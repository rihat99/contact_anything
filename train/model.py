"""Build a SAM-3D-Body model wired for contact-head training.

We patch the checkpoint's ``model_config.yaml`` *before* constructing
the model so the contact tokens, contact head, and the mask
conditioning the checkpoint shipped with are all created natively by
``SAM3DBody``. Checkpoint weights load with ``strict=False`` —
everything in the upstream model (including the v2 mask conditioning)
gets restored, and the new contact modules stay at random init.

After the load we freeze the whole network and unfreeze only the
contact pipeline (anything with ``contact`` in the parameter name —
that's ``contact_embedding``, ``head_contact.*``, ``contact_posemb_linear.*``,
``contact_feat_linear.*``). Backbone, decoder, prompt encoder, MHR/camera
heads stay frozen.

``build_model`` returns ``(model, trainable_names)``.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import torch
import torch.nn as nn

from sam_3d_body.models.meta_arch import SAM3DBody
from sam_3d_body.utils.checkpoint import load_state_dict
from sam_3d_body.utils.config import get_config


def _patch_model_cfg(model_cfg, train_model_cfg, mhr_path: str):
    """Mutate ``model_cfg`` in-place to honour the train config."""
    chead = train_model_cfg["contact_head"]
    num_kp = int(chead["num_keypoint_tokens"])
    num_gl = int(chead["num_global_tokens"])

    model_cfg.defrost()
    # Always rebuild contact tokens from train config.
    model_cfg.MODEL.DECODER.DO_CONTACT_TOKENS = True
    if "CONTACT_HEAD" not in model_cfg.MODEL:
        from yacs.config import CfgNode
        model_cfg.MODEL.CONTACT_HEAD = CfgNode()
    ch = model_cfg.MODEL.CONTACT_HEAD
    ch.NUM_CONTACTS            = num_kp
    ch.NUM_GLOBAL_TOKENS       = num_gl
    ch.NUM_VERTICES            = int(chead["num_vertices"])
    ch.MLP_DEPTH               = int(chead["mlp_depth"])
    ch.MLP_CHANNEL_DIV_FACTOR  = int(chead["mlp_channel_div_factor"])
    ch.POOL_MODE               = str(chead["pool_mode"])
    ch.DROPOUT                 = float(chead["dropout"])
    ch.GRID_SIZE               = int(chead["grid_size"])
    ch.GRID_RADIUS             = float(chead["grid_radius"])

    # Mask conditioning (must be set before model build).
    mask_embed_type = train_model_cfg.get("mask_embed_type", None)
    model_cfg.MODEL.PROMPT_ENCODER.MASK_EMBED_TYPE = mask_embed_type

    # MHR weights path.
    model_cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = mhr_path
    model_cfg.freeze()
    return model_cfg


def _load_checkpoint_weights(model: nn.Module, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    load_state_dict(model, sd, strict=False)


def _trainable_name_filter(name: str) -> bool:
    """Train the contact pipeline only: tokens, head, and the small
    posemb / feat projection layers that update the tokens between
    decoder layers (all of which contain ``contact`` in the name)."""
    return "contact" in name.lower()


def build_model(cfg: dict, device: str = "cuda") -> Tuple[nn.Module, List[str]]:
    """Construct a SAM-3D-Body model ready for contact training.

    Returns ``(model, trainable_param_names)``. The names are produced
    in iteration order over ``model.named_parameters()`` so the caller
    can build an optimiser param group with ``filter(lambda p: p.requires_grad, ...)``
    and a checkpoint dump with the same name list.
    """
    mcfg = cfg["model"]
    ckpt_path = mcfg["checkpoint_path"]
    mhr_path  = mcfg["mhr_model_path"]

    # 1) Load model_config that ships with the checkpoint, then patch.
    model_cfg_path = os.path.join(os.path.dirname(ckpt_path), "model_config.yaml")
    if not os.path.exists(model_cfg_path):
        model_cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), "model_config.yaml",
        )
    model_cfg = get_config(model_cfg_path)
    _patch_model_cfg(model_cfg, mcfg, mhr_path)

    # 2) Build + load weights for the parts that exist in the checkpoint.
    model = SAM3DBody(model_cfg)
    _load_checkpoint_weights(model, ckpt_path)

    # 3) Freeze everything; unfreeze just the contact pipeline.
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if _trainable_name_filter(name):
            p.requires_grad = True

    model.to(device)

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_train:,} / {n_total:,}  ({100 * n_train / n_total:.2f}%)")
    return model, trainable_names
