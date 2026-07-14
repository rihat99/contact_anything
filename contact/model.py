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
``contact_feat_linear.*``, and ``contact_temporal.*`` when enabled). Backbone,
decoder, prompt encoder, MHR/camera heads stay frozen.

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

from .targets import TargetSpec


def _patch_model_cfg(model_cfg, cfg: dict, mhr_path: str):
    """Mutate ``model_cfg`` in-place to honour the train config.

    Sizes the contact tokens from ``model.contact_head.contact_keypoint_indices``
    and builds one head per enabled contact target (``CONTACT_HEAD.TARGETS`` maps
    the target name to its output dim: topology verts for ``vertex``, 22 for
    ``joint``).
    """
    from yacs.config import CfgNode

    chead = cfg["model"]["contact_head"]
    kp = chead.get("contact_keypoint_indices")
    if kp is None:
        kp = list(range(21))
    kp = [int(i) for i in kp]

    out_dims = TargetSpec.from_config(cfg).output_dims()   # {name: dim} for enabled targets

    model_cfg.defrost()
    # Always rebuild contact tokens from train config.
    model_cfg.MODEL.DECODER.DO_CONTACT_TOKENS = True
    if "CONTACT_HEAD" not in model_cfg.MODEL:
        model_cfg.MODEL.CONTACT_HEAD = CfgNode()
    ch = model_cfg.MODEL.CONTACT_HEAD
    ch.KEYPOINT_INDICES        = kp
    ch.NUM_CONTACTS            = len(kp)
    ch.NUM_GLOBAL_TOKENS       = int(chead["num_global_tokens"])
    ch.MLP_DEPTH               = int(chead["mlp_depth"])
    ch.MLP_CHANNEL_DIV_FACTOR  = int(chead["mlp_channel_div_factor"])
    ch.POOL_MODE               = str(chead["pool_mode"])
    ch.DROPOUT                 = float(chead["dropout"])
    ch.GRID_SIZE               = int(chead["grid_size"])
    ch.GRID_RADIUS             = float(chead["grid_radius"])
    ch.TARGETS = CfgNode({name.upper(): int(dim) for name, dim in out_dims.items()})

    # Temporal module (Phase 3). Patched even when disabled so the model config
    # is self-describing; SAM3DBody only builds contact_temporal when ENABLED.
    tcfg = cfg["model"].get("temporal", {}) or {}
    model_cfg.MODEL.TEMPORAL = CfgNode({
        "ENABLED": bool(tcfg.get("enabled", False)),
        "PLACEMENT": str(tcfg.get("placement", "post_decoder")),
        "BOTTLENECK_DIM": int(tcfg.get("bottleneck_dim", 256)),
        "NUM_LAYERS": int(tcfg.get("num_layers", 1)),
        "NUM_HEADS": int(tcfg.get("num_heads", 4)),
        "MLP_RATIO": float(tcfg.get("mlp_ratio", 2.0)),
        "ATTEND": str(tcfg.get("attend", "joint")),
        "CAUSAL": bool(tcfg.get("causal", False)),
        "DROPOUT": float(tcfg.get("dropout", 0.0)),
        "POSITION_SCALE": float(tcfg.get("position_scale", 1.0)),
    })

    # Efficiency flags (Phase 4). Patched even when off so the model config is
    # self-describing; sam3d_body / promptable_decoder read them at their delimited
    # hooks and fall back to old (full-graph) behaviour when the key is absent.
    train_cfg = cfg.get("train", {}) or {}
    model_cfg.MODEL.EFFICIENCY = CfgNode({
        "BACKBONE_NO_GRAD": bool(train_cfg.get("backbone_no_grad", False)),
        "DETACH_INTERM_PREDS": bool(train_cfg.get("detach_interm_preds", False)),
    })

    # Mask conditioning (must be set before model build).
    model_cfg.MODEL.PROMPT_ENCODER.MASK_EMBED_TYPE = cfg["model"].get("mask_embed_type", None)

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


def pin_frozen_eval(model: nn.Module) -> nn.Module:
    """Permanently pin every frozen submodule to eval(); only contact modules
    follow the requested train/eval mode.

    The frozen SAM-3D-Body backbone ships with ``DROP_PATH_RATE 0.1`` (stochastic
    depth) plus dropout, so a global ``model.train()`` would make the frozen
    features the contact head reads nondeterministic. We force all non-contact
    modules to eval and keep them there by overriding ``model.train`` so any
    later ``model.train(True)`` re-pins them. ``model.eval()`` still works since
    it delegates to ``model.train(False)``.

    Contact modules are every submodule whose dotted path contains ``"contact"``
    — ``head_contact``, ``contact_posemb_linear``, ``contact_feat_linear``,
    ``contact_embedding``, ``contact_temporal`` and all their descendants
    (including param-less dropout children, which need train mode to be active).
    This matches the ``"contact"`` param-name freeze filter, so exactly the
    trainable subtree stays trainable.
    """
    contact_modules = [m for name, m in model.named_modules() if "contact" in name.lower()]

    def _apply(mode: bool = True) -> nn.Module:
        nn.Module.train(model, False)                   # every module → eval
        for submodule in contact_modules:
            submodule.training = bool(mode)             # contact subtree → mode
        return model

    model.train = _apply
    _apply(False)
    return model


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
    _patch_model_cfg(model_cfg, cfg, mhr_path)

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

    # Keep frozen backbone/decoder/heads in eval permanently (kills stochastic
    # depth + dropout nondeterminism); only contact modules toggle train/eval.
    pin_frozen_eval(model)

    # backbone_no_grad is only sound if the backbone is fully frozen.
    if model_cfg.MODEL.EFFICIENCY.BACKBONE_NO_GRAD:
        trainable_backbone = [
            n for n, p in model.named_parameters()
            if p.requires_grad and n.startswith("backbone")
        ]
        assert not trainable_backbone, (
            f"train.backbone_no_grad is set but the backbone has trainable params: "
            f"{trainable_backbone}")

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_train:,} / {n_total:,}  ({100 * n_train / n_total:.2f}%)")
    return model, trainable_names
