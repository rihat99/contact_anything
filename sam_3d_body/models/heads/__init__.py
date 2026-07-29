# Copyright (c) Meta Platforms, Inc. and affiliates.

from ..modules import to_2tuple
from .camera_head import PerspectiveHead
from .contact_head import ContactHead
from .force_head import ForceHead
from .mhr_head import MHRHead


def build_head(cfg, head_type="mhr", enable_hand_model=False, default_scale_factor=1.0,
               output_dims=None):
    if head_type == "mhr":
        return MHRHead(
            input_dim=cfg.MODEL.DECODER.DIM,
            mlp_depth=cfg.MODEL.MHR_HEAD.get("MLP_DEPTH", 1),
            mhr_model_path=cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH,
            mlp_channel_div_factor=cfg.MODEL.MHR_HEAD.get("MLP_CHANNEL_DIV_FACTOR", 1),
            enable_hand_model=enable_hand_model,
        )
    elif head_type == "perspective":
        return PerspectiveHead(
            input_dim=cfg.MODEL.DECODER.DIM,
            img_size=to_2tuple(cfg.MODEL.IMAGE_SIZE),
            mlp_depth=cfg.MODEL.get("CAMERA_HEAD", dict()).get("MLP_DEPTH", 1),
            mlp_channel_div_factor=cfg.MODEL.get("CAMERA_HEAD", dict()).get(
                "MLP_CHANNEL_DIV_FACTOR", 1
            ),
            default_scale_factor=default_scale_factor,
        )
    elif head_type == "contact":
        contact_cfg = cfg.MODEL.get("CONTACT_HEAD", dict())
        num_kp  = contact_cfg.get("NUM_CONTACTS", 21)
        num_gbl = contact_cfg.get("NUM_GLOBAL_TOKENS", 0)
        dims = output_dims if output_dims is not None else contact_cfg.get("NUM_VERTICES", 18439)
        return ContactHead(
            input_dim=cfg.MODEL.DECODER.DIM,
            num_contact_tokens=num_kp + num_gbl,
            # ContactHead preserves this arbitrary output size for pooled modes;
            # per-token mode validates that it equals the total token count.
            output_dims=dims,
            mlp_depth=contact_cfg.get("MLP_DEPTH", 2),
            mlp_channel_div_factor=contact_cfg.get("MLP_CHANNEL_DIV_FACTOR", 4),
            pool_mode=contact_cfg.get("POOL_MODE", "attention"),
            dropout=contact_cfg.get("DROPOUT", 0.0),
        )
    elif head_type == "force":
        force_cfg = cfg.MODEL.get("FORCE_HEAD", dict())
        # Force anchors: FORCE_HEAD.KEYPOINT_INDICES when explicitly set,
        # otherwise the contact keypoint anchors (D2, legacy default) — the
        # number of force tokens follows the resolved anchor list.
        force_kp = force_cfg.get("KEYPOINT_INDICES", None)
        if force_kp is not None:
            num_kp = len(force_kp)
        else:
            num_kp = cfg.MODEL.get("CONTACT_HEAD", dict()).get("NUM_CONTACTS", 21)
        return ForceHead(
            input_dim=cfg.MODEL.DECODER.DIM,
            num_force_tokens=num_kp,
            mlp_depth=force_cfg.get("MLP_DEPTH", 2),
            mlp_channel_div_factor=force_cfg.get("MLP_CHANNEL_DIV_FACTOR", 4),
            dropout=force_cfg.get("DROPOUT", 0.0),
        )
    else:
        raise ValueError("Invalid head type: ", head_type)
