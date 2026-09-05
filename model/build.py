"""Build :class:`~model.network.ContactAnything` from a resolved run config.

The config sections map onto the network's sub-config dicts 1:1; everything the
frozen base needs (checkpoint, MHR archive, mask conditioning, the no-grad
efficiency flags) is owned by :class:`~model.wrapper.SAM3DBodyWrapper`, which
also freezes and eval-pins the base. Nothing here touches ``requires_grad``:
the wrapper's parameters are frozen at construction and every module this
builder adds is trainable by construction.

The contact head is the per-token regime — one shared classifier applied to
each anchored token, so the six kindyn_6 anchors give six contact outputs —
or, with ``contact.source: pose_token``, one flat classifier from the pose
token to the six kindyn_6 groups.
"""
from __future__ import annotations

import torch

from model.loss import NUM_KINDYN_GROUPS
from model.network import ContactAnything
from model.wrapper import SAM3DBodyWrapper


def _section(parent: dict, name: str) -> dict | None:
    """``parent[name]`` minus ``enabled``, or ``None`` when off."""
    node = parent[name]
    if not node["enabled"]:
        return None
    return {k: v for k, v in node.items() if k != "enabled"}


def build_model(cfg: dict, device: torch.device | str) -> ContactAnything:
    """Construct the model for ``cfg`` on ``device``, in eval mode.

    :param cfg: resolved run config (see :func:`train.config.load_config`).
    :param device: torch device for the whole model.
    :returns: the composed model; the frozen base is eval-pinned, so a later
        ``model.train(True)`` toggles only the trainable branches.
    """
    mcfg = cfg["model"]
    wrapper = SAM3DBodyWrapper(mcfg["checkpoint_path"], mcfg["mhr_model_path"])

    contact = _section(mcfg, "contact")
    if contact is not None:
        if contact["source"] == "pose_token":
            contact["targets"] = {"joint": NUM_KINDYN_GROUPS}
            contact["pool_mode"] = "flat"
        else:
            num_tokens = len(contact["keypoint_indices"]) + contact["num_global_tokens"]
            contact["targets"] = {"joint": num_tokens}
            contact["pool_mode"] = "per_token"

    model = ContactAnything(
        wrapper,
        contact=contact,
        force=_section(mcfg, "force"),
        motion=_section(mcfg, "motion"),
        cross_modal=_section(mcfg, "cross_modal_temporal"),
        pose_temporal=_section(mcfg, "pose_temporal"),
        finetune_pose_head=bool(mcfg["finetune_pose_head"]),
        finetune_camera_head=bool(mcfg["finetune_camera_head"]),
        extra_token_attention=str(mcfg["extra_token_attention"]),
        smplx=_section(mcfg, "smplx"),
        keypoints2d=_section(mcfg["token_inputs"], "keypoints2d"),
        token_masking=_section(mcfg, "token_masking"),
        camera_twist=_section(mcfg["token_inputs"], "camera_twist"),
        bbox=_section(mcfg["token_inputs"], "bbox"),
        frozen_camera=_section(mcfg["token_inputs"], "frozen_camera"),
    )
    model.to(device)
    model.eval()
    return model
