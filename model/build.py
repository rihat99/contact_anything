"""Build :class:`~model.network.ContactAnything` from a resolved run config.

The config sections map onto the network's sub-config dicts 1:1; everything the
frozen base needs (checkpoint, MHR archive, mask conditioning, the no-grad
efficiency flags) is owned by :class:`~model.wrapper.SAM3DBodyWrapper`, which
also freezes and eval-pins the base. Everything else trains from scratch.
"""
from __future__ import annotations

import torch

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
    # Plain fp32 wherever the frozen base is not bf16: TF32 (10-bit mantissas) puts
    # ~1e-3 relative noise on the per-frame camera output, which the trajectory metrics
    # cube (lifted jitter 448 vs 15 on BEDLAM). Every evaluation / prediction entry point
    # therefore measures without it; scripts/train.py switches it back on for the training
    # steps (15 % faster). cudnn's default is True.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    mcfg = cfg["model"]
    wrapper = SAM3DBodyWrapper(
        mcfg["checkpoint_path"], mcfg["mhr_model_path"],
        autocast_bf16=bool(mcfg["decoder_bf16"]),
        checkpoint_layers=bool(mcfg["decoder_checkpointing"]))
    model = ContactAnything(
        wrapper,
        contact=_section(mcfg, "contact"),
        force=_section(mcfg, "force"),
        cross_modal=_section(mcfg, "cross_modal_temporal"),
        smplx=_section(mcfg, "smplx"),
        token_heads=_section(mcfg, "token_heads"),
        refiner=_section(mcfg, "refiner"),
        contact_set_name=str(cfg["data"]["contact_set"]),
    )
    model.to(device)
    model.eval()
    return model
