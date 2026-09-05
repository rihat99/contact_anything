"""Build :class:`~model.network.ContactAnything` from a resolved run config.

The config sections map onto the network's sub-config dicts 1:1; everything the
frozen base needs (checkpoint, MHR archive, mask conditioning, the no-grad
efficiency flags) is owned by :class:`~model.wrapper.SAM3DBodyWrapper`, which
also freezes and eval-pins the base. The one ``requires_grad`` decision made
here is the stage-2 one: ``model.smplx.checkpoint`` initialises the SMPL-X head
from a stage-1 run and ``model.smplx.frozen`` freezes it, so its per-frame body
becomes a fixed input of the refiner (a frozen head is not part of the run's
own checkpoints — the config's path is re-read on load).
"""
from __future__ import annotations

from pathlib import Path

import torch

from model.network import ContactAnything
from model.wrapper import SAM3DBodyWrapper

_SMPLX_PREFIX = "head_smplx."


def _section(parent: dict, name: str) -> dict | None:
    """``parent[name]`` minus ``enabled``, or ``None`` when off."""
    node = parent[name]
    if not node["enabled"]:
        return None
    return {k: v for k, v in node.items() if k != "enabled"}


def init_smplx_head(model: ContactAnything, smplx_cfg: dict) -> None:
    """Load ``head_smplx`` from ``smplx_cfg["checkpoint"]`` and apply ``frozen``."""
    path = smplx_cfg["checkpoint"]
    if path is not None:
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        # `camera: cliff | ray` heads have identical parameter shapes, so the shapes cannot
        # tell them apart: compare the head definition recorded in the checkpoint's config.
        saved = ckpt["config"]["model"]["smplx"]
        for key in ("camera", "hands", "model_path", "mlp_depth", "mlp_channel_div_factor"):
            if saved[key] != smplx_cfg[key]:
                raise ValueError(
                    f"{path}: its head was built with model.smplx.{key} = {saved[key]!r}, "
                    f"this config says {smplx_cfg[key]!r}")
        state = {k[len(_SMPLX_PREFIX):]: v for k, v in ckpt["state_dict"].items()
                 if k.startswith(_SMPLX_PREFIX)}
        if not state:
            raise ValueError(f"{path}: the checkpoint carries no head_smplx weights")
        model.head_smplx.load_state_dict(state, strict=True)
    if smplx_cfg["frozen"]:
        for p in model.head_smplx.parameters():
            p.requires_grad_(False)


def build_model(cfg: dict, device: torch.device | str) -> ContactAnything:
    """Construct the model for ``cfg`` on ``device``, in eval mode.

    :param cfg: resolved run config (see :func:`train.config.load_config`).
    :param device: torch device for the whole model.
    :returns: the composed model; the frozen base is eval-pinned, so a later
        ``model.train(True)`` toggles only the trainable branches.
    """
    mcfg = cfg["model"]
    wrapper = SAM3DBodyWrapper(mcfg["checkpoint_path"], mcfg["mhr_model_path"])
    model = ContactAnything(
        wrapper,
        contact=_section(mcfg, "contact"),
        force=_section(mcfg, "force"),
        cross_modal=_section(mcfg, "cross_modal_temporal"),
        smplx=_section(mcfg, "smplx"),
        refiner=_section(mcfg, "refiner"),
    )
    if model.head_smplx is not None:
        init_smplx_head(model, mcfg["smplx"])
    model.to(device)
    model.eval()
    return model
