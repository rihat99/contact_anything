"""Load a trained model for inference and run single clips through it.

Scripts that only need predictions (evaluation, rendering, reconstruction) go
through here so the config -> build -> checkpoint recipe lives in one place.
``checkpoint_path=None`` returns the untrained model — the frozen-baseline arm,
whose contact/force/motion heads are at their (zero-gated) initial values.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from data.collate import batch_to_device
from model.build import build_model
from model.network import ContactAnything
from train.checkpoint import load as load_checkpoint
from train.config import load_config


def load_model(
    config_path: str | Path,
    checkpoint_path: Optional[str | Path] = None,
    device: torch.device | str = "cuda",
) -> tuple[ContactAnything, dict]:
    """Build the model of ``config_path`` on ``device`` and restore weights.

    :param checkpoint_path: a run checkpoint, or ``None`` to keep the untrained
        branches (the frozen baseline).
    :returns: ``(model in eval mode, resolved config)``.
    """
    cfg = load_config(config_path)
    model = build_model(cfg, device)
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, map_location=str(device))
    model.eval()
    return model, cfg


@torch.no_grad()
def run_clip(model: ContactAnything, batch: dict,
             device: torch.device | str = "cuda") -> dict:
    """Move one collated clip batch to ``device`` and forward it."""
    return model(batch_to_device(batch, device))
