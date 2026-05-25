"""Save/load only the parameters we actually train.

The frozen SAM-3D-Body weights live in the original checkpoint —
re-saving them every epoch would be ~600 MB of nothing new. Here we
serialise just the names in ``trainable_names`` (contact tokens, contact
head, mask conditioning, LoRA adapters), plus the optimiser, scheduler,
and run state.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn


def _select_state(model: nn.Module, names: Iterable[str]) -> dict:
    sd = model.state_dict()
    names = set(names)
    return {k: v for k, v in sd.items() if k in names}


def save(
    path: str | Path,
    model: nn.Module,
    trainable_names: Iterable[str],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object],
    epoch: int,
    global_step: int,
    best_val: float,
    extra: Optional[dict] = None,
) -> None:
    ckpt = {
        "trainable_state_dict": _select_state(model, trainable_names),
        "trainable_names":      list(trainable_names),
        "optimizer":            optimizer.state_dict(),
        "scheduler":            scheduler.state_dict() if scheduler is not None else None,
        "epoch":                int(epoch),
        "global_step":          int(global_step),
        "best_val":             float(best_val),
    }
    if extra:
        ckpt["extra"] = extra
    torch.save(ckpt, Path(path))


def load(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    map_location: str = "cpu",
) -> dict:
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    missing, unexpected = model.load_state_dict(
        ckpt["trainable_state_dict"], strict=False,
    )
    # Anything trainable that's missing is a real problem; the frozen
    # base weights are expected to be missing here.
    trainable_missing = [m for m in missing if m in set(ckpt["trainable_names"])]
    if trainable_missing:
        raise RuntimeError(f"trainable params missing from checkpoint: {trainable_missing}")
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt
