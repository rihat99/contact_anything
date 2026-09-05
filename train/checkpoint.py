"""Trainable-only checkpoints with a strict name+shape load.

The frozen SAM-3D-Body weights live in the base checkpoint — re-saving them
every epoch would be ~600 MB of nothing new. A checkpoint here carries the
``requires_grad`` parameters (token blocks, heads, the RoPE brick), the
optimizer and scheduler state, the run counters and the resolved config.

Loading is strict: the checkpoint's ``(name, shape)`` set must equal the live
model's trainable set exactly. A mismatch raises with the full diff rather than
silently leaving a branch at random init.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def trainable_names(model: nn.Module) -> list[str]:
    """Names of every ``requires_grad`` parameter, in module order."""
    return [name for name, p in model.named_parameters() if p.requires_grad]


def trainable_state_dict(model: nn.Module) -> dict:
    """State-dict entries for the trainable parameters only."""
    wanted = set(trainable_names(model))
    return {k: v for k, v in model.state_dict().items() if k in wanted}


def _spec(state: dict) -> dict[str, tuple[int, ...]]:
    return {name: tuple(t.shape) for name, t in state.items()}


def save(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object],
    *,
    epoch: int,
    step: int,
    best: float,
    config: dict,
    extra: Optional[dict] = None,
) -> None:
    """Write a checkpoint of the trainable state plus optimizer/run metadata.

    ``extra`` entries (e.g. the trainer's ``ema_raw`` weights) are stored
    alongside and returned untouched by :func:`load`.
    """
    torch.save(
        {
            "state_dict": trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "epoch": int(epoch),
            "step": int(step),
            "best": float(best),
            "config": config,
            **(extra or {}),
        },
        Path(path),
    )


def load(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    map_location: str = "cpu",
) -> dict:
    """Restore a checkpoint into ``model`` (and optionally optimizer/scheduler).

    :returns: the loaded checkpoint dict (``epoch``, ``step``, ``best``, ``config``).
    :raises RuntimeError: if the checkpoint's trainable ``(name, shape)`` set
        differs from the model's, with the full diff.
    """
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError(f"{path}: not a training checkpoint (no state_dict).")

    saved = _spec(ckpt["state_dict"])
    current = _spec(trainable_state_dict(model))
    missing = sorted(set(current) - set(saved))
    unexpected = sorted(set(saved) - set(current))
    reshaped = sorted(
        f"{name}: checkpoint {saved[name]} vs model {current[name]}"
        for name in set(saved) & set(current) if saved[name] != current[name]
    )
    if missing or unexpected or reshaped:
        raise RuntimeError(
            f"{path}: trainable architecture mismatch — refusing to load.\n"
            f"  in the model but not the checkpoint ({len(missing)}): {missing}\n"
            f"  in the checkpoint but not the model ({len(unexpected)}): {unexpected}\n"
            f"  shape mismatches ({len(reshaped)}): {reshaped}")

    model.load_state_dict(ckpt["state_dict"], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt["scheduler"] is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt
