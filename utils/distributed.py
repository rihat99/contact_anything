"""Distributed helpers: rank queries and the exact DDP global weighted mean.

Every loss term in this repo is a weighted mean whose numerator and mass are
ADDITIVE across ranks. That is deliberate: a rank-local normalisation cannot be
undone after the fact, so ranks with unequal (or zero) supervision mass would
silently reweight the objective. :func:`ddp_global_mean_term` is the one place
that turns the additive pair back into a gradient equal to the single-process
global weighted mean.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor


def is_distributed() -> bool:
    """Whether ``torch.distributed`` is initialised with a real process group."""
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main() -> bool:
    return rank() == 0


def ddp_global_mean_term(
    local_numerator: Tensor, global_mass: Tensor | float, world: int | None = None,
) -> Tensor:
    """One rank's loss term whose DDP-averaged gradient is the global weighted mean.

    DDP averages gradients over ``world_size`` ranks, so multiplying an additive
    local numerator by ``world_size / clamp(global_mass, 1)`` makes the averaged
    gradient equal ``sum(numerator) / clamp(sum(mass), 1)``. Exact when a
    non-empty rank's mass is below one and when another rank's mass is zero.
    Outside DDP it degenerates to the plain local weighted mean.

    :param local_numerator: this rank's graph-live weighted numerator (scalar).
    :param global_mass: the all-reduced mass of the same term.
    :param world: process count; ``None`` queries the current process group.
    """
    denominator = torch.as_tensor(
        global_mass, device=local_numerator.device, dtype=local_numerator.dtype
    ).clamp(min=1.0)
    return local_numerator * (float(world_size() if world is None else world)
                              / denominator)
