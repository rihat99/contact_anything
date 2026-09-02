"""Metric closed forms over ADDITIVE sufficient statistics.

Every eval metric in this repo is accumulated as a float64 sum over batches and
all-reduced once per epoch, then divided here. That is what makes a distributed
evaluation exact: a mean of per-batch means would weight a short final batch
like a full one, and a Pearson r cannot be averaged at all.

* Contact: masked confusion counts per output dimension -> P / R / F1 / F2 / IoU.
* Regression: ``(n, sum_p, sum_g, sum_pp, sum_gg, sum_pg)`` -> Pearson r, and
  ``(sum_sq_err, n)`` -> RMSE.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

_EPS = 1e-8

#: Column order of the per-dimension confusion tensor.
COUNT_NAMES = ("tp", "fp", "fn", "tn")


@torch.no_grad()
def contact_counts(
    logits: Tensor, gt: Tensor, mask: Tensor, threshold: float = 0.5,
) -> Tensor:
    """Masked confusion counts per output dimension. ``(B, D) -> (D, 4)`` float64.

    Columns are :data:`COUNT_NAMES`. Only elements with ``mask > 0`` count, so a
    dimension nobody supervised this batch contributes exactly zero counts and
    drops out of every denominator downstream.
    """
    if logits.shape != gt.shape or logits.shape != mask.shape:
        raise ValueError(
            f"logits/gt/mask shapes must match; got {tuple(logits.shape)}, "
            f"{tuple(gt.shape)}, {tuple(mask.shape)}")
    pred = torch.sigmoid(logits) > threshold
    positive = gt > 0.5
    active = mask > 0
    return torch.stack([
        (pred & positive & active).sum(dim=0),
        (pred & ~positive & active).sum(dim=0),
        (~pred & positive & active).sum(dim=0),
        (~pred & ~positive & active).sum(dim=0),
    ], dim=-1).to(torch.float64)


def prf1(counts: Tensor | Sequence[float]) -> dict[str, float]:
    """Precision / recall / F1 / F2 / IoU / accuracy from ``(tp, fp, fn, tn)``."""
    tp, fp, fn, tn = (float(v) for v in counts)
    return {
        "precision": tp / (tp + fp + _EPS),
        "recall": tp / (tp + fn + _EPS),
        "f1": 2 * tp / (2 * tp + fp + fn + _EPS),
        # F-beta with beta = 2: recall carries four times precision's weight.
        "f2": 5 * tp / (5 * tp + fp + 4 * fn + _EPS),
        "iou": tp / (tp + fp + fn + _EPS),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + _EPS),
    }


def pearson_from_stats(
    n: Tensor, sum_p: Tensor, sum_g: Tensor,
    sum_pp: Tensor, sum_gg: Tensor, sum_pg: Tensor,
) -> Tensor:
    """Pearson r from weighted sufficient statistics; ``nan`` when degenerate.

    Degenerate means fewer than two samples or a zero variance on either side —
    a correlation that does not exist must not read as 0.
    """
    cov = n * sum_pg - sum_p * sum_g
    var_p = n * sum_pp - sum_p ** 2
    var_g = n * sum_gg - sum_g ** 2
    denom = (var_p.clamp(min=0) * var_g.clamp(min=0)).sqrt()
    return torch.where(
        (n >= 2) & (denom > 0), cov / denom.clamp(min=1e-30),
        torch.full_like(cov, float("nan")))


def rmse_from_stats(sum_sq_err: Tensor, n: Tensor) -> Tensor:
    """Root mean squared error from ``(sum of squared errors, sample count)``."""
    return torch.where(
        n > 0, (sum_sq_err / n.clamp(min=1.0)).sqrt(),
        torch.full_like(n, float("nan")))


def mean_from_stats(numerator: float, mass: float) -> float:
    """Mass-weighted mean; ``nan`` at zero mass.

    Zero mass — nothing in the split supervised this quantity — must never
    masquerade as a perfect score, so it reports ``nan`` rather than 0.
    """
    return numerator / mass if mass > 0 else float("nan")
