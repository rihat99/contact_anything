"""Mask-aware contact metrics: micro precision / recall / F1 / F2 / IoU.

One implementation shared by the trainer, evaluator and demo. Predictions are
``sigmoid(logits) > threshold``; only elements with ``mask > 0`` count, so a
target's unsupervised rows never enter the confusion matrix (and a target with
zero active elements this batch contributes zero counts — excluded from
denominators when metrics are accumulated micro).
"""
from __future__ import annotations

import torch

_EPS = 1e-8
_COUNT_KEYS = ("tp", "fp", "fn", "tn")


@torch.no_grad()
def contact_counts(
    logits: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5,
) -> dict[str, int]:
    """Masked confusion-matrix counts. ``logits/gt/mask``: ``[..., D]``.

    :returns: ``{'tp', 'fp', 'fn', 'tn'}`` integer counts over ``mask > 0`` elements.
    """
    preds = torch.sigmoid(logits) > threshold
    positive = gt > 0.5
    active = mask > 0
    return {
        "tp": int((preds & positive & active).sum()),
        "fp": int((preds & ~positive & active).sum()),
        "fn": int((~preds & positive & active).sum()),
        "tn": int((~preds & ~positive & active).sum()),
    }


def prf1(counts: dict[str, int]) -> dict[str, float]:
    """Precision / recall / F1 / F2 / IoU / accuracy from confusion counts."""
    tp, fp, fn, tn = (counts[k] for k in _COUNT_KEYS)
    return {
        "precision": tp / (tp + fp + _EPS),
        "recall": tp / (tp + fn + _EPS),
        "f1": 2 * tp / (2 * tp + fp + fn + _EPS),
        # F-beta with beta=2: recall carries four times precision's weight.
        "f2": 5 * tp / (5 * tp + fp + 4 * fn + _EPS),
        "iou": tp / (tp + fp + fn + _EPS),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + _EPS),
    }


@torch.no_grad()
def contact_counts_per_dim(
    logits: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5,
) -> list[dict[str, int]]:
    """Return one masked confusion matrix per output dimension.

    This is primarily useful for the four-extremity head, where a micro score can
    otherwise hide a weak hand or foot behind the other three outputs.
    """
    if logits.shape != gt.shape or logits.shape != mask.shape:
        raise ValueError(
            f"logits, gt, and mask must have the same shape; got "
            f"{tuple(logits.shape)}, {tuple(gt.shape)}, {tuple(mask.shape)}")
    if logits.ndim < 2:
        raise ValueError(f"expected a final output dimension; got shape {tuple(logits.shape)}")
    return [
        contact_counts(logits[..., i], gt[..., i], mask[..., i], threshold)
        for i in range(logits.shape[-1])
    ]


def contact_metrics(
    logits: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5,
) -> dict[str, float]:
    """Convenience: masked counts + derived P/R/F1/IoU/accuracy + ``n_active``."""
    counts = contact_counts(logits, gt, mask, threshold)
    n_active = sum(counts[k] for k in _COUNT_KEYS)
    return {**prf1(counts), **counts, "n_active": n_active}


def zero_counts() -> dict[str, int]:
    """A fresh all-zero confusion-count accumulator."""
    return {k: 0 for k in _COUNT_KEYS}


def add_counts(acc: dict[str, int], counts: dict[str, int]) -> None:
    """In-place accumulate ``counts`` into ``acc``."""
    for k in _COUNT_KEYS:
        acc[k] += counts[k]
