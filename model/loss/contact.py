"""Confidence-weighted BCE over the six kindyn contact groups.

Reads ``out["contact"]["logits"] (B, 6)`` — one logit per contact token, in
:data:`~model.loss.KINDYN_GROUP_NAMES` order — against the collated
``contact_gt`` / ``contact_valid`` / ``contact_conf`` labels.

The supervision weight of an element is ``contact_valid * contact_conf`` (the
confidence factor is switched off by ``contact_supervision.confidence_weights:
false``), so an unlabelled joint-frame contributes exactly nothing and a
low-confidence label contributes proportionally less. Masked elements are
replaced BEFORE the loss rather than multiplied out afterwards: ``NaN * 0`` is
still ``NaN``, and an untracked video frame carries no meaningful logit.

Plain binary cross-entropy: calibrated probabilities, constant gradient scale
(what WHAM / GVHMR / TRACE use for their contact heads).

Metrics are micro P / R / F1 / IoU at threshold 0.5 over the whole split, plus
per-group F1 — a micro score otherwise hides a weak heel behind four strong
limbs — and ``precision_at_r90``: the micro precision at the operating point
whose recall is 0.9, interpolated on the 0.02..0.9 threshold curve (NaN when
no curve point brackets that recall).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from model.loss import KINDYN_GROUP_NAMES, NUM_KINDYN_GROUPS, Loss, LossResult
from utils.metrics import COUNT_NAMES, contact_counts, prf1

#: Prediction threshold of every reported contact metric.
THRESHOLD = 0.5
#: Thresholds of the accumulated P/R curve (``precision_at_r90``).
CURVE_THRESHOLDS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
#: Recall at which the curve precision is reported.
CURVE_RECALL = 0.9


class ContactLoss(Loss):
    """Confidence-weighted BCE on the six contact logits."""

    name = "contact"
    term_names = ("bce",)
    stat_names = tuple(
        f"{group}/{count}" for group in KINDYN_GROUP_NAMES for count in COUNT_NAMES
    ) + tuple(
        f"curve/{threshold}/{count}" for threshold in CURVE_THRESHOLDS for count in COUNT_NAMES)

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        super().__init__(cfg, model, device)
        cs = cfg["contact_supervision"]
        self.weight = float(cs["weight"])
        self.use_confidence = bool(cs["confidence_weights"])

    def __call__(self, out: dict, batch: dict, *, train: bool) -> LossResult:
        logits = out["contact"]["logits"].to(self.device, self.dtype)
        gt = batch["contact_gt"].to(self.device, self.dtype)
        mask = batch["contact_valid"].to(self.device, self.dtype)
        if self.use_confidence:
            mask = mask * batch["contact_conf"].to(self.device, self.dtype)
        if logits.shape != gt.shape:
            raise ValueError(
                f"contact logits {tuple(logits.shape)} do not match the labels "
                f"{tuple(gt.shape)} — the contact head's token count and the "
                f"dataset's group count must agree")

        # An ignored element must not reach the loss at all: NaN * 0 is NaN.
        safe = torch.where(mask > 0, logits, torch.zeros_like(logits))
        per_element = F.binary_cross_entropy_with_logits(safe, gt, reduction="none")
        numerator = self.weight * (per_element * mask).sum()
        mass = float(mask.sum())
        anchor = safe.sum() * 0.0

        detached = logits.detach()
        stats = torch.cat(
            [contact_counts(detached, gt, mask, THRESHOLD).reshape(-1)]
            + [contact_counts(detached, gt, mask, t).sum(dim=0) for t in CURVE_THRESHOLDS])
        scalars = {"n_active": float((mask > 0).sum()),
                   "pos_rate": float((torch.sigmoid(detached) > THRESHOLD)
                                     .to(self.dtype).mean())}
        return LossResult(
            terms=self._terms({"bce": (numerator, mass)}, anchor),
            scalars=scalars,
            stats=stats.to(self.device),
        )

    def metrics(self, stats: Tensor) -> dict[str, float]:
        n_group = NUM_KINDYN_GROUPS * len(COUNT_NAMES)
        counts = stats[:n_group].reshape(NUM_KINDYN_GROUPS, len(COUNT_NAMES))
        curve = stats[n_group:].reshape(len(CURVE_THRESHOLDS), len(COUNT_NAMES))
        micro = prf1(counts.sum(dim=0))
        out = {key: micro[key] for key in ("f1", "precision", "recall", "iou")}
        out["precision_at_r90"] = precision_at_recall(curve, CURVE_RECALL)
        for group, row in zip(KINDYN_GROUP_NAMES, counts):
            out[f"groups/{group}_f1"] = prf1(row)["f1"]
        return out


def precision_at_recall(curve: Tensor, recall: float) -> float:
    """Precision at ``recall`` on the accumulated threshold curve (``(K, 4)``
    counts at increasing thresholds), linearly interpolated between the two
    neighbouring operating points; NaN when the curve does not bracket
    ``recall`` (every point recalls less, or even the highest threshold
    recalls more).
    """
    points = [prf1(row) for row in curve]                # increasing threshold
    prev = None
    for point in points:                                 # recall DEcreases along the curve
        if point["recall"] < recall:
            if prev is None:
                return float("nan")
            span = prev["recall"] - point["recall"]
            frac = (prev["recall"] - recall) / span if span > 0 else 0.0
            return prev["precision"] + frac * (point["precision"] - prev["precision"])
        prev = point
    return float("nan")


__all__ = ["ContactLoss", "THRESHOLD", "CURVE_THRESHOLDS", "precision_at_recall"]
