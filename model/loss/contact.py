"""Focal contact loss over the six kindyn groups.

Reads ``out["contact"]["joint_logits"] (B, 6)`` — one logit per contact token,
in :data:`~model.loss.KINDYN_GROUP_NAMES` order — against the collated
``contact_gt`` / ``contact_valid`` / ``contact_conf`` labels.

The supervision weight of an element is ``contact_valid * contact_conf`` (the
confidence factor is switched off by ``contact_supervision.confidence_weights:
false``), so an unlabelled joint-frame contributes exactly nothing and a
low-confidence label contributes proportionally less. Masked elements are
replaced BEFORE the loss rather than multiplied out afterwards: ``NaN * 0`` is
still ``NaN``, and an untracked video frame carries no meaningful logit.

Asymmetric focal BCE — ``alpha`` weights the POSITIVE class
(``alpha_t = a * gt + (1 - a) * (1 - gt)``) and ``gamma`` focuses on hard
examples. The class prior here is near balanced (~47 % positive over the
labelled joint-frames), so ``alpha`` is not a rare-class correction but a
deliberate precision/recall trade: a false contact invents a downstream force
while a missed one only omits it.

Metrics are micro P / R / F1 / F2 / IoU at threshold 0.5 over the whole split,
plus per-group P / R / F1 — a micro score otherwise hides a weak heel behind
four strong limbs (the heels run a ~3 % positive prior).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from model.loss import (
    KINDYN_GROUP_NAMES,
    NUM_KINDYN_GROUPS,
    Loss,
    LossResult,
)
from utils.metrics import COUNT_NAMES, contact_counts, prf1

#: Prediction threshold of every reported contact metric.
THRESHOLD = 0.5


class ContactLoss(Loss):
    """Confidence-weighted asymmetric focal BCE on the six contact logits."""

    name = "contact"
    stat_names = tuple(
        f"{group}/{count}" for group in KINDYN_GROUP_NAMES for count in COUNT_NAMES)

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        super().__init__(cfg, model, device)
        cs = cfg["contact_supervision"]
        self.weight = float(cs["weight"])
        self.term_names = ("focal",)
        self.alpha = float(cs["focal_alpha"])
        self.gamma = float(cs["focal_gamma"])
        self.use_confidence = bool(cs["confidence_weights"])

    def __call__(self, out: dict, batch: dict, *, train: bool) -> LossResult:
        logits = out["contact"]["joint_logits"].to(self.device, self.dtype)
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
        bce = F.binary_cross_entropy_with_logits(safe, gt, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.alpha * gt + (1.0 - self.alpha) * (1.0 - gt)
        focal = alpha_t * (1.0 - pt) ** self.gamma * bce

        numerator = self.weight * (focal * mask).sum()
        mass = float(mask.sum())
        anchor = safe.sum() * 0.0

        stats = contact_counts(logits.detach(), gt, mask, THRESHOLD).reshape(-1)
        scalars = {"n_active": float((mask > 0).sum()),
                   "pos_rate": float((torch.sigmoid(logits.detach()) > THRESHOLD)
                                     .to(self.dtype).mean())}
        return LossResult(
            terms=self._terms({"focal": (numerator, mass)}, anchor),
            scalars=scalars,
            stats=stats.to(self.device),
        )

    def metrics(self, stats: Tensor) -> dict[str, float]:
        counts = stats.reshape(NUM_KINDYN_GROUPS, len(COUNT_NAMES))
        out = prf1(counts.sum(dim=0))
        for group, row in zip(KINDYN_GROUP_NAMES, counts):
            group_metrics = prf1(row)
            for key in ("precision", "recall", "f1"):
                out[f"{group}/{key}"] = group_metrics[key]
        out["n_active"] = float(counts.sum())
        return out


__all__ = ["ContactLoss", "THRESHOLD"]
