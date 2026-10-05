"""Confidence-weighted BCE over the run's contact slots.

Reads ``out["contact"]["logits"] (B, K)`` — one logit per contact token, in the
run's contact-set slot order (``data.contact_set``: the six kindyn groups, or the
35 named contact frames) — against the collated ``contact_gt`` / ``contact_valid``
/ ``contact_conf`` labels, which carry the same K slots.

The supervision weight of an element is ``contact_valid * contact_conf`` (the
confidence factor is switched off by ``contact_supervision.confidence_weights:
false``), so an unlabelled joint-frame contributes exactly nothing and a
low-confidence label contributes proportionally less. Masked elements are
replaced BEFORE the loss rather than multiplied out afterwards: ``NaN * 0`` is
still ``NaN``, and an untracked video frame carries no meaningful logit.

Plain binary cross-entropy: calibrated probabilities, constant gradient scale
(what WHAM / GVHMR / TRACE use for their contact heads), normalised PER SLOT:
each slot's weighted mean over its supervised rows of the micro-batch, summed
over the slots (``numerator``) against the constant ``mass`` K, the slot count.
Every slot's gradient is therefore ``weight / K`` per row, whatever its label
rate and however many rows the OTHER slots supervise (an unsupervised slot adds
zero): a slot positive 2 % of the time weighs as much as one positive half the
time, and ``weight`` = K / 6 gives a 35-frame run the per-slot gradient the
six-group recipe had at ``weight`` 1. ``class_weights`` (``{positive:
[K], negative: [K]}``, per slot) multiplies the rows on top: ``positive`` is the
false-NEGATIVE penalty (a positive row's BCE), ``negative`` the false-POSITIVE
one. The six groups are far from balanced on the train labels (hands 78-81 %
positive, toes 60 %, heels 3.4-3.7 %), and a plain BCE learns the heels as
"never" (test F1 exactly 0 through round 3). The weights enter a slot's
numerator AND mass, so they reweight rows without changing the term's scale;
the metrics stay unweighted.

``layer_weight`` adds deep supervision: the same weighted BCE on every
INTERMEDIATE layer's logits of an iterative refiner
(``out["contact"]["logits_layers"][:-1]``; the last entry IS ``logits``), pooled
into ONE ``bce_layer`` term — numerators and masses summed over the layers — at
``layer_weight`` times the section's ``weight``. Every layer reads the same
labels, so the term is the final BCE repeated, and the metrics stay the final
layer's.

The reported scores live on TWO levels, so runs of either contact set compare:

* the SIX-GROUP scores (``f1`` / ``precision`` / ``recall`` / ``iou`` /
  ``precision_at_r90``, plus ``groups/<name>_*``) — the headline numbers, and the
  only ones the manual test labels can score. A group's prediction is the MAX
  probability over its member slots (:meth:`~model.contact_frames.ContactSet.fold_max`);
  its label is ``contact_gt_groups`` when the batch carries the manual six-group
  annotation, else the fold of the slot labels: POSITIVE as soon as one SUPERVISED
  member slot is positive, known-NEGATIVE only when every member slot is supervised
  and free, unscored otherwise. A group with no member slot is never counted.
* the SLOT scores (``slots_f1`` / ``slots_precision`` / ``slots_recall`` /
  ``slots_iou``, plus ``slots/<slot>_*``) — the same micro scores over the K slots
  against the slot labels. Under ``kindyn6`` the fold is the identity and the two
  levels coincide exactly.

``precision_at_r90`` is the six-group micro precision at the operating point whose
recall is 0.9, interpolated on the 0.02..0.9 threshold curve (NaN when no curve
point brackets that recall).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from model.loss import (KINDYN_GROUP_NAMES, NUM_KINDYN_GROUPS, ContactSet, Loss,
                        LossResult)
from utils.metrics import COUNT_NAMES, contact_counts, prf1

#: Logit standing in for a group with no member slot (probability 0, always masked out).
_ABSENT_LOGIT = -1.0e30

#: Prediction threshold of every reported contact metric.
THRESHOLD = 0.5
#: Thresholds of the accumulated P/R curve (``precision_at_r90``).
CURVE_THRESHOLDS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
#: Recall at which the curve precision is reported.
CURVE_RECALL = 0.9


def group_members(slots: ContactSet, device: torch.device | str | None = None) -> Tensor:
    """``(6, K)`` boolean membership matrix of the six-group fold of ``slots``."""
    return torch.as_tensor(slots.group_of, device=device)[None, :] == torch.arange(
        NUM_KINDYN_GROUPS, device=device)[:, None]


def fold_to_groups(score: Tensor, gt: Tensor, mask: Tensor, members: Tensor, batch: dict
                   ) -> tuple[Tensor, Tensor, Tensor]:
    """``(B, K)`` slot scores / labels / mask -> the SIX-GROUP ``(B, 6)`` triple.

    The group's score is the max over its member slots (monotone in the probability,
    so a threshold on either side is the same decision). Its label is the batch's
    manual ``contact_gt_groups`` when present, else the fold rule of this module's
    docstring. ``score`` may be logits or probabilities; the mask is any non-negative
    weight (``> 0`` = supervised).
    """
    has_member = members.any(dim=-1)                                  # (6,)
    group_score = torch.where(
        members[None], score[:, None, :],
        torch.full_like(score[:, None, :], _ABSENT_LOGIT)).amax(dim=-1)
    dtype = mask.dtype if mask.is_floating_point() else score.dtype
    if "contact_gt_groups" in batch:
        group_gt = batch["contact_gt_groups"].to(score.device, dtype)
        group_mask = batch["contact_valid_groups"].to(score.device, dtype)
        return group_score, group_gt, group_mask * has_member.to(dtype)
    supervised = mask > 0                                             # (B, K)
    positive = supervised & (gt > 0.5)
    group_pos = (positive[:, None, :] & members[None]).any(dim=-1)     # (B, 6)
    all_supervised = (supervised[:, None, :] | ~members[None]).all(dim=-1) & has_member
    group_mask = (group_pos | all_supervised) & has_member
    return group_score, group_pos.to(dtype), group_mask.to(dtype)


class ContactLoss(Loss):
    """Confidence-weighted BCE on the K contact logits of the run's contact set."""

    name = "contact"

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        super().__init__(cfg, model, device)
        slots = self.contact_set
        self.stat_names = tuple(
            f"slots/{slot}/{count}" for slot in slots.slot_names for count in COUNT_NAMES
        ) + tuple(
            f"{group}/{count}" for group in KINDYN_GROUP_NAMES for count in COUNT_NAMES
        ) + tuple(
            f"curve/{threshold}/{count}" for threshold in CURVE_THRESHOLDS for count in COUNT_NAMES)
        #: ``(6, K)`` membership of the six-group fold (``True`` = that slot is a member).
        self.members = group_members(slots, self.device)
        cs = cfg["contact_supervision"]
        self.weight = float(cs["weight"])
        self.use_confidence = bool(cs["confidence_weights"])
        self.layer_weight = float(cs["layer_weight"])
        self.term_names = ("bce",) + (("bce_layer",) if self.layer_weight > 0.0 else ())
        self.class_weights = None
        if cs["class_weights"] is not None:
            pos = torch.tensor([float(w) for w in cs["class_weights"]["positive"]],
                               dtype=self.dtype, device=self.device)
            neg = torch.tensor([float(w) for w in cs["class_weights"]["negative"]],
                               dtype=self.dtype, device=self.device)
            if pos.numel() != slots.count or neg.numel() != slots.count:
                raise ValueError(
                    "contact_supervision.class_weights.positive / negative need "
                    f"{slots.count} entries each ({slots.name} slot order)")
            self.class_weights = (pos, neg)

    def fold(self, logits: Tensor, gt: Tensor, mask: Tensor, batch: dict
             ) -> tuple[Tensor, Tensor, Tensor]:
        """:func:`fold_to_groups` under this run's contact set."""
        return fold_to_groups(logits, gt, mask, self.members, batch)

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
                f"dataset's data.contact_set slot count must agree")

        weight = mask
        if self.class_weights is not None:
            pos, neg = self.class_weights
            weight = mask * (gt * pos[None, :] + (1.0 - gt) * neg[None, :])
        numerator, mass, anchor = _bce(logits, gt, mask, weight)
        raw = {"bce": (self.weight * numerator, mass)}
        if self.layer_weight > 0.0:
            # Deep supervision: the same BCE on every INTERMEDIATE layer, pooled into one
            # term (the last entry of logits_layers is `logits`, already supervised).
            layer_num = torch.zeros((), device=self.device, dtype=self.dtype)
            layer_mass = 0.0
            for tensor in out["contact"]["logits_layers"][:-1]:
                num, rows, layer_anchor = _bce(
                    tensor.to(self.device, self.dtype), gt, mask, weight)
                layer_num = layer_num + num
                layer_mass += rows
                anchor = anchor + layer_anchor
            raw["bce_layer"] = (self.layer_weight * self.weight * layer_num, layer_mass)

        detached = logits.detach()
        group_logits, group_gt, group_mask = self.fold(detached, gt, mask, batch)
        stats = torch.cat(
            [contact_counts(detached, gt, mask, THRESHOLD).reshape(-1),
             contact_counts(group_logits, group_gt, group_mask, THRESHOLD).reshape(-1)]
            + [contact_counts(group_logits, group_gt, group_mask, t).sum(dim=0)
               for t in CURVE_THRESHOLDS])
        scalars = {"n_active": float((mask > 0).sum()),
                   "pos_rate": float((torch.sigmoid(detached) > THRESHOLD)
                                     .to(self.dtype).mean())}
        return LossResult(
            terms=self._terms(raw, anchor),
            scalars=scalars,
            stats=stats.to(self.device),
        )

    def metrics(self, stats: Tensor) -> dict[str, float]:
        width = len(COUNT_NAMES)
        n_slot = self.contact_set.count * width
        n_group = NUM_KINDYN_GROUPS * width
        slot_counts = stats[:n_slot].reshape(self.contact_set.count, width)
        counts = stats[n_slot:n_slot + n_group].reshape(NUM_KINDYN_GROUPS, width)
        curve = stats[n_slot + n_group:].reshape(len(CURVE_THRESHOLDS), width)
        micro = prf1(counts.sum(dim=0))
        out = {key: micro[key] for key in ("f1", "precision", "recall", "iou")}
        out["precision_at_r90"] = precision_at_recall(curve, CURVE_RECALL)
        for group, row in zip(KINDYN_GROUP_NAMES, counts):
            scores = prf1(row)
            for key in ("f1", "precision", "recall"):
                out[f"groups/{group}_{key}"] = scores[key]
        slot_micro = prf1(slot_counts.sum(dim=0))
        for key in ("f1", "precision", "recall", "iou"):
            out[f"slots_{key}"] = slot_micro[key]
        for slot, row in zip(self.contact_set.slot_names, slot_counts):
            scores = prf1(row)
            for key in ("f1", "precision", "recall"):
                out[f"slots/{slot}_{key}"] = scores[key]
        return out


def _bce(logits: Tensor, gt: Tensor, mask: Tensor, weight: Tensor
         ) -> tuple[Tensor, float, Tensor]:
    """Per-slot weighted BCE ``(numerator, mass, graph anchor)`` of ONE set of logits.

    ``numerator`` sums each slot's weighted mean BCE over its supervised rows
    (an unsupervised slot adds zero), ``mass`` is the slot count K. Ignored
    elements are replaced by zero BEFORE the loss (``NaN * 0`` is still
    ``NaN``), and the anchor is that masked tensor's zero.
    """
    safe = torch.where(mask > 0, logits, torch.zeros_like(logits))
    per_element = F.binary_cross_entropy_with_logits(safe, gt, reduction="none")
    slot_mass = weight.sum(dim=0)                                               # (K,)
    slot_mean = (per_element * weight).sum(dim=0) / slot_mass.clamp(min=1e-12)
    return slot_mean[slot_mass > 0].sum(), float(logits.shape[1]), safe.sum() * 0.0


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


__all__ = ["ContactLoss", "THRESHOLD", "CURVE_THRESHOLDS", "CURVE_RECALL",
           "fold_to_groups", "group_members", "precision_at_recall"]
