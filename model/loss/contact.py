"""Contact loss (BCE or focal) over the six kindyn groups.

Reads ``out["contact"]["joint_logits"] (B, 6)`` — one logit per contact token,
in :data:`~model.loss.KINDYN_GROUP_NAMES` order — against the collated
``contact_gt`` / ``contact_valid`` / ``contact_conf`` labels.

The supervision weight of an element is ``contact_valid * contact_conf`` (the
confidence factor is switched off by ``contact_supervision.confidence_weights:
false``), so an unlabelled joint-frame contributes exactly nothing and a
low-confidence label contributes proportionally less. Masked elements are
replaced BEFORE the loss rather than multiplied out afterwards: ``NaN * 0`` is
still ``NaN``, and an untracked video frame carries no meaningful logit.

Two precision-oriented options reshape that weight AT TRAIN TIME ONLY (both
enter the mass, so the loss stays a weighted mean and the gradient-balanced
``weight`` keeps meaning; the test loss stays the plain protocol so arms remain
comparable):

* ``neg_weight`` multiplies every NEGATIVE (no-contact) row and ``pos_weight``
  (one factor per group) every POSITIVE row — cost-sensitive BCE. ``neg_weight
  > 1`` buys precision (a false contact costs more than a missed one); a
  ``pos_weight`` above 1 on the heel groups (~3 % positive prior) stops the
  head from ignoring them outright.
* ``transition_tolerance: k`` drops the ``k`` frames on either side of every
  GT contact transition of a clip from the LOSS (never from the metrics): the
  automatic train labels are motion-gated, so exact on/off frames are the
  least certain rows, and a quarter of the test errors sit within 2 frames of
  a transition. Transitions are only defined between two consecutive VALID
  frames of the same clip.

Note that a class cost moves the 0.5 operating point by construction (the
minimiser is ``p = w+ q / (w+ q + w- (1 - q))`` for true posterior ``q``, so
``neg_weight 2`` predicts contact only for ``q > 2/3``): compare arms on
``precision_at_r90`` / the threshold curve, not on the 0.5 numbers.

``criterion: bce`` is the plain binary cross-entropy (calibrated
probabilities when the class costs are 1, constant gradient scale — what WHAM /
GVHMR / TRACE use for their contact heads). ``criterion: focal`` is the asymmetric focal BCE:
``alpha`` weights the POSITIVE class (``alpha_t = a * gt + (1 - a) * (1 -
gt)``) and ``gamma`` focuses on hard examples. The class prior here is near
balanced (~47 % positive over the labelled joint-frames), so focal's ``alpha``
is not a rare-class correction but a precision/recall trade, and its
``(1 - p)^gamma`` factor fades the contact gradient as predictions sharpen.

Metrics are micro P / R / F1 / IoU at threshold 0.5 over the whole split, plus
per-group F1 — a micro score otherwise hides a weak heel behind four strong
limbs — and ``precision_at_r90``: the micro precision at the operating point
whose recall is 0.9, interpolated on the 0.02..0.9 threshold curve (the
threshold-free number the precision arm is about; NaN when no curve point
brackets that recall).
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
#: Thresholds of the accumulated P/R curve (``precision_at_r90``).
CURVE_THRESHOLDS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
#: Recall at which the curve precision is reported.
CURVE_RECALL = 0.9


def transition_keep_mask(gt: Tensor, valid: Tensor, seq_len: int, tolerance: int) -> Tensor:
    """``(B*T, D)`` bool: False within ``tolerance`` frames of a GT transition.

    ``gt``/``valid`` are the clip-major flattened labels (``B*T`` rows of ``T``
    consecutive frames each). A transition between frames ``t`` and ``t + 1``
    (both valid, labels differ) drops frames ``t - tolerance + 1 .. t +
    tolerance``. ``tolerance <= 0`` keeps everything.
    """
    rows, dims = gt.shape
    if tolerance <= 0 or seq_len < 2:
        return torch.ones(rows, dims, dtype=torch.bool, device=gt.device)
    tolerance = min(tolerance, seq_len)      # a wider window than the clip drops it whole
    n_clips = rows // seq_len
    positive = (gt > 0.5).view(n_clips, seq_len, dims)
    ok = (valid > 0).view(n_clips, seq_len, dims)
    change = (positive[:, 1:] != positive[:, :-1]) & ok[:, 1:] & ok[:, :-1]   # [n, T-1, D]
    drop = torch.zeros(n_clips, seq_len, dims, dtype=torch.bool, device=gt.device)
    for offset in range(-tolerance + 1, tolerance + 1):
        lo, hi = max(0, offset), min(seq_len, seq_len - 1 + offset)
        drop[:, lo:hi] |= change[:, lo - offset:hi - offset]
    return (~drop).view(rows, dims)


class ContactLoss(Loss):
    """Confidence-weighted BCE / asymmetric focal BCE on the six contact logits."""

    name = "contact"
    stat_names = tuple(
        f"{group}/{count}" for group in KINDYN_GROUP_NAMES for count in COUNT_NAMES
    ) + tuple(
        f"curve/{threshold}/{count}" for threshold in CURVE_THRESHOLDS for count in COUNT_NAMES)

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        super().__init__(cfg, model, device)
        cs = cfg["contact_supervision"]
        self.weight = float(cs["weight"])
        self.criterion = str(cs["criterion"])
        if self.criterion not in ("bce", "focal"):
            raise ValueError(
                f"contact_supervision.criterion must be 'bce' or 'focal'; got "
                f"{self.criterion!r}")
        self.term_names = (self.criterion,)
        self.alpha = float(cs["focal_alpha"])
        self.gamma = float(cs["focal_gamma"])
        self.use_confidence = bool(cs["confidence_weights"])
        self.neg_weight = float(cs["neg_weight"])
        pos_weight = [float(w) for w in cs["pos_weight"]]
        if len(pos_weight) != NUM_KINDYN_GROUPS:
            raise ValueError(
                f"contact_supervision.pos_weight needs {NUM_KINDYN_GROUPS} factors "
                f"({', '.join(KINDYN_GROUP_NAMES)}); got {len(pos_weight)}")
        self.pos_weight = torch.tensor(pos_weight, dtype=self.dtype, device=self.device)
        self.transition_tolerance = int(cs["transition_tolerance"])

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
        per_element = F.binary_cross_entropy_with_logits(safe, gt, reduction="none")
        if self.criterion == "focal":
            pt = torch.exp(-per_element)
            alpha_t = self.alpha * gt + (1.0 - self.alpha) * (1.0 - gt)
            per_element = alpha_t * (1.0 - pt) ** self.gamma * per_element

        # Loss weight: validity x confidence, and at train time x class cost x
        # transition tolerance (the test loss keeps the plain protocol).
        loss_mask = mask
        if train:
            loss_mask = mask * (gt * self.pos_weight + (1.0 - gt) * self.neg_weight)
            if self.transition_tolerance > 0:
                keep = transition_keep_mask(gt, batch["contact_valid"].to(self.device),
                                            int(batch["seq_len"]), self.transition_tolerance)
                loss_mask = loss_mask * keep.to(self.dtype)
        numerator = self.weight * (per_element * loss_mask).sum()
        mass = float(loss_mask.sum())
        anchor = safe.sum() * 0.0

        detached = logits.detach()
        stats = torch.cat(
            [contact_counts(detached, gt, mask, THRESHOLD).reshape(-1)]
            + [contact_counts(detached, gt, mask, t).sum(dim=0) for t in CURVE_THRESHOLDS])
        scalars = {"n_active": float((mask > 0).sum()),
                   "pos_rate": float((torch.sigmoid(detached) > THRESHOLD)
                                     .to(self.dtype).mean())}
        return LossResult(
            terms=self._terms({self.criterion: (numerator, mass)}, anchor),
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


__all__ = ["ContactLoss", "THRESHOLD", "CURVE_THRESHOLDS", "transition_keep_mask",
           "precision_at_recall"]
