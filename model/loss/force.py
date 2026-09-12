"""Supervised six-group force loss against the kindyn ground truth.

Reads ``out["force"]["forces"] (B, 6, 3)`` — body-weight units in the
**body-root frame**, which is the frame the loader rotates the GT into (by the
kindyn root quaternion), so a decoder-token head learns that frame directly and
no camera extrinsics enter this objective at all. A head that predicts in its
OWN body frame (the refiner: ``out["force"]["frame"]`` = world-from-body) has
the GT and the lever arms rotated into that frame first (needs the ``smplx``
GT group for the kindyn root rotation). Groups are in
:data:`~model.loss.KINDYN_GROUP_NAMES` order.

Six terms, every one a ``loss.*`` weight:

* ``force`` — Huber between the prediction and GT VECTORS, summed over the 3
  components, on valid **in-contact** limb-frames. Huber because the GT solve
  has heavy tails (in-contact ``|f|`` p99 ~ 1.6 bw, max 48 bw): quadratic near
  zero for a clean mean, linear past ``huber_delta_bw`` so spikes cannot
  dominate the gradient. Limb-frames whose GT magnitude exceeds ``outlier_bw``
  are excluded outright (solver blowups on bad reconstructions).
* ``magnitude`` / ``direction`` — the same rows with the vector error SPLIT:
  ``magnitude`` is the Huber (same ``huber_delta_bw``) on ``|f_pred| - |f_gt|``;
  ``direction`` is ``1 - cos(f_pred, f_gt)`` on the rows whose GT magnitude is
  at least ``direction_min_bw`` (the direction of a near-zero GT force is
  solver noise — a heel row is below 0.1 bw four times out of ten). The
  predicted norm inside the cosine is clamped at :data:`DIRECTION_EPS_BW`, so
  the term is finite at the zero-initialised head and its gradient there points
  along the GT direction (which also gets ``magnitude`` off its zero-gradient
  start). The split lets the two error kinds be weighted apart: the vector
  Huber charges a 20-degree error on a 1 bw force ten times more than on a
  0.1 bw one.
* ``noncontact`` — L1 magnitude penalty on valid **non-contact** limb-frames,
  where the GT is identically zero by construction (kindyn only solved forces
  where its own contact mask said contact). L1's constant slope at ``|f| -> 0``
  admits exact zeros, which a quadratic never reaches. The model's contact gate
  does this job in the forward pass instead, so gated builds set this to 0.
* ``sum_force`` — Huber on the NET force ``sum_i f_i`` over all six groups
  regardless of the contact mask (GT is exactly zero off-contact, and a gated
  prediction is ~0 there). A row is skipped when force-invalid or when ANY group
  is an outlier: one blown-up group poisons the whole sum.
* ``sum_torque`` — the same on the net torque ``sum_i r_i x f_i`` (bw*m, its own
  ``huber_delta_bwm``) with the loader's root-frame lever arms. The SAME arms
  enter both sides, so the choice of origin is a consistency statement, not a
  physics claim.

``group_weights`` turns the per-limb terms (``force`` / ``magnitude`` /
``direction``) into per-group weighted means — the weights enter numerator AND
mass, so an upweighted group gets proportionally more gradient without changing
the term's scale. That knob exists because with uniform weights the legs
collapse to exactly zero: the hands dominate both the contact rate and the GT
magnitude.

``layer_weight`` adds deep supervision: the four PER-LIMB terms (``force`` /
``magnitude`` / ``direction`` / ``noncontact``; not the sums, which are a
statement about one body's whole wrench) are repeated on every INTERMEDIATE
layer's forces of an iterative refiner (``out["force"]["forces_layers"][:-1]``;
the last entry IS ``forces``) and pooled per term into ``<term>_layer``,
numerators and masses summed over the layers, at ``layer_weight`` times the
term's own weight. Every layer predicts in the SAME body frame
(``out["force"]["frame"]``) and the row masks come from the GT, so the whole GT
side is built once. Metrics stay the final layer's.

``force_supervision.confidence`` weights every term's rows by kindyn's per-frame
solve confidence raised to ``confidence_power`` (1 = the raw confidence, 0.5
compresses it — the corpus confidence is 0.99 at the median and 0.54 at p5, so
the exponent only matters for the low tail; 0 = flat) — into numerator and
mass both, so it reweights rows without changing any term's scale. The reported
metrics stay unweighted: ``mae`` (vector error, in-contact rows), ``mag_mae``
(``||f_pred| - |f_gt||``, same rows), ``angle_deg`` (on the direction rows) and
``noncontact_mag``.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from model.loss import Loss, LossResult
from utils.metrics import mean_from_stats

_TERM_NAMES = ("force", "magnitude", "direction", "noncontact", "sum_force", "sum_torque")
#: The PER-LIMB terms, which read one layer's forces alone, so deep supervision
#: (``layer_weight``) can repeat them on the refiner's intermediate layers.
LAYER_TERM_NAMES = ("force", "magnitude", "direction", "noncontact")
#: Floor of the predicted norm inside the direction cosine (bw): bounds the gradient
#: at ``1 / DIRECTION_EPS_BW`` per row and defines the term at a zero prediction.
DIRECTION_EPS_BW = 0.05


def _norm(x: Tensor) -> Tensor:
    """``|x|`` over the last axis with a zero (not NaN) gradient at ``x = 0``."""
    return torch.sqrt(x.pow(2).sum(dim=-1) + 1e-12)


class ForceLoss(Loss):
    """Kindyn GT-force supervision for the six-token force branch."""

    name = "force"
    stat_names = ("mae_num", "mae_mass", "noncontact_num", "noncontact_mass",
                  "mag_num", "angle_num", "angle_mass")

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        super().__init__(cfg, model, device)
        section = cfg["force_supervision"]
        loss_cfg = section["loss"]
        self.use_confidence = bool(section["confidence"])
        self.confidence_power = float(section["confidence_power"])
        self.weights = {name: float(loss_cfg[name]) for name in _TERM_NAMES}
        self.term_names = tuple(n for n in _TERM_NAMES if self.weights[n] != 0.0)
        if not self.term_names:
            raise ValueError(
                "force_supervision: every loss weight is 0 — disable the section instead")
        self.layer_weight = float(section["layer_weight"])
        #: The subset deep supervision repeats on every intermediate layer.
        self.layer_terms = tuple(n for n in LAYER_TERM_NAMES if self.weights[n] != 0.0)
        if self.layer_weight > 0.0:
            self.term_names += tuple(f"{n}_layer" for n in self.layer_terms)
        self.huber_delta = float(loss_cfg["huber_delta_bw"])
        self.huber_delta_bwm = float(loss_cfg["huber_delta_bwm"])
        self.outlier_bw = float(loss_cfg["outlier_bw"])
        self.direction_min_bw = float(loss_cfg["direction_min_bw"])
        group_weights = loss_cfg["group_weights"]
        self.group_weights = (
            None if group_weights is None
            else torch.tensor([float(w) for w in group_weights],
                              dtype=self.dtype, device=self.device))

    def _limb_terms(self, pred: Tensor, gt: Tensor, unit_gt: Tensor, mag_gt: Tensor,
                    row_weights: tuple[Tensor, Tensor, Tensor]
                    ) -> dict[str, tuple[Tensor, float]]:
        """The four PER-LIMB terms (:data:`LAYER_TERM_NAMES`) of ONE prediction, un-weighted.

        The GT side and the row weights ``(in-contact, direction, off-contact)`` are the
        same for every layer — a layer changes nothing but ``pred``.
        """
        w_contact, w_direction, w_free = row_weights
        mag_pred = _norm(pred)
        huber = F.smooth_l1_loss(
            pred, gt, reduction="none", beta=self.huber_delta).sum(dim=-1)
        huber_mag = F.smooth_l1_loss(
            mag_pred, mag_gt, reduction="none", beta=self.huber_delta)
        # cos(f_pred, f_gt) with the predicted norm floored: finite at f_pred = 0, where
        # the gradient is -u_gt / eps (the zero-init head's way off its start).
        cosine = (pred * unit_gt).sum(dim=-1) / mag_pred.clamp(min=DIRECTION_EPS_BW)
        return {
            "force": ((huber * w_contact).sum(), float(w_contact.sum())),
            "magnitude": ((huber_mag * w_contact).sum(), float(w_contact.sum())),
            "direction": (((1.0 - cosine) * w_direction).sum(), float(w_direction.sum())),
            "noncontact": ((mag_pred * w_free).sum(), float(w_free.sum())),
        }

    def layer_raw(self, layers: list[Tensor], gt: Tensor, unit_gt: Tensor, mag_gt: Tensor,
                  row_weights: tuple[Tensor, Tensor, Tensor]
                  ) -> tuple[dict[str, tuple[Tensor, float]], Tensor]:
        """Deep supervision (``layer_weight``): :meth:`_limb_terms` on the INTERMEDIATE layers.

        One ``<term>_layer`` per term, the layers pooled (numerators and masses summed), at
        ``layer_weight`` times the term's own weight; plus the layers' graph anchor.
        """
        sums = {name: torch.zeros((), device=self.device, dtype=self.dtype)
                for name in self.layer_terms}
        masses = {name: 0.0 for name in self.layer_terms}
        anchor = torch.zeros((), device=self.device, dtype=self.dtype)
        for tensor in layers:
            pred = tensor.to(self.device, self.dtype)
            raw = self._limb_terms(pred, gt, unit_gt, mag_gt, row_weights)
            anchor = anchor + pred.sum() * 0.0
            for name in self.layer_terms:
                numerator, mass = raw[name]
                sums[name] = sums[name] + self.weights[name] * numerator
                masses[name] += mass
        return ({f"{name}_layer": (self.layer_weight * sums[name], masses[name])
                 for name in self.layer_terms}, anchor)

    def __call__(self, out: dict, batch: dict, *, train: bool) -> LossResult:
        pred = out["force"]["forces"].to(self.device, self.dtype)  # (B,K,3)
        anchor = pred.sum() * 0.0
        gt = batch["force_gt"].to(self.device, self.dtype)
        lever = batch["force_lever"].to(self.device, self.dtype)         # (B,K,3)
        frame = out["force"].get("frame")
        if frame is not None:
            # The head predicts in ITS body frame (world-from-body `frame`); the loader's GT is
            # in the kindyn root frame. rel = frame^T R_gt_root is world-independent.
            rel = frame.detach().to(self.device, self.dtype).transpose(1, 2) @ batch[
                "smplx_root_rot"].to(self.device, self.dtype)            # (B,3,3)
            gt = torch.einsum("bij,bkj->bki", rel, gt)
            lever = torch.einsum("bij,bkj->bki", rel, lever)
        if pred.shape != gt.shape:
            raise ValueError(
                f"force prediction {tuple(pred.shape)} does not match the GT "
                f"{tuple(gt.shape)} — model.force.keypoint_indices and the "
                f"dataset's force groups must agree")
        contact = batch["force_contact"].to(self.device)                 # (B,K)
        valid = (batch["force_valid"] & batch["frame_valid"]).to(self.device)
        if self.use_confidence:
            conf = batch["force_conf"].to(self.device, self.dtype).clamp(min=0.0)
            conf = conf ** self.confidence_power                         # (B,)
        else:
            conf = torch.ones_like(valid, dtype=self.dtype)

        mag_gt = _norm(gt)
        mag_pred = _norm(pred)
        in_contact = valid[:, None] & contact
        off_contact = valid[:, None] & ~contact
        outlier = torch.zeros_like(in_contact)
        if self.outlier_bw > 0.0:
            outlier = in_contact & (mag_gt > self.outlier_bw)
            in_contact = in_contact & ~outlier
        direction_rows = in_contact & (mag_gt >= self.direction_min_bw)

        w_contact = in_contact.to(self.dtype) * conf[:, None]            # (B,K)
        w_direction = direction_rows.to(self.dtype) * conf[:, None]
        w_free = off_contact.to(self.dtype) * conf[:, None]
        if self.group_weights is not None:
            if self.group_weights.numel() != pred.shape[1]:
                raise ValueError(
                    f"force_supervision.loss.group_weights has "
                    f"{self.group_weights.numel()} entries but the model "
                    f"predicts {pred.shape[1]} force groups")
            w_contact = w_contact * self.group_weights[None, :]
            w_direction = w_direction * self.group_weights[None, :]

        unit_gt = gt / mag_gt.clamp(min=1e-6)[..., None]
        row_weights = (w_contact, w_direction, w_free)
        raw: dict[str, tuple[Tensor, float]] = self._limb_terms(
            pred, gt, unit_gt, mag_gt, row_weights)

        # Net force / net torque over ALL six groups per eligible row.
        sum_rows = valid & ~outlier.any(dim=-1)
        w_sum = sum_rows.to(self.dtype) * conf
        sum_huber = F.smooth_l1_loss(
            pred.sum(dim=1), gt.sum(dim=1), reduction="none",
            beta=self.huber_delta).sum(dim=-1)
        raw["sum_force"] = ((sum_huber * w_sum).sum(), float(w_sum.sum()))

        torque_rows = sum_rows & torch.isfinite(lever).all(dim=-1).all(dim=-1)
        # Zero the skipped rows' arms BEFORE the cross product: a non-finite
        # lever would otherwise turn `huber * mask` into NaN * 0 = NaN.
        lever_ok = torch.where(
            torque_rows[:, None, None], lever, torch.zeros_like(lever))
        torque_huber = F.smooth_l1_loss(
            torch.linalg.cross(lever_ok, pred, dim=-1).sum(dim=1),
            torch.linalg.cross(lever_ok, gt, dim=-1).sum(dim=1),
            reduction="none", beta=self.huber_delta_bwm).sum(dim=-1)
        w_torque = torque_rows.to(self.dtype) * conf
        raw["sum_torque"] = ((torque_huber * w_torque).sum(), float(w_torque.sum()))

        with torch.no_grad():
            err = torch.linalg.vector_norm(pred - gt, dim=-1)            # (B,K)
            mag_err = (mag_pred - mag_gt).abs()
            angle = torch.rad2deg(torch.acos(
                ((pred * unit_gt).sum(dim=-1) / mag_pred.clamp(min=1e-6)).clamp(-1.0, 1.0)))
            stats = torch.tensor([
                float((err * in_contact).sum()), float(in_contact.sum()),
                float((mag_pred * off_contact).sum()), float(off_contact.sum()),
                float((mag_err * in_contact).sum()),
                float((angle * direction_rows).sum()), float(direction_rows.sum()),
            ], dtype=torch.float64, device=self.device)
        scalars = {
            "mae": mean_from_stats(float(stats[0]), float(stats[1])),
            "n_outlier": float(outlier.sum()),
            "n_rows": float(valid.sum()),
        }
        terms = {name: raw[name] for name in _TERM_NAMES
                 if self.weights[name] != 0.0}
        weighted = {name: (self.weights[name] * numerator, mass)
                    for name, (numerator, mass) in terms.items()}
        if self.layer_weight > 0.0:
            layer_terms, layer_anchor = self.layer_raw(
                out["force"]["forces_layers"][:-1], gt, unit_gt, mag_gt, row_weights)
            weighted.update(layer_terms)
            anchor = anchor + layer_anchor
        return LossResult(terms=self._terms(weighted, anchor),
                          scalars=scalars, stats=stats)

    def metrics(self, stats: Tensor) -> dict[str, float]:
        return {
            "mae": mean_from_stats(float(stats[0]), float(stats[1])),
            "noncontact_mag": mean_from_stats(float(stats[2]), float(stats[3])),
            "mag_mae": mean_from_stats(float(stats[4]), float(stats[1])),
            "angle_deg": mean_from_stats(float(stats[5]), float(stats[6])),
        }


__all__ = ["ForceLoss", "DIRECTION_EPS_BW"]
