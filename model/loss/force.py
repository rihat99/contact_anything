"""Supervised six-group force loss against the kindyn ground truth.

Reads ``out["force"]["forces"] (B, 6, 3)`` — body-weight units in the
**body-root frame**, which is the frame the loader rotates the GT into (by the
kindyn root quaternion), so a decoder-token head learns that frame directly and
no camera extrinsics enter this objective at all. A head that predicts in its
OWN body frame (the refiner: ``out["force"]["frame"]`` = world-from-body) has
the GT and the lever arms rotated into that frame first (needs the ``smplx``
GT group for the kindyn root rotation). Groups are in
:data:`~model.loss.KINDYN_GROUP_NAMES` order.

Four terms:

* ``force`` — Huber between prediction and GT, summed over the 3 components, on
  valid **in-contact** limb-frames. Huber because the GT solve has heavy tails
  (in-contact ``|f|`` p99 ~ 1.6 bw, max 48 bw): quadratic near zero for a clean
  mean, linear past ``huber_delta_bw`` so spikes cannot dominate the gradient.
  Limb-frames whose GT magnitude exceeds ``outlier_bw`` are excluded outright
  (solver blowups on bad reconstructions). ``group_weights`` turns the term into
  a per-group weighted mean — the weights enter numerator AND mass, so an
  upweighted group gets proportionally more gradient without changing the term's
  scale. That knob exists because with uniform weights the legs collapse to
  exactly zero: the hands dominate both the contact rate and the GT magnitude.
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

``force_supervision.confidence`` weights every term's rows by kindyn's per-frame
solve confidence — into numerator and mass both, so it reweights rows without
changing any term's scale. The reported MAE stays unweighted.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from model.loss import Loss, LossResult
from utils.metrics import mean_from_stats

_TERM_NAMES = ("force", "noncontact", "sum_force", "sum_torque")


class ForceLoss(Loss):
    """Kindyn GT-force supervision for the six-token force branch."""

    name = "force"
    stat_names = ("mae_num", "mae_mass", "noncontact_num", "noncontact_mass")

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        super().__init__(cfg, model, device)
        loss_cfg = cfg["force_supervision"]["loss"]
        self.use_confidence = bool(cfg["force_supervision"]["confidence"])
        self.weights = {name: float(loss_cfg[name]) for name in _TERM_NAMES}
        self.term_names = tuple(n for n in _TERM_NAMES if self.weights[n] != 0.0)
        if not self.term_names:
            raise ValueError(
                "force_supervision: every loss weight is 0 — disable the section instead")
        self.huber_delta = float(loss_cfg["huber_delta_bw"])
        self.huber_delta_bwm = float(loss_cfg["huber_delta_bwm"])
        self.outlier_bw = float(loss_cfg["outlier_bw"])
        group_weights = loss_cfg["group_weights"]
        self.group_weights = (
            None if group_weights is None
            else torch.tensor([float(w) for w in group_weights],
                              dtype=self.dtype, device=self.device))

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
        conf = (batch["force_conf"].to(self.device, self.dtype) if self.use_confidence
                else torch.ones_like(valid, dtype=self.dtype))           # (B,)

        in_contact = valid[:, None] & contact
        off_contact = valid[:, None] & ~contact
        outlier = torch.zeros_like(in_contact)
        if self.outlier_bw > 0.0:
            outlier = in_contact & (
                torch.linalg.vector_norm(gt, dim=-1) > self.outlier_bw)
            in_contact = in_contact & ~outlier

        w_contact = in_contact.to(self.dtype) * conf[:, None]            # (B,K)
        w_free = off_contact.to(self.dtype) * conf[:, None]
        if self.group_weights is not None:
            if self.group_weights.numel() != pred.shape[1]:
                raise ValueError(
                    f"force_supervision.loss.group_weights has "
                    f"{self.group_weights.numel()} entries but the model "
                    f"predicts {pred.shape[1]} force groups")
            w_contact = w_contact * self.group_weights[None, :]

        huber = F.smooth_l1_loss(
            pred, gt, reduction="none", beta=self.huber_delta).sum(dim=-1)
        raw: dict[str, tuple[Tensor, float]] = {
            "force": ((huber * w_contact).sum(), float(w_contact.sum())),
            "noncontact": ((pred.abs().sum(dim=-1) * w_free).sum(),
                           float(w_free.sum())),
        }

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
            free_mag = torch.linalg.vector_norm(pred, dim=-1)
            stats = torch.tensor([
                float((err * in_contact).sum()), float(in_contact.sum()),
                float((free_mag * off_contact).sum()), float(off_contact.sum()),
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
        return LossResult(terms=self._terms(weighted, anchor),
                          scalars=scalars, stats=stats)

    def metrics(self, stats: Tensor) -> dict[str, float]:
        return {
            "mae": mean_from_stats(float(stats[0]), float(stats[1])),
            "noncontact_mag": mean_from_stats(float(stats[2]), float(stats[3])),
        }


__all__ = ["ForceLoss"]
