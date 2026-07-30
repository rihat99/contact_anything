"""Supervised six-group force loss against kindyn ground truth.

Reads ``out["force"]["joint_forces"] (B, K, 3)`` — body-weight units in the
**body-root frame** (the corpus loader rotates GT world forces by the kindyn
root quaternion, so the head learns that frame directly; no camera extrinsics
anywhere) — and the collated ``force_gt (B, K, 3)`` / ``force_contact (B, K)``
/ ``force_valid (B)`` batch keys in the kindyn group order
(``left_hand, right_hand, left_foot, right_foot, left_ankle, right_ankle``).

Two terms, mirroring the :class:`~contact.physics.loss.PhysicsLoss` term
contract (each returns ``(weighted_numerator, mass)`` so the trainer's
:func:`~contact.losses.ddp_global_mean_term` reduction gives the exact global
mean under DDP):

- ``force`` — smooth-L1 (Huber) between prediction and GT, summed over the 3
  components, on valid **in-contact** limb-frames. Huber because the GT solve
  has heavy tails (in-contact ``|f|`` p99 ≈ 1.6 bw, max 48 bw): quadratic near
  zero for a clean mean, linear past ``huber_delta_bw`` so spikes cannot
  dominate the gradient. Limb-frames whose GT magnitude exceeds ``outlier_bw``
  are excluded outright (solver blowups on bad reconstructions). Optional
  ``group_weights`` (kindyn group order) turn the term into a per-group
  weighted mean — weights enter both the numerator and the mass, so upweighted
  groups get proportionally more gradient without changing the term's scale.
  Motivation: with uniform weights the first supervised run collapsed the four
  leg groups to exactly zero (hands dominate both contact rate and GT
  magnitude); upweighting the legs counteracts that.
- ``noncontact`` — L1 magnitude penalty on valid **non-contact** limb-frames.
  GT is identically zero there by construction (the kindyn solve only placed
  forces where its contacts_2-derived labels said contact), so this is the
  label-gated zero-force regulariser; L1's constant slope at ``‖f‖ → 0``
  admits exact zeros (the t7hinge lesson), and its weight balances the two
  populations independently of their frame counts.

``target_frame: center`` supervises only row ``T // 2`` of each clip (the
:func:`contact.engine.select_temporal_supervision` convention — odd ``T``
required); the temporal module still attends the full window.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

_TERM_NAMES = ("force", "noncontact")


class ForceSupervisedLoss:
    """GT-force supervision for the six-token force branch.

    :param cfg: resolved run config; reads ``force_supervision.*``.
    :param device: device the loss runs on (predictions are moved to it).
    :param dtype: floating dtype (float32).
    """

    def __init__(
        self,
        cfg: dict,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        fs = cfg["force_supervision"]
        self.target_frame = str(fs["target_frame"])
        loss_cfg = fs["loss"]
        self.weights = {name: float(loss_cfg[name]) for name in _TERM_NAMES}
        self.huber_delta = float(loss_cfg["huber_delta_bw"])
        self.outlier_bw = float(loss_cfg["outlier_bw"])
        gw = loss_cfg["group_weights"]
        self.group_weights = (
            None if gw is None
            else torch.tensor([float(w) for w in gw], dtype=dtype))
        self.device = torch.device(device)
        self.dtype = dtype

    def __call__(self, out: dict, batch: dict) -> tuple[Tensor, dict[str, Any]]:
        return self.forward(out, batch)

    def forward(self, out: dict, batch: dict) -> tuple[Tensor, dict[str, Any]]:
        """Return ``(total, parts)``.

        :param out: forward output — reads ``out["force"]["joint_forces"]
            (B, K, 3)`` (grads live).
        :param batch: reads ``force_gt (B, K, 3)``, ``force_contact (B, K)``,
            ``force_valid (B)``, ``frame_valid (B)``, ``seq_len``.
        :returns: ``(total, parts)`` where ``parts["terms"][name]`` carries
            ``weighted_numerator_tensor`` + ``weight_mass`` for exact DDP
            reduction; ``parts["force_mae"]`` is the headline monitor entry
            (mean prediction-error norm on in-contact limb-frames, bw) in the
            same numerator/mass form for exact eval-side reduction.
        """
        pred = out["force"]["joint_forces"].to(self.device, self.dtype)   # (B, K, 3)
        # Graph-connected zero touching every force param (DDP: force params must
        # stay on the backward graph even when a batch has no supervised frames).
        zero_touch = pred.sum() * 0.0

        gt = batch["force_gt"].to(self.device, self.dtype)
        if pred.shape != gt.shape:
            raise ValueError(
                f"force prediction {tuple(pred.shape)} does not match GT "
                f"{tuple(gt.shape)} — force_keypoint_indices and the dataset's "
                f"force groups must agree")
        contact = batch["force_contact"].to(self.device)                  # (B, K) bool
        valid = (batch["force_valid"] & batch["frame_valid"]).to(self.device)  # (B)

        if self.target_frame == "center":
            seq_len = int(batch["seq_len"])
            if seq_len % 2 == 0:
                raise ValueError(
                    f"center-frame supervision requires an odd seq_len; got {seq_len}")
            n_clips = pred.shape[0] // seq_len
            center = seq_len // 2

            def _center(value: Tensor) -> Tensor:
                return value.reshape(n_clips, seq_len, *value.shape[1:])[:, center]

            pred, gt = _center(pred), _center(gt)
            contact, valid = _center(contact), _center(valid)
        elif self.target_frame != "all":
            raise ValueError(
                f"target_frame must be 'all' or 'center'; got {self.target_frame!r}")

        in_contact = valid[:, None] & contact                             # (rows, K)
        off_contact = valid[:, None] & ~contact
        gt_mag = torch.linalg.vector_norm(gt, dim=-1)                     # (rows, K)
        n_outlier = 0
        if self.outlier_bw > 0.0:
            outlier = in_contact & (gt_mag > self.outlier_bw)
            n_outlier = int(outlier.sum())
            in_contact = in_contact & ~outlier

        huber = F.smooth_l1_loss(
            pred, gt, reduction="none", beta=self.huber_delta).sum(dim=-1)  # (rows, K)
        l1_mag = pred.abs().sum(dim=-1)                                     # (rows, K)
        if self.group_weights is not None:
            if self.group_weights.numel() != pred.shape[1]:
                raise ValueError(
                    f"force_supervision.loss.group_weights has "
                    f"{self.group_weights.numel()} entries but the model predicts "
                    f"{pred.shape[1]} force groups")
            gw = self.group_weights.to(self.device)[None, :]                # (1, K)
            force_num = (huber * in_contact * gw).sum()
            force_mass = float((in_contact * gw).sum())
        else:
            force_num = (huber * in_contact).sum()
            force_mass = float(in_contact.sum())
        terms: dict[str, tuple[Tensor, float]] = {
            "force": (force_num, force_mass),
            "noncontact": ((l1_mag * off_contact).sum(), float(off_contact.sum())),
        }

        err_norm = torch.linalg.vector_norm(pred - gt, dim=-1)            # (rows, K)
        mae_mass = float(in_contact.sum())
        diagnostics: dict[str, Any] = {
            "force_mae": {
                "weighted_numerator_tensor": (err_norm * in_contact).sum().detach(),
                "weight_mass": mae_mass,
                "loss": float((err_norm * in_contact).sum().detach()) / max(mae_mass, 1.0),
            },
            "noncontact_mag": float(
                (torch.linalg.vector_norm(pred, dim=-1) * off_contact).sum().detach()
            ) / max(float(off_contact.sum()), 1.0),
            "per_group_mae": [
                float((err_norm[:, k] * in_contact[:, k]).sum().detach())
                / max(float(in_contact[:, k].sum()), 1.0)
                for k in range(pred.shape[1])
            ],
            "n_outlier_excluded": n_outlier,
            "n_supervised_rows": int(valid.sum()),
        }
        return self._assemble(terms, zero_touch, diagnostics)

    def _assemble(
        self,
        terms: dict[str, tuple[Tensor, float]],
        zero_touch: Tensor,
        diagnostics: dict[str, Any],
    ) -> tuple[Tensor, dict[str, Any]]:
        """Weight, normalise, and package the term contract + logging scalars.

        Every nonzero-weight term is always present (mass 0 when it has no data
        this batch), so the term set is fixed by config — not by per-batch
        supervision — which the trainer needs for consistent DDP all-reduce.
        Each term carries the graph-connected ``zero_touch``.
        """
        parts_terms: dict[str, dict[str, Any]] = {}
        total: Tensor | None = None
        for name in _TERM_NAMES:
            if self.weights[name] == 0.0:
                continue
            raw, mass = terms[name]
            weighted = self.weights[name] * raw + zero_touch
            normalized = weighted / max(mass, 1.0)
            total = normalized if total is None else total + normalized
            parts_terms[name] = {
                "weighted_numerator_tensor": weighted,
                "weight_mass": mass,
                "loss": float(normalized.detach()),
            }
        if total is None:                       # every weight is zero (degenerate)
            total = zero_touch

        parts: dict[str, Any] = {"terms": parts_terms, "loss": float(total.detach())}
        parts.update(diagnostics)
        return total, parts


__all__ = ["ForceSupervisedLoss"]
