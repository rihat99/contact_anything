"""Kindyn-MHR pseudo-GT pose supervision for the pose-temporal module (E2).

Reads ``out["mhr"]["mhr_model_params"] (B, 204)`` — graph-live through the
recomputed final pose output when ``model.pose_temporal`` is enabled — and the
collated ``pose_gt_q (B, 132)`` / ``pose_valid (B)`` / ``frame_valid (B)`` batch
keys (``scripts/convert_kindyn_to_mhr.py`` targets, world-frame MHR ``q``).

The comparison runs in **q space** (the rig's 125 local pose channels): the
prediction's ``mhr_model_params`` go through the BetterHuman body's
``from_classic`` (differentiable in the parameters), the target ``q`` is used
as stored. Parameter space is NOT usable directly — the 130 body-pose slots
project onto the 125-dim ``q`` manifold (the last six slots are coupled;
``to_classic(from_classic(p)) != p`` there), so a parameter-space loss would
chase components the rig cannot represent. The free-flyer root (world- vs
camera-frame, never comparable here) is not supervised.

Terms follow the force/motion ``(weighted_numerator, mass)`` contract so the
trainer's exact-DDP reduction applies: ``pose`` (per-frame wrapped Huber) and,
when ``loss.acc > 0``, ``acc`` — a Huber on the clip-wise SECOND DIFFERENCES of
the q channels against the target's (valid frame triples only). The per-frame
term alone moves the pose toward kindyn without smoothing it (E2 v1: MAE
0.096 -> 0.070 while the acceleration ratio stayed ~5.7x); the acc term is the
explicit smoothness objective. Diagnostics are packed
as a float64 sufficient-statistics vector (see :data:`STAT_NAMES`): the
per-frame pose MAE and the clip-wise pose-channel acceleration RMS of the
prediction and the target — the smoothness pair the experiment is about
(pred/GT ratio 1.0 = as smooth as kindyn; the frozen model's own value is the
epoch-0 row).
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

#: Slots of the float64 stats vector (all-reduced with SUM under DDP).
STAT_NAMES = ("sum_abs_err", "n_channel_rows", "sum_acc2_pred", "sum_acc2_gt",
              "n_acc_rows")
#: Supervised q slice: the 125 local pose channels after the free-flyer root.
POSE_SLOTS = slice(7, 132)
#: Channel count of the supervised slice.
N_POSE_CHANNELS = 125


def _wrap(diff: Tensor) -> Tensor:
    """Wrap channel differences to ``(-pi, pi]`` (euler-like channels)."""
    return torch.remainder(diff + math.pi, 2.0 * math.pi) - math.pi


class PoseSupervisedLoss:
    """Huber on the 130 body-pose parameter slots against kindyn-MHR targets.

    :param cfg: resolved run config; reads ``pose_supervision.*``.
    :param device: device the loss (and the conversion body) runs on.
    """

    def __init__(
        self,
        cfg: dict,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        import better_human as bh

        from .physics.adapter import _resolve_model_path

        ps = cfg["pose_supervision"]
        self.weight = float(ps["loss"]["pose"])
        self.acc_weight = float(ps["loss"].get("acc", 0.0))
        self.huber_delta = float(ps["loss"]["huber_delta"])
        self.device = torch.device(device)
        self.dtype = dtype
        lod = int(ps["mhr"]["lod"])
        self.body = bh.MHR(
            _resolve_model_path(ps["mhr"]["model_path"], lod),
            lod=lod,
            use_expression=False,
            use_correctives=False,
            compute_mass=False,
            dtype=dtype,
            device=self.device,
        )

    def __call__(self, out: dict, batch: dict) -> tuple[Tensor, dict[str, Any]]:
        return self.forward(out, batch)

    def forward(self, out: dict, batch: dict) -> tuple[Tensor, dict[str, Any]]:
        """Return ``(total, parts)`` with the force/motion term contract."""
        from better_human.bodies import MHRClassic

        params = out["mhr"]["mhr_model_params"].to(self.device, self.dtype)
        # q is shape-independent (adapter invariant), so the identity used for
        # the conversion is irrelevant — detach it either way.
        shape = out["mhr"]["shape"].detach().to(self.device, self.dtype)
        _, q_pred = self.body.from_classic(MHRClassic(
            identity_coeffs=shape, model_parameters=params))
        pred = q_pred[..., POSE_SLOTS]                              # (B, 125)
        zero_touch = pred.sum() * 0.0

        q_gt = batch["pose_gt_q"].to(self.device, self.dtype)       # (B, 132)
        valid = (batch["pose_valid"] & batch["frame_valid"]).to(self.device)
        tgt = q_gt[..., POSE_SLOTS]

        diff = _wrap(pred - tgt)                                    # (B, 125)
        huber = F.smooth_l1_loss(
            diff, torch.zeros_like(diff), reduction="none", beta=self.huber_delta)
        mask = valid.to(self.dtype)                                 # (B,)
        mass = float(valid.sum())
        raw = (huber.sum(dim=-1) * mask).sum()

        terms: dict[str, tuple] = {"pose": (self.weight * raw, mass)}
        if self.acc_weight > 0.0:
            seq_len = int(batch["seq_len"])
            acc_raw = pred.new_zeros(())
            acc_mass = 0.0
            if seq_len >= 3 and pred.shape[0] % seq_len == 0:
                n_clips = pred.shape[0] // seq_len
                p = pred.view(n_clips, seq_len, -1)
                g = tgt.view(n_clips, seq_len, -1)
                v = valid.view(n_clips, seq_len)
                v3 = (v[:, 2:] & v[:, 1:-1] & v[:, :-2]).to(self.dtype)
                acc_diff = _wrap((p[:, 2:] - 2.0 * p[:, 1:-1] + p[:, :-2])
                                 - (g[:, 2:] - 2.0 * g[:, 1:-1] + g[:, :-2]))
                acc_huber = F.smooth_l1_loss(
                    acc_diff, torch.zeros_like(acc_diff), reduction="none",
                    beta=self.huber_delta)
                acc_raw = (acc_huber.sum(dim=-1) * v3).sum()
                acc_mass = float(v3.sum())
            terms["acc"] = (self.acc_weight * acc_raw, acc_mass)

        stats = self._diagnostics(pred.detach(), tgt, valid, int(batch["seq_len"]))
        total = None
        parts_terms: dict[str, Any] = {}
        for name, (weighted_raw, term_mass) in terms.items():
            weighted = weighted_raw + zero_touch
            normalized = weighted / max(term_mass, 1.0)
            total = normalized if total is None else total + normalized
            parts_terms[name] = {
                "weighted_numerator_tensor": weighted,
                "weight_mass": term_mass,
                "loss": float(normalized.detach()),
            }
        parts: dict[str, Any] = {
            "terms": parts_terms,
            "loss": float(total.detach()),
            "stats": stats,
            "pose_mae": float(stats[0] / max(float(stats[1]), 1.0)),
            "n_supervised_rows": int(valid.sum()),
        }
        return total, parts

    @torch.no_grad()
    def _diagnostics(
        self, pred: Tensor, tgt: Tensor, valid: Tensor, seq_len: int,
    ) -> Tensor:
        """Float64 sufficient statistics (:data:`STAT_NAMES`).

        The acceleration rows need three consecutive VALID frames of one clip;
        clips are flat clip-major/frame-minor, so a ``(n_clips, T, ...)`` view
        recovers them.
        """
        stats = torch.zeros(len(STAT_NAMES), dtype=torch.float64, device=pred.device)
        err = _wrap(pred - tgt).abs().to(torch.float64)             # (B, 125)
        weight = valid.to(torch.float64)
        stats[0] = (err.sum(dim=-1) * weight).sum()
        stats[1] = weight.sum() * err.shape[-1]
        if seq_len >= 3 and pred.shape[0] % seq_len == 0:
            n_clips = pred.shape[0] // seq_len
            p = pred.view(n_clips, seq_len, -1).to(torch.float64)
            g = tgt.view(n_clips, seq_len, -1).to(torch.float64)
            v = valid.view(n_clips, seq_len)
            v3 = (v[:, 2:] & v[:, 1:-1] & v[:, :-2]).to(torch.float64)
            acc_p = p[:, 2:] - 2.0 * p[:, 1:-1] + p[:, :-2]
            acc_g = g[:, 2:] - 2.0 * g[:, 1:-1] + g[:, :-2]
            stats[2] = ((acc_p ** 2).sum(dim=-1) * v3).sum()
            stats[3] = ((acc_g ** 2).sum(dim=-1) * v3).sum()
            stats[4] = v3.sum() * p.shape[-1]
        return stats


def metrics_from_stats(stats: Tensor) -> dict[str, float]:
    """Eval metrics from the (all-reduced) stats vector."""
    n_chan = max(float(stats[1]), 1.0)
    n_acc = max(float(stats[4]), 1.0)
    acc_pred = float((stats[2] / n_acc) ** 0.5)
    acc_gt = float((stats[3] / n_acc) ** 0.5)
    return {
        "mae": float(stats[0] / n_chan),
        "acc_rms_pred": acc_pred,
        "acc_rms_gt": acc_gt,
        "acc_ratio": acc_pred / max(acc_gt, 1e-12),
        "n_rows": float(stats[1] / N_POSE_CHANNELS),
    }


__all__ = ["PoseSupervisedLoss", "metrics_from_stats", "STAT_NAMES",
           "POSE_SLOTS", "N_POSE_CHANNELS"]
