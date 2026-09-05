"""Supervision of the refiner's motion output against finite-differenced kindyn GT.

Reads ``out["motion"]``: ``vel`` / ``acc`` ``(B, 22, 3)`` — world velocity and
acceleration of the 22 SMPL-X body joints — and ``ang_vel`` / ``ang_acc``
``(B, 3)`` of the root, every vector expressed in the predicted body frame
``frame`` ``(B, 3, 3)`` (world-from-body of the depth-smoothed per-frame root).

Targets are built per clip from the kindyn world joints and root rotation
(``smplx_joints_world`` / ``smplx_root_rot``): central finite differences at
the clip's real frame spacing, Gaussian label smoothing of ``label_smooth_sec``
on the velocity (the acceleration is the smoothed derivative of the smoothed
velocity — the 2026-08 motion round showed raw kindyn derivatives are too noisy
to learn from), then rotated into the predicted body frame so prediction and
target share one frame. The smoothing weights by the derivative's own support
(both neighbours valid), and rows within ``ceil(2 sigma / dt)`` frames of a run
end or a hole are not supervised — their kernel is truncated.

Every quantity is divided by its ``scale`` (the GT RMS, config) before a Huber
of width ``huber_delta`` in those standardized units, so the four terms start on
an equal footing. Metrics: RMSE in physical units and the pooled Pearson
correlation over all components, per quantity.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from model.loss import Loss, LossResult
from model.refiner import (NUM_BODY_JOINTS, angular_velocity, gaussian_smooth,
                           stencil_valid, time_derivative)
from utils.metrics import pearson_from_stats

QUANTITIES = ("vel", "acc", "ang_vel", "ang_acc")
_STATS = ("se", "sum_p", "sum_g", "sum_pg", "sum_pp", "sum_gg", "n")


class MotionLoss(Loss):
    """Standardized Huber on the four motion quantities of the refiner."""

    name = "motion"
    stat_names = tuple(f"{q}/{s}" for q in QUANTITIES for s in _STATS)

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        super().__init__(cfg, model, device)
        section = cfg["motion_supervision"]
        self.sigma = float(section["label_smooth_sec"])
        self.scale = {q: float(section["scale"][q]) for q in QUANTITIES}
        self.weights = {q: float(section["loss"][q]) for q in QUANTITIES}
        self.delta = float(section["loss"]["huber_delta"])
        self.term_names = tuple(q for q in QUANTITIES if self.weights[q] > 0.0)

    def targets(self, batch: dict, frame: Tensor) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """GT motion in the predicted body frame and the per-row validity masks.

        :param frame: ``(B, 3, 3)`` world-from-body of the prediction.
        :returns: ``({quantity: target}, {quantity: mask (B,) bool})``.
        """
        seq_len = int(batch["seq_len"])
        n_frames = frame.shape[0]
        n_clips = n_frames // seq_len
        seconds = batch["frame_pos_sec"].to(self.device, self.dtype).view(n_clips, seq_len)
        valid = (batch["smplx_valid"] & batch["frame_valid"]).to(self.device).view(n_clips, seq_len)
        joints = batch["smplx_joints_world"][:, :NUM_BODY_JOINTS].to(self.device, self.dtype)
        joints = joints.view(n_clips, seq_len, NUM_BODY_JOINTS, 3)
        root = batch["smplx_root_rot"].to(self.device, self.dtype).view(n_clips, seq_len, 3, 3)

        # A derivative exists only where both neighbours are valid; the smoothing must weight
        # by THAT support (a forced-zero derivative next to a hole would otherwise leak into
        # its valid neighbours). Rows within ~2 sigma of a run end or a hole see a truncated
        # kernel, so the loss masks them too: radius = ceil(2 sigma / dt).
        first = stencil_valid(valid, 1)
        second = stencil_valid(valid, 2)
        vel_w = gaussian_smooth(time_derivative(joints, seconds, valid), seconds, first, self.sigma)
        acc_w = gaussian_smooth(time_derivative(vel_w, seconds, first), seconds, second, self.sigma)
        ang_body = angular_velocity(root, seconds, valid)                    # GT body frame
        ang_w = gaussian_smooth((root @ ang_body[..., None])[..., 0], seconds, first, self.sigma)
        ang_acc_w = gaussian_smooth(time_derivative(ang_w, seconds, first), seconds, second, self.sigma)

        to_body = frame.transpose(1, 2)                                      # body-from-world
        targets = {
            "vel": torch.einsum("bij,bkj->bki", to_body, vel_w.reshape(n_frames, NUM_BODY_JOINTS, 3)),
            "acc": torch.einsum("bij,bkj->bki", to_body, acc_w.reshape(n_frames, NUM_BODY_JOINTS, 3)),
            "ang_vel": (to_body @ ang_w.reshape(n_frames, 3, 1))[..., 0],
            "ang_acc": (to_body @ ang_acc_w.reshape(n_frames, 3, 1))[..., 0],
        }
        steps = seconds[:, 1:] - seconds[:, :-1]
        dt = float(steps[steps > 0].median()) if bool((steps > 0).any()) else 0.0
        edge = int(math.ceil(2.0 * self.sigma / dt)) if dt > 0 else 0
        rows_vel = stencil_valid(valid, max(1, edge)).reshape(n_frames)
        rows_acc = stencil_valid(valid, max(2, edge)).reshape(n_frames)
        masks = {"vel": rows_vel, "acc": rows_acc, "ang_vel": rows_vel, "ang_acc": rows_acc}
        return targets, masks

    def __call__(self, out: dict, batch: dict, *, train: bool) -> LossResult:
        motion = out["motion"]
        pred = {q: motion[q].to(self.device, self.dtype) for q in QUANTITIES}
        anchor = sum(p.sum() for p in pred.values()) * 0.0
        # The frame is a fixed input of the loss: a trainable pose path must never lower this
        # term by rotating its root instead of fixing the motion.
        targets, masks = self.targets(batch, motion["frame"].detach().to(self.device, self.dtype))

        raw: dict[str, tuple[Tensor, float]] = {}
        stats = []
        for q in QUANTITIES:
            p, g = pred[q].reshape(pred[q].shape[0], -1), targets[q].reshape(pred[q].shape[0], -1)
            mask = masks[q].to(self.dtype)
            if self.weights[q] > 0.0:
                huber = F.smooth_l1_loss(p / self.scale[q], g / self.scale[q],
                                         reduction="none", beta=self.delta).mean(dim=-1)
                raw[q] = (self.weights[q] * (huber * mask).sum(), float(mask.sum()))
            with torch.no_grad():
                pm, gm = p.detach() * mask[:, None], g * mask[:, None]
                stats += [float(((pm - gm) ** 2).sum()), float(pm.sum()), float(gm.sum()),
                          float((pm * gm).sum()), float((pm * pm).sum()), float((gm * gm).sum()),
                          float(mask.sum() * p.shape[1])]
        return LossResult(
            terms=self._terms(raw, anchor),
            scalars={"n_rows": float(masks["vel"].sum())},
            stats=torch.tensor(stats, dtype=torch.float64, device=self.device))

    def metrics(self, stats: Tensor) -> dict[str, float]:
        out = {}
        for i, q in enumerate(QUANTITIES):
            se, sp, sg, spg, spp, sgg, n = (float(v) for v in stats[7 * i:7 * i + 7])
            out[f"{q}_rmse"] = math.sqrt(se / n) if n > 0 else float("nan")
            out[f"{q}_pearson"] = pearson_from_stats(sp, sg, spg, spp, sgg, n)
        return out


__all__ = ["MotionLoss", "QUANTITIES"]
