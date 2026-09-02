"""Pelvis gravity-view twist loss against the kindyn ground truth.

Reads ``out["motion"]["joint_motion"] (B, 1, 12)`` — the standardized pelvis
twist: linear velocity ``[..., 0:3]`` and acceleration ``[..., 3:6]`` in the
**gravity-view** frame, then the body angular velocity ``[..., 6:9]`` and
acceleration ``[..., 9:12]``.

The gravity-view frame (GVHMR's) has its vertical along the scene's FITTED
gravity — read per scene from ``kindyn_1.npz``, not a world axis; the corpus is
genuinely tilted (median 3.2 deg, max 61 deg) — and its azimuth along the
per-frame camera view direction. Body roll and pitch therefore no longer rotate
the linear target, and the vertical is its own channel. The angular pair is the
SE3-log body rate, which does not depend on the linear frame at all.

GT arrives from the loader in **physical** units (m/s, m/s^2, rad/s, rad/s^2)
and is standardized here with the config's pinned
``motion_supervision.standardize`` table, so the objective is reproducible from
a checkpoint's stored config alone (a registered buffer would not be serialised).

Four Huber terms, one per triple, sharing one mask: an entry contributes when
the frame is motion-valid (the central-difference stencil has support, outside
the scene-edge and gap trims) and frame-valid. During TRAINING the per-frame
outlier bit additionally drops the row — the same kindyn position spike
contaminates the velocity and the acceleration — while evaluation never filters,
so the reported numbers are protocol-stable across runs.

Diagnostics are de-standardized and reported in two ways per quantity: the
Pearson r pooled over the 3 target-axis components (what the head actually
regresses), and the **world-vertical** one — the world vectors projected on the
scene's fitted gravity, positive downward. Both sides go through the identical
conversion, so the comparison stays like-for-like.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from model.loss import Loss, LossResult
from utils.metrics import pearson_from_stats, rmse_from_stats

#: The four regressed triples, in the head's output-channel order.
TERM_NAMES = ("vel", "acc", "ang_vel", "ang_acc")
#: Columns of the per-quantity statistics block. The ``*_vert`` group feeds the
#: world-vertical Pearson r, the ``*_3d`` group the pooled 3-component one
#: (whose sample count is ``3 * n``).
STAT_COLUMNS = ("n", "pred_vert", "gt_vert", "pred_vert_sq", "gt_vert_sq",
                "pred_gt_vert", "sq_err_3d", "pred_3d", "gt_3d", "pred_3d_sq",
                "gt_3d_sq", "pred_gt_3d")


class MotionLoss(Loss):
    """Standardized Huber on the pelvis gravity-view twist."""

    name = "motion"
    stat_names = tuple(f"{term}/{column}"
                       for term in TERM_NAMES for column in STAT_COLUMNS)

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        super().__init__(cfg, model, device)
        ms = cfg["motion_supervision"]
        loss_cfg = ms["loss"]
        self.weights = {name: float(loss_cfg[name]) for name in TERM_NAMES}
        self.term_names = tuple(n for n in TERM_NAMES if self.weights[n] != 0.0)
        if not self.term_names:
            raise ValueError(
                "motion_supervision: every loss weight is 0 — disable the section instead")
        self.huber_delta = float(loss_cfg["huber_delta"])
        # [K][4][3] -> [1, K, 12]: broadcasts over rows and lines up with the
        # head's (vel | acc | ang_vel | ang_acc) output layout.
        mean = torch.tensor(ms["standardize"]["mean"], dtype=self.dtype)
        std = torch.tensor(ms["standardize"]["std"], dtype=self.dtype)
        width = 3 * len(TERM_NAMES)
        self.mean = mean.reshape(1, mean.shape[0], width).to(self.device)
        self.std = std.reshape(1, std.shape[0], width).to(self.device)

    def __call__(self, out: dict, batch: dict, *, train: bool) -> LossResult:
        pred = out["motion"]["joint_motion"].to(self.device, self.dtype)  # (B,K,12)
        anchor = pred.sum() * 0.0
        gt = batch["motion_gt"].to(self.device, self.dtype)
        if pred.shape != gt.shape:
            raise ValueError(
                f"motion prediction {tuple(pred.shape)} does not match the GT "
                f"{tuple(gt.shape)} — model.motion.keypoint_indices and the "
                f"dataset's motion slots must agree")
        if pred.shape[-1] != 3 * len(TERM_NAMES):
            raise ValueError(
                f"motion prediction is {pred.shape[-1]}-wide; the gravity-view "
                f"twist target is {3 * len(TERM_NAMES)}-wide")
        if self.mean.shape[1] != pred.shape[1]:
            raise ValueError(
                f"motion_supervision.standardize has {self.mean.shape[1]} slot "
                f"rows but the model predicts {pred.shape[1]} motion tokens")

        valid = (batch["motion_valid"] & batch["frame_valid"]).to(self.device)
        mask = valid[:, None].expand(-1, pred.shape[1])
        n_outlier = 0
        if train:
            outlier = batch["motion_outlier"].to(self.device)
            n_outlier = int((mask & outlier).sum())
            mask = mask & ~outlier

        huber = F.smooth_l1_loss(
            pred, (gt - self.mean) / self.std, reduction="none", beta=self.huber_delta)
        mass = float(mask.sum())
        raw = {
            name: (self.weights[name]
                   * (huber[..., 3 * i:3 * i + 3].sum(dim=-1) * mask).sum(), mass)
            for i, name in enumerate(TERM_NAMES) if self.weights[name] != 0.0
        }

        stats = self._statistics(pred.detach(), gt, mask, batch)
        scalars = {"n_outlier": float(n_outlier), "n_rows": float(valid.sum())}
        for i, name in enumerate(TERM_NAMES):
            block = stats[i * len(STAT_COLUMNS):(i + 1) * len(STAT_COLUMNS)]
            scalars[f"{name}_rmse"] = float(
                rmse_from_stats(block[6], block[0]))
        return LossResult(terms=self._terms(raw, anchor), scalars=scalars,
                          stats=stats)

    @torch.no_grad()
    def _statistics(
        self, pred: Tensor, gt: Tensor, mask: Tensor, batch: dict,
    ) -> Tensor:
        """De-standardized Pearson / RMSE sufficient statistics. ``[4 * 12]`` float64.

        The vertical component is the world vector projected on the scene's
        fitted ``gravity_world`` (down-positive) — a fixed axis index would be
        wrong for the hundreds of scenes whose gravity is tilted. The linear
        pair rotates by the gravity-view frame's own world rotation, the angular
        pair by the body rotation.
        """
        physical = pred * self.std + self.mean                          # (B,K,12)
        weight = mask.to(torch.float64)                                 # (B,K)
        lin_rot = batch["motion_lin_rot"].to(self.device, self.dtype)   # (B,3,3)
        rot = batch["motion_rot"].to(self.device, self.dtype)
        gravity = batch["gravity_world"].to(self.device, self.dtype)    # (B,3)

        stats = torch.zeros(len(TERM_NAMES), len(STAT_COLUMNS),
                            dtype=torch.float64, device=self.device)
        for i, name in enumerate(TERM_NAMES):
            channels = slice(3 * i, 3 * i + 3)
            p = physical[..., channels].to(torch.float64)               # (B,K,3)
            g = gt[..., channels].to(torch.float64)
            world = lin_rot if name in ("vel", "acc") else rot
            p_world = torch.einsum("bij,bkj->bki", world, physical[..., channels])
            g_world = torch.einsum("bij,bkj->bki", world, gt[..., channels])
            p_vert = (p_world * gravity[:, None, :]).sum(-1).to(torch.float64)
            g_vert = (g_world * gravity[:, None, :]).sum(-1).to(torch.float64)
            weight3 = weight[:, :, None]
            stats[i] = torch.stack([
                weight.sum(),
                (p_vert * weight).sum(), (g_vert * weight).sum(),
                (p_vert * p_vert * weight).sum(), (g_vert * g_vert * weight).sum(),
                (p_vert * g_vert * weight).sum(),
                (((p - g) ** 2).sum(dim=-1) * weight).sum(),
                (p * weight3).sum(), (g * weight3).sum(),
                (p * p * weight3).sum(), (g * g * weight3).sum(),
                (p * g * weight3).sum(),
            ])
        return stats.reshape(-1)

    def metrics(self, stats: Tensor) -> dict[str, float]:
        blocks = stats.reshape(len(TERM_NAMES), len(STAT_COLUMNS))
        out: dict[str, float] = {}
        for i, name in enumerate(TERM_NAMES):
            n, pv, gv, pv2, gv2, pgv, sq3, p3, g3, p32, g32, pg3 = blocks[i]
            out[f"{name}_vert_r"] = float(
                pearson_from_stats(n, pv, gv, pv2, gv2, pgv))
            out[f"{name}_r3d"] = float(
                pearson_from_stats(3.0 * n, p3, g3, p32, g32, pg3))
            out[f"{name}_rmse"] = float(rmse_from_stats(sq3, n))
            out[f"{name}_gt_rms"] = float(rmse_from_stats(g32, n))   # zero-predictor RMSE
        out["n_rows"] = float(blocks[0, 0])
        return out


__all__ = ["MotionLoss", "TERM_NAMES", "STAT_COLUMNS"]
