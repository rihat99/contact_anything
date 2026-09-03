"""Pelvis root-twist loss against the kindyn ground truth.

Reads ``out["motion"]["joint_motion"] (B, 1, 3 * len(terms))`` — the
standardized pelvis twist, three channels per term of ``model.motion.terms``
in the order ``vel, acc, ang_vel, ang_acc``: linear velocity / acceleration
in the ``motion_supervision.linear_frame`` (``gravity_view`` or ``body``),
then the body angular velocity / acceleration (the SE3-log body rate, the
same in either linear frame).

The gravity-view frame (GVHMR's) has its vertical along the scene's FITTED
gravity — read per scene from ``kindyn_1.npz``, not a world axis; the corpus is
genuinely tilted (median 3.2 deg, max 61 deg) — and its azimuth along the
per-frame camera view direction. Body roll and pitch therefore no longer rotate
the linear target, and the vertical is its own channel. The body frame is BVR's
root body twist (``v = R^T p_dot``), the one a roll-out can integrate.

GT arrives from the loader in **physical** units (m/s, m/s^2, rad/s, rad/s^2)
as the full 12-channel twist and is sliced to the head's terms and standardized
here with the config's pinned ``motion_supervision.standardize`` table
(measured per linear frame), so the objective is reproducible from a
checkpoint's stored config alone (a registered buffer would not be serialised).

One Huber term per head term with a non-zero weight, sharing one mask: an
entry contributes when the frame is motion-valid (the central-difference
stencil has support, outside the scene-edge and gap trims) and frame-valid.
During TRAINING the per-frame outlier bit additionally drops the row — the
same kindyn position spike contaminates the velocity and the acceleration —
while evaluation never filters, so the reported numbers are protocol-stable
across runs.

Diagnostics are de-standardized and reported in two ways per quantity: the
Pearson r pooled over the 3 target-axis components (what the head actually
regresses), and the **world-vertical** one — the world vectors projected on the
scene's fitted gravity, positive downward. Both sides go through the identical
conversion, so the comparison stays like-for-like.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from model.loss import Loss, LossResult
from utils.metrics import pearson_from_stats, rmse_from_stats

#: The four possible triples, in the loader's ``motion_gt`` channel order.
TERM_NAMES = ("vel", "acc", "ang_vel", "ang_acc")
#: Columns of the per-quantity statistics block. The ``*_vert`` group feeds the
#: world-vertical Pearson r, the ``*_3d`` group the pooled 3-component one
#: (whose sample count is ``3 * n``).
STAT_COLUMNS = ("n", "pred_vert", "gt_vert", "pred_vert_sq", "gt_vert_sq",
                "pred_gt_vert", "sq_err_3d", "pred_3d", "gt_3d", "pred_3d_sq",
                "gt_3d_sq", "pred_gt_3d")


def standardize_table(
    cfg: dict, terms: Sequence[str], device, dtype=torch.float32,
) -> tuple[Tensor, Tensor]:
    """``(mean, std)`` ``[1, K, 3 * len(terms)]`` of the pinned per-frame table.

    ``motion_supervision.standardize`` is ``[K][4][3]`` over :data:`TERM_NAMES`;
    the head's ``terms`` select and order the triples.
    """
    table = cfg["motion_supervision"]["standardize"]
    if table["mean"] is None or table["std"] is None:
        raise ValueError(
            "motion_supervision.standardize.mean/std must be measured [K][4][3] "
            "tables (null = not measured for this linear_frame)")
    mean = torch.tensor(table["mean"], dtype=dtype)
    std = torch.tensor(table["std"], dtype=dtype)
    if mean.ndim != 3 or mean.shape[1:] != (len(TERM_NAMES), 3) or std.shape != mean.shape:
        raise ValueError(
            f"motion_supervision.standardize must be [K][4][3]; got mean "
            f"{tuple(mean.shape)}, std {tuple(std.shape)}")
    index = [TERM_NAMES.index(t) for t in terms]
    return (mean[:, index].reshape(1, mean.shape[0], -1).to(device),
            std[:, index].reshape(1, std.shape[0], -1).to(device))


def select_terms(gt: Tensor, terms: Sequence[str]) -> Tensor:
    """Slice the loader's ``(B, K, 12)`` twist to ``(B, K, 3 * len(terms))``."""
    return torch.cat([gt[..., 3 * TERM_NAMES.index(t):3 * TERM_NAMES.index(t) + 3]
                      for t in terms], dim=-1)


class MotionLoss(Loss):
    """Standardized Huber on the pelvis twist terms the head emits."""

    name = "motion"

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        super().__init__(cfg, model, device)
        ms = cfg["motion_supervision"]
        loss_cfg = ms["loss"]
        self.terms = list(self.model.motion_terms)
        self.weights = {name: float(loss_cfg[name]) for name in TERM_NAMES}
        missing = [t for t in TERM_NAMES if self.weights[t] != 0.0 and t not in self.terms]
        if missing:
            raise ValueError(
                f"motion_supervision.loss weights {missing} are non-zero but the head "
                f"emits only {self.terms}")
        self.term_names = tuple(n for n in self.terms if self.weights[n] != 0.0)
        if not self.term_names:
            raise ValueError(
                "motion_supervision: every loss weight is 0 — disable the section instead")
        self.stat_names = tuple(f"{term}/{column}"
                                for term in self.terms for column in STAT_COLUMNS)
        self.huber_delta = float(loss_cfg["huber_delta"])
        self.mean, self.std = standardize_table(cfg, self.terms, self.device, self.dtype)

    def __call__(self, out: dict, batch: dict, *, train: bool) -> LossResult:
        pred = out["motion"]["joint_motion"].to(self.device, self.dtype)  # (B,K,3n)
        anchor = pred.sum() * 0.0
        gt = select_terms(batch["motion_gt"].to(self.device, self.dtype), self.terms)
        if pred.shape != gt.shape:
            raise ValueError(
                f"motion prediction {tuple(pred.shape)} does not match the GT "
                f"{tuple(gt.shape)} — model.motion.keypoint_indices / terms and the "
                f"dataset's motion slots must agree")
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
            for i, name in enumerate(self.terms) if self.weights[name] != 0.0
        }

        stats = self._statistics(pred.detach(), gt, mask, batch)
        scalars = {"n_outlier": float(n_outlier), "n_rows": float(valid.sum())}
        for i, name in enumerate(self.terms):
            block = stats[i * len(STAT_COLUMNS):(i + 1) * len(STAT_COLUMNS)]
            scalars[f"{name}_rmse"] = float(
                rmse_from_stats(block[6], block[0]))
        return LossResult(terms=self._terms(raw, anchor), scalars=scalars,
                          stats=stats)

    @torch.no_grad()
    def _statistics(
        self, pred: Tensor, gt: Tensor, mask: Tensor, batch: dict,
    ) -> Tensor:
        """De-standardized Pearson / RMSE sufficient statistics. ``[terms * 12]`` float64.

        The vertical component is the world vector projected on the scene's
        fitted ``gravity_world`` (down-positive) — a fixed axis index would be
        wrong for the hundreds of scenes whose gravity is tilted. The linear
        pair rotates by the linear frame's own world rotation, the angular
        pair by the body rotation.
        """
        physical = pred * self.std + self.mean                          # (B,K,3n)
        weight = mask.to(torch.float64)                                 # (B,K)
        lin_rot = batch["motion_lin_rot"].to(self.device, self.dtype)   # (B,3,3)
        rot = batch["motion_rot"].to(self.device, self.dtype)
        gravity = batch["gravity_world"].to(self.device, self.dtype)    # (B,3)

        stats = torch.zeros(len(self.terms), len(STAT_COLUMNS),
                            dtype=torch.float64, device=self.device)
        for i, name in enumerate(self.terms):
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
        blocks = stats.reshape(len(self.terms), len(STAT_COLUMNS))
        out: dict[str, float] = {}
        for i, name in enumerate(self.terms):
            n, pv, gv, pv2, gv2, pgv, sq3, p3, g3, p32, g32, pg3 = blocks[i]
            out[f"{name}_vert_r"] = float(
                pearson_from_stats(n, pv, gv, pv2, gv2, pgv))
            out[f"{name}_r3d"] = float(
                pearson_from_stats(3.0 * n, p3, g3, p32, g32, pg3))
            out[f"{name}_rmse"] = float(rmse_from_stats(sq3, n))
            out[f"{name}_gt_rms"] = float(rmse_from_stats(g32, n))   # zero-predictor RMSE
        out["n_rows"] = float(blocks[0, 0])
        return out


__all__ = ["MotionLoss", "TERM_NAMES", "STAT_COLUMNS", "standardize_table", "select_terms"]
