"""Jerk prior on the SMPL-X head's predicted trajectory — toward ZERO, no target.

Two smoothness terms on the per-frame SMPL-X prediction, differentiated over
the clip with the SAME 5-point stencils the corpus jerk statistics were
measured with (scratchpad ``massive/jerk_stats.md``, 2026-09-03):

* ``root_lin`` / ``root_ang`` — jerk of the ROOT BODY TWIST: the predicted
  camera-frame root (``pelvis_cam``, ``root_rot``) lifted to the world with the
  frame extrinsics (so camera egomotion is removed), successive relative poses
  ``d[t] = se3_log(T_t^{-1} T_{t+1})``, then ``j[t] = (d[t+1] - d[t] - d[t-1]
  + d[t-2]) / (2 dt^3)`` (m/s^3, rad/s^3).
* ``joints`` — jerk of the ROOT-LOCAL joints ``R_root^T (p_j - p_root)`` (the 21
  articulated body joints; no extrinsics needed, the camera cancels):
  ``j[t] = (-p[t-2] + 2 p[t-1] - 2 p[t+1] + p[t+2]) / (2 dt^3)``.

Each is a Huber toward zero in UNITS OF THE GT p75 (``huber_delta_*``: the
jerk the real motion reaches a quarter of the time at the clip sampler's dt),
so the loss is quadratic on the jerk the GT itself has and linear beyond, and
the two weights ``loss.root`` (shared by the linear + angular root terms) and
``loss.joints`` are unit-free. A stencil row needs frames ``t-2..t+2`` all
frame-valid; the mass is the number of such rows. The gradient reaches the
pose path only (through ``joints_cam`` / ``root_rot``).

Metrics (eval): the RMS jerk of the prediction and of the kindyn GT under the
identical stencils, physical units, so the prior's effect is read directly
against the real motion's jerk. The 2026-08-29 jerk+snap on the MHR path was a
wash on the positional terms and over-smoothed the angular ones — expect a
small effect and watch the GT-relative RMS.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from model.loss import Loss, LossResult
from utils.geometry import lift_to_world, se3_log
from utils.metrics import rmse_from_stats

TERMS = ("root_lin", "root_ang", "joints")
NUM_BODY_JOINTS = 22
#: Frames a jerk stencil row needs (``t-2 .. t+2``).
STENCIL = 5


def stencil_support(valid: Tensor) -> Tensor:
    """Rows whose 5-frame stencil lies inside the clip on valid frames. ``(n, T) -> (n, T-4)``."""
    return (valid[:, :-4] & valid[:, 1:-3] & valid[:, 2:-2] & valid[:, 3:-1] & valid[:, 4:])


def root_twist_jerk(rot_w: Tensor, pos_w: Tensor, dt: Tensor) -> Tensor:
    """Body-twist jerk of a root trajectory. ``(n, T, 3, 3), (n, T, 3), (n,) -> (n, T-4, 6)``."""
    rel_rot = rot_w[:, :-1].transpose(-1, -2) @ rot_w[:, 1:]
    rel_trans = torch.einsum("ntji,ntj->nti", rot_w[:, :-1], pos_w[:, 1:] - pos_w[:, :-1])
    d = se3_log(rel_rot, rel_trans)                                       # (n, T-1, 6)
    jerk = 0.5 * (d[:, 3:] - d[:, 2:-1] - d[:, 1:-2] + d[:, :-3])          # rows t = 2..T-3
    return jerk / (dt ** 3)[:, None, None]


def point_jerk(points: Tensor, dt: Tensor) -> Tensor:
    """Central third difference. ``(n, T, K, 3), (n,) -> (n, T-4, K, 3)``."""
    jerk = (-points[:, :-4] + 2.0 * points[:, 1:-3] - 2.0 * points[:, 3:-1] + points[:, 4:])
    return jerk / (2.0 * dt ** 3)[:, None, None, None]


class SmoothnessLoss(Loss):
    """Huber-toward-zero jerk of the predicted root twist and root-local joints."""

    name = "smoothness"
    metric_group = "smooth"
    stat_names = tuple(f"{who}_{term}_{part}" for who in ("pred", "gt")
                       for term in TERMS for part in ("sq_sum", "count"))

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        super().__init__(cfg, model, device)
        if self.model.head_smplx is None:
            raise ValueError("pose_smoothness needs model.smplx.enabled (the SMPL-X head)")
        loss_cfg = cfg["pose_smoothness"]["loss"]
        self.weights = {"root_lin": float(loss_cfg["root"]), "root_ang": float(loss_cfg["root"]),
                        "joints": float(loss_cfg["joints"])}
        self.deltas = {"root_lin": float(loss_cfg["huber_delta_root_lin"]),
                       "root_ang": float(loss_cfg["huber_delta_root_ang"]),
                       "joints": float(loss_cfg["huber_delta_joints"])}
        if any(d <= 0 for d in self.deltas.values()):
            raise ValueError("pose_smoothness.loss.huber_delta_* must be positive")
        self.term_names = tuple(t for t in TERMS if self.weights[t] > 0.0)
        if not self.term_names:
            raise ValueError(
                "pose_smoothness: both weights are 0 — disable the section instead")

    def _jerks(self, joints: Tensor, root_rot: Tensor, ext: Tensor, seq_len: int,
               dt: Tensor) -> dict[str, Tensor]:
        """Per-row jerks of one (prediction or GT) trajectory, ``{term: (n, T-4, ...)}``."""
        n = joints.shape[0] // seq_len
        pelvis = joints[:, 0]
        rot_w = ext[:, :3, :3].transpose(1, 2) @ root_rot                    # world-from-root
        pos_w = lift_to_world(pelvis[:, None], ext)[:, 0]
        local = torch.einsum("bji,bkj->bki", root_rot, joints[:, 1:] - pelvis[:, None])
        root = root_twist_jerk(rot_w.reshape(n, seq_len, 3, 3),
                               pos_w.reshape(n, seq_len, 3), dt)
        return {
            "root_lin": root[..., :3],
            "root_ang": root[..., 3:],
            "joints": point_jerk(local.reshape(n, seq_len, *local.shape[1:]), dt),
        }

    def __call__(self, out: dict, batch: dict, *, train: bool) -> LossResult:
        smplx = out["smplx"]
        joints = smplx["joints_cam"][:, :NUM_BODY_JOINTS].to(self.device, self.dtype)
        root_rot = smplx["root_rot"].to(self.device, self.dtype)              # cam-from-root
        anchor = (joints.sum() + root_rot.sum()) * 0.0
        seq_len = int(batch["seq_len"])
        n = joints.shape[0] // seq_len
        ext = batch["cam_from_world"].to(self.device, self.dtype)
        seconds = batch["frame_pos_sec"].to(self.device, self.dtype).reshape(n, seq_len)
        raw: dict[str, tuple[Tensor, float]] = {}
        stats = self.empty_stats()
        if seq_len < STENCIL:
            return LossResult(terms=self._terms({t: (anchor, 0.0) for t in self.term_names},
                                                anchor), stats=stats)
        dt = (seconds[:, -1] - seconds[:, 0]) / (seq_len - 1)                # uniform within a clip
        support = stencil_support(
            batch["frame_valid"].to(self.device).reshape(n, seq_len))         # (n, T-4)
        mass = float(support.sum())
        weight = support.to(self.dtype)

        pred = self._jerks(joints, root_rot, ext, seq_len, dt)
        for term in self.term_names:
            scaled = pred[term] / self.deltas[term]
            huber = F.smooth_l1_loss(scaled, torch.zeros_like(scaled), reduction="none", beta=1.0)
            per_row = huber.flatten(2).mean(dim=-1)                          # (n, T-4)
            raw[term] = (self.weights[term] * (per_row * weight).sum(), mass)

        with torch.no_grad():
            gt_rot = ext[:, :3, :3] @ batch["smplx_root_rot"].to(self.device, self.dtype)
            gt_joints_cam = torch.einsum(
                "bij,bkj->bki", ext[:, :3, :3],
                batch["smplx_joints_world"][:, :NUM_BODY_JOINTS].to(self.device, self.dtype),
            ) + ext[:, :3, 3][:, None]
            gt_support = support & stencil_support(
                batch["smplx_valid"].to(self.device).reshape(n, seq_len))
            gt = self._jerks(gt_joints_cam, gt_rot, ext, seq_len, dt)
            for i, term in enumerate(TERMS):
                for j, (values, rows) in enumerate(((pred[term], support), (gt[term], gt_support))):
                    sq = values.detach().to(torch.float64).flatten(2).square().mean(dim=-1)
                    base = 2 * (j * len(TERMS) + i)
                    stats[base] = (sq * rows.to(torch.float64)).sum()
                    stats[base + 1] = rows.sum().to(torch.float64)
        return LossResult(terms=self._terms(raw, anchor), scalars={"n_rows": mass}, stats=stats)

    def metrics(self, stats: Tensor) -> dict[str, float]:
        out: dict[str, float] = {}
        for j, who in enumerate(("pred", "gt")):
            for i, term in enumerate(TERMS):
                base = 2 * (j * len(TERMS) + i)
                out[f"{who}_rms_{term}"] = float(rmse_from_stats(stats[base], stats[base + 1]))
        out["n_rows"] = float(stats[1])
        return out


__all__ = ["SmoothnessLoss", "TERMS", "root_twist_jerk", "point_jerk", "stencil_support"]
