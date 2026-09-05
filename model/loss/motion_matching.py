"""Motion matching: the PREDICTED SMPL-X trajectory differentiated and matched to the GT motion.

The per-frame prediction (``pelvis_cam``, ``root_rot``, ``joints_cam``) is lifted
to the world with the GT extrinsics, and the world root trajectory is
differentiated with the SAME BVR body-twist scheme the kindyn motion targets
use (:func:`data.climbing_videos.kindyn.root_body_twist`)::

    d[t] = se3_log(T_t^-1 T_t+1)
    v[t] = (d[t-1] + d[t]) / (2 dt)          a[t] = (d[t] - d[t-1]) / dt^2

so the pose path itself — not an auxiliary head — is asked to be consistent in
time. Terms (``motion_matching.loss``, each a Huber, inert at weight 0):

- ``root_vel`` / ``root_ang_vel`` / ``root_acc`` / ``root_ang_acc`` — the
  pose-derived body twist vs the kindyn GT twist (``motion_gt``, the SMPL-X
  root differentiated after ``motion_supervision.target_smooth_sec`` smoothing;
  0 = the raw kindyn fit, itself smooth), standardized with the pinned
  ``motion_supervision.standardize`` table like the motion head's own loss.
  The target is smooth and the raw finite difference of a jittery trajectory
  is not, so the term is at once a motion and a smoothness signal.
- ``head_vel`` / ``head_ang_vel`` — the pose-derived twist vs the motion head's
  DETACHED prediction (the old pose->motion consistency): the pose path is
  pulled toward the head's estimate, never the other way round.
- ``joint_vel`` / ``joint_acc`` — root-local joint velocities / accelerations
  (``R_cb^T (joint - pelvis)`` central differences; the camera cancels exactly)
  vs the GT SMPL-X joints differentiated the same way, physical m/s, m/s^2.

A twist row needs frames ``t-1, t, t+1`` valid, so clip boundaries are never
supervised and ``data.clip.frames >= 3`` is required. The root terms compare in
the BODY frame of each side (predicted / GT root rotation): the velocity a
roll-out integrates, the frame the head predicts in. Gradients reach the pose
path only (the motion head is detached). The eval metrics report how the
lifted trajectory differentiates — Pearson r (3 axes pooled) and RMSE of the
pose-derived root twist vs GT, and the joint-velocity RMSE — protocol-stable
(no train-only outlier filtering).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from model.loss import Loss, LossResult
from model.loss.motion import TERM_NAMES, standardize_table
from utils.geometry import lift_to_world, se3_log
from utils.metrics import pearson_from_stats, rmse_from_stats

NUM_BODY_JOINTS = 22
ROOT_TERMS = ("root_vel", "root_acc", "root_ang_vel", "root_ang_acc")   # motion_gt channel order
HEAD_TERMS = ("head_vel", "head_ang_vel")
JOINT_TERMS = ("joint_vel", "joint_acc")
TERMS = ROOT_TERMS + HEAD_TERMS + JOINT_TERMS
#: Per-quantity statistics: pooled over the 3 axes (r3d / RMSE / GT RMS).
STAT_COLUMNS = ("n", "sq_err", "pred", "gt", "pred_sq", "gt_sq", "pred_gt")
STAT_QUANTITIES = ("root_vel", "root_acc", "root_ang_vel", "root_ang_acc", "joint_vel", "joint_acc")


def clip_body_twist(rot_w: Tensor, pos_w: Tensor, dt: Tensor) -> tuple[Tensor, Tensor]:
    """BVR body twist velocity / acceleration of a root trajectory.

    ``(n, T, 3, 3), (n, T, 3), (n,) -> (vel (n, T-2, 6), acc (n, T-2, 6))``
    for rows ``t = 1 .. T-2``; layout ``[linear, angular]``.
    """
    rel_rot = rot_w[:, :-1].transpose(-1, -2) @ rot_w[:, 1:]
    rel_trans = torch.einsum("ntji,ntj->nti", rot_w[:, :-1], pos_w[:, 1:] - pos_w[:, :-1])
    d = se3_log(rel_rot, rel_trans)                                       # (n, T-1, 6)
    vel = 0.5 * (d[:, :-1] + d[:, 1:]) / dt[:, None, None]
    acc = (d[:, 1:] - d[:, :-1]) / (dt * dt)[:, None, None]
    return vel, acc


def central_differences(points: Tensor, dt: Tensor) -> tuple[Tensor, Tensor]:
    """Velocity / acceleration of a point trajectory. ``(n, T, ...) -> 2 x (n, T-2, ...)``."""
    shape = (-1,) + (1,) * (points.ndim - 1)
    vel = (points[:, 2:] - points[:, :-2]) / (2.0 * dt).reshape(shape)
    acc = (points[:, 2:] - 2.0 * points[:, 1:-1] + points[:, :-2]) / (dt * dt).reshape(shape)
    return vel, acc


def interior_support(valid: Tensor) -> Tensor:
    """Rows whose ``t-1, t, t+1`` are all valid. ``(n, T) -> (n, T-2)``."""
    return valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]


class MotionMatchingLoss(Loss):
    """Pose-derived root twist / joint motion vs the kindyn GT and the motion head."""

    name = "motion_matching"
    metric_group = "matching"
    stat_names = tuple(f"{q}/{c}" for q in STAT_QUANTITIES for c in STAT_COLUMNS)

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        super().__init__(cfg, model, device)
        loss_cfg = cfg["motion_matching"]["loss"]
        self.weights = {name: float(loss_cfg[name]) for name in TERMS}
        self.term_names = tuple(n for n in TERMS if self.weights[n] > 0.0)
        if not self.term_names:
            raise ValueError(
                "motion_matching: every loss weight is 0 — disable the section instead")
        if any(self.weights[n] > 0.0 for n in HEAD_TERMS):
            head_terms = list(getattr(self.model, "motion_terms", []))
            if self.model.head_motion is None or not all(
                    t in head_terms for t in ("vel", "ang_vel")):
                raise ValueError(
                    "motion_matching.loss.head_* needs model.motion with terms vel + ang_vel")
        self.huber_delta = float(loss_cfg["huber_delta"])
        self.delta_joint_vel = float(loss_cfg["huber_delta_joint_vel"])
        self.delta_joint_acc = float(loss_cfg["huber_delta_joint_acc"])
        # [1, 1, 12] over (vel, acc, ang_vel, ang_acc): the root terms' units.
        self.mean, self.std = standardize_table(cfg, TERM_NAMES, self.device, self.dtype)
        self.mean, self.std = self.mean[0, 0], self.std[0, 0]
        self.head_mean, self.head_std = None, None
        if self.model.head_motion is not None:
            self.head_mean, self.head_std = standardize_table(
                cfg, self.model.motion_terms, self.device, self.dtype)

    # ------------------------------------------------------------------ trajectories

    @staticmethod
    def _lift(joints: Tensor, root_rot: Tensor, ext: Tensor, n: int, seq_len: int
              ) -> tuple[Tensor, Tensor, Tensor]:
        """World root rotation / position and root-local joints of a camera-frame body."""
        pelvis = joints[:, 0]
        rot_w = ext[:, :3, :3].transpose(1, 2) @ root_rot                    # world-from-root
        pos_w = lift_to_world(pelvis[:, None], ext)[:, 0]
        local = torch.einsum("bji,bkj->bki", root_rot, joints[:, 1:] - pelvis[:, None])
        return (rot_w.reshape(n, seq_len, 3, 3), pos_w.reshape(n, seq_len, 3),
                local.reshape(n, seq_len, -1, 3))

    def __call__(self, out: dict, batch: dict, *, train: bool) -> LossResult:
        smplx = out["smplx"]
        joints = smplx["joints_cam"][:, :NUM_BODY_JOINTS].to(self.device, self.dtype)
        root_rot = smplx["root_rot"].to(self.device, self.dtype)              # cam-from-root
        anchor = (joints.sum() + root_rot.sum()) * 0.0
        seq_len = int(batch["seq_len"])
        n = joints.shape[0] // seq_len
        stats = self.empty_stats()
        if seq_len < 3:
            return LossResult(terms=self._terms({t: (anchor, 0.0) for t in self.term_names},
                                                anchor), stats=stats)
        ext = batch["cam_from_world"].to(self.device, self.dtype)
        seconds = batch["frame_pos_sec"].to(self.device, self.dtype).reshape(n, seq_len)
        dt = (seconds[:, -1] - seconds[:, 0]) / (seq_len - 1)                # uniform within a clip
        frame_valid = batch["frame_valid"].to(self.device).reshape(n, seq_len)
        frame_rows = interior_support(frame_valid)                          # (n, T-2)

        # --- predicted trajectory -> body twist + root-local joint motion ---
        rot_w, pos_w, local = self._lift(joints, root_rot, ext, n, seq_len)
        vel, acc = clip_body_twist(rot_w, pos_w, dt)                          # (n, T-2, 6)
        twist = torch.cat([vel[..., :3], acc[..., :3], vel[..., 3:], acc[..., 3:]], -1)
        twist_std = (twist - self.mean) / self.std                            # (n, T-2, 12)
        joint_vel, joint_acc = central_differences(local, dt)                 # (n, T-2, 21, 3)

        # --- GT ---
        gt = batch["motion_gt"].to(self.device, self.dtype)[:, 0].reshape(n, seq_len, 12)[:, 1:-1]
        gt_std = (gt - self.mean) / self.std
        motion_valid = batch["motion_valid"].to(self.device).reshape(n, seq_len)[:, 1:-1]
        root_rows = frame_rows & motion_valid
        if train:
            outlier = batch["motion_outlier"].to(self.device).reshape(n, seq_len)[:, 1:-1]
            root_rows = root_rows & ~outlier
        gt_rot = ext[:, :3, :3] @ batch["smplx_root_rot"].to(self.device, self.dtype)
        gt_joints_cam = torch.einsum(
            "bij,bkj->bki", ext[:, :3, :3],
            batch["smplx_joints_world"][:, :NUM_BODY_JOINTS].to(self.device, self.dtype),
        ) + ext[:, :3, 3][:, None]
        _, _, gt_local = self._lift(gt_joints_cam, gt_rot, ext, n, seq_len)
        gt_joint_vel, gt_joint_acc = central_differences(gt_local, dt)
        joint_rows = frame_rows & interior_support(
            batch["smplx_valid"].to(self.device).reshape(n, seq_len))

        raw: dict[str, tuple[Tensor, float]] = {}
        root_w = root_rows.to(self.dtype)
        root_mass = float(root_rows.sum())
        for i, name in enumerate(ROOT_TERMS):
            if self.weights[name] <= 0.0:
                continue
            channels = slice(3 * i, 3 * i + 3)
            huber = F.smooth_l1_loss(twist_std[..., channels], gt_std[..., channels],
                                     reduction="none", beta=self.huber_delta)
            raw[name] = (self.weights[name] * (huber.sum(-1) * root_w).sum(), root_mass)
        if any(self.weights[name] > 0.0 for name in HEAD_TERMS):
            motion = out["motion"]
            head = torch.cat([motion["joint_vel"][:, 0], motion["joint_ang_vel"][:, 0]],
                             -1).detach().to(self.device, self.dtype)
            head = (head * self._head_scale("std") + self._head_scale("mean"))
            head = head.reshape(n, seq_len, 6)[:, 1:-1]                       # physical
            head_std = (head - self.mean[[0, 1, 2, 6, 7, 8]]) / self.std[[0, 1, 2, 6, 7, 8]]
            pred_std = twist_std[..., [0, 1, 2, 6, 7, 8]]
            frame_w = frame_rows.to(self.dtype)
            frame_mass = float(frame_rows.sum())
            for name, channels in (("head_vel", slice(0, 3)), ("head_ang_vel", slice(3, 6))):
                if self.weights[name] <= 0.0:
                    continue
                huber = F.smooth_l1_loss(pred_std[..., channels], head_std[..., channels],
                                         reduction="none", beta=self.huber_delta)
                raw[name] = (self.weights[name] * (huber.sum(-1) * frame_w).sum(), frame_mass)
        joint_w = joint_rows.to(self.dtype)
        joint_mass = float(joint_rows.sum())
        for name, pred, target, delta in (
                ("joint_vel", joint_vel, gt_joint_vel, self.delta_joint_vel),
                ("joint_acc", joint_acc, gt_joint_acc, self.delta_joint_acc)):
            if self.weights[name] <= 0.0:
                continue
            huber = F.smooth_l1_loss(pred / delta, target / delta, reduction="none", beta=1.0)
            raw[name] = (self.weights[name] * (huber.flatten(2).mean(-1) * joint_w).sum(),
                         joint_mass)

        with torch.no_grad():
            pairs = [(twist[..., 3 * i:3 * i + 3], gt[..., 3 * i:3 * i + 3], root_rows)
                     for i in range(4)]
            pairs += [(joint_vel.flatten(2), gt_joint_vel.flatten(2), joint_rows),
                      (joint_acc.flatten(2), gt_joint_acc.flatten(2), joint_rows)]
            for q, (p, g, rows) in enumerate(pairs):
                p, g = p.to(torch.float64), g.to(torch.float64)
                w = rows.to(torch.float64)[..., None]
                k = p.shape[-1]
                stats[q * 7:(q + 1) * 7] = torch.stack([
                    w.sum() * k,
                    ((p - g).square() * w).sum(), (p * w).sum(), (g * w).sum(),
                    (p * p * w).sum(), (g * g * w).sum(), (p * g * w).sum()])
        return LossResult(terms=self._terms(raw, anchor),
                          scalars={"n_root_rows": root_mass, "n_joint_rows": joint_mass},
                          stats=stats)

    def _head_scale(self, which: str) -> Tensor:
        """The motion head's (mean | std) rows for ``vel`` then ``ang_vel``. ``[6]``."""
        table = (self.head_mean if which == "mean" else self.head_std)[0, 0]
        terms = list(self.model.motion_terms)
        index = [3 * terms.index("vel") + j for j in range(3)] + [
            3 * terms.index("ang_vel") + j for j in range(3)]
        return table[index]

    def metrics(self, stats: Tensor) -> dict[str, float]:
        out: dict[str, float] = {}
        for q, name in enumerate(STAT_QUANTITIES):
            n, sq, p, g, p2, g2, pg = stats[q * 7:(q + 1) * 7]
            out[f"{name}_r"] = float(pearson_from_stats(n, p, g, p2, g2, pg))
            out[f"{name}_rmse"] = float(rmse_from_stats(sq, n / 3.0))     # per-row 3-vector RMS
            out[f"{name}_gt_rms"] = float(rmse_from_stats(g2, n / 3.0))
        out["n_rows"] = float(stats[0] / 3.0)
        return out


__all__ = ["MotionMatchingLoss", "TERMS", "clip_body_twist", "central_differences",
           "interior_support"]
