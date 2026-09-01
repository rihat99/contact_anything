"""Jerk + snap smoothness penalty on the PREDICTED pose trajectory.

Unlike every other temporal term in this repo, nothing here is matched to a
target: the 3rd and 4th time derivatives of the model's own motion are pushed
toward **zero**. The premise is that real climbing motion has velocity and
acceleration but almost no jerk/snap at the scale a per-frame reconstruction
produces them — what the stencils pick up is frame-to-frame reconstruction
noise, so minimising it is a prior on the temporal block rather than a fit to
kindyn. Acceleration itself is deliberately left alone (``kp_vel``/``kp_acc``
in :mod:`contact.keypoint_supervision` are the terms that match motion to GT).

Three channels, each with its own weight and Huber delta:

* ``joint_jerk`` / ``joint_snap`` — the 70 MHR70 keypoints lifted into the
  metric world with the dataset extrinsics, exactly as ``kp_vel``/``kp_acc``
  lift them (``p_w = R_ext^T (kp3d + pred_cam_t - t_ext)``). Differencing in
  the world removes the camera egomotion; a camera-frame difference would
  charge the model for handheld shake. Joints carry
  :func:`~contact.keypoint_supervision.joint_weight_vector` weights (fingers
  down) and every term is a weighted MEAN over them.
* ``root_pos_jerk`` / ``root_pos_snap`` — the same stencils on the predicted
  world pelvis position (:func:`~contact.motion_consistency.predicted_root_world`),
  i.e. the camera-relative placement whose depth wobble is the classic failure
  of a per-frame reconstruction.
* ``root_rot_jerk`` / ``root_rot_snap`` — the BODY angular derivatives of the
  predicted world-from-root rotation, built the BVR way
  (:func:`~contact.motion_consistency.clip_body_twist`): the staggered
  half-step ``d[k] = so3_log(R_k^T R_{k+1})`` is differenced successively, so
  what is penalised is angular jerk/snap of the body twist — never a naive
  difference of Euler triples.

Stencils (real elapsed seconds; ``dt`` is the clip's mean frame interval)::

    jerk_t = (-p[t-2] + 2 p[t-1] - 2 p[t+1] + p[t+2]) / (2 dt^3)
    snap_t = ( p[t-2] - 4 p[t-1] + 6 p[t] - 4 p[t+1] + p[t+2]) / dt^4
    ang_jerk_t = 0.5 (d[t+1] - d[t] - d[t-1] + d[t-2]) / dt^3
    ang_snap_t =     (d[t+1] - 3 d[t] + 3 d[t-1] - d[t-2]) / dt^4

All four are exact on polynomials of their own order (a cubic trajectory has
constant jerk and zero snap) and all four read the same five frames
``t-2 .. t+2``, so every channel shares one support mask: five consecutive
rows of the SAME clip, each ``frame_valid & cam_valid`` (the world lift needs
the extrinsics; no GT is involved, so ``kp_valid`` is irrelevant here).
``frames_per_clip >= 5`` is required — shorter clips leave every term at zero
mass.

Penalty form: ``smooth_l1_loss(x, 0, beta=delta)`` per element. Huber toward
zero, not L2: the derivative distributions are heavy-tailed (the GT's own
|coord| p99/p75 ratio is ~10x, and 60-fps scenes sit ~6x above 24-fps ones at
identical physical smoothness because the ``1/dt^3``/``1/dt^4`` Jacobians
amplify per-frame noise), and an L2 would let those spikes own the gradient
direction. Each delta is calibrated to the corresponding GT statistic's p75,
so GT-level smoothness sits in the quadratic zone and everything rougher
contributes a bounded, constant-magnitude pull.

Terms follow the ``(weighted_numerator, mass)`` contract so the trainer's
exact-DDP reduction applies; the term set is fixed by the config's nonzero
weights and every numerator carries a graph-connected zero, so no parameter
drops off the backward graph under ``find_unused_parameters=False``.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .keypoint_supervision import joint_weight_vector
from .motion_consistency import (
    predicted_root_world, quat_xyzw_from_matrix, so3_log_xyzw,
)

#: Loss terms, in config/report order. Each has a ``loss.<name>`` weight and a
#: ``loss.huber_delta_<name>`` transition.
TERM_NAMES = ("joint_jerk", "joint_snap", "root_pos_jerk", "root_pos_snap",
              "root_rot_jerk", "root_rot_snap")

#: Frames the stencils read (``t-2 .. t+2``).
STENCIL_WIDTH = 5


def _clip_dt(batch: dict, n_clips: int, seq_len: int,
             device: torch.device, dtype: torch.dtype) -> Tensor:
    """Mean frame interval per clip in seconds. ``(n_clips,)``."""
    pos_sec = batch["frame_pos_sec"].to(device, dtype).reshape(n_clips, seq_len)
    return (pos_sec[:, 1:] - pos_sec[:, :-1]).mean(dim=1).clamp(min=1e-6)


def position_jerk_snap(pos: Tensor, dt: Tensor) -> tuple[Tensor, Tensor]:
    """Central 5-point jerk and snap of a per-clip trajectory.

    :param pos: ``(n_clips, T, ..., 3)`` positions, uniformly sampled in time.
    :param dt: ``(n_clips,)`` frame interval in seconds.
    :returns: ``(jerk, snap)``, each ``(n_clips, T - 4, ..., 3)`` — row ``r``
        is centred on frame ``r + 2``.
    """
    scale = dt.reshape((-1,) + (1,) * (pos.dim() - 1)).to(pos.dtype)
    jerk = (-pos[:, :-4] + 2.0 * pos[:, 1:-3]
            - 2.0 * pos[:, 3:-1] + pos[:, 4:]) / (2.0 * scale**3)
    snap = (pos[:, :-4] - 4.0 * pos[:, 1:-3] + 6.0 * pos[:, 2:-2]
            - 4.0 * pos[:, 3:-1] + pos[:, 4:]) / scale**4
    return jerk, snap


def angular_jerk_snap(rot: Tensor, dt: Tensor) -> tuple[Tensor, Tensor]:
    """BVR body angular jerk and snap of a per-clip rotation trajectory.

    The half-step body rotation increments ``d[k] = so3_log(R_k^T R_{k+1})``
    (``= omega * dt`` at ``k + 1/2``) are differenced successively; the jerk is
    averaged back onto the integer grid the way
    :func:`~contact.motion_consistency.clip_body_twist` averages its velocity,
    so both outputs are centred on frame ``r + 2`` and read frames
    ``r .. r + 4`` — the same five-frame support as
    :func:`position_jerk_snap`.

    :param rot: ``(n_clips, T, 3, 3)`` world-from-root rotations.
    :param dt: ``(n_clips,)`` frame interval in seconds.
    :returns: ``(jerk, snap)``, each ``(n_clips, T - 4, 3)`` in rad/s^3, rad/s^4.
    """
    rel = rot[:, :-1].transpose(-1, -2) @ rot[:, 1:]              # (n, T-1, 3, 3)
    d = so3_log_xyzw(quat_xyzw_from_matrix(rel))                  # (n, T-1, 3)
    scale = dt.reshape(-1, 1, 1).to(d.dtype)
    jerk = 0.5 * (d[:, 3:] - d[:, 2:-1] - d[:, 1:-2] + d[:, :-3]) / scale**3
    snap = (d[:, 3:] - 3.0 * d[:, 2:-1]
            + 3.0 * d[:, 1:-2] - d[:, :-3]) / scale**4
    return jerk, snap


def stencil_support(ok: Tensor) -> Tensor:
    """Rows whose five-frame window is entirely valid. ``(n, T) -> (n, T-4)``."""
    return (ok[:, :-4] & ok[:, 1:-3] & ok[:, 2:-2]
            & ok[:, 3:-1] & ok[:, 4:])


class PoseSmoothnessLoss:
    """Jerk/snap minimisation on the predicted keypoint and root trajectories.

    :param cfg: resolved run config; reads ``pose_smoothness.*``.
    :param device: device the loss runs on.
    :param dtype: floating dtype of the joint channel and of every penalty
        (the root channels differentiate in float64 like
        :mod:`contact.motion_consistency`, then cast).
    """

    def __init__(
        self,
        cfg: dict,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        ps = cfg["pose_smoothness"]
        self.device = torch.device(device)
        self.dtype = dtype
        self.weights = {name: float(ps["loss"][name]) for name in TERM_NAMES}
        self.deltas = {name: float(ps["loss"][f"huber_delta_{name}"])
                       for name in TERM_NAMES}
        self.joint_w = joint_weight_vector(
            ps["joint_weights"]["fingers"], ps["joint_weights"]["face"],
            device=self.device, dtype=dtype)
        self.joint_w_sum = float(self.joint_w.sum())
        self.joints_on = any(
            self.weights[name] > 0.0 for name in ("joint_jerk", "joint_snap"))
        self.root_pos_on = any(
            self.weights[name] > 0.0
            for name in ("root_pos_jerk", "root_pos_snap"))
        self.root_rot_on = any(
            self.weights[name] > 0.0
            for name in ("root_rot_jerk", "root_rot_snap"))

    def __call__(self, out: dict, batch: dict) -> tuple[Tensor, dict[str, Any]]:
        return self.forward(out, batch)

    def forward(self, out: dict, batch: dict) -> tuple[Tensor, dict[str, Any]]:
        """Return ``(total, parts)``.

        :param out: forward output — reads ``out["mhr"]``
            (``pred_keypoints_3d (B, 70, 3)``, ``pred_cam_t (B, 3)``,
            ``global_rot (B, 3)``); gradients flow back into the pose path.
        :param batch: reads ``cam_from_world (B, 4, 4)``, ``cam_valid (B)``,
            ``frame_valid (B)``, ``frame_pos_sec (B)``, ``seq_len`` and — for
            the reference diagnostics only — ``kp3d_world``/``kp_valid``.
        :returns: ``(total, parts)``; ``parts["terms"][name]`` carries
            ``weighted_numerator_tensor`` + ``weight_mass`` (supported rows)
            per active term, alongside the pred/GT RMS diagnostics.
        """
        mhr = out["mhr"]
        kp3d = mhr["pred_keypoints_3d"].to(self.device, self.dtype)   # (B, 70, 3)
        cam_t = mhr["pred_cam_t"].to(self.device, self.dtype)         # (B, 3)
        zero_touch = (mhr["pred_keypoints_3d"].sum() + mhr["pred_cam_t"].sum()
                      + mhr["global_rot"].sum()).to(self.dtype) * 0.0

        n_frames = kp3d.shape[0]
        seq_len = int(batch.get("seq_len", 1))
        if seq_len < STENCIL_WIDTH or n_frames % seq_len:
            return self._assemble(
                {name: (zero_touch, 0.0) for name in TERM_NAMES},
                zero_touch, self._empty_diagnostics())
        n_clips = n_frames // seq_len

        ext = batch["cam_from_world"].to(self.device, self.dtype)     # (B, 4, 4)
        ok = (batch["frame_valid"].to(self.device)
              & batch["cam_valid"].to(self.device)).reshape(n_clips, seq_len)
        support = stencil_support(ok)                                 # (n, T-4)
        sup = support.to(self.dtype)
        mass = float(support.sum())

        terms: dict[str, tuple[Tensor, float]] = {
            name: (zero_touch, mass) for name in TERM_NAMES}
        diagnostics = self._empty_diagnostics()
        diagnostics["n_rows"] = int(support.sum())
        joint_w = self.joint_w[:, None]                               # (70, 1)

        def joint_mean(elementwise: Tensor) -> Tensor:
            """Weighted joint mean, coordinate sum: ``(..., 70, 3) -> (...)``."""
            return (elementwise * joint_w).sum(dim=(-2, -1)) / self.joint_w_sum

        if self.joints_on:
            # World lift, keypoint_supervision's composition exactly.
            pred_world = torch.einsum(
                "bji,bkj->bki", ext[:, :3, :3],
                kp3d + cam_t[:, None] - ext[:, :3, 3][:, None])
            pw = pred_world.reshape(n_clips, seq_len, kp3d.shape[1], 3)
            dt = _clip_dt(batch, n_clips, seq_len, self.device, self.dtype)
            jerk, snap = position_jerk_snap(pw, dt)                # (n, T-4, 70, 3)
            for name, value in (("joint_jerk", jerk), ("joint_snap", snap)):
                if self.weights[name] > 0.0:
                    huber = F.smooth_l1_loss(
                        value, torch.zeros_like(value), reduction="none",
                        beta=self.deltas[name])
                    terms[name] = ((joint_mean(huber) * sup).sum(), mass)
            with torch.no_grad():
                diagnostics["jerk_rms"] = self._joint_rms(jerk, support)
                diagnostics["snap_rms"] = self._joint_rms(snap, support)

        if self.root_pos_on or self.root_rot_on:
            pos_w, rot_w = predicted_root_world(mhr, batch["cam_from_world"])
            dt64 = _clip_dt(batch, n_clips, seq_len, self.device, torch.float64)
            if self.root_pos_on:
                jerk, snap = position_jerk_snap(
                    pos_w.reshape(n_clips, seq_len, 3), dt64)      # (n, T-4, 3)
                self._add_root_terms(
                    terms, diagnostics, "root_pos", jerk, snap, sup, support, mass)
            if self.root_rot_on:
                jerk, snap = angular_jerk_snap(
                    rot_w.reshape(n_clips, seq_len, 3, 3), dt64)
                self._add_root_terms(
                    terms, diagnostics, "root_rot", jerk, snap, sup, support, mass)

        if "kp3d_world" in batch:
            with torch.no_grad():
                gt_ok = ok & batch["kp_valid"].to(self.device).reshape(
                    n_clips, seq_len)
                gt_support = stencil_support(gt_ok)
                diagnostics["n_gt_rows"] = int(gt_support.sum())
                if bool(gt_support.any()):
                    gt = batch["kp3d_world"].to(self.device, self.dtype).reshape(
                        n_clips, seq_len, -1, 3)
                    dt = _clip_dt(batch, n_clips, seq_len, self.device, self.dtype)
                    gt_jerk, gt_snap = position_jerk_snap(gt, dt)
                    diagnostics["gt_jerk_rms"] = self._joint_rms(
                        gt_jerk, gt_support)
                    diagnostics["gt_snap_rms"] = self._joint_rms(
                        gt_snap, gt_support)

        return self._assemble(terms, zero_touch, diagnostics)

    def _add_root_terms(
        self,
        terms: dict[str, tuple[Tensor, float]],
        diagnostics: dict[str, float],
        channel: str,
        jerk: Tensor,
        snap: Tensor,
        sup: Tensor,
        support: Tensor,
        mass: float,
    ) -> None:
        """Fill the ``root_pos``/``root_rot`` term pair and its RMS diagnostics."""
        for order, raw in (("jerk", jerk), ("snap", snap)):
            name = f"{channel}_{order}"
            value = raw.to(self.dtype)
            if self.weights[name] > 0.0:
                huber = F.smooth_l1_loss(
                    value, torch.zeros_like(value), reduction="none",
                    beta=self.deltas[name])
                terms[name] = ((huber.sum(dim=-1) * sup).sum(), mass)
            with torch.no_grad():
                norm = value.norm(dim=-1)[support]
                diagnostics[f"{name}_rms"] = (
                    float(norm.square().mean().sqrt()) if norm.numel() else 0.0)

    def _joint_rms(self, value: Tensor, support: Tensor) -> float:
        """Joint-weighted RMS of the per-joint vector norms over supported rows."""
        selected = value[support]                                  # (rows, 70, 3)
        if not selected.numel():
            return 0.0
        weights = self.joint_w.to(selected.device, selected.dtype)
        squared = selected.square().sum(dim=-1)                    # (rows, 70)
        return float(((squared * weights).sum()
                      / (squared.shape[0] * self.joint_w_sum)).sqrt())

    @staticmethod
    def _empty_diagnostics() -> dict[str, float]:
        """All-zero diagnostic block (an inactive batch reports zeros)."""
        return {"jerk_rms": 0.0, "snap_rms": 0.0,
                "gt_jerk_rms": 0.0, "gt_snap_rms": 0.0,
                "root_pos_jerk_rms": 0.0, "root_pos_snap_rms": 0.0,
                "root_rot_jerk_rms": 0.0, "root_rot_snap_rms": 0.0,
                "n_rows": 0, "n_gt_rows": 0}

    def _assemble(
        self,
        terms: dict[str, tuple[Tensor, float]],
        zero_touch: Tensor,
        diagnostics: dict[str, Any],
    ) -> tuple[Tensor, dict[str, Any]]:
        """Weight, normalise and package (the MotionConsistencyLoss contract)."""
        parts_terms: dict[str, dict[str, Any]] = {}
        total: Tensor | None = None
        for name in TERM_NAMES:
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
        if total is None:                        # every weight is zero (degenerate)
            total = zero_touch
        parts: dict[str, Any] = {"terms": parts_terms,
                                 "loss": float(total.detach())}
        parts.update(diagnostics)
        return total, parts


__all__ = ["PoseSmoothnessLoss", "TERM_NAMES", "STENCIL_WIDTH",
           "position_jerk_snap", "angular_jerk_snap", "stencil_support"]
