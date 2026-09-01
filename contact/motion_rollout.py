"""Roll-out consistency: integrate the predicted velocity, compare displacements.

:mod:`contact.motion_consistency` differentiates the predicted pose and compares
twists. This module runs the other direction: it **integrates** the motion head's
predicted root velocity over the clip and compares the resulting displacement
against (a) the kindyn GT world root path and (b) the same path implied by the
PREDICTED pose. Differentiating amplifies exactly the frequencies where the
pseudo-GT is worst (the camera-depth wobble); integrating suppresses them and
asks the question a per-frame derivative loss cannot — *did the body actually
travel this far over this second?*

Only **displacements over a horizon** are compared, never absolute positions, so
the constant of integration never enters (absolute placement stays the job of
``keypoint_supervision`` / ``motion_consistency``'s anchors). Several horizons
run at once: short ones say roughly what the derivative loss says, long ones
carry the low-frequency constraint.

The predicted velocity is rotated to the world with ``motion_lin_rot`` — the SAME
world-from-linear-frame rotation the loader expressed the target in. Under
``motion_supervision.root_convention: gravity_view`` that rotation is built from
gravity and the camera view direction and carries no body orientation, so the
integral cannot compound the predicted pose's orientation error; the config
validator therefore restricts this loss to that convention.

Terms, in the ``(weighted_numerator, mass)`` contract the trainer's
:func:`~contact.losses.ddp_global_mean_term` reduces exactly under DDP:

- ``gt`` — integrated displacement vs the GT root path (grad -> motion head: a
  low-frequency supervision signal the per-frame Huber cannot give).
- ``pose`` — integrated displacement vs the PREDICTED pose's root path. With
  ``detach_head`` (default) the integrated side is detached, so the gradient
  reaches the pose path ONLY: the pose trajectory is pulled toward the motion
  head's smoother estimate, never the reverse.
- ``rot_gt`` / ``rot_pose`` — the same two comparisons for orientation
  (``motion_supervision.angular`` runs only). The predicted body rate is
  composed over the horizon as ``prod exp(omega dt)`` — body rates multiply on
  the right — and compared geodesically with the GT / predicted relative
  rotation.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .motion_consistency import (
    predicted_root_world, quat_xyzw_from_matrix, so3_log_xyzw)

_LINEAR_TERMS = ("gt", "pose")
_ANGULAR_TERMS = ("rot_gt", "rot_pose")


def so3_exp(vec: Tensor) -> Tensor:
    """Rodrigues exponential of axis-angle vectors, ``(..., 3) -> (..., 3, 3)``."""
    theta = vec.norm(dim=-1, keepdim=True)                            # (..., 1)
    small = theta < 1e-8
    safe = theta.clamp(min=1e-8)
    sin_c = torch.where(small, torch.ones_like(safe), torch.sin(safe) / safe)
    cos_c = torch.where(small, 0.5 * torch.ones_like(safe),
                        (1.0 - torch.cos(safe)) / (safe * safe))
    zeros = torch.zeros_like(vec[..., 0])
    skew = torch.stack([
        zeros, -vec[..., 2], vec[..., 1],
        vec[..., 2], zeros, -vec[..., 0],
        -vec[..., 1], vec[..., 0], zeros,
    ], dim=-1).reshape(*vec.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=vec.dtype, device=vec.device)
    return eye + sin_c[..., None] * skew + cos_c[..., None] * (skew @ skew)


def prefix_rotations(steps: Tensor) -> Tensor:
    """Prefix products of per-step rotations, ``(n, S, 3, 3) -> (n, S + 1, 3, 3)``.

    ``out[:, t] = steps[:, 0] @ ... @ steps[:, t-1]`` (identity at ``t = 0``), so
    the relative rotation over any window is ``out[:, t]^T @ out[:, t + H]`` —
    every horizon for the price of one scan.
    """
    acc = torch.eye(3, dtype=steps.dtype, device=steps.device).expand(
        steps.shape[0], 3, 3)
    prefixes = [acc]
    for step in steps.unbind(dim=1):
        acc = acc @ step
        prefixes.append(acc)
    return torch.stack(prefixes, dim=1)


class MotionRolloutLoss:
    """Horizon-displacement agreement between the integrated velocity and the pose.

    :param cfg: resolved run config; reads ``motion_rollout.*`` and the
        ``motion_supervision`` standardization table (to de-standardize the head).
    :param device: device the loss runs on.
    :param dtype: floating dtype (float32).
    """

    def __init__(
        self,
        cfg: dict,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        mr = cfg["motion_rollout"]
        ms = cfg["motion_supervision"]
        self.horizons = tuple(int(h) for h in mr["horizons"])
        self.detach_head = bool(mr["detach_head"])
        self.angular = bool(ms["angular"])
        self.term_names = _LINEAR_TERMS + (_ANGULAR_TERMS if self.angular else ())
        self.weights = {name: float(mr["loss"][name]) for name in self.term_names}
        self.huber_m = float(mr["loss"]["huber_m"])
        self.huber_rad = float(mr["loss"]["huber_rad"])
        self.device = torch.device(device)
        self.dtype = dtype
        # The head is standardized; the roll-out is metric, so de-standardize with
        # the same pinned table motion_supervision uses. Pelvis is the only slot
        # (validator), so row 0 is it.
        mean = torch.tensor(ms["standardize"]["mean"], dtype=dtype)[0]   # [G, 3]
        std = torch.tensor(ms["standardize"]["std"], dtype=dtype)[0]
        self.mean = mean.reshape(-1).to(self.device)                     # [3G]
        self.std = std.reshape(-1).to(self.device)

    def __call__(self, out: dict, batch: dict) -> tuple[Tensor, dict[str, Any]]:
        return self.forward(out, batch)

    def forward(self, out: dict, batch: dict) -> tuple[Tensor, dict[str, Any]]:
        """Return ``(total, parts)``.

        :param out: forward output — reads ``out["motion"]["joint_motion"]``
            (grads live unless ``detach_head`` for the pose terms) and
            ``out["mhr"]`` (pose path).
        :param batch: reads ``motion_lin_rot``, ``motion_root_pos``,
            ``motion_root_valid``, ``motion_rot``, ``cam_from_world``,
            ``cam_valid``, ``frame_valid``, ``frame_pos_sec`` and ``seq_len``.
        """
        pred = out["motion"]["joint_motion"].to(self.device, self.dtype)  # (B, K, 3G)
        zero_touch = pred.sum() * 0.0 + sum(
            out["mhr"][key].sum() * 0.0
            for key in ("pred_keypoints_3d", "pred_cam_t", "global_rot"))

        seq_len = int(batch.get("seq_len", 1))
        rows = pred.shape[0]
        if seq_len <= min(self.horizons) or rows % seq_len:
            # Clips too short for the shortest horizon carry no window at all
            # (T = 1 still images included). The diagnostics keep their keys so
            # the trainer's eval accumulator never has to guard.
            return self._assemble(
                {name: (zero_touch, 0.0) for name in self.term_names},
                zero_touch,
                {"disp_err_m": 0.0, "n_rows": 0,
                 **({"rot_err_deg": 0.0, "n_rot_rows": 0} if self.angular else {}),
                 **{f"{key}_h{h}": 0 if key == "n_rows" else 0.0
                    for h in self.horizons
                    for key in ("disp_err_m", "n_rows")
                    + (("rot_err_deg",) if self.angular else ())}})
        n_clips = rows // seq_len
        horizons = [h for h in self.horizons if h < seq_len]

        phys = pred[:, 0] * self.std + self.mean                          # (B, 3G)
        lin_rot = batch["motion_lin_rot"].to(self.device, self.dtype).detach()
        vel_w = torch.einsum("bij,bj->bi", lin_rot, phys[:, 0:3])         # (B, 3) world
        vel_w = vel_w.reshape(n_clips, seq_len, 3)

        pos_sec = batch["frame_pos_sec"].to(self.device, self.dtype).reshape(
            n_clips, seq_len)
        dt = (pos_sec[:, 1:] - pos_sec[:, :-1]).clamp(min=1e-6)           # (n, T-1)
        # Trapezoid. Substituting the target's own central difference
        # v[i] = (p[i+1] - p[i-1]) / 2dt telescopes to the displacement of the
        # [1/4, 1/2, 1/4]-smoothed path, i.e. the exact displacement plus
        # (dt^2 / 4)(a[t+H] - a[t]) — sub-millimetre at 25 fps against a 0.1 m
        # Huber knee, and exact for constant velocity.
        step = 0.5 * (vel_w[:, :-1] + vel_w[:, 1:]) * dt[..., None]       # (n, T-1, 3)
        path = torch.cat([
            torch.zeros_like(step[:, :1]), step.cumsum(dim=1)], dim=1)    # (n, T, 3)

        pos_pred_w, rot_pred_w = predicted_root_world(
            out["mhr"], batch["cam_from_world"])
        pose_pos = pos_pred_w.to(self.dtype).reshape(n_clips, seq_len, 3)
        pose_rot = rot_pred_w.to(self.dtype).reshape(n_clips, seq_len, 3, 3)
        gt_pos = batch["motion_root_pos"].to(self.device, self.dtype).reshape(
            n_clips, seq_len, 3)
        gt_rot = batch["motion_rot"].to(self.device, self.dtype).reshape(
            n_clips, seq_len, 3, 3)

        ok = (batch["frame_valid"].to(self.device)
              & batch["cam_valid"].to(self.device)).reshape(n_clips, seq_len)
        root_ok = batch["motion_root_valid"].to(self.device).reshape(
            n_clips, seq_len)
        # A window contributes only when EVERY frame it integrates over is real
        # and extrinsics-valid: prefix-count the invalid frames and take windows
        # whose count is unchanged.
        invalid = torch.cat([
            torch.zeros_like(ok[:, :1], dtype=torch.int32),
            (~ok).to(torch.int32).cumsum(dim=1)], dim=1)                  # (n, T+1)

        rot_prefix = None
        if self.angular:
            omega = phys[:, 6:9].reshape(n_clips, seq_len, 3)
            omega_step = 0.5 * (omega[:, :-1] + omega[:, 1:]) * dt[..., None]
            rot_prefix = prefix_rotations(so3_exp(omega_step))            # (n, T, 3, 3)

        terms = {name: [torch.zeros((), device=self.device, dtype=self.dtype), 0.0]
                 for name in self.term_names}
        # Per-horizon, because a number pooling a 0.12 s error with a 1.2 s one
        # reads as neither. The pooled pair is kept as the headline.
        diagnostics: dict[str, Any] = {}
        err_m, err_m_mass, err_deg, err_deg_mass = 0.0, 0.0, 0.0, 0.0
        for horizon in horizons:
            window = (invalid[:, horizon + 1:] - invalid[:, :-horizon - 1]) == 0
            disp = path[:, horizon:] - path[:, :-horizon]                 # (n, m, 3)
            gt_window = window & root_ok[:, :-horizon] & root_ok[:, horizon:]
            targets = {
                "gt": (gt_pos[:, horizon:] - gt_pos[:, :-horizon], gt_window, disp),
                "pose": (pose_pos[:, horizon:] - pose_pos[:, :-horizon], window,
                         disp.detach() if self.detach_head else disp),
            }
            for name, (target, mask, source) in targets.items():
                if self.weights[name] == 0.0:
                    continue
                huber = F.smooth_l1_loss(
                    source, target, reduction="none", beta=self.huber_m).sum(dim=-1)
                terms[name][0] = terms[name][0] + (huber * mask).sum()
                terms[name][1] += float(mask.sum())
            with torch.no_grad():
                err_h = float(((disp - targets["gt"][0]).norm(dim=-1) * gt_window).sum())
                mass_h = float(gt_window.sum())
                diagnostics[f"disp_err_m_h{horizon}"] = err_h / max(mass_h, 1.0)
                diagnostics[f"n_rows_h{horizon}"] = int(mass_h)
                err_m += err_h
                err_m_mass += mass_h

            if not self.angular:
                continue
            rel_pred = (rot_prefix[:, :-horizon].transpose(-1, -2)
                        @ rot_prefix[:, horizon:])                        # (n, m, 3, 3)
            rot_targets = {
                "rot_gt": (gt_rot[:, :-horizon].transpose(-1, -2) @ gt_rot[:, horizon:],
                           gt_window, rel_pred),
                "rot_pose": (pose_rot[:, :-horizon].transpose(-1, -2)
                             @ pose_rot[:, horizon:], window,
                             rel_pred.detach() if self.detach_head else rel_pred),
            }
            for name, (target, mask, source) in rot_targets.items():
                residual = self._geodesic(source, target)                 # (n, m, 3)
                if self.weights[name] != 0.0:
                    huber = F.smooth_l1_loss(
                        residual, torch.zeros_like(residual), reduction="none",
                        beta=self.huber_rad).sum(dim=-1)
                    terms[name][0] = terms[name][0] + (huber * mask).sum()
                    terms[name][1] += float(mask.sum())
                if name == "rot_gt":
                    with torch.no_grad():
                        rot_h = float((residual.norm(dim=-1) * mask).sum())
                        rot_mass_h = float(mask.sum())
                        diagnostics[f"rot_err_deg_h{horizon}"] = (
                            rot_h / max(rot_mass_h, 1.0) * 180.0 / torch.pi)
                        err_deg += rot_h
                        err_deg_mass += rot_mass_h

        diagnostics["disp_err_m"] = err_m / max(err_m_mass, 1.0)   # pooled over horizons
        diagnostics["n_rows"] = int(err_m_mass)
        if self.angular:
            diagnostics["rot_err_deg"] = (
                err_deg / max(err_deg_mass, 1.0) * 180.0 / torch.pi)
            diagnostics["n_rot_rows"] = int(err_deg_mass)
        return self._assemble(
            {name: (value, mass) for name, (value, mass) in terms.items()},
            zero_touch, diagnostics)

    @staticmethod
    def _geodesic(pred_rel: Tensor, target_rel: Tensor) -> Tensor:
        """``so3_log(R_pred^T R_target)`` for stacked windows, ``(n, m, 3)`` rad."""
        residual = pred_rel.transpose(-1, -2) @ target_rel
        flat = residual.reshape(-1, 3, 3).to(torch.float64)
        return so3_log_xyzw(quat_xyzw_from_matrix(flat)).reshape(
            *residual.shape[:-2], 3).to(pred_rel.dtype)

    def _assemble(
        self,
        terms: dict[str, tuple[Tensor, float]],
        zero_touch: Tensor,
        diagnostics: dict[str, Any],
    ) -> tuple[Tensor, dict[str, Any]]:
        """Weight, normalise and package the term contract (see the motion loss)."""
        parts_terms: dict[str, dict[str, Any]] = {}
        total: Tensor | None = None
        for name in self.term_names:
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
        if total is None:
            total = zero_touch
        parts: dict[str, Any] = {"terms": parts_terms, "loss": float(total.detach())}
        parts.update(diagnostics)
        return total, parts


__all__ = ["MotionRolloutLoss", "so3_exp", "prefix_rotations"]
