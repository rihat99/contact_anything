"""Pose→motion consistency: differentiate the PREDICTED pose, compare as motion.

The predicted per-frame body placement (``out["mhr"]``: hip keypoints +
``pred_cam_t`` + ``global_rot``) is lifted to the metric reconstruction world
with the dataset camera extrinsics, differentiated over the clip with the SAME
body-twist scheme the kindyn motion targets use
(:func:`~contact.data.climbing_corpus.root_body_twist`), standardized with the
``motion_supervision`` pelvis table, and compared against

- ``gt``   — the kindyn GT pelvis twist (``motion_gt``): pulls the POSE path
  toward physically true motion (the classic failure it attacks is depth
  wobble — world-position jitter that reprojects fine but differentiates into
  huge accelerations);
- ``head`` — the motion head's own prediction, **detached**: the pose-derived
  twist is pulled toward the motion estimate, never the other way round. The
  motion head learns from ``motion_supervision`` alone, so a degenerate pose
  trajectory cannot drag it down (the corpus_allmod_mutual failure).

Derivative terms alone leave the ABSOLUTE root placement in a null space: a
constant ``pred_cam_t`` contributes nothing to world velocity under a static
camera, and corpus_allmod_mutual exploited exactly that (camera collapsed to a
constant 9 cm depth while q-space pose error stayed small). Three anchor terms
close it:

- ``pos`` — predicted world mean-hips vs the kindyn root position lifted to the
  same point: ``p_gt + R_gt @ hip_offset_root`` (``motion_root_pos`` /
  ``motion_rot``; the constant offset mean-hips − root-origin was measured on
  the motion-probe artifact, ≈ 9 cm). Huber in metres, every valid row — no
  stencil, so clip boundaries ARE supervised here.
- ``rot`` — geodesic residual ``so3_log(R_pred^T R_gt)`` vs zero, Huber in
  radians (probe: constant frame offset ≈ 1.4°, i.e. none; per-frame model
  error ≈ 7°).
- ``cam_rail`` — trust region on the camera translation: zero inside
  ``cam_rail_margin_m`` of the frozen model's own ``pred_cam_t`` (stashed by
  the recompute hook as ``pred_cam_t_frozen``), linear beyond. Inert for a
  healthy model; makes the collapse optimum unreachable.
- ``rot_rail`` — the same trust region on the global orientation: zero inside
  ``rot_rail_margin_rad`` of the frozen model's own ``global_rot``
  (``global_rot_frozen``), linear beyond. corpus_allmod_consistency (v3)
  proved the necessity: with only the camera railed, the twist terms pinned
  the world orientation near-constant (per-clip spread 0.21° vs GT 2.85°),
  parking it ~55° from GT — the rotation channel was the one open escape.

``angular: false`` (the v4 default stance) restricts the ``gt``/``head`` twist
comparison to the LINEAR vel/acc rows even when ``motion_supervision.angular``
is on. The angular residuals are dominated by the frozen model's ~7° per-frame
orientation wobble differentiated at 30 fps and standardized by the small GT
angular std — exactly the signal that makes a constant orientation the
optimum. The motion head keeps its full 12-dim supervision either way.

World lifting is the composition verified against the motion-probe artifact
(``output/motion_probe_geom``, max deviation 2.4e-7 m)::

    p_world = R_ext^T @ ((mean(kp3d[9], kp3d[10]) + pred_cam_t) - t_ext)
    R_world_from_root = R_ext^T @ diag(1,-1,-1) @ euler_to_rotmat("xyz", global_rot)

(``pred_keypoints_3d`` leave the MHR head already in camera axes; ``global_rot``
is the native-axes euler triple, hence the flip.) Everything stays on the
autograd graph — the recomputed final pose output means the pose bricks and the
fine-tuned pose head receive these gradients.

The interior twist stencil needs frames ``t-1, t, t+1``, so clip boundaries are
never supervised and ``frames_per_clip >= 3`` is required. All comparisons run
in the standardized units of the pinned ``motion_supervision.standardize``
pelvis row; the term contract mirrors :class:`.MotionSupervisedLoss`
(``(weighted_numerator, mass)`` per term for exact DDP reduction).
"""
from __future__ import annotations

from typing import Any

import roma
import torch
import torch.nn.functional as F
from torch import Tensor

from .data.climbing_corpus import MOTION_JOINT_NAMES

#: Camera-vs-native axis flip (``det = +1``, a rotation) — the MHR head's
#: axes-1,2 flip, matching ``contact/physics/adapter.py``.
_FLIP = torch.diag(torch.tensor([1.0, -1.0, -1.0]))

#: MHR70 keypoint indices whose mean is the model's own pelvis (left/right hip).
_HIP_KPS = (9, 10)

#: Small-angle Taylor threshold on ``theta^2``; the lie math below runs in
#: float64 (tiny tensors), mirroring the loader's target derivation exactly.
_TAYLOR_THETA2 = 1e-8


def quat_xyzw_from_matrix(rot: Tensor) -> Tensor:
    """Differentiable rotation-matrix -> ``xyzw`` quaternion. ``(..., 3, 3) -> (..., 4)``.

    Shepperd's method: all four candidate quaternions are formed and the
    best-conditioned one (largest denominator) is selected per element with
    ``torch.where`` — smooth wherever the selection is locally constant.
    """
    m = rot
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    trace = m00 + m11 + m22
    # Four squared denominators (each >= 0): 1+trace and 1+2*mii-trace.
    q_sq = torch.stack([
        1.0 + trace, 1.0 + 2.0 * m00 - trace,
        1.0 + 2.0 * m11 - trace, 1.0 + 2.0 * m22 - trace], dim=-1)
    q_sq = q_sq.clamp(min=0.0)
    best = q_sq.argmax(dim=-1, keepdim=True)
    denom = 0.5 / (q_sq.gather(-1, best).squeeze(-1) + 1e-12).sqrt()

    w0 = 0.25 / denom
    cands = torch.stack([
        torch.stack([(m[..., 2, 1] - m[..., 1, 2]) * denom,
                     (m[..., 0, 2] - m[..., 2, 0]) * denom,
                     (m[..., 1, 0] - m[..., 0, 1]) * denom, w0], dim=-1),
        torch.stack([w0,
                     (m[..., 0, 1] + m[..., 1, 0]) * denom,
                     (m[..., 0, 2] + m[..., 2, 0]) * denom,
                     (m[..., 2, 1] - m[..., 1, 2]) * denom], dim=-1),
        torch.stack([(m[..., 0, 1] + m[..., 1, 0]) * denom, w0,
                     (m[..., 1, 2] + m[..., 2, 1]) * denom,
                     (m[..., 0, 2] - m[..., 2, 0]) * denom], dim=-1),
        torch.stack([(m[..., 0, 2] + m[..., 2, 0]) * denom,
                     (m[..., 1, 2] + m[..., 2, 1]) * denom, w0,
                     (m[..., 1, 0] - m[..., 0, 1]) * denom], dim=-1),
    ], dim=-2)                                             # (..., 4 cands, 4)
    quat = cands.gather(
        -2, best[..., None].expand(*best.shape[:-1], 1, 4)).squeeze(-2)
    return quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def so3_log_xyzw(quat: Tensor) -> Tensor:
    """Rotation vector of an ``xyzw`` unit quaternion. ``(..., 4) -> (..., 3)``.

    Torch mirror of the loader's float64 target derivation — same hemisphere
    flip and small-angle Taylor branch.
    """
    quat = torch.where(quat[..., 3:4] < 0.0, -quat, quat)
    qxyz, qw = quat[..., :3], quat[..., 3:4]
    sin_half2 = (qxyz * qxyz).sum(dim=-1, keepdim=True)
    taylor = sin_half2 < _TAYLOR_THETA2 / 4.0
    sin_half = torch.where(taylor, torch.ones_like(sin_half2), sin_half2).sqrt()
    theta = 2.0 * torch.atan2(sin_half, qw.clamp(-1.0, 1.0))
    factor = torch.where(
        taylor, 2.0 + sin_half2 * (2.0 / 3.0), theta / sin_half.clamp(min=1e-30))
    return factor * qxyz


def se3_log(trans: Tensor, quat: Tensor) -> Tensor:
    """``log`` of the SE3 element ``(trans, quat_xyzw)`` -> ``(..., 6)``.

    The linear part carries the ``V^{-1}(omega)`` correction, layout
    ``(linear, angular)`` — the loader's :func:`se3_log_xyzw` in torch.
    """
    omega = so3_log_xyzw(quat)
    theta2 = (omega * omega).sum(dim=-1, keepdim=True)
    taylor = theta2 < _TAYLOR_THETA2
    theta2_safe = torch.where(taylor, torch.ones_like(theta2), theta2)
    theta = theta2_safe.sqrt()
    cot_half = torch.cos(theta / 2.0) / torch.sin(theta / 2.0).clamp(min=1e-30)
    coeff = torch.where(
        taylor, (1.0 / 12.0) + theta2 / 720.0,
        1.0 / theta2_safe - cot_half / (2.0 * theta))
    zeros = torch.zeros_like(omega[..., :1])
    skew = torch.cat([
        zeros, -omega[..., 2:3], omega[..., 1:2],
        omega[..., 2:3], zeros, -omega[..., 0:1],
        -omega[..., 1:2], omega[..., 0:1], zeros], dim=-1).reshape(*omega.shape, 3)
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device)
    v_inv = eye - 0.5 * skew + coeff[..., None] * (skew @ skew)
    linear = (v_inv @ trans[..., None]).squeeze(-1)
    return torch.cat([linear, omega], dim=-1)


def predicted_root_world(
    mhr_out: dict, cam_from_world: Tensor,
) -> tuple[Tensor, Tensor]:
    """Predicted pelvis world position + world-from-root rotation, ``(B, 3)``/``(B, 3, 3)``.

    Fully differentiable w.r.t. the pose path: keypoints, ``pred_cam_t`` and
    ``global_rot`` all come out of the recomputed final pose output.
    """
    dtype = torch.float64
    kp = mhr_out["pred_keypoints_3d"].to(dtype)
    cam_t = mhr_out["pred_cam_t"].to(dtype)
    ext = cam_from_world.to(kp.device, dtype)
    pelvis_cam = kp[:, list(_HIP_KPS)].mean(dim=1) + cam_t          # camera axes
    rot_ext = ext[:, :3, :3]
    trans_ext = ext[:, :3, 3]
    pos_w = torch.einsum(
        "bji,bj->bi", rot_ext, pelvis_cam - trans_ext)              # R_ext^T (p - t)
    rot_native = roma.euler_to_rotmat(
        "xyz", mhr_out["global_rot"].to(dtype))                     # native axes
    flip = _FLIP.to(kp.device, dtype)
    rot_w = rot_ext.transpose(-1, -2) @ flip @ rot_native
    return pos_w, rot_w


def clip_body_twist(
    pos: Tensor, rot: Tensor, dt: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """BVR body-twist v/a of a world root trajectory, per clip.

    The loader's :func:`root_body_twist` stencil on matrices::

        d[t] = se3_log(T_t^-1 T_t+1);  v[t] = (d[t-1]+d[t]) / 2dt;  a[t] = (d[t]-d[t-1]) / dt^2

    :param pos: ``(n_clips, T, 3)`` world pelvis positions.
    :param rot: ``(n_clips, T, 3, 3)`` world-from-root rotations.
    :param dt: ``(n_clips,)`` frame interval in seconds.
    :returns: ``(vel, acc, ang_vel, ang_acc)``, each ``(n_clips, T, 3)`` —
        boundary frames are zero (they are never supervised).
    """
    rel_rot = rot[:, :-1].transpose(-1, -2) @ rot[:, 1:]            # (n, T-1, 3, 3)
    rel_pos = torch.einsum(
        "ntji,ntj->nti", rot[:, :-1], pos[:, 1:] - pos[:, :-1])
    diff = se3_log(rel_pos, quat_xyzw_from_matrix(rel_rot))         # (n, T-1, 6)

    dt = dt[:, None, None].to(diff.dtype)
    twist = torch.zeros(
        pos.shape[0], pos.shape[1], 6, dtype=diff.dtype, device=diff.device)
    acc = torch.zeros_like(twist)
    twist[:, 1:-1] = 0.5 * (diff[:, :-1] + diff[:, 1:]) / dt
    acc[:, 1:-1] = (diff[:, 1:] - diff[:, :-1]) / (dt * dt)
    return twist[..., :3], acc[..., :3], twist[..., 3:], acc[..., 3:]


class MotionConsistencyLoss:
    """The consistency + absolute-anchor terms, in :class:`.MotionSupervisedLoss`'s contract.

    :param cfg: resolved run config; reads ``motion_consistency.*`` and the
        ``motion_supervision`` table/weights it standardizes against.
    :param device: device the loss runs on.
    """

    def __init__(
        self,
        cfg: dict,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        mc = cfg["motion_consistency"]
        ms = cfg["motion_supervision"]
        self.weights = {"gt": float(mc["loss"]["gt"]),
                        "head": float(mc["loss"]["head"]),
                        "pos": float(mc["loss"]["pos"]),
                        "rot": float(mc["loss"]["rot"]),
                        "cam_rail": float(mc["loss"]["cam_rail"]),
                        "rot_rail": float(mc["loss"]["rot_rail"])}
        self.huber_delta = float(mc["loss"]["huber_delta"])
        self.pos_huber_m = float(mc["loss"]["pos_huber_m"])
        self.rot_huber_rad = float(mc["loss"]["rot_huber_rad"])
        self.cam_rail_margin_m = float(mc["loss"]["cam_rail_margin_m"])
        self.rot_rail_margin_rad = float(mc["loss"]["rot_rail_margin_rad"])
        self.hip_offset_root = torch.tensor(
            [float(v) for v in mc["hip_offset_root"]], dtype=dtype)
        names = tuple(ms.get("joint_names") or MOTION_JOINT_NAMES)
        self.pelvis_slot = names.index("pelvis")
        # The twist comparison covers the angular rows only when BOTH the head
        # is trained on them and the consistency config opts in.
        ms_angular = bool(ms.get("angular", False))
        self.angular = bool(mc["angular"]) and ms_angular
        self.groups = ("vel", "acc") + (("ang_vel", "ang_acc") if self.angular else ())
        n_ms_groups = 2 + (2 if ms_angular else 0)
        # Per-3-component-group weights, shared with motion_supervision so the
        # three pelvis objectives (head<->GT, pose<->GT, pose<->head) weight
        # vel/acc/angular identically.
        group_w = torch.tensor(
            [float(ms["loss"][g]) for g in self.groups], dtype=dtype)
        self.group_weights = group_w.repeat_interleave(3)[None, :]    # (1, 3G)
        width = 3 * len(self.groups)
        mean = torch.tensor(ms["standardize"]["mean"], dtype=dtype)
        std = torch.tensor(ms["standardize"]["std"], dtype=dtype)
        # The standardize table (and the head/GT rows) follow motion_supervision's
        # group count; slice the leading vel/acc groups when angular is off here.
        self.mean = mean[self.pelvis_slot].reshape(n_ms_groups, 3)[
            : len(self.groups)].reshape(1, width)
        self.std = std[self.pelvis_slot].reshape(n_ms_groups, 3)[
            : len(self.groups)].reshape(1, width)
        self.device = torch.device(device)
        self.dtype = dtype
        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)
        self.group_weights = self.group_weights.to(self.device)
        self.hip_offset_root = self.hip_offset_root.to(self.device)

    def __call__(
        self, out: dict, batch: dict, exclude_outliers: bool = True,
    ) -> tuple[Tensor, dict[str, Any]]:
        return self.forward(out, batch, exclude_outliers)

    def forward(
        self, out: dict, batch: dict, exclude_outliers: bool = True,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Return ``(total, parts)``.

        :param out: forward output — reads ``out["mhr"]`` (pose path, grads
            live; ``pred_cam_t_frozen`` for the rail when a pose write path
            exists) and ``out["motion"]["joint_motion"]`` (motion head,
            DETACHED — the head term never trains the motion head).
        :param batch: reads ``cam_from_world``/``cam_valid``, ``frame_valid``,
            ``frame_pos_sec``, ``seq_len`` and the motion GT keys (incl.
            ``motion_root_pos``/``motion_root_valid``/``motion_rot`` for the
            absolute anchors).
        :param exclude_outliers: apply the pelvis outlier bit to the ``gt``
            term (training); evaluation never filters.
        """
        width = self.mean.shape[1]
        head = out["motion"]["joint_motion"][:, self.pelvis_slot, :width].detach(
            ).to(self.device, self.dtype)                             # (B, 3G)
        zero_touch = sum(
            out["mhr"][k].sum() * 0.0
            for k in ("pred_keypoints_3d", "pred_cam_t", "global_rot"))

        term_names = ("gt", "head", "pos", "rot", "cam_rail", "rot_rail")
        seq_len = int(batch.get("seq_len", 1))
        batch_rows = head.shape[0]
        if seq_len < 3 or batch_rows % seq_len:
            return self._assemble(
                {n: (zero_touch, 0.0) for n in term_names}, zero_touch, {})
        n_clips = batch_rows // seq_len

        pos_w_flat, rot_w_flat = predicted_root_world(
            out["mhr"], batch["cam_from_world"])                      # (B,3)/(B,3,3) f64
        pos_w = pos_w_flat.reshape(n_clips, seq_len, 3)
        rot_w = rot_w_flat.reshape(n_clips, seq_len, 3, 3)
        pos_sec = batch["frame_pos_sec"].to(self.device, torch.float64)
        pos_sec = pos_sec.reshape(n_clips, seq_len)
        dt = (pos_sec[:, 1:] - pos_sec[:, :-1]).mean(dim=1).clamp(min=1e-6)

        vel, acc, ang_vel, ang_acc = clip_body_twist(pos_w, rot_w, dt)
        pose_twist = torch.cat(
            ([vel, acc, ang_vel, ang_acc] if self.angular else [vel, acc]),
            dim=-1).reshape(batch_rows, -1).to(self.dtype)            # (B, 3G)
        pose_std = (pose_twist - self.mean) / self.std

        # The stencil at t reads frames t-1, t, t+1: every one must be a real,
        # extrinsics-valid frame. Clip boundaries have no stencil support.
        ok = (batch["frame_valid"].to(self.device)
              & batch["cam_valid"].to(self.device)).reshape(n_clips, seq_len)
        support = torch.zeros_like(ok)
        support[:, 1:-1] = ok[:, :-2] & ok[:, 1:-1] & ok[:, 2:]
        support = support.reshape(batch_rows)                         # (B,)

        gt = batch["motion_gt"][:, self.pelvis_slot, :width].to(
            self.device, self.dtype)
        gt_std = (gt - self.mean) / self.std
        gt_mask = support & batch["motion_valid"].to(self.device)
        if exclude_outliers:
            gt_mask = gt_mask & ~batch["motion_outlier"][:, self.pelvis_slot].to(
                self.device)

        def _term(target: Tensor, mask: Tensor) -> tuple[Tensor, float]:
            huber = F.smooth_l1_loss(
                pose_std, target, reduction="none", beta=self.huber_delta)
            per_row = (huber * self.group_weights).sum(dim=-1)        # (B,)
            return (per_row * mask).sum(), float(mask.sum())

        terms = {"gt": _term(gt_std, gt_mask), "head": _term(head, support)}

        # Absolute root-pose anchors — per-frame, no stencil: every real,
        # extrinsics-valid, kindyn-covered row supervises (boundaries included).
        pose_ok = (batch["frame_valid"].to(self.device)
                   & batch["cam_valid"].to(self.device)
                   & batch["motion_root_valid"].to(self.device))      # (B,)
        rot_gt = batch["motion_rot"].to(self.device, self.dtype)      # (B, 3, 3)
        pos_target = (batch["motion_root_pos"].to(self.device, self.dtype)
                      + torch.einsum("bij,j->bi", rot_gt, self.hip_offset_root))
        pos_pred = pos_w_flat.to(self.dtype)
        pos_huber = F.smooth_l1_loss(
            pos_pred, pos_target, reduction="none",
            beta=self.pos_huber_m).sum(dim=-1)                        # (B,)
        terms["pos"] = ((pos_huber * pose_ok).sum(), float(pose_ok.sum()))

        rot_res = so3_log_xyzw(quat_xyzw_from_matrix(
            rot_w_flat.transpose(-1, -2)
            @ rot_gt.to(torch.float64))).to(self.dtype)               # (B, 3) rad
        rot_huber = F.smooth_l1_loss(
            rot_res, torch.zeros_like(rot_res), reduction="none",
            beta=self.rot_huber_rad).sum(dim=-1)
        terms["rot"] = ((rot_huber * pose_ok).sum(), float(pose_ok.sum()))

        # Camera trust region: pure linear penalty beyond the margin around the
        # frozen model's own translation — exactly zero inside it. Absent
        # pred_cam_t_frozen means no pose write path, so nothing can drift.
        frozen = out["mhr"].get("pred_cam_t_frozen")
        if frozen is not None:
            cam_dev = (out["mhr"]["pred_cam_t"].to(self.device, self.dtype)
                       - frozen.to(self.device, self.dtype)).norm(dim=-1)  # (B,)
            rail = F.relu(cam_dev - self.cam_rail_margin_m)
            rail_mask = batch["frame_valid"].to(self.device)
            terms["cam_rail"] = ((rail * rail_mask).sum(), float(rail_mask.sum()))
        else:
            cam_dev = None
            terms["cam_rail"] = (zero_touch, 0.0)

        # Rotation trust region — the native-axes relative rotation between the
        # predicted and frozen ``global_rot`` (extrinsics cancel in a geodesic
        # distance, so this needs no camera validity).
        frozen_rot = out["mhr"].get("global_rot_frozen")
        if frozen_rot is not None:
            r_pred = roma.euler_to_rotmat(
                "xyz", out["mhr"]["global_rot"].to(torch.float64))
            r_frz = roma.euler_to_rotmat(
                "xyz", frozen_rot.to(self.device, torch.float64))
            rot_dev = so3_log_xyzw(quat_xyzw_from_matrix(
                r_pred.transpose(-1, -2) @ r_frz)).norm(dim=-1).to(self.dtype)
            rot_rail = F.relu(rot_dev - self.rot_rail_margin_rad)
            rot_rail_mask = batch["frame_valid"].to(self.device)
            terms["rot_rail"] = (
                (rot_rail * rot_rail_mask).sum(), float(rot_rail_mask.sum()))
        else:
            rot_dev = None
            terms["rot_rail"] = (zero_touch, 0.0)

        with torch.no_grad():
            # pose_twist is already physical (clip_body_twist output) — compare
            # raw against the raw GT; only the loss terms are standardized.
            diff = (pose_twist - gt).reshape(batch_rows, -1)
            m = gt_mask.to(torch.float64)
            n = m.sum().clamp(min=1.0)
            n_pose = pose_ok.to(torch.float64).sum().clamp(min=1.0)
            diagnostics = {
                "vel_rmse": float(((diff[:, 0:3].to(torch.float64).square()
                                    .sum(-1) * m).sum() / n).sqrt()),
                "acc_rmse": float(((diff[:, 3:6].to(torch.float64).square()
                                    .sum(-1) * m).sum() / n).sqrt()),
                "pos_err_m": float(
                    ((pos_pred - pos_target).norm(dim=-1).to(torch.float64)
                     * pose_ok).sum() / n_pose),
                "rot_err_deg": float(
                    (rot_res.norm(dim=-1).to(torch.float64) * pose_ok).sum()
                    / n_pose * 180.0 / torch.pi),
                "n_gt_rows": int(gt_mask.sum()),
                "n_head_rows": int(support.sum()),
                "n_pose_rows": int(pose_ok.sum()),
            }
            if cam_dev is not None:
                nf = batch["frame_valid"].to(self.device, torch.float64)
                diagnostics["cam_dev_m"] = float(
                    (cam_dev.to(torch.float64) * nf).sum()
                    / nf.sum().clamp(min=1.0))
                diagnostics["rail_frac"] = float(
                    (((cam_dev > self.cam_rail_margin_m).to(torch.float64))
                     * nf).sum() / nf.sum().clamp(min=1.0))
            if rot_dev is not None:
                nf = batch["frame_valid"].to(self.device, torch.float64)
                diagnostics["rot_dev_deg"] = float(
                    (rot_dev.to(torch.float64) * nf).sum()
                    / nf.sum().clamp(min=1.0) * 180.0 / torch.pi)
                diagnostics["rot_rail_frac"] = float(
                    (((rot_dev > self.rot_rail_margin_rad).to(torch.float64))
                     * nf).sum() / nf.sum().clamp(min=1.0))
        return self._assemble(terms, zero_touch, diagnostics)

    def _assemble(
        self,
        terms: dict[str, tuple[Tensor, float]],
        zero_touch: Tensor,
        diagnostics: dict[str, Any],
    ) -> tuple[Tensor, dict[str, Any]]:
        """Weight, normalise and package (the MotionSupervisedLoss contract)."""
        parts_terms: dict[str, dict[str, Any]] = {}
        total: Tensor | None = None
        for name, (raw, mass) in terms.items():
            if self.weights[name] == 0.0:
                continue
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


__all__ = ["MotionConsistencyLoss", "predicted_root_world", "clip_body_twist",
           "quat_xyzw_from_matrix", "so3_log_xyzw", "se3_log"]
