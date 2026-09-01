"""World-frame lift of the model's own predicted body root, plus the SO(3) helpers it needs.

The pose head speaks camera-crop language: ``pred_keypoints_3d`` are
camera-frame metres relative to the crop's principal point, ``pred_cam_t`` is
the crop translation and ``global_rot`` is an ``xyz`` Euler triple on the MHR
head's *native* axes. Anything that wants to compare the prediction against a
metric world quantity (kindyn root poses, GT extrinsics, force positions) has
to undo all three. :func:`predicted_root_world` is that single conversion,
kept differentiable end to end so the gradient reaches the pose path.

The quaternion/log helpers are torch mirrors of the loader's float64 target
derivation — same hemisphere flip, same small-angle Taylor branch — so a
prediction and its target are compared through identical arithmetic.
"""
from __future__ import annotations

import roma
import torch
from torch import Tensor

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


