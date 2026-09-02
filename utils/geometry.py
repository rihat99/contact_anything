"""Torch geometry shared by the loss terms: SO(3) logs and the camera -> world lift.

Frames and units, the bug-prone part every consumer here depends on:

* The pose head speaks camera-crop language. ``pred_keypoints_3d`` /
  ``pred_vertices`` are camera-frame metres RELATIVE to the body root,
  ``pred_cam_t`` is the crop translation that places them, and ``global_rot``
  is an ``xyz`` Euler triple on the MHR head's *native* axes — the camera axes
  differ from those by ``diag(1, -1, -1)`` (a rotation, ``det = +1``).
* ``cam_from_world`` is the dataset's metric OpenCV extrinsic ``[R | t]``
  (world -> camera), so the world lift of an absolute camera-frame point is
  ``p_w = R^T (p_c - t)``.

Anything that compares a prediction against a metric world quantity (kindyn
root poses, GT keypoints, force positions) has to undo all three; doing it in
one place is what keeps the keypoint, contact-consistency and force-consistency
losses on the same convention. Everything is differentiable end to end, so the
gradient reaches the pose path.

The quaternion / log helpers mirror the loader's float64 target derivation —
same hemisphere flip, same small-angle Taylor branch — so a prediction and its
target are compared through identical arithmetic.
"""
from __future__ import annotations

import roma
import torch
from torch import Tensor

#: Camera-vs-native axis flip of the MHR head (axes 1, 2; ``det = +1``).
CAMERA_FROM_NATIVE = torch.diag(torch.tensor([1.0, -1.0, -1.0]))

#: MHR70 left/right hip keypoints. Their mean is the body placement every
#: world lift of the prediction is rooted at (MHR70 has no pelvis keypoint).
HIP_KEYPOINTS = (9, 10)

#: Small-angle Taylor threshold on ``theta^2`` for the quaternion log.
_TAYLOR_THETA2 = 1e-8


def matrix_to_quat_xyzw(rot: Tensor) -> Tensor:
    """Differentiable rotation matrix -> ``xyzw`` quaternion. ``(..., 3, 3) -> (..., 4)``.

    Shepperd's method: all four candidate quaternions are formed and the
    best-conditioned one (largest denominator) is selected per element with
    ``gather`` — smooth wherever the selection is locally constant.
    """
    m = rot
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    trace = m00 + m11 + m22
    # Four squared denominators (each >= 0): 1 + trace and 1 + 2 m_ii - trace.
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
    ], dim=-2)                                              # (..., 4 cands, 4)
    quat = cands.gather(
        -2, best[..., None].expand(*best.shape[:-1], 1, 4)).squeeze(-2)
    return quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def quat_log_xyzw(quat: Tensor) -> Tensor:
    """Rotation vector of an ``xyzw`` unit quaternion. ``(..., 4) -> (..., 3)``.

    Hemisphere-aligned (``w >= 0``) so the log is the shortest rotation, with a
    Taylor branch near identity where ``theta / sin(theta / 2)`` is ill-conditioned.
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


def so3_log(rot: Tensor) -> Tensor:
    """Rotation vector of a rotation matrix. ``(..., 3, 3) -> (..., 3)``.

    Taylor-smooth at the identity, unlike a ``acos`` geodesic angle whose
    gradient explodes there — which is what a trust-region rail needs.
    """
    return quat_log_xyzw(matrix_to_quat_xyzw(rot))


def lift_to_world(points_cam: Tensor, cam_from_world: Tensor) -> Tensor:
    """Absolute camera-frame points -> world. ``(B, K, 3)``, ``(B, 4, 4)`` -> ``(B, K, 3)``.

    ``p_w = R_ext^T (p_c - t_ext)``. Differencing predictions in the world
    (rather than the camera) is what removes the camera egomotion exactly:
    camera-frame differences bury body motion under handheld shake.
    """
    ext = cam_from_world.to(points_cam.device, points_cam.dtype)
    return torch.einsum(
        "bji,bkj->bki", ext[:, :3, :3], points_cam - ext[:, :3, 3][:, None])


def predicted_keypoints_world(
    mhr: dict,
    cam_from_world: Tensor,
    indices: Tensor | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Predicted MHR70 keypoints in world metres. ``(B, K, 3)``.

    :param mhr: the model's MHR readout (``pred_keypoints_3d``, ``pred_cam_t``).
    :param cam_from_world: ``(B, 4, 4)`` metric OpenCV extrinsics.
    :param indices: optional keypoint subset (``None`` = all 70).
    """
    kp = mhr["pred_keypoints_3d"].to(dtype)
    if indices is not None:
        kp = kp[:, indices]
    cam_t = mhr["pred_cam_t"].to(dtype)
    return lift_to_world(kp + cam_t[:, None], cam_from_world)


def predicted_root_world(
    mhr: dict, cam_from_world: Tensor,
) -> tuple[Tensor, Tensor]:
    """Predicted body root in the world: position ``(B, 3)`` and rotation ``(B, 3, 3)``.

    The position is the mean-hips keypoint (:data:`HIP_KEYPOINTS`) placed by
    ``pred_cam_t`` and lifted with the extrinsics; the rotation composes the
    extrinsics, the camera-vs-native flip and the head's Euler triple. Runs in
    float64 — the consumer differentiates it twice.
    """
    dtype = torch.float64
    kp = mhr["pred_keypoints_3d"].to(dtype)
    cam_t = mhr["pred_cam_t"].to(dtype)
    ext = cam_from_world.to(kp.device, dtype)
    pelvis_cam = kp[:, list(HIP_KEYPOINTS)].mean(dim=1) + cam_t
    pos_w = lift_to_world(pelvis_cam[:, None], ext)[:, 0]
    rot_native = roma.euler_to_rotmat("xyz", mhr["global_rot"].to(dtype))
    flip = CAMERA_FROM_NATIVE.to(kp.device, dtype)
    rot_w = ext[:, :3, :3].transpose(-1, -2) @ flip @ rot_native
    return pos_w, rot_w


def windowed_mean(x: Tensor, kernel: Tensor) -> Tensor:
    """Kernel-smooth ``x`` along the time axis with edge replication.

    :param x: ``(..., T, C)`` Euclidean channels.
    :param kernel: ``(L,)`` odd-length non-negative weights (normalised here);
        a single-element kernel returns ``x`` unchanged.
    :returns: ``(..., T, C)`` smoothed; edge frames use replicated boundaries.
    """
    if kernel.numel() == 1:
        return x
    weights = kernel / kernel.sum()
    num_knots = x.shape[-2]
    radius = kernel.numel() // 2
    offsets = torch.arange(-radius, radius + 1, device=x.device)
    centers = torch.arange(num_knots, device=x.device).unsqueeze(-1)
    indices = (centers + offsets).clamp(0, num_knots - 1)      # (T, L)
    return (x[..., indices, :] * weights.view(-1, 1)).sum(-2)


# ------------------------------------------------------------------ SMPL-X head

def rotmat_to_rot6d(rot: Tensor) -> Tensor:
    """Zhou et al. continuous 6D: the first two COLUMNS of ``R``, concatenated.

    ``(..., 3, 3) -> (..., 6)`` laid out ``[col0, col1]`` (the pytorch3d /
    PromptHMR convention; the identity is ``[1,0,0, 0,1,0]``).
    """
    return rot[..., :, :2].transpose(-1, -2).reshape(*rot.shape[:-2], 6)


def rot6d_to_rotmat(six: Tensor) -> Tensor:
    """Gram-Schmidt inverse of :func:`rotmat_to_rot6d`. ``(..., 6) -> (..., 3, 3)``."""
    columns = six.reshape(*six.shape[:-1], 2, 3).transpose(-1, -2)    # (..., 3, 2)
    return roma.special_gramschmidt(columns)


def cliff_cam_to_translation(
    cam: Tensor, bbox_center: Tensor, bbox_size: Tensor, cam_int: Tensor,
) -> Tensor:
    """CLIFF crop weak-perspective ``(s, tx, ty)`` -> full-image camera translation.

    ``t = (tx + 2 (cx - px) / (b s), ty + 2 (cy - py) / (b s), 2 f / (b s))``
    with ``b`` the square crop side in full-image pixels, ``(cx, cy)`` its
    centre, ``(px, py)`` the principal point and ``f`` the focal — the same
    lift SAM 3D Body's camera head performs, minus its native-axis sign flips.

    :param cam: ``(B, 3)``; ``bbox_center`` ``(B, 2)``; ``bbox_size`` ``(B,)``;
        ``cam_int`` ``(B, 3, 3)``.
    :returns: ``(B, 3)`` metres in the OpenCV camera frame.
    """
    s = cam[:, 0].clamp(min=0.05)
    bs = bbox_size * s
    focal = cam_int[:, 0, 0]
    tz = 2.0 * focal / bs
    tx = cam[:, 1] + 2.0 * (bbox_center[:, 0] - cam_int[:, 0, 2]) / bs
    ty = cam[:, 2] + 2.0 * (bbox_center[:, 1] - cam_int[:, 1, 2]) / bs
    return torch.stack([tx, ty, tz], dim=-1)


def translation_to_cliff_cam(
    trans: Tensor, bbox_center: Tensor, bbox_size: Tensor, cam_int: Tensor,
) -> Tensor:
    """Exact inverse of :func:`cliff_cam_to_translation` (the GT proxy target)."""
    focal = cam_int[:, 0, 0]
    z = trans[:, 2].clamp(min=1e-3)
    s = 2.0 * focal / (bbox_size * z)
    tx = trans[:, 0] - (bbox_center[:, 0] - cam_int[:, 0, 2]) * z / focal
    ty = trans[:, 1] - (bbox_center[:, 1] - cam_int[:, 1, 2]) * z / focal
    return torch.stack([s, tx, ty], dim=-1)


def project_to_crop(
    points_cam: Tensor, cam_int: Tensor, affine_trans: Tensor, img_size: Tensor,
    min_depth: float = 0.25,
) -> tuple[Tensor, Tensor]:
    """Camera-frame points -> full-image pixels and the crop-normalized space.

    Mirrors SAM 3D Body's ``_full_to_crop`` exactly: intrinsics -> full-image
    px -> crop affine -> ``/ img_size - 0.5`` (so ``[-0.5, 0.5]`` spans the crop).

    :param points_cam: ``(B, K, 3)``; ``cam_int`` ``(B, 3, 3)``;
        ``affine_trans`` ``(B, 2, 3)`` full -> crop; ``img_size`` ``(B, 2)``.
    :returns: ``(pixels (B, K, 2), crop (B, K, 2))``.
    """
    z = points_cam[..., 2].clamp(min=min_depth)
    u = cam_int[:, 0, 0, None] * points_cam[..., 0] / z + cam_int[:, 0, 2, None]
    v = cam_int[:, 1, 1, None] * points_cam[..., 1] / z + cam_int[:, 1, 2, None]
    pixels = torch.stack([u, v], dim=-1)
    homog = torch.cat([pixels, torch.ones_like(u)[..., None]], dim=-1)   # (B,K,3)
    crop = (homog @ affine_trans.mT) / img_size[:, None] - 0.5
    return pixels, crop


def procrustes_align(pred: Tensor, gt: Tensor) -> Tensor:
    """Similarity-align ``pred`` onto ``gt`` per row (PA-MPJPE). ``(B, K, 3)`` each.

    Closed-form Umeyama in float64, no gradient: returns ``pred`` after the
    best rotation, uniform scale and translation.
    """
    with torch.no_grad():
        p = pred.to(torch.float64)
        g = gt.to(torch.float64)
        mu_p, mu_g = p.mean(dim=1, keepdim=True), g.mean(dim=1, keepdim=True)
        p0, g0 = p - mu_p, g - mu_g
        var_p = (p0 * p0).sum(dim=(1, 2))
        cov = g0.transpose(1, 2) @ p0                              # (B,3,3)
        u, sigma, vh = torch.linalg.svd(cov)
        sign = torch.sign(torch.linalg.det(u @ vh))
        d = torch.ones_like(sigma)
        d[:, -1] = sign
        rot = u @ torch.diag_embed(d) @ vh
        scale = (sigma * d).sum(dim=-1) / var_p.clamp(min=1e-12)
        aligned = scale[:, None, None] * (p0 @ rot.transpose(1, 2)) + mu_g
    return aligned.to(pred.dtype)
