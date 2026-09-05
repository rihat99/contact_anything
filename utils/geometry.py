"""Torch geometry shared by the SMPL-X head and its loss / metrics.

Frames and units, the bug-prone part every consumer here depends on:

* The SMPL-X head predicts the body in the OpenCV CAMERA frame (metres);
  ``cam_from_world`` is the dataset's metric OpenCV extrinsic ``[R | t]``
  (world -> camera), so the world lift of an absolute camera-frame point is
  ``p_w = R^T (p_c - t)``.
* Camera translation has two parametrizations: CLIFF's crop weak-perspective
  proxy ``(s, tx, ty)`` (lifted with the crop box and the focal) and the pelvis
  ray ``(x/z, y/z, log z)``. Both directions of each are here, so the head and
  the GT proxy targets share one arithmetic.

Everything is differentiable end to end except :func:`procrustes_align`
(metrics only).
"""
from __future__ import annotations

import roma
import torch
from torch import Tensor


def lift_to_world(points_cam: Tensor, cam_from_world: Tensor) -> Tensor:
    """Absolute camera-frame points -> world. ``(B, K, 3)``, ``(B, 4, 4)`` -> ``(B, K, 3)``.

    ``p_w = R_ext^T (p_c - t_ext)``.
    """
    ext = cam_from_world.to(points_cam.device, points_cam.dtype)
    return torch.einsum(
        "bji,bkj->bki", ext[:, :3, :3], points_cam - ext[:, :3, 3][:, None])


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


def translation_to_ray(trans: Tensor) -> Tensor:
    """Camera translation ``(x, y, z)`` -> the ray parametrization ``(x/z, y/z, log z)``.

    The bearing is the pelvis pixel's normalized image coordinate ``((u - px) / f,
    (v - py) / f)``; the depth is logarithmic so per-frame noise is relative.
    """
    z = trans[:, 2].clamp(min=1e-3)
    return torch.stack([trans[:, 0] / z, trans[:, 1] / z, torch.log(z)], dim=-1)


def ray_to_translation(ray: Tensor) -> Tensor:
    """Exact inverse of :func:`translation_to_ray`: ``exp(d) (rx, ry, 1)``."""
    z = torch.exp(ray[:, 2])
    return torch.stack([ray[:, 0] * z, ray[:, 1] * z, z], dim=-1)


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


def smplx_q(pelvis: Tensor, root_rot: Tensor, body_rot: Tensor,
            hand_rot: Tensor | None = None) -> Tensor:
    """Assemble the BetterHuman SMPL-X configuration ``(B, 91)`` (``(B, 211)`` with hands).

    ``[pelvis (3), root quat xyzw (4), 21 body-joint quats, (30 finger quats)]`` — the
    root of ``q`` IS the pelvis pose, joint rotations are parent-local.

    :param pelvis: pelvis position ``(B, 3)`` in the frame ``root_rot`` is expressed in.
    :param root_rot: frame-from-root rotation ``(B, 3, 3)``.
    :param body_rot: parent-local body rotations ``(B, 21, 3, 3)``.
    :param hand_rot: parent-local finger rotations ``(B, 30, 3, 3)`` or ``None``.
    """
    parts = [
        pelvis,
        roma.rotmat_to_unitquat(root_rot),                                # xyzw
        roma.rotmat_to_unitquat(body_rot).reshape(pelvis.shape[0], -1),
    ]
    if hand_rot is not None:
        parts.append(roma.rotmat_to_unitquat(hand_rot).reshape(pelvis.shape[0], -1))
    return torch.cat(parts, dim=-1)


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
