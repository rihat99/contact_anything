"""SMPL-X supervision of the from-scratch pose + camera heads, and the pose metrics.

Targets are the corpus kindyn SMPL-X body (BetterHuman ``q`` convention:
root = pelvis pose), lifted from the world into the camera with the frame
extrinsics. The prediction is :class:`~model.heads.SmplxHead`'s output, which
already carries camera-frame joints and the projected 2D points.

Terms (every one a per-frame MEAN over its elements, mass = supervised frames):

* ``kp2d`` — Huber on the crop-normalized ``[-0.5, 0.5]`` 2D body joints. The
  projection is full-frame (true intrinsics, full-image camera translation —
  CLIFF's point: the crop camera cannot express the crop's bearing angle), the
  metric is crop-normalized so its scale is bounded (SAM3D / HMR2.0 practice).
* ``kp3d`` — Huber on pelvis-relative camera-frame joints (metres).
* ``orient`` / ``pose`` — MSE on the raw 6D outputs vs the GT's first two
  rotation-matrix columns (GVHMR / WHAM practice); GT rotations arrive as
  matrices, never as the stored axis-angles (those sit off the principal branch).
* ``betas`` — MSE on the 10 shape coefficients (per-person GT served per frame).
* ``cam`` — Huber on the CLIFF ``(s, tx, ty)`` proxy vs the GT pelvis inverted
  into the same proxy — translation is supervised in proxy space, as every
  surveyed method does, never in metres.

Metrics (eval only) follow WHAM's ``evaluate_3dpw.py`` / GVHMR's
``compute_camcoord_metrics`` line by line, on the 22 SMPL-X body joints and
the 10475 vertices with flat hands:

* every frame is aligned by the MEAN OF THE TWO HIP JOINTS (their
  ``pelvis_idxs = [1, 2]``); the vertices are shifted by that same joint-derived
  pelvis;
* ``mpjpe`` / ``pa_mpjpe`` / ``pve`` (mm) — per-frame mean L2 over joints /
  Procrustes-aligned joints / vertices;
* ``accel`` (m/s^2) — per-frame mean over joints of the L2 error of the second
  finite difference of the aligned joints. The papers divide by ``(1/30 s)^2``
  (their footage is 30 fps); here the division uses each clip's REAL frame
  spacing, so the number is theirs on 30 fps footage and fps-exact on the
  corpus's 24-60 fps scenes. GVHMR's trim (every interior frame; WHAM also
  drops the first and last) is used.

The reduction is frame-weighted (``np.concatenate(all frames).mean()`` in both
repos), which is exactly what the additive ``(sum, count)`` statistics give.
"""
from __future__ import annotations

import roma
import torch
import torch.nn.functional as F
from torch import Tensor

from model.loss import Loss, LossResult
from utils.geometry import (
    procrustes_align,
    project_to_crop,
    rotmat_to_rot6d,
    translation_to_cliff_cam,
)
from utils.metrics import mean_from_stats

_TERM_NAMES = ("kp2d", "kp3d", "orient", "pose", "betas", "cam")
#: Reported metrics, in the order of the statistics vector.
POSE_METRICS = ("mpjpe", "pa_mpjpe", "pve", "accel")
#: Minimum camera-frame depth (metres) for a projectable GT row.
_MIN_DEPTH_M = 0.25
#: The two hip joints whose mean is the alignment pelvis (WHAM/GVHMR
#: ``pelvis_idxs`` for the SMPL 24 / SMPL-X 22 body joint set).
SMPLX_HIPS = (1, 2)


def gt_smplx_camera(batch: dict, device, dtype=torch.float32) -> dict:
    """The kindyn SMPL-X GT of a batch, lifted into each frame's camera.

    :returns: ``joints (B, 22, 3)`` metres, ``root_rot (B, 3, 3)``
        camera-from-root, ``body_rot (B, 21, 3, 3)`` parent-local, ``betas
        (B, 10)``, ``q (B, 91)`` the BetterHuman camera-frame configuration,
        ``valid (B,)`` bool (labelled, tracked, and in front of the camera).
    """
    ext = batch["cam_from_world"].to(device, dtype)                      # (B, 4, 4)
    rot_cw, t_cw = ext[:, :3, :3], ext[:, :3, 3]
    joints_world = batch["smplx_joints_world"].to(device, dtype)
    joints = torch.einsum("bij,bkj->bki", rot_cw, joints_world) + t_cw[:, None]
    root_rot = rot_cw @ batch["smplx_root_rot"].to(device, dtype)
    body_rot = batch["smplx_body_rot"].to(device, dtype)
    valid = (batch["smplx_valid"] & batch["frame_valid"]).to(device)
    valid = valid & (joints[..., 2] > _MIN_DEPTH_M).all(dim=-1)
    return {
        "joints": joints, "root_rot": root_rot, "body_rot": body_rot,
        "betas": batch["smplx_betas"].to(device, dtype),
        "q": smplx_q(joints[:, 0], root_rot, body_rot), "valid": valid,
    }


def smplx_q(pelvis: Tensor, root_rot: Tensor, body_rot: Tensor) -> Tensor:
    """Assemble the 22-joint BetterHuman configuration ``(B, 91)``.

    :param pelvis: pelvis position ``(B, 3)`` (the root of ``q`` IS the pelvis).
    :param root_rot: root rotation ``(B, 3, 3)``.
    :param body_rot: parent-local body rotations ``(B, 21, 3, 3)``.
    """
    return torch.cat([
        pelvis,
        roma.rotmat_to_unitquat(root_rot),                                # xyzw
        roma.rotmat_to_unitquat(body_rot).reshape(pelvis.shape[0], -1),
    ], dim=-1)


def smplx_vertices(body, betas: Tensor, q: Tensor) -> Tensor:
    """Skinned vertices ``(B, 10475, 3)`` of a BetterHuman SMPL-X body (flat hands)."""
    shaped = body.with_shape(betas=betas)
    return body.vertices_from_data(shaped.fk(q))


@torch.no_grad()
def pose_metric_stats(
    pred_joints: Tensor, pred_verts: Tensor, gt_joints: Tensor, gt_verts: Tensor,
    valid: Tensor, seq_len: int, frame_pos_sec: Tensor,
) -> Tensor:
    """Additive ``(sum, count)`` pairs of :data:`POSE_METRICS` — float64 ``[8]``.

    :param pred_joints: ``(B, 22, 3)`` camera metres, ``B = n_clips * seq_len``
        clip-major.
    :param pred_verts: ``(B, V, 3)``.
    :param gt_joints: ``(B, 22, 3)``.
    :param gt_verts: ``(B, V, 3)``.
    :param valid: ``(B,)`` bool rows that count.
    :param seq_len: frames per clip (the acceleration stencil never crosses a clip).
    :param frame_pos_sec: ``(B,)`` elapsed seconds per frame.
    """
    hips = list(SMPLX_HIPS)
    pred_pelvis = pred_joints[:, hips].mean(dim=1, keepdim=True)
    gt_pelvis = gt_joints[:, hips].mean(dim=1, keepdim=True)
    pj, gj = pred_joints - pred_pelvis, gt_joints - gt_pelvis
    pv, gv = pred_verts - pred_pelvis, gt_verts - gt_pelvis
    mask = valid.to(pred_joints.dtype)
    count = float(mask.sum())

    mpjpe = (pj - gj).norm(dim=-1).mean(dim=-1) * 1000.0
    pa_mpjpe = (procrustes_align(pj, gj) - gj).norm(dim=-1).mean(dim=-1) * 1000.0
    pve = (pv - gv).norm(dim=-1).mean(dim=-1) * 1000.0

    accel_sum, accel_count = 0.0, 0.0
    if seq_len >= 3:
        n_clips = pred_joints.shape[0] // seq_len
        pj_t = pj.reshape(n_clips, seq_len, *pj.shape[1:])
        gj_t = gj.reshape(n_clips, seq_len, *gj.shape[1:])
        t = frame_pos_sec.to(pred_joints.dtype).reshape(n_clips, seq_len)
        v = valid.reshape(n_clips, seq_len)
        dt = 0.5 * (t[:, 2:] - t[:, :-2])                                # (n, T-2)

        def second(x: Tensor) -> Tensor:
            return x[:, :-2] - 2.0 * x[:, 1:-1] + x[:, 2:]

        err = (second(pj_t) - second(gj_t)).norm(dim=-1).mean(dim=-1)   # (n, T-2)
        err = err / dt.clamp(min=1e-6) ** 2                             # m/s^2
        row = (v[:, :-2] & v[:, 1:-1] & v[:, 2:] & (dt > 0)).to(err.dtype)
        accel_sum, accel_count = float((err * row).sum()), float(row.sum())

    return torch.tensor([
        float((mpjpe * mask).sum()), count,
        float((pa_mpjpe * mask).sum()), count,
        float((pve * mask).sum()), count,
        accel_sum, accel_count,
    ], dtype=torch.float64)


def pose_metrics_from_stats(stats: Tensor) -> dict[str, float]:
    """:data:`POSE_METRICS` from the summed statistics vector."""
    return {name: mean_from_stats(float(stats[2 * i]), float(stats[2 * i + 1]))
            for i, name in enumerate(POSE_METRICS)}


class SmplxLoss(Loss):
    """Huber keypoint / proxy-camera terms and 6D / betas MSE for the SMPL-X head."""

    name = "smplx"
    metric_group = "pose"
    stat_names = tuple(f"{key}_{part}" for key in POSE_METRICS for part in ("sum", "count"))

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        super().__init__(cfg, model, device)
        loss_cfg = cfg["smplx_supervision"]["loss"]
        self.weights = {name: float(loss_cfg[name]) for name in _TERM_NAMES}
        self.term_names = tuple(n for n in _TERM_NAMES if self.weights[n] > 0.0)
        if not self.term_names:
            raise ValueError(
                "smplx_supervision: every loss weight is 0 — disable the section instead")
        self.delta_2d = float(loss_cfg["huber_delta_2d"])
        self.delta_3d = float(loss_cfg["huber_delta_3d"])
        self.delta_cam = float(loss_cfg["huber_delta_cam"])

    def __call__(self, out: dict, batch: dict, *, train: bool) -> LossResult:
        pred = out["smplx"]
        root_6d = pred["root_6d"].to(self.device, self.dtype)            # (B,6)
        body_6d = pred["body_6d"].to(self.device, self.dtype)            # (B,21,6)
        betas = pred["betas"].to(self.device, self.dtype)                # (B,10)
        cam = pred["cam"].to(self.device, self.dtype)                    # (B,3)
        joints = pred["joints_cam"].to(self.device, self.dtype)          # (B,22,3)
        kp2d_crop = pred["kp2d_crop"].to(self.device, self.dtype)        # (B,22,2)
        anchor = (root_6d.sum() + body_6d.sum() + betas.sum() + cam.sum()) * 0.0

        gt = gt_smplx_camera(batch, self.device, self.dtype)
        gt_joints = gt["joints"]
        mask = gt["valid"].to(self.dtype)                                # (B,)
        mass = float(mask.sum())

        cam_int = batch["cam_int"].to(self.device, self.dtype)
        affine = batch["affine_trans"].to(self.device, self.dtype)
        img_size = batch["img_size"].to(self.device, self.dtype)
        bbox_center = batch["bbox_center"].to(self.device, self.dtype)
        bbox_size = batch["bbox_scale"].to(self.device, self.dtype)[:, 0]
        _, gt_crop = project_to_crop(gt_joints, cam_int, affine, img_size)
        gt_cam = translation_to_cliff_cam(gt_joints[:, 0], bbox_center, bbox_size, cam_int)
        gt_root_6d = rotmat_to_rot6d(gt["root_rot"])
        gt_body_6d = rotmat_to_rot6d(gt["body_rot"])

        raw: dict[str, tuple[Tensor, float]] = {}
        if self.weights["kp2d"] > 0.0:
            huber = F.smooth_l1_loss(kp2d_crop, gt_crop, reduction="none", beta=self.delta_2d)
            raw["kp2d"] = ((huber.mean(dim=(1, 2)) * mask).sum(), mass)
        if self.weights["kp3d"] > 0.0:
            huber = F.smooth_l1_loss(joints - joints[:, :1], gt_joints - gt_joints[:, :1],
                                     reduction="none", beta=self.delta_3d)
            raw["kp3d"] = ((huber.mean(dim=(1, 2)) * mask).sum(), mass)
        if self.weights["orient"] > 0.0:
            raw["orient"] = (((root_6d - gt_root_6d).square().mean(dim=-1) * mask).sum(), mass)
        if self.weights["pose"] > 0.0:
            raw["pose"] = (((body_6d - gt_body_6d).square().mean(dim=(1, 2)) * mask).sum(), mass)
        if self.weights["betas"] > 0.0:
            raw["betas"] = (((betas - gt["betas"]).square().mean(dim=-1) * mask).sum(), mass)
        if self.weights["cam"] > 0.0:
            huber = F.smooth_l1_loss(cam, gt_cam, reduction="none", beta=self.delta_cam)
            raw["cam"] = ((huber.mean(dim=-1) * mask).sum(), mass)

        stats = self.empty_stats()
        if not train:
            # Vertices only at evaluation: the metrics are their sole consumer.
            body = self.model.head_smplx.body(self.device)
            pred_verts = smplx_vertices(body, betas.detach(), pred["q_cam"].detach())
            gt_verts = smplx_vertices(body, gt["betas"], gt["q"])
            stats = pose_metric_stats(
                joints.detach(), pred_verts, gt_joints, gt_verts, gt["valid"],
                int(batch["seq_len"]), batch["frame_pos_sec"].to(self.device),
            ).to(self.device)
        weighted = {name: (self.weights[name] * numerator, term_mass)
                    for name, (numerator, term_mass) in raw.items()}
        return LossResult(terms=self._terms(weighted, anchor), scalars={"n_rows": mass},
                          stats=stats)

    def metrics(self, stats: Tensor) -> dict[str, float]:
        return pose_metrics_from_stats(stats)


__all__ = ["SmplxLoss", "POSE_METRICS", "SMPLX_HIPS", "gt_smplx_camera", "smplx_q",
           "smplx_vertices", "pose_metric_stats", "pose_metrics_from_stats"]
