"""SMPL-X supervision of the from-scratch pose + camera heads, and the pose metrics.

Targets are the corpus kindyn SMPL-X body (BetterHuman ``q`` convention:
root = pelvis pose), lifted from the world into the camera with the frame
extrinsics. The prediction is :class:`~model.heads.SmplxHead`'s output, which
already carries camera-frame joints and the projected 2D points.

Terms (every one a per-frame MEAN over its elements, mass = supervised frames):

* ``kp2d`` — Huber on the 2D body joints. The projection is full-frame (true
  intrinsics, full-image camera translation — CLIFF's point: the crop camera
  cannot express the crop's bearing angle); the error is measured either
  crop-normalized (``kp2d_space: crop``, ``[-0.5, 0.5]`` spans the crop —
  SAM3D / HMR2.0 practice, but its scale rides on the per-frame crop side) or
  in bearing units (``image``: full-image px / f, the camera ray's own space).
* ``kp3d`` — Huber on pelvis-relative camera-frame joints (metres).
* ``orient`` / ``pose`` / ``hand_pose`` — MSE on the raw 6D outputs vs the GT's
  first two rotation-matrix columns (GVHMR / WHAM practice); GT rotations
  arrive as matrices, never as the stored axis-angles (those sit off the
  principal branch). ``hand_pose`` covers the 30 finger joints of a
  ``model.smplx.hands`` head.
* ``betas`` — MSE on the 10 shape coefficients (per-person GT served per frame).

The keypoint terms run over every joint the head emits (22, or 52 with hands)
as a WEIGHTED mean: body joints at 1, finger joints at
``joint_weights.fingers`` (their GT is the least reliable part of the fit and
their lever arm on the body is negligible).
* ``cam`` — Huber on the CLIFF ``(s, tx, ty)`` proxy vs the GT pelvis inverted
  into the same proxy — translation is supervised in proxy space, as every
  surveyed method does, never in metres. ``camera: cliff`` heads only.
* ``pelvis`` — Huber on the absolute camera-frame pelvis (metres).
* ``depth`` / ``bearing`` — Huber on the pelvis ray ``(x/z, y/z, log z)`` of
  the LIFTED pelvis (any camera parametrization): the log depth and the
  bearing separately.
* ``depth_vel`` / ``depth_acc`` / ``bearing_vel`` / ``bearing_acc`` — Huber on
  the first (forward) and second (central) differences of that ray over each
  clip's real elapsed seconds vs the same differences of the GT ray. The GT
  depth is temporally clean (0.25 %/frame incl. real motion) while the token's
  depth noise is white (0.55 %/frame), so matching derivatives asks the
  temporal block to average the measurement across frames — the objective the
  CLIFF proxy ``s`` (95 % crop jitter) could never express.

Metrics (eval only) follow WHAM's ``evaluate_3dpw.py`` / GVHMR's
``compute_camcoord_metrics`` line by line, on the 22 SMPL-X body joints and
the 10475 vertices with flat hands:

* every frame is aligned by the MEAN OF THE TWO HIP JOINTS (their
  ``pelvis_idxs = [1, 2]``); the vertices are shifted by that same joint-derived
  pelvis;
* ``mpjpe`` / ``pa_mpjpe`` / ``pve`` (mm) — per-frame mean L2 over joints /
  Procrustes-aligned joints / vertices;
* ``dlogz_pred`` / ``dlogz_gt`` / ``dlogz_err`` (%/frame) — RMS frame-to-frame
  step of the pelvis log depth: the prediction's, the GT's, and that of their
  difference (the noise alone, real motion removed). The relative-depth jitter
  the cubed world jitter hides behind the far-person clips.
* ``accel`` (m/s^2) — per-frame mean over joints of the L2 error of the second
  finite difference of the aligned joints. The papers divide by ``(1/30 s)^2``
  (their footage is 30 fps); here the division uses each clip's REAL frame
  spacing, so the number is theirs on 30 fps footage and fps-exact on the
  corpus's 24-60 fps scenes. GVHMR's trim (every interior frame; WHAM also
  drops the first and last) is used.
* ``hand_mpjpe`` / ``hand_pa_mpjpe`` (mm, hands heads only) — per-frame mean
  over the 30 finger joints of the L2 error after aligning each hand on its
  own wrist (translation only: still charged for the wrist's orientation) and
  after a per-hand Procrustes alignment of wrist + 15 fingers (pure finger
  articulation). The body metrics always use the 22 body joints and the
  FLAT-hand vertices (the 22-joint body at the body part of ``q``), so they
  stay comparable across hands / no-hands runs and the frozen baseline.

The reduction is frame-weighted (``np.concatenate(all frames).mean()`` in both
repos), which is exactly what the additive ``(sum, count)`` statistics give.
"""
from __future__ import annotations

import math

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
    translation_to_ray,
)
from utils.metrics import mean_from_stats

#: Ray derivative terms: ``<part>_<order>`` over the pelvis ray channels.
_RAY_DIFF_TERMS = ("depth_vel", "depth_acc", "bearing_vel", "bearing_acc")
_TERM_NAMES = ("kp2d", "kp3d", "orient", "pose", "hand_pose", "betas", "cam", "pelvis",
               "depth", "bearing") + _RAY_DIFF_TERMS
#: Reported body metrics, in the order of the statistics vector (the ``dlogz_*``
#: statistics are sums of SQUARES; :func:`pose_metrics_from_stats` takes the root).
POSE_METRICS = ("mpjpe", "pa_mpjpe", "pve", "accel", "pelvis_err", "depth_err", "depth_bias",
                "dlogz_pred", "dlogz_gt", "dlogz_err")
#: Appended for a hands head (wrist-aligned and per-hand Procrustes-aligned finger error).
HAND_METRICS = ("hand_mpjpe", "hand_pa_mpjpe")
#: Minimum camera-frame depth (metres) for a projectable GT row.
_MIN_DEPTH_M = 0.25
#: The two hip joints whose mean is the alignment pelvis (WHAM/GVHMR
#: ``pelvis_idxs`` for the SMPL 24 / SMPL-X 22 body joint set).
SMPLX_HIPS = (1, 2)
NUM_BODY_JOINTS = 22
NUM_HAND_JOINTS = 30
#: Wrist joint of each hand's 15 finger joints (left wrist 20, right wrist 21).
HAND_WRISTS = (20, 21)
#: Width of the body part of ``q`` (pelvis, root quat, 21 joint quats).
BODY_Q_DIM = 91


def gt_smplx_camera(batch: dict, device, dtype=torch.float32, hands: bool = False) -> dict:
    """The kindyn SMPL-X GT of a batch, lifted into each frame's camera.

    :param hands: assemble ``q`` with the finger quaternions (a 52-joint body).
    :returns: ``joints (B, 52, 3)`` metres (22 body joints first), ``root_rot
        (B, 3, 3)`` camera-from-root, ``body_rot (B, 21, 3, 3)`` and
        ``hand_rot (B, 30, 3, 3)`` parent-local, ``betas (B, 10)``, ``q (B, 91 |
        211)`` the BetterHuman camera-frame configuration, ``valid (B,)`` bool
        (labelled, tracked, and in front of the camera).
    """
    ext = batch["cam_from_world"].to(device, dtype)                      # (B, 4, 4)
    rot_cw, t_cw = ext[:, :3, :3], ext[:, :3, 3]
    joints_world = batch["smplx_joints_world"].to(device, dtype)
    joints = torch.einsum("bij,bkj->bki", rot_cw, joints_world) + t_cw[:, None]
    root_rot = rot_cw @ batch["smplx_root_rot"].to(device, dtype)
    body_rot = batch["smplx_body_rot"].to(device, dtype)
    hand_rot = batch["smplx_hand_rot"].to(device, dtype)
    valid = (batch["smplx_valid"] & batch["frame_valid"]).to(device)
    valid = valid & (joints[..., 2] > _MIN_DEPTH_M).all(dim=-1)
    return {
        "joints": joints, "root_rot": root_rot, "body_rot": body_rot,
        "hand_rot": hand_rot, "betas": batch["smplx_betas"].to(device, dtype),
        "q": smplx_q(joints[:, 0], root_rot, body_rot, hand_rot if hands else None),
        "valid": valid,
    }


def smplx_q(pelvis: Tensor, root_rot: Tensor, body_rot: Tensor,
            hand_rot: Tensor | None = None) -> Tensor:
    """Assemble the BetterHuman configuration ``(B, 91)`` (``(B, 211)`` with hands).

    :param pelvis: pelvis position ``(B, 3)`` (the root of ``q`` IS the pelvis).
    :param root_rot: root rotation ``(B, 3, 3)``.
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


def smplx_vertices(body, betas: Tensor, q: Tensor) -> Tensor:
    """Skinned vertices ``(B, 10475, 3)`` of a BetterHuman SMPL-X body at ``q``."""
    shaped = body.with_shape(betas=betas)
    return body.vertices_from_data(shaped.fk(q))


@torch.no_grad()
def pose_metric_stats(
    pred_joints: Tensor, pred_verts: Tensor, gt_joints: Tensor, gt_verts: Tensor,
    valid: Tensor, seq_len: int, frame_pos_sec: Tensor,
) -> Tensor:
    """Additive ``(sum, count)`` pairs of :data:`POSE_METRICS` — float64 ``[20]``
    (``[24]`` with :data:`HAND_METRICS` when ``pred_joints`` carries the fingers).

    :param pred_joints: ``(B, 22 | 52, 3)`` camera metres, ``B = n_clips *
        seq_len`` clip-major; the body metrics use the first 22 rows.
    :param pred_verts: ``(B, V, 3)`` flat-hand vertices.
    :param gt_joints: ``(B, 22 | 52, 3)`` (52 required for the hand metric).
    :param gt_verts: ``(B, V, 3)`` flat-hand vertices.
    :param valid: ``(B,)`` bool rows that count.
    :param seq_len: frames per clip (the acceleration stencil never crosses a clip).
    :param frame_pos_sec: ``(B,)`` elapsed seconds per frame.
    """
    hands = pred_joints.shape[1] == NUM_BODY_JOINTS + NUM_HAND_JOINTS
    hand_stats: list[float] = []
    if hands:
        hand_stats = _hand_metric_stats(pred_joints, gt_joints, valid)
    pred_joints, gt_joints = pred_joints[:, :NUM_BODY_JOINTS], gt_joints[:, :NUM_BODY_JOINTS]
    # Absolute camera-frame pelvis (joint 0) error — the quantity the pelvis-aligned
    # metrics below cannot see (a constant depth offset is invisible to all of them).
    abs_err = (pred_joints[:, 0] - gt_joints[:, 0]) * 1000.0                 # (B, 3) mm
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

    # Relative-depth jitter: sum of squared frame-to-frame steps of log z (%/frame).
    dlogz = [0.0] * 6
    if seq_len >= 2:
        n_clips = pred_joints.shape[0] // seq_len
        lz_pred = torch.log(pred_joints[:, 0, 2].clamp(min=1e-3)).reshape(n_clips, seq_len)
        lz_gt = torch.log(gt_joints[:, 0, 2].clamp(min=1e-3)).reshape(n_clips, seq_len)
        v = valid.reshape(n_clips, seq_len)
        pair = (v[:, 1:] & v[:, :-1]).to(lz_pred.dtype)
        for i, series in enumerate((lz_pred, lz_gt, lz_pred - lz_gt)):
            step = (series[:, 1:] - series[:, :-1]) * 100.0
            dlogz[2 * i] = float((step.square() * pair).sum())
            dlogz[2 * i + 1] = float(pair.sum())

    return torch.tensor([
        float((mpjpe * mask).sum()), count,
        float((pa_mpjpe * mask).sum()), count,
        float((pve * mask).sum()), count,
        accel_sum, accel_count,
        float((abs_err.norm(dim=-1) * mask).sum()), count,
        float((abs_err[:, 2].abs() * mask).sum()), count,
        float((abs_err[:, 2] * mask).sum()), count,
    ] + dlogz + hand_stats, dtype=torch.float64)


def _hand_metric_stats(pred_joints: Tensor, gt_joints: Tensor, valid: Tensor) -> list[float]:
    """``(sum, count)`` pairs of :data:`HAND_METRICS` (mm) over valid rows."""
    per_hand = NUM_HAND_JOINTS // 2
    errors, pa_errors = [], []
    for hand, wrist in enumerate(HAND_WRISTS):
        lo = NUM_BODY_JOINTS + hand * per_hand
        fingers = slice(lo, lo + per_hand)
        pred = pred_joints[:, fingers] - pred_joints[:, wrist:wrist + 1]
        gt = gt_joints[:, fingers] - gt_joints[:, wrist:wrist + 1]
        errors.append((pred - gt).norm(dim=-1))                          # (B, 15)
        # Procrustes over wrist + fingers; the wrist row is dropped from the error.
        origin = torch.zeros_like(pred[:, :1])
        aligned = procrustes_align(torch.cat([origin, pred], dim=1),
                                   torch.cat([origin, gt], dim=1))[:, 1:]
        pa_errors.append((aligned - gt).norm(dim=-1))
    mask = valid.to(pred_joints.dtype)
    err = torch.cat(errors, dim=1).mean(dim=-1) * 1000.0                 # (B,)
    pa_err = torch.cat(pa_errors, dim=1).mean(dim=-1) * 1000.0
    return [float((err * mask).sum()), float(mask.sum()),
            float((pa_err * mask).sum()), float(mask.sum())]


def pose_metrics_from_stats(stats: Tensor) -> dict[str, float]:
    """:data:`POSE_METRICS` (+ :data:`HAND_METRICS`) from the summed statistics vector."""
    names = POSE_METRICS + (HAND_METRICS if len(stats) > 2 * len(POSE_METRICS) else ())
    out = {}
    for i, name in enumerate(names):
        value = mean_from_stats(float(stats[2 * i]), float(stats[2 * i + 1]))
        out[name] = math.sqrt(value) if name.startswith("dlogz") and value == value else value
    return out


class SmplxLoss(Loss):
    """Huber keypoint / proxy-camera terms and 6D / betas MSE for the SMPL-X head."""

    name = "smplx"
    metric_group = "pose"

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        super().__init__(cfg, model, device)
        section = cfg["smplx_supervision"]
        loss_cfg = section["loss"]
        self.hands = bool(self.model.head_smplx.hands)
        self.weights = {name: float(loss_cfg[name]) for name in _TERM_NAMES}
        if self.weights["hand_pose"] > 0.0 and not self.hands:
            raise ValueError("smplx_supervision.loss.hand_pose needs model.smplx.hands")
        self.term_names = tuple(n for n in _TERM_NAMES if self.weights[n] > 0.0)
        if not self.term_names:
            raise ValueError(
                "smplx_supervision: every loss weight is 0 — disable the section instead")
        self.delta_2d = float(loss_cfg["huber_delta_2d"])
        self.delta_3d = float(loss_cfg["huber_delta_3d"])
        self.delta_cam = float(loss_cfg["huber_delta_cam"])
        self.delta_pelvis = float(loss_cfg["huber_delta_pelvis"])
        self.delta_depth = float(loss_cfg["huber_delta_depth"])
        self.delta_bearing = float(loss_cfg["huber_delta_bearing"])
        self.delta_diff = {name: float(loss_cfg[f"huber_delta_{name}"]) for name in _RAY_DIFF_TERMS}
        self.kp2d_space = str(section["kp2d_space"])
        if self.kp2d_space not in ("crop", "image"):
            raise ValueError(f"smplx_supervision.kp2d_space must be crop | image; got {self.kp2d_space!r}")
        if self.weights["cam"] > 0.0 and self.model.head_smplx.camera != "cliff":
            raise ValueError("smplx_supervision.loss.cam needs model.smplx.camera: cliff")
        # Per-joint keypoint weights over the joints the head emits.
        num_joints = self.model.head_smplx.num_joints
        joint_w = torch.ones(num_joints, dtype=self.dtype)
        joint_w[NUM_BODY_JOINTS:] = float(section["joint_weights"]["fingers"])
        self.joint_w = (joint_w / joint_w.sum()).to(self.device)         # sums to 1
        metric_names = POSE_METRICS + (HAND_METRICS if self.hands else ())
        self.stat_names = tuple(f"{key}_{part}" for key in metric_names
                                for part in ("sum", "count"))

    def __call__(self, out: dict, batch: dict, *, train: bool) -> LossResult:
        pred = out["smplx"]
        root_6d = pred["root_6d"].to(self.device, self.dtype)            # (B,6)
        body_6d = pred["body_6d"].to(self.device, self.dtype)            # (B,21,6)
        betas = pred["betas"].to(self.device, self.dtype)                # (B,10)
        pelvis_cam = pred["pelvis_cam"].to(self.device, self.dtype)      # (B,3)
        joints = pred["joints_cam"].to(self.device, self.dtype)          # (B,J,3)
        kp2d_crop = pred["kp2d_crop"].to(self.device, self.dtype)        # (B,J,2)
        kp2d_full = pred["kp2d_full"].to(self.device, self.dtype)        # (B,J,2) px
        anchor = (root_6d.sum() + body_6d.sum() + betas.sum() + pelvis_cam.sum()) * 0.0
        cam = None
        if self.weights["cam"] > 0.0:
            cam = pred["cam"].to(self.device, self.dtype)                # (B,3)
            anchor = anchor + cam.sum() * 0.0
        hand_6d = None
        if self.hands:
            hand_6d = pred["hand_6d"].to(self.device, self.dtype)        # (B,30,6)
            anchor = anchor + hand_6d.sum() * 0.0

        gt = gt_smplx_camera(batch, self.device, self.dtype, hands=self.hands)
        gt_joints = gt["joints"][:, :joints.shape[1]]
        mask = gt["valid"].to(self.dtype)                                # (B,)
        mass = float(mask.sum())

        cam_int = batch["cam_int"].to(self.device, self.dtype)
        affine = batch["affine_trans"].to(self.device, self.dtype)
        img_size = batch["img_size"].to(self.device, self.dtype)
        bbox_center = batch["bbox_center"].to(self.device, self.dtype)
        bbox_size = batch["bbox_scale"].to(self.device, self.dtype)[:, 0]
        gt_full, gt_crop = project_to_crop(gt_joints, cam_int, affine, img_size)
        gt_root_6d = rotmat_to_rot6d(gt["root_rot"])
        gt_body_6d = rotmat_to_rot6d(gt["body_rot"])
        ray_pred = translation_to_ray(pelvis_cam)
        ray_gt = translation_to_ray(gt_joints[:, 0])

        raw: dict[str, tuple[Tensor, float]] = {}
        if self.weights["kp2d"] > 0.0:
            if self.kp2d_space == "crop":
                huber = F.smooth_l1_loss(kp2d_crop, gt_crop, reduction="none", beta=self.delta_2d)
            else:
                focal = cam_int[:, 0, 0, None, None]
                huber = F.smooth_l1_loss(kp2d_full / focal, gt_full / focal, reduction="none",
                                         beta=self.delta_2d)
            raw["kp2d"] = (((huber.mean(dim=-1) * self.joint_w).sum(dim=1) * mask).sum(), mass)
        if self.weights["kp3d"] > 0.0:
            huber = F.smooth_l1_loss(joints - joints[:, :1], gt_joints - gt_joints[:, :1],
                                     reduction="none", beta=self.delta_3d)
            raw["kp3d"] = (((huber.mean(dim=-1) * self.joint_w).sum(dim=1) * mask).sum(), mass)
        if self.weights["orient"] > 0.0:
            raw["orient"] = (((root_6d - gt_root_6d).square().mean(dim=-1) * mask).sum(), mass)
        if self.weights["pose"] > 0.0:
            raw["pose"] = (((body_6d - gt_body_6d).square().mean(dim=(1, 2)) * mask).sum(), mass)
        if self.weights["hand_pose"] > 0.0:
            gt_hand_6d = rotmat_to_rot6d(gt["hand_rot"])
            raw["hand_pose"] = (
                ((hand_6d - gt_hand_6d).square().mean(dim=(1, 2)) * mask).sum(), mass)
        if self.weights["betas"] > 0.0:
            raw["betas"] = (((betas - gt["betas"]).square().mean(dim=-1) * mask).sum(), mass)
        if self.weights["cam"] > 0.0:
            gt_cam = translation_to_cliff_cam(gt_joints[:, 0], bbox_center, bbox_size, cam_int)
            huber = F.smooth_l1_loss(cam, gt_cam, reduction="none", beta=self.delta_cam)
            raw["cam"] = ((huber.mean(dim=-1) * mask).sum(), mass)
        if self.weights["depth"] > 0.0:
            huber = F.smooth_l1_loss(ray_pred[:, 2], ray_gt[:, 2], reduction="none",
                                     beta=self.delta_depth)
            raw["depth"] = ((huber * mask).sum(), mass)
        if self.weights["bearing"] > 0.0:
            huber = F.smooth_l1_loss(ray_pred[:, :2], ray_gt[:, :2], reduction="none",
                                     beta=self.delta_bearing)
            raw["bearing"] = ((huber.mean(dim=-1) * mask).sum(), mass)
        if any(self.weights[name] > 0.0 for name in _RAY_DIFF_TERMS):
            raw.update(self._ray_diff_terms(
                ray_pred, ray_gt, gt["valid"], int(batch["seq_len"]),
                batch["frame_pos_sec"].to(self.device, self.dtype)))
        if self.weights["pelvis"] > 0.0:
            # Absolute camera-frame pelvis in metres: the metric depth anchor the
            # projective / pelvis-relative terms lack (and the DC term a derivative
            # loss such as motion_matching cannot supply).
            huber = F.smooth_l1_loss(joints[:, 0], gt_joints[:, 0], reduction="none",
                                     beta=self.delta_pelvis)
            raw["pelvis"] = ((huber.sum(dim=-1) * mask).sum(), mass)

        stats = self.empty_stats()
        if not train:
            # Vertices only at evaluation: the metrics are their sole consumer.
            # Flat hands on both sides (the 22-joint body at the body part of q)
            # keep PVE comparable across hands / no-hands runs.
            body = self.model.head_smplx.body_flat(self.device)
            pred_verts = smplx_vertices(
                body, betas.detach(), pred["q_cam"].detach()[:, :BODY_Q_DIM])
            gt_verts = smplx_vertices(body, gt["betas"], gt["q"][:, :BODY_Q_DIM])
            stats = pose_metric_stats(
                joints.detach(), pred_verts, gt_joints, gt_verts, gt["valid"],
                int(batch["seq_len"]), batch["frame_pos_sec"].to(self.device),
            ).to(self.device)
        weighted = {name: (self.weights[name] * numerator, term_mass)
                    for name, (numerator, term_mass) in raw.items()}
        return LossResult(terms=self._terms(weighted, anchor), scalars={"n_rows": mass},
                          stats=stats)

    def _ray_diff_terms(self, ray_pred: Tensor, ray_gt: Tensor, valid: Tensor,
                        seq_len: int, seconds: Tensor) -> dict[str, tuple[Tensor, float]]:
        """Velocity / acceleration Huber terms of the pelvis ray over each clip's real seconds.

        Forward difference for the velocity (rows ``t, t+1`` valid), central
        second difference for the acceleration (rows ``t-1, t, t+1``); stencils
        never cross a clip. A clip too short for a stencil leaves the term at
        mass 0 (graph-connected zero).
        """
        zero = ray_pred.sum() * 0.0
        out = {name: (zero, 0.0) for name in _RAY_DIFF_TERMS if self.weights[name] > 0.0}
        n_clips = ray_pred.shape[0] // seq_len
        p = ray_pred.reshape(n_clips, seq_len, 3)
        g = ray_gt.reshape(n_clips, seq_len, 3)
        v = valid.reshape(n_clips, seq_len)
        t = seconds.reshape(n_clips, seq_len)
        stencils = {}
        if seq_len >= 2:
            dt = (t[:, 1:] - t[:, :-1]).clamp(min=1e-6)[..., None]
            stencils["vel"] = ((p[:, 1:] - p[:, :-1]) / dt, (g[:, 1:] - g[:, :-1]) / dt,
                               v[:, 1:] & v[:, :-1])
        if seq_len >= 3:
            dt2 = (0.5 * (t[:, 2:] - t[:, :-2])).clamp(min=1e-6).square()[..., None]
            stencils["acc"] = ((p[:, 2:] - 2.0 * p[:, 1:-1] + p[:, :-2]) / dt2,
                               (g[:, 2:] - 2.0 * g[:, 1:-1] + g[:, :-2]) / dt2,
                               v[:, 2:] & v[:, 1:-1] & v[:, :-2])
        for order, (dp, dg, rows) in stencils.items():
            rows = rows.to(self.dtype)
            row_mass = float(rows.sum())
            for part, channels in (("depth", slice(2, 3)), ("bearing", slice(0, 2))):
                name = f"{part}_{order}"
                if name not in out:
                    continue
                huber = F.smooth_l1_loss(dp[..., channels], dg[..., channels], reduction="none",
                                         beta=self.delta_diff[name])
                out[name] = ((huber.mean(dim=-1) * rows).sum(), row_mass)
        return out

    def metrics(self, stats: Tensor) -> dict[str, float]:
        return pose_metrics_from_stats(stats)


__all__ = ["SmplxLoss", "POSE_METRICS", "HAND_METRICS", "SMPLX_HIPS", "BODY_Q_DIM",
           "gt_smplx_camera", "smplx_q", "smplx_vertices", "pose_metric_stats",
           "pose_metrics_from_stats"]
