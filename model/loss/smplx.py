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
* ``cam`` — Huber on the CLIFF ``(s, tx, ty)`` proxy vs the GT pelvis inverted
  into the same proxy (``camera: cliff`` heads only).
* ``root_bias`` / ``root_shape`` — the WORLD root error of every clip split
  into its mean over the clip's supervised rows (a Huber on the norm with a
  wide knee: the absolute anchor) and the per-frame deviation from that mean
  (a Huber on the norm with a small knee: the trajectory shape, which a
  per-frame Huber past its knee cannot see behind a 10 cm bias). Both are
  counted per frame, so a clip weighs by its rows; single-frame clips make the
  shape term zero and the bias term the plain per-frame error.
* ``depth`` / ``bearing`` — Huber on the pelvis ray ``(x/z, y/z, log z)`` of
  the LIFTED pelvis (any camera parametrization): the log depth and the
  bearing separately — the crop-free absolute anchors a ``ray`` head needs.

The keypoint terms run over every joint the head emits (22, or 52 with hands)
as a WEIGHTED mean: body joints at 1, finger joints at
``joint_weights.fingers``.

Metrics (eval only) follow WHAM's ``evaluate_3dpw.py`` / GVHMR's
``compute_camcoord_metrics`` line by line, on the 22 SMPL-X body joints and
the 10475 vertices with flat hands:

* every frame is aligned by the MEAN OF THE TWO HIP JOINTS (their
  ``pelvis_idxs = [1, 2]``); the vertices are shifted by that same joint-derived
  pelvis;
* ``mpjpe`` / ``pa_mpjpe`` / ``pve`` (mm) — per-frame mean L2 over joints /
  Procrustes-aligned joints / vertices;
* ``pelvis_err`` / ``depth_err`` / ``depth_bias`` (mm) — the absolute
  camera-frame pelvis error, its depth component, and the signed depth error;
* ``dlogz_pred`` / ``dlogz_gt`` / ``dlogz_err`` (%/frame) — RMS frame-to-frame
  step of the pelvis log depth: the prediction's, the GT's, and that of their
  difference (the noise alone, real motion removed);
* ``accel`` (m/s^2) — per-frame mean over joints of the L2 error of the second
  finite difference of the aligned joints, at each clip's REAL frame spacing;
* ``lifted_*`` — GVHMR's global metrics of the per-frame body LIFTED to the
  world with the GT extrinsics of every frame (:mod:`utils.gvhmr_metrics`):
  ``wa_mpjpe100`` / ``w_mpjpe100`` (mm, per-100-frame-chunk alignment),
  ``rte`` (root translation error, % of the GT path) and ``jitter`` (10 m/s^3),
  with ``gt_jitter`` as the floor. Invalid frames are dropped by GVHMR's mask
  compaction before chunking;
* ``hand_mpjpe`` / ``hand_pa_mpjpe`` (mm, hands heads only) — per-frame mean
  over the 30 finger joints after aligning each hand on its own wrist, and
  after a per-hand Procrustes alignment of wrist + 15 fingers.

The reduction is frame-weighted (``np.concatenate(all frames).mean()`` in both
repos), which is exactly what the additive ``(sum, count)`` statistics give.

A refined body (the world keys of :class:`~model.refiner.TemporalRefiner`)
carries a second, research-only group on the SAME clips and validity as
``lifted_jitter`` (:func:`world_diag_stats`), with the world body read as
``J = p + R X`` (world pelvis, world-from-root rotation, root-frame joints):

* ``jitter_root_only`` / ``jitter_rot_only`` / ``jitter_artic_only`` (10 m/s^3)
  — the oracle swap decomposition of the lifted jitter: exactly one of
  ``p`` / ``R`` / ``X`` taken from the prediction, the other two from the GT,
  so each says how much of the jerk that one channel alone would produce
  (``gt_jitter`` is their common floor);
* ``vel_ratio_root`` / ``vel_ratio_rot`` / ``vel_ratio_joints`` — the
  shrinkage detector: the predicted over the GT RMS of the FORWARD difference
  of the world pelvis, of the root rotation (``log(R_t^T R_t+1) / h``) and of
  the root-frame joints. 1 is an unshrunk trajectory, below 1 an amplitude a
  pointwise derivative loss has traded against its noise;
* ``root_err_hf`` / ``root_err_lf`` (mm) — the world pelvis error split by a
  Gaussian at ``_ROOT_SPLIT_SEC`` into what a temporal filter could still
  remove (high pass) and the slow error / bias it cannot (low pass).
"""
from __future__ import annotations

import math

import numpy as np
import roma
import torch
import torch.nn.functional as F
from torch import Tensor

from model.loss import Loss, LossResult
from model.refiner import gaussian_smooth
from utils.geometry import (
    lift_to_world,
    procrustes_align,
    project_to_crop,
    rotmat_to_rot6d,
    smplx_q,
    translation_to_cliff_cam,
    translation_to_ray,
)
from utils.gvhmr_metrics import compute_jitter, global_metrics
from utils.metrics import mean_from_stats

_TERM_NAMES = ("kp2d", "kp3d", "orient", "pose", "hand_pose", "betas", "cam", "root_bias",
               "root_shape", "depth", "bearing")
#: The terms that read nothing but ONE world trajectory's body, so deep supervision
#: (``layer_weight``) can repeat them on the refiner's intermediate layers.
LAYER_TERM_NAMES = ("kp3d", "orient", "pose", "root_bias", "root_shape")
#: Camera-frame body metrics, in the order of the statistics vector (the ``dlogz_*``
#: statistics are sums of SQUARES; :func:`pose_metrics_from_stats` takes the root).
POSE_METRICS = ("mpjpe", "pa_mpjpe", "pve", "accel", "pelvis_err", "depth_err", "depth_bias",
                "dlogz_pred", "dlogz_gt", "dlogz_err")
#: GVHMR global metrics of the camera-lifted world trajectory.
LIFTED_METRICS = ("lifted_wa_mpjpe100", "lifted_w_mpjpe100", "lifted_rte", "lifted_jitter",
                  "gt_jitter")
#: Appended for a hands head (wrist-aligned and per-hand Procrustes-aligned finger error).
HAND_METRICS = ("hand_mpjpe", "hand_pa_mpjpe")
#: Appended for a refined (world) body: the jitter swap decomposition, the velocity
#: amplitude ratios and the high / low pass split of the world root error.
DIAG_METRICS = ("jitter_root_only", "jitter_rot_only", "jitter_artic_only",
                "vel_ratio_root", "vel_ratio_rot", "vel_ratio_joints",
                "root_err_hf", "root_err_lf")
#: Metrics whose statistics are sums of SQUARES — the reported number is the root
#: (for ``vel_ratio_*`` the pair is ``(sum of squares predicted, sum of squares GT)``).
_ROOT_METRICS = frozenset(("dlogz_pred", "dlogz_gt", "dlogz_err", "vel_ratio_root",
                           "vel_ratio_rot", "vel_ratio_joints", "root_err_hf", "root_err_lf"))
#: Gaussian width (seconds) splitting the world root error into high and low pass.
_ROOT_SPLIT_SEC = 0.2
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


def metric_names(hands: bool, world: bool = False) -> tuple[str, ...]:
    """The reported pose metrics, in statistics order.

    :param hands: the head emits the 30 finger joints.
    :param world: the body is refined, so the world diagnostics are reported.
    """
    return (POSE_METRICS + LIFTED_METRICS + (HAND_METRICS if hands else ())
            + (DIAG_METRICS if world else ()))


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


def smplx_vertices(body, betas: Tensor, q: Tensor) -> Tensor:
    """Skinned vertices ``(B, 10475, 3)`` of a BetterHuman SMPL-X body at ``q``."""
    shaped = body.with_shape(betas=betas)
    return body.vertices_from_data(shaped.fk(q))


@torch.no_grad()
def eval_stats(
    pred_joints: Tensor, pred_verts: Tensor, gt_joints: Tensor, gt_verts: Tensor,
    valid: Tensor, batch: dict, world: dict | None = None,
) -> Tensor:
    """Additive ``(sum, count)`` pairs of :func:`metric_names` — float64.

    :param pred_joints: ``(B, 22 | 52, 3)`` camera metres, ``B = n_clips *
        seq_len`` clip-major; the body metrics use the first 22 rows and the
        hand metrics are appended when the fingers are present.
    :param pred_verts: ``(B, V, 3)`` flat-hand vertices.
    :param gt_joints: ``(B, 22 | 52, 3)`` (52 required for the hand metric).
    :param gt_verts: ``(B, V, 3)`` flat-hand vertices.
    :param valid: ``(B,)`` bool rows that count.
    :param batch: the collated batch (``seq_len``, ``frame_pos_sec``,
        ``cam_from_world``, ``smplx_joints_world``).
    :param world: the refined body's world keys (``pelvis_world``,
        ``root_rot_world``, ``joints_world``); ``None`` skips
        :data:`DIAG_METRICS`.
    """
    seq_len = int(batch["seq_len"])
    seconds = batch["frame_pos_sec"].to(pred_joints.device)
    stats = [pose_metric_stats(pred_joints, pred_verts, gt_joints, gt_verts, valid,
                               seq_len, seconds),
             lifted_metric_stats(pred_joints, batch, valid, seq_len)]
    if pred_joints.shape[1] == NUM_BODY_JOINTS + NUM_HAND_JOINTS:
        stats.append(_hand_metric_stats(pred_joints, gt_joints, valid))
    if world is not None:
        stats.append(world_diag_stats(world, batch, valid, seq_len))
    return torch.cat([s.to(torch.float64).cpu() for s in stats])


@torch.no_grad()
def pose_metric_stats(
    pred_joints: Tensor, pred_verts: Tensor, gt_joints: Tensor, gt_verts: Tensor,
    valid: Tensor, seq_len: int, frame_pos_sec: Tensor,
) -> Tensor:
    """Additive ``(sum, count)`` pairs of :data:`POSE_METRICS` — float64 ``[20]``."""
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
    ] + dlogz, dtype=torch.float64)


@torch.no_grad()
def lifted_metric_stats(pred_joints: Tensor, batch: dict, valid: Tensor, seq_len: int) -> Tensor:
    """Additive ``(sum, count)`` pairs of :data:`LIFTED_METRICS` — float64 ``[10]``.

    The camera-frame body is lifted with the GT extrinsics of EVERY frame
    (trusting the camera, not the network) and scored per clip against the
    kindyn world joints; a clip with fewer than two valid rows is skipped.
    """
    device = pred_joints.device
    joints = pred_joints[:, :NUM_BODY_JOINTS].to(torch.float64)
    ext = batch["cam_from_world"].to(device, torch.float64)
    seconds = batch["frame_pos_sec"].to(device, torch.float64)
    gt_world = batch["smplx_joints_world"][:, :NUM_BODY_JOINTS].to(device, torch.float64)
    lifted = lift_to_world(joints, ext).float().cpu()
    gt_world, valid = gt_world.float().cpu(), valid.cpu()
    metrics = ("wa_mpjpe100", "w_mpjpe100", "rte", "jitter")
    stats = torch.zeros(2 * len(LIFTED_METRICS), dtype=torch.float64)
    for clip in range(joints.shape[0] // seq_len):
        rows = slice(clip * seq_len, (clip + 1) * seq_len)
        mask = valid[rows]
        if int(mask.sum()) < 2:
            continue
        dt = seconds[rows][1:] - seconds[rows][:-1]
        fps = float(1.0 / dt.median()) if seq_len > 1 else 1.0
        gt = gt_world[rows][mask]
        result = global_metrics(lifted[rows][mask], gt, fps)
        for m, metric in enumerate(metrics):
            values = np.asarray(result[metric], np.float64)
            stats[2 * m] += float(values.sum())
            stats[2 * m + 1] += float(len(values))
        gt_jitter = np.asarray(compute_jitter(gt, fps=fps), np.float64)
        stats[-2] += float(gt_jitter.sum())
        stats[-1] += float(len(gt_jitter))
    return stats


def _forward_rate(x: Tensor, step: Tensor) -> Tensor:
    """Forward difference ``(x[t+1] - x[t]) / h`` of a ``(T, ...)`` series."""
    return (x[1:] - x[:-1]) / step.reshape(-1, *([1] * (x.dim() - 1)))


@torch.no_grad()
def world_diag_stats(world: dict, batch: dict, valid: Tensor, seq_len: int) -> Tensor:
    """Additive pairs of :data:`DIAG_METRICS` of a refined body — float64 ``[16]``.

    Same clips and validity as :func:`lifted_metric_stats`: one pass per clip
    over the rows GVHMR's mask compaction keeps, at the clip's sampled fps, a
    clip with fewer than two valid rows skipped. The world body is read as
    ``J = p + R X`` on both sides (the prediction's own world keys, the kindyn
    world joints and root rotation), so the three jitter swaps and the three
    velocity ratios are directly comparable to ``lifted_jitter`` / ``gt_jitter``.

    The velocity ratios accumulate the two sums of squares (predicted,
    GT) as the pair, so the split-level ratio is the root of their quotient;
    the root error pairs are ``(sum of squared mm, valid rows)``.
    """
    cpu = torch.device("cpu")
    p_pred = world["pelvis_world"].detach().to(cpu, torch.float64)                # (B, 3)
    r_pred = world["root_rot_world"].detach().to(cpu, torch.float64)              # (B, 3, 3)
    j_pred = world["joints_world"][:, :NUM_BODY_JOINTS].detach().to(cpu, torch.float64)
    j_gt = batch["smplx_joints_world"][:, :NUM_BODY_JOINTS].to(cpu, torch.float64)
    r_gt = batch["smplx_root_rot"].to(cpu, torch.float64)
    p_gt = j_gt[:, 0]
    x_pred = torch.einsum("bji,bkj->bki", r_pred, j_pred - p_pred[:, None])
    x_gt = torch.einsum("bji,bkj->bki", r_gt, j_gt - p_gt[:, None])
    seconds = batch["frame_pos_sec"].to(cpu, torch.float64)
    valid = valid.cpu()

    stats = torch.zeros(2 * len(DIAG_METRICS), dtype=torch.float64)
    for clip in range(p_pred.shape[0] // seq_len):
        rows = slice(clip * seq_len, (clip + 1) * seq_len)
        mask = valid[rows]
        if int(mask.sum()) < 2:
            continue
        dt = seconds[rows][1:] - seconds[rows][:-1]
        fps = float(1.0 / dt.median()) if seq_len > 1 else 1.0
        pp, rp, xp = p_pred[rows][mask], r_pred[rows][mask], x_pred[rows][mask]
        pg, rg, xg = p_gt[rows][mask], r_gt[rows][mask], x_gt[rows][mask]

        # Swap decomposition: one channel predicted, the other two from the GT.
        for m, (p, r, x) in enumerate(((pp, rg, xg), (pg, rp, xg), (pg, rg, xp))):
            joints = (p[:, None] + torch.einsum("bij,bkj->bki", r, x)).float()
            jitter = np.asarray(compute_jitter(joints, fps=fps), np.float64)
            stats[2 * m] += float(jitter.sum())
            stats[2 * m + 1] += float(len(jitter))

        # Velocity amplitudes: forward differences at the real spacing of the kept rows.
        step = (seconds[rows][mask][1:] - seconds[rows][mask][:-1]).clamp(min=1e-6)
        rates = (
            (_forward_rate(pp, step), _forward_rate(pg, step)),
            (roma.rotmat_to_rotvec(rp[:-1].transpose(-1, -2) @ rp[1:]) / step[:, None],
             roma.rotmat_to_rotvec(rg[:-1].transpose(-1, -2) @ rg[1:]) / step[:, None]),
            (_forward_rate(xp, step), _forward_rate(xg, step)),
        )
        for m, (pred_rate, gt_rate) in enumerate(rates, start=3):
            stats[2 * m] += float(pred_rate.square().sum())
            stats[2 * m + 1] += float(gt_rate.square().sum())

        # World root error, split by a Gaussian into what a filter could still remove
        # (high pass) and the slow error it could not. Smoothed on the whole clip with
        # the mask, so invalid rows never contribute (and are zeroed, never summed).
        weight = mask.to(torch.float64)
        err = (p_pred[rows] - p_gt[rows]) * weight[:, None] * 1000.0              # (T, 3) mm
        low = gaussian_smooth(err[None], seconds[rows][None], mask[None], _ROOT_SPLIT_SEC)[0]
        for m, series in enumerate((err - low, low), start=6):
            stats[2 * m] += float((series.square().sum(dim=-1) * weight).sum())
            stats[2 * m + 1] += float(weight.sum())
    return stats


def _hand_metric_stats(pred_joints: Tensor, gt_joints: Tensor, valid: Tensor) -> Tensor:
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
    return torch.tensor([float((err * mask).sum()), float(mask.sum()),
                         float((pa_err * mask).sum()), float(mask.sum())], dtype=torch.float64)


def pose_metrics_from_stats(stats: Tensor, hands: bool, world: bool = False) -> dict[str, float]:
    """:func:`metric_names` from the summed statistics vector."""
    out = {}
    for i, name in enumerate(metric_names(hands, world)):
        value = mean_from_stats(float(stats[2 * i]), float(stats[2 * i + 1]))
        out[name] = math.sqrt(value) if name in _ROOT_METRICS and value == value else value
    return out


class SmplxLoss(Loss):
    """Huber keypoint / camera terms and 6D / betas MSE for the SMPL-X head."""

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
        self.layer_weight = float(section["layer_weight"])
        #: The subset deep supervision repeats on every intermediate layer.
        self.layer_terms = tuple(n for n in LAYER_TERM_NAMES if self.weights[n] > 0.0)
        if self.layer_weight > 0.0:
            self.term_names += tuple(f"{n}_layer" for n in self.layer_terms)
        self.delta = {name: float(loss_cfg[f"huber_delta_{name}"])
                      for name in ("2d", "3d", "cam", "root_bias", "root_shape", "depth",
                                   "bearing")}
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
        # A refined body carries the world keys the research diagnostics read.
        self.world_diag = self.model.refiner is not None
        self.stat_names = tuple(f"{key}_{part}"
                                for key in metric_names(self.hands, self.world_diag)
                                for part in ("sum", "count"))

    def _body_terms(self, joints: Tensor, root_6d: Tensor, body_6d: Tensor,
                    pelvis_world, gt_joints: Tensor, gt_root_6d: Tensor, gt_body_6d: Tensor,
                    mask: Tensor, mass: float, batch: dict) -> dict[str, tuple[Tensor, float]]:
        """The terms ONE body carries (:data:`LAYER_TERM_NAMES`), numerators un-weighted.

        :param pelvis_world: the body's world root — ``None`` lifts its own camera pelvis
            (the final body); an intermediate layer passes its ``pelvis_world_layers`` entry.
        """
        raw: dict[str, tuple[Tensor, float]] = {}
        if self.weights["kp3d"] > 0.0:
            huber = F.smooth_l1_loss(joints - joints[:, :1], gt_joints - gt_joints[:, :1],
                                     reduction="none", beta=self.delta["3d"])
            raw["kp3d"] = (((huber.mean(dim=-1) * self.joint_w).sum(dim=1) * mask).sum(), mass)
        if self.weights["orient"] > 0.0:
            raw["orient"] = (((root_6d - gt_root_6d).square().mean(dim=-1) * mask).sum(), mass)
        if self.weights["pose"] > 0.0:
            raw["pose"] = (((body_6d - gt_body_6d).square().mean(dim=(1, 2)) * mask).sum(), mass)
        if self.weights["root_bias"] > 0.0 or self.weights["root_shape"] > 0.0:
            # World root error, split per clip into its mean over the supervised rows (the
            # absolute anchor) and the per-frame deviation from it (the trajectory shape).
            seq_len = int(batch["seq_len"])
            n_clips = joints.shape[0] // seq_len
            pred_w = (lift_to_world(joints[:, :1], batch["cam_from_world"])[:, 0]
                      if pelvis_world is None else pelvis_world.to(self.device, self.dtype))
            gt_w = batch["smplx_joints_world"][:, 0].to(self.device, self.dtype)
            err = ((pred_w - gt_w) * mask[:, None]).view(n_clips, seq_len, 3)
            rows = mask.view(n_clips, seq_len, 1)
            bias = err.sum(dim=1, keepdim=True) / rows.sum(dim=1, keepdim=True).clamp(min=1.0)
            bias_norm = bias.norm(dim=-1).expand(n_clips, seq_len).reshape(-1)
            shape_norm = ((err - bias) * rows).norm(dim=-1).reshape(-1)
            if self.weights["root_bias"] > 0.0:
                huber = F.smooth_l1_loss(bias_norm, torch.zeros_like(bias_norm),
                                         reduction="none", beta=self.delta["root_bias"])
                raw["root_bias"] = ((huber * mask).sum(), mass)
            if self.weights["root_shape"] > 0.0:
                huber = F.smooth_l1_loss(shape_norm, torch.zeros_like(shape_norm),
                                         reduction="none", beta=self.delta["root_shape"])
                raw["root_shape"] = ((huber * mask).sum(), mass)
        return raw

    def layer_raw(self, pred: dict, gt_joints: Tensor, gt_root_6d: Tensor, gt_body_6d: Tensor,
                  mask: Tensor, mass: float, batch: dict) -> dict[str, tuple[Tensor, float]]:
        """Deep supervision (``layer_weight``): :meth:`_body_terms` on the INTERMEDIATE layers.

        One ``<term>_layer`` per term, the layers pooled (numerators and masses summed), at
        ``layer_weight`` times the term's own weight.
        """
        sums = {name: torch.zeros((), device=self.device, dtype=self.dtype)
                for name in self.layer_terms}
        masses = {name: 0.0 for name in self.layer_terms}
        layers = zip(pred["joints_cam_layers"][:-1], pred["root_6d_layers"][:-1],
                     pred["body_6d_layers"][:-1], pred["pelvis_world_layers"][:-1])
        for joints, root_6d, body_6d, pelvis_world in layers:
            raw = self._body_terms(joints.to(self.device, self.dtype)[:, :gt_joints.shape[1]],
                                   root_6d.to(self.device, self.dtype),
                                   body_6d.to(self.device, self.dtype), pelvis_world,
                                   gt_joints, gt_root_6d, gt_body_6d, mask, mass, batch)
            for name, (numerator, term_mass) in raw.items():
                sums[name] = sums[name] + self.weights[name] * numerator
                masses[name] += term_mass
        return {f"{name}_layer": (self.layer_weight * sums[name], masses[name])
                for name in self.layer_terms}

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
                huber = F.smooth_l1_loss(kp2d_crop, gt_crop, reduction="none", beta=self.delta["2d"])
            else:
                focal = cam_int[:, 0, 0, None, None]
                huber = F.smooth_l1_loss(kp2d_full / focal, gt_full / focal, reduction="none",
                                         beta=self.delta["2d"])
            raw["kp2d"] = (((huber.mean(dim=-1) * self.joint_w).sum(dim=1) * mask).sum(), mass)
        raw.update(self._body_terms(joints, root_6d, body_6d, None, gt_joints, gt_root_6d,
                                    gt_body_6d, mask, mass, batch))
        if self.weights["hand_pose"] > 0.0:
            gt_hand_6d = rotmat_to_rot6d(gt["hand_rot"])
            raw["hand_pose"] = (
                ((hand_6d - gt_hand_6d).square().mean(dim=(1, 2)) * mask).sum(), mass)
        if self.weights["betas"] > 0.0:
            raw["betas"] = (((betas - gt["betas"]).square().mean(dim=-1) * mask).sum(), mass)
        if self.weights["cam"] > 0.0:
            gt_cam = translation_to_cliff_cam(gt_joints[:, 0], bbox_center, bbox_size, cam_int)
            huber = F.smooth_l1_loss(cam, gt_cam, reduction="none", beta=self.delta["cam"])
            raw["cam"] = ((huber.mean(dim=-1) * mask).sum(), mass)
        if self.weights["depth"] > 0.0:
            huber = F.smooth_l1_loss(ray_pred[:, 2], ray_gt[:, 2], reduction="none",
                                     beta=self.delta["depth"])
            raw["depth"] = ((huber * mask).sum(), mass)
        if self.weights["bearing"] > 0.0:
            huber = F.smooth_l1_loss(ray_pred[:, :2], ray_gt[:, :2], reduction="none",
                                     beta=self.delta["bearing"])
            raw["bearing"] = ((huber.mean(dim=-1) * mask).sum(), mass)

        stats = self.empty_stats()
        if not train:
            # Vertices only at evaluation: the metrics are their sole consumer.
            # Flat hands on both sides (the 22-joint body at the body part of q)
            # keep PVE comparable across hands / no-hands runs.
            body = self.model.head_smplx.body_flat(self.device)
            pred_verts = smplx_vertices(
                body, betas.detach(), pred["q_cam"].detach()[:, :BODY_Q_DIM])
            gt_verts = smplx_vertices(body, gt["betas"], gt["q"][:, :BODY_Q_DIM])
            stats = eval_stats(joints.detach(), pred_verts, gt_joints, gt_verts,
                               gt["valid"], batch,
                               world=pred if self.world_diag else None).to(self.device)
        weighted = {name: (self.weights[name] * numerator, term_mass)
                    for name, (numerator, term_mass) in raw.items()}
        if self.layer_weight > 0.0:
            weighted.update(self.layer_raw(pred, gt_joints, gt_root_6d, gt_body_6d,
                                           mask, mass, batch))
        return LossResult(terms=self._terms(weighted, anchor), scalars={"n_rows": mass},
                          stats=stats)

    def metrics(self, stats: Tensor) -> dict[str, float]:
        return pose_metrics_from_stats(stats, self.hands, self.world_diag)


__all__ = ["SmplxLoss", "POSE_METRICS", "LIFTED_METRICS", "HAND_METRICS", "DIAG_METRICS",
           "SMPLX_HIPS", "BODY_Q_DIM", "metric_names", "gt_smplx_camera", "smplx_q",
           "smplx_vertices", "eval_stats", "pose_metric_stats", "lifted_metric_stats",
           "world_diag_stats", "pose_metrics_from_stats"]
