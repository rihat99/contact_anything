"""Keypoint / vertex / rail supervision of the final pose and camera readout.

Supervises the FINAL pose output against ``mhr_sup_1``: the SAM3D model's own
MHR module evaluated at the GT ``(lbs_params, identity)`` — same rig, same
regressor, same vertex topology as the prediction, so there is no cross-rig
offset by construction and all 70 keypoints are usable. The GT lives in world
metres and is lifted to the camera with the dataset extrinsics.

Every term is a weighted MEAN over its elements (joints / vertices), so its
magnitude does not depend on the element count or on the weight vector. Joints
carry a per-joint weight: the 40 finger/thumb keypoints are the least reliable
part of the mesh fit and have a negligible lever arm on the body pose, and the
5 face keypoints inherit whatever the underlying fit did with the head.

Terms:

* ``kp2d`` — the model's own crop-normalized ``[-0.5, 0.5]`` keypoints vs the GT
  projected with the SAME intrinsics and crop affine. The CLIFF-style term, and
  the only gradient path that constrains the camera head's ``(s, tx, ty)``.
* ``kp3d`` — mean-hips-relative camera-frame metres: articulation and global
  orientation, independent of the camera translation.
* ``kp3d_abs`` — ABSOLUTE camera-frame metres: pins ``pred_cam_t`` (depth
  included) to the metric extrinsics.
* ``vert`` / ``vert_abs`` — the same relative / absolute Huber on a vertex
  subset. 70 sparse landmarks leave body VOLUME almost unconstrained, and body
  size drifts freely under the scale/depth ambiguity while every keypoint error
  still looks healthy; ``vert_abs`` is the size + depth anchor, ``vert`` the
  shape term. Both share the relative term's mean-hips root, so they compose
  with ``kp3d`` rather than fighting it over where the body's origin is.
* ``kp_vel`` / ``kp_acc`` — WORLD-frame velocity/acceleration of the predicted
  keypoints vs the finite-differenced GT (central 3-point stencil over the
  clip's real elapsed seconds, interior rows only). Differencing in the world
  removes the camera egomotion exactly; camera-frame differences would bury body
  motion under handheld shake and couple position error to the camera's rotation
  rate. Rows whose GT acceleration exceeds ``outlier_acc`` are dropped.
* ``cam_rail`` / ``rot_rail`` — trust regions on the camera translation and the
  global orientation against the FROZEN model's own outputs:
  ``relu(deviation - margin)``, exactly zero for a healthy model and linear
  beyond. Derivative losses reward temporal constancy and leave the absolute
  placement in a null space, so the collapse region has to be explicitly uphill
  regardless of how the other terms' gradient magnitudes evolve.

Mass is supervised frame rows: fit validity is per-frame, so a row supervises
all its joints or none.
"""
from __future__ import annotations

import math

import roma
import torch
import torch.nn.functional as F
from torch import Tensor

from model.loss import Loss, LossResult
from utils.geometry import (
    so3_log,
    HIP_KEYPOINTS,
    lift_to_world,
)
from utils.metrics import mean_from_stats

#: MHR70 finger/thumb keypoints: the 20 per hand strictly BETWEEN the two wrist
#: entries (41 = right wrist, 62 = left wrist, both excluded — body joints).
FINGER_KEYPOINTS = tuple(range(21, 41)) + tuple(range(42, 62))
#: MHR70 face keypoints: nose, left/right eye, left/right ear.
FACE_KEYPOINTS = (0, 1, 2, 3, 4)
NUM_MHR70 = 70
#: Minimum camera-frame depth (metres) for a projectable GT row.
_MIN_DEPTH_M = 0.25

_TERM_NAMES = ("kp2d", "kp3d", "kp3d_abs", "vert", "vert_abs",
               "kp_vel", "kp_acc", "cam_rail", "rot_rail")
#: Diagnostic quantities, each stored as an additive ``(numerator, mass)`` pair.
_DIAGNOSTICS = ("kp2d_err_crop", "kp3d_err_m", "depth_err_m", "kp_vel_err_ms",
                "kp_acc_err_ms2", "vert_err_m", "vert_size_ratio", "cam_dev_m",
                "rot_dev_deg")


def joint_weight_vector(
    finger_weight: float, face_weight: float,
    device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Per-MHR70-joint loss weights, ``(70,)``; body joints at 1.0."""
    weights = torch.ones(NUM_MHR70, device=device, dtype=dtype)
    weights[list(FINGER_KEYPOINTS)] = float(finger_weight)
    weights[list(FACE_KEYPOINTS)] = float(face_weight)
    return weights


class KeypointLoss(Loss):
    """Huber keypoint / vertex terms plus the camera and orientation rails."""

    name = "keypoint"
    stat_names = tuple(f"{key}_{part}" for key in _DIAGNOSTICS
                       for part in ("num", "mass"))

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        super().__init__(cfg, model, device)
        ks = cfg["keypoint_supervision"]
        loss_cfg = ks["loss"]
        self.weights = {name: float(loss_cfg[name]) for name in _TERM_NAMES}
        self.term_names = tuple(
            n for n in ("kp2d", "kp3d", "kp3d_abs", "vert", "vert_abs", "kp_vel",
                        "kp_acc", "cam_rail", "rot_rail") if self.weights[n] > 0.0)
        if not self.term_names:
            raise ValueError(
                "keypoint_supervision: every loss weight is 0 — disable the section instead")
        self.delta_2d = float(loss_cfg["huber_delta_2d"])
        self.delta_3d = float(loss_cfg["huber_delta_3d"])
        self.delta_vel = float(loss_cfg["huber_delta_vel"])
        self.delta_acc = float(loss_cfg["huber_delta_acc"])
        self.outlier_acc = float(loss_cfg["outlier_acc"])
        self.cam_rail_margin_m = float(loss_cfg["cam_rail_margin_m"])
        self.rot_rail_margin_rad = float(loss_cfg["rot_rail_margin_rad"])
        self.joint_w = joint_weight_vector(
            ks["joint_weights"]["fingers"], ks["joint_weights"]["face"],
            device=self.device, dtype=self.dtype)[:, None]               # (70, 1)
        self.joint_w_sum = float(self.joint_w.sum())

    def _joint_mean(self, elementwise: Tensor) -> Tensor:
        """Weighted joint mean, coordinate sum: ``(..., 70, C) -> (...)``."""
        return (elementwise * self.joint_w).sum(dim=(-2, -1)) / self.joint_w_sum

    def __call__(self, out: dict, batch: dict, *, train: bool) -> LossResult:
        mhr = out["mhr"]
        kp3d = mhr["pred_keypoints_3d"].to(self.device, self.dtype)      # (B,70,3)
        cam_t = mhr["pred_cam_t"].to(self.device, self.dtype)            # (B,3)
        kp2d = mhr["pred_keypoints_2d_cropped"].to(self.device, self.dtype)
        anchor = (kp3d.sum() + cam_t.sum() + kp2d.sum()) * 0.0

        gt_world = batch["kp3d_world"].to(self.device, self.dtype)       # (B,70,3)
        ext = batch["cam_from_world"].to(self.device, self.dtype)        # (B,4,4)
        gt_cam = (torch.einsum("bij,bkj->bki", ext[:, :3, :3], gt_world)
                  + ext[:, :3, 3][:, None])
        valid = (batch["kp_valid"] & batch["frame_valid"]).to(self.device)
        # A GT joint at or behind the camera plane means broken extrinsics for
        # the row — drop the whole row rather than mask per joint.
        valid = valid & (gt_cam[..., 2] > _MIN_DEPTH_M).all(dim=-1)
        mask = valid.to(self.dtype)                                      # (B,)
        mass = float(mask.sum())

        raw: dict[str, tuple[Tensor, float]] = {}
        diagnostics = {name: (0.0, 0.0) for name in _DIAGNOSTICS}

        if self.weights["kp2d"] > 0.0:
            gt_crop = self._project_to_crop(gt_cam, batch)
            huber = F.smooth_l1_loss(kp2d, gt_crop, reduction="none",
                                     beta=self.delta_2d)
            raw["kp2d"] = ((self._joint_mean(huber) * mask).sum(), mass)
            with torch.no_grad():
                diagnostics["kp2d_err_crop"] = (
                    float(((kp2d - gt_crop).norm(dim=-1).mean(dim=-1) * mask).sum()),
                    mass)

        pred_root = kp3d[:, list(HIP_KEYPOINTS)].mean(dim=1, keepdim=True)
        gt_root = gt_cam[:, list(HIP_KEYPOINTS)].mean(dim=1, keepdim=True)
        if self.weights["kp3d"] > 0.0:
            huber = F.smooth_l1_loss(kp3d - pred_root, gt_cam - gt_root,
                                     reduction="none", beta=self.delta_3d)
            raw["kp3d"] = ((self._joint_mean(huber) * mask).sum(), mass)
        if self.weights["kp3d_abs"] > 0.0:
            huber = F.smooth_l1_loss(kp3d + cam_t[:, None], gt_cam,
                                     reduction="none", beta=self.delta_3d)
            raw["kp3d_abs"] = ((self._joint_mean(huber) * mask).sum(), mass)
        with torch.no_grad():
            error = (kp3d + cam_t[:, None]) - gt_cam                     # (B,70,3)
            diagnostics["kp3d_err_m"] = (
                float((error.norm(dim=-1).mean(dim=-1) * mask).sum()), mass)
            diagnostics["depth_err_m"] = (
                float((error[..., 2].abs().mean(dim=-1) * mask).sum()), mass)

        if self.weights["vert"] > 0.0 or self.weights["vert_abs"] > 0.0:
            anchor = anchor + self._vertex_terms(
                mhr, batch, ext, valid, pred_root, gt_root, cam_t, raw, diagnostics)
        if self.weights["kp_vel"] > 0.0 or self.weights["kp_acc"] > 0.0:
            self._velocity_terms(
                mhr, batch, ext, gt_world, valid, raw, diagnostics)
        self._rail_terms(mhr, batch, cam_t, raw, diagnostics)

        stats = torch.tensor(
            [value for name in _DIAGNOSTICS for value in diagnostics[name]],
            dtype=torch.float64, device=self.device)
        scalars = {name: mean_from_stats(*diagnostics[name])
                   for name in _DIAGNOSTICS if diagnostics[name][1] > 0.0}
        scalars["n_rows"] = mass
        weighted = {name: (self.weights[name] * numerator, term_mass)
                    for name, (numerator, term_mass) in raw.items()}
        return LossResult(terms=self._terms(weighted, anchor), scalars=scalars,
                          stats=stats)

    def _project_to_crop(self, gt_cam: Tensor, batch: dict) -> Tensor:
        """GT camera-frame points -> the model's crop-normalized 2D space.

        Mirrors the model's own ``_full_to_crop`` exactly: intrinsics ->
        full-image px -> crop affine -> ``/ img_size - 0.5``.
        """
        cam_int = batch["cam_int"].to(self.device, self.dtype)           # (B,3,3)
        z = gt_cam[..., 2].clamp(min=_MIN_DEPTH_M)
        u = cam_int[:, 0, 0, None] * gt_cam[..., 0] / z + cam_int[:, 0, 2, None]
        v = cam_int[:, 1, 1, None] * gt_cam[..., 1] / z + cam_int[:, 1, 2, None]
        pixels = torch.stack([u, v, torch.ones_like(u)], dim=-1)         # (B,70,3)
        affine = batch["affine_trans"].to(self.device, self.dtype)       # (B,2,3)
        img_size = batch["img_size"].to(self.device, self.dtype)[:, None]
        return (pixels @ affine.mT) / img_size - 0.5

    def _vertex_terms(
        self, mhr: dict, batch: dict, ext: Tensor, valid: Tensor,
        pred_root: Tensor, gt_root: Tensor, cam_t: Tensor,
        raw: dict, diagnostics: dict,
    ) -> Tensor:
        """Relative / absolute vertex-subset Huber; returns the graph anchor."""
        # pred_vertices is root-relative camera-frame metres exactly like
        # pred_keypoints_3d (the head zeroes the root translation and puts the
        # placement in pred_cam_t), always computed and always differentiable.
        indices = batch["vert_indices"].to(self.device)                  # (V,)
        verts = mhr["pred_vertices"].to(self.device, self.dtype)[:, indices]
        gt_world = batch["vert_gt_world"].to(self.device, self.dtype)
        gt_cam = (torch.einsum("bij,bkj->bki", ext[:, :3, :3], gt_world)
                  + ext[:, :3, 3][:, None])
        v_valid = valid & batch["vert_valid"].to(self.device)
        mask = v_valid.to(self.dtype)
        mass = float(mask.sum())

        if self.weights["vert"] > 0.0:
            huber = F.smooth_l1_loss(verts - pred_root, gt_cam - gt_root,
                                     reduction="none", beta=self.delta_3d)
            raw["vert"] = ((huber.mean(dim=-2).sum(-1) * mask).sum(), mass)
        if self.weights["vert_abs"] > 0.0:
            huber = F.smooth_l1_loss(verts + cam_t[:, None], gt_cam,
                                     reduction="none", beta=self.delta_3d)
            raw["vert_abs"] = ((huber.mean(dim=-2).sum(-1) * mask).sum(), mass)
        with torch.no_grad():
            diagnostics["vert_err_m"] = (
                float((((verts + cam_t[:, None]) - gt_cam).norm(dim=-1).mean(dim=-1)
                       * mask).sum()), mass)
            # Mean radius about the body root: the body-SIZE channel, reported as
            # the ratio of the two summed radii so it stays exactly additive.
            diagnostics["vert_size_ratio"] = (
                float(((verts - pred_root).norm(dim=-1).mean(dim=-1) * mask).sum()),
                float(((gt_cam - gt_root).norm(dim=-1).mean(dim=-1) * mask).sum()))
        return verts.sum() * 0.0

    def _velocity_terms(
        self, mhr: dict, batch: dict, ext: Tensor, gt_world: Tensor,
        valid: Tensor, raw: dict, diagnostics: dict,
    ) -> None:
        """World-frame keypoint velocity / acceleration against the GT stencil."""
        rows = gt_world.shape[0]
        seq_len = int(batch["seq_len"])
        # Zero-mass fallbacks keep the term set identical on every rank.
        for name in ("kp_vel", "kp_acc"):
            if self.weights[name] > 0.0:
                raw[name] = (gt_world.new_zeros(()), 0.0)
        if seq_len < 3 or rows % seq_len:
            return
        n_clips = rows // seq_len

        pred_world = lift_to_world(
            mhr["pred_keypoints_3d"].to(self.device, self.dtype)
            + mhr["pred_cam_t"].to(self.device, self.dtype)[:, None], ext)
        pw = pred_world.reshape(n_clips, seq_len, -1, 3)
        gw = gt_world.reshape(n_clips, seq_len, -1, 3)
        pos_sec = batch["frame_pos_sec"].to(self.device, self.dtype).reshape(
            n_clips, seq_len)
        dt = (pos_sec[:, 1:] - pos_sec[:, :-1]).mean(dim=1).clamp(
            min=1e-6)[:, None, None, None]
        vel_pred = (pw[:, 2:] - pw[:, :-2]) / (2.0 * dt)
        vel_gt = (gw[:, 2:] - gw[:, :-2]) / (2.0 * dt)
        acc_pred = (pw[:, 2:] - 2.0 * pw[:, 1:-1] + pw[:, :-2]) / dt.square()
        acc_gt = (gw[:, 2:] - 2.0 * gw[:, 1:-1] + gw[:, :-2]) / dt.square()

        # The stencil at t reads rows t-1, t, t+1 — every one must be a fully
        # valid row of the SAME clip; GT-acceleration outliers drop.
        ok = valid.reshape(n_clips, seq_len)
        support = (ok[:, :-2] & ok[:, 1:-1] & ok[:, 2:]
                   & ~(acc_gt.norm(dim=-1).amax(dim=-1) > self.outlier_acc))
        mask = support.to(self.dtype)
        mass = float(support.sum())
        if self.weights["kp_vel"] > 0.0:
            huber = F.smooth_l1_loss(vel_pred, vel_gt, reduction="none",
                                     beta=self.delta_vel)
            raw["kp_vel"] = ((self._joint_mean(huber) * mask).sum(), mass)
        if self.weights["kp_acc"] > 0.0:
            huber = F.smooth_l1_loss(acc_pred, acc_gt, reduction="none",
                                     beta=self.delta_acc)
            raw["kp_acc"] = ((self._joint_mean(huber) * mask).sum(), mass)
        with torch.no_grad():
            diagnostics["kp_vel_err_ms"] = (
                float(((vel_pred - vel_gt).norm(dim=-1).mean(dim=-1) * mask).sum()),
                mass)
            diagnostics["kp_acc_err_ms2"] = (
                float(((acc_pred - acc_gt).norm(dim=-1).mean(dim=-1) * mask).sum()),
                mass)

    def _rail_terms(
        self, mhr: dict, batch: dict, cam_t: Tensor, raw: dict, diagnostics: dict,
    ) -> None:
        """Trust regions against the frozen model's camera / orientation."""
        if self.weights["cam_rail"] <= 0.0 and self.weights["rot_rail"] <= 0.0:
            return
        valid = batch["frame_valid"].to(self.device)
        mask = valid.to(self.dtype)
        mass = float(valid.sum())
        if self.weights["cam_rail"] > 0.0:
            frozen = mhr["pred_cam_t_frozen"].to(self.device, self.dtype)
            deviation = (cam_t - frozen).norm(dim=-1)                    # (B,) m
            raw["cam_rail"] = (
                (F.relu(deviation - self.cam_rail_margin_m) * mask).sum(), mass)
            with torch.no_grad():
                diagnostics["cam_dev_m"] = (float((deviation * mask).sum()), mass)
        if self.weights["rot_rail"] > 0.0:
            # Native-axes relative rotation through the so3 log, which is
            # Taylor-smooth at the identity — a geodesic acos would have an
            # exploding gradient there.
            pred = roma.euler_to_rotmat(
                "xyz", mhr["global_rot"].to(self.device, torch.float64))
            frozen = roma.euler_to_rotmat(
                "xyz", mhr["global_rot_frozen"].to(self.device, torch.float64))
            deviation = so3_log(pred.transpose(-1, -2) @ frozen).norm(dim=-1).to(self.dtype)
            raw["rot_rail"] = (
                (F.relu(deviation - self.rot_rail_margin_rad) * mask).sum(), mass)
            with torch.no_grad():
                diagnostics["rot_dev_deg"] = (
                    float((deviation * mask).sum()) * 180.0 / math.pi, mass)

    def metrics(self, stats: Tensor) -> dict[str, float]:
        return {
            name: mean_from_stats(float(stats[2 * i]), float(stats[2 * i + 1]))
            for i, name in enumerate(_DIAGNOSTICS)
        }


__all__ = ["KeypointLoss", "joint_weight_vector", "FINGER_KEYPOINTS",
           "FACE_KEYPOINTS", "NUM_MHR70"]
