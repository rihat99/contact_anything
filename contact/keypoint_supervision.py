"""SAM3D-style keypoint + vertex supervision from the MHR-native corpus GT.

Supervises the FINAL pose output's keypoints and a vertex subset against
``mhr_sup_1.npz`` (``kp3d_world``/``vert_gt_world``, loaded with
``load_keypoints``) lifted to the camera with the dataset extrinsics — the
stabilizers for pose/camera fine-tuning. SAM 3D Body itself (arXiv 2602.15989,
Sec. 4) trains L1 on 2D keypoints in the cropped image space and on
pelvis-normalized 3D keypoints; we keep those two spaces and use Huber instead
of L1.

**GT source (2026-08-29 swap).** The targets are the SAM3D model's OWN MHR
module evaluated at the ``mhr_1`` GT ``(lbs_params, identity)`` — same rig, same
sapiens-308-sliced-to-70 regressor, same vertex topology as the prediction. The
previous GT was 13 kindyn SMPL-X ``joints_world`` name-matched onto MHR70, whose
constant cross-rig offset the audit measured at 69-75 % of the keypoint MSE; it
is gone by construction, and all 70 keypoints are now supervised (they exist).

**Per-element means.** Every term below is a weighted MEAN over its elements
(joints / vertices), not a sum, so its magnitude does not depend on the joint
count or on the weight vector. The pre-swap terms were sums over 13 joints —
multiply a historical weight by 13 to reproduce its old magnitude.

Terms:

* ``kp2d`` — ``pred_keypoints_2d_cropped`` (the model's own crop-normalized
  [-0.5, 0.5] space) vs the GT joints projected with the SAME intrinsics and
  crop affine. The CLIFF-style term: the only gradient path that constrains
  the camera head's (s, tx, ty) readout.
* ``kp3d`` — mean-hips-relative camera-frame 3D (metres): articulation +
  global orientation, independent of the camera translation.
* ``kp3d_abs`` — ABSOLUTE camera-frame 3D (metres): pins ``pred_cam_t``
  (depth included) to the metric extrinsics — the anti-collapse anchor the
  upstream recipe never had (no metric extrinsics there; the corpus has them).
* ``vert`` / ``vert_abs`` — the same relative / absolute camera-frame Huber on
  the ``vert_gt_world`` vertex subset, with the prediction sliced out of
  ``pred_vertices`` by the GT's own ``vert_indices``. Keypoints are 70 sparse
  landmarks that leave body VOLUME almost unconstrained; the audit traced the
  regression to exactly that (body size drifted +3.9 % -> -3.3 % under the
  scale/depth ambiguity while keypoint error looked fine). ``vert_abs`` is the
  size + depth anchor, ``vert`` the shape/pose term. Both share the relative
  term's mean-hips root so they compose with ``kp3d``.
* ``cam_rail`` / ``rot_rail`` — trust regions on the camera translation /
  global orientation vs the FROZEN model's own outputs
  (``pred_cam_t_frozen`` / ``global_rot_frozen``, stashed by the recompute):
  ``relu(deviation - margin)`` — exactly zero for a healthy model, linear
  beyond. The v4-proven anti-collapse device: derivative losses reward
  temporal constancy and leave the absolute placement in a null space, so
  the collapse region must be explicitly uphill regardless of how the other
  terms' gradient magnitudes evolve (the v2 probe drifted metres in depth).
* ``kp_vel`` / ``kp_acc`` — WORLD-frame velocity/acceleration of the predicted
  keypoints vs the finite-differenced GT (central 3-point stencil on the
  clip's real elapsed seconds; interior rows only). The predictions are lifted
  to the kindyn world with the GT extrinsics — loss-only use: differencing in
  world removes the camera egomotion exactly (camera-frame differences would
  bury body motion under handheld camera shake and couple position error to
  the camera's rotation rate), and the GT side needs no transform at all
  (``joints_world`` is stored in world). Rows whose GT acceleration exceeds
  ``outlier_acc`` (broken kindyn frames) are dropped.
Joints carry a per-joint weight (:func:`joint_weight_vector`): the 40
finger/thumb keypoints (:data:`MHR70_FINGER_INDICES`) are downweighted by
``finger_weight`` — they are the least reliable part of the mesh fit and their
lever arm on the body pose is negligible — and the 5 face keypoints
(:data:`MHR70_FACE_INDICES`) by ``face_weight`` (the audit found the head GT bad
in 4 of 6 inspected scenes, inherited from kindyn's own SMPL-X fit).

``fit_err_confidence`` optionally weights each row by
``1 / (1 + (fit_err_cm / fit_err_ref_cm)^2)`` — the ``mhr_1`` mesh-fit residual
of that frame (mean 0.68 cm; a badly fitted row is a badly known target).

Terms follow the force/motion ``(weighted_numerator, mass)`` contract so the
trainer's exact-DDP reduction applies; mass = supervised frame rows (a row
supervises all joints or none — fit validity is per-frame). Gradients flow
through the final-readout recompute into the pose path (pose_temporal,
fine-tuned head copies) and the camera copy; every frozen param has
``requires_grad=False``.
"""
from __future__ import annotations

import math
from typing import Any

import roma
import torch
import torch.nn.functional as F
from torch import Tensor

from .data.climbing_corpus import KP_JOINT_NAMES, NUM_MHR70
from .root_world import quat_xyzw_from_matrix, so3_log_xyzw

#: MHR70 keypoint index for each :data:`KP_JOINT_NAMES` entry (name-matched:
#: shoulders 5/6, elbows 7/8, wrists 62/41, hips 9/10, knees 11/12, ankles
#: 13/14, neck 69 — see ``sam_3d_body/metadata/mhr70.py``). No longer used by
#: this loss (all 70 keypoints are supervised); kept for the renderers, which
#: still report the 13-joint subset.
KP_MHR70_INDICES = (5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14, 69)
#: MHR70 finger/thumb keypoints: the 20 per hand strictly BETWEEN the two wrist
#: entries (41 = right-wrist, 62 = left-wrist, both excluded — they are body
#: joints). Right hand 21..40 (thumb/index/middle/ring/pinky x tip + three
#: joints), left hand 42..61 in the same layout — see
#: ``sam_3d_body/metadata/mhr70.py``.
MHR70_FINGER_INDICES = tuple(range(21, 41)) + tuple(range(42, 62))
#: MHR70 face keypoints: nose, left/right eye, left/right ear.
MHR70_FACE_INDICES = (0, 1, 2, 3, 4)
#: Positions of left_hip / right_hip in MHR70 (the centering root for the
#: relative 3D terms; MHR70 has no pelvis keypoint). Matches
#: ``contact.root_world._HIP_KPS``.
_HIP_POSITIONS = (9, 10)
#: Minimum camera-frame depth (metres) for a projectable GT row.
_MIN_DEPTH_M = 0.25

assert len(KP_JOINT_NAMES) == len(KP_MHR70_INDICES)
assert len(set(MHR70_FINGER_INDICES)) == 40
assert 41 not in MHR70_FINGER_INDICES and 62 not in MHR70_FINGER_INDICES


def joint_weight_vector(
    finger_weight: float, face_weight: float,
    device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Per-MHR70-joint loss weights. ``(70,)``, body joints at 1.0.

    :param finger_weight: weight of the 40 :data:`MHR70_FINGER_INDICES`.
    :param face_weight: weight of the 5 :data:`MHR70_FACE_INDICES`.
    """
    weights = torch.ones(NUM_MHR70, device=device, dtype=dtype)
    weights[list(MHR70_FINGER_INDICES)] = float(finger_weight)
    weights[list(MHR70_FACE_INDICES)] = float(face_weight)
    return weights


def row_confidence(
    fit_err_cm: Tensor, ref_cm: float, enabled: bool,
) -> Tensor:
    """Row weight from the ``mhr_1`` mesh-fit residual. ``(B,) -> (B,)``.

    ``1 / (1 + (err / ref)^2)`` — 1.0 for a perfect fit, 0.5 at ``ref_cm``.
    Returns ones when disabled, so callers multiply unconditionally.
    """
    if not enabled:
        return torch.ones_like(fit_err_cm)
    return 1.0 / (1.0 + (fit_err_cm / ref_cm).square())


class KeypointSupervisedLoss:
    """Huber keypoint terms against the kindyn GT (2D crop + camera-frame 3D).

    :param cfg: resolved run config; reads ``keypoint_supervision.*``.
    :param device: device the loss runs on.
    """

    def __init__(self, cfg: dict, device: str = "cuda") -> None:
        ks = cfg["keypoint_supervision"]
        self.device = device
        self.dtype = torch.float32
        self.w_kp2d = float(ks["loss"]["kp2d"])
        self.w_kp3d = float(ks["loss"]["kp3d"])
        self.w_kp3d_abs = float(ks["loss"]["kp3d_abs"])
        self.w_vert = float(ks["loss"]["vert"])
        self.w_vert_abs = float(ks["loss"]["vert_abs"])
        self.w_vel = float(ks["loss"].get("kp_vel", 0.0))
        self.w_acc = float(ks["loss"].get("kp_acc", 0.0))
        self.delta_2d = float(ks["loss"]["huber_delta_2d"])
        self.delta_3d = float(ks["loss"]["huber_delta_3d"])
        self.delta_vel = float(ks["loss"].get("huber_delta_vel", 0.5))
        self.delta_acc = float(ks["loss"].get("huber_delta_acc", 2.0))
        self.outlier_acc = float(ks["loss"].get("outlier_acc", 50.0))
        self.w_cam_rail = float(ks["loss"].get("cam_rail", 0.0))
        self.w_rot_rail = float(ks["loss"].get("rot_rail", 0.0))
        self.cam_rail_margin_m = float(ks["loss"].get("cam_rail_margin_m", 0.5))
        self.rot_rail_margin_rad = float(
            ks["loss"].get("rot_rail_margin_rad", 0.2))
        self.fit_err_confidence = bool(ks["fit_err_confidence"])
        self.fit_err_ref_cm = float(ks["fit_err_ref_cm"])
        #: (70,) per-joint weights; every joint term is a MEAN under them.
        self.joint_w = joint_weight_vector(
            ks["joint_weights"]["fingers"], ks["joint_weights"]["face"],
            device=device, dtype=self.dtype)
        self.joint_w_sum = float(self.joint_w.sum())

    def forward(self, out: dict, batch: dict) -> tuple[Tensor, dict[str, Any]]:
        """Return ``(total, parts)`` with the force/motion term contract."""
        mhr = out["mhr"]
        kp3d = mhr["pred_keypoints_3d"].to(self.dtype)                  # (B,70,3)
        cam_t = mhr["pred_cam_t"].to(self.dtype)                        # (B,3)
        kp2d = mhr["pred_keypoints_2d_cropped"].to(self.dtype)          # (B,70,2)
        n = kp3d.shape[0]
        # Keeps the pose/camera paths on the backward graph on rows with no
        # supervision (DDP find_unused_parameters=False).
        zero_touch = (kp3d.sum() + cam_t.sum() + kp2d.sum()) * 0.0

        gt_world = batch["kp3d_world"].to(kp3d.device, self.dtype)      # (B,70,3)
        ext = batch["cam_from_world"].to(kp3d.device, self.dtype)       # (B,4,4)
        gt_cam = (torch.einsum("bij,bkj->bki", ext[:, :3, :3], gt_world)
                  + ext[:, :3, 3][:, None])                             # (B,70,3)
        valid = (batch["kp_valid"] & batch["cam_valid"]
                 & batch["frame_valid"]).to(kp3d.device)
        # A GT joint at/behind the camera plane means broken extrinsics for the
        # row — drop the whole row rather than mask per joint.
        valid = valid & (gt_cam[..., 2] > _MIN_DEPTH_M).all(dim=-1)
        # Row weight = validity x optional mesh-fit confidence; the mass is the
        # same weight summed, so every term stays a weighted MEAN over rows.
        conf = row_confidence(
            batch["mhr_fit_err_cm"].to(kp3d.device, self.dtype),
            self.fit_err_ref_cm, self.fit_err_confidence)
        mask = valid.to(self.dtype) * conf                              # (B,)
        mass = float(mask.sum())
        jw = self.joint_w.to(kp3d.device)[:, None]                      # (70,1)

        def joint_mean(huber: Tensor) -> Tensor:
            """Weighted joint mean, coordinate sum: ``(..., 70, C) -> (...)``."""
            return (huber * jw).sum(dim=(-2, -1)) / self.joint_w_sum

        terms: dict[str, tuple] = {}
        if self.w_kp2d > 0.0:
            # GT projection mirrors the model's _full_to_crop exactly:
            # intrinsics -> full-image px -> crop affine -> /img_size - 0.5.
            cam_int = batch["cam_int"].to(kp3d.device, self.dtype)      # (B,3,3)
            z = gt_cam[..., 2].clamp(min=_MIN_DEPTH_M)
            u = (cam_int[:, 0, 0, None] * gt_cam[..., 0] / z
                 + cam_int[:, 0, 2, None])
            v = (cam_int[:, 1, 1, None] * gt_cam[..., 1] / z
                 + cam_int[:, 1, 2, None])
            gt_px = torch.stack([u, v, torch.ones_like(u)], dim=-1)     # (B,13,3)
            affine = batch["affine_trans"].to(kp3d.device, self.dtype).reshape(n, 2, 3)
            img_size = batch["img_size"].to(kp3d.device, self.dtype).reshape(n, 1, 2)
            gt_crop = (gt_px @ affine.mT) / img_size - 0.5              # (B,13,2)
            h2d = F.smooth_l1_loss(kp2d, gt_crop, reduction="none",
                                   beta=self.delta_2d)
            terms["kp2d"] = (self.w_kp2d * (joint_mean(h2d) * mask).sum(), mass)
        pred_root = kp3d[:, _HIP_POSITIONS].mean(dim=1, keepdim=True)    # (B,1,3)
        gt_root = gt_cam[:, _HIP_POSITIONS].mean(dim=1, keepdim=True)
        if self.w_kp3d > 0.0:
            h3d = F.smooth_l1_loss(kp3d - pred_root, gt_cam - gt_root,
                                   reduction="none", beta=self.delta_3d)
            terms["kp3d"] = (self.w_kp3d * (joint_mean(h3d) * mask).sum(), mass)
        if self.w_kp3d_abs > 0.0:
            pred_abs = kp3d + cam_t[:, None]
            habs = F.smooth_l1_loss(pred_abs, gt_cam, reduction="none",
                                    beta=self.delta_3d)
            terms["kp3d_abs"] = (
                self.w_kp3d_abs * (joint_mean(habs) * mask).sum(), mass)

        vert_diag: dict[str, float] = {}
        if self.w_vert > 0.0 or self.w_vert_abs > 0.0:
            # pred_vertices is root-relative camera-frame metres, exactly like
            # pred_keypoints_3d (the head zeroes the root translation and puts
            # the placement in pred_cam_t), and it is always computed and always
            # differentiable — mhr_head.forward has no gate on it, and the
            # keypoint regressor already backprops through every vertex, so the
            # subset slice below costs nothing measurable.
            vidx = batch["vert_indices"].to(kp3d.device)                # (V,)
            verts = mhr["pred_vertices"].to(self.dtype)[:, vidx]        # (B,V,3)
            gt_v_world = batch["vert_gt_world"].to(kp3d.device, self.dtype)
            gt_v_cam = (torch.einsum("bij,bkj->bki", ext[:, :3, :3], gt_v_world)
                        + ext[:, :3, 3][:, None])                       # (B,V,3)
            v_valid = valid & batch["vert_valid"].to(kp3d.device)
            v_mask = v_valid.to(self.dtype) * conf
            v_mass = float(v_mask.sum())
            zero_touch = zero_touch + verts.sum() * 0.0
            if self.w_vert > 0.0:
                # Same mean-hips root as kp3d, so the two terms compose rather
                # than fight over where the body's origin is.
                hv = F.smooth_l1_loss(verts - pred_root, gt_v_cam - gt_root,
                                      reduction="none", beta=self.delta_3d)
                terms["vert"] = (
                    self.w_vert * (hv.mean(dim=-2).sum(-1) * v_mask).sum(), v_mass)
            if self.w_vert_abs > 0.0:
                hva = F.smooth_l1_loss(verts + cam_t[:, None], gt_v_cam,
                                       reduction="none", beta=self.delta_3d)
                terms["vert_abs"] = (
                    self.w_vert_abs * (hva.mean(dim=-2).sum(-1) * v_mask).sum(),
                    v_mass)
            with torch.no_grad():
                if v_mass > 0:
                    sel = v_valid
                    vert_diag["vert_err_m"] = float(
                        ((verts + cam_t[:, None])[sel] - gt_v_cam[sel])
                        .norm(dim=-1).mean())
                    # Mean radius about the body root: the body-SIZE channel the
                    # audit found drifting +3.9 % -> -3.3 % unsupervised.
                    vert_diag["vert_size_ratio"] = float(
                        (verts - pred_root)[sel].norm(dim=-1).mean()
                        / (gt_v_cam - gt_root)[sel].norm(dim=-1).mean().clamp(min=1e-6))
                else:
                    vert_diag["vert_err_m"] = 0.0
                    vert_diag["vert_size_ratio"] = 0.0

        vel_diag: dict[str, float] = {}
        if self.w_vel > 0.0 or self.w_acc > 0.0:
            # Zero-mass fallbacks keep the term set identical on every rank
            # (the exact-DDP reducer iterates the same names batch-wise).
            zero = kp3d.new_zeros(())
            if self.w_vel > 0.0:
                terms["kp_vel"] = (zero, 0.0)
            if self.w_acc > 0.0:
                terms["kp_acc"] = (zero, 0.0)
            seq_len = int(batch.get("seq_len", 1))
            if seq_len >= 3 and n % seq_len == 0:
                n_clips = n // seq_len
                pred_world = torch.einsum(
                    "bji,bkj->bki", ext[:, :3, :3],
                    kp3d + cam_t[:, None] - ext[:, :3, 3][:, None])
                pw = pred_world.reshape(n_clips, seq_len, -1, 3)
                gw = gt_world.reshape(n_clips, seq_len, -1, 3)
                pos_sec = batch["frame_pos_sec"].to(
                    kp3d.device, self.dtype).reshape(n_clips, seq_len)
                dt = (pos_sec[:, 1:] - pos_sec[:, :-1]).mean(dim=1).clamp(
                    min=1e-6)[:, None, None, None]                  # (n,1,1,1)
                vel_p = (pw[:, 2:] - pw[:, :-2]) / (2.0 * dt)
                vel_g = (gw[:, 2:] - gw[:, :-2]) / (2.0 * dt)
                acc_p = (pw[:, 2:] - 2.0 * pw[:, 1:-1] + pw[:, :-2]) / dt.square()
                acc_g = (gw[:, 2:] - 2.0 * gw[:, 1:-1] + gw[:, :-2]) / dt.square()
                # The stencil at t reads rows t-1, t, t+1 — every one must be
                # a fully valid row of the SAME clip; GT-acc outliers drop.
                ok = valid.reshape(n_clips, seq_len)
                support = (ok[:, :-2] & ok[:, 1:-1] & ok[:, 2:]
                           & ~(acc_g.norm(dim=-1).amax(dim=-1)
                               > self.outlier_acc))                 # (n, T-2)
                sup = support.to(self.dtype)
                sup_mass = float(support.sum())
                if self.w_vel > 0.0:
                    hv = F.smooth_l1_loss(vel_p, vel_g, reduction="none",
                                          beta=self.delta_vel)
                    terms["kp_vel"] = (
                        self.w_vel * (joint_mean(hv) * sup).sum(), sup_mass)
                if self.w_acc > 0.0:
                    ha = F.smooth_l1_loss(acc_p, acc_g, reduction="none",
                                          beta=self.delta_acc)
                    terms["kp_acc"] = (
                        self.w_acc * (joint_mean(ha) * sup).sum(), sup_mass)
                with torch.no_grad():
                    if sup_mass > 0:
                        srows = support
                        vel_diag["kp_vel_err_ms"] = float(
                            (vel_p - vel_g)[srows].norm(dim=-1).mean())
                        vel_diag["kp_acc_err_ms2"] = float(
                            (acc_p - acc_g)[srows].norm(dim=-1).mean())

        rail_diag: dict[str, float] = {}
        if self.w_cam_rail > 0.0 or self.w_rot_rail > 0.0:
            fv = batch["frame_valid"].to(kp3d.device)
            fv_mask = fv.to(self.dtype)
            fv_mass = float(fv.sum())
            if self.w_cam_rail > 0.0:
                frozen_t = out["mhr"].get("pred_cam_t_frozen")
                if frozen_t is None:
                    terms["cam_rail"] = (kp3d.new_zeros(()), 0.0)
                else:
                    cam_dev = (cam_t - frozen_t.to(cam_t.device, self.dtype)
                               ).norm(dim=-1)                       # (B,)
                    rail = F.relu(cam_dev - self.cam_rail_margin_m)
                    terms["cam_rail"] = (
                        self.w_cam_rail * (rail * fv_mask).sum(), fv_mass)
                    with torch.no_grad():
                        rail_diag["cam_dev_m"] = float(
                            (cam_dev * fv_mask).sum() / max(fv_mass, 1.0))
            if self.w_rot_rail > 0.0:
                frozen_r = out["mhr"].get("global_rot_frozen")
                if frozen_r is None:
                    terms["rot_rail"] = (kp3d.new_zeros(()), 0.0)
                else:
                    # Native-axes relative rotation via the so3 log, which is
                    # Taylor-smooth at I — a geodesic acos would have an
                    # exploding gradient there.
                    r_pred = roma.euler_to_rotmat(
                        "xyz", out["mhr"]["global_rot"].to(torch.float64))
                    r_frz = roma.euler_to_rotmat(
                        "xyz", frozen_r.to(kp3d.device, torch.float64))
                    rot_dev = so3_log_xyzw(quat_xyzw_from_matrix(
                        r_pred.transpose(-1, -2) @ r_frz)
                    ).norm(dim=-1).to(self.dtype)                   # (B,) rad
                    rail = F.relu(rot_dev - self.rot_rail_margin_rad)
                    terms["rot_rail"] = (
                        self.w_rot_rail * (rail * fv_mask).sum(), fv_mass)
                    with torch.no_grad():
                        rail_diag["rot_dev_deg"] = float(
                            (rot_dev * fv_mask).sum() / max(fv_mass, 1.0)
                            * 180.0 / math.pi)

        total = None
        parts_terms: dict[str, Any] = {}
        for name, (weighted_raw, term_mass) in terms.items():
            weighted = weighted_raw + zero_touch
            normalized = weighted / max(term_mass, 1.0)
            total = normalized if total is None else total + normalized
            parts_terms[name] = {
                "weighted_numerator_tensor": weighted,
                "weight_mass": term_mass,
                "loss": float(normalized.detach()),
            }
        parts: dict[str, Any] = {
            "terms": parts_terms,
            "loss": float(total.detach()),
            "n_supervised_rows": int(valid.sum()),
        }
        if self.w_vel > 0.0 or self.w_acc > 0.0:
            parts["kp_vel_err_ms"] = vel_diag.get("kp_vel_err_ms", 0.0)
            parts["kp_acc_err_ms2"] = vel_diag.get("kp_acc_err_ms2", 0.0)
        parts.update(rail_diag)
        parts.update(vert_diag)
        with torch.no_grad():
            if mass > 0:
                sel = valid
                parts["kp2d_err_crop"] = float(
                    (kp2d[sel] - gt_crop[sel]).norm(dim=-1).mean()
                ) if self.w_kp2d > 0.0 else 0.0
                parts["kp3d_err_m"] = float(
                    ((kp3d + cam_t[:, None])[sel] - gt_cam[sel])
                    .norm(dim=-1).mean())
                parts["depth_err_m"] = float(
                    ((kp3d + cam_t[:, None])[sel][..., 2]
                     - gt_cam[sel][..., 2]).abs().mean())
            else:
                parts["kp2d_err_crop"] = 0.0
                parts["kp3d_err_m"] = 0.0
                parts["depth_err_m"] = 0.0
        return total, parts

    __call__ = forward
