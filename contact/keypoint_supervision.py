"""SAM3D-style keypoint supervision from the corpus kindyn GT.

Supervises the FINAL pose output's keypoints against the kindyn world-frame
joints (``kp3d_world``/``kp_valid``, loaded with ``load_keypoints``) lifted to
the camera with the dataset extrinsics — the stabilizers for pose/camera
fine-tuning. SAM 3D Body itself (arXiv 2602.15989, Sec. 4) trains L1 on 2D
keypoints in the cropped image space and on pelvis-normalized 3D keypoints;
we keep those two spaces and use Huber instead of L1 because our pseudo-GT
carries few-cm SMPL-X-joint vs MHR70-keypoint rig offsets. Three terms:

* ``kp2d`` — ``pred_keypoints_2d_cropped`` (the model's own crop-normalized
  [-0.5, 0.5] space) vs the GT joints projected with the SAME intrinsics and
  crop affine. The CLIFF-style term: the only gradient path that constrains
  the camera head's (s, tx, ty) readout.
* ``kp3d`` — mean-hips-relative camera-frame 3D (metres): articulation +
  global orientation, independent of the camera translation.
* ``kp3d_abs`` — ABSOLUTE camera-frame 3D (metres): pins ``pred_cam_t``
  (depth included) to the metric extrinsics — the anti-collapse anchor the
  upstream recipe never had (no metric extrinsics there; the corpus has them).
* ``kp_vel`` / ``kp_acc`` — WORLD-frame velocity/acceleration of the predicted
  keypoints vs the finite-differenced GT (central 3-point stencil on the
  clip's real elapsed seconds; interior rows only). The predictions are lifted
  to the kindyn world with the GT extrinsics — loss-only use: differencing in
  world removes the camera egomotion exactly (camera-frame differences would
  bury body motion under handheld camera shake and couple position error to
  the camera's rotation rate), and the GT side needs no transform at all
  (``joints_world`` is stored in world). Rows whose GT acceleration exceeds
  ``outlier_acc`` (broken kindyn frames) are dropped.

The 13 supervised joints are ``KP_JOINT_NAMES`` (climbing_corpus order)
matched to MHR70 keypoints by name (:data:`KP_MHR70_INDICES`). Terms follow
the force/motion ``(weighted_numerator, mass)`` contract so the trainer's
exact-DDP reduction applies; mass = supervised frame rows (a row supervises
all 13 joints or none — kindyn validity is per-frame). Gradients flow through
the final-readout recompute into the pose path (pose_temporal, fine-tuned head
copies) and the camera copy; every frozen param has ``requires_grad=False``.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .data.climbing_corpus import KP_JOINT_NAMES

#: MHR70 keypoint index for each :data:`KP_JOINT_NAMES` entry (name-matched:
#: shoulders 5/6, elbows 7/8, wrists 62/41, hips 9/10, knees 11/12, ankles
#: 13/14, neck 69 — see ``sam_3d_body/metadata/mhr70.py``).
KP_MHR70_INDICES = (5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14, 69)
#: Positions of left_hip / right_hip inside the 13-joint set (the centering
#: root for the relative 3D term; MHR70 has no pelvis keypoint).
_HIP_POSITIONS = (6, 7)
#: Minimum camera-frame depth (metres) for a projectable GT row.
_MIN_DEPTH_M = 0.25

assert len(KP_JOINT_NAMES) == len(KP_MHR70_INDICES)


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
        self.w_vel = float(ks["loss"].get("kp_vel", 0.0))
        self.w_acc = float(ks["loss"].get("kp_acc", 0.0))
        self.delta_2d = float(ks["loss"]["huber_delta_2d"])
        self.delta_3d = float(ks["loss"]["huber_delta_3d"])
        self.delta_vel = float(ks["loss"].get("huber_delta_vel", 0.5))
        self.delta_acc = float(ks["loss"].get("huber_delta_acc", 2.0))
        self.outlier_acc = float(ks["loss"].get("outlier_acc", 50.0))
        self.kp_idx = torch.tensor(KP_MHR70_INDICES, device=device)

    def forward(self, out: dict, batch: dict) -> tuple[Tensor, dict[str, Any]]:
        """Return ``(total, parts)`` with the force/motion term contract."""
        mhr = out["mhr"]
        kp3d = mhr["pred_keypoints_3d"].to(self.dtype)[:, self.kp_idx]  # (B,13,3)
        cam_t = mhr["pred_cam_t"].to(self.dtype)                        # (B,3)
        kp2d = mhr["pred_keypoints_2d_cropped"].to(self.dtype)[:, self.kp_idx]
        n = kp3d.shape[0]
        # Keeps the pose/camera paths on the backward graph on rows with no
        # supervision (DDP find_unused_parameters=False).
        zero_touch = (kp3d.sum() + cam_t.sum() + kp2d.sum()) * 0.0

        gt_world = batch["kp3d_world"].to(kp3d.device, self.dtype)      # (B,13,3)
        ext = batch["cam_from_world"].to(kp3d.device, self.dtype)       # (B,4,4)
        gt_cam = (torch.einsum("bij,bkj->bki", ext[:, :3, :3], gt_world)
                  + ext[:, :3, 3][:, None])                             # (B,13,3)
        valid = (batch["kp_valid"] & batch["cam_valid"]
                 & batch["frame_valid"]).to(kp3d.device)
        # A GT joint at/behind the camera plane means broken extrinsics for the
        # row — drop the whole row rather than mask per joint.
        valid = valid & (gt_cam[..., 2] > _MIN_DEPTH_M).all(dim=-1)
        mask = valid.to(self.dtype)                                     # (B,)
        mass = float(valid.sum())

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
            terms["kp2d"] = (self.w_kp2d
                             * (h2d.sum(dim=(-2, -1)) * mask).sum(), mass)
        if self.w_kp3d > 0.0:
            pred_rel = kp3d - kp3d[:, _HIP_POSITIONS].mean(dim=1, keepdim=True)
            gt_rel = gt_cam - gt_cam[:, _HIP_POSITIONS].mean(dim=1, keepdim=True)
            h3d = F.smooth_l1_loss(pred_rel, gt_rel, reduction="none",
                                   beta=self.delta_3d)
            terms["kp3d"] = (self.w_kp3d
                             * (h3d.sum(dim=(-2, -1)) * mask).sum(), mass)
        if self.w_kp3d_abs > 0.0:
            pred_abs = kp3d + cam_t[:, None]
            habs = F.smooth_l1_loss(pred_abs, gt_cam, reduction="none",
                                    beta=self.delta_3d)
            terms["kp3d_abs"] = (self.w_kp3d_abs
                                 * (habs.sum(dim=(-2, -1)) * mask).sum(), mass)

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
                        self.w_vel * (hv.sum(dim=(-2, -1)) * sup).sum(),
                        sup_mass)
                if self.w_acc > 0.0:
                    ha = F.smooth_l1_loss(acc_p, acc_g, reduction="none",
                                          beta=self.delta_acc)
                    terms["kp_acc"] = (
                        self.w_acc * (ha.sum(dim=(-2, -1)) * sup).sum(),
                        sup_mass)
                with torch.no_grad():
                    if sup_mass > 0:
                        srows = support
                        vel_diag["kp_vel_err_ms"] = float(
                            (vel_p - vel_g)[srows].norm(dim=-1).mean())
                        vel_diag["kp_acc_err_ms2"] = float(
                            (acc_p - acc_g)[srows].norm(dim=-1).mean())

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
