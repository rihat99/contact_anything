"""Roll-out evaluation: the predicted body twist integrated into a world trajectory.

Eval only (no loss term). For every test clip the motion head's de-standardized
BODY-frame root twist ``[v, omega]`` (``motion_supervision.linear_frame: body``)
is integrated from the first frame's predicted world root pose — the SMPL-X
head's camera-frame root lifted with THAT frame's GT extrinsics; every later
frame uses the prediction only::

    T_{t+1} = T_t · exp(dt_t · (xi_t + xi_{t+1}) / 2)

(the trapezoid inverse of the loader's central-difference twist), and the
SMPL-X head's root-local joints ride on the integrated root. The result is
scored with GVHMR's global metrics (:mod:`utils.gvhmr_metrics`): W-MPJPE100
(first-two-frame alignment per 100-frame chunk), WA-MPJPE100 (per-chunk
similarity alignment), RTE (root translation error over the GT path length,
%) and jitter (10 m/s^3), all against the kindyn world joints. The same
metrics of the ``lifted`` trajectory — the per-frame camera-frame prediction
lifted with the GT extrinsics of EVERY frame, i.e. trusting the camera instead
of the network — are reported next to them, and the GT's own jitter as the
floor. Invalid frames are dropped by GVHMR's mask compaction before chunking.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from model.loss import Loss, LossResult
from model.loss.motion import standardize_table
from utils.geometry import lift_to_world, se3_exp
from utils.gvhmr_metrics import compute_jitter, global_metrics
from utils.metrics import mean_from_stats

NUM_BODY_JOINTS = 22
VARIANTS = ("rollout", "lifted")
METRICS = ("wa_mpjpe100", "w_mpjpe100", "rte", "jitter")


class RolloutLoss(Loss):
    """GVHMR global metrics of the rolled-out and the camera-lifted world trajectory."""

    name = "rollout"
    metric_group = "global"
    term_names = ()
    stat_names = tuple(
        f"{variant}_{metric}_{part}" for variant in VARIANTS for metric in METRICS
        for part in ("sum", "count")) + ("gt_jitter_sum", "gt_jitter_count")

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        super().__init__(cfg, model, device)
        if self.model.head_smplx is None or self.model.head_motion is None:
            raise ValueError("rollout_eval needs the SMPL-X head and the motion head")
        ms = cfg["motion_supervision"]
        if ms["linear_frame"] != "body" or ms["root_source"] != "smplx" or not all(
                t in self.model.motion_terms for t in ("vel", "ang_vel")):
            raise ValueError(
                "rollout_eval integrates the SMPL-X body's twist: motion_supervision."
                "linear_frame must be 'body', root_source 'smplx', and "
                "model.motion.terms must include 'vel' and 'ang_vel'")
        self.mean, self.std = standardize_table(
            cfg, ("vel", "ang_vel"), self.device, torch.float64)          # [1, K, 6]

    def __call__(self, out: dict, batch: dict, *, train: bool) -> LossResult:
        if train:
            return LossResult(terms={}, stats=self.empty_stats())
        return LossResult(terms={}, stats=self._stats(out, batch))

    @torch.no_grad()
    def _stats(self, out: dict, batch: dict) -> Tensor:
        seq_len = int(batch["seq_len"])
        smplx = out["smplx"]
        joints = smplx["joints_cam"][:, :NUM_BODY_JOINTS].to(self.device, torch.float64)
        root_rot = smplx["root_rot"].to(self.device, torch.float64)          # cam-from-root
        motion = out["motion"]
        twist = torch.cat([motion["joint_vel"][:, 0], motion["joint_ang_vel"][:, 0]],
                          dim=-1).to(self.device, torch.float64)
        twist = twist * self.std[0, 0] + self.mean[0, 0]                     # [B, 6] physical
        ext = batch["cam_from_world"].to(self.device, torch.float64)
        seconds = batch["frame_pos_sec"].to(self.device, torch.float64)
        gt_world = batch["smplx_joints_world"][:, :NUM_BODY_JOINTS].to(
            self.device, torch.float64)
        valid = (batch["smplx_valid"] & batch["frame_valid"]).to(self.device)

        lifted = lift_to_world(joints, ext)                                  # [B, 22, 3]
        pelvis = joints[:, 0]
        local = torch.einsum("bji,bkj->bki", root_rot, joints - pelvis[:, None])
        rot_w0 = ext[:, :3, :3].transpose(1, 2) @ root_rot                   # world-from-root
        pos_w0 = lift_to_world(pelvis[:, None], ext)[:, 0]

        stats = torch.zeros(len(self.stat_names), dtype=torch.float64)
        n_clips = joints.shape[0] // seq_len
        for clip in range(n_clips):
            rows = slice(clip * seq_len, (clip + 1) * seq_len)
            rolled = self._roll_out(local[rows], twist[rows], seconds[rows],
                                    rot_w0[rows][0], pos_w0[rows][0])
            mask = valid[rows].cpu()
            if int(mask.sum()) < 2:
                continue
            dt = seconds[rows][1:] - seconds[rows][:-1]
            fps = float(1.0 / dt.median()) if seq_len > 1 else 1.0
            gt = gt_world[rows].float().cpu()[mask]
            for v, pred in enumerate((rolled, lifted[rows])):
                result = global_metrics(pred.float().cpu()[mask], gt, fps)
                for m, metric in enumerate(METRICS):
                    values = np.asarray(result[metric], np.float64)
                    base = 2 * (v * len(METRICS) + m)
                    stats[base] += float(values.sum())
                    stats[base + 1] += float(len(values))
            gt_jitter = np.asarray(compute_jitter(gt, fps=fps), np.float64)
            stats[-2] += float(gt_jitter.sum())
            stats[-1] += float(len(gt_jitter))
        return stats

    @staticmethod
    def _roll_out(local: Tensor, twist: Tensor, seconds: Tensor,
                  rot: Tensor, pos: Tensor) -> Tensor:
        """World joints ``(T, J, 3)`` of one clip from its integrated root trajectory."""
        seq_len = local.shape[0]
        world = torch.empty_like(local)
        for k in range(seq_len):
            world[k] = local[k] @ rot.T + pos
            if k + 1 < seq_len:
                step = 0.5 * (twist[k] + twist[k + 1]) * (seconds[k + 1] - seconds[k])
                d_rot, d_pos = se3_exp(step)
                pos = pos + rot @ d_pos
                rot = rot @ d_rot
        return world

    def metrics(self, stats: Tensor) -> dict[str, float]:
        out: dict[str, float] = {}
        for v, variant in enumerate(VARIANTS):
            for m, metric in enumerate(METRICS):
                base = 2 * (v * len(METRICS) + m)
                out[f"{variant}_{metric}"] = mean_from_stats(
                    float(stats[base]), float(stats[base + 1]))
        out["gt_jitter"] = mean_from_stats(float(stats[-2]), float(stats[-1]))
        return out


__all__ = ["RolloutLoss", "VARIANTS", "METRICS"]
