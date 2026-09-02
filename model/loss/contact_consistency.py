"""Contact -> velocity consistency: a limb the model calls in contact must stand still.

The corpus contact labels are motion-gated *stable* contact (stillness plus
hysteresis in the estimator), so a predicted contact makes a claim about the
WORLD: that extremity is not moving. This loss states the claim as an
objective — the world-frame speed of the six extremity keypoints, weighted by
the model's own contact probability::

    L = sum_rows sum_limbs  p_contact * huber(||v_world||, 0)  /  sum p_contact

It needs no labels: the gate and the trajectory both come out of the forward
pass. Two gradient paths open:

* the **pose** path (``pred_keypoints_3d`` / ``pred_cam_t``): where contact is
  predicted, the pose must stop jittering — the depth-wobble attack applied to
  the limbs that are pinned to the wall;
* the **contact** path, only under ``detach_gate: false``: the head can lower a
  probability to escape a moving limb's penalty. The supervised focal loss is
  the counterweight, and a collapse of the gate shows up immediately as a
  contact-F1 drop.

Velocity is a central difference of the world-lifted keypoints over the clip's
real elapsed seconds: differencing in the world removes the camera egomotion, so
a handheld camera cannot fake limb motion. Rows need the full stencil
(``t-1, t, t+1``) inside one clip, so clip boundaries are never supervised and
``data.clip.frames >= 3`` is required.

Mass is the summed gate, not the row count: normalising by confidence mass keeps
the term's scale when the model predicts few contacts.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from model.loss import KINDYN_GROUP_KEYPOINTS, Loss, LossResult
from utils.geometry import predicted_keypoints_world
from utils.metrics import mean_from_stats


class ContactConsistencyLoss(Loss):
    """Gated world-speed penalty on the six extremity keypoints."""

    name = "contact_consistency"
    stat_names = ("vel_num", "vel_mass")

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        super().__init__(cfg, model, device)
        cc = cfg["contact_consistency"]
        self.detach_gate = bool(cc["detach_gate"])
        self.weight = float(cc["loss"]["vel"])
        self.term_names = ("vel",)
        self.huber_delta_ms = float(cc["loss"]["huber_delta_ms"])
        self.keypoints = torch.tensor(KINDYN_GROUP_KEYPOINTS, device=self.device)

    def __call__(self, out: dict, batch: dict, *, train: bool) -> LossResult:
        mhr = out["mhr"]
        gate_all = out["contact"]["joint_probs"].to(self.device, self.dtype)
        # Graph-connected zero over every tensor the loss consumes: the pose and
        # contact params must stay on the backward graph even on batches with no
        # supported rows.
        anchor = (mhr["pred_keypoints_3d"].sum() + mhr["pred_cam_t"].sum()
                  + out["contact"]["joint_logits"].sum()).to(self.dtype) * 0.0

        rows = gate_all.shape[0]
        seq_len = int(batch["seq_len"])
        if seq_len < 3 or rows % seq_len:
            return LossResult(
                terms=self._terms({"vel": (anchor, 0.0)}, anchor),
                scalars={"n_rows": 0.0},
                stats=self.empty_stats())
        n_clips = rows // seq_len

        world = predicted_keypoints_world(
            mhr, batch["cam_from_world"], self.keypoints, self.dtype)
        positions = world.reshape(n_clips, seq_len, len(KINDYN_GROUP_KEYPOINTS), 3)
        pos_sec = batch["frame_pos_sec"].to(self.device, self.dtype).reshape(
            n_clips, seq_len)
        dt = (pos_sec[:, 1:] - pos_sec[:, :-1]).mean(dim=1).clamp(
            min=1e-6)[:, None, None, None]
        speed = ((positions[:, 2:] - positions[:, :-2]) / (2.0 * dt)).norm(dim=-1)

        gate = gate_all.detach() if self.detach_gate else gate_all
        gate = gate.reshape(n_clips, seq_len, -1)[:, 1:-1]           # (n, T-2, 6)
        # The stencil at t reads frames t-1, t, t+1: each must be a real, valid
        # frame of the SAME clip (the world lift needs its extrinsics).
        ok = batch["frame_valid"].to(self.device).reshape(n_clips, seq_len)
        support = ok[:, :-2] & ok[:, 1:-1] & ok[:, 2:]               # (n, T-2)
        gated = gate * support[..., None]

        elementwise = F.smooth_l1_loss(
            speed, torch.zeros_like(speed), reduction="none",
            beta=self.huber_delta_ms)
        numerator = self.weight * (gated * elementwise).sum()
        mass = float((gate.detach() * support[..., None]).sum())

        with torch.no_grad():
            stats = torch.tensor(
                [float((gate.detach() * support[..., None] * speed).sum()), mass],
                dtype=torch.float64, device=self.device)
            supported = gate.detach()[support]                       # (rows, 6)
            scalars = {
                "vel_ms": mean_from_stats(float(stats[0]), mass),
                "gate_mean": float(supported.mean()) if supported.numel() else 0.0,
                "n_rows": float(support.sum()),
            }
        return LossResult(terms=self._terms({"vel": (numerator, mass)}, anchor),
                          scalars=scalars, stats=stats)

    def metrics(self, stats: Tensor) -> dict[str, float]:
        return {"vel_ms": mean_from_stats(float(stats[0]), float(stats[1]))}


__all__ = ["ContactConsistencyLoss"]
