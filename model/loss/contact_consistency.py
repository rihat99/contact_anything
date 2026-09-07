"""Contact stillness: the refined extremities must not move while labelled in contact.

Reads ``out["smplx"]["joints_world"]`` (the refiner's world joints) at the six
extremity joints of the kindyn groups — wrists 20 / 21, big toes 10 / 11, heels
7 / 8 in :data:`~model.loss.KINDYN_GROUP_NAMES` order — and penalises their
world SPEED (raw central finite difference over the clip's real frame spacing,
m/s) on limb-frames whose contact label is positive: an L1 weighted by
``contact_valid * contact_conf`` (confidence off with ``confidence_weights:
false``). A contact label in this corpus is a motion-gated "stable contact"
(stillness with hysteresis in the estimator), so the target of zero speed is
the label's own definition; the GT's residual in-contact speed (0.13 m/s mean on
train, heavy-tailed) is reported as the floor.

Gradient reaches the pose path only (the labels are data, not the contact
head), so this is a stillness prior on the refined pose, never a way to lower
the loss by predicting less contact. Rows need a central stencil (both
neighbours valid); clip ends carry no term.

Metrics: ``speed`` — mean predicted in-contact extremity speed (m/s, unweighted
by confidence), ``gt_speed`` — the same on the kindyn GT joints.
"""
from __future__ import annotations

import torch
from torch import Tensor

from model.loss import Loss, LossResult
from model.refiner import stencil_valid, time_derivative
from utils.metrics import mean_from_stats

#: SMPL-X body joint of each kindyn group (LH, RH, LF toe, RF toe, LA heel, RA heel).
GROUP_JOINTS = (20, 21, 10, 11, 7, 8)


class ContactConsistencyLoss(Loss):
    """L1 on the in-contact extremity speed of the refined pose."""

    name = "contact_consistency"
    term_names = ("still",)
    stat_names = ("speed_num", "speed_mass", "gt_speed_num", "gt_speed_mass")

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        super().__init__(cfg, model, device)
        section = cfg["contact_consistency"]
        self.weight = float(section["weight"])
        self.use_confidence = bool(section["confidence_weights"])

    def _speed(self, joints_world: Tensor, batch: dict) -> tuple[Tensor, Tensor]:
        """``(speed (B, 6) m/s, rows (B,) bool)`` of the six extremity joints."""
        seq_len = int(batch["seq_len"])
        n_frames = joints_world.shape[0]
        n_clips = n_frames // seq_len
        seconds = batch["frame_pos_sec"].to(self.device, self.dtype).view(n_clips, seq_len)
        valid = batch["frame_valid"].to(self.device).view(n_clips, seq_len)
        points = joints_world[:, list(GROUP_JOINTS)].view(n_clips, seq_len, len(GROUP_JOINTS), 3)
        speed = time_derivative(points, seconds, valid).norm(dim=-1).reshape(n_frames, -1)
        return speed, stencil_valid(valid, 1).reshape(n_frames)

    def __call__(self, out: dict, batch: dict, *, train: bool) -> LossResult:
        joints = out["smplx"]["joints_world"].to(self.device, self.dtype)
        anchor = joints.sum() * 0.0
        speed, rows = self._speed(joints, batch)
        contact = (batch["contact_gt"].to(self.device, self.dtype) > 0.5)
        weight = batch["contact_valid"].to(self.device, self.dtype) * contact.to(self.dtype)
        if self.use_confidence:
            weight = weight * batch["contact_conf"].to(self.device, self.dtype)
        weight = weight * rows.to(self.dtype)[:, None]
        raw = {"still": (self.weight * (speed * weight).sum(), float(weight.sum()))}

        with torch.no_grad():
            count = (contact & batch["contact_valid"].to(self.device).bool() & rows[:, None]).to(self.dtype)
            gt_rows = count * (batch["smplx_valid"].to(self.device).to(self.dtype))[:, None]
            gt_speed, _ = self._speed(batch["smplx_joints_world"].to(self.device, self.dtype), batch)
            stats = torch.tensor([
                float((speed.detach() * count).sum()), float(count.sum()),
                float((gt_speed * gt_rows).sum()), float(gt_rows.sum()),
            ], dtype=torch.float64, device=self.device)
        return LossResult(terms=self._terms(raw, anchor),
                          scalars={"n_rows": float(weight.sum())}, stats=stats)

    def metrics(self, stats: Tensor) -> dict[str, float]:
        return {"speed": mean_from_stats(float(stats[0]), float(stats[1])),
                "gt_speed": mean_from_stats(float(stats[2]), float(stats[3]))}


__all__ = ["ContactConsistencyLoss", "GROUP_JOINTS"]
