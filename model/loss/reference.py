"""Gaussian reference: an annealed pull of the refined body toward a fixed kernel.

The learned-smoothing round runs the refiner with its input Gaussians OFF
(``root_smooth_sec`` / ``pose_smooth_sec`` 0) — the temporal block itself has
to produce a smooth body through its ``pose`` offset head. A zero-initialised
head starts at the RAW per-frame body (lifted jitter ~105 against a GT floor of
~6.5), and the derivative losses alone have to discover a filter before they
can reward one.

This term hands it the answer for the first part of the run and then takes it
away: the reference is the Gaussian-smoothed refiner INPUT (the world root
position at ``root_sigma_sec``, the world root rotation and the 21
parent-local joint rotations at ``pose_sigma_sec``, all detached), the same
fixed kernel the round-4 / round-5 recipe used as the body itself, and the
three terms measure how far the REFINED body is from it:

* ``root`` — Huber on the world root position, metres (``huber_delta_root``);
* ``rot`` — geodesic angle of the world root rotation, radians;
* ``joints`` — geodesic angle of the 21 parent-local joint rotations, radians,
  averaged over the joints.

All three carry the same annealed weight: ``weight`` up to ``anneal_start``,
linearly down to 0 at ``anneal_end``, 0 after (fractions of the run's optimizer
steps, read from :attr:`~model.loss.Loss.progress`, which the trainer sets).
The end of the schedule is the point: the Gaussian is a starting basin, never
the objective — past ``anneal_end`` only the derivative and position losses
score the body, so the block is free to beat the kernel it was seeded from.

Metrics: ``root_mm`` / ``rot_deg`` / ``joints_deg`` — the RMS distance to the
reference in each channel (the drift away from the Gaussian, whatever the
weight is doing) — and ``weight``, the schedule's current value.
"""
from __future__ import annotations

import math

import roma
import torch
import torch.nn.functional as F
from torch import Tensor

from model.loss import Loss, LossResult
from model.refiner import NUM_BODY_JOINTS, gaussian_smooth, smooth_rotations


def geodesic_angle(a: Tensor, b: Tensor) -> Tensor:
    """Angle (rad) between two stacks of rotations ``(..., 3, 3)``.

    The norm of ``log(a^T b)``, floored inside the square root so a pair that
    coincides exactly has a zero — not a NaN — gradient.
    """
    vec = roma.rotmat_to_rotvec(a.transpose(-1, -2) @ b)
    return vec.square().sum(dim=-1).clamp(min=1e-12).sqrt()


class GaussianReferenceLoss(Loss):
    """Annealed distance between the refined body and the Gaussian of the refiner's input."""

    name = "gaussian_reference"
    term_names = ("root", "rot", "joints")
    stat_names = ("root_se", "root_n", "rot_se", "rot_n", "joints_se", "joints_n",
                  "weight_sum", "weight_n")

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        super().__init__(cfg, model, device)
        section = cfg["gaussian_reference"]
        self.root_sigma = float(section["root_sigma_sec"])
        self.pose_sigma = float(section["pose_sigma_sec"])
        self.weight = float(section["weight"])
        self.anneal_start = float(section["anneal_start"])
        self.anneal_end = float(section["anneal_end"])
        self.delta_root = float(section["huber_delta_root"])

    def current_weight(self) -> float:
        """The schedule at :attr:`~model.loss.Loss.progress`."""
        progress = float(self.progress)
        if progress <= self.anneal_start:
            return self.weight
        if progress >= self.anneal_end:
            return 0.0
        return self.weight * (self.anneal_end - progress) / (self.anneal_end - self.anneal_start)

    def reference(self, smplx: dict, batch: dict) -> tuple[Tensor, Tensor, Tensor]:
        """The Gaussian-smoothed refiner INPUT body, detached.

        :returns: ``(world root position (B, 3), world root rotation (B, 3, 3),
            parent-local joint rotations (B, 21, 3, 3))``.
        """
        n_frames = smplx["pelvis_world_in"].shape[0]
        seq_len = int(batch["seq_len"])
        n_clips = n_frames // seq_len
        seconds = batch["frame_pos_sec"].to(self.device, self.dtype).view(n_clips, seq_len)
        valid = batch["frame_valid"].to(self.device).view(n_clips, seq_len)
        with torch.no_grad():
            pelvis = smplx["pelvis_world_in"].detach().to(self.device, self.dtype)
            root = smplx["root_rot_world_in"].detach().to(self.device, self.dtype)
            joints = smplx["body_rot_in"].detach().to(self.device, self.dtype)
            pelvis = gaussian_smooth(pelvis.view(n_clips, seq_len, 3), seconds, valid,
                                     self.root_sigma)
            root = smooth_rotations(root.view(n_clips, seq_len, 3, 3), seconds, valid,
                                    self.pose_sigma)
            joints = smooth_rotations(
                joints.view(n_clips, seq_len, NUM_BODY_JOINTS - 1, 3, 3), seconds, valid,
                self.pose_sigma)
        return (pelvis.reshape(n_frames, 3), root.reshape(n_frames, 3, 3),
                joints.reshape(n_frames, NUM_BODY_JOINTS - 1, 3, 3))

    def __call__(self, out: dict, batch: dict, *, train: bool) -> LossResult:
        smplx = out["smplx"]
        pelvis = smplx["pelvis_world"].to(self.device, self.dtype)
        root = smplx["root_rot_world"].to(self.device, self.dtype)
        joints = smplx["body_rot"].to(self.device, self.dtype)
        pelvis_ref, root_ref, joints_ref = self.reference(smplx, batch)
        rows = batch["frame_valid"].to(self.device).to(self.dtype)
        mass = float(rows.sum())
        weight = self.current_weight()

        root_pos = F.smooth_l1_loss(pelvis, pelvis_ref, reduction="none",
                                    beta=self.delta_root).mean(dim=-1)
        root_ang = geodesic_angle(root_ref, root)
        joint_ang = geodesic_angle(joints_ref, joints)                       # [B, 21]
        raw = {"root": (weight * (root_pos * rows).sum(), mass),
               "rot": (weight * (root_ang * rows).sum(), mass),
               "joints": (weight * (joint_ang.mean(dim=-1) * rows).sum(), mass)}
        anchor = (pelvis.sum() + root.sum() + joints.sum()) * 0.0

        with torch.no_grad():
            stats = torch.tensor([
                float((((pelvis - pelvis_ref) ** 2).sum(dim=-1) * rows).sum()), mass,
                float((root_ang.detach() ** 2 * rows).sum()), mass,
                float(((joint_ang.detach() ** 2).mean(dim=-1) * rows).sum()), mass,
                weight, 1.0], dtype=torch.float64, device=self.device)
        return LossResult(terms=self._terms(raw, anchor),
                          scalars={"weight": weight}, stats=stats)

    def metrics(self, stats: Tensor) -> dict[str, float]:
        root_se, root_n, rot_se, rot_n, joint_se, joint_n, weight, batches = (
            float(v) for v in stats)
        return {"root_mm": 1000.0 * math.sqrt(root_se / max(root_n, 1.0)),
                "rot_deg": math.degrees(math.sqrt(rot_se / max(rot_n, 1.0))),
                "joints_deg": math.degrees(math.sqrt(joint_se / max(joint_n, 1.0))),
                "weight": weight / max(batches, 1.0)}


__all__ = ["GaussianReferenceLoss", "geodesic_angle"]
