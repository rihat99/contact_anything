"""Newton consistency: the predicted contact forces must explain the predicted motion.

The linear half of the physics statement, written so the body mass cancels. With
the forces already in body-weight units, Newton's second law on the root reads::

    a_world / g  =  g_hat_world  +  sum_i f_i^world          (all sides dimensionless)

so the residual minimised here is::

    r = a_world / g  -  g_hat_world  -  R_world<-root . sum_i f_i^root     [bw]

``a_world`` is the second difference of the PREDICTED world root position (hip
keypoints plus ``pred_cam_t``, lifted with the dataset extrinsics),
``g_hat_world`` is kindyn's fitted unit DOWN direction, and the forces are the
six-group head output in the kindyn body-root frame. Sanity check: a climber
hanging still has ``a = 0`` and holds one body weight, ``sum f^world ~ -g_hat``,
so ``||r|| ~ 0``.

Unlike the RNEA physics loss this is a LINEAR balance only — no inertia, no
rotational half, no MHR body, hence no dependence on the MHR archive. What it
buys instead is a coupling the supervised force loss cannot provide: a force
prediction inconsistent with the observed motion costs, and so does a pose
trajectory whose acceleration no predicted contact can explain. Gradients reach
the force head and the pose path.

``motion_rot`` and ``gravity_world`` are kindyn GT and detached: the rotation
that expresses root-frame forces in the world is never itself fitted — an
unrailed rotation would be the cheapest way to satisfy the balance.

The second difference needs the full stencil (``t-1, t, t+1``), so clip
boundaries are never supervised; rows additionally need valid frames, kindyn root
coverage and force validity. Optional pre-smoothing runs a windowed mean over the
world positions before differencing — a second difference of a noisy trajectory
is the loudest signal in this objective, which is also why the term ramps in
(:meth:`ForceConsistencyLoss.weight_scale`) instead of switching on cold.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from model.loss import Loss, LossResult
from utils.geometry import predicted_root_world, windowed_mean
from utils.metrics import mean_from_stats

#: Standard gravity (m/s^2) — the scale that turns an acceleration into body
#: weights. The fitted per-scene ``gravity_world`` supplies the DIRECTION only.
GRAVITY_MS2 = 9.81


class ForceConsistencyLoss(Loss):
    """Linear Newton residual between the predicted forces and the predicted motion."""

    name = "force_consistency"
    stat_names = ("residual_num", "residual_mass")

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        super().__init__(cfg, model, device)
        fc = cfg["force_consistency"]
        self.weight = float(fc["loss"]["residual"])
        self.term_names = ("residual",)
        self.huber_delta_bw = float(fc["loss"]["huber_delta_bw"])
        self.ramp_start = int(fc["ramp"]["start_epoch"])
        self.ramp_epochs = int(fc["ramp"]["epochs"])
        self.smoothing_kernel = torch.tensor(
            [float(w) for w in fc["smoothing_kernel"]],
            dtype=torch.float64, device=self.device)

    def weight_scale(self, epoch: int) -> float:
        """Linear warm-up: 0 before ``ramp.start_epoch``, full ``ramp.epochs`` later."""
        return min(1.0, max(
            0.0, (epoch + 1 - self.ramp_start) / self.ramp_epochs))

    def __call__(self, out: dict, batch: dict, *, train: bool) -> LossResult:
        forces = out["force"]["joint_forces"].to(self.device, self.dtype)
        mhr = out["mhr"]
        anchor = (mhr["pred_keypoints_3d"].sum() + mhr["pred_cam_t"].sum()
                  + forces.sum()).to(self.dtype) * 0.0

        rows = forces.shape[0]
        seq_len = int(batch["seq_len"])
        if seq_len < 3 or rows % seq_len:
            return LossResult(
                terms=self._terms({"residual": (anchor, 0.0)}, anchor),
                scalars={"n_rows": 0.0}, stats=self.empty_stats())
        n_clips = rows // seq_len

        root_world, _ = predicted_root_world(mhr, batch["cam_from_world"])
        positions = windowed_mean(
            root_world.reshape(n_clips, seq_len, 3), self.smoothing_kernel)
        pos_sec = batch["frame_pos_sec"].to(self.device, torch.float64).reshape(
            n_clips, seq_len)
        dt = (pos_sec[:, 1:] - pos_sec[:, :-1]).mean(dim=1).clamp(
            min=1e-6)[:, None, None]
        acceleration = (positions[:, 2:] - 2.0 * positions[:, 1:-1]
                        + positions[:, :-2]) / dt.square()        # (n, T-2, 3) m/s^2

        rot_world_from_root = batch["motion_rot"].to(
            self.device, self.dtype).detach()                          # (B, 3, 3)
        force_world = torch.einsum(
            "bij,bj->bi", rot_world_from_root, forces.sum(dim=1))      # (B, 3) bw
        gravity = batch["gravity_world"].to(self.device, self.dtype).detach()

        interior = slice(1, seq_len - 1)
        residual = (acceleration.to(self.dtype) / GRAVITY_MS2
                    - gravity.reshape(n_clips, seq_len, 3)[:, interior]
                    - force_world.reshape(n_clips, seq_len, 3)[:, interior])

        ok = (batch["frame_valid"].to(self.device)
              & batch["motion_root_valid"].to(self.device)
              & batch["force_valid"].to(self.device)).reshape(n_clips, seq_len)
        support = (ok[:, :-2] & ok[:, 1:-1] & ok[:, 2:]).to(self.dtype)

        row_loss = F.smooth_l1_loss(
            residual, torch.zeros_like(residual), reduction="none",
            beta=self.huber_delta_bw).sum(dim=-1)                      # (n, T-2)
        numerator = self.weight * (row_loss * support).sum()
        mass = float(support.sum())

        with torch.no_grad():
            stats = torch.tensor(
                [float((residual.norm(dim=-1) * support).sum()), mass],
                dtype=torch.float64, device=self.device)
        scalars = {"residual_bw": mean_from_stats(float(stats[0]), mass),
                   "n_rows": mass}
        return LossResult(
            terms=self._terms({"residual": (numerator, mass)}, anchor),
            scalars=scalars, stats=stats)

    def metrics(self, stats: Tensor) -> dict[str, float]:
        return {"residual_bw": mean_from_stats(float(stats[0]), float(stats[1]))}


__all__ = ["ForceConsistencyLoss", "GRAVITY_MS2"]
