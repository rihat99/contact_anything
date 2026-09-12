"""Physics consistency of the refined motion, the predicted contacts and the predicted forces.

The root-wrench residual itself lives in :mod:`model.physics` (:class:`~model.physics.RootWrench`,
shared with the refiner's ``residual_feedback``): the refined world trajectory is a rigid-body
motion of a BetterHuman SMPL-X body, inverse dynamics says which root wrench that motion
requires under gravity, and the six predicted extremity forces are the only external forces
available to supply it. This loss puts a pseudo-Huber on the force (bw) and torque (bw·m)
parts of that residual with separate weights — the torque channel is where force ALLOCATION
lives and was measured ~23× weaker than the force sum in the 2026-07 physics runs.

* **Motion**: the refiner's ``pelvis_world`` / ``root_rot_world`` / ``body_rot``, optionally
  Gaussian-smoothed first (``smooth_sec``).
* **Forces**: the six predicted forces (body-weight units in the refiner's input body frame
  ``out["force"]["frame"]``), optionally gated by the DETACHED predicted contact probability
  (``gate_by_contact``).
* **Gravity**: the scene's fitted unit down vector (``gravity_world``) — supervision, so the
  corpus value is used even when the refiner predicts its own.

Gradient reaches the force head AND the pose path (``detach_pose: false``;
the position and derivative losses anchor the pose). Rows need the ±2 stencil
inside a valid run; clip ends carry no term.

Metrics: ``force`` / ``torque`` — mean residual magnitudes (bw, bw·m) of the
prediction, and ``gt_force`` / ``gt_torque`` — the same residual evaluated on the
kindyn GT trajectory with the kindyn GT forces (the floor this body model and
the joint-origin approximation allow).
"""
from __future__ import annotations

import torch
from torch import Tensor

from model.loss import Loss, LossResult
from model.physics import GROUP_ROBOT_JOINTS, RootWrench, trajectory_derivatives
from utils.metrics import mean_from_stats


def pseudo_huber(x: Tensor, delta: float) -> Tensor:
    """``delta^2 (sqrt(1 + (x / delta)^2) - 1)``: quadratic near 0, linear far out, smooth."""
    return delta * delta * (torch.sqrt(1.0 + (x / delta) ** 2) - 1.0)


class ForceConsistencyLoss(Loss):
    """RNEA root-wrench residual of the refined motion under the predicted forces."""

    name = "force_consistency"
    term_names = ("force", "torque")
    stat_names = ("force_num", "mass", "torque_num", "gt_force_num", "gt_mass", "gt_torque_num")

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        super().__init__(cfg, model, device)
        section = cfg["force_consistency"]
        self.sigma = float(section["smooth_sec"])
        self.detach_pose = bool(section["detach_pose"])
        self.gate = bool(section["gate_by_contact"])
        loss_cfg = section["loss"]
        self.weights = {"force": float(loss_cfg["force"]), "torque": float(loss_cfg["torque"])}
        self.delta = {"force": float(loss_cfg["huber_delta_force"]),
                      "torque": float(loss_cfg["huber_delta_torque"])}
        self.term_names = tuple(t for t in ("force", "torque") if self.weights[t] > 0.0)
        self.wrench = RootWrench(cfg["model"]["smplx"]["model_path"], self.device, self.dtype)

    # ------------------------------------------------------------------ core

    @property
    def body(self):
        """The 22-joint dynamics body (:attr:`~model.physics.RootWrench.body`)."""
        return self.wrench.body

    def residual(self, pelvis: Tensor, root_rot: Tensor, body_rot: Tensor, betas: Tensor,
                 forces_world_bw: Tensor, gravity_down: Tensor, seconds: Tensor,
                 valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """:meth:`~model.physics.RootWrench.residual` under the loss's ``smooth_sec``."""
        return self.wrench.residual(pelvis, root_rot, body_rot, betas, forces_world_bw,
                                    gravity_down, seconds, valid, smooth_sec=self.sigma)

    def _clip_inputs(self, batch: dict):
        seq_len = int(batch["seq_len"])
        n_frames = batch["frame_valid"].shape[0]
        n = n_frames // seq_len
        seconds = batch["frame_pos_sec"].to(self.device, self.dtype).view(n, seq_len)
        valid = batch["frame_valid"].to(self.device).view(n, seq_len)
        gravity = batch["gravity_world"].to(self.device, self.dtype).view(n, seq_len, 3)[:, 0]
        return n, seq_len, seconds, valid, gravity

    def __call__(self, out: dict, batch: dict, *, train: bool) -> LossResult:
        n, t, seconds, valid, gravity = self._clip_inputs(batch)
        smplx, force = out["smplx"], out["force"]
        forces_bw = force["forces"].to(self.device, self.dtype)              # (B, 6, 3) body frame
        anchor = forces_bw.sum() * 0.0
        frame = force["frame"].to(self.device, self.dtype)                   # world-from-body
        if self.gate:
            forces_bw = forces_bw * out["contact"]["probs"].detach().to(self.device, self.dtype)[..., None]
        forces_world = torch.einsum("bij,bkj->bki", frame, forces_bw).view(n, t, 6, 3)
        pelvis = smplx["pelvis_world"].to(self.device, self.dtype)
        root_rot = smplx["root_rot_world"].to(self.device, self.dtype)
        body_rot = smplx["body_rot"].to(self.device, self.dtype)
        betas = smplx["betas"].to(self.device, self.dtype).view(n, t, -1)[:, 0]
        if self.detach_pose:
            pelvis, root_rot, body_rot, betas = (x.detach() for x in (pelvis, root_rot, body_rot, betas))
        else:
            anchor = anchor + (pelvis.sum() + root_rot.sum() + body_rot.sum()) * 0.0
        res_f, res_t, rows = self.residual(
            pelvis.view(n, t, 3), root_rot.view(n, t, 3, 3), body_rot.view(n, t, -1, 3, 3),
            betas, forces_world, gravity, seconds, valid)
        mask = rows.to(self.dtype)
        mass = float(mask.sum())
        raw = {
            "force": (self.weights["force"] * (pseudo_huber(res_f, self.delta["force"]).mean(-1) * mask).sum(), mass),
            "torque": (self.weights["torque"] * (pseudo_huber(res_t, self.delta["torque"]).mean(-1) * mask).sum(), mass),
        }
        raw = {k: v for k, v in raw.items() if k in self.term_names}

        with torch.no_grad():
            stats = [float((res_f.norm(dim=-1) * mask).sum()), mass,
                     float((res_t.norm(dim=-1) * mask).sum())]
            stats += self._gt_stats(batch, n, t, seconds, valid, gravity)
        return LossResult(terms=self._terms(raw, anchor), scalars={"n_rows": mass},
                          stats=torch.tensor(stats, dtype=torch.float64, device=self.device))

    def _gt_stats(self, batch: dict, n: int, t: int, seconds: Tensor, valid: Tensor,
                  gravity: Tensor) -> list[float]:
        """The residual of the kindyn GT motion under the kindyn GT forces (reference floor)."""
        joints = batch["smplx_joints_world"].to(self.device, self.dtype)
        root_rot = batch["smplx_root_rot"].to(self.device, self.dtype)
        gt_forces = batch["force_gt"].to(self.device, self.dtype)            # bw, GT root frame
        gt_forces = gt_forces * batch["force_contact"].to(self.device).to(self.dtype)[..., None]
        forces_world = torch.einsum("bij,bkj->bki", root_rot, gt_forces).view(n, t, 6, 3)
        gt_valid = (batch["smplx_valid"] & batch["force_valid"]).to(self.device).view(n, t) & valid
        res_f, res_t, rows = self.residual(
            joints[:, 0].view(n, t, 3), root_rot.view(n, t, 3, 3),
            batch["smplx_body_rot"].to(self.device, self.dtype).view(n, t, -1, 3, 3),
            batch["smplx_betas"].to(self.device, self.dtype).view(n, t, -1)[:, 0],
            forces_world, gravity, seconds, gt_valid)
        mask = rows.to(self.dtype)
        return [float((res_f.norm(dim=-1) * mask).sum()), float(mask.sum()),
                float((res_t.norm(dim=-1) * mask).sum())]

    def metrics(self, stats: Tensor) -> dict[str, float]:
        return {"force": mean_from_stats(float(stats[0]), float(stats[1])),
                "torque": mean_from_stats(float(stats[2]), float(stats[1])),
                "gt_force": mean_from_stats(float(stats[3]), float(stats[4])),
                "gt_torque": mean_from_stats(float(stats[5]), float(stats[4]))}


__all__ = ["ForceConsistencyLoss", "GROUP_ROBOT_JOINTS", "pseudo_huber", "trajectory_derivatives"]
