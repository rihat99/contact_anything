"""Physics consistency of the refined motion, the predicted contacts and the predicted forces.

The refined world trajectory (:mod:`model.refiner` output) is a rigid-body
motion of a BetterHuman SMPL-X body; inverse dynamics (BetterRobot's RNEA) says
which root wrench that motion requires under gravity, and the six predicted
extremity forces are the only external forces available to supply it. The loss
is the residual of that balance at the free-flyer root:

    tau_root = M(q) a + b(q, v) + g(q) − J^T f_ext          (should be zero)

* **Body**: the 22-joint SMPL-X (``use_hands=False`` — fingers are massless and
  would only add finite-difference noise), shape-baked per clip from the
  refiner's clip-mean betas (``compute_mass=True``: per-part mesh inertias, total
  ~72.6 kg neutral).
* **Motion**: the refined ``pelvis_world`` / ``root_rot_world`` / ``body_rot``
  → BetterHuman ``q`` (91), optionally Gaussian-smoothed first (``smooth_sec``;
  rotations as matrix means projected onto SO(3)), then manifold central
  differences at the clip's real frame spacing for ``v`` and ``a``
  (``model.difference`` — the free-flyer twist is body-local, as RNEA wants).
* **Forces**: the six predicted forces (body-weight units in the refiner's input
  body frame ``out["force"]["frame"]``), optionally gated by the DETACHED predicted
  contact probability (``gate_by_contact``), scaled to newtons by ``m g``, rotated
  into each extremity joint's local frame and applied as a pure force at the
  joint origin (wrists 20/21, big toes 10/11, heels 7/8 — within ~5–8 cm of the
  true contact points; the lever's moment is a knowingly accepted error).
* **Gravity**: the scene's fitted unit down vector (``gravity_world``) × 9.81.
* **Residual**: ``tau[..., :3] / (m g)`` (body weights) and ``tau[..., 3:6] /
  (m g · 1 m)``, each through a pseudo-Huber and its own weight — the torque
  channel is where force ALLOCATION lives and was measured ~23× weaker than the
  force sum in the 2026-07 physics runs, hence the separate ``torque`` weight.

Gradient reaches the force head AND the pose path (``detach_pose: false``;
the position and derivative losses anchor the pose). Rows need the ±2 stencil
inside a valid run; clip ends carry no term.

Metrics: ``force`` / ``torque`` — mean residual magnitudes (bw, bw·m) of the
prediction, and ``gt_force`` / ``gt_torque`` — the same residual evaluated on the
kindyn GT trajectory with the kindyn GT forces (the floor this body model and
the joint-origin approximation allow).
"""
from __future__ import annotations

import dataclasses

import better_robot as br
import roma
import torch
from torch import Tensor

from model.loss import KINDYN_GROUP_NAMES, Loss, LossResult
from model.refiner import gaussian_smooth, smooth_rotations, stencil_valid
from utils.geometry import smplx_q
from utils.metrics import mean_from_stats

#: BetterRobot joint names of the six kindyn groups (SMPL-X wrists, big toes, heels).
GROUP_ROBOT_JOINTS = ("left_wrist", "right_wrist", "left_foot", "right_foot",
                      "left_ankle", "right_ankle")
GRAVITY = 9.81
NUM_BODY_JOINTS = 22


def pseudo_huber(x: Tensor, delta: float) -> Tensor:
    """``delta^2 (sqrt(1 + (x / delta)^2) - 1)``: quadratic near 0, linear far out, smooth."""
    return delta * delta * (torch.sqrt(1.0 + (x / delta) ** 2) - 1.0)


def trajectory_derivatives(model, q: Tensor, seconds: Tensor) -> tuple[Tensor, Tensor]:
    """Manifold central-difference velocity and acceleration at non-uniform spacing.

    :param q: ``(n, T, nq)``; ``seconds`` ``(n, T)``. Interior frames use central
        differences, the two boundary frames one-sided ones (never supervised).
    :returns: ``(v, a)`` each ``(n, T, nv)`` in the model's tangent layout.
    """
    dt_adjacent = (seconds[..., 1:] - seconds[..., :-1]).clamp(min=1e-6).unsqueeze(-1)
    dt_central = (seconds[..., 2:] - seconds[..., :-2]).clamp(min=1e-6).unsqueeze(-1)
    adjacent = model.difference(q[..., :-1, :], q[..., 1:, :]) / dt_adjacent
    central = model.difference(q[..., :-2, :], q[..., 2:, :]) / dt_central
    velocity = torch.cat((adjacent[..., :1, :], central, adjacent[..., -1:, :]), dim=-2)
    central_a = (velocity[..., 2:, :] - velocity[..., :-2, :]) / dt_central
    delta_v = velocity[..., 1:, :] - velocity[..., :-1, :]
    edge0 = delta_v[..., :1, :] / dt_adjacent[..., :1, :]
    edge1 = delta_v[..., -1:, :] / dt_adjacent[..., -1:, :]
    return velocity, torch.cat((edge0, central_a, edge1), dim=-2)


class ForceConsistencyLoss(Loss):
    """RNEA root-wrench residual of the refined motion under the predicted forces."""

    name = "force_consistency"
    term_names = ("force", "torque")
    stat_names = ("force_num", "mass", "torque_num", "gt_force_num", "gt_mass", "gt_torque_num")

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        super().__init__(cfg, model, device)
        import better_human as bh
        section = cfg["force_consistency"]
        self.sigma = float(section["smooth_sec"])
        self.detach_pose = bool(section["detach_pose"])
        self.gate = bool(section["gate_by_contact"])
        loss_cfg = section["loss"]
        self.weights = {"force": float(loss_cfg["force"]), "torque": float(loss_cfg["torque"])}
        self.delta = {"force": float(loss_cfg["huber_delta_force"]),
                      "torque": float(loss_cfg["huber_delta_torque"])}
        self.term_names = tuple(t for t in ("force", "torque") if self.weights[t] > 0.0)
        # The 22-joint dynamics body; fingers never enter the RNEA.
        self.body = bh.SMPLX(
            model_path=str(cfg["model"]["smplx"]["model_path"]), gender="neutral", num_betas=10,
            use_hands=False, use_face=False, compute_mass=True, dtype=self.dtype,
            device=self.device)
        robot = self.body.robot
        self.joint_ids = torch.tensor([robot.joint_id(n) for n in GROUP_ROBOT_JOINTS],
                                      device=self.device)
        self.njoints = int(robot.njoints)

    # ------------------------------------------------------------------ core

    def residual(self, pelvis: Tensor, root_rot: Tensor, body_rot: Tensor, betas: Tensor,
                 forces_world_bw: Tensor, gravity_down: Tensor, seconds: Tensor,
                 valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Root-wrench residual of one batch of clips.

        :param pelvis: ``(n, T, 3)`` world pelvis; ``root_rot`` ``(n, T, 3, 3)``
            world-from-root; ``body_rot`` ``(n, T, 21, 3, 3)`` parent-local.
        :param betas: ``(n, 10)`` per clip.
        :param forces_world_bw: ``(n, T, 6, 3)`` world forces in body weights.
        :param gravity_down: ``(n, 3)`` unit down vector (world).
        :param seconds: ``(n, T)``; ``valid`` ``(n, T)`` bool.
        :returns: ``(force residual (n, T, 3) in bw, torque residual (n, T, 3) in
            bw·m, rows (n, T) bool)``.
        """
        n, t = seconds.shape
        if self.sigma > 0.0:
            pelvis = gaussian_smooth(pelvis, seconds, valid, self.sigma)
            root_rot = smooth_rotations(root_rot, seconds, valid, self.sigma)
            body_rot = smooth_rotations(body_rot, seconds, valid, self.sigma)
        q = smplx_q(pelvis.reshape(n * t, 3), root_rot.reshape(n * t, 3, 3),
                    body_rot.reshape(n * t, NUM_BODY_JOINTS - 1, 3, 3)).view(n, t, -1)

        shaped = self.body.with_shape(betas=betas)                           # (n, ...) values
        values = shaped.robot.values
        robot = shaped.robot.with_values(
            joint_placements=values.joint_placements.unsqueeze(-3),        # (n, 1, J, 7)
            body_inertias=values.body_inertias.unsqueeze(-3))              # (n, 1, J, 10)
        gravity6 = torch.cat([GRAVITY * gravity_down, torch.zeros_like(gravity_down)], dim=-1)
        robot = dataclasses.replace(
            robot, values=dataclasses.replace(robot.values, gravity=gravity6[:, None]))
        mass = values.body_inertias[..., 0].sum(dim=-1)                      # (n,) kg
        mass_g = (mass * GRAVITY).view(n, 1, 1)

        v, a = trajectory_derivatives(robot, q, seconds)
        fk = br.forward_kinematics(robot, q)
        quat = fk.joint_pose_world[..., self.joint_ids, 3:7]                 # (n, T, 6, 4) xyzw
        rot_wj = roma.unitquat_to_rotmat(quat)                               # (n, T, 6, 3, 3)
        f_local = (rot_wj.transpose(-1, -2) @ (forces_world_bw * mass_g[..., None]).unsqueeze(-1)
                   ).squeeze(-1)                                             # newtons, joint frame
        fext = torch.zeros(n, t, self.njoints, 6, dtype=q.dtype, device=q.device)
        fext = fext.index_copy(2, self.joint_ids, torch.cat(
            [f_local, torch.zeros_like(f_local)], dim=-1))
        tau = br.rnea(robot, q, v, a, fext=fext)                             # (n, T, nv)
        rows = stencil_valid(valid, 2)
        return tau[..., :3] / mass_g, tau[..., 3:6] / mass_g, rows

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
