"""Root-wrench residual of a world SMPL-X trajectory under the contact set's forces.

The refined world trajectory is a rigid-body motion of a BetterHuman SMPL-X body;
inverse dynamics (BetterRobot's RNEA) says which root wrench that motion requires
under gravity, and the contact slots' forces are the only external forces available
to supply it. The residual of that balance at the free-flyer root,

    tau_root = M(q) a + b(q, v) + g(q) - J^T f_ext          (should be zero)

is used twice: as the ``force_consistency`` loss and, under
``model.refiner.residual_feedback``, as a per-layer feedback channel of the
refiner (layer k + 1 sees what layer k's forces left unbalanced).

* **Body**: the 22-joint SMPL-X (``use_hands=False`` — fingers are massless and
  would only add finite-difference noise), shape-baked per clip
  (``compute_mass=True``: per-part mesh inertias, total ~72.6 kg neutral).
* **Motion**: world pelvis / root rotation / parent-local joints → BetterHuman
  ``q`` (91), optionally Gaussian-smoothed first (``smooth_sec``), then manifold
  central differences at the clip's real frame spacing for ``v`` and ``a``
  (``model.difference`` — the free-flyer twist is body-local, as RNEA wants). The
  acceleration is the ±2-frame stencil, so a period-2 wobble does not enter it.
* **Forces**: world forces in body-weight units, scaled to newtons by ``m g``, each
  applied at its slot's own world point and accumulated on the slot's parent body-22
  joint as the wrench ``[R_j^T f, R_j^T ((p - p_j) x f)]`` — the force plus the moment
  of the lever from the joint origin to the contact point. A ``kindyn6`` slot IS its
  joint, so its moment vanishes and the balance is the round 8-11 one.
* **Residual**: ``tau[..., :3] / (m g)`` (body weights) and ``tau[..., 3:6] /
  (m g · 1 m)``, both in the root frame; rows need the ±2 stencil inside a valid
  run, clip ends carry none. The 21 body joints' torques ``tau[..., 6:] / (m g · 1 m)``
  come back with it — what the force allocation costs the body's own actuators.
"""
from __future__ import annotations

import dataclasses

import better_robot as br
import roma
import torch
from torch import Tensor

from model.refiner import gaussian_smooth, smooth_rotations, stencil_valid
from utils.geometry import smplx_q

#: The 22 SMPL-X body joints under BetterRobot's names, in SMPL-X order (index 0 = the
#: pelvis = the free-flyer ``root``); a slot's parent-22 index indexes this tuple.
BODY_ROBOT_JOINTS = (
    "root", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist")
GRAVITY = 9.81
NUM_BODY_JOINTS = 22


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


class RootWrench:
    """The residual above on one BetterHuman dynamics body (built once per device)."""

    def __init__(self, model_path: str, device: torch.device | str,
                 dtype: torch.dtype = torch.float32) -> None:
        import better_human as bh
        self.device = torch.device(device)
        self.dtype = dtype
        self.body = bh.SMPLX(
            model_path=str(model_path), gender="neutral", num_betas=10, use_hands=False,
            use_face=False, compute_mass=True, dtype=dtype, device=self.device)
        robot = self.body.robot
        names = tuple(robot.joint_names)[1:]
        if names != BODY_ROBOT_JOINTS:
            raise AssertionError(
                f"the dynamics body's joints are {names}, not {BODY_ROBOT_JOINTS}")
        self.joint_ids = torch.tensor([robot.joint_id(n) for n in BODY_ROBOT_JOINTS],
                                      device=self.device)
        self.njoints = int(robot.njoints)

    def residual(self, pelvis: Tensor, root_rot: Tensor, body_rot: Tensor, betas: Tensor,
                 forces_world_bw: Tensor, points_world: Tensor, parent22: Tensor,
                 gravity_down: Tensor, seconds: Tensor, valid: Tensor,
                 smooth_sec: float = 0.0) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Root-wrench residual of one batch of clips.

        :param pelvis: ``(n, T, 3)`` world pelvis; ``root_rot`` ``(n, T, 3, 3)``
            world-from-root; ``body_rot`` ``(n, T, 21, 3, 3)`` parent-local.
        :param betas: ``(n, 10)`` per clip.
        :param forces_world_bw: ``(n, T, K, 3)`` world forces in body weights.
        :param points_world: ``(n, T, K, 3)`` world point each force acts at.
        :param parent22: ``(K,)`` long — the body-22 joint each slot hangs from
            (:attr:`~model.contact_frames.ContactSet.parent_joint22`).
        :param gravity_down: ``(n, 3)`` unit down vector (world).
        :param seconds: ``(n, T)``; ``valid`` ``(n, T)`` bool.
        :param smooth_sec: Gaussian sigma (s) applied to the motion first (0 = none).
        :returns: ``(force residual (n, T, 3) in bw, torque residual (n, T, 3) in
            bw·m, rows (n, T) bool, joint torques (n, T, nv - 6) in bw·m)`` —
            residuals in the root frame, joint torques in BetterHuman ``q`` DOF order.
        """
        n, t = seconds.shape
        if smooth_sec > 0.0:
            pelvis = gaussian_smooth(pelvis, seconds, valid, smooth_sec)
            root_rot = smooth_rotations(root_rot, seconds, valid, smooth_sec)
            body_rot = smooth_rotations(body_rot, seconds, valid, smooth_sec)
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
        slot_ids = self.joint_ids[parent22.to(self.joint_ids.device)]        # (K,) robot ids
        pose = fk.joint_pose_world[..., slot_ids, :]                         # (n, T, K, 7)
        rot_wj = roma.unitquat_to_rotmat(pose[..., 3:7])                     # (n, T, K, 3, 3)
        f_world = forces_world_bw * mass_g[..., None]                        # newtons
        moment_world = torch.cross(points_world - pose[..., :3], f_world, dim=-1)
        to_joint = rot_wj.transpose(-1, -2)
        f_local = (to_joint @ f_world.unsqueeze(-1)).squeeze(-1)             # joint frame
        m_local = (to_joint @ moment_world.unsqueeze(-1)).squeeze(-1)
        fext = torch.zeros(n, t, self.njoints, 6, dtype=q.dtype, device=q.device)
        fext = fext.index_add(2, slot_ids, torch.cat([f_local, m_local], dim=-1))
        tau = br.rnea(robot, q, v, a, fext=fext)                             # (n, T, nv)
        rows = stencil_valid(valid, 2)
        return tau[..., :3] / mass_g, tau[..., 3:6] / mass_g, rows, tau[..., 6:] / mass_g

    def dof_weights(self, table: dict[str, float]) -> Tensor:
        """``(nv - 6,)`` per-DOF multiplier of the joint torques from a joint-GROUP table.

        The group of a joint is its name without side and index (``left_hip`` -> ``hip``,
        ``spine2`` -> ``spine``); every body joint must be covered.
        """
        robot = self.body.robot
        idx_vs, nvs = [int(i) for i in robot.idx_vs], [int(n) for n in robot.nvs]
        weight = torch.zeros(int(robot.nv) - 6, device=self.device, dtype=self.dtype)
        for joint_id, name in enumerate(robot.joint_names):
            if idx_vs[joint_id] < 6:
                continue                                                 # universe and the root
            group = name.removeprefix("left_").removeprefix("right_").rstrip("0123456789")
            if group not in table:
                raise ValueError(f"joint torque multipliers: no entry for joint group {group!r} "
                                 f"(joint {name!r}); table has {sorted(table)}")
            weight[idx_vs[joint_id] - 6: idx_vs[joint_id] - 6 + nvs[joint_id]] = float(table[group])
        return weight


__all__ = ["BODY_ROBOT_JOINTS", "GRAVITY", "RootWrench", "trajectory_derivatives"]
