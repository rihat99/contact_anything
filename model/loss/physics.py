"""Physics loss: smooth MHR trajectory -> v, a -> RNEA root-wrench residual + regularizers.

Supervises the six kindyn-group force predictions without force labels. Per
batch (flat clip-major ``B = n_clips * T``):

1. **Eligibility** — physics applies to clips with ``T >= min_frames`` and every
   frame valid (``frame_valid`` already means tracked *and* camera-valid).
   Optionally clips whose sampled camera centre jumps more than
   ``physics.max_cam_jump_m`` are dropped as reconstruction discontinuities.
2. **q trajectory** — :class:`~model.loss.physics_adapter.MHRAdapter` maps the
   per-frame MHR params + camera extrinsics to a world-frame ``q`` (detached).
3. **Smoothing** — composed on the manifold: a linear windowed mean for the root
   translation + 125 revolute channels, and a hemisphere-aligned slerp mean for
   the root quaternion.
4. **v, a** — manifold central differences honouring the per-interval ``dt`` from
   ``frame_pos_sec``.
5. **Residual frames** — a frame contributes only when its full stencil is inside
   the clip: kernel radius ``r`` per side for smoothing plus two frames for the
   doubled central difference, i.e. ``{t : 2 + r <= t <= T - 3 - r}``.
6. **fext** — ``f_newtons = pred * m * g`` placed at the six group joint origins
   (zero torque), converted from the head's frame (``model.force.frame``) to
   world and then into each joint's local frame.
7. **RNEA** — gravity is per clip (``physics.gravity * gravity_world``, angular
   zero). ``tau[..., :6]`` is the root residual, ``tau[..., 6:]`` joint torques.

Every term is dimensionless: the residual force is normalised by ``m*g``, the
residual/joint torques by ``m*g*1m``, and the predicted forces are already in
body-weight units. On every call — including fully ineligible batches — each
numerator carries ``joint_forces.sum() * 0`` so the force parameters stay in the
autograd graph under ``find_unused_parameters=False``.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from torch import Tensor

import better_robot as br
from better_robot.lie import so3
from better_robot.tasks import Trajectory, smooth_trajectory

from model.loss import NUM_KINDYN_GROUPS, Loss, LossResult
from model.loss.physics_adapter import MHRAdapter
from utils.geometry import windowed_mean
from utils.metrics import mean_from_stats

if TYPE_CHECKING:
    from model.network import ContactAnything

#: ``out["mhr"]`` keys the adapter consumes.
_MHR_KEYS = ("mhr_model_params", "shape", "pred_cam_t")

#: Loss-term names, in report order.
_TERM_NAMES = (
    "residual",
    "force_noncontact",
    "force_at_contact",
    "force_smooth",
    "force_l2",
    "torque_l2",
    "torque_smooth",
)

#: Terms that need the RNEA call (and therefore at least one residual frame).
_RNEA_TERMS = ("residual", "torque_l2", "torque_smooth")

#: Terms gated by the predicted contact probability.
_GATED_TERMS = ("force_noncontact", "force_at_contact")

_NORM_EPS = 1.0e-12


def _pseudo_huber(x: Tensor, delta: float) -> Tensor:
    """Component-wise pseudo-Huber ρ_δ(x) = δ²·(sqrt(1 + (x/δ)²) − 1).

    Quadratic near 0 (ρ → ½x² as ``x → 0``) and linear far out (ρ ≈ δ·|x| − δ²),
    with a smooth, finite gradient everywhere including ``x = 0`` (unlike the
    classic Huber, whose second derivative is discontinuous at δ). ``δ`` is in the
    same dimensionless units as the residual (force / m·g, torque / m·g·1 m).

    :param x: residual components (any shape).
    :param delta: transition scale δ > 0.
    :returns: ρ_δ(x), same shape as ``x``.
    """
    scaled = x / delta
    return (delta * delta) * (torch.sqrt(1.0 + scaled * scaled) - 1.0)


def _residual_frame_indices(seq_len: int, kernel_radius: int) -> list[int]:
    """Residual frame indices whose full smoothing + double-difference stencil fits.

    :param seq_len: clip length ``T``.
    :param kernel_radius: smoothing half-width ``r`` (``len(kernel) // 2``).
    :returns: sorted indices ``{t : 2 + r <= t <= T - 3 - r}`` (possibly empty).
    """
    lo = 2 + kernel_radius
    hi = seq_len - 3 - kernel_radius
    return list(range(lo, hi + 1)) if hi >= lo else []


def _smooth_configuration(q: Tensor, t_sec: Tensor, kernel: Tensor) -> Tensor:
    """Manifold-compose smoothing over the free-flyer + revolute configuration.

    :param q: ``(n_clips, T, 132)`` free-flyer ``[t(3), quat(4)]`` then 125 pose.
    :param t_sec: ``(n_clips, T)`` frame timestamps (for the ``Trajectory`` pair).
    :param kernel: ``(L,)`` weights; ``L == 1`` is a no-op.
    :returns: ``(n_clips, T, 132)`` smoothed configuration.
    """
    if kernel.numel() == 1:
        return q
    euclidean = torch.cat((q[..., 0:3], q[..., 7:]), dim=-1)    # translation + pose
    euclidean = windowed_mean(euclidean, kernel)
    quaternion = smooth_trajectory(Trajectory(t=t_sec, q=q[..., 3:7]), kernel, kind="so3").q
    return torch.cat((euclidean[..., 0:3], quaternion, euclidean[..., 3:]), dim=-1)


def _trajectory_derivatives(model: br.Model, q: Tensor, t_sec: Tensor) -> tuple[Tensor, Tensor]:
    """q-aligned manifold central-difference velocity and acceleration, non-uniform dt.

    Interior frames use central differences; the two boundary frames use one-sided
    estimates (discarded — never residual frames). Requires ``T >= 3``.

    :param q: ``(n_clips, T, 132)`` configuration.
    :param t_sec: ``(n_clips, T)`` timestamps.
    :returns: ``(velocity, acceleration)`` each ``(n_clips, T, nv)``.
    """
    dt_adjacent = (t_sec[..., 1:] - t_sec[..., :-1]).unsqueeze(-1)      # (., T-1, 1)
    dt_central = (t_sec[..., 2:] - t_sec[..., :-2]).unsqueeze(-1)       # (., T-2, 1)
    adjacent = model.difference(q[..., :-1, :], q[..., 1:, :]) / dt_adjacent
    central = model.difference(q[..., :-2, :], q[..., 2:, :]) / dt_central
    velocity = torch.cat((adjacent[..., :1, :], central, adjacent[..., -1:, :]), dim=-2)

    central_a = (velocity[..., 2:, :] - velocity[..., :-2, :]) / dt_central
    delta_v = velocity[..., 1:, :] - velocity[..., :-1, :]
    edge0 = delta_v[..., :1, :] / dt_adjacent[..., :1, :]
    edge1 = delta_v[..., -1:, :] / dt_adjacent[..., -1:, :]
    acceleration = torch.cat((edge0, central_a, edge1), dim=-2)
    return velocity, acceleration


class PhysicsLoss(Loss):
    """RNEA root-wrench physics loss over the six kindyn-group force predictions.

    :param cfg: resolved run config; reads ``physics.*``, ``model.force.frame``
        and ``mhr_body.*``.
    :param model: the built :class:`~model.network.ContactAnything` (branch and
        token-count validation only; no parameters are read).
    :param device: device for the MHR body and the physics compute.
    """

    name = "physics"
    stat_names = ("raw_residual_num", "raw_residual_mass")

    def __init__(
        self,
        cfg: dict,
        model: "ContactAnything",
        device: torch.device | str = "cuda",
        adapter: MHRAdapter | None = None,
    ) -> None:
        super().__init__(cfg, model, device)
        net = self.model
        physics = cfg["physics"]
        self.frame = str(cfg["model"]["force"]["frame"])
        if self.frame not in ("root", "local_world_aligned", "local"):
            raise ValueError(f"unknown model.force.frame {self.frame!r}")
        self.use_warp = bool(physics["use_warp"])
        self.gravity = float(physics["gravity"])
        self.min_frames = int(physics["min_frames"])
        self.smoothing_kernel = [float(w) for w in physics["smoothing_kernel"]]
        self.kernel_radius = len(self.smoothing_kernel) // 2
        loss_cfg = physics["loss"]
        self.weights = {name: float(loss_cfg[name]) for name in _TERM_NAMES}
        self.f_min = float(loss_cfg["contact_min_bw"])
        # Robust residual: the ``residual`` TERM applies ρ component-wise to the
        # six normalised root-wrench residual components. ``square`` (ρ(x)=x²) is
        # the plain quadratic objective; ``pseudo_huber`` down-weights the heavy
        # tail of the double finite-differencing with a linear tail past δ.
        robust = loss_cfg["residual_robust"]
        self.residual_kind = str(robust["kind"])
        if self.residual_kind not in ("square", "pseudo_huber"):
            raise ValueError(
                f"unknown physics.loss.residual_robust.kind {self.residual_kind!r}")
        self.delta_force = float(robust["delta_force"])
        self.delta_torque = float(robust["delta_torque"])
        # Separate weights on the force / torque parts of the residual objective:
        # the per-limb allocation signal lives in the torque part (~20x weaker than
        # the force sum), so weighting torque up rebalances. ``raw_residual`` — the
        # reported headline — stays unweighted either way. ``gate_frames``
        # restricts the prob-gated force terms to the residual (centre) frames,
        # where a windowed temporal contact model's probs are in-distribution.
        self.residual_force_weight = float(loss_cfg["residual_force_weight"])
        self.residual_torque_weight = float(loss_cfg["residual_torque_weight"])
        self.gate_frames = str(loss_cfg["gate_frames"])
        if self.gate_frames not in ("all", "residual"):
            raise ValueError(f"unknown physics.loss.gate_frames {self.gate_frames!r}")
        # Non-contact penalty form: ``soft_l2`` is the (1 − p)·‖f‖² shrinkage tax —
        # quadratic in ‖f‖, so its stationary point against the residual pull is
        # ‖f‖ ∝ 1/(1 − p), nonzero on every limb. ``hinge_l1`` penalises ‖f‖ with a
        # hinge gate on p — full below ``p_lo``, zero above ``p_hi``, linear
        # between — whose constant slope at ‖f‖ → 0 admits an exact zero-force
        # solution on confidently-free limbs while leaving p ≥ p_hi limbs untaxed.
        gate_cfg = loss_cfg["noncontact_gate"]
        self.noncontact_kind = str(gate_cfg["kind"])
        if self.noncontact_kind not in ("soft_l2", "hinge_l1"):
            raise ValueError(
                f"unknown physics.loss.noncontact_gate.kind {self.noncontact_kind!r}")
        self.noncontact_p_lo = float(gate_cfg["p_lo"])
        self.noncontact_p_hi = float(gate_cfg["p_hi"])
        # Camera-jerk clip filter: drop clips whose per-frame camera-centre jump
        # exceeds this metric threshold (reconstruction discontinuities alias into
        # body acceleration). ``None`` = off.
        jump = physics["max_cam_jump_m"]
        self.max_cam_jump_m = None if jump is None else float(jump)

        self.active_terms = tuple(n for n in _TERM_NAMES if self.weights[n] != 0.0)
        self.term_names = self.active_terms
        if not self.active_terms:
            raise ValueError("physics.enabled but every physics.loss.* weight is 0")
        self.runs_rnea = any(self.weights[n] != 0.0 for n in _RNEA_TERMS)
        self.needs_probs = any(self.weights[n] != 0.0 for n in _GATED_TERMS)

        self.num_groups = NUM_KINDYN_GROUPS
        if net.force_tokens is None:
            raise ValueError("physics.enabled requires model.force.enabled")
        if net.force_tokens.num_tokens != self.num_groups:
            raise ValueError(
                f"physics supervises the {self.num_groups} kindyn groups but the force "
                f"branch has {net.force_tokens.num_tokens} tokens")
        if self.needs_probs and net.head_contact is None:
            raise ValueError(
                "physics.loss.force_noncontact / force_at_contact gate on the predicted "
                "contact probabilities and therefore require model.contact.enabled")

        self.adapter = adapter if adapter is not None else MHRAdapter(
            model_path=cfg["mhr_body"]["model_path"], lod=int(cfg["mhr_body"]["lod"]),
            device=self.device, dtype=self.dtype)
        self._flip = torch.diag(
            torch.tensor([1.0, -1.0, -1.0], device=self.device, dtype=self.dtype))
        njoints = self.adapter.body.robot.njoints
        self._group_to_joint = torch.nn.functional.one_hot(
            self.adapter.group_joint_ids, njoints).to(dtype=self.dtype)   # (6, njoints)

    def __call__(self, out: dict, batch: dict, *, train: bool = True) -> LossResult:
        """Return the physics term set for one batch.

        Physics has no train-only filtering and no ramp, so ``train`` is ignored.

        :param out: model output — reads ``out["mhr"]``,
            ``out["force"]["joint_forces"] (B, 6, 3)`` (grads live), and
            ``out["contact"]["joint_probs"] (B, 6)`` (detached before use, so
            physics never trains the contact head).
        :param batch: reads ``seq_len``, ``frame_pos_sec (B)``, ``frame_valid (B)``,
            ``cam_from_world (B, 4, 4)``, ``gravity_world (B, 3)``, ``cam_jump_m (B)``
            and — for ``model.force.frame == "root"`` — ``motion_rot (B, 3, 3)``.
        """
        forces_all = out["force"]["joint_forces"].to(self.device, self.dtype)
        # Graph-connected zero touching every force param (DDP: force params reach
        # the loss only through this module under find_unused_parameters=False).
        zero_touch = forces_all.sum() * 0.0

        seq_len = int(batch["seq_len"])
        if forces_all.shape[0] % seq_len:
            raise ValueError(
                f"flat batch of {forces_all.shape[0]} rows is not a multiple of "
                f"seq_len {seq_len}")
        if forces_all.shape[1] != self.num_groups:
            raise ValueError(
                f"expected {self.num_groups} force groups, got {forces_all.shape[1]}")
        n_clips = forces_all.shape[0] // seq_len
        eligible, n_jerk_excluded = self._eligible_clips(batch, seq_len, n_clips)

        residual_frames = _residual_frame_indices(seq_len, self.kernel_radius)
        run_rnea = bool(residual_frames) and self.runs_rnea
        run_physics = bool(eligible.any()) and (
            run_rnea or self.weights["force_smooth"] != 0.0)

        terms: dict[str, tuple[Tensor, float]] = {}
        scalars: dict[str, float] = {
            "residual_force": 0.0,
            "residual_torque": 0.0,
            "residual_sat_frac": 0.0,
            "force_std": 0.0,
            "n_eligible_clips": float(int(eligible.sum())),
            "n_residual_frames": float(len(residual_frames)),
            "n_jerk_excluded_clips": float(n_jerk_excluded),
        }
        raw_residual = (0.0, 0.0)

        if eligible.any():
            self._force_gate_terms(out, eligible, seq_len, residual_frames, terms, scalars)
        if run_physics:
            raw_residual = self._physics_terms(
                out, batch, eligible, seq_len, n_clips, residual_frames, run_rnea,
                terms, scalars)

        return self._assemble(terms, zero_touch, scalars, raw_residual)

    def metrics(self, stats: Tensor) -> dict[str, float]:
        """Mass-weighted mean unweighted root-wrench residual from summed stats."""
        return {"residual": mean_from_stats(float(stats[0]), float(stats[1]))}

    # ------------------------------------------------------------ eligibility

    def _eligible_clips(self, batch: dict, seq_len: int, n_clips: int) -> tuple[Tensor, int]:
        """Per-clip eligibility ``(n_clips,)`` (cpu bool) and the jerk-exclusion count.

        The optional camera-jerk filter (``physics.max_cam_jump_m``) only *removes*
        clips: an otherwise-eligible clip whose max ``cam_jump_m`` — the camera-centre
        displacement between consecutive SAMPLED clip frames — exceeds the threshold
        becomes ineligible.
        """
        if seq_len < self.min_frames:
            return torch.zeros(n_clips, dtype=torch.bool), 0
        eligible = batch["frame_valid"].view(n_clips, seq_len).all(dim=1)
        n_jerk_excluded = 0
        if self.max_cam_jump_m is not None:
            cam_jump = batch["cam_jump_m"].view(n_clips, seq_len)
            within = cam_jump.max(dim=1).values <= self.max_cam_jump_m
            n_jerk_excluded = int((eligible & ~within).sum())
            eligible = eligible & within
        return eligible.cpu(), n_jerk_excluded

    # ------------------------------------------------------------ force gate

    def _force_gate_terms(
        self, out: dict, eligible: Tensor, seq_len: int, residual_frames: list[int],
        terms: dict[str, tuple[Tensor, float]], scalars: dict[str, float],
    ) -> None:
        """Contact-gated force terms on eligible clips.

        ``force_l2`` and the ``force_std`` diagnostic always span all frames. The
        prob-gated terms span all frames when ``physics.loss.gate_frames == "all"``
        and only the residual frames when ``"residual"`` — with a windowed temporal
        contact model only the residual (centre) frame's probs are in-distribution,
        so this avoids gating forces against off-window contact predictions.

        ``force_std`` is the across-clip std of each clip's mean head-frame ‖f‖: a
        collapsed force branch predicts a near-constant vector, so this is the
        cheapest online collapse detector. It is detached, never an objective.
        """
        rows = self._eligible_rows(eligible, seq_len)
        n_elig = int(eligible.sum())
        k = self.num_groups
        forces = out["force"]["joint_forces"].to(self.device, self.dtype)[rows] \
            .view(n_elig, seq_len, k, 3)

        magnitude_sq = (forces ** 2).sum(-1)                       # ||f||^2  (., T, 6)
        magnitude = (magnitude_sq + _NORM_EPS).sqrt()
        terms["force_l2"] = (magnitude_sq.sum(), float(n_elig * seq_len * k))
        if n_elig > 1:
            per_clip = magnitude.mean(dim=(1, 2))                  # (n_elig,)
            scalars["force_std"] = float(per_clip.std(unbiased=False).detach())

        if not self.needs_probs:
            return
        probs = out["contact"]["joint_probs"].detach().to(self.device, self.dtype)[rows] \
            .view(n_elig, seq_len, k)
        if self.gate_frames == "residual":
            if not residual_frames:
                return                                             # no in-window frame
            sel = torch.tensor(residual_frames, device=self.device)
            magnitude_sq = magnitude_sq.index_select(1, sel)
            magnitude = magnitude.index_select(1, sel)
            probs = probs.index_select(1, sel)
        gate_mass = float(n_elig * magnitude_sq.shape[1] * k)
        if self.noncontact_kind == "hinge_l1":
            hinge = ((self.noncontact_p_hi - probs)
                     / (self.noncontact_p_hi - self.noncontact_p_lo)).clamp(0.0, 1.0)
            terms["force_noncontact"] = ((hinge * magnitude).sum(), gate_mass)
        else:
            terms["force_noncontact"] = (((1.0 - probs) * magnitude_sq).sum(), gate_mass)
        terms["force_at_contact"] = (
            (probs * torch.relu(self.f_min - magnitude) ** 2).sum(), gate_mass)

    # --------------------------------------------------------------- physics

    def _physics_terms(
        self, out: dict, batch: dict, eligible: Tensor, seq_len: int, n_clips: int,
        residual_frames: list[int], run_rnea: bool,
        terms: dict[str, tuple[Tensor, float]], scalars: dict[str, float],
    ) -> tuple[float, float]:
        """Adapter -> smooth -> FK -> (force_smooth, RNEA residual + torque terms).

        :returns: the unweighted ``(raw_residual_numerator, mass)`` sufficient
            statistics (``(0, 0)`` when RNEA does not run).
        """
        rows = self._eligible_rows(eligible, seq_len)
        n_elig = int(eligible.sum())
        k = self.num_groups
        clip_idx = eligible.nonzero(as_tuple=False).flatten().to(self.device)

        forces = out["force"]["joint_forces"].to(self.device, self.dtype)[rows] \
            .view(n_elig, seq_len, k, 3)
        cam_flat = batch["cam_from_world"].detach().to(self.device, self.dtype)[rows]
        cam = cam_flat.view(n_elig, seq_len, 4, 4)
        t_sec = batch["frame_pos_sec"].to(self.device, self.dtype)[rows] \
            .view(n_elig, seq_len)
        gravity_world = batch["gravity_world"].detach().to(self.device, self.dtype) \
            .view(n_clips, seq_len, 3)[clip_idx][:, 0]                # per clip (., 3)

        mhr_out = {key: out["mhr"][key][rows] for key in _MHR_KEYS}
        body, q = self.adapter.q_from_mhr_out(mhr_out, cam_flat, n_elig, seq_len)
        mass = self.adapter.total_mass(body)                          # (n_elig,)
        kernel = torch.tensor(self.smoothing_kernel, device=self.device, dtype=self.dtype)
        q_smoothed = _smooth_configuration(q, t_sec, kernel)

        fk = br.forward_kinematics(body.robot, q_smoothed, use_warp=self.use_warp)
        group_pose = fk.joint_pose_world.index_select(-2, self.adapter.group_joint_ids)
        r_joint = so3.to_matrix(group_pose[..., 3:])                 # (., T, 6, 3, 3)
        r_w_c = cam[..., :3, :3].transpose(-1, -2)                   # (., T, 3, 3)
        r_w_root = self._world_from_root(batch, rows, n_elig, seq_len, fk)

        if self.weights["force_smooth"] != 0.0:
            world_forces = self._pred_to_world(forces, r_w_c, r_joint, r_w_root)
            delta = world_forces[:, 1:] - world_forces[:, :-1]
            terms["force_smooth"] = ((delta ** 2).sum(), float(n_elig * (seq_len - 1) * k))

        if not run_rnea:
            return 0.0, 0.0
        return self._residual_terms(
            body, q_smoothed, t_sec, forces, mass, gravity_world, r_w_c, r_joint,
            r_w_root, residual_frames, n_elig, terms, scalars)

    def _residual_terms(
        self, body, q_smoothed: Tensor, t_sec: Tensor, forces: Tensor, mass: Tensor,
        gravity_world: Tensor, r_w_c: Tensor, r_joint: Tensor, r_w_root: Tensor,
        residual_frames: list[int], n_elig: int,
        terms: dict[str, tuple[Tensor, float]], scalars: dict[str, float],
    ) -> tuple[float, float]:
        """RNEA residual + torque terms on the residual frames of eligible clips."""
        sel = torch.tensor(residual_frames, device=self.device)     # time axis is dim 1
        n_res = sel.numel()
        velocity, acceleration = _trajectory_derivatives(body.robot, q_smoothed, t_sec)

        q_res = q_smoothed.index_select(1, sel)
        v_res = velocity.index_select(1, sel)
        a_res = acceleration.index_select(1, sel)
        r_w_c_res = r_w_c.index_select(1, sel)
        r_joint_res = r_joint.index_select(1, sel)
        r_w_root_res = None if r_w_root is None else r_w_root.index_select(1, sel)

        mass_g = (mass * self.gravity).view(n_elig, 1, 1)
        fext = self._fext_from_head_newton(
            forces.index_select(1, sel) * mass_g.unsqueeze(-1),
            r_w_c_res, r_joint_res, r_w_root_res)

        grav_linear = self.gravity * gravity_world
        grav6 = torch.cat((grav_linear, torch.zeros_like(grav_linear)), dim=-1).unsqueeze(-2)
        robot = dataclasses.replace(
            body.robot,
            values=dataclasses.replace(body.robot.values, gravity=grav6))
        tau = br.rnea(
            robot, q_res, v_res, a_res, fext=fext, use_warp=self.use_warp
        )                                                             # (n_elig, n_res, nv)

        residual_force = tau[..., :3] / mass_g
        residual_torque = tau[..., 3:6] / mass_g
        joint_torque = tau[..., 6:] / mass_g
        force_sq = (residual_force ** 2).sum(-1)
        torque_sq = (residual_torque ** 2).sum(-1)
        res_mass = float(n_elig * n_res)

        # The ``residual`` objective term applies ρ component-wise and scales the
        # force / torque parts by ``residual_force_weight`` / ``residual_torque_weight``.
        # ``raw_residual`` (the reported metric) is ALWAYS the unweighted physical
        # residual — the comparable headline whose zero-force baseline is ≈ 2.586.
        physical_num = (force_sq + torque_sq).sum()
        w_f, w_tau = self.residual_force_weight, self.residual_torque_weight
        if self.residual_kind == "pseudo_huber":
            residual_num = (w_f * _pseudo_huber(residual_force, self.delta_force).sum()
                            + w_tau * _pseudo_huber(residual_torque, self.delta_torque).sum())
            saturated = int((residual_force.abs() > self.delta_force).sum()
                            + (residual_torque.abs() > self.delta_torque).sum())
            n_components = residual_force.numel() + residual_torque.numel()
            scalars["residual_sat_frac"] = saturated / max(n_components, 1)
        elif w_f == 1.0 and w_tau == 1.0:
            residual_num = physical_num
        else:
            residual_num = w_f * force_sq.sum() + w_tau * torque_sq.sum()
        terms["residual"] = (residual_num, res_mass)
        terms["torque_l2"] = ((joint_torque ** 2).sum(), res_mass)
        if n_res >= 2:
            joint_delta = joint_torque[:, 1:] - joint_torque[:, :-1]
            terms["torque_smooth"] = (
                (joint_delta ** 2).sum(), float(n_elig * (n_res - 1)))
        scalars["residual_force"] = float(force_sq.mean().detach())
        scalars["residual_torque"] = float(torque_sq.mean().detach())
        return float(physical_num.detach()), res_mass

    # ------------------------------------------------------------- helpers

    def _world_from_root(
        self, batch: dict, rows: Tensor, n_elig: int, seq_len: int, fk,
    ) -> Tensor | None:
        """World-from-root rotation ``(n_elig, T, 3, 3)`` for the ``root`` force frame.

        Prefers the kindyn GT ``motion_rot`` (the frame the supervised force loss and
        the Newton consistency loss express root forces in). Without it — physics does
        not itself request the ``motion`` signal group — the same rotation is read off
        the reconstructed MHR free-flyer: the pose converter defines the kindyn body
        root as the MHR root joint, so the two agree up to the model's own pose error.

        :returns: ``None`` for the frames that do not need it.
        """
        if self.frame != "root":
            return None
        # The kindyn GT world-from-root rotation (loaded because signal_needs
        # adds `motion` for physics in the root frame) — never the prediction's
        # own root, which the objective could otherwise rotate to satisfy.
        return batch["motion_rot"].detach().to(self.device, self.dtype)[rows] \
            .view(n_elig, seq_len, 3, 3)

    def _fext_from_head_newton(
        self, forces_newton: Tensor, r_w_c: Tensor, r_joint: Tensor,
        r_w_root: Tensor | None,
    ) -> Tensor:
        """Map per-group head-frame forces (newton) to the RNEA ``fext`` wrench.

        Head frame → world (:meth:`_pred_to_world`) → each group joint's LOCAL frame
        → scattered onto the full joint set as a zero-torque wrench.

        :param forces_newton: ``(., Tsel, 6, 3)`` group forces in newton.
        :param r_w_c: ``(., Tsel, 3, 3)`` world-from-camera rotation.
        :param r_joint: ``(., Tsel, 6, 3, 3)`` world group-joint rotation.
        :param r_w_root: ``(., Tsel, 3, 3)`` world-from-root rotation, or ``None``.
        :returns: ``(., Tsel, njoints, 6)`` external wrench for :func:`better_robot.rnea`.
        """
        world_newton = self._pred_to_world(forces_newton, r_w_c, r_joint, r_w_root)
        local_newton = (r_joint.transpose(-1, -2) @ world_newton.unsqueeze(-1)).squeeze(-1)
        by_joint = torch.einsum("...ci,cj->...ji", local_newton, self._group_to_joint)
        return torch.cat((by_joint, torch.zeros_like(by_joint)), dim=-1)

    def _pred_to_world(
        self, forces: Tensor, r_w_c: Tensor, r_joint: Tensor, r_w_root: Tensor | None,
    ) -> Tensor:
        """Map predicted group forces from the head frame to the world frame.

        ``local`` — the head predicts in each group joint's own frame, so the joint's
        world rotation is the whole map. ``root`` — the head predicts in the kindyn
        body-root frame (the supervised-force convention). ``local_world_aligned`` —
        the head predicts in the per-frame camera frame with SAM's y-up convention, so
        the axis flip ``D = diag(1, -1, -1)`` precedes ``R_w<-c``.

        :param forces: ``(., Tsel, 6, 3)`` in the head frame.
        :param r_w_c: ``(., Tsel, 3, 3)`` world-from-camera rotation.
        :param r_joint: ``(., Tsel, 6, 3, 3)`` world group-joint rotation.
        :param r_w_root: ``(., Tsel, 3, 3)`` world-from-root rotation, or ``None``.
        :returns: ``(., Tsel, 6, 3)`` world-frame forces.
        """
        if self.frame == "local":
            return (r_joint @ forces.unsqueeze(-1)).squeeze(-1)
        if self.frame == "root":
            return (r_w_root.unsqueeze(-3) @ forces.unsqueeze(-1)).squeeze(-1)
        f_cam = (self._flip @ forces.unsqueeze(-1)).squeeze(-1)
        return (r_w_c.unsqueeze(-3) @ f_cam.unsqueeze(-1)).squeeze(-1)

    def _eligible_rows(self, eligible: Tensor, seq_len: int) -> Tensor:
        """Flat clip-major row indices for the eligible clips ``(n_elig * T,)``."""
        clip_idx = eligible.nonzero(as_tuple=False).flatten().to(self.device)
        frames = torch.arange(seq_len, device=self.device)
        return (clip_idx.unsqueeze(-1) * seq_len + frames).flatten()

    def _assemble(
        self, terms: dict[str, tuple[Tensor, float]], zero_touch: Tensor,
        scalars: dict[str, float], raw_residual: tuple[float, float],
    ) -> LossResult:
        """Weight the numerators and package the fixed term set + sufficient stats.

        Every nonzero-weight term is always present (mass 0 when it has no data this
        batch), so the term set is fixed by config — not by per-batch eligibility —
        which the trainer needs for a consistent DDP all-reduce. Each numerator
        carries the graph-connected ``zero_touch`` so every force parameter stays in
        the autograd graph regardless of which terms have data.
        """
        weighted = {
            name: (self.weights[name] * terms[name][0], terms[name][1])
            if name in terms else (torch.zeros((), device=self.device), 0.0)
            for name in self.active_terms
        }
        stats = torch.tensor(raw_residual, dtype=torch.float64, device=self.device)
        return LossResult(
            terms=self._terms(weighted, zero_touch), scalars=scalars, stats=stats)


__all__ = ["PhysicsLoss"]
