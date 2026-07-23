"""Physics loss: smooth MHR trajectory -> v, a -> RNEA root-wrench residual + regularizers.

Supervises the four extremity force predictions without force labels (plan
``README.md`` §3, D5-D9, D12). Per batch (flat clip-major ``B = n_clips * T``):

1. **Eligibility** — physics applies to video clips with ``T >= min_frames``, all
   frames valid, and all frames camera-valid. Still images (``T = 1``) and clips
   with an invalid frame contribute zero. A clip that is otherwise eligible but
   lacks the step-02 camera export (``cam_valid=False``) raises (never a silent
   no-op).
2. **q trajectory** — the step-03 :class:`MHRAdapter` maps the frozen model's
   per-frame MHR params + camera extrinsics to a world-frame ``q`` (detached).
3. **Smoothing** — composed on the manifold: a linear windowed mean for the root
   translation + 125 revolute channels, and a hemisphere-aligned slerp mean for
   the root quaternion (``smooth_trajectory`` cannot take the composite 132-d q).
4. **v, a** — manifold central differences honouring the per-interval ``dt`` from
   ``frame_pos_sec``.
5. **Residual frames** — a frame contributes to the RNEA residual only when its
   full stencil is inside the clip: kernel radius ``r`` per side for smoothing,
   plus two frames for the doubled central difference (velocity central needs
   ``+/-1``; acceleration central of velocity needs another ``+/-1``). So the
   residual frame indices are ``{t : 2 + r <= t <= T - 3 - r}``. For ``T = 5,
   kernel = [1]`` (``r = 0``) this is ``{2}`` (1 frame); for ``T = 7, kernel =
   [1]`` it is ``{2, 3, 4}`` (3 frames). The default kernel ``[0.25, 0.5, 0.25]``
   (``r = 1``) needs ``T >= 7`` for any residual frame.
6. **fext** — ``f_newtons = pred * m * g`` (D5) placed at the four extremity joint
   origins (zero torque). Frame conversion per ``model.force_head.frame`` (D6):
   ``local_world_aligned`` maps the per-frame camera y-up prediction through
   ``D = diag(1, -1, -1)`` then ``R_w<-c`` (world) then the joint rotation
   transpose (local); ``local`` passes straight through.
7. **RNEA** — gravity is per clip (``physics.gravity * gravity_world``, angular
   zero) set on the shaped values without mutating global state. ``tau[..., :6]``
   is the root residual; ``tau[..., 6:]`` the joint torques.

Every loss term is dimensionless (D12): the residual force is normalised by
``m*g``, the residual/joint torques by ``m*g*1m``; the predicted forces are
already in body-weight units. Each term returns ``(weighted_numerator, mass)`` so
the trainer (step 07) forms the exact global DDP mean with
:func:`contact.losses.ddp_global_mean_term`. On every call — including fully
ineligible batches — the returned graph carries ``joint_forces.sum() * 0`` so all
force params stay in the autograd graph under ``find_unused_parameters=False``
(mirrors ``safe_logits`` in ``contact/losses.py``).
"""
from __future__ import annotations

import dataclasses
from typing import Any

import torch
from torch import Tensor

import better_robot as br
from better_robot.lie import so3
from better_robot.tasks import Trajectory, smooth_trajectory

from .adapter import MHRAdapter

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

_NORM_EPS = 1.0e-12


def _pseudo_huber(x: Tensor, delta: float) -> Tensor:
    """Component-wise pseudo-Huber ρ_δ(x) = δ²·(sqrt(1 + (x/δ)²) − 1).

    Quadratic near 0 (ρ → ½x² as ``x → 0``) and linear far out (ρ ≈ δ·|x| − δ²),
    with a smooth, finite gradient everywhere including ``x = 0`` (unlike the classic
    Huber, whose second derivative is discontinuous at δ). ``δ`` is in the same
    dimensionless units as the residual (force / m·g, torque / m·g·1 m).

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


def _linear_windowed_mean(x: Tensor, kernel: Tensor) -> Tensor:
    """Kernel-smooth ``x`` along the time axis with edge replication.

    :param x: ``(..., T, C)`` Euclidean channels.
    :param kernel: ``(L,)`` odd-length non-negative weights (normalised here).
    :returns: ``(..., T, C)`` smoothed; edge frames use replicated boundaries.
    """
    weights = kernel / kernel.sum()
    num_knots = x.shape[-2]
    radius = kernel.numel() // 2
    offsets = torch.arange(-radius, radius + 1, device=x.device)
    centers = torch.arange(num_knots, device=x.device).unsqueeze(-1)
    indices = (centers + offsets).clamp(0, num_knots - 1)      # (T, L)
    windows = x[..., indices, :]                                # (..., T, L, C)
    return (windows * weights.view(-1, 1)).sum(-2)


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
    euclidean = _linear_windowed_mean(euclidean, kernel)
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


class PhysicsLoss:
    """RNEA root-wrench physics loss over the four extremity force predictions.

    :param cfg: resolved run config; reads ``physics.*`` and
        ``model.force_head.frame``.
    :param device: device for the MHR body and physics compute.
    :param dtype: floating dtype (float32).
    :param adapter: optional pre-built :class:`MHRAdapter` (test injection); a
        fresh one is built from ``physics.model_path`` / ``physics.lod`` otherwise.
    """

    def __init__(
        self,
        cfg: dict,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
        adapter: MHRAdapter | None = None,
    ) -> None:
        physics = cfg["physics"]
        self.frame = str(cfg["model"]["force_head"]["frame"])
        self.use_warp = bool(physics["use_warp"])
        self.gravity = float(physics["gravity"])
        self.min_frames = int(physics["min_frames"])
        self.smoothing_kernel = [float(w) for w in physics["smoothing_kernel"]]
        self.kernel_radius = len(self.smoothing_kernel) // 2
        loss_cfg = physics["loss"]
        self.weights = {name: float(loss_cfg[name]) for name in _TERM_NAMES}
        self.f_min = float(loss_cfg["contact_min_bw"])
        # Robust residual (§1): the ``residual`` TERM applies ρ component-wise to the
        # six normalised root-wrench residual components. ``square`` (ρ(x)=x²) is
        # bit-identical to the original objective; ``pseudo_huber`` down-weights the
        # heavy tail (noisy double finite-differencing) with a linear tail past δ.
        # Read via ``.get`` so pre-existing configs/fixtures without the block keep
        # the exact square path.
        robust = loss_cfg.get("residual_robust") or {}
        self.residual_kind = str(robust.get("kind", "square"))
        self.delta_force = float(robust.get("delta_force", 1.0))
        self.delta_torque = float(robust.get("delta_torque", 0.5))
        # Camera-jerk clip filter (§2): drop clips whose per-frame camera-center jump
        # exceeds this metric threshold (reconstruction discontinuities alias into
        # body acceleration). ``None`` = off (default, backwards-identical).
        self.max_cam_jump_m = physics.get("max_cam_jump_m")
        if self.max_cam_jump_m is not None:
            self.max_cam_jump_m = float(self.max_cam_jump_m)

        self.adapter = adapter if adapter is not None else MHRAdapter(
            model_path=physics["model_path"], lod=int(physics["lod"]),
            device=device, dtype=dtype)
        self.device = self.adapter.device
        self.dtype = dtype
        self._flip = torch.diag(
            torch.tensor([1.0, -1.0, -1.0], device=self.device, dtype=dtype))
        njoints = self.adapter.body.robot.njoints
        self._contact_to_joint = torch.nn.functional.one_hot(
            self.adapter.extremity_joint_ids, njoints).to(dtype=dtype)   # (4, njoints)

    def __call__(self, out: dict, batch: dict) -> tuple[Tensor, dict[str, Any]]:
        return self.forward(out, batch)

    def forward(self, out: dict, batch: dict) -> tuple[Tensor, dict[str, Any]]:
        """Return ``(total, parts)``.

        :param out: frozen forward output — reads ``out["mhr"]``,
            ``out["force"]["joint_forces"] (B, 4, 3)`` (grads live), and
            ``out["contact"]["joint_probs"] (B, 4)`` (detached before use, D8).
        :param batch: reads ``seq_len``, ``frame_pos_sec (B)``, ``frame_valid (B)``,
            ``cam_from_world (B, 4, 4)``, ``gravity_world (B, 3)``, ``cam_valid (B)``.
        :returns: ``(total, parts)`` where ``parts["terms"][name]`` carries
            ``weighted_numerator_tensor`` + ``weight_mass`` for exact DDP reduction
            and the remaining ``parts`` keys are detached scalars for logging.
        """
        forces_all = out["force"]["joint_forces"].to(self.device, self.dtype)
        # Graph-connected zero touching every force param (DDP: force params reach
        # the loss only through this module under find_unused_parameters=False).
        zero_touch = forces_all.sum() * 0.0

        seq_len = int(batch["seq_len"])
        n_clips = forces_all.shape[0] // seq_len
        eligible, n_jerk_excluded = self._eligible_clips(batch, seq_len, n_clips)

        residual_frames = _residual_frame_indices(seq_len, self.kernel_radius)
        run_rnea = bool(residual_frames) and any(
            self.weights[name] for name in ("residual", "torque_l2", "torque_smooth"))
        run_physics = eligible.any() and (run_rnea or self.weights["force_smooth"] != 0.0)

        terms: dict[str, tuple[Tensor, float]] = {}
        # Detached logging scalars (+ the ``raw_residual`` headline entry, which the
        # trainer/eval read for the physics_residual monitor; overwritten by
        # ``_residual_terms`` when RNEA runs, else left as a mass-0 no-op).
        diagnostics: dict[str, Any] = {
            "residual_force": 0.0, "residual_torque": 0.0,
            "residual_sat_frac": 0.0, "force_std": 0.0,
            "n_jerk_excluded_clips": int(n_jerk_excluded),
            "raw_residual": {
                "weighted_numerator_tensor": zero_touch.detach(),
                "weight_mass": 0.0, "loss": 0.0},
        }

        if eligible.any():
            self._force_gate_terms(out, eligible, seq_len, terms, diagnostics)
        if run_physics:
            self._physics_terms(
                out, batch, eligible, seq_len, n_clips, residual_frames, run_rnea,
                terms, diagnostics)

        return self._assemble(terms, zero_touch, eligible, residual_frames, diagnostics)

    # ------------------------------------------------------------ eligibility

    def _eligible_clips(
        self, batch: dict, seq_len: int, n_clips: int,
    ) -> tuple[Tensor, int]:
        """Per-clip eligibility ``(n_clips,)`` and the camera-jerk exclusion count.

        Raises on the missing-camera guard (a stale export must never become a
        silent no-op). The optional camera-jerk filter (``physics.max_cam_jump_m``)
        is applied *after* the guard and only *removes* clips (never raises): an
        otherwise-eligible clip whose max ``cam_jump_m`` — the camera-center
        displacement between consecutive SAMPLED clip frames (stride-consistent,
        see ``climbing_videos.py``) — exceeds the threshold becomes ineligible.
        Returns ``(eligible, n_jerk_excluded)``.
        """
        if seq_len < self.min_frames:
            return torch.zeros(n_clips, dtype=torch.bool), 0
        frame_valid = batch["frame_valid"].view(n_clips, seq_len)
        cam_valid = batch["cam_valid"].view(n_clips, seq_len)
        frames_ok = frame_valid.all(dim=1)
        cams_ok = cam_valid.all(dim=1)
        misconfigured = frames_ok & ~cams_ok
        if bool(misconfigured.any()):
            clip = int(misconfigured.nonzero(as_tuple=False).flatten()[0])
            raise ValueError(
                f"physics-eligible clip {clip} (T={seq_len} >= min_frames, frames "
                "valid) has cam_valid=False — the dataset lacks the step-02 camera "
                "export; re-export with extrinsics or make the clip physics-ineligible")
        eligible = frames_ok & cams_ok
        n_jerk_excluded = 0
        if self.max_cam_jump_m is not None and "cam_jump_m" in batch:
            cam_jump = batch["cam_jump_m"].view(n_clips, seq_len)
            within = cam_jump.max(dim=1).values <= self.max_cam_jump_m
            jerk_excluded = eligible & ~within
            n_jerk_excluded = int(jerk_excluded.sum().item())
            eligible = eligible & within
        return eligible.cpu(), n_jerk_excluded

    # ------------------------------------------------------------ force gate

    def _force_gate_terms(
        self, out: dict, eligible: Tensor, seq_len: int,
        terms: dict[str, tuple[Tensor, float]], diagnostics: dict[str, Any],
    ) -> None:
        """Contact-gated force terms on all frames of eligible clips (D8).

        Also records ``force_std`` (§3): the across-clip std of each clip's mean
        head-frame ‖f‖ (mean over its frames and four extremities). A collapsed
        force branch predicts a near-constant vector, so this is the cheapest online
        collapse detector; it is a detached scalar, never part of the objective.
        """
        rows = self._eligible_rows(eligible, seq_len)
        n_elig = int(eligible.sum().item())
        forces = out["force"]["joint_forces"].to(self.device, self.dtype)[rows] \
            .view(n_elig, seq_len, 4, 3)
        probs = out["contact"]["joint_probs"].detach().to(self.device, self.dtype)[rows] \
            .view(n_elig, seq_len, 4)

        magnitude_sq = (forces ** 2).sum(-1)                       # ||f||^2  (., T, 4)
        magnitude = (magnitude_sq + _NORM_EPS).sqrt()
        gate_mass = float(n_elig * seq_len * 4)
        terms["force_noncontact"] = (((1.0 - probs) * magnitude_sq).sum(), gate_mass)
        terms["force_at_contact"] = (
            (probs * torch.relu(self.f_min - magnitude) ** 2).sum(), gate_mass)
        terms["force_l2"] = (magnitude_sq.sum(), gate_mass)
        if n_elig > 1:
            per_clip = magnitude.mean(dim=(1, 2))                  # (n_elig,)
            diagnostics["force_std"] = float(per_clip.std(unbiased=False).detach())

    # --------------------------------------------------------------- physics

    def _physics_terms(
        self, out: dict, batch: dict, eligible: Tensor, seq_len: int, n_clips: int,
        residual_frames: list[int], run_rnea: bool,
        terms: dict[str, tuple[Tensor, float]], diagnostics: dict[str, Any],
    ) -> None:
        """Adapter -> smooth -> FK -> (force_smooth, RNEA residual + torque terms)."""
        rows = self._eligible_rows(eligible, seq_len)
        n_elig = int(eligible.sum().item())
        clip_idx = eligible.nonzero(as_tuple=False).flatten().to(self.device)

        forces = out["force"]["joint_forces"].to(self.device, self.dtype)[rows] \
            .view(n_elig, seq_len, 4, 3)
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
        ext_pose = fk.joint_pose_world.index_select(-2, self.adapter.extremity_joint_ids)
        r_joint = so3.to_matrix(ext_pose[..., 3:])                   # (., T, 4, 3, 3)
        r_w_c = cam[..., :3, :3].transpose(-1, -2)                   # (., T, 3, 3)

        if self.weights["force_smooth"] != 0.0:
            world_forces = self._pred_to_world(forces, r_w_c, r_joint)
            delta = world_forces[:, 1:] - world_forces[:, :-1]
            terms["force_smooth"] = ((delta ** 2).sum(), float(n_elig * (seq_len - 1) * 4))

        if run_rnea:
            self._residual_terms(
                body, q_smoothed, t_sec, forces, mass, gravity_world, r_w_c, r_joint,
                residual_frames, n_elig, terms, diagnostics)

    def _residual_terms(
        self, body, q_smoothed: Tensor, t_sec: Tensor, forces: Tensor, mass: Tensor,
        gravity_world: Tensor, r_w_c: Tensor, r_joint: Tensor, residual_frames: list[int],
        n_elig: int, terms: dict[str, tuple[Tensor, float]], diagnostics: dict[str, Any],
    ) -> None:
        """RNEA residual + torque terms on the residual frames of eligible clips."""
        sel = torch.tensor(residual_frames, device=self.device)     # time axis is dim 1
        n_res = sel.numel()
        velocity, acceleration = _trajectory_derivatives(body.robot, q_smoothed, t_sec)

        q_res = q_smoothed.index_select(1, sel)
        v_res = velocity.index_select(1, sel)
        a_res = acceleration.index_select(1, sel)
        r_w_c_res = r_w_c.index_select(1, sel)
        r_joint_res = r_joint.index_select(1, sel)

        mass_g = (mass * self.gravity).view(n_elig, 1, 1)
        fext = self._fext_from_head_newton(
            forces.index_select(1, sel) * mass_g.unsqueeze(-1), r_w_c_res, r_joint_res)

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

        # The ``residual`` objective term applies ρ component-wise (§1). ``square``
        # reproduces the original ``force_sq + torque_sq`` sum bit-for-bit; the raw
        # physical residual below is emitted unconditionally as the comparable
        # headline (zero-force baseline ≈ 2.586).
        physical_num = (force_sq + torque_sq).sum()
        if self.residual_kind == "pseudo_huber":
            residual_num = (_pseudo_huber(residual_force, self.delta_force).sum()
                            + _pseudo_huber(residual_torque, self.delta_torque).sum())
            saturated = int((residual_force.abs() > self.delta_force).sum()
                            + (residual_torque.abs() > self.delta_torque).sum())
            n_components = residual_force.numel() + residual_torque.numel()
            diagnostics["residual_sat_frac"] = saturated / max(n_components, 1)
        else:
            residual_num = physical_num
        terms["residual"] = (residual_num, res_mass)
        diagnostics["raw_residual"] = {
            "weighted_numerator_tensor": physical_num.detach(),
            "weight_mass": res_mass,
            "loss": float((physical_num / max(res_mass, 1.0)).detach()),
        }
        terms["torque_l2"] = ((joint_torque ** 2).sum(), res_mass)
        if n_res >= 2:
            joint_delta = joint_torque[:, 1:] - joint_torque[:, :-1]
            terms["torque_smooth"] = (
                (joint_delta ** 2).sum(), float(n_elig * (n_res - 1)))
        diagnostics["residual_force"] = float(force_sq.mean().detach())
        diagnostics["residual_torque"] = float(torque_sq.mean().detach())

    # ---------------------------------------------------------- evaluation

    @torch.no_grad()
    def diagnostics(self, out: dict, batch: dict) -> dict[str, Tensor] | None:
        """Per-frame force plausibility tensors for eligible clips (eval only).

        Companion to :meth:`forward` for ``scripts/evaluate.py``: returns raw
        per-(frame, extremity) predicted-force magnitudes (body weight + newton),
        contact probabilities, and the per-frame world-vertical force sum (body
        weight, positive up). :meth:`forward` still supplies the residual headline;
        this never runs RNEA (and for the ``local_world_aligned`` frame needs no FK
        — only the camera rotation and the shaped-body mass).

        :param out: frozen forward output — ``out["force"]["joint_forces"]``,
            ``out["contact"]["joint_probs"]``, ``out["mhr"]``.
        :param batch: same fields :meth:`forward` reads.
        :returns: cpu tensors ``magnitude_bw (M, 4)``, ``magnitude_newton (M, 4)``,
            ``probs (M, 4)``, ``vertical_sum_bw (M,)`` with ``M = n_elig * T`` in
            extremity order ``left_hand, right_hand, left_foot, right_foot``; or
            ``None`` when no clip in the batch is physics-eligible.
        """
        forces_all = out["force"]["joint_forces"].to(self.device, self.dtype)
        seq_len = int(batch["seq_len"])
        n_clips = forces_all.shape[0] // seq_len
        eligible, _ = self._eligible_clips(batch, seq_len, n_clips)
        if not eligible.any():
            return None
        n_elig = int(eligible.sum().item())
        rows = self._eligible_rows(eligible, seq_len)
        clip_idx = eligible.nonzero(as_tuple=False).flatten().to(self.device)

        forces = forces_all[rows].view(n_elig, seq_len, 4, 3)
        probs = out["contact"]["joint_probs"].to(self.device, self.dtype)[rows] \
            .view(n_elig, seq_len, 4)
        cam_flat = batch["cam_from_world"].detach().to(self.device, self.dtype)[rows]
        cam = cam_flat.view(n_elig, seq_len, 4, 4)
        gravity_world = batch["gravity_world"].detach().to(self.device, self.dtype) \
            .view(n_clips, seq_len, 3)[clip_idx][:, 0]                # per clip (., 3)

        mhr_out = {key: out["mhr"][key][rows] for key in _MHR_KEYS}
        body, q = self.adapter.q_from_mhr_out(mhr_out, cam_flat, n_elig, seq_len)
        mass = self.adapter.total_mass(body)                          # (n_elig,)

        r_w_c = cam[..., :3, :3].transpose(-1, -2)                    # (., T, 3, 3)
        r_joint = None
        if self.frame == "local":
            t_sec = batch["frame_pos_sec"].to(self.device, self.dtype)[rows] \
                .view(n_elig, seq_len)
            kernel = torch.tensor(self.smoothing_kernel, device=self.device, dtype=self.dtype)
            fk = br.forward_kinematics(
                body.robot,
                _smooth_configuration(q, t_sec, kernel),
                use_warp=self.use_warp,
            )
            ext_pose = fk.joint_pose_world.index_select(-2, self.adapter.extremity_joint_ids)
            r_joint = so3.to_matrix(ext_pose[..., 3:])

        magnitude_bw = (forces ** 2).sum(-1).add(_NORM_EPS).sqrt()    # (., T, 4)
        mass_g = (mass * self.gravity).view(n_elig, 1, 1)
        magnitude_newton = magnitude_bw * mass_g
        world_forces = self._pred_to_world(forces, r_w_c, r_joint)    # (., T, 4, 3) bw
        up = -gravity_world                                          # unit up (., 3)
        vertical_sum_bw = (world_forces * up[:, None, None, :]).sum(-1).sum(-1)  # (., T)

        def _flat(x: Tensor) -> Tensor:
            return x.reshape(n_elig * seq_len, *x.shape[2:]).detach().cpu()

        return {
            "magnitude_bw": _flat(magnitude_bw),
            "magnitude_newton": _flat(magnitude_newton),
            "probs": _flat(probs),
            "vertical_sum_bw": _flat(vertical_sum_bw),
        }

    @torch.no_grad()
    def affine_residual(self, out: dict, batch: dict) -> dict[str, Tensor] | None:
        """Per-residual-frame affine decomposition of the root-wrench residual (§4).

        The normalised root wrench is **affine** in the head-frame forces: for each
        residual frame ``r(f) = r0 + B · vec(f)`` with ``r0 ∈ R^6`` (zero forces),
        ``B ∈ R^{6×12}`` and ``vec(f)`` the 12-vector of the four extremity forces
        (extremity-major, component-minor — matching ``f_pred`` below). ``r0`` and
        the 12 columns of ``B`` are obtained by one zero-force and twelve unit-force
        no-grad RNEA calls, each mapped through :meth:`_fext_from_head_newton` exactly
        like the prediction. This lets :func:`scripts.evaluate.evaluate_physics`
        compare the network against the best fitted constant and against shuffled
        forces without any autograd — proving or disproving input-dependence.

        :param out: frozen forward output (same fields :meth:`forward` reads).
        :param batch: same camera/timing fields :meth:`forward` reads.
        :returns: cpu tensors ``r0 (Nc, nr, 6)``, ``basis (Nc, nr, 6, 12)``,
            ``f_pred (Nc, nr, 12)``, ``probs (Nc, nr, 4)`` and the ``seq_len`` int
            (``Nc`` eligible clips, ``nr`` residual frames), or ``None`` when the
            batch has no physics-eligible clip or no residual frame.
        """
        seq_len = int(batch["seq_len"])
        forces_all = out["force"]["joint_forces"].to(self.device, self.dtype)
        n_clips = forces_all.shape[0] // seq_len
        eligible, _ = self._eligible_clips(batch, seq_len, n_clips)
        residual_frames = _residual_frame_indices(seq_len, self.kernel_radius)
        if not eligible.any() or not residual_frames:
            return None
        n_elig = int(eligible.sum().item())
        rows = self._eligible_rows(eligible, seq_len)
        clip_idx = eligible.nonzero(as_tuple=False).flatten().to(self.device)

        forces = forces_all[rows].view(n_elig, seq_len, 4, 3)
        probs = out["contact"]["joint_probs"].detach().to(self.device, self.dtype)[rows] \
            .view(n_elig, seq_len, 4)
        cam_flat = batch["cam_from_world"].detach().to(self.device, self.dtype)[rows]
        cam = cam_flat.view(n_elig, seq_len, 4, 4)
        t_sec = batch["frame_pos_sec"].to(self.device, self.dtype)[rows] \
            .view(n_elig, seq_len)
        gravity_world = batch["gravity_world"].detach().to(self.device, self.dtype) \
            .view(n_clips, seq_len, 3)[clip_idx][:, 0]

        mhr_out = {key: out["mhr"][key][rows] for key in _MHR_KEYS}
        body, q = self.adapter.q_from_mhr_out(mhr_out, cam_flat, n_elig, seq_len)
        mass = self.adapter.total_mass(body)
        kernel = torch.tensor(self.smoothing_kernel, device=self.device, dtype=self.dtype)
        q_smoothed = _smooth_configuration(q, t_sec, kernel)
        fk = br.forward_kinematics(body.robot, q_smoothed, use_warp=self.use_warp)
        ext_pose = fk.joint_pose_world.index_select(-2, self.adapter.extremity_joint_ids)
        r_joint = so3.to_matrix(ext_pose[..., 3:])
        r_w_c = cam[..., :3, :3].transpose(-1, -2)

        sel = torch.tensor(residual_frames, device=self.device)
        n_res = sel.numel()
        velocity, acceleration = _trajectory_derivatives(body.robot, q_smoothed, t_sec)
        q_res = q_smoothed.index_select(1, sel)
        v_res = velocity.index_select(1, sel)
        a_res = acceleration.index_select(1, sel)
        r_w_c_res = r_w_c.index_select(1, sel)
        r_joint_res = r_joint.index_select(1, sel)
        mass_g = (mass * self.gravity).view(n_elig, 1, 1)

        grav_linear = self.gravity * gravity_world
        grav6 = torch.cat((grav_linear, torch.zeros_like(grav_linear)), dim=-1).unsqueeze(-2)
        robot = dataclasses.replace(
            body.robot, values=dataclasses.replace(body.robot.values, gravity=grav6))

        def root_residual(head_forces_bw: Tensor) -> Tensor:
            fext = self._fext_from_head_newton(
                head_forces_bw * mass_g.unsqueeze(-1), r_w_c_res, r_joint_res)
            tau = br.rnea(robot, q_res, v_res, a_res, fext=fext, use_warp=self.use_warp)
            return tau[..., :6] / mass_g                             # (n_elig, n_res, 6)

        zero = torch.zeros(n_elig, n_res, 4, 3, device=self.device, dtype=self.dtype)
        r0 = root_residual(zero)
        columns = []
        for extremity in range(4):
            for component in range(3):
                unit = zero.clone()
                unit[:, :, extremity, component] = 1.0
                columns.append(root_residual(unit) - r0)
        basis = torch.stack(columns, dim=-1)                         # (n_elig, n_res, 6, 12)
        f_pred = forces.index_select(1, sel).reshape(n_elig, n_res, 12)
        probs_res = probs.index_select(1, sel)
        return {
            "r0": r0.detach().cpu(),
            "basis": basis.detach().cpu(),
            "f_pred": f_pred.detach().cpu(),
            "probs": probs_res.detach().cpu(),
            "seq_len": seq_len,
        }

    # ------------------------------------------------------------- helpers

    def _fext_from_head_newton(
        self, forces_newton: Tensor, r_w_c: Tensor, r_joint: Tensor,
    ) -> Tensor:
        """Map per-extremity head-frame forces (newton) to the RNEA ``fext`` wrench.

        Head frame → world (:meth:`_pred_to_world`, D6) → each extremity joint's
        LOCAL frame → scattered onto the full joint set as a zero-torque wrench.

        :param forces_newton: ``(., Tsel, 4, 3)`` extremity forces in newton.
        :param r_w_c: ``(., Tsel, 3, 3)`` world-from-camera rotation.
        :param r_joint: ``(., Tsel, 4, 3, 3)`` world extremity-joint rotation.
        :returns: ``(., Tsel, njoints, 6)`` external wrench for :func:`better_robot.rnea`.
        """
        world_newton = self._pred_to_world(forces_newton, r_w_c, r_joint)
        local_newton = (r_joint.transpose(-1, -2) @ world_newton.unsqueeze(-1)).squeeze(-1)
        by_joint = torch.einsum("...ci,cj->...ji", local_newton, self._contact_to_joint)
        return torch.cat((by_joint, torch.zeros_like(by_joint)), dim=-1)

    def _pred_to_world(self, forces: Tensor, r_w_c: Tensor, r_joint: Tensor) -> Tensor:
        """Map predicted extremity forces to the world frame (D6).

        :param forces: ``(., Tsel, 4, 3)`` in the head frame.
        :param r_w_c: ``(., Tsel, 3, 3)`` world-from-camera rotation.
        :param r_joint: ``(., Tsel, 4, 3, 3)`` world joint rotation.
        :returns: ``(., Tsel, 4, 3)`` world-frame forces.
        """
        if self.frame == "local":
            return (r_joint @ forces.unsqueeze(-1)).squeeze(-1)
        f_cam = (self._flip @ forces.unsqueeze(-1)).squeeze(-1)
        return (r_w_c.unsqueeze(-3) @ f_cam.unsqueeze(-1)).squeeze(-1)

    def _eligible_rows(self, eligible: Tensor, seq_len: int) -> Tensor:
        """Flat clip-major row indices for the eligible clips ``(n_elig * T,)``."""
        clip_idx = eligible.nonzero(as_tuple=False).flatten().to(self.device)
        frames = torch.arange(seq_len, device=self.device)
        return (clip_idx.unsqueeze(-1) * seq_len + frames).flatten()

    def _assemble(
        self, terms: dict[str, tuple[Tensor, float]], zero_touch: Tensor,
        eligible: Tensor, residual_frames: list[int], diagnostics: dict[str, Any],
    ) -> tuple[Tensor, dict[str, Any]]:
        """Weight, normalise, and package the term contract + logging scalars.

        Every nonzero-weight term is always present (mass 0 when it has no data
        this batch), so the term set is fixed by config — not by per-batch
        eligibility — which the trainer needs for consistent DDP all-reduce. Each
        term carries the graph-connected ``zero_touch`` so every force param stays
        in the autograd graph regardless of which terms have data.
        """
        parts_terms: dict[str, dict[str, Any]] = {}
        total: Tensor | None = None
        for name in _TERM_NAMES:
            if self.weights[name] == 0.0:
                continue
            entry = terms.get(name)
            if entry is None:
                weighted, mass = zero_touch, 0.0
            else:
                raw, mass = entry
                weighted = self.weights[name] * raw + zero_touch
            normalized = weighted / max(mass, 1.0)
            total = normalized if total is None else total + normalized
            parts_terms[name] = {
                "weighted_numerator_tensor": weighted,
                "weight_mass": mass,
                "loss": float(normalized.detach()),
            }
        if total is None:                       # every weight is zero (degenerate)
            total = zero_touch

        parts: dict[str, Any] = {
            "terms": parts_terms,
            "loss": float(total.detach()),
            "n_eligible_clips": int(eligible.sum().item()),
            "n_residual_frames": len(residual_frames),
        }
        parts.update(diagnostics)
        return total, parts


__all__ = ["PhysicsLoss"]
