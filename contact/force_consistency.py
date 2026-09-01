"""Newton consistency: the predicted contact forces must explain the predicted motion.

The linear half of the physics statement, written so that the body mass
cancels. With the forces already in body-weight units, Newton's second law on
the root reads::

    a_world / g  =  ĝ_world  +  Σ_i f_i^world           (all sides dimensionless)

so the residual this loss minimises is::

    r = a_world / g  -  ĝ_world  -  R_world←root · Σ_i f_i^root      [bw]

``a_world`` is the second difference of the PREDICTED world root position
(:func:`~contact.motion_consistency.predicted_root_world` — hip keypoints +
``pred_cam_t`` lifted with the dataset extrinsics), ``ĝ_world`` the kindyn unit
DOWN direction (``gravity_world``; world y is down-positive), and the forces
are the six-group head output in the kindyn body-root frame, already
contact-gated. Sanity check: a climber hanging still has ``a = 0`` and holds
one body weight, ``Σ f^world ≈ -ĝ``, so ``‖r‖ ≈ 0``.

Unlike :class:`~contact.physics.loss.PhysicsLoss` this is a **linear** balance
only: no RNEA, no MHR body, no inertia — hence no rotational half and no
dependence on the MHR archive. What it buys in exchange is a coupling the
supervised force loss cannot provide: the force head and the pose path are
tied to each other, so a force prediction that is inconsistent with the
observed motion costs, and a pose trajectory whose acceleration is
unexplainable by the predicted contacts costs too. Two gradient paths:

- **force** — through ``out["force"]["joint_forces"]``;
- **pose** — through the keypoints/``pred_cam_t`` inside the world lift.

``motion_rot`` and ``gravity_world`` are kindyn GT and are detached: the
rotation that expresses the root-frame forces in the world is never itself
fitted (an unrailed rotation would be the cheapest way to satisfy the balance).

The second difference needs the full stencil (``t-1, t, t+1``), so clip
boundaries are never supervised and ``frames_per_clip >= 3`` is required; rows
additionally need valid frames, extrinsics, kindyn root coverage and force
validity. Optional pre-smoothing (``smoothing_kernel``, ``[1.0]`` = off) runs
the linear windowed mean :func:`~contact.physics.loss._linear_windowed_mean`
over the world positions before differencing — a second difference of a noisy
trajectory is the loudest signal in this objective, which is also why the
trainer ramps the term in (``weight_scale``) instead of switching it on cold.

Terms follow the ``(weighted_numerator, mass)`` contract that the trainer's
:func:`~contact.losses.ddp_global_mean_term` reduces exactly under DDP; the
term set is fixed by config (mass 0 when a batch has no eligible rows) and
every early return carries a graph-connected zero so no parameter drops out of
the backward graph under ``find_unused_parameters=False``.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .motion_consistency import predicted_root_world
from .physics.loss import _linear_windowed_mean

#: Standard gravity (m/s²) — the scale that turns an acceleration into body
#: weights. The fitted per-scene ``gravity_world`` supplies the DIRECTION only.
GRAVITY_MS2 = 9.81

_TERM_NAMES = ("residual",)


class ForceConsistencyLoss:
    """Linear Newton residual between predicted forces and predicted root motion.

    :param cfg: resolved run config; reads ``force_consistency.*`` (the
        ``ramp`` block is the trainer's — it passes the resulting factor as
        ``weight_scale``).
    :param device: device the loss runs on (predictions are moved to it).
    :param dtype: floating dtype of the residual and the loss (float32); the
        world trajectory and its second difference stay float64 up to the
        residual, as in :class:`~contact.motion_consistency.MotionConsistencyLoss`.
    """

    def __init__(
        self,
        cfg: dict,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        fc = cfg["force_consistency"]
        self.weights = {"residual": float(fc["loss"]["residual"])}
        self.huber_delta_bw = float(fc["loss"]["huber_delta_bw"])
        self.device = torch.device(device)
        self.dtype = dtype
        self.smoothing_kernel = torch.tensor(
            [float(w) for w in fc["smoothing_kernel"]],
            dtype=torch.float64, device=self.device)

    def __call__(
        self, out: dict, batch: dict, weight_scale: float = 1.0,
    ) -> tuple[Tensor, dict[str, Any]]:
        return self.forward(out, batch, weight_scale)

    def forward(
        self, out: dict, batch: dict, weight_scale: float = 1.0,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Return ``(total, parts)``.

        :param out: forward output — reads ``out["mhr"]`` (``pred_keypoints_3d``,
            ``pred_cam_t``, ``global_rot``; the pose path, grads live) and
            ``out["force"]["joint_forces"] (B, 6, 3)`` in body-weight units,
            kindyn body-root frame.
        :param batch: reads ``cam_from_world (B, 4, 4)``, ``motion_rot
            (B, 3, 3)`` (world-from-root GT), ``gravity_world (B, 3)``,
            ``frame_pos_sec (B)``, ``seq_len`` and the validity flags
            ``frame_valid`` / ``cam_valid`` / ``motion_root_valid`` /
            ``force_valid``.
        :param weight_scale: the trainer's ramp factor in ``[0, 1]``; it
            multiplies the numerator only (never the mass), so a ramped term
            keeps its normalisation and is reported as the ``ramp`` diagnostic.
        :returns: ``(total, parts)``; ``parts["terms"]["residual"]`` carries
            ``weighted_numerator_tensor`` + ``weight_mass`` (supported rows)
            for exact DDP reduction, alongside the ``residual_bw`` / ``ramp``
            diagnostics.
        """
        forces = out["force"]["joint_forces"].to(self.device, self.dtype)
        mhr = out["mhr"]
        # Graph-connected zero over the tensors the loss consumes: the force
        # and pose params must stay on the backward graph even on batches with
        # no eligible rows (DDP find_unused_parameters=False).
        zero_touch = (mhr["pred_keypoints_3d"].sum() + mhr["pred_cam_t"].sum()
                      + out["force"]["joint_forces"].sum()).to(self.dtype) * 0.0

        n_frames = forces.shape[0]
        seq_len = int(batch.get("seq_len", 1))
        if seq_len < 3 or n_frames % seq_len:
            return self._assemble(
                {name: (zero_touch, 0.0) for name in _TERM_NAMES}, zero_touch,
                {"residual_bw": 0.0, "ramp": float(weight_scale), "n_rows": 0})
        n_clips = n_frames // seq_len

        pos_w, _ = predicted_root_world(mhr, batch["cam_from_world"])   # (B,3) f64
        positions = _linear_windowed_mean(
            pos_w.reshape(n_clips, seq_len, 3), self.smoothing_kernel)
        pos_sec = batch["frame_pos_sec"].to(self.device, torch.float64).reshape(
            n_clips, seq_len)
        dt = (pos_sec[:, 1:] - pos_sec[:, :-1]).mean(dim=1).clamp(
            min=1e-6)[:, None, None]                                    # (n,1,1)
        acceleration = (positions[:, 2:] - 2.0 * positions[:, 1:-1]
                        + positions[:, :-2]) / dt.square()         # (n,T-2,3) m/s²

        # GT rotation and gravity carry no graph: the world expression of the
        # root-frame forces is never fitted to satisfy the balance.
        rot_world_from_root = batch["motion_rot"].to(
            self.device, self.dtype).detach()                           # (B,3,3)
        force_world = torch.einsum(
            "bij,bj->bi", rot_world_from_root, forces.sum(dim=1))       # (B,3) bw
        gravity = batch["gravity_world"].to(self.device, self.dtype).detach()

        interior = slice(1, seq_len - 1)
        residual = (acceleration.to(self.dtype) / GRAVITY_MS2
                    - gravity.reshape(n_clips, seq_len, 3)[:, interior]
                    - force_world.reshape(n_clips, seq_len, 3)[:, interior])

        ok = (batch["frame_valid"].to(self.device)
              & batch["cam_valid"].to(self.device)
              & batch["motion_root_valid"].to(self.device)
              & batch["force_valid"].to(self.device)).reshape(n_clips, seq_len)
        support = ok[:, :-2] & ok[:, 1:-1] & ok[:, 2:]                  # (n, T-2)

        row_loss = F.smooth_l1_loss(
            residual, torch.zeros_like(residual), reduction="none",
            beta=self.huber_delta_bw).sum(dim=-1)                       # (n, T-2)
        raw = weight_scale * (row_loss * support).sum()
        mass = float(support.sum())

        with torch.no_grad():
            diagnostics = {
                "residual_bw": float(
                    (residual.norm(dim=-1) * support).sum() / max(mass, 1.0)),
                "ramp": float(weight_scale),
                "n_rows": int(support.sum()),
            }
        return self._assemble({"residual": (raw, mass)}, zero_touch, diagnostics)

    def _assemble(
        self,
        terms: dict[str, tuple[Tensor, float]],
        zero_touch: Tensor,
        diagnostics: dict[str, Any],
    ) -> tuple[Tensor, dict[str, Any]]:
        """Weight, normalise and package (the MotionConsistencyLoss contract)."""
        parts_terms: dict[str, dict[str, Any]] = {}
        total: Tensor | None = None
        for name in _TERM_NAMES:
            if self.weights[name] == 0.0:
                continue
            raw, mass = terms[name]
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
        parts: dict[str, Any] = {"terms": parts_terms, "loss": float(total.detach())}
        parts.update(diagnostics)
        return total, parts


__all__ = ["ForceConsistencyLoss", "GRAVITY_MS2"]
