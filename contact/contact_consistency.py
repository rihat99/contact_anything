"""Contact→velocity consistency: a limb the model calls in contact must stand still.

The corpus contact labels are motion-gated "stable contact" (stillness +
hysteresis in the estimator), so a predicted contact makes a claim about the
WORLD: that extremity is not moving. This loss states that claim as an
objective — the world-frame speed of the six extremity keypoints, weighted by
the model's own contact probability::

    L = Σ_rows Σ_limbs  p_contact · huber(‖v_world‖, 0)  /  Σ p_contact

It needs no labels: the gate and the trajectory both come out of the forward
pass. The two gradient paths it opens are

- the **pose** path (``pred_keypoints_3d`` / ``pred_cam_t`` — the recomputed
  final readout): where contact is predicted, the pose must stop jittering.
  That is the same depth-wobble attack :mod:`contact.motion_consistency` makes
  on the root, applied to the limbs that are pinned to the wall;
- the **contact** path, only when ``detach_gate: false``: the head can lower a
  probability to escape a moving limb's penalty. The supervised focal loss is
  the counterweight — there is deliberately no extra guard here (a collapse of
  the gate would show up immediately as a contact-F1 drop).

Velocity is a central difference of the world-lifted keypoints over the clip's
real elapsed seconds, exactly as ``keypoint_supervision``'s ``kp_vel`` term
does it: differencing in the world removes the camera egomotion, so a handheld
camera cannot fake limb motion (camera-frame differences would bury the body
motion under camera shake). Rows need the full stencil (``t-1, t, t+1``) with
valid frames and valid extrinsics, so clip boundaries are never supervised and
``frames_per_clip >= 3`` is required.

Mass is the summed gate, not the row count: a term normalised by confidence
mass keeps its scale when the model predicts few contacts, and matches the
``(weighted_numerator, mass)`` contract the trainer's
:func:`~contact.losses.ddp_global_mean_term` reduces exactly under DDP. The
term set is fixed by config (mass 0 when a batch has no data), and every early
return still carries a graph-connected zero so no parameter drops out of the
backward graph under ``find_unused_parameters=False``.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

#: MHR70 keypoint index per contact/force group, in the corpus kindyn_6 order
#: ``left_hand, right_hand, left_foot, right_foot, left_ankle, right_ankle``
#: (left/right wrist, left/right big-toe tip, left/right heel) — the same
#: order as ``model.force_head.force_keypoint_indices`` and the six-group GT.
EXTREMITY_MHR70_INDICES = (62, 41, 15, 18, 17, 20)

_TERM_NAMES = ("vel",)


class ContactConsistencyLoss:
    """Gated world-speed penalty on the six extremity keypoints.

    :param cfg: resolved run config; reads ``contact_consistency.*``.
    :param device: device the loss runs on (predictions are moved to it).
    :param dtype: floating dtype (float32; the world lift mirrors
        :class:`~contact.keypoint_supervision.KeypointSupervisedLoss`, which
        differences in the library dtype rather than float64).
    """

    def __init__(
        self,
        cfg: dict,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        cc = cfg["contact_consistency"]
        self.detach_gate = bool(cc["detach_gate"])
        self.weights = {"vel": float(cc["loss"]["vel"])}
        self.huber_delta_ms = float(cc["loss"]["huber_delta_ms"])
        self.device = torch.device(device)
        self.dtype = dtype
        self.kp_idx = torch.tensor(EXTREMITY_MHR70_INDICES, device=self.device)

    def __call__(self, out: dict, batch: dict) -> tuple[Tensor, dict[str, Any]]:
        return self.forward(out, batch)

    def forward(self, out: dict, batch: dict) -> tuple[Tensor, dict[str, Any]]:
        """Return ``(total, parts)``.

        :param out: forward output — reads ``out["mhr"]["pred_keypoints_3d"]
            (B, 70, 3)`` and ``out["mhr"]["pred_cam_t"] (B, 3)`` (pose path,
            grads live) plus ``out["contact"]["joint_probs"] (B, 6)`` (the
            gate, detached iff ``detach_gate``) and
            ``out["contact"]["joint_logits"]`` (graph anchor only).
        :param batch: reads ``cam_from_world (B, 4, 4)``, ``cam_valid (B)``,
            ``frame_valid (B)``, ``frame_pos_sec (B)`` and ``seq_len``.
        :returns: ``(total, parts)``; ``parts["terms"]["vel"]`` carries
            ``weighted_numerator_tensor`` + ``weight_mass`` (summed gate) for
            exact DDP reduction, alongside the ``vel_ms`` / ``gate_mean``
            diagnostics.
        """
        mhr = out["mhr"]
        kp3d = mhr["pred_keypoints_3d"].to(self.device, self.dtype)[:, self.kp_idx]
        cam_t = mhr["pred_cam_t"].to(self.device, self.dtype)             # (B, 3)
        gate_full = out["contact"]["joint_probs"].to(self.device, self.dtype)
        # Graph-connected zero over every tensor the loss consumes: the pose
        # and contact params must stay on the backward graph even on batches
        # with no supervised rows (DDP find_unused_parameters=False).
        zero_touch = (mhr["pred_keypoints_3d"].sum() + mhr["pred_cam_t"].sum()
                      + out["contact"]["joint_logits"].sum()).to(self.dtype) * 0.0

        n_frames = kp3d.shape[0]
        seq_len = int(batch.get("seq_len", 1))
        if seq_len < 3 or n_frames % seq_len:
            return self._assemble(
                {name: (zero_touch, 0.0) for name in _TERM_NAMES}, zero_touch,
                {"vel_ms": 0.0, "gate_mean": 0.0, "n_rows": 0})
        n_clips = n_frames // seq_len

        # World lift (keypoint_supervision's composition):
        # p_w = R_ext^T (p_cam - t_ext).
        ext = batch["cam_from_world"].to(self.device, self.dtype)         # (B, 4, 4)
        pred_world = torch.einsum(
            "bji,bkj->bki", ext[:, :3, :3],
            kp3d + cam_t[:, None] - ext[:, :3, 3][:, None])               # (B, 6, 3)
        pw = pred_world.reshape(n_clips, seq_len, len(EXTREMITY_MHR70_INDICES), 3)
        pos_sec = batch["frame_pos_sec"].to(self.device, self.dtype).reshape(
            n_clips, seq_len)
        dt = (pos_sec[:, 1:] - pos_sec[:, :-1]).mean(dim=1).clamp(
            min=1e-6)[:, None, None, None]                               # (n,1,1,1)
        velocity = (pw[:, 2:] - pw[:, :-2]) / (2.0 * dt)             # (n, T-2, 6, 3)
        speed = velocity.norm(dim=-1)                                # (n, T-2, 6)

        gate = gate_full.detach() if self.detach_gate else gate_full
        gate = gate.reshape(n_clips, seq_len, -1)[:, 1:-1]           # (n, T-2, 6)
        # The stencil at t reads frames t-1, t, t+1: each must be a real frame
        # of the SAME clip with valid extrinsics (the lift needs them).
        ok = (batch["frame_valid"].to(self.device)
              & batch["cam_valid"].to(self.device)).reshape(n_clips, seq_len)
        support = ok[:, :-2] & ok[:, 1:-1] & ok[:, 2:]               # (n, T-2)

        elementwise = F.smooth_l1_loss(
            speed, torch.zeros_like(speed), reduction="none",
            beta=self.huber_delta_ms)                                # (n, T-2, 6)
        gate_masked = gate * support[..., None]
        raw = (gate_masked * elementwise).sum()
        mass = float((gate.detach() * support[..., None]).sum())

        with torch.no_grad():
            gated_speed = (gate.detach() * support[..., None] * speed).sum()
            supported_gate = gate.detach()[support]                  # (rows, 6)
            diagnostics = {
                "vel_ms": float(gated_speed / max(mass, 1.0)),
                "gate_mean": (float(supported_gate.mean())
                              if supported_gate.numel() else 0.0),
                "n_rows": int(support.sum()),
            }
        return self._assemble({"vel": (raw, mass)}, zero_touch, diagnostics)

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


__all__ = ["ContactConsistencyLoss", "EXTREMITY_MHR70_INDICES"]
