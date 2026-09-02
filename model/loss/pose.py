"""MHR q-space pose loss against the kindyn-MHR pseudo-ground truth.

Reads ``out["mhr"]["mhr_model_params"] (B, 204)`` — graph-live through the
final-readout recompute whenever anything writes the pose token — and the
collated ``pose_gt_q (B, 132)`` / ``pose_identity (B, 45)`` /
``pose_gt_bones (B, 6)`` / ``pose_gt_scale (B, 68)`` targets.

**Why q space.** The comparison runs on the rig's 125 local pose channels: the
prediction's parameter vector goes through the BetterHuman body's
``from_classic`` (differentiable in the parameters) and the target ``q`` is used
as stored. Parameter space is NOT usable directly — the 130 body-pose slots
project onto the 125-dim ``q`` manifold (the last six slots are coupled, so
``to_classic(from_classic(p)) != p`` there) and a parameter-space loss would
chase components the rig cannot represent. The free-flyer root is world-frame on
the target and camera-frame on the prediction, so it is never supervised.

**Proportions.** ``bones`` and ``scale`` supervise the body's geometry against
the GT ``lbs_params``: the flexible bone slots 130..135 and the 68 per-person
scale slots. ``mhr_model_params`` IS that same 204-vector — the head's 28 scale
coefficients have already been expanded through ``scale_mean + coeffs @
scale_comps`` inside it — so the comparison needs no unit conversion. It does
need a REACHABILITY projection on the scale side (:func:`_scale_subspace`): that
expansion spans only a rank-24 subspace of the 68 slots, and about half of the
GT deviation lies outside it, i.e. a residual the head can never remove and
whose Huber gradient would pull forever on directions that move no geometry.

**Channel normalisation.** ``shape``, ``bones`` and ``scale`` are per-channel
MEANS, so their weights are comparable across the 45 / 6 / 68 channel counts.
``shape`` is an L2 against the GT identity coefficients, which are constant per
person and broadcast over the clip — the per-frame form also acts as a temporal
consistency prior. It is independent of the q term (the q construction detaches
the identity; q itself is shape-independent).

Statistics carry the per-frame q MAE and the clip-wise q-channel acceleration
RMS of prediction and target — the smoothness pair (a ratio of 1.0 is as smooth
as kindyn) — plus the bone/scale MAEs.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from model.loss import Loss, LossResult
from utils.betterhuman import resolve_mhr_archive
from utils.metrics import mean_from_stats

#: Supervised ``q`` slice: the 125 local pose channels after the free-flyer root.
POSE_SLOTS = slice(7, 132)
N_POSE_CHANNELS = 125
#: ``mhr_model_params`` slices: the flexible bone-geometry slots (spine, neck,
#: shoulder width, arm, hip width, leg lengths) and the 68 per-person scale slots.
BONE_SLOTS = slice(130, 136)
SCALE_SLOTS = slice(136, 204)

_TERM_NAMES = ("pose", "shape", "bones", "scale")
def _scale_subspace(
    checkpoint_path: str, device: torch.device, dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Mean and orthogonal projector of the head's REACHABLE bone-scale set.

    The pose head emits 28 coefficients and expands them as
    ``scale_mean + coeffs @ scale_comps``, so the 68 scale slots it can produce
    form an affine subspace of rank 24 (``scale_comps`` is ``(28, 68)`` and rank
    deficient). Measured on the corpus, half of the GT slot deviation from
    ``scale_mean`` lies OUTSIDE that image. Projecting the GT onto the subspace
    first makes the residual reachable, at a ~1 mm cost in GT mesh geometry.

    :returns: ``(scale_mean (68,), projector (68, 68))`` on ``device``.
    """
    state = torch.load(checkpoint_path, map_location="cpu", mmap=True,
                       weights_only=False)
    state = state.get("state_dict", state)
    mean = state["head_pose.scale_mean"].to(device=device, dtype=dtype)
    comps = state["head_pose.scale_comps"].to(device=device, dtype=torch.float32)
    return mean, (torch.linalg.pinv(comps) @ comps).to(dtype)


def _wrap(diff: Tensor) -> Tensor:
    """Wrap channel differences to ``(-pi, pi]`` (the euler-like q channels)."""
    return torch.remainder(diff + math.pi, 2.0 * math.pi) - math.pi


class PoseLoss(Loss):
    """Huber in MHR ``q`` space plus the shape / bone / scale proportion terms."""

    name = "pose"
    stat_names = ("abs_err", "channel_rows", "acc2_pred", "acc2_gt", "acc_rows",
                  "bones_num", "bones_mass", "scale_num", "scale_mass")

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        super().__init__(cfg, model, device)
        import better_human as bh

        loss_cfg = cfg["pose_supervision"]["loss"]
        self.weights = {name: float(loss_cfg[name]) for name in _TERM_NAMES}
        self.term_names = ("pose",) + tuple(
            n for n in _TERM_NAMES[1:] if self.weights[n] > 0.0)
        if all(w == 0.0 for w in self.weights.values()):
            raise ValueError(
                "pose_supervision: every loss weight is 0 — disable the section instead")
        self.huber_delta = float(loss_cfg["huber_delta"])
        self.huber_delta_bones = float(loss_cfg["huber_delta_bones"])
        if self.weights["scale"] > 0.0:
            self.scale_mean, self.scale_proj = _scale_subspace(
                cfg["model"]["checkpoint_path"], self.device, self.dtype)
        lod = int(cfg["mhr_body"]["lod"])
        self.body = bh.MHR(
            resolve_mhr_archive(cfg["mhr_body"]["model_path"], lod),
            lod=lod,
            use_expression=False,
            use_correctives=False,
            compute_mass=False,
            dtype=self.dtype,
            device=self.device,
        )

    def __call__(self, out: dict, batch: dict, *, train: bool) -> LossResult:
        from better_human.bodies import MHRClassic

        params = out["mhr"]["mhr_model_params"].to(self.device, self.dtype)
        # q is shape-independent, so the identity used for the conversion is
        # irrelevant — detach it either way.
        identity = out["mhr"]["shape"].detach().to(self.device, self.dtype)
        _, q_pred = self.body.from_classic(
            MHRClassic(identity_coeffs=identity, model_parameters=params))
        pred = q_pred[..., POSE_SLOTS]                                  # (B, 125)
        anchor = pred.sum() * 0.0

        target = batch["pose_gt_q"].to(self.device, self.dtype)[..., POSE_SLOTS]
        valid = (batch["pose_valid"] & batch["frame_valid"]).to(self.device)
        mask = valid.to(self.dtype)
        mass = float(valid.sum())

        diff = _wrap(pred - target)
        huber = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none",
                                 beta=self.huber_delta)
        raw: dict[str, tuple[Tensor, float]] = {
            "pose": (self.weights["pose"] * (huber.sum(dim=-1) * mask).sum(), mass)
        }

        if self.weights["shape"] > 0.0:
            deviation = (out["mhr"]["shape"].to(self.device, self.dtype)
                         - batch["pose_identity"].to(self.device, self.dtype))
            raw["shape"] = (
                self.weights["shape"]
                * (deviation.square().mean(dim=-1) * mask).sum(), mass)

        proportion_stats = [0.0, 0.0, 0.0, 0.0]
        for i, (name, slots, gt_key) in enumerate(
                (("bones", BONE_SLOTS, "pose_gt_bones"),
                 ("scale", SCALE_SLOTS, "pose_gt_scale"))):
            if self.weights[name] <= 0.0:
                continue
            gt_slots = batch[gt_key].to(self.device, self.dtype)
            if name == "scale":
                # Only the head's rank-24 image is reachable (_scale_subspace).
                gt_slots = self.scale_mean + (gt_slots - self.scale_mean) @ self.scale_proj
            deviation = params[:, slots] - gt_slots
            slot_huber = F.smooth_l1_loss(
                deviation, torch.zeros_like(deviation), reduction="none",
                beta=self.huber_delta_bones)
            raw[name] = (
                self.weights[name] * (slot_huber.mean(dim=-1) * mask).sum(), mass)
            with torch.no_grad():
                proportion_stats[2 * i] = float(
                    (deviation.abs().mean(dim=-1) * mask).sum())
                proportion_stats[2 * i + 1] = mass

        stats = self._statistics(
            pred.detach(), target, valid, int(batch["seq_len"]), proportion_stats)
        scalars = {"mae": mean_from_stats(float(stats[0]), float(stats[1])),
                   "n_rows": float(valid.sum())}
        return LossResult(terms=self._terms(raw, anchor), scalars=scalars,
                          stats=stats)

    @torch.no_grad()
    def _statistics(
        self, pred: Tensor, target: Tensor, valid: Tensor, seq_len: int,
        proportions: list[float],
    ) -> Tensor:
        """Float64 sufficient statistics (:attr:`stat_names`).

        The acceleration rows need three consecutive VALID frames of ONE clip;
        the batch is clip-major / frame-minor, so a ``(n_clips, T, ...)`` view
        recovers them.
        """
        stats = torch.zeros(len(self.stat_names), dtype=torch.float64,
                            device=pred.device)
        err = _wrap(pred - target).abs().to(torch.float64)
        weight = valid.to(torch.float64)
        stats[0] = (err.sum(dim=-1) * weight).sum()
        stats[1] = weight.sum() * err.shape[-1]
        if seq_len >= 3 and pred.shape[0] % seq_len == 0:
            n_clips = pred.shape[0] // seq_len
            p = pred.reshape(n_clips, seq_len, -1).to(torch.float64)
            g = target.reshape(n_clips, seq_len, -1).to(torch.float64)
            v = valid.reshape(n_clips, seq_len)
            support = (v[:, 2:] & v[:, 1:-1] & v[:, :-2]).to(torch.float64)
            acc_p = p[:, 2:] - 2.0 * p[:, 1:-1] + p[:, :-2]
            acc_g = g[:, 2:] - 2.0 * g[:, 1:-1] + g[:, :-2]
            stats[2] = ((acc_p ** 2).sum(dim=-1) * support).sum()
            stats[3] = ((acc_g ** 2).sum(dim=-1) * support).sum()
            stats[4] = support.sum() * p.shape[-1]
        stats[5:9] = torch.tensor(proportions, dtype=torch.float64,
                                  device=pred.device)
        return stats

    def metrics(self, stats: Tensor) -> dict[str, float]:
        n_acc = max(float(stats[4]), 1.0)
        acc_pred = float((stats[2] / n_acc) ** 0.5)
        acc_gt = float((stats[3] / n_acc) ** 0.5)
        return {
            "mae": mean_from_stats(float(stats[0]), float(stats[1])),
            "acc_rms_pred": acc_pred,
            "acc_rms_gt": acc_gt,
            "acc_ratio": acc_pred / max(acc_gt, 1e-12),
            "bones_mae": mean_from_stats(float(stats[5]), float(stats[6])),
            "scale_mae": mean_from_stats(float(stats[7]), float(stats[8])),
            "n_rows": float(stats[1]) / N_POSE_CHANNELS,
        }


__all__ = ["PoseLoss", "POSE_SLOTS", "N_POSE_CHANNELS", "BONE_SLOTS", "SCALE_SLOTS"]
