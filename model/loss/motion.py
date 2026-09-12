"""Supervision of the refiner's motion output against finite-differenced kindyn GT.

Reads ``out["motion"]``: ``vel`` / ``acc`` ``(B, 22, 3)`` — world velocity and
acceleration of the 22 SMPL-X body joints — and ``ang_vel`` / ``ang_acc``
``(B, 3)`` of the root, every vector expressed in the predicted body frame
``frame`` ``(B, 3, 3)`` (world-from-body of the depth-smoothed per-frame root).

Targets are built per clip from the kindyn world joints and root rotation
(``smplx_joints_world`` / ``smplx_root_rot``): central finite differences at
the clip's real frame spacing, Gaussian label smoothing of ``label_smooth_sec``
on the velocity (the acceleration is the smoothed derivative of the smoothed
velocity — the 2026-08 motion round showed raw kindyn derivatives are too noisy
to learn from), then rotated into the predicted body frame so prediction and
target share one frame. The smoothing weights by the derivative's own support
(both neighbours valid), and rows within ``ceil(2 sigma / dt)`` frames of a run
end or a hole are not supervised — their kernel is truncated.

Every quantity is divided by its ``scale`` (the GT RMS, config) before a Huber
of width ``huber_delta`` in those standardized units, so the four terms start on
an equal footing. Metrics: RMSE in physical units and the pooled Pearson
correlation over all components, per quantity.

``stencil: aligned`` replaces the double smoothing by ONE Gaussian on the GT
trajectory (positions; rotations as projected matrix means) followed by the
same stencils the prediction side uses — central velocity, tight centred second
difference (span +-1), the rotational twins :func:`~model.refiner.angular_velocity`
/ :func:`~model.refiner.angular_acceleration` — so target and estimator are the
same operator on two trajectories (the 2026-09-07 audit: the legacy acceleration
target is effectively a 0.17 s Gaussian against a raw second difference, 0.35x
the estimator's RMS). ``legacy`` is the round-4 recipe above.

``stencil: forward`` is ``aligned`` with the two CENTRAL first differences
replaced by one-sided forward ones — velocity ``(x[t+1] - x[t]) / h`` at frame
``t`` and the rotational twin
:func:`~model.refiner.forward_angular_velocity` — on both sides. A central
first difference has exactly zero response to a period-2 alternation, which is
what the jitter metric weighs most; the forward one does not. The rows are
then the stencils' own support (velocity: ``t`` and ``t + 1`` valid;
acceleration: the ``+-1`` neighbours), with no Gaussian edge margin — the
round is run at ``label_smooth_sec: 0``, where the target is the raw GT.

**Pose-derivative matching** (``loss.pose_*`` weights): the same four targets
are also matched by the finite differences of the REFINED trajectory itself
(``out["smplx"]["joints_world"]`` / ``root_rot_world``, the same stencils, no
smoothing on the prediction side), so the pose head is rewarded for a
trajectory whose velocity and acceleration are the GT's — a derivative-level
objective a per-frame position Huber does not provide. Reported as
``pose_<quantity>_rmse`` / ``_pearson`` next to the head's numbers. The
motion-HEAD terms are optional: with their four weights at zero the loss runs
on the ``pose_*`` terms alone and ``out["motion"]`` may be ``None`` (no motion
head built).

**Deep supervision** (``layer_weight``, an iterative refiner only): every
enabled ``pose_*`` term is evaluated again on each INTERMEDIATE layer's
trajectory (``out["smplx"]["joints_world_layers"][:-1]`` and the matching
``root_rot_world_layers``) and reported as ``<term>_layer`` — the layers pooled
into one term (numerators and masses summed), at ``layer_weight`` times the
term's own weight. The metrics stay the final layer's.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from model.loss import Loss, LossResult
from model.refiner import (NUM_BODY_JOINTS, angular_acceleration, angular_velocity,
                           forward_angular_velocity, forward_difference, forward_valid,
                           gaussian_smooth, second_difference, smooth_rotations, stencil_valid,
                           time_derivative)
from utils.metrics import pearson_from_stats

QUANTITIES = ("vel", "acc", "ang_vel", "ang_acc")
STENCILS = ("legacy", "aligned", "forward")
#: The two predictions matched against the targets: the motion head, and the
#: finite differences of the refined pose (``pose_<quantity>`` terms / metrics).
SOURCES = ("head", "pose")
#: Metric-only rows: the refined pose's derivatives against the GT smoothed with
#: :data:`EVAL_SMOOTH_SEC` (``pose_s_<q>``) — the raw GT's second difference is mostly its own
#: noise, so a correlation against it says little about the acceleration a force model gets —
#: and the same with the PREDICTED trajectory smoothed identically (``pose_ss_<q>``): the
#: band-limited comparison, where an amplitude ratio below 1 is real attenuation.
METRIC_SOURCES = SOURCES + ("pose_s", "pose_ss")
EVAL_SMOOTH_SEC = 0.12
_STATS = ("se", "sum_p", "sum_g", "sum_pg", "sum_pp", "sum_gg", "n")


def _tag(source: str, quantity: str) -> str:
    return quantity if source == "head" else f"{source}_{quantity}"


class MotionLoss(Loss):
    """Standardized Huber on the four motion quantities of the refiner."""

    name = "motion"
    stat_names = tuple(f"{_tag(src, q)}/{s}" for src in METRIC_SOURCES for q in QUANTITIES
                       for s in _STATS)

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        super().__init__(cfg, model, device)
        section = cfg["motion_supervision"]
        self.sigma = float(section["label_smooth_sec"])
        self.stencil = str(section["stencil"])
        if self.stencil not in STENCILS:
            raise ValueError(
                f"motion_supervision.stencil must be one of {STENCILS}; got {self.stencil!r}")
        self.scale = {q: float(section["scale"][q]) for q in QUANTITIES}
        self.weights = {_tag(src, q): float(section["loss"][_tag(src, q)])
                        for src in SOURCES for q in QUANTITIES}
        self.delta = float(section["loss"]["huber_delta"])
        self.rms_weights = {q: float(section["loss"][f"pose_{q}_rms"]) for q in QUANTITIES}
        self.ss_weights = {q: float(section["loss"][f"pose_ss_{q}"]) for q in QUANTITIES}
        #: The terms a trajectory carries — the ones deep supervision repeats per layer.
        self.pose_terms = tuple(
            f"pose_{q}" for q in QUANTITIES if self.weights[f"pose_{q}"] > 0.0) + tuple(
            f"pose_{q}_rms" for q in QUANTITIES if self.rms_weights[q] > 0.0)
        self.layer_weight = float(section["layer_weight"])
        self.term_names = tuple(t for t in self.weights if self.weights[t] > 0.0) + tuple(
            f"pose_{q}_rms" for q in QUANTITIES if self.rms_weights[q] > 0.0) + tuple(
            f"pose_ss_{q}" for q in QUANTITIES if self.ss_weights[q] > 0.0) + (
            tuple(f"{t}_layer" for t in self.pose_terms) if self.layer_weight > 0.0 else ())

    def targets(self, batch: dict, n_frames: int, sigma: float | None = None,
                ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """GT motion in the WORLD frame and the per-row validity masks.

        :param sigma: label smoothing width override (s); ``None`` = the config's.
        :returns: ``({quantity: target}, {quantity: mask (B,) bool})``.
        """
        sigma = self.sigma if sigma is None else float(sigma)
        seq_len = int(batch["seq_len"])
        n_clips = n_frames // seq_len
        seconds = batch["frame_pos_sec"].to(self.device, self.dtype).view(n_clips, seq_len)
        valid = (batch["smplx_valid"] & batch["frame_valid"]).to(self.device).view(n_clips, seq_len)
        joints = batch["smplx_joints_world"][:, :NUM_BODY_JOINTS].to(self.device, self.dtype)
        joints = joints.view(n_clips, seq_len, NUM_BODY_JOINTS, 3)
        root = batch["smplx_root_rot"].to(self.device, self.dtype).view(n_clips, seq_len, 3, 3)

        # A derivative exists only where both neighbours are valid; the smoothing must weight
        # by THAT support (a forced-zero derivative next to a hole would otherwise leak into
        # its valid neighbours). Rows within ~2 sigma of a run end or a hole see a truncated
        # kernel, so the loss masks them too: radius = ceil(2 sigma / dt).
        first = stencil_valid(valid, 1)
        second = stencil_valid(valid, 2)
        if self.stencil in ("aligned", "forward"):
            # ONE Gaussian on the trajectory (a no-op at label_smooth_sec 0), then the same
            # stencils the prediction side uses: target and estimator are one operator.
            joints_s = gaussian_smooth(joints, seconds, valid, sigma)
            root_s = smooth_rotations(root, seconds, valid, sigma)
            if self.stencil == "forward":
                vel_w = forward_difference(joints_s, seconds, valid)
                ang_w = (root_s @ forward_angular_velocity(root_s, seconds, valid)[..., None])[..., 0]
            else:
                vel_w = time_derivative(joints_s, seconds, valid)
                ang_w = (root_s @ angular_velocity(root_s, seconds, valid)[..., None])[..., 0]
            acc_w = second_difference(joints_s, seconds, valid)
            ang_acc_w = (root_s @ angular_acceleration(root_s, seconds, valid)[..., None])[..., 0]
        else:
            vel_w = gaussian_smooth(time_derivative(joints, seconds, valid), seconds, first, sigma)
            acc_w = gaussian_smooth(time_derivative(vel_w, seconds, first), seconds, second, sigma)
            ang_body = angular_velocity(root, seconds, valid)                    # GT body frame
            ang_w = gaussian_smooth((root @ ang_body[..., None])[..., 0], seconds, first, sigma)
            ang_acc_w = gaussian_smooth(time_derivative(ang_w, seconds, first), seconds, second, sigma)

        targets = {
            "vel": vel_w.reshape(n_frames, NUM_BODY_JOINTS, 3),
            "acc": acc_w.reshape(n_frames, NUM_BODY_JOINTS, 3),
            "ang_vel": ang_w.reshape(n_frames, 3),
            "ang_acc": ang_acc_w.reshape(n_frames, 3),
        }
        if self.stencil == "forward":
            # The stencils' own support: the forward velocity needs t and t + 1, the tight
            # second difference the +-1 neighbours. No Gaussian edge to widen them by.
            rows_vel = forward_valid(valid).reshape(n_frames)
            rows_acc = first.reshape(n_frames)
        else:
            steps = seconds[:, 1:] - seconds[:, :-1]
            dt = float(steps[steps > 0].median()) if bool((steps > 0).any()) else 0.0
            edge = int(math.ceil(2.0 * sigma / dt)) if dt > 0 else 0
            rows_vel = stencil_valid(valid, max(1, edge)).reshape(n_frames)
            rows_acc = stencil_valid(valid, max(2, edge)).reshape(n_frames)
        masks = {"vel": rows_vel, "acc": rows_acc, "ang_vel": rows_vel, "ang_acc": rows_acc}
        return targets, masks

    def pose_derivatives(self, out: dict, batch: dict) -> dict[str, Tensor]:
        """World velocity / acceleration of the refined joints and root, by raw central
        finite differences of ``out["smplx"]`` (graph-live: this is the pose head's signal)."""
        smplx = out["smplx"]
        return self.trajectory_derivatives(smplx["joints_world"], smplx["root_rot_world"], batch)

    def _smoothed_prediction(self, out: dict, batch: dict, sigma: float) -> tuple[Tensor, Tensor]:
        """The refined world joints and root rotation under the eval Gaussian (graph-live)."""
        smplx = out["smplx"]
        joints = smplx["joints_world"].to(self.device, self.dtype)
        root = smplx["root_rot_world"].to(self.device, self.dtype)
        n_frames = joints.shape[0]
        seq_len = int(batch["seq_len"])
        n_clips = n_frames // seq_len
        seconds = batch["frame_pos_sec"].to(self.device, self.dtype).view(n_clips, seq_len)
        valid = batch["frame_valid"].to(self.device).view(n_clips, seq_len)
        joints = gaussian_smooth(joints.view(n_clips, seq_len, -1, 3), seconds, valid, sigma)
        root = smooth_rotations(root.view(n_clips, seq_len, 3, 3), seconds, valid, sigma)
        return joints.reshape(n_frames, -1, 3), root.reshape(n_frames, 3, 3)

    def trajectory_derivatives(self, joints_world: Tensor, root_world: Tensor,
                               batch: dict) -> dict[str, Tensor]:
        """The same stencils on ONE world trajectory (a layer's, or the final one)."""
        joints = joints_world[:, :NUM_BODY_JOINTS].to(self.device, self.dtype)
        root = root_world.to(self.device, self.dtype)
        n_frames = joints.shape[0]
        seq_len = int(batch["seq_len"])
        n_clips = n_frames // seq_len
        seconds = batch["frame_pos_sec"].to(self.device, self.dtype).view(n_clips, seq_len)
        valid = batch["frame_valid"].to(self.device).view(n_clips, seq_len)
        joints = joints.view(n_clips, seq_len, NUM_BODY_JOINTS, 3)
        root = root.view(n_clips, seq_len, 3, 3)
        if self.stencil == "forward":
            vel = forward_difference(joints, seconds, valid)
            ang = (root @ forward_angular_velocity(root, seconds, valid)[..., None])[..., 0]
        else:
            vel = time_derivative(joints, seconds, valid)
            ang = (root @ angular_velocity(root, seconds, valid)[..., None])[..., 0]   # world frame
        if self.stencil == "legacy":
            acc = time_derivative(vel, seconds, valid)
            ang_acc = time_derivative(ang, seconds, valid)
        else:
            acc = second_difference(joints, seconds, valid)
            ang_acc = (root @ angular_acceleration(root, seconds, valid)[..., None])[..., 0]
        return {"vel": vel.reshape(n_frames, NUM_BODY_JOINTS, 3),
                "acc": acc.reshape(n_frames, NUM_BODY_JOINTS, 3),
                "ang_vel": ang.reshape(n_frames, 3), "ang_acc": ang_acc.reshape(n_frames, 3)}

    def __call__(self, out: dict, batch: dict, *, train: bool) -> LossResult:
        n_frames = out["smplx"]["joints_world"].shape[0]
        targets_world, masks = self.targets(batch, n_frames)
        preds: dict[str, dict[str, Tensor]] = {}
        targets: dict[str, dict[str, Tensor]] = {}
        motion = out["motion"]
        if motion is not None:
            preds["head"] = {q: motion[q].to(self.device, self.dtype) for q in QUANTITIES}
            # The head predicts in its body frame; the frame is a fixed input of the loss (a
            # trainable pose path must never lower this term by rotating its root instead of
            # fixing the motion), so the WORLD targets are rotated into it.
            to_body = motion["frame"].detach().to(self.device, self.dtype).transpose(1, 2)
            targets["head"] = {
                q: (torch.einsum("bij,bkj->bki", to_body, t) if t.dim() == 3
                    else (to_body @ t[..., None])[..., 0]) for q, t in targets_world.items()}
        if any(self.weights[_tag("pose", q)] > 0.0 or self.rms_weights[q] > 0.0
               or self.ss_weights[q] > 0.0 for q in QUANTITIES):
            preds["pose"] = self.pose_derivatives(out, batch)
            targets["pose"] = targets_world
        if "pose" in preds:
            with torch.no_grad():
                targets["pose_s"] = self.targets(batch, n_frames, EVAL_SMOOTH_SEC)[0]
            targets["pose_ss"] = targets["pose_s"]
            preds["pose_s"] = preds["pose"]
            # Band-limited matching (`pose_ss_*` terms): the smoothed prediction against the
            # smoothed target, so the loss asks for the predictable band only and has no shrink
            # direction on the raw trajectory. Metric-only when unweighted.
            with torch.set_grad_enabled(any(w > 0.0 for w in self.ss_weights.values())):
                preds["pose_ss"] = self.trajectory_derivatives(
                    *self._smoothed_prediction(out, batch, EVAL_SMOOTH_SEC), batch)
        anchor = sum(p.sum() for source in preds.values() for p in source.values()) * 0.0

        raw: dict[str, tuple[Tensor, float]] = {}
        stats = []
        for src in METRIC_SOURCES:
            for q in QUANTITIES:
                tag = _tag(src, q)
                if src not in preds:
                    stats += [0.0] * len(_STATS)
                    continue
                p = preds[src][q].reshape(n_frames, -1)
                g = targets[src][q].reshape(n_frames, -1)
                mask = masks[q].to(self.dtype)
                if src in SOURCES and self.weights[tag] > 0.0:
                    raw[tag] = (self.weights[tag] * self._huber_rows(q, p, g, mask).sum(),
                                float(mask.sum()))
                elif src == "pose_ss" and self.ss_weights[q] > 0.0:
                    raw[tag] = (self.ss_weights[q] * self._huber_rows(q, p, g, mask).sum(),
                                float(mask.sum()))
                with torch.no_grad():
                    pm, gm = p.detach() * mask[:, None], g * mask[:, None]
                    stats += [float(((pm - gm) ** 2).sum()), float(pm.sum()), float(gm.sum()),
                              float((pm * gm).sum()), float((pm * pm).sum()), float((gm * gm).sum()),
                              float(mask.sum() * p.shape[1])]
        if "pose" in preds:
            raw.update(self.rms_terms(preds["pose"], targets["pose"], masks, batch))
        if self.layer_weight > 0.0:
            raw.update(self.layer_terms(out["smplx"], targets_world, masks, batch))
        return LossResult(
            terms=self._terms(raw, anchor),
            scalars={"n_rows": float(masks["vel"].sum())},
            stats=torch.tensor(stats, dtype=torch.float64, device=self.device))

    def _huber_rows(self, quantity: str, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Per-row Huber of one quantity in ``scale`` units, zeroed off the mask."""
        huber = F.smooth_l1_loss(pred / self.scale[quantity], target / self.scale[quantity],
                                 reduction="none", beta=self.delta).mean(dim=-1)
        return huber * mask

    def layer_terms(self, smplx: dict, targets: dict[str, Tensor], masks: dict[str, Tensor],
                    batch: dict) -> dict[str, tuple[Tensor, float]]:
        """Deep supervision (``layer_weight``): the pose terms on the INTERMEDIATE trajectories.

        Every enabled ``pose_*`` term is evaluated again on each layer of
        ``smplx["joints_world_layers"][:-1]`` (the last layer IS the supervised body) and
        the layers are pooled into one ``<term>_layer`` term — numerators summed, mass
        summed, so the term is the mean over rows AND layers at the term's own weight
        times ``layer_weight``.
        """
        joints_layers = smplx["joints_world_layers"][:-1]
        root_layers = smplx["root_rot_world_layers"][:-1]
        sums = {name: torch.zeros((), device=self.device, dtype=self.dtype)
                for name in self.pose_terms}
        masses = {name: 0.0 for name in self.pose_terms}
        for joints, root in zip(joints_layers, root_layers):
            preds = self.trajectory_derivatives(joints, root, batch)
            for q in QUANTITIES:
                name = f"pose_{q}"
                if self.weights[name] > 0.0:
                    rows = self._huber_rows(q, preds[q].reshape(preds[q].shape[0], -1),
                                            targets[q].reshape(targets[q].shape[0], -1),
                                            masks[q].to(self.dtype))
                    sums[name] = sums[name] + self.weights[name] * rows.sum()
                    masses[name] += float(masks[q].sum())
            for name, (numerator, mass) in self.rms_terms(preds, targets, masks, batch).items():
                sums[name] = sums[name] + numerator
                masses[name] += mass
        return {f"{name}_layer": (self.layer_weight * sums[name], masses[name])
                for name in self.pose_terms}

    def rms_terms(self, preds: dict[str, Tensor], targets: dict[str, Tensor],
                  masks: dict[str, Tensor], batch: dict) -> dict[str, tuple[Tensor, float]]:
        """Amplitude matching (``loss.pose_<q>_rms``): per clip and per joint, the RMS over
        the supervised rows of the predicted derivative's norm against the target's, Huber in
        ``scale`` units. The pointwise Huber ``E|p - g|^2 = (rms_p - rms_g)^2 + 2 rms_p rms_g
        (1 - r)`` rewards shrinking ``rms_p`` whenever the target is not predictable frame by
        frame (``r < 1``); this keeps only its first half, so the only way down is the GT's
        amplitude — a per-frame-unpredictable acceleration can still set the smoothness."""
        seq_len = int(batch["seq_len"])
        raw: dict[str, tuple[Tensor, float]] = {}
        for q in QUANTITIES:
            if self.rms_weights[q] <= 0.0:
                continue
            p, g = preds[q], targets[q]
            n_frames = p.shape[0]
            n_clips = n_frames // seq_len
            p = p.reshape(n_clips, seq_len, -1, 3)
            g = g.reshape(n_clips, seq_len, -1, 3)
            mask = masks[q].to(self.dtype).view(n_clips, seq_len, 1)
            rows = mask.sum(dim=1)                                        # [n, 1]
            msq_p = ((p ** 2).sum(dim=-1) * mask).sum(dim=1) / rows.clamp(min=1.0)
            msq_g = ((g ** 2).sum(dim=-1) * mask).sum(dim=1) / rows.clamp(min=1.0)
            rms_p = (msq_p + 1e-8).sqrt() / self.scale[q]
            rms_g = (msq_g + 1e-8).sqrt() / self.scale[q]
            huber = F.smooth_l1_loss(rms_p, rms_g, reduction="none", beta=self.delta)
            weight = (rows > 1.0).to(self.dtype).expand_as(huber)
            raw[f"pose_{q}_rms"] = (self.rms_weights[q] * (huber * weight).sum(), float(weight.sum()))
        return raw

    def metrics(self, stats: Tensor) -> dict[str, float]:
        out = {}
        k = len(_STATS)
        for i, tag in enumerate(_tag(src, q) for src in METRIC_SOURCES for q in QUANTITIES):
            se, sp, sg, spg, spp, sgg, n = (float(v) for v in stats[k * i:k * i + k])
            if n <= 0:
                continue                      # the pose terms are off: no metric rows
            out[f"{tag}_rmse"] = math.sqrt(se / n)
            out[f"{tag}_pearson"] = pearson_from_stats(sp, sg, spg, spp, sgg, n)
            out[f"{tag}_amp_ratio"] = math.sqrt(spp / sgg) if sgg > 0 else 0.0
        return out


__all__ = ["MotionLoss", "QUANTITIES", "SOURCES"]
