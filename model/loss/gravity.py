"""The refiner's predicted gravity against the corpus gravity.

Reads ``out["gravity"]["world"] (B, 3)`` — the refiner's unit down vector per
clip, expanded to its frames — and scores it per CLIP against the collated
``gravity_world``: ``1 − cos`` (the ``cos`` term). A rotation of the world rotates
both vectors, so the term is frame-independent like everything else in the
refiner.

The corpus gravity is a measurement on the ``ground`` / ``geocalib`` scenes and
the first camera's down axis on the ``fallback_down`` ones (``gravity_measured``
per frame); ``measured_only`` restricts the supervision to the former. The
metrics always report both populations apart: ``angle_measured`` /
``angle_fallback`` (mean degrees between the prediction and the corpus vector)
and ``prior_angle_measured`` / ``prior_angle_fallback``, the same for the head's
zero-init estimate (``out["gravity"]["prior_world"]``: the pooled camera down
axis), which is the bar a trained head has to beat.

``layer_weight`` repeats the term on the intermediate layers' estimates
(``world_layers[:-1]``) as ``cos_layer``, pooled (mass = clips × layers).
"""
from __future__ import annotations

import torch
from torch import Tensor

from model.loss import Loss, LossResult
from utils.metrics import mean_from_stats


class GravityLoss(Loss):
    """``1 − cos`` between the predicted and the corpus down vector, per clip."""

    name = "gravity"
    term_names = ("cos",)
    stat_names = ("angle_measured", "n_measured", "angle_fallback", "n_fallback",
                  "prior_angle_measured", "prior_angle_fallback")

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        super().__init__(cfg, model, device)
        section = cfg["gravity_supervision"]
        self.weight = float(section["weight"])
        self.measured_only = bool(section["measured_only"])
        self.layer_weight = float(section["layer_weight"])
        if self.layer_weight > 0.0:
            self.term_names = ("cos", "cos_layer")

    def _clips(self, batch: dict) -> tuple[Tensor, Tensor, Tensor]:
        """Per-clip ``(gravity (n, 3), valid (n,), measured (n,))`` from the frame rows."""
        seq_len = int(batch["seq_len"])
        n_clips = batch["frame_valid"].shape[0] // seq_len
        valid = batch["frame_valid"].to(self.device).view(n_clips, seq_len).any(dim=1)
        gravity = batch["gravity_world"].to(self.device, self.dtype).view(n_clips, seq_len, 3)[:, 0]
        measured = batch["gravity_measured"].to(self.device).view(n_clips, seq_len)[:, 0]
        return gravity, valid, measured

    def _cos_term(self, predicted: Tensor, gravity: Tensor, weight: Tensor,
                  seq_len: int) -> Tensor:
        pred = predicted.view(-1, seq_len, 3)[:, 0]
        return ((1.0 - (pred * gravity).sum(dim=-1)) * weight).sum()

    def __call__(self, out: dict, batch: dict, *, train: bool) -> LossResult:
        pred = out["gravity"]
        world = pred["world"].to(self.device, self.dtype)
        anchor = world.sum() * 0.0
        seq_len = int(batch["seq_len"])
        gravity, valid, measured = self._clips(batch)
        weight = valid.to(self.dtype)
        if self.measured_only:
            weight = weight * measured.to(self.dtype)
        mass = float(weight.sum())
        raw = {"cos": (self.weight * self._cos_term(world, gravity, weight, seq_len), mass)}
        if self.layer_weight > 0.0:
            layers = pred["world_layers"][:-1]
            numerator = sum(self._cos_term(w.to(self.device, self.dtype), gravity, weight, seq_len)
                            for w in layers)
            raw["cos_layer"] = (self.layer_weight * self.weight * numerator, mass * len(layers))

        with torch.no_grad():
            def angles(vectors: Tensor) -> Tensor:
                v = vectors.view(-1, seq_len, 3)[:, 0]
                cos = (v * gravity).sum(dim=-1).clamp(-1.0, 1.0)
                return torch.rad2deg(torch.acos(cos))
            angle = angles(world)
            prior = angles(pred["prior_world"].to(self.device, self.dtype))
            is_measured = (valid & measured).to(torch.float64)
            is_fallback = (valid & ~measured).to(torch.float64)
            stats = torch.stack([
                (angle.double() * is_measured).sum(), is_measured.sum(),
                (angle.double() * is_fallback).sum(), is_fallback.sum(),
                (prior.double() * is_measured).sum(), (prior.double() * is_fallback).sum()])
        return LossResult(terms=self._terms(raw, anchor), scalars={"n_clips": mass},
                          stats=stats.to(self.device))

    def metrics(self, stats: Tensor) -> dict[str, float]:
        n_measured, n_fallback = float(stats[1]), float(stats[3])
        return {"angle_measured": mean_from_stats(float(stats[0]), n_measured),
                "angle_fallback": mean_from_stats(float(stats[2]), n_fallback),
                "prior_angle_measured": mean_from_stats(float(stats[4]), n_measured),
                "prior_angle_fallback": mean_from_stats(float(stats[5]), n_fallback)}


__all__ = ["GravityLoss"]
