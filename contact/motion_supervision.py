"""Supervised per-joint velocity/acceleration loss against kindyn ground truth.

Reads ``out["motion"]["joint_motion"] (B, K, 6)`` — **standardized** root-frame
linear velocity (``[..., 0:3]``) and acceleration (``[..., 3:6]``) — and the
collated ``motion_gt (B, K, 6)`` / ``motion_valid (B)`` / ``motion_outlier
(B, K)`` / ``motion_rot (B, 3, 3)`` / ``motion_omega (B, 3)`` batch keys, whose
slot order is ``motion_supervision.joint_names`` (default: all of
:data:`~contact.data.climbing_corpus.MOTION_JOINT_NAMES` — ``left_wrist,
right_wrist, left_foot, right_foot, left_ankle, right_ankle, pelvis``, pelvis
LAST). GT arrives in **physical** units (m/s, m/s²) and is standardized here
with the config's pinned ``motion_supervision.standardize.{mean,std}`` table, so
the objective is reproducible from a checkpoint's stored config alone (a
registered buffer would not be serialised — see ``contact/checkpoint.py``).

Two terms, mirroring the :class:`~contact.force_supervision.ForceSupervisedLoss`
term contract (each returns ``(weighted_numerator, mass)`` so the trainer's
:func:`~contact.losses.ddp_global_mean_term` reduction gives the exact global
mean under DDP):

- ``vel`` — smooth-L1 (Huber) between prediction and standardized GT, summed
  over the 3 components, on valid ``(frame, joint)`` entries.
- ``acc`` — the same on the acceleration triple.

Per-joint standardization is what makes the two scales comparable: the wrists'
root-frame acceleration std is ~2.5× the pelvis's, so a single global scaler
would drown the pelvis token (the one carrying the v1 comparison bar).

Masking. An entry contributes when the frame is motion-valid (central-difference
support present, outside the scene-edge/gap trims) **and** frame-valid. During
**training** the per-``(frame, joint)`` outlier bit additionally zeroes both the
vel and acc terms of that entry — the same kindyn position spike contaminates
both. Evaluation never filters (``exclude_outliers=False``), matching the v1
probe's protocol.

Diagnostics are de-standardized: per-joint 3-D RMSE and pooled 3-component
Pearson statistics in the **target** axes, plus the Pearson statistics of the
**world-vertical** component (see :func:`to_world_linear`; vertical is world
**y**, positive downward — kindyn's ``gravity_world`` is exactly ``[0, 1, 0]``).
They are returned as raw sums so the trainer can all-reduce them into an exact
global correlation.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .data.climbing_corpus import MOTION_JOINT_NAMES

_TERM_NAMES = ("vel", "acc")

#: Columns of the per-(quantity, joint) statistics tensor ``[2, K, 12]``. The
#: ``*_vert`` block feeds the world-vertical Pearson r, the ``*_3d`` block the
#: pooled 3-component Pearson r in target axes (its sample count is ``3 * n``).
STAT_COLUMNS = ("n", "sum_pred_vert", "sum_gt_vert", "sum_pred_vert_sq",
                "sum_gt_vert_sq", "sum_pred_gt_vert", "sum_sq_err_3d",
                "sum_pred_3d", "sum_gt_3d", "sum_pred_3d_sq", "sum_gt_3d_sq",
                "sum_pred_gt_3d")


def to_world_linear(
    vel: Tensor, acc: Tensor, rot: Tensor, omega: Tensor, twist_slots: Tensor,
) -> tuple[Tensor, Tensor]:
    """Root-axis linear vel/acc ``(rows, K, 3)`` -> world axes.

    ``rotated_world`` slots are a plain re-expression (``x_world = R x_root``).
    A ``twist`` slot additionally carries the Coriolis term, since BVR's body
    twist satisfies ``a_world = R (a_body + omega x v_body)``.

    :param rot: ``(rows, 3, 3)`` world-from-root rotation.
    :param omega: ``(rows, 3)`` root body angular velocity.
    :param twist_slots: ``(K,)`` bool — slots whose target is a body twist.
    """
    coriolis = torch.linalg.cross(
        omega[:, None, :].expand_as(vel), vel, dim=-1)                 # (rows, K, 3)
    acc = acc + coriolis * twist_slots[None, :, None].to(acc.dtype)
    return (torch.einsum("rij,rkj->rki", rot, vel),
            torch.einsum("rij,rkj->rki", rot, acc))


class MotionSupervisedLoss:
    """GT vel/acc supervision for the motion-token branch.

    :param cfg: resolved run config; reads ``motion_supervision.*``.
    :param device: device the loss runs on (predictions are moved to it).
    :param dtype: floating dtype (float32).
    """

    def __init__(
        self,
        cfg: dict,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        ms = cfg["motion_supervision"]
        self.target_frame = str(ms["target_frame"])
        self.joint_names = tuple(ms.get("joint_names") or MOTION_JOINT_NAMES)
        # Only the root slot can be a twist; the limbs are always rotated_world.
        twist = str(ms.get("root_convention", "twist")) == "twist"
        self.twist_slots = torch.tensor(
            [twist and name == "pelvis" for name in self.joint_names],
            dtype=torch.bool, device=device)
        loss_cfg = ms["loss"]
        self.weights = {name: float(loss_cfg[name]) for name in _TERM_NAMES}
        self.huber_delta = float(loss_cfg["huber_delta"])
        self.device = torch.device(device)
        self.dtype = dtype
        # [K, 2, 3] -> [1, K, 6] so it broadcasts over the row axis and lines up
        # with the head's (vel | acc) output layout.
        mean = torch.tensor(ms["standardize"]["mean"], dtype=dtype)
        std = torch.tensor(ms["standardize"]["std"], dtype=dtype)
        self.mean = mean.reshape(1, mean.shape[0], 6).to(self.device)
        self.std = std.reshape(1, std.shape[0], 6).to(self.device)

    def __call__(
        self, out: dict, batch: dict, exclude_outliers: bool = True,
    ) -> tuple[Tensor, dict[str, Any]]:
        return self.forward(out, batch, exclude_outliers)

    def forward(
        self, out: dict, batch: dict, exclude_outliers: bool = True,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Return ``(total, parts)``.

        :param out: forward output — reads ``out["motion"]["joint_motion"]
            (B, K, 6)`` in standardized units (grads live).
        :param batch: reads ``motion_gt (B, K, 6)`` (physical units),
            ``motion_valid (B)``, ``motion_outlier (B, K)``,
            ``motion_rot (B, 3, 3)``, ``frame_valid (B)`` and ``seq_len``.
        :param exclude_outliers: apply the per-``(frame, joint)`` outlier bit.
            ``True`` in training, ``False`` at evaluation.
        :returns: ``(total, parts)`` where ``parts["terms"][name]`` carries
            ``weighted_numerator_tensor`` + ``weight_mass`` for exact DDP
            reduction and ``parts["stats"]`` is the ``[2, K, 7]`` float64 tensor
            of Pearson/RMSE sufficient statistics (rows: vel, acc).
        """
        pred = out["motion"]["joint_motion"].to(self.device, self.dtype)   # (B, K, 6)
        # Graph-connected zero touching every motion param (DDP: they must stay
        # on the backward graph even when a batch has no supervised frames).
        zero_touch = pred.sum() * 0.0

        gt = batch["motion_gt"].to(self.device, self.dtype)                # (B, K, 6)
        if pred.shape != gt.shape:
            raise ValueError(
                f"motion prediction {tuple(pred.shape)} does not match GT "
                f"{tuple(gt.shape)} — motion_keypoint_indices and the dataset's "
                f"motion joints must agree")
        if self.mean.shape[1] != pred.shape[1]:
            raise ValueError(
                f"motion_supervision.standardize has {self.mean.shape[1]} joint "
                f"rows but the model predicts {pred.shape[1]} motion tokens")
        valid = (batch["motion_valid"] & batch["frame_valid"]).to(self.device)  # (B)
        outlier = batch["motion_outlier"].to(self.device)                 # (B, K)
        rot = batch["motion_rot"].to(self.device, self.dtype)             # (B, 3, 3)
        omega = batch["motion_omega"].to(self.device, self.dtype)         # (B, 3)

        if self.target_frame == "center":
            seq_len = int(batch["seq_len"])
            if seq_len % 2 == 0:
                raise ValueError(
                    f"center-frame supervision requires an odd seq_len; got {seq_len}")
            n_clips = pred.shape[0] // seq_len
            center = seq_len // 2

            def _center(value: Tensor) -> Tensor:
                return value.reshape(n_clips, seq_len, *value.shape[1:])[:, center]

            pred, gt = _center(pred), _center(gt)
            valid, outlier = _center(valid), _center(outlier)
            rot, omega = _center(rot), _center(omega)
        elif self.target_frame != "all":
            raise ValueError(
                f"target_frame must be 'all' or 'center'; got {self.target_frame!r}")

        mask = valid[:, None].expand_as(outlier)                          # (rows, K)
        if exclude_outliers:
            mask = mask & ~outlier
        n_outlier = int((valid[:, None] & outlier).sum()) if exclude_outliers else 0

        gt_std = (gt - self.mean) / self.std
        huber = F.smooth_l1_loss(
            pred, gt_std, reduction="none", beta=self.huber_delta)        # (rows, K, 6)
        mass = float(mask.sum())
        terms: dict[str, tuple[Tensor, float]] = {
            "vel": ((huber[..., :3].sum(dim=-1) * mask).sum(), mass),
            "acc": ((huber[..., 3:].sum(dim=-1) * mask).sum(), mass),
        }

        diagnostics = self._diagnostics(
            pred, gt, mask, rot, omega, n_outlier, int(valid.sum()))
        return self._assemble(terms, zero_touch, diagnostics)

    @torch.no_grad()
    def _diagnostics(
        self,
        pred: Tensor,
        gt: Tensor,
        mask: Tensor,
        rot: Tensor,
        omega: Tensor,
        n_outlier: int,
        n_rows: int,
    ) -> dict[str, Any]:
        """De-standardized RMSE + Pearson sufficient statistics.

        Two correlations per (quantity, slot): the pooled 3-component one in the
        **target** axes (what the head actually regresses) and the
        **world-vertical** one (:func:`to_world_linear`, so a twist slot picks up
        its Coriolis term). Both predictions and GT go through the same
        conversion, so the comparison stays like-for-like.
        """
        pred_phys = pred * self.std + self.mean                           # (rows, K, 6)
        weight = mask.to(torch.float64)                                   # (rows, K)
        pred_world = to_world_linear(
            pred_phys[..., :3], pred_phys[..., 3:], rot, omega, self.twist_slots)
        gt_world = to_world_linear(
            gt[..., :3], gt[..., 3:], rot, omega, self.twist_slots)

        stats = torch.zeros(2, pred.shape[1], len(STAT_COLUMNS),
                            dtype=torch.float64, device=pred.device)
        rmse = {}
        for i, name in enumerate(_TERM_NAMES):
            sl = slice(0, 3) if name == "vel" else slice(3, 6)
            p = pred_phys[..., sl].to(torch.float64)                      # (rows, K, 3)
            g = gt[..., sl].to(torch.float64)
            # World y (down-positive) of the converted vectors.
            p_vert = pred_world[i][..., 1].to(torch.float64)               # (rows, K)
            g_vert = gt_world[i][..., 1].to(torch.float64)
            sq_err = ((p - g) ** 2).sum(dim=-1)                            # (rows, K)
            stats[i, :, 0] = weight.sum(dim=0)
            stats[i, :, 1] = (p_vert * weight).sum(dim=0)
            stats[i, :, 2] = (g_vert * weight).sum(dim=0)
            stats[i, :, 3] = (p_vert * p_vert * weight).sum(dim=0)
            stats[i, :, 4] = (g_vert * g_vert * weight).sum(dim=0)
            stats[i, :, 5] = (p_vert * g_vert * weight).sum(dim=0)
            stats[i, :, 6] = (sq_err * weight).sum(dim=0)
            # Pooled over the 3 target-axis components (sample count 3 * n).
            weight3 = weight[:, :, None]
            stats[i, :, 7] = (p * weight3).sum(dim=(0, 2))
            stats[i, :, 8] = (g * weight3).sum(dim=(0, 2))
            stats[i, :, 9] = (p * p * weight3).sum(dim=(0, 2))
            stats[i, :, 10] = (g * g * weight3).sum(dim=(0, 2))
            stats[i, :, 11] = (p * g * weight3).sum(dim=(0, 2))
            total_n = float(stats[i, :, 0].sum())
            rmse[f"{name}_rmse"] = float(
                (stats[i, :, 6].sum() / max(total_n, 1.0)) ** 0.5)
        return {
            "stats": stats,
            "n_outlier_excluded": n_outlier,
            "n_supervised_rows": n_rows,
            **rmse,
        }

    def _assemble(
        self,
        terms: dict[str, tuple[Tensor, float]],
        zero_touch: Tensor,
        diagnostics: dict[str, Any],
    ) -> tuple[Tensor, dict[str, Any]]:
        """Weight, normalise, and package the term contract + logging scalars.

        Every nonzero-weight term is always present (mass 0 when it has no data
        this batch), so the term set is fixed by config — not by per-batch
        supervision — which the trainer needs for consistent DDP all-reduce.
        Each term carries the graph-connected ``zero_touch``.
        """
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


def _pearson(n: Tensor, sum_p: Tensor, sum_g: Tensor, sum_pp: Tensor,
             sum_gg: Tensor, sum_pg: Tensor) -> Tensor:
    cov = n * sum_pg - sum_p * sum_g
    var_p = n * sum_pp - sum_p ** 2
    var_g = n * sum_gg - sum_g ** 2
    denom = (var_p.clamp(min=0) * var_g.clamp(min=0)).sqrt()
    return torch.where(
        (n >= 2) & (denom > 0), cov / denom.clamp(min=1e-30),
        torch.full_like(cov, float("nan")))


def pearson_from_stats(stats: Tensor) -> Tensor:
    """Per-(quantity, joint) **world-vertical** Pearson r from the statistics tensor.

    Entries with fewer than two samples or a degenerate variance yield ``nan``.
    """
    return _pearson(stats[..., 0], stats[..., 1], stats[..., 2],
                    stats[..., 3], stats[..., 4], stats[..., 5])


def pearson3d_from_stats(stats: Tensor) -> Tensor:
    """Per-(quantity, joint) Pearson r pooled over the 3 target-axis components.

    Every ``(row, component)`` pair is one sample, so the count is ``3 * n``.
    Unlike the per-axis correlations this one is a single number per slot and
    does not privilege the vertical, which is what the pelvis monitor wants.
    """
    return _pearson(3.0 * stats[..., 0], stats[..., 7], stats[..., 8],
                    stats[..., 9], stats[..., 10], stats[..., 11])


def rmse_from_stats(stats: Tensor) -> Tensor:
    """Per-(quantity, joint) 3-D RMSE (physical units) from the statistics tensor."""
    n = stats[..., 0]
    return torch.where(
        n > 0, (stats[..., 6] / n.clamp(min=1.0)).sqrt(),
        torch.full_like(n, float("nan")))


def gt_rms3d_from_stats(stats: Tensor) -> Tensor:
    """Per-(quantity, joint) 3-D RMS of the GT — the zero-prior's RMSE."""
    n = stats[..., 0]
    return torch.where(
        n > 0, (stats[..., 10] / n.clamp(min=1.0)).sqrt(),
        torch.full_like(n, float("nan")))


__all__ = ["MotionSupervisedLoss", "to_world_linear", "pearson_from_stats",
           "pearson3d_from_stats", "rmse_from_stats", "gt_rms3d_from_stats",
           "STAT_COLUMNS"]
