"""Per-target contact losses: mask-correct Focal BCE + Dice + L1 sparsity.

Each supervision target (``vertex`` / ``joint``) has its own :class:`ContactLoss`
with its own hyper-parameters and receives ``(logits, gt, mask)``, all ``[B, D]``.
``mask`` is a per-element supervision weight (0 = ignore, >0 = supervise; a
confidence weight when > 1 element-fraction). :class:`MultiTargetContactLoss`
holds one loss per enabled target and returns the weighted sum.

Reduction rules (mask-correct):

* Focal / sparsity: masked elements contribute exactly 0; the denominator is the
  mask mass (clamped to 1 so an all-masked batch gives 0, graph-safe).
* Dice: averaged over samples in proportion to their mask mass. This preserves
  the usual mean for fully supervised samples while making an entirely
  low-confidence frame less influential than a high-confidence frame.
* A target with zero active elements in the batch contributes exactly ``0.0``
  (still a tensor function of the logits, so ``backward`` is safe).

Components whose configured weight is zero are not evaluated.  Besides the
locally normalised loss, the implementation retains its weighted numerator and
mask mass.  Those additive quantities let DDP form the exact global masked mean
without trying to undo a rank-local normalisation.

NOTE on InteractVLM bug: their ``HumanContact3DPredictor`` thresholds averaged
probabilities to binary *before* the loss, killing focal gradients. Here we
always work with continuous sigmoid probabilities from raw logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def ddp_global_mean_term(
    local_numerator: torch.Tensor,
    global_mass: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    """Return one rank's loss term for an exact DDP global weighted mean.

    DDP averages gradients across ``world_size`` ranks.  Multiplying an additive
    local numerator by ``world_size / clamp(global_mass, 1)`` therefore makes the
    averaged gradient equal ``sum(local_numerator) / clamp(sum(mass), 1)``.  This
    remains exact when a non-empty rank has mask mass below one and when another
    rank has zero mass.
    """
    denominator = global_mass.clamp(min=1.0).to(
        device=local_numerator.device, dtype=local_numerator.dtype)
    return local_numerator * (float(world_size) / denominator)


class ContactLoss(nn.Module):
    """Combined single-target contact loss: Focal BCE + Dice + L1 sparsity.

    :param focal_alpha: Class-balance factor (>0.5 upweights the rare positive class).
    :param focal_gamma: Focusing exponent (higher -> hard examples weighted more).
    :param focal_weight: Weight of the focal-BCE component.
    :param dice_weight: Weight of the dice component.
    :param sparsity_weight: Weight of the L1 sparsity regulariser.
    :param dice_eps: Numerical-stability epsilon in the dice denominator.
    """

    def __init__(
        self,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        focal_weight: float = 5.0,
        dice_weight: float = 0.5,
        sparsity_weight: float = 0.002,
        dice_eps: float = 1e-5,
    ):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.sparsity_weight = sparsity_weight
        self.dice_eps = dice_eps

    @staticmethod
    def _normalizer(mask: torch.Tensor) -> torch.Tensor:
        return mask.sum().clamp(min=1.0)

    def _focal_bce_numerator(
        self, logits: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """Asymmetric focal BCE's additive confidence-weighted numerator."""
        # NaN * 0 is still NaN. Replace ignored logits before evaluating the loss
        # so an unsupervised invalid video frame is absent from the objective.
        safe_logits = torch.where(mask > 0, logits, torch.zeros_like(logits))
        bce = F.binary_cross_entropy_with_logits(safe_logits, gt, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.focal_alpha * gt + (1.0 - self.focal_alpha) * (1.0 - gt)
        focal = alpha_t * (1.0 - pt) ** self.focal_gamma * bce
        return (focal * mask).sum()

    def _focal_bce(self, logits: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Asymmetric focal BCE, mask-weighted mean. ``logits/gt/mask``: ``[B, D]``."""
        return self._focal_bce_numerator(logits, gt, mask) / self._normalizer(mask)

    def _dice_loss_numerator(
        self, logits: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """Soft-Dice's additive numerator after confidence-mass sample weighting."""
        safe_logits = torch.where(mask > 0, logits, torch.zeros_like(logits))
        probs = torch.sigmoid(safe_logits)
        inter = (probs * gt * mask).sum(dim=1)
        sum_pred = (probs * mask).sum(dim=1)
        sum_gt = (gt * mask).sum(dim=1)
        dice = 1.0 - (2.0 * inter + self.dice_eps) / (sum_pred + sum_gt + self.dice_eps)
        sample_weight = mask.sum(dim=1)
        return (dice * sample_weight).sum()

    def _dice_loss(self, logits: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Soft dice per sample, reduced in proportion to supervision/confidence mass."""
        return self._dice_loss_numerator(logits, gt, mask) / self._normalizer(mask)

    @staticmethod
    def _sparsity_loss_numerator(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """L1 sparsity's additive confidence-weighted numerator."""
        safe_logits = torch.where(mask > 0, logits, torch.zeros_like(logits))
        probs = torch.sigmoid(safe_logits)
        return (probs * mask).sum()

    def _sparsity_loss(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mask-weighted mean predicted probability."""
        return self._sparsity_loss_numerator(logits, mask) / self._normalizer(mask)

    def forward(
        self, logits: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Return ``(total_loss, parts)``.

        :param logits: ``[B, D]`` raw contact logits.
        :param gt: ``[B, D]`` float binary ground truth.
        :param mask: ``[B, D]`` float supervision weights (0 = ignore).
        :returns: ``(total, {'focal', 'dice', 'sparsity', 'loss', 'n_active'})``.
        """
        mass = mask.sum()
        normalizer = mass.clamp(min=1.0)
        # Keep a graph-connected zero so an all-masked batch remains backward-safe.
        safe_logits = torch.where(mask > 0, logits, torch.zeros_like(logits))
        numerator = safe_logits.sum() * 0.0
        parts = {}
        if self.focal_weight != 0.0:
            focal_numerator = self._focal_bce_numerator(logits, gt, mask)
            numerator = numerator + self.focal_weight * focal_numerator
            parts["focal"] = (focal_numerator / normalizer).item()
        if self.dice_weight != 0.0:
            dice_numerator = self._dice_loss_numerator(logits, gt, mask)
            numerator = numerator + self.dice_weight * dice_numerator
            parts["dice"] = (dice_numerator / normalizer).item()
        if self.sparsity_weight != 0.0:
            sparsity_numerator = self._sparsity_loss_numerator(logits, mask)
            numerator = numerator + self.sparsity_weight * sparsity_numerator
            parts["sparsity"] = (sparsity_numerator / normalizer).item()

        total = numerator / normalizer
        parts.update({
            "loss": total.item(),
            "n_active": float((mask > 0).sum().item()),
            "weight_mass": float(mass.item()),
            # Internal autograd value used to build an exact global DDP mean.
            "loss_numerator_tensor": numerator,
        })
        return total, parts


class MultiTargetContactLoss(nn.Module):
    """Sum of per-target :class:`ContactLoss`, weighted by ``targets.*.weight``.

    Built from a resolved run config; only enabled targets get a loss.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        targets = cfg["contact"]["targets"]
        dice_eps = float(cfg["loss"]["dice_eps"])
        self.losses = nn.ModuleDict()
        self.weights: dict[str, float] = {}
        for name in ("vertex", "joint"):
            spec = targets[name]
            if not spec["enabled"]:
                continue
            lcfg = spec["loss"]
            self.losses[name] = ContactLoss(
                focal_alpha=float(lcfg["focal_alpha"]),
                focal_gamma=float(lcfg["focal_gamma"]),
                focal_weight=float(lcfg["focal_weight"]),
                dice_weight=float(lcfg["dice_weight"]),
                sparsity_weight=float(lcfg["sparsity_weight"]),
                dice_eps=dice_eps,
            )
            self.weights[name] = float(spec["weight"])

    @property
    def target_names(self) -> list[str]:
        return list(self.losses.keys())

    def forward(
        self,
        logits_by_target: dict[str, torch.Tensor],
        targets_by_target: dict[str, dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, dict]]:
        """Return ``(total, {target: parts})`` over the enabled targets."""
        total: torch.Tensor | None = None
        parts: dict[str, dict] = {}
        for name, loss_fn in self.losses.items():
            gt = targets_by_target[name]["gt"]
            mask = targets_by_target[name]["mask"]
            loss, part = loss_fn(logits_by_target[name], gt, mask)
            weighted = self.weights[name] * loss
            total = weighted if total is None else total + weighted
            # Additive across ranks; unlike the locally normalised loss, this can
            # be divided by the global confidence mass exactly under DDP.
            part["weighted_numerator_tensor"] = (
                self.weights[name] * part["loss_numerator_tensor"])
            parts[name] = part
        return total, parts
