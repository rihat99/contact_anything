"""Per-target contact losses: mask-correct Focal BCE + Dice + L1 sparsity.

Each supervision target (``vertex`` / ``joint``) has its own :class:`ContactLoss`
with its own hyper-parameters and receives ``(logits, gt, mask)``, all ``[B, D]``.
``mask`` is a per-element supervision weight (0 = ignore, >0 = supervise; a
confidence weight when > 1 element-fraction). :class:`MultiTargetContactLoss`
holds one loss per enabled target and returns the weighted sum.

Reduction rules (mask-correct):

* Focal / sparsity: masked elements contribute exactly 0; the denominator is the
  mask mass (clamped to 1 so an all-masked batch gives 0, graph-safe).
* Dice: averaged only over samples with positive mask mass.
* A target with zero active elements in the batch contributes exactly ``0.0``
  (still a tensor function of the logits, so ``backward`` is safe).

NOTE on InteractVLM bug: their ``HumanContact3DPredictor`` thresholds averaged
probabilities to binary *before* the loss, killing focal gradients. Here we
always work with continuous sigmoid probabilities from raw logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def _focal_bce(self, logits: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Asymmetric focal BCE, mask-weighted mean. ``logits/gt/mask``: ``[B, D]``."""
        bce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.focal_alpha * gt + (1.0 - self.focal_alpha) * (1.0 - gt)
        focal = alpha_t * (1.0 - pt) ** self.focal_gamma * bce
        return (focal * mask).sum() / mask.sum().clamp(min=1.0)

    def _dice_loss(self, logits: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Soft dice, per sample, averaged only over samples with positive mask mass."""
        probs = torch.sigmoid(logits)
        inter = (probs * gt * mask).sum(dim=1)
        sum_pred = (probs * mask).sum(dim=1)
        sum_gt = (gt * mask).sum(dim=1)
        dice = 1.0 - (2.0 * inter + self.dice_eps) / (sum_pred + sum_gt + self.dice_eps)
        has_mask = (mask.sum(dim=1) > 0).to(dice.dtype)
        return (dice * has_mask).sum() / has_mask.sum().clamp(min=1.0)

    def _sparsity_loss(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mask-weighted mean predicted probability."""
        probs = torch.sigmoid(logits)
        return (probs * mask).sum() / mask.sum().clamp(min=1.0)

    def forward(
        self, logits: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Return ``(total_loss, parts)``.

        :param logits: ``[B, D]`` raw contact logits.
        :param gt: ``[B, D]`` float binary ground truth.
        :param mask: ``[B, D]`` float supervision weights (0 = ignore).
        :returns: ``(total, {'focal', 'dice', 'sparsity', 'loss', 'n_active'})``.
        """
        focal = self._focal_bce(logits, gt, mask)
        dice = self._dice_loss(logits, gt, mask)
        sparsity = self._sparsity_loss(logits, mask)
        total = self.focal_weight * focal + self.dice_weight * dice + self.sparsity_weight * sparsity

        parts = {
            "focal": focal.item(),
            "dice": dice.item(),
            "sparsity": sparsity.item(),
            "loss": total.item(),
            "n_active": float((mask > 0).sum().item()),
        }
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
            parts[name] = part
        return total, parts
