"""
Contact vertex loss functions for 3D human body contact prediction.

Combines three complementary losses adapted from InteractVLM
(Dwivedi et al., 2025) for direct 3D vertex prediction:

  1. Focal BCE   — handles class imbalance, focuses on hard examples
  2. Dice        — optimises overlap for small sparse contact regions
  3. L1 Sparsity — regularises towards sparse contact predictions

NOTE on InteractVLM bug:
  Their HumanContact3DPredictor (components.py:243) thresholds averaged
  probabilities to binary BEFORE passing them to the loss:
      pred_3d_contacts = (pred_3d_contacts > threshold).to(dtype)
  Binary values have no gradient information → focal loss is ineffective.
  Here we always work with continuous sigmoid probabilities from raw logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContactLoss(nn.Module):
    """
    Combined contact vertex loss: Focal BCE + Dice + L1 Sparsity.

    Args:
        focal_alpha:     Class-balance factor for focal loss (default 0.25).
        focal_gamma:     Focusing exponent — higher → harder examples weighted more (default 2.0).
        focal_weight:    Weight of focal BCE component in total loss (default 2.0).
        dice_weight:     Weight of dice component in total loss (default 0.5).
        dice_eps:        Epsilon for numerical stability in dice denominator (default 1e-5).
        sparsity_weight: Weight of L1 sparsity regularisation (default 0.01).

    Forward inputs:
        logits:         [B, V] raw (pre-sigmoid) contact logits.
        targets:        [B, V] float binary ground-truth (0 or 1).
        vertex_weights: [B, V] float per-vertex importance weights, optional.
                        OOB vertices should have weight 0.0; visible vertices 1.0.

    Returns:
        (total_loss, component_dict) where component_dict has keys:
        'focal_bce', 'dice', 'sparsity', values are Python floats for logging.
    """

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        focal_weight: float = 2.0,
        dice_weight: float = 0.5,
        dice_eps: float = 1e-5,
        sparsity_weight: float = 0.01,
    ):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.dice_eps = dice_eps
        self.sparsity_weight = sparsity_weight

    # ------------------------------------------------------------------
    # Component A: Focal BCE
    # ------------------------------------------------------------------

    def _focal_bce(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        vertex_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Asymmetric Focal BCE operating on raw logits.

        bce    = BCE_with_logits(logits, targets)              [B, V]
        pt     = exp(-bce)                                      [B, V]
        alpha_t = alpha  for positives (target=1)  — upweights the rare class
                = 1-alpha for negatives (target=0) — downweights the majority
        loss   = alpha_t * (1 - pt)^gamma * bce               [B, V]

        NOTE: alpha should be set > 0.5 when positives are rare (e.g. 0.75 for
        14% contact rate) so that positive vertices receive MORE weight.
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # [B, V]
        pt = torch.exp(-bce)
        # Asymmetric alpha: alpha for positives, (1-alpha) for negatives
        alpha_t = self.focal_alpha * targets + (1.0 - self.focal_alpha) * (1.0 - targets)
        focal = alpha_t * (1.0 - pt) ** self.focal_gamma * bce  # [B, V]

        if vertex_weights is not None:
            focal = focal * vertex_weights
            denom = vertex_weights.sum().clamp(min=1.0)
            return focal.sum() / denom
        return focal.mean()

    # ------------------------------------------------------------------
    # Component B: Dice Loss
    # ------------------------------------------------------------------

    def _dice_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        vertex_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Soft Dice loss computed per sample then averaged over the batch.

        dice_i = 1 - (2 * sum(p_i * t_i) + eps) / (sum(p_i) + sum(t_i) + eps)

        vertex_weights zero out OOB vertices in both prediction and target.
        """
        probs = torch.sigmoid(logits)  # [B, V]

        if vertex_weights is not None:
            probs = probs * vertex_weights
            targets = targets * vertex_weights

        # Per-sample sums over vertices
        intersection = (probs * targets).sum(dim=1)          # [B]
        sum_pred = probs.sum(dim=1)                          # [B]
        sum_gt = targets.sum(dim=1)                          # [B]

        dice = 1.0 - (2.0 * intersection + self.dice_eps) / (sum_pred + sum_gt + self.dice_eps)
        return dice.mean()

    # ------------------------------------------------------------------
    # Component C: L1 Sparsity Regularisation
    # ------------------------------------------------------------------

    def _sparsity_loss(
        self,
        logits: torch.Tensor,
        vertex_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        L1 sparsity: penalise the mean predicted probability.
        Encourages the model to predict sparse contacts (~14-15% of vertices).

        Uses weighted mean when vertex_weights provided.
        """
        probs = torch.sigmoid(logits)  # [B, V]

        if vertex_weights is not None:
            w_sum = vertex_weights.sum().clamp(min=1.0)
            return (probs * vertex_weights).sum() / w_sum
        return probs.mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        vertex_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute combined contact loss.

        Returns:
            total_loss:    Scalar loss tensor (differentiable).
            component_dict: {'focal_bce': float, 'dice': float,
                             'sparsity': float, 'total': float}
        """
        focal = self._focal_bce(logits, targets, vertex_weights)
        dice = self._dice_loss(logits, targets, vertex_weights)
        sparsity = self._sparsity_loss(logits, vertex_weights)

        total = (
            self.focal_weight * focal
            + self.dice_weight * dice
            + self.sparsity_weight * sparsity
        )

        if torch.isnan(total):
            print(
                f"[ContactLoss] NaN detected! "
                f"focal={focal.item():.6f}  dice={dice.item():.6f}  "
                f"sparsity={sparsity.item():.6f}"
            )

        return total, {
            "focal_bce": focal.item(),
            "dice": dice.item(),
            "sparsity": sparsity.item(),
            "total": total.item(),
        }


# ---------------------------------------------------------------------------
# Debug / verification (run: python train/losses.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("ContactLoss — debug verification")
    print("=" * 60)

    B, V = 4, 18439
    # Use recommended hyperparameters (alpha=0.75 for rare positive class)
    loss_fn = ContactLoss(focal_alpha=0.75, focal_gamma=2.0, focal_weight=5.0,
                          dice_weight=0.5, sparsity_weight=0.001)
    passed = 0
    failed = 0

    def check(name, condition, info=""):
        global passed, failed
        status = "PASS" if condition else "FAIL"
        marker = "✓" if condition else "✗"
        print(f"  [{status}] {marker} {name}" + (f"  ({info})" if info else ""))
        if condition:
            passed += 1
        else:
            failed += 1

    # ---- Case 1: all-zeros logits + all-zeros targets ----
    # With asymmetric alpha=0.75: negatives get (1-alpha)=0.25 weight → small focal
    print("\nCase 1: all-zeros logits, all-zeros targets")
    logits = torch.zeros(B, V)
    targets = torch.zeros(B, V)
    total, d = loss_fn(logits, targets)
    check("no NaN", not torch.isnan(total), f"total={total.item():.4f}")
    check("no Inf", not torch.isinf(total))
    check("focal ≥ 0", d["focal_bce"] >= 0, f"{d['focal_bce']:.4f}")
    check("dice ≈ 1.0 (no overlap)", abs(d["dice"] - 1.0) < 0.01, f"{d['dice']:.4f}")
    check("sparsity ≈ 0.5 (sigmoid(0)=0.5)", abs(d["sparsity"] - 0.5) < 0.01, f"{d['sparsity']:.4f}")

    # ---- Case 2: confidently wrong logits + all-ones targets ----
    # Focal loss intentionally down-weights uncertain predictions (logits≈0).
    # Use large negative logits to simulate "confidently predicting no contact".
    print("\nCase 2: large-negative logits (-5), all-ones targets (confidently wrong)")
    logits = torch.full((B, V), -5.0)
    targets = torch.ones(B, V)
    total, d = loss_fn(logits, targets)
    check("no NaN", not torch.isnan(total))
    check("high focal (confidently wrong → high loss)", d["focal_bce"] > 1.0, f"{d['focal_bce']:.4f}")
    check("dice near 1 (no overlap)", d["dice"] > 0.9, f"{d['dice']:.4f}")

    # ---- Case 3: large positive logits + all-ones targets (best case) ----
    # Focal → 0 when pt→1 regardless of alpha (the (1-pt)^gamma term kills it)
    print("\nCase 3: large positive logits (+10), all-ones targets (min loss)")
    logits = torch.full((B, V), 10.0)
    targets = torch.ones(B, V)
    total, d = loss_fn(logits, targets)
    check("no NaN", not torch.isnan(total))
    check("focal near 0 (pt→1 kills focal term)", d["focal_bce"] < 0.001, f"{d['focal_bce']:.6f}")
    check("dice near 0 (good overlap)", d["dice"] < 0.01, f"{d['dice']:.6f}")
    check("sparsity near 1.0", d["sparsity"] > 0.99, f"{d['sparsity']:.4f}")

    # ---- Case 3b: asymmetric alpha — positives weighted more than negatives ----
    print("\nCase 3b: asymmetric alpha — positives vs negatives at same uncertainty")
    logits_unc = torch.zeros(B, V)
    targets_all_pos = torch.ones(B, V)
    targets_all_neg = torch.zeros(B, V)
    _, d_pos = loss_fn(logits_unc, targets_all_pos)
    _, d_neg = loss_fn(logits_unc, targets_all_neg)
    check(
        "focal for positives > focal for negatives (alpha=0.75 > 1-alpha=0.25)",
        d_pos["focal_bce"] > d_neg["focal_bce"],
        f"pos={d_pos['focal_bce']:.4f}  neg={d_neg['focal_bce']:.4f}",
    )

    # ---- Case 4: realistic 14% contact sparsity ----
    print("\nCase 4: realistic 14% contact rate")
    torch.manual_seed(0)
    logits = torch.randn(B, V) * 0.5
    targets = (torch.rand(B, V) < 0.14).float()
    total, d = loss_fn(logits, targets)
    check("no NaN", not torch.isnan(total))
    check("no Inf", not torch.isinf(total))
    check("total loss reasonable (< 10)", total.item() < 10.0, f"total={total.item():.4f}")
    print(f"    focal={d['focal_bce']:.4f}  dice={d['dice']:.4f}  sparsity={d['sparsity']:.4f}")

    # ---- Case 5: vertex_weights masking ----
    print("\nCase 5: vertex_weights — first half zeroed out")
    logits = torch.randn(B, V)
    targets = (torch.rand(B, V) < 0.14).float()
    # All weights=1 except last V//2 vertices set to 0
    vw = torch.ones(B, V)
    vw[:, V // 2 :] = 0.0
    total_masked, d_masked = loss_fn(logits, targets, vertex_weights=vw)
    total_full, d_full = loss_fn(logits[:, : V // 2], targets[:, : V // 2])
    check("no NaN with weights", not torch.isnan(total_masked))
    check(
        "masked loss ≈ half-vertex loss",
        abs(d_masked["focal_bce"] - d_full["focal_bce"]) < 0.05,
        f"masked={d_masked['focal_bce']:.4f}  half={d_full['focal_bce']:.4f}",
    )

    # ---- Case 6: gradient flow ----
    print("\nCase 6: gradient flow")
    logits = torch.randn(B, V, requires_grad=True)
    targets = (torch.rand(B, V) < 0.14).float()
    total, _ = loss_fn(logits, targets)
    total.backward()
    check("logits.grad is not None", logits.grad is not None)
    check("grad has no NaN", not torch.isnan(logits.grad).any())
    check("grad has no Inf", not torch.isinf(logits.grad).any())
    grad_norm = logits.grad.norm().item()
    check("grad norm > 0", grad_norm > 0, f"norm={grad_norm:.6f}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
