"""
Contact loss: weighted BCE for per-vertex contacts + BCE for per-part contacts.

Replaces the previous triple loss (Focal BCE + Dice + L1 Sparsity) with a
simpler formulation:

  vertex_loss = BCE_with_logits(vertex_logits, vertex_targets,
                                pos_weight=vertex_pos_weight)
  part_loss   = BCE_with_logits(part_logits, part_targets)
  total       = vertex_weight * vertex_loss + part_weight * part_loss

pos_weight handles class imbalance for per-vertex contacts (~14% contact rate
→ pos_weight ≈ neg/pos ≈ 6, typically tuned to 5–15).  Part contacts are more
balanced (~30-50% contact rate per part) and do not need special weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContactLoss(nn.Module):
    """
    Combined contact loss for vertex and part contact prediction.

    Args:
        vertex_pos_weight: Scalar weight applied to positive (contact) class in
                           the vertex BCE loss.  Compensates for the ~14% contact
                           rate: set to roughly neg/pos ≈ 6–15.  Default 10.0.
        vertex_weight:     Scale for the vertex BCE component (default 1.0).
        part_weight:       Scale for the part BCE component (default 0.5).
                           Part loss acts as auxiliary supervision.

    Forward args:
        vertex_logits:  [B, V] raw pre-sigmoid vertex contact logits.
        vertex_targets: [B, V] float binary ground-truth (0 or 1).
        part_logits:    [B, 24] raw pre-sigmoid part contact logits.
        part_targets:   [B, 24] float binary ground-truth (0 or 1).

    Returns:
        (total_loss, component_dict) where component_dict has keys:
        'vertex_bce', 'part_bce', 'total'
    """

    def __init__(
        self,
        vertex_pos_weight: float = 10.0,
        vertex_weight: float = 1.0,
        part_weight: float = 0.5,
    ):
        super().__init__()
        self.vertex_weight = vertex_weight
        self.part_weight = part_weight
        self.register_buffer(
            'pos_weight',
            torch.tensor([vertex_pos_weight], dtype=torch.float32),
        )

    def forward(
        self,
        vertex_logits: torch.Tensor,
        vertex_targets: torch.Tensor,
        part_logits: torch.Tensor,
        part_targets: torch.Tensor,
    ) -> tuple:
        vertex_loss = F.binary_cross_entropy_with_logits(
            vertex_logits,
            vertex_targets,
            pos_weight=self.pos_weight,
        )
        part_loss = F.binary_cross_entropy_with_logits(
            part_logits,
            part_targets,
        )
        total = self.vertex_weight * vertex_loss + self.part_weight * part_loss

        return total, {
            'vertex_bce': vertex_loss.item(),
            'part_bce':   part_loss.item(),
            'total':      total.item(),
        }


# ---------------------------------------------------------------------------
# Quick verification (run: python train/losses.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    B, V, P = 4, 6890, 24
    loss_fn = ContactLoss(vertex_pos_weight=10.0, vertex_weight=1.0, part_weight=0.5)
    passed = failed = 0

    def check(name, cond, info=""):
        global passed, failed
        if cond:
            passed += 1
            print(f"  [PASS] {name}" + (f"  ({info})" if info else ""))
        else:
            failed += 1
            print(f"  [FAIL] {name}" + (f"  ({info})" if info else ""))

    print("=== ContactLoss verification ===")

    # Case 1: all zeros
    vl = torch.zeros(B, V); vt = torch.zeros(B, V)
    pl = torch.zeros(B, P); pt = torch.zeros(B, P)
    tot, d = loss_fn(vl, vt, pl, pt)
    check("no NaN all-zeros", not torch.isnan(tot), f"total={tot.item():.4f}")

    # Case 2: gradient flow
    vl = torch.randn(B, V, requires_grad=True)
    pl = torch.randn(B, P, requires_grad=True)
    vt = (torch.rand(B, V) < 0.14).float()
    pt = (torch.rand(B, P) < 0.40).float()
    tot, d = loss_fn(vl, vt, pl, pt)
    tot.backward()
    check("vertex grad not None", vl.grad is not None)
    check("part grad not None", pl.grad is not None)
    check("no NaN in vertex grad", not torch.isnan(vl.grad).any())
    check("no NaN in part grad", not torch.isnan(pl.grad).any())
    check("total > 0", tot.item() > 0, f"total={tot.item():.4f}")
    check("vertex_bce in dict", 'vertex_bce' in d)
    check("part_bce in dict", 'part_bce' in d)
    print(f"  vertex_bce={d['vertex_bce']:.4f}  part_bce={d['part_bce']:.4f}")

    # Case 3: pos_weight registered as buffer (survives .to(device))
    loss_cpu = loss_fn
    check("pos_weight is buffer", isinstance(loss_cpu.pos_weight, torch.Tensor))

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
