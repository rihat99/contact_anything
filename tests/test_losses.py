"""Per-target contact loss: ported numeric self-tests + mask-correct reductions."""
from __future__ import annotations

import pytest
import torch

from contact.losses import ContactLoss, MultiTargetContactLoss

# Recommended hyper-parameters (alpha=0.75 upweights the rare positive class).
_LOSS = ContactLoss(focal_alpha=0.75, focal_gamma=2.0, focal_weight=5.0,
                    dice_weight=0.5, sparsity_weight=0.001)


def _ones_mask(x):
    return torch.ones_like(x)


# ---------------------------------------------------------------- ported numeric

def test_all_zero_logits_and_targets():
    logits = torch.zeros(4, 512)
    gt = torch.zeros(4, 512)
    total, d = _LOSS(logits, gt, _ones_mask(logits))
    assert not torch.isnan(total) and not torch.isinf(total)
    assert d["focal"] >= 0
    assert d["dice"] == pytest.approx(1.0, abs=0.01)             # no overlap
    assert d["sparsity"] == pytest.approx(0.5, abs=0.01)        # sigmoid(0)=0.5


def test_confident_correct_minimises_focal_and_dice():
    logits = torch.full((4, 512), 10.0)
    gt = torch.ones(4, 512)
    total, d = _LOSS(logits, gt, _ones_mask(logits))
    assert d["focal"] < 1e-3                                    # pt->1 kills focal
    assert d["dice"] < 1e-2                                     # perfect overlap
    assert d["sparsity"] > 0.99


def test_asymmetric_alpha_upweights_positives():
    logits = torch.zeros(4, 512)
    _, d_pos = _LOSS(logits, torch.ones_like(logits), _ones_mask(logits))
    _, d_neg = _LOSS(logits, torch.zeros_like(logits), _ones_mask(logits))
    assert d_pos["focal"] > d_neg["focal"]                      # alpha=0.75 > 1-alpha=0.25


def test_gradient_flow():
    logits = torch.randn(4, 512, requires_grad=True)
    gt = (torch.rand(4, 512) < 0.14).float()
    total, _ = _LOSS(logits, gt, _ones_mask(logits))
    total.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    assert logits.grad.norm().item() > 0


# ---------------------------------------------------------------- mask semantics

def test_masked_elements_contribute_zero():
    torch.manual_seed(0)
    logits = torch.randn(4, 512)
    gt = (torch.rand(4, 512) < 0.14).float()
    half = torch.ones(4, 512)
    half[:, 256:] = 0.0
    _, masked = _LOSS(logits, gt, half)
    _, first_half = _LOSS(logits[:, :256], gt[:, :256], torch.ones(4, 256))
    assert masked["focal"] == pytest.approx(first_half["focal"], abs=1e-5)
    assert masked["sparsity"] == pytest.approx(first_half["sparsity"], abs=1e-5)
    assert masked["n_active"] == 4 * 256


def test_fully_masked_sample_excluded_from_dice():
    torch.manual_seed(1)
    logits = torch.randn(3, 128)
    gt = (torch.rand(3, 128) < 0.2).float()
    mask = torch.ones(3, 128)
    mask[1] = 0.0                                               # sample 1 fully masked
    _, both = _LOSS(logits, gt, mask)
    # dice must equal the mean over the two *supervised* samples only
    dice0 = _LOSS(logits[[0]], gt[[0]], torch.ones(1, 128))[1]["dice"]
    dice2 = _LOSS(logits[[2]], gt[[2]], torch.ones(1, 128))[1]["dice"]
    assert both["dice"] == pytest.approx((dice0 + dice2) / 2, abs=1e-5)


def test_zero_active_target_is_zero_and_graph_safe():
    logits = torch.randn(4, 512, requires_grad=True)
    gt = (torch.rand(4, 512) < 0.2).float()
    zero_mask = torch.zeros(4, 512)
    total, d = _LOSS(logits, gt, zero_mask)
    assert d["loss"] == 0.0
    assert d["n_active"] == 0.0
    assert total.requires_grad                                 # still a function of logits
    total.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) == 0.0               # zero grad, no NaN


# ---------------------------------------------------------------- multi-target

def _multi_cfg():
    def loss(alpha):
        return {"focal_alpha": alpha, "focal_gamma": 2.0, "focal_weight": 5.0,
                "dice_weight": 0.5, "sparsity_weight": 0.0}
    return {
        "loss": {"dice_eps": 1e-5},
        "contact": {"targets": {
            "vertex": {"enabled": True, "weight": 1.0, "loss": loss(0.75)},
            "joint": {"enabled": True, "weight": 2.0, "loss": loss(0.5)},
        }},
    }


def test_multi_target_weighted_sum():
    torch.manual_seed(2)
    multi = MultiTargetContactLoss(_multi_cfg())
    assert multi.target_names == ["vertex", "joint"]
    logits = {"vertex": torch.randn(2, 100), "joint": torch.randn(2, 22)}
    targets = {
        "vertex": {"gt": (torch.rand(2, 100) < 0.2).float(), "mask": torch.ones(2, 100)},
        "joint": {"gt": (torch.rand(2, 22) < 0.3).float(), "mask": torch.ones(2, 22)},
    }
    total, parts = multi(logits, targets)
    expected = 1.0 * parts["vertex"]["loss"] + 2.0 * parts["joint"]["loss"]
    assert total.item() == pytest.approx(expected, rel=1e-5)


def test_multi_target_inactive_target_drops_out():
    torch.manual_seed(3)
    multi = MultiTargetContactLoss(_multi_cfg())
    logits = {"vertex": torch.randn(2, 100), "joint": torch.randn(2, 22)}
    targets = {
        "vertex": {"gt": (torch.rand(2, 100) < 0.2).float(), "mask": torch.ones(2, 100)},
        "joint": {"gt": torch.zeros(2, 22), "mask": torch.zeros(2, 22)},   # unsupervised this batch
    }
    total, parts = multi(logits, targets)
    assert parts["joint"]["loss"] == 0.0
    assert total.item() == pytest.approx(1.0 * parts["vertex"]["loss"], rel=1e-5)
