"""Per-target contact loss: ported numeric self-tests + mask-correct reductions."""
from __future__ import annotations

import pytest
import torch

from contact.losses import ContactLoss, MultiTargetContactLoss, ddp_global_mean_term

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


def test_dice_reduction_weights_samples_by_confidence_mass():
    torch.manual_seed(1)
    logits = torch.randn(3, 128)
    gt = (torch.rand(3, 128) < 0.2).float()
    mask = torch.ones(3, 128)
    mask[1] = 0.0                                               # sample 1 fully masked
    _, both = _LOSS(logits, gt, mask)
    # Equal mask mass gives the mean over the two supervised samples.
    dice0 = _LOSS(logits[[0]], gt[[0]], torch.ones(1, 128))[1]["dice"]
    dice2 = _LOSS(logits[[2]], gt[[2]], torch.ones(1, 128))[1]["dice"]
    assert both["dice"] == pytest.approx((dice0 + dice2) / 2, abs=1e-5)

    # Uniformly lower confidence for row 2 must reduce that frame's influence.
    mask[2] = 0.1
    _, weighted = _LOSS(logits, gt, mask)
    assert weighted["dice"] == pytest.approx((dice0 + 0.1 * dice2) / 1.1, abs=1e-5)


def test_confidence_scales_per_label_focal_gradient():
    loss_fn = ContactLoss(focal_alpha=0.5, focal_gamma=2.0, focal_weight=1.0,
                          dice_weight=0.0, sparsity_weight=0.0)
    logits = torch.zeros(1, 2, requires_grad=True)
    gt = torch.ones_like(logits)
    total, parts = loss_fn(logits, gt, torch.tensor([[1.0, 0.25]]))
    total.backward()
    ratio = float(logits.grad[0, 0].abs() / logits.grad[0, 1].abs())
    assert ratio == pytest.approx(4.0, rel=1e-5)
    assert parts["weight_mass"] == pytest.approx(1.25)


def test_focal_only_skips_inactive_components_and_matches_focal_gradient():
    loss_fn = ContactLoss(focal_alpha=0.8, focal_gamma=2.0, focal_weight=5.0,
                          dice_weight=0.0, sparsity_weight=0.0)
    logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    gt = torch.tensor([[1.0, 0.0]])
    mask = torch.ones_like(logits)

    total, parts = loss_fn(logits, gt, mask)
    expected = 5.0 * loss_fn._focal_bce(logits, gt, mask)
    expected_grad = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(total, logits)[0]

    assert set(parts) == {
        "focal", "loss", "n_active", "weight_mass", "loss_numerator_tensor"}
    assert total.item() == pytest.approx(expected.item(), rel=1e-6)
    assert torch.allclose(actual_grad, expected_grad)
    # At equal difficulty alpha=.8 makes a false negative four times costlier.
    assert float(actual_grad[0, 0].abs() / actual_grad[0, 1].abs()) == pytest.approx(4.0)


def _rank_focal_terms(parameter, feature, gt, mask):
    logits = parameter * feature
    loss_fn = ContactLoss(focal_alpha=0.8, focal_gamma=2.0, focal_weight=1.0,
                          dice_weight=0.0, sparsity_weight=0.0)
    _, parts = loss_fn(logits, gt, mask)
    return parts["loss_numerator_tensor"], parts["weight_mass"]


@pytest.mark.parametrize("masks", [
    # Unequal ordinary confidence masses.
    (torch.tensor([[1.0, 0.5]]), torch.tensor([[0.25, 1.0]])),
    # Rank 0 has 0 < mass < 1; rank 1 is entirely unsupervised.
    (torch.tensor([[0.1, 0.2]]), torch.zeros(1, 2)),
])
def test_ddp_numerator_reduction_matches_global_weighted_gradient(masks):
    features = (torch.tensor([[1.0, -0.5]]), torch.tensor([[0.25, 2.0]]))
    targets = (torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]))
    global_mass = torch.tensor(sum(float(mask.sum()) for mask in masks))

    local_grads = []
    local_terms = []
    for feature, gt, mask in zip(features, targets, masks):
        parameter = torch.tensor(0.3, requires_grad=True)
        numerator, _ = _rank_focal_terms(parameter, feature, gt, mask)
        term = ddp_global_mean_term(numerator, global_mass, world_size=2)
        local_terms.append(term.detach())
        local_grads.append(torch.autograd.grad(term, parameter)[0])

    # DDP averages rank gradients and scalar terms.
    ddp_grad = torch.stack(local_grads).mean()
    ddp_value = torch.stack(local_terms).mean()

    parameter = torch.tensor(0.3, requires_grad=True)
    global_numerator = sum(
        _rank_focal_terms(parameter, feature, gt, mask)[0]
        for feature, gt, mask in zip(features, targets, masks)
    )
    reference = global_numerator / global_mass.clamp(min=1.0)
    reference_grad = torch.autograd.grad(reference, parameter)[0]

    assert ddp_value.item() == pytest.approx(reference.item(), rel=1e-6)
    assert ddp_grad.item() == pytest.approx(reference_grad.item(), rel=1e-6)


def test_ddp_all_zero_mass_is_zero_and_backward_safe():
    parameter = torch.tensor(0.3, requires_grad=True)
    numerator, mass = _rank_focal_terms(
        parameter,
        torch.tensor([[1.0, -0.5]]),
        torch.tensor([[1.0, 0.0]]),
        torch.zeros(1, 2),
    )
    term = ddp_global_mean_term(numerator, torch.tensor(mass), world_size=2)
    term.backward()

    assert term.item() == 0.0
    assert parameter.grad is not None
    assert parameter.grad.item() == 0.0


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


def test_masked_nan_logits_are_ignored_without_nan_gradient():
    logits = torch.tensor([[0.0, float("nan")]], requires_grad=True)
    gt = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0]])
    loss_fn = ContactLoss(focal_alpha=0.8, focal_gamma=2.0, focal_weight=5.0,
                          dice_weight=0.0, sparsity_weight=0.0)
    total, parts = loss_fn(logits, gt, mask)
    assert torch.isfinite(total)
    assert parts["n_active"] == 1.0
    total.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1].item() == 0.0


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
