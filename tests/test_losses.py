"""Unit tests of the contact and force supervision terms (CPU, synthetic batches).

* force: the magnitude / direction split is zero where it should be, finite with
  a gradient along the GT at the zero-initialised head, and rows below
  ``direction_min_bw`` stay out of the direction term; ``confidence_power``
  compresses the row weights;
* contact: ``class_weights`` scale the positive / negative rows of the BCE and
  leave the metrics untouched.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch
import yaml

from model.loss.contact import ContactLoss
from model.loss.force import ForceLoss

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def base_cfg():
    return yaml.safe_load((REPO / "configs" / "base.yaml").read_text())


def force_cfg(base_cfg, **loss):
    cfg = copy.deepcopy(base_cfg)
    cfg["force_supervision"]["enabled"] = True
    cfg["force_supervision"]["loss"].update(loss)
    return cfg


def force_batch(n: int = 8, seed: int = 0):
    torch.manual_seed(seed)
    gt = 0.5 * torch.randn(n, 6, 3)
    return {
        "force_gt": gt, "force_lever": 0.3 * torch.randn(n, 6, 3),
        "force_contact": torch.ones(n, 6, dtype=torch.bool),
        "force_valid": torch.ones(n, dtype=torch.bool), "frame_valid": torch.ones(n, dtype=torch.bool),
        "force_conf": torch.full((n,), 0.25),
    }


def force_terms(loss, pred, batch):
    result = loss({"force": {"forces": pred}}, batch, train=True)
    return {k: float(v.numerator) / v.mass for k, v in result.terms.items() if v.mass > 0}, result


def test_force_split_is_zero_on_scaled_and_equal_predictions(base_cfg):
    loss = ForceLoss(force_cfg(base_cfg, force=0.0, magnitude=1.0, direction=1.0, direction_min_bw=0.0),
                     None, "cpu")
    batch = force_batch()
    gt = batch["force_gt"]
    terms, _ = force_terms(loss, gt * 3.0, batch)          # same direction, wrong magnitude
    assert terms["direction"] < 1e-6 and terms["magnitude"] > 0.1
    unit = gt / gt.norm(dim=-1, keepdim=True)
    rotated = torch.stack([-unit[..., 1], unit[..., 0], unit[..., 2]], dim=-1)   # rotate about z
    rotated = rotated / rotated.norm(dim=-1, keepdim=True) * gt.norm(dim=-1, keepdim=True)
    terms, result = force_terms(loss, rotated, batch)        # same magnitude, wrong direction
    assert terms["magnitude"] < 1e-5 and terms["direction"] > 0.1
    metrics = loss.metrics(result.stats)
    assert metrics["mag_mae"] < 1e-5 and 0.0 < metrics["angle_deg"] <= 180.0


def test_force_direction_is_finite_at_zero_and_pushes_along_gt(base_cfg):
    loss = ForceLoss(force_cfg(base_cfg, force=0.0, magnitude=1.0, direction=1.0), None, "cpu")
    batch = force_batch()
    pred = torch.zeros(8, 6, 3, requires_grad=True)
    terms, result = force_terms(loss, pred, batch)
    assert math.isfinite(terms["direction"]) and abs(terms["direction"] - 1.0) < 1e-6
    sum(v.numerator for v in result.terms.values()).backward()
    assert torch.isfinite(pred.grad).all()
    rows = batch["force_gt"].norm(dim=-1) >= loss.direction_min_bw
    cos = torch.nn.functional.cosine_similarity(-pred.grad[rows], batch["force_gt"][rows], dim=-1)
    assert (cos > 0.999).all()                                # descent = the GT direction


def test_force_direction_min_bw_masks_small_rows(base_cfg):
    batch = force_batch()
    mags = batch["force_gt"].norm(dim=-1)
    cut = float(mags.median())
    loss = ForceLoss(force_cfg(base_cfg, force=0.0, direction=1.0, direction_min_bw=cut), None, "cpu")
    _, result = force_terms(loss, torch.randn(8, 6, 3), batch)
    expected = float((mags >= cut).sum()) * 0.25            # rows x confidence
    assert abs(result.terms["direction"].mass - expected) < 1e-4
    assert abs(float(result.stats[6]) - float((mags >= cut).sum())) < 1e-6   # angle rows unweighted


def test_force_confidence_power_compresses_row_weights(base_cfg):
    batch = force_batch()
    for power, expected in ((1.0, 0.25), (0.5, 0.5), (0.0, 1.0)):
        cfg = force_cfg(base_cfg, force=1.0)
        cfg["force_supervision"]["confidence_power"] = power
        _, result = force_terms(ForceLoss(cfg, None, "cpu"), torch.randn(8, 6, 3), batch)
        assert abs(result.terms["force"].mass - 48 * expected) < 1e-4


def test_contact_class_weights_scale_rows_not_metrics(base_cfg):
    torch.manual_seed(0)
    logits, gt = torch.randn(10, 6), (torch.rand(10, 6) > 0.5).float()
    batch = {"contact_gt": gt, "contact_valid": torch.ones(10, 6), "contact_conf": torch.ones(10, 6)}
    plain = ContactLoss(base_cfg, None, "cpu")({"contact": {"logits": logits}}, batch, train=True)
    cfg = copy.deepcopy(base_cfg)
    cfg["contact_supervision"]["class_weights"] = {"positive": [1, 1, 1, 1, 5, 5], "negative": [2, 2, 1, 1, 1, 1]}
    weighted_loss = ContactLoss(cfg, None, "cpu")
    weighted = weighted_loss({"contact": {"logits": logits}}, batch, train=True)
    pos, neg = weighted_loss.class_weights
    w = gt * pos + (1 - gt) * neg
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, gt, reduction="none")
    assert abs(float(weighted.terms["bce"].numerator) - float((bce * w).sum())) < 1e-4
    assert abs(weighted.terms["bce"].mass - float(w.sum())) < 1e-4
    assert abs(plain.terms["bce"].mass - 60.0) < 1e-6
    assert torch.equal(plain.stats, weighted.stats)          # metrics ignore the class weights
    with pytest.raises(ValueError):
        bad = copy.deepcopy(cfg)
        bad["contact_supervision"]["class_weights"]["positive"] = [1, 1, 1]
        ContactLoss(bad, None, "cpu")
