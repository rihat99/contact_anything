"""Tests for the supervised six-group force loss."""
from __future__ import annotations

import math

import pytest
import torch

from contact.force_supervision import ForceSupervisedLoss

K = 6


def make_cfg(**loss_overrides) -> dict:
    loss = {"force": 1.0, "noncontact": 1.0, "sum_force": 0.0, "sum_torque": 0.0,
            "huber_delta_bw": 0.5, "huber_delta_bwm": 0.1, "outlier_bw": 4.0,
            "group_weights": None}
    loss.update(loss_overrides)
    return {"force_supervision": {"target_frame": "all", "loss": loss}}


def make_batch(n_rows: int, gt: torch.Tensor, contact: torch.Tensor,
               valid: torch.Tensor | None = None, seq_len: int = 1,
               lever: torch.Tensor | None = None) -> dict:
    batch = {
        "force_gt": gt,
        "force_contact": contact,
        "force_valid": torch.ones(n_rows, dtype=torch.bool) if valid is None else valid,
        "frame_valid": torch.ones(n_rows, dtype=torch.bool),
        "seq_len": seq_len,
    }
    if lever is not None:
        batch["force_lever"] = lever
    return batch


def out_for(pred: torch.Tensor) -> dict:
    return {"force": {"joint_forces": pred}}


def test_exact_values_single_row():
    """Hand-computed Huber + noncontact L1 on one frame, two active groups."""
    loss_fn = ForceSupervisedLoss(make_cfg(), device="cpu")
    pred = torch.zeros(1, K, 3)
    pred[0, 0] = torch.tensor([0.1, 0.0, 0.0])      # in contact, small error
    pred[0, 2] = torch.tensor([0.0, 2.0, 0.0])      # in contact, large error
    pred[0, 1] = torch.tensor([0.3, 0.0, -0.4])     # NOT in contact -> L1 penalty
    gt = torch.zeros(1, K, 3)
    gt[0, 0] = torch.tensor([0.2, 0.0, 0.0])
    gt[0, 2] = torch.tensor([0.0, 0.5, 0.0])
    contact = torch.zeros(1, K, dtype=torch.bool)
    contact[0, [0, 2]] = True

    total, parts = loss_fn(out_for(pred), make_batch(1, gt, contact))
    # group 0: |e|=0.1 < delta=0.5 -> 0.5*0.1^2/0.5 = 0.01; group 2: |e|=1.5 ->
    # 1.5 - 0.5/2 = 1.25 (smooth_l1 with beta): expected numerator 1.26, mass 2.
    assert parts["terms"]["force"]["weight_mass"] == 2.0
    assert math.isclose(parts["terms"]["force"]["loss"], 1.26 / 2.0, rel_tol=1e-5)
    # noncontact: L1 over remaining 4 groups; only group 1 nonzero: 0.3 + 0.4.
    assert parts["terms"]["noncontact"]["weight_mass"] == 4.0
    assert math.isclose(parts["terms"]["noncontact"]["loss"], 0.7 / 4.0, rel_tol=1e-5)
    assert math.isclose(float(total), 1.26 / 2.0 + 0.7 / 4.0, rel_tol=1e-5)
    # headline: mean error norm over contact entries = (0.1 + 1.5) / 2.
    assert math.isclose(parts["force_mae"]["loss"], 0.8, rel_tol=1e-5)


def test_outlier_frames_excluded():
    loss_fn = ForceSupervisedLoss(make_cfg(outlier_bw=4.0), device="cpu")
    gt = torch.zeros(1, K, 3)
    gt[0, 0, 1] = -10.0                              # 10 bw solver blowup
    gt[0, 1, 1] = -0.5
    contact = torch.zeros(1, K, dtype=torch.bool)
    contact[0, [0, 1]] = True
    _, parts = loss_fn(out_for(torch.zeros(1, K, 3)), make_batch(1, gt, contact))
    assert parts["n_outlier_excluded"] == 1
    assert parts["terms"]["force"]["weight_mass"] == 1.0  # only group 1 remains
    # An excluded limb-frame is dropped, not moved to the noncontact term.
    assert parts["terms"]["noncontact"]["weight_mass"] == 4.0


def test_center_frame_selection():
    """T=3 clips supervise only the middle row; off-center rows are ignored."""
    cfg = make_cfg()
    cfg["force_supervision"]["target_frame"] = "center"
    loss_fn = ForceSupervisedLoss(cfg, device="cpu")
    pred = torch.zeros(3, K, 3)
    pred[0, 0, 0] = 99.0                             # off-center garbage: ignored
    pred[1, 0, 0] = 1.0                              # center row
    gt = torch.zeros(3, K, 3)
    gt[1, 0, 0] = 1.0
    contact = torch.zeros(3, K, dtype=torch.bool)
    contact[:, 0] = True
    total, parts = loss_fn(out_for(pred), make_batch(3, gt, contact, seq_len=3))
    assert parts["terms"]["force"]["weight_mass"] == 1.0
    assert float(total) == pytest.approx(0.0, abs=1e-6)

    cfg["force_supervision"]["target_frame"] = "center"
    with pytest.raises(ValueError, match="odd seq_len"):
        loss_fn(out_for(pred[:2]), make_batch(2, gt[:2], contact[:2], seq_len=2))


def test_empty_supervision_keeps_graph_and_terms():
    """No valid rows: mass-0 terms, zero total, gradient still reaches pred."""
    loss_fn = ForceSupervisedLoss(make_cfg(), device="cpu")
    pred = torch.randn(2, K, 3, requires_grad=True)
    valid = torch.zeros(2, dtype=torch.bool)
    batch = make_batch(2, torch.zeros(2, K, 3), torch.zeros(2, K, dtype=torch.bool),
                       valid=valid)
    total, parts = loss_fn(out_for(pred), batch)
    assert float(total.detach()) == 0.0
    assert set(parts["terms"]) == {"force", "noncontact"}
    assert all(t["weight_mass"] == 0.0 for t in parts["terms"].values())
    total.backward()
    assert pred.grad is not None and torch.all(pred.grad == 0)


def test_invalid_frames_masked():
    loss_fn = ForceSupervisedLoss(make_cfg(), device="cpu")
    gt = torch.ones(2, K, 3)
    contact = torch.ones(2, K, dtype=torch.bool)
    valid = torch.tensor([True, False])
    _, parts = loss_fn(out_for(torch.zeros(2, K, 3)),
                       make_batch(2, gt, contact, valid=valid))
    assert parts["terms"]["force"]["weight_mass"] == float(K)
    assert parts["n_supervised_rows"] == 1


def test_shape_mismatch_raises():
    loss_fn = ForceSupervisedLoss(make_cfg(), device="cpu")
    batch = make_batch(1, torch.zeros(1, K, 3), torch.zeros(1, K, dtype=torch.bool))
    with pytest.raises(ValueError, match="does not match GT"):
        loss_fn(out_for(torch.zeros(1, 4, 3)), batch)


def test_group_weights_weighted_mean():
    """Weights enter numerator AND mass; the headline MAE stays unweighted."""
    weights = [1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
    loss_fn = ForceSupervisedLoss(make_cfg(group_weights=weights), device="cpu")
    pred = torch.zeros(1, K, 3)
    gt = torch.zeros(1, K, 3)
    gt[0, 0, 0] = 0.1                                # hand group, weight 1
    gt[0, 2, 0] = 0.1                                # foot group, weight 2
    contact = torch.zeros(1, K, dtype=torch.bool)
    contact[0, [0, 2]] = True
    _, parts = loss_fn(out_for(pred), make_batch(1, gt, contact))
    # Same per-entry Huber value h = 0.5*0.1^2/0.5 = 0.01; weighted mean
    # (1*h + 2*h) / (1 + 2) = h — equal errors are unaffected by weighting.
    assert parts["terms"]["force"]["weight_mass"] == 3.0
    assert math.isclose(parts["terms"]["force"]["loss"], 0.01, rel_tol=1e-5)
    # Unequal errors: the foot entry dominates 2:1.
    gt[0, 0, 0] = 0.3                                # h_hand = 0.5*0.09/0.5 = 0.09
    _, parts = loss_fn(out_for(pred), make_batch(1, gt, contact))
    assert math.isclose(
        parts["terms"]["force"]["loss"], (1 * 0.09 + 2 * 0.01) / 3.0, rel_tol=1e-5)
    # Headline MAE is unweighted: (0.3 + 0.1) / 2.
    assert math.isclose(parts["force_mae"]["loss"], 0.2, rel_tol=1e-5)


def test_group_weights_length_mismatch_raises():
    loss_fn = ForceSupervisedLoss(make_cfg(group_weights=[1.0, 2.0]), device="cpu")
    batch = make_batch(1, torch.zeros(1, K, 3), torch.zeros(1, K, dtype=torch.bool))
    with pytest.raises(ValueError, match="group_weights"):
        loss_fn(out_for(torch.zeros(1, K, 3)), batch)


def test_zero_weight_term_omitted():
    loss_fn = ForceSupervisedLoss(make_cfg(noncontact=0.0), device="cpu")
    gt = torch.zeros(1, K, 3)
    contact = torch.zeros(1, K, dtype=torch.bool)
    _, parts = loss_fn(out_for(torch.zeros(1, K, 3)), make_batch(1, gt, contact))
    assert set(parts["terms"]) == {"force"}


def test_sum_force_exact_value():
    """Hand-computed Huber on the six-group net force, all groups counted."""
    loss_fn = ForceSupervisedLoss(make_cfg(sum_force=0.25), device="cpu")
    pred = torch.zeros(1, K, 3)
    gt = torch.zeros(1, K, 3)
    gt[0, 0] = torch.tensor([0.2, 0.0, 0.0])
    gt[0, 2] = torch.tensor([0.0, 0.6, 0.0])
    contact = torch.zeros(1, K, dtype=torch.bool)
    contact[0, [0, 2]] = True
    _, parts = loss_fn(out_for(pred), make_batch(1, gt, contact))
    # net error (0.2, 0.6, 0), beta 0.5: 0.5*0.2^2/0.5 + (0.6 - 0.25) = 0.39;
    # the term's loss carries its 0.25 weight (mass 1 row).
    assert parts["terms"]["sum_force"]["weight_mass"] == 1.0
    assert math.isclose(parts["terms"]["sum_force"]["loss"], 0.25 * 0.39, rel_tol=1e-5)


def test_sum_rows_skip_outliers_and_invalid_rows():
    """A row with ANY outlier group leaves the sum terms entirely; so do invalid rows."""
    loss_fn = ForceSupervisedLoss(make_cfg(sum_force=1.0), device="cpu")
    pred = torch.zeros(3, K, 3)
    gt = torch.zeros(3, K, 3)
    gt[1, 0, 1] = -10.0                              # outlier group poisons row 1's sum
    gt[1, 1, 0] = 0.5                                # would register if the row counted
    contact = torch.zeros(3, K, dtype=torch.bool)
    contact[1, [0, 1]] = True
    valid = torch.tensor([True, True, False])        # row 2 is force-invalid
    _, parts = loss_fn(out_for(pred), make_batch(3, gt, contact, valid=valid))
    assert parts["terms"]["sum_force"]["weight_mass"] == 1.0   # row 0 only
    assert parts["terms"]["sum_force"]["loss"] == pytest.approx(0.0, abs=1e-6)


def test_sum_torque_exact_value_and_shared_levers():
    """tau = sum(r x f) with the loader's levers on BOTH sides of the Huber."""
    cfg = make_cfg(force=0.0, noncontact=0.0, sum_torque=1.0)
    loss_fn = ForceSupervisedLoss(cfg, device="cpu")
    pred = torch.zeros(1, K, 3)
    pred[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    gt = torch.zeros(1, K, 3)
    contact = torch.zeros(1, K, dtype=torch.bool)
    contact[0, 0] = True
    lever = torch.zeros(1, K, 3)
    lever[0, 0] = torch.tensor([0.0, 0.0, 1.0])
    _, parts = loss_fn(
        out_for(pred), make_batch(1, gt, contact, lever=lever))
    # tau_pred = (0,0,1) x (1,0,0) = (0,1,0); tau_gt = 0; beta 0.1 -> 1 - 0.05.
    assert parts["terms"]["sum_torque"]["weight_mass"] == 1.0
    assert math.isclose(parts["terms"]["sum_torque"]["loss"], 0.95, rel_tol=1e-5)
    # Identical pred and gt forces give exactly zero torque residual: only the
    # forces differ between the two sides, never the lever arms.
    _, parts = loss_fn(
        out_for(gt.clone()), make_batch(1, gt, contact, lever=torch.randn(1, K, 3)))
    assert parts["terms"]["sum_torque"]["loss"] == pytest.approx(0.0, abs=1e-6)


def test_sum_torque_skips_nonfinite_lever_rows():
    """A NaN lever removes the row from sum_torque only — and must not leak NaN."""
    loss_fn = ForceSupervisedLoss(
        make_cfg(sum_force=1.0, sum_torque=1.0), device="cpu")
    pred = torch.zeros(2, K, 3, requires_grad=True)
    gt = torch.zeros(2, K, 3)
    contact = torch.zeros(2, K, dtype=torch.bool)
    lever = torch.zeros(2, K, 3)
    lever[1, 3, 2] = float("nan")
    total, parts = loss_fn(out_for(pred), make_batch(2, gt, contact, lever=lever))
    assert parts["terms"]["sum_torque"]["weight_mass"] == 1.0
    assert parts["terms"]["sum_force"]["weight_mass"] == 2.0
    assert math.isfinite(float(total.detach()))
    total.backward()
    assert torch.isfinite(pred.grad).all()


def test_center_selection_applies_to_levers():
    """T=3 clips: only the middle row's lever/forces feed the sum terms."""
    cfg = make_cfg(force=0.0, noncontact=0.0, sum_torque=1.0)
    cfg["force_supervision"]["target_frame"] = "center"
    loss_fn = ForceSupervisedLoss(cfg, device="cpu")
    pred = torch.zeros(3, K, 3)
    gt = torch.zeros(3, K, 3)
    contact = torch.zeros(3, K, dtype=torch.bool)
    lever = torch.full((3, K, 3), float("nan"))
    lever[1] = 0.0                                   # only the center row is finite
    _, parts = loss_fn(
        out_for(pred), make_batch(3, gt, contact, seq_len=3, lever=lever))
    assert parts["terms"]["sum_torque"]["weight_mass"] == 1.0
    assert parts["terms"]["sum_torque"]["loss"] == pytest.approx(0.0, abs=1e-6)
