"""Tests for the supervised motion (velocity/acceleration) loss."""
from __future__ import annotations

import math

import pytest
import torch

from contact.motion_supervision import (
    MotionSupervisedLoss,
    pearson3d_from_stats,
    pearson_from_stats,
    rmse_from_stats,
)

#: Three slots; the pelvis LAST, so under the ``twist`` convention slot 2 carries
#: the Coriolis correction and the two limb slots do not.
JOINT_NAMES = ["left_wrist", "right_wrist", "pelvis"]
K = len(JOINT_NAMES)


def make_cfg(mean=None, std=None, root_convention="twist", **loss_overrides) -> dict:
    loss = {"vel": 1.0, "acc": 1.0, "huber_delta": 1.0, "outlier_acc_ms2": 50.0}
    loss.update(loss_overrides)
    return {"motion_supervision": {
        "target_frame": "all",
        "joint_names": list(JOINT_NAMES),
        "root_convention": root_convention,
        "standardize": {
            "mean": mean if mean is not None else [[[0.0] * 3, [0.0] * 3]] * K,
            "std": std if std is not None else [[[1.0] * 3, [1.0] * 3]] * K,
        },
        "loss": loss,
    }}


def make_batch(n_rows: int, gt: torch.Tensor, valid=None, outlier=None,
               rot=None, omega=None, seq_len: int = 1) -> dict:
    return {
        "motion_gt": gt,
        "motion_valid": torch.ones(n_rows, dtype=torch.bool) if valid is None else valid,
        "motion_outlier": (torch.zeros(n_rows, K, dtype=torch.bool)
                           if outlier is None else outlier),
        "motion_rot": (torch.eye(3).expand(n_rows, 3, 3).contiguous()
                       if rot is None else rot),
        # Body-frame conventions: the LINEAR frame is the root frame.
        "motion_lin_rot": (torch.eye(3).expand(n_rows, 3, 3).contiguous()
                           if rot is None else rot),
        # World y down — the pre-regeneration constant these expectations assume.
        "gravity_world": torch.tensor([0.0, 1.0, 0.0]).expand(n_rows, 3).contiguous(),
        # Zero by default: the Coriolis term vanishes and every slot then behaves
        # like `rotated_world`. `test_twist_slot_gets_the_coriolis_term` turns it on.
        "motion_omega": torch.zeros(n_rows, 3) if omega is None else omega,
        "frame_valid": torch.ones(n_rows, dtype=torch.bool),
        "seq_len": seq_len,
    }


def out_for(pred: torch.Tensor) -> dict:
    return {"motion": {"joint_motion": pred}}


def test_exact_values_unit_standardizer():
    """Hand-computed Huber with an identity scaler: one small, one large error."""
    loss_fn = MotionSupervisedLoss(make_cfg(), device="cpu")
    pred = torch.zeros(1, K, 6)
    pred[0, 0, 0] = 0.5                              # vel error 0.5 (quadratic branch)
    pred[0, 1, 3] = 3.0                              # acc error 3.0 (linear branch)
    total, parts = loss_fn(out_for(pred), make_batch(1, torch.zeros(1, K, 6)))
    # smooth_l1 beta=1: 0.5*0.5^2 = 0.125; 3.0 - 0.5 = 2.5. Mass = 1 row x 3 joints.
    assert parts["terms"]["vel"]["weight_mass"] == 3.0
    assert math.isclose(parts["terms"]["vel"]["loss"], 0.125 / 3.0, rel_tol=1e-6)
    assert math.isclose(parts["terms"]["acc"]["loss"], 2.5 / 3.0, rel_tol=1e-6)
    assert math.isclose(float(total), (0.125 + 2.5) / 3.0, rel_tol=1e-6)


def test_standardization_round_trip():
    """A prediction equal to the standardized GT gives exactly zero loss, and the
    de-standardized diagnostics see zero error."""
    mean = [[[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]] * K
    std = [[[2.0, 4.0, 0.5], [10.0, 5.0, 2.0]]] * K
    loss_fn = MotionSupervisedLoss(make_cfg(mean=mean, std=std), device="cpu")
    gt = torch.randn(4, K, 6) * 3.0
    mean_t = torch.tensor(mean).reshape(1, K, 6)
    std_t = torch.tensor(std).reshape(1, K, 6)
    pred = (gt - mean_t) / std_t
    total, parts = loss_fn(out_for(pred), make_batch(4, gt))
    assert float(total) == pytest.approx(0.0, abs=1e-6)
    assert parts["vel_rmse"] == pytest.approx(0.0, abs=1e-4)
    assert parts["acc_rmse"] == pytest.approx(0.0, abs=1e-4)


def test_nonunit_scaler_changes_the_objective():
    """The loss operates on standardized values: a 10x std shrinks the residual."""
    gt = torch.zeros(1, K, 6)
    pred = torch.zeros(1, K, 6)
    pred[0, 0, 3] = 1.0                              # 1 standardized unit of acc error
    _, tight = MotionSupervisedLoss(make_cfg(), device="cpu")(
        out_for(pred), make_batch(1, gt))
    # The prediction is already standardized, so the scaler only moves the TARGET.
    std = [[[1.0] * 3, [10.0] * 3]] * K
    _, loose = MotionSupervisedLoss(make_cfg(std=std), device="cpu")(
        out_for(pred), make_batch(1, gt))
    assert tight["terms"]["acc"]["loss"] == pytest.approx(loose["terms"]["acc"]["loss"])
    # ... but the de-standardized RMSE grows with the scale.
    assert loose["acc_rmse"] == pytest.approx(10.0 * tight["acc_rmse"], rel=1e-5)


def test_invalid_frames_masked():
    loss_fn = MotionSupervisedLoss(make_cfg(), device="cpu")
    gt = torch.ones(2, K, 6)
    valid = torch.tensor([True, False])
    _, parts = loss_fn(out_for(torch.zeros(2, K, 6)), make_batch(2, gt, valid=valid))
    assert parts["terms"]["vel"]["weight_mass"] == float(K)
    assert parts["n_supervised_rows"] == 1


def test_outlier_bit_is_per_joint_and_train_only():
    """The bit removes one (frame, joint) entry from BOTH terms — and only in train."""
    loss_fn = MotionSupervisedLoss(make_cfg(), device="cpu")
    gt = torch.zeros(2, K, 6)
    outlier = torch.zeros(2, K, dtype=torch.bool)
    outlier[0, 1] = True
    batch = make_batch(2, gt, outlier=outlier)
    _, train_parts = loss_fn(out_for(torch.zeros(2, K, 6)), batch, exclude_outliers=True)
    assert train_parts["terms"]["vel"]["weight_mass"] == 2 * K - 1
    assert train_parts["terms"]["acc"]["weight_mass"] == 2 * K - 1
    assert train_parts["n_outlier_excluded"] == 1
    _, eval_parts = loss_fn(out_for(torch.zeros(2, K, 6)), batch, exclude_outliers=False)
    assert eval_parts["terms"]["vel"]["weight_mass"] == 2 * K
    assert eval_parts["n_outlier_excluded"] == 0


def test_center_frame_selection():
    """T=3 clips supervise only the middle row; off-center rows are ignored."""
    cfg = make_cfg()
    cfg["motion_supervision"]["target_frame"] = "center"
    loss_fn = MotionSupervisedLoss(cfg, device="cpu")
    pred = torch.zeros(3, K, 6)
    pred[0, 0, 0] = 99.0                             # off-center garbage: ignored
    total, parts = loss_fn(
        out_for(pred), make_batch(3, torch.zeros(3, K, 6), seq_len=3))
    assert parts["terms"]["vel"]["weight_mass"] == float(K)
    assert float(total) == pytest.approx(0.0, abs=1e-6)

    with pytest.raises(ValueError, match="odd seq_len"):
        loss_fn(out_for(pred[:2]), make_batch(2, torch.zeros(2, K, 6), seq_len=2))


def test_empty_supervision_keeps_graph_and_terms():
    """No valid rows: mass-0 terms, zero total, gradient still reaches pred."""
    loss_fn = MotionSupervisedLoss(make_cfg(), device="cpu")
    pred = torch.randn(2, K, 6, requires_grad=True)
    batch = make_batch(2, torch.zeros(2, K, 6), valid=torch.zeros(2, dtype=torch.bool))
    total, parts = loss_fn(out_for(pred), batch)
    assert float(total.detach()) == 0.0
    assert set(parts["terms"]) == {"vel", "acc"}
    assert all(t["weight_mass"] == 0.0 for t in parts["terms"].values())
    total.backward()
    assert pred.grad is not None and torch.all(pred.grad == 0)


def test_zero_weight_term_omitted():
    loss_fn = MotionSupervisedLoss(make_cfg(acc=0.0), device="cpu")
    _, parts = loss_fn(out_for(torch.zeros(1, K, 6)), make_batch(1, torch.zeros(1, K, 6)))
    assert set(parts["terms"]) == {"vel"}


def test_shape_mismatch_raises():
    loss_fn = MotionSupervisedLoss(make_cfg(), device="cpu")
    batch = make_batch(1, torch.zeros(1, K, 6))
    with pytest.raises(ValueError, match="does not match GT"):
        loss_fn(out_for(torch.zeros(1, K + 1, 6)), batch)


def test_standardizer_joint_count_mismatch_raises():
    loss_fn = MotionSupervisedLoss(
        make_cfg(mean=[[[0.0] * 3, [0.0] * 3]] * 2, std=[[[1.0] * 3, [1.0] * 3]] * 2),
        device="cpu")
    gt = torch.zeros(1, K, 6)
    with pytest.raises(ValueError, match="standardize has 2 joint rows"):
        loss_fn(out_for(torch.zeros(1, K, 6)), make_batch(1, gt))


def test_vertical_stats_use_the_world_from_root_rotation():
    """``motion_rot`` row 1 selects the world-vertical component of the root vector."""
    loss_fn = MotionSupervisedLoss(make_cfg(), device="cpu")
    # R maps root -> world by swapping x and y, so world-y is the root-x component.
    rot = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pred = torch.zeros(1, K, 6)
    pred[0, 0, 0] = 2.0                              # vel root-x
    gt = torch.zeros(1, K, 6)
    gt[0, 0, 0] = 3.0
    _, parts = loss_fn(
        out_for(pred), make_batch(1, gt, rot=rot.expand(1, 3, 3).contiguous()))
    stats = parts["stats"]                           # [2, K, 12]
    assert stats[0, 0, 1] == pytest.approx(2.0)      # sum pred vertical
    assert stats[0, 0, 2] == pytest.approx(3.0)      # sum gt vertical
    assert stats[0, 0, 6] == pytest.approx(1.0)      # sum squared 3-D error
    assert stats[0, 0, 7] == pytest.approx(2.0)      # sum pred, pooled over xyz
    assert stats[0, 0, 8] == pytest.approx(3.0)      # sum gt, pooled over xyz


def test_pearson_and_rmse_from_stats_recover_known_values():
    """Perfectly correlated / anti-correlated streams give r = +1 / -1."""
    loss_fn = MotionSupervisedLoss(make_cfg(), device="cpu")
    rows = 32
    gt = torch.zeros(rows, K, 6)
    gt[:, 0, 1] = torch.linspace(-1.0, 1.0, rows)    # vel root-y
    pred = torch.zeros(rows, K, 6)
    pred[:, 0, 1] = 2.0 * gt[:, 0, 1]                # same direction, 2x amplitude
    pred[:, 1, 1] = -gt[:, 0, 1]                     # joint 1: anti-correlated with its (zero) gt
    _, parts = loss_fn(out_for(pred), make_batch(rows, gt))
    r = pearson_from_stats(parts["stats"])
    rmse = rmse_from_stats(parts["stats"])
    assert float(r[0, 0]) == pytest.approx(1.0, abs=1e-6)
    assert math.isnan(float(r[0, 1]))                # constant-zero gt -> undefined
    assert float(rmse[0, 0]) == pytest.approx(
        float((gt[:, 0, 1] ** 2).mean().sqrt()), rel=1e-5)
    # The pooled 3-component r sees the same single moving axis (the other two
    # are constant zero on both sides and contribute no covariance).
    r3d = pearson3d_from_stats(parts["stats"])
    assert float(r3d[0, 0]) == pytest.approx(1.0, abs=1e-6)
    assert math.isnan(float(r3d[0, 1]))


def test_pooled_3d_r_sees_axes_the_vertical_one_misses():
    """``r3d`` correlates all three target components, not just world-vertical."""
    loss_fn = MotionSupervisedLoss(make_cfg(), device="cpu")
    rows = 64
    ramp = torch.linspace(-1.0, 1.0, rows)
    gt = torch.zeros(rows, K, 6)
    gt[:, 0, 3] = ramp                               # acc root-x only
    pred = torch.zeros(rows, K, 6)
    pred[:, 0, 3] = ramp
    # Identity rotation: world-vertical is root-y, which carries no signal at all.
    _, parts = loss_fn(out_for(pred), make_batch(rows, gt))
    assert math.isnan(float(pearson_from_stats(parts["stats"])[1, 0]))
    assert float(pearson3d_from_stats(parts["stats"])[1, 0]) == pytest.approx(
        1.0, abs=1e-6)


def test_twist_slot_gets_the_coriolis_term():
    """Only the pelvis slot picks up ``omega x v`` on the way to world axes.

    With ``omega = e_z`` and a unit root-x velocity, the correction adds
    ``omega x v = +e_y`` to the acceleration — exactly the world-vertical axis
    under an identity rotation. The limb slots must be untouched.
    """
    rows = 1
    gt = torch.zeros(rows, K, 6)
    gt[:, :, 0] = 1.0                                # vel root-x on every slot
    omega = torch.tensor([[0.0, 0.0, 1.0]])
    batch = make_batch(rows, gt, omega=omega)
    pred = torch.zeros(rows, K, 6)                   # unit standardizer -> pred == gt units

    _, twist = MotionSupervisedLoss(make_cfg(), device="cpu")(out_for(pred), batch)
    _, rotated = MotionSupervisedLoss(
        make_cfg(root_convention="rotated_world"), device="cpu")(out_for(pred), batch)
    # GT world-vertical acceleration: +1 on the pelvis slot, 0 on the limbs.
    assert float(twist["stats"][1, 2, 2]) == pytest.approx(1.0)
    assert float(twist["stats"][1, 0, 2]) == pytest.approx(0.0)
    assert float(rotated["stats"][1, 2, 2]) == pytest.approx(0.0)
    # The objective itself is frame-free, so the loss is identical either way.
    assert twist["loss"] == pytest.approx(rotated["loss"])


# ------------------------------------------------------------------ angular (12-dim)


def make_angular_cfg(**loss_overrides) -> dict:
    """Pelvis-only twist config with the angular pair enabled (K=1, G=4)."""
    cfg = make_cfg(**loss_overrides)
    ms = cfg["motion_supervision"]
    ms["joint_names"] = ["pelvis"]
    ms["angular"] = True
    ms["standardize"] = {"mean": [[[0.0] * 3] * 4], "std": [[[1.0] * 3] * 4]}
    ms["loss"].setdefault("ang_vel", 1.0)
    ms["loss"].setdefault("ang_acc", 1.0)
    return cfg


def make_angular_batch(n_rows: int, gt: torch.Tensor, omega=None) -> dict:
    return {
        "motion_gt": gt,
        "motion_valid": torch.ones(n_rows, dtype=torch.bool),
        "motion_outlier": torch.zeros(n_rows, 1, dtype=torch.bool),
        "motion_rot": torch.eye(3).expand(n_rows, 3, 3).contiguous(),
        "motion_lin_rot": torch.eye(3).expand(n_rows, 3, 3).contiguous(),
        "gravity_world": torch.tensor([0.0, 1.0, 0.0]).expand(n_rows, 3).contiguous(),
        "motion_omega": torch.zeros(n_rows, 3) if omega is None else omega,
        "frame_valid": torch.ones(n_rows, dtype=torch.bool),
        "seq_len": 1,
    }


def test_angular_terms_and_exact_values():
    """Four terms in twist order, [4, 1, 12] stats, hand-computed Huber."""
    loss_fn = MotionSupervisedLoss(make_angular_cfg(), device="cpu")
    assert loss_fn.term_names == ("vel", "acc", "ang_vel", "ang_acc")
    pred = torch.zeros(2, 1, 12)
    pred[0, 0, 6] = 0.5                              # ang_vel error (quadratic branch)
    pred[1, 0, 11] = 3.0                             # ang_acc error (linear branch)
    total, parts = loss_fn(
        out_for(pred), make_angular_batch(2, torch.zeros(2, 1, 12)))
    assert tuple(parts["terms"]) == ("vel", "acc", "ang_vel", "ang_acc")
    assert parts["stats"].shape == (4, 1, 12)
    # smooth_l1 beta=1: 0.5*0.5^2 = 0.125; 3.0 - 0.5 = 2.5. Mass = 2 rows x 1 slot.
    assert math.isclose(parts["terms"]["ang_vel"]["loss"], 0.125 / 2.0, rel_tol=1e-6)
    assert math.isclose(parts["terms"]["ang_acc"]["loss"], 2.5 / 2.0, rel_tol=1e-6)
    assert math.isclose(float(total), (0.125 + 2.5) / 2.0, rel_tol=1e-6)
    for name in ("vel", "acc", "ang_vel", "ang_acc"):
        assert f"{name}_rmse" in parts


def test_angular_world_stats_have_no_coriolis():
    """ω/α convert with a plain rotation while the linear acc keeps its Coriolis.

    Identity rotation, ``omega = e_z``, linear vel ``e_x``: the LINEAR world
    acceleration picks up ``ω × v = e_y`` (vertical +1). ``ang_vel = 2 e_x``
    would leak ``ω × ang_vel = 2 e_y`` into a wrongly-Coriolis'd angular
    acceleration; the correct plain rotation leaves its vertical at exactly 3.
    """
    loss_fn = MotionSupervisedLoss(make_angular_cfg(), device="cpu")
    gt = torch.zeros(1, 1, 12)
    gt[0, 0, 0] = 1.0                                # lin vel root-x
    gt[0, 0, 6] = 2.0                                # ang vel root-x
    gt[0, 0, 10] = 3.0                               # ang acc root-y (vertical)
    omega = torch.tensor([[0.0, 0.0, 1.0]])
    _, parts = loss_fn(
        out_for(torch.zeros(1, 1, 12)), make_angular_batch(1, gt, omega=omega))
    stats = parts["stats"]                           # [4, 1, 12]
    assert float(stats[1, 0, 2]) == pytest.approx(1.0)   # lin acc vert: Coriolis
    assert float(stats[2, 0, 2]) == pytest.approx(0.0)   # ang vel vert: plain R
    assert float(stats[3, 0, 2]) == pytest.approx(3.0)   # ang acc vert: plain R
    # Pooled 3-component sums see the raw target axes either way.
    assert float(stats[2, 0, 8]) == pytest.approx(2.0)
    assert float(stats[3, 0, 8]) == pytest.approx(3.0)


def test_angular_width_mismatch_raises():
    """A 6-wide prediction against the 4-term loss is a config drift, not a crash."""
    loss_fn = MotionSupervisedLoss(make_angular_cfg(), device="cpu")
    with pytest.raises(ValueError, match="angular"):
        loss_fn(out_for(torch.zeros(1, 1, 6)),
                make_angular_batch(1, torch.zeros(1, 1, 6)))
