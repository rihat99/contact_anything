"""Force→motion consistency loss: Newton balance, ramp, masking, gradient routing.

Synthetic clips with analytically known world root trajectories and hand-built
six-group forces, so every expected loss value is a closed-form Huber of a
known residual. Identity extrinsics keep the world root equal to the predicted
camera translation (the world lift itself is covered by
``tests/test_motion_consistency.py``). All float32 (the library's end-to-end
dtype); only the loss's internal trajectory math is float64.
"""
from __future__ import annotations

import math

import pytest
import torch

from contact.force_consistency import GRAVITY_MS2, ForceConsistencyLoss

_FPS = 20.0
_N_GROUPS = 6
#: kindyn world gravity direction: world y, down-positive.
_GRAVITY_DIR = torch.tensor([0.0, 1.0, 0.0])


def _loss(smoothing_kernel=(1.0,), weight_residual: float = 1.0,
          huber_delta_bw: float = 0.5) -> ForceConsistencyLoss:
    return ForceConsistencyLoss(
        {"force_consistency": {
            "enabled": True,
            "ramp": {"start_epoch": 0, "epochs": 0},
            "smoothing_kernel": list(smoothing_kernel),
            "loss": {"residual": weight_residual,
                     "huber_delta_bw": huber_delta_bw}}},
        device="cpu")


def _rotation(angle: float) -> torch.Tensor:
    """World-from-root rotation about world x (a non-trivial GT frame)."""
    return torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(angle), -math.sin(angle)],
        [0.0, math.sin(angle), math.cos(angle)]])


def _out_and_batch(positions: torch.Tensor, force_world_total: torch.Tensor,
                   angle: float = 0.4):
    """Fake forward output + batch for a known root trajectory and net force.

    :param positions: ``(n_clips, T, 3)`` world root positions (metres); with
        identity extrinsics and zero hip keypoints the predicted world root IS
        ``pred_cam_t``.
    :param force_world_total: ``(3,)`` net contact force in the WORLD frame
        (body weights); it is expressed in the root frame and parked on group 0
        so the loss's ``R @ Σ f`` must recover it.
    :returns: ``(out, batch)`` — prediction tensors require grad, and so do the
        GT tensors that must NOT receive one.
    """
    n_clips, seq_len = positions.shape[:2]
    n_frames = n_clips * seq_len
    rot = _rotation(angle)
    forces = torch.zeros(n_frames, _N_GROUPS, 3)
    forces[:, 0] = rot.T @ force_world_total                     # world -> root

    out = {
        "mhr": {
            "pred_keypoints_3d": torch.zeros(n_frames, 70, 3, requires_grad=True),
            "pred_cam_t": positions.reshape(n_frames, 3).clone().requires_grad_(True),
            "global_rot": torch.zeros(n_frames, 3, requires_grad=True),
        },
        "force": {"joint_forces": forces.requires_grad_(True)},
    }
    ones = torch.ones(n_frames, dtype=torch.bool)
    batch = {
        "seq_len": seq_len,
        "cam_from_world": torch.eye(4).expand(n_frames, 4, 4).contiguous(),
        "frame_pos_sec": (torch.arange(n_frames, dtype=torch.float32) % seq_len)
        / _FPS,
        "frame_valid": ones.clone(),
        "cam_valid": ones.clone(),
        "motion_root_valid": ones.clone(),
        "force_valid": ones.clone(),
        "motion_rot": rot.expand(n_frames, 3, 3).contiguous().requires_grad_(True),
        "gravity_world": _GRAVITY_DIR.expand(n_frames, 3).contiguous(),
    }
    return out, batch


def _static(n_clips: int = 2, seq_len: int = 5) -> torch.Tensor:
    return torch.tensor([0.3, -1.2, 4.0]).expand(n_clips, seq_len, 3).contiguous()


def test_supported_hang_is_free():
    """Static root, net contact force exactly one body weight UP => zero residual."""
    out, batch = _out_and_batch(_static(), -_GRAVITY_DIR)
    total, parts = _loss()(out, batch)
    assert parts["terms"].keys() == {"residual"}
    assert parts["residual_bw"] == pytest.approx(0.0, abs=1e-5)
    assert parts["terms"]["residual"]["loss"] == pytest.approx(0.0, abs=1e-6)
    assert parts["terms"]["residual"]["weight_mass"] == 6.0    # 2 clips x 3 rows
    assert parts["ramp"] == 1.0
    assert float(total.detach()) == pytest.approx(0.0, abs=1e-6)


def test_zero_forces_leave_a_full_body_weight_of_gravity():
    """No contact force under a static root => ‖r‖ = 1 bw, a known Huber."""
    out, batch = _out_and_batch(_static(), torch.zeros(3))
    _, parts = _loss(huber_delta_bw=0.5)(out, batch)
    assert parts["residual_bw"] == pytest.approx(1.0, abs=1e-5)
    # r = (0, -1, 0): |r_y| = 1 > beta -> smooth-L1 = 1 - beta/2, other axes 0.
    assert parts["terms"]["residual"]["loss"] == pytest.approx(0.75, abs=1e-5)


def test_free_fall_is_free():
    """No contact force but a genuine 1 g fall along world +y => zero residual."""
    seconds = torch.arange(5, dtype=torch.float32) / _FPS
    fall = 0.5 * GRAVITY_MS2 * seconds.square()
    positions = torch.zeros(1, 5, 3)
    positions[0, :, 1] = fall                       # world y is down-positive
    out, batch = _out_and_batch(positions, torch.zeros(3))
    _, parts = _loss()(out, batch)
    assert parts["residual_bw"] == pytest.approx(0.0, abs=1e-4)


def test_ramp_scales_the_numerator_but_not_the_mass():
    out, batch = _out_and_batch(_static(), torch.zeros(3))
    total, parts = _loss()(out, batch, weight_scale=0.0)
    assert float(total.detach()) == 0.0
    assert parts["terms"]["residual"]["loss"] == 0.0
    assert parts["terms"]["residual"]["weight_mass"] == 6.0     # mass is untouched
    assert parts["ramp"] == 0.0
    assert parts["residual_bw"] == pytest.approx(1.0, abs=1e-5)  # diagnostic unramped
    _, half = _loss()(out, batch, weight_scale=0.5)
    assert half["terms"]["residual"]["loss"] == pytest.approx(0.375, abs=1e-5)
    assert half["ramp"] == 0.5


def test_gradients_reach_the_force_and_pose_paths_only():
    out, batch = _out_and_batch(_static(), torch.zeros(3))
    total, _ = _loss()(out, batch)
    total.backward()
    for tensor, name in ((out["force"]["joint_forces"], "joint_forces"),
                         (out["mhr"]["pred_cam_t"], "pred_cam_t"),
                         (out["mhr"]["pred_keypoints_3d"], "pred_keypoints_3d")):
        assert tensor.grad is not None and float(tensor.grad.abs().sum()) > 0, name
    # The GT rotation is detached, and the orientation half of the pose readout
    # is not part of this (purely linear) balance.
    assert batch["motion_rot"].grad is None
    assert out["mhr"]["global_rot"].grad is None


def test_invalid_root_rows_drop_from_support():
    out, batch = _out_and_batch(_static(n_clips=1), torch.zeros(3))
    batch["motion_root_valid"][0] = False        # only row 1 reads frame 0
    _, parts = _loss()(out, batch)
    assert parts["terms"]["residual"]["weight_mass"] == 2.0
    out, batch = _out_and_batch(_static(n_clips=1), torch.zeros(3))
    batch["motion_root_valid"][2] = False        # centre frame -> every row dies
    total, parts = _loss()(out, batch)
    assert parts["terms"]["residual"]["weight_mass"] == 0.0
    assert parts["residual_bw"] == pytest.approx(0.0, abs=1e-9)
    assert float(total.detach()) == 0.0
    out, batch = _out_and_batch(_static(n_clips=1), torch.zeros(3))
    batch["force_valid"][2] = False
    _, parts = _loss()(out, batch)
    assert parts["terms"]["residual"]["weight_mass"] == 0.0


def _wobbly(n_clips: int = 1, seq_len: int = 5) -> torch.Tensor:
    positions = _static(n_clips, seq_len).clone()
    positions[:, 2, 1] += 0.02                   # 2 cm blip on the centre frame
    positions[:, 3, 0] -= 0.01
    return positions


def _expected_unsmoothed_loss(positions: torch.Tensor, beta: float) -> float:
    """Closed-form loss for identity extrinsics, zero forces, no smoothing."""
    dt = 1.0 / _FPS
    acc = (positions[:, 2:] - 2.0 * positions[:, 1:-1] + positions[:, :-2]) / dt ** 2
    residual = acc / GRAVITY_MS2 - _GRAVITY_DIR
    row = torch.nn.functional.smooth_l1_loss(
        residual, torch.zeros_like(residual), reduction="none", beta=beta).sum(-1)
    return float(row.mean())


def test_unit_kernel_is_exactly_no_smoothing():
    positions = _wobbly()
    out, batch = _out_and_batch(positions, torch.zeros(3))
    _, parts = _loss(smoothing_kernel=(1.0,))(out, batch)
    expected = _expected_unsmoothed_loss(positions, beta=0.5)
    assert parts["terms"]["residual"]["loss"] == pytest.approx(expected, rel=1e-5)
    # A width-3 kernel with all its mass in the centre is the same operator.
    out, batch = _out_and_batch(positions, torch.zeros(3))
    _, centred = _loss(smoothing_kernel=(0.0, 1.0, 0.0))(out, batch)
    assert centred["terms"]["residual"]["loss"] == pytest.approx(expected, rel=1e-5)
    # ... while a real kernel damps the blip's acceleration -> a different value.
    out, batch = _out_and_batch(positions, torch.zeros(3))
    _, smoothed = _loss(smoothing_kernel=(0.25, 0.5, 0.25))(out, batch)
    assert abs(smoothed["terms"]["residual"]["loss"] - expected) > 1e-3


def test_short_clips_are_inactive():
    """T < 3 (still images): the term exists for DDP but carries no mass."""
    out, batch = _out_and_batch(_static(n_clips=2), torch.zeros(3))
    batch["seq_len"] = 1
    total, parts = _loss()(out, batch, weight_scale=0.3)
    assert parts["terms"].keys() == {"residual"}
    assert parts["terms"]["residual"]["weight_mass"] == 0.0
    assert parts["residual_bw"] == 0.0 and parts["ramp"] == 0.3
    assert torch.isfinite(total) and float(total.detach()) == 0.0
    total.backward()                     # zero-touch keeps every path alive
    assert out["force"]["joint_forces"].grad is not None
    assert out["mhr"]["pred_cam_t"].grad is not None


def test_ragged_batches_are_inactive():
    """B not divisible by seq_len (never emitted by the collate, but the DDP
    contract must hold anyway)."""
    out, batch = _out_and_batch(_static(n_clips=2), torch.zeros(3))
    batch["seq_len"] = 4
    _, parts = _loss()(out, batch)
    assert parts["terms"]["residual"]["weight_mass"] == 0.0
