"""Unit tests for the temporal (world-frame vel/acc) keypoint terms.

Synthetic clips with analytically known world trajectories; the static 2D/3D
terms are weighted zero so only the stencil terms are exercised. All float32
(the library's end-to-end dtype).
"""
from __future__ import annotations

import math

import pytest
import torch

from contact.keypoint_supervision import KP_MHR70_INDICES, KeypointSupervisedLoss


def _loss(**overrides) -> KeypointSupervisedLoss:
    loss_cfg = {
        "kp2d": 0.0, "kp3d": 0.0, "kp3d_abs": 0.0,
        "kp_vel": 1.0, "kp_acc": 1.0,
        "huber_delta_2d": 0.05, "huber_delta_3d": 0.1,
        "huber_delta_vel": 0.5, "huber_delta_acc": 2.0,
        "outlier_acc": 50.0,
    }
    loss_cfg.update(overrides)
    return KeypointSupervisedLoss(
        {"keypoint_supervision": {"loss": loss_cfg}}, device="cpu")


def _extrinsics(t: int) -> torch.Tensor:
    """A genuinely moving camera: rotation about y + drifting translation."""
    a = 0.1 * t
    rot = torch.tensor([
        [math.cos(a), 0.0, math.sin(a)],
        [0.0, 1.0, 0.0],
        [-math.sin(a), 0.0, math.cos(a)]])
    ext = torch.eye(4)
    ext[:3, :3] = rot
    ext[:3, 3] = torch.tensor([0.02 * t, -0.01 * t, 3.0])
    return ext


def _batch_and_out(n_clips: int = 2, seq_len: int = 5, fps: float = 30.0):
    """GT: per-joint constant world velocity. Prediction == GT exactly, split
    into ``kp3d + cam_t`` through per-frame MOVING extrinsics."""
    g = torch.Generator().manual_seed(0)
    n = n_clips * seq_len
    x0 = torch.randn(n_clips, 1, 13, 3, generator=g)
    v = 0.3 * torch.randn(n_clips, 1, 13, 3, generator=g)
    t_idx = torch.arange(seq_len, dtype=torch.float32)[None, :, None, None]
    gt_world = (x0 + v * t_idx / fps).reshape(n, 13, 3)

    ext = torch.stack([_extrinsics(t) for _ in range(n_clips)
                       for t in range(seq_len)])
    gt_cam = (torch.einsum("bij,bkj->bki", ext[:, :3, :3], gt_world)
              + ext[:, :3, 3][:, None])
    cam_t = gt_cam.mean(dim=1)
    kp70 = torch.zeros(n, 70, 3)
    kp70[:, list(KP_MHR70_INDICES)] = gt_cam - cam_t[:, None]

    out = {"mhr": {
        "pred_keypoints_3d": kp70,
        "pred_cam_t": cam_t,
        "pred_keypoints_2d_cropped": torch.zeros(n, 70, 2),
    }}
    batch = {
        "kp3d_world": gt_world,
        "cam_from_world": ext,
        "kp_valid": torch.ones(n, dtype=torch.bool),
        "cam_valid": torch.ones(n, dtype=torch.bool),
        "frame_valid": torch.ones(n, dtype=torch.bool),
        "frame_pos_sec": (torch.arange(n, dtype=torch.float32) % seq_len) / fps,
        "seq_len": seq_len,
    }
    return out, batch


def test_exact_prediction_under_moving_camera_is_zero():
    """Pred == GT in world => zero vel/acc loss even though the camera moves —
    the frame-choice property the world lift buys."""
    out, batch = _batch_and_out()
    total, parts = _loss()(out, batch)
    assert parts["terms"]["kp_vel"]["loss"] == pytest.approx(0.0, abs=1e-5)
    assert parts["terms"]["kp_acc"]["loss"] == pytest.approx(0.0, abs=1e-5)
    assert parts["terms"]["kp_vel"]["weight_mass"] == 2 * (5 - 2)
    assert parts["kp_vel_err_ms"] < 1e-4
    assert parts["kp_acc_err_ms2"] < 1e-2


def test_camera_frame_jitter_is_penalized_as_acceleration():
    """A depth blip on one interior frame (invisible-ish statically) shows up
    as a large world acceleration error."""
    out, batch = _batch_and_out()
    out["mhr"]["pred_cam_t"] = out["mhr"]["pred_cam_t"].clone()
    out["mhr"]["pred_cam_t"][2, 2] += 0.03            # 3 cm depth blip, frame 2
    _, parts = _loss()(out, batch)
    assert parts["terms"]["kp_acc"]["loss"] > 1.0
    assert parts["kp_acc_err_ms2"] > 10.0             # 0.03 * 2 / dt^2 / sqrt-ish


def test_stencil_needs_three_valid_rows():
    """Invalidating one frame drops every stencil row that reads it."""
    out, batch = _batch_and_out(n_clips=1)
    _, parts = _loss()(out, batch)
    assert parts["terms"]["kp_vel"]["weight_mass"] == 3
    batch["kp_valid"] = batch["kp_valid"].clone()
    batch["kp_valid"][2] = False                      # centre frame of T=5
    _, parts = _loss()(out, batch)
    # rows 1, 2, 3 all read frame 2 -> zero rows left
    assert parts["terms"]["kp_vel"]["weight_mass"] == 0


def test_gt_outlier_rows_are_dropped():
    """A GT teleport (broken kindyn frame) removes the affected rows."""
    out, batch = _batch_and_out(n_clips=1)
    batch["kp3d_world"] = batch["kp3d_world"].clone()
    batch["kp3d_world"][2, 0] += 5.0                  # one joint jumps 5 m
    _, parts = _loss()(out, batch)
    assert parts["terms"]["kp_acc"]["weight_mass"] == 0


def test_single_frame_batches_fall_back_to_zero_mass():
    """T=1 (still images): terms exist for DDP but carry no mass."""
    out, batch = _batch_and_out(n_clips=1, seq_len=5)
    batch["seq_len"] = 1
    total, parts = _loss()(out, batch)
    assert parts["terms"]["kp_vel"]["weight_mass"] == 0
    assert parts["terms"]["kp_acc"]["weight_mass"] == 0
    assert torch.isfinite(total)


def test_gradients_reach_the_prediction():
    """The vel/acc terms differentiate through kp3d and cam_t."""
    out, batch = _batch_and_out()
    kp = out["mhr"]["pred_keypoints_3d"].clone().requires_grad_(True)
    cam_t = out["mhr"]["pred_cam_t"].clone().requires_grad_(True)
    out["mhr"]["pred_keypoints_3d"] = kp
    out["mhr"]["pred_cam_t"] = cam_t
    total, _ = _loss()(out, batch)
    total.backward()
    assert kp.grad is not None and cam_t.grad is not None
    # An exact match sits at the loss minimum: gradient ~ 0. Perturbed, the
    # depth-blip gradient must be substantially nonzero.
    out["mhr"]["pred_cam_t"] = cam_t.detach().clone().requires_grad_(True)
    with torch.no_grad():
        out["mhr"]["pred_cam_t"][2, 2] += 0.03
    total, _ = _loss()(out, batch)
    total.backward()
    assert float(out["mhr"]["pred_cam_t"].grad.abs().max()) > 1.0
