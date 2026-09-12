"""Unit tests of the refined-body pose diagnostics (CPU, no base model).

Two synthetic clips of a known world body ``J = p + R X`` (a smooth pelvis
path, a fixed-axis root rotation of a smoothly varying angle, smoothly swinging
root-frame joints), scored by :func:`model.loss.smplx.world_diag_stats`:

* a prediction that IS the GT reproduces ``gt_jitter`` in all three swaps, has
  every velocity ratio at 1 and no root error at all;
* white noise on the predicted world pelvis raises ``jitter_root_only`` alone
  and lands in the high-pass half of the root error;
* a trajectory replayed at half amplitude gives every velocity ratio 0.5.
"""
from __future__ import annotations

import math

import pytest
import roma
import torch
from torch import Tensor

from model.loss.smplx import (DIAG_METRICS, LIFTED_METRICS, NUM_BODY_JOINTS, POSE_METRICS,
                              lifted_metric_stats, pose_metrics_from_stats, world_diag_stats)

N_CLIPS, SEQ_LEN, FPS = 2, 24, 25.0
NUM_FRAMES = N_CLIPS * SEQ_LEN


def ground_truth(seed: int = 0) -> dict[str, Tensor]:
    """A smooth world body of :data:`N_CLIPS` clips: ``pelvis``, ``rot``, ``joints_root``.

    The root rotation turns about ONE axis, so scaling its angle scales every
    frame-to-frame increment exactly — what the velocity-ratio test needs.
    """
    torch.manual_seed(seed)
    seconds = (torch.arange(SEQ_LEN, dtype=torch.float32) / FPS).repeat(N_CLIPS)
    t = seconds.reshape(N_CLIPS, SEQ_LEN)
    phase = torch.tensor([0.0, 1.3])[:, None]
    pelvis = torch.stack([0.6 * torch.sin(3.0 * t + phase),
                          0.2 * torch.cos(5.0 * t + phase),
                          1.5 + 0.4 * t], dim=-1).reshape(NUM_FRAMES, 3)
    axis = torch.tensor([0.3, 0.9, -0.2])
    angle = 0.5 * torch.sin(4.0 * t + phase)
    rot = roma.random_rotmat(N_CLIPS).repeat_interleave(SEQ_LEN, dim=0) @ roma.rotvec_to_rotmat(
        angle.reshape(NUM_FRAMES, 1) * axis / axis.norm())
    rest = 0.4 * torch.randn(NUM_BODY_JOINTS, 3)
    swing = torch.sin(6.0 * t + phase)[..., None, None] * (0.15 * torch.randn(NUM_BODY_JOINTS, 3))
    joints_root = rest + swing.reshape(NUM_FRAMES, NUM_BODY_JOINTS, 3)
    joints_root[:, 0] = 0.0                      # the pelvis IS the root frame's origin
    return {"pelvis": pelvis, "rot": rot, "joints_root": joints_root, "seconds": seconds}


def world_joints(pelvis: Tensor, rot: Tensor, joints_root: Tensor) -> Tensor:
    """``p + R X`` — the world joints of a body given by its three channels."""
    return pelvis[:, None] + torch.einsum("bij,bkj->bki", rot, joints_root)


def make_batch(gt: dict[str, Tensor]) -> dict:
    """The batch keys the diagnostics read, with identity extrinsics.

    World == camera, so :func:`lifted_metric_stats` on the GT world joints
    yields the ``gt_jitter`` the swap decomposition must reproduce.
    """
    return {
        "seq_len": SEQ_LEN,
        "frame_pos_sec": gt["seconds"],
        "smplx_joints_world": world_joints(gt["pelvis"], gt["rot"], gt["joints_root"]),
        "smplx_root_rot": gt["rot"],
        "cam_from_world": torch.eye(4).repeat(NUM_FRAMES, 1, 1),
    }


def report(stats: Tensor) -> dict[str, float]:
    """The reported diagnostics, through the real statistics-to-metrics path."""
    prefix = torch.zeros(2 * (len(POSE_METRICS) + len(LIFTED_METRICS)), dtype=torch.float64)
    return pose_metrics_from_stats(torch.cat([prefix, stats]), hands=False, world=True)


def diagnose(pelvis: Tensor, rot: Tensor, joints_root: Tensor,
             gt: dict[str, Tensor]) -> dict[str, float]:
    """Score one predicted body (given by its three channels) against the GT."""
    world = {"pelvis_world": pelvis, "root_rot_world": rot,
             "joints_world": world_joints(pelvis, rot, joints_root)}
    valid = torch.ones(NUM_FRAMES, dtype=torch.bool)
    stats = world_diag_stats(world, make_batch(gt), valid, SEQ_LEN)
    assert stats.shape == (2 * len(DIAG_METRICS),)
    return report(stats)


def gt_jitter(gt: dict[str, Tensor]) -> float:
    """``gt_jitter`` of the same clips, straight out of :func:`lifted_metric_stats`."""
    batch = make_batch(gt)
    valid = torch.ones(NUM_FRAMES, dtype=torch.bool)
    stats = lifted_metric_stats(batch["smplx_joints_world"], batch, valid, SEQ_LEN)
    return float(stats[-2]) / float(stats[-1])


def test_perfect_prediction_is_the_gt_floor():
    gt = ground_truth()
    metrics = diagnose(gt["pelvis"], gt["rot"], gt["joints_root"], gt)
    floor = gt_jitter(gt)
    assert floor > 0.0
    for name in ("jitter_root_only", "jitter_rot_only", "jitter_artic_only"):
        assert metrics[name] == pytest.approx(floor, rel=1e-3, abs=5e-3), name
    for name in ("vel_ratio_root", "vel_ratio_rot", "vel_ratio_joints"):
        assert metrics[name] == pytest.approx(1.0, rel=1e-5), name
    assert metrics["root_err_hf"] == 0.0
    assert metrics["root_err_lf"] == 0.0


def test_pelvis_noise_moves_only_the_root_jitter():
    gt = ground_truth()
    torch.manual_seed(3)
    sigma = 0.01                                                     # 10 mm, white
    metrics = diagnose(gt["pelvis"] + sigma * torch.randn(NUM_FRAMES, 3), gt["rot"],
                       gt["joints_root"], gt)
    floor = gt_jitter(gt)
    assert metrics["jitter_root_only"] > 20.0 * floor
    for name in ("jitter_rot_only", "jitter_artic_only"):
        assert metrics[name] == pytest.approx(floor, rel=1e-3, abs=5e-3), name
    # White noise is (almost) all high pass, and inflates the root velocity alone.
    assert metrics["root_err_hf"] > 4.0 * metrics["root_err_lf"]
    assert metrics["root_err_hf"] == pytest.approx(1000.0 * sigma * math.sqrt(3), rel=0.25)
    assert metrics["vel_ratio_root"] > 1.02
    for name in ("vel_ratio_rot", "vel_ratio_joints"):
        assert metrics[name] == pytest.approx(1.0, rel=1e-5), name


def test_halved_velocities_halve_every_ratio():
    gt = ground_truth()
    # Replay each channel from its own first frame at half the amplitude: every forward
    # difference is exactly halved (the rotation turns about a single fixed axis).
    start = gt["pelvis"].reshape(N_CLIPS, SEQ_LEN, 3)[:, :1]
    pelvis = (start + 0.5 * (gt["pelvis"].reshape(N_CLIPS, SEQ_LEN, 3) - start)
              ).reshape(NUM_FRAMES, 3)
    rot = gt["rot"].reshape(N_CLIPS, SEQ_LEN, 3, 3)
    increment = rot[:, :1].transpose(-1, -2) @ rot
    rot = (rot[:, :1] @ roma.rotvec_to_rotmat(0.5 * roma.rotmat_to_rotvec(increment))
           ).reshape(NUM_FRAMES, 3, 3)
    joints = gt["joints_root"].reshape(N_CLIPS, SEQ_LEN, NUM_BODY_JOINTS, 3)
    joints = (joints[:, :1] + 0.5 * (joints - joints[:, :1])).reshape(
        NUM_FRAMES, NUM_BODY_JOINTS, 3)
    metrics = diagnose(pelvis, rot, joints, gt)
    for name in ("vel_ratio_root", "vel_ratio_rot", "vel_ratio_joints"):
        assert metrics[name] == pytest.approx(0.5, rel=1e-4), name
