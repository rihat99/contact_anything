"""Contact→velocity consistency loss: egomotion cancellation, gating, masking, grads.

Synthetic clips with analytically known WORLD trajectories for the six
extremities, split into ``pred_keypoints_3d + pred_cam_t`` through per-frame
MOVING extrinsics — so a passing "zero loss" test is proof the world lift
cancels camera egomotion, not that the camera stood still. All float32 (the
library's end-to-end dtype).
"""
from __future__ import annotations

import math

import pytest
import torch

from contact.contact_consistency import (
    EXTREMITY_MHR70_INDICES, ContactConsistencyLoss,
)

_FPS = 30.0
_N_LIMBS = len(EXTREMITY_MHR70_INDICES)


def _loss(detach_gate: bool = True, **overrides) -> ContactConsistencyLoss:
    loss_cfg = {"vel": 1.0, "huber_delta_ms": 0.05}
    loss_cfg.update(overrides)
    return ContactConsistencyLoss(
        {"contact_consistency": {
            "enabled": True, "detach_gate": detach_gate, "loss": loss_cfg}},
        device="cpu")


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


def _world(n_clips: int, seq_len: int, velocity: torch.Tensor) -> torch.Tensor:
    """``(n_clips, T, 6, 3)`` world limb positions with per-limb constant velocity.

    :param velocity: ``(6, 3)`` metres per second.
    """
    gen = torch.Generator().manual_seed(0)
    start = torch.randn(n_clips, 1, _N_LIMBS, 3, generator=gen)
    seconds = torch.arange(seq_len, dtype=torch.float32)[None, :, None, None] / _FPS
    return start + velocity[None, None] * seconds


def _out_and_batch(world: torch.Tensor, gate: torch.Tensor):
    """Fake forward output whose world-lifted extremities ARE ``world``.

    :param world: ``(n_clips, T, 6, 3)`` world limb positions.
    :param gate: ``(n_clips * T, 6)`` contact probabilities.
    :returns: ``(out, batch)`` — every prediction tensor requires grad.
    """
    n_clips, seq_len = world.shape[:2]
    n_frames = n_clips * seq_len
    flat = world.reshape(n_frames, _N_LIMBS, 3)
    ext = torch.stack([_extrinsics(t) for _ in range(n_clips)
                       for t in range(seq_len)])
    cam = (torch.einsum("bij,bkj->bki", ext[:, :3, :3], flat)
           + ext[:, :3, 3][:, None])
    cam_t = cam.mean(dim=1)                    # an arbitrary keypoint/camera split
    keypoints = torch.zeros(n_frames, 70, 3)
    keypoints[:, list(EXTREMITY_MHR70_INDICES)] = cam - cam_t[:, None]

    out = {
        "mhr": {"pred_keypoints_3d": keypoints.requires_grad_(True),
                "pred_cam_t": cam_t.requires_grad_(True)},
        "contact": {"joint_probs": gate.clone().requires_grad_(True),
                    "joint_logits": torch.zeros(
                        n_frames, _N_LIMBS, requires_grad=True)},
    }
    batch = {
        "seq_len": seq_len,
        "cam_from_world": ext,
        "cam_valid": torch.ones(n_frames, dtype=torch.bool),
        "frame_valid": torch.ones(n_frames, dtype=torch.bool),
        "frame_pos_sec": (torch.arange(n_frames, dtype=torch.float32) % seq_len)
        / _FPS,
    }
    return out, batch


def _still(n_clips: int = 2, seq_len: int = 5, gate: float = 1.0):
    world = _world(n_clips, seq_len, torch.zeros(_N_LIMBS, 3))
    return _out_and_batch(
        world, torch.full((n_clips * seq_len, _N_LIMBS), gate))


def _one_limb_moving(n_clips: int = 2, seq_len: int = 5, speed: float = 0.3,
                     gate: float = 1.0):
    """Limb 0 slides along world x at ``speed``; the other five stand still."""
    velocity = torch.zeros(_N_LIMBS, 3)
    velocity[0, 0] = speed
    world = _world(n_clips, seq_len, velocity)
    gates = torch.zeros(n_clips * seq_len, _N_LIMBS)
    gates[:, 0] = gate
    return _out_and_batch(world, gates)


def test_still_limbs_under_a_moving_camera_cost_nothing():
    """World-static limbs => zero speed => zero loss, even though the camera
    translates and rotates every frame (the egomotion cancellation)."""
    out, batch = _still()
    total, parts = _loss()(out, batch)
    assert parts["terms"].keys() == {"vel"}
    assert parts["terms"]["vel"]["loss"] == pytest.approx(0.0, abs=1e-6)
    assert parts["vel_ms"] == pytest.approx(0.0, abs=1e-4)
    # 5-frame clips: rows 1..3 have stencil support -> 3 rows x 2 clips x 6 limbs.
    assert parts["terms"]["vel"]["weight_mass"] == pytest.approx(36.0)
    assert parts["n_rows"] == 6
    assert float(total.detach()) == pytest.approx(0.0, abs=1e-6)


def test_moving_limb_is_penalized_where_contact_is_predicted():
    """Gate 1 on the moving limb: the loss is the Huber of its 0.3 m/s speed."""
    out, batch = _one_limb_moving(speed=0.3, gate=1.0)
    _, parts = _loss()(out, batch)
    # |v| = 0.3 > beta = 0.05 -> smooth-L1 = 0.3 - beta/2.
    assert parts["terms"]["vel"]["loss"] == pytest.approx(0.275, abs=1e-4)
    assert parts["terms"]["vel"]["weight_mass"] == pytest.approx(6.0)
    assert parts["vel_ms"] == pytest.approx(0.3, abs=1e-4)
    assert parts["gate_mean"] == pytest.approx(1.0 / _N_LIMBS, abs=1e-6)


def test_zero_gate_removes_both_loss_and_mass():
    """The SAME motion costs nothing when the model predicts no contact."""
    out, batch = _one_limb_moving(speed=0.3, gate=0.0)
    total, parts = _loss()(out, batch)
    assert parts["terms"]["vel"]["weight_mass"] == 0.0
    assert parts["terms"]["vel"]["loss"] == pytest.approx(0.0, abs=1e-9)
    assert float(total.detach()) == pytest.approx(0.0, abs=1e-9)


def test_mass_is_the_summed_gate_over_supported_rows():
    out, batch = _still(n_clips=2, seq_len=5, gate=0.25)
    _, parts = _loss()(out, batch)
    # 2 clips x 3 supported rows x 6 limbs x gate 0.25.
    assert parts["terms"]["vel"]["weight_mass"] == pytest.approx(9.0)
    assert parts["gate_mean"] == pytest.approx(0.25, abs=1e-6)


def test_stencil_needs_three_valid_rows():
    """Invalidating a frame drops every stencil row that reads it."""
    out, batch = _still(n_clips=1, seq_len=5, gate=1.0)
    batch["cam_valid"] = batch["cam_valid"].clone()
    batch["cam_valid"][2] = False            # centre frame of T=5 -> rows 1,2,3 die
    _, parts = _loss()(out, batch)
    assert parts["terms"]["vel"]["weight_mass"] == 0.0
    out, batch = _still(n_clips=1, seq_len=5, gate=1.0)
    batch["frame_valid"] = batch["frame_valid"].clone()
    batch["frame_valid"][0] = False          # only row 1 reads frame 0
    _, parts = _loss()(out, batch)
    assert parts["terms"]["vel"]["weight_mass"] == pytest.approx(12.0)


def test_detached_gate_keeps_gradient_off_the_contact_head():
    out, batch = _one_limb_moving()
    total, _ = _loss(detach_gate=True)(out, batch)
    total.backward()
    assert out["contact"]["joint_probs"].grad is None
    # The logits stay on the graph (zero_touch) so DDP sees no unused param.
    logit_grad = out["contact"]["joint_logits"].grad
    assert logit_grad is not None
    assert float(logit_grad.abs().sum()) == 0.0


def test_undetached_gate_reaches_the_contact_head():
    out, batch = _one_limb_moving()
    total, _ = _loss(detach_gate=False)(out, batch)
    total.backward()
    grad = out["contact"]["joint_probs"].grad
    assert grad is not None and float(grad.abs().sum()) > 0
    # Only the moving limb's interior rows carry a gradient.
    assert float(grad[:, 1:].abs().sum()) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("detach_gate", [True, False])
def test_gradients_reach_the_pose_path(detach_gate: bool):
    out, batch = _one_limb_moving()
    total, _ = _loss(detach_gate=detach_gate)(out, batch)
    total.backward()
    for key in ("pred_keypoints_3d", "pred_cam_t"):
        grad = out["mhr"][key].grad
        assert grad is not None and float(grad.abs().sum()) > 0, key


def test_short_clips_are_inactive():
    """T < 3 (still images): the term exists for DDP but carries no mass."""
    out, batch = _still(n_clips=2, seq_len=5)
    batch["seq_len"] = 1
    total, parts = _loss()(out, batch)
    assert parts["terms"].keys() == {"vel"}
    assert parts["terms"]["vel"]["weight_mass"] == 0.0
    assert parts["vel_ms"] == 0.0 and parts["gate_mean"] == 0.0
    assert torch.isfinite(total) and float(total.detach()) == 0.0
    total.backward()                     # zero-touch keeps every path alive
    assert out["mhr"]["pred_cam_t"].grad is not None
    assert out["contact"]["joint_logits"].grad is not None


def test_ragged_batches_are_inactive():
    """B not divisible by seq_len (never emitted by the collate, but the DDP
    contract must hold anyway)."""
    out, batch = _still(n_clips=2, seq_len=5)
    batch["seq_len"] = 4
    _, parts = _loss()(out, batch)
    assert parts["terms"]["vel"]["weight_mass"] == 0.0
