"""Unit tests of the world-space temporal refiner (CPU, no base model).

* helpers: Gaussian smoothing, finite differences and angular velocity on
  synthetic series;
* identity at initialisation: the refiner returns the per-frame body it was
  given (depth smoothing off, constant betas);
* world-frame independence: a rigid re-definition of the world (extrinsics
  right-multiplied by the inverse transform) leaves EVERY camera-frame and
  body-frame output identical, and moves the world outputs rigidly;
* gradients reach the contact tokens and the pose token.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import roma
import torch
import yaml

from model.refiner import TemporalRefiner, angular_velocity, gaussian_smooth, time_derivative
from utils.geometry import smplx_q

REPO = Path(__file__).resolve().parents[1]
DECODER_DIM = 1024


@pytest.fixture(scope="module")
def body():
    import better_human as bh
    cfg = yaml.safe_load((REPO / "configs" / "base.yaml").read_text())
    return bh.SMPLX(model_path=cfg["model"]["smplx"]["model_path"], gender="neutral",
                    num_betas=10, use_hands=True, use_face=False, compute_mass=False,
                    dtype=torch.float32, device="cpu")


def synthetic(body, n_clips: int = 2, seq_len: int = 12, seed: int = 0):
    """Random per-frame camera-frame bodies + geometry, as the refiner sees them."""
    torch.manual_seed(seed)
    n = n_clips * seq_len
    root_rot = roma.random_rotmat(n)
    body_rot = roma.rotvec_to_rotmat(0.3 * torch.randn(n, 21, 3))
    hand_rot = roma.rotvec_to_rotmat(0.2 * torch.randn(n, 30, 3))
    betas = (0.5 * torch.randn(n_clips, 10)).repeat_interleave(seq_len, dim=0)
    pelvis_cam = torch.tensor([0.1, 0.2, 3.0]) + 0.05 * torch.randn(n, 3)
    q = smplx_q(pelvis_cam, root_rot, body_rot, hand_rot)
    joints_cam = body.with_shape(betas=betas).fk(q).joint_pose_world[..., 1:, :3]
    smplx_out = {"pelvis_cam": pelvis_cam, "root_rot": root_rot, "body_rot": body_rot,
                 "hand_rot": hand_rot, "betas": betas, "joints_cam": joints_cam}
    tokens = torch.randn(n, 7, DECODER_DIM)
    blocks = {"contact": (1, 7), "pose": (0, 1)}
    ext = torch.eye(4).repeat(n, 1, 1)
    ext[:, :3, :3] = roma.rotvec_to_rotmat(0.2 * torch.randn(n, 3))
    ext[:, :3, 3] = torch.randn(n, 3)
    cam_int = torch.tensor([[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]]).repeat(n, 1, 1)
    affine = torch.tensor([[0.5, 0.0, 10.0], [0.0, 0.5, 20.0]]).repeat(n, 1, 1)
    batch = {
        "seq_len": seq_len,
        "frame_pos_sec": (torch.arange(seq_len, dtype=torch.float32) / 25.0).repeat(n_clips),
        "frame_valid": torch.ones(n, dtype=torch.bool),
        "cam_from_world": ext, "cam_int": cam_int, "affine_trans": affine,
        "img_size": torch.full((n, 2), 256.0),
    }
    return smplx_out, tokens, blocks, batch


def make_refiner(randomize: bool, depth_smooth_sec: float = 0.0) -> TemporalRefiner:
    torch.manual_seed(1)
    refiner = TemporalRefiner(DECODER_DIM, ("pose", "contact", "motion", "force"),
                              num_contact_tokens=6, dim=64, num_layers=2, num_heads=4,
                              window=0.5, depth_smooth_sec=depth_smooth_sec, dropout=0.0)
    if randomize:
        for head in refiner.heads.values():
            torch.nn.init.normal_(head[2].weight, std=0.02)
        for block in refiner.temporal.blocks:
            torch.nn.init.normal_(block.proj.weight, std=0.02)
            torch.nn.init.normal_(block.ffn[3].weight, std=0.02)
    return refiner.eval()


# ------------------------------------------------------------------ helpers

def test_gaussian_smooth_preserves_constants_and_ignores_invalid():
    seconds = torch.arange(10, dtype=torch.float32)[None] / 25.0
    valid = torch.ones(1, 10, dtype=torch.bool)
    x = torch.full((1, 10, 3), 2.5)
    assert torch.allclose(gaussian_smooth(x, seconds, valid, 0.1), x)
    x = torch.zeros(1, 10, 1)
    x[0, 5] = 100.0
    valid[0, 5] = False
    smoothed = gaussian_smooth(x, seconds, valid, 0.1)
    assert torch.allclose(smoothed[0, :5], torch.zeros(5, 1)) and torch.allclose(
        smoothed[0, 6:], torch.zeros(4, 1))          # the invalid spike never leaks
    assert smoothed[0, 5, 0] > 0                     # but the invalid frame keeps seeing itself


def test_time_derivative_of_linear_series_is_the_slope():
    seconds = torch.arange(8, dtype=torch.float32)[None] * 0.04
    valid = torch.ones(1, 8, dtype=torch.bool)
    x = 3.0 * seconds[..., None] + 1.0
    assert torch.allclose(time_derivative(x, seconds, valid), torch.full((1, 8, 1), 3.0), atol=1e-4)
    valid[0, 4] = False                               # a hole: neighbours fall back to one-sided
    d = time_derivative(x, seconds, valid)
    assert torch.allclose(d[0, [3, 5]], torch.full((2, 1), 3.0), atol=1e-4)


def test_angular_velocity_of_constant_rate_rotation():
    rate = torch.tensor([0.0, 0.0, 2.0])             # rad/s about the body z axis
    seconds = torch.arange(6, dtype=torch.float32)[None] * 0.04
    base = roma.random_rotmat(1)
    rot = base @ roma.rotvec_to_rotmat(rate * seconds[0, :, None])    # [T, 3, 3]
    omega = angular_velocity(rot[None], seconds, torch.ones(1, 6, dtype=torch.bool))
    assert torch.allclose(omega[0], rate.expand(6, 3), atol=1e-4)


# ------------------------------------------------------------------ the module

def test_identity_at_init(body):
    smplx_out, tokens, blocks, batch = synthetic(body)
    out = make_refiner(randomize=False)(smplx_out, tokens, blocks, batch, body)
    assert torch.allclose(out["smplx"]["joints_cam"], smplx_out["joints_cam"], atol=1e-4)
    assert torch.allclose(out["smplx"]["pelvis_cam"], smplx_out["pelvis_cam"], atol=1e-5)
    assert torch.allclose(out["smplx"]["root_rot"], smplx_out["root_rot"], atol=1e-5)
    assert torch.allclose(out["smplx"]["body_rot"], smplx_out["body_rot"], atol=1e-5)
    assert torch.allclose(out["smplx"]["betas"], smplx_out["betas"], atol=1e-6)
    assert torch.count_nonzero(out["contact"]["logits"]) == 0
    assert torch.count_nonzero(out["force"]["forces"]) == 0
    assert all(torch.count_nonzero(out["motion"][k]) == 0 for k in ("vel", "acc", "ang_vel", "ang_acc"))


def test_world_frame_independence(body):
    smplx_out, tokens, blocks, batch = synthetic(body)
    refiner = make_refiner(randomize=True, depth_smooth_sec=0.2)
    out = refiner(smplx_out, tokens, blocks, batch, body)
    assert torch.count_nonzero(out["contact"]["logits"]) > 0      # the heads are live
    assert (out["smplx"]["joints_cam"] - smplx_out["joints_cam"]).abs().max() > 1e-4

    # Re-define the world: p_new = R0 p_old + t0  =>  cam_from_world_new = cam_from_world @ G^-1.
    torch.manual_seed(7)
    rot0 = roma.random_rotmat(1)[0]
    t0 = torch.tensor([3.0, -2.0, 5.0])
    g_inv = torch.eye(4)
    g_inv[:3, :3] = rot0.T
    g_inv[:3, 3] = -rot0.T @ t0
    moved = dict(batch)
    moved["cam_from_world"] = batch["cam_from_world"] @ g_inv
    out2 = refiner(smplx_out, tokens, blocks, moved, body)

    for key in ("joints_cam", "pelvis_cam", "root_rot", "body_rot", "q_cam", "kp2d_crop", "betas"):
        assert torch.allclose(out["smplx"][key], out2["smplx"][key], atol=1e-4), key
    assert torch.allclose(out["contact"]["logits"], out2["contact"]["logits"], atol=1e-4)
    assert torch.allclose(out["force"]["forces"], out2["force"]["forces"], atol=1e-4)
    for key in ("vel", "acc", "ang_vel", "ang_acc"):
        assert torch.allclose(out["motion"][key], out2["motion"][key], atol=1e-4), key
    # World outputs move rigidly with the frame.
    expected = (rot0 @ out["smplx"]["pelvis_world"].T).T + t0
    assert torch.allclose(out2["smplx"]["pelvis_world"], expected, atol=1e-4)
    assert torch.allclose(out2["smplx"]["root_rot_world"], rot0 @ out["smplx"]["root_rot_world"], atol=1e-4)
    assert torch.allclose(out2["motion"]["frame"], rot0 @ out["motion"]["frame"], atol=1e-4)


def test_depth_smoothing_keeps_the_bearing(body):
    smplx_out, tokens, blocks, batch = synthetic(body)
    smplx_out["pelvis_cam"][:, 2] += 0.3 * torch.randn(smplx_out["pelvis_cam"].shape[0])
    out = make_refiner(randomize=False, depth_smooth_sec=0.3)(smplx_out, tokens, blocks, batch, body)
    ray_in = smplx_out["pelvis_cam"][:, :2] / smplx_out["pelvis_cam"][:, 2:]
    ray_out = out["smplx"]["pelvis_cam"][:, :2] / out["smplx"]["pelvis_cam"][:, 2:]
    assert torch.allclose(ray_in, ray_out, atol=1e-5)
    # Smoothing reduces the frame-to-frame depth roughness.
    seq_len = batch["seq_len"]
    z_in = smplx_out["pelvis_cam"][:, 2].view(-1, seq_len)
    z_out = out["smplx"]["pelvis_cam"][:, 2].view(-1, seq_len)
    assert (z_out[:, 1:] - z_out[:, :-1]).abs().mean() < (z_in[:, 1:] - z_in[:, :-1]).abs().mean()


def test_gradients_reach_the_tokens(body):
    smplx_out, tokens, blocks, batch = synthetic(body)
    tokens = tokens.clone().requires_grad_(True)
    refiner = make_refiner(randomize=True)
    out = refiner(smplx_out, tokens, blocks, batch, body)
    loss = out["smplx"]["joints_cam"].square().sum() + out["contact"]["logits"].square().sum()
    loss.backward()
    grad = tokens.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad[:, 1:].abs().sum() > 0 and grad[:, 0].abs().sum() > 0
    assert all(p.grad is not None for p in refiner.heads["pose"].parameters())


def test_receptive_field_is_local(body):
    """A frame far outside the window x layers horizon cannot influence a frame."""
    smplx_out, tokens, blocks, batch = synthetic(body, n_clips=1, seq_len=60)
    refiner = make_refiner(randomize=True)                     # 2 layers x 0.5 s = 1 s horizon
    out = refiner(smplx_out, tokens, blocks, batch, body)
    tokens2 = tokens.clone()
    tokens2[-1] += 10.0                                        # perturb the LAST frame (t = 2.36 s)
    out2 = refiner(smplx_out, tokens2, blocks, batch, body)
    logits, logits2 = out["contact"]["logits"], out2["contact"]["logits"]
    assert torch.allclose(logits[:20], logits2[:20], atol=1e-5)   # frames < 0.8 s: untouched
    assert (logits[-1] - logits2[-1]).abs().max() > 1e-4
