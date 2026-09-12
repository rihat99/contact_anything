"""Unit tests of the world-space temporal refiner (CPU, no base model).

* helpers: Gaussian smoothing, finite differences and angular velocity on
  synthetic series;
* identity at initialisation: the refiner returns the per-frame body it was
  given (smoothing off, constant betas);
* world-frame independence: a rigid re-definition of the world (extrinsics
  right-multiplied by the inverse transform) leaves EVERY camera-frame and
  body-frame output identical, and moves the world outputs rigidly;
* gradients reach the contact tokens and the pose token;
* iterative refinement: a correction per layer, the per-layer world joints, and
  gradients into the zero-init feedback projection (with and without the
  cumulative-delta feedback channels);
* deep supervision: the motion / SMPL-X ``layer_weight`` terms repeat the body's
  own terms on the intermediate layers, pooled, with mass = rows x layers;
* pose smoothing: the polar projection matches the Procrustes one, a constant
  trajectory is a fixed point, and frame independence survives the camera
  context features;
* the video-interleaved sampler visits every clip once and spreads a step's
  block over distinct videos.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import roma
import torch
import yaml

from data.loaders import VideoInterleavedSampler
from model.loss.motion import MotionLoss
from model.loss.reference import GaussianReferenceLoss
from model.loss.smplx import SmplxLoss
from model.refiner import (TemporalRefiner, angular_velocity, backward_angular_velocity,
                           backward_difference, forward_angular_velocity, forward_difference,
                           forward_valid, gaussian_smooth, project_rotation, smooth_rotations,
                           time_derivative)
from torch import Tensor

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
    # Continuous trajectories (random walks of ~6 deg per frame) rather than independent
    # random rotations: the rotation smoothing assumes adjacent frames are close.
    base = roma.random_rotmat(n_clips).repeat_interleave(seq_len, dim=0)
    walk = torch.cumsum(0.1 * torch.randn(n_clips, seq_len, 3), dim=1).reshape(n, 3)
    root_rot = base @ roma.rotvec_to_rotmat(walk)
    body_walk = torch.cumsum(0.1 * torch.randn(n_clips, seq_len, 21, 3), dim=1).reshape(n, 21, 3)
    body_rot = roma.rotvec_to_rotmat(0.3 * torch.randn(n_clips, 21, 3)).repeat_interleave(
        seq_len, dim=0) @ roma.rotvec_to_rotmat(body_walk)
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
    ext[:, :3, :3] = roma.rotvec_to_rotmat(
        torch.cumsum(0.02 * torch.randn(n_clips, seq_len, 3), dim=1).reshape(n, 3))
    ext[:, :3, 3] = torch.cumsum(0.05 * torch.randn(n_clips, seq_len, 3), dim=1).reshape(n, 3)
    cam_int = torch.tensor([[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]]).repeat(n, 1, 1)
    affine = torch.tensor([[0.5, 0.0, 10.0], [0.0, 0.5, 20.0]]).repeat(n, 1, 1)
    batch = {
        "seq_len": seq_len,
        "frame_pos_sec": (torch.arange(seq_len, dtype=torch.float32) / 25.0).repeat(n_clips),
        "frame_valid": torch.ones(n, dtype=torch.bool),
        "cam_from_world": ext, "cam_int": cam_int, "affine_trans": affine,
        "img_size": torch.full((n, 2), 256.0),
        "bbox_center": torch.tensor([480.0, 520.0]) + 20.0 * torch.randn(n, 2),
        "bbox_scale": torch.full((n, 2), 300.0) + 10.0 * torch.randn(n, 2),
        "gravity_world": torch.tensor([0.0, -1.0, 0.0]).expand(n, 3).clone(),
    }
    return smplx_out, tokens, blocks, batch


ALL_TOKEN = {"local_rotations": True, "gravity": True, "raw_minus_mean": True}
ONE_SIDED_TOKEN = {**ALL_TOKEN, "one_sided_velocity": True, "joint_velocity": True}


ALL_OUTPUTS = ("pose", "contact", "motion", "force", "gravity")


def smplx_model_path() -> str:
    cfg = yaml.safe_load((REPO / "configs" / "base.yaml").read_text())
    return cfg["model"]["smplx"]["model_path"]


def make_refiner(randomize: bool, root_smooth_sec: float = 0.0, pose_smooth_sec: float = 0.0,
                 camera_context: bool = False, learn_smoothing: bool = False,
                 token: dict | None = None, iterative: bool = False,
                 feedback_delta: bool = False, outputs=("pose", "contact", "motion", "force"),
                 camera_axes: bool = False, residual_feedback: bool = False,
                 frame_mask_p: float = 0.0, head_grad_scale: float = 1.0) -> TemporalRefiner:
    torch.manual_seed(1)
    refiner = TemporalRefiner(DECODER_DIM, outputs,
                              num_contact_tokens=6, dim=64, num_layers=2, num_heads=4,
                              window=0.5, root_smooth_sec=root_smooth_sec,
                              pose_smooth_sec=pose_smooth_sec, learn_smoothing=learn_smoothing,
                              token=token, iterative=iterative, feedback_delta=feedback_delta,
                              camera_context=camera_context, camera_axes=camera_axes,
                              residual_feedback=residual_feedback, frame_mask_p=frame_mask_p,
                              head_grad_scale=head_grad_scale,
                              smplx_model_path=smplx_model_path() if residual_feedback else None,
                              dropout=0.0)
    if randomize:
        for head in refiner.heads.values():
            torch.nn.init.normal_(head[2].weight, std=0.02)
        for block in refiner.temporal.blocks:
            torch.nn.init.normal_(block.proj.weight, std=0.02)
            torch.nn.init.normal_(block.ffn[3].weight, std=0.02)
        if refiner.feedback_proj is not None:
            torch.nn.init.normal_(refiner.feedback_proj.weight, std=0.02)
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


def test_forward_difference_of_linear_series_is_the_slope():
    seconds = torch.arange(8, dtype=torch.float32)[None] * 0.04
    valid = torch.ones(1, 8, dtype=torch.bool)
    x = 3.0 * seconds[..., None] + 1.0
    forward = forward_difference(x, seconds, valid)
    assert torch.allclose(forward[0, :-1], torch.full((7, 1), 3.0), atol=1e-4)
    assert torch.count_nonzero(forward[0, -1]) == 0            # the last frame has no successor
    assert forward_valid(valid).tolist() == [[True] * 7 + [False]]
    backward = backward_difference(x, seconds, valid)
    assert torch.allclose(backward[0, 1:], torch.full((7, 1), 3.0), atol=1e-4)
    assert torch.count_nonzero(backward[0, 0]) == 0
    # The point of the stencil: a period-2 alternation is invisible to the central difference.
    alternating = torch.tensor([0.0, 1.0] * 4)[None, :, None]
    assert torch.count_nonzero(time_derivative(alternating, seconds, valid)[0, 1:-1]) == 0
    assert (forward_difference(alternating, seconds, valid)[0, :-1].abs() > 1.0).all()
    valid[0, 4] = False                                        # a hole: both steps across it die
    forward = forward_difference(x, seconds, valid)
    assert torch.count_nonzero(forward[0, 3]) == 0 and torch.count_nonzero(forward[0, 4]) == 0
    assert torch.allclose(forward[0, 5], torch.full((1,), 3.0), atol=1e-4)
    assert forward_valid(valid).tolist() == [[True, True, True, False, False, True, True, False]]


def test_one_sided_angular_velocity_of_constant_rate_rotation():
    rate = torch.tensor([0.0, 0.0, 2.0])
    seconds = torch.arange(6, dtype=torch.float32)[None] * 0.04
    valid = torch.ones(1, 6, dtype=torch.bool)
    rot = (roma.random_rotmat(1) @ roma.rotvec_to_rotmat(rate * seconds[0, :, None]))[None]
    forward = forward_angular_velocity(rot, seconds, valid)
    backward = backward_angular_velocity(rot, seconds, valid)
    assert torch.allclose(forward[0, :-1], rate.expand(5, 3), atol=1e-4)
    assert torch.count_nonzero(forward[0, -1]) == 0
    assert torch.allclose(backward[0, 1:], rate.expand(5, 3), atol=1e-4)
    assert torch.count_nonzero(backward[0, 0]) == 0


def test_angular_velocity_of_constant_rate_rotation():
    rate = torch.tensor([0.0, 0.0, 2.0])             # rad/s about the body z axis
    seconds = torch.arange(6, dtype=torch.float32)[None] * 0.04
    base = roma.random_rotmat(1)
    rot = base @ roma.rotvec_to_rotmat(rate * seconds[0, :, None])    # [T, 3, 3]
    omega = angular_velocity(rot[None], seconds, torch.ones(1, 6, dtype=torch.bool))
    assert torch.allclose(omega[0], rate.expand(6, 3), atol=1e-4)


# ------------------------------------------------------------------ the module

@pytest.mark.parametrize("iterative, token, feedback_delta", [
    (False, None, False), (True, ONE_SIDED_TOKEN, False), (True, ONE_SIDED_TOKEN, True)])
def test_identity_at_init(body, iterative, token, feedback_delta):
    smplx_out, tokens, blocks, batch = synthetic(body)
    refiner = make_refiner(randomize=False, iterative=iterative, token=token,
                           feedback_delta=feedback_delta)
    out = refiner(smplx_out, tokens, blocks, batch, body)
    layers = out["smplx"]["joints_world_layers"]
    assert len(layers) == (refiner.temporal.num_layers if iterative else 1)
    assert layers[-1] is out["smplx"]["joints_world"]
    assert torch.allclose(out["smplx"]["joints_cam"], smplx_out["joints_cam"], atol=1e-4)
    assert torch.allclose(out["smplx"]["pelvis_cam"], smplx_out["pelvis_cam"], atol=1e-5)
    assert torch.allclose(out["smplx"]["root_rot"], smplx_out["root_rot"], atol=1e-5)
    assert torch.allclose(out["smplx"]["body_rot"], smplx_out["body_rot"], atol=1e-5)
    assert torch.allclose(out["smplx"]["betas"], smplx_out["betas"], atol=1e-6)
    assert torch.count_nonzero(out["contact"]["logits"]) == 0
    assert torch.count_nonzero(out["force"]["forces"]) == 0
    assert all(torch.count_nonzero(out["motion"][k]) == 0 for k in ("vel", "acc", "ang_vel", "ang_acc"))


@pytest.mark.parametrize("camera_context, token, iterative, feedback_delta", [
    (False, None, False, False), (True, None, False, False), (True, ALL_TOKEN, False, False),
    (False, ALL_TOKEN, False, False), (True, ONE_SIDED_TOKEN, False, False),
    (False, ONE_SIDED_TOKEN, False, False), (True, ONE_SIDED_TOKEN, True, False),
    (False, ONE_SIDED_TOKEN, True, False), (True, ONE_SIDED_TOKEN, True, True),
    (False, ONE_SIDED_TOKEN, True, True)])
def test_world_frame_independence(body, camera_context, token, iterative, feedback_delta):
    smplx_out, tokens, blocks, batch = synthetic(body)
    refiner = make_refiner(randomize=True, root_smooth_sec=0.2, pose_smooth_sec=0.08,
                           camera_context=camera_context, token=token, iterative=iterative,
                           feedback_delta=feedback_delta)
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
    moved["gravity_world"] = batch["gravity_world"] @ rot0.T      # the down vector moves with the world
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


def test_root_smoothing_is_in_the_world(body):
    # A body at rest in the world seen by a moving camera: its camera-frame pelvis moves with
    # the camera, and the smoothing must leave the WORLD position exactly fixed (smoothing
    # the camera coordinates instead would blur the camera's motion into it).
    smplx_out, tokens, blocks, batch = synthetic(body)
    ext = batch["cam_from_world"]
    p_w = torch.tensor([0.5, -0.2, 1.0])
    at_rest = ext[:, :3, :3] @ p_w + ext[:, :3, 3]
    smplx_out["pelvis_cam"] = at_rest.clone()
    out = make_refiner(randomize=False, root_smooth_sec=0.3)(smplx_out, tokens, blocks, batch, body)
    assert torch.allclose(out["smplx"]["pelvis_world"], p_w.expand_as(out["smplx"]["pelvis_world"]),
                          atol=1e-5)
    # Per-frame noise on the camera-frame pelvis is reduced in the world.
    torch.manual_seed(3)
    smplx_out["pelvis_cam"] = at_rest + 0.05 * torch.randn_like(at_rest)
    out = make_refiner(randomize=False, root_smooth_sec=0.3)(smplx_out, tokens, blocks, batch, body)
    err_in = (out["smplx"]["pelvis_world_in"] - p_w).norm(dim=-1).mean()
    raw_w = torch.einsum("bji,bj->bi", ext[:, :3, :3], smplx_out["pelvis_cam"] - ext[:, :3, 3])
    assert err_in < 0.5 * (raw_w - p_w).norm(dim=-1).mean()


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


def test_iterative_feedback_is_live_and_gets_gradients(body):
    """Every layer corrects the trajectory and the feedback projection is on the loss path."""
    smplx_out, tokens, blocks, batch = synthetic(body)
    refiner = make_refiner(randomize=True, iterative=True, token=ONE_SIDED_TOKEN)
    torch.nn.init.constant_(refiner.heads["pose"][2].bias, 1e-3)   # a non-zero first correction
    out = refiner(smplx_out, tokens, blocks, batch, body)
    layers = out["smplx"]["joints_world_layers"]
    assert len(layers) == refiner.temporal.num_layers == 2
    assert torch.equal(layers[-1], out["smplx"]["joints_world"])
    assert (layers[0] - layers[1]).abs().max() > 1e-5              # the second layer moves it too

    out["smplx"]["joints_world"].sum().backward()
    for name, param in refiner.feedback_proj.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()
        assert param.grad.abs().sum() > 0, name
    with pytest.raises(ValueError):                                # the fed-back channels must be on
        make_refiner(randomize=False, iterative=True, token=ALL_TOKEN)
    with pytest.raises(ValueError):                                # ... and the delta needs them too
        make_refiner(randomize=False, iterative=False, token=ONE_SIDED_TOKEN, feedback_delta=True)


def test_polar_projection_matches_procrustes():
    torch.manual_seed(3)
    base = roma.random_rotmat(500)
    noisy = torch.stack([base @ roma.rotvec_to_rotmat(0.15 * torch.randn(500, 3))
                         for _ in range(5)]).mean(0)
    projected = project_rotation(noisy)
    assert torch.allclose(projected, roma.special_procrustes(noisy), atol=1e-5)
    eye = torch.eye(3).expand(500, 3, 3)
    assert torch.allclose(projected.transpose(-1, -2) @ projected, eye, atol=1e-5)


def test_pose_smoothing_fixes_a_constant_pose(body):
    """Smoothing is a fixed point on a still body; on a jittery one it damps the rotations."""
    smplx_out, tokens, blocks, batch = synthetic(body, n_clips=1, seq_len=20)
    still = {k: (v[:1].expand_as(v).clone() if torch.is_tensor(v) else v) for k, v in smplx_out.items()}
    batch_still = dict(batch)
    batch_still["cam_from_world"] = batch["cam_from_world"][:1].expand_as(batch["cam_from_world"]).clone()
    out = make_refiner(randomize=False, pose_smooth_sec=0.1)(still, tokens, blocks, batch_still, body)
    assert torch.allclose(out["smplx"]["root_rot"], still["root_rot"], atol=1e-5)
    assert torch.allclose(out["smplx"]["body_rot"], still["body_rot"], atol=1e-5)
    assert torch.allclose(out["smplx"]["pelvis_cam"], still["pelvis_cam"], atol=1e-5)
    seconds = batch["frame_pos_sec"][None]
    valid = torch.ones(1, 20, dtype=torch.bool)
    rot = smplx_out["root_rot"][None]
    smooth = smooth_rotations(rot, seconds, valid, 0.1)
    step = lambda r: roma.rotmat_to_rotvec(r[0, :-1].transpose(-1, -2) @ r[0, 1:]).norm(dim=-1).mean()
    assert step(smooth) < step(rot)


def test_video_interleaved_sampler():
    videos = [f"v{i % 7}" for i in range(50)] + ["v_long"] * 30
    world, batch = 2, 4
    samplers = [VideoInterleavedSampler(videos, batch, num_replicas=world, rank=r, seed=1)
                for r in range(world)]
    per_rank = [list(s) for s in samplers]
    assert all(len(idx) == len(samplers[0]) for idx in per_rank)
    seen = sorted(i for idx in per_rank for i in idx)
    assert len(seen) == len(set(seen)) == (len(videos) // (world * batch)) * world * batch
    # The first step's global block (8 clips) comes from 8 distinct videos.
    block = [videos[i] for r in range(world) for i in per_rank[r][:batch]]
    assert len(set(block)) == world * batch
    samplers[0].set_epoch(1)
    assert list(samplers[0]) != per_rank[0]                     # a new epoch, a new deal


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


def test_gaussian_smooth_per_channel_widths_match_scalars():
    torch.manual_seed(3)
    seconds = (torch.arange(10, dtype=torch.float32) / 25.0)[None]
    valid = torch.ones(1, 10, dtype=torch.bool)
    x = torch.randn(1, 10, 3, 4)
    widths = torch.tensor([0.05, 0.1, 0.2])
    per_channel = gaussian_smooth(x, seconds, valid, widths)
    for j, w in enumerate(widths.tolist()):
        assert torch.allclose(per_channel[:, :, j], gaussian_smooth(x[:, :, j], seconds, valid, w), atol=1e-6)
    assert torch.allclose(gaussian_smooth(x, seconds, valid, torch.tensor([0.1])),
                          gaussian_smooth(x, seconds, valid, 0.1), atol=1e-6)


def test_learnable_smoothing_starts_at_the_config_widths_and_gets_gradients(body):
    smplx_out, tokens, blocks, batch = synthetic(body)
    fixed = make_refiner(randomize=False, root_smooth_sec=0.1, pose_smooth_sec=0.08)
    learn = make_refiner(randomize=False, root_smooth_sec=0.1, pose_smooth_sec=0.08, learn_smoothing=True)
    names = {n for n, _ in learn.named_parameters()}
    assert {"log_root_sigma", "log_pose_sigma"} <= names and learn.log_pose_sigma.shape == (22,)
    out_fixed = fixed(smplx_out, tokens, blocks, batch, body)["smplx"]
    out_learn = learn(smplx_out, tokens, blocks, batch, body)["smplx"]
    assert torch.allclose(out_fixed["joints_world"], out_learn["joints_world"], atol=1e-5)
    scalars = learn.smoothing_scalars()
    assert abs(scalars["smoothing/root_sigma"] - 0.1) < 1e-6
    assert abs(scalars["smoothing/joint_sigma_mean"] - 0.08) < 1e-6
    out_learn["joints_world"].pow(2).sum().backward()
    assert learn.log_root_sigma.grad is not None and torch.isfinite(learn.log_root_sigma.grad).all()
    # Leaf joints (feet, head) move no joint position, so their widths see no gradient here.
    grad = learn.log_pose_sigma.grad
    assert grad is not None and torch.isfinite(grad).all() and int((grad != 0).sum()) >= 18
    with pytest.raises(ValueError):
        make_refiner(randomize=False, root_smooth_sec=0.0, pose_smooth_sec=0.08, learn_smoothing=True)



# ------------------------------------------------------------------ the new loss terms

def world_series(n_clips: int = 2, seq_len: int = 16, seed: int = 0):
    """World-space random-walk pose series in the refiner's output layout (no body model)."""
    torch.manual_seed(seed)
    n = n_clips * seq_len
    pelvis = torch.cumsum(0.02 * torch.randn(n_clips, seq_len, 3), dim=1).reshape(n, 3)
    root = roma.rotvec_to_rotmat(
        torch.cumsum(0.05 * torch.randn(n_clips, seq_len, 3), dim=1).reshape(n, 3))
    body = roma.rotvec_to_rotmat(
        torch.cumsum(0.05 * torch.randn(n_clips, seq_len, 21, 3), dim=1).reshape(n, 21, 3))
    batch = {"seq_len": seq_len,
             "frame_pos_sec": (torch.arange(seq_len, dtype=torch.float32) / 25.0).repeat(n_clips),
             "frame_valid": torch.ones(n, dtype=torch.bool)}
    return pelvis, root, body, batch


def base_cfg() -> dict:
    return yaml.safe_load((REPO / "configs" / "base.yaml").read_text())


def test_gaussian_reference_is_zero_on_its_own_reference_and_anneals():
    cfg = base_cfg()
    cfg["gaussian_reference"]["enabled"] = True
    loss = GaussianReferenceLoss(cfg, None, "cpu")
    pelvis, root, body, batch = world_series()
    inputs = {"pelvis_world_in": pelvis, "root_rot_world_in": root, "body_rot_in": body}
    p_ref, r_ref, b_ref = loss.reference(inputs, batch)

    on_reference = loss({"smplx": {**inputs, "pelvis_world": p_ref, "root_rot_world": r_ref,
                                   "body_rot": b_ref}}, batch, train=True)
    assert all(float(t.numerator) / t.mass < 1e-5 for t in on_reference.terms.values())
    metrics = loss.metrics(on_reference.stats)
    assert max(metrics["root_mm"], metrics["rot_deg"], metrics["joints_deg"]) < 1e-3
    assert metrics["weight"] == 1.0

    # The RAW body is at a distance, and the schedule — not the distance — scales the terms.
    raw = {"smplx": {**inputs, "pelvis_world": pelvis, "root_rot_world": root, "body_rot": body}}
    results = {}
    for progress in (0.0, 0.45, 1.0):
        loss.progress = progress
        results[progress] = loss(raw, batch, train=True)
    assert loss.current_weight() == 0.0
    for term in loss.term_names:
        full = float(results[0.0].terms[term].numerator)
        assert full > 0.0
        assert math.isclose(float(results[0.45].terms[term].numerator), 0.5 * full, rel_tol=1e-5)
        assert float(results[1.0].terms[term].numerator) == 0.0
    assert loss.metrics(results[1.0].stats)["weight"] == 0.0
    assert math.isclose(loss.metrics(results[1.0].stats)["root_mm"],
                        loss.metrics(results[0.0].stats)["root_mm"], rel_tol=1e-9)
    assert loss.metrics(results[0.0].stats)["root_mm"] > 1.0      # mm away from the Gaussian


def test_motion_loss_runs_without_a_motion_head():
    cfg = base_cfg()
    cfg["motion_supervision"].update(enabled=True, stencil="forward", label_smooth_sec=0.0)
    cfg["motion_supervision"]["loss"].update(vel=0.0, acc=0.0, ang_vel=0.0, ang_acc=0.0,
                                             pose_vel=1.0, pose_acc=0.2, pose_ang_vel=1.0,
                                             pose_ang_acc=0.5)
    loss = MotionLoss(cfg, None, "cpu")
    assert loss.term_names == ("pose_vel", "pose_acc", "pose_ang_vel", "pose_ang_acc")

    _, root, _, batch = world_series(seed=2)
    n = root.shape[0]
    joints = (0.3 * torch.randn(n, 22, 3)).requires_grad_(True)
    batch = {**batch, "smplx_valid": torch.ones(n, dtype=torch.bool),
             "smplx_joints_world": 0.3 * torch.randn(n, 22, 3),
             "smplx_root_rot": roma.random_rotmat(n)}
    result = loss({"motion": None, "smplx": {"joints_world": joints, "root_rot_world": root}},
                  batch, train=True)
    assert set(result.terms) == set(loss.term_names)
    assert all(t.mass > 0 and torch.isfinite(t.numerator) for t in result.terms.values())
    sum(t.numerator for t in result.terms.values()).backward()
    assert joints.grad is not None and torch.isfinite(joints.grad).all() and joints.grad.abs().sum() > 0
    assert set(loss.metrics(result.stats)) == {
        f"{src}_{q}_{s}" for src in ("pose", "pose_s", "pose_ss") for q in ("vel", "acc", "ang_vel", "ang_acc")
        for s in ("rmse", "pearson", "amp_ratio")}


def test_motion_rms_terms_match_amplitude_not_value():
    cfg = base_cfg()
    cfg["motion_supervision"].update(enabled=True, stencil="forward", label_smooth_sec=0.0)
    cfg["motion_supervision"]["loss"].update(vel=0.0, acc=0.0, ang_vel=0.0, ang_acc=0.0,
                                             pose_vel=0.0, pose_acc=0.0, pose_ang_vel=0.0,
                                             pose_ang_acc=0.0, pose_acc_rms=1.0, pose_ang_acc_rms=1.0)
    loss = MotionLoss(cfg, None, "cpu")
    assert loss.term_names == ("pose_acc_rms", "pose_ang_acc_rms")
    _, root, _, batch = world_series(seed=3)
    n = root.shape[0]
    gt = 0.3 * torch.randn(n, 22, 3)
    batch = {**batch, "smplx_valid": torch.ones(n, dtype=torch.bool),
             "smplx_joints_world": gt, "smplx_root_rot": root.clone()}
    # The same trajectory reversed in time within each clip has the same acceleration
    # amplitude but different per-frame values: the RMS terms must be ~0 on it.
    seq_len = int(batch["seq_len"])
    flipped = gt.view(-1, seq_len, 22, 3).flip(1).reshape(n, 22, 3)
    root_flipped = root.view(-1, seq_len, 3, 3).flip(1).reshape(n, 3, 3)
    same = loss({"motion": None, "smplx": {"joints_world": gt.clone(), "root_rot_world": root.clone()}},
                batch, train=True)
    flip = loss({"motion": None, "smplx": {"joints_world": flipped, "root_rot_world": root_flipped}},
                batch, train=True)
    for term in loss.term_names:
        assert same.terms[term].mass > 0
        assert float(same.terms[term].numerator) < 1e-6
        assert float(flip.terms[term].numerator) / flip.terms[term].mass < 1e-3
    # Halving the trajectory halves its acceleration amplitude: a clearly positive term with
    # a gradient that pushes the amplitude UP (towards the GT), never down.
    half = (0.5 * gt).requires_grad_(True)
    res = loss({"motion": None, "smplx": {"joints_world": half, "root_rot_world": root.clone()}},
               batch, train=True)
    assert float(res.terms["pose_acc_rms"].numerator) / res.terms["pose_acc_rms"].mass > 0.05
    res.terms["pose_acc_rms"].numerator.backward()
    assert float((half.grad * gt).sum()) < 0.0            # descent direction grows the amplitude


# ------------------------------------------------------------------ deep supervision

def gt_keys(batch: dict, joints_world: Tensor, seed: int = 11) -> dict:
    """The kindyn GT keys the SMPL-X / motion losses read, around a world trajectory."""
    torch.manual_seed(seed)
    n = joints_world.shape[0]
    return {**batch,
            "smplx_joints_world": joints_world + 0.02 * torch.randn_like(joints_world),
            "smplx_root_rot": roma.random_rotmat(n),
            "smplx_body_rot": roma.rotvec_to_rotmat(0.3 * torch.randn(n, 21, 3)),
            "smplx_hand_rot": roma.rotvec_to_rotmat(0.2 * torch.randn(n, 30, 3)),
            "smplx_betas": torch.zeros(n, 10),
            "smplx_valid": torch.ones(n, dtype=torch.bool)}


def test_motion_layer_terms_pool_the_intermediate_layers():
    cfg = base_cfg()
    cfg["motion_supervision"].update(enabled=True, stencil="forward", label_smooth_sec=0.0,
                                     layer_weight=0.5)
    cfg["motion_supervision"]["loss"].update(vel=0.0, acc=0.0, ang_vel=0.0, ang_acc=0.0,
                                             pose_vel=1.0, pose_acc=0.0, pose_ang_vel=1.0,
                                             pose_ang_acc=0.0)
    loss = MotionLoss(cfg, None, "cpu")
    assert loss.term_names == ("pose_vel", "pose_ang_vel", "pose_vel_layer", "pose_ang_vel_layer")

    _, root, _, batch = world_series(seed=5)
    n = root.shape[0]
    joints = 0.3 * torch.randn(n, 22, 3)
    batch = {**batch, "smplx_valid": torch.ones(n, dtype=torch.bool),
             "smplx_joints_world": 0.3 * torch.randn(n, 22, 3), "smplx_root_rot": root.clone()}

    def run(layer_joints):
        return loss({"motion": None,
                     "smplx": {"joints_world": joints, "root_rot_world": root,
                               "joints_world_layers": layer_joints + [joints],
                               "root_rot_world_layers": [root] * (len(layer_joints) + 1)}},
                    batch, train=True)

    # Two intermediate layers, both equal to the final trajectory: the pooled layer term is
    # the final term at `layer_weight`, over twice the mass.
    result = run([joints, joints])
    assert set(result.terms) == set(loss.term_names)
    for quantity in ("vel", "ang_vel"):
        base, layer = result.terms[f"pose_{quantity}"], result.terms[f"pose_{quantity}_layer"]
        assert base.mass > 0
        assert layer.mass == pytest.approx(2.0 * base.mass)
        assert float(layer.numerator) == pytest.approx(1.0 * float(base.numerator), rel=1e-6)
        assert (float(layer.numerator) / layer.mass
                == pytest.approx(0.5 * float(base.numerator) / base.mass, rel=1e-6))
    # A different intermediate trajectory moves the layer term and leaves the final one.
    moved = run([joints + 0.05 * torch.randn_like(joints), joints])
    assert float(moved.terms["pose_vel"].numerator) == pytest.approx(
        float(result.terms["pose_vel"].numerator), rel=1e-6)
    assert float(moved.terms["pose_vel_layer"].numerator) > float(
        result.terms["pose_vel_layer"].numerator)
    # The amplitude terms get their own layer twin.
    cfg["motion_supervision"]["loss"].update(pose_acc_rms=1.0)
    assert MotionLoss(cfg, None, "cpu").term_names[-3:] == (
        "pose_vel_layer", "pose_ang_vel_layer", "pose_acc_rms_layer")


def test_smplx_layer_terms_repeat_the_body_terms(body):
    smplx_out, tokens, blocks, batch = synthetic(body)
    refiner = make_refiner(randomize=True, iterative=True, token=ONE_SIDED_TOKEN)
    pred = refiner(smplx_out, tokens, blocks, batch, body)["smplx"]
    batch = gt_keys(batch, pred["joints_world"].detach())

    cfg = base_cfg()
    cfg["smplx_supervision"].update(enabled=True, layer_weight=0.5)
    cfg["smplx_supervision"]["loss"].update(kp2d=0.0, kp3d=5.0, orient=1.0, pose=1.0,
                                            betas=0.0, cam=0.0, root_bias=2.0, root_shape=2.0)
    stub = SimpleNamespace(
        head_smplx=SimpleNamespace(hands=True, num_joints=pred["joints_cam"].shape[1],
                                   camera="ray"),
        refiner=refiner)
    loss = SmplxLoss(cfg, stub, "cpu")
    assert loss.layer_terms == ("kp3d", "orient", "pose", "root_bias", "root_shape")

    # Three layers, the two intermediate ones equal to the final body.
    for key in ("joints_cam", "root_6d", "body_6d", "pelvis_world"):
        pred[f"{key}_layers"] = [pred[key]] * 3
    result = loss({"smplx": pred}, batch, train=True)
    assert set(result.terms) == set(loss.term_names)
    for name in loss.layer_terms:
        base, layer = result.terms[name], result.terms[f"{name}_layer"]
        assert base.mass > 0
        assert layer.mass == pytest.approx(2.0 * base.mass)
        assert (float(layer.numerator) / layer.mass
                == pytest.approx(0.5 * float(base.numerator) / base.mass, rel=1e-4))


def test_band_limited_terms_reach_the_pose():
    cfg = base_cfg()
    cfg["motion_supervision"].update(enabled=True, stencil="forward", label_smooth_sec=0.0)
    cfg["motion_supervision"]["loss"].update(vel=0.0, acc=0.0, ang_vel=0.0, ang_acc=0.0,
                                             pose_vel=1.0, pose_ss_acc=0.1, pose_ss_ang_acc=0.25)
    loss = MotionLoss(cfg, None, "cpu")
    assert loss.term_names == ("pose_vel", "pose_ss_acc", "pose_ss_ang_acc")

    _, root, _, batch = world_series(seed=3)
    n = root.shape[0]
    joints = (0.3 * torch.randn(n, 22, 3)).requires_grad_(True)
    batch = {**batch, "smplx_valid": torch.ones(n, dtype=torch.bool),
             "smplx_joints_world": 0.3 * torch.randn(n, 22, 3),
             "smplx_root_rot": roma.random_rotmat(n)}
    result = loss({"motion": None, "smplx": {"joints_world": joints, "root_rot_world": root}},
                  batch, train=True)
    assert set(result.terms) == set(loss.term_names)
    result.terms["pose_ss_acc"].numerator.backward()
    assert joints.grad is not None and torch.isfinite(joints.grad).all() and joints.grad.abs().sum() > 0


# ------------------------------------------------------------------ round 8: gravity, feedback, masking

def camera_down_prior(batch: dict) -> Tensor:
    """The pooled camera +y axis per clip (world), expanded to the frames."""
    seq_len = int(batch["seq_len"])
    rot_wc = batch["cam_from_world"][:, :3, :3].transpose(1, 2)
    down = rot_wc[:, :, 1].view(-1, seq_len, 3).mean(dim=1)
    down = down / down.norm(dim=-1, keepdim=True)
    return down[:, None].expand(-1, seq_len, 3).reshape(-1, 3)


def test_gravity_head_is_the_camera_axis_at_init(body):
    smplx_out, tokens, blocks, batch = synthetic(body)
    refiner = make_refiner(randomize=False, iterative=True, token=ONE_SIDED_TOKEN,
                           outputs=ALL_OUTPUTS, camera_context=True, camera_axes=True,
                           residual_feedback=True)
    out = refiner(smplx_out, tokens, blocks, batch, body)
    gravity = out["gravity"]
    prior = camera_down_prior(batch)
    assert torch.allclose(gravity["world"], prior, atol=1e-5)
    assert torch.allclose(gravity["prior_world"], prior, atol=1e-5)
    assert torch.allclose(gravity["world"].norm(dim=-1), torch.ones(len(prior)), atol=1e-5)
    rot_wr = out["smplx"]["root_rot_world"]
    assert torch.allclose(gravity["body"], (rot_wr.transpose(1, 2) @ prior[..., None])[..., 0], atol=1e-5)
    assert len(gravity["world_layers"]) == 2 and gravity["world_layers"][-1] is gravity["world"]
    assert len(out["contact"]["logits_layers"]) == 2 and len(out["force"]["forces_layers"]) == 2
    assert torch.allclose(out["smplx"]["joints_cam"], smplx_out["joints_cam"], atol=1e-4)
    with pytest.raises(ValueError):                                # the prior needs the camera axes
        make_refiner(randomize=False, outputs=ALL_OUTPUTS, camera_context=True)
    with pytest.raises(ValueError):                                # the axes extend the context
        make_refiner(randomize=False, camera_axes=True)
    with pytest.raises(ValueError):                                # the residual needs every head
        make_refiner(randomize=False, iterative=True, token=ONE_SIDED_TOKEN,
                     outputs=("pose", "contact", "force"), camera_context=True,
                     camera_axes=True, residual_feedback=True)


def test_world_frame_independence_with_gravity_and_residual_feedback(body):
    smplx_out, tokens, blocks, batch = synthetic(body)
    refiner = make_refiner(randomize=True, iterative=True, token=ONE_SIDED_TOKEN,
                           outputs=ALL_OUTPUTS, camera_context=True, camera_axes=True,
                           residual_feedback=True)
    torch.nn.init.normal_(refiner.heads["gravity"][2].weight, std=0.5)   # a real correction
    out = refiner(smplx_out, tokens, blocks, batch, body)
    assert (out["gravity"]["world"] - out["gravity"]["prior_world"]).abs().max() > 1e-3
    torch.manual_seed(7)
    rot0 = roma.random_rotmat(1)[0]
    t0 = torch.tensor([3.0, -2.0, 5.0])
    g_inv = torch.eye(4)
    g_inv[:3, :3] = rot0.T
    g_inv[:3, 3] = -rot0.T @ t0
    moved = dict(batch)
    moved["cam_from_world"] = batch["cam_from_world"] @ g_inv
    moved["gravity_world"] = batch["gravity_world"] @ rot0.T
    out2 = refiner(smplx_out, tokens, blocks, moved, body)
    for key in ("joints_cam", "pelvis_cam", "root_rot", "body_rot"):
        assert torch.allclose(out["smplx"][key], out2["smplx"][key], atol=1e-4), key
    assert torch.allclose(out["contact"]["logits"], out2["contact"]["logits"], atol=1e-4)
    assert torch.allclose(out["force"]["forces"], out2["force"]["forces"], atol=1e-4)
    assert torch.allclose(out["gravity"]["body"], out2["gravity"]["body"], atol=1e-4)
    assert torch.allclose(out2["gravity"]["world"], (rot0 @ out["gravity"]["world"].T).T, atol=1e-4)


def test_residual_feedback_detaches_the_body_and_keeps_the_forces_live(body):
    smplx_out, tokens, blocks, batch = synthetic(body)
    refiner = make_refiner(randomize=True, iterative=True, token=ONE_SIDED_TOKEN,
                           outputs=ALL_OUTPUTS, camera_context=True, camera_axes=True,
                           residual_feedback=True)
    out = refiner(smplx_out, tokens, blocks, batch, body)
    seq_len = int(batch["seq_len"])
    n = len(batch["frame_valid"])
    pelvis = out["smplx"]["pelvis_world"].detach().requires_grad_(True)
    rot_wr = out["smplx"]["root_rot_world"].detach().requires_grad_(True)
    body_rot = out["smplx"]["body_rot"].detach().requires_grad_(True)
    betas = out["smplx"]["betas"].detach().view(-1, seq_len, 10)[:, 0].clone().requires_grad_(True)
    frame_in = out["smplx"]["root_rot_world_in"].detach().clone().requires_grad_(True)
    gravity = out["gravity"]["world"].detach().clone().requires_grad_(True)
    forces = (0.3 * torch.randn(n, 6, 3)).requires_grad_(True)
    probs = torch.full((n, 6), 0.8).requires_grad_(True)
    seconds = batch["frame_pos_sec"].view(-1, seq_len)
    valid = batch["frame_valid"].view(-1, seq_len)
    residual = refiner._residual(pelvis, rot_wr, body_rot, betas, forces, frame_in, probs,
                                 gravity, seconds, valid)
    rows = valid.clone()
    rows[:, :2] = rows[:, -2:] = False
    assert residual.shape == (n, 6)
    assert residual.view(-1, seq_len, 6)[~rows].abs().max() == 0        # outside the stencil
    assert residual.view(-1, seq_len, 6)[rows].abs().max() > 0
    residual.sum().backward()
    assert forces.grad is not None and forces.grad.abs().sum() > 0     # forces live
    for name, tensor in (("pelvis", pelvis), ("rot_wr", rot_wr), ("body_rot", body_rot),
                         ("betas", betas), ("frame_in", frame_in), ("gravity", gravity),
                         ("probs", probs)):
        assert tensor.grad is None, name                                # everything else detached
    rot_wr, body_rot, betas = rot_wr.detach(), body_rot.detach(), betas.detach()
    # Too short for the +-2 stencil: a graph-connected zero.
    short = refiner._residual(pelvis[:8].detach(), rot_wr[:8], body_rot[:8], betas[:2], forces[:8],
                              frame_in[:8].detach(), probs[:8].detach(), gravity[:8].detach(),
                              seconds[:, :4].reshape(2, 4), valid[:, :4].reshape(2, 4))
    assert short.shape == (8, 6) and short.abs().max() == 0 and short.requires_grad
    # A still body under gravity needs one body weight along -g at the root (the loss's
    # convention): with zero forces the force residual has unit magnitude.
    still = torch.zeros(n, 6, 3)
    p0 = out["smplx"]["pelvis_world"].detach().view(-1, seq_len, 3)[:, :1].expand(-1, seq_len, 3).reshape(n, 3)
    r0 = rot_wr.view(-1, seq_len, 3, 3)[:, :1].expand(-1, seq_len, 3, 3).reshape(n, 3, 3)
    b0 = body_rot.view(-1, seq_len, 21, 3, 3)[:, :1].expand(-1, seq_len, 21, 3, 3).reshape(n, 21, 3, 3)
    res0 = refiner._residual(p0, r0, b0, betas, still, r0, probs, out["gravity"]["world"].detach(),
                             seconds, valid)
    assert torch.allclose(res0.view(-1, seq_len, 6)[rows][:, :3].norm(dim=-1), torch.ones(int(rows.sum())), atol=5e-3)
    # The forces enter in the INPUT body frame `frame_in` and the residual is read in the
    # CURRENT root frame: on the still body the force part moves by exactly the gated sum
    # transported between the two frames (a wrong frame would fail this).
    torch.manual_seed(5)
    frame_in = roma.random_rotmat(1).expand(n, 3, 3)
    with_forces = refiner._residual(p0, r0, b0, betas, forces.detach(), frame_in, probs,
                                    out["gravity"]["world"].detach(), seconds, valid)
    transported = -torch.einsum("bij,bj->bi", r0.transpose(1, 2) @ frame_in,
                                (forces.detach() * probs[..., None]).sum(dim=1))
    delta = (with_forces - res0).view(-1, seq_len, 6)[rows][:, :3]
    assert torch.allclose(delta, transported.view(-1, seq_len, 3)[rows], atol=1e-4)


def test_frame_masking_only_in_training(body):
    smplx_out, tokens, blocks, batch = synthetic(body)
    refiner = make_refiner(randomize=True, iterative=True, token=ONE_SIDED_TOKEN, frame_mask_p=0.9)
    torch.nn.init.normal_(refiner.mask_token, std=1.0)
    plain = make_refiner(randomize=True, iterative=True, token=ONE_SIDED_TOKEN)
    out_eval = refiner(smplx_out, tokens, blocks, batch, body)
    out_plain = plain(smplx_out, tokens, blocks, batch, body)
    assert torch.allclose(out_eval["smplx"]["joints_cam"], out_plain["smplx"]["joints_cam"], atol=1e-6)
    refiner.train()
    torch.manual_seed(0)
    out_train = refiner(smplx_out, tokens, blocks, batch, body)
    assert (out_train["smplx"]["joints_cam"] - out_eval["smplx"]["joints_cam"]).abs().max() > 1e-4
    with pytest.raises(ValueError):
        make_refiner(randomize=False, frame_mask_p=1.0)


def test_gravity_loss_supervises_measured_clips_and_reports_both(body):
    from model.loss.gravity import GravityLoss
    cfg = base_cfg()
    cfg["gravity_supervision"].update({"enabled": True, "weight": 2.0, "measured_only": True,
                                       "layer_weight": 0.5})
    loss = GravityLoss(cfg, None, "cpu")
    seq_len, n_clips = 4, 3
    n = seq_len * n_clips
    torch.manual_seed(2)
    gravity = torch.nn.functional.normalize(torch.randn(n_clips, 3), dim=-1)
    world = torch.nn.functional.normalize(gravity + 0.3 * torch.randn(n_clips, 3), dim=-1)
    layer0 = torch.nn.functional.normalize(torch.randn(n_clips, 3), dim=-1)
    expand = lambda v: v[:, None].expand(n_clips, seq_len, 3).reshape(n, 3)
    measured = torch.tensor([True, False, True]).repeat_interleave(seq_len)
    batch = {"seq_len": seq_len, "frame_valid": torch.ones(n, dtype=torch.bool),
             "gravity_world": expand(gravity), "gravity_measured": measured}
    out = {"gravity": {"world": expand(world).requires_grad_(True), "prior_world": expand(gravity),
                       "world_layers": [expand(layer0), expand(world)]}}
    result = loss(out, batch, train=True)
    assert loss.term_names == ("cos", "cos_layer")
    cos = (1.0 - (world * gravity).sum(-1))
    assert torch.allclose(result.terms["cos"].numerator, 2.0 * cos[[0, 2]].sum(), atol=1e-6)
    assert result.terms["cos"].mass == 2.0
    layer_cos = (1.0 - (layer0 * gravity).sum(-1))
    assert torch.allclose(result.terms["cos_layer"].numerator, 0.5 * 2.0 * layer_cos[[0, 2]].sum(), atol=1e-6)
    assert result.terms["cos_layer"].mass == 2.0
    metrics = loss.metrics(result.stats)
    angle = torch.rad2deg(torch.acos((world * gravity).sum(-1).clamp(-1, 1)))
    assert math.isclose(metrics["angle_measured"], float(angle[[0, 2]].mean()), rel_tol=1e-5)
    assert math.isclose(metrics["angle_fallback"], float(angle[1]), rel_tol=1e-5)
    assert metrics["prior_angle_measured"] < 0.1 and metrics["prior_angle_fallback"] < 0.1   # acos at 1 in fp32
    result.terms["cos"].numerator.backward()
    assert out["gravity"]["world"].grad.abs().sum() > 0


def test_detached_heads_leave_the_trunk_to_the_pose_losses(body):
    smplx_out, tokens, blocks, batch = synthetic(body)
    refiner = make_refiner(randomize=True, iterative=True, token=ONE_SIDED_TOKEN,
                           outputs=ALL_OUTPUTS, camera_context=True, camera_axes=True,
                           residual_feedback=True, head_grad_scale=0.0)
    out = refiner(smplx_out, tokens, blocks, batch, body)
    head_loss = lambda o: (o["contact"]["logits"].square().sum() + o["force"]["forces"].square().sum()
                           + o["gravity"]["world"].sum())
    head_loss(out).backward()
    trunk = [p for n, p in refiner.named_parameters() if n.startswith("temporal.") or n.startswith("input_proj")]
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in trunk)
    # A partial scale passes exactly that fraction of the shared gradient (same values).
    grads = {}
    for scale in (1.0, 0.25):
        half = make_refiner(randomize=True, iterative=True, token=ONE_SIDED_TOKEN,
                            outputs=ALL_OUTPUTS, camera_context=True, camera_axes=True,
                            residual_feedback=True, head_grad_scale=scale)
        o = half(smplx_out, tokens, blocks, batch, body)
        assert torch.allclose(o["contact"]["logits"], out["contact"]["logits"], atol=1e-6)
        head_loss(o).backward()
        grads[scale] = half.input_proj.weight.grad.clone()
    # Not exactly 0.25x: paths through the fed-back values are scaled twice (s^2).
    ratio = grads[0.25].norm() / grads[1.0].norm()
    cosine = (grads[0.25] * grads[1.0]).sum() / (grads[0.25].norm() * grads[1.0].norm())
    assert 0.2 < float(ratio) < 0.3 and float(cosine) > 0.99
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in refiner.heads["contact"].parameters())
    refiner.zero_grad()
    out = refiner(smplx_out, tokens, blocks, batch, body)
    out["smplx"]["joints_world"].sum().backward()                  # the pose loss still trains the trunk
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trunk)
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in refiner.heads["contact"].parameters())


def test_gravity_pooling_votes_with_unit_vectors_and_never_vanishes():
    n_clips, seq_len = 2, 3
    valid = torch.ones(n_clips, seq_len, dtype=torch.bool)
    rot = torch.eye(3).expand(n_clips * seq_len, 3, 3)
    down = torch.tensor([0.0, 1.0, 0.0]).expand(n_clips * seq_len, 3)
    delta = torch.zeros(n_clips * seq_len, 3)
    delta[0] = torch.tensor([10.0, -1.0, 0.0])                     # a huge correction on one frame
    pooled = TemporalRefiner.pool_gravity(delta, rot, down, n_clips, seq_len, valid)
    expected = torch.tensor([1.0, 2.0, 0.0]) / 3.0                  # mean of (1,0,0), (0,1,0), (0,1,0)
    assert torch.allclose(pooled[0], expected / expected.norm(), atol=1e-6)
    assert torch.allclose(pooled[3], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    delta = -down.clone()                                            # every vote cancels the axis
    delta[:seq_len] = torch.tensor([0.0, -2.0, 0.0])                 # (0,1,0) + (0,-2,0) = (0,-1,0)
    delta[seq_len - 1] = torch.tensor([0.0, 0.0, 0.0])               # ... except one frame: (0,1,0)
    pooled = TemporalRefiner.pool_gravity(delta, rot, down, n_clips, seq_len, valid)
    assert torch.allclose(pooled[:seq_len].norm(dim=-1), torch.ones(seq_len), atol=1e-6)
    assert torch.allclose(pooled[0], torch.tensor([0.0, -1.0, 0.0]), atol=1e-6)  # 2:1 majority survives
    delta = torch.zeros(n_clips * seq_len, 3)
    delta[seq_len:] = torch.tensor([0.0, -1.0, 0.0])                 # exactly zero votes: the axis wins
    pooled = TemporalRefiner.pool_gravity(delta, rot, down, n_clips, seq_len, valid)
    assert torch.allclose(pooled[seq_len], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
