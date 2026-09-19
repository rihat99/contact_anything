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
* the optional known-gravity input: who is given it, that a given clip's estimate
  IS the given vector on every layer, that its four token channels carry it in
  the body frame, that nothing reaches the clips that were not given it, and that
  the gravity loss drops the given clips;
* the limb tokens: a seven-token refiner is still the identity at init and still
  world-frame independent (with and without decoder contact tokens), the contact
  and force gradients — each head's loss on its own, at full and at scaled head
  gradient — reach the limb projections, the limb feedback and both attentions of
  the alternating block, and that block alone is an identity at init that mixes a
  frame's slots at once, stays local in time, and matches a per-(clip, slot)
  reference over clips of different spacings, holes and absolute times;
* the gravity force frame (``force_frame: gravity``): a rotation whose up column
  is the clip's own gravity estimate negated, identity at init, the second force
  component points straight up, the world force is unchanged by a rigid
  re-definition of the world, the heading falls back to the body's ``+y``
  when its ``+z`` lies along gravity, and — with limb tokens and per-layer
  gravity estimates that differ — every layer's transported force is the same
  WORLD vector as its raw output in its own frame;
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
from model.rope import CrossModalRopeModule
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
        "gravity_measured": torch.ones(n, dtype=torch.bool),
    }
    return smplx_out, tokens, blocks, batch


ALL_TOKEN = {"local_rotations": True, "gravity": True, "raw_minus_mean": True}
ONE_SIDED_TOKEN = {**ALL_TOKEN, "one_sided_velocity": True, "joint_velocity": True}


ALL_OUTPUTS = ("pose", "contact", "motion", "force", "gravity")
#: ``model.refiner.gravity_input`` as the round-9 arms set it, with the training draw off
#: (the tests exercise the eligibility and the draw separately).
GRAVITY_INPUT = {"enabled": True, "p_given": 1.0, "measured_only": True, "eval_given": True}


def smplx_model_path() -> str:
    cfg = yaml.safe_load((REPO / "configs" / "base.yaml").read_text())
    return cfg["model"]["smplx"]["model_path"]


def make_refiner(randomize: bool, root_smooth_sec: float = 0.0, pose_smooth_sec: float = 0.0,
                 camera_context: bool = False, learn_smoothing: bool = False,
                 token: dict | None = None, iterative: bool = False,
                 feedback_delta: bool = False, outputs=("pose", "contact", "motion", "force"),
                 camera_axes: bool = False, gravity_input: dict | None = None,
                 residual_feedback: bool = False,
                 frame_mask_p: float = 0.0, head_grad_scale: float = 1.0,
                 limb_tokens: bool = False, num_contact_tokens: int = 6,
                 force_frame: str = "body", per_frame: bool = False) -> TemporalRefiner:
    torch.manual_seed(1)
    refiner = TemporalRefiner(DECODER_DIM, outputs,
                              num_contact_tokens=num_contact_tokens, dim=64, num_layers=2,
                              num_heads=4,
                              window=0.5, root_smooth_sec=root_smooth_sec,
                              pose_smooth_sec=pose_smooth_sec, learn_smoothing=learn_smoothing,
                              token=token, iterative=iterative, feedback_delta=feedback_delta,
                              camera_context=camera_context, camera_axes=camera_axes,
                              gravity_input=gravity_input,
                              residual_feedback=residual_feedback, frame_mask_p=frame_mask_p,
                              head_grad_scale=head_grad_scale,
                              smplx_model_path=smplx_model_path() if residual_feedback else None,
                              dropout=0.0, limb_tokens=limb_tokens, force_frame=force_frame,
                              per_frame=per_frame)
    if randomize:
        for head in refiner.heads.values():
            torch.nn.init.normal_(head[2].weight, std=0.02)
        for block in refiner.temporal.blocks:
            # The zero-init output projections of either block variant (one attention, or
            # the alternating pair), then the FFN's.
            for name, module in block.named_children():
                if name.startswith("proj") and isinstance(module, torch.nn.Linear):
                    torch.nn.init.normal_(module.weight, std=0.02)
            torch.nn.init.normal_(block.ffn[3].weight, std=0.02)
        for projection in (refiner.feedback_proj, refiner.limb_feedback_proj):
            if projection is not None:
                torch.nn.init.normal_(projection.weight, std=0.02)
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


@pytest.mark.parametrize("gravity_input", [None, GRAVITY_INPUT])
def test_world_frame_independence_with_gravity_and_residual_feedback(body, gravity_input):
    smplx_out, tokens, blocks, batch = synthetic(body)
    refiner = make_refiner(randomize=True, iterative=True, token=ONE_SIDED_TOKEN,
                           outputs=ALL_OUTPUTS, camera_context=True, camera_axes=True,
                           gravity_input=gravity_input, residual_feedback=True)
    torch.nn.init.normal_(refiner.heads["gravity"][2].weight, std=0.5)   # a real correction
    out = refiner(smplx_out, tokens, blocks, batch, body)
    assert (out["gravity"]["world"] - out["gravity"]["prior_world"]).abs().max() > 1e-3
    assert out["gravity"]["given"].all() == (gravity_input is not None)
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


def test_gravity_loss_supervises_guessed_measured_clips_and_reports_both(body):
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
    # Clip 2's gravity was handed to the refiner: measured, but nothing to learn from.
    given = torch.tensor([False, False, True]).repeat_interleave(seq_len)
    batch = {"seq_len": seq_len, "frame_valid": torch.ones(n, dtype=torch.bool),
             "gravity_world": expand(gravity), "gravity_measured": measured}
    out = {"gravity": {"world": expand(world).requires_grad_(True), "prior_world": expand(gravity),
                       "given": given, "world_layers": [expand(layer0), expand(world)]}}
    result = loss(out, batch, train=True)
    assert loss.term_names == ("cos", "cos_layer")
    cos = (1.0 - (world * gravity).sum(-1))
    assert torch.allclose(result.terms["cos"].numerator, 2.0 * cos[0], atol=1e-6)
    assert result.terms["cos"].mass == 1.0
    layer_cos = (1.0 - (layer0 * gravity).sum(-1))
    assert torch.allclose(result.terms["cos_layer"].numerator, 0.5 * 2.0 * layer_cos[0], atol=1e-6)
    assert result.terms["cos_layer"].mass == 1.0
    metrics = loss.metrics(result.stats)
    angle = torch.rad2deg(torch.acos((world * gravity).sum(-1).clamp(-1, 1)))
    assert math.isclose(metrics["angle_measured"], float(angle[0]), rel_tol=1e-5)
    assert math.isclose(metrics["angle_fallback"], float(angle[1]), rel_tol=1e-5)
    assert metrics["prior_angle_measured"] < 0.1 and metrics["prior_angle_fallback"] < 0.1   # acos at 1 in fp32
    result.terms["cos"].numerator.backward()
    assert out["gravity"]["world"].grad.abs().sum() > 0

    # Every clip given: nothing to score, but the term stays graph-connected (DDP has no
    # unused-parameter tolerance) and the metric reports nothing rather than a perfect score.
    world_all = expand(world).requires_grad_(True)
    out = {"gravity": {"world": world_all, "prior_world": expand(gravity),
                       "given": torch.ones(n, dtype=torch.bool),
                       "world_layers": [expand(layer0), world_all]}}
    result = loss(out, batch, train=True)
    assert result.terms["cos"].mass == 0.0
    result.terms["cos"].numerator.backward()
    assert world_all.grad is not None and world_all.grad.abs().sum() == 0
    assert math.isnan(loss.metrics(result.stats)["angle_measured"])


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


# ------------------------------------------------------------------ round 9: the known-gravity input

def test_gravity_input_eligibility_and_draws(body):
    """Which clips are handed the scene's gravity: eligible, drawn in training."""
    smplx_out, tokens, blocks, batch = synthetic(body)
    seq_len = int(batch["seq_len"])
    n_clips = len(batch["frame_valid"]) // seq_len
    batch = {**batch, "gravity_measured": torch.tensor([True, False]).repeat_interleave(seq_len)}

    def make(**overrides):
        return make_refiner(randomize=False, iterative=True, token=ONE_SIDED_TOKEN,
                            outputs=ALL_OUTPUTS, camera_context=True, camera_axes=True,
                            gravity_input={**GRAVITY_INPUT, **overrides})

    def given(refiner):
        return refiner._given_gravity(batch, n_clips, seq_len, torch.device("cpu")).tolist()

    assert given(make()) == [True, False]                 # eval: every eligible clip, no draw
    assert given(make(eval_given=False)) == [False, False]
    assert given(make(measured_only=False)) == [True, True]
    assert given(make().train()) == [True, False]         # p_given 1: the draw always fires
    assert given(make(p_given=0.0).train()) == [False, False]
    with pytest.raises(ValueError):                       # it replaces the gravity output
        make_refiner(randomize=False, gravity_input={"enabled": True})
    with pytest.raises(ValueError):
        make(p_given=1.5)


def test_given_gravity_is_the_input_and_reaches_only_its_clips(body):
    smplx_out, tokens, blocks, batch = synthetic(body)
    seq_len = int(batch["seq_len"])
    measured = torch.tensor([True, False]).repeat_interleave(seq_len)
    gravity = torch.nn.functional.normalize(
        torch.tensor([[0.2, -0.9, 0.3], [0.0, -1.0, 0.0]]), dim=-1).repeat_interleave(seq_len, dim=0)
    batch = {**batch, "gravity_measured": measured, "gravity_world": gravity}
    # No token.gravity: that channel hands the same vector to EVERY clip, which is what
    # the input replaces (the round-8 recipe has it off).
    token = {**ONE_SIDED_TOKEN, "gravity": False}
    make = lambda randomize: make_refiner(
        randomize=randomize, iterative=True, token=token, outputs=ALL_OUTPUTS,
        camera_context=True, camera_axes=True, gravity_input=GRAVITY_INPUT)

    # At init the refiner is still the per-frame body, and the clip that guesses guesses
    # the camera's down axis — the input changes neither.
    init = make(randomize=False)(smplx_out, tokens, blocks, batch, body)
    assert torch.allclose(init["smplx"]["joints_cam"], smplx_out["joints_cam"], atol=1e-4)
    assert torch.count_nonzero(init["contact"]["logits"]) == 0
    assert torch.allclose(init["gravity"]["world"][seq_len:],
                          camera_down_prior(batch)[seq_len:], atol=1e-5)

    refiner = make(randomize=True)
    torch.nn.init.normal_(refiner.heads["gravity"][2].weight, std=0.5)
    seen = {}
    refiner.geometry_norm.register_forward_pre_hook(lambda _m, args: seen.update(x=args[0]))
    out = refiner(smplx_out, tokens, blocks, batch, body)

    assert torch.equal(out["gravity"]["given"], measured)
    # The given clip's estimate IS the given vector, on every layer; the other one guesses.
    assert torch.equal(out["gravity"]["world"][:seq_len], gravity[:seq_len])
    assert all(torch.equal(w[:seq_len], gravity[:seq_len]) for w in out["gravity"]["world_layers"])
    guessed = out["gravity"]["world"][seq_len:]
    assert (guessed - gravity[seq_len:]).abs().max() > 1e-3
    assert torch.allclose(guessed.norm(dim=-1), torch.ones(seq_len), atol=1e-5)
    # The four token channels (last of the geometry block): the given vector in the body
    # frame, then the flag — both zero on the clip that was not given it.
    rot_wr = out["smplx"]["root_rot_world_in"]
    flag = out["gravity"]["given"][:, None].float()
    expected = torch.cat([(rot_wr.transpose(1, 2) @ gravity[..., None])[..., 0] * flag, flag], dim=-1)
    assert torch.allclose(seen["x"][:, -4:], expected, atol=1e-6)

    # Another gravity moves the given clip and leaves the other one bit-identical.
    other = torch.nn.functional.normalize(
        torch.tensor([[-0.4, -0.8, 0.4], [0.1, -0.9, 0.2]]), dim=-1).repeat_interleave(seq_len, dim=0)
    out2 = refiner(smplx_out, tokens, blocks, {**batch, "gravity_world": other}, body)
    for key, value in (("smplx", "joints_cam"), ("contact", "logits"), ("gravity", "world")):
        a, b = out[key][value], out2[key][value]
        assert torch.equal(a[seq_len:], b[seq_len:]), value
        assert (a[:seq_len] - b[:seq_len]).abs().max() > 1e-6, value


# ------------------------------------------------------------------ round 10: the limb tokens

LIMB_OUTPUTS = ("pose", "gravity", "contact", "force")


def make_limb_refiner(randomize: bool, num_contact_tokens: int = 6) -> TemporalRefiner:
    """A seven-token refiner: iterative, camera context + axes, contact and force heads."""
    return make_refiner(randomize=randomize, iterative=True, token=ONE_SIDED_TOKEN,
                        outputs=LIMB_OUTPUTS, camera_context=True, camera_axes=True,
                        limb_tokens=True, num_contact_tokens=num_contact_tokens)


@pytest.mark.parametrize("num_contact_tokens", [6, 0])
def test_limb_tokens_identity_at_init(body, num_contact_tokens):
    smplx_out, tokens, blocks, batch = synthetic(body)
    refiner = make_limb_refiner(randomize=False, num_contact_tokens=num_contact_tokens)
    n = len(batch["frame_valid"])
    assert refiner.temporal.num_slots == 7 and refiner.temporal.alternating
    assert (refiner.limb_token_norm is None) == (num_contact_tokens == 0)
    out = refiner(smplx_out, tokens, blocks, batch, body)
    assert torch.allclose(out["smplx"]["joints_cam"], smplx_out["joints_cam"], atol=1e-4)
    assert torch.allclose(out["smplx"]["body_rot"], smplx_out["body_rot"], atol=1e-5)
    assert out["contact"]["logits"].shape == (n, 6) and out["force"]["forces"].shape == (n, 6, 3)
    assert torch.count_nonzero(out["contact"]["logits"]) == 0
    assert torch.count_nonzero(out["force"]["forces"]) == 0
    assert len(out["contact"]["logits_layers"]) == len(out["force"]["forces_layers"]) == 2


@pytest.mark.parametrize("num_contact_tokens", [6, 0])
def test_limb_tokens_world_frame_independence(body, num_contact_tokens):
    smplx_out, tokens, blocks, batch = synthetic(body)
    refiner = make_limb_refiner(randomize=True, num_contact_tokens=num_contact_tokens)
    torch.nn.init.normal_(refiner.heads["gravity"][2].weight, std=0.5)   # a real correction
    out = refiner(smplx_out, tokens, blocks, batch, body)
    assert torch.count_nonzero(out["contact"]["logits"]) > 0          # the limb heads are live
    assert out["force"]["forces"].abs().max() > 0
    assert (out["smplx"]["joints_cam"] - smplx_out["joints_cam"]).abs().max() > 1e-4

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
    for key in ("joints_cam", "pelvis_cam", "root_rot", "body_rot", "kp2d_crop"):
        assert torch.allclose(out["smplx"][key], out2["smplx"][key], atol=1e-4), key
    assert torch.allclose(out["contact"]["logits"], out2["contact"]["logits"], atol=1e-4)
    assert torch.allclose(out["force"]["forces"], out2["force"]["forces"], atol=1e-4)
    assert torch.allclose(out["gravity"]["body"], out2["gravity"]["body"], atol=1e-4)
    expected = (rot0 @ out["smplx"]["pelvis_world"].T).T + t0
    assert torch.allclose(out2["smplx"]["pelvis_world"], expected, atol=1e-4)


def test_limb_token_gradients_reach_the_limb_path(body):
    smplx_out, tokens, blocks, batch = synthetic(body)
    refiner = make_limb_refiner(randomize=True)
    out = refiner(smplx_out, tokens, blocks, batch, body)
    (out["contact"]["logits"].square().sum() + out["force"]["forces"].square().sum()).backward()
    grads = dict(refiner.named_parameters())
    reached = ["limb_input_proj.weight", "limb_feedback_proj.weight", "temporal.slot_embed"]
    reached += [f"temporal.blocks.{layer}.{attn}.weight" for layer in range(2)
                for attn in ("qkv_temporal", "qkv_frame")]
    for name in reached:
        grad = grads[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    assert all(p.grad is not None and p.grad.abs().sum() > 0
               for p in refiner.heads["contact"].parameters())

    refiner.zero_grad()
    out = refiner(smplx_out, tokens, blocks, batch, body)
    out["smplx"]["joints_world"].sum().backward()                  # the body path of the pose
    assert refiner.input_proj.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and p.grad.abs().sum() > 0
               for p in refiner.heads["pose"].parameters())


def alternating_module(dim: int = 32, num_slots: int = 3, num_layers: int = 2):
    """A bare alternating transformer and a clip of random tokens for it."""
    torch.manual_seed(0)
    module = CrossModalRopeModule(dim=dim, num_slots=num_slots, num_layers=num_layers,
                                  num_heads=4, mlp_ratio=2.0, dropout=0.0, window=0.5,
                                  time_scale=25.0, alternating=True).eval()
    seq_len = 30
    tokens = torch.randn(seq_len, num_slots, dim)
    seconds = torch.arange(seq_len, dtype=torch.float32) / 25.0
    return module, tokens, seq_len, seconds, torch.ones(seq_len, dtype=torch.bool)


def test_per_frame_refiner_sees_one_frame(body):
    """per_frame: a frame's every output is unchanged when every OTHER frame changes."""
    smplx_out, tokens, blocks, batch = synthetic(body)
    refiner = make_refiner(randomize=True, outputs=LIMB_OUTPUTS, camera_context=True,
                           camera_axes=True, limb_tokens=True, per_frame=True)
    torch.nn.init.normal_(refiner.heads["gravity"][2].weight, std=0.5)
    out = refiner(smplx_out, tokens, blocks, batch, body)
    assert torch.count_nonzero(out["contact"]["logits"]) > 0
    other, other_tokens, _, other_batch = synthetic(body, seed=3)
    keep = torch.zeros(tokens.shape[0], dtype=torch.bool)
    keep[5] = keep[17] = True                                    # one frame of each clip
    mixed_out = {k: torch.where(keep.view(-1, *([1] * (v.dim() - 1))), v, other[k])
                 for k, v in smplx_out.items()}
    mixed_tokens = torch.where(keep[:, None, None], tokens, other_tokens)
    mixed_batch = dict(batch)
    for key in ("cam_from_world", "bbox_center", "bbox_scale"):
        mixed_batch[key] = torch.where(keep.view(-1, *([1] * (batch[key].dim() - 1))),
                                       batch[key], other_batch[key])
    out2 = refiner(mixed_out, mixed_tokens, blocks, mixed_batch, body)
    for key in ("joints_cam", "pelvis_cam", "root_rot", "body_rot", "betas"):
        assert torch.allclose(out["smplx"][key][keep], out2["smplx"][key][keep], atol=1e-4), key
    assert torch.allclose(out["contact"]["logits"][keep], out2["contact"]["logits"][keep], atol=1e-4)
    assert torch.allclose(out["force"]["forces"][keep], out2["force"]["forces"][keep], atol=1e-4)
    assert torch.allclose(out["gravity"]["world"][keep], out2["gravity"]["world"][keep], atol=1e-4)
    assert (out["contact"]["logits"][~keep] - out2["contact"]["logits"][~keep]).abs().max() > 1e-3
    with pytest.raises(ValueError):
        make_refiner(randomize=False, per_frame=True, iterative=True, token=ONE_SIDED_TOKEN)


def test_alternating_module_is_identity_at_init():
    module, tokens, seq_len, seconds, valid = alternating_module()
    assert torch.allclose(module(tokens, seq_len, seconds, valid), tokens, atol=1e-6)


def test_alternating_module_mixes_slots_within_a_frame_and_stays_local():
    """Two layers x 0.5 s = a 1 s horizon in time; the slots of a frame mix immediately."""
    module, tokens, seq_len, seconds, valid = alternating_module()
    for block in module.blocks:
        for projection in (block.proj_temporal, block.proj_frame, block.ffn[3]):
            torch.nn.init.normal_(projection.weight, std=0.05)
    out = module(tokens, seq_len, seconds, valid)
    perturbed = tokens.clone()
    # Slot 0 of the LAST frame (t = 1.16 s). The block LayerNorms its input, so the
    # perturbation must not be a constant shift (which a LayerNorm removes exactly).
    perturbed[-1, 0] += 10.0 * torch.randn(tokens.shape[-1])
    out2 = module(perturbed, seq_len, seconds, valid)
    assert torch.allclose(out[:4], out2[:4], atol=1e-6)            # frames < 0.16 s: untouched
    assert (out[-1, 0] - out2[-1, 0]).abs().max() > 1e-4
    assert (out[-1, 1] - out2[-1, 1]).abs().max() > 1e-4           # ... reaches the other slots
    assert (out[-3] - out2[-3]).abs().max() > 1e-4                 # ... and the nearby frames


# ------------------------------------------------------------------ the gravity force frame

def gravity_frame_refiner(randomize: bool, force_frame: str = "gravity") -> TemporalRefiner:
    """An iterative refiner whose forces are read in the gravity-aligned frame."""
    return make_refiner(randomize=randomize, iterative=True, token=ONE_SIDED_TOKEN,
                        outputs=ALL_OUTPUTS, camera_context=True, camera_axes=True,
                        residual_feedback=True, force_frame=force_frame)


def test_force_frame_gravity_is_identity_at_init(body):
    smplx_out, tokens, blocks, batch = synthetic(body)
    refiner = gravity_frame_refiner(randomize=False)
    out = refiner(smplx_out, tokens, blocks, batch, body)
    assert torch.allclose(out["smplx"]["joints_cam"], smplx_out["joints_cam"], atol=1e-4)
    assert torch.count_nonzero(out["force"]["forces"]) == 0
    assert all(torch.count_nonzero(f) == 0 for f in out["force"]["forces_layers"])
    # The frame is a rotation whose +y column is the clip's down estimate negated.
    frame = out["force"]["frame"]
    eye = torch.eye(3).expand_as(frame)
    assert torch.allclose(frame.transpose(1, 2) @ frame, eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(frame), torch.ones(len(frame)), atol=1e-5)
    assert torch.allclose(frame[:, :, 1], -out["gravity"]["world"], atol=1e-5)
    with pytest.raises(ValueError):                     # the frame needs the gravity estimate
        make_refiner(randomize=False, force_frame="gravity")
    with pytest.raises(ValueError):
        make_refiner(randomize=False, force_frame="world")


def test_force_frame_body_is_unchanged(body):
    """`body` is the default and returns the input body frame itself."""
    smplx_out, tokens, blocks, batch = synthetic(body)
    out = gravity_frame_refiner(randomize=True, force_frame="body")(
        smplx_out, tokens, blocks, batch, body)
    assert torch.equal(out["force"]["frame"], out["smplx"]["root_rot_world_in"])


def test_force_frame_gravity_reads_the_second_component_along_up(body):
    """A head that emits (0, 1, 0) puts one unit of force straight up, against gravity."""
    smplx_out, tokens, blocks, batch = synthetic(body)
    refiner = gravity_frame_refiner(randomize=False)
    torch.nn.init.constant_(refiner.heads["force"][2].bias, 0.0)
    with torch.no_grad():
        refiner.heads["force"][2].bias.view(6, 3)[:, 1] = 1.0
    out = refiner(smplx_out, tokens, blocks, batch, body)
    world = torch.einsum("bij,bkj->bki", out["force"]["frame"], out["force"]["forces"])
    up = -out["gravity"]["world"][:, None].expand_as(world)
    assert torch.allclose(world, up, atol=1e-5)


def test_force_frame_gravity_is_world_frame_independent(body):
    smplx_out, tokens, blocks, batch = synthetic(body)
    refiner = gravity_frame_refiner(randomize=True)
    torch.nn.init.normal_(refiner.heads["force"][2].weight, std=0.2)
    torch.nn.init.normal_(refiner.heads["gravity"][2].weight, std=0.5)
    out = refiner(smplx_out, tokens, blocks, batch, body)
    assert out["force"]["forces"].abs().max() > 0
    world = torch.einsum("bij,bkj->bki", out["force"]["frame"], out["force"]["forces"])

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
    assert torch.allclose(out["force"]["forces"], out2["force"]["forces"], atol=1e-4)
    assert torch.allclose(out["smplx"]["joints_cam"], out2["smplx"]["joints_cam"], atol=1e-4)
    world2 = torch.einsum("bij,bkj->bki", out2["force"]["frame"], out2["force"]["forces"])
    assert torch.allclose(world2, (rot0 @ world.reshape(-1, 3).T).T.reshape(world.shape), atol=1e-4)
    assert torch.allclose(out2["force"]["frame"], rot0 @ out["force"]["frame"], atol=1e-4)


def test_force_frame_gravity_falls_back_when_the_body_lies_along_gravity(body):
    """The heading axis is the body's +y when its +z is (nearly) vertical."""
    refiner = gravity_frame_refiner(randomize=False)
    torch.manual_seed(4)
    rot_wr = roma.random_rotmat(5)
    down = -rot_wr[:, :, 2]                            # gravity along the body's own +z
    frame = refiner._force_frame(down, rot_wr)
    eye = torch.eye(3).expand_as(frame)
    assert torch.allclose(frame.transpose(1, 2) @ frame, eye, atol=1e-5)
    assert torch.allclose(frame[:, :, 1], rot_wr[:, :, 2], atol=1e-5)          # up = -g
    heading = rot_wr[:, :, 1] - (rot_wr[:, :, 1] * rot_wr[:, :, 2]).sum(-1, keepdim=True) * rot_wr[:, :, 2]
    assert torch.allclose(frame[:, :, 2], heading / heading.norm(dim=-1, keepdim=True), atol=1e-5)


# ------------------------------------------------------------------ round 10: stricter checks

def alternating_clips(n_clips: int = 3, seq_len: int = 8, num_slots: int = 3, dim: int = 32,
                      seed: int = 11):
    """Clips of DIFFERENT frame spacings and different invalid frames, for one module.

    A table (cos / sin / mask / slot embedding) repeated over the wrong axis in
    :meth:`~model.rope.CrossModalRopeModule._alternating_tables` is invisible on a single
    clip of evenly spaced, all-valid frames; here every clip has its own spacing, its own
    absolute times and its own holes, and the window bites at a different distance in each.
    """
    torch.manual_seed(seed)
    steps = torch.tensor([[0.04] * seq_len, [0.08] * seq_len,
                          [0.02, 0.02, 0.06, 0.02, 0.10, 0.02, 0.04, 0.02]])
    seconds = torch.cumsum(steps, dim=1) - steps[:, :1] + torch.tensor([[0.0], [0.5], [1.0]])
    valid = torch.ones(n_clips, seq_len, dtype=torch.bool)
    valid[1, 2] = valid[1, 5] = False
    valid[2, 7] = False
    tokens = torch.randn(n_clips * seq_len, num_slots, dim)
    return tokens, seconds, valid


def reference_alternating(module, tokens: Tensor, seconds: Tensor, valid: Tensor) -> Tensor:
    """The alternating stack run clip by clip and slot by slot, tables built per clip."""
    from torch.nn.functional import scaled_dot_product_attention

    from model.rope import frame_keep_mask, rope_cos_sin, rope_rotate
    n_clips, seq_len = seconds.shape
    num_slots, dim = tokens.shape[1], tokens.shape[2]
    heads, head_dim = module.blocks[0].num_heads, module.head_dim
    x = tokens.reshape(n_clips, seq_len, num_slots, dim)
    for block in module.blocks:
        out = torch.empty_like(x)
        for clip in range(n_clips):
            cos, sin = rope_cos_sin(seconds[clip][None] * module.time_scale, head_dim)
            cos, sin = cos[:, None], sin[:, None]
            mask = frame_keep_mask(seconds[clip][None], valid[clip][None], module.window)
            mask = None if mask is None else mask[:, None]
            for slot in range(num_slots):
                row = x[clip, :, slot][None]                                  # [1, T, C]
                normed = block.norm_temporal(row) + module.slot_embed[slot]
                qkv = block.qkv_temporal(normed).reshape(1, seq_len, 3, heads, head_dim)
                query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
                attn = scaled_dot_product_attention(
                    rope_rotate(query, cos, sin), rope_rotate(key, cos, sin), value,
                    attn_mask=mask).transpose(1, 2).reshape(1, seq_len, dim)
                out[clip, :, slot] = (row + block.proj_temporal(attn))[0]
        # The other two branches mix nothing across frames or clips: the block's own.
        frames = out.reshape(n_clips * seq_len, num_slots, dim)
        normed = block.norm_frame(frames) + module.slot_embed[None]
        qkv = block.qkv_frame(normed).reshape(-1, num_slots, 3, heads, head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = scaled_dot_product_attention(query, key, value).transpose(1, 2).reshape(
            -1, num_slots, dim)
        frames = frames + block.proj_frame(attn)
        frames = frames + block.ffn(block.norm_ffn(frames))
        x = frames.reshape(n_clips, seq_len, num_slots, dim)
    return x.reshape(n_clips * seq_len, num_slots, dim)


def test_alternating_tables_are_per_clip_and_per_slot():
    module, _, _, _, _ = alternating_module(dim=32, num_slots=3, num_layers=2)
    torch.manual_seed(5)
    for block in module.blocks:                     # live projections: the attention must act
        for projection in (block.proj_temporal, block.proj_frame, block.ffn[3]):
            torch.nn.init.normal_(projection.weight, std=0.1)
    tokens, seconds, valid = alternating_clips()
    n_clips, seq_len = seconds.shape
    out = module(tokens, seq_len, seconds.reshape(-1), valid.reshape(-1))
    assert torch.allclose(out, reference_alternating(module, tokens, seconds, valid), atol=1e-5)
    # The clips are genuinely different: swapping two of them changes their rows.
    order = torch.tensor([1, 0, 2])
    swapped = module(tokens.reshape(n_clips, seq_len, 3, 32)[order].reshape_as(tokens),
                     seq_len, seconds.reshape(-1), valid.reshape(-1))
    rows = out.reshape(n_clips, seq_len, 3, 32)
    assert (swapped.reshape(n_clips, seq_len, 3, 32)[0] - rows[order][0]).abs().max() > 1e-4


@pytest.mark.parametrize("head, head_grad_scale", [
    ("contact", 1.0), ("force", 1.0), ("contact", 0.3), ("force", 0.3)])
def test_limb_head_losses_alone_reach_the_whole_limb_path(body, head, head_grad_scale):
    """Each limb head on its own trains the limb inputs, the feedback and both attentions."""
    smplx_out, tokens, blocks, batch = synthetic(body)
    refiner = make_refiner(randomize=True, iterative=True, token=ONE_SIDED_TOKEN,
                           outputs=LIMB_OUTPUTS, camera_context=True, camera_axes=True,
                           limb_tokens=True, head_grad_scale=head_grad_scale)
    out = refiner(smplx_out, tokens, blocks, batch, body)
    target = out["contact"]["logits"] if head == "contact" else out["force"]["forces"]
    target.square().sum().backward()
    named = dict(refiner.named_parameters())
    reached = ["limb_input_proj.weight", "limb_feedback_proj.weight", "temporal.slot_embed",
               "limb_output_norm.weight", "input_proj.weight"]
    reached += [f"temporal.blocks.{layer}.{attn}.weight" for layer in range(2)
                for attn in ("qkv_temporal", "qkv_frame", "proj_temporal", "proj_frame")]
    for name in reached:
        grad = named[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    assert all(p.grad is not None and p.grad.abs().sum() > 0
               for p in refiner.heads[head].parameters())


def test_layer_forces_are_transported_into_the_final_gravity_frame(body):
    """Every layer's force means the same WORLD vector after the transport, and the world
    force does not depend on how the world is defined."""
    smplx_out, tokens, blocks, batch = synthetic(body)
    refiner = make_refiner(randomize=True, iterative=True, token=ONE_SIDED_TOKEN,
                           outputs=LIMB_OUTPUTS, camera_context=True, camera_axes=True,
                           limb_tokens=True, force_frame="gravity")
    torch.nn.init.normal_(refiner.heads["force"][2].weight, std=0.2)
    torch.nn.init.normal_(refiner.heads["force"][2].bias, std=0.2)
    torch.nn.init.normal_(refiner.heads["gravity"][2].weight, std=0.5)
    raw_layers: list[Tensor] = []
    refiner.heads["force"].register_forward_hook(
        lambda _m, _in, out: raw_layers.append(out.detach().clone()))
    out = refiner(smplx_out, tokens, blocks, batch, body)

    to_world = lambda frame, force: torch.einsum("bij,bkj->bki", frame, force)
    layers = out["gravity"]["world_layers"]
    assert len(layers) == len(raw_layers) == 2
    assert (layers[0] - layers[1]).abs().max() > 1e-4          # the layers guess differently
    assert out["force"]["forces"].abs().max() > 0
    assert torch.allclose(raw_layers[-1], out["force"]["forces"], atol=1e-6)
    rot_wr_in = out["smplx"]["root_rot_world_in"]
    for layer, (gravity, raw) in enumerate(zip(layers, raw_layers)):
        frame = refiner._force_frame(gravity, rot_wr_in)
        transported = to_world(out["force"]["frame"], out["force"]["forces_layers"][layer])
        assert torch.allclose(to_world(frame, raw), transported, atol=1e-5), layer
    # Layer 0 really is read in another frame: its raw force is not its transported one.
    assert (raw_layers[0] - out["force"]["forces_layers"][0]).abs().max() > 1e-4

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
    for key in ("forces", "forces_layers"):
        first = out["force"][key] if key == "forces" else out["force"][key][0]
        second = out2["force"][key] if key == "forces" else out2["force"][key][0]
        assert torch.allclose(first, second, atol=1e-4), key
        world = to_world(out["force"]["frame"], first)
        world2 = to_world(out2["force"]["frame"], second)
        assert torch.allclose(world2, torch.einsum("ij,bkj->bki", rot0, world), atol=1e-4), key
