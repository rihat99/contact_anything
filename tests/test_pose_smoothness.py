"""Jerk/snap pose-smoothness penalty: stencil exactness, support, grads, config.

The analytic cases drive the loss through a genuinely MOVING camera — the
predicted keypoints/root are split into ``pred_keypoints_3d + pred_cam_t``
against per-frame extrinsics — so an exact "constant jerk, zero snap" reading
is proof the world lift cancels camera egomotion, not that the camera stood
still. Everything is float32 (the library's end-to-end dtype); the root
channels' internal float64 lift is the library's own choice.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import roma
import torch

from contact.config import load_config
from contact.pose_smoothness import (
    PoseSmoothnessLoss, TERM_NAMES, angular_jerk_snap, position_jerk_snap,
    stencil_support,
)

_FPS = 30.0
_DT = 1.0 / _FPS
_NUM_KP = 70
_HIPS = (9, 10)
#: The MHR head's camera-vs-native axis flip (``contact.motion_consistency._FLIP``).
_FLIP = torch.diag(torch.tensor([1.0, -1.0, -1.0]))

_DELTAS = {"huber_delta_joint_jerk": 80.0, "huber_delta_joint_snap": 6400.0,
           "huber_delta_root_pos_jerk": 50.0, "huber_delta_root_pos_snap": 4600.0,
           "huber_delta_root_rot_jerk": 530.0,
           "huber_delta_root_rot_snap": 82000.0}


def _loss(**weights) -> PoseSmoothnessLoss:
    """Loss with the fixture deltas and the named weights (others zero).

    The deltas are pinned here (calibration-scale, not read from the config) so
    the analytic assertions below stay valid when the shipped values are
    re-measured.
    """
    loss_cfg = {name: 0.0 for name in TERM_NAMES}
    loss_cfg.update(_DELTAS)
    loss_cfg.update(weights)
    return PoseSmoothnessLoss(
        {"pose_smoothness": {
            "enabled": True, "loss": loss_cfg,
            "joint_weights": {"fingers": 0.1, "face": 1.0}}},
        device="cpu")


def _extrinsics(t: int) -> torch.Tensor:
    """A genuinely moving camera: rotation about y + drifting translation."""
    angle = 0.07 * t
    rot = torch.tensor([
        [math.cos(angle), 0.0, math.sin(angle)],
        [0.0, 1.0, 0.0],
        [-math.sin(angle), 0.0, math.cos(angle)]])
    ext = torch.eye(4)
    ext[:3, :3] = rot
    ext[:3, 3] = torch.tensor([0.02 * t, -0.01 * t, 3.0])
    return ext


def _polynomial_world(
    n_clips: int, seq_len: int, jerk: float = 0.0, snap: float = 0.0,
) -> torch.Tensor:
    """``(n_clips, T, 70, 3)`` world keypoints with exactly this jerk/snap.

    ``p(s) = p0 + v s + a s^2 / 2 + jerk s^3 / 6 + snap s^4 / 24`` along every
    coordinate of every joint, so the derivative of order 3 (4) is exactly
    ``jerk`` (``snap``) and the lower orders are non-zero noise the stencils
    must annihilate.
    """
    gen = torch.Generator().manual_seed(0)
    start = torch.randn(n_clips, 1, _NUM_KP, 3, generator=gen)
    vel = torch.randn(n_clips, 1, _NUM_KP, 3, generator=gen)
    acc = torch.randn(n_clips, 1, _NUM_KP, 3, generator=gen)
    s = (torch.arange(seq_len, dtype=torch.float32) * _DT)[None, :, None, None]
    return (start + vel * s + 0.5 * acc * s**2
            + jerk * s**3 / 6.0 + snap * s**4 / 24.0)


def _out_and_batch(world: torch.Tensor, rot_world: torch.Tensor | None = None):
    """Fake forward output whose world-lifted keypoints ARE ``world``.

    :param world: ``(n_clips, T, 70, 3)`` world keypoint positions.
    :param rot_world: ``(n_clips, T, 3, 3)`` desired world-from-root rotations;
        ``None`` = identity. ``global_rot`` is solved for so
        ``predicted_root_world`` reproduces them exactly.
    :returns: ``(out, batch)`` — every prediction tensor requires grad.
    """
    n_clips, seq_len = world.shape[:2]
    n_frames = n_clips * seq_len
    flat = world.reshape(n_frames, _NUM_KP, 3)
    ext = torch.stack([_extrinsics(t) for _ in range(n_clips)
                       for t in range(seq_len)])
    cam = (torch.einsum("bij,bkj->bki", ext[:, :3, :3], flat)
           + ext[:, :3, 3][:, None])
    cam_t = cam.mean(dim=1)                     # an arbitrary keypoint/camera split
    keypoints = cam - cam_t[:, None]
    if rot_world is None:
        rot_world = torch.eye(3).expand(n_frames, 3, 3)
    # R_w = R_ext^T @ flip @ euler("xyz", global_rot)  =>  invert for the euler.
    native = _FLIP.T @ ext[:, :3, :3] @ rot_world.reshape(n_frames, 3, 3)
    global_rot = roma.rotmat_to_euler("xyz", native).to(torch.float32)

    out = {"mhr": {"pred_keypoints_3d": keypoints.clone().requires_grad_(True),
                   "pred_cam_t": cam_t.clone().requires_grad_(True),
                   "global_rot": global_rot.clone().requires_grad_(True)}}
    batch = {
        "seq_len": seq_len,
        "cam_from_world": ext,
        "cam_valid": torch.ones(n_frames, dtype=torch.bool),
        "frame_valid": torch.ones(n_frames, dtype=torch.bool),
        "frame_pos_sec": (torch.arange(n_frames, dtype=torch.float32) % seq_len)
        * _DT,
    }
    return out, batch


def _axis_angle_clip(n_clips: int, seq_len: int, angle_of_s) -> torch.Tensor:
    """``(n_clips, T, 3, 3)`` rotations about a FIXED axis by ``angle_of_s(s)``."""
    axis = torch.tensor([0.3, -0.5, 0.81])
    axis = axis / axis.norm()
    s = torch.arange(seq_len, dtype=torch.float32) * _DT
    rotvec = axis[None, :] * angle_of_s(s)[:, None]                  # (T, 3)
    rot = roma.rotvec_to_rotmat(rotvec)                              # (T, 3, 3)
    return rot[None].expand(n_clips, seq_len, 3, 3).contiguous()


# ------------------------------------------------------------ stencil exactness

def test_position_stencils_are_exact_on_a_cubic():
    """A cubic trajectory has constant jerk and exactly zero snap."""
    world = _polynomial_world(2, 9, jerk=60.0)
    jerk, snap = position_jerk_snap(world, torch.full((2,), _DT))
    assert jerk.shape == (2, 5, _NUM_KP, 3)
    assert torch.allclose(jerk, torch.full_like(jerk, 60.0), rtol=2e-3)
    # Exactly zero in exact arithmetic; the residual is the float32 floor of
    # the 1/dt^4 = 8.1e5 Jacobian on metre-scale coordinates (~3 m/s^4, i.e.
    # 0.05 % of the calibrated 6400 m/s^4 delta).
    assert snap.abs().max() < 5.0


def test_position_stencils_are_exact_on_a_quartic():
    """A quartic trajectory has constant snap; the jerk grows linearly."""
    world = _polynomial_world(1, 9, jerk=0.0, snap=5000.0)
    jerk, snap = position_jerk_snap(world, torch.full((1,), _DT))
    assert torch.allclose(snap, torch.full_like(snap, 5000.0), rtol=2e-3)
    # jerk(s) = snap * s, sampled at the stencil centres s = (r + 2) dt.
    centres = (torch.arange(5, dtype=torch.float32) + 2.0) * _DT
    expected = 5000.0 * centres[None, :, None, None]
    assert torch.allclose(jerk, expected.expand_as(jerk), atol=1.0)


def test_angular_stencils_are_exact_on_a_cubic_angle():
    """Fixed axis, cubic angle: angular jerk is constant, angular snap zero."""
    rot = _axis_angle_clip(2, 9, lambda s: 40.0 * s**3 / 6.0)
    jerk, snap = angular_jerk_snap(rot, torch.full((2,), _DT))
    assert jerk.shape == (2, 5, 3)
    axis = torch.tensor([0.3, -0.5, 0.81])
    axis = axis / axis.norm()
    assert torch.allclose(jerk, 40.0 * axis.expand_as(jerk), rtol=5e-3)
    assert snap.abs().max() < 5.0                       # 0 up to float32 rounding


def test_angular_stencils_are_exact_on_a_quartic_angle():
    rot = _axis_angle_clip(1, 9, lambda s: 3.0e4 * s**4 / 24.0)
    _, snap = angular_jerk_snap(rot, torch.full((1,), _DT))
    axis = torch.tensor([0.3, -0.5, 0.81])
    axis = axis / axis.norm()
    assert torch.allclose(snap, 3.0e4 * axis.expand_as(snap), rtol=5e-3)


def test_stencils_use_real_seconds_not_frame_index():
    """Halving dt at fixed per-frame displacement multiplies jerk by 8."""
    world = _polynomial_world(1, 9, jerk=60.0)
    slow, _ = position_jerk_snap(world, torch.full((1,), _DT))
    fast, _ = position_jerk_snap(world, torch.full((1,), _DT / 2.0))
    assert torch.allclose(fast, slow * 8.0, rtol=1e-4)


# ------------------------------------------------------------ loss integration

def test_cubic_motion_under_a_moving_camera_reads_its_analytic_jerk():
    """The world lift cancels egomotion: the diagnostics see exactly 60/0."""
    out, batch = _out_and_batch(_polynomial_world(2, 9, jerk=60.0))
    total, parts = _loss(joint_jerk=1.0, joint_snap=1.0)(out, batch)
    assert parts["jerk_rms"] == pytest.approx(60.0 * math.sqrt(3.0), rel=2e-3)
    assert parts["snap_rms"] < 5.0
    # |jerk| = 60 < beta = 80 -> quadratic zone: 0.5 * 60^2 / 80 per coordinate,
    # summed over 3 coordinates (the joint mean is weight-normalised).
    assert parts["terms"]["joint_jerk"]["loss"] == pytest.approx(
        3.0 * 0.5 * 60.0**2 / 80.0, rel=3e-3)
    assert parts["terms"]["joint_snap"]["loss"] < 1e-3
    assert float(total.detach()) == pytest.approx(
        parts["terms"]["joint_jerk"]["loss"]
        + parts["terms"]["joint_snap"]["loss"], rel=1e-5)


def test_huber_saturates_far_above_the_delta():
    """A jerk 10x the delta contributes |x| - beta/2, not x^2 — the heavy-tail cap."""
    out, batch = _out_and_batch(_polynomial_world(1, 9, jerk=800.0))
    _, parts = _loss(joint_jerk=1.0)(out, batch)
    assert parts["terms"]["joint_jerk"]["loss"] == pytest.approx(
        3.0 * (800.0 - 0.5 * 80.0), rel=3e-3)


def test_root_position_channel_tracks_the_hip_trajectory():
    """The root terms differentiate the predicted world mean-hips."""
    world = _polynomial_world(2, 9, jerk=40.0)
    out, batch = _out_and_batch(world)
    _, parts = _loss(root_pos_jerk=1.0, root_pos_snap=1.0)(out, batch)
    assert parts["root_pos_jerk_rms"] == pytest.approx(
        40.0 * math.sqrt(3.0), rel=2e-3)
    assert parts["root_pos_snap_rms"] < 5.0
    assert parts["terms"]["root_pos_jerk"]["loss"] == pytest.approx(
        3.0 * 0.5 * 40.0**2 / 50.0, rel=3e-3)
    # The hips are two of the 70 joints, so the joint channel sees the same
    # cubic; nothing here is joint-specific.
    assert set(parts["terms"]) == {"root_pos_jerk", "root_pos_snap"}


def test_root_rotation_channel_reads_the_body_angular_jerk():
    """global_rot solved so the world-from-root rotation is the cubic-angle one."""
    n_clips, seq_len = 2, 9
    rot = _axis_angle_clip(n_clips, seq_len, lambda s: 400.0 * s**3 / 6.0)
    out, batch = _out_and_batch(
        _polynomial_world(n_clips, seq_len), rot_world=rot)
    _, parts = _loss(root_rot_jerk=1.0, root_rot_snap=1.0)(out, batch)
    assert parts["root_rot_jerk_rms"] == pytest.approx(400.0, rel=1e-2)
    assert parts["root_rot_snap_rms"] < 50.0
    assert parts["terms"]["root_rot_jerk"]["loss"] > 0.0


def test_static_pose_costs_exactly_nothing():
    gen = torch.Generator().manual_seed(3)
    world = torch.randn(2, 1, _NUM_KP, 3, generator=gen).expand(2, 9, _NUM_KP, 3)
    out, batch = _out_and_batch(world.contiguous())
    total, parts = _loss(**{name: 1.0 for name in TERM_NAMES})(out, batch)
    # Exactly zero in exact arithmetic; ~1e-4 is the float32 stencil floor
    # inside the quadratic zone (see the cubic-snap test).
    assert float(total.detach()) == pytest.approx(0.0, abs=1e-3)
    assert parts["jerk_rms"] < 1e-2 and parts["snap_rms"] < 5.0


# ------------------------------------------------------------ support / masking

def test_support_needs_five_consecutive_valid_rows():
    ok = torch.tensor([[True] * 9])
    assert stencil_support(ok).shape == (1, 5)
    assert bool(stencil_support(ok).all())
    ok = ok.clone()
    ok[0, 4] = False                     # the centre frame kills every row
    assert int(stencil_support(ok).sum()) == 0


def test_invalid_frames_drop_only_the_rows_that_read_them():
    out, batch = _out_and_batch(_polynomial_world(1, 9, jerk=60.0))
    batch["cam_valid"] = batch["cam_valid"].clone()
    batch["cam_valid"][0] = False        # only row 0 reads frame 0
    _, parts = _loss(joint_jerk=1.0)(out, batch)
    assert parts["terms"]["joint_jerk"]["weight_mass"] == pytest.approx(4.0)
    assert parts["n_rows"] == 4
    out, batch = _out_and_batch(_polynomial_world(1, 9, jerk=60.0))
    batch["frame_valid"] = batch["frame_valid"].clone()
    batch["frame_valid"][4] = False
    _, parts = _loss(joint_jerk=1.0)(out, batch)
    assert parts["terms"]["joint_jerk"]["weight_mass"] == 0.0


def test_clip_boundaries_are_never_penalised():
    """Two clips of T=5 give one row each — the windows never straddle clips."""
    out, batch = _out_and_batch(_polynomial_world(2, 5, jerk=60.0))
    _, parts = _loss(joint_jerk=1.0)(out, batch)
    assert parts["terms"]["joint_jerk"]["weight_mass"] == pytest.approx(2.0)


def test_short_clips_are_inactive():
    """T < 5 (still images or 4-frame clips): the terms exist but carry no mass."""
    out, batch = _out_and_batch(_polynomial_world(2, 9, jerk=60.0))
    batch["seq_len"] = 1
    total, parts = _loss(joint_jerk=1.0, root_rot_snap=1.0)(out, batch)
    assert set(parts["terms"]) == {"joint_jerk", "root_rot_snap"}
    assert all(term["weight_mass"] == 0.0 for term in parts["terms"].values())
    assert float(total.detach()) == 0.0 and torch.isfinite(total)
    assert parts["n_rows"] == 0 and parts["jerk_rms"] == 0.0


def test_ragged_batches_are_inactive():
    """B not divisible by seq_len (never emitted by the collate; DDP contract)."""
    out, batch = _out_and_batch(_polynomial_world(2, 9, jerk=60.0))
    batch["seq_len"] = 7
    _, parts = _loss(joint_jerk=1.0)(out, batch)
    assert parts["terms"]["joint_jerk"]["weight_mass"] == 0.0


# ------------------------------------------------------------------- gradients

@pytest.mark.parametrize(
    "weights",
    [{"joint_jerk": 1.0}, {"joint_snap": 1.0}, {"root_pos_jerk": 1.0},
     {"root_rot_jerk": 1.0}, {"root_rot_snap": 1.0}])
def test_gradients_reach_the_pose_path(weights):
    out, batch = _out_and_batch(
        _polynomial_world(2, 9, jerk=60.0),
        rot_world=_axis_angle_clip(2, 9, lambda s: 400.0 * s**3 / 6.0))
    total, _ = _loss(**weights)(out, batch)
    total.backward()
    grads = {key: out["mhr"][key].grad for key in
             ("pred_keypoints_3d", "pred_cam_t", "global_rot")}
    assert all(grad is not None and torch.isfinite(grad).all()
               for grad in grads.values())
    moved = ("global_rot" if "rot" in next(iter(weights))
             else "pred_keypoints_3d")
    assert float(grads[moved].abs().sum()) > 0.0


def test_inactive_batches_still_touch_every_pose_tensor():
    """Zero mass must not orphan a param (DDP find_unused_parameters=False)."""
    out, batch = _out_and_batch(_polynomial_world(2, 9))
    batch["seq_len"] = 1
    total, _ = _loss(joint_jerk=1.0)(out, batch)
    total.backward()
    for key in ("pred_keypoints_3d", "pred_cam_t", "global_rot"):
        grad = out["mhr"][key].grad
        assert grad is not None and float(grad.abs().sum()) == 0.0


def test_the_gt_reference_never_enters_the_loss():
    """kp3d_world only feeds the diagnostics: the objective ignores it entirely."""
    out, batch = _out_and_batch(_polynomial_world(1, 9, jerk=60.0))
    gt = _polynomial_world(1, 9, jerk=200.0).reshape(9, _NUM_KP, 3)
    batch["kp3d_world"] = gt
    batch["kp_valid"] = torch.ones(9, dtype=torch.bool)
    total, parts = _loss(joint_jerk=1.0)(out, batch)
    assert parts["gt_jerk_rms"] == pytest.approx(200.0 * math.sqrt(3.0), rel=2e-3)
    assert parts["n_gt_rows"] == 5
    without_gt, _ = _loss(joint_jerk=1.0)(*_out_and_batch(
        _polynomial_world(1, 9, jerk=60.0)))
    assert float(total.detach()) == pytest.approx(
        float(without_gt.detach()), rel=1e-6)
    assert not gt.requires_grad


def test_gt_diagnostics_respect_kp_valid():
    out, batch = _out_and_batch(_polynomial_world(1, 9, jerk=60.0))
    batch["kp3d_world"] = _polynomial_world(1, 9, jerk=200.0).reshape(
        9, _NUM_KP, 3)
    batch["kp_valid"] = torch.ones(9, dtype=torch.bool)
    batch["kp_valid"][4] = False
    _, parts = _loss(joint_jerk=1.0)(out, batch)
    assert parts["n_gt_rows"] == 0 and parts["gt_jerk_rms"] == 0.0
    assert parts["n_rows"] == 5          # the objective's own support is untouched


# ------------------------------------------------------------- term contract

def test_only_weighted_terms_are_reported():
    out, batch = _out_and_batch(_polynomial_world(1, 9, jerk=60.0))
    _, parts = _loss(joint_snap=2.0, root_rot_jerk=0.5)(out, batch)
    assert set(parts["terms"]) == {"joint_snap", "root_rot_jerk"}
    for term in parts["terms"].values():
        assert set(term) == {"weighted_numerator_tensor", "weight_mass", "loss"}
        assert term["weighted_numerator_tensor"].shape == ()
        assert term["weight_mass"] == pytest.approx(5.0)
        assert term["loss"] == pytest.approx(
            float(term["weighted_numerator_tensor"].detach()) / 5.0, rel=1e-6)


def test_weight_scales_the_term_linearly():
    out, batch = _out_and_batch(_polynomial_world(1, 9, jerk=60.0))
    _, one = _loss(joint_jerk=1.0)(out, batch)
    _, three = _loss(joint_jerk=3.0)(out, batch)
    assert three["terms"]["joint_jerk"]["loss"] == pytest.approx(
        3.0 * one["terms"]["joint_jerk"]["loss"], rel=1e-6)


def test_finger_weights_shape_the_joint_mean():
    """Fingers at 0.1 shrink their share of the joint mean (weights are means)."""
    world = _polynomial_world(1, 9)
    world = world.clone()
    world[:, :, 21:41] += (torch.arange(9, dtype=torch.float32)[None, :, None, None]
                           * _DT)**3 * 1000.0 / 6.0
    out, batch = _out_and_batch(world)
    _, parts = _loss(joint_jerk=1.0)(out, batch)
    fingered = parts["terms"]["joint_jerk"]["loss"]
    loss_obj = _loss(joint_jerk=1.0)
    loss_obj.joint_w = torch.ones_like(loss_obj.joint_w)
    loss_obj.joint_w_sum = float(loss_obj.joint_w.sum())
    _, flat = loss_obj(*_out_and_batch(world))
    assert flat["terms"]["joint_jerk"]["loss"] > fingered


# ------------------------------------------------------------ config validation

_ENABLED = """
base: configs/base.yaml
model:
  pose_temporal: {enabled: true}
contact:
  targets:
    vertex: {enabled: false}
    joint: {enabled: false}
data:
  datasets:
    - {name: climbing_corpus, config: configs/datasets/climbing_corpus_pose.yaml}
  sequence: {frames_per_clip: 8}
pose_supervision: {enabled: true}
pose_smoothness:
  enabled: true
  loss: {joint_jerk: 1.0e-5}
"""


def _write(tmp_path, text: str):
    path = tmp_path / "run.yaml"
    path.write_text(text)
    return path


def test_defaults_are_off_and_documented():
    cfg = load_config("configs/base.yaml")
    ps = cfg["pose_smoothness"]
    assert ps["enabled"] is False
    assert all(ps["loss"][name] == 0.0 for name in TERM_NAMES)
    assert all(ps["loss"][f"huber_delta_{name}"] > 0.0 for name in TERM_NAMES)


def test_valid_experiment_config_loads(tmp_path):
    cfg = load_config(_write(tmp_path, _ENABLED))
    assert cfg["pose_smoothness"]["loss"]["joint_jerk"] == pytest.approx(1e-5)


@pytest.mark.parametrize("override, message", [
    ("loss: {joint_jerk: -1.0}", "finite and >= 0"),
    ("loss: {joint_snap: .nan}", "finite and >= 0"),
    ("loss: {huber_delta_joint_jerk: 0.0}", "finite and positive"),
    ("loss: {huber_delta_root_rot_snap: -3.0}", "finite and positive"),
    ("joint_weights: {fingers: -0.5}", "finite and >= 0"),
])
def test_bad_values_are_rejected(tmp_path, override, message):
    text = _ENABLED.replace("  loss: {joint_jerk: 1.0e-5}",
                            f"  loss: {{joint_jerk: 1.0e-5}}\n  {override}")
    with pytest.raises(ValueError, match=message):
        load_config(_write(tmp_path, text))


def test_unknown_key_is_rejected(tmp_path):
    text = _ENABLED + "  jerk_weight: 1.0\n"
    with pytest.raises(ValueError, match="jerk_weight"):
        load_config(_write(tmp_path, text))


def test_all_zero_weights_are_rejected(tmp_path):
    text = _ENABLED.replace("loss: {joint_jerk: 1.0e-5}", "loss: {joint_jerk: 0.0}")
    with pytest.raises(ValueError, match="does nothing"):
        load_config(_write(tmp_path, text))


def test_frozen_pose_path_is_rejected(tmp_path):
    """A contact-only corpus build has no pose gradient path — reject it."""
    text = """
base: configs/base.yaml
contact:
  targets:
    vertex: {enabled: false}
    joint: {enabled: true}
data:
  datasets:
    - {name: climbing_corpus, config: configs/datasets/climbing_corpus_pose.yaml}
  sequence: {frames_per_clip: 8}
pose_smoothness:
  enabled: true
  loss: {joint_jerk: 1.0e-5}
"""
    with pytest.raises(ValueError, match="trainable pose path"):
        load_config(_write(tmp_path, text))


def test_short_clips_are_rejected(tmp_path):
    text = _ENABLED.replace("frames_per_clip: 8", "frames_per_clip: 4")
    with pytest.raises(ValueError, match="frames_per_clip >= 5"):
        load_config(_write(tmp_path, text))


def test_still_image_only_dataset_is_rejected(tmp_path):
    text = _ENABLED.replace(
        "    - {name: climbing_corpus, config: configs/datasets/climbing_corpus_pose.yaml}",
        "    - {name: damon, config: configs/datasets/damon.yaml}")
    with pytest.raises(ValueError, match="climbing_corpus"):
        load_config(_write(tmp_path, text))


def test_jerksnap_experiment_config_validates():
    # Both arms were retired to the trash 2026-08-30 (pre-v3 cleanup); the
    # verbatim copies in tests/fixtures/ keep this A/B assertion alive.
    cfg = load_config("tests/fixtures/rope_t60_mhrsup_temponly_jerksnap.yaml")
    ps = cfg["pose_smoothness"]
    assert ps["enabled"] is True
    assert all(ps["loss"][name] > 0.0 for name in TERM_NAMES)
    baseline = load_config("tests/fixtures/rope_t60_mhrsup_temponly.yaml")
    assert baseline["pose_smoothness"]["enabled"] is False
    # The A/B differs ONLY by the new section (and the run name).
    for section in ("model", "keypoint_supervision", "pose_supervision",
                    "optim", "data", "train", "loss"):
        assert cfg[section] == baseline[section], section


# --------------------------------------------------------------- GPU semantics

_CKPT = load_config("configs/base.yaml")["model"]["checkpoint_path"]
_CORPUS = Path("/data3/rikhat.akizhanov/better/data/ClimbingVideos")
_SCENE = "MuVpoovQl2M_0001"
_CONVERTED = (_CORPUS / "features" / "human_optim" / "Mu" / "Vp" / _SCENE
              / "mhr_sup_1.npz")


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint missing")
@pytest.mark.skipif(not _CONVERTED.is_file(), reason="mhr_sup_1.npz missing")
def test_gpu_gradients_reach_pose_temporal_and_no_frozen_param():
    """Real checkpoint + a real corpus clip: the penalty trains the temporal
    block alone. Gammas are randomised first — at their zero init the gated
    subtree's inner weights have an exactly-zero (not absent) gradient."""
    from contact.data.climbing_corpus import ClimbingCorpusDataset
    from contact.data.collate import batch_to_device, make_collate
    from contact.engine import forward_model
    from contact.model import build_model
    from contact.targets import TargetSpec

    torch.manual_seed(0)
    cfg = load_config("tests/fixtures/pose_temporal.yaml")
    cfg["pose_smoothness"]["enabled"] = True
    for name in TERM_NAMES:
        cfg["pose_smoothness"]["loss"][name] = 1.0
    model, trainable = build_model(cfg, "cuda")
    assert trainable and all("pose_temporal" in name for name in trainable)
    generator = torch.Generator(device="cuda").manual_seed(7)
    with torch.no_grad():
        for name, param in model.pose_temporal.named_parameters():
            if "gamma" in name:
                param.copy_(0.1 * torch.randn(
                    param.shape, generator=generator, device="cuda"))

    dataset = ClimbingCorpusDataset(
        _CORPUS, scenes=[_SCENE], split="train", frames_per_clip=7,
        frame_stride=1, jitter=False, load_pose=True, load_keypoints=True)
    collate = make_collate(
        tuple(model.cfg.MODEL.IMAGE_SIZE), TargetSpec.from_config(cfg))
    batch = batch_to_device(collate([dataset[0], dataset[1]]), "cuda")

    total, parts = PoseSmoothnessLoss(cfg, device="cuda")(
        forward_model(model, batch), batch)
    assert torch.isfinite(total) and float(total.detach()) > 0.0
    assert set(parts["terms"]) == set(TERM_NAMES)
    assert all(term["weight_mass"] > 0 for term in parts["terms"].values())
    assert parts["jerk_rms"] > 0.0 and parts["gt_jerk_rms"] > 0.0
    total.backward()
    n_nonzero = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert "pose_temporal" in name
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name
            n_nonzero += int(float(param.grad.abs().sum()) > 0.0)
        elif param.grad is not None:
            raise AssertionError(f"frozen param {name} received a gradient")
    assert n_nonzero > 0
