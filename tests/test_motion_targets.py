"""The motion targets themselves: SE3 reference, conventions, slot selection.

Motion v3 changed what the ROOT slot means — from ``R^T`` of the world central
difference (v1/v2) to BetterVideoReconstruction's own body twist. BVR is the
producer of the kindyn fit these labels come from, and it derives velocity and
acceleration in exactly one place
(``tools/smplx_robot/dynamics.py::velocity_acceleration_from_trajectory``) using
BetterRobot's free-flyer ``difference`` = ``se3.log(T_0^-1 T_1)``. This module
pins that equivalence in three independent ways:

1. the numpy float64 log lane against :mod:`better_robot.lie` on random SE3
   elements AND on a real scene's kindyn ``q`` (skipped if better-robot is not
   installed);
2. ``rotated_world`` still reproducing the v1 probe's ``targets.npz`` bit-exactly
   — the eval rows and every published v1/v2 number depend on it;
3. the twist round-tripping to the world acceleration through
   ``a_world = R (a + omega x v)``, which is what pins the Coriolis SIGN (a
   flipped sign lands ~14% off, the correct one ~0.2%).

float64 is deliberate here: it is the production dtype of the target math under
test, so a float32 test would exercise a code path that never runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scipy.ndimage import gaussian_filter1d

from contact.data.climbing_corpus import (
    MOTION_JOINT_NAMES,
    ClimbingCorpusDataset,
    hemisphere_align,
    root_body_twist,
    se3_log_xyzw,
    smooth_root_trajectory,
    so3_log_xyzw,
)
from contact.motion_supervision import to_world_linear

REPO = Path(__file__).resolve().parents[1]
_CORPUS = Path("/data3/rikhat.akizhanov/better/data/ClimbingVideos")
_V1_TARGETS = REPO / "output" / "probe_motion_v1_20260812" / "targets.npz"

requires_corpus = pytest.mark.skipif(
    not (_CORPUS / "scenes" / "scenes.db").is_file(),
    reason="ClimbingVideos corpus not available")
requires_v1_targets = pytest.mark.skipif(
    not _V1_TARGETS.is_file(), reason="v1 motion-probe targets.npz not available")

PELVIS = MOTION_JOINT_NAMES.index("pelvis")
LIMBS = [k for k in range(len(MOTION_JOINT_NAMES)) if k != PELVIS]


def _dataset(scenes, convention="twist", joint_names=None, smooth_sec=0.0,
             frame_stride=1) -> ClimbingCorpusDataset:
    """Default: UNSMOOTHED targets — the definition the v1 reference data uses."""
    return ClimbingCorpusDataset(
        _CORPUS, scenes=list(scenes), split="train", frames_per_clip=7,
        frame_stride=frame_stride, jitter=False, load_motion=True, load_images=False,
        motion_joint_names=joint_names, motion_root_convention=convention,
        motion_target_smooth_sec=smooth_sec)


def _v1_entries(count: int) -> list[dict]:
    manifest = json.loads(str(np.load(_V1_TARGETS, allow_pickle=True)["__manifest__"]))
    return manifest[:count]


def _scene_q(scene: str) -> tuple[np.ndarray, float]:
    """Root configuration ``(N, 7)`` and fps of one scene's first tracked person."""
    shard = f"{scene[0:2]}/{scene[2:4]}"
    kindyn = np.load(
        _CORPUS / "features" / "human_optim" / shard / scene / "kindyn_1.npz",
        allow_pickle=True)
    return (np.asarray(kindyn["q"], np.float32)[0, :, :7],
            float(np.asarray(kindyn["fps"]).item()))


# ------------------------------------------------------- SE3 reference (better-robot)

def _better_robot_lie():
    pytest.importorskip("better_robot", reason="better-robot not installed")
    from better_robot.lie import se3, so3

    return se3, so3


@pytest.mark.parametrize("angle", [1e-9, 1e-5, 1e-2, 1.0, 3.0])
def test_se3_log_matches_better_robot(angle):
    """Our numpy log equals BetterRobot's across the Taylor/full branch split."""
    se3, so3 = _better_robot_lie()
    rng = np.random.default_rng(0)
    axis = rng.normal(size=(256, 3))
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    quat = so3.exp(torch.tensor(axis * angle, dtype=torch.float64)).numpy()
    trans = rng.normal(size=(256, 3)) * 2.0

    reference = se3.log(torch.tensor(
        np.concatenate([trans, quat], -1), dtype=torch.float64)).numpy()
    assert np.abs(se3_log_xyzw(trans, quat) - reference).max() < 1e-12
    assert np.abs(so3_log_xyzw(quat) - reference[:, 3:]).max() < 1e-12


def test_root_body_twist_matches_better_robot_on_a_random_trajectory():
    """The BVR stencil on a synthetic SE3 trajectory, term for term."""
    se3, so3 = _better_robot_lie()
    rng = np.random.default_rng(3)
    n_frames, dt = 40, 1.0 / 29.97
    pos = np.cumsum(rng.normal(size=(n_frames, 3)) * 0.02, axis=0)
    quat = so3.exp(torch.tensor(
        np.cumsum(rng.normal(size=(n_frames, 3)) * 0.05, axis=0),
        dtype=torch.float64)).numpy()
    q_root = np.concatenate([pos, quat], -1)

    vel, acc, omega = root_body_twist(q_root, dt)
    tensor_q = torch.tensor(q_root, dtype=torch.float64)
    diff = se3.log(se3.compose(se3.inverse(tensor_q[:-1]), tensor_q[1:])).numpy()
    vel_ref = np.zeros((n_frames, 6))
    acc_ref = np.zeros((n_frames, 6))
    vel_ref[1:-1] = 0.5 * (diff[:-1] + diff[1:]) / dt
    acc_ref[1:-1] = (diff[1:] - diff[:-1]) / (dt * dt)

    assert np.abs(vel - vel_ref[:, :3]).max() < 1e-12
    assert np.abs(acc - acc_ref[:, :3]).max() < 1e-10
    assert np.abs(omega - vel_ref[:, 3:]).max() < 1e-12


@requires_corpus
@requires_v1_targets
def test_root_body_twist_matches_better_robot_on_a_real_scene():
    """Same equivalence on real kindyn ``q`` (non-unit quats, fractional fps)."""
    se3, _ = _better_robot_lie()
    scene = _v1_entries(1)[0]["scene"]
    q_root, fps = _scene_q(scene)
    dt = 1.0 / fps

    vel, acc, omega = root_body_twist(q_root, dt)
    tensor_q = torch.tensor(q_root, dtype=torch.float64)
    tensor_q[:, 3:] /= tensor_q[:, 3:].norm(dim=-1, keepdim=True)
    diff = se3.log(se3.compose(se3.inverse(tensor_q[:-1]), tensor_q[1:])).numpy()
    vel_ref = np.zeros((len(q_root), 6))
    acc_ref = np.zeros((len(q_root), 6))
    vel_ref[1:-1] = 0.5 * (diff[:-1] + diff[1:]) / dt
    acc_ref[1:-1] = (diff[1:] - diff[:-1]) / (dt * dt)

    assert np.abs(vel - vel_ref[:, :3]).max() < 1e-9
    assert np.abs(acc - acc_ref[:, :3]).max() < 1e-6
    assert np.abs(omega - vel_ref[:, 3:]).max() < 1e-9


# ------------------------------------------------------------- convention behaviour

@requires_corpus
@requires_v1_targets
def test_rotated_world_targets_are_bit_exact_vs_the_v1_probe():
    """``rotated_world`` reproduces the v1 probe's pelvis targets EXACTLY.

    The canonical 7,561 eval rows and every published v1/v2 number are stated on
    these values, so this is an equality, not a tolerance.
    """
    entries = _v1_entries(4)
    dataset = _dataset({e["scene"] for e in entries}, convention="rotated_world")
    v1 = np.load(_V1_TARGETS, allow_pickle=True)
    for entry in entries:
        scene, oid = entry["scene"], entry["object_id"]
        person = list(dataset._scenes[scene]["object_ids"]).index(oid)
        gt = dataset._scenes[scene]["motion_gt"][person]
        assert np.array_equal(gt[:, PELVIS, :3], v1[f"{scene}#{oid}#vel_root"])
        assert np.array_equal(gt[:, PELVIS, 3:], v1[f"{scene}#{oid}#acc_root"])
        assert np.array_equal(
            dataset._scenes[scene]["motion_valid"][person],
            v1[f"{scene}#{oid}#target_valid"])


@requires_corpus
@requires_v1_targets
def test_twist_convention_changes_only_the_pelvis_slot():
    """The six limb slots have no BVR twist counterpart and must stay untouched."""
    entries = _v1_entries(3)
    scenes = {e["scene"] for e in entries}
    rotated = _dataset(scenes, convention="rotated_world")
    twist = _dataset(scenes, convention="twist")
    for scene in scenes:
        gt_rot = rotated._scenes[scene]["motion_gt"]
        gt_tw = twist._scenes[scene]["motion_gt"]
        assert np.array_equal(gt_rot[:, :, LIMBS], gt_tw[:, :, LIMBS])
        valid = twist._scenes[scene]["motion_valid"]
        assert not np.array_equal(gt_rot[:, :, PELVIS][valid],
                                  gt_tw[:, :, PELVIS][valid])


@requires_corpus
@requires_v1_targets
def test_twist_pelvis_round_trips_to_the_world_acceleration():
    """``R (a_twist + omega x v_twist)`` recovers the world central difference.

    This is the sign check on the Coriolis term: with the correct sign the twist
    round-trips onto the raw world acceleration to ~0.2% (the residual is the
    ``V^-1`` correction and the un-adjointed average of two logs); with a flipped
    sign the error is two orders of magnitude larger.
    """
    entry = _v1_entries(1)[0]
    scene, oid = entry["scene"], entry["object_id"]
    dataset = _dataset([scene], convention="twist")
    person = list(dataset._scenes[scene]["object_ids"]).index(oid)
    data = dataset._scenes[scene]
    valid = data["motion_valid"][person]
    gt = torch.from_numpy(data["motion_gt"][person])
    vel_world, acc_world = to_world_linear(
        gt[:, PELVIS:PELVIS + 1, :3], gt[:, PELVIS:PELVIS + 1, 3:],
        torch.from_numpy(data["motion_rot"][person]),
        torch.from_numpy(data["motion_omega"][person]),
        torch.tensor([True]))

    v1 = np.load(_V1_TARGETS, allow_pickle=True)
    for got, want in ((vel_world, v1[f"{scene}#{oid}#vel_world"]),
                      (acc_world, v1[f"{scene}#{oid}#acc_world"])):
        got = got[valid, 0].numpy().astype(np.float64)
        want = want[valid].astype(np.float64)
        relative = (np.linalg.norm(got - want, axis=-1)
                    / np.maximum(np.linalg.norm(want, axis=-1), 1e-9))
        assert np.median(relative) < 0.01, f"median relative error {np.median(relative)}"


# ------------------------------------------------------------- fixed-seconds smoothing

def test_hemisphere_align_removes_double_cover_flips():
    """``q`` and ``-q`` are the same rotation; a sign flip must not survive."""
    rng = np.random.default_rng(11)
    quat = rng.normal(size=(30, 4))
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    flipped = quat.copy()
    flipped[13:] *= -1.0                      # one deliberate double-cover flip
    flipped[21] *= -1.0                       # ... and an isolated one on top

    aligned = hemisphere_align(flipped)
    assert ((aligned[1:] * aligned[:-1]).sum(-1) >= 0).all()
    # Every frame is still the same rotation (up to the global +-1 gauge).
    assert np.allclose(np.abs((aligned * quat).sum(-1)), 1.0)


def test_smoothing_is_blind_to_double_cover_flips():
    """Component-wise quaternion smoothing is only safe after the alignment.

    A sign flip in the middle of a run must not change the smoothed trajectory at
    all; without the alignment it would be filtered as a 180-degree excursion.
    """
    rng = np.random.default_rng(5)
    n_frames = 60
    axis = np.cumsum(rng.normal(size=(n_frames, 3)) * 0.05, axis=0)
    quat = np.concatenate([
        np.sin(np.linalg.norm(axis, axis=-1, keepdims=True) / 2) * axis
        / np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-12),
        np.cos(np.linalg.norm(axis, axis=-1, keepdims=True) / 2)], -1)
    q_root = np.concatenate([np.cumsum(rng.normal(size=(n_frames, 3)) * 0.02, 0), quat], -1)
    valid = np.ones(n_frames, bool)
    flipped = q_root.copy()
    flipped[25:, 3:] *= -1.0

    clean = smooth_root_trajectory(q_root, valid, 3.0)
    dirty = smooth_root_trajectory(flipped, valid, 3.0)
    # Same rotation at every frame (the gauge may differ), same positions.
    assert np.allclose(clean[:, :3], dirty[:, :3])
    assert np.allclose(np.abs((clean[:, 3:] * dirty[:, 3:]).sum(-1)), 1.0, atol=1e-12)
    # Sanity: without the alignment the flip would wreck the filtered quaternion.
    naive = gaussian_filter1d(flipped[:, 3:], sigma=3.0, axis=0, mode="nearest",
                              truncate=4.0)
    naive /= np.linalg.norm(naive, axis=-1, keepdims=True)
    assert np.abs((naive[25] * clean[25, 3:]).sum()) < 0.99


def test_smoothing_stays_inside_valid_runs():
    """Frames across a tracking gap never mix, and invalid frames pass through."""
    n_frames = 40
    q_root = np.zeros((n_frames, 7))
    q_root[:, 6] = 1.0
    q_root[20:, 0] = 100.0                    # huge jump right after the gap
    valid = np.ones(n_frames, bool)
    valid[19] = False

    out = smooth_root_trajectory(q_root, valid, 3.0)
    assert out[19, 0] == 0.0                  # invalid frame returned unchanged
    assert out[18, 0] == 0.0                  # first run never sees the jump
    assert out[20, 0] == pytest.approx(100.0)


def test_smoothing_is_off_at_zero_sigma():
    q_root = np.arange(28, dtype=np.float64).reshape(4, 7)
    assert smooth_root_trajectory(q_root, np.ones(4, bool), 0.0) is q_root


@requires_corpus
@requires_v1_targets
def test_smoothing_shrinks_the_acceleration_target():
    """The fixed-seconds filter removes the sampling-rate wobble, not the motion."""
    entry = _v1_entries(1)[0]
    scene = entry["scene"]
    raw = _dataset([scene], joint_names=["pelvis"], smooth_sec=0.0)
    smooth = _dataset([scene], joint_names=["pelvis"], smooth_sec=0.12)
    valid = raw._scenes[scene]["motion_valid"]
    gt_raw = raw._scenes[scene]["motion_gt"][valid]
    gt_smooth = smooth._scenes[scene]["motion_gt"][valid]
    rms = lambda x: float(np.sqrt((x.astype(np.float64) ** 2).sum(-1).mean()))
    assert rms(gt_smooth[:, 0, 3:]) < 0.7 * rms(gt_raw[:, 0, 3:])
    # Velocity is a far lower-frequency signal: smoothing barely touches it.
    assert rms(gt_smooth[:, 0, :3]) > 0.8 * rms(gt_raw[:, 0, :3])


# ------------------------------------------------------------------- auto stride

@requires_corpus
@requires_v1_targets
def test_auto_stride_holds_the_clip_span_physically_constant():
    entries = _v1_entries(6)
    dataset = _dataset({e["scene"] for e in entries}, frame_stride="auto")
    for scene in dataset._scenes:
        fps = float(dataset._scenes[scene]["fps"])
        stride = dataset.scene_stride(scene)
        assert stride == max(1, round(fps / 25.0))
        span_sec = (dataset.T - 1) * stride / fps
        assert 0.19 <= span_sec <= 0.27, f"{scene}: {span_sec:.3f}s at {fps} fps"


@requires_corpus
@requires_v1_targets
def test_auto_stride_clips_sample_at_that_stride():
    scene = _v1_entries(1)[0]["scene"]
    dataset = _dataset([scene], frame_stride="auto")
    stride = dataset.scene_stride(scene)
    clip = dataset[0]
    positions = [f["frame_position"] for f in clip]
    assert np.all(np.diff(positions) == stride)
    # frame_pos_sec stays REAL seconds, so the temporal encoding is unaffected.
    fps = float(dataset._scenes[scene]["fps"])
    assert clip[-1]["frame_pos_sec"] == pytest.approx((dataset.T - 1) * stride / fps)


def test_bad_frame_stride_rejected():
    with pytest.raises(ValueError, match="frame_stride must be an int or 'auto'"):
        ClimbingCorpusDataset(_CORPUS, scenes=[], split="train", frame_stride="native")


# ------------------------------------------------------------------ slot selection

@requires_corpus
@requires_v1_targets
def test_joint_subset_emits_exactly_the_selected_columns():
    scene = _v1_entries(1)[0]["scene"]
    every = _dataset([scene])
    pelvis_only = _dataset([scene], joint_names=["pelvis"])
    assert pelvis_only.motion_joints == ("pelvis",)
    for key in ("motion_gt", "motion_outlier"):
        full = every._scenes[scene][key]
        assert pelvis_only._scenes[scene][key].shape[2] == 1
        assert np.array_equal(pelvis_only._scenes[scene][key][:, :, 0],
                              full[:, :, PELVIS])
    # Frame dicts follow the same K, and the shared per-frame keys are unchanged.
    frame = pelvis_only[0][0]
    assert frame["motion_gt"].shape == (1, 6)
    assert frame["motion_outlier"].shape == (1,)
    assert frame["motion_omega"].shape == (3,)


@requires_corpus
@pytest.mark.parametrize("names", [["pelvis", "pelvis"], ["nose"]])
def test_bad_motion_joint_names_rejected(names):
    with pytest.raises(ValueError, match="duplicate-free subset"):
        ClimbingCorpusDataset(_CORPUS, scenes=[], split="train", load_motion=True,
                              motion_joint_names=names)


@requires_corpus
def test_bad_root_convention_rejected():
    with pytest.raises(ValueError, match="motion_root_convention must be"):
        ClimbingCorpusDataset(_CORPUS, scenes=[], split="train", load_motion=True,
                              motion_root_convention="world")
