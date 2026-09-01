"""The gravity-view linear frame and the roll-out consistency loss.

Both pieces are exercised on synthetic trajectories whose answer is known in
closed form (constant velocity, constant body rate), so the corpus is never read
and the suite stays a CPU unit test:

* :func:`gravity_view_basis` — orthonormality, the gravity column, and the fact
  that channel 1 of a GV-expressed vector IS its downward component (what makes
  the vertical its own channel);
* :class:`MotionRolloutLoss` — exactness of the trapezoid integral, invariance to
  a constant offset in the target path (the gauge freedom the horizon
  formulation is there to remove), the analytic error of a velocity gap, the
  ``detach_head`` gradient contract, and the SO(3) composition.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from contact.config import DEFAULTS as CONFIG_DEFAULTS
from contact.config import _validate_motion_rollout
from contact.data.climbing_corpus import gravity_view_basis
from contact.motion_rollout import MotionRolloutLoss, so3_exp

T_FRAMES = 8
DT = 0.1
ZERO3 = torch.zeros(3)


def _cfg(horizons=(2, 4), angular=True, detach_head=True, mean=None, std=None,
         **weights) -> dict:
    """A minimal config carrying only what the loss reads."""
    loss = {"gt": 0.0, "pose": 0.0, "rot_gt": 0.0, "rot_pose": 0.0,
            "huber_m": 0.1, "huber_rad": 0.1}
    loss.update(weights)
    groups = 4 if angular else 2
    return {
        "motion_rollout": {"enabled": True, "horizons": list(horizons),
                           "detach_head": detach_head, "loss": loss},
        "motion_supervision": {
            "angular": angular,
            "standardize": {
                "mean": [mean or [[0.0] * 3] * groups],
                "std": [std or [[1.0] * 3] * groups]},
        },
    }


def _clip(vel, omega=None, gt_vel=None, gt_omega=None, gt_offset=ZERO3):
    """One clip with a CONSTANT predicted velocity (and optional body rate)."""
    t = torch.arange(T_FRAMES, dtype=torch.float32) * DT
    motion = torch.zeros(T_FRAMES, 1, 12)
    motion[:, 0, 0:3] = vel
    if omega is not None:
        motion[:, 0, 6:9] = omega
    gt_rot = (torch.eye(3).expand(T_FRAMES, 3, 3).contiguous()
              if gt_omega is None else so3_exp(t[:, None] * gt_omega))
    batch = {
        "seq_len": T_FRAMES,
        "motion_lin_rot": torch.eye(3).expand(T_FRAMES, 3, 3).contiguous(),
        "motion_root_pos": gt_offset + t[:, None] * (vel if gt_vel is None else gt_vel),
        "motion_root_valid": torch.ones(T_FRAMES, dtype=torch.bool),
        "motion_rot": gt_rot,
        "cam_from_world": torch.eye(4).expand(T_FRAMES, 4, 4).contiguous(),
        "cam_valid": torch.ones(T_FRAMES, dtype=torch.bool),
        "frame_valid": torch.ones(T_FRAMES, dtype=torch.bool),
        "frame_pos_sec": t,
    }
    return motion, batch


def _out(motion, cam_t=None):
    """Identity extrinsics + zero keypoints make the pose root exactly ``cam_t``."""
    return {
        "motion": {"joint_motion": motion},
        "mhr": {"pred_keypoints_3d": torch.zeros(T_FRAMES, 11, 3),
                "pred_cam_t": torch.zeros(T_FRAMES, 3) if cam_t is None else cam_t,
                "global_rot": torch.zeros(T_FRAMES, 3)},
    }


def _cameras(n: int) -> np.ndarray:
    """`n` cam-from-world matrices with distinct orientations."""
    axes = torch.tensor([[0.3, 0.1, -0.4], [1.0, 0.2, 0.5], [-0.7, 0.9, 0.2],
                         [0.0, 0.0, 0.0], [0.4, -1.1, 0.3]], dtype=torch.float64)
    ext = np.tile(np.eye(4), (n, 1, 1))
    ext[:, :3, :3] = so3_exp(axes[:n]).numpy()
    return ext


def test_gravity_view_basis_is_orthonormal_and_gravity_aligned():
    gravity = np.array([0.2, 0.9, -0.3])
    gravity /= np.linalg.norm(gravity)
    ext = _cameras(5)
    basis = gravity_view_basis(gravity, ext)

    assert np.allclose(basis @ basis.transpose(0, 2, 1), np.eye(3), atol=1e-9)
    assert np.allclose(np.linalg.det(basis), 1.0, atol=1e-9)     # right-handed
    assert np.allclose(basis[:, :, 1], gravity, atol=1e-9)       # column 1 = down
    # Column 0 is the camera's view direction with the gravity component removed.
    view = ext[:, 2, :3]
    horizontal = view - (view @ gravity)[:, None] * gravity
    horizontal /= np.linalg.norm(horizontal, axis=-1, keepdims=True)
    assert np.allclose(basis[:, :, 0], horizontal, atol=1e-9)


def test_gravity_view_channel_one_is_the_downward_component():
    gravity = np.array([0.0, 0.6, 0.8])
    ext = _cameras(5)
    basis = gravity_view_basis(gravity, ext)
    world = np.random.default_rng(0).normal(size=(5, 3))
    gv = np.einsum("nji,nj->ni", basis, world)                   # R^T v

    assert np.allclose(gv[:, 1], world @ gravity, atol=1e-9)
    assert np.allclose(np.linalg.norm(gv, axis=-1),
                       np.linalg.norm(world, axis=-1), atol=1e-9)


def test_rollout_is_exact_for_constant_velocity_and_ignores_a_constant_offset():
    """A matching path scores zero however far away its origin sits."""
    vel = torch.tensor([0.3, -0.2, 0.1])
    motion, batch = _clip(vel, gt_offset=torch.tensor([5.0, -3.0, 2.0]))
    total, parts = MotionRolloutLoss(_cfg(gt=1.0), device="cpu")(_out(motion), batch)

    assert parts["disp_err_m"] == pytest.approx(0.0, abs=1e-6)
    assert float(total) == pytest.approx(0.0, abs=1e-9)
    assert parts["n_rows"] == (T_FRAMES - 2) + (T_FRAMES - 4)


def test_rollout_error_is_the_velocity_gap_times_the_horizon():
    motion, batch = _clip(ZERO3, gt_vel=torch.tensor([0.1, 0.0, 0.0]))
    _, parts = MotionRolloutLoss(_cfg(gt=1.0), device="cpu")(_out(motion), batch)

    counts = {h: T_FRAMES - h for h in (2, 4)}
    expected = (sum(0.1 * h * DT * counts[h] for h in counts)
                / sum(counts.values()))
    assert parts["disp_err_m"] == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize(
    "weights, detach_head, head_grad, pose_grad",
    [({"gt": 1.0}, True, True, False),        # GT term trains the head only
     ({"pose": 1.0}, True, False, True),      # detached: pose path only
     ({"pose": 1.0}, False, True, True)])     # bidirectional
def test_rollout_gradient_paths(weights, detach_head, head_grad, pose_grad):
    motion, batch = _clip(torch.tensor([0.2, 0.0, 0.0]), gt_vel=ZERO3)
    motion = motion.clone().requires_grad_(True)
    cam_t = torch.zeros(T_FRAMES, 3, requires_grad=True)
    loss = MotionRolloutLoss(_cfg(detach_head=detach_head, **weights), device="cpu")

    total, _ = loss(_out(motion, cam_t), batch)
    total.backward()

    assert (float(motion.grad.abs().sum()) > 0) is head_grad
    assert (float(cam_t.grad.abs().sum()) > 0) is pose_grad


def test_rollout_angular_is_exact_for_a_constant_body_rate():
    """exp(w dt) composed H times is exp(w H dt) — the GT relative rotation."""
    omega = torch.tensor([0.0, 0.4, 0.0])
    motion, batch = _clip(ZERO3, omega=omega, gt_omega=omega)
    total, parts = MotionRolloutLoss(
        _cfg(rot_gt=1.0), device="cpu")(_out(motion), batch)

    assert parts["rot_err_deg"] == pytest.approx(0.0, abs=1e-3)
    assert float(total) == pytest.approx(0.0, abs=1e-9)


def test_rollout_angular_error_is_the_rate_gap_times_the_horizon():
    motion, batch = _clip(ZERO3, omega=torch.tensor([0.0, 0.4, 0.0]),
                          gt_omega=torch.tensor([0.0, 0.3, 0.0]))
    _, parts = MotionRolloutLoss(
        _cfg(rot_gt=1.0), device="cpu")(_out(motion), batch)

    counts = {h: T_FRAMES - h for h in (2, 4)}
    expected = (sum(np.rad2deg(0.1 * h * DT) * counts[h] for h in counts)
                / sum(counts.values()))
    assert parts["rot_err_deg"] == pytest.approx(expected, rel=1e-4)


def test_validator_requires_the_gravity_view_frame_and_a_pose_path():
    cfg = copy.deepcopy(CONFIG_DEFAULTS)
    cfg["motion_rollout"]["enabled"] = True
    cfg["motion_supervision"]["enabled"] = True
    cfg["motion_supervision"]["joint_names"] = ["pelvis"]
    cfg["motion_supervision"]["angular"] = True
    cfg["data"]["sequence"]["frames_per_clip"] = 60

    with pytest.raises(ValueError, match="gravity_view"):
        _validate_motion_rollout(cfg)

    cfg["motion_supervision"]["root_convention"] = "gravity_view"
    with pytest.raises(ValueError, match="trainable pose path"):
        _validate_motion_rollout(cfg)

    cfg["train"]["finetune_pose_head"] = True
    _validate_motion_rollout(cfg)

    cfg["motion_supervision"]["angular"] = False
    with pytest.raises(ValueError, match="angular"):
        _validate_motion_rollout(cfg)

    cfg["motion_supervision"]["angular"] = True
    cfg["data"]["sequence"]["frames_per_clip"] = 3
    with pytest.raises(ValueError, match="frames_per_clip"):
        _validate_motion_rollout(cfg)


def test_clips_shorter_than_the_horizon_are_inert_but_keep_their_keys():
    """T = 1 (still images) and short clips must not crash the eval accumulator."""
    motion, batch = _clip(torch.tensor([0.4, 0.0, 0.0]))
    batch["seq_len"] = 1                                   # every row its own clip
    total, parts = MotionRolloutLoss(
        _cfg(gt=1.0, rot_gt=1.0), device="cpu")(_out(motion), batch)

    assert float(total) == 0.0
    assert parts["n_rows"] == 0
    assert parts["disp_err_m"] == 0.0 and parts["rot_err_deg"] == 0.0
    assert all(term["weight_mass"] == 0.0 for term in parts["terms"].values())


def test_gravity_view_follows_the_camera_not_the_world_azimuth():
    """The defining property: the azimuth comes from the CAMERA.

    Re-labelling the world by a yaw about gravity (camera and vectors rotating
    together) must leave the GV expression untouched — that is what makes the
    target independent of the scene's arbitrary world azimuth. Yawing the camera
    alone, with the world fixed, must move it.
    """
    gravity = np.array([0.1, 0.9, -0.2])
    gravity /= np.linalg.norm(gravity)
    ext = _cameras(4)
    world = np.random.default_rng(1).normal(size=(4, 3))

    def express(extrinsics, vectors):
        return np.einsum("nji,nj->ni", gravity_view_basis(gravity, extrinsics), vectors)

    yaw = so3_exp(torch.tensor(0.7 * gravity)).numpy()          # about gravity itself
    yawed = ext.copy()
    yawed[:, :3, :3] = ext[:, :3, :3] @ yaw.T                   # x_cam = R (R_z^T x')

    # Same scene, re-labelled world: gravity is fixed by the yaw, the axes rotate
    # with it, and the expressed vector is unchanged.
    assert np.allclose(
        gravity_view_basis(gravity, yawed), yaw @ gravity_view_basis(gravity, ext),
        atol=1e-9)
    assert np.allclose(express(yawed, world @ yaw.T), express(ext, world), atol=1e-9)
    # Camera yawed, world fixed: the frame moved, so the expression must differ.
    assert not np.allclose(express(yawed, world), express(ext, world), atol=1e-3)


def test_rollout_de_standardizes_and_rotates_a_time_varying_velocity():
    """The three things the constant-velocity tests cannot see.

    A quadratic world path with a NON-identity linear frame and a NON-trivial
    standardize table: the trapezoid is exact on it (its integrand is linear, and
    the central difference of a quadratic is its exact derivative), so the whole
    chain — de-standardize, rotate to world, integrate — must reproduce the GT
    displacement to machine precision. Dropping the de-standardization,
    transposing the frame rotation, or integrating with rectangles all break it.
    """
    accel = torch.tensor([0.2, -0.5, 0.3])
    vel0 = torch.tensor([0.1, 0.0, -0.2])
    mean = [[0.05, -0.10, 0.20]] + [[0.0] * 3] * 3
    std = [[0.5, 2.0, 1.5]] + [[1.0] * 3] * 3
    frame_rot = so3_exp(torch.tensor([0.3, -0.7, 0.2]))          # world-from-frame

    t = torch.arange(T_FRAMES, dtype=torch.float32) * DT
    world_vel = vel0 + accel * t[:, None]                        # exact derivative
    frame_vel = world_vel @ frame_rot                            # R^T v, as rows
    motion = torch.zeros(T_FRAMES, 1, 12)
    motion[:, 0, 0:3] = (frame_vel - torch.tensor(mean[0])) / torch.tensor(std[0])

    _, batch = _clip(ZERO3)
    batch["motion_lin_rot"] = frame_rot.expand(T_FRAMES, 3, 3).contiguous()
    batch["motion_root_pos"] = vel0 * t[:, None] + 0.5 * accel * (t ** 2)[:, None]

    loss = MotionRolloutLoss(_cfg(gt=1.0, mean=mean, std=std), device="cpu")
    total, parts = loss(_out(motion), batch)
    assert parts["disp_err_m"] == pytest.approx(0.0, abs=1e-6)
    assert float(total) == pytest.approx(0.0, abs=1e-9)


def test_rollout_composes_non_commuting_body_rates_in_the_right_order():
    """Body rates multiply on the RIGHT, and the step is a trapezoid.

    The GT rotations are built here by an independent left-to-right loop, so a
    flipped product order or a rectangle step in the loss shows up as a geodesic
    error rather than cancelling on both sides. The rates deliberately do not
    commute.
    """
    steps_t = torch.arange(T_FRAMES, dtype=torch.float64) * DT
    omega = torch.stack([0.9 * torch.sin(3 * steps_t), 0.7 * torch.cos(2 * steps_t),
                         0.5 + steps_t], dim=-1)                 # (T, 3), non-parallel
    increments = so3_exp(0.5 * (omega[:-1] + omega[1:]) * DT)    # trapezoid, (T-1, 3, 3)
    assert not torch.allclose(increments[0] @ increments[1],
                              increments[1] @ increments[0], atol=1e-4)

    gt_rot = [torch.eye(3, dtype=torch.float64)]
    for increment in increments:                                 # independent composition
        gt_rot.append(gt_rot[-1] @ increment)
    motion, batch = _clip(ZERO3, omega=omega.float())
    batch["motion_rot"] = torch.stack(gt_rot).float()

    _, parts = MotionRolloutLoss(_cfg(rot_gt=1.0), device="cpu")(_out(motion), batch)
    assert parts["rot_err_deg"] == pytest.approx(0.0, abs=1e-3)


def test_rollout_masks_exactly_the_windows_that_span_an_invalid_frame():
    """Window validity is per-frame over the whole span, and the GT term also
    needs both endpoints kindyn-covered — which the pose term does not."""
    motion, batch = _clip(torch.tensor([0.2, 0.0, 0.0]), gt_vel=ZERO3)
    batch["frame_valid"] = torch.ones(T_FRAMES, dtype=torch.bool)
    batch["frame_valid"][4] = False                              # kills windows 2, 3, 4
    batch["motion_root_valid"] = torch.ones(T_FRAMES, dtype=torch.bool)
    batch["motion_root_valid"][7] = False                        # kills the GT half of 5

    _, parts = MotionRolloutLoss(
        _cfg(horizons=(2,), gt=1.0, pose=1.0), device="cpu")(_out(motion), batch)

    # T=8, H=2 -> 6 windows [t, t+2]; frame 4 sits in t=2,3,4.
    assert parts["terms"]["pose"]["weight_mass"] == 3.0
    assert parts["terms"]["gt"]["weight_mass"] == 2.0
    assert parts["n_rows"] == 2
