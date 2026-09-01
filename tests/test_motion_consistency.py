"""Pose→motion consistency loss: config matrix, lie/twist parity, masking, grads.

Fast CPU coverage: the config accept/reject matrix, the torch lie helpers
against the loader's float64 numpy target derivation (the SAME scheme must
produce the SAME numbers), the world-lifting composition, and the loss's
masking/mass/gradient-routing semantics on a fabricated batch. ``-m slow``
proves the full-model gradient routing (pose path AND motion head, never the
frozen base) with the real checkpoint on corpus data.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from contact.config import load_config
from contact.data.climbing_corpus import (
    quat_xyzw_to_matrix, root_body_twist,
)
from contact.motion_consistency import (
    MotionConsistencyLoss, clip_body_twist, predicted_root_world,
    quat_xyzw_from_matrix, se3_log, so3_log_xyzw,
)

REPO = Path(__file__).resolve().parents[1]
_CKPT = load_config(REPO / "configs" / "base.yaml")["model"]["checkpoint_path"]


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "run.yaml"
    p.write_text(text)
    return p


# ---------------------------------------------------------------- config matrix

def test_defaults_disabled():
    cfg = load_config(REPO / "configs" / "base.yaml")
    mc = cfg["motion_consistency"]
    assert mc["enabled"] is False
    assert mc["angular"] is True
    assert mc["loss"] == {
        "gt": 0.005, "head": 0.0025, "huber_delta": 1.0,
        "pos": 0.0, "pos_huber_m": 0.1, "rot": 0.0, "rot_huber_rad": 0.1}
    assert mc["hip_offset_root"] == pytest.approx([-0.009, -0.060, -0.065])
    assert mc["detach_head"] is True


def test_allmod_rope_fixture_validates():
    # The shipped config was retired to the trash 2026-08-30 (pre-v3 cleanup);
    # tests/fixtures/ keeps a verbatim copy so this coverage survives.
    cfg = load_config(REPO / "tests" / "fixtures" / "allmod_rope_t60.yaml")
    assert cfg["motion_consistency"]["enabled"] is True
    assert cfg["model"]["extra_token_attention"] == "mutual"
    # The single all-modality RoPE block is the only temporal path, and it
    # writes the pose token (so pose_supervision must be on).
    xm = cfg["model"]["cross_modal_temporal"]
    assert xm["enabled"] is True
    assert xm["modalities"] == ["pose", "contact", "force", "motion"]
    assert cfg["model"]["pose_temporal"]["enabled"] is False
    assert cfg["pose_supervision"]["enabled"] is True


def test_requires_motion_supervision(tmp_path):
    with pytest.raises(ValueError, match="requires motion_supervision"):
        load_config(_write(tmp_path, """
base: tests/fixtures/pose_temporal.yaml
motion_consistency: {enabled: true}
"""))


def test_requires_a_trainable_pose_path(tmp_path):
    with pytest.raises(ValueError, match="trainable pose path"):
        load_config(_write(tmp_path, """
base: tests/fixtures/motion_pelvis12_angw05.yaml
motion_consistency: {enabled: true}
"""))


def test_requires_pelvis_slot(tmp_path):
    with pytest.raises(ValueError, match="'pelvis' slot"):
        load_config(_write(tmp_path, """
base: tests/fixtures/motion_seven_tokens.yaml
model:
  motion_head: {motion_keypoint_indices: [62, 41]}
  pose_temporal: {enabled: true}
motion_supervision:
  joint_names: [left_wrist, right_wrist]
  standardize:
    mean: [[[0,0,0],[0,0,0]], [[0,0,0],[0,0,0]]]
    std: [[[1,1,1],[1,1,1]], [[1,1,1],[1,1,1]]]
pose_supervision: {enabled: true}
motion_consistency: {enabled: true}
"""))


def test_requires_three_frame_clips(tmp_path):
    with pytest.raises(ValueError, match="frames_per_clip >= 3"):
        load_config(_write(tmp_path, """
base: tests/fixtures/motion_pelvis12_angw05.yaml
model:
  pose_temporal: {enabled: true}
data:
  sequence: {frames_per_clip: 1, frame_stride: 1, jitter: true, target_frame: all}
pose_supervision: {enabled: true}
motion_consistency: {enabled: true}
"""))


def test_zero_weights_rejected(tmp_path):
    with pytest.raises(ValueError, match="does nothing"):
        load_config(_write(tmp_path, """
base: tests/fixtures/motion_pelvis12_angw05.yaml
model:
  pose_temporal: {enabled: true}
pose_supervision: {enabled: true}
motion_consistency:
  enabled: true
  loss: {gt: 0.0, head: 0.0, pos: 0.0, rot: 0.0}
"""))


def test_negative_pos_weight_rejected(tmp_path):
    with pytest.raises(ValueError, match="loss.pos"):
        load_config(_write(tmp_path, """
base: tests/fixtures/motion_pelvis12_angw05.yaml
model:
  pose_temporal: {enabled: true}
pose_supervision: {enabled: true}
motion_consistency: {enabled: true, loss: {pos: -1.0}}
"""))


def test_bad_hip_offset_rejected(tmp_path):
    with pytest.raises(ValueError, match="hip_offset_root"):
        load_config(_write(tmp_path, """
base: tests/fixtures/motion_pelvis12_angw05.yaml
model:
  pose_temporal: {enabled: true}
pose_supervision: {enabled: true}
motion_consistency: {enabled: true, hip_offset_root: [0.0, 0.0]}
"""))


def test_hip_offset_default_follows_root_source(tmp_path):
    """The offset is a property of the GT rig, so its default follows
    ``motion_supervision.root_source``: the kindyn pelvis sits ~9 cm from the
    mean-hips, the MHR root IS the mean-hips. An explicit value still wins."""
    kindyn = load_config(REPO / "configs" / "base.yaml")
    assert kindyn["motion_supervision"]["root_source"] == "kindyn"
    assert kindyn["motion_consistency"]["hip_offset_root"] == pytest.approx(
        [-0.009, -0.060, -0.065])

    mhr = load_config(_write(tmp_path, """
base: configs/base.yaml
motion_supervision: {root_source: mhr}
"""))
    assert mhr["motion_consistency"]["hip_offset_root"] == [0.0, 0.0, 0.0]

    explicit = load_config(_write(tmp_path, """
base: configs/base.yaml
motion_supervision: {root_source: mhr}
motion_consistency: {hip_offset_root: [0.02, -0.06, -0.07]}
"""))
    assert explicit["motion_consistency"]["hip_offset_root"] == pytest.approx(
        [0.02, -0.06, -0.07])


def test_no_arch_signature_key(tmp_path):
    # The loss adds no parameters: enabling it must not change the architecture
    # signature (checkpoints stay interchangeable across the toggle).
    from contact import checkpoint as ckpt_io

    sig = ckpt_io._arch_signature(
        load_config(REPO / "tests" / "fixtures" / "allmod_rope_t60.yaml"))
    assert "motion_consistency" not in sig


# ---------------------------------------------------------------- lie parity

def _random_trajectory(n: int, t: int, seed: int = 0):
    """Smooth-ish random world root trajectory (pos, quat_xyzw)."""
    rng = np.random.RandomState(seed)
    pos = np.cumsum(rng.randn(n, t, 3) * 0.05, axis=1)
    axis = rng.randn(n, t, 3) * 0.15
    angle = np.linalg.norm(axis, axis=-1, keepdims=True)
    quat = np.concatenate([
        axis / np.clip(angle, 1e-9, None) * np.sin(angle / 2.0),
        np.cos(angle / 2.0)], axis=-1)
    return pos, quat


def test_quat_from_matrix_roundtrip():
    _, quat = _random_trajectory(4, 5, seed=1)
    rot = quat_xyzw_to_matrix(quat.reshape(-1, 4)).astype(np.float64)
    back = quat_xyzw_from_matrix(torch.from_numpy(rot))
    ref = torch.from_numpy(quat.reshape(-1, 4))
    # A quaternion and its negation are the same rotation.
    sign = torch.sign((back * ref).sum(-1, keepdim=True))
    assert torch.allclose(back * sign, ref, atol=1e-6)


def test_so3_se3_log_match_the_loader():
    from contact.data.climbing_corpus import se3_log_xyzw, so3_log_xyzw as np_so3

    _, quat = _random_trajectory(3, 4, seed=2)
    quat = quat.reshape(-1, 4)
    trans = np.random.RandomState(3).randn(quat.shape[0], 3)
    np.testing.assert_allclose(
        so3_log_xyzw(torch.from_numpy(quat)).numpy(), np_so3(quat), atol=1e-9)
    np.testing.assert_allclose(
        se3_log(torch.from_numpy(trans), torch.from_numpy(quat)).numpy(),
        se3_log_xyzw(trans, quat), atol=1e-9)


def test_clip_body_twist_matches_the_target_derivation():
    """The torch matrix-path twist == the loader's float64 quat-path twist."""
    pos, quat = _random_trajectory(2, 9, seed=4)
    q_root = np.concatenate([pos, quat], axis=-1)
    dt = 1.0 / 30.0
    vel_np, acc_np, omega_np, ang_acc_np = root_body_twist(q_root, dt)

    rot = quat_xyzw_to_matrix(quat).astype(np.float64)
    vel, acc, omega, ang_acc = clip_body_twist(
        torch.from_numpy(pos), torch.from_numpy(rot),
        torch.full((2,), dt, dtype=torch.float64))
    # atol 1e-6: quat_xyzw_to_matrix emits float32 matrices, so the two paths
    # share only float32 precision on the input rotations.
    for torch_val, np_val in ((vel, vel_np), (acc, acc_np),
                              (omega, omega_np), (ang_acc, ang_acc_np)):
        np.testing.assert_allclose(torch_val.numpy(), np_val, rtol=1e-5, atol=1e-6)


def test_predicted_root_world_identity_extrinsics():
    torch.manual_seed(0)
    b = 6
    kp = torch.randn(b, 70, 3, dtype=torch.float64)
    cam_t = torch.randn(b, 3, dtype=torch.float64)
    euler = torch.randn(b, 3, dtype=torch.float64) * 0.3
    out = {"pred_keypoints_3d": kp, "pred_cam_t": cam_t, "global_rot": euler}
    pos, rot = predicted_root_world(out, torch.eye(4).expand(b, 4, 4))
    assert torch.allclose(pos, kp[:, [9, 10]].mean(1) + cam_t, atol=1e-9)
    # World-from-root carries the native flip.
    import roma
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float64))
    assert torch.allclose(
        rot, flip @ roma.euler_to_rotmat("xyz", euler), atol=1e-9)


# ---------------------------------------------------------------- loss semantics

def _toy_cfg() -> dict:
    return {
        "motion_consistency": {
            "enabled": True,
            "angular": True,
            "hip_offset_root": [0.0, 0.0, 0.0],
            "loss": {"gt": 1.0, "head": 0.5, "huber_delta": 1.0,
                     "pos": 1.0, "pos_huber_m": 0.1,
                     "rot": 1.0, "rot_huber_rad": 0.1},
        },
        "motion_supervision": {
            "enabled": True,
            "joint_names": ["pelvis"],
            "angular": True,
            "loss": {"vel": 1.0, "acc": 1.0, "ang_vel": 0.5, "ang_acc": 0.5,
                     "huber_delta": 1.0, "outlier_acc_ms2": 50.0},
            "standardize": {
                "mean": [[[0.0] * 3] * 4],
                "std": [[[1.0] * 3] * 4],
            },
        },
    }


def _toy_batch(n_clips: int = 2, t: int = 5, fps: float = 30.0) -> dict:
    b = n_clips * t
    # motion_rot defaults to the camera-vs-native flip: with identity
    # extrinsics and global_rot == 0 the predicted world-from-root IS the
    # flip, so the default toy rot residual only carries the random pose.
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    return {
        "seq_len": t,
        "cam_from_world": torch.eye(4).expand(b, 4, 4).clone(),
        "cam_valid": torch.ones(b, dtype=torch.bool),
        "frame_valid": torch.ones(b, dtype=torch.bool),
        "frame_pos_sec": (torch.arange(b, dtype=torch.float32) % t) / fps,
        "motion_gt": torch.zeros(b, 1, 12),
        "motion_valid": torch.ones(b, dtype=torch.bool),
        "motion_outlier": torch.zeros(b, 1, dtype=torch.bool),
        "motion_rot": flip.expand(b, 3, 3).clone(),
        "motion_root_pos": torch.zeros(b, 3),
        "motion_root_valid": torch.ones(b, dtype=torch.bool),
    }


def _toy_out(n_clips: int = 2, t: int = 5, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    b = n_clips * t
    return {
        "mhr": {
            "pred_keypoints_3d": (torch.randn(b, 70, 3) * 0.1).requires_grad_(),
            "pred_cam_t": (torch.randn(b, 3) * 0.1 + torch.tensor([0.0, 0.0, 3.0])
                           ).requires_grad_(),
            "global_rot": (torch.randn(b, 3) * 0.1).requires_grad_(),
        },
        "motion": {"joint_motion": torch.randn(b, 1, 12).requires_grad_()},
    }


def test_gradients_reach_pose_path_only():
    """Every gradient path lands on the pose side; the motion head is DETACHED."""
    loss_fn = MotionConsistencyLoss(_toy_cfg(), device="cpu")
    out, batch = _toy_out(), _toy_batch()
    total, parts = loss_fn(out, batch)
    assert parts["terms"].keys() == {"gt", "head", "pos", "rot"}
    total.backward()
    for key in ("pred_keypoints_3d", "pred_cam_t", "global_rot"):
        grad = out["mhr"][key].grad
        assert grad is not None and float(grad.abs().sum()) > 0, key
    # The head term's target is detached: no gradient may reach the motion head.
    assert out["motion"]["joint_motion"].grad is None


def test_detach_head_false_reaches_motion_head():
    """``detach_head: false`` makes the head term bidirectional."""
    cfg = _toy_cfg()
    cfg["motion_consistency"]["detach_head"] = False
    loss_fn = MotionConsistencyLoss(cfg, device="cpu")
    out, batch = _toy_out(), _toy_batch()
    total, _ = loss_fn(out, batch)
    total.backward()
    grad = out["motion"]["joint_motion"].grad
    assert grad is not None and float(grad.abs().sum()) > 0
    for key in ("pred_keypoints_3d", "pred_cam_t", "global_rot"):
        assert out["mhr"][key].grad is not None, key


def test_boundary_rows_are_never_supervised():
    loss_fn = MotionConsistencyLoss(_toy_cfg(), device="cpu")
    out, batch = _toy_out(), _toy_batch(n_clips=2, t=5)
    _, parts = loss_fn(out, batch)
    # 5-frame clips: rows 1..3 have stencil support -> 3 rows x 2 clips.
    assert parts["terms"]["head"]["weight_mass"] == 6.0
    assert parts["terms"]["gt"]["weight_mass"] == 6.0
    # The absolute anchors need no stencil: every row supervises.
    assert parts["terms"]["pos"]["weight_mass"] == 10.0
    assert parts["terms"]["rot"]["weight_mass"] == 10.0


def test_invalid_camera_kills_the_window():
    loss_fn = MotionConsistencyLoss(_toy_cfg(), device="cpu")
    out, batch = _toy_out(n_clips=1, t=5), _toy_batch(n_clips=1, t=5)
    batch["cam_valid"][2] = False        # center frame -> rows 1, 2, 3 lose support
    _, parts = loss_fn(out, batch)
    assert parts["terms"]["head"]["weight_mass"] == 0.0
    # The anchors only lose the one extrinsics-less row.
    assert parts["terms"]["pos"]["weight_mass"] == 4.0


def test_outlier_bit_gates_gt_but_not_head():
    loss_fn = MotionConsistencyLoss(_toy_cfg(), device="cpu")
    out, batch = _toy_out(n_clips=1, t=5), _toy_batch(n_clips=1, t=5)
    batch["motion_outlier"][:, 0] = True
    _, parts = loss_fn(out, batch, exclude_outliers=True)
    assert parts["terms"]["gt"]["weight_mass"] == 0.0
    assert parts["terms"]["head"]["weight_mass"] == 3.0
    _, parts_eval = loss_fn(out, batch, exclude_outliers=False)
    assert parts_eval["terms"]["gt"]["weight_mass"] == 3.0


def test_short_clips_are_inactive():
    loss_fn = MotionConsistencyLoss(_toy_cfg(), device="cpu")
    out, batch = _toy_out(n_clips=2, t=1), _toy_batch(n_clips=2, t=1)
    total, parts = loss_fn(out, batch)
    assert all(t["weight_mass"] == 0.0 for t in parts["terms"].values())
    total.backward()                     # zero-touch keeps the POSE graph alive
    assert out["mhr"]["pred_cam_t"].grad is not None
    assert out["motion"]["joint_motion"].grad is None    # detached head


def test_rmse_diagnostics_are_physical_units():
    """vel/acc RMSE stay raw m/s / m/s^2 under a non-trivial standardize table
    (regression: the diagnostic once re-applied std/mean to the already
    physical pose twist)."""
    cfg = _toy_cfg()
    cfg["motion_supervision"]["standardize"] = {
        "mean": [[[0.5] * 3] * 4], "std": [[[2.0] * 3] * 4]}
    loss_fn = MotionConsistencyLoss(cfg, device="cpu")
    n_clips, t = 2, 5
    out, batch = _toy_out(n_clips=n_clips, t=t), _toy_batch(n_clips=n_clips, t=t)
    _, parts = loss_fn(out, batch)

    pos_w, rot_w = predicted_root_world(out["mhr"], batch["cam_from_world"])
    pos_sec = batch["frame_pos_sec"].to(torch.float64).reshape(n_clips, t)
    dt = (pos_sec[:, 1:] - pos_sec[:, :-1]).mean(dim=1).clamp(min=1e-6)
    vel, acc, _, _ = clip_body_twist(
        pos_w.reshape(n_clips, t, 3), rot_w.reshape(n_clips, t, 3, 3), dt)
    # motion_gt is zero, every interior row supervised -> RMSE = interior RMS.
    for key, arr in (("vel_rmse", vel), ("acc_rmse", acc)):
        expected = float(arr[:, 1:-1].square().sum(-1).mean().sqrt())
        assert parts[key] == pytest.approx(expected, rel=1e-4), key


def test_still_pose_gives_zero_twist():
    """A frozen-in-place trajectory must produce exactly zero vel/acc."""
    loss_fn = MotionConsistencyLoss(_toy_cfg(), device="cpu")
    out, batch = _toy_out(n_clips=1, t=5), _toy_batch(n_clips=1, t=5)
    with torch.no_grad():
        for key in ("pred_keypoints_3d", "pred_cam_t", "global_rot"):
            out["mhr"][key].copy_(out["mhr"][key][0:1].expand_as(out["mhr"][key]))
    _, parts = loss_fn(out, batch)
    # GT is zeros and the pose twist is zeros -> the gt term is exactly 0.
    assert parts["terms"]["gt"]["loss"] == pytest.approx(0.0, abs=1e-12)


def _predicted_pos(out: dict) -> torch.Tensor:
    """The loss's own pelvis-camera point under identity extrinsics."""
    with torch.no_grad():
        return (out["mhr"]["pred_keypoints_3d"][:, [9, 10]].mean(dim=1)
                + out["mhr"]["pred_cam_t"])


def test_pos_term_anchors_absolute_position():
    loss_fn = MotionConsistencyLoss(_toy_cfg(), device="cpu")
    out, batch = _toy_out(n_clips=1, t=5), _toy_batch(n_clips=1, t=5)
    batch["motion_root_pos"] = _predicted_pos(out)
    _, parts = loss_fn(out, batch)
    assert parts["terms"]["pos"]["loss"] == pytest.approx(0.0, abs=1e-6)
    # A 1 m x-shift of the GT root: |e| = 1 > beta -> smooth-L1 = 1 - beta/2.
    batch["motion_root_pos"][:, 0] += 1.0
    _, parts = loss_fn(out, batch)
    assert parts["terms"]["pos"]["loss"] == pytest.approx(0.95, abs=1e-5)


def test_hip_offset_is_applied():
    cfg = _toy_cfg()
    cfg["motion_consistency"]["hip_offset_root"] = [0.02, -0.06, -0.07]
    loss_fn = MotionConsistencyLoss(cfg, device="cpu")
    out, batch = _toy_out(n_clips=1, t=5), _toy_batch(n_clips=1, t=5)
    # target = p_gt + R_gt @ offset must land exactly on the predicted point.
    offset = torch.tensor(cfg["motion_consistency"]["hip_offset_root"])
    batch["motion_root_pos"] = (_predicted_pos(out)
                                - batch["motion_rot"] @ offset)
    _, parts = loss_fn(out, batch)
    assert parts["terms"]["pos"]["loss"] == pytest.approx(0.0, abs=1e-6)


def test_rot_term_zero_when_aligned_and_active_when_not():
    loss_fn = MotionConsistencyLoss(_toy_cfg(), device="cpu")
    out, batch = _toy_out(n_clips=1, t=5), _toy_batch(n_clips=1, t=5)
    with torch.no_grad():
        out["mhr"]["global_rot"].zero_()      # R_pred_world = flip = motion_rot
    _, parts = loss_fn(out, batch)
    assert parts["terms"]["rot"]["loss"] == pytest.approx(0.0, abs=1e-9)
    # Rotate the GT root 0.2 rad about x: residual 0.2 > beta -> 0.2 - beta/2.
    angle = 0.2
    rx = torch.tensor([[1.0, 0.0, 0.0],
                       [0.0, float(np.cos(angle)), -float(np.sin(angle))],
                       [0.0, float(np.sin(angle)), float(np.cos(angle))]])
    batch["motion_rot"] = batch["motion_rot"] @ rx
    _, parts = loss_fn(out, batch)
    assert parts["terms"]["rot"]["loss"] == pytest.approx(0.15, abs=1e-5)


def test_linear_only_mode_ignores_angular_rows():
    """``angular: false`` — garbage in the angular rows of the head/GT must not
    move the gt/head terms (they compare linear vel/acc only)."""
    cfg = _toy_cfg()
    cfg["motion_consistency"]["angular"] = False
    loss_fn = MotionConsistencyLoss(cfg, device="cpu")
    out, batch = _toy_out(n_clips=1, t=5), _toy_batch(n_clips=1, t=5)
    _, parts_clean = loss_fn(out, batch)
    with torch.no_grad():
        out["motion"]["joint_motion"][..., 6:] = 1e6
    batch["motion_gt"][..., 6:] = -1e6
    _, parts_garbage = loss_fn(out, batch)
    for name in ("gt", "head"):
        assert parts_garbage["terms"][name]["loss"] == pytest.approx(
            parts_clean["terms"][name]["loss"], rel=1e-9)
    # The angular-aware loss DOES see the garbage (sanity of the probe).
    loss_full = MotionConsistencyLoss(_toy_cfg(), device="cpu")
    _, parts_full = loss_full(out, batch)
    assert parts_full["terms"]["gt"]["loss"] > 1e3


def test_root_validity_gates_the_anchors():
    loss_fn = MotionConsistencyLoss(_toy_cfg(), device="cpu")
    out, batch = _toy_out(n_clips=1, t=5), _toy_batch(n_clips=1, t=5)
    batch["motion_root_valid"][:] = False
    _, parts = loss_fn(out, batch)
    assert parts["terms"]["pos"]["weight_mass"] == 0.0
    assert parts["terms"]["rot"]["weight_mass"] == 0.0
    assert parts["terms"]["head"]["weight_mass"] == 3.0   # twists unaffected


# ---------------------------------------------------------------- GPU semantics

_CORPUS = Path("/data3/rikhat.akizhanov/better/data/ClimbingVideos")

slow = pytest.mark.slow
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
needs_ckpt = pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint missing")
needs_corpus = pytest.mark.skipif(
    not (_CORPUS / "scenes" / "scenes.db").is_file(), reason="corpus missing")


@slow
@needs_gpu
@needs_ckpt
@needs_corpus
def test_full_model_gradient_routing(tmp_path):
    """On a real corpus batch every consistency term reaches the pose path
    (head_pose_ft_proj + pose_temporal) and — head DETACHED — never the motion
    branch, never anything frozen. The recompute hook must stash the frozen
    camera/rotation (the keypoint_supervision rails anchor there)."""
    from contact.data.climbing_corpus import ClimbingCorpusDataset
    from contact.data.collate import batch_to_device, make_collate
    from contact.engine import forward_model
    from contact.model import build_model
    from contact.targets import TargetSpec

    torch.manual_seed(0)
    cfg = load_config(_write(tmp_path, """
base: tests/fixtures/motion_pelvis12_angw05.yaml
model:
  pose_temporal: {enabled: true}
train: {finetune_pose_head: true}
pose_supervision: {enabled: true}
motion_consistency:
  enabled: true
  angular: false
  loss: {pos: 5.0, rot: 2.0}
"""))
    model, trainable = build_model(cfg, "cuda")
    model.eval()

    ds = ClimbingCorpusDataset(
        _CORPUS, scenes=["MuVpoovQl2M_0001"], split="train", frames_per_clip=7,
        frame_stride=1, jitter=False, load_motion=True,
        motion_joint_names=["pelvis"], motion_root_convention="twist",
        motion_target_smooth_sec=0.12, motion_outlier_acc_ms2=50.0,
        motion_angular=True, load_pose=True)
    collate = make_collate(
        tuple(model.cfg.MODEL.IMAGE_SIZE), TargetSpec.from_config(cfg))
    batch = batch_to_device(collate([ds[0], ds[1]]), "cuda")

    from contact.motion_consistency import MotionConsistencyLoss
    loss_fn = MotionConsistencyLoss(cfg, device="cuda")
    out = forward_model(model, batch)
    # The pose write path (pose_temporal) recomputed the final output — the
    # frozen camera must be stashed (the keypoint_supervision rails and the
    # renderer anchor on it), and the zero-gated recompute IS the frozen output.
    assert "pred_cam_t_frozen" in out["mhr"]
    assert torch.equal(
        out["mhr"]["pred_cam_t"], out["mhr"]["pred_cam_t_frozen"])
    assert "global_rot_frozen" in out["mhr"]
    assert torch.equal(
        out["mhr"]["global_rot"], out["mhr"]["global_rot_frozen"])
    total, parts = loss_fn(out, batch)
    for name in ("gt", "head", "pos", "rot"):
        assert parts["terms"][name]["weight_mass"] > 0, name
    assert parts["pos_err_m"] < 1.0, "frozen model should start near the GT root"
    total.backward()

    got = {n for n, p in model.named_parameters()
           if p.grad is not None and float(p.grad.abs().sum()) > 0}
    # SPLIT-HEAD: the trainable copy is head_pose_ft_proj; head_pose.proj stays frozen.
    assert any(n.startswith("head_pose_ft_proj.") for n in got), "pose path missed"
    assert any(n.startswith("pose_temporal.") for n in got), "pose brick missed"
    assert not any(n.startswith("head_motion.") for n in got), (
        "the detached head term leaked gradient into the motion head")
    for n in got:
        assert n in set(trainable) or n.startswith("head_pose_ft_proj."), (
            f"frozen param {n} received a gradient")
