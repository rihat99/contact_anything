"""Unit tests for the temporal (world-frame vel/acc) keypoint terms.

Synthetic clips with analytically known world trajectories; the static 2D/3D
terms are weighted zero so only the stencil terms are exercised. All float32
(the library's end-to-end dtype).
"""
from __future__ import annotations

import math

import pytest
import torch

from contact.data.climbing_corpus import NUM_MHR70, NUM_SUP_VERTICES
from contact.keypoint_supervision import (
    MHR70_FACE_INDICES,
    MHR70_FINGER_INDICES,
    KeypointSupervisedLoss,
    joint_weight_vector,
)

#: Default weight mass: 25 body joints + 5 face at 1.0 + 40 fingers at 0.1.
_JOINT_W_SUM = 25.0 + 5.0 + 40.0 * 0.1


def _loss(*, fingers: float = 0.1, face: float = 1.0,
          fit_err_confidence: bool = False, **overrides) -> KeypointSupervisedLoss:
    loss_cfg = {
        "kp2d": 0.0, "kp3d": 0.0, "kp3d_abs": 0.0,
        "kp_vel": 1.0, "kp_acc": 1.0, "vert": 0.0, "vert_abs": 0.0,
        "huber_delta_2d": 0.05, "huber_delta_3d": 0.1,
        "huber_delta_vel": 0.5, "huber_delta_acc": 2.0,
        "outlier_acc": 50.0, "cam_rail": 0.0, "rot_rail": 0.0,
        "cam_rail_margin_m": 0.5, "rot_rail_margin_rad": 0.2,
    }
    loss_cfg.update(overrides)
    return KeypointSupervisedLoss(
        {"keypoint_supervision": {
            "loss": loss_cfg,
            "joint_weights": {"fingers": fingers, "face": face},
            "fit_err_confidence": fit_err_confidence,
            "fit_err_ref_cm": 2.0}},
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


def _batch_and_out(n_clips: int = 2, seq_len: int = 5, fps: float = 30.0):
    """GT: per-joint constant world velocity. Prediction == GT exactly, split
    into ``kp3d + cam_t`` through per-frame MOVING extrinsics."""
    g = torch.Generator().manual_seed(0)
    n = n_clips * seq_len
    # Body-scale spread (~+-1 m): all 70 joints must stay in front of the
    # camera, or the loss's depth guard drops whole rows and the masses below
    # stop being the clean 2 * (T - 2).
    x0 = 0.3 * torch.randn(n_clips, 1, NUM_MHR70, 3, generator=g)
    v = 0.3 * torch.randn(n_clips, 1, NUM_MHR70, 3, generator=g)
    t_idx = torch.arange(seq_len, dtype=torch.float32)[None, :, None, None]
    gt_world = (x0 + v * t_idx / fps).reshape(n, NUM_MHR70, 3)

    ext = torch.stack([_extrinsics(t) for _ in range(n_clips)
                       for t in range(seq_len)])
    gt_cam = (torch.einsum("bij,bkj->bki", ext[:, :3, :3], gt_world)
              + ext[:, :3, 3][:, None])
    cam_t = gt_cam.mean(dim=1)

    # Vertex GT: a rigid box around the body, tracking the same world motion, so
    # an exact prediction is exact for the vertex terms too.
    v_off = torch.linspace(-0.2, 0.2, NUM_SUP_VERTICES)[None, :, None].expand(
        n, NUM_SUP_VERTICES, 3)
    gt_vert_world = gt_world.mean(dim=1, keepdim=True) + v_off
    gt_vert_cam = (torch.einsum("bij,bkj->bki", ext[:, :3, :3], gt_vert_world)
                   + ext[:, :3, 3][:, None])
    pred_vertices = torch.zeros(n, 18439, 3)
    vert_indices = torch.arange(NUM_SUP_VERTICES) * 7
    pred_vertices[:, vert_indices] = gt_vert_cam - cam_t[:, None]

    out = {"mhr": {
        "pred_keypoints_3d": gt_cam - cam_t[:, None],
        "pred_cam_t": cam_t,
        "pred_keypoints_2d_cropped": torch.zeros(n, NUM_MHR70, 2),
        "pred_vertices": pred_vertices,
    }}
    batch = {
        "kp3d_world": gt_world,
        "vert_gt_world": gt_vert_world,
        "vert_valid": torch.ones(n, dtype=torch.bool),
        "vert_indices": vert_indices,
        "mhr_fit_err_cm": torch.full((n,), 0.5),
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


def test_rails_are_zero_inside_the_margin():
    """Small deviations from the frozen stash cost exactly nothing."""
    out, batch = _batch_and_out()
    n = batch["frame_valid"].shape[0]
    frozen_t = out["mhr"]["pred_cam_t"].clone()
    frozen_t[:, 2] += 0.3                                             # < 0.5 m
    out["mhr"]["pred_cam_t_frozen"] = frozen_t
    out["mhr"]["global_rot"] = torch.zeros(n, 3)
    out["mhr"]["global_rot_frozen"] = torch.full((n, 3), 0.05)        # ~5 deg
    _, parts = _loss(kp_vel=0.0, kp_acc=0.0, cam_rail=10.0, rot_rail=10.0)(
        out, batch)
    assert parts["terms"]["cam_rail"]["loss"] == 0.0
    assert parts["terms"]["rot_rail"]["loss"] == 0.0
    assert parts["terms"]["cam_rail"]["weight_mass"] == n
    assert parts["cam_dev_m"] == pytest.approx(0.3, rel=1e-4)


def test_rails_penalize_linearly_beyond_the_margin():
    """A 1 m camera escape costs w * (1 - margin) per row; a 30 deg rotation
    escape costs w * (0.524 - 0.2)."""
    out, batch = _batch_and_out()
    n = batch["frame_valid"].shape[0]
    frozen_t = out["mhr"]["pred_cam_t"].clone()
    frozen_t[:, 2] += 1.0
    out["mhr"]["pred_cam_t_frozen"] = frozen_t
    out["mhr"]["global_rot"] = torch.zeros(n, 3)
    rot = torch.zeros(n, 3)
    rot[:, 1] = math.radians(30.0)
    out["mhr"]["global_rot_frozen"] = rot
    _, parts = _loss(kp_vel=0.0, kp_acc=0.0, cam_rail=10.0, rot_rail=10.0)(
        out, batch)
    assert parts["terms"]["cam_rail"]["loss"] == pytest.approx(
        10.0 * (1.0 - 0.5), rel=1e-5)
    assert parts["terms"]["rot_rail"]["loss"] == pytest.approx(
        10.0 * (math.radians(30.0) - 0.2), rel=1e-3)
    assert parts["rot_dev_deg"] == pytest.approx(30.0, rel=1e-3)


def test_rails_fall_back_to_zero_mass_without_the_stash():
    """No recompute ran (no stash): terms stay in the DDP set with no mass."""
    out, batch = _batch_and_out()
    out["mhr"]["global_rot"] = torch.zeros(batch["frame_valid"].shape[0], 3)
    _, parts = _loss(kp_vel=0.0, kp_acc=0.0, cam_rail=10.0, rot_rail=10.0)(
        out, batch)
    assert parts["terms"]["cam_rail"]["weight_mass"] == 0
    assert parts["terms"]["rot_rail"]["weight_mass"] == 0


# ------------------------------------------------------------- per-joint weights

def test_joint_weight_vector_layout():
    """Fingers and face carry their configured weight; every other MHR70
    keypoint — including both wrists — stays at 1.0."""
    w = joint_weight_vector(0.1, 0.25)
    assert w.shape == (NUM_MHR70,)
    torch.testing.assert_close(w[list(MHR70_FINGER_INDICES)], torch.full((40,), 0.1))
    torch.testing.assert_close(w[list(MHR70_FACE_INDICES)], torch.full((5,), 0.25))
    for body_joint in (41, 62, 9, 10, 13, 14, 69):     # wrists, hips, ankles, neck
        assert float(w[body_joint]) == 1.0
    assert int((w == 1.0).sum()) == NUM_MHR70 - 40 - 5
    assert float(w.sum()) == pytest.approx(_JOINT_W_SUM - 5.0 + 5.0 * 0.25)


def test_per_joint_weights_scale_each_group_in_the_loss():
    """The same error on a finger costs `fingers` times what it costs on a body
    joint; the face group tracks `face`. Terms are means, so the denominator is
    shared and the ratio is exactly the weight ratio."""
    out, batch = _batch_and_out(n_clips=1, seq_len=3)

    def _kp3d_loss(joint: int) -> float:
        mhr = dict(out["mhr"])
        kp = mhr["pred_keypoints_3d"].clone()
        kp[:, joint, 0] += 0.5                         # 50 cm off on one joint
        mhr["pred_keypoints_3d"] = kp
        _, parts = _loss(kp_vel=0.0, kp_acc=0.0, kp3d=1.0)({"mhr": mhr}, batch)
        return parts["terms"]["kp3d"]["loss"]

    body = _kp3d_loss(5)                               # left shoulder
    assert body > 0.0
    assert _kp3d_loss(MHR70_FINGER_INDICES[0]) == pytest.approx(0.1 * body, rel=1e-4)
    assert _kp3d_loss(MHR70_FACE_INDICES[0]) == pytest.approx(body, rel=1e-4)
    # The wrists are body joints, not fingers, even though they bound the block.
    assert _kp3d_loss(62) == pytest.approx(body, rel=1e-4)


def test_row_confidence_halves_the_mass_at_the_reference_residual():
    """fit_err_confidence weights rows by 1 / (1 + (err / ref)^2): at err == ref
    every row weight is 0.5, so the mass halves and the normalized loss does
    not move."""
    out, batch = _batch_and_out(n_clips=1, seq_len=3)
    kp = out["mhr"]["pred_keypoints_3d"].clone()
    kp[:, 5, 0] += 0.5
    out["mhr"]["pred_keypoints_3d"] = kp
    _, plain = _loss(kp_vel=0.0, kp_acc=0.0, kp3d=1.0)(out, batch)
    batch["mhr_fit_err_cm"] = torch.full_like(batch["mhr_fit_err_cm"], 2.0)
    _, weighted = _loss(kp_vel=0.0, kp_acc=0.0, kp3d=1.0,
                        fit_err_confidence=True)(out, batch)
    assert plain["terms"]["kp3d"]["weight_mass"] == 3
    assert weighted["terms"]["kp3d"]["weight_mass"] == pytest.approx(1.5, rel=1e-6)
    assert weighted["terms"]["kp3d"]["loss"] == pytest.approx(
        plain["terms"]["kp3d"]["loss"], rel=1e-5)


# -------------------------------------------------------------- vertex terms

def test_vertex_terms_slice_pred_vertices_by_the_gt_indices():
    """An exact prediction is exactly zero. The fixture writes the GT only at
    `vert_indices` (stride 7) and leaves the other 18055 vertices at zero, so a
    wrong slice could not produce this."""
    out, batch = _batch_and_out(n_clips=1, seq_len=3)
    _, parts = _loss(kp_vel=0.0, kp_acc=0.0, vert=1.0, vert_abs=1.0)(out, batch)
    assert parts["terms"]["vert"]["loss"] == pytest.approx(0.0, abs=1e-6)
    assert parts["terms"]["vert_abs"]["loss"] == pytest.approx(0.0, abs=1e-6)
    assert parts["terms"]["vert"]["weight_mass"] == 3
    assert parts["vert_err_m"] < 1e-5
    assert parts["vert_size_ratio"] == pytest.approx(1.0, rel=1e-4)


def test_vert_sees_body_size_and_vert_abs_sees_depth():
    """The two vertex terms separate the channels they exist for: shrinking the
    body about its root moves `vert` (and the size diagnostic), a pure camera
    depth shift moves `vert_abs` only."""
    out, batch = _batch_and_out(n_clips=1, seq_len=3)
    kp3d = out["mhr"]["pred_keypoints_3d"]
    root = kp3d[:, (9, 10)].mean(dim=1, keepdim=True)
    idx = batch["vert_indices"]

    shrunk = {"mhr": dict(out["mhr"])}
    verts = out["mhr"]["pred_vertices"].clone()
    verts[:, idx] = root + 0.95 * (verts[:, idx] - root)          # 5% smaller
    shrunk["mhr"]["pred_vertices"] = verts
    _, parts = _loss(kp_vel=0.0, kp_acc=0.0, vert=1.0, vert_abs=1.0)(shrunk, batch)
    assert parts["terms"]["vert"]["loss"] > 1e-4
    assert parts["vert_size_ratio"] == pytest.approx(0.95, rel=1e-3)

    deeper = {"mhr": dict(out["mhr"])}
    deeper["mhr"]["pred_cam_t"] = out["mhr"]["pred_cam_t"].clone()
    deeper["mhr"]["pred_cam_t"][:, 2] += 0.1                       # 10 cm deeper
    _, parts = _loss(kp_vel=0.0, kp_acc=0.0, vert=1.0, vert_abs=1.0)(deeper, batch)
    assert parts["terms"]["vert"]["loss"] == pytest.approx(0.0, abs=1e-6)
    assert parts["terms"]["vert_abs"]["loss"] > 1e-3
    assert parts["vert_size_ratio"] == pytest.approx(1.0, rel=1e-4)


def test_vertex_terms_are_masked_by_vert_valid():
    """A row whose GT vertices are missing carries no vertex mass, while the
    keypoint terms on the same row still count."""
    out, batch = _batch_and_out(n_clips=1, seq_len=3)
    batch["vert_valid"] = torch.tensor([True, False, True])
    _, parts = _loss(kp_vel=0.0, kp_acc=0.0, kp3d=1.0, vert=1.0)(out, batch)
    assert parts["terms"]["vert"]["weight_mass"] == 2
    assert parts["terms"]["kp3d"]["weight_mass"] == 3


def test_vertex_gradients_reach_the_prediction():
    """vert/vert_abs differentiate through pred_vertices and pred_cam_t."""
    out, batch = _batch_and_out(n_clips=1, seq_len=3)
    verts = out["mhr"]["pred_vertices"].clone()
    verts[:, batch["vert_indices"]] *= 0.9
    verts = verts.requires_grad_(True)
    cam_t = out["mhr"]["pred_cam_t"].clone().requires_grad_(True)
    out["mhr"]["pred_vertices"] = verts
    out["mhr"]["pred_cam_t"] = cam_t
    total, _ = _loss(kp_vel=0.0, kp_acc=0.0, vert=1.0, vert_abs=1.0)(out, batch)
    total.backward()
    assert verts.grad is not None and float(verts.grad.abs().max()) > 0.0
    assert cam_t.grad is not None and float(cam_t.grad.abs().max()) > 0.0
    # Only the supervised subset receives gradient.
    mask = torch.ones(18439, dtype=torch.bool)
    mask[batch["vert_indices"]] = False
    assert float(verts.grad[:, mask].abs().max()) == 0.0
