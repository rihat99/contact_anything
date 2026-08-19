"""Conditioning features for reconstruction scenes (``cond_rows_from_mhr``).

The peter/BVR prediction path computes ``cond_feat`` rows from the frozen
head's own per-frame reconstruction instead of the corpus artifact. These tests
pin the recipe to the artifact's (``output/motion_probe_geom/
build_cond_features.py``): fixed 0.12 s Gaussian bandwidth, gap-aware runs,
central differences, ``R_ext^T @ diag(1,-1,-1) @ euler_xyz(global_rot)``, and
all-zero rows off the validity support.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from contact.data.climbing_corpus import COND_FEATURE_DIM
from contact.data.reconstruction_scenes import (
    COND_SIGMA_SEC, ReconstructionSceneDataset, cond_rows_from_mhr,
)

_STD = {
    "vel_mean": [0.0, 0.0, 0.0], "vel_std": [1.0, 1.0, 1.0],
    "acc_mean": [0.0, 0.0, 0.0], "acc_std": [1.0, 1.0, 1.0],
}
_FPS = 30.0


def _identity_extrinsics(n: int) -> np.ndarray:
    return np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))


def _reference_rows(pelvis_cam, global_rot, extrinsics, valid, fps, std, clip):
    """Independent reimplementation of the artifact recipe for comparison."""
    import roma
    from scipy.ndimage import gaussian_filter1d

    from contact.data.climbing_corpus import cond_feature_rows

    n = len(valid)
    valid = np.asarray(valid, bool) & np.isfinite(pelvis_cam).all(-1)
    rot_cw = np.asarray(extrinsics, np.float64)[:, :3, :3]
    t_cw = np.asarray(extrinsics, np.float64)[:, :3, 3]
    pos = np.einsum("nji,nj->ni", rot_cw, np.asarray(pelvis_cam, np.float64) - t_cw)

    dt = 1.0 / fps
    vel = np.zeros((n, 3)); acc = np.zeros((n, 3)); ok = np.zeros(n, bool)
    padded = np.pad(valid.astype(np.int8), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    for lo, hi in zip(changes[0::2], changes[1::2]):
        run = np.arange(lo, hi)
        if len(run) < 3:
            continue
        smooth = gaussian_filter1d(
            pos[run], sigma=COND_SIGMA_SEC * fps, axis=0, mode="nearest", truncate=4.0)
        vel[run[1:-1]] = (smooth[2:] - smooth[:-2]) / (2 * dt)
        acc[run[1:-1]] = (smooth[2:] - 2 * smooth[1:-1] + smooth[:-2]) / (dt * dt)
        ok[run[1:-1]] = True
    rot = roma.euler_to_rotmat(
        "xyz", torch.as_tensor(np.nan_to_num(np.asarray(global_rot, np.float64)))).numpy()
    rot_world = np.einsum(
        "nji,jk,nkl->nil", rot_cw, np.diag([1.0, -1.0, -1.0]), rot)
    return cond_feature_rows(vel, acc, rot_world, ok, std, clip)


def test_matches_reference_recipe_with_gap_and_camera():
    rng = np.random.RandomState(3)
    n = 90
    pelvis = np.cumsum(rng.randn(n, 3) * 0.02, axis=0).astype(np.float32)
    global_rot = (rng.randn(n, 3) * 0.3).astype(np.float32)
    ext = _identity_extrinsics(n)
    ext[:, :3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])  # yaw camera
    ext[:, :3, 3] = [0.3, -0.2, 4.0]
    valid = np.ones(n, bool)
    valid[40:47] = False                        # a tracking hole splits the runs
    rows = cond_rows_from_mhr(pelvis, global_rot, ext, valid, _FPS, _STD, 5.0)
    ref = _reference_rows(pelvis, global_rot, ext, valid, _FPS, _STD, 5.0)
    np.testing.assert_array_equal(rows, ref)
    assert rows.shape == (n, COND_FEATURE_DIM) and rows.dtype == np.float32
    # Run endpoints around the hole are invalid ⇒ all-zero rows, bit 0.
    for row in (0, 39, 47, n - 1):
        np.testing.assert_array_equal(rows[row], np.zeros(COND_FEATURE_DIM))
    assert rows[20, 9] == 1.0 and rows[60, 9] == 1.0


def test_constant_acceleration_is_recovered_in_the_interior():
    n = 120
    t = np.arange(n) / _FPS
    pelvis = np.stack([0.5 * t**2, np.zeros(n), np.zeros(n)], -1)  # a = [1,0,0] m/s^2
    rows = cond_rows_from_mhr(
        pelvis, np.zeros((n, 3)), _identity_extrinsics(n),
        np.ones(n, bool), _FPS, _STD, 5.0)
    interior = slice(20, n - 20)                # away from the 'nearest' edge bias
    np.testing.assert_allclose(rows[interior, 3], 1.0, atol=1e-3)   # acc x
    np.testing.assert_allclose(rows[interior, 4:6], 0.0, atol=1e-6)
    # global_rot = 0, R_ext = I ⇒ R_world_from_root = diag(1,-1,-1);
    # gravity [0,1,0] in root axes = [0,-1,0].
    np.testing.assert_allclose(
        rows[interior, 6:9] - np.array([0.0, -1.0, 0.0]), 0.0, atol=1e-6)


def test_nan_frames_never_leak():
    n = 30
    pelvis = np.ones((n, 3), np.float32)
    pelvis[10] = np.nan                          # invalid frame inside the track
    global_rot = np.zeros((n, 3), np.float32)
    global_rot[10] = np.nan
    valid = np.ones(n, bool)
    rows = cond_rows_from_mhr(
        pelvis, global_rot, _identity_extrinsics(n), valid, _FPS, _STD, 5.0)
    assert np.isfinite(rows).all()
    np.testing.assert_array_equal(rows[10], np.zeros(COND_FEATURE_DIM))


def _fake_out_tree(tmp_path):
    from PIL import Image

    n_frames, size = 3, 32
    out = tmp_path / "scene"
    (out / "sam3").mkdir(parents=True)
    (out / "geometry").mkdir()
    (out / "sam3d").mkdir()
    np.savez(out / "sam3" / "bboxes.npz",
             bboxes_per_obj=np.tile([2.0, 2.0, 30.0, 30.0], (1, n_frames, 1)),
             object_ids=np.array([0]))
    np.savez(out / "geometry" / "transform.npz",
             intrinsics_px_orig=np.tile(np.eye(3, dtype=np.float32) * 100, (n_frames, 1, 1)),
             extrinsics=_identity_extrinsics(n_frames).astype(np.float32),
             frame_indices=np.arange(n_frames))
    np.savez(out / "sam3d" / "params.npz",
             valid_mask=np.ones((1, n_frames), bool),
             object_ids=np.array([0]), fps=np.float32(_FPS))
    frames = tmp_path / "frames"
    frames.mkdir()
    for pos in range(n_frames):
        Image.fromarray(np.zeros((size, size, 3), np.uint8)).save(
            frames / f"{pos:06d}.jpg")
    return out, frames


def test_dataset_emits_attached_cond_rows(tmp_path):
    out, frames = _fake_out_tree(tmp_path)
    ds = ReconstructionSceneDataset(out, frames)
    assert "cond_feat" not in ds[0][0]

    rows = np.arange(3 * COND_FEATURE_DIM, dtype=np.float32).reshape(1, 3, -1)
    ds.set_cond_features(rows)
    for index in range(len(ds)):
        (frame,) = ds[index]
        pos = frame["frame_position"]
        got = frame["cond_feat"]
        assert got.dtype == torch.float32 and got.shape == (COND_FEATURE_DIM,)
        np.testing.assert_array_equal(got.numpy(), rows[0, pos])

    with pytest.raises(ValueError, match="cond features"):
        ds.set_cond_features(np.zeros((1, 5, COND_FEATURE_DIM), np.float32))
    ds.set_cond_features(None)
    assert "cond_feat" not in ds[0][0]
