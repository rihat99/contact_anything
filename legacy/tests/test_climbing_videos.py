"""ClimbingVideos loader: completed test labels + window validity/coverage.

Synthetic scenes written to tmp dirs mimic the exporter schema (see
``BetterVideoReconstruction/scripts/export_contact_dataset.py``) so the loader is
exercised without the real corpus.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from contact.config import load_config
from contact.data.climbing_videos import (
    DEFAULT_ROOT,
    ClimbingVideosDataset,
    list_completed_test_scenes,
    list_scenes,
)
from contact.data.collate import make_loaders
from contact.targets import ALWAYS_NON_CONTACT_8, NUM_BODY_22


# ---------------------------------------------------------------- scene builders

def _write_frames(scene_dir: Path, n: int, oid: int) -> None:
    frames = scene_dir / "frames"
    masks = scene_dir / "masks" / f"{oid:02d}"
    frames.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    for pos in range(n):
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(frames / f"{pos:06d}.jpg")
        Image.fromarray(np.full((8, 8), 255, np.uint8)).save(masks / f"{pos:06d}.png")


def _common(n: int, oid: int, valid: np.ndarray,
            extrinsics: np.ndarray | None = None,
            gravity: np.ndarray | None = None) -> dict:
    # Schema-v3 camera keys default to identity extrinsics / downward gravity;
    # camera-specific tests pass their own.
    if extrinsics is None:
        extrinsics = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    if gravity is None:
        gravity = np.array([0.0, 1.0, 0.0], np.float32)
    return dict(
        object_ids=np.array([oid], np.int64),
        frame_indices=np.arange(n, dtype=np.int64),
        bbox=np.tile(np.array([1.0, 1.0, 7.0, 7.0], np.float32), (1, n, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32) * 5.0, (n, 1, 1)),
        valid_mask=valid[None].astype(bool),                # [1, n]
        fps=np.float32(30.0),
        extrinsics=extrinsics.astype(np.float32),           # [n, 4, 4]
        gravity_world=gravity.astype(np.float32),           # [3]
    )


def _train_scene(root: Path, scene: str, n: int, valid: np.ndarray,
                 jc: np.ndarray | None = None, conf: np.ndarray | None = None,
                 oid: int = 7, extrinsics: np.ndarray | None = None,
                 gravity: np.ndarray | None = None) -> None:
    scene_dir = root / "train" / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    _write_frames(scene_dir, n, oid)
    if jc is None:
        jc = np.zeros((1, n, NUM_BODY_22), bool)
    if conf is None:
        conf = np.ones((1, n, NUM_BODY_22), np.float32)
    np.savez(scene_dir / "labels.npz",
             joint_contact_22=jc,
             contact_conf_22=conf,
             **_common(n, oid, valid, extrinsics=extrinsics, gravity=gravity))


def _test_scene(root: Path, scene: str, n: int, valid: np.ndarray, *,
                pending: bool, jc: np.ndarray | None = None,
                annotated: np.ndarray | None = None, oid: int = 7) -> None:
    scene_dir = root / "test" / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    _write_frames(scene_dir, n, oid)
    np.savez(scene_dir / "inputs.npz", **_common(n, oid, valid))
    if pending:
        np.savez(scene_dir / "contacts.npz",
                 joint_contact_22=np.zeros((1, n, NUM_BODY_22), bool),
                 pending=np.bool_(True))
    else:
        if jc is None:
            jc = np.zeros((1, n, NUM_BODY_22), bool)
        if annotated is None:
            annotated = np.ones((1, n, NUM_BODY_22), bool)
        np.savez(scene_dir / "contacts.npz",
                 joint_contact_22=jc, annotated_22=annotated,   # NOTE: no contact_conf_22
                 pending=np.bool_(False))


# ---------------------------------------------------------------- finding 4: labels

def test_train_scene_loads_all_22(tmp_path):
    n = 12
    jc = np.zeros((1, n, NUM_BODY_22), bool)
    jc[0, :, 7] = True                                  # left ankle in contact everywhere
    _train_scene(tmp_path, "vid_0000", n, np.ones(n, bool), jc=jc)
    ds = ClimbingVideosDataset(tmp_path, scenes=["vid_0000"], mode="train",
                               frames_per_clip=4, frame_stride=1, jitter=False)
    assert len(ds) > 0
    frame = ds[0][0]
    assert frame["joint_mask"].shape == (NUM_BODY_22,)
    assert float(frame["joint_mask"].sum()) == NUM_BODY_22   # all-valid -> all supervised
    assert bool(frame["joint_contact"][7] > 0.5)


def test_raw_confidence_is_separate_from_supervision_and_loss_mask(tmp_path):
    n = 4
    conf = np.linspace(0.0, 1.0, NUM_BODY_22, dtype=np.float32)
    conf = np.tile(conf, (1, n, 1))
    _train_scene(tmp_path, "vid_0000", n, np.ones(n, bool), conf=conf)

    plain = ClimbingVideosDataset(tmp_path, scenes=["vid_0000"], mode="val",
                                  frames_per_clip=1, use_confidence_weights=False)
    weighted = ClimbingVideosDataset(tmp_path, scenes=["vid_0000"], mode="val",
                                     frames_per_clip=1, use_confidence_weights=True)
    plain_frame, weighted_frame = plain[0][0], weighted[0][0]

    assert np.allclose(plain_frame["joint_confidence"].numpy(), conf[0, 0])
    assert float(plain_frame["joint_supervised"].sum()) == NUM_BODY_22
    assert np.allclose(plain_frame["joint_mask"].numpy(), np.ones(NUM_BODY_22))
    assert np.allclose(weighted_frame["joint_mask"].numpy(), conf[0, 0])
    assert np.allclose(weighted_frame["joint_confidence"].numpy(), conf[0, 0])
    assert weighted_frame["frame_position"] == 0
    assert weighted_frame["frame_index"] == 0


def test_completed_test_scene_adds_schema_fixed_negative_joints(tmp_path):
    n = 8
    jc = np.zeros((1, n, NUM_BODY_22), bool)
    jc[0, :, 20] = True                                 # left wrist in contact
    annotated = np.zeros((1, n, NUM_BODY_22), bool)
    annotated[0, :, [20, 21, 7, 8]] = True              # only 4 joints annotated
    _test_scene(tmp_path, "vid_0000", n, np.ones(n, bool),
                pending=False, jc=jc, annotated=annotated)

    ds = ClimbingVideosDataset(tmp_path, scenes=["vid_0000"], mode="val",
                               split_dir="test", frames_per_clip=4, frame_stride=1,
                               require_labels=True)      # must NOT raise KeyError
    frame = ds[0][0]
    assert float(frame["joint_mask"].sum()) == 12.0     # 4 observed + 8 fixed negatives
    assert bool(frame["joint_mask"][20] > 0.5) and bool(frame["joint_contact"][20] > 0.5)
    assert float(frame["joint_mask"][0]) == 1.0         # pelvis is a fixed negative
    assert float(frame["joint_contact"][ALWAYS_NON_CONTACT_8].sum()) == 0.0
    assert float(frame["joint_supervised"].sum()) == 12.0
    assert np.allclose(frame["joint_confidence"].numpy(), 1.0)  # no test confidence shipped


def test_wholly_unreviewed_test_frame_stays_unsupervised(tmp_path):
    n = 4
    annotated = np.zeros((1, n, NUM_BODY_22), bool)
    _test_scene(tmp_path, "vid_0000", n, np.ones(n, bool),
                pending=False, annotated=annotated)
    ds = ClimbingVideosDataset(tmp_path, scenes=["vid_0000"], mode="val",
                               split_dir="test", frames_per_clip=1)
    assert float(ds[0][0]["joint_mask"].sum()) == 0.0


def test_test_scene_rejects_contact_on_schema_fixed_joint(tmp_path):
    n = 4
    jc = np.zeros((1, n, NUM_BODY_22), bool)
    jc[..., 0] = True
    annotated = np.zeros((1, n, NUM_BODY_22), bool)
    annotated[..., 20] = True
    _test_scene(tmp_path, "vid_0000", n, np.ones(n, bool),
                pending=False, jc=jc, annotated=annotated)
    with pytest.raises(ValueError, match="schema-fixed"):
        ClimbingVideosDataset(tmp_path, scenes=["vid_0000"], mode="val",
                              split_dir="test", frames_per_clip=1)


def test_pending_test_scene_raises(tmp_path):
    n = 8
    _test_scene(tmp_path, "vid_0000", n, np.ones(n, bool), pending=True)
    with pytest.raises(RuntimeError, match="pending"):
        ClimbingVideosDataset(tmp_path, scenes=["vid_0000"], mode="val",
                              split_dir="test", frames_per_clip=4, frame_stride=1,
                              require_labels=True)


def test_completed_test_scene_discovery_skips_pending(tmp_path):
    n = 5
    _test_scene(tmp_path, "done_0000", n, np.ones(n, bool), pending=False)
    _test_scene(tmp_path, "pending_0000", n, np.ones(n, bool), pending=True)
    assert list_completed_test_scenes(tmp_path) == ["done_0000"]


def test_train_test_loaders_use_all_train_and_manual_test_scenes(tmp_path):
    n = 10
    _train_scene(tmp_path, "train_a_0000", n, np.ones(n, bool))
    _train_scene(tmp_path, "train_b_0000", n, np.ones(n, bool))
    _test_scene(tmp_path, "test_done_0000", n, np.ones(n, bool), pending=False)
    _test_scene(tmp_path, "test_pending_0000", n, np.ones(n, bool), pending=True)

    dataset_cfg = tmp_path / "climbing.yaml"
    dataset_cfg.write_text(f"name: climbing_videos\ndata:\n  root: {tmp_path}\n")
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "climbing_videos_joint_temporal.yaml")
    cfg["data"]["datasets"] = [{
        "name": "climbing_videos", "config": str(dataset_cfg)}]
    cfg["data"]["num_workers"] = 0

    train_loader, test_loader, manifest = make_loaders(cfg, (32, 32))
    train_ds = train_loader.loaders[0].dataset
    test_ds = test_loader.loaders[0].dataset
    assert set(train_ds._scenes) == {"train_a_0000", "train_b_0000"}
    assert set(test_ds._scenes) == {"test_done_0000"}
    key = f"video:{dataset_cfg}"
    assert manifest[key] == {
        "train": ["train_a_0000", "train_b_0000"],
        "test": ["test_done_0000"],
    }

    _, restored_test, restored_manifest = make_loaders(
        cfg, (32, 32), manifest=manifest)
    assert set(restored_test.loaders[0].dataset._scenes) == {"test_done_0000"}
    assert restored_manifest == manifest


def test_require_labels_false_zero_supervision(tmp_path):
    n = 8
    _test_scene(tmp_path, "vid_0000", n, np.ones(n, bool), pending=True)
    ds = ClimbingVideosDataset(tmp_path, scenes=["vid_0000"], mode="val",
                               split_dir="test", frames_per_clip=4, frame_stride=1,
                               require_labels=False)     # no None indexing
    frame = ds[0][0]
    assert float(frame["joint_mask"].sum()) == 0.0
    assert float(frame["joint_contact"].sum()) == 0.0


# ---------------------------------------------------------------- finding 5: windows

def test_partially_invalid_window_rejected(tmp_path):
    n = 8
    valid = np.array([1, 1, 1, 1, 0, 0, 0, 0], bool)    # exactly 4/8 valid over the window
    _train_scene(tmp_path, "vid_0000", n, valid)
    ds = ClimbingVideosDataset(tmp_path, scenes=["vid_0000"], mode="val",
                               frames_per_clip=8, frame_stride=1, jitter=False)
    assert len(ds) == 0, "temporal windows must contain only valid tracked-person rows"


def test_below_50pct_valid_window_rejected(tmp_path):
    n = 8
    valid = np.array([1, 1, 1, 0, 0, 0, 0, 0], bool)    # 3/8 valid -> >50% invalid
    _train_scene(tmp_path, "vid_0000", n, valid)
    ds = ClimbingVideosDataset(tmp_path, scenes=["vid_0000"], mode="val",
                               frames_per_clip=8, frame_stride=1, jitter=False)
    assert len(ds) == 0


def test_zero_bbox_on_invalid_frame_cannot_enter_temporal_clip(tmp_path):
    n = 10
    valid = np.ones(n, bool)
    valid[4] = False
    _train_scene(tmp_path, "vid_0000", n, valid)
    path = tmp_path / "train" / "vid_0000" / "labels.npz"
    with np.load(path) as saved:
        payload = {name: saved[name].copy() for name in saved.files}
    payload["bbox"][0, 4] = 0.0
    np.savez(path, **payload)

    ds = ClimbingVideosDataset(tmp_path, scenes=["vid_0000"], mode="val",
                               frames_per_clip=5, frame_stride=1, jitter=False)
    assert len(ds) == 1                              # base 0 rejected; base 5 survives
    clip = ds[0]
    assert all(frame["frame_valid"] for frame in clip)
    assert all((frame["bbox"][2:] > frame["bbox"][:2]).all() for frame in clip)


def test_degenerate_bbox_on_valid_frame_is_dataset_corruption(tmp_path):
    n = 5
    _train_scene(tmp_path, "vid_0000", n, np.ones(n, bool))
    path = tmp_path / "train" / "vid_0000" / "labels.npz"
    with np.load(path) as saved:
        payload = {name: saved[name].copy() for name in saved.files}
    payload["bbox"][0, 2] = 0.0
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="valid person/frame.*invalid bbox"):
        ClimbingVideosDataset(tmp_path, scenes=["vid_0000"], mode="val",
                              frames_per_clip=5, frame_stride=1)


def test_jittered_window_validity_fallback(tmp_path):
    # Frames 0..7 valid, 8+ invalid. base=0 window is fully valid; jitter can push
    # the start into the invalid tail. The loader must fall back to the all-valid base.
    n = 20
    valid = np.zeros(n, bool)
    valid[:8] = True
    _train_scene(tmp_path, "vid_0000", n, valid)
    ds = ClimbingVideosDataset(tmp_path, scenes=["vid_0000"], mode="train",
                               frames_per_clip=8, frame_stride=1, jitter=True, seed=42)
    assert len(ds) >= 1
    for epoch in range(16):
        ds.set_epoch(epoch)
        for i in range(len(ds)):
            fv = np.array([f["frame_valid"] for f in ds[i]])
            assert fv.all(), f"epoch {epoch} item {i}: invalid frame escaped fallback"


def test_terminal_val_window_covers_tail(tmp_path):
    # T=4 stride=1 -> step=4; n=15 -> max_start=11. Stride tiling hits 0,4,8 and
    # leaves an 11-based tail. Val must append the terminal max_start window; train
    # must not.
    n = 15
    _train_scene(tmp_path, "vid_0000", n, np.ones(n, bool))
    common = dict(scenes=["vid_0000"], frames_per_clip=4, frame_stride=1)
    val = ClimbingVideosDataset(tmp_path, mode="val", jitter=False, **common)
    train = ClimbingVideosDataset(tmp_path, mode="train", jitter=False, **common)
    val_bases = {item[2] for item in val._items}
    train_bases = {item[2] for item in train._items}
    assert 11 in val_bases, f"terminal window missing from val (bases={sorted(val_bases)})"
    assert 11 not in train_bases, "train must not add the terminal window"


# ---------------------------------------------------------------- step 02: cameras

def test_cam_jump_is_sampled_step_displacement(tmp_path):
    """``cam_jump_m`` is the camera-center displacement between consecutive SAMPLED
    clip frames (row 0 = 0.0): a discontinuity on a source frame skipped by
    ``frame_stride=2`` must still appear in the enclosing sampled step — the old
    per-source-frame jump put it on the skipped row, making it invisible — and the
    physics jerk filter must then exclude the clip."""
    import types

    import torch

    from contact.physics.loss import PhysicsLoss

    n = 8
    extr = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    extr[5:, 0, 3] = -7.85            # centre C = -R^T t jumps +7.85 x at 4 -> 5
    _train_scene(tmp_path, "vid_0000", n, np.ones(n, bool), extrinsics=extr)
    ds = ClimbingVideosDataset(tmp_path, scenes=["vid_0000"], mode="val",
                               frames_per_clip=4, frame_stride=2, jitter=False)
    clip = ds[0]
    assert [f["frame_position"] for f in clip] == [0, 2, 4, 6]
    jumps = [f["cam_jump_m"] for f in clip]
    assert jumps[0] == 0.0                              # first clip row
    assert jumps[1] == pytest.approx(0.0) and jumps[2] == pytest.approx(0.0)
    # The 4->6 sampled step encloses the skipped 4->5 discontinuity. The
    # per-source-frame jump at the sampled rows {0,2,4,6} is 0 everywhere.
    assert jumps[3] == pytest.approx(7.85, rel=1e-5)

    # And the jerk filter (real _eligible_clips code, duck-typed self) drops it.
    batch = {
        "frame_valid": torch.ones(4, dtype=torch.bool),
        "cam_valid": torch.ones(4, dtype=torch.bool),
        "cam_jump_m": torch.tensor(jumps, dtype=torch.float32),
    }
    stub = types.SimpleNamespace(min_frames=4, max_cam_jump_m=0.5)
    eligible, n_excluded = PhysicsLoss._eligible_clips(stub, batch, 4, 1)
    assert not bool(eligible.any())
    assert n_excluded == 1


def test_clip_carries_per_frame_camera_and_constant_gravity(tmp_path):
    n = 12
    extr = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    extr[:, 0, 3] = np.arange(n, dtype=np.float32)      # tag each frame by tx = k
    gravity = np.array([0.0, 0.0, 1.0], np.float32)     # non-default, still unit
    _train_scene(tmp_path, "vid_0000", n, np.ones(n, bool),
                 extrinsics=extr, gravity=gravity)
    ds = ClimbingVideosDataset(tmp_path, scenes=["vid_0000"], mode="val",
                               frames_per_clip=4, frame_stride=2, jitter=False)
    clip = ds[0]
    positions = [f["frame_position"] for f in clip]
    assert positions == [0, 2, 4, 6]                    # window offsets respected
    for frame, pos in zip(clip, positions):
        cam = np.asarray(frame["cam_from_world"], np.float32)
        assert cam.shape == (4, 4)
        assert float(cam[0, 3]) == float(pos)           # per-frame extrinsics row
        assert np.allclose(np.asarray(frame["gravity_world"]), gravity)


def test_missing_extrinsics_raises_stale_export(tmp_path):
    n = 6
    _train_scene(tmp_path, "vid_0000", n, np.ones(n, bool))
    path = tmp_path / "train" / "vid_0000" / "labels.npz"
    with np.load(path) as saved:
        payload = {name: saved[name].copy() for name in saved.files
                   if name not in ("extrinsics", "gravity_world")}
    np.savez(path, **payload)                           # drop the camera keys
    with pytest.raises(ValueError, match="stale export"):
        ClimbingVideosDataset(tmp_path, scenes=["vid_0000"], mode="val",
                              frames_per_clip=4, frame_stride=1)


@pytest.mark.slow
@pytest.mark.skipif(not (Path(DEFAULT_ROOT) / "train").is_dir(),
                    reason="ClimbingVideos_v1 not present")
def test_real_scene_cam_from_world_matches_transform():
    import json

    root = Path(DEFAULT_ROOT)
    info = json.loads((root / "dataset_info.json").read_text())
    if int(info.get("schema_version", 0)) < 3:
        pytest.skip("dataset not backfilled to schema v3")
    feat = Path(info["source_data_root"]) / "features"
    for scene in list_scenes(root, "train"):
        ds = ClimbingVideosDataset(root, scenes=[scene], mode="val",
                                   frames_per_clip=4, frame_stride=2, jitter=False)
        if len(ds) > 0:
            break
    else:
        pytest.skip("no train scene long enough for a T=4 window")
    aa, bb = scene[:2], scene[2:4]
    with np.load(feat / "geometry" / aa / bb / scene / "transform.npz") as transform:
        extr = np.asarray(transform["extrinsics"], np.float32)
    for frame in ds[0]:
        pos = frame["frame_position"]
        assert np.allclose(
            np.asarray(frame["cam_from_world"], np.float32), extr[pos], atol=1e-5)
        assert abs(float(np.linalg.norm(np.asarray(frame["gravity_world"]))) - 1.0) < 1e-4
