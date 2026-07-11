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

from contact.data.climbing_videos import ClimbingVideosDataset
from contact.targets import NUM_BODY_22


# ---------------------------------------------------------------- scene builders

def _write_frames(scene_dir: Path, n: int, oid: int) -> None:
    frames = scene_dir / "frames"
    masks = scene_dir / "masks" / f"{oid:02d}"
    frames.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    for pos in range(n):
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(frames / f"{pos:06d}.jpg")
        Image.fromarray(np.full((8, 8), 255, np.uint8)).save(masks / f"{pos:06d}.png")


def _common(n: int, oid: int, valid: np.ndarray) -> dict:
    return dict(
        object_ids=np.array([oid], np.int64),
        frame_indices=np.arange(n, dtype=np.int64),
        bbox=np.tile(np.array([1.0, 1.0, 7.0, 7.0], np.float32), (1, n, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32) * 5.0, (n, 1, 1)),
        valid_mask=valid[None].astype(bool),                # [1, n]
        fps=np.float32(30.0),
    )


def _train_scene(root: Path, scene: str, n: int, valid: np.ndarray,
                 jc: np.ndarray | None = None, conf: np.ndarray | None = None,
                 oid: int = 7) -> None:
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
             **_common(n, oid, valid))


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


def test_completed_test_scene_uses_annotated_mask(tmp_path):
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
    assert float(frame["joint_mask"].sum()) == 4.0      # only annotated joints supervised
    assert bool(frame["joint_mask"][20] > 0.5) and bool(frame["joint_contact"][20] > 0.5)
    assert float(frame["joint_mask"][0]) == 0.0         # unannotated joint ignored
    assert float(frame["joint_supervised"].sum()) == 4.0
    assert np.allclose(frame["joint_confidence"].numpy(), 1.0)  # no test confidence shipped


def test_pending_test_scene_raises(tmp_path):
    n = 8
    _test_scene(tmp_path, "vid_0000", n, np.ones(n, bool), pending=True)
    with pytest.raises(RuntimeError, match="pending"):
        ClimbingVideosDataset(tmp_path, scenes=["vid_0000"], mode="val",
                              split_dir="test", frames_per_clip=4, frame_stride=1,
                              require_labels=True)


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

def test_exactly_50pct_valid_window_accepted(tmp_path):
    n = 8
    valid = np.array([1, 1, 1, 1, 0, 0, 0, 0], bool)    # exactly 4/8 valid over the window
    _train_scene(tmp_path, "vid_0000", n, valid)
    ds = ClimbingVideosDataset(tmp_path, scenes=["vid_0000"], mode="val",
                               frames_per_clip=8, frame_stride=1, jitter=False)
    assert len(ds) == 1, "exactly-50%-valid window must be accepted (reject only >50% invalid)"


def test_below_50pct_valid_window_rejected(tmp_path):
    n = 8
    valid = np.array([1, 1, 1, 0, 0, 0, 0, 0], bool)    # 3/8 valid -> >50% invalid
    _train_scene(tmp_path, "vid_0000", n, valid)
    ds = ClimbingVideosDataset(tmp_path, scenes=["vid_0000"], mode="val",
                               frames_per_clip=8, frame_stride=1, jitter=False)
    assert len(ds) == 0


def test_jittered_window_validity_fallback(tmp_path):
    # Frames 0..7 valid, 8+ invalid. base=0 window is fully valid; jitter can push
    # the start into the invalid tail. The loader must fall back to base so no
    # returned window is >50% invalid.
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
            assert fv.mean() >= 0.5, f"epoch {epoch} item {i}: {fv.mean():.3f} valid"


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
