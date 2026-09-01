"""Tests for the BetterVideoReconstruction out-tree inference dataset."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from contact.config import load_config
from contact.data.collate import make_collate
from contact.data.reconstruction_scenes import ReconstructionSceneDataset, extract_frames
from contact.targets import TargetSpec

N_FRAMES = 4


@pytest.fixture()
def out_tree(tmp_path: Path) -> tuple[Path, Path]:
    """A minimal synthetic ``out/<stem>/`` tree + extracted frames dir."""
    out_dir = tmp_path / "scene"
    (out_dir / "sam3" / "00").mkdir(parents=True)
    (out_dir / "geometry").mkdir()
    (out_dir / "human_optim").mkdir()

    bbox = np.tile(np.array([4, 6, 40, 60], np.int32), (1, N_FRAMES, 1))
    bbox[0, 2] = [10, 10, 10, 10]                       # degenerate box on frame 2
    np.savez(out_dir / "sam3" / "bboxes.npz",
             bboxes_per_obj=bbox, object_ids=np.array([0], np.int32))

    intr = np.tile(np.eye(3, dtype=np.float32) * 100.0, (N_FRAMES, 1, 1))
    extr = np.tile(np.eye(4, dtype=np.float32), (N_FRAMES, 1, 1))
    np.savez(out_dir / "geometry" / "transform.npz",
             intrinsics_px_orig=intr, extrinsics=extr,
             frame_indices=np.arange(N_FRAMES, dtype=np.int32),
             fps=np.float32(30.0), metric=np.bool_(True))

    valid = np.ones((1, N_FRAMES), bool)
    valid[0, 3] = False                                 # untracked frame 3
    np.savez(out_dir / "human_optim" / "contacts_1.npz",
             valid_mask=valid, fps=np.float32(30.0),
             object_ids=np.array([0], np.int32))

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    rng = np.random.default_rng(0)
    for pos in range(N_FRAMES):
        Image.fromarray(rng.integers(0, 255, (64, 48, 3), np.uint8)).save(
            frames_dir / f"{pos:06d}.jpg")
        Image.fromarray(np.full((64, 48), 255, np.uint8)).save(
            out_dir / "sam3" / "00" / f"frame_{pos:06d}.png")
    return out_dir, frames_dir


def test_items_cover_valid_nondegenerate_frames_only(out_tree):
    out_dir, frames_dir = out_tree
    ds = ReconstructionSceneDataset(out_dir, frames_dir)
    assert [item[2] for item in ds._items] == [0, 1]    # frame 2 bad bbox, frame 3 invalid
    data = ds._scenes["scene"]
    assert data["valid_mask"].tolist() == [[True, True, False, False]]
    assert data["fps"] == 30.0
    assert data["frame_indices"].tolist() == list(range(N_FRAMES))


def test_item_collates_with_zero_supervision(out_tree):
    out_dir, frames_dir = out_tree
    ds = ReconstructionSceneDataset(out_dir, frames_dir)
    clip = ds[0]
    assert isinstance(clip, list) and len(clip) == 1
    frame = clip[0]
    assert frame["image"].shape == (64, 48, 3)
    assert frame["mask"].shape == (64, 48)
    assert frame["key"] == "scene#0@0"

    cfg = load_config(REPO / "tests" / "fixtures" / "joint_temporal_center_v2.yaml")
    collate = make_collate((256, 256), TargetSpec.from_config(cfg))
    batch = collate([clip])
    assert batch["img"].shape == (1, 1, 3, 256, 256)
    assert batch["cam_int"].shape == (1, 3, 3)
    assert batch["seq_len"] == 1
    assert float(batch["targets"]["joint"]["mask"].sum()) == 0.0
    assert not bool(batch["cam_valid"].any())           # physics inputs absent by design


def test_window_machinery_accepts_dataset(out_tree):
    import render_climbing_video_contacts as rcv

    out_dir, frames_dir = out_tree
    ds = ReconstructionSceneDataset(out_dir, frames_dir)
    index_map = rcv._frame_index_map(ds, "scene")
    assert set(index_map) == {(0, 0), (0, 1)}
    requests = rcv.sliding_window_requests(
        ds._scenes["scene"]["valid_mask"], seq_len=3, stride=1)
    # The 2-frame valid run collapses to one short window emitting both rows.
    assert requests == {2: [(0, (0, 1), (0, 1))]}


def test_bbox_rows_selected_by_object_id(out_tree):
    out_dir, frames_dir = out_tree
    # sam3 tracked two objects [5, 7]; the human stages kept only object 7.
    bbox = np.stack([
        np.tile(np.array([1, 1, 2, 2], np.int32), (N_FRAMES, 1)),     # object 5
        np.tile(np.array([4, 6, 40, 60], np.int32), (N_FRAMES, 1)),   # object 7
    ])
    np.savez(out_dir / "sam3" / "bboxes.npz",
             bboxes_per_obj=bbox, object_ids=np.array([5, 7], np.int32))
    np.savez(out_dir / "human_optim" / "contacts_1.npz",
             valid_mask=np.ones((1, N_FRAMES), bool), fps=np.float32(30.0),
             object_ids=np.array([7], np.int32))

    ds = ReconstructionSceneDataset(out_dir, frames_dir)
    data = ds._scenes["scene"]
    assert data["object_ids"].tolist() == [7]
    assert data["bbox"].shape == (1, N_FRAMES, 4)
    assert data["bbox"][0, 0].tolist() == [4.0, 6.0, 40.0, 60.0]

    np.savez(out_dir / "human_optim" / "contacts_1.npz",
             valid_mask=np.ones((1, N_FRAMES), bool), fps=np.float32(30.0),
             object_ids=np.array([9], np.int32))
    with pytest.raises(ValueError, match="no sam3 bbox track"):
        ReconstructionSceneDataset(out_dir, frames_dir)


def test_extract_frames_counts(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (48, 32))
    assert writer.isOpened()
    for value in (0, 80, 160, 240):
        writer.write(np.full((32, 48, 3), value, np.uint8))
    writer.release()

    frames_dir = tmp_path / "frames"
    extract_frames(video, frames_dir, 4)
    assert sorted(p.name for p in frames_dir.glob("*.jpg")) == [
        f"{i:06d}.jpg" for i in range(4)]
    with pytest.raises(ValueError, match="decoded"):
        extract_frames(video, tmp_path / "frames5", 5)
    with pytest.raises(ValueError, match="more decodable"):
        extract_frames(video, tmp_path / "frames3", 3)
