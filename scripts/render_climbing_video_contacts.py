"""Render four-extremity contact predictions directly onto ClimbingVideos clips.

The renderer selects one random scene chunk per source video and draws four
circles at the frozen MHR model's predicted left/right wrist and ankle
positions. Contact is red and non-contact is green. No mesh or skeleton is
rendered.

Temporal checkpoints use centered sliding inference. Interior frames take the
third (center) output of a five-frame window. The first and last two frames of
each contiguous person track come from three-frame boundary windows. Tracks
shorter than three frames fall back to per-frame inference so every valid frame
is still rendered. Launching with ``torchrun`` shards the selected videos across
the available ranks; each rank owns one GPU and no DDP model wrapper is needed.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import yaml
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from contact import checkpoint as ckpt_io
from contact.config import load_config
from contact.data.climbing_videos import ClimbingVideosDataset, list_scenes
from contact.data.collate import batch_to_device, make_collate
from contact.data.splits import video_id_from_scene
from contact.engine import forward_model
from contact.model import build_model
from contact.targets import EXTREMITY_4_NAMES, TargetSpec


# OpenCV BGR colors.
CONTACT_COLOR = (45, 45, 235)
FREE_COLOR = (55, 185, 75)
OUTLINE_COLOR = (245, 245, 245)


def select_random_scenes(
    scenes: list[str], count: int, seed: int,
) -> list[tuple[str, str]]:
    """Select at most one random scene per source video, deterministically."""
    by_video: dict[str, list[str]] = defaultdict(list)
    for scene in scenes:
        by_video[video_id_from_scene(scene)].append(scene)
    rng = random.Random(seed)
    video_ids = sorted(by_video)
    rng.shuffle(video_ids)
    selected = []
    for video_id in video_ids[:count]:
        selected.append((video_id, rng.choice(sorted(by_video[video_id]))))
    return selected


def _dataset_root(cfg: dict) -> str:
    entry = next(d for d in cfg["data"]["datasets"] if d["name"] == "climbing_videos")
    dataset_cfg = yaml.safe_load(Path(entry["config"]).read_text()) or {}
    return str(dataset_cfg["data"]["root"])


def temporal_window_requests(
    valid_mask: np.ndarray,
) -> dict[int, list[tuple[int, tuple[int, ...], tuple[int, ...]]]]:
    """Build centered temporal requests for every contiguous valid person track.

    A request is ``(person, frame_positions, emitted_offsets)``. Long tracks use
    one T=3 request for each boundary and overlapping T=5 windows in the
    interior. Only offset 2 (the third frame) is emitted from every T=5 window.
    The boundary T=3 requests emit the exact edge rows they own. This avoids
    predicting any output frame twice while retaining all available context.
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.ndim != 2:
        raise ValueError(f"valid_mask must be [people, frames]; got {valid_mask.shape}")
    requests: dict[int, list[tuple[int, tuple[int, ...], tuple[int, ...]]]] = {
        1: [], 3: [], 5: [],
    }
    for person, row in enumerate(valid_mask):
        padded = np.pad(row.astype(np.int8), (1, 1))
        changes = np.flatnonzero(np.diff(padded))
        for start, end in zip(changes[0::2], changes[1::2]):
            length = int(end - start)
            if length < 3:
                for position in range(int(start), int(end)):
                    requests[1].append((person, (position,), (0,)))
                continue
            if length == 3:
                positions = tuple(range(int(start), int(end)))
                requests[3].append((person, positions, (0, 1, 2)))
                continue

            # Own the first two and last two outputs with T=3 windows. For a
            # four-frame run these are two overlapping windows with disjoint
            # emitted rows.
            first = tuple(range(int(start), int(start + 3)))
            last = tuple(range(int(end - 3), int(end)))
            requests[3].append((person, first, (0, 1)))
            requests[3].append((person, last, (1, 2)))

            # T=5 windows exist from a run length of five onward. Each owns only
            # its center output, which is local offset 2.
            for center in range(int(start + 2), int(end - 2)):
                positions = tuple(range(center - 2, center + 3))
                requests[5].append((person, positions, (2,)))
    return requests


def _frame_index_map(ds: ClimbingVideosDataset, scene: str) -> dict[tuple[int, int], int]:
    """Map valid ``(person, frame_position)`` rows to the dataset's T=1 items."""
    result = {}
    for index, (item_scene, person, frame_position, _) in enumerate(ds._items):
        if item_scene == scene:
            result[(person, frame_position)] = index
    return result


def _predict_requests(
    model,
    ds: ClimbingVideosDataset,
    scene: str,
    requests: list[tuple[int, tuple[int, ...], tuple[int, ...]]],
    seq_len: int,
    batch_size: int,
    device: str,
    collate,
    anchor_indices: list[int],
    probs: np.ndarray,
    points: np.ndarray,
) -> None:
    """Run homogeneous-T requests and write only their requested output rows."""
    if not requests:
        return
    item_index = _frame_index_map(ds, scene)
    data = ds._scenes[scene]
    frame_indices = data["frame_indices"]
    fps = float(data["fps"])

    for lo in tqdm(
        range(0, len(requests), batch_size),
        desc=f"predict {scene} T={seq_len}",
        leave=False,
    ):
        selected = requests[lo:lo + batch_size]
        # Sliding windows overlap heavily. Cache decoded T=1 samples within this
        # batch so neighboring requests do not repeatedly read the same JPEG.
        frame_cache: dict[tuple[int, int], dict] = {}
        clips = []
        for person, positions, _ in selected:
            if len(positions) != seq_len:
                raise AssertionError(
                    f"T={seq_len} request has {len(positions)} positions: {positions}")
            start_index = float(frame_indices[positions[0]])
            clip = []
            for position in positions:
                key = (person, position)
                if key not in item_index:
                    raise RuntimeError(
                        f"{scene}: no valid T=1 dataset row for person/frame {key}")
                if key not in frame_cache:
                    frame_cache[key] = ds[item_index[key]][0]
                frame = dict(frame_cache[key])
                # A T=1 dataset item always has position zero. Rebuild the
                # elapsed timestamp for the assembled temporal clip.
                frame["frame_pos_sec"] = (
                    float(frame_indices[position]) - start_index
                ) / fps
                clip.append(frame)
            clips.append(clip)

        batch = batch_to_device(collate(clips), device)
        with torch.inference_mode():
            output = forward_model(model, batch)
        batch_probs = torch.sigmoid(
            output["contact"]["joint_logits"]
        ).float().cpu().numpy()
        batch_points = (
            output["mhr"]["pred_keypoints_2d"][:, anchor_indices]
            .float().cpu().numpy()
        )

        for clip_index, (person, positions, emitted_offsets) in enumerate(selected):
            for offset in emitted_offsets:
                row = clip_index * seq_len + offset
                frame_position = positions[offset]
                if np.isfinite(probs[person, frame_position]).any():
                    raise AssertionError(
                        f"duplicate prediction for {scene} person={person} "
                        f"frame={frame_position}")
                probs[person, frame_position] = batch_probs[row]
                points[person, frame_position] = batch_points[row]


def _predict_scene(
    model,
    cfg: dict,
    root: str,
    scene: str,
    batch_size: int,
    device: str,
) -> tuple[ClimbingVideosDataset, np.ndarray, np.ndarray]:
    """Return dataset plus ``probs[P,N,4]`` and keypoints ``[P,N,4,2]``."""
    ds = ClimbingVideosDataset(
        root=root,
        scenes=[scene],
        mode="val",
        split_dir="test",
        frames_per_clip=1,
        frame_stride=1,
        jitter=False,
        seed=int(cfg["data"]["seed"]),
        use_confidence_weights=False,
        require_labels=False,
    )
    data = ds._scenes[scene]
    n_people, n_frames = data["valid_mask"].shape
    probs = np.full((n_people, n_frames, 4), np.nan, dtype=np.float32)
    points = np.full((n_people, n_frames, 4, 2), np.nan, dtype=np.float32)
    spec = TargetSpec.from_config(cfg)
    if spec.joint_names != EXTREMITY_4_NAMES:
        raise ValueError(
            f"video renderer requires extremities_4; got {spec.joint_set}: {spec.joint_names}")
    collate = make_collate(tuple(model.cfg.MODEL.IMAGE_SIZE), spec)
    anchor_indices = list(model.contact_keypoint_indices)
    if len(anchor_indices) != 4:
        raise ValueError(f"expected four MHR anchors; got {anchor_indices}")

    temporal_enabled = bool(cfg["model"].get("temporal", {}).get("enabled", False))
    if temporal_enabled:
        requests_by_t = temporal_window_requests(data["valid_mask"])
    else:
        requests_by_t = {
            1: [
                (person, (frame_position,), (0,))
                for item_scene, person, frame_position, _ in ds._items
                if item_scene == scene
            ],
        }
    for seq_len in sorted(requests_by_t):
        _predict_requests(
            model, ds, scene, requests_by_t[seq_len], seq_len, batch_size,
            device, collate, anchor_indices, probs, points,
        )
    return ds, probs, points


def _draw_contacts(
    frame: np.ndarray,
    frame_probs: np.ndarray,
    frame_points: np.ndarray,
    threshold: float,
) -> None:
    height, width = frame.shape[:2]
    radius = max(6, int(round(min(height, width) * 0.009)))
    outline = max(2, radius // 4)
    for person in range(frame_probs.shape[0]):
        for probability, xy in zip(frame_probs[person], frame_points[person]):
            if not np.isfinite(probability) or not np.isfinite(xy).all():
                continue
            x, y = (int(round(float(xy[0]))), int(round(float(xy[1]))))
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            color = CONTACT_COLOR if probability >= threshold else FREE_COLOR
            cv2.circle(frame, (x, y), radius + outline, OUTLINE_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)


def _render_scene(
    scene: str,
    ds: ClimbingVideosDataset,
    probs: np.ndarray,
    points: np.ndarray,
    threshold: float,
    output_path: Path,
) -> dict:
    data = ds._scenes[scene]
    frames_dir = data["dir"] / "frames"
    n_frames = len(data["frame_indices"])
    first = cv2.imread(str(frames_dir / "000000.jpg"), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(frames_dir / "000000.jpg")
    height, width = first.shape[:2]
    fps = float(data["fps"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {output_path}")
    try:
        for frame_position in tqdm(range(n_frames), desc=f"render {scene}", leave=False):
            frame = cv2.imread(
                str(frames_dir / f"{frame_position:06d}.jpg"), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(frames_dir / f"{frame_position:06d}.jpg")
            _draw_contacts(
                frame, probs[:, frame_position], points[:, frame_position], threshold)
            writer.write(frame)
    finally:
        writer.release()

    valid = np.isfinite(probs)
    predicted_contact = valid & (probs >= threshold)
    return {
        "scene": scene,
        "output": output_path.name,
        "frames": n_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "tracked_people": int(probs.shape[0]),
        "valid_extremity_predictions": int(valid.sum()),
        "predicted_contact_fraction": float(
            predicted_contact.sum() / max(int(valid.sum()), 1)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs" / "climbing_videos_joint.yaml")
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--num-videos", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument(
        "--batch-size", type=int, default=12,
        help="clips per GPU forward (12 T=5 clips = the 60-frame training budget)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed video inference requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = f"cuda:{local_rank}"
    else:
        device = args.device

    checkpoint = Path(args.checkpoint).resolve()
    threshold_tag = f"{args.threshold:.2f}".replace(".", "")
    output_dir = (
        args.output_dir if args.output_dir is not None else
        checkpoint.parent / f"video_inference_{args.split}_last_t{threshold_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    root = _dataset_root(cfg)
    scenes = list_scenes(root, args.split)
    selected = select_random_scenes(scenes, args.num_videos, args.seed)
    if len(selected) < args.num_videos:
        raise ValueError(
            f"requested {args.num_videos} distinct source videos, found {len(selected)}")

    print(f"[rank {rank}/{world_size}] Building model on {device} and loading "
          f"{checkpoint.name} …")
    model, _ = build_model(cfg, device)
    state = ckpt_io.load(checkpoint, model, config=cfg, map_location=device)
    model.eval()
    if rank == 0:
        print(f"Checkpoint epoch {state['epoch']}; threshold {args.threshold:.2f}")
        print("Selected:", ", ".join(f"{video_id}:{scene}" for video_id, scene in selected))

    local_records = []
    indexed = list(enumerate(selected, start=1))
    for index, (video_id, scene) in indexed[rank::world_size]:
        print(f"[rank {rank}] [{index}/{len(selected)}] {video_id} -> {scene}")
        ds, probs, points = _predict_scene(
            model, cfg, root, scene, args.batch_size, device)
        output_path = output_dir / f"{index:02d}_{scene}_contacts_t{threshold_tag}.mp4"
        record = _render_scene(scene, ds, probs, points, args.threshold, output_path)
        record["source_video"] = video_id
        record["selection_index"] = index
        local_records.append(record)
        print(f"  saved {output_path} ({record['frames']} frames, {record['fps']:.2f} fps)")

    if world_size > 1:
        gathered: list[list[dict] | None] = [None] * world_size
        dist.all_gather_object(gathered, local_records)
        records = [record for shard in gathered for record in (shard or [])]
    else:
        records = local_records
    records.sort(key=lambda record: record["selection_index"])

    if rank == 0:
        temporal_enabled = bool(cfg["model"].get("temporal", {}).get("enabled", False))
        summary = {
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": int(state["epoch"]),
            "split": args.split,
            "threshold": args.threshold,
            "seed": args.seed,
            "selection": "one seeded-random scene chunk per distinct source video",
            "inference": (
                "centered sliding T=5; T=3 at track boundaries; T=1 only for "
                "tracks shorter than three frames"
                if temporal_enabled else "per-frame T=1"
            ),
            "world_size": world_size,
            "joint_names": list(EXTREMITY_4_NAMES),
            "circle_colors": {"contact": "red", "non_contact": "green"},
            "videos": records,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"Done: {output_dir}")
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
