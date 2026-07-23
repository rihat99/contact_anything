"""Render four-extremity contacts (and optional force arrows) onto ClimbingVideos clips.

The renderer selects one random scene chunk per source video and draws four
circles at the frozen MHR model's predicted left/right wrist and ankle
positions. Contact is red and non-contact is green. With ``--overlay-labels``,
the filled inner circle is the dataset label and the outer ring is the model
prediction. No mesh or skeleton is rendered.

Inference windows follow the config's ``data.sequence`` (``frames_per_clip`` T,
``frame_stride`` s). Every source frame belongs to exactly one stride-``s``
parity subsequence; within each parity a contiguous valid person track is tiled
by centered sliding windows of T sampled frames. Each window owns (emits) a
central block of rows and boundary windows clamp to the track edge to emit the
uncovered edge rows; a track shorter than T collapses to a single window that
emits every row (down to T=1). Every valid (person, frame) is predicted exactly
once. A per-frame (T=1, s=1) config reduces to one forward per frame.

When the checkpoint has a force head, per-extremity **force arrows** are drawn on
top of each contact disk: the predicted 3D force is a metric segment of
``FORCE_METERS_PER_BW`` metres per body weight at the extremity's camera-frame
3D position, perspective-projected through the dataset's per-frame intrinsics
(so on-image direction and foreshortening are the camera's own), then attached
to the extremity's 2D keypoint (colour by extremity). Only the
``local_world_aligned`` force frame is drawn.

Launching with ``torchrun`` shards the selected videos across the available
ranks; each rank owns one GPU and no DDP model wrapper is needed.
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
from contact.data.climbing_videos import (
    ClimbingVideosDataset,
    list_completed_test_scenes,
    list_scenes,
)
from contact.data.collate import batch_to_device, make_collate
from contact.data.splits import video_id_from_scene
from contact.engine import forward_model
from contact.model import build_model
from contact.targets import (
    EXTREMITY_4_NAMES,
    TargetSpec,
    reduce_body22_to_extremities,
)


# OpenCV BGR colors.
CONTACT_COLOR = (45, 45, 235)
FREE_COLOR = (55, 185, 75)
OUTLINE_COLOR = (245, 245, 245)


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    """Convert an ``#rrggbb`` colour to an OpenCV BGR tuple."""
    color = color.lstrip("#")
    red, green, blue = (int(color[i:i + 2], 16) for i in (0, 2, 4))
    return (blue, green, red)


# One arrow colour per force output (left_hand, right_hand, left_foot, right_foot),
# matching scripts/demo_climbing_videos.py::FORCE_COLORS (there in RGB hex).
FORCE_COLORS = ("#e0530f", "#f0a500", "#1c72d8", "#2fb3ad")
FORCE_COLORS_BGR = tuple(_hex_to_bgr(color) for color in FORCE_COLORS)
FORCE_METERS_PER_BW = 1.0       # 3D arrow length in metres per unit body weight of |f|
FORCE_MIN_BW = 1.0e-3           # skip near-zero forces (e.g. a zero-init head)
FORCE_THICKNESS = 8             # arrow line thickness (px); outline adds +3
FORCE_OUTLINE_BGR = (25, 25, 25)
# LWA (camera y-up) -> OpenCV camera frame (y-down, z-forward).
FORCE_FLIP = np.array([1.0, -1.0, -1.0])


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


def _emit_margin(seq_len: int) -> int:
    """Context rows trimmed from each side of a full interior window.

    Emitted rows sit in the window centre with ``margin`` context frames on each
    side; the leading/trailing ``margin`` rows are emitted only by the boundary
    windows that own them. ``(T - 1) // 4`` gives 0 for ``T = 1`` (per-frame)
    and, for the decisive ``T = 16`` force config, ``margin = 3`` — so the emitted
    central block ``{3..12}`` coincides with the physics residual frames.
    """
    return (seq_len - 1) // 4


def _contiguous_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open ``(lo, hi)`` index ranges of each contiguous True run."""
    padded = np.pad(np.asarray(valid, dtype=np.int8), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    return [(int(lo), int(hi)) for lo, hi in zip(changes[0::2], changes[1::2])]


def _centered_windows(length: int, seq_len: int):
    """Tile a contiguous track of ``length`` sampled frames with centered windows.

    Yields ``(window_start, emit_lo, emit_hi)`` in the track's local sampled-index
    space: the window spans ``[window_start, window_start + min(seq_len, length))``
    and OWNS output rows ``[emit_lo, emit_hi)``. The owned ranges partition
    ``[0, length)`` exactly once. A track shorter than ``seq_len`` collapses to a
    single window that emits every row.
    """
    if length <= seq_len:
        yield 0, 0, length
        return
    margin = _emit_margin(seq_len)
    covered = 0
    while covered < length:
        start = min(max(covered - margin, 0), length - seq_len)
        # The final window (clamped to the right edge) emits to the track end;
        # every other window leaves ``margin`` trailing rows for the next window.
        emit_hi = length if start == length - seq_len else start + seq_len - margin
        yield start, covered, emit_hi
        covered = emit_hi


def plan_track_windows(
    valid_row: np.ndarray, seq_len: int, stride: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Plan centered sliding windows for one person's frame track.

    ``valid_row`` is the 1D per-frame validity mask over source frame positions.
    Frames are grouped into ``stride`` parity subsequences (offsets ``0..s-1``)
    so every source frame is covered; each contiguous valid run within a parity
    is tiled by :func:`_centered_windows`.

    :returns: a list of ``(positions, emitted_offsets)``. ``positions`` are the
        source frame positions of one window (length ``min(seq_len, run_len)``,
        stepping by ``stride``); ``emitted_offsets`` index into ``positions`` for
        the rows this window owns. Every valid source frame appears in exactly
        one owned ``(window, offset)``.
    """
    valid_row = np.asarray(valid_row, dtype=bool)
    if valid_row.ndim != 1:
        raise ValueError(f"valid_row must be 1D [frames]; got {valid_row.shape}")
    if seq_len < 1 or stride < 1:
        raise ValueError(f"seq_len and stride must be >= 1; got {seq_len}, {stride}")

    requests: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    n_frames = valid_row.shape[0]
    for offset in range(stride):
        sampled_positions = np.arange(offset, n_frames, stride)      # source positions
        for lo, hi in _contiguous_runs(valid_row[sampled_positions]):
            run_positions = sampled_positions[lo:hi]                 # contiguous in sampled space
            length = int(run_positions.shape[0])
            window_len = min(seq_len, length)
            for start, emit_lo, emit_hi in _centered_windows(length, seq_len):
                positions = tuple(int(p) for p in run_positions[start:start + window_len])
                emitted = tuple(range(emit_lo - start, emit_hi - start))
                requests.append((positions, emitted))
    return requests


def sliding_window_requests(
    valid_mask: np.ndarray, seq_len: int, stride: int,
) -> dict[int, list[tuple[int, tuple[int, ...], tuple[int, ...]]]]:
    """Centered windows for every person, grouped by window length for batching.

    :returns: ``{window_len: [(person, positions, emitted_offsets), ...]}``.
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.ndim != 2:
        raise ValueError(f"valid_mask must be [people, frames]; got {valid_mask.shape}")
    requests: dict[int, list[tuple[int, tuple[int, ...], tuple[int, ...]]]] = defaultdict(list)
    for person, row in enumerate(valid_mask):
        for positions, emitted in plan_track_windows(row, seq_len, stride):
            requests[len(positions)].append((person, positions, emitted))
    return dict(requests)


def _frame_index_map(ds: ClimbingVideosDataset, scene: str) -> dict[tuple[int, int], int]:
    """Map valid ``(person, frame_position)`` rows to the dataset's T=1 items."""
    result = {}
    for index, (item_scene, person, frame_position, _) in enumerate(ds._items):
        if item_scene == scene:
            result[(person, frame_position)] = index
    return result


def _scene_ground_truth(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return four-extremity labels and known-label mask as ``[P,N,4]`` arrays."""
    raw_contact = data.get("joint_contact")
    if raw_contact is None:
        raise ValueError("dataset labels were not loaded for this scene")

    contact = torch.as_tensor(raw_contact, dtype=torch.float32)
    valid = torch.as_tensor(data["valid_mask"], dtype=torch.bool)
    supervised = valid[..., None].expand_as(contact).clone()
    annotated = data.get("annotated")
    if annotated is not None:
        supervised &= torch.as_tensor(annotated, dtype=torch.bool)

    raw_confidence = data.get("contact_conf")
    confidence = (
        torch.ones_like(contact)
        if raw_confidence is None
        else torch.as_tensor(raw_confidence, dtype=torch.float32)
    )
    reduced_contact, reduced_supervised, _ = reduce_body22_to_extremities(
        contact, supervised, confidence)
    return (
        (reduced_contact > 0.5).cpu().numpy(),
        (reduced_supervised > 0).cpu().numpy(),
    )


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
    force_data: dict[str, np.ndarray] | None = None,
) -> None:
    """Run homogeneous-T requests and write only their requested output rows.

    When ``force_data`` is given (force head present, ``local_world_aligned``
    frame), the per-row force vector plus the anchors' camera-frame 3D position
    are stored alongside the 2D anchor keypoints so the render loop can draw
    force arrows.
    """
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
        collect_force = force_data is not None and output.get("force") is not None
        if collect_force:
            batch_forces = output["force"]["joint_forces"].float().cpu().numpy()
            # Camera-frame 3D position of each anchor (keypoints_3d + cam translation),
            # matching demo_climbing_videos._draw_force_arrows' ``point_cam``.
            batch_anchor_cam = (
                output["mhr"]["pred_keypoints_3d"][:, anchor_indices]
                + output["mhr"]["pred_cam_t"][:, None, :]
            ).float().cpu().numpy()

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
                if collect_force:
                    force_data["forces"][person, frame_position] = batch_forces[row]
                    force_data["anchor_cam"][person, frame_position] = batch_anchor_cam[row]


def _predict_scene(
    model,
    cfg: dict,
    root: str,
    scene: str,
    batch_size: int,
    device: str,
    split_dir: str = "test",
    require_labels: bool = False,
    collect_force: bool = False,
) -> tuple[ClimbingVideosDataset, np.ndarray, np.ndarray, dict[str, np.ndarray] | None]:
    """Return dataset, ``probs[P,N,4]``, keypoints ``[P,N,4,2]`` and force data.

    The item store is a per-frame (T=1) dataset; inference windows follow the
    config's ``data.sequence``. ``force_data`` is ``None`` unless ``collect_force``,
    in which case it carries ``forces[P,N,4,3]`` and camera-frame anchor positions
    ``anchor_cam[P,N,4,3]`` for the force-arrow projection.
    """
    ds = ClimbingVideosDataset(
        root=root,
        scenes=[scene],
        mode="val",
        split_dir=split_dir,
        frames_per_clip=1,
        frame_stride=1,
        jitter=False,
        seed=int(cfg["data"]["seed"]),
        use_confidence_weights=False,
        require_labels=require_labels,
    )
    data = ds._scenes[scene]
    n_people, n_frames = data["valid_mask"].shape
    probs = np.full((n_people, n_frames, 4), np.nan, dtype=np.float32)
    points = np.full((n_people, n_frames, 4, 2), np.nan, dtype=np.float32)
    force_data = None
    if collect_force:
        force_data = {
            "forces": np.full((n_people, n_frames, 4, 3), np.nan, dtype=np.float32),
            "anchor_cam": np.full((n_people, n_frames, 4, 3), np.nan, dtype=np.float32),
        }
    spec = TargetSpec.from_config(cfg)
    if spec.joint_names != EXTREMITY_4_NAMES:
        raise ValueError(
            f"video renderer requires extremities_4; got {spec.joint_set}: {spec.joint_names}")
    collate = make_collate(tuple(model.cfg.MODEL.IMAGE_SIZE), spec)
    anchor_indices = list(model.contact_keypoint_indices)
    if len(anchor_indices) != 4:
        raise ValueError(f"expected four MHR anchors; got {anchor_indices}")

    seq = cfg["data"]["sequence"]
    requests_by_t = sliding_window_requests(
        data["valid_mask"], int(seq["frames_per_clip"]), int(seq["frame_stride"]))
    for seq_len in sorted(requests_by_t):
        _predict_requests(
            model, ds, scene, requests_by_t[seq_len], seq_len, batch_size,
            device, collate, anchor_indices, probs, points, force_data,
        )
    return ds, probs, points, force_data


def _draw_contacts(
    frame: np.ndarray,
    frame_probs: np.ndarray,
    frame_points: np.ndarray,
    threshold: float,
    frame_labels: np.ndarray | None = None,
    frame_label_mask: np.ndarray | None = None,
) -> None:
    """Draw prediction disks, or prediction rings around inner label disks."""
    height, width = frame.shape[:2]
    radius = max(6, int(round(min(height, width) * 0.009)))
    outline = max(2, radius // 4)
    if (frame_labels is None) != (frame_label_mask is None):
        raise ValueError("frame_labels and frame_label_mask must be provided together")
    if frame_labels is not None:
        if frame_labels.shape != frame_probs.shape or frame_label_mask.shape != frame_probs.shape:
            raise ValueError(
                "label, label-mask and probability shapes must match; got "
                f"{frame_labels.shape}, {frame_label_mask.shape}, {frame_probs.shape}")
        radius = max(9, int(round(min(height, width) * 0.012)))
        inner_radius = max(4, int(round(radius * 0.48)))

    for person in range(frame_probs.shape[0]):
        for joint, (probability, xy) in enumerate(
            zip(frame_probs[person], frame_points[person])
        ):
            if not np.isfinite(probability) or not np.isfinite(xy).all():
                continue
            x, y = (int(round(float(xy[0]))), int(round(float(xy[1]))))
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            color = CONTACT_COLOR if probability >= threshold else FREE_COLOR
            cv2.circle(frame, (x, y), radius + outline, OUTLINE_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)
            if frame_labels is not None:
                # A thin white separator makes agreement and disagreement easy
                # to read: prediction is the outer ring, label is the inner disk.
                cv2.circle(
                    frame, (x, y), inner_radius + 2, OUTLINE_COLOR, -1, cv2.LINE_AA)
                if bool(frame_label_mask[person, joint]):
                    label_color = (
                        CONTACT_COLOR
                        if bool(frame_labels[person, joint])
                        else FREE_COLOR
                    )
                    cv2.circle(
                        frame, (x, y), inner_radius, label_color, -1, cv2.LINE_AA)


def _draw_force_arrows(
    frame: np.ndarray,
    frame_points: np.ndarray,
    frame_forces: np.ndarray,
    frame_anchor_cam: np.ndarray,
    cam_int: np.ndarray,
) -> None:
    """Draw one predicted-force arrow per extremity per person on top of the disks.

    The force (``local_world_aligned``, camera y-up) is flipped into the OpenCV
    camera frame and treated as a metric 3D segment of ``FORCE_METERS_PER_BW``
    metres per body weight starting at the extremity's camera-frame position.
    Both endpoints are perspective-projected through the dataset's intrinsics —
    on-image direction and foreshortening are the real camera's — and the
    projected segment is attached to the extremity's 2D keypoint (the model's
    3D and the dataset camera do not share an exact projection).

    :param frame_points: ``[P, 4, 2]`` anchor keypoints (full-image px).
    :param frame_forces: ``[P, 4, 3]`` predicted forces (body weight, head frame).
    :param frame_anchor_cam: ``[P, 4, 3]`` camera-frame anchor positions.
    :param cam_int: ``[3, 3]`` dataset camera intrinsics for this frame.
    """
    height, width = frame.shape[:2]
    fx, fy = float(cam_int[0, 0]), float(cam_int[1, 1])
    cx, cy = float(cam_int[0, 2]), float(cam_int[1, 2])
    if not (np.isfinite(fx) and np.isfinite(fy)) or fx <= 0 or fy <= 0:
        return
    for person in range(frame_forces.shape[0]):
        for out_idx in range(frame_forces.shape[1]):
            force = np.asarray(frame_forces[person, out_idx], dtype=np.float64)
            if not np.isfinite(force).all():
                continue
            mag = float(np.linalg.norm(force))
            if mag < FORCE_MIN_BW:
                continue
            point_cam = np.asarray(frame_anchor_cam[person, out_idx], dtype=np.float64)
            if not np.isfinite(point_cam).all() or point_cam[2] <= 1e-3:  # behind camera
                continue
            anchor = np.asarray(frame_points[person, out_idx], dtype=np.float64)
            if not np.isfinite(anchor).all():
                continue
            if anchor[0] < 0 or anchor[0] >= width or anchor[1] < 0 or anchor[1] >= height:
                continue
            tip_cam = point_cam + FORCE_METERS_PER_BW * (FORCE_FLIP * force)
            if tip_cam[2] <= 1e-3:
                continue
            base_px = np.array([fx * point_cam[0] / point_cam[2] + cx,
                                fy * point_cam[1] / point_cam[2] + cy])
            tip_px = np.array([fx * tip_cam[0] / tip_cam[2] + cx,
                               fy * tip_cam[1] / tip_cam[2] + cy])
            delta = tip_px - base_px
            length = float(np.linalg.norm(delta))
            if length < 2.0:  # force points almost along the optical axis
                continue
            color = FORCE_COLORS_BGR[out_idx % len(FORCE_COLORS_BGR)]
            start = (int(round(anchor[0])), int(round(anchor[1])))
            end = (int(round(anchor[0] + delta[0])), int(round(anchor[1] + delta[1])))
            tip_length = float(np.clip(32.0 / max(length, 1.0), 0.1, 0.5))
            cv2.arrowedLine(frame, start, end, FORCE_OUTLINE_BGR,
                            FORCE_THICKNESS + 3, cv2.LINE_AA, tipLength=tip_length)
            cv2.arrowedLine(frame, start, end, color,
                            FORCE_THICKNESS, cv2.LINE_AA, tipLength=tip_length)


def _render_scene(
    scene: str,
    ds: ClimbingVideosDataset,
    probs: np.ndarray,
    points: np.ndarray,
    threshold: float,
    output_path: Path,
    labels: np.ndarray | None = None,
    label_mask: np.ndarray | None = None,
    force_data: dict[str, np.ndarray] | None = None,
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
                frame,
                probs[:, frame_position],
                points[:, frame_position],
                threshold,
                None if labels is None else labels[:, frame_position],
                None if label_mask is None else label_mask[:, frame_position],
            )
            if force_data is not None:
                _draw_force_arrows(
                    frame,
                    points[:, frame_position],
                    force_data["forces"][:, frame_position],
                    force_data["anchor_cam"][:, frame_position],
                    data["intrinsics"][frame_position],
                )
            writer.write(frame)
    finally:
        writer.release()

    valid = np.isfinite(probs)
    predicted_contact = valid & (probs >= threshold)
    record = {
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
    if labels is not None and label_mask is not None:
        active = valid & label_mask
        record.update({
            "known_extremity_labels": int(active.sum()),
            "label_contact_fraction": float(
                (labels & active).sum() / max(int(active.sum()), 1)),
            "prediction_label_agreement": float(
                ((predicted_contact == labels) & active).sum()
                / max(int(active.sum()), 1)),
        })
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs" / "climbing_videos_joint.yaml")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--overlay-labels", action="store_true",
        help="draw dataset label as the inner disk and prediction as the outer ring")
    parser.add_argument("--num-videos", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument(
        "--batch-size", type=int, default=12,
        help="windows per GPU forward; frames per forward = batch-size * T "
             "(T = data.sequence.frames_per_clip)")
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
        checkpoint.parent / (
            f"video_inference_{args.split}_{checkpoint.stem}_t{threshold_tag}"
            f"{'_gtpred' if args.overlay_labels else ''}"
        ))
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    root = _dataset_root(cfg)
    scenes = (
        list_completed_test_scenes(root)
        if args.split == "test" and args.overlay_labels
        else list_scenes(root, args.split)
    )
    selected = select_random_scenes(scenes, args.num_videos, args.seed)
    if len(selected) < args.num_videos:
        raise ValueError(
            f"requested {args.num_videos} distinct source videos, found {len(selected)}")

    print(f"[rank {rank}/{world_size}] Building model on {device} and loading "
          f"{checkpoint.name} …")
    model, _ = build_model(cfg, device)
    state = ckpt_io.load(checkpoint, model, config=cfg, map_location=device)
    model.eval()

    # Draw force arrows only for a local_world_aligned force head; other frames
    # need FK not run here (guarded like scripts/demo_climbing_videos.py).
    force_enabled = bool(cfg["model"]["force_head"]["enabled"])
    force_frame = cfg["model"]["force_head"]["frame"] if force_enabled else None
    collect_force = force_enabled and force_frame == "local_world_aligned"
    if rank == 0 and force_enabled and not collect_force:
        print(f"[force] force_head.frame={force_frame!r} != 'local_world_aligned'; "
              "force arrows need FK not run here — skipping arrows.")
    if rank == 0:
        print(f"Checkpoint epoch {state['epoch']}; threshold {args.threshold:.2f}")
        print("Selected:", ", ".join(f"{video_id}:{scene}" for video_id, scene in selected))

    local_records = []
    indexed = list(enumerate(selected, start=1))
    for index, (video_id, scene) in indexed[rank::world_size]:
        print(f"[rank {rank}] [{index}/{len(selected)}] {video_id} -> {scene}")
        ds, probs, points, force_data = _predict_scene(
            model, cfg, root, scene, args.batch_size, device,
            split_dir=args.split,
            require_labels=args.overlay_labels,
            collect_force=collect_force,
        )
        labels, label_mask = (
            _scene_ground_truth(ds._scenes[scene])
            if args.overlay_labels else (None, None)
        )
        suffix = "gtpred" if args.overlay_labels else "contacts"
        output_path = output_dir / f"{index:02d}_{scene}_{suffix}_t{threshold_tag}.mp4"
        record = _render_scene(
            scene, ds, probs, points, args.threshold, output_path,
            labels=labels, label_mask=label_mask, force_data=force_data)
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
        seq = cfg["data"]["sequence"]
        seq_len = int(seq["frames_per_clip"])
        stride = int(seq["frame_stride"])
        summary = {
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": int(state["epoch"]),
            "split": args.split,
            "threshold": args.threshold,
            "seed": args.seed,
            "selection": "one seeded-random scene chunk per distinct source video",
            "inference": (
                f"per-frame T=1" if seq_len == 1 and stride == 1 else
                f"centered sliding windows of T={seq_len} sampled frames "
                f"(stride {stride}); every source frame predicted exactly once"
            ),
            "frames_per_clip": seq_len,
            "frame_stride": stride,
            "world_size": world_size,
            "joint_names": list(EXTREMITY_4_NAMES),
            "circle_colors": {"contact": "red", "non_contact": "green"},
            "circle_encoding": (
                {"inner_disk": "dataset label", "outer_ring": "model prediction"}
                if args.overlay_labels else {"disk": "model prediction"}
            ),
            "videos": records,
        }
        if collect_force:
            summary["force_arrows"] = {
                "encoding": "one arrow per extremity: the predicted 3D force as a "
                            "metric segment at the extremity's camera-frame position, "
                            "perspective-projected through the dataset per-frame "
                            "intrinsics, attached to the anchor 2D keypoint",
                "colors": dict(zip(EXTREMITY_4_NAMES, FORCE_COLORS)),
                "units": "body weight (dimensionless)",
                "meters_per_body_weight": FORCE_METERS_PER_BW,
                "min_magnitude_bw": FORCE_MIN_BW,
                "frame": force_frame,
            }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"Done: {output_dir}")
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
