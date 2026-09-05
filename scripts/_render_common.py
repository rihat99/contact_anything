"""Shared plumbing for the scripts that run a checkpoint over whole scenes.

The two renderers do the same three things: resolve a scene list from the run
config's dataset yaml, run the whole-scene evaluation clip of every tracked
person through the model, and draw the result onto the corpus JPEG frames.

A test scene is run under the evaluation protocol and no other: ONE clip per
``(scene, person)`` — the longest contiguous valid run, strided like training
and capped at ``data.eval_max_frames`` — so a render shows exactly what
``scripts/evaluate.py`` scores. A train scene has no such protocol (the
whole-scene clip is eval-only, :class:`~data.base.ClipDataset` rejects it for
``split="train"``), so it is run as the training windows themselves: invalid-free
tiles of ``data.clip.frames``.

Either way only the predicted frames are written — a rendered video is the
covered clip(s), at the source fps divided by the clip stride.
"""
from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Sequence

import cv2
import numpy as np
import torch
import yaml

from data import make_collate
from data.base import Clip, ClipDataset
from data.climbing_videos import ClimbingVideosDataset
from data.transforms import crop_size
from train.predict import run_clip

REPO = Path(__file__).resolve().parents[1]
#: Opacity of a drawn mesh over the frame.
MESH_ALPHA = 0.55


def dataset_spec(cfg: dict) -> tuple[Path, int]:
    """Corpus root and contact level from the config's dataset yaml."""
    spec = _dataset_yaml(cfg)
    return Path(spec["root"]), int(spec["contact_level"])


def dataset_camera(cfg: dict) -> str:
    """The dataset yaml's ``camera`` filter (``all`` | ``static`` | ``moving``)."""
    return str(_dataset_yaml(cfg)["camera"])


def _dataset_yaml(cfg: dict) -> dict:
    entries = list(cfg["data"]["datasets"])
    if len(entries) != 1:
        raise ValueError(
            f"the renderers handle exactly one dataset; config lists {entries}")
    path = Path(entries[0])
    spec = yaml.safe_load((path if path.is_absolute() else REPO / path).read_text())
    if spec["name"] != ClimbingVideosDataset.name:
        raise ValueError(
            f"{path}: the renderers need the {ClimbingVideosDataset.name} dataset; "
            f"got {spec['name']!r}")
    return spec


def resolve_scenes(root: Path, split: str, selection: Optional[str],
                   camera: str = "all") -> list[str]:
    """Scene ids to render: all of ``split``, its first N, or a named subset.

    :param selection: ``None`` (every scene), a count (``"5"``), or a
        comma-separated list of scene ids.
    :param camera: the dataset yaml's camera filter (``all`` | ``static`` | ``moving``).
    """
    available = ClimbingVideosDataset.list_scenes(root, split, camera)
    if selection is None:
        return available
    if selection.strip().isdigit():
        count = int(selection)
        if count > len(available):
            raise ValueError(
                f"asked for {count} {split} scenes; only {len(available)} exist")
        return available[:count]
    wanted = [s for s in selection.replace(",", " ").split() if s]
    unknown = [s for s in wanted if s not in available]
    if unknown:
        raise ValueError(f"not {split} scenes of this corpus: {unknown}")
    return wanted


def shard(items: Sequence) -> tuple[list, int, int]:
    """Slice ``items`` for this torchrun rank. ``-> (mine, rank, world_size)``."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return list(items)[rank::world_size], rank, world_size


def build_dataset(
    cfg: dict, root: Path, contact_level: int, scene: str, split: str,
    load: set[str], max_frames: int,
) -> ClimbingVideosDataset:
    """One scene as clips: the whole-scene eval clip per person, or train tiles.

    ``max_frames`` caps the eval clip and is inert on the train split, whose
    clips are ``data.clip.frames`` long by construction.
    """
    return ClimbingVideosDataset(
        root,
        scenes=[scene],
        split=split,
        clip_frames=int(cfg["data"]["clip"]["frames"]),
        stride=cfg["data"]["clip"]["stride"],
        jitter=False,
        seed=int(cfg["data"]["seed"]),
        contact_level=contact_level,
        load=load,
        embedding_dir=(root / "features" / "embedding"
                       if bool(cfg["data"]["embedding_cache"]) else None),
        full_scenes=split == "test",
        max_frames=int(max_frames),
    )


def clip_batches(
    ds: ClipDataset, cfg: dict, model, device: str,
) -> Iterator[tuple[Clip, dict, dict]]:
    """Forward every clip of ``ds``. Yields ``(clip, batch, model output)``.

    The batch keeps its ``frame_index`` / ``key`` rows, which is how a caller
    maps output row ``r`` back to a source frame.
    """
    collate = make_collate(crop_size(cfg["model"]["checkpoint_path"]))
    for index, clip in enumerate(ds.clips):
        batch = collate([ds[index]])
        yield clip, batch, run_clip(model, batch, device)


def project(points_cam: np.ndarray, intr: np.ndarray) -> np.ndarray:
    """Pinhole projection of camera-frame points ``(..., 3)`` to pixels ``(..., 2)``."""
    z = np.clip(points_cam[..., 2:3], 1e-6, None)
    uv = points_cam[..., :2] / z
    return uv * np.array([intr[0, 0], intr[1, 1]]) + np.array([intr[0, 2], intr[1, 2]])


def read_frame(frames_dir: Path, position: int) -> np.ndarray:
    """Read one corpus JPEG as BGR; raises when the frame tree is incomplete."""
    path = frames_dir / f"{position:06d}.jpg"
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(path)
    return frame


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """mp4v writer at ``size = (width, height)``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size)
    if not writer.isOpened():
        raise RuntimeError(f"could not open a video writer for {path}")
    return writer


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


@contextmanager
def nan_means():
    """Silence the all-NaN-slice warning: uncovered frames are NaN by design."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def draw_mesh(img: np.ndarray, verts2d: np.ndarray, verts_cam: np.ndarray,
              faces: np.ndarray, colour) -> None:
    """Painter-sorted, lambert-shaded solid mesh alpha-blended onto ``img`` (BGR)."""
    tri2d = verts2d[faces].astype(np.float32)                   # [F, 3, 2]
    tricam = verts_cam[faces].astype(np.float32)                # [F, 3, 3]
    normals = np.cross(tricam[:, 1] - tricam[:, 0], tricam[:, 2] - tricam[:, 0])
    shade = 0.30 + 0.70 * np.clip(
        np.abs(normals[:, 2]) / (np.linalg.norm(normals, axis=1) + 1e-8), 0.0, 1.0)
    depth = tricam[..., 2].mean(axis=1)
    height, width = img.shape[:2]
    xs, ys = tri2d[..., 0], tri2d[..., 1]
    keep = (np.isfinite(tri2d).all(axis=(1, 2)) & np.isfinite(depth) & (depth > 0.05)
            & (xs.max(1) >= 0) & (xs.min(1) < width)
            & (ys.max(1) >= 0) & (ys.min(1) < height))
    order = np.argsort(-depth[keep])
    colour = np.asarray(colour, np.float32)
    overlay = img.copy()
    for points, factor in zip(tri2d[keep][order].astype(np.int32), shade[keep][order]):
        cv2.fillConvexPoly(overlay, points, tuple(float(c) for c in colour * factor))
    cv2.addWeighted(overlay, MESH_ALPHA, img, 1.0 - MESH_ALPHA, 0.0, dst=img)


def draw_keypoints(img: np.ndarray, points: np.ndarray, colour, radius: int) -> None:
    """Filled discs with a black outline at every finite ``(x, y)`` near the frame."""
    height, width = img.shape[:2]
    for x, y in points:
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if -50 <= x < width + 50 and -50 <= y < height + 50:
            cv2.circle(img, (int(x), int(y)), radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), radius, colour, -1, cv2.LINE_AA)
