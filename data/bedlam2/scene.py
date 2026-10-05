"""BEDLAM2_our corpus: scene discovery, our train/test split, per-scene geometry.

The corpus is read directly from its pipeline tree (no exported dataset):

* ``scenes/scenes.db`` — table ``scenes`` (``scene_id``, ``video_id`` = the
  render job, ``n_frames``, ``fps``, ``width``/``height``) and table
  ``bedlam2`` (``render_job``, ``environment``, ``camera_movement``, ...).
* ``frames/<shard>/<scene>/<pos:06d>.jpg`` — pre-extracted frames; row ``k`` of
  every feature array is frame ``k``.
* ``features/gt/<shard>/<scene>/`` — ``smplx.npz`` (the BetterHuman ``q``
  trajectory), ``contacts.npz`` (35 contact frames), ``forces.npz`` (their
  world-newton wrenches and the world joints), ``gravity.npz``, ``camera.npz``
  (per-frame OpenCV extrinsics + pixel intrinsics), ``bboxes.npz`` and
  ``masks/<pos:06d>.png`` — ONE 8-bit instance map per frame, person ``p`` drawn
  as ``p + 1`` in ``object_ids`` order.

The corpus DB carries no split of its own (every scene is ``train``), so the
split is OURS: :func:`split_scenes` draws ``test_fraction`` of each render job's
scenes, so the test split covers every environment / camera style. Boxes and
masks are generated per scene; a scene without them is dropped from the listing
(:func:`list_scenes`).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from model.contact_frames import FRAMES35, ContactSet, contact_set

#: A frame is usable only when the person is mostly visible: at least this fraction
#: of the body's rendered surface is in the instance mask ...
MIN_VISIBLE_FRACTION = 0.5
#: ... and the visible box is at least this tall (px). Occluded slivers (a 5 px box
#: of a body 110 px tall) pass ``bboxes.valid`` but give a meaningless crop, and the
#: per-frame depth of such a crop dominates the lifted trajectory error.
MIN_BOX_HEIGHT_PX = 80.0


def scene_shard(scene: str) -> str:
    """Two-level ``<s[0:2]>/<s[2:4]>`` shard prefix used throughout the corpus."""
    return f"{scene[0:2]}/{scene[2:4]}"


def gt_dir(root: str | Path, scene: str) -> Path:
    """``features/gt/<shard>/<scene>`` — every ground-truth file of one scene."""
    return Path(root) / "features" / "gt" / scene_shard(scene) / scene


def scenes_by_job(root: str | Path) -> dict[str, list[str]]:
    """``{render job: sorted scene ids}`` of the whole corpus, jobs in sorted order."""
    db_path = Path(root) / "scenes" / "scenes.db"
    if not db_path.is_file():
        raise FileNotFoundError(f"no scene database at {db_path}")
    with sqlite3.connect(db_path) as db:
        rows = db.execute("SELECT video_id, scene_id FROM scenes ORDER BY scene_id")
        jobs: dict[str, list[str]] = {}
        for video_id, scene_id in rows:
            jobs.setdefault(str(video_id), []).append(str(scene_id))
    return {job: jobs[job] for job in sorted(jobs)}


def split_scenes(
    root: str | Path, *, test_fraction: float = 0.02, split_seed: int = 0,
) -> dict[str, list[str]]:
    """Our stratified split: ``max(1, round(fraction * n))`` test scenes per render job.

    One rng seeded with ``split_seed`` draws every job's test scenes, jobs in
    sorted order, so the split is a pure function of ``(corpus, fraction, seed)``
    and every job is represented in the test split.

    :returns: ``{"train": [...], "test": [...]}``, each sorted.
    """
    if not 0.0 < float(test_fraction) < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1); got {test_fraction!r}")
    rng = np.random.default_rng(int(split_seed))
    test: list[str] = []
    train: list[str] = []
    for ids in scenes_by_job(root).values():
        count = max(1, int(round(float(test_fraction) * len(ids))))
        picked = set(rng.choice(len(ids), size=min(count, len(ids)), replace=False).tolist())
        test.extend(ids[i] for i in sorted(picked))
        train.extend(ids[i] for i in range(len(ids)) if i not in picked)
    return {"train": sorted(train), "test": sorted(test)}


def has_labels(root: str | Path, scene: str) -> bool:
    """Whether the person boxes and instance masks of ``scene`` have been generated."""
    directory = gt_dir(root, scene)
    return (directory / "bboxes.npz").is_file() and (directory / "masks").is_dir()


def list_scenes(
    root: str | Path, split: str, *, test_fraction: float = 0.02, split_seed: int = 0,
    allow_missing_labels: bool = False,
) -> list[str]:
    """Scenes of ``split`` whose boxes and masks exist, sorted.

    :param allow_missing_labels: tolerate more than 1 % of the split missing
        (the boxes/masks job is still running).
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test'; got {split!r}")
    wanted = split_scenes(root, test_fraction=test_fraction, split_seed=split_seed)[split]
    kept = [scene for scene in wanted if has_labels(root, scene)]
    dropped = len(wanted) - len(kept)
    if dropped:
        share = dropped / len(wanted)
        print(f"bedlam2 {split}: {dropped}/{len(wanted)} scenes ({share:.1%}) have no "
              f"bboxes.npz / masks yet and were dropped")
        if share > 0.01 and not allow_missing_labels:
            raise RuntimeError(
                f"bedlam2 {split}: {share:.1%} of the split has no boxes / masks — set "
                "allow_missing_labels in the dataset yaml to train on what exists")
    if not kept:
        raise RuntimeError(f"bedlam2 {split}: no scene has boxes / masks")
    return kept


def load_scene(root: Path, scene: str, slots: ContactSet) -> dict:
    """Frames, masks, boxes, cameras and per-slot contact labels of one scene.

    Labels are the GT ``frame_contact`` of the 35 contact frames, supervised
    wherever the person is tracked at confidence 1 (the renderer's contacts are
    exact); under ``kindyn6`` they are folded onto the six groups (a group is in
    contact when any of its frames is).

    :returns: the scene dict :class:`~data.base.ClipDataset` indexes.
    """
    directory = gt_dir(root, scene)
    smplx = np.load(directory / "smplx.npz", allow_pickle=True)
    contacts = np.load(directory / "contacts.npz", allow_pickle=True)
    boxes = np.load(directory / "bboxes.npz", allow_pickle=True)
    camera = np.load(directory / "camera.npz", allow_pickle=True)

    n = int(smplx["num_frames"])
    object_ids = np.asarray(smplx["object_ids"], np.int64).reshape(-1)
    n_people = len(object_ids)
    intrinsics = np.asarray(camera["intrinsics_px"], np.float32)           # [N, 3, 3]
    extrinsics = np.asarray(camera["extrinsics"], np.float32)              # [N, 4, 4]
    bbox = np.asarray(boxes["bbox_visible"], np.float32)                   # [P, N, 4]

    for name, npz, ids in (("contacts", contacts, contacts["object_ids"]),
                           ("bboxes", boxes, boxes["object_ids"])):
        if not np.array_equal(np.asarray(ids, np.int64).reshape(-1), object_ids):
            raise ValueError(
                f"{scene}: {name}.object_ids {np.asarray(ids).tolist()} are not "
                f"smplx.npz's {object_ids.tolist()}")
        if int(npz["num_frames"]) != n:
            raise ValueError(
                f"{scene}: {name} has {int(npz['num_frames'])} frames, smplx.npz has {n}")
    if intrinsics.shape != (n, 3, 3) or extrinsics.shape != (n, 4, 4):
        raise ValueError(
            f"{scene}: camera arrays {intrinsics.shape}/{extrinsics.shape} do not "
            f"match {n} frames")
    if bbox.shape != (n_people, n, 4):
        raise ValueError(
            f"{scene}: bbox_visible {bbox.shape} does not match ({n_people}, {n}, 4)")
    if not bool(np.asarray(camera["metric"]).item()):
        raise ValueError(f"{scene}: camera geometry is not metric")
    if not np.isfinite(extrinsics).all():
        raise ValueError(f"{scene}: extrinsics contain non-finite values")
    frame_indices = np.asarray(camera["frame_indices"], np.int64)
    if not np.array_equal(frame_indices, np.arange(n, dtype=np.int64)):
        raise ValueError(
            f"{scene}: camera.frame_indices is not sequential 0..{n - 1}; "
            f"the frames/ tree would be misaligned")

    visible = (np.asarray(boxes["mask_area"], np.float64)
               / np.maximum(np.asarray(boxes["surface_area"], np.float64), 1.0))
    valid_mask = (
        np.asarray(smplx["valid_mask"], bool)
        & np.asarray(boxes["valid"], bool)
        & np.asarray(boxes["person_valid"], bool)[:, None]
        & np.isfinite(bbox).all(axis=-1)
        & (visible >= MIN_VISIBLE_FRACTION)
        & (bbox[..., 3] - bbox[..., 1] >= MIN_BOX_HEIGHT_PX)
    )

    names = tuple(str(x) for x in contacts["contact_frame_names"])
    if names != tuple(frame[0] for frame in FRAMES35):
        raise ValueError(f"{scene}: contact_frame_names are not the 35 contact frames")
    contact = np.asarray(contacts["frame_contact"], bool)                  # [P, N, 35]
    if contact.shape != (n_people, n, len(names)):
        raise ValueError(
            f"{scene}: frame_contact {contact.shape} does not match "
            f"({n_people}, {n}, {len(names)})")
    if not slots.uses_frames:
        # The files are always per contact FRAME; kindyn6 folds them onto the groups.
        contact = contact_set("frames35").fold_max(contact.astype(np.float32)) > 0.5
    contact_gt = contact.astype(np.float32)

    return {
        "frames_dir": root / "frames" / scene_shard(scene) / scene,
        "mask_dir": directory / "masks",
        "gt_dir": directory,
        "object_ids": object_ids,
        "frame_indices": frame_indices,
        "bbox": bbox,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "valid_mask": valid_mask,
        "fps": float(camera["fps"]),
        "contact_gt": contact_gt,                                     # [P, N, K]
        "contact_valid": np.broadcast_to(
            valid_mask[..., None], contact_gt.shape).astype(np.float32),
        "contact_conf": np.ones(contact_gt.shape, np.float32),
    }
