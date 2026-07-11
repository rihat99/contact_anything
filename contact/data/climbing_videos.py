"""ClimbingVideos_v1 — per-joint (22) contact over windows of video frames.

Dataset root ``/data3/rikhat.akizhanov/datasets/ClimbingVideos_v1`` (schema in
``dataset_info.json``). Only the ``train/`` directory is used here; it is split
into train/val by *source video* (see :func:`contact.data.splits.group_train_val_split`).
The ``test/`` directory holds manual annotations; a scene still marked ``pending``
raises when its labels are requested, while a *completed* scene supervises only
the joints the annotator labelled (``valid_mask`` & ``annotated_22``).

An **item** is one ``(scene, person, window)`` of ``T`` frames at a given stride.
Windows tile each scene with step ``T * stride``; the train split jitters the
window start *statelessly* from ``(seed, epoch, item_index)`` (resume-safe with
persistent workers) via :meth:`set_epoch`, the val split uses the fixed tiles.
Windows in which the person is invalid (``valid_mask``) for more than half of the
frames are skipped.

``__getitem__`` returns a **clip**: a list of ``T`` per-frame dicts, each with the
full RGB image, the person mask (or ``None`` when the file is missing), the xyxy
bbox, the camera intrinsics, the per-joint contact + supervision mask, and
``frame_pos_sec`` (elapsed seconds from the window's first frame).  Raw
``joint_confidence`` and the boolean-valued ``joint_supervised`` mask are also
returned for inspection; ``joint_mask`` remains the loss mask and is confidence-
weighted only when ``use_confidence_weights=True``. The mask is 0 for invalid
frames, so those frames contribute nothing to the loss.

Label semantics: train labels cover all 22 joints (no observable subset) and are
motion-gated *stable* contact — see :mod:`contact.targets` for why they differ
from still-image derived joint contact.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

from ..targets import NUM_BODY_22

DEFAULT_ROOT = "/data3/rikhat.akizhanov/datasets/ClimbingVideos_v1"


def list_scenes(root: str | Path, split_dir: str = "train") -> list[str]:
    """Sorted scene ids under ``{root}/{split_dir}`` that carry a labels/inputs npz."""
    base = Path(root) / split_dir
    label_file = "labels.npz" if split_dir == "train" else "inputs.npz"
    return sorted(s.name for s in base.iterdir() if s.is_dir() and (s / label_file).is_file())


class ClimbingVideosDataset(Dataset):
    """Windowed per-joint contact clips from one split of ClimbingVideos_v1."""

    supervised_targets = frozenset({"joint"})
    topology = None            # joint target is not bound to a vertex topology
    name = "climbing_videos"

    def __init__(
        self,
        root: str = DEFAULT_ROOT,
        scenes: Optional[Sequence[str]] = None,
        mode: str = "train",                # "train" (jittered) | "val" (fixed tiles)
        split_dir: str = "train",
        frames_per_clip: int = 8,
        frame_stride: int = 2,
        jitter: bool = True,
        seed: int = 42,
        use_confidence_weights: bool = False,
        require_labels: bool = True,
    ):
        super().__init__()
        if mode not in ("train", "val"):
            raise ValueError(f"mode must be 'train' or 'val'; got {mode!r}")
        if split_dir not in ("train", "test"):
            raise ValueError(f"split_dir must be 'train' or 'test'; got {split_dir!r}")
        self.root = Path(root)
        self.split_dir = split_dir
        self.mode = mode
        self.T = int(frames_per_clip)
        self.stride = int(frame_stride)
        self.jitter = bool(jitter) and mode == "train"
        self.seed = int(seed)
        self.use_confidence_weights = bool(use_confidence_weights)
        self._epoch = 0

        if scenes is None:
            scenes = list_scenes(root, split_dir)

        self._scenes: dict[str, dict] = {}
        self._items: list[tuple[str, int, int, int]] = []   # (scene, person, base_start, jitter_range)
        span = (self.T - 1) * self.stride
        step = self.T * self.stride
        for scene in scenes:
            data = self._load_scene(scene, require_labels)
            self._scenes[scene] = data
            num_frames = len(data["frame_indices"])
            valid_mask = data["valid_mask"]                 # [P, N] bool
            max_start = num_frames - 1 - span
            if max_start < 0:
                continue                                    # scene too short for one window
            for person in range(valid_mask.shape[0]):
                bases = list(range(0, max_start + 1, step))
                # Val windows are the scored tiles: when the stride tiling leaves a
                # tail (max_start not hit), append a terminal window so those frames
                # are covered too. Duplicate frames are allowed (val may count a few
                # boundary frames twice — acceptable, documented).
                if self.mode == "val" and bases and bases[-1] != max_start:
                    bases.append(max_start)
                for base in bases:
                    positions = base + np.arange(self.T) * self.stride
                    if valid_mask[person, positions].mean() < 0.5:
                        continue                            # >50% invalid -> skip window
                    jitter_range = max(1, min(step, max_start - base + 1))
                    self._items.append((scene, person, base, jitter_range))

    # ------------------------------------------------------------------ loading

    def _load_scene(self, scene: str, require_labels: bool) -> dict:
        scene_dir = self.root / self.split_dir / scene
        annotated = None
        if self.split_dir == "train":
            npz = np.load(scene_dir / "labels.npz", allow_pickle=True)
            joint_contact = npz["joint_contact_22"]
            contact_conf = npz["contact_conf_22"]
        else:
            npz = np.load(scene_dir / "inputs.npz", allow_pickle=True)
            joint_contact = contact_conf = None
            if require_labels:
                contacts = np.load(scene_dir / "contacts.npz", allow_pickle=True)
                if bool(contacts["pending"]):
                    raise RuntimeError(
                        f"{scene}: contacts.npz is a pending placeholder (all-zero) — "
                        f"joint labels are unavailable for the test split")
                joint_contact = contacts["joint_contact_22"]
                # Completed manual annotations are tri-state: only the joints the
                # annotator labelled carry supervision (valid_mask & annotated_22
                # downstream); unannotated entries are ignored, not false negatives.
                annotated = contacts["annotated_22"].astype(bool)
                # The completed test set does not ship confidence weights.
                if self.use_confidence_weights and "contact_conf_22" in contacts.files:
                    contact_conf = contacts["contact_conf_22"]
        return {
            "dir": scene_dir,
            "object_ids": npz["object_ids"].astype(np.int64),
            "frame_indices": npz["frame_indices"].astype(np.int64),
            "bbox": npz["bbox"].astype(np.float32),          # [P, N, 4] xyxy
            "intrinsics": npz["intrinsics"].astype(np.float32),  # [N, 3, 3]
            "valid_mask": npz["valid_mask"].astype(bool),    # [P, N]
            "fps": float(npz["fps"]),
            "joint_contact": joint_contact,                  # [P, N, 22] bool or None
            "contact_conf": contact_conf,                    # [P, N, 22] f32 or None
            "annotated": annotated,                          # [P, N, 22] bool or None (test)
        }

    # ------------------------------------------------------------------ epoch / jitter

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used by the stateless train-window jitter."""
        self._epoch = int(epoch)

    def _window_start(self, base: int, jitter_range: int, item_index: int) -> int:
        if not self.jitter:
            return base
        rng = np.random.default_rng([self.seed, self._epoch, item_index])
        return base + int(rng.integers(0, jitter_range))

    # ------------------------------------------------------------------ access

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> list[dict]:
        scene, person, base, jitter_range = self._items[index]
        data = self._scenes[scene]
        start = self._window_start(base, jitter_range, index)
        positions = start + np.arange(self.T) * self.stride
        # Validity was checked at `base`, but jitter picks a different start; if the
        # jittered window is >50% invalid fall back deterministically to `base`
        # (guaranteed >=50% valid at build), so training batches are never mostly
        # masked. Deterministic given (seed, epoch, index).
        if start != base and data["valid_mask"][person, positions].mean() < 0.5:
            start = base
            positions = base + np.arange(self.T) * self.stride

        oid = int(data["object_ids"][person])
        mask_dir = data["dir"] / "masks" / f"{oid:02d}"
        frames_dir = data["dir"] / "frames"
        frame_indices = data["frame_indices"]
        fps = data["fps"]
        start_time = float(frame_indices[start])

        clip = []
        for pos in positions:
            pos = int(pos)
            image = np.array(Image.open(frames_dir / f"{pos:06d}.jpg").convert("RGB"), np.uint8)
            mask_path = mask_dir / f"{pos:06d}.png"
            mask = np.array(Image.open(mask_path), np.uint8) if mask_path.is_file() else None

            valid = bool(data["valid_mask"][person, pos])
            if data["joint_contact"] is None:              # require_labels=False -> no labels
                joint_gt = np.zeros(NUM_BODY_22, np.float32)
                joint_supervised = np.zeros(NUM_BODY_22, np.float32)
                joint_confidence = np.zeros(NUM_BODY_22, np.float32)
            else:
                joint_gt = data["joint_contact"][person, pos].astype(np.float32)      # [22]
                joint_supervised = np.full(NUM_BODY_22, float(valid), dtype=np.float32)
                if data["annotated"] is not None:          # completed test: ignore unannotated joints
                    joint_supervised *= data["annotated"][person, pos].astype(np.float32)
                if data["contact_conf"] is None:
                    joint_confidence = np.ones(NUM_BODY_22, np.float32)
                else:
                    joint_confidence = np.clip(
                        data["contact_conf"][person, pos].astype(np.float32), 0.0, 1.0)

            # Keep the training contract intact: ``joint_mask`` is the score mask,
            # optionally confidence-weighted.  The two explicit fields let tools
            # such as the dataset viewer distinguish an unannotated joint from a
            # supervised label whose confidence happens to be zero.
            joint_mask = joint_supervised.copy()
            if self.use_confidence_weights:
                joint_mask *= joint_confidence

            clip.append({
                "image": image,
                "mask": mask,
                "bbox": data["bbox"][person, pos],                                    # [4] xyxy
                "cam_int": data["intrinsics"][pos],                                   # [3, 3]
                "joint_contact": torch.from_numpy(joint_gt),
                "joint_mask": torch.from_numpy(joint_mask),
                "joint_supervised": torch.from_numpy(joint_supervised),
                "joint_confidence": torch.from_numpy(joint_confidence),
                "frame_pos_sec": (float(frame_indices[pos]) - start_time) / fps,
                "frame_position": pos,
                "frame_index": int(frame_indices[pos]),
                "frame_valid": valid,
                "key": f"{scene}#{oid}@{pos}",
                "dataset": "climbing_videos",
            })
        return clip
