"""Lazy dataset registry and small decoded-instance cache for the viewer."""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from contact.data import (
    ClimbingImagesDataset,
    DamonDataset,
    LemonDataset,
    RichDataset,
)
# The viewer still reads the exported ClimbingVideos_v1 from disk; its loader is
# retired from the training pipeline and lives in legacy/.
from legacy.climbing_videos import ClimbingVideosDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "datasets"


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    label: str
    target: str
    splits: tuple[str, ...]
    default_split: str
    modes: tuple[str, ...] = ("frame",)
    description: str = ""

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "target": self.target,
            "splits": list(self.splits),
            "default_split": self.default_split,
            "modes": list(self.modes),
            "description": self.description,
        }


SPECS: dict[str, DatasetSpec] = {
    "climbing_videos": DatasetSpec(
        "climbing_videos", "ClimbingVideos v1", "joint", ("train", "test"), "train",
        ("frame", "sequence"),
        "Per-joint stable-contact labels. Each person-frame or deterministic clip is one instance.",
    ),
    "climbing_images": DatasetSpec(
        "climbing_images", "ClimbingImages v1", "vertex", ("all",), "all",
        description="Climbing still images with per-vertex SMPL contact labels.",
    ),
    "damon": DatasetSpec(
        "damon", "DAMON / DECO", "vertex", ("trainval", "test"), "trainval",
        description="Still images with per-vertex SMPL contact labels.",
    ),
    "lemon": DatasetSpec(
        "lemon", "LEMON / 3DIR", "vertex", ("train", "val"), "train",
        description="Still images with per-vertex SMPL-H contact labels.",
    ),
    "rich": DatasetSpec(
        "rich", "RICH", "vertex", ("train", "val", "test"), "val",
        description="BSTRO-format still images with per-vertex SMPL contact labels.",
    ),
}


class ViewerDataset:
    """Thread-safe wrapper with a deliberately tiny decoded-item LRU."""

    def __init__(self, raw: Any, spec: DatasetSpec, split: str, mode: str,
                 clip_length: int, stride: int):
        self.raw = raw
        self.spec = spec
        self.split = split
        self.mode = mode
        self.clip_length = clip_length
        self.stride = stride
        self._cache: OrderedDict[int, Any] = OrderedDict()
        self._cache_size = 4 if spec.id == "climbing_videos" else 8
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self.raw)

    def item(self, index: int) -> Any:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        with self._lock:
            if index in self._cache:
                item = self._cache.pop(index)
                self._cache[index] = item
                return item
            item = self.raw[index]
            self._cache[index] = item
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return item


class DatasetManager:
    """Validate selections and instantiate only datasets the user opens."""

    def __init__(self):
        self._datasets: OrderedDict[tuple, ViewerDataset] = OrderedDict()
        self._dataset_cache_size = 5
        self._lock = threading.RLock()

    def catalog(self) -> list[dict]:
        return [spec.as_dict() for spec in SPECS.values()]

    def get(self, dataset_id: str, split: str, mode: str = "frame",
            clip_length: int = 5, stride: int = 1) -> ViewerDataset:
        spec = SPECS.get(dataset_id)
        if spec is None:
            raise ValueError(f"unknown dataset {dataset_id!r}")
        if split not in spec.splits:
            raise ValueError(f"split must be one of {spec.splits}; got {split!r}")
        if mode not in spec.modes:
            raise ValueError(f"mode must be one of {spec.modes}; got {mode!r}")
        if not 1 <= int(clip_length) <= 16:
            raise ValueError("clip_length must be between 1 and 16")
        if not 1 <= int(stride) <= 8:
            raise ValueError("stride must be between 1 and 8")

        # Frame mode is canonicalized so UI-only sequence controls do not create
        # duplicate caches. All still-image datasets are one frame by definition.
        if mode == "frame":
            clip_length, stride = 1, 1
        key = (dataset_id, split, mode, int(clip_length), int(stride))
        with self._lock:
            if key in self._datasets:
                dataset = self._datasets.pop(key)
                self._datasets[key] = dataset
                return dataset
            raw = self._build(dataset_id, split, mode, int(clip_length), int(stride))
            dataset = ViewerDataset(raw, spec, split, mode, int(clip_length), int(stride))
            self._datasets[key] = dataset
            while len(self._datasets) > self._dataset_cache_size:
                self._datasets.popitem(last=False)
            return dataset

    @staticmethod
    def _config(name: str) -> Path:
        return CONFIG_DIR / f"{name}.yaml"

    def _build(self, dataset_id: str, split: str, mode: str,
               clip_length: int, stride: int) -> Any:
        if dataset_id == "climbing_videos":
            cfg_path = self._config("climbing_videos")
            root = yaml.safe_load(cfg_path.read_text())["data"]["root"]
            return ClimbingVideosDataset(
                root=root,
                mode="val",                  # fixed, deterministic windows for inspection
                split_dir=split,
                frames_per_clip=1 if mode == "frame" else clip_length,
                frame_stride=1 if mode == "frame" else stride,
                jitter=False,
                use_confidence_weights=False,
                require_labels=True,
            )
        if dataset_id == "climbing_images":
            return ClimbingImagesDataset.from_config(self._config("climbing_images"))
        if dataset_id == "damon":
            return DamonDataset.from_config(self._config("damon"), split=split)
        if dataset_id == "lemon":
            return LemonDataset(split=split)
        if dataset_id == "rich":
            return RichDataset(split=split)
        raise AssertionError(dataset_id)
