"""ClimbingImages_v1 dataset — image + per-vertex SMPL (6890) contact labels.

Self-contained and prebuilt by ``dataset/build_climbing_v1.py``: no DB, no
BetterImageReconstruction sidecars, no SMPL-X conversion at load time — just
``metadata.npz`` (all labels stacked by index) plus one jpg + one png per
climber-item. The builder is where filtering and SMPL-X -> SMPL conversion
happen; see it (and ``dataset_info.json`` in the dataset dir) for provenance.

Per item::

    {
        "image":   uint8 [H, W, 3],
        "contact": float32 tensor [6890]     (SMPL contact_surface),
        "key":     str  "{sha}#{climber}",
        "dataset": "climbing",
        "mask":    uint8 [H, W]              (this climber's person mask),
        "bbox":    float32 [4]               (xyxy, from mask),
        "focal":   float                     (cam_int[0, 0], pixels),
        "cam_int": float32 [3, 3]            (pixel-scale intrinsics),
        "vertex_topology": "smpl",
        "smpl":    {betas[10], body_pose[69], global_orient[3], transl[3]},
        "depth_errors": {sapiens_*, body_*, stitch_*},
    }
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

DEFAULT_ROOT = "/data3/rikhat.akizhanov/datasets/ClimbingImages_v1"

_SMPL_KEYS = ("betas", "body_pose", "global_orient", "transl")
_DEPTH_KEYS = ("sapiens_scale", "sapiens_shift", "sapiens_inliers",
               "body_rmse", "body_inliers", "body_valid",
               "stitch_rmse", "stitch_err_local", "stitch_err_global", "stitch_ok")


class ClimbingImagesDataset(Dataset):
    """One climber-item per index, read from the prebuilt v1 dataset."""

    def __init__(self, root: str = DEFAULT_ROOT):
        super().__init__()
        self.root = Path(root)
        meta = self.root / "metadata.npz"
        if not meta.is_file():
            raise FileNotFoundError(
                f"{meta} not found — build it with dataset/build_climbing_v1.py")
        d = np.load(meta, allow_pickle=True)
        self.m = {k: d[k] for k in d.files}
        self.num_vertices = int(self.m["num_vertices"])

    @classmethod
    def from_config(cls, config) -> "ClimbingImagesDataset":
        if isinstance(config, (str, Path)):
            config = yaml.safe_load(Path(config).read_text())
        return cls(root=config["data"]["root"])

    def __len__(self) -> int:
        return len(self.m["key"])

    def __getitem__(self, idx: int) -> dict:
        m = self.m
        img  = np.array(Image.open(self.root / "images" / f"{idx:05d}.jpg").convert("RGB"), np.uint8)
        mask = np.array(Image.open(self.root / "masks" / f"{idx:05d}.png"), np.uint8)
        contact = np.unpackbits(m["contact"][idx])[:self.num_vertices].astype(np.float32)
        cam = m["cam_int"][idx].astype(np.float32)
        return {
            "image":           img,
            "contact":         torch.from_numpy(contact),
            "key":             str(m["key"][idx]),
            "dataset":         "climbing",
            "mask":            mask,
            "bbox":            m["bbox"][idx].astype(np.float32),
            "focal":           float(cam[0, 0]),
            "cam_int":         cam,
            "vertex_topology": "smpl",
            "smpl":            {k: m[k][idx].astype(np.float32) for k in _SMPL_KEYS},
            "depth_errors":    {k: m[k][idx].item() for k in _DEPTH_KEYS},
        }
