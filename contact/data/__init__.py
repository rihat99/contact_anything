"""Per-vertex contact datasets.

All loaders return the same dict schema. ``contact`` is a float32 tensor
whose length matches the body model topology — 6890 for SMPL-family
datasets, 10475 for SMPL-X. The optional ``vertex_topology`` field
("smpl" or "smplx") lets downstream code pick the right template; it
defaults to ``"smpl"`` when absent.

    {
        "image":   uint8 ndarray [H, W, 3] RGB  (or None for label-only access),
        "contact": float32 tensor [V],
        "key":     str,
        "dataset": str,
        "mask":    ndarray or None,
        "bbox":    ndarray [4] or None,
        "focal":   float or None,
        "vertex_topology": "smpl" | "smplx"  (optional, defaults to "smpl"),
    }

``ClimbingImagesDataset`` (SMPL, 6890) additionally carries converted SMPL
pose params and the depth-fit errors recorded by the BetterImageReconstruction
pipeline; it reads a prebuilt dataset (``scripts/build_climbing_images.py``).
See ``climbing_images.py`` for the full schema.
"""

from .climbing_corpus import ClimbingCorpusDataset
from .climbing_images import ClimbingImagesDataset
from .climbing_videos import ClimbingVideosDataset
from .damon import DamonDataset
from .lemon import LemonDataset
from .rich import RichDataset

__all__ = [
    "ClimbingCorpusDataset",
    "ClimbingImagesDataset",
    "ClimbingVideosDataset",
    "DamonDataset",
    "LemonDataset",
    "RichDataset",
]
