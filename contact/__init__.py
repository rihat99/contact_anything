"""Contact-head library for the SAM-3D-Body fork.

Model build + freeze/eval pinning (``model``), loss (``losses``),
checkpoint I/O (``checkpoint``), and the per-vertex contact datasets
(``data``). The datasets are re-exported here for convenience.
"""

from .data import (
    ClimbingImagesDataset,
    DamonDataset,
    LemonDataset,
    RichDataset,
)

__all__ = [
    "ClimbingImagesDataset",
    "DamonDataset",
    "LemonDataset",
    "RichDataset",
]
