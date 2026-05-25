"""SMPL-topology contact datasets (6890 vertices).

All loaders return the same dict schema::

    {
        "image":   uint8 ndarray [H, W, 3] RGB  (or None for label-only access),
        "contact": float32 tensor [6890],
        "key":     str,
        "dataset": str,
        "mask":    None,    # placeholder
        "bbox":    ndarray [4] or None,
        "focal":   float or None,
    }
"""

from .contact import ContactDataset
from .damon import DamonDataset
from .lemon import LemonDataset
from .rich import RichDataset

__all__ = ["ContactDataset", "DamonDataset", "LemonDataset", "RichDataset"]
