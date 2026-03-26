"""
Shared utilities for DAMON dataset classes.

Provides low-level helpers for mask loading, contact label conversion,
instance index construction, and split utilities used by all DAMON datasets.
"""
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOD_VERTEX_COUNTS = {
    0: 73639,
    1: 18439,
    2: 10661,
    3:  4899,
    4:  2461,
    5:   971,
    6:   595,
}

SMPL_NUM_VERTS = 6890


# ---------------------------------------------------------------------------
# Mask / name helpers
# ---------------------------------------------------------------------------

def load_mask_png(path) -> Optional[np.ndarray]:
    """Load a binary mask PNG as bool [H, W]. Returns None if file missing or unreadable."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    return gray > 127


def sanitize_object_name(name: str) -> str:
    """Convert an object name to a safe filename component (lowercase, underscores only)."""
    return re.sub(r'[^a-z0-9_]', '', name.lower().replace(' ', '_'))


# ---------------------------------------------------------------------------
# Contact label helpers
# ---------------------------------------------------------------------------

def dense_label_from_indices(indices: np.ndarray, n_verts: int) -> np.ndarray:
    """Convert a sparse vertex index array to a dense binary int64 array [n_verts]."""
    label = np.zeros(n_verts, dtype=np.int64)
    if len(indices) > 0:
        label[indices.astype(np.int64)] = 1
    return label


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def infer_split_from_npz_path(npz_path: str) -> str:
    """
    Infer dataset split ('trainval' or 'test') from NPZ filename.

    Expects the filename stem to contain 'trainval' or 'test'.
    Examples:
      hot_dca_trainval_contact_lod1.npz  →  'trainval'
      hot_dca_test.npz                   →  'test'
    """
    stem = Path(npz_path).stem
    if 'trainval' in stem:
        return 'trainval'
    if 'test' in stem:
        return 'test'
    raise ValueError(
        f"Cannot infer split from '{npz_path}'. "
        "Filename must contain 'trainval' or 'test'."
    )


def image_level_split(n_samples: int, val_ratio: float = 0.2, seed: int = 42):
    """
    Compute reproducible train / val image-index splits.

    Returns:
        (train_indices, val_indices) — sorted lists of integer indices.
    """
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(n_samples)
    n_val = max(1, int(round(n_samples * val_ratio)))
    val_indices = sorted(shuffled[:n_val].tolist())
    train_indices = sorted(shuffled[n_val:].tolist())
    return train_indices, val_indices


# ---------------------------------------------------------------------------
# Instance index
# ---------------------------------------------------------------------------

def build_instance_index(
    masks_v2_dir: str,
    split: str,
    mode: str,
) -> list:
    """
    Build a flat list of (sample_idx, obj_order) pairs for instance-mode datasets.

    Args:
        masks_v2_dir: Root of the masks_v2 directory.
        split:        'trainval' or 'test'.
        mode:         'instance_contact' or 'instance_all'.
                      instance_contact — only objects with non-empty contact vertices
                                         and a valid best detection (>= 0).
                      instance_all     — every non-person detected object.

    Returns:
        Sorted list of (sample_idx, obj_order) integer tuples.

    Result is cached to {masks_v2_dir}/{split}/instance_index_{mode}.npy.
    Delete the cache file to force a rebuild.
    """
    split_dir = Path(masks_v2_dir) / split
    cache_path = split_dir / f"instance_index_{mode}.npy"

    if cache_path.exists():
        arr = np.load(str(cache_path))
        return [tuple(int(x) for x in row) for row in arr]

    index = []
    for sample_dir in sorted(split_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        try:
            sample_idx = int(sample_dir.name)
        except ValueError:
            continue

        meta_path = sample_dir / "metadata.npz"
        if not meta_path.exists():
            continue

        meta = np.load(str(meta_path), allow_pickle=True)
        n_objs = len(meta["object_names"])

        for i in range(1, n_objs):  # skip index 0 (person)
            if mode == 'instance_contact':
                cv_smpl = meta[f"contact_vertices_smpl_{i}"]
                best_det = int(meta[f"best_detection_{i}"])
                if len(cv_smpl) > 0 and best_det >= 0:
                    index.append((sample_idx, i))
            else:  # instance_all
                index.append((sample_idx, i))

    index.sort()

    if index:
        np.save(str(cache_path), np.array(index, dtype=np.int32))

    return index
