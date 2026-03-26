"""
DAMON dataset classes for SAM-3D-Body contact training.

DamonDataset
    Raw-image dataset. Topology ('mhr' or 'smpl') and operation mode are
    constructor parameters — no subclasses needed.

DamonPrecomputedDataset
    Loads precomputed DINOv3 features (.pt) instead of raw images.
    Always MHR topology (SAM-3D-Body specific).

Operation modes
---------------
classic
    One item per image. Returns (image, bbox, cam_k), contact_label[V].
    contact_label is the merged union of all per-object contacts.

instance_contact
    One item per (image, contact-object) pair.
    Only includes objects with ≥1 contact vertex and a valid detection.
    Requires masks_v2_dir.

instance_all
    One item per (image, detected-object) pair, all non-person objects.
    contact_label is zeros for objects with no contact.
    Requires masks_v2_dir.

Instance mode return format
---------------------------
  (inputs_dict, label_dict)

  inputs_dict:
    image / feature  — np.ndarray [H,W,3] uint8 (DamonDataset)
                       or float16 tensor [C,H,W] (DamonPrecomputedDataset)
    person_mask      — np.ndarray [H,W] bool or None
    object_mask      — np.ndarray [H,W] bool or None
    person_bbox      — float32 tensor [4]  (x1, y1, x2, y2)
    object_bbox      — float32 tensor [4], zeros if no valid detection
    cam_k            — float32 tensor [3,3]
    ori_img_size     — float32 tensor [2] (H, W) [DamonPrecomputedDataset only]
    pred_kp2d        — float32 tensor [70,2]     [DamonPrecomputedDataset only]
    pred_kp3d        — float32 tensor [70,3]     [DamonPrecomputedDataset only]

  label_dict:
    contact_label — int64 tensor [V], dense binary
    object_name   — str (e.g. 'skateboard')
    has_contact   — bool

Note: object_name (str) prevents the default DataLoader collate_fn from batching
label_dicts automatically. Pass a custom collate_fn to your DataLoader.
"""
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset

try:
    from .damon_utils import (
        LOD_VERTEX_COUNTS,
        SMPL_NUM_VERTS,
        build_instance_index,
        dense_label_from_indices,
        image_level_split,
        infer_split_from_npz_path,
        load_mask_png,
        sanitize_object_name,
    )
except ImportError:
    from damon_utils import (  # noqa: E402 — direct execution inside dataset/
        LOD_VERTEX_COUNTS,
        SMPL_NUM_VERTS,
        build_instance_index,
        dense_label_from_indices,
        image_level_split,
        infer_split_from_npz_path,
        load_mask_png,
        sanitize_object_name,
    )


# ---------------------------------------------------------------------------
# DamonDataset
# ---------------------------------------------------------------------------

class DamonDataset(Dataset):
    """
    DAMON human-contact dataset loading raw images.

    topology='mhr', lod=1  →  18 439-vertex MHR mesh (default)
    topology='smpl'        →  6 890-vertex SMPL mesh
    """

    def __init__(
        self,
        contact_npz_path: str,
        detect_npz_path: Optional[str] = None,
        topology: str = 'mhr',
        lod: int = 1,
        mode: str = 'classic',
        masks_v2_dir: Optional[str] = None,
        data_root: Optional[str] = None,
        transform=None,
    ):
        """
        Args:
            contact_npz_path: NPZ with keys 'imgname' and 'contact_label' [N, V].
            detect_npz_path:  NPZ with 'imgname', 'bbox' [N,4], 'cam_k' [N,3,3]. Optional.
            topology:         'mhr' or 'smpl'.
            lod:              MHR LOD level 0–6. Ignored for topology='smpl'.
            mode:             'classic', 'instance_contact', or 'instance_all'.
            masks_v2_dir:     Path to masks_v2 root dir. Required for instance modes.
            data_root:        Image root. Falls back to DAMON_DATA_ROOT env var, then
                              '/data3/rikhat.akizhanov/DECO'.
            transform:        Optional transform applied to PIL Image (classic mode only).
        """
        super().__init__()
        _validate_args(topology, lod, mode, masks_v2_dir)

        self.topology = topology
        self.lod = lod
        self.mode = mode
        self.masks_v2_dir = masks_v2_dir
        self.transform = transform
        self.num_vertices = LOD_VERTEX_COUNTS[lod] if topology == 'mhr' else SMPL_NUM_VERTS
        self.data_root = _resolve_data_root(data_root)

        self.imgnames, self.contact_labels = _load_contact_npz(contact_npz_path, self.num_vertices)
        self.bboxes, self.cam_ks = _load_detect_npz(detect_npz_path)

        self._instance_index = None
        if mode != 'classic':
            self._split = infer_split_from_npz_path(contact_npz_path)
            self._instance_index = build_instance_index(masks_v2_dir, self._split, mode)
            print(f"Loaded DAMON {topology.upper()} ({mode}): {len(self._instance_index)} instances")
        else:
            avg = self.contact_labels.sum() / len(self.imgnames)
            print(f"Loaded DAMON {topology.upper()} (classic): {len(self.imgnames)} samples, avg contacts: {avg:.1f}")

    def __len__(self):
        return len(self._instance_index) if self.mode != 'classic' else len(self.imgnames)

    def __getitem__(self, idx):
        if self.mode == 'classic':
            return self._getitem_classic(idx)
        return self._item_for_pair(*self._instance_index[idx])

    def _getitem_classic(self, idx):
        img = _open_image(os.path.join(self.data_root, self.imgnames[idx]))
        w, h = img.size
        if self.transform is not None:
            img = self.transform(img)
        if isinstance(img, Image.Image):
            img = np.array(img, dtype=np.uint8)
        bbox = torch.from_numpy(self.bboxes[idx]).float() if self.bboxes is not None \
            else torch.tensor([0., 0., float(w), float(h)])
        cam_k = torch.from_numpy(self.cam_ks[idx]).float() if self.cam_ks is not None \
            else _default_cam_k(w, h)
        return (img, bbox, cam_k), torch.from_numpy(self.contact_labels[idx]).long()

    def _item_for_pair(self, sample_idx, obj_order):
        meta, masks_dir = _load_meta(self.masks_v2_dir, self._split, sample_idx)
        object_name = str(list(meta['object_names'])[obj_order])
        img = np.array(_open_image(os.path.join(self.data_root, str(meta['imgname']))), dtype=np.uint8)
        h, w = img.shape[:2]
        person_mask, person_bbox = _load_person_mask(sample_idx, masks_dir, meta, h, w)
        object_mask, object_bbox = _load_object_mask(sample_idx, obj_order, object_name, masks_dir, meta)
        cam_k = torch.from_numpy(self.cam_ks[sample_idx]).float() if self.cam_ks is not None \
            else _default_cam_k(w, h)
        vert_key = f"contact_vertices_{'mhr' if self.topology == 'mhr' else 'smpl'}_{obj_order}"
        contact_label = torch.from_numpy(dense_label_from_indices(meta[vert_key], self.num_vertices))
        return (
            {'image': img, 'person_mask': person_mask, 'object_mask': object_mask,
             'person_bbox': person_bbox, 'object_bbox': object_bbox, 'cam_k': cam_k},
            {'contact_label': contact_label, 'object_name': object_name,
             'has_contact': bool(contact_label.any())},
        )

    @classmethod
    def split_train_val(
        cls,
        contact_npz_path: str,
        detect_npz_path: Optional[str] = None,
        topology: str = 'mhr',
        lod: int = 1,
        mode: str = 'classic',
        masks_v2_dir: Optional[str] = None,
        val_ratio: float = 0.2,
        seed: int = 42,
        data_root: Optional[str] = None,
    ):
        """
        Split at image level into train and val Subset objects.

        The split is deterministic (seed=42 by default) and always at the image
        level — for instance modes, every instance of an image lands in the same
        subset.
        """
        full = cls(contact_npz_path, detect_npz_path, topology=topology, lod=lod,
                   mode=mode, masks_v2_dir=masks_v2_dir, data_root=data_root)
        return _make_split(full, val_ratio, seed, mode)


# ---------------------------------------------------------------------------
# DamonPrecomputedDataset
# ---------------------------------------------------------------------------

class DamonPrecomputedDataset(Dataset):
    """
    DAMON dataset that loads precomputed DINOv3 features (.pt) instead of images.

    Skips the backbone at training time for faster iteration.
    Always MHR topology.

    Classic mode returns (tuple, contact_label) — backward-compatible:
        (feature[C,H,W], bbox[4], cam_k[3,3], ori_img_size[2],
         pred_kp2d[70,2], pred_kp3d[70,3]), contact_label[V]

    Instance modes return (inputs_dict, label_dict) — see module docstring.
    """

    def __init__(
        self,
        contact_npz_path: str,
        detect_npz_path: str,
        features_dir: str,
        predictions_npz_path: Optional[str] = None,
        lod: int = 1,
        mode: str = 'classic',
        masks_v2_dir: Optional[str] = None,
        data_root: Optional[str] = None,
    ):
        """
        Args:
            contact_npz_path:     Contact label NPZ.
            detect_npz_path:      Detect NPZ (bbox + cam_k). Required.
            features_dir:         Dir containing {idx:04d}.pt feature files.
            predictions_npz_path: Optional merged predictions NPZ with keys
                                  'pred_keypoints_2d' [N,70,2] and
                                  'pred_keypoints_3d' [N,70,3].
            lod:                  MHR LOD level (default 1 → 18 439 vertices).
            mode:                 'classic', 'instance_contact', or 'instance_all'.
            masks_v2_dir:         Required for instance modes.
            data_root:            Image root for computing original image sizes.
        """
        super().__init__()
        _validate_args('mhr', lod, mode, masks_v2_dir)

        self.lod = lod
        self.mode = mode
        self.masks_v2_dir = masks_v2_dir
        self.num_vertices = LOD_VERTEX_COUNTS[lod]
        self.features_dir = Path(features_dir)
        self.data_root = _resolve_data_root(data_root)

        self.imgnames, self.contact_labels = _load_contact_npz(contact_npz_path, self.num_vertices)
        self.bboxes, self.cam_ks = _load_detect_npz(detect_npz_path)
        n = len(self.imgnames)

        self.ori_img_sizes = self._load_or_compute_sizes(n)

        self.pred_kp2d = self.pred_kp3d = None
        if predictions_npz_path and os.path.exists(predictions_npz_path):
            pred = np.load(predictions_npz_path, allow_pickle=True)
            self.pred_kp2d = pred['pred_keypoints_2d'].astype(np.float32)  # [N, 70, 2]
            self.pred_kp3d = pred['pred_keypoints_3d'].astype(np.float32)  # [N, 70, 3]
            assert self.pred_kp2d.shape[0] == n

        assert (self.features_dir / "0000.pt").exists(), \
            f"Feature file not found: {self.features_dir / '0000.pt'}"

        self._instance_index = None
        if mode != 'classic':
            self._split = infer_split_from_npz_path(contact_npz_path)
            self._instance_index = build_instance_index(masks_v2_dir, self._split, mode)
            print(f"Loaded DamonPrecomputed ({mode}): {len(self._instance_index)} instances")
        else:
            print(f"Loaded DamonPrecomputed (classic): {n} samples from {features_dir}")

    def _load_or_compute_sizes(self, n):
        cache = self.features_dir / "ori_img_sizes.npy"
        if cache.exists():
            return np.load(str(cache)).astype(np.float32)
        print("  Computing image sizes (one-time, will be cached)...")
        sizes = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            try:
                with Image.open(os.path.join(self.data_root, self.imgnames[i])) as img:
                    w, h = img.size
                    sizes[i] = [h, w]
            except Exception:
                sizes[i] = [max(self.bboxes[i][3], 480), max(self.bboxes[i][2], 640)]
        np.save(str(cache), sizes)
        return sizes

    def _load_feature(self, sample_idx: int):
        return torch.load(
            str(self.features_dir / f"{sample_idx:04d}.pt"),
            map_location='cpu',
            weights_only=True,
        )

    def _get_predictions(self, sample_idx: int):
        if self.pred_kp2d is not None:
            return (
                torch.from_numpy(self.pred_kp2d[sample_idx]).float(),
                torch.from_numpy(self.pred_kp3d[sample_idx]).float(),
            )
        return torch.zeros(70, 2), torch.zeros(70, 3)

    def __len__(self):
        return len(self._instance_index) if self.mode != 'classic' else len(self.imgnames)

    def __getitem__(self, idx):
        if self.mode == 'classic':
            return self._getitem_classic(idx)
        return self._item_for_pair(*self._instance_index[idx])

    def _getitem_classic(self, idx):
        pred_kp2d, pred_kp3d = self._get_predictions(idx)
        return (
            self._load_feature(idx),
            torch.from_numpy(self.bboxes[idx]).float(),
            torch.from_numpy(self.cam_ks[idx]).float(),
            torch.from_numpy(self.ori_img_sizes[idx]).float(),
            pred_kp2d,
            pred_kp3d,
        ), torch.from_numpy(self.contact_labels[idx]).long()

    def _item_for_pair(self, sample_idx, obj_order):
        meta, masks_dir = _load_meta(self.masks_v2_dir, self._split, sample_idx)
        object_name = str(list(meta['object_names'])[obj_order])
        h, w = int(self.ori_img_sizes[sample_idx][0]), int(self.ori_img_sizes[sample_idx][1])
        person_mask, person_bbox = _load_person_mask(sample_idx, masks_dir, meta, h, w)
        object_mask, object_bbox = _load_object_mask(sample_idx, obj_order, object_name, masks_dir, meta)
        pred_kp2d, pred_kp3d = self._get_predictions(sample_idx)
        contact_label = torch.from_numpy(
            dense_label_from_indices(meta[f"contact_vertices_mhr_{obj_order}"], self.num_vertices)
        )
        return (
            {
                'feature': self._load_feature(sample_idx),
                'person_mask': person_mask,
                'object_mask': object_mask,
                'person_bbox': person_bbox,
                'object_bbox': object_bbox,
                'cam_k': torch.from_numpy(self.cam_ks[sample_idx]).float(),
                'ori_img_size': torch.from_numpy(self.ori_img_sizes[sample_idx]).float(),
                'pred_kp2d': pred_kp2d,
                'pred_kp3d': pred_kp3d,
            },
            {'contact_label': contact_label, 'object_name': object_name,
             'has_contact': bool(contact_label.any())},
        )

    @classmethod
    def split_train_val(
        cls,
        contact_npz_path: str,
        detect_npz_path: str,
        features_dir: str,
        predictions_npz_path: Optional[str] = None,
        lod: int = 1,
        mode: str = 'classic',
        masks_v2_dir: Optional[str] = None,
        val_ratio: float = 0.2,
        seed: int = 42,
        data_root: Optional[str] = None,
    ):
        """Split at image level into train and val Subset objects."""
        full = cls(contact_npz_path, detect_npz_path, features_dir,
                   predictions_npz_path=predictions_npz_path, lod=lod,
                   mode=mode, masks_v2_dir=masks_v2_dir, data_root=data_root)
        return _make_split(full, val_ratio, seed, mode)


# ---------------------------------------------------------------------------
# Config-driven factory
# ---------------------------------------------------------------------------

def build_datasets(cfg):
    """
    Build (train, val, test) datasets from a config node.

    Expected cfg.DATASET keys:
        CONTACT_NPZ.TRAINVAL  — trainval contact NPZ
        CONTACT_NPZ.TEST      — test contact NPZ (optional)
        DETECT_NPZ.TRAINVAL   — trainval detect NPZ (optional for DamonDataset)
        DETECT_NPZ.TEST       — test detect NPZ
        LOD                   — MHR LOD (default 1)
        MODE                  — 'classic', 'instance_contact', 'instance_all' (default 'classic')
        MASKS_V2_DIR          — masks_v2 root (required if MODE != 'classic')
        DATA_ROOT             — image root
        VAL_RATIO             — val fraction (default 0.2)
        SEED                  — random seed (default 42)
        FEATURES_DIR          — precomputed features dir; enables DamonPrecomputedDataset
        PREDICTIONS_NPZ       — merged predictions NPZ (optional, for DamonPrecomputed)
    """
    lod = cfg.DATASET.get('LOD', 1)
    mode = cfg.DATASET.get('MODE', 'classic')
    masks_v2_dir = cfg.DATASET.get('MASKS_V2_DIR', None)
    data_root = cfg.DATASET.get('DATA_ROOT', None)
    val_ratio = cfg.DATASET.get('VAL_RATIO', 0.2)
    seed = cfg.DATASET.get('SEED', 42)
    contact_npz = cfg.DATASET.CONTACT_NPZ
    detect_npz = cfg.DATASET.get('DETECT_NPZ', {})
    features_dir = cfg.DATASET.get('FEATURES_DIR', None)
    predictions_npz = cfg.DATASET.get('PREDICTIONS_NPZ', None)

    split_kw = dict(lod=lod, mode=mode, masks_v2_dir=masks_v2_dir,
                    val_ratio=val_ratio, seed=seed, data_root=data_root)

    if features_dir:
        train, val = DamonPrecomputedDataset.split_train_val(
            contact_npz.TRAINVAL, detect_npz.get('TRAINVAL'), features_dir,
            predictions_npz_path=predictions_npz, **split_kw,
        )
        test = DamonPrecomputedDataset(
            contact_npz.TEST, detect_npz.get('TEST'),
            features_dir.replace('trainval', 'test'),
            predictions_npz_path=(predictions_npz.replace('trainval', 'test') if predictions_npz else None),
            lod=lod, mode=mode, masks_v2_dir=masks_v2_dir, data_root=data_root,
        ) if contact_npz.get('TEST') else None
    else:
        train, val = DamonDataset.split_train_val(
            contact_npz.TRAINVAL, detect_npz.get('TRAINVAL'), **split_kw,
        )
        test = DamonDataset(
            contact_npz.TEST, detect_npz.get('TEST'),
            lod=lod, mode=mode, masks_v2_dir=masks_v2_dir, data_root=data_root,
        ) if contact_npz.get('TEST') else None

    return train, val, test


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_args(topology, lod, mode, masks_v2_dir):
    if topology not in ('mhr', 'smpl'):
        raise ValueError(f"topology must be 'mhr' or 'smpl', got '{topology}'")
    if mode not in ('classic', 'instance_contact', 'instance_all'):
        raise ValueError(f"mode must be 'classic', 'instance_contact', or 'instance_all', got '{mode}'")
    if topology == 'mhr' and lod not in LOD_VERTEX_COUNTS:
        raise ValueError(f"lod must be 0–6, got {lod}")
    if mode != 'classic' and masks_v2_dir is None:
        raise ValueError(f"masks_v2_dir is required for mode='{mode}'")


def _resolve_data_root(data_root):
    return data_root or os.environ.get('DAMON_DATA_ROOT', '/data3/rikhat.akizhanov/DECO')


def _load_contact_npz(path, num_vertices):
    data = np.load(path, allow_pickle=True)
    imgnames = data['imgname']
    labels = data['contact_label']
    if labels.dtype != np.int64:
        labels = (labels > 0.5).astype(np.int64)
    assert labels.shape == (len(imgnames), num_vertices), (
        f"Expected contact_label ({len(imgnames)}, {num_vertices}), got {labels.shape}"
    )
    return imgnames, labels


def _load_detect_npz(path):
    if path is None:
        return None, None
    data = np.load(path, allow_pickle=True)
    return data['bbox'].astype(np.float32), data['cam_k'].astype(np.float32)


def _open_image(path: str) -> Image.Image:
    try:
        return Image.open(path).convert('RGB')
    except Exception as e:
        raise FileNotFoundError(f"Could not load image '{path}': {e}")


def _default_cam_k(width, height):
    f = float(max(width, height))
    return torch.tensor([[f, 0., width/2.], [0., f, height/2.], [0., 0., 1.]])


def _load_meta(masks_v2_dir, split, sample_idx):
    """Return (metadata_npz, masks_dir_path)."""
    sample_dir = Path(masks_v2_dir) / split / f"{sample_idx:04d}"
    meta = np.load(str(sample_dir / "metadata.npz"), allow_pickle=True)
    return meta, sample_dir / "masks"


def _load_person_mask(sample_idx, masks_dir, meta, img_h, img_w):
    mask = load_mask_png(masks_dir / f"{sample_idx:04d}_person_0.png")
    best = int(meta["best_detection_0"])
    bbox = torch.from_numpy(meta["bboxes_0"][best].astype(np.float32)) if best >= 0 \
        else torch.tensor([0., 0., float(img_w), float(img_h)])
    return mask, bbox


def _load_object_mask(sample_idx, obj_order, object_name, masks_dir, meta):
    best = int(meta[f"best_detection_{obj_order}"])
    if best < 0:
        return None, torch.zeros(4)
    safe = sanitize_object_name(object_name)
    mask = load_mask_png(masks_dir / f"{sample_idx:04d}_{safe}_{best}.png")
    bbox = torch.from_numpy(meta[f"bboxes_{obj_order}"][best].astype(np.float32))
    return mask, bbox


def _make_split(full_dataset, val_ratio, seed, mode):
    """Return (train_subset, val_subset) split at image level."""
    train_img, val_img = image_level_split(len(full_dataset.imgnames), val_ratio=val_ratio, seed=seed)
    if mode == 'classic':
        return Subset(full_dataset, train_img), Subset(full_dataset, val_img)
    train_set, val_set = set(train_img), set(val_img)
    train_flat = [i for i, (s, _) in enumerate(full_dataset._instance_index) if s in train_set]
    val_flat   = [i for i, (s, _) in enumerate(full_dataset._instance_index) if s in val_set]
    print(f"Split (seed={seed}): {len(train_flat)} train, {len(val_flat)} val instances")
    return Subset(full_dataset, train_flat), Subset(full_dataset, val_flat)
