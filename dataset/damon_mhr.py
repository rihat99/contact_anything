"""
PyTorch Dataset for Damon with MHR contact labels.

Loads from two separate NPZ files:
  - contact NPZ:  imgname, contact_label  (LOD-specific)
  - detect NPZ:   imgname, bbox, cam_k    (LOD-independent, optional)

Train / val / test splits are handled by passing separate npz files.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple


# LOD → vertex count (mirrors BodyConverter.LOD_VERTEX_COUNTS)
LOD_VERTEX_COUNTS = {
    0: 73639,
    1: 18439,
    2: 10661,
    3:  4899,
    4:  2461,
    5:   971,
    6:   595,
}


class DamonMHRDataset(Dataset):
    """
    Dataset for Damon images with MHR contact labels, bounding boxes, and camera parameters.

    Args:
        contact_npz_path: Path to the contact .npz file.
            Required keys:
            - 'imgname':       Array of image filenames
            - 'contact_label': Binary contact labels  (N, V)  where V = LOD vertex count
        detect_npz_path: Path to the detect .npz file (optional).
            Expected keys:
            - 'imgname': must match contact_npz imgnames
            - 'bbox':    Bounding boxes (N, 4) in [x1, y1, x2, y2]
            - 'cam_k':   Camera intrinsics (N, 3, 3)
        lod: MHR Level of Detail (0–6, default 1 → 18439 verts).
        data_root: Root directory for images. If None, tries:
            1. DAMON_DATA_ROOT environment variable
            2. Parent directory of contact_npz_path
        transform: Optional transform applied to the PIL Image before returning.

    Returns per item:
        (image, bbox, cam_k), contact_label
            image  — numpy uint8 RGB array [H, W, 3]
            bbox   — float32 tensor [4]  (x1, y1, x2, y2)
            cam_k  — float32 tensor [3, 3]
        contact_label — int64 tensor [V] (binary)
    """

    def __init__(
        self,
        contact_npz_path: str,
        detect_npz_path: Optional[str] = None,
        lod: int = 1,
        data_root: Optional[str] = None,
        transform=None,
    ):
        super().__init__()

        if lod not in LOD_VERTEX_COUNTS:
            raise ValueError(f"lod must be 0–6, got {lod}")

        self.contact_npz_path = contact_npz_path
        self.detect_npz_path = detect_npz_path
        self.lod = lod
        self.num_vertices = LOD_VERTEX_COUNTS[lod]
        self.transform = transform

        # --- Load contact NPZ ---
        print(f"Loading Damon MHR contact dataset from {contact_npz_path}")
        contact_data = np.load(contact_npz_path, allow_pickle=True)

        self.imgnames = contact_data['imgname']
        self.contact_labels = contact_data['contact_label']  # (N, V)

        num_samples = len(self.imgnames)
        assert self.contact_labels.shape == (num_samples, self.num_vertices), (
            f"Expected contact_labels shape ({num_samples}, {self.num_vertices}) "
            f"for LOD{lod}, got {self.contact_labels.shape}"
        )

        # --- Load detect NPZ (optional) ---
        if detect_npz_path is not None:
            print(f"  Loading detect data from {detect_npz_path}")
            detect_data = np.load(detect_npz_path, allow_pickle=True)

            # Sanity check: imgnames must match
            det_imgnames = detect_data['imgname']
            assert len(det_imgnames) == num_samples and (det_imgnames == self.imgnames).all(), (
                f"imgname mismatch between contact NPZ ({num_samples}) "
                f"and detect NPZ ({len(det_imgnames)})"
            )

            self.bboxes = detect_data['bbox'].astype(np.float32)   # (N, 4)
            self.cam_ks = detect_data['cam_k'].astype(np.float32)  # (N, 3, 3)
            assert self.bboxes.shape == (num_samples, 4), (
                f"Expected bbox shape ({num_samples}, 4), got {self.bboxes.shape}"
            )
            assert self.cam_ks.shape == (num_samples, 3, 3), (
                f"Expected cam_k shape ({num_samples}, 3, 3), got {self.cam_ks.shape}"
            )
            print(f"  Bounding boxes and camera intrinsics available")
            self.has_bboxes = True
            self.has_cam_ks = True
        else:
            self.bboxes = None
            self.cam_ks = None
            self.has_bboxes = False
            self.has_cam_ks = False
            print(f"  No detect NPZ — will use full-image bbox and default focal length")

        # --- Resolve data root ---
        if data_root is None:
            data_root = os.environ.get('DAMON_DATA_ROOT', None)
            if data_root is None:
                data_root = os.path.dirname(os.path.dirname(contact_npz_path))
                print(
                    f"Warning: data_root not specified. Using default: {data_root}\n"
                    f"Set DAMON_DATA_ROOT env var or pass data_root explicitly if images are missing."
                )

        self.data_root = data_root

        total_contacts = int(self.contact_labels.sum())
        print(
            f"Loaded {num_samples} samples | LOD{lod} ({self.num_vertices} verts) | "
            f"total contact vertices: {total_contacts} | "
            f"avg per sample: {total_contacts / num_samples:.1f}"
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.imgnames)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Tuple[np.ndarray, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns:
            (image, bbox, cam_k), contact_label
        """
        imgname = self.imgnames[idx]
        img_path = os.path.join(self.data_root, imgname)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load image {img_path}\n"
                f"Image name: {imgname}\n"
                f"Data root: {self.data_root}\n"
                f"Error: {e}\n"
                f"Hint: Set data_root parameter or DAMON_DATA_ROOT environment variable."
            )

        width, height = image.size

        if self.transform is not None:
            image = self.transform(image)

        # Return as numpy array (SAM-3D-Body transforms applied in data_utils.py)
        if isinstance(image, Image.Image):
            image = np.array(image, dtype=np.uint8)  # [H, W, 3] RGB uint8

        # Bounding box
        if self.has_bboxes:
            bbox = torch.from_numpy(self.bboxes[idx]).float()
        else:
            bbox = torch.tensor([0.0, 0.0, float(width), float(height)])

        # Camera intrinsics
        if self.has_cam_ks:
            cam_k = torch.from_numpy(self.cam_ks[idx]).float()
        else:
            focal = float(max(width, height))
            cam_k = torch.tensor([
                [focal, 0.0, width / 2.0],
                [0.0, focal, height / 2.0],
                [0.0, 0.0, 1.0],
            ])

        contact_label = torch.from_numpy(self.contact_labels[idx]).long()

        return (image, bbox, cam_k), contact_label

    # ------------------------------------------------------------------
    # Train / val splitting
    # ------------------------------------------------------------------

    @classmethod
    def split_train_val(
        cls,
        contact_npz_path: str,
        detect_npz_path: Optional[str] = None,
        lod: int = 1,
        val_ratio: float = 0.2,
        seed: int = 42,
        data_root: Optional[str] = None,
    ):
        """
        Load the full dataset and split into train and val subsets.

        Args:
            contact_npz_path:  Path to the contact NPZ file.
            detect_npz_path:   Path to the detect NPZ file (optional).
            lod:               MHR LOD level (default 1).
            val_ratio:         Fraction of samples to use for validation (default 0.2).
            seed:              Random seed for reproducible shuffling (default 42).
            data_root:         Root directory for images.

        Returns:
            (train_subset, val_subset) — torch.utils.data.Subset objects.
        """
        from torch.utils.data import Subset

        full_dataset = cls(
            contact_npz_path=contact_npz_path,
            detect_npz_path=detect_npz_path,
            lod=lod,
            data_root=data_root,
        )
        n = len(full_dataset)

        rng = np.random.RandomState(seed)
        shuffled = rng.permutation(n)

        n_val = max(1, int(round(n * val_ratio)))
        val_indices   = sorted(shuffled[:n_val].tolist())
        train_indices = sorted(shuffled[n_val:].tolist())

        print(
            f"Split (seed={seed}): {len(train_indices)} train, "
            f"{len(val_indices)} val (val_ratio={val_ratio})"
        )
        return Subset(full_dataset, train_indices), Subset(full_dataset, val_indices)

    # ------------------------------------------------------------------

    def get_sample_info(self, idx: int) -> dict:
        """Return metadata dictionary for a sample (debugging / statistics)."""
        contact_label = self.contact_labels[idx]
        num_contacts = int(contact_label.sum())
        info = {
            'idx': idx,
            'imgname': self.imgnames[idx],
            'num_contacts': num_contacts,
            'contact_ratio': num_contacts / self.num_vertices,
        }
        if self.has_bboxes:
            bbox = self.bboxes[idx]
            info['bbox'] = bbox.tolist()
            info['bbox_size'] = [float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])]
        if self.has_cam_ks:
            cam_k = self.cam_ks[idx]
            info['focal_length'] = [float(cam_k[0, 0]), float(cam_k[1, 1])]
            info['principal_point'] = [float(cam_k[0, 2]), float(cam_k[1, 2])]
        return info


class DamonPrecomputedDataset(Dataset):
    """Dataset that loads precomputed DINOv3 features + predictions instead of raw images.

    Features are loaded from .pt files, predictions from a merged .npz.
    This skips the backbone entirely during training.

    Returns per item:
        (feature, bbox, cam_k, ori_img_size, pred_kp2d, pred_kp3d), contact_label
            feature      — float16 tensor [C, H, W]  (e.g. [1280, 32, 32])
            bbox         — float32 tensor [4]
            cam_k        — float32 tensor [3, 3]
            ori_img_size — float32 tensor [2]  (H, W)
            pred_kp2d    — float32 tensor [70, 2] or zeros if no predictions
            pred_kp3d    — float32 tensor [70, 3] or zeros if no predictions
        contact_label — int64 tensor [V]
    """

    def __init__(
        self,
        contact_npz_path: str,
        detect_npz_path: str,
        features_dir: str,
        predictions_npz_path: Optional[str] = None,
        lod: int = 1,
        data_root: Optional[str] = None,
    ):
        super().__init__()
        if lod not in LOD_VERTEX_COUNTS:
            raise ValueError(f"lod must be 0–6, got {lod}")

        self.lod = lod
        self.num_vertices = LOD_VERTEX_COUNTS[lod]
        self.features_dir = Path(features_dir)

        # --- Load contact NPZ ---
        print(f"Loading precomputed dataset from {contact_npz_path}")
        contact_data = np.load(contact_npz_path, allow_pickle=True)
        self.imgnames = contact_data['imgname']
        self.contact_labels = contact_data['contact_label']
        num_samples = len(self.imgnames)
        assert self.contact_labels.shape == (num_samples, self.num_vertices)

        # --- Load detect NPZ (required for bbox/cam_k) ---
        detect_data = np.load(detect_npz_path, allow_pickle=True)
        self.bboxes = detect_data['bbox'].astype(np.float32)
        self.cam_ks = detect_data['cam_k'].astype(np.float32)

        # --- Resolve data_root to get original image sizes ---
        if data_root is None:
            data_root = os.environ.get('DAMON_DATA_ROOT', None)
            if data_root is None:
                data_root = os.path.dirname(os.path.dirname(contact_npz_path))
        self.data_root = data_root

        # --- Precompute original image sizes (needed for ray conditioning) ---
        # Try to load from a cache file first, otherwise compute from images
        cache_path = self.features_dir / "ori_img_sizes.npy"
        if cache_path.exists():
            self.ori_img_sizes = np.load(str(cache_path)).astype(np.float32)
            print(f"  Loaded cached ori_img_sizes from {cache_path}")
        else:
            print(f"  Computing original image sizes (one-time)...")
            self.ori_img_sizes = np.zeros((num_samples, 2), dtype=np.float32)
            for i in range(num_samples):
                img_path = os.path.join(self.data_root, self.imgnames[i])
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        self.ori_img_sizes[i] = [h, w]
                except Exception:
                    # Fallback: derive from bbox extents
                    self.ori_img_sizes[i] = [
                        max(self.bboxes[i][3], 480),
                        max(self.bboxes[i][2], 640),
                    ]
            np.save(str(cache_path), self.ori_img_sizes)
            print(f"  Cached ori_img_sizes to {cache_path}")

        # --- Load predictions NPZ (optional, for pose supervision) ---
        self.pred_kp2d = None
        self.pred_kp3d = None
        if predictions_npz_path and os.path.exists(predictions_npz_path):
            print(f"  Loading predictions from {predictions_npz_path}")
            pred_data = np.load(predictions_npz_path, allow_pickle=True)
            self.pred_kp2d = pred_data['pred_keypoints_2d'].astype(np.float32)  # [N, 70, 2]
            self.pred_kp3d = pred_data['pred_keypoints_3d'].astype(np.float32)  # [N, 70, 3]
            assert self.pred_kp2d.shape[0] == num_samples
        else:
            print(f"  No predictions NPZ — pose supervision targets will be zeros")

        # Verify features exist
        sample_feat = self.features_dir / "0000.pt"
        assert sample_feat.exists(), f"Feature file not found: {sample_feat}"

        print(f"  {num_samples} samples, features from {features_dir}")

    def __len__(self):
        return len(self.imgnames)

    def __getitem__(self, idx):
        # Load precomputed feature
        feat_path = self.features_dir / f"{idx:04d}.pt"
        feature = torch.load(str(feat_path), map_location='cpu', weights_only=True)  # [C, H, W] float16

        bbox = torch.from_numpy(self.bboxes[idx]).float()
        cam_k = torch.from_numpy(self.cam_ks[idx]).float()
        ori_img_size = torch.from_numpy(self.ori_img_sizes[idx]).float()  # [2] (H, W)
        contact_label = torch.from_numpy(self.contact_labels[idx]).long()

        if self.pred_kp2d is not None:
            pred_kp2d = torch.from_numpy(self.pred_kp2d[idx]).float()
            pred_kp3d = torch.from_numpy(self.pred_kp3d[idx]).float()
        else:
            pred_kp2d = torch.zeros(70, 2)
            pred_kp3d = torch.zeros(70, 3)

        return (feature, bbox, cam_k, ori_img_size, pred_kp2d, pred_kp3d), contact_label

    @classmethod
    def split_train_val(
        cls,
        contact_npz_path: str,
        detect_npz_path: str,
        features_dir: str,
        predictions_npz_path: Optional[str] = None,
        lod: int = 1,
        val_ratio: float = 0.2,
        seed: int = 42,
        data_root: Optional[str] = None,
    ):
        from torch.utils.data import Subset

        full_dataset = cls(
            contact_npz_path=contact_npz_path,
            detect_npz_path=detect_npz_path,
            features_dir=features_dir,
            predictions_npz_path=predictions_npz_path,
            lod=lod,
            data_root=data_root,
        )
        n = len(full_dataset)
        rng = np.random.RandomState(seed)
        shuffled = rng.permutation(n)
        n_val = max(1, int(round(n * val_ratio)))
        val_indices = sorted(shuffled[:n_val].tolist())
        train_indices = sorted(shuffled[n_val:].tolist())
        print(f"Split (seed={seed}): {len(train_indices)} train, {len(val_indices)} val")
        return Subset(full_dataset, train_indices), Subset(full_dataset, val_indices)


# ---------------------------------------------------------------------------
# Convenience helpers for building train / val / test datasets from config
# ---------------------------------------------------------------------------

def build_datasets(cfg):
    """
    Build train, val, and test dataset objects from a config node.

    Train and val come from the same combined contact npz file (CONTACT_NPZ.TRAINVAL);
    they are split by index using a fixed random seed.

    Expected config keys under cfg.DATASET:
        CONTACT_NPZ.TRAINVAL  — path to the contact npz for train+val
        CONTACT_NPZ.TEST      — path to the contact npz for test (optional)
        DETECT_NPZ.TRAINVAL   — path to the detect npz for train+val (optional)
        DETECT_NPZ.TEST       — path to the detect npz for test (optional)
        LOD                   — MHR LOD level (default 1)
        DATA_ROOT             — root directory for images (or use DAMON_DATA_ROOT env var)
        VAL_RATIO             — fraction of samples used for val (default 0.2)
        SEED                  — random seed for reproducible split (default 42)
    """
    data_root = cfg.DATASET.get('DATA_ROOT', None)
    val_ratio = cfg.DATASET.get('VAL_RATIO', 0.2)
    seed      = cfg.DATASET.get('SEED', 42)
    lod       = cfg.DATASET.get('LOD', 1)

    contact_npz = cfg.DATASET.CONTACT_NPZ
    detect_npz  = cfg.DATASET.get('DETECT_NPZ', {})

    train_ds, val_ds = DamonMHRDataset.split_train_val(
        contact_npz_path=contact_npz.TRAINVAL,
        detect_npz_path=detect_npz.get('TRAINVAL', None),
        lod=lod,
        val_ratio=val_ratio,
        seed=seed,
        data_root=data_root,
    )

    test_ds = None
    if contact_npz.get('TEST', None):
        test_ds = DamonMHRDataset(
            contact_npz_path=contact_npz.TEST,
            detect_npz_path=detect_npz.get('TEST', None),
            lod=lod,
            data_root=data_root,
        )

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Standalone test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Damon MHR Dataset")
    parser.add_argument("--contact_npz_path", type=str, required=True,
                        help="Path to the contact .npz file")
    parser.add_argument("--detect_npz_path", type=str, default=None,
                        help="Path to the detect .npz file (optional)")
    parser.add_argument("--lod", type=int, default=1, help="MHR LOD level (default 1)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root directory for images")
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()

    print("=" * 80)
    print("Testing Damon MHR Dataset")
    print("=" * 80)

    dataset = DamonMHRDataset(
        contact_npz_path=args.contact_npz_path,
        detect_npz_path=args.detect_npz_path,
        lod=args.lod,
        data_root=args.data_root,
    )
    print(f"\nDataset length: {len(dataset)}")

    for i in range(min(args.num_samples, len(dataset))):
        print(f"\nSample {i}:")
        info = dataset.get_sample_info(i)
        print(f"  Image: {info['imgname']}")
        print(f"  Contacts: {info['num_contacts']} ({info['contact_ratio']*100:.2f}%)")
        if 'bbox' in info:
            print(f"  BBox: {info['bbox']}  size: {info['bbox_size']}")
        if 'focal_length' in info:
            print(f"  fx={info['focal_length'][0]:.1f}  fy={info['focal_length'][1]:.1f}  "
                  f"cx={info['principal_point'][0]:.1f}  cy={info['principal_point'][1]:.1f}")

        try:
            (image, bbox, cam_k), contact_label = dataset[i]
            print(f"  Image shape: {image.shape}")
            print(f"  BBox: {bbox}")
            print(f"  cam_k shape: {cam_k.shape}")
            print(f"  contact_label shape: {contact_label.shape}, sum={contact_label.sum()}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 80)
