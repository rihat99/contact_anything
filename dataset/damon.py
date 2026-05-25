"""DAMON / DECO dataset — image + per-vertex SMPL (6890) contact labels.

Loads from the original DECO release NPZ. Each split is one NPZ:
  ``hot_dca_trainval.npz`` and ``hot_dca_test.npz`` under
  ``datasets/Release_Datasets/damon/``. Image paths inside the NPZ are
  relative to the dataset root (``datasets/HOT-Annotated/images/...``).

If ``masks_dir`` is set, a precomputed person mask is loaded from
``{masks_dir}/{split}/{idx:06d}.png`` (uint8 binary). When the file
isn't there the field is ``None``. ``bbox`` (xyxy, float32) is
derived from the mask whenever the mask is present so downstream
training has a clean person crop without a separate detect npz.

If ``cam_params_dir`` is set, MoGe2-estimated intrinsics are loaded
from ``{cam_params_dir}/{split}.npz`` (keys: ``cam_int`` ``(N, 3, 3)``
float32 — absolute-pixel K, ``image_size`` ``(N, 2)`` int32 — ``(H, W)``).
The ``focal`` field is overridden with ``K[0, 0]`` and a new
``cam_int`` field is added.

Returns a dict that matches the other loaders in this package:
    {
        "image":   uint8 ndarray [H, W, 3],
        "contact": float32 tensor [6890],
        "key":     str,
        "dataset": "damon",
        "mask":    uint8 ndarray [H, W] or None,
        "bbox":    float32 ndarray [4] (xyxy) or None,
        "focal":   float or None,
        "cam_int": float32 ndarray [3, 3] or None,
    }
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

NUM_VERTICES = 6890
DEFAULT_ROOT = "/data3/rikhat.akizhanov/datasets/DECO"
SPLIT_NPZ = {
    "trainval": "datasets/Release_Datasets/damon/hot_dca_trainval.npz",
    "test":     "datasets/Release_Datasets/damon/hot_dca_test.npz",
}


class DamonDataset(Dataset):
    def __init__(
        self,
        root: str = DEFAULT_ROOT,
        split: str = "trainval",
        npz_path: Optional[str] = None,
        masks_dir: Optional[str] = None,
        cam_params_dir: Optional[str] = None,
    ):
        super().__init__()
        if split not in SPLIT_NPZ and npz_path is None:
            raise ValueError(f"split must be one of {list(SPLIT_NPZ)}; got {split!r}")
        self.root = Path(root)
        self.split = split
        npz_path = Path(npz_path) if npz_path else self.root / SPLIT_NPZ[split]

        d = np.load(str(npz_path), allow_pickle=True)
        self.imgnames = d["imgname"]
        contact = d["contact_label"]
        assert contact.shape[1] == NUM_VERTICES, contact.shape
        self.contact_labels = (contact > 0.5).astype(np.float32)
        self.cam_ks = d["cam_k"].astype(np.float32) if "cam_k" in d.files else None

        self.masks_dir = Path(masks_dir) / split if masks_dir else None

        # Precomputed MoGe2 intrinsics (one .npz per split). When the
        # file exists, ``cam_int[i]`` overrides the per-sample focal
        # length; otherwise the field stays ``None`` and downstream
        # code falls back to ``self.cam_ks``. The optional ``done`` mask
        # marks which indices were actually computed (so a half-finished
        # precompute doesn't silently hand out zero matrices).
        self.cam_params = None
        if cam_params_dir is not None:
            p = Path(cam_params_dir) / f"{split}.npz"
            if p.is_file():
                cp = np.load(str(p))
                K = cp["cam_int"].astype(np.float32)
                done = cp["done"].astype(bool) if "done" in cp.files else (
                    K.reshape(K.shape[0], -1).any(axis=1)
                )
                self.cam_params = {
                    "cam_int":    K,
                    "image_size": cp["image_size"].astype(np.int32),
                    "done":       done,
                }

    @classmethod
    def from_config(cls, config, split: str = "trainval") -> "DamonDataset":
        """Build from a config dict or path to a YAML file.

        Expected structure::

            data:
              root: <abs path>
              splits:
                trainval: <rel npz path>
                test:     <rel npz path>
            masks:
              dir: <abs path>     # optional
            cam_params:
              dir: <abs path>     # optional, holds {split}.npz files
        """
        if isinstance(config, (str, Path)):
            config = yaml.safe_load(Path(config).read_text())
        data = config["data"]
        npz_rel = data["splits"][split]
        return cls(
            root=data["root"],
            split=split,
            npz_path=str(Path(data["root"]) / npz_rel),
            masks_dir=(config.get("masks") or {}).get("dir"),
            cam_params_dir=(config.get("cam_params") or {}).get("dir"),
        )

    def __len__(self) -> int:
        return len(self.imgnames)

    def _load_mask(self, idx: int) -> Optional[np.ndarray]:
        if self.masks_dir is None:
            return None
        p = self.masks_dir / f"{idx:06d}.png"
        if not p.is_file():
            return None
        return np.array(Image.open(p), dtype=np.uint8)

    def _load_cam_int(self, idx: int) -> Optional[np.ndarray]:
        if self.cam_params is None or not self.cam_params["done"][idx]:
            return None
        return self.cam_params["cam_int"][idx].copy()

    @staticmethod
    def _mask_to_bbox(mask: np.ndarray, pad_frac: float = 0.05) -> Optional[np.ndarray]:
        """Tight xyxy bbox of the mask's foreground, expanded by ``pad_frac``.

        Returns ``None`` if the mask has no foreground pixels.
        """
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            return None
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        pad_x = (x1 - x0) * pad_frac
        pad_y = (y1 - y0) * pad_frac
        H, W = mask.shape[:2]
        return np.array(
            [max(0.0, x0 - pad_x), max(0.0, y0 - pad_y),
             min(W - 1.0, x1 + pad_x), min(H - 1.0, y1 + pad_y)],
            dtype=np.float32,
        )

    def __getitem__(self, idx: int) -> dict:
        rel = str(self.imgnames[idx])
        img = np.array(Image.open(self.root / rel).convert("RGB"), dtype=np.uint8)
        mask = self._load_mask(idx)
        bbox = self._mask_to_bbox(mask) if mask is not None else None
        cam_int = self._load_cam_int(idx)
        if cam_int is not None:
            focal = float(cam_int[0, 0])
        elif self.cam_ks is not None:
            focal = float(self.cam_ks[idx, 0, 0])
        else:
            focal = None
        return {
            "image":   img,
            "contact": torch.from_numpy(self.contact_labels[idx]),
            "key":     rel,
            "dataset": "damon",
            "mask":    mask,
            "bbox":    bbox,
            "focal":   focal,
            "cam_int": cam_int,
        }
