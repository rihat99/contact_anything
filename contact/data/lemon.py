"""LEMON / 3DIR dataset — image + per-vertex SMPL-H (6890) contact labels.

SMPL-H shares the SMPL body topology (6890 verts), so labels are
directly usable on a SMPL mesh.

Image paths come from ``txt_scripts/{split}.txt`` (entries look like
``Data/Images/<Class>/<action>/<Class>_<action>_<n>.jpg``); the
``Data/`` prefix is stripped to resolve them under ``root``. The
matching contact pkl lives at the same relative path with ``Images``
replaced by ``smplh_contact_pkl`` and ``.jpg`` by ``.pkl``.
"""

from pathlib import Path

import joblib as jl
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

NUM_VERTICES = 6890
DEFAULT_ROOT = "/data3/rikhat.akizhanov/datasets/3DIR_release"


class LemonDataset(Dataset):
    def __init__(
        self,
        root: str = DEFAULT_ROOT,
        split: str = "train",
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split

        txt = self.root / "txt_scripts" / f"{split}.txt"
        rels = [line.removeprefix("Data/")
                for line in txt.read_text().splitlines() if line.strip()]

        # Keep only entries whose contact pkl actually exists on disk.
        self.entries: list[tuple[str, str]] = []
        for img_rel in rels:
            contact_rel = (img_rel
                           .replace("Images/", "smplh_contact_pkl/", 1)
                           .replace(".jpg", ".pkl"))
            if (self.root / contact_rel).is_file():
                self.entries.append((img_rel, contact_rel))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        img_rel, contact_rel = self.entries[idx]
        img = np.array(Image.open(self.root / img_rel).convert("RGB"), dtype=np.uint8)
        contact = jl.load(self.root / contact_rel).astype(np.float32)
        assert contact.shape == (NUM_VERTICES,), contact.shape
        return {
            "image":   img,
            "contact": torch.from_numpy(contact),
            "key":     img_rel,
            "dataset": "lemon",
            "mask":    None,
            "bbox":    None,
            "focal":   None,
        }
