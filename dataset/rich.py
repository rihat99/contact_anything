"""RICH dataset — image + per-vertex SMPL (6890) contact labels.

Uses the BSTRO TSV-DB format shipped with the BSTRO paper
(``rich_for_bstro_tsv_db.zip``). The on-disk layout per split is:

    rich_for_bstro_tsv_db/
        {split}.label.tsv       — ``key\\t{"contact": [6890 floats], ...}`` per row
        {split}.label.lineidx   — byte offsets into the corresponding tsv
        {split}.img.tsv         — ``key\\t<base64-encoded JPEG>`` per row
        {split}.img.lineidx
        {split}.hw.tsv          — image height/width metadata
        ...

The image TSVs are huge (28-90 GB per split) and must be extracted from
the zip once. If the .img.tsv is missing, the loader still works for
labels/keys, but ``image`` is ``None`` — useful for stats but not for
the viewer.

Extract the zip to enable images:

    cd /data3/rikhat.akizhanov/datasets/RICH
    unzip rich_for_bstro_tsv_db.zip
"""

import base64
import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

NUM_VERTICES = 6890
DEFAULT_ROOT = "/data3/rikhat.akizhanov/datasets/RICH/rich_for_bstro_tsv_db"


class _TSV:
    """Random-access reader for a single BSTRO-style TSV file.

    Each instance keeps a file handle plus the byte-offset table from
    the matching ``.lineidx``. Reopen-on-fork friendly: the offsets are
    plain ints, the file handle is rebuilt per worker process.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = str(path)
        lineidx = os.path.splitext(self.path)[0] + ".lineidx"
        with open(lineidx) as f:
            self.offsets = [int(x) for x in f.read().splitlines()]
        self._fp = None
        self._pid = None

    def __len__(self) -> int:
        return len(self.offsets)

    def _ensure_open(self) -> None:
        pid = os.getpid()
        if self._fp is None or self._pid != pid:
            self._fp = open(self.path, "rb")
            self._pid = pid

    def __getitem__(self, idx: int) -> list[bytes]:
        self._ensure_open()
        self._fp.seek(self.offsets[idx])
        return self._fp.readline().rstrip(b"\n").split(b"\t")


class RichDataset(Dataset):
    """RICH dataset (BSTRO TSV format)."""

    SPLITS = ("train", "val", "test")

    def __init__(
        self,
        root: str = DEFAULT_ROOT,
        split: str = "test",
        require_images: bool = False,
    ):
        super().__init__()
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}; got {split!r}")
        self.root = Path(root)
        self.split = split

        label_tsv = self.root / f"{split}.label.tsv"
        if not label_tsv.is_file():
            raise FileNotFoundError(
                f"RICH label tsv not found: {label_tsv}.\n"
                f"Extract rich_for_bstro_tsv_db.zip first."
            )
        self.label_tsv = _TSV(label_tsv)

        img_tsv = self.root / f"{split}.img.tsv"
        if img_tsv.is_file():
            self.img_tsv: Optional[_TSV] = _TSV(img_tsv)
        elif require_images:
            raise FileNotFoundError(
                f"RICH image tsv not found: {img_tsv}.\n"
                f"Extract rich_for_bstro_tsv_db.zip to enable images."
            )
        else:
            self.img_tsv = None

    def __len__(self) -> int:
        return len(self.label_tsv)

    def __getitem__(self, idx: int) -> dict:
        key_b, label_b = self.label_tsv[idx]
        key = key_b.decode()
        ann = json.loads(label_b)[0]
        contact = np.asarray(ann["contact"], dtype=np.float32)
        assert contact.shape == (NUM_VERTICES,), contact.shape

        image = None
        if self.img_tsv is not None:
            _, img_b = self.img_tsv[idx]
            buf = np.frombuffer(base64.b64decode(img_b), dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # center/scale describe a square crop in original image coords; convert to xyxy bbox.
        cx, cy = ann["center"]
        s = float(ann["scale"]) * 200.0 / 2.0
        bbox = np.array([cx - s, cy - s, cx + s, cy + s], dtype=np.float32)

        return {
            "image":   image,
            "contact": torch.from_numpy(contact),
            "key":     key,
            "dataset": "rich",
            "mask":    None,
            "bbox":    bbox,   # crop from BSTRO annotation
            "focal":   None,
        }
