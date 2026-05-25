"""Training data — DamonDataset → SAM-3D-Body batch.

Walks the new ``DamonDataset`` (returns image / mask / bbox-from-mask /
cam_int / contact), runs the SAM-3D-Body ``TopdownAffine`` transform per
sample so the cropped image *and* the cropped mask end up on the model's
input grid, then assembles the batch dict that ``SAM3DBody.forward_step``
consumes.

This is the only place the train loop touches the SAM-3D-Body transform
API — keep it thin so swapping in LEMON/RICH is a one-liner upstream.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, default_collate
from torchvision.transforms import ToTensor

from sam_3d_body.data.transforms import (
    Compose, GetBBoxCenterScale, TopdownAffine, VisionTransformWrapper,
)
from sam_3d_body.data.utils.prepare_batch import NoCollate

from dataset.damon import DamonDataset


# -------------------------------------------------------------------- transform

def _build_transform(image_size: Tuple[int, int]):
    """SAM-3D-Body's standard top-down crop pipeline at the model resolution."""
    return Compose([
        GetBBoxCenterScale(),
        TopdownAffine(input_size=image_size, use_udp=False),
        VisionTransformWrapper(ToTensor()),
    ])


# -------------------------------------------------------------------- collate

def _process_sample(sample: dict, transform):
    """Run the SAM-3D-Body transform on a single dataset item.

    ``DamonDataset`` returns a uint8 image, a uint8 mask, an xyxy bbox,
    and an absolute-pixel cam_int. We hand them to the transform and
    let it warp both image and mask into the model's crop frame.
    """
    img  = sample["image"]
    mask = sample["mask"]
    bbox = sample["bbox"]
    if bbox is None:
        raise RuntimeError(f"sample {sample.get('key', '?')} has no bbox")
    if mask is None:
        H, W = img.shape[:2]
        mask = np.zeros((H, W, 1), dtype=np.uint8)
    elif mask.ndim == 2:
        mask = mask[..., None]

    data_info = dict(
        img=img,
        bbox=bbox.astype(np.float32),
        bbox_format="xyxy",
        mask=mask,
        mask_score=np.array(1.0, dtype=np.float32),
    )
    out = transform(data_info)
    # Mask comes out of TopdownAffine as float32 [H, W] or [H, W, 1].
    # Normalise to [0, 1] for the conv-stack in mask_downscaling.
    m = out["mask"]
    if m.ndim == 3:
        m = m[..., 0]
    out["mask"] = (m.astype(np.float32) / 255.0)
    return out


def make_collate(image_size: Tuple[int, int]):
    transform = _build_transform(image_size)

    def _collate(batch):
        per_sample = [_process_sample(s, transform) for s in batch]
        cam_ints   = torch.stack([
            torch.as_tensor(s["cam_int"], dtype=torch.float32) for s in batch
        ], dim=0)  # [B, 3, 3]
        contacts   = torch.stack([s["contact"].float() for s in batch], dim=0)
        first_img  = batch[0]["image"]  # kept for hand-decoder code paths

        keys = ["img", "img_size", "ori_img_size", "bbox_center", "bbox_scale",
                "bbox", "affine_trans", "mask", "mask_score"]
        out = {}
        for k in keys:
            if k not in per_sample[0]:
                continue
            tensors = [
                s[k] if isinstance(s[k], torch.Tensor) else torch.as_tensor(s[k])
                for s in per_sample
            ]
            stacked = torch.stack(tensors, dim=0).float()  # [B, ...]
            out[k]  = stacked.unsqueeze(1)                 # [B, 1, ...]
        # Mask wants an extra channel dim: [B, 1, 1, H, W]
        if "mask" in out and out["mask"].dim() == 4:
            out["mask"] = out["mask"].unsqueeze(2)
        out["person_valid"] = torch.ones((len(batch), 1))
        out["cam_int"]      = cam_ints
        out["img_ori"]      = [NoCollate(first_img)]
        out["contact"]      = contacts
        return out

    return _collate


def batch_to_device(batch: dict, device: str) -> dict:
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)
    return batch


# -------------------------------------------------------------------- splits

def make_loaders(cfg: dict, image_size: Tuple[int, int]) -> Tuple[DataLoader, DataLoader]:
    """Build train + val DataLoaders from the trainval split of DAMON."""
    dcfg = cfg["data"]
    ds   = DamonDataset.from_config(dcfg["dataset_config"], split="trainval")
    n    = len(ds)
    rng  = np.random.default_rng(int(dcfg.get("seed", 42)))
    idx  = rng.permutation(n)
    n_val = int(round(n * float(dcfg.get("val_ratio", 0.15))))
    val_idx, train_idx = idx[:n_val].tolist(), idx[n_val:].tolist()

    collate = make_collate(image_size)
    bs   = int(dcfg["batch_size"])
    nw   = int(dcfg["num_workers"])
    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=bs, shuffle=True, num_workers=nw, drop_last=True,
        collate_fn=collate, pin_memory=False,
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=bs, shuffle=False, num_workers=nw, drop_last=False,
        collate_fn=collate, pin_memory=False,
        persistent_workers=nw > 0,
    )
    print(f"Damon split: train={len(train_idx)} val={len(val_idx)} "
          f"(val_ratio={dcfg.get('val_ratio', 0.15)}, seed={dcfg.get('seed', 42)})")
    return train_loader, val_loader
