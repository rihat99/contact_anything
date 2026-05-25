#!/usr/bin/env python3
"""Precompute MoGe2 camera intrinsics for DAMON.

Mirrors what ``SAM3DBodyEstimator`` does at inference time: feed each
full-resolution RGB image through MoGe2, denormalise the resulting
``intrinsics`` to absolute pixels, then override ``fx`` with ``fy`` so
the matrix is aspect-correct in the SAM-3D-Body sense::

    fov_x_deg via build_fov_estimator -> intrinsics[0, 0] = intrinsics[1, 1]

The output for every split is a single compressed NPZ stored at
``{config.cam_params.dir}/{split}.npz`` with::

    cam_int    (N, 3, 3) float32   absolute-pixel intrinsics
    image_size (N, 2)    int32     (H, W) at the time of inference

Image loading happens in DataLoader worker processes so the GPU stays
busy. Inference runs one image at a time because DAMON has too many
distinct (H, W) shapes for fixed-shape batching to pay off.

Usage::

    python scripts/precompute_cam_params_damon.py --split trainval
    python scripts/precompute_cam_params_damon.py --split test --resume

GPU is fixed to 0 by default (``CUDA_VISIBLE_DEVICES=0``).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from dataset.damon import DamonDataset  # noqa: E402

_DEFAULT_MODEL = "Ruicheng/moge-2-vitl-normal"


# -------------------------------------------------------------------- MoGe2

def _load_moge2(model_id: str, device: str, dtype: torch.dtype):
    from moge.model.v2 import MoGeModel
    return MoGeModel.from_pretrained(model_id).to(device).to(dtype).eval()


def _denormalize_K(norm_K: np.ndarray, H: int, W: int) -> np.ndarray:
    """Same convention as ``tools/build_fov_estimator.py`` in SAM-3D-Body.

    MoGe2 returns ``[fx/W, 0, cx/W; 0, fy/H, cy/H; 0, 0, 1]``. Scale to
    absolute pixels and then *override* ``fx`` with ``fy`` so the focal
    length is reported consistently across landscape and portrait crops
    — this matches what the SAM-3D-Body inference pipeline ingests.
    """
    fx = float(norm_K[0, 0]) * W
    fy = float(norm_K[1, 1]) * H
    cx = float(norm_K[0, 2]) * W
    cy = float(norm_K[1, 2]) * H
    K = np.array(
        [[fy, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return K


def _run_moge2(model, image_rgb: np.ndarray, device: str) -> np.ndarray:
    """Single-image MoGe2 inference → absolute-pixel 3×3 K."""
    H, W = image_rgb.shape[:2]
    t = (
        torch.from_numpy(image_rgb)
        .to(device, non_blocking=True)
        .float()
        .div_(255.0)
        .permute(2, 0, 1)
        .contiguous()
    )
    out = model.infer(t)
    norm_K = out["intrinsics"].detach().float().cpu().numpy()
    return _denormalize_K(norm_K, H, W)


# -------------------------------------------------------------------- loader

class _Indexed(torch.utils.data.Dataset):
    """Yields ``(global_idx, image_uint8)`` for the precompute pipeline."""

    def __init__(self, base: DamonDataset, indices: list[int]):
        self.base = base
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        item = self.base[idx]
        return idx, item["image"]


def _collate(batch):
    return [x[0] for x in batch], [x[1] for x in batch]


# -------------------------------------------------------------------- main

def run(config_path: Path, split: str, num_workers: int,
        start: int, end: int, resume: bool, model_id: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_dir = Path(cfg["cam_params"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"{split}.npz"

    ds = DamonDataset(
        root=cfg["data"]["root"], split=split,
        npz_path=str(Path(cfg["data"]["root"]) / cfg["data"]["splits"][split]),
        masks_dir=None,
        cam_params_dir=None,
    )

    N = len(ds)
    end = N if end < 0 else min(end, N)
    indices = list(range(start, end))

    # If resuming, only run indices that are missing from the existing npz.
    existing_K = np.zeros((N, 3, 3), dtype=np.float32)
    existing_sz = np.zeros((N, 2), dtype=np.int32)
    done_mask = np.zeros(N, dtype=bool)
    if resume and out_npz.is_file():
        cp = np.load(str(out_npz))
        existing_K = cp["cam_int"].astype(np.float32)
        existing_sz = cp["image_size"].astype(np.int32)
        done_mask = cp["done"].astype(bool) if "done" in cp.files else (
            existing_K.reshape(N, -1).any(axis=1)
        )
        indices = [i for i in indices if not done_mask[i]]

    print(f"[{split}] processing {len(indices)} / {end - start} samples "
          f"(resume={resume}) -> {out_npz}")
    if not indices:
        return

    loader = DataLoader(
        _Indexed(ds, indices),
        batch_size=1, num_workers=num_workers,
        collate_fn=_collate, pin_memory=False,
        persistent_workers=num_workers > 0,
    )

    print(f"Loading MoGe2 ({model_id}) ...")
    model = _load_moge2(model_id, "cuda", torch.bfloat16)

    cam_int = existing_K.copy()
    image_size = existing_sz.copy()
    done = done_mask.copy()
    save_every = 500
    pbar = tqdm(total=len(indices), desc=split)

    with torch.inference_mode():
        for n_processed, (idxs, imgs) in enumerate(loader, start=1):
            for idx, img in zip(idxs, imgs):
                K = _run_moge2(model, img, "cuda")
                cam_int[idx] = K
                image_size[idx] = (img.shape[0], img.shape[1])
                done[idx] = True
                pbar.update(1)
            # Periodic checkpoint so a long run survives interruptions.
            if n_processed % save_every == 0:
                np.savez_compressed(
                    out_npz, cam_int=cam_int, image_size=image_size, done=done,
                )

    pbar.close()
    np.savez_compressed(
        out_npz, cam_int=cam_int, image_size=image_size, done=done,
    )
    n_done = int(done.sum())
    print(f"[{split}] done → {out_npz}  ({n_done}/{N} samples written)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(REPO / "configs" / "damon.yaml"))
    p.add_argument("--split", default="trainval", choices=("trainval", "test"))
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--model", default=_DEFAULT_MODEL)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run(Path(args.config), args.split, args.num_workers,
        args.start, args.end, args.resume, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
