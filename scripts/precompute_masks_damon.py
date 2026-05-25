#!/usr/bin/env python3
"""Precompute SAM3 person masks for DAMON.

For every sample in the chosen split we run a single SAM3 "person"
prompt, keep the detection with the largest bounding box, and write
the binary mask (uint8 0/255 PNG, original image resolution) to
``{config.masks.dir}/{split}/{idx:06d}.png``.

Image loading happens in DataLoader worker processes so the GPU stays
busy; inference batches the workers' images together.

Usage::

    python scripts/precompute_masks_damon.py --split trainval
    python scripts/precompute_masks_damon.py --split test --resume

GPU is fixed to 0 by default (``CUDA_VISIBLE_DEVICES=0``).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
SAM3_REPO = REPO / "third_party" / "sam3"
SAM3_CKPT = Path(
    "/data3/rikhat.akizhanov/.cache/huggingface/hub/models--facebook--sam3/"
    "snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
)
BPE_PATH  = SAM3_REPO / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

sys.path.insert(0, str(REPO))
from dataset.damon import DamonDataset  # noqa: E402


# -------------------------------------------------------------------- SAM3 setup

def _import_sam3():
    sys.path.insert(0, str(SAM3_REPO))
    from sam3 import build_sam3_image_model
    from sam3.train.data.sam3_image_dataset import (
        Datapoint, FindQueryLoaded, Image as SAMImage, InferenceMetadata,
    )
    from sam3.train.data.collator import collate_fn_api as collate
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.transforms.basic_for_api import (
        ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI,
    )
    from sam3.eval.postprocessors import PostProcessImage
    return SimpleNamespace(
        build=build_sam3_image_model,
        Datapoint=Datapoint, FindQueryLoaded=FindQueryLoaded,
        SAMImage=SAMImage, InferenceMetadata=InferenceMetadata,
        collate=collate, to_device=copy_data_to_device,
        ComposeAPI=ComposeAPI, RandomResizeAPI=RandomResizeAPI,
        ToTensorAPI=ToTensorAPI, NormalizeAPI=NormalizeAPI,
        PostProcessImage=PostProcessImage,
    )


def build_sam3(threshold: float = 0.5):
    S = _import_sam3()
    model = S.build(bpe_path=str(BPE_PATH), load_from_HF=False,
                    checkpoint_path=str(SAM3_CKPT))
    model.eval()
    transform = S.ComposeAPI(transforms=[
        S.RandomResizeAPI(sizes=1008, max_size=1008, square=True,
                          consistent_transform=False),
        S.ToTensorAPI(),
        S.NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    postproc = S.PostProcessImage(
        max_dets_per_img=-1, iou_type="segm",
        use_original_sizes_box=True, use_original_sizes_mask=True,
        convert_mask_to_rle=False, detection_threshold=threshold, to_cpu=False,
    )
    return model, transform, postproc, S


def _make_datapoint(S, qid: int, image: np.ndarray):
    """Build a SAM3 datapoint with a single 'person' text query."""
    pil = Image.fromarray(image)
    h, w = image.shape[:2]
    dp = S.Datapoint(find_queries=[], images=[])
    dp.images = [S.SAMImage(data=pil, objects=[], size=[h, w])]
    dp.find_queries.append(S.FindQueryLoaded(
        query_text="person",
        image_id=0, object_ids_output=[], is_exhaustive=True,
        query_processing_order=0,
        inference_metadata=S.InferenceMetadata(
            coco_image_id=qid, original_image_id=qid,
            original_category_id=1, original_size=[h, w],
            object_id=0, frame_index=0,
        ),
    ))
    return dp


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
    idxs = [x[0] for x in batch]
    imgs = [x[1] for x in batch]
    return idxs, imgs


# -------------------------------------------------------------------- main

def run(config_path: Path, split: str, batch_size: int, num_workers: int,
        start: int, end: int, resume: bool, threshold: float) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    masks_dir = Path(cfg["masks"]["dir"]) / split
    masks_dir.mkdir(parents=True, exist_ok=True)

    ds = DamonDataset(
        root=cfg["data"]["root"], split=split,
        npz_path=str(Path(cfg["data"]["root"]) / cfg["data"]["splits"][split]),
        masks_dir=None,  # don't load existing masks during precompute
    )

    end = len(ds) if end < 0 else min(end, len(ds))
    indices = list(range(start, end))
    if resume:
        indices = [i for i in indices if not (masks_dir / f"{i:06d}.png").exists()]
    print(f"[{split}] processing {len(indices)} / {end - start} samples "
          f"(resume={resume})")
    if not indices:
        return

    loader = DataLoader(
        _Indexed(ds, indices),
        batch_size=batch_size, num_workers=num_workers,
        collate_fn=_collate, pin_memory=False, persistent_workers=num_workers > 0,
    )

    print("Loading SAM3 ...")
    model, transform, postproc, S = build_sam3(threshold=threshold)
    counter = 1
    pbar = tqdm(total=len(indices), desc=split)

    with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
        for idxs, imgs in loader:
            datapoints, kept_qids, kept_idxs = [], [], []
            for idx, img in zip(idxs, imgs):
                dp = _make_datapoint(S, counter, img)
                try:
                    dp = transform(dp)
                except Exception as exc:
                    print(f"[idx={idx}] transform failed: {exc}")
                    pbar.update(1)
                    counter += 1
                    continue
                datapoints.append(dp)
                kept_qids.append(counter)
                kept_idxs.append(idx)
                counter += 1
            if not datapoints:
                continue

            batch = S.collate(datapoints, dict_key="d")["d"]
            batch = S.to_device(batch, torch.device("cuda"), non_blocking=True)
            out = model(batch)
            results = postproc.process_results(out, batch.find_metadatas)

            for idx, qid in zip(kept_idxs, kept_qids):
                r = results.get(qid)
                if r is None or len(r["scores"]) == 0:
                    pbar.update(1)
                    continue
                b = r["boxes"].float()
                areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                best = areas.argmax().item()
                m = r["masks"][best].squeeze().cpu().float().numpy()
                m = (m > 0).astype(np.uint8) * 255
                Image.fromarray(m).save(masks_dir / f"{idx:06d}.png")
                pbar.update(1)

    pbar.close()
    print(f"[{split}] done → {masks_dir}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(REPO / "configs" / "damon.yaml"))
    p.add_argument("--split", default="trainval",
                   choices=("trainval", "test"))
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run(Path(args.config), args.split, args.batch_size, args.num_workers,
        args.start, args.end, args.resume, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
