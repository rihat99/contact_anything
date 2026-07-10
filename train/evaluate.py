"""Evaluate a trained contact head on a held-out test set.

Reports global (micro-averaged) precision / recall / F1 over all
samples x vertices. Two eval sets, both held out from every run:

  damon     — DAMON official *test* split (hot_dca_test.npz); no run
              trained on it (all DAMON training used the trainval split).
  climbing  — the climbing samples held out by BOTH climbing runs, i.e.
              exp1-val (climbing-only split) INTERSECT exp2-val (combined
              split). 131 samples that neither climbing model trained on,
              and that the DAMON-only model never saw either.

The model architecture is identical across the three runs, so we build
once from any config and load each run's ``best.pth`` (which carries only
the trained contact params) on top of the frozen SAM-3D-Body base.

Usage::

    CUDA_VISIBLE_DEVICES=0 python train/evaluate.py \
        --checkpoint train/output/<run>/best.pth --eval both \
        --out train/output/cross_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from train import checkpoint as ckpt_io
from train.data import batch_to_device, make_collate
from train.model import build_model
from train.train import _forward
from dataset.climbing import ClimbingImagesDataset
from dataset.damon import DamonDataset

# Architecture is shared; any run config builds the right model.
BUILD_CONFIG = REPO / "train" / "config_climbing.yaml"
DAMON_CONFIG = REPO / "configs" / "damon.yaml"


def climbing_test_indices() -> list[int]:
    """Climbing samples held out by BOTH climbing runs (clean for all three).

    Reproduces the two seed-42 splits exactly: exp1 permutes 5383 climbing
    items; exp2 permutes the 9767-item concat (DAMON 0..4383, climbing
    4384..9766). The intersection of their validation halves is the set no
    climbing model trained on.
    """
    v1 = set(np.random.default_rng(42).permutation(5383)[:807].tolist())
    idx2 = np.random.default_rng(42).permutation(9767)
    climb_val2 = {c - 4384 for c in idx2[:1465].tolist() if c >= 4384}
    return sorted(v1 & climb_val2)


def build_eval_dataset(which: str):
    """Return ``(dataset, n_used, n_total)`` for the named held-out set."""
    if which == "damon":
        ds = DamonDataset.from_config(str(DAMON_CONFIG), split="test")
        valid = [
            i for i in range(len(ds))
            if (ds.masks_dir / f"{i:06d}.png").is_file()
            and ds.cam_params is not None and bool(ds.cam_params["done"][i])
        ]
        return Subset(ds, valid), len(valid), len(ds)
    if which == "climbing":
        ds = ClimbingImagesDataset()
        idxs = climbing_test_indices()
        return Subset(ds, idxs), len(idxs), len(ds)
    raise ValueError(f"unknown eval set {which!r}")


@torch.no_grad()
def evaluate(model, loader, device: str) -> dict:
    tp = fp = fn = tn = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        preds = torch.sigmoid(_forward(model, batch)) > 0.5
        gt = batch["contact"].bool()
        tp += int((preds & gt).sum())
        fp += int((preds & ~gt).sum())
        fn += int((~preds & gt).sum())
        tn += int((~preds & ~gt).sum())
    eps = 1e-8
    return {
        "precision": tp / (tp + fp + eps),
        "recall":    tp / (tp + fn + eps),
        "f1":        2 * tp / (2 * tp + fp + fn + eps),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval", default="both", choices=["damon", "climbing", "both"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="append one result JSON per line here")
    args = ap.parse_args()

    cfg = yaml.safe_load(BUILD_CONFIG.read_text())
    model, _ = build_model(cfg, args.device)
    ckpt_io.load(args.checkpoint, model)
    model.eval()
    image_size = tuple(model.cfg.MODEL.IMAGE_SIZE)
    collate = make_collate(image_size)

    which = ["damon", "climbing"] if args.eval == "both" else [args.eval]
    for w in which:
        ds, n_used, n_total = build_eval_dataset(w)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate, pin_memory=False,
        )
        res = evaluate(model, loader, args.device)
        res.update(checkpoint=args.checkpoint, eval=w, n_used=n_used, n_total=n_total)
        print(f"[{w}] n={n_used}/{n_total}  "
              f"P={res['precision']:.4f}  R={res['recall']:.4f}  F1={res['f1']:.4f}")
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(res) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
