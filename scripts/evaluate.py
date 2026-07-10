"""Evaluate a trained contact checkpoint on its config's validation split.

Builds the model + val loader from ``--config`` (the same
``contact.data.collate.make_loaders`` the trainer uses), runs the frozen base +
loaded contact weights over the val split, and reports micro-averaged per-target
precision / recall / F1 / IoU via ``contact.metrics``. Works for both a vertex
config (e.g. DAMON) and a joint config (e.g. ClimbingVideos val split).

Usage::

    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
        --config configs/damon_baseline.yaml \
        --checkpoint output/<run>/best.pth --out output/eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from contact import checkpoint as ckpt_io
from contact.config import load_config
from contact.data.collate import batch_to_device, make_loaders
from contact.engine import forward_contact
from contact.metrics import add_counts, contact_counts, prf1, zero_counts
from contact.model import build_model


@torch.no_grad()
def evaluate(model, loader, targets: list[str], device: str) -> dict:
    counts = {t: zero_counts() for t in targets}
    for batch in loader:
        batch = batch_to_device(batch, device)
        contact = forward_contact(model, batch)
        for t in targets:
            tgt = batch["targets"][t]
            add_counts(counts[t], contact_counts(contact[f"{t}_logits"], tgt["gt"], tgt["mask"]))
    return {t: {**prf1(counts[t]), **counts[t]} for t in targets}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="append one result JSON per line here")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, _ = build_model(cfg, args.device)
    state = ckpt_io.load(args.checkpoint, model, config=cfg)
    model.eval()

    # Reproduce the exact split the checkpoint was validated on *when this eval
    # config uses the same datasets as training* (fail if members vanished from
    # disk). For a cross-eval on a different config, re-derive as usual.
    manifest = None
    if state.get("split_manifest") is not None:
        trained_datasets = (state.get("config", {}) or {}).get("data", {}).get("datasets")
        if trained_datasets == cfg["data"]["datasets"]:
            manifest = state["split_manifest"]

    _, val_loader, _ = make_loaders(cfg, tuple(model.cfg.MODEL.IMAGE_SIZE), manifest=manifest)
    targets = [t for t in ("vertex", "joint") if cfg["contact"]["targets"][t]["enabled"]]

    results = evaluate(model, val_loader, targets, args.device)
    for t, res in results.items():
        print(f"[{t}] P={res['precision']:.4f}  R={res['recall']:.4f}  "
              f"F1={res['f1']:.4f}  IoU={res['iou']:.4f}  "
              f"(tp={res['tp']} fp={res['fp']} fn={res['fn']})")
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps({"checkpoint": args.checkpoint,
                                "config": str(args.config), "results": results}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
