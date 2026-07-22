"""Evaluate a trained contact checkpoint on validation or manual video test data.

Builds the model + val loader from ``--config`` (the same
``contact.data.collate.make_loaders`` the trainer uses), runs the frozen base +
loaded contact weights over the val split, and reports micro-averaged per-target
precision / recall / F1 / F2 / IoU via ``contact.metrics``. Works for both a
vertex config (e.g. DAMON) and a joint config. ``--split test`` is supported for
ClimbingVideos and reads the physical manually annotated test directory.

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
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from contact import checkpoint as ckpt_io
from contact.config import load_config
from contact.data.climbing_videos import ClimbingVideosDataset
from contact.data.collate import batch_to_device, make_collate, make_loaders
from contact.engine import forward_contact, select_temporal_supervision
from contact.metrics import (
    add_counts,
    contact_counts,
    contact_counts_per_dim,
    prf1,
    zero_counts,
)
from contact.model import build_model
from contact.targets import TargetSpec


@torch.no_grad()
def evaluate(
    model,
    loader,
    targets: list[str],
    device: str,
    *,
    threshold: float = 0.5,
    curve_thresholds: tuple[float, ...] = (),
    output_names: dict[str, tuple[str, ...]] | None = None,
    target_frame: str = "all",
) -> dict:
    thresholds = tuple(sorted(set((float(threshold), *map(float, curve_thresholds)))))
    counts = {th: {t: zero_counts() for t in targets} for th in thresholds}
    per_output = {}
    for target, names in (output_names or {}).items():
        per_output[target] = {name: zero_counts() for name in names}
    for batch in loader:
        batch = batch_to_device(batch, device)
        contact = forward_contact(model, batch)
        logits_by_target, selected_targets = select_temporal_supervision(
            {t: contact[f"{t}_logits"] for t in targets},
            batch["targets"],
            int(batch.get("seq_len", 1)),
            target_frame,
        )
        for t in targets:
            tgt = selected_targets[t]
            logits = logits_by_target[t]
            for th in thresholds:
                add_counts(counts[th][t], contact_counts(
                    logits, tgt["gt"], tgt["mask"], threshold=th))
            if t in per_output:
                dim_counts = contact_counts_per_dim(
                    logits, tgt["gt"], tgt["mask"], threshold=threshold)
                for name, current in zip(per_output[t], dim_counts):
                    add_counts(per_output[t][name], current)

    results = {
        t: {**prf1(counts[float(threshold)][t]), **counts[float(threshold)][t]}
        for t in targets
    }
    for target, named_counts in per_output.items():
        results[target]["per_output"] = {
            name: {**prf1(value), **value} for name, value in named_counts.items()
        }
    if curve_thresholds:
        for target in targets:
            results[target]["threshold_curve"] = [
                {"threshold": th, **prf1(counts[th][target]), **counts[th][target]}
                for th in thresholds
            ]
    return results


def _manual_test_loader(cfg: dict, image_size: tuple[int, int], spec: TargetSpec):
    video_entries = [d for d in cfg["data"]["datasets"] if d["name"] == "climbing_videos"]
    if len(video_entries) != 1 or len(cfg["data"]["datasets"]) != 1:
        raise ValueError("--split test requires a ClimbingVideos-only data config")
    dataset_cfg = yaml.safe_load(Path(video_entries[0]["config"]).read_text()) or {}
    sequence = cfg["data"]["sequence"]
    ds = ClimbingVideosDataset(
        root=dataset_cfg["data"]["root"],
        mode="val",
        split_dir="test",
        frames_per_clip=int(sequence["frames_per_clip"]),
        frame_stride=int(sequence["frame_stride"]),
        jitter=False,
        seed=int(cfg["data"]["seed"]),
        use_confidence_weights=bool(
            cfg["contact"]["targets"]["joint"]["use_confidence_weights"]),
    )
    clips_per_batch = max(
        1, int(cfg["data"]["frames_per_batch"]) // int(sequence["frames_per_clip"]))
    return DataLoader(
        ds,
        batch_size=clips_per_batch,
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        collate_fn=make_collate(image_size, spec),
        pin_memory=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument(
        "--curve-thresholds",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="comma-separated thresholds for the saved precision/recall curve; empty disables",
    )
    ap.add_argument("--out", default=None, help="append one result JSON per line here")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, _ = build_model(cfg, args.device)
    state = ckpt_io.load(args.checkpoint, model, config=cfg)
    model.eval()

    spec = TargetSpec.from_config(cfg)
    if args.split == "test":
        val_loader = _manual_test_loader(cfg, tuple(model.cfg.MODEL.IMAGE_SIZE), spec)
    else:
        # Reproduce the checkpoint's exact grouped validation split when this
        # config uses the same datasets as training.
        manifest = None
        if state.get("split_manifest") is not None:
            trained_datasets = (state.get("config", {}) or {}).get("data", {}).get("datasets")
            if trained_datasets == cfg["data"]["datasets"]:
                manifest = state["split_manifest"]
        _, val_loader, _ = make_loaders(
            cfg, tuple(model.cfg.MODEL.IMAGE_SIZE), manifest=manifest)
    targets = [t for t in ("vertex", "joint") if cfg["contact"]["targets"][t]["enabled"]]
    curve_thresholds = tuple(
        float(value) for value in args.curve_thresholds.split(",") if value.strip())
    output_names = {"joint": spec.joint_names} if "joint" in targets else {}

    results = evaluate(
        model,
        val_loader,
        targets,
        args.device,
        threshold=args.threshold,
        curve_thresholds=curve_thresholds,
        output_names=output_names,
        target_frame=str(cfg["data"]["sequence"]["target_frame"]),
    )
    for t, res in results.items():
        print(f"[{t}] P={res['precision']:.4f}  R={res['recall']:.4f}  "
              f"F1={res['f1']:.4f}  F2={res['f2']:.4f}  IoU={res['iou']:.4f}  "
              f"(tp={res['tp']} fp={res['fp']} fn={res['fn']})")
        for name, values in res.get("per_output", {}).items():
            print(f"  {name:>10s}: P={values['precision']:.4f} R={values['recall']:.4f} "
                  f"F1={values['f1']:.4f} F2={values['f2']:.4f}")
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps({"checkpoint": args.checkpoint,
                                "config": str(args.config), "split": args.split,
                                "threshold": args.threshold, "results": results}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
