"""Qualitative per-frame inference for a ClimbingVideos joint checkpoint.

Each figure contains the source frame, the frozen model's predicted MHR pose,
the confidence-coloured ground-truth canonical skeleton, and the predicted
canonical skeleton. ClimbingVideos does not provide projected joint positions,
so the contact skeleton is deliberately schematic rather than overlaid.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from contact import checkpoint as ckpt_io
from contact.config import load_config
from contact.data.climbing_videos import ClimbingVideosDataset, list_scenes
from contact.data.collate import batch_to_device, make_collate
from contact.data.splits import video_id_from_scene
from contact.engine import forward_model
from contact.model import build_model
from contact.targets import (
    EXTREMITY_4_GROUPS,
    SMPLX_BODY_22,
    TargetSpec,
    reduce_body22_to_extremities,
)
from scripts.demo import _mhr_overlay_arrays, overlay_mesh_on_image_2d
from viewer.skeleton import JOINT_COORDS, JOINT_EDGES

COLOR_CONTACT = np.array([0.87, 0.12, 0.16])
COLOR_FREE = np.array([0.08, 0.62, 0.36])
COLOR_UNCERTAIN = np.array([0.64, 0.66, 0.69])


def _mix_color(contact: bool, confidence: float) -> np.ndarray:
    target = COLOR_CONTACT if contact else COLOR_FREE
    amount = float(np.clip(confidence, 0.0, 1.0))
    return COLOR_UNCERTAIN + amount * (target - COLOR_UNCERTAIN)


def _draw_skeleton(
    ax,
    states: np.ndarray,
    confidence: np.ndarray,
    supervised: np.ndarray,
    title: str,
    subtitle: str,
) -> None:
    coords = np.asarray(JOINT_COORDS, dtype=np.float32)
    for a, b in JOINT_EDGES:
        ax.plot(coords[[a, b], 0], coords[[a, b], 1], color="#d8dbe1",
                linewidth=2.4, zorder=1)
    for index, (x, y) in enumerate(coords):
        if supervised[index]:
            color = _mix_color(bool(states[index]), float(confidence[index]))
            ax.scatter(x, y, s=145, color=color, edgecolor="white", linewidth=1.4, zorder=3)
        else:
            ax.scatter(x, y, s=145, facecolor="white", edgecolor=COLOR_UNCERTAIN,
                       linewidth=1.5, zorder=3)
        if states[index] and supervised[index]:
            dx = 2.5 if x >= 50 else -2.5
            ha = "left" if x >= 50 else "right"
            ax.text(x + dx, y, SMPLX_BODY_22[index].replace("_", " "),
                    fontsize=7.5, color="#7f1117", ha=ha, va="center", zorder=4)
    ax.set_xlim(-5, 105)
    ax.set_ylim(108, 0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.text(0.5, -0.025, subtitle, transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color="#555b64")


def _counts(pred: np.ndarray, gt: np.ndarray, active: np.ndarray) -> dict:
    pred, gt = pred & active, gt & active
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt & active).sum())
    fn = int((~pred & gt & active).sum())
    tn = int((~pred & ~gt & active).sum())
    eps = 1e-8
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "f1": 2 * tp / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
    }


def _expand_for_skeleton(
    values: np.ndarray,
    confidence: np.ndarray,
    supervised: np.ndarray,
    spec: TargetSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand a semantic joint-set vector onto the canonical body-22 skeleton."""
    if spec.joint_set == "smplx_body_22":
        return values, confidence, supervised
    states22 = np.zeros(22, dtype=values.dtype)
    confidence22 = np.zeros(22, dtype=confidence.dtype)
    supervised22 = np.zeros(22, dtype=bool)
    for output_index, group in enumerate(EXTREMITY_4_GROUPS):
        states22[list(group)] = values[output_index]
        confidence22[list(group)] = confidence[output_index]
        supervised22[list(group)] = supervised[output_index]
    return states22, confidence22, supervised22


def _reduced_label_arrays(sample: dict, spec: TargetSpec):
    if spec.joint_set == "smplx_body_22":
        return (
            sample["joint_contact"].numpy(),
            sample["joint_supervised"].numpy(),
            sample["joint_confidence"].numpy(),
        )
    return tuple(
        value.numpy() for value in reduce_body22_to_extremities(
            sample["joint_contact"],
            sample["joint_supervised"],
            sample["joint_confidence"],
        )
    )


def _make_figure(
    sample: dict,
    out: dict,
    probs: np.ndarray,
    target_gt: np.ndarray,
    target_mask: np.ndarray,
    spec: TargetSpec,
    threshold: float,
):
    reduced_gt, reduced_supervised, reduced_confidence = _reduced_label_arrays(sample, spec)
    gt = target_gt > 0.5
    supervised = reduced_supervised > 0.5
    confidence = reduced_confidence
    active = target_mask > 0
    pred = probs > threshold
    score = _counts(pred, gt, active)
    pred_confidence = np.clip(2.0 * np.abs(probs - 0.5), 0.0, 1.0)
    gt22, gt_conf22, gt_supervised22 = _expand_for_skeleton(
        reduced_gt > 0.5, confidence, supervised, spec)
    pred22, pred_conf22, pred_supervised22 = _expand_for_skeleton(
        pred, pred_confidence, np.ones(spec.joint_dims, dtype=bool), spec)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), dpi=180,
                             gridspec_kw={"height_ratios": [1.0, 1.05]})
    image = sample["image"]
    axes[0, 0].imshow(image)
    x1, y1, x2, y2 = np.asarray(sample["bbox"]).astype(int)
    axes[0, 0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   linewidth=1.8, edgecolor="#f1c40f", facecolor="none"))
    axes[0, 0].set_title("Source frame", fontsize=13, fontweight="bold")
    axes[0, 0].set_axis_off()

    pred_v2, pred_vcam, mhr_faces = _mhr_overlay_arrays(out["mhr"])
    overlay_mesh_on_image_2d(axes[0, 1], image, pred_v2, pred_vcam, mhr_faces,
                             None, "Predicted pose (frozen MHR)")

    gt_mean = float(confidence[supervised].mean()) if supervised.any() else 0.0
    _draw_skeleton(
        axes[1, 0], gt22, gt_conf22, gt_supervised22,
        f"Ground truth · {spec.joint_set}",
        f"{int(gt[active].sum())} contacts · mean label confidence {gt_mean:.0%}",
    )
    _draw_skeleton(
        axes[1, 1], pred22, pred_conf22, pred_supervised22,
        f"Prediction · threshold {threshold:.2f}",
        f"{int(pred.sum())} contacts · F1 {score['f1']:.3f} · IoU {score['iou']:.3f}",
    )
    title = (f"{sample['key']}   ·   P {score['precision']:.3f}   "
             f"R {score['recall']:.3f}   F1 {score['f1']:.3f}")
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0.015, 1, 0.975])
    return fig, score, gt, pred, confidence


def _video_dataset(cfg: dict, state: dict, split: str) -> ClimbingVideosDataset:
    video_spec = next(d for d in cfg["data"]["datasets"] if d["name"] == "climbing_videos")
    root = yaml.safe_load(Path(video_spec["config"]).read_text())["data"]["root"]
    split_dir = "test" if split == "test" else "train"
    scenes = list_scenes(root, split_dir)
    if split == "val":
        key = f"video:{video_spec['config']}"
        manifest = state.get("split_manifest") or {}
        if key not in manifest:
            raise RuntimeError(f"checkpoint split manifest has no {key!r}")
        val_videos = set(manifest[key]["val"])
        scenes = [scene for scene in scenes if video_id_from_scene(scene) in val_videos]
    return ClimbingVideosDataset(
        root,
        scenes=scenes,
        mode="val",
        split_dir=split_dir,
        frames_per_clip=1,
        frame_stride=1,
        jitter=False,
        seed=int(cfg["data"]["seed"]),
        use_confidence_weights=bool(
            cfg["contact"]["targets"]["joint"]["use_confidence_weights"]),
    )


def _stratified_picks(
    ds: ClimbingVideosDataset, spec: TargetSpec, count: int, seed: int,
) -> list[int]:
    buckets = {0: [], 1: [], 2: []}
    for index, (scene, person, base, _) in enumerate(ds._items):
        data = ds._scenes[scene]
        contacts = torch.as_tensor(data["joint_contact"][person, base], dtype=torch.float32)
        supervised = torch.full((22,), float(data["valid_mask"][person, base]))
        if data["annotated"] is not None:
            supervised *= torch.as_tensor(data["annotated"][person, base])
        confidence = (
            torch.ones(22) if data["contact_conf"] is None else
            torch.as_tensor(np.nan_to_num(
                data["contact_conf"][person, base], nan=0.0, posinf=1.0, neginf=0.0
            ).clip(0.0, 1.0))
        )
        if spec.joint_set == "extremities_4":
            contacts, supervised, _ = reduce_body22_to_extremities(
                contacts, supervised, confidence)
        n_contact = int(((contacts > 0.5) & (supervised > 0)).sum())
        buckets[min(n_contact, 2)].append(index)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    picks = []
    base_quota, remainder = divmod(count, 3)
    for bucket_id in range(3):
        quota = base_quota + int(bucket_id < remainder)
        picks.extend(buckets[bucket_id][:quota])
    if len(picks) < count:
        used = set(picks)
        remainder_pool = [i for values in buckets.values() for i in values if i not in used]
        rng.shuffle(remainder_pool)
        picks.extend(remainder_pool[:count - len(picks)])
    rng.shuffle(picks)
    return picks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "climbing_videos_joint.yaml")
    ap.add_argument("--num-samples", type=int, default=12)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    threshold_tag = f"{args.threshold:.2f}".replace(".", "")
    out_dir = (Path(args.output_dir) if args.output_dir else
               checkpoint.parent / f"inference_{args.split}_last_t{threshold_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    print("Building joint-contact model …")
    model, _ = build_model(cfg, args.device)
    state = ckpt_io.load(checkpoint, model, config=cfg, map_location=args.device)
    model.eval()
    ds = _video_dataset(cfg, state, args.split)
    target_spec = TargetSpec.from_config(cfg)
    picks = _stratified_picks(
        ds, target_spec, min(args.num_samples, len(ds)), args.seed)
    collate = make_collate(tuple(model.cfg.MODEL.IMAGE_SIZE), target_spec)
    print(f"{args.split.capitalize()} frames: {len(ds)}; rendering {len(picks)} "
          f"from last checkpoint at threshold {args.threshold:.2f}")

    records = []
    for run_index, ds_index in enumerate(picks):
        sample = ds[ds_index][0]
        batch = collate([[sample]])
        target_gt = batch["targets"]["joint"]["gt"][0].numpy()
        target_mask = batch["targets"]["joint"]["mask"][0].numpy()
        batch = batch_to_device(batch, args.device)
        with torch.inference_mode():
            out = forward_model(model, batch)
        probs = torch.sigmoid(out["contact"]["joint_logits"][0]).float().cpu().numpy()
        fig, score, gt, pred, confidence = _make_figure(
            sample, out, probs, target_gt, target_mask, target_spec, args.threshold)
        path = out_dir / f"sample_{run_index:02d}_idx{ds_index}_f1{score['f1']:.3f}.png"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        record = {
            "dataset_index": ds_index,
            "key": sample["key"],
            "figure": path.name,
            "metrics": score,
            "joint_set": target_spec.joint_set,
            "output_names": list(target_spec.joint_names),
            "gt_contact_joints": [target_spec.joint_names[i] for i in np.flatnonzero(gt)],
            "pred_contact_joints": [target_spec.joint_names[i] for i in np.flatnonzero(pred)],
            "label_confidence": confidence.tolist(),
            "predicted_probability": probs.tolist(),
        }
        records.append(record)
        print(f"[{run_index + 1:02d}/{len(picks):02d}] {sample['key']}  "
              f"F1={score['f1']:.3f} IoU={score['iou']:.3f} -> {path.name}")

    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(state["epoch"]),
        "split": args.split,
        "threshold": args.threshold,
        "joint_set": target_spec.joint_set,
        "output_names": list(target_spec.joint_names),
        "selection": "stratified by 0, 1, and 2+ ground-truth contact outputs",
        "samples": records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Done. Figures and summary: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
