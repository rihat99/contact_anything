"""Qualitative clip inference for a ClimbingVideos joint/force checkpoint.

Each sampled item is a clip of the config's ``data.sequence.frames_per_clip``
frames, forwarded whole so temporal / force attention (and the physics-frame
window) see the same cross-frame context they were trained with; the figure is
rendered for the clip's center frame.

Each figure contains that frame, the frozen model's predicted MHR pose, the
confidence-coloured ground-truth canonical skeleton, and the predicted canonical
skeleton. ClimbingVideos does not provide projected joint positions, so the
contact skeleton is deliberately schematic rather than overlaid.

When the checkpoint has a force head, per-extremity **force arrows** are drawn on
the predicted-pose panel (anchor = the extremity's 2D keypoint, direction = the
3D force projected through the model's intrinsics, length ∝ magnitude in body
weights). Lacking a trained force checkpoint, ``--warm-start`` builds the untrained
force branch from the config's ``model.init_contact_checkpoint`` (zero-init head →
no visible arrows until trained).
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

# One arrow colour per force output (left_hand, right_hand, left_foot, right_foot).
FORCE_COLORS = ("#e0530f", "#f0a500", "#1c72d8", "#2fb3ad")
FORCE_PIXELS_PER_BW = 70.0      # arrow pixel length per unit body weight of |f|
FORCE_MIN_BW = 1.0e-3           # skip near-zero forces (e.g. a zero-init head)


def _draw_force_arrows(ax, mhr: dict, forces: np.ndarray, anchor_indices, frame: str) -> None:
    """Overlay per-extremity predicted force arrows on a full-image axes.

    Anchor = the extremity's projected 2D keypoint; direction = the predicted 3D
    force projected through the model's own intrinsics; pixel length ∝ magnitude
    in body weights; colour by extremity. Only the ``local_world_aligned`` frame
    is drawn — the joint-local frame needs FK not run here.

    :param mhr: ``out["mhr"]`` (reads ``pred_keypoints_3d``, ``pred_keypoints_2d``,
        ``pred_cam_t``, ``focal_length``).
    :param forces: ``[K, 3]`` predicted forces (body weight, head frame).
    :param anchor_indices: MHR70 keypoint indices the force tokens are anchored to.
    :param frame: ``model.force_head.frame``.
    """
    if frame != "local_world_aligned":
        return
    kp3d = mhr["pred_keypoints_3d"][0].cpu().numpy()          # [70, 3] camera frame
    kp2d = mhr["pred_keypoints_2d"][0].cpu().numpy()          # [70, 2] full-img px
    cam_t = mhr["pred_cam_t"][0].cpu().numpy()                # [3]
    focal = float(mhr["focal_length"][0].cpu())
    flip = np.array([1.0, -1.0, -1.0])                        # LWA (cam y-up) -> y-down
    for out_idx, anchor_idx in enumerate(anchor_indices):
        f_pred = np.asarray(forces[out_idx], dtype=np.float64)
        mag = float(np.linalg.norm(f_pred))
        if mag < FORCE_MIN_BW:
            continue
        point_cam = kp3d[anchor_idx] + cam_t                  # camera-space anchor
        if point_cam[2] <= 1e-3:                              # behind the camera
            continue
        anchor = kp2d[anchor_idx]
        # Principal point that reproduces the model's own 2D keypoint at this anchor,
        # so the pinhole projection of a step along the force is exactly consistent.
        cx = anchor[0] - focal * point_cam[0] / point_cam[2]
        cy = anchor[1] - focal * point_cam[1] / point_cam[2]
        step_cam = point_cam + (0.05 / mag) * (flip * f_pred)   # 5 cm along force dir
        tip_x = focal * step_cam[0] / step_cam[2] + cx
        tip_y = focal * step_cam[1] / step_cam[2] + cy
        direction = np.array([tip_x - anchor[0], tip_y - anchor[1]])
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            continue
        tip = anchor + (FORCE_PIXELS_PER_BW * mag) * (direction / norm)
        color = FORCE_COLORS[out_idx % len(FORCE_COLORS)]
        ax.annotate("", xy=(tip[0], tip[1]), xytext=(anchor[0], anchor[1]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.6,
                                    shrinkA=0, shrinkB=0), zorder=5)
        ax.text(tip[0], tip[1], f"{mag:.2f}", color=color, fontsize=8.5,
                fontweight="bold", ha="left", va="center", zorder=6)


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
    force_anchor_indices=None,
    force_frame: str | None = None,
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
    pose_title = "Predicted pose (frozen MHR)"
    draw_force = out.get("force") is not None and force_anchor_indices is not None
    if draw_force:
        pose_title = "Predicted pose (frozen MHR) + force arrows"
    overlay_mesh_on_image_2d(axes[0, 1], image, pred_v2, pred_vcam, mhr_faces,
                             None, pose_title)
    if draw_force:
        _draw_force_arrows(
            axes[0, 1], out["mhr"], out["force"]["joint_forces"][0].cpu().numpy(),
            force_anchor_indices, str(force_frame))

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
        entry = manifest.get(key)
        if entry is None or "val" not in entry:
            detail = (
                " — this run was trained with data.eval_split=test, so its manifest "
                "holds train/test scene lists (there is no val split); use --split test"
                if entry is not None and "test" in entry else "")
            raise RuntimeError(
                f"checkpoint split manifest has no val split for {key!r}{detail}")
        val_videos = set(entry["val"])
        scenes = [scene for scene in scenes if video_id_from_scene(scene) in val_videos]
    # Match the training-time clip so temporal / force checkpoints see the same
    # cross-frame context they were trained with (a T=1 clip would collapse the
    # temporal attention and the physics-frame window). The center frame is rendered.
    seq = cfg["data"]["sequence"]
    return ClimbingVideosDataset(
        root,
        scenes=scenes,
        mode="val",
        split_dir=split_dir,
        frames_per_clip=int(seq["frames_per_clip"]),
        frame_stride=int(seq["frame_stride"]),
        jitter=False,
        seed=int(cfg["data"]["seed"]),
        use_confidence_weights=bool(
            cfg["contact"]["targets"]["joint"]["use_confidence_weights"]),
    )


def _clip_bucket(ds: ClimbingVideosDataset, spec: TargetSpec, index: int) -> int:
    """Contact-count bucket (0 / 1 / 2+) of the clip's RENDERED (center) frame.

    ``main`` renders ``clip[T // 2]``, which for a val-mode (jitter-off) clip is
    source frame ``base + (T // 2) * stride`` — the same sampling arithmetic
    ``__getitem__`` uses. Stratifying by the clip's *base* frame instead would
    label the figure by a frame up to ``(T // 2) * stride`` source frames away
    from the one actually shown (16 frames at T=16 stride 2).
    """
    scene, person, base, _ = ds._items[index]
    data = ds._scenes[scene]
    center_pos = base + (ds.T // 2) * ds.stride
    contacts = torch.as_tensor(
        data["joint_contact"][person, center_pos], dtype=torch.float32)
    supervised = torch.full((22,), float(data["valid_mask"][person, center_pos]))
    if data["annotated"] is not None:
        supervised *= torch.as_tensor(data["annotated"][person, center_pos])
    confidence = (
        torch.ones(22) if data["contact_conf"] is None else
        torch.as_tensor(np.nan_to_num(
            data["contact_conf"][person, center_pos], nan=0.0, posinf=1.0, neginf=0.0
        ).clip(0.0, 1.0))
    )
    if spec.joint_set == "extremities_4":
        contacts, supervised, _ = reduce_body22_to_extremities(
            contacts, supervised, confidence)
    n_contact = int(((contacts > 0.5) & (supervised > 0)).sum())
    return min(n_contact, 2)


def _stratified_picks(
    ds: ClimbingVideosDataset, spec: TargetSpec, count: int, seed: int,
) -> list[int]:
    buckets = {0: [], 1: [], 2: []}
    for index in range(len(ds._items)):
        buckets[_clip_bucket(ds, spec, index)].append(index)
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


def _select_frame(value, index: int, num_frames: int):
    """Slice a model-output structure down to a single frame, keeping a leading
    batch dim of length 1 so the figure helpers' ``[0]`` indexing selects that frame.

    The clip is forwarded flat (``[T, ...]``) so temporal / force attention see the
    whole window; the figure is then rendered for one frame. Tensors / lists whose
    leading length is the clip length are sliced; anything else passes through.
    """
    if torch.is_tensor(value):
        if value.dim() >= 1 and value.shape[0] == num_frames:
            return value[index:index + 1]
        return value
    if isinstance(value, dict):
        return {k: _select_frame(v, index, num_frames) for k, v in value.items()}
    if isinstance(value, (list, tuple)) and len(value) == num_frames:
        return type(value)([value[index]])
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument(
        "--warm-start", action="store_true",
        help="build an untrained force branch warm-started from the config's "
             "model.init_contact_checkpoint (no force checkpoint yet). The zero-init "
             "force head predicts no force, so arrows appear only once trained.")
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "climbing_videos_joint.yaml")
    ap.add_argument("--num-samples", type=int, default=12)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print("Building joint-contact model …")
    model, _ = build_model(cfg, args.device)
    if args.warm_start:
        init_ckpt = cfg["model"].get("init_contact_checkpoint")
        if not init_ckpt:
            ap.error("--warm-start requires model.init_contact_checkpoint in the config")
        state = ckpt_io.initialize_common_contact(
            init_ckpt, model, config=cfg, map_location=args.device)
        ref_path = Path(init_ckpt).resolve()
    elif args.checkpoint:
        state = ckpt_io.load(args.checkpoint, model, config=cfg, map_location=args.device)
        ref_path = Path(args.checkpoint).resolve()
    else:
        ap.error("--checkpoint is required unless --warm-start is given")
    model.eval()

    threshold_tag = f"{args.threshold:.2f}".replace(".", "")
    out_dir = (Path(args.output_dir) if args.output_dir else
               ref_path.parent / f"inference_{args.split}_last_t{threshold_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    force_anchor_indices = None
    force_frame = None
    if cfg["model"]["force_head"]["enabled"]:
        force_anchor_indices = cfg["model"]["contact_head"].get("contact_keypoint_indices")
        force_frame = cfg["model"]["force_head"]["frame"]

    ds = _video_dataset(cfg, state, args.split)
    target_spec = TargetSpec.from_config(cfg)
    picks = _stratified_picks(
        ds, target_spec, min(args.num_samples, len(ds)), args.seed)
    collate = make_collate(tuple(model.cfg.MODEL.IMAGE_SIZE), target_spec)
    print(f"{args.split.capitalize()} frames: {len(ds)}; rendering {len(picks)} "
          f"from last checkpoint at threshold {args.threshold:.2f}")

    records = []
    for run_index, ds_index in enumerate(picks):
        clip = ds[ds_index]                       # list of T frame dicts
        num_frames = len(clip)
        center = num_frames // 2                   # frame rendered from the clip
        batch = collate([clip])                    # flat [T, ...] batch, seq_len=T
        target_gt = batch["targets"]["joint"]["gt"][center].numpy()
        target_mask = batch["targets"]["joint"]["mask"][center].numpy()
        batch = batch_to_device(batch, args.device)
        with torch.inference_mode():
            out = forward_model(model, batch)
        # Reduce the clip's outputs to the center frame so the figure helpers (which
        # index row 0) render that frame's pose / contacts / forces.
        out = _select_frame(out, center, num_frames)
        sample = clip[center]
        probs = torch.sigmoid(out["contact"]["joint_logits"][0]).float().cpu().numpy()
        fig, score, gt, pred, confidence = _make_figure(
            sample, out, probs, target_gt, target_mask, target_spec, args.threshold,
            force_anchor_indices=force_anchor_indices, force_frame=force_frame)
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
        "checkpoint": str(ref_path),
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
