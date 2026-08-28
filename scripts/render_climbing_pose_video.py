"""Render side-by-side pose overlays (frozen vs finetuned) on ClimbingVideos clips.

For each selected test scene the script runs the SAME pose-only build twice —
once freshly initialized (zero-init temporal gates + deepcopy'd head copies =
bit-identical to the frozen SAM3D model) and once with the trained checkpoint
loaded — and writes an mp4 with two panels per frame:

    left  = frozen model      right = finetuned checkpoint

Each panel overlays the predicted MHR mesh (painter-sorted, lambert-shaded,
alpha-blended) plus the kindyn GT 13-keypoint dots (world joints lifted through
the per-frame extrinsics and projected with the per-frame intrinsics). A
per-scene trajectory PNG plots mean-hips camera depth (GT vs frozen vs tuned)
and per-frame 3D / 2D keypoint errors, and ``summary.json`` aggregates the
per-scene mean errors for both passes.

Inference windows follow the config's ``data.sequence`` exactly like
``render_climbing_video_contacts.py`` (every valid (person, frame) predicted
once). ``--num-shards/--shard-index`` split the selected scenes over multiple
launches (one GPU each).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import render_climbing_video_contacts as rcv

from contact import checkpoint as ckpt_io
from contact.config import load_config
from contact.data.climbing_corpus import KP_JOINT_NAMES, ClimbingCorpusDataset
from contact.data.collate import batch_to_device, make_collate
from contact.engine import forward_model
from contact.keypoint_supervision import KP_MHR70_INDICES
from contact.model import build_model
from contact.targets import TargetSpec

_HIP_POSITIONS = (6, 7)  # left_hip / right_hip rows of KP_JOINT_NAMES

MESH_COLOR = (208, 178, 130)      # BGR neutral steel blue
GT_COLOR = (80, 220, 80)          # GT keypoint dots
PRED_KP_COLOR = (200, 80, 220)    # predicted keypoint dots
MESH_ALPHA = 0.55


def _predict_pose_pass(
    model, ds: ClimbingCorpusDataset, scene: str, batch_size: int,
    device: str, collate, seq_cfg: dict,
) -> dict:
    """One full-scene pass; returns per-(person, frame) mesh + keypoint arrays."""
    data = ds._scenes[scene]
    frame_indices = data["frame_indices"]
    fps = float(data["fps"])
    n_people, n_frames = data["valid_mask"].shape
    item_index = rcv._frame_index_map(ds, scene)
    kp_idx = list(KP_MHR70_INDICES)

    out: dict = {}

    def _alloc(n_verts: int):
        out["verts2d"] = np.full((n_people, n_frames, n_verts, 2), np.nan, np.float16)
        out["verts_cam"] = np.full((n_people, n_frames, n_verts, 3), np.nan, np.float16)
        out["kp2d"] = np.full((n_people, n_frames, len(kp_idx), 2), np.nan, np.float32)
        out["kp3d_cam"] = np.full((n_people, n_frames, len(kp_idx), 3), np.nan, np.float32)

    requests_by_t = rcv.sliding_window_requests(
        data["valid_mask"], int(seq_cfg["frames_per_clip"]), int(seq_cfg["frame_stride"]))
    for seq_len in sorted(requests_by_t):
        requests = requests_by_t[seq_len]
        for lo in tqdm(range(0, len(requests), batch_size),
                       desc=f"predict {scene} T={seq_len}", leave=False):
            selected = requests[lo:lo + batch_size]
            frame_cache: dict[tuple[int, int], dict] = {}
            clips = []
            for person, positions, _ in selected:
                start_index = float(frame_indices[positions[0]])
                clip = []
                for position in positions:
                    key = (person, position)
                    if key not in frame_cache:
                        frame_cache[key] = ds[item_index[key]][0]
                    frame = dict(frame_cache[key])
                    frame["frame_pos_sec"] = (
                        float(frame_indices[position]) - start_index) / fps
                    clip.append(frame)
                clips.append(clip)

            batch = batch_to_device(collate(clips), device)
            with torch.inference_mode():
                output = forward_model(model, batch)
            mhr = output["mhr"]
            verts2d = mhr["pred_keypoints_2d_verts"].float().cpu().numpy()
            verts_cam = (
                mhr["pred_vertices"] + mhr["pred_cam_t"][:, None, :]
            ).float().cpu().numpy()
            kp2d = mhr["pred_keypoints_2d"][:, kp_idx].float().cpu().numpy()
            kp3d_cam = (
                mhr["pred_keypoints_3d"][:, kp_idx] + mhr["pred_cam_t"][:, None, :]
            ).float().cpu().numpy()
            if "faces" not in out:
                faces = mhr["faces"]
                faces = faces.cpu().numpy() if torch.is_tensor(faces) else np.asarray(faces)
                out["faces"] = faces.astype(np.int64)
                _alloc(verts2d.shape[1])

            for clip_index, (person, positions, emitted_offsets) in enumerate(selected):
                for offset in emitted_offsets:
                    row = clip_index * seq_len + offset
                    pos = positions[offset]
                    out["verts2d"][person, pos] = verts2d[row]
                    out["verts_cam"][person, pos] = verts_cam[row]
                    out["kp2d"][person, pos] = kp2d[row]
                    out["kp3d_cam"][person, pos] = kp3d_cam[row]
    return out


def _draw_mesh(img: np.ndarray, verts2d: np.ndarray, verts_cam: np.ndarray,
               faces: np.ndarray) -> None:
    """Painter-sorted, lambert-shaded solid mesh, alpha-blended onto ``img``."""
    tri2d = verts2d[faces].astype(np.float32)             # [F, 3, 2]
    tricam = verts_cam[faces].astype(np.float32)          # [F, 3, 3]
    normals = np.cross(tricam[:, 1] - tricam[:, 0], tricam[:, 2] - tricam[:, 0])
    norm = np.linalg.norm(normals, axis=1) + 1e-8
    shade = 0.30 + 0.70 * np.clip(np.abs(normals[:, 2]) / norm, 0.0, 1.0)
    depth = tricam[..., 2].mean(axis=1)
    keep = np.isfinite(tri2d).all(axis=(1, 2)) & np.isfinite(depth) & (depth > 0.05)
    # Cull triangles fully outside the image (cheap bbox test).
    h, w = img.shape[:2]
    xs, ys = tri2d[..., 0], tri2d[..., 1]
    keep &= (xs.max(1) >= 0) & (xs.min(1) < w) & (ys.max(1) >= 0) & (ys.min(1) < h)
    order = np.argsort(-depth[keep])
    tri2d = tri2d[keep][order].astype(np.int32)
    shade = shade[keep][order]
    color = np.asarray(MESH_COLOR, np.float32)
    overlay = img.copy()
    for pts, s in zip(tri2d, shade):
        cv2.fillConvexPoly(overlay, pts, tuple(float(c) for c in color * s))
    cv2.addWeighted(overlay, MESH_ALPHA, img, 1.0 - MESH_ALPHA, 0.0, dst=img)


def _project(points_cam: np.ndarray, intr: np.ndarray) -> np.ndarray:
    """Perspective projection of camera-frame points [..., 3] -> pixels [..., 2]."""
    z = np.clip(points_cam[..., 2:3], 1e-6, None)
    uv = points_cam[..., :2] / z
    return uv * np.array([intr[0, 0], intr[1, 1]]) + np.array([intr[0, 2], intr[1, 2]])


def _draw_kp(img: np.ndarray, pts: np.ndarray, color, radius: int) -> None:
    h, w = img.shape[:2]
    for x, y in pts:
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if -50 <= x < w + 50 and -50 <= y < h + 50:
            cv2.circle(img, (int(x), int(y)), radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)


def _panel(frame: np.ndarray, pass_data: dict, frame_pos: int, gt2d: np.ndarray | None,
           label: str) -> np.ndarray:
    img = frame.copy()
    n_people = pass_data["verts2d"].shape[0]
    for person in range(n_people):
        v2d = pass_data["verts2d"][person, frame_pos].astype(np.float32)
        if not np.isfinite(v2d).any():
            continue
        vcam = pass_data["verts_cam"][person, frame_pos].astype(np.float32)
        _draw_mesh(img, v2d, vcam, pass_data["faces"])
        _draw_kp(img, pass_data["kp2d"][person, frame_pos], PRED_KP_COLOR, 3)
    if gt2d is not None:
        for person in range(gt2d.shape[0]):
            _draw_kp(img, gt2d[person], GT_COLOR, 4)
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (25, 25, 25), -1)
    cv2.putText(img, label, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (240, 240, 240), 2, cv2.LINE_AA)
    return img


def _scene_gt(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GT keypoints in camera frame + pixels: cam [P,N,13,3], px [P,N,13,2], valid [P,N]."""
    kp_w = data["kp3d_world"]                       # [P, N, 13, 3]
    valid = data["kp_valid"] & data["valid_mask"]   # [P, N]
    ext = data["extrinsics"]                        # [N, 4, 4]
    cam = np.einsum("nij,pnkj->pnki", ext[:, :3, :3], kp_w) + ext[None, :, None, :3, 3]
    px = np.full(cam.shape[:-1] + (2,), np.nan, np.float32)
    for n in range(cam.shape[1]):
        px[:, n] = _project(cam[:, n], data["intrinsics"][n])
    cam = np.where(valid[:, :, None, None], cam, np.nan).astype(np.float32)
    px = np.where(valid[:, :, None, None], px, np.nan)
    return cam, px, valid


def _pass_errors(pass_data: dict, gt_cam: np.ndarray, gt_px: np.ndarray) -> dict:
    """Per-frame mean errors vs GT (nan where nothing valid)."""
    err3d = np.linalg.norm(pass_data["kp3d_cam"] - gt_cam, axis=-1)  # [P, N, 13]
    err2d = np.linalg.norm(pass_data["kp2d"] - gt_px, axis=-1)
    depth = pass_data["kp3d_cam"][:, :, _HIP_POSITIONS, 2].mean(axis=2)  # [P, N]
    with np.errstate(invalid="ignore"):
        return {
            "err3d": np.nanmean(err3d, axis=(0, 2)),
            "err2d": np.nanmean(err2d, axis=(0, 2)),
            "depth": np.nanmean(depth, axis=0),
        }


def _trajectory_plot(scene: str, fps: float, gt_cam: np.ndarray,
                     frozen: dict, tuned: dict, path: Path) -> None:
    gt_depth = np.nanmean(gt_cam[:, :, _HIP_POSITIONS, 2].mean(axis=2), axis=0)
    t = np.arange(len(gt_depth)) / fps
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(t, gt_depth, "k-", lw=2, label="kindyn GT")
    axes[0].plot(t, frozen["depth"], color="tab:blue", lw=1.2, label="frozen")
    axes[0].plot(t, tuned["depth"], color="tab:orange", lw=1.2, label="finetuned")
    axes[0].set_ylabel("mean-hips depth [m]")
    axes[0].legend(loc="best")
    axes[1].plot(t, frozen["err3d"], color="tab:blue", lw=1.2, label="frozen")
    axes[1].plot(t, tuned["err3d"], color="tab:orange", lw=1.2, label="finetuned")
    axes[1].set_ylabel("mean 3D kp err [m]")
    axes[1].legend(loc="best")
    axes[2].plot(t, frozen["err2d"], color="tab:blue", lw=1.2, label="frozen")
    axes[2].plot(t, tuned["err2d"], color="tab:orange", lw=1.2, label="finetuned")
    axes[2].set_ylabel("mean 2D kp err [px]")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="best")
    fig.suptitle(f"{scene} — 13 kindyn keypoints, camera frame")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _render_video(scene: str, data: dict, frozen: dict, tuned: dict,
                  gt_px: np.ndarray, path: Path, scale: float) -> dict:
    frames_dir = data["frames_dir"]
    n_frames = len(data["frame_indices"])
    first = cv2.imread(str(frames_dir / "000000.jpg"), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(frames_dir / "000000.jpg")
    height, width = first.shape[:2]
    out_w, out_h = int(round(2 * width * scale)), int(round(height * scale))
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(data["fps"]), (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    try:
        for pos in tqdm(range(n_frames), desc=f"render {scene}", leave=False):
            frame = cv2.imread(str(frames_dir / f"{pos:06d}.jpg"), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(frames_dir / f"{pos:06d}.jpg")
            gt = gt_px[:, pos]
            left = _panel(frame, frozen, pos, gt, "frozen")
            right = _panel(frame, tuned, pos, gt, "finetuned + temporal")
            combo = np.concatenate([left, right], axis=1)
            if scale != 1.0:
                combo = cv2.resize(combo, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(combo)
    finally:
        writer.release()
    return {"frames": n_frames, "size": [out_w, out_h], "fps": float(data["fps"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--num-videos", type=int, default=6)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=12,
                        help="windows per forward (frames = batch-size * T)")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="output video scale factor on the 2-panel canvas")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    output_dir = (args.output_dir if args.output_dir is not None
                  else checkpoint.parent / f"pose_videos_{args.split}_{checkpoint.stem}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    cfg["train"]["compile_backbone"] = False  # no warmup for a render script
    root, contact_level = rcv._dataset_options(cfg)
    scenes = rcv.list_corpus_scenes(root, args.split)
    selected = rcv.select_random_scenes(scenes, args.num_videos, args.seed)
    selected = selected[args.shard_index::args.num_shards]
    print(f"[shard {args.shard_index}/{args.num_shards}] scenes:",
          ", ".join(scene for _, scene in selected))

    spec = TargetSpec.from_config(cfg)
    print("Building model …")
    model, _ = build_model(cfg, args.device)
    model.eval()
    collate = make_collate(tuple(model.cfg.MODEL.IMAGE_SIZE), spec)
    seq_cfg = cfg["data"]["sequence"]

    records = []
    scene_ctx = {}
    frozen_passes = {}
    for video_id, scene in selected:
        ds = ClimbingCorpusDataset(
            root, scenes=[scene], split=args.split,
            frames_per_clip=1, frame_stride=1, jitter=False,
            seed=int(cfg["data"]["seed"]), contact_level=contact_level,
            use_confidence_weights=False, require_labels=False,
            load_keypoints=True,
        )
        data = ds._scenes[scene]
        scene_ctx[scene] = (video_id, ds, data, *_scene_gt(data)[:2])
        # Fresh init = zero temporal gates + identity head copies = frozen model.
        frozen_passes[scene] = _predict_pose_pass(
            model, ds, scene, args.batch_size, args.device, collate, seq_cfg)

    state = ckpt_io.load(checkpoint, model, config=cfg, map_location=args.device)
    print(f"loaded {checkpoint.name} (epoch {state['epoch']})")
    model.eval()

    for video_id, scene in selected:
        _, ds, data, gt_cam, gt_px = scene_ctx[scene]
        tuned = _predict_pose_pass(
            model, ds, scene, args.batch_size, args.device, collate, seq_cfg)
        frozen = frozen_passes.pop(scene)

        errors = {"frozen": _pass_errors(frozen, gt_cam, gt_px),
                  "tuned": _pass_errors(tuned, gt_cam, gt_px)}
        _trajectory_plot(scene, float(data["fps"]), gt_cam,
                         errors["frozen"], errors["tuned"],
                         output_dir / f"{scene}_trajectory.png")
        info = _render_video(scene, data, frozen, tuned, gt_px,
                             output_dir / f"{scene}.mp4", args.scale)

        record = {"video": video_id, "scene": scene, **info}
        for name in ("frozen", "tuned"):
            record[name] = {
                "err3d_m": float(np.nanmean(errors[name]["err3d"])),
                "err3d_median_m": float(np.nanmedian(errors[name]["err3d"])),
                "err2d_px": float(np.nanmean(errors[name]["err2d"])),
                "err2d_median_px": float(np.nanmedian(errors[name]["err2d"])),
                "depth_mae_m": float(np.nanmean(np.abs(
                    errors[name]["depth"]
                    - np.nanmean(gt_cam[:, :, _HIP_POSITIONS, 2].mean(2), 0)))),
            }
        records.append(record)
        print(json.dumps(record, indent=2))

    summary_path = output_dir / f"summary_shard{args.shard_index}.json"
    summary_path.write_text(json.dumps(records, indent=2))
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
