"""Render frozen-vs-checkpoint pose overlays on ClimbingVideos scenes.

Each scene is run twice — once by the untrained build (zero-gated temporal
bricks + deepcopy'd head copies, i.e. bit-identical to the frozen SAM-3D-Body
model) and once with the checkpoint loaded — and written as a two-panel mp4:

    left = frozen                     right = checkpoint

Every panel carries the predicted MHR mesh (painter-sorted, lambert-shaded,
alpha-blended), the predicted MHR70 keypoints and the GT MHR70 keypoints
(``mhr_sup_1`` world keypoints lifted through the scene's per-frame extrinsics
and projected with its intrinsics). A per-scene PNG plots mean-hips camera depth
(GT / frozen / checkpoint) and the per-frame 3D and 2D keypoint errors, and the
same errors are printed per scene.

Only the frames of the evaluation clip are written (see
:mod:`scripts._render_common`); under ``torchrun`` the scenes are sharded over
ranks.

    python scripts/render_pose_video.py --config configs/allmod_rope_t60_gv.yaml \
        --checkpoint output/<run>/best.pth --scenes 5 --out output/<run>/pose
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _render_common as rc                                     # noqa: E402
from train.checkpoint import load as load_checkpoint            # noqa: E402
from train.checkpoint import trainable_state_dict               # noqa: E402
from train.predict import load_model                            # noqa: E402

#: MHR70 left/right hip rows — GT and prediction share this index space.
HIPS = (9, 10)
MESH_BGR = (208, 178, 130)
GT_KP_BGR = (80, 220, 80)
PRED_KP_BGR = (200, 80, 220)
MESH_ALPHA = 0.55


@contextmanager
def nan_means():
    """Silence the all-NaN-slice warning: uncovered frames are NaN by design."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def predict_pass(model, ds, cfg: dict, device: str, with_gt: bool) -> dict:
    """Run every whole-scene clip; scatter meshes, keypoints and (once) the GT."""
    data = ds.scene_data(ds.clips[0].scene)
    n_people, n_frames = data["valid_mask"].shape
    out: dict = {
        "kp2d": np.full((n_people, n_frames, 70, 2), np.nan, np.float32),
        "kp3d_cam": np.full((n_people, n_frames, 70, 3), np.nan, np.float32),
        "covered": np.zeros((n_people, n_frames), bool),
    }
    if with_gt:
        out["gt_cam"] = np.full((n_people, n_frames, 70, 3), np.nan, np.float32)
        out["gt_px"] = np.full((n_people, n_frames, 70, 2), np.nan, np.float32)

    for clip, batch, output in rc.clip_batches(ds, cfg, model, device):
        rows = batch["frame_index"].tolist()
        person = clip.person
        mhr = output["mhr"]
        verts2d = rc.to_numpy(mhr["pred_keypoints_2d_verts"])
        verts_cam = rc.to_numpy(mhr["pred_vertices"] + mhr["pred_cam_t"][:, None, :])
        kp2d = rc.to_numpy(mhr["pred_keypoints_2d"])
        kp3d_cam = rc.to_numpy(mhr["pred_keypoints_3d"] + mhr["pred_cam_t"][:, None, :])
        if "faces" not in out:
            faces = mhr["faces"]
            out["faces"] = np.asarray(
                faces.cpu().numpy() if torch.is_tensor(faces) else faces, np.int64)
            out["verts2d"] = np.full(
                (n_people, n_frames, verts2d.shape[1], 2), np.nan, np.float16)
            out["verts_cam"] = np.full(
                (n_people, n_frames, verts2d.shape[1], 3), np.nan, np.float16)
        if with_gt:
            # GT keypoints are world-frame metres; the camera frame and its
            # projection are the scene's own, per frame.
            extrinsics = rc.to_numpy(batch["cam_from_world"])
            gt_world = rc.to_numpy(batch["kp3d_world"])
            gt_cam = (np.einsum("nij,nkj->nki", extrinsics[:, :3, :3], gt_world)
                      + extrinsics[:, None, :3, 3])
            gt_valid = rc.to_numpy(batch["kp_valid"]) > 0
            cam_int = rc.to_numpy(batch["cam_int"])

        for row, position in enumerate(rows):
            out["covered"][person, position] = True
            out["verts2d"][person, position] = verts2d[row]
            out["verts_cam"][person, position] = verts_cam[row]
            out["kp2d"][person, position] = kp2d[row]
            out["kp3d_cam"][person, position] = kp3d_cam[row]
            if with_gt and gt_valid[row]:
                out["gt_cam"][person, position] = gt_cam[row]
                out["gt_px"][person, position] = rc.project(gt_cam[row], cam_int[row])
    return out


def draw_mesh(img: np.ndarray, verts2d: np.ndarray, verts_cam: np.ndarray,
              faces: np.ndarray) -> None:
    """Painter-sorted, lambert-shaded solid mesh alpha-blended onto ``img``."""
    tri2d = verts2d[faces].astype(np.float32)                   # [F, 3, 2]
    tricam = verts_cam[faces].astype(np.float32)                # [F, 3, 3]
    normals = np.cross(tricam[:, 1] - tricam[:, 0], tricam[:, 2] - tricam[:, 0])
    shade = 0.30 + 0.70 * np.clip(
        np.abs(normals[:, 2]) / (np.linalg.norm(normals, axis=1) + 1e-8), 0.0, 1.0)
    depth = tricam[..., 2].mean(axis=1)
    height, width = img.shape[:2]
    xs, ys = tri2d[..., 0], tri2d[..., 1]
    keep = (np.isfinite(tri2d).all(axis=(1, 2)) & np.isfinite(depth) & (depth > 0.05)
            & (xs.max(1) >= 0) & (xs.min(1) < width)
            & (ys.max(1) >= 0) & (ys.min(1) < height))
    order = np.argsort(-depth[keep])
    colour = np.asarray(MESH_BGR, np.float32)
    overlay = img.copy()
    for points, factor in zip(tri2d[keep][order].astype(np.int32), shade[keep][order]):
        cv2.fillConvexPoly(overlay, points, tuple(float(c) for c in colour * factor))
    cv2.addWeighted(overlay, MESH_ALPHA, img, 1.0 - MESH_ALPHA, 0.0, dst=img)


def draw_keypoints(img: np.ndarray, points: np.ndarray, colour, radius: int) -> None:
    height, width = img.shape[:2]
    for x, y in points:
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if -50 <= x < width + 50 and -50 <= y < height + 50:
            cv2.circle(img, (int(x), int(y)), radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), radius, colour, -1, cv2.LINE_AA)


def panel(frame: np.ndarray, predictions: dict, gt_px: np.ndarray, position: int,
          label: str) -> np.ndarray:
    img = frame.copy()
    for person in range(predictions["covered"].shape[0]):
        if not predictions["covered"][person, position]:
            continue
        draw_mesh(img, predictions["verts2d"][person, position].astype(np.float32),
                  predictions["verts_cam"][person, position].astype(np.float32),
                  predictions["faces"])
        draw_keypoints(img, predictions["kp2d"][person, position], PRED_KP_BGR, 3)
    for person in range(gt_px.shape[0]):
        draw_keypoints(img, gt_px[person, position], GT_KP_BGR, 4)
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (25, 25, 25), -1)
    cv2.putText(img, label, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (240, 240, 240), 2, cv2.LINE_AA)
    return img


def pass_errors(predictions: dict, gt_cam: np.ndarray, gt_px: np.ndarray) -> dict:
    """Per-frame mean keypoint errors and mean-hips depth (NaN where nothing valid)."""
    with nan_means():
        return {
            "err3d": np.nanmean(
                np.linalg.norm(predictions["kp3d_cam"] - gt_cam, axis=-1), axis=(0, 2)),
            "err2d": np.nanmean(
                np.linalg.norm(predictions["kp2d"] - gt_px, axis=-1), axis=(0, 2)),
            "depth": np.nanmean(
                predictions["kp3d_cam"][:, :, HIPS, 2].mean(axis=2), axis=0),
        }


def trajectory_plot(scene: str, times: np.ndarray, gt_depth: np.ndarray,
                    frozen: dict, tuned: dict, positions: np.ndarray,
                    path: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(times, gt_depth[positions], "k-", lw=2, label="kindyn GT")
    axes[0].set_ylabel("mean-hips depth [m]")
    axes[1].set_ylabel("mean 3D kp err [m]")
    axes[2].set_ylabel("mean 2D kp err [px]")
    axes[2].set_xlabel("time [s]")
    for name, errors, colour in (("frozen", frozen, "tab:blue"),
                                 ("checkpoint", tuned, "tab:orange")):
        axes[0].plot(times, errors["depth"][positions], color=colour, lw=1.2, label=name)
        axes[1].plot(times, errors["err3d"][positions], color=colour, lw=1.2, label=name)
        axes[2].plot(times, errors["err2d"][positions], color=colour, lw=1.2, label=name)
    for axis in axes:
        axis.legend(loc="best")
    figure.suptitle(f"{scene} — 70 MHR70 keypoints, camera frame")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def render_video(ds, scene: str, frozen: dict, tuned: dict, gt_px: np.ndarray,
                 positions: np.ndarray, scale: float, path: Path) -> dict:
    data = ds.scene_data(scene)
    fps = float(data["fps"]) / ds.scene_stride(scene)
    first = rc.read_frame(data["frames_dir"], int(positions[0]))
    height, width = first.shape[:2]
    size = (int(round(2 * width * scale)), int(round(height * scale)))
    writer = rc.open_writer(path, fps, size)
    try:
        for position in positions:
            frame = rc.read_frame(data["frames_dir"], int(position))
            combined = np.concatenate(
                [panel(frame, frozen, gt_px, position, "frozen"),
                 panel(frame, tuned, gt_px, position, "checkpoint")], axis=1)
            if scale != 1.0:
                combined = cv2.resize(combined, size, interpolation=cv2.INTER_AREA)
            writer.write(combined)
    finally:
        writer.release()
    return {"frames": int(positions.size), "size": list(size), "fps": round(fps, 3)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", default="none",
                        help="checkpoint path for the right panel, or 'none' "
                             "(both panels then show the frozen model)")
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--scenes", default=None,
                        help="scene count (e.g. '5') or a comma-separated id list")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="scale factor of the two-panel canvas")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="test-split clip-length cap "
                             "(default: data.eval_max_frames; inert on --split train, "
                             "which runs data.clip.frames tiles)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    checkpoint = None if str(args.checkpoint).lower() == "none" else args.checkpoint
    # One resident build, two trainable states: `load_model(config, None)` IS
    # the frozen arm, and loading the checkpoint into it gives the other. A
    # second full build would duplicate the 6 GB frozen base for nothing.
    model, cfg = load_model(args.config, None, args.device)
    frozen_state = {k: v.detach().clone() for k, v in trainable_state_dict(model).items()}
    tuned_state = frozen_state
    if checkpoint is not None:
        load_checkpoint(checkpoint, model, map_location=args.device)
        tuned_state = {k: v.detach().clone()
                       for k, v in trainable_state_dict(model).items()}
        model.load_state_dict(frozen_state, strict=False)
    root, contact_level = rc.dataset_spec(cfg)
    max_frames = args.max_frames or int(cfg["data"]["eval_max_frames"])

    scenes, rank, world_size = rc.shard(
        rc.resolve_scenes(root, args.split, args.scenes))
    print(f"[rank {rank}/{world_size}] {len(scenes)} scene(s) on {args.device}; "
          f"right panel = {checkpoint or 'none (untrained)'}")

    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for index, scene in enumerate(scenes, start=1):
        ds = rc.build_dataset(cfg, root, contact_level, scene, args.split,
                              {"keypoints"}, max_frames)
        frozen = predict_pass(model, ds, cfg, args.device, with_gt=True)
        model.load_state_dict(tuned_state, strict=False)
        tuned = predict_pass(model, ds, cfg, args.device, with_gt=False)
        model.load_state_dict(frozen_state, strict=False)
        gt_cam, gt_px = frozen["gt_cam"], frozen["gt_px"]
        positions = np.flatnonzero(frozen["covered"].any(axis=0))
        if positions.size == 0:
            raise RuntimeError(f"{scene}: no frame was covered by an evaluation clip")

        errors = {"frozen": pass_errors(frozen, gt_cam, gt_px),
                  "checkpoint": pass_errors(tuned, gt_cam, gt_px)}
        with nan_means():
            gt_depth = np.nanmean(gt_cam[:, :, HIPS, 2].mean(axis=2), axis=0)
        times = (positions - positions[0]) / float(ds.scene_data(scene)["fps"])
        trajectory_plot(scene, times, gt_depth, errors["frozen"], errors["checkpoint"],
                        positions, args.out / f"{scene}_trajectory.png")
        record = {"scene": scene,
                  **render_video(ds, scene, frozen, tuned, gt_px, positions,
                                 args.scale, args.out / f"{scene}.mp4")}
        for name in ("frozen", "checkpoint"):
            record[name] = {
                "err3d_m": float(np.nanmean(errors[name]["err3d"][positions])),
                "err3d_median_m": float(np.nanmedian(errors[name]["err3d"][positions])),
                "err2d_px": float(np.nanmean(errors[name]["err2d"][positions])),
                "err2d_median_px": float(np.nanmedian(errors[name]["err2d"][positions])),
                "depth_mae_m": float(np.nanmean(np.abs(
                    errors[name]["depth"][positions] - gt_depth[positions]))),
            }
        records.append(record)
        print(f"[rank {rank}] [{index}/{len(scenes)}] {json.dumps(record, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
