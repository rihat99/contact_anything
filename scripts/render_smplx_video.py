"""Render kindyn-GT vs frozen-MHR vs SMPL-X-head overlays on ClimbingVideos scenes.

Every test scene is run once under the evaluation protocol (one clip per person,
see :mod:`scripts._render_common`) and written as a three-panel mp4:

    left = kindyn SMPL-X GT (hands flat)   middle = frozen MHR   right = SMPL-X head

Each panel carries that arm's mesh (painter-sorted, lambert-shaded,
alpha-blended) and its keypoints; the GT 22-joint body skeleton is drawn in
green on every panel for reference. The banner shows the arm's per-frame
hip-centred 12-joint MPJPE (the 12 joints both skeletons name — shoulders,
elbows, wrists, hips, knees, ankles — which is also how the frozen MHR is
scored on the same 12 named joints). A per-scene PNG plots the
mean-hips camera depth (GT / frozen / head) and the per-frame 12-joint 3D and
2D errors of both arms; a JSON record per scene is printed.

    python scripts/render_smplx_video.py --config configs/smplx_probe.yaml \
        --checkpoint output/<run>/best.pth --scenes 5 --out output/<run>/render
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _render_common as rc                                     # noqa: E402
from render_pose_video import draw_keypoints, draw_mesh, nan_means   # noqa: E402
from model.loss.smplx import SMPLX_HIPS, gt_smplx_camera, smplx_vertices   # noqa: E402

#: The 12 joints both skeletons name: shoulders, elbows, hips, knees, ankles,
#: wrists (left, right) — SMPL-X body indices and the MHR70 keypoint indices.
SMPLX12 = (16, 17, 18, 19, 1, 2, 4, 5, 7, 8, 20, 21)
MHR12 = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 62, 41)
MHR_HIPS = (9, 10)
from train.predict import load_model                            # noqa: E402

#: SMPL-X body kinematic tree (child -> parent), joints 1..21.
SMPLX_PARENTS = (0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
#: Bones among the 12 common joints, as index pairs into the 12-lists.
BONES12 = ((0, 2), (2, 10), (1, 3), (3, 11), (4, 6), (6, 8), (5, 7), (7, 9),
           (0, 1), (4, 5), (0, 4), (1, 5))
GT_BGR, FROZEN_BGR, HEAD_BGR = (80, 220, 80), (208, 178, 130), (120, 120, 235)
ARMS = ("gt", "frozen", "head")
LABELS = {"gt": "kindyn SMPL-X GT", "frozen": "frozen MHR", "head": "SMPL-X head"}
MESH_COLOURS = {"gt": (120, 200, 120), "frozen": FROZEN_BGR, "head": HEAD_BGR}
KP_COLOURS = {"gt": GT_BGR, "frozen": (200, 80, 220), "head": (60, 60, 255)}


def _project(points_cam: np.ndarray, cam_int: np.ndarray) -> np.ndarray:
    return np.stack([rc.project(points_cam[i], cam_int[i]) for i in range(len(points_cam))])


def predict_pass(model, ds, cfg: dict, device: str) -> dict:
    """Run every whole-scene clip; scatter the three arms' meshes and keypoints."""
    data = ds.scene_data(ds.clips[0].scene)
    n_people, n_frames = data["valid_mask"].shape
    body = model.head_smplx.body(torch.device(device))
    smplx_faces = np.asarray(body.structure.faces.cpu().numpy(), np.int64)
    out: dict = {"covered": np.zeros((n_people, n_frames), bool),
                 "faces": {"gt": smplx_faces, "head": smplx_faces}}

    def slot(key, *shape, dtype=np.float32):
        out[key] = np.full((n_people, n_frames) + shape, np.nan, dtype)

    for clip, batch, output in rc.clip_batches(ds, cfg, model, device):
        rows = batch["frame_index"].tolist()
        person = clip.person
        mhr, sx = output["mhr"], output["smplx"]
        dev = torch.device(device)
        cam_int = rc.to_numpy(batch["cam_int"])
        if "faces_frozen_done" not in out:
            faces = mhr["faces"]
            out["faces"]["frozen"] = np.asarray(
                faces.cpu().numpy() if torch.is_tensor(faces) else faces, np.int64)
            out["faces_frozen_done"] = True
            n_mhr = mhr["pred_vertices"].shape[1]
            n_sx = int(body.structure.faces.max()) + 1
            for arm, nv in (("gt", n_sx), ("frozen", n_mhr), ("head", n_sx)):
                slot(f"{arm}_verts2d", nv, 2, dtype=np.float16)
                slot(f"{arm}_verts_cam", nv, 3, dtype=np.float16)
            slot("frozen_kp3d", 70, 3); slot("frozen_kp2d", 70, 2)
            slot("head_kp3d", 22, 3); slot("head_kp2d", 22, 2)
            slot("gt_kp3d", 22, 3); slot("gt_kp2d", 22, 2)

        # frozen MHR: root-relative vertices/keypoints placed by pred_cam_t
        frozen_verts = rc.to_numpy(mhr["pred_vertices"] + mhr["pred_cam_t"][:, None])
        frozen_kp3d = rc.to_numpy(mhr["pred_keypoints_3d"] + mhr["pred_cam_t"][:, None])
        # SMPL-X head: its own q / betas through the same body -> vertices
        head_data = body.with_shape(betas=sx["betas"]).fk(sx["q_cam"])
        head_verts = rc.to_numpy(body.vertices_from_data(head_data))
        head_kp3d = rc.to_numpy(sx["joints_cam"][:, :22])
        # GT: kindyn SMPL-X lifted to the camera through the head's own body
        gt = gt_smplx_camera(batch, dev, hands=model.head_smplx.hands)
        gt_verts = rc.to_numpy(smplx_vertices(body, gt["betas"], gt["q"]))
        gt_kp3d = rc.to_numpy(gt["joints"][:, :22])
        gt_valid = rc.to_numpy(batch["smplx_valid"]) > 0

        arms = {"frozen": (frozen_verts, frozen_kp3d), "head": (head_verts, head_kp3d),
                "gt": (gt_verts, gt_kp3d)}
        for row, position in enumerate(rows):
            out["covered"][person, position] = True
            for arm, (verts, kp3d) in arms.items():
                if arm == "gt" and not gt_valid[row]:
                    continue
                out[f"{arm}_verts_cam"][person, position] = verts[row]
                out[f"{arm}_verts2d"][person, position] = rc.project(verts[row], cam_int[row])
                out[f"{arm}_kp3d"][person, position] = kp3d[row]
                out[f"{arm}_kp2d"][person, position] = rc.project(kp3d[row], cam_int[row])
    return out


def hip_centred_12(kp12: np.ndarray) -> np.ndarray:
    """Centre a ``(..., 12, 3)`` common-joint set at its own mean hips."""
    return kp12 - kp12[..., 4:6, :].mean(axis=-2, keepdims=True)


def pass_errors(pred: dict) -> dict:
    """Per-frame errors of both arms vs the GT, NaN where nothing is valid."""
    gt12 = pred["gt_kp3d"][:, :, list(SMPLX12)]
    gt12_px = pred["gt_kp2d"][:, :, list(SMPLX12)]
    gt_hips = pred["gt_kp3d"][:, :, list(SMPLX_HIPS)].mean(axis=2)
    arms = {"frozen": (pred["frozen_kp3d"][:, :, list(MHR12)],
                       pred["frozen_kp2d"][:, :, list(MHR12)],
                       pred["frozen_kp3d"][:, :, list(MHR_HIPS)].mean(axis=2)),
            "head": (pred["head_kp3d"][:, :, list(SMPLX12)],
                     pred["head_kp2d"][:, :, list(SMPLX12)],
                     pred["head_kp3d"][:, :, list(SMPLX_HIPS)].mean(axis=2))}
    errors = {"gt_depth": None}
    with nan_means():
        errors["gt_depth"] = np.nanmean(gt_hips[..., 2], axis=0)
        for arm, (kp12, px12, hips) in arms.items():
            errors[arm] = {
                "mpjpe12_mm": 1000.0 * np.nanmean(np.linalg.norm(
                    hip_centred_12(kp12) - hip_centred_12(gt12), axis=-1), axis=(0, 2)),
                "per_person_mpjpe12_mm": 1000.0 * np.nanmean(np.linalg.norm(
                    hip_centred_12(kp12) - hip_centred_12(gt12), axis=-1), axis=2),
                "err2d_px": np.nanmean(np.linalg.norm(px12 - gt12_px, axis=-1), axis=(0, 2)),
                "hips_mm": 1000.0 * np.nanmean(np.linalg.norm(hips - gt_hips, axis=-1), axis=0),
                "depth": np.nanmean(hips[..., 2], axis=0),
            }
    return errors


def draw_bones(img: np.ndarray, points: np.ndarray, bones, colour, thickness: int) -> None:
    height, width = img.shape[:2]
    for a, b in bones:
        pa, pb = points[a], points[b]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        if max(abs(pa).max(), abs(pb).max()) > 4 * max(height, width):
            continue
        cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), colour,
                 thickness, cv2.LINE_AA)


def panel(frame: np.ndarray, pred: dict, errors: dict, arm: str, position: int) -> np.ndarray:
    img = frame.copy()
    n_people = pred["covered"].shape[0]
    for person in range(n_people):
        if not pred["covered"][person, position]:
            continue
        verts2d = pred[f"{arm}_verts2d"][person, position].astype(np.float32)
        if np.isfinite(verts2d).all():
            draw_mesh(img, verts2d, pred[f"{arm}_verts_cam"][person, position].astype(np.float32),
                      pred["faces"][arm], MESH_COLOURS[arm])
        kp2d = pred[f"{arm}_kp2d"][person, position]
        if arm == "frozen":
            draw_bones(img, kp2d[list(MHR12)], BONES12, KP_COLOURS[arm], 2)
            draw_keypoints(img, kp2d[list(MHR12)], KP_COLOURS[arm], 3)
        else:
            bones = [(child + 1, parent) for child, parent in enumerate(SMPLX_PARENTS)]
            draw_bones(img, kp2d, bones, KP_COLOURS[arm], 2)
            draw_keypoints(img, kp2d, KP_COLOURS[arm], 3)
        # the GT skeleton on every panel, thin, for reference
        if arm != "gt":
            gt = pred["gt_kp2d"][person, position]
            draw_bones(img, gt, [(c + 1, p) for c, p in enumerate(SMPLX_PARENTS)], GT_BGR, 1)
            draw_keypoints(img, gt, GT_BGR, 2)
    text = LABELS[arm]
    if arm != "gt":
        value = errors[arm]["mpjpe12_mm"][position]
        if np.isfinite(value):
            text += f"  {value:.0f} mm"                 # hip-centred 12-joint MPJPE
    font = min(0.8, img.shape[1] / 640.0)
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (25, 25, 25), -1)
    cv2.putText(img, text, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, font, (240, 240, 240), 2,
                cv2.LINE_AA)
    return img


def trajectory_plot(scene: str, times: np.ndarray, errors: dict, positions: np.ndarray,
                    path: Path) -> None:
    figure, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    axes[0].plot(times, errors["gt_depth"][positions], "k-", lw=2, label="kindyn GT")
    axes[0].set_ylabel("mean-hips depth [m]")
    axes[1].set_ylabel("hip-centred 12-joint MPJPE [mm]")
    axes[2].set_ylabel("abs mean-hips error [mm]")
    axes[3].set_ylabel("12-joint 2D error [px]")
    axes[3].set_xlabel("time [s]")
    for arm, colour in (("frozen", "tab:blue"), ("head", "tab:red")):
        axes[0].plot(times, errors[arm]["depth"][positions], color=colour, lw=1.2, label=arm)
        axes[1].plot(times, errors[arm]["mpjpe12_mm"][positions], color=colour, lw=1.2, label=arm)
        axes[2].plot(times, errors[arm]["hips_mm"][positions], color=colour, lw=1.2, label=arm)
        axes[3].plot(times, errors[arm]["err2d_px"][positions], color=colour, lw=1.2, label=arm)
    for axis in axes:
        axis.legend(loc="best")
    figure.suptitle(f"{scene} — frozen MHR vs SMPL-X head against the kindyn SMPL-X GT")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def render_video(ds, scene: str, pred: dict, errors: dict, positions: np.ndarray,
                 scale: float, path: Path) -> dict:
    data = ds.scene_data(scene)
    fps = float(data["fps"]) / ds.scene_stride(scene)
    first = rc.read_frame(data["frames_dir"], int(positions[0]))
    height, width = first.shape[:2]
    size = (int(round(3 * width * scale)), int(round(height * scale)))
    writer = rc.open_writer(path, fps, size)
    try:
        for position in positions:
            frame = rc.read_frame(data["frames_dir"], int(position))
            combined = np.concatenate(
                [panel(frame, pred, errors, arm, position) for arm in ARMS], axis=1)
            if scale != 1.0:
                combined = cv2.resize(combined, size, interpolation=cv2.INTER_AREA)
            writer.write(combined)
    finally:
        writer.release()
    return {"frames": int(positions.size), "size": list(size), "fps": round(fps, 3)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True,
                        help="checkpoint path, or 'none' (the untrained mean-body head)")
    parser.add_argument("--scenes", default=None,
                        help="scene count (e.g. '5') or a comma-separated id list")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="scale factor of the three-panel canvas")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="clip-length cap (default: data.eval_max_frames)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    checkpoint = None if str(args.checkpoint).lower() == "none" else args.checkpoint
    model, cfg = load_model(args.config, checkpoint, args.device)
    if model.head_smplx is None:
        raise ValueError("the config has no model.smplx head to render")
    root, contact_level = rc.dataset_spec(cfg)
    max_frames = args.max_frames or int(cfg["data"]["eval_max_frames"])
    scenes, rank, world_size = rc.shard(rc.resolve_scenes(root, "test", args.scenes))
    print(f"[rank {rank}/{world_size}] {len(scenes)} scene(s) on {args.device}; "
          f"checkpoint = {checkpoint or 'none (untrained)'}")

    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for index, scene in enumerate(scenes, start=1):
        ds = rc.build_dataset(cfg, root, contact_level, scene, "test", {"smplx"}, max_frames)
        pred = predict_pass(model, ds, cfg, args.device)
        positions = np.flatnonzero(pred["covered"].any(axis=0))
        if positions.size == 0:
            raise RuntimeError(f"{scene}: no frame was covered by an evaluation clip")
        errors = pass_errors(pred)
        times = (positions - positions[0]) / float(ds.scene_data(scene)["fps"])
        trajectory_plot(scene, times, errors, positions, args.out / f"{scene}_trajectory.png")
        record = {"scene": scene, **render_video(
            ds, scene, pred, errors, positions, args.scale, args.out / f"{scene}.mp4")}
        with nan_means():
            for arm in ("frozen", "head"):
                record[arm] = {
                    "mpjpe12_mm": float(np.nanmean(errors[arm]["mpjpe12_mm"][positions])),
                    "hips_mm": float(np.nanmean(errors[arm]["hips_mm"][positions])),
                    "err2d_px": float(np.nanmean(errors[arm]["err2d_px"][positions])),
                    "depth_mae_m": float(np.nanmean(np.abs(
                        errors[arm]["depth"][positions] - errors["gt_depth"][positions]))),
                }
        records.append(record)
        print(f"[rank {rank}] [{index}/{len(scenes)}] {json.dumps(record, indent=2)}")
    (args.out / f"records_rank{rank}.json").write_text(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
