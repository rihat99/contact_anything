"""Predict contacts and 3D forces on BetterVideoReconstruction out-trees.

For every scene ``<out-root>/<stem>/`` with pipeline inputs
(``sam3/bboxes.npz``, ``geometry/transform.npz``) the source video is decoded to
frames and the checkpoint is run over :class:`data.reconstruction.
ReconstructionSceneDataset`'s clips — tiled windows of the config's clip length
at stride 1, so every frame of a window-sized valid run is predicted. Rows no
window covers stay NaN.

Two files are written into ``<stem>/predictions/``:

``contacts.npz``
    per-group probabilities and thresholded booleans.
``forces.npz`` (``--force-name``)
    per-group 3D forces in body-weight units, in the head's own frame plus a
    world-frame copy. A ``root``-frame head (the supervised regime) is rotated
    by the kindyn root quaternion (``human_optim/kindyn_1.npz`` ``q[..., 3:7]``,
    xyzw, world-from-root), so that tree must have run the dynamics stage; a
    ``local_world_aligned`` head is rotated by the axis flip into the OpenCV
    camera and then by ``cam_from_world``.

Group order is the kindyn one everywhere: ``left_hand, right_hand, left_foot
(toe), right_foot, left_ankle (heel), right_ankle``.

    python scripts/predict_reconstruction.py \
        --config output/<run>/config.yaml --checkpoint output/<run>/best.pth \
        --out-root ../BetterVideoReconstruction/out \
        --videos ../BetterVideoReconstruction/data
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _render_common as rc                                     # noqa: E402
from data.climbing_videos.kindyn import quat_xyzw_to_matrix     # noqa: E402
from data.climbing_videos.scene import rows_by_object_id        # noqa: E402
from data.reconstruction import ReconstructionSceneDataset, extract_frames  # noqa: E402
from model.loss import KINDYN_GROUP_NAMES                       # noqa: E402
from train.predict import load_model                            # noqa: E402

#: local_world_aligned (camera y-up) -> OpenCV camera frame (y-down, z-forward).
LWA_TO_CAM = np.array([1.0, -1.0, -1.0], np.float32)


def checkpoint_epoch(path: str | None) -> int:
    """Training epoch stored in a checkpoint; ``-1`` for the untrained model."""
    if path is None:
        return -1
    return int(torch.load(path, map_location="cpu", weights_only=False)["epoch"])


def predict_scene(model, ds, cfg: dict, device: str) -> dict:
    """Run every clip of one out-tree. ``-> {probs, forces, kp2d, kp3d_cam}``.

    Arrays are ``[P, N, ...]`` over the tree's people and frames; rows no clip
    covered stay NaN.
    """
    data = ds.scene_data(ds.scene)
    n_people, n_frames = data["valid_mask"].shape
    n_contact = model.contact_tokens.num_tokens if model.contact_tokens else 0
    n_force = model.force_tokens.num_tokens if model.force_tokens else 0
    out = {
        "probs": np.full((n_people, n_frames, n_contact), np.nan, np.float32),
        "forces": np.full((n_people, n_frames, n_force, 3), np.nan, np.float32),
        "kp2d": np.full((n_people, n_frames, 70, 2), np.nan, np.float32),
        "kp3d_cam": np.full((n_people, n_frames, 70, 3), np.nan, np.float32),
    }
    for clip, batch, output in rc.clip_batches(ds, cfg, model, device):
        mhr = output["mhr"]
        kp2d = rc.to_numpy(mhr["pred_keypoints_2d"])
        kp3d_cam = rc.to_numpy(mhr["pred_keypoints_3d"] + mhr["pred_cam_t"][:, None, :])
        probs = (rc.to_numpy(output["contact"]["joint_probs"])
                 if output["contact"] is not None else None)
        forces = (rc.to_numpy(output["force"]["joint_forces"])
                  if output["force"] is not None else None)
        for row, position in enumerate(batch["frame_index"].tolist()):
            out["kp2d"][clip.person, position] = kp2d[row]
            out["kp3d_cam"][clip.person, position] = kp3d_cam[row]
            if probs is not None:
                out["probs"][clip.person, position] = probs[row]
            if forces is not None:
                out["forces"][clip.person, position] = forces[row]
    return out


def provenance(ds, video: Path, checkpoint: str, epoch: int, cfg: dict,
               clip_frames: int) -> dict:
    """The identity block both npz files carry (BetterVideoReconstruction reads it)."""
    data = ds.scene_data(ds.scene)
    return {
        "limbs": np.asarray(list(KINDYN_GROUP_NAMES)),
        "object_ids": data["object_ids"].astype(np.int32),
        "frame_indices": data["frame_indices"].astype(np.int32),
        "valid_mask": data["valid_mask"],
        "fps": np.float32(data["fps"]),
        "source_video": str(video),
        "checkpoint": checkpoint,
        "checkpoint_epoch": np.int32(epoch),
        "exp_name": str(cfg["output"]["exp_name"]),
        "windows": (f"tiled windows T={clip_frames} stride=1 over invalid-free "
                    f"runs; uncovered rows are NaN"),
    }


def world_forces(forces: np.ndarray, out_dir: Path, ds, force_frame: str,
                 scene: str) -> tuple[np.ndarray, dict]:
    """Rotate head-frame forces to the world frame. ``-> (forces_world, extra keys)``."""
    data = ds.scene_data(scene)
    if force_frame == "local_world_aligned":
        # LWA (camera y-up) -> OpenCV camera, then camera -> world by the
        # cam_from_world rotation (rotation only: body-weight units are scale-free).
        cam = forces * LWA_TO_CAM[None, None, None, :]
        rotation = data["extrinsics"][:, :3, :3]                   # [N, 3, 3]
        return (np.einsum("nji,pnkj->pnki", rotation, cam).astype(np.float32),
                {"lwa_to_opencv_cam_flip": LWA_TO_CAM})
    kindyn = np.load(out_dir / "human_optim" / "kindyn_1.npz", allow_pickle=True)
    q = rows_by_object_id(np.asarray(kindyn["q"], np.float32), kindyn["object_ids"],
                          data["object_ids"], scene, "kindyn")      # [P, N, nq]
    valid = rows_by_object_id(np.asarray(kindyn["valid_mask"], bool),
                              kindyn["object_ids"], data["object_ids"], scene, "kindyn")
    if q.shape[1] != forces.shape[1]:
        raise ValueError(
            f"{scene}: kindyn covers {q.shape[1]} frames but the tree has "
            f"{forces.shape[1]}")
    world = np.einsum("pnij,pnkj->pnki",
                      quat_xyzw_to_matrix(q[..., 3:7]), forces).astype(np.float32)
    world[~valid] = np.nan                       # no root orientation on those rows
    return world, {"root_rotation_source":
                   "human_optim/kindyn_1.npz q[...,3:7] xyzw world-from-root"}


def run_scene(args, model, cfg: dict, scene: str, video: Path, work_root: Path,
              epoch: int, force_frame: str | None) -> None:
    out_dir = args.out_root / scene
    pred_dir = out_dir / "predictions"
    targets = {"contact": pred_dir / "contacts.npz", "force": pred_dir / args.force_name}
    wanted = ([targets["contact"]] if model.contact_tokens is not None else []) + \
             ([targets["force"]] if model.force_tokens is not None else [])
    if args.skip_existing and all(path.is_file() for path in wanted):
        print(f"{scene}: predictions exist, skipping")
        return

    n_frames = len(np.load(out_dir / "geometry" / "transform.npz")["frame_indices"])
    frames_dir = work_root / scene
    print(f"{scene}: extracting {n_frames} frames …")
    extract_frames(video, frames_dir, n_frames)
    clip_frames = min(int(cfg["data"]["clip"]["frames"]), n_frames)
    ds = ReconstructionSceneDataset(out_dir, frames_dir, scene=scene,
                                    clip_frames=clip_frames, stride=1)
    preds = predict_scene(model, ds, cfg, args.device)
    pred_dir.mkdir(parents=True, exist_ok=True)
    identity = provenance(ds, video, str(args.checkpoint), epoch, cfg, clip_frames)

    if model.contact_tokens is not None:
        probs = preds["probs"]
        anchors = preds["kp2d"][:, :, list(model.contact_tokens.keypoint_indices)]
        contacts = np.where(np.isfinite(probs), probs >= args.threshold, False)
        np.savez_compressed(
            targets["contact"], probs=probs, contacts=contacts.astype(bool),
            threshold=np.float32(args.threshold), anchor_points_2d=anchors, **identity)
        print(f"  contacts.npz: {int(np.isfinite(probs).all(-1).sum())} predicted "
              f"person-frames, contact fraction "
              f"{float(contacts[np.isfinite(probs)].mean()):.3f}")

    if model.force_tokens is not None:
        forces = preds["forces"]
        anchors = preds["kp2d"][:, :, list(model.force_tokens.keypoint_indices)]
        anchor_cam = preds["kp3d_cam"][:, :, list(model.force_tokens.keypoint_indices)]
        forces_world, extra = world_forces(forces, out_dir, ds, force_frame, scene)
        if model.contact_tokens is not None:
            extra["contact_probs"] = preds["probs"]
        np.savez_compressed(
            targets["force"], forces=forces, forces_world=forces_world,
            anchor_points_2d=anchors, anchor_cam=anchor_cam, units="body_weight",
            force_frame=force_frame, **extra, **identity)
        magnitude = np.linalg.norm(forces, axis=-1)
        magnitude = magnitude[np.isfinite(magnitude)]
        print(f"  {targets['force'].name}: mean |f| {float(magnitude.mean()):.3f} bw, "
              f"frac >0.05 bw {float((magnitude > 0.05).mean()):.3f}")

    shutil.rmtree(frames_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", default="none",
                        help="checkpoint path, or 'none' for the untrained model")
    parser.add_argument("--out-root", type=Path, required=True,
                        help="root of pipeline out-trees (one subdir per scene)")
    parser.add_argument("--videos", type=Path, required=True,
                        help="root the video pattern is resolved against")
    parser.add_argument("--video-pattern", default="{scene}.mp4",
                        help="source video relative to --videos ({scene} placeholder)")
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="scene subset (default: every out-tree with pipeline inputs)")
    parser.add_argument("--force-name", default="forces.npz",
                        help="filename of the force predictions inside predictions/")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="frame-extraction scratch dir (default: a temp dir)")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    scenes = args.scenes or sorted(
        d.name for d in args.out_root.iterdir()
        if d.is_dir() and not d.name.startswith("_")
        and (d / "sam3" / "bboxes.npz").is_file()
        and (d / "geometry" / "transform.npz").is_file())
    if not scenes:
        raise SystemExit(f"no scenes with pipeline inputs under {args.out_root}")
    videos = {s: args.videos / args.video_pattern.format(scene=s) for s in scenes}
    missing = [s for s in scenes if not videos[s].is_file()]
    if missing:
        raise SystemExit(f"missing source videos for scenes: {missing}")

    checkpoint = None if str(args.checkpoint).lower() == "none" else args.checkpoint
    model, cfg = load_model(args.config, checkpoint, args.device)
    if model.contact_tokens is None and model.force_tokens is None:
        raise SystemExit("this build has neither a contact nor a force head — "
                         "there is nothing to predict")
    force_frame = cfg["model"]["force"]["frame"] if model.force_tokens else None
    if force_frame == "local":
        raise SystemExit("force.frame 'local' cannot be written in world coordinates "
                         "without forward kinematics this script does not run")

    epoch = checkpoint_epoch(checkpoint)
    work_root = args.work_dir or Path(tempfile.mkdtemp(prefix="predict_reconstruction_"))
    work_root.mkdir(parents=True, exist_ok=True)
    failures = []
    for index, scene in enumerate(scenes, start=1):
        print(f"[{index}/{len(scenes)}] {scene}")
        try:
            run_scene(args, model, cfg, scene, videos[scene], work_root, epoch,
                      force_frame)
        except Exception as error:          # a broken scene must not kill the batch
            print(f"  FAILED — {error}")
            failures.append(scene)
    if args.work_dir is None:
        shutil.rmtree(work_root, ignore_errors=True)
    if failures:
        print(f"Done with {len(failures)} failed scene(s): {failures}")
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
