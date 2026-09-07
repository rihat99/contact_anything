"""Predict contacts, 3D forces and the refined SMPL-X body on BetterVideoReconstruction out-trees.

For every scene ``<out-root>/<stem>/`` with pipeline inputs (``sam3/bboxes.npz``,
``geometry/transform.npz``) the source video is decoded to frames and the checkpoint
runs over :class:`data.reconstruction.ReconstructionSceneDataset` under the
``scripts/predict_test.py`` protocol: every contiguous tracked run of every person at
the config's clip stride (``auto`` = the per-scene ~25 fps stride), tiled into
``--max-frames``-row windows overlapping by ``--overlap`` rows, each row keeping the
window it sits deepest inside. Rows no window covers (source frames between stride
steps) stay NaN. Labels are not needed — the tree only has to carry cameras and boxes.

Files written into ``<stem>/predictions/`` (arrays ``[P, N, ...]`` over the tree's
people and frames, NaN / False where not predicted):

``contacts.npz``
    six-group probabilities, thresholded booleans and the anchors' pixels.
``forces.npz`` (``--force-name``)
    ``forces`` in the refiner's body frame and ``forces_world`` — rotated with the
    model's OWN world-from-body root, i.e. into the world of ``geometry/transform.npz``
    (a decoder-level force head is rotated with the per-frame SMPL-X root instead) —
    plus the camera-frame anchors, body-weight units.
``smplx.npz``
    the refined body in the ``predict_test.py`` layout: ``q_cam``, ``betas``,
    ``joints_cam``, ``joints_world``, ``pelvis_cam``, ``covered``.

Group order is the kindyn one everywhere: ``left_hand, right_hand, left_foot (toe),
right_foot, left_ankle (heel), right_ankle``; the anchors are the refined body's
SMPL-X wrist / toe / ankle joints.

    python scripts/predict_reconstruction.py --config configs/stage2_v2_force.yaml \
        --checkpoint output/<run>/last.pth \
        --out-root ../BetterVideoReconstruction-dev/peter/out_climb_wall_2_single \
        --videos ../BetterVideoReconstruction-dev/peter/climb_wall_2 \
        --video-pattern "{scene}/cam_left.mp4" --force-name forces_sup.npz
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
from data.base import Clip                                      # noqa: E402
from data.reconstruction import ReconstructionSceneDataset, extract_frames  # noqa: E402
from model.loss import KINDYN_GROUP_NAMES                       # noqa: E402
from model.loss.contact_consistency import GROUP_JOINTS         # noqa: E402
from predict_test import Q_FULL, pad_hands, windows             # noqa: E402
from train.predict import load_model                            # noqa: E402

NUM_GROUPS = len(KINDYN_GROUP_NAMES)


def checkpoint_epoch(path: str | None) -> int:
    """Training epoch stored in a checkpoint; ``-1`` for the untrained model."""
    if path is None:
        return -1
    return int(torch.load(path, map_location="cpu", weights_only=False)["epoch"])


def project_rows(points_cam: np.ndarray, cam_int: np.ndarray) -> np.ndarray:
    """Per-row pinhole projection: ``(B, K, 3)`` camera points with ``(B, 3, 3)`` intrinsics."""
    z = np.clip(points_cam[..., 2:3], 1e-6, None)
    focal = cam_int[:, None, [0, 1], [0, 1]]
    centre = cam_int[:, None, :2, 2]
    return points_cam[..., :2] / z * focal + centre


def predict_scene(model, ds: ReconstructionSceneDataset, cfg: dict, device: str,
                  max_rows: int, overlap: int) -> dict:
    """Run the tiled windows of one out-tree; scatter the rows into per-frame arrays."""
    scene = ds.scene
    data = ds.scene_data(scene)
    n_people, n_frames = data["valid_mask"].shape
    stride = ds.scene_stride(scene)
    n_joints = model.head_smplx.num_joints
    has_contact, has_force = model.has_contact, model.has_force
    nan = lambda *shape: np.full((n_people, n_frames, *shape), np.nan, np.float32)  # noqa: E731
    out = {
        "q_cam": nan(Q_FULL), "betas": nan(10), "joints_cam": nan(n_joints, 3),
        "joints_world": nan(n_joints, 3), "pelvis_cam": nan(3),
        "anchor_cam": nan(NUM_GROUPS, 3), "anchor_2d": nan(NUM_GROUPS, 2),
        "covered": np.zeros((n_people, n_frames), bool),
        "stride": np.int32(stride),
    }
    if has_contact:
        out["probs"] = nan(NUM_GROUPS)
    if has_force:
        out["forces"], out["forces_world"] = nan(NUM_GROUPS, 3), nan(NUM_GROUPS, 3)
    # Distance of the kept prediction from its window's edge (rows); a later window
    # overwrites a row only from deeper inside itself.
    depth = np.full((n_people, n_frames), -1, np.int64)
    clips = []
    for person in range(n_people):
        for start, n_rows in windows(data["valid_mask"][person], stride, max_rows, overlap):
            clips.append(Clip(scene, person, start, n_rows, 1))
    ds.clips = clips
    for clip, batch, output in rc.clip_batches(ds, cfg, model, device):
        sx = output["smplx"]
        ext = batch["cam_from_world"]
        q = pad_hands(rc.to_numpy(sx["q_cam"]))
        betas, joints = rc.to_numpy(sx["betas"]), rc.to_numpy(sx["joints_cam"])
        pelvis = rc.to_numpy(sx["pelvis_cam"])
        if "joints_world" in sx:
            joints_world = rc.to_numpy(sx["joints_world"])
        else:                                       # per-frame head: lift with the tree's cameras
            rot_wc = ext[:, :3, :3].transpose(1, 2)
            joints_world = rc.to_numpy(torch.einsum(
                "bij,bkj->bki", rot_wc, sx["joints_cam"] - ext[:, None, :3, 3]))
        anchor_cam = joints[:, list(GROUP_JOINTS)]
        anchor_2d = project_rows(anchor_cam, rc.to_numpy(batch["cam_int"]))
        probs = rc.to_numpy(output["contact"]["probs"]) if has_contact else None
        forces = forces_world = None
        if has_force:
            fr = output["force"]
            frame = fr.get("frame")                 # refiner: world-from-body of its forces
            if frame is None:                       # decoder head: the kindyn root-frame convention
                frame = ext[:, :3, :3].transpose(1, 2) @ sx["root_rot"]
            forces = rc.to_numpy(fr["forces"])
            forces_world = np.einsum("bij,bkj->bki", rc.to_numpy(frame), forces)
        rows = batch["frame_index"].tolist()
        p = clip.person
        for row, position in enumerate(rows):
            d = min(row, len(rows) - 1 - row)
            if d <= depth[p, position]:
                continue
            depth[p, position] = d
            out["q_cam"][p, position] = q[row]
            out["betas"][p, position] = betas[row]
            out["joints_cam"][p, position] = joints[row]
            out["joints_world"][p, position] = joints_world[row]
            out["pelvis_cam"][p, position] = pelvis[row]
            out["anchor_cam"][p, position] = anchor_cam[row]
            out["anchor_2d"][p, position] = anchor_2d[row]
            out["covered"][p, position] = True
            if probs is not None:
                out["probs"][p, position] = probs[row]
            if forces is not None:
                out["forces"][p, position] = forces[row]
                out["forces_world"][p, position] = forces_world[row]
    out["windows"] = np.array([(c.person, c.start, c.frames) for c in clips], np.int32)
    return out


def provenance(ds, preds: dict, video: Path, checkpoint: str, epoch: int, cfg: dict,
               max_rows: int, overlap: int) -> dict:
    """The identity block every file carries (BetterVideoReconstruction reads it).

    ``valid_mask`` is the PREDICTED coverage (tracked rows at the clip stride), so a
    consumer never draws an uncovered row.
    """
    data = ds.scene_data(ds.scene)
    return {
        "limbs": np.asarray(list(KINDYN_GROUP_NAMES)),
        "object_ids": data["object_ids"].astype(np.int32),
        "frame_indices": data["frame_indices"].astype(np.int32),
        "valid_mask": preds["covered"],
        "tracked": np.asarray(data["valid_mask"], bool),
        "fps": np.float32(data["fps"]),
        "stride": preds["stride"],
        "source_video": str(video),
        "checkpoint": checkpoint,
        "checkpoint_epoch": np.int32(epoch),
        "exp_name": str(cfg["output"]["exp_name"]),
        "windows": (f"tiled windows of {max_rows} rows overlapping {overlap} at stride "
                    f"{int(preds['stride'])} over tracked runs; each row keeps the window it "
                    f"sits deepest inside; uncovered rows are NaN"),
    }


def run_scene(args, model, cfg: dict, scene: str, video: Path, work_root: Path,
              epoch: int) -> None:
    out_dir = args.out_root / scene
    pred_dir = out_dir / "predictions"
    targets = {"contact": pred_dir / "contacts.npz", "force": pred_dir / args.force_name,
               "smplx": pred_dir / "smplx.npz"}
    wanted = [targets["smplx"]] + ([targets["contact"]] if model.has_contact else []) + \
             ([targets["force"]] if model.has_force else [])
    if args.skip_existing and all(path.is_file() for path in wanted):
        print(f"{scene}: predictions exist, skipping")
        return

    n_frames = len(np.load(out_dir / "geometry" / "transform.npz")["frame_indices"])
    frames_dir = work_root / scene
    print(f"{scene}: extracting {n_frames} frames …")
    extract_frames(video, frames_dir, n_frames)
    max_rows = int(args.max_frames)
    ds = ReconstructionSceneDataset(out_dir, frames_dir, scene=scene,
                                    clip_frames=max_rows, stride=cfg["data"]["clip"]["stride"])
    preds = predict_scene(model, ds, cfg, args.device, max_rows, int(args.overlap))
    pred_dir.mkdir(parents=True, exist_ok=True)
    identity = provenance(ds, preds, video, str(args.checkpoint), epoch, cfg,
                          max_rows, int(args.overlap))
    covered = int(preds["covered"].sum())
    print(f"  {covered}/{int(ds.scene_data(scene)['valid_mask'].sum())} tracked person-frames "
          f"predicted (stride {int(preds['stride'])}, {len(preds['windows'])} windows)")

    np.savez_compressed(
        targets["smplx"], q_cam=preds["q_cam"], betas=preds["betas"],
        joints_cam=preds["joints_cam"], joints_world=preds["joints_world"],
        pelvis_cam=preds["pelvis_cam"], covered=preds["covered"],
        hands=np.bool_(model.head_smplx.hands), **identity)

    if model.has_contact:
        probs = preds["probs"]
        contacts = np.where(np.isfinite(probs), probs >= args.threshold, False)
        np.savez_compressed(
            targets["contact"], probs=probs, contacts=contacts.astype(bool),
            threshold=np.float32(args.threshold), anchor_points_2d=preds["anchor_2d"],
            **identity)
        print(f"  contacts.npz: contact fraction "
              f"{float(contacts[np.isfinite(probs)].mean()):.3f}")

    if model.has_force:
        extra = {"contact_probs": preds["probs"]} if model.has_contact else {}
        np.savez_compressed(
            targets["force"], forces=preds["forces"], forces_world=preds["forces_world"],
            anchor_points_2d=preds["anchor_2d"], anchor_cam=preds["anchor_cam"],
            units="body_weight", force_frame="refiner_body",
            root_rotation_source="the model's own world-from-body root (refiner input frame)",
            **extra, **identity)
        magnitude = np.linalg.norm(preds["forces"], axis=-1)
        finite = np.isfinite(magnitude)
        line = f"  {targets['force'].name}: mean |f| {float(magnitude[finite].mean()):.3f} bw"
        if model.has_contact:
            on = finite & (preds["probs"] >= args.threshold)
            line += (f", limbs with p>={args.threshold:g}: {float(magnitude[on].mean()):.3f} bw, "
                     f"others {float(magnitude[finite & ~on].mean()):.3f} bw")
        print(line)

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
    parser.add_argument("--max-frames", type=int, default=240,
                        help="window length in rows (~18 GiB peak at 240)")
    parser.add_argument("--overlap", type=int, default=120,
                        help="rows shared by consecutive windows")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="frame-extraction scratch dir (default: a temp dir)")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not 0 <= args.overlap < args.max_frames:
        raise SystemExit(f"--overlap {args.overlap} must be in [0, --max-frames {args.max_frames})")

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
    if model.head_smplx is None:
        raise SystemExit("this build has no SMPL-X head — the anchors need the body")
    epoch = checkpoint_epoch(checkpoint)
    print(f"{len(scenes)} scene(s) on {args.device}; checkpoint {checkpoint} (epoch {epoch}); "
          f"contact={model.has_contact} force={model.has_force}; windows {args.max_frames} rows, "
          f"overlap {args.overlap}")
    work_root = args.work_dir or Path(tempfile.mkdtemp(prefix="predict_reconstruction_"))
    work_root.mkdir(parents=True, exist_ok=True)
    failures = []
    for index, scene in enumerate(scenes, start=1):
        print(f"[{index}/{len(scenes)}] {scene}", flush=True)
        try:
            run_scene(args, model, cfg, scene, videos[scene], work_root, epoch)
        except Exception as error:          # a broken scene must not kill the batch
            print(f"  FAILED — {type(error).__name__}: {error}", flush=True)
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
