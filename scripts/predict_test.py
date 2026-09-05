"""Dump a checkpoint's SMPL-X (+ contact) predictions over whole test scenes.

The results viewer (``scripts/view_results.py``) reads these dumps, so every
model — present and future — is compared through the same file: run this once
per run with its best checkpoint.

    python scripts/predict_test.py --config configs/baseline.yaml \
        --checkpoint output/<run>/best.pth

Protocol: the test split of the config's dataset yaml (its ``camera`` filter
included). Every contiguous tracked run of every person is predicted at the
evaluation stride (``data.clip.stride``, ``auto`` = the per-scene ~25 fps
stride), tiled into windows of ``--max-frames`` rows overlapping by
``--overlap`` rows; each row keeps the window it sits deepest inside (farthest
from an edge). The defaults (240 / 120, ~18 GiB peak) put every row >= 60 rows
inside its window; against a single whole-scene pass the tiling differs by
<= 5 mm max / 0.3 mm mean on the joints (docs/viewer.md). Source frames between
stride steps are not predicted. The forward runs with TF32 like evaluate.py, so
``joints_cam`` is the head's own FK of ``q_cam`` to ~0.6 mm, not an fp32 one.

Writes ``<run>/predictions/<scene>.npz`` next to the checkpoint — per person and
source frame (NaN / False where not predicted): the BetterHuman camera-frame
configuration ``q_cam (P, N, 211)`` (finger quaternions identity for a
hands-free head), ``betas (P, N, 10)``, ``joints_cam (P, N, J, 3)``,
``pelvis_cam (P, N, 3)``, ``covered (P, N)``, ``tracked (P, N)`` (the dataset's
person-frame validity), ``contact_probs (P, N, 6)`` when the build predicts contact and
``forces_world (P, N, 6, 3)`` (body-weight units, WORLD frame: the refiner's body-frame
forces rotated with its own root) when it predicts forces — plus
``predictions/manifest.json`` with the checkpoint, epoch, config and protocol.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _render_common as rc                                     # noqa: E402
from data.base import Clip, valid_runs                          # noqa: E402
from train.predict import load_model                            # noqa: E402

#: Width of the full BetterHuman SMPL-X configuration (pelvis, root, 21 body, 30 fingers).
Q_FULL = 211
NUM_HAND_QUATS = 30


def windows(valid: np.ndarray, stride: int, max_rows: int, overlap: int) -> list[tuple[int, int]]:
    """``(start_frame, n_rows)`` windows tiling every valid run of one person.

    A run shorter than ``max_rows`` rows is one window; a longer one is tiled
    with step ``max_rows - overlap`` and a terminal window aligned to its end.
    """
    if overlap >= max_rows:
        raise ValueError(f"overlap {overlap} must be smaller than max_rows {max_rows}")
    out = []
    for run_start, run_len in valid_runs(valid):
        n_rows = (run_len - 1) // stride + 1
        if n_rows <= max_rows:
            out.append((run_start, n_rows))
            continue
        step = max_rows - overlap
        starts = list(range(0, n_rows - max_rows + 1, step))
        if starts[-1] != n_rows - max_rows:
            starts.append(n_rows - max_rows)
        out.extend((run_start + s * stride, max_rows) for s in starts)
    return out


def pad_hands(q: np.ndarray) -> np.ndarray:
    """``(B, 91) -> (B, 211)`` with identity finger quaternions (``xyzw``)."""
    if q.shape[1] == Q_FULL:
        return q
    fingers = np.tile(np.array([0.0, 0.0, 0.0, 1.0], np.float32), (q.shape[0], NUM_HAND_QUATS))
    return np.concatenate([q, fingers], axis=1)


def predict_scene(model, ds, cfg: dict, device: str, max_rows: int, overlap: int) -> dict:
    """Run the tiled windows of one scene; scatter the rows into per-frame arrays."""
    scene = ds.scenes[0]
    data = ds.scene_data(scene)
    n_people, n_frames = data["valid_mask"].shape
    stride = ds.scene_stride(scene)
    n_joints = model.head_smplx.num_joints
    has_contact = model.has_contact
    out = {
        "q_cam": np.full((n_people, n_frames, Q_FULL), np.nan, np.float32),
        "betas": np.full((n_people, n_frames, 10), np.nan, np.float32),
        "joints_cam": np.full((n_people, n_frames, n_joints, 3), np.nan, np.float32),
        "pelvis_cam": np.full((n_people, n_frames, 3), np.nan, np.float32),
        "covered": np.zeros((n_people, n_frames), bool),
        "stride": np.int32(stride),
    }
    if has_contact:
        out["contact_probs"] = np.full((n_people, n_frames, 6), np.nan, np.float32)
    has_force = model.has_force
    if has_force:
        out["forces_world"] = np.full((n_people, n_frames, 6, 3), np.nan, np.float32)
    # Distance of the kept prediction from its window's edge (rows); a later
    # window overwrites a row only from deeper inside itself.
    depth = np.full((n_people, n_frames), -1, np.int64)
    clips = []
    for person in range(n_people):
        for start, n_rows in windows(data["valid_mask"][person], stride, max_rows, overlap):
            clips.append(Clip(scene, person, start, n_rows, 1))
    ds.clips = clips
    for clip, batch, output in rc.clip_batches(ds, cfg, model, device):
        sx = output["smplx"]
        q = pad_hands(rc.to_numpy(sx["q_cam"]))
        betas, joints = rc.to_numpy(sx["betas"]), rc.to_numpy(sx["joints_cam"])
        pelvis = rc.to_numpy(sx["pelvis_cam"])
        probs = rc.to_numpy(output["contact"]["probs"]) if has_contact else None
        forces = None
        if has_force:
            fr = output["force"]
            frame = fr.get("frame")                     # refiner: world-from-body of its forces
            if frame is None:                           # decoder head: the kindyn root-frame convention
                ext = batch["cam_from_world"]
                frame = ext[:, :3, :3].transpose(1, 2) @ sx["root_rot"]
            forces = np.einsum("bij,bkj->bki", rc.to_numpy(frame), rc.to_numpy(fr["forces"]))
        rows = batch["frame_index"].tolist()
        for row, position in enumerate(rows):
            d = min(row, len(rows) - 1 - row)
            if d <= depth[clip.person, position]:
                continue
            depth[clip.person, position] = d
            p = clip.person
            out["q_cam"][p, position] = q[row]
            out["betas"][p, position] = betas[row]
            out["joints_cam"][p, position] = joints[row]
            out["pelvis_cam"][p, position] = pelvis[row]
            out["covered"][p, position] = True
            if probs is not None:
                out["contact_probs"][p, position] = probs[row]
            if forces is not None:
                out["forces_world"][p, position] = forces[row]
    out["windows"] = np.array([(c.person, c.start, c.frames) for c in clips], np.int32)
    return out


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=rc.REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: <checkpoint dir>/predictions)")
    parser.add_argument("--scenes", default=None,
                        help="scene count (e.g. '5') or a comma-separated id list")
    parser.add_argument("--max-frames", type=int, default=240,
                        help="window length in rows (~18 GiB peak at 240)")
    parser.add_argument("--overlap", type=int, default=120,
                        help="rows shared by consecutive windows")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, cfg = load_model(args.config, args.checkpoint, args.device)
    if model.head_smplx is None:
        raise SystemExit("the config has no model.smplx head — nothing to dump")
    root, contact_level = rc.dataset_spec(cfg)
    camera = rc.dataset_camera(cfg)
    max_rows = int(args.max_frames)
    if not 0 <= args.overlap < max_rows:
        raise SystemExit(f"--overlap {args.overlap} must be in [0, --max-frames {max_rows})")
    scenes = rc.resolve_scenes(root, "test", args.scenes, camera)
    out_dir = args.out or args.checkpoint.resolve().parent / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    epoch = int(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["epoch"])
    print(f"{len(scenes)} test scene(s) [camera={camera}] on {args.device}; "
          f"checkpoint {args.checkpoint} (epoch {epoch}); windows {max_rows} rows, "
          f"overlap {args.overlap}")

    manifest = {
        "checkpoint": str(args.checkpoint.resolve()), "epoch": epoch,
        "exp_name": str(cfg["output"]["exp_name"]), "config": str(args.config),
        "git": git_head(), "hands": bool(model.head_smplx.hands),
        "camera_head": str(model.head_smplx.camera),
        "contact": model.has_contact, "force": model.has_force,
        "protocol": {"max_rows": max_rows, "overlap": args.overlap,
                     "stride": str(cfg["data"]["clip"]["stride"]), "camera": camera},
        "scenes": {},
    }
    for index, scene in enumerate(scenes, start=1):
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        ds = rc.build_dataset(cfg, root, contact_level, scene, "test", set(), max_rows)
        pred = predict_scene(model, ds, cfg, args.device, max_rows, args.overlap)
        data = ds.scene_data(scene)
        np.savez_compressed(
            out_dir / f"{scene}.npz", **pred,
            object_ids=data["object_ids"].astype(np.int32),
            tracked=np.asarray(data["valid_mask"], bool),
            fps=np.float32(data["fps"]),
            hands=np.bool_(model.head_smplx.hands),
            checkpoint=str(args.checkpoint.resolve()), epoch=np.int32(epoch),
            exp_name=str(cfg["output"]["exp_name"]))
        covered = int(pred["covered"].sum())
        tracked = int(data["valid_mask"].sum())
        manifest["scenes"][scene] = {
            "people": int(data["valid_mask"].shape[0]), "frames": int(data["valid_mask"].shape[1]),
            "stride": int(pred["stride"]), "covered": covered, "tracked": tracked,
            "windows": int(len(pred["windows"]))}
        mem = (f", peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB"
               if args.device.startswith("cuda") else "")
        print(f"[{index}/{len(scenes)}] {scene}: {covered}/{tracked} tracked person-frames "
              f"predicted (stride {int(pred['stride'])}, {len(pred['windows'])} windows){mem}",
              flush=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(scenes)} scene(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
