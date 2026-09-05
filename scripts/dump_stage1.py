"""Dump the per-frame (stage-1) body over whole scenes, with the kindyn GT alongside.

    python scripts/dump_stage1.py --config configs/stage1.yaml \
        --checkpoint output/<run>/best.pth --split train --scenes 150
    python scripts/dump_stage1.py --config configs/stage1.yaml \
        --checkpoint output/<run>/best.pth --split test

Feeds ``scripts/analyze_stage1.py`` (train/test gap of the per-frame model,
depth-smoothing calibration, GT motion scales). Every tracked person-frame at
the ``auto`` stride (~25 fps) is predicted, in windows of ``--max-frames`` rows
(the model is per-frame, so windows do not interact).

Writes ``<checkpoint dir>/dump_<split>/<scene>.npz`` — per person ``P`` and
source frame ``N`` (NaN / False where not predicted): ``pelvis_cam (P, N, 3)``,
``root_rot_cam (P, N, 3, 3)``, ``joints_cam (P, N, 22, 3)``, ``betas (P, N,
10)``, ``covered (P, N)``; the GT ``gt_joints_world (P, N, 22, 3)``,
``gt_root_rot (P, N, 3, 3)``, ``gt_valid (P, N)``; ``cam_from_world (N, 4,
4)``, ``tracked (P, N)``, ``fps``, ``stride``, ``object_ids``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _render_common as rc                                     # noqa: E402
from data.base import Clip, valid_runs                          # noqa: E402
from data.climbing_videos import ClimbingVideosDataset          # noqa: E402
from train.predict import load_model                            # noqa: E402

NUM_BODY_JOINTS = 22


def windows(valid: np.ndarray, stride: int, max_rows: int) -> list[tuple[int, int]]:
    """Non-overlapping ``(start_frame, n_rows)`` tiles over every valid run."""
    out = []
    for run_start, run_len in valid_runs(valid):
        n_rows = (run_len - 1) // stride + 1
        for first in range(0, n_rows, max_rows):
            out.append((run_start + first * stride, min(max_rows, n_rows - first)))
    return out


def dump_scene(model, cfg: dict, root: Path, contact_level: int, scene: str, split: str,
               device: str, max_rows: int) -> dict:
    ds = ClimbingVideosDataset(
        root, scenes=[scene], split=split, clip_frames=1, stride="auto", jitter=False,
        seed=int(cfg["data"]["seed"]), contact_level=contact_level, load={"smplx"},
        embedding_dir=(root / "features" / "embedding"
                       if bool(cfg["data"]["embedding_cache"]) else None))
    data = ds.scene_data(scene)
    n_people, n_frames = data["valid_mask"].shape
    stride = ds.scene_stride(scene)
    nan = np.nan
    out = {
        "pelvis_cam": np.full((n_people, n_frames, 3), nan, np.float32),
        "root_rot_cam": np.full((n_people, n_frames, 3, 3), nan, np.float32),
        "joints_cam": np.full((n_people, n_frames, NUM_BODY_JOINTS, 3), nan, np.float32),
        "betas": np.full((n_people, n_frames, 10), nan, np.float32),
        "covered": np.zeros((n_people, n_frames), bool),
        "gt_joints_world": np.full((n_people, n_frames, NUM_BODY_JOINTS, 3), nan, np.float32),
        "gt_root_rot": np.full((n_people, n_frames, 3, 3), nan, np.float32),
        "gt_valid": np.zeros((n_people, n_frames), bool),
        "cam_from_world": np.asarray(data["extrinsics"], np.float32),
        "tracked": np.asarray(data["valid_mask"], bool),
        "fps": np.float32(data["fps"]),
        "stride": np.int32(stride),
        "object_ids": np.asarray(data["object_ids"]).astype(np.int32),
    }
    ds.clips = [Clip(scene, person, start, n_rows, 1)
                for person in range(n_people)
                for start, n_rows in windows(data["valid_mask"][person], stride, max_rows)]
    for clip, batch, output in rc.clip_batches(ds, cfg, model, device):
        sx = output["smplx"]
        rows = batch["frame_index"].tolist()
        p = clip.person
        out["pelvis_cam"][p, rows] = rc.to_numpy(sx["pelvis_cam"])
        out["root_rot_cam"][p, rows] = rc.to_numpy(sx["root_rot"])
        out["joints_cam"][p, rows] = rc.to_numpy(sx["joints_cam"])[:, :NUM_BODY_JOINTS]
        out["betas"][p, rows] = rc.to_numpy(sx["betas"])
        out["covered"][p, rows] = True
        out["gt_joints_world"][p, rows] = batch["smplx_joints_world"].cpu().numpy()[:, :NUM_BODY_JOINTS]
        out["gt_root_rot"][p, rows] = batch["smplx_root_rot"].cpu().numpy()
        out["gt_valid"][p, rows] = batch["smplx_valid"].cpu().numpy()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--scenes", default=None,
                        help="scene count (e.g. '150') or a comma-separated id list; default all")
    parser.add_argument("--max-frames", type=int, default=120, help="rows per forward window")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: <checkpoint dir>/dump_<split>)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, cfg = load_model(args.config, args.checkpoint, args.device)
    if model.head_smplx is None:
        raise SystemExit("the config has no model.smplx head — nothing to dump")
    root, contact_level = rc.dataset_spec(cfg)
    scenes = rc.resolve_scenes(root, args.split, args.scenes, rc.dataset_camera(cfg))
    out_dir = args.out or args.checkpoint.resolve().parent / f"dump_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(scenes)} {args.split} scene(s) -> {out_dir}")
    for index, scene in enumerate(scenes, start=1):
        pred = dump_scene(model, cfg, root, contact_level, scene, args.split, args.device,
                          int(args.max_frames))
        np.savez_compressed(out_dir / f"{scene}.npz", **pred)
        print(f"[{index}/{len(scenes)}] {scene}: {int(pred['covered'].sum())}/"
              f"{int(pred['tracked'].sum())} tracked person-frames", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
