"""Stage-1 diagnostics from ``scripts/dump_stage1.py`` dumps.

    python scripts/analyze_stage1.py --train output/<run>/dump_train --test output/<run>/dump_test

Three questions the stage-2 design depends on (docs/refiner.md):

1. **Train / test gap of the per-frame model** — the refiner trains on stage-1
   predictions of scenes stage 1 was trained on. Camera-frame MPJPE (hips-mean
   aligned, 22 joints), absolute pelvis / depth error and the GVHMR lifted
   jitter, per split.
2. **Depth-smoothing calibration** — Gaussian smoothing of the pelvis log depth
   in camera coordinates (bearing kept), swept over ``--sigmas`` seconds: world
   pelvis error, absolute world joint error and lifted jitter after smoothing.
3. **GT motion scales** — RMS of the kindyn world joint velocity / acceleration
   and root angular velocity / acceleration after the ``--label-smooth`` label
   smoothing (the ``motion_supervision.scale`` numbers).

Series are handled per contiguous covered run of one person at the dump stride;
derivatives use the refiner's own helpers so the numbers match the loss.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.base import valid_runs                                # noqa: E402
from model.refiner import angular_velocity, gaussian_smooth, time_derivative  # noqa: E402
from utils.gvhmr_metrics import compute_jitter                  # noqa: E402

HIPS = (1, 2)


def runs_of(dump: dict, person: int):
    """Yield per contiguous covered+valid run: (frame indices, seconds) arrays."""
    ok = dump["covered"][person] & dump["gt_valid"][person] & dump["tracked"][person]
    stride = int(dump["stride"])
    positions = np.flatnonzero(ok)
    if positions.size == 0:
        return
    # Split where the gap is not exactly one stride.
    breaks = np.flatnonzero(np.diff(positions) != stride) + 1
    for chunk in np.split(positions, breaks):
        if len(chunk) >= 3:
            yield chunk, chunk.astype(np.float64) / float(dump["fps"])


def jitter(points: np.ndarray, fps: float) -> np.ndarray:
    """GVHMR jitter (10 m/s^3) of ``(T, K, 3)`` world joints, per interior frame."""
    return np.asarray(compute_jitter(torch.as_tensor(points, dtype=torch.float32), fps=fps))


def lift(points_cam: np.ndarray, ext: np.ndarray) -> np.ndarray:
    """``(T, K, 3)`` camera points -> world with ``(T, 4, 4)`` cam_from_world."""
    rot, t = ext[:, :3, :3], ext[:, :3, 3]
    return np.einsum("tji,tkj->tki", rot, points_cam - t[:, None])


def smooth_depth(pelvis_cam: np.ndarray, seconds: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing of log z along time, bearing kept. ``(T, 3) -> (T, 3)``."""
    if sigma <= 0:
        return pelvis_cam
    t = torch.as_tensor(seconds, dtype=torch.float32)[None]
    valid = torch.ones_like(t, dtype=torch.bool)
    log_z = torch.log(torch.as_tensor(pelvis_cam[:, 2:3], dtype=torch.float32).clamp(min=1e-3))[None]
    z = torch.exp(gaussian_smooth(log_z, t, valid, sigma))[0, :, 0].numpy()
    return np.stack([pelvis_cam[:, 0] / pelvis_cam[:, 2] * z,
                     pelvis_cam[:, 1] / pelvis_cam[:, 2] * z, z], axis=-1)


def analyze_split(files: list[Path], sigmas: list[float], label_smooth: float) -> dict:
    acc = {"mpjpe": [], "pelvis_err": [], "depth_err": [], "depth_bias": [],
           "jitter": [], "gt_jitter": [], "frames": 0, "runs": 0}
    sweep = {s: {"pelvis_world": [], "joint_world": [], "jitter": []} for s in sigmas}
    motion_sq = {"vel": [0.0, 0], "acc": [0.0, 0], "ang_vel": [0.0, 0], "ang_acc": [0.0, 0]}
    for path in files:
        dump = dict(np.load(path))
        fps_eff = float(dump["fps"]) / int(dump["stride"])
        for person in range(dump["covered"].shape[0]):
            for rows, seconds in runs_of(dump, person):
                ext = dump["cam_from_world"][rows]
                joints = dump["joints_cam"][person, rows]                  # (T, 22, 3)
                pelvis = dump["pelvis_cam"][person, rows]                  # (T, 3)
                gt_world = dump["gt_joints_world"][person, rows]           # (T, 22, 3)
                gt_cam = np.einsum("tij,tkj->tki", ext[:, :3, :3], gt_world) + ext[:, :3, 3][:, None]
                # camera-frame metrics
                pj = joints - joints[:, list(HIPS)].mean(1, keepdims=True)
                gj = gt_cam - gt_cam[:, list(HIPS)].mean(1, keepdims=True)
                acc["mpjpe"].append(np.linalg.norm(pj - gj, axis=-1).mean(-1))
                abs_err = (joints[:, 0] - gt_cam[:, 0])
                acc["pelvis_err"].append(np.linalg.norm(abs_err, axis=-1))
                acc["depth_err"].append(np.abs(abs_err[:, 2]))
                acc["depth_bias"].append(abs_err[:, 2])
                lifted = lift(joints, ext)
                acc["jitter"].append(jitter(lifted, fps_eff))
                acc["gt_jitter"].append(jitter(gt_world, fps_eff))
                acc["frames"] += len(rows)
                acc["runs"] += 1
                # depth-smoothing sweep
                for s in sigmas:
                    pelvis_s = smooth_depth(pelvis, seconds, s)
                    shifted = joints + (pelvis_s - pelvis)[:, None]
                    world = lift(shifted, ext)
                    sweep[s]["pelvis_world"].append(np.linalg.norm(world[:, 0] - gt_world[:, 0], axis=-1))
                    sweep[s]["joint_world"].append(np.linalg.norm(world - gt_world, axis=-1).mean(-1))
                    sweep[s]["jitter"].append(jitter(world, fps_eff))
                # GT motion scales
                t = torch.as_tensor(seconds, dtype=torch.float32)[None]
                valid = torch.ones_like(t, dtype=torch.bool)
                p = torch.as_tensor(gt_world)[None]
                v = gaussian_smooth(time_derivative(p, t, valid), t, valid, label_smooth)
                a = gaussian_smooth(time_derivative(v, t, valid), t, valid, label_smooth)
                root = torch.as_tensor(dump["gt_root_rot"][person, rows])[None]
                w_body = angular_velocity(root, t, valid)
                w = gaussian_smooth((root @ w_body[..., None])[..., 0], t, valid, label_smooth)
                al = gaussian_smooth(time_derivative(w, t, valid), t, valid, label_smooth)
                interior = slice(2, -2)
                for key, series in (("vel", v), ("acc", a), ("ang_vel", w), ("ang_acc", al)):
                    x = series[0, interior]
                    motion_sq[key][0] += float((x ** 2).sum())
                    motion_sq[key][1] += x.numel()
    summary = {k: float(np.concatenate(v).mean()) for k, v in acc.items() if isinstance(v, list) and v}
    summary["frames"], summary["runs"] = acc["frames"], acc["runs"]
    summary["sweep"] = {s: {k: float(np.concatenate(v).mean()) for k, v in d.items()}
                        for s, d in sweep.items()}
    summary["motion_rms"] = {k: (sq / n) ** 0.5 if n else float("nan") for k, (sq, n) in motion_sq.items()}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, default=None, help="dump_train directory")
    parser.add_argument("--test", type=Path, default=None, help="dump_test directory")
    parser.add_argument("--sigmas", default="0,0.1,0.2,0.3,0.5,0.75,1.0",
                        help="depth-smoothing sigmas (seconds) to sweep")
    parser.add_argument("--label-smooth", type=float, default=0.12,
                        help="motion_supervision.label_smooth_sec used for the GT scales")
    args = parser.parse_args()
    sigmas = [float(s) for s in args.sigmas.split(",")]
    splits = [(name, path) for name, path in (("train", args.train), ("test", args.test)) if path]
    if not splits:
        raise SystemExit("pass --train and/or --test")

    results = {name: analyze_split(sorted(path.glob("*.npz")), sigmas, args.label_smooth)
               for name, path in splits}
    print("\n## per-frame model, camera frame (mm; jitter 10 m/s^3)\n")
    print("| split | runs | frames | mpjpe | pelvis_err | depth_err | depth_bias | lifted jitter | gt jitter |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, r in results.items():
        print(f"| {name} | {r['runs']} | {r['frames']} | {1000 * r['mpjpe']:.1f} | "
              f"{1000 * r['pelvis_err']:.1f} | {1000 * r['depth_err']:.1f} | "
              f"{1000 * r['depth_bias']:+.1f} | {r['jitter']:.1f} | {r['gt_jitter']:.1f} |")
    print("\n## depth smoothing sweep (world, mm; jitter 10 m/s^3)\n")
    header = "| sigma (s) | " + " | ".join(
        f"{name} pelvis | {name} joints | {name} jitter" for name in results) + " |"
    print(header)
    print("|---|" + "---|" * (3 * len(results)))
    for s in sigmas:
        cells = []
        for r in results.values():
            d = r["sweep"][s]
            cells += [f"{1000 * d['pelvis_world']:.1f}", f"{1000 * d['joint_world']:.1f}",
                      f"{d['jitter']:.1f}"]
        print(f"| {s:g} | " + " | ".join(cells) + " |")
    print(f"\n## GT motion RMS after {args.label_smooth:g} s label smoothing (motion_supervision.scale)\n")
    for name, r in results.items():
        rms = r["motion_rms"]
        print(f"{name}: vel {rms['vel']:.3f} m/s, acc {rms['acc']:.3f} m/s^2, "
              f"ang_vel {rms['ang_vel']:.3f} rad/s, ang_acc {rms['ang_acc']:.3f} rad/s^2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
