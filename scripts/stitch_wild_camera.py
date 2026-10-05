"""Replace the tail of a wild tree's camera trajectory with a separate VGGT solve of that tail.

VGGT-Omega solves the whole clip in one forward; a stretch of blurred, featureless frames (a
dyno, a whip pan) can leave every camera AFTER it jittering by degrees per frame although the
same frames solve cleanly on their own. This takes that clean solve of frames ``start..N-1``
(``vggt/camera.npz`` of a clip cut at ``start``) and registers it onto the tree's metric world
with ONE similarity transform: the scale from ALL the overlapping frames (the ratio of the two
trajectories' spreads — a short window barely moves and cannot fix a scale), the rotation as the
chordal mean of the per-frame camera orientations and the translation as the mean centre offset
over the ``--fit-frames`` frames after the cut (the full solve drifts as well as jitters, so a
global fit misplaces the join; 30 frames average the jitter and stay local). Rows ``start..`` of ``geometry/transform.npz`` get the registered extrinsics
and the tail solve's intrinsics; the previous file stays beside it as
``transform_before_stitch.npz``. The fused scene / human points are NOT re-fused (the
contact / force pipeline reads only the cameras).

    .venv/bin/python scripts/stitch_wild_camera.py --tree ../data/willd_videos/bvr_out/climbing_dyno \\
        --tail <tail-solve>/vggt/camera.npz --start 270
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def rotation_angle_deg(rotations: np.ndarray) -> np.ndarray:
    """Angle of each rotation matrix in ``(..., 3, 3)``, degrees."""
    trace = np.trace(rotations, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))


def camera_centres(extr: np.ndarray) -> np.ndarray:
    """World-frame camera centres of cam_from_world matrices ``(N, 4, 4)``."""
    return np.einsum("nji,nj->ni", extr[:, :3, :3], -extr[:, :3, 3])


def fit_similarity(extr_ref: np.ndarray, extr_tail: np.ndarray,
                   fit_frames: int) -> tuple[np.ndarray, float, np.ndarray]:
    """``world_ref = scale * R @ world_tail + t`` from two cam_from_world stacks of the same frames.

    Scale over every frame, rotation and translation over the first ``fit_frames``.
    """
    centre_ref, centre_tail = camera_centres(extr_ref), camera_centres(extr_tail)
    dev_ref, dev_tail = centre_ref - centre_ref.mean(0), centre_tail - centre_tail.mean(0)
    scale = float(np.sqrt((dev_ref ** 2).sum() / (dev_tail ** 2).sum()))
    rot_ref = np.transpose(extr_ref[:fit_frames, :3, :3], (0, 2, 1))       # world_from_cam
    rot_tail = np.transpose(extr_tail[:fit_frames, :3, :3], (0, 2, 1))
    u, _, vt = np.linalg.svd(np.einsum("nij,nkj->ik", rot_ref, rot_tail))   # sum R_ref R_tail^T
    rotation = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
    translation = centre_ref[:fit_frames].mean(0) - scale * (centre_tail[:fit_frames] @ rotation.T).mean(0)
    return rotation, scale, translation


def register(extr_tail: np.ndarray, rotation: np.ndarray, scale: float,
             translation: np.ndarray) -> np.ndarray:
    """cam_from_world of the tail solve expressed in the reference world."""
    out = np.tile(np.eye(4, dtype=np.float32), (len(extr_tail), 1, 1))
    rot_cw = extr_tail[:, :3, :3] @ rotation.T                    # cam_from_ref rotation
    out[:, :3, :3] = rot_cw
    out[:, :3, 3] = scale * extr_tail[:, :3, 3] - rot_cw @ translation
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=Path, required=True, help="the clip's pipeline tree")
    parser.add_argument("--tail", type=Path, required=True,
                        help="vggt/camera.npz of the clip cut at --start")
    parser.add_argument("--start", type=int, required=True, help="first frame of the tail solve")
    parser.add_argument("--fit-frames", type=int, default=30,
                        help="frames after the cut the rotation + translation are fitted on")
    args = parser.parse_args()

    path = args.tree / "geometry" / "transform.npz"
    ref = dict(np.load(path, allow_pickle=True))
    tail = np.load(args.tail, allow_pickle=True)
    n_frames = len(ref["extrinsics"])
    if args.start + int(tail["num_frames"]) != n_frames or int(tail["num_frames_total"]) != n_frames - args.start:
        raise ValueError(f"tail solve covers {int(tail['num_frames'])} frames, the tree has "
                         f"{n_frames - args.start} from frame {args.start}")
    width, height = int(ref["image_width"]), int(ref["image_height"])
    extr_tail = np.asarray(tail["extrinsics"], np.float64)
    extr_ref = np.asarray(ref["extrinsics"][args.start:], np.float64)

    rotation, scale, translation = fit_similarity(extr_ref, extr_tail, args.fit_frames)
    stitched = register(extr_tail, rotation, scale, translation)
    residual = rotation_angle_deg(np.einsum("nij,nik->njk", stitched[:, :3, :3], extr_ref[:, :3, :3]))
    centre_err = np.linalg.norm(camera_centres(stitched) - camera_centres(extr_ref), axis=1)
    before = rotation_angle_deg(np.einsum("nij,nik->njk", extr_ref[:-1, :3, :3], extr_ref[1:, :3, :3]))
    after = rotation_angle_deg(np.einsum("nij,nik->njk", stitched[:-1, :3, :3], stitched[1:, :3, :3]))
    join = rotation_angle_deg(ref["extrinsics"][args.start - 1, :3, :3] @ stitched[0, :3, :3].T)
    join_cm = 100 * np.linalg.norm(camera_centres(np.asarray(ref["extrinsics"][args.start - 1:args.start], np.float64))
                                   - camera_centres(stitched[:1]))
    print(f"registration: scale {scale:.4f} over {len(extr_ref)} frames, rotation "
          f"{rotation_angle_deg(rotation):.2f} deg over {args.fit_frames}; residual vs the full solve: rotation median "
          f"{np.median(residual):.2f} / p90 {np.percentile(residual, 90):.2f} deg, centre median "
          f"{np.median(centre_err):.3f} m")
    print(f"frame-to-frame rotation, median (p90): before {np.median(before):.2f} ({np.percentile(before, 90):.2f}) "
          f"-> after {np.median(after):.2f} ({np.percentile(after, 90):.2f}) deg; step across the "
          f"join {join:.2f} deg / {join_cm:.1f} cm")

    backup = path.with_name("transform_before_stitch.npz")
    if backup.exists():
        raise FileExistsError(f"{backup} exists: the tree was stitched before")
    path.rename(backup)
    out = dict(ref)
    out["extrinsics"] = np.concatenate([ref["extrinsics"][:args.start], stitched.astype(np.float32)])
    intr_norm = np.asarray(tail["intrinsics"], np.float32)
    intr_px = intr_norm.copy()
    intr_px[:, 0, :] *= width
    intr_px[:, 1, :] *= height
    out["intrinsics_px_orig"] = np.concatenate([ref["intrinsics_px_orig"][:args.start], intr_px])
    out["intrinsics"] = np.concatenate([ref["intrinsics"][:args.start], intr_norm])
    for key in ("fov_x_deg", "fov_y_deg"):
        out[key] = np.concatenate([ref[key][:args.start], np.asarray(tail[key], np.float32)])
    out["camera_source"] = np.str_(f"frames {args.start}+ from a separate VGGT solve, registered by "
                                   f"scripts/stitch_wild_camera.py (scale {scale:.4f})")
    np.savez(path, **out)
    print(f"wrote {path} (previous file: {backup})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
