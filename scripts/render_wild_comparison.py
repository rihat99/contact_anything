"""Learned vs optimisation force arrows on the in-the-wild clips, as cropped frames.

For every clip with a learned dump under ``--learned-root/<stem>/predictions/<learned-dir>/``
and an optimisation dump under ``--optim-root/<stem>/predictions/<optim-dir>/`` two folders of
``--side`` x ``--side`` JPEG squares (quality 92) are written, both cut with the SAME window
(:func:`render_wild_overlays.crop_windows` over the learned skeleton and both runs' arrow tips):

* ``<out>/original/<stem>/<frame>.jpg`` — the frame as is, and
* ``<out>/comparison/<stem>/<frame>.jpg`` — the frame fogged towards white with the learned
  forces as yellow arrows over the optimisation's red ones, both drawn from the learned
  skeleton's joints (no skeleton).

Each folder also gets ``<stem>.mp4``. The drawing helpers and the crop are those of
``scripts/render_wild_overlays.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_wild_overlays import (ARROW_M_PER_BW, ARROW_MIN_BW, ARROW_PALETTES, IMWRITE_PARAMS,  # noqa: E402
                                  crop_frame, crop_windows, draw_arrows, fog,
                                  load_people, pixels_per_metre)

LEARNED_PALETTE = ARROW_PALETTES["yellow"]
OPTIM_PALETTE = ARROW_PALETTES["red"]


def draw_forces(img: np.ndarray, person: dict, f: int, intrinsics: np.ndarray,
                palette: tuple) -> None:
    """The person's arrows at frame ``f`` on the full-resolution ``img``."""
    force = person["force"][f]
    anchors = person["anchors"][f]
    on = np.isfinite(force).all(-1) & (np.linalg.norm(np.nan_to_num(force), axis=-1) >= ARROW_MIN_BW)
    if on.any():
        draw_arrows(img, anchors[on], anchors[on] + force[on] * ARROW_M_PER_BW, intrinsics, 1.0,
                    pixels_per_metre(person["body"][f], intrinsics, 1.0), palette)


def render_clip(stem: str, learned_root: Path, learned_dir: str, optim_root: Path, optim_dir: str,
                out: Path, side: int, fog_amount: float) -> None:
    learned, clip = load_people(learned_root / stem, learned_dir)
    optim, _ = load_people(optim_root / stem, optim_dir)
    intrinsics, n_frames = clip["intrinsics"], clip["n_frames"]
    # The optimisation's forces are drawn from the LEARNED skeleton's joints, like the learned ones.
    optim = [{"covered": p["covered"], "body": q["body"], "anchors": q["anchors"], "force": p["force"]}
             for p, q in zip(optim, learned)]
    windows = crop_windows(learned + optim, intrinsics, n_frames, clip["width"], clip["height"],
                           clip["fps"])
    dirs = {name: out / name / stem for name in ("original", "comparison")}
    writers = {}
    for name, folder in dirs.items():
        folder.mkdir(parents=True, exist_ok=True)
        writers[name] = cv2.VideoWriter(str(folder.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"),
                                        clip["fps"], (side, side))
    cap = cv2.VideoCapture(str(clip["video"]))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {clip['video']}")
    try:
        for f in range(n_frames):
            ok, img = cap.read()
            if not ok:
                raise ValueError(f"{clip['video']}: decoded {f} frames, the dump has {n_frames}")
            original = crop_frame(img, windows[f], side)
            img = fog(img, fog_amount)
            for person in optim:
                if person["covered"][f]:
                    draw_forces(img, person, f, intrinsics[f], OPTIM_PALETTE)
            for person in learned:                                  # the learned arrows on top
                if person["covered"][f]:
                    draw_forces(img, person, f, intrinsics[f], LEARNED_PALETTE)
            comparison = crop_frame(img, windows[f], side)
            for name, frame in (("original", original), ("comparison", comparison)):
                cv2.imwrite(str(dirs[name] / f"{f:06d}.jpg"), frame, IMWRITE_PARAMS["jpg"])
                writers[name].write(frame)
    finally:
        cap.release()
        for writer in writers.values():
            writer.release()
    print(f"  {stem}: {n_frames} frames at {side}x{side} into {out}/{{original,comparison}}/{stem}",
          flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--learned-root", type=Path, required=True)
    parser.add_argument("--learned-dir", default="joint_frames35")
    parser.add_argument("--optim-root", type=Path, required=True)
    parser.add_argument("--optim-dir", default="bvr_optim")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stems", nargs="*", default=None, help="default: every clip with both dumps")
    parser.add_argument("--side", type=int, default=768)
    parser.add_argument("--fog", type=float, default=0.35, help="blend towards white, 0 = none")
    args = parser.parse_args()
    stems = args.stems or sorted(
        d.name for d in args.learned_root.iterdir()
        if (d / "predictions" / args.learned_dir / "smplx.npz").is_file()
        and (args.optim_root / d.name / "predictions" / args.optim_dir / "smplx.npz").is_file())
    if not stems:
        raise SystemExit("no clip with both dumps")
    for index, stem in enumerate(stems, start=1):
        print(f"[{index}/{len(stems)}] {stem}", flush=True)
        render_clip(stem, args.learned_root, args.learned_dir, args.optim_root, args.optim_dir,
                    args.out, args.side, args.fog)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
