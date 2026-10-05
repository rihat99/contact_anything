"""Paper figure: one climb_wall_3 clip — a video frame beside the four limbs' force traces.

Left: one frame of ``cam_left.mp4``, cropped to the climber. Right: one panel per limb with
the size of the limb's force in newtons — the instrumented board dark, the learned model
(PACT) and the optimisation on top. A dotted line marks the frame shown; the median direction
error of each method (angle to the board where both read above ``CONTACT_N``) prints to the
console.

``--video`` writes the same figure as ``<out>.mp4`` instead: the crop of the still (one fixed
box) plays the whole clip, a solid cursor sweeps the traces, and PACT's in-contact limb forces
are drawn on the video as yellow shaded arrows (``render_wild_overlays.py``'s arrows: the hand
arrow from the middle-finger base, the foot slots summed onto the ankle) at
``--arrow-m-per-bw`` metres per body weight, the shaft thinned with the length so the arrows keep
the overlay's proportions. An arrow that leaves the crop's top carries on over white above the
photo (the figure's margin), so the photo stays the still's crop. The frames are piped into
``--ffmpeg`` (H.264).

Our body-weight output becomes newtons through the RECONSTRUCTED mass (``total_mass`` of the
clip's ``human_optim/kindyn_1.npz``), the mass the optimisation solved its forces with, so
both rows read in the same mass (``--mass optim`` of ``score_rig.py``).

    .venv/bin/python scripts/fig_climb_wall_3.py --clip a_16_03_part1 \\
        --out output_7/logs/figures_20260925/climb_wall_3_a_16_03_part1
    .venv/bin/python scripts/fig_climb_wall_3.py --clip a_16_03_part1 --video \\
        --ffmpeg ~/miniconda3/envs/estmf/bin/ffmpeg \\
        --out output_7/logs/figures_20260925/climb_wall_3_a_16_03_part1
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
_BVR = _ROOT.parent / "BetterVideoReconstruction"
for _p in (_BVR, _BVR / "scripts", _BVR / "scripts" / "diagnostics"):
    sys.path.insert(0, str(_p))

from _render_common import project                                   # noqa: E402
from compare_gt_contact import limb_forces                           # noqa: E402
from model.contact_frames import contact_set                         # noqa: E402
from model.physics import GRAVITY                                    # noqa: E402
from render_wild_overlays import (ARROW_M, ARROW_M_PER_BW, ARROW_MIN_BW,  # noqa: E402
                                  ARROW_PALETTES, draw_arrows, load_people, pixels_per_metre)
from score_rig import GROUP_LIMBS, board_forces                      # noqa: E402

TREE = _BVR / "peter" / "out_climb_wall_3_single"
VIDEOS = _BVR / "peter" / "climb_wall_3"
LIMB_TITLES = ("Left hand", "Right hand", "Left foot", "Right foot")
CONTACT_N = 50.0
BOARDS = ("Instrumented boards", "#222222", 2.0)
METHODS = (("PACT", "#0072B2", 1.2), ("Optimization", "#D55E00", 1.2))
#: The time cursor: dotted behind the traces on the still, solid and on top in the video.
STILL_CURSOR = {"color": "0.45", "lw": 0.8, "ls": (0, (1.5, 2.5)), "zorder": 0}
VIDEO_CURSOR = {"color": "0.1", "lw": 1.8, "ls": "-", "zorder": 3}
#: Video arrow length. The raised hands' arrows leave the top of the frame beyond 0.4 (on
#: a_16_03_part1: 86 px at 0.8, 138 px at 1.0) and continue above the photo; 0.8 stays clear of
#: the legend.
VIDEO_ARROW_M_PER_BW = 0.8
ARROW_HEADROOM_PX = 12              # white rows past the highest arrow tip: its head's black rim
PAD_INCHES = 0.1                    # margin around the tight bounding box, as savefig's "tight"
STYLE = {"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.5,
         "legend.fontsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
         "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
         "pdf.fonttype": 42, "ps.fonttype": 42}


def learned_forces(pred_dir: Path, n: int, body_weight: float) -> np.ndarray:
    """``(n, 4, 3)`` world-frame limb forces in newtons: the in-contact slots summed per limb."""
    forces = np.load(pred_dir / "forces_sup.npz", allow_pickle=True)
    slots = contact_set(str(forces["contact_set"]))
    bw = np.nan_to_num(np.asarray(forces["forces_world"][0][:n], np.float64))
    on = np.nan_to_num(np.asarray(forces["contact_probs"][0][:n], np.float64)) >= 0.5
    bw6 = slots.fold_sum(bw * on[..., None])
    return np.stack([bw6[:, list(g)].sum(1) for g in GROUP_LIMBS], 1) * body_weight


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between ``(n, 4, 3)`` vectors, NaN where either is below ``CONTACT_N``."""
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    both = (na > CONTACT_N) & (nb > CONTACT_N)
    cos = (a * b).sum(-1) / np.where(both, na * nb, 1.0)
    return np.where(both, np.degrees(np.arccos(np.clip(cos, -1, 1))), np.nan)


def choose_frame(boards: np.ndarray, valid: np.ndarray) -> int:
    """The frame nearest the clip's middle where every board reads above 100 N."""
    loaded = (np.linalg.norm(boards, axis=-1) > 100.0).all(1) & valid
    candidates = np.flatnonzero(loaded)
    if len(candidates) == 0:
        raise SystemExit("no frame with all four boards loaded")
    return int(candidates[np.argmin(np.abs(candidates - len(boards) / 2))])


def read_frame(video: Path, frame: int) -> np.ndarray:
    """One frame of ``video`` (BGR)."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot decode frame {frame}")
    return img


def crop_box(joints_px: np.ndarray, image_shape: tuple, aspect: float) -> tuple[int, int, int, int]:
    """``x0, y0, x1, y1`` around the projected body with a margin, at ``aspect`` = w/h, inside the image."""
    height, width = image_shape[:2]
    cx, cy = (joints_px.min(0) + joints_px.max(0)) / 2
    half_h = np.ptp(joints_px[:, 1]) / 2 + 0.10 * height
    half_w = max(half_h * aspect, np.ptp(joints_px[:, 0]) / 2 + 0.03 * width)
    half_h = half_w / aspect
    return (int(max(cx - half_w, 0)), int(max(cy - half_h, 0)),
            int(min(cx + half_w, width)), int(min(cy + half_h, height)))


def draw_figure(width: float, image: np.ndarray, time: np.ndarray, board_size: np.ndarray,
                rows: list, cursor_sec: float, cursor: dict) -> tuple:
    """The figure: ``image`` (RGB crop) beside the four limb panels, a cursor at ``cursor_sec``.

    :param rows: ``(force (n, 4, 3), valid (n,))`` per entry of :data:`METHODS`.
    :returns: ``(fig, image artist, the four cursor lines)``.
    """
    plt.rcParams.update(STYLE)
    fig = plt.figure(figsize=(width, 0.66 * width))
    outer = fig.add_gridspec(1, 2, width_ratios=(0.9, 1.7), wspace=0.16,
                             left=0.01, right=0.99, top=0.90, bottom=0.09)
    ax_img = fig.add_subplot(outer[0])
    image_artist = ax_img.imshow(image)
    ax_img.set_axis_off()

    blocks = outer[1].subgridspec(4, 1, hspace=0.45)
    shared = None
    cursors = []
    for limb in range(4):
        ax_f = fig.add_subplot(blocks[limb], sharex=shared)
        shared = shared or ax_f
        ax_f.plot(time, board_size[:, limb], color=BOARDS[1], lw=BOARDS[2], label=BOARDS[0])
        for (label, colour, lw), (force, valid) in zip(METHODS, rows):
            size = np.where(valid, np.linalg.norm(force[:, limb], axis=-1), np.nan)
            ax_f.plot(time, size, color=colour, lw=lw, label=label)
        cursors.append(ax_f.axvline(cursor_sec, **cursor))
        ax_f.grid(True, color="0.93", lw=0.5)
        ax_f.set_axisbelow(True)
        ax_f.margins(x=0.01)
        ax_f.set_ylim(0, 470)
        ax_f.set_yticks((0, 200, 400))
        ax_f.set_ylabel("Force [N]")
        ax_f.set_title(LIMB_TITLES[limb], loc="left", fontsize=9, pad=2.5,
                       fontweight="semibold")
        if limb < 3:
            ax_f.tick_params(labelbottom=False)
        else:
            ax_f.set_xlabel("Time [s]")
    handles, labels = shared.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=2.0)
    return fig, image_artist, cursors


def tight_pixels(fig, width_px: int) -> tuple[int, int, int, int]:
    """Set ``fig``'s dpi so its tight bounding box is ``width_px`` wide; that box in canvas pixels.

    Returns ``x0, y0, x1, y1`` (rows from the top), padded by ``PAD_INCHES``, inside the
    canvas and with an even width and height (yuv420p needs both).
    """
    fig.canvas.draw()
    box = fig.get_tightbbox(fig.canvas.get_renderer()).padded(PAD_INCHES)
    fig.set_dpi(width_px / box.width)
    fig.canvas.draw()
    dpi = fig.dpi
    canvas_w, canvas_h = fig.canvas.get_width_height()
    x0, x1 = max(int(box.x0 * dpi), 0), min(int(np.ceil(box.x1 * dpi)), canvas_w)
    y0, y1 = max(canvas_h - int(np.ceil(box.y1 * dpi)), 0), min(canvas_h - int(box.y0 * dpi), canvas_h)
    return x0, y0, x1 - (x1 - x0) % 2, y1 - (y1 - y0) % 2


def arrow_headroom(person: dict, intrinsics: np.ndarray, box: tuple, n: int,
                   m_per_bw: float) -> int:
    """Rows above the crop that the clip's arrows reach, plus ``ARROW_HEADROOM_PX`` (0 when none do)."""
    highest = float(box[1])
    for f in np.flatnonzero(person["covered"][:n]):
        force = np.nan_to_num(person["force"][f])
        on = np.linalg.norm(force, axis=-1) >= ARROW_MIN_BW
        if on.any():
            tips = project(person["anchors"][f][on] + force[on] * m_per_bw, intrinsics[f])
            highest = min(highest, float(tips[:, 1].min()))
    return int(np.ceil(box[1] - highest)) + ARROW_HEADROOM_PX if highest < box[1] else 0


def write_video(fig, image_artist, cursors: list, video: Path, box: tuple, person: dict,
                intrinsics: np.ndarray, n: int, fps: float, m_per_bw: float, out: Path,
                ffmpeg: str, width_px: int) -> None:
    """Play the clip in the figure's image panel with PACT's force arrows, sweep the cursors, encode ``out``.

    :param person: :func:`render_wild_overlays.load_people`'s entry (camera-frame joints,
        arrow anchors and forces in body weight).
    """
    # The panel image is the crop below `headroom` white rows; the axes keep the crop's limits
    # and the image draws past them, so the photo sits where the still has it.
    headroom = arrow_headroom(person, intrinsics, box, n, m_per_bw)
    crop_w, crop_h = box[2] - box[0], box[3] - box[1]
    image_artist.set_extent((-0.5, crop_w - 0.5, crop_h - 0.5, -headroom - 0.5))
    image_artist.set_clip_on(False)
    image_artist.axes.set_xlim(-0.5, crop_w - 0.5)
    image_artist.axes.set_ylim(crop_h - 0.5, -0.5)
    panel_from_image = np.array([[1.0, 0.0, -box[0]], [0.0, 1.0, headroom - box[1]], [0.0, 0.0, 1.0]])
    x0, y0, x1, y1 = tight_pixels(fig, width_px)
    encoder = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{x1 - x0}x{y1 - y0}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264",
         "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(out)], stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(video))
    palette = ARROW_PALETTES["yellow"]
    try:
        for f in range(n):
            ok, img = cap.read()
            if not ok:
                raise RuntimeError(f"{video}: decoded {f} frames, the figure has {n}")
            panel = np.full((headroom + crop_h, crop_w, 3), 255, np.uint8)
            panel[headroom:] = img[box[1]:box[3], box[0]:box[2]]
            if person["covered"][f]:
                force, anchors = person["force"][f], person["anchors"][f]
                on = np.isfinite(force).all(-1) & (
                    np.linalg.norm(np.nan_to_num(force), axis=-1) >= ARROW_MIN_BW)
                if on.any():
                    draw_arrows(panel, anchors[on], anchors[on] + force[on] * m_per_bw,
                                panel_from_image @ intrinsics[f], 1.0,
                                pixels_per_metre(person["body"][f], intrinsics[f], 1.0), palette,
                                thickness_m=ARROW_M * m_per_bw / ARROW_M_PER_BW)
            image_artist.set_data(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
            for line in cursors:
                line.set_xdata([f / fps, f / fps])
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[y0:y1, x0:x1, :3]
            encoder.stdin.write(np.ascontiguousarray(rgb).tobytes())
    finally:
        cap.release()
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError(f"{ffmpeg} failed on {out}")
    print(f"{out}: {n} frames at {fps:.0f} fps, {x1 - x0}x{y1 - y0}, "
          f"arrows {m_per_bw} m/bw, {headroom} px above the photo")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clip", default="a_16_03_part1")
    parser.add_argument("--run", default="joint_frames35")
    parser.add_argument("--frame", type=int, default=None,
                        help="frame shown, and the crop's frame in the video (default: auto)")
    parser.add_argument("--out", type=Path, required=True, help="path without extension")
    parser.add_argument("--width", type=float, default=7.0, help="figure width in inches")
    parser.add_argument("--video", action="store_true", help="write <out>.mp4 instead of the still")
    parser.add_argument("--video-width", type=int, default=1920, help="video width in pixels")
    parser.add_argument("--arrow-m-per-bw", type=float, default=VIDEO_ARROW_M_PER_BW,
                        help="video arrow length in metres per body weight")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"),
                        help="an ffmpeg with libx264 (default: the one on PATH)")
    args = parser.parse_args()
    if args.video and not args.ffmpeg:
        parser.error("--video needs an ffmpeg with libx264: none on PATH, pass --ffmpeg")

    tree = TREE / args.clip
    pred_dir = tree / "predictions" / args.run
    video = VIDEOS / args.clip / "cam_left.mp4"
    smplx = np.load(pred_dir / "smplx.npz", allow_pickle=True)
    camera = np.load(tree / "geometry" / "transform.npz", allow_pickle=True)
    solve = np.load(tree / "human_optim" / "kindyn_1.npz", allow_pickle=True)
    n = min(int(smplx["covered"].shape[1]), len(camera["extrinsics"]))
    fps = float(smplx["fps"])
    covered = np.asarray(smplx["covered"][0][:n], bool)
    mass = float(solve["total_mass"][0])
    boards = board_forces(TREE, args.clip, n)
    optim, _ = limb_forces(tree / "human_optim" / "kindyn_1.npz", n)
    optim_valid = np.asarray(solve["valid_mask"][0][:n], bool)
    learned = learned_forces(pred_dir, n, mass * GRAVITY)
    rows = [(learned, covered), (np.asarray(optim, np.float64), optim_valid)]
    frame = args.frame if args.frame is not None else choose_frame(boards, covered & optim_valid)
    time = np.arange(n) / fps
    board_size = np.linalg.norm(boards, axis=-1)

    height = 0.66 * args.width
    joints_px = project(np.asarray(smplx["joints_cam"][0][frame], np.float64),
                        np.asarray(camera["intrinsics_px_orig"][frame], np.float64))
    panel_aspect = (0.9 / 2.6 * args.width) / (height * (0.90 - 0.09))
    img = read_frame(video, frame)
    box = crop_box(joints_px, img.shape, panel_aspect)
    crop = cv2.cvtColor(img[box[1]:box[3], box[0]:box[2]], cv2.COLOR_BGR2RGB)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.video:
        fig, image_artist, cursors = draw_figure(args.width, crop, time, board_size, rows, 0.0,
                                                 VIDEO_CURSOR)
        people, _ = load_people(tree, args.run)
        write_video(fig, image_artist, cursors, video, box, people[0],
                    np.asarray(camera["intrinsics_px_orig"], np.float64), n, fps,
                    args.arrow_m_per_bw, args.out.with_suffix(".mp4"), args.ffmpeg,
                    args.video_width)
        return 0
    fig, _, _ = draw_figure(args.width, crop, time, board_size, rows, frame / fps, STILL_CURSOR)
    for suffix in (".pdf", ".png"):
        fig.savefig(args.out.with_suffix(suffix), dpi=300, bbox_inches="tight",
                    pad_inches=PAD_INCHES)
    angles = {label: np.nanmedian(angle_deg(force, boards)[valid])
              for (label, _, _), (force, valid) in zip(METHODS, rows)}
    print(f"{args.clip}: frame {frame}, mass {mass:.1f} kg, {n} frames at {fps:.0f} fps; "
          f"median angle " + ", ".join(f"{k} {v:.0f} deg" for k, v in angles.items())
          + f" -> {args.out}.pdf/.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
