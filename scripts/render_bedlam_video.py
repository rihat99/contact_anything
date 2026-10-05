"""Draw one BEDLAM scene's predicted contacts and forces beside its vertical-GRF graph.

The input is a whole-scene dump of :mod:`scripts.predict_test` (``slot_points_world``,
``contact_probs``, ``forces_world`` in body-weight units and world axes) and the scene's
own ground truth under ``features/gt/<shard>/<scene>/`` — ``forces.npz`` (world newtons
per contact frame, ``frame_contact``, ``total_mass``, ``gravity_world``) and
``camera.npz`` (per-frame OpenCV ``extrinsics`` + ``intrinsics_px``).

The picture is BVR's (``scripts/diagnostics/render_sup_force_overlays.py``): the video
pane on the left carries a two-ring disc per contact slot at that slot's projected world
point — outer ring = the model's contact head above the threshold, inner disc = the GT
label of that frame — and one arrow per slot along the force on the body, green for the
prediction and red for the GT, both projected so an arrow shortens with distance. The
right pane is five stacked vertical-GRF panels (left foot, right foot, left hand, right
hand, all 35 slots summed) drawn once, with only a cursor moving. Vertical is the
component along UP = ``-gravity_world``; body weights become newtons through the scene's
own ``total_mass`` — the mass the labels were made with.

Writes ``<out>/<scene>_overlay.mp4`` and ``<out>/<scene>_grf.png``.

    python scripts/render_bedlam_video.py \\
        --dump output_7/<run>/predictions_examples/0b000925.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _render_common as rc                                     # noqa: E402
from data.bedlam2.scene import gt_dir, scene_shard              # noqa: E402
from model.contact_frames import contact_set                    # noqa: E402

DEFAULT_ROOT = Path("/home/rikhat.akizhanov/better/data/BEDLAM2_our")
GRAVITY = 9.81
#: Arrow length in metres per unit body weight — BVR's scale, so the two read alike.
METRES_PER_BW = 0.35
#: Below this an arrow is under a pixel of ink.
MIN_ARROW_BW = 0.02

CONTACT_BGR = (45, 45, 235)
FREE_BGR = (55, 185, 75)
WHITE_BGR = (245, 245, 245)
PRED_FORCE_BGR = (60, 200, 60)
GT_FORCE_BGR = (40, 40, 220)
CURSOR_BGR = (30, 30, 205)

#: The five graph panels: (name, slot indices summed) — a plate foot is toes + heel.
PANELS = (("left foot", (2, 4)), ("right foot", (3, 5)),
          ("left hand", (0,)), ("right hand", (1,)), ("all slots", None))


def _fold_vertical(forces: np.ndarray, up: np.ndarray, slots) -> np.ndarray:
    """``[N, 5]`` vertical load per graph panel of ``forces [N, 35, 3]`` (newtons)."""
    groups = slots.fold_sum(np.nan_to_num(forces))                      # [N, 6, 3]
    columns = []
    for _, members in PANELS:
        vector = forces.sum(1) if members is None else groups[:, list(members)].sum(1)
        columns.append(vector @ up)
    out = np.stack(columns, 1)
    return np.where(np.isfinite(forces).all((1, 2))[:, None], out, np.nan)


def _grf_figure(time: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                size_px: tuple[int, int], title: str | None):
    """The five stacked panels at ``size_px = (width, height)``. ``-> (fig, axes)``."""
    scale = size_px[1] / 720.0
    fig, axes = plt.subplots(
        len(PANELS), 1, figsize=(size_px[0] / 100.0, size_px[1] / 100.0), dpi=100,
        sharex=True)
    for axis, (name, _), truth, guess in zip(axes, PANELS, gt.T, pred.T):
        axis.plot(time, truth, color="black", lw=1.4 * scale, label="GT")
        axis.plot(time, guess, color="#1f9d3a", lw=1.4 * scale, label="predicted")
        axis.text(0.008, 0.95, name, transform=axis.transAxes, ha="left", va="top",
                  fontsize=11 * scale, color="0.15")
        axis.tick_params(labelsize=9 * scale)
        axis.grid(alpha=0.25, lw=0.6 * scale)
        # Headroom for the in-panel name, so it never sits on a curve.
        low, high = axis.get_ylim()
        axis.set_ylim(low, high + 0.22 * (high - low))
    axes[0].legend(loc="upper right", fontsize=9 * scale, ncol=2, framealpha=0.9,
                   borderpad=0.3, handlelength=1.4)
    axes[-1].set_xlim(0.0, float(time[-1]))
    axes[-1].set_xlabel("time [s]", fontsize=9 * scale)
    axes[len(PANELS) // 2].set_ylabel("vertical force [N]", fontsize=10 * scale)
    if title is not None:
        axes[0].set_title(title, fontsize=10 * scale)
    fig.subplots_adjust(left=0.10, right=0.995, top=0.94 if title else 0.995,
                        bottom=0.075, hspace=0.08)
    fig.canvas.draw()
    return fig, axes


def _panel_image(fig, axes) -> tuple[np.ndarray, callable, int, int]:
    """``(BGR panel, time -> pixel column, cursor top, cursor bottom)``."""
    panel = np.asarray(fig.canvas.buffer_rgba())[..., :3][..., ::-1].copy()
    height = panel.shape[0]
    column_of = lambda t: int(round(axes[0].transData.transform((t, 0.0))[0]))  # noqa: E731
    top = height - int(round(axes[0].transAxes.transform((0.0, 1.0))[1]))
    bottom = height - int(round(axes[-1].transAxes.transform((0.0, 0.0))[1]))
    return panel, column_of, top, bottom


def _project(points_cam: np.ndarray, intr: np.ndarray, scale: float) -> np.ndarray:
    """Pixels of camera-frame points, NaN behind the camera, at the render scale."""
    pixels = rc.project(points_cam, intr) * scale
    return np.where((points_cam[..., 2:3] > 1e-3), pixels, np.nan)


def _draw_discs(pane: np.ndarray, pixels: np.ndarray, pred: np.ndarray | None,
                truth: np.ndarray | None, radius: int) -> None:
    """One two-ring disc per slot: outer ring = prediction, inner disc = GT label."""
    for slot, (x, y) in enumerate(pixels):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        centre = (int(round(x)), int(round(y)))
        if pred is not None:
            cv2.circle(pane, centre, radius, CONTACT_BGR if pred[slot] else FREE_BGR,
                       -1, cv2.LINE_AA)
            cv2.circle(pane, centre, max(1, radius - 2), WHITE_BGR, -1, cv2.LINE_AA)
        if truth is not None:
            cv2.circle(pane, centre, max(1, radius - 3),
                       CONTACT_BGR if truth[slot] else FREE_BGR, -1, cv2.LINE_AA)


def _draw_arrows(pane: np.ndarray, anchor_cam: np.ndarray, force_bw: np.ndarray,
                 rotation: np.ndarray, intr: np.ndarray, scale: float, colour) -> None:
    """One arrow per loaded slot from its anchor along the force, both ends projected."""
    for start, force in zip(anchor_cam, force_bw):
        if not (np.isfinite(start).all() and np.isfinite(force).all()):
            continue
        if np.linalg.norm(force) < MIN_ARROW_BW or start[2] <= 1e-3:
            continue
        end = start + (rotation @ force) * METRES_PER_BW
        if end[2] <= 1e-3:
            continue
        tail, head = (tuple(int(round(v)) for v in _project(p, intr, scale))
                      for p in (start, end))
        cv2.arrowedLine(pane, tail, head, colour, max(1, int(round(2 * scale))),
                        cv2.LINE_AA, tipLength=0.25)


def _text(pane: np.ndarray, lines) -> None:
    """Caption lines at a fixed readable size, black-outlined, top left."""
    for row, (colour, line) in enumerate(lines):
        origin = (10, 20 + 18 * row)
        cv2.putText(pane, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0),
                    3, cv2.LINE_AA)
        cv2.putText(pane, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour,
                    1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, type=Path,
                        help="a predict_test.py whole-scene dump (<scene>.npz)")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="BEDLAM2_our corpus root")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default <dump>/../../render_examples)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="contact probability above which the model claims contact")
    parser.add_argument("--scale", type=float, default=0.5, help="video downscale")
    args = parser.parse_args()

    scene = args.dump.stem
    out_dir = args.out or args.dump.parent.parent / "render_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    dump = np.load(args.dump, allow_pickle=True)
    gt = np.load(gt_dir(args.root, scene) / "forces.npz", allow_pickle=True)
    camera = np.load(gt_dir(args.root, scene) / "camera.npz", allow_pickle=True)
    frames_dir = args.root / "frames" / scene_shard(scene) / scene

    slots = contact_set(str(dump["contact_set"]))
    names = [str(x) for x in dump["slot_names"]]
    if names != [str(x) for x in gt["contact_frame_names"]]:
        raise ValueError(f"{scene}: the dump's slot_names are not the GT contact frames")
    if dump["forces_world"].shape[0] != 1:
        raise ValueError(f"{scene}: this renderer draws one person, the dump holds "
                         f"{dump['forces_world'].shape[0]}")

    mass = float(np.asarray(gt["total_mass"], float)[0])
    down = np.asarray(gt["gravity_world"], float)
    up = -down / np.linalg.norm(down)
    fps = float(dump["fps"])

    covered = np.asarray(dump["covered"][0], bool)                      # [N]
    valid = np.asarray(gt["valid_mask"][0], bool)                       # [N]
    points_world = np.asarray(dump["slot_points_world"][0], np.float64)  # [N, 35, 3]
    probs = np.asarray(dump["contact_probs"][0], np.float64)            # [N, 35]
    pred_bw = np.asarray(dump["forces_world"][0], np.float64)           # [N, 35, 3] bw
    gt_newton = np.asarray(gt["frame_forces"][0], np.float64)           # [N, 35, 3] N
    gt_contact = np.asarray(gt["frame_contact"][0], bool)               # [N, 35]
    n_frames = len(covered)

    # Uncovered / invalid rows carry nothing: NaN in the graphs, nothing on the video.
    pred_newton = np.where(covered[:, None, None], pred_bw * mass * GRAVITY, np.nan)
    gt_newton = np.where(valid[:, None, None], gt_newton, np.nan)
    pred_vertical = _fold_vertical(pred_newton, up, slots)              # [N, 5]
    gt_vertical = _fold_vertical(gt_newton, up, slots)

    both = covered & valid
    mae = np.abs(pred_vertical[both] - gt_vertical[both]).mean(0)
    title = (f"{scene} — vertical GRF (up = -gravity), mass {mass:.1f} kg;  MAE [N]: "
             + ",  ".join(f"{name} {value:.0f}"
                          for (name, _), value in zip(PANELS, mae)))
    time = np.arange(n_frames) / fps

    figure, axes = _grf_figure(time, gt_vertical, pred_vertical, (1400, 900), title)
    png = out_dir / f"{scene}_grf.png"
    figure.savefig(png, dpi=100)
    plt.close(figure)

    first = rc.read_frame(frames_dir, 0)
    pane_size = (int(round(first.shape[1] * args.scale)),
                 int(round(first.shape[0] * args.scale)))
    figure, axes = _grf_figure(time, gt_vertical, pred_vertical,
                               (int(round(pane_size[1] * 1.35)), pane_size[1]), None)
    panel, column_of, cursor_top, cursor_bottom = _panel_image(figure, axes)
    plt.close(figure)

    radius = max(3, int(round(7 * args.scale)))
    video = out_dir / f"{scene}_overlay.mp4"
    writer = rc.open_writer(
        video, fps, (pane_size[0] + panel.shape[1], pane_size[1]))
    try:
        for frame in range(n_frames):
            pane = cv2.resize(rc.read_frame(frames_dir, frame), pane_size,
                              interpolation=cv2.INTER_AREA)
            extrinsic = np.asarray(camera["extrinsics"][frame], np.float64)
            intr = np.asarray(camera["intrinsics_px"][frame], np.float64)
            points_cam = (points_world[frame] @ extrinsic[:3, :3].T) + extrinsic[:3, 3]
            pixels = _project(points_cam, intr, args.scale)
            _draw_discs(pane, pixels,
                        probs[frame] >= args.threshold if covered[frame] else None,
                        gt_contact[frame] if valid[frame] else None, radius)
            if valid[frame]:
                _draw_arrows(pane, points_cam, gt_newton[frame] / (mass * GRAVITY),
                             extrinsic[:3, :3], intr, args.scale, GT_FORCE_BGR)
            if covered[frame]:
                _draw_arrows(pane, points_cam, pred_bw[frame], extrinsic[:3, :3],
                             intr, args.scale, PRED_FORCE_BGR)
            _text(pane, [
                (WHITE_BGR, f"frame {frame:4d}   vertical total  pred "
                            f"{pred_vertical[frame, -1]:7.0f} N   GT "
                            f"{gt_vertical[frame, -1]:7.0f} N"),
                (PRED_FORCE_BGR, "green arrow = predicted force"),
                (GT_FORCE_BGR, "red arrow = GT force"),
                (WHITE_BGR, "disc: outer = pred contact, inner = GT "
                            "(red = contact, green = free)"),
            ])
            cursor = panel.copy()
            column = column_of(frame / fps)
            cursor[cursor_top:cursor_bottom, max(0, column - 1):column + 2] = CURSOR_BGR
            writer.write(np.hstack([pane, cursor]))
    finally:
        writer.release()

    print(f"{scene}: {n_frames} frames, mass {mass:.1f} kg, "
          f"covered {int(covered.sum())}, GT valid {int(valid.sum())}")
    for (name, _), value, truth, guess in zip(
            PANELS, mae, np.nanmean(gt_vertical, 0), np.nanmean(pred_vertical, 0)):
        print(f"  {name:>10s}: MAE {value:8.1f} N   GT mean {truth:8.1f} N   "
              f"pred mean {guess:8.1f} N")
    print(f"wrote {video}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
