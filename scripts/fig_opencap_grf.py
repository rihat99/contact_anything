"""Paper figure: one OpenCap clip's ground reaction force, plates vs our two methods.

Two rows (left / right foot) by three columns (vertical, anterior-posterior, medio-lateral),
in % of the MEASURED body weight, after the benchmark's per-activity low-pass. The plates
are drawn thick and dark, the learned model and the optimisation on top; both of ours are
read in the subject's measured mass (their newtons rescaled by measured / reconstructed
mass, the ``measured mass (rescale)`` rows of the benchmark tables). Frames the plates did
not watch (drop-jump box) or a method did not cover are left blank.

    .venv/bin/python scripts/fig_opencap_grf.py --clip subject11_walking3_cam1 \\
        --out output_7/logs/figures_20260923/opencap_grf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opencap_review import CUTOFF_HZ, GRAVITY, PROCESSED, lowpass, trial_type  # noqa: E402

#: (label, run directory under ``processed/out/<clip>/``, colour, line width).
METHODS = [
    ("Ours (learned)", "method_joint_frames35", "#0072B2", 1.3),
    ("Ours (optimisation)", "human_optim_floorplane_edgev", "#D55E00", 1.3),
]
PLATES = ("Force plates", "#222222", 2.4)
AXES = (("Vertical", 1), ("Anterior-posterior", 0), ("Medio-lateral", 2))
FEET = ("Left foot", "Right foot")
STYLE = {"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9.5,
         "axes.titlesize": 10, "legend.fontsize": 9, "xtick.labelsize": 8.5,
         "ytick.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
         "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42}


def method_forces(clip: str, run: str, n: int, body_weight: float, cutoff_hz: float,
                  fps: float) -> tuple[np.ndarray, np.ndarray]:
    """``(n, 2, 3)`` %BW foot forces in the measured mass and the ``(n,)`` rows the run covered."""
    path = PROCESSED / "out" / clip / run / "estimated_forces.npz"
    est = np.load(path, allow_pickle=True)
    cols = [list(map(str, est["limbs"])).index(name) for name in ("left_foot", "right_foot")]
    m = min(n, len(est["force"]))
    newtons = np.asarray(est["force"], float)[:m, cols]
    if "total_mass" in est.files:
        own_mass = float(est["total_mass"])
    else:                                    # the optimisation: its mass lives in the solve
        own_mass = float(np.load(path.parent / "kindyn_1.npz", allow_pickle=True)["total_mass"][0])
    newtons = newtons * body_weight / (own_mass * GRAVITY)
    forces = np.zeros((n, 2, 3))
    forces[:m] = lowpass(newtons, cutoff_hz, fps) / body_weight * 100
    valid = np.zeros(n, bool)
    if "valid_mask" in est.files:
        valid[:m] = np.asarray(est["valid_mask"], bool)[:m]
    else:
        kindyn = np.load(path.parent / "kindyn_1.npz", allow_pickle=True)
        valid[:m] = np.asarray(kindyn["valid_mask"], bool)[0, :m]
    return forces, valid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clip", default="subject11_walking3_cam1")
    parser.add_argument("--out", type=Path, required=True, help="path without extension")
    parser.add_argument("--width", type=float, default=7.0, help="figure width in inches")
    args = parser.parse_args()

    gt = np.load(PROCESSED / args.clip / "forces.npz", allow_pickle=True)
    fps, kind = float(gt["fps"]), trial_type(args.clip)
    body_weight = float(gt["mass_kg"]) * GRAVITY
    plates_raw = np.asarray(gt["force"], float)[:, 2:]
    captured = np.asarray(gt["captured"], bool)[:, 2:]
    n = len(plates_raw)
    plates = lowpass(np.nan_to_num(plates_raw), CUTOFF_HZ[kind], fps) / body_weight * 100
    time = np.arange(n) / fps
    methods = [(label, colour, width, *method_forces(args.clip, run, n, body_weight,
                                                     CUTOFF_HZ[kind], fps))
               for label, run, colour, width in METHODS]

    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(2, 3, figsize=(args.width, 0.52 * args.width), sharex=True)
    for foot in range(2):
        for column, (title, k) in enumerate(AXES):
            axis = axes[foot, column]
            axis.axhline(0, color="0.8", lw=0.7, zorder=0)
            axis.plot(time, np.where(captured[:, foot], plates[:, foot, k], np.nan),
                      color=PLATES[1], lw=PLATES[2], label=PLATES[0], solid_capstyle="round",
                      zorder=1)
            for label, colour, width, forces, valid in methods:
                shown = captured[:, foot] & valid
                axis.plot(time, np.where(shown, forces[:, foot, k], np.nan), color=colour,
                          lw=width, label=label, zorder=2)
            axis.grid(True, color="0.92", lw=0.6)
            axis.set_axisbelow(True)
            axis.margins(x=0.01)
            if foot == 0:
                axis.set_title(title)
            else:
                axis.set_xlabel("Time [s]")
            if column == 0:
                axis.set_ylabel(f"{FEET[foot]}\nforce [% BW]")
    ylim = [ax.get_ylim() for ax in axes[:, 0]]
    for axis in axes[:, 0]:
        axis.set_ylim(min(y[0] for y in ylim), max(y[1] for y in ylim))
    for column in (1, 2):
        ylim = [ax.get_ylim() for ax in axes[:, column]]
        for axis in axes[:, column]:
            axis.set_ylim(min(y[0] for y in ylim), max(y[1] for y in ylim))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.4, columnspacing=2.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.8, w_pad=1.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(args.out.with_suffix(suffix), dpi=300, bbox_inches="tight")
    print(f"{args.clip}: mass {float(gt['mass_kg']):.1f} kg, {n} frames at {fps:.0f} fps, "
          f"{CUTOFF_HZ[kind]:.0f} Hz low-pass -> {args.out}.pdf/.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
