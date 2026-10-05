"""Per-clip review of every method's GRF against the OpenCap plates: one figure and one flag row per clip.

For each of the 64 natural clips draws the two feet × three OpenSim ground axes (AP, V up,
ML) in %BW of the measured mass — the plates thick underneath, every method on top — after
the scorer's per-activity low-pass, and tabulates what a reviewer should look at: the
video-to-mocap sync offset and its uncertainty flag, the plates' loaded fraction, and per
method the vertical correlation over stance and the stance F1. Rows are flagged where one of
OUR methods' vertical correlation falls below ``--flag-r`` (a sign of a swapped foot, a sync
alias, or a broken solve) so the eye goes to the clips that need it; the two external methods
are drawn and tabulated but do not flag (they fail on most clips for reasons of their own).

Reads ``processed/<clip>/forces.npz`` (synced plates) and ``processed/out/<clip>/<run>/
estimated_forces.npz``. Writes ``<out>/<clip>.png`` and ``<out>/review.md``.

Usage::

    .venv/bin/python scripts/opencap_review.py --out output_7/logs/opencap_review_<date>/
    .venv/bin/python scripts/opencap_review.py --out <dir> --clip subject11_walking2_cam1 --runs joint --measured-mass
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import butter, filtfilt  # noqa: E402

PROCESSED = Path("/home/rikhat.akizhanov/better/data/LabValidation_withVideos/processed")
CLIPS = PROCESSED / "out" / "_methods" / "clips_natural.txt"

#: (label, run directory under out/<clip>/, colour).
RUNS = [
    ("Li et al.", "method_estmf", "#999999"),
    ("PhysPT", "method_physpt", "#CC79A7"),
    ("optim edgev", "human_optim_floorplane_edgev", "#009E73"),
    ("optim massedgev", "human_optim_floorplane_massedgev", "#56B4E9"),
    ("climbing", "method_climbing_frames35", "#E69F00"),
    ("bedlam", "method_bedlam_frames35", "#0072B2"),
    ("joint", "method_joint_frames35", "#000000"),
]
PLATES = "#D55E00"
AXES = ("AP", "V (up)", "ML")
CUTOFF_HZ = {"walking": 6.0, "squats": 4.0, "STS": 4.0, "DJ": 30.0}
STANCE_BW = 0.05
LOADED_N = 20.0
GRAVITY = 9.81


def trial_type(clip: str) -> str:
    return re.sub(r"\d+(p\d+)?$", "", clip.split("_")[1])


def lowpass(x: np.ndarray, cutoff_hz: float, fps: float) -> np.ndarray:
    if cutoff_hz >= 0.5 * fps:
        return x
    b, a = butter(2, cutoff_hz / (0.5 * fps))
    return filtfilt(b, a, x, axis=0)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def review_clip(clip: str, out_dir: Path, flag_r: float, measured_mass: bool) -> dict:
    gt = np.load(PROCESSED / clip / "forces.npz", allow_pickle=True)
    fps, kind = float(gt["fps"]), trial_type(clip)
    body_weight = float(gt["mass_kg"]) * GRAVITY
    plates_raw = np.asarray(gt["force"], float)[:, 2:]                       # (N, 2, 3) feet
    captured = np.asarray(gt["captured"], bool)[:, 2:]
    n = len(plates_raw)
    plates = lowpass(np.nan_to_num(plates_raw), CUTOFF_HZ[kind], fps) / body_weight * 100
    sync_path = PROCESSED / clip / "sync.json"
    sync = json.loads(sync_path.read_text()) if sync_path.exists() else {}
    row = {"clip": clip, "type": kind, "frames": n,
           "sync": sync.get("mocap_offset_frames", np.nan),
           "sync uncertain": bool(sync.get("uncertain", False)),
           "loaded frac": float((np.nan_to_num(plates_raw[..., 1]) > STANCE_BW * body_weight).mean())}

    fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharex=True)
    time = np.arange(n) / fps
    for foot in range(2):
        for k in range(3):
            axis = axes[foot, k]
            axis.axhline(0, color="0.85", lw=0.8)
            shown = np.where(captured[:, foot], plates[:, foot, k], np.nan)
            axis.plot(time, shown, color=PLATES, lw=2.6, label="plates")
            axis.grid(True, color="0.92", lw=0.6)
            axis.set_axisbelow(True)
    worst = np.inf
    for label, run, colour in RUNS:
        path = PROCESSED / "out" / clip / run / "estimated_forces.npz"
        if not path.exists():
            row[f"{label} r"] = np.nan
            continue
        est = np.load(path, allow_pickle=True)
        cols = [list(map(str, est["limbs"])).index(name) for name in ("left_foot", "right_foot")]
        m = min(n, len(est["force"]))
        newtons = np.asarray(est["force"], float)[:m, cols]
        if measured_mass:               # body-weight units read in the subject's measured mass
            newtons = newtons * body_weight / (float(est["total_mass"]) * GRAVITY)
        ours = np.zeros((n, 2, 3))
        ours[:m] = lowpass(newtons, CUTOFF_HZ[kind], fps)
        ours = ours / body_weight * 100
        valid = np.zeros(n, bool)
        if "valid_mask" in est.files:
            valid[:m] = np.asarray(est["valid_mask"], bool)[:m]
        else:                       # our optimisation: the mask lives in the solve's file
            kindyn = np.load(path.parent / "kindyn_1.npz", allow_pickle=True)
            valid[:m] = np.asarray(kindyn["valid_mask"], bool)[0, :m]
        for foot in range(2):
            watched = captured[:, foot] & valid
            stance = watched & (plates[:, foot, 1] > STANCE_BW * 100)
            for k in range(3):
                axes[foot, k].plot(time, np.where(watched, ours[:, foot, k], np.nan),
                                   color=colour, lw=1.1, label=label)
            r = pearson(ours[stance, foot, 1], plates[stance, foot, 1])
            row[f"{label} r{'LR'[foot]}"] = r
            if np.isfinite(r) and label not in ("Li et al.", "PhysPT"):   # flag on OUR methods
                worst = min(worst, r)
        watched = captured & valid[:, None]
        label_loaded = (plates_raw[..., 1] > LOADED_N)[watched]
        pred_loaded = (ours[..., 1] * body_weight / 100 > LOADED_N)[watched]
        tp = (label_loaded & pred_loaded).sum()
        precision = tp / max((pred_loaded).sum(), 1)
        recall = tp / max(label_loaded.sum(), 1)
        row[f"{label} F1"] = float(2 * precision * recall / max(precision + recall, 1e-9))
    row["flag"] = bool(worst < flag_r) or row["sync uncertain"]
    for k, name in enumerate(AXES):
        axes[0, k].set_title(f"{name} [%BW]", fontsize=10)
        axes[1, k].set_xlabel("time [s]", fontsize=9)
    axes[0, 0].set_ylabel("left foot", fontsize=10)
    axes[1, 0].set_ylabel("right foot", fontsize=10)
    axes[0, 1].legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    fig.suptitle(f"{clip}  (sync {row['sync']:+.1f} fr{', UNCERTAIN' if row['sync uncertain'] else ''}, "
                 f"mass {float(gt['mass_kg']):.1f} kg{' (predictions scaled to it)' if measured_mass else ''}, "
                 f"{CUTOFF_HZ[kind]:.0f} Hz low-pass)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / f"{clip}.png", dpi=110)
    plt.close(fig)
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clips", type=Path, default=CLIPS)
    ap.add_argument("--clip", nargs="*", default=None, help="these clips only (default: the --clips list)")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="method labels to draw and tabulate (default: all of RUNS)")
    ap.add_argument("--measured-mass", action="store_true",
                    help="read our body-weight forces in the subject's measured mass instead of the "
                         "reconstructed one (force x mass_kg / total_mass; our methods only)")
    ap.add_argument("--flag-r", type=float, default=0.5,
                    help="flag a clip when any method's stance vertical correlation is below this")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    clips = args.clip or [c.strip() for c in args.clips.read_text().splitlines() if c.strip()]
    if args.runs is not None:
        unknown = set(args.runs) - {label for label, _, _ in RUNS}
        if unknown:
            raise SystemExit(f"unknown --runs {sorted(unknown)}; choose from {[l for l, _, _ in RUNS]}")
        RUNS[:] = [entry for entry in RUNS if entry[0] in args.runs]
    rows = [review_clip(clip, args.out, args.flag_r, args.measured_mass) for clip in clips]

    labels = [label for label, _, _ in RUNS]
    head = ["clip", "type", "frames", "sync fr", "loaded"] + [f"{l} rL/rR/F1" for l in labels]
    lines = [f"# OpenCap review — {len(rows)} clips", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in rows:
        cells = [("**" if row["flag"] else "") + row["clip"] + ("**" if row["flag"] else ""),
                 row["type"], str(row["frames"]),
                 f"{row['sync']:+.1f}{' ?' if row['sync uncertain'] else ''}",
                 f"{row['loaded frac']:.2f}"]
        for label in labels:
            r_l, r_r, f1 = row.get(f"{label} rL", np.nan), row.get(f"{label} rR", np.nan), row.get(f"{label} F1", np.nan)
            fmt = lambda v: "—" if not np.isfinite(v) else f"{v:.2f}"  # noqa: E731
            cells.append(f"{fmt(r_l)} / {fmt(r_r)} / {fmt(f1)}")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f"Bold = one of OUR methods' stance vertical correlation below {args.flag_r} on either foot, "
              "or an uncertain sync. r = Pearson correlation of the low-passed vertical force over "
              "that foot's stance frames; F1 = the method's loaded state (vertical > 20 N) against the "
              "plate's, both feet. One PNG per clip beside this file.", ""]
    (args.out / "review.md").write_text("\n".join(lines))
    flagged = [r["clip"] for r in rows if r["flag"]]
    print(f"{len(rows)} clips reviewed, {len(flagged)} flagged: {flagged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
