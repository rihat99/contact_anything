"""Gait metrics of the vertical ground reaction force on the OpenCap walking trials, in % BW.

The biomechanics reading of :mod:`scripts.score_opencap`: the same dumps, the same plates,
but only the ``walking*`` clips, only the VERTICAL component (the projection on ``up`` =
``-gravity_world``), everything in percent of the subject's MEASURED body weight
(``mass_kg`` x g), and the numbers a gait paper reports per STANCE rather than per frame.

A stance is a contiguous run of frames on which that foot's plate reads above
:data:`~scripts.score_opencap.LOADED_N` (20 N), at least ``MIN_STANCE_S`` long, that starts
and ends inside the frames the plate was watching (a stance cut by the clip's edge is not a
stance). Inside each plate stance:

* **peak 1 / valley / peak 2** — the vertical GRF at the loading-response peak (max over the
  first half of the stance), the mid-stance valley (min over the middle half) and the
  push-off peak (max over the second half), % BW; plates and ours, then the signed error
  ours − plates. Ours is read on the rows the dump predicted (stride 2 at 60 fps = 30 Hz,
  ±17 ms), never interpolated.
* **impulse** — the vertical impulse of the stance, ∫ F dt, in BW·s (ours integrated over
  its predicted rows with their own spacing); signed error.
* **stance time** — the plate stance duration in seconds vs OURS: our own vertical force
  above 20 N, the longest run overlapping the plate stance; signed error in seconds.
* **peak timing** — frames between the plates' overall stance maximum and ours.

Per whole clip (all scored frames, not just the stances): **MAE** and **RMSE** of the
per-foot vertical GRF in % BW, the same over stance frames only, and the Pearson
correlation of each foot's vertical force over time. Loading rate is NOT reported: at
30 Hz effective sampling the rising edge is two or three samples.

Rows: ``optimisation`` (BVR's solve) and every run named. ``POOLED`` rows pool stances /
frames over the walking clips; a second pooled row leaves ``subject4_walking1`` out, whose
per-foot assignment is anti-phase with the plates for every method (see
``output_7/logs/opencap_20260920.md``).

    python scripts/score_opencap_gait.py --runs bedlam_frames35_ep4 climbing_frames35 \
        --out output_7/logs/opencap_gait_<date>.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.score_opencap import (  # noqa: E402
    GRAVITY, LOADED_N, PROCESSED, clip_names, optimisation_row, plates, prediction_row,
)

MIN_STANCE_S = 0.1
FEET = (2, 3)
FOOT_NAME = ("L", "R")
STANCE_COLUMNS = ("stances", "peak1 plate", "peak1 ours", "peak1 err", "valley plate",
                  "valley ours", "valley err", "peak2 plate", "peak2 ours", "peak2 err",
                  "impulse plate", "impulse ours", "impulse err", "stance s plate",
                  "stance s ours", "stance err s", "peak timing fr")
FRAME_COLUMNS = ("frames", "MAE %BW", "RMSE %BW", "stance MAE %BW", "stance RMSE %BW",
                 "corr L", "corr R")


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """``(start, end_exclusive)`` of every run of True."""
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def vertical_bw(row: dict, gt: dict) -> np.ndarray:
    """``(N, 2)`` vertical foot force in % BW on the method's predicted rows, NaN elsewhere."""
    force = row["force"][:, list(FEET)] @ gt["up"]
    force = np.where(row["rows"][:, None], force, np.nan)
    return 100.0 * force / (gt["mass_kg"] * GRAVITY)


def plate_vertical_bw(gt: dict) -> np.ndarray:
    """``(N, 2)`` plate vertical force in % BW, NaN where the plate did not see the foot."""
    force = gt["force"][:, list(FEET)] @ gt["up"]
    seen = gt["captured"][:, list(FEET)] & np.isfinite(force)
    return np.where(seen, 100.0 * force / (gt["mass_kg"] * GRAVITY), np.nan)


def stance_metrics(plate: np.ndarray, ours: np.ndarray, start: int, end: int,
                   fps: float, bw_n: float) -> dict[str, float] | None:
    """One stance's row, or None when ours has no predicted row inside it."""
    p = plate[start:end]
    o = ours[start:end]
    rows = np.flatnonzero(np.isfinite(o))
    if rows.size < 3:
        return None
    n = end - start
    half, quarter = n // 2, n // 4

    def segment(values: np.ndarray, lo: int, hi: int, reduce) -> float:
        seg = values[lo:hi]
        seg = seg[np.isfinite(seg)]
        return float(reduce(seg)) if seg.size else float("nan")

    peak1_p, peak1_o = segment(p, 0, half, np.max), segment(o, 0, half, np.max)
    valley_p = segment(p, quarter, n - quarter, np.min)
    valley_o = segment(o, quarter, n - quarter, np.min)
    peak2_p, peak2_o = segment(p, half, n, np.max), segment(o, half, n, np.max)
    impulse_p = float(np.nansum(p) / fps / 100.0)
    impulse_o = float(np.trapezoid(o[rows], rows / fps) / 100.0)
    # our stance = the longest run of our vertical force above the plate threshold that
    # overlaps the plate stance, searched over the whole clip on our predicted rows
    ours_loaded = np.nan_to_num(ours, nan=-1.0) > (100.0 * LOADED_N / bw_n)
    filled = ours_loaded.copy()
    predicted = np.isfinite(ours)
    # carry each predicted row's state across the unpredicted rows that follow it
    last = False
    for i in range(len(filled)):
        if predicted[i]:
            last = bool(ours_loaded[i])
        filled[i] = last
    overlapping = [(a, b) for a, b in runs_of(filled) if a < end and b > start]
    if overlapping:
        a, b = max(overlapping, key=lambda r: r[1] - r[0])
        stance_o = (b - a) / fps
    else:
        stance_o = 0.0
    stance_p = n / fps
    peak_t_p = start + int(np.nanargmax(p))
    peak_t_o = start + rows[int(np.argmax(o[rows]))]
    return {
        "stances": 1.0,
        "peak1 plate": peak1_p, "peak1 ours": peak1_o, "peak1 err": peak1_o - peak1_p,
        "valley plate": valley_p, "valley ours": valley_o, "valley err": valley_o - valley_p,
        "peak2 plate": peak2_p, "peak2 ours": peak2_o, "peak2 err": peak2_o - peak2_p,
        "impulse plate": impulse_p, "impulse ours": impulse_o,
        "impulse err": impulse_o - impulse_p,
        "stance s plate": stance_p, "stance s ours": stance_o,
        "stance err s": stance_o - stance_p,
        "peak timing fr": float(peak_t_o - peak_t_p),
    }


def frame_metrics(plate: np.ndarray, ours: np.ndarray, stance_mask: np.ndarray) -> dict:
    """Whole-clip (or pooled) per-frame numbers over the frames both sides have."""
    both = np.isfinite(plate) & np.isfinite(ours)
    err = (ours - plate)[both]
    in_stance = (ours - plate)[both & stance_mask]
    corr = []
    for foot in range(2):
        m = both[:, foot]
        corr.append(float(np.corrcoef(plate[m, foot], ours[m, foot])[0, 1])
                    if m.sum() > 2 else float("nan"))
    return {
        "frames": float(both.sum()),
        "MAE %BW": float(np.abs(err).mean()) if err.size else float("nan"),
        "RMSE %BW": float(np.sqrt((err ** 2).mean())) if err.size else float("nan"),
        "stance MAE %BW": float(np.abs(in_stance).mean()) if in_stance.size else float("nan"),
        "stance RMSE %BW": (float(np.sqrt((in_stance ** 2).mean()))
                            if in_stance.size else float("nan")),
        "corr L": corr[0], "corr R": corr[1],
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    """Mean over stances of every column; ``stances`` = the count; errors also as |mean|."""
    if not rows:
        return {k: float("nan") for k in STANCE_COLUMNS}
    out = {k: float(np.nanmean([r[k] for r in rows])) for k in STANCE_COLUMNS}
    out["stances"] = float(len(rows))
    return out


def fmt(key: str, value: float) -> str:
    if not np.isfinite(value):
        return "—"
    if key in ("stances", "frames"):
        return f"{int(value)}"
    if key.startswith("corr"):
        return f"{value:.2f}"
    if key.endswith(" s") or key.startswith("stance err") or key.startswith("impulse"):
        return f"{value:+.3f}" if "err" in key else f"{value:.3f}"
    if key == "peak timing fr":
        return f"{value:+.1f}"
    return f"{value:+.1f}" if "err" in key else f"{value:.1f}"


def table(rows: dict[str, dict[str, float]], columns: tuple[str, ...], first: str) -> list[str]:
    lines = ["| " + " | ".join([first, *columns]) + " |",
             "|" + "---|" * (len(columns) + 1)]
    for label, values in rows.items():
        lines.append("| " + " | ".join([label, *(fmt(c, values[c]) for c in columns)]) + " |")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--source", type=Path, default=PROCESSED)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    clips = [c for c in clip_names(args.source, args.runs) if "_walking" in c]
    if not clips:
        raise SystemExit("no walking clip has plates and every run's dump")
    labels = ["optimisation", *args.runs]
    stance_rows: dict[str, dict[str, list[dict]]] = {l: {c: [] for c in clips} for l in labels}
    frame_series: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        l: {} for l in labels}
    notes = []

    for clip in clips:
        clip_dir = args.source / "out" / clip
        gt = plates(clip_dir)
        bw_n = gt["mass_kg"] * GRAVITY
        plate = plate_vertical_bw(gt)                                        # (N, 2) %BW
        rows = {"optimisation": optimisation_row(clip_dir, gt)}
        for run in args.runs:
            rows[run] = prediction_row(clip_dir / "predictions" / run, gt, args.threshold)
        stance_mask = np.zeros_like(plate, dtype=bool)
        per_foot = []
        for foot in range(2):
            found = plate_stances(plate[:, foot], gt["fps"], bw_n)
            per_foot.append(found)
            for a, b in found:
                stance_mask[a:b, foot] = True
        notes.append(f"- `{clip}`: {gt['n']} frames at {gt['fps']:.0f} fps, body weight "
                     f"{bw_n:.0f} N ({gt['mass_kg']:.1f} kg); plate stances L "
                     f"{len(per_foot[0])}, R {len(per_foot[1])}")
        for label in labels:
            ours = vertical_bw(rows[label], gt)
            for foot in range(2):
                for a, b in per_foot[foot]:
                    m = stance_metrics(plate[:, foot], ours[:, foot], a, b, gt["fps"], bw_n)
                    if m is not None:
                        m["foot"] = FOOT_NAME[foot]
                        stance_rows[label][clip].append(m)
            frame_series[label][clip] = (plate, ours, stance_mask)

    lines = [f"# OpenCap walking trials — vertical GRF gait metrics in % BW "
             f"({len(clips)} clips, runs {', '.join(f'`{r}`' for r in args.runs)})", "",
             *notes, "",
             "Stance rows are means over plate-defined stances (both feet); errors are "
             "signed ours − plates. Impulse in BW·s. Ours sampled at its predicted rows only.",
             ""]
    keep = [c for c in clips if c != "subject4_walking1_cam1"]
    for title, subset in (("Per stance — pooled over all walking clips", clips),
                          ("Per stance — without subject4_walking1", keep)):
        block = {label: aggregate([r for c in subset for r in stance_rows[label][c]])
                 for label in labels}
        lines += [f"## {title}", "", *table(block, STANCE_COLUMNS, "method"), ""]
    for title, subset in (("Per frame — pooled over all walking clips", clips),
                          ("Per frame — without subject4_walking1", keep)):
        block = {}
        for label in labels:
            plate = np.concatenate([frame_series[label][c][0] for c in subset])
            ours = np.concatenate([frame_series[label][c][1] for c in subset])
            mask = np.concatenate([frame_series[label][c][2] for c in subset])
            block[label] = frame_metrics(plate, ours, mask)
        lines += [f"## {title}", "", *table(block, FRAME_COLUMNS, "method"), ""]
    for clip in clips:
        block_s = {label: aggregate(stance_rows[label][clip]) for label in labels}
        block_f = {label: frame_metrics(*frame_series[label][clip]) for label in labels}
        lines += [f"## {clip}", "", *table(block_s, STANCE_COLUMNS, "method"), "",
                  *table(block_f, FRAME_COLUMNS, "method"), ""]
        for label in labels:
            for r in stance_rows[label][clip]:
                lines.append(f"- {label} {r['foot']}: peak1 {r['peak1 plate']:.0f}/"
                             f"{r['peak1 ours']:.0f}, valley {r['valley plate']:.0f}/"
                             f"{r['valley ours']:.0f}, peak2 {r['peak2 plate']:.0f}/"
                             f"{r['peak2 ours']:.0f} %BW, stance {r['stance s plate']:.2f}/"
                             f"{r['stance s ours']:.2f} s")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    return 0


def plate_stances(plate: np.ndarray, fps: float, bw_n: float) -> list[tuple[int, int]]:
    """Plate stances of one foot (% BW series): loaded runs >= MIN_STANCE_S that start and
    end on watched frames (a run touching the clip edge or an unseen frame is dropped)."""
    seen = np.isfinite(plate)
    loaded = seen & (np.nan_to_num(plate) > 100.0 * LOADED_N / bw_n)
    out = []
    for start, end in runs_of(loaded):
        if end - start < MIN_STANCE_S * fps:
            continue
        if start == 0 or end == len(plate) or not seen[start - 1] or not seen[end]:
            continue
        out.append((start, end))
    return out


if __name__ == "__main__":
    sys.exit(main())
