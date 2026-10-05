"""The OpenCap LabValidation GRF tables for the paper, from BVR's plate-format scorer.

Reads the ``metrics.npz`` that ``BetterVideoReconstruction/scripts/diagnostics/
compare_opencap_plates.py`` writes per run under ``processed/out/_plates_<run>_clips_natural/``
and prints two tables in the biomechanics form (MAE of the vertical / anterior-posterior /
medio-lateral GRF in % body weight, per foot, aggregated clip → subject → mean over subjects):

1. **Walking**, over STANCE frames (plate vertical > 5 % BW), 6 Hz low-pass — the protocol of
   OpenCap Monocular (Gilon et al. 2026), whose published rows are appended.
2. **Every activity** over WHOLE trials, OpenCap's per-activity cutoffs — the protocol of
   OpenCap 2023 (Uhlrich et al.), whose published rows are appended.

Every method appears twice: newtons from the method's OWN body mass (the reconstructed SMPL-X
mass for ours, PhysPT's SMPL mass, Li et al.'s fixed 74.3 kg table) scored against the
measured body weight (``own mass``), and the same forces read as fractions of the method's own
body weight against the plates' fractions of the measured one — a post-hoc rescale by
measured / own mass, which for a body-weight-output model is the reading with the subject's
MEASURED mass (``measured mass (rescale)``). The optimisation run solved with the measured
mass inside the solve is its own row. The literature numbers are as given by the user / the papers and are not
re-derived here.

Usage::

    .venv/bin/python scripts/opencap_paper_table.py --out output_7/logs/opencap_paper_tables_<date>.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

OUT = Path("/home/rikhat.akizhanov/better/data/LabValidation_withVideos/processed/out")

#: (label, compare directory, whether the solve itself consumed the measured mass).
RUNS = [
    ("Li et al. 2019 (our SAM-3D init, Sapiens 2D joints, THEIR contact recogniser)",
     "_plates_method_estmf_rec_clips_natural", False),
    ("PhysPT 2024 (our SAM-3D init; yaw + translation registered to mocap markers)", "_plates_method_physpt_clips_natural", False),
    ("Ours, optimisation (edgev)", "_plates_floorplane_edgev_clips_natural", False),
    ("Ours, optimisation (edgev, measured mass INSIDE the solve)", "_plates_floorplane_massedgev_clips_natural", True),
    ("Ours, learned, ClimbingVideos (frames35)", "_plates_method_climbing_frames35_clips_natural", False),
    ("Ours, learned, BEDLAM2 (frames35)", "_plates_method_bedlam_frames35_clips_natural", False),
    ("Ours, learned, BEDLAM2 then ClimbingVideos + BEDLAM2 50/50 (frames35)",
     "_plates_method_joint_frames35_clips_natural", False),
]

ACTIVITIES = ["walking", "squats", "STS", "DJ", "ALL (mean of activities)"]

#: OpenCap Monocular 2026, walking, stance phase, %BW (V / AP / ML); user-supplied.
WALKING_LITERATURE = [
    ("OpenCap 2023 (two cameras, physics simulation)", 12.2, 3.1, 1.2),
    ("GaitDynamics 2026 (mocap kinematics, learned)", 3.9, 1.3, 0.6),
    ("OpenCap Monocular 2026 (one camera, learned)", 9.7, 4.4, 1.7),
]

#: OpenCap 2023 (two cameras), whole trials, %BW per activity; from the paper's supplement.
OPENCAP_2023 = {
    "walking": (8.21, 2.06, 1.09), "squats": (6.44, 1.32, 5.66), "STS": (5.68, 1.90, 3.19),
    "DJ": (25.15, 8.91, 5.31), "ALL (mean of activities)": (11.37, 3.55, 3.81),
}


def load(directory: str) -> dict | None:
    path = OUT / directory / "metrics.npz"
    return np.load(path, allow_pickle=True)["rows"].item() if path.exists() else None


def cell(value: float | None) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{value:.1f}"


def triple(row: dict, keys: tuple[str, str, str]) -> str:
    return " | ".join(cell(row.get(k)) for k in keys)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None, help="write the markdown here too")
    args = ap.parse_args(argv)

    stance = {"own mass": ("MAE V [%BW]", "MAE AP [%BW]", "MAE ML [%BW]"),
              "measured mass (rescale)": ("MAE V own-BW [%BW]", "MAE AP own-BW [%BW]", "MAE ML own-BW [%BW]")}
    whole = {"own mass": ("MAE V all [%BW]", "MAE AP all [%BW]", "MAE ML all [%BW]"),
             "measured mass (rescale)": ("MAE V all own-BW [%BW]", "MAE AP all own-BW [%BW]",
                         "MAE ML all own-BW [%BW]")}

    lines = ["# OpenCap LabValidation — GRF MAE, %BW", "",
             "## Walking, stance frames (OpenCap Monocular protocol)", "",
             "| method | mass | clips | subj | V | AP | ML |", "|---|---|---|---|---|---|---|"]
    for label, directory, solved_with_gt in RUNS:
        rows = load(directory)
        if rows is None:
            lines.append(f"| {label} | | | | (not scored) | | |")
            continue
        walking = rows["walking"]
        for mass, keys in stance.items():
            if solved_with_gt and mass == "own mass":
                continue
            lines.append(f"| {label} | {mass} | {walking['clips']:.0f} | {walking['subjects']:.0f} "
                         f"| {triple(walking, keys)} |")
    for label, v, ap, ml in WALKING_LITERATURE:
        lines.append(f"| {label} | measured | — | 10 | {v:.1f} | {ap:.1f} | {ml:.1f} |")

    lines += ["", "## Every activity, whole trials (OpenCap 2023 protocol)", "",
              "| method | mass | " + " | ".join(f"{a} V / AP / ML" for a in ACTIVITIES) + " |",
              "|---|---|" + "---|" * len(ACTIVITIES)]
    for label, directory, solved_with_gt in RUNS:
        rows = load(directory)
        if rows is None:
            continue
        for mass, keys in whole.items():
            if solved_with_gt and mass == "own mass":
                continue
            lines.append(f"| {label} | {mass} | " + " | ".join(
                " / ".join(cell(rows[a].get(k)) for k in keys) if a in rows else "—"
                for a in ACTIVITIES) + " |")
    lines.append("| OpenCap 2023 (two cameras, physics simulation) | measured | " + " | ".join(
        " / ".join(f"{x:.1f}" for x in OPENCAP_2023[a]) for a in ACTIVITIES) + " |")

    lines += ["", "## Stance-frame V / AP / ML per activity (same runs, measured mass by rescale)", "",
              "| method | " + " | ".join(f"{a}" for a in ACTIVITIES) + " |",
              "|---|" + "---|" * len(ACTIVITIES)]
    for label, directory, _ in RUNS:
        rows = load(directory)
        if rows is None:
            continue
        lines.append(f"| {label} | " + " | ".join(
            " / ".join(cell(rows[a].get(k)) for k in stance["measured mass (rescale)"]) if a in rows else "—"
            for a in ACTIVITIES) + " |")

    lines += ["", "Protocol (BVR `compare_opencap_plates.py`): both sides low-pass filtered "
              "(4th-order zero-lag Butterworth, 6 Hz walking / 4 Hz squats and STS / drop jump "
              "unfiltered at 60 Hz); forces in % of the MEASURED body weight; per foot; stance = "
              "that foot's plate vertical above 5 % BW; whole trial = every frame the plates "
              "watched that foot and the method covered; clip → subject → mean over subjects per "
              "activity; ALL = mean of the four activities. `own mass` = the method's native newtons "
              "(its own reconstructed body mass: SMPL-X shape mass for ours, PhysPT's SMPL mass, Li "
              "et al.'s fixed 74.3 kg table) against the measured body weight; `measured mass "
              "(rescale)` = the same forces rescaled post hoc by measured / own mass (for a "
              "body-weight-output model this IS the measured-mass reading; for the physics methods "
              "it is NOT a rerun of their dynamics with that mass). Unmeasured plate rows and "
              "uncovered method rows hold their nearest neighbour before the filter and are masked "
              "afterwards. 64 'natural' clips (walking1-4, squats1, STS1, DJ1-4) of 9 subjects "
              "(subjects 2,3,4,5,7,8,9,10,11); six further natural clips with an uncertain "
              "video-to-mocap sync are excluded, among them all three subject8 walking clips, so "
              "walking has 8 subjects. Baselines: Li et al. runs on our SAM-3D initialisation, "
              "Sapiens 2D joints and ITS OWN contact recogniser (Li et al.'s released CNNs on "
              "joint crops, no measured or optimised contact label enters; on these clips the "
              "recogniser calls a hand in contact on ~70 % of frames and never lifts a foot, and the "
              "estimator lets those phantom hand contacts carry load: 93 N mean upward hand force over "
              "the 64 clips, 17 % of the total upward load, 45 N / 7 % on walking — that load is "
              "missing from its foot forces); PhysPT on our SAM-3D "
              "initialisation, its world registered to the mocap markers by a yaw + translation "
              "fit. Literature rows are quoted, not re-derived: OpenCap Monocular's walking "
              "numbers are stance-phase MAE on 10 subjects × 6 walking trials (incl. trunk-sway "
              "variants), its Figure 5 labels the two horizontal axes the other way round (ML 4.4 "
              "/ AP 1.7 for Monocular, ML 3.1 / AP 1.2 for two-camera) — unresolved; OpenCap "
              "2023's are whole-trial MAE over 10 subjects.",
              ""]
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
