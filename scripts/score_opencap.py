"""Score our prediction dumps on the OpenCap LabValidation force plates.

Twelve clips of ``LabValidation_withVideos`` (``subject{2,3,4}_{DJ1,squats1,STS1,walking1}
_cam1``) are the only place outside the climbing rig where a whole-body force is MEASURED
on the same single-view pixels our model reads: two laboratory force plates, filtered and
already assigned per foot by the dataset, carried into the reconstruction's world by the
dataset's own calibration (``human_optim/sensor_forces.npz``, written by BVR's
``scripts/opencap_sensor_forces.py`` — no fitted parameter, ``gravity_check_deg`` 0.7-1.9°).
Nothing in our model or in BVR's solve ever sees a plate.

Rows are ``optimisation`` (BVR's own solve, ``human_optim/kindyn_1.npz``'s
``contact_forces_world`` summed per limb — feet = ``left_ankle`` 7 + ``left_foot`` 10 and
``right_ankle`` 8 + ``right_foot`` 11, hands = the wrists 20 / 21) and every run name given
on the command line, whose dumps ``scripts/predict_reconstruction.py`` wrote into
``<clip>/predictions/<run>/``. A dump's 35 contact-frame slots fold onto the six kindyn
groups (forces by VECTOR SUM :meth:`~model.contact_frames.ContactSet.fold_sum`,
probabilities by MAX) and those onto the four plate limbs the way the boards do — hand =
hand group, foot = foot (toe) group + ankle (heel) group.

Units: a dump's forces are BODY WEIGHTS, so newtons need a body weight. The PRIMARY reading
uses the subject's MEASURED mass (``sensor_forces.npz::mass_kg``, 62.6-78.2 kg) × 9.81; the
``vert tot MAE (recon) N`` column repeats the total vertical MAE with the RECONSTRUCTED body
mass (``kindyn_1.npz::total_mass``, 57.0-62.3 kg) instead — a sensitivity check on the mass,
not a correction, and ``nan`` for the optimisation row, whose forces are already newtons
solved with that reconstructed mass.

Vertical is the component along UP = ``-gravity_world`` of the clip's ``kindyn_1.npz``
(within ``gravity_check_deg`` of the mocap floor's own up axis); the plates read POSITIVE UP
on a standing subject, which the header's static check prints.

Scored rows: a limb-frame counts only where the row was PREDICTED (the dump's ``valid_mask``
— stride 2 leaves every other video frame NaN; the optimisation's ``valid_mask`` covers
every frame, so its rows are ~2× ours) AND the plates were watching that limb
(``captured``, false during a drop jump's box phase or when a foot stands off a plate). A
plate frame that is NaN drops out of every number; it never scores as zero. Row ``i`` of
every array is video frame ``i`` (checked: the dumps' ``frame_indices`` run ``0..N-1`` and
the plate, body and dump arrays all have the same length).

Columns (per clip, POOLED over frames, and a MEAN OF CLIPS row that weights clips equally):

* **frames** — scored rows (frames with at least one scored foot).
* **vert L / vert R MAE N** — mean absolute error of that foot's vertical force.
* **vert tot MAE N** — the same for the two feet summed as vectors, on the frames where
  BOTH feet are scored. This is the whole-body support number.
* **vert tot MAE (recon) N** — the mass sensitivity check described above.
* **|F| L / |F| R MAE N** — mean ``‖ours − plates‖`` of the FULL 3D VECTOR per foot.
* **corr tot vert** — Pearson correlation of the TOTAL vertical force over time (timing and
  shape, immune to a constant scale error). NaN where fewer than three frames are scored.
* **angle deg** — mean angle between the two foot-force vectors on the limb-frames where
  both read above :data:`CONTACT_N` (50 N).
* **share pp** — the load split between the two feet in percentage points (each foot's
  share of the frame's total ``|F|``, ``|Δ|`` averaged), over the frames where the plate
  total clears :data:`CONTACT_N`. Below that there is no measured split to compare against.
* **phantom N** — mean of OUR vertical force on a foot the plates WERE watching and found
  under :data:`LOADED_N` (20 N): support we invented. It never enters ``share pp``.
* **hand |F| N / hand max N** — mean and max size of our two hand forces. The plates measure
  the hands as exactly zero in all three trial types, so this is pure phantom support.
* **F1 / P / R** — contact of each foot against plate-loaded (that foot's plate vertical >
  :data:`LOADED_N`), pooled over the two feet and all scored frames. Ours is the folded
  slot probability at ``--threshold``; the optimisation row uses its own solved contact
  labels (``kindyn_1.npz::joint_contact`` on joints 7/10 and 8/11) — a DIFFERENT quantity
  from a probability threshold, so the two are not a like-for-like comparison.
* **onset Δ fr** — video frames between the plates' and our own total vertical force first
  crossing :data:`ONSET_N` (50 N), both searched from the first frame the plates watched
  both feet. The synchronisation evidence; it only exists on the drop jumps, where the box
  phase leaves the plates unloaded, and is blank (``—``) where the clip starts loaded. Our
  side is searched over PREDICTED rows only, so with stride 2 its resolution is 2 frames.

Differences from BVR's own ``scripts/diagnostics/compare_opencap_forces.py``, whose
definitions this mirrors wherever they exist (``|F| MAE``, ``Fy(up) MAE``, ``corr``,
``angle``, ``share err``, ``phantom``, ``hand |F|``, ``onset Δ``):

1. BVR rotates both sides back into the OpenSim mocap ground frame (``R_world_from_mocap``)
   and calls ``y`` up; we stay in the reconstruction world and project on ``-gravity_world``.
   The two up axes agree to ``gravity_check_deg`` (0.7-1.9°) and every other column here is
   rotation-invariant.
2. BVR's ``Fy(up) MAE`` pools the two feet into one number; we report the two feet
   separately plus the vector total, which BVR reports as ``total err`` (a 3D norm, not a
   vertical MAE).
3. BVR's ``corr`` is the correlation of the per-foot force SIZE over all foot-frames; ours
   is the correlation of the total VERTICAL force over time.
4. BVR's ``Fy mass-sc`` multiplies its newtons by ``mass_gt / mass_ours``; our primary
   column is already in the measured mass and the ``(recon)`` column is the reciprocal
   check.
5. BVR has no contact columns and no per-run comparison; it scores one solve directory.
6. BVR also prints ``cop gap``, ``lin/ang resid``, ``mass ours/gt`` and a whole-clip ``lag``
   scan; none of those are here (``cop gap`` and the residuals price its solve, not a dump).

    .venv/bin/python scripts/score_opencap.py --runs bedlam_frames35_ep4 climbing_frames35 \\
        --out output_7/logs/opencap_20260920.md --plots output_7/logs/opencap_plots_20260920/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
import numpy as np                                                   # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from model.contact_frames import contact_set                         # noqa: E402

#: The processed OpenCap tree written by BetterVideoReconstruction.
PROCESSED = Path("/home/rikhat.akizhanov/better/data/LabValidation_withVideos/processed")
#: Standard gravity, the constant every body-weight unit in this repository uses.
GRAVITY = 9.81
#: A plate reading below this is a measured "no load" (BVR's ``opencap_plates.LOADED_N``).
LOADED_N = 20.0
#: Both sides above this: the frames where a direction or a share means anything.
CONTACT_N = 50.0
#: The total vertical force must clear this for the onset check to call it a landing.
ONSET_N = 50.0
#: Plate channels, in ``sensor_forces.npz::limbs`` order.
LIMBS = ("left_hand", "right_hand", "left_foot", "right_foot")
#: Kindyn groups making up each plate channel: hand = hand, foot = toe group + heel group.
GROUP_LIMBS = ((0,), (1,), (2, 4), (3, 5))
#: 52-joint members of each plate channel in BVR's solve (wrists; ankle + foot per side).
OPTIM_JOINTS = ((20,), (21,), (7, 10), (8, 11))
#: Columns of every table, in order.
COLUMNS = ("frames", "vert L MAE N", "vert R MAE N", "vert tot MAE N",
           "vert tot MAE (recon) N", "|F| L MAE N", "|F| R MAE N", "corr tot vert",
           "angle deg", "share pp", "phantom N", "hand |F| N", "hand max N",
           "F1", "P", "R", "onset Δ fr")
#: Columns printed to three decimals; the rest to one (newtons) or as an integer.
_RATES = frozenset(("corr tot vert", "F1", "P", "R"))
#: Quantities that cannot be formed by concatenating clips (one unbroken time axis needed).
PER_CLIP_ONLY = ("onset Δ fr",)
#: Okabe-Ito colours for the runs, after the plates (black) and the optimisation (grey).
RUN_COLOURS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")


# ------------------------------------------------------------------ the clip's own data

def clip_names(source: Path, runs: list[str]) -> list[str]:
    """Every clip under ``<source>/out`` that has plates and a dump of every run."""
    out = source / "out"
    return sorted(
        c.name for c in out.iterdir()
        if c.is_dir() and not c.name.startswith("_")
        and (c / "human_optim" / "sensor_forces.npz").is_file()
        and all((c / "predictions" / run / "forces_sup.npz").is_file() for run in runs))


def plates(clip_dir: Path) -> dict:
    """The measured side of one clip: forces, what the plates saw, the masses, up."""
    sensor = np.load(clip_dir / "human_optim" / "sensor_forces.npz", allow_pickle=True)
    kindyn = np.load(clip_dir / "human_optim" / "kindyn_1.npz", allow_pickle=True)
    channels = [str(x) for x in sensor["limbs"]]
    if tuple(channels) != LIMBS:
        raise ValueError(f"{clip_dir.name}: plate channels are {channels}, expected {LIMBS}")
    gravity = np.asarray(kindyn["gravity_world"], np.float64).reshape(3)
    up = -gravity / np.linalg.norm(gravity)
    force = np.asarray(sensor["reaction_world"], np.float64)
    return {
        "force": force,                                        # (N, 4, 3) newtons, our world
        "captured": np.asarray(sensor["captured"], bool),      # (N, 4)
        "up": up,
        "n": len(force),
        "fps": float(np.asarray(sensor["video_fps"], np.float64)),
        "mass_kg": float(np.asarray(sensor["mass_kg"], np.float64)),
        "recon_mass_kg": float(np.asarray(kindyn["total_mass"], np.float64)[0]),
        "gravity_check_deg": float(np.asarray(sensor["gravity_check_deg"], np.float64)),
    }


def optimisation_row(clip_dir: Path, gt: dict) -> dict:
    """BVR's own solve: its limb forces (already newtons) and its own contact labels."""
    kindyn = np.load(clip_dir / "human_optim" / "kindyn_1.npz", allow_pickle=True)
    n = gt["n"]
    solved = np.asarray(kindyn["contact_forces_world"][0], np.float64)[:n]
    contact = np.asarray(kindyn["joint_contact"][0], bool)[:n]
    return {
        "force": np.stack([solved[:, list(j)].sum(1) for j in OPTIM_JOINTS], 1),
        "force_recon": None,                    # already solved with the reconstructed mass
        "contact": np.stack([contact[:, list(j)].any(1) for j in OPTIM_JOINTS[2:]], 1),
        "rows": np.asarray(kindyn["valid_mask"][0], bool)[:n],
    }


def prediction_row(pred_dir: Path, gt: dict, threshold: float) -> dict:
    """One of our dumps: its 35 slots folded onto the four plate limbs, in newtons."""
    dump = np.load(pred_dir / "forces_sup.npz", allow_pickle=True)
    slots = contact_set(str(dump["contact_set"]))
    n = gt["n"]
    forces = np.asarray(dump["forces_world"][0], np.float64)[:n]           # (n, 35, 3) bw
    probs = np.asarray(dump["contact_probs"][0], np.float64)[:n]           # (n, 35)
    rows = (np.asarray(dump["valid_mask"][0], bool)[:n]
            & np.isfinite(forces).all((-1, -2)))
    group_force = slots.fold_sum(np.nan_to_num(forces))                    # (n, 6, 3) bw
    group_prob = slots.fold_max(np.nan_to_num(probs))                      # (n, 6)
    limb_bw = np.stack([group_force[:, list(g)].sum(1) for g in GROUP_LIMBS], 1)
    limb_bw = np.where(rows[:, None, None], limb_bw, np.nan)
    contact = np.stack([(group_prob[:, list(g)] >= threshold).any(1)
                        for g in GROUP_LIMBS[2:]], 1)
    return {
        "force": limb_bw * (gt["mass_kg"] * GRAVITY),
        "force_recon": limb_bw * (gt["recon_mass_kg"] * GRAVITY),
        "contact": contact,
        "rows": rows,
    }


# ------------------------------------------------------------------ one scored series

def series(row: dict, gt: dict) -> dict:
    """Everything one method contributes on one clip, masked to the scored limb-frames.

    A foot-frame is scored where the method predicted the row and the plates were watching
    that foot; everything else is NaN (``False`` for the masks) and drops out of every
    reduction below. The vertical components are projections on ``up``.
    """
    up, feet = gt["up"], [2, 3]
    scored = gt["captured"][:, feet] & row["rows"][:, None] & np.isfinite(
        gt["force"][:, feet]).all(-1)
    measured = np.where(scored[..., None], gt["force"][:, feet], np.nan)
    ours = np.where(scored[..., None], row["force"][:, feet], np.nan)
    recon = (np.where(scored[..., None], row["force_recon"][:, feet], np.nan)
             if row["force_recon"] is not None else np.full_like(ours, np.nan))
    return {
        "measured": measured, "ours": ours, "recon": recon,
        "measured_up": measured @ up, "ours_up": ours @ up, "recon_up": recon @ up,
        "both": scored.all(1),
        "scored": scored,
        "contact": np.where(scored, row["contact"], False),
        "loaded": np.where(scored, np.nan_to_num(measured @ up) > LOADED_N, False),
        "quiet": scored & (np.nan_to_num(measured @ up) < LOADED_N),
        "hands": np.where(row["rows"][:, None, None], row["force"][:, :2], np.nan),
        "fps": gt["fps"],
    }


def onset_delay(clip: dict) -> float:
    """Frames between the plates' first loaded frame and ours, or NaN when undefined.

    Both onsets are searched from the first frame the plates watched BOTH feet, so a drop
    jump's unmeasured box phase does not count as unloaded; NaN where the clip is already
    loaded there (no onset to time) or where either side never crosses.
    """
    watching = clip["scored"].all(1)
    if not watching.any():
        return float("nan")
    start = int(np.argmax(watching))
    plate_up = np.nan_to_num(clip["measured_up"]).sum(1) > ONSET_N
    ours_up = np.nan_to_num(clip["ours_up"]).sum(1) > ONSET_N
    if plate_up[start] or not plate_up[start:].any() or not ours_up[start:].any():
        return float("nan")
    return float(np.argmax(ours_up[start:]) - np.argmax(plate_up[start:]))


def metrics(clip: dict) -> dict[str, float]:
    """Every column of one table row, from one clip or a pooled stack of clips."""
    measured, ours = clip["measured"], clip["ours"]
    size_m, size_o = np.linalg.norm(measured, axis=-1), np.linalg.norm(ours, axis=-1)
    both = clip["both"]
    total_gap = np.abs(np.nansum(clip["ours_up"][both], 1) - np.nansum(
        clip["measured_up"][both], 1))
    has_recon = np.isfinite(clip["recon_up"]).any()      # the optimisation carries none
    total_recon = (np.abs(np.nansum(clip["recon_up"][both], 1)
                          - np.nansum(clip["measured_up"][both], 1))
                   if has_recon else np.full(int(both.sum()), np.nan))
    pair = (np.isfinite(clip["ours_up"][both]).all(1)
            & np.isfinite(clip["measured_up"][both]).all(1))
    ours_tot = clip["ours_up"][both][pair].sum(1)
    plate_tot = clip["measured_up"][both][pair].sum(1)
    loud = (size_m > CONTACT_N) & (size_o > CONTACT_N)              # NaN compares False
    cos = np.clip((measured * ours).sum(-1)[loud] / (size_m[loud] * size_o[loud]), -1.0, 1.0)
    split = both & (np.linalg.norm(np.nan_to_num(measured).sum(1), axis=-1) > CONTACT_N)
    share = lambda f: f / np.maximum(f.sum(1, keepdims=True), 1e-6)  # noqa: E731
    hand_size = np.linalg.norm(clip["hands"], axis=-1)
    got, want = clip["contact"][clip["scored"]], clip["loaded"][clip["scored"]]
    tp, fp, fn = int((got & want).sum()), int((got & ~want).sum()), int((~got & want).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "frames": float(clip["scored"].any(1).sum()),
        "vert L MAE N": float(np.nanmean(np.abs(clip["ours_up"][:, 0]
                                                - clip["measured_up"][:, 0]))),
        "vert R MAE N": float(np.nanmean(np.abs(clip["ours_up"][:, 1]
                                                - clip["measured_up"][:, 1]))),
        "vert tot MAE N": float(np.nanmean(total_gap)) if both.any() else float("nan"),
        "vert tot MAE (recon) N": (float(np.nanmean(total_recon))
                                   if both.any() and np.isfinite(total_recon).any()
                                   else float("nan")),
        "|F| L MAE N": float(np.nanmean(np.linalg.norm(ours[:, 0] - measured[:, 0], axis=-1))),
        "|F| R MAE N": float(np.nanmean(np.linalg.norm(ours[:, 1] - measured[:, 1], axis=-1))),
        "corr tot vert": (float(np.corrcoef(ours_tot, plate_tot)[0, 1])
                          if pair.sum() > 2 and ours_tot.std() > 0 and plate_tot.std() > 0
                          else float("nan")),
        "angle deg": (float(np.degrees(np.arccos(cos)).mean()) if cos.size else float("nan")),
        "share pp": (100.0 * float(np.nanmean(np.abs(share(size_o[split])
                                                     - share(size_m[split]))))
                     if split.any() else float("nan")),
        "phantom N": (float(np.nanmean(clip["ours_up"][clip["quiet"]]))
                      if clip["quiet"].any() else float("nan")),
        "hand |F| N": (float(np.nanmean(hand_size)) if np.isfinite(hand_size).any()
                       else float("nan")),
        "hand max N": (float(np.nanmax(hand_size)) if np.isfinite(hand_size).any()
                       else float("nan")),
        "F1": (2 * precision * recall / max(precision + recall, 1e-9)
               if tp + fp + fn else float("nan")),
        "P": precision if tp + fp + fn else float("nan"),
        "R": recall if tp + fp + fn else float("nan"),
        "onset Δ fr": onset_delay(clip),
    }


def pooled(clips: list[dict]) -> dict:
    """Concatenate the per-clip series along time, so the pooled row pools FRAMES."""
    keys = ("measured", "ours", "recon", "measured_up", "ours_up", "recon_up", "both",
            "scored", "contact", "loaded", "quiet", "hands")
    return {**{k: np.concatenate([c[k] for c in clips]) for k in keys},
            "fps": clips[0]["fps"]}


# ------------------------------------------------------------------ plots

def gapped(values: np.ndarray, rows: np.ndarray, stride: int) -> np.ndarray:
    """``values`` with NaN kept wherever a gap is longer than the prediction stride.

    Consecutive predicted rows sit ``stride`` frames apart by construction, so joining them
    is drawing the prediction, not interpolating it; a longer gap (an untracked run, a
    plate that stopped watching) stays a break in the line.
    """
    out = np.full(len(values), np.nan)
    out[rows] = values[rows]
    index = np.flatnonzero(rows)
    for start, end in zip(index[:-1], index[1:]):
        if end - start <= stride:
            out[start:end + 1] = np.interp(np.arange(start, end + 1), (start, end),
                                           (out[start], out[end]))
    return out


def plot_clip(clip: str, gt: dict, rows: dict[str, dict], stride: dict[str, int],
              path: Path) -> None:
    """Vertical GRF vs time for one clip: left foot, right foot, total, ours vs the plates.

    Every series is masked to the frames the plates watched that foot — the frames the
    table scores — and each method is drawn only on the rows it predicted (dots), joined
    across its own stride but never across a longer gap (:func:`gapped`).
    """
    up, feet = gt["up"], [2, 3]
    time = np.arange(gt["n"]) / gt["fps"]
    seen = gt["captured"][:, feet] & np.isfinite(gt["force"][:, feet]).all(-1)
    plate_up = np.where(seen, gt["force"][:, feet] @ up, np.nan)
    panels = [("left foot", plate_up[:, 0]), ("right foot", plate_up[:, 1]),
              ("total", np.where(seen.all(1), np.nansum(plate_up, 1), np.nan))]
    colours = dict(zip([k for k in rows if k != "optimisation"], RUN_COLOURS))
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.0), sharex=True)
    for panel, (axis, (title, measured)) in enumerate(zip(axes, panels)):
        axis.axhline(0.0, color="0.85", lw=0.8)
        axis.plot(time, measured, color="black", lw=2.2, label="plates")
        for label, row in rows.items():
            ours = np.where(seen, row["force"][:, feet] @ up, np.nan)
            values = (np.where(seen.all(1), np.nansum(ours, 1), np.nan) if title == "total"
                      else ours[:, panel])
            line = gapped(values, row["rows"] & np.isfinite(values), stride[label])
            style = dict(color="0.55", lw=1.4, ls="--") if label == "optimisation" else dict(
                color=colours[label], lw=1.4)
            axis.plot(time, line, label=label, **style)
            axis.plot(time[row["rows"]], values[row["rows"]], ".", ms=2.6,
                      color=style["color"])
        axis.set_ylabel(f"{title}\nF up [N]", fontsize=9)
        axis.grid(True, color="0.92", lw=0.6)
        axis.set_axisbelow(True)
    axes[-1].set_xlabel("time [s]", fontsize=9)
    axes[0].legend(loc="upper right", fontsize=8, frameon=False, ncol=len(rows) + 1)
    fig.suptitle(f"{clip} — vertical ground reaction force "
                 f"(measured mass {gt['mass_kg']:.1f} kg = {gt['mass_kg'] * GRAVITY:.0f} N, "
                 f"reconstructed {gt['recon_mass_kg']:.1f} kg)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ table rendering

def cell(key: str, value: float) -> str:
    """One markdown cell: ``—`` where the quantity has no value on that row."""
    if not np.isfinite(value):
        return "—"
    if key == "frames":
        return f"{value:.0f}"
    if key == "onset Δ fr":
        return f"{value:+.0f}"
    return f"{value:.3f}" if key in _RATES else f"{value:.1f}"


def table(rows: dict[str, dict[str, float]], first: str) -> list[str]:
    """Markdown table of ``{row name: metrics}``, one column per entry of :data:`COLUMNS`."""
    out = ["| " + first + " | " + " | ".join(c.replace("|", "\\|") for c in COLUMNS) + " |",
           "|" + "---|" * (len(COLUMNS) + 1)]
    for name, values in rows.items():
        out.append(f"| {name} | " + " | ".join(cell(c, values[c]) for c in COLUMNS) + " |")
    return out


# ------------------------------------------------------------------ driver

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True,
                        help="run names whose dumps sit in <clip>/predictions/<run>/")
    parser.add_argument("--source", type=Path, default=PROCESSED,
                        help="the processed LabValidation root")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="contact probability threshold of every run")
    parser.add_argument("--out", type=Path, default=None, help="markdown file to write")
    parser.add_argument("--plots", type=Path, default=None,
                        help="directory for one vertical-GRF PNG per clip")
    args = parser.parse_args(argv)

    clips = clip_names(args.source, args.runs)
    if not clips:
        raise SystemExit(f"no clip under {args.source}/out has plates and every run's dump")
    labels = ["optimisation", *args.runs]
    per_clip: dict[str, dict[str, dict[str, float]]] = {label: {} for label in labels}
    stacks: dict[str, list[dict]] = {label: [] for label in labels}
    header_lines = []

    for clip in clips:
        clip_dir = args.source / "out" / clip
        gt = plates(clip_dir)
        rows = {"optimisation": optimisation_row(clip_dir, gt)}
        strides = {"optimisation": 1}
        for run in args.runs:
            pred_dir = clip_dir / "predictions" / run
            rows[run] = prediction_row(pred_dir, gt, args.threshold)
            strides[run] = int(np.load(pred_dir / "forces_sup.npz",
                                       allow_pickle=True)["stride"])
        for label in labels:
            scored = series(rows[label], gt)
            stacks[label].append(scored)
            per_clip[label][clip] = metrics(scored)
        still = gt["captured"][:, 2:].all(1) & np.isfinite(gt["force"][:, 2:]).all((-1, -2))
        plate_up = float(np.nanmedian((gt["force"][still][:, 2:] @ gt["up"]).sum(1))) \
            if still.any() else float("nan")
        header_lines.append(
            f"- `{clip}`: {gt['n']} frames at {gt['fps']:.0f} fps, "
            f"{int(still.sum())} both-feet frames, median plate total up "
            f"{plate_up:+.0f} N vs body weight {gt['mass_kg'] * GRAVITY:.0f} N "
            f"(measured {gt['mass_kg']:.1f} kg, reconstructed {gt['recon_mass_kg']:.1f} kg), "
            f"gravity check {gt['gravity_check_deg']:.1f}°, predicted rows "
            + ", ".join(f"{label} {int(rows[label]['rows'].sum())}" for label in labels))
        print(header_lines[-1], flush=True)
        if args.plots is not None:
            args.plots.mkdir(parents=True, exist_ok=True)
            plot_clip(clip, gt, rows, strides, args.plots / f"{clip}.png")

    pooled_rows, mean_rows = {}, {}
    for label in labels:
        pooled_rows[label] = metrics(pooled(stacks[label]))
        for key in PER_CLIP_ONLY:                 # not formable by concatenating clips
            pooled_rows[label][key] = float("nan")
        values = per_clip[label]
        mean_rows[label] = {
            k: (float(np.nanmean([v[k] for v in values.values()]))
                if np.isfinite([v[k] for v in values.values()]).any() else float("nan"))
            for k in COLUMNS}

    lines = [f"# OpenCap LabValidation — vertical GRF vs the force plates "
             f"({len(clips)} clips, runs {', '.join(f'`{r}`' for r in args.runs)}, "
             f"contact threshold {args.threshold})", "",
             "## Pooled over frames", "", *table(pooled_rows, "method"), "",
             "## Mean of clips", "", *table(mean_rows, "method"), ""]
    for label in labels:
        lines += [f"## Per clip — `{label}`", "", *table(per_clip[label], "clip"), ""]
    lines += ["## Clips", "", *header_lines, "",
              "Newtons from body weights use the subject's MEASURED mass; "
              "`vert tot MAE (recon) N` repeats the total with the reconstructed body mass "
              "(nan for the optimisation, whose newtons are already solved with it). "
              "Vertical is the component along `-gravity_world`. Only rows the method "
              "predicted and the plates saw are scored; the optimisation covers every "
              "frame and our dumps every second one (stride 2). See the module docstring "
              "for every column and for the differences from BVR's "
              "`compare_opencap_forces.py`.", ""]
    text = "\n".join(lines)
    print("\n" + text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    if args.plots is not None:
        print(f"wrote {len(clips)} PNG(s) to {args.plots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
