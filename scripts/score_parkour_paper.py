"""Score the LAAS Parkour rig exactly as Li et al. (CVPR'19) and the dataset's evaluator do.

Three numbers per method, all of them the dataset's own (``Parkour-dataset/lib/parkour_evaluator.py``):

* **Procrustes MPJPE** (mm) — a per-frame similarity fit WITH SCALE of the 16 Vicon joints,
  mean over joints and frames, then the unweighted mean over the clips of an action
  (kv / mu / pu / sv) and over all 28 clips. The evaluator scores frames
  ``0 .. num_frames_in_video - 4``, which is exactly the frame range BVR's importer kept,
  so every processed frame is scored. A similarity fit is blind to the world frame, so
  this column is the same under either force convention below.
* **Mean linear force error** (N) and **mean moment error** (N·m) per channel
  (``l_sole, r_sole, l_hand, r_hand``), over the frames the rig CAPTURED (the evaluator
  zeroes the uncaptured ones and divides by the captured count), per clip and then
  averaged over clips. No alignment of any kind: the wrench is expressed at the contact
  joint with axes parallel to the MoCap world, the camera being carried there by the fixed
  rotation ``parkourR_c = [[-1,0,0],[0,0,-1],[0,-1,0]]`` of
  ``Estimating-3D-Motion-Forces/optimizer.py::ExpressLocalForcesInParkourFrame`` — the paper
  assumes a level camera looking along the MoCap ``+y`` and fits nothing.

Newtons use the paper's GENERIC 74.6 kg body ("we assume a generic human physical model of
mass 74.6 kg for all the subjects") for every row: our dumps and PhysPT's store body-weight
units of the reconstructed body, so they are multiplied by ``74.6 * 9.81``; the optimisation
trees store newtons of their own reconstructed mass, so they are divided by that mass's body
weight first.

A channel's linear force is the SUM of that limb's slot forces (a dump's slots and the
optimisation's 52 joints alike fold onto the channel through the 22-joint parent: wrist and
fingers to the hand, foot and ankle to the sole). Its moment is
``sum_slots (anchor - p_ref) x f`` about OUR ankle joint (soles) or OUR wrist joint (hands).
The GT sole moment is the plate's about the ankle, so the sole columns compare like with
like; the GT HAND moment is the instrumented bar's about the SENSOR frame (the dataset only
rotates it), so no method's hand moment is comparable to it — Li's own numbers there are
131 / 134 N·m.

A fourth table carries the climb_wall_3 rig table's force-agreement columns
(``scripts/score_rig.py::force_columns``) read on the same Parkour-frame channel forces, pooled
over every captured channel-frame of the 28 clips: ``angle`` (mean angle between the vectors
where both read above 50 N), ``corr`` (Pearson correlation of the sizes), ``share pp`` (per-frame
load-share error), and the RNEA root-wrench residual of each row's own pose under its own
forces (``RNEA f bw`` / ``RNEA tau bw·m``, ``score_rig``'s reading; the optimisation rows report
the residual their solve stored). ``--sam3d`` adds the raw per-frame SAM 3D Body fit as a
pose-only row.

``--check-li <dir>`` validates the metric functions against Li's own stored per-clip results
(``<clip>.pkl``) by feeding them his stored, already-Parkour-frame arrays (``<clip>_pred.npz``:
``joints_16``, ``forces_parkour``). Run it before trusting any table.

    python scripts/score_parkour_paper.py --check-li \\
        ../data/Estimating-3D-Motion-Forces/validation/results/ours/run/evaluation/ours_fps25_s7
    python scripts/score_parkour_paper.py --runs climbing_frames35 physpt \\
        --out output_7/logs/parkour_paper_20260922.md
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from model.contact_frames import body22_parent, contact_set                       # noqa: E402
from model.physics import RootWrench                                              # noqa: E402
from score_parkour_rig import (CORRESPONDENCE, PARKOUR, clip_rotation, joints16,  # noqa: E402
                               procrustes_mpjpe)
from score_rig import force_columns, predicted_residual                           # noqa: E402
from viewer.bodies import load_body                                               # noqa: E402

#: Camera -> Parkour MoCap world, the paper's fixed rotation (no per-clip fit).
PARKOUR_R_C = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]])
#: The generic body the paper puts every method's forces in newtons with.
GENERIC_MASS_KG = 74.6
GRAVITY = 9.81
#: The force-agreement columns, in table order.
RIG_COLUMNS = ("angle", "corr", "share pp", "RNEA f bw", "RNEA tau bw·m")
#: Li's own body: the sum of his fixed inertia table
#: (``Estimating-3D-Motion-Forces/person_models/full_body_inertia.pkl``), the mass his
#: newtons are solved with — the paper says 74.6, the released model weighs this.
LI_MASS_KG = 74.308
#: PhysPT's outputs, whose own body mass is shape-dependent (its SMPL betas per clip).
PHYSPT = _ROOT.parent / "other" / "PhysPT"
#: The four force channels, in the dataset's order (its GT names l_ankle / r_ankle /
#: l_fingers / r_fingers).
CHANNELS = ("l_sole", "r_sole", "l_hand", "r_hand")
#: Processed ``forces.npz`` limb columns (left_hand, right_hand, left_foot, right_foot)
#: in :data:`CHANNELS` order (only the check mode reads the processed copy).
PROCESSED_TO_CHANNEL = (2, 3, 0, 1)
#: The ground truth the dataset's evaluator reads: the ORIGINAL release, shipped inside
#: Li's estimator. The copy under ``data/Parkour-dataset/`` that the processed corpus was
#: imported from is a reduced one — every reading under 30 N zeroed — and reproduces
#: neither the paper's numbers nor Li's stored ones.
DEFAULT_GT = (_ROOT.parent / "data" / "Estimating-3D-Motion-Forces" / "data"
              / "Parkour-dataset" / "gt_motion_forces")
#: 22-joint parent -> channel, Li's own convention: everything on a leg below the hip
#: (knee, ankle, heel, toes) loads that side's sole channel, and the wrist and every finger
#: load that side's hand channel. Every other parent (head, elbows, shoulders, hips, sacrum,
#: chest, ...) carries no Parkour channel.
CHANNEL_OF_PARENT22 = {4: 0, 7: 0, 10: 0, 5: 1, 8: 1, 11: 1, 20: 2, 21: 3}
#: The joint the moment of each channel is taken about, in the 52-joint SMPL-X.
REFERENCE_JOINT52 = (7, 8, 20, 21)
#: Li's own evaluator results for the rerun on our inputs with his contact recogniser.
DEFAULT_LI_RERUN = (_ROOT.parent / "data" / "Estimating-3D-Motion-Forces" / "validation"
                    / "results" / "ours" / "run" / "evaluation" / "ours_rec_s7")
LI_RERUN_LABEL = ("Li et al. 2019 (rerun: our SAM-3D init, Sapiens 2D, their recogniser, "
                  "25 fps) - own evaluator")
#: ``--check-li`` fails above this difference from Li's own stored numbers.
CHECK_TOLERANCE = 1e-9
#: The optimisation trees, as ``score_rig`` labels them.
OPTIM_LABELS = {"human_optim": "optimisation", "human_optim_scenefree": "optimisation (scene-free)"}
#: Dumps whose run name does not say what the row is.
RUN_LABELS = {"estmf_rec": "estmf_rec (the same Li rerun through our dump round-trip)"}
#: Parkour techniques, by the clip-name prefix ``report_parkour.py`` groups on.
ACTIONS = (("kv", "Kong-vault"), ("mu", "Muscle-up"), ("pu", "Pull-up"), ("sv", "Safety-vault"))
#: The paper's own table, quoted (Table 4/5 of Li et al. 2019).
PAPER_ROWS = {
    "SMPLify (paper)": dict(mpjpe=(121.75, 147.41, 120.48, 169.36, 139.69)),
    "HMR (paper)": dict(mpjpe=(111.36, 140.16, 132.44, 149.64, 135.65)),
    "Li et al. 2019 (paper)": dict(mpjpe=(98.42, 125.21, 119.92, 138.45, 122.11),
                                   force=(144.23, 138.21, 107.91, 113.42),
                                   moment=(23.71, 22.32, 131.13, 134.21)),
}


def physpt_mass(clip: str) -> float:
    """PhysPT's OWN body mass (kg) for one clip: its betas through its own mass model."""
    if clip not in _PHYSPT_MASS:
        sys.path.insert(0, str(PHYSPT))
        import config                                                         # noqa: PLC0415
        from models.smpl_phys import SMPL                                     # noqa: PLC0415

        config.PHYSICS_PATH = str(PHYSPT / "assets" / "data" / "physics.pkl")
        model = SMPL(model_path=str(PHYSPT / "assets" / "data" / "smpl" / "neutral.pkl"))
        betas = torch.as_tensor(np.asarray(
            json.loads((PHYSPT / "parkour" / f"{clip}_sam3d.json").read_text())["pred_beta"],
            np.float32)[:1])
        shaped = torch.tensordot(betas, model.shapedirs, dims=([1], [0])) \
            + model.v_template.unsqueeze(0)
        joints = torch.matmul(shaped.transpose(1, 2), model.regressor).transpose(1, 2)
        _PHYSPT_MASS[clip] = float(np.asarray(model.compute_massinertia(shaped, joints)[0]).sum())
    return _PHYSPT_MASS[clip]


_PHYSPT_MASS: dict[str, float] = {}


# ------------------------------------------------------------------ the dataset's metrics

def channel_errors(predicted: np.ndarray, truth: np.ndarray,
                   captured: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One clip's mean linear-force and moment errors per channel, the evaluator's way.

    :param predicted: ``(n, 4, 6)`` wrenches ``[f, m]`` in the Parkour frame.
    :param truth: ``(n, 4, 6)`` the rig's own wrenches.
    :param captured: ``(n, 4)`` bool — the frames the rig measured that channel on.
    :returns: ``((4,), (4,))`` newtons and newton-metres, NaN for a channel that was
        never captured in this clip.
    """
    difference = np.where(captured[..., None], predicted - truth, 0.0)
    counts = captured.sum(0).astype(np.float64)
    counts = np.where(counts, counts, np.nan)
    errors = [np.linalg.norm(difference[..., part], axis=-1).sum(0) / counts
              for part in (slice(0, 3), slice(3, 6))]
    return errors[0], errors[1]


def pose_error(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, int]:
    """Procrustes MPJPE (mm) over the frames the method covered, and the number it did not."""
    finite = np.isfinite(predicted).all(axis=(1, 2))
    return procrustes_mpjpe(predicted[finite], truth[finite])[0], int((~finite).sum())


# ------------------------------------------------------------------ the rig

def load_truth(gt_dir: Path) -> dict[str, dict]:
    """Every clip's Vicon joints, measured wrenches and captured mask, over the scored window.

    The evaluator scores frames ``0 .. num_frames_in_video - 4``; the shipped GT rows past
    that are dropped here, so every array is exactly the scored window (and the same length
    as the processed corpus and every dump).
    """
    with open(gt_dir / "contact_forces_local.pkl", "rb") as handle:
        forces = pickle.load(handle, encoding="latin-1")
    with open(gt_dir / "joint_3d_positions.pkl", "rb") as handle:
        joints = pickle.load(handle, encoding="latin-1")
    truth = {}
    for clip, entry in forces.items():
        n = len(entry["contact_forces_local"]) - 3
        truth[clip] = {
            "n": n,
            "joints16": np.asarray(joints[clip]["joint_3d_positions"], np.float64)[:n],
            "wrench": np.asarray(entry["contact_forces_local"], np.float64)[:n, :4],
            "captured": ~np.asarray(entry["mask_uncaptured_forces"], bool)[:n, :4],
        }
    return truth


def extrinsics(tree: Path, clip: str, n: int) -> np.ndarray:
    """``(n, 4, 4)`` ``cam_from_world`` of the clip's reconstruction tree."""
    return np.asarray(np.load(tree / clip / "geometry" / "transform.npz")["extrinsics"],
                      np.float64)[:n]


# ------------------------------------------------------------------ the rows

class Fk:
    """The 52-joint BetterHuman SMPL-X every row's 16 Parkour joints are re-posed with."""

    def __init__(self, model_path: str, device: torch.device) -> None:
        self.body = load_body(device, model_path)
        self.device = device
        self.joint_names = [str(name) for name in self.body.structure.joint_names]

    def joints16(self, q: np.ndarray, betas: np.ndarray,
                 ext: np.ndarray | None) -> np.ndarray:
        """``(T, 16, 3)`` world joints of a ``(T, 211)`` q; ``ext`` lifts a camera-frame q."""
        return joints16(q, betas, self.body, ext, self.joint_names, self.device)


def channel_wrench(forces_n: np.ndarray, points: np.ndarray, channel_of_slot: np.ndarray,
                   reference_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fold per-slot forces onto the four channels, summing force and moment.

    :param forces_n: ``(n, S, 3)`` newtons in one frame (the tree's world).
    :param points: ``(n, S, 3)`` where each slot's force acts, same frame.
    :param channel_of_slot: ``(S,)`` channel index per slot, ``-1`` for none.
    :param reference_points: ``(n, 4, 3)`` the joint each channel's moment is taken about.
    :returns: ``((n, 4, 3), (n, 4, 3))`` force and moment.
    """
    n = len(forces_n)
    force = np.zeros((n, 4, 3))
    moment = np.zeros((n, 4, 3))
    for channel in range(4):
        slots = np.flatnonzero(channel_of_slot == channel)
        if not slots.size:
            continue
        force[:, channel] = forces_n[:, slots].sum(1)
        lever = points[:, slots] - reference_points[:, channel, None]
        moment[:, channel] = np.cross(lever, forces_n[:, slots]).sum(1)
    return force, moment


def native_mass_kg(run: str, clip: str, tree_mass: float) -> float:
    """The body mass a row's forces were SOLVED with, which the generic 74.6 kg replaces.

    A dump stores body weights of the tree's reconstructed body (``score_rig``'s convention,
    the divisor ``scripts/estmf_dumps.py`` and ``scripts/physpt_dumps.py`` used), so the
    native newtons are always ``bw * tree_mass * g``; what differs is the mass the METHOD
    itself assumed — its own fixed skeleton for Li, its own shaped SMPL for PhysPT, and, for
    a model that predicts body weights, the reconstructed body itself.
    """
    if run.startswith("estmf"):
        return LI_MASS_KG
    if run == "physpt":
        return physpt_mass(clip)
    return tree_mass


def prediction_row(pred_dir: Path, tree: Path, clip: str, n: int, body: Fk,
                   newton: float, wrench: RootWrench, fps: float) -> dict:
    """One dump: its 16 Parkour joints, its channel wrenches in the tree's world (N), its residual.

    ``newton`` turns the dump's body-weight units into newtons of the generic body.
    Both readings are returned: the raw force sum (the rig table's reading) and the force
    gated by the predicted contact probability. The RNEA residual is ``score_rig``'s: the
    dumped body under the probability-gated dumped forces at the dumped slot points.
    """
    dump = np.load(pred_dir / "smplx.npz", allow_pickle=True)
    covered = np.asarray(dump["covered"][0][:n], bool)
    ext = extrinsics(tree, clip, n)
    q_cam = np.asarray(dump["q_cam"][0][:n], np.float32)
    betas = np.nan_to_num(np.asarray(dump["betas"][0][:n], np.float32))
    pose16 = body.joints16(q_cam, betas, ext.astype(np.float32))
    forces = np.load(pred_dir / "forces_sup.npz", allow_pickle=True)
    slots = contact_set(str(forces["contact_set"]))
    forces_bw = np.nan_to_num(np.asarray(forces["forces_world"][0][:n], np.float64))
    forces_n = forces_bw * newton
    probs = np.nan_to_num(np.asarray(forces["contact_probs"][0][:n], np.float64))
    anchor_cam = np.nan_to_num(np.asarray(forces["anchor_cam"][0][:n], np.float64))
    points = np.einsum("tji,tkj->tki", ext[:, :3, :3], anchor_cam - ext[:, None, :3, 3])
    channel_of_slot = np.asarray([CHANNEL_OF_PARENT22.get(parent, -1)
                                  for parent in slots.parent_joint22], np.int64)
    joints = np.nan_to_num(np.asarray(dump["joints_world"][0][:n], np.float64))
    held = np.where(covered[:, None], np.nan_to_num(q_cam[:, :91]), q_cam[np.flatnonzero(covered)[0], :91])
    res_f, res_t, rows = predicted_residual(
        wrench, tree, clip, {"n": n, "fps": fps, "valid": np.ones(n, bool)}, dump, held, betas,
        forces_bw * probs[..., None], anchor_cam, np.asarray(slots.parent_joint22, np.int64),
        covered)
    row = {"joints16": np.where(covered[:, None, None], pose16, np.nan),
           "dropped_n": float(np.linalg.norm(forces_n[:, channel_of_slot < 0], axis=-1).sum()),
           "total_n": float(np.linalg.norm(forces_n, axis=-1).sum()),
           "rnea": (res_f, res_t), "rnea_rows": rows & covered}
    for suffix, values in (("", forces_n), ("_gated", forces_n * probs[..., None])):
        force, moment = channel_wrench(values, points, channel_of_slot,
                                       joints[:, REFERENCE_JOINT52])
        row[f"force{suffix}"], row[f"moment{suffix}"] = force, moment
    return row


def optimisation_row(tree: Path, clip: str, directory: str, n: int, body: Fk) -> dict:
    """One of Peter's solves: its joints, and its solved forces rescaled to the generic body."""
    kindyn = np.load(tree / clip / directory / "kindyn_1.npz", allow_pickle=True)
    joints = np.asarray(kindyn["joints_world"][0][:n], np.float64)
    covered = np.asarray(kindyn["valid_mask"][0][:n], bool)
    pose16 = body.joints16(np.asarray(kindyn["q"][0][:n], np.float32),
                           np.asarray(kindyn["betas"][0], np.float32), None)
    mass = float(np.asarray(kindyn["total_mass"], np.float64)[0])
    forces_n = np.asarray(kindyn["contact_forces_world"][0][:n], np.float64) \
        * (GENERIC_MASS_KG / mass)
    channel_of_slot = np.asarray([CHANNEL_OF_PARENT22.get(body22_parent(j), -1)
                                  for j in range(joints.shape[1])], np.int64)
    force, moment = channel_wrench(forces_n, joints, channel_of_slot,
                                   joints[:, REFERENCE_JOINT52])
    stored = np.asarray(kindyn["base_wrench"][0], np.float64)[:n] / (mass * GRAVITY)
    return {"joints16": np.where(covered[:, None, None], pose16, np.nan),
            "force": force, "moment": moment, "force_gated": force, "moment_gated": moment,
            "dropped_n": float(np.linalg.norm(forces_n[:, channel_of_slot < 0], axis=-1).sum()),
            "total_n": float(np.linalg.norm(forces_n, axis=-1).sum()),
            "rnea": (np.linalg.norm(stored[:, :3], axis=-1), np.linalg.norm(stored[:, 3:6], axis=-1)),
            "rnea_rows": covered}


def sam3d_row(tree: Path, clip: str, n: int, body: Fk) -> dict:
    """The raw per-frame SAM 3D Body fit: its 16 Parkour joints, no force."""
    from trivial_baselines import init_body
    q_cam, betas, valid = init_body(tree / clip, body.body, body.device)
    pose16 = body.joints16(q_cam[:n], betas, extrinsics(tree, clip, n).astype(np.float32))
    blank = np.full((n, 4, 3), np.nan)
    return {"joints16": np.where(valid[:n, None, None], pose16, np.nan),
            "force": blank, "moment": blank, "force_gated": blank, "moment_gated": blank,
            "dropped_n": 0.0, "total_n": 0.0, "rnea": (np.full(n, np.nan), np.full(n, np.nan)),
            "rnea_rows": np.zeros(n, bool)}


def to_parkour(row: dict, ref: dict, tree: Path, clip: str, fitted: bool,
               gated: bool) -> np.ndarray:
    """``(n, 4, 6)`` the row's channel wrenches in the Parkour frame.

    ``fitted`` swaps the paper's fixed camera rotation for ``score_parkour_rig``'s per-clip
    similarity fit of the row's own world joints onto the Vicon ones — NOT the paper protocol.
    """
    suffix = "_gated" if gated else ""
    if fitted:
        finite = np.isfinite(row["joints16"]).all(axis=(1, 2))
        rotation = clip_rotation(row["joints16"][finite], ref["joints16"][finite])
        force = row[f"force{suffix}"] @ rotation.T
        moment = row[f"moment{suffix}"] @ rotation.T
    else:
        rotation = PARKOUR_R_C @ extrinsics(tree, clip, ref["n"])[:, :3, :3]
        force = np.einsum("tij,tkj->tki", rotation, row[f"force{suffix}"])
        moment = np.einsum("tij,tkj->tki", rotation, row[f"moment{suffix}"])
    return np.concatenate([force, moment], axis=-1)


# ------------------------------------------------------------------ accumulation and tables

class Accumulator:
    """Per-clip numbers of one table row; every mean over clips is unweighted."""

    def __init__(self) -> None:
        self.mpjpe: dict[str, list[float]] = {}
        self.force: list[np.ndarray] = []
        self.moment: list[np.ndarray] = []
        self.uncovered = 0
        self.dropped_n = 0.0
        self.total_n = 0.0
        self.pairs: list[tuple[np.ndarray, np.ndarray]] = []
        self.rnea = [0.0, 0.0, 0.0]

    def add(self, clip: str, mpjpe: float, uncovered: int, force: np.ndarray,
            moment: np.ndarray) -> None:
        self.mpjpe.setdefault(clip[:2], []).append(mpjpe)
        self.force.append(force)
        self.moment.append(moment)
        self.uncovered += uncovered

    def add_rig(self, predicted: np.ndarray, truth: np.ndarray, captured: np.ndarray,
                rnea: tuple[np.ndarray, np.ndarray] | None, rows: np.ndarray | None) -> None:
        """``(n, 4, 3)`` Parkour-frame channel forces on both sides and the row's residual."""
        if np.isfinite(predicted).any():
            self.pairs.append((np.where(captured[..., None], predicted, np.nan),
                               np.where(captured[..., None], truth, np.nan)))
        if rnea is not None:
            force, torque = rnea
            keep = rows & np.isfinite(force)
            self.rnea[0] += float(force[keep].sum())
            self.rnea[1] += float(torque[keep].sum())
            self.rnea[2] += float(keep.sum())

    def rig_cells(self) -> list[float]:
        cells = [float("nan")] * 3
        if self.pairs:
            ours = np.concatenate([p for p, _ in self.pairs])
            truth = np.concatenate([t for _, t in self.pairs])
            weight = np.full(len(ours), GENERIC_MASS_KG * GRAVITY)
            columns = force_columns(truth, ours, weight, np.full(len(ours), np.nan))
            cells = [columns[c] for c in RIG_COLUMNS[:3]]
        n = self.rnea[2]
        return cells + [self.rnea[0] / n if n else float("nan"),
                        self.rnea[1] / n if n else float("nan")]

    def pose_cells(self) -> list[float]:
        per_action = [float(np.mean(self.mpjpe[prefix])) if prefix in self.mpjpe else float("nan")
                      for prefix, _ in ACTIONS]
        return per_action + [float(np.mean([v for values in self.mpjpe.values() for v in values]))]

    def force_cells(self, key: str) -> list[float]:
        values = np.nanmean(np.stack(self.force if key == "force" else self.moment), axis=0)
        return list(values) + [float(values[:2].mean()), float(values[2:].mean())]


def table(header: list[str], columns: list[str], rows: dict[str, list[float]],
          decimals: int = 2) -> list[str]:
    lines = header + ["", "| method | " + " | ".join(columns) + " |",
                      "|" + "---|" * (len(columns) + 1)]
    for label, values in rows.items():
        cells = " | ".join("-" if not np.isfinite(v) else f"{v:.{decimals}f}" for v in values)
        lines.append(f"| {label} | {cells} |")
    return lines + [""]




# ------------------------------------------------------------------ validation against Li

def check_li(results: Path, gt_dir: Path) -> int:
    """Reproduce Li's own stored per-clip numbers from his stored Parkour-frame arrays.

    Also reports how far the processed corpus's ``forces.npz`` is from the ground truth
    the evaluator reads — the two releases of the GT are not the same data.
    """
    truth = load_truth(gt_dir)
    worst = {"mpjpe": 0.0, "force": 0.0, "moment": 0.0}
    processed = {"rows": 0, "max": 0.0, "mask": 0}
    clips = sorted(path.name[:-len("_pred.npz")] for path in results.glob("*_pred.npz"))
    for clip in clips:
        with open(results / f"{clip}.pkl", "rb") as handle:
            stored = pickle.load(handle, encoding="latin-1")[clip]
        predicted = np.load(results / f"{clip}_pred.npz", allow_pickle=True)
        ref = truth[clip]
        mpjpe, _ = pose_error(np.asarray(predicted["joints_16"], np.float64), ref["joints16"])
        force, moment = channel_errors(np.asarray(predicted["forces_parkour"], np.float64)[:, :4],
                                       ref["wrench"], ref["captured"])
        for key, got, want in (("mpjpe", mpjpe, float(stored["mpjpe"]["procrustes"])),
                               ("force", force, np.asarray(stored["mean_linear_force_errors"])[:4]),
                               ("moment", moment, np.asarray(stored["mean_torque_errors"])[:4])):
            worst[key] = max(worst[key], float(np.max(np.abs(np.asarray(got) - want))))
        corpus = np.load(PARKOUR / clip / "forces.npz")
        wrench = np.asarray(corpus["forces"], np.float64)[:, PROCESSED_TO_CHANNEL]
        processed["rows"] += int((np.abs(wrench - ref["wrench"]).max(-1) > 1e-3).sum())
        processed["max"] = max(processed["max"], float(np.abs(wrench - ref["wrench"]).max()))
        processed["mask"] += int((np.asarray(corpus["captured"], bool)[:, PROCESSED_TO_CHANNEL]
                                  != ref["captured"]).sum())
        print(f"{clip}: mpjpe {mpjpe:.6f} (stored {float(stored['mpjpe']['procrustes']):.6f}), "
              f"force {np.array2string(force, precision=3)}", flush=True)
    print(f"\n{len(clips)} clips, max |difference| vs Li's stored results: "
          f"mpjpe {worst['mpjpe']:.3e} mm, linear force {worst['force']:.3e} N, "
          f"moment {worst['moment']:.3e} N.m")
    print(f"processed corpus vs {gt_dir}: {processed['rows']} differing channel-frames, "
          f"max |difference| {processed['max']:.2f}, {processed['mask']} captured-mask "
          f"disagreements")
    failures = [f"{key} off by {value:.3e}" for key, value in worst.items()
                if value > CHECK_TOLERANCE]
    if not clips:
        print(f"FAILED: no `<clip>_pred.npz` under {results}")
        return 1
    if failures:
        print(f"FAILED: {', '.join(failures)} (tolerance {CHECK_TOLERANCE:.0e})")
        return 1
    print("OK")
    return 0


def li_row(results: Path, truth: dict[str, dict], clips: list[str]) -> tuple[Accumulator, float]:
    """One row from Li's OWN evaluator output, rescaled to the paper's generic body.

    Pose comes from the stored ``mpjpe`` (nothing to rescale). The forces are recomputed
    from the stored Parkour-frame wrenches (``<clip>_pred.npz``) times ``74.6 / 74.308``,
    his fixed skeleton's own mass; the stored per-clip means are the check, and the largest
    difference between them and the same computation WITHOUT the rescaling is returned.
    """
    accumulator = Accumulator()
    worst = 0.0
    for clip in clips:
        with open(results / f"{clip}.pkl", "rb") as handle:
            stored = pickle.load(handle, encoding="latin-1")[clip]
        wrench = np.asarray(np.load(results / f"{clip}_pred.npz",
                                    allow_pickle=True)["forces_parkour"], np.float64)[:, :4]
        ref = truth[clip]
        native = channel_errors(wrench, ref["wrench"], ref["captured"])
        for got, want in zip(native, (np.asarray(stored["mean_linear_force_errors"])[:4],
                                      np.asarray(stored["mean_torque_errors"])[:4])):
            worst = max(worst, float(np.nanmax(np.abs(got - want))))
        force, moment = channel_errors(wrench * (GENERIC_MASS_KG / LI_MASS_KG),
                                       ref["wrench"], ref["captured"])
        accumulator.add(clip, float(stored["mpjpe"]["procrustes"]), 0, force, moment)
        accumulator.add_rig(wrench[:, :, :3] * (GENERIC_MASS_KG / LI_MASS_KG),
                            ref["wrench"][:, :, :3], ref["captured"], None, None)
    return accumulator, worst


# ------------------------------------------------------------------ the report

def zero_reference(truth: dict[str, dict], clips: list[str]) -> dict[str, list[float]]:
    """What a method that predicts NO force at all scores — the size of the truth itself."""
    force, moment = [], []
    for clip in clips:
        ref = truth[clip]
        errors = channel_errors(np.zeros_like(ref["wrench"]), ref["wrench"], ref["captured"])
        force.append(errors[0])
        moment.append(errors[1])
    cells = {}
    for key, values in (("force", force), ("moment", moment)):
        mean = np.nanmean(np.stack(values), axis=0)
        cells[key] = list(mean) + [float(mean[:2].mean()), float(mean[2:].mean())]
    return cells


def mass_sentence(values: list[tuple[float, float]]) -> str:
    """How one row's newtons were made: its native mass over the clips and the conversion."""
    tree = np.asarray([v[0] for v in values], np.float64)
    native = np.asarray([v[1] for v in values], np.float64)
    span = lambda a: (f"{a[0]:.3f} kg" if np.ptp(a) < 1e-6 else  # noqa: E731
                      f"{a.min():.3f}-{a.max():.3f} kg")
    if np.isnan(tree).all():
        return (f"native mass {span(native)} (Li's fixed skeleton, the sum of"
                f" `person_models/full_body_inertia.pkl`); his stored Parkour-frame wrenches"
                f" x {GENERIC_MASS_KG}/{native[0]:.3f}")
    if np.allclose(tree, native):
        return (f"native mass = the reconstructed body, {span(tree)}; newtons rescaled by"
                f" {GENERIC_MASS_KG}/that mass (a body-weight dump is simply"
                f" x {GENERIC_MASS_KG} x g)")
    return (f"native mass {span(native)} (the method's own body); dump bw x the tree's"
            f" {span(tree)} x g x {GENERIC_MASS_KG}/that native mass")


def report(rows: dict[str, dict[str, Accumulator]], clips: list[str], truth: dict[str, dict],
           gt_dir: Path, notes: list[str], li: dict[str, Accumulator],
           masses: dict[str, list[tuple[float, float]]]) -> list[str]:
    """The paper-shaped tables, the quoted paper rows, and the two controls."""
    paper = lambda key: {label: list(PAPER_ROWS[label][key])  # noqa: E731
                         + [float(np.mean(PAPER_ROWS[label][key][:2])),
                            float(np.mean(PAPER_ROWS[label][key][2:]))]
                         for label in PAPER_ROWS if key in PAPER_ROWS[label]}
    zero = zero_reference(truth, clips)
    pose = {label: list(PAPER_ROWS[label]["mpjpe"]) + [float("nan")] for label in PAPER_ROWS} | {
        label: accumulator.pose_cells() + [float("nan")] for label, accumulator in li.items()} | {
        label: variants["paper"].pose_cells() + [float(variants["paper"].uncovered)]
        for label, variants in rows.items()}
    force = paper("force") | {
        label: accumulator.force_cells("force") for label, accumulator in li.items()} | {
        label: variants["paper"].force_cells("force") for label, variants in rows.items()} | {
        "zero forces (reference)": zero["force"]}
    moment = paper("moment") | {
        label: accumulator.force_cells("moment") for label, accumulator in li.items()} | {
        label: variants["paper"].force_cells("moment") for label, variants in rows.items()} | {
        "zero forces (reference)": zero["moment"]}

    lines = [f"# LAAS Parkour, the paper's protocol ({len(clips)} clips)", "",
             f"Ground truth: `{gt_dir}`.", ""]
    lines += table(["## Procrustes MPJPE [mm], by technique"],
                   [label for _, label in ACTIONS] + ["Avg", "uncovered frames"], pose)
    lines += table(["## Mean linear force error [N], by contact channel"],
                   list(CHANNELS) + ["soles", "hands"], force)
    lines += table(["## Mean moment error [N.m], by contact channel"],
                   list(CHANNELS) + ["soles", "hands"], moment)
    lines += table(["## Force agreement and RNEA residual, pooled over the captured"
                    " channel-frames", "",
                    "The climb_wall_3 rig table's columns (`scripts/score_rig.py::force_columns`)"
                    " read on the same Parkour-frame channel forces as the force table"
                    " (ungated, fixed rotation, generic mass — which the three agreement"
                    " columns do not depend on): the mean angle between the vectors where"
                    " both read above 50 N, the Pearson correlation of the sizes, the"
                    " per-frame load-share error in percentage points. `RNEA` is the root"
                    " wrench residual of the row's own pose under its own forces in body"
                    " weights of the reconstructed body (`score_rig`'s reading: the dumped"
                    " slot forces gated by the predicted probability, the ±2 stencil rows); the"
                    " optimisation rows report the residual their own solve stored, which is"
                    " not the same quantity."],
                   list(RIG_COLUMNS),
                   {label: accumulator.rig_cells() for label, accumulator in li.items()} | {
                    label: variants["paper"].rig_cells() for label, variants in rows.items()},
                   decimals=3)
    if all("gated" in variants for variants in rows.values()):
        lines += table(["## Secondary reading: forces gated by the predicted contact"
                        " probability", "",
                        "NOT the rig table's reading — `scripts/score_rig.py::prediction_row`"
                        " sums the RAW slot forces for its force MAE and gates only its RNEA"
                        " residual. This table multiplies every slot force by that slot's"
                        " predicted contact probability before folding it onto a channel;"
                        " everything else is the paper protocol. The optimisation rows and"
                        " Li's row are unchanged (no probabilities to gate with)."],
                       list(CHANNELS) + ["soles", "hands"],
                       {label: variants["gated"].force_cells("force")
                        for label, variants in rows.items()})
    lines += table(["## NOT the paper protocol: the per-clip fitted rotation", "",
                    "The same (ungated) forces carried into the MoCap axes by"
                    " `score_parkour_rig`'s per-clip similarity fit of the row's own world"
                    " joints onto the Vicon ones, instead of the paper's fixed camera"
                    " rotation. Pose is itself a similarity fit and does not change."],
                   [f"{c} N" for c in CHANNELS] + [f"{c} N.m" for c in CHANNELS],
                   {label: variants["fitted"].force_cells("force")[:4]
                    + variants["fitted"].force_cells("moment")[:4]
                    for label, variants in rows.items()})
    lines += [
        "## Protocol", "",
        "Pose: the dataset's own evaluator (`Parkour-dataset/lib/parkour_evaluator.py`) —"
        " a per-frame Procrustes fit WITH SCALE of the 16 Vicon joints, mean over joints and"
        " frames, unweighted mean over the clips of a technique and over all clips. The"
        " evaluator's window is frames `0 .. N-4` of the raw capture, which is exactly the"
        " frame range the processed corpus keeps, so every processed frame is scored. A frame"
        " a method did not cover has no pose at all; those frames are left out of that clip's"
        " mean and counted in the last column.", "",
        "Forces: mean Euclidean error of the linear part (N) and of the moment (N.m) per"
        " channel over the frames the rig captured, per clip, then the unweighted mean over"
        " clips — the evaluator's `compute_mean_force_errors`. NO alignment: each method's"
        " camera-frame wrench is carried into the MoCap axes by the paper's fixed"
        " `parkourR_c = [[-1,0,0],[0,0,-1],[0,-1,0]]`. A channel's force is the SUM of its"
        " limb's slot forces, ungated — the same reading as the climb_wall_3 rig table"
        " (`scripts/score_rig.py::prediction_row`, which gates only its RNEA residual); the"
        " gated reading is the secondary table above. A channel's moment is"
        " `sum (anchor - p_ref) x f` about our ankle (soles) or our wrist (hands).", "",
        f"Newtons: the paper's generic {GENERIC_MASS_KG} kg body for every row. Every dump"
        " stores body weights of the TREE's reconstructed body, so its native newtons are"
        " `bw * tree_mass * g` and the generic ones are those times"
        " `74.6 / (the mass the METHOD itself assumed)`. Per row:",
    ] + [f"- {label}: {mass_sentence(values)}" for label, values in masses.items()] + [
        "",
        "The GT sole moment is the plate's moment about the ankle joint. Li's OWN sole"
        " moment is not: `optimizer.py::ExpressLocalForcesInParkourFrame` rotates each"
        " contact wrench and sums it WITHOUT translating it, so on a toe-only frame his sole"
        " moment is about the toe. The reviewer measured that difference at 0.01 / 0.14 N.m"
        " over the 797 toe-contact frames — small, but the sole columns are not exactly the"
        " same quantity. The GT HAND moment is the instrumented bar's moment about the"
        " SENSOR frame (the dataset only rotates it), so no method's hand moment is"
        " comparable to it — Li's own published numbers there are 131 / 134 N.m, of the same"
        " size as the GT itself.", "",
        "Two conversions lose moment. PhysPT's dump (`scripts/physpt_dumps.py`) replaces the"
        " distributed vertex forces of a foot by their sum at a magnitude-weighted centroid,"
        " which does not preserve the moment of non-parallel forces: the discarded"
        " `sum (p_i - centroid) x f_i` is 0.3 N.m on average, 8.6 N.m at most. Li's dump"
        " (`scripts/estmf_dumps.py`) keeps his linear forces and their points but not his"
        " solved ANGULAR wrench, so the `estmf_rec` row's moments are point-force moments"
        " only; his own evaluator row above carries the full wrench.", "",
        "Channels follow Li's own mapping: everything on a leg below the hip (knee, ankle,"
        " heel, toes) loads that side's sole channel and the wrist and fingers load that"
        " side's hand channel. A slot on the head, an elbow, a shoulder, a hip, the sacrum or"
        " the chest has no Parkour channel and is dropped; the last lines say how much force"
        " that is per row.", "",
        "The ground truth is the ORIGINAL release of `contact_forces_local.pkl`, the copy"
        " shipped inside Li's estimator, which is what the dataset's evaluator reads and what"
        " the published numbers were computed against. The copy under"
        " `data/Parkour-dataset/gt_motion_forces/`, which the processed corpus"
        " (`processed/<clip>/forces.npz`) was imported from, is a REDUCED one: every reading"
        " under 30 N is zeroed, which empties most of the hand moments. Scoring against it"
        " reproduces neither Li's own stored per-clip results nor the paper.", "",
        "`zero forces (reference)` is the error of predicting no force at all — the size of"
        " the truth itself, and the bar every row has to beat.",
    ]
    lines += [""] + notes
    lines += [f"`{label}`: {variants['paper'].uncovered} frames with no prediction (scored as"
              f" no pose and no force), {variants['paper'].dropped_n:.0f} N.frames of slot"
              " force on no Parkour channel"
              f" ({100 * variants['paper'].dropped_n / max(variants['paper'].total_n, 1e-9):.1f}"
              " % of the row's total)."
              for label, variants in rows.items()]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=Path, default=PARKOUR / "out")
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--optim", nargs="*", default=["human_optim", "human_optim_scenefree"])
    parser.add_argument("--check-li", type=Path, default=None)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--li-pkl", nargs=2, action="append", metavar=("DIR", "LABEL"),
                        default=[[str(DEFAULT_LI_RERUN), LI_RERUN_LABEL]],
                        help="a directory of Li's own evaluator results and its row label")
    parser.add_argument("--gate", action="store_true",
                        help="also report the forces gated by the predicted contact probability")
    parser.add_argument("--sam3d", action="store_true",
                        help="add the raw per-frame SAM 3D Body fit as a pose-only row")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.check_li is not None:
        return check_li(args.check_li, args.gt)

    truth = load_truth(args.gt)
    clips = sorted(d.name for d in args.tree.iterdir()
                   if (d / "geometry" / "transform.npz").is_file())
    model_path = yaml.safe_load(
        (_ROOT / "configs" / "base.yaml").read_text())["model"]["smplx"]["model_path"]
    body = Fk(model_path, torch.device(args.device))
    wrench = RootWrench(model_path, torch.device(args.device))
    labels: dict[str, tuple[str, str]] = {}
    if args.sam3d:
        labels["SAM 3D Body"] = ("sam3d", "")
    labels |= {OPTIM_LABELS.get(directory, directory): ("optim", directory)
               for directory in args.optim}
    present = [run for run in args.runs
               if all((args.tree / clip / "predictions" / run).is_dir() for clip in clips)]
    missing = sorted(set(args.runs) - set(present))
    notes = [f"Not scored, no dump under `predictions/`: {', '.join(missing)}."] if missing else []
    labels |= {RUN_LABELS.get(run, run): ("pred", run) for run in present}
    variants = [("paper", False, False), ("fitted", True, False)]
    if args.gate:
        variants.append(("gated", False, True))
    rows = {label: {key: Accumulator() for key, _, _ in variants} for label in labels}
    masses: dict[str, list[tuple[float, float]]] = {}

    for clip in clips:
        ref = truth[clip]
        tree_mass = float(np.asarray(np.load(args.tree / clip / "human_optim" / "kindyn_1.npz",
                                             allow_pickle=True)["total_mass"], np.float64)[0])
        fps = float(np.load(args.tree / clip / "geometry" / "transform.npz")["fps"])
        for label, (kind, run) in labels.items():
            mass = tree_mass
            if kind == "sam3d":
                row = sam3d_row(args.tree, clip, ref["n"], body)
            elif kind == "optim":
                row = optimisation_row(args.tree, clip, run, ref["n"], body)
            else:
                mass = native_mass_kg(run, clip, tree_mass)
                row = prediction_row(args.tree / clip / "predictions" / run, args.tree, clip,
                                     ref["n"], body, tree_mass * GRAVITY * GENERIC_MASS_KG / mass,
                                     wrench, fps)
            masses.setdefault(label, []).append((tree_mass, mass))
            mpjpe, uncovered = pose_error(row["joints16"], ref["joints16"])
            for key, fitted, gated in variants:
                accumulator = rows[label][key]
                parkour = to_parkour(row, ref, args.tree, clip, fitted, gated)
                force, moment = channel_errors(parkour, ref["wrench"], ref["captured"])
                accumulator.add(clip, mpjpe, uncovered, force, moment)
                accumulator.dropped_n += row["dropped_n"]
                accumulator.total_n += row["total_n"]
                if key == "paper":
                    accumulator.add_rig(parkour[:, :, :3], ref["wrench"][:, :, :3],
                                        ref["captured"], row["rnea"], row["rnea_rows"])
        print(f"{clip}: {ref['n']} frames", flush=True)

    li: dict[str, Accumulator] = {}
    for directory, label in args.li_pkl:
        if not all((Path(directory) / f"{clip}.pkl").is_file() for clip in clips):
            notes.append(f"Not scored, no `<clip>.pkl` under `{directory}`: {label}.")
            continue
        li[label], worst = li_row(Path(directory), truth, clips)
        masses[label] = [(float("nan"), LI_MASS_KG)]
        print(f"{label}: native force/moment means reproduce his stored ones to {worst:.3e}")
    text = "\n".join(report(rows, clips, truth, args.gt, notes, li, masses)) + "\n"
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
