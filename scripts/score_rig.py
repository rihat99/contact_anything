"""Score our prediction dumps and Peter's optimisation on the climb_wall_3 measured rig.

The rig (``BetterVideoReconstruction/peter/``) is the only place where pose, contact and
force are all MEASURED on the same clips: a calibrated three-camera fit gives the body
(``climb_wall_3_gt/human/<clip>/human_optim/kindyn_1.npz``, whose world IS the left
camera's frame), and four instrumented holds give each limb's contact state and its force
in newtons (``sensor_forces.npz``). Every scored row saw the same single-view pixels of
``cam_left.mp4``.

Rows are ``optimisation`` (the tree's ``human_optim``), every run name given on the
command line, whose dumps ``scripts/predict_reconstruction.py`` wrote into
``<clip>/predictions/<run>/``
(``--legacy-dirs`` reads the older ``<clip>/predictions_<run>/`` layout instead), and with
``--sam3d`` the raw per-frame SAM 3D Body fit every row starts from (``sam3d/smplx_params.npz``,
pose columns only).

Columns, all pooled over the 68 clips:

* **pose** — the corpus test protocol, not BVR's: :func:`model.loss.smplx.pose_metric_stats`
  for the pelvis-aligned camera-frame ``MPJPE`` / ``PA-MPJPE`` / ``PVE`` / ``Accel`` and
  :func:`utils.gvhmr_metrics.global_metrics` for the world ``WA-MPJPE100`` /
  ``W-MPJPE100`` / ``RTE`` / ``jitter``, each weighted by the rows (segments, for the two
  chunked ones) it was measured on. ``WA-``/``W-MPJPE`` and ``RTE`` read each row's OWN
  world trajectory (alignment makes the choice of world frame immaterial); ``jitter`` reads
  the CAMERA-frame trajectory, which is the frame the three-camera reference lives in — the
  tree's per-frame extrinsics carry their own wobble, and lifting through them would score a
  different signal from the reference's. The reference's own jitter is the floor, printed
  above the table (in that same camera frame); ``jitter world`` is the same number read on
  the row's own world trajectory, where the tree's extrinsics no longer contribute. Both bodies are re-posed from their stored BetterHuman ``q``
  with the same neutral SMPL-X, so joints and vertices compare like with like; the
  FK-vs-stored check above the table is how far that re-posing lands from each row's own
  stored joints.
* **contacts** — ``F1`` / ``P`` / ``R`` of the four board limbs against
  ``human_optim/contacts_1.npz``'s ``limb_contact``, pooled over limbs and frames. A dump's
  own slots (its ``contact_set``) fold onto the six kindyn groups first — the max
  probability over a group's member slots — and those fold onto the four limbs the way
  ``tools/human_optim/boards.py`` does — hand = hand, foot = toe ∪ heel — at the dump's own
  threshold; the optimisation rows' labels are read
  from their ``contacts_2.npz`` exactly as ``score_gt_contacts.py`` reads them.
* **forces** — each row's limb force against the measured one. A dump's slots fold onto
  the six kindyn groups and those onto the four limbs, both by VECTOR SUM and carry body-weight units, so the boards' newtons need a
  body weight to compare against. ``--mass optim`` (the default) uses the RECONSTRUCTED
  body's mass — ``total_mass`` of the clip's ``human_optim/kindyn_1.npz``, 48.8-66.3 kg —
  which is the mass the corpus forces were put in body weights with in the first place and
  the one ``compare_gt_contact.py`` normalises its residual by; ``--mass subject`` uses the
  participant's MEASURED mass instead (``mass_kg`` in the GT ``kindyn_1.npz``: 72.47 kg for
  A, 47.99 kg for B), which reads ~10 % lower in newtons. ``vec MAE bw% (GT mass)`` reads the
  boards in body weights of the MEASURED mass and compares them to our body-weight output
  as it is (the optimisation rows' newtons are read back in the reconstructed body weight
  they were solved in; no reconstructed mass on the boards' side). ``vec MAE`` is the error of the FULL 3D VECTOR,
  ``size MAE`` the error of its length (the column BVR's tables print); ``LH`` / ``RH`` /
  ``LF`` / ``RF MAE N`` split the vector error per limb. ``angle`` is the
  MEAN angle between the vectors on limb-frames where both read above 50 N, ``corr`` the
  Pearson correlation of the sizes, and ``share pp`` the load-share error read PER FRAME
  (:func:`compare_gt_contact.force_metrics`: each limb's share of that frame's total,
  ``|Δ|`` averaged over limbs and frames) — not the clip-averaged reading BVR's
  ``compare_climb_wall_3_runs.py`` puts in its table.
* **RNEA** — the root-wrench residual of the PREDICTED pose under the PREDICTED forces
  applied at the dumped slot points (``anchor_cam``, lifted into the tree's world) and
  gated by the predicted contact probabilities (:meth:`model.physics.RootWrench.residual`,
  the ``force_consistency`` call): linear part in body weights, angular in bw·m, mean over
  the rows the ±2 stencil covers. The optimisation rows are not recomputed — their solve
  already stores its own root residual (``base_wrench``, normalised by that solve's own
  body mass, which is what ``compare_gt_contact.py`` reads), so those cells report the
  stored value and are NOT the same quantity as ours.

    python scripts/score_rig.py --runs L_body D16_notoken --legacy-dirs \
        --out output_6/logs/rig_20260919.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import roma
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.append(str(_ROOT / "scripts"))
_BVR = _ROOT.parent / "BetterVideoReconstruction"
for _p in (_BVR, _BVR / "scripts", _BVR / "scripts" / "diagnostics"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from compare_gt_contact import CONTACT_N, limb_forces                 # noqa: E402
from score_frame_contacts import _FOLD_10                            # noqa: E402
from tools.human_optim.boards import LIMBS                           # noqa: E402

from model.contact_frames import ContactSet, contact_set             # noqa: E402
from model.loss.smplx import NUM_BODY_JOINTS, pose_metric_stats      # noqa: E402
from model.physics import GRAVITY, RootWrench                        # noqa: E402
from utils.gvhmr_metrics import compute_jitter, global_metrics       # noqa: E402

#: Kindyn group indices of the four board limbs (LH, RH, L foot = toe + heel, R foot).
GROUP_LIMBS = ((0,), (1,), (2, 4), (3, 5))
#: Width of the 22-joint part of a BetterHuman ``q`` (pelvis, root quat, 21 joint quats).
BODY_Q_DIM = 91
#: The optimisation rows, in table order: label -> the run directory under ``<clip>/``.
OPTIM_RUNS = {"optimisation": "human_optim"}
#: Table columns, in order.
COLUMNS = ("MPJPE", "PA-MPJPE", "PVE", "Accel", "WA-MPJPE", "W-MPJPE", "RTE", "jitter",
           "jitter world", "F1", "P", "R", "vec MAE bw%", "vec RMSE bw%", "vec MAE bw% (GT mass)",
           "vec MAE N", "LH MAE N", "RH MAE N", "LF MAE N", "RF MAE N", "size MAE N",
           "size RMSE N", "angle", "corr", "share pp", "RNEA f bw", "RNEA tau bw·m")
#: The force columns, blank on a row that predicts no force.
_FORCE_COLUMNS = COLUMNS[COLUMNS.index("vec MAE bw%"):]
#: Columns printed to three decimals (rates and correlations); the rest to two.
_RATES = frozenset(("F1", "P", "R", "corr", "RNEA f bw", "RNEA tau bw·m"))


# ------------------------------------------------------------------ the rig's own data

def clip_names(gt_root: Path) -> list[str]:
    return sorted(d.name for d in (gt_root / "human").iterdir()
                  if d.is_dir() and not d.name.startswith("_"))


def reference(gt_root: Path, clip: str) -> dict:
    """The three-camera body and everything the rig measured on one clip."""
    kindyn = np.load(gt_root / "human" / clip / "human_optim" / "kindyn_1.npz",
                     allow_pickle=True)
    contacts = np.load(gt_root / "human" / clip / "human_optim" / "contacts_1.npz",
                       allow_pickle=True)
    names = [str(n) for n in contacts["limb_names"]]
    n = int(kindyn["num_frames"])
    return {
        "n": n,
        "fps": float(np.load(gt_root / "human" / clip / "cameras.npz",
                             allow_pickle=True)["fps"]),
        "q": np.asarray(kindyn["q"][0][:n, :BODY_Q_DIM], np.float32),
        "betas": np.asarray(kindyn["betas"][0], np.float32),
        "joints": np.asarray(kindyn["joints_world"][0][:n, :NUM_BODY_JOINTS], np.float32),
        "valid": np.asarray(kindyn["valid_mask"][0][:n], bool),
        "contact": np.asarray(contacts["limb_contact"][0], bool)[:n,
                                                                 [names.index(l) for l in LIMBS]],
        "mass_kg": float(kindyn["mass_kg"]),
    }


def board_forces(tree: Path, clip: str, n: int) -> np.ndarray:
    """``(n, 4, 3)`` measured limb reaction forces in newtons, the tree's world."""
    sensor = np.load(tree / clip / "human_optim" / "sensor_forces.npz", allow_pickle=True)
    channels = [str(x) for x in sensor["limbs"]]
    if channels != LIMBS:
        raise ValueError(f"{clip}: sensor channels are {channels}, expected {LIMBS}")
    return np.asarray(sensor["reaction_world"], np.float64)[:n]


# ------------------------------------------------------------------ bodies

class Body:
    """The neutral 22-joint BetterHuman SMPL-X every row is re-posed with."""

    def __init__(self, model_path: str, device: torch.device) -> None:
        import better_human as bh
        self.device = device
        self.body = bh.SMPLX(model_path=model_path, gender="neutral", num_betas=10,
                             use_hands=False, use_face=False, compute_mass=False,
                             dtype=torch.float32, device=device)

    def pose(self, q: np.ndarray, betas: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """``(T, 91)`` configurations and ``(T, 10)`` shapes -> joints ``(T, 22, 3)``, vertices."""
        q_t = torch.as_tensor(q, dtype=torch.float32, device=self.device)
        betas_t = torch.as_tensor(np.broadcast_to(betas, (len(q), 10)).copy(),
                                  dtype=torch.float32, device=self.device)
        fk = self.body.with_shape(betas=betas_t).fk(q_t)
        return fk.joint_pose_world[..., 1:, :3], self.body.vertices_from_data(fk)


def to_camera(points: torch.Tensor, ext: np.ndarray) -> torch.Tensor:
    """``(T, K, 3)`` world points through ``(T, 4, 4)`` ``cam_from_world``."""
    e = torch.as_tensor(ext, dtype=points.dtype, device=points.device)
    return torch.einsum("tij,tkj->tki", e[:, :3, :3], points) + e[:, None, :3, 3]


def body_rotations(q: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """``(T, 91)`` q -> root rotation ``(T, 3, 3)`` and the 21 parent-local ``(T, 21, 3, 3)``."""
    q_t = torch.as_tensor(q, dtype=torch.float32, device=device)
    root = roma.unitquat_to_rotmat(q_t[:, 3:7])
    joints = roma.unitquat_to_rotmat(q_t[:, 7:BODY_Q_DIM].reshape(-1, 21, 4))
    return root, joints


# ------------------------------------------------------------------ one row on one clip

def optimisation_row(tree: Path, clip: str, optim_dir: str, ref: dict, body: Body) -> dict:
    """One of Peter's solves: pose, its own contact labels, its solved forces, its residual."""
    n = ref["n"]
    kindyn = np.load(tree / clip / optim_dir / "kindyn_1.npz", allow_pickle=True)
    ext = np.asarray(np.load(tree / clip / "geometry" / "transform.npz",
                             allow_pickle=True)["extrinsics"], np.float64)[:n]
    joints_world, verts_world = body.pose(
        np.asarray(kindyn["q"][0][:n, :BODY_Q_DIM], np.float32),
        np.asarray(kindyn["betas"][0], np.float32))
    force_n, _ = limb_forces(tree / clip / optim_dir / "kindyn_1.npz", n)
    contact = np.load(tree / clip / optim_dir / "contacts_2.npz", allow_pickle=True)
    names = [str(f) for f in contact["contact_frame_names"]]
    frame_contact = np.asarray(contact["frame_contact"][0], bool)[:n]
    wrench = np.asarray(kindyn["base_wrench"][0], np.float64)[:n]
    body_weight = float(np.asarray(kindyn["total_mass"], np.float64)[0]) * GRAVITY
    return {
        "joints_cam": to_camera(joints_world, ext), "verts_cam": to_camera(verts_world, ext),
        "joints_world": joints_world.detach().cpu().numpy().astype(np.float64),
        "fk_check": (joints_world,
                     np.asarray(kindyn["joints_world"][0][:n, :NUM_BODY_JOINTS], np.float64)),
        "valid": np.asarray(kindyn["valid_mask"][0][:n], bool),
        "contact": np.stack([frame_contact[:, [names.index(f) for f in _FOLD_10[limb]
                                               if f in names]].any(1) for limb in LIMBS], 1),
        "force_n": np.asarray(force_n, np.float64)[:n],
        "rnea": (np.linalg.norm(wrench[:, :3], axis=-1) / body_weight,
                 np.linalg.norm(wrench[:, 3:6], axis=-1) / body_weight),
        "rnea_rows": np.asarray(kindyn["valid_mask"][0][:n], bool),
    }


def sam3d_row(tree: Path, clip: str, ref: dict, body: Body, body52) -> dict:
    """The raw per-frame SAM 3D Body fit: pose only, no contact, no force."""
    from trivial_baselines import init_body
    n = ref["n"]
    q_cam, betas, valid = init_body(tree / clip, body52, body.device)
    ext = np.asarray(np.load(tree / clip / "geometry" / "transform.npz",
                             allow_pickle=True)["extrinsics"], np.float64)[:n]
    joints_cam, verts_cam = body.pose(q_cam[:n, :BODY_Q_DIM], betas)
    cam = joints_cam.detach().cpu().numpy().astype(np.float64)
    joints_world = np.einsum("tji,tkj->tki", ext[:, :3, :3], cam - ext[:, None, :3, 3])
    nan = np.full(n, np.nan)
    return {
        "joints_cam": joints_cam, "verts_cam": verts_cam, "joints_world": joints_world,
        "fk_check": (joints_cam, cam), "valid": valid[:n], "contact": None,
        "force_n": np.full((n, 4, 3), np.nan), "rnea": (nan, nan), "rnea_rows": np.zeros(n, bool),
    }


def dump_slots(dump) -> ContactSet:
    """The contact set a dump was written with (older dumps carry none: the six groups)."""
    name = str(dump["contact_set"]) if "contact_set" in dump.files else "kindyn6"
    return contact_set(name)


def prediction_row(pred_dir: Path, tree: Path, clip: str, ref: dict, body: Body,
                   wrench: RootWrench, threshold: float, force_name: str) -> dict:
    """One of our dumps: the refined body, its contact probabilities and its forces.

    The dump's K slots are folded onto the six kindyn groups first (probabilities by MAX,
    forces by SUM), because the boards are per limb and the fold onto them is per group.
    """
    n = ref["n"]
    dump = np.load(pred_dir / "smplx.npz", allow_pickle=True)
    slots = dump_slots(dump)
    q_cam = np.asarray(dump["q_cam"][0][:n, :BODY_Q_DIM], np.float32)
    betas = np.asarray(dump["betas"][0][:n], np.float32)
    covered = np.asarray(dump["covered"][0][:n], bool)
    q_cam = np.where(covered[:, None], q_cam, ref["q"])          # a NaN row would poison the FK
    betas = np.nan_to_num(betas)
    joints_cam, verts_cam = body.pose(q_cam, betas)
    joints_world = np.asarray(dump["joints_world"][0][:n, :NUM_BODY_JOINTS], np.float64)

    has_contact = (pred_dir / "contacts.npz").is_file()     # a no-contact build dumps none
    probs = (np.asarray(np.load(pred_dir / "contacts.npz", allow_pickle=True)["probs"][0][:n],
                        np.float64) if has_contact else np.ones((n, slots.count)))
    probs6 = slots.fold_max(np.nan_to_num(probs))           # (n, 6), -inf where no member
    contact6 = probs6 >= threshold
    forces = np.load(pred_dir / force_name, allow_pickle=True)
    forces_bw = np.nan_to_num(np.asarray(forces["forces_world"][0][:n], np.float64))
    forces_bw6 = slots.fold_sum(forces_bw)
    force_bw4 = np.stack([forces_bw6[:, list(g)].sum(1) for g in GROUP_LIMBS], 1)
    points_cam = np.nan_to_num(np.asarray(forces["anchor_cam"][0][:n], np.float64))

    res_f, res_t, rows = predicted_residual(
        wrench, tree, clip, ref, dump, q_cam, betas,
        forces_bw * np.nan_to_num(probs)[..., None], points_cam,
        np.asarray(slots.parent_joint22, np.int64), covered)
    return {
        "joints_cam": joints_cam, "verts_cam": verts_cam, "joints_world": joints_world,
        "fk_check": (joints_cam,
                     np.asarray(dump["joints_cam"][0][:n, :NUM_BODY_JOINTS], np.float64)),
        "valid": covered,
        "contact": (np.stack([contact6[:, list(g)].any(1) for g in GROUP_LIMBS], 1)
                    if has_contact else None),
        "force_n": force_bw4 * (ref["body_weight_n"]),
        "rnea": (res_f, res_t), "rnea_rows": rows,
    }


def predicted_residual(wrench: RootWrench, tree: Path, clip: str, ref: dict, dump,
                       q_cam: np.ndarray, betas: np.ndarray, forces_world_bw: np.ndarray,
                       points_cam: np.ndarray, parent22: np.ndarray,
                       covered: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``RootWrench.residual`` on the dumped body, as ``force_consistency`` calls it."""
    n = ref["n"]
    device = wrench.device
    ext = np.asarray(np.load(tree / clip / "geometry" / "transform.npz",
                             allow_pickle=True)["extrinsics"], np.float32)[:n]
    rot_wc = torch.as_tensor(ext[:, :3, :3], device=device).transpose(-1, -2)
    t_cw = torch.as_tensor(ext[:, :3, 3], device=device)
    points_world = torch.einsum(
        "tij,tkj->tki", rot_wc,
        torch.as_tensor(points_cam, dtype=torch.float32, device=device) - t_cw[:, None])
    root_cam, body_rot = body_rotations(q_cam, device)
    root_world = rot_wc @ root_cam
    pelvis = torch.as_tensor(
        np.nan_to_num(np.asarray(dump["joints_world"][0][:n, 0], np.float32)), device=device)
    gravity = np.asarray(np.load(tree / clip / "geocalib" / "gravity.npz",
                                 allow_pickle=True)["gravity_world"], np.float32).reshape(3)
    gravity = torch.as_tensor(gravity / np.linalg.norm(gravity), device=device)[None]
    seconds = torch.arange(n, device=device, dtype=torch.float32)[None] / ref["fps"]
    valid = torch.as_tensor(covered & ref["valid"], device=device)[None]
    res_f, res_t, rows, _ = wrench.residual(
        pelvis[None], root_world[None], body_rot[None],
        torch.as_tensor(betas[covered].mean(0) if covered.any() else betas[0],
                        device=device)[None],
        torch.as_tensor(forces_world_bw, dtype=torch.float32, device=device)[None],
        points_world[None], torch.as_tensor(parent22, device=device),
        gravity, seconds, valid)
    return (res_f[0].norm(dim=-1).cpu().numpy(), res_t[0].norm(dim=-1).cpu().numpy(),
            rows[0].cpu().numpy())


# ------------------------------------------------------------------ accumulation

class Accumulator:
    """Pooled statistics of one table row over the clips."""

    def __init__(self) -> None:
        self.pose = torch.zeros(20, dtype=torch.float64)
        self.world = {k: [0.0, 0.0] for k in ("wa_mpjpe100", "w_mpjpe100", "rte", "jitter",
                                              "jitter_world")}
        self.counts = dict(tp=0, fp=0, fn=0)
        self.boards, self.ours, self.mass_g, self.subject_g = [], [], [], []
        self.rnea = [0.0, 0.0, 0.0]
        self.fk_check = []

    def add_pose(self, row: dict, ref: dict, gt_joints, gt_verts, valid: np.ndarray) -> None:
        keep = torch.as_tensor(valid, device=gt_joints.device)
        seconds = torch.arange(ref["n"], device=gt_joints.device,
                               dtype=torch.float32) / ref["fps"]
        self.pose += pose_metric_stats(
            torch.nan_to_num(row["joints_cam"]), torch.nan_to_num(row["verts_cam"]),
            gt_joints, gt_verts, keep, ref["n"], seconds)
        mask = valid & np.isfinite(row["joints_world"]).all(axis=(-1, -2))
        if mask.sum() >= 2:
            pred = torch.as_tensor(row["joints_world"][mask], dtype=torch.float32)
            gt = torch.as_tensor(ref["joints"][mask], dtype=torch.float32)
            for name, values in global_metrics(pred, gt, ref["fps"]).items():
                if name == "jitter":            # the world reading; the column is the camera one
                    name = "jitter_world"
                self.world[name][0] += float(np.asarray(values, np.float64).sum())
                self.world[name][1] += float(len(values))
        if mask.sum() >= 4:
            camera = torch.nan_to_num(row["joints_cam"]).detach().cpu()[
                torch.as_tensor(mask)].float()
            values = compute_jitter(camera, fps=ref["fps"])
            self.world["jitter"][0] += float(np.asarray(values, np.float64).sum())
            self.world["jitter"][1] += float(len(values))
        fk, stored = row["fk_check"]
        err = fk.detach().cpu().numpy().astype(np.float64) - stored
        if np.isfinite(err[valid]).all() and valid.any():
            self.fk_check.append(float(np.linalg.norm(err[valid], axis=-1).mean() * 1e3))

    def add_contact(self, predicted, truth: np.ndarray, rows: np.ndarray) -> None:
        if predicted is None:
            return
        got, want = predicted[rows], truth[rows]
        self.counts["tp"] += int((got & want).sum())
        self.counts["fp"] += int((got & ~want).sum())
        self.counts["fn"] += int((~got & want).sum())

    def add_force(self, boards: np.ndarray, ours: np.ndarray, body_weight_n: float,
                  subject_weight_n: float, rows: np.ndarray) -> None:
        if not np.isfinite(ours).any():
            return
        self.boards.append(boards[rows])
        self.ours.append(np.where(np.isfinite(boards[rows]), ours[rows], np.nan))
        self.mass_g.append(np.full(int(rows.sum()), body_weight_n))
        self.subject_g.append(np.full(int(rows.sum()), subject_weight_n))

    def add_rnea(self, row: dict, valid: np.ndarray) -> None:
        force, torque = row["rnea"]
        rows = row["rnea_rows"] & valid & np.isfinite(force)
        self.rnea[0] += float(force[rows].sum())
        self.rnea[1] += float(torque[rows].sum())
        self.rnea[2] += float(rows.sum())

    # -------------------------------------------------------------- the numbers

    def metrics(self) -> dict[str, float]:
        p = self.pose.numpy()
        out = {"MPJPE": p[0] / p[1], "PA-MPJPE": p[2] / p[3], "PVE": p[4] / p[5],
               "Accel": p[6] / max(p[7], 1.0)}
        for name, key in (("WA-MPJPE", "wa_mpjpe100"), ("W-MPJPE", "w_mpjpe100"),
                          ("RTE", "rte"), ("jitter", "jitter"),
                          ("jitter world", "jitter_world")):
            total, count = self.world[key]
            out[name] = total / count if count else float("nan")
        tp, fp, fn = (self.counts[k] for k in ("tp", "fp", "fn"))
        if tp + fp + fn:
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            out["P"], out["R"] = precision, recall
            out["F1"] = 2 * precision * recall / max(precision + recall, 1e-9)
        else:                                   # no contact head: unscored, forces ungated
            out["P"] = out["R"] = out["F1"] = float("nan")
        out |= (force_columns(np.concatenate(self.boards), np.concatenate(self.ours),
                              np.concatenate(self.mass_g), np.concatenate(self.subject_g))
                if self.ours else {c: float("nan") for c in _FORCE_COLUMNS})
        n = self.rnea[2]
        out["RNEA f bw"] = self.rnea[0] / n if n else float("nan")
        out["RNEA tau bw·m"] = self.rnea[1] / n if n else float("nan")
        return out


def force_columns(boards: np.ndarray, ours: np.ndarray, mass_g: np.ndarray,
                  subject_g: np.ndarray) -> dict[str, float]:
    """The force half of a row: ``(R, 4, 3)`` newtons on both sides, ``(R,)`` body weights (N).

    ``mass_g`` is the body weight ``ours`` was read in newtons with, ``subject_g`` the
    measured one.
    """
    measured, solved = np.linalg.norm(boards, axis=-1), np.linalg.norm(ours, axis=-1)
    bw = mass_g[:, None]
    vector = np.linalg.norm(ours - boards, axis=-1)
    vector_gt_mass = np.linalg.norm(ours / bw[..., None] - boards / subject_g[:, None, None],
                                    axis=-1)
    both = (measured > CONTACT_N) & (solved > CONTACT_N)
    cos = np.clip((boards * ours).sum(-1)[both] / (measured[both] * solved[both]), -1, 1)
    share = lambda m: m / np.maximum(np.nansum(m, 1, keepdims=True), 1e-6)  # noqa: E731
    pair = np.isfinite(solved) & np.isfinite(measured)
    return {
        "vec MAE bw%": 100.0 * np.nanmean(vector / bw),
        "vec RMSE bw%": 100.0 * np.sqrt(np.nanmean((vector / bw) ** 2)),
        "vec MAE bw% (GT mass)": (100.0 * np.nanmean(vector_gt_mass)
                                  if np.isfinite(vector_gt_mass).any() else float("nan")),
        "vec MAE N": np.nanmean(vector),
        **{f"{limb} MAE N": np.nanmean(vector[:, k])
           for k, limb in enumerate(("LH", "RH", "LF", "RF"))},
        "size MAE N": np.nanmean(np.abs(solved - measured)),
        "size RMSE N": np.sqrt(np.nanmean((solved - measured) ** 2)),
        "angle": float(np.degrees(np.arccos(cos)).mean()),
        "corr": float(np.corrcoef(solved[pair], measured[pair])[0, 1]),
        "share pp": 100.0 * float(np.nanmean(np.abs(share(solved) - share(measured)))),
    }


# ------------------------------------------------------------------ driver

def markdown(rows: dict[str, dict[str, float]], header: list[str]) -> str:
    cell = lambda name, v: ("nan" if not np.isfinite(v) else  # noqa: E731
                            f"{v:.3f}" if name in _RATES else f"{v:.2f}")
    lines = header + ["| method | " + " | ".join(COLUMNS) + " |",
                      "|" + "---|" * (len(COLUMNS) + 1)]
    for label, values in rows.items():
        lines.append(f"| {label} | " + " | ".join(cell(c, values[c]) for c in COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=Path,
                        default=_BVR / "peter" / "out_climb_wall_3_single")
    parser.add_argument("--gt", type=Path, default=_BVR / "peter" / "climb_wall_3_gt")
    parser.add_argument("--runs", nargs="*", default=[],
                        help="run names whose dumps sit in <clip>/predictions/<run>/")
    parser.add_argument("--legacy-dirs", action="store_true",
                        help="read <clip>/predictions_<run>/ instead")
    parser.add_argument("--sam3d", action="store_true",
                        help="add the raw per-frame SAM 3D Body fit as a pose-only row")
    parser.add_argument("--force-name", default="forces_sup.npz")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mass", choices=("optim", "subject"), default="optim",
                        help="the body weight our body-weight-unit forces are read in newtons "
                             "with: the reconstructed body's mass (default) or the "
                             "participant's measured one")
    parser.add_argument("--clips", type=int, default=None, help="score the first N clips only")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smplx-model", default=None,
                        help="SMPL-X npz (default: the one in configs/base.yaml)")
    parser.add_argument("--out", type=Path, default=None, help="markdown file to write")
    args = parser.parse_args()

    model_path = args.smplx_model or yaml.safe_load(
        (_ROOT / "configs" / "base.yaml").read_text())["model"]["smplx"]["model_path"]
    device = torch.device(args.device)
    body = Body(model_path, device)
    wrench = RootWrench(model_path, device)

    clips = clip_names(args.gt)[:args.clips]
    rows = {}
    if args.sam3d:
        from viewer.bodies import load_body
        body52 = load_body(device, model_path)
        rows["SAM 3D Body"] = (Accumulator(), ("sam3d", ""))
    rows |= {label: (Accumulator(), ("optim", directory))
             for label, directory in OPTIM_RUNS.items()}
    for run in args.runs:
        rows[run] = (Accumulator(), ("pred", f"predictions_{run}" if args.legacy_dirs
                                     else f"predictions/{run}"))
    gt_jitter, frames = [0.0, 0.0], 0

    for index, clip in enumerate(clips, start=1):
        ref = reference(args.gt, clip)
        ref["body_weight_n"] = GRAVITY * (ref["mass_kg"] if args.mass == "subject" else float(
            np.load(args.tree / clip / "human_optim" / "kindyn_1.npz",
                    allow_pickle=True)["total_mass"][0]))
        gt_joints, gt_verts = body.pose(ref["q"], ref["betas"])
        boards = board_forces(args.tree, clip, ref["n"])
        frames += int(ref["valid"].sum())
        if ref["valid"].sum() >= 4:
            values = compute_jitter(torch.as_tensor(ref["joints"][ref["valid"]],
                                                    dtype=torch.float32), fps=ref["fps"])
            gt_jitter[0] += float(np.asarray(values, np.float64).sum())
            gt_jitter[1] += float(len(values))
        ref["fk_check"] = float(np.linalg.norm(
            gt_joints.cpu().numpy().astype(np.float64) - ref["joints"],
            axis=-1)[ref["valid"]].mean() * 1e3)

        for label, (acc, (kind, directory)) in rows.items():
            path = args.tree / clip / directory
            if not path.exists():
                raise SystemExit(f"{clip}: {path} does not exist")
            if kind == "sam3d":
                row = sam3d_row(args.tree, clip, ref, body, body52)
            elif kind == "optim":
                row = optimisation_row(args.tree, clip, directory, ref, body)
            else:
                row = prediction_row(path, args.tree, clip, ref, body, wrench, args.threshold,
                                     args.force_name)
            valid = ref["valid"] & row["valid"]
            acc.add_pose(row, ref, gt_joints, gt_verts, valid)
            acc.add_contact(row["contact"], ref["contact"], valid)
            acc.add_force(boards, row["force_n"], ref["body_weight_n"], GRAVITY * ref["mass_kg"],
                          valid)
            acc.add_rnea(row, valid)
        print(f"[{index}/{len(clips)}] {clip}", flush=True)

    table = {label: acc.metrics() for label, (acc, _) in rows.items()}
    checks = ", ".join(f"{label} {np.mean(acc.fk_check):.2f} mm"
                       for label, (acc, _) in rows.items() if acc.fk_check)
    header = [
        f"climb_wall_3 GT rig, left camera: {len(clips)} trials, {frames} reference frames; "
        f"reference reposing check {ref['fk_check']:.2f} mm; FK-vs-stored joints check: {checks}",
        f"GT jitter floor {gt_jitter[0] / max(gt_jitter[1], 1.0):.1f}; "
        f"accel at the native fps ({ref['fps']:.0f})",
    ]
    text = markdown(table, header)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
