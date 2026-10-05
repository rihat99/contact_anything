"""Score our prediction dumps and the optimisation on the LAAS Parkour rig, ``score_rig.py`` style.

The Parkour dataset measures 16 Vicon joints and, per limb, the force of the floor plates
and the instrumented bar (28 clips, 25 fps). Rows are the same as on climb_wall_3
(``scripts/score_rig.py``): the optimisation tree ``human_optim`` (estimated contacts) and its
scene-free variant ``human_optim_scenefree`` (no scene terms, kindyn from kinematic_1) when
every clip has them — no measured-contact solve is scored anywhere — plus every dump named on
the command line
(``<clip>/predictions/<run>/``, the learned models and the trivial baselines alike).

Two things differ from the climbing rig, and both come from the truth living in the MoCap
world rather than ours:

* **pose** — no reference body, only 16 joints, each row's taken from its own 52-joint body at
  the 16 corresponding joints. The columns are the climbing table's: ``MPJPE`` (per frame,
  root-relative; the root is the hip midpoint because Vicon has no pelvis, and the whole clip
  is first turned into the MoCap axes by ONE rotation fitted on its joints), the paper's
  per-frame Procrustes ``PA-MPJPE``, and GVHMR's ``WA-MPJPE`` / ``W-MPJPE`` (100-frame chunks,
  aligned by a similarity on the whole chunk / on its first two frames).
* **forces** — the same per-clip rotation (the method's joints onto the Vicon ones,
  centroids removed) carries the method's world forces into the MoCap axes, so
  the vector columns (``vec MAE``, ``angle``) are filled instead of blank. Forces in newtons
  use the reconstructed body's mass, as on the climbing rig; the dataset states no subject
  mass, so ``vec MAE bw% (GT mass)`` is blank.

Contacts are scored against the plates' own labels (``processed/<clip>/contacts.npz``).
Limb-frames the rig never captured are NaN in the truth and drop out of every number.
``RNEA`` is exactly ``score_rig``'s: the root-wrench residual of the dumped pose under the
dumped forces (the optimisation rows report their stored residual).

    python scripts/score_parkour_rig.py --runs climbing_frames35 bedlam_frames35 \\
        baseline_equal baseline_rnea --out output_7/logs/parkour_rig.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
_BVR = _ROOT.parent / "BetterVideoReconstruction"
for _p in (_BVR, _BVR / "scripts", _BVR / "scripts" / "diagnostics"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.human_optim.boards import LIMBS                                                   # noqa: E402

from model.physics import GRAVITY, RootWrench                                                # noqa: E402
from score_rig import (Body, OPTIM_RUNS, _RATES, force_columns, optimisation_row,            # noqa: E402
                       prediction_row)
from utils.gvhmr_metrics import CHUNK_LENGTH, compute_jpe, first_align_joints, global_align_joints  # noqa: E402
from viewer.bodies import load_body                                                          # noqa: E402

PARKOUR = _ROOT.parent / "data" / "Parkour-dataset" / "processed"
#: Each Parkour joint and the joint of our 52-joint body standing in for it, and the three
#: pose alignments — the same definitions as BVR's ``compare_parkour_methods.py`` (not imported:
#: that module pulls in pip ``smplx``, which this environment does not have).
CORRESPONDENCE = {
    "l_hip": "left_hip", "l_knee": "left_knee", "l_ankle": "left_ankle", "l_toes": "left_foot",
    "r_hip": "right_hip", "r_knee": "right_knee", "r_ankle": "right_ankle", "r_toes": "right_foot",
    "l_shoulder": "left_shoulder", "l_elbow": "left_elbow", "l_wrist": "left_wrist",
    "l_fingers": "left_middle1",
    "r_shoulder": "right_shoulder", "r_elbow": "right_elbow", "r_wrist": "right_wrist",
    "r_fingers": "right_middle1",
}


def procrustes_mpjpe(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """The paper's pose metric: per-frame similarity fit, mean residual mm, and its scale."""
    residuals, scales = [], []
    for pred, gt in zip(predicted * 1e3, truth * 1e3):
        pred_centred, gt_centred = pred - pred.mean(0), gt - gt.mean(0)
        cross = pred_centred.T @ gt_centred
        u, _, vt = np.linalg.svd(cross)
        v = vt.T
        rotation = v @ np.diag([1.0, 1.0, np.linalg.det(v) * np.linalg.det(u)]) @ u.T
        scale = np.trace(rotation @ cross) / (pred_centred ** 2).sum()
        residuals.append(np.linalg.norm(pred_centred @ (scale * rotation).T - gt_centred, axis=1))
        scales.append(scale)
    return float(np.mean(residuals)), float(np.mean(scales))


def clip_rotation(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """One rotation for the whole clip taking a prediction's axes onto the MoCap world's."""
    pred = (predicted - predicted.mean(1, keepdims=True)).reshape(-1, 3)
    gt = (truth - truth.mean(1, keepdims=True)).reshape(-1, 3)
    u, _, vt = np.linalg.svd(pred.T @ gt)
    v = vt.T
    return v @ np.diag([1.0, 1.0, np.linalg.det(v) * np.linalg.det(u)]) @ u.T


def root_mpjpe(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Root-relative MPJPE (mm): one rotation for the clip, then the hip midpoint per frame."""
    turned = predicted @ clip_rotation(predicted, truth).T
    hips = [list(CORRESPONDENCE).index(n) for n in ("l_hip", "r_hip")]
    root = lambda j: j[:, hips].mean(1, keepdims=True)  # noqa: E731
    return float(np.linalg.norm(((turned - root(turned)) - (truth - root(truth))) * 1e3,
                                axis=-1).mean())


def world_mpjpe(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """GVHMR's WA-MPJPE and W-MPJPE (mm) over 100-frame chunks of the 16 joints."""
    wa, w = [], []
    for start in range(0, len(truth), CHUNK_LENGTH):
        gt = torch.as_tensor(truth[start:start + CHUNK_LENGTH], dtype=torch.float32)
        pred = torch.as_tensor(predicted[start:start + CHUNK_LENGTH], dtype=torch.float32)
        wa.append(compute_jpe(gt, global_align_joints(gt, pred)))
        w.append(compute_jpe(gt, first_align_joints(gt, pred)))
    return float(np.concatenate(wa).mean() * 1e3), float(np.concatenate(w).mean() * 1e3)


COLUMNS = ("MPJPE", "PA-MPJPE", "WA-MPJPE", "W-MPJPE", "F1", "P", "R", "vec MAE bw%", "vec RMSE bw%",
           "vec MAE bw% (GT mass)", "vec MAE N", "size MAE N", "size RMSE N", "angle", "corr",
           "share pp", "RNEA f bw", "RNEA tau bw·m")


def reference(clip: str) -> dict:
    """The Vicon joints, the plate contacts and the frame count of one clip."""
    vicon = np.load(PARKOUR / clip / "joints_3d.npz")
    names = [str(x) for x in vicon["joint_names"]]
    contacts = np.load(PARKOUR / clip / "contacts.npz")
    limbs = [str(x) for x in contacts["limbs"]]
    n = len(np.load(PARKOUR / "out" / clip / "geometry" / "transform.npz")["frame_indices"])
    joints = np.asarray(vicon["joints"], np.float64)[:n, [names.index(j) for j in CORRESPONDENCE]]
    return {"n": n, "fps": float(vicon["fps"]), "joints16": joints,
            "valid": np.isfinite(joints).all(axis=(1, 2)),
            "contact": np.asarray(contacts["contacts"], bool)[:n, [limbs.index(l) for l in LIMBS]],
            "q": np.zeros((n, 91), np.float32), "mass_kg": float("nan")}


def joints16(q: np.ndarray, betas: np.ndarray, body, ext: np.ndarray | None,
             joint_names: list[str], device) -> np.ndarray:
    """``(T, 16, 3)`` Parkour joints of a full ``q (T, 211)``; ``ext`` lifts a camera-frame q."""
    with torch.no_grad():
        shaped = body.with_shape(betas=torch.as_tensor(
            np.broadcast_to(betas, (len(q), 10)).copy(), device=device))
        joints = shaped.fk(torch.as_tensor(np.nan_to_num(q), device=device)).joint_pose_world[:, 1:, :3]
        if ext is not None:
            e = torch.as_tensor(ext, dtype=torch.float32, device=device)
            joints = torch.einsum("tji,tkj->tki", e[:, :3, :3], joints - e[:, None, :3, 3])
    columns = [joint_names.index(CORRESPONDENCE[name]) for name in CORRESPONDENCE]
    return joints[:, columns].cpu().numpy().astype(np.float64)


class Accumulator:
    def __init__(self) -> None:
        self.pose = {k: [] for k in ("MPJPE", "PA-MPJPE", "WA-MPJPE", "W-MPJPE")}
        self.counts = dict(tp=0, fp=0, fn=0)
        self.boards, self.ours, self.mass_g = [], [], []
        self.rnea = [0.0, 0.0, 0.0]

    def add(self, row: dict, pred16: np.ndarray, ref: dict, boards: np.ndarray,
            body_weight_n: float) -> None:
        keep = ref["valid"] & row["valid"] & np.isfinite(pred16).all(axis=(1, 2))
        if keep.sum() < 2:
            return
        pred, truth = pred16[keep], ref["joints16"][keep]
        wa, w = world_mpjpe(pred, truth)
        for key, value in (("MPJPE", root_mpjpe(pred, truth)), ("PA-MPJPE", procrustes_mpjpe(pred, truth)[0]),
                           ("WA-MPJPE", wa), ("W-MPJPE", w)):
            self.pose[key].append(value)
        rotation = clip_rotation(pred16[keep], ref["joints16"][keep])
        if row["contact"] is not None:
            got, want = row["contact"][keep], ref["contact"][keep]
            self.counts["tp"] += int((got & want).sum())
            self.counts["fp"] += int((got & ~want).sum())
            self.counts["fn"] += int((~got & want).sum())
        turned = row["force_n"][keep] @ rotation.T
        self.boards.append(boards[keep])
        self.ours.append(np.where(np.isfinite(boards[keep]), turned, np.nan))
        self.mass_g.append(np.full(int(keep.sum()), body_weight_n))
        force, torque = row["rnea"]
        rows = row["rnea_rows"] & keep & np.isfinite(force)
        self.rnea[0] += float(force[rows].sum())
        self.rnea[1] += float(torque[rows].sum())
        self.rnea[2] += float(rows.sum())

    def metrics(self) -> dict[str, float]:
        out = {key: float(np.mean(values)) for key, values in self.pose.items()}
        tp, fp, fn = (self.counts[k] for k in ("tp", "fp", "fn"))
        if tp + fp + fn:
            out["P"], out["R"] = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            out["F1"] = 2 * out["P"] * out["R"] / max(out["P"] + out["R"], 1e-9)
        else:
            out["P"] = out["R"] = out["F1"] = float("nan")
        mass_g = np.concatenate(self.mass_g)
        out |= force_columns(np.concatenate(self.boards), np.concatenate(self.ours), mass_g,
                             np.full_like(mass_g, np.nan))
        n = self.rnea[2]
        out["RNEA f bw"] = self.rnea[0] / n if n else float("nan")
        out["RNEA tau bw·m"] = self.rnea[1] / n if n else float("nan")
        return out


def markdown(rows: dict[str, dict[str, float]], header: list[str]) -> str:
    cell = lambda name, v: ("nan" if not np.isfinite(v) else  # noqa: E731
                            f"{v:.3f}" if name in _RATES else f"{v:.2f}")
    lines = header + ["| method | " + " | ".join(COLUMNS) + " |", "|" + "---|" * (len(COLUMNS) + 1)]
    for label, values in rows.items():
        lines.append(f"| {label} | " + " | ".join(cell(c, values[c]) for c in COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=Path, default=PARKOUR / "out")
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--force-name", default="forces_sup.npz")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--clips", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    model_path = yaml.safe_load((_ROOT / "configs" / "base.yaml").read_text())["model"]["smplx"]["model_path"]
    device = torch.device(args.device)
    body22, body52 = Body(model_path, device), load_body(device, model_path)
    wrench = RootWrench(model_path, device)
    clips = sorted(d.name for d in args.tree.iterdir()
                   if (d / "human_optim" / "kindyn_1.npz").is_file())[:args.clips]
    joint_names = [str(x) for x in np.load(
        args.tree / clips[0] / "human_optim" / "kindyn_1.npz", allow_pickle=True)["joint_names"]]
    runs = {label: directory for label, directory in OPTIM_RUNS.items()
            if "GT contact" not in label} | {"optimisation (scene-free)": "human_optim_scenefree"}
    rows = {label: (Accumulator(), ("optim", d)) for label, d in runs.items()
            if all((args.tree / c / d / "kindyn_1.npz").is_file() for c in clips)}
    for run in args.runs:
        rows[run] = (Accumulator(), ("pred", f"predictions/{run}"))
    frames = 0

    for index, clip in enumerate(clips, start=1):
        ref = reference(clip)
        n = ref["n"]
        ext = np.asarray(np.load(args.tree / clip / "geometry" / "transform.npz")["extrinsics"],
                         np.float32)[:n]
        mass = float(np.load(args.tree / clip / "human_optim" / "kindyn_1.npz",
                             allow_pickle=True)["total_mass"][0])
        ref["body_weight_n"] = GRAVITY * mass
        sensor = np.load(args.tree / clip / "human_optim" / "sensor_forces.npz", allow_pickle=True)
        channels = [str(x) for x in sensor["limbs"]]
        boards = np.asarray(sensor["reaction_world"], np.float64)[:n, [channels.index(l) for l in LIMBS]]
        frames += int(ref["valid"].sum())
        for label, (acc, (kind, directory)) in rows.items():
            path = args.tree / clip / directory
            if kind == "optim":
                row = optimisation_row(args.tree, clip, directory, ref, body22)
                kindyn = np.load(path / "kindyn_1.npz", allow_pickle=True)
                pred16 = joints16(np.asarray(kindyn["q"][0][:n], np.float32),
                                  np.asarray(kindyn["betas"][0], np.float32), body52, None,
                                  joint_names, device)
            else:
                row = prediction_row(path, args.tree, clip, ref, body22, wrench, args.threshold,
                                     args.force_name)
                dump = np.load(path / "smplx.npz", allow_pickle=True)
                q = np.asarray(dump["q_cam"][0][:n], np.float32)
                betas = np.nan_to_num(np.asarray(dump["betas"][0][:n], np.float32))
                pred16 = joints16(q, betas, body52, ext, joint_names, device)
            acc.add(row, pred16, ref, boards, ref["body_weight_n"])
        print(f"[{index}/{len(clips)}] {clip}", flush=True)

    table = {label: acc.metrics() for label, (acc, _) in rows.items()}
    header = [f"LAAS Parkour rig: {len(clips)} clips, {frames} reference frames; forces rotated "
              "into the MoCap world by each clip's joint alignment; newtons use the reconstructed "
              "body's mass (no subject mass is published)"]
    text = markdown(table, header)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
