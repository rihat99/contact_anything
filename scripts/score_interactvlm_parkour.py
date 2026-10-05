"""Score InteractVLM's per-vertex contacts on the Parkour clips at the limb level.

The vertices are folded onto the four measured limbs exactly as
``trivial_baselines.py::contacts_interactvlm`` does (``lbs`` grouping: every SMPL vertex
belongs to its argmax skinning-weight joint; a limb is in contact when a tenth of its
vertices are above the file's threshold) and compared with the plates' own labels
``<clip>/contacts.npz``. Precision / recall / F1 are pooled over clips and limbs; the
always-in-contact predictor is printed as the reference. Numbers are reported over all
limb-frames and over the rows the rig actually measured (``captured``).

    python scripts/score_interactvlm_parkour.py --queries scene wall
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
PARKOUR = _ROOT.parent / "data" / "Parkour-dataset" / "processed"
OTHERS = PARKOUR / "others"
SMPL_NEUTRAL = _ROOT.parent / "BetterHuman" / "models" / "smpl" / "SMPL_NEUTRAL.npz"

LIMBS = ("left_hand", "right_hand", "left_foot", "right_foot")
LBS_JOINTS = {"left_hand": [20, 22], "right_hand": [21, 23],
              "left_foot": [7, 10], "right_foot": [8, 11]}
VERTEX_FRACTION = 0.10


def vertex_groups() -> dict[str, np.ndarray]:
    """Limb -> SMPL vertex indices, by argmax skinning weight."""
    owner = np.asarray(np.load(SMPL_NEUTRAL, allow_pickle=True)["weights"]).argmax(1)
    return {limb: np.flatnonzero(np.isin(owner, ids)) for limb, ids in LBS_JOINTS.items()}


def predicted(path: Path, groups: dict[str, np.ndarray], n: int) -> np.ndarray:
    """``(n, 4)`` limb contact from a ``contacts_vertex.npz``."""
    vertex = np.load(path, allow_pickle=True)
    above = np.asarray(vertex["contact_smpl"], np.float32) > float(vertex["threshold"])
    hit = np.stack([above[:, groups[limb]].mean(1) >= VERTEX_FRACTION for limb in LIMBS], 1)
    frames = np.asarray(vertex["frame_indices"], int)
    keep = frames < n
    out = np.zeros((n, 4), bool)
    out[frames[keep]] = hit[keep]
    return out


def truth(clip: str) -> tuple[np.ndarray, np.ndarray]:
    """``(contacts (N, 4), captured (N, 4))`` in the ``LIMBS`` order."""
    data = np.load(PARKOUR / clip / "contacts.npz")
    names = [str(x) for x in data["limbs"]]
    order = [names.index(limb) for limb in LIMBS]
    return (np.asarray(data["contacts"], bool)[:, order],
            np.asarray(data["captured"], bool)[:, order])


def scores(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float, int, int]:
    """``(precision, recall, f1, n_pred_positive, n_gt_positive)``."""
    tp = int((pred & gt).sum())
    precision = tp / max(int(pred.sum()), 1)
    recall = tp / max(int(gt.sum()), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1, int(pred.sum()), int(gt.sum())


def report(name: str, pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> None:
    precision, recall, f1, n_pred, n_gt = scores(pred[mask], gt[mask])
    print(f"{name:<34} P {precision:.4f}  R {recall:.4f}  F1 {f1:.4f}"
          f"  pred+ {n_pred}  gt+ {n_gt}  rows {int(mask.sum())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", nargs="+", default=["scene", "wall"])
    args = parser.parse_args()

    groups = vertex_groups()
    clips = sorted(d.name for d in (PARKOUR / "out").iterdir()
                   if d.is_dir() and not d.name.startswith("_"))
    gt_all, cap_all = [], []
    pred_all: dict[str, list[np.ndarray]] = {q: [] for q in args.queries}
    per_clip: dict[str, dict[str, tuple[float, float, float, int, int]]] = {}
    for clip in clips:
        contacts, captured = truth(clip)
        n = len(contacts)
        gt_all.append(contacts)
        cap_all.append(captured)
        per_clip[clip] = {}
        for query in args.queries:
            path = OTHERS / f"interactvlm_{query}" / clip / "contacts_vertex.npz"
            pred = predicted(path, groups, n)
            pred_all[query].append(pred)
            per_clip[clip][query] = scores(pred, contacts)

    gt = np.concatenate(gt_all)
    captured = np.concatenate(cap_all)
    every = np.ones_like(gt, bool)

    print(f"{len(clips)} clips, {len(gt)} frames, {gt.size} limb-frames, "
          f"gt positive rate {gt.mean():.4f}, captured rate {captured.mean():.4f}\n")

    for mask, label in ((every, "all limb-frames"), (captured, "captured limb-frames only")):
        print(f"--- pooled over clips and limbs, {label}")
        for query in args.queries:
            report(f"interactvlm_{query}", np.concatenate(pred_all[query]), gt, mask)
        report("always in contact", np.ones_like(gt), gt, mask)
        print()

    for query in args.queries:
        print(f"--- per limb, {query}, all limb-frames")
        pred = np.concatenate(pred_all[query])
        for i, limb in enumerate(LIMBS):
            column = np.zeros_like(gt, bool)
            column[:, i] = True
            report(f"  {limb}", pred, gt, column)
        print()

    for query in args.queries:
        print(f"--- per clip, {query} (all limb-frames)")
        for clip in clips:
            precision, recall, f1, n_pred, n_gt = per_clip[clip][query]
            print(f"  {clip:<12} P {precision:.3f}  R {recall:.3f}  F1 {f1:.3f}"
                  f"  pred+ {n_pred}  gt+ {n_gt}")
        print()


if __name__ == "__main__":
    main()
