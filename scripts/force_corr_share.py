"""Force correlation and load-share error of prediction dumps against the corpus kindyn forces.

The two force numbers of the climb_wall_2 board comparison
(``BetterVideoReconstruction-dev/scripts/diagnostics/compare_climb_wall_2.py``), computed the
same way on the annotated corpus test scenes from a run's ``predictions/<scene>.npz`` dumps
(``scripts/predict_test.py``: whole scenes at the evaluation stride):

* the six kindyn groups are folded into the four limbs of the board rig — hand = hand,
  foot = toe + heel (vector sum) — and each limb's force SIZE is taken per frame;
* rows = every predicted (``covered``) person-frame the kindyn solve marks valid, pooled over
  all scenes and people;
* ``corr`` = Pearson correlation of the pooled per-limb sizes (prediction vs GT, all four
  limbs flattened together);
* ``share`` = each limb's share of the mean total force, in per cent, and the reported error is
  the mean absolute deviation from the GT shares in percentage points.

Both are unit-free (body weights here, newtons on the boards) and use ALL rows, in contact or
not, exactly like the board table.

    python scripts/force_corr_share.py output/<run> [output/<run> ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.climbing_videos import kindyn, scene as scene_io      # noqa: E402

#: Kindyn group indices of the four board limbs (LH, RH, L foot = toe + heel, R foot).
LIMBS = ((0,), (1,), (2, 4), (3, 5))
LIMB_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot")
DATASET_YAML = Path(__file__).resolve().parents[1] / "configs" / "datasets" / "climbing_videos.yaml"


def limb_sizes(forces: np.ndarray) -> np.ndarray:
    """``(R, 6, 3)`` group vectors -> ``(R, 4)`` limb force sizes."""
    return np.stack([np.linalg.norm(forces[:, list(g)].sum(1), axis=-1) for g in LIMBS], 1)


def pooled_sizes(run: Path, root: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Prediction and GT limb sizes ``(T, 4)`` pooled over the run's dumped test scenes."""
    pred, gt, scenes = [], [], 0
    for path in sorted((run / "predictions").glob("*.npz")):
        dump = np.load(path, allow_pickle=True)
        if "forces_world" not in dump.files:
            raise ValueError(f"{path.name}: the dump carries no forces")
        data = scene_io.load_scene(root, path.stem, "test", 1)
        object_ids = data["object_ids"]
        n = len(data["frame_indices"])
        forces = kindyn.load_forces(path.stem, data["human_dir"], object_ids, n)
        pred_forces = scene_io.rows_by_object_id(
            np.asarray(dump["forces_world"], np.float32), dump["object_ids"], object_ids,
            path.stem, "prediction dump")                                   # [P, N, 6, 3]
        covered = scene_io.rows_by_object_id(
            np.asarray(dump["covered"], bool), dump["object_ids"], object_ids,
            path.stem, "prediction dump")                                   # [P, N]
        rows = covered & forces["force_valid"] & np.isfinite(pred_forces).all(axis=(-1, -2))
        pred.append(limb_sizes(pred_forces[rows]))
        gt.append(limb_sizes(forces["force_gt"][rows]))
        scenes += 1
    return np.concatenate(pred), np.concatenate(gt), scenes


def corr_and_share(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Pearson correlation of the pooled sizes and the load-share error (pp) + the two share vectors."""
    corr = float(np.corrcoef(pred.reshape(-1), gt.reshape(-1))[0, 1])
    share = 100.0 * pred.mean(0) / pred.mean(0).sum()
    gt_share = 100.0 * gt.mean(0) / gt.mean(0).sum()
    return corr, float(np.abs(share - gt_share).mean()), share, gt_share


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--root", type=Path, default=None,
                        help="corpus root (default: the dataset yaml's)")
    args = parser.parse_args()
    root = args.root or Path(yaml.safe_load(DATASET_YAML.read_text())["root"])
    print(f"{'run':<44s} {'scenes':>6s} {'rows':>7s} {'corr':>6s} {'share pp':>8s}   shares pred | gt ({', '.join(LIMB_NAMES)})")
    for run in args.runs:
        pred, gt, scenes = pooled_sizes(run, root)
        corr, share_err, share, gt_share = corr_and_share(pred, gt)
        print(f"{run.name:<44s} {scenes:6d} {len(pred):7d} {corr:6.3f} {share_err:8.2f}   "
              + " ".join(f"{s:5.1f}" for s in share) + " | " + " ".join(f"{s:5.1f}" for s in gt_share))


if __name__ == "__main__":
    main()
