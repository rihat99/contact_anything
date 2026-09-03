"""Evaluate a checkpoint on the annotated test scenes (one clip per scene/person).

    python scripts/evaluate.py --config configs/allmod_rope_t60_gv.yaml \
        --checkpoint output/<run>/best.pth
    python scripts/evaluate.py --config configs/allmod_rope_t60_gv.yaml \
        --checkpoint none            # the untrained (frozen-baseline) arm

Prints every ``loss_test/*`` term and ``metric_*/*`` metric the enabled losses report, and — when the contact
branch is on — a precision/recall/F1 threshold curve plus per-group scores at
``--threshold``. The report is mirrored to ``<output.dir>/logs/<run>_eval.log``
(``untrained_eval.log`` for ``--checkpoint none``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import build_datasets                        # noqa: E402
from data.loaders import build_loaders                 # noqa: E402
from train.config import signal_needs                  # noqa: E402
from train.logger import tee_output                    # noqa: E402
from train.predict import load_model                   # noqa: E402
from train.trainer import evaluate_losses              # noqa: E402
from model.loss import KINDYN_GROUP_NAMES, build_losses  # noqa: E402
from model.loss.contact import CURVE_THRESHOLDS       # noqa: E402

GROUPS = KINDYN_GROUP_NAMES
CURVE = CURVE_THRESHOLDS
_EPS = 1e-8


class ContactCurve:
    """Per-group confusion counts at several thresholds, over the test split."""

    def __init__(self, thresholds):
        self.thresholds = tuple(thresholds)
        self.counts = torch.zeros(len(self.thresholds), len(GROUPS), 4,
                                  dtype=torch.float64)

    def __call__(self, out: dict, batch: dict) -> None:
        if out["contact"] is None:
            return
        probs = out["contact"]["joint_probs"].detach().float().cpu()
        gt = batch["contact_gt"].detach().float().cpu() > 0.5
        valid = batch["contact_valid"].detach().float().cpu() > 0
        for i, threshold in enumerate(self.thresholds):
            pred = probs > threshold
            for j, counts in enumerate(
                    (pred & gt & valid, pred & ~gt & valid,
                     ~pred & gt & valid, ~pred & ~gt & valid)):
                self.counts[i, :, j] += counts.sum(dim=0).to(torch.float64)

    @staticmethod
    def _prf1(tp, fp, fn):
        precision = tp / (tp + fp + _EPS)
        recall = tp / (tp + fn + _EPS)
        return precision, recall, 2 * tp / (2 * tp + fp + fn + _EPS)

    def report(self, threshold: float) -> None:
        print("\nthreshold curve (micro over the six kindyn groups)")
        print(f"  {'thr':>5s} {'P':>7s} {'R':>7s} {'F1':>7s} {'TP':>9s} "
              f"{'FP':>9s} {'FN':>9s}")
        for i, value in enumerate(self.thresholds):
            tp, fp, fn, _ = self.counts[i].sum(dim=0).tolist()
            precision, recall, f1 = self._prf1(tp, fp, fn)
            print(f"  {value:5.2f} {precision:7.4f} {recall:7.4f} {f1:7.4f} "
                  f"{tp:9.0f} {fp:9.0f} {fn:9.0f}")
        if threshold not in self.thresholds:
            return
        index = self.thresholds.index(threshold)
        print(f"\nper group at threshold {threshold}")
        print(f"  {'group':>12s} {'P':>7s} {'R':>7s} {'F1':>7s} {'pos':>8s}")
        for j, name in enumerate(GROUPS):
            tp, fp, fn, _ = self.counts[index, j].tolist()
            precision, recall, f1 = self._prf1(tp, fp, fn)
            print(f"  {name:>12s} {precision:7.4f} {recall:7.4f} {f1:7.4f} "
                  f"{tp + fn:8.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="checkpoint path, or 'none' for the untrained model")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-scenes", type=int, default=None,
                        help="smoke runs: use only the first N test scenes")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    checkpoint = None if args.checkpoint.lower() == "none" else args.checkpoint
    model, cfg = load_model(args.config, checkpoint, args.device)
    run = "untrained" if checkpoint is None else Path(checkpoint).resolve().parent.name
    tee_output(Path(cfg["output"]["dir"]) / "logs" / f"{run}_eval.log")
    print(f"config: {args.config}   checkpoint: {checkpoint or 'none (untrained)'}")
    _, test_sets = build_datasets(cfg, signal_needs(cfg), limit_scenes=args.limit_scenes)
    _, test_loader = build_loaders(cfg, [], test_sets)
    losses = build_losses(cfg, model, args.device)

    curve = ContactCurve(sorted({*CURVE, float(args.threshold)}))
    metrics = evaluate_losses(model, test_loader, losses, args.device,
                              hook=curve if cfg["model"]["contact"]["enabled"] else None)

    print(f"\ncheckpoint: {checkpoint or 'none (untrained)'}")
    for tag in sorted(metrics):
        print(f"  {tag:<44s} {metrics[tag]:.6f}")
    if cfg["model"]["contact"]["enabled"]:
        curve.report(float(args.threshold))


if __name__ == "__main__":
    main()
