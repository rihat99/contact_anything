"""Evaluate a checkpoint on the annotated test scenes (one clip per scene/person).

    python scripts/evaluate.py --config configs/final/final_full.yaml \
        --checkpoint output_6/<run>/best.pth
    python scripts/evaluate.py --config configs/final/final_full.yaml \
        --checkpoint none            # the untrained (frozen-baseline) arm

Prints every ``loss_test/*`` term and ``metric_*/*`` metric the enabled losses report,
and — when the contact branch is on — a precision/recall/F1 threshold curve over the SIX
KINDYN GROUPS plus per-group scores at ``--threshold``, and, when the run predicts more
slots than that (``data.contact_set: frames35``), the same per-SLOT table. A group's
prediction is the max over its member slots and its label the fold of theirs (or the
manual six-group annotation when the test batch carries one) — see
:mod:`model.loss.contact`. The report is mirrored to ``<output.dir>/logs/<run>_eval.log``
(``untrained_eval.log`` for ``--checkpoint none``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import build_datasets                        # noqa: E402
from data.loaders import build_loaders                 # noqa: E402
from train.config import signal_needs                  # noqa: E402
from train.logger import tee_output                    # noqa: E402
from train.predict import load_model                   # noqa: E402
from train.trainer import evaluate_tests              # noqa: E402
from model.contact_frames import contact_set                    # noqa: E402
from model.loss import KINDYN_GROUP_NAMES, build_losses         # noqa: E402
from model.loss.contact import (CURVE_THRESHOLDS, fold_to_groups,  # noqa: E402
                                group_members)

GROUPS = KINDYN_GROUP_NAMES
CURVE = CURVE_THRESHOLDS
_EPS = 1e-8


class ContactCurve:
    """Confusion counts at several thresholds over the test split, on both levels.

    :attr:`counts` is the SIX-GROUP table (the headline scores, and the level the
    manual test labels live on); :attr:`slot_counts` the run's own K slots. Under
    ``kindyn6`` the fold is the identity and the two are equal.
    """

    def __init__(self, thresholds, slots=None):
        self.thresholds = tuple(thresholds)
        self.slots = slots or contact_set("kindyn6")
        self.members = group_members(self.slots)
        self.counts = torch.zeros(len(self.thresholds), len(GROUPS), 4, dtype=torch.float64)
        self.slot_counts = torch.zeros(len(self.thresholds), self.slots.count, 4,
                                       dtype=torch.float64)

    def __call__(self, out: dict, batch: dict) -> None:
        if out["contact"] is None:
            return
        probs = out["contact"]["probs"].detach().float().cpu()
        gt = batch["contact_gt"].detach().float().cpu()
        valid = batch["contact_valid"].detach().float().cpu()
        cpu_batch = {key: batch[key].detach().float().cpu()
                     for key in ("contact_gt_groups", "contact_valid_groups") if key in batch}
        group_probs, group_gt, group_valid = fold_to_groups(
            probs, gt, valid, self.members, cpu_batch)
        for table, score, truth, mask in (
                (self.counts, group_probs, group_gt > 0.5, group_valid > 0),
                (self.slot_counts, probs, gt > 0.5, valid > 0)):
            for i, threshold in enumerate(self.thresholds):
                pred = score > threshold
                for j, counts in enumerate(
                        (pred & truth & mask, pred & ~truth & mask,
                         ~pred & truth & mask, ~pred & ~truth & mask)):
                    table[i, :, j] += counts.sum(dim=0).to(torch.float64)

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
        self._table(f"per group at threshold {threshold}", "group", GROUPS,
                    self.counts[index])
        if self.slots.count != len(GROUPS):
            self._table(f"per {self.slots.name} slot at threshold {threshold}", "slot",
                        self.slots.slot_names, self.slot_counts[index])

    def _table(self, title: str, label: str, names, counts) -> None:
        width = max(12, max(len(name) for name in names))
        print(f"\n{title}")
        print(f"  {label:>{width}s} {'P':>7s} {'R':>7s} {'F1':>7s} {'pos':>8s}")
        for j, name in enumerate(names):
            tp, fp, fn, _ = counts[j].tolist()
            precision, recall, f1 = self._prf1(tp, fp, fn)
            print(f"  {name:>{width}s} {precision:7.4f} {recall:7.4f} {f1:7.4f} "
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
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the metrics as json (output.frozen_metrics format)")
    args = parser.parse_args()


    checkpoint = None if args.checkpoint.lower() == "none" else args.checkpoint
    model, cfg = load_model(args.config, checkpoint, args.device)
    run = "untrained" if checkpoint is None else Path(checkpoint).resolve().parent.name
    tee_output(Path(cfg["output"]["dir"]) / "logs" / f"{run}_eval.log")
    print(f"config: {args.config}   checkpoint: {checkpoint or 'none (untrained)'}")
    _, test_sets = build_datasets(cfg, signal_needs(cfg), limit_scenes=args.limit_scenes)
    _, tests = build_loaders(cfg, [], test_sets)
    losses = build_losses(cfg, model, args.device)

    curve = ContactCurve(sorted({*CURVE, float(args.threshold)}),
                         contact_set(cfg["data"]["contact_set"]))
    metrics = evaluate_tests(model, tests, losses, args.device,
                             hook=curve if model.has_contact else None)

    print(f"\ncheckpoint: {checkpoint or 'none (untrained)'}")
    for tag in sorted(metrics):
        print(f"  {tag:<44s} {metrics[tag]:.6f}")
    if model.has_contact:
        curve.report(float(args.threshold))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"config": str(args.config), "checkpoint": checkpoint or "none",
             "metrics": {tag: float(v) for tag, v in metrics.items()}}, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
