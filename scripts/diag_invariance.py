"""Invariance diagnostics of a trained model on the test clips (plan Phase 4).

    python scripts/diag_invariance.py --config configs/final.yaml \
        --checkpoint output_2/<run>/last.pth --json output_2/audits/invariance/<run>.json

Each row re-scores the SAME checkpoint on the SAME test clips under one perturbation of the
model's input, always against the unperturbed GT:

* ``none`` — the reference.
* ``reverse`` — the clip played backwards (timestamps re-based so they stay increasing); an
  offline smoother is reversal-equivariant, so nothing should change.
* ``shuffle:k`` — the per-frame CONTENT (token, camera, box) is randomly permuted inside blocks
  of ``k`` frames while the timestamps stay in place; ``shuffle:all`` permutes the whole clip.
  Scored two ways: ``shuffle:k`` maps every output back to its content's frame (a per-frame
  map is invariant, anything that reads the timestamp is not); ``shuffle:k/slot`` scores the
  output where it was computed, against the GT of that timestamp (an interpolating smoother
  stays close, a per-frame map is off by the displacement inside the block).
* ``decimate:k`` — every ``k``-th frame only, spacing channels updated; scored on the kept frames
  next to the reference restricted to the same frames (``decimate:k/ref``).
* ``window:s`` — the temporal attention half-width set to ``s`` seconds at test time.

Outputs are mapped back to the original frame order before scoring, so every sequence metric
(accel, jitter, motion correlations) is computed in real time order.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import build_datasets                        # noqa: E402
from data.collate import batch_to_device               # noqa: E402
from data.loaders import build_loaders                 # noqa: E402
from evaluate import ContactCurve                      # noqa: E402
from model.loss import KINDYN_GROUP_NAMES, build_losses  # noqa: E402
from train.config import signal_needs                  # noqa: E402
from train.predict import load_model                   # noqa: E402

REPORT = ("contact/f1", "contact/precision", "contact/recall", "pose/mpjpe", "pose/pa_mpjpe",
          "pose/accel", "pose/lifted_jitter", "force/mae", "force/angle_deg",
          "motion/vel_pearson", "motion/acc_pearson")
_KEEP_ORDER = ("frame_pos_sec",)


def _index(batch: dict, idx: torch.Tensor, n: int) -> dict:
    """Every entry with a leading frame axis of length ``n`` re-indexed by ``idx``."""
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == n:
            out[key] = value[idx.to(value.device)]
        elif isinstance(value, list) and len(value) == n:
            out[key] = [value[i] for i in idx.tolist()]
        else:
            out[key] = value
    return out


def _index_out(out, idx: torch.Tensor, n: int):
    if isinstance(out, dict):
        return {k: _index_out(v, idx, n) for k, v in out.items()}
    if isinstance(out, torch.Tensor) and out.dim() > 0 and out.shape[0] == n:
        return out[idx.to(out.device)]
    return out


def permutation(mode: str, n: int, generator: torch.Generator) -> torch.Tensor:
    if mode == "reverse":
        return torch.arange(n - 1, -1, -1)
    block = n if mode == "shuffle:all" else int(mode.split(":")[1])
    perm = torch.arange(n)
    for start in range(0, n, block):
        stop = min(start + block, n)
        perm[start:stop] = start + torch.randperm(stop - start, generator=generator)
    return perm


class Scorer:
    """Sums every loss's sufficient statistics + the contact threshold curve."""

    def __init__(self, losses, device, threshold: float):
        self.losses = losses
        self.stats = {loss.name: torch.zeros(len(loss.stat_names), dtype=torch.float64,
                                             device=device) for loss in losses}
        self.curve = ContactCurve((threshold,))
        self.threshold = threshold

    def add(self, out: dict, batch: dict) -> None:
        if out["contact"] is not None:
            self.curve(out, batch)
        for loss in self.losses:
            self.stats[loss.name] += loss(out, batch, train=False).stats.to(
                self.stats[loss.name].device, torch.float64)

    def metrics(self) -> dict[str, float]:
        metrics = {}
        for loss in self.losses:
            for key, value in loss.metrics(self.stats[loss.name]).items():
                metrics[f"{loss.metric_group}/{key}"] = float(value)
        return metrics


@torch.no_grad()
def score(model, loader, losses, device, modes: list[str], threshold: float,
          seed: int) -> dict[str, dict[str, float]]:
    scorers = {}
    for mode in modes:
        scorers[mode] = Scorer(losses, device, threshold)
        if mode.startswith("decimate"):
            scorers[mode + "/ref"] = Scorer(losses, device, threshold)
        if mode.startswith("shuffle"):
            scorers[mode + "/slot"] = Scorer(losses, device, threshold)
    generator = torch.Generator().manual_seed(seed)
    trained_window = model.refiner.temporal.window
    for batch in tqdm(loader, desc="clips"):
        batch = batch_to_device(batch, device)
        n = int(batch["seq_len"])
        assert batch["frame_pos_sec"].shape[0] == n, "one clip per batch"
        reference = None
        for mode in modes:
            model.refiner.temporal.window = trained_window
            if mode == "none":
                reference = model(batch)
                scorers[mode].add(reference, batch)
            elif mode == "reverse" or mode.startswith("shuffle"):
                perm = permutation(mode, n, generator)
                perturbed = _index(batch, perm, n)
                if mode == "reverse":
                    sec = batch["frame_pos_sec"]
                    perturbed["frame_pos_sec"] = sec[-1] - sec[perm]
                else:
                    for key in _KEEP_ORDER:
                        perturbed[key] = batch[key]
                out = model(perturbed)
                if mode != "reverse":
                    scorers[mode + "/slot"].add(out, batch)
                scorers[mode].add(_index_out(out, torch.argsort(perm), n), batch)
            elif mode.startswith("decimate"):
                step = int(mode.split(":")[1])
                idx = torch.arange(0, n, step)
                sub = _index(batch, idx, n)
                sub["seq_len"] = len(idx)
                scorers[mode].add(model(sub), sub)
                if reference is None:
                    reference = model(batch)
                scorers[mode + "/ref"].add(_index_out(reference, idx, n), sub)
            elif mode.startswith("window"):
                model.refiner.temporal.window = float(mode.split(":")[1])
                scorers[mode].add(model(batch), batch)
            else:
                raise ValueError(f"unknown mode {mode!r}")
    model.refiner.temporal.window = trained_window
    results = {}
    for mode, scorer in scorers.items():
        metrics = scorer.metrics()
        counts = scorer.curve.counts[0].sum(dim=0).tolist()
        if sum(counts) > 0:
            tp, fp, fn, _ = counts
            metrics["contact/f1"] = 2 * tp / max(2 * tp + fp + fn, 1e-8)
            metrics["contact/precision"] = tp / max(tp + fp, 1e-8)
            metrics["contact/recall"] = tp / max(tp + fn, 1e-8)
            for j, group in enumerate(KINDYN_GROUP_NAMES):
                tp, fp, fn, _ = scorer.curve.counts[0, j].tolist()
                metrics[f"contact/f1_{group}"] = 2 * tp / max(2 * tp + fp + fn, 1e-8)
                metrics[f"contact/precision_{group}"] = tp / max(tp + fp, 1e-8)
        results[mode] = metrics
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--modes", default="none,reverse,shuffle:2,shuffle:4,shuffle:all,"
                        "decimate:2,window:0.01,window:0.15")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--limit-scenes", type=int, default=None)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, cfg = load_model(args.config, args.checkpoint, args.device)
    if model.refiner is None:
        raise SystemExit("the diagnostics need a refiner build")
    _, test_sets = build_datasets(cfg, signal_needs(cfg), limit_scenes=args.limit_scenes)
    _, loader = build_loaders(cfg, [], test_sets)
    losses = build_losses(cfg, model, args.device)
    results = score(model, loader, losses, args.device, args.modes.split(","),
                    args.threshold, args.seed)

    columns = [c for c in REPORT if any(c in m for m in results.values())]
    print(f"\ncheckpoint {args.checkpoint}  trained window {model.refiner.temporal.window} s")
    print(f"{'mode':>16s} " + " ".join(f"{c.split('/')[1]:>12s}" for c in columns))
    for mode, metrics in results.items():
        print(f"{mode:>16s} " + " ".join(
            f"{metrics[c]:12.4f}" if c in metrics else f"{'-':>12s}" for c in columns))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"checkpoint": args.checkpoint, "results": results},
                                        indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
