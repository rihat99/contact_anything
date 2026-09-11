"""The Gaussian front (plan C1): re-score one checkpoint over the refiner's smoothing width.

    python scripts/diag_sigma_sweep.py --config configs/final.yaml \
        --checkpoint output_2/<run>/last.pth --json output_2/audits/sigma_sweep/<run>.json

Overrides the refiner's ``pose_smooth_sec`` (and ``root_smooth_sec = ratio × pose_smooth_sec``,
the ratio of the config) at test time (``--checkpoint none`` = stage 1 + the Gaussian alone).
Besides the loss metrics each row reports the retained motion amplitude: the RMS world-joint
speed of the prediction over the GT's (22 body joints, valid frames).
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
from diag_invariance import Scorer                     # noqa: E402
from model.loss import build_losses                    # noqa: E402
from train.config import signal_needs                  # noqa: E402
from train.predict import load_model                   # noqa: E402

REPORT = ("pose/mpjpe", "pose/pa_mpjpe", "pose/accel", "pose/lifted_jitter", "pose/gt_jitter",
          "motion/vel_pearson", "motion/acc_pearson", "still/speed", "contact/f1")


def speed_sums(out: dict, batch: dict) -> torch.Tensor:
    """``[pred_sq, gt_sq]`` sums of squared world-joint speeds over valid adjacent frames."""
    valid = batch["frame_valid"].bool() & batch["smplx_valid"].bool()
    pred = out["smplx"]["joints_world"][:, :22].double()
    gt = batch["smplx_joints_world"][:, :22].double()
    dt = batch["frame_pos_sec"].double().diff().clamp(min=1e-3)
    pair = valid[1:] & valid[:-1]
    v_pred = (pred[1:] - pred[:-1])[pair] / dt[pair, None, None]
    v_gt = (gt[1:] - gt[:-1])[pair] / dt[pair, None, None]
    return torch.stack([(v_pred ** 2).sum(), (v_gt ** 2).sum()])


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sigmas", default="0.02,0.04,0.06,0.08,0.10,0.12,0.16")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    checkpoint = None if args.checkpoint.lower() == "none" else args.checkpoint
    model, cfg = load_model(args.config, checkpoint, args.device)
    refiner = model.refiner
    if refiner is None or refiner.learn_smoothing or refiner.pose_smooth_sec <= 0.0:
        raise SystemExit("the sweep needs a refiner with fixed, positive smoothing widths")
    ratio = refiner.root_smooth_sec / refiner.pose_smooth_sec
    _, test_sets = build_datasets(cfg, signal_needs(cfg), limit_scenes=args.limit_scenes)
    _, loader = build_loaders(cfg, [], test_sets)
    losses = build_losses(cfg, model, args.device)
    sigmas = [float(s) for s in args.sigmas.split(",")]
    scorers = {s: Scorer(losses, args.device, 0.5) for s in sigmas}
    speeds = {s: torch.zeros(2, dtype=torch.float64) for s in sigmas}
    for batch in tqdm(loader, desc="clips"):
        batch = batch_to_device(batch, args.device)
        for s in sigmas:
            refiner.pose_smooth_sec, refiner.root_smooth_sec = s, ratio * s
            out = model(batch)
            scorers[s].add(out, batch)
            speeds[s] += speed_sums(out, batch).cpu()

    results = {}
    for s in sigmas:
        metrics = scorers[s].metrics()
        counts = scorers[s].curve.counts[0].sum(dim=0).tolist()
        if sum(counts) > 0:
            tp, fp, fn, _ = counts
            metrics["contact/f1"] = 2 * tp / max(2 * tp + fp + fn, 1e-8)
        metrics["speed_ratio"] = float((speeds[s][0] / speeds[s][1].clamp(min=1e-12)).sqrt())
        results[f"{s:.3f}"] = metrics
    columns = [c for c in REPORT if any(c in m for m in results.values())] + ["speed_ratio"]
    print(f"\ncheckpoint {checkpoint or 'none'}  root/pose sigma ratio {ratio:.3f}")
    print(f"{'sigma_pose':>10s} " + " ".join(f"{c.split('/')[-1]:>12s}" for c in columns))
    for key, metrics in results.items():
        print(f"{key:>10s} " + " ".join(
            f"{metrics[c]:12.4f}" if c in metrics else f"{'-':>12s}" for c in columns))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"checkpoint": checkpoint, "config": str(args.config),
                                         "results": results}, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
