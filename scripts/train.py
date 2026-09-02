"""Train the contact / force / motion branches on the climbing corpus.

    python scripts/train.py --config configs/allmod_rope_t60_gv.yaml
    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
        scripts/train.py --config configs/allmod_rope_t60_gv.yaml
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import build_datasets                        # noqa: E402
from data.loaders import build_loaders                 # noqa: E402
from model.build import build_model                    # noqa: E402
from model.loss import build_losses                    # noqa: E402
from train.config import load_config, signal_needs     # noqa: E402
from train.trainer import Trainer                      # noqa: E402


def resolve_resume(cfg: dict, resume: str | None) -> Path | None:
    """Map ``--resume`` (``None`` | ``"auto"`` | a path) to a checkpoint path."""
    if resume is None:
        return None
    if resume != "auto":
        path = Path(resume)
        if not path.exists():
            raise FileNotFoundError(f"--resume: {path} does not exist")
        return path
    exp_name = cfg["output"]["exp_name"]
    stamped = re.compile(re.escape(exp_name) + r"_\d{8}_\d{6}$")
    candidates = sorted(
        (p for p in Path(cfg["output"]["dir"]).glob(f"{exp_name}_*/last.pth")
         if stamped.match(p.parent.name)),
        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"--resume auto: no '{exp_name}_YYYYMMDD_HHMMSS/last.pth' under "
            f"{cfg['output']['dir']}")
    return candidates[-1]


def run_dir(cfg: dict, resume: Path | None, is_main: bool, distributed: bool) -> Path:
    """The run directory: the resumed one, or a fresh timestamped one on rank 0."""
    if resume is not None:
        return resume.resolve().parent
    out_dir = None
    if is_main:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(cfg["output"]["dir"]) / f"{cfg['output']['exp_name']}_{stamp}"
    if distributed:
        shared = [str(out_dir) if out_dir is not None else None]
        dist.broadcast_object_list(shared, src=0)
        out_dir = Path(shared[0])
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"Output: {out_dir}")
    if distributed:
        dist.barrier()
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=str, default=None,
                        help="'auto' (newest <exp>_*/last.pth) or a checkpoint path")
    parser.add_argument("--limit-scenes", type=int, default=None,
                        help="smoke runs: use only the first N scenes of each split")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = args.device
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = f"cuda:{local_rank}"

    try:
        cfg = load_config(args.config)
        torch.manual_seed(int(cfg["data"]["seed"]))
        torch.cuda.manual_seed_all(int(cfg["data"]["seed"]))

        resume = resolve_resume(cfg, args.resume)
        out_dir = run_dir(cfg, resume, rank == 0, world_size > 1)

        model = build_model(cfg, device)
        train_sets, test_sets = build_datasets(
            cfg, signal_needs(cfg), limit_scenes=args.limit_scenes)
        train_loader, test_loader = build_loaders(
            cfg, train_sets, test_sets, rank=rank, world_size=world_size)
        losses = build_losses(cfg, model, device)
        if rank == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"Trainable: {trainable:,} / {total:,} "
                  f"({100 * trainable / total:.2f}%)")
            print(f"Losses: {[loss.name for loss in losses]}")
        Trainer(cfg, model, losses, train_loader, test_loader, device,
                out_dir=out_dir, resume=resume).fit()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
