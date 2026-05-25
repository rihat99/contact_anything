"""Train the contact head on DAMON.

Slim and modular: model build in ``train/model.py``, data pipeline in
``train/data.py``, loss in ``train/losses.py``, checkpoint I/O in
``train/checkpoint.py``. This file owns the loop, scheduler, and logs.

Usage::

    python train/train.py                       # default config
    python train/train.py --config train/config.yaml
    python train/train.py --resume train/output/contact_2026..._best.pth
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from train import checkpoint as ckpt_io
from train.data import batch_to_device, make_loaders
from train.losses import ContactLoss
from train.model import build_model


# -------------------------------------------------------------------- helpers

def _setup_output(cfg: dict, config_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg["output"]["dir"]) / f"{cfg['output']['exp_name']}_{stamp}"
    (out_dir / "tensorboard").mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out_dir / "config.yaml")
    print(f"Output: {out_dir}")
    return out_dir


def _build_scheduler(optimizer, optim_cfg):
    epochs  = int(optim_cfg["epochs"])
    warmup  = int(optim_cfg.get("warmup_epochs", 0))
    lr_min  = float(optim_cfg.get("lr_min", 0.0))

    main = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup, 1), eta_min=lr_min,
    )
    if warmup <= 0:
        return main
    warm = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warm, main], milestones=[warmup],
    )


@torch.no_grad()
def _metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    preds = (torch.sigmoid(logits) > 0.5)
    gt    = targets.bool()
    tp = (preds & gt).float().sum()
    fp = (preds & ~gt).float().sum()
    fn = (~preds & gt).float().sum()
    tn = (~preds & ~gt).float().sum()
    eps = 1e-8
    return {
        "accuracy":  ((tp + tn) / (tp + tn + fp + fn + eps)).item(),
        "precision": (tp / (tp + fp + eps)).item(),
        "recall":    (tp / (tp + fn + eps)).item(),
        "f1":        (2 * tp / (2 * tp + fp + fn + eps)).item(),
        "iou":       (tp / (tp + fp + fn + eps)).item(),
    }


def _forward(model, batch: dict) -> torch.Tensor:
    """Single forward through SAM-3D-Body. Returns contact_logits [B, V]."""
    model._initialize_batch(batch)
    out = model.forward_step(batch, decoder_type="body")
    if out.get("contact") is None:
        raise RuntimeError("model produced no contact output — check DO_CONTACT_TOKENS.")
    return out["contact"]["contact_logits"]


# -------------------------------------------------------------------- loop

class Trainer:
    def __init__(self, config_path: Path, device: str = "cuda"):
        self.cfg = yaml.safe_load(config_path.read_text())
        self.device = device
        self.out_dir = _setup_output(self.cfg, config_path)
        self.writer  = SummaryWriter(self.out_dir / "tensorboard")

        self.model, self.trainable_names = build_model(self.cfg, device)
        # Use the model's actual input resolution (set by checkpoint config).
        image_size = tuple(self.model.cfg.MODEL.IMAGE_SIZE)
        self.train_loader, self.val_loader = make_loaders(self.cfg, image_size)

        lcfg = self.cfg["loss"]
        self.loss_fn = ContactLoss(
            focal_alpha     = float(lcfg["focal_alpha"]),
            focal_gamma     = float(lcfg["focal_gamma"]),
            focal_weight    = float(lcfg["focal_weight"]),
            dice_weight     = float(lcfg["dice_weight"]),
            dice_eps        = float(lcfg["dice_eps"]),
            sparsity_weight = float(lcfg["sparsity_weight"]),
        )

        ocfg = self.cfg["optim"]
        self.grad_clip = float(ocfg["grad_clip"])
        self.optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=float(ocfg["lr"]), weight_decay=float(ocfg["weight_decay"]),
        )
        self.scheduler = _build_scheduler(self.optimizer, ocfg)
        self.epochs    = int(ocfg["epochs"])

        ofcfg = self.cfg["output"]
        self.log_freq  = int(ofcfg.get("log_freq", 10))
        self.val_freq  = int(ofcfg.get("val_freq", 1))
        self.save_freq = int(ofcfg.get("save_freq", 5))

        self.epoch       = 0
        self.global_step = 0
        self.best_val    = float("inf")

    # ---------------------------------------------------------------- training

    def _train_epoch(self) -> dict:
        self.model.train()
        running = {"loss": 0.0, "iou": 0.0, "f1": 0.0}
        n = 0
        pbar = tqdm(self.train_loader, desc=f"epoch {self.epoch}")
        for batch in pbar:
            batch = batch_to_device(batch, self.device)
            logits = _forward(self.model, batch)
            target = batch["contact"]
            loss, parts = self.loss_fn(logits, target)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in self.model.parameters() if p.requires_grad),
                    self.grad_clip,
                )
            self.optimizer.step()

            m = _metrics(logits.detach(), target)
            running["loss"] += loss.item()
            running["iou"]  += m["iou"]
            running["f1"]   += m["f1"]
            n += 1

            if self.global_step % self.log_freq == 0:
                self.writer.add_scalar("train/loss",       loss.item(), self.global_step)
                self.writer.add_scalar("train/focal_bce",  parts["focal_bce"], self.global_step)
                self.writer.add_scalar("train/dice",       parts["dice"], self.global_step)
                self.writer.add_scalar("train/sparsity",   parts["sparsity"], self.global_step)
                self.writer.add_scalar("train/iou",        m["iou"], self.global_step)
                self.writer.add_scalar("train/f1",         m["f1"], self.global_step)
                self.writer.add_scalar("train/lr",
                                       self.optimizer.param_groups[0]["lr"],
                                       self.global_step)
            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             iou=f"{m['iou']:.3f}",
                             f1=f"{m['f1']:.3f}")
            self.global_step += 1
        return {k: v / max(n, 1) for k, v in running.items()}

    @torch.no_grad()
    def _validate(self) -> dict:
        self.model.eval()
        agg = {"loss": 0.0, "focal_bce": 0.0, "dice": 0.0, "sparsity": 0.0,
               "iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0,
               "accuracy": 0.0}
        n = 0
        for batch in tqdm(self.val_loader, desc="val"):
            batch  = batch_to_device(batch, self.device)
            logits = _forward(self.model, batch)
            target = batch["contact"]
            loss, parts = self.loss_fn(logits, target)
            m = _metrics(logits, target)
            agg["loss"]      += loss.item()
            agg["focal_bce"] += parts["focal_bce"]
            agg["dice"]      += parts["dice"]
            agg["sparsity"]  += parts["sparsity"]
            for k in ("iou", "f1", "precision", "recall", "accuracy"):
                agg[k] += m[k]
            n += 1
        return {k: v / max(n, 1) for k, v in agg.items()}

    # ---------------------------------------------------------------- top-level

    def fit(self, resume: str | None = None):
        if resume:
            print(f"Resuming from {resume}")
            state = ckpt_io.load(resume, self.model, self.optimizer, self.scheduler)
            self.epoch       = state["epoch"] + 1
            self.global_step = state["global_step"]
            self.best_val    = state["best_val"]

        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch
            t = self._train_epoch()
            print(f"epoch {epoch:3d}  train loss {t['loss']:.4f}  "
                  f"iou {t['iou']:.4f}  f1 {t['f1']:.4f}")

            if epoch % self.val_freq == 0:
                v = self._validate()
                print(f"           val  loss {v['loss']:.4f}  "
                      f"iou {v['iou']:.4f}  f1 {v['f1']:.4f}  "
                      f"prec {v['precision']:.4f}  rec {v['recall']:.4f}")
                for k, x in v.items():
                    self.writer.add_scalar(f"val/{k}", x, epoch)
                if v["loss"] < self.best_val:
                    self.best_val = v["loss"]
                    self._save("best.pth")

            if self.save_freq > 0 and epoch > 0 and epoch % self.save_freq == 0:
                self._save(f"epoch_{epoch:04d}.pth")

            self.scheduler.step()

        self._save("final.pth")
        self.writer.close()

    def _save(self, name: str):
        path = self.out_dir / name
        ckpt_io.save(
            path, self.model, self.trainable_names,
            self.optimizer, self.scheduler,
            self.epoch, self.global_step, self.best_val,
        )
        size_mb = path.stat().st_size / 2**20
        print(f"  saved {name}  ({size_mb:.1f} MB)")


# -------------------------------------------------------------------- entrypoint

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=REPO / "train" / "config.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    Trainer(args.config, device=args.device).fit(resume=args.resume)


if __name__ == "__main__":
    main()
