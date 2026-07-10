"""Train the contact head(s) on still-image and/or video contact datasets.

Slim and modular: config load/validate in ``contact/config.py``, model build in
``contact/model.py``, data pipeline in ``contact/data/collate.py``, the shared
forward in ``contact/engine.py``, per-target loss in ``contact/losses.py``,
metrics in ``contact/metrics.py``, checkpoint I/O in ``contact/checkpoint.py``,
and wandb/tensorboard logging in ``contact/tracking.py``. This file owns the
loop, scheduler, monitor-based best selection, and resume.

Usage::

    python scripts/train.py --config configs/damon_baseline.yaml
    python scripts/train.py --config configs/climbing_videos_joint.yaml
    python scripts/train.py --config configs/X.yaml --resume auto
    python scripts/train.py --config configs/X.yaml --resume output/<run>/last.pth
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from contact import checkpoint as ckpt_io
from contact.config import load_config
from contact.data.collate import batch_to_device, make_loaders
from contact.engine import forward_contact
from contact.losses import MultiTargetContactLoss
from contact.metrics import add_counts, contact_counts, prf1, zero_counts
from contact.model import build_model
from contact.tracking import RunLogger


# The per-target metrics a monitor may select (direction: all "max"; only
# ``val/loss`` is "min"). Mirrors the keys produced by ``contact.metrics.prf1``.
_MONITOR_METRICS = ("precision", "recall", "f1", "iou", "accuracy")


# -------------------------------------------------------------------- helpers

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


def _resume_config_diffs(saved: dict, current: dict) -> list[str]:
    """Identity-defining config diffs for an auto-resume candidate.

    Compared: ``model``, ``contact``, ``data.datasets``, ``data.sequence``,
    ``optim`` (minus ``epochs``) and ``output.monitor``. Allowed to differ across
    a resume (not compared): ``optim.epochs`` (extend training) and all
    ``logging.*``.
    """
    diffs: list[str] = []
    for section in ("model", "contact"):
        if saved.get(section) != current.get(section):
            diffs.append(f"  {section}: differs")
    for key in ("datasets", "sequence"):
        if (saved.get("data", {}) or {}).get(key) != (current.get("data", {}) or {}).get(key):
            diffs.append(f"  data.{key}: differs")
    s_optim = {k: v for k, v in (saved.get("optim") or {}).items() if k != "epochs"}
    c_optim = {k: v for k, v in (current.get("optim") or {}).items() if k != "epochs"}
    if s_optim != c_optim:
        diffs.append("  optim (excl. epochs): differs")
    if (saved.get("output", {}) or {}).get("monitor") != (current.get("output", {}) or {}).get("monitor"):
        diffs.append("  output.monitor: differs")
    return diffs


def _resolve_resume(cfg: dict, resume: str | None) -> Path | None:
    """Map ``--resume`` (``None`` | ``"auto"`` | path) to a checkpoint path.

    ``"auto"`` picks the newest ``{exp_name}_YYYYMMDD_HHMMSS/last.pth`` under
    ``output.dir`` — restricted to the *current experiment identity* (exact
    ``exp_name``, not a prefix) — and hard-fails if that run's saved config
    differs from the current one on any identity-defining section (see
    :func:`_resume_config_diffs`).
    """
    if resume is None:
        return None
    output_dir = Path(cfg["output"]["dir"])
    if resume == "auto":
        exp_name = cfg["output"]["exp_name"]
        stamped = re.compile(re.escape(exp_name) + r"_\d{8}_\d{6}$")
        candidates = sorted(
            (p for p in output_dir.glob(f"{exp_name}_*/last.pth")
             if stamped.match(p.parent.name)),
            key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(
                f"--resume auto: no '{exp_name}_YYYYMMDD_HHMMSS/last.pth' under {output_dir}")
        chosen = candidates[-1]
        saved_cfg_path = chosen.parent / "config.yaml"
        if saved_cfg_path.exists():
            saved_cfg = yaml.safe_load(saved_cfg_path.read_text()) or {}
            diffs = _resume_config_diffs(saved_cfg, cfg)
            if diffs:
                raise RuntimeError(
                    f"--resume auto selected {chosen} but its config differs from the "
                    f"current one on identity-defining sections:\n" + "\n".join(diffs)
                    + "\nOnly optim.epochs and logging.* may change across a resume; "
                    "start a fresh run or point --resume at an explicit checkpoint.")
        return chosen
    path = Path(resume)
    if not path.exists():
        raise FileNotFoundError(f"--resume: checkpoint {path} does not exist")
    return path


# -------------------------------------------------------------------- loop

class Trainer:
    def __init__(self, config_path: Path, device: str = "cuda", resume: str | None = None):
        self.cfg = load_config(config_path)
        self.device = device

        resume_ckpt = _resolve_resume(self.cfg, resume)
        if resume_ckpt is not None:
            self.out_dir = resume_ckpt.resolve().parent
            print(f"Resuming from {resume_ckpt}  (run dir {self.out_dir})")
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.out_dir = Path(self.cfg["output"]["dir"]) / f"{self.cfg['output']['exp_name']}_{stamp}"
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / "config.yaml").write_text(yaml.safe_dump(self.cfg, sort_keys=False))
            print(f"Output: {self.out_dir}")

        self.model, self.trainable_names = build_model(self.cfg, device)
        image_size = tuple(self.model.cfg.MODEL.IMAGE_SIZE)
        self.train_loader, self.val_loader, self.split_manifest = make_loaders(self.cfg, image_size)
        if resume_ckpt is None:
            (self.out_dir / "split_manifest.json").write_text(
                json.dumps(self.split_manifest, indent=2))

        self.loss_fn = MultiTargetContactLoss(self.cfg).to(device)
        self.targets = self.loss_fn.target_names
        primary = self.cfg["contact"]["primary_target"]
        self.primary = primary if primary in self.targets else self.targets[0]

        ocfg = self.cfg["optim"]
        self.grad_clip = float(self.cfg["loss"]["grad_clip"])
        self.optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=float(ocfg["lr"]), weight_decay=float(ocfg["weight_decay"]),
        )
        self.scheduler = _build_scheduler(self.optimizer, ocfg)
        self.epochs    = int(ocfg["epochs"])

        ofcfg = self.cfg["output"]
        self.log_freq  = int(ofcfg["log_freq"])
        self.val_freq  = int(ofcfg["val_freq"])
        self.save_freq = int(ofcfg["save_freq"])
        self.monitor   = str(ofcfg["monitor"])
        self._validate_monitor()
        self.monitor_mode = "min" if self.monitor == "val/loss" else "max"

        self.epoch        = 0
        self.global_step  = 0
        self.best_metric  = float("inf") if self.monitor_mode == "min" else float("-inf")
        self.wandb_run_id = None

        # ---- resume state (weights/optim/sched/RNG restored) before logger init ----
        if resume_ckpt is not None:
            state = ckpt_io.load(
                resume_ckpt, self.model, self.optimizer, self.scheduler,
                config=self.cfg, restore_rng=True, map_location=device)
            saved_manifest = state.get("split_manifest")
            if saved_manifest is not None and saved_manifest != self.split_manifest:
                raise RuntimeError(
                    "resume split-manifest drift: the checkpoint's train/val split no "
                    "longer matches what the current data produces — resuming would "
                    "train/validate on a different split.\n"
                    f"  checkpoint: {saved_manifest}\n  current:    {self.split_manifest}")
            if state["monitor"] != self.monitor:
                raise ValueError(
                    f"resume monitor mismatch: checkpoint monitored {state['monitor']!r} "
                    f"but this run monitors {self.monitor!r} — best_metric is not comparable")
            self.epoch        = state["epoch"] + 1
            self.global_step  = state["global_step"]
            self.best_metric  = state["best_metric"]
            self.wandb_run_id = state.get("wandb_run_id")
            print(f"Resumed at epoch {self.epoch}  step {self.global_step}  "
                  f"best {self.monitor}={self.best_metric:.4f}")

        self.logger = RunLogger(self.cfg, self.out_dir, self.out_dir.name,
                                resume_id=self.wandb_run_id)
        if self.wandb_run_id is None:
            self.wandb_run_id = self.logger.run_id   # persist the fresh id into checkpoints

        self._print_run_summary()

    # ---------------------------------------------------------------- utilities

    def _validate_monitor(self) -> None:
        """Accept only ``val/loss`` or ``val/{enabled_target}_{metric}``.

        Any other name (e.g. ``val/joint_loss``) is rejected up-front rather than
        crashing later in :meth:`_monitor_value` when the metric is looked up.
        """
        valid = {"val/loss"} | {
            f"val/{t}_{k}" for t in self.targets for k in _MONITOR_METRICS}
        if self.monitor not in valid:
            raise ValueError(
                f"output.monitor {self.monitor!r} is not a valid metric; choose one of "
                f"{sorted(valid)}")

    def _monitor_value(self, val: dict) -> float:
        rest = self.monitor.split("/", 1)[-1]
        if rest == "loss":
            return val["loss"]
        target, key = rest.split("_", 1)
        return val["metrics"][target][key]

    def _is_better(self, value: float) -> bool:
        return value > self.best_metric if self.monitor_mode == "max" else value < self.best_metric

    def _logits(self, contact: dict) -> dict:
        return {t: contact[f"{t}_logits"] for t in self.targets}

    def _print_run_summary(self) -> None:
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        fpb = int(self.cfg["data"]["frames_per_batch"])
        clip_len = int(self.cfg["data"]["sequence"]["frames_per_clip"])
        has_video = any(d["name"] == "climbing_videos" for d in self.cfg["data"]["datasets"])
        layout = (f"video: {max(1, fpb // clip_len)} clips x T={clip_len}"
                  if has_video else f"stills: {fpb} frames x T=1")
        print(f"Trainable params: {n_train:,} | targets={self.targets} primary={self.primary} | "
              f"monitor={self.monitor} ({self.monitor_mode})")
        print(f"Batch budget: frames_per_batch={fpb} -> {layout} | "
              f"train batches/epoch={len(self.train_loader)} val batches={len(self.val_loader)}")

    # ---------------------------------------------------------------- training

    def _train_epoch(self) -> dict:
        self.model.train()
        running_loss, n, epoch_frames, skipped = 0.0, 0, 0, 0
        counts = {t: zero_counts() for t in self.targets}
        window_frames, window_start = 0, time.perf_counter()
        pbar = tqdm(self.train_loader, desc=f"epoch {self.epoch}")
        for batch in pbar:
            batch  = batch_to_device(batch, self.device)
            frames = int(batch["img"].shape[0])
            epoch_frames += frames
            window_frames += frames

            logits = self._logits(forward_contact(self.model, batch))
            loss, parts = self.loss_fn(logits, batch["targets"])

            # A batch with zero active supervision (e.g. an all-invalid video
            # window) has zero gradient — but AdamW weight decay would still nudge
            # the weights. Skip the optimiser step entirely for those.
            active = sum(parts[t]["n_active"] for t in self.targets)
            self.optimizer.zero_grad(set_to_none=True)
            if active > 0:
                loss.backward()
                grad_norm = self._clip_grads()
                self.optimizer.step()
            else:
                skipped += 1
                grad_norm = 0.0

            running_loss += loss.item()
            n += 1
            batch_counts = {}
            for t in self.targets:
                tgt = batch["targets"][t]
                bc = contact_counts(logits[t].detach(), tgt["gt"], tgt["mask"])
                batch_counts[t] = bc
                add_counts(counts[t], bc)

            if self.global_step % self.log_freq == 0:
                now = time.perf_counter()
                fps = window_frames / max(now - window_start, 1e-9)
                window_frames, window_start = 0, now
                scalars = {
                    "train/loss": loss.item(),
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "train/grad_norm": grad_norm,
                    "train/frames_per_sec": fps,
                }
                for t in self.targets:
                    m = prf1(batch_counts[t])
                    scalars[f"train/{t}/focal"]     = parts[t]["focal"]
                    scalars[f"train/{t}/dice"]      = parts[t]["dice"]
                    scalars[f"train/{t}/sparsity"]  = parts[t]["sparsity"]
                    scalars[f"train/{t}/precision"] = m["precision"]
                    scalars[f"train/{t}/recall"]    = m["recall"]
                    scalars[f"train/{t}/f1"]        = m["f1"]
                    scalars[f"train/{t}/iou"]       = m["iou"]
                self.logger.log(scalars, self.global_step)

            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             f1=f"{prf1(batch_counts[self.primary])['f1']:.3f}")
            self.global_step += 1

        metrics = {t: prf1(counts[t]) for t in self.targets}
        return {"loss": running_loss / max(n, 1), "metrics": metrics,
                "frames": epoch_frames, "skipped": skipped}

    def _clip_grads(self) -> float:
        """Clip trainable grads; return the post-clip global grad norm."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        max_norm = self.grad_clip if self.grad_clip > 0 else float("inf")
        total = torch.nn.utils.clip_grad_norm_(params, max_norm).item()
        return min(total, self.grad_clip) if self.grad_clip > 0 else total

    @torch.no_grad()
    def _validate(self) -> dict:
        self.model.eval()
        running_loss, n = 0.0, 0
        counts = {t: zero_counts() for t in self.targets}
        for batch in tqdm(self.val_loader, desc="val"):
            batch  = batch_to_device(batch, self.device)
            logits = self._logits(forward_contact(self.model, batch))
            loss, _ = self.loss_fn(logits, batch["targets"])
            running_loss += loss.item()
            n += 1
            for t in self.targets:
                tgt = batch["targets"][t]
                add_counts(counts[t], contact_counts(logits[t], tgt["gt"], tgt["mask"]))
        return {"loss": running_loss / max(n, 1),
                "metrics": {t: prf1(counts[t]) for t in self.targets}}

    # ---------------------------------------------------------------- top-level

    def fit(self):
        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch
            if hasattr(self.train_loader, "set_epoch"):
                self.train_loader.set_epoch(epoch)

            t0 = time.perf_counter()
            t = self._train_epoch()
            epoch_time = time.perf_counter() - t0
            summary = "  ".join(
                f"{name}[f1={t['metrics'][name]['f1']:.3f} iou={t['metrics'][name]['iou']:.3f}]"
                for name in self.targets)
            skip_note = f"  [{t['skipped']} unsupervised batches skipped]" if t["skipped"] else ""
            print(f"epoch {epoch:3d}  train loss {t['loss']:.4f}  {summary}  "
                  f"({epoch_time:.1f}s, {t['frames'] / max(epoch_time, 1e-9):.1f} frames/s){skip_note}")

            val_metric = None
            if epoch % self.val_freq == 0:
                v = self._validate()
                vsummary = "  ".join(
                    f"{name}[f1={v['metrics'][name]['f1']:.3f} p={v['metrics'][name]['precision']:.3f} "
                    f"r={v['metrics'][name]['recall']:.3f}]" for name in self.targets)
                print(f"           val  loss {v['loss']:.4f}  {vsummary}")
                val_scalars = {"val/loss": v["loss"]}
                for name in self.targets:
                    for key, val in v["metrics"][name].items():
                        val_scalars[f"val/{name}/{key}"] = val
                self.logger.log(val_scalars, self.global_step)
                val_metric = self._monitor_value(v)

            self.logger.log({"train/epoch_time_sec": epoch_time,
                             "train/epoch_frames_per_sec": t["frames"] / max(epoch_time, 1e-9),
                             "train/skipped_batches": t["skipped"]},
                            self.global_step)

            self.scheduler.step()

            # Checkpoints saved AFTER scheduler.step so their state (weights, optim,
            # scheduler, RNG) reflects "ready to start epoch+1" — resume is seamless.
            if val_metric is not None and self._is_better(val_metric):
                self.best_metric = val_metric
                self._save("best.pth")
                print(f"           new best {self.monitor}={val_metric:.4f}")
            if self.save_freq > 0 and epoch > 0 and epoch % self.save_freq == 0:
                self._save(f"epoch_{epoch:04d}.pth")
            self._save("last.pth")

        self._save("final.pth")
        self.logger.close()

    def _save(self, name: str):
        path = self.out_dir / name
        ckpt_io.save(
            path, self.model, self.trainable_names,
            self.optimizer, self.scheduler,
            epoch=self.epoch, global_step=self.global_step,
            best_metric=self.best_metric, monitor=self.monitor,
            config=self.cfg, wandb_run_id=self.wandb_run_id,
            split_manifest=self.split_manifest,
        )
        size_mb = path.stat().st_size / 2**20
        print(f"  saved {name}  ({size_mb:.1f} MB)")


# -------------------------------------------------------------------- entrypoint

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=REPO / "configs" / "damon_baseline.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", type=str, default=None,
                   help="'auto' (newest */last.pth under output.dir) or a checkpoint path")
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    Trainer(args.config, device=args.device, resume=args.resume).fit()


if __name__ == "__main__":
    main()
