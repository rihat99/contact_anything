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
    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
        scripts/train.py --config configs/climbing_videos_joint.yaml
    python scripts/train.py --config configs/X.yaml --resume auto
    python scripts/train.py --config configs/X.yaml --resume output/<run>/last.pth
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from contact import checkpoint as ckpt_io
from contact.config import load_config
from contact.data.collate import batch_to_device, make_loaders
from contact.engine import forward_contact
from contact.losses import MultiTargetContactLoss, ddp_global_mean_term
from contact.metrics import (
    add_counts,
    contact_counts,
    contact_counts_per_dim,
    prf1,
    zero_counts,
)
from contact.model import build_model
from contact.targets import TargetSpec
from contact.tracking import RunLogger


# The per-target metrics a monitor may select (direction: all "max"; only
# ``val/loss`` is "min"). Mirrors the keys produced by ``contact.metrics.prf1``.
_MONITOR_METRICS = ("precision", "recall", "f1", "f2", "iou", "accuracy")


class _ContactForward(nn.Module):
    """Give DDP a conventional ``forward`` around SAM-3D-Body's step API."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, batch: dict) -> dict:
        return forward_contact(self.model, batch)


class _NullLogger:
    """No-op logger used by non-zero distributed ranks."""

    run_id = None

    def log(self, scalars: dict, step: int) -> None:
        pass

    def close(self) -> None:
        pass


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
    for key in ("datasets", "sequence", "eval_split"):
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
    def __init__(
        self,
        config_path: Path,
        device: str = "cuda",
        resume: str | None = None,
        *,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
    ):
        self.cfg = load_config(config_path)
        self.device = device
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_rank = int(local_rank)
        self.is_main = self.rank == 0
        self.distributed = self.world_size > 1

        resume_ckpt = _resolve_resume(self.cfg, resume)
        if resume_ckpt is not None:
            self.out_dir = resume_ckpt.resolve().parent
            if self.is_main:
                print(f"Resuming from {resume_ckpt}  (run dir {self.out_dir})")
        else:
            out_dir = None
            if self.is_main:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_dir = Path(self.cfg["output"]["dir"]) / f"{self.cfg['output']['exp_name']}_{stamp}"
            if self.distributed:
                shared = [str(out_dir) if out_dir is not None else None]
                dist.broadcast_object_list(shared, src=0)
                out_dir = Path(shared[0])
            self.out_dir = out_dir
            if self.is_main:
                self.out_dir.mkdir(parents=True, exist_ok=True)
                (self.out_dir / "config.yaml").write_text(yaml.safe_dump(self.cfg, sort_keys=False))
                print(f"Output: {self.out_dir}")
            if self.distributed:
                dist.barrier()

        self.model, self.trainable_names = build_model(self.cfg, device)
        warm_state = None
        warm_path = self.cfg["model"].get("init_contact_checkpoint")
        if resume_ckpt is None and warm_path:
            warm_state = ckpt_io.initialize_common_contact(
                warm_path, self.model, config=self.cfg, map_location=device)
            if self.is_main:
                print(
                    f"Warm-started {len(warm_state['warm_start_loaded_names'])} contact "
                    f"parameters from {warm_path}; initialized "
                    f"{len(warm_state['warm_start_new_names'])} temporal parameters")
        image_size = tuple(self.model.cfg.MODEL.IMAGE_SIZE)
        self.train_loader, self.eval_loader, self.split_manifest = make_loaders(
            self.cfg,
            image_size,
            manifest=(warm_state or {}).get("split_manifest"),
            distributed_rank=self.rank,
            distributed_world_size=self.world_size,
        )
        if resume_ckpt is None and self.is_main:
            (self.out_dir / "split_manifest.json").write_text(
                json.dumps(self.split_manifest, indent=2))

        self.loss_fn = MultiTargetContactLoss(self.cfg).to(device)
        self.targets = self.loss_fn.target_names
        target_spec = TargetSpec.from_config(self.cfg)
        # Named per-output reporting is useful for compact semantic heads. Avoid
        # creating thousands of metric streams for a vertex target.
        self.output_names = (
            {"joint": target_spec.joint_names}
            if "joint" in self.targets and target_spec.joint_dims <= 32 else {}
        )
        primary = self.cfg["contact"]["primary_target"]
        self.primary = primary if primary in self.targets else self.targets[0]
        self.eval_split = str(self.cfg["data"]["eval_split"])

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
        self.monitor_mode = "min" if self.monitor.endswith("/loss") else "max"

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
                    "resume split-manifest drift: the checkpoint's train/eval split no "
                    "longer matches what the current data produces — resuming would "
                    "train/evaluate on a different split.\n"
                    f"  checkpoint: {saved_manifest}\n  current:    {self.split_manifest}")
            if state["monitor"] != self.monitor:
                raise ValueError(
                    f"resume monitor mismatch: checkpoint monitored {state['monitor']!r} "
                    f"but this run monitors {self.monitor!r} — best_metric is not comparable")
            self.epoch        = state["epoch"] + 1
            self.global_step  = state["global_step"]
            self.best_metric  = state["best_metric"]
            self.wandb_run_id = state.get("wandb_run_id")
            if self.is_main:
                print(f"Resumed at epoch {self.epoch}  step {self.global_step}  "
                      f"best {self.monitor}={self.best_metric:.4f}")

        forward_module = _ContactForward(self.model)
        if self.distributed:
            self.forward_module = DistributedDataParallel(
                forward_module,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
            )
        else:
            self.forward_module = forward_module

        if self.is_main:
            self.logger = RunLogger(self.cfg, self.out_dir, self.out_dir.name,
                                    resume_id=self.wandb_run_id)
            if self.wandb_run_id is None:
                self.wandb_run_id = self.logger.run_id
        else:
            self.logger = _NullLogger()

        if self.distributed:
            shared_id = [self.wandb_run_id if self.is_main else None]
            dist.broadcast_object_list(shared_id, src=0)
            self.wandb_run_id = shared_id[0]

        if self.is_main:
            self._print_run_summary()

    # ---------------------------------------------------------------- utilities

    def _validate_monitor(self) -> None:
        """Accept only the configured eval prefix and an available metric.

        Any other name (e.g. ``val/joint_loss``) is rejected up-front rather than
        crashing later in :meth:`_monitor_value` when the metric is looked up.
        """
        valid = {f"{self.eval_split}/loss"} | {
            f"{self.eval_split}/{t}_{k}"
            for t in self.targets for k in _MONITOR_METRICS
        }
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

    def _reduce_epoch_stats(
        self,
        running_loss: float,
        n: int,
        frames: int,
        skipped: int,
        counts: dict[str, dict[str, int]],
    ) -> tuple[float, int, int, int, dict[str, dict[str, int]]]:
        """Sum epoch statistics across ranks for globally correct metrics."""
        if not self.distributed:
            return running_loss, n, frames, skipped, counts
        count_keys = ("tp", "fp", "fn", "tn")
        values = [float(running_loss), float(n), float(frames), float(skipped)]
        for target in self.targets:
            values.extend(float(counts[target][key]) for key in count_keys)
        packed = torch.tensor(values, dtype=torch.float64, device=self.device)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        values = packed.cpu().tolist()
        reduced_counts = {target: zero_counts() for target in self.targets}
        offset = 4
        for target in self.targets:
            for key in count_keys:
                reduced_counts[target][key] = int(values[offset])
                offset += 1
        return values[0], int(values[1]), int(values[2]), int(values[3]), reduced_counts

    def _reduce_output_counts(
        self, counts: dict[str, list[dict[str, int]]],
    ) -> dict[str, list[dict[str, int]]]:
        """All-reduce compact per-output confusion matrices across ranks."""
        if not self.distributed or not counts:
            return counts
        count_keys = ("tp", "fp", "fn", "tn")
        layout = [(target, index) for target, values in counts.items()
                  for index in range(len(values))]
        packed = torch.tensor(
            [float(counts[target][index][key])
             for target, index in layout for key in count_keys],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        values = packed.cpu().tolist()
        reduced = {target: [zero_counts() for _ in target_counts]
                   for target, target_counts in counts.items()}
        offset = 0
        for target, index in layout:
            for key in count_keys:
                reduced[target][index][key] = int(values[offset])
                offset += 1
        return reduced

    def _ddp_weighted_loss(self, loss: torch.Tensor, parts: dict) -> tuple[torch.Tensor, bool]:
        """Make locally normalized target losses equal the global masked mean.

        DDP averages gradients across ranks. Each target exposes its additive
        weighted numerator; dividing it by the all-reduced confidence mass (and
        cancelling DDP's average) gives the exact global weighted-loss gradient,
        including when a rank has mass below one or zero.
        """
        if not self.distributed:
            active = any(parts[t]["weight_mass"] > 0 for t in self.targets)
            return loss, active
        local_mass = torch.tensor(
            [parts[t]["weight_mass"] for t in self.targets],
            dtype=torch.float64,
            device=self.device,
        )
        global_mass = local_mass.clone()
        dist.all_reduce(global_mass, op=dist.ReduceOp.SUM)
        scaled = None
        for i, target in enumerate(self.targets):
            term = ddp_global_mean_term(
                parts[target]["weighted_numerator_tensor"],
                global_mass[i],
                self.world_size,
            )
            scaled = term if scaled is None else scaled + term
        return scaled, bool(global_mass.sum().item() > 0)

    def _print_run_summary(self) -> None:
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        fpb = int(self.cfg["data"]["frames_per_batch"])
        clip_len = int(self.cfg["data"]["sequence"]["frames_per_clip"])
        has_video = any(d["name"] == "climbing_videos" for d in self.cfg["data"]["datasets"])
        layout = (f"video: {max(1, fpb // clip_len)} clips x T={clip_len}"
                  if has_video else f"stills: {fpb} frames x T=1")
        print(f"Trainable params: {n_train:,} | targets={self.targets} primary={self.primary} | "
              f"monitor={self.monitor} ({self.monitor_mode})")
        print(f"Batch budget: {fpb} frames/rank x {self.world_size} rank(s) "
              f"= {fpb * self.world_size} global frames -> {layout} per rank | "
              f"train batches/epoch={len(self.train_loader)} "
              f"{self.eval_split} batches={len(self.eval_loader)}")

    # ---------------------------------------------------------------- training

    def _train_epoch(self) -> dict:
        self.model.train()
        running_loss, n, epoch_frames, skipped = 0.0, 0, 0, 0
        counts = {t: zero_counts() for t in self.targets}
        output_counts = {
            target: [zero_counts() for _ in names]
            for target, names in self.output_names.items()
        }
        window_frames, window_start = 0, time.perf_counter()
        pbar = tqdm(self.train_loader, desc=f"epoch {self.epoch}", disable=not self.is_main)
        for batch in pbar:
            batch  = batch_to_device(batch, self.device)
            frames = int(batch["img"].shape[0])
            epoch_frames += frames
            window_frames += frames

            logits = self._logits(self.forward_module(batch))
            loss, parts = self.loss_fn(logits, batch["targets"])

            # A batch with zero active supervision (e.g. an all-invalid video
            # window) has zero gradient — but AdamW weight decay would still nudge
            # the weights. Skip the optimiser step entirely for those.
            loss, active = self._ddp_weighted_loss(loss, parts)
            finite = torch.tensor(
                int(bool(torch.isfinite(loss).item())), device=self.device,
                dtype=torch.int32)
            if self.distributed:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not bool(finite.item()):
                raise FloatingPointError(
                    f"non-finite loss at epoch={self.epoch} global_step={self.global_step}; "
                    "optimizer step was not executed")
            self.optimizer.zero_grad(set_to_none=True)
            if active:
                loss.backward()
                grad_norm = self._clip_grads()
                if not math.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"non-finite gradient norm at epoch={self.epoch} "
                        f"global_step={self.global_step}; optimizer step was not executed")
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
                if t in output_counts:
                    for accumulated, current in zip(
                        output_counts[t],
                        contact_counts_per_dim(
                            logits[t].detach(), tgt["gt"], tgt["mask"]),
                    ):
                        add_counts(accumulated, current)

            if self.global_step % self.log_freq == 0:
                now = time.perf_counter()
                fps = self.world_size * window_frames / max(now - window_start, 1e-9)
                window_frames, window_start = 0, now
                scalars = {
                    "train/loss": loss.item(),
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "train/grad_norm": grad_norm,
                    "train/frames_per_sec": fps,
                }
                for t in self.targets:
                    m = prf1(batch_counts[t])
                    for component in ("focal", "dice", "sparsity"):
                        if component in parts[t]:
                            scalars[f"train/{t}/{component}"] = parts[t][component]
                    scalars[f"train/{t}/precision"] = m["precision"]
                    scalars[f"train/{t}/recall"]    = m["recall"]
                    scalars[f"train/{t}/f1"]        = m["f1"]
                    scalars[f"train/{t}/f2"]        = m["f2"]
                    scalars[f"train/{t}/iou"]       = m["iou"]
                    if t in self.output_names:
                        tgt = batch["targets"][t]
                        current_dims = contact_counts_per_dim(
                            logits[t].detach(), tgt["gt"], tgt["mask"])
                        for output_name, current in zip(self.output_names[t], current_dims):
                            output_metrics = prf1(current)
                            for key in ("precision", "recall", "f1", "f2"):
                                scalars[f"train/{t}/{output_name}/{key}"] = output_metrics[key]
                self.logger.log(scalars, self.global_step)

            if self.is_main:
                pbar.set_postfix(loss=f"{loss.item():.3f}",
                                 f1=f"{prf1(batch_counts[self.primary])['f1']:.3f}")
            self.global_step += 1

        running_loss, n, epoch_frames, skipped, counts = self._reduce_epoch_stats(
            running_loss, n, epoch_frames, skipped, counts)
        output_counts = self._reduce_output_counts(output_counts)
        metrics = {t: prf1(counts[t]) for t in self.targets}
        per_output = {
            target: {
                name: prf1(value)
                for name, value in zip(self.output_names[target], target_counts)
            }
            for target, target_counts in output_counts.items()
        }
        return {"loss": running_loss / max(n, 1), "metrics": metrics,
                "per_output": per_output, "frames": epoch_frames, "skipped": skipped}

    def _clip_grads(self) -> float:
        """Clip trainable grads; return the post-clip global grad norm."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        max_norm = self.grad_clip if self.grad_clip > 0 else float("inf")
        total = torch.nn.utils.clip_grad_norm_(params, max_norm).item()
        return min(total, self.grad_clip) if self.grad_clip > 0 else total

    @torch.no_grad()
    def _evaluate(self) -> dict:
        self.model.eval()
        running_loss, n = 0.0, 0
        counts = {t: zero_counts() for t in self.targets}
        output_counts = {
            target: [zero_counts() for _ in names]
            for target, names in self.output_names.items()
        }
        for batch in tqdm(
            self.eval_loader, desc=self.eval_split, disable=not self.is_main,
        ):
            batch  = batch_to_device(batch, self.device)
            logits = self._logits(self.forward_module(batch))
            loss, _ = self.loss_fn(logits, batch["targets"])
            running_loss += loss.item()
            n += 1
            for t in self.targets:
                tgt = batch["targets"][t]
                add_counts(counts[t], contact_counts(logits[t], tgt["gt"], tgt["mask"]))
                if t in output_counts:
                    for accumulated, current in zip(
                        output_counts[t],
                        contact_counts_per_dim(logits[t], tgt["gt"], tgt["mask"]),
                    ):
                        add_counts(accumulated, current)
        running_loss, n, _, _, counts = self._reduce_epoch_stats(
            running_loss, n, 0, 0, counts)
        output_counts = self._reduce_output_counts(output_counts)
        return {"loss": running_loss / max(n, 1),
                "metrics": {t: prf1(counts[t]) for t in self.targets},
                "per_output": {
                    target: {
                        name: prf1(value)
                        for name, value in zip(self.output_names[target], target_counts)
                    }
                    for target, target_counts in output_counts.items()
                }}

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
                f"{name}[f1={t['metrics'][name]['f1']:.3f} f2={t['metrics'][name]['f2']:.3f} "
                f"iou={t['metrics'][name]['iou']:.3f}]"
                for name in self.targets)
            skip_note = f"  [{t['skipped']} unsupervised batches skipped]" if t["skipped"] else ""
            if self.is_main:
                print(f"epoch {epoch:3d}  train loss {t['loss']:.4f}  {summary}  "
                      f"({epoch_time:.1f}s, {t['frames'] / max(epoch_time, 1e-9):.1f} frames/s){skip_note}")

            val_metric = None
            if epoch % self.val_freq == 0:
                v = self._evaluate()
                vsummary = "  ".join(
                    f"{name}[f1={v['metrics'][name]['f1']:.3f} f2={v['metrics'][name]['f2']:.3f} "
                    f"p={v['metrics'][name]['precision']:.3f} "
                    f"r={v['metrics'][name]['recall']:.3f}]" for name in self.targets)
                if self.is_main:
                    print(f"           {self.eval_split:<4s} loss {v['loss']:.4f}  {vsummary}")
                val_scalars = {f"{self.eval_split}/loss": v["loss"]}
                for name in self.targets:
                    for key, val in v["metrics"][name].items():
                        val_scalars[f"{self.eval_split}/{name}/{key}"] = val
                    for output_name, output_metrics in v["per_output"].get(name, {}).items():
                        for key, val in output_metrics.items():
                            val_scalars[
                                f"{self.eval_split}/{name}/{output_name}/{key}"
                            ] = val
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
                if self.is_main:
                    self._save("best.pth")
                    print(f"           new best {self.monitor}={val_metric:.4f}")
            if (self.is_main and self.save_freq > 0 and epoch > 0
                    and epoch % self.save_freq == 0):
                self._save(f"epoch_{epoch:04d}.pth")
            if self.is_main:
                self._save("last.pth")
            if self.distributed:
                dist.barrier()

        if self.is_main:
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

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = args.device
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("multi-process training requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = f"cuda:{local_rank}"

    try:
        Trainer(
            args.config,
            device=device,
            resume=args.resume,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        ).fit()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
