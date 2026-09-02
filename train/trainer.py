"""Generic training loop over the :class:`~model.loss.Loss` interface.

The trainer knows nothing about contact, force, motion or pose. Every
supervision term arrives as a ``(numerator, mass)`` pair, and the total loss is

    sum over losses, over terms:  weight_scale(epoch) * numerator * W / max(M, 1)

where ``M`` is the term's mass summed over ranks and ``W`` the world size — the
factor that cancels DDP's gradient averaging, so the averaged gradient equals
the single-process global weighted mean exactly. Masses for every term of every
loss are packed into ONE float64 vector and all-reduced once per step. A batch
whose terms all have zero global mass carries no gradient, so its optimizer
step is skipped (AdamW's decay would otherwise still move the weights).

Evaluation is stats-based: each loss returns an additive float64 sufficient-
statistics vector, summed over batches, all-reduced once, and turned into
metrics by the loss itself. Metric tags are ``{split}/{loss.name}/{metric}``.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from data.collate import batch_to_device
from data.loaders import set_epoch
from model.loss import Loss
from train import checkpoint as ckpt_io
from train.config import monitor_mode
from train.logger import Logger
from utils.distributed import ddp_global_mean_term, is_distributed, rank, world_size


def build_scheduler(optimizer: torch.optim.Optimizer, optim_cfg: dict):
    """Linear warmup for ``warmup_epochs`` then cosine annealing to ``lr_min``."""
    epochs = int(optim_cfg["epochs"])
    warmup = int(optim_cfg["warmup_epochs"])
    lr_min = float(optim_cfg["lr_min"])

    def factor_for(base_lr: float):
        floor = lr_min / base_lr            # every group anneals to the same lr_min

        def factor(epoch: int) -> float:
            if epoch < warmup:
                return 0.01 + 0.99 * epoch / warmup
            progress = min((epoch - warmup) / max(epochs - warmup, 1), 1.0)
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return factor

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, [factor_for(float(g["lr"])) for g in optimizer.param_groups])


@torch.no_grad()
def evaluate_losses(module, loader, losses: Sequence[Loss], device,
                    *, distributed: bool = False, is_main: bool = True,
                    hook=None) -> dict:
    """Accumulate every loss's sufficient statistics over ``loader`` -> metrics.

    Each loss's float64 ``stats`` vector is summed over batches and all-reduced
    once; ``metrics()`` turns the global sums into the reported values, tagged
    ``test/{loss.name}/{metric}``. ``test/loss`` is the sum over every term of
    its global mass-weighted mean (un-ramped: ``weight_scale`` is a train-time
    warm-up only). ``hook(out, batch)`` sees every forward for callers that need
    raw predictions (the threshold curve).
    """
    stats = {loss.name: torch.zeros(len(loss.stat_names), dtype=torch.float64,
                                    device=device)
             for loss in losses}
    # Config-derived layout: a rank whose test shard is empty still reduces
    # buffers of the same shape as every other rank.
    keys = [(loss.name, term) for loss in losses for term in loss.term_names]
    term_sums: dict[tuple[str, str], list[float]] = {k: [0.0, 0.0] for k in keys}
    for batch in tqdm(loader, desc="test", disable=not is_main):
        batch = batch_to_device(batch, device)
        out = module(batch)
        if hook is not None:
            hook(out, batch)
        for loss in losses:
            result = loss(out, batch, train=False)
            stats[loss.name] += result.stats.to(device, torch.float64)
            for term, value in result.terms.items():
                acc = term_sums[(loss.name, term)]
                acc[0] += float(value.numerator.detach())
                acc[1] += float(value.mass)

    packed = torch.tensor([term_sums[k] for k in keys] or [[0.0, 0.0]],
                          dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        for name in stats:
            dist.all_reduce(stats[name], op=dist.ReduceOp.SUM)

    metrics: dict[str, float] = {
        "test/loss": float(sum(num / max(mass, 1.0)
                               for num, mass in packed.tolist()[:len(keys)]))
    }
    for loss in losses:
        for key, value in loss.metrics(stats[loss.name]).items():
            metrics[f"test/{loss.name}/{key}"] = float(value)
    return metrics


class Trainer:
    """Fit a model against a list of losses, evaluating on the test split each epoch.

    :param cfg: resolved run config.
    :param model: the model to train (already on ``device``).
    :param losses: the enabled losses, in ``build_losses`` order.
    :param train_loader: clip loader over the train scenes.
    :param test_loader: one-clip-per-(scene, person) loader over the test scenes.
    :param device: training device.
    :param out_dir: run directory (checkpoints + tensorboard).
    :param resume: checkpoint to restore weights/optimizer/scheduler/counters from.
    """

    _FT_PREFIXES = ("head_pose_ft_proj.", "head_camera_ft_proj.")

    def __init__(
        self,
        cfg: dict,
        model: torch.nn.Module,
        losses: Sequence[Loss],
        train_loader,
        test_loader,
        device: torch.device | str,
        *,
        out_dir: str | Path,
        resume: Optional[str | Path] = None,
    ):
        self.cfg = cfg
        self.model = model
        self.losses = list(losses)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.out_dir = Path(out_dir)

        self.distributed = is_distributed()
        self.world_size = world_size()
        self.rank = rank()
        self.is_main = self.rank == 0

        optim_cfg = cfg["optim"]
        self.epochs = int(optim_cfg["epochs"])
        self.grad_clip = float(optim_cfg["grad_clip"])
        self.optimizer = self._build_optimizer(optim_cfg)
        self.scheduler = build_scheduler(self.optimizer, optim_cfg)

        out_cfg = cfg["output"]
        self.log_freq = int(out_cfg["log_freq"])
        self.save_freq = int(out_cfg["save_freq"])
        self.monitor = str(out_cfg["monitor"])
        self.monitor_mode = monitor_mode(self.monitor)

        self.epoch = 0
        self.step = 0
        self.best = float("inf") if self.monitor_mode == "min" else float("-inf")
        if resume is not None:
            state = ckpt_io.load(resume, self.model, self.optimizer, self.scheduler,
                                 map_location=str(device))
            self.epoch = int(state["epoch"]) + 1
            self.step = int(state["step"])
            self.best = float(state["best"])
            if self.is_main:
                print(f"Resumed from {resume}: epoch {self.epoch}, step {self.step}, "
                      f"best {self.monitor}={self.best:.4f}")

        self.module = self.model
        if self.distributed:
            self.module = DistributedDataParallel(
                self.model,
                device_ids=[torch.device(device).index],
                output_device=torch.device(device).index,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        self.logger = Logger(self.out_dir, enabled=self.is_main)

    # ------------------------------------------------------------------ setup

    def _build_optimizer(self, optim_cfg: dict) -> torch.optim.Optimizer:
        """AdamW; the fine-tuned head copies form their own lr-scaled group."""
        trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        base = [p for n, p in trainable if not n.startswith(self._FT_PREFIXES)]
        head = [p for n, p in trainable if n.startswith(self._FT_PREFIXES)]
        groups = [{"params": base}]
        if head:
            groups.append({
                "params": head,
                "lr": float(optim_cfg["lr"]) * float(optim_cfg["head_lr_scale"]),
            })
        return torch.optim.AdamW(
            groups, lr=float(optim_cfg["lr"]),
            weight_decay=float(optim_cfg["weight_decay"]))

    # ------------------------------------------------------------- reductions

    @staticmethod
    def _layout(results: dict) -> list[tuple[str, str]]:
        """``(loss name, term name)`` pairs in a rank-stable order."""
        return [(name, term)
                for name, result in results.items()
                for term in sorted(result.terms)]

    def _global_masses(self, results: dict, layout) -> torch.Tensor:
        """All-reduced mass of every term, in ``layout`` order."""
        masses = torch.tensor(
            [float(results[name].terms[term].mass) for name, term in layout],
            dtype=torch.float64, device=self.device)
        if self.distributed:
            dist.all_reduce(masses, op=dist.ReduceOp.SUM)
        return masses

    def _total(self, results: dict, layout, masses: torch.Tensor,
               scales: dict) -> torch.Tensor:
        total = None
        for (name, term), mass in zip(layout, masses):
            contribution = scales[name] * ddp_global_mean_term(
                results[name].terms[term].numerator, mass)
            total = contribution if total is None else total + contribution
        return total

    def _clip_grads(self) -> float:
        """Clip trainable grads to ``optim.grad_clip``; return the raw total norm.

        The finiteness check must run on the RAW norm: an inf/NaN raw makes
        ``clip_grad_norm_`` scale every gradient by 0/NaN while the capped value
        ``min(inf, grad_clip)`` still looks perfectly finite.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        cap = self.grad_clip if self.grad_clip > 0 else float("inf")
        raw = float(torch.nn.utils.clip_grad_norm_(params, cap))
        if not math.isfinite(raw):
            raise FloatingPointError(
                f"non-finite raw gradient norm ({raw}) at epoch={self.epoch} "
                f"step={self.step}; the optimizer step was not executed")
        return raw

    # ------------------------------------------------------------------ train

    def train_epoch(self) -> float:
        """One pass over the train loader; returns the mean step loss."""
        self.model.train()
        scales = {loss.name: loss.weight_scale(self.epoch) for loss in self.losses}
        running, steps, skipped = 0.0, 0, 0
        window_frames, window_start = 0, time.perf_counter()
        pbar = tqdm(self.train_loader, desc=f"epoch {self.epoch}", disable=not self.is_main)
        for batch in pbar:
            batch = batch_to_device(batch, self.device)
            window_frames += int(batch["bbox_center"].shape[0])

            out = self.module(batch)
            results = {loss.name: loss(out, batch, train=True) for loss in self.losses}
            layout = self._layout(results)
            masses = self._global_masses(results, layout)
            total = self._total(results, layout, masses, scales)

            finite = torch.tensor(int(bool(torch.isfinite(total).item())),
                                  device=self.device, dtype=torch.int32)
            if self.distributed:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not bool(finite.item()):
                raise FloatingPointError(
                    f"non-finite loss at epoch={self.epoch} step={self.step}; "
                    "the optimizer step was not executed")

            self.optimizer.zero_grad(set_to_none=True)
            active = bool(masses.sum().item() > 0)
            if active:
                total.backward()
                grad_norm = self._clip_grads()
                self.optimizer.step()
            else:
                skipped += 1
                grad_norm = 0.0

            running += float(total.item())
            steps += 1
            if self.step % self.log_freq == 0:
                now = time.perf_counter()
                scalars = {
                    "train/loss": float(total.item()),
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "train/grad_norm": grad_norm,
                    "train/frames_per_sec": (
                        self.world_size * window_frames / max(now - window_start, 1e-9)),
                }
                window_frames, window_start = 0, now
                for name, result in results.items():
                    for term, value in result.terms.items():
                        scalars[f"train/{name}/{term}"] = (
                            float(value.numerator.detach()) / value.mass
                            if value.mass > 0 else 0.0)
                    for key, value in result.scalars.items():
                        scalars[f"train/{name}/{key}"] = float(value)
                self.logger.log(scalars, self.step)

            if self.is_main:
                postfix = {"loss": f"{float(total.item()):.3f}"}
                if "contact" in results:
                    contact = next(l for l in self.losses if l.name == "contact")
                    postfix["f1"] = f"{contact.metrics(results['contact'].stats)['f1']:.3f}"
                pbar.set_postfix(**postfix)
            self.step += 1

        if skipped and self.is_main:
            print(f"  [{skipped} batches without supervision — optimizer step skipped]")
        return running / max(steps, 1)

    # ------------------------------------------------------------------- eval

    def evaluate(self) -> dict:
        """Full-scene test evaluation; returns the flat ``test/*`` scalar dict."""
        self.model.eval()
        return evaluate_losses(self.module, self.test_loader, self.losses, self.device,
                               distributed=self.distributed, is_main=self.is_main)

    # -------------------------------------------------------------------- fit

    def fit(self) -> None:
        """Train for ``optim.epochs``, evaluating and checkpointing every epoch."""
        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch
            set_epoch(self.train_loader, epoch)

            start = time.perf_counter()
            train_loss = self.train_epoch()
            elapsed = time.perf_counter() - start
            metrics = self.evaluate()
            self.scheduler.step()

            if self.is_main:
                print(f"epoch {epoch:3d}  train loss {train_loss:.4f}  "
                      f"({elapsed:.1f}s)   test loss {metrics['test/loss']:.4f}")
                for loss in self.losses:
                    prefix = f"test/{loss.name}/"
                    body = "  ".join(
                        f"{tag[len(prefix):]} {value:.4f}"
                        for tag, value in metrics.items()
                        if tag.startswith(prefix) and "/" not in tag[len(prefix):])
                    print(f"           {loss.name:<10s} {body}")
            self.logger.log({"train/epoch_loss": train_loss,
                             "train/epoch_time_sec": elapsed, **metrics}, self.step)

            if self.monitor not in metrics:
                raise KeyError(
                    f"output.monitor {self.monitor!r} is not among the evaluated "
                    f"metrics: {sorted(metrics)}")
            value = metrics[self.monitor]
            improved = (value < self.best if self.monitor_mode == "min"
                        else value > self.best)
            if improved:
                self.best = value
            if self.is_main:
                self._save("last.pth")
                if improved:
                    self._save("best.pth")
                    print(f"           new best {self.monitor}={value:.4f}")
                if self.save_freq > 0 and epoch > 0 and epoch % self.save_freq == 0:
                    self._save(f"epoch_{epoch:04d}.pth")
            if self.distributed:
                dist.barrier()
        self.logger.close()

    def _save(self, name: str) -> None:
        path = self.out_dir / name
        ckpt_io.save(path, self.model, self.optimizer, self.scheduler,
                     epoch=self.epoch, step=self.step, best=self.best, config=self.cfg)
        print(f"  saved {name}  ({path.stat().st_size / 2**20:.1f} MB)")
