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
from contact.config import DEFAULTS as CONFIG_DEFAULTS
from contact.config import _deep_merge, load_config
from contact.data.collate import batch_to_device, make_loaders
from contact.engine import forward_model, select_temporal_supervision
from contact.losses import MultiTargetContactLoss, ddp_global_mean_term
from contact.metrics import (
    add_counts,
    contact_counts,
    contact_counts_per_dim,
    prf1,
    zero_counts,
)
from contact.model import _trainable_name_filter, build_model
from contact.targets import TargetSpec
from contact.tracking import RunLogger


# The per-target metrics a monitor may select (direction: all "max"; only
# ``val/loss`` is "min"). Mirrors the keys produced by ``contact.metrics.prf1``.
_MONITOR_METRICS = ("precision", "recall", "f1", "f2", "iou", "accuracy")


class _ContactForward(nn.Module):
    """Give DDP a conventional ``forward`` around SAM-3D-Body's step API.

    Returns the full forward output (``"contact"``, ``"mhr"``, ``"force"``): the
    contact loss reads ``out["contact"]`` and the physics loss reads
    ``out["mhr"]`` / ``out["force"]`` — both must share one DDP-wrapped forward so
    every trainable param sits on a single backward graph.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, batch: dict) -> dict:
        out = forward_model(self.model, batch)
        if out.get("contact") is None:
            raise RuntimeError("model produced no contact output — check DO_CONTACT_TOKENS.")
        return out


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
    """Identity-defining config diffs for a resume candidate.

    Compared: ``model``, ``contact``, ``physics`` (whole section), top-level
    ``loss`` (carries ``grad_clip``), ``data.datasets``, ``data.sequence``,
    ``data.eval_split``, ``optim`` (minus ``epochs``) and ``output.monitor``.
    Allowed to differ across a resume (not compared): ``optim.epochs`` (extend
    training) and all ``logging.*``.

    Every compared section is normalised over the schema defaults on BOTH sides
    before comparison, so a historical config that predates a backward-identical
    key (e.g. ``physics.loss.residual_robust``/``physics.max_cam_jump_m``,
    ``model.temporal.window_frames``, or the whole disabled ``physics`` section)
    compares equal to a current resolution holding those defaults — absent keys
    and explicit defaults are the same run.
    """
    def _normalized(cfg: dict, section: str) -> dict:
        return _deep_merge(CONFIG_DEFAULTS[section], cfg.get(section) or {})

    diffs: list[str] = []
    for section in ("model", "contact", "physics", "loss"):
        if _normalized(saved, section) != _normalized(current, section):
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


def _ensure_resume_identity(saved: dict, current: dict, context: str) -> None:
    """Raise unless ``saved`` and ``current`` agree on every identity-defining section.

    Shared by BOTH resume paths: ``--resume auto`` (checked against the run dir's
    ``config.yaml`` before selection) and an explicit ``--resume PATH`` (checked
    against the checkpoint's stored config after load) — an explicit path must not
    bypass the physics/loss/optim identity checks the auto path enforces.

    :param context: message prefix naming the offending source (e.g.
        ``"--resume auto selected <path> but its"``).
    """
    diffs = _resume_config_diffs(saved, current)
    if diffs:
        raise RuntimeError(
            f"{context} config differs from the current one on identity-defining "
            "sections:\n" + "\n".join(diffs)
            + "\nOnly optim.epochs and logging.* may change across a resume; "
            "start a fresh run instead.")


def _physics_residual_headline(numerator: float, mass: float, *, required: bool) -> float:
    """Finalize the mass-weighted physics-residual eval headline.

    Zero residual mass — every clip physics-ineligible or jerk-excluded — must
    never masquerade as a perfect residual: it reports ``NaN`` (which
    ``_is_better`` rejects in both monitor directions), and when the residual IS
    the monitor it raises instead — a jerk threshold or data change that silently
    excludes everything would otherwise starve best-model selection forever.

    :param numerator: all-reduced residual numerator (Σ over frames).
    :param mass: all-reduced residual mass (Σ ``n_elig * n_res``).
    :param required: whether ``output.monitor`` is ``{split}/physics_residual``.
    :raises RuntimeError: when ``required`` and ``mass == 0``.
    """
    if mass > 0:
        return numerator / mass
    if required:
        raise RuntimeError(
            "output.monitor is the physics residual but the evaluation produced no "
            "residual data: every clip in the split was physics-ineligible or "
            "excluded by physics.max_cam_jump_m. Loosen the jerk threshold, check "
            "the split's clip length vs physics.min_frames/smoothing_kernel, or "
            "monitor a contact metric instead.")
    return float("nan")


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
            _ensure_resume_identity(
                saved_cfg, cfg, f"--resume auto selected {chosen} but its")
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
        # Params to serialise: every contact/force param, whether or not currently
        # trainable. In regime (a) (``freeze_contact``) the warm-started contact
        # branch is frozen and thus absent from ``trainable_names`` (force-only), yet
        # it is not in the base checkpoint either — persisting it keeps the checkpoint
        # self-contained so evaluate/demo/resume never run with a random contact head.
        # The arch fingerprint still keys on ``trainable_names`` (see checkpoint.save).
        self.saved_names = [
            n for n, _ in self.model.named_parameters() if _trainable_name_filter(n)]
        warm_state = None
        warm_path = self.cfg["model"].get("init_contact_checkpoint")
        if resume_ckpt is None and warm_path:
            warm_state = ckpt_io.initialize_common_contact(
                warm_path, self.model, config=self.cfg, map_location=device)
            if self.is_main:
                print(
                    f"Warm-started {len(warm_state['warm_start_loaded_names'])} contact "
                    f"parameters from {warm_path}; initialized "
                    f"{len(warm_state['warm_start_new_names'])} new (temporal/force) parameters")
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

        # Physics-force objective (step 06/07). Built once on the training device;
        # loads the MHR body inside. Regime (a) (``train.freeze_contact``) drops the
        # contact loss from the total but still logs its metrics.
        self.physics_enabled = bool(self.cfg["physics"]["enabled"])
        self.freeze_contact = bool(self.cfg["train"]["freeze_contact"])
        self.physics_loss = None
        if self.physics_enabled:
            from contact.physics.loss import PhysicsLoss
            self.physics_loss = PhysicsLoss(self.cfg, device=device)
        elif self.cfg["model"]["force_head"]["enabled"]:
            # Trainer-only guard (evaluate/demo legitimately build force models
            # without a physics objective): the force branch is supervised solely
            # by the physics loss, so training it without one leaves the force
            # params gradient-less — a silent no-op single-process and a
            # find_unused_parameters=False crash under DDP.
            raise ValueError(
                "model.force_head.enabled requires physics.enabled=true for "
                "training: without the physics loss the force params never "
                "receive gradients")

        # Regime-(b) physics-gradient leak guard: the documented physics/contact
        # gradient isolation holds only in regime (a) (contact frozen). When physics
        # trains alongside a TRAINABLE contact branch, physics gradients reach the
        # contact head through force->contact attention (the vendored mask permits it,
        # sam3d_body.py). Warn loudly (do NOT raise — a joint regime may be intended).
        if self.physics_enabled and not self.freeze_contact and self.is_main:
            leaking = [n for n, p in self.model.named_parameters()
                       if p.requires_grad and "contact" in n]
            if leaking:
                bar = "!" * 78
                print(
                    f"\n{bar}\n"
                    "WARNING: regime (b) physics-gradient leak into the contact head\n"
                    "  physics.enabled=true with train.freeze_contact=false and "
                    f"{len(leaking)} trainable\n"
                    "  'contact'-named params. Physics gradients reach the contact head via\n"
                    "  force->contact attention, so the contact head is trained by BOTH its\n"
                    "  labels AND the physics residual. The documented gradient isolation\n"
                    "  holds only in regime (a) (contact frozen). Set train.freeze_contact=true\n"
                    "  for the isolated force-only objective if this is not intended.\n"
                    f"{bar}\n")

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
        self.target_frame = str(self.cfg["data"]["sequence"]["target_frame"])

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
        # `.../loss` and the physics-residual pseudo-target are minimised; the
        # classification metrics (f1/iou/...) are maximised.
        self.monitor_mode = (
            "min" if self.monitor.endswith(("/loss", "/physics_residual")) else "max")

        self.epoch        = 0
        self.global_step  = 0
        self.best_metric  = float("inf") if self.monitor_mode == "min" else float("-inf")
        self.wandb_run_id = None

        # ---- resume state (weights/optim/sched/RNG restored) before logger init ----
        if resume_ckpt is not None:
            state = ckpt_io.load(
                resume_ckpt, self.model, self.optimizer, self.scheduler,
                config=self.cfg, restore_rng=True, map_location=device)
            # Explicit ``--resume PATH`` must enforce the same identity checks the
            # auto path does (physics/loss/optim/monitor live outside the arch
            # signature ckpt_io.load verifies) — checked against the checkpoint's
            # own stored config so a moved/copied run dir cannot dodge it.
            _ensure_resume_identity(
                state.get("config") or {}, self.cfg,
                f"--resume {resume_ckpt}: the checkpoint's stored")
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
        crashing later in :meth:`_monitor_value` when the metric is looked up. When
        physics is enabled, ``{split}/physics_residual`` is a valid monitor
        pseudo-target (its residual term only, stored under
        ``metrics["physics"]["residual"]``; it is NOT a contact target).
        """
        valid = {f"{self.eval_split}/loss"} | {
            f"{self.eval_split}/{t}_{k}"
            for t in self.targets for k in _MONITOR_METRICS
        }
        if getattr(self, "physics_enabled", False):
            valid.add(f"{self.eval_split}/physics_residual")
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

    def _supervision(self, contact: dict, batch: dict) -> tuple[dict, dict]:
        """Return logits/targets for the configured temporal prediction rows."""
        return select_temporal_supervision(
            self._logits(contact),
            batch["targets"],
            int(batch.get("seq_len", 1)),
            self.target_frame,
        )

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

    def _ddp_physics_loss(
        self, total: torch.Tensor, parts: dict,
    ) -> tuple[torch.Tensor, bool]:
        """Fold PhysicsLoss's per-term ``(numerator, mass)`` into the exact DDP mean.

        Reuses the contact machinery (:func:`ddp_global_mean_term`): each physics
        term is normalised by its all-reduced global mass so DDP's gradient average
        equals the single-process global weighted mean. The term set is fixed by the
        nonzero-weight physics config (step 06), so every rank iterates the same
        terms; every term's numerator carries the graph-connected ``joint_forces``
        zero, keeping all force params on the backward graph under
        ``find_unused_parameters=False`` even on physics-ineligible batches.
        """
        terms = parts["terms"]
        names = list(terms)
        if not self.distributed:
            active = any(terms[n]["weight_mass"] > 0 for n in names)
            return total, active
        local_mass = torch.tensor(
            [terms[n]["weight_mass"] for n in names],
            dtype=torch.float64, device=self.device,
        )
        global_mass = local_mass.clone()
        dist.all_reduce(global_mass, op=dist.ReduceOp.SUM)
        scaled = None
        for i, name in enumerate(names):
            term = ddp_global_mean_term(
                terms[name]["weighted_numerator_tensor"], global_mass[i], self.world_size)
            scaled = term if scaled is None else scaled + term
        return scaled, bool(global_mass.sum().item() > 0)

    @staticmethod
    def _physics_scalars(prefix: str, phys_parts: dict) -> dict:
        """Flatten PhysicsLoss parts into loggable ``{prefix}/physics/*`` scalars.

        Per-term local losses + masses (residual and every active regulariser) plus
        the residual-force/torque diagnostics and the eligibility counts.
        """
        scalars = {f"{prefix}/physics/loss": phys_parts["loss"]}
        for name, term in phys_parts["terms"].items():
            scalars[f"{prefix}/physics/{name}"] = term["loss"]
            scalars[f"{prefix}/physics/{name}_mass"] = term["weight_mass"]
        scalars[f"{prefix}/physics/residual_force"] = phys_parts["residual_force"]
        scalars[f"{prefix}/physics/residual_torque"] = phys_parts["residual_torque"]
        # Raw (un-robustified) physical residual headline + collapse/robustness/jerk
        # diagnostics (§1-§3): raw_residual is the comparable physics_residual value,
        # force_std the cheapest online collapse detector, residual_sat_frac the Huber
        # tail fraction (0 under square), n_jerk_excluded_clips the camera-jerk drops.
        scalars[f"{prefix}/physics/raw_residual"] = phys_parts["raw_residual"]["loss"]
        scalars[f"{prefix}/physics/residual_sat_frac"] = phys_parts["residual_sat_frac"]
        scalars[f"{prefix}/physics/force_std"] = phys_parts["force_std"]
        scalars[f"{prefix}/physics/n_eligible_clips"] = phys_parts["n_eligible_clips"]
        scalars[f"{prefix}/physics/n_residual_frames"] = phys_parts["n_residual_frames"]
        scalars[f"{prefix}/physics/n_jerk_excluded_clips"] = phys_parts["n_jerk_excluded_clips"]
        return scalars

    def _print_run_summary(self) -> None:
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        fpb = int(self.cfg["data"]["frames_per_batch"])
        clip_len = int(self.cfg["data"]["sequence"]["frames_per_clip"])
        has_video = any(d["name"] == "climbing_videos" for d in self.cfg["data"]["datasets"])
        layout = (f"video: {max(1, fpb // clip_len)} clips x T={clip_len}"
                  if has_video else f"stills: {fpb} frames x T=1")
        supervised = (
            f"center row only ({max(1, fpb // clip_len)} labels/rank)"
            if self.target_frame == "center"
            else "all rows"
        )
        print(f"Trainable params: {n_train:,} | targets={self.targets} primary={self.primary} | "
              f"monitor={self.monitor} ({self.monitor_mode}) | supervision={supervised}")
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

            out = self.forward_module(batch)
            logits, targets = self._supervision(out["contact"], batch)
            contact_loss, parts = self.loss_fn(logits, targets)
            contact_loss, contact_active = self._ddp_weighted_loss(contact_loss, parts)

            # Regime (a) (``freeze_contact``): contact params are frozen, so the
            # contact loss is dropped from the objective (its metrics stay logged)
            # and physics is the sole objective. Otherwise the total is contact +
            # physics (either may be absent).
            phys_parts = None
            physics_active = False
            total = None if self.freeze_contact else contact_loss
            if self.physics_loss is not None:
                phys_total, phys_parts = self.physics_loss(out, batch)
                phys_loss, physics_active = self._ddp_physics_loss(phys_total, phys_parts)
                total = phys_loss if total is None else total + phys_loss
            if total is None:
                raise RuntimeError(
                    "train.freeze_contact is set but no physics objective is enabled — "
                    "nothing to train")

            # A batch with zero active supervision (an all-invalid video window, or a
            # fully physics-ineligible clip) has zero gradient — but AdamW weight
            # decay would still nudge the weights. Skip the optimiser step for those.
            active = (physics_active if self.freeze_contact
                      else (contact_active or physics_active))
            loss = total
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
                # _clip_grads raises FloatingPointError on a non-finite RAW norm:
                # checking the post-clip value would let an inf raw slip through as
                # min(inf, grad_clip) while clipping had already NaN'd the grads.
                grad_norm_raw, grad_norm = self._clip_grads()
                self.optimizer.step()
            else:
                skipped += 1
                grad_norm_raw = grad_norm = 0.0

            running_loss += loss.item()
            n += 1
            batch_counts = {}
            for t in self.targets:
                tgt = targets[t]
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
                    "train/grad_norm_raw": grad_norm_raw,
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
                        tgt = targets[t]
                        current_dims = contact_counts_per_dim(
                            logits[t].detach(), tgt["gt"], tgt["mask"])
                        for output_name, current in zip(self.output_names[t], current_dims):
                            output_metrics = prf1(current)
                            for key in ("precision", "recall", "f1", "f2"):
                                scalars[f"train/{t}/{output_name}/{key}"] = output_metrics[key]
                if phys_parts is not None:
                    scalars.update(self._physics_scalars("train", phys_parts))
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

    def _clip_grads(self) -> tuple[float, float]:
        """Clip trainable grads; return ``(raw, post)`` global grad norms.

        ``raw`` is the pre-clip total norm (exactly what ``clip_grad_norm_`` measures
        and returns); ``post`` applies the ``loss.grad_clip`` cap. Logging both makes
        the true gradient scale visible — the post-clip value alone pegs at the cap.

        :raises FloatingPointError: when the RAW norm is non-finite. The finiteness
            check must run on ``raw``: an inf/NaN raw makes ``clip_grad_norm_``
            scale the grads by 0/NaN (corrupting them), yet the capped post value
            ``min(inf, grad_clip)`` would look perfectly finite.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        max_norm = self.grad_clip if self.grad_clip > 0 else float("inf")
        raw = torch.nn.utils.clip_grad_norm_(params, max_norm).item()
        if not math.isfinite(raw):
            raise FloatingPointError(
                f"non-finite raw gradient norm ({raw}) at epoch={self.epoch} "
                f"global_step={self.global_step}; optimizer step was not executed")
        post = min(raw, self.grad_clip) if self.grad_clip > 0 else raw
        return raw, post

    @torch.no_grad()
    def _evaluate(self) -> dict:
        self.model.eval()
        running_loss, n = 0.0, 0
        counts = {t: zero_counts() for t in self.targets}
        output_counts = {
            target: [zero_counts() for _ in names]
            for target, names in self.output_names.items()
        }
        # Physics residual: exact global (batch- and rank-wise) mass-weighted mean of
        # the RAW physical residual (raw_residual — un-robustified, comparable across
        # runs; falls back to the residual term for old parts). Regularizer terms are
        # deliberately excluded — their mix should not decide "best". Reuses
        # PhysicsLoss's additive numerator/mass.
        phys_residual_num, phys_residual_mass = 0.0, 0.0
        for batch in tqdm(
            self.eval_loader, desc=self.eval_split, disable=not self.is_main,
        ):
            batch  = batch_to_device(batch, self.device)
            out = self.forward_module(batch)
            logits, targets = self._supervision(out["contact"], batch)
            loss, _ = self.loss_fn(logits, targets)
            running_loss += loss.item()
            n += 1
            if self.physics_loss is not None:
                _, phys_parts = self.physics_loss(out, batch)
                rterm = phys_parts.get("raw_residual") or phys_parts["terms"].get("residual")
                if rterm is not None:
                    phys_residual_num += float(rterm["weighted_numerator_tensor"].detach())
                    phys_residual_mass += rterm["weight_mass"]
            for t in self.targets:
                tgt = targets[t]
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
        metrics = {t: prf1(counts[t]) for t in self.targets}
        if self.physics_loss is not None:
            if self.distributed:
                packed = torch.tensor(
                    [phys_residual_num, phys_residual_mass],
                    dtype=torch.float64, device=self.device)
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                phys_residual_num, phys_residual_mass = packed.cpu().tolist()
            metrics["physics"] = {"residual": _physics_residual_headline(
                phys_residual_num, phys_residual_mass,
                required=self.monitor.endswith("/physics_residual"))}
        return {"loss": running_loss / max(n, 1),
                "metrics": metrics,
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
                phys_note = ""
                if "physics" in v["metrics"]:
                    phys_note = f"  physics_residual {v['metrics']['physics']['residual']:.4f}"
                if self.is_main:
                    print(f"           {self.eval_split:<4s} loss {v['loss']:.4f}  {vsummary}{phys_note}")
                val_scalars = {f"{self.eval_split}/loss": v["loss"]}
                for name in self.targets:
                    for key, val in v["metrics"][name].items():
                        val_scalars[f"{self.eval_split}/{name}/{key}"] = val
                    for output_name, output_metrics in v["per_output"].get(name, {}).items():
                        for key, val in output_metrics.items():
                            val_scalars[
                                f"{self.eval_split}/{name}/{output_name}/{key}"
                            ] = val
                # Physics residual pseudo-target (guarded out of the per-target loop
                # above so the f1-hardcoded summary never KeyErrors on "physics").
                if "physics" in v["metrics"]:
                    val_scalars[f"{self.eval_split}/physics/residual"] = (
                        v["metrics"]["physics"]["residual"])
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
            saved_names=self.saved_names,
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
