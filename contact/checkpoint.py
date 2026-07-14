"""Schema-versioned checkpoints for the trainable contact parameters (v2).

The frozen SAM-3D-Body weights live in the original checkpoint — re-saving them
every epoch would be ~600 MB of nothing new. Here we serialise only the params
named in ``trainable_names`` (contact tokens, contact head(s), the contact
posemb/feat projections, and the contact temporal module when enabled), plus the
optimiser, scheduler, run state, resolved config, wandb run id and RNG states.

**Hard-fail loading.** Every checkpoint carries ``schema_version`` and an *arch
fingerprint* — a sha256 over the sorted ``(trainable name, shape)`` pairs. Loading
into a model whose trainable architecture differs (e.g. different
``contact_keypoint_indices`` or head output dims) raises with a diff of the
mismatched entries instead of silently proceeding with a randomly initialised
head (the old ``strict=False`` flow could).
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn

from .targets import joint_set_names, joint_set_num_outputs, topology_num_vertices

SCHEMA_VERSION = 2


# -------------------------------------------------------------------- fingerprint

def _current_trainable_spec(model: nn.Module) -> list[tuple[str, tuple[int, ...]]]:
    """Sorted ``(name, shape)`` pairs for *every* ``requires_grad`` param.

    Computed from the live model, independent of any name list stored in a
    checkpoint, so a current model with **extra** trainable params (e.g. the
    temporal module) or **missing** ones is detected on load — not silently
    accepted.
    """
    spec = [
        (name, tuple(p.shape))
        for name, p in model.named_parameters()
        if p.requires_grad
    ]
    return sorted(spec)


def _arch_spec(model: nn.Module, names: Iterable[str]) -> list[tuple[str, tuple[int, ...]]]:
    """Sorted ``(name, shape)`` pairs for the named params — the arch identity."""
    wanted = set(names)
    spec = [
        (name, tuple(p.shape))
        for name, p in model.named_parameters()
        if name in wanted
    ]
    return sorted(spec)


def _arch_signature(config: Optional[dict]) -> Optional[dict]:
    """Canonical forward-architecture signature for the trainable contact stack.

    Captures semantics that leave the trainable param **shapes** unchanged yet
    change what the head learns — reordered anchor indices, grid params, temporal
    placement/attend/causal, mask conditioning, and the frozen base checkpoint —
    so a same-shape but different-architecture checkpoint is rejected on load.
    Returns ``None`` when ``config`` is ``None`` (signature comparison skipped).
    """
    if config is None:
        return None
    model_cfg = config.get("model", {}) or {}
    chead = dict(model_cfg.get("contact_head", {}) or {})
    kp = chead.get("contact_keypoint_indices")
    if kp is None:
        kp = list(range(21))
    topology = config.get("contact", {}).get("topology", "smpl")
    tgts = config.get("contact", {}).get("targets", {}) or {}
    targets_layout = {}
    joint_layout = None
    for name in ("vertex", "joint"):
        if tgts.get(name, {}).get("enabled", False):
            if name == "vertex":
                targets_layout[name] = topology_num_vertices(topology)
            else:
                joint_set = str(tgts[name].get("joint_set", "smplx_body_22"))
                names = joint_set_names(joint_set)
                targets_layout[name] = joint_set_num_outputs(joint_set)
                joint_layout = {
                    "joint_set": joint_set,
                    "names": list(names),
                    "dim": len(names),
                }

    tcfg = model_cfg.get("temporal", {}) or {}
    if tcfg.get("enabled", False):
        temporal = {
            "enabled": True,
            "placement": str(tcfg.get("placement", "post_decoder")),
            "attend": str(tcfg.get("attend", "joint")),
            "causal": bool(tcfg.get("causal", False)),
            "bottleneck_dim": int(tcfg.get("bottleneck_dim", 256)),
            "num_layers": int(tcfg.get("num_layers", 1)),
            "num_heads": int(tcfg.get("num_heads", 4)),
            "mlp_ratio": float(tcfg.get("mlp_ratio", 2.0)),
            "position_scale": float(tcfg.get("position_scale", 1.0)),
        }
    else:
        temporal = {"enabled": False}

    return {
        "anchor_indices": [int(i) for i in kp],
        "num_global_tokens": int(chead.get("num_global_tokens", 3)),
        "pool_mode": str(chead.get("pool_mode", "concat")),
        "mlp_depth": int(chead.get("mlp_depth", 4)),
        "mlp_channel_div_factor": int(chead.get("mlp_channel_div_factor", 2)),
        "grid_size": int(chead.get("grid_size", 5)),
        "grid_radius": float(chead.get("grid_radius", 0.1)),
        "dropout": float(chead.get("dropout", 0.1)),
        "topology": str(topology),
        "targets": {k: int(v) for k, v in sorted(targets_layout.items())},
        "joint_layout": joint_layout,
        "temporal": temporal,
        "mask_embed_type": model_cfg.get("mask_embed_type", None),
        "base_checkpoint": Path(str(model_cfg.get("checkpoint_path", ""))).name,
    }


def _fingerprint(spec: list[tuple[str, tuple[int, ...]]]) -> str:
    """sha256 over the sorted ``(name, shape)`` pairs."""
    h = hashlib.sha256()
    for name, shape in spec:
        h.update(name.encode("utf-8"))
        h.update(repr(tuple(int(d) for d in shape)).encode("utf-8"))
    return h.hexdigest()


def _select_state(model: nn.Module, names: Iterable[str]) -> dict:
    sd = model.state_dict()
    names = set(names)
    return {k: v for k, v in sd.items() if k in names}


# -------------------------------------------------------------------- RNG

def _capture_rng() -> dict:
    """Snapshot torch (CPU + all CUDA), numpy and python RNG states."""
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _cpu_byte(state: torch.Tensor) -> torch.Tensor:
    """A CPU ``uint8`` copy — RNG generators reject states on CUDA / wrong dtype.

    ``load(..., map_location="cuda")`` moves every stored tensor (including RNG
    ByteTensors) onto CUDA, so coerce back before handing them to the generators.
    """
    return state.cpu().to(torch.uint8)


def _restore_rng(rng: Optional[dict]) -> None:
    if not rng:
        return
    torch.set_rng_state(_cpu_byte(rng["torch_cpu"]))
    if rng.get("torch_cuda") is not None and torch.cuda.is_available():
        cuda_states = [_cpu_byte(s) for s in rng["torch_cuda"]]
        # Restore only when the device count matches what we saved.
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])


# -------------------------------------------------------------------- save / load

def save(
    path: str | Path,
    model: nn.Module,
    trainable_names: Iterable[str],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object],
    *,
    epoch: int,
    global_step: int,
    best_metric: float,
    monitor: str,
    config: dict,
    wandb_run_id: Optional[str] = None,
    split_manifest: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> None:
    """Write a schema-v2 checkpoint of the trainable state and run metadata."""
    trainable_names = list(trainable_names)
    spec = _arch_spec(model, trainable_names)
    ckpt = {
        "schema_version":       SCHEMA_VERSION,
        "arch_fingerprint":     _fingerprint(spec),
        "arch_spec":            [(n, list(s)) for n, s in spec],
        "arch_signature":       _arch_signature(config),
        "trainable_state_dict": _select_state(model, trainable_names),
        "trainable_names":      trainable_names,
        "optimizer":            optimizer.state_dict(),
        "scheduler":            scheduler.state_dict() if scheduler is not None else None,
        "epoch":                int(epoch),
        "global_step":          int(global_step),
        "best_metric":          float(best_metric),
        "monitor":              str(monitor),
        "config":               config,
        "wandb_run_id":         wandb_run_id,
        "split_manifest":       split_manifest,
        "rng":                  _capture_rng(),
    }
    if extra:
        ckpt["extra"] = extra
    torch.save(ckpt, Path(path))


def _check_schema(
    ckpt: dict, model: nn.Module, path: str | Path, config: Optional[dict] = None,
) -> None:
    """Raise a clear error if the checkpoint is not a compatible v2 checkpoint."""
    version = ckpt.get("schema_version")
    if version != SCHEMA_VERSION:
        raise RuntimeError(
            f"{path}: schema_version {version!r} != {SCHEMA_VERSION}. This is not a "
            f"v2 contact checkpoint (old checkpoints are throwaway — retrain). "
            f"Refusing to load to avoid a silently random-initialised head."
        )

    # Compare against the *complete* current trainable spec (both directions):
    # a model with extra trainable params (temporal, a second head) no longer
    # slips through by only checking the checkpoint's own name list.
    have_spec = _current_trainable_spec(model)
    have_fp = _fingerprint(have_spec)
    if have_fp != ckpt["arch_fingerprint"]:
        want = {n: tuple(s) for n, s in ckpt["arch_spec"]}
        got = {n: tuple(s) for n, s in have_spec}
        mismatches = []
        for name in sorted(set(want) | set(got)):
            if want.get(name) != got.get(name):
                mismatches.append(
                    f"  {name}: checkpoint={want.get(name, 'ABSENT')} "
                    f"model={got.get(name, 'ABSENT')}")
        raise RuntimeError(
            f"{path}: arch fingerprint mismatch — the model was built with a "
            f"different contact architecture than this checkpoint.\n"
            f"fingerprint checkpoint={ckpt['arch_fingerprint'][:12]} "
            f"model={have_fp[:12]}\nmismatched trainable params:\n"
            + "\n".join(mismatches))

    # Same-shape semantic mismatch (reordered anchors, grid params, temporal
    # placement/attend/causal, mask conditioning, base checkpoint identity).
    want_sig = ckpt.get("arch_signature")
    got_sig = _arch_signature(config)
    if config is not None and want_sig is not None and want_sig != got_sig:
        diffs = [
            f"  {key}: checkpoint={want_sig.get(key, 'ABSENT')!r} "
            f"run={got_sig.get(key, 'ABSENT')!r}"
            for key in sorted(set(want_sig) | set(got_sig))
            if want_sig.get(key) != got_sig.get(key)
        ]
        raise RuntimeError(
            f"{path}: arch signature mismatch — the trainable params share the "
            f"checkpoint's shapes but the architecture differs semantically:\n"
            + "\n".join(diffs))


def load(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    *,
    config: Optional[dict] = None,
    restore_rng: bool = False,
    map_location: str = "cpu",
) -> dict:
    """Load a v2 checkpoint into ``model``; optionally restore optimiser/scheduler/RNG.

    :param config: the current run config; when given, the checkpoint's stored
        architecture *signature* (anchor indices, head/temporal/mask settings,
        base checkpoint) is compared against it and a mismatch hard-fails.
    :param restore_rng: when ``True`` (resume-training path), restore torch/numpy/
        python RNG so the next epoch's data ordering matches an uninterrupted run.
    :raises RuntimeError: if the checkpoint is not schema v2 or its trainable-param
        fingerprint / architecture signature does not match ``model`` (never
        silently random-inits).
    :returns: the loaded checkpoint dict (metadata + state).
    """
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict) or "trainable_state_dict" not in ckpt:
        raise RuntimeError(f"{path}: not a contact checkpoint (no trainable_state_dict).")

    _check_schema(ckpt, model, path, config)

    missing, unexpected = model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
    # The frozen base weights are expected to be "missing" here; a *trainable*
    # name missing means the fingerprint check let something slip — treat as fatal.
    trainable_missing = [m for m in missing if m in set(ckpt["trainable_names"])]
    if trainable_missing:
        raise RuntimeError(f"{path}: trainable params missing from checkpoint: {trainable_missing}")

    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if restore_rng:
        _restore_rng(ckpt.get("rng"))
    return ckpt


def initialize_common_contact(
    path: str | Path,
    model: nn.Module,
    *,
    config: dict,
    map_location: str = "cpu",
) -> dict:
    """Warm-start common contact parameters while allowing a new temporal module.

    This is deliberately narrower than :func:`load`: it does not restore the
    optimiser, scheduler, epoch, or RNG, and the only trainable parameters that
    may be absent from the source checkpoint are ``contact_temporal.*``. All
    forward semantics other than the temporal configuration must match exactly.
    It is intended for starting temporal fine-tuning from a per-frame contact
    checkpoint without weakening strict resume checks.
    """
    path = Path(path)
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict) or "trainable_state_dict" not in ckpt:
        raise RuntimeError(f"{path}: not a contact checkpoint (no trainable_state_dict).")
    if ckpt.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"{path}: schema_version {ckpt.get('schema_version')!r} != {SCHEMA_VERSION}")

    source_sig = dict(ckpt.get("arch_signature") or {})
    target_sig = dict(_arch_signature(config) or {})
    source_temporal = source_sig.pop("temporal", {"enabled": False})
    target_temporal = target_sig.pop("temporal", {"enabled": False})
    if source_sig != target_sig:
        diffs = [
            f"  {key}: checkpoint={source_sig.get(key, 'ABSENT')!r} "
            f"run={target_sig.get(key, 'ABSENT')!r}"
            for key in sorted(set(source_sig) | set(target_sig))
            if source_sig.get(key) != target_sig.get(key)
        ]
        raise RuntimeError(
            f"{path}: warm-start architecture mismatch outside temporal module:\n"
            + "\n".join(diffs))
    if source_temporal.get("enabled", False):
        raise RuntimeError(
            f"{path}: warm-start source already has temporal enabled; use strict resume "
            "for the same architecture or start from a per-frame checkpoint")
    if not target_temporal.get("enabled", False):
        raise RuntimeError(
            "initialize_common_contact is only valid when the target enables temporal")

    current = dict(model.named_parameters())
    source_state = ckpt["trainable_state_dict"]
    problems = []
    for name, value in source_state.items():
        if name not in current:
            problems.append(f"  {name}: absent from target model")
        elif tuple(value.shape) != tuple(current[name].shape):
            problems.append(
                f"  {name}: checkpoint={tuple(value.shape)} model={tuple(current[name].shape)}")
    if problems:
        raise RuntimeError(f"{path}: incompatible warm-start parameters:\n" + "\n".join(problems))

    current_trainable = {name for name, p in current.items() if p.requires_grad}
    missing = sorted(current_trainable - set(source_state))
    invalid_missing = [name for name in missing if not name.startswith("contact_temporal.")]
    if invalid_missing:
        raise RuntimeError(
            f"{path}: warm start is missing non-temporal trainable parameters: "
            f"{invalid_missing}")

    model.load_state_dict(source_state, strict=False)
    ckpt["warm_start_loaded_names"] = sorted(source_state)
    ckpt["warm_start_new_names"] = missing
    return ckpt
