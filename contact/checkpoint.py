"""Schema-versioned checkpoints for the trainable contact/force parameters (v2).

The frozen SAM-3D-Body weights live in the original checkpoint — re-saving them
every epoch would be ~600 MB of nothing new. Here we serialise only the params
named in ``saved_names`` (contact tokens, contact head(s), the contact
posemb/feat projections, the shared ``cross_modal_temporal`` block, and — when
enabled — the force tokens / ``head_force``), plus the optimiser, scheduler, run
state, resolved config, wandb run id and RNG states.

**Self-contained regime (a).** In regime (a) (``train.freeze_contact``) the
warm-started contact branch is frozen, so it is *not* in the ``requires_grad``
trainable set — yet it is not part of the base checkpoint either. ``save`` takes a
``saved_names`` superset (every ``contact``/``force`` param, whether or not it is
currently trainable) so those frozen contact weights ride along; the arch
fingerprint stays computed over the ``requires_grad`` trainable set only, so the
hard-fail identity check is unchanged. ``load`` restores every saved param and
hard-fails if the frozen contact branch is incomplete.

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

from .targets import joint_set_names, joint_set_num_outputs

SCHEMA_VERSION = 2


# -------------------------------------------------------------------- fingerprint

def _current_trainable_spec(model: nn.Module) -> list[tuple[str, tuple[int, ...]]]:
    """Sorted ``(name, shape)`` pairs for *every* ``requires_grad`` param.

    Computed from the live model, independent of any name list stored in a
    checkpoint, so a current model with **extra** trainable params (e.g. the
    cross-modal temporal block) or **missing** ones is detected on load — not
    silently accepted.
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
    hyperparameters, mask conditioning, and the frozen base checkpoint — so a
    same-shape but different-architecture checkpoint is rejected on load.
    Returns ``None`` when ``config`` is ``None`` (signature comparison skipped).

    The ``motion``, ``cond_input``, ``cross_modal_temporal`` and ``pose_temporal``
    keys appear only when their branch is enabled, and the comparison is an exact
    dict equality.

    Known, deliberate gap: the RoPE sub-signatures omit ``dropout`` and
    ``max_rel_sec``. ``max_rel_sec`` is an inference attention-window choice (like
    ``frames_per_clip``), not a weight-shape or head-semantic key, so a
    long-sequence eval must load its training checkpoint.
    """
    if config is None:
        return None
    model_cfg = config.get("model", {}) or {}
    chead = dict(model_cfg.get("contact_head", {}) or {})
    kp = chead.get("contact_keypoint_indices")
    if kp is None:
        kp = list(range(21))
    tgts = config.get("contact", {}).get("targets", {}) or {}
    targets_layout = {}
    joint_layout = None
    if tgts.get("joint", {}).get("enabled", False):
        joint_set = str(tgts["joint"].get("joint_set", "smplx_body_22"))
        names = joint_set_names(joint_set)
        targets_layout["joint"] = joint_set_num_outputs(joint_set)
        joint_layout = {
            "joint_set": joint_set,
            "names": list(names),
            "dim": len(names),
        }

    # Force head (step 04): same shape-invariant-but-semantic capture as the
    # contact head. `frame` changes what the head learns (LWA vs joint-local
    # forces), so it belongs here; loss weights / physics: numbers do NOT.
    fhcfg = model_cfg.get("force_head", {}) or {}
    if fhcfg.get("enabled", False):
        force = {
            "enabled": True,
            "frame": str(fhcfg.get("frame", "local_world_aligned")),
            "mlp_depth": int(fhcfg.get("mlp_depth", 2)),
            "mlp_channel_div_factor": int(fhcfg.get("mlp_channel_div_factor", 4)),
            "dropout": float(fhcfg.get("dropout", 0.0)),
        }
        # Decoupled force anchors change what each token regresses, so they are
        # semantic. Added only when set: legacy checkpoints (null = inherit the
        # contact anchors) keep their stored signatures byte-identical.
        force_kp = fhcfg.get("force_keypoint_indices")
        if force_kp is not None:
            force["force_keypoint_indices"] = [int(i) for i in force_kp]
        # The contact gate changes what the head weights mean (they learn the
        # pre-gate magnitude), so it is semantic too. Added only when enabled,
        # for the same legacy-signature reason as the anchors above.
        gate_cfg = fhcfg.get("contact_gate", {}) or {}
        if gate_cfg.get("enabled", False):
            force["contact_gate"] = {
                "enabled": True,
                "sharpness": float(gate_cfg.get("sharpness", 4.0)),
            }
    else:
        force = {"enabled": False}

    # Motion head (motion tokens v2). Added to the returned signature ONLY when
    # the motion branch is enabled, so every checkpoint written before it existed
    # keeps a byte-identical stored signature (the comparison in `_check_schema`
    # is an exact dict equality).
    sig_extra: dict = {}
    mhcfg = model_cfg.get("motion_head", {}) or {}
    if mhcfg.get("enabled", False):
        sig_extra["motion"] = {
            "enabled": True,
            "motion_keypoint_indices": [
                int(i) for i in (mhcfg.get("motion_keypoint_indices") or [])],
            "mlp_depth": int(mhcfg.get("mlp_depth", 2)),
            "mlp_channel_div_factor": int(mhcfg.get("mlp_channel_div_factor", 4)),
            "dropout": float(mhcfg.get("dropout", 0.0)),
        }
        # Unanchored motion tokens (no per-layer posemb/feature update) are a
        # different architecture with a different param set. Added only when
        # non-default, so every anchored checkpoint keeps a byte-identical
        # stored signature (same rule as `force_keypoint_indices`).
        if not mhcfg.get("anchored", True):
            sig_extra["motion"]["anchored"] = False

    # The all-modality RoPE temporal block. Enabled-only, like `motion`: absent
    # from every checkpoint written before it existed. The modality list changes
    # which token slices the weights were trained on AND sizes the learned slot
    # embedding, so it is semantic; `time_scale` changes what the rotary
    # positions mean.
    xmcfg = model_cfg.get("cross_modal_temporal", {}) or {}
    if xmcfg.get("enabled", False):
        sig_extra["cross_modal_temporal"] = {
            "enabled": True,
            "type": str(xmcfg.get("type", "rope")),
            "modalities": sorted(str(m) for m in (xmcfg.get("modalities") or [])),
            "num_layers": int(xmcfg.get("num_layers", 4)),
            "num_heads": int(xmcfg.get("num_heads", 16)),
            "mlp_ratio": float(xmcfg.get("mlp_ratio", 2.0)),
            "time_scale": float(xmcfg.get("time_scale", 25.0)),
        }
    # The mutual decoder mask changes what every appended token was trained to
    # read (contact attends force/motion and vice versa) — semantic. Emitted
    # only when set, so pre-existing causal signatures stay byte-identical.
    if model_cfg.get("extra_token_attention", "mutual") == "mutual":
        sig_extra["extra_token_attention"] = "mutual"

    # Pose/camera-head fine-tune (train.finetune_pose_head /
    # train.finetune_camera_head). Enabled-only: the checkpoint then carries
    # the fine-tuned COPY weights (head_pose_ft_proj / head_camera_ft_proj,
    # split-head), which must never load into (or be expected by) a
    # frozen-pose run. "split": True distinguishes from the pre-split scheme
    # whose weights lived at head_pose.proj (loads hard-fail on the param
    # diff either way).
    if (config.get("train", {}) or {}).get("finetune_pose_head", False):
        sig_extra["pose_head_finetune"] = {"enabled": True, "split": True}
    if (config.get("train", {}) or {}).get("finetune_camera_head", False):
        sig_extra["camera_head_finetune"] = {"enabled": True}

    # Pose-token temporal module (E2). Enabled-only, like `motion`: absent from
    # every checkpoint written before it existed.
    ptcfg = model_cfg.get("pose_temporal", {}) or {}
    if ptcfg.get("enabled", False):
        sig_extra["pose_temporal"] = {
            "enabled": True,
            "type": "rope",
            "num_layers": int(ptcfg.get("num_layers", 4)),
            "num_heads": int(ptcfg.get("num_heads", 16)),
            "mlp_ratio": float(ptcfg.get("mlp_ratio", 2.0)),
            "time_scale": float(ptcfg.get("time_scale", 25.0)),
        }

    # Input conditioning (model.cond_input). Same enabled-only rule as `motion`:
    # every checkpoint written before it existed keeps a byte-identical stored
    # signature. The standardization literals and the clip ARE captured — they
    # define what the projections' learned weights mean, so a checkpoint trained
    # under different literals is a different model, not a rescaled one.
    ccfg = model_cfg.get("cond_input", {}) or {}
    if ccfg.get("enabled", False):
        std = ccfg.get("standardize", {}) or {}
        sig_extra["cond_input"] = {
            "enabled": True,
            "clip": float(ccfg.get("clip", 5.0)),
            "standardize": {
                key: [float(v) for v in (std.get(key) or [])]
                for key in ("vel_mean", "vel_std", "acc_mean", "acc_std")
            },
        }
        # Added only when set: every bare-linear checkpoint written before the
        # MLP encoder existed keeps a byte-identical stored signature.
        if ccfg.get("encoder_hidden", None) is not None:
            sig_extra["cond_input"]["encoder_hidden"] = int(ccfg["encoder_hidden"])
        # Added only when non-default (same stability rule): the injection site
        # changes what the learned projections mean, so a post_decoder run must
        # never silently load a pre_decoder checkpoint or vice versa.
        if str(ccfg.get("injection", "pre_decoder")) != "pre_decoder":
            sig_extra["cond_input"]["injection"] = str(ccfg["injection"])

    return {
        **sig_extra,
        "anchor_indices": [int(i) for i in kp],
        "num_global_tokens": int(chead.get("num_global_tokens", 3)),
        "pool_mode": str(chead.get("pool_mode", "concat")),
        "mlp_depth": int(chead.get("mlp_depth", 4)),
        "mlp_channel_div_factor": int(chead.get("mlp_channel_div_factor", 2)),
        "grid_size": int(chead.get("grid_size", 5)),
        "grid_radius": float(chead.get("grid_radius", 0.1)),
        "dropout": float(chead.get("dropout", 0.1)),
        "targets": {k: int(v) for k, v in sorted(targets_layout.items())},
        "joint_layout": joint_layout,
        "force": force,
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
    saved_names: Optional[Iterable[str]] = None,
) -> None:
    """Write a schema-v2 checkpoint of the trainable state and run metadata.

    :param trainable_names: the ``requires_grad`` param names — the arch identity
        (fingerprint + spec). Unchanged by ``saved_names`` so the hard-fail load
        check still compares against the live model's trainable set exactly.
    :param saved_names: superset of ``trainable_names`` whose tensors are actually
        serialised. Defaults to ``trainable_names``. In regime (a) the caller passes
        every ``contact``/``force`` param so the frozen-but-not-in-base contact
        weights are persisted (otherwise consumers would run with random contact
        weights). The extra frozen names are also recorded under
        ``frozen_saved_names`` so ``load`` can tell a self-contained regime-(a)
        checkpoint from a legacy one.
    """
    trainable_names = list(trainable_names)
    saved_names = list(saved_names) if saved_names is not None else list(trainable_names)
    spec = _arch_spec(model, trainable_names)
    frozen_saved_names = sorted(set(saved_names) - set(trainable_names))
    ckpt = {
        "schema_version":       SCHEMA_VERSION,
        "arch_fingerprint":     _fingerprint(spec),
        "arch_spec":            [(n, list(s)) for n, s in spec],
        "arch_signature":       _arch_signature(config),
        "trainable_state_dict": _select_state(model, saved_names),
        "trainable_names":      trainable_names,
        "frozen_saved_names":   frozen_saved_names,
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
    if config is not None and want_sig is not None:
        # `force` is absent from a contact-only signature on either side;
        # default it to disabled on BOTH so such a pair still compares equal.
        disabled = {"force": {"enabled": False}}
        want_sig = {**disabled, **want_sig}
        got_sig = {**disabled, **(got_sig or {})}
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


def _check_frozen_contact_complete(ckpt: dict, model: nn.Module, path: str | Path) -> None:
    """Reject a regime-(a) checkpoint whose frozen contact branch is incomplete.

    A regime-(a) checkpoint (``train.freeze_contact``) is self-contained: it
    carries the warm-started, then frozen, contact weights alongside the trainable
    force ones. It must hold the model's FULL contact-named set — a partial (or
    absent) contact branch would leave the rest at random init, the exact silent
    failure the hard-fail guarantee forbids.
    """
    ckpt_config = ckpt.get("config") or {}
    if not bool((ckpt_config.get("train") or {}).get("freeze_contact", False)):
        return
    saved = {name for name in ckpt["trainable_state_dict"] if "contact" in name.lower()}
    wanted = {name for name, _ in model.named_parameters() if "contact" in name.lower()}
    missing = sorted(wanted - saved)
    if missing:
        raise RuntimeError(
            f"{path}: regime-(a) checkpoint is missing {len(missing)} contact "
            f"param(s) of the model (e.g. {missing[:5]}). Loading it would leave "
            f"them randomly initialised; retrain to write a self-contained "
            f"checkpoint.")


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
        silently random-inits), or if a regime-(a) checkpoint's frozen contact
        branch is incomplete (:func:`_check_frozen_contact_complete`).
    :returns: the loaded checkpoint dict (metadata + state).
    """
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict) or "trainable_state_dict" not in ckpt:
        raise RuntimeError(f"{path}: not a contact checkpoint (no trainable_state_dict).")

    _check_schema(ckpt, model, path, config)

    # Self-contained-checkpoint completeness: every frozen name the checkpoint
    # CLAIMS to carry (regime (a) contact branch) must actually have a tensor in
    # the state dict and a home in the model — a truncated/partial checkpoint must
    # hard-fail here, never load a partially random frozen branch.
    frozen_names = ckpt.get("frozen_saved_names") or []
    if frozen_names:
        state_keys = set(ckpt["trainable_state_dict"])
        model_params = {name for name, _ in model.named_parameters()}
        missing_tensors = sorted(set(frozen_names) - state_keys)
        missing_params = sorted(set(frozen_names) - model_params)
        if missing_tensors or missing_params:
            raise RuntimeError(
                f"{path}: frozen_saved_names inconsistency — the checkpoint is "
                f"corrupt/partial or the model differs.\n"
                f"  names without a saved tensor: {missing_tensors}\n"
                f"  names absent from the model:  {missing_params}")

    missing, unexpected = model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
    # The frozen base weights are expected to be "missing" here; a *trainable*
    # name missing means the fingerprint check let something slip — treat as fatal.
    trainable_missing = [m for m in missing if m in set(ckpt["trainable_names"])]
    if trainable_missing:
        raise RuntimeError(f"{path}: trainable params missing from checkpoint: {trainable_missing}")

    _check_frozen_contact_complete(ckpt, model, path)

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
    """Warm-start common contact parameters, allowing new temporal and/or force params.

    This is deliberately narrower than :func:`load`: it does not restore the
    optimiser, scheduler, epoch, or RNG. The only trainable parameters that may be
    absent from the source checkpoint are ``cross_modal_temporal.*`` (adding the
    all-modality temporal block to a per-frame checkpoint), any param whose name
    contains ``"force"`` (the force branch — force tokens/linears, ``head_force``)
    or ``"motion"``, and the ``*_cond_linear`` input-conditioning projections.
    Everything else must match exactly: those keys are exempted from the
    arch-signature comparison symmetrically, and the precondition is that the
    target enables the cross-modal temporal block OR the force branch. It starts a
    temporal fine-tune, or a regime-(a) force warm-start, from a per-frame contact
    checkpoint without weakening the strict resume checks.

    The source may itself carry the temporal block provided the target's block is
    architecturally identical (same signature dict): its ``cross_modal_temporal.*``
    params then load like any other param — no longer "allowed missing", they
    simply load. A temporal source with a differing/disabled target block raises.
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
    # Exempt the modules a warm start may introduce from the signature comparison,
    # symmetrically on both sides: the cross-modal temporal block (temporal
    # fine-tune) and the force branch (regime (a)). A contact-only source has no
    # force keys; a force-enabled target does — popping both keeps the remaining
    # semantics an exact match.
    source_temporal = source_sig.pop(
        "cross_modal_temporal", {"enabled": False})
    target_temporal = target_sig.pop(
        "cross_modal_temporal", {"enabled": False})
    source_sig.pop("force", None)
    target_force = target_sig.pop("force", {"enabled": False})
    # Same treatment for the two later branches a warm start may introduce (or
    # drop): the motion tokens and the input conditioning. Both are absent from
    # every earlier checkpoint's signature, and their params are allowed-missing
    # below, so comparing them here would reject warm starts the loader supports.
    # One asymmetry survives the pop: a source that TRAINED cond projections must
    # not have them transplanted across a different injection site — pre_decoder
    # weights steer the decoder's inputs, post_decoder ones offset its outputs.
    # The tensors are shape-identical, so only this check separates them.
    src_injection = (source_sig.get("cond_input") or {}).get("injection", "pre_decoder")
    tgt_injection = (target_sig.get("cond_input") or {}).get("injection", "pre_decoder")
    if src_injection != tgt_injection and any(
            "cond_linear" in name for name in ckpt["trainable_state_dict"]):
        raise RuntimeError(
            f"{path}: warm-start source carries cond projections trained with "
            f"injection={src_injection!r} but the target uses "
            f"injection={tgt_injection!r}; the weights are shape-compatible yet "
            "semantically different")
    for key in ("motion", "cond_input", "pose_temporal"):
        source_sig.pop(key, None)
        target_sig.pop(key, None)
    if source_sig != target_sig:
        diffs = [
            f"  {key}: checkpoint={source_sig.get(key, 'ABSENT')!r} "
            f"run={target_sig.get(key, 'ABSENT')!r}"
            for key in sorted(set(source_sig) | set(target_sig))
            if source_sig.get(key) != target_sig.get(key)
        ]
        raise RuntimeError(
            f"{path}: warm-start architecture mismatch outside the temporal/force/"
            f"motion/cond_input modules:\n" + "\n".join(diffs))
    # A temporal source is allowed IFF the target's temporal architecture is
    # identical: then its ``cross_modal_temporal.*`` params simply load like any
    # other param. A differing (or disabled) target block still raises — the
    # weights would be shape-compatible yet semantically wrong.
    if source_temporal.get("enabled", False) and source_temporal != target_temporal:
        raise RuntimeError(
            f"{path}: warm-start source has cross_modal_temporal enabled but the "
            "target's block differs (or is disabled); a temporal source may only "
            "warm-start an identical temporal architecture. Source="
            f"{source_temporal!r} target={target_temporal!r}")
    if not (target_temporal.get("enabled", False) or target_force.get("enabled", False)):
        raise RuntimeError(
            "initialize_common_contact is only valid when the target enables the "
            "cross-modal temporal block or the force branch")

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
    invalid_missing = [
        name for name in missing
        if "cross_modal" not in name.lower()
        and "force" not in name.lower()
        and "motion" not in name.lower()
        and "cond_linear" not in name
    ]
    if invalid_missing:
        raise RuntimeError(
            f"{path}: warm start is missing trainable parameters outside the "
            f"temporal/force/motion/cond_input modules: {invalid_missing}")

    model.load_state_dict(source_state, strict=False)
    ckpt["warm_start_loaded_names"] = sorted(source_state)
    ckpt["warm_start_new_names"] = missing
    return ckpt
