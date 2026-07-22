"""Config loader for contact-head training.

Plain YAML with a single ``base:`` include and strict key validation — no
hydra, no omegaconf. :func:`load_config` reads a run config, deep-merges it
over the file named by its ``base:`` key (path relative to the repo root),
fills documented defaults, and hard-errors on any key that is not part of the
schema (typo protection). The resolved config is a plain ``dict``.

The schema mirrors ``configs/base.yaml``; the two must agree. The ``train.*``
efficiency flags (Phase 4) narrow the autograd graph during contact training
(they are grad-asserted no-ops for the trainable params); default ``true``.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml

from .targets import JOINT_SET_NAMES, topology_num_vertices

REPO_ROOT = Path(__file__).resolve().parents[1]

_CKPT_DIR = (
    "/data3/rikhat.akizhanov/.cache/huggingface/hub/"
    "models--facebook--sam-3d-body-dinov3/snapshots/"
    "11aaa346c7204874a1cbafe3d39a979080b2c55a"
)

# The schema *is* the default tree: every allowed key appears here with its
# default value. A ``dict`` value is a nested namespace (recursed into for both
# defaulting and validation); any other value (scalar, ``None``, list) is a
# leaf — its concrete value is not schema-checked.
DEFAULTS: dict[str, Any] = {
    "base": None,
    "model": {
        "checkpoint_path": f"{_CKPT_DIR}/model.ckpt",
        "mhr_model_path": f"{_CKPT_DIR}/assets/mhr_model.pt",
        "init_contact_checkpoint": None,   # optional contact-only warm start (not optimiser resume)
        "mask_embed_type": "v2",
        "contact_head": {
            "contact_keypoint_indices": None,   # None = list(range(21))
            "num_global_tokens": 3,
            "pool_mode": "concat",
            "dropout": 0.1,
            "mlp_depth": 4,
            "mlp_channel_div_factor": 2,
            "grid_size": 5,
            "grid_radius": 0.1,
        },
        "temporal": {                       # Phase 3: ContactTemporalModule
            "enabled": False,
            "placement": "post_decoder",    # post_decoder | between_layers | pre_decoder
            "bottleneck_dim": 256,           # project 1024-d contact tokens before attention
            "num_layers": 1,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "attend": "joint",              # joint (T*K tokens) | per_token (T per slot)
            "causal": False,
            "dropout": 0.0,
            "position_scale": 1.0,           # multiplier on elapsed seconds before time PE
        },
    },
    "contact": {
        "topology": "smpl",             # smpl(6890) | smplx(10475) | mhr -> NotImplementedError
        "primary_target": "vertex",     # headline metric target
        "targets": {
            "vertex": {
                "enabled": True,
                "weight": 1.0,
                "loss": {
                    "focal_alpha": 0.75,
                    "focal_gamma": 2.0,
                    "focal_weight": 5.0,
                    "dice_weight": 0.5,
                    "sparsity_weight": 0.002,
                },
            },
            "joint": {
                "enabled": False,
                "joint_set": "smplx_body_22",  # smplx_body_22 | extremities_4
                "weight": 1.0,
                "supervise_subset": None,       # None=all 22; 'observable_14'; or index list
                "derive_from_vertex": False,    # OFF by default (semantics differ, see targets.py)
                "use_confidence_weights": False,
                "loss": {
                    "focal_alpha": 0.5,
                    "focal_gamma": 2.0,
                    "focal_weight": 5.0,
                    "dice_weight": 0.5,
                    "sparsity_weight": 0.0,
                },
            },
        },
    },
    "data": {
        "datasets": [],                 # list of {name, config[, split]}
        "eval_split": "val",           # val | test (manual ClimbingVideos annotations)
        "val_ratio": 0.15,
        "seed": 42,
        "frames_per_batch": 32,         # B_clips = frames_per_batch // T (homogeneous T)
        "num_workers": 16,
        "sequence": {
            "frames_per_clip": 8,
            "frame_stride": 2,
            "jitter": True,
            "target_frame": "all",       # all | center (loss/metrics rows per clip)
        },
    },
    "loss": {"dice_eps": 1.0e-5, "grad_clip": 1.0},
    "train": {                          # Phase 4 efficiency flags (grad-asserted no-ops)
        "detach_interm_preds": True,    # run interm MHR/camera preds under no_grad
        "backbone_no_grad": True,       # wrap only the frozen backbone call in no_grad
    },
    "optim": {
        "lr": 1.0e-4,
        "weight_decay": 1.0e-4,
        "epochs": 20,
        "warmup_epochs": 1,
        "lr_min": 1.0e-5,
    },
    "logging": {
        "wandb": {          # consumed in Phase 4; present for forward-compat
            "enabled": True,
            "project": "contact-anything",
            "entity": None,
            "tags": [],
            "mode": "online",
        },
        "tensorboard": True,
        "tensorboard_metrics": None,    # null = all; otherwise exact scalar-tag allowlist
    },
    "output": {
        "dir": "./output",
        "exp_name": "contact",
        "log_freq": 10,
        "val_freq": 1,
        "save_freq": 5,
        "monitor": "val/vertex_f1",
    },
}

_KNOWN_DATASETS = frozenset({"damon", "climbing", "climbing_videos"})
_KNOWN_TARGETS = frozenset({"vertex", "joint"})
_TEMPORAL_PLACEMENTS = frozenset({"post_decoder", "between_layers", "pre_decoder"})
_TEMPORAL_ATTEND = frozenset({"joint", "per_token"})
_KNOWN_JOINT_SETS = frozenset(JOINT_SET_NAMES)
_CONTACT_POOL_MODES = frozenset({"attention", "concat", "per_token"})


def _deep_merge(base: dict, override: dict) -> dict:
    """Return ``base`` with ``override`` recursively applied (override wins).

    Both sides are deep-copied, so the resolved config never aliases nested
    dicts/lists in ``DEFAULTS`` (or the raw YAML): mutating one resolved config
    cannot leak into ``DEFAULTS`` or a later ``load_config`` in the same process.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_raw(path: Path) -> dict:
    """Read one config and splice in its ``base:`` include (child over base)."""
    raw = yaml.safe_load(path.read_text()) or {}
    base = raw.get("base")
    if base:
        base_raw = _load_raw(REPO_ROOT / base)
        raw = _deep_merge(base_raw, raw)
    return raw


def _validate_keys(node: dict, schema: dict, path: str = "") -> None:
    """Reject any key absent from ``schema`` or with the wrong namespace type.

    A schema value that is a ``dict`` marks a *namespace*: the config must supply a
    mapping there too (a scalar/``null`` would otherwise skip validation and fail
    later in unrelated build code). Non-dict schema values are leaves (their
    concrete value is not type-checked).
    """
    for key, value in node.items():
        dotted = f"{path}.{key}" if path else key
        if key not in schema:
            raise ValueError(f"unknown config key: {dotted!r}")
        sub = schema[key]
        if isinstance(sub, dict):
            if not isinstance(value, dict):
                raise ValueError(
                    f"config key {dotted!r} must be a mapping (namespace); "
                    f"got {type(value).__name__}")
            _validate_keys(value, sub, dotted)


def _validate_semantics(cfg: dict) -> None:
    """Cross-key checks that pure key validation cannot express."""
    topology = cfg["contact"]["topology"]
    topology_num_vertices(topology)   # raises NotImplementedError for 'mhr', ValueError for unknown

    primary = cfg["contact"]["primary_target"]
    if primary not in _KNOWN_TARGETS:
        raise ValueError(f"contact.primary_target must be one of {sorted(_KNOWN_TARGETS)}; got {primary!r}")

    targets = cfg["contact"]["targets"]
    if not any(targets[name]["enabled"] for name in _KNOWN_TARGETS):
        raise ValueError("no contact target is enabled — enable at least one of vertex/joint")

    joint_cfg = targets["joint"]
    joint_set = joint_cfg["joint_set"]
    if joint_set not in _KNOWN_JOINT_SETS:
        raise ValueError(
            f"contact.targets.joint.joint_set must be one of {sorted(_KNOWN_JOINT_SETS)}; "
            f"got {joint_set!r}")
    if joint_set == "extremities_4" and joint_cfg["supervise_subset"] is not None:
        raise ValueError(
            "contact.targets.joint.supervise_subset must be null for "
            "joint_set='extremities_4'")

    contact_head = cfg["model"]["contact_head"]
    pool_mode = str(contact_head["pool_mode"])
    if pool_mode not in _CONTACT_POOL_MODES:
        raise ValueError(
            f"model.contact_head.pool_mode must be one of {sorted(_CONTACT_POOL_MODES)}; "
            f"got {pool_mode!r}")
    num_global = int(contact_head["num_global_tokens"])
    if num_global < 0:
        raise ValueError("model.contact_head.num_global_tokens must be non-negative")
    if pool_mode == "per_token":
        anchors = contact_head["contact_keypoint_indices"]
        num_anchors = 21 if anchors is None else len(anchors)
        token_count = num_anchors + num_global
        output_dims = {
            "vertex": topology_num_vertices(topology),
            "joint": len(JOINT_SET_NAMES[joint_set]),
        }
        mismatched = {
            name: output_dims[name]
            for name in _KNOWN_TARGETS
            if targets[name]["enabled"] and output_dims[name] != token_count
        }
        if mismatched:
            raise ValueError(
                "model.contact_head.pool_mode='per_token' requires every enabled "
                f"target output dimension to equal the total token count {token_count}; "
                f"got {mismatched}")

    temporal = cfg["model"]["temporal"]
    if temporal["placement"] not in _TEMPORAL_PLACEMENTS:
        raise ValueError(
            f"model.temporal.placement must be one of {sorted(_TEMPORAL_PLACEMENTS)}; "
            f"got {temporal['placement']!r}")
    if temporal["attend"] not in _TEMPORAL_ATTEND:
        raise ValueError(
            f"model.temporal.attend must be one of {sorted(_TEMPORAL_ATTEND)}; "
            f"got {temporal['attend']!r}")
    bottleneck_dim = int(temporal["bottleneck_dim"])
    num_heads = int(temporal["num_heads"])
    if bottleneck_dim <= 0:
        raise ValueError("model.temporal.bottleneck_dim must be positive")
    if num_heads <= 0 or bottleneck_dim % num_heads:
        raise ValueError(
            "model.temporal.bottleneck_dim must be divisible by "
            f"model.temporal.num_heads; got {bottleneck_dim} and {num_heads}")
    if int(temporal["num_layers"]) <= 0:
        raise ValueError("model.temporal.num_layers must be positive")
    if float(temporal["mlp_ratio"]) <= 0:
        raise ValueError("model.temporal.mlp_ratio must be positive")
    position_scale = float(temporal["position_scale"])
    if not math.isfinite(position_scale) or position_scale <= 0:
        raise ValueError("model.temporal.position_scale must be finite and positive")

    sequence = cfg["data"]["sequence"]
    target_frame = str(sequence["target_frame"])
    if target_frame not in ("all", "center"):
        raise ValueError("data.sequence.target_frame must be 'all' or 'center'")
    frames_per_clip = int(sequence["frames_per_clip"])
    if frames_per_clip <= 0:
        raise ValueError("data.sequence.frames_per_clip must be positive")
    if target_frame == "center" and frames_per_clip % 2 == 0:
        raise ValueError(
            "data.sequence.target_frame='center' requires an odd frames_per_clip")

    for entry in cfg["data"]["datasets"]:
        if not isinstance(entry, dict) or "name" not in entry or "config" not in entry:
            raise ValueError(f"each data.datasets entry needs 'name' and 'config'; got {entry!r}")
        if entry["name"] not in _KNOWN_DATASETS:
            raise ValueError(
                f"unknown dataset {entry['name']!r}; choose from {sorted(_KNOWN_DATASETS)}")
        extra = set(entry) - {"name", "config", "split"}
        if extra:
            raise ValueError(f"unknown keys in data.datasets entry: {sorted(extra)}")

    eval_split = str(cfg["data"]["eval_split"])
    if eval_split not in ("val", "test"):
        raise ValueError("data.eval_split must be 'val' or 'test'")
    if eval_split == "test" and (
        len(cfg["data"]["datasets"]) != 1
        or cfg["data"]["datasets"][0]["name"] != "climbing_videos"
    ):
        raise ValueError(
            "data.eval_split='test' requires a ClimbingVideos-only data config")

    tb_metrics = cfg["logging"]["tensorboard_metrics"]
    if tb_metrics is not None and (
        not isinstance(tb_metrics, list)
        or not all(isinstance(metric, str) and metric for metric in tb_metrics)
    ):
        raise ValueError("logging.tensorboard_metrics must be null or a list of scalar tags")


def load_config(path: str | Path) -> dict:
    """Load, merge (``base:`` include), default-fill and validate a run config.

    :param path: Path to the run YAML.
    :returns: The resolved config as a plain ``dict``.
    :raises ValueError: on any unknown key or invalid value.
    :raises NotImplementedError: if ``contact.topology`` is ``"mhr"``.
    """
    raw = _load_raw(Path(path))
    merged = _deep_merge(DEFAULTS, raw)
    _validate_keys(merged, DEFAULTS)
    _validate_semantics(merged)
    return merged
