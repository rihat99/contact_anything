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
            "blind_to_image": False,        # ablation: no image path into contact tokens
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
            "window_frames": None,           # None|odd>=3: attend only the central window
        },
        "force_head": {                     # Step 04: per-extremity 3D force regression
            "enabled": False,
            "force_keypoint_indices": None, # None = inherit the contact anchors (legacy);
                                            # else own MHR70 anchor list (enables force-only builds)
            "frame": "local_world_aligned", # local_world_aligned | local (consumed by physics loss)
            "mlp_depth": 2,
            "mlp_channel_div_factor": 4,
            "dropout": 0.0,
        },
        "force_temporal": {                 # Step 05: temporal attention over force tokens
            "enabled": False,               # requires model.force_head.enabled
            "bottleneck_dim": 256,          # project 1024-d force tokens before attention
            "num_layers": 1,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "attend": "per_token",          # joint (T*K tokens per clip) | per_token (T per slot)
            "causal": False,
            "dropout": 0.0,
            "position_scale": 1.0,          # multiply elapsed seconds before sinusoidal time PE
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
    "physics": {                        # Step 06: RNEA root-wrench physics loss on forces
        "enabled": False,               # requires model.force_head.enabled + a video dataset
        "use_warp": False,              # opt into BetterRobot's fused CUDA FK/RNEA lanes
        "model_path": None,             # null -> $BETTERHUMAN_MODELS_DIR then sibling checkout
        "lod": 1,                       # MHR level of detail (1 matches SAM-3D-Body)
        "gravity": 9.81,                # magnitude m/s^2; DIRECTION is per scene (gravity_world)
        "min_frames": 5,                # clips shorter than this are physics-ineligible
        "smoothing_kernel": [0.25, 0.5, 0.25],   # odd-length; [1.0] = smoothing off
        "max_cam_jump_m": None,         # null = off; else drop clips whose sampled-step camera-center jump exceeds this (m)
        "loss": {
            "residual": 1.0,            # sum rho(r_f) + rho(r_tau) root-wrench residual (objective)
            "residual_robust": {        # per-component residual robustifier (rho); square = original
                "kind": "square",       # square | pseudo_huber
                "delta_force": 1.0,     # pseudo-Huber transition for the 3 force components
                "delta_torque": 0.5,    # pseudo-Huber transition for the 3 torque components
            },
            "residual_force_weight": 1.0,   # weight on the force part of the residual objective
            "residual_torque_weight": 1.0,  # weight on the torque part (allocation signal ~20x weaker)
            "force_noncontact": 1.0,    # non-contact force penalty (form: noncontact_gate)
            "noncontact_gate": {        # force_noncontact form
                "kind": "soft_l2",      # soft_l2 = (1-p)*||f||^2 | hinge_l1 = hinge(p)*||f|| (exact zeros)
                "p_lo": 0.2,            # hinge_l1: full penalty at p <= p_lo
                "p_hi": 0.5,            # hinge_l1: zero penalty at p >= p_hi (linear ramp between)
            },
            "force_at_contact": 0.1,    # p*relu(contact_min_bw - ||f||)^2 at contacts
            "contact_min_bw": 0.05,     # min force at a contact, units of body weight
            "gate_frames": "all",       # all | residual: frames the prob-gated force terms use
            "force_smooth": 0.1,        # ||f_t - f_{t-1}||^2 on world-frame forces
            "force_l2": 0.01,           # ||f||^2
            "torque_l2": 0.01,          # ||tau_j||^2 (residual frames)
            "torque_smooth": 0.0,       # ||tau_j(t) - tau_j(t-1)||^2 (>=2 residual frames)
        },
    },
    "force_supervision": {              # supervised GT-force loss (corpus kindyn forces)
        "enabled": False,               # requires model.force_head.enabled; excludes physics.enabled
        "target_frame": "center",       # center | all (rows per clip contributing to the loss)
        "loss": {
            "force": 1.0,               # Huber on in-contact limb-frames (bw units)
            "huber_delta_bw": 0.5,      # smooth-L1 quadratic->linear transition (bw)
            "outlier_bw": 4.0,          # exclude limb-frames with |gt| above this (0 = off)
            "noncontact": 1.0,          # L1 zero-force penalty on non-contact limb-frames
        },
    },
    "loss": {"dice_eps": 1.0e-5, "grad_clip": 1.0},
    "train": {                          # Phase 4 efficiency flags (grad-asserted no-ops)
        "detach_interm_preds": True,    # run interm MHR/camera preds under no_grad
        "backbone_no_grad": True,       # wrap only the frozen backbone call in no_grad
        "freeze_contact": False,        # regime (a): freeze contact, train force branch only
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

_KNOWN_DATASETS = frozenset({"damon", "climbing", "climbing_videos", "climbing_corpus"})
_KNOWN_TARGETS = frozenset({"vertex", "joint"})
_TEMPORAL_PLACEMENTS = frozenset({"post_decoder", "between_layers", "pre_decoder"})
_TEMPORAL_ATTEND = frozenset({"joint", "per_token"})
_KNOWN_JOINT_SETS = frozenset(JOINT_SET_NAMES)
_CONTACT_POOL_MODES = frozenset({"attention", "concat", "per_token"})
_FORCE_FRAMES = frozenset({"local_world_aligned", "local", "root"})


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


def _validate_temporal_common(node: dict, path: str) -> None:
    """Validate the attend/shape/scale clauses shared by ``model.temporal`` and
    ``model.force_temporal``.

    Placement is validated separately (``force_temporal`` has no placement key —
    it is fixed to ``post_decoder``).
    """
    if node["attend"] not in _TEMPORAL_ATTEND:
        raise ValueError(
            f"{path}.attend must be one of {sorted(_TEMPORAL_ATTEND)}; "
            f"got {node['attend']!r}")
    bottleneck_dim = int(node["bottleneck_dim"])
    num_heads = int(node["num_heads"])
    if bottleneck_dim <= 0:
        raise ValueError(f"{path}.bottleneck_dim must be positive")
    if num_heads <= 0 or bottleneck_dim % num_heads:
        raise ValueError(
            f"{path}.bottleneck_dim must be divisible by "
            f"{path}.num_heads; got {bottleneck_dim} and {num_heads}")
    if int(node["num_layers"]) <= 0:
        raise ValueError(f"{path}.num_layers must be positive")
    if float(node["mlp_ratio"]) <= 0:
        raise ValueError(f"{path}.mlp_ratio must be positive")
    position_scale = float(node["position_scale"])
    if not math.isfinite(position_scale) or position_scale <= 0:
        raise ValueError(f"{path}.position_scale must be finite and positive")


_PHYSICS_WEIGHT_KEYS = frozenset({
    "residual", "force_noncontact", "force_at_contact", "contact_min_bw",
    "force_smooth", "force_l2", "torque_l2", "torque_smooth",
})

_RESIDUAL_ROBUST_KINDS = frozenset({"square", "pseudo_huber"})

_GATE_FRAMES = frozenset({"all", "residual"})

_NONCONTACT_GATE_KINDS = frozenset({"soft_l2", "hinge_l1"})


def _validate_physics(cfg: dict, force_head: dict) -> None:
    """Validate the ``physics:`` section (step 06). ``physics:`` numbers stay out of
    the checkpoint arch signature.
    """
    physics = cfg["physics"]
    gravity = float(physics["gravity"])
    if not math.isfinite(gravity) or gravity <= 0:
        raise ValueError("physics.gravity must be finite and positive")
    if int(physics["lod"]) < 0:
        raise ValueError("physics.lod must be a non-negative integer")

    kernel = physics["smoothing_kernel"]
    if (not isinstance(kernel, list) or len(kernel) == 0 or len(kernel) % 2 == 0
            or not all(isinstance(w, (int, float)) and not isinstance(w, bool) for w in kernel)):
        raise ValueError("physics.smoothing_kernel must be a non-empty odd-length list of numbers")
    if any(w < 0 for w in kernel):
        raise ValueError("physics.smoothing_kernel weights must be non-negative")
    if sum(kernel) <= 0:
        raise ValueError("physics.smoothing_kernel weights must have a positive sum")

    for key in _PHYSICS_WEIGHT_KEYS:
        value = float(physics["loss"][key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"physics.loss.{key} must be finite and >= 0")

    # Robust residual + camera-jerk filter (read via ``.get`` so the direct
    # ``_validate_physics`` unit test — which passes a pre-schema physics dict —
    # still runs; ``load_config`` always supplies both from the defaults).
    robust = physics["loss"].get("residual_robust")
    if robust is not None:
        if robust["kind"] not in _RESIDUAL_ROBUST_KINDS:
            raise ValueError(
                "physics.loss.residual_robust.kind must be one of "
                f"{sorted(_RESIDUAL_ROBUST_KINDS)}; got {robust['kind']!r}")
        for key in ("delta_force", "delta_torque"):
            delta = float(robust[key])
            if not math.isfinite(delta) or delta <= 0:
                raise ValueError(
                    f"physics.loss.residual_robust.{key} must be finite and positive")

    # Force/torque residual weights + gated-frame restriction (read via ``.get`` for
    # the same pre-schema-fixture reason as ``residual_robust`` above).
    for key in ("residual_force_weight", "residual_torque_weight"):
        weight = float(physics["loss"].get(key, 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"physics.loss.{key} must be finite and >= 0")
    gate_frames = str(physics["loss"].get("gate_frames", "all"))
    if gate_frames not in _GATE_FRAMES:
        raise ValueError(
            f"physics.loss.gate_frames must be one of {sorted(_GATE_FRAMES)}; "
            f"got {gate_frames!r}")

    # Non-contact penalty form (read via ``.get`` for the same pre-schema-fixture
    # reason as ``residual_robust`` above).
    gate = physics["loss"].get("noncontact_gate")
    if gate is not None:
        if gate["kind"] not in _NONCONTACT_GATE_KINDS:
            raise ValueError(
                "physics.loss.noncontact_gate.kind must be one of "
                f"{sorted(_NONCONTACT_GATE_KINDS)}; got {gate['kind']!r}")
        p_lo, p_hi = float(gate["p_lo"]), float(gate["p_hi"])
        if not (math.isfinite(p_lo) and math.isfinite(p_hi)
                and 0.0 <= p_lo < p_hi <= 1.0):
            raise ValueError(
                "physics.loss.noncontact_gate must satisfy 0 <= p_lo < p_hi <= 1; "
                f"got p_lo={p_lo}, p_hi={p_hi}")

    max_cam_jump = physics.get("max_cam_jump_m")
    if max_cam_jump is not None:
        max_cam_jump = float(max_cam_jump)
        if not math.isfinite(max_cam_jump) or max_cam_jump <= 0:
            raise ValueError("physics.max_cam_jump_m must be null or a finite positive number")

    # The physics_residual monitor reads the raw RNEA residual, which is computed
    # only when the residual objective actually runs — a zero residual weight would
    # leave the monitor permanently without data (the trainer then raises at eval).
    monitor = str((cfg.get("output") or {}).get("monitor") or "")
    if monitor.endswith("physics_residual") and float(physics["loss"]["residual"]) == 0.0:
        raise ValueError(
            "output.monitor '.../physics_residual' requires physics.loss.residual > 0 "
            "(the raw residual headline is computed only when the RNEA residual "
            "objective runs)")

    min_frames = int(physics["min_frames"])
    if min_frames < 3:
        raise ValueError("physics.min_frames must be >= 3")

    if not physics["enabled"]:
        return
    if not force_head["enabled"]:
        raise ValueError(
            "physics.enabled requires model.force_head.enabled=true "
            "(the physics loss supervises the predicted forces)")
    # ``.get`` so the direct pre-schema unit test (minimal force_head dict)
    # still runs; ``load_config`` always supplies ``frame`` from the defaults.
    if str(force_head.get("frame", "local_world_aligned")) == "root":
        raise ValueError(
            "physics.enabled requires model.force_head.frame 'local_world_aligned' "
            "or 'local' — the physics loss rotates predictions into the world "
            "through the camera extrinsics, which the 'root' frame does not use")
    if not any(entry["name"] in ("climbing_videos", "climbing_corpus")
               for entry in cfg["data"]["datasets"]):
        raise ValueError(
            "physics.enabled requires a video dataset (climbing_videos or "
            "climbing_corpus) in data.datasets")
    frames_per_clip = int(cfg["data"]["sequence"]["frames_per_clip"])
    if frames_per_clip < min_frames:
        raise ValueError(
            "physics.enabled requires data.sequence.frames_per_clip "
            f">= physics.min_frames ({frames_per_clip} < {min_frames})")
    # Residual frames are {2+r <= t <= T-3-r} with r = kernel radius (formula from
    # contact/physics/loss.py::_residual_frame_indices, duplicated here so config
    # validation does not import better_robot). Empty set = the residual objective —
    # the whole point of the physics loss — is silently dead.
    radius = len(kernel) // 2
    if 2 + radius > frames_per_clip - 3 - radius:
        raise ValueError(
            f"physics.enabled with frames_per_clip={frames_per_clip} and a "
            f"smoothing kernel of radius {radius} leaves zero residual frames "
            "({2+r <= t <= T-3-r}) — raise data.sequence.frames_per_clip to at "
            f"least {5 + 2 * radius} or shorten physics.smoothing_kernel")


def _validate_force_supervision(cfg: dict, force_head: dict) -> None:
    """Validate the ``force_supervision:`` section (supervised kindyn forces)."""
    fs = cfg["force_supervision"]
    target_frame = str(fs["target_frame"])
    if target_frame not in ("all", "center"):
        raise ValueError("force_supervision.target_frame must be 'all' or 'center'")
    for key in ("force", "noncontact", "outlier_bw"):
        value = float(fs["loss"][key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"force_supervision.loss.{key} must be finite and >= 0")
    delta = float(fs["loss"]["huber_delta_bw"])
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("force_supervision.loss.huber_delta_bw must be finite and positive")

    if not fs["enabled"]:
        return
    if not force_head["enabled"]:
        raise ValueError(
            "force_supervision.enabled requires model.force_head.enabled=true "
            "(the supervised loss reads the predicted forces)")
    if cfg["physics"]["enabled"]:
        raise ValueError(
            "force_supervision.enabled and physics.enabled are mutually exclusive — "
            "pick one supervision signal for the force branch")
    if not any(entry["name"] == "climbing_corpus" for entry in cfg["data"]["datasets"]):
        raise ValueError(
            "force_supervision.enabled requires a climbing_corpus dataset in "
            "data.datasets (GT forces come from the corpus kindyn_1.npz)")
    if target_frame == "center" and int(cfg["data"]["sequence"]["frames_per_clip"]) % 2 == 0:
        raise ValueError(
            "force_supervision.target_frame='center' requires an odd "
            "data.sequence.frames_per_clip")


def _validate_semantics(cfg: dict) -> None:
    """Cross-key checks that pure key validation cannot express."""
    topology = cfg["contact"]["topology"]
    topology_num_vertices(topology)   # raises NotImplementedError for 'mhr', ValueError for unknown

    primary = cfg["contact"]["primary_target"]
    if primary not in _KNOWN_TARGETS:
        raise ValueError(f"contact.primary_target must be one of {sorted(_KNOWN_TARGETS)}; got {primary!r}")

    targets = cfg["contact"]["targets"]
    contact_enabled = any(targets[name]["enabled"] for name in _KNOWN_TARGETS)
    force_head = cfg["model"]["force_head"]
    force_kp = force_head["force_keypoint_indices"]
    if not contact_enabled and not (force_head["enabled"] and force_kp is not None):
        # Force-only builds (no contact tokens/head at all) are legal, but only
        # with the force branch on and its own explicit anchors — null anchors
        # inherit from the contact tokens, which do not exist here.
        raise ValueError(
            "no contact target is enabled — enable at least one of vertex/joint, "
            "or configure a force-only build (model.force_head.enabled=true with "
            "explicit model.force_head.force_keypoint_indices)")

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
    if temporal["enabled"] and not contact_enabled:
        raise ValueError(
            "model.temporal.enabled requires an enabled contact target (the contact "
            "temporal module attends the contact tokens, which a force-only build "
            "does not create — use model.force_temporal for the force tokens)")
    if temporal["placement"] not in _TEMPORAL_PLACEMENTS:
        raise ValueError(
            f"model.temporal.placement must be one of {sorted(_TEMPORAL_PLACEMENTS)}; "
            f"got {temporal['placement']!r}")
    if (contact_head["blind_to_image"] and temporal["enabled"]
            and temporal["placement"] == "pre_decoder"):
        raise ValueError(
            "model.contact_head.blind_to_image with model.temporal.placement="
            "'pre_decoder' is a no-op: the pre_decoder branch exists only to build "
            "the image tensor the anchored update samples, and blind_to_image "
            "removes that update")
    _validate_temporal_common(temporal, "model.temporal")
    window_frames = temporal["window_frames"]
    if window_frames is not None and (int(window_frames) < 3 or int(window_frames) % 2 == 0):
        raise ValueError(
            f"model.temporal.window_frames must be null or an odd int >= 3; "
            f"got {window_frames!r}")

    if force_head["frame"] not in _FORCE_FRAMES:
        raise ValueError(
            f"model.force_head.frame must be one of {sorted(_FORCE_FRAMES)}; "
            f"got {force_head['frame']!r}")
    if force_kp is not None and (
        not isinstance(force_kp, list) or len(force_kp) == 0
        or not all(isinstance(i, int) and not isinstance(i, bool) for i in force_kp)
        or not all(0 <= i < 70 for i in force_kp)
    ):
        raise ValueError(
            "model.force_head.force_keypoint_indices must be null or a non-empty "
            f"list of MHR70 indices in [0, 70); got {force_kp!r}")
    if force_head["enabled"]:
        if force_kp is None:
            joint_enabled = targets["joint"]["enabled"]
            if not (joint_enabled and joint_set == "extremities_4" and pool_mode == "per_token"):
                raise ValueError(
                    "model.force_head.enabled requires the joint target enabled with "
                    "joint_set='extremities_4' and model.contact_head.pool_mode='per_token' "
                    "(force tokens reuse the four extremity contact anchors), or explicit "
                    "model.force_head.force_keypoint_indices to decouple the anchors")
        elif cfg["physics"]["enabled"]:
            raise ValueError(
                "physics.enabled requires model.force_head.force_keypoint_indices=null "
                "(the physics loss gates on the four extremity contact probabilities, "
                "which is only sound when the force anchors are the contact anchors)")

    force_temporal = cfg["model"]["force_temporal"]
    if force_temporal["enabled"] and not force_head["enabled"]:
        raise ValueError(
            "model.force_temporal.enabled requires model.force_head.enabled=true "
            "(force temporal attends the force tokens)")
    _validate_temporal_common(force_temporal, "model.force_temporal")

    _validate_physics(cfg, force_head)
    _validate_force_supervision(cfg, force_head)

    if cfg["train"]["freeze_contact"]:
        if not contact_enabled:
            raise ValueError(
                "train.freeze_contact=true requires an enabled contact target — a "
                "force-only build has no contact branch to freeze")
        if not force_head["enabled"]:
            raise ValueError(
                "train.freeze_contact=true requires model.force_head.enabled=true "
                "(there is nothing else to train)")
        if cfg["model"]["init_contact_checkpoint"] is None:
            raise ValueError(
                "train.freeze_contact=true requires model.init_contact_checkpoint "
                "(warm-start the frozen contact branch)")

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
        or cfg["data"]["datasets"][0]["name"] not in ("climbing_videos", "climbing_corpus")
    ):
        raise ValueError(
            "data.eval_split='test' requires a single climbing_videos or "
            "climbing_corpus dataset (the manually annotated test split)")

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
