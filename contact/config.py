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

from .targets import JOINT_SET_NAMES

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
        "extra_token_attention": "mutual",  # decoder mask over the appended token blocks:
                                            # mutual (default) = contact/force/motion fully
                                            # inter-attend (original tokens still attend none
                                            # of them); causal = legacy regime — no earlier
                                            # block attends a later one
                                            # (original ⊥ contact ⊥ force ⊥ motion)
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
        "force_head": {                     # Step 04: per-extremity 3D force regression
            "enabled": False,
            "force_keypoint_indices": None, # None = inherit the contact anchors (legacy);
                                            # else own MHR70 anchor list (enables force-only builds)
            "frame": "local_world_aligned", # local_world_aligned | local (consumed by physics loss)
            "mlp_depth": 2,
            "mlp_channel_div_factor": 4,
            "dropout": 0.0,
            "contact_gate": {               # gate the final force output by the (detached)
                "enabled": False,           # kindyn_6 contact logits, one per force group
                "sharpness": 4.0,           # gate = sigmoid(sharpness * contact logit)
            },
        },
        "motion_head": {                    # motion tokens v2: per-joint linear vel + acc
            "enabled": False,
            # Always explicit (no contact inheritance, no global tokens). Default =
            # the six kindyn force anchors + MHR70 9 (left hip) for the pelvis token.
            "motion_keypoint_indices": [62, 41, 15, 18, 17, 20, 9],
            # false = no per-layer anchored token update (posemb + grid-sampled
            # image features): pure learned queries. The anchor list then only
            # names/counts the slots.
            "anchored": True,
            "mlp_depth": 2,
            "mlp_channel_div_factor": 4,
            "dropout": 0.0,
        },
        "pose_temporal": {                  # RoPE temporal attention over the POSE token (E2)
            "enabled": False,               # DELIBERATE exception to the frozen-pose rule:
                                            # the final MHR output is recomputed from a
                                            # temporally-mixed pose token (zero-init gates
                                            # = frozen behavior at init)
            "type": "rope",                 # only value
            "time_scale": 25.0,             # seconds -> RoPE positions
            "max_rel_sec": 2.5,             # attention window half-width (seconds);
                                            # null disables the window
            "num_layers": 4,
            "num_heads": 16,                # decoder dim / num_heads must be even
            "mlp_ratio": 2.0,
            "dropout": 0.0,
        },
        "cross_modal_temporal": {           # THE post-decoder mixing brick: ONE temporal
            "enabled": False,               # transformer over the CONCATENATION of the chosen
                                            # modality token blocks. Every listed token attends
                                            # every other across the clip's frames; the dt = 0
                                            # diagonal gives within-frame cross-modal attention.
            "type": "rope",                 # only value: a native-dim block with
                                            # relative-time rotary positions
            "modalities": ["contact", "force"],  # >= 2 of pose|contact|force|motion; each needs
                                            # its branch enabled. 'pose' WRITES the pose token:
                                            # the final MHR output is recomputed from it
                                            # (needs pose_supervision for training)
            "num_layers": 4,
            "num_heads": 16,                # decoder dim / num_heads must be even
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "time_scale": 25.0,             # rope only: seconds -> RoPE positions
            "max_rel_sec": 2.5,             # rope only: attention window half-width (seconds);
                                            # null disables the window
        },
    },
    "contact": {
        "primary_target": "joint",      # headline metric target
        "targets": {
            "joint": {
                "enabled": False,
                "joint_set": "smplx_body_22",  # smplx_body_22 | extremities_4 | kindyn_6
                "weight": 1.0,
                "supervise_subset": None,       # None=all 22; 'observable_14'; or index list
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
        # Load precomputed backbone embeddings (<root>/features/embedding, bf16)
        # for climbing_corpus datasets and skip the frozen backbone entirely.
        # Build the cache with scripts/precompute_embeddings.py first; a missing
        # per-frame file is a hard error (never silent live fallback).
        "embedding_cache": False,
        "sequence": {
            "frames_per_clip": 8,
            "frame_stride": 2,
            "jitter": True,
            "target_frame": "all",       # all | center (loss/metrics rows per clip)
            "eval_full_scenes": False,   # eval/test protocol: one single-pass clip per
                                         # (scene, person) = the longest valid run
                                         # (batch = 1 clip; long-sequence modules only)
            "eval_max_frames": None,     # cap the full-scene clip length (frames after
                                         # striding); None = whole run. ~0.1 GiB/frame
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
        "gt_frame": "root",             # root | world — coordinate frame the loader emits the GT
                                        # forces/levers in (world yaw is unobservable from a crop)
        "units": "bw",                  # bw | newtons — GT force units; the loss deltas/cuts
                                        # (huber_delta_bw, outlier_bw) are in these units
        "confidence": True,             # weight every loss term's rows by kindyn's per-frame
                                        # force_confidence (numerator AND mass, exact DDP mean)
        "loss": {
            "force": 1.0,               # Huber on in-contact limb-frames (bw units)
            "huber_delta_bw": 0.5,      # smooth-L1 quadratic->linear transition (bw)
            "outlier_bw": 4.0,          # exclude limb-frames with |gt| above this (0 = off)
            "noncontact": 1.0,          # L1 zero-force penalty on non-contact limb-frames
            "sum_force": 0.0,           # Huber on the six-group net force vs GT (all groups)
            "sum_torque": 0.0,          # Huber on the net torque sum(r x f) vs GT (bw*m)
            "huber_delta_bwm": 0.1,     # sum_torque smooth-L1 transition (bw*m)
            "group_weights": None,      # per-group Huber weights (kindyn order); null = uniform
        },
    },
    "motion_supervision": {             # supervised kindyn vel/acc loss (motion tokens v2)
        "enabled": False,               # requires model.motion_head.enabled
        "target_frame": "all",          # all | center (rows per clip contributing to the loss)
        "joint_names": None,            # null = all 7 motion joints; else an ordered subset
        "root_convention": "twist",     # pelvis slot LINEAR frame: twist (BVR body twist) |
                                        # rotated_world (R^T of the world central difference) |
                                        # gravity_view (GVHMR's frame: vertical = the scene's
                                        # FITTED gravity, azimuth = the camera view direction —
                                        # gravity-aligned and free of body roll/pitch, so pose
                                        # orientation error no longer rotates the target).
                                        # The angular pair is the SE3-log body rate regardless.
                                        # `standardize` is FRAME-specific: recompute it when
                                        # this changes.
        "root_source": "kindyn",        # rig the targets are differentiated from: kindyn
                                        # (SMPL-X joints_world + kindyn root) | mhr
                                        # (mhr_sup_1 MHR70 limbs + mhr_1 q_world root — the
                                        # SAME rig as the prediction, so the pose-derived
                                        # twist carries no cross-rig offset). The two have
                                        # DIFFERENT target distributions: `standardize` is
                                        # per-source and must be recomputed when this changes.
        "angular": False,               # append the root twist's angular vel/acc (12-dim target;
                                        # requires twist + joint_names ['pelvis'])
        "target_smooth_sec": 0.12,      # Gaussian width (s) on the root trajectory; 0 = raw
        "standardize": {                # per-joint per-component tables, [K][2][3] (vel; acc)
            "mean": None,               # root axes, m/s and m/s^2; required when enabled
            "std": None,                # (angular: [K][4][3] — vel, acc, ang_vel, ang_acc)
        },
        "loss": {
            "vel": 1.0,                 # Huber weight on the standardized velocity
            "acc": 1.0,                 # Huber weight on the standardized acceleration
            "ang_vel": 1.0,             # Huber weight on the angular velocity (angular runs)
            "ang_acc": 1.0,             # Huber weight on the angular acceleration (angular runs)
            "huber_delta": 1.0,         # smooth-L1 transition (standardized units)
            "outlier_acc_ms2": 50.0,    # TRAIN-only per-(frame, joint) cut on |acc_world|; 0 = off
        },
    },
    "contact_consistency": {            # world-frame velocity of the six extremity keypoints
        "enabled": False,               # weighted by the PREDICTED contact probability —
                                        # stable contact implies a still joint. Requires the
                                        # kindyn_6 joint target + a trainable pose path.
        "detach_gate": True,            # detach the contact probs in the gate (grad -> pose
                                        # path only; false also pushes probs DOWN at moving
                                        # joints — the supervised focal loss must counter)
        "loss": {
            "vel": 1.0,                 # weight on gate * huber(|v_world|)
            "huber_delta_ms": 0.5,      # smooth-L1 transition (m/s)
        },
    },
    "force_consistency": {              # linear Newton residual in body-weight units:
        "enabled": False,               # a_root/g must equal gravity + the net predicted
                                        # contact force (root->world via the GT kindyn root
                                        # rotation). Mass cancels in bw units. Grad -> pose
                                        # (via the root acceleration) + force head.
        "ramp": {                       # weight warm-up (the residual is unstable early):
            "start_epoch": 3,           # scale 0 before this epoch, then linear
            "epochs": 3,                # ... reaching 1.0 at start_epoch + epochs - 1
        },
        "smoothing_kernel": [0.25, 0.5, 0.25],
                                        # windowed mean on the predicted world root before
                                        # double-differencing; [1.0] = off
        "loss": {
            "residual": 1.0,            # weight on huber(residual) (dimensionless, bw)
            "huber_delta_bw": 1.0,      # smooth-L1 transition (bw)
        },
    },
    "pose_supervision": {               # kindyn-MHR pseudo-GT pose loss (E2)
        "enabled": False,               # requires model.pose_temporal.enabled + corpus
                                        # mhr_1.npz files (scripts/convert_kindyn_to_mhr.py)
        "loss": {
            "pose": 1.0,                # Huber weight on the 125 local MHR q channels
            "acc": 0.0,                 # Huber weight on clip-wise q second differences
                                        # (pred vs GT) — the explicit smoothness term
            "shape": 0.0,               # L2 on the 45 blendshape outputs vs the mhr_1 v2
                                        # GT identity (per-person, mesh-fitted) — the
                                        # full-parameter alternative to shape_rail
            "bones": 0.0,               # Huber on the 6 per-frame flexible bone-geometry
                                        # slots (lbs 130..135: spine/neck/shoulder-width/
                                        # arm/hip-width/leg lengths) vs the mhr_1 v3 GT.
                                        # Per-channel mean. The audit: unsupervised AND
                                        # unrailed, 98% of the body-size drift lived here
            "scale": 0.0,               # Huber on the 68 per-person scale slots (lbs
                                        # 136..203) vs the same GT — the head's 28 PCA
                                        # coefficients already expanded inside
                                        # mhr_model_params. Per-channel mean
            "huber_delta_bones": 0.05,  # smooth-L1 transition for bones/scale (slot units)
            "shape_rail": 0.0,          # L2 pinning the 45 blendshape outputs to the FROZEN
                                        # readout's own values (shape_frozen stash) — nothing
                                        # else supervises them
            "scale_rail": 0.0,          # same L2 for the 28 bone-scale outputs. REDUNDANT
                                        # and opposed once loss.scale is on (the rail pins
                                        # to the frozen value the GT is correcting)
            "huber_delta": 0.1,         # smooth-L1 transition (radians)
        },
        "fit_err_confidence": False,    # weight bones/scale rows by the mhr_1 mesh-fit
                                        # residual: 1 / (1 + (fit_err_cm / ref)^2)
        "fit_err_ref_cm": 2.0,          # residual at which the row weight is 0.5
                                        # (corpus mean fit_err is 0.68 cm)
        "mhr": {                        # BetterHuman archive for q <-> params conversion
            "model_path": None,         # null resolves like the physics adapter
            "lod": 1,
        },
    },
    "keypoint_supervision": {           # kindyn joints_world keypoint losses — the SAM3D-style
        "enabled": False,               # stabilizers for pose/camera fine-tuning (video scenes)
        "loss": {
            "kp2d": 1.0,                # Huber on crop-normalized 2D reprojection (the CLIFF-
                                        # style term that constrains the camera head)
            "kp3d": 0.5,                # Huber on mean-hips-relative camera-frame 3D (metres)
            "kp3d_abs": 0.25,           # Huber on ABSOLUTE camera-frame 3D (metres) — pins
                                        # pred_cam_t depth with the metric extrinsics
            "vert": 0.0,                # Huber on the mean-hips-relative camera-frame vertex
                                        # subset (metres): body SHAPE, which 70 sparse
                                        # landmarks barely constrain
            "vert_abs": 0.0,            # same ABSOLUTE — the body-SIZE + depth anchor. The
                                        # audit's regression channel: size drifted
                                        # +3.9% -> -3.3% while keypoint error looked fine
            "kp_vel": 0.0,              # Huber on WORLD-frame keypoint velocity (central
                                        # stencil over the clip; extrinsics loss-only) vs the
                                        # finite-differenced kindyn joints_world
            "kp_acc": 0.0,              # same for acceleration — the explicit smoothness term
            "huber_delta_2d": 0.05,     # crop-normalized units (crop spans [-0.5, 0.5])
            "huber_delta_3d": 0.1,      # metres
            "huber_delta_vel": 0.5,     # m/s
            "huber_delta_acc": 2.0,     # m/s^2
            "outlier_acc": 50.0,        # drop rows whose GT keypoint acc exceeds this (m/s^2)
            "cam_rail": 0.0,            # trust region on pred_cam_t vs the FROZEN model's own
                                        # output (pred_cam_t_frozen): relu(|dev| - margin) —
                                        # zero for a healthy model, closes the constant-offset
                                        # null space of the derivative terms
            "rot_rail": 0.0,            # same trust region on global_rot (geodesic radians)
            "cam_rail_margin_m": 0.5,   # rail margin (metres)
            "rot_rail_margin_rad": 0.2, # rail margin (radians, ~11.5 deg)
        },
        # Per-joint weights. Every joint term is a weighted MEAN over the 70
        # MHR70 keypoints, so these do not change the term's overall scale —
        # only the relative pull of each group. NOTE the pre-2026-08-29 terms
        # were SUMS over 13 joints: multiply a historical loss weight by 13.
        "joint_weights": {
            "fingers": 0.1,             # the 40 finger/thumb keypoints (wrists excluded):
                                        # least reliable part of the mesh fit, negligible
                                        # lever arm on the body pose
            "face": 1.0,                # nose/eyes/ears. The audit found the head GT bad in
                                        # 4 of 6 inspected scenes (inherited from kindyn's
                                        # own SMPL-X fit) — the knob to turn down if the
                                        # head still misbehaves
        },
        "fit_err_confidence": False,    # weight rows by the mhr_1 mesh-fit residual:
                                        # 1 / (1 + (fit_err_cm / ref)^2)
        "fit_err_ref_cm": 2.0,          # residual at which the row weight is 0.5
    },
    "loss": {"dice_eps": 1.0e-5, "grad_clip": 1.0},
    "train": {                          # Phase 4 efficiency flags (grad-asserted no-ops)
        "detach_interm_preds": True,    # run interm MHR/camera preds under no_grad
        "backbone_no_grad": True,       # wrap only the frozen backbone call in no_grad
        "compile_backbone": False,      # torch.compile the frozen backbone (~1.2x step)
        "freeze_contact": False,        # regime (a): freeze contact, train force branch only
        "finetune_pose_head": False,    # train a COPY of head_pose.proj applied to the FINAL
                                        # pose token only — in-decoder interm predictions keep
                                        # the frozen original (split-head). Deliberate exception
                                        # to the frozen-pose rule; needs pose_supervision or
                                        # keypoint_supervision
        "finetune_camera_head": False,  # same split for head_camera.proj (s, tx, ty readout);
                                        # needs keypoint_supervision (kp2d) — the only
                                        # loss that constrains the camera
        "pose_head_lr_scale": 0.1,      # lr multiplier for the fine-tuned head param group(s)
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
            "enabled": False,   # user 2026-08-28: wandb off by default
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
        "monitor": "test/joint_f1",
    },
}

_KNOWN_DATASETS = frozenset({"climbing_corpus"})
_KNOWN_TARGETS = frozenset({"joint"})
_MODALITIES = ("pose", "contact", "force", "motion")
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


def _validate_temporal_shape(node: dict, path: str) -> None:
    """Validate the block-shape clauses shared by every temporal section."""
    if int(node["num_layers"]) <= 0:
        raise ValueError(f"{path}.num_layers must be positive")
    if int(node["num_heads"]) <= 0:
        raise ValueError(f"{path}.num_heads must be positive")
    if float(node["mlp_ratio"]) <= 0:
        raise ValueError(f"{path}.mlp_ratio must be positive")
    dropout = float(node["dropout"])
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"{path}.dropout must be in [0, 1); got {dropout}")


def _validate_rope_temporal(node: dict, path: str) -> None:
    """Validate a RoPE temporal section.

    RoPE blocks run natively at the decoder dim (no bottleneck adapter) and
    are bidirectional-only, so the sliding-window keys have no counterpart
    here.
    """
    _validate_temporal_shape(node, path)
    time_scale = float(node["time_scale"])
    if not math.isfinite(time_scale) or time_scale <= 0:
        raise ValueError(f"{path}.time_scale must be finite and positive")
    max_rel_sec = node["max_rel_sec"]
    if max_rel_sec is not None:
        max_rel_sec = float(max_rel_sec)
        if not math.isfinite(max_rel_sec) or max_rel_sec <= 0:
            raise ValueError(
                f"{path}.max_rel_sec must be finite and positive, or null")




def _validate_pose_temporal(node: dict) -> None:
    """Validate ``model.pose_temporal`` (RoPE only — the sliding module is gone)."""
    path = "model.pose_temporal"
    ptype = node.get("type", "rope")
    if ptype != "rope":
        raise ValueError(
            f"{path}.type must be 'rope'; the sliding-window temporal module is "
            f"retired. Got {ptype!r}")
    _validate_rope_temporal(node, path)


def _enabled_modalities(cfg: dict) -> dict:
    """Which modality token blocks the configured build creates."""
    contact_enabled = any(
        cfg["contact"]["targets"][name]["enabled"] for name in _KNOWN_TARGETS)
    return {
        "pose": True,
        "contact": contact_enabled,
        "force": bool(cfg["model"]["force_head"]["enabled"]),
        "motion": bool(cfg["model"]["motion_head"]["enabled"]),
    }


def _validate_modalities(cfg: dict, node: dict, path: str, min_count: int) -> None:
    """Validate a ``modalities`` list against the build's enabled branches."""
    mods = node["modalities"]
    if (not isinstance(mods, list) or len(mods) < min_count
            or len(set(mods)) != len(mods)
            or any(m not in _MODALITIES for m in mods)):
        raise ValueError(
            f"{path}.modalities must be a duplicate-free list of >= {min_count} "
            f"of {list(_MODALITIES)}; got {mods!r}")
    available = _enabled_modalities(cfg)
    missing = [m for m in mods if not available[m]]
    if missing:
        raise ValueError(
            f"{path}.modalities {missing} have no token block in this build "
            "(enable the corresponding contact target / force_head / motion_head)")


_PHYSICS_WEIGHT_KEYS = frozenset({
    "residual", "force_noncontact", "force_at_contact", "contact_min_bw",
    "force_smooth", "force_l2", "torque_l2", "torque_smooth",
})

_RESIDUAL_ROBUST_KINDS = frozenset({"square", "pseudo_huber"})

_GATE_FRAMES = frozenset({"all", "residual"})

_NONCONTACT_GATE_KINDS = frozenset({"soft_l2", "hinge_l1"})

# The kindyn motion joints (contact.data.climbing_corpus.MOTION_JOINT_NAMES);
# duplicated so config validation stays free of loader/torch imports.
_MOTION_JOINT_NAMES = ("left_wrist", "right_wrist", "left_foot", "right_foot",
                       "left_ankle", "right_ankle", "pelvis")
_NUM_MOTION_JOINTS = len(_MOTION_JOINT_NAMES)

_ROOT_CONVENTIONS = frozenset({"twist", "rotated_world", "gravity_view"})


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
    if not any(entry["name"] == "climbing_corpus" for entry in cfg["data"]["datasets"]):
        raise ValueError(
            "physics.enabled requires a climbing_corpus dataset in data.datasets "
            "(per-frame extrinsics + gravity come from the corpus loader)")
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
    if str(fs["gt_frame"]) not in ("root", "world"):
        raise ValueError("force_supervision.gt_frame must be 'root' or 'world'")
    if str(fs["units"]) not in ("bw", "newtons"):
        raise ValueError("force_supervision.units must be 'bw' or 'newtons'")
    if not isinstance(fs["confidence"], bool):
        raise ValueError("force_supervision.confidence must be a boolean")
    for key in ("force", "noncontact", "sum_force", "sum_torque", "outlier_bw"):
        value = float(fs["loss"][key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"force_supervision.loss.{key} must be finite and >= 0")
    for key in ("huber_delta_bw", "huber_delta_bwm"):
        delta = float(fs["loss"][key])
        if not math.isfinite(delta) or delta <= 0:
            raise ValueError(f"force_supervision.loss.{key} must be finite and positive")
    group_weights = fs["loss"]["group_weights"]
    if group_weights is not None:
        if not isinstance(group_weights, (list, tuple)) or not group_weights:
            raise ValueError(
                "force_supervision.loss.group_weights must be null or a non-empty list")
        for w in group_weights:
            if not isinstance(w, (int, float)) or not math.isfinite(float(w)) or float(w) <= 0:
                raise ValueError(
                    "force_supervision.loss.group_weights entries must be finite and > 0")
        anchors = force_head.get("force_keypoint_indices")
        if anchors is not None and len(group_weights) != len(anchors):
            raise ValueError(
                f"force_supervision.loss.group_weights has {len(group_weights)} entries "
                f"but model.force_head.force_keypoint_indices has {len(anchors)}")

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


def _pose_trainable_paths(cfg: dict) -> list:
    """Names of the enabled config paths that move the frozen pose outputs."""
    paths = []
    if cfg["model"]["pose_temporal"]["enabled"]:
        paths.append("model.pose_temporal")
    if cfg["train"]["finetune_pose_head"]:
        paths.append("train.finetune_pose_head")
    node = cfg["model"]["cross_modal_temporal"]
    if node["enabled"] and "pose" in node["modalities"]:
        paths.append("model.cross_modal_temporal (pose modality)")
    return paths


def _validate_motion(cfg: dict) -> None:
    """Validate ``model.motion_head`` / ``motion_supervision`` (motion tokens v2)."""
    motion_head = cfg["model"]["motion_head"]
    motion_kp = motion_head["motion_keypoint_indices"]
    if (not isinstance(motion_kp, list) or len(motion_kp) == 0
            or not all(isinstance(i, int) and not isinstance(i, bool) for i in motion_kp)
            or not all(0 <= i < 70 for i in motion_kp)):
        raise ValueError(
            "model.motion_head.motion_keypoint_indices must be a non-empty list of "
            f"MHR70 indices in [0, 70); got {motion_kp!r}")
    if not isinstance(motion_head["anchored"], bool):
        raise ValueError(
            "model.motion_head.anchored must be a boolean; got "
            f"{motion_head['anchored']!r}")

    _validate_pose_temporal(cfg["model"]["pose_temporal"])

    ps = cfg["pose_supervision"]
    for key in ("pose", "acc", "shape", "shape_rail", "scale_rail", "bones",
                "scale"):
        value = float(ps["loss"][key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"pose_supervision.loss.{key} must be finite and >= 0")
    for key in ("huber_delta", "huber_delta_bones"):
        value = float(ps["loss"][key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"pose_supervision.loss.{key} must be finite and positive")
    _validate_fit_err_confidence(ps, "pose_supervision")
    if ps["enabled"] and float(ps["loss"]["scale"]) > 0 and float(
            ps["loss"]["scale_rail"]) > 0:
        raise ValueError(
            "pose_supervision.loss.scale and loss.scale_rail are opposed — the "
            "rail pins the 28 scale coefficients to the FROZEN readout while the "
            "GT term corrects them. Set scale_rail to 0")
    if ps["enabled"]:
        if not _pose_trainable_paths(cfg):
            raise ValueError(
                "pose_supervision.enabled requires a trainable pose path: "
                "model.pose_temporal.enabled, train.finetune_pose_head, or the "
                "'pose' modality in model.cross_modal_temporal")
        if not any(entry["name"] == "climbing_corpus"
                   for entry in cfg["data"]["datasets"]):
            raise ValueError(
                "pose_supervision.enabled requires a climbing_corpus dataset in "
                "data.datasets (pose pseudo-GT comes from the corpus mhr_1.npz)")

    ms = cfg["motion_supervision"]
    target_frame = str(ms["target_frame"])
    if target_frame not in ("all", "center"):
        raise ValueError("motion_supervision.target_frame must be 'all' or 'center'")
    if ms["root_source"] not in ("kindyn", "mhr"):
        raise ValueError(
            f"motion_supervision.root_source must be 'kindyn' or 'mhr'; "
            f"got {ms['root_source']!r}")
    if ms["root_convention"] not in _ROOT_CONVENTIONS:
        raise ValueError(
            f"motion_supervision.root_convention must be one of "
            f"{sorted(_ROOT_CONVENTIONS)}; got {ms['root_convention']!r}")
    joint_names = ms["joint_names"]
    if joint_names is not None and (
            not isinstance(joint_names, list) or not joint_names
            or any(name not in _MOTION_JOINT_NAMES for name in joint_names)
            or len(set(joint_names)) != len(joint_names)):
        raise ValueError(
            "motion_supervision.joint_names must be null (all seven) or a "
            f"duplicate-free subset of {list(_MOTION_JOINT_NAMES)}; got "
            f"{joint_names!r}")
    angular = ms["angular"]
    if not isinstance(angular, bool):
        raise ValueError(
            f"motion_supervision.angular must be a boolean; got {angular!r}")
    # The angular pair is the SE3-log twist's own components — it only exists
    # for the root slot and only under the twist convention.
    if angular and ms["root_convention"] not in ("twist", "gravity_view"):
        raise ValueError(
            "motion_supervision.angular requires root_convention 'twist' or "
            "'gravity_view' (the angular pair is the body rate under both)")
    if angular and joint_names != ["pelvis"]:
        raise ValueError(
            "motion_supervision.angular requires joint_names=['pelvis'] "
            "(angular targets exist for the root slot only); got "
            f"{joint_names!r}")
    # The loader expresses only the ROOT slot in the gravity-view frame, and the
    # loss de-rotates every slot with one matrix, so a mixed slot list would
    # report the limbs in the wrong frame (a silent diagnostics bug).
    if ms["root_convention"] == "gravity_view" and (
            joint_names or list(_MOTION_JOINT_NAMES)) != ["pelvis"]:
        raise ValueError(
            "motion_supervision.root_convention='gravity_view' requires "
            "joint_names=['pelvis'] (the limb slots stay in root axes)")
    smooth = ms["target_smooth_sec"]
    if (isinstance(smooth, bool) or not isinstance(smooth, (int, float))
            or not math.isfinite(float(smooth)) or float(smooth) < 0):
        raise ValueError(
            "motion_supervision.target_smooth_sec must be a finite number >= 0 "
            f"(seconds; 0 = raw kindyn derivatives); got {smooth!r}")
    for key in ("vel", "acc", "ang_vel", "ang_acc", "outlier_acc_ms2"):
        value = float(ms["loss"][key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"motion_supervision.loss.{key} must be finite and >= 0")
    delta = float(ms["loss"]["huber_delta"])
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("motion_supervision.loss.huber_delta must be finite and positive")

    if not ms["enabled"]:
        return
    if not motion_head["enabled"]:
        raise ValueError(
            "motion_supervision.enabled requires model.motion_head.enabled=true "
            "(the supervised loss reads the predicted motion)")
    # Token k, joint_names[k] and standardize row k are matched purely by
    # POSITION — nothing downstream can detect a reordered anchor list, so the
    # arity is the only mechanical check available. The name list is duplicated
    # here (as `_MOTION_JOINT_NAMES`) so config validation stays torch/loader-free.
    names = list(joint_names or _MOTION_JOINT_NAMES)
    if len(motion_kp) != len(names):
        raise ValueError(
            "motion_supervision.enabled requires exactly "
            f"{len(names)} model.motion_head.motion_keypoint_indices (one per "
            f"motion_supervision.joint_names entry: {', '.join(names)}); "
            f"got {len(motion_kp)}")
    if not any(entry["name"] == "climbing_corpus" for entry in cfg["data"]["datasets"]):
        raise ValueError(
            "motion_supervision.enabled requires a climbing_corpus dataset in "
            "data.datasets (GT vel/acc come from the corpus kindyn_1.npz)")
    # Targets are native-rate derivatives (dt = 1/fps). Stride 1 shows the model
    # exactly that rate; 'auto' (max(1, round(fps/25)) per scene) shows a
    # fixed PHYSICAL window instead, which only makes sense because
    # `target_smooth_sec` makes the label a band-limited physical quantity rather
    # than a per-sample difference. Any other fixed stride is a silent mismatch.
    stride = cfg["data"]["sequence"]["frame_stride"]
    if stride != "auto" and int(stride) != 1:
        raise ValueError(
            "motion_supervision.enabled requires data.sequence.frame_stride=1 or "
            "'auto' (GT velocity/acceleration are native-rate derivatives)")
    if target_frame == "center" and int(cfg["data"]["sequence"]["frames_per_clip"]) % 2 == 0:
        raise ValueError(
            "motion_supervision.target_frame='center' requires an odd "
            "data.sequence.frames_per_clip")
    # The standardization table is pinned in the config (never a buffer) so the
    # loss stays reproducible from a checkpoint's stored config alone.
    n_groups = 4 if angular else 2
    group_order = "vel, acc, ang_vel, ang_acc" if angular else "vel then acc"
    for key in ("mean", "std"):
        table = ms["standardize"][key]
        if (not isinstance(table, list) or len(table) != len(motion_kp)
                or not all(isinstance(row, list) and len(row) == n_groups
                           for row in table)
                or not all(isinstance(part, list) and len(part) == 3
                           for row in table for part in row)
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                           and math.isfinite(float(v))
                           for row in table for part in row for v in part)):
            raise ValueError(
                f"motion_supervision.standardize.{key} must be a finite "
                f"[{len(motion_kp)}][{n_groups}][3] nested list (one row per "
                f"motion token, {group_order}, xyz)")
    if any(float(v) <= 0 for row in ms["standardize"]["std"] for part in row for v in part):
        raise ValueError("motion_supervision.standardize.std entries must be positive")
    # Label smoothing is implemented for the ROOT trajectory only: the six limb
    # slots keep raw central differences of `joints_world` while their `R^T` uses
    # the SMOOTHED root rotation, so the pair would describe two different
    # bandwidths. Fail rather than ship that silently.
    limbs = [name for name in names if name != "pelvis"]
    if float(ms["target_smooth_sec"]) > 0 and limbs:
        raise ValueError(
            "motion_supervision.target_smooth_sec > 0 is implemented for the "
            f"pelvis slot only; joint_names also selects {limbs}. Limb-target "
            "smoothing is unimplemented (their positions are not smoothed, only "
            "the frame they are rotated into) — set target_smooth_sec: 0.0 or "
            "restrict joint_names to ['pelvis']")






def _validate_contact_consistency(cfg: dict) -> None:
    """Validate ``contact_consistency`` (gated zero-velocity at predicted contacts)."""
    cc = cfg["contact_consistency"]
    vel = float(cc["loss"]["vel"])
    if not math.isfinite(vel) or vel < 0:
        raise ValueError("contact_consistency.loss.vel must be finite and >= 0")
    delta = float(cc["loss"]["huber_delta_ms"])
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError(
            "contact_consistency.loss.huber_delta_ms must be finite and positive")
    if not isinstance(cc["detach_gate"], bool):
        raise ValueError("contact_consistency.detach_gate must be a boolean")
    if not cc["enabled"]:
        return
    if vel == 0.0:
        raise ValueError("contact_consistency.enabled with loss.vel=0 does nothing")
    joint = cfg["contact"]["targets"]["joint"]
    if not (joint["enabled"] and str(joint["joint_set"]) == "kindyn_6"):
        raise ValueError(
            "contact_consistency.enabled requires the kindyn_6 joint contact target "
            "(its six outputs match the six extremity keypoints 1:1)")
    if not _pose_trainable_paths(cfg):
        raise ValueError(
            "contact_consistency.enabled requires a trainable pose path — with the "
            "gate detached the velocity side carries the only gradient")
    if not any(entry["name"] == "climbing_corpus"
               for entry in cfg["data"]["datasets"]):
        raise ValueError(
            "contact_consistency.enabled requires a climbing_corpus dataset "
            "(the world lift needs the per-frame camera extrinsics)")
    if int(cfg["data"]["sequence"]["frames_per_clip"]) < 3:
        raise ValueError(
            "contact_consistency.enabled requires frames_per_clip >= 3 "
            "(central-difference velocity stencil)")


def _validate_force_consistency(cfg: dict) -> None:
    """Validate ``force_consistency`` (linear Newton residual, bw units)."""
    fc = cfg["force_consistency"]
    weight = float(fc["loss"]["residual"])
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("force_consistency.loss.residual must be finite and >= 0")
    delta = float(fc["loss"]["huber_delta_bw"])
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError(
            "force_consistency.loss.huber_delta_bw must be finite and positive")
    kernel = fc["smoothing_kernel"]
    if (not isinstance(kernel, (list, tuple)) or len(kernel) % 2 != 1
            or not all(math.isfinite(float(v)) for v in kernel)):
        raise ValueError(
            "force_consistency.smoothing_kernel must be an odd-length list of "
            "finite floats ([1.0] disables smoothing)")
    ramp = fc["ramp"]
    if int(ramp["start_epoch"]) < 0:
        raise ValueError("force_consistency.ramp.start_epoch must be >= 0")
    if int(ramp["epochs"]) < 1:
        raise ValueError("force_consistency.ramp.epochs must be >= 1")
    if not fc["enabled"]:
        return
    if weight == 0.0:
        raise ValueError("force_consistency.enabled with loss.residual=0 does nothing")
    if not cfg["model"]["force_head"]["enabled"]:
        raise ValueError(
            "force_consistency.enabled requires model.force_head.enabled=true")
    fs = cfg["force_supervision"]
    if not fs["enabled"]:
        raise ValueError(
            "force_consistency.enabled requires force_supervision.enabled=true "
            "(the residual assumes the supervised bw/root force convention)")
    if str(fs["units"]) != "bw" or str(fs["gt_frame"]) != "root":
        raise ValueError(
            "force_consistency.enabled requires force_supervision units 'bw' and "
            "gt_frame 'root' (the residual formula divides by m*g and rotates the "
            "net force with the kindyn root rotation)")
    if not cfg["motion_supervision"]["enabled"]:
        raise ValueError(
            "force_consistency.enabled requires motion_supervision.enabled=true "
            "(the root rotation and its validity come from the motion targets)")
    if int(cfg["data"]["sequence"]["frames_per_clip"]) < 3:
        raise ValueError(
            "force_consistency.enabled requires frames_per_clip >= 3 "
            "(double-difference acceleration stencil)")




def _validate_fit_err_confidence(section: dict, name: str) -> None:
    """Validate the shared ``fit_err_confidence`` / ``fit_err_ref_cm`` pair."""
    if not isinstance(section["fit_err_confidence"], bool):
        raise ValueError(f"{name}.fit_err_confidence must be a boolean")
    ref = float(section["fit_err_ref_cm"])
    if not math.isfinite(ref) or ref <= 0:
        raise ValueError(f"{name}.fit_err_ref_cm must be finite and positive")


def _validate_keypoint_supervision(cfg: dict) -> None:
    """Validate ``keypoint_supervision`` (kindyn keypoint losses) and the
    ``train.finetune_camera_head`` flag whose only objectives live here."""
    ks = cfg["keypoint_supervision"]
    for key in ("kp2d", "kp3d", "kp3d_abs", "kp_vel", "kp_acc",
                "vert", "vert_abs", "cam_rail", "rot_rail"):
        value = float(ks["loss"][key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"keypoint_supervision.loss.{key} must be finite and >= 0")
    for key in ("fingers", "face"):
        value = float(ks["joint_weights"][key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"keypoint_supervision.joint_weights.{key} must be finite and >= 0")
    if float(ks["joint_weights"]["fingers"]) == 0 and float(
            ks["joint_weights"]["face"]) == 0:
        raise ValueError(
            "keypoint_supervision.joint_weights cannot zero both groups and the "
            "body joints are fixed at 1.0 — this config downweights nothing")
    _validate_fit_err_confidence(ks, "keypoint_supervision")
    for key in ("huber_delta_2d", "huber_delta_3d", "huber_delta_vel",
                "huber_delta_acc", "outlier_acc", "cam_rail_margin_m",
                "rot_rail_margin_rad"):
        value = float(ks["loss"][key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"keypoint_supervision.loss.{key} must be finite and positive")
    if ks["enabled"]:
        if not any(float(ks["loss"][k]) > 0
                   for k in ("kp2d", "kp3d", "kp3d_abs", "kp_vel",
                             "kp_acc", "vert", "vert_abs")):
            raise ValueError(
                "keypoint_supervision.enabled requires a positive loss weight")
        if (any(float(ks["loss"][k]) > 0
                for k in ("kp_vel", "kp_acc"))
                and int(cfg["data"]["sequence"]["frames_per_clip"]) < 3):
            raise ValueError(
                "keypoint_supervision kp_vel/kp_acc require "
                "data.sequence.frames_per_clip >= 3 (the stencil reads "
                "frames t-1, t, t+1)")
        if not (_pose_trainable_paths(cfg)
                or cfg["train"]["finetune_camera_head"]):
            raise ValueError(
                "keypoint_supervision.enabled requires a trainable pose or "
                "camera path (a pose path or train.finetune_camera_head) — "
                "otherwise the keypoint losses reach no parameters")
        if not any(entry["name"] == "climbing_corpus"
                   for entry in cfg["data"]["datasets"]):
            raise ValueError(
                "keypoint_supervision.enabled requires a climbing_corpus "
                "dataset in data.datasets (GT keypoints/vertices come from the "
                "corpus mhr_sup_1.npz)")
    if cfg["train"]["finetune_camera_head"] and not (
            ks["enabled"] and float(ks["loss"]["kp2d"]) > 0):
        raise ValueError(
            "train.finetune_camera_head requires keypoint_supervision with a "
            "positive kp2d weight — no other loss constrains the camera head")




def _validate_semantics(cfg: dict) -> None:
    """Cross-key checks that pure key validation cannot express."""
    primary = cfg["contact"]["primary_target"]
    if primary not in _KNOWN_TARGETS:
        raise ValueError(f"contact.primary_target must be one of {sorted(_KNOWN_TARGETS)}; got {primary!r}")

    targets = cfg["contact"]["targets"]
    contact_enabled = any(targets[name]["enabled"] for name in _KNOWN_TARGETS)
    force_head = cfg["model"]["force_head"]
    force_kp = force_head["force_keypoint_indices"]
    motion_head = cfg["model"]["motion_head"]
    if (not contact_enabled and not (force_head["enabled"] and force_kp is not None)
            and not motion_head["enabled"]
            and not cfg["model"]["pose_temporal"]["enabled"]
            and not cfg["train"]["finetune_pose_head"]
            and not cfg["train"]["finetune_camera_head"]):
        # Force-only builds (no contact tokens/head at all) are legal, but only
        # with the force branch on and its own explicit anchors — null anchors
        # inherit from the contact tokens, which do not exist here. Motion-only,
        # pose-temporal-only and fine-tuned-heads-only builds are legal too.
        raise ValueError(
            "the joint contact target is disabled — enable it, "
            "or configure a force-only build (model.force_head.enabled=true with "
            "explicit model.force_head.force_keypoint_indices), a motion-only "
            "build (model.motion_head.enabled=true), a pose-temporal build "
            "(model.pose_temporal.enabled=true), or a fine-tuned-heads build "
            "(train.finetune_pose_head / train.finetune_camera_head)")

    joint_cfg = targets["joint"]
    joint_set = joint_cfg["joint_set"]
    if joint_set not in _KNOWN_JOINT_SETS:
        raise ValueError(
            f"contact.targets.joint.joint_set must be one of {sorted(_KNOWN_JOINT_SETS)}; "
            f"got {joint_set!r}")
    if joint_set != "smplx_body_22" and joint_cfg["supervise_subset"] is not None:
        raise ValueError(
            "contact.targets.joint.supervise_subset must be null for "
            f"joint_set={joint_set!r}")

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
        output_dims = {"joint": len(JOINT_SET_NAMES[joint_set])}
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

    cross_modal = cfg["model"]["cross_modal_temporal"]
    xm_type = cross_modal.get("type", "rope")
    if xm_type != "rope":
        raise ValueError(
            f"model.cross_modal_temporal.type must be 'rope'; got {xm_type!r}")
    _validate_rope_temporal(cross_modal, "model.cross_modal_temporal")
    if cross_modal["enabled"]:
        _validate_modalities(
            cfg, cross_modal, "model.cross_modal_temporal", min_count=2)

    eta = cfg["model"]["extra_token_attention"]
    if eta not in ("causal", "mutual"):
        raise ValueError(
            f"model.extra_token_attention must be 'causal' or 'mutual'; got {eta!r}")

    lr_scale = float(cfg["train"]["pose_head_lr_scale"])
    if not math.isfinite(lr_scale) or lr_scale <= 0:
        raise ValueError("train.pose_head_lr_scale must be finite and positive")

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

    contact_gate = force_head["contact_gate"]
    sharpness = float(contact_gate["sharpness"])
    if not math.isfinite(sharpness) or sharpness <= 0:
        raise ValueError(
            "model.force_head.contact_gate.sharpness must be finite and positive")
    if contact_gate["enabled"]:
        if not force_head["enabled"]:
            raise ValueError(
                "model.force_head.contact_gate.enabled requires "
                "model.force_head.enabled=true (there is no force output to gate)")
        if not (targets["joint"]["enabled"] and joint_set == "kindyn_6"
                and pool_mode == "per_token"):
            raise ValueError(
                "model.force_head.contact_gate.enabled requires the joint contact "
                "target enabled with joint_set='kindyn_6' and "
                "model.contact_head.pool_mode='per_token' (each force group is "
                "gated by its own aligned contact output)")
        # The contact->force gate map is the identity over the six kindyn groups
        # (FORCE_GATE_CONTACT_MAP in the force head; arity duplicated here so
        # config validation stays torch-free).
        if force_kp is None or len(force_kp) != 6:
            raise ValueError(
                "model.force_head.contact_gate.enabled requires explicit "
                "model.force_head.force_keypoint_indices with 6 entries (the "
                "contact->force gate map is the identity over the six kindyn "
                f"groups); got {force_kp!r}")

    _validate_physics(cfg, force_head)
    _validate_force_supervision(cfg, force_head)
    _validate_motion(cfg)
    _validate_contact_consistency(cfg)
    _validate_force_consistency(cfg)
    _validate_keypoint_supervision(cfg)

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
        xm = cfg["model"]["cross_modal_temporal"]
        if xm["enabled"] and "contact" in xm["modalities"]:
            raise ValueError(
                "train.freeze_contact=true is incompatible with 'contact' in "
                "model.cross_modal_temporal.modalities: the block's params do not "
                "carry 'contact' in their names, so they would stay trainable and "
                "move the frozen contact outputs")
        if cfg["model"]["extra_token_attention"] == "mutual":
            raise ValueError(
                "train.freeze_contact=true is incompatible with "
                "model.extra_token_attention='mutual': the frozen contact tokens "
                "would attend the trainable force/motion tokens, so the frozen "
                "contact outputs would move during force training")

    sequence = cfg["data"]["sequence"]
    target_frame = str(sequence["target_frame"])
    if target_frame not in ("all", "center"):
        raise ValueError("data.sequence.target_frame must be 'all' or 'center'")
    frames_per_clip = int(sequence["frames_per_clip"])
    if frames_per_clip <= 0:
        raise ValueError("data.sequence.frames_per_clip must be positive")
    # `auto` = per-scene max(1, round(fps / 25)) (ClimbingCorpusDataset.scene_stride).
    # Checked here rather than at the consumer: several scripts do int(frame_stride).
    stride = sequence["frame_stride"]
    if stride != "auto" and (isinstance(stride, bool)
                             or not isinstance(stride, int) or stride <= 0):
        raise ValueError(
            f"data.sequence.frame_stride must be a positive int or 'auto'; got {stride!r}")
    # The corpus loader resolves `auto` per scene; the motion and pose pipelines
    # ride it. demo.py, render_climbing_video_contacts.py,
    # render_climbing_pose_video.py, predict_reconstruction.py and the viewer all
    # still do int(frame_stride) and would die on it with an opaque ValueError, so
    # the contact/force configs stay int. (evaluate.py handles `auto` since
    # 2026-08-30 — this guard never protected it anyway: a motion/pose config is
    # exactly the kind you then evaluate, and it crashed before scoring a batch.)
    if stride == "auto" and not (cfg["motion_supervision"]["enabled"]
                                 or cfg["pose_supervision"]["enabled"]):
        raise ValueError(
            "data.sequence.frame_stride: auto requires motion_supervision.enabled "
            "or pose_supervision.enabled (the contact/force CLIs read this key as "
            "a plain int)")
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
        or cfg["data"]["datasets"][0]["name"] != "climbing_corpus"
    ):
        raise ValueError(
            "data.eval_split='test' requires a single climbing_corpus dataset "
            "(the manually annotated test split)")


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
    """
    raw = _load_raw(Path(path))
    merged = _deep_merge(DEFAULTS, raw)
    _validate_keys(merged, DEFAULTS)
    _validate_semantics(merged)
    return merged
