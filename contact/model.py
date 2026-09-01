"""Build a SAM-3D-Body model wired for contact-, force- and motion-head training.

We patch the checkpoint's ``model_config.yaml`` *before* constructing
the model so the contact tokens, contact head(s), the optional force
tokens / force head, the optional motion tokens / motion head, and the
mask conditioning the checkpoint shipped with are all created natively by
``SAM3DBody``. Checkpoint weights load with ``strict=False`` — everything
in the upstream model (including the v2 mask conditioning) gets restored,
and the new contact/force/motion modules stay at random init.

After the load we freeze the whole network and unfreeze only the contact,
force **and motion** pipelines: any param whose name contains ``contact``,
``force`` or ``motion`` (contact tokens/head/posemb/feat, the force tokens,
``head_force``, and the motion tokens, ``head_motion``) — plus the shared
post-decoder ``cross_modal_temporal`` block and ``pose_temporal``. Backbone,
decoder, prompt encoder, MHR/camera heads stay frozen. Regime (a) (``train.freeze_contact``)
re-freezes the contact params after the unfreeze so only the force branch
trains. Frozen modules are eval-pinned (:func:`pin_frozen_eval`).

``build_model`` returns ``(model, trainable_names)``.
"""
from __future__ import annotations

import copy
import os
from typing import List, Tuple

import torch
import torch.nn as nn

from sam_3d_body.models.meta_arch import SAM3DBody
from sam_3d_body.utils.checkpoint import load_state_dict
from sam_3d_body.utils.config import get_config

from .targets import TargetSpec


def _patch_model_cfg(model_cfg, cfg: dict, mhr_path: str):
    """Mutate ``model_cfg`` in-place to honour the train config.

    Sizes the contact tokens from ``model.contact_head.contact_keypoint_indices``
    and builds one head per enabled contact target (``CONTACT_HEAD.TARGETS`` maps
    the target name to its output dim: topology verts for ``vertex``, 22 for
    ``joint``).
    """
    from yacs.config import CfgNode

    chead = cfg["model"]["contact_head"]
    kp = chead.get("contact_keypoint_indices")
    if kp is None:
        kp = list(range(21))
    kp = [int(i) for i in kp]

    out_dims = TargetSpec.from_config(cfg).output_dims()   # {name: dim} for enabled targets

    model_cfg.defrost()
    # Rebuild contact tokens from train config; no enabled contact target means
    # a force-only build with no contact tokens/head at all. CONTACT_HEAD is
    # still patched below (self-describing config; the force branch reads the
    # shared GRID_SIZE / GRID_RADIUS from it).
    model_cfg.MODEL.DECODER.DO_CONTACT_TOKENS = bool(out_dims)
    # Decoder mask over the appended token blocks: causal (block-triangular) or
    # mutual (contact/force/motion inter-attend; original tokens still blind).
    model_cfg.MODEL.EXTRA_TOKEN_ATTENTION = str(
        cfg["model"].get("extra_token_attention", "mutual"))
    if "CONTACT_HEAD" not in model_cfg.MODEL:
        model_cfg.MODEL.CONTACT_HEAD = CfgNode()
    ch = model_cfg.MODEL.CONTACT_HEAD
    ch.KEYPOINT_INDICES        = kp
    ch.NUM_CONTACTS            = len(kp)
    ch.NUM_GLOBAL_TOKENS       = int(chead["num_global_tokens"])
    ch.MLP_DEPTH               = int(chead["mlp_depth"])
    ch.MLP_CHANNEL_DIV_FACTOR  = int(chead["mlp_channel_div_factor"])
    ch.POOL_MODE               = str(chead["pool_mode"])
    ch.DROPOUT                 = float(chead["dropout"])
    ch.GRID_SIZE               = int(chead["grid_size"])
    ch.GRID_RADIUS             = float(chead["grid_radius"])
    ch.BLIND_TO_IMAGE          = bool(chead["blind_to_image"])
    ch.TARGETS = CfgNode({name.upper(): int(dim) for name, dim in out_dims.items()})

    # Pose-token temporal module (E2). The DELIBERATE exception to the
    # frozen-pose rule: when enabled, the final MHR output is recomputed from a
    # temporally-mixed pose token (zero-init gates = frozen behavior at init).
    ptcfg = cfg["model"].get("pose_temporal", {}) or {}
    _pt_max_rel = ptcfg.get("max_rel_sec", 2.5)
    model_cfg.MODEL.POSE_TEMPORAL = CfgNode({
        "ENABLED": bool(ptcfg.get("enabled", False)),
        "TYPE": str(ptcfg.get("type", "rope")),
        "TIME_SCALE": float(ptcfg.get("time_scale", 25.0)),
        "MAX_REL_SEC": None if _pt_max_rel is None else float(_pt_max_rel),
        "NUM_LAYERS": int(ptcfg.get("num_layers", 4)),
        "NUM_HEADS": int(ptcfg.get("num_heads", 16)),
        "MLP_RATIO": float(ptcfg.get("mlp_ratio", 2.0)),
        "DROPOUT": float(ptcfg.get("dropout", 0.0)),
    })

    # Force head + tokens (steps 04+). Patched even when disabled so the model
    # config is self-describing; SAM3DBody only builds the force stack when
    # DO_FORCE_TOKENS. The `frame` key is consumed by the physics loss (step 06),
    # not the model, so it is not mirrored here.
    fhcfg = cfg["model"].get("force_head", {}) or {}
    force_kp = fhcfg.get("force_keypoint_indices")
    gate_cfg = fhcfg.get("contact_gate", {}) or {}
    model_cfg.MODEL.DECODER.DO_FORCE_TOKENS = bool(fhcfg.get("enabled", False))
    model_cfg.MODEL.FORCE_HEAD = CfgNode({
        # None = inherit the contact anchors (legacy); else the force tokens
        # get their own MHR70 anchor list (required for force-only builds).
        "KEYPOINT_INDICES":       None if force_kp is None else [int(i) for i in force_kp],
        "MLP_DEPTH":              int(fhcfg.get("mlp_depth", 2)),
        "MLP_CHANNEL_DIV_FACTOR": int(fhcfg.get("mlp_channel_div_factor", 4)),
        "DROPOUT":                float(fhcfg.get("dropout", 0.0)),
        # Contact-gated final force output (sigmoid of the detached extremity
        # contact logits, fixed 4->6 group map — see heads/force_head.py).
        "CONTACT_GATE_ENABLED":   bool(gate_cfg.get("enabled", False)),
        "CONTACT_GATE_SHARPNESS": float(gate_cfg.get("sharpness", 4.0)),
    })

    # Motion head + tokens (motion tokens v2). Patched even when disabled so the
    # model config is self-describing; SAM3DBody only builds the motion stack
    # when DO_MOTION_TOKENS. Anchors are always explicit (no contact inheritance,
    # no global tokens), so the token count is the anchor-list length.
    mhcfg = cfg["model"].get("motion_head", {}) or {}
    motion_kp = mhcfg.get("motion_keypoint_indices") or []
    model_cfg.MODEL.DECODER.DO_MOTION_TOKENS = bool(mhcfg.get("enabled", False))
    # Head width follows the supervision target: 12 when the angular pair is on.
    motion_angular = bool(
        cfg.get("motion_supervision", {}).get("angular", False))
    model_cfg.MODEL.MOTION_HEAD = CfgNode({
        "KEYPOINT_INDICES":       [int(i) for i in motion_kp],
        # False = no per-layer anchored token update; the motion tokens are pure
        # learned queries and the two anchored-update projections are not built.
        "ANCHORED":               bool(mhcfg.get("anchored", True)),
        "MLP_DEPTH":              int(mhcfg.get("mlp_depth", 2)),
        "MLP_CHANNEL_DIV_FACTOR": int(mhcfg.get("mlp_channel_div_factor", 4)),
        "DROPOUT":                float(mhcfg.get("dropout", 0.0)),
        "OUTPUT_DIMS":            12 if motion_angular else 6,
    })

    # Cross-modal temporal module: THE post-decoder mixing brick — ONE temporal
    # transformer (rope or the revived sinusoidal window block) over the
    # concatenation of the listed modality token blocks. Patched even when
    # disabled so the model config is self-describing; SAM3DBody only builds
    # cross_modal_temporal when ENABLED.
    xmcfg = cfg["model"].get("cross_modal_temporal", {}) or {}
    _xm_max_rel = xmcfg.get("max_rel_sec", 2.5)
    _xm_bottleneck = xmcfg.get("bottleneck_dim", 256)
    model_cfg.MODEL.CROSS_MODAL_TEMPORAL = CfgNode({
        "ENABLED": bool(xmcfg.get("enabled", False)),
        "TYPE": str(xmcfg.get("type", "rope")),
        "MODALITIES": [str(m) for m in (xmcfg.get("modalities") or [])],
        "NUM_LAYERS": int(xmcfg.get("num_layers", 4)),
        "NUM_HEADS": int(xmcfg.get("num_heads", 16)),
        "MLP_RATIO": float(xmcfg.get("mlp_ratio", 2.0)),
        "DROPOUT": float(xmcfg.get("dropout", 0.0)),
        "TIME_SCALE": float(xmcfg.get("time_scale", 25.0)),
        "MAX_REL_SEC": None if _xm_max_rel is None else float(_xm_max_rel),
        "BOTTLENECK_DIM": None if _xm_bottleneck is None else int(_xm_bottleneck),
        "POSITION_SCALE": float(xmcfg.get("position_scale", 25.0)),
        "CAUSAL": bool(xmcfg.get("causal", False)),
    })

    # Efficiency flags (Phase 4). Patched even when off so the model config is
    # self-describing; sam3d_body / promptable_decoder read them at their delimited
    # hooks and fall back to old (full-graph) behaviour when the key is absent.
    train_cfg = cfg.get("train", {}) or {}
    model_cfg.MODEL.EFFICIENCY = CfgNode({
        "BACKBONE_NO_GRAD": bool(train_cfg.get("backbone_no_grad", False)),
        "DETACH_INTERM_PREDS": bool(train_cfg.get("detach_interm_preds", False)),
    })

    # Mask conditioning (must be set before model build).
    model_cfg.MODEL.PROMPT_ENCODER.MASK_EMBED_TYPE = cfg["model"].get("mask_embed_type", None)

    # MHR weights path.
    model_cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = mhr_path
    model_cfg.freeze()
    return model_cfg


def _load_checkpoint_weights(model: nn.Module, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    load_state_dict(model, sd, strict=False)


def _trainable_name_filter(name: str) -> bool:
    """Train the contact, force and motion pipelines only: their tokens, heads,
    and the small posemb / feat projection layers that update the tokens between
    decoder layers (all of which contain ``contact``, ``force`` or ``motion`` in
    the dotted name) — plus the shared post-decoder ``cross_modal`` block
    (the all-modality RoPE temporal transformer) and
    ``pose_temporal``, the deliberate exception that is allowed to move the
    frozen pose outputs (E2; ``head_pose`` itself never matches, the substring
    check is on the full module name — its optional fine-tune is a separate
    explicit flag in :func:`build_model`)."""
    lname = name.lower()
    return ("contact" in lname or "force" in lname or "motion" in lname
            or "cross_modal" in lname or "pose_temporal" in lname)


def _subtree_requires_grad(module: nn.Module) -> Tuple[bool, bool]:
    """``(any_trainable, all_trainable)`` over the module's recursive parameters.

    A param-less subtree returns ``(False, False)`` (nothing trainable to toggle).
    """
    reqs = [p.requires_grad for p in module.parameters(recurse=True)]
    if not reqs:
        return False, False
    return any(reqs), all(reqs)


def pin_frozen_eval(model: nn.Module) -> nn.Module:
    """Pin every frozen submodule to eval(); only trainable subtrees follow the
    requested train/eval mode.

    The frozen SAM-3D-Body backbone ships with ``DROP_PATH_RATE 0.1`` (stochastic
    depth) plus dropout, so a global ``model.train()`` would make the frozen
    features the trainable heads read nondeterministic. We force everything to
    eval and keep it there by overriding ``model.train`` so any later
    ``model.train(True)`` re-pins the frozen parts. ``model.eval()`` still works
    since it delegates to ``model.train(False)``.

    The toggled set is derived from ``requires_grad`` at call time (not a name
    list), so it tracks whichever branch is training — contact only, force only
    (regime a: ``freeze_contact``), or both. A subtree whose parameters are *all*
    trainable follows ``mode`` in full, including its param-less ``nn.Dropout``
    children (a rule keyed on direct trainable params would silently disable them);
    a fully-frozen subtree (e.g. a contact head frozen in regime a) stays eval;
    a mixed container is descended into. For a contact-only build this reproduces
    the previous ``"contact"``-name behaviour exactly.
    """
    def _propagate(module: nn.Module, mode: bool) -> None:
        any_trainable, all_trainable = _subtree_requires_grad(module)
        if not any_trainable:
            return                                      # fully frozen → stays eval
        if all_trainable:
            nn.Module.train(module, mode)               # trainable subtree → mode
            return
        for child in module.children():                 # mixed → descend
            _propagate(child, mode)

    def _apply(mode: bool = True) -> nn.Module:
        nn.Module.train(model, False)                   # every module → eval
        _propagate(model, bool(mode))
        return model

    model.train = _apply
    _apply(False)
    return model


def build_model(cfg: dict, device: str = "cuda") -> Tuple[nn.Module, List[str]]:
    """Construct a SAM-3D-Body model ready for contact training.

    Returns ``(model, trainable_param_names)``. The names are produced
    in iteration order over ``model.named_parameters()`` so the caller
    can build an optimiser param group with ``filter(lambda p: p.requires_grad, ...)``
    and a checkpoint dump with the same name list.
    """
    mcfg = cfg["model"]
    ckpt_path = mcfg["checkpoint_path"]
    mhr_path  = mcfg["mhr_model_path"]

    # 1) Load model_config that ships with the checkpoint, then patch.
    model_cfg_path = os.path.join(os.path.dirname(ckpt_path), "model_config.yaml")
    if not os.path.exists(model_cfg_path):
        model_cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), "model_config.yaml",
        )
    model_cfg = get_config(model_cfg_path)
    _patch_model_cfg(model_cfg, cfg, mhr_path)

    # 2) Build + load weights for the parts that exist in the checkpoint.
    model = SAM3DBody(model_cfg)
    _load_checkpoint_weights(model, ckpt_path)

    # 3) Freeze everything; unfreeze the contact + force + motion pipelines.
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if _trainable_name_filter(name):
            p.requires_grad = True

    # Optional pose/camera-head fine-tune (split-head): the ORIGINAL heads stay
    # frozen and keep producing every in-decoder intermediate prediction (whose
    # keypoint-token refresh feeds back into the frozen decoder — training the
    # shared head would perturb the frozen model layer by layer), while a COPY
    # of the projection FFN — initialized identical, so init behavior is
    # exactly the frozen model — is applied to the FINAL pose token only (the
    # meta-arch's final-readout recompute). Deliberate exception to the
    # frozen-pose rule; train.py enforces a pose/keypoint objective and gives
    # these params their own lr-scaled optimizer group.
    if cfg.get("train", {}).get("finetune_pose_head", False):
        model.head_pose_ft_proj = copy.deepcopy(model.head_pose.proj)
        for p in model.head_pose_ft_proj.parameters():
            p.requires_grad = True
    if cfg.get("train", {}).get("finetune_camera_head", False):
        model.head_camera_ft_proj = copy.deepcopy(model.head_camera.proj)
        for p in model.head_camera_ft_proj.parameters():
            p.requires_grad = True

    # Regime (a): warm-start from a contact checkpoint and train the force branch
    # only. Re-freeze every contact param after the normal unfreeze (force params,
    # named force_*, do not match "contact"). Config validation requires an init
    # contact checkpoint and force_head.enabled when this is set.
    if cfg.get("train", {}).get("freeze_contact", False):
        for name, p in model.named_parameters():
            if "contact" in name.lower():
                p.requires_grad = False

    model.to(device)

    # Keep frozen backbone/decoder/heads in eval permanently (kills stochastic
    # depth + dropout nondeterminism); only contact modules toggle train/eval.
    pin_frozen_eval(model)

    # backbone_no_grad is only sound if the backbone is fully frozen.
    if model_cfg.MODEL.EFFICIENCY.BACKBONE_NO_GRAD:
        trainable_backbone = [
            n for n, p in model.named_parameters()
            if p.requires_grad and n.startswith("backbone")
        ]
        assert not trainable_backbone, (
            f"train.backbone_no_grad is set but the backbone has trainable params: "
            f"{trainable_backbone}")

    # Optional torch.compile of the frozen backbone (no trainable params there,
    # so checkpoints — trainable-only weights — never see the _orig_mod prefix).
    if cfg["train"]["compile_backbone"]:
        model.backbone = torch.compile(model.backbone)

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_train:,} / {n_total:,}  ({100 * n_train / n_total:.2f}%)")
    return model, trainable_names
