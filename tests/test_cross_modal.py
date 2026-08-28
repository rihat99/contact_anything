"""Cross-modal temporal + frame attention + pose-head fine-tune wiring.

Fast tests cover config validation, the yaml->yacs bridge, the arch signature
and the freeze filter; ``-m slow`` covers the GPU semantics with the real
checkpoint (zero-init no-op, modality isolation, trainable sets).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from contact import checkpoint as ckpt_io
from contact.config import _pose_trainable_paths, load_config
from contact.model import _trainable_name_filter

REPO = Path(__file__).resolve().parents[1]
JOINT_CFG = REPO / "configs" / "old" / "climbing_videos_joint.yaml"
_CKPT = load_config(REPO / "configs" / "base.yaml")["model"]["checkpoint_path"]


def _write(tmp_path, text):
    p = tmp_path / "run.yaml"
    p.write_text(text)
    return p


def _joint_cfg_text(extra: str) -> str:
    """A joint-target corpus config (contact enabled) plus ``extra`` overrides."""
    return f"""
base: {JOINT_CFG}
{extra}
"""


# ---------------------------------------------------------------- config defaults

def test_defaults_disabled():
    cfg = load_config(REPO / "configs" / "base.yaml")
    assert cfg["model"]["cross_modal_temporal"]["enabled"] is False
    assert cfg["model"]["frame_attn"]["enabled"] is False
    assert cfg["train"]["finetune_pose_head"] is False
    assert cfg["train"]["pose_head_lr_scale"] == pytest.approx(0.1)


# ---------------------------------------------------------------- validation

def test_cross_modal_needs_two_modalities(tmp_path):
    with pytest.raises(ValueError, match="cross_modal_temporal.modalities"):
        load_config(_write(tmp_path, _joint_cfg_text(
            "model: {cross_modal_temporal: {enabled: true, modalities: [contact]}}")))


def test_cross_modal_rejects_unknown_and_duplicate_modalities(tmp_path):
    for bad in ("[contact, hands]", "[contact, contact]"):
        with pytest.raises(ValueError, match="cross_modal_temporal.modalities"):
            load_config(_write(tmp_path, _joint_cfg_text(
                f"model: {{cross_modal_temporal: {{enabled: true, modalities: {bad}}}}}")))


def test_cross_modal_rejects_disabled_branch(tmp_path):
    # JOINT_CFG has no force head -> 'force' has no token block.
    with pytest.raises(ValueError, match="no token block"):
        load_config(_write(tmp_path, _joint_cfg_text(
            "model: {cross_modal_temporal: {enabled: true, "
            "modalities: [contact, force]}}")))


def test_cross_modal_accepts_pose_contact(tmp_path):
    cfg = load_config(_write(tmp_path, _joint_cfg_text(
        "model: {cross_modal_temporal: {enabled: true, modalities: [pose, contact]}}")))
    assert cfg["model"]["cross_modal_temporal"]["modalities"] == ["pose", "contact"]


def test_frame_attn_needs_frame_context(tmp_path):
    # A pose-only build has no other tokens in the frame to attend.
    with pytest.raises(ValueError, match="at least one of the contact"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
model:
  pose_temporal: {enabled: true}
  frame_attn: {enabled: true, modalities: [pose]}
contact:
  targets:
    vertex: {enabled: false}
    joint: {enabled: false}
"""))
    # With a contact branch present the same listing is legal: keys/values
    # always span every enabled modality's tokens of the frame.
    cfg = load_config(_write(tmp_path, _joint_cfg_text(
        "model: {frame_attn: {enabled: true, modalities: [pose]}}")))
    assert cfg["model"]["frame_attn"]["modalities"] == ["pose"]


def test_extra_token_attention_validation(tmp_path):
    assert load_config(REPO / "configs" / "base.yaml")["model"][
        "extra_token_attention"] == "mutual"
    with pytest.raises(ValueError, match="extra_token_attention"):
        load_config(_write(tmp_path, _joint_cfg_text(
            "model: {extra_token_attention: sideways}")))
    # JOINT_CFG has only the contact branch -> mutual is vacuous (identical
    # mask to causal) but, as the repo default, must be legal for every build.
    cfg = load_config(_write(tmp_path, _joint_cfg_text(
        "model: {extra_token_attention: mutual}")))
    assert cfg["model"]["extra_token_attention"] == "mutual"
    cfg = load_config(_write(tmp_path, _joint_cfg_text(
        "model:\n"
        "  extra_token_attention: mutual\n"
        "  force_head: {enabled: true, force_keypoint_indices: [62, 41, 13, 14],"
        " frame: root}")))
    assert cfg["model"]["extra_token_attention"] == "mutual"


def test_mutual_mask_incompatible_with_freeze_contact(tmp_path):
    with pytest.raises(ValueError, match="extra_token_attention='mutual'"):
        load_config(_write(tmp_path, _joint_cfg_text(
            "model:\n"
            "  init_contact_checkpoint: some.pth\n"
            "  extra_token_attention: mutual\n"
            "  force_head: {enabled: true, force_keypoint_indices: [62, 41, 13, 14],"
            " frame: root}\n"
            "train: {freeze_contact: true}")))


def test_pose_head_lr_scale_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="pose_head_lr_scale"):
        load_config(_write(tmp_path, _joint_cfg_text(
            "train: {pose_head_lr_scale: 0.0}")))


def test_pose_supervision_accepts_finetune_pose_head(tmp_path):
    # No pose_temporal needed anymore: the fine-tuned head is a trainable path.
    cfg = load_config(_write(tmp_path, _joint_cfg_text(
        "train: {finetune_pose_head: true}\n"
        "pose_supervision: {enabled: true}")))
    assert _pose_trainable_paths(cfg) == ["train.finetune_pose_head"]


def test_pose_trainable_paths_lists_modality_bricks(tmp_path):
    cfg = load_config(_write(tmp_path, _joint_cfg_text(
        "model: {cross_modal_temporal: {enabled: true, modalities: [pose, contact]}}\n"
        "pose_supervision: {enabled: true}")))
    assert _pose_trainable_paths(cfg) == ["model.cross_modal_temporal (pose modality)"]


# ---------------------------------------------------------------- freeze filter

def test_freeze_filter_matches_new_bricks_and_not_head_pose():
    assert _trainable_name_filter("cross_modal_temporal.blocks.0.gamma_attn")
    assert _trainable_name_filter("frame_attn.contact.blocks.0.gamma_attn")
    assert _trainable_name_filter("frame_attn.pose.token_in_proj.weight")
    assert not _trainable_name_filter("head_pose.proj.layers.0.weight")
    assert not _trainable_name_filter("head_pose.hand_pose_comps")


# ---------------------------------------------------------------- yacs bridge

def test_bridge_carries_new_sections_to_yacs(tmp_path):
    from yacs.config import CfgNode as CN

    from contact.model import _patch_model_cfg

    cfg = load_config(_write(tmp_path, _joint_cfg_text(
        "model:\n"
        "  extra_token_attention: mutual\n"
        "  force_head: {enabled: true, force_keypoint_indices: [62, 41, 13, 14],"
        " frame: root}\n"
        "  cross_modal_temporal: {enabled: true, modalities: [pose, contact]}\n"
        "  frame_attn: {enabled: true, modalities: [contact]}")))
    mc = CN()
    mc.MODEL = CN()
    mc.MODEL.DECODER = CN()
    mc.MODEL.PROMPT_ENCODER = CN()
    mc.MODEL.MHR_HEAD = CN()
    _patch_model_cfg(mc, cfg, mhr_path="unused.pt")
    assert mc.MODEL.CROSS_MODAL_TEMPORAL.ENABLED is True
    assert mc.MODEL.CROSS_MODAL_TEMPORAL.MODALITIES == ["pose", "contact"]
    assert mc.MODEL.FRAME_ATTN.ENABLED is True
    assert mc.MODEL.FRAME_ATTN.MODALITIES == ["contact"]
    assert mc.MODEL.EXTRA_TOKEN_ATTENTION == "mutual"


# ---------------------------------------------------------------- arch signature

def test_signature_captures_bricks_only_when_enabled(tmp_path):
    base = load_config(_write(tmp_path, _joint_cfg_text("")))
    sig = ckpt_io._arch_signature(base)
    assert "cross_modal_temporal" not in sig
    assert "frame_attn" not in sig
    assert "pose_head_finetune" not in sig
    assert "extra_token_attention" not in sig      # causal is never recorded

    cfg = load_config(_write(tmp_path, _joint_cfg_text(
        "model:\n"
        "  cross_modal_temporal: {enabled: true, modalities: [pose, contact]}\n"
        "  frame_attn: {enabled: true, modalities: [contact]}\n"
        "train: {finetune_pose_head: true}\n"
        "pose_supervision: {enabled: true}")))
    sig2 = ckpt_io._arch_signature(cfg)
    assert sig2["cross_modal_temporal"]["modalities"] == ["contact", "pose"]  # sorted
    # The attend_all key is a pinned constant now (the config knob was removed;
    # stored signatures compare by exact equality).
    assert sig2["frame_attn"]["attend_all"] is True
    assert sig2["pose_head_finetune"] == {"enabled": True, "split": True}
    assert "camera_head_finetune" not in sig2
    assert sig != sig2

    cfg3 = load_config(_write(tmp_path, _joint_cfg_text(
        "train: {finetune_camera_head: true}\n"
        "keypoint_supervision: {enabled: true}")))
    sig3 = ckpt_io._arch_signature(cfg3)
    assert sig3["camera_head_finetune"] == {"enabled": True}
    assert "pose_head_finetune" not in sig3


def test_signature_captures_mutual_mask(tmp_path):
    cfg = load_config(_write(tmp_path, _joint_cfg_text(
        "model:\n"
        "  extra_token_attention: mutual\n"
        "  force_head: {enabled: true, force_keypoint_indices: [62, 41, 13, 14],"
        " frame: root}")))
    assert ckpt_io._arch_signature(cfg)["extra_token_attention"] == "mutual"


# ---------------------------------------------------------------- GPU semantics

pytest_slow = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint missing"),
]

_NOISE_MARGIN = 8.0
_NOISE_FLOOR_EPS = 1e-6


def _gpu_build(extra: str, tmp_path):
    from contact.model import build_model

    torch.manual_seed(0)
    cfg = load_config(_write(tmp_path, _joint_cfg_text(extra)))
    model, trainable = build_model(cfg, "cuda")
    model.eval()
    return model, trainable, cfg


def _gpu_batch(cfg, model, n=4):
    import numpy as np

    from contact.data.collate import batch_to_device, make_collate
    from contact.targets import NUM_BODY_22, TargetSpec

    rng = np.random.RandomState(1234)
    frames = []
    for t in range(n):
        gt = torch.zeros(NUM_BODY_22)
        gt[t % NUM_BODY_22] = 1.0
        frames.append({
            "image": (rng.rand(200, 160, 3) * 255).astype(np.uint8),
            "mask": (np.ones((200, 160), np.uint8) * 255),
            "bbox": np.array([10.0, 10.0, 150.0, 190.0], np.float32),
            "cam_int": (np.eye(3, dtype=np.float32) * 500.0),
            "joint_contact": gt,
            "joint_mask": torch.ones(NUM_BODY_22),
            "joint_supervised": torch.ones(NUM_BODY_22),
            "joint_confidence": torch.ones(NUM_BODY_22),
            "frame_pos_sec": t * 0.1,
            "frame_valid": True,
        })
    collate = make_collate(tuple(model.cfg.MODEL.IMAGE_SIZE), TargetSpec.from_config(cfg))
    return batch_to_device(collate([frames]), "cuda")   # one T=n clip


def _heat_gammas(module, seed=13):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if "gamma" in name:
                p.copy_(torch.randn(p.shape, generator=gen, device="cuda"))


def _mhr_floats(out):
    return {k: v.detach().float().clone() for k, v in out["mhr"].items()
            if torch.is_tensor(v) and v.is_floating_point()}


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint missing")
def test_cross_modal_gpu_semantics(tmp_path):
    """Zero-init == dropped module (noise floor); hot gammas move the contact
    logits but leave the frozen MHR outputs at the floor ('pose' not listed)."""
    from contact.engine import forward_model

    model, trainable, cfg = _gpu_build(
        "model:\n"
        "  cross_modal_temporal: {enabled: true, modalities: [pose, contact]}",
        tmp_path)
    assert any("cross_modal_temporal" in n for n in trainable)
    batch = _gpu_batch(cfg, model)

    out_live = forward_model(model, batch)
    xm = model.cross_modal_temporal
    del model.cross_modal_temporal
    off_a = forward_model(model, batch)
    off_b = forward_model(model, batch)
    model.cross_modal_temporal = xm

    floor_c = (off_a["contact"]["joint_logits"] - off_b["contact"]["joint_logits"]).abs().max()
    diff_c = (out_live["contact"]["joint_logits"] - off_a["contact"]["joint_logits"]).abs().max()
    assert float(diff_c) <= _NOISE_MARGIN * float(floor_c) + _NOISE_FLOOR_EPS

    _heat_gammas(model.cross_modal_temporal)
    out_hot = forward_model(model, batch)
    moved = (out_hot["contact"]["joint_logits"] - off_a["contact"]["joint_logits"]).abs().max()
    assert float(moved) > 100.0 * (float(floor_c) + _NOISE_FLOOR_EPS)

    # 'pose' IS a listed modality here — hot gammas must move the MHR outputs
    # (that is the point of the pose modality)...
    mhr_a, mhr_hot = _mhr_floats(off_a), _mhr_floats(out_hot)
    assert float((mhr_hot["body_pose"] - mhr_a["body_pose"]).abs().max()) > 1e-3


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint missing")
def test_frame_attn_gpu_semantics(tmp_path):
    """Zero-init == dropped module (noise floor); hot gammas move contact while
    the frozen MHR outputs stay at the floor (contact is the only written
    modality; the pose token is read-only context)."""
    from contact.engine import forward_model

    model, trainable, cfg = _gpu_build(
        "model:\n"
        "  frame_attn: {enabled: true, modalities: [contact]}",
        tmp_path)
    assert any("frame_attn.contact" in n for n in trainable)
    batch = _gpu_batch(cfg, model)

    out_live = forward_model(model, batch)
    fa = model.frame_attn
    del model.frame_attn
    off_a = forward_model(model, batch)
    off_b = forward_model(model, batch)
    model.frame_attn = fa

    floor_c = (off_a["contact"]["joint_logits"] - off_b["contact"]["joint_logits"]).abs().max()
    diff_c = (out_live["contact"]["joint_logits"] - off_a["contact"]["joint_logits"]).abs().max()
    assert float(diff_c) <= _NOISE_MARGIN * float(floor_c) + _NOISE_FLOOR_EPS

    _heat_gammas(model.frame_attn["contact"])
    out_hot = forward_model(model, batch)
    moved = (out_hot["contact"]["joint_logits"] - off_a["contact"]["joint_logits"]).abs().max()
    assert float(moved) > 100.0 * (float(floor_c) + _NOISE_FLOOR_EPS)

    mhr_a, mhr_b, mhr_hot = _mhr_floats(off_a), _mhr_floats(off_b), _mhr_floats(out_hot)
    for key in ("body_pose", "global_rot", "pred_keypoints_3d", "pred_vertices"):
        floor = (mhr_a[key] - mhr_b[key]).abs().max()
        diff = (mhr_hot[key] - mhr_a[key]).abs().max()
        assert float(diff) <= _NOISE_MARGIN * float(floor) + _NOISE_FLOOR_EPS, key


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint missing")
def test_finetune_pose_head_trainable_set(tmp_path):
    model, trainable, _ = _gpu_build(
        "train: {finetune_pose_head: true}\n"
        "pose_supervision: {enabled: true}",
        tmp_path)
    head = [n for n in trainable if n.startswith("head_pose")]
    # Split-head: the trainable params are the COPY's, never the original's.
    assert head and all(n.startswith("head_pose_ft_proj.") for n in head)
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert any(n.startswith("head_pose.proj.") for n in frozen)
    assert "head_pose.hand_pose_comps" in frozen
    assert any(n.startswith("head_pose_hand.") for n in frozen)
    # The copy starts exactly equal to the original (frozen init behavior).
    for (na, pa), (nb, pb) in zip(
            sorted(model.head_pose.proj.named_parameters()),
            sorted(model.head_pose_ft_proj.named_parameters())):
        assert na == nb and torch.equal(pa, pb)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint missing")
def test_mutual_mask_gpu_semantics(tmp_path):
    """extra_token_attention='mutual' opens contact→force attention inside the
    frozen decoder (contact logits move vs the same-weights causal build well
    above the repeat floor) while the base tokens stay blind to the appended
    blocks (every MHR output at the floor)."""
    from contact.engine import forward_model

    force_line = ("model:\n"
                  "  force_head: {enabled: true,"
                  " force_keypoint_indices: [62, 41, 13, 14], frame: root}\n")

    model_c, _, cfg_c = _gpu_build(force_line, tmp_path)
    batch = _gpu_batch(cfg_c, model_c)
    causal_a = forward_model(model_c, batch)
    causal_b = forward_model(model_c, batch)
    contact_ref = causal_a["contact"]["joint_logits"].detach().float().clone()
    floor_c = float((contact_ref
                     - causal_b["contact"]["joint_logits"].float()).abs().max())
    mhr_ref = _mhr_floats(causal_a)
    mhr_floor = {k: float((v - _mhr_floats(causal_b)[k]).abs().max())
                 for k, v in mhr_ref.items()}
    weights_ref = model_c.contact_embedding.weight.detach().clone()
    del model_c, causal_a, causal_b
    torch.cuda.empty_cache()

    model_m, _, cfg_m = _gpu_build(
        force_line + "  extra_token_attention: mutual", tmp_path)
    # Same seed, same build order: the two models share every weight — the
    # attention mask is the ONLY difference between the two forwards.
    assert torch.equal(model_m.contact_embedding.weight.detach(), weights_ref)
    mutual = forward_model(model_m, _gpu_batch(cfg_m, model_m))

    moved = float((mutual["contact"]["joint_logits"].float() - contact_ref).abs().max())
    assert moved > 100.0 * (floor_c + _NOISE_FLOOR_EPS), (
        "mutual mask did not open contact→force attention")

    mhr_mut = _mhr_floats(mutual)
    for key in ("body_pose", "global_rot", "pred_keypoints_3d", "pred_vertices"):
        diff = float((mhr_mut[key] - mhr_ref[key]).abs().max())
        limit = _NOISE_MARGIN * mhr_floor[key] + _NOISE_FLOOR_EPS
        assert diff <= limit, f"MHR {key!r} moved {diff:.2e} > {limit:.2e}"


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint missing")
def test_head_split_isolation_gpu(tmp_path):
    """Perturbing the fine-tuned head COPIES moves ONLY the final readout.

    The in-decoder trajectory — contact logits and the stashed frozen anchors
    (produced by the original heads) — stays at the CUDA repeat floor, proving
    the frozen decoder never sees the fine-tuned weights (the old shared-head
    scheme failed exactly this: per-layer interm predictions feed the keypoint
    token refresh). At init (copies == originals) the final outputs match a
    no-finetune build.
    """
    from contact.engine import forward_model

    model, _, cfg = _gpu_build(
        "train: {finetune_pose_head: true, finetune_camera_head: true}\n"
        "pose_supervision: {enabled: true}\n"
        "keypoint_supervision: {enabled: true}",
        tmp_path)
    batch = _gpu_batch(cfg, model)
    with torch.no_grad():
        ref_a = forward_model(model, batch)
        ref_b = forward_model(model, batch)
    contact_ref = ref_a["contact"]["joint_logits"].detach().float().clone()
    floor = float((contact_ref
                   - ref_b["contact"]["joint_logits"].float()).abs().max())
    mhr_ref = _mhr_floats(ref_a)
    mhr_floor = {k: float((v - _mhr_floats(ref_b)[k]).abs().max())
                 for k, v in mhr_ref.items()}

    # Init equality: the ft build's final readout (recompute with identical
    # copy weights) matches a no-finetune build on the same batch.
    model0, _, cfg0 = _gpu_build("", tmp_path)
    with torch.no_grad():
        out0 = forward_model(model0, _gpu_batch(cfg0, model0))
    for key in ("pred_cam_t", "mhr_model_params"):
        delta = float((ref_a["mhr"][key].float()
                       - out0["mhr"][key].float()).abs().max())
        assert delta <= max(10 * mhr_floor[key], 1e-4), (key, delta)

    with torch.no_grad():
        for module in (model.head_pose_ft_proj, model.head_camera_ft_proj):
            for param in module.parameters():
                param.add_(torch.randn_like(param) * 0.01)
        out_p = forward_model(model, batch)

    # Decoder-side unchanged: contact logits + frozen anchors at the floor.
    d_contact = float((out_p["contact"]["joint_logits"].float()
                       - contact_ref).abs().max())
    assert d_contact <= max(2 * floor, 1e-6), (d_contact, floor)
    for key in ("pred_cam_t_frozen", "global_rot_frozen"):
        delta = float((out_p["mhr"][key].float()
                       - ref_a["mhr"][key].float()).abs().max())
        assert delta <= max(2 * mhr_floor.get(key, 0.0), 1e-6), (key, delta)

    # Final readout moved well above the floor (both heads).
    for key in ("pred_cam_t", "mhr_model_params"):
        delta = float((out_p["mhr"][key].float()
                       - ref_a["mhr"][key].float()).abs().max())
        assert delta > max(100 * mhr_floor[key], 1e-3), (key, delta)
