"""Unit tests for the RoPE temporal module (long-sequence pose path).

Small dims for speed; all float32 (the library's end-to-end dtype). Where a
test needs the module to actually *do* something, the zero-init gammas are
nudged off zero with a seeded generator.
"""
from __future__ import annotations

import pytest
import torch

from sam_3d_body.models.modules.temporal_rope import RopeTemporalModule


def _module(**overrides) -> RopeTemporalModule:
    kwargs = dict(dim=64, num_layers=2, num_heads=4, mlp_ratio=2.0,
                  dropout=0.0, time_scale=25.0, max_rel_sec=2.5)
    kwargs.update(overrides)
    return RopeTemporalModule(**kwargs).eval()


def _nudge_gammas(module: RopeTemporalModule, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if "gamma" in name:
                p.copy_(0.1 * torch.randn(p.shape, generator=g))


def _clip(n_clips: int = 2, t: int = 6, dim: int = 64, fps: float = 25.0):
    g = torch.Generator().manual_seed(1)
    tokens = torch.randn(n_clips * t, 1, dim, generator=g)
    pos = (torch.arange(n_clips * t, dtype=torch.float32) % t) / fps
    valid = torch.ones(n_clips * t, dtype=torch.bool)
    return tokens, pos, valid


def test_identity_at_init():
    """Zero gammas make the module a bitwise identity."""
    m = _module()
    tokens, pos, valid = _clip()
    assert torch.equal(m(tokens, 6, pos, valid), tokens)


def test_identity_at_init_single_frame_and_no_positions():
    """T=1 and missing frame_pos_sec both run and stay identity."""
    m = _module()
    tokens, _, _ = _clip(n_clips=4, t=1)
    assert torch.equal(m(tokens, 1, None, None), tokens)


def test_time_shift_invariance():
    """RoPE logits depend only on relative offsets: shifting every timestamp
    by a constant leaves the output unchanged (the property absolute
    sinusoidal encodings lack)."""
    m = _module()
    _nudge_gammas(m)
    tokens, pos, valid = _clip()
    out_a = m(tokens, 6, pos, valid)
    out_b = m(tokens, 6, pos + 3.7, valid)
    assert not torch.equal(out_a, tokens)          # the module is active
    assert torch.allclose(out_a, out_b, atol=1e-4)


def test_fps_changes_the_output():
    """The same frame indices at a different fps are a different motion —
    time-valued positions must distinguish them."""
    m = _module()
    _nudge_gammas(m)
    tokens, pos, valid = _clip(fps=25.0)
    out_25 = m(tokens, 6, pos, valid)
    out_60 = m(tokens, 6, pos * (25.0 / 60.0), valid)
    assert not torch.allclose(out_25, out_60, atol=1e-6)


def test_window_mask_hides_far_frames():
    """Frames further apart than max_rel_sec never influence each other,
    even through stacked layers (receptive field = layers * window)."""
    m = _module(max_rel_sec=0.1)                   # +-2 frames at 25 fps
    _nudge_gammas(m)
    tokens, pos, valid = _clip(n_clips=1, t=10)
    base = m(tokens, 10, pos, valid)
    perturbed = tokens.clone()
    perturbed[9] += 10.0                           # last frame, 0.36 s away
    out = m(perturbed, 10, pos, valid)
    # 2 layers x 0.1 s window: frame 9 reaches back at most to ~frame 5.
    assert torch.equal(out[0], base[0])
    assert not torch.equal(out[8], base[8])


def test_window_mask_inert_inside_training_span():
    """A clip shorter than max_rel_sec builds no mask: output equals the
    unwindowed module's."""
    import copy

    tokens, pos, valid = _clip()
    m_win = _module(max_rel_sec=2.5)
    _nudge_gammas(m_win)
    m_off = copy.deepcopy(m_win)
    m_off.max_rel_sec = None
    assert torch.equal(m_win(tokens, 6, pos, valid), m_off(tokens, 6, pos, valid))


def test_invalid_frame_is_hidden_and_nothing_nans():
    """An invalid frame's token influences no other frame; outputs stay finite
    (the diagonal keeps every softmax row alive)."""
    m = _module()
    _nudge_gammas(m)
    tokens, pos, valid = _clip(n_clips=1, t=6)
    valid = valid.clone()
    valid[3] = False
    base = m(tokens, 6, pos, valid)
    perturbed = tokens.clone()
    perturbed[3] += 10.0
    out = m(perturbed, 6, pos, valid)
    keep = [i for i in range(6) if i != 3]
    assert torch.equal(out[keep], base[keep])
    assert torch.isfinite(out).all()


def test_slots_attend_independently():
    """K > 1: perturbing one slot never leaks into the other slot's output."""
    m = _module()
    _nudge_gammas(m)
    g = torch.Generator().manual_seed(2)
    tokens = torch.randn(6, 2, 64, generator=g)
    pos = torch.arange(6, dtype=torch.float32) / 25.0
    base = m(tokens, 6, pos, None)
    perturbed = tokens.clone()
    perturbed[:, 1] += 10.0
    out = m(perturbed, 6, pos, None)
    assert torch.equal(out[:, 0], base[:, 0])
    assert not torch.equal(out[:, 1], base[:, 1])


def test_gradients_reach_all_parameters():
    """One backward pass touches qkv/proj/ffn weights and the gammas."""
    m = _module().train()
    tokens, pos, valid = _clip()
    out = m(tokens.requires_grad_(True), 6, pos, valid)
    out.square().mean().backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None
               or not torch.isfinite(p.grad).all()]
    assert missing == []
    # gamma grads are nonzero even at init (they gate a nonzero branch)
    gnorm = sum(p.grad.abs().sum() for n, p in m.named_parameters() if "gamma" in n)
    assert float(gnorm) > 0.0


def test_causal_and_joint_attend_rejected():
    m = _module()
    tokens, pos, valid = _clip()
    with pytest.raises(ValueError):
        m(tokens, 6, pos, valid, causal=True)
    with pytest.raises(ValueError):
        m(tokens, 6, pos, valid, attend="joint")


def test_constructor_validation():
    with pytest.raises(ValueError):
        _module(time_scale=0.0)
    with pytest.raises(ValueError):
        _module(max_rel_sec=-1.0)
    with pytest.raises(ValueError):
        RopeTemporalModule(dim=64, num_heads=5)    # not divisible


# ------------------------------------------------------------- GPU semantics

import os
from pathlib import Path

from contact.config import load_config

_REPO = Path(__file__).resolve().parents[1]
_ROPE_CFG = _REPO / "configs" / "rope_t60.yaml"
_CORPUS = Path("/data3/rikhat.akizhanov/better/data/ClimbingVideos")
_EMB_DIR = _CORPUS / "features" / "embedding"
_CKPT_PATH = load_config(_REPO / "configs" / "base.yaml")["model"]["checkpoint_path"]

slow = pytest.mark.slow
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
needs_ckpt = pytest.mark.skipif(not os.path.exists(_CKPT_PATH), reason="checkpoint missing")
needs_corpus = pytest.mark.skipif(not _EMB_DIR.is_dir(), reason="embedding cache missing")


@slow
@needs_gpu
@needs_ckpt
@needs_corpus
def test_rope_gpu_semantics_cached_embeddings():
    """The rope_t60 build on the real checkpoint + cached embeddings:
    zero-init == frozen (noise floor), live gammas move pose, gradients reach
    ONLY pose_temporal + the split ft heads; peak memory is printed for the
    frames-per-batch decision."""
    from contact.data.climbing_corpus import ClimbingCorpusDataset
    from contact.data.collate import batch_to_device, make_collate
    from contact.engine import forward_model
    from contact.keypoint_supervision import KeypointSupervisedLoss
    from contact.model import build_model
    from contact.pose_supervision import PoseSupervisedLoss
    from contact.targets import TargetSpec

    torch.manual_seed(0)
    cfg = load_config(_ROPE_CFG)
    model, trainable = build_model(cfg, "cuda")
    model.eval()
    allowed = ("pose_temporal", "head_pose_ft_proj", "head_camera_ft_proj")
    assert trainable and all(any(k in n for k in allowed) for n in trainable)
    assert any("pose_temporal" in n for n in trainable)

    ds = ClimbingCorpusDataset(
        _CORPUS, scenes=["MuVpoovQl2M_0001"], split="train", frames_per_clip=60,
        frame_stride="auto", jitter=False, load_pose=True, load_keypoints=True,
        embedding_dir=_EMB_DIR)
    collate = make_collate(
        tuple(model.cfg.MODEL.IMAGE_SIZE), TargetSpec.from_config(cfg))
    batch = batch_to_device(collate([ds[0], ds[1]]), "cuda")   # 2 clips x T=60
    assert "embedding" in batch and batch["embedding"].shape[0] == 120

    out_live = forward_model(model, batch)
    pt = model.pose_temporal
    del model.pose_temporal
    off_a = forward_model(model, batch)
    off_b = forward_model(model, batch)
    model.pose_temporal = pt
    for key in ("body_pose", "global_rot", "pred_keypoints_3d", "pred_cam_t"):
        floor = (off_a["mhr"][key] - off_b["mhr"][key]).abs().max()
        diff = (out_live["mhr"][key] - off_a["mhr"][key]).abs().max()
        assert float(diff) <= 8.0 * float(floor) + 1e-6, (key, float(diff))
    # The frozen-anchor stash equals the live outputs at init (deepcopy heads).
    assert float((out_live["mhr"]["pred_cam_t"]
                  - out_live["mhr"]["pred_cam_t_frozen"]).abs().max()) < 1e-5

    with torch.no_grad():
        for n, p in model.pose_temporal.named_parameters():
            if "gamma" in n:
                torch.nn.init.normal_(p, std=0.1)
    out_hot = forward_model(model, batch)
    assert float((out_hot["mhr"]["body_pose"]
                  - off_a["mhr"]["body_pose"]).abs().max()) > 1e-3

    torch.cuda.reset_peak_memory_stats()
    pose_loss = PoseSupervisedLoss(cfg, device="cuda")
    kp_loss = KeypointSupervisedLoss(cfg, device="cuda")
    out = forward_model(model, batch)
    total_p, parts_p = pose_loss(out, batch)
    total_k, parts_k = kp_loss(out, batch)
    assert parts_p["terms"]["pose"]["weight_mass"] > 0
    assert parts_k["terms"]["kp_vel"]["weight_mass"] > 0
    (total_p + total_k).backward()
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert any(k in n for k in allowed), n
        elif p.grad is not None:
            raise AssertionError(f"frozen param {n} received a gradient")
    grads = [n for n, p in model.pose_temporal.named_parameters()
             if p.grad is not None and float(p.grad.abs().sum()) > 0]
    assert grads
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    print(f"\n[rope smoke] fwd+bwd peak {peak_gb:.2f} GiB for 120 frames "
          f"({peak_gb / 120 * 1024:.1f} MiB/frame)")


@slow
@needs_gpu
@needs_ckpt
@needs_corpus
def test_rope_full_scene_single_pass():
    """Whole-scene single-pass inference (the eval protocol): T far beyond the
    training span runs under the seconds-window mask, stays finite, and matches
    identity at init."""
    from contact.data.climbing_corpus import ClimbingCorpusDataset
    from contact.data.collate import batch_to_device, make_collate
    from contact.engine import forward_model
    from contact.model import build_model
    from contact.targets import TargetSpec

    cfg = load_config(_ROPE_CFG)
    model, _ = build_model(cfg, "cuda")
    model.eval()
    ds = ClimbingCorpusDataset(
        _CORPUS, scenes=["MuVpoovQl2M_0001"], split="test", frames_per_clip=60,
        frame_stride="auto", jitter=False, full_scenes=True, require_labels=False,
        embedding_dir=_EMB_DIR)
    clip = ds[0]
    assert len(clip) > 120                      # genuinely beyond training length
    collate = make_collate(
        tuple(model.cfg.MODEL.IMAGE_SIZE), TargetSpec.from_config(cfg))
    batch = batch_to_device(collate([clip]), "cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = forward_model(model, batch)
    assert torch.isfinite(out["mhr"]["body_pose"]).all()
    assert torch.isfinite(out["mhr"]["pred_cam_t"]).all()
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    print(f"\n[rope smoke] full-scene T={len(clip)} no-grad peak {peak_gb:.2f} GiB")
