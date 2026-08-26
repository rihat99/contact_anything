"""ContactTemporalModule unit tests (CPU): no-op init, masks, PE, shapes."""
from __future__ import annotations

import pytest
import torch

from sam_3d_body.models.modules.temporal import (
    ContactTemporalModule,
    frame_visibility,
    sinusoidal_time_encoding,
)

_DIM = 16
_HEADS = 4
_K = 3


def _module(**kw):
    defaults = dict(dim=_DIM, num_layers=2, num_heads=_HEADS, mlp_ratio=2.0,
                    attend="joint", causal=False, dropout=0.0)
    defaults.update(kw)
    m = ContactTemporalModule(**defaults)
    m.eval()
    return m


# ---------------------------------------------------------------- zero-gamma no-op

@pytest.mark.parametrize("attend", ["joint", "per_token"])
@pytest.mark.parametrize("seq_len,n_clips", [(1, 3), (4, 2)])
def test_zero_gamma_is_exact_identity(attend, seq_len, n_clips):
    m = _module(attend=attend)
    b_flat = n_clips * seq_len
    tokens = torch.randn(b_flat, _K, _DIM)
    pos = torch.arange(b_flat, dtype=torch.float32) * 0.1
    valid = torch.ones(b_flat, dtype=torch.bool)
    out = m(tokens, seq_len, pos, valid)
    assert out.shape == tokens.shape
    assert torch.equal(out, tokens), "zero-gamma module must be an exact identity"


def test_zero_gamma_identity_with_invalid_and_causal():
    # Masking must not poison the identity (no NaN from all-masked rows).
    m = _module(attend="joint", causal=True)
    tokens = torch.randn(6, _K, _DIM)
    pos = torch.arange(6, dtype=torch.float32)
    valid = torch.tensor([True, False, True, True, True, False])  # 2 clips, T=3
    out = m(tokens, 3, pos, valid)
    assert torch.equal(out, tokens)


@pytest.mark.parametrize("attend", ["joint", "per_token"])
def test_bottleneck_adapter_is_exact_identity_and_gets_live_gate_gradients(attend):
    m = _module(attend=attend, bottleneck_dim=8, num_heads=2)
    tokens = torch.randn(10, _K, _DIM, requires_grad=True)
    out = m(tokens, 5, torch.arange(10, dtype=torch.float32), None)
    assert torch.equal(out, tokens)
    out.square().sum().backward()
    gammas = [p for n, p in m.named_parameters() if "gamma_" in n]
    assert gammas and all(g.grad is not None for g in gammas)
    assert hasattr(m, "token_in_proj") and hasattr(m, "token_out_proj")


# ---------------------------------------------------------------- visibility matrix

def test_frame_visibility_causal_with_invalid_frame():
    # T=3, frame 1 invalid, causal: hand-written expected allowed matrix.
    valid = torch.tensor([True, False, True])
    allowed = frame_visibility(3, valid, causal=True)
    expected = torch.tensor([
        [True,  False, False],   # q0 sees only its own frame
        [True,  True,  False],   # q1 (invalid) still sees itself + valid past
        [True,  False, True],    # q2 sees valid past (0) + self, NOT invalid 1
    ])
    assert torch.equal(allowed, expected)


def test_frame_visibility_noncausal_all_valid_is_full():
    allowed = frame_visibility(4, torch.ones(4, dtype=torch.bool), causal=False)
    assert torch.equal(allowed, torch.ones(4, 4, dtype=torch.bool))


def test_frame_visibility_causal_all_valid_is_lower_triangular():
    allowed = frame_visibility(4, torch.ones(4, dtype=torch.bool), causal=True)
    assert torch.equal(allowed, torch.tril(torch.ones(4, 4, dtype=torch.bool)))


def test_causal_per_token_outputs_do_not_depend_on_future_frames():
    """Changing future token features cannot change an earlier prediction."""
    torch.manual_seed(7)
    module = _module(attend="per_token", causal=True)
    with torch.no_grad():
        for block in module.blocks:
            block.gamma_attn.fill_(0.2)
            block.gamma_ffn.fill_(0.2)

    tokens = torch.randn(5, _K, _DIM)
    changed = tokens.clone()
    changed[3:] += 10.0
    positions = torch.arange(5, dtype=torch.float32)

    original_out = module(tokens, 5, positions)
    changed_out = module(changed, 5, positions)

    assert torch.equal(original_out[:3], changed_out[:3])
    assert not torch.equal(original_out[3:], changed_out[3:])


def test_frame_visibility_diagonal_never_fully_masked():
    # Every row must keep at least the diagonal so softmax never sees an all -inf row.
    for causal in (True, False):
        allowed = frame_visibility(3, torch.zeros(3, dtype=torch.bool), causal=causal)
        assert torch.equal(torch.diagonal(allowed), torch.ones(3, dtype=torch.bool))
        assert bool(allowed.any(dim=1).all())


def test_joint_token_mask_kron_expansion():
    # attend='joint' expands the [T,T] frame mask to [T*K, T*K] token blocks.
    m = _module(attend="joint", causal=True, num_heads=1)
    valid = torch.tensor([True, False, True])           # 1 clip, T=3
    blocked = m._attn_mask(3, valid, n_clips=1, num_heads=1, per_slot=2,
                           causal=True, device=torch.device("cpu"))
    allowed_tok = ~blocked[0]                            # [T*K, T*K], K=2
    frame_allowed = frame_visibility(3, valid, causal=True)
    expected = frame_allowed.repeat_interleave(2, 0).repeat_interleave(2, 1)
    assert torch.equal(allowed_tok, expected)


def test_attn_mask_none_when_noncausal_all_valid():
    m = _module(attend="joint", causal=False)
    blocked = m._attn_mask(4, torch.ones(8, dtype=torch.bool), n_clips=2,
                           num_heads=_HEADS, per_slot=_K, causal=False,
                           device=torch.device("cpu"))
    assert blocked is None


# ---------------------------------------------------------------- positional encoding

def test_pe_zeros_at_t1():
    m = _module()
    pe = m._pos_emb(torch.zeros(3), seq_len=1, n_clips=3, dim=_DIM,
                    device=torch.device("cpu"), dtype=torch.float32)
    assert pe.shape == (3, 1, _DIM)
    assert torch.equal(pe, torch.zeros_like(pe))


def test_pe_nonzero_for_multiframe():
    m = _module()
    pos = torch.tensor([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])  # 2 clips, T=3
    pe = m._pos_emb(pos, seq_len=3, n_clips=2, dim=_DIM,
                    device=torch.device("cpu"), dtype=torch.float32)
    assert pe.shape == (2, 3, _DIM)
    assert not torch.equal(pe, torch.zeros_like(pe))
    # identical positions -> identical encodings (both clips share 0,0.5,1.0)
    assert torch.equal(pe[0], pe[1])


def test_position_scale_separates_adjacent_30fps_frames():
    pos = torch.tensor([0.0, 1.0 / 30.0])
    weak = _module(position_scale=1.0)._pos_emb(
        pos, seq_len=2, n_clips=1, dim=_DIM,
        device=torch.device("cpu"), dtype=torch.float32,
    )[0]
    strong = _module(position_scale=30.0)._pos_emb(
        pos, seq_len=2, n_clips=1, dim=_DIM,
        device=torch.device("cpu"), dtype=torch.float32,
    )[0]
    weak_similarity = torch.nn.functional.cosine_similarity(weak[0], weak[1], dim=0)
    strong_similarity = torch.nn.functional.cosine_similarity(strong[0], strong[1], dim=0)
    assert strong_similarity < weak_similarity - 1e-3


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_position_scale_must_be_finite_and_positive(scale):
    with pytest.raises(ValueError, match="position_scale must be finite and positive"):
        _module(position_scale=scale)


def test_sinusoidal_encoding_shape_and_even_dim():
    enc = sinusoidal_time_encoding(torch.tensor([0.0, 1.0, 2.0]), _DIM)
    assert enc.shape == (3, _DIM)
    # pos=0 -> sin block 0, cos block 1 (standard sinusoidal absolute encoding)
    assert torch.allclose(enc[0, : _DIM // 2], torch.zeros(_DIM // 2))
    assert torch.allclose(enc[0, _DIM // 2:], torch.ones(_DIM // 2))
    with pytest.raises(ValueError):
        sinusoidal_time_encoding(torch.tensor([0.0]), 15)


# ---------------------------------------------------------------- shapes / asserts

@pytest.mark.parametrize("attend", ["joint", "per_token"])
def test_output_shape_matches_input(attend):
    m = _module(attend=attend)
    tokens = torch.randn(8, _K, _DIM)  # 2 clips, T=4
    out = m(tokens, 4, torch.arange(8, dtype=torch.float32), torch.ones(8, dtype=torch.bool))
    assert out.shape == tokens.shape


def test_divisibility_assert():
    m = _module()
    tokens = torch.randn(7, _K, _DIM)   # 7 not divisible by 4
    with pytest.raises(AssertionError):
        m(tokens, 4, None, None)


def test_nonzero_gamma_changes_output_but_stays_finite():
    m = _module(attend="joint", causal=True)
    with torch.no_grad():
        for blk in m.blocks:
            blk.gamma_attn.normal_()
            blk.gamma_ffn.normal_()
    tokens = torch.randn(6, _K, _DIM)
    valid = torch.tensor([True, False, True, False, False, False])  # 2 clips, T=3
    out = m(tokens, 3, torch.arange(6, dtype=torch.float32), valid)
    assert torch.isfinite(out).all()
    assert not torch.equal(out, tokens)


def test_per_token_attention_cannot_mix_limb_slots():
    torch.manual_seed(0)
    m = _module(attend="per_token")
    with torch.no_grad():
        for block in m.blocks:
            block.gamma_attn.fill_(0.1)
            block.gamma_ffn.fill_(0.1)
    tokens = torch.randn(5, _K, _DIM)
    changed = tokens.clone()
    changed[:, 0] += 10.0
    original_out = m(tokens, 5, torch.arange(5, dtype=torch.float32), None)
    changed_out = m(changed, 5, torch.arange(5, dtype=torch.float32), None)
    assert torch.equal(original_out[:, 1:], changed_out[:, 1:])


def test_noncausal_center_output_backpropagates_to_every_context_frame():
    """A center-only loss still trains from all five temporal input rows."""
    torch.manual_seed(11)
    module = _module(attend="per_token", causal=False, num_layers=1)
    with torch.no_grad():
        module.blocks[0].gamma_attn.fill_(0.2)
        module.blocks[0].gamma_ffn.fill_(0.2)

    tokens = torch.randn(5, _K, _DIM, requires_grad=True)
    output = module(tokens, 5, torch.arange(5, dtype=torch.float32))
    output[2].sum().backward()

    gradient_per_frame = tokens.grad.abs().sum(dim=(1, 2))
    assert bool((gradient_per_frame > 0).all())


def test_gate_params_present_and_zero_init():
    # Under the model the attr is `contact_temporal`, so every dotted param name
    # gains that prefix; the freeze/eval filters key off "contact". Here we just
    # confirm the zero-init gates exist.
    m = _module()
    gammas = [p for n, p in m.named_parameters() if "gamma_" in n]
    assert gammas, "no gate parameters found"
    assert all(bool((g == 0).all()) for g in gammas)


# ---------------------------------------------------------------- force temporal path

# `force_temporal` is a second ContactTemporalModule instance (post_decoder,
# attend='per_token' by default, D11). The two tests below pin the module-level
# contract the force path relies on, reusing the shared `_module` factory.

def test_force_temporal_zero_gamma_is_exact_identity():
    m = _module(attend="per_token")
    tokens = torch.randn(8, _K, _DIM)                       # 2 clips, T=4
    pos = torch.arange(8, dtype=torch.float32) * 0.1
    valid = torch.ones(8, dtype=torch.bool)
    out = m(tokens, 4, pos, valid)
    assert torch.equal(out, tokens), "zero-gamma force temporal must be an exact identity"


def test_force_temporal_per_token_cannot_mix_limb_slots():
    torch.manual_seed(0)
    m = _module(attend="per_token")
    with torch.no_grad():
        for block in m.blocks:
            block.gamma_attn.fill_(0.1)
            block.gamma_ffn.fill_(0.1)
    tokens = torch.randn(5, _K, _DIM)
    changed = tokens.clone()
    changed[:, 0] += 10.0
    out = m(tokens, 5, torch.arange(5, dtype=torch.float32), None)
    out_changed = m(changed, 5, torch.arange(5, dtype=torch.float32), None)
    assert torch.equal(out[:, 1:], out_changed[:, 1:])


# ---------------------------------------------------------------- centered window

def _live_module(window_frames=None, attend="per_token", num_layers=1, causal=False):
    """A window module with deterministic non-zero gates (off the identity)."""
    torch.manual_seed(3)
    m = _module(attend=attend, num_layers=num_layers, causal=causal,
                window_frames=window_frames)
    with torch.no_grad():
        for block in m.blocks:
            block.gamma_attn.normal_(std=0.3)
            block.gamma_ffn.normal_(std=0.3)
    return m


def test_window_matches_native_subclip_and_passes_through_outside():
    # T=7, window=5 (lo=1): frames 1..5 must equal running the SAME module on the
    # extracted central-5 subclip (positions re-zeroed); frames 0 and 6 pass through.
    m = _live_module(window_frames=5)
    n_clips, T = 2, 7
    tokens = torch.randn(n_clips * T, _K, _DIM)
    pos = torch.cat([torch.arange(T, dtype=torch.float32) for _ in range(n_clips)])
    valid = torch.ones(n_clips * T, dtype=torch.bool)

    out = m(tokens, T, pos, valid).view(n_clips, T, _K, _DIM)

    # Native T=5 reference on the central window with re-zeroed positions.
    clips = tokens.view(n_clips, T, _K, _DIM)
    sub = clips[:, 1:6].reshape(n_clips * 5, _K, _DIM)
    sub_pos = (pos.view(n_clips, T)[:, 1:6] - pos.view(n_clips, T)[:, 1:2]).reshape(-1)
    ref = m(sub, 5, sub_pos, valid.view(n_clips, T)[:, 1:6].reshape(-1))
    ref = ref.view(n_clips, 5, _K, _DIM)

    assert torch.equal(out[:, 1:6], ref)                       # window == native T=5
    tok = tokens.view(n_clips, T, _K, _DIM)
    assert torch.equal(out[:, 0], tok[:, 0])                   # pre-window pass-through
    assert torch.equal(out[:, 6], tok[:, 6])                   # post-window pass-through


def test_window_inactive_when_seq_len_leq_window():
    # T=5 with window=5 must byte-match the no-window module (same weights).
    torch.manual_seed(3)
    m_win = _module(attend="per_token", num_layers=1, window_frames=5)
    torch.manual_seed(3)
    m_no = _module(attend="per_token", num_layers=1)
    with torch.no_grad():
        for a, b in zip(m_win.blocks, m_no.blocks):
            a.gamma_attn.normal_(std=0.3); b.gamma_attn.copy_(a.gamma_attn)
            a.gamma_ffn.normal_(std=0.3); b.gamma_ffn.copy_(a.gamma_ffn)
    tokens = torch.randn(10, _K, _DIM)                          # 2 clips, T=5
    pos = torch.cat([torch.arange(5, dtype=torch.float32)] * 2)
    assert torch.equal(m_win(tokens, 5, pos, None), m_no(tokens, 5, pos, None))


def test_window_t1_unchanged():
    # T=1 stills: window is inactive (1 <= 5) -> exact identity at zero gammas.
    m = _module(attend="per_token", window_frames=5)
    tokens = torch.randn(3, _K, _DIM)
    out = m(tokens, 1, torch.zeros(3), torch.ones(3, dtype=torch.bool))
    assert torch.equal(out, tokens)


def test_window_joint_attend_path():
    m = _live_module(window_frames=5, attend="joint")
    n_clips, T = 2, 7
    tokens = torch.randn(n_clips * T, _K, _DIM)
    pos = torch.cat([torch.arange(T, dtype=torch.float32) for _ in range(n_clips)])
    out = m(tokens, T, pos, None).view(n_clips, T, _K, _DIM)
    tok = tokens.view(n_clips, T, _K, _DIM)
    assert torch.equal(out[:, 0], tok[:, 0]) and torch.equal(out[:, 6], tok[:, 6])
    assert not torch.equal(out[:, 1:6], tok[:, 1:6])           # window frames moved


def test_window_ignores_out_of_window_invalidity():
    # An invalid frame OUTSIDE the window is never attended, so marking it invalid
    # cannot change the result vs. all-valid.
    m = _live_module(window_frames=5)
    n_clips, T = 1, 7
    tokens = torch.randn(n_clips * T, _K, _DIM)
    pos = torch.arange(T, dtype=torch.float32)
    all_valid = torch.ones(T, dtype=torch.bool)
    marked = all_valid.clone()
    marked[0] = False                                          # frame 0 is outside [1,6)
    assert torch.equal(m(tokens, T, pos, all_valid), m(tokens, T, pos, marked))


def test_window_requires_centered_offset():
    # T=8, window=5 -> (8-5)=3 is odd -> the window cannot be centered.
    m = _module(attend="per_token", window_frames=5)
    tokens = torch.randn(8, _K, _DIM)
    with pytest.raises(AssertionError, match="exactly centered"):
        m(tokens, 8, torch.arange(8, dtype=torch.float32), None)


@pytest.mark.parametrize("bad", [4, 2, 0, -1])
def test_window_frames_must_be_odd_ge_3(bad):
    with pytest.raises(ValueError, match="window_frames must be an odd int >= 3"):
        _module(attend="per_token", window_frames=bad)


# ---------------------------------------------------------------- config plumbing

def _sig_cfg(window_frames):
    """Minimal config dict for :func:`contact.checkpoint._arch_signature`."""
    return {
        "model": {
            "contact_head": {"contact_keypoint_indices": [62, 41, 13, 14],
                             "num_global_tokens": 0, "pool_mode": "per_token"},
            "temporal": {"enabled": True,
                         "attend": "per_token", "causal": False,
                         "bottleneck_dim": 256, "num_layers": 1, "num_heads": 4,
                         "mlp_ratio": 2.0, "position_scale": 30.0,
                         "window_frames": window_frames},
        },
        "contact": {"topology": "smplx",
                    "targets": {"joint": {"enabled": True,
                                          "joint_set": "extremities_4"}}},
    }


def test_window_frames_excluded_from_arch_signature():
    # An inference attention-window choice must not change the arch identity, so a
    # windowed run loads its non-windowed checkpoint.
    from contact.checkpoint import _arch_signature
    assert _arch_signature(_sig_cfg(None)) == _arch_signature(_sig_cfg(5))


def test_signature_still_emits_placement_for_stored_checkpoints():
    # The placement key left the config schema (post_decoder is the only
    # placement), but stored checkpoint signatures contain it and the
    # comparison is exact equality — the default must keep being emitted.
    from contact.checkpoint import _arch_signature
    assert _arch_signature(_sig_cfg(None))["temporal"]["placement"] == "post_decoder"


def test_bridge_carries_window_frames_to_yacs():
    # The yaml->yacs bridge must surface window_frames as MODEL.TEMPORAL.WINDOW_FRAMES
    # (kept as-is, incl. None) for the meta_arch construction hook.
    from pathlib import Path

    from yacs.config import CfgNode as CN

    from contact.config import load_config
    from contact.model import _patch_model_cfg

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")
    cfg["model"]["temporal"]["enabled"] = True
    cfg["model"]["temporal"]["window_frames"] = 5

    mc = CN()
    mc.MODEL = CN()
    mc.MODEL.DECODER = CN()
    mc.MODEL.PROMPT_ENCODER = CN()
    mc.MODEL.MHR_HEAD = CN()
    _patch_model_cfg(mc, cfg, mhr_path="unused.pt")
    assert mc.MODEL.TEMPORAL.WINDOW_FRAMES == 5
