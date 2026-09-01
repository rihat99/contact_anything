"""Unit tests for the all-modality RoPE temporal block (CPU, float32).

Covers the guarantees the block inherits from the retired bricks (zero-gate
identity at init, frame-validity masking, seconds window) plus the two it adds:
RoPE relative-time semantics across a multi-token-per-frame sequence, and the
learned slot embedding that carries token identity.
"""
from __future__ import annotations

import pytest
import torch

from sam_3d_body.models.modules.cross_modal_rope import CrossModalRopeModule

_DIM = 64
_SLOTS = 3


def _module(**overrides) -> CrossModalRopeModule:
    kwargs = dict(dim=_DIM, num_slots=_SLOTS, num_layers=2, num_heads=4,
                  mlp_ratio=2.0, dropout=0.0, time_scale=25.0, max_rel_sec=2.5)
    kwargs.update(overrides)
    return CrossModalRopeModule(**kwargs).eval()


def _nudge(module: CrossModalRopeModule, seed: int = 0) -> None:
    """Move the zero-init gammas (and the slot embedding) off their init so the
    block actually mixes."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "gamma" in name:
                param.copy_(0.1 * torch.randn(param.shape, generator=gen))
        module.slot_embed.copy_(
            torch.randn(module.slot_embed.shape, generator=gen))


def _clip(n_clips: int = 2, t: int = 6, slots: int = _SLOTS, fps: float = 25.0):
    gen = torch.Generator().manual_seed(1)
    tokens = torch.randn(n_clips * t, slots, _DIM, generator=gen)
    pos = (torch.arange(n_clips * t, dtype=torch.float32) % t) / fps
    valid = torch.ones(n_clips * t, dtype=torch.bool)
    return tokens, pos, valid


# ---------------------------------------------------------------- zero-gate init

def test_identity_at_init():
    """Zero gammas make the block a bitwise identity — enabling it changes no
    output, which is what keeps the frozen pose/MHR path intact."""
    module = _module()
    tokens, pos, valid = _clip()
    assert torch.equal(module(tokens, 6, pos, valid), tokens)


def test_identity_at_init_survives_a_nonzero_slot_embedding():
    """The slot embedding lives INSIDE the gated branch, so a hot embedding on
    its own still cannot move the output."""
    module = _module()
    with torch.no_grad():
        module.slot_embed.copy_(torch.randn(module.slot_embed.shape))
    tokens, pos, valid = _clip()
    assert torch.equal(module(tokens, 6, pos, valid), tokens)


def test_identity_at_init_single_frame_and_no_positions():
    """T=1 (single images) and a missing frame_pos_sec both run and stay
    identity; at T=1 the block degenerates to within-frame attention."""
    module = _module()
    tokens, _, _ = _clip(n_clips=4, t=1)
    assert torch.equal(module(tokens, 1, None, None), tokens)


def test_single_frame_mixes_within_the_frame():
    """T=1 is not a passthrough once the gates are hot: the slots of the one
    frame still attend each other (this subsumes the retired frame_attn)."""
    module = _module()
    _nudge(module)
    tokens, _, _ = _clip(n_clips=4, t=1)
    out = module(tokens, 1, None, None)
    assert not torch.equal(out, tokens)
    # Frames stay independent — perturbing row 0 leaves row 1 untouched.
    perturbed = tokens.clone()
    perturbed[0] += 10.0
    out_p = module(perturbed, 1, None, None)
    assert torch.equal(out_p[1:], out[1:])


# ---------------------------------------------------------------- RoPE semantics

def test_time_shift_invariance():
    """RoPE logits depend only on relative offsets: shifting every timestamp by
    a constant leaves the output unchanged (absolute encodings cannot do this)."""
    module = _module()
    _nudge(module)
    tokens, pos, valid = _clip()
    out_a = module(tokens, 6, pos, valid)
    out_b = module(tokens, 6, pos + 3.7, valid)
    assert not torch.equal(out_a, tokens)          # the block is active
    assert torch.allclose(out_a, out_b, atol=1e-4)


def test_scaling_dt_changes_the_output():
    """The same frame indices at a different fps are a different motion —
    time-valued positions must distinguish them."""
    module = _module()
    _nudge(module)
    tokens, pos, valid = _clip(fps=25.0)
    out_25 = module(tokens, 6, pos, valid)
    out_60 = module(tokens, 6, pos * (25.0 / 60.0), valid)
    assert not torch.allclose(out_25, out_60, atol=1e-6)


def test_all_slots_of_a_frame_share_one_rope_position():
    """Within-frame pairs must see dt = 0. Two clips whose frames differ only by
    a per-frame time shift are the same problem, so a permutation of slots INSIDE
    one frame is the only thing that can change a same-frame interaction — the
    positions themselves carry no slot index."""
    module = _module(num_layers=1)
    _nudge(module)
    tokens, pos, valid = _clip(n_clips=1, t=4)
    # Collapse the clip to one instant: every token now sits at dt = 0 from
    # every other, so the result must be independent of the (constant) stamp.
    flat_a = module(tokens, 4, torch.zeros(4), valid)
    flat_b = module(tokens, 4, torch.full((4,), 9.5), valid)
    assert torch.allclose(flat_a, flat_b, atol=1e-5)
    # ...and differs from the genuinely time-separated clip.
    assert not torch.allclose(flat_a, module(tokens, 4, pos, valid), atol=1e-5)


def test_window_mask_hides_far_frames():
    """Frames further apart than max_rel_sec never influence each other."""
    module = _module(max_rel_sec=0.1)              # +-2 frames at 25 fps
    _nudge(module)
    tokens, pos, valid = _clip(n_clips=1, t=10)
    base = module(tokens, 10, pos, valid)
    perturbed = tokens.clone()
    perturbed[9] += 10.0                           # last frame, 0.36 s away
    out = module(perturbed, 10, pos, valid)
    assert torch.equal(out[0], base[0])            # unreachable through 2 layers
    assert not torch.equal(out[8], base[8])


def test_window_mask_inert_inside_training_span():
    """A clip shorter than max_rel_sec builds no mask at all."""
    import copy

    tokens, pos, valid = _clip()
    windowed = _module(max_rel_sec=2.5)
    _nudge(windowed)
    unwindowed = copy.deepcopy(windowed)
    unwindowed.max_rel_sec = None
    assert torch.equal(windowed(tokens, 6, pos, valid),
                       unwindowed(tokens, 6, pos, valid))


def test_invalid_frame_is_hidden_and_nothing_nans():
    """An invalid frame's tokens influence no other frame; outputs stay finite
    (its own frame stays visible, so no softmax row is ever empty)."""
    module = _module()
    _nudge(module)
    tokens, pos, valid = _clip(n_clips=1, t=6)
    valid = valid.clone()
    valid[3] = False
    base = module(tokens, 6, pos, valid)
    perturbed = tokens.clone()
    perturbed[3] += 10.0
    out = module(perturbed, 6, pos, valid)
    keep = [i for i in range(6) if i != 3]
    assert torch.equal(out[keep], base[keep])
    assert torch.isfinite(out).all()


def test_clips_are_independent():
    """Flattened clips must not leak into each other."""
    module = _module()
    _nudge(module)
    tokens, pos, valid = _clip(n_clips=2, t=6)
    base = module(tokens, 6, pos, valid)
    perturbed = tokens.clone()
    perturbed[6:] += 10.0                          # entire second clip
    out = module(perturbed, 6, pos, valid)
    assert torch.equal(out[:6], base[:6])


# ---------------------------------------------------------------- slot identity

def test_slots_are_not_interchangeable():
    """The learned slot embedding breaks permutation equivariance: swapping two
    slots' token values does NOT just swap their outputs (RoPE alone carries no
    token identity, so without the embedding it would)."""
    module = _module(num_slots=2)
    _nudge(module)
    gen = torch.Generator().manual_seed(7)
    tokens = torch.randn(6, 2, _DIM, generator=gen)
    pos = torch.arange(6, dtype=torch.float32) / 25.0
    base = module(tokens, 6, pos, None)
    swapped = module(tokens.flip(1), 6, pos, None).flip(1)
    assert not torch.allclose(base, swapped, atol=1e-4)


def test_slots_mix_with_each_other():
    """Cross-modal mixing: perturbing one slot reaches the others (the retired
    per-modality blocks could not do this)."""
    module = _module()
    _nudge(module)
    tokens, pos, valid = _clip(n_clips=1, t=6)
    base = module(tokens, 6, pos, valid)
    perturbed = tokens.clone()
    perturbed[:, 1] += 10.0
    out = module(perturbed, 6, pos, valid)
    assert not torch.equal(out[:, 0], base[:, 0])


def test_slot_embedding_is_deterministic_per_position():
    """Slot k always reads row k of the embedding, at every frame of the clip —
    the canonical (pose, contact, force, motion) concatenation order the model
    builds is what gives each row its meaning."""
    module = _module(num_layers=1, num_slots=2)
    _nudge(module)
    with torch.no_grad():                          # make the two rows identical
        module.slot_embed[1].copy_(module.slot_embed[0])
    gen = torch.Generator().manual_seed(8)
    tokens = torch.randn(4, 2, _DIM, generator=gen)
    pos = torch.arange(4, dtype=torch.float32) / 25.0
    base = module(tokens, 4, pos, None)
    swapped = module(tokens.flip(1), 4, pos, None).flip(1)
    assert torch.allclose(base, swapped, atol=1e-5)   # equivariant again


# ---------------------------------------------------------------- plumbing

def test_gradients_reach_all_parameters():
    module = _module().train()
    tokens, pos, valid = _clip()
    out = module(tokens.requires_grad_(True), 6, pos, valid)
    out.square().mean().backward()
    bad = [n for n, p in module.named_parameters()
           if p.grad is None or not torch.isfinite(p.grad).all()]
    assert bad == []
    gamma_norm = sum(p.grad.abs().sum()
                     for n, p in module.named_parameters() if "gamma" in n)
    assert float(gamma_norm) > 0.0


def test_shape_and_dtype_are_preserved():
    module = _module()
    _nudge(module)
    tokens, pos, valid = _clip(n_clips=3, t=5)
    out = module(tokens, 5, pos, valid)
    assert out.shape == tokens.shape
    assert out.dtype == torch.float32


def test_constructor_validation():
    with pytest.raises(ValueError):
        _module(time_scale=0.0)
    with pytest.raises(ValueError):
        _module(max_rel_sec=-1.0)
    with pytest.raises(ValueError):
        _module(num_slots=0)
    with pytest.raises(ValueError):
        _module(num_heads=5)                       # dim not divisible


def test_forward_rejects_a_mismatched_token_count():
    module = _module()
    tokens, pos, valid = _clip(slots=_SLOTS + 1)
    with pytest.raises(AssertionError, match="token count"):
        module(tokens, 6, pos, valid)


def test_forward_rejects_a_ragged_batch():
    module = _module()
    tokens, _, _ = _clip(n_clips=1, t=7)
    with pytest.raises(AssertionError, match="not divisible"):
        module(tokens, 4, None, None)
