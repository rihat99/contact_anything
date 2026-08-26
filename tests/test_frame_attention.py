"""FrameAttentionModule unit tests (CPU): no-op init, frame independence."""
from __future__ import annotations

import pytest
import torch

from sam_3d_body.models.modules.frame_attention import FrameAttentionModule

_DIM = 16
_K = 3
_S = 7   # context tokens


def _module(**kw):
    defaults = dict(dim=_DIM, num_layers=2, num_heads=4, mlp_ratio=2.0,
                    dropout=0.0)
    defaults.update(kw)
    m = FrameAttentionModule(**defaults)
    m.eval()
    return m


def _heat(m, seed=13):
    """Move every zero-init gamma off zero so the module actually mixes."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "gamma" in name:
                p.copy_(torch.randn(p.shape, generator=gen))
    return m


# ---------------------------------------------------------------- zero-gamma no-op

@pytest.mark.parametrize("bottleneck_dim", [None, 8])
def test_zero_gamma_is_exact_identity(bottleneck_dim):
    m = _module(bottleneck_dim=bottleneck_dim,
                num_heads=2 if bottleneck_dim else 4)
    tokens = torch.randn(6, _K, _DIM)
    ctx = torch.randn(6, _S, _DIM)
    out = m(tokens, ctx)
    assert out.shape == tokens.shape
    assert torch.equal(out, tokens), "zero-gamma module must be an exact identity"


def test_gate_params_present_and_zero_init():
    m = _module(num_layers=2)
    gammas = [(n, p) for n, p in m.named_parameters() if "gamma" in n]
    assert len(gammas) == 4                      # attn + ffn per layer
    assert all(torch.equal(p, torch.zeros_like(p)) for _, p in gammas)


# ---------------------------------------------------------------- shape contract

def test_context_shape_must_match_tokens():
    tokens = torch.randn(4, _K, _DIM)
    with pytest.raises(AssertionError, match="context"):
        _module()(tokens, torch.randn(3, _S, _DIM))     # frame-count mismatch
    with pytest.raises(AssertionError, match="context"):
        _module()(tokens, torch.randn(4, _S, _DIM + 1))  # dim mismatch


# ---------------------------------------------------------------- frame isolation

def test_frames_are_independent():
    """No temporal mixing: perturbing frame i changes ONLY frame i's output."""
    m = _heat(_module(bottleneck_dim=8, num_heads=2))
    tokens = torch.randn(6, _K, _DIM)
    ctx = torch.randn(6, _S, _DIM)
    base = m(tokens, ctx)

    poked = tokens.clone()
    poked[2] += 1.0
    out = m(poked, ctx)
    changed = (out - base).abs().amax(dim=(1, 2))
    assert float(changed[2]) > 1e-4
    others = torch.cat([changed[:2], changed[3:]])
    assert torch.equal(others, torch.zeros_like(others))


def test_module_reads_the_context():
    m = _heat(_module(bottleneck_dim=8, num_heads=2))
    tokens = torch.randn(5, _K, _DIM)
    ctx = torch.randn(5, _S, _DIM)
    base = m(tokens, ctx)
    poked_ctx = ctx.clone()
    poked_ctx[1] += 1.0
    out = m(tokens, poked_ctx)
    changed = (out - base).abs().amax(dim=(1, 2))
    assert float(changed[1]) > 1e-4
    others = torch.cat([changed[:1], changed[2:]])
    assert torch.equal(others, torch.zeros_like(others))


# ---------------------------------------------------------------- gradients

def test_bottleneck_gate_gradients_flow():
    m = _module(bottleneck_dim=8, num_heads=2)
    tokens = torch.randn(4, _K, _DIM, requires_grad=True)
    ctx = torch.randn(4, _S, _DIM)
    out = m(tokens, ctx)
    assert torch.equal(out, tokens)
    out.square().sum().backward()
    gammas = [p for n, p in m.named_parameters() if "gamma_" in n]
    assert all(p.grad is not None for p in gammas)
    # The first (innermost) attention gamma sees a live gradient even at init.
    assert any(float(p.grad.abs().sum()) > 0 for p in gammas)
