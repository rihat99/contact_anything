"""ForceHead unit tests (CPU): shapes, zero-init output, per-token independence,
and the contact-gated final output."""
from __future__ import annotations

import pytest
import torch

from sam_3d_body.models.heads.force_head import (
    FORCE_GATE_CONTACT_MAP,
    ForceHead,
    contact_gate_forces,
)

_DIM = 16
_K = 4


def _head(**kw) -> ForceHead:
    defaults = dict(input_dim=_DIM, num_force_tokens=_K, mlp_depth=2,
                    mlp_channel_div_factor=2, dropout=0.0)
    defaults.update(kw)
    return ForceHead(**defaults)


def _randomize_final_linear(head: ForceHead, seed: int = 0) -> None:
    """Move the zero-init output layer off zero so forces are non-trivial."""
    gen = torch.Generator().manual_seed(seed)
    final = [m for m in head.proj.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        final.weight.copy_(torch.randn(final.weight.shape, generator=gen))
        final.bias.copy_(torch.randn(final.bias.shape, generator=gen))


def test_maps_tokens_to_three_vectors():
    head = _head().eval()
    _randomize_final_linear(head)
    out = head(torch.randn(3, _K, _DIM))
    assert out.shape == (3, _K, 3)


def test_zero_init_predicts_exactly_zero_force():
    head = _head().eval()
    out = head(torch.randn(5, _K, _DIM))
    assert torch.equal(out, torch.zeros_like(out)), (
        "zero-init ForceHead must predict exactly zero force")


def test_backpropagates_to_tokens_and_params():
    head = _head()
    _randomize_final_linear(head)          # else upstream grads are all zero at init
    tokens = torch.randn(2, _K, _DIM, requires_grad=True)

    head(tokens).sum().backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    assert float(tokens.grad.abs().sum()) > 0.0
    assert all(p.grad is not None for p in head.parameters())


def test_per_token_mlp_is_shared_and_preserves_token_identity():
    head = _head().eval()
    _randomize_final_linear(head)
    tokens = torch.randn(2, _K, _DIM)
    permutation = torch.tensor([2, 0, 3, 1])

    expected = head(tokens)[:, permutation]
    actual = head(tokens[:, permutation])
    assert torch.equal(actual, expected)


@torch.no_grad()
def test_perturbing_one_token_changes_only_its_row():
    head = _head().eval()
    _randomize_final_linear(head)
    tokens = torch.randn(2, _K, _DIM)
    base = head(tokens)

    perturbed = tokens.clone()
    perturbed[:, 1] += 1.0
    changed = head(perturbed)

    for k in range(_K):
        moved = float((changed[:, k] - base[:, k]).abs().sum())
        if k == 1:
            assert moved > 0.0, "perturbed token did not change its own row"
        else:
            assert moved == 0.0, f"token {k} moved when only token 1 was perturbed"


def test_rejects_runtime_token_count_mismatch():
    head = _head()
    try:
        head(torch.randn(2, _K + 1, _DIM))
    except ValueError as err:
        assert "input=5, configured=4" in str(err)
    else:
        raise AssertionError("expected a token-count mismatch ValueError")


# ---------------------------------------------------------------- contact gate

def test_gate_map_is_identity_and_sharpness_math():
    """The kindyn_6 contact outputs match the force groups 1:1, so each group is
    scaled by sigmoid(sharpness * its OWN contact logit)."""
    assert FORCE_GATE_CONTACT_MAP == (0, 1, 2, 3, 4, 5)
    logits = torch.tensor([[1.0, -0.5, 0.25, -2.0, 3.0, 0.0]])
    forces = torch.ones(1, 6, 3)
    gated = contact_gate_forces(forces, logits, sharpness=4.0)
    expected = torch.sigmoid(4.0 * logits[0])
    torch.testing.assert_close(gated, expected[None, :, None].expand(1, 6, 3))
    # Sharpness 4.0 anchor: logit 1.0 (p 0.73) -> gate 0.982.
    assert float(gated[0, 0, 0]) == pytest.approx(0.98201, abs=1e-4)
    # Identity map: all six logits differ here, so all six gate factors differ
    # (heel force follows its ankle's contact output, not the foot's).
    assert len(set(gated[0, :, 0].tolist())) == 6


def test_gate_saturates_at_confident_logits():
    logits = torch.tensor([[6.0, -6.0, 6.0, -6.0, 6.0, -6.0]])
    gated = contact_gate_forces(torch.ones(1, 6, 3), logits, sharpness=4.0)
    assert float((gated[:, [0, 2, 4]] - 1.0).abs().max()) < 1e-6   # contact -> ~1
    assert float(gated[:, [1, 3, 5]].abs().max()) < 1e-6           # free -> ~0


def test_gate_detaches_contact_logits_but_not_forces():
    """The gate path must carry ZERO gradient into the contact logits (they are
    detached unconditionally) while the raw forces keep their gradient."""
    logits = torch.randn(2, 6, requires_grad=True)
    forces = torch.randn(2, 6, 3, requires_grad=True)
    contact_gate_forces(forces, logits, sharpness=4.0).sum().backward()
    assert logits.grad is None
    assert forces.grad is not None
    expected = torch.sigmoid(
        4.0 * logits.detach()[:, list(FORCE_GATE_CONTACT_MAP)])
    torch.testing.assert_close(forces.grad, expected[..., None].expand(2, 6, 3))


def test_gate_rejects_wrong_shapes():
    with pytest.raises(ValueError, match="one group per gate-map entry"):
        contact_gate_forces(torch.ones(1, 4, 3), torch.zeros(1, 6), 4.0)
    # Four-extremity logits no longer fit the identity map over six groups.
    with pytest.raises(ValueError, match="produced only"):
        contact_gate_forces(torch.ones(1, 6, 3), torch.zeros(1, 4), 4.0)
