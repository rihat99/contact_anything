"""The blind-contact-token ablation: contact tokens receive no image features.

``model.contact_head.blind_to_image`` cuts both image paths into the contact
tokens — the decoder's image cross-attention is gated off for their rows, and the
keypoint-anchored update (grid-sampled features + 2D keypoint posemb) never runs.
What remains is self-attention over the frozen SAM-3D-Body tokens.

Blindness is proven compositionally, because the contact tokens are *supposed* to
keep an indirect image dependency through the body tokens:

1. the gate makes a token's layer output independent of the image context
   (:func:`test_gate_makes_row_independent_of_context`) and provably does not
   disturb any other row (:func:`test_gate_leaves_other_rows_bit_identical`);
2. the model wires that gate onto exactly the contact rows
   (:func:`test_gate_covers_exactly_the_contact_rows`);
3. the anchored update cannot run — its two projections are not even built
   (:func:`test_blind_model_drops_the_anchored_projections`), so a forward pass
   completing at all is proof the contact block never reaches it.

The gate and config tests run on CPU; the model tests need the real checkpoint and
are marked slow.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from contact.config import load_config
from sam_3d_body.models.modules.transformer import TransformerDecoderLayer

REPO = Path(__file__).resolve().parents[1]
BLIND_CFG = REPO / "configs" / "old" / "climbing_videos_joint_temporal_center_blind.yaml"
BASELINE_CFG = REPO / "configs" / "old" / "climbing_videos_joint_temporal_center_v2.yaml"
_CKPT = load_config(REPO / "configs" / "base.yaml")["model"]["checkpoint_path"]

needs_model = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint missing"),
]

_TOKEN_DIMS = 32
_CONTEXT_DIMS = 24
_NUM_TOKENS = 7
_GATED_ROWS = [5, 6]                 # a trailing block, like the contact tokens
_UNGATED_ROWS = [i for i in range(_NUM_TOKENS) if i not in _GATED_ROWS]


def _layer() -> TransformerDecoderLayer:
    torch.manual_seed(0)
    layer = TransformerDecoderLayer(
        token_dims=_TOKEN_DIMS, context_dims=_CONTEXT_DIMS,
        num_heads=4, head_dims=8, mlp_dims=16,
    )
    return layer.eval()


def _gate() -> torch.Tensor:
    gate = torch.ones(1, _NUM_TOKENS, 1)
    gate[:, _GATED_ROWS] = 0.0
    return gate


def _inputs(seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randn(2, _NUM_TOKENS, _TOKEN_DIMS, generator=generator)
    context = torch.randn(2, 11, _CONTEXT_DIMS, generator=generator)
    return tokens, context


# ---------------------------------------------------------------- gate mechanics


def test_gate_makes_row_independent_of_context():
    """A gated row's layer output does not depend on the image context at all.

    Within one layer a row's only context-dependent term is its cross-attention
    output, so zeroing that term must make two unrelated contexts produce the
    identical row.
    """
    layer = _layer()
    tokens, context = _inputs()
    _, other_context = _inputs(seed=99)
    gate = _gate()

    with torch.no_grad():
        out_a, _ = layer(tokens, context, x_context_gate=gate)
        out_b, _ = layer(tokens, other_context * 7.0 - 3.0, x_context_gate=gate)

    assert torch.equal(out_a[:, _GATED_ROWS], out_b[:, _GATED_ROWS])
    # The ungated rows must still read the image, or the gate proves nothing.
    assert not torch.allclose(out_a[:, _UNGATED_ROWS], out_b[:, _UNGATED_ROWS])


def test_gate_leaves_other_rows_bit_identical():
    """Gating some rows must not perturb any other row by a single ulp.

    Cross-attention is independent per query row (keys/values are image-only), so
    the ungated rows have to match a completely ungated forward exactly.
    """
    layer = _layer()
    tokens, context = _inputs()

    with torch.no_grad():
        ungated_out, _ = layer(tokens, context)
        gated_out, _ = layer(tokens, context, x_context_gate=_gate())

    assert torch.equal(ungated_out[:, _UNGATED_ROWS], gated_out[:, _UNGATED_ROWS])
    assert not torch.equal(ungated_out[:, _GATED_ROWS], gated_out[:, _GATED_ROWS])


def test_gate_none_matches_upstream_forward():
    """The default (``None``) path stays byte-for-byte the unablated behaviour."""
    layer = _layer()
    tokens, context = _inputs()

    with torch.no_grad():
        without_arg, _ = layer(tokens, context)
        explicit_none, _ = layer(tokens, context, x_context_gate=None)
        all_ones, _ = layer(tokens, context,
                            x_context_gate=torch.ones(1, _NUM_TOKENS, 1))

    assert torch.equal(without_arg, explicit_none)
    assert torch.equal(without_arg, all_ones)


def test_gate_survives_backward():
    """A zero gate must not poison gradients the way a fully-masked row would."""
    layer = _layer()
    tokens, context = _inputs()
    tokens.requires_grad_(True)

    out, _ = layer(tokens, context, x_context_gate=_gate())
    out.sum().backward()

    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


# ---------------------------------------------------------------- config surface


def test_blind_defaults_to_off():
    cfg = load_config(REPO / "configs" / "base.yaml")
    assert cfg["model"]["contact_head"]["blind_to_image"] is False


def test_blind_config_differs_from_baseline_in_one_knob():
    """The ablation must be the baseline recipe plus exactly this one flag."""
    cfg = load_config(BLIND_CFG)
    baseline = load_config(BASELINE_CFG)
    assert cfg["model"]["contact_head"]["blind_to_image"] is True
    assert baseline["model"]["contact_head"]["blind_to_image"] is False

    cfg["model"]["contact_head"]["blind_to_image"] = False
    cfg["output"]["exp_name"] = baseline["output"]["exp_name"]
    cfg["base"] = baseline["base"]
    assert cfg == baseline


# ---------------------------------------------------------------- model wiring


def _build_blind():
    from contact.model import build_model
    torch.manual_seed(0)
    cfg = load_config(BLIND_CFG)
    model, trainable = build_model(cfg, "cuda")
    return model, cfg, trainable


def _clip_batch(cfg, model, num_frames: int = 2):
    from contact.data.collate import batch_to_device, make_collate
    from contact.targets import NUM_BODY_22, TargetSpec

    rng = np.random.RandomState(0)
    frames = [{
        "image": (rng.rand(200, 160, 3) * 255).astype(np.uint8),
        "mask": np.ones((200, 160), np.uint8) * 255,
        "bbox": np.array([10.0, 10.0, 150.0, 190.0], np.float32),
        "cam_int": np.eye(3, dtype=np.float32) * 500.0,
        "joint_contact": torch.zeros(NUM_BODY_22),
        "joint_mask": torch.ones(NUM_BODY_22),
        "joint_supervised": torch.ones(NUM_BODY_22),
        "joint_confidence": torch.ones(NUM_BODY_22),
        "frame_pos_sec": t * 0.1,
        "frame_valid": True,
    } for t in range(num_frames)]
    spec = TargetSpec.from_config(cfg)
    collate = make_collate(tuple(model.cfg.MODEL.IMAGE_SIZE), spec)
    return batch_to_device(collate([frames]), "cuda")


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint missing")
def test_blind_model_drops_the_anchored_projections():
    """The anchored-update projections are not built at all.

    They would otherwise be trainable params that never receive a gradient, which
    DDP rejects without ``find_unused_parameters``. Their absence is also what
    makes the anchored update *impossible* rather than merely skipped.
    """
    model, _, trainable = _build_blind()

    assert model.contact_blind_to_image is True
    assert not hasattr(model, "contact_posemb_linear")
    assert not hasattr(model, "contact_feat_linear")
    assert not any("contact_posemb" in name or "contact_feat" in name
                   for name in trainable)
    # Tokens, head and the temporal block must still train.
    assert any("contact_embedding" in name for name in trainable)
    assert any("head_contact" in name for name in trainable)
    assert any("contact_temporal" in name for name in trainable)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint missing")
def test_gate_covers_exactly_the_contact_rows():
    """The gate reaching the decoder is zero on the contact rows and one elsewhere."""
    from contact.engine import forward_model

    model, cfg, _ = _build_blind()
    model.eval()
    batch = _clip_batch(cfg, model)
    seen = {}

    decoder_cls = type(model.decoder)
    original = decoder_cls.forward

    def spy(self, *args, **kwargs):
        if self is model.decoder:
            seen["gate"] = kwargs.get("token_context_gate")
        return original(self, *args, **kwargs)

    decoder_cls.forward = spy
    try:
        with torch.no_grad():
            forward_model(model, batch)
    finally:
        decoder_cls.forward = original

    gate = seen["gate"]
    assert gate is not None, "the blind model must pass a cross-attention gate"
    flat = gate.reshape(-1)
    contact_start = gate.shape[1] - model.total_contact_tokens
    zero_rows = (flat == 0).nonzero().flatten().tolist()
    assert zero_rows == list(range(contact_start, gate.shape[1]))
    assert float(flat[:contact_start].min()) == 1.0


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint missing")
def test_blind_every_trainable_param_receives_grad():
    """DDP rejects params that never get a gradient — none may be left dangling."""
    from contact.engine import forward_model

    model, cfg, trainable = _build_blind()
    model.train()
    batch = _clip_batch(cfg, model)

    out = forward_model(model, batch)
    out["contact"]["joint_logits"].square().mean().backward()

    named = dict(model.named_parameters())
    missing = [name for name in trainable if named[name].grad is None]
    assert not missing, f"params with no gradient (DDP would fail): {missing}"


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint missing")
def test_blind_contact_logits_still_respond_to_the_image():
    """Blind means no *direct* image path, not a disconnected head.

    The contact tokens still self-attend the body tokens, which are image-derived,
    so a different image must still move the logits. Without this the ablation
    could be passing for the trivial reason that nothing reaches the head.
    """
    from contact.engine import forward_model

    model, cfg, _ = _build_blind()
    model.eval()
    batch = _clip_batch(cfg, model)

    with torch.no_grad():
        first = forward_model(model, batch)["contact"]["joint_logits"].clone()
        batch["img"] = torch.flip(batch["img"], dims=[-1])
        second = forward_model(model, batch)["contact"]["joint_logits"].clone()

    assert not torch.allclose(first, second)
