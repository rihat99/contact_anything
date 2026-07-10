"""GPU grad-flow test for the Phase-4 efficiency flags.

Real SAM-3D-Body checkpoint + temporal (post_decoder) enabled. Asserts:

* one training step with ``backbone_no_grad`` + ``detach_interm_preds`` on gives
  every trainable (``contact*``) param a finite, non-``None`` grad, a finite loss,
  and leaves every frozen param's grad ``None``;
* flipping both flags off vs on produces the *same* trainable-param grads
  (rtol 1e-3) on the same seeded batch — proving the flags only prune dead-end
  graph and never change the gradients the optimiser sees.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from contact.config import load_config
from contact.data.collate import batch_to_device, make_collate
from contact.engine import forward_contact
from contact.losses import MultiTargetContactLoss
from contact.model import build_model
from contact.targets import NUM_BODY_22, TargetSpec

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPORAL_CFG = os.path.join(REPO, "configs", "climbing_videos_joint_temporal.yaml")
_CKPT = load_config(os.path.join(REPO, "configs", "base.yaml"))["model"]["checkpoint_path"]

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint missing"),
]


def _synth_frames(n: int):
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
            "frame_pos_sec": t * 0.1,
            "frame_valid": True,
        })
    return frames


def _batch(cfg, model, frames, seq_len=4):
    spec = TargetSpec.from_config(cfg)
    collate = make_collate(tuple(model.cfg.MODEL.IMAGE_SIZE), spec)
    items = [frames[i:i + seq_len] for i in range(0, len(frames), seq_len)]
    return batch_to_device(collate(items), "cuda")


def _randomize_gammas(model, seed=7):
    """Move every temporal residual gate off zero so its subtree is on a live path."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for name, p in model.contact_temporal.named_parameters():
            if "gamma" in name:
                p.copy_(torch.randn(p.shape, generator=gen, device="cuda"))


def _set_flags(model, on: bool):
    model.cfg.defrost()
    model.cfg.MODEL.EFFICIENCY.BACKBONE_NO_GRAD = on
    model.cfg.MODEL.EFFICIENCY.DETACH_INTERM_PREDS = on
    model.cfg.freeze()
    model.decoder.detach_interm_preds = on
    model.decoder_hand.detach_interm_preds = on


def _step_grads(model, batch, loss_fn, seed=7):
    """One fwd/bwd with deterministic dropout; return (loss, {name: grad})."""
    model.zero_grad(set_to_none=True)
    model.train()
    torch.manual_seed(seed)                       # pin contact-head dropout masks
    contact = forward_contact(model, batch)
    logits = {t: contact[f"{t}_logits"] for t in loss_fn.target_names}
    loss, _ = loss_fn(logits, batch["targets"])
    loss.backward()
    grads = {n: p.grad.detach().clone()
             for n, p in model.named_parameters() if p.grad is not None}
    return loss.detach(), grads


@pytest.fixture(scope="module")
def model_batch_loss():
    torch.manual_seed(0)
    cfg = load_config(TEMPORAL_CFG)
    cfg["train"]["detach_interm_preds"] = True
    cfg["train"]["backbone_no_grad"] = True
    model, _ = build_model(cfg, "cuda")
    batch = _batch(cfg, model, _synth_frames(8), seq_len=4)
    loss_fn = MultiTargetContactLoss(cfg).to("cuda")
    try:
        yield model, batch, loss_fn
    finally:
        del model
        torch.cuda.empty_cache()


def test_flags_on_every_contact_param_gets_grad(model_batch_loss):
    model, batch, loss_fn = model_batch_loss
    _set_flags(model, True)
    loss, grads = _step_grads(model, batch, loss_fn)

    assert torch.isfinite(loss).all(), "loss is not finite"
    n_contact = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert "contact" in name.lower(), f"unexpected trainable param {name}"
            assert p.grad is not None, f"trainable {name} received no grad"
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
            n_contact += 1
        else:
            assert p.grad is None, f"frozen {name} unexpectedly has a grad"
    assert n_contact > 0


@pytest.mark.parametrize("placement", ["post_decoder", "between_layers", "pre_decoder"])
def test_every_trainable_param_gets_nonzero_grad(placement):
    """Every trainable contact param gets a *nonzero* grad, for all placements.

    A ``p.grad is not None`` check passes even for params on a multiply-by-zero
    path (temporal weights at zero-gate init). We move the gates off zero first,
    then require ``p.grad.abs().sum() > 0`` for every trainable param.
    """
    torch.manual_seed(0)
    cfg = load_config(TEMPORAL_CFG)
    cfg["model"]["temporal"]["placement"] = placement
    cfg["train"]["detach_interm_preds"] = True
    cfg["train"]["backbone_no_grad"] = True
    model, _ = build_model(cfg, "cuda")
    try:
        _randomize_gammas(model)                     # gates off zero -> live subtree
        batch = _batch(cfg, model, _synth_frames(8), seq_len=4)
        loss_fn = MultiTargetContactLoss(cfg).to("cuda")
        loss, _ = _step_grads(model, batch, loss_fn)
        assert torch.isfinite(loss).all()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"{name} received no grad"
                assert float(p.grad.abs().sum()) > 0, f"{name} received an all-zero grad"
    finally:
        del model
        torch.cuda.empty_cache()


def test_flags_do_not_change_gradients(model_batch_loss):
    model, batch, loss_fn = model_batch_loss

    _set_flags(model, False)
    loss_off, grads_off = _step_grads(model, batch, loss_fn)
    _set_flags(model, True)
    loss_on, grads_on = _step_grads(model, batch, loss_fn)

    assert torch.isfinite(loss_off).all() and torch.isfinite(loss_on).all()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable, "no trainable params?"
    for name in trainable:
        assert name in grads_off and name in grads_on, f"{name} missing a grad in one run"
        assert torch.allclose(grads_off[name], grads_on[name], rtol=1e-3, atol=1e-5), (
            f"grad for {name} changed between flags off/on: "
            f"max|diff|={(grads_off[name] - grads_on[name]).abs().max():.2e}")
