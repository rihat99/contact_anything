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
from contact.engine import forward_contact, forward_model
from contact.losses import MultiTargetContactLoss
from contact.model import build_model
from contact.targets import NUM_BODY_22, TargetSpec

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPORAL_CFG = os.path.join(
    REPO, "configs", "old", "climbing_videos_joint_temporal_center_v2.yaml")
JOINT_CFG = os.path.join(REPO, "configs", "old", "climbing_videos_joint.yaml")
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
            "joint_supervised": torch.ones(NUM_BODY_22),
            "joint_confidence": torch.ones(NUM_BODY_22),
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


def test_every_trainable_param_gets_nonzero_grad():
    placement = "post_decoder"
    """Every trainable contact param gets a *nonzero* grad.

    A ``p.grad is not None`` check passes even for params on a multiply-by-zero
    path (temporal weights at zero-gate init). We move the gates off zero first,
    then require ``p.grad.abs().sum() > 0`` for every trainable param.
    """
    torch.manual_seed(0)
    cfg = load_config(TEMPORAL_CFG)
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


# ---------------------------------------------------------------- force branch (step 04)

def _force_cfg(freeze_contact: bool) -> dict:
    cfg = load_config(JOINT_CFG)                    # extremities_4, per_token joint target
    cfg["model"]["force_head"]["enabled"] = True
    cfg["train"]["detach_interm_preds"] = True
    cfg["train"]["backbone_no_grad"] = True
    cfg["train"]["freeze_contact"] = freeze_contact
    return cfg


def _randomize_force_head_final(model, seed=7):
    """Move the zero-init force output layer off zero so a magnitude force term
    has a nonzero gradient (at exact zero-init ``d(f^2)/dθ = 2f·f' = 0``)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    linears = [m for m in model.head_force.proj.modules() if isinstance(m, torch.nn.Linear)]
    final = linears[-1]
    with torch.no_grad():
        final.weight.copy_(torch.randn(final.weight.shape, generator=gen, device="cuda"))
        final.bias.copy_(torch.randn(final.bias.shape, generator=gen, device="cuda"))


def _force_loss_step(model, batch, loss_fn, seed=7):
    """One fwd/bwd whose loss touches both branches. The real physics force loss
    is step 06; here a graph-connected ``force`` term only exercises grad flow."""
    _randomize_force_head_final(model)              # non-zero forces -> live grads
    model.zero_grad(set_to_none=True)
    model.train()
    torch.manual_seed(seed)
    out = forward_model(model, batch)
    logits = {t: out["contact"][f"{t}_logits"] for t in loss_fn.target_names}
    loss, _ = loss_fn(logits, batch["targets"])
    loss = loss + out["force"]["joint_forces"].pow(2).sum()
    loss.backward()
    return loss.detach()


def _randomize_force_temporal_gammas(model, seed=13):
    """Move every force_temporal residual gate off zero so its subtree is live."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for name, p in model.force_temporal.named_parameters():
            if "gamma" in name:
                p.copy_(torch.randn(p.shape, generator=gen, device="cuda"))


def test_force_temporal_params_get_nonzero_grad():
    """Every force_temporal param gets a *nonzero* grad from a force-reading loss
    across a T>1 clip, with the gammas AND the zero-init force head final layer
    randomized. A zero final head layer would zero every upstream force grad
    (including force_temporal.*) regardless of the gammas, so the test would pass
    spuriously at true init unless the head is moved off zero first."""
    torch.manual_seed(0)
    cfg = _force_cfg(freeze_contact=False)
    cfg["model"]["force_temporal"]["enabled"] = True
    model, _ = build_model(cfg, "cuda")
    try:
        _randomize_force_temporal_gammas(model)     # gates off zero -> live subtree
        batch = _batch(cfg, model, _synth_frames(8), seq_len=4)   # 2 clips, T=4
        loss_fn = MultiTargetContactLoss(cfg).to("cuda")
        loss = _force_loss_step(model, batch, loss_fn)   # randomizes head, reads joint_forces
        assert torch.isfinite(loss).all()

        ft_params = [(n, p) for n, p in model.named_parameters() if "force_temporal" in n]
        assert ft_params, "no force_temporal params found"
        for name, p in ft_params:
            assert p.requires_grad, f"force_temporal {name} not trainable"
            assert p.grad is not None, f"force_temporal {name} received no grad"
            assert float(p.grad.abs().sum()) > 0, f"force_temporal {name} got an all-zero grad"
    finally:
        del model
        torch.cuda.empty_cache()


def test_force_enabled_only_contact_or_force_are_trainable():
    torch.manual_seed(0)
    cfg = _force_cfg(freeze_contact=False)
    model, _ = build_model(cfg, "cuda")
    try:
        batch = _batch(cfg, model, _synth_frames(4), seq_len=1)
        loss_fn = MultiTargetContactLoss(cfg).to("cuda")
        loss = _force_loss_step(model, batch, loss_fn)
        assert torch.isfinite(loss).all()

        n_train = 0
        for name, p in model.named_parameters():
            lname = name.lower()
            if p.requires_grad:
                assert "contact" in lname or "force" in lname, f"unexpected trainable {name}"
                n_train += 1
            else:
                assert p.grad is None, f"frozen {name} unexpectedly has a grad"
        assert n_train > 0
        # Both branches present and trainable (regime b: joint + physics jointly).
        assert any("force" in n.lower() and p.requires_grad
                   for n, p in model.named_parameters())
        assert any("contact" in n.lower() and p.requires_grad
                   for n, p in model.named_parameters())
    finally:
        del model
        torch.cuda.empty_cache()


def test_freeze_contact_trains_force_only_and_pins_eval():
    torch.manual_seed(0)
    cfg = _force_cfg(freeze_contact=True)
    model, _ = build_model(cfg, "cuda")
    try:
        batch = _batch(cfg, model, _synth_frames(4), seq_len=1)
        loss_fn = MultiTargetContactLoss(cfg).to("cuda")
        loss = _force_loss_step(model, batch, loss_fn)
        assert torch.isfinite(loss).all()

        contact_params = [(n, p) for n, p in model.named_parameters() if "contact" in n.lower()]
        force_params = [(n, p) for n, p in model.named_parameters() if "force" in n.lower()]
        assert contact_params and force_params

        # Contact frozen: no requires_grad, no grad.
        for name, p in contact_params:
            assert not p.requires_grad, f"contact {name} still trainable under freeze_contact"
            assert p.grad is None, f"frozen contact {name} received a grad"
        # Every force param trains; at least one gets a nonzero grad (plumbing live).
        for name, p in force_params:
            assert p.requires_grad, f"force {name} not trainable"
            assert p.grad is not None, f"force {name} received no grad"
        assert any(float(p.grad.abs().sum()) > 0 for _, p in force_params)
        # Frozen base params never get grads.
        for name, p in model.named_parameters():
            if "contact" not in name.lower() and "force" not in name.lower():
                assert p.grad is None, f"frozen base {name} received a grad"

        # Eval-pin: after train(True) with freeze_contact, contact submodules report
        # eval, force submodules train.
        model.train(True)
        checked_force = checked_contact = 0
        for mod_name, m in model.named_modules():
            if not mod_name:
                continue
            if "head_force" in mod_name or mod_name.startswith("force_"):
                assert m.training is True, f"force module {mod_name} not in train mode"
                checked_force += 1
            elif "head_contact" in mod_name or mod_name.startswith("contact_"):
                assert m.training is False, f"frozen contact module {mod_name} in train mode"
                checked_contact += 1
        assert checked_force > 0 and checked_contact > 0
    finally:
        del model
        torch.cuda.empty_cache()
