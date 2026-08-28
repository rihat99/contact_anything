"""GPU invariance tests: the force branch never perturbs frozen MHR or contact.

Real SAM-3D-Body checkpoint required. Mirrors ``test_temporal_invariance.py``.

The force tokens are appended *after* the contact tokens and the asymmetric mask
blocks every earlier token block from attending them (D1), so both the frozen
MHR/pose outputs and the contact logits are *mathematically* independent of every
force parameter. The exact guarantee is a zero Jacobian; the forward values only
agree to within the CUDA noise floor (the longer token sequence can reorder SDPA
reductions), so noise-floor assertions calibrate against two force-disabled runs.

The force head's final linear is zero-initialised, so at init the *upstream* force
params (embedding / posemb / feat linears) have identically zero gradient through
the head. The sanity check that force outputs depend on force params therefore
randomises the final layer first; a separate test documents the zero-init state.
"""
from __future__ import annotations

import contextlib
import os

import numpy as np
import pytest
import torch

from contact.config import load_config
from contact.data.collate import batch_to_device, make_collate
from contact.engine import forward_model
from contact.model import build_model
from contact.targets import NUM_BODY_22, TargetSpec

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOINT_CFG = os.path.join(REPO, "configs", "old", "climbing_videos_joint.yaml")
_CKPT = load_config(os.path.join(REPO, "configs", "base.yaml"))["model"]["checkpoint_path"]

_NOISE_MARGIN = 8.0
_NOISE_FLOOR_EPS = 1e-6

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint missing"),
]


# ---------------------------------------------------------------- helpers

_FORCE_SIGNAL = 1e-3


def _cfg(force_temporal: bool = False) -> dict:
    cfg = load_config(JOINT_CFG)            # extremities_4, per_token joint target
    cfg["model"]["force_head"]["enabled"] = True
    if force_temporal:
        cfg["model"]["force_temporal"]["enabled"] = True
    return cfg


def _build(force_temporal: bool = False):
    torch.manual_seed(0)
    cfg = _cfg(force_temporal)
    model, _ = build_model(cfg, "cuda")
    model.eval()
    return model, cfg


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


def _batch(cfg, model, frames, seq_len=1):
    spec = TargetSpec.from_config(cfg)
    collate = make_collate(tuple(model.cfg.MODEL.IMAGE_SIZE), spec)
    if seq_len == 1:
        items = list(frames)                    # each frame is its own T=1 clip
    else:
        assert len(frames) % seq_len == 0
        items = [frames[i:i + seq_len] for i in range(0, len(frames), seq_len)]
    return batch_to_device(collate(items), "cuda")


def _outputs(model, batch):
    """Frozen outputs (MHR floats + contact logits) as a flat float dict."""
    out = forward_model(model, batch)
    res = {"__contact__": out["contact"]["joint_logits"].detach().float().clone()}
    for key, val in out["mhr"].items():
        if torch.is_tensor(val) and val.is_floating_point():
            res[key] = val.detach().float().clone()
    return res


def _max_abs(a, b):
    return float((a - b).abs().max())


@contextlib.contextmanager
def _force_disabled(model):
    model.cfg.defrost()
    model.cfg.MODEL.DECODER.DO_FORCE_TOKENS = False
    model.cfg.freeze()
    try:
        yield
    finally:
        model.cfg.defrost()
        model.cfg.MODEL.DECODER.DO_FORCE_TOKENS = True
        model.cfg.freeze()


@contextlib.contextmanager
def _force_temporal_disabled(model):
    """Drop the ``force_temporal`` submodule so the post-decoder hook takes its
    ``getattr(..., None)`` skip path — the exact code path of a disabled build."""
    ft = model.force_temporal
    del model.force_temporal
    try:
        yield
    finally:
        model.force_temporal = ft


def _randomize_force_temporal_gammas(model, seed=13):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for name, p in model.force_temporal.named_parameters():
            if "gamma" in name:
                p.copy_(torch.randn(p.shape, generator=gen, device="cuda"))


def _force_params(model):
    return [(n, p) for n, p in model.named_parameters() if "force" in n.lower()]


def _final_force_linear(model):
    linears = [m for m in model.head_force.proj.modules() if isinstance(m, torch.nn.Linear)]
    return linears[-1]


def _randomize_force_head_final(model, seed=7):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    final = _final_force_linear(model)
    with torch.no_grad():
        final.weight.copy_(torch.randn(final.weight.shape, generator=gen, device="cuda"))
        final.bias.copy_(torch.randn(final.bias.shape, generator=gen, device="cuda"))


def _noise_floor(model, batch):
    with _force_disabled(model):
        a = _outputs(model, batch)
        b = _outputs(model, batch)
    return a, {k: _max_abs(a[k], b[k]) for k in a}


def _assert_within_floor(enabled, disabled, floor, keys, tag):
    for key in keys:
        limit = _NOISE_MARGIN * floor[key] + _NOISE_FLOOR_EPS
        diff = _max_abs(enabled[key], disabled[key])
        assert diff <= limit, (
            f"[{tag}] output {key!r} moved {diff:.2e} > {limit:.2e} "
            f"(base noise {floor[key]:.2e}) — force leaked into a frozen output")


# ---------------------------------------------------------------- (a) exact Jacobian isolation

def test_force_params_have_no_mhr_or_contact_jacobian():
    """Randomised force head: MHR + contact have a zero force Jacobian; force moves."""
    model, cfg = _build()
    try:
        _randomize_force_head_final(model)          # upstream force params now live
        batch = _batch(cfg, model, _synth_frames(4))
        out = forward_model(model, batch)           # grad enabled (no no_grad)
        fparams = [p for _, p in _force_params(model)]
        assert fparams, "no force params found"

        # Sanity: force outputs DO depend on force params (graph is live).
        force = out["force"]["joint_forces"].float().sum()
        assert force.requires_grad
        gf = torch.autograd.grad(force, fparams, allow_unused=True, retain_graph=True)
        assert any(g is not None and float(g.abs().sum()) > 0 for g in gf), (
            "force outputs have no gradient w.r.t. force params — test not exercising it")

        # Contact logits: exactly-zero force Jacobian (D1 -> regime (a) preserves contact).
        contact = out["contact"]["joint_logits"].float().sum()
        if contact.requires_grad:
            gc = torch.autograd.grad(contact, fparams, allow_unused=True, retain_graph=True)
            for g in gc:
                assert g is None or float(g.abs().sum()) == 0.0, (
                    "contact logits have a nonzero force Jacobian")

        # Every MHR output: no grad required, or an exactly-zero force Jacobian.
        checked = 0
        for key, val in out["mhr"].items():
            if not (torch.is_tensor(val) and val.is_floating_point()):
                continue
            checked += 1
            if not val.requires_grad:
                continue
            grads = torch.autograd.grad(val.float().sum(), fparams,
                                        allow_unused=True, retain_graph=True)
            for g in grads:
                assert g is None or float(g.abs().sum()) == 0.0, (
                    f"MHR output {key!r} has a nonzero force Jacobian")
        assert checked > 0
    finally:
        del model
        torch.cuda.empty_cache()


def test_force_temporal_moves_force_and_isolates_frozen():
    """force_temporal enabled + gammas live: force moves across frames while MHR
    and contact keep an exactly-zero Jacobian w.r.t. all force params (now
    including ``force_temporal.*``)."""
    model, cfg = _build(force_temporal=True)
    try:
        _randomize_force_head_final(model)          # nonzero forces to move
        batch = _batch(cfg, model, _synth_frames(4), seq_len=4)   # one T=4 clip

        # Baseline forces with force_temporal an exact identity (module removed).
        with _force_temporal_disabled(model):
            f_base = forward_model(model, batch)["force"]["joint_forces"]
            f_base = f_base.detach().float().clone()

        # Live temporal: cross-frame mixing changes the per-frame forces.
        _randomize_force_temporal_gammas(model)
        out = forward_model(model, batch)           # grad enabled (no no_grad)
        f_live = out["force"]["joint_forces"]
        moved = _max_abs(f_live.detach().float(), f_base)
        assert moved > _FORCE_SIGNAL, (
            f"force_temporal barely moved the forces across frames ({moved:.2e})")

        fparams = [p for _, p in _force_params(model)]
        ft_params = [p for n, p in _force_params(model) if "force_temporal" in n]
        assert ft_params, "no force_temporal params picked up by the 'force' filter"

        # Sanity: force output depends on the force_temporal params (graph is live).
        gft = torch.autograd.grad(f_live.float().sum(), ft_params,
                                  allow_unused=True, retain_graph=True)
        assert any(g is not None and float(g.abs().sum()) > 0 for g in gft), (
            "force output has no gradient w.r.t. force_temporal params")

        # Contact logits: exactly-zero force Jacobian.
        contact = out["contact"]["joint_logits"].float().sum()
        if contact.requires_grad:
            gc = torch.autograd.grad(contact, fparams, allow_unused=True, retain_graph=True)
            for g in gc:
                assert g is None or float(g.abs().sum()) == 0.0, (
                    "contact logits have a nonzero force Jacobian")

        # Every MHR output: no grad required, or an exactly-zero force Jacobian.
        checked = 0
        for key, val in out["mhr"].items():
            if not (torch.is_tensor(val) and val.is_floating_point()):
                continue
            checked += 1
            if not val.requires_grad:
                continue
            grads = torch.autograd.grad(val.float().sum(), fparams,
                                        allow_unused=True, retain_graph=True)
            for g in grads:
                assert g is None or float(g.abs().sum()) == 0.0, (
                    f"MHR output {key!r} has a nonzero force Jacobian")
        assert checked > 0
    finally:
        del model
        torch.cuda.empty_cache()


def test_zero_init_force_head_final_layer_gets_grad_upstream_does_not():
    """At zero-init only the final linear drives the force output; upstream is dead."""
    model, cfg = _build()                           # zero-init head, NOT randomized
    try:
        batch = _batch(cfg, model, _synth_frames(4))
        out = forward_model(model, batch)
        force = out["force"]["joint_forces"].float().sum()

        final = _final_force_linear(model)
        final_ids = {id(final.weight), id(final.bias)}
        upstream = [p for _, p in _force_params(model) if id(p) not in final_ids]

        gfin = torch.autograd.grad(force, [final.weight, final.bias],
                                   allow_unused=True, retain_graph=True)
        assert all(g is not None and float(g.abs().sum()) > 0 for g in gfin), (
            "zero-init final force linear must still receive gradient")

        gup = torch.autograd.grad(force, upstream, allow_unused=True)
        assert all(g is None or float(g.abs().sum()) == 0.0 for g in gup), (
            "upstream force params must have zero grad through a zero-init head")
    finally:
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- (b) noise-floor

def test_force_branch_within_noise_floor():
    model, cfg = _build()
    try:
        batch = _batch(cfg, model, _synth_frames(4))
        disabled, floor = _noise_floor(model, batch)
        enabled = _outputs(model, batch)            # zero-init force branch live
        _assert_within_floor(enabled, disabled, floor, list(disabled), "force")
    finally:
        del model
        torch.cuda.empty_cache()


def test_contact_gate_wires_final_force_output():
    """Gated build (six kindyn_6 contact tokens matched 1:1 to the six force
    groups): the forward's final force output equals raw * sigmoid(sharpness *
    joint logits[gate map]), with the ungated tensor preserved under
    ``joint_forces_raw``."""
    from sam_3d_body.models.heads.force_head import FORCE_GATE_CONTACT_MAP

    torch.manual_seed(0)
    cfg = load_config(
        os.path.join(REPO, "configs", "old", "climbing_corpus_joint_force_cond_sum1_postdec.yaml"))
    model, _ = build_model(cfg, "cuda")
    model.eval()
    try:
        _randomize_force_head_final(model)          # nonzero raw forces
        batch = _batch(cfg, model, _synth_frames(2))
        with torch.no_grad():
            out = forward_model(model, batch)
        assert sorted(out["force"]) == ["joint_forces", "joint_forces_raw"]
        raw = out["force"]["joint_forces_raw"].float()
        assert float(raw.abs().max()) > 0
        gate = torch.sigmoid(
            4.0 * out["contact"]["joint_logits"].float()
            [:, list(FORCE_GATE_CONTACT_MAP)])
        torch.testing.assert_close(
            out["force"]["joint_forces"].float(), raw * gate.unsqueeze(-1))
    finally:
        del model
        torch.cuda.empty_cache()
