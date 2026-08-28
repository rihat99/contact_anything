"""GPU invariance tests: the motion branch never perturbs the frozen MHR outputs.

Real SAM-3D-Body checkpoint required. Mirrors ``test_force_invariance.py`` for the
motion-only builds (``tests/fixtures/motion_seven_tokens.yaml``
with seven motion tokens and ``configs/old/climbing_corpus_motion_pelvis_t7.yaml``
with one: no contact tokens, no force tokens, K motion tokens appended last).

The motion tokens are appended *after* every other token block and the asymmetric
mask blocks every earlier block from attending them, so the frozen MHR/pose
outputs are *mathematically* independent of every motion parameter. The exact
guarantee is a zero Jacobian; the forward values only agree to within the CUDA
noise floor (the longer token sequence can reorder SDPA reductions), so
noise-floor assertions calibrate against two motion-disabled runs.

The motion head's final linear is zero-initialised, so at init the *upstream*
motion params (embedding / posemb / feat linears) have identically zero gradient
through the head. The sanity check that motion outputs depend on motion params
therefore randomises the final layer first; a separate test documents the
zero-init state.
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
from contact.targets import TargetSpec

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOTION_CFG = os.path.join(REPO, "tests", "fixtures", "motion_seven_tokens.yaml")
PELVIS_CFG = os.path.join(REPO, "configs", "old", "climbing_corpus_motion_pelvis_t7.yaml")
#: Both shipped motion builds: seven anchored tokens (v2) and one (v3, pelvis).
MOTION_CFGS = (MOTION_CFG, PELVIS_CFG)
_CKPT = load_config(os.path.join(REPO, "configs", "base.yaml"))["model"]["checkpoint_path"]

_NOISE_MARGIN = 8.0
_NOISE_FLOOR_EPS = 1e-6
_MOTION_SIGNAL = 1e-3

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint missing"),
]


# ---------------------------------------------------------------- helpers

def _cfg(motion_temporal: bool = True, config_path: str = MOTION_CFG) -> dict:
    cfg = load_config(config_path)              # motion-only build, K anchored tokens
    cfg["model"]["motion_temporal"]["enabled"] = motion_temporal
    return cfg


def _build(motion_temporal: bool = True, config_path: str = MOTION_CFG):
    torch.manual_seed(0)
    cfg = _cfg(motion_temporal, config_path)
    model, trainable = build_model(cfg, "cuda")
    model.eval()
    return model, trainable, cfg


def _synth_frames(n: int):
    rng = np.random.RandomState(1234)
    return [{
        "image": (rng.rand(200, 160, 3) * 255).astype(np.uint8),
        "mask": (np.ones((200, 160), np.uint8) * 255),
        "bbox": np.array([10.0, 10.0, 150.0, 190.0], np.float32),
        "cam_int": (np.eye(3, dtype=np.float32) * 500.0),
        "frame_pos_sec": t * 0.1,
        "frame_valid": True,
    } for t in range(n)]


def _batch(cfg, model, frames, seq_len=1):
    spec = TargetSpec.from_config(cfg)
    collate = make_collate(
        tuple(model.cfg.MODEL.IMAGE_SIZE), spec,
        motion_joints=len(cfg["motion_supervision"]["joint_names"] or range(7)))
    if seq_len == 1:
        items = list(frames)                    # each frame is its own T=1 clip
    else:
        assert len(frames) % seq_len == 0
        items = [frames[i:i + seq_len] for i in range(0, len(frames), seq_len)]
    return batch_to_device(collate(items), "cuda")


def _mhr_outputs(model, batch):
    """Frozen MHR outputs as a flat float dict."""
    out = forward_model(model, batch)
    return {
        key: val.detach().float().clone()
        for key, val in out["mhr"].items()
        if torch.is_tensor(val) and val.is_floating_point()
    }


def _max_abs(a, b):
    return float((a - b).abs().max())


@contextlib.contextmanager
def _motion_disabled(model):
    model.cfg.defrost()
    model.cfg.MODEL.DECODER.DO_MOTION_TOKENS = False
    model.cfg.freeze()
    try:
        yield
    finally:
        model.cfg.defrost()
        model.cfg.MODEL.DECODER.DO_MOTION_TOKENS = True
        model.cfg.freeze()


@contextlib.contextmanager
def _motion_temporal_disabled(model):
    """Drop the ``motion_temporal`` submodule(s) so the hooks take their
    ``getattr(..., None)`` skip paths — the exact code path of a disabled build."""
    mt = model.motion_temporal
    del model.motion_temporal
    try:
        yield
    finally:
        model.motion_temporal = mt


def _randomize_motion_temporal_gammas(model, seed=13):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    modules = [model.motion_temporal]
    with torch.no_grad():
        for module in modules:
            for name, p in module.named_parameters():
                if "gamma" in name:
                    p.copy_(torch.randn(p.shape, generator=gen, device="cuda"))


def _motion_params(model):
    return [(n, p) for n, p in model.named_parameters() if "motion" in n.lower()]


def _final_motion_linear(model):
    linears = [m for m in model.head_motion.proj.modules()
               if isinstance(m, torch.nn.Linear)]
    return linears[-1]


def _randomize_motion_head_final(model, seed=7):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    final = _final_motion_linear(model)
    with torch.no_grad():
        final.weight.copy_(torch.randn(final.weight.shape, generator=gen, device="cuda"))
        final.bias.copy_(torch.randn(final.bias.shape, generator=gen, device="cuda"))


def _noise_floor(model, batch):
    with _motion_disabled(model):
        a = _mhr_outputs(model, batch)
        b = _mhr_outputs(model, batch)
    return a, {k: _max_abs(a[k], b[k]) for k in a}


# ---------------------------------------------------------------- (a) build shape

@pytest.mark.parametrize("config_path", MOTION_CFGS)
def test_motion_only_build_forward_and_trainable_set(config_path):
    """Trainable set = exactly the motion branch; no contact/force module exists."""
    model, trainable, cfg = _build(config_path=config_path)
    anchors = cfg["model"]["motion_head"]["motion_keypoint_indices"]
    n_tokens = len(anchors)
    try:
        assert trainable and all("motion" in name.lower() for name in trainable)
        assert not any("contact" in name.lower() or "force" in name.lower()
                       for name, _ in model.named_parameters())
        assert not hasattr(model, "head_contact")
        assert not hasattr(model, "head_force")
        assert model.num_motion_tokens == n_tokens
        assert model.motion_keypoint_indices == anchors

        batch = _batch(cfg, model, _synth_frames(2))
        out = forward_model(model, batch)
        assert out["contact"] is None
        assert out["force"] is None
        assert out["motion"]["joint_motion"].shape == (2, n_tokens, 6)
        assert out["motion"]["joint_vel"].shape == (2, n_tokens, 3)
        assert out["motion"]["joint_acc"].shape == (2, n_tokens, 3)
        torch.testing.assert_close(
            out["motion"]["joint_motion"][..., :3], out["motion"]["joint_vel"])

        # pin_frozen_eval with no contact/force modules: train(True) flips only the
        # motion branch; the frozen backbone stays eval-pinned.
        model.train(True)
        assert not model.backbone.training
        assert model.head_motion.training
        assert model.motion_temporal.training
    finally:
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- (b) exact Jacobian isolation

@pytest.mark.parametrize("config_path", MOTION_CFGS)
def test_motion_params_have_no_mhr_jacobian(config_path):
    """Randomised motion head: MHR has a zero motion Jacobian; motion moves."""
    model, _, cfg = _build(motion_temporal=False, config_path=config_path)
    try:
        _randomize_motion_head_final(model)         # upstream motion params now live
        batch = _batch(cfg, model, _synth_frames(4))
        out = forward_model(model, batch)           # grad enabled (no no_grad)
        mparams = [p for _, p in _motion_params(model)]
        assert mparams, "no motion params found"

        # Sanity: motion outputs DO depend on motion params (graph is live).
        motion = out["motion"]["joint_motion"].float().sum()
        assert motion.requires_grad
        gm = torch.autograd.grad(motion, mparams, allow_unused=True, retain_graph=True)
        assert any(g is not None and float(g.abs().sum()) > 0 for g in gm), (
            "motion outputs have no gradient w.r.t. motion params — test not exercising it")

        # Every MHR output: no grad required, or an exactly-zero motion Jacobian.
        checked = 0
        for key, val in out["mhr"].items():
            if not (torch.is_tensor(val) and val.is_floating_point()):
                continue
            checked += 1
            if not val.requires_grad:
                continue
            grads = torch.autograd.grad(val.float().sum(), mparams,
                                        allow_unused=True, retain_graph=True)
            for g in grads:
                assert g is None or float(g.abs().sum()) == 0.0, (
                    f"MHR output {key!r} has a nonzero motion Jacobian")
        assert checked > 0
    finally:
        del model
        torch.cuda.empty_cache()


def test_motion_temporal_moves_motion_and_isolates_mhr():
    """motion_temporal enabled + gammas live: motion moves across frames while MHR
    keeps an exactly-zero Jacobian w.r.t. all motion params (now including
    ``motion_temporal.*``)."""
    model, _, cfg = _build(motion_temporal=True)
    try:
        _randomize_motion_head_final(model)         # nonzero motion to move
        batch = _batch(cfg, model, _synth_frames(4), seq_len=4)   # one T=4 clip

        # Baseline with motion_temporal an exact identity (module removed).
        with _motion_temporal_disabled(model):
            m_base = forward_model(model, batch)["motion"]["joint_motion"]
            m_base = m_base.detach().float().clone()

        # Live temporal: cross-frame mixing changes the per-frame outputs.
        _randomize_motion_temporal_gammas(model)
        out = forward_model(model, batch)           # grad enabled (no no_grad)
        m_live = out["motion"]["joint_motion"]
        moved = _max_abs(m_live.detach().float(), m_base)
        assert moved > _MOTION_SIGNAL, (
            f"motion_temporal barely moved the outputs across frames ({moved:.2e})")

        mparams = [p for _, p in _motion_params(model)]
        mt_params = [p for n, p in _motion_params(model) if "motion_temporal" in n]
        assert mt_params, "no motion_temporal params picked up by the 'motion' filter"

        gmt = torch.autograd.grad(m_live.float().sum(), mt_params,
                                  allow_unused=True, retain_graph=True)
        assert any(g is not None and float(g.abs().sum()) > 0 for g in gmt), (
            "motion output has no gradient w.r.t. motion_temporal params")

        checked = 0
        for key, val in out["mhr"].items():
            if not (torch.is_tensor(val) and val.is_floating_point()):
                continue
            checked += 1
            if not val.requires_grad:
                continue
            grads = torch.autograd.grad(val.float().sum(), mparams,
                                        allow_unused=True, retain_graph=True)
            for g in grads:
                assert g is None or float(g.abs().sum()) == 0.0, (
                    f"MHR output {key!r} has a nonzero motion Jacobian")
        assert checked > 0
    finally:
        del model
        torch.cuda.empty_cache()


def test_zero_init_motion_head_final_layer_gets_grad_upstream_does_not():
    """At zero-init only the final linear drives the motion output; upstream is dead."""
    model, _, cfg = _build(motion_temporal=False)   # zero-init head, NOT randomized
    try:
        batch = _batch(cfg, model, _synth_frames(4))
        out = forward_model(model, batch)
        assert float(out["motion"]["joint_motion"].detach().abs().max()) == 0.0
        motion = out["motion"]["joint_motion"].float().sum()

        final = _final_motion_linear(model)
        final_ids = {id(final.weight), id(final.bias)}
        upstream = [p for _, p in _motion_params(model) if id(p) not in final_ids]

        gfin = torch.autograd.grad(motion, [final.weight, final.bias],
                                   allow_unused=True, retain_graph=True)
        assert all(g is not None and float(g.abs().sum()) > 0 for g in gfin), (
            "zero-init final motion linear must still receive gradient")

        gup = torch.autograd.grad(motion, upstream, allow_unused=True)
        assert all(g is None or float(g.abs().sum()) == 0.0 for g in gup), (
            "upstream motion params must have zero grad through a zero-init head")
    finally:
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- (c) noise floor

def test_motion_branch_within_noise_floor():
    model, _, cfg = _build(motion_temporal=True)
    try:
        batch = _batch(cfg, model, _synth_frames(4), seq_len=4)
        disabled, floor = _noise_floor(model, batch)
        enabled = _mhr_outputs(model, batch)        # zero-init motion branch live
        for key in disabled:
            limit = _NOISE_MARGIN * floor[key] + _NOISE_FLOOR_EPS
            diff = _max_abs(enabled[key], disabled[key])
            assert diff <= limit, (
                f"MHR output {key!r} moved {diff:.2e} > {limit:.2e} "
                f"(base noise {floor[key]:.2e}) — motion leaked into a frozen output")
    finally:
        del model
        torch.cuda.empty_cache()
