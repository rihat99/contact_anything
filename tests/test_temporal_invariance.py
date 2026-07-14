"""GPU invariance tests: the temporal hooks never perturb frozen pose/MHR outputs.

Real SAM-3D-Body checkpoint required.

**Why not ``torch.equal``.** The frozen SAM-3D-Body forward is *not* run-to-run
bit-deterministic on CUDA: two identical forward passes of the same model on the
same batch already differ by ~1e-7 (contact logits) / ~5e-7 (keypoints), even
under ``torch.use_deterministic_algorithms``. That noise floor is independent of
the temporal module. So these tests **calibrate against that floor**: we measure
it (two disabled passes), then require that

* zero-gamma temporal, and the pose/MHR/keypoint stream under *non-zero* gamma,
  differ from the disabled run by no more than the base nondeterminism, while
* non-zero-gamma contact logits move by **orders of magnitude** more than the floor.

The asymmetric token mask (pose rows never attend to contact tokens) plus the
private image copy for ``pre_decoder`` make pose isolation exact *mathematically*;
the residual we observe is pure CUDA noise.

Disabling is done by dropping the ``contact_temporal`` submodule so the hooks hit
their ``getattr(..., None)`` skip branch — the exact code path of a disabled build.
"""
from __future__ import annotations

import contextlib
import os

import numpy as np
import pytest
import torch

from contact.config import load_config
from contact.data.collate import batch_to_device, make_collate
from contact.model import build_model
from contact.targets import NUM_BODY_22, TargetSpec

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPORAL_CFG = os.path.join(REPO, "configs", "climbing_videos_joint_temporal.yaml")
_CKPT = load_config(os.path.join(REPO, "configs", "base.yaml"))["model"]["checkpoint_path"]

# Margins: enabled-vs-disabled must stay within a few × the base noise floor; a
# real contact change is > CONTACT_SIGNAL (orders of magnitude above the floor).
_NOISE_MARGIN = 8.0
_NOISE_FLOOR_EPS = 1e-6
_CONTACT_SIGNAL = 1e-3

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint missing"),
]


# ---------------------------------------------------------------- helpers

def _cfg(placement: str, attend: str = "joint") -> dict:
    cfg = load_config(TEMPORAL_CFG)
    cfg["model"]["temporal"]["placement"] = placement
    cfg["model"]["temporal"]["attend"] = attend
    return cfg


def _build(placement: str, attend: str = "joint"):
    torch.manual_seed(0)
    cfg = _cfg(placement, attend)
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


def _batch(cfg, model, frames, seq_len):
    spec = TargetSpec.from_config(cfg)
    collate = make_collate(tuple(model.cfg.MODEL.IMAGE_SIZE), spec)
    if seq_len == 1:
        items = list(frames)                       # each frame is its own T=1 clip
    else:
        assert len(frames) % seq_len == 0
        items = [frames[i:i + seq_len] for i in range(0, len(frames), seq_len)]
    return batch_to_device(collate(items), "cuda")


def _outputs(model, batch):
    model._initialize_batch(batch)
    with torch.no_grad():
        out = model.forward_step(batch, decoder_type="body")
    res = {"__contact__": out["contact"]["joint_logits"].detach().float().clone()}
    for key, val in out["mhr"].items():
        if torch.is_tensor(val) and val.is_floating_point():
            res[key] = val.detach().float().clone()
    return res


def _max_abs(a, b):
    return float((a - b).abs().max())


@contextlib.contextmanager
def _temporal_disabled(model):
    tm = model.contact_temporal
    del model.contact_temporal          # hooks take the getattr(..., None) skip path
    try:
        yield
    finally:
        model.contact_temporal = tm


def _randomize_gammas(model, seed=7):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for name, p in model.contact_temporal.named_parameters():
            if "gamma" in name:
                p.copy_(torch.randn(p.shape, generator=gen, device="cuda"))


def _noise_floor(model, batch):
    """Base run-to-run nondeterminism per output key (temporal disabled)."""
    with _temporal_disabled(model):
        a = _outputs(model, batch)
        b = _outputs(model, batch)
    return a, {k: _max_abs(a[k], b[k]) for k in a}


def _assert_within_floor(enabled, disabled, floor, keys, tag):
    for key in keys:
        limit = _NOISE_MARGIN * floor[key] + _NOISE_FLOOR_EPS
        diff = _max_abs(enabled[key], disabled[key])
        assert diff <= limit, (
            f"[{tag}] output {key!r} moved {diff:.2e} > {limit:.2e} "
            f"(base noise {floor[key]:.2e}) — temporal leaked into a frozen output")


# ---------------------------------------------------------------- (a) zero-gamma == disabled

@pytest.mark.parametrize("placement", ["post_decoder", "between_layers", "pre_decoder"])
def test_zero_gamma_matches_disabled(placement):
    model, cfg = _build(placement)
    try:
        batch = _batch(cfg, model, _synth_frames(8), seq_len=4)
        disabled, floor = _noise_floor(model, batch)
        enabled = _outputs(model, batch)                 # zero gammas at init
        # (a) EVERY output — contact + pose — within the base noise floor.
        _assert_within_floor(enabled, disabled, floor, list(enabled), placement)
    finally:
        del model
        torch.cuda.empty_cache()


# -------------------------------------------------------- (a') exact pose isolation

@pytest.mark.parametrize("placement", ["post_decoder", "between_layers", "pre_decoder"])
def test_temporal_params_have_no_pose_jacobian(placement):
    """Exact (noise-free) proof: pose/MHR outputs have a zero temporal Jacobian.

    With non-zero temporal gates, differentiate every frozen pose/MHR output w.r.t.
    every ``contact_temporal`` parameter: each gradient must be ``None`` (the output
    does not depend on it) or bitwise zero. Complements the noise-floor tests.
    """
    model, cfg = _build(placement)
    try:
        _randomize_gammas(model)                         # nonzero gates -> temporal live
        batch = _batch(cfg, model, _synth_frames(8), seq_len=4)
        model._initialize_batch(batch)
        out = model.forward_step(batch, decoder_type="body")   # grad enabled (no no_grad)
        tparams = [p for _, p in model.contact_temporal.named_parameters()]

        # Sanity: contact logits DO depend on the temporal params (graph is live).
        contact = out["contact"]["joint_logits"].float().sum()
        assert contact.requires_grad
        gc = torch.autograd.grad(contact, tparams, allow_unused=True, retain_graph=True)
        assert any(g is not None and float(g.abs().sum()) > 0 for g in gc), (
            f"[{placement}] contact logits have no temporal gradient — test not "
            f"exercising the module")

        # Every pose/MHR output: no grad required (cannot depend on any trainable
        # param) or an exactly-zero temporal Jacobian.
        checked = 0
        for key, val in out["mhr"].items():
            if not (torch.is_tensor(val) and val.is_floating_point()):
                continue
            checked += 1
            if not val.requires_grad:
                continue
            grads = torch.autograd.grad(val.float().sum(), tparams,
                                        allow_unused=True, retain_graph=True)
            for g in grads:
                assert g is None or float(g.abs().sum()) == 0.0, (
                    f"[{placement}] MHR output {key!r} has a nonzero temporal Jacobian")
        assert checked > 0
    finally:
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- (b) nonzero-gamma pose isolation

@pytest.mark.parametrize("placement", ["post_decoder", "between_layers", "pre_decoder"])
def test_nonzero_gamma_isolates_pose(placement):
    model, cfg = _build(placement)
    try:
        batch = _batch(cfg, model, _synth_frames(8), seq_len=4)
        disabled, floor = _noise_floor(model, batch)

        _randomize_gammas(model)
        enabled = _outputs(model, batch)

        # (b) pose/MHR/keypoints stay within the noise floor...
        pose_keys = [k for k in enabled if k != "__contact__"]
        _assert_within_floor(enabled, disabled, floor, pose_keys, placement)
        # ...while contact logits move far beyond it (real signal).
        contact_diff = _max_abs(enabled["__contact__"], disabled["__contact__"])
        assert contact_diff > _CONTACT_SIGNAL, (
            f"[{placement}] non-zero gamma barely changed contact ({contact_diff:.2e})")
        assert contact_diff > 100 * (floor["__contact__"] + _NOISE_FLOOR_EPS)
    finally:
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- (c) T=1 vs T=4

def test_zero_gamma_t1_matches_t4():
    model, cfg = _build("post_decoder")
    try:
        frames = _synth_frames(8)
        batch_t4 = _batch(cfg, model, frames, seq_len=4)
        # base noise floor for contact logits on the T=4 batch
        _, floor = _noise_floor(model, batch_t4)

        out_t4 = _outputs(model, batch_t4)
        out_t1 = _outputs(model, _batch(cfg, model, frames, seq_len=1))
        # Zero-gamma temporal is an exact identity and the frozen forward ignores
        # seq_len, so per-frame contact logits agree to within CUDA noise (not
        # bit-exact: two shapes -> different kernels + base nondeterminism).
        diff = _max_abs(out_t4["__contact__"], out_t1["__contact__"])
        assert torch.allclose(out_t4["__contact__"], out_t1["__contact__"],
                              rtol=1e-4, atol=1e-5), f"T1 vs T4 contact diff {diff:.2e}"
        assert diff <= _NOISE_MARGIN * floor["__contact__"] + 1e-5
    finally:
        del model
        torch.cuda.empty_cache()
