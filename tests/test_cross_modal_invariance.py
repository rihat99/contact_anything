"""GPU invariance tests: the all-modality RoPE block never perturbs the frozen
pose/MHR outputs unless ``pose`` is a listed (written) modality.

Real SAM-3D-Body checkpoint required. Successor of the retired
``test_temporal_invariance.py`` (the sliding-window ``contact_temporal`` module).

**Why not ``torch.equal``.** The frozen SAM-3D-Body forward is *not* run-to-run
bit-deterministic on CUDA: two identical forward passes of the same model on the
same batch already differ by ~1e-7 (contact logits) / ~5e-7 (keypoints), even
under ``torch.use_deterministic_algorithms``. That noise floor is independent of
the temporal module, so these tests **calibrate against it**: measure it (two
disabled passes), then require that

* the zero-gate block, and the pose/MHR stream under *non-zero* gates, differ
  from the disabled run by no more than the base nondeterminism, while
* non-zero-gate contact logits move by **orders of magnitude** more.

The asymmetric decoder mask (the original tokens never attend the appended
blocks) makes the pose isolation exact *mathematically*; the residual observed
here is pure CUDA noise. ``test_block_params_have_no_pose_jacobian`` proves the
exact statement directly.

Disabling is done by dropping the ``cross_modal_temporal`` submodule so the hook
hits its ``getattr(..., None)`` skip branch — the code path of a disabled build.
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
JOINT_CFG = os.path.join(REPO, "tests", "fixtures", "climbing_videos_joint.yaml")
_CKPT = load_config(os.path.join(REPO, "configs", "base.yaml"))["model"]["checkpoint_path"]

# Margins: enabled-vs-disabled must stay within a few x the base noise floor; a
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

def _cfg(modalities: list[str]) -> dict:
    cfg = load_config(JOINT_CFG)
    cfg["model"]["cross_modal_temporal"] = {
        "enabled": True, "modalities": modalities, "num_layers": 2,
        "num_heads": 16, "mlp_ratio": 2.0, "dropout": 0.0,
        "time_scale": 25.0, "max_rel_sec": 2.5,
    }
    if "force" in modalities:
        cfg["model"]["force_head"] = {
            **cfg["model"]["force_head"], "enabled": True,
            "force_keypoint_indices": [62, 41, 13, 14], "frame": "root"}
    return cfg


def _build(modalities: list[str]):
    torch.manual_seed(0)
    cfg = _cfg(modalities)
    model, trainable = build_model(cfg, "cuda")
    model.eval()
    return model, cfg, trainable


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
    collate = make_collate(
        tuple(model.cfg.MODEL.IMAGE_SIZE), TargetSpec.from_config(cfg))
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
def _block_disabled(model):
    block = model.cross_modal_temporal
    del model.cross_modal_temporal      # the hook takes its getattr(..., None) path
    try:
        yield
    finally:
        model.cross_modal_temporal = block


def _randomize_gammas(model, seed=7):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for name, param in model.cross_modal_temporal.named_parameters():
            if "gamma" in name:
                param.copy_(torch.randn(param.shape, generator=gen, device="cuda"))


def _noise_floor(model, batch):
    """Base run-to-run nondeterminism per output key (block disabled)."""
    with _block_disabled(model):
        a = _outputs(model, batch)
        b = _outputs(model, batch)
    return a, {k: _max_abs(a[k], b[k]) for k in a}


def _assert_within_floor(enabled, disabled, floor, keys, tag):
    for key in keys:
        limit = _NOISE_MARGIN * floor[key] + _NOISE_FLOOR_EPS
        diff = _max_abs(enabled[key], disabled[key])
        assert diff <= limit, (
            f"[{tag}] output {key!r} moved {diff:.2e} > {limit:.2e} "
            f"(base noise {floor[key]:.2e}) — the block leaked into a frozen output")


# ------------------------------------------------------- (a) zero gate == disabled

def test_zero_gate_matches_disabled():
    model, cfg, trainable = _build(["contact", "force"])
    try:
        assert any("cross_modal_temporal" in n for n in trainable)
        batch = _batch(cfg, model, _synth_frames(8), seq_len=4)
        disabled, floor = _noise_floor(model, batch)
        enabled = _outputs(model, batch)                 # zero gammas at init
        # EVERY output — contact + pose/MHR — within the base noise floor.
        _assert_within_floor(enabled, disabled, floor, list(enabled), "zero-gate")
    finally:
        del model
        torch.cuda.empty_cache()


# --------------------------------------------------- (a') exact pose isolation

def test_block_params_have_no_pose_jacobian():
    """Exact (noise-free) proof: with ``pose`` NOT listed, every frozen pose/MHR
    output has a bitwise-zero Jacobian w.r.t. every block parameter."""
    model, cfg, _ = _build(["contact", "force"])
    try:
        _randomize_gammas(model)                         # nonzero gates -> block live
        batch = _batch(cfg, model, _synth_frames(8), seq_len=4)
        model._initialize_batch(batch)
        out = model.forward_step(batch, decoder_type="body")   # grad enabled
        params = [p for _, p in model.cross_modal_temporal.named_parameters()]

        # Sanity: the contact logits DO depend on the block (the graph is live).
        contact = out["contact"]["joint_logits"].float().sum()
        assert contact.requires_grad
        grads = torch.autograd.grad(contact, params, allow_unused=True,
                                    retain_graph=True)
        assert any(g is not None and float(g.abs().sum()) > 0 for g in grads), (
            "contact logits have no block gradient — test not exercising the module")

        checked = 0
        for key, val in out["mhr"].items():
            if not (torch.is_tensor(val) and val.is_floating_point()):
                continue
            checked += 1
            if not val.requires_grad:
                continue
            grads = torch.autograd.grad(val.float().sum(), params,
                                        allow_unused=True, retain_graph=True)
            for grad in grads:
                assert grad is None or float(grad.abs().sum()) == 0.0, (
                    f"MHR output {key!r} has a nonzero cross-modal Jacobian")
        assert checked > 0
    finally:
        del model
        torch.cuda.empty_cache()


# ------------------------------------------- (b) nonzero gate isolates pose

def test_nonzero_gate_isolates_pose():
    model, cfg, _ = _build(["contact", "force"])
    try:
        batch = _batch(cfg, model, _synth_frames(8), seq_len=4)
        disabled, floor = _noise_floor(model, batch)

        _randomize_gammas(model)
        enabled = _outputs(model, batch)

        pose_keys = [k for k in enabled if k != "__contact__"]
        _assert_within_floor(enabled, disabled, floor, pose_keys, "hot-gate")
        contact_diff = _max_abs(enabled["__contact__"], disabled["__contact__"])
        assert contact_diff > _CONTACT_SIGNAL, (
            f"non-zero gate barely changed contact ({contact_diff:.2e})")
        assert contact_diff > 100 * (floor["__contact__"] + _NOISE_FLOOR_EPS)
    finally:
        del model
        torch.cuda.empty_cache()


# ------------------------------------------- (b') pose listed -> pose MOVES

def test_pose_modality_moves_the_frozen_readout():
    """The deliberate exception: listing ``pose`` writes the pose token, so a hot
    block MUST move the recomputed final MHR output."""
    model, cfg, _ = _build(["pose", "contact"])
    try:
        batch = _batch(cfg, model, _synth_frames(8), seq_len=4)
        disabled, floor = _noise_floor(model, batch)
        _randomize_gammas(model)
        enabled = _outputs(model, batch)
        moved = _max_abs(enabled["body_pose"], disabled["body_pose"])
        assert moved > 100 * (floor["body_pose"] + _NOISE_FLOOR_EPS), moved
    finally:
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- (c) T=1 vs T=4

def test_zero_gate_t1_matches_t4():
    model, cfg, _ = _build(["contact", "force"])
    try:
        frames = _synth_frames(8)
        batch_t4 = _batch(cfg, model, frames, seq_len=4)
        _, floor = _noise_floor(model, batch_t4)

        out_t4 = _outputs(model, batch_t4)
        out_t1 = _outputs(model, _batch(cfg, model, frames, seq_len=1))
        # A zero-gate block is an exact identity and the frozen forward ignores
        # seq_len, so per-frame contact logits agree to within CUDA noise (not
        # bit-exact: two shapes -> different kernels + base nondeterminism).
        diff = _max_abs(out_t4["__contact__"], out_t1["__contact__"])
        assert torch.allclose(out_t4["__contact__"], out_t1["__contact__"],
                              rtol=1e-4, atol=1e-5), f"T1 vs T4 contact diff {diff:.2e}"
        assert diff <= _NOISE_MARGIN * floor["__contact__"] + 1e-5
    finally:
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------- (d) RoPE relative time

def test_uniform_time_shift_leaves_outputs_unchanged():
    """The RoPE property end-to-end: shifting every ``frame_pos_sec`` by a
    constant is not a different clip, so the contact logits must not move."""
    model, cfg, _ = _build(["contact", "force"])
    try:
        frames = _synth_frames(8)
        batch = _batch(cfg, model, frames, seq_len=4)
        _, floor = _noise_floor(model, batch)
        _randomize_gammas(model)
        base = _outputs(model, batch)

        shifted = _batch(cfg, model,
                         [{**f, "frame_pos_sec": f["frame_pos_sec"] + 4.25}
                          for f in frames], seq_len=4)
        out = _outputs(model, shifted)
        diff = _max_abs(out["__contact__"], base["__contact__"])
        assert diff <= _NOISE_MARGIN * floor["__contact__"] + 1e-4, diff

        # ...while a genuine fps change (scaled dt) DOES move them.
        scaled = _batch(cfg, model,
                        [{**f, "frame_pos_sec": f["frame_pos_sec"] * 4.0}
                         for f in frames], seq_len=4)
        out_scaled = _outputs(model, scaled)
        assert _max_abs(out_scaled["__contact__"], base["__contact__"]) > \
            100 * (floor["__contact__"] + _NOISE_FLOOR_EPS)
    finally:
        del model
        torch.cuda.empty_cache()
