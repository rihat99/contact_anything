"""GPU tests: the conditioning projections start as an exact no-op and stay isolated.

Real SAM-3D-Body checkpoint required. Mirrors ``test_force_invariance.py``.

``model.cond_input`` adds one zero-init linear per token block whose output is
ADDED to the contact / force token stream — to the initial token embeddings
(``injection: pre_decoder``) or to the decoder's token outputs
(``injection: post_decoder``). Two properties matter and are proven here on the
real model:

1. **Init equivalence** — at initialisation both projections emit exactly zero, so
   a conditioned run is bit-identical to the unconditioned one no matter what the
   feature contains. The A/B pair therefore starts from the same point.
2. **Isolation** — the injection changes token *values* only (no mask edit, no new
   token), so the frozen MHR/pose outputs stay independent of the new params even
   once the projections are randomised: the asymmetric mask still stops every
   original token from attending the contact/force block.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from contact.config import load_config
from contact.data.climbing_corpus import COND_FEATURE_DIM
from contact.data.collate import batch_to_device, make_collate
from contact.engine import forward_model
from contact.model import build_model
from contact.targets import NUM_BODY_22, TargetSpec

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_POSTDEC_CFG = os.path.join(
    REPO, "configs", "old", "climbing_corpus_joint_force_cond_sum1_postdec.yaml")
_CKPT = load_config(os.path.join(REPO, "configs", "base.yaml"))["model"]["checkpoint_path"]

# The retired shipped A/B ladder is reconstructed from the kept production
# config: the bare-linear pre_decoder cond arm, and the same build with the
# conditioning stripped back to its defaults.
_COND_TEXT = """
base: configs/old/climbing_corpus_joint_force_cond_sum1_postdec.yaml
model:
  cond_input:
    encoder_hidden: null
    injection: pre_decoder
"""
_UNCOND_TEXT = """
base: configs/old/climbing_corpus_joint_force_cond_sum1_postdec.yaml
model:
  cond_input:
    enabled: false
    features_path: null
    encoder_hidden: null
    injection: pre_decoder
    standardize: {vel_mean: null, vel_std: null, acc_mean: null, acc_std: null}
"""

_NOISE_MARGIN = 8.0
_NOISE_FLOOR_EPS = 1e-6

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint missing"),
]

_SEQ_LEN = 7


@pytest.fixture(scope="module")
def cfg_pair(tmp_path_factory):
    """(unconditioned, bare-linear pre_decoder cond) config paths."""
    d = tmp_path_factory.mktemp("cond_pair")
    uncond = d / "uncond.yaml"
    uncond.write_text(_UNCOND_TEXT)
    cond = d / "cond.yaml"
    cond.write_text(_COND_TEXT)
    return str(uncond), str(cond)


@pytest.fixture(scope="module")
def built(cfg_pair):
    torch.manual_seed(0)
    cfg = load_config(cfg_pair[1])
    model, trainable = build_model(cfg, "cuda")
    model.eval()
    return model, cfg, trainable


def _frames(n: int):
    rng = np.random.RandomState(1234)
    frames = []
    for t in range(n):
        gt = torch.zeros(NUM_BODY_22)
        gt[t % NUM_BODY_22] = 1.0
        # A plausible standardized row: v/a inside the clip, unit gravity, bit 1.
        feat = np.zeros(COND_FEATURE_DIM, np.float32)
        feat[:6] = rng.uniform(-3.0, 3.0, 6).astype(np.float32)
        feat[6:9] = np.array([0.05, -0.75, -0.43], np.float32)
        feat[9] = 1.0
        frames.append({
            "image": (rng.rand(200, 160, 3) * 255).astype(np.uint8),
            "mask": (np.ones((200, 160), np.uint8) * 255),
            "bbox": np.array([10.0, 10.0, 150.0, 190.0], np.float32),
            "cam_int": (np.eye(3, dtype=np.float32) * 500.0),
            "joint_contact": gt,
            "joint_mask": torch.ones(NUM_BODY_22),
            "joint_supervised": torch.ones(NUM_BODY_22),
            "joint_confidence": torch.ones(NUM_BODY_22),
            "frame_pos_sec": t * 0.04,
            "frame_valid": True,
            "cond_feat": torch.from_numpy(feat),
        })
    return frames


def _batches(cfg, model):
    """One clip, twice: with the conditioning feature and with it zeroed.

    The two batches must be identical everywhere else — the images/bboxes are
    built once and only ``cond_feat`` differs, so any output movement is the
    feature's doing and nothing else's.
    """
    collate = make_collate(tuple(model.cfg.MODEL.IMAGE_SIZE), TargetSpec.from_config(cfg))
    batch = collate([_frames(_SEQ_LEN)])
    zeroed = dict(batch)
    zeroed["cond_feat"] = torch.zeros_like(batch["cond_feat"])
    assert float(batch["cond_feat"].abs().max()) > 0.0
    return batch_to_device(batch, "cuda"), batch_to_device(zeroed, "cuda")


def _outputs(model, batch):
    """Every float output as a flat dict: MHR, contact logits, forces."""
    out = forward_model(model, batch)
    res = {"__contact__": out["contact"]["joint_logits"].detach().float().clone(),
           "__force__": out["force"]["joint_forces"].detach().float().clone(),
           "__force_raw__": out["force"]["joint_forces_raw"].detach().float().clone()}
    for key, val in out["mhr"].items():
        if torch.is_tensor(val) and val.is_floating_point():
            res[key] = val.detach().float().clone()
    return res


def _max_abs(a, b):
    return float((a - b).abs().max())


def _randomize_cond_projections(model, seed=11):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for linear in (model.contact_cond_linear, model.force_cond_linear):
            linear.weight.copy_(
                torch.randn(linear.weight.shape, generator=gen, device="cuda") * 0.05)


def _zero_cond_projections(model):
    with torch.no_grad():
        for linear in (model.contact_cond_linear, model.force_cond_linear):
            linear.weight.zero_()


def _final_force_linear(model):
    return [m for m in model.head_force.proj.modules()
            if isinstance(m, torch.nn.Linear)][-1]


def _randomize_force_head_final(model, seed=7):
    """The force head's last linear is zero-init, which would mask every upstream
    effect on ``joint_forces``; give it real weights first."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    final = _final_force_linear(model)
    with torch.no_grad():
        final.weight.copy_(torch.randn(final.weight.shape, generator=gen, device="cuda"))
        final.bias.copy_(torch.randn(final.bias.shape, generator=gen, device="cuda"))


def _zero_force_head_final(model):
    final = _final_force_linear(model)
    with torch.no_grad():
        final.weight.zero_()
        final.bias.zero_()


# ---------------------------------------------------------------- build

def test_projections_exist_zero_init_and_train(built):
    model, _, trainable = built
    for name in ("contact_cond_linear", "force_cond_linear"):
        linear = getattr(model, name)
        assert linear.in_features == COND_FEATURE_DIM
        assert linear.out_features == model.cfg.MODEL.DECODER.DIM
        assert torch.equal(linear.weight, torch.zeros_like(linear.weight))
        # No bias: it would be a per-block constant trained even on all-zero
        # (invalid) feature rows, i.e. a difference channel between the A/B arms
        # that carries no motion information.
        assert linear.bias is None
        assert f"{name}.weight" in trainable
        assert dict(model.named_parameters())[f"{name}.weight"].requires_grad


def test_experiment_pair_shares_every_other_weight(cfg_pair):
    """Under one seed the two arms differ by the two zero projections and nothing else.

    The trainer seeds the global RNG from ``data.seed`` before building, and the
    conditioning projections are constructed inside ``fork_rng``, so the
    conditioned build must reproduce the baseline's random init parameter for
    parameter — the A/B pair's only difference is the conditioning itself.
    """
    base_cfg, cond_cfg = cfg_pair
    seed = load_config(base_cfg)["data"]["seed"]
    assert seed == load_config(cond_cfg)["data"]["seed"]

    def _trainable(cfg_path):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model, names = build_model(load_config(cfg_path), "cuda")
        params = dict(model.named_parameters())
        return {n: params[n].detach().clone() for n in names}

    base = _trainable(base_cfg)
    cond = _trainable(cond_cfg)
    assert set(cond) - set(base) == {
        "contact_cond_linear.weight", "force_cond_linear.weight"}
    assert set(base) - set(cond) == set()
    mismatched = [n for n in base if not torch.equal(base[n], cond[n])]
    assert mismatched == [], (
        f"{len(mismatched)}/{len(base)} shared trainable tensors differ: "
        f"{mismatched[:5]}")
    for name in set(cond) - set(base):
        assert torch.equal(cond[name], torch.zeros_like(cond[name])), name


# ---------------------------------------------------------------- init equivalence

def test_injection_is_bit_exact_at_init(built):
    """The added term is EXACTLY zero, so the conditioned token embeddings are the
    unconditioned ones bit for bit — whatever the feature contains."""
    model, _, _ = built
    _zero_cond_projections(model)
    feat = torch.randn(_SEQ_LEN, COND_FEATURE_DIM, device="cuda") * 10.0
    for name, embedding in (("contact_cond_linear", model.contact_embedding),
                            ("force_cond_linear", model.force_embedding)):
        delta = getattr(model, name)(feat)
        assert torch.equal(delta, torch.zeros_like(delta)), name
        emb = embedding.weight[None, :, :].repeat(_SEQ_LEN, 1, 1)
        assert torch.equal(emb + delta.unsqueeze(1), emb), name


def test_conditioned_model_is_the_baseline_at_init(built):
    """Zero-init projection => the whole forward is unchanged with and without features.

    This is the A/B pair's starting point: the conditioned run and its matched
    baseline begin from the same model. The bound is the model's own repeat-run
    floor (the compiled bf16 backbone is not bitwise reproducible; ~1e-7 on MHR
    outputs, ~1e-4 on the projected 2D keypoints), measured here on two identical
    forwards, exactly as ``test_force_invariance.py`` does.
    """
    model, cfg, _ = built
    batch, zero_batch = _batches(cfg, model)
    _zero_cond_projections(model)
    _randomize_force_head_final(model)          # so joint_forces is not trivially 0
    try:
        with_cond = _outputs(model, batch)
        floor = {key: _max_abs(with_cond[key], _outputs(model, batch)[key])
                 for key in with_cond}
        without = _outputs(model, zero_batch)
        assert float(with_cond["__force_raw__"].abs().max()) > 0.0
        for key in with_cond:
            limit = _NOISE_MARGIN * floor[key] + _NOISE_FLOOR_EPS
            diff = _max_abs(with_cond[key], without[key])
            assert diff <= limit, f"{key} moved {diff:.2e} > {limit:.2e} at init"
    finally:
        _zero_force_head_final(model)


# ---------------------------------------------------------------- reach + isolation

def test_projected_features_reach_contact_and_force(built):
    """Once the projections are non-zero the features must actually change the heads."""
    model, cfg, _ = built
    batch, zero_batch = _batches(cfg, model)
    _zero_cond_projections(model)
    _randomize_force_head_final(model)
    try:
        base = _outputs(model, batch)
        _randomize_cond_projections(model)
        moved = _outputs(model, batch)
        # Bias-free, so a zero feature row projects to exactly zero; the
        # zero-feature batch is compared anyway to keep the assertion honest
        # about WHICH input moved the head.
        zeroed = _outputs(model, zero_batch)
        for key in ("__contact__", "__force_raw__"):
            assert _max_abs(moved[key], base[key]) > 1e-4, key
            assert _max_abs(moved[key], zeroed[key]) > 1e-5, f"{key} ignores the feature"
    finally:
        _zero_cond_projections(model)
        _zero_force_head_final(model)


def test_frozen_mhr_is_unaffected_by_the_projections(built):
    """The mask invariant survives: no MHR output depends on a conditioning param."""
    model, cfg, _ = built
    batch, _ = _batches(cfg, model)
    _zero_cond_projections(model)
    base = _outputs(model, batch)
    floor = {key: _max_abs(base[key], _outputs(model, batch)[key]) for key in base}
    try:
        _randomize_cond_projections(model)
        moved = _outputs(model, batch)
        mhr_keys = [k for k in base if not k.startswith("__")]
        assert mhr_keys, "no MHR outputs captured"
        for key in mhr_keys:
            limit = _NOISE_MARGIN * floor[key] + _NOISE_FLOOR_EPS
            diff = _max_abs(moved[key], base[key])
            assert diff <= limit, f"MHR output {key!r} moved {diff:.2e} > {limit:.2e}"
    finally:
        _zero_cond_projections(model)


# ---------------------------------------------------------------- MLP encoder
# (the kept postdec production config IS the MLP-encoder arm)


@pytest.fixture(scope="module")
def built_mlp():
    torch.manual_seed(0)
    cfg = load_config(_POSTDEC_CFG)
    model, trainable = build_model(cfg, "cuda")
    model.eval()
    return model, cfg, trainable


def _cond_output_linear(module):
    """Last Linear of a cond projection — the zero-init output layer."""
    return [m for m in module.modules() if isinstance(m, torch.nn.Linear)][-1]


def test_mlp_encoder_builds_zero_output_and_trains(built_mlp):
    model, _, trainable = built_mlp
    for name in ("contact_cond_linear", "force_cond_linear"):
        module = getattr(model, name)
        assert isinstance(module, torch.nn.Sequential), name
        hidden, out = module[0], _cond_output_linear(module)
        assert hidden.in_features == COND_FEATURE_DIM and hidden.out_features == 64
        assert out.out_features == model.cfg.MODEL.DECODER.DIM
        assert torch.equal(out.weight, torch.zeros_like(out.weight))
        assert out.bias is None
        # The hidden layer keeps its real default init (only the OUTPUT is gated).
        assert float(hidden.weight.detach().abs().max()) > 0.0
        for pname in (f"{name}.0.weight", f"{name}.0.bias", f"{name}.2.weight"):
            assert pname in trainable, pname
            assert dict(model.named_parameters())[pname].requires_grad, pname


def test_mlp_conditioned_model_is_the_baseline_at_init(built_mlp):
    """Zero OUTPUT layer => the MLP emits exact zeros, whatever the feature."""
    model, cfg, _ = built_mlp
    batch, zero_batch = _batches(cfg, model)
    _randomize_force_head_final(model)
    try:
        feat = torch.randn(_SEQ_LEN, COND_FEATURE_DIM, device="cuda") * 10.0
        for name in ("contact_cond_linear", "force_cond_linear"):
            delta = getattr(model, name)(feat)
            assert torch.equal(delta, torch.zeros_like(delta)), name
        with_cond = _outputs(model, batch)
        floor = {key: _max_abs(with_cond[key], _outputs(model, batch)[key])
                 for key in with_cond}
        without = _outputs(model, zero_batch)
        for key in with_cond:
            limit = _NOISE_MARGIN * floor[key] + _NOISE_FLOOR_EPS
            diff = _max_abs(with_cond[key], without[key])
            assert diff <= limit, f"{key} moved {diff:.2e} > {limit:.2e} at init"
    finally:
        _zero_force_head_final(model)


def test_mlp_reaches_heads_but_not_frozen_mhr(built_mlp):
    """Randomized output layers move contact/force yet leave every MHR output
    at the repeat-run noise floor (the mask invariant, MLP edition)."""
    model, cfg, _ = built_mlp
    batch, zero_batch = _batches(cfg, model)
    _randomize_force_head_final(model)
    outs = [_cond_output_linear(getattr(model, n))
            for n in ("contact_cond_linear", "force_cond_linear")]
    try:
        base = _outputs(model, batch)
        floor = {key: _max_abs(base[key], _outputs(model, batch)[key]) for key in base}
        gen = torch.Generator(device="cuda").manual_seed(11)
        with torch.no_grad():
            for out in outs:
                out.weight.copy_(
                    torch.randn(out.weight.shape, generator=gen, device="cuda") * 0.05)
        moved = _outputs(model, batch)
        zeroed = _outputs(model, zero_batch)
        for key in ("__contact__", "__force_raw__"):
            assert _max_abs(moved[key], base[key]) > 1e-4, key
            # An MLP has a bias path (GELU(b1) through the output layer), so a
            # zero feature row no longer projects to exactly zero — but the
            # FEATURE must still be what separates these two batches.
            assert _max_abs(moved[key], zeroed[key]) > 1e-5, f"{key} ignores the feature"
        for key in [k for k in base if not k.startswith("__")]:
            limit = _NOISE_MARGIN * floor[key] + _NOISE_FLOOR_EPS
            diff = _max_abs(moved[key], base[key])
            assert diff <= limit, f"MHR output {key!r} moved {diff:.2e} > {limit:.2e}"
    finally:
        with torch.no_grad():
            for out in outs:
                out.weight.zero_()
        _zero_force_head_final(model)


def test_mlp_hidden_layer_gets_gradient_once_output_is_nonzero(built_mlp):
    """The staged-unlock property: zero output layer => zero (but present) hidden
    grads; a non-zero output layer opens the path to the hidden layer."""
    model, cfg, _ = built_mlp
    batch, _ = _batches(cfg, model)
    _randomize_force_head_final(model)
    out = _cond_output_linear(model.force_cond_linear)
    try:
        model.zero_grad(set_to_none=True)
        forward_model(model, batch)["force"]["joint_forces_raw"].square().sum().backward()
        hidden = model.force_cond_linear[0]
        assert out.weight.grad is not None
        assert float(out.weight.grad.abs().max()) > 0.0
        assert hidden.weight.grad is not None
        assert float(hidden.weight.grad.abs().max()) == 0.0   # gated by the zero output

        model.zero_grad(set_to_none=True)
        gen = torch.Generator(device="cuda").manual_seed(13)
        with torch.no_grad():
            out.weight.copy_(
                torch.randn(out.weight.shape, generator=gen, device="cuda") * 0.05)
        forward_model(model, batch)["force"]["joint_forces_raw"].square().sum().backward()
        assert float(hidden.weight.grad.abs().max()) > 0.0
    finally:
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            out.weight.zero_()
        _zero_force_head_final(model)


def test_conditioning_params_have_gradients_from_the_force_output(built):
    """Grad flow: the force output must depend on the conditioning projections."""
    model, cfg, _ = built
    batch, _ = _batches(cfg, model)
    _zero_cond_projections(model)
    # The force head's last linear is zero-init, which would zero the gradient of
    # ANY upstream parameter; give it real weights so the path is exercised.
    _randomize_force_head_final(model)
    try:
        model.zero_grad(set_to_none=True)
        out = forward_model(model, batch)
        out["force"]["joint_forces_raw"].square().sum().backward()
        for name in ("force_cond_linear", "contact_cond_linear"):
            grad = getattr(model, name).weight.grad
            assert grad is not None, name
        assert float(model.force_cond_linear.weight.grad.abs().max()) > 0.0
    finally:
        model.zero_grad(set_to_none=True)
        _zero_force_head_final(model)


# ---------------------------------------------------------------- post_decoder injection

@pytest.fixture(scope="module")
def built_postdec():
    torch.manual_seed(0)
    cfg = load_config(_POSTDEC_CFG)
    model, trainable = build_model(cfg, "cuda")
    model.eval()
    return model, cfg, trainable


def test_postdec_builds_zero_init_and_trains(built_postdec):
    model, _, trainable = built_postdec
    assert model.cond_input_injection == "post_decoder"
    for name in ("contact_cond_linear", "force_cond_linear"):
        out = _cond_output_linear(getattr(model, name))
        assert torch.equal(out.weight, torch.zeros_like(out.weight)), name
        assert out.bias is None
        assert f"{name}.2.weight" in trainable


def test_postdec_conditioned_model_is_the_baseline_at_init(built_postdec):
    """Zero OUTPUT layer => the late injection adds exact zeros at init."""
    model, cfg, _ = built_postdec
    batch, zero_batch = _batches(cfg, model)
    _randomize_force_head_final(model)
    try:
        with_cond = _outputs(model, batch)
        floor = {key: _max_abs(with_cond[key], _outputs(model, batch)[key])
                 for key in with_cond}
        without = _outputs(model, zero_batch)
        for key in with_cond:
            limit = _NOISE_MARGIN * floor[key] + _NOISE_FLOOR_EPS
            diff = _max_abs(with_cond[key], without[key])
            assert diff <= limit, f"{key} moved {diff:.2e} > {limit:.2e} at init"
    finally:
        _zero_force_head_final(model)


def test_postdec_reaches_heads_but_not_frozen_mhr(built_postdec):
    """Randomized output layers move contact/force yet leave every MHR output at
    the repeat-run noise floor. Post-decoder edition: the injection sits entirely
    outside the decoder graph, after the pose/MHR readout."""
    model, cfg, _ = built_postdec
    batch, zero_batch = _batches(cfg, model)
    _randomize_force_head_final(model)
    outs = [_cond_output_linear(getattr(model, n))
            for n in ("contact_cond_linear", "force_cond_linear")]
    try:
        base = _outputs(model, batch)
        floor = {key: _max_abs(base[key], _outputs(model, batch)[key]) for key in base}
        gen = torch.Generator(device="cuda").manual_seed(11)
        with torch.no_grad():
            for out in outs:
                out.weight.copy_(
                    torch.randn(out.weight.shape, generator=gen, device="cuda") * 0.05)
        moved = _outputs(model, batch)
        zeroed = _outputs(model, zero_batch)
        for key in ("__contact__", "__force_raw__"):
            assert _max_abs(moved[key], base[key]) > 1e-4, key
            # The MLP bias path makes a zero row project to a non-zero constant,
            # but the FEATURE must still be what separates these two batches.
            assert _max_abs(moved[key], zeroed[key]) > 1e-5, f"{key} ignores the feature"
        for key in [k for k in base if not k.startswith("__")]:
            limit = _NOISE_MARGIN * floor[key] + _NOISE_FLOOR_EPS
            diff = _max_abs(moved[key], base[key])
            assert diff <= limit, f"MHR output {key!r} moved {diff:.2e} > {limit:.2e}"
    finally:
        with torch.no_grad():
            for out in outs:
                out.weight.zero_()
        _zero_force_head_final(model)


def test_postdec_cond_params_get_gradients_from_both_heads(built_postdec):
    """Grad flow through the late injection: force AND contact outputs must
    depend on their cond projections."""
    model, cfg, _ = built_postdec
    batch, _ = _batches(cfg, model)
    _randomize_force_head_final(model)
    try:
        model.zero_grad(set_to_none=True)
        out = forward_model(model, batch)
        (out["force"]["joint_forces_raw"].square().sum()
         + out["contact"]["joint_logits"].square().sum()).backward()
        for name in ("force_cond_linear", "contact_cond_linear"):
            grad = _cond_output_linear(getattr(model, name)).weight.grad
            assert grad is not None, name
            assert float(grad.abs().max()) > 0.0, name
    finally:
        model.zero_grad(set_to_none=True)
        _zero_force_head_final(model)
