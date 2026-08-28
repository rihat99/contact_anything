"""Decoupled force anchors + force-only builds (no contact branch).

Fast CPU coverage: the config accept/reject matrix for
``model.force_head.force_keypoint_indices``, the block-triangular token
attention mask in all three modes (contact-only / contact+force / force-only),
force-head sizing from its own anchor list, the ``_patch_model_cfg`` wiring,
and stability of the checkpoint arch signature (the new config key must not
leak into stored signatures — the t7hinge checkpoint stays loadable).

Slow GPU coverage (real SAM-3D-Body checkpoint): force-only ``build_model``
semantics (trainable set, absent contact modules, ``out["contact"] is None``,
eval pinning), the MHR noise-floor invariance of the force-only branch, and
loading the shipped t7hinge checkpoint through the strict identity machinery.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from yacs.config import CfgNode

from contact import checkpoint as ckpt_io
from contact.config import load_config
from sam_3d_body.models.heads import build_head
from sam_3d_body.models.meta_arch.sam3d_body import SAM3DBody

REPO = Path(__file__).resolve().parents[1]
_CKPT = load_config(REPO / "configs" / "base.yaml")["model"]["checkpoint_path"]
_T7HINGE_CFG = REPO / "configs" / "old" / "climbing_videos_force_warmstart_t7hinge.yaml"
_T7HINGE_BEST = REPO / "output" / "climb4_force_t7hinge_20260724_121450" / "best.pth"

# Six-anchor list of the supervised force experiment, kindyn column order
# [LH, RH, LF(toe), RF(toe), LA(heel), RA(heel)].
_SIX_ANCHORS = [62, 41, 15, 18, 17, 20]


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "run.yaml"
    p.write_text(text)
    return p


_FORCE_ONLY = """
base: configs/base.yaml
model:
  force_head:
    enabled: true
    force_keypoint_indices: [62, 41, 15, 18, 17, 20]
  force_temporal: {enabled: true}
contact:
  targets:
    vertex: {enabled: false}
    joint: {enabled: false}
"""

_COEXIST = """
base: configs/base.yaml
model:
  contact_head: {contact_keypoint_indices: [62, 41, 13, 14], num_global_tokens: 0,
                 pool_mode: per_token}
  force_head:
    enabled: true
    force_keypoint_indices: [62, 41, 15, 18, 17, 20]
contact:
  primary_target: joint
  targets:
    vertex: {enabled: false}
    joint: {enabled: true, joint_set: extremities_4, supervise_subset: null}
"""


# ---------------------------------------------------------------- config matrix

def test_force_only_with_explicit_anchors_accepted(tmp_path):
    cfg = load_config(_write(tmp_path, _FORCE_ONLY))
    assert cfg["model"]["force_head"]["force_keypoint_indices"] == _SIX_ANCHORS
    assert cfg["contact"]["targets"]["vertex"]["enabled"] is False
    assert cfg["contact"]["targets"]["joint"]["enabled"] is False
    assert cfg["model"]["force_temporal"]["enabled"] is True


def test_force_only_without_anchors_rejected(tmp_path):
    # Null anchors inherit from the contact tokens, which a force-only build
    # does not create — reject instead of silently defaulting to 21 tokens.
    with pytest.raises(ValueError, match="force_keypoint_indices"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
model:
  force_head: {enabled: true}
contact:
  targets:
    vertex: {enabled: false}
    joint: {enabled: false}
"""))


def test_force_only_without_force_head_rejected(tmp_path):
    with pytest.raises(ValueError, match="no contact target is enabled"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
contact:
  targets:
    vertex: {enabled: false}
    joint: {enabled: false}
"""))


def test_contact_plus_force_null_anchors_stays_legacy(tmp_path):
    cfg = load_config(_write(tmp_path, """
base: configs/base.yaml
model:
  contact_head: {contact_keypoint_indices: [62, 41, 13, 14], num_global_tokens: 0,
                 pool_mode: per_token}
  force_head: {enabled: true}
contact:
  primary_target: joint
  targets:
    vertex: {enabled: false}
    joint: {enabled: true, joint_set: extremities_4, supervise_subset: null}
"""))
    assert cfg["model"]["force_head"]["force_keypoint_indices"] is None


def test_contact_four_anchors_coexist_with_six_force_anchors(tmp_path):
    # Supported: the anchors are fully decoupled and the asymmetric mask keeps
    # contact independent of force regardless of the two lists.
    cfg = load_config(_write(tmp_path, _COEXIST))
    assert cfg["model"]["contact_head"]["contact_keypoint_indices"] == [62, 41, 13, 14]
    assert cfg["model"]["force_head"]["force_keypoint_indices"] == _SIX_ANCHORS


def test_explicit_force_anchors_with_physics_rejected(tmp_path):
    # The physics loss gates on the four extremity contact probabilities, which
    # is only sound when the force anchors inherit the contact anchors.
    with pytest.raises(ValueError, match=r"physics.enabled requires .*force_keypoint_indices=null"):
        load_config(_write(tmp_path, _COEXIST + """
physics: {enabled: true}
data:
  datasets:
    - {name: climbing_corpus, config: configs/datasets/climbing_corpus.yaml}
  sequence: {frames_per_clip: 8}
"""))


@pytest.mark.parametrize("bad", ["[]", "[70]", "[-1, 62]", "[62.5]", "[true]", "62"])
def test_bad_force_anchor_values_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="force_keypoint_indices must be null or a non-empty"):
        load_config(_write(tmp_path, f"""
base: configs/base.yaml
model:
  force_head:
    enabled: true
    force_keypoint_indices: {bad}
contact:
  targets:
    vertex: {{enabled: false}}
    joint: {{enabled: false}}
"""))


def test_force_only_rejects_contact_temporal(tmp_path):
    # The contact temporal module attends contact tokens, which do not exist.
    with pytest.raises(ValueError, match="model.temporal.enabled requires an enabled contact target"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
model:
  temporal: {enabled: true}
  force_head:
    enabled: true
    force_keypoint_indices: [62, 41, 15, 18, 17, 20]
contact:
  targets:
    vertex: {enabled: false}
    joint: {enabled: false}
"""))


def test_force_only_rejects_freeze_contact(tmp_path):
    with pytest.raises(ValueError, match="freeze_contact.*no contact branch"):
        load_config(_write(tmp_path, _FORCE_ONLY + "train: {freeze_contact: true}\n"))


def test_legacy_configs_still_validate():
    # The relaxations must not loosen anything for contact-enabled configs.
    cfg = load_config(_T7HINGE_CFG)
    assert cfg["train"]["freeze_contact"] is True
    assert cfg["model"]["force_head"]["enabled"] is True
    assert cfg["model"]["force_head"]["force_keypoint_indices"] is None


# ---------------------------------------------------------------- attention mask

def _assert_block_pattern(mask: torch.Tensor, boundaries: list[int]) -> None:
    """Assert the block-triangular pattern: block i attends block j iff j <= i."""
    assert mask.dtype == torch.bool
    num_total = mask.shape[-1]
    starts = [0] + boundaries
    ends = boundaries + [num_total]
    for qi, (qs, qe) in enumerate(zip(starts, ends)):
        for ki, (ks, ke) in enumerate(zip(starts, ends)):
            block = mask[:, qs:qe, ks:ke]
            if ki <= qi:
                assert bool(block.all()), f"block {qi}->{ki} must be fully allowed"
            else:
                assert bool((~block).all()), f"block {qi}->{ki} must be fully barred"


def test_token_mask_contact_only():
    mask = SAM3DBody._build_block_token_mask(2, 10, [6], torch.device("cpu"))
    assert mask.shape == (2, 10, 10)
    _assert_block_pattern(mask, [6])


def test_token_mask_contact_plus_force():
    mask = SAM3DBody._build_block_token_mask(3, 12, [5, 9], torch.device("cpu"))
    _assert_block_pattern(mask, [5, 9])


def test_token_mask_force_only():
    mask = SAM3DBody._build_block_token_mask(1, 9, [7], torch.device("cpu"))
    _assert_block_pattern(mask, [7])
    # Force tokens attend everything; original tokens never attend force tokens.
    assert bool(mask[0, 7:, :].all())
    assert not bool(mask[0, :7, 7:].any())


def test_token_mask_matches_legacy_inline_construction():
    # Bit-identical to the pre-refactor inline code for the contact+force case.
    contact_start, force_start, num_total = 5, 9, 12
    legacy = torch.ones(2, num_total, num_total, dtype=torch.bool)
    legacy[:, :contact_start, contact_start:] = False
    legacy[:, :force_start, force_start:] = False
    built = SAM3DBody._build_block_token_mask(
        2, num_total, [contact_start, force_start], torch.device("cpu"))
    assert torch.equal(built, legacy)


# ---------------------------------------------------------------- force head sizing

def _force_head_cfg(force_kp, num_contacts: int = 4) -> CfgNode:
    return CfgNode({"MODEL": {
        "DECODER": {"DIM": 16},
        "CONTACT_HEAD": {"NUM_CONTACTS": num_contacts},
        "FORCE_HEAD": {"KEYPOINT_INDICES": force_kp, "MLP_DEPTH": 2,
                       "MLP_CHANNEL_DIV_FACTOR": 2, "DROPOUT": 0.0},
    }})


def test_force_head_sized_by_own_anchor_list():
    head = build_head(_force_head_cfg(_SIX_ANCHORS), "force")
    assert head.num_force_tokens == 6
    assert head(torch.randn(2, 6, 16)).shape == (2, 6, 3)


def test_force_head_falls_back_to_contact_token_count():
    head = build_head(_force_head_cfg(None, num_contacts=4), "force")
    assert head.num_force_tokens == 4


# ---------------------------------------------------------------- model-config patching

def _min_model_cfg() -> CfgNode:
    return CfgNode({"MODEL": {"DECODER": {}, "PROMPT_ENCODER": {}, "MHR_HEAD": {}}})


def test_patch_model_cfg_force_only_disables_contact_tokens(tmp_path):
    from contact.model import _patch_model_cfg

    cfg = load_config(_write(tmp_path, _FORCE_ONLY))
    model_cfg = _patch_model_cfg(_min_model_cfg(), cfg, "mhr.pt")
    assert model_cfg.MODEL.DECODER.DO_CONTACT_TOKENS is False
    assert model_cfg.MODEL.DECODER.DO_FORCE_TOKENS is True
    assert model_cfg.MODEL.FORCE_HEAD.KEYPOINT_INDICES == _SIX_ANCHORS
    assert len(model_cfg.MODEL.CONTACT_HEAD.TARGETS) == 0


def test_patch_model_cfg_legacy_force_inherits_contact_anchors(tmp_path):
    from contact.model import _patch_model_cfg

    cfg = load_config(_write(tmp_path, """
base: configs/base.yaml
model:
  contact_head: {contact_keypoint_indices: [62, 41, 13, 14], num_global_tokens: 0,
                 pool_mode: per_token}
  force_head: {enabled: true}
contact:
  primary_target: joint
  targets:
    vertex: {enabled: false}
    joint: {enabled: true, joint_set: extremities_4, supervise_subset: null}
"""))
    model_cfg = _patch_model_cfg(_min_model_cfg(), cfg, "mhr.pt")
    assert model_cfg.MODEL.DECODER.DO_CONTACT_TOKENS is True
    assert model_cfg.MODEL.DECODER.DO_FORCE_TOKENS is True
    assert model_cfg.MODEL.FORCE_HEAD.KEYPOINT_INDICES is None
    assert model_cfg.MODEL.CONTACT_HEAD.KEYPOINT_INDICES == [62, 41, 13, 14]


# ---------------------------------------------------------------- signature stability

def test_arch_signature_has_no_force_anchor_key():
    # The new config key must not leak into the semantic signature: stored
    # t7hinge signatures predate it, and any new key inside `force` would make
    # their exact-dict comparison spuriously mismatch on load.
    sig = ckpt_io._arch_signature(load_config(_T7HINGE_CFG))
    assert sig["force"] == {
        "enabled": True,
        "frame": "local_world_aligned",
        "mlp_depth": 2,
        "mlp_channel_div_factor": 4,
        "dropout": 0.0,
    }


@pytest.mark.skipif(not _T7HINGE_BEST.exists(), reason="t7hinge checkpoint missing")
def test_t7hinge_stored_signature_unchanged():
    ckpt = torch.load(_T7HINGE_BEST, map_location="cpu", weights_only=False)
    assert ckpt_io._arch_signature(load_config(_T7HINGE_CFG)) == ckpt["arch_signature"]


# ---------------------------------------------------------------- slow GPU integration

_slow_gpu = pytest.mark.slow


def _skip_unless_gpu_ckpt():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if not os.path.exists(_CKPT):
        pytest.skip("checkpoint missing")


_NOISE_MARGIN = 8.0
_NOISE_FLOOR_EPS = 1e-6


def _build_force_only(tmp_path):
    from contact.model import build_model

    torch.manual_seed(0)
    cfg = load_config(_write(tmp_path, _FORCE_ONLY))
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


def _batch(cfg, model, frames):
    from contact.data.collate import batch_to_device, make_collate
    from contact.targets import TargetSpec

    spec = TargetSpec.from_config(cfg)
    collate = make_collate(tuple(model.cfg.MODEL.IMAGE_SIZE), spec)
    return batch_to_device(collate(list(frames)), "cuda")


def _mhr_outputs(model, batch):
    from contact.engine import forward_model

    out = forward_model(model, batch)
    return {
        key: val.detach().float().clone()
        for key, val in out["mhr"].items()
        if torch.is_tensor(val) and val.is_floating_point()
    }


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
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


@_slow_gpu
def test_force_only_build_forward_and_trainable_set(tmp_path):
    _skip_unless_gpu_ckpt()
    from contact.engine import forward_model

    model, trainable, cfg = _build_force_only(tmp_path)
    try:
        # Trainable set = exactly the force branch; no contact module exists.
        assert trainable and all("force" in name.lower() for name in trainable)
        assert not any(
            "contact" in name.lower() for name, _ in model.named_parameters())
        assert not hasattr(model, "head_contact")
        assert not hasattr(model, "contact_embedding")
        assert model.num_force_tokens == len(_SIX_ANCHORS)
        assert model.force_keypoint_indices == _SIX_ANCHORS

        batch = _batch(cfg, model, _synth_frames(2))
        out = forward_model(model, batch)
        assert out["contact"] is None
        assert out["force"]["joint_forces"].shape == (2, len(_SIX_ANCHORS), 3)

        # pin_frozen_eval with no contact modules: train(True) flips only the
        # force branch; the frozen backbone stays eval-pinned.
        model.train(True)
        assert not model.backbone.training
        assert model.head_force.training
        assert model.force_temporal.training
    finally:
        del model
        torch.cuda.empty_cache()


@_slow_gpu
def test_force_only_mhr_within_noise_floor(tmp_path):
    _skip_unless_gpu_ckpt()
    model, _, cfg = _build_force_only(tmp_path)
    try:
        batch = _batch(cfg, model, _synth_frames(2))
        with _force_disabled(model):
            base = _mhr_outputs(model, batch)
            rerun = _mhr_outputs(model, batch)
        floor = {key: _max_abs(base[key], rerun[key]) for key in base}
        live = _mhr_outputs(model, batch)
        for key in base:
            limit = _NOISE_MARGIN * floor[key] + _NOISE_FLOOR_EPS
            diff = _max_abs(live[key], base[key])
            assert diff <= limit, (
                f"MHR output {key!r} moved {diff:.2e} > {limit:.2e} "
                f"(base noise {floor[key]:.2e}) — force-only branch leaked into MHR")
    finally:
        del model
        torch.cuda.empty_cache()


@_slow_gpu
def test_t7hinge_checkpoint_loads_into_current_build():
    _skip_unless_gpu_ckpt()
    if not _T7HINGE_BEST.exists():
        pytest.skip("t7hinge checkpoint missing")
    from contact.model import build_model

    cfg = load_config(_T7HINGE_CFG)
    torch.manual_seed(0)
    model, _ = build_model(cfg, "cuda")
    try:
        # Hard-fail identity machinery: fingerprint + signature + weights.
        ckpt = ckpt_io.load(_T7HINGE_BEST, model, config=cfg, map_location="cuda")
        assert ckpt["schema_version"] == ckpt_io.SCHEMA_VERSION
    finally:
        del model
        torch.cuda.empty_cache()
