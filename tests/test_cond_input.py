"""Input conditioning (``model.cond_input``): config matrix, wiring, feature math.

Fast CPU coverage for feeding the frozen model's own smoothed pelvis velocity /
acceleration into the contact and force tokens — the config accept/reject matrix,
``_patch_model_cfg`` plumbing, freeze-filter coverage of the new projections,
arch-signature stability (the new key must stay out of every signature written
before conditioning existed), and :func:`cond_feature_rows` against a hand
computation on the real artifact. The retired cond A/B ladder's configs are
gone; the kept production arm (`cond_sum1_postdec`) anchors the enabled-path
assertions and a minimal inline fixture covers the bare-linear pre_decoder
variant.

The init-equivalence proof (a conditioned build IS the unconditioned one at
initialisation, because both projections are zero-init) needs the real
checkpoint and lives in ``test_cond_input_invariance.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from yacs.config import CfgNode

from contact import checkpoint as ckpt_io
from contact.config import load_config
from contact.data.climbing_corpus import COND_FEATURE_DIM, cond_feature_rows

REPO = Path(__file__).resolve().parents[1]
_POSTDEC_CFG = REPO / "configs" / "climbing_corpus_joint_force_cond_sum1_postdec.yaml"
_T7HINGE_CFG = REPO / "configs" / "climbing_videos_force_warmstart_t7hinge.yaml"
_FEATURES = REPO / "output" / "motion_probe_geom" / "cond_features.npz"

requires_features = pytest.mark.skipif(
    not _FEATURES.is_file(), reason="cond_features.npz artifact not available")

_STD = """
    standardize:
      vel_mean: [0.0, 0.0, 0.0]
      vel_std: [1.0, 1.0, 1.0]
      acc_mean: [0.0, 0.0, 0.0]
      acc_std: [2.0, 2.0, 2.0]
"""

# Bare-linear pre_decoder conditioning on the kept force-only corpus config
# (no cond in the base, so the removal-based reject tests below stay honest).
_COND_ON = """
base: configs/climbing_corpus_force_supervised.yaml
model:
  cond_input:
    enabled: true
    features_path: output/motion_probe_geom/cond_features.npz
""" + _STD


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "run.yaml"
    p.write_text(text)
    return p


# ---------------------------------------------------------------- config matrix

def test_disabled_by_default():
    cfg = load_config(REPO / "configs" / "base.yaml")
    cond = cfg["model"]["cond_input"]
    assert cond["enabled"] is False
    assert cond["features_path"] is None
    assert cond["clip"] == 5.0
    assert cond["standardize"] == {
        "vel_mean": None, "vel_std": None, "acc_mean": None, "acc_std": None}


def test_enabled_config_accepted(tmp_path):
    cfg = load_config(_write(tmp_path, _COND_ON))
    assert cfg["model"]["cond_input"]["enabled"] is True
    assert cfg["model"]["cond_input"]["standardize"]["acc_std"] == [2.0, 2.0, 2.0]


def test_enabled_requires_features_path(tmp_path):
    text = _COND_ON.replace(
        "    features_path: output/motion_probe_geom/cond_features.npz\n", "")
    with pytest.raises(ValueError, match="features_path"):
        load_config(_write(tmp_path, text))


def test_enabled_requires_standardization_literals(tmp_path):
    text = _COND_ON.replace("      acc_std: [2.0, 2.0, 2.0]\n", "")
    with pytest.raises(ValueError, match="acc_std must be a finite 3-list"):
        load_config(_write(tmp_path, text))


def test_standardization_std_must_be_positive(tmp_path):
    text = _COND_ON.replace("      vel_std: [1.0, 1.0, 1.0]", "      vel_std: [1.0, 0.0, 1.0]")
    with pytest.raises(ValueError, match="vel_std entries must be positive"):
        load_config(_write(tmp_path, text))


def test_clip_must_be_positive(tmp_path):
    text = _COND_ON + "    clip: 0.0\n"
    with pytest.raises(ValueError, match="clip must be a finite positive number"):
        load_config(_write(tmp_path, text))


def test_enabled_requires_a_corpus_dataset(tmp_path):
    text = """
base: configs/climbing_videos_joint.yaml
data:
  eval_split: val
  datasets:
    - {name: damon, config: configs/datasets/damon.yaml}
model:
  cond_input:
    enabled: true
    features_path: output/motion_probe_geom/cond_features.npz
""" + _STD
    with pytest.raises(ValueError, match="requires a climbing_corpus dataset"):
        load_config(_write(tmp_path, text))


def test_enabled_requires_tokens_to_condition(tmp_path):
    text = """
base: configs/climbing_corpus_motion_pelvis_t7.yaml
model:
  cond_input:
    enabled: true
    features_path: output/motion_probe_geom/cond_features.npz
""" + _STD
    with pytest.raises(ValueError, match="no tokens to condition"):
        load_config(_write(tmp_path, text))


# ------------------------------------------------------------- pinned literals

def test_pinned_standardization_is_the_artifact_scale():
    """Sanity-bound the pinned literals (recomputing them needs the artifact)."""
    std = load_config(_POSTDEC_CFG)["model"]["cond_input"]["standardize"]
    assert all(0.3 < v < 0.5 for v in std["vel_std"])
    assert all(1.5 < v < 2.0 for v in std["acc_std"])
    assert all(abs(v) < 0.1 for v in std["vel_mean"] + std["acc_mean"])


# ---------------------------------------------------------------- model wiring

def _min_model_cfg() -> CfgNode:
    return CfgNode({"MODEL": {"DECODER": {}, "PROMPT_ENCODER": {}, "MHR_HEAD": {}}})


def test_patch_model_cfg_carries_the_switch(tmp_path):
    from contact.model import _patch_model_cfg

    model_cfg = _patch_model_cfg(
        _min_model_cfg(), load_config(_write(tmp_path, _COND_ON)), "mhr.pt")
    assert model_cfg.MODEL.COND_INPUT.ENABLED is True
    assert model_cfg.MODEL.COND_INPUT.FEAT_DIM == COND_FEATURE_DIM


def test_patch_model_cfg_disabled_is_still_self_describing():
    from contact.model import _patch_model_cfg

    model_cfg = _patch_model_cfg(
        _min_model_cfg(), load_config(REPO / "configs" / "base.yaml"), "mhr.pt")
    assert model_cfg.MODEL.COND_INPUT.ENABLED is False
    assert model_cfg.MODEL.COND_INPUT.FEAT_DIM == COND_FEATURE_DIM


def test_zero_init_projection_is_an_exact_no_op():
    """The injection arithmetic itself: a zero-init linear adds bitwise nothing.

    The full-model version of this (real checkpoint, real token embeddings) is
    ``test_cond_input_invariance.py::test_injection_is_bit_exact_at_init``.
    """
    import torch
    from torch import nn

    linear = nn.Linear(COND_FEATURE_DIM, 32)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    feat = torch.randn(7, COND_FEATURE_DIM) * 10.0
    delta = linear(feat)
    assert torch.equal(delta, torch.zeros_like(delta))
    emb = torch.randn(7, 6, 32)
    assert torch.equal(emb + delta.unsqueeze(1), emb)


def test_forked_construction_leaves_the_rng_stream_untouched():
    """The mechanism that keeps the A/B pair's shared init identical.

    ``nn.Linear`` consumes RNG in its default init before the zeros overwrite it,
    which would shift every module built afterwards. The model builds the two
    conditioning projections inside ``torch.random.fork_rng``; this is that
    contract in isolation (the whole-model version is
    ``test_cond_input_invariance.py::test_experiment_pair_shares_every_other_weight``).
    """
    import torch
    from torch import nn

    def stream(build_extra: bool) -> torch.Tensor:
        torch.manual_seed(42)
        nn.Linear(8, 8)                          # stands in for the shared modules
        if build_extra:
            with torch.random.fork_rng(devices=[]):
                extra = nn.Linear(COND_FEATURE_DIM, 32)
                nn.init.zeros_(extra.weight)
        return nn.Linear(8, 8).weight.detach()   # a module built after the hook

    assert torch.equal(stream(False), stream(True))


def test_cond_projections_pass_the_trainable_filter():
    from contact.model import _trainable_name_filter

    for name in ("contact_cond_linear.weight", "force_cond_linear.weight"):
        assert _trainable_name_filter(name), name


# ---------------------------------------------------------------- MLP encoder

def test_encoder_hidden_defaults_to_bare_linear(tmp_path):
    assert load_config(REPO / "configs" / "base.yaml")["model"]["cond_input"][
        "encoder_hidden"] is None
    assert load_config(_write(tmp_path, _COND_ON))["model"]["cond_input"][
        "encoder_hidden"] is None


@pytest.mark.parametrize("bad", ["0", "-4", "true", "2.5"])
def test_encoder_hidden_rejects_non_positive_ints(tmp_path, bad):
    text = _COND_ON + f"    encoder_hidden: {bad}\n"
    with pytest.raises(ValueError, match="encoder_hidden"):
        load_config(_write(tmp_path, text))


def test_patch_model_cfg_carries_encoder_hidden(tmp_path):
    from contact.model import _patch_model_cfg

    text = _COND_ON + "    encoder_hidden: 64\n"
    model_cfg = _patch_model_cfg(
        _min_model_cfg(), load_config(_write(tmp_path, text)), "mhr.pt")
    assert model_cfg.MODEL.COND_INPUT.ENCODER_HIDDEN == 64
    model_cfg = _patch_model_cfg(
        _min_model_cfg(), load_config(_write(tmp_path, _COND_ON)), "mhr.pt")
    assert model_cfg.MODEL.COND_INPUT.ENCODER_HIDDEN is None


def test_mlp_encoder_zero_output_layer_is_an_exact_no_op():
    """The MLP variant keeps the init contract: zero OUTPUT layer, exact zeros."""
    import torch
    from torch import nn

    out = nn.Linear(64, 32, bias=False)
    nn.init.zeros_(out.weight)
    mlp = nn.Sequential(nn.Linear(COND_FEATURE_DIM, 64), nn.GELU(), out)
    feat = torch.randn(7, COND_FEATURE_DIM) * 10.0
    delta = mlp(feat)
    assert torch.equal(delta, torch.zeros_like(delta))


def test_mlp_encoder_params_pass_filter_and_warmstart_exemption():
    from contact.model import _trainable_name_filter

    for name in ("contact_cond_linear.0.weight", "contact_cond_linear.0.bias",
                 "contact_cond_linear.2.weight", "force_cond_linear.0.weight",
                 "force_cond_linear.0.bias", "force_cond_linear.2.weight"):
        assert _trainable_name_filter(name), name
        # `initialize_common_contact` exempts missing params by this substring.
        assert "cond_linear" in name


# ---------------------------------------------------------------- signature stability

def test_arch_signature_omits_cond_key_when_disabled():
    # Checkpoints written before conditioning existed store signatures without
    # this key, and the comparison in `_check_schema` is an exact dict equality.
    assert "cond_input" not in ckpt_io._arch_signature(load_config(_T7HINGE_CFG))
    assert "cond_input" not in ckpt_io._arch_signature(
        load_config(REPO / "configs" / "base.yaml"))


def test_arch_signature_carries_cond_key_when_enabled():
    sig = ckpt_io._arch_signature(load_config(_POSTDEC_CFG))
    assert sig["cond_input"]["enabled"] is True
    assert sig["cond_input"]["clip"] == 5.0
    assert sig["cond_input"]["standardize"]["vel_std"] == [0.37997, 0.38611, 0.40752]


def test_arch_signature_separates_different_standardizations(tmp_path):
    other = load_config(_POSTDEC_CFG)
    other["model"]["cond_input"]["standardize"]["acc_std"] = [1.0, 1.0, 1.0]
    assert ckpt_io._arch_signature(other) != ckpt_io._arch_signature(
        load_config(_POSTDEC_CFG))


def test_arch_signature_encoder_key_only_when_set(tmp_path):
    # Bare-linear checkpoints (the retired A/B pair) keep byte-identical stored
    # signatures; an MLP-encoder run is a different architecture.
    bare = ckpt_io._arch_signature(load_config(_write(tmp_path, _COND_ON)))
    assert "encoder_hidden" not in bare["cond_input"]
    mlp = ckpt_io._arch_signature(load_config(_POSTDEC_CFG))
    assert mlp["cond_input"]["encoder_hidden"] == 64


# ---------------------------------------------------------------- feature math

_ROT_Z90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], np.float32)
_STANDARDIZE = {
    "vel_mean": [0.0, 0.0, 0.0], "vel_std": [1.0, 2.0, 4.0],
    "acc_mean": [1.0, 1.0, 1.0], "acc_std": [1.0, 1.0, 1.0],
}


def test_cond_feature_rows_layout_and_rotation():
    vel = np.array([[1.0, 2.0, 3.0]], np.float32)
    acc = np.array([[1.0, 1.0, 1.0]], np.float32)
    rot = _ROT_Z90[None]
    rows = cond_feature_rows(vel, acc, rot, np.array([True]), _STANDARDIZE, 5.0)
    assert rows.shape == (1, COND_FEATURE_DIM) and rows.dtype == np.float32
    # R^T v with R = Rz(90 deg): world x maps onto -root y, world y onto +root x.
    np.testing.assert_allclose(rows[0, :3], [2.0 / 1.0, -1.0 / 2.0, 3.0 / 4.0], atol=1e-6)
    np.testing.assert_allclose(rows[0, 3:6], [0.0, -2.0, 0.0], atol=1e-6)
    # Gravity [0, 1, 0] (world y down) expressed in root axes, unstandardized.
    np.testing.assert_allclose(rows[0, 6:9], [1.0, 0.0, 0.0], atol=1e-6)
    assert rows[0, 9] == 1.0


def test_cond_feature_rows_clips_and_zeroes_invalid():
    vel = np.array([[100.0, 0.0, 0.0], [1.0, 2.0, 3.0]], np.float32)
    acc = np.array([[-100.0, 0.0, 0.0], [1.0, 1.0, 1.0]], np.float32)
    rot = np.tile(np.eye(3, dtype=np.float32), (2, 1, 1))
    rows = cond_feature_rows(vel, acc, rot, np.array([True, False]), _STANDARDIZE, 5.0)
    assert rows[0, 0] == 5.0 and rows[0, 3] == -5.0
    np.testing.assert_array_equal(rows[1], np.zeros(COND_FEATURE_DIM, np.float32))


@requires_features
def test_cond_feature_rows_matches_hand_computation_on_a_real_entry():
    artifact = np.load(_FEATURES, allow_pickle=True)
    meta = json.loads(str(artifact["__meta__"]))
    name = meta["entries"][0]["name"]
    std = load_config(_POSTDEC_CFG)["model"]["cond_input"]["standardize"]

    rot = artifact[f"{name}#R_pred_world_from_root"]
    vel = artifact[f"{name}#vel_smooth_world"]
    acc = artifact[f"{name}#acc_smooth_world_alt"]        # sigma 0.12 s (better 3D r)
    valid = artifact[f"{name}#feat_valid"].astype(bool)
    rows = cond_feature_rows(vel, acc, rot, valid, std, 5.0)

    # Hand computation on one valid frame, written out component by component.
    t = int(np.flatnonzero(valid)[len(np.flatnonzero(valid)) // 2])
    r_t = np.asarray(rot[t], np.float64)
    for axis in range(3):
        v_root = float(sum(r_t[j, axis] * float(vel[t][j]) for j in range(3)))
        a_root = float(sum(r_t[j, axis] * float(acc[t][j]) for j in range(3)))
        g_root = float(r_t[1, axis])                     # gravity_world = [0, 1, 0]
        v_z = (v_root - std["vel_mean"][axis]) / std["vel_std"][axis]
        a_z = (a_root - std["acc_mean"][axis]) / std["acc_std"][axis]
        assert rows[t, axis] == pytest.approx(min(max(v_z, -5.0), 5.0), abs=1e-6)
        assert rows[t, 3 + axis] == pytest.approx(min(max(a_z, -5.0), 5.0), abs=1e-6)
        assert rows[t, 6 + axis] == pytest.approx(g_root, abs=1e-6)
    assert rows[t, 9] == 1.0
    # The gravity block is a unit direction on every valid row.
    norms = np.linalg.norm(rows[valid][:, 6:9], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
    # Clipping is the only nonlinearity: nothing leaves [-5, 5].
    assert float(np.abs(rows[:, :6]).max()) <= 5.0


@requires_features
def test_real_artifact_covers_every_corpus_scene_person():
    """The join key is ``"<scene>__p<object_id>"`` for every corpus track."""
    from contact.data.climbing_corpus import (
        DEFAULT_ROOT, list_annotated_test_scenes, list_corpus_scenes, scene_shard,
    )

    corpus = Path(DEFAULT_ROOT)
    if not (corpus / "scenes" / "scenes.db").is_file():
        pytest.skip("ClimbingVideos corpus not available")
    artifact = np.load(_FEATURES, allow_pickle=True)
    keys = set(artifact.files)
    scenes = list_corpus_scenes(corpus, "train") + list_annotated_test_scenes(corpus)
    missing = []
    for scene in scenes:
        contacts = np.load(
            corpus / "features" / "human_optim" / scene_shard(scene) / scene
            / "contacts_1.npz", allow_pickle=True)
        for oid in np.asarray(contacts["object_ids"]).reshape(-1):
            if f"{scene}__p{int(oid)}#frame_idx" not in keys:
                missing.append(f"{scene}__p{int(oid)}")
    assert missing == []


# ---------------------------------------------------------------- injection site

def test_injection_defaults_to_pre_decoder(tmp_path):
    assert load_config(REPO / "configs" / "base.yaml")["model"]["cond_input"][
        "injection"] == "pre_decoder"
    assert load_config(_write(tmp_path, _COND_ON))["model"]["cond_input"][
        "injection"] == "pre_decoder"


def test_injection_rejects_unknown_values(tmp_path):
    text = _COND_ON + "    injection: between_layers\n"
    with pytest.raises(ValueError, match="injection must be 'pre_decoder'"):
        load_config(_write(tmp_path, text))


def test_patch_model_cfg_carries_injection(tmp_path):
    from contact.model import _patch_model_cfg

    text = _COND_ON + "    injection: post_decoder\n"
    model_cfg = _patch_model_cfg(
        _min_model_cfg(), load_config(_write(tmp_path, text)), "mhr.pt")
    assert model_cfg.MODEL.COND_INPUT.INJECTION == "post_decoder"
    model_cfg = _patch_model_cfg(
        _min_model_cfg(), load_config(_write(tmp_path, _COND_ON)), "mhr.pt")
    assert model_cfg.MODEL.COND_INPUT.INJECTION == "pre_decoder"


def test_arch_signature_injection_key_only_when_post_decoder(tmp_path):
    # Every pre_decoder checkpoint written before the key existed keeps a
    # byte-identical stored signature.
    sig = ckpt_io._arch_signature(load_config(_write(tmp_path, _COND_ON)))
    assert "injection" not in sig["cond_input"]
    sig = ckpt_io._arch_signature(load_config(_POSTDEC_CFG))
    assert sig["cond_input"]["injection"] == "post_decoder"
