"""Pose-temporal module (E2): config matrix, freeze-filter exception, signature,
and GPU invariance semantics.

The pose-temporal module is the ONE deliberate exception to the frozen-pose
rule: zero-init gates keep init behavior exactly frozen (noise-floor asserted),
live gammas move the MHR outputs (that is the point), and only params named
``pose_temporal`` train. Fast tests cover config/signature/filter; ``-m slow``
covers the GPU semantics with the real checkpoint.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from contact import checkpoint as ckpt_io
from contact.config import load_config
from contact.model import _trainable_name_filter

REPO = Path(__file__).resolve().parents[1]
_POSE_CFG = REPO / "tests" / "fixtures" / "pose_temporal.yaml"


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "run.yaml"
    p.write_text(text)
    return p


# ---------------------------------------------------------------- config matrix

def test_shipped_pose_experiment_validates():
    cfg = load_config(_POSE_CFG)
    assert cfg["model"]["pose_temporal"]["enabled"] is True
    assert cfg["pose_supervision"]["enabled"] is True
    assert cfg["output"]["monitor"] == "test/pose_mae"


def test_pose_temporal_only_build_accepted(tmp_path):
    cfg = load_config(_write(tmp_path, """
base: configs/base.yaml
model:
  pose_temporal: {enabled: true}
contact:
  targets:
    vertex: {enabled: false}
    joint: {enabled: false}
"""))
    assert cfg["model"]["pose_temporal"]["enabled"] is True


def test_pose_supervision_requires_pose_temporal(tmp_path):
    with pytest.raises(ValueError, match="requires a trainable pose path"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
pose_supervision: {enabled: true}
data:
  datasets:
    - {name: climbing_corpus, config: configs/datasets/climbing_corpus_pose.yaml}
"""))


def test_pose_supervision_requires_corpus(tmp_path):
    with pytest.raises(ValueError, match="requires a climbing_corpus dataset"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
model:
  pose_temporal: {enabled: true}
contact:
  targets:
    vertex: {enabled: false}
    joint: {enabled: false}
pose_supervision: {enabled: true}
"""))


# ------------------------------------------------------------- filter/signature

def test_pose_temporal_passes_the_filter_but_head_pose_does_not():
    assert _trainable_name_filter("pose_temporal.blocks.0.gamma_attn")
    assert _trainable_name_filter("pose_temporal.blocks.0.qkv.weight")
    # The frozen MHR head must NEVER match the exception.
    assert not _trainable_name_filter("head_pose.mhr.character_torch.skeleton.pmi")
    assert not _trainable_name_filter("head_pose.hand_pose_comps_ori")
    assert not _trainable_name_filter("init_pose.weight")


def test_pose_temporal_in_signature_only_when_enabled(tmp_path):
    base = load_config(REPO / "configs" / "base.yaml")
    assert "pose_temporal" not in (ckpt_io._arch_signature(base) or {})
    cfg = load_config(_POSE_CFG)
    sig = ckpt_io._arch_signature(cfg)
    assert sig["pose_temporal"]["enabled"] is True
    assert sig["pose_temporal"]["num_layers"] == 2


# ---------------------------------------------------------------- GPU semantics

_CKPT = load_config(REPO / "configs" / "base.yaml")["model"]["checkpoint_path"]
_CORPUS = Path("/data3/rikhat.akizhanov/better/data/ClimbingVideos")
_CONVERTED = (_CORPUS / "features" / "human_optim" / "Mu" / "Vp"
              / "MuVpoovQl2M_0001" / "mhr_1.npz")

slow = pytest.mark.slow
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
needs_ckpt = pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint missing")
needs_mhr1 = pytest.mark.skipif(not _CONVERTED.is_file(), reason="mhr_1.npz missing")


@slow
@needs_gpu
@needs_ckpt
@needs_mhr1
def test_pose_temporal_gpu_semantics():
    """Zero-init == frozen (noise floor); live gammas move pose; the loss's
    gradient reaches pose_temporal params ONLY."""
    from contact.data.climbing_corpus import ClimbingCorpusDataset
    from contact.data.collate import batch_to_device, make_collate
    from contact.engine import forward_model
    from contact.model import build_model
    from contact.pose_supervision import PoseSupervisedLoss
    from contact.targets import TargetSpec

    torch.manual_seed(0)
    cfg = load_config(_POSE_CFG)
    model, trainable = build_model(cfg, "cuda")
    model.eval()
    assert trainable and all("pose_temporal" in n for n in trainable)

    ds = ClimbingCorpusDataset(
        _CORPUS, scenes=["MuVpoovQl2M_0001"], split="train", frames_per_clip=7,
        frame_stride=1, jitter=False, load_pose=True)
    collate = make_collate(
        tuple(model.cfg.MODEL.IMAGE_SIZE), TargetSpec.from_config(cfg))
    batch = batch_to_device(collate([ds[0], ds[1]]), "cuda")

    out_live = forward_model(model, batch)
    pt = model.pose_temporal
    del model.pose_temporal
    off_a = forward_model(model, batch)
    off_b = forward_model(model, batch)
    model.pose_temporal = pt
    for key in ("body_pose", "global_rot", "pred_keypoints_3d", "pred_vertices"):
        floor = (off_a["mhr"][key] - off_b["mhr"][key]).abs().max()
        diff = (out_live["mhr"][key] - off_a["mhr"][key]).abs().max()
        assert float(diff) <= 8.0 * float(floor) + 1e-6, (key, float(diff))

    with torch.no_grad():
        for n, p in model.pose_temporal.named_parameters():
            if "gamma" in n:
                torch.nn.init.normal_(p, std=0.1)
    out_hot = forward_model(model, batch)
    moved = (out_hot["mhr"]["body_pose"] - off_a["mhr"]["body_pose"]).abs().max()
    assert float(moved) > 1e-3

    loss_fn = PoseSupervisedLoss(cfg, device="cuda")
    total, parts = loss_fn(forward_model(model, batch), batch)
    assert parts["terms"]["pose"]["weight_mass"] > 0
    total.backward()
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert "pose_temporal" in n
        elif p.grad is not None:
            raise AssertionError(f"frozen param {n} received a gradient")


# ------------------------------------------------------------- dataset validation

def test_validate_targets_accepts_a_pose_only_corpus_dataset():
    """A pose-supervised run has no contact targets: the corpus dataset counts
    as supervising through its pose pseudo-GT (load_pose), mirroring the
    force/motion exemptions."""
    from contact.targets import validate_targets

    class _Ds:
        name = "climbing_corpus"
        supervised_targets = frozenset({"joint"})
        topology = None
        load_pose = True

    cfg = {
        "contact": {
            "topology": "smplx",
            "primary_target": "joint",
            "targets": {
                "vertex": {"enabled": False},
                "joint": {"enabled": False, "joint_set": "smplx_body_22",
                          "supervise_subset": None, "derive_from_vertex": False,
                          "use_confidence_weights": False},
            },
        },
        "pose_supervision": {"enabled": True},
    }
    validate_targets(cfg, [_Ds()])                    # no raise
    _Ds.load_pose = False
    with pytest.raises(ValueError, match="supervises none"):
        validate_targets(cfg, [_Ds()])


# ------------------------------------------------------------- shape/scale rail

def test_shape_scale_rail_terms():
    """L2 rail vs the frozen stash: exact values, masking, zero-mass fallback.
    The MHR body is stubbed — the rail never touches q conversion."""
    pytest.importorskip("better_human")
    from contact.pose_supervision import PoseSupervisedLoss

    loss = PoseSupervisedLoss.__new__(PoseSupervisedLoss)
    loss.weight = 0.0
    loss.acc_weight = 0.0
    loss.shape_w = 0.0
    loss.shape_rail_w = 1.0
    loss.scale_rail_w = 2.0
    loss.huber_delta = 0.1
    loss.bones_w = 0.0
    loss.scale_w = 0.0
    loss.huber_delta_bones = 0.05
    loss.fit_err_confidence = False
    loss.fit_err_ref_cm = 2.0
    loss.device = torch.device("cpu")
    loss.dtype = torch.float32

    class _StubBody:
        @staticmethod
        def from_classic(classic):
            return None, classic.model_parameters[:, :132]

    loss.body = _StubBody()

    n = 4
    shape = torch.randn(n, 45)
    scale = torch.randn(n, 28)
    out = {"mhr": {
        "mhr_model_params": torch.zeros(n, 204),
        "shape": shape + 0.5,
        "scale": scale,
        "shape_frozen": shape,
        "scale_frozen": scale,
    }}
    batch = {"pose_gt_q": torch.zeros(n, 132),
             "pose_valid": torch.ones(n, dtype=torch.bool),
             "frame_valid": torch.tensor([True, True, True, False]),
             "seq_len": 1}
    _, parts = loss(out, batch)
    # scale matches frozen exactly; shape deviates by 0.5 in every channel.
    assert parts["terms"]["scale_rail"]["loss"] == pytest.approx(0.0, abs=1e-10)
    assert parts["terms"]["shape_rail"]["weight_mass"] == 3
    assert parts["terms"]["shape_rail"]["loss"] == pytest.approx(
        45 * 0.5 ** 2, rel=1e-5)
    assert parts["shape_dev"] == pytest.approx(0.5, rel=1e-5)

    # Missing stash (no recompute ran): zero-mass fallback, term still present
    # so the DDP term set stays rank-identical.
    del out["mhr"]["shape_frozen"], out["mhr"]["scale_frozen"]
    _, parts = loss(out, batch)
    assert parts["terms"]["shape_rail"]["weight_mass"] == 0


def test_shape_gt_identity_term():
    """L2 vs the mhr_1 v2 GT identity: exact value, pose&frame validity mask."""
    pytest.importorskip("better_human")
    from contact.pose_supervision import PoseSupervisedLoss

    loss = PoseSupervisedLoss.__new__(PoseSupervisedLoss)
    loss.weight = 0.0
    loss.acc_weight = 0.0
    loss.shape_w = 1.0
    loss.shape_rail_w = 0.0
    loss.scale_rail_w = 0.0
    loss.huber_delta = 0.1
    loss.bones_w = 0.0
    loss.scale_w = 0.0
    loss.huber_delta_bones = 0.05
    loss.fit_err_confidence = False
    loss.fit_err_ref_cm = 2.0
    loss.device = torch.device("cpu")
    loss.dtype = torch.float32

    class _StubBody:
        @staticmethod
        def from_classic(classic):
            return None, classic.model_parameters[:, :132]

    loss.body = _StubBody()

    n = 4
    identity = torch.randn(n, 45)
    out = {"mhr": {
        "mhr_model_params": torch.zeros(n, 204),
        "shape": identity + 0.5,
        "scale": torch.zeros(n, 28),
    }}
    batch = {"pose_gt_q": torch.zeros(n, 132),
             "pose_valid": torch.tensor([True, True, True, False]),
             "frame_valid": torch.tensor([True, True, False, True]),
             "pose_identity": identity,
             "seq_len": 1}
    _, parts = loss(out, batch)
    # Rows 0 and 1 are supervised; every channel deviates by exactly 0.5.
    assert parts["terms"]["shape"]["weight_mass"] == 2
    # Per-channel mean, not a 45-channel sum (the weight is per channel now).
    assert parts["terms"]["shape"]["loss"] == pytest.approx(0.25, rel=1e-5)


def test_bones_and_scale_terms():
    """Huber vs the GT lbs slots read straight off mhr_model_params: exact
    per-channel-mean value, validity mask, and the fit-err row confidence."""
    pytest.importorskip("better_human")
    from contact.pose_supervision import PoseSupervisedLoss

    loss = PoseSupervisedLoss.__new__(PoseSupervisedLoss)
    loss.weight = 0.0
    loss.acc_weight = 0.0
    loss.shape_w = 0.0
    loss.shape_rail_w = 0.0
    loss.scale_rail_w = 0.0
    loss.huber_delta = 0.1
    loss.bones_w = 1.0
    loss.scale_w = 1.0
    loss.huber_delta_bones = 1.0            # keep every channel quadratic
    loss.fit_err_confidence = False
    loss.fit_err_ref_cm = 2.0
    loss.device = torch.device("cpu")
    loss.dtype = torch.float32
    # Identity subspace: this test pins the Huber arithmetic, not the projection
    # (test_scale_target_is_projected_onto_the_reachable_subspace covers that).
    loss.scale_mean = torch.zeros(68)
    loss.scale_proj = torch.eye(68)

    class _StubBody:
        @staticmethod
        def from_classic(classic):
            return None, classic.model_parameters[:, :132]

    loss.body = _StubBody()

    n = 4
    params = torch.zeros(n, 204)
    params[:, 130:136] = 0.4                # deviates from GT by exactly 0.4
    params[:, 136:204] = 0.2                # ... and 0.2
    out = {"mhr": {"mhr_model_params": params, "shape": torch.zeros(n, 45),
                   "scale": torch.zeros(n, 28)}}
    batch = {"pose_gt_q": torch.zeros(n, 132),
             "pose_valid": torch.tensor([True, True, True, False]),
             "frame_valid": torch.tensor([True, True, False, True]),
             "pose_gt_bones": torch.zeros(n, 6),
             "pose_gt_scale": torch.zeros(n, 68),
             "mhr_fit_err_cm": torch.full((n,), 2.0),
             "seq_len": 1}
    _, parts = loss(out, batch)
    # Rows 0 and 1 only. Huber with beta=1 is 0.5 * d^2 in the quadratic regime.
    assert parts["terms"]["bones"]["weight_mass"] == 2
    assert parts["terms"]["bones"]["loss"] == pytest.approx(0.5 * 0.4 ** 2, rel=1e-5)
    assert parts["terms"]["scale"]["loss"] == pytest.approx(0.5 * 0.2 ** 2, rel=1e-5)
    assert parts["bones_mae"] == pytest.approx(0.4, rel=1e-5)
    assert parts["scale_mae"] == pytest.approx(0.2, rel=1e-5)

    # fit_err_confidence: fit_err == ref halves every row weight, so the mass
    # halves while the (mass-normalized) loss value is unchanged.
    loss.fit_err_confidence = True
    _, parts = loss(out, batch)
    assert parts["terms"]["bones"]["weight_mass"] == pytest.approx(1.0, rel=1e-6)
    assert parts["terms"]["bones"]["loss"] == pytest.approx(0.5 * 0.4 ** 2, rel=1e-5)


@needs_ckpt
def test_scale_target_is_projected_onto_the_reachable_subspace():
    """The head reaches the 68 scale slots only as ``scale_mean + c @ comps``
    (rank 24), so the raw GT is half unreachable: the loss must compare against
    the GT's PROJECTION, which the head can actually attain."""
    pytest.importorskip("better_human")
    from contact.pose_supervision import PoseSupervisedLoss, _scale_subspace

    mean, proj = _scale_subspace(_CKPT, torch.device("cpu"), torch.float32)
    assert mean.shape == (68,) and proj.shape == (68, 68)
    # A projector: symmetric, idempotent, and rank-deficient (trace = rank).
    torch.testing.assert_close(proj, proj.T, atol=1e-5, rtol=0)
    torch.testing.assert_close(proj @ proj, proj, atol=1e-5, rtol=0)
    assert float(proj.trace()) == pytest.approx(24.0, abs=1e-3)

    loss = PoseSupervisedLoss.__new__(PoseSupervisedLoss)
    for name, value in (("weight", 0.0), ("acc_weight", 0.0), ("shape_w", 0.0),
                        ("shape_rail_w", 0.0), ("scale_rail_w", 0.0),
                        ("bones_w", 0.0), ("scale_w", 1.0), ("huber_delta", 0.1),
                        ("huber_delta_bones", 1.0), ("fit_err_confidence", False),
                        ("fit_err_ref_cm", 2.0)):
        setattr(loss, name, value)
    loss.device = torch.device("cpu")
    loss.dtype = torch.float32
    loss.scale_mean = mean
    loss.scale_proj = proj

    class _StubBody:
        @staticmethod
        def from_classic(classic):
            return None, classic.model_parameters[:, :132]

    loss.body = _StubBody()

    n = 2
    torch.manual_seed(0)
    gt = mean + torch.randn(n, 68)                       # mostly unreachable
    reachable = mean + (gt - mean) @ proj
    params = torch.zeros(n, 204)
    params[:, 136:204] = reachable                       # the head predicts it exactly
    batch = {"pose_gt_q": torch.zeros(n, 132),
             "pose_valid": torch.ones(n, dtype=torch.bool),
             "frame_valid": torch.ones(n, dtype=torch.bool),
             "pose_gt_bones": torch.zeros(n, 6),
             "pose_gt_scale": gt,
             "mhr_fit_err_cm": torch.zeros(n),
             "seq_len": 1}
    out = {"mhr": {"mhr_model_params": params, "shape": torch.zeros(n, 45),
                   "scale": torch.zeros(n, 28)}}
    _, parts = loss(out, batch)
    # Predicting the projection exactly is a perfect score; against the raw GT
    # the same prediction would carry a permanent unreachable residual.
    assert parts["terms"]["scale"]["loss"] == pytest.approx(0.0, abs=1e-9)
    assert parts["scale_mae"] == pytest.approx(0.0, abs=1e-6)
    assert float((gt - reachable).abs().mean()) > 0.1


# ----------------------------------------------------------------- rope variant

def test_rope_type_validates_and_signs(tmp_path):
    cfg = load_config(_write(tmp_path, """
base: configs/base.yaml
model:
  pose_temporal:
    enabled: true
    type: rope
    num_layers: 4
    num_heads: 16
    dropout: 0.1
contact:
  targets:
    vertex: {enabled: false}
    joint: {enabled: false}
"""))
    sig = ckpt_io._arch_signature(cfg)
    assert sig["pose_temporal"] == {
        "enabled": True, "type": "rope", "num_layers": 4,
        "num_heads": 16, "mlp_ratio": 2.0, "time_scale": 25.0}


def test_rope_signature_depends_on_the_hyperparameters(tmp_path):
    """Two rope builds that differ only in depth must not share a signature."""
    def _cfg(layers):
        return load_config(_write(tmp_path, f"""
base: configs/base.yaml
model:
  pose_temporal: {{enabled: true, type: rope, num_layers: {layers}}}
contact:
  targets:
    vertex: {{enabled: false}}
    joint: {{enabled: false}}
"""))
    sig_a = ckpt_io._arch_signature(_cfg(4))["pose_temporal"]
    sig_b = ckpt_io._arch_signature(_cfg(2))["pose_temporal"]
    assert sig_a["type"] == sig_b["type"] == "rope"
    assert sig_a != sig_b


def test_retired_sliding_keys_rejected(tmp_path):
    # The sliding-window module is gone: its config keys must hard-error rather
    # than be silently ignored by a rope build.
    for key in ("causal: true", "attend: joint", "bottleneck_dim: 256",
                "position_scale: 30.0"):
        with pytest.raises(ValueError, match="unknown config key"):
            load_config(_write(tmp_path, f"""
base: configs/base.yaml
model:
  pose_temporal: {{enabled: true, type: rope, {key}}}
contact:
  targets:
    vertex: {{enabled: false}}
    joint: {{enabled: false}}
"""))


def test_unknown_pose_temporal_type_rejected(tmp_path):
    with pytest.raises(ValueError, match="type must be 'rope'"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
model:
  pose_temporal: {enabled: true, type: sliding}
contact:
  targets:
    vertex: {enabled: false}
    joint: {enabled: false}
"""))


def test_rope_max_rel_sec_null_accepted(tmp_path):
    cfg = load_config(_write(tmp_path, """
base: configs/base.yaml
model:
  pose_temporal: {enabled: true, type: rope, max_rel_sec: null}
contact:
  targets:
    vertex: {enabled: false}
    joint: {enabled: false}
"""))
    assert cfg["model"]["pose_temporal"]["max_rel_sec"] is None
