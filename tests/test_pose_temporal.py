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
_POSE_CFG = REPO / "configs" / "climbing_corpus_pose_temporal.yaml"


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
    with pytest.raises(ValueError, match="requires model.pose_temporal.enabled"):
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
    assert _trainable_name_filter("pose_temporal.token_in_proj.weight")
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
