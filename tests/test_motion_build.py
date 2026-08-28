"""Motion tokens v2: config matrix, token mask, head sizing, model-cfg patching.

Fast CPU coverage for the motion branch — the config accept/reject matrix
(motion-only builds are legal; motion_temporal needs the head; the standardize
table must match the anchor list; native-rate targets forbid a strided clip), the
three-block token attention mask, ``MotionHead`` sizing/zero-init, the
``_patch_model_cfg`` wiring, and stability of the checkpoint arch signature (the
new keys must stay out of every signature written before the motion branch).
The ``anchored: false`` variant (pure learned queries) is covered here too; the
one assertion needing a real build — that its two anchored-update projections do
not exist — is marked slow.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from yacs.config import CfgNode

from contact import checkpoint as ckpt_io
from contact.config import load_config
from sam_3d_body.models.heads import build_head
from sam_3d_body.models.meta_arch.sam3d_body import SAM3DBody

REPO = Path(__file__).resolve().parents[1]
_MOTION_CFG = REPO / "tests" / "fixtures" / "motion_seven_tokens.yaml"
_PELVIS_CFG = REPO / "configs" / "old" / "climbing_corpus_motion_pelvis_t7.yaml"
_T7HINGE_CFG = REPO / "configs" / "old" / "climbing_videos_force_warmstart_t7hinge.yaml"

#: Seven motion anchors: the six kindyn force anchors + MHR70 9 (left hip) for pelvis.
_SEVEN_ANCHORS = [62, 41, 15, 18, 17, 20, 9]
#: Canonical motion slot order (contact.data.climbing_corpus.MOTION_JOINT_NAMES).
_MOTION_JOINT_NAMES = ("left_wrist", "right_wrist", "left_foot", "right_foot",
                       "left_ankle", "right_ankle", "pelvis")


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "run.yaml"
    p.write_text(text)
    return p


_MOTION_ONLY = """
base: configs/base.yaml
model:
  motion_head:
    enabled: true
    motion_keypoint_indices: [62, 41, 15, 18, 17, 20, 9]
  motion_temporal: {enabled: true}
contact:
  targets:
    vertex: {enabled: false}
    joint: {enabled: false}
"""


# ---------------------------------------------------------------- config matrix

def test_motion_only_build_accepted(tmp_path):
    cfg = load_config(_write(tmp_path, _MOTION_ONLY))
    assert cfg["model"]["motion_head"]["enabled"] is True
    assert cfg["model"]["motion_head"]["motion_keypoint_indices"] == _SEVEN_ANCHORS
    assert cfg["model"]["motion_temporal"]["enabled"] is True
    assert cfg["contact"]["targets"]["vertex"]["enabled"] is False
    assert cfg["contact"]["targets"]["joint"]["enabled"] is False


def _with_anchored(value: str) -> str:
    """``_MOTION_ONLY`` with an ``anchored:`` line inside ``model.motion_head``."""
    return _MOTION_ONLY.replace(
        "    motion_keypoint_indices: [62, 41, 15, 18, 17, 20, 9]",
        f"    motion_keypoint_indices: [62, 41, 15, 18, 17, 20, 9]\n    anchored: {value}")


def test_motion_unanchored_config_accepted(tmp_path):
    """``anchored: false`` = pure learned queries; the anchors still name the slots."""
    cfg = load_config(_write(tmp_path, _with_anchored("false")))
    assert cfg["model"]["motion_head"]["anchored"] is False
    # The anchor list is untouched: it still defines K and the supervision order.
    assert cfg["model"]["motion_head"]["motion_keypoint_indices"] == _SEVEN_ANCHORS


def test_motion_anchored_accepted_and_defaults_to_true(tmp_path):
    assert load_config(
        _write(tmp_path, _with_anchored("true")))["model"]["motion_head"]["anchored"] is True
    assert load_config(
        _write(tmp_path, _MOTION_ONLY))["model"]["motion_head"]["anchored"] is True


@pytest.mark.parametrize("bad", ["0", "'false'", "null"])
def test_bad_motion_anchored_value_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="motion_head.anchored must be a boolean"):
        load_config(_write(tmp_path, _with_anchored(bad)))


def test_shipped_motion_experiment_validates():
    cfg = load_config(_MOTION_CFG)
    assert cfg["motion_supervision"]["enabled"] is True
    assert cfg["motion_supervision"]["target_frame"] == "all"
    assert cfg["data"]["sequence"] == {
        "frames_per_clip": 7, "frame_stride": 1, "jitter": True, "target_frame": "all"}
    assert len(cfg["motion_supervision"]["standardize"]["mean"]) == len(_SEVEN_ANCHORS)
    # Monitored on the quantity the pre-registered v1 bars use (pelvis, not the
    # 7-joint mean, and not val/loss — ~31% of val loss entries are outlier rows
    # that carry no training gradient).
    assert cfg["output"]["monitor"] == "val/motion_acc_vert_r_pelvis"


def test_no_enabled_target_still_rejected_when_motion_is_off(tmp_path):
    with pytest.raises(ValueError, match="no contact target is enabled"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
contact:
  targets:
    vertex: {enabled: false}
    joint: {enabled: false}
"""))


def test_motion_temporal_requires_motion_head_enabled(tmp_path):
    with pytest.raises(
        ValueError, match="motion_temporal.enabled requires model.motion_head.enabled"
    ):
        load_config(_write(tmp_path, """
base: configs/base.yaml
model:
  motion_temporal: {enabled: true}
"""))


@pytest.mark.parametrize("bad", ["[]", "[70]", "[-1, 62]", "[62.5]", "[true]", "62"])
def test_bad_motion_anchor_values_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="motion_keypoint_indices must be a non-empty"):
        load_config(_write(tmp_path, f"""
base: configs/base.yaml
model:
  motion_head:
    enabled: true
    motion_keypoint_indices: {bad}
contact:
  targets:
    vertex: {{enabled: false}}
    joint: {{enabled: false}}
"""))


def test_motion_temporal_bad_divisibility_rejected(tmp_path):
    with pytest.raises(
        ValueError, match="model.motion_temporal.bottleneck_dim must be divisible"
    ):
        load_config(_write(
            tmp_path,
            _MOTION_ONLY.replace(
                "motion_temporal: {enabled: true}",
                "motion_temporal: {enabled: true, bottleneck_dim: 256, num_heads: 7}"),
        ))


def test_motion_supervision_requires_motion_head(tmp_path):
    with pytest.raises(
        ValueError, match="motion_supervision.enabled requires model.motion_head.enabled"
    ):
        load_config(_write(tmp_path, """
base: configs/base.yaml
motion_supervision: {enabled: true}
"""))


def test_motion_supervision_requires_corpus_dataset(tmp_path):
    with pytest.raises(ValueError, match="requires a climbing_corpus dataset"):
        load_config(_write(tmp_path, _MOTION_ONLY + """
motion_supervision:
  enabled: true
  standardize:
    mean: [[[0,0,0],[0,0,0]], [[0,0,0],[0,0,0]], [[0,0,0],[0,0,0]], [[0,0,0],[0,0,0]],
           [[0,0,0],[0,0,0]], [[0,0,0],[0,0,0]], [[0,0,0],[0,0,0]]]
    std: [[[1,1,1],[1,1,1]], [[1,1,1],[1,1,1]], [[1,1,1],[1,1,1]], [[1,1,1],[1,1,1]],
          [[1,1,1],[1,1,1]], [[1,1,1],[1,1,1]], [[1,1,1],[1,1,1]]]
"""))


def test_motion_supervision_rejects_strided_clips(tmp_path):
    # Targets are native-rate central differences: stride > 1 would show the model
    # frames stride/fps apart while the target is a 1/fps derivative.
    with pytest.raises(ValueError, match="requires data.sequence.frame_stride=1"):
        load_config(_write(tmp_path, """
base: tests/fixtures/motion_seven_tokens.yaml
data:
  sequence: {frames_per_clip: 7, frame_stride: 2, jitter: true, target_frame: all}
"""))


def test_motion_supervision_center_requires_odd_clip(tmp_path):
    with pytest.raises(ValueError, match="target_frame='center' requires an odd"):
        load_config(_write(tmp_path, """
base: tests/fixtures/motion_seven_tokens.yaml
motion_supervision: {target_frame: center}
data:
  sequence: {frames_per_clip: 8, frame_stride: 1, jitter: true, target_frame: all}
"""))


def test_standardize_table_must_match_anchor_count(tmp_path):
    with pytest.raises(ValueError, match=r"standardize.std must be a finite \[7\]\[2\]\[3\]"):
        load_config(_write(tmp_path, """
base: tests/fixtures/motion_seven_tokens.yaml
motion_supervision:
  standardize:
    std: [[[1,1,1],[1,1,1]]]
"""))


def test_standardize_std_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="standardize.std entries must be positive"):
        load_config(_write(tmp_path, """
base: tests/fixtures/motion_seven_tokens.yaml
motion_supervision:
  standardize:
    std: [[[1,1,1],[1,1,1]], [[1,1,1],[1,1,1]], [[1,1,1],[1,1,1]], [[1,1,1],[1,1,1]],
          [[1,1,1],[1,1,1]], [[1,1,1],[1,1,1]], [[1,1,1],[0,1,1]]]
"""))


def test_motion_supervision_enabled_requires_standardize_table(tmp_path):
    with pytest.raises(ValueError, match="standardize.mean must be a finite"):
        load_config(_write(tmp_path, _MOTION_ONLY + """
data:
  datasets:
    - {name: climbing_corpus, config: configs/datasets/climbing_corpus_motion.yaml}
  eval_split: val
  sequence: {frames_per_clip: 7, frame_stride: 1, jitter: true, target_frame: all}
motion_supervision: {enabled: true}
"""))


def test_motion_supervision_requires_exactly_seven_anchors(tmp_path):
    # Token k <-> joint_names[k] <-> standardize row k is convention-order;
    # a wrong LENGTH is the only mechanically detectable mistake, so it must fail
    # at config load, not three minutes into a run.
    with pytest.raises(ValueError, match="requires exactly 7"):
        load_config(_write(tmp_path, """
base: tests/fixtures/motion_seven_tokens.yaml
model:
  motion_head:
    motion_keypoint_indices: [62, 41, 15, 18, 17, 20]
"""))


def test_anchor_count_must_match_the_joint_name_subset(tmp_path):
    with pytest.raises(ValueError, match="requires exactly 1 .*pelvis"):
        load_config(_write(tmp_path, """
base: configs/old/climbing_corpus_motion_pelvis_t7.yaml
model:
  motion_head:
    motion_keypoint_indices: [9, 62]
"""))


@pytest.mark.parametrize("names", [
    "[pelvis, pelvis]", "[nose]", "[]", "pelvis",
])
def test_bad_joint_names_rejected(tmp_path, names):
    with pytest.raises(ValueError, match="joint_names must be null"):
        load_config(_write(tmp_path, f"""
base: configs/old/climbing_corpus_motion_pelvis_t7.yaml
motion_supervision:
  joint_names: {names}
"""))


def test_limb_slots_with_label_smoothing_rejected(tmp_path):
    # Smoothing is root-only: a limb slot would carry a RAW central difference
    # expressed in a SMOOTHED frame. Must fail loudly, not ship silently.
    with pytest.raises(ValueError, match="pelvis slot only"):
        load_config(_write(tmp_path, """
base: configs/old/climbing_corpus_motion_pelvis_t7.yaml
model:
  motion_head:
    motion_keypoint_indices: [9, 62]
motion_supervision:
  joint_names: [pelvis, left_wrist]
  standardize:
    mean: [[[0,0,0],[0,0,0]], [[0,0,0],[0,0,0]]]
    std: [[[1,1,1],[1,1,1]], [[1,1,1],[1,1,1]]]
"""))


def test_limb_slots_allowed_without_label_smoothing():
    # The v2 experiment is exactly this case: seven slots, target_smooth_sec 0.
    cfg = load_config(_MOTION_CFG)
    assert cfg["motion_supervision"]["joint_names"] is None
    assert cfg["motion_supervision"]["target_smooth_sec"] == 0.0


def test_auto_stride_rejected_without_motion_supervision(tmp_path):
    # evaluate.py / demo.py / the renderers read frame_stride as a plain int.
    with pytest.raises(ValueError, match="frame_stride: auto requires"):
        load_config(_write(tmp_path, """
base: configs/old/climbing_videos_joint.yaml
data:
  sequence: {frames_per_clip: 7, frame_stride: auto, jitter: true, target_frame: center}
"""))


def test_bad_root_convention_rejected(tmp_path):
    with pytest.raises(ValueError, match="root_convention must be one of"):
        load_config(_write(tmp_path, """
base: configs/old/climbing_corpus_motion_pelvis_t7.yaml
motion_supervision:
  root_convention: world
"""))


def test_shipped_pelvis_experiment_validates():
    """The v3 run: one twist-convention pelvis token, val-less, r3d monitor."""
    cfg = load_config(_PELVIS_CFG)
    assert cfg["model"]["motion_head"]["motion_keypoint_indices"] == [9]
    assert cfg["motion_supervision"]["joint_names"] == ["pelvis"]
    assert cfg["motion_supervision"]["root_convention"] == "twist"
    assert cfg["motion_supervision"]["target_smooth_sec"] == 0.12
    assert len(cfg["motion_supervision"]["standardize"]["mean"]) == 1
    # No val split: every curated train scene trains, the manual test scenes score.
    assert cfg["data"]["eval_split"] == "test"
    # `auto` stride = fixed PHYSICAL clip span at every corpus frame rate.
    assert cfg["data"]["sequence"] == {
        "frames_per_clip": 7, "frame_stride": "auto", "jitter": True,
        "target_frame": "all"}
    assert cfg["output"]["monitor"] == "test/motion_acc_r3d_pelvis"
    assert cfg["optim"]["epochs"] == 10


def test_v2_experiment_keeps_its_legacy_target_definition():
    # The defaults flipped to `twist` / 0.12 s smoothing; the v2 checkpoint trained
    # on raw `rotated_world` targets and its config must keep saying so.
    ms = load_config(_MOTION_CFG)["motion_supervision"]
    assert ms["root_convention"] == "rotated_world"
    assert ms["target_smooth_sec"] == 0.0
    assert load_config(_MOTION_CFG)["data"]["sequence"]["frame_stride"] == 1


def test_motion_defaults_load():
    cfg = load_config(REPO / "configs" / "base.yaml")
    assert cfg["model"]["motion_head"] == {
        "enabled": False,
        "motion_keypoint_indices": _SEVEN_ANCHORS,
        "anchored": True,
        "mlp_depth": 2,
        "mlp_channel_div_factor": 4,
        "dropout": 0.0,
    }
    assert cfg["model"]["motion_temporal"] == {
        "enabled": False,
        "bottleneck_dim": 256,
        "num_layers": 1,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "attend": "per_token",
        "causal": False,
        "dropout": 0.0,
        "position_scale": 1.0,
    }
    assert cfg["motion_supervision"] == {
        "enabled": False,
        "target_frame": "all",
        "joint_names": None,
        "root_convention": "twist",
        "angular": False,
        "target_smooth_sec": 0.12,
        "standardize": {"mean": None, "std": None},
        "loss": {"vel": 1.0, "acc": 1.0, "ang_vel": 1.0, "ang_acc": 1.0,
                 "huber_delta": 1.0, "outlier_acc_ms2": 50.0},
    }


# ---------------------------------------------------------------- monitor plumbing

class _MonitorStub:
    """Minimal stand-in exposing what ``_validate_monitor`` reads (see test_monitor.py)."""

    def __init__(self, monitor, motion_supervised=True, eval_split="val",
                 motion_joint_names=_MOTION_JOINT_NAMES):
        self.monitor = monitor
        self.targets = ()
        self.eval_split = eval_split
        self.motion_supervised = motion_supervised
        self.motion_joint_names = motion_joint_names


def _train_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_mod", REPO / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_monitor_accepts_motion_metrics_and_rejects_them_when_disabled():
    tm = _train_module()
    for mon in ("val/loss", "val/motion_acc_vert_r_pelvis", "val/motion_vel_vert_r_mean",
                "val/motion_acc_r3d_pelvis", "val/motion_acc_r3d_mean",
                "val/motion_acc_rmse_left_wrist", "val/motion_vel_rmse_mean"):
        tm.Trainer._validate_monitor(_MonitorStub(mon))                 # no raise
    with pytest.raises(ValueError, match="not a valid metric"):
        tm.Trainer._validate_monitor(
            _MonitorStub("val/motion_acc_vert_r_pelvis", motion_supervised=False))
    with pytest.raises(ValueError, match="not a valid metric"):
        tm.Trainer._validate_monitor(_MonitorStub("val/motion_acc_vert_r_nose"))
    # The valid set follows the CONFIGURED slots: a pelvis-only run must not
    # accept a limb metric it will never produce.
    tm.Trainer._validate_monitor(
        _MonitorStub("test/motion_acc_r3d_pelvis", eval_split="test",
                     motion_joint_names=("pelvis",)))
    with pytest.raises(ValueError, match="not a valid metric"):
        tm.Trainer._validate_monitor(
            _MonitorStub("test/motion_acc_r3d_left_wrist", eval_split="test",
                         motion_joint_names=("pelvis",)))


def test_monitor_value_reaches_the_motion_metrics_dict():
    # The monitor name is `{split}/motion_<key>` (single slash, the val/force_mae
    # convention `_monitor_value` parses) — NOT the `val/motion/<key>` log tag.
    tm = _train_module()
    stub = _MonitorStub("val/motion_acc_vert_r_pelvis")
    val = {"loss": 9.0, "metrics": {"motion": {"acc_vert_r_pelvis": 0.31,
                                               "vel_rmse_mean": 1.2}}}
    assert tm.Trainer._monitor_value(stub, val) == pytest.approx(0.31)
    stub.monitor = "val/motion_vel_rmse_mean"
    assert tm.Trainer._monitor_value(stub, val) == pytest.approx(1.2)


@pytest.mark.parametrize("monitor,mode", [
    ("val/motion_acc_vert_r_pelvis", "max"),
    ("val/motion_vel_vert_r_mean", "max"),
    ("val/motion_acc_rmse_pelvis", "min"),
    ("val/motion_vel_rmse_mean", "min"),
    ("val/loss", "min"),
])
def test_monitor_direction_inference(monitor, mode):
    # Mirrors the expression in Trainer.__init__ (correlations up, errors down).
    inferred = ("min" if (monitor.endswith(("/loss", "/physics_residual", "/force_mae"))
                          or "_rmse_" in monitor) else "max")
    assert inferred == mode


# ---------------------------------------------------------------- outlier sentinel

_CORPUS = Path("/data3/rikhat.akizhanov/better/data/ClimbingVideos")
requires_corpus = pytest.mark.skipif(
    not (_CORPUS / "scenes" / "scenes.db").is_file(),
    reason="ClimbingVideos corpus not available")


@requires_corpus
def test_outlier_threshold_zero_disables_the_flag():
    """``outlier_acc_ms2: 0`` is the documented "off" sentinel.

    Without the guard, ``|acc| > 0`` is true for every entry, the whole TRAIN loss
    mask goes False, every batch reports ``active=False`` and the run takes zero
    optimiser steps while looking perfectly healthy.
    """
    from contact.data.climbing_corpus import ClimbingCorpusDataset, list_corpus_scenes

    scene = list_corpus_scenes(_CORPUS, "train")[0]
    common = dict(scenes=[scene], split="train", frames_per_clip=7, frame_stride=1,
                  jitter=False, load_motion=True, load_images=False)
    off = ClimbingCorpusDataset(_CORPUS, motion_outlier_acc_ms2=0.0, **common)
    on = ClimbingCorpusDataset(_CORPUS, motion_outlier_acc_ms2=1.0, **common)
    off_mask = off._scenes[scene]["motion_outlier"]
    on_mask = on._scenes[scene]["motion_outlier"]
    assert off_mask.dtype == bool and off_mask.shape == on_mask.shape
    assert not off_mask.any(), "threshold 0 must disable the outlier flag entirely"
    # Sanity: the guard changes behaviour rather than the data being outlier-free.
    assert on_mask.any(), "a 1 m/s^2 threshold should flag something"


# ---------------------------------------------------------------- dataset validation

class _FakeDataset:
    def __init__(self, name, supervised, topology, load_motion=False):
        self.name = name
        self.supervised_targets = frozenset(supervised)
        self.topology = topology
        self.load_motion = load_motion


def _motion_only_cfg(motion_supervised: bool) -> dict:
    return {
        "contact": {
            "topology": "smpl",
            "primary_target": "vertex",
            "targets": {
                "vertex": {"enabled": False},
                "joint": {
                    "enabled": False,
                    "joint_set": "smplx_body_22",
                    "supervise_subset": None,
                    "derive_from_vertex": False,
                    "use_confidence_weights": False,
                },
            },
        },
        "motion_supervision": {"enabled": motion_supervised},
    }


def test_motion_dataset_counts_as_supervising():
    from contact.targets import validate_targets

    ds = _FakeDataset("climbing_corpus", {"joint"}, None, load_motion=True)
    validate_targets(_motion_only_cfg(True), [ds])          # no raise


def test_motion_dataset_without_motion_supervision_still_rejected():
    from contact.targets import validate_targets

    ds = _FakeDataset("climbing_corpus", {"joint"}, None, load_motion=True)
    with pytest.raises(ValueError, match="supervises none"):
        validate_targets(_motion_only_cfg(False), [ds])


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


def test_token_mask_contact_force_motion():
    mask = SAM3DBody._build_block_token_mask(1, 20, [8, 14, 17], torch.device("cpu"))
    assert mask.shape == (1, 20, 20)
    _assert_block_pattern(mask, [8, 14, 17])


def test_token_mask_motion_only():
    mask = SAM3DBody._build_block_token_mask(2, 11, [4], torch.device("cpu"))
    _assert_block_pattern(mask, [4])
    # Motion tokens attend everything; original tokens never attend motion tokens.
    assert bool(mask[0, 4:, :].all())
    assert not bool(mask[0, :4, 4:].any())


# ---------------------------------------------------------------- motion head

def _motion_head_cfg(anchors: list[int], dim: int = 16) -> CfgNode:
    return CfgNode({"MODEL": {
        "DECODER": {"DIM": dim},
        "MOTION_HEAD": {"KEYPOINT_INDICES": anchors, "MLP_DEPTH": 2,
                        "MLP_CHANNEL_DIV_FACTOR": 2, "DROPOUT": 0.0},
    }})


def test_motion_head_sized_by_anchor_list_and_zero_init():
    head = build_head(_motion_head_cfg(_SEVEN_ANCHORS), "motion")
    assert head.num_motion_tokens == 7
    out = head(torch.randn(2, 7, 16))
    assert out.shape == (2, 7, 6)
    # Zero-init final linear -> standardized-mean (exactly zero) prediction at init.
    assert torch.equal(out, torch.zeros_like(out))


def test_motion_head_rejects_wrong_token_count():
    head = build_head(_motion_head_cfg(_SEVEN_ANCHORS), "motion")
    with pytest.raises(ValueError, match="motion-head input token count"):
        head(torch.randn(2, 6, 16))


# ---------------------------------------------------------------- model-config patching

def _min_model_cfg() -> CfgNode:
    return CfgNode({"MODEL": {"DECODER": {}, "PROMPT_ENCODER": {}, "MHR_HEAD": {}}})


def test_patch_model_cfg_motion_only(tmp_path):
    from contact.model import _patch_model_cfg

    cfg = load_config(_write(tmp_path, _MOTION_ONLY))
    model_cfg = _patch_model_cfg(_min_model_cfg(), cfg, "mhr.pt")
    assert model_cfg.MODEL.DECODER.DO_CONTACT_TOKENS is False
    assert model_cfg.MODEL.DECODER.DO_FORCE_TOKENS is False
    assert model_cfg.MODEL.DECODER.DO_MOTION_TOKENS is True
    assert model_cfg.MODEL.MOTION_HEAD.KEYPOINT_INDICES == _SEVEN_ANCHORS
    assert model_cfg.MODEL.MOTION_TEMPORAL.ENABLED is True
    assert len(model_cfg.MODEL.CONTACT_HEAD.TARGETS) == 0


def test_patch_model_cfg_threads_the_anchored_flag(tmp_path):
    from contact.model import _patch_model_cfg

    cfg = load_config(_write(tmp_path, _with_anchored("false")))
    assert _patch_model_cfg(
        _min_model_cfg(), cfg, "mhr.pt").MODEL.MOTION_HEAD.ANCHORED is False
    cfg = load_config(_write(tmp_path, _MOTION_ONLY))
    assert _patch_model_cfg(
        _min_model_cfg(), cfg, "mhr.pt").MODEL.MOTION_HEAD.ANCHORED is True


def test_patch_model_cfg_motion_disabled_is_still_self_describing():
    from contact.model import _patch_model_cfg

    cfg = load_config(REPO / "configs" / "base.yaml")
    model_cfg = _patch_model_cfg(_min_model_cfg(), cfg, "mhr.pt")
    assert model_cfg.MODEL.DECODER.DO_MOTION_TOKENS is False
    assert model_cfg.MODEL.MOTION_HEAD.KEYPOINT_INDICES == _SEVEN_ANCHORS
    assert model_cfg.MODEL.MOTION_HEAD.ANCHORED is True
    assert model_cfg.MODEL.MOTION_TEMPORAL.ENABLED is False


def test_motion_params_pass_the_trainable_filter():
    from contact.model import _trainable_name_filter

    for name in ("motion_embedding.weight", "head_motion.proj.layers.0.0.weight",
                 "motion_temporal.blocks.0.gamma_attn", "motion_feat_linear.bias"):
        assert _trainable_name_filter(name), name
    assert not _trainable_name_filter("backbone.blocks.0.attn.qkv.weight")


# ---------------------------------------------------------------- signature stability

def test_arch_signature_omits_motion_keys_when_disabled():
    # Checkpoints written before the motion branch store signatures without these
    # keys, and the comparison in `_check_schema` is an exact dict equality.
    sig = ckpt_io._arch_signature(load_config(_T7HINGE_CFG))
    assert "motion" not in sig
    assert "motion_temporal" not in sig


def test_arch_signature_carries_motion_keys_when_enabled():
    sig = ckpt_io._arch_signature(load_config(_MOTION_CFG))
    assert sig["motion"] == {
        "enabled": True,
        "motion_keypoint_indices": _SEVEN_ANCHORS,
        "mlp_depth": 2,
        "mlp_channel_div_factor": 4,
        "dropout": 0.0,
    }
    assert sig["motion_temporal"]["enabled"] is True
    assert sig["motion_temporal"]["attend"] == "joint"
    assert sig["motion_temporal"]["num_layers"] == 2
    assert sig["motion_temporal"]["position_scale"] == pytest.approx(30.0)


def test_arch_signature_records_only_the_non_default_anchoring(tmp_path):
    """Anchored (default) signatures stay byte-identical; unanchored ones say so.

    Unanchored builds have a different trainable param set (no motion posemb/feat
    linears), so the signature must separate them — but only by a key that every
    already-written anchored checkpoint also omits.
    """
    anchored = ckpt_io._arch_signature(load_config(_write(tmp_path, _MOTION_ONLY)))
    unanchored = ckpt_io._arch_signature(
        load_config(_write(tmp_path, _with_anchored("false"))))
    assert "anchored" not in anchored["motion"]
    assert unanchored["motion"].pop("anchored") is False
    assert unanchored == anchored          # nothing else moved


# ---------------------------------------------------------------- angular (12-dim)

_ANGULAR_TABLE = """
    mean: [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    std: [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
"""

_ANGULAR_BASE = """
base: configs/base.yaml
model:
  motion_head:
    enabled: true
    motion_keypoint_indices: [9]
contact:
  targets:
    vertex: {enabled: false}
    joint: {enabled: false}
data:
  datasets:
  - {name: climbing_corpus, config: configs/datasets/climbing_corpus_motion.yaml}
  sequence: {frames_per_clip: 7, frame_stride: auto}
motion_supervision:
  enabled: true
  joint_names: [pelvis]
  angular: true
  standardize:
"""


def test_angular_config_accepted_and_widens_the_head(tmp_path):
    from contact.model import _patch_model_cfg

    cfg = load_config(_write(tmp_path, _ANGULAR_BASE + _ANGULAR_TABLE))
    assert cfg["motion_supervision"]["angular"] is True
    model_cfg = _patch_model_cfg(_min_model_cfg(), cfg, "mhr.pt")
    assert model_cfg.MODEL.MOTION_HEAD.OUTPUT_DIMS == 12
    # Non-angular configs keep the 6-wide head.
    base = load_config(REPO / "configs" / "base.yaml")
    assert _patch_model_cfg(
        _min_model_cfg(), base, "mhr.pt").MODEL.MOTION_HEAD.OUTPUT_DIMS == 6


def test_angular_requires_twist_convention(tmp_path):
    with pytest.raises(ValueError, match="angular requires root_convention"):
        load_config(_write(tmp_path, _ANGULAR_BASE.replace(
            "  angular: true", "  angular: true\n  root_convention: rotated_world",
        ) + _ANGULAR_TABLE))


def test_angular_requires_pelvis_only(tmp_path):
    with pytest.raises(ValueError, match="angular requires joint_names"):
        load_config(_write(tmp_path, _ANGULAR_BASE.replace(
            "joint_names: [pelvis]", "joint_names: [left_wrist, pelvis]",
        ).replace("motion_keypoint_indices: [9]",
                  "motion_keypoint_indices: [62, 9]") + _ANGULAR_TABLE))


def test_angular_standardize_needs_four_groups(tmp_path):
    with pytest.raises(ValueError, match=r"\[1\]\[4\]\[3\]"):
        load_config(_write(tmp_path, _ANGULAR_BASE + """
    mean: [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    std: [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
"""))


def test_motion_head_supports_twelve_outputs():
    cfg = _motion_head_cfg([9])
    cfg.MODEL.MOTION_HEAD.OUTPUT_DIMS = 12
    head = build_head(cfg, "motion")
    out = head(torch.randn(2, 1, 16))
    assert out.shape == (2, 1, 12)
    assert torch.equal(out, torch.zeros_like(out))


# ---------------------------------------------------------------- unanchored build (slow)

_BASE_CKPT = load_config(REPO / "configs" / "base.yaml")["model"]["checkpoint_path"]


def _build_motion(anchored: bool):
    from contact.model import build_model

    cfg = load_config(_MOTION_CFG)
    cfg["model"]["motion_head"]["anchored"] = anchored
    torch.manual_seed(0)
    return build_model(cfg, "cuda")


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not Path(_BASE_CKPT).exists(), reason="checkpoint missing")
def test_unanchored_build_drops_the_anchored_projections():
    """``anchored: false`` builds no projections at all — no dead trainable params.

    They would otherwise never receive a gradient, which DDP rejects without
    ``find_unused_parameters``; their absence is also what makes the per-layer
    anchored update *impossible* rather than merely skipped.
    """
    model, trainable = _build_motion(False)

    assert model.motion_anchored is False
    assert not hasattr(model, "motion_posemb_linear")
    assert not hasattr(model, "motion_feat_linear")
    assert not any("motion_posemb" in name or "motion_feat" in name
                   for name in trainable)
    # Tokens, head and the temporal block must still train.
    assert any("motion_embedding" in name for name in trainable)
    assert any("head_motion" in name for name in trainable)
    assert any("motion_temporal" in name for name in trainable)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not Path(_BASE_CKPT).exists(), reason="checkpoint missing")
def test_anchored_build_keeps_the_anchored_projections():
    model, trainable = _build_motion(True)

    assert model.motion_anchored is True
    assert any("motion_posemb_linear" in name for name in trainable)
    assert any("motion_feat_linear" in name for name in trainable)
