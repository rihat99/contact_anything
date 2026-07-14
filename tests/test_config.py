"""Config loader: base-include merge, unknown-key rejection, mhr rejection."""
from __future__ import annotations

from pathlib import Path

import pytest

from contact.config import load_config

REPO = Path(__file__).resolve().parents[1]


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "run.yaml"
    p.write_text(text)
    return p


def test_base_yaml_loads_with_defaults():
    cfg = load_config(REPO / "configs" / "base.yaml")
    assert cfg["contact"]["topology"] == "smpl"
    assert cfg["contact"]["targets"]["vertex"]["enabled"] is True
    assert cfg["contact"]["targets"]["joint"]["enabled"] is False
    assert cfg["data"]["frames_per_batch"] == 32
    assert cfg["loss"]["grad_clip"] == 1.0


def test_base_include_deep_merges_child_over_base(tmp_path):
    cfg = load_config(_write(tmp_path, """
base: configs/base.yaml
optim:
  lr: 5.0e-4
output:
  exp_name: merged
"""))
    # overridden
    assert cfg["optim"]["lr"] == pytest.approx(5.0e-4)
    assert cfg["output"]["exp_name"] == "merged"
    # inherited siblings survive the deep merge
    assert cfg["optim"]["epochs"] == 20
    assert cfg["output"]["log_freq"] == 10
    assert cfg["contact"]["targets"]["vertex"]["loss"]["focal_alpha"] == pytest.approx(0.75)


def test_ported_baselines_keep_semantics():
    damon = load_config(REPO / "configs" / "damon_baseline.yaml")
    assert [d["name"] for d in damon["data"]["datasets"]] == ["damon"]
    assert damon["contact"]["targets"]["vertex"]["enabled"] is True
    assert damon["contact"]["targets"]["joint"]["enabled"] is False
    assert damon["model"]["contact_head"]["pool_mode"] == "concat"

    joint = load_config(REPO / "configs" / "climbing_videos_joint.yaml")
    assert joint["contact"]["primary_target"] == "joint"
    assert joint["contact"]["targets"]["vertex"]["enabled"] is False
    assert joint["contact"]["targets"]["joint"]["enabled"] is True
    assert joint["contact"]["targets"]["joint"]["joint_set"] == "extremities_4"
    assert joint["contact"]["targets"]["joint"]["supervise_subset"] is None
    assert joint["contact"]["targets"]["joint"]["use_confidence_weights"] is True
    assert joint["contact"]["targets"]["joint"]["loss"] == {
        "focal_alpha": 0.6,
        "focal_gamma": 2.0,
        "focal_weight": 5.0,
        "dice_weight": 0.0,
        "sparsity_weight": 0.0,
    }
    assert joint["model"]["temporal"]["enabled"] is False
    assert joint["data"]["sequence"]["frames_per_clip"] == 1
    assert joint["data"]["sequence"]["frame_stride"] == 1
    assert joint["data"]["frames_per_batch"] == 64
    assert joint["optim"]["lr"] == pytest.approx(4.0e-4)
    assert joint["logging"]["wandb"]["enabled"] is False
    assert joint["data"]["eval_split"] == "test"
    assert joint["logging"]["tensorboard_metrics"] == [
        "train/loss", "train/lr", "train/grad_norm", "train/joint/f1",
        "test/loss", "test/joint/precision", "test/joint/recall", "test/joint/f1",
    ]
    assert joint["model"]["contact_head"]["contact_keypoint_indices"] == [62, 41, 13, 14]
    assert joint["model"]["contact_head"]["num_global_tokens"] == 0
    assert joint["model"]["contact_head"]["pool_mode"] == "per_token"
    assert joint["output"]["exp_name"] == "climb4_frame"
    assert joint["output"]["monitor"] == "test/joint_f1"

    temporal = load_config(REPO / "configs" / "climbing_videos_joint_temporal.yaml")
    assert temporal["model"]["temporal"] == {
        "enabled": True,
        "placement": "post_decoder",
        "bottleneck_dim": 256,
        "num_layers": 1,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "attend": "per_token",
        "causal": False,
        "dropout": 0.0,
        "position_scale": 30.0,
    }
    assert temporal["data"]["sequence"]["frames_per_clip"] == 5
    assert temporal["data"]["sequence"]["frame_stride"] == 1
    assert temporal["data"]["frames_per_batch"] == 60
    assert temporal["optim"]["lr"] == pytest.approx(3.75e-4)
    assert temporal["model"]["init_contact_checkpoint"] is None
    assert temporal["output"]["exp_name"] == "climb4_t5"


def test_unknown_top_level_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown config key: 'bogus'"):
        load_config(_write(tmp_path, "base: configs/base.yaml\nbogus: 1\n"))


def test_unknown_nested_key_reports_dotted_path(tmp_path):
    with pytest.raises(ValueError, match=r"unknown config key: 'model.contact_head.typo'"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
model:
  contact_head:
    typo: 3
"""))


def test_mhr_topology_raises_not_implemented(tmp_path):
    with pytest.raises(NotImplementedError):
        load_config(_write(tmp_path, "base: configs/base.yaml\ncontact:\n  topology: mhr\n"))


def test_unknown_topology_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown topology"):
        load_config(_write(tmp_path, "base: configs/base.yaml\ncontact:\n  topology: banana\n"))


def test_unknown_joint_set_rejected(tmp_path):
    with pytest.raises(ValueError, match="joint_set must be one of"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
contact:
  targets:
    joint:
      joint_set: hands_and_feet_and_maybe_tail
"""))


def test_unknown_contact_pool_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match="pool_mode must be one of"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
model:
  contact_head:
    pool_mode: magical_pool
"""))


def test_per_token_pool_requires_one_output_per_total_token(tmp_path):
    with pytest.raises(ValueError, match="output dimension.*total token count 4"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
model:
  contact_head:
    contact_keypoint_indices: [62, 41, 13, 14]
    num_global_tokens: 0
    pool_mode: per_token
"""))


def test_extremity_joint_set_requires_full_four_output_space(tmp_path):
    with pytest.raises(ValueError, match="supervise_subset must be null"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
contact:
  targets:
    joint:
      joint_set: extremities_4
      supervise_subset: observable_14
"""))


def test_extremity_joint_set_is_accepted(tmp_path):
    cfg = load_config(_write(tmp_path, """
base: configs/base.yaml
contact:
  targets:
    joint:
      joint_set: extremities_4
      supervise_subset: null
"""))
    assert cfg["contact"]["targets"]["joint"]["joint_set"] == "extremities_4"


def test_no_enabled_target_rejected(tmp_path):
    with pytest.raises(ValueError, match="no contact target is enabled"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
contact:
  targets:
    vertex: {enabled: false}
    joint: {enabled: false}
"""))


def test_unknown_dataset_name_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown dataset"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
data:
  datasets:
    - {name: nope, config: x.yaml}
"""))


def test_manual_test_eval_requires_climbing_videos_only(tmp_path):
    with pytest.raises(ValueError, match="ClimbingVideos-only"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
data:
  eval_split: test
  datasets:
    - {name: damon, config: configs/datasets/damon.yaml}
"""))


def test_tensorboard_metric_filter_must_be_a_string_list(tmp_path):
    with pytest.raises(ValueError, match="tensorboard_metrics"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
logging:
  tensorboard_metrics: train/loss
"""))


@pytest.mark.parametrize("value", ["0", "-1", ".nan", ".inf"])
def test_temporal_position_scale_must_be_finite_and_positive(tmp_path, value):
    with pytest.raises(ValueError, match="position_scale must be finite and positive"):
        load_config(_write(tmp_path, f"""
base: configs/base.yaml
model:
  temporal:
    position_scale: {value}
"""))


def test_namespace_scalar_rejected(tmp_path):
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(_write(tmp_path, "base: configs/base.yaml\nmodel: 5\n"))


def test_nested_namespace_null_rejected(tmp_path):
    # A null where a namespace is expected must fail up-front, not defer to build.
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
model:
  contact_head: null
"""))


def test_resolved_config_does_not_alias_defaults(tmp_path):
    # Mutating one resolved config's untouched nested dict must not leak into
    # DEFAULTS (and thus a later load in the same process).
    a = load_config(_write(tmp_path, "base: configs/base.yaml\n"))
    a["contact"]["targets"]["vertex"]["loss"]["focal_alpha"] = 0.123
    b = load_config(REPO / "configs" / "base.yaml")
    assert b["contact"]["targets"]["vertex"]["loss"]["focal_alpha"] == pytest.approx(0.75)
