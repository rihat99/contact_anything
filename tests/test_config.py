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
    assert cfg["data"]["sequence"]["target_frame"] == "all"
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
        "window_frames": None,
    }
    assert temporal["data"]["sequence"]["frames_per_clip"] == 5
    assert temporal["data"]["sequence"]["frame_stride"] == 1
    assert temporal["data"]["sequence"]["target_frame"] == "all"
    assert temporal["data"]["frames_per_batch"] == 60
    assert temporal["optim"]["lr"] == pytest.approx(3.75e-4)
    assert temporal["model"]["init_contact_checkpoint"] is None
    assert temporal["output"]["exp_name"] == "climb4_t5"

    center = load_config(REPO / "configs" / "climbing_videos_joint_temporal_center.yaml")
    assert center["model"]["temporal"]["enabled"] is True
    assert center["model"]["temporal"]["causal"] is False
    assert center["data"]["sequence"]["frames_per_clip"] == 5
    assert center["data"]["sequence"]["target_frame"] == "center"
    assert center["model"]["init_contact_checkpoint"] is None
    assert center["output"]["exp_name"] == "climb4_t5mid"


def test_force_warmstart_temporal_launch_config():
    cfg = load_config(REPO / "configs" / "climbing_videos_force_warmstart_temporal.yaml")
    assert cfg["model"]["init_contact_checkpoint"] == (
        "output/climb4_frame_20260716_122726/best.pth"
    )
    assert cfg["model"]["temporal"]["enabled"] is False
    assert cfg["model"]["force_temporal"]["enabled"] is True
    assert cfg["physics"]["use_warp"] is True
    assert cfg["data"]["eval_split"] == "test"
    assert cfg["output"]["monitor"] == "test/physics_residual"


def test_force_warmstart_t16_launch_config():
    cfg = load_config(REPO / "configs" / "climbing_videos_force_warmstart_t16.yaml")
    # Regime (a), TEMPORAL contact source (climb4_t5, measured F1 0.8878 at
    # T=16/stride 2), force-only training.
    assert cfg["train"]["freeze_contact"] is True
    assert cfg["model"]["force_head"]["enabled"] is True
    assert cfg["model"]["init_contact_checkpoint"] == (
        "output/climb4_t5_20260716_183709/best.pth")
    # The contact temporal block must byte-match the source run's architecture
    # (configs/climbing_videos_joint_temporal.yaml) or the warm start hard-fails.
    source = load_config(REPO / "configs" / "climbing_videos_joint_temporal.yaml")
    assert cfg["model"]["temporal"] == source["model"]["temporal"]
    assert cfg["model"]["temporal"]["enabled"] is True
    assert cfg["model"]["force_temporal"] == {
        "enabled": True, "bottleneck_dim": 256, "num_layers": 1, "num_heads": 4,
        "mlp_ratio": 2.0, "attend": "per_token", "causal": False, "dropout": 0.0,
        "position_scale": 30.0}
    # Decisive-run knobs: 16-frame clips at stride 2 -> 10 residual frames.
    assert cfg["data"]["sequence"] == {
        "frames_per_clip": 16, "frame_stride": 2, "jitter": True, "target_frame": "all"}
    assert cfg["data"]["frames_per_batch"] == 32
    # Robust residual + jerk filter + relaxed regularisers.
    assert cfg["physics"]["loss"]["residual_robust"] == {
        "kind": "pseudo_huber", "delta_force": 1.0, "delta_torque": 0.5}
    assert cfg["physics"]["max_cam_jump_m"] == pytest.approx(0.5)
    assert cfg["physics"]["loss"]["force_noncontact"] == pytest.approx(2.0)
    assert cfg["physics"]["loss"]["force_at_contact"] == 0.0
    assert cfg["loss"]["grad_clip"] == pytest.approx(5.0)
    assert cfg["optim"] == {
        "lr": pytest.approx(1.0e-4), "weight_decay": pytest.approx(1.0e-4),
        "epochs": 30, "warmup_epochs": 1, "lr_min": pytest.approx(1.0e-6)}
    assert cfg["output"]["exp_name"] == "climb4_force_t16"
    assert cfg["output"]["monitor"] == "test/physics_residual"


def test_residual_robust_defaults_are_square():
    cfg = load_config(REPO / "configs" / "base.yaml")
    assert cfg["physics"]["loss"]["residual_robust"] == {
        "kind": "square", "delta_force": 1.0, "delta_torque": 0.5}
    assert cfg["physics"]["max_cam_jump_m"] is None


def test_residual_robust_bad_kind_rejected(tmp_path):
    with pytest.raises(ValueError, match="residual_robust.kind must be one of"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
physics:
  loss:
    residual_robust: {kind: welsch}
"""))


def test_residual_robust_nonpositive_delta_rejected(tmp_path):
    with pytest.raises(ValueError, match="residual_robust.delta_force must be finite and positive"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
physics:
  loss:
    residual_robust: {delta_force: 0.0}
"""))


def test_noncontact_gate_defaults_are_soft_l2():
    cfg = load_config(REPO / "configs" / "base.yaml")
    assert cfg["physics"]["loss"]["noncontact_gate"] == {
        "kind": "soft_l2", "p_lo": 0.2, "p_hi": 0.5}


def test_noncontact_gate_bad_kind_rejected(tmp_path):
    with pytest.raises(ValueError, match="noncontact_gate.kind must be one of"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
physics:
  loss:
    noncontact_gate: {kind: hard_step}
"""))


def test_noncontact_gate_bad_thresholds_rejected(tmp_path):
    with pytest.raises(ValueError, match="must satisfy 0 <= p_lo < p_hi <= 1"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
physics:
  loss:
    noncontact_gate: {kind: hinge_l1, p_lo: 0.5, p_hi: 0.5}
"""))


def test_max_cam_jump_m_nonpositive_rejected(tmp_path):
    with pytest.raises(ValueError, match="max_cam_jump_m must be null or a finite positive"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
physics:
  max_cam_jump_m: -1.0
"""))


def test_physics_residual_monitor_requires_residual_weight(tmp_path):
    # The physics_residual monitor reads the raw RNEA residual, computed only when
    # the residual objective runs — a zero weight would starve the monitor forever.
    with pytest.raises(ValueError, match="physics_residual.*requires physics.loss.residual > 0"):
        load_config(_write(tmp_path, """
base: configs/climbing_videos_force_warmstart_t16.yaml
physics:
  loss:
    residual: 0.0
"""))


def test_center_target_frame_requires_odd_clip_length(tmp_path):
    with pytest.raises(ValueError, match="requires an odd frames_per_clip"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
data:
  sequence:
    frames_per_clip: 4
    target_frame: center
"""))


def test_unknown_target_frame_rejected(tmp_path):
    with pytest.raises(ValueError, match="target_frame must be 'all' or 'center'"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
data:
  sequence:
    target_frame: almost_middle
"""))


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


_FORCE_JOINT = """
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
"""


def test_force_head_defaults_load():
    cfg = load_config(REPO / "configs" / "base.yaml")
    assert cfg["model"]["force_head"] == {
        "enabled": False,
        "force_keypoint_indices": None,
        "frame": "local_world_aligned",
        "mlp_depth": 2,
        "mlp_channel_div_factor": 4,
        "dropout": 0.0,
    }
    assert cfg["train"]["freeze_contact"] is False


def test_force_head_on_extremities_per_token_is_accepted(tmp_path):
    cfg = load_config(_write(tmp_path, _FORCE_JOINT))
    assert cfg["model"]["force_head"]["enabled"] is True
    assert cfg["model"]["force_head"]["frame"] == "local_world_aligned"


def test_force_head_requires_extremities_per_token_joint_target(tmp_path):
    with pytest.raises(ValueError, match="extremities_4.*per_token"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
model:
  force_head: {enabled: true}
"""))


def test_force_head_bad_frame_rejected(tmp_path):
    with pytest.raises(ValueError, match="force_head.frame must be one of"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
model:
  force_head: {frame: world}
"""))


def test_force_temporal_defaults_load():
    cfg = load_config(REPO / "configs" / "base.yaml")
    assert cfg["model"]["force_temporal"] == {
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


_FORCE_TEMPORAL = """
base: configs/base.yaml
model:
  contact_head: {contact_keypoint_indices: [62, 41, 13, 14], num_global_tokens: 0,
                 pool_mode: per_token}
  force_head: {enabled: true}
  force_temporal: {enabled: true%(extra)s}
contact:
  primary_target: joint
  targets:
    vertex: {enabled: false}
    joint: {enabled: true, joint_set: extremities_4, supervise_subset: null}
"""


def test_force_temporal_on_extremities_is_accepted(tmp_path):
    cfg = load_config(_write(tmp_path, _FORCE_TEMPORAL % {"extra": ""}))
    assert cfg["model"]["force_temporal"]["enabled"] is True
    assert cfg["model"]["force_temporal"]["attend"] == "per_token"


def test_force_temporal_requires_force_head_enabled(tmp_path):
    with pytest.raises(ValueError, match="force_temporal.enabled requires model.force_head.enabled"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
model:
  force_temporal: {enabled: true}
"""))


def test_force_temporal_bad_divisibility_rejected(tmp_path):
    with pytest.raises(ValueError, match="model.force_temporal.bottleneck_dim must be divisible"):
        load_config(_write(tmp_path, _FORCE_TEMPORAL % {"extra": ", bottleneck_dim: 256, num_heads: 7"}))


def test_freeze_contact_requires_force_enabled(tmp_path):
    with pytest.raises(ValueError, match="freeze_contact.*force_head.enabled"):
        load_config(_write(tmp_path, """
base: configs/base.yaml
train: {freeze_contact: true}
"""))


def test_freeze_contact_requires_init_contact_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="freeze_contact.*init_contact_checkpoint"):
        load_config(_write(tmp_path, _FORCE_JOINT + "train: {freeze_contact: true}\n"))


def test_freeze_contact_with_init_checkpoint_is_accepted(tmp_path):
    cfg = load_config(_write(tmp_path, """
base: configs/base.yaml
model:
  init_contact_checkpoint: /tmp/frame.pth
  contact_head: {contact_keypoint_indices: [62, 41, 13, 14], num_global_tokens: 0,
                 pool_mode: per_token}
  force_head: {enabled: true}
contact:
  primary_target: joint
  targets:
    vertex: {enabled: false}
    joint: {enabled: true, joint_set: extremities_4, supervise_subset: null}
train: {freeze_contact: true}
"""))
    assert cfg["train"]["freeze_contact"] is True
    assert cfg["model"]["init_contact_checkpoint"] == "/tmp/frame.pth"


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
    with pytest.raises(ValueError, match="single climbing_videos or"):
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


def test_temporal_window_frames_defaults_null():
    cfg = load_config(REPO / "configs" / "base.yaml")
    assert cfg["model"]["temporal"]["window_frames"] is None


def test_temporal_window_frames_accepts_odd(tmp_path):
    cfg = load_config(_write(tmp_path, """
base: configs/base.yaml
model:
  temporal:
    window_frames: 5
"""))
    assert cfg["model"]["temporal"]["window_frames"] == 5


@pytest.mark.parametrize("bad", [4, 2, 1])
def test_temporal_window_frames_rejects_non_odd_ge_3(tmp_path, bad):
    with pytest.raises(ValueError, match="window_frames must be null or an odd int >= 3"):
        load_config(_write(tmp_path, f"""
base: configs/base.yaml
model:
  temporal:
    window_frames: {bad}
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
