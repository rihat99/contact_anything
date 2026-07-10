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
    assert joint["contact"]["targets"]["joint"]["supervise_subset"] is None
    assert 41 in joint["model"]["contact_head"]["contact_keypoint_indices"]
    assert 62 in joint["model"]["contact_head"]["contact_keypoint_indices"]


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
