"""Run-config loading: a single ``base:`` include, strict keys, cross-key checks.

``configs/base.yaml`` IS the schema: every allowed key appears there with its
default, a mapping value marks a namespace, anything else is a leaf whose
concrete value is not type-checked. :func:`load_config` reads a run yaml,
splices in the file named by its ``base:`` key (path relative to the repo
root), deep-merges child over base, rejects any key the schema does not
define, and runs the handful of cross-key checks that pure key validation
cannot express.

:func:`signal_needs` derives which optional dataset signal groups the run
must load (``forces``/``motion``/``pose``/``keypoints``) from the enabled
losses — that is never configured directly.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "configs" / "base.yaml"

_MODALITY_ORDER = ("pose", "contact", "force", "motion")
_MONITOR_MAX = ("f1", "f2", "iou", "r3d", "precision", "recall", "accuracy")
_MONITOR_MIN = ("mae", "err", "loss", "residual", "rmse")


def _deep_merge(base: dict, override: dict) -> dict:
    """``base`` with ``override`` recursively applied; both sides deep-copied."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_raw(path: Path) -> dict:
    """Read one yaml and splice in its ``base:`` include (child over base)."""
    raw = yaml.safe_load(path.read_text()) or {}
    base = raw.get("base")
    if base:
        raw = _deep_merge(_load_raw(REPO_ROOT / base), raw)
    return raw


def _validate_keys(node: dict, schema: dict, path: str = "") -> None:
    """Reject any key absent from ``schema`` or any namespace given a scalar."""
    for key, value in node.items():
        dotted = f"{path}.{key}" if path else key
        if key not in schema:
            raise ValueError(f"unknown config key: {dotted!r}")
        sub = schema[key]
        if isinstance(sub, dict):
            if not isinstance(value, dict):
                raise ValueError(
                    f"config key {dotted!r} must be a mapping (namespace); "
                    f"got {type(value).__name__}")
            _validate_keys(value, sub, dotted)


def monitor_mode(monitor: str) -> str:
    """``"max"`` or ``"min"`` for a metric tag, by the suffix rule in base.yaml."""
    tail = monitor.rsplit("/", 1)[-1]
    if any(token in tail for token in _MONITOR_MAX):
        return "max"
    if any(token in tail for token in _MONITOR_MIN):
        return "min"
    raise ValueError(
        f"output.monitor {monitor!r}: cannot tell max from min — the metric name "
        f"must contain one of {_MONITOR_MAX} (max) or {_MONITOR_MIN} (min)")


def enabled_branches(cfg: dict) -> dict:
    """Which modality token blocks the configured build creates."""
    return {
        "pose": True,
        "contact": bool(cfg["model"]["contact"]["enabled"]),
        "force": bool(cfg["model"]["force"]["enabled"]),
        "motion": bool(cfg["model"]["motion"]["enabled"]),
    }


def enabled_losses(cfg: dict) -> list[str]:
    """Names of the enabled losses, in :func:`model.loss.build_losses` order."""
    sections = (("contact", "contact_supervision"), ("force", "force_supervision"),
                ("motion", "motion_supervision"), ("pose", "pose_supervision"),
                ("keypoint", "keypoint_supervision"),
                ("contact_consistency", "contact_consistency"),
                ("force_consistency", "force_consistency"), ("physics", "physics"))
    return [name for name, section in sections if cfg[section]["enabled"]]


def signal_needs(cfg: dict) -> set[str]:
    """Optional dataset signal groups the enabled losses require."""
    physics = cfg["physics"]["enabled"]
    needs: set[str] = set()
    if (cfg["force_supervision"]["enabled"] or physics
            or cfg["force_consistency"]["enabled"]):
        needs.add("forces")
    if (cfg["motion_supervision"]["enabled"] or cfg["force_consistency"]["enabled"]
            or (physics and cfg["model"]["force"]["frame"] == "root")):
        needs.add("motion")           # the kindyn world-from-root rotation
    if cfg["pose_supervision"]["enabled"]:
        needs.add("pose")
    if cfg["keypoint_supervision"]["enabled"]:
        needs.add("keypoints")
    return needs


def validate(cfg: dict) -> None:
    """Cross-key checks: branch/loss/modality coherence and the monitor tag."""
    model = cfg["model"]
    branches = enabled_branches(cfg)
    cross_modal = model["cross_modal_temporal"]
    modalities = list(cross_modal["modalities"]) if cross_modal["enabled"] else []

    if cross_modal["enabled"]:
        if len(set(modalities)) != len(modalities) or any(
                m not in _MODALITY_ORDER for m in modalities):
            raise ValueError(
                "model.cross_modal_temporal.modalities must be a duplicate-free "
                f"subset of {list(_MODALITY_ORDER)}; got {modalities!r}")
        if len(modalities) < 2:
            raise ValueError(
                "model.cross_modal_temporal.modalities needs >= 2 entries "
                f"(there is nothing to mix otherwise); got {modalities!r}")
        missing = [m for m in modalities if not branches[m]]
        if missing:
            raise ValueError(
                f"model.cross_modal_temporal.modalities {missing} have no token "
                "block in this build — enable model.contact / model.force / "
                "model.motion accordingly")

    pose_sup = cfg["pose_supervision"]["enabled"]
    kp_sup = cfg["keypoint_supervision"]["enabled"]
    pose_writers = [
        name for name, on in (
            ("model.cross_modal_temporal.modalities['pose']", "pose" in modalities),
            ("model.pose_temporal", model["pose_temporal"]["enabled"]),
            ("model.finetune_pose_head", model["finetune_pose_head"]),
            ("model.finetune_camera_head", model["finetune_camera_head"]),
        ) if on
    ]
    if pose_writers and not (pose_sup or kp_sup):
        raise ValueError(
            f"{', '.join(pose_writers)} write(s) the pose readout but neither "
            "pose_supervision nor keypoint_supervision is enabled — nothing "
            "would train the written pose")
    if model["finetune_camera_head"] and not kp_sup:
        raise ValueError(
            "model.finetune_camera_head requires keypoint_supervision.enabled "
            "(kp2d is the only loss that constrains the camera)")

    if cfg["motion_supervision"]["enabled"]:
        if not branches["motion"]:
            raise ValueError(
                "motion_supervision.enabled requires model.motion.enabled")
        if "motion" not in modalities:
            raise ValueError(
                "motion_supervision.enabled requires 'motion' in "
                "model.cross_modal_temporal.modalities — a per-frame head cannot "
                "represent a derivative")

    if model["force"]["contact_gate"]["enabled"]:
        if not branches["contact"]:
            raise ValueError(
                "model.force.contact_gate.enabled requires model.contact.enabled "
                "(each force group is gated by its own aligned contact output)")
        if len(model["contact"]["keypoint_indices"]) != 6:
            raise ValueError(
                "model.force.contact_gate.enabled requires the six kindyn_6 "
                "contact anchors; got "
                f"{model['contact']['keypoint_indices']!r}")

    if cfg["contact_supervision"]["enabled"] and not branches["contact"]:
        raise ValueError(
            "contact_supervision.enabled requires model.contact.enabled")
    if cfg["force_supervision"]["enabled"] and not branches["force"]:
        raise ValueError(
            "force_supervision.enabled requires model.force.enabled")
    if cfg["force_consistency"]["enabled"] and not (
            cfg["force_supervision"]["enabled"] and cfg["motion_supervision"]["enabled"]):
        raise ValueError(
            "force_consistency.enabled requires force_supervision.enabled (bw forces "
            "in the kindyn root frame) and motion_supervision.enabled (the GT root "
            "rotation)")
    if cfg["physics"]["enabled"]:
        if not branches["force"]:
            raise ValueError("physics.enabled requires model.force.enabled")
        if cfg["force_supervision"]["enabled"]:
            raise ValueError(
                "physics.enabled and force_supervision.enabled are mutually "
                "exclusive supervision regimes for the same force output")

    monitor = str(cfg["output"]["monitor"])
    parts = monitor.split("/")
    if monitor != "test/loss" and (
            len(parts) != 3 or parts[0] != "test" or parts[1] not in enabled_losses(cfg)):
        raise ValueError(
            f"output.monitor {monitor!r} must be 'test/loss' or 'test/<loss>/<metric>' "
            f"with <loss> one of the enabled losses {enabled_losses(cfg)}")
    monitor_mode(monitor)


def load_config(path: str | Path) -> dict:
    """Load, merge, key-validate and cross-validate a run config.

    :param path: run yaml (its ``base:`` include is resolved against the repo root).
    :returns: the resolved config as a plain ``dict``.
    :raises ValueError: on an unknown key or a failed cross-key check.
    """
    schema: dict[str, Any] = yaml.safe_load(SCHEMA_PATH.read_text())
    cfg = _deep_merge(schema, _load_raw(Path(path)))
    _validate_keys(cfg, schema)
    validate(cfg)
    return cfg
