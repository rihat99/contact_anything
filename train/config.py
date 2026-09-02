"""Run-config loading: a single ``base:`` include, strict keys, cross-key checks.

``configs/base.yaml`` IS the schema: every allowed key appears there with its
default, a mapping value marks a namespace, anything else is a leaf whose
concrete value is not type-checked. :func:`load_config` reads a run yaml,
splices in the file named by its ``base:`` key (path relative to the repo
root), deep-merges child over base, rejects any key the schema does not
define, and runs the handful of cross-key checks that pure key validation
cannot express.

:func:`signal_needs` derives which optional dataset signal groups the run
must load (``forces``/``motion``/``pose``/``keypoints``/``smplx``) from the
enabled losses — that is never configured directly.
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
_MONITOR_MIN = ("mae", "err", "loss", "residual", "rmse", "mpjpe", "pve", "accel")
#: Tensorboard metric section of every loss (``metric_<group>/...``); a loss
#: whose group is not its own name is listed here.
METRIC_GROUPS = {"smplx": "pose"}


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
    contact = cfg["model"]["contact"]
    return {
        "pose": True,
        "contact": bool(contact["enabled"]) and contact["source"] == "tokens",
        "force": bool(cfg["model"]["force"]["enabled"]),
        "motion": bool(cfg["model"]["motion"]["enabled"]),
    }


def enabled_losses(cfg: dict) -> list[str]:
    """Names of the enabled losses, in :func:`model.loss.build_losses` order."""
    sections = (("contact", "contact_supervision"), ("force", "force_supervision"),
                ("motion", "motion_supervision"), ("pose", "pose_supervision"),
                ("keypoint", "keypoint_supervision"),
                ("smplx", "smplx_supervision"),
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
    if cfg["smplx_supervision"]["enabled"]:
        needs.add("smplx")
    return needs


def validate(cfg: dict) -> None:
    """Cross-key checks: branch/loss/modality coherence and the monitor tag."""
    model = cfg["model"]
    branches = enabled_branches(cfg)
    cross_modal = model["cross_modal_temporal"]
    modalities = list(cross_modal["modalities"]) if cross_modal["enabled"] else []

    if model["contact"]["enabled"] and model["contact"]["source"] not in (
            "tokens", "pose_token"):
        raise ValueError(
            "model.contact.source must be 'tokens' or 'pose_token'; got "
            f"{model['contact']['source']!r}")
    if cross_modal["enabled"]:
        if len(set(modalities)) != len(modalities) or any(
                m not in _MODALITY_ORDER for m in modalities):
            raise ValueError(
                "model.cross_modal_temporal.modalities must be a duplicate-free "
                f"subset of {list(_MODALITY_ORDER)}; got {modalities!r}")
        if not modalities:
            raise ValueError(
                "model.cross_modal_temporal.modalities needs >= 1 entry; got []")
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
    smplx_sup = cfg["smplx_supervision"]["enabled"]
    if pose_writers and not (pose_sup or kp_sup or smplx_sup):
        raise ValueError(
            f"{', '.join(pose_writers)} write(s) the pose readout but none of "
            "pose_supervision / keypoint_supervision / smplx_supervision is "
            "enabled — nothing would train the written pose")
    if model["smplx"]["enabled"]:
        # The SMPL-X head replaces the MHR readout (its recompute is skipped),
        # so nothing may consume a written MHR pose in that build.
        mhr_consumers = [
            name for name, on in (
                ("pose_supervision", pose_sup), ("keypoint_supervision", kp_sup),
                ("model.finetune_pose_head", model["finetune_pose_head"]),
                ("model.finetune_camera_head", model["finetune_camera_head"]),
                ("physics", cfg["physics"]["enabled"]),
            ) if on]
        if mhr_consumers:
            raise ValueError(
                f"model.smplx.enabled is exclusive with {', '.join(mhr_consumers)}: "
                "the SMPL-X head is the pose output and the MHR readout is not "
                "recomputed")
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

    optim = cfg["optim"]
    betas = optim["betas"]
    if len(betas) != 2 or not all(0.0 <= float(b) < 1.0 for b in betas):
        raise ValueError(f"optim.betas must be two values in [0, 1); got {betas}")
    if not 0.0 <= float(optim["ema"]) < 1.0:
        raise ValueError(f"optim.ema must lie in [0, 1); got {optim['ema']}")
    if int(optim["warmup_steps"]) < 0:
        raise ValueError("optim.warmup_steps must be >= 0")
    if cfg["contact_supervision"]["enabled"]:
        if not model["contact"]["enabled"]:
            raise ValueError(
                "contact_supervision.enabled requires model.contact.enabled")
        if cfg["contact_supervision"]["criterion"] not in ("bce", "focal"):
            raise ValueError(
                "contact_supervision.criterion must be 'bce' or 'focal'; got "
                f"{cfg['contact_supervision']['criterion']!r}")
        cs = cfg["contact_supervision"]
        if float(cs["neg_weight"]) <= 0 or any(float(w) <= 0 for w in cs["pos_weight"]):
            raise ValueError("contact_supervision.neg_weight / pos_weight must be positive")
        if len(cs["pos_weight"]) != 6:
            raise ValueError(
                f"contact_supervision.pos_weight needs six factors (kindyn_6 order); "
                f"got {len(cs['pos_weight'])}")
        if int(cs["transition_tolerance"]) < 0:
            raise ValueError("contact_supervision.transition_tolerance must be >= 0")
    if cfg["smplx_supervision"]["enabled"] and not model["smplx"]["enabled"]:
        raise ValueError("smplx_supervision.enabled requires model.smplx.enabled")
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
    groups = sorted({METRIC_GROUPS.get(name, name) for name in enabled_losses(cfg)})
    parts = monitor.split("/")
    if monitor != "loss_test/total" and (
            len(parts) != 2 or not parts[0].startswith("metric_")
            or parts[0][len("metric_"):] not in groups):
        raise ValueError(
            f"output.monitor {monitor!r} must be 'loss_test/total' or "
            f"'metric_<group>/<name>' with <group> one of {groups}")
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
