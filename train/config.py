"""Run-config loading: a single ``base:`` include, strict keys, cross-key checks.

``configs/base.yaml`` IS the schema: every allowed key appears there with its
default, a mapping value marks a namespace, anything else is a leaf whose
concrete value is not type-checked. :func:`load_config` reads a run yaml,
splices in the file named by its ``base:`` key (path relative to the repo
root), deep-merges child over base, rejects any key the schema does not
define, and runs the handful of cross-key checks that pure key validation
cannot express.

:func:`signal_needs` derives which optional dataset signal groups the run
must load (``forces`` / ``smplx``) from the enabled losses — that is never
configured directly.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "configs" / "base.yaml"

_MODALITY_ORDER = ("pose", "contact", "force")
_REFINER_OUTPUTS = ("pose", "contact", "motion", "force")
_MONITOR_MAX = ("f1", "iou", "precision", "recall", "pearson")
_MONITOR_MIN = ("mae", "err", "loss", "mpjpe", "pve", "accel", "rte", "jitter", "bias",
                "mag", "dlogz", "rmse")
#: Tensorboard metric section of every loss (``metric_<group>/...``); a loss
#: whose group is not its own name is listed here.
METRIC_GROUPS = {"smplx": "pose"}
#: MHR70 anchors the six kindyn groups are supervised at (contact AND force).
NUM_GROUPS = 6


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


def enabled_losses(cfg: dict) -> list[str]:
    """Names of the enabled losses, in :func:`model.loss.build_losses` order."""
    sections = (("contact", "contact_supervision"), ("force", "force_supervision"),
                ("smplx", "smplx_supervision"), ("motion", "motion_supervision"))
    return [name for name, section in sections if cfg[section]["enabled"]]


def signal_needs(cfg: dict) -> set[str]:
    """Optional dataset signal groups the enabled losses require."""
    needs: set[str] = set()
    if cfg["force_supervision"]["enabled"]:
        needs.add("forces")
    if cfg["smplx_supervision"]["enabled"] or cfg["motion_supervision"]["enabled"]:
        needs.add("smplx")
    if "force" in refiner_outputs(cfg):
        needs.add("smplx")          # the kindyn root rotation re-frames the force GT
    return needs


def refiner_outputs(cfg: dict) -> set[str]:
    """The refiner's output heads (empty when ``model.refiner`` is off)."""
    refiner = cfg["model"]["refiner"]
    return {str(o) for o in refiner["outputs"]} if refiner["enabled"] else set()


def _check_anchors(section: str, indices) -> None:
    indices = [int(i) for i in indices]
    if len(indices) != NUM_GROUPS or len(set(indices)) != NUM_GROUPS or any(
            not 0 <= i < 70 for i in indices):
        raise ValueError(
            f"{section}.keypoint_indices must be {NUM_GROUPS} distinct MHR70 indices "
            f"(one per kindyn group, in group order); got {indices}")


def validate(cfg: dict) -> None:
    """Cross-key checks: branch/loss/modality coherence and the monitor tag."""
    model = cfg["model"]
    contact_on = bool(model["contact"]["enabled"])
    force_on = bool(model["force"]["enabled"])
    smplx = model["smplx"]
    if contact_on:
        _check_anchors("model.contact", model["contact"]["keypoint_indices"])
    if force_on:
        _check_anchors("model.force", model["force"]["keypoint_indices"])

    cross_modal = model["cross_modal_temporal"]
    modalities = [str(m) for m in cross_modal["modalities"]] if cross_modal["enabled"] else []
    if cross_modal["enabled"]:
        if not modalities or len(set(modalities)) != len(modalities) or any(
                m not in _MODALITY_ORDER for m in modalities):
            raise ValueError(
                "model.cross_modal_temporal.modalities must be a non-empty duplicate-free "
                f"subset of {list(_MODALITY_ORDER)}; got {modalities!r}")
        blocks = {"pose": True, "contact": contact_on, "force": force_on}
        missing = [m for m in modalities if not blocks[m]]
        if missing:
            raise ValueError(
                f"model.cross_modal_temporal.modalities {missing} have no token block in "
                "this build — enable model.contact / model.force accordingly")
        if cross_modal["window"] is not None and not float(cross_modal["window"]) > 0.0:
            raise ValueError("model.cross_modal_temporal.window must be positive or null")
        if not float(cross_modal["time_scale"]) > 0.0:
            raise ValueError("model.cross_modal_temporal.time_scale must be positive")
        if "pose" in modalities and not smplx["enabled"]:
            raise ValueError(
                "model.cross_modal_temporal writes the pose token ('pose' listed) but "
                "nothing reads it — enable model.smplx")

    if smplx["camera"] not in ("cliff", "ray"):
        raise ValueError(f"model.smplx.camera must be 'cliff' or 'ray'; got {smplx['camera']!r}")
    sup = cfg["smplx_supervision"]
    if sup["kp2d_space"] not in ("crop", "image"):
        raise ValueError(
            f"smplx_supervision.kp2d_space must be 'crop' or 'image'; got {sup['kp2d_space']!r}")
    if sup["enabled"]:
        if not smplx["enabled"]:
            raise ValueError("smplx_supervision.enabled requires model.smplx.enabled")
        if float(sup["loss"]["hand_pose"]) > 0 and not smplx["hands"]:
            raise ValueError(
                "smplx_supervision.loss.hand_pose > 0 requires model.smplx.hands (the head "
                "regresses no finger rotations otherwise)")
        if smplx["camera"] == "ray" and float(sup["loss"]["cam"]) > 0.0:
            raise ValueError(
                "smplx_supervision.loss.cam supervises the CLIFF (s, tx, ty) proxy, which "
                "model.smplx.camera: ray does not produce — set it to 0")
    if smplx["frozen"] and not smplx["enabled"]:
        raise ValueError("model.smplx.frozen requires model.smplx.enabled")
    if smplx["checkpoint"] is not None and not smplx["enabled"]:
        raise ValueError("model.smplx.checkpoint requires model.smplx.enabled")
    if smplx["frozen"] and smplx["checkpoint"] is None:
        raise ValueError(
            "model.smplx.frozen without model.smplx.checkpoint would freeze a random head")

    refiner = model["refiner"]
    outputs = refiner_outputs(cfg)
    if refiner["enabled"]:
        listed = [str(o) for o in refiner["outputs"]]
        if not listed or len(set(listed)) != len(listed) or any(
                o not in _REFINER_OUTPUTS for o in listed):
            raise ValueError(
                "model.refiner.outputs must be a non-empty duplicate-free subset of "
                f"{list(_REFINER_OUTPUTS)}; got {listed!r}")
        if not smplx["enabled"]:
            raise ValueError("model.refiner needs the per-frame body: enable model.smplx")
        if not smplx["frozen"]:
            raise ValueError(
                "model.refiner needs a FROZEN per-frame body (model.smplx.frozen with a "
                "stage-1 checkpoint): a trainable pose path under the motion / force losses "
                "is the shrinkage shortcut")
        if int(refiner["num_layers"]) < 1:
            raise ValueError("model.refiner.num_layers must be >= 1")
        if cross_modal["enabled"]:
            raise ValueError(
                "model.refiner IS the temporal model — disable model.cross_modal_temporal")
        if "force" in outputs and force_on:
            raise ValueError(
                "model.refiner.outputs lists force AND model.force is enabled: two force heads")
        if not float(refiner["window"]) > 0.0:
            raise ValueError("model.refiner.window must be positive")
        if not float(refiner["time_scale"]) > 0.0:
            raise ValueError("model.refiner.time_scale must be positive")
        if float(refiner["depth_smooth_sec"]) < 0.0:
            raise ValueError("model.refiner.depth_smooth_sec must be >= 0")
        if int(refiner["dim"]) % int(refiner["num_heads"]) != 0:
            raise ValueError("model.refiner.dim must be divisible by num_heads")
        # Every head must receive a loss (DDP runs with find_unused_parameters=False).
        needs = {"pose": "smplx_supervision", "contact": "contact_supervision",
                 "motion": "motion_supervision", "force": "force_supervision"}
        for output in sorted(outputs):
            if not cfg[needs[output]]["enabled"]:
                raise ValueError(
                    f"model.refiner.outputs lists {output!r} but {needs[output]} is disabled")
        if sup["enabled"] and float(sup["loss"]["cam"]) > 0.0:
            raise ValueError(
                "smplx_supervision.loss.cam supervises the CLIFF proxy, which the refined "
                "body does not carry — set it to 0 under model.refiner")

    if cfg["contact_supervision"]["enabled"] and not (contact_on or "contact" in outputs):
        raise ValueError(
            "contact_supervision.enabled requires model.contact.enabled or a refiner "
            "'contact' output")
    if cfg["force_supervision"]["enabled"] and not (force_on or "force" in outputs):
        raise ValueError(
            "force_supervision.enabled requires model.force.enabled or a refiner 'force' output")
    # A decoder-level head with no loss never receives a gradient (DDP hard-errors).
    if contact_on and "contact" not in outputs and not cfg["contact_supervision"]["enabled"]:
        raise ValueError(
            "model.contact.enabled builds a decoder contact head: enable contact_supervision "
            "or give the refiner the 'contact' output")
    if force_on and not cfg["force_supervision"]["enabled"]:
        raise ValueError("model.force.enabled builds a force head: enable force_supervision")
    motion = cfg["motion_supervision"]
    if motion["enabled"]:
        if "motion" not in outputs:
            raise ValueError("motion_supervision.enabled requires a refiner 'motion' output")
        if float(motion["label_smooth_sec"]) < 0.0:
            raise ValueError("motion_supervision.label_smooth_sec must be >= 0")
        if any(float(v) <= 0.0 for v in motion["scale"].values()):
            raise ValueError("motion_supervision.scale values must be positive")
        weights = {k: float(v) for k, v in motion["loss"].items() if k != "huber_delta"}
        if any(w < 0.0 for w in weights.values()) or not any(w > 0.0 for w in weights.values()):
            raise ValueError("motion_supervision.loss weights must be >= 0 with at least one > 0")
        if int(cfg["data"]["clip"]["frames"]) < 5:
            raise ValueError(
                "motion_supervision needs data.clip.frames >= 5 (acceleration rows need two "
                "valid neighbours on each side)")

    optim = cfg["optim"]
    betas = optim["betas"]
    if len(betas) != 2 or not all(0.0 <= float(b) < 1.0 for b in betas):
        raise ValueError(f"optim.betas must be two values in [0, 1); got {betas}")
    if not 0.0 <= float(optim["ema"]) < 1.0:
        raise ValueError(f"optim.ema must lie in [0, 1); got {optim['ema']}")
    if int(optim["warmup_steps"]) < 0:
        raise ValueError("optim.warmup_steps must be >= 0")

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
