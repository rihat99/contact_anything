"""Monitor whitelist follows the configured evaluation split."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _train_module():
    spec = importlib.util.spec_from_file_location("train_mod", REPO / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Stub:
    """Minimal stand-in exposing the two attributes ``_validate_monitor`` reads."""

    def __init__(self, monitor, targets, eval_split="val"):
        self.monitor = monitor
        self.targets = targets
        self.eval_split = eval_split


def test_monitor_accepts_loss_and_target_metrics():
    tm = _train_module()
    for mon in ("val/loss", "val/joint_f1", "val/vertex_iou", "val/joint_precision",
                "val/vertex_recall", "val/joint_f2", "val/joint_accuracy"):
        tm.Trainer._validate_monitor(_Stub(mon, ["vertex", "joint"]))   # no raise

    for mon in ("test/loss", "test/joint_f1", "test/joint_precision"):
        tm.Trainer._validate_monitor(
            _Stub(mon, ["joint"], eval_split="test"))


def test_monitor_rejects_wrong_evaluation_prefix():
    tm = _train_module()
    with pytest.raises(ValueError, match="not a valid metric"):
        tm.Trainer._validate_monitor(
            _Stub("val/joint_f1", ["joint"], eval_split="test"))


def test_monitor_rejects_target_loss_name():
    # 'val/joint_loss' used to pass validation then crash in _monitor_value.
    tm = _train_module()
    with pytest.raises(ValueError, match="not a valid metric"):
        tm.Trainer._validate_monitor(_Stub("val/joint_loss", ["joint"]))


def test_monitor_rejects_disabled_target():
    tm = _train_module()
    with pytest.raises(ValueError, match="not a valid metric"):
        tm.Trainer._validate_monitor(_Stub("val/vertex_f1", ["joint"]))


def test_monitor_rejects_unknown_metric():
    tm = _train_module()
    with pytest.raises(ValueError, match="not a valid metric"):
        tm.Trainer._validate_monitor(_Stub("val/joint_dice", ["joint"]))
