"""Monitor whitelist follows the configured evaluation split, plus the train-loop
helpers that share the module (resume-identity diffs and gradient clipping)."""
from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]


def _train_module():
    spec = importlib.util.spec_from_file_location("train_mod", REPO / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Stub:
    """Minimal stand-in exposing the attributes ``_validate_monitor`` reads."""

    def __init__(self, monitor, targets, eval_split="val", physics_enabled=False):
        self.monitor = monitor
        self.targets = targets
        self.eval_split = eval_split
        self.physics_enabled = physics_enabled


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


def test_monitor_accepts_physics_residual_when_enabled():
    # The physics-residual pseudo-target is valid only when physics is enabled; it is
    # NOT a contact target (it must never enter the f1-hardcoded summary).
    tm = _train_module()
    tm.Trainer._validate_monitor(
        _Stub("val/physics_residual", ["joint"], physics_enabled=True))  # no raise
    tm.Trainer._validate_monitor(
        _Stub("test/physics_residual", ["joint"], eval_split="test",
              physics_enabled=True))                                     # no raise


def test_monitor_rejects_physics_residual_when_physics_disabled():
    tm = _train_module()
    with pytest.raises(ValueError, match="not a valid metric"):
        tm.Trainer._validate_monitor(_Stub("val/physics_residual", ["joint"]))


# ---------------------------------------------------- resume-identity config diffs

def _resume_base() -> dict:
    return {
        "model": {"a": 1}, "contact": {"b": 2},
        "physics": {"smoothing_kernel": [0.25, 0.5, 0.25]},
        "loss": {"grad_clip": 1.0},
        "data": {"datasets": [], "sequence": {}, "eval_split": "test"},
        "optim": {"lr": 1e-4, "epochs": 20}, "output": {"monitor": "test/physics_residual"},
    }


def test_resume_diff_identical_config_has_no_diffs():
    tm = _train_module()
    base = _resume_base()
    assert tm._resume_config_diffs(base, base) == []


def test_resume_diff_flags_physics_smoothing_kernel():
    # A physics change (e.g. the smoothing kernel) is now identity-defining.
    tm = _train_module()
    base = _resume_base()
    changed = {**base, "physics": {"smoothing_kernel": [1.0]}}
    diffs = tm._resume_config_diffs(base, changed)
    assert any("physics" in d for d in diffs)


def test_resume_diff_flags_loss_grad_clip():
    # The top-level loss section (carrying grad_clip) is now compared.
    tm = _train_module()
    base = _resume_base()
    changed = {**base, "loss": {"grad_clip": 5.0}}
    diffs = tm._resume_config_diffs(base, changed)
    assert any(d.strip().startswith("loss") for d in diffs)


def test_resume_diff_normalizes_historical_physics_configs():
    # A historical saved config that predates residual_robust / max_cam_jump_m (or
    # the whole disabled physics section) must compare equal to a current
    # resolution holding the backward-identical defaults — absent keys and explicit
    # defaults are the same run.
    import copy

    from contact.config import DEFAULTS

    tm = _train_module()
    current = _resume_base()
    current["physics"] = copy.deepcopy(DEFAULTS["physics"])

    saved = copy.deepcopy(current)
    del saved["physics"]["max_cam_jump_m"]
    del saved["physics"]["loss"]["residual_robust"]
    assert tm._resume_config_diffs(saved, current) == []

    no_physics = {k: v for k, v in current.items() if k != "physics"}
    assert tm._resume_config_diffs(no_physics, current) == []

    # A REAL physics difference must still flag after normalization.
    changed = copy.deepcopy(current)
    changed["physics"]["smoothing_kernel"] = [1.0]
    assert any("physics" in d for d in tm._resume_config_diffs(changed, current))


def test_ensure_resume_identity_shared_by_both_resume_paths():
    # The helper the explicit --resume PATH branch now calls (Trainer.__init__)
    # and the auto path already called: silent on identical, RuntimeError on diff.
    tm = _train_module()
    base = _resume_base()
    tm._ensure_resume_identity(base, base, "ctx")            # no raise
    changed = {**base, "loss": {"grad_clip": 5.0}}
    with pytest.raises(RuntimeError, match="identity-defining"):
        tm._ensure_resume_identity(
            base, changed, "--resume out/run/last.pth: the checkpoint's stored")


# ------------------------------------------------------- physics residual headline

def test_physics_residual_headline_zero_mass_semantics():
    # mass > 0 -> exact mean; zero mass -> NaN (never a perfect 0), and RAISES when
    # the physics residual is the monitor (all clips excluded must be loud).
    import math

    tm = _train_module()
    assert tm._physics_residual_headline(3.0, 2.0, required=True) == pytest.approx(1.5)
    assert math.isnan(tm._physics_residual_headline(0.0, 0.0, required=False))
    with pytest.raises(RuntimeError, match="no residual data"):
        tm._physics_residual_headline(0.0, 0.0, required=True)


# --------------------------------------------------------------- gradient clipping

def _grad_stub(model: nn.Module, grad_clip: float):
    return types.SimpleNamespace(model=model, grad_clip=grad_clip,
                                 epoch=0, global_step=0)


def test_clip_grads_clips_large_gradient_to_cap():
    tm = _train_module()
    model = nn.Linear(4, 4)
    for p in model.parameters():
        p.grad = torch.full_like(p, 10.0)                     # norm >> 1
    raw, post = tm.Trainer._clip_grads(_grad_stub(model, grad_clip=1.0))
    assert raw > 1.0                                          # true pre-clip norm measured
    assert raw >= post
    assert post == pytest.approx(1.0)                         # capped at grad_clip


def test_clip_grads_leaves_small_gradient_unchanged():
    tm = _train_module()
    model = nn.Linear(2, 2)
    for p in model.parameters():
        p.grad = torch.full_like(p, 0.01)                     # norm < cap
    raw, post = tm.Trainer._clip_grads(_grad_stub(model, grad_clip=5.0))
    assert raw >= post
    assert raw == pytest.approx(post)                         # below the cap -> untouched


def test_clip_grads_raises_on_nonfinite_gradient():
    # An inf raw norm must raise BEFORE the optimizer step: clip_grad_norm_ has
    # already scaled the grads by clip/inf (NaN), while the capped post value
    # min(inf, clip) would look perfectly finite to a post-clip check.
    tm = _train_module()
    model = nn.Linear(2, 2)
    for p in model.parameters():
        p.grad = torch.full_like(p, float("inf"))
    with pytest.raises(FloatingPointError, match="non-finite raw gradient"):
        tm.Trainer._clip_grads(_grad_stub(model, grad_clip=5.0))
