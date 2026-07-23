"""Checkpoint v2: roundtrip, hard-fail on schema/fingerprint mismatch, RNG restore.

CPU-only and fast — a tiny ``nn.Module`` stands in for the real SAM-3D-Body model
so the fingerprint / schema / RNG logic is exercised without a GPU or checkpoint.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from contact import checkpoint as ckpt_io


class _Tiny(nn.Module):
    """A 'contact_*' trainable head + a frozen base, mirroring the real freeze split."""

    def __init__(self, dim: int = 4):
        super().__init__()
        self.contact_head = nn.Linear(dim, dim)
        self.frozen_base = nn.Linear(dim, dim)
        for p in self.frozen_base.parameters():
            p.requires_grad = False


class _TinyPlus(_Tiny):
    """``_Tiny`` with an *extra* trainable contact param (a temporal-like head)."""

    def __init__(self, dim: int = 4):
        super().__init__(dim)
        self.contact_temporal = nn.Linear(dim, dim)


class _TinyForce(_Tiny):
    """``_Tiny`` with an extra 'force'-named trainable head (the force branch)."""

    def __init__(self, dim: int = 4):
        super().__init__(dim)
        self.head_force = nn.Linear(dim, dim)


def _force_cfg(force_enabled: bool, grid_size: int = 5) -> dict:
    cfg = {
        "model": {"checkpoint_path": "snap/model.ckpt",
                  "contact_head": {"grid_size": grid_size}},
        "contact": {"topology": "smpl", "targets": {}},
    }
    if force_enabled:
        cfg["model"]["force_head"] = {
            "enabled": True, "frame": "local_world_aligned",
            "mlp_depth": 2, "mlp_channel_div_factor": 4, "dropout": 0.0}
    return cfg


# A pair of same-shape configs differing only semantically (grid_size).
_CFG_A = {"model": {"checkpoint_path": "snap/model.ckpt", "contact_head": {"grid_size": 5}},
          "contact": {"topology": "smpl", "targets": {}}}
_CFG_B = {"model": {"checkpoint_path": "snap/model.ckpt", "contact_head": {"grid_size": 7}},
          "contact": {"topology": "smpl", "targets": {}}}


def _joint_cfg(joint_set: str) -> dict:
    return {
        "model": {"checkpoint_path": "snap/model.ckpt", "contact_head": {"grid_size": 5}},
        "contact": {
            "topology": "smpl",
            "targets": {
                "joint": {"enabled": True, "joint_set": joint_set},
            },
        },
    }


def _temporal_cfg(enabled: bool, position_scale: float = 30.0,
                  force_enabled: bool = False) -> dict:
    cfg = {
        "model": {"checkpoint_path": "snap/model.ckpt",
                  "contact_head": {"grid_size": 5}},
        "contact": {"topology": "smpl", "targets": {}},
    }
    cfg["model"]["temporal"] = (
        {"enabled": True, "placement": "post_decoder", "attend": "per_token",
         "causal": False, "bottleneck_dim": 2, "num_layers": 1, "num_heads": 1,
         "mlp_ratio": 2.0, "position_scale": position_scale}
        if enabled else {"enabled": False})
    if force_enabled:
        cfg["model"]["force_head"] = {
            "enabled": True, "frame": "local_world_aligned",
            "mlp_depth": 2, "mlp_channel_div_factor": 4, "dropout": 0.0}
    return cfg


def _regime_a_cfg(init_contact: str | None = None) -> dict:
    """A regime-(a) config: force enabled + ``train.freeze_contact``."""
    cfg = _force_cfg(force_enabled=True)
    cfg["train"] = {"freeze_contact": True}
    if init_contact is not None:
        cfg["model"]["init_contact_checkpoint"] = init_contact
    return cfg


def _freeze_contact(model: nn.Module) -> nn.Module:
    """Freeze every 'contact'-named param — mirrors regime (a) (train.freeze_contact)."""
    for name, p in model.named_parameters():
        if "contact" in name.lower():
            p.requires_grad = False
    return model


def _trainable_names(model: nn.Module) -> list[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def _saved_names(model: nn.Module) -> list[str]:
    """Superset train.py serialises: every contact/force param (contact/model.py
    ``_trainable_name_filter``), whether or not currently trainable."""
    return [n for n, _ in model.named_parameters()
            if "contact" in n.lower() or "force" in n.lower()]


def _opt_sched(model: nn.Module):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
    return opt, sched


def _save(path, model, monitor="val/vertex_f1", best=0.5, run_id="run-abc"):
    opt, sched = _opt_sched(model)
    opt.step()  # give the optimiser some state
    sched.step()
    ckpt_io.save(path, model, _trainable_names(model), opt, sched,
                 epoch=3, global_step=42, best_metric=best, monitor=monitor,
                 config={"contact": {"topology": "smpl"}}, wandb_run_id=run_id)


# ---------------------------------------------------------------- roundtrip

def test_save_load_roundtrip(tmp_path):
    src = _Tiny()
    path = tmp_path / "ck.pth"
    _save(path, src)

    dst = _Tiny()
    assert not torch.allclose(src.contact_head.weight, dst.contact_head.weight)
    state = ckpt_io.load(path, dst)

    assert torch.allclose(src.contact_head.weight, dst.contact_head.weight)
    assert state["schema_version"] == ckpt_io.SCHEMA_VERSION
    assert state["epoch"] == 3 and state["global_step"] == 42
    assert state["best_metric"] == 0.5 and state["monitor"] == "val/vertex_f1"
    assert state["wandb_run_id"] == "run-abc"
    assert state["config"]["contact"]["topology"] == "smpl"


def test_load_restores_optimizer_and_scheduler(tmp_path):
    src = _Tiny()
    path = tmp_path / "ck.pth"
    _save(path, src)

    dst = _Tiny()
    opt, sched = _opt_sched(dst)
    before = sched.get_last_lr()[0]
    ckpt_io.load(path, dst, opt, sched)
    # the saved scheduler had stepped once -> restored lr differs from a fresh one
    assert sched.last_epoch == 1
    assert sched.get_last_lr()[0] != before


# ---------------------------------------------------------------- hard-fail

def test_schema_mismatch_raises(tmp_path):
    path = tmp_path / "old.pth"
    torch.save({"trainable_state_dict": {}, "trainable_names": [], "schema_version": 1}, path)
    with pytest.raises(RuntimeError, match="schema_version"):
        ckpt_io.load(path, _Tiny())


def test_missing_schema_raises(tmp_path):
    path = tmp_path / "nover.pth"
    torch.save({"trainable_state_dict": {}, "trainable_names": []}, path)
    with pytest.raises(RuntimeError, match="schema_version"):
        ckpt_io.load(path, _Tiny())


def test_fingerprint_mismatch_raises(tmp_path):
    path = tmp_path / "ck.pth"
    _save(path, _Tiny(dim=4))
    # Different architecture (shapes differ) -> fingerprint mismatch, listed clearly.
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        ckpt_io.load(path, _Tiny(dim=8))


def test_not_a_checkpoint_raises(tmp_path):
    path = tmp_path / "junk.pth"
    torch.save({"hello": "world"}, path)
    with pytest.raises(RuntimeError, match="not a contact checkpoint"):
        ckpt_io.load(path, _Tiny())


def test_extra_trainable_param_in_model_raises(tmp_path):
    # Reverse direction: a checkpoint WITHOUT the temporal head loaded into a model
    # that HAS it must fail — otherwise the extra params stay randomly initialised.
    path = tmp_path / "ck.pth"
    _save(path, _Tiny(dim=4))
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        ckpt_io.load(path, _TinyPlus(dim=4))


def test_temporal_warm_start_loads_common_params_only(tmp_path):
    source = _Tiny()
    path = tmp_path / "frame.pth"
    source_cfg = {
        "model": {"checkpoint_path": "snap/model.ckpt", "contact_head": {},
                  "temporal": {"enabled": False}},
        "contact": {"topology": "smpl", "targets": {}},
    }
    target_cfg = {
        "model": {"checkpoint_path": "snap/model.ckpt", "contact_head": {},
                  "temporal": {"enabled": True, "placement": "post_decoder",
                               "bottleneck_dim": 2, "num_layers": 1,
                               "num_heads": 1, "mlp_ratio": 2.0,
                               "attend": "joint", "causal": False}},
        "contact": {"topology": "smpl", "targets": {}},
    }
    opt, sched = _opt_sched(source)
    ckpt_io.save(path, source, _trainable_names(source), opt, sched,
                 epoch=0, global_step=1, best_metric=0.0,
                 monitor="val/vertex_f1", config=source_cfg)

    target = _TinyPlus()
    before_temporal = target.contact_temporal.weight.detach().clone()
    state = ckpt_io.initialize_common_contact(path, target, config=target_cfg)
    assert torch.equal(target.contact_head.weight, source.contact_head.weight)
    assert torch.equal(target.contact_temporal.weight, before_temporal)
    assert state["warm_start_new_names"] == [
        "contact_temporal.bias", "contact_temporal.weight"]
    # the diff must name the unmatched trainable param
    with pytest.raises(RuntimeError, match="contact_temporal"):
        ckpt_io.load(path, _TinyPlus(dim=4))


def test_force_warm_start_loads_contact_leaves_force_fresh(tmp_path):
    # Regime (a): a contact-only checkpoint (no force keys in its arch signature)
    # warm-starts a force-enabled model. Contact params load; the force branch stays
    # fresh; the force keys are exempted from the signature comparison symmetrically.
    source = _Tiny()
    path = tmp_path / "contact.pth"
    opt, sched = _opt_sched(source)
    ckpt_io.save(path, source, _trainable_names(source), opt, sched,
                 epoch=0, global_step=1, best_metric=0.0, monitor="val/joint_f1",
                 config=_force_cfg(force_enabled=False))

    target = _TinyForce()
    before_force = target.head_force.weight.detach().clone()
    state = ckpt_io.initialize_common_contact(
        path, target, config=_force_cfg(force_enabled=True))
    assert torch.equal(target.contact_head.weight, source.contact_head.weight)  # loaded
    assert torch.equal(target.head_force.weight, before_force)                  # fresh
    assert state["warm_start_new_names"] == ["head_force.bias", "head_force.weight"]
    assert state["warm_start_loaded_names"] == [
        "contact_head.bias", "contact_head.weight"]


def test_force_warm_start_rejects_other_arch_mismatch(tmp_path):
    # A non-force/non-temporal semantic difference (grid_size) must still hard-fail
    # even though the force keys are now exempted.
    source = _Tiny()
    path = tmp_path / "contact.pth"
    opt, sched = _opt_sched(source)
    ckpt_io.save(path, source, _trainable_names(source), opt, sched,
                 epoch=0, global_step=1, best_metric=0.0, monitor="val/joint_f1",
                 config=_force_cfg(force_enabled=False, grid_size=5))
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        ckpt_io.initialize_common_contact(
            path, _TinyForce(), config=_force_cfg(force_enabled=True, grid_size=7))


def test_force_warm_start_requires_temporal_or_force_target(tmp_path):
    # Neither temporal nor force enabled in the target -> the precondition fires.
    source = _Tiny()
    path = tmp_path / "contact.pth"
    opt, sched = _opt_sched(source)
    ckpt_io.save(path, source, _trainable_names(source), opt, sched,
                 epoch=0, global_step=1, best_metric=0.0, monitor="val/joint_f1",
                 config=_force_cfg(force_enabled=False))
    with pytest.raises(RuntimeError, match="temporal module or the force branch"):
        ckpt_io.initialize_common_contact(
            path, _Tiny(), config=_force_cfg(force_enabled=False))


def test_force_model_checkpoint_roundtrip(tmp_path):
    # Strict resume of a force-enabled model: save -> load reproduces the trainable
    # state exactly (force + contact) and the resolved config signature matches.
    src = _TinyForce()
    path = tmp_path / "force.pth"
    opt, sched = _opt_sched(src)
    opt.step(); sched.step()
    ckpt_io.save(path, src, _trainable_names(src), opt, sched,
                 epoch=2, global_step=5, best_metric=0.1,
                 monitor="val/physics_residual", config=_force_cfg(force_enabled=True))

    dst = _TinyForce()
    assert not torch.allclose(src.head_force.weight, dst.head_force.weight)
    state = ckpt_io.load(path, dst, config=_force_cfg(force_enabled=True))
    assert torch.equal(src.head_force.weight, dst.head_force.weight)
    assert torch.equal(src.contact_head.weight, dst.contact_head.weight)
    assert state["monitor"] == "val/physics_residual"
    assert state["epoch"] == 2 and state["global_step"] == 5


# ------------------------------------------ regime (a): self-contained + recovery

def test_regime_a_checkpoint_is_self_contained(tmp_path):
    # Regime (a): contact frozen, force-only trainable. save() with the saved_names
    # superset must persist the frozen contact weights so a fresh model + load()
    # restores contact AND force bit-exact (never a random contact head).
    src = _freeze_contact(_TinyForce())
    trainable = _trainable_names(src)
    assert all("force" in n for n in trainable)            # force-only trainable
    assert any("contact" in n for n in _saved_names(src))  # contact still serialised

    path = tmp_path / "regime_a.pth"
    opt, sched = _opt_sched(src)
    opt.step(); sched.step()
    ckpt_io.save(path, src, trainable, opt, sched,
                 epoch=1, global_step=2, best_metric=0.0,
                 monitor="test/physics_residual", config=_regime_a_cfg(),
                 saved_names=_saved_names(src))

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert any("contact" in k for k in ckpt["trainable_state_dict"])   # contact tensors present
    assert ckpt["frozen_saved_names"] == ["contact_head.bias", "contact_head.weight"]

    dst = _freeze_contact(_TinyForce())
    assert not torch.allclose(src.contact_head.weight, dst.contact_head.weight)
    assert not torch.allclose(src.head_force.weight, dst.head_force.weight)
    ckpt_io.load(path, dst, config=_regime_a_cfg())   # freeze_contact set, contact present -> no recovery
    assert torch.equal(src.contact_head.weight, dst.contact_head.weight)   # frozen contact restored
    assert torch.equal(src.head_force.weight, dst.head_force.weight)       # force restored


def test_legacy_regime_a_recovers_contact_from_init(tmp_path):
    # A LEGACY regime-(a) checkpoint (force tensors only, the old bug) must recover
    # its dropped contact branch from the config's init_contact_checkpoint on load().
    init_source = _Tiny()
    init_path = tmp_path / "contact_init.pth"
    opt, sched = _opt_sched(init_source)
    ckpt_io.save(init_path, init_source, _trainable_names(init_source), opt, sched,
                 epoch=0, global_step=1, best_metric=0.0, monitor="val/joint_f1",
                 config=_force_cfg(force_enabled=False))

    legacy = _freeze_contact(_TinyForce())
    legacy_path = tmp_path / "legacy_force.pth"
    opt, sched = _opt_sched(legacy)
    # saved_names defaults to trainable_names (force-only) -> reproduces the bug.
    ckpt_io.save(legacy_path, legacy, _trainable_names(legacy), opt, sched,
                 epoch=2, global_step=3, best_metric=0.0,
                 monitor="test/physics_residual",
                 config=_regime_a_cfg(init_contact=str(init_path)))
    ckpt = torch.load(legacy_path, map_location="cpu", weights_only=False)
    assert not any("contact" in k for k in ckpt["trainable_state_dict"])   # contact-less

    dst = _freeze_contact(_TinyForce())
    ckpt_io.load(legacy_path, dst, config=_regime_a_cfg(init_contact=str(init_path)))
    assert torch.equal(dst.contact_head.weight, init_source.contact_head.weight)  # recovered
    assert torch.equal(dst.head_force.weight, legacy.head_force.weight)           # from legacy


def test_legacy_regime_a_missing_init_raises(tmp_path):
    legacy = _freeze_contact(_TinyForce())
    legacy_path = tmp_path / "legacy_force.pth"
    opt, sched = _opt_sched(legacy)
    missing = str(tmp_path / "does_not_exist.pth")
    ckpt_io.save(legacy_path, legacy, _trainable_names(legacy), opt, sched,
                 epoch=0, global_step=0, best_metric=0.0,
                 monitor="test/physics_residual",
                 config=_regime_a_cfg(init_contact=missing))
    dst = _freeze_contact(_TinyForce())
    with pytest.raises(RuntimeError, match="missing/unreadable"):
        ckpt_io.load(legacy_path, dst, config=_regime_a_cfg(init_contact=missing))


def test_legacy_regime_a_no_init_raises(tmp_path):
    legacy = _freeze_contact(_TinyForce())
    legacy_path = tmp_path / "legacy_force.pth"
    opt, sched = _opt_sched(legacy)
    ckpt_io.save(legacy_path, legacy, _trainable_names(legacy), opt, sched,
                 epoch=0, global_step=0, best_metric=0.0,
                 monitor="test/physics_residual", config=_regime_a_cfg())
    dst = _freeze_contact(_TinyForce())
    with pytest.raises(RuntimeError, match="init_contact_checkpoint"):
        ckpt_io.load(legacy_path, dst, config=_regime_a_cfg())


def test_truncated_frozen_saved_names_raises(tmp_path):
    # frozen_saved_names is a CLAIM of self-containment: a checkpoint whose claimed
    # frozen tensor is missing from the state dict (truncated/corrupt file) must
    # hard-fail — never load a partially random frozen contact branch.
    src = _freeze_contact(_TinyForce())
    path = tmp_path / "truncated.pth"
    opt, sched = _opt_sched(src)
    ckpt_io.save(path, src, _trainable_names(src), opt, sched,
                 epoch=0, global_step=0, best_metric=0.0,
                 monitor="test/physics_residual", config=_regime_a_cfg(),
                 saved_names=_saved_names(src))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    del ckpt["trainable_state_dict"]["contact_head.weight"]      # truncate one tensor
    torch.save(ckpt, path)
    with pytest.raises(RuntimeError, match="frozen_saved_names"):
        ckpt_io.load(path, _freeze_contact(_TinyForce()), config=_regime_a_cfg())


def test_partial_contact_state_raises_not_recovers(tmp_path):
    # SOME but not the model's FULL contact-named set in the state is a corrupt /
    # incompatible regime-(a) save: it must neither pass as self-contained (leaving
    # the missing tensors random) nor trigger legacy recovery — it raises.
    src = _freeze_contact(_TinyForce())
    trainable = _trainable_names(src)
    partial = trainable + ["contact_head.weight"]                # weight, NOT bias
    path = tmp_path / "partial.pth"
    opt, sched = _opt_sched(src)
    ckpt_io.save(path, src, trainable, opt, sched,
                 epoch=0, global_step=0, best_metric=0.0,
                 monitor="test/physics_residual", config=_regime_a_cfg(),
                 saved_names=partial)
    with pytest.raises(RuntimeError, match="PARTIAL contact branch"):
        ckpt_io.load(path, _freeze_contact(_TinyForce()), config=_regime_a_cfg())


# ------------------------------------------ warm start from a temporal source (FIX 2)

def test_temporal_source_warm_start_loads_contact_temporal(tmp_path):
    # A temporal source is allowed when the target's temporal architecture is
    # identical: contact_temporal.* then LOADS (not left fresh).
    source = _TinyPlus()
    path = tmp_path / "t5.pth"
    opt, sched = _opt_sched(source)
    ckpt_io.save(path, source, _trainable_names(source), opt, sched,
                 epoch=0, global_step=1, best_metric=0.0, monitor="test/joint_f1",
                 config=_temporal_cfg(enabled=True, position_scale=30.0))

    target = _TinyPlus()
    assert not torch.allclose(target.contact_temporal.weight, source.contact_temporal.weight)
    state = ckpt_io.initialize_common_contact(
        path, target, config=_temporal_cfg(enabled=True, position_scale=30.0))
    assert torch.equal(target.contact_temporal.weight, source.contact_temporal.weight)  # LOADED
    assert torch.equal(target.contact_head.weight, source.contact_head.weight)
    assert state["warm_start_new_names"] == []   # nothing missing — contact_temporal loaded


def test_temporal_source_mismatched_temporal_raises(tmp_path):
    source = _TinyPlus()
    path = tmp_path / "t5.pth"
    opt, sched = _opt_sched(source)
    ckpt_io.save(path, source, _trainable_names(source), opt, sched,
                 epoch=0, global_step=1, best_metric=0.0, monitor="test/joint_f1",
                 config=_temporal_cfg(enabled=True, position_scale=30.0))
    with pytest.raises(RuntimeError, match="temporal architecture differs"):
        ckpt_io.initialize_common_contact(
            path, _TinyPlus(),
            config=_temporal_cfg(enabled=True, position_scale=1.0))


def test_temporal_source_disabled_target_raises(tmp_path):
    # Target temporal disabled but force enabled (so the precondition would pass) —
    # a temporal source must still raise because the temporal architecture differs.
    source = _TinyPlus()
    path = tmp_path / "t5.pth"
    opt, sched = _opt_sched(source)
    ckpt_io.save(path, source, _trainable_names(source), opt, sched,
                 epoch=0, global_step=1, best_metric=0.0, monitor="test/joint_f1",
                 config=_temporal_cfg(enabled=True, position_scale=30.0))
    with pytest.raises(RuntimeError, match="temporal architecture differs"):
        ckpt_io.initialize_common_contact(
            path, _TinyForce(),
            config=_temporal_cfg(enabled=False, force_enabled=True))


def test_same_shape_semantic_mismatch_raises(tmp_path):
    # Identical param shapes, different architecture semantics (grid_size) -> the
    # shape fingerprint passes but the arch signature catches it.
    path = tmp_path / "ck.pth"
    opt, sched = _opt_sched(_Tiny())
    ckpt_io.save(path, _Tiny(), _trainable_names(_Tiny()), opt, sched,
                 epoch=0, global_step=0, best_metric=0.0, monitor="val/vertex_f1",
                 config=_CFG_A)
    with pytest.raises(RuntimeError, match="signature mismatch"):
        ckpt_io.load(path, _Tiny(), config=_CFG_B)


def test_temporal_position_scale_is_checkpointed_semantics(tmp_path):
    def cfg(scale):
        return {
            "model": {
                "checkpoint_path": "snap/model.ckpt",
                "contact_head": {"grid_size": 5},
                "temporal": {
                    "enabled": True,
                    "placement": "post_decoder",
                    "attend": "per_token",
                    "causal": False,
                    "bottleneck_dim": 256,
                    "num_layers": 1,
                    "num_heads": 4,
                    "mlp_ratio": 2.0,
                    "position_scale": scale,
                },
            },
            "contact": {"topology": "smpl", "targets": {}},
        }

    model = _Tiny()
    opt, sched = _opt_sched(model)
    path = tmp_path / "ck.pth"
    ckpt_io.save(
        path, model, _trainable_names(model), opt, sched,
        epoch=0, global_step=0, best_metric=0.0,
        monitor="test/joint_f1", config=cfg(30.0),
    )
    with pytest.raises(RuntimeError, match="signature mismatch"):
        ckpt_io.load(path, _Tiny(), config=cfg(1.0))


def test_matching_signature_loads(tmp_path):
    path = tmp_path / "ck.pth"
    opt, sched = _opt_sched(_Tiny())
    ckpt_io.save(path, _Tiny(), _trainable_names(_Tiny()), opt, sched,
                 epoch=0, global_step=0, best_metric=0.0, monitor="val/vertex_f1",
                 config=_CFG_A)
    ckpt_io.load(path, _Tiny(), config=_CFG_A)   # same signature -> no raise


def test_checkpoint_signature_records_extremity_semantics():
    signature = ckpt_io._arch_signature(_joint_cfg("extremities_4"))
    assert signature["targets"]["joint"] == 4
    assert signature["joint_layout"] == {
        "joint_set": "extremities_4",
        "names": ["left_hand", "right_hand", "left_foot", "right_foot"],
        "dim": 4,
    }


def test_same_shape_different_joint_set_signature_raises(tmp_path):
    # The tiny stand-in deliberately has identical parameter shapes; semantic
    # layout metadata must still prevent cross-loading body-22/extremity heads.
    path = tmp_path / "ck.pth"
    model = _Tiny()
    opt, sched = _opt_sched(model)
    ckpt_io.save(path, model, _trainable_names(model), opt, sched,
                 epoch=0, global_step=0, best_metric=0.0, monitor="val/joint_f1",
                 config=_joint_cfg("smplx_body_22"))
    with pytest.raises(RuntimeError, match="joint_layout"):
        ckpt_io.load(path, _Tiny(), config=_joint_cfg("extremities_4"))


def test_split_manifest_roundtrips(tmp_path):
    path = tmp_path / "ck.pth"
    manifest = {"images": {"train": [0, 2, 3], "val": [1]},
                "video:cfg": {"train": ["vidA"], "val": ["vidB"]}}
    opt, sched = _opt_sched(_Tiny())
    ckpt_io.save(path, _Tiny(), _trainable_names(_Tiny()), opt, sched,
                 epoch=0, global_step=0, best_metric=0.0, monitor="val/vertex_f1",
                 config=_CFG_A, split_manifest=manifest)
    state = ckpt_io.load(path, _Tiny())
    assert state["split_manifest"] == manifest


# ---------------------------------------------------------------- RNG restore

def test_rng_restore_reproduces_stream(tmp_path):
    path = tmp_path / "ck.pth"
    model = _Tiny()

    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    torch.rand(5); np.random.rand(5); random.random()   # advance the streams

    _save(path, model)
    expected_t = torch.rand(3)
    expected_n = np.random.rand(3)
    expected_p = [random.random() for _ in range(3)]

    # Perturb every stream, then restore from the checkpoint.
    torch.manual_seed(999); np.random.seed(999); random.seed(999)
    torch.rand(100); np.random.rand(100); [random.random() for _ in range(100)]

    ckpt_io.load(path, model, restore_rng=True)
    assert torch.allclose(torch.rand(3), expected_t)
    assert np.allclose(np.random.rand(3), expected_n)
    assert [random.random() for _ in range(3)] == expected_p


def test_load_without_restore_rng_leaves_stream(tmp_path):
    path = tmp_path / "ck.pth"
    model = _Tiny()
    _save(path, model)

    torch.manual_seed(7)
    ref = torch.rand(3)
    torch.manual_seed(7)
    ckpt_io.load(path, model, restore_rng=False)   # must NOT touch the RNG
    assert torch.allclose(torch.rand(3), ref)
