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


def _trainable_names(model: nn.Module) -> list[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


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
