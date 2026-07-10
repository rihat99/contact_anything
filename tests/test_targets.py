"""Targets: topology sizes, nearest-joint ownership, joint derivation, validation."""
from __future__ import annotations

import pytest
import torch

from contact.targets import (
    NUM_BODY_22,
    OBSERVABLE_14,
    SMPLX_BODY_22,
    TOPOLOGY_VERTS,
    TargetSpec,
    compute_vertex_joint_owner,
    derive_joint_contact,
    topology_num_vertices,
    validate_targets,
)

# left/right wrist indices in the 22-joint body set (folded hands land here).
_LEFT_WRIST, _RIGHT_WRIST = 20, 21


def test_topology_sizes():
    assert TOPOLOGY_VERTS == {"smpl": 6890, "smplx": 10475}
    assert topology_num_vertices("smpl") == 6890
    assert topology_num_vertices("smplx") == 10475


def test_mhr_topology_not_implemented():
    with pytest.raises(NotImplementedError):
        topology_num_vertices("mhr")


def test_joint_set_names_and_observable():
    assert len(SMPLX_BODY_22) == NUM_BODY_22 == 22
    assert SMPLX_BODY_22[_LEFT_WRIST] == "left_wrist"
    assert SMPLX_BODY_22[_RIGHT_WRIST] == "right_wrist"
    assert OBSERVABLE_14 == [1, 2, 4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21]


def test_ownership_every_joint_owns_at_least_25_verts():
    owner = compute_vertex_joint_owner()
    assert owner.shape == (6890,)
    assert owner.dtype == torch.int64
    assert int(owner.min()) >= 0 and int(owner.max()) < NUM_BODY_22
    counts = torch.bincount(owner, minlength=NUM_BODY_22)
    assert int(counts.min()) >= 25, f"a joint owns <25 verts: {counts.tolist()}"
    assert int((counts == 0).sum()) == 0, "some joint orphaned (0 verts)"


def test_wrists_own_hand_vertices_after_fold():
    # After folding SMPL hand joints (22/23) into the wrists (20/21), the wrists
    # must own substantially more than a bare wrist joint would — the hand region.
    owner = compute_vertex_joint_owner()
    counts = torch.bincount(owner, minlength=NUM_BODY_22)
    median = float(counts.float().median())
    assert int(counts[_LEFT_WRIST]) > median
    assert int(counts[_RIGHT_WRIST]) > median


def test_derive_joint_contact_single_joint():
    owner = compute_vertex_joint_owner()
    vc = torch.zeros(6890)
    vc[owner == 7] = 1.0                       # light up all vertices owned by joint 7
    jc = derive_joint_contact(vc, owner)
    assert jc.shape == (NUM_BODY_22,)
    assert bool(jc[7] > 0.5)
    assert float(jc.sum()) == pytest.approx(1.0)  # exactly one joint hot


def test_derive_joint_contact_batched_and_amax():
    owner = compute_vertex_joint_owner()
    vc = torch.zeros(3, 6890)
    vc[1, owner == _LEFT_WRIST] = 1.0          # only one owned vertex per joint needed
    jc = derive_joint_contact(vc, owner)
    assert jc.shape == (3, NUM_BODY_22)
    assert bool(jc[1, _LEFT_WRIST] > 0.5)
    assert float(jc[0].sum()) == 0.0
    assert float(jc[2].sum()) == 0.0


def test_target_spec_output_dims():
    cfg = _base_cfg(vertex=True, joint=True)
    spec = TargetSpec.from_config(cfg)
    assert spec.output_dims() == {"vertex": 6890, "joint": 22}

    cfg_joint = _base_cfg(vertex=False, joint=True)
    assert TargetSpec.from_config(cfg_joint).output_dims() == {"joint": 22}


# ---------------------------------------------------------------- validation

class _FakeDataset:
    def __init__(self, name, supervised, topology):
        self.name = name
        self.supervised_targets = frozenset(supervised)
        self.topology = topology


def test_validate_targets_accepts_matching_setup():
    cfg = _base_cfg(vertex=True, joint=False)
    validate_targets(cfg, [_FakeDataset("damon", {"vertex"}, "smpl")])  # no raise


def test_validate_targets_rejects_unsupervised_target():
    cfg = _base_cfg(vertex=True, joint=True)   # joint enabled but no joint dataset, derive off
    with pytest.raises(ValueError, match="not supervised"):
        validate_targets(cfg, [_FakeDataset("damon", {"vertex"}, "smpl")])


def test_validate_targets_rejects_topology_mismatch():
    cfg = _base_cfg(vertex=True, joint=False)   # topology smpl
    with pytest.raises(ValueError, match="topology"):
        validate_targets(cfg, [_FakeDataset("climbing", {"vertex"}, "smplx")])


def test_validate_targets_derive_from_vertex_supervises_joint():
    cfg = _base_cfg(vertex=True, joint=True)
    cfg["contact"]["targets"]["joint"]["derive_from_vertex"] = True
    validate_targets(cfg, [_FakeDataset("damon", {"vertex"}, "smpl")])  # no raise


def test_validate_targets_video_supervises_joint():
    cfg = _base_cfg(vertex=False, joint=True)
    validate_targets(cfg, [_FakeDataset("climbing_videos", {"joint"}, None)])  # no raise


def test_validate_targets_rejects_dataset_with_no_enabled_target():
    # joint-only config: a DAMON (vertex-only) loader alongside a video loader
    # satisfies the *global* joint check but is itself wholly unsupervised.
    cfg = _base_cfg(vertex=False, joint=True)
    datasets = [_FakeDataset("climbing_videos", {"joint"}, None),
                _FakeDataset("damon", {"vertex"}, "smpl")]
    with pytest.raises(ValueError, match="supervises none"):
        validate_targets(cfg, datasets)


# ---------------------------------------------------------------- per-sample betas

def test_ownership_shifts_with_betas():
    neutral = compute_vertex_joint_owner()
    shaped = compute_vertex_joint_owner(betas=[3.0, -3.0, 2.0, -2.0, 1.5, 0, 0, 0, 0, 0])
    assert int((neutral != shaped).sum()) > 0, "extreme betas moved no vertex owner"


def test_assemble_batch_uses_per_sample_betas():
    betas = [3.0, -3.0, 2.0, -2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    neutral = compute_vertex_joint_owner()
    shaped = compute_vertex_joint_owner(betas=betas)
    moved = int((neutral != shaped).nonzero()[0][0])       # a boundary vertex that moved
    j_neutral, j_shaped = int(neutral[moved]), int(shaped[moved])

    spec = TargetSpec.from_config(_derive_cfg())
    contact = torch.zeros(6890)
    contact[moved] = 1.0

    shaped_frame = {"contact": contact, "smpl": {"betas": betas}}
    gt_shaped = spec.assemble_batch([shaped_frame])["joint"]["gt"][0]
    assert bool(gt_shaped[j_shaped] > 0.5)                 # lit at the shaped owner
    assert not bool(gt_shaped[j_neutral] > 0.5)            # NOT the neutral owner

    neutral_frame = {"contact": contact}                   # no betas -> neutral map
    gt_neutral = spec.assemble_batch([neutral_frame])["joint"]["gt"][0]
    assert bool(gt_neutral[j_neutral] > 0.5)


def _derive_cfg() -> dict:
    return {
        "contact": {
            "topology": "smpl",
            "primary_target": "joint",
            "targets": {
                "vertex": {"enabled": False},
                "joint": {"enabled": True, "supervise_subset": None,
                          "derive_from_vertex": True},
            },
        }
    }


# ---------------------------------------------------------------- helpers

def _base_cfg(vertex: bool, joint: bool) -> dict:
    return {
        "contact": {
            "topology": "smpl",
            "primary_target": "vertex" if vertex else "joint",
            "targets": {
                "vertex": {"enabled": vertex},
                "joint": {
                    "enabled": joint,
                    "supervise_subset": None,
                    "derive_from_vertex": False,
                },
            },
        }
    }
