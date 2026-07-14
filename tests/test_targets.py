"""Targets: topology sizes, nearest-joint ownership, joint derivation, validation."""
from __future__ import annotations

import pytest
import torch

from contact.targets import (
    EXTREMITY_4_GROUPS,
    EXTREMITY_4_NAMES,
    NUM_BODY_22,
    NUM_EXTREMITY_4,
    OBSERVABLE_14,
    SMPLX_BODY_22,
    TOPOLOGY_VERTS,
    TargetSpec,
    compute_vertex_joint_owner,
    derive_joint_contact,
    reduce_body22_to_extremities,
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
    assert EXTREMITY_4_NAMES == ("left_hand", "right_hand", "left_foot", "right_foot")
    assert EXTREMITY_4_GROUPS == ((20,), (21,), (7, 10), (8, 11))
    assert NUM_EXTREMITY_4 == 4


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

    cfg_extremities = _base_cfg(vertex=False, joint=True)
    cfg_extremities["contact"]["targets"]["joint"]["joint_set"] = "extremities_4"
    spec_extremities = TargetSpec.from_config(cfg_extremities)
    assert spec_extremities.output_dims() == {"joint": 4}
    assert spec_extremities.joint_names == EXTREMITY_4_NAMES


def test_reduce_body22_to_extremities_tri_state_and_confidence():
    # Eight rows cover the complete two-member foot truth table plus the
    # singleton hand behavior. Values not explicitly set remain free/unknown.
    contact = torch.zeros(8, NUM_BODY_22)
    supervised = torch.zeros_like(contact)
    confidence = torch.zeros_like(contact)

    # 0: both known free -> known free, confidence mean = .6
    supervised[0, [7, 10]] = 1
    confidence[0, [7, 10]] = torch.tensor([0.4, 0.8])
    # 1: ankle positive, foot free -> contact; only positive confidence counts
    supervised[1, [7, 10]] = 1
    contact[1, 7] = 1
    confidence[1, [7, 10]] = torch.tensor([0.3, 0.95])
    # 2: foot/toe positive, ankle free -> symmetric OR behavior
    supervised[2, [7, 10]] = 1
    contact[2, 10] = 1
    confidence[2, [7, 10]] = torch.tensor([0.95, 0.45])
    # 3: both positive -> max positive confidence
    supervised[3, [7, 10]] = 1
    contact[3, [7, 10]] = 1
    confidence[3, [7, 10]] = torch.tensor([0.55, 0.9])
    # 4: known positive + unknown -> contact remains known
    supervised[4, 10] = 1
    contact[4, 10] = 1
    confidence[4, 10] = 0.7
    # 5: known negative + unknown -> partial negative is ignored
    supervised[5, 7] = 1
    confidence[5, 7] = 0.8
    # 6: both unknown -> ignored
    # 7: singleton hand known positive preserves its confidence
    supervised[7, 20] = 1
    contact[7, 20] = 1
    confidence[7, 20] = 0.65

    gt, sup, conf = reduce_body22_to_extremities(contact, supervised, confidence)
    assert gt.shape == sup.shape == conf.shape == (8, NUM_EXTREMITY_4)
    assert torch.allclose(gt[:7, 2], torch.tensor([0, 1, 1, 1, 1, 0, 0]).float())
    assert torch.allclose(sup[:7, 2], torch.tensor([1, 1, 1, 1, 1, 0, 0]).float())
    assert torch.allclose(conf[:7, 2], torch.tensor([0.6, 0.3, 0.45, 0.9, 0.7, 0, 0]))
    assert gt[7, 0] == 1 and sup[7, 0] == 1 and conf[7, 0] == pytest.approx(0.65)


def test_extremity_assemble_batch_uses_raw_supervision_and_confidence():
    cfg = _base_cfg(vertex=False, joint=True)
    jcfg = cfg["contact"]["targets"]["joint"]
    jcfg["joint_set"] = "extremities_4"
    jcfg["use_confidence_weights"] = True
    spec = TargetSpec.from_config(cfg)

    contact = torch.zeros(NUM_BODY_22)
    supervised = torch.zeros(NUM_BODY_22)
    confidence = torch.zeros(NUM_BODY_22)
    # Known left-hand contact.
    contact[20] = 1
    supervised[20] = 1
    confidence[20] = 0.75
    # Known-free left foot: mean confidence .5.
    supervised[[7, 10]] = 1
    confidence[[7, 10]] = torch.tensor([0.2, 0.8])
    # Right foot is only a partial negative and must remain ignored despite its
    # nonzero confidence. A preweighted mask alone could not distinguish this.
    supervised[8] = 1
    confidence[8] = 0.9
    frame = {
        "joint_contact": contact,
        "joint_supervised": supervised,
        "joint_confidence": confidence,
        "joint_mask": supervised * confidence,
    }
    target = spec.assemble_batch([frame])["joint"]
    assert torch.allclose(target["gt"], torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(target["mask"], torch.tensor([[0.75, 0.0, 0.5, 0.0]]))

    jcfg["use_confidence_weights"] = False
    unweighted = TargetSpec.from_config(cfg).assemble_batch([frame])["joint"]
    assert torch.allclose(unweighted["mask"], torch.tensor([[1.0, 0.0, 1.0, 0.0]]))


def test_extremity_assemble_requires_raw_annotation_fields():
    cfg = _base_cfg(vertex=False, joint=True)
    cfg["contact"]["targets"]["joint"]["joint_set"] = "extremities_4"
    frame = {
        "joint_contact": torch.zeros(NUM_BODY_22),
        "joint_mask": torch.ones(NUM_BODY_22),
    }
    with pytest.raises(ValueError, match="requires raw body-22"):
        TargetSpec.from_config(cfg).assemble_batch([frame])


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
                    "joint_set": "smplx_body_22",
                    "supervise_subset": None,
                    "derive_from_vertex": False,
                    "use_confidence_weights": False,
                },
            },
        }
    }
