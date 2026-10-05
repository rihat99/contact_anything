"""Supervision under ``data.contact_set: frames35`` (CPU, synthetic batches, float32).

The losses are indexed by the run's contact set, and every reported contact number
exists on two levels: the K SLOTS and the six kindyn GROUPS they fold onto. These
tests pin the fold and the slot-point contract:

* the six-group fold of the contact labels — a group with one positive SUPERVISED
  member is positive even when another member is unknown; a group whose members are
  all supervised and free is a known negative; a group with no positive and an
  unknown member is not scored at all; and a batch that carries the manual
  ``contact_gt_groups`` block is scored against THAT instead;
* the fold of the predictions is the MAX over the member slots, so one firing slot
  fires its group;
* under ``kindyn6`` the fold is the identity: the ``slots_*`` block and the group
  block are the same numbers, and both match an explicit reference;
* the force loss's ``groups_mae`` sums the member slots' force vectors on both
  sides, on the rows where a member is in contact;
* ``contact_consistency`` reads ``out["smplx"]["slot_points_world"]``, with the GT
  floor at each slot's parent joint;
* ``force_consistency`` applies each slot's force at its slot POINT: moving a slot
  point off its parent joint shifts the torque residual by exactly ``-(r x f)``.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

from model.contact_frames import KINDYN_GROUP_NAMES, contact_set
from model.loss.contact import THRESHOLD, ContactLoss
from model.loss.contact_consistency import ContactConsistencyLoss
from model.loss.force import ForceLoss
from model.loss.force_consistency import ForceConsistencyLoss
from utils.geometry import smplx_q
from utils.metrics import prf1

REPO = Path(__file__).resolve().parents[1]
FPS = 25.0
SLOTS = contact_set("frames35")
K = SLOTS.count


@pytest.fixture(scope="module")
def base_cfg():
    return yaml.safe_load((REPO / "configs" / "base.yaml").read_text())


def cfg_for(base_cfg: dict, set_name: str, **sections) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["contact_set"] = set_name
    for section, values in sections.items():
        cfg[section].update(values)
    return cfg


def contact_loss(base_cfg, set_name: str, **section) -> ContactLoss:
    return ContactLoss(cfg_for(base_cfg, set_name, contact_supervision=dict(
        enabled=True, weight=1.0, confidence_weights=True, layer_weight=0.0, **section)),
        None, "cpu")


def out_of(logits: torch.Tensor) -> dict:
    return {"contact": {"logits": logits, "probs": torch.sigmoid(logits)}}


#: Slot index of each named frames35 slot.
IDX = {name: i for i, name in enumerate(SLOTS.slot_names)}
#: The members of the two hand groups (four slots each) — the fold's interesting case.
LEFT_HAND = [IDX[n] for n in ("palm_left", "finger_left", "palm_distal_left", "thumb_left")]


# ------------------------------------------------------------------ the six-group fold

def fold_batch() -> tuple[dict, torch.Tensor]:
    """Four rows exercising the fold rule on the left hand, plus logits.

    Row 0: one positive supervised member, one unknown member -> the group is POSITIVE.
    Row 1: every member supervised and free                   -> known NEGATIVE.
    Row 2: no positive, one member unknown                    -> NOT scored.
    Row 3: nothing supervised at all                          -> NOT scored.
    """
    n = 4
    gt = torch.zeros(n, K)
    valid = torch.zeros(n, K)
    valid[0, LEFT_HAND] = 1.0
    valid[0, LEFT_HAND[1]] = 0.0                       # one member unknown
    gt[0, LEFT_HAND[0]] = 1.0                          # and one supervised positive
    valid[1, LEFT_HAND] = 1.0                          # all supervised, all free
    valid[2, LEFT_HAND] = 1.0
    valid[2, LEFT_HAND[2]] = 0.0                       # unknown member, no positive
    batch = {"contact_gt": gt, "contact_valid": valid, "contact_conf": torch.ones(n, K)}
    logits = torch.full((n, K), -4.0)
    logits[0, LEFT_HAND[3]] = 4.0                      # one member fires -> the group fires
    logits[1, LEFT_HAND[1]] = 4.0                      # a false positive on a free group
    return batch, logits


def group_counts(loss: ContactLoss, stats: torch.Tensor) -> torch.Tensor:
    """``(6, 4)`` group confusion counts out of the statistics vector."""
    start = loss.contact_set.count * 4
    return stats[start:start + 6 * 4].reshape(6, 4)


def test_group_fold_rule_of_the_slot_labels(base_cfg):
    loss = contact_loss(base_cfg, "frames35")
    batch, logits = fold_batch()
    stats = loss(out_of(logits), batch, train=True).stats
    counts = group_counts(loss, stats)
    left = KINDYN_GROUP_NAMES.index("left_hand")
    # Rows 2 and 3 are not scored; row 0 is a true positive (a member fired), row 1 a
    # false positive (all members free, one fired).
    assert counts[left].tolist() == [1.0, 1.0, 0.0, 0.0]
    # No other group saw a supervised slot at all.
    assert counts.sum() == pytest.approx(2.0)
    metrics = loss.metrics(stats)
    assert metrics["groups/left_hand_recall"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["groups/left_hand_precision"] == pytest.approx(0.5, abs=1e-6)


def test_group_prediction_is_the_max_over_the_members(base_cfg):
    loss = contact_loss(base_cfg, "frames35")
    n = 1
    batch = {"contact_gt": torch.ones(n, K), "contact_valid": torch.ones(n, K),
             "contact_conf": torch.ones(n, K)}
    logits = torch.full((n, K), -4.0)
    counts = group_counts(loss, loss(out_of(logits), batch, train=True).stats)
    assert counts[:, 0].sum() == 0.0 and counts[:, 2].sum() == 6.0     # all six missed
    logits[0, LEFT_HAND[2]] = 4.0                                      # ONE member fires
    counts = group_counts(loss, loss(out_of(logits), batch, train=True).stats)
    left = KINDYN_GROUP_NAMES.index("left_hand")
    assert counts[left].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert counts[:, 0].sum() == 1.0


def test_manual_group_labels_replace_the_fold(base_cfg):
    """A test batch carrying ``contact_gt_groups`` is scored against THOSE labels."""
    loss = contact_loss(base_cfg, "frames35")
    batch, logits = fold_batch()
    batch["contact_gt_groups"] = torch.zeros(4, 6)
    batch["contact_valid_groups"] = torch.ones(4, 6)
    batch["contact_gt_groups"][:, KINDYN_GROUP_NAMES.index("right_ankle")] = 1.0
    counts = group_counts(loss, loss(out_of(logits), batch, train=True).stats)
    # Every row of every group is scored now (4 x 6 = 24 counts), and the right heel is
    # the only positive: never predicted, so four false negatives.
    assert counts.sum() == pytest.approx(24.0)
    assert counts[KINDYN_GROUP_NAMES.index("right_ankle")].tolist() == [0.0, 0.0, 4.0, 0.0]
    # The slot block is untouched by the manual group labels.
    slot_counts = loss(out_of(logits), batch, train=True).stats[:K * 4].reshape(K, 4)
    assert slot_counts.sum() == pytest.approx(
        float((batch["contact_valid"] > 0).sum()))


def test_kindyn6_fold_is_the_identity(base_cfg):
    """Under ``kindyn6`` the group block, the slot block and an explicit reference agree."""
    torch.manual_seed(4)
    loss = contact_loss(base_cfg, "kindyn6")
    n = 24
    gt = (torch.rand(n, 6) > 0.5).float()
    batch = {"contact_gt": gt, "contact_valid": (torch.rand(n, 6) > 0.25).float(),
             "contact_conf": torch.ones(n, 6)}
    logits = torch.randn(n, 6)
    metrics = loss.metrics(loss(out_of(logits), batch, train=True).stats)
    for key in ("f1", "precision", "recall", "iou"):
        assert metrics[f"slots_{key}"] == pytest.approx(metrics[key], rel=1e-9)
    for group in KINDYN_GROUP_NAMES:
        for key in ("f1", "precision", "recall"):
            assert metrics[f"slots/{group}_{key}"] == pytest.approx(
                metrics[f"groups/{group}_{key}"], rel=1e-9)
    pred = torch.sigmoid(logits) > THRESHOLD
    active, positive = batch["contact_valid"] > 0, gt > 0.5
    reference = prf1([float((pred & positive & active).sum()),
                      float((pred & ~positive & active).sum()),
                      float((~pred & positive & active).sum()),
                      float((~pred & ~positive & active).sum())])
    for key in ("f1", "precision", "recall", "iou"):
        assert metrics[key] == pytest.approx(reference[key], rel=1e-9)


def test_class_weights_need_one_entry_per_slot(base_cfg):
    with pytest.raises(ValueError, match="frames35 slot order"):
        contact_loss(base_cfg, "frames35",
                     class_weights={"positive": [1.0] * 6, "negative": [1.0] * 6})


# ------------------------------------------------------------------ force: groups_mae

def force_loss(base_cfg, set_name: str) -> ForceLoss:
    cfg = cfg_for(base_cfg, set_name, force_supervision=dict(enabled=True, confidence=False))
    cfg["force_supervision"]["loss"].update(
        force=1.0, magnitude=0.0, direction=0.0, noncontact=0.0,
        sum_force=0.0, sum_torque=0.0)
    return ForceLoss(cfg, None, "cpu")


def test_groups_mae_sums_the_member_slots(base_cfg):
    """The four left-hand slots' forces are one group vector on both sides."""
    loss = force_loss(base_cfg, "frames35")
    n = 1
    gt = torch.zeros(n, K, 3)
    pred = torch.zeros(n, K, 3)
    contact = torch.zeros(n, K, dtype=torch.bool)
    contact[0, LEFT_HAND] = True
    # Each member is 0.25 bw off along x, but the four errors cancel in pairs: the
    # GROUP error is zero while the per-slot MAE is 0.25.
    for i, slot in enumerate(LEFT_HAND):
        gt[0, slot, 0] = 0.5
        pred[0, slot, 0] = 0.5 + (0.25 if i % 2 == 0 else -0.25)
    batch = {"force_gt": gt, "force_lever": torch.zeros(n, K, 3), "force_contact": contact,
             "force_valid": torch.ones(n, dtype=torch.bool),
             "frame_valid": torch.ones(n, dtype=torch.bool), "force_conf": torch.ones(n)}
    metrics = loss.metrics(loss({"force": {"forces": pred}}, batch, train=True).stats)
    assert metrics["mae"] == pytest.approx(0.25, rel=1e-5)
    assert metrics["groups_mae"] == pytest.approx(0.0, abs=1e-6)

    # One member off by 0.4 with the others exact: the group carries the whole error,
    # and only the one group with an in-contact member is scored.
    pred = gt.clone()
    pred[0, LEFT_HAND[0], 1] = 0.4
    metrics = loss.metrics(loss({"force": {"forces": pred}}, batch, train=True).stats)
    assert metrics["groups_mae"] == pytest.approx(0.4, rel=1e-5)
    assert metrics["mae"] == pytest.approx(0.4 / 4, rel=1e-5)


def test_groups_mae_is_the_slot_mae_under_kindyn6(base_cfg):
    torch.manual_seed(9)
    loss = force_loss(base_cfg, "kindyn6")
    n = 8
    contact = torch.rand(n, 6) > 0.3
    gt = 0.5 * torch.randn(n, 6, 3) * contact[..., None].float()
    batch = {"force_gt": gt, "force_lever": torch.zeros(n, 6, 3), "force_contact": contact,
             "force_valid": torch.ones(n, dtype=torch.bool),
             "frame_valid": torch.ones(n, dtype=torch.bool), "force_conf": torch.ones(n)}
    metrics = loss.metrics(
        loss({"force": {"forces": 0.4 * torch.randn(n, 6, 3)}}, batch, train=True).stats)
    assert metrics["groups_mae"] == pytest.approx(metrics["mae"], rel=1e-9)


# ------------------------------------------------------------------ contact stillness

def test_stillness_reads_the_slot_points(base_cfg):
    """The loss moves with ``slot_points_world``; its floor with the slots' parent joints."""
    cfg = cfg_for(base_cfg, "frames35", contact_consistency=dict(
        enabled=True, weight=1.0, confidence_weights=True, stencil="forward"))
    loss = ContactConsistencyLoss(cfg, None, "cpu")
    seq_len, amplitude = 8, 0.02
    sign = torch.tensor([1.0 if t % 2 == 0 else -1.0 for t in range(seq_len)])
    slot = IDX["thumb_left"]
    points = torch.zeros(seq_len, K, 3)
    points[:, slot, 0] = amplitude * sign
    gt_joints = torch.zeros(seq_len, 52, 3)
    gt_joints[:, SLOTS.parent_joint52[slot], 0] = 0.5 * amplitude * sign
    contact = torch.zeros(seq_len, K)
    contact[:, slot] = 1.0
    batch = {"seq_len": seq_len,
             "frame_pos_sec": torch.arange(seq_len, dtype=torch.float32) / FPS,
             "frame_valid": torch.ones(seq_len, dtype=torch.bool),
             "contact_gt": contact, "contact_valid": torch.ones(seq_len, K),
             "contact_conf": torch.ones(seq_len, K),
             "smplx_valid": torch.ones(seq_len, dtype=torch.bool),
             "smplx_joints_world": gt_joints}
    points = points.requires_grad_(True)
    result = loss({"smplx": {"slot_points_world": points}}, batch, train=True)
    metrics = loss.metrics(result.stats)
    speed = 2.0 * amplitude * FPS
    assert metrics["speed"] == pytest.approx(speed, rel=1e-5)
    assert metrics["gt_speed"] == pytest.approx(0.5 * speed, rel=1e-5)
    # Only the labelled slot carries gradient.
    result.terms["still"].numerator.backward()
    assert float(points.grad[:, slot].abs().sum()) > 0.0
    others = [i for i in range(K) if i != slot]
    assert float(points.grad[:, others].abs().sum()) == 0.0


# ------------------------------------------------------------------ the wrench's lever

STATIC_FRAMES = 7
MID = STATIC_FRAMES // 2


def test_slot_point_off_its_joint_adds_its_moment(base_cfg):
    """A frames35 slot's force acts at its POINT: the offset's moment enters the torque."""
    cfg = cfg_for(base_cfg, "frames35", force_consistency=dict(
        enabled=True, gate_by_contact=False, smooth_sec=0.0))
    loss = ForceConsistencyLoss(cfg, None, "cpu")
    n, t = 1, STATIC_FRAMES
    pelvis = torch.zeros(n, t, 3)
    root_rot = torch.eye(3).expand(n, t, 3, 3).contiguous()
    body_rot = torch.eye(3).expand(n, t, 21, 3, 3).contiguous()
    betas = torch.zeros(n, 10)
    seconds = (torch.arange(t, dtype=torch.float32) / FPS)[None]
    valid = torch.ones(n, t, dtype=torch.bool)
    gravity = torch.tensor([[0.0, -1.0, 0.0]])
    forces = torch.zeros(n, t, K, 3)
    slot = IDX["heel_left"]
    forces[:, :, slot, 1] = 1.0                       # 1 bw straight up on one slot

    q = smplx_q(pelvis.reshape(n * t, 3), root_rot.reshape(n * t, 3, 3),
                body_rot.reshape(n * t, 21, 3, 3))
    joints = loss.wrench.body.with_shape(
        betas=betas[:, None].expand(n, t, 10).reshape(n * t, 10)
    ).fk(q).joint_pose_world[..., 1:, :3]
    at_joint = joints[:, list(loss.parent22)].view(n, t, K, 3)
    offset = at_joint.clone()
    lever = torch.tensor([0.0, 0.0, 0.1])             # 10 cm along +z
    offset[:, :, slot] = offset[:, :, slot] + lever

    parent22 = loss.parent22
    base = loss.wrench.residual(pelvis, root_rot, body_rot, betas, forces, at_joint,
                                parent22, gravity, seconds, valid)
    moved = loss.wrench.residual(pelvis, root_rot, body_rot, betas, forces, offset,
                                 parent22, gravity, seconds, valid)
    rows = base[2]
    assert bool(rows[0, MID])
    # The force residual is untouched; the torque residual loses exactly the moment the
    # offset applies, ``-(r x f)`` (the residual SUBTRACTS the external wrench), which
    # here is 0.1 m x 1 bw about the x axis in the root frame (identity).
    assert torch.allclose(base[0][rows], moved[0][rows], atol=1e-4)
    delta = (moved[1] - base[1])[0, MID]
    assert torch.allclose(delta, -torch.linalg.cross(lever, torch.tensor([0.0, 1.0, 0.0])),
                          atol=1e-3)
