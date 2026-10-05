"""The contact sets: WHERE contact and force are predicted on the body.

A contact set names the slots every contact / force tensor in this repository is
indexed by — the decoder tokens, the refiner's limb tokens, the heads' outputs
and the loaders' labels all agree on it. Two sets exist:

* ``kindyn6`` — the six kindyn groups (``left_hand, right_hand, left_foot (toe),
  right_foot, left_ankle (heel), right_ankle``): each slot is a BODY JOINT
  (the wrists, the big toes, the heels) and the decoder tokens anchor at an
  MHR70 keypoint. The round 5-11 protocol.
* ``frames35`` — the 35 named contact frames of the BetterVideoReconstruction
  body (``better_contacts.json``): each slot is a MESH VERTEX rigidly attached
  to its parent joint (BetterHuman contact frames, posed by forward kinematics),
  and the decoder tokens anchor at that vertex's image projection through the
  MHR body (the SMPL-X vertex is a barycentric point of one MHR triangle).

Both sets fold onto the six kindyn groups through the parent joint (the hand
groups take the wrist and every finger frame, the toe groups the foot joint's
frames, the heel groups the ankle's; the other frames belong to no group), which
is how a 35-slot prediction is scored against the six-group manual test labels
and the rig boards.

This module is import-light (numpy + json) so the config layer and the loaders
can read it; the torch helpers that pose the frames live in :mod:`model.refiner`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: The six kindyn groups, in kindyn's ``contact_force_joints`` column order.
KINDYN_GROUP_NAMES = (
    "left_hand", "right_hand", "left_foot", "right_foot", "left_ankle", "right_ankle",
)
NUM_KINDYN_GROUPS = len(KINDYN_GROUP_NAMES)
#: MHR70 keypoint anchoring each kindyn group's decoder token (wrists, big-toe tips, heels).
KINDYN_GROUP_KEYPOINTS = (62, 41, 15, 18, 17, 20)
#: SMPL-X body joint of each kindyn group (wrists 20/21, big toes 10/11, heels 7/8).
KINDYN_GROUP_JOINTS = (20, 21, 10, 11, 7, 8)
#: 52-joint membership of each group (the hands aggregate wrist + 15 fingers).
LEFT_HAND_GROUP_52 = (20,) + tuple(range(22, 37))
RIGHT_HAND_GROUP_52 = (21,) + tuple(range(37, 52))
KINDYN_GROUPS_52 = (LEFT_HAND_GROUP_52, RIGHT_HAND_GROUP_52, (10,), (11,), (7,), (8,))

#: The authored frame set of the 35 contact frames (BetterHuman studio data; the same
#: file BetterVideoReconstruction poses its labels and forces with).
FRAMES35_JSON = Path(
    "/home/rikhat.akizhanov/better/BetterHuman/studio_data/frames/smplx_neutral/better_contacts.json")
#: SMPL-X (10475 vertices) -> MHR: one MHR triangle id + barycentric weights per SMPL-X
#: vertex (BetterVideoReconstruction's ``mhr2smplx_mapping.npz``).
MHR_MAPPING_NPZ = Path(
    "/home/rikhat.akizhanov/better/BetterVideoReconstruction/third_party/MHR/tools/"
    "mhr_smpl_conversion/assets/mhr2smplx_mapping.npz")

#: The 35 frames: (name, parent joint in the 52-joint SMPL-X, SMPL-X vertex id), in the
#: order of ``contact_frame_names`` in every ``contacts*.npz`` / ``kindyn_1.npz`` /
#: ``forces.npz`` of both corpora. ``tests/test_contact_frames.py`` checks this table
#: against the JSON and BetterHuman's own parent resolution.
FRAMES35 = (
    ("heel_left", 7, 8924), ("heel_right", 8, 8712),
    ("toe_above_left", 10, 5895), ("toe_above_right", 11, 8589),
    ("palm_left", 20, 4704), ("palm_right", 21, 7440),
    ("finger_right", 42, 7813), ("finger_left", 27, 5077),
    ("sit_left", 0, 3463), ("sit_right", 0, 6224),
    ("elbow_left", 18, 4374), ("elbow_right", 19, 7110),
    ("back_right", 14, 8278), ("back_left", 13, 5566),
    ("ball_inner_left", 10, 5906), ("ball_outer_left", 10, 5916),
    ("ball_inner_right", 11, 8600), ("ball_outer_right", 11, 8610),
    ("big_toe_left", 10, 5847), ("big_toe_right", 11, 8541),
    ("knee_left", 4, 3677), ("knee_right", 5, 6438),
    ("sacrum", 0, 5613),
    ("shoulder_right", 17, 6822), ("shoulder_left", 16, 4078),
    ("hip_side_left", 1, 3958), ("hip_side_right", 2, 6706),
    ("chest", 6, 3855),
    ("pelvis_front_left", 0, 4420), ("pelvis_front_right", 0, 7156),
    ("head", 15, 9003),
    ("palm_distal_left", 25, 4903), ("palm_distal_right", 40, 7639),
    ("thumb_right", 51, 8114), ("thumb_left", 36, 5380),
)

CONTACT_SETS = ("kindyn6", "frames35")


def body22_parent(joint52: int) -> int:
    """The 22-body ancestor of a 52-joint index: finger joints fold onto their wrist."""
    if joint52 < 22:
        return joint52
    return 20 if joint52 < 37 else 21


def group_of_joint52(joint52: int) -> int:
    """Kindyn group index of a 52-joint parent, or ``-1`` when it belongs to none."""
    for group, members in enumerate(KINDYN_GROUPS_52):
        if joint52 in members:
            return group
    return -1


@dataclass(frozen=True)
class ContactSet:
    """One contact set: the slots, their parent joints and their image anchors.

    :param name: ``kindyn6`` | ``frames35``.
    :param slot_names: one name per slot.
    :param parent_joint52: parent joint of each slot in the 52-joint SMPL-X.
    :param vertex_ids: SMPL-X vertex of each slot (``frames35``), else ``None``.
    :param frame_json: the BetterHuman frame-set file to build the body's contact
        frames from (``frames35``), else ``None``.
    """

    name: str
    slot_names: tuple[str, ...]
    parent_joint52: tuple[int, ...]
    vertex_ids: tuple[int, ...] | None
    frame_json: Path | None

    @property
    def count(self) -> int:
        return len(self.slot_names)

    @property
    def parent_joint22(self) -> tuple[int, ...]:
        """Parent of each slot in the 22-joint body (the dynamics body)."""
        return tuple(body22_parent(j) for j in self.parent_joint52)

    @property
    def group_of(self) -> np.ndarray:
        """``(count,)`` kindyn group of each slot, ``-1`` for none (the six-group fold)."""
        return np.asarray([group_of_joint52(j) for j in self.parent_joint52], np.int64)

    @property
    def uses_frames(self) -> bool:
        """Whether the slots are posed contact frames (else body joints)."""
        return self.vertex_ids is not None

    def fold_max(self, values: np.ndarray) -> np.ndarray:
        """``(..., count) -> (..., 6)``: the max over each group's member slots (probabilities
        / logits / labels); a group with no member is ``-inf``."""
        values = np.asarray(values)
        out = np.full(values.shape[:-1] + (NUM_KINDYN_GROUPS,), -np.inf, values.dtype)
        group_of = self.group_of
        for group in range(NUM_KINDYN_GROUPS):
            members = np.flatnonzero(group_of == group)
            if members.size:
                out[..., group] = values[..., members].max(axis=-1)
        return out

    def fold_sum(self, values: np.ndarray) -> np.ndarray:
        """``(..., count, k) -> (..., 6, k)``: the sum over each group's member slots (forces)."""
        values = np.asarray(values)
        out = np.zeros(values.shape[:-2] + (NUM_KINDYN_GROUPS,) + values.shape[-1:], values.dtype)
        group_of = self.group_of
        for group in range(NUM_KINDYN_GROUPS):
            members = np.flatnonzero(group_of == group)
            if members.size:
                out[..., group, :] = values[..., members, :].sum(axis=-2)
        return out


def contact_set(name: str) -> ContactSet:
    """The contact set called ``name``."""
    if name == "kindyn6":
        return ContactSet("kindyn6", KINDYN_GROUP_NAMES, KINDYN_GROUP_JOINTS, None, None)
    if name == "frames35":
        return ContactSet(
            "frames35", tuple(f[0] for f in FRAMES35), tuple(f[1] for f in FRAMES35),
            tuple(f[2] for f in FRAMES35), FRAMES35_JSON)
    raise ValueError(f"unknown contact set {name!r}; choose from {CONTACT_SETS}")


def frames35_json_vertices() -> dict[str, int]:
    """``{name: vertex}`` of the authored JSON, in authored order (for the consistency test)."""
    raw = json.loads(FRAMES35_JSON.read_text())
    return {name: (entry["vertex"] if isinstance(entry, dict) else int(entry))
            for name, entry in raw.items()}


__all__ = [
    "CONTACT_SETS", "ContactSet", "FRAMES35", "FRAMES35_JSON", "KINDYN_GROUPS_52",
    "KINDYN_GROUP_JOINTS", "KINDYN_GROUP_KEYPOINTS", "KINDYN_GROUP_NAMES", "MHR_MAPPING_NPZ",
    "NUM_KINDYN_GROUPS", "body22_parent", "contact_set", "frames35_json_vertices",
    "group_of_joint52",
]
