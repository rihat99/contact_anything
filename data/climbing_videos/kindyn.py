"""kindyn ground truth: contact forces and the SMPL-X body.

``features/human_optim/<shard>/<scene>/kindyn_1.npz`` is the per-scene inverse
dynamics solve. Two products are read here.

**Forces.** The solve places wrenches on ~35 named contact frames (world-frame
newtons). Each frame maps to one of the six groups through its PARENT JOINT:
the hand groups aggregate the wrist plus every finger frame, the foot groups
the big-toe/ball frames, the ankle groups the heels. Frames whose parent
belongs to no group (knees, elbows, back, ...) are dropped — corpus-wide they
carry ~4 % of the total force magnitude. Forces are divided by ``total_mass *
g`` (body-weight units) and rotated into the body-root frame by the kindyn root
quaternion, so no extrinsics enter the objective. ``force_lever`` is each
group joint's offset from the pelvis in the same frame (metres).

**SMPL-X body.** The fitted ``q`` trajectory of BetterHuman's
``SMPLX(use_face=False, use_hands=True, num_betas=10)`` — root = pelvis pose,
parent-local joint quaternions — plus world joints and per-person betas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .scene import (
    GROUP_NAMES,
    LEFT_HAND_GROUP_52,
    N_JOINTS_52,
    NUM_GROUPS,
    RIGHT_HAND_GROUP_52,
    rows_by_object_id,
)

GRAVITY_MAG = 9.81
#: The joint names kindyn stores for the six group columns (hands by wrist).
KINDYN_FORCE_JOINTS = (
    "left_wrist", "right_wrist", "left_foot", "right_foot", "left_ankle", "right_ankle",
)
#: 52-joint membership per group (hands aggregate wrist + fingers).
GROUPS_52 = (LEFT_HAND_GROUP_52, RIGHT_HAND_GROUP_52, (10,), (11,), (7,), (8,))


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Rotation matrices from ``xyzw`` quaternions (normalized internally).

    The kindyn root quaternion ``q[..., 3:7]`` uses this layout and ``R(q)`` is
    world-from-root. ``(..., 4) -> (..., 3, 3)`` float32.
    """
    quat = np.asarray(quat, np.float32)
    quat = quat / np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8, None)
    x, y, z, w = (quat[..., i] for i in range(4))
    rot = np.empty(quat.shape[:-1] + (3, 3), np.float32)
    rot[..., 0, 0] = 1 - 2 * (y * y + z * z)
    rot[..., 0, 1] = 2 * (x * y - z * w)
    rot[..., 0, 2] = 2 * (x * z + y * w)
    rot[..., 1, 0] = 2 * (x * y + z * w)
    rot[..., 1, 1] = 1 - 2 * (x * x + z * z)
    rot[..., 1, 2] = 2 * (y * z - x * w)
    rot[..., 2, 0] = 2 * (x * z - y * w)
    rot[..., 2, 1] = 2 * (y * z + x * w)
    rot[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


# ------------------------------------------------------------------ forces

def load_forces(scene: str, human_dir: Path, object_ids: np.ndarray, n: int) -> dict:
    """Six-group GT contact forces in body-weight units, body-root frame.

    :returns: ``force_gt (P, N, 6, 3)``, ``force_contact (P, N, 6)`` bool,
        ``force_lever (P, N, 6, 3)`` metres, ``force_valid (P, N)``,
        ``force_conf (P, N)``.
    """
    kindyn = np.load(human_dir / "kindyn_1.npz", allow_pickle=True)
    kindyn_ids = np.asarray(kindyn["object_ids"])
    frame_names = [str(x) for x in kindyn["contact_frame_names"]]
    parents = np.asarray(kindyn["contact_frame_parents"], np.int64).reshape(-1)
    n_cframes = len(frame_names)
    if parents.shape != (n_cframes,) or (parents < 0).any() or (
            parents >= N_JOINTS_52).any():
        raise ValueError(
            f"{scene}: contact_frame_parents is not {n_cframes} valid 52-joint indices")
    group_of = np.full(n_cframes, -1, np.int64)
    for g, members in enumerate(GROUPS_52):
        group_of[np.isin(parents, list(members))] = g
    for g, name in enumerate(GROUP_NAMES):
        if not (group_of == g).any():
            raise ValueError(
                f"{scene}: no kindyn contact frame maps to force group {name!r}")

    def _rows(key, dtype):
        return rows_by_object_id(
            np.asarray(kindyn[key], dtype), kindyn_ids, object_ids, scene, "kindyn")

    frame_forces = _rows("frame_forces", np.float32)          # [P, N, F, 3] world newtons
    frame_contact = _rows("frame_contact", bool)              # [P, N, F]
    q = _rows("q", np.float32)                                # [P, N, 211]
    total_mass = rows_by_object_id(
        np.asarray(kindyn["total_mass"], np.float32).reshape(-1),
        kindyn_ids, object_ids, scene, "kindyn")              # [P] kg
    force_valid = _rows("valid_mask", bool)                   # [P, N]
    force_conf = _rows("force_confidence", np.float32)        # [P, N]
    joints_world = _rows("joints_world", np.float32)          # [P, N, J, 3] world m
    joint_names = [str(x) for x in kindyn["joint_names"]]

    n_people = len(object_ids)
    if frame_forces.shape != (n_people, n, n_cframes, 3):
        raise ValueError(
            f"{scene}: frame_forces {frame_forces.shape} does not match "
            f"({n_people}, {n}, {n_cframes}, 3)")
    if frame_contact.shape != (n_people, n, n_cframes):
        raise ValueError(
            f"{scene}: frame_contact {frame_contact.shape} does not match "
            f"({n_people}, {n}, {n_cframes})")
    if joints_world.shape != (n_people, n, len(joint_names), 3):
        raise ValueError(
            f"{scene}: joints_world {joints_world.shape} does not match "
            f"({n_people}, {n}, {len(joint_names)}, 3)")
    missing_joints = [name for name in ("pelvis",) + KINDYN_FORCE_JOINTS
                      if name not in joint_names]
    if missing_joints:
        raise ValueError(f"{scene}: kindyn joint_names is missing {missing_joints}")
    if not np.isfinite(frame_forces).all():
        raise ValueError(f"{scene}: frame_forces contain non-finite values")
    if not np.isfinite(force_conf).all():
        raise ValueError(f"{scene}: force_confidence contains non-finite values")
    force_conf = np.clip(force_conf, 0.0, 1.0).astype(np.float32)
    if (not np.isfinite(total_mass).all() or (total_mass <= 0).any()
            or not np.isfinite(np.asarray(kindyn["betas"])).all()):
        raise ValueError(f"{scene}: kindyn total_mass/betas are not sane")

    # Fold frames -> groups: forces sum, contact ORs, over the member frames.
    forces_n = np.stack(
        [frame_forces[:, :, group_of == g].sum(axis=2) for g in range(NUM_GROUPS)],
        axis=2)                                               # [P, N, 6, 3] world newtons
    group_contact = np.stack(
        [frame_contact[:, :, group_of == g].any(axis=2) for g in range(NUM_GROUPS)],
        axis=2)                                               # [P, N, 6]
    # Forces are only ever solved under the contact mask: a nonzero force on an
    # uncontacted group means corrupted data. (Zero force during contact is
    # possible in principle, so the converse is not asserted.)
    if bool((np.linalg.norm(forces_n, axis=-1) > 0)[~group_contact].any()):
        raise ValueError(
            f"{scene}: nonzero contact force on a group with no contact label")

    forces_out = forces_n / (total_mass[:, None, None, None] * GRAVITY_MAG)
    # Lever arms for the net-torque term: the six group joints' offsets from the
    # pelvis. Not checked for finiteness — uncovered frames may hold garbage and
    # the loss skips them.
    pelvis = joint_names.index("pelvis")
    group_joints = [joint_names.index(name) for name in KINDYN_FORCE_JOINTS]
    lever = joints_world[:, :, group_joints] - joints_world[:, :, [pelvis]]
    # q[3:7] is the root quaternion, xyzw, R(q) = world-from-root (verified
    # against the stored axis-angle global_orient); rotate world -> root.
    rot = quat_xyzw_to_matrix(q[..., 3:7])                    # [P, N, 3, 3]
    forces_out = np.einsum("pnji,pnkj->pnki", rot, forces_out)
    lever = np.einsum("pnji,pnkj->pnki", rot, lever)
    return {
        "force_gt": forces_out.astype(np.float32),
        "force_contact": group_contact,
        "force_lever": lever.astype(np.float32),
        "force_valid": force_valid,
        "force_conf": force_conf,
    }


# ------------------------------------------------------------------ SMPL-X

#: Body joints of the corpus SMPL-X (``smplx_mid``): pelvis + 21 articulated.
NUM_SMPLX_BODY_JOINTS = 22
#: Finger joints: 15 per hand (index, middle, pinky, ring, thumb x 3), left then right.
NUM_SMPLX_HAND_JOINTS = 30
NUM_SMPLX_JOINTS = NUM_SMPLX_BODY_JOINTS + NUM_SMPLX_HAND_JOINTS
NUM_SMPLX_BETAS = 10
#: ``q`` layout: ``[pelvis_world (3), root quat xyzw (4), 51 x joint quat xyzw]``
#: (21 body joints, then the 30 finger joints).
SMPLX_Q_DIM = 211
_SMPLX_BODY_Q = slice(7, 7 + 4 * (NUM_SMPLX_BODY_JOINTS - 1))
_SMPLX_HAND_Q = slice(_SMPLX_BODY_Q.stop, _SMPLX_BODY_Q.stop + 4 * NUM_SMPLX_HAND_JOINTS)


def load_smplx(scene: str, human_dir: Path, object_ids: np.ndarray, n: int) -> dict:
    """SMPL-X body GT from ``kindyn_1.npz`` (BetterHuman ``q`` convention).

    The root of ``q`` IS the pelvis pose: ``q[:3]`` equals ``joints_world[0]``
    and ``q[3:7]`` is the world-from-root quaternion (the stored classic
    ``transl`` differs from it only by the shape-dependent pelvis offset
    ``J0(beta)``). Joint rotations are parent-local: the 21 body joints and
    the 30 finger joints (raw local rotations — the classic hand mean is not
    part of the ``q`` convention); face and expression are dropped (zero
    corpus-wide). Invalid rows are zeroed / set to the identity so nothing
    downstream ever multiplies a NaN by a zero mask.

    :returns: ``smplx_joints_world (P, N, 52, 3)`` metres (22 body joints,
        then the 30 finger joints), ``smplx_root_rot (P, N, 3, 3)``
        world-from-root, ``smplx_body_rot (P, N, 21, 3, 3)`` parent-local
        joints 1..21, ``smplx_hand_rot (P, N, 30, 3, 3)`` parent-local finger
        joints, ``smplx_betas (P, 10)`` per person, ``smplx_valid (P, N)`` bool.
    """
    kindyn = np.load(human_dir / "kindyn_1.npz", allow_pickle=True)
    if str(kindyn["model_type"]) != "smplx_mid" or int(kindyn["num_betas"]) != NUM_SMPLX_BETAS:
        raise ValueError(
            f"{scene}: kindyn body is {str(kindyn['model_type'])!r} with "
            f"{int(kindyn['num_betas'])} betas; expected smplx_mid / {NUM_SMPLX_BETAS}")
    kindyn_ids = np.asarray(kindyn["object_ids"])

    def _rows(key, dtype):
        return rows_by_object_id(
            np.asarray(kindyn[key], dtype), kindyn_ids, object_ids, scene, "kindyn")

    q = _rows("q", np.float32)                                # [P, N, 211]
    valid = _rows("valid_mask", bool)                         # [P, N]
    joints = _rows("joints_world", np.float32)                # [P, N, 52, 3]
    betas = _rows("betas", np.float32)                        # [P, 10]
    n_people = len(object_ids)
    if q.shape != (n_people, n, SMPLX_Q_DIM):
        raise ValueError(f"{scene}: kindyn q {q.shape} != ({n_people}, {n}, {SMPLX_Q_DIM})")
    if joints.shape != (n_people, n, NUM_SMPLX_JOINTS, 3):
        raise ValueError(
            f"{scene}: kindyn joints_world {joints.shape} is not (P, N, {NUM_SMPLX_JOINTS}, 3)")
    if betas.shape != (n_people, NUM_SMPLX_BETAS) or not np.isfinite(betas).all():
        raise ValueError(f"{scene}: kindyn betas {betas.shape} are not (P, 10) finite")
    valid = valid & np.isfinite(q).all(axis=-1) & np.isfinite(joints).all(axis=(2, 3))

    q = np.where(valid[..., None], q, 0.0).astype(np.float32)
    root_rot = quat_xyzw_to_matrix(q[..., 3:7])                             # [P, N, 3, 3]
    body_rot = quat_xyzw_to_matrix(
        q[..., _SMPLX_BODY_Q].reshape(n_people, n, NUM_SMPLX_BODY_JOINTS - 1, 4))
    hand_rot = quat_xyzw_to_matrix(
        q[..., _SMPLX_HAND_Q].reshape(n_people, n, NUM_SMPLX_HAND_JOINTS, 4))
    eye = np.eye(3, dtype=np.float32)
    root_rot = np.where(valid[..., None, None], root_rot, eye)
    body_rot = np.where(valid[..., None, None, None], body_rot, eye)
    hand_rot = np.where(valid[..., None, None, None], hand_rot, eye)
    joints = np.where(valid[..., None, None], joints, 0.0)
    return {
        "smplx_joints_world": joints.astype(np.float32),
        "smplx_root_rot": root_rot.astype(np.float32),
        "smplx_body_rot": body_rot.astype(np.float32),
        "smplx_hand_rot": hand_rot.astype(np.float32),
        "smplx_betas": betas,
        "smplx_valid": valid,
    }
