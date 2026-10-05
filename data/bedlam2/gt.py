"""BEDLAM2_our ground truth: contact forces and the SMPL-X body.

``features/gt/<shard>/<scene>/forces.npz`` is the frozen-body inverse dynamics
solve on the renderer's own trajectory, in exactly kindyn's conventions: world
newtons on the 35 named contact frames, the ``q`` trajectory of BetterHuman's
``SMPLX(use_face=False, use_hands=True, num_betas=10)`` and the 52 world joints.
Forces are divided by ``total_mass * g`` and rotated into the body-root frame by
the root quaternion, as in :mod:`data.climbing_videos.kindyn`; ``force_lever`` is
each slot's PARENT joint's offset from the pelvis in the same frame.

Unlike the corpus there is no label confidence — the renderer's contacts and the
solve's forces are ground truth, so ``force_conf`` is ``1`` everywhere — and
``gravity.npz`` is always a MEASUREMENT (the scene's own up axis).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from model.contact_frames import ContactSet, contact_set

from ..climbing_videos.kindyn import GRAVITY_MAG, quat_xyzw_to_matrix, smplx_fields


def load_gravity(directory: Path, scene: str) -> dict:
    """``gravity_world`` (unit down, world) and ``gravity_measured`` of one scene."""
    gravity = np.load(directory / "gravity.npz", allow_pickle=True)
    down = np.asarray(gravity["gravity_world"], np.float32).reshape(-1)
    if down.shape != (3,) or not np.isfinite(down).all() or not (
            0.99 < float(np.linalg.norm(down)) < 1.01):
        raise ValueError(f"{scene}: gravity_world is not a finite unit 3-vector: {down}")
    return {"gravity_world": (down / np.linalg.norm(down)).astype(np.float32),
            "gravity_measured": bool(np.asarray(gravity["reliable"]).item())}


def load_forces(scene: str, directory: Path, n: int, *, slots: ContactSet) -> dict:
    """Per-slot GT contact forces in body-weight units, body-root frame.

    :returns: ``force_gt (P, N, K, 3)``, ``force_contact (P, N, K)`` bool,
        ``force_lever (P, N, K, 3)`` metres, ``force_valid (P, N)``,
        ``force_conf (P, N)`` (all ones), ``gravity_world (3,)``,
        ``gravity_measured`` bool.
    """
    forces = np.load(directory / "forces.npz", allow_pickle=True)
    frame_forces = np.asarray(forces["frame_forces"], np.float32)   # [P, N, 35, 3] world N
    frame_contact = np.asarray(forces["frame_contact"], bool)       # [P, N, 35]
    joints_world = np.asarray(forces["joints_world"], np.float32)   # [P, N, 52, 3]
    q = np.asarray(forces["q"], np.float32)                         # [P, N, 211]
    total_mass = np.asarray(forces["total_mass"], np.float32).reshape(-1)     # [P] kg
    valid = np.asarray(forces["valid_mask"], bool)                  # [P, N]
    n_people = len(total_mass)
    n_cframes = frame_contact.shape[-1]
    joint_names = [str(x) for x in forces["joint_names"]]

    if frame_forces.shape != (n_people, n, n_cframes, 3):
        raise ValueError(
            f"{scene}: frame_forces {frame_forces.shape} does not match "
            f"({n_people}, {n}, {n_cframes}, 3)")
    if joints_world.shape[:2] != (n_people, n) or joints_world.shape[2] < 52:
        raise ValueError(
            f"{scene}: joints_world {joints_world.shape} is not (P, N, >=52, 3)")
    if not np.isfinite(frame_forces).all():
        raise ValueError(f"{scene}: frame_forces contain non-finite values")
    if not np.isfinite(total_mass).all() or (total_mass <= 0).any():
        raise ValueError(f"{scene}: total_mass is not positive and finite")
    if joint_names[0] != "pelvis":
        raise ValueError(f"{scene}: joint 0 is {joint_names[0]!r}, not the pelvis")

    if slots.uses_frames:
        forces_n, slot_contact = frame_forces, frame_contact
    else:
        # The solve is always per contact FRAME; kindyn6 folds it onto the groups.
        frames = contact_set("frames35")
        forces_n = frames.fold_sum(frame_forces)
        slot_contact = frames.fold_max(frame_contact.astype(np.float32)) > 0.5
    if bool((np.linalg.norm(forces_n, axis=-1) > 0)[~slot_contact].any()):
        raise ValueError(f"{scene}: nonzero contact force on a slot with no contact label")

    forces_out = forces_n / (total_mass[:, None, None, None] * GRAVITY_MAG)
    lever = joints_world[:, :, list(slots.parent_joint52)] - joints_world[:, :, [0]]
    rot = quat_xyzw_to_matrix(q[..., 3:7])                          # [P, N, 3, 3]
    return {
        "force_gt": np.einsum("pnji,pnkj->pnki", rot, forces_out).astype(np.float32),
        "force_contact": slot_contact,
        "force_lever": np.einsum("pnji,pnkj->pnki", rot, lever).astype(np.float32),
        "force_valid": valid,
        "force_conf": np.ones((n_people, n), np.float32),
        **load_gravity(directory, scene),
    }


def load_smplx(scene: str, directory: Path, n: int) -> dict:
    """SMPL-X body GT: ``smplx.npz``'s ``q`` / ``betas`` and ``forces.npz``'s world joints.

    Both files carry the same trajectory; the joints are taken from the solve so
    they are the exact positions its wrenches were placed on.
    """
    smplx = np.load(directory / "smplx.npz", allow_pickle=True)
    forces = np.load(directory / "forces.npz", allow_pickle=True)
    if str(smplx["model_type"]) != "smplx_mid" or int(smplx["num_betas"]) != 10:
        raise ValueError(
            f"{scene}: body is {str(smplx['model_type'])!r} with "
            f"{int(smplx['num_betas'])} betas; expected smplx_mid / 10")
    joints = np.asarray(forces["joints_world"], np.float32)[:, :, :52]
    return {
        **load_gravity(directory, scene),
        **smplx_fields(
            scene, np.asarray(smplx["q"], np.float32),
            np.asarray(smplx["valid_mask"], bool) & np.asarray(forces["valid_mask"], bool),
            joints, np.asarray(smplx["betas"], np.float32), n),
    }
