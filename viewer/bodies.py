"""The three SMPL-X body sources of a test scene, posed in the metric world frame.

Every source becomes the same skinning payload — one rest mesh + rest skeleton
per person and per-frame world joint transforms — so the browser does the LBS
(one upload per body, 52 bone poses per frame) and the viewer treats the
sources identically:

* ``gt`` — the kindyn SMPL-X trajectory (``human_optim/kindyn_1.npz``: world
  ``q (P, N, 211)``, one ``betas`` per person). Exact: identity and pose are the
  archive's.
* ``frozen`` — the frozen SAM 3D Body prediction refit to SMPL-X by the corpus
  pipeline (``sam3d/smplx_params.npz``, classic smplx-PyPI params in the
  CAMERA frame), converted per frame to the BetterHuman ``q`` and folded into
  the world with the frame's extrinsics.
* ``predicted`` — a run's ``predictions/<scene>.npz`` (``scripts/predict_test.py``):
  the SMPL-X head's camera-frame ``q_cam`` per frame, folded into the world
  the same way.

Frozen and predicted bodies regress ``betas`` per frame; the mesh is uploaded
once at the person's MEDIAN identity over its frames, while the skeleton (and
the bone transforms that pose the mesh) comes from the exact per-frame FK. The
browser LBS also drops SMPL-X's pose correctives (~6 mm mean on a climbing pose).

World-frame bodies are what the ``world`` regime shows; the ``camera`` regime
re-expresses the SAME transforms with each frame's ``cam_from_world`` (see
:mod:`viewer.scene`), which for the frozen and predicted bodies is exactly
their native camera-frame output.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

#: The SMPL-X archive every model in this repo is built on (``model.smplx.model_path``).
SMPLX_MODEL_PATH = "/data3/rikhat.akizhanov/better/BetterHuman/models/smplx/SMPLX_NEUTRAL.npz"
NUM_JOINTS = 52
NUM_BODY_JOINTS = 22
Q_FULL = 211
_BODY_CACHE: dict = {}


@dataclass
class Person:
    """One person's skinning data (``N`` = the scene's frame count)."""

    oid: int
    v_shaped: np.ndarray      # (V, 3) rest mesh at the display identity
    j_rest: np.ndarray        # (52, 3) rest joints of that identity
    weights: np.ndarray       # (V, 52) top-4 skin weights, rows sum to 1
    bone_wxyz: np.ndarray     # (N, 52, 4) world joint rotations, NaN where invalid
    bone_pos: np.ndarray      # (N, 52, 3) world joint positions, NaN where invalid
    valid: np.ndarray         # (N,) bool
    betas: np.ndarray         # (10,) the display identity
    betas_std: float          # per-frame betas spread (0 for a per-person identity)


@dataclass
class BodySource:
    """One body source of a scene: shared topology + one :class:`Person` per tracked person."""

    name: str
    faces: np.ndarray         # (F, 3) int32
    parents: np.ndarray       # (52,) int32
    joint_names: tuple
    object_ids: np.ndarray    # (P,) dataset person order
    people: list              # [Person | None] in ``object_ids`` order


def load_body(device: torch.device | str = "cpu", model_path: str = SMPLX_MODEL_PATH):
    """The 52-joint BetterHuman SMPL-X body, built once per device."""
    import better_human as bh

    key = (str(device), model_path)
    if key not in _BODY_CACHE:
        _BODY_CACHE[key] = bh.SMPLX(
            model_path=model_path, gender="neutral", num_betas=10, use_hands=True,
            use_face=False, compute_mass=False, dtype=torch.float32, device=device)
    return _BODY_CACHE[key]


def top4_weights(dense: np.ndarray) -> np.ndarray:
    """Keep each vertex's four largest skin weights, renormalised (viser keeps top-4 unnormalised)."""
    rows = np.arange(dense.shape[0])[:, None]
    keep = np.argsort(dense, axis=1)[:, -4:]
    out = np.zeros_like(dense)
    out[rows, keep] = dense[rows, keep]
    return (out / out.sum(1, keepdims=True)).astype(np.float32)


def rows_by_id(source_ids: np.ndarray, wanted: int) -> int | None:
    ids = [int(x) for x in np.asarray(source_ids).reshape(-1)]
    return ids.index(int(wanted)) if int(wanted) in ids else None


def _fk_world(body, betas: torch.Tensor, q: torch.Tensor,
              world_from_cam: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame world joint poses ``(B, 52, 4) xyzw, (B, 52, 3)`` of ``q`` at ``betas``.

    :param betas: ``(B, 10)`` per-frame identities (the FK is exact per frame).
    :param q: ``(B, 211)`` BetterHuman configuration — world, or camera when
        ``world_from_cam`` ``(B, 4, 4)`` is given (then folded into the world).
    """
    from better_robot.lie import so3

    shaped = body.with_shape(betas=betas)
    poses = shaped.fk(q).joint_pose_world[:, 1:]                          # drop the universe row
    quat, pos = poses[..., 3:7], poses[..., :3]
    if world_from_cam is None:
        return quat, pos
    rot = world_from_cam[:, None, :3, :3] @ so3.to_matrix(quat)            # (B, 52, 3, 3)
    pos = (world_from_cam[:, None, :3, :3] @ pos[..., None])[..., 0] + world_from_cam[:, None, :3, 3]
    return so3.from_matrix(rot), pos


def skin_person(body, oid: int, betas: np.ndarray, q: np.ndarray, valid: np.ndarray,
                world_from_cam: np.ndarray | None, device) -> Person | None:
    """Skinning data of one person from per-frame ``q (N, 211)`` and ``betas (N, 10)``.

    Rows outside ``valid`` are ignored (they may be NaN). The mesh identity is the
    median of the valid rows' betas; the bone poses are the exact per-frame FK.
    """
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return None
    n = len(valid)
    betas_v = np.asarray(betas, np.float32)[idx]
    identity = np.median(betas_v, axis=0).astype(np.float32)
    spread = float(betas_v.std(axis=0).mean()) if idx.size > 1 else 0.0
    with torch.no_grad():
        q_t = torch.as_tensor(np.asarray(q, np.float32)[idx], device=device)
        b_t = torch.as_tensor(betas_v, device=device)
        wfc = (None if world_from_cam is None
               else torch.as_tensor(np.asarray(world_from_cam, np.float32)[idx], device=device))
        quat, pos = _fk_world(body, b_t, q_t, wfc)
        shaped = body.with_shape(betas=torch.as_tensor(identity, device=device)[None])
        v_shaped = shaped.values.v_shaped[0].cpu().numpy()
        j_rest = shaped.values.rest_joints[0].cpu().numpy()
        weights = top4_weights(shaped.values.skinning_weight_matrix.cpu().numpy())
    bone_wxyz = np.full((n, NUM_JOINTS, 4), np.nan, np.float32)
    bone_pos = np.full((n, NUM_JOINTS, 3), np.nan, np.float32)
    bone_wxyz[idx] = quat.cpu().numpy()[..., [3, 0, 1, 2]]                 # xyzw -> wxyz
    bone_pos[idx] = pos.cpu().numpy()
    return Person(oid=int(oid), v_shaped=v_shaped.astype(np.float32),
                  j_rest=j_rest.astype(np.float32), weights=weights,
                  bone_wxyz=bone_wxyz, bone_pos=bone_pos, valid=np.asarray(valid, bool),
                  betas=identity, betas_std=spread)


def _topology(body) -> tuple[np.ndarray, np.ndarray, tuple]:
    faces = body.structure.faces.cpu().numpy().astype(np.int32)
    parents = np.asarray(list(body.structure.parents), np.int32)
    return faces, parents, tuple(body.structure.joint_names)


def gt_source(kindyn_path: Path, object_ids: np.ndarray, n_frames: int, device) -> BodySource:
    """The kindyn SMPL-X GT (world ``q``, one identity per person)."""
    body = load_body(device)
    faces, parents, names = _topology(body)
    kd = np.load(kindyn_path, allow_pickle=True)
    q_all, betas_all = np.asarray(kd["q"], np.float32), np.asarray(kd["betas"], np.float32)
    valid_all = np.asarray(kd["valid_mask"], bool)
    if q_all.shape[1] != n_frames:
        raise ValueError(f"{kindyn_path}: {q_all.shape[1]} frames but the scene has {n_frames}")
    people = []
    for oid in object_ids:
        row = rows_by_id(kd["object_ids"], oid)
        if row is None:
            people.append(None)
            continue
        n = q_all.shape[1]
        betas = np.repeat(betas_all[row][None], n, axis=0)
        people.append(skin_person(body, oid, betas, q_all[row], valid_all[row], None, device))
    return BodySource("gt", faces, parents, names, np.asarray(object_ids, np.int32), people)


def frozen_source(params_path: Path, extrinsics: np.ndarray, object_ids: np.ndarray,
                  device) -> BodySource:
    """The frozen SAM 3D Body refit (classic camera-frame params) folded into the world."""
    from better_human.bodies.smpl_family.smplx import SMPLXClassic

    body = load_body(device)
    faces, parents, names = _topology(body)
    sx = np.load(params_path)
    valid_all = np.asarray(sx["valid_mask"], bool)
    n = valid_all.shape[1]
    world_from_cam = np.linalg.inv(np.asarray(extrinsics, np.float64)).astype(np.float32)
    if len(world_from_cam) != n:
        raise ValueError(f"{params_path}: {n} frames but {len(world_from_cam)} extrinsics")
    people = []
    for oid in object_ids:
        row = rows_by_id(sx["object_ids"], oid)
        if row is None or not valid_all[row].any():
            people.append(None)
            continue
        idx = np.flatnonzero(valid_all[row])

        def tensor(key):
            return torch.as_tensor(np.ascontiguousarray(sx[key][row][idx], np.float32), device=device)

        with torch.no_grad():
            shaped = body.with_shape(betas=tensor("betas"))
            q_cam = shaped.from_classic(SMPLXClassic(
                global_orient=tensor("global_orient"), body_pose=tensor("body_pose"),
                transl=tensor("transl"), left_hand_pose=tensor("left_hand_pose"),
                right_hand_pose=tensor("right_hand_pose"), num_pca_comps=None)).cpu().numpy()
        q_full = np.full((n, Q_FULL), np.nan, np.float32)
        q_full[idx] = q_cam
        people.append(skin_person(body, oid, sx["betas"][row], q_full, valid_all[row],
                                  world_from_cam, device))
    return BodySource("frozen", faces, parents, names, np.asarray(object_ids, np.int32), people)


def predicted_source(npz_path: Path, extrinsics: np.ndarray, device) -> BodySource:
    """A run's dumped SMPL-X head output (camera-frame ``q_cam``) folded into the world."""
    body = load_body(device)
    faces, parents, names = _topology(body)
    pred = np.load(npz_path)
    covered = np.asarray(pred["covered"], bool)
    object_ids = np.asarray(pred["object_ids"], np.int32)
    world_from_cam = np.linalg.inv(np.asarray(extrinsics, np.float64)).astype(np.float32)
    if len(world_from_cam) != covered.shape[1]:
        raise ValueError(f"{npz_path}: {covered.shape[1]} frames but {len(world_from_cam)} extrinsics")
    people = [skin_person(body, oid, pred["betas"][p], pred["q_cam"][p], covered[p],
                          world_from_cam, device)
              for p, oid in enumerate(object_ids)]
    return BodySource("predicted", faces, parents, names, object_ids, people)
