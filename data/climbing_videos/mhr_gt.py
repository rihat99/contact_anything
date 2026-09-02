"""MHR-native ground truth: ``mhr_1.npz`` pose and ``mhr_sup_1.npz`` geometry.

``mhr_1.npz`` (``scripts/data/convert_kindyn_to_mhr.py``) is the kindyn SMPL-X
trajectory re-fitted as a world-frame MHR body: the ``q_world`` configuration
per frame, the shared per-person ``identity``, and the full fitted
``lbs_params`` row — the exact vector the MHR module is called with, so its
slots line up 1:1 with ``out["mhr"]["mhr_model_params"]``.

``mhr_sup_1.npz`` (``scripts/data/precompute_mhr_supervision.py``) is the SAM3D
model's OWN MHR module evaluated at those GT parameters: all 70 MHR70 keypoints
and a fixed 384-vertex template subset, in the metric world frame. GT and
prediction therefore come from the same rig and the same keypoint regressor —
there is no cross-rig bias to fight.

The same archives supply the motion root: the free-flyer whose twist the motion
target is, built as ``(mean-hips position, q_world root quaternion)``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .scene import rows_by_object_id

#: MHR70 keypoint count (the sapiens-308 regressor sliced to its first 70 rows).
NUM_MHR70 = 70
#: Vertex-subset size stored per frame (farthest-point sampled on the template).
NUM_SUP_VERTICES = 384
#: Pinned ``mhr_sup_1.npz`` schema.
MHR_SUP_SCHEMA = 1
#: Minimum ``mhr_1.npz`` converter version (the first storing ``lbs_params``).
MHR_CONVERTER_VERSION = 3
#: ``lbs_params`` slices. Slots 130..135 are the flexible bone-geometry params
#: (spine/neck/shoulder-width/arm/hip-width/leg lengths) — the tail of the pose
#: head's own 136-dim pose block; slots 136..203 are the 68 per-person scale
#: slots the head reaches through ``scale_mean + coeffs @ scale_comps``.
MHR_BONE_SLOTS = slice(130, 136)
MHR_SCALE_SLOTS = slice(136, 204)
NUM_MHR_BONES = 6
NUM_MHR_SCALES = 68
#: MHR70 left/right hip keypoints. Their mean is the body placement the pose
#: predictions are lifted from, so it is also the motion root's position half.
HIP_KEYPOINTS = (9, 10)


def _open_mhr(scene: str, human_dir: Path, n: int):
    path = human_dir / "mhr_1.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"{scene}: {path} missing — run scripts/data/convert_kindyn_to_mhr.py")
    mhr = np.load(path, allow_pickle=True)
    if int(mhr["num_frames"]) != n:
        raise ValueError(
            f"{scene}: mhr_1 has {int(mhr['num_frames'])} frames, contacts has {n}")
    version = int(mhr["converter_version"])
    if version < MHR_CONVERTER_VERSION or "lbs_params" not in mhr:
        raise ValueError(
            f"{scene}: mhr_1.npz converter_version {version} lacks lbs_params — "
            f"regenerate with scripts/data/convert_kindyn_to_mhr.py "
            f"(v{MHR_CONVERTER_VERSION})")
    return mhr


def _open_sup(scene: str, human_dir: Path, n: int):
    path = human_dir / "mhr_sup_1.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"{scene}: {path} missing — run scripts/data/precompute_mhr_supervision.py")
    sup = np.load(path, allow_pickle=True)
    schema = int(sup["schema_version"])
    if schema != MHR_SUP_SCHEMA:
        raise ValueError(
            f"{scene}: mhr_sup_1 schema {schema} != {MHR_SUP_SCHEMA} — regenerate "
            f"with scripts/data/precompute_mhr_supervision.py")
    if int(sup["num_frames"]) != n:
        raise ValueError(
            f"{scene}: mhr_sup_1 has {int(sup['num_frames'])} frames, contacts has {n}")
    return sup


def _sup_keypoints(scene: str, sup, mhr_ids: np.ndarray, object_ids: np.ndarray,
                   n: int) -> np.ndarray:
    """``mhr_sup_1`` world keypoints in dataset person order. ``(P, N, 70, 3)``.

    ``mhr_sup_1.npz`` stores no ``object_ids`` — its person rows ARE the
    ``mhr_1`` rows by construction — so the person axis is resolved with
    ``mhr_1``'s ids after checking the array shape agrees.
    """
    kp_raw = np.asarray(sup["kp_world"], np.float32)
    if kp_raw.shape[0] != len(mhr_ids):
        raise ValueError(
            f"{scene}: mhr_sup_1 person axis {kp_raw.shape[0]} does not match "
            f"mhr_1's {len(mhr_ids)} object ids")
    kp = rows_by_object_id(kp_raw, mhr_ids, object_ids, scene, "mhr_sup_1")
    if kp.shape != (len(object_ids), n, NUM_MHR70, 3):
        raise ValueError(
            f"{scene}: mhr_sup_1 kp_world {kp.shape} does not match "
            f"({len(object_ids)}, {n}, {NUM_MHR70}, 3)")
    return kp


def _lbs_targets(
    mhr, mhr_ids: np.ndarray, object_ids: np.ndarray, scene: str,
    pose_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bone and scale targets from an open ``mhr_1.npz``.

    * bones — the flexible geometry slots, the person's MEDIAN over valid rows
      served on every frame. The converter re-fits them per frame, but a body's
      proportions do not change within a scene: the frame-to-frame spread
      (per-slot std 0.07-0.14, ~70 % of the between-person spread) is fit
      freedom, not signal.
    * scale — the 68 per-person scale slots, verified exactly constant across
      fitted rows, taken from the person's first valid row; a person with no
      valid row gets zeros and is masked by ``pose_valid`` anyway.

    :param pose_valid: ``(P, N)`` per-person validity, already id-aligned.
    :returns: ``(bones (P, N, 6), scales (P, 68))`` float32.
    """
    lbs = rows_by_object_id(
        np.asarray(mhr["lbs_params"], np.float32), mhr_ids, object_ids,
        scene, "mhr_1")                                              # [P, N, 204]
    if lbs.shape[:2] != pose_valid.shape or lbs.shape[2] != MHR_SCALE_SLOTS.stop:
        raise ValueError(
            f"{scene}: mhr_1 lbs_params {lbs.shape} does not match "
            f"{pose_valid.shape} x {MHR_SCALE_SLOTS.stop}")
    bones = np.zeros(pose_valid.shape + (NUM_MHR_BONES,), np.float32)
    scales = np.zeros((lbs.shape[0], NUM_MHR_SCALES), np.float32)
    for person in range(lbs.shape[0]):
        rows = np.flatnonzero(pose_valid[person])
        if len(rows):
            bones[person] = np.median(lbs[person, rows, MHR_BONE_SLOTS], axis=0)
            scales[person] = lbs[person, rows[0], MHR_SCALE_SLOTS]
    return bones, scales


def load_pose(scene: str, human_dir: Path, object_ids: np.ndarray, n: int) -> dict:
    """MHR pose pseudo-GT: world ``q``, identity, bone geometry and scales.

    Only ``valid_mask`` rows were fitted (the rest carry the raw per-frame init),
    so the loss masks on ``pose_valid``.

    :returns: ``pose_gt_q (P, N, 132)``, ``pose_valid (P, N)``,
        ``pose_identity (P, 45)``, ``pose_gt_bones (P, N, 6)``,
        ``pose_gt_scale (P, 68)``.
    """
    mhr = _open_mhr(scene, human_dir, n)
    mhr_ids = np.asarray(mhr["object_ids"])
    q_world = rows_by_object_id(
        np.asarray(mhr["q_world"], np.float32), mhr_ids, object_ids, scene, "mhr_1")
    pose_valid = rows_by_object_id(
        np.asarray(mhr["valid_mask"], bool), mhr_ids, object_ids, scene, "mhr_1")
    identity = rows_by_object_id(
        np.asarray(mhr["identity"], np.float32), mhr_ids, object_ids, scene, "mhr_1")
    bones, scales = _lbs_targets(mhr, mhr_ids, object_ids, scene, pose_valid)
    return {"pose_gt_q": q_world, "pose_valid": pose_valid,
            "pose_identity": identity, "pose_gt_bones": bones,
            "pose_gt_scale": scales}


def load_keypoints(scene: str, human_dir: Path, object_ids: np.ndarray, n: int) -> dict:
    """MHR-native keypoint + vertex GT in the metric world frame.

    NaN rows (frames the fit did not cover) come back as exact zeros with their
    validity bit False; the losses mask on the bit and lift these world arrays
    into the camera with the frame's ``cam_from_world``.

    :returns: ``kp3d_world (P, N, 70, 3)``, ``kp_valid (P, N)``,
        ``vert_gt_world (P, N, V, 3)``, ``vert_valid (P, N)``,
        ``vert_indices (V,)`` into ``pred_vertices``.
    """
    mhr = _open_mhr(scene, human_dir, n)
    mhr_ids = np.asarray(mhr["object_ids"])
    sup = _open_sup(scene, human_dir, n)
    kp3d = _sup_keypoints(scene, sup, mhr_ids, object_ids, n)
    vert_raw = np.asarray(sup["verts_world"], np.float32)
    if vert_raw.shape[0] != len(mhr_ids):
        raise ValueError(
            f"{scene}: mhr_sup_1 vertex person axis {vert_raw.shape[0]} does not "
            f"match mhr_1's {len(mhr_ids)} object ids")
    verts = rows_by_object_id(vert_raw, mhr_ids, object_ids, scene, "mhr_sup_1")
    vert_indices = np.asarray(sup["vert_indices"], np.int64).reshape(-1)
    if verts.shape != (len(object_ids), n, NUM_SUP_VERTICES, 3) or len(
            vert_indices) != NUM_SUP_VERTICES:
        raise ValueError(
            f"{scene}: mhr_sup_1 verts_world {verts.shape} / vert_indices "
            f"{vert_indices.shape} do not match ({len(object_ids)}, {n}, "
            f"{NUM_SUP_VERTICES}, 3)")
    kp_valid = np.isfinite(kp3d).all(axis=(2, 3))
    vert_valid = np.isfinite(verts).all(axis=(2, 3))
    return {
        "kp3d_world": np.where(kp_valid[:, :, None, None], kp3d, 0.0).astype(np.float32),
        "kp_valid": kp_valid,
        "vert_gt_world": np.where(
            vert_valid[:, :, None, None], verts, 0.0).astype(np.float32),
        "vert_valid": vert_valid,
        "vert_indices": vert_indices,
    }


def motion_root(
    scene: str, human_dir: Path, object_ids: np.ndarray, n: int, fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The free-flyer trajectory the pelvis motion target is differentiated from.

    The pair is (MEAN-HIPS position, ROOT orientation) — deliberately NOT
    ``q_world[..., :7]``. The MHR free-flyer root is anchored ~0.93 m from the
    hips with a leg-pose-dependent 0.28 m spread, so its twist is a different
    physical quantity. The mean-hips pairing is exactly what the prediction side
    lifts to the world (``mean(kp[9, 10]) + pred_cam_t``, orientation from
    ``global_rot``), so GT and prediction are the same construction on the same
    rig and the hip offset is zero by construction.

    Rows the fit did not cover are NaN in the archives; they come back as the
    identity root and are reported invalid. No supervised row ever reads one
    (the stencil needs three consecutive valid frames).

    :returns: ``(root7 (P, N, 7) float32, valid (P, N) bool)``.
    """
    mhr = _open_mhr(scene, human_dir, n)
    sup = _open_sup(scene, human_dir, n)
    src_fps = float(np.asarray(mhr["fps"]).item())
    if not np.isfinite(src_fps) or src_fps <= 0:
        raise ValueError(f"{scene}: bad mhr_1 fps {src_fps}")
    if abs(src_fps - fps) > 1e-6:
        raise ValueError(f"{scene}: mhr_1 fps {src_fps} != contacts fps {fps}")
    mhr_ids = np.asarray(mhr["object_ids"])
    q_world = rows_by_object_id(
        np.asarray(mhr["q_world"], np.float32), mhr_ids, object_ids, scene, "mhr_1")
    valid = rows_by_object_id(
        np.asarray(mhr["valid_mask"], bool), mhr_ids, object_ids, scene, "mhr_1")
    kp = _sup_keypoints(scene, sup, mhr_ids, object_ids, n)
    hips = kp[:, :, HIP_KEYPOINTS].mean(axis=2)                       # [P, N, 3]
    root7 = np.concatenate([hips, q_world[..., 3:7]], axis=-1)        # [P, N, 7]
    valid = (valid & np.isfinite(kp).all(axis=(2, 3))
             & np.isfinite(root7).all(axis=-1))
    identity_root = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], np.float32)
    return np.where(valid[:, :, None], root7, identity_root).astype(np.float32), valid
