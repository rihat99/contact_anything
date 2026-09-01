"""ClimbingVideos corpus — per-joint contact (+ GT force) clips read from ``features/``.

Training-time replacement for the exported ClimbingVideos_v1 loader
(``ClimbingVideosDataset``, retired to ``legacy/climbing_videos.py``): reads the
raw pipeline corpus at ``/data3/rikhat.akizhanov/better/data/ClimbingVideos``
directly — ``scenes/scenes.db`` (scene selection + train/test split),
``features/human_optim/<shard>/<scene>/contacts_<level>.npz`` (52-joint labels,
folded to the 22 SMPL-X body joints exactly as the v1 exporter did),
``features/sam3`` (bboxes + person masks), ``features/geometry/transform.npz``
(per-frame original-resolution intrinsics + metric ``cam_from_world``
extrinsics), ``features/annotation`` (manual tri-state test labels) and
optionally ``features/human_optim/<shard>/<scene>/kindyn_1.npz`` (solved GT
contact forces for the six extremity groups). Frames come from the
pre-extracted ``frames/<shard>/<scene>/<pos:06d>.jpg`` tree
(``scripts/extract_corpus_frames.py``, sequential-decode JPEG q95 — row ``k``
of every feature array is frame ``k``).

An **item** is one ``(scene, person, window)`` of ``T`` frames at a given
stride, exactly as in the v1 loader: windows tile each scene with step
``T * stride``, the train split jitters the window start *statelessly* from
``(seed, epoch, item_index)`` via :meth:`set_epoch`, and val/test use the fixed
tiles (plus a terminal window covering the tail). Windows containing an
invalid frame are skipped; a tracked frame whose bbox is degenerate is demoted
to invalid (as in :mod:`contact.data.reconstruction_scenes`).

``__getitem__`` returns a **clip** (list of ``T`` per-frame dicts) matching the
v1 item contract, with two deliberate differences: ``gravity_world`` is the
kindyn convention's world-y-down direction — kindyn's FITTED per-scene vector
when forces load, the exact ``[0, 1, 0]`` fallback otherwise — rather than the
v1 camera-0-derived direction, and ``load_forces=True`` adds per-frame GT forces:

* ``force_gt`` ``[6, 3]`` float32 — solved contact force per group in
  :data:`FORCE_GROUP_NAMES` order, folded from kindyn's ~35 contact frames by
  parent joint (hands sum palm + fingers + thumb into the wrist, feet the
  toe/ball frames, ankles the heels; see :meth:`_load_forces`). Units follow
  ``force_units`` (default ``bw``: ``newtons / (total_mass * 9.81)``); frame
  follows ``force_frame`` (default ``root``: ``f_root = R(q_xyzw)^T @ f_world``
  where ``q[3:7]`` is the kindyn root quaternion in ``xyzw`` order, numerically
  verified against the stored axis-angle ``global_orient``; ``R(q)`` is
  world-from-root).
* ``force_contact`` ``[6]`` bool — kindyn's per-frame solve mask
  (``frame_contact``) folded per group with the same frame -> group map.
  A zero force means *unlabeled*, not measured-zero.
* ``force_lever`` ``[6, 3]`` float32 — lever arm (metres) of each group's
  kindyn joint from the pelvis, in the same ``force_frame`` as ``force_gt``.
  Consumed by the net-torque consistency loss; may be non-finite on frames the
  solve did not cover (the loss skips those rows).
* ``force_valid`` bool — frame valid and covered by the kindyn solve.
* ``force_conf`` float — kindyn's per-frame solve confidence in ``[0, 1]``
  (loss row weight when ``force_supervision.confidence``).

``load_motion=True`` adds the per-frame kindyn motion targets (motion tokens v2):

* ``motion_gt`` ``[K, 6|12]`` float32 — linear velocity (``[..., 0:3]``, m/s) and
  linear acceleration (``[..., 3:6]``, m/s²) of the ``motion_joint_names`` slots
  (default: all of :data:`MOTION_JOINT_NAMES`, pelvis LAST), in **body-root**
  axes. The six limb slots are central differences of ``joints_world`` over the
  FULL scene trajectory at the scene's stored (fractional) fps, rotated with the
  same ``R(q_xyzw)^T`` as the forces. The ``pelvis`` slot follows
  ``motion_root_convention``: ``"twist"`` (default) is BVR's own body twist,
  :func:`root_body_twist`; ``"rotated_world"`` is the same ``R^T`` central
  difference as the limbs (the motion-tokens-v1/v2 convention).
* ``motion_valid`` bool — frame valid, central-difference support present, and
  outside the 2-frame trims at each scene edge / around every validity gap.
* ``motion_outlier`` ``[K]`` bool — per-joint ``|acc|`` above
  ``motion_outlier_acc_ms2`` (kindyn ``1/dt²`` jitter on 50/60-fps scenes; a
  threshold of ``0`` disables the flag). A train-only filter bit; eval never
  applies it.
* ``motion_rot`` ``[3, 3]`` float32 — ``R(q_xyzw)`` (world-from-root) at that
  frame, so the metric can report world-vertical components.
* ``motion_omega`` ``[3]`` float32 — body angular velocity of the root (the
  angular part of the same twist). Under the ``twist`` convention the world
  acceleration is ``R (a + omega x v)``, so the metric needs it to reach world
  axes.
* ``motion_root_pos`` ``[3]`` float32 — the smoothed kindyn root position in
  the metric WORLD frame (``q_root[:3]``, same trajectory ``motion_rot`` comes
  from). With ``motion_rot`` it is the absolute root-pose anchor of the
  consistency loss (``motion_consistency.loss.pos/rot``).
* ``motion_root_valid`` bool — the raw kindyn-coverage bit for that frame.
  Unlike ``motion_valid`` it needs no stencil support or edge trim: an
  absolute pose is per-frame.

``load_keypoints=True`` adds the MHR-native GT the SAM3D-style keypoint and
vertex losses are supervised by, read from ``mhr_sup_1.npz`` (the SAM3D model's
OWN MHR module evaluated at the ``mhr_1`` GT parameters — same rig and same
keypoint regressor as the predictions, so there is no cross-rig bias):

* ``kp3d_world`` ``[70, 3]`` float32 — all MHR70 keypoints, metres in the metric
  WORLD frame (the losses map them into the camera with the frame's
  ``cam_from_world``, which video items always carry). Exactly zero on rows the
  fit did not cover.
* ``kp_valid`` bool — frame valid and the GT row finite.
* ``vert_gt_world`` ``[V, 3]`` float32 — the ``V`` = :data:`NUM_SUP_VERTICES`
  template-subset vertices, same frame and masking rule.
* ``vert_valid`` bool — frame valid and the GT vertex row finite.
* ``vert_indices`` ``[V]`` int64 — the subset's indices into the model's
  ``pred_vertices``; scene-constant (identical corpus-wide).

``load_pose`` / ``load_keypoints`` additionally emit ``mhr_fit_err_cm`` float —
the ``mhr_1`` mesh-fit residual of that row (cm), the optional row-confidence
weight of the MHR-native metric losses.

``cond_features_path`` adds ``cond_feat`` ``[10]`` float32, the *input*-side
conditioning feature (``model.cond_input``): standardized root-frame smoothed
velocity (0:3) and acceleration (3:6), the gravity direction in root axes (6:9)
and a validity bit (9). Unlike everything above it is **label-free** — it is
derived from the frozen model's own reconstructed pelvis trajectory plus the
dataset extrinsics (see :func:`cond_feature_rows`), so it is available at
inference time. Frames outside the artifact (or outside its validity mask) get
exact zeros with bit 0.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset

from ..targets import ALWAYS_NON_CONTACT_8, NUM_BODY_22, OBSERVABLE_14
from .splits import group_train_val_split, video_id_from_scene

DEFAULT_ROOT = "/data3/rikhat.akizhanov/better/data/ClimbingVideos"

# Curated-corpus scene filter (the source of truth for what trains): boulder
# scenes a human kept, VLM-classed as climbing (1) or bouldering (2), not rope
# supported. dataset_split is the DB's train/test assignment.
_SCENE_QUERY = (
    "SELECT scene_id FROM scenes WHERE human_selected=1 AND vlm_category IN (1,2) "
    "AND vlm_rope_supported=0 AND dataset_split=? ORDER BY scene_id"
)

# 52 SMPLXMid joints = 22 body (0-21) + 30 fingers (22-51). Each hand folds the
# wrist + its 15 finger joints (the v1 exporter's convention, kept bit-exact).
N_JOINTS_52 = 52
LEFT_HAND_GROUP_52 = (20,) + tuple(range(22, 37))
RIGHT_HAND_GROUP_52 = (21,) + tuple(range(37, 52))
_HAND_FOLDS = ((20, LEFT_HAND_GROUP_52), (21, RIGHT_HAND_GROUP_52))

# Pinned label schema of contacts_<level>.npz (2026-08-27 corpus regeneration).
CONTACT_LABEL_SCHEMA = 2

#: Canonical six force groups, in kindyn's ``contact_force_joints`` column order.
#: ``*_foot`` is the big-toe joint (SMPL-X 10/11), ``*_ankle`` the heel (7/8).
FORCE_GROUP_NAMES = (
    "left_hand", "right_hand", "left_foot", "right_foot", "left_ankle", "right_ankle",
)
NUM_FORCE_GROUPS = len(FORCE_GROUP_NAMES)
# The joint names kindyn stores for those columns (hands are named by wrist).
KINDYN_FORCE_JOINTS = (
    "left_wrist", "right_wrist", "left_foot", "right_foot", "left_ankle", "right_ankle",
)
# 52-joint membership per force group (hands aggregate wrist + fingers).
FORCE_GROUPS_52 = (LEFT_HAND_GROUP_52, RIGHT_HAND_GROUP_52, (10,), (11,), (7,), (8,))

#: Motion-target joints (motion tokens v2): the six kindyn force joints — hands at
#: the WRIST, exactly where kindyn attaches the wrench — plus the pelvis LAST.
#: Resolved by name from ``kindyn["joint_names"]``; the 52-joint indices are
#: ``(20, 21, 10, 11, 7, 8, 0)``.
MOTION_JOINT_NAMES = KINDYN_FORCE_JOINTS + ("pelvis",)
NUM_MOTION_JOINTS = len(MOTION_JOINT_NAMES)
#: Frames trimmed at each scene edge and on both sides of every validity gap.
#: Stricter than the 1 frame a central difference needs — kept at 2 so the eval
#: rows stay identical to the v1 motion probe's.
MOTION_EDGE_TRIM = 2
#: Default per-(frame, joint) world-acceleration cut (m/s^2) flagged as an
#: outlier; the run config (``motion_supervision.loss.outlier_acc_ms2``) is the
#: authority and is threaded in by :func:`contact.data.collate.make_loaders`.
MOTION_OUTLIER_ACC_MS2 = 50.0
#: Default Gaussian width (SECONDS) applied to the root trajectory before
#: differentiating, so the label bandwidth does not depend on the scene's fps.
#: ``0`` reproduces the raw v1/v2 targets; the config knob
#: ``motion_supervision.target_smooth_sec`` is the authority.
MOTION_TARGET_SMOOTH_SEC = 0.12
#: Reference frame rate (Hz) the ``auto`` clip stride normalises scenes to, so a
#: T-frame clip spans the same physical time at every corpus fps.
MOTION_REFERENCE_FPS = 25.0

#: SMPL-X kindyn joints with a clean MHR70 keypoint correspondence, canonical
#: order shared with contact/keypoint_supervision.py (which owns the MHR70 side).
KP_JOINT_NAMES = (
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "neck",
)

#: MHR70 keypoint count. Since the MHR-native supervision swap the keypoint
#: losses supervise ALL 70 (from ``mhr_sup_1.npz``) rather than the 13
#: name-matched kindyn joints above: the GT is the SAM3D model's own MHR module
#: evaluated at the GT parameters, so the ~12 cm cross-rig bias that
#: ``joints_world`` carried against MHR70 is gone by construction.
NUM_MHR70 = 70
#: Vertex-subset size stored per frame in ``mhr_sup_1.npz`` (farthest-point
#: sampled on the MHR template for full-body coverage).
NUM_SUP_VERTICES = 384
#: Pinned ``mhr_sup_1.npz`` schema (``scripts/precompute_mhr_supervision.py``).
MHR_SUP_SCHEMA = 1
#: Minimum ``mhr_1.npz`` converter version. v3 is the first to store
#: ``lbs_params``, which the bone/scale supervision needs.
MHR_CONVERTER_VERSION = 3
#: ``lbs_params`` slices. Slots 130..135 are the flexible bone-geometry
#: params (spine/neck/shoulder-width/arm/hip-width/leg lengths) — the tail of the
#: pose head's own 136-dim pose block; slots 136..203 are the 68 per-person scale
#: slots the head reaches through ``scale_mean + coeffs @ scale_comps``. Both are
#: the SAME vector as ``out["mhr"]["mhr_model_params"]``, so they compare
#: directly with no unit conversion.
MHR_BONE_SLOTS = slice(130, 136)
MHR_SCALE_SLOTS = slice(136, 204)
NUM_MHR_BONES = 6
NUM_MHR_SCALES = 68
#: MHR70 keypoint index of each non-pelvis :data:`MOTION_JOINT_NAMES` slot —
#: wrists, big-toe tips (kindyn "foot") and heels (kindyn "ankle"), i.e. the same
#: anchors the contact/force tokens use. Read when ``motion_root_source='mhr'``.
MOTION_MHR70_INDICES = (62, 41, 15, 18, 17, 20)
#: MHR70 left/right hip keypoints. Their mean is the body placement the pose
#: predictions are lifted from (``contact.motion_consistency._HIP_KPS``), so it
#: is also the position half of the ``mhr`` motion root.
_MHR70_HIP_KPS = (9, 10)

#: Width of the ``cond_feat`` input-conditioning vector (``model.cond_input``):
#: standardized root-frame velocity (3) + acceleration (3) + the gravity
#: direction in root axes (3) + a validity bit.
COND_FEATURE_DIM = 10

GRAVITY_MAG = 9.81
# Fallback world down direction (world y is down). Since the 2026-08-27 corpus
# regeneration every kindyn solve carries a per-scene FITTED ``gravity_world``
# (tilts of up to ~27 deg from +y occur); force-loading scenes override this
# constant with the fitted vector. The v1 export instead *derived* gravity from
# camera 0 — do not mix the two.
GRAVITY_WORLD = np.array([0.0, 1.0, 0.0], np.float32)

# BetterContactAnnotator's 14 manual joints -> SMPL-X 22-body index (by name).
# Hands land on the wrists (fingers are already folded there in 22-joint space);
# "foot" is the big-toe joint, kept distinct from the ankle.
ANNOTATION_TO_SMPLX22 = {
    "left_hand": 20, "right_hand": 21, "left_foot": 10, "right_foot": 11,
    "left_ankle": 7, "right_ankle": 8, "left_knee": 4, "right_knee": 5,
    "left_elbow": 18, "right_elbow": 19, "left_shoulder": 16, "right_shoulder": 17,
    "left_hip": 1, "right_hip": 2,
}


def scene_shard(scene: str) -> str:
    """Two-level ``<s[0:2]>/<s[2:4]>`` shard prefix used throughout the corpus."""
    return f"{scene[0:2]}/{scene[2:4]}"


def list_corpus_scenes(corpus_root: str | Path, dataset_split: str = "train") -> list[str]:
    """Sorted curated scene ids of one DB ``dataset_split`` (``train``/``test``)."""
    db_path = Path(corpus_root) / "scenes" / "scenes.db"
    if not db_path.is_file():
        raise FileNotFoundError(f"no scene database at {db_path}")
    with sqlite3.connect(db_path) as db:
        return [row[0] for row in db.execute(_SCENE_QUERY, (dataset_split,))]


def list_annotated_test_scenes(corpus_root: str | Path) -> list[str]:
    """Test scenes whose manual ``annotation.npz`` exists (labels are available)."""
    corpus = Path(corpus_root)
    return [
        scene for scene in list_corpus_scenes(corpus, "test")
        if (corpus / "features" / "annotation" / scene_shard(scene) / scene
            / "annotation.npz").is_file()
    ]


def embedding_path(
    embedding_dir: str | Path, scene: str, object_id: int, position: int,
) -> Path:
    """Cache file for one person-frame crop's frozen-backbone embedding.

    ``<embedding_dir>/<shard>/<scene>/<oid:02d>/<pos:06d>.npy`` — an int16 bit
    view of the bf16 ``[1280, 32, 32]`` backbone output (the tensor
    ``forward_pose_branch`` produces right before its fp32 cast). Written by
    ``scripts/precompute_embeddings.py``; read back via
    ``torch.from_numpy(np.load(path)).view(torch.bfloat16)``.
    """
    return (Path(embedding_dir) / scene_shard(scene) / scene
            / f"{object_id:02d}" / f"{position:06d}.npy")


def merge_contacts_52_to_22(
    jc52: np.ndarray, conf52: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold 52-joint contacts + confidence into the 22 SMPL-X body joints.

    Port of the v1 exporter's fold (``export_contact_dataset.py``): joints 0-21
    pass through; each hand ORs the wrist + its 15 fingers. The folded
    confidence is the strongest *touching* member when the hand is in contact
    (max) and the weakest member when it is free (min over all 16 — one
    occluded finger makes the whole-hand free label uncertain).

    A non-finite confidence entry means the joint is **not assessed** by the
    confidence method (since the 2026-08-27 regeneration: spine1/spine3/neck
    and every individual finger joint are NaN on all frames; NaN never
    coincides with ``joint_contact=True``). Unassessed entries are ignored in
    the hand folds and become confidence ``0.0`` for the pass-through joints.

    :param jc52: ``(P, N, 52)`` bool contact labels.
    :param conf52: ``(P, N, 52)`` float32 label confidence in ``[0, 1]``
        (non-finite = not assessed).
    :returns: ``(jc22 (P,N,22) bool, conf22 (P,N,22) float32)``.
    """
    jc52 = np.asarray(jc52, bool)
    conf52 = np.asarray(conf52, np.float32)
    if jc52.ndim != 3 or jc52.shape[-1] != N_JOINTS_52:
        raise ValueError(f"joint_contact must have shape (P,N,52), got {jc52.shape}")
    if conf52.shape != jc52.shape:
        raise ValueError(
            f"label_confidence shape {conf52.shape} does not match contacts {jc52.shape}")
    finite = np.isfinite(conf52)
    if bool(((conf52 < 0.0) | (conf52 > 1.0))[finite].any()):
        raise ValueError("finite label_confidence must be within [0, 1]")

    jc22 = jc52[..., :NUM_BODY_22].copy()
    conf22 = np.where(finite, conf52, 0.0)[..., :NUM_BODY_22].copy()
    for wrist, group in _HAND_FOLDS:
        sub_lbl = jc52[..., group]                                      # (P, N, 16)
        sub_conf = conf52[..., group]
        sub_fin = finite[..., group]
        lbl = sub_lbl.any(axis=-1)
        # strongest touching vote / weakest member of the free AND — over the
        # ASSESSED members only (a NaN finger casts no vote either way).
        conf_touch = np.where(sub_lbl & sub_fin, sub_conf, -np.inf).max(axis=-1)
        conf_free = np.where(sub_fin, sub_conf, np.inf).min(axis=-1)
        folded = np.where(lbl, conf_touch, conf_free)
        jc22[..., wrist] = lbl
        conf22[..., wrist] = np.where(np.isfinite(folded), folded, 0.0)
    return jc22, conf22


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Rotation matrices from ``xyzw`` quaternions (normalized internally).

    The kindyn root quaternion ``q[..., 3:7]`` uses this layout, and
    ``R(q_xyzw)`` equals ``R(global_orient)`` (world-from-root) — verified
    numerically in ``tests/test_climbing_corpus.py``.

    :param quat: ``(..., 4)`` quaternions, ``xyzw`` order.
    :returns: ``(..., 3, 3)`` float32 rotation matrices.
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


def _quat_conjugate_xyzw(quat: np.ndarray) -> np.ndarray:
    """Conjugate (= inverse, for unit quaternions) of an ``xyzw`` quaternion."""
    return np.concatenate([-quat[..., :3], quat[..., 3:]], axis=-1)


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two ``xyzw`` quaternions (``better_robot.lie.so3``)."""
    ax, ay, az, aw = (a[..., i] for i in range(4))
    bx, by, bz, bw = (b[..., i] for i in range(4))
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def _quat_act_xyzw(quat: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Rotate ``point`` by the ``xyzw`` quaternion (``better_robot.lie.so3.act``)."""
    qxyz, qw = quat[..., :3], quat[..., 3:]
    return point + 2.0 * np.cross(qxyz, np.cross(qxyz, point) + qw * point)


def _hat3(vec: np.ndarray) -> np.ndarray:
    """Skew-symmetric ``(..., 3, 3)`` matrix of an ``(..., 3)`` vector."""
    zero = np.zeros(vec.shape[:-1], vec.dtype)
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    return np.stack([
        np.stack([zero, -z, y], axis=-1),
        np.stack([z, zero, -x], axis=-1),
        np.stack([-y, x, zero], axis=-1),
    ], axis=-2)


#: Small-angle cutoff on ``theta**2`` for the float64 log lane, mirroring
#: ``better_robot.lie.so3._TAYLOR_THETA2_FP64``.
_TAYLOR_THETA2_FP64 = 1e-8


def so3_log_xyzw(quat: np.ndarray) -> np.ndarray:
    """Rotation vector of an ``xyzw`` unit quaternion. ``(..., 4) -> (..., 3)``.

    float64 mirror of :func:`better_robot.lie.so3.log` — same hemisphere flip and
    same small-angle Taylor branch — so the motion targets follow the exact
    scheme BetterVideoReconstruction differentiated the kindyn trajectory with.
    """
    quat = np.asarray(quat, np.float64)
    quat = np.where(quat[..., 3:4] < 0.0, -quat, quat)
    qxyz, qw = quat[..., :3], quat[..., 3:4]
    sin_half2 = (qxyz * qxyz).sum(axis=-1, keepdims=True)
    taylor = sin_half2 < _TAYLOR_THETA2_FP64 / 4.0
    sin_half = np.sqrt(np.where(taylor, 1.0, sin_half2))
    theta = 2.0 * np.arctan2(sin_half, np.clip(qw, -1.0, 1.0))
    factor = np.where(taylor, 2.0 + sin_half2 * (2.0 / 3.0),
                      theta / np.maximum(sin_half, 1e-30))
    return factor * qxyz


def se3_log_xyzw(trans: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """``log`` of the SE3 element ``(trans, quat_xyzw)``. ``-> (..., 6)``.

    float64 mirror of :func:`better_robot.lie.se3.log`: the returned linear part
    carries the ``V^{-1}(omega) = I - W/2 + coeff * W^2`` correction (``W =
    hat(omega)``), so it is a true manifold tangent rather than ``R^T dp``.
    Layout is ``(linear, angular)``.
    """
    omega = so3_log_xyzw(quat)
    theta2 = (omega * omega).sum(axis=-1, keepdims=True)
    taylor = theta2 < _TAYLOR_THETA2_FP64
    theta2_safe = np.where(taylor, 1.0, theta2)
    theta = np.sqrt(theta2_safe)
    cot_half = np.cos(theta / 2.0) / np.maximum(np.sin(theta / 2.0), 1e-30)
    coeff = np.where(
        taylor,
        (1.0 / 12.0) + theta2 / 720.0,
        1.0 / theta2_safe - cot_half / (2.0 * theta),
    )
    skew = _hat3(omega)
    v_inv = np.eye(3) - 0.5 * skew + coeff[..., None] * (skew @ skew)
    linear = np.einsum("...ij,...j->...i", v_inv, np.asarray(trans, np.float64))
    return np.concatenate([linear, omega], axis=-1)


def hemisphere_align(quat: np.ndarray) -> np.ndarray:
    """Remove double-cover sign flips from a quaternion sequence. ``(N, 4)``.

    ``q`` and ``-q`` are the same rotation, and the kindyn solve is free to
    switch between them from frame to frame. Component-wise filtering would read
    such a flip as a 180-degree excursion, so the sequence is first made
    hemisphere-consistent (``dot(q_t, q_t+1) >= 0`` everywhere) by propagating a
    cumulative sign.
    """
    dots = (quat[1:] * quat[:-1]).sum(axis=-1)
    sign = np.concatenate([[1.0], np.cumprod(np.where(dots < 0.0, -1.0, 1.0))])
    return quat * sign[:, None]


def smooth_root_trajectory(
    q_root: np.ndarray, valid: np.ndarray, sigma_samples: float,
) -> np.ndarray:
    """Gaussian-smooth a free-flyer trajectory. ``(N, 7) -> (N, 7)``.

    Smoothing runs INSIDE each contiguous run of valid frames — filtering across
    a tracking gap would invent motion — and invalid frames come back unchanged.
    Positions are filtered component-wise; quaternions are hemisphere-aligned
    (:func:`hemisphere_align`) first, then filtered component-wise and
    renormalized.

    The width is given in SAMPLES so the caller controls the physical bandwidth:
    passing ``sigma_sec * fps`` makes the label spectrum fps-independent, which
    is the point — raw kindyn pelvis ``|a|`` RMS runs 3.4 m/s^2 at 24 fps against
    13.3 at 60 fps for the same activity, i.e. mostly sampling-rate artifact.

    :param valid: ``(N,)`` per-frame kindyn validity.
    :param sigma_samples: Gaussian width in frames; ``<= 0`` returns the input.
    """
    if sigma_samples <= 0.0:
        return q_root
    out = np.array(q_root, np.float64, copy=True)
    breaks = np.flatnonzero(np.diff(valid.astype(np.int8)) != 0) + 1
    for run in np.split(np.arange(len(q_root)), breaks):
        if not valid[run[0]] or len(run) < 2:
            continue
        out[run, :3] = gaussian_filter1d(
            out[run, :3], sigma=sigma_samples, axis=0, mode="nearest", truncate=4.0)
        quat = gaussian_filter1d(
            hemisphere_align(out[run, 3:7]), sigma=sigma_samples, axis=0,
            mode="nearest", truncate=4.0)
        out[run, 3:7] = quat / np.clip(
            np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8, None)
    return out


def gravity_view_basis(gravity: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    """World-from-gravity-view rotations, ``(3,) | (N, 3)`` + ``(N, 4, 4) -> (N, 3, 3)``.

    GVHMR's Gravity-View frame: the vertical axis is gravity (column 1, DOWN
    positive — the sign convention the world-vertical diagnostics already use)
    and the azimuth is the camera's view direction projected onto the horizontal
    plane. The frame is therefore gravity-aligned, uniquely defined per frame,
    and independent of both the arbitrary world azimuth and the body's pose — so
    a pelvis roll/pitch error no longer rotates the linear target. Columns are
    ``[forward, down, right]``, right-handed.

    :param gravity: unit DOWN direction in world axes (per scene or per frame).
    :param extrinsics: ``(N, 4, 4)`` cam-from-world; row 2 of the rotation block
        is the camera's +z (forward) axis expressed in world axes.
    """
    n = extrinsics.shape[0]
    down = np.broadcast_to(
        np.asarray(gravity, np.float64).reshape(-1, 3), (n, 3)).copy()
    down /= np.linalg.norm(down, axis=-1, keepdims=True)
    view = np.asarray(extrinsics[:, 2, :3], np.float64)              # camera +z in world
    fwd = view - (view * down).sum(-1, keepdims=True) * down
    # Degenerate only when the camera looks along gravity: fall back to the world
    # axis least parallel to it so the basis stays defined and deterministic.
    fallback = np.eye(3)[np.argmin(np.abs(down), axis=-1)]
    fallback = fallback - (fallback * down).sum(-1, keepdims=True) * down
    fwd = np.where(
        np.linalg.norm(fwd, axis=-1, keepdims=True) > 1e-6, fwd, fallback)
    fwd /= np.linalg.norm(fwd, axis=-1, keepdims=True)
    return np.stack([fwd, down, np.cross(fwd, down)], axis=-1)


def fitted_gravity_world(scene: str, human_dir: Path) -> np.ndarray:
    """The scene's FITTED unit down direction, from ``kindyn_1.npz``.

    The corpus regeneration made ``gravity_world`` a per-scene FIT (tilts of tens
    of degrees), not the old ``[0, 1, 0]`` constant. The gravity-view frame must
    use the fitted vector: a wrong gravity rotates every target in the scene by a
    constant the network cannot possibly infer.
    """
    path = human_dir / "kindyn_1.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"{scene}: {path} missing — the gravity_view frame needs kindyn's gravity")
    gravity = np.asarray(
        np.load(path, allow_pickle=True)["gravity_world"], np.float64).reshape(3)
    norm = float(np.linalg.norm(gravity))
    if not np.isfinite(gravity).all() or not 0.9 < norm < 1.1:
        raise ValueError(
            f"{scene}: kindyn gravity_world {gravity.tolist()} is not a unit vector")
    return gravity / norm


def root_body_twist(q_root: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BVR's body-twist velocity/acceleration of the free-flyer root.

    Mirrors ``BetterVideoReconstruction/tools/smplx_robot/dynamics.py::
    velocity_acceleration_from_trajectory`` — the ONE place BVR derives v/a from
    the fitted ``q`` — for the free-flyer (root) joint::

        d[t] = se3_log(T_t^-1 T_t+1)        # BetterRobot's `difference`
        v[t] = (d[t-1] + d[t]) / (2 dt)
        a[t] = (d[t] - d[t-1]) / dt^2

    The result therefore lives in the ROOT-LOCAL (body) frame as a proper twist:
    its linear acceleration equals ``R^T p_ddot - omega x v_body``, ~7% away from
    the plain ``R^T`` re-expression of the world acceleration.

    :param q_root: ``(..., N, 7)`` kindyn root configuration — world position
        ``[0:3]`` and world-from-root ``xyzw`` quaternion ``[3:7]``.
    :param dt: frame interval in seconds (``1 / fps``).
    :returns: ``(vel, acc, omega, ang_acc)``, each ``(..., N, 3)`` float64 — the
        LINEAR parts of ``v``/``a``, then the ANGULAR parts: body angular
        velocity and its derivative. Boundary frames are zero (they are never
        target-valid).
    """
    q_root = np.asarray(q_root, np.float64)
    pos, quat = q_root[..., :3], q_root[..., 3:7]
    quat = quat / np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8, None)
    quat_inv = _quat_conjugate_xyzw(quat[..., :-1, :])
    rel_trans = _quat_act_xyzw(quat_inv, pos[..., 1:, :] - pos[..., :-1, :])
    rel_quat = _quat_mul_xyzw(quat_inv, quat[..., 1:, :])
    diff = se3_log_xyzw(rel_trans, rel_quat)                     # (..., N-1, 6)

    twist = np.zeros(q_root.shape[:-1] + (6,), np.float64)
    acc = np.zeros_like(twist)
    twist[..., 1:-1, :] = 0.5 * (diff[..., :-1, :] + diff[..., 1:, :]) / dt
    acc[..., 1:-1, :] = (diff[..., 1:, :] - diff[..., :-1, :]) / (dt * dt)
    return twist[..., :3], acc[..., :3], twist[..., 3:], acc[..., 3:]


def cond_feature_rows(
    vel_world: np.ndarray,
    acc_world: np.ndarray,
    rot_world_from_root: np.ndarray,
    valid: np.ndarray,
    standardize: dict,
    clip: float,
) -> np.ndarray:
    """Assemble the 10-d ``cond_feat`` rows from a ``cond_features.npz`` entry.

    Everything is rotated into the PREDICTED root axes (``x_root = R^T x_world``,
    the artifact's own ``root_frame_use`` recipe), velocity and acceleration are
    standardized with the pinned literals and clamped to ``+-clip``, and invalid
    rows are zeroed so an unusable frame is indistinguishable from a missing one.

    :param vel_world: ``(F, 3)`` smoothed world velocity (m/s).
    :param acc_world: ``(F, 3)`` smoothed world acceleration (m/s^2).
    :param rot_world_from_root: ``(F, 3, 3)`` predicted world-from-root rotation.
    :param valid: ``(F,)`` bool validity mask of the artifact.
    :param standardize: ``vel_mean``/``vel_std``/``acc_mean``/``acc_std``, each 3.
    :param clip: clamp on the standardized components.
    :returns: ``(F, 10)`` float32 — standardized v (0:3), a (3:6), the gravity
        direction in root axes (6:9), the validity bit (9).
    """
    rot = np.asarray(rot_world_from_root, np.float64)
    valid = np.asarray(valid, bool)
    vel_root = np.einsum("fji,fj->fi", rot, np.asarray(vel_world, np.float64))
    acc_root = np.einsum("fji,fj->fi", rot, np.asarray(acc_world, np.float64))
    grav_root = np.einsum("fji,j->fi", rot, GRAVITY_WORLD.astype(np.float64))

    vel_z = (vel_root - np.asarray(standardize["vel_mean"], np.float64)) / np.asarray(
        standardize["vel_std"], np.float64)
    acc_z = (acc_root - np.asarray(standardize["acc_mean"], np.float64)) / np.asarray(
        standardize["acc_std"], np.float64)
    clip = float(clip)
    feat = np.concatenate([
        np.clip(vel_z, -clip, clip),
        np.clip(acc_z, -clip, clip),
        grav_root,
        valid[:, None].astype(np.float64),
    ], axis=-1)
    return (feat * valid[:, None]).astype(np.float32)


def fold_force_group_contact(jc52: np.ndarray) -> np.ndarray:
    """Fold 52-joint contact into per-force-group contact.

    :param jc52: ``(P, N, 52)`` bool contact labels (kindyn's ``joint_contact``,
        bit-identical to ``contacts_2.npz``).
    :returns: ``(P, N, 6)`` bool, groups in :data:`FORCE_GROUP_NAMES` order.
    """
    jc52 = np.asarray(jc52, bool)
    return np.stack(
        [jc52[..., list(group)].any(axis=-1) for group in FORCE_GROUPS_52], axis=-1)


def _annotation_to_22(
    ann_contacts: np.ndarray,
    ann_names: list[str],
    ann_oids: list[int],
    ann_ignored: np.ndarray,
    data_oids: list[int],
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a manual tri-state annotation onto the 22 SMPL-X body joints.

    Port of the v1 exporter's ``--fill-test`` mapping: labels map by joint
    *name*, people match by object id, unlabeled (-1) joint-frames and the 8
    joints absent from the manual schema stay unannotated, and ignored people
    are zeroed entirely.

    :param ann_contacts: ``(P_ann, 14, N)`` int8 tri-state
        (-1 unlabeled / 0 free / 1 contact).
    :returns: ``(joint_contact_22 (P,N,22) bool, annotated_22 (P,N,22) bool)``
        in dataset person order.
    """
    ann_contacts = np.asarray(ann_contacts)
    if ann_contacts.shape != (len(ann_oids), len(ann_names), n_frames):
        raise ValueError(
            "manual contact shape does not match object_ids/joint_names/frame count: "
            f"{ann_contacts.shape} vs ({len(ann_oids)}, {len(ann_names)}, {n_frames})")
    if len(set(ann_oids)) != len(ann_oids) or len(set(data_oids)) != len(data_oids):
        raise ValueError("manual and dataset object_ids must each be unique")
    missing_names = sorted(set(ANNOTATION_TO_SMPLX22) - set(ann_names))
    if missing_names:
        raise ValueError(f"manual annotation is missing joints: {missing_names}")
    if not np.isin(ann_contacts, (-1, 0, 1)).all():
        raise ValueError("manual contacts must be tri-state values -1/0/1")
    ann_ignored = np.asarray(ann_ignored, bool)
    if ann_ignored.shape != (len(ann_oids),):
        raise ValueError(
            f"ignored shape {ann_ignored.shape} does not match "
            f"{len(ann_oids)} annotation people")

    jc = np.zeros((len(data_oids), n_frames, NUM_BODY_22), bool)
    annotated = np.zeros((len(data_oids), n_frames, NUM_BODY_22), bool)
    ann_col = {name: i for i, name in enumerate(ann_names)}
    oid_to_row = {int(oid): i for i, oid in enumerate(ann_oids)}
    for person, oid in enumerate(data_oids):
        row = oid_to_row.get(int(oid))
        if row is None or ann_ignored[row]:     # never annotated / annotator ignored
            continue
        for name, sidx in ANNOTATION_TO_SMPLX22.items():
            tri = ann_contacts[row, ann_col[name]]          # (N,) int8
            jc[person, :, sidx] = tri == 1
            annotated[person, :, sidx] = tri != -1
    return jc, annotated


def _rows_by_object_id(
    array: np.ndarray, source_ids: np.ndarray, wanted_ids: np.ndarray,
    scene: str, what: str,
) -> np.ndarray:
    """Select ``array`` person rows so they align with ``wanted_ids`` order."""
    source = [int(x) for x in np.asarray(source_ids).reshape(-1)]
    wanted = [int(x) for x in np.asarray(wanted_ids).reshape(-1)]
    missing = [oid for oid in wanted if oid not in source]
    if missing:
        raise ValueError(
            f"{scene}: object ids {missing} have no {what} row (available: {source})")
    return np.asarray(array)[[source.index(oid) for oid in wanted]]


#: Free-flyer root of an uncovered frame: origin + identity ``xyzw`` quaternion.
#: Keeps the smoothing/twist stencils finite; those rows are masked out anyway.
_IDENTITY_ROOT = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], np.float32)


def _check_source_fps(scene: str, what: str, src_fps: float, fps: float) -> None:
    """Reject a motion source whose frame rate disagrees with the contacts."""
    if not np.isfinite(src_fps) or src_fps <= 0:
        raise ValueError(f"{scene}: bad {what} fps {src_fps}")
    if abs(src_fps - fps) > 1e-6:
        raise ValueError(f"{scene}: {what} fps {src_fps} != contacts fps {fps}")


def _mhr_lbs_targets(
    mhr, mhr_ids: np.ndarray, object_ids: np.ndarray, scene: str,
    pose_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bone / scale / fit-error targets from an open ``mhr_1.npz`` (converter v3).

    ``lbs_params`` is the exact vector the MHR module is called with, so its
    slots line up 1:1 with ``out["mhr"]["mhr_model_params"]``:

    * bones — the flexible geometry slots (:data:`MHR_BONE_SLOTS`), the
      person's MEDIAN over valid rows served on every frame. The converter
      re-fits them per frame, but a body's proportions do not change within a
      scene: the frame-to-frame spread (per-slot std 0.07-0.14, ~70 % of the
      between-person spread) is fit freedom, not signal, so the median is the
      target and the per-frame jitter is not chased;
    * scale — the 68 :data:`MHR_SCALE_SLOTS`, per-person CONSTANT (verified
      exactly constant across every fitted row), taken from the person's first
      valid row; a person with no valid row gets zeros and is masked by
      ``pose_valid`` being False everywhere anyway;
    * ``fit_err_cm`` — the mesh-fit residual per row, the optional
      row-confidence weight of the metric losses. Non-finite entries (the
      NaN-padded invalid rows) become 0.

    :param mhr: an open ``mhr_1.npz``.
    :param pose_valid: ``(P, N)`` per-person validity, already id-aligned.
    """
    version = int(mhr["converter_version"])
    if version < MHR_CONVERTER_VERSION or "lbs_params" not in mhr:
        raise ValueError(
            f"{scene}: mhr_1.npz converter_version {version} lacks lbs_params — "
            f"regenerate with scripts/convert_kindyn_to_mhr.py (v{MHR_CONVERTER_VERSION})")
    lbs = _rows_by_object_id(
        np.asarray(mhr["lbs_params"], np.float32), mhr_ids, object_ids,
        scene, "mhr_1")                                          # [P, N, 204]
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
    fit_err = np.asarray(
        _rows_by_object_id(np.asarray(mhr["fit_err_cm"], np.float32), mhr_ids,
                           object_ids, scene, "mhr_1"), np.float32)
    fit_err = np.where(np.isfinite(fit_err), fit_err, 0.0).astype(np.float32)
    return bones.astype(np.float32), scales, fit_err


def _longest_valid_run(valid: np.ndarray) -> tuple[int, int]:
    """Start and length of the longest True run in a 1-D bool mask (0 if none)."""
    best_start = best_len = 0
    start = None
    for i, v in enumerate(valid.tolist() + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best_len:
                best_start, best_len = start, i - start
            start = None
    return best_start, best_len


class ClimbingCorpusDataset(Dataset):
    """Windowed per-joint contact (+ force) clips straight from the corpus.

    Duck-types the retired v1 ``ClimbingVideosDataset`` (``legacy/
    climbing_videos.py``: ``supervised_targets``/``topology``/``name``/
    ``set_epoch``, list-of-frame-dict clips, ``_scenes``/``_items`` internals),
    so the existing collate and sliding-window machinery run unchanged.

    :param corpus_root: corpus root containing ``scenes/``, ``features/``, ``frames/``.
    :param scenes: explicit scene ids; ``None`` discovers them from the DB and,
        for ``train``/``val``, applies the grouped source-video split.
    :param split: ``"train"`` (jittered windows) | ``"val"`` (held-out source
        videos, fixed tiles) | ``"test"`` (DB test split, manual labels, fixed tiles).
    :param frames_per_clip: window length ``T``.
    :param frame_stride: source-frame stride within a window, or ``"auto"`` for
        ``max(1, round(fps / 25))`` **per scene** — a clip then spans the same
        physical time everywhere (0.20-0.26 s at T=7) instead of 2.9x more of it
        at 24 fps than at 60. See :meth:`scene_stride`.
    :param jitter: stateless train-window jitter (train split only).
    :param seed: seed for both the grouped split and the window jitter.
    :param val_fraction: source-video fraction held out for ``val``.
    :param contact_level: label readout level — ``contacts_1.npz`` or ``contacts_2.npz``.
    :param use_confidence_weights: multiply ``joint_mask`` by label confidence.
    :param require_labels: on ``test``, require the manual annotation (raise when
        missing; discovery skips unannotated scenes). Train labels are always loaded.
    :param load_forces: read ``kindyn_1.npz`` and emit ``force_gt`` /
        ``force_contact`` / ``force_lever`` / ``force_valid`` / ``force_conf``
        per frame; the scene's ``gravity_world`` becomes kindyn's fitted vector.
    :param force_frame: coordinate frame of ``force_gt``/``force_lever`` —
        ``"root"`` (body-root, the default; no extrinsics anywhere) or
        ``"world"`` (kindyn's metric world, whose yaw is NOT observable from a
        single crop — pair it with a loss that maps predictions into world).
    :param force_units: ``"bw"`` (force divided by ``total_mass * g``,
        dimensionless, the default) or ``"newtons"`` (raw solve output).
    :param load_motion: read ``kindyn_1.npz`` and emit ``motion_gt`` /
        ``motion_valid`` / ``motion_outlier`` / ``motion_rot`` / ``motion_omega``
        per frame.
    :param motion_joint_names: ordered subset of :data:`MOTION_JOINT_NAMES` to
        emit (``None`` = all seven). The slot order IS this tuple's order.
    :param motion_root_convention: how the ``pelvis`` slot is expressed —
        ``"twist"`` (BVR's body twist, see :func:`root_body_twist`) or
        ``"rotated_world"`` (``R^T`` of the world central difference, the
        motion-tokens-v1/v2 convention). The six limb slots are always
        ``rotated_world``: BVR defines no linear twist for non-root joints.
    :param motion_target_smooth_sec: Gaussian width in SECONDS applied to the
        root trajectory before differentiating (:func:`smooth_root_trajectory`),
        so the label bandwidth is fps-independent. ``0`` differentiates the raw
        kindyn fit (the v1/v2 target). Note this also re-derives ``motion_rot``
        and ``motion_omega`` from the smoothed trajectory, so the limb slots'
        ``R^T`` uses the smoothed rotation too.
    :param motion_outlier_acc_ms2: acceleration magnitude above which a
        ``(frame, joint)`` motion target is flagged as an outlier (train-only
        filtering; the loss owns whether the bit is applied). ``0`` disables the
        flag entirely.
    :param motion_angular: append the root slot's body angular velocity and
        acceleration to ``motion_gt`` (``[K, 12]``). Requires the ``twist``
        convention and a pelvis-only joint list.
    :param motion_root_source: rig the motion targets are differentiated from —
        ``"kindyn"`` (SMPL-X ``joints_world`` + kindyn root) or ``"mhr"``
        (``mhr_sup_1`` MHR70 limbs + the ``mhr_1`` ``q_world`` root). The two
        have different target distributions; ``motion_supervision.standardize``
        is source-specific.
    :param load_keypoints: read ``mhr_sup_1.npz`` and emit ``kp3d_world`` /
        ``kp_valid`` / ``vert_gt_world`` / ``vert_valid`` / ``vert_indices`` per
        frame — the MHR70 world keypoints and vertex subset the
        keypoint losses reproject.
    :param cond_features_path: ``cond_features.npz`` to read the input-side
        conditioning feature from (``None`` = no ``cond_feat`` emitted). Entries
        are keyed ``"<scene>__p<object_id>"`` and joined per frame on ``frame_idx``.
    :param cond_standardize: ``vel_mean``/``vel_std``/``acc_mean``/``acc_std``
        literals (each 3), required with ``cond_features_path``.
    :param cond_clip: clamp on the standardized conditioning components.
    :param load_images: read frame JPEGs + person masks in ``__getitem__``.
        ``False`` returns ``image``/``mask`` as ``None`` (metadata-only access;
        such items cannot go through the training collate).
    :param embedding_dir: precomputed-embedding root (``features/embedding``,
        see :func:`embedding_path`). When set, every frame additionally carries
        ``embedding`` — the bf16 ``[1280, 32, 32]`` frozen-backbone output — and
        the model skips the backbone. A missing file raises (a stale or
        incomplete cache must never silently fall back to live compute).
        Frame JPEGs are then NOT pixel-decoded: the model provably never reads
        ``batch["img"]`` values in the cached path (metadata only), so
        ``image`` stays ``None``, ``img_wh`` carries the header-read full-frame
        size, and the collate emits a zero crop (masks still decode — mask
        conditioning runs live).
    """

    supervised_targets = frozenset({"joint"})
    topology = None            # joint target is not bound to a vertex topology
    name = "climbing_corpus"

    def __init__(
        self,
        corpus_root: str = DEFAULT_ROOT,
        scenes: Optional[Sequence[str]] = None,
        split: str = "train",
        frames_per_clip: int = 8,
        frame_stride: int | str = 2,
        jitter: bool = True,
        seed: int = 42,
        val_fraction: float = 0.15,
        contact_level: int = 1,
        use_confidence_weights: bool = False,
        require_labels: bool = True,
        load_forces: bool = False,
        force_frame: str = "root",
        force_units: str = "bw",
        load_motion: bool = False,
        motion_joint_names: Optional[Sequence[str]] = None,
        motion_root_convention: str = "twist",
        motion_target_smooth_sec: float = MOTION_TARGET_SMOOTH_SEC,
        motion_outlier_acc_ms2: float = MOTION_OUTLIER_ACC_MS2,
        motion_angular: bool = False,
        motion_root_source: str = "kindyn",
        load_pose: bool = False,
        load_keypoints: bool = False,
        cond_features_path: Optional[str] = None,
        cond_standardize: Optional[dict] = None,
        cond_clip: float = 5.0,
        load_images: bool = True,
        embedding_dir: Optional[str | Path] = None,
        full_scenes: bool = False,
        eval_max_frames: Optional[int] = None,
    ):
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val' or 'test'; got {split!r}")
        if full_scenes and split == "train":
            raise ValueError("full_scenes is an eval protocol; split must be val/test")
        self.full_scenes = bool(full_scenes)
        self.eval_max_frames = None if eval_max_frames is None else int(eval_max_frames)
        if self.eval_max_frames is not None and self.eval_max_frames < 1:
            raise ValueError("eval_max_frames must be a positive int or None")
        if contact_level not in (1, 2):
            raise ValueError(f"contact_level must be 1 or 2; got {contact_level!r}")
        self.corpus_root = Path(corpus_root)
        self.split = split
        self.mode = "train" if split == "train" else "val"
        self.T = int(frames_per_clip)
        if frame_stride == "auto":
            self.stride = "auto"
        elif isinstance(frame_stride, str):
            raise ValueError(f"frame_stride must be an int or 'auto'; got {frame_stride!r}")
        else:
            self.stride = int(frame_stride)
        self.jitter = bool(jitter) and split == "train"
        self.seed = int(seed)
        self.val_fraction = float(val_fraction)
        self.contact_level = int(contact_level)
        self.use_confidence_weights = bool(use_confidence_weights)
        self.require_labels = bool(require_labels)
        self.load_forces = bool(load_forces)
        if force_frame not in ("root", "world"):
            raise ValueError(f"force_frame must be 'root' or 'world'; got {force_frame!r}")
        if force_units not in ("bw", "newtons"):
            raise ValueError(f"force_units must be 'bw' or 'newtons'; got {force_units!r}")
        self.force_frame = str(force_frame)
        self.force_units = str(force_units)
        self.load_motion = bool(load_motion)
        self.motion_joints = tuple(
            MOTION_JOINT_NAMES if motion_joint_names is None else motion_joint_names)
        unknown = [n for n in self.motion_joints if n not in MOTION_JOINT_NAMES]
        if unknown or len(set(self.motion_joints)) != len(self.motion_joints):
            raise ValueError(
                f"motion_joint_names must be a duplicate-free subset of "
                f"{list(MOTION_JOINT_NAMES)}; got {list(self.motion_joints)}")
        if motion_root_convention not in ("twist", "rotated_world", "gravity_view"):
            raise ValueError(
                "motion_root_convention must be 'twist', 'rotated_world' or "
                f"'gravity_view'; got {motion_root_convention!r}")
        self.motion_root_convention = str(motion_root_convention)
        self.motion_target_smooth_sec = float(motion_target_smooth_sec)
        if not np.isfinite(self.motion_target_smooth_sec) or self.motion_target_smooth_sec < 0:
            raise ValueError(
                "motion_target_smooth_sec must be finite and >= 0; got "
                f"{motion_target_smooth_sec!r}")
        self.motion_outlier_acc_ms2 = float(motion_outlier_acc_ms2)
        self.motion_angular = bool(motion_angular)
        if motion_root_source not in ("kindyn", "mhr"):
            raise ValueError(
                f"motion_root_source must be 'kindyn' or 'mhr'; got {motion_root_source!r}")
        self.motion_root_source = str(motion_root_source)
        self.load_pose = bool(load_pose)
        self.load_keypoints = bool(load_keypoints)
        # Angular twist targets exist for the root slot only; mirroring the
        # config validator keeps the class safe for direct construction.
        # The GV frame is applied to the ROOT slot only (the limbs keep root axes
        # under every convention), and the loss de-rotates all slots with one
        # matrix — so a mixed slot list would report the limbs in the wrong frame.
        if self.motion_root_convention == "gravity_view" and self.motion_joints != ("pelvis",):
            raise ValueError(
                "motion_root_convention='gravity_view' is implemented for the "
                "pelvis slot only (the limb slots stay in root axes); got "
                f"motion_joint_names={list(self.motion_joints)}")
        if self.motion_angular and (
                self.motion_root_convention not in ("twist", "gravity_view")
                or self.motion_joints != ("pelvis",)):
            raise ValueError(
                "motion_angular requires motion_root_convention 'twist' or "
                "'gravity_view' (the angular pair is the SE3-log body rate under "
                "both) and motion_joint_names=['pelvis']; got "
                f"{self.motion_root_convention!r}, {list(self.motion_joints)}")
        self.cond_features_path = (
            None if cond_features_path is None else str(cond_features_path))
        if self.cond_features_path is not None:
            missing = [key for key in ("vel_mean", "vel_std", "acc_mean", "acc_std")
                       if not (cond_standardize or {}).get(key)]
            if missing:
                raise ValueError(
                    f"cond_features_path requires cond_standardize entries {missing}")
        self.cond_standardize = dict(cond_standardize or {})
        self.cond_clip = float(cond_clip)
        self.load_images = bool(load_images)
        self.embedding_dir = None if embedding_dir is None else Path(embedding_dir)
        self._epoch = 0

        if scenes is None:
            if split == "test":
                scenes = (
                    list_annotated_test_scenes(self.corpus_root)
                    if require_labels
                    else list_corpus_scenes(self.corpus_root, "test")
                )
            else:
                all_train = list_corpus_scenes(self.corpus_root, "train")
                train_videos, val_videos = group_train_val_split(
                    (video_id_from_scene(s) for s in all_train),
                    self.val_fraction, self.seed)
                keep = train_videos if split == "train" else val_videos
                scenes = [s for s in all_train if video_id_from_scene(s) in keep]

        # Opened once for the whole scene loop (a zip member read per scene) and
        # dropped before the loaders fork their workers.
        self._cond_npz = (
            None if self.cond_features_path is None
            else np.load(self.cond_features_path, allow_pickle=True))
        self._scenes: dict[str, dict] = {}
        # (scene, person, base_start, jitter_range, t_frames); t_frames == self.T
        # for tiled windows, per-item for full_scenes.
        self._items: list[tuple[str, int, int, int, int]] = []
        for scene in scenes:
            data = self._load_scene(scene)
            self._scenes[scene] = data
            stride = self.scene_stride(scene)
            span = (self.T - 1) * stride
            step = self.T * stride
            num_frames = len(data["frame_indices"])
            valid_mask = data["valid_mask"]                 # [P, N] bool
            if self.full_scenes:
                # One clip per (scene, person): the longest contiguous valid
                # run, strided like training. Single-pass whole-scene protocol
                # for the long-sequence (RoPE) temporal module.
                for person in range(valid_mask.shape[0]):
                    base, run_len = _longest_valid_run(valid_mask[person])
                    if run_len < 1:
                        continue
                    t_frames = (run_len - 1) // stride + 1
                    # GPU-memory cap: the frozen per-frame path costs ~0.1 GiB
                    # per frame at inference, so uncapped 500-frame scenes OOM
                    # a 48 GB card. Truncation keeps the run's head.
                    if self.eval_max_frames is not None:
                        t_frames = min(t_frames, self.eval_max_frames)
                    self._items.append((scene, person, base, 1, t_frames))
                continue
            max_start = num_frames - 1 - span
            if max_start < 0:
                continue                                    # scene too short for one window
            for person in range(valid_mask.shape[0]):
                bases = list(range(0, max_start + 1, step))
                # Val/test windows are the scored tiles: when the stride tiling
                # leaves a tail, append a terminal window so those frames are
                # covered too (a few boundary frames may score twice — accepted).
                if self.mode == "val" and bases and bases[-1] != max_start:
                    bases.append(max_start)
                for base in bases:
                    positions = base + np.arange(self.T) * stride
                    if not valid_mask[person, positions].all():
                        continue  # every temporal row needs a real bbox/camera crop
                    jitter_range = max(1, min(step, max_start - base + 1))
                    self._items.append((scene, person, base, jitter_range, self.T))
        if self._cond_npz is not None:
            self._cond_npz.close()
            self._cond_npz = None

    def scene_stride(self, scene: str) -> int:
        """Frame stride used inside this scene's clips.

        A fixed integer stride is returned as-is; ``"auto"`` resolves to
        ``max(1, round(fps / 25))``, which holds the clip's PHYSICAL span roughly
        constant (T=7 spans 0.20-0.26 s at every corpus fps) instead of letting a
        60-fps scene show 2.9x less time than a 24-fps one. Targets stay
        native-rate derivatives — under fixed-seconds label smoothing they are a
        physical quantity, not a per-sample difference, so the wider spacing is a
        context choice, not a semantic mismatch.
        """
        if self.stride != "auto":
            return self.stride
        return max(1, int(round(float(self._scenes[scene]["fps"]) / MOTION_REFERENCE_FPS)))

    # ------------------------------------------------------------------ loading

    def _load_scene(self, scene: str) -> dict:
        features = self.corpus_root / "features"
        shard = scene_shard(scene)
        human_dir = features / "human_optim" / shard / scene
        sam3_dir = features / "sam3" / shard / scene
        contacts = np.load(
            human_dir / f"contacts_{self.contact_level}.npz", allow_pickle=True)
        boxes = np.load(sam3_dir / "bboxes.npz", allow_pickle=True)
        transform = np.load(
            features / "geometry" / shard / scene / "transform.npz", allow_pickle=True)

        n = int(contacts["num_frames"])
        object_ids = np.asarray(contacts["object_ids"], np.int64).reshape(-1)
        n_people = len(object_ids)
        valid_mask = np.asarray(contacts["valid_mask"], bool)             # [P, N]
        intrinsics = np.asarray(transform["intrinsics_px_orig"], np.float32)  # [N, 3, 3]
        extrinsics = np.asarray(transform["extrinsics"], np.float32)      # [N, 4, 4] cam-from-world
        bbox = _rows_by_object_id(
            np.asarray(boxes["bboxes_per_obj"], np.float32),
            boxes["object_ids"], object_ids, scene, "sam3 bbox track")    # [P, N, 4] xyxy px

        if valid_mask.shape != (n_people, n):
            raise ValueError(
                f"{scene}: valid_mask {valid_mask.shape} does not match "
                f"({n_people}, {n})")
        if bbox.shape != (n_people, n, 4):
            raise ValueError(
                f"{scene}: bboxes_per_obj {bbox.shape} does not match "
                f"({n_people}, {n}, 4)")
        if intrinsics.shape != (n, 3, 3) or extrinsics.shape != (n, 4, 4):
            raise ValueError(
                f"{scene}: camera arrays {intrinsics.shape}/{extrinsics.shape} do not "
                f"match {n} frames")
        if not bool(np.asarray(transform["metric"]).item()):
            raise ValueError(
                f"{scene}: geometry is still up-to-scale (metric=False) — the corpus "
                f"scale stage has not run")
        if not np.isfinite(extrinsics).all():
            raise ValueError(f"{scene}: extrinsics contain non-finite values")
        for name, npz in (("contacts", contacts), ("sam3/bboxes", boxes),
                          ("geometry/transform", transform)):
            if "frame_indices" not in npz.files:
                continue
            frame_indices = np.asarray(npz["frame_indices"], np.int64)
            if not np.array_equal(frame_indices, np.arange(n, dtype=np.int64)):
                raise ValueError(
                    f"{scene}: {name}.frame_indices is not sequential 0..{n - 1}; "
                    f"the frames/ tree would be misaligned")

        # A tracked frame whose box is degenerate cannot be cropped — demote it
        # to invalid rather than failing the scene (reconstruction_scenes rule).
        bbox_good = (
            np.isfinite(bbox).all(axis=-1)
            & (bbox[..., 2] > bbox[..., 0])
            & (bbox[..., 3] > bbox[..., 1])
        )
        valid_mask = valid_mask & bbox_good

        joint_contact = contact_conf = annotated = None
        if self.split in ("train", "val"):
            schema = int(np.asarray(contacts["contact_label_schema"]).item())
            if schema != CONTACT_LABEL_SCHEMA:
                raise ValueError(
                    f"{scene}: contacts_{self.contact_level} contact_label_schema="
                    f"{schema}, expected {CONTACT_LABEL_SCHEMA}")
            joint_contact, contact_conf = merge_contacts_52_to_22(
                contacts["joint_contact"], contacts["joint_label_confidence"])
        elif self.require_labels:
            joint_contact, annotated = self._load_test_labels(scene, object_ids, n)

        # World camera centres C = -R^T t for the physics camera-jerk filter;
        # the jump is computed in __getitem__ between consecutive SAMPLED frames.
        cam_centers = -np.einsum(
            "nji,nj->ni", extrinsics[:, :3, :3], extrinsics[:, :3, 3]
        ).astype(np.float32)

        data = {
            "dir": human_dir,
            "frames_dir": self.corpus_root / "frames" / shard / scene,
            "mask_dir": sam3_dir,
            "object_ids": object_ids,
            "frame_indices": np.arange(n, dtype=np.int64),
            "bbox": bbox,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,                            # [N, 4, 4] cam-from-world
            "gravity_world": GRAVITY_WORLD.copy(),               # [3] exact, downward
            "cam_centers": cam_centers,                          # [N, 3] world camera centres (m)
            "valid_mask": valid_mask,
            "fps": float(contacts["fps"]),
            "joint_contact": joint_contact,                  # [P, N, 22] bool or None
            "contact_conf": contact_conf,                    # [P, N, 22] f32 or None
            "annotated": annotated,                          # [P, N, 22] bool or None (test)
        }
        if self.load_forces:
            data.update(self._load_forces(scene, human_dir, object_ids, n))
        if self.load_motion:
            # The motion diagnostics project the world vectors on this vector and
            # the gravity_view frame is built from it, so it must be the FITTED
            # one whether or not `load_forces` (which sets it too) is on. The
            # corpus tilt reaches 61 deg, so the GRAVITY_WORLD constant above
            # would silently mis-report every vertical statistic.
            data["gravity_world"] = fitted_gravity_world(scene, human_dir).astype(
                np.float32)
            data.update(self._load_motion(
                scene, human_dir, object_ids, n, float(contacts["fps"]), extrinsics))
        if self.load_pose:
            data.update(self._load_pose(scene, human_dir, object_ids, n))
        if self.load_keypoints:
            data.update(self._load_keypoints(scene, human_dir, object_ids, n))
        if self.cond_features_path is not None:
            data.update(self._load_cond(scene, object_ids, n))
        return data

    def _load_pose(
        self, scene: str, human_dir: Path, object_ids: np.ndarray, n: int,
    ) -> dict:
        """Kindyn-MHR pseudo-GT pose targets from ``mhr_1.npz`` (E2).

        Written by ``scripts/convert_kindyn_to_mhr.py``: a world-frame MHR ``q``
        trajectory fitted to the kindyn joints. Only ``valid_mask`` rows were
        fitted (the rest carry the raw per-frame init) — the loss masks on it.
        """
        path = human_dir / "mhr_1.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"{scene}: {path} missing — run scripts/convert_kindyn_to_mhr.py")
        mhr = np.load(path, allow_pickle=True)
        if int(mhr["num_frames"]) != n:
            raise ValueError(
                f"{scene}: mhr_1 has {int(mhr['num_frames'])} frames, contacts has {n}")
        mhr_ids = np.asarray(mhr["object_ids"])
        q_world = _rows_by_object_id(
            np.asarray(mhr["q_world"], np.float32), mhr_ids, object_ids,
            scene, "mhr_1")                                     # [P, N, 132]
        pose_valid = _rows_by_object_id(
            np.asarray(mhr["valid_mask"], bool), mhr_ids, object_ids,
            scene, "mhr_1")                                     # [P, N]
        if "identity" not in mhr:
            raise ValueError(
                f"{scene}: mhr_1.npz lacks 'identity' — regenerate with "
                f"scripts/convert_kindyn_to_mhr.py (converter v2)")
        identity = _rows_by_object_id(
            np.asarray(mhr["identity"], np.float32), mhr_ids, object_ids,
            scene, "mhr_1")                                     # [P, 45]
        bones, scales, fit_err = _mhr_lbs_targets(mhr, mhr_ids, object_ids, scene,
                                                  pose_valid)
        return {"pose_gt_q": q_world, "pose_valid_mask": pose_valid,
                "pose_identity": identity, "pose_gt_bones": bones,
                "pose_gt_scale": scales, "mhr_fit_err_cm": fit_err}

    def _load_keypoints(
        self, scene: str, human_dir: Path, object_ids: np.ndarray, n: int,
    ) -> dict:
        """MHR-native keypoint + vertex GT from ``mhr_sup_1.npz``.

        Written by ``scripts/precompute_mhr_supervision.py``: the SAM3D model's
        OWN MHR module evaluated at the ``mhr_1`` GT ``(lbs_params, identity)``,
        so the GT keypoints/vertices come from the same rig and the same
        sapiens-308-sliced-to-70 regressor as the predictions. That kills the
        cross-rig bias the kindyn ``joints_world`` GT carried (the audit measured
        it at 69-75 % of the keypoint MSE); the losses lift these world-frame
        arrays into the camera with the frame's ``cam_from_world``.

        ``mhr_sup_1.npz`` stores no ``object_ids`` — its person rows are the
        ``mhr_1`` rows by construction — so the person axis is resolved with
        ``mhr_1``'s ids, which are checked against the array shape. NaN rows (the
        frames the fit did not cover) come back as exact zeros with the validity
        bit False; the loss masks on the bit.
        """
        mhr_path = human_dir / "mhr_1.npz"
        if not mhr_path.is_file():
            raise FileNotFoundError(
                f"{scene}: {mhr_path} missing — run scripts/convert_kindyn_to_mhr.py")
        mhr = np.load(mhr_path, allow_pickle=True)
        mhr_ids = np.asarray(mhr["object_ids"])
        path = human_dir / "mhr_sup_1.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"{scene}: {path} missing — run scripts/precompute_mhr_supervision.py")
        sup = np.load(path, allow_pickle=True)
        schema = int(sup["schema_version"])
        if schema != MHR_SUP_SCHEMA:
            raise ValueError(
                f"{scene}: mhr_sup_1 schema {schema} != {MHR_SUP_SCHEMA} — "
                f"regenerate with scripts/precompute_mhr_supervision.py")
        if int(sup["num_frames"]) != n:
            raise ValueError(
                f"{scene}: mhr_sup_1 has {int(sup['num_frames'])} frames, contacts has {n}")
        n_people = len(object_ids)
        kp_raw = np.asarray(sup["kp_world"], np.float32)
        vert_raw = np.asarray(sup["verts_world"], np.float32)
        if kp_raw.shape[0] != len(mhr_ids) or vert_raw.shape[0] != len(mhr_ids):
            raise ValueError(
                f"{scene}: mhr_sup_1 person axis {kp_raw.shape[0]}/{vert_raw.shape[0]} "
                f"does not match mhr_1's {len(mhr_ids)} object ids")
        kp3d = _rows_by_object_id(kp_raw, mhr_ids, object_ids, scene, "mhr_sup_1")
        verts = _rows_by_object_id(vert_raw, mhr_ids, object_ids, scene, "mhr_sup_1")
        vert_indices = np.asarray(sup["vert_indices"], np.int64).reshape(-1)
        if kp3d.shape != (n_people, n, NUM_MHR70, 3):
            raise ValueError(
                f"{scene}: mhr_sup_1 kp_world {kp3d.shape} does not match "
                f"({n_people}, {n}, {NUM_MHR70}, 3)")
        if verts.shape != (n_people, n, NUM_SUP_VERTICES, 3) or len(
                vert_indices) != NUM_SUP_VERTICES:
            raise ValueError(
                f"{scene}: mhr_sup_1 verts_world {verts.shape} / vert_indices "
                f"{vert_indices.shape} do not match ({n_people}, {n}, "
                f"{NUM_SUP_VERTICES}, 3)")
        kp_valid = np.isfinite(kp3d).all(axis=(2, 3))                  # [P, N]
        vert_valid = np.isfinite(verts).all(axis=(2, 3))               # [P, N]
        _, _, fit_err = _mhr_lbs_targets(
            mhr, mhr_ids, object_ids, scene,
            _rows_by_object_id(np.asarray(mhr["valid_mask"], bool), mhr_ids,
                               object_ids, scene, "mhr_1"))
        return {
            "kp3d_world": np.where(kp_valid[:, :, None, None], kp3d, 0.0).astype(
                np.float32),                                      # [P, N, 70, 3] world
            "kp_valid": kp_valid,                                 # [P, N] bool
            "vert_gt_world": np.where(
                vert_valid[:, :, None, None], verts, 0.0).astype(
                    np.float32),                                  # [P, N, V, 3] world
            "vert_valid": vert_valid,                             # [P, N] bool
            "vert_indices": vert_indices,                         # [V] into pred_vertices
            "mhr_fit_err_cm": fit_err,                            # [P, N] cm
        }

    def _load_test_labels(
        self, scene: str, object_ids: np.ndarray, n: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Manual tri-state labels mapped to 22 joints (v1 test-label semantics)."""
        path = (self.corpus_root / "features" / "annotation" / scene_shard(scene)
                / scene / "annotation.npz")
        if not path.is_file():
            raise RuntimeError(
                f"{scene}: {path} is missing — manual joint labels are unavailable "
                f"for the test split")
        ann = np.load(path, allow_pickle=True)
        if int(ann["annotation_version"]) < 2:
            raise ValueError(f"{scene}: expected manual annotation schema v2")
        if int(ann["num_frames"]) != n:
            raise ValueError(
                f"{scene}: annotation has {int(ann['num_frames'])} frames, scene has {n}")
        joint_contact, annotated = _annotation_to_22(
            ann["contacts"],
            [str(x) for x in ann["joint_names"]],
            [int(x) for x in ann["object_ids"]],
            ann["ignored"],
            [int(x) for x in object_ids],
            n,
        )
        # On a reviewed frame the 8 joints outside the manual schema are
        # schema-defined non-contact, not unknown (v1 loader normalization).
        # They are structurally False in joint_contact here.
        reviewed = annotated[..., OBSERVABLE_14].any(axis=-1)
        annotated[..., ALWAYS_NON_CONTACT_8] |= reviewed[..., None]
        return joint_contact, annotated

    def _load_cond(
        self, scene: str, object_ids: np.ndarray, n: int,
    ) -> dict:
        """Read the input-conditioning feature for this scene's tracked people.

        The artifact is keyed by ``"<scene>__p<object_id>"`` and stores its own
        ``frame_idx``, which may be shorter than the scene (bbox-degenerate
        frames are skipped) or have internal holes. Rows are scattered back onto
        the scene's frame axis; everything it does not cover stays exactly zero
        (validity bit included), which is also what a wholly missing entry gives.
        """
        cond = np.zeros((len(object_ids), n, COND_FEATURE_DIM), np.float32)
        for person, oid in enumerate(object_ids):
            key = f"{scene}__p{int(oid)}"
            if f"{key}#frame_idx" not in self._cond_npz.files:
                continue
            frame_idx = np.asarray(self._cond_npz[f"{key}#frame_idx"], np.int64)
            rows = cond_feature_rows(
                self._cond_npz[f"{key}#vel_smooth_world"],
                self._cond_npz[f"{key}#acc_smooth_world_alt"],
                self._cond_npz[f"{key}#R_pred_world_from_root"],
                self._cond_npz[f"{key}#feat_valid"],
                self.cond_standardize,
                self.cond_clip,
            )
            keep = (frame_idx >= 0) & (frame_idx < n)
            cond[person, frame_idx[keep]] = rows[keep]
        return {"cond_feat": cond}                           # [P, N, 10] f32

    def _load_forces(
        self, scene: str, human_dir: Path, object_ids: np.ndarray, n: int,
    ) -> dict:
        """Load kindyn GT forces, folding the contact frames into the six groups.

        Since the 2026-08-27 corpus regeneration kindyn stores per-frame forces
        on ~35 named contact frames (``frame_forces``, newtons, world frame).
        Each frame maps to a group through its PARENT JOINT
        (``contact_frame_parents`` -> :data:`FORCE_GROUPS_52` membership): the
        hand groups aggregate the wrist plus every finger frame (palm, fingers,
        thumb sum into the wrist), the foot groups the big-toe/ball/toe frames,
        the ankle groups the heels. Frames whose parent belongs to no group
        (knees, sit, elbows, back, shoulders, chest, head, ...) are dropped —
        corpus-wide they carry ~4 % of the total force magnitude. Group contact
        is the OR of the member frames' ``frame_contact`` (the mask the solve
        placed forces under). Units follow ``force_units`` (``bw`` divides by
        ``total_mass * g``), the coordinate frame follows ``force_frame``
        (``root`` rotates world -> body-root by the kindyn root quaternion).
        Also emits the per-frame solve confidence and the scene's FITTED
        ``gravity_world``.
        """
        kindyn = np.load(human_dir / "kindyn_1.npz", allow_pickle=True)
        kindyn_ids = np.asarray(kindyn["object_ids"])
        frame_names = [str(x) for x in kindyn["contact_frame_names"]]
        parents = np.asarray(kindyn["contact_frame_parents"], np.int64).reshape(-1)
        n_cframes = len(frame_names)
        if parents.shape != (n_cframes,) or (parents < 0).any() or (
                parents >= N_JOINTS_52).any():
            raise ValueError(
                f"{scene}: contact_frame_parents is not {n_cframes} valid "
                f"52-joint indices")
        group_of = np.full(n_cframes, -1, np.int64)
        for g, members in enumerate(FORCE_GROUPS_52):
            group_of[np.isin(parents, list(members))] = g
        for g, name in enumerate(FORCE_GROUP_NAMES):
            if not (group_of == g).any():
                raise ValueError(
                    f"{scene}: no kindyn contact frame maps to force group "
                    f"{name!r} — the corpus schema changed again?")
        frame_forces = _rows_by_object_id(
            np.asarray(kindyn["frame_forces"], np.float32),
            kindyn_ids, object_ids, scene, "kindyn")      # [P, N, F, 3] newtons, world
        frame_contact = _rows_by_object_id(
            np.asarray(kindyn["frame_contact"], bool),
            kindyn_ids, object_ids, scene, "kindyn")      # [P, N, F]
        q = _rows_by_object_id(
            np.asarray(kindyn["q"], np.float32),
            kindyn_ids, object_ids, scene, "kindyn")          # [P, N, 211]
        total_mass = _rows_by_object_id(
            np.asarray(kindyn["total_mass"], np.float32).reshape(-1),
            kindyn_ids, object_ids, scene, "kindyn")          # [P] kg
        force_valid = _rows_by_object_id(
            np.asarray(kindyn["valid_mask"], bool),
            kindyn_ids, object_ids, scene, "kindyn")          # [P, N]
        force_conf = _rows_by_object_id(
            np.asarray(kindyn["force_confidence"], np.float32),
            kindyn_ids, object_ids, scene, "kindyn")          # [P, N]
        joints_world = _rows_by_object_id(
            np.asarray(kindyn["joints_world"], np.float32),
            kindyn_ids, object_ids, scene, "kindyn")          # [P, N, J, 3] m, world
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
            raise ValueError(
                f"{scene}: kindyn joint_names is missing {missing_joints}")
        if not np.isfinite(frame_forces).all():
            raise ValueError(f"{scene}: frame_forces contain non-finite values")
        if not np.isfinite(force_conf).all():
            raise ValueError(f"{scene}: force_confidence contains non-finite values")
        force_conf = np.clip(force_conf, 0.0, 1.0).astype(np.float32)
        if (not np.isfinite(total_mass).all() or (total_mass <= 0).any()
                or not np.isfinite(np.asarray(kindyn["betas"])).all()):
            raise ValueError(f"{scene}: kindyn total_mass/betas are not sane")
        gravity = np.asarray(kindyn["gravity_world"], np.float32).reshape(3)
        grav_norm = float(np.linalg.norm(gravity))
        if not np.isfinite(gravity).all() or not 0.9 < grav_norm < 1.1:
            raise ValueError(
                f"{scene}: kindyn gravity_world {gravity.tolist()} is not a "
                f"unit direction")
        gravity = (gravity / grav_norm).astype(np.float32)

        # Fold frames -> groups: forces sum, contact ORs, over the member frames.
        forces_n = np.stack(
            [frame_forces[:, :, group_of == g].sum(axis=2)
             for g in range(NUM_FORCE_GROUPS)], axis=2)   # [P, N, 6, 3] newtons, world
        group_contact = np.stack(
            [frame_contact[:, :, group_of == g].any(axis=2)
             for g in range(NUM_FORCE_GROUPS)], axis=2)   # [P, N, 6]
        # Forces are only ever solved under the contact mask: a nonzero force on
        # an uncontacted group means corrupted data. (The converse — zero force
        # during contact — is possible in principle, so it is not asserted.)
        nonzero = np.linalg.norm(forces_n, axis=-1) > 0
        if bool((nonzero & ~group_contact).any()):
            raise ValueError(
                f"{scene}: nonzero contact force on a group with no contact label")

        forces_out = forces_n
        if self.force_units == "bw":
            forces_out = forces_n / (total_mass[:, None, None, None] * GRAVITY_MAG)
        # Lever arms for the net-torque consistency loss: the six group joints'
        # world offsets from the pelvis. Resolved by NAME once per scene. Not
        # checked for finiteness — uncovered frames may hold garbage; the loss
        # skips them.
        pelvis = joint_names.index("pelvis")
        group_joints = [joint_names.index(name) for name in KINDYN_FORCE_JOINTS]
        lever = joints_world[:, :, group_joints] - joints_world[:, :, [pelvis]]
        if self.force_frame == "root":
            # q[3:7] is the root quaternion, xyzw, R(q) = world-from-root
            # (verified against the stored axis-angle global_orient); rotate
            # world -> root, forces and lever arms with the same rotation.
            rot = quat_xyzw_to_matrix(q[..., 3:7])            # [P, N, 3, 3]
            forces_out = np.einsum("pnji,pnkj->pnki", rot, forces_out)
            lever = np.einsum("pnji,pnkj->pnki", rot, lever)
        return {
            "force_gt": forces_out.astype(np.float32),   # [P, N, 6, 3] units/frame per cfg
            "force_contact": group_contact,              # [P, N, 6] bool
            "force_lever": lever.astype(np.float32),     # [P, N, 6, 3] m, same frame
            "force_valid": force_valid,                  # [P, N] bool
            "force_conf": force_conf,                    # [P, N] solve confidence [0, 1]
            "gravity_world": gravity,                    # [3] FITTED unit down direction
        }

    def _motion_sources(
        self, scene: str, human_dir: Path, object_ids: np.ndarray, n: int, fps: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Raw trajectories the motion targets are differentiated from.

        Returns ``(joints_world (P, N, 7, 3), root7 (P, N, 7), valid (P, N),
        fps)`` — world-frame positions of :data:`MOTION_JOINT_NAMES` (pelvis
        LAST) and the free-flyer root configuration (position + ``xyzw``
        quaternion) the body twist is taken from.

        ``motion_root_source`` picks the rig:

        * ``kindyn`` — kindyn's SMPL-X ``joints_world`` and its 211-dim ``q``
          root (the v1/v2 targets).
        * ``mhr`` — the MHR-native source: the six limb slots are the
          :data:`MOTION_MHR70_INDICES` columns of ``mhr_sup_1``'s ``kp_world``,
          and the root is the (MEAN-HIPS position, ``q_world`` root quaternion)
          pair — the same free-flyer ``motion_consistency`` builds from the
          PREDICTION, so the two twists are the same construction on the same
          rig and ``hip_offset_root`` is exactly zero. It is deliberately NOT
          ``q_world[..., :7]``: the MHR free-flyer root sits ~0.93 m from the
          hips with a leg-pose-dependent 0.28 m spread, so its twist is a
          different physical quantity. Rows the fit did not cover are
          NaN in the archives; they are zeroed (root: identity) and reported
          invalid, and no supervised row ever reads one (the central-difference
          stencil needs three consecutive valid frames).

        NOTE the two sources have DIFFERENT target distributions — the MHR root
        is not the kindyn pelvis — so ``motion_supervision.standardize`` is
        source-specific and must be recomputed when this flag changes.
        """
        if self.motion_root_source == "kindyn":
            kindyn = np.load(human_dir / "kindyn_1.npz", allow_pickle=True)
            kindyn_ids = np.asarray(kindyn["object_ids"])
            joint_names = [str(x) for x in kindyn["joint_names"]]
            missing = [name for name in MOTION_JOINT_NAMES if name not in joint_names]
            if missing:
                raise ValueError(f"{scene}: kindyn joint_names is missing {missing}")
            if int(kindyn["num_frames"]) != n:
                raise ValueError(
                    f"{scene}: kindyn has {int(kindyn['num_frames'])} frames, contacts "
                    f"has {n} — the derivative frame indexing would be misaligned")
            src_fps = float(np.asarray(kindyn["fps"]).item())
            _check_source_fps(scene, "kindyn", src_fps, fps)
            cols = [joint_names.index(name) for name in MOTION_JOINT_NAMES]
            joints_world = _rows_by_object_id(
                np.asarray(kindyn["joints_world"], np.float32),
                kindyn_ids, object_ids, scene, "kindyn")[:, :, cols]
            root7 = _rows_by_object_id(
                np.asarray(kindyn["q"], np.float32),
                kindyn_ids, object_ids, scene, "kindyn")[..., :7]
            valid = _rows_by_object_id(
                np.asarray(kindyn["valid_mask"], bool),
                kindyn_ids, object_ids, scene, "kindyn")
            return joints_world, root7, valid, src_fps

        mhr_path = human_dir / "mhr_1.npz"
        sup_path = human_dir / "mhr_sup_1.npz"
        for path, maker in ((mhr_path, "scripts/convert_kindyn_to_mhr.py"),
                            (sup_path, "scripts/precompute_mhr_supervision.py")):
            if not path.is_file():
                raise FileNotFoundError(f"{scene}: {path} missing — run {maker}")
        mhr = np.load(mhr_path, allow_pickle=True)
        sup = np.load(sup_path, allow_pickle=True)
        if int(sup["schema_version"]) != MHR_SUP_SCHEMA:
            raise ValueError(
                f"{scene}: mhr_sup_1 schema {int(sup['schema_version'])} != "
                f"{MHR_SUP_SCHEMA} — regenerate with {sup_path.name}'s script")
        for name, archive in (("mhr_1", mhr), ("mhr_sup_1", sup)):
            if int(archive["num_frames"]) != n:
                raise ValueError(
                    f"{scene}: {name} has {int(archive['num_frames'])} frames, "
                    f"contacts has {n} — the derivative frame indexing would be "
                    f"misaligned")
        src_fps = float(np.asarray(mhr["fps"]).item())
        _check_source_fps(scene, "mhr_1", src_fps, fps)
        mhr_ids = np.asarray(mhr["object_ids"])
        q_world = _rows_by_object_id(
            np.asarray(mhr["q_world"], np.float32), mhr_ids, object_ids,
            scene, "mhr_1")                                        # [P, N, 132]
        valid = _rows_by_object_id(
            np.asarray(mhr["valid_mask"], bool), mhr_ids, object_ids,
            scene, "mhr_1")                                        # [P, N]
        kp_raw = np.asarray(sup["kp_world"], np.float32)
        if kp_raw.shape[0] != len(mhr_ids):
            raise ValueError(
                f"{scene}: mhr_sup_1 person axis {kp_raw.shape[0]} does not match "
                f"mhr_1's {len(mhr_ids)} object ids")
        kp = _rows_by_object_id(kp_raw, mhr_ids, object_ids, scene, "mhr_sup_1")
        if kp.shape[1:] != (n, NUM_MHR70, 3):
            raise ValueError(
                f"{scene}: mhr_sup_1 kp_world {kp.shape} does not match "
                f"(P, {n}, {NUM_MHR70}, 3)")
        # The free-flyer trajectory is (MEAN-HIPS position, ROOT orientation) —
        # NOT q_world[..., :7]. The MHR free-flyer root is anchored ~0.93 m from
        # the hips (measured: |root - mean_hips| = 0.933 m, and the residual is
        # leg-pose dependent with a 0.28 m spread, so it is NOT a rigid body
        # offset). Using it directly would make the "pelvis" slot the twist of a
        # foot-level frame — a different physical quantity from the kindyn
        # pelvis, and from what the motion head is anchored to.
        #
        # This pairing is exactly what motion_consistency builds on the
        # PREDICTION side (``p_w = mean(kp[9,10]) + pred_cam_t`` lifted by the
        # extrinsics, ``R_w`` from ``global_rot``), so GT and prediction are the
        # same construction on the same rig and ``hip_offset_root`` is exactly
        # zero for this source.
        hips = kp[:, :, _MHR70_HIP_KPS].mean(axis=2)               # [P, N, 3]
        root7 = np.concatenate([hips, q_world[..., 3:7]], axis=-1)  # [P, N, 7]
        valid = (valid & np.isfinite(kp).all(axis=(2, 3))
                 & np.isfinite(root7).all(axis=-1))
        joints_world = np.concatenate(
            [kp[:, :, list(MOTION_MHR70_INDICES)], hips[:, :, None]], axis=2)
        joints_world = np.where(valid[:, :, None, None], joints_world, 0.0)
        root7 = np.where(valid[:, :, None], root7, _IDENTITY_ROOT)
        return (joints_world.astype(np.float32), root7.astype(np.float32),
                valid, src_fps)

    def _load_motion(
        self, scene: str, human_dir: Path, object_ids: np.ndarray, n: int, fps: float,
        extrinsics: np.ndarray,
    ) -> dict:
        """Linear vel/acc of the motion joints in the configured linear frame.

        Computed once per scene over the FULL trajectory (never per clip) in
        float64. The six limb slots are world central differences rotated with
        the SAME einsum the forces use; the ``pelvis`` slot follows
        ``motion_root_convention`` (see :func:`root_body_twist`): ``twist`` and
        ``rotated_world`` keep body-root axes, ``gravity_view`` expresses the
        root's linear vel/acc in the gravity + camera-view frame instead (see
        :func:`gravity_view_basis`) — the angular pair is the SE3-log body rate
        under every convention. ``motion_lin_rot`` is the world-from-LINEAR-frame
        rotation, which is what turns the linear target back into world axes.
        With ``motion_angular`` the root slot appends the body angular velocity and
        acceleration (``[..., 6:12]``). Only the derived ``[N, K, 6|12]`` arrays
        are cached (~25 MB corpus-wide); the raw 52-joint positions are not.
        """
        n_people = len(object_ids)
        joints_world, root7, src_valid, src_fps = self._motion_sources(
            scene, human_dir, object_ids, n, fps)
        if joints_world.shape != (n_people, n, NUM_MOTION_JOINTS, 3):
            raise ValueError(
                f"{scene}: motion joints_world {joints_world.shape} does not match "
                f"({n_people}, {n}, {NUM_MOTION_JOINTS}, 3)")
        if not np.isfinite(joints_world[src_valid]).all():
            raise ValueError(
                f"{scene}: non-finite motion joint position on a valid frame")

        kindyn_fps = src_fps
        dt = 1.0 / kindyn_fps
        pos = joints_world.astype(np.float64)
        vel = np.zeros_like(pos)
        acc = np.zeros_like(pos)
        vel[:, 1:-1] = (pos[:, 2:] - pos[:, :-2]) / (2.0 * dt)
        acc[:, 1:-1] = (pos[:, 2:] - 2.0 * pos[:, 1:-1] + pos[:, :-2]) / (dt * dt)

        # Fixed-PHYSICAL-width label smoothing (see `smooth_root_trajectory`):
        # everything derived from the root — the twist, R and omega — comes from
        # the smoothed trajectory, so the target, the frame it is expressed in
        # and the world conversion all describe the same motion.
        q_root = np.stack([
            smooth_root_trajectory(
                root7[person].astype(np.float64), src_valid[person],
                self.motion_target_smooth_sec * kindyn_fps)
            for person in range(n_people)])                        # [P, N, 7]

        rot = quat_xyzw_to_matrix(q_root[..., 3:7])                # [P, N, 3, 3] world-from-root
        vel_out = np.einsum("pnji,pnkj->pnki", rot, vel)           # [P, N, 7, 3] root axes
        acc_out = np.einsum("pnji,pnkj->pnki", rot, acc)
        # Outlier magnitudes are the WORLD ones for the R^T slots (a rotation
        # preserves the norm), and the twist's own for the root slot below.
        acc_mag = np.linalg.norm(acc, axis=-1)                     # [P, N, 7]
        # The root slot's BVR-exact body twist. Always computed: `motion_omega`
        # (the body angular velocity) is what turns a root-frame linear vector
        # back into world axes under either convention.
        twist_vel, twist_acc, omega, ang_acc = root_body_twist(q_root, dt)
        pelvis = MOTION_JOINT_NAMES.index("pelvis")
        lin_rot = rot
        if self.motion_root_convention == "twist":
            vel_out[:, :, pelvis] = twist_vel
            acc_out[:, :, pelvis] = twist_acc
            acc_mag[:, :, pelvis] = np.linalg.norm(twist_acc, axis=-1)
        elif self.motion_root_convention == "gravity_view":
            # A pure RE-EXPRESSION of the same body twist, so the target keeps the
            # smoothed trajectory the twist is derived from (the raw central
            # difference of the root would silently drop `target_smooth_sec`, and
            # its acceleration runs ~4x larger). World first — a_world =
            # R (a_body + omega x v_body), the Coriolis relation
            # `contact.motion_supervision.to_world_linear` pins — then into the
            # gravity-view frame.
            world_vel = np.einsum("pnij,pnj->pni", rot, twist_vel)
            world_acc = np.einsum(
                "pnij,pnj->pni", rot, twist_acc + np.cross(omega, twist_vel))
            lin_rot = np.broadcast_to(
                gravity_view_basis(fitted_gravity_world(scene, human_dir), extrinsics),
                (n_people, n, 3, 3))
            vel_out[:, :, pelvis] = np.einsum("pnji,pnj->pni", lin_rot, world_vel)
            acc_out[:, :, pelvis] = np.einsum("pnji,pnj->pni", lin_rot, world_acc)
            acc_mag[:, :, pelvis] = np.linalg.norm(world_acc, axis=-1)

        # Validity: v1's rule verbatim (build_targets.py) — the eval rows depend
        # on it. Central-diff support (n-1, n, n+1 inside the scene AND
        # kindyn-valid), then MOTION_EDGE_TRIM frames trimmed at each scene edge
        # and on both sides of every validity gap.
        target_valid = np.zeros((n_people, n), bool)
        for person in range(n_people):
            valid = src_valid[person]
            diff_ok = np.zeros(n, bool)
            if n >= 3:
                diff_ok[1:n - 1] = valid[2:] & valid[1:-1] & valid[:-2]
            keep = np.zeros(n, bool)
            keep[MOTION_EDGE_TRIM:n - MOTION_EDGE_TRIM] = True
            gaps = np.flatnonzero(~valid)
            for offset in range(-MOTION_EDGE_TRIM, MOTION_EDGE_TRIM + 1):
                neighbours = gaps + offset
                neighbours = neighbours[(neighbours >= 0) & (neighbours < n)]
                keep[neighbours] = False
            target_valid[person] = diff_ok & keep

        # Per-(frame, joint) outlier bit on the WORLD acceleration magnitude
        # (kindyn 1/dt^2 jitter on 50/60-fps scenes). A per-frame drop would
        # discard 14.3% of train frames and hurt the pelvis token most.
        # A threshold of 0 means OFF (the documented sentinel, mirroring
        # ``force_supervision``'s ``outlier_bw``): without this guard every entry
        # would compare ``> 0`` true, mask the whole train loss, and the run would
        # take zero optimiser steps while looking healthy.
        outlier = np.zeros(vel.shape[:3], bool)                       # [P, N, 7]
        if self.motion_outlier_acc_ms2 > 0.0:
            outlier = acc_mag > self.motion_outlier_acc_ms2
        # Keep only the configured slots (default: all seven, in canonical order).
        cols_out = [MOTION_JOINT_NAMES.index(name) for name in self.motion_joints]
        parts = [vel_out, acc_out]
        if self.motion_angular:
            # Angular twist components exist for the root slot only; __init__
            # already restricted angular targets to pelvis-only joint lists.
            ang_vel_out = np.zeros_like(vel_out)
            ang_acc_out = np.zeros_like(acc_out)
            ang_vel_out[:, :, pelvis] = omega
            ang_acc_out[:, :, pelvis] = ang_acc
            parts += [ang_vel_out, ang_acc_out]
        motion_gt = np.concatenate(parts, -1)[:, :, cols_out]
        return {
            "motion_gt": motion_gt.astype(np.float32),              # [P,N,K,6|12] root frame
            "motion_valid": target_valid,                           # [P, N] bool
            "motion_outlier": outlier[:, :, cols_out],              # [P, N, K] bool
            "motion_rot": rot.astype(np.float32),                   # [P,N,3,3] world-from-root
            "motion_lin_rot": np.ascontiguousarray(                  # [P,N,3,3] world-from-
                lin_rot, np.float32),                                # LINEAR frame
            "motion_omega": omega.astype(np.float32),               # [P,N,3] body angular vel
            "motion_root_pos": q_root[..., :3].astype(np.float32),  # [P,N,3] world (smoothed)
            "motion_root_valid": src_valid.copy(),                  # [P, N] bool (no stencil)
        }

    # ------------------------------------------------------------------ epoch / jitter

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used by the stateless train-window jitter."""
        self._epoch = int(epoch)

    def _window_start(self, base: int, jitter_range: int, item_index: int) -> int:
        if not self.jitter:
            return base
        rng = np.random.default_rng([self.seed, self._epoch, item_index])
        return base + int(rng.integers(0, jitter_range))

    # ------------------------------------------------------------------ access

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> list[dict]:
        scene, person, base, jitter_range, t_frames = self._items[index]
        data = self._scenes[scene]
        stride = self.scene_stride(scene)
        start = self._window_start(base, jitter_range, index)
        positions = start + np.arange(t_frames) * stride
        # The indexed base window is all-valid. If jitter crosses a tracking gap,
        # fall back deterministically so invalid-frame bboxes never reach the crop.
        if start != base and not data["valid_mask"][person, positions].all():
            start = base
            positions = base + np.arange(t_frames) * stride

        oid = int(data["object_ids"][person])
        frame_indices = data["frame_indices"]
        fps = data["fps"]
        start_time = float(frame_indices[start])

        clip = []
        for row, pos in enumerate(positions):
            pos = int(pos)
            image = mask = img_wh = None
            if self.load_images:
                frame_path = data["frames_dir"] / f"{pos:06d}.jpg"
                if self.embedding_dir is not None:
                    # Cached-embedding rows never read pixels (the backbone is
                    # skipped); the header open reads only the JPEG dimensions.
                    with Image.open(frame_path) as im:
                        img_wh = im.size                              # (W, H)
                else:
                    image = np.array(Image.open(frame_path).convert("RGB"), np.uint8)
                mask_path = data["mask_dir"] / f"{oid:02d}" / f"frame_{pos:06d}.png"
                mask = np.array(Image.open(mask_path), np.uint8) if mask_path.is_file() else None

            valid = bool(data["valid_mask"][person, pos])
            if data["joint_contact"] is None:              # test with require_labels=False
                joint_gt = np.zeros(NUM_BODY_22, np.float32)
                joint_supervised = np.zeros(NUM_BODY_22, np.float32)
                joint_confidence = np.zeros(NUM_BODY_22, np.float32)
            else:
                joint_gt = data["joint_contact"][person, pos].astype(np.float32)      # [22]
                joint_supervised = np.full(NUM_BODY_22, float(valid), dtype=np.float32)
                if data["annotated"] is not None:          # test: ignore unannotated joints
                    joint_supervised *= data["annotated"][person, pos].astype(np.float32)
                if data["contact_conf"] is None:
                    joint_confidence = np.ones(NUM_BODY_22, np.float32)
                else:
                    joint_confidence = np.clip(
                        data["contact_conf"][person, pos].astype(np.float32), 0.0, 1.0)

            joint_mask = joint_supervised.copy()
            if self.use_confidence_weights:
                joint_mask *= joint_confidence

            frame = {
                "image": image,
                "img_wh": img_wh,
                "mask": mask,
                "bbox": data["bbox"][person, pos],                                    # [4] xyxy
                "cam_int": data["intrinsics"][pos],                                   # [3, 3]
                "cam_from_world": data["extrinsics"][pos],                            # [4, 4]
                "gravity_world": data["gravity_world"],                               # [3]
                # Camera-center displacement (m) from the PREVIOUS SAMPLED frame
                # of this clip (stride-consistent; row 0 = 0.0).
                "cam_jump_m": float(np.linalg.norm(
                    data["cam_centers"][pos]
                    - data["cam_centers"][int(positions[row - 1])]
                )) if row > 0 and valid else 0.0,
                "joint_contact": torch.from_numpy(joint_gt),
                "joint_mask": torch.from_numpy(joint_mask),
                "joint_supervised": torch.from_numpy(joint_supervised),
                "joint_confidence": torch.from_numpy(joint_confidence),
                "frame_pos_sec": (float(frame_indices[pos]) - start_time) / fps,
                "frame_position": pos,
                "frame_index": int(frame_indices[pos]),
                "frame_valid": valid,
                "key": f"{scene}#{oid}@{pos}",
                "dataset": self.name,
            }
            if self.embedding_dir is not None:
                bits = np.load(embedding_path(self.embedding_dir, scene, oid, pos))
                frame["embedding"] = torch.from_numpy(bits).view(torch.bfloat16)
            if self.load_forces:
                frame["force_gt"] = torch.from_numpy(data["force_gt"][person, pos])   # [6, 3]
                frame["force_contact"] = torch.from_numpy(
                    data["force_contact"][person, pos])                               # [6] bool
                frame["force_lever"] = torch.from_numpy(
                    data["force_lever"][person, pos])                                 # [6, 3]
                frame["force_valid"] = valid and bool(data["force_valid"][person, pos])
                frame["force_conf"] = float(data["force_conf"][person, pos])
            if self.load_motion:
                frame["motion_gt"] = torch.from_numpy(
                    data["motion_gt"][person, pos])                               # [K, 6|12]
                frame["motion_outlier"] = torch.from_numpy(
                    data["motion_outlier"][person, pos])                          # [K] bool
                frame["motion_rot"] = torch.from_numpy(
                    data["motion_rot"][person, pos])                              # [3, 3]
                frame["motion_lin_rot"] = torch.from_numpy(
                    data["motion_lin_rot"][person, pos])                          # [3, 3]
                frame["motion_omega"] = torch.from_numpy(
                    data["motion_omega"][person, pos])                            # [3]
                frame["motion_valid"] = valid and bool(data["motion_valid"][person, pos])
                frame["motion_root_pos"] = torch.from_numpy(
                    data["motion_root_pos"][person, pos])                         # [3]
                frame["motion_root_valid"] = valid and bool(
                    data["motion_root_valid"][person, pos])
            if self.load_pose:
                frame["pose_gt_q"] = torch.from_numpy(
                    data["pose_gt_q"][person, pos])                           # [132]
                frame["pose_valid"] = valid and bool(
                    data["pose_valid_mask"][person, pos])
                frame["pose_identity"] = torch.from_numpy(
                    data["pose_identity"][person])                            # [45] static
                frame["pose_gt_bones"] = torch.from_numpy(
                    data["pose_gt_bones"][person, pos])                       # [6]
                frame["pose_gt_scale"] = torch.from_numpy(
                    data["pose_gt_scale"][person])                            # [68] static
            if self.load_keypoints:
                frame["kp3d_world"] = torch.from_numpy(
                    data["kp3d_world"][person, pos])                          # [70, 3]
                frame["kp_valid"] = valid and bool(data["kp_valid"][person, pos])
                frame["vert_gt_world"] = torch.from_numpy(
                    data["vert_gt_world"][person, pos])                       # [V, 3]
                frame["vert_valid"] = valid and bool(data["vert_valid"][person, pos])
                # Scene-constant view (no copy); the collate keeps one row.
                frame["vert_indices"] = torch.from_numpy(data["vert_indices"])  # [V]
            if self.load_pose or self.load_keypoints:
                # Mesh-fit residual of the mhr_1 row — the optional confidence
                # weight of the MHR-native metric losses. Both branches read the
                # same mhr_1 array, so one key serves either.
                frame["mhr_fit_err_cm"] = float(
                    data["mhr_fit_err_cm"][person, pos])
            if self.cond_features_path is not None:
                frame["cond_feat"] = torch.from_numpy(data["cond_feat"][person, pos])  # [10]
            clip.append(frame)
        return clip
