"""kindyn ground truth: contact forces, the fitted gravity, and the pelvis twist.

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

**Motion.** The pelvis target is the free-flyer body twist of a trajectory
supplied by the caller (:mod:`data.climbing_videos.mhr_gt` builds it from the
MHR rig), Gaussian-smoothed in SECONDS so the label bandwidth does not depend
on the scene's fps, differentiated with BVR's SE3-log stencil, and its LINEAR
half re-expressed in the gravity-view frame: vertical = the scene's FITTED
gravity, azimuth = the camera view direction. The angular half is the SE3-log
body rate under any linear convention. Layout is
``[lin_vel, lin_acc, ang_vel, ang_acc]``, 12 channels.

Gravity is the per-scene FITTED unit down vector, not a world axis: it tilts
from +y by a median 3.2 deg and up to 61 deg over the corpus, so treating world
y as up is wrong for hundreds of scenes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

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
#: Frames trimmed at each scene edge and on both sides of every validity gap —
#: stricter than the single frame a central difference needs.
MOTION_EDGE_TRIM = 2
#: Small-angle cutoff on ``theta**2``, mirroring ``better_robot.lie.so3``.
_TAYLOR_THETA2_FP64 = 1e-8


# ------------------------------------------------------------------ geometry

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


def _quat_conjugate_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.concatenate([-quat[..., :3], quat[..., 3:]], axis=-1)


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two ``xyzw`` quaternions."""
    ax, ay, az, aw = (a[..., i] for i in range(4))
    bx, by, bz, bw = (b[..., i] for i in range(4))
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def _quat_act_xyzw(quat: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Rotate ``point`` by the ``xyzw`` quaternion."""
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


def so3_log_xyzw(quat: np.ndarray) -> np.ndarray:
    """Rotation vector of an ``xyzw`` unit quaternion. ``(..., 4) -> (..., 3)``.

    float64 mirror of ``better_robot.lie.so3.log`` — same hemisphere flip and
    the same small-angle Taylor branch — so the targets follow the exact scheme
    BetterVideoReconstruction differentiated its fitted trajectory with.
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

    float64 mirror of ``better_robot.lie.se3.log``: the linear part carries the
    ``V^{-1}(omega) = I - W/2 + coeff * W^2`` correction (``W = hat(omega)``),
    so it is a true manifold tangent rather than ``R^T dp``. Layout is
    ``(linear, angular)``.
    """
    omega = so3_log_xyzw(quat)
    theta2 = (omega * omega).sum(axis=-1, keepdims=True)
    taylor = theta2 < _TAYLOR_THETA2_FP64
    theta2_safe = np.where(taylor, 1.0, theta2)
    theta = np.sqrt(theta2_safe)
    cot_half = np.cos(theta / 2.0) / np.maximum(np.sin(theta / 2.0), 1e-30)
    coeff = np.where(taylor, (1.0 / 12.0) + theta2 / 720.0,
                     1.0 / theta2_safe - cot_half / (2.0 * theta))
    skew = _hat3(omega)
    v_inv = np.eye(3) - 0.5 * skew + coeff[..., None] * (skew @ skew)
    linear = np.einsum("...ij,...j->...i", v_inv, np.asarray(trans, np.float64))
    return np.concatenate([linear, omega], axis=-1)


def hemisphere_align(quat: np.ndarray) -> np.ndarray:
    """Remove double-cover sign flips from a quaternion sequence. ``(N, 4)``.

    ``q`` and ``-q`` are the same rotation and the fit is free to switch between
    them frame to frame; component-wise filtering would read such a flip as a
    180-degree excursion.
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
    first, then filtered and renormalized.

    The width is given in SAMPLES so the caller controls the physical bandwidth:
    ``sigma_sec * fps`` makes the label spectrum fps-independent, which is the
    point — raw pelvis ``|a|`` RMS runs 3.4 m/s^2 at 24 fps against 13.3 at 60
    fps for the same activity, i.e. mostly sampling-rate artifact.

    :param valid: ``(N,)`` per-frame validity.
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
    """World-from-gravity-view rotations. ``(3,) | (N, 3)`` + ``(N, 4, 4) -> (N, 3, 3)``.

    GVHMR's Gravity-View frame: the vertical axis is gravity (column 1, DOWN
    positive) and the azimuth is the camera's view direction projected onto the
    horizontal plane. The frame is gravity-aligned, uniquely defined per frame
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
    view = np.asarray(extrinsics[:, 2, :3], np.float64)
    fwd = view - (view * down).sum(-1, keepdims=True) * down
    # Degenerate only when the camera looks along gravity: fall back to the world
    # axis least parallel to it so the basis stays defined and deterministic.
    fallback = np.eye(3)[np.argmin(np.abs(down), axis=-1)]
    fallback = fallback - (fallback * down).sum(-1, keepdims=True) * down
    fwd = np.where(np.linalg.norm(fwd, axis=-1, keepdims=True) > 1e-6, fwd, fallback)
    fwd /= np.linalg.norm(fwd, axis=-1, keepdims=True)
    return np.stack([fwd, down, np.cross(fwd, down)], axis=-1)


def root_body_twist(
    q_root: np.ndarray, dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """BVR's body-twist velocity/acceleration of the free-flyer root.

    Mirrors ``BetterVideoReconstruction/tools/smplx_robot/dynamics.py::
    velocity_acceleration_from_trajectory`` — the ONE place BVR derives v/a from
    a fitted ``q`` — for the free-flyer joint::

        d[t] = se3_log(T_t^-1 T_t+1)
        v[t] = (d[t-1] + d[t]) / (2 dt)
        a[t] = (d[t] - d[t-1]) / dt^2

    The result lives in the ROOT-LOCAL (body) frame as a proper twist: its
    linear acceleration equals ``R^T p_ddot - omega x v_body``, ~7 % away from
    the plain ``R^T`` re-expression of the world acceleration.

    :param q_root: ``(..., N, 7)`` world position ``[0:3]`` + world-from-root
        ``xyzw`` quaternion ``[3:7]``.
    :param dt: frame interval in seconds.
    :returns: ``(vel, acc, omega, ang_acc)``, each ``(..., N, 3)`` float64.
        Boundary frames are zero (they are never target-valid).
    """
    q_root = np.asarray(q_root, np.float64)
    pos, quat = q_root[..., :3], q_root[..., 3:7]
    quat = quat / np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8, None)
    quat_inv = _quat_conjugate_xyzw(quat[..., :-1, :])
    rel_trans = _quat_act_xyzw(quat_inv, pos[..., 1:, :] - pos[..., :-1, :])
    rel_quat = _quat_mul_xyzw(quat_inv, quat[..., 1:, :])
    diff = se3_log_xyzw(rel_trans, rel_quat)                       # (..., N-1, 6)

    twist = np.zeros(q_root.shape[:-1] + (6,), np.float64)
    acc = np.zeros_like(twist)
    twist[..., 1:-1, :] = 0.5 * (diff[..., :-1, :] + diff[..., 1:, :]) / dt
    acc[..., 1:-1, :] = (diff[..., 1:, :] - diff[..., :-1, :]) / (dt * dt)
    return twist[..., :3], acc[..., :3], twist[..., 3:], acc[..., 3:]


# ------------------------------------------------------------------ gravity

def _unit_gravity(scene: str, raw: np.ndarray) -> np.ndarray:
    gravity = np.asarray(raw, np.float64).reshape(3)
    norm = float(np.linalg.norm(gravity))
    if not np.isfinite(gravity).all() or not 0.9 < norm < 1.1:
        raise ValueError(
            f"{scene}: kindyn gravity_world {gravity.tolist()} is not a unit vector")
    return gravity / norm


def fitted_gravity_world(scene: str, human_dir: Path) -> np.ndarray:
    """The scene's FITTED unit down direction, from ``kindyn_1.npz``.

    A wrong gravity rotates every gravity-view target in the scene by a constant
    the network cannot possibly infer, so the fitted vector is mandatory.
    """
    path = human_dir / "kindyn_1.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"{scene}: {path} missing — the gravity-view frame needs kindyn's gravity")
    return _unit_gravity(scene, np.load(path, allow_pickle=True)["gravity_world"])


# ------------------------------------------------------------------ forces

def load_forces(scene: str, human_dir: Path, object_ids: np.ndarray, n: int) -> dict:
    """Six-group GT contact forces in body-weight units, body-root frame.

    :returns: ``force_gt (P, N, 6, 3)``, ``force_contact (P, N, 6)`` bool,
        ``force_lever (P, N, 6, 3)`` metres, ``force_valid (P, N)``,
        ``force_conf (P, N)`` and the scene's fitted ``gravity_world (3,)``.
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
        "gravity_world": _unit_gravity(
            scene, kindyn["gravity_world"]).astype(np.float32),
    }


# ------------------------------------------------------------------ motion

def motion_targets(
    root7: np.ndarray, src_valid: np.ndarray, fps: float, gravity: np.ndarray,
    extrinsics: np.ndarray, smooth_sec: float, outlier_acc_ms2: float,
) -> dict:
    """Pelvis gravity-view twist targets from a free-flyer trajectory.

    The whole scene is differentiated at once in float64 (never per clip).
    Everything derived from the root — the twist, ``R`` and ``omega`` — comes
    from the SMOOTHED trajectory, so the target, the frame it is expressed in
    and the world conversion all describe the same motion. The linear half is
    converted world-first (``a_world = R (a_body + omega x v_body)``, the
    Coriolis relation) and then into the gravity-view frame.

    :param root7: ``(P, N, 7)`` world position + world-from-root ``xyzw`` quat.
    :param src_valid: ``(P, N)`` coverage of the source fit.
    :param gravity: ``(3,)`` fitted unit down vector.
    :param extrinsics: ``(N, 4, 4)`` cam-from-world (the gravity-view azimuth).
    :param smooth_sec: Gaussian width in seconds applied before differentiating.
    :param outlier_acc_ms2: world ``|a|`` above which a row is flagged an
        outlier (a train-only filter bit); ``0`` disables the flag.
    :returns: ``motion_gt (P, N, 1, 12)``, ``motion_valid (P, N)``,
        ``motion_outlier (P, N, 1)``, ``motion_rot``/``motion_lin_rot``
        ``(P, N, 3, 3)``, ``motion_omega (P, N, 3)``,
        ``motion_root_pos (P, N, 3)``, ``motion_root_valid (P, N)``.
    """
    n_people, n = src_valid.shape
    dt = 1.0 / fps
    q_root = np.stack([
        smooth_root_trajectory(
            root7[person].astype(np.float64), src_valid[person], smooth_sec * fps)
        for person in range(n_people)])                                # [P, N, 7]

    rot = quat_xyzw_to_matrix(q_root[..., 3:7])                        # world-from-root
    twist_vel, twist_acc, omega, ang_acc = root_body_twist(q_root, dt)
    world_vel = np.einsum("pnij,pnj->pni", rot, twist_vel)
    world_acc = np.einsum("pnij,pnj->pni", rot, twist_acc + np.cross(omega, twist_vel))
    lin_rot = np.broadcast_to(
        gravity_view_basis(gravity, extrinsics), (n_people, n, 3, 3))
    vel_out = np.einsum("pnji,pnj->pni", lin_rot, world_vel)
    acc_out = np.einsum("pnji,pnj->pni", lin_rot, world_acc)

    # Validity: central-difference support (n-1, n, n+1 inside the scene AND
    # source-valid), then MOTION_EDGE_TRIM frames trimmed at each scene edge and
    # on both sides of every validity gap.
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

    # Outlier bit on the WORLD acceleration magnitude (the fit's 1/dt^2 jitter
    # on 50/60-fps scenes). A threshold of 0 means OFF — without that sentinel
    # every entry would compare ``> 0`` true and mask the whole train loss.
    outlier = np.zeros((n_people, n, 1), bool)
    if outlier_acc_ms2 > 0.0:
        outlier[..., 0] = np.linalg.norm(world_acc, axis=-1) > outlier_acc_ms2
    motion_gt = np.concatenate(
        [vel_out, acc_out, omega, ang_acc], axis=-1)[:, :, None]       # [P, N, 1, 12]
    return {
        "motion_gt": motion_gt.astype(np.float32),
        "motion_valid": target_valid,
        "motion_outlier": outlier,
        "motion_rot": rot.astype(np.float32),
        "motion_lin_rot": np.ascontiguousarray(lin_rot, np.float32),
        "motion_omega": omega.astype(np.float32),
        "motion_root_pos": q_root[..., :3].astype(np.float32),
        "motion_root_valid": src_valid.copy(),
    }


# ------------------------------------------------------------------ SMPL-X

#: Body joints of the corpus SMPL-X (``smplx_mid``): pelvis + 21 articulated.
NUM_SMPLX_BODY_JOINTS = 22
NUM_SMPLX_BETAS = 10
#: ``q`` layout: ``[pelvis_world (3), root quat xyzw (4), 51 x joint quat xyzw]``.
SMPLX_Q_DIM = 211
_SMPLX_BODY_Q = slice(7, 7 + 4 * (NUM_SMPLX_BODY_JOINTS - 1))


def load_smplx(scene: str, human_dir: Path, object_ids: np.ndarray, n: int) -> dict:
    """SMPL-X body GT from ``kindyn_1.npz`` (BetterHuman ``q`` convention).

    The root of ``q`` IS the pelvis pose: ``q[:3]`` equals ``joints_world[0]``
    and ``q[3:7]`` is the world-from-root quaternion (the stored classic
    ``transl`` differs from it only by the shape-dependent pelvis offset
    ``J0(beta)``). Joint rotations are parent-local; hands, face and expression
    are dropped (hands never move a body joint; face/expression are zero
    corpus-wide). Invalid rows are zeroed / set to the identity so nothing
    downstream ever multiplies a NaN by a zero mask.

    :returns: ``smplx_joints_world (P, N, 22, 3)`` metres,
        ``smplx_root_rot (P, N, 3, 3)`` world-from-root,
        ``smplx_body_rot (P, N, 21, 3, 3)`` parent-local joints 1..21,
        ``smplx_betas (P, 10)`` per person, ``smplx_valid (P, N)`` bool.
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
    if joints.shape[:2] != (n_people, n) or joints.shape[2] < NUM_SMPLX_BODY_JOINTS:
        raise ValueError(f"{scene}: kindyn joints_world {joints.shape} is not (P, N, >=22, 3)")
    if betas.shape != (n_people, NUM_SMPLX_BETAS) or not np.isfinite(betas).all():
        raise ValueError(f"{scene}: kindyn betas {betas.shape} are not (P, 10) finite")
    joints = joints[:, :, :NUM_SMPLX_BODY_JOINTS]
    valid = valid & np.isfinite(q).all(axis=-1) & np.isfinite(joints).all(axis=(2, 3))

    q = np.where(valid[..., None], q, 0.0).astype(np.float32)
    root_rot = quat_xyzw_to_matrix(q[..., 3:7])                             # [P, N, 3, 3]
    body_rot = quat_xyzw_to_matrix(
        q[..., _SMPLX_BODY_Q].reshape(n_people, n, NUM_SMPLX_BODY_JOINTS - 1, 4))
    eye = np.eye(3, dtype=np.float32)
    root_rot = np.where(valid[..., None, None], root_rot, eye)
    body_rot = np.where(valid[..., None, None, None], body_rot, eye)
    joints = np.where(valid[..., None, None], joints, 0.0)
    return {
        "smplx_joints_world": joints.astype(np.float32),
        "smplx_root_rot": root_rot.astype(np.float32),
        "smplx_body_rot": body_rot.astype(np.float32),
        "smplx_betas": betas,
        "smplx_valid": valid,
    }
