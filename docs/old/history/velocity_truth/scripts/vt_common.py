"""Shared loading for the velocity-matching forensics (read-only, scratchpad only).

Every source is reduced to the SAME per-(scene, person) payload, in the metric
WORLD frame plus the camera-frame pelvis:

    root_pos_w   (N, 3)      pelvis world position
    root_quat_w  (N, 4)      world-from-root quaternion, xyzw
    body_q       (N, 21, 4)  parent-local body joint quaternions 1..21, xyzw
    joints_w     (N, 22, 3)  body-22 world joint positions
    valid        (N,)        bool

GT comes straight from ``kindyn_1.npz`` (the same arrays
``data/climbing_videos/kindyn.py::load_smplx`` feeds the velocity loss).
Predicted runs come from ``scripts/predict_test.py`` dumps (camera-frame
``q_cam``/``joints_cam``) lifted with the scene extrinsics exactly the way
``model/loss/velocity.py`` lifts them.  The frozen SAM3D refit goes through
``viewer/bodies.py::frozen_source`` (classic params -> BetterHuman q -> FK ->
world), whose conventions are the verified ones.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path("/data3/rikhat.akizhanov/better/contact_anything_dev")
CORPUS = Path("/data3/rikhat.akizhanov/better/data/ClimbingVideos")
sys.path.insert(0, str(REPO))

from data.climbing_videos import scene_shard                    # noqa: E402
from data.climbing_videos.camera import smooth_cameras          # noqa: E402

NUM_BODY = 22
BODY_Q = slice(7, 7 + 4 * 21)

#: run label -> (predictions dir, camera_smooth_sec used at train/eval time)
RUNS = {
    "static_baseline": ("output/static_baseline_20260903_191808", 0.0),
    "static_ray": ("output/static_ray_20260904_171141", 0.0),
    "tb_projzero": ("output/tb_projzero_20260904_225421", 0.0),
    "tvel_ray": ("output/tvel_ray_20260905_130637", 0.25),
    "tvel_cliff": ("output/tvel_cliff_20260905_130652", 0.25),
}


def test_scenes() -> list[str]:
    """The 16 static test scenes, from the static_ray dump (all dumps share them)."""
    d = REPO / RUNS["static_ray"][0] / "predictions"
    return sorted(p.stem for p in d.glob("*.npz"))


def rows_by_id(source_ids, wanted: int) -> int | None:
    ids = [int(x) for x in np.asarray(source_ids).reshape(-1)]
    return ids.index(int(wanted)) if int(wanted) in ids else None


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """xyzw quaternion -> rotation matrix. ``(..., 4) -> (..., 3, 3)``."""
    q = np.asarray(q, np.float64)
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    out = np.empty(q.shape[:-1] + (3, 3), np.float64)
    out[..., 0, 0] = 1 - 2 * (y * y + z * z)
    out[..., 0, 1] = 2 * (x * y - z * w)
    out[..., 0, 2] = 2 * (x * z + y * w)
    out[..., 1, 0] = 2 * (x * y + z * w)
    out[..., 1, 1] = 1 - 2 * (x * x + z * z)
    out[..., 1, 2] = 2 * (y * z - x * w)
    out[..., 2, 0] = 2 * (x * z - y * w)
    out[..., 2, 1] = 2 * (y * z + x * w)
    out[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """Rotation matrix -> xyzw quaternion (scipy, exact)."""
    from scipy.spatial.transform import Rotation
    shape = r.shape[:-2]
    q = Rotation.from_matrix(np.asarray(r, np.float64).reshape(-1, 3, 3)).as_quat()
    return q.reshape(shape + (4,))


def scene_cameras(scene: str, smooth_sec: float = 0.0) -> dict:
    """``extrinsics (N,4,4)``, ``intrinsics``, ``fps`` — optionally Gaussian-smoothed."""
    shard = scene_shard(scene)
    tf = np.load(CORPUS / "features" / "geometry" / shard / scene / "transform.npz")
    extr = np.asarray(tf["extrinsics"], np.float32)
    intr = np.asarray(tf["intrinsics_px_orig"], np.float32)
    fps = float(tf["fps"])
    if smooth_sec > 0.0:
        intr, extr = smooth_cameras(intr, extr, smooth_sec * fps)
    return {"extrinsics": np.asarray(extr, np.float64), "intrinsics": np.asarray(intr, np.float64),
            "fps": fps}


def load_gt(scene: str, object_ids: np.ndarray) -> dict:
    """Kindyn GT per person, aligned to ``object_ids`` (dump order)."""
    shard = scene_shard(scene)
    kd = np.load(CORPUS / "features" / "human_optim" / shard / scene / "kindyn_1.npz",
                 allow_pickle=True)
    q_all = np.asarray(kd["q"], np.float64)
    j_all = np.asarray(kd["joints_world"], np.float64)
    v_all = np.asarray(kd["valid_mask"], bool)
    v_all = v_all & np.isfinite(q_all).all(-1) & np.isfinite(j_all).all((2, 3))
    gravity = np.asarray(kd["gravity_world"], np.float64)
    gravity = gravity / max(np.linalg.norm(gravity), 1e-12)
    people = []
    for oid in object_ids:
        row = rows_by_id(kd["object_ids"], oid)
        if row is None:
            people.append(None)
            continue
        q, j, v = q_all[row], j_all[row], v_all[row]
        people.append({
            "root_pos_w": q[:, :3].copy(),
            "root_quat_w": q[:, 3:7].copy(),
            "body_q": q[:, BODY_Q].reshape(-1, 21, 4).copy(),
            "joints_w": j[:, :NUM_BODY].copy(),
            "valid": v.copy(),
        })
    return {"people": people, "gravity": gravity, "betas": np.asarray(kd["betas"], np.float64)}


def lift(q_cam: np.ndarray, joints_cam: np.ndarray, extr: np.ndarray) -> dict:
    """Camera-frame BetterHuman ``q`` + joints -> the world payload."""
    rot_cw = extr[:, :3, :3]
    rot_wc = np.transpose(rot_cw, (0, 2, 1))
    center = -np.einsum("nij,nj->ni", rot_wc, extr[:, :3, 3])
    pos_c = q_cam[:, :3]
    root_rot_c = quat_to_matrix(q_cam[:, 3:7])
    return {
        "root_pos_w": np.einsum("nij,nj->ni", rot_wc, pos_c) + center,
        "root_quat_w": matrix_to_quat(rot_wc @ root_rot_c),
        "body_q": q_cam[:, BODY_Q].reshape(-1, 21, 4).copy(),
        "joints_w": np.einsum("nij,nkj->nki", rot_wc, joints_cam[:, :NUM_BODY]) + center[:, None],
    }


def load_run(run: str, scene: str, extr: np.ndarray) -> dict:
    """One prediction dump's people, lifted with ``extr``."""
    path = REPO / RUNS[run][0] / "predictions" / f"{scene}.npz"
    pred = np.load(path)
    covered = np.asarray(pred["covered"], bool)
    q_cam = np.asarray(pred["q_cam"], np.float64)
    joints = np.asarray(pred["joints_cam"], np.float64)
    object_ids = np.asarray(pred["object_ids"], np.int32)
    people = []
    for p in range(covered.shape[0]):
        v = covered[p] & np.isfinite(q_cam[p]).all(-1) & np.isfinite(joints[p]).all((1, 2))
        qc = np.where(v[:, None], q_cam[p], 0.0)
        qc[~v, 3:7] = np.array([0.0, 0.0, 0.0, 1.0])
        jc = np.where(v[:, None, None], joints[p], 0.0)
        out = lift(qc, jc, extr)
        out["valid"] = v
        people.append(out)
    tracked = np.asarray(pred["tracked"], bool) if "tracked" in pred.files else None
    return {"people": people, "object_ids": object_ids, "stride": int(pred["stride"]),
            "tracked": tracked}


def load_frozen(scene: str, extr: np.ndarray, object_ids: np.ndarray, parents: np.ndarray) -> dict:
    """The frozen SAM3D SMPL-X refit via ``viewer/bodies.py::frozen_source``."""
    from viewer.bodies import frozen_source
    shard = scene_shard(scene)
    src = frozen_source(CORPUS / "features" / "sam3d" / shard / scene / "smplx_params.npz",
                        np.asarray(extr, np.float32), object_ids, "cpu")
    people = []
    for person in src.people:
        if person is None:
            people.append(None)
            continue
        quat_w = np.asarray(person.bone_wxyz, np.float64)[..., [1, 2, 3, 0]]   # wxyz -> xyzw
        pos_w = np.asarray(person.bone_pos, np.float64)
        valid = np.asarray(person.valid, bool)
        rot_w = quat_to_matrix(np.where(valid[:, None, None], quat_w,
                                        np.array([0.0, 0.0, 0.0, 1.0])))
        body_local = np.einsum("njab,njac->njbc", rot_w[:, parents[1:NUM_BODY]], rot_w[:, 1:NUM_BODY])
        people.append({
            "root_pos_w": np.where(valid[:, None], pos_w[:, 0], 0.0),
            "root_quat_w": np.where(valid[:, None], quat_w[:, 0], np.array([0.0, 0.0, 0.0, 1.0])),
            "body_q": matrix_to_quat(body_local),
            "joints_w": np.where(valid[:, None, None], pos_w[:, :NUM_BODY], 0.0),
            "valid": valid,
        })
    return {"people": people}


def smplx_parents() -> np.ndarray:
    """SMPL-X kinematic parents (52 joints) from the BetterHuman archive."""
    from viewer.bodies import load_body
    return np.asarray(list(load_body("cpu").structure.parents), np.int64)


__all__ = ["REPO", "CORPUS", "RUNS", "NUM_BODY", "scene_shard", "test_scenes", "scene_cameras", "load_gt",
           "load_run", "load_frozen", "smplx_parents", "quat_to_matrix", "matrix_to_quat",
           "rows_by_id", "lift"]
