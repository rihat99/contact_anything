"""Render the ``mhr_1`` v2 pose pseudo-GT against the FROZEN model on video frames.

The pose audit's GT-side control: before blaming the pose losses for a bad
finetuned pose, this script shows what the *target* itself looks like. Per
frame it writes two panels over the original video frame:

    left  = frozen SAM3D prediction     right = mhr_1 v2 GT mesh

The GT panel poses a BetterHuman MHR body at the stored world-frame ``q_world``
(the exact ``from_classic`` convention ``contact/pose_supervision.py`` compares
against), applies the stored per-person 45 ``identity`` blendshapes, maps
world -> camera with the scene's per-frame ``cam_from_world`` extrinsics and
projects with the per-frame intrinsics. Both panels therefore share one
projection: the frozen model's own vertex projection is a pinhole with EXACTLY
the dataset's per-frame ``fx, fy`` and the image centre (verified to 0.000 px),
so the two meshes are directly comparable in pixels.

Head vertices are tinted in both panels and a 12 cm axis triad is drawn at the
``c_head`` joint (native x/y/z = B/G/R), because the open question is head
orientation.

``mhr_1.npz`` stores ``identity`` (45 surface blendshapes) but NOT the fitted
``proportions`` (73 bone scales), and in this rig the SKELETON is driven by the
proportions alone — identity moves the surface only. ``--gt-proportions``
selects what the GT body's skeleton uses: ``frozen`` (the frozen model's own
per-scene median, the closest available match, default) or ``zero`` (the MHR
mean skeleton, GT-only but a different body). Both variants' joint agreement
with kindyn is reported in the summary either way.

Alongside the videos it writes, per scene, a summary JSON (GT self-consistency,
identity magnitudes, head/neck q-channel statistics) and a PNG of the six
head/neck ``q`` channels over time, GT vs frozen.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import render_climbing_pose_video as rpv
import render_climbing_video_contacts as rcv

from contact.config import load_config
from contact.data.climbing_corpus import (
    KP_JOINT_NAMES, NUM_MHR70, ClimbingCorpusDataset, _rows_by_object_id,
)
from contact.data.collate import batch_to_device, make_collate
from contact.engine import forward_model
from contact.model import build_model
from contact.pose_supervision import POSE_SLOTS
from contact.targets import TargetSpec

#: kindyn SMPL-X joint -> MHR native joint, for the cross-rig GT joint check
#: (the ``scripts/convert_kindyn_to_mhr.py`` map, restricted to the 13
#: :data:`KP_JOINT_NAMES` columns the renderer already carries).
KP_TO_MHR_NATIVE = {
    "left_shoulder": "l_uparm", "right_shoulder": "r_uparm",
    "left_elbow": "l_lowarm", "right_elbow": "r_lowarm",
    "left_wrist": "l_wrist", "right_wrist": "r_wrist",
    "left_hip": "l_upleg", "right_hip": "r_upleg",
    "left_knee": "l_lowleg", "right_knee": "r_lowleg",
    "left_ankle": "l_talocrural", "right_ankle": "r_talocrural",
    "neck": "c_neck",
}
#: MHR archive used by ``pose_supervision`` / ``convert_kindyn_to_mhr``.
MHR_ARCHIVE = "/data3/rikhat.akizhanov/better/BetterHuman/models/MHR/converted/mhr_lod1.npz"
#: SAM's native-to-camera axis flip (``contact/physics/adapter.py``).
D_FLIP = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
#: The six head/neck ``q`` channels — the last six of the 132-wide vector.
HEAD_CHANNEL_NAMES = ("neck_bend", "neck_lean", "neck_twist",
                      "head_bend", "head_lean", "head_twist")
#: Native joints whose skinning weight defines the tinted head patch.
HEAD_SKIN_JOINTS = ("c_head", "c_jaw", "c_jaw_null", "l_eye", "r_eye",
                    "l_eye_null", "r_eye_null", "c_head_null")

FROZEN_COLOR = (208, 178, 130)     # BGR steel blue (matches the pose renderer)
GT_COLOR_MESH = (150, 200, 245)    # BGR warm sand
HEAD_TINT = (90, 90, 235)          # BGR red-ish head patch
AXIS_COLORS = ((255, 90, 90), (90, 220, 90), (90, 90, 255))  # x, y, z in BGR
MESH_ALPHA = 0.55


def _head_face_mask(body, faces: np.ndarray) -> np.ndarray:
    """Boolean ``(F,)`` mask of triangles whose three vertices skin to the head."""
    names = body.structure.joint_names
    columns = [names.index(name) for name in HEAD_SKIN_JOINTS if name in names]
    weight = body.values.skinning_weight_matrix[:, columns].sum(-1)
    is_head = (weight > 0.5).cpu().numpy()
    return is_head[faces].all(axis=1)


def _predict_frozen_pass(
    model, ds: ClimbingCorpusDataset, scene: str, batch_size: int, device: str,
    collate, requests_by_t: dict, head_joint: int,
) -> dict:
    """Frozen-model pass; ``rpv._predict_pose_pass`` plus the head frame and params.

    Adds ``model_params (P, N, 204)``, ``shape (P, N, 45)`` (for the q-channel
    statistics) and the ``c_head`` camera-frame pose. ``pred_joint_coords`` is
    already axis-flipped (``= D @ native``, verified to 1e-5 cm) while
    ``joint_global_rots`` is native, so the camera rotation is ``D @ R_native``.
    """
    data = ds._scenes[scene]
    frame_indices = data["frame_indices"]
    fps = float(data["fps"])
    n_people, n_frames = data["valid_mask"].shape
    item_index = rcv._frame_index_map(ds, scene)
    kp_idx = list(range(NUM_MHR70))
    flip = torch.as_tensor(D_FLIP, device=device)

    out: dict = {
        "kp2d": np.full((n_people, n_frames, len(kp_idx), 2), np.nan, np.float32),
        "kp3d_cam": np.full((n_people, n_frames, len(kp_idx), 3), np.nan, np.float32),
        "model_params": np.full((n_people, n_frames, 204), np.nan, np.float32),
        "shape": np.full((n_people, n_frames, 45), np.nan, np.float32),
        "scale": None,
        "head_pos_cam": np.full((n_people, n_frames, 3), np.nan, np.float32),
        "head_rot_cam": np.full((n_people, n_frames, 3, 3), np.nan, np.float32),
    }

    for seq_len in sorted(requests_by_t):
        requests = requests_by_t[seq_len]
        for lo in tqdm(range(0, len(requests), batch_size),
                       desc=f"frozen {scene} T={seq_len}", leave=False):
            selected = requests[lo:lo + batch_size]
            clips = []
            for person, positions, _ in selected:
                start_index = float(frame_indices[positions[0]])
                clip = []
                for position in positions:
                    frame = dict(ds[item_index[(person, position)]][0])
                    frame["frame_pos_sec"] = (
                        float(frame_indices[position]) - start_index) / fps
                    clip.append(frame)
                clips.append(clip)

            batch = batch_to_device(collate(clips), device)
            with torch.inference_mode():
                output = forward_model(model, batch)
            mhr = output["mhr"]
            cam_t = mhr["pred_cam_t"].float()
            verts2d = mhr["pred_keypoints_2d_verts"].float().cpu().numpy()
            verts_cam = (mhr["pred_vertices"].float() + cam_t[:, None]).cpu().numpy()
            kp2d = mhr["pred_keypoints_2d"][:, kp_idx].float().cpu().numpy()
            kp3d_cam = (mhr["pred_keypoints_3d"][:, kp_idx].float()
                        + cam_t[:, None]).cpu().numpy()
            head_pos = (mhr["pred_joint_coords"].float()[:, head_joint]
                        + cam_t).cpu().numpy()
            head_rot = (flip @ mhr["joint_global_rots"].float()[:, head_joint]
                        ).cpu().numpy()
            params = mhr["mhr_model_params"].float().cpu().numpy()
            shape = mhr["shape"].float().cpu().numpy()
            scale = mhr["scale"].float().cpu().numpy()
            if "faces" not in out:
                out["faces"] = np.asarray(
                    mhr["faces"].cpu() if torch.is_tensor(mhr["faces"])
                    else mhr["faces"]).astype(np.int64)
                n_verts = verts2d.shape[1]
                out["verts2d"] = np.full(
                    (n_people, n_frames, n_verts, 2), np.nan, np.float16)
                out["verts_cam"] = np.full(
                    (n_people, n_frames, n_verts, 3), np.nan, np.float16)
                out["scale"] = np.full(
                    (n_people, n_frames, scale.shape[-1]), np.nan, np.float32)

            for clip_index, (person, positions, offsets) in enumerate(selected):
                for offset in offsets:
                    row = clip_index * seq_len + offset
                    pos = positions[offset]
                    out["verts2d"][person, pos] = verts2d[row]
                    out["verts_cam"][person, pos] = verts_cam[row]
                    out["kp2d"][person, pos] = kp2d[row]
                    out["kp3d_cam"][person, pos] = kp3d_cam[row]
                    out["model_params"][person, pos] = params[row]
                    out["shape"][person, pos] = shape[row]
                    out["scale"][person, pos] = scale[row]
                    out["head_pos_cam"][person, pos] = head_pos[row]
                    out["head_rot_cam"][person, pos] = head_rot[row]
    return out


def _gt_pass(
    body0, data: dict, proportions: torch.Tensor | None, device: str,
    chunk: int, head_joint: int, kp_body_ids: list[int],
) -> dict:
    """Pose the mhr_1 GT body per frame and project it like the frozen panel.

    :param body0: unshaped :class:`better_human.MHR` (LOD1).
    :param proportions: ``(73,)`` bone scales, or ``None`` for the rig default.
    :param head_joint: ``robot.body_names`` row of ``c_head``.
    :param kp_body_ids: ``robot.body_names`` rows matching :data:`KP_JOINT_NAMES`.
    """
    from better_robot import forward_kinematics
    from better_robot.lie import so3

    q_all = data["pose_gt_q"]
    valid = data["pose_valid_mask"] & data["valid_mask"]
    identity = data["pose_identity"]
    extrinsics = data["extrinsics"]
    intrinsics = data["intrinsics"]
    n_people, n_frames = valid.shape
    n_verts = int(body0.values.v_template.shape[0])

    out = {
        "faces": body0.structure.faces.cpu().numpy().astype(np.int64),
        "verts2d": np.full((n_people, n_frames, n_verts, 2), np.nan, np.float16),
        "verts_cam": np.full((n_people, n_frames, n_verts, 3), np.nan, np.float16),
        "kp2d": np.full((n_people, n_frames, len(kp_body_ids), 2), np.nan, np.float32),
        "kp3d_cam": np.full((n_people, n_frames, len(kp_body_ids), 3), np.nan,
                            np.float32),
        "head_pos_cam": np.full((n_people, n_frames, 3), np.nan, np.float32),
        "head_rot_cam": np.full((n_people, n_frames, 3, 3), np.nan, np.float32),
        "joints_world": np.full((n_people, n_frames, len(kp_body_ids), 3), np.nan,
                                np.float32),
    }
    for person in range(n_people):
        rows = np.flatnonzero(valid[person])
        if rows.size == 0:
            continue
        groups = {"identity": torch.as_tensor(identity[person], device=device)}
        if proportions is not None:
            groups["proportions"] = proportions
        body = body0.with_shape(**groups)
        for lo in range(0, rows.size, chunk):
            sel = rows[lo:lo + chunk]
            q = torch.as_tensor(q_all[person, sel], device=device)
            with torch.no_grad():
                fk = forward_kinematics(body.robot, q)
                verts_w = body.vertices_from_data(fk).cpu().numpy()
                pose_w = fk.joint_pose_world
                joints_w = pose_w[..., :3].cpu().numpy()
                head_rot_w = so3.to_matrix(pose_w[:, head_joint, 3:7]).cpu().numpy()
            rot = extrinsics[sel, :3, :3]
            trans = extrinsics[sel, :3, 3]
            verts_cam = np.einsum("nij,nkj->nki", rot, verts_w) + trans[:, None]
            kp_w = joints_w[:, kp_body_ids]
            kp_cam = np.einsum("nij,nkj->nki", rot, kp_w) + trans[:, None]
            head_cam = np.einsum("nij,nj->ni", rot, joints_w[:, head_joint]) + trans
            intr = intrinsics[sel]
            out["verts2d"][person, sel] = _project_batch(verts_cam, intr)
            out["verts_cam"][person, sel] = verts_cam
            out["kp2d"][person, sel] = _project_batch(kp_cam, intr)
            out["kp3d_cam"][person, sel] = kp_cam
            out["joints_world"][person, sel] = kp_w
            out["head_pos_cam"][person, sel] = head_cam
            out["head_rot_cam"][person, sel] = np.einsum(
                "nij,njk->nik", rot, head_rot_w)
    return out


def _project_batch(points_cam: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Project ``(n, k, 3)`` camera points with per-frame ``(n, 3, 3)`` intrinsics."""
    z = np.clip(points_cam[..., 2:3], 1e-6, None)
    uv = points_cam[..., :2] / z
    focal = np.stack([intrinsics[:, 0, 0], intrinsics[:, 1, 1]], -1)[:, None]
    principal = np.stack([intrinsics[:, 0, 2], intrinsics[:, 1, 2]], -1)[:, None]
    return uv * focal + principal


def _draw_mesh(img: np.ndarray, verts2d: np.ndarray, verts_cam: np.ndarray,
               faces: np.ndarray, base_color, head_mask: np.ndarray) -> None:
    """Painter-sorted lambert mesh with the head patch tinted (``rpv._draw_mesh``)."""
    tri2d = verts2d[faces].astype(np.float32)
    tricam = verts_cam[faces].astype(np.float32)
    normals = np.cross(tricam[:, 1] - tricam[:, 0], tricam[:, 2] - tricam[:, 0])
    norm = np.linalg.norm(normals, axis=1) + 1e-8
    shade = 0.30 + 0.70 * np.clip(np.abs(normals[:, 2]) / norm, 0.0, 1.0)
    depth = tricam[..., 2].mean(axis=1)
    keep = np.isfinite(tri2d).all(axis=(1, 2)) & np.isfinite(depth) & (depth > 0.05)
    height, width = img.shape[:2]
    xs, ys = tri2d[..., 0], tri2d[..., 1]
    keep &= (xs.max(1) >= 0) & (xs.min(1) < width) & (ys.max(1) >= 0) & (ys.min(1) < height)
    order = np.argsort(-depth[keep])
    tri2d = tri2d[keep][order].astype(np.int32)
    shade = shade[keep][order]
    is_head = head_mask[keep][order]
    base = np.asarray(base_color, np.float32)
    tint = np.asarray(HEAD_TINT, np.float32)
    overlay = img.copy()
    for pts, lit, head in zip(tri2d, shade, is_head):
        color = tint if head else base
        cv2.fillConvexPoly(overlay, pts, tuple(float(c) for c in color * lit))
    cv2.addWeighted(overlay, MESH_ALPHA, img, 1.0 - MESH_ALPHA, 0.0, dst=img)


def _draw_axes(img: np.ndarray, position: np.ndarray, rotation: np.ndarray,
               intrinsics: np.ndarray, length: float = 0.12) -> None:
    """Project a camera-frame axis triad at ``position`` onto ``img``."""
    if not (np.isfinite(position).all() and np.isfinite(rotation).all()):
        return
    tips = position[None] + length * rotation.T          # rows = x, y, z axes
    pts = _project_batch(np.concatenate([position[None], tips])[None], intrinsics[None])[0]
    origin = pts[0]
    if not np.isfinite(origin).all():
        return
    for axis in range(3):
        tip = pts[1 + axis]
        if not np.isfinite(tip).all():
            continue
        cv2.line(img, (int(origin[0]), int(origin[1])), (int(tip[0]), int(tip[1])),
                 AXIS_COLORS[axis], 2, cv2.LINE_AA)


def _panel(frame: np.ndarray, pass_data: dict, frame_pos: int, gt2d: np.ndarray,
           intrinsics: np.ndarray, color, head_mask: np.ndarray,
           label: str) -> np.ndarray:
    img = frame.copy()
    for person in range(pass_data["verts2d"].shape[0]):
        verts2d = pass_data["verts2d"][person, frame_pos].astype(np.float32)
        if not np.isfinite(verts2d).any():
            continue
        verts_cam = pass_data["verts_cam"][person, frame_pos].astype(np.float32)
        _draw_mesh(img, verts2d, verts_cam, pass_data["faces"], color, head_mask)
        _draw_axes(img, pass_data["head_pos_cam"][person, frame_pos],
                   pass_data["head_rot_cam"][person, frame_pos], intrinsics)
    for person in range(gt2d.shape[0]):
        rpv._draw_kp(img, gt2d[person], rpv.GT_COLOR, 4)
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (25, 25, 25), -1)
    cv2.putText(img, label, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (240, 240, 240), 2, cv2.LINE_AA)
    return img


def _render_video(scene: str, data: dict, frozen: dict, gt: dict, gt_px: np.ndarray,
                  head_mask: np.ndarray, path: Path, scale: float,
                  gt_label: str) -> dict:
    frames_dir = data["frames_dir"]
    n_frames = len(data["frame_indices"])
    first = cv2.imread(str(frames_dir / "000000.jpg"), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(frames_dir / "000000.jpg")
    height, width = first.shape[:2]
    out_w, out_h = int(round(2 * width * scale)), int(round(height * scale))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(data["fps"]), (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    try:
        for pos in tqdm(range(n_frames), desc=f"render {scene}", leave=False):
            frame = cv2.imread(str(frames_dir / f"{pos:06d}.jpg"), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(frames_dir / f"{pos:06d}.jpg")
            intr = data["intrinsics"][pos]
            left = _panel(frame, frozen, pos, gt_px[:, pos], intr, FROZEN_COLOR,
                          head_mask, "frozen SAM3D prediction")
            right = _panel(frame, gt, pos, gt_px[:, pos], intr, GT_COLOR_MESH,
                           head_mask, gt_label)
            combo = np.concatenate([left, right], axis=1)
            if scale != 1.0:
                combo = cv2.resize(combo, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(combo)
    finally:
        writer.release()
    return {"frames": n_frames, "size": [out_w, out_h], "fps": float(data["fps"])}


def _wrap(diff: np.ndarray) -> np.ndarray:
    """Wrap radian differences to ``(-pi, pi]``."""
    return np.remainder(diff + np.pi, 2.0 * np.pi) - np.pi


def _frozen_q(body0, frozen: dict, valid: np.ndarray, device: str) -> np.ndarray:
    """The frozen model's own ``q (P, N, 132)`` via the training-path conversion."""
    from better_human.bodies import MHRClassic

    n_people, n_frames = valid.shape
    out = np.full((n_people, n_frames, 132), np.nan, np.float32)
    for person in range(n_people):
        rows = np.flatnonzero(valid[person])
        if rows.size == 0:
            continue
        params = torch.as_tensor(frozen["model_params"][person, rows], device=device)
        shape = torch.as_tensor(frozen["shape"][person, rows], device=device)
        with torch.no_grad():
            _, q = body0.from_classic(
                MHRClassic(identity_coeffs=shape, model_parameters=params))
        out[person, rows] = q.cpu().numpy()
    return out


def _channel_stats(q_gt: np.ndarray, q_frozen: np.ndarray,
                   valid: np.ndarray) -> dict:
    """Per-channel GT vs frozen statistics over the scene's valid rows."""
    rows_gt = q_gt[valid]                                   # (R, 132)
    rows_fr = q_frozen[valid]
    keep = np.isfinite(rows_gt).all(1) & np.isfinite(rows_fr).all(1)
    rows_gt, rows_fr = rows_gt[keep], rows_fr[keep]
    diff = np.abs(_wrap(rows_gt - rows_fr))
    head = {}
    for offset, name in enumerate(HEAD_CHANNEL_NAMES):
        channel = 132 - len(HEAD_CHANNEL_NAMES) + offset
        head[name] = {
            "q_index": channel,
            "gt_mean": float(rows_gt[:, channel].mean()),
            "gt_std": float(rows_gt[:, channel].std()),
            "gt_min": float(rows_gt[:, channel].min()),
            "gt_max": float(rows_gt[:, channel].max()),
            "frozen_mean": float(rows_fr[:, channel].mean()),
            "frozen_std": float(rows_fr[:, channel].std()),
            "frozen_min": float(rows_fr[:, channel].min()),
            "frozen_max": float(rows_fr[:, channel].max()),
            "abs_diff_mean": float(diff[:, channel].mean()),
            "abs_diff_median": float(np.median(diff[:, channel])),
        }
    local = slice(POSE_SLOTS.start, POSE_SLOTS.stop)
    med = np.median(diff[:, local], axis=0)
    order = np.argsort(-med)[:10]
    return {
        "n_rows": int(rows_gt.shape[0]),
        "head_channels": head,
        "all125_abs_diff_mean_rad": float(diff[:, local].mean()),
        "all125_abs_diff_median_rad": float(np.median(diff[:, local])),
        "gt_std_rad_mean_over_125": float(rows_gt[:, local].std(0).mean()),
        "frozen_std_rad_mean_over_125": float(rows_fr[:, local].std(0).mean()),
        "gt_near_constant_channels": int((rows_gt[:, local].std(0) < 1e-3).sum()),
        "frozen_near_constant_channels": int((rows_fr[:, local].std(0) < 1e-3).sum()),
        "worst10_channels": [
            {"local_index": int(i), "q_index": int(i + POSE_SLOTS.start),
             "abs_diff_median_rad": float(med[i]),
             "gt_std_rad": float(rows_gt[:, i + POSE_SLOTS.start].std())}
            for i in order],
    }


def _head_chain_stats(body0, q: np.ndarray, valid: np.ndarray,
                      device: str) -> dict:
    """Is the neck/head split identified, or only its composition?

    The six head/neck channels rotate one chain ``c_spine3 -> c_neck ->
    c_head``, so a surface fit can trade neck against head freely. Compares the
    per-frame jumpiness of the individual CHANNELS with that of the COMPOSED
    head-relative-to-chest orientation: channels jumping while the composition
    stays smooth means the decomposition is degenerate and a per-channel loss
    supervises an arbitrary split.
    """
    from better_robot import forward_kinematics
    from better_robot.lie import so3

    body_names = list(body0.robot.body_names)
    head, chest = body_names.index("c_head"), body_names.index("c_spine3")
    person = int(np.argmax(valid.sum(1)))
    rows = np.flatnonzero(valid[person] & np.isfinite(q[person]).all(-1))
    if rows.size < 3:
        return {}
    # Consecutive-frame deltas only (a gap would fake a jump).
    step = np.flatnonzero(np.diff(rows) == 1)
    with torch.no_grad():
        pose = forward_kinematics(
            body0.robot, torch.as_tensor(q[person, rows], device=device)
        ).joint_pose_world
        rot = so3.to_matrix(pose[..., 3:7])
        rel = rot[:, chest].transpose(-1, -2) @ rot[:, head]
        delta = rel[step].transpose(-1, -2) @ rel[step + 1]
        geodesic = np.degrees(
            so3.log(so3.from_matrix(delta)).norm(dim=-1).cpu().numpy())
    channels = np.degrees(np.abs(np.diff(q[person, rows][:, -6:], axis=0))[step])
    return {
        "composed_head_vs_chest_deg_per_frame": {
            "median": float(np.median(geodesic)),
            "p99": float(np.percentile(geodesic, 99)),
            "max": float(geodesic.max())},
        "channel_deg_per_frame": {
            name: {"median": float(np.median(channels[:, i])),
                   "p99": float(np.percentile(channels[:, i], 99)),
                   "max": float(channels[:, i].max())}
            for i, name in enumerate(HEAD_CHANNEL_NAMES)},
        "n_steps": int(step.size),
    }


def _head_plot(scene: str, fps: float, q_gt: np.ndarray, q_frozen: np.ndarray,
               valid: np.ndarray, path: Path) -> None:
    person = int(np.argmax(valid.sum(1)))
    time = np.arange(q_gt.shape[1]) / fps
    fig, axes = plt.subplots(6, 1, figsize=(11, 12), sharex=True)
    for offset, (name, axis) in enumerate(zip(HEAD_CHANNEL_NAMES, axes)):
        channel = 132 - len(HEAD_CHANNEL_NAMES) + offset
        axis.plot(time, q_gt[person, :, channel], "k-", lw=1.4, label="mhr_1 v2 GT")
        axis.plot(time, q_frozen[person, :, channel], color="tab:blue", lw=1.2,
                  label="frozen model")
        axis.set_ylabel(f"{name}\n[rad]")
        if offset == 0:
            axis.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"{scene} — head/neck q channels (person {person})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _joint_check(gt: dict, data: dict, kin_world: np.ndarray,
                 valid: np.ndarray) -> dict:
    """GT-mesh joints vs kindyn ``joints_world`` — metres and pixels (check 4b).

    Cross-RIG by construction: ``gt["joints_world"]`` are the BetterHuman MHR
    body's own joint origins posed at the mhr_1 q, ``kin_world`` is kindyn's
    SMPL-X rig (:func:`_kindyn_joints`). The loader's ``kp3d_world`` can no
    longer stand in for the kindyn side — since the MHR-native supervision swap
    it IS the MHR70 GT, which would turn this into an MHR-vs-MHR self-check.
    """
    pred = gt["joints_world"]
    both = valid & np.isfinite(pred).all(axis=(2, 3)) & np.isfinite(kin_world).all(axis=(2, 3))
    if not both.any():
        return {}
    dist = np.linalg.norm(pred - kin_world, axis=-1)[both] * 100.0
    px = np.linalg.norm(gt["kp2d"] - _gt_pixels(kin_world, data), axis=-1)[both]
    return {
        "joint3d_median_cm": float(np.median(dist)),
        "joint3d_mean_cm": float(dist.mean()),
        "joint3d_p95_cm": float(np.percentile(dist, 95)),
        "reproj_median_px": float(np.nanmedian(px)),
        "reproj_p95_px": float(np.nanpercentile(px, 95)),
        "per_joint_median_cm": {
            name: float(np.median(dist[:, i]))
            for i, name in enumerate(KP_JOINT_NAMES)},
    }


def _kindyn_joints(data: dict, scene: str) -> np.ndarray:
    """kindyn ``joints_world`` restricted to :data:`KP_JOINT_NAMES`, id-aligned.

    ``(P, N, 13, 3)`` metres, world frame — the cross-rig reference the MHR-native
    keypoint GT no longer provides.
    """
    kindyn = np.load(data["dir"] / "kindyn_1.npz", allow_pickle=True)
    names = [str(x) for x in kindyn["joint_names"]]
    joints = _rows_by_object_id(
        np.asarray(kindyn["joints_world"], np.float32), kindyn["object_ids"],
        data["object_ids"], scene, "kindyn_1")
    return joints[:, :, [names.index(name) for name in KP_JOINT_NAMES]]


def _gt_pixels(kp_world: np.ndarray, data: dict) -> np.ndarray:
    """World keypoints -> pixels through the scene's own camera."""
    ext = data["extrinsics"]
    cam = np.einsum("nij,pnkj->pnki", ext[:, :3, :3], kp_world) + ext[None, :, None, :3, 3]
    px = np.full(cam.shape[:-1] + (2,), np.nan, np.float32)
    for frame in range(cam.shape[1]):
        px[:, frame] = rpv._project(cam[:, frame], data["intrinsics"][frame])
    return px


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", required=True,
                        help="scene names; a leading space escapes names that "
                             "start with '-' (argparse would read them as flags)")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO / "output" / "mhr_gt_vs_frozen")
    parser.add_argument("--gt-proportions", choices=("frozen", "zero"),
                        default="frozen",
                        help="skeleton bone scales for the GT body (mhr_1 does "
                             "not store the fitted ones)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--gt-chunk", type=int, default=32)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--no-video", action="store_true",
                        help="statistics only (skip the mp4s)")
    args = parser.parse_args()

    import better_human as bh

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    cfg["train"]["compile_backbone"] = False
    root, contact_level = rcv._dataset_options(cfg)
    scenes = [name.strip() for name in args.scenes][args.shard_index::args.num_shards]
    print(f"[shard {args.shard_index}/{args.num_shards}] scenes: {', '.join(scenes)}")

    spec = TargetSpec.from_config(cfg)
    model, _ = build_model(cfg, args.device)
    model.eval()
    collate = make_collate(tuple(model.cfg.MODEL.IMAGE_SIZE), spec)
    seq_cfg = cfg["data"]["sequence"]

    body0 = bh.MHR(MHR_ARCHIVE, lod=1, use_expression=False, use_correctives=False,
                   compute_mass=False, device=args.device)
    proportion_indices = body0._tree.proportion_parameter_indices
    body_names = list(body0.robot.body_names)
    head_body = body_names.index("c_head")
    head_native = body0.structure.joint_names.index("c_head")
    kp_body_ids = [body_names.index(KP_TO_MHR_NATIVE[name]) for name in KP_JOINT_NAMES]
    faces = body0.structure.faces.cpu().numpy().astype(np.int64)
    head_mask = _head_face_mask(body0, faces)

    records = []
    for scene in scenes:
        ds = ClimbingCorpusDataset(
            root, scenes=[scene], split=args.split, frames_per_clip=1,
            frame_stride=1, jitter=False, seed=int(cfg["data"]["seed"]),
            contact_level=contact_level, use_confidence_weights=False,
            require_labels=False, load_keypoints=True, load_pose=True)
        data = ds._scenes[scene]
        requests = rpv.full_scene_requests(
            data["valid_mask"],
            max(1, int(round(float(data["fps"]) / 25.0)))
            if seq_cfg["frame_stride"] == "auto" else int(seq_cfg["frame_stride"]),
            args.max_frames)
        frozen = _predict_frozen_pass(model, ds, scene, args.batch_size,
                                      args.device, collate, requests, head_native)
        if not np.array_equal(frozen["faces"], faces):
            raise RuntimeError(f"{scene}: frozen and BetterHuman MHR topologies differ")

        valid = data["pose_valid_mask"] & data["valid_mask"]
        flat = frozen["model_params"].reshape(-1, 204)
        flat = flat[np.isfinite(flat).all(1)]
        frozen_props = torch.as_tensor(flat, device=args.device).index_select(
            -1, proportion_indices).median(0).values
        variants = {"frozen": frozen_props, "zero": None}
        gt = _gt_pass(body0, data, variants[args.gt_proportions], args.device,
                      args.gt_chunk, head_body, kp_body_ids)
        other = "zero" if args.gt_proportions == "frozen" else "frozen"
        gt_other = _gt_pass(body0, data, variants[other], args.device,
                            args.gt_chunk, head_body, kp_body_ids)

        # MHR70 GT keypoints (mhr_sup_1, same rig as the prediction) for the
        # frozen-model error; kindyn's own SMPL-X joints for the cross-rig check.
        mhr_world = data["kp3d_world"]                       # [P, N, 70, 3]
        mhr_valid = valid & data["kp_valid"]
        gt_px = np.where(mhr_valid[:, :, None, None], _gt_pixels(mhr_world, data), np.nan)
        kin_world = _kindyn_joints(data, scene)              # [P, N, 13, 3]
        kin_valid = valid & np.isfinite(kin_world).all(axis=(2, 3))

        q_gt = np.where(valid[:, :, None], data["pose_gt_q"], np.nan)
        q_frozen = _frozen_q(body0, frozen, valid, args.device)
        stats = _channel_stats(q_gt, q_frozen, valid)
        _head_plot(scene, float(data["fps"]), q_gt, q_frozen, valid,
                   output_dir / f"{scene}_head_channels.png")

        mhr_npz = np.load(data["dir"] / "mhr_1.npz", allow_pickle=True)
        fit_err = np.asarray(mhr_npz["fit_err_cm"], np.float32)
        identity = data["pose_identity"]
        record = {
            "scene": scene,
            "n_people": int(valid.shape[0]),
            "n_valid_frames": int(valid.sum()),
            "fps": float(data["fps"]),
            "gt_proportions": args.gt_proportions,
            "fit_err_cm": {
                "mean": float(np.nanmean(fit_err)),
                "median": float(np.nanmedian(fit_err)),
                "p95": float(np.nanpercentile(fit_err, 95)),
                "max": float(np.nanmax(fit_err))},
            "joint_vs_kindyn_med_cm_stored": np.asarray(
                mhr_npz["joint_vs_kindyn_med_cm"]).tolist(),
            "gt_joint_check": {
                args.gt_proportions: _joint_check(gt, data, kin_world, kin_valid),
                other: _joint_check(gt_other, data, kin_world, kin_valid)},
            "identity": {
                "abs_mean": np.abs(identity).mean(1).tolist(),
                "abs_max": np.abs(identity).max(1).tolist(),
                "frozen_shape_abs_mean": float(
                    np.nanmean(np.abs(frozen["shape"]))),
                "frozen_shape_abs_max": float(np.nanmax(np.abs(frozen["shape"]))),
                "frozen_scale_abs_mean": float(np.nanmean(np.abs(frozen["scale"])))},
            "q_channels": stats,
            "head_chain_gt": _head_chain_stats(body0, q_gt, valid, args.device),
            "head_chain_frozen": _head_chain_stats(body0, q_frozen, valid,
                                                   args.device),
            "frozen_err": {},
        }
        frozen_err = rpv._pass_errors(
            frozen, np.where(mhr_valid[:, :, None, None], _cam_frame(mhr_world, data),
                             np.nan), gt_px)
        record["frozen_err"] = {
            "err3d_m": float(np.nanmean(frozen_err["err3d"])),
            "err2d_px": float(np.nanmean(frozen_err["err2d"]))}

        if not args.no_video:
            label = ("mhr_1 v2 GT (identity + frozen proportions)"
                     if args.gt_proportions == "frozen"
                     else "mhr_1 v2 GT (identity, default proportions)")
            record.update(_render_video(
                scene, data, frozen, gt, gt_px, head_mask,
                output_dir / f"{scene}.mp4", args.scale, label))
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "q_channels"},
                         indent=2))
        print("head channels:", json.dumps(stats["head_channels"], indent=2))

    path = output_dir / f"summary_shard{args.shard_index}.json"
    path.write_text(json.dumps(records, indent=2))
    print(f"wrote {path}")
    return 0


def _cam_frame(kp_world: np.ndarray, data: dict) -> np.ndarray:
    """World keypoints -> camera frame ``(P, N, K, 3)``."""
    ext = data["extrinsics"]
    return (np.einsum("nij,pnkj->pnki", ext[:, :3, :3], kp_world)
            + ext[None, :, None, :3, 3]).astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
