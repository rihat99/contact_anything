"""Cropped video next to two fixed 3D views of the predicted body and forces, per wild clip.

For every out-tree ``<out-root>/<stem>/`` with ``predictions/<run>/{smplx,forces_sup}.npz`` one
``<out>/<stem>.mp4`` of three ``--side`` squares in a row is written:

* the source frame cropped around the person (``render_wild_overlays.crop_windows``, the same
  window as ``results_frames/``) with the white skeleton and the yellow force arrows of
  ``results/`` drawn on it (no fog);
* the ``camera view``: the SMPL-X mesh rebuilt from the dumped ``q_cam`` + betas, in the metric
  world with gravity down, seen by a FIXED virtual camera that looks from the source camera's
  side (its first-frame heading, levelled) and is pulled back so the WHOLE trajectory stays in
  frame — unless that would leave the body small (``FOLLOW_IF_FARTHER``): then the camera keeps the
  body large and slides after the pelvis path smoothed over ``FOLLOW_SMOOTH_SEC`` (a dolly: it
  translates, never turns, and the grid shows the world motion), zooming slowly so that every frame
  within ``ZOOM_HOLD_SEC`` stays inside the panel; and
* the ``side view``: the same scene from a camera turned ``--side-deg`` about the vertical.

The body is BVR's light blue, each joint that carries contact slots gets one thick yellow arrow
(the in-contact slots' world forces summed, ``ARROW_M_PER_BW_3D`` metres per body weight, folded
like the 2D overlays; the hand arrow starts at the middle-finger base, the foot arrow mid-foot),
a soft floor with a grid at the lowest foot carries the body's shadow (the mesh projected flat along
the key light). Lit per view by a key, a fill and a rim (``LIGHTS``), rendered at ``SUPERSAMPLE`` x
and shrunk, over a gradient.
Rendered headless with pyrender over EGL (``PYOPENGL_PLATFORM=egl`` is set here).

``--tight`` writes JPEG frames instead — ``<out>/{video,camera,side}/<stem>/<frame>.jpg`` and the
three side by side in ``<out>/united/<stem>/<frame>.jpg`` — with every panel a tight square around
the person (the video crop on the joints alone, the 3D views on a per-frame camera around that
frame's body and arrows, padded by ``TIGHT_VIDEO_PAD`` / ``TIGHT_PAD``); the video panel is undrawn.

    .venv/bin/python scripts/render_wild_3d.py --out-root ../data/willd_videos/out \\
        --out ../data/willd_videos/results_3d --stems backflip
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import pyrender
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _render_common import open_writer                                       # noqa: E402
from render_wild_overlays import (ANCHOR_JOINT, ARROW_M_PER_BW, ARROW_MIN_BW, ARROW_PALETTES,  # noqa: E402
                                  FOLD_JOINT, IMWRITE_PARAMS, crop_frame, crop_windows, draw_arrows,
                                  draw_skeleton, joint_forces_cam, load_people, pixels_per_metre)
from model.contact_frames import body22_parent, contact_set                  # noqa: E402
from viewer.bodies import load_body                                          # noqa: E402

BODY_RGB = (166, 189, 219)          # BVR's light blue
ARROW_RGB = (255, 196, 0)           # golden yellow: reads against the pale floor and the blue body
GROUND_RGB = (246, 246, 248)        # a soft floor that receives the key light's shadow
GRID_RGB = (196, 198, 204)
GRID_LINE_R = 0.006                 # metres: the grid is drawn as thin rods, a GL line is one pixel
BACKGROUND_TOP = (252, 252, 254)    # vertical gradient behind the scene (BGR-agnostic greys)
BACKGROUND_BOTTOM = (226, 229, 235)
SUPERSAMPLE = 2                     # render at this multiple of --side and shrink (anti-aliasing)
#: Studio lights relative to each view: (name, yaw deg, pitch deg, intensity), all directional. Yaw
#: turns the light about the view's up axis (positive = to the viewer's left), pitch raises it above
#: the line of sight. The key also casts the body's shadow onto the floor (a planar projection).
LIGHTS = (("key", -35.0, 45.0, 2.6),      # upper left
          ("fill", 50.0, 10.0, 1.0),      # soft, from the right, near eye level
          ("rim", 160.0, 35.0, 1.8))      # from behind, outlines the body against the floor
AMBIENT = 0.22
SHADOW_RGB = (0.66, 0.66, 0.69)     # opaque and unlit: the flattened triangles overlap and face both ways
SHADOW_LIFT_M = 0.009               # the flat shadow floats this far above the floor, over the grid rods
ARROW_SHAFT_R = 0.035               # metres
ARROW_HEAD_R = 0.09
ARROW_HEAD_L = 0.2
ARROW_M_PER_BW_3D = 1.0             # metres per body weight (longer than the 2D overlays' 0.7)
GRID_CELL_M = 0.5
GRID_MARGIN_CELLS = 30              # the floor runs this far past the trajectory
FOG_M = 6.0                         # beyond the trajectory the floor fades into the backdrop over this distance
GHOST_ALPHA = 0.45                  # arrow parts hidden inside the body show through at this opacity
YFOV_DEG = 40.0
TIGHT_PAD = 0.08                    # --tight: padding of the 3D views
TIGHT_VIDEO_PAD = 0.15              # --tight: padding of the video crop (a little air around the person)
TIGHT_SMOOTH_SEC = 0.3              # --tight: the per-frame 3D camera is smoothed this much
VIEW_PAD = 0.07                     # the trajectory box keeps this fraction of the frame free per side
ELEVATION_DEG = 12.0                # the virtual cameras look slightly down
FEET_JOINTS = (7, 8, 10, 11)        # ankles and toes: the ground sits under their lowest point
FOLLOW_SMOOTH_SEC = 0.3             # dolly camera: its target is the pelvis path smoothed this much
ZOOM_HOLD_SEC = 1.0                 # dolly zoom: the distance covers every frame within this much, then smoothed
FOLLOW_IF_FARTHER = 1.4             # dolly when the fixed camera would sit this much farther than the dolly
FOOT_ANCHOR = {7: 10, 8: 11}        # the foot's arrow starts halfway from the ankle to the toe joint


def unit(v: np.ndarray) -> np.ndarray:
    return v / max(np.linalg.norm(v), 1e-9)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """OpenGL camera pose (world from camera, camera looks down its -z)."""
    forward = unit(target - eye)
    right = unit(np.cross(forward, up))
    true_up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, true_up, -forward, eye
    return pose


def view_direction(heading: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Unit camera-to-scene direction: ``heading`` (horizontal) tilted down by ``ELEVATION_DEG``."""
    elevation = np.deg2rad(ELEVATION_DEG)
    return unit(np.cos(elevation) * heading - np.sin(elevation) * up)


def fit_distance(offsets: np.ndarray, direction: np.ndarray, up: np.ndarray, yfov: float,
                 pad: float = VIEW_PAD) -> float:
    """How far back along ``-direction`` a camera aimed at the origin must sit to keep every offset in frame.

    :param offsets: ``(M, 3)`` points relative to the camera's look-at target.
    :param pad: fraction of the frame kept free on each side.
    """
    radius = np.linalg.norm(offsets, axis=1).max()
    distance = radius / np.tan(yfov / 2) * (1 + pad)
    limit = np.tan(yfov / 2) * (1 - 2 * pad)
    for _ in range(20):                                                       # tighten from the sphere bound
        pose = look_at(-direction * distance, np.zeros(3), up)
        local = (offsets - pose[:3, 3]) @ pose[:3, :3]                        # camera frame, -z forward
        extent = (np.abs(local[:, :2]) / -local[:, 2:3]).max()               # square image, aspect 1
        if extent <= limit:
            break
        distance *= extent / limit * 1.02
    return float(distance)


def smooth_series(values: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing over the first axis with edge padding."""
    half = int(3 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, ((half, half),) + ((0, 0),) * (values.ndim - 1), mode="edge")
    return np.stack([np.convolve(padded[(slice(None),) + idx], kernel, mode="valid")
                     for idx in np.ndindex(values.shape[1:])], -1).reshape(values.shape)


def dolly_distances(points: np.ndarray, point_frame: np.ndarray, target: np.ndarray, direction: np.ndarray,
                    up: np.ndarray, yfov: float, fps: float) -> np.ndarray:
    """Per-frame dolly distance ``(N,)``: each frame's own fit, held over ``ZOOM_HOLD_SEC`` and smoothed.

    The running maximum over a window wider than the smoothing kernel keeps the smoothed distance at
    or above every frame's need; the final ``maximum`` guards the few places where it is not.
    """
    n_frames = len(target)
    need = np.zeros(n_frames)
    for f in np.unique(point_frame):
        rows = point_frame == f
        need[f] = fit_distance(points[rows] - target[f], direction, up, yfov)
    known = need > 0
    frames = np.arange(n_frames)
    need = np.interp(frames, frames[known], need[known])
    hold = int(ZOOM_HOLD_SEC * fps)
    padded = np.pad(need, hold, mode="edge")
    held = np.array([padded[f:f + 2 * hold + 1].max() for f in range(n_frames)])
    return np.maximum(smooth_series(held, hold / 3), need)


def tight_camera(points: np.ndarray, point_frame: np.ndarray, n_frames: int, direction: np.ndarray,
                 up: np.ndarray, yfov: float, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame targets ``(N, 3)`` and distances ``(N,)`` that frame each frame's own points tightly.

    The box centre of the frame's joints and arrow tips is the target; both are smoothed over
    ``TIGHT_SMOOTH_SEC`` (gaps interpolated), the distance then raised back to each frame's need.
    """
    target = np.full((n_frames, 3), np.nan)
    need = np.full(n_frames, np.nan)
    for f in np.unique(point_frame):
        pts = points[point_frame == f]
        target[f] = 0.5 * (pts.min(0) + pts.max(0))
        need[f] = fit_distance(pts - target[f], direction, up, yfov, TIGHT_PAD)
    known = np.isfinite(need)
    frames = np.arange(n_frames)
    target = np.stack([np.interp(frames, frames[known], target[known, k]) for k in range(3)], 1)
    need = np.interp(frames, frames[known], need[known])
    sigma = TIGHT_SMOOTH_SEC * fps
    target = smooth_series(target, sigma)
    distance = np.maximum(smooth_series(need, sigma), np.array([
        fit_distance(points[point_frame == f] - target[f], direction, up, yfov, TIGHT_PAD) if known[f] else 0.0
        for f in frames]))
    return target, distance


def camera_targets(dump: dict, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame look-at targets ``(N, 3)`` for the fixed camera (constant) and the dolly (smoothed pelvis).

    The dolly target is the covered frames' mean pelvis, linearly interpolated over gaps and
    Gaussian-smoothed over ``FOLLOW_SMOOTH_SEC``, so the camera translates slowly and never turns.
    """
    covered, pelvis = dump["covered"], dump["joints_world"][:, :, 0]         # (P, N), (P, N, 3)
    n_frames = covered.shape[1]
    known = covered.any(0)
    mean = np.stack([np.nanmean(np.where(covered[:, f, None], pelvis[:, f], np.nan), axis=0)
                     for f in range(n_frames)])                               # (N, 3), NaN where nobody
    frames = np.arange(n_frames)
    path = np.stack([np.interp(frames, frames[known], mean[known, k]) for k in range(3)], 1)
    follow = smooth_series(path, FOLLOW_SMOOTH_SEC * fps)
    lo, hi = path[known].min(0), path[known].max(0)
    fixed = np.tile(0.5 * (lo + hi), (n_frames, 1))
    return fixed, follow


def arrow_mesh(start: np.ndarray, force: np.ndarray) -> trimesh.Trimesh:
    """A cylinder + cone along ``force`` (world, body weight) from ``start``."""
    length = np.linalg.norm(force) * ARROW_M_PER_BW_3D
    head = min(ARROW_HEAD_L, 0.5 * length)
    shaft = trimesh.creation.cylinder(ARROW_SHAFT_R, length - head, sections=24)
    shaft.apply_translation([0, 0, (length - head) / 2])
    cone = trimesh.creation.cone(ARROW_HEAD_R, head, sections=24)
    cone.apply_translation([0, 0, length - head])
    arrow = trimesh.util.concatenate([shaft, cone])
    rotation = trimesh.geometry.align_vectors([0, 0, 1], unit(force))
    arrow.apply_transform(rotation)
    arrow.apply_translation(start)
    return arrow


def ground_meshes(points: np.ndarray, up: np.ndarray, height: float) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """A floor plane and a grid of thin rods on it, perpendicular to ``up`` at ``height`` along it."""
    right = unit(np.cross(up, [1.0, 0.0, 0.0] if abs(up[0]) < 0.9 else [0.0, 0.0, 1.0]))
    forward = np.cross(right, up)
    local = (points - up * height) @ np.stack([right, forward], 1)
    lo = np.floor(local.min(0) / GRID_CELL_M - GRID_MARGIN_CELLS) * GRID_CELL_M
    hi = np.ceil(local.max(0) / GRID_CELL_M + GRID_MARGIN_CELLS) * GRID_CELL_M
    origin = up * height

    def rod(a: np.ndarray, b: np.ndarray) -> trimesh.Trimesh:
        return trimesh.creation.cylinder(GRID_LINE_R, segment=np.stack([a, b]), sections=6)

    rods = [rod(origin + right * a + forward * lo[1], origin + right * a + forward * hi[1])
            for a in np.arange(lo[0], hi[0] + 1e-6, GRID_CELL_M)]
    rods += [rod(origin + right * lo[0] + forward * b, origin + right * hi[0] + forward * b)
             for b in np.arange(lo[1], hi[1] + 1e-6, GRID_CELL_M)]
    corners = np.stack([origin + right * a + forward * b for a, b in
                        ((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1]))])
    faces = [[0, 1, 2], [0, 2, 3]]
    if np.cross(corners[1] - corners[0], corners[2] - corners[0]) @ up < 0:
        faces = [[0, 2, 1], [0, 3, 2]]                                       # wind the floor to face up
    return trimesh.Trimesh(corners, faces, process=False), trimesh.util.concatenate(rods)


def body_meshes(body, betas: np.ndarray, q_cam: np.ndarray, covered: np.ndarray,
                world_from_cam: np.ndarray, device) -> tuple[list[np.ndarray | None], np.ndarray]:
    """Per-frame world vertices ``(V, 3)`` (None when uncovered) and the faces."""
    frames = np.flatnonzero(covered)
    verts: list[np.ndarray | None] = [None] * len(covered)
    for start in range(0, len(frames), 64):
        idx = frames[start:start + 64]
        b = torch.as_tensor(betas[idx], dtype=torch.float32, device=device)
        q = torch.as_tensor(q_cam[idx], dtype=torch.float32, device=device)
        shaped = body.with_shape(betas=b)
        v = shaped.vertices_from_data(shaped.fk(q)).cpu().numpy()             # (B, V, 3) camera frame
        w = world_from_cam[idx]
        v = np.einsum("bij,bvj->bvi", w[:, :3, :3], v) + w[:, None, :3, 3]
        for f, vf in zip(idx, v):
            verts[f] = vf
    return verts, body.structure.faces.cpu().numpy().astype(np.int32)


def anchor_points(joints_world: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """Arrow starts ``(P, N, J, 3)``: mid-foot for the ankles, the middle-finger base for the wrists."""
    out = []
    for j in joints:
        j = int(j)
        if j in FOOT_ANCHOR:
            out.append(0.5 * (joints_world[:, :, j] + joints_world[:, :, FOOT_ANCHOR[j]]))
        else:
            out.append(joints_world[:, :, ANCHOR_JOINT.get(j, j)])
    return np.stack(out, axis=2)


def light_pose(view: np.ndarray, yaw_deg: float, pitch_deg: float, centre: np.ndarray,
               distance: float) -> np.ndarray:
    """A light aimed at ``centre`` from ``distance`` away, turned by yaw / pitch off the view's line of sight."""
    right, true_up, back = view[:3, 0], view[:3, 1], view[:3, 2]
    yaw, pitch = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    towards_light = np.cos(pitch) * (np.cos(yaw) * back + np.sin(yaw) * (-right)) + np.sin(pitch) * true_up
    return look_at(centre + unit(towards_light) * distance, centre, true_up)


def studio_lights(view: np.ndarray, centre: np.ndarray) -> list[tuple]:
    """Key, fill and rim for one view as ``(light, pose)`` pairs (the key first)."""
    return [(pyrender.DirectionalLight(intensity=intensity), light_pose(view, yaw, pitch, centre, 5.0))
            for _, yaw, pitch, intensity in LIGHTS]


def planar_shadow(verts: np.ndarray, faces: np.ndarray, light_pose: np.ndarray, up: np.ndarray,
                  height: float) -> trimesh.Trimesh:
    """``verts`` projected onto the floor along the light's direction: the body's shadow as a flat mesh."""
    direction = -light_pose[:3, 2]                                            # a light shines down its -z
    along = direction @ up
    if along > -1e-3:                                                         # light at or below the floor
        direction = direction - up * (along + 0.3)
        along = direction @ up
    t = (verts @ up - height) / -along
    flat = verts + t[:, None] * direction + up * SHADOW_LIFT_M
    return trimesh.Trimesh(flat, faces, process=False)


def background(side: int) -> np.ndarray:
    """Vertical gradient ``(side, side, 3)`` BGR, light at the top."""
    t = np.linspace(0.0, 1.0, side, dtype=np.float32)[:, None, None]
    top, bottom = np.array(BACKGROUND_TOP, np.float32), np.array(BACKGROUND_BOTTOM, np.float32)
    return np.broadcast_to(top * (1 - t) + bottom * t, (side, side, 3)).astype(np.uint8)


def load_dump(tree: Path, pred_dir: str) -> dict:
    pred = np.load(tree / "predictions" / pred_dir / "smplx.npz", allow_pickle=True)
    forces = np.load(tree / "predictions" / pred_dir / "forces_sup.npz", allow_pickle=True)
    camera = np.load(tree / "geometry" / "transform.npz", allow_pickle=True)
    extrinsics = np.asarray(camera["extrinsics"], np.float64)
    identity = np.tile(np.eye(3, dtype=np.float32), (len(extrinsics), 1, 1))
    slots = contact_set(str(pred["contact_set"]))
    slot_parent = np.asarray([FOLD_JOINT.get(body22_parent(j), body22_parent(j))
                              for j in slots.parent_joint52], np.int64)
    joints = np.unique(slot_parent)
    down = np.asarray(pred["gravity_world"], np.float64).reshape(-1, 3)
    down = np.median(down[np.isfinite(down).all(1)], axis=0)
    return {"q_cam": np.asarray(pred["q_cam"], np.float32), "betas": np.asarray(pred["betas"], np.float32),
            "covered": np.asarray(pred["covered"], bool), "joints_world": np.asarray(pred["joints_world"], np.float32),
            "world_from_cam": np.linalg.inv(extrinsics),
            "force_world": np.stack([joint_forces_cam(forces["forces_world"][p], forces["contact_probs"][p],
                                                      slot_parent, joints, identity)
                                     for p in range(len(pred["covered"]))]),           # (P, N, J, 3)
            "anchor_world": anchor_points(np.asarray(pred["joints_world"], np.float32), joints),  # (P, N, J, 3)
            "up": (-unit(down)).astype(np.float64)}


def render_view(renderer: pyrender.OffscreenRenderer, scene: pyrender.Scene, pose: np.ndarray,
                lights: list[tuple], yfov: float) -> tuple[np.ndarray, np.ndarray]:
    """RGBA and depth of ``scene`` from ``pose`` under ``lights`` (added for the call, then removed)."""
    nodes = [scene.add(pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0), pose=pose)]
    nodes += [scene.add(light, pose=light_pose) for light, light_pose in lights]
    rgba, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    for node in nodes:
        scene.remove_node(node)
    return rgba, depth


def draw_overlay(img: np.ndarray, people: list[dict], f: int, intrinsics: np.ndarray) -> None:
    """The ``results/`` overlay on the full frame ``img``: white skeleton, yellow arrows, no fog."""
    for person in people:
        if not person["covered"][f]:
            continue
        draw_skeleton(img, person["body"][f], intrinsics, 1.0)
        force, anchors = person["force"][f], person["anchors"][f]
        on = np.isfinite(force).all(-1) & (np.linalg.norm(np.nan_to_num(force), axis=-1) >= ARROW_MIN_BW)
        if on.any():
            draw_arrows(img, anchors[on], anchors[on] + force[on] * ARROW_M_PER_BW, intrinsics, 1.0,
                        pixels_per_metre(person["body"][f], intrinsics, 1.0), ARROW_PALETTES["yellow"])


def render_clip(tree: Path, pred_dir: str, out: Path, side: int, side_deg: float, device,
                tight: bool, views: tuple[str, ...]) -> None:
    people, clip = load_people(tree, pred_dir)
    dump = load_dump(tree, pred_dir)
    up = dump["up"]
    n_people, n_frames = dump["covered"].shape
    body = load_body(device)
    verts = [body_meshes(body, dump["betas"][p], dump["q_cam"][p], dump["covered"][p],
                         dump["world_from_cam"], device)[0] for p in range(n_people)]
    faces = body.structure.faces.cpu().numpy().astype(np.int32)

    # Everything the fixed cameras must keep in frame: joints and arrow tips of every covered frame.
    covered = dump["covered"]
    joints = dump["joints_world"][covered]                                    # (M, 52, 3)
    tips = dump["anchor_world"] + np.nan_to_num(dump["force_world"]) * ARROW_M_PER_BW_3D
    frame_ids = np.broadcast_to(np.arange(n_frames)[None, :], covered.shape)[covered]        # (M,)
    points = np.concatenate([joints.reshape(-1, 3), tips[covered].reshape(-1, 3)])
    point_frame = np.concatenate([np.repeat(frame_ids, joints.shape[1]), np.repeat(frame_ids, tips.shape[2])])
    finite = np.isfinite(points).all(1)
    points, point_frame = points[finite], point_frame[finite]
    yfov = np.deg2rad(YFOV_DEG)
    cam0 = dump["world_from_cam"][0]
    heading = cam0[:3, :3] @ np.array([0.0, 0.0, 1.0])                       # OpenCV camera looks down +z
    heading = unit(heading - up * (heading @ up))
    turned = unit(np.cos(np.deg2rad(side_deg)) * heading + np.sin(np.deg2rad(side_deg)) * np.cross(up, heading))
    directions = {"camera": view_direction(heading, up), "side": view_direction(turned, up)}
    directions = {name: directions[name] for name in views}
    fixed_target, follow_target = camera_targets(dump, clip["fps"])
    # Fixed camera when the whole path fits at body scale, else a dolly that keeps that scale and slides.
    distances, targets, modes = {}, {}, {}                       # distances: (N,) per view
    for name, direction in directions.items():
        if tight:
            targets[name], distances[name] = tight_camera(points, point_frame, n_frames, direction, up, yfov,
                                                          clip["fps"])
            modes[name] = "tight"
            continue
        fixed_d = fit_distance(points - fixed_target[point_frame], direction, up, yfov)
        follow_d = dolly_distances(points, point_frame, follow_target, direction, up, yfov, clip["fps"])
        if fixed_d > FOLLOW_IF_FARTHER * follow_d.max():
            distances[name], targets[name], modes[name] = follow_d, follow_target, "dolly"
        else:
            distances[name], targets[name], modes[name] = np.full(n_frames, fixed_d), fixed_target, "fixed"
    poses = {name: [look_at(targets[name][f] - directions[name] * distances[name][f], targets[name][f], up)
                    for f in range(n_frames)] for name in directions}
    ground_height = float((joints[:, list(FEET_JOINTS)] @ up).min()) - 0.01

    if tight:
        windows = crop_windows(people, clip["intrinsics"], n_frames, clip["width"], clip["height"], clip["fps"],
                               pad=TIGHT_VIDEO_PAD, tips=False)
        folders = {name: out / name / tree.name for name in ("video", *views, "united")}
        for folder in folders.values():
            folder.mkdir(parents=True, exist_ok=True)
    else:
        windows = crop_windows(people, clip["intrinsics"], n_frames, clip["width"], clip["height"], clip["fps"])
    body_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[*(c / 255 for c in BODY_RGB), 1.0],
                                                       metallicFactor=0.05, roughnessFactor=0.55)
    arrow_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[*(c / 255 for c in ARROW_RGB), 1.0],
                                                        emissiveFactor=[0.30, 0.20, 0.0],
                                                        metallicFactor=0.0, roughnessFactor=0.45)
    grid_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[*(c / 255 for c in GRID_RGB), 1.0],
                                                       metallicFactor=0.0, roughnessFactor=1.0)
    floor_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[*(c / 255 for c in GROUND_RGB), 1.0],
                                                        metallicFactor=0.0, roughnessFactor=1.0)
    plane, rods = ground_meshes(points, up, ground_height)
    ground = [pyrender.Mesh.from_trimesh(plane, material=floor_material, smooth=False),
              pyrender.Mesh.from_trimesh(rods, material=grid_material, smooth=False)]
    lights = {name: [studio_lights(poses[name][f], targets[name][f]) for f in range(n_frames)] for name in poses}
    shadow_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.0, 0.0, 0.0, 1.0], metallicFactor=0.0,
                                                         roughnessFactor=1.0, emissiveFactor=list(SHADOW_RGB),
                                                         doubleSided=True)             # unlit: one flat tone
    local_radius = np.linalg.norm(points - follow_target[point_frame], axis=1).max()
    fog_start = {name: distances[name] + local_radius for name in poses}                     # (N,) per view
    big = side * SUPERSAMPLE
    backdrop = background(big)
    renderer = pyrender.OffscreenRenderer(big, big)
    writer = None if tight else open_writer(out / f"{tree.name}.mp4", clip["fps"], ((1 + len(views)) * side, side))
    cap = cv2.VideoCapture(str(clip["video"]))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {clip['video']}")
    try:
        for f in range(n_frames):
            ok, img = cap.read()
            if not ok:
                raise ValueError(f"{clip['video']}: decoded {f} frames, the dump has {n_frames}")
            if not tight:
                draw_overlay(img, people, f, clip["intrinsics"][f])
            panels = [crop_frame(img, windows[f], side)]
            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[AMBIENT] * 3)
            for mesh in ground:
                scene.add(mesh)
            arrows = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[AMBIENT] * 3)
            for p in range(n_people):
                if verts[p][f] is None:
                    continue
                scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(verts[p][f], faces, process=False),
                                                     material=body_material, smooth=True))
                for anchor, force in zip(dump["anchor_world"][p, f], dump["force_world"][p, f]):
                    if np.isfinite(force).all() and np.linalg.norm(force) >= ARROW_MIN_BW:
                        mesh = pyrender.Mesh.from_trimesh(arrow_mesh(anchor, force), material=arrow_material,
                                                          smooth=False)
                        scene.add(mesh)
                        arrows.add(mesh)
            for name in views:
                shadows = [scene.add(pyrender.Mesh.from_trimesh(
                    planar_shadow(verts[p][f], faces, lights[name][f][0][1], up, ground_height),
                    material=shadow_material, smooth=False)) for p in range(n_people) if verts[p][f] is not None]
                rgba, depth = render_view(renderer, scene, poses[name][f], lights[name][f], yfov)
                for node in shadows:
                    scene.remove_node(node)
                alpha = rgba[..., 3:4].astype(np.float32) / 255
                colour = cv2.cvtColor(np.ascontiguousarray(rgba[..., :3]), cv2.COLOR_RGB2BGR).astype(np.float32)
                fog = np.clip((depth - fog_start[name][f]) / FOG_M, 0.0, 1.0)[..., None] * (depth > 0)[..., None]
                alpha = alpha * (1 - fog)
                panel = colour * alpha + backdrop * (1 - alpha)
                if len(arrows.mesh_nodes):                        # ghost the arrow parts hidden inside the body
                    arrow_rgba, arrow_depth = render_view(renderer, arrows, poses[name][f], lights[name][f], yfov)
                    hidden = (arrow_rgba[..., 3] > 0) & (depth > 0) & (arrow_depth > depth + 1e-3)
                    arrow_bgr = cv2.cvtColor(np.ascontiguousarray(arrow_rgba[..., :3]), cv2.COLOR_RGB2BGR)
                    panel[hidden] = (1 - GHOST_ALPHA) * panel[hidden] + GHOST_ALPHA * arrow_bgr[hidden]
                panel = panel.astype(np.uint8)
                panels.append(cv2.resize(panel, (side, side), interpolation=cv2.INTER_AREA))
            if tight:
                for folder, panel in zip(folders.values(), panels + [np.concatenate(panels, axis=1)]):
                    cv2.imwrite(str(folder / f"{f:06d}.jpg"), panel, IMWRITE_PARAMS["jpg"])
            else:
                writer.write(np.concatenate(panels, axis=1))
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        renderer.delete()
    print(f"  {tree.name}: {n_frames} frames, {n_people} people, "
          + ", ".join(f"{name} {modes[name]} {distances[name].min():.1f}-{distances[name].max():.1f} m"
                      for name in poses)
          + (f" -> {out}/{{video,camera,side,united}}/{tree.name}/" if tight else f" -> {out / tree.name}.mp4"),
          flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--pred-dir", default="joint_frames35")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stems", nargs="*", default=None, help="default: every clip with a dump")
    parser.add_argument("--side", type=int, default=768, help="pixels per square panel")
    parser.add_argument("--side-deg", type=float, default=90.0, help="side view: turn about the vertical")
    parser.add_argument("--views", default="camera,side",
                        help="which 3D views to render, in order: camera and/or side")
    parser.add_argument("--tight", action="store_true",
                        help="frames instead of an mp4: <out>/{video,camera,side,united}/<stem>/<frame>.jpg, every "
                             "panel a tight square around the person (TIGHT_PAD), no captions")
    args = parser.parse_args()
    views = tuple(args.views.split(","))
    if not views or any(v not in ("camera", "side") for v in views):
        raise SystemExit(f"--views must list camera and/or side, got {args.views!r}")
    stems = args.stems or sorted(d.name for d in args.out_root.iterdir()
                                 if (d / "predictions" / args.pred_dir / "smplx.npz").is_file())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(stems, start=1):
        print(f"[{index}/{len(stems)}] {stem}", flush=True)
        render_clip(args.out_root / stem, args.pred_dir, args.out, args.side, args.side_deg, device, args.tight,
                    views)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
