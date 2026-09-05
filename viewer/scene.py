"""``SceneView`` — the viser nodes of one scene and their per-frame update.

Every node lives under one ``/clip`` root frame and is built in the metric WORLD
frame once. The two viewing regimes differ only in that root's pose:

* ``world`` — the root is the identity; the per-frame camera frustums, their
  path and the gravity arrow show, and the scene is oriented with the fitted
  gravity down.
* ``camera`` — the root is posed at ``cam_from_world(f)`` every frame, so the
  whole world is re-expressed in the CURRENT camera's OpenCV axes: the camera
  is a fixed frustum at the origin and the bodies are seen exactly as the
  models output them (no lifting), the GT lifted INTO the camera the way the
  losses see it.

Meshes are viser skinned meshes (LBS in the browser): one upload per person and
source, 52 bone poses per frame. Skeletons are the 22 body joints (a point cloud)
plus their bones (line segments), one node pair per person and source.

Contacts and forces (predicted from the run's dump, GT from the corpus) sit on
the body they belong to: six markers at the group joints coloured by
probability / label, and one line per group from the joint along the
world-frame force (``force_scale`` metres per body weight).
"""
from __future__ import annotations

import numpy as np

from .bodies import NUM_BODY_JOINTS
from .loading import SOURCES, SceneData

#: RGB per body source — the palette of ``scripts/render_smplx_video.py``.
COLORS = {"predicted": (235, 110, 110), "gt": (110, 205, 110), "frozen": (120, 170, 235)}
_CAM_MAX_FRUSTUMS = 24
_CAM_COLOR = (255, 140, 0)
_GRAVITY_COLOR = (235, 40, 40)
_JOINT_RADIUS, _BONE_WIDTH = 0.03, 4.0
#: Body-22 joint of each kindyn group (LH, RH wrists; LF, RF toes; LA, RA heels).
GROUP_JOINTS = (20, 21, 10, 11, 7, 8)
_CONTACT_SIZE, _FORCE_WIDTH = 0.06, 6.0
_CONTACT_ON, _CONTACT_OFF, _CONTACT_UNKNOWN = (40, 230, 60), (120, 120, 120), (55, 55, 55)


def contact_colors(values: np.ndarray) -> np.ndarray:
    """``(6,)`` probabilities / labels (NaN = unknown) -> ``(6, 3)`` uint8 grey→green."""
    v = np.asarray(values, np.float64)
    known = np.isfinite(v)
    t = np.clip(np.where(known, v, 0.0), 0.0, 1.0)[:, None]
    rgb = (1.0 - t) * np.array(_CONTACT_OFF) + t * np.array(_CONTACT_ON)
    rgb[~known] = _CONTACT_UNKNOWN
    return rgb.astype(np.uint8)


def _wxyz_from_matrix(rot: np.ndarray) -> np.ndarray:
    """Rotation matrices ``(..., 3, 3)`` -> ``wxyz`` unit quaternions ``(..., 4)``."""
    import viser.transforms as vt

    flat = np.asarray(rot, np.float64).reshape(-1, 3, 3)
    return np.stack([vt.SO3.from_matrix(r).wxyz for r in flat]).reshape(rot.shape[:-2] + (4,))


def camera_pose(extrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cam_from_world ``[R | t]`` -> the camera's world pose ``(position, wxyz)``."""
    rot, t = extrinsic[:3, :3], extrinsic[:3, 3]
    return (-rot.T @ t).astype(np.float64), _wxyz_from_matrix(rot.T)


def onboard_target(extrinsic: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Viewer camera ``(position, look_at, up)`` riding the source camera.

    ``None`` is the camera regime: the camera IS the origin looking down +z.
    """
    if extrinsic is None:
        return np.zeros(3), np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0])
    rot, t = extrinsic[:3, :3], extrinsic[:3, 3]
    centre = -rot.T @ t
    return centre, centre + rot.T @ np.array([0.0, 0.0, 1.0]), rot.T @ np.array([0.0, -1.0, 0.0])


class SceneView:
    """Build and own the nodes of one scene; :meth:`dispose` removes them all."""

    def __init__(self, server, data: SceneData, layers: dict, *, regime: str,
                 opacity: float, point_size: float, camera_scale: float,
                 force_scale: float = 0.3) -> None:
        self.server = server
        self.data = data
        self.layers = layers
        self.regime = regime
        self.opacity = float(opacity)
        self.camera_scale = float(camera_scale)
        self.force_scale = float(force_scale)
        self.n_frames = data.n_frames
        self._frame = -1
        self._visible: dict[int, bool] = {}          # id(handle) -> last pushed visibility
        ss = server.scene
        self.root = ss.add_frame("/clip", show_axes=False)

        # Per-frame root pose of the camera regime: displayed = cam_from_world @ world.
        self.cam_wxyz = _wxyz_from_matrix(data.extrinsics[:, :3, :3])
        self.cam_pos = data.extrinsics[:, :3, 3].astype(np.float64)
        centres = -np.einsum("nji,nj->ni", data.extrinsics[:, :3, :3],
                             data.extrinsics[:, :3, 3]).astype(np.float64)
        self.camera_centres = centres
        self.focus_world = self._focus()

        # -- scene cloud --
        self.scene_node = ss.add_point_cloud(
            "/clip/scene", data.scene_points, data.scene_colors, point_size=point_size,
            point_shape="circle", visible=False)

        # -- world-regime cameras: strided frustums, the centre path, a cursor --
        aspect = data.width / data.height
        self.frustums = []
        step = max(1, int(np.ceil(data.n_frames / _CAM_MAX_FRUSTUMS)))
        for k in range(0, data.n_frames, step):
            pos, wxyz = camera_pose(data.extrinsics[k])
            self.frustums.append(ss.add_camera_frustum(
                f"/clip/cameras/cam_{k:04d}", fov=float(data.fov_y[k]), aspect=aspect,
                scale=self.camera_scale, color=_CAM_COLOR, line_width=1.5,
                wxyz=wxyz, position=pos, visible=False))
        self.path = (ss.add_spline_catmull_rom(
            "/clip/cameras/path", centres.astype(np.float32), color=_CAM_COLOR,
            line_width=2.0, visible=False) if data.n_frames > 1 else None)
        self.cursor = ss.add_icosphere(
            "/clip/cameras/cursor", radius=max(self.camera_scale * 0.4, 0.02),
            color=(235, 40, 40), position=tuple(centres[0].tolist()), visible=False)
        # -- camera-regime camera: a fixed frustum at the origin (outside /clip) --
        self.origin_frustum = ss.add_camera_frustum(
            "/cam", fov=float(data.fov_y[0]), aspect=aspect, scale=self.camera_scale,
            color=_CAM_COLOR, line_width=2.0, visible=False)
        # -- gravity: a 1 m arrow from the body focus straight down (world regime) --
        seg = np.stack([self.focus_world, self.focus_world + data.gravity], 0)[None]
        self.gravity_node = ss.add_line_segments(
            "/clip/gravity", seg.astype(np.float32), colors=_GRAVITY_COLOR, line_width=4.0,
            visible=False)

        # -- bodies: skinned mesh + skeleton per (source, person) --
        self.bodies: dict[str, list] = {}
        for name in SOURCES:
            src = data.sources[name]
            entries = []
            for person in src.people:
                if person is None:
                    entries.append(None)
                    continue
                ident = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32),
                                (person.j_rest.shape[0], 1))
                mesh = ss.add_mesh_skinned(
                    f"/clip/body/{name}/p{person.oid:02d}", person.v_shaped, src.faces,
                    bone_wxyzs=ident, bone_positions=person.j_rest,
                    skin_weights=person.weights, color=COLORS[name],
                    opacity=self.opacity, visible=False)
                joints = ss.add_point_cloud(
                    f"/clip/skel/{name}/p{person.oid:02d}/joints",
                    person.j_rest[:NUM_BODY_JOINTS], COLORS[name], point_size=_JOINT_RADIUS,
                    point_shape="circle", visible=False)
                parents = src.parents[:NUM_BODY_JOINTS]
                child = np.flatnonzero(parents >= 0)
                bones = ss.add_line_segments(
                    f"/clip/skel/{name}/p{person.oid:02d}/bones",
                    np.stack([person.j_rest[parents[child]], person.j_rest[child]], 1),
                    colors=COLORS[name], line_width=_BONE_WIDTH, visible=False)
                entries.append({"person": person, "mesh": mesh, "joints": joints,
                                "bones": bones, "child": child, "parent": parents[child]})
            self.bodies[name] = entries

        # -- contacts + forces on the predicted and GT bodies --
        self.overlays: dict[str, list] = {}
        zeros6 = np.zeros((6, 3), np.float32)
        for name in ("predicted", "gt"):
            if name not in data.contacts and name not in data.forces:
                continue
            entries = []
            for pidx, person in enumerate(data.sources[name].people):
                if person is None:
                    continue
                markers = ss.add_point_cloud(
                    f"/clip/contact/{name}/p{person.oid:02d}", zeros6,
                    np.tile(np.array(_CONTACT_OFF, np.uint8), (6, 1)), point_size=_CONTACT_SIZE,
                    point_shape="circle", visible=False)
                arrows = ss.add_line_segments(
                    f"/clip/force/{name}/p{person.oid:02d}", np.zeros((6, 2, 3), np.float32),
                    colors=COLORS[name], line_width=_FORCE_WIDTH, visible=False)
                entries.append({"person": person, "pidx": pidx, "markers": markers, "arrows": arrows})
            self.overlays[name] = entries
        self.set_regime(regime)

    # -- framing --
    def _focus(self) -> np.ndarray:
        """Mean pelvis over every source's valid frames (world), the orbit centre."""
        roots = []
        for src in self.data.sources.values():
            for person in src.people:
                if person is None:
                    continue
                pelvis = person.bone_pos[person.valid, 0]
                if len(pelvis):
                    roots.append(pelvis.mean(0))
        if roots:
            return np.mean(roots, axis=0).astype(np.float64)
        return self.camera_centres.mean(0) + np.array([0.0, 0.0, 3.0])

    def home_view(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """``(position, look_at)`` of the opening view in the current regime."""
        if self.regime == "camera":
            ext = self.data.extrinsics[frame]
            focus = ext[:3, :3] @ self.focus_world + ext[:3, 3]
            dist = float(np.linalg.norm(focus))
            back = -focus / dist if dist > 1e-3 else np.array([0.0, 0.0, -1.0])
            return focus + back * (dist + 1.0), focus            # 1 m behind the camera origin
        d = self.camera_centres.mean(0) - self.focus_world
        dist = float(np.linalg.norm(d))
        if dist < 1e-3:
            d, dist = np.array([0.0, 0.0, -1.0]), 1.0
        return self.focus_world + (d / dist) * max(dist * 1.2, 2.0), self.focus_world

    def up_direction(self):
        return (0.0, -1.0, 0.0) if self.regime == "camera" else tuple(
            float(x) for x in -self.data.gravity)

    # -- regime + layers --
    def set_regime(self, regime: str) -> None:
        if regime not in ("camera", "world"):
            raise ValueError(f"regime must be camera | world, got {regime!r}")
        self.regime = regime
        self.server.scene.set_up_direction(self.up_direction())
        if regime == "world":
            self.root.wxyz, self.root.position = (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        self.apply_static_layers()
        if self._frame >= 0:
            self.apply_frame(self._frame)

    def _show(self, handle, on: bool) -> None:
        key = id(handle)
        if self._visible.get(key) is not on:
            handle.visible = on
            self._visible[key] = on

    def apply_static_layers(self) -> None:
        world = self.regime == "world"
        with self.server.atomic():
            self._show(self.scene_node, self.layers["scene"])
            for h in self.frustums:
                self._show(h, world and self.layers["cameras"])
            if self.path is not None:
                self._show(self.path, world and self.layers["cameras"])
            self._show(self.cursor, world and self.layers["cameras"])
            self._show(self.origin_frustum, (not world) and self.layers["cameras"])
            self._show(self.gravity_node, world and self.layers["gravity"])

    def set_opacity(self, alpha: float) -> None:
        self.opacity = float(alpha)
        with self.server.atomic():
            for entries in self.bodies.values():
                for e in entries:
                    if e is not None:
                        e["mesh"].opacity = self.opacity

    def set_point_size(self, size: float) -> None:
        self.scene_node.point_size = float(size)

    def set_force_scale(self, scale: float) -> None:
        self.force_scale = float(scale)
        if self._frame >= 0:
            self.apply_frame(self._frame)

    def set_camera_scale(self, scale: float) -> None:
        self.camera_scale = float(scale)
        with self.server.atomic():
            for h in self.frustums + [self.origin_frustum]:
                h.scale = self.camera_scale

    # -- per frame --
    def apply_frame(self, f: int) -> None:
        f = int(np.clip(f, 0, max(self.n_frames - 1, 0)))
        with self.server.atomic():
            if self.regime == "camera":
                self.root.wxyz, self.root.position = self.cam_wxyz[f], self.cam_pos[f]
                self.origin_frustum.fov = float(self.data.fov_y[f])
            else:
                self.cursor.position = tuple(float(x) for x in self.camera_centres[f])
            for name, entries in self.bodies.items():
                mesh_on, skel_on = self.layers[f"mesh_{name}"], self.layers[f"skel_{name}"]
                for e in entries:
                    if e is None:
                        continue
                    person = e["person"]
                    live = bool(person.valid[f])
                    self._show(e["mesh"], mesh_on and live)
                    if mesh_on and live:
                        wxyz, pos = person.bone_wxyz[f], person.bone_pos[f]
                        for j, bone in enumerate(e["mesh"].bones):
                            bone.wxyz, bone.position = wxyz[j], pos[j]
                    self._show(e["joints"], skel_on and live)
                    self._show(e["bones"], skel_on and live)
                    if skel_on and live:
                        pos = person.bone_pos[f]
                        e["joints"].points = pos[:NUM_BODY_JOINTS]
                        e["bones"].points = np.stack([pos[e["parent"]], pos[e["child"]]], 1)
            for name, entries in self.overlays.items():
                contacts = self.data.contacts.get(name)
                forces = self.data.forces.get(name)
                c_on = bool(self.layers.get(f"contact_{name}", False)) and contacts is not None
                f_on = bool(self.layers.get(f"force_{name}", False)) and forces is not None
                for e in entries:
                    person, pidx = e["person"], e["pidx"]
                    live = bool(person.valid[f])
                    joints = person.bone_pos[f][list(GROUP_JOINTS)] if live else None
                    show_c = c_on and live and bool(np.isfinite(contacts[pidx, f]).any())
                    self._show(e["markers"], show_c)
                    if show_c:
                        e["markers"].points = joints.astype(np.float32)
                        e["markers"].colors = contact_colors(contacts[pidx, f])
                    show_f = f_on and live and bool(np.isfinite(forces[pidx, f]).any())
                    self._show(e["arrows"], show_f)
                    if show_f:
                        vec = np.nan_to_num(forces[pidx, f]) * self.force_scale
                        e["arrows"].points = np.stack([joints, joints + vec], 1).astype(np.float32)
        self._frame = f

    def dispose(self) -> None:
        """Remove the whole subtree (a handle's ``remove`` drops only that node
        server-side; the children would be replayed to every new client)."""
        self.server.scene.remove_by_name("/cam")
        self.server.scene.remove_by_name("/clip")
