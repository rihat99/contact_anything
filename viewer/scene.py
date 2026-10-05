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
source, 52 bone poses per frame. Skeletons follow the BetterVideoReconstruction
viewer's look (``tools/viewer/scene.py``): all 52 joints as icospheres — the 30
finger joints at 30 % size — and one cylinder per bone, placed with the
shortest-arc rotation of +z onto the bone.

Contacts and forces (predicted from the run's dump, GT from the corpus) sit on
the body they belong to: one sphere per contact frame at its point on the skin,
red in contact and green free, a thin connector back to its parent joint, and
one red arrow per frame along the world-frame force (``force_scale`` metres per
body weight).
"""
from __future__ import annotations

import numpy as np

from .bodies import NUM_BODY_JOINTS, NUM_JOINTS
from .loading import SceneData

#: RGB per body source: the predicted body in BVR's light blue (its first per-person colour),
#: the GT green, the frozen body lilac (BVR's fourth) so it stays apart from the predicted one.
COLORS = {"predicted": (166, 189, 219), "gt": (110, 205, 110), "frozen": (204, 166, 224)}
_CAM_MAX_FRUSTUMS = 24
_CAM_COLOR = (255, 140, 0)
_GRAVITY_COLOR = (235, 40, 40)
_GROUND_COLOR = (150, 150, 150)
_GROUND_HALF_M = 10.0
#: BVR's skeleton look: joint sphere radius, bone radius (m) and the finger scale.
_SKEL_R, _SKEL_RB, _SKEL_FINGER_SCALE = 0.028, 0.012, 0.30
#: The predicted body's skeleton grey (BVR ``_SKEL_RGB``); GT and frozen keep ``COLORS``.
_SKEL_RGB = (120, 128, 140)
#: Contact frames: sphere radius, the connector's radius fraction and its pale colour.
_FRAME_R, _FRAME_CONN_SCALE, _FRAME_CONN_RGB = 0.012, 0.3, (150, 190, 160)
#: Force arrows (BVR ``_DYN_FORCE_RGB`` and the plate-arrow geometry).
_FORCE_RGB, _FORCE_SHAFT, _FORCE_HEAD_R, _FORCE_HEAD_L = (235, 45, 45), 0.012, 0.028, 0.05
#: The two force displays: each slot's own force at its point on the skin, or a joint's
#: slots summed at the joint.
FORCE_MODES = ("per-frame", "per-joint")
#: Forces below this (body weights) draw no arrow.
FORCE_MIN_BW = 0.05
_CONTACT_ON, _CONTACT_OFF, _CONTACT_UNKNOWN = (235, 45, 45), (60, 200, 90), (120, 120, 120)


def contact_colors(values: np.ndarray) -> np.ndarray:
    """``(K,)`` probabilities / labels -> ``(K, 3)`` uint8: red in contact (≥ 0.5),
    green free, grey where the value is NaN (unknown)."""
    v = np.asarray(values, np.float64)
    known = np.isfinite(v)
    rgb = np.where((known & (v >= 0.5))[:, None], np.array(_CONTACT_ON), np.array(_CONTACT_OFF))
    rgb[~known] = _CONTACT_UNKNOWN
    return rgb.astype(np.uint8)


def _wxyz_shortest_arc_z(d: np.ndarray) -> tuple[float, float, float, float]:
    """Shortest-arc quaternion (``wxyz``) rotating a unit cylinder's +z axis onto ``d``."""
    d = np.asarray(d, np.float64)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    d = d / n
    cos = float(d[2])
    if cos > 0.999999:
        return (1.0, 0.0, 0.0, 0.0)
    if cos < -0.999999:
        return (0.0, 1.0, 0.0, 0.0)                       # antiparallel: 180° about x
    v = np.array([-d[1], d[0], 0.0])                      # z × d
    s = np.sqrt((1.0 + cos) * 2.0)
    q = np.array([s * 0.5, v[0] / s, v[1] / s, v[2] / s])
    q = q / np.linalg.norm(q)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _unit_cylinder(color: tuple):
    """A unit +z trimesh cylinder (radius 1, height 1) in ``color``, scaled per frame
    into the skeleton's bones and the contact frames' connectors."""
    import trimesh

    cyl = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=10)
    cyl.visual.vertex_colors = np.array([*color, 255], np.uint8)
    return cyl


def _wxyz_from_matrix(rot: np.ndarray) -> np.ndarray:
    """Rotation matrices ``(..., 3, 3)`` -> ``wxyz`` unit quaternions ``(..., 4)``."""
    import viser.transforms as vt

    flat = np.asarray(rot, np.float64).reshape(-1, 3, 3)
    return np.stack([vt.SO3.from_matrix(r).wxyz for r in flat]).reshape(rot.shape[:-2] + (4,))


def _wxyz_z_to(normal: np.ndarray) -> np.ndarray:
    """The rotation taking +z onto the unit ``normal`` (``wxyz``)."""
    z = np.array([0.0, 0.0, 1.0])
    axis, cos = np.cross(z, normal), float(np.dot(z, normal))
    if np.linalg.norm(axis) < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0]) if cos > 0 else np.array([0.0, 1.0, 0.0, 0.0])
    axis = axis / np.linalg.norm(axis)
    half = 0.5 * np.arccos(np.clip(cos, -1.0, 1.0))
    return np.concatenate([[np.cos(half)], np.sin(half) * axis])


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
    """Build and own the nodes of one scene under the persistent ``root`` frame.

    :meth:`dispose` removes every node except the skinned body meshes: removing a
    skinned mesh crashes the viser 1.1.1 frontend (the page goes white), while
    re-adding one under the same name supersedes it cleanly. So the meshes are named
    by source and person INDEX, the next scene re-adds them by name, and a mesh the
    next scene does not need stays hidden.
    """

    def __init__(self, server, root, data: SceneData, layers: dict, *, regime: str,
                 opacity: float, point_size: float, camera_scale: float,
                 force_scale: float = 0.3, force_mode: str = "per-frame") -> None:
        self.server = server
        self.data = data
        self.layers = layers
        self.regime = regime
        self.force_mode = force_mode
        self.opacity = float(opacity)
        self.camera_scale = float(camera_scale)
        self.force_scale = float(force_scale)
        self.n_frames = data.n_frames
        self._frame = -1
        self._visible: dict[int, bool] = {}          # id(handle) -> last pushed visibility
        ss = server.scene
        self.root = root

        # Per-frame root pose of the camera regime: displayed = cam_from_world @ world.
        self.cam_wxyz = _wxyz_from_matrix(data.extrinsics[:, :3, :3])
        self.cam_pos = data.extrinsics[:, :3, 3].astype(np.float64)
        centres = -np.einsum("nji,nj->ni", data.extrinsics[:, :3, :3],
                             data.extrinsics[:, :3, 3]).astype(np.float64)
        self.camera_centres = centres
        self.focus_world = self._focus()

        # -- scene cloud --
        self.scene_node = (ss.add_point_cloud(
            "/clip/scene", data.scene_points, data.scene_colors, point_size=point_size,
            point_shape="circle", visible=False) if len(data.scene_points) else None)

        # -- GT ground plane: a grid with a translucent fill, world regime --
        self.ground_node = None
        if data.ground is not None:
            side = 2.0 * _GROUND_HALF_M
            self.ground_node = ss.add_grid(
                "/clip/ground", width=side, height=side, plane="xy", cell_size=0.5,
                section_size=1.0, plane_color=_GROUND_COLOR, plane_opacity=0.25,
                wxyz=_wxyz_z_to(data.ground["normal"]), position=data.ground["centroid"],
                visible=False)

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

        # -- bodies: skinned mesh + 52-joint sphere/cylinder skeleton per (source, person) --
        bone_cyl = {name: _unit_cylinder(_SKEL_RGB if name == "predicted" else COLORS[name])
                    for name in data.sources}
        finger = np.arange(NUM_JOINTS) >= NUM_BODY_JOINTS
        self.bodies: dict[str, list] = {}
        for name, src in data.sources.items():
            entries = []
            for pidx, person in enumerate(src.people):
                if person is None:
                    entries.append(None)
                    continue
                ident = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32),
                                (person.j_rest.shape[0], 1))
                mesh = ss.add_mesh_skinned(
                    f"/clip/body/{name}/p{pidx}", person.v_shaped, src.faces,
                    bone_wxyzs=ident, bone_positions=person.j_rest,
                    skin_weights=person.weights, color=COLORS[name],
                    opacity=self.opacity, visible=False)
                skel = f"/clip/skel/{name}/p{person.oid:02d}"
                colour = _SKEL_RGB if name == "predicted" else COLORS[name]
                joints = [ss.add_icosphere(
                    f"{skel}/joint_{j:02d}",
                    radius=_SKEL_R * (_SKEL_FINGER_SCALE if finger[j] else 1.0),
                    color=colour, visible=False) for j in range(NUM_JOINTS)]
                bones = [(ss.add_mesh_trimesh(f"{skel}/bone_{j:02d}", bone_cyl[name],
                                              visible=False),
                          j, int(src.parents[j]),
                          _SKEL_RB * (_SKEL_FINGER_SCALE if finger[j] else 1.0))
                         for j in range(NUM_JOINTS) if src.parents[j] >= 0]
                entries.append({"person": person, "mesh": mesh, "joints": joints,
                                "bones": bones})
            self.bodies[name] = entries

        # -- contact frames + forces on the predicted and GT bodies: one sphere per slot
        #    at its point on the skin with a connector back to its parent joint (the
        #    per-joint display drops the connectors: the marker IS the joint), and one
        #    red arrows node per person, re-added every frame --
        self.overlays: dict[str, list] = {}
        link_cyl = _unit_cylinder(_FRAME_CONN_RGB)
        counts = {"per-frame": len(data.slot_names), "per-joint": len(data.slot_joints)}
        for name in ("predicted", "gt"):
            if name not in data.contacts and name not in data.forces:
                continue
            entries = []
            for pidx, person in enumerate(data.sources[name].people):
                if person is None:
                    continue
                nodes = {}
                for mode, count in counts.items():
                    path = f"/clip/contact/{name}/{mode}/p{person.oid:02d}"
                    nodes[mode] = {
                        "spheres": [ss.add_icosphere(f"{path}/frame_{i:02d}", radius=_FRAME_R,
                                                     color=_CONTACT_OFF, visible=False)
                                    for i in range(count)],
                        "links": ([ss.add_mesh_trimesh(f"{path}/link_{i:02d}", link_cyl,
                                                       visible=False) for i in range(count)]
                                  if mode == "per-frame" else [])}
                entries.append({"person": person, "pidx": pidx, "nodes": nodes,
                                "arrows": None, "oid": person.oid})
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
            if self.scene_node is not None:
                self._show(self.scene_node, self.layers["scene"])
            for h in self.frustums:
                self._show(h, world and self.layers["cameras"])
            if self.path is not None:
                self._show(self.path, world and self.layers["cameras"])
            self._show(self.cursor, world and self.layers["cameras"])
            self._show(self.origin_frustum, (not world) and self.layers["cameras"])
            self._show(self.gravity_node, world and self.layers["gravity"])
            if self.ground_node is not None:
                self._show(self.ground_node, world and self.layers["ground"])

    def set_opacity(self, alpha: float) -> None:
        self.opacity = float(alpha)
        with self.server.atomic():
            for entries in self.bodies.values():
                for e in entries:
                    if e is not None:
                        e["mesh"].opacity = self.opacity

    def set_point_size(self, size: float) -> None:
        if self.scene_node is not None:
            self.scene_node.point_size = float(size)

    def set_force_scale(self, scale: float) -> None:
        self.force_scale = float(scale)
        if self._frame >= 0:
            self.apply_frame(self._frame)

    def set_force_mode(self, mode: str) -> None:
        self.force_mode = mode
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
                    self._pose_skeleton(e, f, skel_on and live)
            per_joint = self.force_mode == "per-joint"
            for name, entries in self.overlays.items():
                if per_joint:
                    contacts, forces = self.data.contacts_joint.get(name), self.data.forces_joint.get(name)
                else:
                    contacts, forces = self.data.contacts.get(name), self.data.forces.get(name)
                c_on = bool(self.layers.get(f"contact_{name}", False)) and contacts is not None
                f_on = bool(self.layers.get(f"force_{name}", False)) and forces is not None
                for e in entries:
                    person, pidx = e["person"], e["pidx"]
                    live = bool(person.valid[f])
                    for mode, nodes in e["nodes"].items():
                        if mode != self.force_mode:
                            self._hide_markers(nodes)
                    nodes = e["nodes"][self.force_mode]
                    if per_joint:
                        points = person.bone_pos[f][self.data.slot_joints] if live else None
                    else:
                        points = self.data.slot_points[name][pidx, f] if name in self.data.slot_points else None
                    live = live and points is not None and bool(np.isfinite(points).all())
                    show_c = c_on and live and bool(np.isfinite(contacts[pidx, f]).any())
                    if show_c:
                        colors = contact_colors(contacts[pidx, f])
                        joints = person.bone_pos[f][self.data.slot_parent]
                        for i, sphere in enumerate(nodes["spheres"]):
                            sphere.position = tuple(float(x) for x in points[i])
                            sphere.color = tuple(int(c) for c in colors[i])
                            self._show(sphere, True)
                        for i, link in enumerate(nodes["links"]):
                            self._place_cylinder(link, joints[i], points[i],
                                                 _FRAME_R * _FRAME_CONN_SCALE)
                    else:
                        self._hide_markers(nodes)
                    show_f = f_on and live and bool(np.isfinite(forces[pidx, f]).any())
                    self._set_arrows(e, name, points if show_f else None,
                                     forces[pidx, f] if show_f else None)
        self._frame = f

    def _pose_skeleton(self, entry: dict, f: int, on: bool) -> None:
        """Place one body's 52 joint spheres and its bone cylinders at frame ``f``."""
        if not on:
            for sphere in entry["joints"]:
                self._show(sphere, False)
            for handle, *_rest in entry["bones"]:
                self._show(handle, False)
            return
        pos = entry["person"].bone_pos[f]
        for j, sphere in enumerate(entry["joints"]):
            sphere.position = tuple(float(x) for x in pos[j])
            self._show(sphere, True)
        for handle, j, parent, radius in entry["bones"]:
            self._place_cylinder(handle, pos[parent], pos[j], radius)

    def _place_cylinder(self, handle, a: np.ndarray, b: np.ndarray, radius: float) -> None:
        """Stretch a unit cylinder handle from ``a`` to ``b``; hide it on a degenerate bone."""
        d = b - a
        length = float(np.linalg.norm(d))
        if not np.isfinite(length) or length < 1e-6:
            self._show(handle, False)
            return
        handle.position = tuple(float(x) for x in (a + b) * 0.5)
        handle.wxyz = _wxyz_shortest_arc_z(d)
        handle.scale = (radius, radius, length)
        self._show(handle, True)

    def _hide_markers(self, nodes: dict) -> None:
        """Hide one display's contact-frame spheres and their connectors."""
        for handle in nodes["spheres"] + nodes["links"]:
            self._show(handle, False)

    def _set_arrows(self, entry: dict, name: str, points: np.ndarray | None,
                    forces: np.ndarray | None) -> None:
        """Re-add one person's force arrows (reusing an arrows handle leaks the node)."""
        if entry["arrows"] is not None:
            entry["arrows"].remove()
            entry["arrows"] = None
        if points is None:
            return
        forces = np.nan_to_num(np.asarray(forces, np.float64))
        keep = np.linalg.norm(forces, axis=-1) >= FORCE_MIN_BW      # a tiny arrow is clutter
        vec = forces * self.force_scale
        if not keep.any():
            return
        seg = np.stack([points[keep], points[keep] + vec[keep]], 1).astype(np.float32)
        entry["arrows"] = self.server.scene.add_arrows(
            f"/clip/force/{name}/p{entry['oid']:02d}", points=seg, colors=_FORCE_RGB,
            shaft_radius=_FORCE_SHAFT, head_radius=_FORCE_HEAD_R, head_length=_FORCE_HEAD_L)

    def dispose(self) -> None:
        """Remove this scene's nodes and hide its body meshes (see the class docstring)."""
        handles = [self.scene_node, self.ground_node, self.path, self.cursor,
                   self.origin_frustum, self.gravity_node, *self.frustums]
        for entries in self.bodies.values():
            for e in entries:
                if e is not None:
                    e["mesh"].visible = False
                    handles += e["joints"] + [b[0] for b in e["bones"]]
        for entries in self.overlays.values():
            for e in entries:
                handles.append(e["arrows"])
                e["arrows"] = None
                for nodes in e["nodes"].values():
                    handles += nodes["spheres"] + nodes["links"]
        with self.server.atomic():
            for h in handles:
                if h is not None:
                    h.remove()
