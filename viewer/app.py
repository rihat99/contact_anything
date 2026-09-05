"""``Viewer`` — the persistent sidebar, scene switching and playback; ``view_results``.

Sidebar (top to bottom): model (run) and scene selection with a pager, the
viewing regime (camera / world) and the onboard-camera toggle, the video pane,
playback (play / fps / frame slider), one checkbox per body source (mesh and
skeleton together) plus a global skeleton switch, the mesh opacity, the
contact / force overlays (predicted and GT, force scale), the static layers
(scene cloud, cameras, gravity), sizes, and an info panel with the run's
checkpoint and the scene's per-source errors against the GT.
"""
from __future__ import annotations

import random
import threading
import time
from pathlib import Path

import numpy as np

from .loading import SOURCES, SceneData, list_runs, list_scenes, load_scene
from .scene import SceneView, onboard_target

_REGIMES = ("camera", "world")
#: Scenes kept in memory (a 400-frame scene with its video pane is ~100 MB).
_CACHE_SCENES = 6
_SOURCE_LABELS = {"predicted": "predicted", "gt": "GT (kindyn)", "frozen": "frozen (SAM 3D)"}


def _blank_image() -> np.ndarray:
    return np.full((2, 2, 3), 255, np.uint8)


class Viewer:
    """Owns the GUI state and swaps the active :class:`SceneView`."""

    def __init__(self, server, runs: list[Path], *, corpus: Path, device: str,
                 video: bool, opacity: float, point_size: float, camera_scale: float,
                 run: str | None = None) -> None:
        self.server = server
        self.runs = runs
        self.corpus = corpus
        self.device = device
        self.video = video
        self.cache: dict[tuple[str, str], SceneData] = {}    # insertion-ordered LRU
        self.view: SceneView | None = None
        self._switching = False
        self.layers = {"scene": True, "cameras": True, "gravity": True,
                       "contact_predicted": True, "force_predicted": True,
                       "contact_gt": False, "force_gt": False}
        # One toggle per body source (mesh + skeleton together) and one global
        # skeleton switch; the scene reads the derived mesh_<src> / skel_<src> keys.
        self.show = {name: True for name in SOURCES}
        self.skeleton = False
        self._derive_body_layers()
        # viser fires GUI callbacks on its own thread; the playback loop and every
        # callback mutate the scene under one reentrant lock (a slider ``.value``
        # set invokes its callback synchronously on the same thread).
        self._lock = threading.RLock()
        self._std_fov: float | None = None

        g = server.gui
        g.add_markdown("### contact_anything · results viewer")
        with g.add_folder("model"):
            names = tuple(r.name for r in runs)
            initial = run if run in names else names[0]
            self.run = g.add_dropdown("run", names, initial_value=initial)
            self.run.on_update(lambda _: self._on_run())
        with g.add_folder("scene"):
            scenes = tuple(list_scenes(self._run_dir()))
            self.scene = g.add_dropdown("scene", scenes, initial_value=scenes[0])
            self.scene.on_update(lambda _: self._on_scene())
            g.add_button("◀ prev").on_click(lambda _: self._nav(-1))
            g.add_button("random").on_click(lambda _: self._nav(0))
            g.add_button("next ▶").on_click(lambda _: self._nav(1))
            self.pos = g.add_markdown("")
        with g.add_folder("view"):
            self.regime = g.add_dropdown("frame", _REGIMES, initial_value="camera")
            self.regime.on_update(lambda _: self._on_regime())
            self.follow = g.add_checkbox("onboard camera", False)
            self.follow.on_update(lambda _: self._on_follow())
            g.add_button("reset view").on_click(lambda _: self._reset_all())
        self.video_img = None
        if video:
            with g.add_folder("video"):
                self.video_img = g.add_image(_blank_image(), format="jpeg", jpeg_quality=80)
        self.play_folder = g.add_folder("playback")
        with self.play_folder:
            self.play = g.add_checkbox("play", False)
            self.fps = g.add_slider("fps", 1, 60, 1, 30)
        self.frame = None
        with g.add_folder("bodies"):
            self.checks = {}
            self.source_checks = {}
            for name in SOURCES:
                c = g.add_checkbox(_SOURCE_LABELS[name], self.show[name])
                c.on_update(self._body_cb)
                self.source_checks[name] = c
            self.skeleton_check = g.add_checkbox("skeletons (all)", self.skeleton)
            self.skeleton_check.on_update(self._body_cb)
            self.opacity = g.add_slider("mesh opacity", 0.1, 1.0, 0.05, float(opacity))
            self.opacity.on_update(lambda _: self._with_view(
                lambda v: v.set_opacity(float(self.opacity.value))))
        with g.add_folder("contacts & forces"):
            for key, label in (("contact_predicted", "contacts (predicted)"),
                               ("force_predicted", "forces (predicted)"),
                               ("contact_gt", "contacts (GT labels)"),
                               ("force_gt", "forces (GT kindyn)")):
                c = g.add_checkbox(label, self.layers[key])
                c.on_update(self._overlay_cb(key))
                self.checks[key] = c
            self.force_scale = g.add_slider("force scale (m per body weight)", 0.05, 1.5, 0.05, 0.3)
            self.force_scale.on_update(lambda _: self._with_view(
                lambda v: v.set_force_scale(float(self.force_scale.value))))
        with g.add_folder("layers"):
            for key, label in (("scene", "scene points"), ("cameras", "cameras"),
                               ("gravity", "gravity (world)")):
                c = g.add_checkbox(label, self.layers[key])
                c.on_update(self._layer_cb(key))
                self.checks[key] = c
        with g.add_folder("sizes", expand_by_default=False):
            self.point_size = g.add_slider("scene points", 0.002, 0.08, 0.002, float(point_size))
            self.point_size.on_update(lambda _: self._with_view(
                lambda v: v.set_point_size(float(self.point_size.value))))
            self.cam_size = g.add_slider("camera frustum", 0.02, 0.6, 0.02, float(camera_scale))
            self.cam_size.on_update(lambda _: self._with_view(
                lambda v: v.set_camera_scale(float(self.cam_size.value))))
        self.info = g.add_markdown("")
        server.on_client_connect(self._on_client_connect)
        self.load(self.scene.value)

    # -- helpers --
    def _run_dir(self) -> Path:
        return next(r for r in self.runs if r.name == self.run.value)

    def _cur(self) -> int:
        return int(self.frame.value) if self.frame is not None else 0

    def _with_view(self, fn) -> None:
        with self._lock:
            if self.view is not None:
                fn(self.view)

    def _derive_body_layers(self) -> None:
        for name in SOURCES:
            self.layers[f"mesh_{name}"] = self.show[name]
            self.layers[f"skel_{name}"] = self.show[name] and self.skeleton

    def _body_cb(self, _) -> None:
        for name, c in self.source_checks.items():
            self.show[name] = bool(c.value)
        self.skeleton = bool(self.skeleton_check.value)
        self._derive_body_layers()
        with self._lock:
            if self.view is not None:
                self.view.apply_frame(self._cur())

    def _layer_cb(self, key: str):
        def cb(_) -> None:
            self.layers[key] = bool(self.checks[key].value)
            with self._lock:
                if self.view is not None:
                    self.view.apply_static_layers()
        return cb

    def _overlay_cb(self, key: str):
        def cb(_) -> None:
            self.layers[key] = bool(self.checks[key].value)
            with self._lock:
                if self.view is not None:
                    self.view.apply_frame(self._cur())
        return cb

    # -- selection --
    def _on_run(self) -> None:
        scenes = tuple(list_scenes(self._run_dir()))
        keep = self.scene.value if self.scene.value in scenes else scenes[0]
        # Re-listing the options may reset the dropdown and fire its callback;
        # suppress that and load exactly once.
        self._switching = True
        try:
            self.scene.options = scenes
            self.scene.value = keep
        finally:
            self._switching = False
        self.load(keep)

    def _on_scene(self) -> None:
        if not getattr(self, "_switching", False):
            self.load(self.scene.value)

    def _nav(self, delta: int) -> None:
        scenes = list(self.scene.options)
        n = len(scenes)
        if n < 2:
            return
        cur = scenes.index(self.scene.value)
        self.scene.value = scenes[(cur + (random.randrange(1, n) if delta == 0 else delta)) % n]

    def _on_regime(self) -> None:
        with self._lock:
            if self.view is not None:
                self.view.set_regime(self.regime.value)
        self._reset_all()

    # -- cameras --
    def _on_client_connect(self, client) -> None:
        client.camera.on_update(lambda _cam: self._on_cam_lock(client))
        with self._lock:
            onboard = self.follow.value and self.view is not None
            if onboard:
                self._drive(client, self._cur())
        if not onboard:
            self._home(client)

    def _on_follow(self) -> None:
        with self._lock:
            if self.view is None:
                return
            if self.follow.value:
                if self._std_fov is None:
                    for client in self.server.get_clients().values():
                        self._std_fov = float(client.camera.fov)
                        break
                for client in self.server.get_clients().values():
                    self._drive(client, self._cur())
            else:
                self._reset_all()

    def _reset_all(self) -> None:
        for client in self.server.get_clients().values():
            self._home(client)

    def _home(self, client) -> None:
        with self._lock:
            if self.view is None:
                return
            pos, look = self.view.home_view(self._cur())
            up = self.view.up_direction()
        client.camera.up_direction = up
        if self._std_fov is not None:
            client.camera.fov = self._std_fov
        client.camera.position = tuple(float(x) for x in pos)
        client.camera.look_at = tuple(float(x) for x in look)

    def _drive(self, client, f: int) -> None:
        """Ride the source camera: its pose in the world regime, the origin in the camera one."""
        if self.view is None:
            return
        data = self.view.data
        ext = data.extrinsics[min(f, data.n_frames - 1)] if self.view.regime == "world" else None
        pos, look, up = onboard_target(ext)
        cam = client.camera
        cam.position = tuple(float(x) for x in pos)
        cam.up_direction = tuple(float(x) for x in up)
        cam.look_at = tuple(float(x) for x in look)
        cam.fov = float(data.fov_y[min(f, data.n_frames - 1)])

    def _on_cam_lock(self, client) -> None:
        with self._lock:
            if not (self.follow.value and self.view is not None):
                return
            f = self._cur()
            data = self.view.data
            ext = data.extrinsics[min(f, data.n_frames - 1)] if self.view.regime == "world" else None
            pos = onboard_target(ext)[0]
            if float(np.linalg.norm(np.asarray(client.camera.position) - pos)) < 1e-2:
                return
            self._drive(client, f)

    # -- frames --
    def _apply_current(self) -> None:
        with self._lock:
            f = self._cur()
            if self.view is not None:
                self.view.apply_frame(f)
            if self.video_img is not None and self.view is not None and self.view.data.video is not None:
                video = self.view.data.video
                self.video_img.image = video[min(f, len(video) - 1)]
            if self.follow.value:
                for client in self.server.get_clients().values():
                    self._drive(client, f)

    def _info_md(self, data: SceneData) -> str:
        m = data.manifest
        ckpt = Path(m.get("checkpoint", "?")).name
        lines = [f"**{data.scene}** — {data.n_frames} frames · {data.fps:.3g} fps · "
                 f"{data.valid_mask.shape[0]} person(s) · {data.width}x{data.height}",
                 f"run **{data.run}** · {ckpt} · epoch {m.get('epoch', '?')} · "
                 f"{'hands' if m.get('hands') else 'body'} · camera head {m.get('camera_head', '?')}",
                 f"predicted {int(data.covered.sum())}/{int(data.valid_mask.sum())} tracked "
                 f"person-frames (stride {data.stride}"
                 + (f", {int(data.held.sum())} held" if data.held.any() else "") + ")"]
        for name in ("predicted", "frozen"):
            met = data.metrics.get(name)
            if met:
                lines.append(f"{_SOURCE_LABELS[name]} vs GT: MPJPE **{met['mpjpe_mm']:.1f} mm** · "
                             f"pelvis {met['pelvis_mm']:.0f} mm ({met['frames']} frames)")
        spread = {name: [p.betas_std for p in data.sources[name].people if p is not None]
                  for name in ("predicted", "frozen")}
        lines.append("betas per-frame std: " + " · ".join(
            f"{name} {np.mean(v):.3f}" for name, v in spread.items() if v)
            + " (meshes at the median identity)")
        for name in ("predicted", "gt"):
            parts = []
            if name in data.contacts:
                c = data.contacts[name]
                parts.append(f"contact rate {np.nanmean(c > 0.5):.2f} on {int(np.isfinite(c).sum())} labels")
            if name in data.forces:
                mag = np.linalg.norm(data.forces[name], axis=-1)
                parts.append(f"mean |f| {np.nanmean(mag):.2f} bw (max {np.nanmax(mag):.1f})")
            if parts:
                lines.append(f"{_SOURCE_LABELS[name]}: " + " · ".join(parts))
        return "\n\n".join(lines)

    def load(self, scene: str) -> None:
        run = self._run_dir()
        print(f"[viewer] loading {run.name} / {scene} …", flush=True)
        with self._lock:
            if self.view is not None:
                self.view.dispose()
                self.view = None
            key = (run.name, scene)
            data = self.cache.pop(key, None)
            if data is None:
                data = load_scene(run, scene, self.corpus, self.device, video=self.video)
            self.cache[key] = data
            while len(self.cache) > _CACHE_SCENES:
                self.cache.pop(next(iter(self.cache)))
            self.view = SceneView(
                self.server, data, self.layers, regime=self.regime.value,
                opacity=float(self.opacity.value), point_size=float(self.point_size.value),
                camera_scale=float(self.cam_size.value),
                force_scale=float(self.force_scale.value))
            if self.frame is not None:
                self.frame.remove()
            with self.play_folder:
                self.frame = self.server.gui.add_slider(
                    "frame", 0, max(data.n_frames - 1, 1), 1, 0)
            self.frame.on_update(lambda _: self._apply_current())
            self.fps.value = int(np.clip(round(data.fps), 1, 60))
            if self.video_img is not None:
                self.video_img.visible = data.video is not None
            scenes = list(self.scene.options)
            self.pos.content = f"scene **{scenes.index(scene) + 1} / {len(scenes)}**"
            self.info.content = self._info_md(data)
            self._apply_current()
        self._reset_all()
        print(f"[viewer] {scene}: {data.n_frames} frames, metrics {data.metrics}", flush=True)

    def run_loop(self) -> None:
        """Advance the frame while ``play`` is on (the slider callback redraws)."""
        while True:
            with self._lock:
                playing = (self.frame is not None and self.play.value
                           and self.view is not None and self.view.n_frames > 0)
                if playing:
                    self.frame.value = (int(self.frame.value) + 1) % self.view.n_frames
            time.sleep(1.0 / float(self.fps.value) if playing else 0.05)


def view_results(output: Path, corpus: Path, *, port: int = 8090, device: str = "cuda",
                 video: bool = True, run: str | None = None, opacity: float = 0.85,
                 point_size: float = 0.02, camera_scale: float = 0.15) -> None:
    """Serve every run with predictions under ``output`` in one viser viewer."""
    import torch
    import viser

    runs = list_runs(output)
    if not runs:
        raise FileNotFoundError(
            f"no <run>/predictions/*.npz under {output} — run scripts/predict_test.py first")
    dev = device if torch.cuda.is_available() or device == "cpu" else "cpu"
    server = viser.ViserServer(port=port)
    server.gui.configure_theme(control_width="large")
    server.scene.set_background_image(_blank_image())
    print(f"[viewer] {len(runs)} run(s) at http://localhost:{port} — "
          f"{', '.join(r.name for r in runs)}", flush=True)
    viewer = Viewer(server, runs, corpus=corpus, device=dev, video=video, opacity=opacity,
                    point_size=point_size, camera_scale=camera_scale, run=run)
    viewer.run_loop()
