"""One test scene's data for the viewer: cameras, footage, scene cloud, the three bodies.

Numpy only (viser never enters here); the :class:`SceneData` is cached per
``(run, scene)`` by the app. Runs are discovered as ``<output>/<run>/predictions/``
directories written by ``scripts/predict_test.py``; the GT and frozen bodies and
the cameras come straight from the corpus feature tree.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from data.climbing_videos import kindyn as kindyn_io, scene_shard
from data.climbing_videos import scene as scene_io

from .bodies import NUM_BODY_JOINTS, BodySource, frozen_source, gt_source, predicted_source

#: Display order of the sources (and the dict order of :attr:`SceneData.sources`).
SOURCES = ("predicted", "gt", "frozen")
#: Mean of the two hip joints: the alignment pelvis of the pose metrics.
_HIPS = (1, 2)


@dataclass
class SceneData:
    """Everything the viewer draws for one ``(run, scene)``, in the metric world frame."""

    run: str
    scene: str
    n_frames: int
    fps: float
    stride: int                      # prediction stride (source frames per predicted row)
    extrinsics: np.ndarray           # (N, 4, 4) cam_from_world, OpenCV, metric
    intrinsics: np.ndarray           # (N, 3, 3) full-frame pixels
    width: int
    height: int
    fov_y: np.ndarray                # (N,) vertical field of view, radians
    gravity: np.ndarray              # (3,) unit DOWN vector (kindyn fitted)
    scene_points: np.ndarray         # (M, 3)
    scene_colors: np.ndarray         # (M, 3) uint8
    video: np.ndarray | None         # (N, h, w, 3) uint8 corpus frames (sidebar pane)
    sources: dict                    # name -> BodySource (SOURCES order)
    valid_mask: np.ndarray           # (P, N) tracked person-frames
    covered: np.ndarray              # (P, N) predicted rows (before the stride hold)
    held: np.ndarray                 # (P, N) rows shown by holding the previous prediction
    metrics: dict = field(default_factory=dict)   # source -> {mpjpe_mm, pelvis_mm, frames}
    manifest: dict = field(default_factory=dict)  # the run's predictions/manifest.json
    contacts: dict = field(default_factory=dict)  # predicted | gt -> (P, N, 6) prob / label, NaN unknown
    forces: dict = field(default_factory=dict)    # predicted | gt -> (P, N, 6, 3) world, body-weight units


def list_runs(output_root: Path) -> list[Path]:
    """Run directories under ``output_root`` that carry a predictions dump."""
    root = Path(output_root)
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir()
                  if d.is_dir() and any((d / "predictions").glob("*.npz")))


def list_scenes(run: Path) -> list[str]:
    """Scene ids a run has predictions for."""
    return sorted(p.stem for p in (Path(run) / "predictions").glob("*.npz"))


def _read_frames(frames_dir: Path, n: int, max_dim: int) -> np.ndarray | None:
    """The corpus JPEG frames, longest side capped at ``max_dim``. ``None`` if absent."""
    from PIL import Image

    if not (frames_dir / "000000.jpg").is_file():
        return None

    def read(position: int) -> np.ndarray | None:
        path = frames_dir / f"{position:06d}.jpg"
        if not path.is_file():
            return None
        with Image.open(path) as im:
            im = im.convert("RGB")
            scale = max_dim / max(im.size)
            if scale < 1.0:
                im = im.resize((round(im.width * scale), round(im.height * scale)), Image.BOX)
            return np.asarray(im, np.uint8)

    with ThreadPoolExecutor(8) as pool:
        frames = list(pool.map(read, range(n)))
    first = next((f for f in frames if f is not None), None)
    if first is None:
        return None
    out = np.zeros((n,) + first.shape, np.uint8)
    for i, frame in enumerate(frames):
        if frame is not None and frame.shape == first.shape:
            out[i] = frame
    return out


def _hold_stride_gaps(source: BodySource, stride: int, tracked: np.ndarray) -> np.ndarray:
    """Fill the frames between stride steps with the previous predicted row.

    Only TRACKED frames within ``stride`` of a predicted row are filled, so a hold
    never extends a body past the end of its run. Returns the ``(P, N)`` mask of
    held rows (the prediction only exists on the stride grid; the hold keeps the
    body from blinking at 60 fps). Call AFTER the metrics: it mutates the people.
    """
    held = np.zeros(tracked.shape, bool)
    if stride <= 1:
        return held
    for p, person in enumerate(source.people):
        if person is None:
            continue
        last = -stride
        for f in range(len(person.valid)):
            if person.valid[f]:
                last = f
            elif 0 < f - last < stride and tracked[p, f]:
                person.bone_wxyz[f] = person.bone_wxyz[last]
                person.bone_pos[f] = person.bone_pos[last]
                person.valid[f] = True
                held[p, f] = True
    return held


def _hold_rows(values: np.ndarray, covered: np.ndarray, held: np.ndarray) -> np.ndarray:
    """Forward-fill the per-frame ``values (P, N, ...)`` onto the stride-held frames."""
    out = np.array(values, np.float32, copy=True)
    for p in range(out.shape[0]):
        last = None
        for f in range(out.shape[1]):
            if covered[p, f]:
                last = f
            elif held[p, f] and last is not None:
                out[p, f] = out[p, last]
    return out


def _gt_contacts_forces(corpus: Path, scene: str, object_ids: np.ndarray, n: int
                        ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """The manual six-group contact labels (NaN = not annotated) and the kindyn GT forces
    in the WORLD frame (body-weight units, NaN where kindyn is invalid); ``None`` when a
    scene lacks the archive."""
    shard = scene_shard(scene)
    human_dir = corpus / "features" / "human_optim" / shard / scene
    labels = forces = None
    try:
        data = scene_io.load_scene(corpus, scene, "test", 1)
        labels = np.asarray(data["contact_gt"], np.float32)
        labels[np.asarray(data["contact_valid"]) <= 0] = np.nan
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[viewer] {scene}: no manual contact labels ({exc})", flush=True)
    try:
        kd = kindyn_io.load_forces(scene, human_dir, object_ids, n)         # bw, GT root frame
        raw = np.load(human_dir / "kindyn_1.npz", allow_pickle=True)
        q = scene_io.rows_by_object_id(np.asarray(raw["q"], np.float32), np.asarray(raw["object_ids"]),
                                       object_ids, scene, "kindyn")
        rot = kindyn_io.quat_xyzw_to_matrix(q[..., 3:7])                    # world-from-root
        forces = np.einsum("pnij,pnkj->pnki", rot, kd["force_gt"]).astype(np.float32)
        forces[~np.asarray(kd["force_valid"], bool)] = np.nan
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[viewer] {scene}: no kindyn GT forces ({exc})", flush=True)
    return labels, forces


def _pose_metrics(source: BodySource, gt: BodySource) -> dict | None:
    """Body-22 MPJPE (mean-hips aligned) and absolute pelvis error vs the GT, in mm."""
    errors, pelvis = [], []
    for person, ref in zip(source.people, gt.people):
        if person is None or ref is None:
            continue
        both = person.valid & ref.valid
        a = person.bone_pos[both, :NUM_BODY_JOINTS]
        b = ref.bone_pos[both, :NUM_BODY_JOINTS]
        if not len(a):
            continue
        a0 = a - a[:, list(_HIPS)].mean(1, keepdims=True)
        b0 = b - b[:, list(_HIPS)].mean(1, keepdims=True)
        errors.append(np.linalg.norm(a0 - b0, axis=-1).mean(1))
        pelvis.append(np.linalg.norm(a[:, 0] - b[:, 0], axis=-1))
    if not errors:
        return None
    errors, pelvis = np.concatenate(errors), np.concatenate(pelvis)
    return {"mpjpe_mm": float(errors.mean() * 1000.0),
            "pelvis_mm": float(pelvis.mean() * 1000.0), "frames": int(len(errors))}


def load_scene(run: Path, scene: str, corpus: Path, device, *, video: bool = True,
               max_dim: int = 400) -> SceneData:
    """Load one scene of one run into a :class:`SceneData`."""
    run, corpus = Path(run), Path(corpus)
    shard = scene_shard(scene)
    features = corpus / "features"
    pred_path = run / "predictions" / f"{scene}.npz"
    transform = np.load(features / "geometry" / shard / scene / "transform.npz")
    extrinsics = np.asarray(transform["extrinsics"], np.float32)
    intrinsics = np.asarray(transform["intrinsics_px_orig"], np.float32)
    n = len(extrinsics)
    width, height = int(transform["image_width"]), int(transform["image_height"])
    fov_y = 2.0 * np.arctan(height / (2.0 * intrinsics[:, 1, 1].astype(np.float64)))

    kindyn_path = features / "human_optim" / shard / scene / "kindyn_1.npz"
    kindyn = np.load(kindyn_path, allow_pickle=True)
    gravity = np.asarray(kindyn["gravity_world"], np.float64)
    gravity = (gravity / max(np.linalg.norm(gravity), 1e-9)).astype(np.float32)

    pred = np.load(pred_path)
    object_ids = np.asarray(pred["object_ids"], np.int32)
    stride = int(pred["stride"])
    covered = np.asarray(pred["covered"], bool)
    if "tracked" in pred.files:
        valid_mask = np.asarray(pred["tracked"], bool)
    else:                       # dumps before 2026-09-05: the raw contacts_1 validity
        contacts = np.load(features / "human_optim" / shard / scene / "contacts_1.npz",
                           allow_pickle=True)
        valid_mask = np.asarray(contacts["valid_mask"], bool)
    if valid_mask.shape != covered.shape or covered.shape[1] != n:
        raise ValueError(f"{pred_path}: covered {covered.shape} vs tracked {valid_mask.shape} "
                         f"vs {n} frames")
    sources = {
        "predicted": predicted_source(pred_path, extrinsics, device),
        "gt": gt_source(kindyn_path, object_ids, n, device),
        "frozen": frozen_source(features / "sam3d" / shard / scene / "smplx_params.npz",
                                extrinsics, object_ids, device),
    }
    # Metrics on the real predictions only; the stride hold below duplicates rows.
    metrics = {name: _pose_metrics(sources[name], sources["gt"])
               for name in ("predicted", "frozen")}
    held = _hold_stride_gaps(sources["predicted"], stride, valid_mask)
    contacts, forces = {}, {}
    if "contact_probs" in pred.files:
        contacts["predicted"] = _hold_rows(pred["contact_probs"], covered, held)
    if "forces_world" in pred.files:
        forces["predicted"] = _hold_rows(pred["forces_world"], covered, held)
    gt_labels, gt_forces = _gt_contacts_forces(corpus, scene, object_ids, n)
    if gt_labels is not None:
        contacts["gt"] = gt_labels
    if gt_forces is not None:
        forces["gt"] = gt_forces

    cloud = np.load(features / "geometry" / shard / scene / "scene.npz")
    manifest_path = run / "predictions" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    # A dump in progress has no manifest yet; the npz carries the run identity too.
    for key in ("checkpoint", "epoch", "exp_name", "hands"):
        if key not in manifest and key in pred.files:
            manifest[key] = pred[key].item() if pred[key].ndim == 0 else pred[key]
    return SceneData(
        run=run.name, scene=scene, n_frames=n, fps=float(transform["fps"]), stride=stride,
        extrinsics=extrinsics, intrinsics=intrinsics, width=width, height=height,
        fov_y=fov_y.astype(np.float32), gravity=gravity,
        scene_points=np.asarray(cloud["points"], np.float32),
        scene_colors=np.asarray(cloud["colors"], np.uint8),
        video=_read_frames(corpus / "frames" / shard / scene, n, max_dim) if video else None,
        sources=sources, valid_mask=valid_mask, covered=covered, held=held,
        metrics=metrics, manifest=manifest, contacts=contacts, forces=forces)
