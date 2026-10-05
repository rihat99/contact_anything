"""One test scene's data for the viewer: cameras, footage, scene cloud, the three bodies.

Numpy only (viser never enters here); the :class:`SceneData` is cached per
``(run, scene)`` by the app. Runs are discovered as ``<output>/<run>/predictions/``
directories written by ``scripts/predict_test.py``; the GT and frozen bodies and
the cameras come straight from the corpus feature tree of the dataset the run was
trained on (:func:`run_corpus`, from the run's ``config.yaml``):

* ``climbing_videos`` — cameras and scene cloud from ``features/geometry``, the GT
  body / gravity / forces from ``features/human_optim/kindyn_1.npz``, the manual
  contact labels, the frozen SAM 3D body from ``features/sam3d``;
* ``bedlam2`` — everything under ``features/gt/<shard>/<scene>/`` (``camera.npz``,
  ``forces.npz`` = the GT body and its 35-frame forces, ``contacts.npz`` labels);
  the renders carry no scene cloud and no frozen SAM 3D refit, so those stay empty.

:func:`load_wild_scene` is the other mode: an in-the-wild video processed by
``scripts/predict_reconstruction.py`` into ``<root>/<stem>/`` (``geometry/transform.npz``
+ ``predictions/<run>/{smplx,contacts,forces_sup}.npz``). There is no GT, no frozen
body, no scene cloud and no ground plane there — only the ``predicted`` source — and
the pane frames come from the dump's ``source_video`` instead of a corpus frame tree.
"""
from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from data.bedlam2 import scene as bedlam_io
from data.climbing_videos import scene_shard
from data.climbing_videos import scene as scene_io
from data.climbing_videos.kindyn import GRAVITY_MAG
from model.contact_frames import FRAMES35, body22_parent, contact_set

from .bodies import (NUM_BODY_JOINTS, BodySource, empty_source, frozen_source,
                     gt_source, predicted_source, rows_by_id)

_REPO = Path(__file__).resolve().parents[1]

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
    scene_points: np.ndarray         # (M, 3); empty when the corpus has no scene cloud
    scene_colors: np.ndarray         # (M, 3) uint8
    video: list | None               # (N,) JPEG-encoded pane frames (decoded on display)
    sources: dict                    # name -> BodySource (the sources this mode has)
    valid_mask: np.ndarray           # (P, N) tracked person-frames
    covered: np.ndarray              # (P, N) predicted rows (before the stride hold)
    held: np.ndarray                 # (P, N) rows shown by holding the previous prediction
    ground: dict | None              # GT ground plane {normal (3,), centroid (3,)} or None
    slot_names: tuple                # the K contact slots of the run (35 frames or the 6 groups)
    slot_parent: np.ndarray          # (K,) parent joint of each slot in the 52-joint body
    slot_joints: np.ndarray          # (J,) the distinct parent joints, the per-joint fold's rows
    metrics: dict = field(default_factory=dict)   # source -> {mpjpe_mm, pelvis_mm, frames}
    manifest: dict = field(default_factory=dict)  # the run's predictions/manifest.json
    contacts: dict = field(default_factory=dict)  # predicted | gt -> (P, N, K) prob / label, NaN unknown
    forces: dict = field(default_factory=dict)    # predicted | gt -> (P, N, K, 3) world, body-weight units
    slot_points: dict = field(default_factory=dict)   # predicted | gt -> (P, N, K, 3) world slot positions
    contacts_joint: dict = field(default_factory=dict)  # source -> (P, N, J) max over the joint's slots
    forces_joint: dict = field(default_factory=dict)    # source -> (P, N, J, 3) sum over the joint's slots


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


def run_corpus(run: Path) -> tuple[str, Path]:
    """``(dataset name, corpus root)`` of a run: its ``config.yaml``'s first dataset yaml."""
    config = yaml.safe_load((Path(run) / "config.yaml").read_text())
    dataset = yaml.safe_load((_REPO / config["data"]["datasets"][0]).read_text())
    return str(dataset["name"]), Path(dataset["root"])


def decode_pane(frame: bytes) -> np.ndarray:
    """One stored pane frame back to an ``(h, w, 3)`` uint8 RGB image."""
    from PIL import Image

    with Image.open(io.BytesIO(frame)) as im:
        return np.asarray(im.convert("RGB"), np.uint8)


def _read_frames(frames_dir: Path, n: int, max_dim: int) -> list | None:
    """The corpus JPEG frames, longest side capped at ``max_dim``, re-encoded as JPEG
    bytes (a 1500-frame pane is ~30 MB that way instead of gigabytes). ``None`` if absent."""
    from PIL import Image

    if not (frames_dir / "000000.jpg").is_file():
        return None

    def read(position: int) -> bytes | None:
        path = frames_dir / f"{position:06d}.jpg"
        if not path.is_file():
            return None
        with Image.open(path) as im:
            im = im.convert("RGB")
            scale = max_dim / max(im.size)
            if scale < 1.0:
                im = im.resize((round(im.width * scale), round(im.height * scale)), Image.BOX)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=85)
            return buf.getvalue()

    with ThreadPoolExecutor(8) as pool:
        frames = list(pool.map(read, range(n)))
    first = next((f for f in frames if f is not None), None)
    if first is None:
        return None
    return [frame if frame is not None else first for frame in frames]


def _read_video(path: Path, n: int, max_dim: int) -> list:
    """The source mp4 decoded sequentially into ``n`` JPEG pane frames.

    Decodable frame ``k`` is prediction row ``k``; the container header over-reports
    the count, so the decode runs until ``read()`` fails and the result must be exactly
    ``n`` frames long.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        scale = max_dim / max(bgr.shape[:2])
        if scale < 1.0:
            bgr = cv2.resize(bgr, (round(bgr.shape[1] * scale), round(bgr.shape[0] * scale)),
                             interpolation=cv2.INTER_AREA)
        frames.append(cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
    cap.release()
    if len(frames) != n:
        raise ValueError(f"{path}: decoded {len(frames)} frames but the tree has {n}")
    return frames


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


def _ground_plane(gravity_path: Path) -> dict | None:
    """The corpus' fitted ground plane (``ground_normal`` / ``ground_centroid`` of the gravity
    file, when it was accepted), ``None`` otherwise."""
    if not gravity_path.is_file():
        return None
    g = np.load(gravity_path, allow_pickle=True)
    if "ground_accepted" not in g.files or not bool(np.asarray(g["ground_accepted"]).item()):
        return None
    normal = np.asarray(g["ground_normal"], np.float64).reshape(3)
    return {"normal": normal / max(np.linalg.norm(normal), 1e-9),
            "centroid": np.asarray(g["ground_centroid"], np.float64).reshape(3)}


def _gt_slots(body_path: Path, object_ids: np.ndarray, n: int, slots
              ) -> tuple[np.ndarray, np.ndarray]:
    """The GT solve's per-contact-frame labels and forces (``frame_contact`` /
    ``frame_forces`` of ``kindyn_1.npz`` or BEDLAM's ``forces.npz``) on the run's slots:
    labels NaN where the solve is invalid, forces in the WORLD frame in body-weight units."""
    body = np.load(body_path, allow_pickle=True)
    names = tuple(str(x) for x in body["contact_frame_names"])
    if names != tuple(frame[0] for frame in FRAMES35):
        raise ValueError(f"{body_path}: contact_frame_names are not the 35 contact frames")
    frames = contact_set("frames35")
    contact = np.asarray(body["frame_contact"], np.float32)                         # (P', N, 35)
    force = np.asarray(body["frame_forces"], np.float32)                            # (P', N, 35, 3) N
    force = force / (np.asarray(body["total_mass"], np.float32).reshape(-1)[:, None, None, None]
                     * GRAVITY_MAG)
    if not slots.uses_frames:
        contact, force = frames.fold_max(contact), frames.fold_sum(force)
    valid = np.asarray(body["valid_mask"], bool)
    contact[~valid] = np.nan
    force[~valid] = np.nan
    if contact.shape[1] != n:
        raise ValueError(f"{body_path}: {contact.shape[1]} frames but the scene has {n}")
    k = contact.shape[-1]
    labels = np.full((len(object_ids), n, k), np.nan, np.float32)
    forces = np.full((len(object_ids), n, k, 3), np.nan, np.float32)
    for p, oid in enumerate(object_ids):
        row = rows_by_id(body["object_ids"], oid)
        if row is not None:
            labels[p], forces[p] = contact[row], force[row]
    return labels, forces


def _rotate(wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec (..., 3)`` by the unit quaternions ``wxyz (..., 4)``."""
    w, q = wxyz[..., :1], wxyz[..., 1:]
    cross = np.cross(q, vec)
    return vec + 2.0 * w * cross + 2.0 * np.cross(q, cross)


def _slot_points(source: BodySource, slots, vertex: np.ndarray | None) -> np.ndarray:
    """World position of every slot on every person's posed body, ``(P, N, K, 3)``.

    A contact frame is a mesh vertex rigidly attached to its parent joint, so its
    posed position is the joint's bone transform applied to the vertex's rest offset
    (exactly how BetterHuman poses the frames). A group slot sits on its joint.
    NaN where the person is invalid.
    """
    parent = np.asarray(slots.parent_joint52)
    n = max((len(p.valid) for p in source.people if p is not None), default=0)
    out = np.full((len(source.people), n, len(parent), 3), np.nan, np.float32)
    for p, person in enumerate(source.people):
        if person is None:
            continue
        pos = person.bone_pos[:, parent]                                        # (N, K, 3)
        if vertex is not None:
            offset = person.v_shaped[vertex] - person.j_rest[parent]            # (K, 3) rest
            pos = pos + _rotate(person.bone_wxyz[:, parent], offset[None])
        out[p] = pos
    return out


#: A slot whose contact probability / label is below this contributes no force to its joint.
CONTACT_THRESHOLD = 0.5


def _in_contact(forces: np.ndarray, contacts: np.ndarray) -> np.ndarray:
    """``forces (P, N, K, 3)`` with the slots below :data:`CONTACT_THRESHOLD` zeroed (NaN kept)."""
    off = np.isfinite(forces).all(-1) & ~(np.nan_to_num(contacts) >= CONTACT_THRESHOLD)
    return np.where(off[..., None], 0.0, forces).astype(np.float32)


def _fold_joints(values: np.ndarray, slot_parent: np.ndarray, joints: np.ndarray,
                 reduce: str) -> np.ndarray:
    """Fold per-slot ``values (P, N, K[, 3])`` onto the distinct parent joints ``(P, N, J[, 3])``:
    ``max`` for labels / probabilities, ``sum`` for forces; a joint whose slots are all
    NaN stays NaN."""
    out = np.full(values.shape[:2] + (len(joints),) + values.shape[3:], np.nan, np.float32)
    for j, joint in enumerate(joints):
        part = values[:, :, slot_parent == joint]
        known = np.isfinite(part).reshape(part.shape[:3] + (-1,)).any(-1)      # (P, N, k)
        if reduce == "max":
            folded = np.nanmax(np.where(known, part, -np.inf), axis=2)
        else:
            folded = np.nansum(np.where(known[..., None], part, 0.0), axis=2)
        out[:, :, j] = np.where(known.any(-1)[..., None] if values.ndim == 4 else known.any(-1),
                                folded, np.nan)
    return out


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


def load_scene(run: Path, scene: str, device, *, video: bool = True,
               max_dim: int = 400) -> SceneData:
    """Load one scene of one run into a :class:`SceneData`."""
    run = Path(run)
    dataset, corpus = run_corpus(run)
    shard = scene_shard(scene)
    features = corpus / "features"
    pred_path = run / "predictions" / f"{scene}.npz"
    if dataset == "bedlam2":
        gt_dir = bedlam_io.gt_dir(corpus, scene)
        camera_path, body_path = gt_dir / "camera.npz", gt_dir / "forces.npz"
        gravity_path = gt_dir / "gravity.npz"
        frozen_path = cloud_path = None
    else:
        gt_dir = features / "human_optim" / shard / scene
        camera_path = features / "geometry" / shard / scene / "transform.npz"
        body_path = gt_dir / "kindyn_1.npz"
        gravity_path = Path(scene_io.gravity_path(corpus, scene))
        frozen_path = features / "sam3d" / shard / scene / "smplx_params.npz"
        cloud_path = features / "geometry" / shard / scene / "scene.npz"
    camera = np.load(camera_path)
    extrinsics = np.asarray(camera["extrinsics"], np.float32)
    intrinsics = np.asarray(camera["intrinsics_px_orig"], np.float32)
    n = len(extrinsics)
    width, height = int(camera["image_width"]), int(camera["image_height"])
    fov_y = 2.0 * np.arctan(height / (2.0 * intrinsics[:, 1, 1].astype(np.float64)))

    body = np.load(body_path, allow_pickle=True)
    gravity = np.asarray(body["gravity_world"], np.float64)
    gravity = (gravity / max(np.linalg.norm(gravity), 1e-9)).astype(np.float32)

    pred = np.load(pred_path)
    object_ids = np.asarray(pred["object_ids"], np.int32)
    stride = int(pred["stride"])
    covered = np.asarray(pred["covered"], bool)
    if "tracked" in pred.files:
        valid_mask = np.asarray(pred["tracked"], bool)
    else:                       # dumps before 2026-09-05: the raw contacts_1 validity
        contacts = np.load(gt_dir / "contacts_1.npz", allow_pickle=True)
        valid_mask = np.asarray(contacts["valid_mask"], bool)
    if valid_mask.shape != covered.shape or covered.shape[1] != n:
        raise ValueError(f"{pred_path}: covered {covered.shape} vs tracked {valid_mask.shape} "
                         f"vs {n} frames")
    sources = {
        "predicted": predicted_source(pred_path, extrinsics, device),
        "gt": gt_source(body_path, object_ids, n, device),
        "frozen": (frozen_source(frozen_path, extrinsics, object_ids, device)
                   if frozen_path is not None else empty_source("frozen", object_ids, device)),
    }
    # Metrics on the real predictions only; the stride hold below duplicates rows.
    metrics = {name: _pose_metrics(sources[name], sources["gt"])
               for name in ("predicted", "frozen")}
    held = _hold_stride_gaps(sources["predicted"], stride, valid_mask)
    # Everything below is per SLOT of the run's contact set (the 35 frames, or the six
    # groups of an older dump); the per-joint fold sums / maxes a joint's slots.
    slots = contact_set(str(pred["contact_set"]) if "contact_set" in pred.files else "kindyn6")
    vertex = np.asarray([frame[2] for frame in FRAMES35]) if slots.uses_frames else None
    # The per-joint fold sums a slot's force at its BODY joint: the hand frames (palm,
    # fingers, thumb) all land on the wrist, as in BVR's viewer.
    slot_parent = np.asarray([body22_parent(j) for j in slots.parent_joint52], np.int64)
    slot_joints = np.unique(slot_parent)
    contacts, forces, slot_points = {}, {}, {}
    if "contact_probs" in pred.files:
        contacts["predicted"] = _hold_rows(pred["contact_probs"], covered, held)
    if "forces_world" in pred.files:
        forces["predicted"] = _hold_rows(pred["forces_world"], covered, held)
    slot_points["predicted"] = _slot_points(sources["predicted"], slots, vertex)
    try:
        contacts["gt"], forces["gt"] = _gt_slots(body_path, object_ids, n, slots)
        slot_points["gt"] = _slot_points(sources["gt"], slots, vertex)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[viewer] {scene}: no GT contact frames ({exc})", flush=True)
    contacts_joint = {k: _fold_joints(v, slot_parent, slot_joints, "max") for k, v in contacts.items()}
    forces_joint = {k: _fold_joints(_in_contact(v, contacts[k]), slot_parent, slot_joints, "sum")
                    for k, v in forces.items() if k in contacts}

    if cloud_path is not None:
        cloud = np.load(cloud_path)
        scene_points = np.asarray(cloud["points"], np.float32)
        scene_colors = np.asarray(cloud["colors"], np.uint8)
    else:
        scene_points, scene_colors = np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    manifest_path = run / "predictions" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    # A dump in progress has no manifest yet; the npz carries the run identity too.
    for key in ("checkpoint", "epoch", "exp_name", "hands"):
        if key not in manifest and key in pred.files:
            manifest[key] = pred[key].item() if pred[key].ndim == 0 else pred[key]
    return SceneData(
        run=run.name, scene=scene, n_frames=n, fps=float(camera["fps"]), stride=stride,
        extrinsics=extrinsics, intrinsics=intrinsics, width=width, height=height,
        fov_y=fov_y.astype(np.float32), gravity=gravity,
        scene_points=scene_points, scene_colors=scene_colors,
        video=_read_frames(corpus / "frames" / shard / scene, n, max_dim) if video else None,
        sources=sources, valid_mask=valid_mask, covered=covered, held=held,
        ground=_ground_plane(gravity_path), slot_names=tuple(slots.slot_names),
        slot_parent=slot_parent, slot_joints=slot_joints,
        metrics=metrics, manifest=manifest, contacts=contacts, forces=forces,
        slot_points=slot_points, contacts_joint=contacts_joint, forces_joint=forces_joint)


def list_wild_runs(root: Path) -> list[str]:
    """Prediction run names found under the wild tree's ``<stem>/predictions/``."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted({d.name for stem in root.iterdir() if stem.is_dir()
                   for d in (stem / "predictions").glob("*") if (d / "smplx.npz").is_file()})


def list_wild_scenes(root: Path, run: str) -> list[str]:
    """Video stems of the wild tree that carry ``run``'s predictions."""
    root = Path(root)
    return sorted(stem.name for stem in root.iterdir()
                  if (stem / "predictions" / run / "smplx.npz").is_file())


def load_wild_scene(root: Path, scene: str, run: str, device, *, video: bool = True,
                    max_dim: int = 400) -> SceneData:
    """Load one in-the-wild video's predictions into a :class:`SceneData`.

    Only the ``predicted`` source exists: no GT body, no frozen body, no scene cloud,
    no ground plane and no pose metrics. The world's down direction is the median of
    the model's own finite ``gravity_world`` rows.
    """
    tree = Path(root) / scene
    pred_dir = tree / "predictions" / run
    camera = np.load(tree / "geometry" / "transform.npz")
    extrinsics = np.asarray(camera["extrinsics"], np.float32)
    intrinsics = np.asarray(camera["intrinsics_px_orig"], np.float32)
    n = len(extrinsics)
    width, height = int(camera["image_width"]), int(camera["image_height"])
    fov_y = 2.0 * np.arctan(height / (2.0 * intrinsics[:, 1, 1].astype(np.float64)))

    smplx_path = pred_dir / "smplx.npz"
    sx = np.load(smplx_path)
    covered = np.asarray(sx["covered"], bool)
    valid_mask = np.asarray(sx["tracked"], bool)
    stride = int(sx["stride"])
    if covered.shape[1] != n:
        raise ValueError(f"{smplx_path}: {covered.shape[1]} rows but {n} extrinsics")
    down = np.asarray(sx["gravity_world"], np.float64).reshape(-1, 3)
    down = down[np.isfinite(down).all(1)]
    if not len(down):
        raise ValueError(f"{smplx_path}: gravity_world has no finite row")
    down = np.median(down, axis=0)
    gravity = (down / max(np.linalg.norm(down), 1e-9)).astype(np.float32)

    sources = {"predicted": predicted_source(smplx_path, extrinsics, device)}
    held = _hold_stride_gaps(sources["predicted"], stride, valid_mask)
    slots = contact_set(str(sx["contact_set"]))
    vertex = np.asarray([frame[2] for frame in FRAMES35]) if slots.uses_frames else None
    # The per-joint fold sums a slot's force at its BODY joint: the hand frames (palm,
    # fingers, thumb) all land on the wrist, as in BVR's viewer.
    slot_parent = np.asarray([body22_parent(j) for j in slots.parent_joint52], np.int64)
    slot_joints = np.unique(slot_parent)
    contacts = {"predicted": _hold_rows(np.load(pred_dir / "contacts.npz")["probs"],
                                        covered, held)}
    forces = {"predicted": _hold_rows(np.load(pred_dir / "forces_sup.npz")["forces_world"],
                                      covered, held)}
    slot_points = {"predicted": _slot_points(sources["predicted"], slots, vertex)}
    source_video = Path(str(sx["source_video"]))
    if not source_video.is_absolute():
        source_video = _REPO / source_video
    return SceneData(
        run=run, scene=scene, n_frames=n, fps=float(sx["fps"]), stride=stride,
        extrinsics=extrinsics, intrinsics=intrinsics, width=width, height=height,
        fov_y=fov_y.astype(np.float32), gravity=gravity,
        scene_points=np.zeros((0, 3), np.float32), scene_colors=np.zeros((0, 3), np.uint8),
        video=_read_video(source_video, n, max_dim) if video else None,
        sources=sources, valid_mask=valid_mask, covered=covered, held=held, ground=None,
        slot_names=tuple(slots.slot_names), slot_parent=slot_parent, slot_joints=slot_joints,
        manifest={"checkpoint": str(sx["checkpoint"]), "epoch": int(sx["checkpoint_epoch"]),
                  "hands": bool(sx["hands"])},
        contacts=contacts, forces=forces, slot_points=slot_points,
        contacts_joint={"predicted": _fold_joints(contacts["predicted"], slot_parent,
                                                  slot_joints, "max")},
        forces_joint={"predicted": _fold_joints(_in_contact(forces["predicted"], contacts["predicted"]),
                                                slot_parent, slot_joints, "sum")})
