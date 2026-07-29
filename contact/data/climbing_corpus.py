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
kindyn convention's exact ``[0, 1, 0]`` (world y is down) rather than the v1
camera-0-derived direction, and ``load_forces=True`` adds per-frame GT forces:

* ``force_gt`` ``[6, 3]`` float32 — solved contact force per group in
  :data:`FORCE_GROUP_NAMES` order, in body-weight units
  (``newtons / (total_mass * 9.81)``), rotated **world -> body-root**:
  ``f_root = R(q_xyzw)^T @ f_world`` where ``q[3:7]`` is the kindyn root
  quaternion in ``xyzw`` order (numerically verified against the stored
  axis-angle ``global_orient``; ``R(q)`` is world-from-root).
* ``force_contact`` ``[6]`` bool — the kindyn/contacts_2 contact mask the
  forces were solved under, folded per group (hands = wrist + 15 fingers).
  A zero force means *unlabeled*, not measured-zero.
* ``force_valid`` bool — frame valid and covered by the kindyn solve.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
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

# Pinned confidence schema of contacts_<level>.npz (same pin as the v1 exporter).
CONTACT_CONFIDENCE_SCHEMA = 8

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

GRAVITY_MAG = 9.81
# Exact scene-world down direction of every kindyn solve (world y is down). The
# v1 export instead *derived* gravity from camera 0 — do not mix the two.
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


def merge_contacts_52_to_22(
    jc52: np.ndarray, conf52: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold 52-joint contacts + confidence into the 22 SMPL-X body joints.

    Port of the v1 exporter's fold (``export_contact_dataset.py``): joints 0-21
    pass through; each hand ORs the wrist + its 15 fingers. The folded
    confidence is the strongest *touching* member when the hand is in contact
    (max) and the weakest member when it is free (min over all 16 — one
    occluded finger makes the whole-hand free label uncertain).

    :param jc52: ``(P, N, 52)`` bool contact labels.
    :param conf52: ``(P, N, 52)`` float32 label confidence in ``[0, 1]``.
    :returns: ``(jc22 (P,N,22) bool, conf22 (P,N,22) float32)``.
    """
    jc52 = np.asarray(jc52, bool)
    conf52 = np.asarray(conf52, np.float32)
    if jc52.ndim != 3 or jc52.shape[-1] != N_JOINTS_52:
        raise ValueError(f"joint_contact must have shape (P,N,52), got {jc52.shape}")
    if conf52.shape != jc52.shape:
        raise ValueError(
            f"label_confidence shape {conf52.shape} does not match contacts {jc52.shape}")
    if not np.isfinite(conf52).all() or bool(((conf52 < 0.0) | (conf52 > 1.0)).any()):
        raise ValueError("label_confidence must be finite and within [0, 1]")

    jc22 = jc52[..., :NUM_BODY_22].copy()
    conf22 = conf52[..., :NUM_BODY_22].copy()
    for wrist, group in _HAND_FOLDS:
        sub_lbl = jc52[..., group]                                      # (P, N, 16)
        sub_conf = conf52[..., group]
        lbl = sub_lbl.any(axis=-1)
        conf_touch = np.where(sub_lbl, sub_conf, -np.inf).max(axis=-1)  # strongest touching vote
        conf_free = sub_conf.min(axis=-1)                               # weakest member of the free AND
        jc22[..., wrist] = lbl
        conf22[..., wrist] = np.where(lbl, conf_touch, conf_free)
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
    :param frame_stride: source-frame stride within a window.
    :param jitter: stateless train-window jitter (train split only).
    :param seed: seed for both the grouped split and the window jitter.
    :param val_fraction: source-video fraction held out for ``val``.
    :param contact_level: label readout level — ``contacts_1.npz`` or ``contacts_2.npz``.
    :param use_confidence_weights: multiply ``joint_mask`` by label confidence.
    :param require_labels: on ``test``, require the manual annotation (raise when
        missing; discovery skips unannotated scenes). Train labels are always loaded.
    :param load_forces: read ``kindyn_1.npz`` and emit ``force_gt`` /
        ``force_contact`` / ``force_valid`` per frame.
    :param load_images: read frame JPEGs + person masks in ``__getitem__``.
        ``False`` returns ``image``/``mask`` as ``None`` (metadata-only access;
        such items cannot go through the training collate).
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
        frame_stride: int = 2,
        jitter: bool = True,
        seed: int = 42,
        val_fraction: float = 0.15,
        contact_level: int = 1,
        use_confidence_weights: bool = False,
        require_labels: bool = True,
        load_forces: bool = False,
        load_images: bool = True,
    ):
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val' or 'test'; got {split!r}")
        if contact_level not in (1, 2):
            raise ValueError(f"contact_level must be 1 or 2; got {contact_level!r}")
        self.corpus_root = Path(corpus_root)
        self.split = split
        self.mode = "train" if split == "train" else "val"
        self.T = int(frames_per_clip)
        self.stride = int(frame_stride)
        self.jitter = bool(jitter) and split == "train"
        self.seed = int(seed)
        self.val_fraction = float(val_fraction)
        self.contact_level = int(contact_level)
        self.use_confidence_weights = bool(use_confidence_weights)
        self.require_labels = bool(require_labels)
        self.load_forces = bool(load_forces)
        self.load_images = bool(load_images)
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

        self._scenes: dict[str, dict] = {}
        self._items: list[tuple[str, int, int, int]] = []   # (scene, person, base_start, jitter_range)
        span = (self.T - 1) * self.stride
        step = self.T * self.stride
        for scene in scenes:
            data = self._load_scene(scene)
            self._scenes[scene] = data
            num_frames = len(data["frame_indices"])
            valid_mask = data["valid_mask"]                 # [P, N] bool
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
                    positions = base + np.arange(self.T) * self.stride
                    if not valid_mask[person, positions].all():
                        continue  # every temporal row needs a real bbox/camera crop
                    jitter_range = max(1, min(step, max_start - base + 1))
                    self._items.append((scene, person, base, jitter_range))

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
            schema = int(np.asarray(contacts["confidence_schema"]).item())
            if schema != CONTACT_CONFIDENCE_SCHEMA:
                raise ValueError(
                    f"{scene}: contacts_{self.contact_level} confidence_schema={schema}, "
                    f"expected {CONTACT_CONFIDENCE_SCHEMA}")
            joint_contact, contact_conf = merge_contacts_52_to_22(
                contacts["joint_contact"], contacts["label_confidence"])
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
        return data

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

    def _load_forces(
        self, scene: str, human_dir: Path, object_ids: np.ndarray, n: int,
    ) -> dict:
        """Load kindyn GT forces: body-weight units, rotated world -> body-root."""
        kindyn = np.load(human_dir / "kindyn_1.npz", allow_pickle=True)
        kindyn_ids = np.asarray(kindyn["object_ids"])
        force_joints = _rows_by_object_id(
            kindyn["contact_force_joints"], kindyn_ids, object_ids, scene, "kindyn")
        for row in force_joints:
            if tuple(str(x) for x in row) != KINDYN_FORCE_JOINTS:
                raise ValueError(
                    f"{scene}: contact_force_joints {[str(x) for x in row]} != "
                    f"{list(KINDYN_FORCE_JOINTS)}")
        forces_n = _rows_by_object_id(
            np.asarray(kindyn["contact_forces"], np.float32),
            kindyn_ids, object_ids, scene, "kindyn")          # [P, N, 6, 3] newtons, world
        q = _rows_by_object_id(
            np.asarray(kindyn["q"], np.float32),
            kindyn_ids, object_ids, scene, "kindyn")          # [P, N, 211]
        total_mass = _rows_by_object_id(
            np.asarray(kindyn["total_mass"], np.float32).reshape(-1),
            kindyn_ids, object_ids, scene, "kindyn")          # [P] kg
        force_valid = _rows_by_object_id(
            np.asarray(kindyn["valid_mask"], bool),
            kindyn_ids, object_ids, scene, "kindyn")          # [P, N]
        group_contact = fold_force_group_contact(
            _rows_by_object_id(
                np.asarray(kindyn["joint_contact"], bool),
                kindyn_ids, object_ids, scene, "kindyn"))     # [P, N, 6]

        n_people = len(object_ids)
        if forces_n.shape != (n_people, n, NUM_FORCE_GROUPS, 3):
            raise ValueError(
                f"{scene}: contact_forces {forces_n.shape} does not match "
                f"({n_people}, {n}, {NUM_FORCE_GROUPS}, 3)")
        if not np.isfinite(forces_n).all():
            raise ValueError(f"{scene}: contact_forces contain non-finite values")
        if (not np.isfinite(total_mass).all() or (total_mass <= 0).any()
                or not np.isfinite(np.asarray(kindyn["betas"])).all()):
            raise ValueError(f"{scene}: kindyn total_mass/betas are not sane")
        # Forces are only ever solved under the contact mask: a nonzero force on
        # an uncontacted group means corrupted data. (The converse — zero force
        # during contact — is possible in principle, so it is not asserted.)
        nonzero = np.linalg.norm(forces_n, axis=-1) > 0
        if bool((nonzero & ~group_contact).any()):
            raise ValueError(
                f"{scene}: nonzero contact force on a group with no contact label")

        forces_bw = forces_n / (total_mass[:, None, None, None] * GRAVITY_MAG)
        # q[3:7] is the root quaternion, xyzw, R(q) = world-from-root (verified
        # against the stored axis-angle global_orient); rotate world -> root.
        rot = quat_xyzw_to_matrix(q[..., 3:7])                # [P, N, 3, 3]
        force_root = np.einsum("pnji,pnkj->pnki", rot, forces_bw).astype(np.float32)
        return {
            "force_gt": force_root,                          # [P, N, 6, 3] bw, root frame
            "force_contact": group_contact,                  # [P, N, 6] bool
            "force_valid": force_valid,                      # [P, N] bool
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
        scene, person, base, jitter_range = self._items[index]
        data = self._scenes[scene]
        start = self._window_start(base, jitter_range, index)
        positions = start + np.arange(self.T) * self.stride
        # The indexed base window is all-valid. If jitter crosses a tracking gap,
        # fall back deterministically so invalid-frame bboxes never reach the crop.
        if start != base and not data["valid_mask"][person, positions].all():
            start = base
            positions = base + np.arange(self.T) * self.stride

        oid = int(data["object_ids"][person])
        frame_indices = data["frame_indices"]
        fps = data["fps"]
        start_time = float(frame_indices[start])

        clip = []
        for row, pos in enumerate(positions):
            pos = int(pos)
            image = mask = None
            if self.load_images:
                image = np.array(
                    Image.open(data["frames_dir"] / f"{pos:06d}.jpg").convert("RGB"),
                    np.uint8)
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
            if self.load_forces:
                frame["force_gt"] = torch.from_numpy(data["force_gt"][person, pos])   # [6, 3]
                frame["force_contact"] = torch.from_numpy(
                    data["force_contact"][person, pos])                               # [6] bool
                frame["force_valid"] = valid and bool(data["force_valid"][person, pos])
            clip.append(frame)
        return clip
