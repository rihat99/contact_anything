"""ClimbingVideos corpus: scene discovery and per-scene geometry + contact labels.

The corpus is read directly from its pipeline tree (no exported dataset):

* ``scenes/scenes.db`` — the curated scene filter and the train/test split.
* ``frames/<shard>/<scene>/<pos:06d>.jpg`` — pre-extracted frames; row ``k`` of
  every feature array is frame ``k``.
* ``features/sam3/<shard>/<scene>`` — ``bboxes.npz`` person tracks and
  ``<oid:02d>/frame_<pos:06d>.png`` person masks.
* ``features/geometry/<shard>/<scene>/transform.npz`` — per-frame
  original-resolution intrinsics and METRIC ``cam_from_world`` extrinsics.
* ``features/human_optim/<shard>/<scene>/contacts_<level>.npz`` — 52-joint
  automatic contact labels with per-joint confidence (train).
* ``features/annotation/<shard>/<scene>/annotation.npz`` — the manual tri-state
  labels of the test split.

Labels are folded 52 -> 22 SMPL-X body joints (each hand ORs its wrist and 15
finger joints) and then onto the six kindyn contact/force groups. Train labels
are supervised wherever the person is tracked; test labels only where the
annotator marked the joint.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

#: Curated-corpus filter: boulder scenes a human kept, VLM-classed as climbing
#: or bouldering, not rope supported. ``dataset_split`` is the DB's assignment.
_SCENE_QUERY = (
    "SELECT scene_id FROM scenes WHERE human_selected=1 AND vlm_category IN (1,2) "
    "AND vlm_rope_supported=0 AND dataset_split=?{camera} ORDER BY scene_id"
)
#: ``camera`` filter of :func:`list_scenes` -> SQL clause on the DB's per-scene
#: ``static_camera`` flag (129 static / 843 moving curated scenes; the static
#: set is 113 train + 16 annotated test scenes).
_CAMERA_CLAUSE = {"all": "", "static": " AND static_camera=1", "moving": " AND static_camera=0"}

#: 52 SMPLXMid joints = 22 body (0-21) + 30 fingers (22-51).
N_JOINTS_52 = 52
NUM_BODY_22 = 22
LEFT_HAND_GROUP_52 = (20,) + tuple(range(22, 37))
RIGHT_HAND_GROUP_52 = (21,) + tuple(range(37, 52))
_HAND_FOLDS = ((20, LEFT_HAND_GROUP_52), (21, RIGHT_HAND_GROUP_52))

#: Pinned label schema of ``contacts_<level>.npz``.
CONTACT_LABEL_SCHEMA = 2

#: The six contact/force groups, in kindyn's ``contact_force_joints`` column
#: order. ``*_foot`` is the big-toe joint, ``*_ankle`` the heel.
GROUP_NAMES = (
    "left_hand", "right_hand", "left_foot", "right_foot", "left_ankle", "right_ankle",
)
NUM_GROUPS = len(GROUP_NAMES)
#: Body-22 source joint of each group: hands are the wrists (fingers are folded
#: there by the 52->22 fold), the toe groups the foot joints, the heels the ankles.
GROUP_BODY22 = ((20,), (21,), (10,), (11,), (7,), (8,))
#: MHR70 keypoint anchors of the same six groups.

#: Joints an annotator can label in the manual test set.
OBSERVABLE_14 = [1, 2, 4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21]
#: The manual protocol does not expose these; on a reviewed frame the schema
#: defines them as non-contact rather than unknown.
ALWAYS_NON_CONTACT_8 = [0, 3, 6, 9, 12, 13, 14, 15]

#: The annotator's 14 joints -> SMPL-X 22-body index, by name.
ANNOTATION_TO_SMPLX22 = {
    "left_hand": 20, "right_hand": 21, "left_foot": 10, "right_foot": 11,
    "left_ankle": 7, "right_ankle": 8, "left_knee": 4, "right_knee": 5,
    "left_elbow": 18, "right_elbow": 19, "left_shoulder": 16, "right_shoulder": 17,
    "left_hip": 1, "right_hip": 2,
}

#: Fallback world down direction, used only by scenes loaded without kindyn.
GRAVITY_WORLD = np.array([0.0, 1.0, 0.0], np.float32)


def scene_shard(scene: str) -> str:
    """Two-level ``<s[0:2]>/<s[2:4]>`` shard prefix used throughout the corpus."""
    return f"{scene[0:2]}/{scene[2:4]}"


def list_scenes(root: str | Path, dataset_split: str, camera: str = "all") -> list[str]:
    """Sorted curated scene ids of one DB ``dataset_split`` (``train``/``test``).

    :param camera: ``all`` | ``static`` | ``moving`` — the DB's ``static_camera`` flag.
    """
    if camera not in _CAMERA_CLAUSE:
        raise ValueError(f"camera must be one of {sorted(_CAMERA_CLAUSE)}; got {camera!r}")
    db_path = Path(root) / "scenes" / "scenes.db"
    if not db_path.is_file():
        raise FileNotFoundError(f"no scene database at {db_path}")
    query = _SCENE_QUERY.format(camera=_CAMERA_CLAUSE[camera])
    with sqlite3.connect(db_path) as db:
        return [row[0] for row in db.execute(query, (dataset_split,))]


def list_train_scenes(root: str | Path, camera: str = "all") -> list[str]:
    """Sorted curated scene ids of the DB's train split."""
    return list_scenes(root, "train", camera)


def list_test_scenes(root: str | Path, camera: str = "all") -> list[str]:
    """Test scenes whose manual ``annotation.npz`` exists (labels are available)."""
    corpus = Path(root)
    return [
        scene for scene in list_scenes(corpus, "test", camera)
        if (corpus / "features" / "annotation" / scene_shard(scene) / scene
            / "annotation.npz").is_file()
    ]


def embedding_path(
    embedding_dir: str | Path, scene: str, object_id: int, position: int,
) -> Path:
    """Cache file of one person-frame crop's frozen-backbone embedding.

    ``<embedding_dir>/<shard>/<scene>/<oid:02d>/<pos:06d>.npy`` — an int16 bit
    view of the bf16 ``[1280, 32, 32]`` backbone output. Read back with
    ``torch.from_numpy(np.load(path)).view(torch.bfloat16)``.
    """
    return (Path(embedding_dir) / scene_shard(scene) / scene
            / f"{object_id:02d}" / f"{position:06d}.npy")


def rows_by_object_id(
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


def merge_contacts_52_to_22(
    jc52: np.ndarray, conf52: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold 52-joint contacts + confidence into the 22 SMPL-X body joints.

    Joints 0-21 pass through; each hand ORs the wrist and its 15 fingers. The
    folded confidence is the strongest *touching* member when the hand is in
    contact and the weakest member when it is free (one occluded finger makes
    the whole-hand free label uncertain).

    A non-finite confidence entry means the joint is **not assessed** (spine1 /
    spine3 / neck and every individual finger joint are NaN on all frames; NaN
    never coincides with a positive label). Unassessed entries cast no vote in
    the hand folds and become confidence ``0.0`` for the pass-through joints.

    :param jc52: ``(P, N, 52)`` bool contact labels.
    :param conf52: ``(P, N, 52)`` float32 confidence in ``[0, 1]`` (non-finite
        = not assessed).
    :returns: ``(jc22 (P, N, 22) bool, conf22 (P, N, 22) float32)``.
    """
    jc52 = np.asarray(jc52, bool)
    conf52 = np.asarray(conf52, np.float32)
    if jc52.ndim != 3 or jc52.shape[-1] != N_JOINTS_52:
        raise ValueError(f"joint_contact must have shape (P,N,52), got {jc52.shape}")
    if conf52.shape != jc52.shape:
        raise ValueError(
            f"label_confidence shape {conf52.shape} does not match contacts {jc52.shape}")
    finite = np.isfinite(conf52)
    if bool(((conf52 < 0.0) | (conf52 > 1.0))[finite].any()):
        raise ValueError("finite label_confidence must be within [0, 1]")

    jc22 = jc52[..., :NUM_BODY_22].copy()
    conf22 = np.where(finite, conf52, 0.0)[..., :NUM_BODY_22].copy()
    for wrist, group in _HAND_FOLDS:
        sub_lbl = jc52[..., group]
        sub_conf = conf52[..., group]
        sub_fin = finite[..., group]
        lbl = sub_lbl.any(axis=-1)
        conf_touch = np.where(sub_lbl & sub_fin, sub_conf, -np.inf).max(axis=-1)
        conf_free = np.where(sub_fin, sub_conf, np.inf).min(axis=-1)
        folded = np.where(lbl, conf_touch, conf_free)
        jc22[..., wrist] = lbl
        conf22[..., wrist] = np.where(np.isfinite(folded), folded, 0.0)
    return jc22, conf22


def reduce_body22_to_groups(
    contact: np.ndarray, supervised: np.ndarray, confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce body-22 labels onto :data:`GROUP_BODY22` with tri-state OR semantics.

    A group is a known positive when any *supervised* member is positive, even
    if another member is unknown. It is a known negative only when every member
    is supervised and free; a partial negative stays ignored. Positive
    confidence is the maximum over supervised positive members, known-free
    confidence the mean over all members. A single-member group (every kindyn
    group) degenerates to a passthrough of its joint's label and confidence,
    zeroed where the joint is unsupervised.

    :param contact: ``(..., 22)`` labels (``> 0.5`` is contact).
    :param supervised: ``(..., 22)`` supervision mask (``> 0`` is supervised).
    :param confidence: ``(..., 22)`` confidence; non-finite counts as ``0``.
    :returns: ``(contact_6, supervised_6, confidence_6)`` float32.
    """
    contact = np.asarray(contact)
    supervised = np.asarray(supervised)
    confidence = np.asarray(confidence)
    if contact.shape != supervised.shape or contact.shape != confidence.shape:
        raise ValueError(
            "body-22 contact/supervised/confidence shapes must match; got "
            f"{contact.shape}, {supervised.shape}, {confidence.shape}")
    if contact.shape[-1] != NUM_BODY_22:
        raise ValueError(f"body-22 reduction expects (..., 22); got {contact.shape}")

    is_contact = contact > 0.5
    is_supervised = supervised > 0
    conf = np.clip(np.nan_to_num(
        confidence.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    out_contact, out_supervised, out_confidence = [], [], []
    for group in GROUP_BODY22:
        idx = list(group)
        group_contact = is_contact[..., idx]
        group_supervised = is_supervised[..., idx]
        group_conf = conf[..., idx]
        supervised_positive = group_contact & group_supervised
        positive = supervised_positive.any(axis=-1)
        all_known = group_supervised.all(axis=-1)
        positive_conf = np.where(supervised_positive, group_conf, -np.inf).max(axis=-1)
        free_conf = group_conf.mean(axis=-1)
        out_contact.append(positive)
        out_supervised.append(positive | all_known)
        out_confidence.append(np.where(
            positive, positive_conf, np.where(all_known, free_conf, 0.0)))
    return (
        np.stack(out_contact, axis=-1).astype(np.float32),
        np.stack(out_supervised, axis=-1).astype(np.float32),
        np.stack(out_confidence, axis=-1).astype(np.float32),
    )


def _annotation_to_22(
    ann_contacts: np.ndarray, ann_names: list[str], ann_oids: list[int],
    ann_ignored: np.ndarray, data_oids: list[int], n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a manual tri-state annotation onto the 22 SMPL-X body joints.

    Labels map by joint *name*, people match by object id, unlabeled (-1)
    joint-frames and the 8 joints absent from the manual schema stay
    unannotated, and ignored people are zeroed entirely.

    :param ann_contacts: ``(P_ann, 14, N)`` int8 tri-state (-1 / 0 / 1).
    :returns: ``(joint_contact (P, N, 22) bool, annotated (P, N, 22) bool)`` in
        dataset person order.
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
        if row is None or ann_ignored[row]:      # never annotated / annotator ignored
            continue
        for name, sidx in ANNOTATION_TO_SMPLX22.items():
            tri = ann_contacts[row, ann_col[name]]
            jc[person, :, sidx] = tri == 1
            annotated[person, :, sidx] = tri != -1
    return jc, annotated


def _load_test_labels(
    root: Path, scene: str, object_ids: np.ndarray, n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Manual tri-state labels of one test scene, mapped to 22 joints."""
    path = root / "features" / "annotation" / scene_shard(scene) / scene / "annotation.npz"
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
        ann["contacts"], [str(x) for x in ann["joint_names"]],
        [int(x) for x in ann["object_ids"]], ann["ignored"],
        [int(x) for x in object_ids], n)
    reviewed = annotated[..., OBSERVABLE_14].any(axis=-1)
    annotated[..., ALWAYS_NON_CONTACT_8] |= reviewed[..., None]
    return joint_contact, annotated


def load_scene(root: Path, scene: str, split: str, contact_level: int) -> dict:
    """Frames, masks, boxes, cameras and six-group contact labels of one scene.

    :param split: ``"train"`` (automatic ``contacts_<level>`` labels, supervised
        wherever the person is tracked) or ``"test"`` (manual annotation).
    :returns: the scene dict :class:`~data.base.ClipDataset` indexes — camera
        arrays, ``valid_mask``, ``fps`` and ``contact_gt``/``contact_valid``/
        ``contact_conf`` ``(P, N, 6)``.
    """
    features = root / "features"
    shard = scene_shard(scene)
    human_dir = features / "human_optim" / shard / scene
    sam3_dir = features / "sam3" / shard / scene
    contacts = np.load(human_dir / f"contacts_{contact_level}.npz", allow_pickle=True)
    boxes = np.load(sam3_dir / "bboxes.npz", allow_pickle=True)
    transform = np.load(
        features / "geometry" / shard / scene / "transform.npz", allow_pickle=True)

    n = int(contacts["num_frames"])
    object_ids = np.asarray(contacts["object_ids"], np.int64).reshape(-1)
    n_people = len(object_ids)
    valid_mask = np.asarray(contacts["valid_mask"], bool)                  # [P, N]
    intrinsics = np.asarray(transform["intrinsics_px_orig"], np.float32)   # [N, 3, 3]
    extrinsics = np.asarray(transform["extrinsics"], np.float32)           # [N, 4, 4]
    bbox = rows_by_object_id(
        np.asarray(boxes["bboxes_per_obj"], np.float32),
        boxes["object_ids"], object_ids, scene, "sam3 bbox track")         # [P, N, 4]

    if valid_mask.shape != (n_people, n):
        raise ValueError(
            f"{scene}: valid_mask {valid_mask.shape} does not match ({n_people}, {n})")
    if bbox.shape != (n_people, n, 4):
        raise ValueError(
            f"{scene}: bboxes_per_obj {bbox.shape} does not match ({n_people}, {n}, 4)")
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

    # A tracked frame whose box is degenerate cannot be cropped — demote it to
    # invalid rather than failing the scene.
    bbox_good = (
        np.isfinite(bbox).all(axis=-1)
        & (bbox[..., 2] > bbox[..., 0])
        & (bbox[..., 3] > bbox[..., 1])
    )
    valid_mask = valid_mask & bbox_good

    if split == "train":
        schema = int(np.asarray(contacts["contact_label_schema"]).item())
        if schema != CONTACT_LABEL_SCHEMA:
            raise ValueError(
                f"{scene}: contacts_{contact_level} contact_label_schema={schema}, "
                f"expected {CONTACT_LABEL_SCHEMA}")
        joint_contact, conf22 = merge_contacts_52_to_22(
            contacts["joint_contact"], contacts["joint_label_confidence"])
        supervised22 = np.broadcast_to(
            valid_mask[..., None], joint_contact.shape).astype(np.float32)
    else:
        joint_contact, annotated = _load_test_labels(root, scene, object_ids, n)
        conf22 = np.ones(joint_contact.shape, np.float32)
        supervised22 = (valid_mask[..., None] & annotated).astype(np.float32)
    contact_gt, contact_valid, contact_conf = reduce_body22_to_groups(
        joint_contact.astype(np.float32), supervised22, conf22)

    # World camera centres C = -R^T t; the per-clip jump is measured between
    # consecutive SAMPLED frames in the dataset.
    cam_centers = -np.einsum(
        "nji,nj->ni", extrinsics[:, :3, :3], extrinsics[:, :3, 3]).astype(np.float32)

    return {
        "human_dir": human_dir,
        "frames_dir": root / "frames" / shard / scene,
        "mask_dir": sam3_dir,
        "object_ids": object_ids,
        "frame_indices": np.arange(n, dtype=np.int64),
        "bbox": bbox,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "gravity_world": GRAVITY_WORLD.copy(),
        "cam_centers": cam_centers,
        "valid_mask": valid_mask,
        "fps": float(contacts["fps"]),
        "contact_gt": contact_gt,                                     # [P, N, 6]
        "contact_valid": contact_valid,                               # [P, N, 6]
        "contact_conf": contact_conf,                                 # [P, N, 6]
    }
