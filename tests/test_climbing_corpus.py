"""Tests for the corpus-direct ClimbingVideos training dataset."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]

from contact.config import load_config
from contact.data.climbing_corpus import (
    FORCE_GROUPS_52,
    COND_FEATURE_DIM,
    MOTION_MHR70_INDICES,
    NUM_MHR70,
    NUM_MHR_BONES,
    NUM_MHR_SCALES,
    NUM_SUP_VERTICES,
    ClimbingCorpusDataset,
    FORCE_GROUP_NAMES,
    GRAVITY_WORLD,
    KINDYN_FORCE_JOINTS,
    KP_JOINT_NAMES,
    cond_feature_rows,
    fold_force_group_contact,
    list_annotated_test_scenes,
    list_corpus_scenes,
    merge_contacts_52_to_22,
    quat_xyzw_to_matrix,
    scene_shard,
)
from contact.data.collate import make_collate, make_loaders
from contact.targets import TargetSpec

CORPUS = Path("/data3/rikhat.akizhanov/better/data/ClimbingVideos")
requires_corpus = pytest.mark.skipif(
    not (CORPUS / "scenes" / "scenes.db").is_file(),
    reason="ClimbingVideos corpus not available")

N_FRAMES = 8
FPS = 25.0
GRAVITY_MAG = 9.81
# Root rotation 90 deg about +z, xyzw (matches the kindyn q[3:7] layout).
QUAT_Z90 = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], np.float32)
#: 52-joint (SMPL-X body) indices of KP_JOINT_NAMES, same order — the synthetic
#: kindyn joint axis is named with them so the loader can resolve the keypoints.
KP_JOINT_INDICES = (16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8, 12)


# ------------------------------------------------------------------ fixture

def _default_labels52(n_people: int, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """52-joint labels exercising the hand fold: finger-only left-hand contact."""
    jc = np.zeros((n_people, n_frames, 52), bool)
    conf = np.full((n_people, n_frames, 52), 0.9, np.float32)
    jc[..., 25] = True          # a left finger touches; the wrist itself does not
    conf[..., 25] = 0.7
    conf[..., 20] = 0.2
    conf[..., 40] = 0.15        # weakest member of the free right hand
    jc[..., 10] = True          # left foot (big toe) touches
    conf[..., 10] = 0.6
    conf[..., 7] = 0.55         # left ankle free
    return jc, conf


def _write_mhr(
    human: Path, oids: np.ndarray, valid: np.ndarray, *,
    nan_frames: tuple[int, ...] = (),
) -> None:
    """Write synthetic ``mhr_1.npz`` (converter v3) + ``mhr_sup_1.npz`` (schema 1).

    Every array is an exact affine ramp in (person, frame, element) so the tests
    can assert values rather than shapes:

    * ``kp_world[p, n, k]  = [1 + 0.01 k, 2 + 0.1 n, 3 + p]``
    * ``verts_world[p,n,v] = [4 + 0.001 v, 5 + 0.1 n, 6 + p]``
    * ``lbs_params[p, n]`` — bones (130..135) ``0.1 (1 + j) + 0.01 n``,
      scales (136..203) ``0.02 (1 + j) + p`` (per-person CONSTANT, as in the
      real archives)
    * ``fit_err_cm[p, n] = 0.5 + 0.1 n``

    ``nan_frames`` NaNs those frames of ``kp_world``/``verts_world``, which is
    how the real files mark rows the fit did not cover.
    """
    n_people, n_frames = valid.shape
    persons = np.arange(n_people, dtype=np.float32).reshape(n_people, 1, 1)
    frames_f = np.arange(n_frames, dtype=np.float32).reshape(1, n_frames, 1)
    zeros3 = np.zeros((n_people, n_frames, 1), np.float32)

    k_idx = np.arange(NUM_MHR70, dtype=np.float32).reshape(1, 1, NUM_MHR70)
    kp = np.stack(np.broadcast_arrays(
        1.0 + 0.01 * k_idx + zeros3, 2.0 + 0.1 * frames_f + 0.0 * k_idx,
        3.0 + persons + 0.0 * k_idx), axis=-1).astype(np.float32)
    v_idx = np.arange(NUM_SUP_VERTICES, dtype=np.float32).reshape(
        1, 1, NUM_SUP_VERTICES)
    verts = np.stack(np.broadcast_arrays(
        4.0 + 0.001 * v_idx + zeros3, 5.0 + 0.1 * frames_f + 0.0 * v_idx,
        6.0 + persons + 0.0 * v_idx), axis=-1).astype(np.float32)
    for f in nan_frames:
        kp[:, f] = np.nan
        verts[:, f] = np.nan

    lbs = np.zeros((n_people, n_frames, 204), np.float32)
    lbs[..., 130:136] = (
        0.1 * (1.0 + np.arange(NUM_MHR_BONES, dtype=np.float32)) + 0.01 * frames_f)
    lbs[..., 136:204] = (
        0.02 * (1.0 + np.arange(NUM_MHR_SCALES, dtype=np.float32)) + persons)
    q_world = np.zeros((n_people, n_frames, 132), np.float32)
    # A deliberately foot-level free-flyer root: the loader must NOT use it as
    # the motion root (the real MHR root sits ~0.93 m from the hips).
    q_world[..., 1] = 9.0 + 0.1 * frames_f[..., 0]
    q_world[..., 3:7] = QUAT_Z90
    np.savez(
        human / "mhr_1.npz", object_ids=oids, q_world=q_world, valid_mask=valid,
        identity=(0.03 * (1.0 + np.arange(45, dtype=np.float32))
                  + persons[:, :, 0]).astype(np.float32),
        lbs_params=lbs,
        fit_err_cm=np.broadcast_to(
            0.5 + 0.1 * frames_f[..., 0], (n_people, n_frames)).astype(np.float32),
        num_frames=np.int32(n_frames), fps=np.float32(FPS),
        converter_version=np.int32(3))
    np.savez(
        human / "mhr_sup_1.npz", kp_world=kp, verts_world=verts,
        vert_indices=(np.arange(NUM_SUP_VERTICES, dtype=np.int64) * 7),
        kp_vs_kindyn_med_cm=np.full(n_people, 3.1, np.float32),
        schema_version=np.int32(1), source_converter_version=np.int32(3),
        num_frames=np.int32(n_frames), fps=np.float32(FPS))


def _write_scene(
    root: Path,
    sid: str,
    *,
    n_people: int = 1,
    labels52: tuple[np.ndarray, np.ndarray] | None = None,
    invalid_frames: tuple[int, ...] = (),
    kindyn_id_order: list[int] | None = None,
    with_annotation: bool = False,
    mhr_nan_frames: tuple[int, ...] = (),
) -> None:
    """Write one synthetic corpus scene (features + frames + masks)."""
    shard = scene_shard(sid)
    oids = np.arange(n_people, dtype=np.int32)
    human = root / "features" / "human_optim" / shard / sid
    sam3 = root / "features" / "sam3" / shard / sid
    geom = root / "features" / "geometry" / shard / sid
    frames = root / "frames" / shard / sid
    for d in (human, sam3, geom, frames):
        d.mkdir(parents=True, exist_ok=True)

    jc52, conf52 = labels52 if labels52 is not None else _default_labels52(n_people, N_FRAMES)
    valid = np.ones((n_people, N_FRAMES), bool)
    for f in invalid_frames:
        valid[:, f] = False
        jc52[:, f] = False
    contacts = dict(
        num_frames=np.int64(N_FRAMES), object_ids=oids, valid_mask=valid,
        fps=np.float32(FPS), joint_contact=jc52, joint_label_confidence=conf52,
        contact_label_schema=np.int32(2),
        frame_indices=np.arange(N_FRAMES, dtype=np.int32),
    )
    np.savez(human / "contacts_1.npz", **contacts)
    np.savez(human / "contacts_2.npz", **contacts)

    # kindyn (2026-08-27 schema): per-contact-frame forces in newtons, world
    # frame, zero exactly off the frame_contact mask. The left hand splits its
    # 0.1 bw across palm + finger (the fold must sum them into the wrist); the
    # knee frame maps to NO group and must be dropped by the loader.
    group_contact = fold_force_group_contact(jc52)                      # [P, N, 6]
    masses = 60.0 + 10.0 * np.arange(n_people, dtype=np.float32)        # per-person kg
    cframe_names = np.array(
        ["palm_left", "finger_left", "palm_right", "toe_left", "heel_left",
         "toe_right", "heel_right", "knee_left"], "<U16")
    cframe_parents = np.array([20, 27, 21, 10, 7, 11, 8, 4], np.int64)
    n_cf = len(cframe_names)
    frame_contact = np.zeros((n_people, N_FRAMES, n_cf), bool)
    frame_contact[..., 0] = group_contact[..., 0]                       # palm_left
    frame_contact[..., 1] = group_contact[..., 0]                       # finger_left
    frame_contact[..., 3] = group_contact[..., 2]                       # toe_left
    frame_contact[..., 7] = True                                        # knee (dropped)
    frame_forces = np.zeros((n_people, N_FRAMES, n_cf, 3), np.float32)
    for p in range(n_people):
        bw = masses[p] * GRAVITY_MAG
        frame_forces[p, :, 0] = np.array([0.06 * bw, 0.0, 0.0], np.float32)
        frame_forces[p, :, 1] = np.array([0.04 * bw, 0.0, 0.0], np.float32)
        frame_forces[p, :, 3] = np.array([0.0, -0.5 * bw, 0.0], np.float32)
        frame_forces[p, :, 7] = np.array([0.0, -0.2 * bw, 0.0], np.float32)
    frame_forces[~frame_contact] = 0.0
    q = np.zeros((n_people, N_FRAMES, 211), np.float32)
    q[..., 3:7] = QUAT_Z90
    # World joints for the lever arms: everything at the pelvis except a known
    # world offset for the left wrist (+0.5 x) and the left foot (+0.8 y).
    joint_names = [f"joint_{j}" for j in range(52)]
    joint_names[0] = "pelvis"
    for name, jidx in zip(KINDYN_FORCE_JOINTS, (20, 21, 10, 11, 7, 8)):
        joint_names[jidx] = name
    for name, jidx in zip(KP_JOINT_NAMES, KP_JOINT_INDICES):
        joint_names[jidx] = name
    joints_world = np.tile(
        np.array([1.0, 2.0, 3.0], np.float32), (n_people, N_FRAMES, 52, 1))
    joints_world[..., 20, 0] += 0.5                                     # left wrist
    joints_world[..., 10, 1] += 0.8                                     # left foot (toe)
    joints_world[..., 12, 2] += 0.4                                     # neck (keypoints)
    order = list(range(n_people)) if kindyn_id_order is None else kindyn_id_order
    np.savez(
        human / "kindyn_1.npz",
        object_ids=oids[order],
        contact_frame_names=cframe_names, contact_frame_parents=cframe_parents,
        frame_forces=frame_forces[order], frame_contact=frame_contact[order],
        force_confidence=np.full((n_people, N_FRAMES), 0.9, np.float32)[order],
        gravity_world=np.array([0.0, 1.0, 0.0], np.float32),
        q=q[order], total_mass=masses[order],
        num_frames=np.int64(N_FRAMES), fps=np.float32(FPS),
        valid_mask=valid[order], joint_contact=jc52[order],
        joint_names=np.array(joint_names), joints_world=joints_world[order],
        betas=np.zeros((n_people, 10), np.float32),
    )

    _write_mhr(human, oids, valid, nan_frames=mhr_nan_frames)

    bbox = np.tile(np.array([2, 3, 20, 30], np.float32), (n_people, N_FRAMES, 1))
    np.savez(sam3 / "bboxes.npz", bboxes_per_obj=bbox.astype(np.int32),
             object_ids=oids, frame_indices=np.arange(N_FRAMES, dtype=np.int32))
    extr = np.tile(np.eye(4, dtype=np.float32), (N_FRAMES, 1, 1))
    extr[:, 2, 3] = -0.1 * np.arange(N_FRAMES, dtype=np.float32)        # camera centre z = 0.1k
    np.savez(geom / "transform.npz",
             intrinsics_px_orig=np.tile(np.eye(3, dtype=np.float32) * 100.0, (N_FRAMES, 1, 1)),
             extrinsics=extr, frame_indices=np.arange(N_FRAMES, dtype=np.int32),
             fps=np.float32(FPS), metric=np.bool_(True))

    rng = np.random.default_rng(0)
    for pos in range(N_FRAMES):
        Image.fromarray(rng.integers(0, 255, (32, 24, 3), np.uint8)).save(
            frames / f"{pos:06d}.jpg")
        for oid in oids:
            (sam3 / f"{int(oid):02d}").mkdir(exist_ok=True)
            Image.fromarray(np.full((32, 24), 255, np.uint8)).save(
                sam3 / f"{int(oid):02d}" / f"frame_{pos:06d}.png")

    if with_annotation:
        ann_dir = root / "features" / "annotation" / shard / sid
        ann_dir.mkdir(parents=True, exist_ok=True)
        names = ("left_hand", "right_hand", "left_foot", "right_foot", "left_ankle",
                 "right_ankle", "left_knee", "right_knee", "left_elbow", "right_elbow",
                 "left_shoulder", "right_shoulder", "left_hip", "right_hip")
        tri = np.zeros((n_people, 14, N_FRAMES), np.int8)
        tri[0, 0] = 1                                       # left hand in contact
        tri[0, 2] = -1                                      # left foot unlabeled
        ignored = np.zeros(n_people, bool)
        if n_people > 1:
            ignored[1] = True
        np.savez(ann_dir / "annotation.npz",
                 contacts=tri, joint_names=np.array(names), object_ids=oids,
                 ignored=ignored, annotation_version=np.int32(2),
                 num_frames=np.int64(N_FRAMES))


def _make_db(root: Path, rows: list[tuple]) -> None:
    (root / "scenes").mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(root / "scenes" / "scenes.db") as db:
        db.execute(
            "CREATE TABLE scenes (scene_id TEXT PRIMARY KEY, video_id TEXT, "
            "dataset_split TEXT, human_selected INTEGER, vlm_category INTEGER, "
            "vlm_rope_supported INTEGER)")
        db.executemany("INSERT INTO scenes VALUES (?,?,?,?,?,?)", rows)


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """A minimal synthetic corpus: 3 curated train scenes (2 videos), 1 test scene."""
    root = tmp_path / "corpus"
    _make_db(root, [
        ("vidA_0000", "vidA", "train", 1, 1, 0),
        ("vidA_0001", "vidA", "train", 1, 2, 0),
        ("vidB_0000", "vidB", "train", 1, 1, 0),
        ("vidC_0000", "vidC", "test", 1, 1, 0),
        ("vidX_0000", "vidX", "train", 0, 1, 0),   # not human-selected
        ("vidY_0000", "vidY", "train", 1, 3, 0),   # wrong vlm category
        ("vidZ_0000", "vidZ", "train", 1, 1, 1),   # rope supported
    ])
    for sid in ("vidA_0000", "vidA_0001", "vidB_0000"):
        _write_scene(root, sid)
    _write_scene(root, "vidC_0000", with_annotation=True)
    return root


# ------------------------------------------------------------------ discovery / split

def test_scene_discovery_applies_curated_filter(corpus):
    assert list_corpus_scenes(corpus, "train") == ["vidA_0000", "vidA_0001", "vidB_0000"]
    assert list_corpus_scenes(corpus, "test") == ["vidC_0000"]
    assert list_annotated_test_scenes(corpus) == ["vidC_0000"]


def test_grouped_split_never_straddles_a_video(corpus):
    kwargs = dict(frames_per_clip=2, frame_stride=2, seed=42, val_fraction=0.5)
    train = ClimbingCorpusDataset(corpus, split="train", jitter=False, **kwargs)
    val = ClimbingCorpusDataset(corpus, split="val", **kwargs)
    train_scenes, val_scenes = set(train._scenes), set(val._scenes)
    assert train_scenes | val_scenes == {"vidA_0000", "vidA_0001", "vidB_0000"}
    assert not train_scenes & val_scenes
    # round(2 videos * 0.5) = 1 held-out video; vidA's chunks stay together.
    vids = {"vidA_0000", "vidA_0001"}
    assert vids <= train_scenes or vids <= val_scenes


# ------------------------------------------------------------------ items

def test_item_contract_and_dtypes(corpus):
    ds = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="train", frames_per_clip=2,
        frame_stride=2, jitter=False, load_forces=True)
    clip = ds[0]
    assert isinstance(clip, list) and len(clip) == 2
    frame = clip[0]
    assert set(frame) == {
        "image", "img_wh", "mask", "bbox", "cam_int", "cam_from_world",
        "gravity_world", "cam_jump_m", "joint_contact", "joint_mask",
        "joint_supervised", "joint_confidence", "frame_pos_sec",
        "frame_position", "frame_index", "frame_valid", "key", "dataset",
        "force_gt", "force_contact", "force_lever", "force_valid", "force_conf",
    }
    assert frame["img_wh"] is None                 # only set on embedding-cache rows
    assert frame["image"].shape == (32, 24, 3) and frame["image"].dtype == np.uint8
    assert frame["mask"].shape == (32, 24)
    assert frame["bbox"].shape == (4,)
    assert frame["cam_int"].shape == (3, 3) and frame["cam_int"].dtype == np.float32
    assert frame["cam_from_world"].shape == (4, 4)
    np.testing.assert_array_equal(frame["gravity_world"], GRAVITY_WORLD)
    for key in ("joint_contact", "joint_mask", "joint_supervised", "joint_confidence"):
        assert frame[key].shape == (22,) and frame[key].dtype == torch.float32
    assert frame["force_gt"].shape == (6, 3) and frame["force_gt"].dtype == torch.float32
    assert frame["force_contact"].shape == (6,) and frame["force_contact"].dtype == torch.bool
    assert frame["force_lever"].shape == (6, 3) and frame["force_lever"].dtype == torch.float32
    assert frame["force_valid"] is True
    assert frame["force_conf"] == pytest.approx(0.9)
    assert frame["key"] == "vidA_0000#0@0" and frame["dataset"] == "climbing_corpus"
    # cam centres move 0.1 m per source frame -> 0.2 m per stride-2 sampled step.
    assert clip[0]["cam_jump_m"] == 0.0
    assert clip[1]["cam_jump_m"] == pytest.approx(0.2, abs=1e-6)
    assert clip[1]["frame_pos_sec"] == pytest.approx(2 / FPS)

    no_force = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="train", frames_per_clip=2,
        frame_stride=2, jitter=False)[0][0]
    assert "force_gt" not in no_force and "force_valid" not in no_force

    metadata_only = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="train", frames_per_clip=2,
        frame_stride=2, jitter=False, load_images=False)[0][0]
    assert metadata_only["image"] is None and metadata_only["mask"] is None


def test_hand_fold_confidence_rule(corpus):
    jc52, conf52 = _default_labels52(1, 1)
    jc22, conf22 = merge_contacts_52_to_22(jc52, conf52)
    # Left hand: finger 25 touches -> contact, confidence = strongest touching vote.
    assert bool(jc22[0, 0, 20]) and conf22[0, 0, 20] == pytest.approx(0.7)
    # Right hand free: confidence = weakest member of the 16-joint AND.
    assert not bool(jc22[0, 0, 21]) and conf22[0, 0, 21] == pytest.approx(0.15)
    # Body joints pass through.
    assert bool(jc22[0, 0, 10]) and conf22[0, 0, 10] == pytest.approx(0.6)
    assert conf22[0, 0, 7] == pytest.approx(0.55)

    frame = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="train", frames_per_clip=1,
        frame_stride=1, jitter=False)[0][0]
    assert float(frame["joint_contact"][20]) == 1.0
    assert float(frame["joint_confidence"][20]) == pytest.approx(0.7)
    assert float(frame["joint_confidence"][21]) == pytest.approx(0.15)
    # Confidence weighting multiplies into the loss mask only when requested.
    assert float(frame["joint_mask"][21]) == 1.0
    weighted = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="train", frames_per_clip=1,
        frame_stride=1, jitter=False, use_confidence_weights=True)[0][0]
    assert float(weighted["joint_mask"][21]) == pytest.approx(0.15)

    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        merge_contacts_52_to_22(jc52, conf52 + 2.0)

    # NaN = "not assessed" (new-schema fingers/spine): casts no vote in the
    # folds and passes through as confidence 0.0.
    conf_nan = conf52.copy()
    conf_nan[..., 22:37] = np.nan       # every left finger unassessed
    conf_nan[..., 3] = np.nan           # spine1 unassessed
    jc_nan = jc52.copy()
    jc_nan[..., 25] = False             # contact never coincides with NaN
    jc22n, conf22n = merge_contacts_52_to_22(jc_nan, conf_nan)
    assert conf22n[0, 0, 20] == pytest.approx(0.2)   # wrist = only assessed member
    assert conf22n[0, 0, 3] == 0.0


def test_contact_level_selects_npz(corpus):
    human = corpus / "features" / "human_optim" / "vi" / "dA" / "vidA_0000"
    with np.load(human / "contacts_2.npz", allow_pickle=True) as npz:
        override = dict(npz)
    override["joint_contact"] = np.zeros_like(override["joint_contact"])
    override["joint_contact"][..., 4] = True                # only the left knee
    np.savez(human / "contacts_2.npz", **override)
    frame = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="train", frames_per_clip=1,
        frame_stride=1, jitter=False, contact_level=2)[0][0]
    assert float(frame["joint_contact"][4]) == 1.0
    assert float(frame["joint_contact"][20]) == 0.0


# ------------------------------------------------------------------ windowing

def test_windowing_tiles_and_val_terminal_window(corpus):
    # N=8, T=3, stride=1: span=2, step=3, max_start=5 -> train bases [0, 3].
    train = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="train", frames_per_clip=3,
        frame_stride=1, jitter=False)
    assert [(base, rng) for _, _, base, rng, _ in train._items] == [(0, 3), (3, 3)]
    # Val appends the terminal window 5 so the tail frames are scored.
    val = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="val", frames_per_clip=3, frame_stride=1)
    assert [base for _, _, base, _, _ in val._items] == [0, 3, 5]
    positions = [f["frame_position"] for f in val[2]]
    assert positions == [5, 6, 7]
    assert [f["frame_pos_sec"] for f in val[2]] == pytest.approx([0.0, 1 / FPS, 2 / FPS])


def test_invalid_and_degenerate_frames_skip_windows(corpus):
    _write_scene(corpus, "vidB_0000", invalid_frames=(4,))
    ds = ClimbingCorpusDataset(
        corpus, scenes=["vidB_0000"], split="train", frames_per_clip=3,
        frame_stride=1, jitter=False)
    assert [base for _, _, base, _, _ in ds._items] == [0]  # window [3,4,5] skipped

    # A valid-tracked frame with a degenerate bbox is demoted to invalid.
    sam3 = corpus / "features" / "sam3" / "vi" / "dA" / "vidA_0000"
    with np.load(sam3 / "bboxes.npz", allow_pickle=True) as npz:
        boxes = dict(npz)
    boxes["bboxes_per_obj"] = boxes["bboxes_per_obj"].copy()
    boxes["bboxes_per_obj"][0, 4] = [10, 10, 10, 10]
    np.savez(sam3 / "bboxes.npz", **boxes)
    ds = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="train", frames_per_clip=3,
        frame_stride=1, jitter=False)
    assert not bool(ds._scenes["vidA_0000"]["valid_mask"][0, 4])
    assert [base for _, _, base, _, _ in ds._items] == [0]


def test_jitter_is_stateless_and_falls_back_over_gaps(corpus):
    _write_scene(corpus, "vidB_0000", invalid_frames=(6,))
    ds = ClimbingCorpusDataset(
        corpus, scenes=["vidB_0000"], split="train", frames_per_clip=2,
        frame_stride=1, jitter=True, seed=7)
    twin = ClimbingCorpusDataset(
        corpus, scenes=["vidB_0000"], split="train", frames_per_clip=2,
        frame_stride=1, jitter=True, seed=7)
    # Windows [0,1], [2,3], [4,5] survive; [6,7] contains the invalid frame.
    assert [base for _, _, base, _, _ in ds._items] == [0, 2, 4]
    for epoch in (0, 1, 5):
        ds.set_epoch(epoch)
        twin.set_epoch(epoch)
        for index, (_, _, base, jitter_range, _) in enumerate(ds._items):
            start = ds._window_start(base, jitter_range, index)
            assert base <= start < base + jitter_range
            assert start == twin._window_start(base, jitter_range, index)
    # Base 4 may jitter to 5, whose window [5, 6] crosses the gap -> falls back.
    index = 2
    for epoch in range(30):
        ds.set_epoch(epoch)
        if ds._window_start(4, ds._items[index][3], index) == 5:
            assert [f["frame_position"] for f in ds[index]] == [4, 5]
            break
    else:
        raise AssertionError("jitter never drew the boundary start; widen the search")


# ------------------------------------------------------------------ forces

def test_force_rotation_units_and_contact_mask(corpus):
    ds = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="train", frames_per_clip=1,
        frame_stride=1, jitter=False, load_forces=True)
    frame = ds[0][0]
    # Groups in contact: left hand (finger fold) and left foot.
    assert frame["force_contact"].tolist() == [True, False, True, False, False, False]
    # World forces 0.1 bw x / -0.5 bw y rotated by R(q)^T with q = Rz(90 deg):
    # x -> -y, y -> x.
    torch.testing.assert_close(
        frame["force_gt"][0], torch.tensor([0.0, -0.1, 0.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        frame["force_gt"][2], torch.tensor([-0.5, 0.0, 0.0]), atol=1e-6, rtol=0)
    assert frame["force_gt"][1].abs().sum() == 0.0          # free groups carry zero
    assert frame["force_valid"] is True
    assert FORCE_GROUP_NAMES == (
        "left_hand", "right_hand", "left_foot", "right_foot", "left_ankle", "right_ankle")


def test_force_lever_rotates_pelvis_offsets_to_root(corpus):
    """Lever arms = R(q)^T @ (group joint - pelvis), metres, root frame."""
    ds = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="train", frames_per_clip=1,
        frame_stride=1, jitter=False, load_forces=True)
    lever = ds[0][0]["force_lever"]
    # World offsets +0.5 x (left wrist) / +0.8 y (left foot) under Rz(90 deg)^T:
    # x -> -y, y -> x (same rotation as the forces).
    torch.testing.assert_close(
        lever[0], torch.tensor([0.0, -0.5, 0.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        lever[2], torch.tensor([0.8, 0.0, 0.0]), atol=1e-6, rtol=0)
    # The other four groups sit exactly at the pelvis in the fixture.
    assert float(lever[[1, 3, 4, 5]].abs().max()) == 0.0


def test_force_lever_group_resolution_is_by_name(corpus):
    """Permuting the kindyn joint axis (pelvis away from row 0) changes nothing;
    a missing group/pelvis name is a hard error."""
    human = corpus / "features" / "human_optim" / "vi" / "dA" / "vidA_0000"
    with np.load(human / "kindyn_1.npz", allow_pickle=True) as npz:
        kindyn = dict(npz)
    perm = np.roll(np.arange(52), 5)
    kindyn["joint_names"] = np.asarray(kindyn["joint_names"])[perm]
    kindyn["joints_world"] = np.asarray(kindyn["joints_world"])[:, :, perm]
    np.savez(human / "kindyn_1.npz", **kindyn)
    lever = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="train", frames_per_clip=1,
        frame_stride=1, jitter=False, load_forces=True)[0][0]["force_lever"]
    torch.testing.assert_close(
        lever[0], torch.tensor([0.0, -0.5, 0.0]), atol=1e-6, rtol=0)

    kindyn["joint_names"] = np.asarray(
        ["nope" if str(n) == "pelvis" else str(n) for n in kindyn["joint_names"]])
    np.savez(human / "kindyn_1.npz", **kindyn)
    with pytest.raises(ValueError, match="missing \\['pelvis'\\]"):
        ClimbingCorpusDataset(
            corpus, scenes=["vidA_0000"], split="train", frames_per_clip=1,
            frame_stride=1, jitter=False, load_forces=True)


def test_force_units_and_frame_options(corpus):
    """force_units='newtons' skips the bw division; force_frame='world' skips
    the root rotation (forces AND lever arms stay world vectors)."""
    ds = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="train", frames_per_clip=1,
        frame_stride=1, jitter=False, load_forces=True,
        force_frame="world", force_units="newtons")
    frame = ds[0][0]
    bw = 60.0 * GRAVITY_MAG
    torch.testing.assert_close(
        frame["force_gt"][0], torch.tensor([0.1 * bw, 0.0, 0.0]), atol=1e-3, rtol=0)
    torch.testing.assert_close(
        frame["force_gt"][2], torch.tensor([0.0, -0.5 * bw, 0.0]), atol=1e-3, rtol=0)
    torch.testing.assert_close(
        frame["force_lever"][0], torch.tensor([0.5, 0.0, 0.0]), atol=1e-6, rtol=0)
    with pytest.raises(ValueError, match="force_frame"):
        ClimbingCorpusDataset(corpus, scenes=["vidA_0000"], force_frame="camera")
    with pytest.raises(ValueError, match="force_units"):
        ClimbingCorpusDataset(corpus, scenes=["vidA_0000"], force_units="kg")


def test_fitted_gravity_replaces_the_constant(corpus):
    """The scene's gravity_world is kindyn's fitted vector (normalised); a
    non-unit stored vector is corrupt data."""
    human = corpus / "features" / "human_optim" / "vi" / "dA" / "vidA_0000"
    with np.load(human / "kindyn_1.npz", allow_pickle=True) as npz:
        kindyn = dict(npz)
    tilted = np.array([0.1, 0.99, -0.05], np.float32)
    kindyn["gravity_world"] = tilted
    np.savez(human / "kindyn_1.npz", **kindyn)
    frame = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="train", frames_per_clip=1,
        frame_stride=1, jitter=False, load_forces=True)[0][0]
    np.testing.assert_allclose(
        frame["gravity_world"], tilted / np.linalg.norm(tilted), atol=1e-6)

    kindyn["gravity_world"] = np.array([0.0, 5.0, 0.0], np.float32)
    np.savez(human / "kindyn_1.npz", **kindyn)
    with pytest.raises(ValueError, match="unit direction"):
        ClimbingCorpusDataset(
            corpus, scenes=["vidA_0000"], split="train", frames_per_clip=1,
            frame_stride=1, jitter=False, load_forces=True)


def test_force_valid_follows_kindyn_and_frame_validity(corpus):
    _write_scene(corpus, "vidB_0000", invalid_frames=(0,))
    ds = ClimbingCorpusDataset(
        corpus, scenes=["vidB_0000"], split="val", frames_per_clip=1,
        frame_stride=1, load_forces=True)
    by_pos = {clip[0]["frame_position"]: clip[0] for clip in (ds[i] for i in range(len(ds)))}
    assert 0 not in by_pos                                  # invalid frame -> no window
    assert by_pos[1]["force_valid"] is True


def test_nonzero_force_off_the_contact_mask_raises(corpus):
    human = corpus / "features" / "human_optim" / "vi" / "dA" / "vidA_0000"
    with np.load(human / "kindyn_1.npz", allow_pickle=True) as npz:
        kindyn = dict(npz)
    kindyn["frame_forces"] = kindyn["frame_forces"].copy()
    kindyn["frame_forces"][0, 0, 2] = [1.0, 0.0, 0.0]      # palm_right has no contact
    np.savez(human / "kindyn_1.npz", **kindyn)
    with pytest.raises(ValueError, match="no contact label"):
        ClimbingCorpusDataset(
            corpus, scenes=["vidA_0000"], split="train", frames_per_clip=1,
            frame_stride=1, jitter=False, load_forces=True)


def test_two_person_scene_aligns_kindyn_rows_by_object_id(corpus):
    # kindyn stores its person rows in reversed object order; masses differ, so a
    # row mix-up would change the body-weight normalisation.
    _write_scene(corpus, "vidB_0000", n_people=2, kindyn_id_order=[1, 0])
    ds = ClimbingCorpusDataset(
        corpus, scenes=["vidB_0000"], split="train", frames_per_clip=1,
        frame_stride=1, jitter=False, load_forces=True)
    frames = {}
    for i in range(len(ds)):
        frame = ds[i][0]
        if frame["frame_position"] == 0:
            frames[frame["key"]] = frame
    # Forces were written as fixed bw fractions per person, so after the correct
    # per-person mass normalisation both persons read identically.
    for key in ("vidB_0000#0@0", "vidB_0000#1@0"):
        torch.testing.assert_close(
            frames[key]["force_gt"][0], torch.tensor([0.0, -0.1, 0.0]),
            atol=1e-6, rtol=0)

    # A kindyn tree missing one person is corrupt -> hard error.
    human = corpus / "features" / "human_optim" / "vi" / "dB" / "vidB_0000"
    with np.load(human / "kindyn_1.npz", allow_pickle=True) as npz:
        kindyn = dict(npz)
    kindyn["object_ids"] = np.array([5, 0], np.int32)
    np.savez(human / "kindyn_1.npz", **kindyn)
    with pytest.raises(ValueError, match="no kindyn row"):
        ClimbingCorpusDataset(
            corpus, scenes=["vidB_0000"], split="train", frames_per_clip=1,
            frame_stride=1, jitter=False, load_forces=True)


# ------------------------------------------------------------------ keypoints

def test_keypoints_and_vertices_come_from_mhr_sup(corpus):
    """load_keypoints emits all 70 MHR70 keypoints and the vertex subset from
    mhr_sup_1.npz — the MHR-native GT — plus the mhr_1 mesh-fit residual."""
    ds = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="train", frames_per_clip=2,
        frame_stride=2, jitter=False, load_images=False, load_keypoints=True)
    frame = ds[0][0]                                     # scene frame 0, person 0
    assert frame["kp3d_world"].shape == (NUM_MHR70, 3)
    assert frame["kp3d_world"].dtype == torch.float32
    assert frame["vert_gt_world"].shape == (NUM_SUP_VERTICES, 3)
    assert frame["vert_indices"].shape == (NUM_SUP_VERTICES,)
    assert frame["vert_indices"].dtype == torch.int64
    assert frame["kp_valid"] is True and frame["vert_valid"] is True
    assert frame["cam_from_world"].shape == (4, 4)       # extrinsics ride along

    k = torch.arange(NUM_MHR70, dtype=torch.float32)
    expected_kp = torch.stack(
        [1.0 + 0.01 * k, torch.full((NUM_MHR70,), 2.0), torch.full((NUM_MHR70,), 3.0)],
        dim=-1)
    torch.testing.assert_close(frame["kp3d_world"], expected_kp, atol=1e-6, rtol=0)
    v = torch.arange(NUM_SUP_VERTICES, dtype=torch.float32)
    expected_v = torch.stack(
        [4.0 + 0.001 * v, torch.full((NUM_SUP_VERTICES,), 5.0),
         torch.full((NUM_SUP_VERTICES,), 6.0)], dim=-1)
    torch.testing.assert_close(frame["vert_gt_world"], expected_v, atol=1e-6, rtol=0)
    torch.testing.assert_close(
        frame["vert_indices"], torch.arange(NUM_SUP_VERTICES) * 7)
    assert frame["mhr_fit_err_cm"] == pytest.approx(0.5)

    plain = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="train", frames_per_clip=2,
        frame_stride=2, jitter=False, load_images=False)[0][0]
    for key in ("kp3d_world", "kp_valid", "vert_gt_world", "vert_valid",
                "vert_indices", "mhr_fit_err_cm"):
        assert key not in plain


def test_keypoint_path_never_reads_kindyn(corpus):
    """The MHR-native swap removed the last kindyn read from the keypoint path:
    the loader must work with kindyn_1.npz gone entirely."""
    human = corpus / "features" / "human_optim" / "vi" / "dA" / "vidA_0000"
    (human / "kindyn_1.npz").rename(human / "kindyn_1.npz.moved")
    frame = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="train", frames_per_clip=2,
        frame_stride=2, jitter=False, load_images=False, load_keypoints=True,
        load_pose=True)[0][0]
    assert frame["kp3d_world"].shape == (NUM_MHR70, 3)
    assert frame["pose_gt_bones"].shape == (NUM_MHR_BONES,)


def test_kp_and_vert_valid_follow_mhr_fit_coverage(corpus):
    """NaN rows of mhr_sup_1 (the frames the fit did not cover) come back as
    exact zeros with the bit cleared, independently per array."""
    human = corpus / "features" / "human_optim" / "vi" / "dA" / "vidA_0000"
    with np.load(human / "mhr_sup_1.npz") as npz:
        sup = dict(npz)
    sup["kp_world"] = sup["kp_world"].copy()
    sup["kp_world"][0, 1] = np.nan                       # keypoints lost on frame 1
    sup["verts_world"] = sup["verts_world"].copy()
    sup["verts_world"][0, 2] = np.nan                    # vertices lost on frame 2
    np.savez(human / "mhr_sup_1.npz", **sup)
    clip = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="train", frames_per_clip=N_FRAMES,
        frame_stride=1, jitter=False, load_images=False, load_keypoints=True)[0]
    assert [f["kp_valid"] for f in clip[:4]] == [True, False, True, True]
    assert [f["vert_valid"] for f in clip[:4]] == [True, True, False, True]
    assert float(clip[1]["kp3d_world"].abs().sum()) == 0.0
    assert float(clip[2]["vert_gt_world"].abs().sum()) == 0.0
    assert float(clip[3]["kp3d_world"].abs().sum()) > 0.0
    assert float(clip[3]["vert_gt_world"].abs().sum()) > 0.0


def test_mhr_sup_schema_is_pinned(corpus):
    """A schema bump is a hard error, never a silent shape mismatch."""
    human = corpus / "features" / "human_optim" / "vi" / "dA" / "vidA_0000"
    with np.load(human / "mhr_sup_1.npz") as npz:
        sup = dict(npz)
    sup["schema_version"] = np.int32(2)
    np.savez(human / "mhr_sup_1.npz", **sup)
    with pytest.raises(ValueError, match="mhr_sup_1 schema 2"):
        ClimbingCorpusDataset(
            corpus, scenes=["vidA_0000"], split="train", frames_per_clip=2,
            frame_stride=2, jitter=False, load_images=False, load_keypoints=True)


def test_pose_targets_carry_bones_scale_and_fit_err(corpus):
    """load_pose adds the GT bone-geometry slots (per-person median, constant on
    every frame), the 68 scale slots (per person, constant) and the fit residual."""
    clip = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="train", frames_per_clip=N_FRAMES,
        frame_stride=1, jitter=False, load_images=False, load_pose=True)[0]
    j = torch.arange(NUM_MHR_BONES, dtype=torch.float32)
    # The fixture ramps the bone slots by 0.01 per frame, but a body's proportions
    # do not change within a scene: that spread is converter fit freedom, so the
    # target is the person's MEDIAN over the 8 valid rows (median n = 3.5) served
    # unchanged on every frame.
    expected_bones = 0.1 * (1.0 + j) + 0.035
    for row in (0, 3, N_FRAMES - 1):
        torch.testing.assert_close(
            clip[row]["pose_gt_bones"], expected_bones, atol=1e-6, rtol=0)
    expected_scale = 0.02 * (1.0 + torch.arange(NUM_MHR_SCALES, dtype=torch.float32))
    for row in (0, 3, N_FRAMES - 1):                     # per-person CONSTANT
        torch.testing.assert_close(
            clip[row]["pose_gt_scale"], expected_scale, atol=1e-6, rtol=0)
    assert clip[0]["mhr_fit_err_cm"] == pytest.approx(0.5)
    assert clip[3]["mhr_fit_err_cm"] == pytest.approx(0.8)


def test_motion_root_source_mhr_uses_the_mhr_hips_and_root_rotation(corpus):
    """root_source='mhr' differentiates the (MEAN-HIPS position, root
    quaternion) free-flyer — the same construction motion_consistency builds
    from the prediction — not q_world's foot-level root."""
    common = dict(
        scenes=["vidA_0000"], split="train", frames_per_clip=N_FRAMES,
        frame_stride=1, jitter=False, load_images=False, load_motion=True,
        motion_joint_names=["pelvis"], motion_target_smooth_sec=0.0)
    kin = ClimbingCorpusDataset(corpus, motion_root_source="kindyn", **common)[0]
    mhr = ClimbingCorpusDataset(corpus, motion_root_source="mhr", **common)[0]
    frame = mhr[3]
    assert frame["motion_valid"] is True
    # The kindyn pelvis is static in the fixture -> zero velocity. The MHR hips
    # ramp +0.1 m/frame along world y = 2.5 m/s at 25 fps, which R(QUAT_Z90)^T
    # (world -> root) maps onto root +x.
    torch.testing.assert_close(
        frame["motion_gt"][0, :3], torch.tensor([2.5, 0.0, 0.0]), atol=1e-4, rtol=0)
    torch.testing.assert_close(
        kin[3]["motion_gt"][0, :3], torch.zeros(3), atol=1e-6, rtol=0)
    # mean-hips x = 1 + 0.01 * (9 + 10) / 2; the foot-level q_world root (y = 9.3
    # on this frame) is deliberately NOT what lands here.
    torch.testing.assert_close(
        frame["motion_root_pos"], torch.tensor([1.095, 2.3, 3.0]), atol=1e-4, rtol=0)
    torch.testing.assert_close(
        frame["motion_rot"], torch.as_tensor(quat_xyzw_to_matrix(QUAT_Z90)),
        atol=1e-6, rtol=0)
    # The six limb slots read the MHR70 anchor columns of kp_world.
    limbs = ClimbingCorpusDataset(
        corpus, motion_root_source="mhr",
        **{**common, "motion_joint_names": None})[0][3]["motion_gt"]
    assert limbs.shape == (7, 6)
    assert MOTION_MHR70_INDICES == (62, 41, 15, 18, 17, 20)


def test_motion_root_source_is_validated(corpus):
    with pytest.raises(ValueError, match="motion_root_source"):
        ClimbingCorpusDataset(
            corpus, scenes=["vidA_0000"], split="train", frames_per_clip=2,
            frame_stride=1, jitter=False, load_images=False,
            motion_root_source="smplx")


def _corpus_run_config(run_dir: Path, corpus: Path, extra: str = "") -> dict:
    """A minimal joint-contact corpus run config pointed at a synthetic corpus."""
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_yaml = run_dir / "dataset.yaml"
    dataset_yaml.write_text(f"name: climbing_corpus\ndata:\n  root: {corpus}\n")
    run_yaml = run_dir / "run.yaml"
    run_yaml.write_text(
        "base: tests/fixtures/climbing_videos_joint.yaml\n"
        f"data: {{datasets: [{{name: climbing_corpus, config: {dataset_yaml}}}]}}\n"
        + extra)
    return load_config(run_yaml)


def test_make_loaders_threads_keypoint_supervision(corpus, tmp_path):
    """keypoint_supervision.enabled reaches every corpus dataset as load_keypoints."""
    off = _corpus_run_config(tmp_path / "off", corpus)
    train_loader, eval_loader, _ = make_loaders(off, (256, 256))
    assert [ldr.dataset.load_keypoints for ldr in train_loader.loaders] == [False]
    assert [ldr.dataset.load_keypoints for ldr in eval_loader.loaders] == [False]

    on = _corpus_run_config(
        tmp_path / "on", corpus,
        "model: {pose_temporal: {enabled: true}}\n"
        "keypoint_supervision: {enabled: true}\n")
    train_loader, eval_loader, _ = make_loaders(on, (256, 256))
    assert [ldr.dataset.load_keypoints for ldr in train_loader.loaders] == [True]
    assert [ldr.dataset.load_keypoints for ldr in eval_loader.loaders] == [True]


# ------------------------------------------------------------------ test split

def test_manual_test_labels_match_v1_semantics(corpus):
    ds = ClimbingCorpusDataset(
        corpus, scenes=["vidC_0000"], split="test", frames_per_clip=1, frame_stride=1)
    frame = ds[0][0]
    supervised = frame["joint_supervised"]
    # left hand annotated contact; right hand annotated free.
    assert float(frame["joint_contact"][20]) == 1.0 and float(supervised[20]) == 1.0
    assert float(frame["joint_contact"][21]) == 0.0 and float(supervised[21]) == 1.0
    # left foot (big toe) unlabeled (-1) -> outside the score mask.
    assert float(supervised[10]) == 0.0
    # On a reviewed frame the 8 schema-fixed joints are supervised negatives.
    assert float(supervised[0]) == 1.0 and float(frame["joint_contact"][0]) == 0.0
    # The completed test set ships no confidence -> ones.
    assert frame["joint_confidence"].tolist() == [1.0] * 22


def test_ignored_person_is_fully_unsupervised(corpus):
    _make_db(corpus / "two", [("vidD_0000", "vidD", "test", 1, 1, 0)])
    _write_scene(corpus / "two", "vidD_0000", n_people=2, with_annotation=True)
    ds = ClimbingCorpusDataset(
        corpus / "two", split="test", frames_per_clip=1, frame_stride=1)
    for i in range(len(ds)):
        frame = ds[i][0]
        supervised = float(frame["joint_supervised"].sum())
        if frame["key"].startswith("vidD_0000#1"):
            assert supervised == 0.0                        # annotator ignored person 1
        else:
            assert supervised > 0.0


def test_test_split_requires_or_skips_missing_annotation(corpus):
    ann = (corpus / "features" / "annotation" / "vi" / "dC" / "vidC_0000"
           / "annotation.npz")
    ann.unlink()
    assert list_annotated_test_scenes(corpus) == []
    with pytest.raises(RuntimeError, match="manual joint labels are unavailable"):
        ClimbingCorpusDataset(
            corpus, scenes=["vidC_0000"], split="test", frames_per_clip=1, frame_stride=1)
    # Discovery under require_labels skips the scene; require_labels=False keeps
    # it label-free (all-zero supervision).
    assert len(ClimbingCorpusDataset(corpus, split="test", frames_per_clip=1,
                                     frame_stride=1)._scenes) == 0
    frame = ClimbingCorpusDataset(
        corpus, split="test", frames_per_clip=1, frame_stride=1,
        require_labels=False)[0][0]
    assert float(frame["joint_supervised"].sum()) == 0.0
    assert float(frame["joint_mask"].sum()) == 0.0


# ------------------------------------------------------------------ collate

def test_clip_collates_into_training_batch(corpus):
    ds = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="train", frames_per_clip=2,
        frame_stride=2, jitter=False, load_forces=True)
    cfg = load_config(REPO / "tests" / "fixtures" / "climbing_videos_joint.yaml")
    collate = make_collate((256, 256), TargetSpec.from_config(cfg))
    batch = collate([ds[0]])
    assert batch["img"].shape == (2, 1, 3, 256, 256)
    assert batch["cam_int"].shape == (2, 3, 3)
    assert batch["seq_len"] == 2
    assert batch["cam_from_world"].shape == (2, 4, 4)
    assert bool(batch["cam_valid"].all())
    assert batch["force_gt"].shape == (2, 6, 3)
    assert batch["force_lever"].shape == (2, 6, 3)
    torch.testing.assert_close(
        batch["gravity_world"], torch.tensor([[0.0, 1.0, 0.0]] * 2))
    # extremities_4 reduction: left hand + left foot in contact, right side free.
    joint = batch["targets"]["joint"]
    assert joint["gt"].shape == (2, 4)
    assert joint["gt"][0].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert bool((joint["mask"] > 0).all())


def test_clip_collates_kindyn6_targets(corpus):
    """The joint contact+force experiment config reduces the clip to six kindyn groups."""
    ds = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="train", frames_per_clip=2,
        frame_stride=2, jitter=False, load_forces=True)
    cfg = load_config(REPO / "tests" / "fixtures" / "joint_force_cond_postdec.yaml")
    collate = make_collate((256, 256), TargetSpec.from_config(cfg))
    joint = collate([ds[0]])["targets"]["joint"]
    assert joint["gt"].shape == (2, 6)
    # kindyn order LH, RH, LF=toe, RF=toe, LA=heel, RA=heel: the folded left
    # hand (finger 25) and the left big-toe joint touch; both heels are free.
    assert joint["gt"][0].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert bool((joint["mask"] > 0).all())


# ------------------------------------------------------------------ cond input

#: Standardization literals for the synthetic conditioning artifact below.
COND_STD = {"vel_mean": [0.0, 0.0, 0.0], "vel_std": [1.0, 1.0, 1.0],
            "acc_mean": [0.0, 0.0, 0.0], "acc_std": [2.0, 2.0, 2.0]}


def _write_cond_features(path: Path, entries: dict) -> Path:
    """Write a synthetic ``cond_features.npz``: ``{entry: (frame_idx, valid)}``."""
    arrays = {}
    for name, (frame_idx, valid) in entries.items():
        n = len(frame_idx)
        arrays[f"{name}#frame_idx"] = np.asarray(frame_idx, np.int64)
        arrays[f"{name}#feat_valid"] = np.asarray(valid, bool)
        # Distinct per-frame values so a mis-join is visible, not averaged away.
        arrays[f"{name}#vel_smooth_world"] = np.stack([
            np.array([0.1 * f, 0.2 * f, 0.3 * f], np.float32) for f in frame_idx])
        arrays[f"{name}#acc_smooth_world_alt"] = np.stack([
            np.array([1.0 + f, 2.0 + f, 3.0 + f], np.float32) for f in frame_idx])
        arrays[f"{name}#R_pred_world_from_root"] = np.tile(
            quat_xyzw_to_matrix(QUAT_Z90).astype(np.float32), (n, 1, 1))
    np.savez(path, **arrays)
    return path


def test_cond_feature_joins_on_frame_idx_and_zeroes_the_rest(corpus, tmp_path):
    # Frames 0, 1 and 7 are absent from the artifact; frame 4 is present but
    # marked invalid. Both cases must come back as exact zeros with bit 0.
    path = _write_cond_features(
        tmp_path / "cond.npz",
        {"vidA_0000__p0": (np.arange(2, 7), np.array([1, 1, 0, 1, 1], bool))})
    ds = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="train", frames_per_clip=N_FRAMES,
        frame_stride=1, jitter=False, load_images=False,
        cond_features_path=str(path), cond_standardize=COND_STD, cond_clip=5.0)
    clip = ds[0]
    assert [f["frame_index"] for f in clip] == list(range(N_FRAMES))
    feats = torch.stack([f["cond_feat"] for f in clip])
    assert feats.shape == (N_FRAMES, COND_FEATURE_DIM) and feats.dtype == torch.float32
    for pos in (0, 1, 4, 7):
        assert float(feats[pos].abs().sum()) == 0.0, pos
    for pos in (2, 3, 5, 6):
        assert float(feats[pos, 9]) == 1.0

    expected = cond_feature_rows(
        np.array([[0.1 * 2, 0.2 * 2, 0.3 * 2]], np.float32),
        np.array([[1.0 + 2, 2.0 + 2, 3.0 + 2]], np.float32),
        quat_xyzw_to_matrix(QUAT_Z90).astype(np.float32)[None],
        np.array([True]), COND_STD, 5.0)
    torch.testing.assert_close(feats[2], torch.from_numpy(expected[0]))


def test_cond_feature_missing_entry_is_all_zeros(corpus, tmp_path):
    path = _write_cond_features(
        tmp_path / "cond.npz", {"someone_else__p0": (np.arange(4), np.ones(4, bool))})
    ds = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="train", frames_per_clip=2,
        frame_stride=1, jitter=False, load_images=False,
        cond_features_path=str(path), cond_standardize=COND_STD, cond_clip=5.0)
    for frame in ds[0]:
        assert float(frame["cond_feat"].abs().sum()) == 0.0


def test_cond_features_require_standardization(corpus, tmp_path):
    path = _write_cond_features(
        tmp_path / "cond.npz", {"vidA_0000__p0": (np.arange(4), np.ones(4, bool))})
    with pytest.raises(ValueError, match="cond_standardize entries"):
        ClimbingCorpusDataset(
            corpus, scenes=["vidA_0000"], split="train", frames_per_clip=2,
            frame_stride=1, jitter=False, load_images=False,
            cond_features_path=str(path), cond_standardize={"vel_mean": [0.0] * 3})


def test_cond_feat_collates_with_and_without_features(corpus, tmp_path):
    cfg = load_config(REPO / "tests" / "fixtures" / "joint_force_cond_postdec.yaml")
    collate = make_collate((256, 256), TargetSpec.from_config(cfg))
    common = dict(
        scenes=["vidA_0000"], split="train", frames_per_clip=2, frame_stride=2,
        jitter=False, load_forces=True)

    # No features configured: the key is still emitted, as zeros (a still-image
    # dataset in a mixed batch looks the same), so the model's zero-init
    # projections are exercised on every step.
    plain = collate([ClimbingCorpusDataset(corpus, **common)[0]])
    assert plain["cond_feat"].shape == (2, COND_FEATURE_DIM)
    assert plain["cond_feat"].dtype == torch.float32
    assert float(plain["cond_feat"].abs().sum()) == 0.0

    path = _write_cond_features(
        tmp_path / "cond.npz",
        {"vidA_0000__p0": (np.arange(N_FRAMES), np.ones(N_FRAMES, bool))})
    ds = ClimbingCorpusDataset(
        corpus, cond_features_path=str(path), cond_standardize=COND_STD,
        cond_clip=5.0, **common)
    clip = ds[0]
    batch = collate([clip])
    torch.testing.assert_close(
        batch["cond_feat"], torch.stack([f["cond_feat"] for f in clip]))
    assert float(batch["cond_feat"].abs().sum()) > 0.0


# ------------------------------------------------------------------ real corpus

@requires_corpus
def test_real_scene_discovery_counts():
    # 2026-08-27 corpus update (better contacts/forces/poses): 864 train and
    # 108 curated test scenes; annotation COMPLETE since 2026-08-29 (was 61).
    assert len(list_corpus_scenes(CORPUS, "train")) == 864
    assert len(list_corpus_scenes(CORPUS, "test")) == 108
    assert len(list_annotated_test_scenes(CORPUS)) == 108


def _real_scene() -> str:
    scenes = list_corpus_scenes(CORPUS, "train")
    return "0aow0AvNZ2A_0004" if "0aow0AvNZ2A_0004" in scenes else scenes[0]


def _matrix_from_axis_angle(aa: np.ndarray) -> np.ndarray:
    """Rodrigues in float32 (independent reimplementation for the check)."""
    aa = np.asarray(aa, np.float32)
    theta = float(np.linalg.norm(aa))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    x, y, z = aa / theta
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], np.float32)
    return (np.eye(3, dtype=np.float32) + np.sin(theta) * skew
            + (1.0 - np.cos(theta)) * (skew @ skew)).astype(np.float32)


@requires_corpus
def test_real_root_quaternion_is_xyzw_world_from_root():
    sid = _real_scene()
    npz = np.load(CORPUS / "features" / "human_optim" / scene_shard(sid) / sid
                  / "kindyn_1.npz", allow_pickle=True)
    q = np.asarray(npz["q"], np.float32)
    global_orient = np.asarray(npz["global_orient"], np.float32)
    valid = np.asarray(npz["valid_mask"], bool)
    checked = 0
    for person in range(q.shape[0]):
        for t in range(0, q.shape[1], 5):
            if not valid[person, t]:
                continue
            from_quat = quat_xyzw_to_matrix(q[person, t, 3:7])
            from_aa = _matrix_from_axis_angle(global_orient[person, t])
            assert float(np.abs(from_quat - from_aa).max()) < 1e-4
            checked += 1
    assert checked > 5


@requires_corpus
def test_real_forces_support_body_weight_and_rotation_preserves_norms():
    sid = _real_scene()
    ds = ClimbingCorpusDataset(
        CORPUS, scenes=[sid], split="train", frames_per_clip=1, frame_stride=1,
        jitter=False, load_forces=True, load_images=False)
    data = ds._scenes[sid]
    npz = np.load(CORPUS / "features" / "human_optim" / scene_shard(sid) / sid
                  / "kindyn_1.npz", allow_pickle=True)
    # Reference fold, independently of the loader: frames -> groups by parent.
    parents = np.asarray(npz["contact_frame_parents"], np.int64)
    group_of = np.full(len(parents), -1, np.int64)
    for g, members in enumerate(FORCE_GROUPS_52):
        group_of[np.isin(parents, list(members))] = g
    frame_forces = np.asarray(npz["frame_forces"], np.float32)
    forces_world = np.stack(
        [frame_forces[:, :, group_of == g].sum(axis=2) for g in range(6)], axis=2)
    mass = np.asarray(npz["total_mass"], np.float32)
    forces_world_bw = forces_world / (mass[:, None, None, None] * 9.81)
    valid = data["force_valid"]

    # On valid rows force_contact matches the zero pattern of the solved forces.
    nonzero = np.linalg.norm(forces_world, axis=-1) > 0
    np.testing.assert_array_equal(nonzero[valid], data["force_contact"][valid])

    # The world->root rotation preserves per-group magnitudes.
    np.testing.assert_allclose(
        np.linalg.norm(data["force_gt"], axis=-1)[valid],
        np.linalg.norm(forces_world_bw, axis=-1)[valid], atol=1e-5)

    # The loader's gravity is kindyn's FITTED unit vector, and on contact
    # frames the world force sum supports ~1 body weight against it.
    ghat = np.asarray(npz["gravity_world"], np.float64)
    ghat = ghat / np.linalg.norm(ghat)
    np.testing.assert_allclose(data["gravity_world"], ghat, atol=1e-6)
    frame_contact = (data["force_contact"] & valid[..., None]).any(axis=-1)
    proj = (forces_world_bw.sum(axis=2) @ ghat)[frame_contact]
    assert -1.15 < float(proj.mean()) < -0.85


@requires_corpus
def test_real_clip_collates_without_reading_frames():
    sid = _real_scene()
    ds = ClimbingCorpusDataset(
        CORPUS, scenes=[sid], split="train", frames_per_clip=2, frame_stride=2,
        jitter=False, load_forces=True, load_images=False)
    clip = ds[0]
    rng = np.random.default_rng(0)
    for frame in clip:                      # frames/ extraction may be in flight
        frame["image"] = rng.integers(0, 255, (64, 48, 3), np.uint8)
        frame["mask"] = None
    cfg = load_config(REPO / "tests" / "fixtures" / "climbing_videos_joint.yaml")
    collate = make_collate((256, 256), TargetSpec.from_config(cfg))
    batch = collate([clip])
    assert batch["img"].shape == (2, 1, 3, 256, 256)
    assert batch["targets"]["joint"]["gt"].shape == (2, 4)
    assert bool(batch["cam_valid"].all())
    assert batch["frame_valid"].tolist() == [True, True]


def test_full_scenes_emits_one_longest_run_clip(corpus):
    # N=8 all valid: one clip per person covering the whole scene.
    ds = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="val", frames_per_clip=3,
        frame_stride=1, jitter=False, full_scenes=True)
    assert [(base, t) for _, _, base, _, t in ds._items] == [(0, 8)]
    clip = ds[0]
    assert [f["frame_position"] for f in clip] == list(range(8))

    # An internal gap: the longest contiguous run wins.
    _write_scene(corpus, "vidB_0000", invalid_frames=(2,))
    ds = ClimbingCorpusDataset(
        corpus, scenes=["vidB_0000"], split="val", frames_per_clip=3,
        frame_stride=1, jitter=False, full_scenes=True)
    assert [(base, t) for _, _, base, _, t in ds._items] == [(3, 5)]

    # Stride shortens the clip: T = (run_len - 1) // stride + 1.
    ds = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="val", frames_per_clip=3,
        frame_stride=2, jitter=False, full_scenes=True)
    assert [(base, t) for _, _, base, _, t in ds._items] == [(0, 4)]
    assert [f["frame_position"] for f in ds[0]] == [0, 2, 4, 6]


def test_full_scenes_rejected_for_train_split(corpus):
    with pytest.raises(ValueError, match="eval protocol"):
        ClimbingCorpusDataset(
            corpus, scenes=["vidA_0000"], split="train", frames_per_clip=3,
            full_scenes=True)


def test_full_scenes_eval_max_frames_caps_the_clip(corpus):
    ds = ClimbingCorpusDataset(
        corpus, scenes=["vidA_0000"], split="val", frames_per_clip=3,
        frame_stride=1, jitter=False, full_scenes=True, eval_max_frames=5)
    assert [(base, t) for _, _, base, _, t in ds._items] == [(0, 5)]
    assert [f["frame_position"] for f in ds[0]] == list(range(5))


def test_gravity_view_targets_are_the_smoothed_twist_re_expressed(corpus):
    """`gravity_view` changes the AXES of the pelvis linear target, nothing else.

    Lifting both conventions back to the world with their own rotations must give
    the same vectors (with the twist's Coriolis term on the acceleration side),
    which pins the einsum directions, the Coriolis sign, and — because the
    comparison is against the SMOOTHED twist — the fact that the GV branch
    differentiates the smoothed trajectory rather than the raw one.
    """
    human = corpus / "features" / "human_optim" / "vi" / "dA" / "vidA_0000"
    with np.load(human / "kindyn_1.npz", allow_pickle=True) as npz:
        kindyn = dict(npz)
    tilted = np.array([0.25, 0.95, -0.18], np.float32)
    tilted /= np.linalg.norm(tilted)
    kindyn["gravity_world"] = tilted
    np.savez(human / "kindyn_1.npz", **kindyn)
    # Spin the MHR root: with the fixture's constant orientation omega is zero and
    # the Coriolis term of the world lift would be untestable.
    with np.load(human / "mhr_1.npz", allow_pickle=True) as npz:
        mhr = dict(npz)
    angle = 0.15 * np.arange(N_FRAMES, dtype=np.float32)
    mhr["q_world"][:, :, 3:7] = np.stack(
        [np.zeros_like(angle), np.zeros_like(angle),
         np.sin(angle / 2), np.cos(angle / 2)], axis=-1)[None]      # xyzw about z
    np.savez(human / "mhr_1.npz", **mhr)

    common = dict(
        scenes=["vidA_0000"], split="train", frames_per_clip=N_FRAMES, frame_stride=1,
        jitter=False, load_images=False, load_motion=True, motion_root_source="mhr",
        motion_joint_names=["pelvis"], motion_target_smooth_sec=0.12)
    twist = ClimbingCorpusDataset(
        corpus, motion_root_convention="twist", **common)._scenes["vidA_0000"]
    gview = ClimbingCorpusDataset(
        corpus, motion_root_convention="gravity_view", **common)._scenes["vidA_0000"]

    valid = gview["motion_valid"]
    assert valid.any(), "fixture has no supervised motion row to compare"
    # The GV frame is a genuinely different frame (else this test is vacuous).
    assert not np.allclose(gview["motion_lin_rot"], gview["motion_rot"], atol=1e-3)
    np.testing.assert_allclose(gview["motion_rot"], twist["motion_rot"], atol=1e-6)

    def to_world(scene, rot_key, columns, coriolis):
        vec = scene["motion_gt"][:, :, 0, columns]
        if coriolis:
            vec = vec + np.cross(scene["motion_omega"], scene["motion_gt"][:, :, 0, 0:3])
        return np.einsum("pnij,pnj->pni", scene[rot_key], vec)

    for columns, coriolis in ((slice(0, 3), False), (slice(3, 6), True)):
        gv_world = to_world(gview, "motion_lin_rot", columns, False)
        twist_world = to_world(twist, "motion_rot", columns, coriolis)
        np.testing.assert_allclose(gv_world[valid], twist_world[valid], atol=1e-4)

    # Channel 1 of the GV target IS the downward component of the world velocity.
    world_vel = to_world(twist, "motion_rot", slice(0, 3), False)
    np.testing.assert_allclose(
        gview["motion_gt"][:, :, 0, 1][valid], (world_vel @ tilted)[valid], atol=1e-4)
