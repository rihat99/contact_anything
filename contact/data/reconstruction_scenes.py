"""BetterVideoReconstruction ``out/<stem>/`` trees as a per-frame inference dataset.

Reads the raw pipeline outputs directly (no exported ClimbingVideos_v1 scene):
``sam3/bboxes.npz`` (per-person xyxy boxes), ``sam3/<obj:02d>/frame_*.png``
person masks, ``geometry/transform.npz`` (per-frame ``intrinsics_px_orig`` and
metric ``cam_from_world`` extrinsics) and ``human_optim/contacts_1.npz``
(``valid_mask``/``fps``; falls back to ``sam3d/params.npz`` when the contacts
stage has not run). Field choices mirror
``BetterVideoReconstruction/scripts/export_contact_dataset.py`` so inputs match
the training-time ClimbingVideos_v1 domain; video frames are extracted with the
same OpenCV-decode + JPEG-quality-95 re-encode (:func:`extract_frames`).

The class is **label-free and inference-only**: every item is one valid
``(person, frame)`` returned as a length-1 clip whose dict matches
:class:`~contact.data.climbing_corpus.ClimbingCorpusDataset` items (zero-filled
supervision, exactly the ``require_labels=False`` path), so the centered
sliding-window machinery in ``scripts/render_climbing_video_contacts.py``
(``sliding_window_requests`` / ``_predict_requests``) runs on it unchanged.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..targets import NUM_BODY_22

FRAME_JPEG_QUALITY = 95     # matches export_contact_dataset.py's default export


def extract_frames(video_path: Path, out_dir: Path, n_expected: int) -> None:
    """Decode ``video_path`` sequentially into ``out_dir/<pos:06d>.jpg``.

    The pipeline enumerated frames by sequential OpenCV decode order, so the
    k-th decoded frame IS out-tree frame k (container headers may over-report
    the frame count; only the decodable count matters). Raises when the number
    of decodable frames differs from ``n_expected``. Existing complete
    extractions are reused.
    """
    out_dir = Path(out_dir)
    if len(list(out_dir.glob("*.jpg"))) == n_expected:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    try:
        count = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if count >= n_expected:
                raise ValueError(
                    f"{video_path}: more decodable frames than the out-tree's "
                    f"{n_expected} — frame indexing would be ambiguous")
            Image.fromarray(bgr[..., ::-1]).save(
                out_dir / f"{count:06d}.jpg", quality=FRAME_JPEG_QUALITY)
            count += 1
    finally:
        cap.release()
    if count != n_expected:
        raise ValueError(
            f"{video_path}: decoded {count} frames, out-tree arrays expect {n_expected}")


class ReconstructionSceneDataset(Dataset):
    """One ``out/<stem>/`` tree as per-frame (T=1) inference items.

    :param out_dir: the scene's pipeline output tree (``out/<stem>/``).
    :param frames_dir: directory of pre-extracted ``<pos:06d>.jpg`` frames
        (see :func:`extract_frames`).
    :param scene: scene id used in item keys; defaults to ``out_dir.name``.
    """

    supervised_targets = frozenset()
    topology = None
    name = "reconstruction_scenes"

    def __init__(self, out_dir: str | Path, frames_dir: str | Path, scene: str | None = None):
        super().__init__()
        out_dir = Path(out_dir)
        self.frames_dir = Path(frames_dir)
        scene = out_dir.name if scene is None else str(scene)
        self.scene = scene

        boxes = np.load(out_dir / "sam3" / "bboxes.npz", allow_pickle=True)
        transform = np.load(out_dir / "geometry" / "transform.npz", allow_pickle=True)
        bbox = np.asarray(boxes["bboxes_per_obj"], np.float32)        # [P_box, N, 4] xyxy px
        intrinsics = np.asarray(transform["intrinsics_px_orig"], np.float32)  # [N, 3, 3]
        extrinsics = np.asarray(transform["extrinsics"], np.float32)  # [N, 4, 4] cam-from-world

        contacts_path = out_dir / "human_optim" / "contacts_1.npz"
        if contacts_path.is_file():
            source = np.load(contacts_path, allow_pickle=True)
        else:                                   # contacts stage not run for this tree
            source = np.load(out_dir / "sam3d" / "params.npz", allow_pickle=True)
        valid_mask = np.asarray(source["valid_mask"], bool)           # [P, N]
        fps = float(source["fps"]) if "fps" in source.files else float(transform["fps"])

        # SAM 3 may track more objects than the human stages kept (e.g. bystanders
        # dropped by sam3d/human_optim). Predict for the kept people only, selecting
        # their bbox rows by object id.
        box_ids = [int(x) for x in np.asarray(boxes["object_ids"]).reshape(-1)]
        object_ids = np.asarray(source["object_ids"], np.int64).reshape(-1)
        missing_ids = [i for i in object_ids.tolist() if i not in box_ids]
        if missing_ids:
            raise ValueError(
                f"{scene}: object ids {missing_ids} have no sam3 bbox track "
                f"(available: {box_ids})")
        bbox = bbox[[box_ids.index(i) for i in object_ids.tolist()]]  # [P, N, 4]

        n_people, n_frames = valid_mask.shape
        if bbox.shape != (n_people, n_frames, 4):
            raise ValueError(
                f"{scene}: bboxes_per_obj {bbox.shape} does not match "
                f"valid_mask {valid_mask.shape}")
        if intrinsics.shape != (n_frames, 3, 3) or extrinsics.shape != (n_frames, 4, 4):
            raise ValueError(
                f"{scene}: camera arrays {intrinsics.shape}/{extrinsics.shape} do not "
                f"match {n_frames} frames")

        # A tracked frame whose box is degenerate cannot be cropped — treat it as
        # invalid rather than failing the scene (pipeline stages tolerate items).
        bbox_good = (
            np.isfinite(bbox).all(axis=-1)
            & (bbox[..., 2] > bbox[..., 0])
            & (bbox[..., 3] > bbox[..., 1])
        )
        valid_mask = valid_mask & bbox_good

        self._scenes = {scene: {
            "dir": out_dir,
            "object_ids": object_ids,
            "frame_indices": np.arange(n_frames, dtype=np.int64),
            "bbox": bbox,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "valid_mask": valid_mask,
            "fps": fps,
        }}
        self._items = [
            (scene, person, pos, 1)
            for person in range(n_people)
            for pos in range(n_frames)
            if valid_mask[person, pos]
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> list[dict]:
        scene, person, pos, _ = self._items[index]
        data = self._scenes[scene]
        oid = int(data["object_ids"][person])

        image = np.array(
            Image.open(self.frames_dir / f"{pos:06d}.jpg").convert("RGB"), np.uint8)
        mask_path = data["dir"] / "sam3" / f"{oid:02d}" / f"frame_{pos:06d}.png"
        mask = np.array(Image.open(mask_path), np.uint8) if mask_path.is_file() else None

        zeros = torch.zeros(NUM_BODY_22, dtype=torch.float32)
        frame = {
            "image": image,
            "mask": mask,
            "bbox": data["bbox"][person, pos],
            "cam_int": data["intrinsics"][pos],
            "joint_contact": zeros,
            "joint_mask": zeros.clone(),
            "joint_supervised": zeros.clone(),
            "joint_confidence": zeros.clone(),
            "frame_pos_sec": 0.0,
            "frame_position": pos,
            "frame_index": pos,
            "frame_valid": True,
            "key": f"{scene}#{oid}@{pos}",
            "dataset": self.name,
        }
        return [frame]
