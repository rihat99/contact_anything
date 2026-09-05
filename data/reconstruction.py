"""BetterVideoReconstruction ``out/<stem>/`` trees as an inference dataset.

Reads a raw pipeline output tree directly: ``sam3/bboxes.npz`` (per-person xyxy
boxes), ``sam3/<oid:02d>/frame_*.png`` person masks,
``geometry/transform.npz`` (per-frame ``intrinsics_px_orig`` and metric
``cam_from_world``) and ``human_optim/contacts_1.npz`` for ``valid_mask``/``fps``
(falling back to ``sam3d/params.npz`` when the contacts stage has not run).
Video frames are extracted with the same sequential OpenCV decode + JPEG-95
re-encode the corpus tree uses (:func:`extract_frames`), so frame ``k`` of the
tree is row ``k`` of every feature array.

The dataset is **label-free and inference-only**: clips carry the frame schema's
input block (image, mask, box, cameras, clip timing) and no ground truth, so
predictions can run on scenes the corpus knows nothing about.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .base import ClipDataset

FRAME_JPEG_QUALITY = 95


def extract_frames(video_path: Path, out_dir: Path, n_expected: int) -> None:
    """Decode ``video_path`` sequentially into ``out_dir/<pos:06d>.jpg``.

    The pipeline enumerated frames by sequential decode order, so the k-th
    decoded frame IS out-tree frame k (container headers may over-report the
    frame count; only the decodable count matters). Raises when the number of
    decodable frames differs from ``n_expected``. Complete extractions are
    reused.
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


class ReconstructionSceneDataset(ClipDataset):
    """One ``out/<stem>/`` tree as clips (default: per-frame ``T = 1`` items).

    :param out_dir: the scene's pipeline output tree.
    :param frames_dir: directory of pre-extracted ``<pos:06d>.jpg`` frames.
    :param scene: scene id used in item keys; defaults to ``out_dir.name``.
    :param clip_frames: clip length — 1 for a per-frame model, ``T`` for a
        temporal one.
    :param stride: source-frame stride inside a clip.
    """

    name = "reconstruction"

    def __init__(
        self,
        out_dir: str | Path,
        frames_dir: str | Path,
        scene: Optional[str] = None,
        *,
        clip_frames: int = 1,
        stride: int | str = 1,
    ):
        self.out_dir = Path(out_dir)
        self.frames_dir = Path(frames_dir)
        self.scene = self.out_dir.name if scene is None else str(scene)
        super().__init__([self.scene], split="test", clip_frames=clip_frames,
                         stride=stride, jitter=False)

    def _load_scene(self, scene: str) -> dict:
        boxes = np.load(self.out_dir / "sam3" / "bboxes.npz", allow_pickle=True)
        transform = np.load(
            self.out_dir / "geometry" / "transform.npz", allow_pickle=True)
        bbox = np.asarray(boxes["bboxes_per_obj"], np.float32)
        intrinsics = np.asarray(transform["intrinsics_px_orig"], np.float32)
        extrinsics = np.asarray(transform["extrinsics"], np.float32)

        contacts_path = self.out_dir / "human_optim" / "contacts_1.npz"
        source = np.load(
            contacts_path if contacts_path.is_file()
            else self.out_dir / "sam3d" / "params.npz", allow_pickle=True)
        valid_mask = np.asarray(source["valid_mask"], bool)               # [P, N]
        fps = (float(source["fps"]) if "fps" in source.files
               else float(transform["fps"]))

        # SAM 3 may track more objects than the human stages kept (bystanders
        # dropped by sam3d/human_optim). Predict for the kept people only.
        box_ids = [int(x) for x in np.asarray(boxes["object_ids"]).reshape(-1)]
        object_ids = np.asarray(source["object_ids"], np.int64).reshape(-1)
        missing = [i for i in object_ids.tolist() if i not in box_ids]
        if missing:
            raise ValueError(
                f"{scene}: object ids {missing} have no sam3 bbox track "
                f"(available: {box_ids})")
        bbox = bbox[[box_ids.index(i) for i in object_ids.tolist()]]      # [P, N, 4]

        n_people, n_frames = valid_mask.shape
        if bbox.shape != (n_people, n_frames, 4):
            raise ValueError(
                f"{scene}: bboxes_per_obj {bbox.shape} does not match "
                f"valid_mask {valid_mask.shape}")
        if intrinsics.shape != (n_frames, 3, 3) or extrinsics.shape != (n_frames, 4, 4):
            raise ValueError(
                f"{scene}: camera arrays {intrinsics.shape}/{extrinsics.shape} do "
                f"not match {n_frames} frames")
        # A tracked frame whose box is degenerate cannot be cropped.
        valid_mask = valid_mask & (
            np.isfinite(bbox).all(axis=-1)
            & (bbox[..., 2] > bbox[..., 0])
            & (bbox[..., 3] > bbox[..., 1]))

        scene_data = {
            "object_ids": object_ids,
            "frame_indices": np.arange(n_frames, dtype=np.int64),
            "bbox": bbox,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "valid_mask": valid_mask,
            "fps": fps,
        }
        return scene_data

    def _frame(
        self, scene: str, data: dict, person: int, position: int, row: int,
        positions: np.ndarray,
    ) -> dict:
        oid = int(data["object_ids"][person])
        image = np.array(
            Image.open(self.frames_dir / f"{position:06d}.jpg").convert("RGB"), np.uint8)
        mask_path = self.out_dir / "sam3" / f"{oid:02d}" / f"frame_{position:06d}.png"
        mask = np.array(Image.open(mask_path), np.uint8) if mask_path.is_file() else None
        frame = {
            "image": image,
            "img_wh": None,
            "mask": mask,
            "bbox": data["bbox"][person, position],
            "cam_int": data["intrinsics"][position],
            "cam_from_world": data["extrinsics"][position],
            "frame_pos_sec": float(position - int(positions[0])) / data["fps"],
            "frame_index": position,
            "frame_valid": True,
            "key": f"{scene}#{oid}@{position}",
        }
        return frame
