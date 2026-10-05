"""``Bedlam2Dataset`` — windowed BEDLAM2_our clips with the requested GT.

Composes :mod:`data.bedlam2.scene` (frames, instance masks, boxes, cameras,
per-slot contact labels) and :mod:`data.bedlam2.gt` (GT forces, the SMPL-X body)
into the same frame schema :mod:`data.base` documents and
:class:`~data.climbing_videos.ClimbingVideosDataset` emits, so the two corpora
can be concatenated into one loader.

There is no embedding or pose-token cache for BEDLAM: the live path (decode the
frame, run the backbone) is the only one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image

from model.contact_frames import contact_set as contact_set_of

from ..base import SIGNAL_GROUPS, ClipDataset
from . import gt, scene as scene_io


class Bedlam2Dataset(ClipDataset):
    """Windowed ``(scene, person)`` clips of the BEDLAM2_our corpus.

    :param root: corpus root containing ``scenes/``, ``features/``, ``frames/``.
    :param scenes: explicit scene ids; ``None`` discovers them from the DB and
        our stratified split.
    :param split: ``"train"`` or ``"test"`` (:func:`data.bedlam2.scene.split_scenes`).
    :param contact_set: ``kindyn6`` | ``frames35`` — the slots the contact labels
        and forces are emitted in (:mod:`model.contact_frames`).
    :param load: signal groups to emit, a subset of ``{"forces", "smplx"}``.
    :param test_fraction: share of each render job's scenes in the test split.
    :param split_seed: seed of that draw.
    :param allow_missing_labels: tolerate more than 1 % of a split having no
        boxes / masks yet.

    Windowing parameters (``clip_frames``, ``stride``, ``jitter``, ``seed``,
    ``full_scenes``, ``max_frames``) are :class:`~data.base.ClipDataset`'s.
    """

    name = "bedlam2"

    @staticmethod
    def list_scenes(
        root: str | Path, split: str, *, test_fraction: float = 0.02,
        split_seed: int = 0, allow_missing_labels: bool = False,
    ) -> list[str]:
        """Scene ids of ``split`` without loading them."""
        return scene_io.list_scenes(
            root, split, test_fraction=test_fraction, split_seed=split_seed,
            allow_missing_labels=allow_missing_labels)

    @staticmethod
    def video_of(scene: str) -> str:
        """Source video of a scene id: a BEDLAM scene IS its own video."""
        return scene

    @classmethod
    def scene_ids(cls, spec: dict, split: str) -> list[str]:
        """Scene ids of ``split`` for a dataset yaml ``spec``."""
        return cls.list_scenes(
            Path(spec["root"]), split, test_fraction=float(spec["test_fraction"]),
            split_seed=int(spec["split_seed"]),
            allow_missing_labels=bool(spec["allow_missing_labels"]))

    @classmethod
    def from_spec(
        cls, spec: dict, *, embedding_cache: bool, pose_token_cache: bool, **kwargs,
    ) -> "Bedlam2Dataset":
        """Build from a dataset yaml plus the run's cache flags (neither is supported)."""
        if embedding_cache or pose_token_cache:
            raise ValueError(
                "bedlam2 has no embedding / pose-token cache: run it on the live path "
                "(data.embedding_cache and data.pose_token_cache off)")
        return cls(
            Path(spec["root"]),
            test_fraction=float(spec["test_fraction"]),
            split_seed=int(spec["split_seed"]),
            allow_missing_labels=bool(spec["allow_missing_labels"]),
            **kwargs)

    def __init__(
        self,
        root: str | Path,
        scenes: Optional[Sequence[str]] = None,
        *,
        split: str = "train",
        clip_frames: int = 8,
        stride: int | str = "auto",
        jitter: bool = True,
        seed: int = 42,
        contact_set: str = "frames35",
        load: Iterable[str] = (),
        full_scenes: bool = False,
        max_frames: Optional[int] = None,
        test_fraction: float = 0.02,
        split_seed: int = 0,
        allow_missing_labels: bool = False,
    ):
        self.root = Path(root)
        self.slots = contact_set_of(contact_set)
        self.load = frozenset(load)
        unknown = self.load - SIGNAL_GROUPS
        if unknown:
            raise ValueError(
                f"unknown signal group(s) {sorted(unknown)}; "
                f"choose from {sorted(SIGNAL_GROUPS)}")
        if scenes is None:
            scenes = self.list_scenes(
                self.root, split, test_fraction=test_fraction, split_seed=split_seed,
                allow_missing_labels=allow_missing_labels)
        super().__init__(
            scenes, split=split, clip_frames=clip_frames, stride=stride,
            jitter=jitter, seed=seed, full_scenes=full_scenes, max_frames=max_frames)

    # ------------------------------------------------------------------ loading

    def _load_scene(self, scene: str) -> dict:
        data = scene_io.load_scene(self.root, scene, self.slots)
        n = len(data["frame_indices"])
        if "forces" in self.load:
            data.update(gt.load_forces(scene, data["gt_dir"], n, slots=self.slots))
        if "smplx" in self.load:
            data.update(gt.load_smplx(scene, data["gt_dir"], n))
        return data

    # ------------------------------------------------------------------ frames

    def _frame(
        self, scene: str, data: dict, person: int, position: int, row: int,
        positions: np.ndarray,
    ) -> dict:
        oid = int(data["object_ids"][person])
        valid = bool(data["valid_mask"][person, position])
        image = np.array(
            Image.open(data["frames_dir"] / f"{position:06d}.jpg").convert("RGB"), np.uint8)
        # One instance map per frame: person p is drawn as p + 1 (object_ids order).
        instances = np.array(
            Image.open(data["mask_dir"] / f"{position:06d}.png"), np.uint8)
        mask = ((instances == person + 1) * 255).astype(np.uint8)

        frame = {
            "image": image,
            "img_wh": None,
            "mask": mask,
            "bbox": data["bbox"][person, position],                     # [4] xyxy
            "cam_int": data["intrinsics"][position],                    # [3, 3]
            "cam_from_world": data["extrinsics"][position],             # [4, 4]
            "frame_pos_sec": float(
                data["frame_indices"][position]
                - data["frame_indices"][int(positions[0])]) / data["fps"],
            "frame_index": int(data["frame_indices"][position]),
            "frame_valid": valid,
            "key": f"{scene}#{oid}@{position}",
            "contact_gt": data["contact_gt"][person, position],         # [K]
            "contact_valid": data["contact_valid"][person, position],   # [K]
            "contact_conf": data["contact_conf"][person, position],     # [K]
        }
        if "forces" in self.load:
            frame["force_gt"] = data["force_gt"][person, position]            # [K, 3]
            frame["force_contact"] = data["force_contact"][person, position]  # [K]
            frame["force_lever"] = data["force_lever"][person, position]      # [K, 3]
            frame["force_conf"] = float(data["force_conf"][person, position])
            frame["force_valid"] = valid and bool(data["force_valid"][person, position])
            frame["gravity_world"] = data["gravity_world"]                    # [3] per scene
            frame["gravity_measured"] = bool(data["gravity_measured"])        # per scene
        if "smplx" in self.load:
            frame["gravity_world"] = data["gravity_world"]
            frame["gravity_measured"] = bool(data["gravity_measured"])
            frame["smplx_joints_world"] = data["smplx_joints_world"][person, position]
            frame["smplx_root_rot"] = data["smplx_root_rot"][person, position]   # [3, 3]
            frame["smplx_body_rot"] = data["smplx_body_rot"][person, position]   # [21, 3, 3]
            frame["smplx_hand_rot"] = data["smplx_hand_rot"][person, position]   # [30, 3, 3]
            frame["smplx_betas"] = data["smplx_betas"][person]                   # [10]
            frame["smplx_valid"] = valid and bool(data["smplx_valid"][person, position])
        return frame
