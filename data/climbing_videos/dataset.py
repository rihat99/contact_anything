"""``ClimbingVideosDataset`` — windowed corpus clips with the requested GT.

Composes :mod:`data.climbing_videos.scene` (frames, masks, boxes, cameras,
six-group contact labels) and :mod:`data.climbing_videos.kindyn` (GT forces,
the SMPL-X body GT) into the frame schema documented in :mod:`data.base`.

Which signal groups load is decided by the caller (``load``), never by the
dataset yaml: the trainer derives it from which losses are enabled.

With an ``embedding_dir`` every frame carries the cached bf16 ``[1280, 32, 32]``
frozen-backbone output and the model skips the backbone. Frame JPEGs are then
NOT pixel-decoded — only their header, for the full-frame size — because the
model provably never reads the crop's values on that path. A missing cache file
raises: a stale or incomplete cache must never silently fall back to live
compute. Masks still decode (mask conditioning runs live).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from ..base import SIGNAL_GROUPS, ClipDataset
from . import kindyn, scene as scene_io


class ClimbingVideosDataset(ClipDataset):
    """Windowed ``(scene, person)`` clips of the ClimbingVideos corpus.

    :param root: corpus root containing ``scenes/``, ``features/``, ``frames/``.
    :param scenes: explicit scene ids; ``None`` discovers them from the DB
        (train split, or the annotated test scenes).
    :param split: ``"train"`` (automatic labels, jittered windows) or ``"test"``
        (manual annotation, fixed windows).
    :param contact_level: label readout level — ``contacts_1`` or ``contacts_2``.
    :param load: signal groups to emit, a subset of ``{"forces", "smplx"}``.
    :param embedding_dir: precomputed-embedding root (``features/embedding``).
    :param camera_filter: ``all`` | ``static`` | ``moving`` (the DB's
        ``static_camera`` flag), used when ``scenes`` is ``None``.

    Windowing parameters (``clip_frames``, ``stride``, ``jitter``, ``seed``,
    ``full_scenes``, ``max_frames``) are :class:`~data.base.ClipDataset`'s.
    """

    name = "climbing_videos"

    @staticmethod
    def list_scenes(root: str | Path, split: str, camera: str = "all") -> list[str]:
        """Scene ids of ``split`` without loading them (train, or annotated test).

        :param camera: ``all`` | ``static`` | ``moving`` (the DB's ``static_camera`` flag).
        """
        return (scene_io.list_train_scenes(root, camera) if split == "train"
                else scene_io.list_test_scenes(root, camera))

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
        contact_level: int = 1,
        load: Iterable[str] = (),
        embedding_dir: Optional[str | Path] = None,
        pose_token_dir: Optional[str | Path] = None,
        full_scenes: bool = False,
        max_frames: Optional[int] = None,
        camera_filter: str = "all",
    ):
        self.root = Path(root)
        if int(contact_level) not in (1, 2):
            raise ValueError(f"contact_level must be 1 or 2; got {contact_level!r}")
        self.contact_level = int(contact_level)
        self.load = frozenset(load)
        unknown = self.load - SIGNAL_GROUPS
        if unknown:
            raise ValueError(
                f"unknown signal group(s) {sorted(unknown)}; "
                f"choose from {sorted(SIGNAL_GROUPS)}")
        self.embedding_dir = None if embedding_dir is None else Path(embedding_dir)
        self.pose_token_dir = None if pose_token_dir is None else Path(pose_token_dir)
        if self.pose_token_dir is not None and self.embedding_dir is not None:
            raise ValueError("pose_token_dir supersedes embedding_dir; give one")
        if scenes is None:
            scenes = self.list_scenes(self.root, split, camera_filter)
        super().__init__(
            scenes, split=split, clip_frames=clip_frames, stride=stride,
            jitter=jitter, seed=seed, full_scenes=full_scenes, max_frames=max_frames)

    # ------------------------------------------------------------------ loading

    def _load_scene(self, scene: str) -> dict:
        data = scene_io.load_scene(self.root, scene, self.split, self.contact_level)
        human_dir, object_ids = data["human_dir"], data["object_ids"]
        n = len(data["frame_indices"])
        if self.pose_token_dir is not None:
            data.update(self._load_pose_tokens(scene, object_ids, n, data["valid_mask"]))
        gravity_path = scene_io.gravity_path(self.root, scene)
        if "forces" in self.load:
            data.update(kindyn.load_forces(scene, human_dir, object_ids, n,
                                           gravity_path=gravity_path))
        if "smplx" in self.load:
            data.update(kindyn.load_smplx(scene, human_dir, object_ids, n,
                                          gravity_path=gravity_path))
        return data

    def _load_pose_tokens(
        self, scene: str, object_ids: np.ndarray, n: int, valid_mask: np.ndarray,
    ) -> dict:
        """The scene's cached frozen pose tokens (int16 bf16 bits) and image sizes."""
        path = scene_io.pose_token_path(self.pose_token_dir, scene)
        if not path.is_file():
            raise FileNotFoundError(
                f"{scene}: no pose-token cache at {path} "
                "(scripts/data/precompute_pose_tokens.py)")
        cache = np.load(path)
        tokens = scene_io.rows_by_object_id(
            np.asarray(cache["tokens"], np.int16), cache["object_ids"], object_ids,
            scene, "pose-token cache")                                  # [P, N, C]
        cached_valid = scene_io.rows_by_object_id(
            np.asarray(cache["valid"], bool), cache["object_ids"], object_ids,
            scene, "pose-token cache")                                  # [P, N]
        img_wh = np.asarray(cache["img_wh"], np.int64)                  # [N, 2]
        if tokens.shape[1] != n or img_wh.shape != (n, 2):
            raise ValueError(
                f"{scene}: pose-token cache covers {tokens.shape[1]} frames / "
                f"img_wh {img_wh.shape}; the scene has {n}")
        missing = valid_mask & ~cached_valid
        if missing.any():
            raise ValueError(
                f"{scene}: {int(missing.sum())} valid person-frames have no cached "
                "pose token — the cache is stale; re-run precompute_pose_tokens.py")
        return {"pose_token": tokens, "img_wh": img_wh}

    # ------------------------------------------------------------------ frames

    def _frame(
        self, scene: str, data: dict, person: int, position: int, row: int,
        positions: np.ndarray,
    ) -> dict:
        oid = int(data["object_ids"][person])
        valid = bool(data["valid_mask"][person, position])
        image = img_wh = mask = None
        frame_path = data["frames_dir"] / f"{position:06d}.jpg"
        if self.pose_token_dir is not None:
            img_wh = tuple(int(v) for v in data["img_wh"][position])    # (W, H)
        elif self.embedding_dir is None:
            image = np.array(Image.open(frame_path).convert("RGB"), np.uint8)
        else:
            with Image.open(frame_path) as im:
                img_wh = im.size                                        # (W, H)
        if self.pose_token_dir is None:
            mask_path = data["mask_dir"] / f"{oid:02d}" / f"frame_{position:06d}.png"
            mask = (np.array(Image.open(mask_path), np.uint8)
                    if mask_path.is_file() else None)

        frame = {
            "image": image,
            "img_wh": img_wh,
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
            "contact_gt": data["contact_gt"][person, position],         # [6]
            "contact_valid": data["contact_valid"][person, position],   # [6]
            "contact_conf": data["contact_conf"][person, position],     # [6]
        }
        if self.pose_token_dir is not None:
            frame["geometry_only"] = True
            frame["pose_token"] = torch.from_numpy(
                np.ascontiguousarray(data["pose_token"][person, position])
            ).view(torch.bfloat16)
        elif self.embedding_dir is not None:
            bits = np.load(
                scene_io.embedding_path(self.embedding_dir, scene, oid, position))
            frame["embedding"] = torch.from_numpy(bits).view(torch.bfloat16)
        if "forces" in self.load:
            frame["force_gt"] = data["force_gt"][person, position]           # [6, 3]
            frame["force_contact"] = data["force_contact"][person, position]  # [6]
            frame["force_lever"] = data["force_lever"][person, position]     # [6, 3]
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
