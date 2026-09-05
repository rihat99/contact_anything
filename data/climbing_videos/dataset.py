"""``ClimbingVideosDataset`` — windowed corpus clips with the requested GT.

Composes :mod:`data.climbing_videos.scene` (frames, masks, boxes, cameras,
six-group contact labels), :mod:`data.climbing_videos.kindyn` (GT forces, the
fitted gravity, the pelvis gravity-view twist, the SMPL-X body GT) and
:mod:`data.climbing_videos.mhr_gt` (MHR pose ``q``, bones, scales, keypoints,
vertices) and :mod:`data.climbing_videos.sapiens` (2D keypoint INPUT) into the
frame schema documented in :mod:`data.base`.

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
from . import camera, kindyn, mhr_gt, sapiens, scene as scene_io

#: Default Gaussian width (SECONDS) applied to the root trajectory before
#: differentiating, so the label bandwidth does not depend on the scene's fps.
MOTION_SMOOTH_SEC = 0.12
#: Default per-frame world-acceleration cut (m/s^2) flagged as an outlier.
MOTION_OUTLIER_ACC_MS2 = 50.0


class ClimbingVideosDataset(ClipDataset):
    """Windowed ``(scene, person)`` clips of the ClimbingVideos corpus.

    :param root: corpus root containing ``scenes/``, ``features/``, ``frames/``.
    :param scenes: explicit scene ids; ``None`` discovers them from the DB
        (train split, or the annotated test scenes).
    :param split: ``"train"`` (automatic labels, jittered windows) or ``"test"``
        (manual annotation, fixed windows).
    :param contact_level: label readout level — ``contacts_1`` or ``contacts_2``.
    :param load: signal groups to emit, a subset of
        ``{"forces", "motion", "pose", "keypoints", "smplx", "keypoints2d", "camera"}``.
    :param embedding_dir: precomputed-embedding root (``features/embedding``).
    :param motion_smooth_sec: Gaussian width in seconds applied to the root
        trajectory before differentiating.
    :param motion_outlier_acc_ms2: world ``|a|`` above which a motion row is
        flagged an outlier; ``0`` disables the flag.
    :param motion_linear_frame: frame of the pelvis linear vel/acc target —
        ``"gravity_view"`` or ``"body"`` (the BVR root body twist).
    :param motion_root_source: trajectory the twist is differentiated from —
        ``"mhr"`` (MHR pseudo-GT mean-hips + root) or ``"smplx"`` (kindyn
        SMPL-X pelvis + root rotation, the body the SMPL-X head predicts).

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
        full_scenes: bool = False,
        max_frames: Optional[int] = None,
        motion_smooth_sec: float = MOTION_SMOOTH_SEC,
        motion_outlier_acc_ms2: float = MOTION_OUTLIER_ACC_MS2,
        motion_linear_frame: str = "gravity_view",
        motion_root_source: str = "mhr",
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
        self.motion_smooth_sec = float(motion_smooth_sec)
        if not np.isfinite(self.motion_smooth_sec) or self.motion_smooth_sec < 0:
            raise ValueError(
                f"motion_smooth_sec must be finite and >= 0; got {motion_smooth_sec!r}")
        self.motion_outlier_acc_ms2 = float(motion_outlier_acc_ms2)
        if motion_linear_frame not in kindyn.MOTION_LINEAR_FRAMES:
            raise ValueError(
                f"motion_linear_frame must be one of {kindyn.MOTION_LINEAR_FRAMES}; "
                f"got {motion_linear_frame!r}")
        self.motion_linear_frame = str(motion_linear_frame)
        if motion_root_source not in kindyn.MOTION_ROOT_SOURCES:
            raise ValueError(
                f"motion_root_source must be one of {kindyn.MOTION_ROOT_SOURCES}; "
                f"got {motion_root_source!r}")
        self.motion_root_source = str(motion_root_source)
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
        if "forces" in self.load:
            data.update(kindyn.load_forces(scene, human_dir, object_ids, n))
        if "motion" in self.load:
            # The gravity-view frame and the vertical diagnostics are built from
            # this vector, so it must be the FITTED one whether or not forces
            # load (which set it too): the corpus tilt reaches 61 degrees.
            gravity = kindyn.fitted_gravity_world(scene, human_dir)
            data["gravity_world"] = gravity.astype(np.float32)
            if self.motion_root_source == "smplx":
                root7, src_valid = kindyn.smplx_motion_root(
                    kindyn.load_smplx(scene, human_dir, object_ids, n))
            else:
                root7, src_valid = mhr_gt.motion_root(
                    scene, human_dir, object_ids, n, data["fps"])
            data.update(kindyn.motion_targets(
                root7, src_valid, data["fps"], gravity, data["extrinsics"],
                self.motion_smooth_sec, self.motion_outlier_acc_ms2,
                linear_frame=self.motion_linear_frame))
        if "pose" in self.load:
            data.update(mhr_gt.load_pose(scene, human_dir, object_ids, n))
        if "keypoints" in self.load:
            data.update(mhr_gt.load_keypoints(scene, human_dir, object_ids, n))
        if "smplx" in self.load:
            data.update(kindyn.load_smplx(scene, human_dir, object_ids, n))
        if "keypoints2d" in self.load:
            data.update(sapiens.load_keypoints2d(self.root, scene, object_ids, n))
        return data

    # ------------------------------------------------------------------ frames

    def _frame(
        self, scene: str, data: dict, person: int, position: int, row: int,
        positions: np.ndarray,
    ) -> dict:
        oid = int(data["object_ids"][person])
        valid = bool(data["valid_mask"][person, position])
        image = img_wh = None
        frame_path = data["frames_dir"] / f"{position:06d}.jpg"
        if self.embedding_dir is None:
            image = np.array(Image.open(frame_path).convert("RGB"), np.uint8)
        else:
            with Image.open(frame_path) as im:
                img_wh = im.size                                        # (W, H)
        mask_path = data["mask_dir"] / f"{oid:02d}" / f"frame_{position:06d}.png"
        mask = np.array(Image.open(mask_path), np.uint8) if mask_path.is_file() else None

        frame = {
            "image": image,
            "img_wh": img_wh,
            "mask": mask,
            "bbox": data["bbox"][person, position],                     # [4] xyxy
            "cam_int": data["intrinsics"][position],                    # [3, 3]
            "cam_from_world": data["extrinsics"][position],             # [4, 4]
            "gravity_world": data["gravity_world"],                     # [3]
            "cam_jump_m": float(np.linalg.norm(
                data["cam_centers"][position]
                - data["cam_centers"][int(positions[row - 1])]
            )) if row > 0 and valid else 0.0,
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
        if self.embedding_dir is not None:
            bits = np.load(
                scene_io.embedding_path(self.embedding_dir, scene, oid, position))
            frame["embedding"] = torch.from_numpy(bits).view(torch.bfloat16)
        if "forces" in self.load:
            frame["force_gt"] = data["force_gt"][person, position]           # [6, 3]
            frame["force_contact"] = data["force_contact"][person, position]  # [6]
            frame["force_lever"] = data["force_lever"][person, position]     # [6, 3]
            frame["force_conf"] = float(data["force_conf"][person, position])
            frame["force_valid"] = valid and bool(data["force_valid"][person, position])
        if "motion" in self.load:
            frame["motion_gt"] = data["motion_gt"][person, position]         # [1, 12]
            frame["motion_outlier"] = data["motion_outlier"][person, position]
            frame["motion_rot"] = data["motion_rot"][person, position]       # [3, 3]
            frame["motion_lin_rot"] = data["motion_lin_rot"][person, position]
            frame["motion_omega"] = data["motion_omega"][person, position]   # [3]
            frame["motion_valid"] = valid and bool(
                data["motion_valid"][person, position])
            frame["motion_root_pos"] = data["motion_root_pos"][person, position]
            frame["motion_root_valid"] = valid and bool(
                data["motion_root_valid"][person, position])
        if "pose" in self.load:
            frame["pose_gt_q"] = data["pose_gt_q"][person, position]         # [132]
            frame["pose_valid"] = valid and bool(data["pose_valid"][person, position])
            frame["pose_identity"] = data["pose_identity"][person]           # [45]
            frame["pose_gt_bones"] = data["pose_gt_bones"][person, position]  # [6]
            frame["pose_gt_scale"] = data["pose_gt_scale"][person]           # [68]
        if "keypoints" in self.load:
            frame["kp3d_world"] = data["kp3d_world"][person, position]       # [70, 3]
            frame["kp_valid"] = valid and bool(data["kp_valid"][person, position])
            frame["vert_gt_world"] = data["vert_gt_world"][person, position]  # [V, 3]
            frame["vert_valid"] = valid and bool(data["vert_valid"][person, position])
            frame["vert_indices"] = data["vert_indices"]                     # [V]
        if "smplx" in self.load:
            frame["smplx_joints_world"] = data["smplx_joints_world"][person, position]
            frame["smplx_root_rot"] = data["smplx_root_rot"][person, position]   # [3, 3]
            frame["smplx_body_rot"] = data["smplx_body_rot"][person, position]   # [21, 3, 3]
            frame["smplx_hand_rot"] = data["smplx_hand_rot"][person, position]   # [30, 3, 3]
            frame["smplx_betas"] = data["smplx_betas"][person]                   # [10]
            frame["smplx_valid"] = valid and bool(data["smplx_valid"][person, position])
        if "keypoints2d" in self.load:
            frame["kp2d_in"] = data["kp2d_in"][person, position]             # [70, 3]
            frame["kp2d_in_valid"] = valid and bool(
                data["kp2d_in_valid"][person, position])
        if "camera" in self.load:
            frame["cam_twist"] = self._camera_twist(data, row, positions)     # [6]
        return frame

    @staticmethod
    def _camera_twist(data: dict, row: int, positions: np.ndarray) -> np.ndarray:
        """The camera's own twist at clip row ``row`` (see :mod:`.camera`)."""
        extrinsics, fps = data["extrinsics"], data["fps"]
        seconds = data["frame_indices"][positions] / fps
        has_prev, has_next = row > 0, row < len(positions) - 1
        return camera.row_camera_twist(
            extrinsics[int(positions[row - 1])] if has_prev else None,
            extrinsics[int(positions[row])],
            extrinsics[int(positions[row + 1])] if has_next else None,
            float(seconds[row] - seconds[row - 1]) if has_prev else 0.0,
            float(seconds[row + 1] - seconds[row]) if has_next else 0.0,
        )
