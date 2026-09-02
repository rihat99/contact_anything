"""Clip datasets: the windowing contract and the per-frame schema.

A dataset **item** is a *clip*: a list of ``T`` per-frame dicts of one
``(scene, person)`` at a fixed source-frame stride. :class:`ClipDataset` owns
everything that does not depend on what a scene contains — how a scene is cut
into windows, the stateless per-epoch jitter of the training windows, and the
single-clip whole-scene evaluation protocol — and delegates the two
data-dependent steps to subclasses (:meth:`ClipDataset._load_scene`,
:meth:`ClipDataset._frame`).

Frame schema
------------

Every frame dict carries this block:

===================  ==========================  =================================
key                  type / shape                meaning
===================  ==========================  =================================
``image``            uint8 ``(H, W, 3)`` | None   full frame RGB (None when cached)
``img_wh``           ``(W, H)`` | None           full-frame size when no ``image``
``mask``             uint8 ``(H, W)`` | None     person mask
``bbox``             float ``(4,)``              person box, xyxy px
``cam_int``          float ``(3, 3)``            intrinsics of the full frame
``cam_from_world``   float ``(4, 4)``            OpenCV extrinsics, metric
``gravity_world``    float ``(3,)``              fitted unit DOWN vector (per scene)
``cam_jump_m``       float                       camera-centre displacement (m) from
                                                 the previous SAMPLED clip frame
``frame_pos_sec``    float                       seconds since the clip's first frame
``frame_index``      int                         source frame index
``frame_valid``      bool                        tracked with a usable box (cameras
                                                 are validated per scene)
``key``              str                         ``"{scene}#{oid}@{frame_index}"``
``embedding``        bf16 ``(1280, 32, 32)``     frozen-backbone cache (optional)
``contact_gt``       float ``(6,)``              six-group contact label (0/1)
``contact_valid``    float ``(6,)``              1 where the label is supervised
``contact_conf``     float ``(6,)``              label confidence in ``[0, 1]``
===================  ==========================  =================================

plus, per requested signal group (``load``):

``forces``
    ``force_gt`` ``(6, 3)`` body-weight units in the body-root frame,
    ``force_contact`` ``(6,)`` bool, ``force_lever`` ``(6, 3)`` metres in the
    same frame, ``force_conf`` float, ``force_valid`` bool.
``motion``
    ``motion_gt`` ``(1, 12)`` standardizable pelvis twist (gravity-view linear
    vel/acc, body angular vel/acc), ``motion_outlier`` ``(1,)`` bool,
    ``motion_rot`` ``(3, 3)`` world-from-root, ``motion_lin_rot`` ``(3, 3)``
    world-from-gravity-view, ``motion_omega`` ``(3,)`` body angular velocity,
    ``motion_valid`` bool, ``motion_root_pos`` ``(3,)`` world, ``motion_root_valid`` bool.
``pose``
    ``pose_gt_q`` ``(132,)`` MHR world configuration, ``pose_valid`` bool,
    ``pose_identity`` ``(45,)``, ``pose_gt_bones`` ``(6,)``, ``pose_gt_scale`` ``(68,)``.
``keypoints``
    ``kp3d_world`` ``(70, 3)`` metres, ``kp_valid`` bool, ``vert_gt_world``
    ``(V, 3)``, ``vert_valid`` bool, ``vert_indices`` ``(V,)`` int64
    (scene-constant).
``smplx``
    ``smplx_joints_world`` ``(22, 3)`` metres (row 0 = pelvis), ``smplx_root_rot``
    ``(3, 3)`` world-from-root, ``smplx_body_rot`` ``(21, 3, 3)`` parent-local,
    ``smplx_betas`` ``(10,)`` per person, ``smplx_valid`` bool.

The six contact/force groups are ``left_hand, right_hand, left_foot (toe),
right_foot, left_ankle (heel), right_ankle`` in that fixed order everywhere.

An inference-only dataset (:mod:`data.reconstruction`) emits the input half of
the table and no label at all; the collate stacks whatever is there, so long as
every frame of a batch agrees.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, NamedTuple, Optional, Sequence

import numpy as np
from torch.utils.data import Dataset

#: Frame rate (Hz) the ``auto`` stride normalises scenes to, so a ``T``-frame
#: clip spans the same physical time at every corpus fps.
REFERENCE_FPS = 25.0

SIGNAL_GROUPS = frozenset({"forces", "motion", "pose", "keypoints", "smplx"})


class Clip(NamedTuple):
    """One windowed item: ``frames`` rows of ``person`` starting at ``start``.

    ``jitter_range`` is the width of the stateless per-epoch start jitter (1 =
    fixed). The first four fields are the tuple scripts consume.
    """

    scene: str
    person: int
    start: int
    frames: int
    jitter_range: int


def valid_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    """``(start, length)`` of every contiguous ``True`` run in a 1-D mask, in order."""
    runs = []
    start = None
    for i, v in enumerate(valid.tolist() + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - start))
            start = None
    return runs


def longest_valid_run(valid: np.ndarray) -> tuple[int, int]:
    """Start and length of the longest ``True`` run in a 1-D mask (``(0, 0)`` if none)."""
    return max(valid_runs(valid), key=lambda run: run[1], default=(0, 0))


class ClipDataset(Dataset, ABC):
    """Windowed ``(scene, person)`` clips over a scene-indexed source.

    Training windows tile each scene with step ``T * stride`` and are jittered
    per epoch from ``(seed, epoch, item_index)`` — statelessly, so a resumed run
    reproduces the same windows without carrying RNG state. A window containing
    an invalid frame is dropped (every temporal row needs a real crop), and a
    jittered start that would cross a tracking gap falls back to its base.

    Evaluation uses ONE clip per ``(scene, person)``: the longest contiguous
    valid run, strided like training and capped at ``max_frames`` (the frozen
    per-frame path costs ~0.1 GiB per frame at inference, so uncapped 500-frame
    scenes OOM a 48 GB card; truncation keeps the run's head).

    :param scenes: scene ids to load, in order.
    :param split: ``"train"`` (jittered tiles) or ``"test"``.
    :param clip_frames: window length ``T``.
    :param stride: source-frame stride, or ``"auto"`` for the per-scene
        ``max(1, round(fps / 25))`` that holds a clip's PHYSICAL span roughly
        constant across the corpus's 24-60 fps range.
    :param jitter: enable the train-window jitter.
    :param seed: seed of the window jitter.
    :param full_scenes: use the single-clip whole-scene protocol (eval only).
    :param max_frames: row cap of a full-scene clip.
    """

    name: str = "clips"

    def __init__(
        self,
        scenes: Sequence[str],
        *,
        split: str = "train",
        clip_frames: int = 8,
        stride: int | str = "auto",
        jitter: bool = True,
        seed: int = 42,
        full_scenes: bool = False,
        max_frames: Optional[int] = None,
    ):
        super().__init__()
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test'; got {split!r}")
        if full_scenes and split == "train":
            raise ValueError("full_scenes is an eval protocol; split must be 'test'")
        if stride != "auto" and (isinstance(stride, str) or int(stride) < 1):
            raise ValueError(f"stride must be a positive int or 'auto'; got {stride!r}")
        if max_frames is not None and int(max_frames) < 1:
            raise ValueError("max_frames must be a positive int or None")
        self.split = split
        self.T = int(clip_frames)
        self.stride = "auto" if stride == "auto" else int(stride)
        self.jitter = bool(jitter) and split == "train"
        self.seed = int(seed)
        self.full_scenes = bool(full_scenes)
        self.max_frames = None if max_frames is None else int(max_frames)
        self._epoch = 0
        self._scenes: dict[str, dict] = {}
        self.clips: list[Clip] = []
        for scene in scenes:
            self._scenes[scene] = self._load_scene(scene)
            self.clips.extend(self._scene_clips(scene))

    # ------------------------------------------------------------------ hooks

    @abstractmethod
    def _load_scene(self, scene: str) -> dict:
        """Load one scene's metadata and ground truth.

        Must provide at least ``valid_mask`` ``(P, N)`` bool, ``fps`` float,
        ``frame_indices`` ``(N,)`` int64 and ``object_ids`` ``(P,)``.
        """

    @abstractmethod
    def _frame(
        self, scene: str, data: dict, person: int, position: int, row: int,
        positions: np.ndarray,
    ) -> dict:
        """Build the frame dict of clip row ``row`` (source frame ``position``)."""

    # ------------------------------------------------------------------ access

    @property
    def scenes(self) -> list[str]:
        """Scene ids in this dataset, in load order."""
        return list(self._scenes)

    def scene_data(self, scene: str) -> dict:
        """The loaded scene dict (cameras, ``valid_mask``, labels, ...) of ``scene``."""
        return self._scenes[scene]

    # ------------------------------------------------------------------ windows

    def scene_stride(self, scene: str) -> int:
        """Frame stride used inside this scene's clips (resolves ``"auto"``)."""
        if self.stride != "auto":
            return self.stride
        return max(1, int(round(float(self._scenes[scene]["fps"]) / REFERENCE_FPS)))

    def _scene_clips(self, scene: str) -> Iterable[Clip]:
        data = self._scenes[scene]
        stride = self.scene_stride(scene)
        valid_mask = data["valid_mask"]
        num_frames = len(data["frame_indices"])
        if self.full_scenes:
            for person in range(valid_mask.shape[0]):
                base, run_len = longest_valid_run(valid_mask[person])
                if run_len < 1:
                    continue
                frames = (run_len - 1) // stride + 1
                if self.max_frames is not None:
                    frames = min(frames, self.max_frames)
                yield Clip(scene, person, base, frames, 1)
            return
        span = (self.T - 1) * stride
        step = self.T * stride
        max_start = num_frames - 1 - span
        if max_start < 0:
            return                              # scene too short for one window
        for person in range(valid_mask.shape[0]):
            if self.split == "train":
                for base in range(0, max_start + 1, step):
                    positions = base + np.arange(self.T) * stride
                    if valid_mask[person, positions].all():
                        yield Clip(scene, person, base, self.T,
                                   max(1, min(step, max_start - base + 1)))
                continue
            # Tiled eval (non-full-scene datasets, e.g. reconstruction out-trees)
            # covers every valid frame: each contiguous valid run is tiled on its
            # own, with a terminal window over the run's tail (a few boundary
            # frames score twice).
            for run_start, run_len in valid_runs(valid_mask[person]):
                run_max_start = run_start + run_len - 1 - span
                if run_max_start < run_start:
                    continue                    # run too short for one window
                bases = list(range(run_start, run_max_start + 1, step))
                if bases[-1] != run_max_start:
                    bases.append(run_max_start)
                for base in bases:
                    yield Clip(scene, person, base, self.T, 1)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch the stateless window jitter is derived from."""
        self._epoch = int(epoch)

    # ------------------------------------------------------------------ access

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> list[dict]:
        clip = self.clips[index]
        data = self._scenes[clip.scene]
        stride = self.scene_stride(clip.scene)
        start = clip.start
        if self.jitter and clip.jitter_range > 1:
            rng = np.random.default_rng([self.seed, self._epoch, index])
            start = clip.start + int(rng.integers(0, clip.jitter_range))
        positions = start + np.arange(clip.frames) * stride
        # The base window is all-valid by construction; a jittered one that
        # crosses a tracking gap falls back so invalid bboxes never get cropped.
        if start != clip.start and not data["valid_mask"][clip.person, positions].all():
            positions = clip.start + np.arange(clip.frames) * stride
        return [
            self._frame(clip.scene, data, clip.person, int(pos), row, positions)
            for row, pos in enumerate(positions)
        ]
