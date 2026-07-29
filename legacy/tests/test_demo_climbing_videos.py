from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from contact.config import load_config
from contact.data.climbing_videos import ClimbingVideosDataset
from contact.targets import NUM_BODY_22, TargetSpec
from scripts.demo_climbing_videos import (
    _clip_bucket,
    _expand_for_skeleton,
    _reduced_label_arrays,
    _video_dataset,
)
from test_climbing_videos import _train_scene


def _spec() -> TargetSpec:
    return TargetSpec.from_config(load_config("configs/climbing_videos_joint.yaml"))


def test_extremity_predictions_expand_to_wrist_and_ankle_foot_nodes():
    states, confidence, supervised = _expand_for_skeleton(
        np.array([True, False, True, False]),
        np.array([0.9, 0.8, 0.7, 0.6]),
        np.ones(4, dtype=bool),
        _spec(),
    )
    assert states[[20, 21, 7, 10, 8, 11]].tolist() == [True, False, True, True, False, False]
    assert confidence[[7, 10]].tolist() == [0.7, 0.7]
    assert supervised.sum() == 6


def test_demo_reduction_uses_same_target_semantics_as_training():
    contact = torch.zeros(22)
    contact[7] = 1.0
    supervised = torch.ones(22)
    confidence = torch.ones(22)
    confidence[7] = 0.35
    gt, sup, conf = _reduced_label_arrays({
        "joint_contact": contact,
        "joint_supervised": supervised,
        "joint_confidence": confidence,
    }, _spec())
    assert gt.tolist() == [0.0, 0.0, 1.0, 0.0]
    assert sup.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert conf[2] == np.float32(0.35)


def test_clip_bucket_uses_center_frame(tmp_path):
    """Stratification must bucket by the RENDERED (center) frame — the source
    frame ``base + (T // 2) * stride`` that ``main`` renders — not the base frame,
    which at T=16/stride 2 lies 16 source frames away from the figure shown."""
    n = 8
    jc = np.zeros((1, n, NUM_BODY_22), bool)
    jc[0, 4, 7] = True                     # contact ONLY at source frame 4
    _train_scene(tmp_path, "vid_0000", n, np.ones(n, bool), jc=jc)
    ds = ClimbingVideosDataset(tmp_path, scenes=["vid_0000"], mode="val",
                               frames_per_clip=4, frame_stride=2, jitter=False)
    # Val tiling: base 0 plus the terminal base 1. Rendered (center) source frames
    # are base + (T//2)*stride = 4 and 5; the BASE frames (0, 1) carry no contact,
    # so base-frame bucketing would put both clips in bucket 0.
    bases = [item[2] for item in ds._items]
    assert sorted(bases) == [0, 1]
    spec = types.SimpleNamespace(joint_set="smplx_body_22")
    buckets = {base: _clip_bucket(ds, spec, index)
               for index, base in enumerate(bases)}
    assert buckets[0] == 1                 # center source frame 4 IS in contact
    assert buckets[1] == 0                 # center source frame 5 is not


def test_val_split_on_test_manifest_names_the_fix(tmp_path):
    """A checkpoint trained with data.eval_split=test holds a train/test manifest;
    asking the demo for --split val must say to use --split test."""
    _train_scene(tmp_path, "vid_0000", 6, np.ones(6, bool))
    dataset_cfg = tmp_path / "climbing.yaml"
    dataset_cfg.write_text(f"name: climbing_videos\ndata:\n  root: {tmp_path}\n")
    cfg = {"data": {"datasets": [
        {"name": "climbing_videos", "config": str(dataset_cfg)}]}}
    state = {"split_manifest": {
        f"video:{dataset_cfg}": {"train": ["vid_0000"], "test": []}}}
    with pytest.raises(RuntimeError, match="use --split test"):
        _video_dataset(cfg, state, "val")
