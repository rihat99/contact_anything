from __future__ import annotations

import numpy as np
import torch

from contact.config import load_config
from contact.targets import TargetSpec
from scripts.demo_climbing_videos import _expand_for_skeleton, _reduced_label_arrays


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
