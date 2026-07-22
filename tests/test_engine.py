"""Shared contact-forward and temporal-supervision helpers."""
from __future__ import annotations

import pytest
import torch

from contact.engine import select_temporal_supervision


def _inputs(rows: int = 10, dims: int = 4):
    logits = torch.arange(rows * dims, dtype=torch.float32).reshape(rows, dims)
    logits.requires_grad_()
    targets = {
        "joint": {
            "gt": torch.arange(rows, dtype=torch.float32).unsqueeze(1).expand(-1, dims),
            "mask": torch.ones(rows, dims),
        }
    }
    return {"joint": logits}, targets


def test_all_frame_supervision_is_unchanged():
    logits, targets = _inputs()
    selected_logits, selected_targets = select_temporal_supervision(
        logits, targets, seq_len=5, target_frame="all")
    assert selected_logits is logits
    assert selected_targets is targets


def test_center_supervision_selects_one_middle_row_per_clip_and_gradient():
    logits, targets = _inputs()
    selected_logits, selected_targets = select_temporal_supervision(
        logits, targets, seq_len=5, target_frame="center")

    assert selected_logits["joint"].shape == (2, 4)
    assert torch.equal(selected_logits["joint"], logits["joint"][[2, 7]])
    assert selected_targets["joint"]["gt"][:, 0].tolist() == [2.0, 7.0]
    assert selected_targets["joint"]["mask"].shape == (2, 4)

    selected_logits["joint"].sum().backward()
    active_rows = logits["joint"].grad.abs().sum(dim=1).nonzero().flatten().tolist()
    assert active_rows == [2, 7]


def test_center_supervision_requires_odd_sequence_length():
    logits, targets = _inputs(rows=8)
    with pytest.raises(ValueError, match="odd seq_len"):
        select_temporal_supervision(logits, targets, seq_len=4, target_frame="center")


def test_center_supervision_rejects_inconsistent_flattened_rows():
    logits, targets = _inputs()
    targets["joint"]["mask"] = targets["joint"]["mask"][:-1]
    with pytest.raises(ValueError, match="row counts disagree"):
        select_temporal_supervision(logits, targets, seq_len=5, target_frame="center")

