from __future__ import annotations

import pytest
import torch

from contact.metrics import contact_counts_per_dim, prf1


def test_f2_weights_false_negatives_more_than_false_positives():
    one_fp = prf1({"tp": 4, "fp": 1, "fn": 0, "tn": 0})
    one_fn = prf1({"tp": 4, "fp": 0, "fn": 1, "tn": 0})
    assert one_fn["f2"] < one_fp["f2"]
    assert one_fp["f2"] == pytest.approx(20 / 21)
    assert one_fn["f2"] == pytest.approx(20 / 24)


def test_contact_counts_per_dim_respects_each_dimension_mask():
    logits = torch.tensor([[10.0, 10.0], [-10.0, -10.0]])
    gt = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    left, right = contact_counts_per_dim(logits, gt, mask)
    assert left == {"tp": 1, "fp": 0, "fn": 0, "tn": 1}
    assert right == {"tp": 0, "fp": 0, "fn": 1, "tn": 0}


def test_contact_counts_per_dim_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        contact_counts_per_dim(torch.zeros(2, 4), torch.zeros(2, 3), torch.zeros(2, 4))
