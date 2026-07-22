from __future__ import annotations

import torch

from scripts.evaluate import evaluate


class _ContactModel(torch.nn.Module):
    def _initialize_batch(self, batch):
        pass

    def forward_step(self, batch, decoder_type="body"):
        assert decoder_type == "body"
        return {"contact": {"joint_logits": batch["logits"]}}


def test_evaluate_reports_named_outputs_f2_and_threshold_curve():
    loader = [{
        "logits": torch.tensor([[10.0, -10.0], [-10.0, 10.0]]),
        "targets": {"joint": {
            "gt": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "mask": torch.ones(2, 2),
        }},
    }]
    result = evaluate(
        _ContactModel(),
        loader,
        ["joint"],
        "cpu",
        threshold=0.5,
        curve_thresholds=(0.3, 0.7),
        output_names={"joint": ("left_hand", "right_hand")},
    )["joint"]

    assert result["f1"] > 0.999
    assert result["f2"] > 0.999
    assert set(result["per_output"]) == {"left_hand", "right_hand"}
    assert all(value["f2"] > 0.999 for value in result["per_output"].values())
    assert [point["threshold"] for point in result["threshold_curve"]] == [0.3, 0.5, 0.7]


def test_evaluate_center_policy_ignores_wrong_noncenter_rows():
    # Two T=5 clips. Only flattened rows 2 and 7 are correct; every other row is
    # deliberately wrong and must not affect center-only metrics.
    gt = torch.zeros(10, 2)
    gt[:, 0] = 1.0
    logits = torch.full((10, 2), -10.0)
    logits[:, 1] = 10.0
    logits[[2, 7], 0] = 10.0
    logits[[2, 7], 1] = -10.0
    loader = [{
        "seq_len": 5,
        "logits": logits,
        "targets": {"joint": {"gt": gt, "mask": torch.ones(10, 2)}},
    }]

    result = evaluate(
        _ContactModel(), loader, ["joint"], "cpu", target_frame="center",
    )["joint"]
    assert result["f1"] > 0.999
    assert (result["tp"], result["fp"], result["fn"]) == (2, 0, 0)
