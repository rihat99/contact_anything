"""Compact experiment-logging surface."""
from __future__ import annotations

from contact.tracking import RunLogger


class _Writer:
    def __init__(self):
        self.values = []

    def add_scalar(self, key, value, step):
        self.values.append((key, value, step))


def test_tensorboard_exact_metric_filter_hides_detailed_streams():
    logger = RunLogger.__new__(RunLogger)
    logger.tb = _Writer()
    logger.wandb = None
    logger.tensorboard_metrics = frozenset({"train/loss", "test/joint/f1"})

    logger.log({
        "train/loss": 0.4,
        "train/frames_per_sec": 12.0,
        "test/joint/f1": 0.8,
        "test/joint/left_hand/f1": 0.7,
    }, step=5)

    assert logger.tb.values == [
        ("train/loss", 0.4, 5),
        ("test/joint/f1", 0.8, 5),
    ]
