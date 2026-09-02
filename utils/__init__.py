"""Shared helpers with no dependency on the model, the data or the trainer.

* :mod:`utils.geometry` — torch SO(3) logs and the camera -> world lift of the
  model's own predictions (metres, OpenCV extrinsics).
* :mod:`utils.metrics` — masked contact confusion counts and the Pearson / RMSE
  closed forms over additive sufficient statistics.
* :mod:`utils.distributed` — rank / world-size queries and the exact DDP global
  weighted mean.
"""
