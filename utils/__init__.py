"""Shared helpers with no dependency on the model, the data or the trainer.

* :mod:`utils.geometry` — camera parametrizations, projection and the camera ->
  world lift (metres, OpenCV extrinsics).
* :mod:`utils.gvhmr_metrics` — GVHMR's global trajectory metrics.
* :mod:`utils.metrics` — masked contact confusion counts and the mean over
  additive sufficient statistics.
* :mod:`utils.distributed` — rank / world-size queries and the exact DDP global
  weighted mean.
"""
