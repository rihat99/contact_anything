"""Physics bridge — frozen SAM-3D-Body outputs to a BetterHuman MHR body.

:class:`MHRAdapter` maps the frozen model's per-frame MHR parameters and the
dataset's camera extrinsics onto a shaped MHR body and a world-frame ``q``
trajectory in the metric reconstruction world. The RNEA physics loss (step 06)
consumes its output.
"""
from __future__ import annotations

from .adapter import EXTREMITY_OUTPUT_NAMES, MHRAdapter
from .loss import PhysicsLoss

__all__ = ["EXTREMITY_OUTPUT_NAMES", "MHRAdapter", "PhysicsLoss"]
