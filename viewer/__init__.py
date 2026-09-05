"""Results viewer: predicted / GT / frozen SMPL-X bodies of the test scenes in viser.

* :mod:`viewer.bodies` — the three body sources posed in the metric world frame
  (numpy + BetterHuman FK, no viser),
* :mod:`viewer.loading` — one scene's cameras, footage, scene cloud and bodies,
* :mod:`viewer.scene` — the viser nodes of one scene and their per-frame update,
* :mod:`viewer.app` — the persistent sidebar and the :func:`~viewer.app.view_results`
  entry point (``scripts/view_results.py`` is the CLI).
"""
from __future__ import annotations

from .app import view_results

__all__ = ["view_results"]
