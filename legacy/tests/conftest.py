"""Archived tests for the retired ClimbingVideos_v1 loader — not collectable.

These files import ``contact.data.climbing_videos`` / ``scripts.
demo_climbing_videos``, both of which now live in ``legacy/``; collecting them
would fail at import. The suite runs from ``tests/`` only; this guard also
keeps a bare ``pytest`` at the repo root from picking them up.
"""

collect_ignore_glob = ["test_*.py"]
