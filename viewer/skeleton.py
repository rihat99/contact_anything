"""Canonical SMPL-X body-22 skeleton metadata used by the browser UI.

ClimbingVideos_v1 stores joint labels but no pose or projected joint positions.
The diagram is therefore deliberately canonical rather than an image overlay.
Its ordering and edges match SMPL-X; the 2-D positions are a normalized frontal
projection chosen for legibility in a small SVG.
"""
from __future__ import annotations

from contact.targets import SMPLX_BODY_22

JOINT_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11),
    (9, 12), (9, 13), (9, 14),
    (12, 15),
    (13, 16), (14, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21),
)

# Coordinates are percentages in a 100 x 108 viewBox.
JOINT_COORDS: tuple[tuple[float, float], ...] = (
    (50, 54), (57, 60), (43, 60), (50, 47),
    (58, 76), (42, 76), (50, 40), (57, 92),
    (43, 92), (50, 34), (61, 99), (39, 99),
    (50, 24), (57, 28), (43, 28), (50, 11),
    (65, 26), (35, 26), (79, 30), (21, 30),
    (92, 27), (8, 27),
)


def skeleton_payload() -> dict:
    return {
        "joint_names": list(SMPLX_BODY_22),
        "joint_coords": [list(p) for p in JOINT_COORDS],
        "edges": [list(e) for e in JOINT_EDGES],
    }
