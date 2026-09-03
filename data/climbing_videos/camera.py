"""Camera self-motion of a clip row: the camera's own twist, an INPUT to the model.

The corpus extrinsics ``cam_from_world`` are metric, so between two sampled
rows the camera's rigid motion is known exactly. It is expressed the way BVR
expresses the body's motion (:func:`data.climbing_videos.kindyn.root_body_twist`):
``d[t] = se3_log(C_t C_{t+1}^{-1})`` — the NEXT camera pose seen from the
CURRENT camera frame — divided by the real elapsed seconds, so the input is a
twist in the current camera's axes (m/s, rad/s), invariant to the world origin
and azimuth and to the scene's fps. Interior rows use the central estimate
``(d[t-1] + d[t]) / (t_{t+1} - t_{t-1})``, the clip's first/last row the
one-sided one. A one-row clip gets zero.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .kindyn import se3_log_xyzw


def _relative_log(cam_a: np.ndarray, cam_b: np.ndarray) -> np.ndarray:
    """``se3_log(C_a C_b^{-1})``: camera ``b``'s pose expressed in camera ``a``. ``(6,)``."""
    rel = np.asarray(cam_a, np.float64) @ np.linalg.inv(np.asarray(cam_b, np.float64))
    quat = Rotation.from_matrix(rel[:3, :3]).as_quat()                 # xyzw
    return se3_log_xyzw(rel[:3, 3], quat)


def row_camera_twist(
    cam_prev: np.ndarray | None, cam: np.ndarray, cam_next: np.ndarray | None,
    dt_prev: float, dt_next: float,
) -> np.ndarray:
    """Camera twist of one clip row. ``-> (6,)`` float32 ``[linear m/s, angular rad/s]``.

    :param cam_prev: ``(4, 4)`` cam-from-world of the previous sampled row, or ``None``.
    :param cam: ``(4, 4)`` of this row.
    :param cam_next: ``(4, 4)`` of the next sampled row, or ``None``.
    :param dt_prev: seconds from the previous row to this one.
    :param dt_next: seconds from this row to the next.
    """
    total, span = np.zeros(6, np.float64), 0.0
    if cam_prev is not None:
        total += _relative_log(cam_prev, cam)
        span += float(dt_prev)
    if cam_next is not None:
        total += _relative_log(cam, cam_next)
        span += float(dt_next)
    if span <= 0.0:
        return np.zeros(6, np.float32)
    return (total / span).astype(np.float32)
