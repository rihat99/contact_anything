"""Sapiens 2D keypoints of the corpus: the model's optional keypoint INPUT.

``features/sapiens/<shard>/<scene>/pose.npz`` holds, per tracked person and
frame, the 308 Goliath keypoints in full-image pixels with a detector score.
The first 70 Goliath keypoints are the MHR70 set in MHR order (checked against
the projected MHR GT: 7.9 px median over 40 train scenes), so the loader emits
those 70 and the model picks its subset by MHR70 index — the same indices
address the frozen readout's ``pred_keypoints_2d`` at deployment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .scene import rows_by_object_id, scene_shard

#: The keypoint scheme whose leading 70 entries are the MHR70 set.
KEYPOINT_SCHEME = "goliath_308"
NUM_GOLIATH = 308
NUM_MHR70 = 70


def load_keypoints2d(root: Path, scene: str, object_ids: np.ndarray, n: int) -> dict:
    """Sapiens keypoints of one scene, MHR70 subset, pixels + score.

    A keypoint with a non-finite coordinate or score is emitted as a zero row
    with score 0 (the model drops zero-score keypoints); frames sapiens did not
    track are all-zero with ``kp2d_in_valid`` False.

    :returns: ``kp2d_in (P, N, 70, 3)`` float32 ``[u, v, score]`` full-image
        pixels, ``kp2d_in_valid (P, N)`` bool.
    """
    path = root / "features" / "sapiens" / scene_shard(scene) / scene / "pose.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{scene}: no sapiens keypoints at {path}")
    pose = np.load(path, allow_pickle=True)
    scheme = str(pose["keypoint_scheme"])
    if scheme != KEYPOINT_SCHEME:
        raise ValueError(
            f"{scene}: sapiens keypoint_scheme {scheme!r}, expected {KEYPOINT_SCHEME!r}")
    if int(pose["num_frames"]) != n:
        raise ValueError(
            f"{scene}: sapiens has {int(pose['num_frames'])} frames, scene has {n}")
    keypoints = rows_by_object_id(
        np.asarray(pose["keypoints"], np.float32), pose["object_ids"], object_ids,
        scene, "sapiens keypoints")                                    # [P, N, 308, 2]
    scores = rows_by_object_id(
        np.asarray(pose["keypoint_scores"], np.float32), pose["object_ids"],
        object_ids, scene, "sapiens keypoint_scores")                  # [P, N, 308]
    valid = rows_by_object_id(
        np.asarray(pose["valid_mask"], bool), pose["object_ids"], object_ids,
        scene, "sapiens valid_mask")                                   # [P, N]
    n_people = len(object_ids)
    if keypoints.shape != (n_people, n, NUM_GOLIATH, 2) or scores.shape != (
            n_people, n, NUM_GOLIATH) or valid.shape != (n_people, n):
        raise ValueError(
            f"{scene}: sapiens arrays {keypoints.shape} / {scores.shape} / "
            f"{valid.shape} do not match ({n_people}, {n}, {NUM_GOLIATH})")
    kp = np.concatenate(
        [keypoints[:, :, :NUM_MHR70], scores[:, :, :NUM_MHR70, None]], axis=-1)
    finite = np.isfinite(kp).all(axis=-1)
    kp = np.where((finite & valid[:, :, None])[..., None], kp, 0.0)
    return {"kp2d_in": kp.astype(np.float32), "kp2d_in_valid": valid}
