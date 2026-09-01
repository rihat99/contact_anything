"""Contact/force/motion/pose library for the SAM-3D-Body fork.

Model build + freeze/eval pinning (``model``), losses (``losses``,
``force_supervision``, ``motion_supervision``, ``pose_supervision``,
``keypoint_supervision``, ...), checkpoint I/O (``checkpoint``) and the
climbing-corpus datasets (``data``).
"""

from .data import ClimbingCorpusDataset

__all__ = ["ClimbingCorpusDataset"]
