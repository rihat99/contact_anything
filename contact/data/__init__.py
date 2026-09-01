"""Datasets for the climbing corpus.

``ClimbingCorpusDataset`` reads the raw ClimbingVideos corpus directly (scene
DB + pre-extracted ``frames/`` tree + ``features/`` archives) and emits video
clips with joint-level contact, kindyn forces, motion twists and MHR pose
pseudo-GT. ``ReconstructionSceneDataset`` is the inference-only mirror for
BetterVideoReconstruction out-trees.
"""

from .climbing_corpus import ClimbingCorpusDataset
from .reconstruction_scenes import ReconstructionSceneDataset

__all__ = [
    "ClimbingCorpusDataset",
    "ReconstructionSceneDataset",
]
