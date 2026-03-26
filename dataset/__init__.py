"""
Datasets for SAM-3D-Body contact training.

    DamonDataset              — raw images, MHR or SMPL topology, three modes
    DamonPrecomputedDataset   — precomputed DINOv3 features, MHR topology
    build_datasets(cfg)       — config-driven factory returning (train, val, test)
"""
from .damon_dataset import DamonDataset, DamonPrecomputedDataset, build_datasets

__all__ = ['DamonDataset', 'DamonPrecomputedDataset', 'build_datasets']
