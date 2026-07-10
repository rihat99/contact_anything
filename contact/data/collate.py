"""Batch assembly — datasets -> SAM-3D-Body batch, images and clips alike.

Each dataset item is either a single frame-dict (still-image datasets) or a
**clip** (list of ``T`` frame-dicts, from :class:`ClimbingVideosDataset`). The
collate normalises stills to length-1 clips, asserts the batch is homogeneous in
``T``, flattens ``[B_clips, T, ...]`` to a model batch ``[B_clips*T, ...]``, runs
the SAM-3D-Body top-down transform per frame, and builds the per-target
``{name: {"gt": [B, D], "mask": [B, D]}}`` supervision from
:meth:`contact.targets.TargetSpec.assemble_batch`.

Mixed image+video training uses **batch-level interleaving**: one DataLoader per
T-group, interleaved by remaining length (:class:`InterleavedLoader`) — never a
mixed-T batch.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.transforms import ToTensor

from sam_3d_body.data.transforms import (
    Compose, GetBBoxCenterScale, TopdownAffine, VisionTransformWrapper,
)
from sam_3d_body.data.utils.prepare_batch import NoCollate

from ..targets import TargetSpec, validate_targets
from .climbing_images import ClimbingImagesDataset
from .climbing_videos import ClimbingVideosDataset, list_scenes
from .damon import DamonDataset
from .splits import group_train_val_split, train_val_indices, video_id_from_scene


# -------------------------------------------------------------------- transform

def _build_transform(image_size: Tuple[int, int]):
    """SAM-3D-Body's standard top-down crop pipeline at the model resolution."""
    return Compose([
        GetBBoxCenterScale(),
        TopdownAffine(input_size=image_size, use_udp=False),
        VisionTransformWrapper(ToTensor()),
    ])


def _process_sample(frame: dict, transform):
    """Run the SAM-3D-Body transform on a single frame dict (image/mask/bbox)."""
    img  = frame["image"]
    mask = frame["mask"]
    bbox = frame["bbox"]
    if bbox is None:
        raise RuntimeError(f"frame {frame.get('key', '?')} has no bbox")
    has_mask = mask is not None
    if mask is None:
        H, W = img.shape[:2]
        mask = np.zeros((H, W, 1), dtype=np.uint8)
    elif mask.ndim == 2:
        mask = mask[..., None]

    # mask_score>0 tells the model this is a real mask (sam3d_body.py:883-889);
    # a substituted all-zeros mask must score 0.0 so it is treated as "no mask".
    data_info = dict(
        img=img,
        bbox=np.asarray(bbox, dtype=np.float32),
        bbox_format="xyxy",
        mask=mask,
        mask_score=np.array(1.0 if has_mask else 0.0, dtype=np.float32),
    )
    out = transform(data_info)
    m = out["mask"]
    if m.ndim == 3:
        m = m[..., 0]
    out["mask"] = (m.astype(np.float32) / 255.0)
    return out


# -------------------------------------------------------------------- collate

def make_collate(image_size: Tuple[int, int], spec: TargetSpec):
    transform = _build_transform(image_size)

    def _collate(batch):
        clips = [item if isinstance(item, list) else [item] for item in batch]
        seq_len = len(clips[0])
        if any(len(c) != seq_len for c in clips):
            raise AssertionError("homogeneous-T batches only: all clips must share the same length")
        frames = [f for clip in clips for f in clip]   # flatten -> B = B_clips * T

        per_sample = [_process_sample(f, transform) for f in frames]
        cam_ints = torch.stack([
            torch.as_tensor(f["cam_int"], dtype=torch.float32) for f in frames
        ], dim=0)

        keys = ["img", "img_size", "ori_img_size", "bbox_center", "bbox_scale",
                "bbox", "affine_trans", "mask", "mask_score"]
        out = {}
        for k in keys:
            if k not in per_sample[0]:
                continue
            tensors = [
                s[k] if isinstance(s[k], torch.Tensor) else torch.as_tensor(s[k])
                for s in per_sample
            ]
            out[k] = torch.stack(tensors, dim=0).float().unsqueeze(1)   # [B, 1, ...]
        if "mask" in out and out["mask"].dim() == 4:
            out["mask"] = out["mask"].unsqueeze(2)                      # [B, 1, 1, H, W]

        out["person_valid"] = torch.ones((len(frames), 1))
        out["cam_int"]      = cam_ints
        out["img_ori"]      = [NoCollate(f["image"]) for f in frames]
        out["targets"]      = spec.assemble_batch(frames)
        out["seq_len"]      = seq_len
        out["frame_pos_sec"] = torch.tensor(
            [float(f.get("frame_pos_sec", 0.0)) for f in frames], dtype=torch.float32)
        out["frame_valid"] = torch.tensor(
            [bool(f.get("frame_valid", True)) for f in frames], dtype=torch.bool)
        return out

    return _collate


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    return value


def batch_to_device(batch: dict, device: str) -> dict:
    for k, v in batch.items():
        batch[k] = _to_device(v, device)
    return batch


# -------------------------------------------------------------------- interleave

class InterleavedLoader:
    """Interleave whole batches from several DataLoaders (one per T-group).

    Batches are drawn proportionally to each loader's remaining length, seeded
    per epoch, so image (T=1) and video (T) batches mix without ever forming a
    mixed-T batch. :meth:`set_epoch` reseeds the interleave order and forwards to
    any child dataset exposing ``set_epoch`` (the video jitter).
    """

    def __init__(self, loaders: list[DataLoader], seed: int = 42):
        self.loaders = loaders
        self.seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        for loader in self.loaders:
            ds = loader.dataset
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

    def __len__(self) -> int:
        return sum(len(loader) for loader in self.loaders)

    def __iter__(self):
        iters = [iter(loader) for loader in self.loaders]
        remaining = [len(loader) for loader in self.loaders]
        rng = random.Random(self.seed + self._epoch)
        while any(remaining):
            choice = rng.choices(range(len(iters)), weights=remaining)[0]
            try:
                yield next(iters[choice])
                remaining[choice] -= 1
            except StopIteration:
                remaining[choice] = 0


# -------------------------------------------------------------------- builders

def _build_image_dataset(spec: dict):
    """Construct one still-image sub-dataset from a ``data.datasets`` entry."""
    name = spec["name"]
    if name == "damon":
        return DamonDataset.from_config(spec["config"], split=spec.get("split", "trainval"))
    if name == "climbing":
        return ClimbingImagesDataset.from_config(spec["config"])
    raise ValueError(f"{name!r} is not a still-image dataset")


def _video_root(config: str) -> str:
    return yaml.safe_load(Path(config).read_text())["data"]["root"]


def _manifest_image_indices(manifest: dict, n: int) -> Tuple[list, list]:
    """Read the still-image split from a manifest; reject out-of-range indices."""
    if "images" not in manifest:
        raise ValueError("split manifest has no 'images' entry but the config has image datasets")
    train_idx = [int(i) for i in manifest["images"]["train"]]
    val_idx = [int(i) for i in manifest["images"]["val"]]
    referenced = set(train_idx) | set(val_idx)
    if referenced and max(referenced) >= n:
        raise ValueError(
            f"split manifest references image index {max(referenced)} but the current "
            f"dataset has only {n} items — the source data changed since the checkpoint")
    return train_idx, val_idx


def _manifest_video_ids(manifest: dict, key: str, scenes: list) -> Tuple[set, set]:
    """Read the video-group split from a manifest; reject ids missing from disk."""
    if key not in manifest:
        raise ValueError(f"split manifest has no {key!r} entry but the config lists that video dataset")
    train_vids = set(manifest[key]["train"])
    val_vids = set(manifest[key]["val"])
    present = {video_id_from_scene(sc) for sc in scenes}
    missing = (train_vids | val_vids) - present
    if missing:
        raise ValueError(
            f"split manifest {key!r} references source videos not present on disk: "
            f"{sorted(missing)} — the source data changed since the checkpoint")
    return train_vids, val_vids


def make_loaders(cfg: dict, image_size: Tuple[int, int], manifest: dict | None = None):
    """Build interleaved train + val loaders from ``data.datasets``.

    Still-image datasets (damon/climbing) are concatenated and split randomly by
    ``val_ratio``/``seed`` (T=1 clips). ClimbingVideos datasets are split by
    source video (no video crosses train/val) into windowed clips (T). Batches
    keep a fixed ``frames_per_batch`` budget: ``B_clips = frames_per_batch // T``.

    :param manifest: when given, splits are taken **from the manifest** instead of
        re-derived (``{"images": {"train": [idx...], "val": [idx...]},
        "video:<config>": {"train": [vid...], "val": [vid...]}}``); missing
        members raise. Used by evaluate/resume to reproduce the exact split a
        checkpoint was trained on rather than the current directory's derivation.
    :returns: ``(train_loader, val_loader, split_manifest)`` — the manifest is the
        one used (echoing the input when supplied), so callers can persist it.
    """
    dcfg = cfg["data"]
    specs = list(dcfg["datasets"])
    spec_obj = TargetSpec.from_config(cfg)
    collate = make_collate(image_size, spec_obj)

    frames_per_batch = int(dcfg["frames_per_batch"])
    num_workers = int(dcfg["num_workers"])
    val_ratio = float(dcfg["val_ratio"])
    seed = int(dcfg["seed"])
    seq = dcfg["sequence"]
    clip_len = int(seq["frames_per_clip"])
    use_conf = bool(cfg["contact"]["targets"]["joint"]["use_confidence_weights"])

    image_specs = [s for s in specs if s["name"] in ("damon", "climbing")]
    video_specs = [s for s in specs if s["name"] == "climbing_videos"]

    datasets_for_validation = []
    train_parts: list[tuple] = []   # (dataset, batch_size, shuffle)
    val_parts: list[tuple] = []
    out_manifest: dict = {}

    if image_specs:
        subsets = [_build_image_dataset(s) for s in image_specs]
        datasets_for_validation += subsets
        ds = subsets[0] if len(subsets) == 1 else ConcatDataset(subsets)
        if manifest is not None:
            train_idx, val_idx = _manifest_image_indices(manifest, len(ds))
        else:
            train_idx, val_idx = train_val_indices(len(ds), val_ratio, seed)
        train_parts.append((Subset(ds, train_idx), frames_per_batch, True))
        val_parts.append((Subset(ds, val_idx), frames_per_batch, False))
        out_manifest["images"] = {"train": [int(i) for i in train_idx],
                                  "val": [int(i) for i in val_idx]}
        sizes = " + ".join(f"{s['name']}={len(d)}" for s, d in zip(image_specs, subsets))
        print(f"Image datasets [{sizes}] -> total={len(ds)} "
              f"train={len(train_idx)} val={len(val_idx)} (val_ratio={val_ratio}, seed={seed})")

    if video_specs:
        clips_per_batch = max(1, frames_per_batch // clip_len)
        for s in video_specs:
            root = _video_root(s["config"])
            scenes = list_scenes(root, "train")
            key = f"video:{s['config']}"
            if manifest is not None:
                train_vids, val_vids = _manifest_video_ids(manifest, key, scenes)
            else:
                train_vids, val_vids = group_train_val_split(
                    (video_id_from_scene(sc) for sc in scenes), val_ratio, seed)
            train_scenes = [sc for sc in scenes if video_id_from_scene(sc) in train_vids]
            val_scenes = [sc for sc in scenes if video_id_from_scene(sc) in val_vids]
            train_ds = ClimbingVideosDataset(
                root, scenes=train_scenes, mode="train",
                frames_per_clip=clip_len, frame_stride=int(seq["frame_stride"]),
                jitter=bool(seq["jitter"]), seed=seed,
                use_confidence_weights=use_conf)
            val_ds = ClimbingVideosDataset(
                root, scenes=val_scenes, mode="val",
                frames_per_clip=clip_len, frame_stride=int(seq["frame_stride"]),
                jitter=False, seed=seed,
                use_confidence_weights=use_conf)
            datasets_for_validation.append(train_ds)
            train_parts.append((train_ds, clips_per_batch, True))
            val_parts.append((val_ds, clips_per_batch, False))
            out_manifest[key] = {"train": sorted(train_vids), "val": sorted(val_vids)}
            print(f"ClimbingVideos [{s['config']}]: videos train={len(train_vids)} val={len(val_vids)} "
                  f"| scenes train={len(train_scenes)} val={len(val_scenes)} "
                  f"| clips train={len(train_ds)} val={len(val_ds)} (T={clip_len})")

    validate_targets(cfg, datasets_for_validation)

    def _loader(dataset, batch_size, shuffle):
        # persistent_workers=False so each epoch's workers fork fresh from the main
        # process *after* set_epoch — the stateless video jitter reads the updated
        # epoch (persistent workers would freeze it at epoch 0). Re-fork is cheap:
        # the dataset (window index / npz metadata) is built once and only copied.
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, drop_last=shuffle, collate_fn=collate,
            pin_memory=False, persistent_workers=False)

    train_loader = InterleavedLoader([_loader(*p) for p in train_parts], seed=seed)
    val_loader = InterleavedLoader([_loader(*p) for p in val_parts], seed=seed)
    return train_loader, val_loader, out_manifest
