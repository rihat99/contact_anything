"""Clips in, one flat model batch out.

The collate is generic: it crops every frame with
:func:`data.transforms.process_frame`, then stacks whatever else the frames
carry. There are no per-signal fallbacks — a key is emitted iff EVERY frame in
the batch carries it, and a partially present key is a bug, not a zero-fill.

Clips must share their length ``T``; the batch is the flattened
``[B_clips * T, ...]`` sequence in frame-major order (clip 0's frames first),
which is the layout the temporal bricks reshape with ``seq_len``. There is no
person dimension.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch

from .transforms import build_transform, process_frame

#: Frame keys consumed by the crop; everything else is stacked as it is.
_CROP_INPUTS = ("image", "img_wh", "mask", "bbox")
#: Geometry the crop produces, in the names the wrapper consumes (``img`` only
#: on the live-image path).
_CROP_OUTPUTS = ("img", "img_size", "ori_img_size", "bbox_center", "bbox_scale",
                 "bbox", "affine_trans", "mask", "mask_score")


def _stack(values: list) -> torch.Tensor:
    """Stack per-frame values into ``[B, ...]``, floats normalised to float32."""
    out = torch.stack([
        v if isinstance(v, torch.Tensor) else torch.as_tensor(v) for v in values])
    if out.is_floating_point() and out.dtype != torch.bfloat16:
        out = out.float()
    return out


def make_collate(image_size: Tuple[int, int]):
    """Build the collate for a given model crop resolution."""
    transform = build_transform(image_size)
    transform_imageless = build_transform(image_size, imageless=True)

    def collate(clips: Sequence[list[dict]]) -> dict:
        seq_len = len(clips[0])
        if any(len(clip) != seq_len for clip in clips):
            raise ValueError(
                "homogeneous-T batches only: all clips must share the same length")
        frames = [f for clip in clips for f in clip]

        cropped = [process_frame(f, transform, transform_imageless) for f in frames]
        # The whole geometry block is float32: the model mixes these tensors in
        # one arithmetic expression (the CLIFF condition), and the integer sizes
        # would otherwise ride in on a type-promotion accident.
        batch = {key: _stack([c[key] for c in cropped]).float()
                 for key in _CROP_OUTPUTS if key in cropped[0]}

        keys = dict.fromkeys(k for f in frames for k in f if k not in _CROP_INPUTS)
        for key in keys:
            missing = sum(key not in f for f in frames)
            if missing:
                raise ValueError(
                    f"batch key {key!r} is present on some frames and missing on "
                    f"{missing} others — datasets must agree on their signals")
            if isinstance(frames[0][key], str):
                batch[key] = [f[key] for f in frames]
            else:
                batch[key] = _stack([f[key] for f in frames])
        batch["seq_len"] = seq_len
        return batch

    return collate


def batch_to_device(batch: dict, device) -> dict:
    """Move every tensor to ``device`` in place; ints and string lists pass through."""
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=True)
    return batch
