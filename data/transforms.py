"""SAM-3D-Body's top-down crop applied to one frame dict.

:func:`process_frame` runs the model's own transform pipeline (bbox -> centre +
padded scale -> affine warp to the model resolution) on a frame's image, mask
and box, producing exactly the geometry keys the wrapper consumes:
``img``, ``img_size``, ``ori_img_size``, ``bbox_center``, ``bbox_scale``,
``bbox``, ``affine_trans``, ``mask``, ``mask_score``.

Frames whose ``image`` is None but which carry a header-read ``img_wh`` (the
embedding-cache path) take an imageless fast path: identical geometry and an
identical mask warp, but no JPEG decode and a zero crop for ``img`` — the model
provably never reads the crop's values when ``embedding`` is present.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml
from torchvision.transforms import ToTensor

from model.sam_3d_body.data.transforms import (
    Compose, GetBBoxCenterScale, TopdownAffine, VisionTransformWrapper,
)


def crop_size(checkpoint_path: str | Path) -> Tuple[int, int]:
    """The model's crop resolution, read from the checkpoint's ``model_config.yaml``.

    The config lives next to the checkpoint or one directory up, as in the
    HuggingFace snapshot layout.
    """
    directory = Path(checkpoint_path).parent
    for candidate in (directory / "model_config.yaml",
                      directory.parent / "model_config.yaml"):
        if candidate.is_file():
            size = yaml.safe_load(candidate.read_text())["MODEL"]["IMAGE_SIZE"]
            return int(size[0]), int(size[1])
    raise FileNotFoundError(f"no model_config.yaml next to {checkpoint_path}")


def build_transform(image_size: Tuple[int, int], *, imageless: bool = False):
    """SAM-3D-Body's standard top-down crop pipeline at the model resolution.

    :param imageless: drop the torchvision wrapper (it requires ``img``); the
        geometric transforms run identically without one — ``TopdownAffine``
        skips a missing ``img`` and the warp matrix depends only on the bbox.
    """
    transforms = [
        GetBBoxCenterScale(),
        TopdownAffine(input_size=image_size, use_udp=False),
    ]
    if not imageless:
        transforms.append(VisionTransformWrapper(ToTensor()))
    return Compose(transforms)


def process_frame(frame: dict, transform, transform_imageless=None) -> dict:
    """Crop one frame dict. ``-> {img, img_size, ori_img_size, bbox_*, mask*}``
    (no ``img`` on the embedding-cache path: the model never reads pixels there).

    :param transform: the pipeline from :func:`build_transform`.
    :param transform_imageless: its imageless variant, required for frames on
        the embedding-cache path.
    """
    img: Optional[np.ndarray] = frame["image"]
    mask: Optional[np.ndarray] = frame["mask"]
    bbox = frame["bbox"]
    if bbox is None:
        raise RuntimeError(f"frame {frame['key']} has no bbox")
    bbox = np.asarray(bbox, dtype=np.float32)
    if (bbox.shape != (4,) or not np.isfinite(bbox).all()
            or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
        raise RuntimeError(
            f"frame {frame['key']} has invalid xyxy bbox {bbox.tolist()}")
    imageless = img is None and frame["img_wh"] is not None
    has_mask = mask is not None
    if mask is None:
        w, h = frame["img_wh"] if imageless else (img.shape[1], img.shape[0])
        mask = np.zeros((h, w, 1), dtype=np.uint8)
    elif mask.ndim == 2:
        mask = mask[..., None]

    # mask_score > 0 tells the model this is a real mask; a substituted
    # all-zeros mask must score 0.0 so it is treated as "no mask given".
    data_info = dict(
        bbox=bbox,
        bbox_format="xyxy",
        mask=mask,
        mask_score=np.array(1.0 if has_mask else 0.0, dtype=np.float32),
    )
    if imageless:
        if transform_imageless is None:
            raise RuntimeError(
                f"frame {frame['key']} has no decoded image "
                "(embedding-cache path) but no imageless transform was given")
        out = transform_imageless(data_info)
        out["ori_img_size"] = np.array(frame["img_wh"])                 # [W, H]
    else:
        data_info["img"] = img
        out = transform(data_info)
    warped = out["mask"]
    if warped.ndim == 3:
        warped = warped[..., 0]
    out["mask"] = warped.astype(np.float32) / 255.0
    return out
