"""
Dataset utilities for preparing batches for SAM-3D-Body training.

Uses the same transform pipeline as the reference inference code
(GetBBoxCenterScale -> TopdownAffine -> ToTensor) to ensure batch format
is identical to what the model expects.

Two entry points:
  - prepare_training_batch:  original ETH-style (shared camera, N persons in 1 image).
  - prepare_damon_batch:     DAMON-style (B independent samples, each with its own
                              camera K; produces [B, 1, ...] shaped tensors).
"""

import numpy as np
import torch
from torch.utils.data import default_collate
from torchvision.transforms import ToTensor

from sam_3d_body.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)
from sam_3d_body.data.utils.prepare_batch import NoCollate


# Build the same transform pipeline used by SAM3DBodyEstimator
# Body padding=1.25 (default), no UDP, input_size matches model config
_body_transform = Compose(
    [
        GetBBoxCenterScale(),  # padding=1.25 by default
        TopdownAffine(input_size=(896, 896), use_udp=False),
        VisionTransformWrapper(ToTensor()),
    ]
)


def _make_transform(target_size=(896, 896)):
    if target_size == (896, 896):
        return _body_transform
    return Compose(
        [
            GetBBoxCenterScale(),
            TopdownAffine(input_size=target_size, use_udp=False),
            VisionTransformWrapper(ToTensor()),
        ]
    )


def _process_one(img, bbox, transform):
    """Run SAM-3D-Body transforms for a single (image, bbox) pair."""
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    if isinstance(bbox, torch.Tensor):
        bbox = bbox.numpy()

    height, width = img.shape[:2]
    data_info = dict(img=img)
    data_info["bbox"] = bbox.astype(np.float32)
    data_info["bbox_format"] = "xyxy"
    data_info["mask"] = np.zeros((height, width, 1), dtype=np.uint8)
    data_info["mask_score"] = np.array(0.0, dtype=np.float32)
    return transform(data_info)


# ---------------------------------------------------------------------------
# Original helper (ETH-style: N persons from 1 image, shared camera)
# ---------------------------------------------------------------------------

def prepare_training_batch(images, bboxes, cam_params, target_size=(896, 896), device='cuda'):
    """
    Prepare a batch for SAM-3D-Body model (ETH-style).

    Creates a batch with shape [1, N, ...] where N is the number of persons
    (all from the same image / same camera).

    Args:
        images: list of N numpy arrays (H, W, 3) RGB uint8
        bboxes: list of N numpy arrays (4,) [x1, y1, x2, y2]
        cam_params: dict with keys fx, fy, cx, cy
        target_size: (H, W) must match model config
        device: torch device string

    Returns:
        batch dict with all keys required by SAM-3D-Body
    """
    transform = _make_transform(target_size)
    orig_img = images[0] if isinstance(images[0], np.ndarray) else images[0].numpy()

    data_list = [_process_one(img, bbox, transform) for img, bbox in zip(images, bboxes)]
    batch = default_collate(data_list)
    max_num_person = batch["img"].shape[0]

    for key in ["img", "img_size", "ori_img_size", "bbox_center", "bbox_scale",
                "bbox", "affine_trans", "mask", "mask_score"]:
        if key in batch:
            batch[key] = batch[key].unsqueeze(0).float()
    if "mask" in batch:
        batch["mask"] = batch["mask"].unsqueeze(2)
    batch["person_valid"] = torch.ones((1, max_num_person))

    cam_int = torch.tensor(
        [[[cam_params['fx'], 0, cam_params['cx']],
          [0, cam_params['fy'], cam_params['cy']],
          [0, 0, 1]]],
        dtype=torch.float32,
    )
    batch["cam_int"] = cam_int
    batch["img_ori"] = [NoCollate(orig_img)]

    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)

    return batch


# ---------------------------------------------------------------------------
# DAMON-style helper: B independent samples, each with its own camera K
# ---------------------------------------------------------------------------

def prepare_damon_batch(images, bboxes, cam_ks, target_size=(896, 896), device='cuda'):
    """
    Prepare a batch for SAM-3D-Body from DAMON dataset samples.

    Each sample is one person with its own camera intrinsics K.
    Produces tensors with shape [B, 1, ...] and cam_int [B, 3, 3].

    Args:
        images: list/tensor of B numpy arrays (H_i, W_i, 3) RGB uint8
        bboxes: list/tensor of B arrays/tensors (4,)  [x1, y1, x2, y2]
        cam_ks: list/tensor of B arrays/tensors (3, 3)  camera intrinsics
        target_size: (H, W) — must match model IMAGE_SIZE config
        device: torch device string

    Returns:
        batch dict with all keys required by SAM-3D-Body, plus cam_int [B, 3, 3]
    """
    transform = _make_transform(target_size)
    B = len(images)

    per_sample = []
    for img, bbox in zip(images, bboxes):
        per_sample.append(_process_one(img, bbox, transform))

    # Stack each key into [B, ...] then unsqueeze person dim -> [B, 1, ...]
    keys_to_stack = ["img", "img_size", "ori_img_size", "bbox_center", "bbox_scale",
                     "bbox", "affine_trans", "mask", "mask_score"]

    batch = {}
    for key in keys_to_stack:
        if key in per_sample[0]:
            tensors = [
                s[key] if isinstance(s[key], torch.Tensor) else torch.as_tensor(s[key])
                for s in per_sample
            ]
            stacked = torch.stack(tensors, dim=0).float()  # [B, ...]
            batch[key] = stacked.unsqueeze(1)  # [B, 1, ...]

    if "mask" in batch:
        batch["mask"] = batch["mask"].unsqueeze(2)  # [B, 1, 1, H, W]  (binary mask channel)

    batch["person_valid"] = torch.ones((B, 1))

    # Per-sample camera intrinsics: stack into [B, 3, 3]
    cam_int_list = []
    for ck in cam_ks:
        if isinstance(ck, torch.Tensor):
            cam_int_list.append(ck.float())
        else:
            cam_int_list.append(torch.tensor(ck, dtype=torch.float32))
    batch["cam_int"] = torch.stack(cam_int_list, dim=0)  # [B, 3, 3]

    # img_ori: store original first image for potential hand processing
    orig_img = images[0] if isinstance(images[0], np.ndarray) else np.array(images[0])
    batch["img_ori"] = [NoCollate(orig_img)]

    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)

    return batch


# ---------------------------------------------------------------------------
# Precomputed-features helper: skips image loading, uses cached backbone output
# ---------------------------------------------------------------------------

def prepare_damon_batch_precomputed(
    features, bboxes, cam_ks, ori_img_sizes, target_size=(896, 896), device='cuda',
):
    """
    Prepare a batch using precomputed backbone features instead of raw images.

    We still need bbox_center, bbox_scale, affine_trans, and a dummy img tensor
    (for ray conditioning shape) — so we run the transform pipeline on small
    dummy images with the correct (H, W).

    Args:
        features:      tensor [B, C, Hf, Wf]  precomputed backbone embeddings
        bboxes:        tensor [B, 4]
        cam_ks:        tensor [B, 3, 3]
        ori_img_sizes: tensor [B, 2]  (H, W) of original images
        target_size:   (H, W) model input size
        device:        torch device string

    Returns:
        (batch_dict, features_tensor)
            batch_dict: has img (dummy), bbox_center, bbox_scale, affine_trans,
                        ori_img_size, cam_int, person_valid — all on device
            features_tensor: [B, C, Hf, Wf] float32 on device
    """
    transform = _make_transform(target_size)
    B = features.shape[0]

    per_sample = []
    for i in range(B):
        h, w = int(ori_img_sizes[i, 0].item()), int(ori_img_sizes[i, 1].item())
        # Minimal dummy image — only bbox geometry matters
        dummy_img = np.zeros((h, w, 3), dtype=np.uint8)
        bbox_np = bboxes[i].numpy() if isinstance(bboxes[i], torch.Tensor) else bboxes[i]
        per_sample.append(_process_one(dummy_img, bbox_np, transform))

    keys_to_stack = ["img", "img_size", "ori_img_size", "bbox_center", "bbox_scale",
                     "bbox", "affine_trans", "mask", "mask_score"]
    batch = {}
    for key in keys_to_stack:
        if key in per_sample[0]:
            tensors = [
                s[key] if isinstance(s[key], torch.Tensor) else torch.as_tensor(s[key])
                for s in per_sample
            ]
            stacked = torch.stack(tensors, dim=0).float()
            batch[key] = stacked.unsqueeze(1)  # [B, 1, ...]

    if "mask" in batch:
        batch["mask"] = batch["mask"].unsqueeze(2)

    batch["person_valid"] = torch.ones((B, 1))

    cam_int_list = []
    for i in range(B):
        ck = cam_ks[i]
        if isinstance(ck, torch.Tensor):
            cam_int_list.append(ck.float())
        else:
            cam_int_list.append(torch.tensor(ck, dtype=torch.float32))
    batch["cam_int"] = torch.stack(cam_int_list, dim=0)

    batch["img_ori"] = [NoCollate(np.zeros((2, 2, 3), dtype=np.uint8))]

    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)

    features_out = features.float().to(device)
    return batch, features_out
