#!/usr/bin/env python3
"""Precompute SAM-3D-Body pose predictions and DINOv3 image features for DAMON dataset.

For each sample:
- Run SAM-3D-Body body decoder (using pre-generated person masks for better bbox/conditioning)
- Save pose predictions (keypoints, MHR params, camera) per split as .npz
- Save DINOv3 encoder features per sample as .pt (float16)

Output structure:
  {output_dir}/features/{split}/{idx:04d}.pt       # [1280, 56, 56] float16
  {output_dir}/predictions/{split}/{idx:04d}.npz   # per-sample (for resume)
  {output_dir}/predictions/{split}_predictions.npz # merged per-split
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT = "/data3/rikhat.akizhanov/DECO"
DEFAULT_MASK_DIR = os.path.join(os.path.dirname(__file__), "damon_mhr_contact", "masks")
DEFAULT_NPZ_DIR = os.path.join(os.path.dirname(__file__), "damon_mhr_contact")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "damon_mhr_contact")
DEFAULT_CHECKPOINT_DIR = (
    "/data3/rikhat.akizhanov/human_global_motion/data/"
    "sam-3d-body-checkpoints/sam-3d-body-dinov3"
)
DEFAULT_TRAIN_CONFIG = os.path.join(os.path.dirname(__file__), "..", "train", "config.yaml")


# ---------------------------------------------------------------------------
# Mask / bbox helpers
# ---------------------------------------------------------------------------

def load_person_mask(mask_dir: str, split: str, idx: int):
    """Load pre-generated person mask (object order 000, single detection per image).

    Returns (mask_uint8 [H, W, 1], bbox [4]) or (None, None) on failure.
    """
    mask_path = Path(mask_dir) / split / f"{idx:04d}" / "masks" / f"{idx:04d}_000.png"
    if not mask_path.exists():
        return None, None

    gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None, None

    mask_bin = (gray > 127).astype(np.uint8) * 255
    coords = cv2.findNonZero(mask_bin)
    if coords is None:
        return None, None

    x, y, w, h = cv2.boundingRect(coords)
    if w == 0 or h == 0:
        return None, None

    bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
    return mask_bin[:, :, np.newaxis], bbox


# ---------------------------------------------------------------------------
# Per-sample prediction extraction
# ---------------------------------------------------------------------------

def extract_predictions(mhr_out, bbox_used, mask_available, imgname_str):
    """Extract scalar/array predictions from model mhr output dict."""
    def _np(x, idx=0):
        if isinstance(x, torch.Tensor):
            return x[idx].cpu().float().numpy()
        if isinstance(x, np.ndarray):
            return x[idx].astype(np.float32)
        return np.float32(x)

    return {
        "imgname": imgname_str,
        "pred_keypoints_3d": _np(mhr_out["pred_keypoints_3d"]),      # [70, 3]
        "pred_keypoints_2d": _np(mhr_out["pred_keypoints_2d"]),      # [70, 2]
        "pred_cam_t": _np(mhr_out["pred_cam_t"]),                    # [3]
        "focal_length": np.float32(_np(mhr_out["focal_length"])),    # scalar
        "pred_pose_raw": _np(mhr_out["pred_pose_raw"]),              # [266]
        "global_rot": _np(mhr_out["global_rot"]),                    # [3]
        "body_pose_params": _np(mhr_out["body_pose"]),               # [130]
        "hand_pose_params": _np(mhr_out["hand"]),                    # [108]
        "scale_params": _np(mhr_out["scale"]),                       # [28]
        "shape_params": _np(mhr_out["shape"]),                       # [45]
        "mhr_model_params": _np(mhr_out["mhr_model_params"]),        # [195]
        "pred_joint_coords": _np(mhr_out["pred_joint_coords"]),      # [127, 3]
        "bbox_used": bbox_used.astype(np.float32),                   # [4]
        "mask_available": np.bool_(mask_available),
    }


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_split(
    split: str,
    data_root: str,
    npz_dir: str,
    mask_dir: str,
    output_dir: str,
    model,
    transform,
    start_idx: int,
    end_idx: int,
    resume: bool,
):
    from sam_3d_body.data.utils.prepare_batch import prepare_batch
    from sam_3d_body.utils import recursive_to

    # Load dataset metadata (detect NPZ has imgname, bbox, cam_k — all we need)
    detect_npz = os.path.join(npz_dir, f"hot_dca_{split}_detect.npz")

    print(f"\n{'='*60}")
    print(f"Loading {detect_npz} ...")
    detect_data = np.load(detect_npz, allow_pickle=True)
    imgnames = detect_data["imgname"]
    n_samples = len(imgnames)
    fallback_bboxes = detect_data["bbox"]   # [N, 4]
    fallback_camks = detect_data["cam_k"]   # [N, 3, 3]

    end = n_samples if end_idx < 0 else min(end_idx, n_samples)
    indices = list(range(start_idx, end))
    print(f"Samples to process: {len(indices)}  (idx {start_idx}..{end - 1})")

    # Output dirs
    feat_dir = Path(output_dir) / "features" / split
    pred_dir = Path(output_dir) / "predictions" / split
    feat_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    stats = {"processed": 0, "skipped_resume": 0, "skipped_error": 0, "no_mask": 0}

    pbar = tqdm(total=len(indices), desc=split)

    for idx in indices:
        feat_path = feat_dir / f"{idx:04d}.pt"
        pred_path = pred_dir / f"{idx:04d}.npz"

        if resume and feat_path.exists() and pred_path.exists():
            stats["skipped_resume"] += 1
            pbar.update(1)
            continue

        imgname_str = str(imgnames[idx])

        # Load image (RGB)
        img_path = os.path.join(data_root, imgname_str)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            warnings.warn(f"[idx={idx}] Cannot load image: {img_path}")
            stats["skipped_error"] += 1
            pbar.update(1)
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Load mask → bbox
        mask_hwc, bbox_from_mask = load_person_mask(mask_dir, split, idx)
        if mask_hwc is not None:
            boxes = bbox_from_mask.reshape(1, 4)
            masks_arr = mask_hwc.reshape(1, h, w, 1).astype(np.uint8)
            masks_score = np.ones(1, dtype=np.float32)
            mask_available = True
        else:
            stats["no_mask"] += 1
            boxes = fallback_bboxes[idx].reshape(1, 4).astype(np.float32)
            masks_arr = None
            masks_score = None
            mask_available = False

        # Camera intrinsics from detect NPZ
        cam_k = torch.tensor(fallback_camks[idx], dtype=torch.float32).unsqueeze(0)  # [1, 3, 3]

        # Build batch
        try:
            batch = prepare_batch(img, transform, boxes, masks_arr, masks_score)
            batch = recursive_to(batch, "cuda")
            model._initialize_batch(batch)
            batch["cam_int"] = cam_k.to(batch["img"])
        except Exception as exc:
            warnings.warn(f"[idx={idx}] Batch prep failed: {exc}")
            stats["skipped_error"] += 1
            pbar.update(1)
            continue

        # Forward pass
        try:
            with torch.no_grad():
                pose_output = model.forward_step(batch, decoder_type="body")
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
            warnings.warn(f"[idx={idx}] Inference failed: {exc}")
            stats["skipped_error"] += 1
            pbar.update(1)
            continue

        # Save image features [1280, 56, 56] as float16
        try:
            feat = pose_output["image_embeddings"]  # [B*N, C, H, W]
            feat = feat[0].half().cpu()             # [C, H, W] = [1280, 56, 56]
            torch.save(feat, str(feat_path))
        except Exception as exc:
            warnings.warn(f"[idx={idx}] Feature save failed: {exc}")
            stats["skipped_error"] += 1
            pbar.update(1)
            continue

        # Save per-sample predictions
        try:
            mhr = pose_output["mhr"]
            preds = extract_predictions(mhr, boxes[0], mask_available, imgname_str)
            np.savez(str(pred_path), **preds)
        except Exception as exc:
            warnings.warn(f"[idx={idx}] Prediction save failed: {exc}")
            # Feature was saved; still count as partial success but flag error
            stats["skipped_error"] += 1
            pbar.update(1)
            continue

        stats["processed"] += 1
        pbar.update(1)

    pbar.close()
    print(f"Split '{split}' done: {stats}")

    # Merge per-sample predictions into a single NPZ
    _merge_predictions(pred_dir, output_dir, split, start_idx, end)

    return stats


def _merge_predictions(pred_dir: Path, output_dir: str, split: str, start_idx, end):
    """Merge all per-sample .npz files into one split-level .npz."""
    print(f"Merging predictions for split '{split}' ...")
    all_preds = {}
    missing = 0

    for idx in range(start_idx, end):
        pred_path = pred_dir / f"{idx:04d}.npz"
        if not pred_path.exists():
            missing += 1
            continue

        data = np.load(str(pred_path), allow_pickle=True)
        for key in data.files:
            val = data[key]
            if key not in all_preds:
                all_preds[key] = []
            all_preds[key].append(val)

    if not all_preds:
        print(f"  No predictions found to merge for split '{split}'")
        return

    merged = {}
    for key, vals in all_preds.items():
        if key == "imgname":
            merged[key] = np.array(vals, dtype=object)
        elif key == "mask_available":
            merged[key] = np.array(vals, dtype=bool)
        else:
            merged[key] = np.stack(vals, axis=0)

    out_path = Path(output_dir) / "predictions" / f"{split}_predictions.npz"
    np.savez(str(out_path), **merged)
    n = len(next(iter(merged.values())))
    print(f"  Saved {n} samples → {out_path}  (missing: {missing})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Precompute SAM-3D-Body predictions and DINOv3 features for DAMON"
    )
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--mask_dir", default=DEFAULT_MASK_DIR)
    parser.add_argument("--npz_dir", default=DEFAULT_NPZ_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--train_config", default=DEFAULT_TRAIN_CONFIG,
                        help="Training config.yaml (used for IMAGE_SIZE override)")
    parser.add_argument("--splits", nargs="+", default=["trainval", "test"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1, help="-1 means process all")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed samples")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Add project root to path so sam_3d_body is importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import yaml
    from sam_3d_body.build_models import load_sam_3d_body_local
    from sam_3d_body.data.transforms import (
        Compose, GetBBoxCenterScale, TopdownAffine, VisionTransformWrapper,
    )
    from torchvision.transforms import ToTensor

    # Read IMAGE_SIZE from training config (overrides checkpoint's baked config)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)
    image_size = tuple(train_cfg["MODEL"]["IMAGE_SIZE"])
    print(f"Using IMAGE_SIZE={image_size} from {args.train_config}")

    print(f"Loading SAM-3D-Body from {args.checkpoint_dir} ...")
    model, model_cfg = load_sam_3d_body_local(args.checkpoint_dir)
    model.eval()

    # Override model config IMAGE_SIZE so backbone produces correct spatial dims
    model_cfg.defrost()
    model_cfg.MODEL.IMAGE_SIZE = list(image_size)
    model_cfg.freeze()

    transform = Compose([
        GetBBoxCenterScale(),
        TopdownAffine(input_size=image_size, use_udp=False),
        VisionTransformWrapper(ToTensor()),
    ])

    for split in args.splits:
        process_split(
            split=split,
            data_root=args.data_root,
            npz_dir=args.npz_dir,
            mask_dir=args.mask_dir,
            output_dir=args.output_dir,
            model=model,
            transform=transform,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            resume=args.resume,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
