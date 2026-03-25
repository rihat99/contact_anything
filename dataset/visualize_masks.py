#!/usr/bin/env python3
"""Visualize v2 SAM3 masks overlaid on original DAMON images.

Color coding (translucent overlay):
  Blue  — person mask
  Green — contact object, best-disambiguation detection
           (has contact vertices + best_detection >= 0)
  Red   — non-contact object detections OR non-best contact detections

Saves one image per sample to:
  {masks_v2_dir}/../test_masks/{split}/{idx:04d}.jpg
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT = "/data3/rikhat.akizhanov/DECO"
DEFAULT_MASKS_V2_DIR = os.path.join(
    os.path.dirname(__file__), "damon_mhr_contact", "masks_v2"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "damon_mhr_contact", "test_masks"
)

# RGBA overlay colours (BGR for cv2)
COLOR_PERSON  = (200,  80,  20)   # blue-ish
COLOR_CONTACT = ( 30, 200,  30)   # green
COLOR_OTHER   = ( 30,  30, 220)   # red
ALPHA = 0.45                       # mask opacity


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def overlay_mask(img: np.ndarray, mask: np.ndarray, color: tuple, alpha: float):
    """Apply a translucent coloured overlay on img where mask is True (in-place)."""
    if mask is None or not mask.any():
        return
    colored = np.zeros_like(img)
    colored[:] = color
    mask3 = mask[:, :, np.newaxis]
    img[:] = np.where(mask3, (img * (1 - alpha) + colored * alpha).astype(np.uint8), img)


def load_mask(path: Path) -> np.ndarray:
    """Load a binary mask PNG as bool [H, W]."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    return gray > 127


# ---------------------------------------------------------------------------
# Per-sample visualisation
# ---------------------------------------------------------------------------

def visualize_sample(
    idx: int,
    split: str,
    data_root: str,
    masks_v2_dir: str,
    out_dir: Path,
):
    sample_dir = Path(masks_v2_dir) / split / f"{idx:04d}"
    meta_path = sample_dir / "metadata.npz"
    if not meta_path.exists():
        return False

    meta = np.load(str(meta_path), allow_pickle=True)
    imgname = str(meta["imgname"])
    object_names = list(meta["object_names"])
    num_detections = meta["num_detections"]
    n_objs = len(object_names)

    # Load original image
    img_path = os.path.join(data_root, imgname)
    img = cv2.imread(img_path)
    if img is None:
        return False

    masks_dir = sample_dir / "masks"

    # Draw order: non-best contact detections (red) → non-contact (red) → contact-best (green) → person (blue)
    # Collect layers in draw order so important masks appear on top
    layers_red   = []   # (mask_array,)
    layers_green = []
    layers_blue  = []

    for obj_order in range(n_objs):
        safe_name = object_names[obj_order].lower().replace(" ", "_")
        n_det = int(num_detections[obj_order])
        best_det = int(meta[f"best_detection_{obj_order}"])
        cv_smpl = meta[f"contact_vertices_smpl_{obj_order}"]
        has_contact_verts = len(cv_smpl) > 0

        if obj_order == 0:
            # Person — always blue, single mask
            mask_path = masks_dir / f"{idx:04d}_{safe_name}_0.png"
            m = load_mask(mask_path)
            if m is not None:
                layers_blue.append(m)
            continue

        for det_idx in range(n_det):
            mask_path = masks_dir / f"{idx:04d}_{safe_name}_{det_idx}.png"
            m = load_mask(mask_path)
            if m is None:
                continue

            is_best = (det_idx == best_det)
            is_contact = has_contact_verts and best_det >= 0 and is_best

            if is_contact:
                layers_green.append(m)
            else:
                layers_red.append(m)

    # Composite: red first (bottom), then green, then blue (top)
    for m in layers_red:
        overlay_mask(img, m, COLOR_OTHER, ALPHA)
    for m in layers_green:
        overlay_mask(img, m, COLOR_CONTACT, ALPHA)
    for m in layers_blue:
        overlay_mask(img, m, COLOR_PERSON, ALPHA)

    # Add a small legend and object labels
    _draw_legend(img, object_names, num_detections, meta, n_objs)

    out_path = out_dir / f"{idx:04d}.jpg"
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return True


def _draw_legend(img, object_names, num_detections, meta, n_objs):
    """Draw per-object label strip at the bottom of the image."""
    h, w = img.shape[:2]
    line_h = 22
    pad = 6
    strip_h = line_h * n_objs + pad * 2
    # Dark semi-transparent strip
    strip = img[max(0, h - strip_h):h, :].copy()
    cv2.rectangle(img, (0, max(0, h - strip_h)), (w, h), (20, 20, 20), -1)
    img[max(0, h - strip_h):h, :] = cv2.addWeighted(
        img[max(0, h - strip_h):h, :], 0.3, strip, 0.7, 0
    )

    for obj_order in range(n_objs):
        name = object_names[obj_order]
        n_det = int(num_detections[obj_order])
        best_det = int(meta[f"best_detection_{obj_order}"])
        cv_smpl = meta[f"contact_vertices_smpl_{obj_order}"]
        has_contact_verts = len(cv_smpl) > 0

        if obj_order == 0:
            color = COLOR_PERSON
            label = f"[person]  det=1"
        elif has_contact_verts and best_det >= 0:
            color = COLOR_CONTACT
            dis_scores = meta[f"disambiguation_scores_{obj_order}"]
            best_score = float(dis_scores[best_det]) if len(dis_scores) > best_det else 0.0
            label = f"[CONTACT] {name}  det={n_det}  best={best_det}  score={best_score:.2f}  cv={len(cv_smpl)}"
        else:
            color = COLOR_OTHER
            reason = "no_cv" if not has_contact_verts else "no_det"
            label = f"[other]   {name}  det={n_det}  ({reason})"

        y = max(0, h - strip_h) + pad + obj_order * line_h + line_h - 4
        cv2.putText(img, label, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize v2 SAM3 masks overlaid on DAMON images"
    )
    parser.add_argument("--split", default="trainval", choices=["trainval", "test"])
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--masks_v2_dir", default=DEFAULT_MASKS_V2_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1, help="-1 = all")
    args = parser.parse_args()

    split_mask_dir = Path(args.masks_v2_dir) / args.split
    if not split_mask_dir.exists():
        print(f"ERROR: mask dir not found: {split_mask_dir}")
        return

    # Collect available sample indices
    available = sorted(
        int(p.name) for p in split_mask_dir.iterdir()
        if p.is_dir() and (p / "metadata.npz").exists()
    )

    start = args.start_idx
    end = available[-1] + 1 if args.end_idx < 0 else args.end_idx
    indices = [i for i in available if start <= i < end]
    print(f"Visualising {len(indices)} samples from split '{args.split}'")

    out_dir = Path(args.output_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for idx in tqdm(indices, desc=args.split):
        if visualize_sample(idx, args.split, args.data_root, args.masks_v2_dir, out_dir):
            ok += 1

    print(f"Done. Saved {ok}/{len(indices)} images → {out_dir}")


if __name__ == "__main__":
    main()
