#!/usr/bin/env python3
"""Generate SAM3 masks and bounding boxes for the DAMON dataset.

For each sample:
- Segment "person" (always, object order 0)
- Segment each contact object from contact_label_objectwise (skip "supporting")

Output structure:
  {output_dir}/{split}/{idx:04d}/
    metadata.npz   (imgname, object_names, num_detections, bboxes_*, scores_*)
    masks/
      {idx:04d}_{obj_order:03d}.png        # single detection
      {idx:04d}_{obj_order:03d}_{det}.png  # multiple detections (det=0,1,2,...)
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAM3_REPO = "/data3/rikhat.akizhanov/human_global_motion/sam3"
BPE_PATH = f"{SAM3_REPO}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
CHECKPOINT_PATH = "/data3/rikhat.akizhanov/human_global_motion/data/sam3-checkpoints/sam3.pt"

DEFAULT_DATA_ROOT = "/data3/rikhat.akizhanov/DECO/"
DEFAULT_NPZ_DIR = "/data3/rikhat.akizhanov/DECO/datasets/Release_Datasets/damon/"
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "damon_mhr_contact", "masks"
)

# ---------------------------------------------------------------------------
# SAM3 imports (after CUDA_VISIBLE_DEVICES is set in __main__)
# ---------------------------------------------------------------------------
def _import_sam3():
    sys.path.insert(0, SAM3_REPO)

    from sam3 import build_sam3_image_model
    from sam3.train.data.sam3_image_dataset import (
        Datapoint,
        FindQueryLoaded,
        Image as SAMImage,
        InferenceMetadata,
    )
    from sam3.train.data.collator import collate_fn_api as collate
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.transforms.basic_for_api import (
        ComposeAPI,
        RandomResizeAPI,
        ToTensorAPI,
        NormalizeAPI,
    )
    from sam3.eval.postprocessors import PostProcessImage

    return (
        build_sam3_image_model,
        Datapoint,
        FindQueryLoaded,
        SAMImage,
        InferenceMetadata,
        collate,
        copy_data_to_device,
        ComposeAPI,
        RandomResizeAPI,
        ToTensorAPI,
        NormalizeAPI,
        PostProcessImage,
    )


# ---------------------------------------------------------------------------
# Datapoint helpers (mirrors sam3_image_batched_inference.ipynb)
# ---------------------------------------------------------------------------
_GLOBAL_COUNTER = 1


def _reset_counter():
    global _GLOBAL_COUNTER
    _GLOBAL_COUNTER = 1


def _create_empty_datapoint(Datapoint):
    return Datapoint(find_queries=[], images=[])


def _set_image(datapoint, pil_image, SAMImage):
    w, h = pil_image.size  # PIL returns (width, height)
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]


def _add_text_prompt(datapoint, text_query, FindQueryLoaded, InferenceMetadata):
    global _GLOBAL_COUNTER
    # images[0].size is stored as (height, width); notebook uses confusing
    # variable names w/h here — just follow the pattern verbatim
    img_h, img_w = datapoint.images[0].size
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=_GLOBAL_COUNTER,
                original_image_id=_GLOBAL_COUNTER,
                original_category_id=1,
                original_size=[img_h, img_w],
                object_id=0,
                frame_index=0,
            ),
        )
    )
    _GLOBAL_COUNTER += 1
    return _GLOBAL_COUNTER - 1


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_mask_png(mask_tensor, path: Path):
    """Save a [1, H, W] or [H, W] bool/float tensor as binary PNG."""
    arr = mask_tensor.squeeze().cpu().float().numpy()
    arr = (arr > 0).astype(np.uint8) * 255
    Image.fromarray(arr).save(str(path))


def _save_sample(
    sample_idx: int,
    sample_dir: Path,
    imgname: str,
    object_names: list,
    results_per_object: list,
):
    """Save masks and metadata.npz for one sample."""
    mask_dir = sample_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    num_objects = len(object_names)
    num_detections = np.zeros(num_objects, dtype=np.int32)
    save_dict = {
        "imgname": np.array(imgname),
        "object_names": np.array(object_names),
    }

    for obj_order, result in enumerate(results_per_object):
        if result is None or len(result["scores"]) == 0:
            num_detections[obj_order] = 0
            save_dict[f"bboxes_{obj_order}"] = np.zeros((0, 4), dtype=np.float32)
            save_dict[f"scores_{obj_order}"] = np.zeros(0, dtype=np.float32)
            continue

        scores = result["scores"]
        boxes = result["boxes"]
        masks = result["masks"]

        # For person (object order 0), keep only the detection with largest bbox area
        if obj_order == 0 and len(scores) > 1:
            b = boxes.float()
            areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            best = areas.argmax().item()
            scores = scores[best : best + 1]
            boxes = boxes[best : best + 1]
            masks = [masks[best]]

        n_det = len(scores)
        num_detections[obj_order] = n_det

        save_dict[f"bboxes_{obj_order}"] = boxes.cpu().float().numpy().astype(np.float32)
        save_dict[f"scores_{obj_order}"] = scores.cpu().float().numpy().astype(np.float32)

        for det_idx, mask in enumerate(masks):
            if n_det == 1:
                fname = f"{sample_idx:04d}_{obj_order:03d}.png"
            else:
                fname = f"{sample_idx:04d}_{obj_order:03d}_{det_idx}.png"
            _save_mask_png(mask, mask_dir / fname)

    save_dict["num_detections"] = num_detections
    np.savez(str(sample_dir / "metadata.npz"), **save_dict)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_split(
    split: str,
    data_root: str,
    npz_dir: str,
    output_dir: str,
    model,
    transform,
    postprocessor,
    batch_size: int,
    start_idx: int,
    end_idx: int,
    resume: bool,
    sam3_classes: tuple,
):
    (
        _,
        Datapoint,
        FindQueryLoaded,
        SAMImage,
        InferenceMetadata,
        collate,
        copy_data_to_device,
        *_rest,
    ) = sam3_classes

    npz_path = os.path.join(npz_dir, f"hot_dca_{split}.npz")
    print(f"\n{'='*60}")
    print(f"Loading {npz_path} ...")
    data = np.load(npz_path, allow_pickle=True)
    imgnames = data["imgname"]
    contact_labels = data["contact_label_objectwise"]
    n_samples = len(imgnames)

    end = n_samples if end_idx < 0 else min(end_idx, n_samples)
    indices = list(range(start_idx, end))
    print(f"Samples to process: {len(indices)}  (idx {start_idx}..{end - 1})")

    split_out = Path(output_dir) / split
    split_out.mkdir(parents=True, exist_ok=True)

    stats = {
        "processed": 0,
        "skipped_resume": 0,
        "skipped_error": 0,
        "no_person_det": 0,
    }

    current_batch_size = batch_size
    i = 0
    pbar = tqdm(total=len(indices), desc=split)

    while i < len(indices):
        raw_batch = indices[i : i + current_batch_size]

        # --- resume filter ---
        if resume:
            to_process = []
            for idx in raw_batch:
                if (split_out / f"{idx:04d}" / "metadata.npz").exists():
                    stats["skipped_resume"] += 1
                    pbar.update(1)
                else:
                    to_process.append(idx)
            if not to_process:
                i += len(raw_batch)
                continue
        else:
            to_process = raw_batch

        # --- build datapoints ---
        _reset_counter()
        datapoints = []
        query_map = {}  # query_id -> (sample_idx, obj_order)
        sample_meta = {}  # sample_idx -> (imgname_str, object_names, query_ids)
        valid_indices = []

        for idx in to_process:
            img_path = os.path.join(data_root, imgnames[idx])
            try:
                pil_img = Image.open(img_path).convert("RGB")
            except Exception as exc:
                warnings.warn(f"[idx={idx}] Cannot load image {img_path}: {exc}")
                stats["skipped_error"] += 1
                pbar.update(1)
                continue

            dp = _create_empty_datapoint(Datapoint)
            _set_image(dp, pil_img, SAMImage)

            object_names = ["person"]
            query_ids = []

            # Person is always object order 0
            qid = _add_text_prompt(dp, "person", FindQueryLoaded, InferenceMetadata)
            query_map[qid] = (idx, 0)
            query_ids.append(qid)

            # Contact objects
            obj_order = 1
            for obj_name in contact_labels[idx].keys():
                if obj_name == "supporting":
                    continue
                text_prompt = obj_name.replace("_", " ")
                qid = _add_text_prompt(
                    dp, text_prompt, FindQueryLoaded, InferenceMetadata
                )
                query_map[qid] = (idx, obj_order)
                query_ids.append(qid)
                object_names.append(obj_name)
                obj_order += 1

            try:
                dp = transform(dp)
            except Exception as exc:
                warnings.warn(f"[idx={idx}] Transform failed: {exc}")
                stats["skipped_error"] += 1
                pbar.update(1)
                continue

            datapoints.append(dp)
            sample_meta[idx] = (str(imgnames[idx]), object_names, query_ids)
            valid_indices.append(idx)

        if not datapoints:
            i += len(raw_batch)
            continue

        # --- inference ---
        try:
            batch = collate(datapoints, dict_key="dummy")["dummy"]
            batch = copy_data_to_device(
                batch, torch.device("cuda"), non_blocking=True
            )
            with torch.no_grad():
                output = model(batch)
            processed_results = postprocessor.process_results(
                output, batch.find_metadatas
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and current_batch_size > 1:
                current_batch_size = max(1, current_batch_size // 2)
                print(f"\n[OOM] Reducing batch_size to {current_batch_size}")
                torch.cuda.empty_cache()
                continue  # retry same i without advancing
            else:
                warnings.warn(
                    f"Inference failed for batch at i={i}: {exc}"
                )
                for idx in valid_indices:
                    stats["skipped_error"] += 1
                    pbar.update(1)
                i += len(raw_batch)
                continue

        # --- save results ---
        for idx in valid_indices:
            imgname_str, object_names, query_ids = sample_meta[idx]
            results_per_object = []
            for obj_order_local, qid in enumerate(query_ids):
                result = processed_results.get(qid)
                results_per_object.append(result)
                if obj_order_local == 0 and (
                    result is None or len(result["scores"]) == 0
                ):
                    stats["no_person_det"] += 1

            sample_dir = split_out / f"{idx:04d}"
            try:
                _save_sample(
                    idx, sample_dir, imgname_str, object_names, results_per_object
                )
                stats["processed"] += 1
            except Exception as exc:
                warnings.warn(f"[idx={idx}] Save failed: {exc}")
                stats["skipped_error"] += 1
            pbar.update(1)

        i += len(raw_batch)

    pbar.close()
    print(f"Split '{split}' done: {stats}")
    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate SAM3 masks/bboxes for DAMON dataset"
    )
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--npz_dir", default=DEFAULT_NPZ_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--detection_threshold", type=float, default=0.5)
    parser.add_argument(
        "--splits", nargs="+", default=["trainval", "test"]
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples that already have metadata.npz",
    )
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument(
        "--end_idx", type=int, default=-1, help="-1 means process all"
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Torch settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sam3_classes = _import_sam3()
    (
        build_sam3_image_model,
        _Datapoint,
        _FindQueryLoaded,
        _SAMImage,
        _InferenceMetadata,
        collate,
        copy_data_to_device,
        ComposeAPI,
        RandomResizeAPI,
        ToTensorAPI,
        NormalizeAPI,
        PostProcessImage,
    ) = sam3_classes

    print("Loading SAM3 model ...")
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
        model = build_sam3_image_model(
            bpe_path=BPE_PATH,
            load_from_HF=False,
            checkpoint_path=CHECKPOINT_PATH,
        )
        model.eval()

        transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=1008, max_size=1008, square=True, consistent_transform=False
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        postprocessor = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=args.detection_threshold,
            to_cpu=False,
        )

        for split in args.splits:
            process_split(
                split=split,
                data_root=args.data_root,
                npz_dir=args.npz_dir,
                output_dir=args.output_dir,
                model=model,
                transform=transform,
                postprocessor=postprocessor,
                batch_size=args.batch_size,
                start_idx=args.start_idx,
                end_idx=args.end_idx,
                resume=args.resume,
                sam3_classes=sam3_classes,
            )

    print("\nAll done.")


if __name__ == "__main__":
    main()
