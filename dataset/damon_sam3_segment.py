#!/usr/bin/env python3
"""Generate SAM3 masks and bounding boxes for the DAMON dataset (v2).

For each sample:
- Segment "person" (always, object order 0)
- Segment each contact object from contact_label_objectwise (skip "supporting")

Output structure (v2):
  {output_dir}/{split}/{idx:04d}/
    metadata.npz   (imgname, object_names, num_detections, bboxes_*, scores_*,
                    contact_vertices_smpl_*, contact_vertices_mhr_*,
                    contact_body_parts_*, best_detection_*,
                    disambiguation_scores_*, metadata_version=2)
    masks/
      {idx:04d}_{object_name}_{det_idx}.png   # always named by object and det index

Disambiguation: when SAM3 finds multiple detections for a contact object, uses
person-mask overlap + 2D joint proximity (from precomputed predictions) to pick
the most likely contacted detection.  Requires --predictions_dir to be populated.
"""

import argparse
import os
import pickle
import re
import sys
import warnings
from pathlib import Path

import cv2
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
    os.path.dirname(__file__), "damon_mhr_contact", "masks_v2"
)
DEFAULT_PREDICTIONS_DIR = os.path.join(
    os.path.dirname(__file__), "damon_mhr_contact", "predictions"
)
DEFAULT_SMPL_SEG_PATH = (
    "/data3/rikhat.akizhanov/climbing/deco/data/smpl_partSegmentation_mapping.pkl"
)
DEFAULT_SMPL_MODEL_PATH = (
    "/data3/rikhat.akizhanov/human_global_motion/better_human/models/smpl/SMPL_NEUTRAL.npz"
)

# ---------------------------------------------------------------------------
# SMPL body part (smpl_index 0-23) → representative MHR70 keypoint indices
# ---------------------------------------------------------------------------
# part2num: Global=0, L_Thigh=1, R_Thigh=2, Spine=3, L_Calf=4, R_Calf=5,
#           Spine1=6, L_Foot=7, R_Foot=8, Spine2=9, L_Toes=10, R_Toes=11,
#           Neck=12, L_Shoulder=13, R_Shoulder=14, Head=15, L_UpperArm=16,
#           R_UpperArm=17, L_ForeArm=18, R_ForeArm=19, L_Hand=20, R_Hand=21,
#           Jaw=22, L_Eye=23
# MHR70 indices: 5=L_shoulder, 6=R_shoulder, 7=L_elbow, 8=R_elbow,
#   9=L_hip, 10=R_hip, 11=L_knee, 12=R_knee, 13=L_ankle, 14=R_ankle,
#   15-17=L_foot, 18-20=R_foot, 41=R_wrist, 62=L_wrist,
#   63=L_olecranon, 64=R_olecranon, 67=L_acromion, 68=R_acromion, 69=neck
SMPL_PART_TO_MHR_KP = {
    0:  [9, 10],              # Global (pelvis) → hips
    1:  [9, 11],              # L_Thigh → L_hip, L_knee
    2:  [10, 12],             # R_Thigh → R_hip, R_knee
    3:  [9, 10],              # Spine → hips
    4:  [11, 13],             # L_Calf → L_knee, L_ankle
    5:  [12, 14],             # R_Calf → R_knee, R_ankle
    6:  [9, 10],              # Spine1 → hips
    7:  [13, 15, 16, 17],     # L_Foot → L_ankle, L_toes, L_heel
    8:  [14, 18, 19, 20],     # R_Foot → R_ankle, R_toes, R_heel
    9:  [5, 6],               # Spine2 → shoulders
    10: [15, 16, 17],         # L_Toes → L_foot keypoints
    11: [18, 19, 20],         # R_Toes → R_foot keypoints
    12: [69],                 # Neck → neck
    13: [5, 67],              # L_Shoulder → L_shoulder, L_acromion
    14: [6, 68],              # R_Shoulder → R_shoulder, R_acromion
    15: [0, 1, 2, 3, 4],      # Head → nose, eyes, ears
    16: [7, 63, 65],          # L_UpperArm → L_elbow, L_olecranon, L_cubital
    17: [8, 64, 66],          # R_UpperArm → R_elbow, R_olecranon, R_cubital
    18: [7, 63, 65],          # L_ForeArm → L_elbow area
    19: [8, 64, 66],          # R_ForeArm → R_elbow area
    20: [62],                 # L_Hand → L_wrist
    21: [41],                 # R_Hand → R_wrist
    22: [0],                  # Jaw → nose
    23: [1, 3],               # L_Eye → L_eye, L_ear
}

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
# Object-name helpers
# ---------------------------------------------------------------------------

def _sanitize_object_name(name: str) -> str:
    """Convert object name to a filesystem-safe lowercase string."""
    safe = name.lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]", "", safe)


# ---------------------------------------------------------------------------
# Disambiguation helpers
# ---------------------------------------------------------------------------

def _mask_png_to_array(path: Path) -> np.ndarray:
    """Load a binary mask PNG as a bool [H, W] numpy array."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    return gray > 127


def _dilated_overlap(mask_a: np.ndarray, mask_b: np.ndarray, radius: int = 10) -> int:
    """Number of pixels in mask_b that overlap with the dilation of mask_a."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    dilated = cv2.dilate(mask_a.astype(np.uint8), kernel)
    return int((dilated & mask_b.astype(np.uint8)).sum())


def _disambiguate_detections(
    person_mask: np.ndarray,          # [H, W] bool
    det_masks: list,                   # list of [H, W] bool arrays
    det_boxes: np.ndarray,             # [n_det, 4]  (x1, y1, x2, y2)
    det_scores: np.ndarray,            # [n_det]
    contact_verts_smpl: np.ndarray,    # SMPL vertex indices in contact (may be empty)
    smpl_part_index: np.ndarray,       # [6890] vertex → SMPL body part id
    pred_kp2d: np.ndarray,             # [70, 2] 2D joint positions in image pixels
) -> tuple:
    """Return (best_detection_idx, combined_scores_array [n_det])."""
    n_det = len(det_masks)
    if n_det == 0:
        return -1, np.zeros(0, dtype=np.float32)

    person_pixels = int(person_mask.sum())

    # --- Signal A: person-mask overlap + adjacency (weight 0.3) ---
    overlap_scores = np.zeros(n_det, dtype=np.float32)
    for d, det_mask in enumerate(det_masks):
        if det_mask is None:
            continue
        if person_pixels > 0:
            overlap = int((person_mask & det_mask).sum())
            adjacency = _dilated_overlap(person_mask, det_mask, radius=10)
            overlap_scores[d] = (overlap + 0.5 * adjacency) / person_pixels

    # --- Signal B: 2D joint proximity (weight 0.5) ---
    if len(contact_verts_smpl) == 0:
        joint_scores = np.full(n_det, 0.5, dtype=np.float32)
    else:
        contact_parts = np.unique(smpl_part_index[contact_verts_smpl])
        relevant_kp_set = set()
        for part_id in contact_parts.tolist():
            for kp in SMPL_PART_TO_MHR_KP.get(part_id, []):
                relevant_kp_set.add(kp)

        if not relevant_kp_set:
            joint_scores = np.full(n_det, 0.5, dtype=np.float32)
        else:
            kp_positions = pred_kp2d[list(relevant_kp_set)]  # [K, 2]
            img_h, img_w = person_mask.shape
            diag = float(np.sqrt(img_h ** 2 + img_w ** 2)) + 1e-6

            joint_scores = np.zeros(n_det, dtype=np.float32)
            for d in range(n_det):
                x1, y1, x2, y2 = det_boxes[d]
                bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
                margin_x, margin_y = bw * 0.2, bh * 0.2
                in_box = (
                    (kp_positions[:, 0] >= x1 - margin_x)
                    & (kp_positions[:, 0] <= x2 + margin_x)
                    & (kp_positions[:, 1] >= y1 - margin_y)
                    & (kp_positions[:, 1] <= y2 + margin_y)
                )
                fraction_in = float(in_box.mean())

                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                dists = np.sqrt(
                    (kp_positions[:, 0] - cx) ** 2 + (kp_positions[:, 1] - cy) ** 2
                )
                proximity = 1.0 - float(dists.min()) / diag
                joint_scores[d] = 0.5 * fraction_in + 0.5 * proximity

    # --- Signal C: SAM3 confidence (weight 0.2) ---
    max_score = float(det_scores.max()) if det_scores.max() > 0 else 1.0
    conf_scores = (det_scores.astype(np.float32) / max_score)

    combined = 0.3 * overlap_scores + 0.5 * joint_scores + 0.2 * conf_scores
    best_idx = int(combined.argmax())
    return best_idx, combined


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
    contact_verts_smpl_per_object: list,   # list of np.ndarray[int64] (SMPL indices)
    contact_verts_mhr_per_object: list,    # list of np.ndarray[int64] (MHR LOD1 indices)
    smpl_part_index: np.ndarray,           # [6890] vertex → part id
    pred_kp2d: np.ndarray,                 # [70, 2] projected 2D joints
):
    """Save masks and metadata.npz for one sample (v2 format)."""
    mask_dir = sample_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    num_objects = len(object_names)
    num_detections = np.zeros(num_objects, dtype=np.int32)
    save_dict = {
        "imgname": np.array(imgname),
        "object_names": np.array(object_names),
        "metadata_version": np.int32(2),
    }

    person_mask_arr = None  # retained for overlap scoring of subsequent objects

    for obj_order, result in enumerate(results_per_object):
        safe_name = _sanitize_object_name(object_names[obj_order])
        cv_smpl = contact_verts_smpl_per_object[obj_order]
        cv_mhr = contact_verts_mhr_per_object[obj_order]

        # Contact body parts (unique SMPL part ids)
        if len(cv_smpl) > 0:
            body_parts = np.unique(smpl_part_index[cv_smpl]).astype(np.int32)
        else:
            body_parts = np.zeros(0, dtype=np.int32)

        save_dict[f"contact_vertices_smpl_{obj_order}"] = cv_smpl
        save_dict[f"contact_vertices_mhr_{obj_order}"] = cv_mhr
        save_dict[f"contact_body_parts_{obj_order}"] = body_parts

        if result is None or len(result["scores"]) == 0:
            num_detections[obj_order] = 0
            save_dict[f"bboxes_{obj_order}"] = np.zeros((0, 4), dtype=np.float32)
            save_dict[f"scores_{obj_order}"] = np.zeros(0, dtype=np.float32)
            save_dict[f"best_detection_{obj_order}"] = np.int32(-1)
            save_dict[f"disambiguation_scores_{obj_order}"] = np.zeros(0, dtype=np.float32)
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

        boxes_np = boxes.cpu().float().numpy().astype(np.float32)
        scores_np = scores.cpu().float().numpy().astype(np.float32)
        save_dict[f"bboxes_{obj_order}"] = boxes_np
        save_dict[f"scores_{obj_order}"] = scores_np

        # Save mask PNGs and collect numpy arrays for disambiguation
        det_mask_arrays = []
        for det_idx, mask in enumerate(masks):
            fname = f"{sample_idx:04d}_{safe_name}_{det_idx}.png"
            mask_path = mask_dir / fname
            _save_mask_png(mask, mask_path)
            det_mask_arrays.append(_mask_png_to_array(mask_path))

        # Retain person mask for subsequent objects
        if obj_order == 0:
            person_mask_arr = det_mask_arrays[0] if det_mask_arrays else None
            save_dict[f"best_detection_{obj_order}"] = np.int32(0)
            save_dict[f"disambiguation_scores_{obj_order}"] = scores_np.copy()
        elif n_det == 1:
            save_dict[f"best_detection_{obj_order}"] = np.int32(0)
            save_dict[f"disambiguation_scores_{obj_order}"] = scores_np.copy()
        else:
            # Multiple detections: disambiguate
            pm = person_mask_arr if person_mask_arr is not None else np.zeros(
                (det_mask_arrays[0].shape if det_mask_arrays[0] is not None else (1, 1)),
                dtype=bool,
            )
            best_idx, dis_scores = _disambiguate_detections(
                person_mask=pm,
                det_masks=det_mask_arrays,
                det_boxes=boxes_np,
                det_scores=scores_np,
                contact_verts_smpl=cv_smpl,
                smpl_part_index=smpl_part_index,
                pred_kp2d=pred_kp2d,
            )
            save_dict[f"best_detection_{obj_order}"] = np.int32(best_idx)
            save_dict[f"disambiguation_scores_{obj_order}"] = dis_scores

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
    smpl_part_index: np.ndarray,
    body_converter,
    predictions_dir: str,
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

    # --- load precomputed predictions (2D keypoints) ---
    pred_kp2d_all = None
    merged_pred_path = Path(predictions_dir) / f"{split}_predictions.npz"
    if merged_pred_path.exists():
        pred_data = np.load(merged_pred_path, allow_pickle=True)
        pred_kp2d_all = pred_data["pred_keypoints_2d"]  # [N, 70, 2]
        print(f"Loaded merged predictions: {pred_kp2d_all.shape}")
    else:
        print(f"[WARNING] Merged predictions not found at {merged_pred_path}."
              f" Samples without per-sample predictions will be skipped.")

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
            # --- load 2D keypoints (required) ---
            pred_kp2d = None
            if pred_kp2d_all is not None and idx < len(pred_kp2d_all):
                pred_kp2d = pred_kp2d_all[idx]
            else:
                per_sample_path = Path(predictions_dir) / split / f"{idx:04d}.npz"
                if per_sample_path.exists():
                    ps = np.load(per_sample_path, allow_pickle=True)
                    pred_kp2d = ps["pred_keypoints_2d"]  # [70, 2]
            if pred_kp2d is None:
                warnings.warn(
                    f"[idx={idx}] No precomputed predictions found — skipping."
                )
                stats["skipped_error"] += 1
                pbar.update(1)
                continue

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
            # Build SMPL contact vertex lists per object
            contact_verts_smpl = [np.zeros(0, dtype=np.int64)]  # person: empty
            contact_verts_mhr = [np.zeros(0, dtype=np.int64)]   # person: empty

            # Person is always object order 0
            qid = _add_text_prompt(dp, "person", FindQueryLoaded, InferenceMetadata)
            query_map[qid] = (idx, 0)
            query_ids.append(qid)

            # Contact objects
            obj_order = 1
            for obj_name, verts_list in contact_labels[idx].items():
                if obj_name == "supporting":
                    continue
                text_prompt = obj_name.replace("_", " ")
                qid = _add_text_prompt(
                    dp, text_prompt, FindQueryLoaded, InferenceMetadata
                )
                query_map[qid] = (idx, obj_order)
                query_ids.append(qid)
                object_names.append(obj_name)

                # SMPL contact vertex indices for this object
                smpl_verts = np.array(verts_list, dtype=np.int64)
                contact_verts_smpl.append(smpl_verts)

                # Convert to MHR LOD1 indices via BodyConverter
                if len(smpl_verts) > 0 and body_converter is not None:
                    binary = np.zeros(6890, dtype=np.float32)
                    binary[smpl_verts] = 1.0
                    result_conv = body_converter.smpl_to_mhr(
                        contacts=torch.from_numpy(binary), target_lod=1
                    )
                    mhr_binary = result_conv.contacts.cpu().numpy()  # [18439]
                    mhr_verts = np.where(mhr_binary > 0)[0].astype(np.int64)
                else:
                    mhr_verts = np.zeros(0, dtype=np.int64)
                contact_verts_mhr.append(mhr_verts)

                obj_order += 1

            try:
                dp = transform(dp)
            except Exception as exc:
                warnings.warn(f"[idx={idx}] Transform failed: {exc}")
                stats["skipped_error"] += 1
                pbar.update(1)
                continue

            datapoints.append(dp)
            sample_meta[idx] = (
                str(imgnames[idx]),
                object_names,
                query_ids,
                contact_verts_smpl,
                contact_verts_mhr,
                pred_kp2d,
            )
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
            (
                imgname_str,
                object_names,
                query_ids,
                cv_smpl_list,
                cv_mhr_list,
                pred_kp2d,
            ) = sample_meta[idx]
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
                    idx,
                    sample_dir,
                    imgname_str,
                    object_names,
                    results_per_object,
                    contact_verts_smpl_per_object=cv_smpl_list,
                    contact_verts_mhr_per_object=cv_mhr_list,
                    smpl_part_index=smpl_part_index,
                    pred_kp2d=pred_kp2d,
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
        description="Generate SAM3 masks/bboxes for DAMON dataset (v2)"
    )
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--npz_dir", default=DEFAULT_NPZ_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--predictions_dir", default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--smpl_seg_path", default=DEFAULT_SMPL_SEG_PATH)
    parser.add_argument("--smpl_model_path", default=DEFAULT_SMPL_MODEL_PATH)
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

    # --- Load SMPL part segmentation ---
    print(f"Loading SMPL part segmentation from {args.smpl_seg_path} ...")
    with open(args.smpl_seg_path, "rb") as f:
        seg_data = pickle.load(f, encoding="latin1")
    smpl_part_index = seg_data["smpl_index"].astype(np.int64)  # [6890]

    # --- Load BodyConverter for SMPL→MHR LOD1 conversion ---
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from mhr_smpl_conversion.body_converter import BodyConverter

    print(f"Loading SMPL faces from {args.smpl_model_path} ...")
    smpl_npz = np.load(args.smpl_model_path, allow_pickle=True)
    smpl_faces = smpl_npz["f"].astype(np.int64)  # [13776, 3]

    print("Building BodyConverter ...")
    body_converter = BodyConverter(smpl_faces=smpl_faces, device="cpu")

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
                smpl_part_index=smpl_part_index,
                body_converter=body_converter,
                predictions_dir=args.predictions_dir,
            )

    print("\nAll done.")


if __name__ == "__main__":
    main()
