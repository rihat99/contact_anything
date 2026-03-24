# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork of **SAM 3D Body** (Meta Superintelligence Labs) — a single-image 3D human mesh recovery (HMR) model. The fork adds a **Contact Head** that predicts per-vertex contact states (which of 18,439 MHR mesh vertices are in contact with the environment), trained on the DAMON/DECO dataset.

## Environment Setup

```bash
conda create -n sam_3d_body python=3.11 -y
conda activate sam_3d_body

pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest xtcocotools loguru optree fvcore black pycocotools tensorboard huggingface_hub

pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
```

## Key Commands

Use this python to run code in terminal
```
PYTHON=/data3/rikhat.akizhanov/miniconda3/envs/sam3d/bin/python
```

**Train contact head:**
```bash
python train/train_contact.py
```

**Evaluate a checkpoint:**
```bash
python train/evaluate.py --checkpoint train/output/<run_folder>/best_model.pth
```

**Inference demo (visualize GT vs predicted contacts):**
```bash
python train/inference_demo.py --checkpoint train/output/<run_folder>/best_model.pth --num_samples 10
```

**Verify setup before training:**
```bash
python train/test_setup.py
```

**Monitor training:**
```bash
tensorboard --logdir train/output/contact_head_eth/tensorboard/
```

**General inference demo:**
```bash
python demo.py --image_folder <path> --output_folder <path> --checkpoint_path ./model.ckpt --mhr_path ./mhr_model.pt
```

**Interactive path setup:**
```bash
python train/setup_paths.py
```

**Precompute SAM-3D-Body predictions and DINOv3 features for DAMON:**
```bash
# Full run (with resume support)
CUDA_VISIBLE_DEVICES=0 python dataset/damon_sam3d_precompute.py --resume --gpu 0

# Test on first N samples of one split
CUDA_VISIBLE_DEVICES=0 python dataset/damon_sam3d_precompute.py --splits test --end_idx 5 --gpu 0
```

**Generate SAM3 segmentation masks and bounding boxes for DAMON (person + contact objects):**
```bash
# Full dataset, both splits, with resume support
CUDA_VISIBLE_DEVICES=0 python dataset/damon_sam3_segment.py --resume --gpu 0

# Test on first N samples of one split
CUDA_VISIBLE_DEVICES=0 python dataset/damon_sam3_segment.py --splits trainval --end_idx 10 --gpu 0
```

## Architecture

### Model Pipeline

1. **Backbone** (`sam_3d_body/models/backbones/`) — DINOv3-H (840M) or ViT-H (631M) encodes the input image into dense embeddings.

2. **Promptable Transformer Decoder** (`sam_3d_body/models/decoders/`) — Multi-layer cross-attention decoder with typed query tokens:
   - 1 pose token → initial pose/shape estimate
   - 70 keypoint query tokens → 2D/3D keypoint predictions
   - 0–1 prompt tokens → optional user-provided 2D keypoint or mask prompts
   - **21 contact tokens** ← new addition; one per first 21 MHR70 keypoints (body joints + toes/heels)

3. **Prediction Heads** (`sam_3d_body/models/heads/`):
   - `mhr_head.py` — Predicts MHR parameters (pose: 260D, shape: 45D, scale: 28D, hand: 108D, face: 72D)
   - `camera_head.py` — Predicts camera scale/translation
   - `contact_head.py` — **New.** Mean-pools the 21 contact tokens → 2-layer MLP → 18,439 binary vertex logits

4. **Main Model** (`sam_3d_body/models/meta_arch/sam3d_body.py`) — `SAM3DBody` orchestrates the full forward pass. The model key `contact_head` stores the new head; `contact_tokens` in the decoder stores the new learnable queries.

### Contact Head Details (`sam_3d_body/models/heads/contact_head.py`)

- Input: `[B, 21, C]` contact tokens
- Operation: mean-pool → FFN (depth=2, hidden=C//4) → `[B, 18439]` logits
- Loss: binary cross-entropy with positive class weighting (heavily imbalanced: ~14–15% contact rate)
- Training freezes everything except contact head weights and contact tokens

### Dataset (`dataset/`)

- `damon_mhr.py` — PyTorch `Dataset` wrapping NPZ files
- NPZ keys: `imgname` (image paths), `contact_label` (binary `[N, 18439]`), `bbox` (optional), `cam_k` (optional)
- Train set: ~4,384 samples; Test set: ~785 samples; single person per image throughout
- Data root configured via `DATASET.DATA_ROOT` in `train/config.yaml`

**Precomputed data:**

```
dataset/damon_mhr_contact/
  masks/{split}/{idx:04d}/
    metadata.npz                          — object_names, num_detections, bboxes_*, scores_*
    masks/{idx:04d}_000.png               — person mask (largest bbox, always object 0)
    masks/{idx:04d}_001.png               — first contact object (single detection)
    masks/{idx:04d}_001_0.png             — first contact object, detection 0 (multiple)
    masks/{idx:04d}_001_1.png             — first contact object, detection 1 (multiple)
  features/{split}/{idx:04d}.pt           — DINOv3 encoder features [1280, 56, 56] float16
  predictions/{split}/{idx:04d}.npz       — per-sample pose predictions (for resume)
  predictions/{split}_predictions.npz    — merged pose predictions for full split
```

Generated by:
- `dataset/damon_sam3_segment.py` — SAM3 open-vocabulary segmentation; text prompts are `"person"` and contact object names from `contact_label_objectwise` (skips `"supporting"`); multiple person detections are deduplicated by keeping the largest bbox
- `dataset/damon_sam3d_precompute.py` — SAM-3D-Body pose predictions and DINOv3 features; uses person mask from above to derive a tighter bbox, falling back to `hot_dca_{split}_detect.npz` bbox if no mask exists

`hot_dca_{split}_detect.npz` (keys: `imgname`, `bbox`, `cam_k`) is the source of truth for image list and camera intrinsics.

Prediction NPZ keys per sample: `imgname`, `pred_keypoints_3d` [70,3], `pred_keypoints_2d` [70,2], `pred_cam_t` [3], `focal_length`, `pred_pose_raw` [266], `global_rot` [3], `body_pose_params`, `hand_pose_params` [108], `scale_params` [28], `shape_params` [45], `mhr_model_params`, `pred_joint_coords` [127,3], `bbox_used` [4], `mask_available`.

## Configuration (`train/config.yaml`)

All training is controlled by `train/config.yaml`. Key sections:

| Section | Purpose |
|---|---|
| `MODEL.CHECKPOINT_PATH` | Pre-trained SAM 3D Body checkpoint (`.ckpt`) |
| `MODEL.MHR_MODEL_PATH` | MHR parametric model weights (`.pt`) |
| `MODEL.CONTACT_HEAD` | Architecture of the contact head (num tokens, vertices, MLP depth) |
| `TRAIN.USE_FP16` | Keep `false` — MHR sparse ops are incompatible with fp16 |
| `TRAIN.POS_WEIGHT` | Class weight for imbalanced labels; `null` = auto-compute |
| `DATASET.TRAINVAL_NPZ` / `TEST_NPZ` | Paths to DAMON/DECO npz files |
| `OUTPUT.DIR` | Base output dir; each run creates `EXP_NAME_YYYYMMDD_HHMMSS/` |

## Training Outputs

Each run creates `train/output/<EXP_NAME_YYYYMMDD_HHMMSS>/`:
- `best_model.pth` — checkpoint with best validation loss
- `final_model.pth` — last epoch checkpoint
- `checkpoint_epoch_N.pth` — periodic saves (every `SAVE_FREQ` epochs)
- `tensorboard/` — loss/metric logs
- `config.yaml` — copy of run config

## Checkpoints

Pre-trained base models are on HuggingFace (access required):
- `facebook/sam-3d-body-dinov3` (DINOv3-H backbone, default)
- `facebook/sam-3d-body-vith` (ViT-H backbone)

Local paths on this machine are set in `train/config.yaml` under `MODEL.CHECKPOINT_PATH` and `MODEL.MHR_MODEL_PATH`.
