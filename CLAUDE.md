# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork of **SAM 3D Body** (Meta Superintelligence Labs) — a single-image 3D human mesh recovery (HMR) model. The fork implements **Step 1** of a multi-step contact prediction proposal: a separate **InteractionDecoder** that cross-attends to the frozen body decoder's pose token and predicts per-vertex contact states (18,439 MHR mesh vertices), trained on the DAMON/DECO dataset.

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

**Verify setup before training:**
```bash
python train/test_setup.py --config configs/step1_contact.yaml
```

**Train contact head (Step 1):**
```bash
python train/train_contact.py --config configs/step1_contact.yaml
```

**Evaluate a checkpoint:**
```bash
python train/evaluate.py --checkpoint train/output/<run_folder>/best_model.pth
```

**Inference demo (visualize GT vs predicted contacts):**
```bash
python train/inference_demo.py --checkpoint train/output/<run_folder>/best_model.pth --num_samples 10
```

**Monitor training:**
```bash
tensorboard --logdir train/output/step1_contact_*/tensorboard/
```

**General inference demo:**
```bash
python demo.py --image_folder <path> --output_folder <path> --checkpoint_path ./model.ckpt --mhr_path ./mhr_model.pt
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

## Architecture (Step 1)

### Forward Pass

```
Image (precomputed DINOv3 features cached on disk)
  ↓
[Body Decoder — FROZEN]  →  body_tokens[:, 0:1, :]  (MHR pose token, [B, 1, 1024])
  ↓                                ↓ detach
image_embeddings [B, 1280, 56, 56] ─→ [InteractionDecoder] → contact_tokens [B, 16, 1024]
                                                                     ↓
                                                            [ContactHead] → logits [B, 18439]
```

At training time the DINOv3 backbone is skipped — precomputed features are loaded from disk. The body decoder still runs (frozen) on those features each forward pass and provides its pose token as body context.

### Model Pipeline

1. **Backbone** (`sam_3d_body/models/backbones/`) — DINOv3-H (840M) or ViT-H (631M). **Frozen and bypassed at training time** via precomputed features.

2. **Promptable Transformer Decoder** (`sam_3d_body/models/decoders/promptable_decoder.py`) — **Frozen.** Produces `body_tokens` (all query token states after decoding); the MHR pose token at index 0 is exposed as `output["body_tokens"]` for the interaction decoder.

3. **InteractionDecoder** (`sam_3d_body/models/decoders/interaction_decoder.py`) — **NEW, trainable.**
   - K=16 learnable contact query embeddings
   - 4 transformer decoder layers, each: self-attn → cross-attn to image features → cross-attn to body pose token (detached)
   - Image features projected from 1280 → 1024 via `nn.Linear`
   - Output: `[B, 16, 1024]` contact tokens

4. **ContactHead** (`sam_3d_body/models/heads/contact_head.py`) — **NEW, trainable.**
   - Attention pooling over 16 contact tokens → `[B, 1024]`
   - 2-layer MLP (hidden=256) → `[B, 18439]` raw logits

5. **Prediction Heads** (`sam_3d_body/models/heads/`):
   - `mhr_head.py` — MHR parameters (frozen)
   - `camera_head.py` — camera scale/translation (frozen)
   - `contact_head.py` — per-vertex contact logits (trainable)

6. **Main Model** (`sam_3d_body/models/meta_arch/sam3d_body.py`) — Two additions:
   - `forward_pose_branch(batch, precomputed_features=None)` — skips backbone when precomputed features provided
   - `output["body_tokens"]` — exposes pose token `[B, 1, 1024]` from body decoder

### Trainable Parameters

| Module | Status |
|--------|--------|
| Backbone | Frozen (bypassed at train time) |
| Body Decoder | Frozen |
| `model.interaction_decoder` | **Trainable** |
| `model.head_contact` | **Trainable** |

### Dataset (`dataset/`)

- `damon_mhr.py` — PyTorch `Dataset` wrapping NPZ files
- NPZ keys: `imgname` (image paths), `contact_label` (binary `[N, 18439]`), `bbox` (optional), `cam_k` (optional)
- Train set: ~4,384 samples; Test set: ~785 samples; single person per image throughout
- Data root configured via `DATASET.DATA_ROOT` in `configs/step1_contact.yaml`

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
- `dataset/damon_sam3_segment.py` — SAM3 open-vocabulary segmentation
- `dataset/damon_sam3d_precompute.py` — SAM-3D-Body pose predictions and DINOv3 features

`hot_dca_{split}_detect.npz` (keys: `imgname`, `bbox`, `cam_k`) is the source of truth for image list and camera intrinsics.

Prediction NPZ keys per sample: `imgname`, `pred_keypoints_3d` [70,3], `pred_keypoints_2d` [70,2], `pred_cam_t` [3], `focal_length`, `pred_pose_raw` [266], `global_rot` [3], `body_pose_params`, `hand_pose_params` [108], `scale_params` [28], `shape_params` [45], `mhr_model_params`, `pred_joint_coords` [127,3], `bbox_used` [4], `mask_available`.

## Configuration

Configs live in `configs/`:

| File | Purpose |
|------|---------|
| `configs/step1_contact.yaml` | **Active config** — Step 1 training (InteractionDecoder + ContactHead) |
| `configs/base_sam3d.yaml` | Reference — documents frozen SAM3D base model settings |

Key sections in `configs/step1_contact.yaml`:

| Section | Purpose |
|---------|---------|
| `MODEL.CHECKPOINT_PATH` | Pre-trained SAM 3D Body checkpoint (`.ckpt`) |
| `MODEL.MHR_MODEL_PATH` | MHR parametric model weights (`.pt`) |
| `MODEL.INTERACTION_DECODER` | InteractionDecoder architecture (K tokens, layers, heads, ffn_dim) |
| `MODEL.CONTACT_HEAD` | ContactHead MLP depth, pool mode, vertex count |
| `TRAIN.USE_FP16` | Keep `false` — MHR sparse ops are incompatible with fp16 |
| `TRAIN.USE_PRECOMPUTED_FEATURES` | `true` — skips DINOv3 backbone at training time |
| `DATASET.CONTACT_NPZ` | Paths to DAMON LOD1 contact label NPZ files |
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

Local paths on this machine are set in `configs/step1_contact.yaml` under `MODEL.CHECKPOINT_PATH` and `MODEL.MHR_MODEL_PATH`.
