# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fork of **SAM 3D Body** (Meta) — single-image 3D human mesh recovery — extended with a
**contact prediction head**. The base model is frozen; only contact-named parameters train.
Contact can be predicted **per-vertex** (SMPL 6890 / SMPL-X 10475; MHR not implemented) and/or
**per-joint** (SMPL-X body-22 or four climbing extremities), on single images or on **video clips** via an optional
temporal attention module that provably does not change the frozen model's pose (MHR) predictions.

## Environment Setup

```bash
conda create -n sam_3d_body python=3.11 -y
conda activate sam_3d_body

pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest xtcocotools loguru optree fvcore black pycocotools tensorboard huggingface_hub

pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
```

Use this python to run code in terminal:
```
PYTHON=/data3/rikhat.akizhanov/miniconda3/envs/sam3d/bin/python
```

## Key Commands

All commands run from the repo root.

```bash
# Train (config = experiment yaml; resume: --resume auto | --resume path/to/last.pth)
python scripts/train.py --config configs/damon_baseline.yaml
python scripts/train.py --config configs/climbing_videos_joint_temporal.yaml
# Two-GPU DDP (`data.frames_per_batch` is per GPU):
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/train.py --config configs/climbing_videos_joint.yaml

# Evaluate on grouped val or the physical manually annotated ClimbingVideos test split.
# Reports P/R/F1/F2/IoU, per-extremity metrics and a threshold curve.
python scripts/evaluate.py --checkpoint output/<run>/best.pth --config configs/<experiment>.yaml
python scripts/evaluate.py --checkpoint output/<run>/last.pth \
  --config configs/climbing_videos_joint.yaml --split test --threshold 0.3

# Qualitative demo (GT vs predicted contacts)
python scripts/demo.py --checkpoint output/<run>/best.pth --config configs/<experiment>.yaml --num_samples 10
python scripts/demo_climbing_videos.py --checkpoint output/<run>/last.pth \
  --config configs/climbing_videos_joint.yaml --split test --threshold 0.3

# Tests (fast CPU suite ~15s; add --runslow-style GPU tests via -m slow)
python -m pytest tests/ -q -m "not slow"
python -m pytest tests/ -q                    # everything incl. GPU invariance/grad-flow tests

# Logging: wandb project "contact-anything" (box is logged in) + optional tensorboard
tensorboard --logdir output/<run>/tensorboard/

# Data preparation
python scripts/precompute_masks_damon.py          # SAM3 person masks for DAMON
python scripts/precompute_cam_params_damon.py     # MoGe2 intrinsics for DAMON
python scripts/build_climbing_images.py --config configs/datasets/climbing_images.yaml
python -m viewer --port 8765                      # Contact Atlas dataset browser
```

## Repository Layout

| Path | Purpose |
|---|---|
| `sam_3d_body/` | Vendored SAM 3D Body fork. Upstream code untouched; our additions are the contact head/tokens, `models/modules/temporal.py`, and hooks delimited by `# --- contact temporal hook ---` / efficiency-flag comments. |
| `contact/` | Our library: `config.py` (yaml + `base:` include + strict validation), `model.py` (build/freeze/eval-pin), `targets.py`, `losses.py`, `metrics.py`, `engine.py` (shared forward), `checkpoint.py` (schema v2), `tracking.py` (wandb+TB), `data/` (loaders, collate, splits). |
| `scripts/` | Thin CLIs: train, evaluate, demo, build_climbing_images, precompute_*, render_results_table. |
| `configs/` | `base.yaml` (all defaults, commented) + experiment overrides; `configs/datasets/*.yaml` = dataset paths/options. |
| `tests/` | pytest suite (`-m slow` = GPU integration: temporal invariance, grad flow). |
| `viewer/` | Standalone FastAPI dataset inspector with frame/sequence video skeleton views and still-image contact meshes. |
| `tools/` | Legacy `view_dataset.py` browser and `climbing_contact_stats.py` (source-tree stats, SMPL-X). |
| `legacy/` | Superseded code kept for reference — see `legacy/README.md` for why each item is there and what functionality it still uniquely has. |
| `conversion/smplx_smpl_conversion/` | SMPL-X→SMPL vertex/param/contact conversion (used by the climbing-images builder). |
| `output/` | New training runs (gitignored). `train/output/` holds pre-refactor historical runs. |

## Architecture

1. **Backbone** — DINOv3-H (bf16, frozen) → image embeddings `[B,1280,32,32]`.
2. **Promptable decoder** (6 layers, dim 1024) with typed tokens. Contact adds keypoint-anchored
   tokens (configurable MHR70 anchor indices, `model.contact_head.contact_keypoint_indices`;
   default = first 21; the four-extremity config uses `[62,41,13,14]` for left/right
   wrist then left/right ankle) + `num_global_tokens` extra tokens.
   An **asymmetric attention mask** stops all original tokens from attending to contact tokens —
   pose/keypoint outputs are unaffected by anything contact-side.
3. **Per-target contact heads** — `head_contact` is an `nn.ModuleDict`: pooled modes support
   `vertex` → `[B, 6890|10475]` or body-22 joint logits. `pool_mode: per_token` applies one
   shared classifier independently to each token; ClimbingVideos uses four tokens → `[B,4]`.
   Output: `out["contact"]["<target>_logits"]`.
4. **Temporal module** (`model.temporal`, optional) — zero-init-gated pre-LN attention blocks over
   the frames of a clip, order-aware via sinusoidal encoding of real elapsed seconds
   (`frame_pos_sec`), optional frame-level causal mask. Placements: `post_decoder` (default),
   `between_layers` (runs at decoder layers 0–4, shared weights), `pre_decoder` (experimental,
   contact-private bottlenecked feature branch). Batches are homogeneous-T flattened clips
   (`[B_clips*T, ...]` + `seq_len`/`frame_pos_sec`/`frame_valid`); single images are T=1.

### Invariants (do not break)

- **Freeze filter is name-based**: only params whose dotted name contains `"contact"` train
  (tokens, heads, posemb/feat linears, `contact_temporal*`). Any new trainable module must carry
  "contact" in its attribute path.
- **Frozen modules are eval-pinned** (`contact/model.py::pin_frozen_eval`): `model.train(True)`
  re-pins backbone/decoder/MHR+camera heads to eval (the backbone has DROP_PATH_RATE 0.1 — train
  mode would make it stochastic). Only contact modules toggle.
- **MHR invariance**: `tests/test_temporal_invariance.py` proves pose/MHR outputs stay within the
  frozen model's CUDA noise floor while contact logits move orders of magnitude. Run after any
  change to decoder hooks.
- `TRAIN.USE_FP16` stays as shipped (backbone bf16); decoder/MHR heads stay fp32 (MHR sparse ops
  are fp16-incompatible).

## Configuration

Experiments are small yamls with `base: configs/base.yaml` (deep-merge; unknown keys hard-error).
Key sections (see `configs/base.yaml` for full commented defaults):

| Section | Controls |
|---|---|
| `model.contact_head` | anchor indices, global tokens, pooling (`concat`/`attention`/`per_token`), MLP, grid sampling |
| `model.temporal` | enabled, placement, layers/heads, `attend: joint|per_token`, `causal` |
| `contact.topology` | `smpl` / `smplx` (`mhr` → NotImplementedError) |
| `contact.targets.vertex/joint` | enabled, weight, loss params, `joint_set`, subset masking, `derive_from_vertex`, confidence weights |
| `data.datasets` | list of `{name, config, split}`; `frames_per_batch` (memory-flat batch budget), `sequence.{frames_per_clip,frame_stride,jitter}` |
| `train` | `backbone_no_grad`, `detach_interm_preds` (both true; ~20% faster, grad-asserted no-ops) |
| `logging` | wandb (project `contact-anything`) + tensorboard |
| `output` | run dir, `monitor` (e.g. `val/vertex_f1`; `*_f1`/`*_iou`→max, `loss`→min) |

## Datasets

| name | granularity | labels | topology |
|---|---|---|---|
| `damon` (DECO) | still | per-vertex | SMPL 6890 |
| `climbing_images` (ClimbingImages_v1) | still | per-vertex (+SMPL params) | SMPL 6890 |
| `climbing_videos` (ClimbingVideos_v1) | video clips | raw body-22; training can reduce to four extremities | SMPL-X joints |
| `lemon`, `rich` | still (viewer-only) | per-vertex | SMPL(-H) 6890 |

ClimbingVideos label semantics (important):
- **Train labels are automatic and cover all 22 joints**. Test labels manually annotate 14
  observable joints; fingers are folded into the wrist/hand labels, and the other eight joints
  are fixed non-contact on reviewed frames. The loader **raises** if test labels are requested
  while `contacts.npz` has `pending=True`.
- Video joint labels are **motion-gated "stable contact"** (stillness/hysteresis in the exporter),
  a different task from instantaneous contact derived from still-image vertices — which is why
  `derive_from_vertex` defaults to false.
- Train/val split is grouped by **source video** (chunks of one video never straddle splits).
- The four-output target order is `left_hand, right_hand, left_foot, right_foot`; each foot is
  `ankle OR foot`. A known positive wins under partial annotation, while a known negative needs
  both source joints annotated. Confidence is max over positive evidence and mean when both are
  known free. The current experiment uses focal-only loss (`alpha=0.8`, `gamma=2`) and exact
  global confidence-mass reduction under DDP.

Datasets emit per-target `(gt, sup_mask)`; missing targets are fully masked, so image (vertex) and
video (joint) datasets mix at batch level (homogeneous-T interleaving).

## Training Outputs & Checkpoints

Each run: `output/<EXP_NAME_YYYYMMDD_HHMMSS>/` with `best.pth` (by `output.monitor`), rolling
`last.pth`, periodic `epoch_XXXX.pth`, `config.yaml` (resolved), `tensorboard/`.
Checkpoints are **schema v2**: trainable-only weights + optimizer/scheduler + epoch/step + best
metric + resolved config + wandb run id + RNG states. Loading **hard-fails with a param diff** on
architecture mismatch (never silent random init). `--resume auto` reproduces the uninterrupted
run exactly (RNG + stateless window jitter restored). Pre-refactor checkpoints are incompatible.

Base model checkpoints (HuggingFace, paths set in `configs/base.yaml`):
`facebook/sam-3d-body-dinov3` (default) / `facebook/sam-3d-body-vith`.
