# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fork of **SAM 3D Body** (Meta) — single-image 3D human mesh recovery — extended with a
**contact prediction head** and an optional **3D contact-force head**. The base model is frozen;
only contact- and force-named parameters train.
Contact can be predicted **per-vertex** (SMPL 6890 / SMPL-X 10475; MHR not implemented) and/or
**per-joint** (SMPL-X body-22 or four climbing extremities), on single images or on **video clips** via an optional
temporal attention module that provably does not change the frozen model's pose (MHR) predictions.
The force head regresses one 3D vector per climbing extremity, supervised by **physics** (an RNEA
root-wrench residual over reconstructed motion) instead of labels — see `docs/forces.md`.

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
python scripts/train.py --config configs/climbing_videos_joint.yaml
python scripts/train.py --config configs/climbing_videos_joint_temporal_center_v2.yaml
# Two-GPU DDP (`data.frames_per_batch` is per GPU):
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/train.py --config configs/climbing_videos_joint.yaml

# Force training. Physics (RNEA) regime (a) needs the editable better-robot / better-human from
# the sibling ../BetterRobot / ../BetterHuman checkouts (step-01 env wiring); the MHR archive
# resolves via that checkout, or set $BETTERHUMAN_MODELS_DIR. See docs/forces.md.
python scripts/train.py --config configs/climbing_videos_force_warmstart_t7hinge.yaml
# Supervised kindyn forces (force-only build, six groups, no physics/extrinsics):
python scripts/train.py --config configs/climbing_corpus_force_supervised.yaml

# Evaluate on grouped val or the manually annotated corpus test split (30 scenes).
# Reports P/R/F1/F2/IoU, per-extremity metrics and a threshold curve.
python scripts/evaluate.py --checkpoint output/<run>/best.pth --config configs/<experiment>.yaml
python scripts/evaluate.py --checkpoint output/<run>/last.pth \
  --config configs/climbing_videos_joint.yaml --split test --threshold 0.3
# Force runs add physics-consistency metrics (physics_residual, per-extremity force magnitudes
# split by predicted contact, gate-violation rates, vertical-force-sum). Lacking a trained force
# checkpoint, --warm-start builds the untrained force branch from the config's init_contact_checkpoint.
python scripts/evaluate.py --config configs/climbing_videos_force_warmstart_t7hinge.yaml --warm-start --split test

# Qualitative demo (GT vs predicted contacts; force arrows when the checkpoint has a force head)
python scripts/demo.py --checkpoint output/<run>/best.pth --config configs/<experiment>.yaml --num_samples 10
# Rendered corpus videos (contact disks + force arrows; shards over torchrun ranks)
python scripts/render_climbing_video_contacts.py --checkpoint output/<run>/best.pth \
  --config configs/climbing_videos_joint_temporal_center_v2.yaml --split test --overlay-labels

# Tests (fast CPU suite ~15s; add --runslow-style GPU tests via -m slow)
python -m pytest tests/ -q -m "not slow"
python -m pytest tests/ -q                    # everything incl. GPU invariance/grad-flow tests

# Logging: wandb project "contact-anything" (box is logged in) + optional tensorboard
tensorboard --logdir output/<run>/tensorboard/

# Data preparation
python scripts/extract_corpus_frames.py           # corpus frames/ JPEG tree (361 scenes, q95)
python scripts/precompute_masks_damon.py          # SAM3 person masks for DAMON
python scripts/precompute_cam_params_damon.py     # MoGe2 intrinsics for DAMON
python scripts/build_climbing_images.py --config configs/datasets/climbing_images.yaml
python -m viewer --port 8765                      # Contact Atlas dataset browser
```

## Repository Layout

| Path | Purpose |
|---|---|
| `sam_3d_body/` | Vendored SAM 3D Body fork. Upstream code untouched; our additions are the contact head/tokens, `models/modules/temporal.py`, and hooks delimited by `# --- contact temporal hook ---` / efficiency-flag comments. |
| `contact/` | Our library: `config.py` (yaml + `base:` include + strict validation), `model.py` (build/freeze/eval-pin), `targets.py`, `losses.py`, `metrics.py`, `engine.py` (shared forward), `checkpoint.py` (schema v2), `tracking.py` (wandb+TB), `data/` (loaders, collate, splits), `physics/` (`adapter.py` MHR bridge + `loss.py` RNEA residual). |
| `scripts/` | Thin CLIs: train, evaluate, demo, build_climbing_images, precompute_*, render_results_table. |
| `configs/` | `base.yaml` (all defaults, commented) + the kept experiment overrides (self-contained, one per `output/` run + the supervised-force experiment); `configs/datasets/*.yaml` = dataset paths/options. Retired experiment yamls live in `legacy/configs/`. |
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
   pose/keypoint outputs are unaffected by anything contact-side. The optional **force tokens**
   (`model.force_head`; four inheriting the contact anchors, or an explicit
   `force_keypoint_indices` list — force-only builds with no contact tokens/head at all are legal)
   are appended *after* the contact tokens and the mask is extended so no earlier token block
   attends a later one.
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
5. **Force head** (`model.force_head`, optional) — K force tokens → `head_force` (zero-init)
   regressing `out["force"]["joint_forces"] [B,K,3]`, dimensionless (units of body weight); an
   optional `model.force_temporal` block mirrors the contact temporal module (post_decoder only).
   Two mutually exclusive supervision signals:
   - **Physics** (`physics.enabled`, K=4 inheriting the extremity contact anchors, order
     `left_hand,right_hand,left_foot,right_foot`): no labels — `contact/physics/` supervises.
     `adapter.py` (`MHRAdapter`) maps frozen per-frame MHR params + dataset camera extrinsics onto
     a BetterHuman **MHR** body and a world-frame `q` trajectory; `loss.py` (`PhysicsLoss`)
     smooths `q`, finite-differences to v/a, runs **RNEA** with the predicted forces as external
     wrenches, and minimises the 6D root-wrench residual (plus contact-gated / smoothness / L2
     regularisers).
   - **Supervised kindyn forces** (`force_supervision.enabled`, force-only build with six explicit
     anchors `[62,41,15,18,17,20]` = kindyn groups `LH,RH,LF(toe),RF(toe),LA(heel),RA(heel)`):
     `contact/force_supervision.py` trains against the corpus `kindyn_1.npz` GT forces —
     body-weight units, body-root frame, no extrinsics anywhere in the objective.
   Full formulation, frames, and conventions: `docs/forces.md`.

### Invariants (do not break)

- **Freeze filter is name-based**: only params whose dotted name contains `"contact"` **or**
  `"force"` train (tokens, heads, posemb/feat linears, `contact_temporal*`, `force_*`). Any new
  trainable module must carry "contact" or "force" in its attribute path.
- **Mask invariant**: no earlier token block attends a later one — original ⊥ {contact, force},
  contact ⊥ force. Force tokens attend everything, so contact/MHR outputs have an exactly-zero
  Jacobian w.r.t. every force param (D1); forward values agree only to the CUDA noise floor.
- **Frozen modules are eval-pinned** (`contact/model.py::pin_frozen_eval`): `model.train(True)`
  re-pins backbone/decoder/MHR+camera heads to eval (the backbone has DROP_PATH_RATE 0.1 — train
  mode would make it stochastic). The toggled set is **requires_grad-derived** at call time (not a
  name list): a fully-trainable subtree follows the mode in full (incl. its param-less dropout),
  a fully-frozen subtree (e.g. a contact head frozen by `train.freeze_contact`) stays eval, a
  mixed container is descended into.
- **MHR invariance**: `tests/test_temporal_invariance.py` (temporal) and
  `tests/test_force_invariance.py` (force) prove pose/MHR + contact outputs stay within the frozen
  model's CUDA noise floor while the new branch's outputs move. Run after any change to decoder
  hooks.
- **Physics-loss gradient isolation (regime (a))**: the physics loss consumes the frozen model's
  outputs (`out["mhr"]`, camera extrinsics) and the contact probs (`out["contact"]["joint_probs"]`)
  all **detached** — gradients reach **force** params only, and physics never trains the frozen
  base. This isolation from the **contact** head is exact only in regime (a) (`train.freeze_contact`,
  contact frozen). In regime (b) (contact trainable) force→contact attention leaks physics gradients
  into the trainable contact params (the vendored mask permits that direction); the trainer warns,
  and the detach-fix is deferred. See `docs/forces.md`.
- `TRAIN.USE_FP16` stays as shipped (backbone bf16); decoder/MHR heads stay fp32 (MHR sparse ops
  are fp16-incompatible).

## Configuration

Experiments are small yamls with `base: configs/base.yaml` (deep-merge; unknown keys hard-error).
Key sections (see `configs/base.yaml` for full commented defaults):

| Section | Controls |
|---|---|
| `model.contact_head` | anchor indices, global tokens, pooling (`concat`/`attention`/`per_token`), MLP, grid sampling |
| `model.temporal` | enabled, placement, layers/heads, `attend: joint|per_token`, `causal` |
| `model.force_head` / `model.force_temporal` | force branch: enabled, `frame: local_world_aligned|local|root`, `force_keypoint_indices` (null = inherit contact anchors; explicit list enables force-only builds), MLP; force temporal (post_decoder only) |
| `physics` | RNEA loss: enabled, MHR `model_path`/`lod`, `gravity`, `min_frames`, `smoothing_kernel`, per-term `loss.*` weights (all dimensionless) |
| `force_supervision` | supervised kindyn GT-force loss (exclusive with `physics`): `target_frame`, Huber `force`/`huber_delta_bw`, `outlier_bw` cut, `noncontact` L1 |
| `train.freeze_contact` | regime (a): freeze contact, train force only (requires `model.init_contact_checkpoint`) |
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
| `climbing_corpus` (raw ClimbingVideos corpus) | video clips | raw body-22 (contacts_1/2, 52→22 fold); training can reduce to four extremities; optional kindyn GT **forces** for six groups (`left_hand, right_hand, left_foot`=toe`, right_foot, left_ankle`=heel`, right_ankle`) in bw units, body-root frame | SMPL-X joints |
| `lemon`, `rich` | still (viewer-only) | per-vertex | SMPL(-H) 6890 |
| `climbing_videos` (ClimbingVideos_v1 export) | **legacy** — loader in `legacy/climbing_videos.py`; viewer-only | raw body-22 | SMPL-X joints |

ClimbingVideos corpus label semantics (important):
- The corpus is read **directly** from `/data3/.../better/data/ClimbingVideos`
  (`scenes/scenes.db` curated split: 331 train / 30 test scenes; pre-extracted `frames/` JPEG
  tree; `features/` contacts, sam3 masks/bboxes, geometry, kindyn). The exported
  ClimbingVideos_v1 dataset is redundant and its loader is legacy.
- **Train labels are automatic and cover all 22 joints** (contacts_1 by default; the 52→22 hand
  fold is bit-exact with the v1 exporter). Test labels manually annotate 14 observable joints;
  fingers are folded into the wrist/hand labels, and the other eight joints are fixed
  non-contact on reviewed frames. Test-scene discovery requires `annotation.npz` to exist.
- Video joint labels are **motion-gated "stable contact"** (stillness/hysteresis in the
  estimator), a different task from instantaneous contact derived from still-image vertices —
  which is why `derive_from_vertex` defaults to false. The same gap applies to **forces** (R8):
  stable-contact labels ≠ instantaneous load, so the physics loss gates on predicted probs, not
  labels (D8). The **supervised** force loss instead gates on kindyn's own contact mask
  (= contacts_2, the mask the forces were solved under).
- Video scenes also carry **per-frame camera extrinsics** (`cam_from_world`, OpenCV, metric) and
  `gravity_world` = exactly `[0, 1, 0]` (kindyn convention, world y down — not v1's camera-0
  derivation) for the physics loss (still images carry `cam_valid=False`). See `docs/forces.md`.
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
