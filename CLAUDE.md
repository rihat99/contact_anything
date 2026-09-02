# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Fork of **SAM 3D Body** (Meta, single-image 3D human mesh recovery) extended with
**per-joint contact**, **3D contact-force**, **root-motion** and **pose** heads trained on
climbing **video clips** from the ClimbingVideos corpus. The base model is frozen; only the
appended token blocks, the post-decoder RoPE temporal transformer, the heads and the optional
split-head pose/camera fine-tune copies train.

**2026-09-01 — new era.** The repo was rebuilt twice in one day: the model layer
(`model/`) and then everything else (`data/`, `model/loss/`, `train/`, `utils/`, `scripts/`).
The old `contact/` library, per-vertex contact, still-image datasets, the validation split,
wandb, the sliding-window temporal modules and every non-selected alternative inside the
losses are **gone**. There is **no backwards compatibility** with older configs or
checkpoints and **no test suite** (deleted by design). Everything removed is recoverable
from `main` (commit `3a94a9c`) and from `/data3/rikhat.akizhanov/trash/`; historical runs
live in `../contact_anything/output/old/`.

## Environment

```
CONDA=/data3/rikhat.akizhanov/miniconda3/envs/sam3d          # python + torchrun live in $CONDA/bin
PYTHON=$CONDA/bin/python                                     # neither is on the login PATH
```
Run everything from the repo root. Scripts insert the repo root at the head of `sys.path`
before importing (`scripts/train.py` would otherwise shadow the `train/` package).
Corpus (read-only): `/data3/rikhat.akizhanov/better/data/ClimbingVideos`. Physics needs the
sibling `../BetterRobot` / `../BetterHuman` checkouts (or `$BETTERHUMAN_MODELS_DIR`).

## Key commands

```bash
# Train (resume: --resume auto | --resume path/to/last.pth; --limit-scenes N for smoke runs)
python scripts/train.py --config configs/allmod_rope_t60_gv.yaml
CUDA_VISIBLE_DEVICES=0,1 $CONDA/bin/torchrun --standalone --nproc-per-node=2 \
  scripts/train.py --config configs/allmod_rope_t60_gv.yaml     # data.frames_per_batch is per GPU

# Evaluate on the annotated corpus test split (full-scene protocol; --checkpoint none = frozen baseline)
python scripts/evaluate.py --config configs/allmod_rope_t60_gv.yaml --checkpoint output/<run>/best.pth

# Renders and inference
python scripts/render_video.py --config <yaml> --checkpoint <pth> --split test --scenes 5 --out <dir>
python scripts/render_pose_video.py --config <yaml> --checkpoint <pth> --scenes 3 --out <dir>
python scripts/predict_reconstruction.py --config <yaml> --checkpoint <pth> --root <BVR out-tree>

# Data preparation (scripts/data/)
python scripts/data/extract_frames.py                       # frames/ JPEG tree
CUDA_VISIBLE_DEVICES=0 python scripts/data/precompute_embeddings.py --split all --shard-index 0 --num-shards 4
python scripts/data/convert_kindyn_to_mhr.py                # mhr_1.npz pose pseudo-GT per scene
python scripts/data/precompute_mhr_supervision.py           # mhr_sup_1.npz keypoint/vertex GT

tensorboard --logdir output/<run>/tensorboard/
```

## Layout

| Path | Purpose |
|---|---|
| `model/sam_3d_body/` | Vendored SAM-3D-Body fork, near-upstream (`c259bfc`). Only extension: a generic **extra-token-block** mechanism (append + asymmetric mask + per-layer update callbacks + blind gate), the precomputed-embedding path and the `proj=` split-head override. Never mentions contact/force/motion. |
| `model/wrapper.py` | `SAM3DBodyWrapper`: builds/freezes/eval-pins the base; `forward(img|embedding, geometry, blocks, attention)` → tokens/bounds/frozen MHR; `decode_pose(pose_token, ctx, proj_pose, proj_camera)`. |
| `model/tokens.py` `rope.py` `heads.py` | `LearnedTokenBlock` (token embeddings + anchored posemb/feat update), `RopeTemporalModule` + `CrossModalRopeModule`, `ContactHead`/`ForceHead`/`MotionHead` + contact gate. |
| `model/network.py` `build.py` | `ContactAnything` (wrapper → cross-modal → pose-temporal → final readout with ft copies → heads); `build_model(cfg, device)`. |
| `model/loss/` | One `Loss` interface (`__init__.py`) and one file per term: `contact` (focal), `force`, `motion`, `pose`, `keypoint`, `contact_consistency`, `force_consistency`, `physics` (+ `physics_adapter`). `build_losses(cfg, model, device)`. |
| `data/` | `base.py` = `ClipDataset` ABC (windowing, jitter, full-scene eval) **and the frame schema** (module docstring); `climbing_videos/` (`scene.py` discovery/cameras/labels, `kindyn.py` forces + motion GT, `mhr_gt.py` pose/keypoint GT, `dataset.py`); `transforms.py`, `collate.py` (generic), `loaders.py`, `reconstruction.py` (BVR out-trees, inference only). `build_datasets(cfg, needs)`. |
| `train/` | `config.py` (yaml + `base:` include, **`configs/base.yaml` is the schema**), `checkpoint.py`, `trainer.py`, `logger.py` (tensorboard), `predict.py` (`load_model`, `run_clip`). |
| `utils/` | `geometry.py` (torch SO(3)/lifting/windowed mean), `metrics.py`, `distributed.py`, `betterhuman.py`. |
| `scripts/` | Thin CLIs (above) + `scripts/data/` preparation scripts. |
| `configs/` | `base.yaml` (every key, its default, one comment) + experiment overrides + `datasets/climbing_videos.yaml`. |
| `output/` | Runs (gitignored): `<exp>_<YYYYMMDD_HHMMSS>/{config.yaml, last.pth, best.pth, epoch_XXXX.pth, tensorboard/}`. |

## Architecture

1. **Backbone** DINOv3-H (bf16, frozen) → `[B,1280,32,32]`; with `data.embedding_cache` the
   loader emits the cached embedding and the backbone is skipped (frame JPEGs not decoded).
2. **Promptable decoder** (frozen, dim 1024) with the pose token at index 0. Our
   `LearnedTokenBlock`s (contact 6 anchored at MHR70 `[62,41,15,18,17,20]` + optional global
   tokens, force 6, motion 1 at the pelvis) are appended behind an **asymmetric mask**:
   original tokens never attend the appended blocks, so the frozen pose/MHR outputs have an
   exactly-zero Jacobian w.r.t. every trainable parameter unless a pose-writing brick is on.
   `model.extra_token_attention`: `mutual` (blocks inter-attend) | `causal` (block-triangular).
3. **Cross-modal temporal** (`model.cross_modal_temporal`) — THE post-decoder mixing brick:
   one zero-gated RoPE transformer over the concatenated modality blocks (≥ 2 of
   pose/contact/force/motion, canonical order) across a clip's frames; RoPE position =
   real elapsed seconds × `time_scale`, identical for all tokens of a frame (within-frame
   mixing is the dt = 0 diagonal); `max_rel_sec` window; `frame_valid` masking. Listing
   `pose` writes the pose token and the final MHR is recomputed.
4. **Pose temporal** (`model.pose_temporal`) — the same block on the pose token alone, after
   the cross-modal brick; redundant when `pose` is a listed modality.
5. **Split-head fine-tune** (`model.finetune_pose_head` / `finetune_camera_head`) —
   deepcopy'd trainable copies of the projection FFNs applied to the FINAL readout only
   (in-decoder intermediate predictions keep the frozen originals), own optimizer group at
   `optim.lr × optim.head_lr_scale`. The frozen readout's `pred_cam_t/global_rot/shape/scale`
   are stashed as `*_frozen` (rail anchors).
6. **Heads**: contact `[B,6]` logits (per-token shared FFN); force `[B,6,3]` in body-weight
   units, `model.force.frame` root | local_world_aligned | local, optionally gated by the
   detached contact logits; motion `[B,1,12]` standardized gravity-view pelvis twist
   (lin vel, lin acc, ang vel, ang acc).

### Losses (`model/loss/`, all on one interface)

Each loss returns `LossResult(terms, scalars, stats)`: `terms[name] = (numerator, mass)`
additive pairs (term weights applied INSIDE the loss, numerators graph-connected even at
zero mass), `stats` a float64 additive vector for eval, `metrics(stats)` the reported
numbers. The trainer all-reduces masses once → exact global weighted means under DDP; eval
sums `stats` across batches and ranks. Metric tags: `{split}/{loss.name}/{metric}`.

| section | supervises | with |
|---|---|---|
| `contact_supervision` | six-group contact logits | focal BCE, kindyn label confidence weights |
| `force_supervision` | forces (root frame, bw) | Huber + noncontact L1 + net force/torque vs kindyn GT, `force_confidence` row weights |
| `motion_supervision` | pelvis 12-dim twist | Huber on standardized GT (pinned `standardize` table), train-only outlier cut |
| `pose_supervision` | 125 local MHR q channels + shape/bones/scale | Huber vs `mhr_1.npz` (kindyn refit as MHR), BetterHuman q-space |
| `keypoint_supervision` | camera-frame keypoints/vertices, world vel/acc | Huber vs `mhr_sup_1.npz`; `cam_rail`/`rot_rail` trust regions vs the frozen readout |
| `contact_consistency` | pose path | predicted-contact-gated zero world velocity of the six extremities |
| `force_consistency` | force + pose paths | linear Newton residual (bw, mass cancels), epoch ramp |
| `physics` | forces (exclusive with force_supervision) | RNEA root-wrench residual on a BetterHuman MHR body, six groups mapped to wrists/toe-balls/ankles |

### Invariants (do not break)

- **Freeze boundary = the wrapper.** Everything under `model.wrapper` is frozen and
  eval-pinned (`wrapper.train()` re-pins); everything else in `ContactAnything` trains. The
  checkpoint stores trainable tensors only and hard-fails with a name/shape diff on mismatch.
- **Mask invariant**: original tokens never attend appended blocks; pose/MHR change only via
  the `pose` modality, `pose_temporal` or the head fine-tune copies (`model.writes_pose`).
- **fp16**: backbone bf16, decoder/MHR heads fp32 (MHR sparse ops are fp16-incompatible).
- **Frozen model noise floor** ~5e-4 px run-to-run; warm up before bitwise compares;
  building SAM-3 elsewhere flips global `allow_tf32`.

## Configuration

`configs/base.yaml` **is** the schema: every allowed key with its default. Experiment yamls
set `base: configs/base.yaml` and override; unknown keys hard-error with the dotted path;
`train/config.py::validate` holds the few cross-key checks (modalities ⊆ enabled branches,
pose writers need a pose loss, camera fine-tune needs keypoints, motion loss needs the
motion modality, physics xor force_supervision, monitor max/min by suffix). Which GT signal
groups a dataset loads is **derived** from the enabled losses (`signal_needs`), never
configured. One dataset yaml: `configs/datasets/climbing_videos.yaml` (root, contact_level).

Sections: `model.{contact,force,motion,cross_modal_temporal,pose_temporal,finetune_*}`,
`mhr_body` (BetterHuman archive), `data.{datasets,embedding_cache,frames_per_batch,
num_workers,seed,clip.{frames,stride,jitter},eval_max_frames}`, the eight loss sections
above, `optim.{lr,weight_decay,epochs,warmup_epochs,lr_min,grad_clip,head_lr_scale}`,
`output.{dir,exp_name,log_freq,save_freq,monitor}`.

## Data (ClimbingVideos corpus, read directly)

- `scenes/scenes.db` curated split: 864 train / 108 test scenes; test scenes need
  `annotation.npz` (manual labels on 14 observable joints; the other eight are fixed
  non-contact). Train labels are automatic (`contacts_{level}.npz`, 52 → 22 joint fold).
- **Six groups everywhere, fixed order**: `left_hand, right_hand, left_foot (toe),
  right_foot, left_ankle (heel), right_ankle`; `KINDYN_GROUP_NAMES` in `model/loss`.
  Test-label fold: a known positive wins under partial annotation, a known negative needs
  all source joints annotated. Video labels are motion-gated "stable contact".
- `kindyn_1.npz`: GT forces on 35 named frames (world newtons) folded into the six groups by
  parent joint and converted to body-weight units in the body-root frame; per-frame
  `force_confidence`; per-scene **fitted** `gravity_world` (unit, down-positive, tilted up to
  61° from +y — never assume world y is up). Motion GT = pelvis twist from the MHR rig
  (`mhr_1` root + mean hips), 0.12 s Gaussian smoothing, BVR body-twist stencil.
- Per-frame metric extrinsics (`cam_from_world`, OpenCV) on every scene.
- Clips: `data.clip.frames` × stride (`auto` = `max(1, round(fps/25))`), tiled with
  stateless per-epoch jitter; **eval = ONE clip per (scene, person)**, the longest valid run,
  capped at `data.eval_max_frames`, batch = 1 clip. Batches are flat `[B_clips·T, ...]` with
  `seq_len`, `frame_pos_sec`, `frame_valid`. Full frame/batch schema: `data/base.py`.
- Embedding cache: `features/embedding/` bf16 `[1280,32,32]` per (scene, oid, frame); missing
  files hard-error; bit-exact only at the backbone batch shape it was built with (35).

## Conventions

- Never `rm`: move to `/data3/rikhat.akizhanov/trash/<name>_<date>/`. Commit/push only on
  explicit instruction. Shared GPU box: never touch other users' processes.
- Skepticism: log raw metrics and trajectories; the user owns verdicts. No validation split
  (train on all train scenes, evaluate on test only).
- Keep the codebase pruned: no `.get()` fallbacks on schema-guaranteed keys, no history
  narration in comments, no options nobody runs.
