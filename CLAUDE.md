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
python scripts/train.py --config configs/temporal_posetoken.yaml
CUDA_VISIBLE_DEVICES=6,7 $CONDA/bin/torchrun --standalone --nproc-per-node=2 \
  scripts/train.py --config configs/temporal_tokens.yaml        # data.frames_per_batch is per GPU
# Frozen SAM3D-as-SMPL-X baseline on the SAME test protocol -> the `frozen` tensorboard run
# (output.frozen_metrics points at the json; recompute whenever eval_max_frames/stride change)
python scripts/eval_frozen_smplx.py --config configs/temporal_posetoken.yaml --out output/frozen_sam3d_smplx.json

# Evaluate on the annotated corpus test split (full-scene protocol; --checkpoint none = frozen baseline)
python scripts/evaluate.py --config configs/allmod_rope_t60_gv.yaml --checkpoint output/<run>/best.pth

# Renders and inference
python scripts/render_video.py --config <yaml> --checkpoint <pth> --split test --scenes 5 --out <dir>
python scripts/render_pose_video.py --config <yaml> --checkpoint <pth> --scenes 3 --out <dir>
python scripts/render_smplx_video.py --config configs/smplx_probe.yaml --checkpoint <pth> --scenes 8 --out <dir>   # GT | frozen MHR | SMPL-X head
python scripts/predict_reconstruction.py --config <yaml> --checkpoint <pth> --root <BVR out-tree>

# Data preparation (scripts/data/)
python scripts/data/extract_frames.py                       # frames/ JPEG tree
CUDA_VISIBLE_DEVICES=0 python scripts/data/precompute_embeddings.py --split all --shard-index 0 --num-shards 4
python scripts/data/convert_kindyn_to_mhr.py                # mhr_1.npz pose pseudo-GT per scene
python scripts/data/precompute_mhr_supervision.py           # mhr_sup_1.npz keypoint/vertex GT

tensorboard --logdir output/<run>/tensorboard/    # sections: optim, loss_train, loss_test, metric_<group>; run `frozen` = baseline line
```

## Layout

| Path | Purpose |
|---|---|
| `model/sam_3d_body/` | Vendored SAM-3D-Body fork, near-upstream (`c259bfc`). Only extension: a generic **extra-token-block** mechanism (append + asymmetric mask + per-layer update callbacks + blind gate), the precomputed-embedding path and the `proj=` split-head override. Never mentions contact/force/motion. |
| `model/wrapper.py` | `SAM3DBodyWrapper`: builds/freezes/eval-pins the base; `forward(img|embedding, geometry, blocks, attention)` → tokens/bounds/frozen MHR; `decode_pose(pose_token, ctx, proj_pose, proj_camera)`. |
| `model/tokens.py` `rope.py` `heads.py` | `LearnedTokenBlock` (token embeddings + anchored posemb/feat update), `RopeTemporalModule` + `CrossModalRopeModule`, `ContactHead`/`ForceHead`/`MotionHead` + contact gate, `SmplxHead` (pose-token probe). |
| `model/network.py` `build.py` | `ContactAnything` (wrapper → cross-modal → pose-temporal → final readout with ft copies → heads); `build_model(cfg, device)`. |
| `model/loss/` | One `Loss` interface (`__init__.py`) and one file per term: `contact` (BCE | focal), `force`, `motion`, `pose`, `keypoint`, `smplx`, `contact_consistency`, `force_consistency`, `physics` (+ `physics_adapter`). `build_losses(cfg, model, device)`. |
| `data/` | `base.py` = `ClipDataset` ABC (windowing, jitter, full-scene eval) **and the frame schema** (module docstring); `climbing_videos/` (`scene.py` discovery/cameras/labels, `kindyn.py` forces + motion + SMPL-X GT, `mhr_gt.py` pose/keypoint GT, `dataset.py`); `transforms.py`, `collate.py` (generic), `loaders.py`, `reconstruction.py` (BVR out-trees, inference only). `build_datasets(cfg, needs)`. |
| `train/` | `config.py` (yaml + `base:` include, **`configs/base.yaml` is the schema**), `checkpoint.py`, `trainer.py`, `logger.py` (tensorboard sections + the `frozen` baseline run), `predict.py` (`load_model`, `run_clip`). |
| `utils/` | `geometry.py` (torch SO(3)/6D/CLIFF camera proxy/projection/Procrustes/lifting/windowed mean), `metrics.py`, `distributed.py`, `betterhuman.py`. |
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
6. **Heads**: contact `[B,6]` logits — `model.contact.source: tokens` = one shared FFN per
   learned contact token; `pose_token` = NO token block, one flat FFN (`C→C/4→6`, same depth)
   straight from the final pose token; force `[B,6,3]` in body-weight
   units, `model.force.frame` root | local_world_aligned | local, optionally gated by the
   detached contact logits; motion `[B,1,12]` standardized gravity-view pelvis twist
   (lin vel, lin acc, ang vel, ang acc).
7. **SMPL-X pose-token probe** (`model.smplx`, `SmplxHead`) — reads the FINAL pose token
   (index 0; the same token the frozen MHR + camera heads read; never written) with two
   from-scratch FFNs of SAM3D's own head shape (`C→C→D`, zero-init last linear, residual on
   a fixed mean: upright facing the camera, identity joints, zero betas, CLIFF `(1,0,0)`)
   and regresses the corpus SMPL-X body in the CAMERA frame under BetterHuman's `q`
   convention (root = pelvis pose): root + 21 body 6D rotations, 10 betas, and a CLIFF
   crop weak-perspective `(s,tx,ty)` lifted with the crop box + true focal to the pelvis
   position. BetterHuman's 22-joint FK (hands never move a body joint; face/expression are
   zero corpus-wide; kindyn `scale` is identically 1) and the full-frame projection run
   inside the head, so `out["smplx"]` carries params, `joints_cam [B,22,3]`, `kp2d_full`
   px and `kp2d_crop` in `[-0.5,0.5]`. Legal with every token block and temporal brick
   off (145 tokens, only the head trains; `configs/smplx_probe.yaml`, result 2026-09-02:
   59 mm MPJPE / 40 PA in ~3 epochs). With the head enabled the SMPL-X body IS the pose
   output: the final MHR recompute is skipped even when a temporal brick writes the pose
   token (`out["mhr"]` = the frozen readout) and the MHR-consuming losses / head fine-tunes
   are rejected by the config validation. The frozen model's own SMPL-X numbers come from
   the corpus refit `features/sam3d/<shard>/<scene>/smplx_params.npz` (camera-frame
   classic params; `q[:3] = transl + pelvis_offset(β)`), scored OFFLINE by
   `scripts/eval_frozen_smplx.py` and drawn as the `frozen` tensorboard run.
8. **Temporal probes** (`configs/temporal_posetoken.yaml` / `temporal_tokens.yaml`, 2026-09-02):
   `cross_modal_temporal` over `[pose]` alone (K = 1, a plain 4-layer RoPE transformer on the
   pose token — a single modality is legal) vs over `[pose, contact]` (K = 7); the SMPL-X head
   reads the mixed pose token, contact comes from the pose token (flat head) or from the six
   mixed contact tokens. RoPE audited 2026-09-02 (`~/.claude/handoffs/smplx_probe/rope_review.md`):
   positions = true seconds from the per-scene fps, auto stride identical in train/eval, all
   relative-time / fps-equivalence / mask properties verified; known caveats: eval clips
   (T=120-360, ±2.5 s window) give each frame ~2× the keys seen at T=60 training, and 9/32
   RoPE frequency pairs are near-constant inside a 2.5 s window (base 10000 is an LLM default).
   Results (5 epochs, test, WHAM metrics; frozen 61.1 / 44.1 / 78.1 mm, accel 11.9): pose-token
   65.1 / 45.7 / 85.7, F1 0.909; tokens 60.7 / 41.4 / 78.4, F1 0.917 — tokens ahead at every
   epoch. Both lost epoch 0 to the per-epoch warm-up bug (fixed the same day: per-step
   `optim.warmup_steps`).
9. **Round 2 on the tokens build** (2026-09-02 evening; only the winner
   `configs/temporal_tokens_b8_lr2.yaml` + its run survive, the three sibling arms were trashed
   to `/data3/rikhat.akizhanov/trash/round2_arms_20260902/`): the trained-state gradient noise
   scale is 16 / 108 / 239 clips (SMPL-X head / temporal block / contact head) vs 2 clips per
   step, so the arms ran 4 clips/GPU (8/step, 459 steps/epoch). At the UNSCALED lr 1e-4 that
   LOSES on pose at equal epochs (66.2 mm, step-limited); with sqrt lr scaling (lr 2e-4) it is
   the best run so far: 60.1 / 41.2 / 77.9, F1 0.919, P@R0.9 0.926, in a quarter of the steps.
   The cost-weighted contact loss (`neg_weight 2`, heel `pos_weight 4`, `transition_tolerance 2`;
   options kept, default off) was worse along the whole P/R curve (P@R0.9 0.919 vs 0.928) — a
   threshold on the plain model buys precision for free; heels stay unpredicted. The optimizer
   hygiene bundle (`betas .95`, `decay_1d false`, `ema`, `grad_clip_per_group`; options kept)
   gained 1.5-4 mm on pose at lr 1e-4 but cost ~0.005 F1, not attributable within the bundle.
   Gradient surgery (PCGrad) was measured pointless: contact/pose gradients on the shared block
   have cos ≈ 0 with balanced magnitudes.

### Losses (`model/loss/`, all on one interface)

Each loss returns `LossResult(terms, scalars, stats)`: `terms[name] = (numerator, mass)`
additive pairs (term weights applied INSIDE the loss, numerators graph-connected even at
zero mass), `stats` a float64 additive vector for eval, `metrics(stats)` the reported
numbers. The trainer all-reduces masses once → exact global weighted means under DDP; eval
sums `stats` across batches and ranks. Tensorboard sections: `optim/*`, `loss_train/total`
+ `loss_train/<loss>.<term>`, the same under `loss_test/`, and `metric_<group>/<metric>`
(`Loss.metric_group`, = the loss name except `smplx` → `pose`). Per-batch diagnostic
`scalars` are NOT logged. `output.monitor` names one of those tags.

| section | supervises | with |
|---|---|---|
| `contact_supervision` | six-group contact logits | `criterion: bce` (default; calibrated, constant gradient scale) or `focal` (alpha/gamma), kindyn label confidence weights; precision knobs (2026-09-02): `neg_weight` (cost of a negative row), `pos_weight` (six per-group positive costs, heels ~3 % prior), `transition_tolerance` (drop k frames either side of a GT transition from the LOSS only) — all enter the loss mass. Metrics: micro f1/precision/recall/iou + `precision_at_r90` (0.1..0.9 threshold curve, interpolated) + `groups/<group>_f1` |
| `force_supervision` | forces (root frame, bw) | Huber + noncontact L1 + net force/torque vs kindyn GT, `force_confidence` row weights |
| `motion_supervision` | pelvis 12-dim twist | Huber on standardized GT (pinned `standardize` table), train-only outlier cut |
| `pose_supervision` | 125 local MHR q channels + shape/bones/scale | Huber vs `mhr_1.npz` (kindyn refit as MHR), BetterHuman q-space |
| `keypoint_supervision` | camera-frame keypoints/vertices, world vel/acc | Huber vs `mhr_sup_1.npz`; `cam_rail`/`rot_rail` trust regions vs the frozen readout |
| `smplx_supervision` | the SMPL-X head (kindyn SMPL-X GT lifted to the camera) | `kp2d` Huber in crop-normalized units of the FULL-FRAME projection, `kp3d` Huber pelvis-relative metres, `orient`/`pose` MSE on raw 6D vs the GT matrix columns, `betas` MSE, `cam` Huber on the CLIFF proxy (GT pelvis inverted into `(s,tx,ty)`). Metrics (eval only, group `pose`) follow WHAM/GVHMR code line by line on the 22 body joints + 10475 vertices, flat hands: frames aligned by the MEAN OF THE TWO HIPS (`pelvis_idxs [1,2]`), `mpjpe`/`pa_mpjpe`/`pve` mm, `accel` = second finite difference of the aligned joints divided by the REAL dt² (m/s² — WHAM's `accel * 30**2` on 30 fps footage, fps-exact here), frame-weighted reduction. `metric_pose/*` |
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
≥ 1 modality, `contact.source` ∈ {tokens, pose_token} and the `contact` modality needs tokens,
pose writers need a pose loss (pose / keypoint / smplx), `smplx` excludes the MHR consumers,
camera fine-tune needs keypoints, motion loss needs the motion modality, physics xor
force_supervision, `contact_supervision.criterion` ∈ {bce, focal} (+ positive `neg_weight`/`pos_weight[6]`, `transition_tolerance` ≥ 0), monitor tag format
`loss_test/total` | `metric_<group>/<name>` and max/min by suffix). Which GT signal
groups a dataset loads is **derived** from the enabled losses (`signal_needs`), never
configured. One dataset yaml: `configs/datasets/climbing_videos.yaml` (root, contact_level).

Sections: `model.{contact,force,motion,cross_modal_temporal,pose_temporal,finetune_*,smplx}`,
`mhr_body` (BetterHuman archive), `data.{datasets,embedding_cache,frames_per_batch,
num_workers,seed,clip.{frames,stride,jitter},eval_max_frames}`, the nine loss sections
above, `optim.{lr,weight_decay,epochs,warmup_steps,lr_min,grad_clip,grad_clip_per_group,betas,decay_1d,ema,head_lr_scale}`,
`output.{dir,exp_name,log_freq,save_freq,monitor,frozen_metrics}`.

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
  SMPL-X GT (`smplx` group): `q (211) = [pelvis_world, root quat xyzw, 51 joint quats]` of
  BetterHuman's `SMPLX(use_face=False, use_hands=True, num_betas=10)` — `q[:3]` IS
  `joints_world[0]`; the stored classic `transl` differs by the shape-dependent `J0(β)`
  only. Served as `smplx_joints_world (22,3)`, `smplx_root_rot`, `smplx_body_rot (21,3,3)`,
  `smplx_betas (10)`, `smplx_valid`. The stored axis-angles are off the principal branch —
  never regress them directly.
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
