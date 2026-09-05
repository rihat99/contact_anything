# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Fork of **SAM 3D Body** (Meta, single-image 3D human mesh recovery) extended with
**per-joint contact**, an optional **3D contact-force** head and a from-scratch **SMPL-X pose**
head, trained on climbing **video clips** from the ClimbingVideos corpus. The base model is
frozen; only the appended token blocks, the post-decoder RoPE temporal transformer and the
heads train.

**2026-09-05 — simplified.** Everything that no run uses today was removed (the MHR-writing
pose path with its split-head fine-tune and MHR pseudo-GT, the motion head and its four losses,
the physics / Newton force losses, the ray depth priors and pose-token inputs, the velocity /
smoothness / matching losses, the block variants, the contact-loss knobs, the token-masking
and 2D-keypoint inputs). Pre-cleanup code is in git (`dev` before this commit) and under
`/data3/rikhat.akizhanov/trash/simplify_20260905/`; the earlier rounds' write-ups moved to
`docs/history/`. There is **no backwards compatibility** with older configs or checkpoints.

**2026-09-05 — the two-stage pipeline (`docs/refiner.md`).** The pose is no longer improved with
image-side temporal models. **Stage 1** (`configs/stage1.yaml`) is the per-frame SMPL-X + CLIFF
model alone (no contact tokens, no temporal block), trained once and frozen. **Stage 2**
(`configs/stage2.yaml`) appends the contact tokens to the frozen decoder and runs the
**world-space temporal refiner** (`model/refiner.py`) behind the frozen stage-1 body: depth
smoothing, world lift with the camera extrinsics, a world-independent per-frame token, a local
RoPE transformer, and zero-init heads for the pose offset, contact, motion and forces. Contact is
trained in stage 2 only. Tests: `tests/test_refiner.py` (CPU).

## Environment

```
CONDA=/data3/rikhat.akizhanov/miniconda3/envs/sam3d          # python + torchrun live in $CONDA/bin
PYTHON=$CONDA/bin/python                                     # neither is on the login PATH
```
Run everything from the repo root. Scripts insert the repo root at the head of `sys.path`
before importing (`scripts/train.py` would otherwise shadow the `train/` package).
Corpus (read-only): `/data3/rikhat.akizhanov/better/data/ClimbingVideos`. The SMPL-X head
needs the sibling `../BetterHuman` checkout (`better_human` + `models/smplx/SMPLX_NEUTRAL.npz`).

## Key commands

```bash
# Train (resume: --resume auto | --resume path/to/last.pth; --limit-scenes N for smoke runs).
# Rank 0's console goes to output/logs/<run>.log by itself; redirect anything else you launch
# into output/logs/ too — never write files into output/ itself.
CUDA_VISIBLE_DEVICES=0,1 $PYTHON -m torch.distributed.run --standalone --nproc-per-node=2 \
    scripts/train.py --config configs/baseline.yaml          # or $CONDA/bin/torchrun
$PYTHON scripts/train.py --config configs/static_ray.yaml    # single GPU

# Frozen SAM3D-as-SMPL-X baseline on the SAME test protocol -> the `frozen` tensorboard run
# (output.frozen_metrics; recompute whenever eval_max_frames / stride / dataset change)
$PYTHON scripts/eval_frozen_smplx.py --config configs/baseline.yaml --out output/frozen_sam3d_smplx.json

# Evaluate on the annotated corpus test split (full-scene protocol; --checkpoint none = untrained,
# which for a stage-2 config IS "stage 1 + depth smoothing"; --json writes a frozen_metrics file)
$PYTHON scripts/evaluate.py --config configs/baseline.yaml --checkpoint output/<run>/best.pth

# Two-stage pipeline: stage 1 (per-frame body), its diagnostics, then stage 2 (refiner)
CUDA_VISIBLE_DEVICES=0,5 $CONDA/bin/torchrun --standalone --nproc-per-node=2 scripts/train.py --config configs/stage1.yaml
$PYTHON scripts/dump_stage1.py --config configs/stage1.yaml --checkpoint output/<stage1>/best.pth --split train --scenes 150
$PYTHON scripts/dump_stage1.py --config configs/stage1.yaml --checkpoint output/<stage1>/best.pth --split test
$PYTHON scripts/analyze_stage1.py --train output/<stage1>/dump_train --test output/<stage1>/dump_test
#   -> train/test gap, depth_smooth_sec sweep, motion_supervision.scale numbers; then set
#   model.smplx.checkpoint in configs/stage2.yaml and train it like stage 1
$PYTHON -m pytest tests/ -q                                  # refiner unit tests (CPU, ~3 min)

# Renders (mp4 per test scene; shard scenes over ranks with torchrun)
$PYTHON scripts/render_video.py --config configs/baseline.yaml --checkpoint output/<run>/best.pth \
    --scenes 5 --out output/<run>/render_contact --overlay-labels --gt-panel --scale 0.5
$PYTHON scripts/render_smplx_video.py --config configs/baseline.yaml --checkpoint output/<run>/best.pth \
    --scenes 5 --out output/<run>/render_pose             # GT | frozen MHR | SMPL-X head panels

# Results viewer (docs/viewer.md): dump a run's whole-scene test predictions once, then serve
# every run's predicted | GT | frozen SMPL-X bodies plus contact markers and force arrows
# (predicted and GT) in viser (port 8082 is the BVR viewer's)
$PYTHON scripts/predict_test.py --config configs/baseline.yaml --checkpoint output/<run>/best.pth
CUDA_VISIBLE_DEVICES=5 $PYTHON scripts/view_results.py --port 8090

# Inference on BetterVideoReconstruction out-trees (contacts + forces, no labels needed)
$PYTHON scripts/predict_reconstruction.py --config configs/baseline.yaml \
    --checkpoint output/<run>/best.pth --out-root ../BetterVideoReconstruction/out --videos <videos>

# Data preparation (scripts/data/)
$PYTHON scripts/data/extract_frames.py                 # corpus frames/ JPEG tree
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/data/precompute_embeddings.py --split all --shard-index 0 --num-shards 4
```

## Layout

| Path | Purpose |
|---|---|
| `model/sam_3d_body/` | Vendored SAM 3D Body fork. Our additions are delimited by `# --- <name> hook ---` comments: the extra-token-block hook (append learned blocks behind the asymmetric mask, per-layer update callbacks, expose the final sequence) and the efficiency hooks (precomputed embeddings, `backbone_no_grad`, `detach_interm_preds`). |
| `model/wrapper.py` | `SAM3DBodyWrapper`: builds / freezes / eval-pins the base; `forward(img|embedding, geometry, blocks)` → final tokens, block bounds, the frozen MHR readout. |
| `model/tokens.py` `rope.py` `heads.py` | `LearnedTokenBlock` (token embeddings + anchored posemb/feat update), `CrossModalRopeModule` (the temporal brick), `ContactHead` / `ForceHead` (per-token FFNs), `SmplxHead`. |
| `model/refiner.py` | `TemporalRefiner` (stage 2): depth smoothing → world lift → world-independent token → RoPE transformer → zero-init pose / contact / motion / force heads → FK back into every camera; plus the masked time-series helpers (`gaussian_smooth`, `time_derivative`, `angular_velocity`). |
| `model/network.py` `build.py` | `ContactAnything` composes the above; `build_model(cfg, device)` maps the yaml sections onto it and applies `model.smplx.checkpoint` / `frozen`. |
| `model/loss/` | One `Loss` interface (`__init__.py`) and one file per term: `contact` (BCE), `force`, `smplx` (+ every pose metric), `motion` (refiner velocities / accelerations). |
| `data/` | `base.py` = `ClipDataset` ABC (windowing, jitter, full-scene eval) **and the frame schema** (module docstring); `climbing_videos/` (`scene.py` DB + labels, `kindyn.py` forces + SMPL-X GT, `dataset.py`); `reconstruction.py` (label-free BVR out-trees); `collate.py`, `loaders.py`, `transforms.py`. |
| `train/` | `config.py` (schema = `configs/base.yaml`, cross-key checks, `signal_needs`), `trainer.py` (DDP-exact weighted means, EMA, per-module clipping, per-step warm-up + cosine), `checkpoint.py` (trainable-only, strict), `logger.py` (tensorboard + `tee_output`), `predict.py` (`load_model`). |
| `utils/` | `geometry.py` (camera parametrizations, projection, world lift), `gvhmr_metrics.py`, `metrics.py`, `distributed.py`. |
| `scripts/` | Thin CLIs (above); `_render_common.py` shares the scene / clip plumbing and the drawing helpers; `dump_stage1.py` + `analyze_stage1.py` are the stage-1 diagnostics. |
| `tests/` | `test_refiner.py`: world-frame independence, identity at init, receptive-field locality, gradient flow (CPU, BetterHuman body). |
| `viewer/` | viser results viewer (`scripts/view_results.py`, `docs/viewer.md`). |
| `configs/` | `base.yaml` (the schema, every key with its default), `stage1.yaml`, `stage2.yaml`, `baseline.yaml`, `static_ray.yaml`, `datasets/*.yaml`. |
| `docs/` | `refiner.md` (the two-stage pipeline: design + results), `results.md` (every recorded number, incl. the trashed runs), `viewer.md`, `history/` (the 2026-09-03/05 round write-ups; their code is gone). |
| `output/` | Run directories `<exp_name>_<stamp>/` (`best.pth`, `last.pth`, `config.yaml`, `tensorboard/`), the frozen-baseline jsons, `logs/`. |

## Architecture

1. **Backbone** DINOv3-H (bf16, frozen) → `[B,1280,32,32]`; with `data.embedding_cache` the
   loader emits the cached embedding and the backbone is skipped (frame JPEGs not decoded).
2. **Promptable decoder** (frozen, dim 1024) with the pose token at index 0. Our
   `LearnedTokenBlock`s (contact 6, force 6; anchored at the MHR70 keypoints of the six kindyn
   groups `[62,41,15,18,17,20]`) are appended behind an **asymmetric mask**: original tokens
   never attend the appended blocks (the blocks attend everything), so the frozen pose/MHR
   readout has an exactly-zero Jacobian w.r.t. every trainable parameter. After every
   intermediate layer each anchored token receives the posemb of its keypoint's interm 2D
   position and the backbone features grid-sampled there.
3. **Cross-modal temporal** (`model.cross_modal_temporal`) — THE post-decoder mixing brick:
   one RoPE transformer of pre-LN residual blocks (zero-init output projections = exact
   identity at init) over the concatenated modality blocks (≥ 1 of pose / contact / force,
   canonical order) across a clip's frames. RoPE position = real elapsed seconds ×
   `time_scale`, identical for all tokens of a frame (within-frame mixing is the offset-0
   diagonal); a hard `window` in seconds (receptive field +`window` per layer) and
   `frame_valid` masking; a learned slot embedding tells the tokens apart. Listing `pose`
   writes the token the SMPL-X head reads.
4. **Heads** (`model/heads.py`): contact `[B,6]` logits and force `[B,6,3]` (body-weight
   units, body-root frame, zero-init) are one shared FFN per token block. The **SMPL-X head**
   reads the FINAL pose token with two from-scratch FFNs of SAM3D's own head shape
   (`C→C→D`, zero-init last linear, residual on a fixed mean) and regresses the corpus SMPL-X
   body in the CAMERA frame under BetterHuman's `q` convention (root = pelvis pose): root + 21
   body 6D rotations (+ 30 finger joints with `hands`), 10 betas, and the pelvis position as
   either the CLIFF crop weak-perspective `(s,tx,ty)` lifted with the crop box + true focal
   (`camera: cliff`) or the pelvis ray `(x/z, y/z, log z)` about a fixed 3.5 m (`camera: ray`,
   crop-free). BetterHuman's FK and the full-frame projection run inside the head, so
   `out["smplx"]` carries params, `joints_cam`, `kp2d_full` px and `kp2d_crop`. The SMPL-X body
   IS the pose output; `out["mhr"]` stays the frozen readout (used by the renders' frozen panel
   and by `predict_reconstruction.py`'s anchor pixels). The frozen model's own SMPL-X numbers
   come from the corpus refit `features/sam3d/<shard>/<scene>/smplx_params.npz`, scored offline
   by `scripts/eval_frozen_smplx.py` and drawn as the `frozen` tensorboard run.
5. **Temporal refiner** (`model.refiner`, stage 2; `docs/refiner.md`) — behind the frozen
   per-frame body. Pelvis log-depth Gaussian smoothing in camera coordinates
   (`depth_smooth_sec`, bearing kept) → world lift with `cam_from_world`, clip-mean betas →
   per-frame token = root-frame joint positions + body-frame root linear/angular velocity +
   frame spacing + betas + projected pose token + projected contact tokens (`pose_token`,
   `pose_token_dim`, `contact_token_dim`) → `CrossModalRopeModule` with ONE slot (`dim`,
   `num_layers`, `num_heads`, `window` seconds per layer) → zero-init heads listed in
   `outputs`: `pose` (6D deltas right-multiplied onto the root and the 21 joints + a body-frame
   root shift), `contact` (replaces the decoder contact head), `motion` (body-frame vel / acc
   of the 22 joints + root angular vel / acc, `out["motion"]` with its `frame`), `force`
   (replaces decoder force tokens). FK in the world, mapped back into every camera → the
   output keeps the SmplxHead layout. **Frame independence is a design rule**: nothing that
   enters or leaves the transformer refers to the world frame (tested). At init the refiner is
   exactly stage 1 + depth smoothing.

### Losses (`model/loss/`, all on one interface)

Each loss returns `LossResult(terms, scalars, stats)`: `terms[name] = (numerator, mass)`
additive pairs (term weights applied INSIDE the loss, numerators graph-connected even at
zero mass), `stats` a float64 additive vector for eval, `metrics(stats)` the reported
numbers. The trainer all-reduces masses once → exact global weighted means under DDP; eval
sums `stats` across batches and ranks. Tensorboard sections: `optim/*`, `loss_train/total`
+ `loss_train/<loss>.<term>`, the same under `loss_test/`, and `metric_<group>/<metric>`
(`Loss.metric_group` = the loss name except `smplx` → `pose`). `output.monitor` names one tag.

| section | supervises | with |
|---|---|---|
| `contact_supervision` | six-group contact logits | confidence-weighted BCE; metrics f1 / precision / recall / iou (thr 0.5), `precision_at_r90`, per-group f1 |
| `force_supervision` | forces (root frame, bw) | Huber on in-contact rows + noncontact L1 + net force / torque vs kindyn GT, `force_confidence` row weights, `group_weights`; metrics mae, noncontact_mag |
| `motion_supervision` | the refiner's `motion` output | Huber on `vel` / `acc` / `ang_vel` / `ang_acc` divided by `scale` (GT RMS), vs kindyn world joints / root finite-differenced with `label_smooth_sec` Gaussian smoothing and rotated into the predicted body frame; metrics `<q>_rmse`, `<q>_pearson` |
| `smplx_supervision` | the SMPL-X head (or the refined body) | `kp2d` Huber on the full-frame projection (crop-normalized or bearing units), `kp3d` pelvis-relative, 6D MSE `orient` / `pose` / `hand_pose`, `betas` MSE, `cam` (CLIFF proxy) or the crop-free anchors `depth` / `bearing` / `pelvis`. Metrics (`metric_pose/*`, WHAM/GVHMR protocol): mpjpe / pa_mpjpe / pve / accel, pelvis_err / depth_err / depth_bias, dlogz_pred / gt / err, the camera-lifted GVHMR globals `lifted_wa_mpjpe100` / `lifted_w_mpjpe100` / `lifted_rte` / `lifted_jitter` + `gt_jitter`, hand_mpjpe / hand_pa_mpjpe |

### Invariants (do not break)

- **Freeze boundary = the wrapper.** Everything under `model.wrapper` is frozen and
  eval-pinned (`wrapper.train()` re-pins); everything else in `ContactAnything` trains. The
  checkpoint stores trainable tensors only and hard-fails with a name/shape diff on mismatch.
- **Mask invariant**: original tokens never attend appended blocks; the frozen MHR readout
  never changes. The pose output is the SMPL-X head, which reads the (mixed) pose token, or —
  under `model.refiner` — the refined body derived from it.
- **Refiner frame independence**: inputs and outputs of the temporal transformer are
  root-/body-frame quantities only (`tests/test_refiner.py::test_world_frame_independence`).
- A frozen SMPL-X head (`model.smplx.frozen`) is NOT in the run's checkpoints — the stage-1
  path in `model.smplx.checkpoint` is re-read on load, so keep that run directory.
- **fp16**: backbone bf16, decoder/MHR heads fp32 (MHR sparse ops are fp16-incompatible).
- **Frozen model noise floor** ~5e-4 px run-to-run; warm up before bitwise compares;
  building SAM-3 elsewhere flips global `allow_tf32`.

## Configuration

`configs/base.yaml` **is** the schema: every allowed key with its default, commented.
Experiment yamls set `base: configs/base.yaml` and override; unknown keys hard-error with the
dotted path; `train/config.py::validate` holds the cross-key checks (six distinct MHR70 anchors
per token block, modalities ⊆ the enabled blocks, `pose` listed needs `model.smplx`, each loss
needs its branch, `hand_pose` needs `hands`, `cam` is CLIFF-only, monitor tag format and
max/min by suffix). Which GT signal groups a dataset loads (`forces` / `smplx`) is **derived**
from the enabled losses (`signal_needs`), never configured. Dataset yamls:
`configs/datasets/climbing_videos.yaml` (root, contact_level, `camera: all | static | moving`
= the DB's `static_camera` flag) and `climbing_videos_static.yaml` (the 113 / 16-scene static
subset).

Sections: `model.{contact, force, cross_modal_temporal, smplx (+ checkpoint, frozen), refiner}`,
`data.{datasets, embedding_cache, frames_per_batch, num_workers, seed, clip.{frames, stride,
jitter}, eval_max_frames}`, the four loss sections (`motion_supervision` needs a refiner
`motion` output; every refiner output needs its loss enabled — DDP has no unused-parameter
tolerance), `optim.{lr, weight_decay, epochs, warmup_steps,
lr_min, grad_clip, betas, ema}` (no decay on 1-d params and per-module clipping are fixed
behaviour), `output.{dir, exp_name, log_freq, save_freq, monitor, frozen_metrics}`.

## Data (ClimbingVideos corpus, read directly)

- `scenes/scenes.db` curated split: 864 train / 108 test scenes (`static_camera` flag: 113 / 16
  static); test scenes need `annotation.npz` (manual labels on 14 observable joints; the other
  eight are fixed non-contact). Train labels are automatic (`contacts_{level}.npz`, 52 → 22
  joint fold).
- **Six groups everywhere, fixed order**: `left_hand, right_hand, left_foot (toe),
  right_foot, left_ankle (heel), right_ankle`; `KINDYN_GROUP_NAMES` in `model/loss`.
  Test-label fold: a known positive wins under partial annotation, a known negative needs
  all source joints annotated. Video labels are motion-gated "stable contact".
- `kindyn_1.npz`: GT forces on 35 named frames (world newtons) folded into the six groups by
  parent joint and converted to body-weight units in the body-root frame; per-frame
  `force_confidence`. SMPL-X GT (`smplx` group): `q (211) = [pelvis_world, root quat xyzw,
  51 joint quats]` of BetterHuman's `SMPLX(use_face=False, use_hands=True, num_betas=10)` —
  `q[:3]` IS `joints_world[0]`; served as `smplx_joints_world (52,3)`, `smplx_root_rot`,
  `smplx_body_rot (21,3,3)`, `smplx_hand_rot (30,3,3)`, `smplx_betas (10)`, `smplx_valid`.
  The stored axis-angles are off the principal branch — never regress them directly.
- Per-frame metric extrinsics (`cam_from_world`, OpenCV) on every scene.
- Clips: `data.clip.frames` × stride (`auto` = `max(1, round(fps/25))`), tiled with
  stateless per-epoch jitter; **eval = ONE clip per (scene, person)**, the longest valid run,
  capped at `data.eval_max_frames`, batch = 1 clip. Batches are flat `[B_clips·T, ...]` with
  `seq_len`, `frame_pos_sec`, `frame_valid`. Full frame/batch schema: `data/base.py`.
- Embedding cache: `features/embedding/` bf16 `[1280,32,32]` per (scene, oid, frame); missing
  files hard-error; bit-exact only at the backbone batch shape it was built with (35).

## Results so far

`docs/results.md` holds every recorded number. Headlines (108-scene test, one clip per
person, 120-frame cap): frozen SAM3D refit 61.1 mm MPJPE / 44.1 PA / accel 11.9; the
`baseline.yaml` recipe (as run `hands`, 5 epochs) 61.0 / 42.0 / F1 0.919; the per-frame SMPL-X
probe 57.6 / 38.9 (no temporal block). On the 16-scene static subset the lifted-trajectory
jitter is 126 for the frozen model vs a GT floor of 6.35; the best non-smoothing run reached 52.
The temporal block over image tokens never learned to denoise the per-frame pose (see
`docs/history/`), which is why the pipeline pivoted to the two-stage refiner — its results live
in `docs/refiner.md`.

## Conventions

- Never `rm`: move to `/data3/rikhat.akizhanov/trash/<name>_<date>/`. Commit/push only on
  explicit instruction. Shared GPU box: never touch other users' processes.
- `output/` holds run directories and the frozen-baseline jsons only; every console transcript
  or launch log goes to `output/logs/` (the train/evaluate CLIs tee themselves there).
- Skepticism: log raw metrics and trajectories; the user owns verdicts. No validation split
  (train on all train scenes, evaluate on test only).
- Keep the codebase pruned: no `.get()` fallbacks on schema-guaranteed keys, no history
  narration in comments, no options nobody runs.
