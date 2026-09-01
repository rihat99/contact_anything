# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fork of **SAM 3D Body** (Meta) — single-image 3D human mesh recovery — extended with
**per-joint contact**, **3D contact-force**, **root-motion** and **pose** heads trained on
climbing video clips. The base model is frozen; only contact-, force-, motion-,
cross_modal- and pose_temporal-named parameters train, plus the optional split-head
pose/camera fine-tune copies.

Contact is predicted **per-joint** (SMPL-X body-22, four climbing extremities, or the six
kindyn force groups) on **video clips**; a single post-decoder RoPE transformer
(`cross_modal_temporal`) mixes the modalities across a clip's frames and provably does not
change the frozen model's pose (MHR) predictions unless `pose` is an explicitly listed
modality. The force head regresses one 3D vector per kindyn contact group, supervised
against the corpus's own kindyn GT forces.

**2026-09-01 — new era.** Per-vertex contact prediction, the DAMON / ClimbingImages /
LEMON / RICH still-image datasets, the sliding-window temporal module, the embedding
noise+CutMix augment, the motion roll-out and motion-consistency losses, the jerk/snap
pose-smoothness loss, the `docs/` tree, the `viewer/`, `tools/`, `legacy/`, `plan/` and
`conversion/` trees and the whole test suite were **deleted**. There is no backwards
compatibility with older configs or checkpoints. Everything removed is recoverable from
the `main` branch (commit `3a94a9c`) and from
`/data3/rikhat.akizhanov/trash/contact_anything_dev_20260901/`; historical runs live in
`../contact_anything/output/old/`.

## Environment Setup

```bash
conda create -n sam_3d_body python=3.11 -y
conda activate sam_3d_body

pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines xtcocotools loguru optree fvcore black pycocotools tensorboard huggingface_hub

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
python scripts/train.py --config configs/allmod_rope_t60_gv.yaml
# Two-GPU DDP (`data.frames_per_batch` is per GPU):
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/train.py --config configs/allmod_rope_t60_gv.yaml

# Evaluate on the manually annotated corpus test split.
# Reports P/R/F1/F2/IoU, per-group metrics and a threshold curve; force runs add
# force MAE and per-group magnitudes split by predicted contact.
python scripts/evaluate.py --checkpoint output/<run>/best.pth \
  --config configs/allmod_rope_t60_gv.yaml --split test --threshold 0.3
# Without a trained force checkpoint, --warm-start builds the untrained force branch
# from the config's init_contact_checkpoint.

# Motion-head evaluation against the kindyn twist targets (+ trivial baselines)
python scripts/evaluate_motion.py --checkpoint output/<run>/best.pth \
  --config configs/allmod_rope_t60_gv.yaml

# Rendered corpus videos (contact disks + force arrows; shards over torchrun ranks)
python scripts/render_climbing_video_contacts.py --checkpoint output/<run>/best.pth \
  --config configs/allmod_rope_t60_gv.yaml --split test --overlay-labels
# Side-by-side frozen-vs-finetuned pose overlays
python scripts/render_climbing_pose_video.py --checkpoint output/<run>/best.pth \
  --config configs/allmod_rope_t60_gv.yaml

# Inference on BetterVideoReconstruction out-trees (contacts.npz / forces.npz per scene)
python scripts/predict_reconstruction.py --out-root <bvr-out> \
  --contact-checkpoint output/<run>/best.pth --config configs/allmod_rope_t60_gv.yaml

# Logging: tensorboard (wandb off by default)
tensorboard --logdir output/<run>/tensorboard/

# Data preparation
python scripts/extract_corpus_frames.py           # corpus frames/ JPEG tree (q95)
python scripts/convert_kindyn_to_mhr.py           # mhr_1.npz  — kindyn SMPL-X refit as MHR q
python scripts/precompute_mhr_supervision.py      # mhr_sup_1.npz — GT keypoints/vertices on the model's own rig
# bf16 frozen-backbone embedding cache (corpus features/embedding, ~0.8 TB; one
# process per GPU; data.embedding_cache: true consumes it and skips the backbone)
CUDA_VISIBLE_DEVICES=0 python scripts/precompute_embeddings.py --split all --shard-index 0 --num-shards 4
```

## Repository Layout

| Path | Purpose |
|---|---|
| `sam_3d_body/` | Vendored SAM 3D Body fork. Upstream code untouched; our additions are the contact/force/motion heads+tokens, `models/modules/temporal_rope.py` (pose token) + `models/modules/cross_modal_rope.py` (the all-modality block), and hooks delimited by `# --- <name> hook ---` / efficiency-flag comments. |
| `contact/` | Our library: `config.py` (yaml + `base:` include + strict validation), `model.py` (build/freeze/eval-pin), `targets.py`, `losses.py`, `metrics.py`, `engine.py` (shared forward), `checkpoint.py` (schema v2), `tracking.py` (TB+wandb), `root_world.py` (predicted body root → metric world + SO(3) helpers), the loss modules (`force_supervision`, `motion_supervision`, `pose_supervision`, `keypoint_supervision`, `contact_consistency`, `force_consistency`), `data/` (corpus loader, collate, splits), `physics/` (`adapter.py` MHR bridge + `loss.py` RNEA residual). |
| `scripts/` | Thin CLIs: train, evaluate, evaluate_motion, predict_reconstruction, the two renderers, extract_corpus_frames, convert_kindyn_to_mhr, precompute_mhr_supervision, precompute_embeddings. |
| `configs/` | `base.yaml` (all defaults, commented) + experiment overrides; `configs/datasets/*.yaml` = dataset paths/options. `allmod_rope_t60_gv.yaml` is the current experiment, self-contained on `base.yaml`. |
| `output/` | Training runs (gitignored). Historical runs live in `../contact_anything/output/old/`. |

There is **no test suite** — deliberately removed 2026-09-01.

## Architecture

1. **Backbone** — DINOv3-H (bf16, frozen) → image embeddings `[B,1280,32,32]`. With
   `data.embedding_cache: true` these are read from disk and the backbone never runs.
2. **Promptable decoder** (6 layers, dim 1024) with typed tokens. Contact adds keypoint-anchored
   tokens (configurable MHR70 anchor indices, `model.contact_head.contact_keypoint_indices`;
   default = first 21; the kindyn_6 config uses `[62,41,15,18,17,20]`) + `num_global_tokens`
   extra tokens.
   An **asymmetric attention mask** stops all original tokens from attending to contact tokens —
   pose/keypoint outputs are unaffected by anything contact-side. The optional **force tokens**
   (`model.force_head`; inheriting the contact anchors, or an explicit
   `force_keypoint_indices` list — force-only builds with no contact tokens/head at all are legal)
   are appended *after* the contact tokens, then the **motion tokens**. Among the appended blocks
   `model.extra_token_attention` picks the regime: `mutual` (default) lets contact/force/motion
   fully inter-attend; `causal` extends the mask so no earlier block attends a later one.
   Original tokens attend none of the appended blocks under either regime, so pose/MHR stay
   isolated either way.
3. **Contact head** — `head_contact` is an `nn.ModuleDict` keyed by target (only `joint`
   exists). Pooled modes (`concat`/`attention`) emit joint logits; `pool_mode: per_token`
   applies one shared classifier independently to each token, so the token count must equal
   the output dimension (kindyn_6 uses six tokens → `[B,6]`).
   Output: `out["contact"]["joint_logits"]`.
4. **Temporal mixing** — ONE post-decoder block, `model.cross_modal_temporal` (item 7).
   Batches are homogeneous-T flattened clips (`[B_clips*T, ...]` +
   `seq_len`/`frame_pos_sec`/`frame_valid`).
5. **Force head** (`model.force_head`, optional) — K force tokens → `head_force` (zero-init)
   regressing `out["force"]["joint_forces"] [B,K,3]`, dimensionless (units of body weight);
   its temporal window comes from listing `force` in `cross_modal_temporal`.
   Two mutually exclusive supervision signals:
   - **Supervised kindyn forces** (`force_supervision.enabled`, six explicit anchors
     `[62,41,15,18,17,20]` = kindyn groups `LH,RH,LF(toe),RF(toe),LA(heel),RA(heel)`):
     `contact/force_supervision.py` trains against the corpus `kindyn_1.npz` GT forces —
     body-weight units, body-root frame, no extrinsics anywhere in the objective.
     This is the live path.
   - **Physics** (`physics.enabled`, K=4 inheriting the extremity contact anchors, order
     `left_hand,right_hand,left_foot,right_foot`): no labels — `contact/physics/` supervises.
     `adapter.py` (`MHRAdapter`) maps frozen per-frame MHR params + dataset camera extrinsics onto
     a BetterHuman **MHR** body and a world-frame `q` trajectory; `loss.py` (`PhysicsLoss`)
     smooths `q`, finite-differences to v/a, runs **RNEA** with the predicted forces as external
     wrenches, and minimises the 6D root-wrench residual (plus contact-gated / smoothness / L2
     regularisers). Needs the sibling `../BetterRobot` / `../BetterHuman` checkouts and the MHR
     archive (or `$BETTERHUMAN_MODELS_DIR`); no current experiment yaml enables it.
6. **Motion head** (`model.motion_head`, optional) — anchored motion tokens regressing
   standardized root-frame vel/acc per slot, `out["motion"]["joint_motion"] [B,K,6|12]`
   (12 with `motion_supervision.angular`: + the root twist's angular vel/acc). A temporal
   window is mandatory in practice (a per-frame head cannot represent a derivative), so a
   motion-supervised config MUST list `motion` in `cross_modal_temporal`. Supervision:
   `motion_supervision` (kindyn twist targets, σ0.12 s label smoothing, pelvis-only for
   `gravity_view`).
7. **Cross-modal temporal** (`model.cross_modal_temporal`, optional) — THE post-decoder
   mixing brick: ONE zero-gated **RoPE temporal transformer**
   (`sam_3d_body/models/modules/cross_modal_rope.py`, sharing `_RopeBlock` with
   `temporal_rope.py`) over the CONCATENATION of the chosen modality token blocks
   (`modalities` ⊆ {pose, contact, force, motion}, ≥ 2, always reordered into the canonical
   sequence order pose < contact < force < motion). Per clip the attention sequence is
   `T` frames × `K` tokens, frame-major; the RoPE position of a token is its frame's real
   elapsed seconds (`frame_pos_sec × time_scale`), IDENTICAL for every token of the frame —
   so within-frame pairs attend un-rotated (within-frame cross-modal attention is the
   `dt = 0` diagonal) and across-frame pairs see only relative time. Token identity comes
   from a learned `[K, dim]` slot embedding added to the LayerNormed input INSIDE each
   block's gated attention branch (never to the residual stream), so the block is still an
   exact identity at init. Frame-level masking: `max_rel_sec` seconds window +
   `frame_valid` (an invalid frame is hidden from every other frame, its own stays visible).
   Deliberately relaxes D1 among the participants; listing `pose` writes the pose token
   (final MHR recomputed — needs `pose_supervision`). Runs BEFORE `pose_temporal`.
8. **Pose temporal** (`model.pose_temporal`, optional) — a deliberate exception to
   the frozen-pose rule: a zero-gated RoPE temporal module on the pose token (index 0) alone,
   run AFTER `cross_modal_temporal`; the FINAL MHR output is recomputed from the updated
   token (interm preds and all other token blocks see the untouched one). Redundant when
   `cross_modal_temporal` already lists `pose`. Supervised by `pose_supervision` against
   kindyn-MHR pseudo-GT (`scripts/convert_kindyn_to_mhr.py` writes `mhr_1.npz` per scene —
   the kindyn SMPL-X trajectory refit as a world-frame MHR `q`, ~0.5 cm joint residual),
   compared in q space (125 local channels; the free-flyer root is never supervised).
   `type: rope` is the only value: the long-sequence **RoPE temporal transformer**
   (`sam_3d_body/models/modules/temporal_rope.py`, GVHMR-informed): pre-LN blocks at the
   native decoder dim (no bottleneck), per-head RoPE on q/k with **time-valued positions**
   (`frame_pos_sec × time_scale` — exact under the corpus's variable fps), zero-init gates
   (identity at init), bidirectional only, and a seconds-based local window
   (`max_rel_sec` = training clip span) so a T=60-trained model runs
   **single-pass on whole scenes** without exposing untrained relative offsets.
   Eval protocol: `data.sequence.eval_full_scenes` emits ONE clip per test
   (scene, person) — the longest valid run, `eval_max_frames`-capped
   (~0.06-0.1 GiB/frame no-grad).
9. **Pose/camera-head fine-tune** (`train.finetune_pose_head` / `train.finetune_camera_head`) —
   SPLIT-HEAD: the ORIGINAL heads stay frozen and keep producing every in-decoder
   intermediate prediction (whose per-layer keypoint-token refresh feeds back into the frozen
   decoder — training the shared head perturbed the frozen model layer by layer, the earlier
   divergence mechanism), while trainable COPIES of the projection FFNs
   (`head_pose_ft_proj` / `head_camera_ft_proj`, deepcopy-initialized so init behavior is
   exactly frozen) are applied to the FINAL pose token only, via the final-readout recompute.
   Copies form their own optimizer param group at `optim.lr × train.pose_head_lr_scale`.
   Pose finetune requires `pose_supervision` or `keypoint_supervision`; camera finetune
   requires `keypoint_supervision` (kp2d — the only camera-constraining loss). The checkpoint
   carries the copy weights (`pose_head_finetune` {enabled, split} / `camera_head_finetune`
   in the arch signature). Side effect of the split: `pred_cam_t_frozen`/`global_rot_frozen`
   (the rail anchors) are genuinely the FROZEN model's outputs even under head finetuning.

### Invariants (do not break)

- **Freeze filter is name-based**: only params whose dotted name contains `"contact"`, `"force"`,
  `"motion"`, `"cross_modal"` **or** `"pose_temporal"` train (tokens, heads,
  posemb/feat linears, `force_*`, `motion_*`, `cross_modal_temporal`,
  `pose_temporal`). Any new trainable module must carry one of those in its
  attribute path. The pose outputs move only via `pose_temporal`, the `pose` modality of
  `cross_modal_temporal`, or the explicit `train.finetune_pose_head`/`finetune_camera_head`
  flags (which build trainable head COPIES `head_pose_ft_proj`/`head_camera_ft_proj` outside
  the name filter — train.py adds them to the saved-name set).
- **Mask invariant**: inside the decoder the original tokens NEVER attend the appended blocks —
  pose/MHR outputs have an exactly-zero Jacobian w.r.t. every contact/force/motion param under
  either mask regime. Under `extra_token_attention: causal` additionally no earlier
  appended block attends a later one (contact ⊥ {force, motion}, force ⊥ motion), so contact
  outputs have an exactly-zero Jacobian w.r.t. every force **and motion** param (D1); `mutual`
  (the default) deliberately opens full inter-attention among the appended blocks (D1 gone
  between them — incompatible with `train.freeze_contact`, and captured in the arch signature).
  The POST-decoder block `cross_modal_temporal` relaxes D1 the same way **among its listed
  modalities**; the frozen pose/MHR outputs remain isolated unless `pose` is a listed
  (written) modality.
- **Frozen modules are eval-pinned** (`contact/model.py::pin_frozen_eval`): `model.train(True)`
  re-pins backbone/decoder/MHR+camera heads to eval (the backbone has DROP_PATH_RATE 0.1 — train
  mode would make it stochastic). The toggled set is **requires_grad-derived** at call time (not a
  name list): a fully-trainable subtree follows the mode in full (incl. its param-less dropout),
  a fully-frozen subtree (e.g. a contact head frozen by `train.freeze_contact`) stays eval, a
  mixed container is descended into.
- **Physics-loss gradient isolation (regime (a))**: the physics loss consumes the frozen model's
  outputs (`out["mhr"]`, camera extrinsics) and the contact probs (`out["contact"]["joint_probs"]`)
  all **detached** — gradients reach **force** params only, and physics never trains the frozen
  base. This isolation from the **contact** head is exact only in regime (a) (`train.freeze_contact`,
  contact frozen). In regime (b) (contact trainable) force→contact attention leaks physics gradients
  into the trainable contact params (the vendored mask permits that direction); the trainer warns,
  and the detach-fix is deferred.
- `TRAIN.USE_FP16` stays as shipped (backbone bf16); decoder/MHR heads stay fp32 (MHR sparse ops
  are fp16-incompatible).

## Configuration

Experiments are small yamls with `base: configs/base.yaml` (deep-merge; unknown keys hard-error).
Key sections (see `configs/base.yaml` for full commented defaults):

| Section | Controls |
|---|---|
| `model.contact_head` | anchor indices, global tokens, pooling (`concat`/`attention`/`per_token`), MLP, grid sampling |
| `model.force_head` | force branch: enabled, `frame: local_world_aligned\|local\|root`, `force_keypoint_indices` (null = inherit contact anchors; explicit list enables force-only builds), MLP, `contact_gate` |
| `force_supervision` | supervised kindyn GT-force loss (exclusive with `physics`): `target_frame` (center\|all rows), `gt_frame` (root\|world), `units` (bw\|newtons), `confidence` (weight rows by kindyn force_confidence), Huber `force`/`huber_delta_bw`, `outlier_bw` cut, `noncontact` L1 |
| `physics` | RNEA loss: enabled, MHR `model_path`/`lod`, `gravity`, `min_frames`, `smoothing_kernel`, per-term `loss.*` weights (all dimensionless) |
| `model.motion_head` | motion tokens: explicit anchors; `anchored: false` = pure learned queries (no in-decoder keypoint-anchored update; the index list only names/counts slots). Needs `motion` listed in `cross_modal_temporal` to see across frames |
| `motion_supervision` | kindyn twist vel/acc loss: `joint_names`, `root_source` (kindyn\|mhr), `root_convention` (twist\|rotated_world\|gravity_view), `angular` (12-dim root target), `target_smooth_sec`, standardize `[K][2\|4][3]` |
| `model.cross_modal_temporal` | THE post-decoder mixing brick: ONE RoPE temporal transformer over the chosen modality blocks (`modalities` ≥ 2 of pose/contact/force/motion), `num_layers`/`num_heads`/`mlp_ratio`/`dropout`/`time_scale`/`max_rel_sec`. Cross-modality AND cross-frame; within-frame mixing is its `dt = 0` diagonal |
| `model.extra_token_attention` | decoder mask among the appended blocks: `mutual` (contact/force/motion inter-attend, default) \| `causal` (block-triangular) |
| `model.pose_temporal` / `pose_supervision` | pose branch: zero-gated pose-token RoPE temporal module (`type: rope` is the only value; `time_scale`/`max_rel_sec`) + kindyn-MHR pseudo-GT q-space Huber (`mhr_1.npz` via `scripts/convert_kindyn_to_mhr.py`); `loss.shape` = L2 on the 45 blendshapes vs the per-person GT `identity`; `loss.shape_rail`/`scale_rail` = L2 pinning the head's 45 blendshape / 28 bone-scale outputs to the FROZEN readout's stashed `shape_frozen`/`scale_frozen` (nothing else supervises them — the pose/keypoint losses are girth-blind) |
| `keypoint_supervision` | SAM3D-style stabilizers from the MHR-native GT (`mhr_sup_1.npz`): `kp2d` crop-normalized reprojection (constrains the camera), `kp3d` hips-relative camera-frame, `kp3d_abs` absolute metric anchor, `vert`/`vert_abs` mesh-vertex subsets; `kp_vel`/`kp_acc` WORLD-frame keypoint velocity/acceleration (central stencil over the clip's real elapsed seconds, predictions lifted with the GT extrinsics — loss-only use; camera-frame differences would bury body motion under camera egomotion), GT-acc outlier rows dropped; `cam_rail`/`rot_rail` trust regions vs the frozen `pred_cam_t`/`global_rot`; corpus loader flag `load_keypoints` |
| `contact_consistency` | predicted-contact-gated zero-velocity: world-frame velocity of the six extremity MHR70 keypoints ([62,41,15,18,17,20], kindyn_6 order) from the predicted pose + extrinsics, weighted by the predicted contact prob (`detach_gate`, default true → grad reaches the pose path only); requires the kindyn_6 joint target + a trainable pose path + T ≥ 3 |
| `force_consistency` | linear Newton residual in bw units (mass cancels): `a_root_world/g − gravity_world − R_root→world·Σf_pred` Huber, acceleration from the smoothed predicted world root (`contact/root_world.py` lifting), rotation = GT kindyn `motion_rot`; grad → pose + force head; `ramp.{start_epoch,epochs}` warm-up (unstable early); requires force_supervision (bw/root) + motion_supervision + T ≥ 3 |
| `train.freeze_contact` | regime (a): freeze contact, train force only (requires `model.init_contact_checkpoint`) |
| `train.finetune_pose_head` / `finetune_camera_head` | split-head fine-tune: trainable COPY of `head_pose.proj`/`head_camera.proj` applied to the FINAL readout only (in-decoder interm preds keep the frozen originals) at `lr × pose_head_lr_scale`; pose needs `pose_supervision` or `keypoint_supervision`, camera needs `keypoint_supervision` (kp2d) |
| `contact.targets.joint` | enabled, weight, loss params, `joint_set` (`smplx_body_22`\|`extremities_4`\|`kindyn_6`), subset masking, confidence weights |
| `data.datasets` | list of `{name, config, split}` (only `climbing_corpus` exists); `frames_per_batch` (memory-flat batch budget), `sequence.{frames_per_clip,frame_stride,jitter,target_frame,eval_full_scenes,eval_max_frames}`; `embedding_cache` (loaders emit precomputed bf16 backbone embeddings from `features/embedding` — built by `scripts/precompute_embeddings.py` — and the model skips the frozen backbone; missing files hard-error; frame JPEGs are not pixel-decoded — `img` is a zero crop, masks still decode) |
| `train` | `backbone_no_grad`, `detach_interm_preds` (both true; ~20% faster, grad-asserted no-ops), `compile_backbone` |
| `logging` | tensorboard + optional wandb (project `contact-anything`, off by default) |
| `output` | run dir, `monitor` (e.g. `test/joint_f1`; `*_f1`/`*_iou`→max, `loss`/`mae`→min) |

## Datasets

| name | granularity | labels |
|---|---|---|
| `climbing_corpus` (raw ClimbingVideos corpus) | video clips | raw body-22 contact (contacts_1/2, 52→22 fold); training reduces to four extremities or the six kindyn groups. Optional kindyn GT **forces** for six groups (`left_hand, right_hand, left_foot`=toe`, right_foot, left_ankle`=heel`, right_ankle`) in bw units, body-root frame; kindyn twist motion targets; MHR pose pseudo-GT (`mhr_1`) and MHR-native keypoint/vertex GT (`mhr_sup_1`) |

`ReconstructionSceneDataset` (`contact/data/reconstruction_scenes.py`) is the inference-only
mirror for BetterVideoReconstruction out-trees; it carries no labels.

ClimbingVideos corpus label semantics (important):
- The corpus is read **directly** from `/data3/.../better/data/ClimbingVideos`
  (`scenes/scenes.db` curated split: 864 train / 108 test scenes, 61 test scenes annotated as
  of 2026-08-28; pre-extracted `frames/` JPEG tree; `features/` contacts, sam3 masks/bboxes,
  geometry, kindyn).
  **2026-08-27 corpus regeneration**: better contact/force/pose GT under a new archive schema
  (`contact_label_schema` 2; kindyn stores forces on 35 named contact frames, world-frame
  newtons, fitted per-scene `gravity_world`, per-frame `force_confidence`; contact confidence
  uses NaN = joint not assessed). The loader folds the frames into the six groups by parent
  joint (hands sum palm+fingers+thumb into the wrist; ~4 % of force mass on non-group frames
  is dropped), converts to bw/root by default (`force_supervision.gt_frame`/`units` flip to
  world/newtons), weights force-loss rows by `force_confidence`
  (`force_supervision.confidence`), and emits the FITTED gravity. Results against the
  pre-regeneration corpus are not comparable.
- **Train labels are automatic and cover all 22 joints** (contacts_1 by default). Test labels
  manually annotate 14 observable joints; fingers are folded into the wrist/hand labels, and
  the other eight joints are fixed non-contact on reviewed frames. Test-scene discovery
  requires `annotation.npz` to exist.
- Video joint labels are **motion-gated "stable contact"** (stillness/hysteresis in the
  estimator), not instantaneous surface contact. The same gap applies to **forces** (R8):
  stable-contact labels ≠ instantaneous load, so the physics loss gates on predicted probs, not
  labels (D8). The **supervised** force loss instead gates on kindyn's own contact mask
  (= contacts_2, the mask the forces were solved under).
- Scenes carry **per-frame camera extrinsics** (`cam_from_world`, OpenCV, metric) and
  `gravity_world` — since the 2026-08-27 regeneration a **per-scene FITTED unit down vector** read
  from `kindyn_1.npz` (one `(3,)` vector per scene), NOT the `[0, 1, 0]` constant and NOT v1's
  camera-0 derivation. Measured over the 864 train scenes it tilts from +y by median 3.2°, p90
  27.5°, max 61.4° (519 scenes > 1°, 163 > 15°), so anything treating world y as "up" is wrong for
  hundreds of scenes. The loader emits the fitted vector whenever `load_forces` **or** `load_motion`
  is on; consumers are the physics/force-consistency losses, the motion diagnostics' vertical
  projection, and the `gravity_view` motion frame.
- Train/val split is grouped by **source video** (chunks of one video never straddle splits).
  With `data.eval_split: test` every curated train scene trains and the annotated test scenes
  evaluate.
- The four-output target order is `left_hand, right_hand, left_foot, right_foot`; each foot is
  `ankle OR foot`. A known positive wins under partial annotation, while a known negative needs
  both source joints annotated. Confidence is max over positive evidence and mean when both are
  known free. Loss reduction uses exact global confidence-mass reduction under DDP.

## Training Outputs & Checkpoints

Each run: `output/<EXP_NAME_YYYYMMDD_HHMMSS>/` with `best.pth` (by `output.monitor`), rolling
`last.pth`, periodic `epoch_XXXX.pth`, `config.yaml` (resolved), `split_manifest.json`,
`tensorboard/`.
Checkpoints are **schema v2**: trainable-only weights + optimizer/scheduler + epoch/step + best
metric + resolved config + wandb run id + RNG states. Loading **hard-fails with a param diff** on
architecture mismatch (never silent random init). `--resume auto` reproduces the uninterrupted
run exactly (RNG + stateless window jitter restored). Pre-2026-09-01 checkpoints are incompatible.

Base model checkpoints (HuggingFace, paths set in `configs/base.yaml`):
`facebook/sam-3d-body-dinov3` (default) / `facebook/sam-3d-body-vith`.
