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
python scripts/train.py --config configs/old/climbing_videos_joint.yaml
python scripts/train.py --config configs/old/climbing_videos_joint_temporal_center_v2.yaml
# Two-GPU DDP (`data.frames_per_batch` is per GPU):
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/train.py --config configs/old/climbing_videos_joint.yaml

# Force training. Physics (RNEA) regime (a) needs the editable better-robot / better-human from
# the sibling ../BetterRobot / ../BetterHuman checkouts (step-01 env wiring); the MHR archive
# resolves via that checkout, or set $BETTERHUMAN_MODELS_DIR. See docs/forces.md.
python scripts/train.py --config configs/old/climbing_videos_force_warmstart_t7hinge.yaml
# Supervised kindyn forces (force-only build, six groups, no physics/extrinsics):
python scripts/train.py --config configs/old/climbing_corpus_force_supervised.yaml

# Evaluate on grouped val or the manually annotated corpus test split (31 scenes).
# Reports P/R/F1/F2/IoU, per-extremity metrics and a threshold curve.
python scripts/evaluate.py --checkpoint output/<run>/best.pth --config configs/<experiment>.yaml
python scripts/evaluate.py --checkpoint output/<run>/last.pth \
  --config configs/old/climbing_videos_joint.yaml --split test --threshold 0.3
# Force runs add physics-consistency metrics (physics_residual, per-extremity force magnitudes
# split by predicted contact, gate-violation rates, vertical-force-sum). Lacking a trained force
# checkpoint, --warm-start builds the untrained force branch from the config's init_contact_checkpoint.
python scripts/evaluate.py --config configs/old/climbing_videos_force_warmstart_t7hinge.yaml --warm-start --split test

# Qualitative demo (GT vs predicted contacts; force arrows when the checkpoint has a force head)
python scripts/demo.py --checkpoint output/<run>/best.pth --config configs/<experiment>.yaml --num_samples 10
# Rendered corpus videos (contact disks + force arrows; shards over torchrun ranks)
python scripts/render_climbing_video_contacts.py --checkpoint output/<run>/best.pth \
  --config configs/old/climbing_videos_joint_temporal_center_v2.yaml --split test --overlay-labels

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
# bf16 frozen-backbone embedding cache (corpus features/embedding, ~0.8 TB; one
# process per GPU; data.embedding_cache: true consumes it and skips the backbone)
CUDA_VISIBLE_DEVICES=0 python scripts/precompute_embeddings.py --split all --shard-index 0 --num-shards 4
python -m viewer --port 8765                      # Contact Atlas dataset browser
```

## Repository Layout

| Path | Purpose |
|---|---|
| `sam_3d_body/` | Vendored SAM 3D Body fork. Upstream code untouched; our additions are the contact head/tokens, `models/modules/temporal.py`, `models/modules/frame_attention.py`, and hooks delimited by `# --- contact temporal hook ---` / efficiency-flag comments. |
| `contact/` | Our library: `config.py` (yaml + `base:` include + strict validation), `model.py` (build/freeze/eval-pin), `targets.py`, `losses.py`, `metrics.py`, `engine.py` (shared forward), `checkpoint.py` (schema v2), `tracking.py` (wandb+TB), `data/` (loaders, collate, splits), `physics/` (`adapter.py` MHR bridge + `loss.py` RNEA residual). |
| `scripts/` | Thin CLIs: train, evaluate, demo, build_climbing_images, precompute_*, render_results_table. |
| `configs/` | `base.yaml` (all defaults, commented) + new-era experiment overrides; `configs/datasets/*.yaml` = dataset paths/options. Pre-2026-08-27 experiment yamls are archived in `configs/old/` (self-contained, causal mask pinned; their runs live in `output/old/`), v1-era yamls in `legacy/configs/`. |
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
   are appended *after* the contact tokens. Among the appended blocks
   `model.extra_token_attention` picks the regime: `mutual` (default since 2026-08-27) lets
   contact/force/motion fully inter-attend; `causal` (the legacy default) extends the mask so no
   earlier block attends a later one. Original tokens attend none of the appended blocks under
   either regime, so pose/MHR stay isolated either way.
3. **Per-target contact heads** — `head_contact` is an `nn.ModuleDict`: pooled modes support
   `vertex` → `[B, 6890|10475]` or body-22 joint logits. `pool_mode: per_token` applies one
   shared classifier independently to each token; ClimbingVideos uses four tokens → `[B,4]`.
   Output: `out["contact"]["<target>_logits"]`.
4. **Temporal module** (`model.temporal`, optional) — zero-init-gated pre-LN attention blocks over
   the frames of a clip, order-aware via sinusoidal encoding of real elapsed seconds
   (`frame_pos_sec`), optional frame-level causal mask. Post-decoder only (the in-decoder
   placements — between_layers, between_layers_cross, temporal conv, pre_decoder — were
   controlled negatives and are retired). Batches are homogeneous-T flattened clips
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
6. **Motion head** (`model.motion_head`, optional) — anchored motion tokens regressing
   standardized root-frame vel/acc per slot, `out["motion"]["joint_motion"] [B,K,6|12]`
   (12 with `motion_supervision.angular`: + the root twist's angular vel/acc). The
   `model.motion_temporal` module is post-decoder self-attention over the motion tokens
   (mandatory in practice — a per-frame head cannot represent a derivative). Supervision:
   `motion_supervision` (kindyn twist targets, σ0.12 s label smoothing, pelvis-only for
   angular).
7. **Pose temporal** (`model.pose_temporal`, optional; E2) — a deliberate exception to
   the frozen-pose rule: the pose modality's per-modality temporal block — a zero-gated
   temporal module on the pose token (index 0), run with the contact/force/motion temporal
   blocks (after `cross_modal_temporal`); the FINAL MHR output is recomputed from the updated
   token (interm preds and all other token blocks see the untouched one). Supervised by `pose_supervision` against
   kindyn-MHR pseudo-GT (`scripts/convert_kindyn_to_mhr.py` writes `mhr_1.npz` per scene —
   the kindyn SMPL-X trajectory refit as a world-frame MHR `q`, ~0.5 cm joint residual),
   compared in q space (125 local channels; the free-flyer root is never supervised).
8. **Cross-modal temporal** (`model.cross_modal_temporal`, optional) — ONE zero-gated temporal
   block (attend=joint) over the CONCATENATION of the chosen modality token blocks
   (`modalities` ⊆ {pose, contact, force, motion}, ≥ 2, canonical sequence order), run
   post-decoder BEFORE the per-modality temporal blocks: every participating token attends
   every other across the clip's frames. Deliberately relaxes D1 among the participants;
   listing `pose` writes the pose token (final MHR recomputed — needs `pose_supervision`).
9. **Frame attention** (`model.frame_attn`, optional) — per-frame (NO temporal mixing)
   zero-gated attention run AFTER every temporal block, one own-weights module per listed
   modality (`sam_3d_body/models/modules/frame_attention.py`). Each module's keys/values
   always span every enabled modality's post-temporal tokens of that frame (pose token
   included, read-only unless `pose` is listed). All updates come from one consistent
   snapshot, then apply — order-independent.
10. **Pose/camera-head fine-tune** (`train.finetune_pose_head` / `train.finetune_camera_head`) —
   SPLIT-HEAD since 2026-08-27: the ORIGINAL heads stay frozen and keep producing every
   in-decoder intermediate prediction (whose per-layer keypoint-token refresh feeds back into
   the frozen decoder — training the shared head perturbed the frozen model layer by layer,
   the earlier divergence mechanism), while trainable COPIES of the projection FFNs
   (`head_pose_ft_proj` / `head_camera_ft_proj`, deepcopy-initialized so init behavior is
   exactly frozen) are applied to the FINAL pose token only, via the final-readout recompute.
   Copies form their own optimizer param group at `optim.lr × train.pose_head_lr_scale`.
   Pose finetune requires `pose_supervision` or `keypoint_supervision`; camera finetune
   requires `keypoint_supervision` (kp2d — the only camera-constraining loss) or
   `motion_consistency`. The checkpoint carries the copy weights (`pose_head_finetune`
   {enabled, split} / `camera_head_finetune` in the arch signature). Side effect of the split:
   `pred_cam_t_frozen`/`global_rot_frozen` (the rail anchors) are now genuinely the FROZEN
   model's outputs even under head finetuning.
11. **Pose→motion consistency** (`motion_consistency`, optional; no parameters — a pure loss,
   `contact/motion_consistency.py`) — lifts the PREDICTED per-frame body placement to the
   metric world with the dataset extrinsics (`p_w = R_ext^T((mean(kp[9,10]) + pred_cam_t) −
   t_ext)`, `R_w = R_ext^T · diag(1,−1,−1) · euler("xyz", global_rot)` — the composition
   verified against the motion-probe artifact), differentiates it with the kindyn BVR
   body-twist stencil, standardizes with the `motion_supervision` pelvis table, and applies
   Huber terms: vs the kindyn GT twist (`loss.gt`, gradient → pose path; attacks depth
   wobble) and vs the motion head's **detached** prediction (`loss.head`, gradient → pose
   path ONLY — the head is never dragged toward a degenerate pose trajectory). Twist rows
   need stencil support (t−1, t, t+1), so clip boundaries are never twist-supervised.
   Derivative terms alone leave the ABSOLUTE root placement in a null space (a constant
   `pred_cam_t` is invisible under a static camera — the corpus_allmod_mutual collapse:
   camera at a constant 9 cm depth, kp2d ~12k px off-person, motion head dragged down).
   Three anchor terms close it: `loss.pos` (world mean-hips vs kindyn root +
   `hip_offset_root`, the measured ≈9 cm constant; Huber metres, every valid row incl.
   boundaries), `loss.rot` (`so3_log(R_pred^T R_gt)`, Huber radians; probe: no constant
   frame offset), and `loss.cam_rail` (trust region vs the frozen model's own `pred_cam_t`,
   stashed by the recompute hook as `out["mhr"]["pred_cam_t_frozen"]`: zero inside
   `cam_rail_margin_m`, linear beyond — inert for a healthy model). v4 adds
   `loss.rot_rail` (the same trust region on `global_rot` vs the stashed
   `global_rot_frozen`, geodesic radians — v3 proved the unrailed orientation is
   the next escape channel: pinned near-constant, ~55° from GT) and
   `angular: false` (the gt/head twist comparison drops the angular rows — pure
   differentiated orientation wobble, the signal that rewards constancy). GT root poses ship as
   `motion_root_pos`/`motion_root_valid` (+ existing `motion_rot`) with `load_motion`.

### Invariants (do not break)

- **Freeze filter is name-based**: only params whose dotted name contains `"contact"`, `"force"`,
  `"motion"`, `"cross_modal"`, `"frame_attn"` **or** `"pose_temporal"` train (tokens, heads,
  posemb/feat linears, `contact_temporal*`, `force_*`, `motion_*`, `cross_modal_temporal`,
  `frame_attn.*`, `pose_temporal`). Any new trainable module must carry one of those in its
  attribute path. The pose outputs move only via `pose_temporal`, the `pose` modality of the
  cross-modal bricks, or the explicit `train.finetune_pose_head`/`finetune_camera_head`
  flags (which build trainable head COPIES `head_pose_ft_proj`/`head_camera_ft_proj` outside
  the name filter — train.py adds them to the saved-name set).
- **Mask invariant**: inside the decoder the original tokens NEVER attend the appended blocks —
  pose/MHR outputs have an exactly-zero Jacobian w.r.t. every contact/force/motion param under
  either mask regime. Under `extra_token_attention: causal` (the legacy default; the repo
  default is `mutual` since 2026-08-27) additionally no earlier
  appended block attends a later one (contact ⊥ {force, motion}, force ⊥ motion), so contact
  outputs have an exactly-zero Jacobian w.r.t. every force **and motion** param (D1); `mutual`
  deliberately opens full inter-attention among the appended blocks (D1 gone between them —
  incompatible with `train.freeze_contact`, and captured in the arch signature). The
  POST-decoder bricks `cross_modal_temporal` and `frame_attn` relax D1 the same way **among
  their listed modalities**; the frozen pose/MHR outputs remain isolated unless `pose` is a
  listed (written) modality.
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
| `force_supervision` | supervised kindyn GT-force loss (exclusive with `physics`): `target_frame` (center\|all rows), `gt_frame` (root\|world), `units` (bw\|newtons), `confidence` (weight rows by kindyn force_confidence), Huber `force`/`huber_delta_bw`, `outlier_bw` cut, `noncontact` L1 |
| `model.motion_head` / `model.motion_temporal` | motion tokens: explicit anchors + post-decoder temporal self-attention; `anchored: false` = pure learned queries (no in-decoder keypoint-anchored update; the index list only names/counts slots) |
| `model.cross_modal_temporal` | ONE temporal block over the chosen modality blocks (`modalities` ≥ 2 of pose/contact/force/motion) — cross-modality mixing across frames |
| `model.frame_attn` | per-frame attention after the temporal blocks: per-modality own-weights blocks whose keys/values span every modality's tokens of the frame |
| `model.extra_token_attention` | decoder mask among the appended blocks: `mutual` (contact/force/motion inter-attend, default since 2026-08-27) \| `causal` (block-triangular, legacy) |
| `motion_supervision` | kindyn twist vel/acc loss: `joint_names`, `root_convention`, `angular` (12-dim root target), `target_smooth_sec`, standardize `[K][2|4][3]` |
| `motion_consistency` | pose→motion consistency + absolute anchors: the PREDICTED pose lifted to world (extrinsics) and differentiated (BVR body twist) — pelvis vel/acc Huber vs kindyn GT (`loss.gt`) and vs the DETACHED motion head (`loss.head`), both grad → pose path only; plus `loss.pos`/`loss.rot` (absolute world root pose vs kindyn + `hip_offset_root`), `loss.cam_rail`/`loss.rot_rail` (trust regions vs the frozen `pred_cam_t`/`global_rot`) closing the constant-camera/-orientation null spaces, and `angular: false` restricting the twist comparison to the linear rows; requires `motion_supervision` + a trainable pose path + T ≥ 3 |
| `model.pose_temporal` / `pose_supervision` | E2 pose branch: zero-gated pose-token temporal module + kindyn-MHR pseudo-GT q-space Huber (`mhr_1.npz` via `scripts/convert_kindyn_to_mhr.py`); `loss.shape_rail`/`scale_rail` = L2 pinning the head's 45 blendshape / 28 bone-scale outputs to the FROZEN readout's stashed `shape_frozen`/`scale_frozen` (nothing else supervises them — the pose/keypoint losses are girth-blind, and the s1 probe showed the ft head warping them to \|x\|~5) |
| `train.freeze_contact` | regime (a): freeze contact, train force only (requires `model.init_contact_checkpoint`) |
| `train.finetune_pose_head` / `finetune_camera_head` | split-head fine-tune: trainable COPY of `head_pose.proj`/`head_camera.proj` applied to the FINAL readout only (in-decoder interm preds keep the frozen originals) at `lr × pose_head_lr_scale`; pose needs `pose_supervision` or `keypoint_supervision`, camera needs `keypoint_supervision` (kp2d) or `motion_consistency` |
| `keypoint_supervision` | SAM3D-style stabilizers from kindyn `joints_world` (13 joints ↔ MHR70 by name): `kp2d` crop-normalized reprojection (constrains the camera), `kp3d` hips-relative camera-frame, `kp3d_abs` absolute metric anchor; `kp_vel`/`kp_acc` WORLD-frame keypoint velocity/acceleration (central stencil over the clip's real elapsed seconds, predictions lifted with the GT extrinsics — loss-only use; camera-frame differences would bury body motion under camera egomotion) vs finite-differenced `joints_world`, GT-acc outlier rows dropped; corpus loader flag `load_keypoints` |
| `contact.topology` | `smpl` / `smplx` (`mhr` → NotImplementedError) |
| `contact.targets.vertex/joint` | enabled, weight, loss params, `joint_set`, subset masking, `derive_from_vertex`, confidence weights |
| `data.datasets` | list of `{name, config, split}`; `frames_per_batch` (memory-flat batch budget), `sequence.{frames_per_clip,frame_stride,jitter}`; `embedding_cache` (corpus loaders emit precomputed bf16 backbone embeddings from `features/embedding` — built by `scripts/precompute_embeddings.py` — and the model skips the frozen backbone; missing files hard-error; frame JPEGs are not pixel-decoded — `img` is a zero crop, masks still decode) |
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
  (`scenes/scenes.db` curated split: 864 train / 108 test scenes, 31 test scenes annotated so
  far; pre-extracted `frames/` JPEG tree; `features/` contacts, sam3 masks/bboxes, geometry,
  kindyn). The exported ClimbingVideos_v1 dataset is redundant and its loader is legacy.
  **2026-08-27 corpus regeneration**: better contact/force/pose GT under a new archive schema
  (`contact_label_schema` 2; kindyn stores forces on 35 named contact frames, world-frame
  newtons, fitted per-scene `gravity_world`, per-frame `force_confidence`; contact confidence
  uses NaN = joint not assessed). The loader folds the frames into the six groups by parent
  joint (hands sum palm+fingers+thumb into the wrist; ~4 % of force mass on non-group frames
  is dropped), converts to bw/root by default (`force_supervision.gt_frame`/`units` flip to
  world/newtons), weights force-loss rows by `force_confidence`
  (`force_supervision.confidence`), and emits the FITTED gravity (tilts up to ~27°).
  `mhr_1.npz` pose pseudo-GT must be re-generated for the new scenes. Results against the
  pre-regeneration corpus are not comparable.
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
  known free. The current experiment uses focal-only loss (`alpha=0.6`, `gamma=2`) and exact
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
