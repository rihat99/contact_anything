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
`docs/old/history/`. There is **no backwards compatibility** with older configs or checkpoints.

**2026-09-05 — the two-stage pipeline (`docs/old/refiner.md`).** The pose is no longer improved with
image-side temporal models. **Stage 1** (`configs/stage1.yaml`) is the per-frame SMPL-X + CLIFF
model alone (no contact tokens, no temporal block), trained once and frozen. **Stage 2**
(`configs/stage2.yaml`) appends the contact tokens to the frozen decoder and runs the
**world-space temporal refiner** (`model/refiner.py`) behind the frozen stage-1 body: depth
smoothing, world lift with the camera extrinsics, a world-independent per-frame token, a local
RoPE transformer, and zero-init heads for the pose offset, contact, motion and forces. Contact is
trained in stage 2 only. Tests: `tests/test_refiner.py` (CPU).

**2026-09-08 — round 5 and the final model (`docs/old/architecture_2.md`).** `configs/final.yaml`
(self-contained on `base.yaml`) is the recipe: the frozen stage-1 body + the fixed Gaussian in
the world IS the pose (no pose head — every learned correction measured worse), the token
channels `model.refiner.token`, `motion_supervision.stencil: aligned`, contact / motion / force
heads on cached pose tokens, 64 clips per step; `final_rnea.yaml` adds the RNEA residual
(force direction −1°, off-contact force ×2). Runs `output_2/final*`; the round-5 arms are
written up in `docs/old/round5_2026-09-07.md` (seed spread: F1 0.002, MPJPE 0.02 mm — smaller
differences are noise).

**2026-09-11 — round 6 (`docs/old/round6_2026-09-11.md`) and cleanup.** Contact on the final recipe
is static pose (F1 0.888) + world motion (+0.039); the frozen pose token is worth +0.005 (two
seeds, precision); the velocity channels' frame (lifted vs camera) does not matter; the limb
labels carry no image evidence and the confidence weights mute the rows where the image would
decide (`scripts/diag_label_anatomy.py`). Every run except stage 1 and `output_2/final*`, the
round-3/4/5/6 configs and the two round-6 token options (`velocity`, `velocity_frame`) went to
`/data3/rikhat.akizhanov/trash/cleanup_20260911/`; the round-3 refiner's `eval.json` stays as
`output/round3_refiner_eval.json` (the final model's tensorboard reference line).

**2026-09-12 — round 7 (`docs/old/round7_2026-09-11.md`): the block smooths the body itself.** The
refiner's input Gaussians are replaced by the temporal block: one-sided (Nyquist-visible)
forward-difference velocity losses on the raw GT (`motion_supervision.stencil: forward`, no
acceleration term), one-sided rate channels for the root and every joint (`token.one_sided_velocity`,
`token.joint_velocity`), iterative per-layer refinement with feedback (`model.refiner.iterative`) and
deep supervision (`layer_weight`). Recipe `configs/smooth/P_k30.yaml` (30 epochs, `best.pth` = min
jitter): 54.86 mm / jitter 5.9 (GT floor 6.9) vs the Gaussian body 55.5 / 2.7, paired +0.6 mm;
in-band acceleration r 0.77 vs 0.70. Central differences were the root cause (zero response at
Nyquist); acceleration losses of every kind (pointwise, RMS, smoothed-target, band-limited) do not
help; the pelvis error (110 mm) is stage-1 low-frequency bias. Runs `output_3/`, configs
`configs/smooth/`. Contact and force heads were OFF this round.

**2026-09-12 — round 8 (`docs/round8_2026-09-12.md`): gravity, contacts and forces on the smooth
body.** The GT gravity channel is gone (`token.gravity` was never needed for the pose); the refiner
predicts gravity itself (`camera_axes` + a per-layer `gravity` head: a body-frame correction on the
camera's down axis, unit votes pooled per clip; `gravity_supervision` on the scenes whose corpus
gravity was MEASURED — 40 % of the corpus gravity is the first camera's down axis, and the two
measurements disagree by 10–20°, so the head's 1° gain over the camera axis is inside the target
noise). Contact / force heads read every layer and feed back; the contact and gravity gradients cost
the shared trunk 3.5 / 1 jitter units, so `model.refiner.head_grad_scale: 0.3` scales what the heads
send into the trunk (0 = probes: pose intact, F1 −0.05). `contact_consistency` (forward stencil) puts
the in-contact speed at the GT floor for +0.25 mm; the RNEA residual as a LOSS (`force_consistency`,
`model/physics.py`) is the one force change that helps (angle 22 → 20.5°), the residual as per-layer
FEEDBACK (`residual_feedback`) adds nothing on top, `frame_mask_p` 0.1 is a null result. Recipe
`configs/r8/F2m_scale03.yaml` (run `output_3/F2m_scale03_20260912_154016`): 53.9 mm / jitter 6.0 /
F1 0.924 / force MAE 0.179 / angle 20.6° on the 107-scene split (the test split and 56 annotations
changed mid-round; P_k30 re-scored there 53.9 / 5.6). Arm configs `configs/r8/`, runs `output_3/`.

**2026-09-13 — round 9 (`docs/round9_2026-09-13.md`): the regenerated corpus (1419 / 109 scenes; pose tokens
rebuilt with `scripts/data/precompute_pose_tokens.py`, now in main, live backbone), an optional KNOWN-gravity
input (`model.refiner.gravity_input`: four token channels, per-clip Bernoulli `p_given` on measured clips in
training, `eval_given` at inference; a given clip's gravity output IS the input and the gravity loss skips
it), and a strength-weighted joint-torque term (`force_consistency.loss.joint_torque` +
`joint_torque_multipliers`, BVR's table, from the same RNEA on the detached body). The torque term alone
only LOWERS THE HANDS; with the force total pinned (`force_supervision.loss.sum_force`, root residual force
5 / δ 0.2) and the toes supervised in magnitude it re-allocates: recipe `configs/r9/D_total_feet.yaml`
(run `output_4/D_total_feet_20260913_134817`; 53.6 mm / jitter 5.4 / F1 0.922 / force 0.163 bw (0.157 given) /
20.5°; corpus feet 0.87 of GT (was 0.76); climb_wall_2 MAE 83 N / share 3.9 pp / corr 0.62). `predict_test.py`
dumps read the pose-token cache (`rc.build_dataset(..., pose_tokens=True)`); renderers stay live. Arms
`configs/r9/` (A–D + `*_given_eval` / `*_predict` twins), runs `output_4/`, docs `docs/round8_force_anatomy.md`
(why the feet were low) and `docs/round9_2026-09-13.md`.

**2026-09-14 — round 10 (`docs/round10_2026-09-14.md`): limb tokens + decoder contact tokens + gravity force
frame.** `model.refiner.limb_tokens` runs SEVEN tokens per frame — the body token and one per kindyn group
carrying that extremity's geometry and its DECODER CONTACT TOKEN (`model.contact` on, live decoder off the
rebuilt `features/embedding` cache, 1.29 TB) — through alternating per-slot temporal RoPE attention and
within-frame attention (`model/rope.py::_AlternatingBlock`); the contact / force heads read the limb tokens and
each limb gets its own feedback. `model.refiner.force_frame: gravity` reads the forces in a gravity-aligned,
body-headed frame (`out["force"]["frame"]` = world-from-frame; the GT hand directions concentrate 0.79 in the
root frame vs 0.88-0.90 there). Recipe `configs/final/final_full.yaml` (run `output_5/L_limb_20260913_180958`, use
`last.pth`: contact peaks at epochs 3-5, forces at the end) vs the 16-epoch control `configs/r10/D16.yaml`:
F1 0.926 / 0.930 vs 0.920 (precision, toes, transitions +0.08), force 0.160 bw (0.157 given) / 20.7° /
off-contact 0.026 vs 0.165 / 20.4 / 0.038, MPJPE 52.9 vs 53.7; climb_wall_2 MAE 75.4 N / F1 0.963 (optimisation
78.6 / 0.957) with the hands fixed and the feet unchanged (share 5.5 pp). Live training is DISK-bound (5 GB of
embeddings per step, ~23 min/epoch on four GPUs). `limb_tokens` + `head_grad_scale 0` leaves `limb_output_norm`
unused (DDP error).

**2026-09-19 — new box + cleanup.** The work moved to a machine with 8 x RTX PRO 6000 Blackwell
(96 GB each); `/data3` is gone, the corpus and the siblings live under `/home/rikhat.akizhanov/better/`,
and the environment is a `uv` project venv instead of the conda env. Everything up to and including
round 9 went to `../trash/cleanup_20260919/`: the run trees `output/` `output_3/` `output_4/`, the
configs `r8/` `r9/` `smooth/` `final*` `baseline` `static_ray` `stage1_eval_auto`, the round-5 one-off
scripts (`audit_rnea.py`, `audit_targets.py`, `diag_sigma_sweep.py`), the `contact_anything_gvhmr`
worktree and the branches `GVHMR` / `legacy` / `worktree-agent-*` / `origin/dev` / `origin/proposal`
(`dev` and `legacy` were ancestors of `main`; the ones that were not — `GVHMR`, `proposal` — are
git bundles in that same trash directory). `configs/r10/D16.yaml` was FLATTENED onto `configs/base.yaml`
— it was the root of a 19-deep chain through the trashed rounds — and carries their rationale in its
comments; every round-10 config resolves exactly as before. The frozen stage-1 body and the
frozen-baseline jsons moved out of `output/` into `checkpoints/` (their recorded `model.smplx.model_path`
was rewritten to the new BetterHuman path; the originals are in the trash). The SAM 3D Body snapshot
was re-downloaded (gated: the HF token is in `~/.cache/huggingface/token`), the DINOv3 hub repo is in
`~/.cache/torch/hub`, and the 1.29 TB `features/embedding` cache was rebuilt (83 frames/s per GPU at
batch 35, 8 shards).

**2026-09-19 — speed on the new box (measured end to end, `--limit-scenes` runs).** The cached-token
path is CPU-launch-bound: 32 clips in one micro-batch cost 470 ms against 4 x 366 ms accumulated, so
`D16.yaml` now runs `frames_per_batch: 1920` / `accumulate_steps: 1` (5,500 frames/s, ~1.5 min/epoch on
ONE GPU, 6 GB; 64 clips = 11,000 frames/s). The live-decoder path is GPU-bound in the frozen decoder
(~2 ms per frame forward + its backward): `L_limb.yaml` runs `frames_per_batch: 480` / no accumulation
with `decoder_checkpointing: false` (55 GB per GPU), 179 frames/s per GPU, ~8 min/epoch on 4 GPUs.
`pin_memory` is off in `build_loaders`: pinning the 1.26 GB embedding batches in the main process cost
16 % of the L step. Every round-10 config keeps its clips per step per GPU. Dead ends, measured: flash
attention (SDPA already), TF32, `torch.compile` (2x slower), a 16-worker loader (the mask PNG decode +
warp is 8 ms per frame but hidden behind the step), `decoder_bf16` (4 %). The live backbone in
training is 55 frames/s — the cache stays.

**2026-09-19 — round 11, the LADDER (`configs/final/`, runs `output_6/`).** The one-off ablations are
replaced by a ladder that adds the pieces one at a time, every rung trained from scratch (the
stage-1 / stage-2 split is gone: `model.smplx.checkpoint` / `frozen`, `warm_start`,
`optim.head_lr_scale`, `configs/stage1.yaml`, `configs/r10/`, `dump_stage1.py` and
`analyze_stage1.py` are in `../trash/ladder_cleanup_20260919/`; the per-frame SMPL-X + camera
heads train jointly with everything else in every run). Rungs, each = the previous + one thing:
`final_base` (heads on the pose token: `model.token_heads`, no temporal model, no camera) →
`final_temporal` (+ `cross_modal_temporal` over the pose token; NO temporal losses: velocity or
stillness losses on the world-lifted per-frame body collapse the CLIFF depth, −336 mm) →
`final_world` (the refiner: lift, world-independent token, one offset head; the velocity and
stillness losses start here, on the refined body) → `final_iter` (+ `iterative`, deep supervision) → `final_limb` (+ six contact limb tokens
with DECODER contact tokens: live decoder off the embedding cache) → `final_limb12` (+ six force
limb tokens with decoder force tokens, `model.refiner.force_tokens`) → `final_rnea` (+ the RNEA
residual loss) → `final_full` (+ joint torques) = the model. Ablations off `final_full`:
`final_nocontact` (no contact anywhere), `final_rf_{full,0.5,0.125}` (receptive field; the default
is ±0.25 s TOTAL = 3 layers × `window` 0.0833). Same schedule everywhere: 10 epochs, 120-frame
clips, 32 clips per step, lr 3e-4, warm-up 300, cosine to 1e-6, EMA. New: `smplx_supervision.loss`
always scores the per-frame heads' body and `smplx_supervision.refined` the refined one; every
build lifts its per-frame body into the world (`out["smplx"]["*_world"]`), so the motion, stillness
and RNEA losses / metrics run without a refiner; `force_consistency` at zero weights reports the
residual metrics only; the gravity loss averages per-frame estimates over the clip;
`model.refiner.window: null` = the whole clip. Rig dumps go to `predictions/<run>/` and are scored
by `scripts/score_rig.py` (force MAE in BW % with the reconstructed mass and, for our rows,
with the subject's MEASURED mass, per-frame share error, RNEA residual of the predicted
pose / contacts / forces); the old rig dumps are in `../trash/cw3_predictions_20260919/`.
Results (`docs/round11_ladder.md`, `output_6/logs/{eval_table,rig_12runs}_20260919.*`, last.pth,
10 epochs): the token block is the pose step (94 → 58 mm, the ladder's best pose), the refiner the
trajectory step (jitter 89 → 12, RNEA residual halved) at a pose cost on this schedule (61 mm), the
contact limb tokens the contact / force step (F1 0.901 → 0.923, precision 0.94, off-contact force
halved, rig F1 0.939 / 86 N / share 8.1 pp vs the optimisation's 0.916 / 94 N / 10.8 pp); force limb
tokens, RNEA loss (torque residual 0.08 → 0.05, pose −3 mm) and joint torques move little; no contact
anywhere = same forces, better pose; receptive field full = 0.5 = 0.25 s, 0.125 s drops. Every
from-scratch run is still improving at epoch 10 (the round-10 body sat at 52.9 mm after 30 + 16).

**2026-09-21 — in-the-wild videos (`../data/willd_videos/`, trees under `out/<stem>/`).** The joint
CV + BEDLAM model (`output_7/joint_frames35_*/last.pth`, `configs/frames/joint_frames35.yaml`) runs on
arbitrary footage of any length: `scripts/prepare_wild.py` builds the out-tree the reconstruction
loader reads — BVR's SAM 3 stage as a subprocess in BVR's venv (`configs/wild_sam3.toml`: the
LARGEST frame-0 person only, 200-frame propagation parts) and a STATIC camera in
`geometry/transform.npz` (identity extrinsics, MoGe-2's first-frame intrinsics; MoGe is installed in
the venv with `uv pip install --no-deps`, outside `uv.lock`); stems with a full pipeline tree under
`--moving-root` (the moving-camera clips: BVR's `scripts/pipeline.py` with `configs/wild_pipeline.toml`
— tracker → VGGT-Omega on EVERY frame, chunked → Sapiens → SAM 3D Body → SMPL-X → fuse → metric
scale, no human optimisation; `wild_pipeline_duo.toml` keeps two people — into
`../data/willd_videos/bvr_out/`) get `sam3/` + `geometry/` symlinked instead, every tracked person
predicted. VGGT-Omega fits ~1000 frames on a 96 GB card and not 2050: floor_exercise_2 was decimated
to 30 fps (the 60 fps original is in `originals_60fps/`). A tree with the
tracker and the cameras only takes every tracked person from `sam3/bboxes.npz`
(`data/reconstruction.py`). `predict_reconstruction.py` (tiled 240-row windows, all frames at
30 fps) now also dumps the model's own gravity (`gravity_world` / `gravity_body` in `smplx.npz`);
`scripts/render_wild_overlays.py` writes `predictions/<run>/overlay.mp4` (`--png`: the originals into
`<stem>/frames/` and the overlays into `predictions/<run>/overlay/` as PNG frames too): the frame
fogged towards white (`--fog 0` = none) with the 22-joint skeleton and one thick red arrow per parent
joint (its slots' forces summed, 0.7 m per body weight, the hand arrows drawn from the middle-finger
base, the foot frames folded onto the ankle, strokes sized by the pelvis depth; no mesh, no
GPU); `--results-dir ../data/willd_videos/results` writes the unfogged `<stem>.mp4` there instead;
`scripts/view_results.py --wild <root>` browses the trees in the viser viewer (no GT, world = the
camera frame oriented by the predicted gravity), whose bodies now draw the way BVR's viewer draws
them (52-joint sphere + cylinder skeleton, contact-frame spheres with a `contact frames` toggle,
red force arrows). Container headers over-report frame counts — every stage counts by sequential
decode. Logs `output_7/logs/wild_*_20260921.log`.

**2026-09-21 — the frames35 ladder (`configs/frames/ladder/`, runs `output_8/`).** The round-11
ladder re-run on the 35 contact frames: every rung = its `configs/final/final_*.yaml` plus the
climbing_frames35 recipe (`contact_set: frames35`, per-slot BCE weight 35/6, uniform force weights,
no force tokens), so `frames_base` → `frames_temporal` → `frames_world` → `frames_iter` →
`frames_limb` (35 contact tokens) → `frames_rnea` → the full model = `output_7/climbing_frames35_*`
(the six-group rung 6, force tokens, has no counterpart); the ablations `frames_nocontact` (contact
off, but each frame keeps its own limb token through 35 decoder FORCE tokens + `force_tokens`, as the
six-group ablation kept its six) and `frames_rf_{full,0.5,0.125}` inherit `climbing_frames35.yaml`.
Same schedule and scoring as round 11 (`output_8/logs/chain_frames35.sh`: train, fp32 eval of
last / best, test dumps, climb_wall_3 rig dumps; `finalize_frames35.sh` writes the eval and rig
tables `output_8/logs/{eval_table,rig}_frames35_20260922.*`).

## Environment

```
PYTHON=.venv/bin/python                                      # uv project venv: python 3.12, torch 2.13+cu129
```
`uv sync` rebuilds it from `pyproject.toml` + `uv.lock`; `torchrun` is `.venv/bin/torchrun`.
Run everything from the repo root. Scripts insert the repo root at the head of `sys.path`
before importing (`scripts/train.py` would otherwise shadow the `train/` package).
Corpus (read-only): `/home/rikhat.akizhanov/better/data/ClimbingVideos`. The SMPL-X head
needs the sibling `../BetterHuman` checkout (`better_human` + `models/smplx/SMPLX_NEUTRAL.npz`).

## Key commands

```bash
# Train (resume: --resume auto | --resume path/to/last.pth; --limit-scenes N for smoke runs).
# Rank 0's console goes to <output.dir>/logs/<run>.log by itself; redirect anything else you
# launch into that logs/ too — never write files into the run tree itself.
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --standalone --nproc-per-node=2 \
    scripts/train.py --config configs/final/final_full.yaml  # the model (live decoder, 2 GPUs)
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/train.py --config configs/final/final_iter.yaml   # rungs 1-4: cached pose tokens, 1 GPU

# Frozen SAM3D-as-SMPL-X baseline on the SAME test protocol -> the `frozen` tensorboard run
# (output.frozen_metrics; recompute whenever eval_max_frames / stride / dataset change)
$PYTHON scripts/eval_frozen_smplx.py --config configs/final/final_full.yaml --out checkpoints/frozen_sam3d_smplx.json

# Evaluate on the annotated corpus test split (full-scene protocol; --checkpoint none = untrained;
# --json writes a frozen_metrics file)
$PYTHON scripts/evaluate.py --config configs/final/final_full.yaml --checkpoint output_6/<run>/last.pth
$PYTHON scripts/eval_table.py output_6/final_*                          # one table over several runs' eval.json
$PYTHON scripts/paired_ci.py output_6/<ref_run> output_6/<run> --protocol capped
#   (video-clustered paired bootstrap over predict_test.py dumps: differences to the first run with 95 % intervals)
$PYTHON scripts/diag_invariance.py --config configs/final/final_full.yaml --checkpoint output_6/<run>/last.pth  # reverse / shuffle / decimate / window
$PYTHON -m pytest tests/ -q                                  # refiner unit tests (CPU, ~35 s)

# Renders (mp4 per test scene; shard scenes over ranks with torchrun)
$PYTHON scripts/render_video.py --config configs/final/final_full.yaml --checkpoint output_6/<run>/last.pth \
    --scenes 5 --out output_6/<run>/render_contact --overlay-labels --gt-panel --scale 0.5
$PYTHON scripts/render_smplx_video.py --config configs/final/final_full.yaml --checkpoint output_6/<run>/last.pth \
    --scenes 5 --out output_6/<run>/render_pose           # GT | frozen MHR | SMPL-X head panels

# Results viewer (docs/old/viewer.md): dump a run's whole-scene test predictions once, then serve
# every run's predicted | GT | frozen SMPL-X bodies plus contact markers and force arrows
# (predicted and GT) in viser (port 8082 is the BVR viewer's)
$PYTHON scripts/predict_test.py --config configs/final/final_full.yaml --checkpoint output_6/<run>/last.pth
CUDA_VISIBLE_DEVICES=5 $PYTHON scripts/view_results.py --port 8090

# Inference on BetterVideoReconstruction out-trees (contacts + forces, no labels needed) into
# <clip>/predictions/<run name>/, then the climb_wall_3 rig table (pose vs the three-camera fit,
# contacts + forces vs the boards, RNEA residual of the predictions)
$PYTHON scripts/predict_reconstruction.py --config configs/final/final_full.yaml \
    --checkpoint output_6/<run>/last.pth --out-root ../BetterVideoReconstruction/peter/out_climb_wall_3_single \
    --videos ../BetterVideoReconstruction/peter/climb_wall_3 --video-pattern "{scene}/cam_left.mp4" \
    --pred-dir final_full --force-name forces_sup.npz
$PYTHON scripts/score_rig.py --runs final_full final_base ... --out output_6/logs/rig_<date>.md
bash output_8/logs/chain_frames35.sh 1,2 frames_limb frames_rf_0.125   # frames35 ladder: train + score, one config after another

# In-the-wild videos of any length: SAM 3 tracks + a static MoGe-2 camera (one GPU per process), the
# joint model on every frame, mesh + per-joint force overlays, and the viewer's wild mode
CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/prepare_wild.py --videos ../data/willd_videos/videos --out ../data/willd_videos/out
CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/predict_reconstruction.py --config configs/frames/joint_frames35.yaml \
    --checkpoint output_7/<run>/last.pth --out-root ../data/willd_videos/out --videos ../data/willd_videos/videos \
    --video-pattern "{scene}.mp4" --pred-dir joint_frames35 --force-name forces_sup.npz
$PYTHON scripts/render_wild_overlays.py --out-root ../data/willd_videos/out --pred-dir joint_frames35
CUDA_VISIBLE_DEVICES=5 $PYTHON scripts/view_results.py --wild ../data/willd_videos/out --port 8090

# Data preparation (scripts/data/)
$PYTHON scripts/data/extract_frames.py                 # corpus frames/ JPEG tree
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/data/precompute_embeddings.py --split all --shard-index 0 --num-shards 4
$PYTHON scripts/data/precompute_pose_tokens.py         # features/pose_token (frozen FINAL pose token per frame)
```

## Layout

| Path | Purpose |
|---|---|
| `model/sam_3d_body/` | Vendored SAM 3D Body fork. Our additions are delimited by `# --- <name> hook ---` comments: the extra-token-block hook (append learned blocks behind the asymmetric mask, per-layer update callbacks, expose the final sequence) and the efficiency hooks (precomputed embeddings, `backbone_no_grad`, `detach_interm_preds`). |
| `model/wrapper.py` | `SAM3DBodyWrapper`: builds / freezes / eval-pins the base; `forward(img|embedding, geometry, blocks)` → final tokens, block bounds, the frozen MHR readout. |
| `model/tokens.py` `rope.py` `heads.py` | `LearnedTokenBlock` (token embeddings + anchored posemb/feat update), `CrossModalRopeModule` (the temporal brick), `ContactHead` / `ForceHead` (per-token FFNs), `SmplxHead`. |
| `model/token_heads.py` | `PoseTokenHeads`: contact / force / gravity heads read straight off the pose token (the ladder's first rungs, no temporal model); outputs in the refiner's layout. |
| `model/refiner.py` | `TemporalRefiner`: depth + pose smoothing → world lift → world-independent token (+ camera context, camera axes) → RoPE transformer, iterative with per-layer pose / contact / motion / force / gravity heads and feedback (`residual_feedback` = the RNEA residual, `head_grad_scale`, `frame_mask_p`) → FK back into every camera; plus the masked time-series helpers (`gaussian_smooth`, `smooth_rotations` / `project_rotation`, `time_derivative`, `angular_velocity`). |
| `model/physics.py` | `RootWrench`: BetterRobot RNEA root-wrench residual of a world SMPL-X trajectory under six extremity forces (shared by `force_consistency` and the refiner's residual feedback). |
| `model/network.py` `build.py` | `ContactAnything` composes the above; `build_model(cfg, device)` maps the yaml sections onto it and applies `model.smplx.checkpoint` / `frozen`. |
| `model/loss/` | One `Loss` interface (`__init__.py`) and one file per term: `contact` (BCE), `force`, `smplx` (+ every pose metric), `motion` (refiner velocities / accelerations + pose-derivative matching), `contact_consistency` (in-contact stillness of the refined extremities, forward stencil), `force_consistency` (RNEA root-wrench residual, BetterRobot + BetterHuman), `gravity` (the predicted down vector vs the corpus gravity on measured scenes). |
| `data/` | `base.py` = `ClipDataset` ABC (windowing, jitter, full-scene eval) **and the frame schema** (module docstring); `climbing_videos/` (`scene.py` DB + labels, `kindyn.py` forces + SMPL-X GT, `dataset.py`); `reconstruction.py` (label-free BVR out-trees); `collate.py`, `loaders.py`, `transforms.py`. |
| `train/` | `config.py` (schema = `configs/base.yaml`, cross-key checks, `signal_needs`), `trainer.py` (DDP-exact weighted means, EMA, per-module clipping, per-step warm-up + cosine), `checkpoint.py` (trainable-only, strict), `logger.py` (tensorboard + `tee_output`), `predict.py` (`load_model`). |
| `utils/` | `geometry.py` (camera parametrizations, projection, world lift), `gvhmr_metrics.py`, `metrics.py`, `distributed.py`. |
| `scripts/` | Thin CLIs (above); `_render_common.py` shares the scene / clip plumbing and the drawing helpers; `prepare_wild.py` + `render_wild_overlays.py` (in-the-wild trees and their overlays); `score_rig.py` scores the climb_wall_3 rig dumps (`--sam3d` adds the raw SAM 3D Body pose row, per-limb force MAE columns); `score_parkour_paper.py` scores the LAAS Parkour trees under Li's own protocol plus the rig's agreement / RNEA columns; `trivial_baselines.py` (smoothed SAM-3D + measured or InteractVLM contacts + equal split / RNEA least-norm forces), `estmf_dumps.py` / `physpt_dumps.py` (Li et al. 2019 and PhysPT outputs as prediction dumps); `paper_table_forces.py` assembles the two into the paper's combined LaTeX table; `fig_opencap_grf.py` / `fig_climb_wall_3.py` (paper figures: one OpenCap clip plates vs learned vs optimisation, one climb_wall_3 clip with a frame + limb traces + angle error, into `output_7/logs/figures_<date>/`); `eval_table.py`, `paired_ci.py`, `diag_invariance.py` (scoring / diagnostics), `diag_label_anatomy.py` (round-6 label anatomy over prediction dumps), `force_corr_share.py` (Peter's corr / share on the corpus). |
| `tests/` | `test_refiner.py`: world-frame independence (with / without camera context), identity at init, pose smoothing (polar projection, still-body fixed point), receptive-field locality, gradient flow, the video-interleaved sampler (CPU, BetterHuman body). |
| `viewer/` | viser results viewer (`scripts/view_results.py`, corpus dumps or `--wild` out-trees; `docs/old/viewer.md`). |
| `configs/` | `base.yaml` (the schema, every key with its default), `final/` (the ladder: `final_base` → `final_temporal` → `final_world` → `final_iter` → `final_limb` → `final_limb12` → `final_rnea` → `final_full`, each inheriting the previous; the ablations `final_nocontact`, `final_rf_*` off `final_full`), `datasets/*.yaml` (`all` / `static` / `moving` camera subsets). |
| `docs/` | `architecture.md` (the current model in plain words), `results.md` (what worked and what did not, in easy words), `round10_2026-09-14.md` (limb tokens, decoder contact tokens, gravity force frame), `round9_2026-09-13.md` + `round8_force_anatomy.md` (round 9 and the force anatomy behind it), `round8_plan.md` + `round8_2026-09-12.md` (the round-8 design and write-up); `old/` holds every earlier document: `architecture_2.md` (the round-5 model), `round5` / `round6` / `round7` write-ups, `refiner.md` (the two-stage pipeline: rounds 1-4), `plan.md`, `results.md` (every recorded number, incl. the trashed runs), `viewer.md`, `history/`. |
| `output_6/` | Round-11 ladder runs; `output_7/` the frames35 / BEDLAM / joint runs; `output_8/` the frames35 ladder. Run directories `<exp_name>_<stamp>/` (`best.pth`, `last.pth`, `config.yaml`, `eval.json`, `predictions/`, `tensorboard/`) and `logs/`. `output_5/` holds the round-10 runs (their configs are in the trash); rounds 1-9 are in `../trash/cleanup_20260919/`. |
| `checkpoints/` | Not a run tree: the frozen-baseline jsons (`frozen_sam3d_smplx*.json`, `round3_refiner_eval.json`) and the retired stage-1 body (`stage1_20260905_180319/`, nothing loads it any more). Gitignored, never deleted. |

## Architecture

1. **Backbone** DINOv3-H (bf16, frozen) → `[B,1280,32,32]`; with `data.embedding_cache` the
   loader emits the cached embedding and the backbone is skipped (frame JPEGs not decoded).
   With `data.pose_token_cache` (builds with NO learned decoder token: `model.contact` /
   `model.force` off, no cross_modal; needs smplx + refiner) the loader emits the cached frozen
   FINAL pose token (`features/pose_token`, one npz per scene, built by the GVHMR worktree's
   `scripts/data/precompute_pose_tokens.py`) and backbone + decoder are both skipped
   (`out["mhr"]` None; no image / mask / embedding loaded) — ~20x faster per batch, the token
   within ~0.5 % of the live pass (bf16 storage + TF32 noise), so evaluate such runs through a
   `pose_token_cache: false` twin config.
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
5. **Temporal refiner** (`model.refiner`; `docs/old/refiner.md`) — behind the per-frame body
   (trained jointly; the per-frame body is also lifted into the world by the network for the
   losses, `out["smplx_per_frame"]` keeps it when the refiner rewrites `out["smplx"]`). Without a
   refiner, `model.token_heads` reads contact / force / gravity off the pose token instead. World lift with `cam_from_world`, THEN Gaussian smoothing of the world
   pelvis position (`root_smooth_sec`; never in camera coordinates — those carry the camera's
   motion and the lift then fails to cancel it, the round-2 jitter source) and a shorter
   Gaussian on the world root rotation and the parent-local joint rotations (matrix mean
   projected onto SO(3), `project_rotation`; `pose_smooth_sec`, 0.08 s is MPJPE-neutral and
   removes most of the rotation jitter; `learn_smoothing` turns both widths into trainable
   log-sigmas — one for the root position, one per rotation (root + 21 joints) — in their own
   optimizer group at `optim.lr × optim.smoothing_lr_scale`, logged as `smoothing/*`), clip-mean betas →
   per-frame token = root-frame joint positions (FK of the smoothed pose) + body-frame root
   linear/angular velocity + frame spacing + betas + optional camera context (`camera_context`:
   pelvis→camera direction in the body frame, log depth, crop-box bearing and angular size) +
   projected pose token + projected contact tokens (`pose_token`,
   `pose_token_dim`, `contact_token_dim`) → `CrossModalRopeModule` with ONE slot (`dim`,
   `num_layers`, `num_heads`, `window` seconds per layer) → zero-init heads listed in
   `outputs`: `pose` (6D deltas right-multiplied onto the root and the 21 joints + a body-frame
   root shift), `contact` (replaces the decoder contact head), `motion` (body-frame vel / acc
   of the 22 joints + root angular vel / acc, `out["motion"]` with its `frame`), `force`
   (replaces decoder force tokens). FK in the world, mapped back into every camera → the
   output keeps the SmplxHead layout. **Frame independence is a design rule**: nothing that
   enters or leaves the transformer refers to the world frame (tested). At init the refiner is
   exactly the per-frame body + the input smoothing. `limb_tokens` / `force_tokens` add six
   contact / six force limb tokens (alternating per-slot temporal and within-frame attention).

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
| `contact_supervision` | six-group contact logits | confidence-weighted BCE, optional per-group `class_weights` (`positive` = false-negative penalty, `negative` = false-positive penalty; train labels are 78-81 % positive for hands, 60 % toes, 3.5 % heels); metrics f1 / precision / recall / iou (thr 0.5), `precision_at_r90`, per-group f1 |
| `force_supervision` | forces (root frame, bw) | on in-contact rows: vector Huber `force`, or the split `magnitude` (Huber on \|f\|) + `direction` (1 − cos on rows with \|f_gt\| ≥ `direction_min_bw`, predicted norm floored at 0.05 bw); + noncontact L1 + net force / torque vs kindyn GT; `force_confidence` ^ `confidence_power` row weights, `group_weights`; metrics mae, mag_mae, angle_deg, noncontact_mag |
| `motion_supervision` | the refiner's `motion` output, and (`loss.pose_*`) the finite differences of the REFINED pose | Huber on `vel` / `acc` / `ang_vel` / `ang_acc` divided by `scale` (GT RMS), vs kindyn world joints / root finite-differenced with `label_smooth_sec` Gaussian smoothing (rotated into the predicted body frame for the head; world frame for the pose derivatives, prediction side unsmoothed); metrics `<q>_rmse`, `<q>_pearson`, `pose_<q>_*` |
| `contact_consistency` | the refined extremities (wrists, toes, heels) | L1 on their world speed on limb-frames labelled in contact (label × confidence weights, gradient → pose path); metrics `speed`, `gt_speed` (the GT floor, ~0.13 m/s) |
| `force_consistency` | the refined motion + predicted forces (+ detached contact probs as gate) | BetterRobot RNEA root-wrench residual on the 22-joint BetterHuman SMPL-X (optional `smooth_sec` pre-smoothing, default 0 since the refined motion is smoothed at the refiner's input; manifold central differences, forces at the extremity joint origins, scene gravity); pseudo-Huber on the force (bw) and torque (bw·m) parts with separate weights; metrics `force`, `torque`, `gt_force`, `gt_torque` (kindyn GT under kindyn forces) |
| `smplx_supervision` | the SMPL-X head (or the refined body) | `kp2d` Huber on the full-frame projection (crop-normalized or bearing units), `kp3d` pelvis-relative, 6D MSE `orient` / `pose` / `hand_pose`, `betas` MSE, `cam` (CLIFF proxy), the crop-free ray anchors `depth` / `bearing`, or the clip-level WORLD root anchor `root_bias` (clip-mean error) / `root_shape` (per-frame deviation from it). Metrics (`metric_pose/*`, WHAM/GVHMR protocol): mpjpe / pa_mpjpe / pve / accel, pelvis_err / depth_err / depth_bias, dlogz_pred / gt / err, the camera-lifted GVHMR globals `lifted_wa_mpjpe100` / `lifted_w_mpjpe100` / `lifted_rte` / `lifted_jitter` + `gt_jitter`, hand_mpjpe / hand_pa_mpjpe |

### Invariants (do not break)

- **Freeze boundary = the wrapper.** Everything under `model.wrapper` is frozen and
  eval-pinned (`wrapper.train()` re-pins); everything else in `ContactAnything` trains. The
  checkpoint stores trainable tensors only and hard-fails with a name/shape diff on mismatch.
- **Mask invariant**: original tokens never attend appended blocks; the frozen MHR readout
  never changes. The pose output is the SMPL-X head, which reads the (mixed) pose token, or —
  under `model.refiner` — the refined body derived from it.
- **Refiner frame independence**: inputs and outputs of the temporal transformer are
  root-/body-frame quantities only (`tests/test_refiner.py::test_world_frame_independence`).
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

Sections: `model.{checkpoint_path, mhr_model_path, decoder_bf16, decoder_checkpointing, contact,
force, cross_modal_temporal, smplx, token_heads, refiner}` (`decoder_checkpointing`
recomputes the frozen decoder's layers in the backward — ~60 MB/frame instead of ~105;
`decoder_bf16` runs it under bf16 autocast with fp32 MHR / keypoint readouts — measured
harmless, 0.01 mm, but also useless: no speed, little memory), `data.{datasets, embedding_cache, pose_token_cache,
frames_per_batch, num_workers, seed, clip.{frames, stride, jitter}, interleave_videos,
eval_max_frames}` (`interleave_videos` deals an epoch's clips out round-robin over the source
videos so a step's global batch spans as many videos as clips), the six loss sections
(`motion_supervision` needs a refiner `motion` output, its `loss.pose_*` and
`contact_consistency` a refiner `pose` output, `force_consistency` the `force` output (with no `pose` output the residual regularises the forces on the fixed body);
every refiner output needs its loss enabled — DDP has no unused-parameter tolerance),
`smplx_supervision.loss` = the per-frame heads' body, `.refined` = the refined one; `force_consistency` with
every weight 0 = the residual metrics only), `optim.{lr, weight_decay, epochs, accumulate_steps, warmup_steps, lr_min, grad_clip, betas, ema}`
(`accumulate_steps` micro-batches per optimizer step; no decay on 1-d params and per-module
clipping are fixed behaviour), `output.{dir, exp_name, log_freq, save_freq, eval_every, monitor,
frozen_metrics}`.

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

`docs/old/results.md` holds every recorded number. Headlines (108-scene test, one clip per
person, 120-frame cap): frozen SAM3D refit 61.1 mm MPJPE / 44.1 PA / accel 11.9; the
`baseline.yaml` recipe (as run `hands`, 5 epochs) 61.0 / 42.0 / F1 0.919; the per-frame SMPL-X
probe 57.6 / 38.9 (no temporal block). On the 16-scene static subset the lifted-trajectory
jitter is 126 for the frozen model vs a GT floor of 6.35; the best non-smoothing run reached 52.
The temporal block over image tokens never learned to denoise the per-frame pose (see
`docs/old/history/`), which is why the pipeline pivoted to the two-stage refiner — its results live
in `docs/old/refiner.md`.

## Conventions

- Never `rm`: move to `/home/rikhat.akizhanov/better/trash/<name>_<date>/` (`../trash/`). Commit/push only on
  explicit instruction. Shared GPU box: never touch other users' processes.
- `output_6/` holds run directories only; every console transcript or launch log goes to
  `output_6/logs/` (the train/evaluate CLIs tee themselves there).
- Skepticism: log raw metrics and trajectories; the user owns verdicts. No validation split
  (train on all train scenes, evaluate on test only).
- Keep the codebase pruned: no `.get()` fallbacks on schema-guaranteed keys, no history
  narration in comments, no options nobody runs.
