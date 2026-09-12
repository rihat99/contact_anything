# The two-stage pipeline: per-frame body, world-space temporal refiner

*Concept page. Configs: `configs/stage1.yaml`, `configs/stage2.yaml` (round 1; 2026-09-08: in the trash with `stage2_v2*.yaml`; the round-3/4 configs
and every `output/stage2_*` run followed on 2026-09-11, `/data3/rikhat.akizhanov/trash/cleanup_20260911/` — the round-3 `eval.json` stays as `output/round3_refiner_eval.json`, the final model's tensorboard reference),
`configs/stage2_v2.yaml` (round 2). Code: `model/refiner.py`, `model/loss/motion.py`,
`model/loss/contact_consistency.py`. Measurement scripts: `scripts/dump_stage1.py`,
`scripts/analyze_stage1.py`.*

## Why the pivot (2026-09-05)

Every 2026-08/09 attempt to improve the per-frame pose with a temporal model over the frozen
decoder's image tokens failed the same way: the tokens carry no velocity information beyond the
pose readout, a temporal block over them degenerates to clip pooling, and velocity losses on the
pose path collapse into shrinkage (`docs/old/history/`). The per-frame SMPL-X head on the frozen pose
token remains the best pose model we have (57.6 mm MPJPE vs 61.1 for the frozen MHR refit). Its
one large, structured error is the pelvis depth, which jitters frame to frame (lifted jitter ~110
against a GT of ~7, 10 m/s³ units), and that noise is nearly removed by smoothing the depth alone.

So the pose problem is split. **Stage 1** is that per-frame model, trained once and frozen.
**Stage 2** never looks at pixels again for the pose: it lifts the per-frame bodies to the world
with the known camera motion, smooths the depth, and runs a small temporal transformer over the
resulting world-space motion — a *fixer* for the pose and the natural place to read contact,
velocity, acceleration and contact forces, which are all properties of the motion, not of a frame.

## Stage 1 — the per-frame body (`configs/stage1.yaml`)

Frozen SAM 3D Body (DINOv3-H backbone + promptable decoder) with two from-scratch heads on the
final pose token: an SMPL-X head (root orientation, 21 body joints, 30 finger joints, 10 betas as
6D / raw residuals on a fixed mean) and a CLIFF camera head (crop weak-perspective `(s, tx, ty)`
lifted to full-image metres with the crop box and the focal). No contact tokens — contact is a
stage-2 output — and no temporal block. Training samples single frames, every 5th source frame
with per-epoch jitter, 64 frames per GPU (the 2026-09-02 probe recipe), on all 864 train scenes.

## Stage 2 — the refiner (`configs/stage2.yaml`, `model/refiner.py`)

The frozen decoder runs live with the six learned contact tokens appended (they need gradients),
the stage-1 heads are loaded from their checkpoint and frozen (`model.smplx.checkpoint`,
`frozen: true`), and the refiner consumes the per-frame body, the pose token and the contact
tokens of a clip.

### Frame independence — the design rule

Nothing that enters or leaves the temporal transformer refers to the world frame. Inputs are
root-frame joint positions and body-frame root velocities; outputs are body-frame / parent-local
corrections and body-frame vectors. Re-defining the world by any rigid transform leaves every
camera-frame and body-frame output bit-for-bit unchanged and moves the world outputs rigidly
(`tests/test_refiner.py::test_world_frame_independence`). The world frame is only used to *carry*
the motion between cameras.

### Steps inside the forward

1. **Depth smoothing.** The pelvis log depth is Gaussian-smoothed along time in camera coordinates
   with `depth_smooth_sec`; the bearing `(x/z, y/z)` is kept, so the body slides along its own
   image ray. Only the pelvis position changes; the body shape relative to the pelvis is untouched.
2. **Lift.** `p_w = R^T (p_c − t)`, `R_world_root = R^T R_cam_root` with the frame's
   `cam_from_world`. Betas are averaged over the clip (one body per person).
3. **Per-frame token.** Two LayerNorms (geometry and projected tokens separately), concatenation,
   linear to `dim`: the 21 non-root body-joint positions in the root frame (63), the root's
   linear velocity in the body frame (3) and angular velocity in the body frame (3) from finite
   differences of the lifted trajectory, the frame spacing in 25-fps frames (1), the mean betas
   (10), the pose token projected 1024→256, the six contact tokens projected 1024→64 each (384).
   No gravity direction, no heading, no absolute position (user decision: the model must work
   from motion alone).
4. **Temporal transformer.** The RoPE block of `model/rope.py` with one slot per frame: positions
   are the frames' real elapsed seconds, attention is masked to `±window` seconds per layer
   (default 0.5 s over 4 layers → a receptive field of about ±2 s), bidirectional, invalid frames
   hidden. Frames outside the horizon provably cannot influence a frame
   (`test_receptive_field_is_local`).
5. **Heads**, each a two-layer MLP with a zero-initialised last linear:
   * `pose` — 6D rotation deltas right-multiplied onto the root (body frame) and onto the 21 body
     joints (parent-local), plus a root position delta expressed in the body frame;
   * `contact` — six logits (kindyn groups LH, RH, LF toe, RF toe, LA heel, RA heel);
   * `motion` — world velocity and acceleration of the 22 joints and angular velocity /
     acceleration of the root, all expressed in the input body frame;
   * `force` — six 3D forces, body-weight units, in the same input body frame; the force loss
     rotates the kindyn GT (given in the GT root frame) and its lever arms into that frame with
     `frame^T R_gt_root`, a world-independent relative rotation.
   Finger rotations pass through unchanged.
6. **Decode.** FK in the world with the mean betas, then into every camera with the extrinsics.
   The output has the `SmplxHead` layout (`q_cam`, `joints_cam`, `pelvis_cam`, 2D projections),
   so the existing SMPL-X loss, all pose metrics, `predict_test.py` and the viewer apply unchanged.

At initialisation the refiner is "stage 1 + depth smoothing + clip-mean betas": the RoPE blocks
are identities and every head is zero, but the betas are already averaged over the clip (a
2–3 mm effect on the joints in the review's measurement). Evaluating the untrained stage-2 model
(`scripts/evaluate.py --checkpoint none`) gives that reference row for free; the raw stage-1 row
comes from evaluating the stage-1 checkpoint itself at `stride: auto`.

### Supervision (round 1: plain supervised terms only)

| loss | target | notes |
|---|---|---|
| `contact_supervision` | six-group kindyn contact labels | confidence-weighted BCE, unchanged |
| `force_supervision` | kindyn GT forces, root frame, bw | Huber + non-contact L1, unchanged |
| `smplx_supervision` | kindyn SMPL-X body in each camera | on the REFINED body; `kp2d 0`, `kp3d 5`, `orient 1`, `pose 1`, `pelvis 1`; `betas` and `cam` 0 (the frozen head's clip-mean betas carry no gradient) |
| `motion_supervision` | finite differences of the kindyn world joints / root | Gaussian label smoothing σ 0.12 s; GT rotated into the predicted body frame; standardized by `scale`; Huber |

No consistency, physics or smoothness terms. The motion GT smoothing follows the 2026-08 finding
that raw kindyn derivatives are too noisy to learn from; the acceleration is the smoothed
derivative of the smoothed velocity. The smoothing weights by the derivative's own support (both
neighbours valid — a forced-zero derivative next to a hole must not leak into its neighbours), and
rows within `ceil(2σ/dt)` frames (6 at 25 fps) of a clip end or a hole are not supervised because
their kernel is truncated. The prediction's body frame enters both the motion and the force loss
detached, and the config layer requires the stage-1 head to be frozen under the refiner: a
trainable pose path under these losses would rediscover the velocity-shrinkage shortcut.

### Operational notes

* A frozen stage-1 head is **not** stored in stage-2 checkpoints; `model.smplx.checkpoint` is
  re-read on every load (the builder also checks that the checkpoint's head definition — camera
  type, hands, body model — matches the config, since a `cliff` and a `ray` head have identical
  parameter shapes). Keep the stage-1 run directory, and do not retrain it in place.
* `scripts/predict_test.py` tiles long scenes into 240-row windows; depth smoothing and the
  clip-mean betas are per window, so the exported body can step at window seams. The evaluation
  protocol (one clip per person, 120-row cap) is unaffected.
* The motion head and the pose head are not tied by any consistency term in round 1, so
  `out["motion"]` is a separate estimate, not the derivative of the refined pose.

### Known risk: stage-1 leakage

The refiner trains on stage-1 predictions of scenes stage 1 was trained on, which are cleaner than
its test predictions. `scripts/dump_stage1.py` + `scripts/analyze_stage1.py` measure the train/test
gap of the per-frame model (MPJPE, pelvis / depth error, lifted jitter) before stage 2 is launched;
a 2-fold stage 1 is the fallback if the gap is material.

## Round 2 (`configs/stage2_v2.yaml`, 2026-09-05 evening)

Round 1 left three facts: the pose head was inert (a per-frame Huber does not reward denoising),
the batch was 4 clips per step, and the linear motion numbers trailed the angular ones. Round 2
changes the inputs, the objective and the optimisation, in that order of expected effect.

### Smoothing everything, not only the depth

Round 1 smoothed the pelvis depth alone. The sweep below (`scripts/analyze_stage1.py
--pose-sigmas`, on the stage-1 dumps, depth already smoothed at 0.25 s) smooths the world root
rotation, the root-frame joint positions (a proxy for the joint rotations) and the camera bearing
with one Gaussian σ:

| σ (s) | train mpjpe | train pelvis | train accel | train jitter | test mpjpe | test pelvis | test accel | test jitter |
|---|---|---|---|---|---|---|---|---|
| 0 | 49.38 | 85.4 | 12.39 | 66.3 | 58.88 | 113.2 | 11.56 | 70.2 |
| 0.03 | 48.69 | 85.2 | 5.39 | 28.8 | 58.30 | 113.0 | 4.87 | 36.6 |
| 0.05 | 48.40 | 85.2 | 3.97 | 24.0 | 57.99 | 112.9 | 3.39 | 32.0 |
| **0.08** | 48.56 | 85.3 | 3.57 | 22.9 | **57.97** | 113.0 | **2.91** | **30.8** |
| 0.12 | 49.81 | 86.2 | 3.62 | 22.6 | 58.79 | 113.7 | 2.91 | 30.5 |
| 0.16 | 51.89 | 87.9 | 3.73 | 22.5 | 60.33 | 115.2 | 3.00 | 30.4 |
| 0.25 | 58.08 | 93.8 | 3.88 | 22.5 | 65.23 | 120.3 | 3.15 | 30.4 |

(camera-frame MPJPE mm, world pelvis mm, acceleration error m/s², lifted jitter 10 m/s³.) The
per-frame rotation noise is white and much faster than the motion: σ 0.05–0.08 s *improves* MPJPE
by ~0.9 mm and cuts the acceleration error 4× and the lifted jitter from 70 to 31 (GT 7); beyond
0.12 s real motion is blurred. `pose_smooth_sec: 0.08`. In the refiner the joint and root
rotations are smoothed as matrices (Gaussian mean projected back onto SO(3) with a scaled Newton
polar iteration — `project_rotation`; the batched GPU SVD of a Procrustes projection took seconds),
the bearing in camera coordinates, and the input joints are the FK of the smoothed pose.

### Camera context in the token

Round 1's token was motion-only. Round 2 appends seven frame-independent numbers: the direction
from the pelvis to the camera in the body frame (where the per-frame depth error points), the
pelvis log depth, and the crop box's bearing and angular size (`(c − p) / f`, `b / f` — the inputs
of the CLIFF lift). They exist so the refiner can express a ray-aligned depth correction and learn
the per-scene depth bias as a function of the crop geometry (the stage-1 pelvis error, 114 mm, is
mostly that bias). Camera *motion* needs no extra input: the root velocity is computed after the
world lift, and the change of the camera direction across frames is visible to the transformer.
The world-frame-independence test covers the new features (the camera moves with the world).

### Root position: relative to what?

The refiner's root output stays a body-frame position delta on top of the smoothed stage-1
pelvis — relative to the model's own per-frame estimate, which is anchored to the camera through
the CLIFF lift — supervised by the absolute camera-frame pelvis Huber and, new, by the derivative
matching below. The alternative (predict a body-frame root velocity and integrate it from a clip
anchor, WHAM/GVHMR style) decouples the trajectory shape from the per-frame depth noise but adds
drift and an anchor choice; it is deferred to a later arm (user decision).

### Derivative-level pose objectives

* **Velocity / acceleration matching of the refined trajectory** (`motion_supervision.loss.pose_*`):
  the raw central finite differences of the refined world joints and root (`pose_vel`, `pose_acc`,
  `pose_ang_vel`, `pose_ang_acc`) are matched — same targets, same standardisation, same masks — to
  the Gaussian-smoothed GT derivatives the motion head is trained on. The prediction side is NOT
  smoothed, so jitter in the refined trajectory is penalised directly; the position terms
  (`kp3d`, `orient`, `pose`, `pelvis`) keep the amplitude honest. This is the objective the 2026-09-04
  smoother round showed works on a lifted trajectory (108 → 9.7 jitter with MPJPE unchanged) and
  differs from the image-model velocity losses that collapsed into shrinkage: here the motion is an
  input, not something to be predicted from pixels.
* **Contact stillness** (`contact_consistency`): the world speed of the refined wrists / toes / heels
  on limb-frames labelled in contact, L1 weighted by label × confidence, gradient to the pose path
  only. GT gating (user decision): the corpus label IS motion-gated stillness — measured GT
  in-contact speed 0.13 m/s mean (rms 0.26, median 0.07) against 0.94 m/s free on 70 train scenes —
  and a wrong contact prediction can then never damp real motion. Metrics `speed` / `gt_speed`.

### Batch, throughput and precision

* **Memory** (`scratch probe, GPU 5, T = 60`): fp32 240 frames/GPU → 28.3 GiB peak, 174 frames/s;
  bf16 autocast of the decoder → 23.5 GiB, same speed; fp32 + per-layer activation checkpointing
  (`model.decoder_checkpointing`) → 17.7 GiB at 240 frames, ~60 MB/frame (the stored layer inputs
  are dominated by the 1024 × 1280 image context of the two-way attention), 600 frames overflow the
  card. bf16 + checkpointing at 480 frames: 31.8 GiB, 166 frames/s. The step is CPU-launch-bound
  (744 ms of GPU work in a 1.4 s step at 240 frames), so larger per-GPU batches raise throughput.
* **bf16 degradation** (untrained stage 2, 30 test scenes, stage-1 body): MPJPE 59.567 → 59.577,
  PA 40.374 → 40.378, pelvis 117.84 → 117.89 mm, lifted jitter 64.83 → 64.84 — 0.01 mm, negligible,
  but bf16 also bought nothing (no speed, ~17 % memory), so the run stays tf32 (fp32 weights).
* **Round 2 batch**: 8 clips × 60 frames per GPU × 4 GPUs × `optim.accumulate_steps: 2` =
  **64 clips per optimizer step** (round 1: 4), from 64 distinct source videos whenever the epoch
  still has that many (`data.interleave_videos`: the corpus has 84 train videos for its 864 scenes;
  the sampler deals each video's shuffled clips out round-robin, and a step's global block takes
  consecutive positions of that stream). 60-frame clips (2.4 s) double the clip count of round 1's
  120-frame clips for the same frame budget; the block's receptive field (3 layers × ±0.5 s) still
  fits inside a clip, and evaluation still runs whole 120-row scenes.
* Block depth 3 (user), window 0.5 s per layer; lr 3e-4 (round 1: 2e-4 at 1/16 of the batch),
  100 warm-up optimizer steps, 50 epochs (~8.5 min per epoch incl. eval); a second 50-epoch phase with the force-consistency loss follows (user plan).

**Untrained round 2** (smoothing + camera context, all heads zero; reference protocol,
scratch `eval/stage2_v2_untrained.json`): MPJPE **55.47** / PA **38.00** / PVE 71.60, accel 2.76,
pelvis 114.2, lifted jitter **25.1** — already better than round 1's trained model on every pose
metric (55.97 / 38.37 / 72.26, jitter 63.4). The pose smoothing is worth more than round 1's
training was.

**First launch (original balance, run `stage2_v2_20260905_231055`, stopped after 3 epochs and
trashed; a 30-epoch launch before it was aborted in its first epoch):** kp3d 5 / pose 1 / pelvis 1,
pose_vel 1 / pose_acc 1 / pose_ang_vel 1 / pose_ang_acc 1, contact_consistency 1. Test MPJPE
59.38 → 58.07 → 57.48 over epochs 0–2 (PA 40.8 → 40.2 → 40.05), i.e. 2–4 mm WORSE than the
untrained model; contact F1 0.852 → 0.901, lifted jitter 25.8 → 25.4, motion head ang_vel Pearson
0.93 by epoch 2. The pose-head diagnostic on the epoch-2 checkpoint (20 scenes) confirmed the head
as the cause: head ON 61.9 / PA 41.6, head OFF 59.9 / 39.8; the corrections it emitted were 3.7°
mean / 16.6° max per frame on the joints (round 1: 1.2° / 3.5°). On a training batch the weighted
terms were kp3d 0.052, pelvis 0.117, pose 0.016 against pose_acc 0.48, pose_vel 0.10, contact
stillness 0.24: the raw second difference of the prediction against the smoothed GT acceleration
dominated the head's objective and rewards damping, at the cost of articulation.

**Rebalanced launch (run `output/stage2_v2_20260905_234616` — THE round-2 run):** kp3d 10 / pose 2
/ pelvis 2 (position anchors ×2), pose_acc 0.2 / pose_ang_acc 0.5 (pose_vel / pose_ang_vel kept at
1), contact_consistency 0.5; everything else unchanged (`configs/stage2_v2.yaml`, GPUs 0/4/5/6,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, console `output/logs/stage2_v2_console.log`).

### Round-2 result (run `stage2_v2_20260905_234616`, 50 epochs, 57 optimizer steps of 64 clips each)

Per-epoch test trajectory (reference protocol):

| epoch | contact f1 | P@R90 | force mae | mpjpe | pa | pelvis | lifted jitter | motion vel r | acc r | ang_vel r | ang_acc r | pose_vel r | pose_acc r | contact speed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| untrained | – | – | – | 55.47 | 38.00 | 114.2 | 25.1 | – | – | – | – | – | – | – |
| 0 | 0.852 | 0.780 | 0.291 | 56.33 | 38.40 | 113.5 | 25.3 | 0.26 | 0.16 | 0.55 | 0.27 | 0.85 | 0.52 | 0.235 |
| 4 | 0.921 | 0.927 | 0.227 | **55.86** | 38.34 | 113.7 | 25.2 | 0.74 | 0.58 | 0.94 | 0.89 | 0.85 | 0.51 | 0.232 |
| 12 | **0.930** | **0.941** | 0.211 | 55.93 | 38.50 | 113.6 | 25.4 | 0.83 | 0.71 | 0.96 | 0.91 | 0.86 | 0.51 | 0.229 |
| 20 | 0.926 | 0.937 | 0.207 | 55.97 | 38.54 | 112.8 | 22.5 | 0.84 | 0.76 | 0.96 | 0.91 | 0.87 | 0.62 | 0.211 |
| 30 | 0.920 | 0.932 | 0.205 | 56.02 | 38.58 | 112.6 | 21.0 | 0.86 | 0.78 | 0.96 | 0.92 | 0.88 | 0.65 | 0.206 |
| **49** | 0.919 | 0.929 | **0.205** | 56.06 | 38.61 | **112.5** | **20.5** | **0.86** | **0.79** | **0.96** | **0.92** | **0.88** | **0.66** | **0.204** |

(`best.pth` = epoch 4 by the MPJPE monitor; `last.pth` = epoch 49, the model phase 2 starts from.
GT floors: lifted jitter 7.2, in-contact speed 0.176 m/s.)

Reading:

* **Pose**: MPJPE stays within 0.6 mm of the untrained floor for the whole run (55.86–56.06 vs
  55.47) — the rebalanced head no longer damages articulation, but it does not improve positions
  either; what it learns is denoising: lifted jitter 25.3 → 20.5 (round 1: 63.4, raw stage 1: 109),
  the pose-derivative acceleration Pearson 0.52 → 0.66, and the in-contact extremity speed
  0.235 → 0.204 m/s (GT 0.176). The denoising only started around epoch 14, once the heads had
  settled, and was still creeping at epoch 49.
* **Contact**: peaks at epoch 12 (F1 0.930 / P@R90 0.941, above round 1's 0.925 / 0.937) and then
  over-fits slowly (0.919 at 49; the test contact BCE rises from epoch 12 on) — the classic
  large-batch / long-schedule pattern for the BCE head. For a contact-first checkpoint use
  `epoch_0010.pth`.
* **Forces**: MAE 0.205 bw (round 1: 0.216), still improving at the end.
* **Motion head**: velocity / acceleration Pearson 0.86 / 0.79 (round 1: 0.81 / 0.70), angular
  0.96 / 0.92 — the smoothed inputs and the larger batch lift the linear quantities the most.

### Phase 2: force consistency (`configs/stage2_v2_force.yaml`, `model/loss/force_consistency.py`)

After the 50-epoch round-2 run, the same model is warm-started from its best checkpoint
(`model.warm_start`: trainable weights only, fresh optimizer and a 50-epoch cosine at lr 1e-4) and
trained with one more loss: the inverse-dynamics residual of the refined motion under the predicted
forces. BetterRobot's RNEA on the 22-joint BetterHuman SMPL-X (per-part mesh inertias baked per clip
from the clip-mean betas, ~72.6 kg neutral) gives the root wrench the refined trajectory requires
under the scene's fitted gravity — `q` from the refined world pelvis / root rotation / joint
rotations (Gaussian-smoothed at `smooth_sec` 0.12 s first, rotations as SO(3)-projected matrix
means), `v` / `a` by manifold central differences at the real frame spacing — minus the six
predicted forces (body weights → newtons by `m g`, gated by the detached predicted contact
probability, rotated into each extremity joint's local frame and applied at the joint origin:
wrists, big toes, heels). The residual's force part is in body weights, its torque part in bw·m;
each goes through a pseudo-Huber with its own weight (torque × 5: the allocation channel, measured
~23× weaker than the force sum in the 2026-07 physics runs). Gradient reaches the force head and
the pose path (user decisions: joint origins, forces + pose, 22-joint body, warm start). Metrics:
the mean residual magnitudes of the prediction and, as the floor, of the kindyn GT motion under
the kindyn GT forces (`gt_force` / `gt_torque`). Tests: `tests/test_force_consistency.py` (a body
at rest needs one body weight at the root; a body weight at a toe balances it; rigid world
independence; gradients).

Smoke (8 scenes, 1 epoch, warm-started from run 1's epoch-1 best): residual force 0.48 bw /
torque 0.095 bw·m against a GT floor of 0.27 / 0.071 — the kindyn GT itself does not balance under
this body model and the joint-origin forces, so the floor, not zero, is the target. At weights
1 / 5 the two terms were ~1.7 % of the total loss (0.052 + 0.017 of 4.02), so the phase-2 config
uses 5 / 25 (≈ the motion terms' share).

### Phase-2 result (run `stage2_v2_force_20260906_063633`, stopped by the user at epoch 22 of 50)

Warm start from run 1's `last.pth`, lr 1e-4, physics weights 5 / 25. Test trajectory (reference
protocol; residuals in bw / bw·m, `res_gt_*` = the kindyn GT under the kindyn forces):

| epoch | f1 | P@R90 | force mae | off-contact |f| | mpjpe | pa | pelvis | lifted jitter | motion vel r | acc r | pose_acc r | contact speed | res force | res torque | res gt force | res gt torque |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run 1 final | 0.919 | 0.929 | 0.205 | 0.019 | 56.06 | 38.61 | 112.5 | 20.5 | 0.86 | 0.79 | 0.66 | 0.204 | – | – | – | – |
| 0 | 0.918 | 0.928 | 0.202 | 0.025 | 56.05 | 38.63 | 112.5 | 20.5 | 0.86 | 0.79 | 0.66 | 0.204 | 0.267 | 0.071 | 0.206 | 0.065 |
| 2 | 0.916 | 0.926 | 0.201 | 0.027 | 56.07 | 38.66 | 112.6 | 20.5 | 0.86 | 0.79 | 0.66 | 0.204 | 0.248 | 0.064 | 0.206 | 0.065 |
| 5 | 0.920 | 0.930 | 0.199 | 0.028 | 56.07 | 38.64 | 112.4 | 20.4 | 0.87 | 0.79 | 0.67 | 0.203 | 0.237 | 0.060 | 0.206 | 0.065 |
| 10 | 0.918 | 0.928 | 0.200 | 0.029 | 56.06 | 38.64 | 112.4 | 20.3 | 0.87 | 0.79 | 0.67 | 0.203 | 0.232 | 0.058 | 0.206 | 0.065 |
| 20 | 0.919 | 0.927 | 0.199 | 0.029 | 56.06 | 38.64 | 112.3 | 20.2 | 0.87 | 0.80 | 0.67 | 0.202 | 0.231 | 0.056 | 0.206 | 0.065 |
| **22** | 0.918 | 0.927 | **0.199** | 0.029 | 56.06 | 38.64 | **112.2** | **20.2** | **0.87** | **0.80** | **0.67** | **0.202** | **0.230** | **0.056** | 0.206 | 0.065 |

Reading:

* The physics residual falls from 0.267 → 0.230 bw (force) and 0.071 → 0.056 bw·m (torque) and
  plateaus from epoch ~10. The torque part goes BELOW the GT floor (0.065) from epoch 2 on — the
  model finds force allocations more consistent with its own (smoothed) motion than the kindyn
  solution is with the kindyn motion under this body model; the force part stays 12 % above its
  floor (0.206), which is the level the joint-origin approximation and the 22-joint mass model
  allow.
* What it buys elsewhere is small but consistent: force MAE 0.205 → 0.199 bw, pelvis error
  112.5 → 112.2 mm, lifted jitter 20.5 → 20.2, motion head acceleration Pearson 0.79 → 0.80; the
  off-contact force magnitude rises 0.019 → 0.029 bw (the residual wants some force wherever the
  gate leaks). Pose (MPJPE 56.06) and contact (F1 0.918) are untouched.
* Run stopped at epoch 22 (user) once the residual had plateaued; `last.pth` = epoch 22, evaluated
  in `eval.json` with predictions in `predictions/`.

## Round 3 (`configs/stage2_v3.yaml`, 2026-09-06): the root in the world frame

### Where round 2's residual jitter came from

Oracle swap decomposition on the 108 test scenes (cap-120 protocol, GVHMR jitter 10 m/s³;
scripts in `~/.claude/handoffs/jitter_decomp_20260906/`, dumps `output/stage1_*/dump_test` and
`output/stage2_v2_force_*/predictions`). World joints `J = p + R X`; each of the root position
`p`, root rotation `R` and root-frame articulation `X` is taken from the prediction or the GT:

| variant | stage 1 raw | round 2 phase 2 |
|---|---|---|
| everything predicted | 105.5 | 20.3 |
| GT root position, predicted rotation + articulation | 48.9 | 5.6 |
| only root position predicted | 89.9 | 22.0 |
| only camera depth of the root predicted | 85.1 | 19.1 |
| only camera bearing of the root predicted | 23.5 | 12.0 |
| only root rotation predicted | 36.0 | 7.2 |
| only articulation predicted | 44.3 | 6.5 |
| GT everything (floor) | 6.5 | 6.5 |

Rotation and articulation are solved (slightly over-smoothed); the root position carries all of
the residual. Split by camera type, root-position-only variants with GT rotation and articulation:

| root position source | static | moving | pelvis error, moving (mm) |
|---|---|---|---|
| GT | 5.2 | 6.7 | 0 |
| GT + a constant camera-frame bias | 5.2 | 6.8 | – |
| GT depth smoothed 0.25 s in camera coordinates | 8.3 | 25.9 | 17.8 |
| GT bearing smoothed 0.08 s in camera coordinates | 8.4 | 14.8 | 5.4 |
| stage-1 root smoothed in camera coordinates (the round-2 refiner input) | 10.6 | 29.9 | 101.0 |
| round-2 output root | 9.8 | 24.1 | 98.5 |
| raw stage-1 root smoothed 0.25 s in WORLD coordinates | 5.0 | 6.3 | 105.9 |
| raw stage-1 root smoothed 0.10 s in WORLD coordinates | 5.4 | 6.7 | 103.4 |

The camera-frame depth of the body contains the camera's own motion (GT joints in camera
coordinates: jitter 29.2 on moving scenes vs 6.7 in the world; the camera centre itself 31.4).
Smoothing the depth before the lift strips that motion from the signal, and the lift then
re-applies its inverse, which no longer cancels — even the perfect GT depth smoothed that way
lands at 26. The same raw root smoothed AFTER the lift sits at the floor at the same absolute
error. A constant camera-frame bias is harmless (camera rotation is smooth). The round-2 head
recovered a fifth of the damage (29.9 → 24.1) because its only lever is a residual on the
already corrupted root.

The stage-1 depth error is not white (autocorrelation 0.88 at lag 1, 0.46 at lag 10, 0.05 at
lag 25); its slow part carries no jitter. The root error of the round-2 output, split per clip
into a mean and the per-frame wander around it, along and across the camera ray (RMS mm):

| | along the ray | across |
|---|---|---|
| clip-mean error (bias) | 103 | 16 |
| fluctuation about the clip mean | 88 | 26 |
| GT root motion about its clip mean | 183 | 235 |

so the error is depth, half bias and half slow wander, and a third of the frames sit past the
0.1 m knee of the per-frame pelvis Huber, where the term is L1 and zero-mean fluctuations cancel
to first order. That loss neither created the jitter nor could remove it, and it barely moved the
depth (bias 106 → 104, wander 94 → 92 mm over 72 epochs).

### The change

* **Lift, then smooth** (`root_smooth_sec` 0.10 s on the world pelvis; `depth_smooth_sec` and the
  camera-coordinate bearing smoothing are gone). The root rotation and joint rotations were
  already smoothed after the lift (`pose_smooth_sec` 0.08 s, unchanged). The camera context's log
  depth is that of the smoothed root re-projected into the camera.
* **Two-scale world root anchor** replacing the per-frame camera-frame `pelvis` Huber: per clip
  and person, `e_t = p_pred − p_gt` in world, `root_bias` = Huber(|mean_t e_t|, knee 0.1 m) and
  `root_shape` = Huber(|e_t − mean_t e_t|, knee 0.02 m), both counted per supervised frame,
  weights 2 / 2. The shape term is quadratic at the wander's scale and is not blinded by the
  bias; the along-ray SNR (183 vs 88 mm) puts its minimum at the GT shape, not at a damped one
  (watch the predicted / GT deviation RMS along the ray as the shrinkage diagnostic).
* Everything else as round 2 (64 clips per step, lr 3e-4, camera context, 3 × 0.5 s); 10 epochs
  as a quick check, `frozen_metrics` = round 2's final eval (the row to beat on tensorboard).
  The round-1/2 yamls no longer validate (retired keys); their runs keep their own `config.yaml`.

**Untrained round 3** (reference protocol, `output/stage1_20260905_180319/eval_stage2_v3_untrained.json`):
MPJPE 55.47 / PA 38.00 / PVE 71.60 (= round 2 untrained: the articulation path is untouched),
lifted jitter **2.75** (round 2 untrained 25.1, trained 20.2; GT 7.2 — now over-smoothed rather
than jittery), pelvis 116.8 / depth 110.3 mm (round 2 untrained 114.2 / 107.8: isotropic world
smoothing averages the along-ray error 2.6 mm less well than the along-ray camera smoothing did),
`dlogz_pred` 0.49 vs GT 0.47 (the camera-frame depth step now matches the GT's — the camera
motion is preserved), wa_mpjpe100 78.0, rte 4.35. Loss values on the test set: root_bias 0.106,
root_shape 0.129 (kp3d 0.103).

### Round-3 result (run `output/stage2_v3_20260906_124223`, 10 epochs, 57 steps of 64 clips each, GPUs 0–3)

Per-epoch test trajectory (reference protocol; console `output/logs/stage2_v3_console.log`, final
`eval.json` via `output/logs/stage2_v3_final_eval.log`):

| epoch | contact f1 | P@R90 | force mae | mpjpe | pa | pelvis | wa_mpjpe100 | lifted jitter | motion vel r | acc r | pose_acc r | contact speed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| untrained | – | – | – | 55.47 | 38.00 | 116.8 | 78.0 | 2.75 | – | – | – | – |
| 0 | 0.852 | 0.780 | 0.292 | 56.39 | 38.41 | 116.0 | 78.5 | 3.64 | 0.27 | 0.16 | 0.58 | 0.247 |
| 2 | 0.900 | 0.887 | 0.242 | **55.85** | **38.29** | 115.6 | 77.7 | 3.46 | 0.67 | 0.44 | 0.58 | 0.240 |
| 4 | 0.921 | 0.927 | 0.228 | 55.90 | 38.34 | 114.6 | 76.1 | 3.53 | 0.75 | 0.57 | 0.61 | 0.223 |
| 6 | 0.923 | 0.933 | 0.222 | 55.96 | 38.43 | 113.8 | 75.2 | 3.71 | 0.78 | 0.62 | 0.63 | 0.215 |
| 8 | 0.923 | 0.935 | 0.219 | 56.00 | 38.48 | 113.5 | 74.8 | 3.85 | 0.78 | 0.64 | 0.63 | 0.211 |
| **9** | **0.924** | **0.935** | **0.219** | 56.01 | 38.50 | **113.4** | **74.7** | 3.90 | **0.79** | **0.64** | **0.63** | **0.211** |
| round 2, epoch 4 (same steps) | 0.921 | 0.927 | 0.227 | 55.86 | 38.34 | 113.7 | – | 25.2 | 0.74 | 0.58 | 0.51 | 0.232 |
| round 2, epoch 49 (5× the steps) | 0.919 | 0.929 | 0.205 | 56.06 | 38.61 | 112.5 | 73.0 | 20.5 | 0.86 | 0.79 | 0.66 | 0.204 |

(`best.pth` = epoch 2 by the MPJPE monitor; `last.pth` = epoch 9, the evaluated model. The cosine
schedule ran to its end at epoch 9, so these are a completed short schedule, not a mid-run snapshot.)

Reading:

* **Jitter**: 3.9 at the end against 20.5 for round 2 and a GT floor of 7.2 — the lift-then-smooth
  input removed the camera-made residual entirely; the model now sits BELOW the floor (the 0.10 s
  root + 0.08 s rotation Gaussians blur some real motion), and training moved it back toward the
  GT's own roughness (2.75 → 3.9 over the run). The camera-frame depth step `dlogz_pred` ends at
  0.468 vs the GT's 0.468.
* **Root**: pelvis error 116.8 → 113.4 mm in 10 epochs (round 2: 114.2 → 113.7 at the same step
  count, 112.5 after 5× the steps); wa_mpjpe100 74.7 (round 2 final 73.0); rte 3.98 (3.83). The
  anchor split recovers the 2.6 mm the isotropic world smoothing had cost and was still moving at
  the last epoch; the ~100 mm depth wander itself remains.
* **Articulation**: MPJPE / PA within 0.5 mm of round 2 at every epoch; `best.pth` at epoch 2 =
  55.85 / 38.29.
* **Contact / force / motion**: at the same step count round 3 matches or beats round 2 on every
  head (F1 0.924 vs 0.921, force MAE 0.219 vs 0.227, pose-derivative acceleration Pearson 0.63 vs
  0.51); the motion head's own acceleration Pearson (0.64) trails round 2's 50-epoch 0.79 only by
  the schedule length.
* **Swap decomposition of the round-3 output** (same protocol as the round-2 table above, dumps in
  the run's `predictions/`): only root position predicted 7.1, only root rotation 7.3, only
  articulation 6.0, GT root position with predicted rotation + articulation 5.0, everything
  predicted 3.9, floor 6.5 — every component now sits at the GT floor (round 2: root position 22.0).
  Root-only by camera type: static 6.2 (floor 5.2), moving 7.3 (floor 6.7; round 2: 24.1). The
  all-predicted total is below every single-swap variant because the predicted parts are jointly
  smoother than the GT, i.e. the residual is over-smoothing, not noise.


## Round 4 (`configs/stage2_v4*.yaml`, 2026-09-06/07): five arms — results under "Results"

Round 3 with two changes, run as an ablation of five 15-epoch arms chained on GPUs 0/1/4/7
(64 clips per step; everything else — world-frame root smoothing 0.10 / 0.08 s, `root_bias` /
`root_shape`, derivative matching, `contact_consistency` 0.5, contact weight 5, the vector-Huber
force loss with non-contact L1 1 — is round 3):

| arm | config | differs from A by |
|---|---|---|
| A full | `stage2_v4.yaml` | — |
| B | `stage2_v4_nofc.yaml` | no RNEA force-consistency loss |
| C | `stage2_v4_nomotion.yaml` | no motion losses (neither the head prediction nor the pose-derivative matching), hence no `motion` output |
| D | `stage2_v4_noforce.yaml` | no force head, no supervised force loss, no RNEA |
| E | `stage2_v4_nosmooth.yaml` | no Gaussian smoothing of the lifted root / rotations before the temporal block |
| F | `stage2_v4_notokens.yaml` | no contact tokens in the SAM 3D Body decoder (the frozen upstream model untouched; the refiner token loses its 6 × 64 contact-token features); trained on the cached frozen pose token (`data.pose_token_cache`), added 2026-09-07 |
| G | `stage2_v4_window1.yaml` | the temporal block's receptive field cut to ONE frame (`window` 0.01 s, below every sampled frame gap): a per-frame refiner with arm A's token, smoothing and losses; added 2026-09-07 |

The chain script is `output/logs/stage2_v4_chain.sh`; console logs `output/logs/<exp>_console.log`.

Arm F's cache: with no learned decoder token the frozen pass is a fixed function of the
person-frame, so the loader reads its FINAL pose token from `features/pose_token` (built by the
GVHMR worktree's `scripts/data/precompute_pose_tokens.py`, one npz per scene, bf16 bits as
int16) and the network skips backbone + decoder (`out["mhr"]` is `None`). Against the live pass
(same cached embedding, same crop geometry — verified equal) the cached token differs by
~0.5 % of its norm (bf16 storage + TF32 shape noise), which moves single SMPL-X joints by up
to ~4 mm; one 480-frame train batch runs forward + backward in 0.11 s at 4.1 GiB instead of
2.1 s at 28 GiB. Arm F is evaluated through the LIVE pass (`stage2_v4_notokens_live.yaml`) so
its numbers are computed like the other arms'.

### Contact loss: heel class weight

Train label balance (`contacts_1`, 276 599 supervised rows per group, the supervised-row
positive rate / the neg:pos ratio):

| group | positive rate | neg : pos |
|---|---|---|
| left / right hand | 0.78 / 0.81 | 0.28 / 0.24 |
| left / right toe | 0.60 / 0.61 | 0.67 / 0.64 |
| left / right heel | 0.034 / 0.037 | 28.5 / 25.9 |

Test (manual annotation) is the same shape: heels 0.069 / 0.036 positive. A plain BCE learns
the heels as "never" (heel F1 exactly 0 in rounds 1–3). `contact_supervision.class_weights`
multiplies each row by `positive[g]` (a positive row: the false-negative penalty) or
`negative[g]` (the false-positive penalty), into numerator and mass. Round 4 sets the heel
positives to 5 and every other weight to 1: at threshold 0.5 that moves the heel decision to a
calibrated probability of ~0.17. The metrics stay unweighted. (The first attempts used
square-root balancing on all six groups — hands 1 / 2, toes 1 / 1.25, heels 5 / 1 — together
with a contact weight of 10; the user judged that rebalance wrong after three aborted
launches, below.)

### Force consistency without pre-smoothing

`force_consistency.smooth_sec` is 0: the refined motion is already smoothed at the refiner's
input (round 3's jitter 3.9 vs the GT's 7.2), so the RNEA differentiates the refined
trajectory directly. Round 4 enables the loss from the first epoch at round 2 phase 2's
weights (5 / 25); round 3 itself ran without it. In arm E nothing is smoothed anywhere, so the
RNEA (and the derivative matching) see the raw stage-1 motion: on the smoke batch the
untrained force residual is 1.79 bw against 0.81 with the smoothing, the pose-derivative terms
2–4× larger.

### Knobs built for round 4 and dropped (in `configs/base.yaml`, default off)

* **Force magnitude / direction split** — `force_supervision.loss.magnitude` (Huber on
  `|f_pred| − |f_gt|`, 0.5 bw knee) + `direction` (`1 − cos(f_pred, f_gt)` on rows with
  `|f_gt| ≥ direction_min_bw`, 0.1 bw: the direction of a near-zero GT force is solver noise —
  fraction of in-contact rows below 0.1 bw on train: hands 4–5 %, toes 17–19 %, heels
  35–45 %). The predicted norm inside the cosine is floored at 0.05 bw, which bounds the row
  gradient at 20 and, at the zero-initialised head, points it along the GT direction (the plain
  magnitude Huber has a zero gradient at a zero prediction). Metrics `mag_mae` / `angle_deg`
  are reported whenever the force loss runs.
* **`confidence_power`** — kindyn's per-frame `force_confidence` raised to an exponent before
  it weights the rows (numerator and mass). Train confidence is 0.987 at the median, 0.924 at
  p25, 0.538 at p5 (12.6 % of rows below 0.8), so the exponent only acts on the low tail (at
  0.5 a p5 row weighs 0.73 instead of 0.54).
* **Learnable smoothing widths** (`model.refiner.learn_smoothing`) — one log-sigma for the
  world root position and one per rotation (root + 21 joints), initialised from the two
  `*_smooth_sec`, in their own optimizer group at `optim.lr × optim.smoothing_lr_scale`, logged
  as `smoothing/*`. RETRACTED after attempt 2: the widths grew monotonically (root 0.10 →
  0.20 s, joints 0.08 → 0.13 / 0.16 s by epoch 7) — the derivative-matching, RNEA and stillness
  terms all get cheaper with more smoothing and the per-frame pose Huber cannot hold them —
  and MPJPE rose from its epoch-2 best 56.26 to 57.20 while the jitter fell to 1.5 (GT 6.9).
  Learnable widths need a cap or a counter-loss.

### The three aborted launches (2026-09-06, 19:37–21:45; runs in the trash)

All with the split force loss, `confidence_power` 0.5, the six-group class weights, RNEA
8 / 40 from epoch 0 and, in 1–2, the learnable widths (lr × 6).

* Attempt 1 (contact 5, non-contact L1 1, 50 epochs) killed after epoch 0. Its epoch-0
  contact numbers (F1 0.852 / P 0.76 / R 0.97) equalled round 3's epoch 0 exactly; the one
  difference was the non-contact force magnitude, 0.116 bw against round 3's 0.023 — the RNEA
  residual loads the free limbs and the L1 is the counterweight.
* Attempt 2 (contact 10, non-contact L1 5) killed at epoch 8: contact F1 0.886 / P 0.871 /
  R 0.901 at epoch 2 (heels 0.002 / 0.004), force MAE 0.228 bw / 23.5°, non-contact 0.050 bw,
  MPJPE 56.26 at epoch 2 then rising (the widths, above).
* Attempt 3 (widths fixed at 0.10 / 0.08 s, 30 epochs) stopped after epoch 1 (MPJPE 56.57)
  when the contact rebalance itself was judged wrong.

## Evaluation protocol

`scripts/evaluate.py` on the annotated test scenes, one clip per (scene, person), the longest valid
run at the `auto` stride capped at `data.eval_max_frames`. Stage-1 numbers in `docs/old/results.md`
style require `stride: auto` (the stage-1 training config strides its own per-epoch test clips by
5). Reference rows for every stage-2 result: stage 1 raw, stage 1 + depth smoothing + clip-mean betas
(the untrained refiner), then the trained refiner.

## Results

### Stage 1 — run `stage1_20260905_180319` (2 GPUs, 434 steps/epoch, ~2.5 min/epoch)

Per-epoch test metrics under the run's OWN protocol (whole test scenes at stride 5, 120-row cap;
`accel` at dt = 0.2 s is not comparable with the stride-auto tables). MPJPE / PA-MPJPE / PVE in mm:

| epoch | 0 | 1 | 3 | 5 | 8 | **10** | 12 | 15 | 19 |
|---|---|---|---|---|---|---|---|---|---|
| mpjpe | 89.9 | 69.8 | 61.4 | 59.8 | 59.2 | **59.13** | 59.17 | 59.28 | 59.40 |
| pa_mpjpe | 66.7 | 50.0 | 41.8 | 40.4 | 39.8 | 39.62 | 39.56 | 39.55 | 39.58 |
| pve | 114.8 | 88.4 | 76.9 | 75.4 | 74.8 | **74.69** | 74.75 | 74.89 | 75.08 |

`best.pth` = epoch 10 (MPJPE monitor). After epoch 10 the articulation (PA) keeps improving by
hundredths while MPJPE and PVE drift up by ~0.3 mm — the global orientation / camera part starts
to over-fit slightly; the cosine schedule ran to epoch 19 regardless. The run was interrupted once
at epoch 13 by an external SIGTERM and resumed exactly from `last.pth` (`--resume auto`).

**Reference protocol** (`scripts/evaluate.py`, stride auto, 120-row cap, 108 scenes —
`output/stage1_20260905_180319/eval_auto.json`, transcript `output/logs/stage1_dump_test_eval.log`):

| model | mpjpe | pa | pve | accel | pelvis_err | depth_err | lifted jitter | gt jitter | wa_mpjpe100 | w_mpjpe100 | rte | hand / pa |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen SAM3D refit (docs/old/results.md) | 61.07 | 44.06 | 78.08 | 11.93 | – | – | – | – | – | – | – | – |
| per-frame probe 2026-09-02 (docs/old/results.md) | 57.60 | 38.86 | 74.74 | 11.15 | – | – | – | – | – | – | – | – |
| **stage 1, epoch 10** | **56.28** | **38.49** | **72.61** | 11.07 | 119.9 | 113.7 | 109.2 | 7.2 | 82.5 | 138.3 | 4.73 | 32.7 / 3.9 |

### Stage-1 diagnostics (`scripts/analyze_stage1.py`, transcript `output/logs/stage1_analyze.log`)

Dumps: 150 train scenes (156 person-runs, 38.7k frames) and all 108 test scenes (109 runs, 30.0k
frames), every tracked frame at the auto stride, whole scenes (so the test numbers here are not the
120-row-capped protocol above).

**Train / test gap of the per-frame model** (camera frame, mm; jitter 10 m/s³):

| split | mpjpe | pelvis_err | depth_err | depth_bias | lifted jitter | gt jitter |
|---|---|---|---|---|---|---|
| train (in-sample) | 49.4 | 91.4 | 85.1 | −5.2 | 128.7 | 10.3 |
| test | 58.9 | 121.3 | 114.2 | +1.6 | 126.7 | 7.8 |

The systematic error is ~16 % (MPJPE) to ~25 % (depth) smaller on scenes stage 1 was trained on;
the frame-to-frame noise (jitter) is identical. So the refiner trains on inputs with the right
noise but a smaller bias than it will meet at test — it will tend to under-correct systematic
errors. Decision pending (user): accept for round 1, or 2-fold stage 1.

**Depth-smoothing sweep** (pelvis log depth, bearing kept; world mm; jitter 10 m/s³):

| σ (s) | train pelvis | train joints | train jitter | test pelvis | test joints | test jitter |
|---|---|---|---|---|---|---|
| 0 | 91.4 | 105.8 | 128.7 | 121.3 | 137.6 | 126.7 |
| 0.08 | 87.8 | 102.6 | 66.4 | 117.1 | 133.8 | 70.3 |
| 0.12 | 86.6 | 101.5 | 66.3 | 115.5 | 132.3 | 70.2 |
| 0.2 | **85.4** | **100.4** | 66.3 | 113.7 | 130.6 | 70.2 |
| 0.3 | 85.8 | 100.7 | 66.3 | **113.1** | **130.0** | 70.2 |
| 0.5 | 90.0 | 104.3 | 66.3 | 115.0 | 131.7 | 70.2 |
| 1.0 | 105.2 | 118.0 | 66.4 | 125.8 | 141.8 | 70.2 |

Smoothing the depth alone halves the lifted jitter (127 → 70) at any σ ≥ 0.08 s and improves the
absolute pelvis by 6–8 mm; the remaining jitter (70 vs a GT of 8) is orientation / articulation
noise, which is the refiner's job. `depth_smooth_sec: 0.25` (between the two minima).

**GT motion RMS** after the 0.12 s label smoothing, train scenes (→ `motion_supervision.scale`):
vel 0.40 m/s, acc 1.18 m/s², ang_vel 0.57 rad/s, ang_acc 1.81 rad/s² (test: 0.39 / 1.15 / 0.51 / 1.52).

### Stage 2

Reference protocol (stride auto, 120-row cap, 108 scenes). The untrained stage-2 model is the
stage-1 body after depth smoothing (σ 0.25 s) and clip-mean betas, before any learned correction
(`output/stage1_20260905_180319/eval_stage2_untrained.json`):

| model | mpjpe | pa | pve | accel | pelvis_err | depth_err | dlogz_pred / err | wa_mpjpe100 | w_mpjpe100 | rte | lifted jitter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stage 1 raw | 56.28 | 38.49 | 72.61 | 11.07 | 119.9 | 113.7 | 1.37 / 1.30 | 82.5 | 138.3 | 4.73 | 109.2 |
| stage 2 untrained (smoothing + mean betas) | 56.21 | 38.49 | 72.53 | 10.96 | 114.3 | 107.8 | 0.30 / 0.35 | 74.4 | 136.3 | 3.95 | 63.0 |

Depth smoothing costs nothing on the articulated metrics (MPJPE / PA / PVE unchanged) and buys
the absolute pelvis 5.6 mm, the depth-step noise ×4 (`dlogz_pred` 1.37 → 0.30 %/frame, close to
the GT's 0.47) and the lifted jitter 109 → 63 (GT 7.2).

**Run `stage2_20260905_193527`** (configs/stage2.yaml, GPUs 0 + 5, 401 steps/epoch of 2 × 120-frame
clips per GPU, ~11 min/epoch, 10 epochs, best = last; `output/stage2_20260905_193527/eval.json`,
transcript `output/logs/stage2_eval_predict.log`). Per-epoch test trajectory:

| epoch | contact f1 | P | R | P@R90 | force mae (bw) | mpjpe | lifted jitter | vel r | acc r | ang_vel r | ang_acc r |
|---|---|---|---|---|---|---|---|---|---|---|---|
| untrained | – | – | – | – | – | 56.21 | 63.0 | – | – | – | – |
| 0 | 0.862 | 0.776 | 0.969 | 0.817 | 0.271 | 56.36 | 63.3 | 0.54 | 0.24 | 0.86 | 0.29 |
| 1 | 0.905 | 0.869 | 0.943 | 0.904 | 0.246 | 56.14 | 63.1 | 0.65 | 0.49 | 0.89 | 0.82 |
| 2 | 0.918 | 0.892 | 0.947 | 0.925 | 0.236 | 56.06 | 63.1 | 0.71 | 0.59 | 0.92 | 0.87 |
| 4 | 0.926 | 0.911 | 0.942 | 0.936 | 0.223 | 56.03 | 63.2 | 0.76 | 0.66 | 0.94 | 0.90 |
| 6 | 0.926 | 0.917 | 0.935 | 0.937 | 0.218 | 56.00 | 63.2 | 0.79 | 0.68 | 0.95 | 0.91 |
| **9** | **0.925** | **0.920** | **0.931** | **0.937** | **0.216** | **55.97** | 63.4 | **0.81** | **0.70** | **0.95** | **0.91** |

Final row, the full picture (reference protocol):

* **Contact** F1 0.925 / IoU 0.861 / P@R90 0.937 — above every previous run (best before:
  0.922 / 0.855 / 0.932, the 2026-09-03 camera-posetoken arm). Per group: LH 0.968, RH 0.974,
  LF 0.891, RF 0.899, heels 0 (704 + 394 positives, never predicted — unchanged from every run).
* **Forces** MAE 0.216 bw on in-contact limb-frames, 0.018 bw mean magnitude off contact.
* **Motion** RMSE / Pearson: velocity 0.18 m/s / 0.81, acceleration 0.69 m/s² / 0.70, root angular
  velocity 0.14 rad/s / 0.95, angular acceleration 0.55 rad/s² / 0.91 (2026-08 motion heads on image
  tokens reached acc r ≈ 0.35 at best).
* **Pose**: MPJPE 55.97 / PA 38.37 / PVE 72.26 vs 56.21 / 38.49 / 72.53 untrained — the pose head
  buys 0.25 mm; lifted jitter 63.4 (63.0 untrained, 109 raw stage 1); pelvis error 113.8 mm.

**Pose-head diagnostic** (12 test scenes, 1427 frames, `output/logs/stage2_diag_pose_head.log`):
the corrections the head emits are tiny — root rotation 0.4° mean (max 1.4°), body joints 1.2°
mean / 3.5° max-per-frame, root shift 9 mm (p95 17 mm), joint displacement 11 mm — and zeroing the
head changes MPJPE by 0.35 mm and jitter by +0.3 (55.7 with vs 55.4 without). The learned
correction is a small, near-static bias fix; it did not learn to denoise the trajectory. The
articulation jitter that remains after depth smoothing (63 vs 7 GT) is a derivative-level quantity
that a per-frame Huber on joint positions barely rewards removing.

### Round 4 — five arms (2026-09-06/07, 15 epochs each, GPUs 0/1/4/7, ~2 h per arm)

Runs `stage2_v4_20260906_215038` (A), `stage2_v4_nofc_20260907_004942` (B),
`stage2_v4_nomotion_20260907_024703` (C), `stage2_v4_noforce_20260907_045218` (D),
`stage2_v4_nosmooth_20260907_064922` (E); each run's `last.pth` (the final model: `best.pth` is
the MPJPE minimum, an early epoch with immature heads) evaluated under the reference protocol
into `<run>/eval.json` (transcripts `output/logs/<exp>_posteval_console.log`) and dumped for
the viewer (`<run>/predictions/`). Round 3 (10 epochs, no RNEA, no heel weight) is the row to
beat. Contact at threshold 0.5; forces in bw; RNEA residuals in bw / bw·m against the GT floor
0.209 / 0.050; Pearson r of the motion head:

| arm | f1 | P | R | mpjpe | pa | pve | wa-mpjpe | w-mpjpe | jitter | force mae | angle | corr | share pp | rnea f / t | vel r | acc r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen SAM3D, SMPL-X refit | – | – | – | 61.10 | 44.15 | 78.03 | 81.2 | 134.6 | 113.7 | – | – | – | – | – | – | – |
| stage 1 raw (no smoothing) | – | – | – | 56.32 | 38.61 | 72.61 | 83.0 | 138.8 | 109.2 | – | – | – | – | – | – | – |
| round 3 | **0.924** | **0.920** | 0.928 | 56.01 | 38.50 | 71.80 | 74.7 | 132.5 | 3.9 | 0.219 | – | 0.63 | 7.8 | – | 0.79 | 0.64 |
| A full | 0.917 | 0.905 | 0.929 | 56.16 | 38.80 | 71.85 | 75.3 | 133.4 | 3.9 | **0.190** | **21.3** | 0.67 | 1.6 | **0.240 / 0.053** | 0.79 | **0.66** |
| B no RNEA | 0.916 | 0.903 | 0.929 | 56.10 | 38.68 | 71.79 | 75.2 | 133.0 | 4.2 | 0.191 | 22.4 | **0.70** | 2.2 | – | 0.79 | 0.66 |
| C no motion losses | 0.916 | 0.903 | 0.928 | **55.36** | **38.04** | **71.46** | 78.3 | 135.6 | 3.9 | 0.191 | 21.5 | 0.65 | **1.4** | 0.283 / 0.060 | – | – |
| D no forces | 0.915 | 0.900 | **0.931** | 56.09 | 38.68 | 71.79 | 75.1 | 132.8 | 4.0 | – | – | – | – | – | 0.79 | 0.65 |
| E no input smoothing | 0.912 | 0.899 | 0.926 | 57.90 | 40.64 | 73.23 | 77.5 | 137.7 | 78.0 | 0.191 | 21.4 | 0.67 | 1.6 | 0.278 / 0.060 | 0.78 | 0.65 |
| F no contact tokens | 0.911 | 0.894 | 0.929 | 56.20 | 38.81 | 71.86 | **74.7** | **132.2** | 3.6 | 0.192 | 21.5 | 0.65 | 1.5 | 0.233 / 0.052 | 0.80 | 0.67 |
| G one-frame window | 0.901 | 0.886 | 0.916 | 56.57 | 38.92 | 72.20 | 78.9 | 136.8 | 4.4 | 0.193 | 21.7 | 0.63 | **1.3** | 0.288 / 0.056 | 0.75 | 0.45 |

The two reference rows are the per-frame models with NO temporal processing at all, scored
under the same protocol and metric code on 2026-09-07: the frozen SAM 3D Body output refit to
SMPL-X (`scripts/eval_frozen_smplx.py`, `output/frozen_sam3d_smplx.json` — camera frame
pelvis error 112.7 mm, depth error 104.5 mm, accel 11.9 m/s²) and stage 1's own SMPL-X +
CLIFF readout (`configs/stage1_eval_auto.yaml`, `output/stage1_20260905_180319/
eval_auto_20260907.json` — pelvis 119.9, depth 113.6, accel 11.0). Their lifted jitter
(114 / 109 vs the GT's 6.9) is what the refiner's input smoothing removes.

`wa-mpjpe` / `w-mpjpe`: WHAM's world-frame MPJPE over 100-frame segments of the lifted
trajectory (WA: each segment rigidly aligned; W: aligned on its first frames), mm. `corr` /
`share pp`: the two force numbers of the climb_wall_2 board table computed on the corpus
(`scripts/force_corr_share.py` on the runs' whole-scene prediction dumps, 108 scenes,
29 963 person-frames with a valid kindyn solve): the six groups folded into the four board
limbs (foot = toe + heel), `corr` the Pearson correlation of the pooled per-limb force sizes,
`share pp` the mean absolute deviation of each limb's share of the mean total force from the
GT shares (GT: hands 26.8 / 32.2 %, feet 19.2 / 21.8 %). Both use every row, in contact or not.


Per group at 0.5 (P / R / F1; heel positives 704 / 394 of ~10.5k rows each):

| arm | LH | RH | L toe | R toe | L heel | R heel |
|---|---|---|---|---|---|---|
| round 3 | 0.968 (f1) | 0.974 (f1) | 0.891 (f1) | 0.899 (f1) | 0 / 0 / 0 | 0 / 0 / 0 |
| A full | 0.954 / 0.982 / 0.968 | 0.960 / 0.988 / 0.974 | 0.876 / 0.899 / 0.887 | 0.880 / 0.915 / 0.897 | 0.185 / 0.132 / 0.154 | 0.090 / 0.099 / 0.094 |
| B no RNEA | 0.952 / 0.982 / 0.967 | 0.960 / 0.989 / 0.974 | 0.868 / 0.893 / 0.881 | 0.877 / 0.919 / 0.897 | 0.223 / 0.158 / 0.185 | 0.072 / 0.079 / 0.075 |
| C no motion losses | 0.958 / 0.983 / 0.970 | 0.959 / 0.988 / 0.973 | 0.880 / 0.891 / 0.885 | 0.871 / 0.915 / 0.893 | 0.218 / 0.159 / 0.184 | 0.091 / 0.122 / 0.105 |
| D no forces | 0.950 / 0.985 / 0.968 | 0.956 / 0.987 / 0.972 | 0.868 / 0.903 / 0.886 | 0.871 / 0.912 / 0.891 | 0.198 / 0.148 / 0.169 | 0.137 / 0.147 / 0.142 |
| E no input smoothing | 0.946 / 0.977 / 0.961 | 0.955 / 0.988 / 0.971 | 0.867 / 0.890 / 0.878 | 0.869 / 0.916 / 0.892 | 0.219 / 0.153 / 0.180 | 0.110 / 0.117 / 0.113 |
| F no contact tokens | 0.949 / 0.978 / 0.964 | 0.961 / 0.988 / 0.975 | 0.863 / 0.895 / 0.879 | 0.879 / 0.925 / 0.901 | 0.152 / 0.125 / 0.137 | 0.067 / 0.119 / 0.086 |
| G one-frame window | 0.941 / 0.967 / 0.954 | 0.947 / 0.981 / 0.964 | 0.844 / 0.889 / 0.866 | 0.849 / 0.898 / 0.873 | 0.122 / 0.078 / 0.095 | 0.119 / 0.114 / 0.116 |

What the arms say (all differences between A/B/C/D on the hands / toes are within ±0.01,
the run-to-run noise):

* **Heel class weight** (every arm vs round 3): the heels go from never predicted to predicted on
  ~4.5 % of frames (true rate 6.7 % / 3.7 %), so the volume is right, but 3 in 4 of those flags
  are wrong (precision 0.07–0.22, recall 0.08–0.16, F1 0.08–0.18). Those ~800 heel false
  positives are the whole micro-F1 loss (0.924 → 0.916: micro precision 0.920 → 0.903, recall
  unchanged) — the hands and toes did not move. Mid-run (arm A, epoch 8, whole test set) the
  same picture: left heel P 0.11 / R 0.08, right 0.08 / 0.09. The weight makes the head fire at
  the base rate; it does not make the heel signal readable from the inputs.
* **RNEA** (A vs B): force direction 22.4° → 21.3° and the residual 0.240 / 0.053 (floor
  0.209 / 0.050) at the price of the off-contact force magnitude 0.020 → 0.043 bw (the residual
  loads the free limbs; round 2 phase 2 saw the same); force MAE, contacts, pose and motion
  identical; jitter 4.2 → 3.9.
* **Motion losses** (A vs C): removing both the motion-head terms and the pose-derivative
  matching is the only change that moves the pose: MPJPE 56.16 → **55.36**, PA 38.80 → 38.04,
  PVE 71.85 → 71.46 — the best stage-2 pose so far (stage 1 raw 56.28, untrained stage 2 56.21)
  — with the SAME lifted jitter (3.9: the input smoothing does the denoising, the derivative
  terms buy none). The derivative terms do hold the absolute root: pelvis error 112.9 → 115.4
  mm and the RNEA residual 0.240 → 0.283. So the matching terms trade 0.8 mm of articulation
  for 2.5 mm of root placement and physics consistency.
* **Forces** (B vs D): the force head is a free rider — contacts, pose, jitter and motion are
  identical with and without it.
* **Force allocation** (corr / share): every round-4 arm puts the load where kindyn puts it
  (share error 1.4–2.2 pp; round 3, at 10 epochs, still under-loaded the feet: 12 / 13 % of
  the total against the GT's 19 / 22 %, 7.8 pp). The RNEA loss trades correlation for
  allocation (A 0.67 / 1.6 vs B 0.70 / 2.2) — the same trade the board table showed for the
  learned model against the optimisation.
* **Input smoothing** (A vs E): without the 0.10 / 0.08 s Gaussian smoothing before the block
  the lifted jitter is 78 (vs 3.9; stage 1 raw 109) and the pose 57.9 / PA 40.6 (vs 56.2 /
  38.8) after 15 epochs — the additive block does not learn to denoise (the 2026-09-04 probes
  found the same), and the derivative / RNEA terms on the raw motion cost 1.7 mm of pose.
  Contacts (0.912) and forces (0.191 bw, 21.4°) are unaffected: they never needed the smoothing.
* **Contact tokens** (A vs F, run `stage2_v4_notokens_20260907_101252`, 15 epochs in 8.5 min on
  the pose-token cache; evaluated live — the cached-path evaluation agrees to the third decimal,
  `eval_cached.json`): without the six learned decoder tokens the contact F1 is 0.911 vs 0.917
  (precision 0.894 vs 0.905 at the same recall; P@R90 0.914 vs 0.922; heels 0.137 / 0.086 vs
  0.154 / 0.094) while pose, jitter, forces, RNEA and motion are unchanged (56.20 / 3.6 / 0.192
  / 21.5° / 0.233). The tokens' image attention is worth ~0.01 of contact precision — small but
  the only signal the refiner gets from the image beyond the pose token; everything else in the
  contact pipeline is carried by the body trajectory. Arm F is the cheapest build by far (no
  backbone, no decoder: 33 s per epoch against ~480 s).
* **Cross-frame attention** (A vs G, run `stage2_v4_window1_20260907_110658`: `window` 0.01 s,
  so every frame attends only itself in all three layers; token, smoothing and losses as A):
  the only arm besides E that loses on every axis at once — contact F1 0.901 vs 0.917 (both
  precision and recall, every group: hands −0.01, toes −0.02, heels −0.06 / +0.02), MPJPE
  56.57 vs 56.16, wa-mpjpe 78.9 vs 75.3, jitter 4.4 vs 3.9, RNEA residual 0.288 vs 0.240, and
  the motion head's acceleration Pearson 0.45 vs 0.66 (velocity 0.75 vs 0.79 — the token's
  own velocity features carry that). Forces are indifferent (MAE 0.193, angle 21.7°, corr
  0.63, share 1.3 pp — the best allocation, within noise of the others). So the temporal block
  earns its keep on contacts, motion and physics consistency, not on the per-frame pose.
* Per-epoch trajectories (`~/.claude/handoffs/round4_prep_20260906/arms_summary.txt`): every
  arm's pose is flat from epoch ~4, contacts / forces plateau by epoch ~8, the test loss rises
  slowly after epoch ~7 while the train loss keeps falling (the usual mild over-fit of the
  contact BCE and force Huber; metrics do not degrade).
