# The two-stage pipeline: per-frame body, world-space temporal refiner

*Concept page. Configs: `configs/stage1.yaml`, `configs/stage2.yaml`. Code: `model/refiner.py`,
`model/loss/motion.py`. Measurement scripts: `scripts/dump_stage1.py`, `scripts/analyze_stage1.py`.*

## Why the pivot (2026-09-05)

Every 2026-08/09 attempt to improve the per-frame pose with a temporal model over the frozen
decoder's image tokens failed the same way: the tokens carry no velocity information beyond the
pose readout, a temporal block over them degenerates to clip pooling, and velocity losses on the
pose path collapse into shrinkage (`docs/history/`). The per-frame SMPL-X head on the frozen pose
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

## Evaluation protocol

`scripts/evaluate.py` on the annotated test scenes, one clip per (scene, person), the longest valid
run at the `auto` stride capped at `data.eval_max_frames`. Stage-1 numbers in `docs/results.md`
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
| frozen SAM3D refit (docs/results.md) | 61.07 | 44.06 | 78.08 | 11.93 | – | – | – | – | – | – | – | – |
| per-frame probe 2026-09-02 (docs/results.md) | 57.60 | 38.86 | 74.74 | 11.15 | – | – | – | – | – | – | – | – |
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
