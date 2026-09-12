# Making the temporal block temporal: local integer RoPE, Lie-algebra velocity matching, smoothed cameras (2026-09-05)

Round goal (user): make the post-decoder temporal block actually use its neighbours. Everything
runs on the STATIC-camera subset (113 train / 16 annotated test scenes), pose only — contacts and
hands are OFF for this round. Two arms; the second pair (velocity weights 8 / 8 / 40) was launched 2026-09-05 13:06 on GPUs 0+1 (ray)
and 4+5 (cliff) and STOPPED by the user at 13:26 (§2.2, defect §2.3) — nothing is running. History: the first pair (2 / 4 / 20, 12:52) was killed at ray epoch 8 / cliff epoch 5 (§2.1);
the cliff arm's very first start on GPUs 3+4 was aborted after 3 minutes when GPU 3 was withdrawn
(`trash/tvel_cliff_gpu3_aborted_20260905`).

| arm | config | camera convention | run dir |
|---|---|---|---|
| ray | `configs/tvel_ray.yaml` | ray head on the frozen prior, bbox + frozen-camera token inputs, kp2d in bearing units, `depth` + `bearing` terms (static_ray R1 minus its vel/acc stencils) | `output/tvel_ray_<stamp>/` |
| cliff | `configs/tvel_cliff.yaml` | legacy static_baseline convention: CLIFF (s, tx, ty) head, proxy `cam` term, crop kp2d, no camera token inputs | `output/tvel_cliff_<stamp>/` |

Everything else is identical: 4 clips/step (2 per GPU × 2), lr 2e-4, 30 epochs (3090 steps,
500 warm-up), `eval_max_frames 120`, monitor `metric_pose/mpjpe`, frozen reference
`output/frozen_sam3d_smplx_static16.json`. Peak memory 10.6 GiB per GPU (smoke run).

## 1. The block (`model/rope.py`)

* **Integer, fps-agnostic RoPE** (`position: index`, the new default): the RoPE position of a
  token is its clip ROW index (0, 1, 2, ...), one sampled frame = one unit, whatever the scene's
  frame rate. `position: seconds` keeps the previous convention (`frame_pos_sec × time_scale`);
  the five kept runs' configs are pinned to it so their checkpoints evaluate unchanged.
* **±4-frame window per layer** (`window: 4`, position units): every attention op sees at most 9
  frames; over the 4 layers the receptive field is ±16 frames (0.64 s at 25 fps). The window is a
  hard keep-mask (plus the `frame_valid` rule), so whole-scene inference never exposes an
  untrained offset. Verified: mask row sums `[5, 6, 7, 8, 9, 9, ...]`, a perturbation at frame 11
  leaves frame 0 bit-identical through 2 layers (RF 8), a constant time shift in `seconds` mode
  changes the output by 5e-7, and `index` equals `seconds` at 25 fps to 0.0 when the seconds
  window is `4/25 + 1e-6` — at exactly `4/25` the float compare `dist <= window` is a knife-edge
  (review: 26 of 1600 mask entries flip), one more reason for the integer default.
* **Plain pre-LN block** (`gate_init: none`, the new default): `x + Proj(Attn(LN x))`,
  `x + FFN(LN x)`, attention / FFN output projections zero-initialised — an exact identity at
  init (`torch.equal`) with a first-order gradient to the projections from step one (the
  `tb_projzero` finding), and no LayerScale gates to open. `zero_gate` / `zero_proj` survive as
  legacy values for the old checkpoints.

## 2. Velocity matching on the Lie algebra (`model/loss/velocity.py`, `velocity_matching`)

Terms: `root_vel`, `root_ang_vel` (the pelvis pose as SE3), `joint_ang_vel` (the 21 parent-local
joint rotations as SO3), each a Huber on `value / delta`, Huber deltas = the GT's own
per-component RMS at the auto stride (40 static train scenes: root 0.42 m/s, root angular 0.60
rad/s, joints 0.87 rad/s → 0.4 / 0.6 / 0.9).

Design, agreed with the user:

1. **Increments are group differences**, never differences of 6D / quaternion / log-of-absolute:
   root `d[t] = se3_log(T_t^{-1} T_{t+1}) / dt` (BetterRobot `se3.log`, the linear part with the
   `V^{-1}` coupling — the true tangent, not `R^T Δp`), joints `so3_log(R_t^T R_{t+1}) / dt`.
   The predicted body is lifted to the world with the (smoothed) extrinsics first, the GT is the
   kindyn SMPL-X world trajectory. Torch increments match the loader's float64 mirror to 2e-7.
2. **One-step FORWARD increments** (rows t, t+1; `T − 1` rows per clip). Central stencils and
   BVR's `(d[t−1] + d[t]) / 2` have zero response at the Nyquist frequency: an alternating
   frame-to-frame jitter is invisible to them. The one-step increment sees it.
3. **Shared comparison frame.** A body-frame twist lives in ITS trajectory's body frame; with an
   orientation error `E_t = R_gt^T R_pred` the predicted vectors are rotated by `E_t` relative
   to the GT's, so a plain comparison charges the velocity term for the absolute orientation
   error that `orient` / `pose` already own (~9 % of |v| at 5°). The predicted increment is
   therefore transported into the GT frame at t (`v → E_t v`, `so3.act`) before the Huber — for
   the angular parts exactly the same as comparing spatial-frame increments `log(R_{t+1} R_t^T)`
   (the linear part is a plain frame change, no `[p]×` adjoint term). Verified: a constant
   BODY-side offset `R_pred = R_gt C` leaves the transported increments equal to the GT's to 2e-7
   while the untransported angular ones differ by 0.011 rad/s. The review adds the mirror fact: a
   constant WORLD-side offset `R_pred = D R_gt` is NOT cancelled by the transport (the untransported
   body-frame comparison would cancel that one instead) — the body-side offset (a root-convention
   mismatch between the head and kindyn) is the one this design targets. Same transport per joint
   on the parent-local rotations.
4. Gradients reach the pose path only; no motion head. Metrics `metric_velocity/*`: Pearson r
   pooled over components, RMSE and GT RMS per term (physical units).

Weights: `root_vel 8, root_ang_vel 8, joint_ang_vel 40` (v2, launched 13:06 — see §2.1 for why the first
pair was restarted). The first pair used `2 / 4 / 20`, chosen from the per-term gradient norms at init
(scratchpad `tvel/calib_grads.py`, 3 batches of 2 clips, ray arm; the total-gradient L2 norm over
the trainable parameters of each term at weight 1):

| term | grad @1 | config weight | grad @weight | value @weight |
|---|---|---|---|---|
| smplx.kp2d (bearing units) | 3.48 | 10 | 34.8 | 0.37 |
| smplx.kp3d | 1.75 | 5 | 8.8 | 0.88 |
| smplx.orient | 7.29 | 1 | 7.3 | 0.61 |
| smplx.pose | 0.54 | 1 | 0.54 | 0.09 |
| smplx.betas | 6.63 | 0.1 | 0.66 | 0.08 |
| smplx.depth | 2.86 | 10 | 28.6 | 0.44 |
| smplx.bearing | 4.39 | 2 | 8.8 | 0.007 |
| velocity.root_vel | 11.43 | 2 | 22.9 | 0.74 |
| velocity.root_ang_vel | 1.55 | 4 | 6.2 | 1.19 |
| velocity.joint_ang_vel | 0.18 | 20 | 3.6 | 4.5 |

At 2 / 4 / 20 the velocity block is ~27 % of the total gradient norm at init (the 2026-09-04 matching
arms: 90 % at 1 / 0.5 / 1 slowed the pose, ~2 % at 0.05 / 0.1 / 1 did nothing); at 8 / 8 / 40 it is
~60 % (91 + 12 + 7 vs 89). CLIFF arm at init: kp2d 17.9, kp3d 6.8, orient 7.4, cam 5.2, root_vel@1
17.5 — the same weights are used so the arms differ only in the camera convention.

### 2.1 First pair (2 / 4 / 20), 12:52-13:06, killed by the user at ray epoch 8 / cliff epoch 5

Partial runs + logs in `/data3/rikhat.akizhanov/trash/tvel_v1_weights_2_4_20_20260905/`. Test metrics per
epoch (16 static scenes; jitter = `lifted_jitter`, GT 6.35; dlogz = `dlogz_pred` %/frame, GT 0.275):

| arm | ep | mpjpe | pa | depth_err | dlogz | jitter | root_vel r / rmse | root_ang_vel r | joint_ang_vel r |
|---|---|---|---|---|---|---|---|---|---|
| ray | 0 | 261.6 | 182.2 | 95 | 0.487 | 91.7 | 0.46 / 0.38 | 0.46 | 0.31 |
| ray | 1 | 201.6 | 159.1 | 126 | 0.447 | 87.6 | 0.48 / 0.36 | 0.68 | 0.43 |
| ray | 2 | 174.1 | 135.9 | 126 | 0.436 | 84.3 | 0.49 / 0.35 | 0.74 | 0.48 |
| ray | 3 | 153.8 | 110.4 | 131 | 0.438 | 83.6 | 0.49 / 0.35 | 0.72 | 0.50 |
| ray | 4 | 117.6 | 78.8 | 121 | 0.439 | 83.1 | 0.48 / 0.36 | 0.77 | 0.52 |
| ray | 5 | 105.4 | 67.6 | 111 | 0.445 | 83.8 | 0.48 / 0.36 | 0.80 | 0.53 |
| ray | 6 | 98.2 | 64.4 | 113 | 0.441 | 83.0 | 0.49 / 0.36 | 0.79 | 0.53 |
| ray | 7 | 88.5 | 60.8 | 110 | 0.440 | 83.1 | 0.49 / 0.35 | 0.81 | 0.53 |
| ray | 8 | 82.9 | 59.7 | 113 | 0.434 | 81.9 | 0.49 / 0.35 | 0.82 | 0.53 |
| cliff | 0 | 289.9 | 196.4 | 1521 | 1.056 | 62.1 | 0.17 / 0.53 | 0.43 | 0.27 |
| cliff | 1 | 213.5 | 160.0 | 1065 | 0.487 | 58.8 | 0.41 / 0.32 | 0.63 | 0.39 |
| cliff | 2 | 182.0 | 140.5 | 888 | 0.452 | 60.9 | 0.45 / 0.31 | 0.71 | 0.46 |
| cliff | 3 | 167.7 | 123.5 | 850 | 0.454 | 60.8 | 0.44 / 0.32 | 0.73 | 0.48 |
| cliff | 4 | 135.3 | 93.8 | 823 | 0.467 | 60.6 | 0.43 / 0.33 | 0.72 | 0.50 |
| cliff | 5 | 118.9 | 75.0 | 859 | 0.483 | 59.6 | 0.43 / 0.33 | 0.76 | 0.52 |

Reading (the reason for the restart): the ray arm's pose converged faster than tb_projzero (83 vs 104
mm at epoch 8) but its DEPTH jitter never moved — dlogz_pred 0.44 %/frame at every epoch (tb_projzero
0.21 at epoch 8; the depth noise alone at 3 m and 25 fps is 0.0044 × 3 × 25 = 0.33 m/s, which is the
flat root_vel RMSE of 0.35) and the lifted jitter plateaued at 82-84 (tb_projzero 55, static_ray R1 68
at the same epoch). The retired ray recipe's depth_vel / depth_acc stencils (~60 gradient units at init)
were the depth-denoising signal; `root_vel` at weight 2 (23 units over three axes) did not replace them.
The CLIFF arm's absolute depth sat 0.82-0.86 m too close from epoch 3 on (static_baseline ended at
125 mm): under the CLIFF lift the predicted velocity noise scales with the predicted depth, so a
velocity term rewards under-estimating it, and the proxy `cam` term (5 units) could not hold it. The
user chose to restart both arms at 8 / 8 / 40 rather than finish the pair.

### 2.2 Second pair (8 / 8 / 40), 13:06-13:26, STOPPED by the user at ray epoch 8 / cliff epoch 6

Run dirs `output/tvel_ray_20260905_130637` (best.pth = epoch 8) and `output/tvel_cliff_20260905_130652`
(best = epoch 3) are kept as they are; both runs are incomplete (9 and 7 of 30 epochs).

| arm | ep | mpjpe | pa | depth_err | dlogz | jitter | root_vel r / rmse | root_ang_vel r | joint_ang_vel r |
|---|---|---|---|---|---|---|---|---|---|
| ray | 0 | 273.4 | 200.2 | 117 | 0.456 | 88.1 | 0.47 / 0.36 | 0.47 | 0.36 |
| ray | 1 | 224.8 | 179.1 | 199 | 0.391 | 78.3 | 0.51 / 0.32 | 0.68 | 0.45 |
| ray | 2 | 204.3 | 162.6 | 222 | 0.378 | 74.2 | 0.52 / 0.31 | 0.74 | 0.48 |
| ray | 3 | 197.3 | 147.2 | 229 | 0.387 | 74.1 | 0.52 / 0.31 | 0.74 | 0.50 |
| ray | 4 | 179.5 | 126.9 | 225 | 0.391 | 75.3 | 0.52 / 0.31 | 0.74 | 0.51 |
| ray | 5 | 145.3 | 100.7 | 217 | 0.393 | 74.8 | 0.52 / 0.32 | 0.75 | 0.51 |
| ray | 6 | 125.5 | 82.5 | 227 | 0.386 | 72.9 | 0.52 / 0.31 | 0.77 | 0.52 |
| ray | 7 | 112.7 | 73.4 | 230 | 0.383 | 72.1 | 0.53 / 0.31 | 0.79 | 0.52 |
| ray | 8 | 104.7 | 69.1 | 240 | 0.377 | 70.1 | 0.53 / 0.30 | 0.79 | 0.53 |
| cliff | 0 | 328.5 | 230.2 | 2417 | 1.065 | 41.2 | 0.18 / 0.39 | 0.42 | 0.30 |
| cliff | 1 | 289.3 | 211.6 | 2248 | 0.494 | 38.0 | 0.45 / 0.26 | 0.62 | 0.42 |
| cliff | 2 | 273.3 | 205.3 | 2113 | 0.447 | 39.3 | 0.49 / 0.25 | 0.71 | 0.47 |
| cliff | 3 | 264.4 | 197.4 | 2096 | 0.449 | 39.2 | 0.48 / 0.26 | 0.74 | 0.49 |
| cliff | 4 | 264.7 | 194.4 | 2126 | 0.462 | 38.6 | 0.48 / 0.26 | 0.75 | 0.51 |
| cliff | 5 | 271.7 | 195.5 | 2189 | 0.485 | 37.3 | 0.47 / 0.26 | 0.75 | 0.51 |
| cliff | 6 | 268.7 | 193.0 | 2147 | 0.486 | 38.1 | 0.46 / 0.26 | 0.74 | 0.52 |

Reading: against the 2 / 4 / 20 pair at equal epochs the ray arm trades pose for smoothness — jitter
82 → 70 and dlogz 0.43 → 0.38 at epoch 8, but mpjpe 83 → 105 and depth_err 113 → 240 mm (bias
negative and growing). Neither weight setting approaches tb_projzero's 55 / 0.21 at epoch 8, let
alone the GT floor. The CLIFF arm never recovered its absolute depth (2.1-2.4 m too close from
epoch 0 on, mpjpe stalled ~265-290). Both are the SAME defect of the linear root term, §2.3.

### 2.3 Defect found 2026-09-05 13:35: the metric root_vel term rewards shrinking a noisy trajectory

The predicted world velocity of the pelvis scales with the predicted depth (`x = u z / f`), and so
does its NOISE. For a depth-noisy prediction the Huber of `v_pred − v_gt` is dominated by the noise
term, so it FALLS when the whole trajectory is pulled toward the camera — a variance/bias trade the
absolute-depth terms (`depth` 10 in the ray arm, the proxy `cam` alone in the CLIFF arm) then have to
fight. Measured on the 16 static test tracks (GT pelvis + multiplicative depth noise at the model's
own level 0.44 %/frame, GT orientation, `delta` 0.4 m/s; scratchpad `tvel/scale_bias`):

| depth scale s on the prediction | no noise | noise 0.44 %/frame | noise 0.6 % | **v / z (own depth), 0.44 %** |
|---|---|---|---|---|
| 0.5 | 0.055 | 0.183 | 0.271 | 0.252 |
| 0.7 | 0.022 | 0.254 | 0.397 | 0.252 |
| 0.9 | 0.003 | 0.352 | 0.551 | 0.252 |
| 1.0 | 0.000 | 0.410 | 0.636 | 0.252 |
| 1.2 | 0.010 | 0.538 | 0.817 | 0.252 |

With noise the loss is monotone in s down to s = 0.5 (0.41 → 0.18): the term prefers a body half as
far away. Dividing each trajectory's linear increment by its OWN camera depth (a dimensionless
velocity, units 1/s — the metric analogue of the retired bearing / log-depth rates) makes the term
exactly scale-invariant (flat 0.252) while keeping the se3 increment and the transport; the angular
root term and the joint term are scale-free already. Not yet changed in the code (user decision pending).

**Follow-up (2026-09-05 afternoon):** the scale bias is one instance of a general shrinkage
incentive of pointwise derivative losses on noisy per-frame estimates — the runs reduced the velocity
loss by shrinking the amplitude of the predicted motion (joint velocity 0.49 of GT, root rotation
0.72) instead of denoising, and the block never filtered (branch gain 0.03). Full analysis, evidence,
literature and design options: `docs/velocity_matching_truth_2026-09-05.md`.

## 3. Camera smoothing (`data.camera_smooth_sec`, `data/climbing_videos/camera.py::smooth_cameras`)

Per-frame camera estimates on the static scenes (16 test + 24 train; `transform.npz`):

| quantity | typical per-frame step | notes |
|---|---|---|
| focal length | 0.03–0.10 % (US6c-J7Rlls_0000: 0.45 %) | several scenes drift 0.5–4.5 % over the clip (slow, not noise) |
| principal point | 0 | constant per scene |
| rotation | 0.011–0.035° (US6c: 0.065°) | total end-to-start 0.03–2.2° |
| camera centre | 0.5–2 mm (US6c: 3.9 mm) | spread around the mean 0.5–3 mm; US6c / s-ArwEzr-2M_0025 drift 8–12 cm |

Gaussian along time, per scene, σ = `camera_smooth_sec × fps` frames (0.25 s ≈ 6 frames at 25
fps), edge-replicated: `f_x f_y c_x c_y` component-wise; the extrinsics as the world-from-camera
free-flyer pose (centre + hemisphere-aligned quaternion average, `kindyn.smooth_root_trajectory`)
recomposed into a rigid `cam_from_world` (orthonormality 1e-7). At σ 0.25 s the per-frame steps
drop 6–10× (4HuRoofxxMI_0002: f 0.047 → 0.004 %, rotation 0.011 → 0.001°, centre 0.48 → 0.05 mm),
the mean focal moves < 0.002 %. Applied at scene load BEFORE every consumer — the frozen decoder's
CLIFF / ray-map conditioning, the SMPL-X head's lift and projection, the GT lifted into the
camera, the camera twist and clip jumps — at train and test alike. Two consequences the review
flagged: (i) `metric_global/lifted_*` lifts the prediction with the SMOOTHED extrinsics, so the
jitter numbers are not strictly comparable with the reference runs (raw cameras); the camera
centre's ~0.7 mm white noise contributes about 5 units of jitter (10 m/s³) — negligible against
52, but the size of the GT floor (6.35), i.e. it matters in the regime this round targets. The
frozen baseline is therefore re-scored with `camera_smooth_sec 0.25`
(`output/frozen_sam3d_smplx_static16_camsmooth025.json`, §5) to measure that share directly;
the frozen SAM3D refit itself (`features/sam3d`) is NOT recomputed (user decision). (ii)
`smooth_cameras` treats every frame as valid, so on MOVING scenes a genuine camera-tracking
discontinuity would be smeared over ±4σ — irrelevant on the static subset, a trap for
`camera: all`.

## 4. What to watch

* `metric_global/lifted_jitter` (GT 6.35 on this set; tb_projzero 52.4, static_ray 56.7) and
  `metric_pose/mpjpe` (static_ray 65.2, static_baseline 60.6, frozen 57.9).
* `metric_velocity/root_vel_r`, `root_ang_vel_r`, `joint_ang_vel_r` — the pose-derived one-step
  velocities vs GT (r 0.24 / 0.10 / 0.05 for the untrained smoke model).
* `scripts/diag_temporal.py` on a checkpoint: attention mass by offset inside the ±4 window,
  projection norms (the gate-less block has no gammas), `same_frame` / `bypass` ablations.

## 5. Frozen baseline under smoothed cameras (review follow-up)

`scripts/eval_frozen_smplx.py --config configs/tvel_ray.yaml` (→ `camera_smooth_sec 0.25` on the test
set) vs the kept raw-camera json: mpjpe 57.871 → 57.873, pa 43.785 → 43.785, depth_err 97.16 → 97.14,
pelvis_err 105.38 → 105.37, accel 11.791 → 11.789, dlogz_gt 0.278 → 0.275. The pose-metric frozen line
is unchanged to 0.01 mm. The frozen script reports no `lifted_jitter`, so the camera-noise share of
the jitter metric stays the review's estimate (~5 units of 10 m/s³ from the ~0.7 mm centre noise);
`output/frozen_sam3d_smplx_static16_camsmooth025.json` is kept next to the raw one.

## 6. Review (Opus, read-only, 2026-09-05 13:35) and fixes

No training-affecting bug. Confirmed: identity at init for all three `gate_init` values (train mode,
dropout on), receptive field exactly ±4 × layers, bit-exact legacy equivalence of the pinned configs
(state dicts + forward, optimizer parameter order unchanged), the transport algebra and its
localisation on the joint tensors, invalid-row masking (no leak), the camera recomposition, every
config loads. Fixed after the review: `scripts/diag_temporal.py` crashed on a contact-less build
(guarded); `METRIC_GROUPS` now maps `motion_matching → matching`; dead NaN clause and an unused
constant removed; the doc / docstring claims made precise (body-side vs world-side offset, angular vs
linear spatial-frame equivalence, the seconds-window knife-edge). Open, deliberately not changed
mid-run: no GT-outlier gating in the velocity loss (the Huber's linear regime bounds a glitch's
gradient; `motion_matching` drops `motion_outlier` rows), and the eval-side camera smoothing
(§3). Review handoff: `~/.claude/handoffs/tvel_review_context.md`.

## 7. Round 2 (2026-09-05 evening): correlation loss, two-layer block, RoPE on seconds again — `tvel2_ray`

User decisions after `docs/velocity_matching_truth_2026-09-05.md`: replace the pointwise Huber by the
correlation form, shrink the block to two layers, re-calibrate the weights, put RoPE back on real
elapsed seconds, and make sure no loss depends on the source frame rate. Launched 15:53 on GPUs 0 + 5
(`configs/tvel2_ray.yaml`, log `output/logs/tvel2_ray_20260905_155257.log`). The CLIFF arm was not
relaunched (its depth collapse was the scale bias the correlation form removes; it can follow on the
same GPUs).

### 7.1 The loss (`model/loss/velocity.py`)

Same se3 / so3 one-step increments, same transport into the GT frame; the comparison is now, per clip
and term, `1 − r` with `r` the plain Pearson correlation of the transported predicted increments vs
the GT increments pooled over the clip's valid rows and the term's components. No stop-gradient, no
amplitude term (the toy follow-ups: a detached denominator inverts into amplitude growth, an
RMS-matching term brings back half the shrink). A clip needs ≥ 4 valid rows; the term's mass is the
number of contributing clips. The predicted variance is clamped from below at 1e-3 × the GT variance
(a 3 % amplitude) so the zero-motion init has a bounded gradient; above the clamp the correlation is
exact. New metric `metric_velocity/*_pred_rms` next to `*_gt_rms`: the amplitude ratio is the
shrinkage indicator (0.49 / 0.72 in the stopped runs; should sit near 1).

Unit checks (scratchpad `tvel2/check_corr.py`): `r = 1.0` for pred = GT, unchanged under a 3.7× scale
+ shift and under a different `dt` (fps), identical for a noisy prediction and 0.2× that prediction,
`d(1 − r)/d(scale) = 5e-8` at scale 1 (amplitude-blind), finite gradient at a constant prediction
(the clamp regime), clips with < 4 rows excluded. An earlier ADDITIVE floor of 1e-2 × var_gt was
rejected: it left `r = 0.995` at pred = GT and a −0.014 amplitude gradient (a small amplification
bias); the clamp has neither.

fps independence: the Pearson is invariant to the common `1/dt` factor of a clip's rows and to the
frame count, so a 24, 25 or 30 fps scene contributes the same loss for the same motion; the
per-frame terms never see time; camera smoothing is σ in seconds; the metrics stay in physical
units (÷ dt). The one remaining fps dependence is the clip DURATION at a fixed frame count (60 frames
= 2.0–2.5 s), which changes what a clip's correlation averages over, not its scale.

### 7.2 The block

`num_layers 2` (toy: one residual layer gains per-frame accuracy under the correlation loss, four
lose), `position: seconds` (RoPE on `frame_pos_sec × 25`), `window: 0.18` s per layer = ±4 frames at
24–25 fps, ±5 at 30 (never a knife-edge: 4/25 = 0.160 and 5/30 = 0.167 inside, 5/25 = 6/30 = 0.200
outside). `seconds` / `0.18` are the base defaults again; the stopped `tvel_*` configs are pinned to
`index` / `4` so their checkpoints evaluate unchanged. Verified (scratchpad
`tvel2/verify_rope_seconds.py`): keys per query 9 / 9 / 11 / 11 at 24 / 25 / 29.97 / 30 fps
(5–6 at clip edges), exact identity at init in train mode with dropout, time-shift invariance to
1.6e-5 for a 123 s shift, `index` (window 4) and `seconds` (0.18 s, 25 fps) outputs equal to 4.8e-7
with the same random weights, T = 1 runs, and a random perturbation of frame 10 reaches exactly
frames 2–18 through two layers at 25 fps (0–20 at 30 fps). (A first version of that test perturbed
all 64 channels by the same constant — invisible after LayerNorm — and wrongly showed no
propagation.)

### 7.3 Weight calibration (`tvel2/calib_grads.py`, 3 batches of 2 clips, L2 norm of each term's
### gradient over the trainable parameters at weight 1)

| operating point | kp2d@10 | kp3d@5 | orient | pose | betas@0.1 | depth@10 | bearing@2 | root_vel@1 | root_ang_vel@1 | joint_ang_vel@1 |
|---|---|---|---|---|---|---|---|---|---|---|
| init (tvel2 build, 2 layers) | 23.6 | 8.8 | 6.7 | 0.51 | 0.60 | 42.1 | 7.6 | 16.6 | 117.7 | 15.2 |
| tvel_ray best (shrunk head) | 18.0 | 7.2 | 2.3 | 0.66 | 0.45 | 147 | 3.0 | 17.5 | 6.8 | 2.2 |
| **static_ray best (healthy head)** | 3.3 | 2.0 | 0.34 | 0.18 | 0.19 | 19.8 | 0.73 | **26.9** | **4.0** | **1.56** |

At init the two rotation terms sit in the zero-motion clamp regime (the head outputs a constant mean
pose: `r = 0`, gradient set by the clamp) — not representative. Weights were set at the healthy
operating point so that each correlation term carries ~8–9 gradient units and the three together
~50 % of the total (per-frame block ~26.5, dominated by `depth`): **root_vel 0.3, root_ang_vel 2,
joint_ang_vel 6** (block 8.1 + 8.1 + 9.4). The init transient (root_ang_vel 2 × 118 ≈ 235 units for
the first steps) is covered by the 500-step warm-up; the one-epoch smoke run at 1 / 1 / 1 trained
through it (root_ang_vel r 0.72, pred/GT RMS 1.19 after one epoch).

### 7.4 What to watch

`metric_velocity/*_pred_rms / *_gt_rms` (near 1 = no shrink; 0.5 = the old failure), `*_r`,
`metric_global/lifted_jitter` (references: static_ray 56.7, tb_projzero 52.4, tvel_ray 70 at ep 8),
`metric_pose/mpjpe` (65.2 / 72.3 / 104.7), `depth_err` / `depth_bias` (116 / 100 / 240, bias −33),
`dlogz_pred` (0.16 / 0.16 / 0.38; GT 0.275).

### 7.5 Outcome (stopped by the user at epoch 10, ~16:08)

| ep | mpjpe | pa | depth_err (bias) | dlogz_pred | jitter | root_vel r / pred RMS | root_ang_vel r / pred RMS | joint_ang_vel r / pred RMS |
|---|---|---|---|---|---|---|---|---|
| 0 | 373.0 | 216.1 | 105 (+49) | 0.531 | 98.1 | 0.45 / 0.450 | 0.53 / 0.071 | 0.43 / 0.043 |
| 3 | 173.8 | 138.1 | 106 (+37) | 0.504 | 97.2 | 0.46 / 0.437 | 0.78 / 0.506 | 0.50 / 0.215 |
| 6 | 84.3 | 59.4 | 98 (+24) | 0.509 | 95.9 | 0.47 / 0.445 | 0.80 / 0.542 | 0.52 / 0.419 |
| 8 | 73.0 | 52.9 | – | 0.508 | 96.3 | 0.47 / 0.448 | 0.81 / 0.555 | 0.52 / 0.460 |
| 10 | – | – | – | – | – | 0.46 / 0.453 | 0.81 / 0.565 | 0.52 / 0.477 |

GT RMS 0.288 / 0.486 / 0.752. Reading: the shrink is gone — root-rotation amplitude 1.15 of GT, joints
climbing (0.63 at epoch 10, still growing out of the zero init), depth bias positive throughout, and the
pose is the best of any velocity run at equal epoch (73 at epoch 8 vs static_ray 83 / tvel_ray 105). But
NOTHING is smoothed: jitter flat at 96–98, dlogz flat at 0.51 (GT 0.275), root_vel r flat at 0.46.
Branch gain of the two residual layers at epoch 6: 0.065 per layer (tvel_ray: 0.032 at epoch 8), token
third-difference ratio 0.81 through the block — twice the old speed, on the toy's slow-kernel curve
(thousands of steps to a useful gain). The depth signal is also weak by construction: `root_vel` at
weight 0.3 carries ~8 gradient units where the retired depth stencils carried ~60 when they took
static_ray to dlogz 0.16. Run and dumps moved to `trash/output_cleanup_20260905/` with the tvel_ray /
tvel_cliff runs (the forensics dumps of §2 live there now).

## 8. Round 3 (2026-09-05 evening): block learning rate, heavier root_vel, non-residual convex first layer — `tvel3_ray`

User decision: stop `tvel2_ray`, apply the three levers together (`configs/tvel3_ray.yaml`, on top of
tvel2_ray). One-epoch smoke passed (23.26 M trainable, +2.1 M for the convex layer); launched 16:17 on GPUs
0 + 5, log `output/logs/tvel3_ray_20260905_161720.log`, run dir `output/tvel3_ray_<stamp>/`. Output folder
cleaned the same evening: the tvel_ray / tvel_cliff / tvel2_ray runs and the launch / smoke logs went to
`/data3/rikhat.akizhanov/trash/output_cleanup_20260905/`; the five kept runs and the frozen jsons remain.

1. **`optim.temporal_lr_scale 10`** — the temporal bricks (`cross_modal_temporal` / `pose_temporal`
   parameter names) form their own AdamW group at 10 × the base lr (`train/trainer.py::_build_optimizer`;
   groups: base, head copies × `head_lr_scale`, temporal × `temporal_lr_scale`). The toy's branch
   gradient under the correlation loss is consistent (SNR −10), so a larger step should turn into gain.
2. **`root_vel 3`** (× 10) — ~80 gradient units at the healthy operating point, the depth-denoising
   signal the ray stencils used to provide; `root_ang_vel 2`, `joint_ang_vel 6` unchanged.
3. **`convex_first: true`** (`model/rope.py::_ConvexPoolLayer`) — one NON-residual layer in front of the
   residual blocks: RoPE q·k attention over the same ±0.18 s window and validity mask, values = the raw
   tokens (no value / output projection, no skip), heads averaged, a learnable per-head self-logit bias
   (init `convex_self_bias 2.0` → self weight 0.48 over 9 keys, the σ ≈ 0.03 s average the frozen-token
   probe showed to be accuracy-neutral). The frame's token is REPLACED by a convex combination of its
   window, so noise can be removed rather than diluted (§2.5 of the truth doc, TCMR). Per slot (never
   mixes modalities), T = 1 is the identity; the module is deliberately NOT an identity at init any more.
   Verified (scratchpad `tvel2/verify_convex.py`): rows sum to 1, interior self weight 0.484 at init,
   T = 1 exact identity, time-shift invariance 1e-5, receptive field of the convex layer exactly ±4
   frames and ±12 with the two residual layers, K = 2 slots isolated (slot-0 change 0.0 when slot 1 is
   perturbed), gradients reach the bias and q/k, `convex_first: false` bit-identical to before;
   optimizer groups on a dummy model: base 2e-4, head copies 2e-5, temporal 2e-3.

Diagnostics caveat: `scripts/diag_temporal.py` / `branch_gain.py` iterate `module.blocks`; the convex
layer is outside that list (their `bypass` ablation leaves it in place).

4. **`depth_prior: constant`** (added 16:25 after five epochs of the first start). With `frozen` the ray
   head adds the frame's OWN noisy frozen pelvis ray at the output and the head has to cancel that noise
   from the token — possible while the residual stream kept the frame's own frozen-camera embedding, but
   the convex layer replaces the token by a window average, so the per-frame information is gone and the
   prior's noise passes straight through: dlogz sat at 0.53–0.54 (the prior's own level) for epochs 0–4
   while the rotation correlations climbed faster than in round 2 (root 0.80 / joints 0.53 at epoch 3 vs
   0.78 / 0.50). The user asked whether the prior is needed at all; `constant` is the no-prior form (log
   3.5 m is only the centring of the zero-init head). The frozen ray stays a token INPUT
   (`token_inputs.frozen_camera`), averaged by the block like everything else — the no-input ablation
   (`static_ray_noprior`, 215 vs 116 mm depth error) is why it is kept. First start (frozen prior, 16:17,
   epochs 0–4) moved to `trash/output_cleanup_20260905/tvel3_ray_20260905_161730` with its log
   (`tvel3_ray_frozenprior_...`); relaunched 16:24, log `output/logs/tvel3_ray_20260905_162433.log`.

### 8.1 Outcome (constant prior; stopped by the user at epoch 7, ~16:35 — "stop everything for now")

| ep | mpjpe | pa | depth_err (bias) | dlogz_pred | jitter | root_vel r / pred RMS | root_ang_vel r / pred RMS | joint_ang_vel r / pred RMS |
|---|---|---|---|---|---|---|---|---|
| 0 | 385.1 | 207.5 | 1163 (+176) | 0.099 | 7.6 | 0.44 / 0.141 | 0.66 / 0.063 | 0.51 / 0.032 |
| 1 | 299.1 | 229.3 | 507 (+118) | 0.437 | 15.2 | 0.46 / 0.441 | 0.70 / 0.279 | 0.51 / 0.098 |
| 2 | 236.3 | 178.0 | 390 (+178) | 0.389 | 27.4 | 0.45 / 0.403 | 0.67 / 0.388 | 0.50 / 0.235 |
| 3 | 198.4 | 130.4 | 423 (+212) | 0.344 | 35.1 | 0.49 / 0.385 | 0.66 / 0.486 | 0.49 / 0.326 |
| 4 | 164.6 | 102.2 | 311 (+108) | 0.365 | 32.4 | 0.45 / 0.373 | 0.70 / 0.407 | 0.48 / 0.399 |
| 5 | 144.5 | 93.9 | 336 (+228) | 0.357 | 32.4 | 0.48 / 0.372 | 0.72 / 0.471 | 0.48 / 0.413 |
| 6 | 142.8 | 88.7 | 282 (+201) | 0.373 | 28.2 | 0.49 / 0.394 | 0.74 / 0.475 | 0.49 / 0.432 |

(Epoch 0 is the init artefact of the constant prior: every frame at 3.5 m, hence the trivially low
dlogz / jitter.) Reading: with no per-frame output prior the convex block finally smooths — jitter 28
vs 96–98 in rounds 1–2 (GT 6.4), dlogz 0.37 vs 0.51–0.54 (GT 0.275), the root-velocity correlation moved
for the first time (0.49) — but the head struggles to recover the ABSOLUTE depth from the averaged token
(error 282 mm, bias swinging +100…+230; the frozen-prior runs sit at ~100) and the pose converges about
half as fast (143 vs 84 at epoch 6 in round 2). The frozen-prior first start (§8 item 4) had the
opposite profile: pose 132 at epoch 4, depth 98, but dlogz stuck at the prior's 0.54 and jitter 98.
Partial run kept for inspection: `output/tvel3_ray_20260905_162441/` (epochs 0–6; `best.pth` = epoch
6), log `output/logs/tvel3_ray_20260905_162433.log`.

**Open next step (not built):** `depth_prior: frozen_pooled` — apply the convex layer's head-averaged
attention weights to the frozen pelvis ray (bearing + log depth) and use THAT as the output prior. The
prior then carries no per-frame noise the head would have to cancel (the failure of item 4), keeps the
frozen model's absolute depth (the failure of the constant prior), and the head is back to a small
residual. Small change: the convex layer already computes the weights; `network.py` would pool the
frozen ray with them before the head. Everything is stopped; nothing is running on any GPU.
