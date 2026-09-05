# Ray camera head: bearing + log-depth on the frozen measurement (2026-09-04)

Follow-up of `docs/jitter_2026-09-04.md` §14 (the high-pass diagnostic of the lift inputs).
Everything here runs on the STATIC-camera subset of the ClimbingVideos corpus.

**Status (2026-09-05 cleanup):** kept in the code — the ray head (`model.smplx.camera: ray`,
`depth_prior: frozen | constant`), the bbox / frozen-camera token inputs, `kp2d_space: image`,
the depth / bearing (vel / acc) terms and the `dlogz_*` metrics; R1 (`configs/static_ray.yaml`,
`output/static_ray_20260904_171141`) is the kept reference. Removed — the prior kernel
(`depth_prior: frozen_smoothed`, R3) and the output smoother (explicit smoothing, excluded by the
user); R2 / R3 and their configs are in `/data3/rikhat.akizhanov/trash/cleanup_20260905/`.

## 1. Why the target changes

The diagnostic measured the per-frame high-pass energy (RMS Δlog per frame, %) of every factor of
the CLIFF lift `t_Z = 2f / (b s)` on the static test clips:

| quantity | test (19 clips) | train (80) |
|---|---|---|
| log b (crop side) | 3.32 | 1.59 |
| log s_GT = log 2f/(b t_Z,GT) | 3.31 | 1.59 |
| log s_pred | 3.25 | 1.61 |
| log(b s)_GT | 0.48 | 0.22 |
| log(b s)_pred | 0.67 | 0.55 |
| GT log t_Z | 0.37 | 0.21 |
| pred log t_Z | 0.60 | 0.56 |
| frozen SAM3D log t_Z | 0.57 | 0.53 |

`r(Δlog s_GT, Δlog b) = −0.99`: the proxy target `s` is 95 % crop jitter by construction, so a
temporal block trained on it must transmit each frame's `b` exactly and cannot average depth
across frames (the earlier bbox-input probe and every smoothing loss on `s` were fighting the data
term). The bbox-free product `b s` still moves 0.55 %/frame for the prediction vs 0.22-0.28 for the
GT, uncorrelated with the GT (r −0.07), identical to the frozen SAM3D's own depth jitter: the
token's per-frame depth noise is white, the GT depth is clean, and nothing in the network was ever
asked to average the one against the other.

Decision (user, 2026-09-04): change the camera target to the pelvis **ray** — bearing
`(x/z, y/z)` and log depth — so that the target moves 0.25 %/frame instead of 3.3, hand the block
the frozen SAM3D depth as an explicit per-frame measurement, and match depth velocity/acceleration
to the clean GT. Receptive field checked: the cross-modal block is 4 bidirectional RoPE layers with
a ±2.5 s window; training clips are 60 frames at the 25 fps auto stride (2.4 s), eval clips 120
frames. Every frame sees > 1 s on each side, nothing to change.

## 2. What was built

* `model.smplx.camera: ray` (`model/heads.py::SmplxHead`): `proj_cam` outputs
  `Δ(rx, ry, d)`; pelvis `= exp(d) · (rx, ry, 1)` with `(rx, ry, d) = prior + Δ`. The crop box never
  enters the lift. `depth_prior: frozen` = the frozen readout's own pelvis ray of the frame
  (`utils.geometry.frozen_pelvis_camera`: MHR mean hips + `pred_cam_t`, detached);
  `constant` = `(0, 0, log depth_prior_m)`. Under `ray` the head emits `ray` and `cam = None`; the
  output smoother (`pelvis_space: ray`) composes unchanged.
  The frozen SAM3D measurement was chosen over training our own per-frame head (user decision):
  zero training, available at deployment, the same white noise as the token, the MHR-vs-SMPL-X pelvis
  gap is a small near-constant offset — measured on a train batch: frozen depth − GT depth mean
  0.000 m, |.| 0.033 m at 4.9 m (the corpus GT was itself initialised from SAM3D).
* `model.token_inputs.frozen_camera` (`model/inputs.py::FrozenCameraInput`): the frozen pelvis
  ray → zero-init MLP → added to the pose token before the temporal block, next to the CLIFF bbox
  vector `model.token_inputs.bbox` (`[(cx−px)/f, (cy−py)/f, log(b/f)]` — the log form is the
  additive one for a log-depth output; it existed but was OFF in every static config). Both are
  exact identities at init; both are pose writers in `train/config.py`.
* `smplx_supervision`: `kp2d_space: image` measures the 2D keypoint error on the FULL image in
  bearing units (px / f — the ray's own space, independent of the per-frame crop side; the crop
  version's scale rides on `b`). New terms on the lifted pelvis ray (any camera parametrization):
  `depth` / `bearing` Huber, `depth_vel` / `bearing_vel` (forward difference over the clip's real
  seconds vs the GT's), `depth_acc` / `bearing_acc` (central second difference). `cam` (the CLIFF
  proxy) must be 0 under `ray`.
* Metric `metric_pose/dlogz_pred|gt|err`: RMS frame-to-frame step of the pelvis log depth in
  %/frame — the prediction's, the GT's, and that of their difference (noise alone, real motion
  removed). Logged next to the cubed world jitter, which the far-person clips dominate.
* Smoke (`scratchpad camera/ray_smoke.py`, 4 train batches of 2 clips, untrained): ray round trip
  exact (1e-6), pelvis == frozen pelvis at init (5e-7), every term finite, vel/acc masses 118/116 of
  120 rows. Per-term gradient norms at weight 1 (the weight calibration in `configs/static_ray.yaml`):
  kp2d 4.2, kp3d 1.4, orient 7.7, pose 0.6, hand_pose 0.6, betas 7.4, depth 2.3, bearing 5.1,
  depth_vel 1.5, depth_acc 57.8, bearing_vel 0.15, bearing_acc 5.4, pelvis 57.2, contact 2.7.
  The metre-space `pelvis` term is dropped (weight 0): its gradient scales with z, i.e. weights the
  far people — the log-depth term is the anchor instead.

**Adversarial review (read-only Opus agent, 2026-09-04 17:50):** no correctness bug found. Verified:
`pred_keypoints_3d + pred_cam_t` IS the frozen model's absolute OpenCV camera-metre pelvis (the MHR head
applies the native flip itself, the camera head uses the true focal + principal point), hips indices
9/10, stencils never cross a clip, the term set is fixed at every `T` (DDP-safe), the sum-of-squares
`dlogz` reduction is exact across batches/ranks, no gradient reaches the frozen base, every consumer of
`out["smplx"]["cam"]` is guarded. Fixed on its advice: the frozen pelvis depth is clamped at 0.25 m
before the ray (the frozen camera head never clamps its scale); `frozen_smoothed` requires `camera:
ray` (else its kernel scalars would be unused under DDP); the three kept static configs pointed their
`frozen_metrics` at the 108-scene json (now the 16-scene one); stale docstrings/comments. Noted, no
action: at init the camera terms deliver exactly zero gradient to the block and the input MLPs
(`proj_cam`'s last layer is zero-init), so the measurement input only starts learning once `proj_cam`
has moved — a slow start, not a block. `prior_smoother` averages LOG depth, the output smoother
averages metres.

## 3. Static split change (scenes.db, backed up as `scenes/scenes.db.bak_20260904_static_flags`)

Whole-scene per-frame focal check of all 134 static scenes: five carry per-frame focal JUMPS
(max |Δlog f| 8-18 %/frame) with a camera centre wandering 0.1-1.6 m — reconstructions of a camera
that was not static, and the GT depth jumps with f (§14). Flipped to `static_camera = 0` (user
decision): test `075EYBxcBtA_0011` (f range 33.6 %), `075EYBxcBtA_0003` (29.3 %),
`075EYBxcBtA_0022` (14.1 %); train `Qqst4r-FT1w_0111` (15.8 %), `H4hAegl_wfM_0006` (9.7 %).
Kept: scenes whose f drifts smoothly (per-frame < 0.3 %, camera spread < 2 cm, e.g.
`Ul96DmN2M3s_0011` 11.1 % range, `3e_DbnS7yS0_0054` 13.9 %) — the GT depth co-drifts smoothly, not
noise. Static set now 113 train / 16 test scenes. Pre-2026-09-04 static numbers (19 test scenes)
are not comparable; the kept checkpoints were re-scored:

| 16-scene static test | frozen SAM3D | static_baseline | SM-6 (`static_sm_split_acc`) |
|---|---|---|---|
| lifted_jitter (GT 6.35) | — | 118.0 | 9.10 |
| dlogz_pred / gt / err (%/frame) | 0.596 / 0.278 / 0.597 | 0.583 / 0.278 / 0.586 | 0.174 / 0.278 / 0.241 |
| mpjpe / pa / pve (mm) | 57.9 / 43.8 / 74.5 | 60.6 / 44.5 / 77.4 | 60.5 / 44.4 / 79.2 |
| accel (m/s²) | 11.8 | 10.3 | 2.6 |
| pelvis_err / depth_err (mm) | 105 / 97 | 132 / 125 | 120 / 113 |
| lifted WA-MPJPE100 / RTE | — | 94.0 / 5.63 | 79.9 / 4.11 |
| contact F1 | — | 0.876 | 0.869 |

Reading: SM-6's depth step (0.174) is BELOW the GT's (0.278) — the output smoother removes real
motion too, and the noise that remains (dlogz_err 0.24) is what keeps its jitter above the GT. The
frozen model has the best absolute depth (97 mm) of the three; the ray head starts exactly there.

## 4. Runs

| run | config | GPUs | what |
|---|---|---|---|
| R1 `static_ray` | `configs/static_ray.yaml` | 0+5 | ray head, frozen prior + frozen input + bbox input, image kp2d, depth/bearing vel+acc |
| R2 `static_ray_noprior` | `configs/static_ray_noprior.yaml` | 4+7 | R1 without the measurement: constant prior (4 m), no frozen input |

| R3 `static_ray_smprior` | `configs/static_ray_smprior.yaml` | 6 (single GPU, 2 clips/step) | R1 with the frozen ray DENOISED by a learned convex Gaussian kernel (`depth_prior: frozen_smoothed`, σ 0.1 s init, plain average) before the residual |

R1/R2: 30 epochs, lr 2e-4, no smoother, no motion matching, 4 clips/step (2 per GPU — the SM-6 recipe; the
8-clip recipe OOMs while other users hold 10-18 GB per card).

## 5. Results

**R1 vs R2, epochs 0-8 (16-scene test, EMA weights):**

| epoch | R1 jitter | R1 dlogz_pred / err | R1 depth_err (bias) mm | R2 jitter | R2 dlogz_pred / err | R2 depth_err (bias) |
|---|---|---|---|---|---|---|
| 0 | 88.7 | 0.458 / 0.472 | 139 (+86) | 20.2 | 0.045 / 0.273 | 1229 (+429) |
| 2 | 79.0 | 0.373 / 0.402 | 225 (+122) | 67.4 | 0.265 / 0.329 | 589 (+200) |
| 4 | 76.8 | 0.364 / 0.392 | 221 (+66) | 75.3 | 0.324 / 0.363 | 295 (+89) |
| 6 | 73.4 | 0.322 / 0.359 | 170 (+69) | 73.7 | 0.297 / 0.347 | 232 (+80) |
| 8 | 68.0 | 0.270 / 0.331 | 151 (+69) | 71.6 | 0.265 / 0.330 | 215 (+78) |

R1 and R2 are the same run on jitter and depth step: the measurement (as prior AND as token input) buys
nothing through epoch 8, and both sit far above SM-6's trajectory at the same epochs (24 at epoch 4, 12
at epoch 8). `scripts/diag_temporal.py` on R1's epoch-5 checkpoint (8 test clips): the block attends
NEAR-UNIFORMLY over the whole ±2.5 s window (cross-frame mass 0.989, ~95 effective frames, mean |dt|
1.03 s) with tiny gates (|γ_attn| 0.14 L2, 0.012 max element; |γ_ffn| 0.08) — it pools the clip, it
does not locally smooth, and while the gates are that small the q/k projections learn too slowly to
shape a local kernel (the zero-gate chicken-and-egg that kept every earlier run's block a per-frame
fix). Output sensitivity full vs same-frame: 30 mm on raw joints, 1.8 mm hip-aligned — what little the
block does is global placement. The residual head therefore cannot cancel each frame's own measurement
noise (`d = d_SAM + Δd` needs `Δd = −noise`, which needs the block to average `d_SAM` across frames
AND the zero-init input MLP to have embedded it — neither happened in 8 epochs). Absolute depth is also
WORSE than the frozen start (151-215 mm vs 97): the trained residual drifts the depth by +7 cm on
average.

**R3** puts the averaging where it is structural: the prior ray is the frozen ray passed through the
learned convex kernel (the SM-6 mechanism, σ learned at `smoother_lr_scale`), and the residual +
depth vel/acc matching can add real motion back (SM-6's over-smoothing is the thing to beat). Smoke:
the smoothed prior's depth step is already ~2-3× smaller than the frozen one at init; kernel scalars get
gradients (log σ_depth −0.26 on the first batch). Launched 17:29 on GPU 6 alone (2 clips/step — both
GPU pairs were busy).

**R3 through epoch 5 (single GPU, 2 clips/step):** jitter 28.9, 34.2, 39.8, 42.1, 43.7, 44.0; dlogz_pred
0.209 → 0.159 (GT 0.278); depth_err 97-100 mm (frozen 97), bias +10-26. The prior kernel learned
σ_bearing 0.08 s, σ_depth 0.29 s (self logits 0.8 / 0.4): it widened the depth kernel well past the
0.1 s init — over-smoothing the depth (below the GT's own step) while the jitter ROSE. R4 (kernel on
the output) was launched and killed on the user's instruction (no output kernel).

### 5.1 Where the jitter is now (scratchpad camera/dump_ray.py + ray_decomp.py, 16 test clips)

Per-frame high-pass of the pelvis log-depth components (×100; d1 = RMS first difference per frame,
d3 = RMS third difference, the quantity the cubed jitter sees; d3/d1 = 3.16 for white noise; lag-1
autocorrelation of d1: −0.5 white, ~1 smooth):

| R3 ep5 | d1 | d3 | d3/d1 | ac | | R1 ep16 | d1 | d3 | d3/d1 | ac |
|---|---|---|---|---|---|---|---|---|---|---|
| frozen ray | 0.544 | 1.367 | 2.5 | 0.01 | | frozen ray | 0.544 | 1.367 | 2.5 | 0.01 |
| prior (kernel) | 0.155 | 0.034 | 0.2 | 0.96 | | residual | 0.457 | 1.194 | 2.6 | −0.06 |
| residual | 0.038 | 0.081 | 2.1 | 0.23 | | final | 0.184 | 0.379 | 2.1 | 0.22 |
| final | 0.159 | 0.076 | 0.5 | 0.90 | | GT | 0.278 | 0.143 | 0.5 | 0.89 |
| GT | 0.278 | 0.143 | 0.5 | 0.89 | | final − GT | 0.284 | 0.399 | 1.4 | 0.49 |
| final − GT | 0.231 | 0.162 | 0.7 | 0.85 | | | | | | |

R3's final depth is SMOOTHER than the GT in both first and third difference: depth is no longer a
jitter source (it over-smooths instead — the error still moves 0.23 %/frame because real motion was
averaged away). R1 (epoch 16) did learn to cancel the frozen noise partly (residual anti-correlated
with it, final d3 0.38 vs frozen 1.37) — slowly, through the zero-gated block.

Oracle decomposition — swap one predicted component for its GT (or for the prior) and recompute the
GVHMR jitter (10 m/s³; GT 6.35) on all 22 joints / the 6 torso joints / the 6 extremities:

| variant | R3 ep5 all / torso / extrem | R1 ep16 all / torso / extrem |
|---|---|---|
| pred | 43.9 / 24.0 / 79.1 | 57.8 / 40.2 / 90.4 |
| GT depth | 43.5 / 22.9 / 79.4 | 49.8 / 29.2 / 87.1 |
| GT pelvis (depth + bearing) | 42.2 / 20.5 / 79.2 | 45.8 / 21.8 / 87.0 |
| GT root rotation | 34.2 / 15.8 / 67.6 | 52.4 / 35.6 / 84.4 |
| GT local pose (articulation) | 35.9 / 23.0 / 57.6 | 51.3 / 40.6 / 70.1 |
| only pelvis predicted (rot + local GT) | 14.4 / 12.5 / 17.8 | 36.2 / 35.0 / 38.7 |
| only depth predicted | 10.7 / 8.8 / 14.3 | 27.9 / 26.5 / 30.4 |
| only bearing predicted | 11.5 / 9.7 / 14.8 | 20.8 / 19.3 / 23.6 |
| only root rotation predicted | 33.0 / 18.7 / 56.1 | 34.7 / 19.7 / 58.9 |
| only local pose predicted | 31.7 / 11.2 / 67.6 | 37.2 / 12.6 / 80.4 |
| all GT | 6.35 / 4.4 / 9.8 | 6.35 / 4.4 / 9.8 |
| frozen raw ray instead of the head's | 106.4 | 107.6 |

Reading: in R3 the camera is fixed — GT depth changes the jitter by 0.5, the whole pelvis by 1.7, and
the pelvis alone (everything else GT) sits at 14 vs GT 6.35 (mostly bearing now). What is left is
the per-frame POSE readout: the root rotation alone gives 33 and the articulation alone 32 (together
~44 in quadrature). Both are per-frame outputs of the pose head on the same token — the same white
per-frame noise the depth had, now in orientation and joint angles, hitting the extremities hardest
(79 vs torso 24). R1 shows the same rotation/articulation floor (35 / 37) under a still-noisy pelvis.

The frozen SAM3D refit itself (features/sam3d smplx_params, same clips; scratchpad camera/frozen_dump.py):
jitter 126 / torso 114 / extremities 150; only-depth 110, only-bearing 25, only-rotation 44, only-local 58.
So the trained heads already cut the articulation share (58 → 32) and the rotation share (44 → 33)
relative to the frozen refit, and the ray head removed the depth share (110 → 11); what remains is the
per-frame noise of the same token expressed in orientation and joint angles.

### 5.2 Is the input the source? Crop-track probe (scratchpad camera/crop_probe.py, frozen model only)

The frozen model re-run on the 16 test clips through the IMAGE path (no embedding cache) with the corpus
SAM3 bbox track replaced per mode; per-frame noise of its outputs (RMS first / third difference over the
valid rows; rot = geodesic step of `global_rot` in degrees; kp = root-relative MHR70 keypoints in mm;
jitter = GVHMR jitter of the world-lifted MHR keypoints, prediction only):

| crop track | crop step Δlog b (%/fr) | depth d1 / d3 (%) | bearing d1 / d3 | rot d1 / d3 (°) | kp d1 / d3 (mm) | world jitter | depth err (mm) |
|---|---|---|---|---|---|---|---|
| raw corpus boxes | 1.38 | 0.543 / 1.365 | 0.260 / 0.249 | 2.15 / 3.99 | 33.7 / 67.2 | 126.2 | 99 |
| Gaussian σ 0.3 s on centre + size | 0.57 | 0.531 / 1.310 | 0.259 / 0.232 | 2.12 / 3.80 | 33.8 / 69.3 | 120.0 | 99 |
| Gaussian σ 1.0 s | 0.23 | 0.528 / 1.297 | 0.259 / 0.246 | 2.13 / 3.81 | 33.7 / 72.0 | 118.2 | 97 |
| union of the boxes within ±0.5 s | 1.00 | 0.555 / 1.362 | 0.259 / 0.236 | 2.23 / 4.38 | 34.0 / 66.7 | 127.5 | 100 |
| one fixed box per person and scene | 0.00 | 0.632 / 1.465 | 0.254 / 0.226 | 2.29 / 4.26 | 34.3 / 65.6 | 136.4 | 125 |
| raw boxes, NO mask prompt (mask_score 0) | 1.38 | 0.546 / 1.376 | 0.260 / 0.250 | 2.16 / 4.00 | 34.1 / 70.0 | 128.4 | 99 |
| raw boxes, cached backbone maps averaged in time, σ 0.08 s | 1.38 | 0.827 / 1.989 | 0.284 / 0.388 | 1.47 / **0.35** | 22.3 / **13.1** | 114.7 | 102 |
| raw boxes, maps averaged σ 0.2 s | 1.38 | 1.052 / 2.045 | 0.299 / 0.387 | 1.21 / **0.32** | 20.1 / **15.6** | 117.6 | 117 |
| **σ 1.0 s boxes + maps averaged σ 0.08 s** | 0.23 | **0.281 / 0.215** | 0.234 / **0.034** | 1.47 / **0.25** | 22.2 / **12.2** | **19.5** | 95 |
| σ 1.0 s boxes + maps averaged σ 0.2 s | 0.23 | 0.263 / 0.209 | 0.203 / 0.033 | 1.22 / 0.21 | 19.9 / 14.1 | 18.7 | 96 |
| σ 1.0 s boxes + maps averaged σ 0.04 s | 0.23 | 0.319 / 0.273 | 0.243 / 0.043 | 1.63 / 0.56 | 24.3 / 17.2 | 25.6 | 96 |
| σ 1.0 s boxes + maps averaged σ 0.12 s | 0.23 | 0.267 / 0.211 | 0.224 / 0.033 | 1.37 / 0.20 | 21.2 / 12.8 | 18.8 | 94 |
| σ 1.0 s boxes + maps averaged σ 0.3 s | 0.23 | 0.266 / 0.221 | 0.182 / 0.032 | 1.06 / 0.22 | 17.9 / 11.7 | 18.9 | 102 |
| fixed box per scene + maps averaged σ 0.08 s | 0.00 | 0.323 / 0.248 | 0.225 / 0.040 | 1.49 / 0.31 | 22.9 / 13.5 | 22.2 | 125 |
| σ 0.3 s boxes + maps averaged σ 0.08 s | 0.57 | 0.292 / 0.280 | 0.235 / 0.033 | 1.47 / 0.23 | 22.0 / 10.1 | 20.3 | 96 |
| (GT, for reference: SMPL-X camera-frame root rotation, 22 root-relative body joints) | | 0.278 / 0.143 | 0.241 / 0.060 | 1.65 / 0.34 | 17.8 / 6.7 | 6.35 | 0 |

Reading: the crop-track jitter is NOT where the noise enters. Cutting it 6× (σ 1.0 s) lowers the
frozen model's depth / rotation / jitter noise by only 5-6 %; removing it entirely (one fixed box)
makes every output noisier (the person is smaller in the crop) — the per-frame noise is intrinsic to
the frame → backbone → decoder mapping (appearance change between consecutive frames), not to where
the crop sits. Stabilising the boxes is worth ~5 %, not the 10× the jitter target needs.
The per-frame SAM3 mask prompt is not a source either: dropping it entirely leaves every noise
number unchanged (the decoder's mask conditioning is inert for noise).

**Where the noise is made — the backbone maps.** Averaging the cached `[1280, 32, 32]` backbone
output across neighbouring frames (σ 0.08 s ≈ ±2 frames, raw crops) removes the orientation noise
11× (rot d3 3.99 → 0.35°) and the articulation noise 5× (kp d3 67 → 13 mm) — so those two are
created by frame-to-frame variation of the backbone features, not by the decoder — while the DEPTH
and bearing noise get WORSE (1.37 → 1.99, 0.25 → 0.39): the maps live in CROP coordinates and the
raw crop jumps 1.4 %/frame, so averaging misaligned maps blurs the person's apparent size and
position in the crop, exactly the two quantities `s`, `tx`, `ty` read. Aligned-crop + averaged-map
variants: with the crop track smoothed (σ 1.0 s, so consecutive maps are aligned to 0.23 %/frame) AND
the maps averaged over ±2 frames, the FROZEN model — no training — drops from jitter 126 to 19.5 with
the depth error unchanged (95 vs 99 mm): depth step 0.28 %/frame = the GT's own, bearing/rotation/joint
third differences 7-16× lower. The per-frame noise of every output is made in the backbone maps and
is removed by averaging ALIGNED maps over a ~0.1 s window; the remaining 19.5 vs 6.35 is what a
±2-frame window cannot average (and any real-motion blur it adds).
σ sweep (σ 1.0 s boxes): 0.04 s → 25.6, 0.08 → 19.5, 0.12 → 18.8, 0.2 → 18.7, 0.3 → 18.9 (depth
error 96 / 95 / 94 / 96 / 102 mm): the knee is at ~0.1 s; beyond it the jitter floor (~19) does not
move and the depth bias starts drifting positive. At σ ≥ 0.12 s the orientation third difference is
already BELOW the GT's (0.20° vs 0.34°) — the root rotation is over-smoothed — while the
root-relative keypoints stay ~2× the GT's (12.8 vs 6.7 mm; MHR70 vs 22 joints, so only indicative).

### 5.3 Is the reconstructed camera a noise source? (scratchpad camera/cam_probe.py, user question)

Per-frame motion of the reconstructed `cam_from_world` / intrinsics on the 16 static test clips:
camera centre 1.4 mm/frame (third difference 4.0 mm), rotation 0.026°/frame (0.074°), focal
0.066 %/frame. Lifting with the raw per-frame extrinsics vs the per-clip MEAN camera vs a σ 0.3 s
smoothed camera:

| trajectory | raw ext | mean ext | smoothed ext |
|---|---|---|---|
| R3 ep5 prediction | 43.9 | 42.2 | 42.2 |
| frozen SAM3D refit | 126.1 | 126.2 | 126.2 |
| GT → camera (raw) → world | 6.35 (= GT) | 11.5 | 11.5 |

The camera track accounts for ≤ 2 of R3's 44 (and nothing of the frozen 126). The camera-frame
training target does carry the camera's own per-frame jitter (a perfect camera-frame predictor
lifted with a constant camera would score 11.5, not 6.35), but the evaluation lifts with the same
per-frame extrinsics, so that component cancels. Camera smoothing does not remove the noise.

**R1 final (epoch 29, EMA):** jitter 56.7, dlogz_pred / err 0.161 / 0.272, mpjpe 65.2, depth_err
116 (+53 bias), contact F1 0.852 (baseline 0.876). Slowly denoised the depth through the block
(0.60 → 0.16 %/frame) but never below the pose-readout floor; absolute depth worse than the frozen
start and contact F1 down 2.4 points.

### 5.4 R3 final (epoch 29, EMA; single GPU, 2 clips/step)

jitter 51.3 (rose monotonically from 28.9 at epoch 0), dlogz_pred / err 0.160 / 0.247, mpjpe 65.5,
depth_err 109 (+42 bias), contact F1 0.872. Kernel learned σ_bearing 0.074 s, σ_depth 0.333 s.
Decomposition (scratchpad camera/v3_r3_final_test.npz):

| only this component predicted | R3 ep5 | R3 final |
|---|---|---|
| all predicted | 43.9 (torso 24 / extrem 79) | 51.2 (28 / 92) |
| depth | 10.7 | 14.8 |
| bearing | 11.5 | 11.6 |
| root rotation | 33.0 | 34.6 |
| articulation | 31.7 | 42.1 |

Final depth: d1 0.160, d3 0.176 (GT 0.278 / 0.143) — smooth, but the residual on the smoothed
prior grew (d3 0.08 → 0.18, white). The jitter growth over training is the ARTICULATION share
(32 → 42; extremities 79 → 92) as the per-frame pose head sharpens for mpjpe (83 → 65 mm): every
per-frame accuracy gain on a noisy token buys per-frame noise in the joint angles. Same mechanism as
the depth before the ray head, now in the pose channels.

## 6. Summary (2026-09-04 evening)

1. The ray camera head + frozen measurement removes depth as a jitter source (110 → 11-15) and
   keeps the frozen model's absolute depth (R3 97-109 mm vs baseline 125), but the temporal block
   never learned to average anything (near-uniform pooling, gates ~0.01), so the measurement is
   only useful through the explicit prior kernel — which over-smooths (σ_depth 0.33 s).
2. With depth fixed, the jitter floor (44-51 vs GT 6.35) is root ROTATION (~34) and ARTICULATION
   (32-42): the same white per-frame token noise, now in the pose readout, worst at the extremities.
3. That noise is NOT from the bbox track, the mask prompt or the reconstructed camera (each ≤ 5 %).
   It is created in the backbone feature maps: averaging ALIGNED maps over ~0.1 s (σ 1 s crop track
   + σ 0.08-0.12 s map average) takes the FROZEN model from 126 to 19 with no training and no
   depth-accuracy cost; the crop alignment is essential (raw crops: depth gets worse).
4. Open (user's call): (a) an aligned-crop embedding cache + input-map averaging as the data-side
   fix (the static subset is 134 scenes), then the heads on top; (b) make the block do the averaging
   (needs a locality prior / larger gates — it has not learned it in any run); (c) both. Output
   kernels are excluded by the user.
