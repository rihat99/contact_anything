# The five additions of 2026-09-03 (reference)

Every item is an experiment arm on the `configs/temporal_tokens_b8_lr2.yaml` recipe (contact
tokens + SMPL-X head + one RoPE temporal block; 8 clips/step, lr 2e-4, 5 epochs). Baseline =
the recorded `output/temporal_tokens_b8_lr2_20260902_203707` (trained WITHOUT the optimizer
hygiene bundle that is now the `base.yaml` default, so every delta folds the bundle in).
Results: `docs/results.md`.

## 1. Hands in the SMPL-X head — `configs/hands.yaml`

| what | where |
|---|---|
| head regresses the 30 finger joints (6D each), 52-joint BetterHuman body, `q` 211 | `model.smplx.hands: true` (`model/heads.py::SmplxHead`) |
| GT finger rotations | loader key `smplx_hand_rot (30, 3, 3)` (`smplx_joints_world` is `(52, 3)`, body first) |
| losses | `smplx_supervision.loss.hand_pose` (6D MSE), `joint_weights.fingers` (finger weight inside kp2d / kp3d; body joints 1, normalised to sum 1) |
| metrics | `hand_mpjpe` (wrist-translation aligned, mm), `hand_pa_mpjpe` (per-hand Procrustes over wrist + 15 fingers); body metrics unchanged (22 joints, flat-hand vertices via `body_flat`) |

Verified: FK of the kindyn `q` reproduces `joints_world` to 5e-7 m; identity-hand vertices
differ from the 22-joint body by ≤ 8 mm; the no-hands build is unchanged.

## 2. 2D keypoints as an INPUT to the pose token — `configs/keypoints.yaml`

GVHMR-style early fusion. The frame's keypoints, crop-normalised exactly like SAM 3D Body's
`_full_to_crop` (`[-0.5, 0.5]` spans the crop) plus their detector score, go through a
zero-init MLP (`3K -> C -> C`) whose output is ADDED to the pose token before the temporal
block (`model/inputs.py::KeypointInput`, wired in `ContactAnything.forward`). Zero-init keeps
the initial function exactly the frozen one.

| key | meaning |
|---|---|
| `model.token_inputs.keypoints2d.enabled` | on/off (a pose writer: counts as writing the pose token) |
| `source: sapiens` | corpus detections `features/sapiens/<shard>/<scene>/pose.npz` (Goliath-308; the first 70 are the MHR70 set in MHR order — 7.9 px median vs the projected MHR GT over 40 scenes); loader signal group `keypoints2d` -> batch `kp2d_in (B, 70, 3)` px + score, `kp2d_in_valid (B,)` |
| `source: mhr` | the frozen readout's own `pred_keypoints_2d` (score 1) — what deployment has; evaluate a sapiens-trained checkpoint under this source to measure the gap (scratchpad `massive/keypoints_eval_mhr.yaml`) |
| `indices` | MHR70 subset, default the 23 body keypoints 0-20 + 41 + 62 (COCO-17 + toes/heels + wrists) |
| `min_score` | keypoints scoring below are zeroed (3 % of body keypoints, 86 px median error); so are keypoints more than half a crop outside it |
| `noise_std`, `drop_prob` | train-only Gaussian noise (crop units) / per-keypoint drop |

## 3. Temporal span masking — `configs/masking.yaml`

Train-only corruption of the temporal block's input (`model/inputs.py::TokenMasking`): every
frame starts a span with probability `frac / mean span`, spans run `span_min..span_max`
frames, and ALL K tokens of a corrupted frame (pose + contact + ...) are replaced — by a
learned `[K, C]` mask embedding (`replace: mask`) or by the same slots of a random frame of
another clip in the batch (`replace: swap`). The keypoint input is overwritten with them.
Neighbours are untouched, every loss still scores the corrupted frames, eval is clean.
Measured at the defaults (`frac 0.15`, spans 1-5): 14 % of frames masked, span lengths 1-11
(overlaps), 4.5 % of clips unmasked. Keys: `model.token_masking.{enabled, frac, span_min,
span_max, replace}`; requires `cross_modal_temporal`.

## 4. Camera motion in, body-frame root velocity out, roll-out eval — `configs/camera_*.yaml`

| piece | where |
|---|---|
| camera input | signal group `camera` -> batch `cam_twist (B, 6)`: the camera's own twist `[m/s, rad/s]` in the CURRENT camera frame between the clip's sampled rows (`data/climbing_videos/camera.py`: `log(C_t C_{t+1}^-1) / dt`, central average of the two neighbouring increments, one-sided at clip ends); `model.token_inputs.camera_twist.enabled` -> `TwistInput` (zero-init `6 -> C -> C`) ADDED to the motion tokens, or to the pose token when the motion head reads it |
| motion head | `model.motion.source: tokens` (learned token, per-token head) or `pose_token` (flat head on the mixed pose token — route C-A); `model.motion.terms` = the 3-channel triples the head emits, subset of `[vel, acc, ang_vel, ang_acc]` in that order; outputs `out["motion"]["joint_<term>"]` |
| target | `motion_supervision.linear_frame: body` (BVR root body twist `v = R^T p_dot`) and `root_source: smplx` (kindyn SMPL-X pelvis + root rotation — the SMPL-X head's body; `mhr` = the legacy MHR mean-hips root); `standardize` re-measured per frame/root (SMPL-X root, body: `configs/camera_posetoken.yaml`; recipe: 864 train scenes, 273,039 rows, winsorised 0.1-99.9 %, reproduces the recorded gravity-view table bit-for-bit) |
| roll-out eval | `rollout_eval.enabled` -> `model/loss/rollout.py` (eval only, `metric_global/*`): integrates the de-standardized body twist from the first frame's predicted world root (trapezoid `se3_exp`), joints ride on the integrated root; GVHMR kernels verbatim in `utils/gvhmr_metrics.py` |

Metrics (per clip, invalid frames compacted first as GVHMR does; frame-weighted means):
`rollout_wa_mpjpe100` (WA-MPJPE100: similarity alignment per 100-frame chunk, mm),
`rollout_w_mpjpe100` (W-MPJPE100: first-two-frame alignment per chunk), `rollout_rte` (root
translation error over the GT path length, %), `rollout_jitter` (third difference × fps³ / 10,
the clip's real sampled fps rather than GVHMR's hard-coded 30); the same four as `lifted_*` for
the per-frame prediction lifted with the GT camera of EVERY frame (trusting the camera instead
of the network), and `gt_jitter`. Check: the GT twist rolled out from the GT first pose
reproduces the raw GT world joints to 18 mm over 90 frames (10 mm of it integration drift
against the σ 0.12 s smoothed root). Camera twist on the train clip rows: linear std
[0.16, 0.11, 0.33] m/s, angular std [0.05, 0.07, 0.03] rad/s; 5 % of rows are linearly static.

## 5. Jerk smoothness prior — `configs/smooth_{mild,strong}.yaml`

`pose_smoothness` (`model/loss/smoothness.py`): Huber toward ZERO of the predicted jerk, no
target. Root: the SMPL-X root lifted to the world with the frame extrinsics, relative poses
`d[t] = se3_log(T_t^-1 T_{t+1})`, `j = (d[t+1] - d[t] - d[t-1] + d[t-2]) / (2 dt^3)` (linear
m/s³ + angular rad/s³, both under `loss.root`). Joints: root-local `R_root^T (p_j - p_root)` of
the 21 articulated joints (no extrinsics), `j = (-p[t-2] + 2p[t-1] - 2p[t+1] + p[t+2]) / (2 dt^3)`
(`loss.joints`). The Huber runs in units of the GT p75 at the sampler's dt
(`huber_delta_root_lin 20.65`, `_root_ang 52.55`, `_joints 29.35`; scratchpad
`massive/jerk_stats.md`), so both weights are unit-free. A row needs frames `t-2..t+2` valid.
Metrics `metric_smooth/{pred,gt}_rms_{root_lin,root_ang,joints}` (physical units, identical
stencils on the kindyn GT).

Measured on the trained b8_lr2 checkpoint: prediction RMS jerk 409 / 275 / 177 vs GT
50 / 59 / 74 (six test scenes); at weight 1 the prior's gradient norm is 268 (root lin) + 60
(root ang) / 24 (joints) against the SMPL-X loss's 4.4 and the contact loss's 3.1 (four train
batches) — hence `mild` (root 0.004, joints 0.05 ≈ 25-30 % of the SMPL-X gradient) and
`strong` (0.012 / 0.15). GT facts from the statistics (3671 train clips): hips and spine1 are
rigid to the pelvis (root-local jerk 0); jerk scales as dt⁻³ and the raw stencil numerator
does not grow with dt, i.e. part of the GT jerk is per-frame fit noise, not motion.
