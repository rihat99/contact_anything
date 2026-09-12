# Camera motion and root velocity — investigation (2026-09-04)

Why the camera-twist arm (C-A, `configs/camera_posetoken.yaml`, `output/camera_posetoken_20260903_152304`)
predicts the root twist poorly (vel r3d 0.64, ang_vel 0.59; roll-out RTE 8.4 % vs 0.75 % with the GT
twist), and what the data says the fix is. Everything below is measured; scripts live in the session
scratchpad `camera/` (`check_chain_gt.py`, `dump_ca.py`, `analyze_ca.py`, `smooth_bar.py`).

**Status (2026-09-05):** the twist-head route is dropped (§5); `motion_matching` (§4) and the pelvis
anchor stay in the code with `static_matching_anchor_raw` as their reference run; the joint-20 / ×3
arms launched at the end of §9 were stopped by the user and are in
`/data3/rikhat.akizhanov/trash/static_anchor_x3_joint_stopped_20260904/`. The ray camera head
(`camera_ray_2026-09-04.md`) then replaced the CLIFF proxy as the depth target.

## 1. No convention bug in the camera-twist path

Reconstructing the body-frame pelvis velocity from the loader's own camera twist
`(v_c, ω_c)`, the camera-frame pelvis `p_c`, its camera-frame velocity and the cam-from-body rotation,

    v_body = R_cb^T (v_c + ω_c × p_c + ṗ_c),     ω_body = R_cb^T ω_c + ω_rel

reproduces the loader's BVR body twist of the raw GT root on 60 train scenes (15,549 rows):

| quantity | r (3 axes pooled) | RMSE |
|---|---|---|
| linear, vs the raw GT twist | **1.0000** | 0.004 m/s |
| linear, vs the σ 0.12 s smoothed target `motion_gt` | 0.968 | 0.163 m/s (the smoothing gap) |
| angular, vs `motion_gt` | 0.945 | 0.36 rad/s |

The batch `cam_twist` equals a recomputation from the batch extrinsics and `frame_pos_sec` to 1.5e-5
(collate/ordering verified). Signs, frames, dt and the se3-log layout are all consistent with the GT.

## 2. The camera explains little of the body's velocity

Variance split of the GT world pelvis velocity (`share` = ‖term‖² / ‖GT‖²):

| set | camera-only term `R_cw^T (v_c + ω_c × p_c)` | camera-relative term `R_cw^T ṗ_c` | cross |
|---|---|---|---|
| 60 train scenes (body frame) | share 0.15, r 0.28 | share 1.22, r 0.90 | −0.07 |
| 108 test clips (world frame) | share 0.47, r 0.16 | share 1.59, r 0.81 | **−1.05** |

In the corpus the climber moves in the image far more than the camera moves in the world; on the test
set the two terms are strongly anti-correlated (a tracking camera: the person's camera-frame motion
cancels the camera's own), so the body velocity is a small difference of two larger terms. The camera
angular velocity (RMS 0.05 rad/s) is negligible against the body's (0.76-0.9 rad/s). The dominant input
the head needs is therefore the person's camera-frame motion across frames — a temporal derivative of
the per-frame placement — not the camera twist.

## 3. What the C-A motion head actually does (test set, 12,542 rows)

| prediction | vel r3d | vel RMSE (m/s) | error std in camera axes x / y / **z** | ang r3d | ang RMSE |
|---|---|---|---|---|---|
| motion head | 0.642 | 0.321 | 0.158 / 0.177 / 0.215 | 0.587 | 0.620 |
| motion head, camera twist input ZEROED | 0.630 | 0.325 | 0.159 / 0.183 / 0.215 | 0.590 | 0.619 |
| per-frame pose lifted with the GT camera, differentiated raw | 0.539 | 0.640 | 0.187 / 0.177 / **0.586** | 0.705 | 0.819 |
| … root smoothed σ 0.12 s before differentiating | **0.802** | **0.280** | 0.051 / 0.068 / 0.267 | **0.942** | **0.257** |
| … σ 0.25 s | 0.833 | 0.231 | 0.071 / 0.102 / 0.195 | 0.913 | 0.336 |

Regressing the head's world velocity on the two GT terms: `pred = 0.36·cam_only + 0.31·rel − 0.02`
(0.29 / 0.30 with the input zeroed). Three facts:

1. **The head barely uses the camera input** (r3d 0.642 → 0.630 without it) and shrinks both components
   to ~⅓ — the regression-to-the-mean of a Huber regressor that cannot resolve the signal; the roll-out
   drift follows directly from that shrinkage.
2. **The per-frame pose path already holds a far better motion estimate than the head extracts**: the
   camera-lifted trajectory, smoothed with the target's own σ, differentiates to r 0.80 / 0.94 (linear /
   angular), better than the head on both. Its raw derivative is ruined by one thing — **depth**: the
   camera-z error std is 0.59 m/s raw vs 0.05-0.07 m/s laterally after smoothing.
3. Post-hoc smoothing of the lifted trajectory (GVHMR metrics, 108 test clips; the head's roll-out is
   WA 150 / W 262 / RTE 8.4 / jitter 38.6):

   | lifted trajectory | WA-MPJPE100 | W-MPJPE100 | RTE % | jitter (GT 7.2) |
   |---|---|---|---|---|
   | raw | 80.5 | 133.9 | 4.39 | 96.5 |
   | root smoothed σ 0.12 s | 77.1 | 131.1 | 4.05 | 38.7 |
   | root + local joints smoothed σ 0.12 s | 76.8 | 131.0 | 4.05 | 1.4 |

   WA / W barely move under smoothing: they are dominated by low-frequency depth/scale error, not jitter.

**The same head on the 19 static-camera test scenes** (no camera egomotion at all; `scripts/evaluate.py`
with `configs/datasets/climbing_videos_static.yaml`): vel r3d **0.55**, ang_vel 0.64, roll-out RTE 8.7 %,
lifted RTE 5.0 %, MPJPE 58.7 — no better than on moving cameras. The camera's motion is not what limits
the twist head.

**Conclusion.** The twist head is the wrong place to put the motion: it learns a shrunk regression of a
signal the pose path already contains. Making the POSE trajectory itself temporally consistent (the
motion-matching loss, `model/loss/motion_matching.py`) attacks the actual error — per-frame depth wobble —
and gives a world trajectory for free through the known camera (the `lifted_*` metrics ARE its roll-out).
The camera input matters only through the camera-only term, ≤ 15-47 % of the variance, and the head does
not use it anyway; whether the camera twist should be smoothed (raw-minus-σ0.12 s RMS is 55 % of its RMS)
is moot until a model uses it.

## 4. Motion matching (`motion_matching`, built 2026-09-04)

The predicted `pelvis_cam` / `root_rot` lifted with the GT extrinsics, differentiated with the loader's
BVR scheme (`d = se3_log(T_t^-1 T_t+1)`, `v = (d[t-1] + d[t]) / 2dt`, `a = (d[t] − d[t-1]) / dt²`; torch
mirror exact to 1e-7) and matched to `motion_gt` (σ 0.12 s smoothed, standardized by the
`motion_supervision` table); root-local joint velocities / accelerations vs the GT SMPL-X joints
differentiated the same way (m/s, camera cancels); optional Huber vs the motion head's DETACHED twist.
Grad → pose path only. Fed the GT body, the root_vel term floors at 0.44 (raw vs smoothed GT) and the
joint terms at 0 — the optimum is a smooth trajectory. Metrics `metric_matching/*`: r / RMSE of the
pose-derived twist and joint motion vs GT (raw, no smoothing — the trained model must be smooth itself).

## 5. Static-camera subset

`scenes.db` flags `static_camera` (115 train / 19 test curated scenes, 36k / 4.7k frames, 413 / 19
clips at T = 60). Measured end-to-start camera displacement: median 4 mm, p90 3.3 cm (train), a few
outliers up to 1.6-1.9 m / 16° (mislabelled). Dataset yaml key `camera: all | static | moving`
(`configs/datasets/climbing_videos_static.yaml`). User decisions (2026-09-04, after the audit): solve the static case first; the twist head is DROPPED
(the world trajectory is the camera-lifted per-frame prediction, `metric_global/lifted_*`, which
`rollout_eval` now reports without a motion head); raise the matching weights if matching does not do
its job; body frame stays. Probes (both the hands recipe on the static subset, 30 epochs ≈ 1550 steps):
`configs/static_matching.yaml` (ST-M: matching root_vel 1 / root_ang_vel 0.5 / joint_vel 1, no head, no
camera input) and `configs/static_matching_x3.yaml` (weights ×3). The aborted headed probes
(`static_motionhead` = twist head alone on the static subset, killed at epoch 10: vel r3d 0.25,
ang 0.15, MPJPE 86 — the head learns the twist SLOWLY even with no camera motion; `static_matching`
with the head, killed at epoch 8: matching root_vel r 0.48, ang_vel 0.71, lifted jitter 100 → 72) are in
`/data3/rikhat.akizhanov/trash/static_probes_headed_20260904/`.

## 6. Matching gradient balance (2026-09-04, scratchpad `camera/check_matching_grads.py`)

Per-term gradient norm over the trainable params, each matching term at weight 1, 4 static train
batches of one 60-frame clip:

| term | converged `hands` checkpoint | `static_matching` epoch 5 |
|---|---|---|
| SMPL-X loss (all terms, config weights) | 6.1 | 113 |
| of which `cam` (the only absolute-depth anchor) | 0.64 | 9.9 |
| contact | 12.1 | 10.6 |
| matching root_vel | 62.7 | 47 |
| matching root_ang_vel | 13.7 | 107 |
| matching joint_vel | 0.8 | 2.7 |
| matching root_acc / root_ang_acc | 1547 / 339 | 1301 / 2830 |

At weights 1 / 0.5 / 1 the matching terms carry ~90 % of the gradient of a converged model (63 vs 6)
and 100× the `cam` term; with per-group clipping (norm 1) the noisy high-pass derivative gradient sets
the step direction in the SMPL-X head and the temporal block. Observed: `static_matching` MPJPE 190 at
epoch 5 vs 131 for the head-only probe, ×3 weights 268, `cam` loss rising through training, pose-derived
vel r ≈ 0.03 until the pose passes ~160 mm (the headed probe then jumped to 0.5 within 3 epochs).
Balanced arm: `configs/static_matching_bal.yaml` (0.05 / 0.1 / 1.0 → matching gradient ≈ 3.1 + 1.4 +
0.8, i.e. ≈ the SMPL-X loss late in training, ~2 % of it early). Acceleration terms need ~0.001.
Frame check on the C-A dump: body-vs-body (the loss) r 0.533 = world-vs-world 0.539 (orientation error
median 4.9°) — the frame choice is not the problem.

Run log (static subset, 19 test clips): `static_matching` (1 / 0.5 / 1) epoch 11: MPJPE 88.4, pose-derived vel r
0.54 / ang 0.81 / joint 0.61, lifted WA 101 / RTE 4.6 / jitter 73. `static_matching_x3` (killed by the user at epoch
11 for the balanced arm): MPJPE 120, vel r 0.56, ang 0.82, lifted WA 115 / RTE 5.7 / jitter 60 — heavier matching =
smoother but slower pose. `static_matching_bal` (0.05 / 0.1 / 1) launched 18:49. `static_baseline` (no matching,
no head) queued behind the 1× run as the reference.
`static_matching_bal` (0.05 / 0.1 / 1) killed by the user at epoch 14 as too weak: MPJPE 70.5 (the 1× arm needed
epoch 22 for 71.8 — no pose slowdown at balanced weights), vel r 0.55 / ang 0.82 / joint 0.65, lifted WA 91.6 /
RTE 4.95 / jitter 86 (1× arm: 73). Next arm `static_matching_velacc` (19:16): velocity + acceleration matching at
gradient-adjusted weights root_vel 0.2 / root_ang_vel 0.3 / joint_vel 2 / root_acc 0.005 / root_ang_acc 0.01 /
joint_acc 0.3 (≈ 5× the SMPL-X gradient on a converged model, ≈ 0.8× early).

## 7. First static results (2026-09-04 evening; 19 static test clips, per-frame dumps in scratchpad `camera/`)

`static_matching` (1 / 0.5 / 1, 30 epochs) final: MPJPE 69.4, pose-derived vel r 0.58 / ang 0.84 / joint 0.65, lifted
WA 88.8 / RTE 4.5 / jitter 75. Lifted-trajectory analysis against the full-corpus C-A model on the SAME clips:

| model | raw lifted vel r / RMSE | depth-vel err std (raw → σ0.12 s) | jitter (raw → σ0.12 s) | pelvis depth error RMS | per-clip depth bias RMS | mean depth bias |
|---|---|---|---|---|---|---|
| C-A (full corpus, no matching) | 0.556 / 0.526 | 0.50 → 0.27 m/s | 97 → 37 | 0.20 m (4.7 %) | 0.16 m | +0.005 m |
| static_matching 1× | 0.597 / 0.422 | 0.39 → 0.27 m/s | 74 → 32 | **0.40 m (7.7 %)** | **0.38 m** | **−0.22 m** |

Lateral pelvis error is 2-4 cm in both; the fast (< 0.5 s) depth residual is 5.9 cm in both. Matching removed some
frame-to-frame wobble (jitter 97 → 74, raw depth-velocity noise 0.50 → 0.39) but NOT the slow depth error (after
smoothing both models are identical at 0.27 m/s), and the 1× arm's ABSOLUTE depth drifted: a −22 cm mean bias and a
per-clip constant offset of 38 cm RMS — invisible to every reported metric (MPJPE is pelvis-aligned; WA/W-MPJPE and
RTE align the first frames; the derivative loss is blind to a constant offset under a static camera). Whether the
drift is matching's doing or a static-subset effect is settled by `static_baseline` (no matching, same data).

## 8. Static-subset round, final (2026-09-04 ~20:20; 30 epochs ≈ 1550 steps each, 19 static test clips)

| arm | matching weights (vel / ang / joint [+ acc]) | MPJPE | PA | lifted WA / W | RTE % | jitter (GT 6.7) | pose-derived vel r / ang r | pelvis depth RMS / mean bias / per-clip offset |
|---|---|---|---|---|---|---|---|---|
| `static_baseline` (no matching) | — | **62.2** | 44.7 | 95.5 / 139 | 5.59 | 108 | 0.51 / 0.80 | 0.22 m / +0.04 / 0.17 |
| `static_matching` 1× | 1 / 0.5 / 1 | 69.4 | 48.8 | **88.8** / 139 | **4.52** | 75 | **0.58 / 0.84** | 0.40 m / **−0.22** / 0.38 |
| `static_matching_velacc` | 0.2 / 0.3 / 2 + 0.005 / 0.01 / 0.3 | 73.5 | 51.5 | 92.8 / 138 | 4.86 | 73 | 0.55 / 0.84 | 0.27 m / −0.07 / 0.23 |
| `static_matching_bal` (killed ep 14) | 0.05 / 0.1 / 1 | 70.5 @14 | 50.8 | 91.6 / — | 4.95 | 86 | 0.55 / 0.82 | — |
| C-A, full corpus, no matching (reference on the same clips) | — | 58.7 | — | 81.3 / 130 | 4.98 | 97 | 0.56 / 0.81 | 0.20 m / +0.01 / 0.16 |

Post-hoc σ 0.12 s smoothing of the lifted root brings every model to the same place (vel r 0.72-0.78, depth-velocity
error 0.27-0.32 m/s, jitter 24-37) — matching removes part of the frame-to-frame wobble the smoother would remove
anyway, and none of the arms improves the SLOW depth error. Verdicts: (1) velocity matching at 1× buys smoother,
better-integrated trajectories (jitter 108 → 75, RTE 5.6 → 4.5, WA 95 → 89) at a pose cost (MPJPE 62 → 69) and
with a −22 cm absolute depth drift — the baseline proves the drift is matching's (0.22 m, no bias, same data);
(2) acceleration matching adds nothing over velocity matching (jitter 73 vs 75) and costs more pose; (3) the
absolute depth needs its own anchor — hence `smplx_supervision.loss.pelvis` and the new `metric_pose/pelvis_err`,
`depth_err`, `depth_bias`. Launched 20:20: `static_matching_anchor` (1× matching + pelvis 3.0, GPUs 0+5) and
`static_matching_anchor_raw` (same + unsmoothed matching targets, GPUs 4+6).

## 9. Unsmoothed targets + pelvis anchor (2026-09-04 ~22:20)

User decision: the kindyn fit is smooth (GT jitter 6.7), the σ 0.12 s target smoothing is dropped
(`motion_supervision.target_smooth_sec: 0` is now the base default) and EVERY matching run trained on smoothed
targets is retired to `/data3/rikhat.akizhanov/trash/smoothed_target_matching_runs_20260904/` (`static_matching`,
`static_matching_velacc`, `static_matching_anchor`; for the record the smoothed-target anchor arm ended at MPJPE 67.4,
depth 150 mm / +26, jitter 81). The `standardize` table was measured on smoothed twists (raw RMS is ~13 % larger) —
a unit-scale detail, not re-measured.

Valid static arms, 30 epochs, 19 static test clips (pose-derived r now vs the RAW GT twist, so not comparable to
the smoothed-target numbers above):

| arm | MPJPE | PA | depth err / bias (mm) | lifted WA / W | RTE % | jitter | vel r / ang r (raw GT) |
|---|---|---|---|---|---|---|---|
| `static_baseline` (no matching) | **62.2** | 44.7 | 224 / +38 | 95.5 / 139 | 5.59 | 108 | — |
| `static_matching_anchor_raw` (1 / 0.5 / 1, pelvis 3) | 65.0 | 47.0 | **145 / +25** | **89.6 / 130** | **4.75** | **84** | 0.51 / 0.81 |

The anchor removes the depth drift entirely (per-clip offset 0.166 m vs the retired 1× arm's 0.376, baseline 0.174)
and matching still smooths (jitter 108 → 84, RTE 5.6 → 4.75, WA 95 → 90) for 2.8 mm of pose. Post-hoc smoothing
of the anchor_raw root alone takes its jitter 84 → 32: the root wobble is ~60 % of the remaining jitter, the
root-local joints the rest — and the joint term at weight 1 has a gradient of ~0.8 (effectively off).
Launched 22:20 on the idle pairs: `static_matching_anchor_joint` (anchor_raw + joint_vel 20) and
`static_matching_anchor_x3` (root 3 / 1.5, joint 20, pelvis 6 — heavier matching now that depth is anchored).
