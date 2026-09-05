# Results

Every number below is from `scripts/evaluate.py` on the **annotated test split** under one
protocol: 108 scenes, ONE clip per (scene, person) = the longest valid run, `stride: auto`
(≈ every frame at 25 fps), capped at `data.eval_max_frames = 120` → 12,868 scored frames.
Pose metrics follow WHAM/GVHMR (22 SMPL-X body joints, hips-mean alignment; `mpjpe` /
`pa_mpjpe` / `pve` in mm, `accel` in m/s² at the real frame dt). Contact metrics are micro
over the six kindyn groups at threshold 0.5 (`f1` / `precision` / `recall` / `iou`),
`P@R0.9` = precision at 90 % recall interpolated on the threshold curve, per-group F1 in the
order LH, RH, LF (toe), RF, LA (heel), RA. Transcripts: `output/logs/<run>_eval.log`.

**Historical page (2026-09-05 simplification).** Every run below is in
`/data3/rikhat.akizhanov/trash/` (`runs_pruned_20260904/`, `cleanup_20260905/runs_removed/`,
`simplify_20260905/output/`, …) and every config named below except `baseline.yaml` (the former
`hands.yaml`) and `static_ray.yaml` (now without the frozen depth prior, the bbox / frozen-camera
inputs and the vel/acc terms) has been removed from the code along with the mechanisms they
exercised (MHR pose path, motion head, matching / velocity / smoothness losses, block variants).
The numbers stand as the record of what those mechanisms did. The static-camera jitter round
(2026-09-04/05) is summarised in the last section; the round write-ups are in `docs/history/`.

## Main table (2026-09-02 runs, evaluated 2026-09-03)

| run | config / checkpoint | mpjpe | pa | pve | accel | f1 | prec | rec | iou | P@R0.9 | LH | RH | LF | RF | LA | RA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen SAM3D (SMPL-X refit) | `output/frozen_sam3d_smplx.json` | 61.07 | 44.06 | 78.08 | 11.93 | – | – | – | – | – | – | – | – | – | – | – |
| SMPL-X probe (per frame) | `smplx_probe.yaml` / `smplx_probe_20260902_093206` ¹ | **57.60** | **38.86** | **74.74** | 11.15 | – | – | – | – | – | – | – | – | – | – | – |
| temporal, pose token | `temporal_posetoken.yaml` / `temporal_posetoken_20260902_132430` | 65.08 | 45.74 | 85.72 | 10.56 | 0.909 | 0.911 | 0.907 | 0.834 | 0.915 | 0.967 | 0.968 | 0.833 | 0.885 | 0 | 0 |
| temporal, contact tokens | `temporal_tokens.yaml` / run trashed 2026-09-03 ² | 60.69 | 41.42 | 78.43 | 10.49 | 0.917 | 0.913 | 0.920 | 0.847 | 0.925 | 0.963 | 0.970 | 0.878 | 0.883 | 0 | 0 |
| temporal, contact tokens, 8 clips/step, lr 2e-4 | `temporal_tokens_b8_lr2.yaml` / `temporal_tokens_b8_lr2_20260902_203707` | 60.07 | 41.21 | 77.86 | **10.47** | **0.919** | **0.917** | **0.921** | **0.850** | **0.926** | 0.964 | 0.971 | 0.885 | 0.884 | 0 | 0 |

¹ The probe's own config strides every 5th frame (`clip: {frames: 1, stride: 5}`); under THAT
protocol it scores 60.40 / 39.88 / 77.04 and accel 1.71 (dt = 0.2 s — not comparable). The
row above re-evaluates the same checkpoint at `stride: auto` so it scores the same frames as
the other rows (`output/logs/smplx_probe_evalauto.log`).
² Numbers from its 2026-09-02 evaluation transcript (kept in
`/data3/rikhat.akizhanov/trash/output_housekeeping_20260903/temporal_tokens_eval.log`).

### Training budgets (not equal — read the table with these in mind)

| run | batch per step | steps/epoch | epochs | steps | frames seen | lr | warm-up |
|---|---|---|---|---|---|---|---|
| SMPL-X probe | 64 single frames (every 5th frame, per-epoch jitter) | 869 | 10 of 30 (stopped; best = epoch 8) | ~7.8k to best | ~0.5 M | 1e-4 | per-epoch (epoch 0 at 1 % lr) |
| temporal, pose token | 2 clips × 60 frames (1 clip/GPU × 2) | 1835 | 5 | 9175 | 1.1 M | 1e-4 | per-epoch (epoch 0 at 1 % lr) |
| temporal, contact tokens | 2 clips × 60 frames | 1835 | 5 | 9175 | 1.1 M | 1e-4 | per-epoch (epoch 0 at 1 % lr) |
| … 8 clips/step, lr 2e-4 | 8 clips × 60 frames (4 clips/GPU × 2) | 459 | 5 | 2295 | 1.1 M | 2e-4 | per-step, 500 steps |

All four trained with AdamW betas (0.9, 0.999), weight decay 0.01 on every parameter, one
global gradient-norm clip at 1.0 and no EMA — i.e. WITHOUT the optimizer hygiene bundle that
became the `base.yaml` default on 2026-09-03. A run's own `output/<run>/config.yaml` is the
authority on what it used.

### Contact threshold curve (micro P / R)

| run | thr 0.3 | thr 0.5 | thr 0.7 | thr 0.9 |
|---|---|---|---|---|
| temporal, pose token | 0.882 / 0.944 | 0.911 / 0.907 | 0.943 / 0.835 | 0.973 / 0.664 |
| temporal, contact tokens | 0.891 / 0.947 | 0.913 / 0.920 | 0.936 / 0.869 | 0.966 / 0.719 |
| … 8 clips/step, lr 2e-4 | 0.893 / 0.948 | 0.917 / 0.921 | 0.938 / 0.864 | 0.968 / 0.707 |

Heels (LA / RA, 3 % prior) are never predicted by any run (F1 = 0 at every threshold).

## 2026-09-03 additions round (`docs/additions_2026-09-03.md`)

Every arm = the b8_lr2 recipe (8 clips × 60 frames per step, lr 2e-4, 5 epochs = 2295 steps,
per-step warm-up 500) PLUS the optimizer hygiene bundle that is the `base.yaml` default since
2026-09-03 (betas (0.9, 0.95), `decay_1d false`, EMA 0.999, per-group clip) — the recorded
b8_lr2 baseline trained WITHOUT it (user call: no re-run; the deltas fold the bundle in, which
was worth −1.5 to −4 mm at equal epochs in the round-2 arm C). Same eval protocol as above.

| arm | config / run | mpjpe | pa | pve | accel | f1 | prec | rec | iou | P@R0.9 | LH | RH | LF | RF | LA | RA | arm-specific |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline b8_lr2 | `temporal_tokens_b8_lr2.yaml` / `…_20260902_203707` | 60.07 | 41.21 | 77.86 | 10.47 | 0.919 | 0.917 | 0.921 | 0.850 | 0.926 | 0.964 | 0.971 | 0.885 | 0.884 | 0 | 0 | – |
| H hands (30 finger joints in the SMPL-X head) | `hands.yaml` / `hands_20260903_134106` | 60.99 | 42.04 | 78.27 | 10.33 | 0.919 | 0.915 | 0.924 | 0.851 | 0.924 | 0.962 | 0.967 | 0.891 | 0.886 | 0.034 | 0.035 | hand_mpjpe 36.02, hand_pa_mpjpe 3.99 |
| K keypoints (sapiens 2D body keypoints → pose token), eval with sapiens | `keypoints.yaml` / `keypoints_20260903_140922` | **59.67** | **41.15** | **77.71** | 10.42 | **0.921** | **0.920** | 0.922 | **0.853** | **0.932** | 0.966 | 0.969 | 0.892 | 0.887 | 0.031 | 0 | train = test source |
| K, eval with the DEPLOYMENT source (frozen MHR keypoints, score 1) | scratchpad `massive/keypoints_eval_mhr.yaml`, same checkpoint | 62.14 | 42.22 | 80.38 | 10.43 | 0.921 | 0.920 | 0.922 | 0.854 | 0.932 | 0.966 | 0.969 | 0.893 | 0.887 | 0.036 | 0 | `output/logs/keypoints_*_eval_mhr.log` |
| K, keypoint input dropped on every frame | scratchpad `massive/eval_keypoints_none.py`, same checkpoint | 103.64 | 64.93 | 122.58 | 10.70 | 0.921 | 0.921 | 0.920 | 0.853 | 0.932 | 0.966 | 0.969 | 0.892 | 0.886 | 0.005 | 0 | `…_eval_none.log` |
| M masking (1-5-frame spans, ~15 %, learned mask token on all 7 slots) | `masking.yaml` / `masking_20260903_145800` | 61.34 | 42.04 | 79.50 | 10.27 | 0.920 | 0.912 | **0.928** | 0.852 | 0.926 | 0.965 | 0.970 | 0.888 | 0.890 | 0.008 | 0 | eval uncorrupted |
| C-A camera twist → pose token, body root twist head on it | `camera_posetoken.yaml` / `camera_posetoken_20260903_152304` | 60.69 | 41.79 | 79.90 | 10.34 | **0.922** | 0.917 | 0.927 | **0.855** | 0.929 | 0.965 | 0.969 | 0.892 | 0.892 | 0.052 | 0 | motion + global table below |

Per-epoch test trajectory (mpjpe / contact F1 at the end of epochs 0…4):

| arm | ep 0 | ep 1 | ep 2 | ep 3 | ep 4 |
|---|---|---|---|---|---|
| baseline b8_lr2 | 101.4 / 0.877 | 79.6 / 0.902 | 67.7 / 0.907 | 61.5 / 0.918 | 60.1 / 0.919 |
| H hands | 108.6 / 0.884 | 71.4 / 0.906 | 64.4 / 0.917 | 61.9 / 0.919 | 61.0 / 0.919 |
| K keypoints (sapiens at test) | 97.2 / 0.885 | 68.3 / 0.906 | 62.5 / 0.917 | 60.5 / 0.920 | 59.7 / 0.921 |
| M masking | 98.8 / 0.884 | 70.1 / 0.903 | 64.3 / 0.916 | 62.2 / 0.920 | 61.3 / 0.920 |
| C-A camera, pose token | 100.0 / 0.885 | 69.1 / 0.906 | 63.7 / 0.919 | 61.4 / 0.921 | 60.7 / 0.922 |

Contact threshold curve (micro P / R):

| arm | thr 0.3 | thr 0.5 | thr 0.7 | thr 0.9 |
|---|---|---|---|---|
| baseline b8_lr2 | 0.893 / 0.948 | 0.917 / 0.921 | 0.938 / 0.864 | 0.968 / 0.707 |
| H hands | 0.895 / 0.949 | 0.915 / 0.924 | 0.936 / 0.871 | 0.967 / 0.730 |
| K keypoints (sapiens) | 0.898 / 0.948 | 0.920 / 0.922 | 0.941 / 0.873 | 0.968 / 0.724 |
| K keypoints (MHR source) | 0.898 / 0.948 | 0.920 / 0.922 | 0.940 / 0.874 | 0.968 / 0.725 |
| M masking | 0.891 / 0.953 | 0.912 / 0.928 | 0.936 / 0.876 | 0.966 / 0.724 |
| C-A camera, pose token | 0.894 / 0.950 | 0.918 / 0.927 | 0.936 / 0.878 | 0.964 / 0.731 |

**H, does the finger head articulate?** (scratchpad `massive/check_hands_flat.py`, transcript
`output/logs/check_hands_flat.log`; 24 test scenes, 2,857 valid frames; the hand metrics
recomputed with three finger sources on the SAME predicted body / betas / camera):

| fingers | hand_mpjpe (wrist-translation aligned) | hand_pa_mpjpe (per-hand Procrustes) |
|---|---|---|
| predicted | 38.3 | 3.9 |
| flat (identity rotations) | 45.8 | 12.6 |
| GT finger rotations (ceiling) | 37.7 | 0.2 |

The head recovers ~70 % of the flat-to-GT Procrustes gap; the wrist-aligned error is at the
ceiling set by the body's wrist ORIENTATION (37.7 with perfect fingers), i.e. it measures the
body pose, not the fingers. Body metrics: +0.9 mm mpjpe / +0.8 pa / +0.4 pve vs the baseline
(inside the epoch-4→3 spread of 1.5 mm; the bundle's expected gain did not show), contact
unchanged. Note H's `best.pth` is chosen by mpjpe = epoch 4 (last).

**K, what the keypoint input buys and costs.** With the SAME source at train and test (sapiens,
23 body keypoints crop-normalised + score, zero-init MLP added to the pose token before the
temporal block) K is the best pose row of the round (−0.4 mm mpjpe, −0.15 pve vs the baseline,
+0.002 F1 / +0.006 P@R0.9 on contact) and converges faster (per-epoch 97 / 68 / 62.5 / 60.5 /
59.7 vs 101 / 80 / 68 / 61.5 / 60.1). But the trained path RELIES on the input: dropping it
(zero embedding on every frame — out of distribution, `drop_prob` was 0) costs +44 mm, and the
deployment source (the frozen model's own MHR keypoints at score 1, the only keypoints
available at inference) lands at 62.14, i.e. 2.1 mm WORSE than the baseline that has no
keypoint input at all. The contact head ignores the input under every source (F1 0.921 ±
0.0002, identical curves). => as trained, the sapiens-keypoint input is a train-time-only gain;
a deployable version needs the deployment source (or its noise) at train time — untested
(optional arms: `noise_std 0.02, drop_prob 0.1`; `source: mhr` training).

**M, span masking.** Pose +1.3 mm mpjpe / +0.8 pa / +1.6 pve vs the baseline (the worst pose
row of the round so far, though still inside 2 mm); contact F1 +0.001 with the operating point
shifted toward recall (R +0.007, P −0.005; the P/R curve and P@R0.9 are unchanged). Converges
like the others in epochs 0-2 (98.8 / 70.1 / 64.3 vs 101 / 80 / 68) and falls behind at
epochs 3-4. Whether the corruption changed the block's attention regime (baseline: near-uniform
clip pooling, same-frame mass 1-3 %) is measured by the temporal diagnostic on this run
(`output/logs/diag_temporal_masking.log`, `output/masking_20260903_145800/diag_temporal/`):

| run | same-frame mass (L0-3) | mean \|dt\| (s) | eff. frames | mass \|d\| > 20 steps | ‖γ_attn‖ / ‖γ_ffn‖ | same_frame Δ mpjpe / Δ F1 | bypass Δ mpjpe / Δ F1 |
|---|---|---|---|---|---|---|---|
| baseline b8_lr2 | 0.012 / 0.030 / 0.022 / 0.019 | 0.69-0.71 | 9.5-11.6 | 0.48-0.52 | 0.39-0.47 / 0.16-0.19 | +0.04 / −0.026 | +17.4 / −0.056 |
| M masking | 0.019 / 0.019 / 0.024 / 0.016 | 0.70-0.79 | 36-49 | 0.51-0.60 | 0.52-0.59 / 0.17-0.23 | +0.08 / −0.032 | +56.4 / −0.106 |

Masking pushed the block the OPPOSITE way from "use the neighbours": the attention is flatter
and wider (4× the effective frames, more mass beyond 20 steps), the gates are larger and the
block carries more of the per-frame transform (bypass now costs 56 mm instead of 17) — but the
temporal context still buys nothing for pose (same_frame Δ +0.08 mm) and only precision for
contact. Per-clip sensitivity full vs same_frame: hip-aligned joints move 9.6 mm (baseline 5.7)
without changing the metrics. => span masking is not the lever for a temporal pose gain.

**C-A / C-B, camera twist in, body root twist out, GVHMR roll-out eval.** Motion metrics are
Pearson r of the de-standardized prediction vs the kindyn SMPL-X root body twist (σ 0.12 s) over
the 12,641 twist-supported test rows (`vert` = world-vertical component, `r3d` = pooled xyz,
rmse / gt_rms in m/s and rad/s). Global metrics (`metric_global/*`, GVHMR kernels verbatim in
`utils/gvhmr_metrics.py`, 100-frame chunks, the clip's real fps): `rollout_*` = the predicted
per-frame body rolled out from the FIRST frame's predicted world root by integrating the
predicted twist (trapezoid se3 exp); `lifted_*` = every frame's prediction lifted with that
frame's GT camera (an oracle-camera reference, not a deployable trajectory); `gt_jitter` = the
GT's own jitter under the same kernel (7.2).

| arm | vel r3d / vert | ang vel r3d / vert | vel rmse (gt rms) | roll-out WA / W-MPJPE100 (mm) | roll-out RTE (%) | roll-out jitter | lifted WA / W | lifted RTE | lifted jitter |
|---|---|---|---|---|---|---|---|---|---|
| C-A pose token | 0.635 / 0.703 | 0.587 / 0.434 | 0.320 (0.421) | 150.1 / 262.4 | 8.36 | 38.7 | 80.5 / 133.9 | 4.39 | 97.1 |

C-A per epoch — vel r3d 0.26 / 0.40 / 0.58 / 0.63 / 0.64, ang vel r3d 0.19 / 0.25 / 0.45 /
0.56 / 0.59; roll-out WA / W / RTE 200 / 339 / 11.2 → 182 / 317 / 10.4 → 160 / 282 / 8.5 →
152 / 267 / 8.2 → 150 / 262 / 8.4; lifted WA / W / RTE 122 / 178 / 5.8 → 81 / 134 / 4.4 (the
lifted numbers only track the per-frame pose). Body pose +0.6 mm mpjpe / +0.6 pa / +2.0 pve vs
the baseline, contact +0.003 F1 / +0.003 P@R0.9 (the best contact row of the round, within
noise of K). Reference brackets (same checkpoint, same predicted body, different twist:
scratchpad `massive/rollout_bracket.py`, `output/logs/rollout_bracket_camera_posetoken.log`):
all 108 test clips:

| twist integrated (C-A body) | roll-out WA / W-MPJPE100 | RTE (%) | jitter |
|---|---|---|---|
| predicted (the C-A head) | 150.1 / 262.4 | 8.36 | 38.6 |
| GT twist (a perfect head; edge rows nearest-valid) | 51.1 / 82.0 | 0.75 | 38.6 |
| zero twist (root frozen at frame 0) | 208.1 / 366.0 | 14.7 | 38.5 |
| (per-frame lifted with the oracle camera, for reference) | 80.5 / 133.9 | 4.39 | 96.5 |

The predicted twist closes 37 % of the static→perfect gap on WA-MPJPE100 and 43 % on RTE.
Roll-out jitter is twist-independent (38.6 under all three) — it is the root-local joint
noise of the per-frame pose, not the trajectory; the lifted jitter (96.5) is the per-frame
root-placement wobble. A perfect twist beats the oracle-camera lifting by 30-50 mm, so on this
model the per-frame camera-frame root placement, not the integration, is the weaker link.

**S1 / S2, jerk prior.** `metric_smooth/*` = RMS jerk over the twist-supported test rows
(12,436): the world-lifted root body twist's linear (m/s³) and angular (rad/s³) jerk and the
root-local joints' jerk (m/s³), 5-point stencils at the eval stride, prediction vs the kindyn GT
under the same stencil. The baseline's numbers come from evaluating its checkpoint with the
metrics switched on (scratchpad `massive/b8lr2_eval_smooth.yaml`,
`output/logs/temporal_tokens_b8_lr2_20260902_203707_eval.log`; its pose metrics reproduce the
recorded row exactly).

| arm | pred RMS jerk root lin / ang / joints | GT RMS | mpjpe | pa | pve | accel |
|---|---|---|---|---|---|---|
| baseline b8_lr2 (no prior) | 402 / 680 / 225 | 38 / 76 / 72 | 60.07 | 41.21 | 77.86 | 10.47 |

## Retired round-2 arms (runs + configs trashed 2026-09-02, numbers from their evaluations)

All three = `temporal_tokens.yaml` at 8 clips/step (4 clips/GPU × 2 GPUs, 459 steps/epoch,
5 epochs, per-step warm-up 500), lr 1e-4 unless stated.

| arm | change | mpjpe | pa | pve | accel | f1 | prec | rec | P@R0.9 |
|---|---|---|---|---|---|---|---|---|---|
| A `temporal_tokens_b8` | batch only | 66.2 | 47.3 | 87.9 | 10.4 | 0.920 | 0.918 | 0.922 | 0.928 |
| B `temporal_tokens_b8_prec` | A + `neg_weight 2`, heel `pos_weight 4`, `transition_tolerance 2` | 66.3 | 47.3 | 88.3 | – | 0.905 | 0.924 | 0.887 | 0.919 |
| C `temporal_tokens_b8_hyg` | A + hygiene bundle (betas .95, `decay_1d false`, EMA .999, per-group clip) | 64.7 | 44.8 | 83.7 | 10.2 | 0.916 | 0.910 | 0.922 | 0.921 |
| D `temporal_tokens_b8_lr2` (kept, main table) | A + lr 2e-4 | 60.1 | 41.2 | 77.9 | 10.5 | 0.919 | 0.917 | 0.921 | 0.926 |

## Temporal-block diagnostic (`scripts/diag_temporal.py`, b8_lr2 run, 2026-09-03)

Attention statistics on the first 24 test scenes (2,857 valid frames, eval protocol: T ≤ 120,
±2.5 s window ≈ ±65 steps at 0.038 s/step); ablations on all 108 scenes. Files:
`output/temporal_tokens_b8_lr2_20260902_203707/diag_temporal/{summary.json, temporal_profile.png,
slot_mixing.png, same_frame_mass.png}`, transcript `output/logs/diag_temporal_b8_lr2.log`.

| layer | self mass | same-frame mass | cross-frame mass | mean \|dt\| (s) | eff. frames | same-frame per head min / med / max | ‖γ_attn‖ | ‖γ_ffn‖ |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.003 | 0.012 | 0.988 | 0.69 | 9.5 | 0.004 / 0.010 / 0.026 | 0.47 | 0.19 |
| 1 | 0.004 | 0.030 | 0.970 | 0.68 | 11.5 | 0.005 / 0.025 / 0.143 | 0.45 | 0.17 |
| 2 | 0.004 | 0.022 | 0.978 | 0.70 | 11.6 | 0.002 / 0.014 / 0.092 | 0.39 | 0.17 |
| 3 | 0.005 | 0.019 | 0.981 | 0.71 | 10.7 | 0.006 / 0.021 / 0.053 | 0.39 | 0.16 |

Mass by |offset| band (clip steps), layer 0 → 3: |d| = 0: 0.012 / 0.030 / 0.022 / 0.019;
|d| = 1: 0.025 / 0.053 / 0.039 / 0.040; 2–5: 0.17 / 0.19 / 0.16 / 0.14; 6–20: 0.54 / 0.44 / 0.48 /
0.51; > 20: 0.48 / 0.50 / 0.52 / 0.51. No head is collapsed to the diagonal (max same-frame mass
0.14); the profile is a broad skirt over the whole window (≈ 20× from dt = 0 to the window edge)
— clip-wide context pooling rather than local smoothing. Slot mixing: every query slot reads
mostly its own slot and the hand slots; the pose query keeps 34–56 % on the pose slot.

Output ablations (`full` = as trained; `same_frame` = window closed to the query's own frame, so
only within-frame cross-modal mixing survives; `bypass` = all gates zeroed = block removed):

| metric | full | same_frame | Δ | bypass | Δ |
|---|---|---|---|---|---|
| mpjpe | 60.07 | 60.03 | −0.04 | 77.49 | +17.42 |
| pa_mpjpe | 41.21 | 41.23 | +0.02 | 49.54 | +8.34 |
| pve | 77.86 | 77.58 | −0.28 | 92.92 | +15.06 |
| accel | 10.47 | 10.63 | +0.16 | 10.53 | +0.06 |
| contact f1 | 0.919 | 0.893 | −0.026 | 0.863 | −0.056 |
| precision | 0.917 | 0.849 | −0.068 | 0.797 | −0.120 |
| recall | 0.921 | 0.942 | +0.021 | 0.942 | +0.021 |
| P@R0.9 | 0.926 | 0.869 | −0.057 | 0.815 | −0.112 |

Per-frame sensitivity full vs same_frame (attention subset): contact-probability |Δ| mean 0.064,
p90 0.165; SMPL-X joints move 17.4 mm raw / 5.7 mm hip-aligned on average without changing the
metrics. Caveat: both ablation arms put the block in an attention regime it never trained in
(7 or 0 keys instead of ~875), so their contact losses mix "temporal information removed" with
out-of-distribution behaviour; the pose invariance is the robust finding.

## Static-camera jitter round (2026-09-04/05)

Static subset of the corpus (`configs/datasets/climbing_videos_static.yaml`: 113 train / 16
annotated test scenes after the 2026-09-04 focal-jump flips), 30 epochs, `eval_max_frames 120`.
`jitter` = GVHMR lifted-trajectory jitter (10 m/s³; GT 6.35 on this set), `dlogz` = RMS
frame-to-frame step of the pelvis log depth in %/frame (prediction / prediction − GT; GT 0.278),
`depth_err` = absolute pelvis depth error (mm). Investigation logs: `docs/jitter_2026-09-04.md`,
`docs/camera_ray_2026-09-04.md`, `docs/temporal_block_2026-09-04.md`, explainer
`docs/trajectory_jitter_explained.md`. Transcripts `output/logs/<run>*.log`.

| run (static subset, 16-scene test) | clips/step | jitter | mpjpe | pa | depth_err | dlogz pred / err | F1 | status |
|---|---|---|---|---|---|---|---|---|
| frozen SAM3D (SMPL-X refit) | – | 126 (MHR70 lift) | 57.9 | 43.8 | 97 | 0.596 / 0.597 | – | reference, `output/frozen_sam3d_smplx_static16.json` |
| `static_baseline` (CLIFF camera, no matching) | 8 | 118.0 | 60.6 | 44.5 | 125 | 0.583 / 0.586 | 0.876 | kept |
| `static_matching_anchor_raw` (CLIFF + matching 1 / 0.5 / 1 + pelvis anchor 3) | 8 | 91.4 (19-scene set: 83.6) | 63.8 | 47.0 | 132 | 0.433 / 0.459 | 0.848 | kept (motion-matching reference; re-scored 2026-09-05, `output/logs/static_matching_anchor_raw_*_eval.log`) |
| `static_ray` R1 (ray camera on the frozen prior, bbox + frozen-camera inputs, image kp2d, depth/bearing vel + acc; `zero_gate`) | 4 | 56.7 | 65.2 | – | 116 | 0.161 / 0.272 | 0.852 | kept |
| `tb_projzero` A (R1 recipe + `gate_init: zero_proj`) | 2 | 52.4 (min 50.5 at ep 13) | 72.3 | – | 100 | 0.158 / – | 0.874 | kept — best without explicit smoothing |
| SM-6 `static_sm_split_acc` (matching + anchor + learned convex OUTPUT smoother + acc matching) | 4 | 9.10 | 60.5 | 44.4 | 113 | 0.174 / 0.241 | 0.869 | rejected: explicit smoothing (code removed) |
| R3 `static_ray_smprior` (R1 + convex kernel on the frozen PRIOR) | 2 | 51.3 | 65.5 | – | 109 | 0.160 / 0.247 | 0.872 | rejected: explicit smoothing (code removed) |
| R2 `static_ray_noprior` (R1 with a constant prior, no frozen input) | 4 | = R1 through ep 8 (71.6) | – | – | 215 | 0.265 / 0.330 | – | ablation, stopped ep 13 |
| B `tb_loc` (A + learnable locality bias σ 0.1 s) | 2 | 50.8 | 72.7 | – | 103 | 0.176 / – | 0.877 | no gain over A; σ widened to 0.13-0.52 s, self bias −0.3…−1.9 (code removed) |
| C `tb_locwide` (σ init 1.0 s) | 2 | 52.1 | 72.6 | – | 103 | 0.158 / – | 0.895 | σ widened further (0.9-1.9 s) (code removed) |
| D `tb_window` (R1 block, `max_rel_sec 0.25`) | 2 | 60.0 | 68.9 | 47.4 | 115 | 0.251 / 0.292 | 0.841 | worse than R1; a hard window does not make the zero-gate block denoise |
| E `tb_stage2` (R1 heads frozen at lr 0, fresh zero_proj + locality block) | 2 | 57.0 | 65.5 | – | 114 | 0.177 / – | 0.862 | = R1: the block alone cannot denoise (warm-start / freeze code removed) |
| F / H `tb_mix*` (convex mixing path, first version) | 2 | killed ep 5 (75 / 66) | – | – | – | – | – | gate never opened; the path mixed slots (bug found by a forced sweep), fixed same-slot version untrained (code removed) |
| GVHMR token arm `static_anchor_raw_gvhmr` (2D keypoints as the main token) | 8 | 76.0 at ep 19 | 157.6 | 116.6 | – | – | – | far too slow to train on 113 scenes; stopped ep 21 (code removed) |

Reading: the ray camera head removed depth as a jitter source (depth-only share 110 → 11-15) and
the zero-projection block parametrisation is worth ~5 points on top; every arm that tried to make
the residual softmax block a local average converged to dilution (`x + c·avg`, self bias negative,
kernels wide) with a floor of ~50. The remaining jitter is per-frame rotation / articulation noise
of the pose readout, created in the backbone feature maps (averaging the frozen pose token over
~0.1 s gives 14 with no training — `temporal_block_2026-09-04.md` §2). Explicit smoothing reaches
9-10 but is excluded by the user.
