# The final model (2026-09-08): what it is, why each piece is there

*Concept page. Config `configs/final.yaml`; code `model/refiner.py`, `model/heads.py`,
`model/rope.py`, `model/loss/`. Measurements behind every choice: `round5_2026-09-07.md`,
`refiner.md`. The earlier, more ambitious proposal is `history/architecture_proposal_2026-09-07.md`; this page replaces it.*

## In one paragraph

A frozen per-frame body estimator gives one SMPL-X body per video frame. We lift those bodies
into the world with the known camera, smooth them with a fixed Gaussian, and that smoothed
trajectory **is** the pose output — nothing learned touches it, because nothing learned ever
beat the Gaussian. A small temporal transformer then reads the trajectory (plus a few
body-frame channels and the frozen image token) and predicts, per frame, which of six body
parts are in contact with the wall, the contact forces on them, and the body's velocities and
accelerations. The transformer is the only trained part: 10.9 M parameters out of 1.3 billion.

```
video frame ──► SAM 3D Body (frozen) ──► pose token ──► SMPL-X + camera head (frozen, stage 1)
                                                                │
                                                     per-frame body in the camera
                                                                │  cam_from_world
                                                                ▼
                                          world lift ─► Gaussian (root 0.10 s, rotations 0.08 s)
                                                                │
                                              ┌─────────────────┴───────────────┐
                                              ▼                                 ▼
                                    THE BODY (pose output)            observation token (538-d)
                                                                                │
                                                                RoPE transformer, 3 × 0.5 s window
                                                                                │
                                                         ┌──────────────────────┼─────────────────┐
                                                         ▼                      ▼                 ▼
                                                  contact (6 logits)   forces (6 × 3, bw)   motion (vel / acc)
```

## 1. Inputs

Everything the model reads per frame:

| input | where it comes from | used by |
|---|---|---|
| frozen pose token (1024-d) | SAM 3D Body's decoder, cached once per person-frame (`features/pose_token`) | stage-1 head, token channel |
| camera intrinsics and `cam_from_world` (per frame) | corpus geometry | world lift, camera context |
| person box | corpus tracks | stage-1 camera lift, camera context |
| gravity direction in the world | corpus SMPL-X fit (per scene) | token channel |

The backbone and decoder never run during training: the cached token is all the model reads
from the image. One training epoch on 864 scenes takes about two minutes on one GPU.

## 2. Stage 1 — the per-frame body (frozen)

Two small MLPs on the pose token (run `output/stage1_20260905_180319`, trained once, 2026-09-05):
an SMPL-X head (root orientation, 21 body joints and 30 finger joints as 6D rotations, 10 betas,
all as residuals on a fixed mean pose) and a CLIFF camera head (crop-space scale and offset,
lifted to metres with the box and the true focal length). Output: the body in the camera frame.
On the test protocol it scores 57.6 mm MPJPE by itself and jitters heavily frame to frame
(lifted-trajectory jitter about 110 against a ground-truth 7, in 10 m/s³ units), almost all of
it depth and rotation noise.

We keep it frozen. Making it trainable behind a stop-gradient (round 5, arm J1) improved its
own body by 0.3 mm, overfit from epoch 7, and made the downstream body 0.85 mm worse.

## 3. The body — world lift and a fixed Gaussian

1. Lift the pelvis and root rotation into the world with `cam_from_world`.
2. Gaussian-smooth the world pelvis position (σ 0.10 s) and, on the rotation manifold, the
   root rotation and the 21 parent-local joint rotations (σ 0.08 s; a matrix mean projected
   back onto SO(3)). Betas are averaged over the clip.
3. Forward kinematics in the world; map back into every camera for the 2D readouts.

Two rules from earlier rounds: smooth in the **world**, never in the camera (camera coordinates
carry the camera's own motion, and smoothing them breaks the lift), and smooth **both** the root
and the rotations (the per-frame fit's root and orientation errors cancel each other; smoothing
one alone raises jitter).

Why fixed: on the 108 test clips, stage 1 plus this Gaussian scores 55.4–55.5 mm MPJPE and
jitter 3, untrained. Every learned alternative measured worse — a learned global width stayed
where it started, a per-frame adaptive kernel learned a Nyquist-passing comb (jitter 22) and,
once its loss could see that, merely matched the Gaussian, a pose-delta head trained with motion
losses shrank the motion and cost 0.4–0.8 mm, and a transformer with no kernel at all did not
smooth (57.8 mm, jitter 71). Section 7 has the numbers. The Gaussian is the ceiling we found, so
the model does not carry a pose head.

## 4. The observation token (538 numbers per frame)

Everything the transformer sees is expressed relative to the body or the camera — never the
world frame. The unit test `tests/test_refiner.py::test_world_frame_independence` enforces it:
rotating and translating the whole world leaves every output unchanged.

| channel group | dims | what it is |
|---|---|---|
| joint positions in the root frame | 63 | FK of the smoothed body, 21 joints relative to the pelvis, rotated into the root frame |
| root velocity (body frame) | 3 | finite difference of the world pelvis, rotated into the body |
| root angular velocity (body frame) | 3 | from adjacent root rotations |
| frame spacing | 1 | local Δt × 25 (the corpus mixes 24–60 fps) |
| betas | 10 | clip mean |
| camera context | 7 | direction to the camera in the body frame, log depth, box bearing and angular size — where the per-frame depth error points and how large the crop was |
| parent-local rotations | 126 | the 21 smoothed joint rotations as 6D |
| gravity in the body frame | 3 | the scene's down vector rotated into the root — pitch and roll relative to gravity |
| raw minus local mean | 66 | the pelvis and the root-frame joints minus their ±2-frame mean — what the smoothing removed |
| projected pose token | 256 | a linear map of the frozen 1024-d decoder token |

The 282 geometry numbers and the 256 token numbers are LayerNormed separately, concatenated,
and projected to the transformer width. The last three groups were added in round 5: gravity
and local rotations are what a force regressor needs (a static load points along gravity),
and together they cut force MAE from 0.192 to 0.183 body-weights and the direction error from
21.5° to 19.9° with contact and pose unchanged. The pose token is worth +0.005 F1, all
precision, on two seeds of this recipe (`round6_2026-09-11.md`; round 5 had measured +0.001 on
a weaker recipe) — at the reading rule's noise edge, and free.

## 5. The temporal block

`CrossModalRopeModule`, width 512, 3 pre-LayerNorm residual blocks, 8 heads, FFN ×4, dropout
0.1; 9.5 M parameters. Rotary position encoding on real elapsed seconds (×25), so attention
depends only on time differences and a model trained on 60-frame clips runs single-pass on a
whole scene. Each layer attends only inside ±0.5 s, so the receptive field is ±1.5 s.
Attention and FFN output projections start at zero, so the block is the identity at
initialisation. Bidirectional: this is an offline video model.

The window matters. A one-frame window loses 0.018 contact F1 and drops transition F1 from
0.39 to 0.24 on three seeds; a 0.15 s window (the "small" block of the earlier proposal) loses
0.012 F1 and 0.165 transition F1. Contact transitions need more than 0.3 s of context.

## 6. Heads and losses

Each head is `Linear(512, 512) → GELU → Linear(512, out)` with the last layer zero-initialised,
on the LayerNormed transformer output.

| head | output | loss (`model/loss/`) |
|---|---|---|
| contact | 6 logits: left hand, right hand, left toe, right toe, left heel, right heel | BCE against the corpus labels, confidence-weighted, weight 5, all class weights 1 |
| force | 6 × 3 in the body frame, body-weight units | Huber (δ 0.5 bw) on frames with a force label, L1 towards zero off contact, weighted by the label confidence |
| motion | body-frame velocity and acceleration of 22 joints (132) + root angular velocity and acceleration (6) | Huber on each, divided by the ground-truth RMS scale; targets = one 0.12 s Gaussian on the GT trajectory, then the same finite-difference stencils the prediction side uses |

The smoothed body is also scored against the corpus SMPL-X fit (`smplx_supervision`) so every
run reports MPJPE, PA-MPJPE, jitter and the rest; with no pose head those terms carry no
gradient.

Two loss-side lessons are baked in. The motion targets use **aligned stencils**: the earlier
recipe smoothed the GT derivatives twice and compared them with central differences of the
prediction, which are exactly blind to a period-two (Nyquist) component; that blindness is what
let the adaptive kernel grow its comb. And there is no coupling loss: contact-frame wrenches, a
stillness loss on contacting limbs and the force–contact hinge all measured within seed noise or
worse. The one physics term with a measurable effect, the RNEA root-wrench residual, is a trade
(about 1° of force direction against twice the off-contact force) and is off in
`final.yaml`, on in `final_rnea.yaml`; the residual's floor on the ground truth itself is
0.19 body-weights with six lumped forces, so it can never be driven to zero.

## 7. What was tried and dropped

All on the same 108-clip test protocol; seed spread of the recipe is 0.002 F1, 0.013 transition
F1, 0.02 mm MPJPE, 0.0005 bw force MAE, 0.4° angle (three seeds).

2026-09-08: the code of the dropped mechanisms (the output convex smoother in its fixed / global /
adaptive kinds, the stop-gradient for a trainable per-frame head, the rigid scene camera, the
contact-wrench RNEA body) was removed with the arms that used it; both are in
`/data3/rikhat.akizhanov/trash/cleanup_20260908/` (`pre_prune_tree/`, `output_2_arms_pruned/`).

| idea | measured | verdict |
|---|---|---|
| learned global kernel width | stays at 0.078 → 0.081 s; = fixed Gaussian | no gain |
| per-frame adaptive kernel | jitter 22 (comb kernel); with aligned losses = fixed Gaussian | no gain |
| pose-delta head + derivative losses | shrinks motion (speed ratio 0.92 → 0.83), +0.4 to +0.8 mm | removed |
| no kernel, block learns to smooth | 57.8 mm, jitter 71 | fails |
| trainable per-frame head (stop-gradient) | overfits, refined body −0.85 mm | removed |
| smaller block (128-d, 2 layers, 0.15 s) | −0.012 F1, −0.165 transition F1 | keep the block |
| RNEA root-wrench residual | about −1° force direction, +0.000 MAE, 2× off-contact force (round 4 B vs A, and final vs F_tok) | a trade: off by default, `final_rnea.yaml` keeps it |
| contact-frame wrenches in the residual | within noise | removed |
| force branch → contacts | +0.0015 F1, +0.023 transition F1 (1.5–2× spread, one seed) | unresolved, kept (it is the force output anyway) |
| pose token as input | +0.005 F1 (precision) on two seeds, round 6 | kept, free |
| decoder contact tokens (image features at the joints) | +0.004 F1 over the cached path, one seed | not worth the 20× slower path |
| heel class weight ×5 | fires at the base rate, 3/4 wrong | weight 1 |
| frozen "static" extrinsics | worse on every metric, +12 mm pelvis error at test | removed |
| smoothing in the camera frame | lift no longer cancels camera motion; jitter 30 | world frame only |

## 8. Training recipe

* Data: all 864 train scenes, 60-frame clips at 25 fps (stride = round(fps / 25)), stateless
  per-epoch window jitter, clips dealt round-robin over source videos so a batch spans many
  videos. Evaluation: one clip per (scene, person) on the 108 annotated test scenes, capped at
  120 frames, plus whole-scene dumps for the bootstrap intervals.
* Batch: 8 clips × 60 frames per micro-batch, 8 micro-batches per optimizer step = 64 clips.
  15 epochs = 860 steps. AdamW lr 3e-4 (betas 0.9 / 0.95, weight decay 0.01, none on 1-d
  parameters), 100 warm-up steps then cosine to 1e-6, gradient clip 1.0 per module, EMA 0.999.
* One GPU, about 35 minutes. Two seeds (42, 1) are trained; runs `output_2/final_*`.

## 9. What the model does with time (diagnostics on the recipe)

`scripts/diag_invariance.py` re-scores a checkpoint under input perturbations:

* Reversing the clip leaves the body untouched and costs 0.02 contact F1, all of it precision,
  uniformly over the six parts: the contact head reads the arrow of time (onsets are detected
  better than offsets, so the labels' own timing is asymmetric).
* Shuffling the content inside 2–4-frame blocks changes nothing; shuffling the whole clip
  destroys contact (0.84 F1). The head is order-blind locally and temporal globally.
* Halving the frame rate is harmless (≤ 0.003 F1, ≤ 0.15 mm).
* Removing attention at test time costs 0.03 F1 and halves the acceleration correlation.

## 10. Results

Capped protocol (108 test clips, 120 frames, `last.pth`), two seeds of the final model against
the round-4 recipe with token channels (F_tok, which still carried a pose head, RNEA, the
stillness loss and the ×5 heel weight) and the untrained Gaussian body:

| run | F1 | P | R | P@R90 | hands F1 (L / R) | toes F1 (L / R) | heels F1 | MPJPE | PA | jitter | force MAE | angle | off-contact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final s42 | 0.927 | 0.924 | 0.931 | 0.938 | 0.969 / 0.976 | 0.894 / 0.903 | 0 / 0 | 55.51 | 38.12 | 2.75 | 0.187 | 21.4 | 0.022 |
| final s1 | 0.926 | 0.922 | 0.931 | 0.937 | 0.967 / 0.975 | 0.892 / 0.906 | 0 / 0 | 55.51 | 38.12 | 2.75 | 0.188 | 21.4 | 0.021 |
| final + RNEA residual (`final_rnea.yaml`, s42) | 0.925 | 0.923 | 0.926 | 0.934 | – | – | 0 / 0 | 55.51 | 38.12 | 2.75 | 0.182 | 20.5 | 0.045 |
| F_tok | 0.912 | 0.896 | 0.928 | 0.917 | 0.964 / 0.975 | 0.874 / 0.900 | 0.09 / 0.12 | 56.20 | 38.83 | 3.09 | 0.183 | 19.9 | 0.048 |
| stage 1 + Gaussian, untrained | – | – | – | – | – | – | – | 55.49 | 38.10 | 2.98 | – | – | – |

Reading, against the seed spread of section 7:

* **Pose**: the Gaussian body, 55.51 mm / jitter 2.75, 0.7 mm better than every trained pose
  head of the earlier rounds, identical on both seeds by construction.
* **Contact**: +0.015 micro F1 and +0.028 precision over F_tok. Most of it is the heels: with
  the class weight at 1 the head never fires on heels (3.5 % positives, unresolvable from the
  video; the ×5 weight produced ~1 000 false positives for ~100 true ones). The rest is real
  but small: toes +0.02 / +0.004, hands +0.005 / +0.001, precision at 90 % recall 0.917 → 0.938.
* **Forces**: MAE 0.187 vs 0.183 and direction 21.4° vs 19.9° — the final is WORSE than F_tok
  here, by 8× and 4× the seed spread. Round 4 measured the same thing: the RNEA residual is
  worth about 1° of direction at the price of twice the off-contact force (0.022 vs 0.048 bw
  here). Section 7's "within noise" line for RNEA is therefore wrong for the angle; a variant
  with the residual back on, as a regulariser of the forces on the fixed body
  (`configs/final_rnea.yaml`, `force_consistency` no longer requires a pose head), is in the
  table. It behaves as predicted: direction 21.4° → 20.5°, MAE 0.187 → 0.182, off-contact
  force 0.022 → 0.045 bw, contact F1 0.927 → 0.925. Which side of that trade to take is a
  product decision (clean off-contact zeros vs 1° of direction); the page's default is the
  clean one.
* **Motion head**: velocity correlation 0.81, acceleration 0.60 (F_tok 0.80 / 0.67; the
  aligned targets are a different, tighter estimator, so the two acceleration numbers are not
  the same quantity).

Paired, video-clustered intervals (`output_2/audits/r5_seeds/final_{capped,whole}.txt`),
final s42 minus F_tok, capped protocol unless noted:

| metric | difference | interval |
|---|---|---|
| contact F1 | +0.015 | [+0.009, +0.024] |
| precision | +0.027 | [+0.019, +0.040] |
| left / right hand F1 | +0.006 / +0.001 | [+0.002, +0.011] / n.s. |
| left / right toe F1 | +0.019 / +0.003 | [+0.007, +0.033] / n.s. |
| transition F1 (capped / whole) | +0.023 / +0.036 | [−0.011, +0.054] / [+0.014, +0.060] |
| force MAE | +0.0045 (worse) | [−0.000, +0.009] |
| force direction | +1.5° (worse) | [+0.9, +2.4] |
| off-contact force | −0.027 bw | [−0.031, −0.023] |
| MPJPE | −0.70 mm | [−0.94, −0.50] |

The two final seeds differ by ≤ 0.001 F1, ≤ 0.005 transition F1, 0.001 bw and 0.1° on every
row.

Force size correlation and load-share error on the corpus test set (`scripts/force_corr_share.py`,
whole scenes, four board limbs, all rows): final corr 0.70 / share 2.1 pp, final_rnea 0.68 / 1.2 pp,
F_tok 0.67 / 1.3 pp, round-4 A 0.67 / 1.6 pp — the final model correlates best, the RNEA variant
allocates best.

**Peter's instrumented wall** (`BetterVideoReconstruction/peter/out_climb_wall_2_single`, 7 trials,
2 535 frames; `scripts/predict_reconstruction.py` + BVR `compare_climb_wall_2.py`; the reconstruction
loader now reads `gravity_world` from each tree's `kindyn_1.npz` for the gravity channel):

| method | force MAE (N) | median angle | size corr | share error (pp) | contact P / R / F1 |
|---|---|---|---|---|---|
| optimization (kindyn solve, the pipeline's own) | 77.7 | 10.5° | 0.67 | 1.3 | 0.947 / 0.968 / 0.957 |
| final | 103.7 | 10.0° | 0.55 | 8.3 | 0.926 / 0.984 / 0.954 |
| final + RNEA | 85.7 | 10.9° | 0.60 | 6.4 | 0.928 / 0.973 / 0.950 |
| previous learned row (round-4 A, with RNEA) | 93.2 | 12.8° | 0.45 | 8.4 | 0.920 / 0.991 / 0.954 |

Out of domain the residual matters more than on the corpus: without it the final model
under-predicts the wall's total load by about 30 % (per-trial total size ratio 0.66–0.72 vs
0.82–0.90 before), which is the MAE gap; with it the total comes back (MAE 85.7 N, the best
learned row so far) at the cost of 0.9° of direction. For deployment on measured rigs
`final_rnea.yaml` is the better choice and is what the wall tree now carries (2026-09-08: per-trial total size ratio 0.86–0.92, total-vector median angle 3–5°); on the corpus metrics the two are a wash. Against the three F seeds (no token channels) the picture is the same, with force MAE then
0.005 better rather than worse and the direction equal.

## 11. How to run

```bash
PY=/data3/rikhat.akizhanov/miniconda3/envs/sam3d/bin/python
CUDA_VISIBLE_DEVICES=0 $PY scripts/train.py --config configs/final.yaml
$PY scripts/evaluate.py --config configs/final.yaml --checkpoint output_2/final_<stamp>/last.pth
$PY scripts/predict_test.py --config configs/final.yaml --checkpoint output_2/final_<stamp>/last.pth
$PY scripts/predict_reconstruction.py --config configs/final.yaml --checkpoint ... --out-root ... --videos ...
```

The refiner needs the stage-1 run directory (`model.smplx.checkpoint`) to exist: the frozen
head is re-read from it, not stored in the run's checkpoints.
