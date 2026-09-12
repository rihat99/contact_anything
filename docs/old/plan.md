# Plan: one jointly trained model, and making the temporal / physics claims testable (2026-09-07)

*Planning page. Follows the 2026-09-07 adversarial review
(`~/.claude/handoffs/adversarial_review_20260907.md`) and its independent second review by
codex / gpt-6-astra (`~/.claude/handoffs/adversarial_review2_20260907.codex.out`), which
corrected the first one on several points that changed this plan (listed in the last section).
The two-stage split is retired: everything trains together. The proposed model with every
block and loss is specified in `history/architecture_proposal_2026-09-07.md`; this page says what to run, in which order,
and what would falsify each claim.*

## What the reviews established

* **Temporal context already helps contacts.** Arm G (one-frame window) finished at F1 0.901
  vs 0.917 for the full model; a paired bootstrap by source video over the prediction dumps
  gives +0.016 F1 with a 95 % interval [+0.011, +0.023]. One seed, but not noise. The decoder
  contact tokens add +0.005 [+0.0002, +0.011]; forces and RNEA add nothing measurable.
* **Pose accuracy is not a target.** The best train-fitted linear filter gains 0.48 mm of 57;
  that is a linear-baseline result on 14 of 84 training videos, not an information bound, but
  nothing in the current data suggests a temporal pose gain worth chasing.
* **The derivative targets are internally inconsistent.** `motion.py` smooths the velocity
  target once and the acceleration target twice (≈ 0.17 s effective) while the pose path is
  differentiated raw; the position targets are raw GT. One trajectory cannot satisfy all three.
* **The learner's RNEA is not the producer's RNEA.** Learner: composed central stencil
  (`(q[t+2] − 2q[t] + q[t−2]) / 4dt²`), pure forces at the joint origins, uniform-density body,
  its own `m·g`. Producer (BetterVideoReconstruction, GT regenerated 2026-09-06 16–17 h): 3-point
  stencil at spacing `round(fps/30)`, contact wrenches `[f, r × f]` at the 12 contact frames,
  **Dempster segment densities** (BVR commit `c3a0627`; mean 1021 kg/m³, mass +2 %, `total_mass`
  stored in `kindyn_1.npz`), strength-weighted joint-torque effort terms (commit `845ef17`) that
  resolve the hands-vs-feet split. The 0.209 bw "GT floor" is a convention mismatch.
* **Contacts are not loads.** 17.1 % of the manual positive contacts (6 784 rows) carry under
  0.05 bw; a loss that makes contact imply load is wrong by construction.
* **Evaluation moves the numbers more than the models do**: capped vs whole-scene stage-1 MPJPE
  differ by 2.6 mm; the referenced `accel` is computed after hips alignment and cannot see root
  acceleration; no seed or video-cluster uncertainty has been reported so far.

## Claims to make hold, and what "hold" means

All with paired bootstrap intervals clustered by source video over the test clips, 3 seeds,
checkpoint rule fixed in advance (last epoch, EMA weights). A claim that does not beat its
control with an interval excluding zero is reported as not holding.

| # | claim | passes when |
|---|---|---|
| C1 | the model smooths the pose by itself, no Gaussian anywhere | on identical raw per-frame input, the learned smoother lies on or beyond the Pareto front of the fixed Gaussian family (referenced position error, referenced accel error, world-root accel error vs retained motion amplitude, swept over σ); the content-dependent kernel beats the learned global kernel; no lag (kernel centre of mass at 0 ± 1 frame) |
| C2 | the force branch improves contacts | with force supervision as the ONLY added factor (no coupling, no RNEA), contact transition metrics (onset / offset timing, boundary F1 ±2 frames) and hand precision at R = 0.90 improve; occupancy and loaded-support contact are scored separately |
| C3 | image features help contacts and forces beyond the predicted trajectory | trained controls with identical trajectory generation, parameter budget and dropout policy: trajectory-only → + pose token → + anchor features improve the C2 metrics and force MAE / angle / corrIC; the increment survives window > 1 frame |
| C4 | the RNEA loss is useful | after the learner matches the producer's conventions: (a) with force labels on nested video subsets, RNEA on all scenes lowers test force MAE / angle vs supervised-only AND vs two cheap controls (static root balance; a direct constrained force solve on the predicted pose + contacts); (b) off-contact force stays ≤ 0.02 bw; (c) world-root / CoM referenced accel does not worsen |
| C5 | the model uses temporal context (dynamics, not a per-frame map) | (a) a matched-capacity temporal model beats its matched-capacity per-frame twin (same preprocessing, no clip-level features); (b) a CAUSAL transition-forecasting head (release / grab within 0.4 s from the past only) beats persistence, a contact-duration hazard baseline and a pose + velocity per-frame baseline; (c) invariance diagnostics behave as predicted (below), reported as diagnostics, not pass / fail |

Every table carries the trivial baselines: per-frame output + the best fixed Gaussian, speed
threshold contact (micro F1 0.874), equal-split static force (0.24 bw / 30°), ridge on the
per-frame trajectory with predicted levers (0.192 bw / 20.0°).

## Phase −1 — audits before anything is built (2 days, no new training)

The second review's top four actions. Each is cheap and can change the rest of the plan.

1. **Target consistency.** Derive position, velocity and acceleration targets from ONE GT
   trajectory (raw, or one Gaussian applied once to the positions, σ swept 0 / 0.05 / 0.08 s),
   with explicitly aligned stencils: velocity = adjacent increment at the interval midpoint,
   acceleration = centred second difference at the frame. Measure, on the round-4 arm A
   checkpoint, the gradient conflict between the position and derivative terms on the pose
   output (cosine of the two gradients, per joint) before and after. Falsified if the conflict
   and the velocity attenuation (0.49 of GT in the 2026-09 rounds) do not move.
2. **Reproduce the producer's RNEA on the stored GT.** Same body (`density="Dempster"`, mass =
   `total_mass`), same stencil and spacing, wrenches with levers at the 12 contact frames (the
   loader currently drops groups carrying ~4 % of the force), same support rows. The residual
   on GT motion + GT forces must fall well below 0.209 bw; whatever remains is the producer's
   own residual (its stored `base_wrench` averages 0.15 bw) and is the real floor. Falsified if
   the residual changes by < 10 % after exact matching.
3. **Token audit.** Add the body-frame gravity direction and the parent-local 6D rotations to
   the refiner token of the round-4 recipe (nothing else changed) and rerun arm A once. The
   ridge check predicts 0.005–0.015 bw / 1–3° from gravity alone. Falsified if a matched
   ablation excludes gains above 0.003 bw / 0.5°.
4. **Freeze the evaluation protocol** (`scripts/evaluate.py`): whole valid runs tiled with
   overlapping context and each frame scored once; consistent beta aggregation; world-root and
   CoM referenced accelerations next to the hips-aligned `accel`; absolute world-root error and
   fixed-scale alignment next to the similarity-aligned globals; static and moving cameras
   separately; per-clip dumps + `scripts/paired_ci.py` (bootstrap by source video); the
   checkpoint rule fixed. Re-score the seven round-4 arms under it and rerun A and G with two
   more seeds each: the +0.016 F1 temporal effect must survive seeds and the new context.

## The joint model

One `ContactAnything` with no learned decoder token, trained end to end on the cached path
(`history/architecture_proposal_2026-09-07.md` has every block, channel and loss):

```
cached pose token ─► SMPL-X + camera head (trainable) ─► per-frame body ──┐
cached anchor image features (6 anchors, pooled DINOv3 samples) ──────────┤
                                                                          ▼
      world lift ─► observation token ─► temporal block ─► convex-kernel smoother
      ─► heads (pose delta, contact, loaded support, force, motion, causal transitions)
      ─► refined body in every camera
```

* `model.smplx.frozen: false`. From scratch is the default; a **warm start** from the old
  stage 1 is a legitimate arm (it is not contaminated, the videos do not overlap), and it
  removes the non-stationary input distribution the refiner sees while the head trains.
* **Gradient partition** (`model.refiner.detach_input`, default **true**) with a COMPLETE
  boundary: the refiner reads the per-frame body, the residual base pose, the betas, the
  camera frames and every geometry-derived channel through one stop-gradient. `false` is the
  fully joint arm J2, the only arm that tests "the auxiliary tasks improve the per-frame body".
* **Shrinkage monitor** (replaces the global kill criterion): predicted / GT speed ratio and
  phase per joint, per frequency band (< 1, 1–3, > 3 Hz) and per contact state, logged every
  epoch; a run is flagged, not stopped, when any joint band falls below 0.85.
* **Both outputs supervised and evaluated.** Per-frame losses on the per-frame body and on the
  refined body (the refined output gets its own camera readout so the `cam` / ray terms apply
  to both); derivative, RNEA and motion terms on the refined body only; metrics `_frame` and
  refined for the same weights.
* **Curriculum in optimizer updates**, not epochs: temporal / physics terms off for the first
  300 updates, each ramped separately over the next 300 (heads stay attached with zero weight
  so DDP sees every parameter).
* **Leak monitoring**: train / test gap of `smplx.kp3d_frame` per epoch; contained by weight
  decay and the small heads; refiner-side dropout does not regularise the per-frame head.

## Phase 0 — infrastructure

1. **Caches.** Extend `precompute_pose_tokens.py` with the frozen anchor features, POOLED
   before storage: per anchor the mean and the centre sample of the `grid_size 5` DINOv3 patch
   (6 × 2 × 1280 bf16 = 30 KB per frame, 15× the pose token; the raw 5×5 grid would be 384 KB).
   Benchmark the loader before training.
2. **Joint-model plumbing.** `smplx.frozen: false` under the refiner, the complete
   `detach_input` boundary, the `_frame` metrics, the update-based curriculum
   (`optim.temporal_start_step`, `temporal_ramp_steps`), the shrinkage monitor, the gap scalar,
   the refined-body camera readout.
3. **Metrics** (from Phase −1 item 4) plus contact transition metrics, occupancy vs
   loaded-support contact, `corrIC`. Drop `corr` / `share pp` from the headline.
4. **Stencils and targets** (from Phase −1 item 1) in every loss: `time_derivative`,
   `angular_velocity`, `pose_derivatives`, `contact_consistency`, `force_consistency`. All
   position / velocity / acceleration targets from one trajectory. `label_smooth_sec` stays for
   the motion head only.
5. **RNEA producer parity** (from Phase −1 item 2): `density: Dempster`, mass from
   `total_mass`, 3-point stencil at `round(fps/30)`, lever wrenches at the contact frames (the
   loader keeps the frame positions), the dropped groups restored or their share documented,
   hard-gate option, `detach_pose` fixed to detach before the force rotation.
6. **Token hygiene.** Fixed per-channel standardisation from train-only statistics of the
   actual input distribution (recomputed once after the per-frame warm-up); gravity in the body
   frame; parent-local 6D rotations of the 21 joints and the root rotation relative to the
   gravity frame (never a world rotation); per-frame betas with a causal running mean instead
   of the clip mean (the clip mean is a sequence-wide feature that leaks the future).
7. **Convex-kernel smoother block** (`history/architecture_proposal_2026-09-07.md`): fixed / global / adaptive modes, support
   in seconds, initialised from the fixed Gaussian, centre-of-mass and width logged.
8. **Augmentation hooks** (`aug.*`, below).
9. **Fixed camera** (`data.fixed_camera`, below).

## Augmentations

* **Image-feature dropout** (`aug.token_dropout`): per-frame probability 0.3 AND whole-clip
  probability 0.1, so the feature-less regime the "dropped" evaluation uses is actually seen in
  training (independent per-frame dropout at 0.4 never drops a 60-frame clip). Dropped frames
  get a learned "missing" embedding at the refiner input. The per-frame head always sees its
  token. This is regularisation and a within-model diagnostic; the C3 claim rests on the
  trained controls, not on the dropped evaluation.
* **Frame masking** (`aug.frame_mask` 0.2, spans of 2–6 frames): the masked frame's OBSERVATION
  is removed from every input path — the transformer token, the residual base pose, the
  smoother's neighbour set and the derived-velocity channels of its neighbours — and replaced by
  the mask embedding; its TARGET stays valid. Observation validity and target validity are two
  masks. It teaches interpolation; it supports C5 only through the causal forecasting head.
* **Realistic noise synthesis** (`aug.synth_noise`): kept as a low-priority option. Its AR(1)
  component is discrete OU noise; the only part that adds something is the per-clip bias
  sampling from out-of-sample residuals. Synthetic clips are identifiable by their missing
  token, so they must be mixed with real token-dropped clips. Run only if S-adaptive trails
  S-global while its train loss keeps falling.
* Not used: mirroring (later), time reversal.

## Model size and schedule

Trainable parameters: SMPL-X + camera head ≈ 2.4 M (two 1024-wide FFNs, with hands), refiner
`dim 128, num_layers 2, num_heads 4, mlp_ratio 2, dropout 0` ≈ 0.35 M, projections 1024→64 and
6 × 2 × (1280→16). `window 0.15 s` per layer. Weight decay 0.05, lr 1e-3, 200 warm-up updates,
**2 000 updates** on all scenes (≈ 7.7 M frame presentations, 2.3× round 4) with EMA as is.
These are fixed from S0 onward; no arm changes size, schedule or initialisation together with
its factor. 64 clips × 60 frames per update. 3 seeds per arm, ~1 h per run on the cached path.

## Phase 1 — smoothing without the Gaussian (C1)

Runs on the fixed-camera track first (all arms, 3 seeds), then the winners on all scenes.
Nested arms, identical support (±0.24 s), inputs, residual capacity and targets:

* **S-fixed**: the smoother block in `fixed` mode = the old Gaussian inside the new model; its
  σ sweep is the Pareto front every other arm is scored against.
* **S-global**: one learned kernel per joint group (position / rotation), initialised from the
  fixed Gaussian. Tests whether the fixed width is the right width.
* **S-adaptive**: kernel logits emitted per joint and frame by the transformer, initialised to
  the same Gaussian. Tests content dependence. A convex kernel passes a common translation bias
  through unchanged, so any accuracy gain must come from the residual path and is reported
  separately.
* **S-RNEA**: S-adaptive + RNEA (producer conventions), soft gate; **S-RNEA-hard**: the hard
  detached gate (prob > 0.5). Two arms, one factor each.
* **J2**: the best of the above with `detach_input: false`.
* Per-frame twin for C5(a): window 1 frame AND the smoother support 0 AND no `raw − mean`
  channel, otherwise the "per-frame" arm is still temporal.
* Losses on the refined body: kp3d, orient / pose, root_bias / root_shape (the demeaned root
  position error; it is not a derivative loss) vs the consistent targets of Phase 0 item 4;
  `pose_vel` / `pose_acc` weights set from measured gradient norms on the pose output, not from
  loss values; contact stillness OFF on the pose path.
* Diagnostics: kernel width and centre of mass vs joint speed and contact state;
  `jitter − gt_jitter` (a diagnostic, not a bound: the GT jitter is the solver's); the
  shrinkage monitor; window sweep 1 frame / 0.15 / 0.3 s.
* Go / no-go: S-adaptive on or beyond the fixed front AND ahead of S-global → the raw-input
  model is the recipe. Otherwise the smoothing stays a fixed or global kernel and the paper says
  the learned block matches but does not beat it.

## Phase 2 — contacts, forces, image features (C2, C3), sequential

Not a factorial. Three questions in order, one factor per step, 3 seeds each:

1. **Appearance increment, force branch OFF**: refiner inputs {trajectory only, + pose token,
   + anchor features}, window 0.15 s, `aug.token_dropout` on where features exist. Report the
   C2 metrics and the paired differences. The current decoder-token interval suggests ≤ 0.005 F1
   is what there is to find.
2. **Force supervision ON** on the best input set of step 1: Huber + noncontact L1 only, no
   coupling, no RNEA. This is the C2 test.
3. **Coupling** as its own factor: a **loaded-support head** (second binary target
   `|f_gt| ≥ 0.05 bw` from the force GT, its own BCE) and a force-derived readout
   `sigmoid(k(|f| − θ))` scored against THAT target, never against manual contact. The
   minimum-force hinge on predicted contacts is dropped (it contradicts 17 % of the positive
   rows and, gated by a detached probability, carries no contact gradient anyway).
* Contact head unchanged (BCE, heel weight 1). Heels are audited (annotator edit rate vs
  agreement with the stillness readout), not discarded.
* Per step: micro / per-group F1, P@R90, transition metrics, occupancy vs loaded support,
  force MAE / angle / corrIC.

## Phase 3 — RNEA (C4), after Phase −1 item 2

* **Nested video-level label subsets** 10 / 25 / 50 / 100 % with three draws each (10 % is
  eight videos; composition dominates otherwise), RNEA on all scenes vs off.
* **Controls on the same subsets**: static root balance as a loss; a direct constrained force
  solve (friction cone, NNLS) on the predicted pose + contacts at test time. RNEA must beat both.
* Every arm loads `gravity_world` (today it comes only with the force signal group); the
  held-out scenes' force-contact masks and confidences are never used as inputs.
* Off-contact |f| ≤ 0.02 bw, residual vs the reproduced producer floor, world-root / CoM accel.

## Phase 4 — temporality and dynamics (C5)

* **Matched-capacity twins**: the per-frame twin of Phase 1 vs the temporal model, same
  preprocessing, no clip-level features (causal running-mean betas).
* **Causal transition forecasting**: a head over a causal prefix (attention masked to the past,
  the smoother and derivative channels causal for this head, no future-derived input), targets
  "releases / grabs within 0.4 s" from the label transitions; baselines: persistence, a
  contact-duration hazard model, a per-frame pose + velocity MLP. Window 1 frame vs 0.15 / 0.3 s
  past context, trajectory-only vs + image features. Posture alone can predict transitions, so
  the claim is "beyond posture and history", nothing more.
* **Frame masking** on the best Phase-2 recipe: masked-position error (interpolation) and the
  C5 metrics, reported separately.
* **Invariance diagnostics** on the best model (not pass / fail): timestamp-preserving content
  permutation inside the window (should degrade), joint permutation of timestamps and content
  (should not), time reversal (a correct offline smoother is reversal-equivariant, the causal
  forecasting head is not), frame decimation with the spacing channels updated (robustness is
  desirable). Each result is interpreted against that expectation.
* **Window curve** (1 frame, 0.15, 0.3, 0.6 s) for every metric of C1–C4 on one figure.

## Fixed-camera track (development loop, and the camera-motion ablation)

The static subset (`configs/datasets/climbing_videos_static.yaml`: 113 train / 16 test scenes,
21 / 2 minutes) is the fast loop for Phase 1 and the sequential Phase 2, and the camera becomes
an explicit factor. The "static" VGGT extrinsics wobble; measured on the stage-1 dumps
(45 static scenes) and, per the second review, propagated to the GT joints:

| wobble about the per-scene mean | median | worst scene |
|---|---|---|
| rotation, rms | 0.04° | 0.91° (`RVL7DuOL9EU_0114`) |
| camera centre, rms | 2.2 mm | 23.7 mm (`US6c-J7Rlls_0000`; `s-ArwEzr-2M_0025` 22.4) |
| induced joint-coordinate change, rms | 2.5 mm | 43.9 mm (`RVL7DuOL9EU_0114`; `BFFCd9gLmXo_0020` 19.2) |

Rotation × subject distance dominates, so the gate is on the **induced joint displacement**
(`data.fixed_camera_max_mm` 15 mm applied to the GT joints re-expressed through the frozen
camera), not on the camera centre. The GT pelvis acceleration is 9 % higher in the per-frame
camera frame than in the world, so the wobble is extrinsic noise and the GT world trajectory
is kept as it is: **fixed-fixed** = per-scene chordal-mean rotation + mean translation
(`data.fixed_camera`). `camera_context` is NOT constant under a fixed camera (it is body-relative
and box-relative); it stays a per-frame channel.

* **Iteration speed**: 10–15 min per run at 600 updates (the budget scaled with the data),
  memorisation contained as on the full set.
* **A clean smoothing problem**: constant extrinsic → the lift is rigid, camera-frame and
  world-frame smoothing coincide, every jitter number is the body's own.
* **The camera ablation that isolates something**: the same static training data with raw vs
  frozen extrinsics, evaluated on all 16 static test scenes with the rotation and induced-
  displacement diagnostics per scene. Static-trained vs all-trained on the moving test scenes
  is reported as descriptive only (it changes data size, domain and motion distribution).
* Caveats: 16 test clips give wide intervals (whole-scene evaluation mandatory, frame counts
  reported); the static scenes are a biased sample; nothing is claimed from this track alone.

## Root pose (the camera / crop concern)

* SMPL-X head with `camera: ray`; plus a **crop / focal perturbation test** (jitter the box and
  the focal at test time, measure the root response), because the cached token still comes from
  a crop and monocular depth stays ambiguous.
* **Persistent contact anchors** (WHAM-style, replacing the same-frame anchored coordinates,
  which cancel the root exactly: `p − mean(e_i) = −R · mean(r_i)`): at a predicted contact onset
  the extremity's world position is latched (detached) and held until release; the root head
  reads the root relative to the latched anchors and predicts a delta; the along-ray wander
  (88 mm rms) is visible as drift of the body away from a held point. Nothing latched → the
  current body-frame delta.
* Gravity in the token (Phase 0), static and moving cameras reported separately.

## Status after round 5 (2026-09-07 evening; numbers in `round5_2026-09-07.md`)

| claim | status | what decided it |
|---|---|---|
| C1 smoothing without the Gaussian | **not met** | untrained stage 1 + Gaussian σ 0.06 = 55.42 mm beats every trained arm (best 55.79); S_global stays at its initial width; S_adaptive learned a comb kernel (jitter 22) because the legacy derivative losses are Nyquist-blind; with aligned stencils it matches F (56.22 / 3.76) and nothing more; no kernel at all = no smoothing (S_none_tok 57.8 / 71) |
| C2 forces improve contacts | unresolved | force branch on − off: F1 +0.0015, precision +0.0055, transition +0.023 — 1.5–2× the seed spread, one seed |
| C3 image features beyond the trajectory | not found (pose token); untested (anchor features) | trajectory-only = with pose token on every metric; the anchor cache was not built |
| C4 RNEA useful | untouched beyond conventions | contact-frame wrenches change nothing; the six-group floor is 0.186 bw (producer parity done); the subset ladder was not run |
| C5a temporal context | **held** | G (one-frame window) × 3 seeds: −0.018 F1, transition F1 0.387 → 0.24; test-time window 0.01 costs 0.03 F1; invariance: contact is locally order-blind, globally temporal, reads the arrow of time (reversal −0.02 F1, all groups) |
| C5b causal forecasting | not built | — |
| joint model J1 | negative | trainable per-frame head overfits (test loss rises from epoch 7), refined body −0.85 mm |
| token channels (Phase 0) | **held** | F_tok: force MAE 0.192 → 0.183, angle 21.5 → 19.9°, contact / MPJPE unchanged (seed spread 0.0005 / 0.4°) |
| model size / schedule | keep the round-4 block | the plan's 0.35 M block loses 0.012 F1 and 0.165 transition F1 (0.3 s receptive field) |
| fixed-camera track | **negative** (two seeds) | frozen extrinsics: training on them is worse on every metric under either test protocol (F1 −0.004 to −0.011, MPJPE +0.2 to +0.8, force MAE +0.006 to +0.010); testing under them adds 12 mm pelvis error; jitter is already at the GT floor on the static subset with raw extrinsics |

Seed spread (F × 3): F1 0.002, transition F1 0.013, MPJPE 0.02 mm, force MAE 0.0005, angle 0.4°.
Reading rule (user): ~0.005 F1 / ~0.1 mm is noise even with an interval that excludes zero.

## Order and cost

Phase −1 audits (2 days) → Phase 0 (2 days) → Phase 1 on the fixed-camera track (half a day)
→ Phase 1 winners on all scenes (half a day) → Phase 2 sequential (1 day) → Phase 3 and 4
(1 day) → the raw-vs-fixed extrinsics ablation (half a day). Nothing after Phase 0 runs the
backbone or the decoder.

## What the second review changed (2026-09-07)

C1 acceptance was an empty set (jitter ≤ 2.75 and ≥ 7.2) → Pareto front vs the fixed family.
Same-frame contact-anchored root coordinates cancel the root → persistent latched anchors.
The transition head leaked the future through the bidirectional block and the clip-mean betas
→ causal head, causal betas. Frame masking was bypassed by the residual base and the kernel
neighbours → observation vs target validity. The contact–force hinge contradicted 17 % of the
positive rows → loaded-support head. `root_shape` has no stencil. The anchor cache was 384 KB
per frame → pooled. The fixed-camera gate missed rotation × distance → induced-displacement
gate; `camera_context` is not constant. Reversal degradation is not a dynamics test →
invariance diagnostics. The factorial became sequential. The learner's RNEA must first match
the producer (stencil, levers, Dempster densities, total mass, torque terms). Phase −1 audits
precede the rewrite.
