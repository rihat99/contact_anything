# Plan: making the temporal / physics claims testable (2026-09-07)

*Planning page. Follows the 2026-09-07 adversarial review
(`~/.claude/handoffs/adversarial_review_20260907.md`, analyses in
`~/.claude/handoffs/adversarial_review_20260907_analyses/`). Pose accuracy is NOT a target of this
plan: the train-fitted linear ceiling for any trajectory-only refiner is ~0.5 mm of 57 and half of
the remaining error inside a 60-frame window is a per-clip constant.*

## Claims to make hold, and what "hold" means

| # | claim | passes when (all with paired bootstrap CIs over the 108 test clips, 3 seeds) |
|---|---|---|
| C1 | the model smooths the pose by itself, no Gaussian at its input | referenced accel error and lifted jitter ≤ the fixed-Gaussian baseline, jitter ≥ the GT floor (not below it), MPJPE within 0.3 mm of the Gaussian baseline |
| C2 | the force branch improves contacts | contact transition metrics (onset / offset timing, boundary F1 ±2 frames) and hand precision at R = 0.90 improve with the force branch, at matched steps |
| C3 | image features help contacts and forces | the same metrics improve from trajectory-only → + pose token → + anchor image features, and the gain survives with window > 1 frame |
| C4 | the RNEA loss is useful | (a) with force labels on a fraction of the scenes, RNEA on all scenes lowers test force MAE / angle vs supervised-only on the same fraction; (b) RNEA lowers the referenced accel error (physics as the smoother); off-contact force does not rise |
| C5 | the model uses temporal context (dynamics, not a per-frame map) | metrics degrade under window = 1 frame, under frame shuffling and under time reversal at test time; a contact-transition forecasting head (release / grab within 0.4 s) beats its per-frame twin |

Every table also carries the trivial baselines from the review: Gaussian-only pose (55.47), speed
threshold contact (micro F1 0.874), equal-split static force (0.24 bw / 30°), ridge on the stage-1
trajectory (0.191 bw / 19.8°). A claim that does not beat its trivial baseline with a CI is reported
as not holding.

## Phase 0 — infrastructure (do all of it before any new training)

1. **Cross-fitted stage 1.** Train stage 1 twice on disjoint halves of the 864 scenes (split by
   `video_id`), dump each half with the other fold's model, dump test with both and average
   nothing (pick fold A for test, report B as a check). Stage 2 trains ONLY on out-of-fold
   outputs. Cost ~2 × 1 h. Removes the in-sample leak (train kp3d 0.065 vs test 0.104).
2. **Stage-2 dataset from dumps.** Extend `dump_stage1.py` to write, per scene: stage-1 body,
   the frozen pose token, and **frozen anchor image features** — the `grid_size 5` DINOv3 features
   grid-sampled at the six MHR anchors of the frozen decoder's interm keypoints (what
   `LearnedTokenBlock` reads; no learned token, so it is cacheable), stored bf16 like
   `features/pose_token`. Stage 2 then runs with no backbone and no decoder: ~30 s per epoch,
   which is what makes 3 seeds × factorial arms affordable.
3. **Metrics.** Add: referenced accel error at the adjacent stencil (exists: `accel`); jitter
   minus GT jitter (signed); contact transition metrics (onset / offset timing error in frames,
   boundary F1 within ±2 frames, per group); in-contact-only force size correlation
   (`corrIC`); per-clip metric dumps and a paired-bootstrap script (`scripts/paired_ci.py`).
   Drop `corr` / `share pp` from the headline (an equal split beats them).
4. **Whole-scene evaluation.** `eval_max_frames` → tile the whole run (or ≥ 360) so the scored
   frames are not the first 4.8 s only; report static and moving cameras separately.
5. **Stencils.** Every loss derivative (`time_derivative`, `angular_velocity`,
   `pose_derivatives`, contact stillness, `force_consistency`) becomes a forward / adjacent
   difference so the objective sees the Nyquist band the jitter metric measures. Derivative
   targets are the RAW GT differences (Huber absorbs the noise); `label_smooth_sec` stays for
   the motion head only, never for the pose-derivative terms.
6. **Token hygiene.** Replace `geometry_norm` by a fixed per-channel standardisation from the
   train dumps; add the body-frame gravity direction (`gravity_world` rotated by the root) and the
   6D of the root + 21 joints to the token (the output is rotation deltas, the input must carry the
   rotations); drop the constant `dt` channel or keep it standardised.
7. **Augmentation hooks** (section below): image-feature dropout, frame masking and the
   residual-noise sampler, each an `aug.*` config key defaulting to off, applied in the
   stage-2 dataset / collate on the cached path so they cost nothing per step.

## Augmentations (built in Phase 0, switched per arm)

None of these touch the label ceiling or the per-clip bias; they fight memorisation at this
data scale and make the temporal / image comparisons fair. Mirroring is deliberately not in
this round.

* **Image-feature dropout** (`aug.token_dropout`, default 0.4): per frame, with that
  probability, zero the projected pose token and the anchor image features together and add a
  learned "missing" embedding. Purpose: the contact / force heads cannot memorise scenes
  through the 1024-d token, and a model that can lean on the image path alone would tell
  nothing about whether image features ADD to the trajectory (C3). Evaluation runs with the
  features present; a second evaluation with them dropped gives the trajectory-only twin of
  the same model for free.
* **Frame masking** (`aug.frame_mask`, default 0.2, spans of 2–6 frames): the masked frames'
  whole token (geometry + image features) is replaced by a learned mask embedding; the pose,
  contact and force losses are applied at the masked positions as at the others. A per-frame
  map cannot solve this, so the temporal block has to open. It teaches interpolation, not
  dynamics, so it supports C5 only together with the transition-forecasting head and the
  shuffle / reversal tests; the frame-masking arm is compared with and without on every C5
  metric. Masked frames keep their RoPE position and their `frame_valid`; the smoothing /
  derivative helpers see them as valid so no edge effects are introduced.
* **Realistic noise synthesis** (`aug.synth_noise`, Phase 1 only, used if S1 is data-starved):
  extra training clips made from the GT trajectory plus a corruption sampled from the
  measured stage-1 residuals, not white noise (the real error is coloured and biased; white
  noise teaches a skill the test set does not need). Per clip: a constant offset drawn from the
  empirical per-clip bias distribution of the train dumps (root along-ray, root rotation,
  per-joint root-frame), plus an AR(1) process with the measured 0.48 s correlation time and
  the measured per-joint fast-error RMS; rotations corrupted by right-multiplying a random
  small rotation series. These clips have no image features (the token path is dropped for
  them) and are mixed 1:1 with the real clips. The noise statistics come from the OUT-OF-FOLD
  dumps of Phase 0 step 1, never from in-sample outputs.
* Not used: white / OU noise on the inputs, time reversal (kept as a temporality test),
  mirroring (later round: needs the caches on flipped frames and a reflection test suite).

## Model size and schedule (all phases)

`dim 128, num_layers 2, num_heads 4, mlp_ratio 2, dropout 0` (~0.35 M params; the current 11.4 M
block stayed at identity), `window 0.15 s` per layer, pose token 1024→64, anchor features
6 × (1280 → 16), weight decay 0.05, lr 1e-3 with 200 warm-up steps, **4000 optimizer steps**
(the round-4 arms had 855; the toy sandbox needed 300 to leave the identity and 3000 to converge),
64 clips × 60 frames per step, EMA as is, 3 seeds per arm. Training clips 60 frames; the receptive
field 2 × 0.15 s fits inside them.

## Phase 1 — smoothing without the Gaussian (C1)

* **S0** current recipe with Gaussian input (the reference; re-run under the new stage-1 folds).
* **S1** raw lifted trajectory in (no `root_smooth_sec` / `pose_smooth_sec`), the extra
  channel `raw − 5-frame-mean` so noisiness is visible, and a **learned convex kernel head**: per
  joint and per frame the transformer emits logits over ±K = 6 frames, softmax, the output
  rotation is the projected weighted mean of the raw neighbours (`project_rotation`), root
  position likewise; a small zero-init residual on top. Initialise the logits flat over ±2
  frames (a poor smoother, so the learned shaping is visible). This gives the model the
  operator class a Gaussian belongs to and asks it to beat the fixed width by using content.
* **S2** = S1 + RNEA with adjacent stencils (physics as the smoother), gate = hard detached
  contact (prob > 0.5) so free limbs cannot absorb the residual.
* Losses for both: kp3d, orient / pose, root_bias / root_shape vs raw GT; `pose_vel` /
  `pose_acc` at adjacent stencils vs raw GT differences, weights ≤ 0.2 of the position terms;
  contact stillness OFF on the pose path (its L1-to-zero is the damping source).
* Diagnostics: learned kernel width vs joint speed (should narrow in fast motion, widen when
  still); `jitter − gt_jitter` per epoch (must not go negative); window sweep 1 frame / 0.15 /
  0.3 s.
* **S1n** = S1 + `aug.synth_noise` (GT trajectories with sampled stage-1-like residuals, 1:1
  with the real clips): run only if S1 trails S0 while its train loss keeps falling, i.e. the
  kernel head is data-starved rather than mis-specified.
* Go / no-go: S1, S1n or S2 meets C1 → proceed with the raw-input model everywhere. None
  does → the smoothing stays explicit and the paper says the learned block matches but does not
  beat a fixed kernel; move on with S0.

## Phase 2 — contacts, forces, image features (C2, C3), factorial on the cache

Arms (3 seeds each, ~2 min per run): inputs {trajectory only, + pose token, + anchor image
features} × force branch {off, on} × window {1 frame, 0.15 s}. Twelve cells, 36 runs.

* `aug.token_dropout 0.4` ON in every cell that has image features; each such model is
  evaluated twice, with the features present and with them dropped, so the image-minus-
  trajectory difference is also available within one model (a paired comparison that does not
  depend on seed variance between arms).
* Contact head unchanged (BCE, heel weight 1 — the heel weight only chases annotation noise).
* Force branch "on" = supervised force loss + a **contact–force coupling** in both directions:
  `noncontact` L1 as now, plus `hinge(θ − |f|)` on rows predicted in contact (θ 0.05 bw): a
  limb the model calls in contact must carry load. Also a **force-derived contact readout**
  `sigmoid(k(|f| − θ))` evaluated as a second contact predictor (the label's own force arm
  unions in frames carrying ≥ 0.02 bw, so this readout is fair).
* Report per cell: micro / per-group F1, P@R90, transition metrics, force MAE / angle /
  corrIC, and the paired differences force-on minus force-off and image minus trajectory.
* Expected honest outcome: frame-level F1 is saturated (0.92 vs a 0.874 threshold and a
  0.892 label ceiling); the room is in hand precision (~1 800 false positives) and in
  transition timing. If image features and the force branch show nothing there with CIs, C2 /
  C3 are reported as not holding at this data scale.

## Phase 3 — RNEA (C4)

* **Label-scarcity sweep**: force labels kept on 10 / 25 / 50 / 100 % of the train scenes
  (by video), RNEA on all scenes vs off; test force MAE / angle / corrIC per fraction. The
  claim is "physics substitutes for labels", the only form of RNEA usefulness the literature
  supports (PhysPT moves joint error ≤ 0.3 mm).
* RNEA variants: adjacent stencils, hard gate, torque weight as now; log off-contact |f|
  (must stay ≤ 0.02 bw) and the residual vs the GT floor (0.209 / 0.050).
* Smoothing effect: S2 vs S1 from Phase 1 on referenced accel.

## Phase 4 — temporality and dynamics (C5)

* **Transition forecasting head**: per frame and group, "releases within 0.4 s" / "grabs
  within 0.4 s" (labels from the contact label's transitions), trained with the rest; compare
  window 1 frame vs 0.15 / 0.3 s and trajectory-only vs + image features. A per-frame model
  cannot solve this; a model that learned load transfer can.
* **Frame masking** (`aug.frame_mask 0.2`) on the best Phase-2 recipe, with vs without:
  masked-position pose / contact / force error (interpolation quality) and every C5 metric.
  Masking that improves the masked-position numbers but nothing else means the block learned
  to interpolate, not dynamics; report it that way.
* **Test-time perturbations** on the best Phase-2 model: shuffle frames inside the window,
  reverse time, drop every other frame; each must degrade contacts / forces / accel by more
  than the CI, else the block is a per-frame map.
* **Window curve** (1 frame, 0.15, 0.3, 0.6 s) for every metric of C1–C4 on one figure.

## Root pose (the camera / crop concern)

* Stage 1 with `camera: ray` (crop-free, the 2026-09-04 result) in the cross-fitted runs; the
  CLIFF lift turns bbox noise into depth noise through the focal.
* Stage-2 root in **contact-anchored coordinates**: for each frame the root position is
  expressed relative to the mean world position of the extremities currently predicted in
  contact (detached), and the root head predicts a delta in that frame; a hand on a hold is a
  world-fixed point, so the along-ray wander (88 mm RMS) is visible as motion of the body
  relative to its anchors. Falls back to the current body-frame delta when nothing is in contact.
* Gravity direction in the token (Phase 0), `root_shape` at adjacent stencils, static and
  moving cameras reported separately.

## Order and cost

Phase 0 (2 days) → Phase 1 (1 day of runs) → Phase 2 (half a day of runs, all cached) → Phase 3
and 4 (1 day). Everything after Phase 0 runs on the dump cache without the backbone; the
decoder is never needed again unless C3 holds and a learned image token is worth its 15× cost.
