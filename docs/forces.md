# Contact forces — what we built, how it trains, and what we learned

This page explains the force-prediction extension from start to finish: what the
model outputs, where the training signal comes from (physics, not labels), every
loss term and why it exists, how the training runs are set up, and — importantly —
why our first attempt collapsed to a constant "everything pushes up" answer and
what we changed to fix it. It is written to be readable top to bottom; the code
lives in `contact/physics/` and `sam_3d_body/models/heads/force_head.py`, and the
knobs in `configs/base.yaml` (`physics:`, `model.force_head`).

## The idea in one paragraph

We want the model to look at a climbing video and say, for each hand and foot,
what 3D force that limb is exerting against the wall. Nobody has labels for this —
you cannot annotate newtons by watching a video. But physics gives us a teacher
for free: if we reconstruct the climber's motion over a clip, Newton's laws tell
us exactly what the *total* external force and torque on the body must have been.
So we let the network guess per-limb forces, plug those guesses into the equations
of motion of the reconstructed body, and penalise whatever imbalance remains. If
the guessed forces explain the motion, the imbalance is zero. That imbalance —
the "residual wrench" at the root of the body — is the loss.

## What we added to the model

The base model is a frozen SAM-3D-Body: a DINOv3 backbone plus a promptable
transformer decoder that already predicts pose (MHR parameters) and, from our
earlier work, per-extremity contact probabilities. On top of that we add:

- **Four force tokens**, one per extremity, anchored to the same keypoints the
  contact tokens use (`[62, 41, 13, 14]` = left wrist, right wrist, left ankle,
  right ankle). They ride through the same decoder as everything else.
- **A force head** (`force_head.py`) that reads those four tokens and regresses
  one 3D vector each: `out["force"]["joint_forces"]` of shape `[B, 4, 3]`, in
  the fixed order `left_hand, right_hand, left_foot, right_foot`. The final
  linear layer is **zero-initialised**, so an untrained model predicts exactly
  zero force everywhere — more on why that is nice below.
- **An optional force temporal module** (`model.force_temporal`) — the same
  gated cross-frame attention block the contact branch uses, so each limb's
  force can look at neighbouring frames of the clip. It attends `per_token`
  (each limb mixes only with itself through time).

One structural rule keeps all of this safe: the decoder's attention mask is
asymmetric. The force tokens are appended *after* the contact tokens, and no
earlier token block is allowed to attend a later one. Original tokens never see
contact or force tokens; contact tokens never see force tokens. The practical
consequence, which `tests/test_force_invariance.py` proves numerically, is that
adding the whole force branch changes the pose and contact outputs by exactly
nothing — their Jacobian with respect to every force parameter is identically
zero. We can bolt this on without any risk to what already works.

Only parameters whose dotted name contains `"contact"` or `"force"` are
trainable at all (the freeze filter is name-based); everything else in the
network stays frozen and eval-pinned.

## What the numbers mean: units and frame

**Units.** The head predicts force in units of the person's **body weight**
(`m·g`), not newtons. A prediction of `(0, 1, 0)` means "one body weight,
straight up". This keeps the regression target around 1.0 regardless of whether
the climber weighs 55 kg or 95 kg, and the physics loss converts to newtons
internally using the per-clip shaped body mass. It also makes the zero-init
head meaningful: "no force" is a real, physical starting point, and the loss at
initialisation equals the pure-kinematics baseline — the model starts from an
honest "I don't know yet" rather than random noise.

**Frame.** The head predicts in the `local_world_aligned` frame: the axes of the
current camera, but with y pointing up (the un-flipped OpenCV frame — what a
level camera would call right/up/forward). This is deliberate. The
reconstruction world's yaw is arbitrary per scene, so the network could never
learn it from a single crop; the camera-aligned frame is something the network
can actually infer from the image. The physics loss rotates these predictions
into the world (through the known camera extrinsics) before using them. There is
also a `local` option (the extremity joint's own frame); `local_world_aligned`
is the default and the only one our renderers draw.

Forces are applied at the wrist/ankle **joint origins** with no torque
component. Real contact happens on the palm and sole, slightly offset, and grips
can transmit moments — that `r×f` error is a known, accepted v1 simplification.

## Where the supervision comes from

### Rebuilding the motion in one static world

The frozen model gives us the body's pose per frame, but in *camera*
coordinates — and our cameras move. If we ignored that, camera motion would
masquerade as body acceleration and poison the physics. So every ClimbingVideos
scene carries **per-frame camera extrinsics** (`cam_from_world`, OpenCV
convention, metric scale) exported from the video reconstruction pipeline,
plus a per-scene **gravity direction** (`gravity_world`, the first camera's
down-axis mapped into world coordinates) and the scale factor used. The adapter
(`contact/physics/adapter.py::MHRAdapter`) composes the per-frame camera pose
with the model's camera-frame body pose so that every frame of the clip lands
in one static, metric world. The result is a configuration trajectory `q(t)`
for a BetterHuman MHR body — a floating-base skeleton with 132 configuration
variables whose mass and inertia come from the predicted body shape.

Cameras are a **dataset** input, not a model input: the network itself never
sees extrinsics and stays deployable on plain images. The "someone provides
cameras" contract lives only in the dataset schema and the loss.

### Newton's laws as the loss

Given `q(t)`, the loss (`contact/physics/loss.py::PhysicsLoss`) does four
things:

1. **Smooths** the trajectory with a small windowed kernel (`[0.25, 0.5, 0.25]`),
   done properly on the manifold — positions and joint angles get a weighted
   mean, the root orientation gets a hemisphere-aligned quaternion average.
2. **Differentiates** twice: finite differences give velocity and acceleration,
   using the real elapsed time between frames and the SE(3) logarithm for the
   free-floating root.
3. **Runs inverse dynamics (RNEA)** with the predicted forces attached as
   external forces at the four extremities:
   `tau = M(q)·a + b(q,v) + g(q) − Jᵀ·f_ext`.
   RNEA answers the question "what generalised forces would have been needed to
   produce this motion, given these external forces?"
4. **Reads the answer at the root.** The first six components of `tau` are the
   force and torque that would have to act on the free-floating root joint. But
   nothing actuates a human's root — no motor holds you in space. For a
   physically consistent explanation, those six numbers must be **zero**. Their
   magnitude is the residual we minimise.

The intuition: gravity pulls the body down, the body accelerates however the
video says it accelerates, and the only things that can reconcile the two are
the contact forces. If the network predicts them right, the books balance at
the root. If it predicts nothing, the residual is exactly the unexplained
gravity + acceleration — the "pure kinematics" baseline the zero-init head
starts from.

Everything is normalised to dimensionless units: residual force by body weight
`m·g`, torques by `m·g·1 m`. A note for the curious: RNEA never inverts the
mass matrix, which matters because MHR's mass matrix is singular (mimic joints,
zero-mass cosmetic bodies) — inverse dynamics works fine, forward dynamics
would not.

### Which frames actually get supervised

A frame only contributes to the residual if its full stencil fits inside the
clip: one frame of smoothing radius on each side, plus two more on each side
for the double central difference. With the default kernel the residual frames
are `{t : 3 ≤ t ≤ T − 4}`, which means **a clip needs T ≥ 7 to produce any
physics signal at all**. This bit us: contact training used T=5, where the
physics objective is silently dead. The first force runs used T=8 (2 residual
frames per clip); the current config uses **T=16 with frame stride 2**, giving
10 residual frames per clip and, because stride 2 doubles the time step,
roughly 4× less amplification of reconstruction noise through the double
derivative (velocity scales as 1/dt, acceleration as 1/dt²).

One honest limitation: the acceleration is formed by central-differencing the
velocity with a doubled interval, which is exact only for uniformly spaced
frames. Our frames are uniformly spaced, so this is currently harmless, but the
formula would be wrong for non-uniform sampling; a rewrite is deferred.

### Throwing away clips the reconstruction ruined

The camera compensation is only as good as the reconstruction, and a few scenes
have genuine glitches — train scene `45KmZUc0CzA_0007` contains a 7.85 m
camera-centre jump between two adjacent frames. Fed through a double
derivative, a jump like that turns into an absurd fake acceleration that the
forces can never explain. So the dataset emits, per clip row, the metric
distance between the camera centres of **consecutive sampled clip frames**
(`cam_jump_m`), and the loss drops any clip whose largest jump exceeds
`physics.max_cam_jump_m` (0.5 m in the current config). Two details matter:
the distance is measured between *sampled* frames, because with stride 2 a
glitch on a skipped frame would be invisible to per-source-frame checks (an
audit found 12 of 18 large stride-2 jumps were exactly that); and the filter
keys on upstream camera evidence, never on the model's own residual, so the
force branch cannot learn to game it. Excluded clips are counted and logged.
If filtering ever excludes *everything*, the evaluation headline becomes NaN
and the trainer raises — an empty split must never read as a perfect score.

## The loss terms, one by one

The total physics loss is a weighted sum (`physics.loss.*`, all dimensionless):

**`residual`** — the root-wrench residual described above; the actual training
objective. By default it is the plain squared norm `‖r_f‖² + ‖r_τ‖²`. The
current config instead applies a component-wise **pseudo-Huber**
`ρ_δ(x) = δ²(√(1+(x/δ)²) − 1)` with `delta_force = 1.0`, `delta_torque = 0.5`:
quadratic near zero, linear past δ. The reason is that the supervision is
heavy-tailed — the residual comes from double-differencing a noisy
reconstruction, and while the median per-clip residual is around 1, the p99 is
~18–28 and the worst clips reach the hundreds, concentrated in a few scenes. In
squared-error land those few clips dominate every gradient step; the linear
tail caps their influence. Importantly, the **reported and monitored number is
always the raw, un-robustified residual**, so runs remain comparable no matter
which robustifier trained them. (Comparable under the same clip protocol, that
is — changing T or stride changes dt, the stencil, and which rows are scored,
so numbers from a T=8 run cannot be compared to a T=16 run without
re-evaluating under the same protocol. We learned to be careful with this.)

**`force_noncontact`** — the term that forbids cheating. Without it there is a
trivial solution: put whatever forces you like on the airborne limbs and zero
the residual with pure fiction. Penalising force wherever the model itself says
"no contact" removes that escape. The penalty has two config-selectable forms
(`physics.loss.noncontact_gate.kind`):

- `soft_l2` (default, the original) — `(1−p)·‖f‖²`. Being quadratic, its
  stationary point against the residual's pull is `‖f‖ ∝ 1/(1−p)` on **every**
  limb: it shrinks forces, it never zeros them. The t7mid run measured exactly
  this equilibrium — `‖f‖` a smooth increasing function of `p` (corr 0.72),
  free limbs holding ~0.1 bw, and raising the weight cannot fix it because the
  soft `(1−p)` weight also taxes true-contact limbs whose `p` is merely 0.6–0.9.
- `hinge_l1` — `hinge(p)·‖f‖` with `hinge(p) = clamp((p_hi − p)/(p_hi − p_lo),
  0, 1)`: full penalty at `p ≤ p_lo`, none at `p ≥ p_hi`, linear ramp between.
  The L1 magnitude has a constant slope at `‖f‖ → 0` (eps-smoothed, so the
  gradient is finite at exactly zero), which creates a **dead zone**: on a limb
  with `p ≤ p_lo` the exact-zero force is a stationary point whenever the
  per-limb penalty slope `(w/4)·hinge(p)` exceeds the residual's pull. The
  hinge decouples killing free-limb force from taxing contact limbs.

Two subtleties shared by both forms: the gate uses the
**predicted** probability, not the ground-truth label, because our video labels
mark motion-gated *stable* contact — a hand can be load-bearing for an instant
without counting as a stable contact, so the labels are the wrong gate for
instantaneous force. And the probability is **detached**, so the force loss
can neither train the contact head nor learn to inflate contact probabilities
to license fictitious forces.

**`force_at_contact`** — `p·relu(min_force − ‖f‖)²`, a weak nudge that contact
should carry at least a little force. We ship it **disabled** (weight 0.0), and
the reason is a genuine trap worth recording: the force head is zero-init, and
this term's gradient at `f = 0` is exactly zero (the softened magnitude
`√(‖f‖²+ε)` has vanishing derivative there). It cannot lift the head off zero;
it can only shape predictions that are already nonzero. It sat in early configs
looking useful and doing nothing.

**`force_smooth`, `force_l2`, `torque_l2`, `torque_smooth`** — small
regularisers: forces should change smoothly over time, shouldn't be huge, and
the implied joint torques shouldn't be huge either. In the current config these
are deliberately light (0.02 / 0.001 / 0.001 / 0), an order of magnitude below
the first run — we measured that they were *not* the cause of the collapse
(they were 30–1000× smaller than the residual term), but there is no reason to
let a magnitude prior argue with the physics.

## How training runs

Two regimes exist:

- **Regime (a), warm-start — the one we use.** Load the contact branch
  (including its temporal module) from a trained contact checkpoint, freeze it
  (`train.freeze_contact: true`), and train only the force branch. Contact
  quality is exactly the source checkpoint's, guaranteed untouched, and the
  gradient isolation is exact: the physics loss consumes pose, cameras, and
  contact probabilities all detached, so gradients flow into force parameters
  only.
- **Regime (b), scratch** — train contact and force together. This works, but
  with a documented wrinkle: force tokens are allowed to attend contact tokens,
  so physics gradients can leak into the trainable contact head through that
  attention path. The trainer prints a loud warning; a detach-fix inside the
  vendored attention is deferred. All shipped force runs use regime (a).

The warm-start source deserves a note, because we measured our way out of a
wrong assumption here. The natural choice was the best temporal contact
checkpoint, but temporal checkpoints were trained at T=5, and the physics needs
T=16. Does a T=5-trained temporal module survive T=16 inference? It depends
which one: `climb4_t5` (trained to predict all frames) is nearly a passthrough
and holds F1 0.888 at T=16/stride 2 (vs 0.897 native); `climb4_t5mid` (trained
to predict the centre frame) genuinely uses its temporal context and
**collapses to F1 0.70** outside its training window. So the current config
warm-starts from `climb4_t5` and pins the temporal architecture to byte-match
the source; the checkpoint loader hard-fails on any architecture mismatch
rather than silently reinitialising.

Other training mechanics worth knowing: the gradient clip is 5.0 (raised from
1.0 — see the collapse story below) and the raw pre-clip norm is logged every
step; a non-finite raw norm raises immediately instead of being masked by the
clip. Checkpoints of regime-(a) runs are **self-contained**: they store the
frozen warm-started contact tensors alongside the trainable force tensors, so
evaluation, demo, and resume reconstruct the exact deployed model from the one
file. (The first implementation saved only trainable parameters, which meant
every downstream consumer silently ran with a randomly initialised contact
branch — test F1 0.15 instead of 0.889. Legacy force-only checkpoints
auto-recover their contact weights from the config's `init_contact_checkpoint`.)
The monitored metric is `test/physics_residual` — the raw residual, minimised.

## Why the forces kept converging to "always up"

This is the most instructive part of the project so far, so it gets its own
section. The first force run (T=8) converged to a near-constant answer: about
one body weight, pointing up, spread across the limbs — roughly the same on
every frame, with almost no dependence on the image. Understanding why turned
out to be a lesson in what the physics objective can and cannot see.

**The residual mostly constrains the *sum* of the forces.** Split the root
wrench into its force part and its torque part. The force part is just
Newton: the sum of all external forces must equal mass × acceleration minus
gravity. Climbing is quasi-static — accelerations are small — so on almost
every frame this says "the four forces must add up to ≈ 1 body weight, up".
Notice what it does *not* say: nothing about which limb carries the load. A
constant prediction of ¼ body weight up on every limb satisfies the force part
of the residual on nearly every frame *without ever looking at the image*.
That is exactly the solution the run found; it is not a bug in optimisation,
it is a real, deep minimum of that part of the objective. And it answers the
"why up?" question directly: gravity is the dominant term in the books the
forces must balance, so the cheapest input-independent answer is the one that
cancels gravity — up.

**Allocation lives in the torque part, and the torque part is weak.** What
distinguishes "load on the left hand" from "load on the right foot" is the
lever arm: the same force at different application points produces different
torques about the root. So the per-limb information is there — but only in the
torque residual, and on our data the torque part is ~23× smaller than the
force part (median 0.033 vs 0.74 in the units of the loss). The one signal
that could break the symmetry is a whisper next to the force-sum's shout.

**Even in principle, one frame does not pin down four forces.** The root
wrench is 6 numbers; four unknown 3D forces are 12. With two limbs in contact
the reachable wrench space has rank ≤ 5. The problem is underdetermined
frame-by-frame, and the soft contact gate does not change that (a probability
reweights a penalty; it does not remove unknowns). Disambiguation has to come
from accumulating many frames, the noncontact gate, and temporal smoothness —
which makes it doubly important that the weak allocation signal survives
optimisation.

**Noise and clipping finished the job.** The supervision is heavy-tailed: a
handful of clips with bad reconstruction produce residuals in the tens to
hundreds against a median near 1. Under squared error those clips dominated
whole batches. Meanwhile the raw gradient norm ran at 15–28 against a clip
threshold of 1.0 — meaning *every* step was clipped ~20×, and the update
direction was effectively "whatever the heavy tail says", drowning the subtle
torque signal entirely. The regularisers, which we initially suspected of
selecting the low-variance constant solution, were measured to be irrelevant
(30–1000× smaller than the residual); the story was force-sum dominance plus
tail noise plus over-clipping.

**What we changed.** Each diagnosis got a counter: pseudo-Huber tames the
tails; T=16/stride-2 gives 5× more residual frames per clip and ~4× less
differencing noise (raising the SNR of the torque whisper); the camera-jerk
filter removes the worst supervision outright; the clip moved to 5.0 with the
raw norm logged so silent 20× clipping can never happen unnoticed; the
noncontact gate was doubled (it is the strongest allocation signal we control);
and the regularisers were lightened so nothing argues with physics. Just as
important, we built the test that makes collapse *visible*: the root wrench is
affine in the predicted forces, so the evaluator computes, in closed form, the
best possible **constant** force solution and the residual of the network's
own predictions **shuffled across clips**. An input-dependent model must beat
both. The collapsed T=8 model, re-scored under the new protocol, loses to the
fitted constant (0.285 vs 0.271) — it was worse than not looking at the input.
The redesigned run passes at epoch 2: network 0.266 < constant 0.271 <
shuffled 0.299, with per-limb correlation between force magnitude and contact
probability of 0.27–0.52 (previously ≈ 0 for the feet) and contact F1 intact.

**What still points up, and why that is partly correct.** Even in the
redesigned run, limbs the model believes are airborne still carry ~0.2 body
weight of mostly-upward force at epoch 2. Some of this is simply an
unconverged model — the run was stopped early, the residual was still
descending, and the noncontact gate grinds those forces down over epochs. But
part of the upward bias is honest physics: the *sum* genuinely must be one
body weight up, and until the torque signal has fully allocated the load, the
optimiser hedges by spreading the mandatory total across limbs it is unsure
about. Watching the free-limb magnitude fall while the residual and the
affine margins improve is exactly the signature of the allocation being
learned; watching it plateau would mean the torque signal is still too weak,
and the next levers would be a stronger gate, longer clips, or better
reconstructions.

## How we evaluate

`scripts/evaluate.py`, on runs with the force branch enabled, reports alongside
the usual contact metrics:

- the **raw physics residual** (the headline and monitor, robustifier-independent);
- the **affine input-dependence baselines** described above — zero forces,
  best-fitted constant, the network, and clip-shuffled network predictions,
  each with mean and tail quantiles, plus an explicit PASS/FAIL line ("network
  beats constant AND shuffled");
- the **vertical force sum** (should sit near 1 body weight for quasi-static
  climbing; 0.886 at epoch 2);
- **gate-violation rates** (mean force on predicted-noncontact limbs, and the
  fraction of predicted contacts carrying less than the minimum force);
- per-extremity force magnitudes split by predicted contact state, in body
  weights and newtons;
- the residual saturation fraction (how often components exceed the
  pseudo-Huber δ) and the number of jerk-excluded clips.

For qualitative checks, `scripts/render_climbing_video_contacts.py` renders
test videos with contact disks (inner = label, outer ring = prediction) and
**force arrows**: the predicted 3D force is treated as a metric segment (1 m
per body weight) at the extremity's camera-frame position and
perspective-projected through the dataset's per-frame intrinsics, so on-image
direction and foreshortening are the real camera's. The retired
`legacy/demo_climbing_videos.py` drew force arrows on still panels too (via the
model's own intrinsics, with screen-space length — a simpler scheme than the
video renderer's).

## Assumptions and known limitations

- **Gravity = the first camera's down-axis.** We assume the camera is level at
  scene start; an initial tilt biases the residual (camera *motion* is
  compensated, initial *orientation* is trusted). The direction is stored in
  the data (`gravity_world`), so a better upstream estimate can replace it
  without code changes.
- **Reconstruction quality is the real supervision quality.** Extrinsics
  drift and scale error feed straight into fake accelerations. The jerk filter
  removes the catastrophic cases; per-scene residual monitoring is the tool
  for the rest.
- **Uniform body density** (1000 kg/m³; LOD-1 neutral mass ≈ 81.5 kg), with
  inertias recomputed from the predicted shape.
- **Forces at joint origins, no contact moments.** Palm/sole offsets and grip
  torques are unmodelled (`r×f` error), accepted for v1.
- **Stable-contact labels ≠ instantaneous load.** This is why the gate uses
  predicted probabilities; the mismatch is inherited and documented.
- **Regime (b) gradient leak** through force→contact attention (warned at
  runtime, fix deferred). Regime (a) is exact.
- **Performance**: the adapter builds the shaped body twice per call and FK is
  kernel-launch-bound; known, measured, deliberately un-optimised until it
  actually hurts.
- **Checkpoint signature omits temporal dropout** (every shipped artifact uses
  0.0; adding the key now would orphan existing checkpoints — to be absorbed
  at the next signature-version bump).

## Supervised forces from the corpus (kindyn)

The physics residual taught us its structural lesson (§ the collapse): the force part of the
root wrench pins only the *sum*, and the allocation signal in the torque part is ~23× weaker.
The corpus removes the need to fight that geometry. Its reconstruction pipeline already runs an
inverse-dynamics solve per scene (`features/human_optim/<scene>/kindyn_1.npz`): forces solved
jointly with the pose under an RNEA base-wrench objective, for **six** contact groups instead of
our four — each foot splits into its big-toe joint and its heel. Those solved forces become
plain regression labels, and `contact/force_supervision.py` trains the force branch against
them directly. Physics and supervised training are **mutually exclusive** by config validation
(`physics.enabled` vs `force_supervision.enabled`): one supervision signal per run.

**The six groups and their anchors.** The head grows to six force tokens with its own MHR70
anchor list, decoupled from the contact anchors via `model.force_head.force_keypoint_indices`:
`[62, 41, 15, 18, 17, 20]` = left wrist, right wrist, left big-toe tip, right big-toe tip, left
heel, right heel — exactly kindyn's column order `left_hand, right_hand, left_foot(toe),
right_foot(toe), left_ankle(heel), right_ankle(heel)`. Hands still fold the fifteen finger
joints onto the wrist.

**Frame and units.** GT forces are stored in newtons in the scene world frame; the loader
(`contact/data/climbing_corpus.py`) normalises by each person's solved body weight
(`total_mass · g`) and rotates **world → body-root** with the kindyn root quaternion
(`q[3:7]`, xyzw, verified numerically against the stored axis-angle `global_orient`). The head
predicts in that body-root frame directly (`model.force_head.frame: root`) — **no camera
extrinsics appear anywhere in the objective**, unlike the physics loss whose
`local_world_aligned` frame needs the per-frame camera to reach the world. Recovering world
forces at analysis time is the inverse rotation through the predicted root pose (plus
extrinsics if a camera frame is wanted).

**The loss** (`ForceSupervisedLoss`) has two terms, in the same `(numerator, mass)` contract as
the physics terms so DDP reduction stays an exact global mean:

- **force** — smooth-L1 (Huber, `huber_delta_bw` 0.5) between prediction and GT on valid
  **in-contact** limb-frames. Huber because the solver's tails are heavy (in-contact `|f|` p99
  ≈ 1.6 bw, max 48 bw): quadratic near zero for a clean mean, linear past the delta so spikes
  cannot dominate the gradient. On top of that, limb-frames whose GT magnitude exceeds
  `outlier_bw` (4 bw) are **excluded outright** — those are solver blowups on bad
  reconstructions, ~0.1 % of frames, not supervision.
- **noncontact** — plain L1 on the predicted magnitude of valid **non-contact** limb-frames.
  The gate is kindyn's own contact mask (bit-identical to contacts_2 — the exact mask the
  forces were solved under), not predicted probabilities: there is no contact branch in this
  run at all. GT is identically zero there by construction (zero force means *unlabeled*, not
  measured-zero), and L1's constant slope at `‖f‖ → 0` admits exact zeros — the t7hinge lesson
  carried over.

**Force-only build.** Setting both contact targets off with explicit force anchors builds a
model with *no contact tokens and no contact head at all* — the six force tokens are the only
trainable addition, the block-triangular mask degenerates to original ⊥ force, and
`out["contact"]` is `None`. `tests/test_force_only_build.py` proves the trainable set, the mask
pattern, and MHR noise-floor invariance for this shape. The first experiment is
`configs/climbing_corpus_force_supervised.yaml` (`corpus6_force_sup_t7`): T=7 / stride 1 clips
with center-frame supervision, force temporal attention on and `attend: joint` (the allocation
coupling argument from t7mid still applies), monitor `val/force_mae` (mean in-contact error
norm, bw, minimised).

## Pointers

- Code: `contact/physics/adapter.py` (SAM ↔ BetterHuman bridge),
  `contact/physics/loss.py` (objective + diagnostics + affine baselines),
  `contact/force_supervision.py` (supervised kindyn-force loss),
  `sam_3d_body/models/heads/force_head.py`.
- Configs: `configs/climbing_videos_force_warmstart_t7hinge.yaml` (the kept physics run,
  flattened + heavily annotated), `configs/climbing_corpus_force_supervised.yaml` (supervised
  kindyn forces); defaults in `configs/base.yaml`. Retired physics-run configs
  (`_warmstart`, `_scratch`, `_t16`, `_t7mid`, …) are archived in `legacy/configs/`.
- Design record with the original decision/risk register: `plan/README.md`
  and `plan/for_agents/`.
