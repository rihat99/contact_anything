# Contact forces

Predict a 3D **contact force** at each of the four climbing extremities on top of
the frozen SAM-3D-Body + contact stack. No force labels exist; supervision is
**physics** — the RNEA root-wrench residual of the reconstructed motion. This
page is the narrative record of the formulation, units, frames, world/gravity
conventions, and the assumptions/risks that outlive `plan/`. Cross-check against
`contact/physics/` and `configs/base.yaml` (`physics:`, `model.force_head`).

## What the model outputs

- Four **force tokens** (keypoint-anchored, reusing the contact anchor indices
  `model.contact_head.contact_keypoint_indices` = `[62, 41, 13, 14]`), an optional
  **force temporal** block (`model.force_temporal`, a second `ContactTemporalModule`,
  post-decoder only), and a **force head** (`sam_3d_body/models/heads/force_head.py`,
  zero-init final linear) regressing one 3D vector per extremity:
  `out["force"]["joint_forces"] [B, 4, 3]`.
- Output order: `left_hand, right_hand, left_foot, right_foot` (same as the
  four-extremity contact joint target; see `contact.physics.EXTREMITY_OUTPUT_NAMES`).
- The force tokens are appended **after** the contact tokens and the asymmetric
  attention mask is extended (`token_mask[:, :force_start, force_start:] = False`)
  so no earlier token block attends a later one — MHR/pose **and** contact logits
  keep an exactly-zero Jacobian w.r.t. every force param (`tests/test_force_invariance.py`).

## Force units — body weight

The head predicts **dimensionless** force in units of the shaped model's body
weight `m·g` (decision D5); the physics loss converts to newtons as
`f_newtons = pred · m · g` where `m` is the per-clip shaped mass (kg) and
`g = physics.gravity` (9.81). Zero-init means the model starts at "no forces": the
RNEA residual then equals the pure-kinematics baseline, a meaningful curriculum
start. Body-weight units keep the regression target O(1) and transfer across body
shapes (D5, D12).

## Physics objective — RNEA root-wrench residual

Over a T-frame clip the frozen model yields per-frame MHR body + camera params.
The step-03 adapter (`contact/physics/adapter.py::MHRAdapter`) maps them onto a
BetterHuman **MHR** body (floating-base, `nq=132`) and a world-frame configuration
trajectory `q`. The step-06 loss (`contact/physics/loss.py::PhysicsLoss`):

1. smooths `q` **on the manifold** (windowed mean for the translation + 125
   revolute channels, hemisphere-aligned slerp mean for the root quaternion);
2. finite-differences to velocity and acceleration via `Model.difference`
   (SE3 log for the free-flyer block), honouring the per-interval `dt` from
   `frame_pos_sec`;
3. places the predicted forces as external wrenches `fext` at the four extremity
   joint origins and runs **RNEA** (inverse dynamics):
   `tau = M(q)·a + b(q,v) + g(q) − Jᵀ·fext`;
4. minimises the **6D residual wrench at the free-flyer root** `tau[..., :6]` — an
   unactuated joint whose generalised force must be zero for physically consistent
   motion. `tau[..., :3]` is the residual force, `tau[..., 3:6]` the residual
   torque; `tau[..., 6:]` are the joint torques (regularised, not driven to zero).

Every term is dimensionless (D12): residual force is normalised by `m·g`,
residual/joint torques by `m·g·1 m`. RNEA needs no mass-matrix inverse, so MHR's
singular CRBA (mimic overcompleteness + zero-mass cosmetic bodies) is a non-issue;
forward dynamics is off the table but not needed.

### Supervision terms (`physics.loss.*`)

- `residual` — the root-wrench residual above (the training objective). By default
  (`physics.loss.residual_robust.kind: square`) it is `‖r_f‖² + ‖r_τ‖²`, bit-for-bit
  the original. Set `kind: pseudo_huber` to apply a **component-wise** pseudo-Huber
  `ρ_δ(x) = δ²(√(1+(x/δ)²) − 1)` to the three residual-force and three residual-torque
  components (`delta_force`, `delta_torque`, both dimensionless): quadratic near zero,
  linear past `δ`, taming the heavy-tailed supervision from noisy double
  finite-differencing (batch residual p99 ≫ median). The **evaluation headline and the
  `{split}/physics_residual` monitor read the RAW, un-robustified residual**
  (`raw_residual`, emitted on every RNEA batch regardless of `kind`) so runs stay
  comparable across robustifiers. **Comparable only under the same clip protocol**,
  though: `dt`, the smoothing stencil, and which rows are scored all change with
  `frames_per_clip`/`frame_stride` — the historical numbers (zero-force baseline
  ≈ 2.586; collapsed run best ≈ 1.60) were measured at T=8/stride 1 and must be
  re-measured under a new protocol (e.g. T=16/stride 2) before being used as
  baselines. `residual_sat_frac` (fraction of residual components past `δ`; 0 under
  `square`) is logged to watch the robust tail. When the split yields **zero
  residual mass** the headline is `NaN`, and the trainer raises if
  `physics_residual` is the monitor.
- `force_noncontact` — `(1−p)·‖f‖²`, a strong penalty on force at **non-contact**
  extremities. Without it the residual is trivially zeroable by fictitious forces
  on airborne limbs. The gate `p` is the **detached** predicted contact prob (D8):
  physics must never train the contact head, and detaching stops the force loss
  from inflating contact probs to license fictitious forces. GT labels are *not*
  used as the gate — the video labels are motion-gated *stable* contact, a
  different quantity from instantaneous load (R8).
- `force_at_contact` — `p·relu(contact_min_bw − ‖f‖)²`, a weak opposite penalty
  discouraging near-zero force at contact extremities. **Caveat:** the force head is
  zero-init and this term has **zero gradient at `f = 0`** (the magnitude uses the
  `√(‖f‖²+ε)` softening, so `∂‖f‖/∂f → 0` as `f → 0`), so it cannot by itself lift
  the head off zero — it only shapes an already-nonzero prediction. The decisive
  `climbing_videos_force_warmstart_t16` config therefore sets it to `0.0`.
- `force_smooth`, `force_l2`, `torque_l2`, `torque_smooth` — smoothness / L2
  regularisers on world-frame forces and RNEA joint torques.

### Gradient isolation is regime-dependent

The physics loss consumes the frozen MHR pose, camera extrinsics and contact probs
**detached**, so its gradients reach only the **force** params — *in regime (a)*
(`train.freeze_contact`, contact frozen). In **regime (b)** (contact trainable
alongside physics) the isolation is **incomplete**: force tokens attend the contact
tokens, and the vendored attention mask permits that direction, so physics gradients
reach the trainable contact head through force→contact attention. The trainer prints
a loud warning in this case. A detach-fix lives in the vendored attention and is
deferred; the shipped force runs all use regime (a), where the isolation is exact.

## Force frame (`model.force_head.frame`)

The head predicts in a **human-centric** frame that must be learnable from a
single crop — never the reconstruction world (its yaw is arbitrary per scene). Two
options (D6), converted to RNEA's per-joint LOCAL `fext` in `PhysicsLoss._pred_to_world`:

- `local_world_aligned` (**default**) — the per-frame **camera y-up axes** (the
  un-flipped camera frame: what a level camera calls up/right/forward). Convert:
  `f_world = R_w←c · D · f_pred` with `D = diag(1, −1, −1)` (camera y-down ↔ native
  y-up), then rotate into the joint LOCAL frame.
- `local` — the extremity joint frame; passes straight through to `fext`.

Forces act at the wrist/ankle **joint origins** with zero torque component; offset
contact points (palm/sole, grip moments → `r×f`) are out of scope v1 (R6).

## World and gravity conventions (moving cameras)

ClimbingVideos cameras move, so the static-camera assumption is invalid: camera
motion would alias into body acceleration and corrupt the residual. The dataset
therefore carries **per-frame camera extrinsics** (step 02), exported from the
BetterVideoReconstruction pipeline's fuse+scale stages:

- `cam_from_world [B, 4, 4]` — camera-from-world, **OpenCV** convention, metric
  after the scale stage. Provenance: `features/geometry/.../transform.npz`
  (`extrinsics`, cumulative `scale` applied to translations). Frame 0 ≈ identity, so
  the physics **world is the metric camera-0 reconstruction frame**. The adapter
  composes SAM's camera-frame root pose with `T_w←c = cam_from_world⁻¹` (plus the
  `D` un-flip) so every frame's body lands in one static world per scene (D7).
- `gravity_world [B, 3]` — the **first camera's +y axis mapped to world** (OpenCV
  +y points down, so this is the downward unit vector; empirically ≈ `[0, 1, 0]`).
  RNEA gravity is set per clip as `physics.gravity · gravity_world` via
  `values.gravity` (differentiable, no global mutation). "Up" in the plausibility
  metrics is `−gravity_world`.
- `cam_valid [B]` — still images and stale exports carry `cam_valid=False` and are
  physics-ineligible; an otherwise-eligible video clip lacking cameras **raises**
  (a stale export must never become a silent no-op, D13).

Cameras are a **dataset input, not a model input** (D13): the model stays
deployable on plain images/video; the "someone provides extrinsics" contract lives
only in the dataset schema, and flows batch → physics loss.

### Camera-jerk clip filtering

Camera-motion compensation is only as good as the reconstruction. Where the
extrinsics jump discontinuously the correction is wrong and the error feeds body
acceleration through double finite differencing. The dataset therefore emits a
per-clip-row `cam_jump_m` — the metric distance between the camera centres
`C = −Rᵀt` of **consecutive sampled clip frames** (row 0 of each clip = 0) — and
`PhysicsLoss` drops any clip whose `cam_jump_m.max()` exceeds
`physics.max_cam_jump_m` (`null` = off). The **sampled-step** definition is
deliberate: with `frame_stride > 1` a discontinuity on a *skipped* source frame is
invisible to per-source-frame jumps (a real-data audit found 12 of 18 stride-2
displacements over 0.5 m missed that way — including the 7.85 m jump below for
even-parity clips), while the net sampled-step displacement bounds any intra-gap
jump from below; the threshold therefore applies to the per-sampled-step
displacement. This filters on **upstream camera evidence, not the model
residual**, so it cannot be gamed by the force branch. Excluded clips are counted
(`n_jerk_excluded_clips`); unlike the missing-camera guard (which raises), a jerk
exclusion is silent — but if it excludes *everything*, the eval headline becomes
`NaN` and the trainer **raises** when `physics_residual` is the monitor (zero data
must never read as a perfect residual). *War story:* train scene
`45KmZUc0CzA_0007` has a 7.85 m jump at source frames 262→263, and other scenes
have `> 0.5 m` discontinuities; `0.5` is the shipped threshold in the decisive
config.

## Residual-frame stencil — a frame-count pitfall

A frame contributes to the residual only when its full stencil fits inside the
clip: smoothing radius `r = len(smoothing_kernel)//2` per side, plus two frames for
the doubled central difference (velocity `±1`, acceleration of velocity `±1`). The
residual frame indices are `{t : 2 + r ≤ t ≤ T − 3 − r}` (`_residual_frame_indices`).

The default kernel `[0.25, 0.5, 0.25]` (`r = 1`) needs **`T ≥ 7` for any residual
frame at all** — with a smaller `T` the residual objective is silently dead. The
`T = 8` force configs give exactly **2 residual frames** (`{3, 4}`); the decisive
`T = 16, stride 2` config gives **10** (`{3..12}`) — more supervision rows and, via
the doubled `dt`, ~4× less finite-difference noise. Set `frames_per_clip` with this
in mind.

**Non-uniform-`dt` limitation:** `_trajectory_derivatives` forms the acceleration by
central-differencing the velocity with the *doubled-interval* `dt`, which is exact
only for uniform frame spacing. ClimbingVideos frames are uniformly spaced (constant
`stride`), so this is currently harmless; a rewrite is deferred (documented, not
fixed).

## Evaluation & demo

- `scripts/evaluate.py` adds physics-consistency metrics when the run enables the
  force branch + the RNEA loss (contact metrics unchanged): the headline
  `physics_residual` (the RAW residual, same as the training monitor), the
  vertical-force-sum distribution (≈1 body weight for quasi-static climbing), the two
  gate-violation rates (mean ‖f‖ on predicted non-contact frames; fraction of
  predicted contact frames with ‖f‖ < `contact_min_bw`), and per-extremity force
  magnitudes (body weight + newton) split by predicted contact state. Lacking a
  trained force checkpoint, run `--warm-start` to exercise the pipeline on the
  zero-init head.
- **Affine input-dependence baselines** (the decisive collapse test). The root
  wrench is *affine* in the head-frame forces — per residual frame
  `r(f) = r0 + B·vec(f)`, `B ∈ ℝ^{6×12}`, obtained from one zero-force and twelve
  unit-force no-grad RNEA calls per batch (`PhysicsLoss.affine_residual`). Streaming
  `r0`, `B` and the predicted forces over the split, the evaluator reports the raw
  residual (mean + median/p90/p99/max) of **(a)** zero forces, **(b)** the best fitted
  constant 12-DoF force (closed-form least squares on the accumulated normal
  equations), **(c)** the network, and **(d)** shuffled per-clip force trajectories
  (5 permutations, mean±std), plus head-frame per-limb/component across-frame std and
  Pearson `corr(‖f‖, prob)`. **An input-dependent model must beat BOTH (b) and (d)**;
  the printout states `PASS`/`FAIL`. A collapsed constant solution fails by
  construction (it ties (b) and is unaffected by (d)). `train/physics/force_std`
  (across-clip std of mean ‖f‖) is the cheap online counterpart.
- `scripts/demo_climbing_videos.py` draws per-extremity force arrows on the
  predicted-pose panel (anchor = the extremity's 2D keypoint, direction = the 3D
  force projected through the model's intrinsics, length ∝ magnitude in body
  weights, colour by extremity) — behind an `out["force"]` presence check, for the
  `local_world_aligned` frame.

## Assumptions & risks (from `plan/README.md` §5)

- **R2 — Gravity direction = first camera's +y.** Assumes the camera is level at
  scene start; per-scene constant thereafter (camera *motion* is compensated via
  extrinsics, initial *tilt* is not). Wrong tilt biases the residual. `gravity_world`
  makes the assumption data-visible and swappable (future: pipeline-side gravity
  estimation; `values.gravity` is differentiable). Not v1.
- **R3 — Extrinsics quality.** Camera motion is compensated exactly where the
  reconstruction is right; the residual risk shifts to VGGT drift and the body-ruler
  scale estimate — errors there feed accelerations through double finite
  differencing. **Monitor per-scene residuals; drop pathological scenes if needed.**
  *War story:* train scene `45KmZUc0CzA_0007` has a **7.85 m/frame camera-center
  jump** (at frame 262 of 329) — a reconstruction quirk; candidates like it may
  deserve exclusion if their per-scene residuals are pathological.
- **R4 — Metric scale consistency.** SAM's `pred_cam_t` (from intrinsics) and the
  pipeline's world scale (body-ruler, exported as `cam_scale`) must agree; both are
  body-anchored so gross mismatch is unlikely. Disagreement shows up as spurious
  root acceleration; the step-03 FK acceptance test bounds the per-frame part. Mass
  tracks predicted shape (`with_shape` recomputes body inertias from the shaped mesh).
- **R5 — Uniform density 1000 kg/m³**, LOD1 mass ≈ 81.5 kg neutral. Fine for v1.
- **R6 — Forces applied at wrist/ankle joint origins.** Real contact points are
  offset (sole, palm, grip moments) → unmodeled `r×f` torque error. Accepted v1.
- **R7 — Perf.** MHR FK is kernel-launch-bound (~203 BR joints); physics adds
  FK+RNEA fwd+bwd per step. *War story / accepted cost:* the adapter runs
  `from_classic` **twice per call** (once for the clip's centre-frame shaped body,
  once for the full-batch `q`) — a known, deliberately un-optimised cost. Escape
  hatches (`use_warp`, fewer physics frames per batch) exist; do not pre-optimise.
- **R8 — Label semantics.** Motion-gated "stable contact" labels ≠ instantaneous
  load; gating the force loss by predicted probs (D8) inherits this bias. Accepted,
  documented.
- **Temporal signature omits `dropout` (deliberate).** The checkpoint arch
  signature's `temporal`/`force_temporal` blocks do not record `dropout`
  (`contact/checkpoint.py::_arch_signature`): every shipped artifact uses `0.0`,
  and adding the key now would spuriously mismatch existing checkpoints whose
  stored signatures lack it. To be absorbed at the next signature-version bump.

## Pointers

- Code: `contact/physics/adapter.py` (SAM↔BetterHuman bridge),
  `contact/physics/loss.py` (objective + `diagnostics` for eval),
  `sam_3d_body/models/heads/force_head.py`.
- Configs: `configs/climbing_videos_force_warmstart.yaml` (regime a: warm-start,
  contact frozen, force-only), `configs/climbing_videos_force_scratch.yaml`
  (regime b: contact + force jointly), `_temporal` variants; defaults in
  `configs/base.yaml` (`physics:`, `model.force_head`, `model.force_temporal`).
- Design record: `plan/README.md` (decisions D1–D13, risks R1–R9), `plan/for_agents/`.
