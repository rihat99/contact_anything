# The pose→motion consistency loss — deep dive

This page explains one loss, `contact/motion_consistency.py`, in full: what it computes, the
geometry it relies on, every one of its six terms, the null-space failures that shaped it, and the
open problems it still has. It exists as its own page because this loss is the project's current
working frontier — [losses.md](losses.md#motion-consistency-loss) carries the short version, and
the experiment chronology (with all run numbers) is in
[experiments.md](experiments.md#10-the-motion-consistency-saga-2026-08-25-to-26).

Status, stated up front: across three training attempts (v2, v3, v4) the loss has **never beaten
the plain allmod baseline** on contact, force or motion metrics (the comparison is not perfectly
controlled — all three consistency configs also raise the non-contact force penalty to 0.2 where
allmod has 0.0, and v2 additionally switched the decoder mask to `mutual`). Each attempt failed
through a different unpinned degree of freedom, and each failure is understood. The loss remains interesting
for one reason no other objective covers: it is the only training signal in the project that
touches the **quality of the world-space pose trajectory** — where the body actually is over time,
as opposed to per-frame joint angles.

## Why this loss exists

The model has two ways of talking about motion, and nothing forcing them to agree:

1. The **pose path** predicts, per frame, a body configuration plus a camera translation
   (`pred_cam_t`) and a global orientation (`global_rot`). String the frames together and you get
   an implied trajectory of the body through the world.
2. The **motion head** directly predicts the pelvis' velocity and acceleration per frame,
   supervised by kindyn targets (`motion_supervision`).

The per-frame pose losses cannot see trajectory quality at all. Pose supervision compares local
joint angles in q-space and deliberately never touches the root; the frozen model's own training
was single-image. The result is **depth wobble**: frame-to-frame jitter in the predicted camera
depth that reprojects almost perfectly in 2-D (it is nearly invisible to any image-space loss) but,
once differentiated, turns into physically absurd accelerations. The motion-probe work measured the
consequence: differentiating the frozen model's raw world trajectory gives acceleration RMS several
times the true motion's.

The consistency loss closes the loop: **lift the predicted pose into the metric world,
differentiate it with the same body-twist stencil the motion targets use, and require the result
to agree with the kindyn ground truth and with the motion head.** If it worked, the pose path
would inherit a temporal smoothness prior with physical units, and pose and motion would stop being allowed to
contradict each other.

It is a pure loss — no parameters, nothing in the architecture signature. Enabling it changes
gradients, not the model.

![Dataflow of the consistency loss: predicted pose lifted to the world, differentiated, and
compared against kindyn GT, the detached motion head, and the frozen model's own
predictions](figures/consistency_dataflow.png)

## Step 1 — lift the predicted pose to the world

Three predicted quantities enter, all taken from the final MHR output. When a brick-level pose
write path is active (`pose_temporal`, or `pose` listed in a cross-modal brick) that is the
**recomputed** output — rebuilt from the pose token after the bricks have written it — and when
only `train.finetune_pose_head` is on it is the ordinary head output; either way all three are
live functions of the trainable pose path:

- `pred_keypoints_3d` — the 70 MHR keypoints in camera axes; the loss uses the mean of indices
  9 and 10 (left and right hip) as "the pelvis",
- `pred_cam_t` — the predicted camera-space translation of the body,
- `global_rot` — the body's global orientation as an euler triple in the model's native axes.

The dataset supplies per-frame camera extrinsics `cam_from_world` (OpenCV convention, metric
scale, estimated by the BVR reconstruction pipeline; see
[data.md](data.md) for the coordinate conventions). The lift is:

```
p_world            = R_ext^T · ( mean(kp3d[9], kp3d[10]) + pred_cam_t − t_ext )
R_world_from_root  = R_ext^T · diag(1, −1, −1) · euler_xyz(global_rot)
```

Two details that are easy to get wrong, both verified:

- The keypoints leave the MHR head **already in camera axes**, while `global_rot` is expressed in
  the model's **native** axes — hence the `diag(1, −1, −1)` flip (a proper rotation, determinant
  +1) sitting only on the rotation path. The same flip lives in the physics adapter.
- The whole composition was checked against a stored geometry artifact
  (`output/motion_probe_geom`): maximum deviation **2.4 × 10⁻⁷ m** from the independently computed
  reference trajectory.

The lift runs in float64 (the tensors are tiny — one point and one matrix per frame) and stays on
the autograd graph end to end.

![The world lift and the twist stencil: the camera-frame pelvis carried into the y-down metric
world by the extrinsics, and the three-frame finite-difference
footprint](figures/consistency_world_lift.png)

### The frozen stash — where the rails get their anchor

The recompute hook in the vendored model (`sam_3d_body/models/meta_arch/sam3d_body.py`) does one
extra thing when a pose write path exists: before overwriting the final outputs with the
recomputed ones, it stashes the **pre-write** predictions, detached, as
`out["mhr"]["pred_cam_t_frozen"]` and `out["mhr"]["global_rot_frozen"]`. These are "what the model
would have said had the trainable bricks not touched the pose token", and they are what the two
rail terms anchor to. The stash exists only when a **brick-level** pose write runs — the recompute
hook fires for `pose_temporal` and for the cross-modal bricks listing `pose`, and nowhere else.
Without one, the rails silently deactivate. For a config with no trainable pose path at all that
is correct (nothing can drift). But `train.finetune_pose_head` **alone** also counts as a valid
pose path, passes config validation, and triggers no recompute: in that configuration the pose
outputs can drift through `head_pose.proj` while both rails are structurally absent. Nothing
currently rejects that combination — treat it as a known footgun.

One caveat that turned out to matter enormously (see the v4 failure below): "pre-write" is not the
same as "fully frozen" when `train.finetune_pose_head` is on. The stash is computed with the
*currently fine-tuned* pose-head projection, so if that head drifts, the anchor drifts with it.

## Step 2 — differentiate: the BVR body twist

The lifted world trajectory is differentiated with the **same body-twist stencil the motion
targets were built with**. Per clip, with frame interval `Δt` (measured from `frame_pos_sec`, the
frames' real timestamps):

```
d[t]  =  se3_log( T_t⁻¹ · T_{t+1} )              # relative pose, in the body frame at t
v[t]  =  ( d[t−1] + d[t] ) / 2Δt                 # central velocity
a[t]  =  ( d[t] − d[t−1] ) / Δt²                 # central acceleration
```

`se3_log` maps a relative rigid transform to a 6-vector (3 linear + 3 angular, with the proper
`V⁻¹` coupling between them). Because the relative pose is taken in the body frame at `t`, the
result is a **body twist** — velocities expressed in the climber's own frame, invariant to where
the world origin is.

The lie-group helpers `so3_log_xyzw` and `se3_log` are torch mirrors of the float64 numpy code
the data loader uses to derive the targets, and parity tests assert both paths agree to float32
tolerance on random trajectories (`quat_xyzw_from_matrix`, which has no loader counterpart, is
tested as the exact inverse of the loader's quaternion-to-matrix map). This is not cosmetic: a
subtly different log map or hemisphere convention would put a permanent bias between prediction
and target.

Shared stencil does **not** mean identical preprocessing, and the difference matters. The targets
are derived from the **σ = 0.12 s Gaussian-smoothed** root trajectory at kindyn's **native frame
rate**; the prediction is differentiated **raw** at the clip's **sampled** `Δt`. The smoothing
half is deliberate — the prediction's frame-to-frame wobble is penalized against clean targets,
which is the entire point — but the rate half is an open subtlety: with `frame_stride` above 1
(including `auto`), the prediction's finite differences are taken at a multiple of the target's
`Δt`, and nothing validates that combination for this loss.

The stencil at `t` needs frames `t−1, t, t+1`, all real and extrinsics-valid. Consequences:

- clip **boundary frames are never twist-supervised** (they lack a neighbor),
- `frames_per_clip ≥ 3` is a hard config requirement,
- still images (`T = 1`) make the whole loss inert — every term returns zero mass.

## Step 3 — standardize and compare

Raw twists mix units (m/s, m/s², rad/s, rad/s²) and scales. Before comparison, the pose-derived
pelvis twist is standardized with the **same fixed mean/std table** `motion_supervision` uses for
the motion head's pelvis targets, and each 3-component group (velocity, acceleration, and the
angular pair — included only when **both** `motion_supervision.angular` and
`motion_consistency.angular` are on) is weighted with the **same group weights** as the motion
loss. The
three pelvis objectives in the system — motion head vs GT, pose-derived vs GT, pose-derived vs
head — therefore weight vel/acc identically and operate in the same normalized units.

The four comparison terms (`gt`, `head`, `pos`, `rot`) are Huber-style smooth-L1 (transition
`huber_delta` = 1.0 in standardized units for the twist terms; 0.1 m and 0.1 rad for the absolute
anchors); the two rails are plain hinges with no smooth zone.

## The six terms

| term | compares | against | rows supervised | shipped weight (v3 / v4) |
|---|---|---|---|---|
| `gt` | pose-derived pelvis twist | kindyn GT twist | stencil-valid interior rows, outlier-filtered | 1.0 / 1.0 |
| `head` | pose-derived pelvis twist | motion head output, **detached** | stencil-valid interior rows | 0.5 / 0.5 |
| `pos` | lifted world pelvis `p_world` | kindyn root + hip offset | every valid frame, boundaries included | 5.0 / 5.0 |
| `rot` | lifted world orientation | kindyn root rotation | every valid frame, boundaries included | 2.0 / 2.0 |
| `cam_rail` | `pred_cam_t` | the model's own frozen `pred_cam_t` | every real frame | 10.0 / 10.0 |
| `rot_rail` | `global_rot` | the model's own frozen `global_rot` | every real frame | — / 10.0 |

### `gt` — the reason the loss exists

The pose-derived standardized twist versus the kindyn GT twist. This is the term that actually
attacks depth wobble: jitter that image-space losses cannot see becomes a large, directly
penalized acceleration error here. The gt term respects the same per-joint outlier bit the motion
loss uses (kindyn's heavy-tailed frames are excluded during training, never during evaluation).

### `head` — and why the detach is the entire design

The pose-derived twist versus the motion head's own pelvis prediction — with the head's output
**detached**. Gradient flows only into the pose path: the pose trajectory is pulled toward the
motion estimate, never the reverse. The motion head keeps learning from `motion_supervision`
alone.

The first version (v2) did not detach, and it demonstrated exactly why this matters: when the pose
trajectory collapsed (below), the bidirectional term dragged the motion head down with it —
velocity correlation fell from 0.53 to 0.21. A consistency constraint between a healthy branch and
a collapsing one, applied symmetrically, converges on the collapse.

### `pos` and `rot` — absolute anchors

Derivative terms cannot see constants (the next section is entirely about this), so two per-frame
absolute terms pin the world pose itself:

- `pos`: Huber (0.1 m) between the lifted mean-hips point and the **smoothed** kindyn root
  position (the same σ = 0.12 s smoothed trajectory the twist targets come from) carried to
  the same anatomical point — `p_gt + R_gt · hip_offset_root`, where `hip_offset_root =
  [−0.009, −0.060, −0.065]` m (norm ≈ 9 cm) is the constant mean-hips-minus-root-origin offset
  measured over 363 scenes of the motion-probe artifact. Without the offset the term would demand
  a 9 cm bias.
- `rot`: Huber (0.1 rad) on the geodesic residual `so3_log(R_pred^T · R_gt)` against zero. The
  probe showed the frozen model has **no constant orientation offset** from kindyn (~1.4°) and a
  ~7° per-frame error, so this term starts near its target and genuinely fights per-frame error
  rather than some calibration constant.

Both are per-frame — no stencil — so clip boundary frames *are* supervised here, and both require
the frame to be real, extrinsics-valid, and kindyn-covered (`motion_root_valid`).

### `cam_rail` and `rot_rail` — trust regions, not targets

```
cam_rail  =  relu( ‖pred_cam_t − pred_cam_t_frozen‖    − 0.5 m )
rot_rail  =  relu( geodesic(global_rot, global_rot_frozen) − 0.2 rad )
```

A rail is **exactly zero** while the prediction stays within the margin of what the frozen model
itself would predict, and grows linearly beyond. For a healthy model the rails contribute nothing
— they are not an opinion about where the camera should be; they are a wall around "how far from
the frozen model's answer a trainable brick is allowed to move it". The rails need no extrinsics
(both comparisons live in camera/native axes; the extrinsics cancel out of the geodesic), so they
stay active even on frames whose camera estimate is invalid.

## Null spaces — why terms 3–6 exist at all

This is the transferable lesson of the whole effort, so it gets its own section. A derivative-only
objective is blind to constants: any direction in which the prediction can move *uniformly across
the clip* changes no velocity and no acceleration, and is therefore a **null space** — and if
moving in that direction happens to reduce the noise the objective penalizes, the optimizer will
take it. Each version of this loss pinned the null space it knew about, and the optimization
pressure relocated to the next one:

- **v2 — the camera.** Only the two derivative terms `gt` + `head`, no anchors. A *constant*
  `pred_cam_t` contributes nothing to world velocity under a mostly-static camera, and a constant
  depth kills depth wobble by killing depth. The camera collapsed to a constant 9 cm from the
  camera (sane value: ~5.5 m); reprojected keypoints landed ~12,000 px off the person; the
  un-detached head term pulled the motion head down with it.
- **v3 — the orientation.** Anchors (`pos`, `rot`), detach, and `cam_rail` fixed the camera
  cleanly. The pressure moved to the one channel still unpinned: the **angular** twist residuals
  are the frozen model's ~7°-per-frame orientation wobble, differenced at 30 fps and divided by a
  small GT angular std — enormous numbers whose cheapest minimizer is a *constant orientation*.
  The world orientation froze (per-clip spread 0.21° vs GT's 2.85°) and parked ~55° from the
  truth; the `rot` anchor at weight 2 was roughly 10× too weak to stop it.
- **v4 — the anchor itself.** Two changes: `angular: false` removed the noise source (the twist
  comparison keeps only linear rows; the motion head keeps its full 12-dim supervision), and
  `rot_rail` walled the orientation. Both worked as designed — and the leak moved *into the
  mechanism that defines the rails*. With `train.finetune_pose_head` on, the "frozen" stash is
  computed by the currently fine-tuned head, so when the head drifts, **it drags the anchor along
  with the offender** and the rails structurally cannot see the drift. Orientation error vs GT
  climbed 6.7° → 28° in two epochs while the rail read a nearly-inert ~2.8°.

![The collapse saga: derivative supervision rewards a constant pose; v2 escaped through the
camera, v3 through the orientation, v4 through the head that anchors its own
rails](figures/consistency_null_spaces.png)

Two mechanism-level conclusions worth keeping:

1. **Anchoring to your own model is only safe while that part of the model is actually frozen.**
   A rail anchored to a quantity the optimizer can move is a rail bolted to the thing it is
   supposed to restrain.
2. **Detaching stops gradients, not features.** v3's motion degradation happened *despite* the
   detach: the modalities share the cross-modal and frame-attention bricks, so a corrupted pose
   token contaminates the activations every other head reads. Gradient isolation and feature
   isolation are different properties.

## Which rows supervise what

A single clip of `T` frames, per term:

| term | needs stencil (t−1, t, t+1) | needs valid extrinsics | needs kindyn coverage | outlier-filtered | boundaries |
|---|---|---|---|---|---|
| `gt` | yes | yes | yes (`motion_valid`) | yes (train only) | never |
| `head` | yes | yes | no | no | never |
| `pos`, `rot` | no | yes | yes (`motion_root_valid`) | no | supervised |
| `cam_rail`, `rot_rail` | no | no | no | no | supervised |

Note the deliberate asymmetry between `gt` and `head`: the head term ignores the outlier bit and
kindyn coverage, because it compares the model with itself — its only requirement is stencil
support (a real, extrinsics-valid frame triple).

Every term reports `(weighted_numerator, mass)` in the same contract as the other loss modules, so
the DDP reduction is the exact global mean (see
[losses.md](losses.md#the-ddp-contract-and-why-exactness-matters)); a term with no eligible rows
this batch contributes a graph-connected zero rather than disappearing, which keeps multi-GPU
ranks in lockstep.

## Gradient reach — test-enforced

All six terms send gradient to the **pose write paths only**: `pose_temporal`, the pose-designated
parts of the cross-modal bricks, and (when enabled) the fine-tuned `head_pose.proj`. The motion
head receives nothing (its output enters detached); the frozen base receives nothing; kindyn data
and extrinsics are plain tensors.

`tests/test_motion_consistency.py` pins all of this: the config accept/reject matrix, parity of
the torch lie helpers with the loader's target derivation, the world-lift composition under
identity extrinsics (the extrinsics half of the lift was verified once against the probe artifact,
not in a test), boundary masking, the outlier asymmetry between `gt` and `head`, short-clip
inertness, and — in the `-m slow` variant, on the real checkpoint — that gradients land on the
pose path and nowhere else.

## Reading the diagnostics

Training logs, every `logging.log_freq` steps under `train/consistency/*`, quantities designed to
make the failure modes above visible early (evaluation logs only the per-term losses, not these
diagnostics):

| key | meaning | healthy | collapse signature |
|---|---|---|---|
| `vel_rmse`, `acc_rmse` | pose-derived vs GT pelvis twist, physical units (m/s, m/s²) | drifting down | `acc_rmse` dropping *fast* while anchors rise = smoothing by constancy |
| `pos_err_m` | mean world position error vs the anchored GT point | ~0.17 m at init | growing = translation escape (v4 late) |
| `rot_err_deg` | mean geodesic orientation error vs GT | ~7° at init | 20°+ and climbing = orientation escape (v3: ~55°) |
| `cam_dev_m` / `rail_frac` | mean camera deviation from the stash / fraction of frames beyond the margin | ≤ ~0.35 m / ~0 | `rail_frac` ≫ 0 = the wall is load-bearing |
| `rot_dev_deg` / `rot_rail_frac` | same for orientation | a few degrees / ~0 | **beware**: stays small under head drift (v4) because the stash drifts too |
| `gt_mass`, `head_mass`, `pos_mass`, … | per-term supervision masses | stable | near-zero = data/masking problem, not learning |

The one trap: the rail diagnostics measure deviation **from the stash**, not from the truth. Under
`finetune_pose_head` the stash itself can drift, so a serene `rot_dev_deg` next to a climbing
`rot_err_deg` is not a contradiction — it is precisely the v4 signature.

## Configuration reference

The block below is the **shipped v4 configuration** (`climbing_corpus_allmod_consistency_v4.yaml`)
— not the base defaults. The base defaults are `angular: true` and **all four** `pos` / `rot` /
`cam_rail` / `rot_rail` weights at **0.0**: writing `motion_consistency: {enabled: true}` and
nothing else reproduces the anchorless v2 recipe, with every null space on this page open. Set the
anchors and rails explicitly.

```yaml
motion_consistency:
  enabled: true
  angular: false            # include angular twist rows in gt/head (needs
                            #   motion_supervision.angular too; v4 stance: false —
                            #   the angular residuals reward a constant orientation)
  hip_offset_root: [-0.009, -0.060, -0.065]   # mean-hips − root origin, metres (measured)
  loss:
    gt: 1.0                 # pose-derived twist vs kindyn GT
    head: 0.5               # vs the DETACHED motion head
    huber_delta: 1.0        # twist Huber transition, standardized units
    pos: 5.0                # absolute world root position anchor
    pos_huber_m: 0.1        #   its Huber transition, metres
    rot: 2.0                # absolute root orientation anchor
    rot_huber_rad: 0.1      #   its Huber transition, radians
    cam_rail: 10.0          # camera trust region weight
    cam_rail_margin_m: 0.5  #   zero inside this margin
    rot_rail: 10.0          # orientation trust region weight (v4 addition)
    rot_rail_margin_rad: 0.2
```

Hard requirements enforced at config load: `motion_supervision.enabled` (the loss standardizes
with its table and compares against its targets and head), `pelvis` among the motion joints, a
trainable pose write path (otherwise nothing can receive the gradient), `frames_per_clip ≥ 3`, and
at least one nonzero term weight. The loss adds nothing to the architecture signature — a
checkpoint trained with it loads anywhere the architecture matches.

## Status and open problems

The score so far is 0-for-3 against the allmod baseline (with the caveat that the comparisons
also differ in the non-contact force weight, and v2 in the decoder mask), each defeat teaching a
specific lesson: camera null space → orientation null space → self-anchoring under a trainable
head. The camera- and orientation-level defenses now hold. Two fronts remain open: the interaction
with `train.finetune_pose_head`, and translation — v4's linear-only twist re-routed the remaining
pressure into the position channel (its `pos` term at epoch 2: 3.26 vs v3's 2.60; motion velocity
`r3d` 0.171 vs v3's 0.225). Candidate next steps, in rough order of confidence:

1. **Freeze the pose head again** (drop `finetune_pose_head`). The pose can then move only through
   the zero-gated bricks, every rail anchor is computed by genuinely frozen parameters, and the v4
   leak is closed by construction. Cost: the pose path loses its highest-bandwidth write channel.
2. **Anchor the rails to truly frozen predictions.** Keep the fine-tuned head, but compute the
   stash with the *original* head weights (a detached copy of `head_pose.proj` from the
   checkpoint). Same idea as 1 restricted to the anchor computation — the head stays trainable but
   can no longer move its own wall.
3. **Raise the ground-truth anchors** (`pos`, `rot`) until fighting them costs more than the
   wobble pays. Blunter: it turns the consistency loss into partial world-pose supervision, and
   the right weights are an empirical search.
4. **A world-trajectory evaluation metric.** Today the loss is judged by contact/force/motion
   metrics it was never primarily aimed at; the quantity it actually targets — world-trajectory
   pose quality (something like world-frame pelvis error and acceleration RMS vs kindyn on the
   test scenes) — is not a first-class evaluation number. Until it is, even a working version
   could look like a tie.

Options 1 and 3 were named in the v4 post-mortem; 2 and 4 follow directly from the mechanisms
documented above.

## Where to read next

- [experiments.md §10](experiments.md#10-the-motion-consistency-saga-2026-08-25-to-26) — the full
  v2/v3/v4 chronology with every number behind the summaries here.
- [losses.md](losses.md) — how this loss composes with the other five objectives and the DDP
  reduction contract.
- [architecture.md](architecture.md) — the pose write paths this loss trains, and the recompute
  hook that makes `pred_cam_t` a live function of them.
- [data.md](data.md) — the coordinate frames, the extrinsics, and how the kindyn targets are made.
