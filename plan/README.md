# Forces Extension — Implementation Plan

Extend contact_anything with **3D contact-force prediction** at the four climbing extremities,
supervised by **physics (RNEA root-wrench residual)** instead of labels. This file is the
architecture + decisions record; `for_agents/NN_*.md` are self-contained briefs, one per
implementation step. Agents: read this file first, then your brief.

Status: **all steps (01–08) implemented 2026-07-22** (uncommitted working tree). Plan written +
adversarially reviewed against all three codebases; revised same day for **moving cameras**:
per-frame metric extrinsics from the reconstruction pipeline become a dataset input (step 02);
the physics world is the reconstruction world, not the camera frame. Briefs renumbered 01–08.
Force branch, physics loss, training regimes, eval metrics, demo arrows, and docs (`docs/forces.md`)
are in place; no trained force checkpoint exists yet (pipeline exercised warm-started, numbers
pending a training run). This file remains the design record.

---

## 1. Goal

- Four **force tokens** (keypoint-anchored, mirroring contact tokens), an optional **force
  temporal block**, and a **force head** regressing one 3D vector per extremity:
  `out["force"]["joint_forces"] [B, 4, 3]`, order `left_hand, right_hand, left_foot,
  right_foot` (same as the contact joint target).
- No GT forces. Supervision: over a T-frame clip (T≥5), take the frozen model's per-frame MHR
  body + camera outputs, smooth them, finite-difference to velocity/acceleration on the
  manifold, run **RNEA with the predicted forces as external wrenches**, and minimize the
  **6D residual wrench at the free-flyer root** (an unactuated joint — its generalized force
  must be zero for physically consistent motion). Per-frame **camera extrinsics provided by
  the dataset** (exported from the video-reconstruction pipeline, step 02) place every
  frame's body in one static metric world, so camera motion never aliases into body
  acceleration.
- Contact predictions gate forces: strong penalty on force at non-contact extremities
  (otherwise the residual is trivially zeroable by fictitious forces on airborne limbs),
  weak opposite penalty on near-zero force at contact extremities, plus smoothness/L2
  regularizers on forces and RNEA joint torques.
- Two training regimes: **(a) warm-start** from a contact checkpoint with contact params
  frozen, train force branch only; **(b) from scratch**, contact from labels + forces from
  physics jointly.

## 2. What exists (verified 2026-07-22; line numbers approximate, names authoritative)

### contact_anything (this repo)
- Contact tokens: `sam_3d_body/models/meta_arch/sam3d_body.py` `_initialze_model` ~207–256
  (embedding + `contact_posemb_linear` FFN + `contact_feat_linear`), injected in
  `forward_decoder` ~566–599. **Asymmetric mask** ~589–599:
  `token_mask[:, :contact_emb_start_idx, contact_emb_start_idx:] = False` (True=allowed) —
  no original token attends contact tokens. Per-layer anchored update:
  `contact_token_update_fn` ~2126–2260 (2D-keypoint posemb + grid-sampled backbone feats,
  anchors = `MODEL.CONTACT_HEAD.KEYPOINT_INDICES`; extremity config uses MHR70
  `[62, 41, 13, 14]` = left_wrist, right_wrist, left_ankle, right_ankle).
- Contact head: `sam_3d_body/models/heads/contact_head.py::ContactHead`, `per_token` mode =
  one shared FFN per token → `[B, 4]`; assembled into `out["contact"]` in `forward_decoder`
  ~672–698.
- Temporal: `sam_3d_body/models/modules/temporal.py::ContactTemporalModule` — pre-LN blocks,
  zero-init `gamma` gates (exact identity at init), `attend: joint|per_token`, optional causal
  mask, sinusoidal PE over `frame_pos_sec`. Batch fields `seq_len` (int), `frame_pos_sec [B]`,
  `frame_valid [B]` come from `contact/data/collate.py`; clips are flattened clip-major
  (`rows [c*T : (c+1)*T]` = clip c in temporal order).
- Frozen per-frame outputs in `out["mhr"]` (see `mhr_head.py::MHRHead.forward` ~346–367 and
  `camera_head.py`): `mhr_model_params [B,204]` (the Momentum compact vector actually fed to
  MHR: `[global_trans*10 (3), global_rot euler (3), body_pose (130), scales (68)]`),
  `shape [B,45]`, `scale [B,28]`, `global_rot [B,3]` (euler ZYX, camera frame),
  `pred_cam_t [B,3]` (camera-frame root translation), `focal_length [B]`,
  `pred_joint_coords [B,127,3]`, `joint_global_rots [B,127,3,3]`,
  `pred_keypoints_3d [B,70,3]` — note `pred_vertices`/`pred_keypoints_3d`/`pred_joint_coords`
  have axes 1,2 **sign-flipped** ("camera system difference": MHR native space is y-up,
  camera space y-down), while `joint_global_rots` is NOT flipped (stays in native frame) —
  mhr_head ~339–343 vs ~365. Also `mhr_model_params[:3]` is **zeroed** (mhr_head ~296) —
  the real root translation only exists as `pred_cam_t`.
- Freeze: `contact/model.py::_trainable_name_filter` (~108) = literal `"contact"` substring;
  `pin_frozen_eval` (~115) re-pins non-contact modules to eval on every `train(True)`.
- Warm start: `contact/checkpoint.py::initialize_common_contact` (~316) — loads common contact
  params, currently allows only `contact_temporal.*` to be new.
- Loss/DDP: `contact/losses.py::MultiTargetContactLoss` + `ddp_global_mean_term` (exact global
  masked mean); trainer wiring in `scripts/train.py::_ddp_weighted_loss` (~414).
- Config: `contact/config.py` — `DEFAULTS` dict IS the schema, deep-merge with `base:`,
  unknown keys hard-error, semantics in `_validate_semantics`.
- Tests to mirror: `tests/test_temporal_invariance.py` (noise-floor + exact Jacobian
  isolation), `tests/test_grad_flow.py`, `tests/test_temporal.py` (CPU identity/mask units).

### BetterRobot (`/data3/rikhat.akizhanov/better/BetterRobot`, pkg `better-robot`)
- `rnea(model, q, v, a, *, fext=None, data=None, use_warp=False) -> tau`
  (`src/better_robot/dynamics/rnea.py` ~263). `tau = M(q)a + b(q,v) + g(q) − Jᵀ·fext`,
  shape `(B..., nv)`; for a free-flyer model **`tau[..., :6]` is the 6D root wrench in the
  base LOCAL frame** — our residual. `fext: (B..., njoints, 6)` = per-joint wrench
  `[fx,fy,fz,τx,τy,τz]` in that joint's **LOCAL** frame. Gravity lives in
  `values.gravity` (6-vec, default `[0,0,-9.81,0,0,0]`), differentiable, per-clip
  overridable. Batched, fp32, GPU, autograd through q/v/a/fext all gradcheck-tested
  (`tests/tasks/test_contact_forces.py`, `tests/autograd/`).
- **Reference prototype of our loss**: `src/better_robot/tasks/contact_forces.py` —
  `solve_contact_forces` minimizes `tau[:6]` with force regularizers;
  `_trajectory_derivatives(model, q, dt)` = manifold central differences using
  `Model.difference` (SE3 log for the free-flyer block). Manifold smoothing:
  `src/better_robot/tasks/smoothing.py::smooth_trajectory`.
- Frames: pinocchio-style `"world" | "local" | "local_world_aligned"`
  (`kinematics/jacobian.py`); wrench transforms via `spatial/force.py::Force.se3_action`.

### BetterHuman (`/data3/rikhat.akizhanov/better/BetterHuman`, pkg `better-human`)
- `MHR` body (`src/better_human/bodies/mhr/body.py`): loads
  `models/MHR/converted/mhr_lod{0..6}.npz`, builds a BetterRobot tree — 127 native joints,
  203 BR joints, `nq=132` (`[3 trans, 4 quat scalar-last, 125 pose]`), `nv=131`,
  **free-flyer root**, mass/inertia from mesh (uniform 1000 kg/m³, ~81.5 kg at LOD1).
- `MHRClassic(identity_coeffs[45], model_parameters[204], expression[72]|None)`;
  `from_classic(classic) -> (shaped_body, q)` and lossless `to_classic` round trip.
  Momentum partitions (`bodies/mhr/archive.py::parameter_partitions`): `[0:6]` root
  tx,ty,tz + euler-XYZ, 125 pose channels at `[6..129, 151]`, 73 proportion channels at
  `[130..150, 152..203]`. Conventions: meters, **y-up**, z-forward.
- FK: `body.fk(q) -> better_robot.Data`; native 127-joint poses via
  `body.native_joint_poses(data)`.
- CRBA is **singular** for MHR (mimic overcompleteness + zero-mass cosmetic bodies) —
  forward dynamics is off the table, but RNEA (inverse dynamics) needs no M⁻¹ and is fine.
- **Gap 1**: MHR + RNEA + fext has never been run end-to-end (RNEA is only tested on SMPL).
- **Gap 2**: the sam3d conda env has a **stale editable `better_human` 0.0.1** pointing at
  `/data3/rikhat.akizhanov/human_global_motion/...` (a different, pypose-based project) and
  **no `better_robot`**. Env wiring is step 01. **[RESOLVED by step 01 — see
  `for_agents/01_results.md`: both gaps closed, MHR+RNEA+fext validated, benchmarks
  recorded.]**

### ClimbingVideos reconstruction pipeline (verified 2026-07-22)

(`/data3/rikhat.akizhanov/better/BetterVideoReconstruction` — modifiable — plus its working
data at `better/data/ClimbingVideos`.)
- Per-scene metric cameras: `features/geometry/<sid[:2]>/<sid[2:4]>/<sid>/transform.npz` —
  `extrinsics (N,4,4) f32`, **camera-from-world, OpenCV convention**, translation metric
  after the scale stage (`metric=True`, cumulative `scale`;
  `scripts/stages/estimate_scale.py::_metricize_geometry` ~273–302 does
  `extr[:, :3, 3] *= s`). Frame 0 ≈ identity → world ≈ the metric camera-0 frame. Row k ↔
  exported frame k (sequential `frame_indices`, exporter-asserted).
- Exporter `scripts/export_contact_dataset.py` already `np.load`s `transform.npz` but
  exports only `intrinsics_px_orig`; its `common` dict (~303–313) feeds both train
  `labels.npz` and test `inputs.npz` symmetrically. ClimbingVideos_v1 (331 train / 30 test
  scenes, schema v2) has **no extrinsics today** — step 02 adds `extrinsics`,
  `gravity_world`, `cam_scale` and backfills existing scenes without re-decoding video.

## 3. Architecture

```
ClimbingVideos_v1 ──► batch: images/labels + cam_from_world [B,4,4], gravity_world [B,3]
        │                                        (step 02: exported from the pipeline)
        ▼
frozen SAM 3D Body ──► out["mhr"] per frame (detached)
        │                       │
        │                       ▼ contact/physics/adapter.py     (step 03)
        │              MHRClassic → from_classic → shaped MHR body,
        │              q trajectory [B_clips, T, 132] in the metric reconstruction
        │              world (camera pose composed via cam_from_world⁻¹),
        │              extremity BR joint ids
        │                       │
        ├─► contact tokens ─► contact head ─► joint_probs [B,4] ─┐ (detached gate)
        │                                                        │
        └─► force tokens ─► (force temporal) ─► force head ──────┤
             (steps 04, 05)     out["force"]["joint_forces"]     │
                                [B,4,3], units of body weight    ▼
                                            contact/physics/loss.py   (step 06)
                                smooth(q) → v,a (manifold central diffs, dt from
                                frame_pos_sec) → fext (frame-convert, N = pred·m·g)
                                → rnea (per-clip gravity ∝ gravity_world) → tau
                                loss = ‖tau[:,:6]‖² (normalized) + gated/smooth/L2 regs
```

New code lives in `contact/physics/` (subpackage, like `contact/data/`): `adapter.py`
(SAM↔BetterHuman bridge — one reason to change: the mapping) and `loss.py` (objective — one
reason to change: the loss design). Model-side additions follow the existing contact hooks in
the vendored `sam_3d_body` with the same `# --- contact ... ---` delimiter style.

## 4. Design decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **Force tokens appended after contact tokens; mask generalized so every earlier token block is blocked from attending later blocks** (`token_mask[:, :force_emb_start_idx, force_emb_start_idx:] = False` in addition to the existing contact line). Force tokens attend everything. | Preserves MHR invariance *and* gives contact logits an exactly-zero Jacobian w.r.t. all force params. Forward values agree only within the CUDA noise floor (the longer token sequence can change SDPA reduction order) — never assert `torch.equal`; regime (a) warm start reproduces contact predictions to noise floor. Test-enforced (step 03). |
| D2 | **Force tokens reuse the contact anchor indices** (`MODEL.CONTACT_HEAD.KEYPOINT_INDICES`), own embedding + posemb/feat linears. No global force tokens. | Forces exist exactly where contacts are predicted; a separate index list is speculative config (YAGNI). |
| D3 | **Trainable filter becomes the substring pair `("contact", "force")`** (hard-coded, both in `_trainable_name_filter`); **eval-pin becomes requires_grad-derived**: pin everything, then set `.training = mode` on every module in a subtree whose root *recursively owns ≥1 trainable param* — propagated top-down so param-less children (Dropout!) inside a trainable head toggle too. New modules are named `force_*`. | Two regimes need contact-frozen-but-force-training; a name list in config is rope nobody asked for. Subtree propagation keeps a *trainable* head's dropout active (a naive "modules with direct trainable params" rule would silently disable it) while a *frozen* contact head's dropout stays off in regime (a) — the name-based pin gets the latter wrong. |
| D4 | **Regime switch = `train.freeze_contact: bool`** (default false). When true: contact params get `requires_grad=False` after the normal unfreeze, contact loss is excluded from the total (metrics still logged), and config validation requires `model.init_contact_checkpoint`. | One boolean, self-documenting; avoids a free-form pattern language. |
| D5 | **Force head predicts dimensionless force in units of body weight** (m·g of the shaped model); physics converts to newtons. Final head layer zero-init. | O(1) regression target, no magic `scale_newtons` constant, model starts at "no forces" (residual = pure kinematics baseline — a meaningful curriculum start). |
| D6 | **Force frame config `model.force_head.frame: local_world_aligned \| local`** — LWA = **per-frame camera y-up axes** (the un-flipped camera frame: what a level camera calls up/right/forward), LOCAL = the extremity joint frame. The head never predicts in the reconstruction world — its yaw is arbitrary per scene and not inferable from an image crop. Conversion to RNEA's per-joint LOCAL fext: LWA → `f_world = R_w←c·D·f_pred`, then `R_jointᵀ·f_world`; LOCAL passes through. Pure force at the joint origin, zero torque component. | The prediction frame must be learnable from pixels; camera y-up is the closest learnable proxy to "human world aligned". Matches the user's two requested variants and RNEA's native fext contract. Offset contact points (`r×f` moments, grip torques) are explicitly out of scope v1 (BetterRobot defers them too). |
| D7 | **Physics world = the metric reconstruction world.** The dataset provides per-frame `cam_from_world [4,4]` (camera-from-world, OpenCV, metric — the pipeline's fuse+scale output, exported in step 02); the adapter composes SAM's camera-frame root pose with `T_w←c = cam_from_world⁻¹` (plus the un-flip `D = diag(1,-1,-1)`) so q lives in one static world per scene. **Gravity = per-scene `gravity_world` unit vector exported with the dataset** (defined as the first camera's +y axis mapped to world — the stated assumption until the pipeline estimates true gravity), magnitude `physics.gravity` (default 9.81), applied per clip via `values.gravity`. | Moving cameras are common in ClimbingVideos; without extrinsics, camera motion aliases into body acceleration and corrupts the residual. The pipeline already computed metric world cameras — using them is free. First-camera-y lives in the data, swappable without touching training code. |
| D8 | **Contact gate uses predicted `joint_probs`, detached.** Non-contact penalty `(1−p)·‖f‖²`; contact-min penalty `p·relu(f_min−‖f‖)²`. | Physics must never train the contact head (labels do); detaching prevents the force loss from inflating contact probs to license fictitious forces. GT labels are deliberately NOT used as the gate: video labels are motion-gated *stable* contact, a different quantity from instantaneous load. |
| D9 | **Smoothing + differentiation on the manifold, dt from `frame_pos_sec`**: smooth the detached q trajectory (BetterRobot `smooth_trajectory` or equivalent kernel + slerp/sclerp for the root), central differences via `Model.difference` (pattern: `tasks/contact_forces.py::_trajectory_derivatives`). Residual only on interior frames whose full stencil (smoothing + two differences) lies inside the valid window. | Quaternion root forbids naive subtraction; real-time dt is already in the batch; edge frames would otherwise inject garbage accelerations. |
| D10 | **Force is NOT a `contact.targets` entry.** New top-level `physics:` config section + `model.force_head` / `model.force_temporal`. `TargetSpec`/`validate_targets` untouched. | The physics loss has no `(gt, mask)` pair; shoehorning it into the target system would corrupt a clean abstraction. |
| D11 | **Force temporal = a second `ContactTemporalModule` instance named `force_temporal`, `post_decoder` placement only, `attend: per_token` default.** | Reuse over reimplementation; the audited between_layers/pre_decoder placements were near-inert for contact — supporting them for force is speculative. |
| D12 | **All losses dimensionless**: residual force part normalized by m·g, residual torque part by m·g·1 m; torque regularizers likewise. | Weights become O(1) and transfer across body shapes/scales. |
| D13 | **Cameras are a dataset input, not a model input.** The model never consumes extrinsics; `cam_from_world`/`gravity_world`/`cam_valid` flow batch → physics loss only. Still-image datasets carry `cam_valid=False` and are physics-ineligible; an otherwise-eligible video clip without cameras raises (stale export must not become a silent no-op). | Keeps the model deployable on plain images/video; the "someone provides extrinsics" contract lives in one place — the dataset schema. |

## 5. Assumptions & risks (record in docs; revisit before scaling up)

- **R1 — MHR+RNEA+fext untested end-to-end.** Mitigated first, in step 01, with synthetic
  static-equilibrium tests + a benchmark, before any model code is written.
- **R2 — Gravity direction = first camera's +y.** Assumes the camera is level at scene
  start; per-scene constant thereafter (camera *motion* is compensated via extrinsics,
  initial *tilt* is not). Wrong tilt biases the residual. The exported `gravity_world` key
  makes the assumption data-visible and swappable (future: pipeline-side gravity
  estimation; `values.gravity` is even differentiable). Not v1.
- **R3 — Extrinsics quality.** Camera motion is compensated exactly where the
  reconstruction is right; the residual risk shifts to VGGT drift and the body-ruler scale
  estimate (`prefit-dual`) — errors there feed accelerations through double finite
  differencing. Monitor per-scene residuals; drop pathological scenes if needed.
- **R4 — Metric scale consistency.** Two scales must agree: SAM's `pred_cam_t` (from
  intrinsics) and the pipeline's world scale (body-ruler, exported as `cam_scale`). Both
  are body-anchored so gross mismatch is unlikely; disagreement shows up as spurious root
  acceleration, and the step-03 FK acceptance test bounds the per-frame part. Mass tracks
  predicted shape: `with_shape` recomputes body inertias from the shaped mesh (verified —
  `BetterHuman/.../bodies/mhr/deformation.py` ~68–96).
- **R5 — Uniform density 1000 kg/m³, LOD1 mass ≈ 81.5 kg neutral.** Fine for v1.
- **R6 — Forces applied at wrist/ankle joint origins** — real contact points are offset
  (sole, palm, grip moments) → unmodeled `r×f` torque error. Accepted v1.
- **R7 — Perf.** MHR FK is kernel-launch-bound (~203 BR joints); physics adds FK+RNEA
  fwd+bwd per step. Step 01 benchmarks at realistic batch; escape hatches: `use_warp` lane,
  fewer physics frames per batch. Do not pre-optimize.
- **R8 — Label semantics.** Motion-gated "stable contact" labels ≠ instantaneous load;
  gating by predicted probs (D8) inherits this bias. Accepted, documented.
- **R9 — sam3d env surgery.** Step 01 removes the stale `better_human` 0.0.1 editable from
  the shared sam3d env; anything else using that env's stale package breaks. Surface loudly
  in that step's report.

## 6. Step order

```
01 dynamics deps + end-to-end MHR/RNEA validation    DONE — see for_agents/01_results.md
02 camera extrinsics: pipeline export + dataset + loader plumbing            DONE
03 MHR adapter (frozen outs + cams → body + world-frame q) needs 01, 02      DONE
04 force tokens + head + freeze/eval-pin generalization                      DONE
05 force temporal block                              needs 04                DONE
06 physics loss (smooth → v,a → RNEA → residual+regs) needs 03, 04           DONE
07 training integration (trainer, warm start, regimes, configs, DDP) needs 04, 06 (05 optional) DONE
08 eval + demo + docs                                needs 07                DONE
```

04/05 are independent of 01–03 and may run in parallel with them. Each step lands with its
tests green (`PYTHON=/data3/rikhat.akizhanov/miniconda3/envs/sam3d/bin/python -m pytest
tests/ -q -m "not slow"`; slow GPU suites where the brief says so) and must not break the
existing suite or the CLAUDE.md invariants (as amended by step 03).

## 7. Verification strategy

- **Physics correctness** is anchored twice: step 01 (BetterHuman MHR standalone: zero-fext
  residual = gravity wrench; synthetic support forces reduce it) and step 05 (same through
  our adapter+loss, plus gradients reach the force tensor).
- **Isolation** is anchored by extending the existing invariance suite: MHR outputs AND
  contact logits must have zero Jacobian w.r.t. every force param; force outputs must move.
- **Adapter truth**: BetterHuman FK from adapted world-frame q, mapped back through
  `cam_from_world`, must reproduce the frozen model's own `pred_joint_coords` (camera
  frame) within tolerance on real checkpoint outputs — the only acceptable proof that the
  204-param mapping, the extrinsics inversion, and the flip composition are all right.
  (Known wrinkle: SAM composes its 204 vector as 136 pose + 68 scales while Momentum
  partitions 6+125+73 — do NOT assume the layouts line up; the FK test decides.)
- **Training plumbing**: grad-flow tests per regime (regime a: contact grads None, force
  grads nonzero; regime b: both nonzero), DDP exact-reduction test for the physics masses,
  warm-start round trip (contact ckpt → force model → contact logits within noise floor).
  DDP gotcha (found in review): the trainer runs `find_unused_parameters=False`, and force
  params reach the loss graph only through the physics loss — every batch must therefore
  touch `out["force"]` with a graph-connected zero term even when no clip is
  physics-eligible (same trick as `safe_logits.sum()*0.0` in `contact/losses.py`).
