# Step 06 — Physics loss: smoothing → v,a → RNEA → residual + regularizers

Depends on steps 03 (adapter) and 04 (`out["force"]`). Read `plan/README.md` D5–D9, D12,
risks R2–R8, and the BetterRobot §2 notes. Reference implementation to study before writing
anything: `BetterRobot/src/better_robot/tasks/contact_forces.py` (`solve_contact_forces`,
`_trajectory_derivatives`) and `BetterRobot/src/better_robot/tasks/smoothing.py`.

Deliverable: `contact/physics/loss.py` — a `PhysicsLoss` module + the `physics:` config
section. It consumes, per batch (flat `B = n_clips*T`, clip-major):
- `out["mhr"]` (detached — q trajectory via the step-03 adapter),
- `out["force"]["joint_forces"] [B,4,3]` (dimensionless, units of body weight; grads live),
- `out["contact"]["joint_probs"] [B,4]` (**detach before use** — D8),
- `batch["seq_len"]`, `batch["frame_pos_sec"] [B]`, `batch["frame_valid"] [B]`,
- `batch["cam_from_world"] [B,4,4]`, `batch["gravity_world"] [B,3]`, `batch["cam_valid"] [B]`
  (step 02 — metric camera-from-world extrinsics + per-scene downward unit vector).

## Pipeline (per batch)

1. **Eligibility**: physics applies only to clips with `seq_len >= physics.min_frames`
   (default 5), all frames valid, and all frames `cam_valid`. Still-image batches (T=1) and
   ineligible clips contribute zero numerator and zero mass — batch-level mixing then works
   like missing contact targets. **Misconfiguration guard**: a clip that is otherwise
   eligible (video, T ≥ min_frames, frames valid) but has `cam_valid=False` means the
   dataset lacks the step-02 camera export — raise with the clip key, never skip silently
   (a silently all-ineligible run would "train" nothing). **DDP requirement**: the trainer wraps with
   `find_unused_parameters=False`, and force params reach the graph ONLY through this loss —
   so on EVERY call (including fully ineligible batches) the returned numerator must include
   a graph-connected zero touching `out["force"]["joint_forces"]`
   (`joint_forces.sum() * 0.0`; mirror `safe_logits` in `contact/losses.py` ~144). This
   keeps every force param (embedding, linears, head, force_temporal) in the autograd graph
   each step.
2. **q trajectory**: adapter (`q_from_mhr_out(mhr_out, cam_from_world, ...)`) →
   `q [n_clips, T, 132]` in the metric reconstruction world (detached), shaped body,
   per-clip mass `m`, extremity joint ids.
3. **Smoothing** (D9): kernel-smooth q along T on the manifold. BetterRobot's
   `smooth_trajectory` only accepts q of dim 4 or 7 (verified) — it canNOT take the
   composite 132-dim q. So compose: plain 1D convolution for the 125 revolute channels +
   translation, and the root SE(3)/quaternion block via `smooth_trajectory` (dim-7 input)
   or a hemisphere-aligned weighted quaternion mean from `better_robot.lie`. Kernel from
   `physics.smoothing_kernel` (odd length; `[1.0]` = off). No padding tricks — frames whose
   smoothing stencil leaves the window are simply not residual frames (see 5).
4. **v, a**: manifold central differences with per-interval `dt` from `frame_pos_sec`
   (`Model.difference`-based; follow `_trajectory_derivatives` but honor non-uniform dt).
5. **Stencil mask**: a residual frame must have its full stencil (smoothing halo + one
   central difference for v + one more for a) inside the clip. Compute the set of valid
   interior frames once from T and kernel length; T=5 with kernel `[1]` → central 3 or 1
   frames depending on the chosen difference scheme — document the scheme in a docstring
   and test it.
6. **fext assembly**: `f_newtons = pred * m * g` (D5, `g = physics.gravity` magnitude).
   Frame conversion per `model.force_head.frame` (D6):
   - `local_world_aligned` (= per-frame camera y-up axes): `f_cam = D @ f_pred` with
     `D = diag(1,-1,-1)`, then `f_world = R_w←c[t] @ f_cam`
     (`R_w←c = cam_from_world[t,:3,:3].T`), then `f_local = R_jointᵀ @ f_world` with
     `R_joint` = world-frame joint rotation from FK at the (smoothed) q — all detached;
   - `local`: use as-is (already the joint frame).
   Scatter into `fext [n_clips, T_res, 203, 6]` at the 4 extremity joint ids, torque
   component zero. All other joints zero.
7. **RNEA**: `tau = rnea(robot, q, v, a, fext=fext)` on the residual frames. Gravity is
   **per clip**: linear part of `values.gravity` = `physics.gravity * gravity_world[clip]`
   (the exported downward unit vector; with frame-0-anchored worlds ≈ `[0, 9.81, 0]` —
   OpenCV y-down world, so "down" is +y), angular part zero. Set it on the shaped values
   without mutating global state (confirm the 6-vec layout `[linear, angular]` against
   BetterRobot's default `[0,0,-9.81,0,0,0]`). `tau[..., :6]` = root residual, `tau[..., 6:]` = joint
   torques. Remember the adapter's batched-shape contract (step 03): the shaped robot's
   values carry a singleton time axis so `[n_clips, T_res, ...]` q/v/a/fext broadcast.

## Loss terms (all dimensionless — D12; `w_* = physics.loss.*` weights)

Let `p` = detached contact probs, `f` = predicted force (units of body weight, so `‖f‖=1`
is one body weight), `r_f = tau[:3]/(m·g)`, `r_τ = tau[3:6]/(m·g·1m)`,
`τ_j = tau[6:]/(m·g·1m)`:

- `residual`: `‖r_f‖² + ‖r_τ‖²` — THE objective, default weight 1.0.
- `force_noncontact`: `(1−p)·‖f‖²` — strong, default 1.0. (Computed on all frames of
  eligible clips, not just residual frames — the prediction should be clean everywhere.)
- `force_at_contact`: `p·relu(f_min − ‖f‖)²` with `f_min = physics.loss.contact_min_bw`
  (units of body weight, default 0.05) — weak, default 0.1.
- `force_smooth`: `‖f_t − f_{t−1}‖²` over adjacent frames, computed on the **world-frame**
  forces (so camera rotation within a clip doesn't masquerade as force change; for the
  `local` head frame this still conflates joint rotation — note it in the docstring, keep
  the term anyway).
- `force_l2`: `‖f‖²`, small (default 0.01).
- `torque_l2`: `‖τ_j‖²`, small (default 0.01), residual frames only.
- `torque_smooth`: `‖τ_j(t) − τ_j(t−1)‖²` over adjacent residual frames — default 0.0
  (needs ≥2 residual frames; only meaningful for T≥7 with the recommended scheme).

Return the same contract `MultiTargetContactLoss` uses so the trainer (step 07) can do exact
DDP reduction: per-term `(numerator_tensor, mass)` where mass = count of contributing
(frame × extremity or frame) elements, plus a `parts` dict of detached scalars for logging.
Reuse `contact/losses.py::ddp_global_mean_term` — do not reinvent it.

## Config (`contact/config.py` DEFAULTS + commented block in `configs/base.yaml`)

```yaml
physics:
  enabled: false
  model_path: null          # null → $BETTERHUMAN_MODELS_DIR (per step 01's decision)
  lod: 1
  gravity: 9.81             # magnitude m/s²; DIRECTION is per scene: batch["gravity_world"]
  min_frames: 5
  smoothing_kernel: [0.25, 0.5, 0.25]
  loss:
    residual: 1.0
    force_noncontact: 1.0
    force_at_contact: 0.1
    contact_min_bw: 0.05
    force_smooth: 0.1
    force_l2: 0.01
    torque_l2: 0.01
    torque_smooth: 0.0
```

Validation: `physics.enabled` requires `model.force_head.enabled`, a video dataset in
`data.datasets`, and `sequence.frames_per_clip >= physics.min_frames >= 3`. Weights ≥ 0.
`physics:` numbers stay OUT of the checkpoint arch signature (step 04 note).

## Tests — `tests/test_physics_loss.py` (CPU; BetterRobot/BetterHuman run fine on CPU)

Use the real MHR body at LOD1 (env from step 01) but tiny synthetic "predictions":
1. **Static equilibrium**: constant q over T, identity `cam_from_world`,
   `gravity_world=[0,1,0]`, zero forces → residual ≈ gravity wrench (‖r_f‖ ≈ 1 body
   weight, dimensionless ≈ 1). Then inject the analytic supporting force from step 01's
   script at the extremities → linear residual part drops by ≥ 10×.
2. **Gradients**: loss.backward() yields finite nonzero grad on a leaf force tensor; zero
   grad on the contact-probs tensor (detachment proof); no grad reaches q.
3. **Stencil/eligibility**: T=1 clips and clips with an invalid frame produce zero mass —
   but the returned numerator is still graph-connected to the force tensor (backward on it
   yields a zero, non-None grad: the DDP guarantee). The residual-frame index set matches
   the documented scheme for T=5 and T=7. An otherwise-eligible video clip with
   `cam_valid=False` raises (the misconfiguration guard).
4. **Gating**: p=0 everywhere → force_noncontact strictly increasing in ‖f‖; p=1 with f=0 →
   force_at_contact > 0; p=1 with ‖f‖>f_min → force_at_contact = 0.
5. **Dimensionlessness**: doubling mass with the same dimensionless setup leaves the
   normalized residual invariant (guards D12).
6. **Frame equivariance**: apply one rigid transform to every `cam_from_world` (world
   re-anchored) while rotating `gravity_world` consistently → all loss terms unchanged
   (this is the test that catches w2c-vs-c2w and flip-sign mistakes).

Mark anything needing the real checkpoint or GPU `@pytest.mark.slow`; the five above should
not need either.

## Out of scope

Trainer integration, DDP wiring, metrics/monitor, experiment configs (all step 07).
