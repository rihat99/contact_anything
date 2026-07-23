# Step 07 — Training integration: trainer wiring, warm start, regimes, configs

Depends on steps 04 + 06 (05 optional). Read `plan/README.md` D3/D4, §7, and the trainer
notes in §2. Files: `scripts/train.py`, `contact/engine.py`, `contact/checkpoint.py`,
`contact/config.py`, `configs/`.

## Trainer (`scripts/train.py`)

1. Build `PhysicsLoss` (with the step-03 adapter inside it) when `physics.enabled`; move it
   to the training device once.
2. Per step: `total = contact_ddp_loss + physics_ddp_loss`. Physics terms go through the
   SAME exact-global-mean machinery as contact (`_ddp_weighted_loss` ~414–440 all-reduces
   per-target masses; extend it or add a sibling that consumes PhysicsLoss's
   `(numerator, mass)` pairs — reuse `ddp_global_mean_term`). The zero-active skip
   (`if active:` ~496–506) becomes: step if contact-active OR physics-active. DDP runs with
   `find_unused_parameters=False` (~298–303): correctness on physics-inactive batches
   relies on PhysicsLoss's always-graph-connected zero term (step 06) — include such a
   batch in the smoke test below.
3. **Regime (a)** (`train.freeze_contact: true`): exclude the contact loss from `total`
   (its params are frozen; keep computing/logging its metrics), physics is the objective.
4. Logging: `train/physics_*` parts (residual, each regularizer, masses) through the
   existing tracking path (wandb + TB); same keys under `val/` in validation.
5. Validation: compute physics parts on the val loader each eval epoch (no grads). New
   monitorable metric `val/physics_residual` (the residual term only, not the regularizers —
   the regularizer mix shouldn't decide "best"). Wiring warning (verified): monitor names
   are generated as `self.targets × _MONITOR_METRICS` (`train.py` ~331–334), the val
   summary hardcodes classification keys (`f1/f2/precision/recall`) per target (~643–645),
   and `_monitor_value` indexes `val["metrics"][target][key]` (~344–345). "physics" is NOT
   a contact target — do not add it to `self.targets` (the f1-hardcoded summary would
   KeyError). Register it as a monitor pseudo-target with its single `residual` metric:
   extend `_validate_monitor` to accept `{split}/physics_residual` explicitly, store the
   value under `val["metrics"]["physics"]["residual"]` (guarded out of the classification
   summary loop), mode: `residual` → min.

## Warm start (`contact/checkpoint.py`)

6. Generalize `initialize_common_contact` (~316–385). Three distinct blockers to fix, not
   one (verified against the code):
   - **Param allowance** (~376): params allowed to be absent from the source become
     `contact_temporal.`-prefixed OR any name containing `"force"`. Everything else keeps
     the current strictness (hard-fail on any other diff — never silent).
   - **Signature comparison** (~340–344): it pops only `temporal` before comparing source
     vs target `_arch_signature`; step 04 added force_head/force_temporal keys, so a
     contact-only source vs force-enabled target would hard-fail. Pop/exempt the force keys
     symmetrically, same as temporal.
   - **Precondition** (~358–360): "only valid when the target enables temporal" — relax to
     "target enables temporal OR force" (a force warm-start with temporal disabled is the
     regime-(a) default and must work).
   The existing use case (contact ckpt → temporal-enabled contact model) must keep working —
   its test proves it.
7. Config validation cross-check (already added in step 04): `freeze_contact` ⇒
   `model.init_contact_checkpoint` set. Resume (`--resume`) of a force run must keep
   working unchanged — `_arch_signature` already covers force fields (steps 04/05); add a
   checkpoint round-trip test for a force-enabled model (save → load → identical trainable
   state, RNG restored).

## Experiment configs

8. `configs/climbing_videos_force_warmstart.yaml` — regime (a): base
   `configs/climbing_videos_joint.yaml`; force_head + physics enabled,
   `train.freeze_contact: true`, `model.init_contact_checkpoint: <path placeholder + comment
   pointing at the best contact run>`, monitor `val/physics_residual`,
   `sequence.frames_per_clip: 8`. **Frame-count pitfall (from step 06's landed scheme —
   read the `contact/physics/loss.py` docstring)**: residual frames are
   `{2+r ≤ t ≤ T−3−r}` with `r = len(smoothing_kernel)//2`; the default kernel
   (`[0.25,0.5,0.25]`, r=1) yields ZERO residual frames for T<7 — the residual objective
   would be silently dead. T=8 + default kernel → 2 residual frames. State this coupling
   in a yaml comment.
9. `configs/climbing_videos_force_scratch.yaml` — regime (b): contact joint target + force
   + physics all on, `freeze_contact: false`, monitor stays `val/joint_f1` (comment: physics
   residual is logged; switch monitor once contact plateaus if desired).
   Optional `_temporal` variants only if step 05 landed — do not speculate otherwise.

## Tests

- `tests/test_checkpoint.py` additions: warm start contact→force model (synthetic tiny
  checkpoint): contact params loaded, force params fresh, any *other* mismatch still
  hard-fails; regression: the existing contact→temporal warm start unchanged.
- Warm-start behavioral test (GPU slow, real ckpt if available): after
  `initialize_common_contact` + `freeze_contact`, contact `joint_logits` equal the
  source-model's within noise floor (leans on the D1 mask guarantee from step 04).
- DDP reduction unit test (CPU, mirror the existing exact-mean test pattern for
  `ddp_global_mean_term`): two simulated ranks with different physics masses reproduce the
  single-process global mean.
- End-to-end smoke (GPU slow): 2 training steps on a real climbing_videos micro-batch with
  physics enabled — loss finite, force params change, frozen params don't, no NaN.

## Out of scope

Eval-script metrics, demo rendering, docs (step 08). Hyperparameter tuning — the default
weights in step 06 are starting points; record, don't optimize.
