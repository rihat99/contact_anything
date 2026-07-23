# Step 08 — Evaluation metrics, demo overlays, documentation

Depends on step 07. Read `plan/README.md` §1, §5 (risks — several must land in docs).

## Evaluation (`scripts/evaluate.py`)

With no GT forces, evaluation = physics consistency + plausibility statistics on the chosen
split (val or the manual test split):

1. `physics_residual` (dimensionless, mean over residual frames) — headline number.
2. Per-extremity mean/percentile force magnitude (in body weights AND newtons) split by
   predicted contact state, and the two gate-violation rates: mean ‖f‖ on non-contact
   frames (want ≈0) and fraction of contact frames with ‖f‖ < contact_min_bw.
3. Plausibility: distribution of the vertical force sum over extremities in body weights
   (≈1 for quasi-static climbing), per-clip residual-vs-time summary.
4. Print in the existing table style; log to wandb when enabled. Keep the contact metrics
   section untouched for force-enabled checkpoints (contact still evaluates as before).

## Demo (`scripts/demo_climbing_videos.py`)

5. When the checkpoint has a force head: draw per-extremity force arrows on the rendered
   frames — anchor at the extremity's 2D keypoint, direction = predicted 3D force projected
   through the camera intrinsics, length ∝ magnitude in body weights, color by extremity
   (reuse the existing color scheme); annotate magnitude. Keep the GT/predicted contact
   visualization as is. No new script — extend the existing one behind a presence check.

## Documentation

6. `CLAUDE.md` (authoritative project guide — keep its telegraphic style):
   - Project Overview + Architecture: force tokens/head/temporal, `contact/physics/`
     (adapter + loss), the RNEA residual objective in two sentences.
   - Invariants: already amended in step 04 (verify wording); add "physics loss consumes
     detached frozen outputs and detached contact probs — gradients reach force params
     only".
   - Key Commands: the two force configs, the step-01 env note
     (`better_robot`/`better_human` editables + `BETTERHUMAN_MODELS_DIR` if that route was
     chosen).
   - Datasets/labels section: one line on the stable-contact vs instantaneous-force
     semantic gap (risk R8).
6b. Module docstrings that predate the force work and now under-describe their file —
   known gap: `contact/model.py` top-of-file docstring still describes only the contact
   pipeline (steps 04/05 generalized the freeze filter + eval-pin). Sweep the touched
   modules for similar stale docstrings while writing docs.
7. `docs/forces.md` (new, narrative — this repo's docs/ is flat): the physics formulation
   (residual wrench, force units, frames LWA/LOCAL, the reconstruction-world convention —
   `cam_from_world`/`gravity_world` provenance from the BetterVideoReconstruction
   fuse+scale stages, step 02), and the recorded assumptions R2–R6/R8 from
   `plan/README.md` §5 so they outlive the plan folder.
8. Update `plan/README.md` status line; mark step checklist done. The plan folder stays —
   it is the design record.

## Acceptance

- `evaluate.py` runs on a force checkpoint (or, lacking a trained one, on a warm-started
  untrained model — numbers meaningless but pipeline exercised) without disturbing
  contact-only evaluation (run once on an existing contact checkpoint to prove it).
- Demo produces frames with arrows on a real clip.
- Docs match what was actually built (cross-check each claim against the code, not the
  plan).

## Out of scope

New viewer features, training-quality investigations, threshold tuning.
