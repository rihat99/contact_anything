# Step 01 — Wire BetterRobot/BetterHuman into the sam3d env and validate MHR dynamics end-to-end

> **Agent log — completed 2026-07-22:** Replaced the stale BetterHuman editable, installed
> BetterRobot/BetterHuman with no dependency upgrades, and passed both upstream suites plus
> the CPU/CUDA validator. Key findings: resolve MHR70 extremities by name (ankles are native
> `l_foot`/`r_foot`), override BetterRobot's z-down default to MHR's y-up gravity, and retain
> NumPy 1.26.4. The installed BetterRobot checkout was already dirty; see `01_results.md`.

Read `plan/README.md` first (§2 "What exists", risks R1/R7/R9). No model/training code in
this step. Goal: after this step, `import better_robot, better_human` works in the sam3d env,
and we have proof + a benchmark that MHR + RNEA + external forces behaves correctly.

Python for everything: `PYTHON=/data3/rikhat.akizhanov/miniconda3/envs/sam3d/bin/python`.

## Part A — environment wiring

Current state of `/data3/rikhat.akizhanov/miniconda3/envs/sam3d`:
- has a **stale editable `better_human` v0.0.1** pointing at
  `/data3/rikhat.akizhanov/human_global_motion/better_human/src` (an older pypose-based
  project, NOT `/data3/rikhat.akizhanov/better/BetterHuman`);
- has **no `better_robot`**.

Tasks:
1. `grep -rn "better_human\|better_robot"` over the contact_anything source tree first —
   confirm nothing currently imports the stale package (expected: nothing does).
2. `pip uninstall better_human` (the stale 0.0.1; also remove any leftover
   `__editable__.better_human-0.0.1.pth` in site-packages).
3. Install editables **without letting pip touch torch/numpy**:
   `pip install --no-deps -e /data3/rikhat.akizhanov/better/BetterRobot`
   then `pip install --no-deps -e /data3/rikhat.akizhanov/better/BetterHuman`,
   plus any *small* missing runtime deps they need (`yourdfpy`, `scipy`) installed normally.
   Do NOT upgrade torch or numpy: BetterRobot declares `torch>=2.4, numpy>=2.0` — check the
   env's actual versions; if numpy is <2.0, verify imports and Part B still pass on the
   existing numpy rather than upgrading (an upgrade could break detectron2/sam_3d_body —
   if numpy<2 actually breaks BetterRobot at runtime, STOP and report; do not upgrade
   unilaterally).
4. Smoke: `better_robot.__version__` (0.2.0), `better_human` loads
   `MHR(...)` from `/data3/rikhat.akizhanov/better/BetterHuman/models/MHR/converted/mhr_lod1.npz`
   (pass `model_path` explicitly or set `BETTERHUMAN_MODELS_DIR`; decide and record which —
   an env var baked into configs is preferable to hardcoded absolute paths in code).
5. Run the relevant upstream test subsets inside the sam3d env to prove compatibility:
   `BetterHuman/tests/bodies/test_mhr.py` and `BetterRobot/tests/tasks/test_contact_forces.py`.

Report loudly (risk R9): removing the stale editable affects anything else that uses this
conda env with the old better_human. List what you uninstalled.

## Part B — end-to-end MHR + RNEA + fext validation (the never-tested path)

Key APIs (see plan/README.md §2 for details):
- `bh.MHR(...)`, `MHRClassic(identity[45], model_parameters[204], expression|None)`,
  `from_classic -> (shaped_body, q[132])`, `body.fk(q)`, `body.robot` (BetterRobot Model),
  `body.robot.joint_id(name)` / `body.structure.joint_names` (127 native names).
- `better_robot.rnea(model, q, v, a, fext=...)`; `fext (B..., njoints=203, 6)` per-joint
  LOCAL-frame wrench `[f, τ]`; `tau[..., :6]` = root wrench (base LOCAL frame);
  gravity in `model.values.gravity`.
- Reference: `BetterRobot/src/better_robot/tasks/contact_forces.py` (`solve_contact_forces`,
  `_trajectory_derivatives`) and its test `tests/tasks/test_contact_forces.py`.

Write a standalone validation script `plan/for_agents/artifacts/validate_mhr_dynamics.py`
(committed; it is evidence, not product code) that, on the neutral shaped MHR at LOD1:

1. **Gravity wrench**: static pose (v=a=0, fext=None) → `tau[:6]` linear-force norm ≈
   total_mass·9.81 (report mass; expect ≈ 81.5 kg neutral). Direction consistent with
   `values.gravity` and MHR's y-up convention.
2. **Support force closes the loop**: identify the wrist/ankle native joints by name (record
   the exact names — SAM anchors are MHR70 `[62,41,13,14]` = l_wrist, r_wrist, l_ankle,
   r_ankle; find the corresponding `joint_names` entries and BR joint ids). Apply a synthetic
   upward world-frame support force at one or two extremities (rotate into joint LOCAL frame
   via FK rotations: `f_local = Rᵀ f_world`), chosen so the linear part of the root residual
   cancels → assert the force part of `tau[:6]` drops to ~0 (torque part need not vanish for
   an arbitrary pose — assert only what is provable, or optimize the fext with a few Adam
   steps to show the full wrench is reducible, mirroring the BetterRobot test).
3. **Autograd**: `tau[:6].pow(2).sum().backward()` produces finite, nonzero grads on a
   leaf fext tensor.
4. **Motion**: a small synthetic q trajectory (e.g., sinusoidal root bob), manifold central
   differences (`Model.difference`-based, copy the `_trajectory_derivatives` pattern) → v, a
   → rnea runs, residual finite and larger than the static case.
5. **Benchmark** (risk R7): time FK+RNEA forward+backward through fext on GPU at
   B = 8 clips × 5 frames = 40 (and 128 for headroom), fp32. Report ms/iter. Also measure
   `with_shape` cost separately (it runs an internal neutral FK). Note: `with_shape` is
   known to recompute body inertias from the shaped mesh
   (`BetterHuman/src/better_human/bodies/mhr/deformation.py` ~68–96) — sanity-check that
   a non-neutral identity changes total mass, and record the number.

## Acceptance

- Both imports + upstream test subsets pass in the sam3d env.
- `validate_mhr_dynamics.py` runs clean in the sam3d env (CPU for 1–4; GPU for 5) and prints
  the assertions/numbers above.
- Report: exact extremity joint names/ids, total mass, benchmark numbers, the
  `with_shape`→inertia answer, numpy/torch versions kept, and anything uninstalled.

## Out of scope

- Any change to contact_anything source, configs, or tests (the script under
  `plan/for_agents/artifacts/` is the only new file).
- Any change to BetterRobot/BetterHuman source. If something there is actually broken,
  STOP and report — do not fix in place.
