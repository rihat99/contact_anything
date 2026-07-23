# Step 01 results — dynamics dependencies and MHR validation

Status: **PASS**. Step 01 is complete; no Step 02 product code was started.

## Deviations and important caveats

1. The plan's MHR70 indices `[62, 41, 13, 14]` are SAM keypoint indices, not indices into
   BetterHuman's 127 native joints. The validator resolves anatomical names through
   `native_pose_joint_indices`. MHR has no native joints named `l_ankle` or `r_ankle`, so the
   ankle anchors map to the nearly coincident native `l_foot` and `r_foot` origins. Using the
   numeric MHR70 indices directly would select incorrect joints.
2. BetterRobot constructs MHR with default gravity `[0, 0, -9.81]`, while MHR declares
   `+y` up. The validator explicitly uses `[0, -9.81, 0]`, matching project decision D7. No
   BetterRobot or BetterHuman source was changed.
3. BetterRobot and BetterHuman declare `numpy>=2.0`, but sam3d has NumPy 1.26.4. Per the
   plan, NumPy was not upgraded. Imports, both requested upstream suites, and the complete
   dynamics path pass on 1.26.4.
4. The validator accepts `--model-path`, then `BETTERHUMAN_MODELS_DIR`, then a sibling
   BetterHuman-checkout fallback. The recorded run used the fallback instead of changing a
   persistent global environment variable.
5. `with_shape` is timed for one identity. An exploratory batch of eight shape bakes was not
   stable as a repeated benchmark; Step 01 only requires its separate cost, not batched shape
   inference. FK+RNEA still uses the required batches 40 and 128.
6. The BetterRobot editable at commit `117648f` was already dirty before installation
   (residual/docs/tests changes and one untracked residual test). Dynamics files were not in
   that dirty set, and this step did not modify the external checkout, but the editable points
   to live, non-reproducible working-tree state. BetterHuman was clean at `c8da7ec`.

## Environment wiring

- Python: 3.12.12
- PyTorch: 2.8.0+cu128 (CUDA 12.8)
- NumPy: 1.26.4, unchanged
- SciPy: 1.17.0, already installed
- yourdfpy: 0.0.60, already installed
- Removed: stale `better_human 0.0.1` editable from
  `/data3/rikhat.akizhanov/human_global_motion/better_human`; its
  `__editable__.better_human-0.0.1.pth` was removed by pip.
- Installed with `--no-deps`: `better-robot 0.2.0` from BetterRobot, then
  `better-human 0.1.0.dev0` from BetterHuman.

This changes a shared conda environment. Any external process that relied on the old
pypose-based BetterHuman API must restart and migrate or restore the old editable.

The repository source search found no Python import of either package outside `plan/`; other
matches are only model-asset path strings.

## Upstream compatibility gates

- `BetterRobot/tests/tasks/test_contact_forces.py`: **7 passed** in 1.88 s.
- `BetterHuman/tests/bodies/test_mhr.py`: **21 passed** in 26.52 s, including CUDA.
- Warnings were existing autodiff-fallback and float32-gradcheck warnings; there were no test
  failures.

## MHR dynamics evidence

Artifact: `plan/for_agents/artifacts/validate_mhr_dynamics.py`

Model: LOD1, fp32, `nq=132`, `nv=131`, `njoints=203`, total mass
`81.481842 kg`. The run used
`/data3/rikhat.akizhanov/better/BetterHuman/models/MHR/converted/mhr_lod1.npz`.

| SAM anchor | MHR70 | Native joint (index) | BetterRobot joint (id) |
|---|---:|---|---|
| left wrist | 62 | `l_wrist` (78) | `l_wrist_ry` (139) |
| right wrist | 41 | `r_wrist` (42) | `r_wrist_ry` (87) |
| left ankle | 13 | `l_foot` (4) | `l_lowleg_twist` (8) |
| right ankle | 14 | `r_foot` (20) | `r_lowleg_twist` (31) |

CPU assertions and measured values:

- Static root-force norm: `799.336914 N`, equal to `mass * 9.81`.
- Two-foot world-up support, rotated into each joint's local frame: root linear residual
  `0.00012555 N` (`1.57e-7` of static). The remaining `52.2004 Nm` torque is expected because
  arbitrary point supports need not cancel the full moment.
- Backpropagation through a leaf `fext` is finite and nonzero; full gradient norm
  `22726.47`, extremity gradient norm `3197.76`.
- Nine-frame manifold-differenced root bob is finite; maximum dynamic wrench norm
  `1574.285`, versus static `799.342`.
- Non-neutral identity `linspace(-0.5, 0.5, 45)` changes recomputed mass from
  `81.481842 kg` to `73.100899 kg`, confirming that `with_shape` recomputes inertias.

## CUDA benchmark

GPU: NVIDIA RTX 6000 Ada Generation, fp32. Times are synchronized wall-clock medians from
five measured iterations after one warmup; parentheses show inclusive Q1–Q3.

| Workload | B=40 | B=128 |
|---|---:|---:|
| RNEA forward + backward through `fext` (includes internal FK) | 443.807 ms (442.789–444.146) | 442.970 ms (442.690–451.626) |
| Explicit FK + RNEA forward + backward through `fext` | 579.769 ms (579.690–580.844) | 583.712 ms (582.027–584.698) |

Separate `with_shape` forward for one identity, including neutral FK and inertia
recomputation: **136.307 ms** (136.241–136.501).

## Files produced

- `plan/for_agents/artifacts/validate_mhr_dynamics.py`
- `plan/for_agents/01_results.md`
- Short completion log added to `plan/for_agents/01_dynamics_deps.md`

No contact-anything source/config/test file and no external BetterRobot/BetterHuman source
file was changed.
