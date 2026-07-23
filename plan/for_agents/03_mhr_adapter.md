# Step 03 — MHR adapter: frozen-model outputs + cameras → BetterHuman body + world-frame q

Depends on steps 01 (env wired; **read `01_results.md`** — joint-id resolution caveat) and
02 (batch carries `cam_from_world`/`gravity_world`). Read `plan/README.md` §2–§5
(especially D7, D13, R2–R4, and the 204-param layout warning in §7).

Deliverable: `contact/physics/__init__.py` + `contact/physics/adapter.py` — the single place
that knows how SAM 3D Body's outputs map onto BetterHuman's MHR and how camera extrinsics
place the body in the reconstruction world. Everything is detached (`torch.no_grad()`
semantics — gradients never flow into the frozen model; the physics loss in step 06 receives
q/v/a as constants and only the predicted forces carry grads).

## Inputs (batch is flat `B = B_clips*T`, clip-major)

From `out["mhr"]` (produced by `sam_3d_body/models/heads/mhr_head.py::MHRHead.forward`
~346–367 and the camera head):
- `mhr_model_params [B,204]` — the vector actually fed to SAM's MHR forward:
  `cat([global_trans (3), global_rot euler (3), body_pose (130)]) (=136), scales (68)`.
  **`global_trans` is zeroed** (`mhr_head.py` ~296) — slots `[0:3]` carry no information;
  the real root translation exists only as `pred_cam_t`. Do not hunt for a translation here.
- `shape [B,45]` → `MHRClassic.identity_coeffs`. `face [B,72]` is zeroed → expression=None
  or zeros (verify which BetterHuman expects for `use_expression=True` vs constructing the
  body with `use_expression=False` — prefer the latter, simpler).
- `global_rot [B,3]` (euler ZYX, camera frame), `pred_cam_t [B,3]` (camera-frame root
  translation), `focal_length [B]`.
- `pred_joint_coords [B,127,3]` — SAM's own MHR joint centers, **camera frame after the
  axes-1,2 sign flip** (mhr_head ~340–343). This is your ground truth for the acceptance
  test.

From the batch (step 02): `cam_from_world [B,4,4]` (camera-from-world, OpenCV, metric),
`cam_valid [B]`. Callers only invoke the adapter on clips where `cam_valid` is all-True.

## The mapping problem (do not guess — verify)

SAM composes 204 as `[6 root | 130 pose | 68 scales]`; Momentum/BetterHuman partitions the
same 204 slots as root `[0:6]`, pose `[6..129, 151]` (125), proportions `[130..150, 152..203]`
(73) — see `BetterHuman/src/better_human/bodies/mhr/archive.py::parameter_partitions`.
These are the SAME 204-slot vector if SAM's MHR is the same Momentum compact_v6 rig — in
that case `MHRClassic(identity=shape, model_parameters=mhr_model_params)` handles the body
channels and the partition difference is only bookkeeping. But verify against
`sam_3d_body`'s vendored MHR implementation (follow what `mhr_model_params` feeds into) and
let the FK test decide. If slot ordering differs, build the permutation explicitly with a
comment citing both sources.

**Root rotation hazard (no permutation can fix this)**: SAM's `global_rot` is euler **ZYX**
(`roma.rotmat_to_euler("ZYX", ...)`, `mhr_head.py` ~293–295), while `from_classic` builds
the root via `root_parameter_transform` + `so3.from_euler` in ITS convention
(`BetterHuman/.../bodies/mhr/body.py` ~267–268). Recommended: sidestep root channels
entirely — feed `MHRClassic` with zeroed root slots to get the body-pose q, then overwrite
the free-flyer block directly: `q[..., 0:3]` and `q[..., 3:7]` (scalar-last quaternion)
from the world-frame composition below, going rotation-matrix → quat, never through euler
round trips.

## Frame composition (decision D7)

Three frames are in play:
1. **MHR native** (y-up, meters) — BetterHuman's world; `joint_global_rots` lives here.
2. **Camera** (OpenCV: x right, y down, z forward) — SAM reports points here via
   `flip = diag(1,-1,-1)` applied to native-space points, plus `pred_cam_t`:
   `X_cam = D @ X_native + pred_cam_t`, `D = diag(1,-1,-1)` (det=+1, a rotation).
   Read `camera_project` (`sam3d_body.py` ~1101–1154) to pin the exact composition.
3. **Reconstruction world** (metric, static per scene, ≈ camera-0 frame) — the physics
   world. Per frame t: `T_w←c[t] = inv(cam_from_world[t])`, i.e.
   `R_w←c = R_ext.T`, `t_w←c = -R_ext.T @ t_ext` (use the transpose form, not a generic
   4×4 inverse).

Free-flyer root per frame:
```
R_root_world[t] = R_w←c[t] @ D @ R_root_native[t]
t_root_world[t] = R_w←c[t] @ pred_cam_t[t] + t_w←c[t]
```
where `R_root_native` comes from `global_rot` (via rotation matrix, see hazard above).
Work it out on paper once, implement once, and let the FK test judge: adapter FK world
joints, mapped back to camera space via `cam_from_world[t]`, must equal
`pred_joint_coords[t] (+ pred_cam_t[t], per SAM's exact composition)`.

Gravity is NOT the adapter's business — the loss reads `batch["gravity_world"]` directly
(step 06). Sanity expectation to note in the test: with frame-0-anchored worlds a standing
climber's feet are at larger +y than their head (OpenCV y-down world).

## API (keep it this small)

```python
class MHRAdapter:
    def __init__(self, model_path: str | None, lod: int, device, dtype=torch.float32): ...
        # loads bh.MHR once; resolves the 4 extremity BR joint ids BY NAME (order:
        # left_hand, right_hand, left_foot, right_foot)
    @torch.no_grad()
    def q_from_mhr_out(self, mhr_out: dict, cam_from_world: Tensor, n_clips: int,
                       seq_len: int) -> tuple[ShapedBody, Tensor]:
        # (shaped body, q [n_clips, T, 132]) in the reconstruction world
```

plus attributes the loss needs: `extremity_joint_ids [4]`, `robot` (BetterRobot Model of the
shaped body), and per-clip total mass (from body inertias — these track shape, since
`with_shape` recomputes them from the shaped mesh). Shape/scale vary per frame in principle;
use the clip's center frame (or mean) shape for the body — one body per clip, q per frame.
`model_path=None` resolves like step 01's validator: `$BETTERHUMAN_MODELS_DIR`, then the
sibling BetterHuman checkout.

**Extremity ids (from `01_results.md` — resolve by NAME, not by MHR70 number)**: the MHR70
indices `[62,41,13,14]` are SAM keypoint indices, not BetterHuman joint indices. MHR has no
native `l_ankle`/`r_ankle`; the ankle anchors map to the nearly coincident native `l_foot`/
`r_foot` origins. Reference resolution (LOD1): `l_wrist`(78)→BR `l_wrist_ry`(139),
`r_wrist`(42)→BR `r_wrist_ry`(87), `l_foot`(4)→BR `l_lowleg_twist`(8),
`r_foot`(20)→BR `r_lowleg_twist`(31). Follow the name-based resolution in
`artifacts/validate_mhr_dynamics.py`; assert the resolved *names*, let numeric ids float.

**Batched-shape × time-axis gotcha (will error if ignored)**: a per-clip shaped body has
batched value tables (`body_inertias [n_clips, nbodies, 10]`, etc.). BetterRobot's execution
batching right-aligns batch axes, so q of shape `[n_clips, T, nq]` against `[n_clips, ...]`
values FAILS with an explicit error (`BetterRobot/src/better_robot/data_model/
execution_batch.py` ~94–102: "add a singleton time axis"). The adapter must return the
shaped robot with a singleton time axis inserted in its batched values
(`[n_clips, 1, ...]`), so downstream FK/RNEA broadcasts over T. Test this path — the
BetterRobot reference task never exercises batched shapes.

Keep helper functions module-private. No config plumbing yet (step 06 wires `physics:`
config to this constructor).

## Tests — `tests/test_physics_adapter.py`

1. **CPU, synthetic**: round-trip sanity — build a q, `to_classic`, feed through the adapter
   math with synthetic `cam_from_world` (include a non-identity rigid transform), get the
   same q back (guards the permutation/translation handling AND the w2c inversion).
2. **GPU + real checkpoint + real dataset clip, `@pytest.mark.slow`** (mirror the fixture
   pattern of `tests/test_temporal_invariance.py`; needs step 02's backfilled dataset): run
   the frozen model on a real clip, adapter → `body.fk(q)` → `native_joint_poses` → map to
   camera space via `cam_from_world` → compare to the model's own `pred_joint_coords`
   (+ camera translation). Tolerance: start at 5 mm mean / 2 cm max per joint; if the rig
   differs slightly (correctives), report the achieved numbers and justify the final
   tolerance in the test comment. **This test is the step's definition of done** — if it
   can't be made to pass, stop and report the discrepancy rather than loosening the
   tolerance past usefulness.
3. Extremity ids: assert the 4 resolved joints' FK positions match the model's MHR70
   keypoints `[62,41,13,14]` (`pred_keypoints_3d`, mapped through the same frames) within
   the same tolerance.

Follow repo test conventions (`tests/` layout, float32, no skipped edge cases — see the
write-tests notes in existing tests). Fast suite must stay green.

## Out of scope

- Any force/loss code (step 06), any model-side changes (step 04), config schema (step 06).
- Gravity estimation, per-frame shape variation, smoothing (loss-side, step 06).
