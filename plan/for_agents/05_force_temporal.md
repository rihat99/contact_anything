# Step 05 — Force temporal block

Depends on step 04. Read `plan/README.md` D11 and §2 (temporal module).

Small step: give the force tokens their own temporal attention, reusing the existing class.

## Tasks

1. `sam3d_body.py::_initialze_model`: when `MODEL.FORCE_TEMPORAL.ENABLED`, construct
   `self.force_temporal = ContactTemporalModule(dim=DECODER.DIM, ..., placement="post_decoder")`
   from `MODEL.FORCE_TEMPORAL.*` (mirror the contact_temporal construction ~257–277 but
   **post_decoder only** — no between_layers/pre_decoder for force; do not pass image_dim
   machinery it won't use, if the ctor allows).
2. `forward_decoder`: apply `force_temporal` to the force token slice right before
   `ForceHead`, exactly as contact does at ~678–683, feeding the same
   `(seq_len, frame_pos_sec, frame_valid)` from `_contact_temporal_fields`.
3. `contact/model.py::_patch_model_cfg`: patch `MODEL.FORCE_TEMPORAL.{ENABLED, NUM_LAYERS,
   NUM_HEADS, MLP_RATIO, ATTEND, CAUSAL, DROPOUT, BOTTLENECK_DIM, POSITION_SCALE}` from a new
   `model.force_temporal` config section (defaults mirror `model.temporal`; `attend:
   per_token` default; no `placement` key — it is fixed).
4. `contact/config.py`: `model.force_temporal` in `DEFAULTS` + `configs/base.yaml` commented
   block. Validation: enabled requires `model.force_head.enabled`; reuse the existing
   temporal validation clauses (bottleneck divisibility etc.) — factor, don't duplicate,
   if the check is identical.
5. `contact/checkpoint.py::_arch_signature`: add force_temporal fields (slot left in step 04).
6. Naming note: the attribute MUST be `force_temporal` (trainable filter + warm-start
   allowance in step 07 key on the `force` substring).

## Tests

- CPU (`tests/test_temporal.py` additions or a small `tests/test_force_temporal.py`):
  zero-gamma exact identity for the force path (`torch.equal`), per_token no-limb-mixing —
  reuse existing helpers instead of copying them.
- Extend `tests/test_force_invariance.py` (from step 04): with force_temporal enabled and
  gammas randomized, MHR outputs and contact logits still have zero Jacobian w.r.t. all
  force params (now including `force_temporal.*`), and force outputs move across frames.
- Extend grad-flow: every `force_temporal` param receives a nonzero grad when a dummy loss
  reads `out["force"]["joint_forces"]` across a T>1 clip with randomized gammas — AND with
  the force head's zero-init final layer randomized: a zero final layer blocks all upstream
  grads (including `force_temporal.*`) regardless of gammas, so the test would fail
  spuriously at true init.

## Out of scope

Joint attend mode across contact+force tokens, between_layers/pre_decoder placements,
any loss/trainer work.
