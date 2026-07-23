# Step 04 — Force tokens, force head, freeze/eval-pin generalization

Independent of steps 01–02. Read `plan/README.md` §2 (contact token mechanism), decisions
D1–D5, §7. This step touches the vendored `sam_3d_body` (same hook style as the contact
additions — delimit new blocks with `# --- force hook ---` comments) and `contact/`.

## Model side (`sam_3d_body/models/meta_arch/sam3d_body.py`)

Mirror the contact token machinery, gated on a new `cfg.MODEL.DECODER.DO_FORCE_TOKENS`:

1. `_initialze_model` (contact block at ~207–256): create `self.force_embedding`
   (`nn.Embedding(num_force_tokens, DECODER.DIM)`), `self.force_posemb_linear`,
   `self.force_feat_linear`. `num_force_tokens = len(self.contact_keypoint_indices)` —
   force anchors ARE the contact anchors (D2); require `DO_CONTACT_TOKENS` when
   `DO_FORCE_TOKENS` (assert with a clear message). No global force tokens.
2. `forward_decoder` (~566–599): append force tokens **after** contact tokens; record
   `force_emb_start_idx`. Extend the asymmetric mask (D1): keep the existing contact line
   and add `token_mask[:, :force_emb_start_idx, force_emb_start_idx:] = False` — original
   AND contact tokens never attend force tokens; force tokens attend everything.
3. `contact_token_update_fn` (~2126–2260): apply the same anchored update (2D-keypoint
   posemb + grid-sampled features, same validity masking) to the force tokens using the
   force linears. Factor the shared sampling logic rather than copy-pasting ~80 lines —
   e.g. extract a private helper taking (start_idx, count, posemb_linear, feat_linear).
   Do not change the contact path's numerics.
4. Head: new `sam_3d_body/models/heads/force_head.py::ForceHead` — per-token FFN mapping
   `[B, K, C] -> [B, K, 3]` (mirror `ContactHead`'s `per_token` branch; output dim 3, no
   sigmoid). **Zero-init the final linear** (D5: model starts predicting zero force).
   Register in the heads registry like `contact_head`. In `forward_decoder` (~672–698):
   slice force tokens, apply head, emit `out["force"] = {"joint_forces": [B, 4, 3]}`
   (dimensionless, units of body weight — D5). Wire `out["force"]` through the same output
   paths that carry `out["contact"]` (main forward ~1376–1383 and the prompt-iter paths —
   grep every place `out["contact"]` is set).

## `contact/` side

5. `contact/model.py::_patch_model_cfg`: patch `MODEL.FORCE_HEAD.{MLP_DEPTH,
   MLP_CHANNEL_DIV_FACTOR, DROPOUT}` + `MODEL.DECODER.DO_FORCE_TOKENS` from the run config
   (mirror ~51–66). The `frame` key is consumed by the physics loss (step 06), not the model.
6. **Freeze filter** (D3): `_trainable_name_filter` → matches if `"contact"` or `"force"`
   in the dotted name. **Eval-pin** (D3): rewrite `pin_frozen_eval` to derive the toggled
   set from `requires_grad` at call time — on `train(mode)`: pin all modules eval, then set
   `.training = mode` for every module in each subtree whose root **recursively owns ≥1
   trainable param**, propagated top-down. The propagation matters: heads contain param-less
   `nn.Dropout` children (e.g. FFN dropout, `contact_head.dropout: 0.1` default) which must
   toggle with their trainable parent — a rule keyed on "modules with direct trainable
   params" would silently disable dropout in trainable heads. With the subtree rule,
   contact-only configs behave exactly as today (assert in tests: after `train(True)` on a
   contact-only build, the module-wise `.training` map equals the current implementation's),
   and in regime (a) a frozen contact head keeps dropout off.
7. **`train.freeze_contact`** (D4): config key (default false). In `build_model`, after the
   normal unfreeze, set `requires_grad=False` on params whose name contains `"contact"`
   (force params, named `force_*`, don't match). Validation: `freeze_contact: true` requires
   `model.init_contact_checkpoint` set and `model.force_head.enabled: true`.
8. `contact/config.py`: add `model.force_head: {enabled: false, frame:
   "local_world_aligned", mlp_depth, mlp_channel_div_factor, dropout}` to `DEFAULTS` (with
   comments, matching `configs/base.yaml` style — also add the commented block to
   `configs/base.yaml`). Semantics in `_validate_semantics`: force_head requires the joint
   target enabled with `joint_set: extremities_4` and `pool_mode: per_token`; `frame` in
   the two allowed values.
9. `contact/checkpoint.py::_arch_signature`: include the force_head fields (and later
   force_temporal — leave a spot). Loss weights / `physics:` numbers must NOT enter the
   signature (changing a weight is not an architecture change). Heads-up for step 07: the
   warm-start path (`initialize_common_contact`) compares source vs target signatures after
   popping only `temporal` — once force keys exist here, that comparison must learn to
   exempt them too (step 07 owns that change; just don't be surprised that warm-start
   breaks until then).

## Tests

- CPU unit (`tests/test_force_head.py`): ForceHead shapes, zero-init → zero output,
  per-token independence (perturb token k → only row k changes).
- Config (`tests/test_config.py` additions): defaults load; validation failures for
  force-without-contact, bad frame, freeze_contact without init checkpoint.
- **Invariance, GPU `@pytest.mark.slow`** (extend `tests/test_temporal_invariance.py`
  pattern into a new `tests/test_force_invariance.py`):
  1. Exact Jacobian: grad of every MHR output w.r.t. every `force_*` param is None/zero;
     grad of `out["contact"]["joint_logits"]` w.r.t. every `force_*` param is None/zero
     (this is the D1 guarantee that regime (a) preserves contact behavior);
     grad of `out["force"]["joint_forces"]` w.r.t. force params is nonzero (sanity) —
     **randomize the zero-init final head layer first**: at exact zero-init the upstream
     force params (embedding/linears) have identically zero grad through the head, so the
     sanity check would fail spuriously. Test the final layer's own grad at zero-init
     separately.
  2. Noise-floor: enabling the force branch leaves MHR outputs and contact logits within
     the measured CUDA noise floor of the force-disabled model. (Noise floor, not
     `torch.equal` — the longer token sequence can change SDPA reduction order; the exact
     guarantee is the Jacobian, the forward agrees to noise only.)
- **Grad flow, GPU slow** (extend `tests/test_grad_flow.py`): with force enabled, every
  trainable param name contains "contact" or "force"; with `freeze_contact: true`, contact
  params have `requires_grad=False` and get no grads while all force params do; frozen-base
  params never get grads. Also assert eval-pin: after `model.train(True)` with
  freeze_contact, contact submodules report `.training == False`, force submodules `True`.

## Docs touchpoint

Update the "Invariants (do not break)" section of `CLAUDE.md`: freeze filter is now
"contact OR force" substring; eval-pin is requires_grad-derived; mask invariant now reads
"no earlier token block attends a later one (original ⊥ {contact, force}, contact ⊥ force)".

## Out of scope

- Temporal for force tokens (step 05), physics loss / `physics:` config (06), trainer &
  warm-start (07). `out["force"]` is produced but consumed by nothing yet — that is fine.
