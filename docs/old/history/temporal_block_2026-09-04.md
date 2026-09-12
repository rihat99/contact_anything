# Making the temporal block denoise (2026-09-04 evening)

**Status (2026-09-05 cleanup):** of §3 only `gate_init: zero_proj` survives in the code (arm A,
`configs/tb_projzero.yaml`, run `output/tb_projzero_20260904_225421`). The locality bias, the
mixing path, `model.init_checkpoint` / `init_skip` and `optim.freeze` were removed (a full patch of
the pre-cleanup code is in `/data3/rikhat.akizhanov/trash/cleanup_20260905/code_before_cleanup/`); arms B, C, D, E and their configs are in
`/data3/rikhat.akizhanov/trash/cleanup_20260905/runs_removed/` and `configs_removed/`, F / H in `/data3/rikhat.akizhanov/trash/runs_mix_crossslot_20260905/`.

Companion to `docs/camera_ray_2026-09-04.md`. That day's runs fixed the DEPTH share of the jitter
(ray camera head on the frozen prior) and left the pose-readout share: on the 16-scene static test set
(GT jitter 6.35) R1 ends at 56.7 and R3 at 51.3, with root rotation (~34) and articulation (32-42)
as the remaining sources — white per-frame token noise created in the backbone maps. The user's
instruction for this round: no explicit smoothing anywhere (no output kernel, no prior kernel, no
map averaging in the loader) — the MODEL has to learn the averaging; small careful arms; GPUs with
low utilisation and enough memory.

## 1. Why the block did not denoise

`scripts/diag_temporal.py` on R1 (epoch 5): cross-frame attention mass 0.99, ~95 effective frames
out of the 120-frame eval clip, mean |dt| 1.03 s, gates |γ_attn| 0.14 L2 / 0.012 max. The block is a
near-uniform CLIP pooler with a tiny gate — a global bias correction, not a local filter. Three
mechanisms, all structural:

1. **Zero-gate parametrisation.** `x + γ·proj(attn)` with `γ = 0` and a RANDOM `proj`. At init the
   gradient reaches only γ (through the random projection of a uniform average, an almost useless
   direction) and `proj`/`qkv` get exactly zero gradient (verified: |∇proj| = 0, |∇γ| = 12 on a
   toy block). Learning the value/projection path is second-order — it has to wait for γ to open,
   and γ stays small because its gradient direction is noise. Every zero-gate run's gates ended
   ~0.01 (b8_lr2, R1, R3).
2. **No locality.** With random q/k the softmax over 60 × 7 keys is near-uniform; a decaying
   kernel in |dt| would have to be built out of coordinated RoPE phases across many frequencies —
   learnable in principle, slow in practice, and it never happened in ~1500 steps.
3. **The per-frame head outruns it.** The SMPL-X head gets a first-order gradient from step one and
   sharpens on the noisy token; the articulation share of the jitter GREW during R3 (32 → 42) as
   mpjpe improved (83 → 65 mm).

## 2. Is token-space averaging enough? (probe, no training)

`scratchpad camera/token_probe.py`: the FROZEN model's final pose token (decoder output, index 0)
averaged in time with a Gaussian (σ s, valid frames of the clip) and re-read by the FROZEN MHR +
camera heads. Same statistics as the crop probe (§5.2 of the camera doc): RMS third differences
(d3) of the outputs, GVHMR jitter of the world-lifted MHR keypoints.

| what is averaged | crop track | depth d1 / d3 (%) | bearing d3 | rot d3 (°) | kp d3 (mm) | world jitter | depth err (mm) |
|---|---|---|---|---|---|---|---|
| nothing (frozen model) | raw | 0.543 / 1.365 | 0.249 | 3.99 | 67.2 | 126.2 | 99 |
| backbone maps, σ 0.08 s | raw | 0.827 / 1.989 | 0.388 | 0.35 | 13.1 | 114.7 | 102 |
| backbone maps, σ 0.08 s | σ 1.0 s | 0.281 / 0.215 | 0.034 | 0.25 | 12.2 | 19.5 | 95 |
| **pose token, σ 0.08 s** | raw | 0.903 / 2.076 | 0.401 | **0.14** | **2.9** | 116.8 | 104 |
| pose token, σ 0.12 s | raw | 1.016 / 2.086 | 0.400 | 0.08 | 1.8 | 117.5 | 109 |
| **pose token, σ 0.08 s** | σ 1.0 s | 0.258 / 0.199 | 0.031 | 0.14 | 2.8 | **14.2** | 96 |
| pose token, σ 0.12 s | σ 1.0 s | 0.236 / 0.198 | 0.030 | 0.08 | 1.7 | 13.6 | 95 |
| pose token, σ 0.2 s | σ 1.0 s | 0.210 / 0.195 | 0.028 | 0.05 | 1.0 | 13.0 | 95 |
| GT (SMPL-X camera-frame root, 22 root-relative joints) | | 0.278 / 0.143 | 0.060 | 0.34 | 6.7 | 6.35 | 0 |

Reading:
* A LINEAR local average of the pose token — exactly what an attention layer with a local kernel
  and identity value/projection computes — removes the rotation noise 28× and the articulation
  noise 23× (better than averaging the maps: the token → pose readout is close to linear). With a
  smooth crop track the frozen model goes from 126 to 14 with no training and no depth-accuracy
  cost. **The block can do this in principle; it just never learned to.**
* At σ ≥ 0.08 s the token average already OVER-smooths rotation and articulation relative to the
  GT's own third differences (0.14° vs 0.34°, 2.9 vs 6.7 mm) while the world jitter keeps falling
  — the jitter metric is dominated by residual noise, so it rewards width; a content-adaptive
  kernel (attention) could keep real motion and still remove the noise.
* Depth / bearing are the crop-alignment problem again (raw crops: worse; smooth crops: fixed).
  Under the ray head the depth is `frozen prior(t) + Δ(token)`, so cancelling the prior's white
  noise needs the head to SUBTRACT the frame's own measurement from the averaged one — the
  residual stream carries the frame's own `frozen_camera` embedding and the attention branch the
  averaged one, so it is representable (R1 learned it partially: residual anti-correlated with the
  frozen noise), but it is a harder function than "read the averaged token".

## 3. What was built (model/rope.py, train/trainer.py, train/checkpoint.py, configs/base.yaml)

All learnable; no fixed kernel anywhere.

* **`gate_init: zero_proj`** — the block keeps `x + γ·proj(attn)` but starts with `γ = 1` and
  ZERO `proj` / FFN-output weights (the residual-branch zero-init of Fixup / ControlNet / GPT-2
  scaling). Still an exact identity at init (`torch.equal` verified for both gate forms, with and
  without the locality prior); the projections now get a first-order gradient from step one
  (toy block: |∇proj| 147, |∇ffn_out| 694 vs 0 / 0 under `zero_gate`).
* **`locality: {enabled, init_sigma_sec, init_self_bias}`** — a learnable Gaussian-in-time logit
  bias per head, `logit += −dt² / (2 σ_h²) + b_h·[same token]`, `dt` = real elapsed seconds
  (exact under the corpus's variable fps, same positions RoPE uses), combined with the keep-mask
  (`-inf` on hidden keys). The model can widen σ_h to "no prior" or sharpen it; σ / b train at
  `optim.locality_lr_scale` (20×, the `smoother_lr_scale` argument: O(1) scalars Adam moves ~lr per
  step). Verified against a manual recompute (max |diff| 0), invalid / out-of-window keys keep
  exactly zero mass, heads with σ 0.05 / 0.1 / 0.2 / 1.0 s show the expected local-to-flat
  profiles. `scripts/diag_temporal.py` recomputes the attention with the bias (SDPA-verified) and
  prints σ / self-bias per layer and the projection norms next to the gates.
* **`model.init_checkpoint` / `init_skip`** — warm start by name from another run's checkpoint
  (shapes must match, names absent keep their init, `init_skip` prefixes are left fresh); NOT a
  resume (optimizer / epoch start at 0).
* **`optim.freeze: [prefixes]`** — an lr-0 parameter group: the weights stay "trainable" (EMA,
  checkpoint, evaluate/dump scripts unchanged) but never move; the scheduler factor of an lr-0
  group is 0. Prefixes matching nothing hard-error.

* **`mix: {enabled, init_logit}`** (added after the round-1 epoch-5 read, §5.1) — a convex temporal
  MIXING path on the residual stream itself: `x ← x + g ⊙ (Σ_k w_k x_k − x)` with `w` the block's own
  (locality-biased, content-modulated) attention weights and `g = sigmoid(logit)` per channel
  (init −4 = 1.8 %). Rationale: a residual softmax block can only ADD a weighted average of
  projected tokens; it cannot subtract its own token's noise, so the best it can do is dilute
  (`x + c·avg`, the head rescaling by `1/(1+c)`), which needs `c → ∞` against weight decay. With
  `g → 1` the path REPLACES the token by its attention-weighted average — verified to reproduce the
  probe's Gaussian average exactly (max |diff| 3e-7 with the content logits off), first-order
  gradient to the gate at init (|∇| 19), the attention recompute unchanged. Trains at
  `locality_lr_scale` like the other kernel scalars.

## 4. Arms (all on the static_ray recipe, 30 epochs, 2 clips/step on one GPU, 16-scene test)

| arm | config | what it tests |
|---|---|---|
| A `tb_projzero` | `gate_init: zero_proj` | the gate parametrisation alone — does the block find locality itself once its projections can learn? |
| B `tb_loc` | zero_proj + locality σ init 0.1 s | the locality prior at the probe's width |
| C `tb_locwide` | zero_proj + locality σ init 1.0 s | locality LEARNED, not imposed: init near-uniform; does σ shrink toward ~0.1 s by itself? |
| D `tb_window` | zero_gate (R1's block) + `max_rel_sec: 0.25` | a hard ±0.25 s window on the unchanged block |
| E `tb_stage2` | warm start from R1 final, every head / token / input at lr 0, FRESH zero_proj + locality block | two-stage: the loss can only improve by denoising the token |
| F `tb_mix` | B + the convex mixing path (init logit −4) | round 2: can the block REPLACE its token by the local average? |
| G `tb_mix_noloc` | zero_proj + mixing, no locality prior | round 2: does content attention find the kernel once it can replace? |
| H `tb_mix_stage2` | E's warm start + freeze, fresh zero_proj + locality + mixing block | round 2: the pure block test with the replace path |

Success bar (user): jitter "very close to GT" (6.35 on this set) — "70-90 is absolutely bad, 60 is
not a win". Judge every arm by the block diagnostic (effective frames, gate / projection norms, σ)
and the oracle decomposition (rotation / articulation shares) next to the jitter, so a lower
jitter is attributed to the block and not to the head.

## 5. Results

### 5.1 Round 1 at epochs 5-8 (EMA weights, 16-scene test; R1 for reference: 73.4 at epoch 6, 68.0 at 8)

| arm | jitter ep 5 / 8 | mpjpe ep 8 | depth_err ep 8 | dlogz_pred ep 8 | block at epoch 5 (scratchpad temporal/ckpt_block_stats.py) |
|---|---|---|---|---|---|
| A projzero | 64.1 / 54.6 | 104 | 119 | 0.208 | γ ≡ 1, ‖W_proj‖ 4.4, ‖W_ffn_out‖ 5.9 (R1 zero_gate at ep 5: ‖γ‖ 0.146, max 0.012, random ‖W_proj‖ 18.8 → effective ≤ 0.2) |
| B loc σ0.1 | 80.9 / 60.9 | 105 | 129 | 0.278 | σ drifted WIDER 0.10 → 0.11-0.22 s; self bias NEGATIVE (−0.1 … −0.8) |
| C locwide σ1.0 | 64.0 / 55.1 | 108 | 119 | 0.208 | σ drifted wider still, 1.0 → 0.9-1.4 s; self bias negative |
| E stage2 (R1 heads frozen, fresh block) | 64.6 / 60.2 (59.2 at 9) | 72 | 106 | 0.223 | σ 0.13-0.25 s, self bias negative; jitter flat at ~60 from epoch 6 |

Reading: the zero-projection block writes 20× more than R1's ever did and all arms are ahead of R1
at the same epoch, but (i) every locality width drifts WIDER, never toward the probe's 0.1 s, and the
self bias goes negative in every layer — the block wants the NEIGHBOURS' average as an additive
signal on top of its own token; (ii) with the heads frozen (E) the block alone plateaus at ~60. Both
are what a residual softmax block can do at best: `x + c·avg` dilutes the token's noise by `1/(1+c)`
per layer, it never subtracts it — the "replace" operation the probe validated is not in its
function class. Hence the mixing path (§3, arms F-H).

### 5.2 Mixing arms at epoch 5

| arm | jitter ep 0-5 | mix gate g (mean / max) at ep 5 | σ (s) | self bias |
|---|---|---|---|---|
| F tb_mix | 86.8, 85.6, 87.7, 85.0, 82.4, 74.8 | 0.016-0.021 / 0.028 (init 0.018) | 0.10-0.21 | −0.7 … +0.1 |
| H tb_mix_stage2 | 85.3, 76.4, 76.4, 67.6, 65.8, 65.6 | 0.016-0.022 / 0.040 | 0.11-0.28 | −1.3 … +0.1 |

The mixing gate does NOT open — after ~250 steps at 20× lr (Adam could have moved the logit by ~1
unit) it sits at its init in both arms, heads training or frozen. So at these operating points the
loss does not reward replacing the token by its attention-weighted local average, or its gradient
on the gate is sign-inconsistent. Two suspects under test: the gate is shared by the pose AND the six
contact slots of a frame (per-frame contact labels may pull against averaging the contact tokens —
added `mix.per_slot` = one gate row per slot), and the benefit may not exist at all at the trained
head's operating point (forced-gate sweep on the stage-2 checkpoint, §5.3).

### 5.3 Forced-gate sweeps on the stage-2 mixing arm (epoch 5; scratchpad temporal/force_mix_eval.py, force_local_eval.py)

Setting the gate to `g` by hand on the R1-heads-frozen checkpoint (test protocol; trained = 64.9 jitter / 70.3 mpjpe):

| forced mixing | jitter | mpjpe | dlogz_pred | F1 |
|---|---|---|---|---|
| all slots, g = 0.25 / 0.5 / 1.0 (block's own attention) | 77 / 91 / 98 | 150 / 197 / 207 | 0.41 / 0.50 / 0.54 | 0.86 / 0.69 / 0.63 |
| pose slot, layer 0, content logits OFF, Gaussian σ 0.05 / 0.1 / 0.2 s, g = 1 | 103 / 99 / 99 | 290 / 291 / 295 | 0.54 | 0.85 |
| block bypassed (γ = 0, no mixing) | 92.9 | 68.9 | 0.44 | 0.61 |

Replacing the token by the block's attention average is DESTRUCTIVE, and so is a "local" σ 0.05 s
average — the opposite of the probe. Cause (found from this table): the mixing weights ran over all
`T × K` tokens, so the "local average of the pose token" was an average of the pose token WITH the six
contact tokens of the neighbouring frames (the locality bias knows frames, not slots; only the
content logits can tell slots apart). The probe averaged pose tokens with pose tokens. That also
explains the closed gate: with cross-slot weights, opening it is always harmful. Fixed after the
sweep: the mixing path is now SAME-SLOT (attention row restricted to the query's own slot and
renormalised — a temporal filter per token; verified to reproduce the per-token Gaussian average
exactly and to leave the other slots untouched). The fixed path has NOT been trained: the user
stopped all experiments at this point (2026-09-05 ~00:40). The two cross-slot mixing runs are in
`trash/runs_mix_crossslot_20260905/`.

### 5.4 Final numbers of the round-1 arms (stopped at epoch 27-29 of 30, EMA; R1 final 56.7 / 65.2 / 116 / 0.852)

| arm | jitter (min → last) | mpjpe | depth_err | dlogz_pred | F1 | block at epoch 15 |
|---|---|---|---|---|---|---|
| A projzero | 50.5 (ep 13) → 52.4 | 72.3 | 100 | 0.158 | 0.874 | ‖W_proj‖ 7.1, ‖W_ffn_out‖ 10.7 (final) |
| B loc σ0.1 | 50.1 (ep 18) → 50.8 | 72.7 | 103 | 0.176 | 0.877 | σ 0.13-0.52 s (layer 0 median 0.38), self bias −0.3 … −1.9 |
| C locwide σ1.0 | 51.2 → 52.1 | 72.6 | 103 | 0.158 | 0.895 | σ 0.86-1.85 s (wider than init), self bias −0.4 … −2.5 |
| E stage2 (R1 heads frozen) | 56.8 (ep 17) → 57.0 (ep 29) | 65.5 | 114 | 0.177 | 0.862 | σ 0.15-0.64 s, self bias −0.1 … −2.7 |
| D window 0.25 s (zero_gate) | 81.6 (ep 0) → 60.0 (ep 29, monotone) | 68.9 | 115 | 0.251 | 0.841 | ran to completion from the queue after the stop (00:06-01:13, unnoticed); worse than R1's 56.7 — a hard local window does not make the zero-gate block denoise |

Reading: the zero-projection parametrisation is worth ~5 jitter points over R1 (52 vs 57) and
converges in ~13 epochs instead of never; the locality prior adds nothing on top (A ≈ B ≈ C), and
in every arm the learned kernel WIDENS (to 0.2-0.5 s, or 1.3-1.8 s from the wide init) with a
strongly negative self bias: the block settles on "add the neighbours' average, excluding myself" —
the dilution regime — and the jitter floor of that regime is ~50 with trainable heads, ~57 with R1's
frozen heads. Nothing in round 1 approaches the probe's 14 (token replaced by a local same-token
average) or the target 6.35.

## 6. Where this stands (2026-09-05 00:45, all experiments stopped by the user)

1. **The block can denoise in principle** — a same-token local average of the pose token is enough
   (frozen model 126 → 14 with a smooth crop track; rotation/articulation noise 28×/23× down).
2. **The residual softmax block never learns it** because it can only ADD; the best additive
   strategy is dilution (`x + c·avg_neighbours`), which is exactly what all arms converged to
   (self bias negative, kernels wide), floor ~50-57 on the 16-scene set.
3. **A replace path is needed and is now built correctly** (`mix`, same-slot, per-slot gate
   option, exact at g = 1) but untrained. The first mixing attempt mixed slots and was rightly
   refused by the loss; that is why its gate never opened — not evidence against replacement.
4. **Next experiment** (not launched): F' = zero_proj + locality σ 0.1 + same-slot mixing with
   `per_slot: true` (pose gate free, contact gates free), and H' = the same on R1's frozen heads.
   Watch the pose-slot gate: if it opens toward 1 with σ ~0.1 s, the block has learned the probe's
   filter; if it stays closed under a same-slot kernel, the loss itself does not reward it and the
   supervision (per-frame kp2d/kp3d/orient) has to change before any block can.
5. **Cleanup (2026-09-05, user request):** everything that did not work or was built for one
   arm was removed from the code — locality bias, mixing path, warm start / freeze — and the
   failed arms trashed (see the status note at the top). `gate_init: zero_proj` stays as the one
   gain. Reviving F'/H' means restoring the same-slot mixing path from
   `/data3/rikhat.akizhanov/trash/cleanup_20260905/code_before_cleanup/model/rope.py`.
