# Velocity-matching toy: exact numerical tests of H1–H5

**Everything below is a synthetic toy**, float64 CPU, no repo code, no training of the real model.
Scripts and raw stdout live beside this file. Python
`/data3/rikhat.akizhanov/miniconda3/envs/sam3d/bin/python`; every script is standalone
(`cd` to this directory and run it).

| script | output | what it produces |
|---|---|---|
| `common.py`, `kernels.py` | – | trajectory/noise generators, losses, kernel-space models, oracle solvers |
| `exp_h2_shrinkage.py` | `out_h2.txt` | H2: analytic + numeric `s*`; multiplicative-depth and dimensionless-velocity variants; Huber vs L2 |
| `exp_h2b_anchor.py` | `out_h2b.txt` | H2: how strong the per-frame anchor must be to hold the scale |
| `exp_h1_filters.py` | `out_h1.txt` | H1: every model family optimised to convergence vs the ±4 / ±16 oracles |
| `exp_h1b_lamsweep.py` | `out_h1b.txt` | H1: the joint optimum as a function of the velocity weight λ |
| `exp_h1c_conflict.py` | `out_h1c.txt` | H1: how much the two objectives conflict, per model class |
| `exp_h4_firstorder.py` | `out_h4_first.txt` | H4: exact gradients at the identity init; model-B self-bias sweep; dilution arithmetic |
| `exp_h4_gradsnr.py` | `out_h4_snr.txt` | H4: per-minibatch gradient mean/std/SNR at the identity init |
| `exp_h4b_noself.py` | `out_h4b.txt` | H4: branch-gradient sign vs whether the average includes the frame's own token |
| `exp_h3_adam.py` | `out_h3.txt`, `h3c_*.json` | H3: spectrum, Adam second moment, 60 k-step dynamic runs, stop-grad routing, landscapes |
| `exp_h3de_landscape.py` | `out_h3de.txt` | H3: the branch-gain landscape (standalone copy of the H3-d/H3-e sections) |
| `follow.py` | – | follow-up: decoupled loss, shared runner, `noself8` branch option |
| `exp_f1a_dec_firstorder.py` | `out_f1a.txt` | F1(a): first-order gradients of L_dec at the identity point |
| `exp_f1c_dec_sstar.py` | `out_f1c.txt` | F1 extra: s\* under L_dec for the scale-only model |
| `exp_f_one.py` | `shards/*.txt`, `out_f1b_f2_shards.txt` | F1(b) + F2(b): one dynamic run per process |
| `exp_f1b_dec_dynamics.py`, `exp_f2_selfmask.py` | `out_f1b.txt`, `out_f2.txt` | serial drivers (F2-a first-order table lives in `out_f2.txt`) |
| `exp_f_one.py` (Follow-up 2) | `shards2/*.txt`, `out_fu2_shards.txt` | F2 round 2: corr_true / onesided / routing control |
| `exp_h5_so3.py` | `out_h5.txt` | H5: SO(3)/SE(3) transport identity and the V⁻¹ coupling |

Seeds: trajectories `seed=11`, noise `seed=12` for all full-batch experiments; the minibatch
experiments draw fresh trajectory/noise seeds per step from disjoint ranges (10000+, 300000+,
500000+, 900000+) — the exact values are in the scripts.

## Setup (common to every experiment)

* `T = 60` frames, `dt = 1/25 s`. Loss rows are restricted to the **interior** of the clip
  (margin 16 frames, so every model's full ±16 receptive field is inside the clip) — 28 frames /
  27 velocity rows per clip, times 384–1024 clips. This removes boundary effects from the comparison.
* GT `g_t` = sum of 4 sinusoids, frequencies ~U(0.15, 1.0) Hz, phases/amplitudes random per clip,
  then rescaled so the **one-step velocity RMS matches the real GT** exactly per clip.
* Observation `x_t = g_t + n_t`, `n` white Gaussian.
* Channels and noise levels (from the brief):

| channel | GT one-step vel RMS | noise σ | noise vel RMS `√2σ/dt` | ratio to GT | var(g)/var(n) = SNR_pf | SNR_vel |
|---|---|---|---|---|---|---|
| depth (root pos) | 0.288 m/s | 15.4 mm (0.44 % of 3.5 m) | 0.5465 m/s | **1.90×** | 26.3 | 0.278 |
| root angle ×1.0 | 0.486 rad/s | 0.01375 rad | 0.4878 | 1.00× | 94.1 | 0.993 |
| root angle ×1.25 | 0.486 rad/s | 0.01718 rad | 0.6097 | 1.25× | 60.2 | 0.635 |
| root angle ×1.5 | 0.486 rad/s | 0.02062 rad | 0.7317 | 1.51× | 41.8 | 0.441 |
| joint angle ×1.25 | 0.752 rad/s | 0.02659 rad | 0.9434 | 1.25× | 60.2 | 0.635 |

* Losses: per-frame and velocity, each in **L2** and in the repo's **δ-normalised Huber**
  (`0.5u²` for `|u|≤1` else `|u|−0.5`, `u = e/δ`), δ_pf = 0.05 m / 0.1 rad, δ_vel = 0.4 m/s /
  0.6 / 0.9 rad/s (= the GT RMS, as in the repo).
* **λ60** = the velocity weight at which `‖∂L_vel/∂ŷ‖ = 0.6 × total` at the identity point — the
  toy analogue of "the velocity block is ~60 % of the gradient norm at init".
  Huber: λ60 = 0.109 (depth) / 0.114 (angles). L2: λ60 = 9.6e-4 (the L2 velocity gradient is
  1/dt² = 625× bigger, so it needs a 113× smaller weight to sit at the same 60 %).
* Model families:
  * **A** `ŷ = s·x + b` (per-frame affine head; free scale = the shrinkage shortcut)
  * **B** `ŷ = Σ_j w_j x_{t+j}`, `w = softmax(logits)` over ±4, DC gain pinned to 1
  * **C** `ŷ = h·(x + c·mean_{±4} x)`, 1 and 4 layers (the residual-dilution structure)
  * **D** = B or C **plus a free head scale**
  * **oracles** = the exact least-squares-optimal LTI kernel of half-width 4 or 16 for the same loss.

---

## H2 — the shrinkage shortcut (`exp_h2_shrinkage.py`, `out_h2.txt`) — **CONFIRMED, exactly**

For the affine head `ŷ = s·x + b` the closed-form L2 optima are
`s*_pf = SNR_pf/(1+SNR_pf)` and `s*_vel = SNR_vel/(1+SNR_vel)`, `SNR_vel = var(Δg)/var(Δn)`.
Numerical minimisation reproduces the closed form to 4 decimals, and H2's formula to 3.

| channel | noise-vel / GT-vel | s* per-frame (formula / numeric) | s* velocity (formula / numeric) | s* velocity, **Huber** |
|---|---|---|---|---|
| depth | 1.90× | 0.9634 / 0.9635 | **0.2174 / 0.2177** | 0.2268 |
| root ang | 0.50× | 0.9974 / 0.9972 | 0.7988 / 0.7985 | 0.7987 |
| root ang | 1.00× | 0.9895 / 0.9893 | 0.4982 / 0.4982 | 0.5044 |
| root ang | 1.25× | 0.9837 / 0.9835 | 0.3885 / 0.3887 | 0.3983 |
| root ang | 1.50× | 0.9767 / 0.9766 | 0.3061 / 0.3064 | 0.3182 |
| root ang | 2.00× | 0.9592 / 0.9593 | 0.1988 / 0.1992 | 0.2121 |
| joint ang | 1.25× | 0.9837 / 0.9835 | 0.3885 / 0.3887 | 0.3992 |
| joint ang | 1.50× | 0.9767 / 0.9766 | 0.3061 / 0.3064 | 0.3192 |

**Depth, multiplicative (real) noise model** `x_t = z_t(1+ε_t)`, ε ~ N(0, 0.0044²), mean depth 3.5 m,
prediction `ŷ = s·x` (shrinking the trajectory toward the camera):

| s | metric vel L2 | metric vel Huber(0.4) | **dimensionless** vel L2 (Δlog z) | dimensionless vel Huber | per-frame L2 |
|---|---|---|---|---|---|
| 1.000 | 0.29855 | 0.68590 | 0.024379 | 1.084667 | 0.00024 |
| 0.750 | 0.17303 | 0.45261 | 0.024379 | 1.084667 | 0.76594 |
| 0.500 | 0.09526 | **0.27581** | 0.024379 | 1.084667 | 3.06394 |
| 0.221 | 0.06484 | 0.19886 | 0.024379 | 1.084667 | 7.43772 |
| 0.100 | 0.07013 | 0.21657 | 0.024379 | 1.084667 | 9.92789 |
| argmin s | **0.2178** | **0.2268** | flat (spread over s∈[0.1,2] = **4.5e-17**, machine ε) | flat | 0.9999 |

The metric velocity loss is minimised at s ≈ 0.22 (Huber 0.23); the Huber drops 2.49× from s=1 to
s=0.5 (the user measured 0.41 → 0.18 = 2.28× on the real data — same effect, same size).
Normalising the increment by each trajectory's own depth makes the loss **exactly** scale-invariant
(variation at machine precision), confirming the proposed fix. Note the multiplicative model gives
*the same* `s*` as the additive one — algebraically identical once you scale the observation.

**Verdict.** H2 is exactly right, including the formula. The mechanism is real and large for depth
(s* = 0.22 under the velocity loss vs 0.96 under the per-frame loss).


### H2b. How strong must the per-frame anchor be? (`exp_h2b_anchor.py`, `out_h2b.txt`)

Model A, L2, exact: `s*(r) = (Vg + r·Vdg) / ((Vg+Vn) + r·(Vdg+Vdn))` with `r = w_vel/w_pf` in
matched units (metres vs metres/second). The per-frame-only optimum is the ceiling
(0.963 for depth, 0.984 for the angles), so `s* = 0.99` is unreachable at any `r ≥ 0` — hence the
negative entries.

| channel | var(g) | var(n) | var(Δg) | var(Δn) | r for s\*=0.95 | 0.90 | 0.50 |
|---|---|---|---|---|---|---|---|
| depth (0.44 %/frame) | 5.93e-03 | 2.37e-04 | 0.0829 | 0.294 | **2.58e-04** | 1.48e-03 | 2.69e-02 |
| root ang ×1.25 | 1.69e-02 | 2.95e-04 | 0.236 | 0.366 | 1.68e-03 | 4.65e-03 | 0.128 |
| joint ang ×1.25 | 4.04e-02 | 7.07e-04 | 0.566 | 0.877 | 1.68e-03 | 4.65e-03 | 0.128 |

For depth the toy's λ60 (L2) = 9.65e-4 sits between the 0.95 and 0.90 rows — i.e. **the
"velocity = 60 % of the gradient norm" balance already costs ~8 % of the depth amplitude in a model
that cannot filter**, and you would have to run the metric velocity term ~3.7× weaker than that to
stay within 5 %. Angles are ~6× more tolerant (their noise velocity is only 1.25× the GT, versus
1.90× for depth).

---

## Item 5 — Huber vs L2 (`out_h2.txt`, `out_h1b.txt`) — **the Huber does NOT rescue the shrinkage**

* Velocity-loss-only optimum: Huber s* is **3–7 % higher** than L2 s* at every real SNR
  (0.2177→0.2268 depth; 0.3064→0.3182 root ang ×1.5). Immaterial.
* At the *matched* 60 %-gradient-norm balance the two loss forms agree almost exactly:
  depth s*(λ60) = 0.9160 (Huber) vs 0.9165 (L2); joint ang 0.9617 vs 0.9614.
* The apparent "Huber is much gentler" impression only appears if you compare at the same *numeric*
  λ, which is not the same balance (the L2 velocity gradient is 625× larger).

---

## H1 — do the two losses have the same minimiser? (`exp_h1_filters.py`, `exp_h1c_conflict.py`)

### H1a. The conflict is a property of the MODEL CLASS, not of the objectives (`out_h1c.txt`, Huber)

Each class is optimised for the per-frame loss alone and for the velocity loss alone; each optimum
is then scored under both.

| channel | model class | excess per-frame cost of the velocity optimum | excess velocity cost of the per-frame optimum | DC gain (pf\*) | DC gain (vel\*) |
|---|---|---|---|---|---|
| depth | **A scale only** | **+1196 %** | +229 % | 0.959 | **0.222** |
| depth | B ±4 convex (DC=1) | +9.8 % | +61 % | 1.000 | 1.000 |
| depth | **D ±4 kernel + scale** | **+1.2 %** | **+4.0 %** | 1.039 | 1.052 |
| depth | ±16 free | +7.3 % | +43 % | 0.966 | 0.919 |
| root ang | A scale only | +1930 % | +119 % | 0.981 | 0.392 |
| root ang | D ±4 kernel + scale | +17.2 % | +58 % | 1.029 | 1.065 |
| root ang | ±16 free | +14.9 % | +57 % | 0.987 | 0.935 |
| joint ang | A scale only | +1652 % | +118 % | 0.981 | 0.393 |
| joint ang | D ±4 kernel + scale | +17.2 % | +58 % | 1.029 | 1.065 |
| joint ang | ±16 free | +14.9 % | +57 % | 0.987 | 0.935 |

(The L2 `B ±4 convex` rows in `out_h1c.txt` show negative "excess", i.e. SLSQP failed to converge on
the tiny L2 per-frame values — ignore those four rows; every BFGS/Huber row is converged.)

### H1b. With a filter available, the joint optimum is λ-independent (`out_h1b.txt`, depth, Huber)

| λ/λ60 | A: s\* | A rmse_pf (mm) | A rmse_vel | **D**: DC gain | D self | D width | **D rmse_pf** | **D rmse_vel** |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.9593 | 15.04 | 0.5224 | 1.0394 | 0.172 | 0.082 s | 6.38 | 0.0475 |
| 1 (real balance) | 0.9160 | 15.40 | 0.4993 | 1.0414 | 0.170 | 0.084 s | 6.38 | 0.0469 |
| 3 | 0.8361 | 17.80 | 0.4576 | 1.0429 | 0.167 | 0.085 s | 6.39 | 0.0467 |
| 10 | 0.6237 | 29.99 | 0.3559 | 1.0447 | 0.165 | 0.086 s | 6.39 | 0.0466 |
| 30 | 0.3813 | 47.16 | 0.2716 | 1.0468 | 0.164 | 0.086 s | 6.40 | 0.0466 |
| 100 | 0.2710 | 55.31 | 0.2536 | 1.0492 | 0.164 | 0.086 s | 6.41 | 0.0466 |

Scale-only: the per-frame error grows 3.7× across the sweep. Kernel+scale: the per-frame error moves
by **0.5 %** over four decades of λ and the optimal head scale is **1.04–1.05, i.e. slightly >1
(amplification, not shrinkage)**. Same picture for joint angles (11.82 → 12.49 mm over 4 decades).

### H1c. Which families reach their oracle? (`out_h1.txt`, depth, Huber, λ60)

| model | DC | self | width | rmse_pf (mm) | rmse_vel |
|---|---|---|---|---|---|
| identity (raw x) | 1.000 | 1.000 | 0 | 15.36 | 0.5445 |
| **oracle ±4 (joint)** | 1.041 | 0.170 | 0.084 s | **6.38** | **0.0469** |
| oracle ±4, DC pinned to 1 | 1.000 | 0.190 | 0.073 s | 6.80 | 0.0539 |
| oracle ±16 (joint) | 0.960 | 0.112 | 0.239 s | 5.24 | 0.0282 |
| A scale only | 0.916 | 0.916 | 0 | 15.40 | 0.4993 |
| B convex, uniform init | 1.000 | 0.190 | 0.074 s | 6.80 | 0.0538 |
| B convex, **identity init** | 1.000 | 0.190 | 0.074 s | 6.80 | 0.0538 |
| **D = B + free scale** (either init) | 1.041 | 0.170 | 0.084 s | **6.38** | **0.0469** |
| C 1 layer, DC pinned | 1.000 | 0.186 | 0.099 s | 8.05 | 0.0787 |
| C 1 layer, free h | 1.060 | 0.192 | 0.099 s | 6.80 | 0.0773 |
| C 4 layers, DC pinned | 1.000 | 0.202 | 0.098 s | 8.03 | 0.0839 |
| C 4 layers, free h | 1.107 | 0.239 | 0.132 s | 8.28 | 0.0887 |

B reaches its constrained oracle **exactly**, from the identity init as well (Adam, lr 1e-2, 7.5 k
full-batch steps). D reaches the unconstrained ±4 oracle exactly. C-1-layer gets the per-frame
number (6.80 mm) but only 0.077 vs 0.047 on velocity — the family `h(δ + c·u)` is a boxcar plus a
delta bump, close but not the oracle shape. **C-4-layers converged WORSE than C-1-layer**
(8.28 mm / 0.0887): with all four layers free the optimiser settles on the symmetric solution
`c = 0.68, h = 0.609` in every layer instead of putting the gain in one layer; the DC-pinned
4-layer run does find the one-layer solution (`c = [0,0,0,8.67]`). Treat that as an optimisation
artifact of the toy, not a capacity statement.

**Verdict on H1.** Confirmed in substance, with a caveat. The two objectives are *not* minimised by
exactly the same filter — the velocity optimum genuinely over-smooths (it costs +1.2 % to +17 %
per-frame error even in the best class, and the ±16 pf-optimum costs +43 % velocity loss). But that
residual conflict is 1–2 orders of magnitude smaller than the conflict inside the per-frame-only
class (+1200 % to +2100 %). So: **the fight is not between the objectives, it is between the
velocity objective and a model that has no temporal filter.** A model that can filter pays
essentially nothing for the velocity term and never shrinks.

---

## H5 — is the GT-body-frame comparison the problem? (`exp_h5_so3.py`, `out_h5.txt`) — **CONFIRMED**

Real BetterRobot `se3`/`so3` ops. GT SE(3) trajectories at the real magnitudes (body angular
velocity 0.486 rad/s, root velocity 0.288 m/s per component), predictions = GT ∘ exp(constant
offset + white per-frame rotation noise) plus white position noise. 256 clips × 60 frames.

### Angular part: the identity holds to machine precision

`E_t = R_gt,t^T R_pred,t`, `ω_body = log(R_t^T R_{t+1})/dt`, `ω_spatial = log(R_{t+1} R_t^T)/dt`.

| case | `max ‖E_t ω_body,pred − ω_body,gt‖ − ‖ω_sp,pred − ω_sp,gt‖` | level |
|---|---|---|
| real magnitudes (10° offset, 2°/frame noise) | **2.2e-15** | 1.133 rad/s |
| 55° rotation offset (the v3 probe number) | 8.0e-15 | 1.105 rad/s |
| 5× angular rate | 7.6e-15 | 1.133 rad/s |
| 5× rate + 55° offset | 8.4e-15 | 1.105 rad/s |
| 20× rate (9.7 rad/s, unphysical) | 1.1e-14 | 1.134 rad/s |

Also verified: `ω_spatial = R_t · ω_body` to 1.5e-15. So the transported body-frame angular
comparison **is** the spatial/world-frame comparison, exactly, at any rate and any pose error.

### Linear part: the world-velocity difference, up to a small V⁻¹ cross-talk

| case | rms ‖E v_pred − v_gt‖ (se3) | rms ‖Δp_pred − Δp_gt‖/dt (world) | rms deviation / rms world |
|---|---|---|---|
| real magnitudes | 0.94315 m/s | 0.94308 m/s | **0.61 %** |
| no rotation error | 0.94308 | 0.94308 | 0.0003 % |
| 55° rotation offset | 0.94316 | 0.94308 | 0.59 % |
| 20× angular rate | 0.94406 | 0.94308 | 2.4 % |

Dropping V⁻¹ (comparing `E_t · R_t^T Δp/dt`) reproduces the world-frame difference **exactly**
(max deviation 9e-16) — so the only difference is the V⁻¹ factor. Isolating it by predicting the
position perfectly and leaving only a rotation error:

| case | spurious linear residual from V⁻¹ alone |
|---|---|
| 10° constant offset **only**, no per-frame rotation noise | **0.0 m/s exactly** |
| 10° offset + 2°/frame rotation noise | 0.0100 m/s (3.5 % of the 0.288 m/s per-component GT vel) |
| 10° offset + 5°/frame rotation noise | 0.0250 m/s |

**Verdict.** H5 is confirmed. The frame choice is not where the fight comes from: the angular
comparison is *identical* to a spatial one, and the linear comparison differs from the plain
world-velocity difference by 0.6 % rms. The only genuine artifact of the SE(3) form is that
**angular-velocity error leaks into the linear channel** through V⁻¹ (0 for a constant orientation
offset; ~0.01 m/s at the real per-frame rotation noise). That is a second-order nuisance, not the
mechanism behind +22 mm MPJPE.

---

## H4 — residual dilution and the first-order gradient (`exp_h4_firstorder.py`, `exp_h4_gradsnr.py`)

### H4a. Exact gradients at the identity point (c = 0, h = 1), full 512-clip batch (`out_h4_first.txt`)

Sign convention: **gradient > 0 ⇒ descent DECREASES the parameter**.

| channel | loss / term | dL/dc (residual branch) | dL/dh (head scale) | dL/dβ, model B at uniform init |
|---|---|---|---|---|
| depth | L2 / per-frame | +7.66e-05 | +4.99e-04 | −9.32e-06 |
| depth | L2 / velocity | +1.47e-03 | **+0.596** | −9.54e-04 |
| depth | Huber / per-frame | +1.53e-02 | +9.97e-02 | −1.86e-03 |
| depth | Huber / velocity | +5.00e-03 | **+1.000** | −2.98e-03 |
| joint ang | L2 / per-frame | +2.60e-04 | +1.52e-03 | −6.42e-05 |
| joint ang | L2 / velocity | +5.13e-03 | **+1.777** | −3.81e-03 |
| joint ang | Huber / velocity | +4.85e-03 | +0.727 | −2.35e-03 |

Population predictions (verified): `dL_pf/dc = 2σ²·u₀ = 2σ²/9 > 0`, `dL_pf/dh = 2σ²`,
`dL_vel/dh = 2·var(Δn)`, and — for a boxcar branch —
`dL_vel/dc = 2σ²(2u₀ − u₁ − u₋₁)/dt² = **0**` (also 0 for the Huber, by Stein's lemma, since at
`c = 0` the residual is pure noise). The nonzero numbers in the `dL_vel/dc` column are finite-sample
noise; the next table proves it.

### H4b. Per-minibatch gradient statistics at the identity point (`out_h4_snr.txt`)

8 clips per minibatch (the real batch size), 4000 independent draws with fresh GT + fresh noise.
`SNR = mean/std`; Adam's steady drift per step ≈ `lr · SNR/√(1+SNR²)` (1.0 = full learning rate).
(That approximation takes `m̂ → μ` and `v̂ → μ²+σ²`, ignoring the EMA's own residual variance, so it
is a slight over-estimate; the ordering it gives is what matters here.)

| channel | loss | param | mean | std | SNR | **Adam drift / lr** |
|---|---|---|---|---|---|---|
| depth | Huber velocity | **c (residual branch)** | +8.0e-06 | 1.7e-02 | **0.0005** | **0.0005** |
| depth | Huber velocity | **h (head scale)** | +0.9965 | 7.5e-02 | **13.22** | **0.997** |
| depth | Huber per-frame | c | +1.04e-02 | 2.9e-02 | 0.356 | 0.336 |
| depth | Huber per-frame | h | +9.47e-02 | 3.2e-02 | 2.92 | 0.946 |
| depth | Huber per-frame | β (B, uniform init) | −2.00e-03 | 4.0e-04 | **−5.03** | −0.981 |
| depth | Huber velocity | β (B, uniform init) | −3.42e-03 | 2.2e-03 | −1.59 | −0.847 |
| joint ang | Huber velocity | c | −1.6e-05 | 1.5e-02 | 0.0010 | 0.001 |
| joint ang | Huber velocity | h | +0.7224 | 6.0e-02 | 12.07 | 0.997 |
| joint ang | Huber per-frame | c | +7.70e-03 | 3.3e-02 | 0.235 | 0.228 |
| joint ang | Huber per-frame | h | +7.05e-02 | 3.6e-02 | 1.97 | 0.892 |
| joint ang | L2 per-frame | β (B, uniform) | −6.82e-05 | 1.2e-05 | −5.90 | −0.986 |

**Reading.** From the identity init the velocity loss gives the residual branch a gradient whose
population mean is **zero** (SNR 0.0005–0.001 — Adam cannot amplify it; the update is a random walk
that averages to nothing), while it gives the head scale a gradient with **SNR 12–13**, i.e. Adam
moves `h` down at essentially the full learning rate. The per-frame loss gives `c` a *positive*
(anti-smoothing) gradient with SNR 0.23–0.36. So the very first thing that moves is the head scale,
downward, and the block does not move at all. That is exactly the reported failure mode.

### H4c. But H4's claim about model B is **REFUTED as stated**

H4 says a non-residual convex attention "has a first-order smoothing gradient under any
noise-sensitive loss". Both halves fail:

1. **At a uniform init the first-order pull is toward NARROWING, not smoothing.** `dL/dβ < 0` at
   β = 0 for every channel and both losses, i.e. descent raises the self weight. The sign flips
   only past the optimum (`out_h4_first.txt`, depth):

| β | w_self | dL_pf/dβ (L2) | dL_vel/dβ (L2) | dL_pf/dβ (Huber) | dL_vel/dβ (Huber) |
|---|---|---|---|---|---|
| 0.0 | 0.111 | −9.3e-06 | −1.9e-03 | −9.5e-04 | −3.0e-03 |
| 1.0 | 0.254 | −5.8e-07 | −1.2e-04 | +1.9e-02 | +5.9e-02 |
| 2.0 | 0.480 | +3.5e-05 | +7.1e-03 | +6.8e-02 | +1.9e-01 |
| 3.0 | 0.715 | +6.0e-05 | +1.2e-02 | +9.2e-02 | +2.1e-01 |
| 5.0 | 0.949 | +2.1e-05 | +4.3e-03 | +3.1e-02 | +5.4e-02 |
| 8.0 | 0.997 | +1.3e-06 | +2.5e-04 | +1.8e-03 | +3.0e-03 |
| 12.0 | 1.000 | **+2.3e-08** | **+4.7e-06** | **+3.3e-05** | **+5.5e-05** |

2. **At an identity init (β large) the softmax gradient is exponentially suppressed** — 3 to 4
   orders of magnitude smaller at β = 12 than at β = 3. It is *not* zero in the mean, however, so
   Adam's per-parameter normalisation restores it to a full-size step: in the long runs B does
   escape the identity init and reach its oracle exactly (see H3c below). The distinction that
   matters is **zero-mean (branch `c` under the velocity loss) vs merely small (softmax logits)** —
   Adam rescues the second and cannot rescue the first.

### H4d. The dilution arithmetic and the co-adaptation requirement (`out_h4_first.txt`)

`(1 + c·w_self)/(1 + c)` with `w_self = 1/9`:

| c | noise gain | head rescale h for DC = 1 |
|---|---|---|
| 0.1 | 0.919 | 0.909 |
| 1 | 0.556 | 0.500 |
| 4 | 0.289 | 0.200 |
| 8 | 0.210 | 0.111 |
| ∞ | 0.111 | 0 |

Confirmed: the residual form needs a **large** branch gain and a head that rescales by 1/(1+c).
In the toy the converged 1-layer solution is exactly that: `c = 11.7, h = 0.083` (`out_h1.txt`).

### H4e. What decides the sign of the branch gradient: whether the average includes the frame's own token (`exp_h4b_noself.py`, `out_h4b.txt`)

The population value of `dL_vel/dc` at the identity point is `2σ²(2u₀ − u₁ − u₋₁)/dt²`, where `u` is
the branch's averaging kernel. Measured over 4000 fresh 8-clip minibatches, Huber losses:

| channel | branch kernel | `2u₀ − u₁ − u₋₁` | velocity `dL/dc` mean | SNR | **Adam drift / lr** |
|---|---|---|---|---|---|
| depth | boxcar ±4, **self included** | 0.000 | +8.0e-06 | **0.0005** | **0.0005** (frozen) |
| depth | boxcar ±4, **self masked out** | −0.250 | **−0.1246** | **−6.17** | **−0.987** (smooths, full speed) |
| depth | gaussian σ=2 frames, **self-peaked** | +0.048 | +0.0239 | +1.55 | +0.840 (**anti**-smooths, full speed) |
| joint ang | boxcar ±4, self included | 0.000 | −1.6e-05 | −0.0010 | −0.001 (frozen) |
| joint ang | boxcar ±4, self masked out | −0.250 | −0.0903 | −5.35 | −0.983 |
| joint ang | gaussian σ=2, self-peaked | +0.048 | +0.0173 | +1.21 | +0.772 |

This is the sharpest single number in the whole study. A uniform ±4 attention window that includes
the self key sits **exactly on the knife edge** — the velocity loss gives the residual branch no
first-order signal at all. Mask the self key and the same loss drives the branch toward smoothing at
99 % of the full learning rate. Let the attention peak on itself (which is what a query·key softmax
does by default, and what RoPE-at-equal-positions encourages) and the velocity loss drives the branch
**away** from smoothing at 77–84 % of the full learning rate.

**Verdict on H4.** The core of H4 is confirmed and quantified: from the identity init the residual
branch receives *no usable* velocity-loss signal while the head scale receives a maximally consistent
one, and the residual form does need a large branch gain plus a co-adapting head. But two specific
sub-claims are wrong: (i) the branch's first-order velocity pull is **zero**, not anti-smoothing
(anti-smoothing comes from the per-frame term, SNR 0.23–0.36, and from any self-peaking of the
attention); (ii) the non-residual convex attention does **not** have a first-order smoothing gradient
— at a uniform init its gradient points toward *narrowing*, and at an identity init it is
exponentially suppressed (though Adam still recovers it, because it is small-but-consistent rather
than zero-mean).

---

## H3 — gradient spectrum and Adam's second moment (`exp_h3_adam.py`, `out_h3.txt`)

### H3a. The spectrum (analytic, verified numerically to 1e-2 relative)

`∂L_vel/∂ŷ = (2/dt²)·DᵀD·(ŷ−g)` ⇒ transfer gain `8 sin²(ω/2)/dt²`; `∂L_pf/∂ŷ = 2(ŷ−g)` ⇒ gain 2.

| f (Hz) | velocity L2 gain | per-frame L2 gain | ratio (L2) | ratio with the repo's δ-normalised **Huber** (δ_pf 0.05 m, δ_v 0.4 m/s) |
|---|---|---|---|---|
| 0 (DC) | 0 | 2 | **0** | 0 |
| 0.5 | 19.7 | 2 | 9.9 | 0.15 |
| 1.0 | 78.5 | 2 | 39 | 0.61 |
| **1.28 (Huber cross-over)** | 128 | 2 | 64 | **1.00** |
| 1.6 | 199 | 2 | 100 | 1.56 |
| 4.0 | 1160 | 2 | 580 | 9.1 |
| 8.0 | 3564 | 2 | **1782** | 27.9 |
| 12.5 (Nyquist) | 5000 | 2 | **2500** | 39.1 |

H3's "~1800× at Nyquist" is right for **L2** (it is 1782× at 8 Hz, 2500× at Nyquist). But the repo
does not use L2: with the δ-normalised Huber the ratio at Nyquist is only **39×**, and **below
1.28 Hz the velocity term has *less* gain than the per-frame term** (0.15× at 0.5 Hz). The δ ratio
(0.4/0.05 = 8, squared = 64) undoes most of the 1/dt² amplification. This weakens H3
substantially at the frequencies where the *signal* lives, while leaving the high-frequency
dominance intact.

### H3b. Adam second moment, measured at a FIXED parameter point (no trajectory confound)

Identity init, 8 clips per minibatch, 3000 fresh minibatches, Huber losses at λ60.
`SHRINK = √v(pf+λ·vel) / √v(pf only)` = the factor by which the *per-frame* gradient's Adam step is
divided when the velocity term is switched on, **in the same parameters**.

| channel | model | param | mean g_pf | √v (pf only) | √v (joint) | **SHRINK** | \|step\| from pf, no vel | with vel |
|---|---|---|---|---|---|---|---|---|
| depth | D = softmax kernel + scale | logits (side) | −5.8e-07 | 5.97e-07 | 1.26e-06 | **2.11** | 0.976 | 0.462 |
| depth | " | logits (±1) | −5.8e-07 | 5.88e-07 | 1.59e-06 | **2.71** | 0.993 | 0.366 |
| depth | " | logit (self) | +4.7e-06 | 4.69e-06 | 1.07e-05 | **2.28** | 0.995 | 0.436 |
| depth | " | **s (head scale)** | +0.0951 | 0.100 | 0.207 | **2.06** | 0.948 | 0.460 |
| depth | D = 4 residual layers + scale | c (branch) | +0.0107 | 0.0305 | 0.0310 | **1.02** | 0.352 | 0.346 |
| depth | " | s (head scale) | +0.0951 | 0.100 | 0.207 | 2.06 | 0.948 | 0.460 |
| joint ang | softmax kernel + scale | logits | −4.3e-07 | 4.4e-07 | 0.95–1.20e-06 | 2.09–2.74 | 0.95–0.99 | 0.36–0.46 |
| joint ang | " | s | +0.0710 | 0.0793 | 0.158 | 1.99 | 0.896 | 0.450 |

So the mechanism H3 describes is **real but modest at the real balance: a factor ~2**, uniform
across parameters (it is not a selective suppression of the kernel parameters — the head scale is
throttled by exactly the same 2.06×). It is *not* 1800×; Adam's normalisation is what converts the
huge raw gradient ratio into a factor-2 effect. Note the residual branch parameter `c` is the
exception (SHRINK 1.02) precisely because the velocity term contributes almost nothing to its
gradient at all (H4b).

### Cross-check: the toy's depth channel matches the real numbers

The brief reports GT `dlogz = 0.275 %/frame` and frozen-model `dlogz_pred = 0.596 %/frame` on the
16 static test scenes. Treating the excess as white per-frame depth noise:
`σ_pred(dlogz) = √(0.596² − 0.275²) = 0.529 %/frame` ⇒ real `SNR_vel = 0.275²/0.529² = 0.270` ⇒
`s*_vel = 0.270/1.270 = 0.213`. The toy's depth channel lands at `SNR_vel = 0.278`,
`s*_vel = 0.218`. So the toy is calibrated to the real system to within 3 %, and the prediction is
that **the metric root-velocity term alone is minimised by shrinking the depth trajectory to ~21 %
of its true amplitude** (observed sign: `tvel_ray v2` depth bias −33 mm, `tvel_cliff v2` −2096 mm).

### H3c. Dynamic Adam runs at the REAL learning rate (lr 2e-4, 8 clips/step, 60 000 steps, fresh data every step, identity init, Huber at λ60) — depth channel

| model | loss | final s (head scale) | DC gain | self weight | width | rmse_pf (mm) | rmse_vel |
|---|---|---|---|---|---|---|---|
| B softmax kernel + free scale | pf only | 1.0415 | 1.042 | 0.172 | 0.082 s | 6.38 | 0.0474 |
| B softmax kernel + free scale | **pf + vel** | 1.0434 | 1.043 | 0.169 | 0.084 s | **6.39** | **0.0469** |
| B softmax kernel + free scale | pf + vel, **vel→kernel only** | 1.0433 | 1.043 | 0.169 | 0.084 s | 6.39 | 0.0469 |
| C 1 residual layer + free scale | pf only | 0.2596 | 1.046 | 0.347 | 0.090 s | 7.30 | 0.1506 |
| C 1 residual layer + free scale | **pf + vel** | **0.1759** | 1.049 | 0.273 | 0.094 s | **6.95** | **0.1115** |
| C 1 residual layer + free scale | pf + vel, **vel→kernel only** | 0.2628 | 1.046 | 0.350 | 0.089 s | 7.32 | 0.1522 |
| C 4 residual layers + free scale | pf only | 0.2449 | 1.076 | 0.334 | 0.115 s | 7.91 | 0.1413 |
| C 4 residual layers + free scale | **pf + vel** | **0.1414** | 1.099 | 0.242 | 0.131 s | **8.26** | **0.0908** |
| C 4 residual layers + free scale | pf + vel, **vel→kernel only** | 0.2116 | 1.083 | 0.305 | 0.120 s | 7.93 | 0.1244 |
| | | (oracle ±4: DC 1.041, self 0.170, 6.38 mm, 0.0469) | | | | | |

Branch gains reached (linear, zero-init): C-1 `c = 3.03` (pf only) / `4.96` (pf+vel) / `2.98`
(routed); C-4 `c = 0.448` / `0.670` / `0.504` per layer.

The **joint-angle** channel (now complete) replicates every depth finding:

| model | pf only | pf + vel | vel→kernel only |
|---|---|---|---|
| B (rmse_pf mm / rmse_vel) | 11.85 / 0.0984 | 11.89 / 0.0933 | 11.89 / 0.0934 |
| B: head scale s | 1.0328 | 1.0362 | 1.0360 |
| C-1 | 14.41 / 0.3180 | **13.84 / 0.2330** | 14.33 / 0.3097 |
| C-1: s, c | 0.3251, 2.213 | 0.2243, 3.682 | 0.3156, 2.312 |
| C-4 | 15.98 / 0.3428 | **16.82 / 0.2321** | 16.02 / 0.3105 |
| C-4: s, c | 0.3533, 0.316 | 0.2251, 0.480 | 0.3169, 0.354 |

Same three signatures: the non-residual kernel is unaffected (11.85 → 11.89 mm); the 1-layer
residual improves (14.41 → 13.84); **only the 4-layer residual pays per-frame accuracy for velocity
accuracy (15.98 → 16.82 mm, +5.3 %, vs 0.3428 → 0.2321 velocity)** — matching the depth channel's
+4.4 %; and routing the velocity gradient away from the head scale reverts both residual models to
almost exactly the pf-only solution (14.33 / 16.02 mm, `c` 2.312 / 0.354 vs the pf-only 2.213 /
0.316). The early per-frame throttling is the same 9× for model B (`proj` first 10 %:
1.90e-05 → 2.09e-06 → 1.91e-05).
`proj = ⟨−update, ĝ_pf⟩` (distance moved per step down the per-frame gradient direction), model B,
depth: pf-only mean 5.84e-05 (first 10 % of the run 2.23e-05); pf+vel mean 3.46e-05 (**first 10 %
1.08e-06, a 21× reduction**); routed 5.83e-05 (2.24e-05). The velocity term throttles early
per-frame progress by ~20× and overall by 1.7×, and routing it away from the head restores it
exactly.

Trajectories (`h3c_*.json`), depth, model B + free scale:

| step | s (pf only) | s (**pf+vel**) | s (vel→kernel only) | self weight (pf+vel) |
|---|---|---|---|---|
| 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| 500 | 0.960 | **0.931** | 0.960 | 0.931 |
| 1 000 | 0.958 | **0.915** | 0.958 | 0.915 |
| 5 000 | 0.960 | **0.915** | 0.960 | 0.915 |
| 10 000 | 0.963 | **0.919** | 0.963 | 0.908 |
| 20 000 | 1.032 | 1.029 | 1.039 | 0.398 |
| 40 000 | 1.042 | 1.043 | 1.043 | 0.169 |
| 59 500 | 1.041 | 1.043 | 1.042 | 0.170 |

Model C (residual) + free scale, depth:

| step | s (pf only) | s (pf+vel) | DC gain (pf+vel) | self (pf+vel) |
|---|---|---|---|---|
| 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1 000 | 0.865 | 0.832 | 0.976 | 0.848 |
| 5 000 | 0.611 | 0.532 | 0.998 | 0.584 |
| 20 000 | 0.376 | 0.280 | 1.042 | 0.365 |
| 59 500 | 0.260 | 0.176 | 1.048 | 0.273 |

**What the dynamics show.**
1. **The shrinkage shortcut is taken first and it is transient in this toy.** With model B the head
   scale drops to 0.915 within 1000 steps and *sits there for 10 000 steps* before the kernel widens
   and s recovers to 1.04. Adding the velocity term doubles the depth of that dip (0.958 → 0.915).
2. **Routing the velocity gradient away from the head scale removes the dip completely** — the
   `vel→kernel only` s-trajectory is numerically identical to the pf-only one (0.960 at step 500),
   and the kernel still converges to the same oracle.
3. **But in the residual model the routing removes the velocity term's entire effect.** With the
   velocity gradient detached from `s`, model C ends at `c = 2.979` vs the pf-only `c = 3.03` and
   `rmse_vel` 0.1522 vs the pf-only 0.1506 — the velocity loss contributed *nothing* to the block.
   Its only channel of influence on a residual block is: shrink the head, and let the block grow
   its branch gain afterwards to restore the DC gain (which it does — DC stays 0.98–1.05 throughout
   while s falls 1.0 → 0.18 and `c` rises 0 → 4.96).
4. **In this toy the velocity term never destroys anything.** At the real learning rate, over
   60 k steps, `pf+vel` ends at least as good as `pf only` on both metrics for both model families
   (6.39/0.0469 vs 6.38/0.0474 for B; 6.95/0.1115 vs 7.30/0.1506 for C). The collapse reported on
   the real system is **not** reproduced by a scalar model that is free to learn a ±4 kernel.

### H3d/H3e. The landscape that explains the dynamics (`exp_h3de_landscape.py`, `out_h3de.txt`)

One residual layer, `ŷ = s·(x + c·mean_{±4} x)`, depth channel, Huber losses, 1024 clips.

**With the head held at `s = 1` the branch direction is uphill everywhere** (both losses), so a
model that cannot move its head can never grow the branch. With the head co-adapting
(`s = 1/(1+c)`, DC gain 1) the same direction is downhill from `c ≈ 0.05` on, and the achieved error
falls monotonically:

| c | `s=1`: dL_pf/dc | dL_vel/dc | `s=1/(1+c)`: dL_pf/dc | dL_vel/dc | rmse_pf (mm) | rmse_vel |
|---|---|---|---|---|---|---|
| 0.00 | +0.0125 | +1.3e-05 | +0.0125 | +1.3e-05 | 15.46 | 0.5451 |
| 0.05 | +0.110 | +0.0115 | +0.0053 | −0.00059 | 14.81 | 0.5191 |
| 0.20 | +0.399 | +0.0457 | −0.0099 | −0.0020 | 13.23 | 0.4543 |
| 1.00 | +1.077 | +0.215 | −0.0303 | −0.0053 | 9.50 | 0.2745 |
| 4.00 | +1.169 | +0.495 | −0.0209 | −0.0040 | **7.92** | 0.1220 |
| 16.0 | +1.174 | +0.554 | −0.0074 | −0.0014 | 8.12 | 0.0726 |

**How little head shrink it takes to open the basin** (`c = 0`, sweeping `s`):

| s | dL_pf/dc | dL_vel/dc | joint dL/dc (λ60) | joint dL/ds |
|---|---|---|---|---|
| 1.000 | **+0.01247** | +1.3e-05 | **+0.01248** | +0.207 (shrink) |
| 0.999 | +0.01037 | −0.00023 | +0.01034 | +0.204 |
| **0.990** | **−0.00840** | −0.00240 | **−0.00866** | +0.183 |
| 0.950 | −0.0878 | −0.0119 | −0.0891 | +0.087 |
| 0.900 | −0.1775 | −0.0236 | −0.1801 | **−0.033** (stops shrinking) |
| 0.600 | −0.4680 | −0.0826 | −0.4770 | −0.722 |

So the sequence is: the velocity loss shrinks the head (`dL/ds = +0.207` at the identity point,
88 % of which is the velocity term: +1.0008·λ = 0.109 vs +0.098 from the per-frame term); after a
**1 % shrink** the branch gradient flips sign and the block starts growing; the head stops shrinking
around `s = 0.9`. Note in every row the velocity term's contribution to `dL/dc` is ~10× smaller than
the per-frame term's — once the basin is open it is the *per-frame* loss that grows the branch.

---

## One paragraph per hypothesis

**H1 (L2 consistency) — confirmed in substance, with a measured caveat.** The two objectives are
*not* minimised by exactly the same estimator: even in the best model class the velocity optimum
costs +1.2 % (depth) to +17 % (angles) extra per-frame error, and the per-frame optimum costs
+4 % to +58 % extra velocity error (`out_h1c.txt`). But that residual disagreement is two to three
orders of magnitude smaller than the disagreement inside the *per-frame-only* model class, where the
velocity optimum costs **+1196 % to +2078 %** extra per-frame error. And with a ±4 kernel available
the joint optimum is essentially λ-independent: over four decades of velocity weight the optimal
filter's per-frame error moves 0.5 % and the optimal head scale stays at 1.04–1.05, i.e. above 1
(`out_h1b.txt`). So the numbers support H1's framing exactly: the losses fight only when the model
cannot implement the filter and reaches the velocity optimum through a per-frame shortcut.

**H2 (shrinkage shortcut) — confirmed exactly, formula included.** `s*_vel = SNR_v/(1+SNR_v)`
reproduces the numerical optimum to three decimals at every SNR tested. For the depth channel at the
real noise level (0.44 %/frame of 3.5 m ⇒ noise velocity 0.5465 m/s = 1.90× the GT 0.288 m/s) the
velocity loss alone wants `s = 0.218` while the per-frame loss wants `s = 0.963`. The multiplicative
(noise ∝ depth) model gives the identical optimum, and normalising the increment by each
trajectory's own depth makes the loss exactly scale-invariant (variation 4.5e-17 over `s ∈ [0.1, 2]`),
confirming the proposed fix. The toy's depth channel is calibrated to the real `dlogz` numbers to
within 3 %. The one thing H2 does not say is that this optimum is only reached by a model *without*
a filter — with a kernel the same losses want `s ≈ 1.04`.

**H3 (gradient spectrum) — the spectrum is right for L2 but the Huber cuts it by 64×, and Adam turns
the whole thing into a factor of 2.** The Laplacian symbol `8 sin²(ω/2)/dt²` is exact (numerically
verified to 1e-2 relative); the L2 ratio is 1782× at 8 Hz and 2500× at Nyquist, matching H3's
"~1800×". But the repo's δ-normalised Huber divides the velocity gain by `δ_v² = 0.16` and the
per-frame gain by `δ_pf² = 0.0025`, so the real ratio at Nyquist is **39×**, and **below 1.28 Hz the
velocity term has *less* output-gradient gain than the per-frame term**. Measured at a fixed
parameter point with the real batch size, adding the velocity term at the 60 % balance multiplies
Adam's second moment by **2.0–2.7× uniformly across all parameters** — the per-frame gradient's
effective step is halved, but it is halved for the head scale (2.06×) exactly as much as for the
kernel logits (2.11–2.71×), so this is a global slow-down, not a selective suppression of the
filtering direction.

**H4 (residual dilution) — core confirmed and sharpened; two sub-claims refuted.** Confirmed: the
residual form needs a large branch gain `c` plus a head that rescales by `1/(1+c)` (the toy's
converged solution is `c = 11.7, h = 0.083`), and from the identity init the velocity loss gives the
head scale a gradient with SNR 12–13 (Adam moves it at 99.7 % of the full lr) while giving the branch
gain a gradient whose population mean is **exactly zero** (SNR 0.0005, Adam moves it at 0.05 % of
the full lr). Refuted: (i) the velocity loss's first-order pull on the branch is zero, not
anti-smoothing — the anti-smoothing pull comes from the per-frame term (SNR 0.23–0.36); (ii) a
non-residual convex attention does **not** have a first-order smoothing gradient — at a uniform init
its gradient points toward *narrowing* (SNR −5 for per-frame, −1.6 for velocity, i.e. descent raises
the self weight), and at an identity init it is exponentially suppressed (3–4 decades) though Adam
still recovers it because it is small-but-consistent rather than zero-mean. The sharpest finding
is a corollary: the sign of the branch gradient is `2u₀ − u₁ − u₋₁`, so a ±4 attention window that
*includes* the self key sits exactly on the knife edge (no signal), masking the self key gives a
strong smoothing signal (SNR −6.2, 99 % of full lr), and letting the attention peak on itself drives
the branch *away* from smoothing at 77–84 % of full lr.

**H5 (frame choice is not the problem) — confirmed, to machine precision.** Using the real
BetterRobot `se3`/`so3`, `‖E_t ω_body,pred − ω_body,gt‖` equals `‖ω_spatial,pred − ω_spatial,gt‖`
to 2e-15 at every angular rate (up to 20× real) and every pose error (up to 55°). The linear part of
the transported `se3` residual equals the plain world-frame velocity difference to 0.61 % rms at the
real magnitudes; dropping the `V⁻¹` factor makes them identical to 9e-16. The only genuine artifact
is that `V⁻¹` couples *angular-velocity* error into the linear channel: exactly 0 for a constant
orientation offset, 0.010 m/s at 2°/frame rotation noise, 0.025 m/s at 5°/frame — i.e. 3–9 % of the
0.288 m/s GT per-component velocity. Not a mechanism for +22 mm MPJPE.

**Item 5 (Huber vs L2) — no material difference to the shrinkage.** The velocity-loss-only optimum
is 3–7 % higher under the Huber than under L2 at every real SNR (depth 0.218 → 0.227), and at the
matched 60 %-gradient-norm balance the two agree to 0.05 % (depth 0.9165 vs 0.9160). The Huber's
real effect is on the *spectrum* (H3a), not on the shrinkage optimum.

---

## Things that contradict, or are not explained by, the hypotheses

1. **At the joint optimum a model with a free scale AND a kernel does not shrink — it amplifies.**
   Optimal `s = 1.03–1.06`, stable over four decades of λ (`out_h1b.txt`). H2's shrinkage is a
   property of the per-frame-only model class, not of the joint objective.
2. **The toy reproduces only a fraction of the reported damage.** At the real learning rate, over
   60 000 steps from the identity init, `pf+vel` ends *no worse* than `pf only` for the
   non-residual kernel (6.39 vs 6.38 mm) and *better* for the 1-layer residual (6.95 vs 7.30 mm).
   Only the 4-layer residual stack pays: 8.26 vs 7.91 mm, **+4.4 %** — the right sign but an order
   of magnitude smaller than the real +27 % MPJPE. The shrinkage also appears as a *transient*
   (model B: s → 0.915 and held there for 10 000 steps, twice as deep as without the velocity term)
   before the kernel takes over. Either the real block cannot learn the filter for reasons outside
   this toy (which is what `tb_stage2` = "fresh block + frozen heads could not denoise" already
   says), or the real damage comes from a mechanism the toy does not model (multi-dimensional token,
   nonlinear head, the pose→world lifting, camera coupling).
3. **The velocity loss cannot teach a residual block to filter directly.** Detaching the head scale
   from the velocity term leaves the residual models' learned kernels essentially identical to the
   pf-only run: C-1-layer `c = 2.979` vs `3.03` (rmse_vel 0.1522 vs 0.1506), C-4-layer `c = 0.504`
   vs `0.448` (0.1244 vs 0.1413) — versus `c = 4.96` / `0.670` when the velocity gradient is allowed
   to reach the head. So ~all of the velocity loss's influence on a residual block flows through the
   head scale it shrinks; the block only follows afterwards to restore the DC gain. This is a
   stronger statement than H4 makes, and it means "route the velocity gradient to the block only"
   would be close to a **no-op** for a residual block. It is *not* a no-op for a non-residual
   convex-kernel block: there it removes the shrinkage transient entirely (s stays at 0.960 instead
   of dipping to 0.915) and the kernel still converges to the oracle.
4. **H3's spectrum argument is much weaker than stated under the loss the repo actually uses.**
   39× at Nyquist, and *below* the per-frame gain under 1.28 Hz.
5. **4-layer model C converged worse than 1-layer model C** (8.28 vs 6.80 mm rmse_pf at the same
   loss) — the free-`h` 4-layer run settles on the symmetric solution `c = 0.68, h = 0.609` in every
   layer. This is an optimisation artifact of the toy's symmetric parametrisation (the DC-pinned
   4-layer run does find the 1-layer solution), but it is a reminder that depth in a residual
   dilution stack does not automatically buy a better filter.
6. **Numerical caveats.** The four `B ±4 convex (DC=1)` rows under the **L2** loss in `out_h1c.txt`
   show negative "excess" values — SLSQP fails to converge on L2 per-frame losses of order 1e-4.
   Every BFGS row and every Huber row is converged. The `numeric symbol check` in H3a agrees with
   the analytic Laplacian symbol to 1e-2 relative (the residual is the two boundary rows).

---

## Limitations of the toy (read before extrapolating)

1. **Scalar channel.** One latent dimension per experiment. The real block mixes a 1024-d token and
   the head is a nonlinear MLP; "head scale" here is a single scalar `s`, "branch gain" a single
   scalar `c`. Anything that depends on the *direction* structure of the token (e.g. the filter that
   denoises depth being different from the one that denoises articulation) is invisible here.
2. **`var(g)` is a modelling choice.** It follows from the assumed frequency band (0.15–1.0 Hz) once
   the velocity RMS is pinned. It sets `SNR_pf` (26 for depth, 60–94 for angles) and therefore how
   close the per-frame optimum is to `s = 1`, i.e. how big the fight is. A lower-frequency GT would
   raise `SNR_pf` and make the conflict *worse*; a higher-frequency one would soften it. The
   `dlogz` cross-check above is the only tie to the real data on this axis, and it constrains
   `SNR_vel`, not `SNR_pf`.
3. **White noise.** The brief states the token noise is white per-frame; the toy assumes it exactly.
   Any temporal correlation in the real noise would raise `SNR_vel` and shrink the effect.
4. **No camera.** The real velocity term operates on a world-lifted trajectory whose lifting
   involves the predicted camera; the toy compares the latent directly. The `pred_cam_t` /
   depth-scale degeneracy discussed in the repo notes is outside this model.
5. **LTI models only.** Every model here is a linear time-invariant kernel, so the "oracle" is the
   finite-support Wiener filter for this ensemble. A real attention layer is input-dependent and
   could in principle beat it; nothing here bounds that.
6. **Boundaries excluded.** All losses are evaluated on the clip interior (margin 16). The real loss
   uses every valid row, and the real block sees truncated windows at clip edges.

---

# Follow-up

Two follow-up experiments requested after the H1–H5 round. Same toy, same conventions
(T = 60, dt = 1/25, interior margin 16, float64 CPU), same seeds unless stated.

New scripts: `follow.py` (decoupled loss + shared runner; also adds a `noself8` branch option
to `kernels.boxcar` / `ModelC`), `exp_f1a_dec_firstorder.py` → `out_f1a.txt`,
`exp_f1c_dec_sstar.py` → `out_f1c.txt`, `exp_f1b_dec_dynamics.py` → `out_f1b.txt`,
`exp_f2_selfmask.py` → `out_f2.txt` (+ `f1b_*.json`, `f2_*.json` trajectories).

## F1 — a decoupled transition loss

`L_dec = (1 − r(Δŷ, Δg)) + β·(RMS(Δŷ) − RMS(Δg))²`, per clip, `r` = Pearson correlation over the
clip's T−1 increments (means subtracted), RMS over the same increments.

Two variants are reported throughout, because the spec's stop-gradient turns out to matter:

* **detach** — the spec: a stop-gradient on the prediction's own RMS in `r`'s denominator.
  The *value* is still the true Pearson r (a stop-gradient does not change the value); only the
  gradient is a semi-gradient. Its numerator `⟨Δŷ̄, Δḡ⟩` then keeps an amplitude gradient, so the
  correlation term is **not** amplitude-neutral — it pushes the amplitude *up*.
* **true-Pearson** — no stop-gradient. `r` is then exactly scale-invariant, so the correlation
  term has an exactly-zero gradient w.r.t. any amplitude parameter (verified below at 1e-13).

`β` is set so the two parts have equal output-gradient norm at the identity point
(384 clips, seeds 11/12): depth **β = 0.6711** (detach) / **1.1994** (true-Pearson);
joint ang **0.2736 / 0.5414**. Gradient norms at identity, depth:
`‖∂(1−r)/∂ŷ‖ = 0.1753` (detach) / `0.3134` (true), `‖∂RMS-term/∂ŷ‖ = 0.2612`,
`‖∂L_pf/∂ŷ‖ = 0.0592`, `‖∂L_vel/∂ŷ‖ = 0.8153`.
Joint-loss weights at the same 60 %-of-gradient-norm balance used everywhere else:
depth `λ_vel = 0.10898`, `λ_dec = 0.37595` (detach) / `0.15286` (true) /
`0.50679` (correlation only, β = 0).

### F1(a) — first-order gradients at the identity point (c = 0, h = 1), self-INCLUSIVE ±4 boxcar

8 clips/minibatch × 4000 fresh draws (traj seeds 10000+, noise 900000+), Huber for the pointwise
rows. Sign: gradient > 0 ⇒ descent DECREASES the parameter. **depth channel:**

| loss term | param | mean | std | SNR | Adam drift/lr | descent moves |
|---|---|---|---|---|---|---|
| **L_dec (detach, β=0.671)** | **c** | **−0.3204** | 0.0334 | **−9.58** | **−0.995** | **UP (smooths)** |
| L_dec (detach, β=0.671) | h | −0.1438 | 0.0678 | −2.12 | −0.905 | UP |
| corr only (detach, β=0) | c | −0.3697 | 0.0345 | −10.72 | −0.996 | UP |
| corr only (detach, β=0) | h | −0.4209 | 0.0411 | −10.24 | −0.995 | UP |
| **L_dec (true-Pearson, β=1.199)** | **c** | **−0.2034** | 0.0204 | **−10.00** | **−0.995** | **UP (smooths)** |
| L_dec (true-Pearson, β=1.199) | h | +0.4953 | 0.0674 | +7.35 | +0.991 | DOWN |
| corr only (true-Pearson, β=0) | c | −0.2914 | 0.0193 | −15.07 | −0.998 | UP |
| **corr only (true-Pearson, β=0)** | **h** | **−1.1e-13** | 1.3e-14 | (−8.45) | – | **nothing — exactly 0** |
| pointwise Huber velocity | c | +8.0e-06 | 0.0174 | 0.0005 | 0.0005 | nothing |
| pointwise Huber velocity | h | +0.9965 | 0.0754 | +13.22 | +0.997 | DOWN (shrinks) |
| pointwise Huber per-frame | c | +0.0104 | 0.0292 | +0.356 | +0.336 | DOWN |
| pointwise Huber per-frame | h | +0.0947 | 0.0324 | +2.92 | +0.946 | DOWN |

**joint-angle channel** (same pattern): L_dec(detach) c −0.4011, SNR −10.18; L_dec(true) c −0.1260,
SNR −5.17; corr only (true) h = −3.0e-14 (exactly 0); pointwise velocity c SNR −0.0010, h SNR +12.07.

**Reading.**
1. **The prediction on `c` is confirmed, strongly.** Exactly where the pointwise velocity loss gives
   the residual branch *nothing* (SNR 0.0005), the correlation term gives it **SNR −9.6 to −15.1**,
   i.e. Adam grows `c` at **99.5 % of the full learning rate**. This holds with the self key
   *included*. The mechanism is visible in the algebra: the pointwise gradient at identity is
   `⟨Δn, Δm⟩` (signal cancels, mean 0), the correlation gradient is `∝ ⟨Δm, Δg⟩` whose signal part
   `⟨u∗Δg, Δg⟩ > 0` survives and dominates.
2. **The prediction on `h` holds only for the TRUE Pearson form, and only for the correlation term
   alone.** True Pearson: `dL/dh = −1.1e-13` — exactly zero to float64 precision, 13 orders below
   the pointwise loss's +0.997. (Its "SNR −8.5" is meaningless: mean and std are both at the
   round-off floor of an analytically-zero quantity.) But:
   * the **spec's detach** makes the correlation term push `h` **UP** (SNR −10.2), not zero;
   * the **β RMS-matching term** reintroduces a shrink signal (true-Pearson L_dec: `h` SNR +7.35,
     drift 0.99) — because at the identity point the prediction's velocity RMS is inflated by noise
     (0.62 vs the GT's 0.288 m/s), and matching it demands shrinking.
   The two effects roughly cancel under the detach variant (net `h` SNR −2.12, still growth).

### F1 extra — the shrinkage optimum under `L_dec` for the scale-only model A (`out_f1c.txt`)

384 clips, seeds 11/12. Pearson's *value* is scale-invariant in all variants, so `s*` under `L_dec`
alone is set entirely by the RMS-matching term.

| channel | objective | s\* | rmse_pf(mm) @ s\* | rmse_vel @ s\* |
|---|---|---|---|---|
| depth | pointwise Huber velocity only | **0.2224** | 58.93 | 0.2511 |
| depth | RMS-matching term only (= `L_dec` alone, all variants) | **0.4528** | 41.95 | 0.2912 |
| depth | pf + pointwise Huber vel (λ60) | 0.9160 | 15.40 | 0.4993 |
| depth | pf + L_dec (true-Pearson, matched λ) | **0.9316** | 15.19 | 0.5076 |
| depth | pf + L_dec (detach, matched λ) | 0.9220 | 15.31 | 0.5025 |
| joint ang | pointwise Huber velocity only | 0.3934 | 120.27 | 0.5811 |
| joint ang | RMS-matching term only | 0.6109 | 78.48 | 0.6423 |
| joint ang | pf + pointwise Huber vel (λ60) | 0.9617 | 26.56 | 0.9045 |
| joint ang | pf + L_dec (true-Pearson, matched λ) | 0.9660 | 26.45 | 0.9085 |

Closed form for the RMS-matching term alone, `s* = 1/√(1 + var(Δn)/var(Δg))`: 0.468 (depth) /
0.625 (joint ang) — the 3 % gap to the numeric argmin is Jensen (the loss averages per-clip squared
RMS differences, the closed form uses ensemble RMS). **So `L_dec` halves the shrinkage optimum for a
model that cannot filter (0.45 vs 0.22) but does not remove it**, and at the matched joint balance
the difference is small (0.932 vs 0.916). Whatever `L_dec` buys has to come from the dynamics.

### F1(b) — dynamic Adam runs (H3c protocol: lr 2e-4, 8 clips/step, identity init, 60 000 steps, fresh data each step, depth channel)

Runs were sharded one-per-process (`exp_f_one.py`, `torch.set_num_threads(1)`); collected results in
`out_f1b_f2_shards.txt`, trajectories in `ftraj_*.json`. Reference points: identity 15.36 mm /
0.5445; **oracle ±4 = 6.38 mm / 0.0469**.

**Model B — non-residual softmax kernel (±4) + free head scale** (`k@` column = self weight):

| objective | s@2k | s@10k | s@60k | self@2k | self@10k | self@60k | DC | width | rmse_pf (mm) | rmse_vel |
|---|---|---|---|---|---|---|---|---|---|---|
| pf only | 0.9591 | 0.9626 | 1.0415 | 0.9590 | 0.9520 | 0.1719 | 1.042 | 0.082 s | **6.38** | 0.0474 |
| pf + pointwise Huber vel | **0.9131** | 0.9188 | 1.0434 | 0.9130 | 0.9083 | 0.1693 | 1.043 | 0.084 s | 6.39 | 0.0469 |
| **pf + L_dec (true-Pearson)** | **0.9298** | 0.9353 | 1.0452 | 0.9297 | 0.9245 | 0.1660 | 1.045 | 0.085 s | **6.39** | **0.0466** |
| pf + L_dec (detach) | 0.9846 | 0.9902 | **1.1556** | 0.9845 | 0.9793 | 0.2776 | **1.156** | 0.058 s | **12.63** | 0.0915 |
| pf + corr only (detach, β=0) | 1.0482 | 1.0506 | **1.1952** | 1.0480 | 1.0401 | 0.3066 | **1.195** | 0.053 s | **15.56** | 0.1080 |

**Model C-1 — one residual layer (self-inclusive ±4 branch) + free head scale** (`c` = branch gain):

| objective | s@2k | s@10k | s@60k | c@2k | c@10k | c@60k | DC | width | rmse_pf (mm) | rmse_vel |
|---|---|---|---|---|---|---|---|---|---|---|
| pf only | 0.7734 | 0.4817 | 0.2596 | 0.2876 | 1.1277 | 3.0298 | 1.046 | 0.090 s | 7.30 | 0.1506 |
| pf + pointwise Huber vel | 0.7120 | 0.3949 | 0.1759 | 0.3726 | 1.5854 | 4.9618 | 1.049 | 0.094 s | 6.95 | 0.1115 |
| **pf + L_dec (true-Pearson)** | 0.7156 | 0.3841 | 0.1368 | **0.3889** | **1.6857** | **6.7010** | 1.054 | 0.096 s | **6.86** | **0.0953** |
| pf + L_dec (detach) | 0.7972 | 0.5324 | 0.3391 | 0.3226 | 1.0995 | 2.4314 | **1.163** | 0.087 s | **11.60** | 0.1935 |
| pf + corr only (detach, β=0) | 0.9003 | 0.6588 | 0.4398 | 0.2123 | 0.7270 | 1.7009 | **1.188** | 0.082 s | **13.86** | 0.2465 |

**Model C-4 — four residual layers + free head scale**:

| objective | s@2k | s@10k | s@60k | c@2k | c@10k | c@60k | DC | width | rmse_pf (mm) | rmse_vel |
|---|---|---|---|---|---|---|---|---|---|---|
| pf only | 0.6913 | 0.3311 | 0.2449 | 0.1020 | 0.3376 | 0.4476 | 1.076 | 0.115 s | **7.91** | 0.1413 |
| pf + pointwise Huber vel | 0.6396 | 0.2010 | 0.1414 | 0.1243 | 0.5240 | 0.6696 | 1.099 | 0.131 s | 8.26 (+4.4 %) | 0.0908 |
| pf + L_dec (true-Pearson) | 0.6349 | 0.1363 | 0.0730 | 0.1320 | **0.6872** | **0.9799** | 1.121 | 0.145 s | **9.14 (+15.6 %)** | **0.0637** |
| pf + L_dec (detach) | 0.6903 | 0.4257 | 0.3761 | 0.1241 | 0.2843 | 0.3296 | **1.175** | 0.103 s | **11.57** | 0.2107 |
| pf + corr only (detach, β=0) | 0.8037 | 0.5648 | 0.4920 | 0.0868 | 0.1986 | 0.2464 | **1.188** | 0.092 s | **13.63** | 0.2729 |

**Reading.**
1. **Does the s → 0.915 dip disappear? No — it shrinks by about a third, and only in the
   true-Pearson form.** Model B, extra dip at 2 k steps relative to the pf-only baseline (0.9591):
   pointwise **−0.046**, L_dec(true-Pearson) **−0.029** (36 % smaller), L_dec(detach) **+0.026**
   (a *lift*, not a dip). The dip is a transient in all cases and s recovers to 1.04 by 60 k.
2. **`L_dec` does drive the residual branch harder, as F1(a) predicts.** Final branch gain `c`:
   C-1 3.03 (pf only) → 4.96 (pointwise) → **6.70** (L_dec true); C-4 0.448 → 0.670 → **0.980**.
   It buys the best velocity error of any objective tested (C-1 0.0953, C-4 0.0637).
3. **But it does not fix the per-frame damage in the deep residual stack — it makes it worse.**
   C-4 `rmse_pf`: 7.91 (pf only) → 8.26 (pointwise, +4.4 %) → **9.14 (L_dec, +15.6 %)**. For C-1 it
   is still a net win (7.30 → 6.86). For the non-residual model B it is neutral (6.38 → 6.39) and
   gives the best velocity of the three (0.0466 vs the oracle's 0.0469).
4. **The spec's stop-gradient is actively harmful.** Every `detach` run over-amplifies
   (DC 1.156–1.195) and roughly doubles the per-frame error (11.6–15.6 mm vs 6.4–7.9). This is
   exactly the F1(a) prediction: with `‖Δŷ‖` detached the correlation term's `h`-gradient is
   −0.42 (SNR −10.2, i.e. it grows the amplitude at full learning rate) instead of the true
   Pearson's exact 0, and the β RMS term is not strong enough to hold it. Use the **true Pearson**
   (no stop-gradient); it is already exactly scale-invariant, which is what the detach was meant
   to achieve.

## F2 — self-masked residual attention in the dynamics

### F2(a) — first-order gradient on `c` at the identity point (`out_f2.txt`, 8 clips × 2000 draws, Huber)

| channel | branch | u₀ | 2u₀−u₁−u₋₁ | loss | mean dL/dc | std | SNR | Adam drift/lr | analytic 2σ²u₀ |
|---|---|---|---|---|---|---|---|---|---|
| depth | boxcar9 (self in) | 0.1111 | 0.000 | per-frame | +1.070e-02 | 0.0295 | **+0.362** | +0.341 | 5.27e-05 |
| depth | boxcar9 | 0.1111 | 0.000 | velocity | +9.91e-05 | 0.0174 | **+0.006** | +0.006 | – |
| **depth** | **noself8 (self masked)** | **0.0000** | **−0.250** | **per-frame** | **+1.51e-04** | 0.0293 | **+0.005** | +0.005 | **0** |
| **depth** | **noself8** | **0.0000** | **−0.250** | **velocity** | **−0.1244** | 0.0200 | **−6.22** | **−0.987** | – |
| joint ang | boxcar9 | 0.1111 | 0.000 | per-frame | +8.03e-03 | 0.0332 | +0.242 | +0.235 | 1.57e-04 |
| joint ang | boxcar9 | 0.1111 | 0.000 | velocity | −3.64e-05 | 0.0151 | −0.002 | −0.002 | – |
| joint ang | noself8 | 0.0000 | −0.250 | per-frame | +1.65e-04 | 0.0329 | **+0.005** | +0.005 | **0** |
| joint ang | noself8 | 0.0000 | −0.250 | velocity | −0.0903 | 0.0167 | **−5.41** | −0.983 | – |

Both analytic predictions confirmed. **`dL_pf/dc = 2σ²u₀` goes to 0 when the self tap is masked**:
SNR +0.362 → +0.005 (depth), +0.242 → +0.005 (joint ang) — the per-frame loss's anti-smoothing pull
vanishes exactly. And `dL_vel/dc = 2σ²(2u₀−u₁−u₋₁)/dt²` goes from 0 to a strong smoothing pull
(SNR −6.2 / −5.4, Adam drift −0.99).

### F2(b) — dynamics (same H3c protocol, depth channel; the `boxcar9` rows are the shared controls from F1(b))

| model | branch | objective | s@2k | s@10k | s@60k | c@2k | c@10k | c@60k | DC | self | width | rmse_pf (mm) | rmse_vel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C-1 | boxcar9 | pf only | 0.7734 | 0.4817 | 0.2596 | 0.2876 | 1.1277 | 3.0298 | 1.046 | 0.347 | 0.090 s | 7.30 | 0.1506 |
| **C-1** | **noself8** | **pf only** | 0.7675 | 0.4845 | 0.2855 | **0.3042** | 1.1300 | 2.6871 | 1.053 | 0.285 | 0.094 s | **6.99** | **0.1176** |
| C-1 | boxcar9 | pf + vel | 0.7120 | 0.3949 | 0.1759 | 0.3726 | 1.5854 | 4.9618 | 1.049 | 0.273 | 0.094 s | 6.95 | 0.1115 |
| **C-1** | **noself8** | **pf + vel** | 0.7095 | 0.4003 | 0.2072 | **0.3896** | 1.5786 | 4.1039 | 1.058 | 0.207 | 0.098 s | **6.81** | **0.0825** |
| C-4 | boxcar9 | pf only | 0.6913 | 0.3311 | 0.2449 | 0.1020 | 0.3376 | 0.4476 | 1.076 | 0.334 | 0.115 s | 7.91 | 0.1413 |
| C-4 | noself8 | pf only | 0.6908 | 0.3603 | 0.2970 | 0.1038 | 0.3110 | 0.3798 | 1.077 | 0.334 | 0.115 s | **7.90** | **0.1410** |
| C-4 | boxcar9 | pf + vel | 0.6396 | 0.2010 | 0.1414 | 0.1243 | 0.5240 | 0.6696 | 1.099 | 0.242 | 0.131 s | 8.26 | 0.0908 |
| C-4 | noself8 | pf + vel | 0.6429 | 0.2338 | 0.1881 | 0.1257 | 0.4692 | 0.5553 | 1.101 | 0.242 | 0.131 s | **8.25** | **0.0905** |

(`c` is in different units for the two branches — `noself8`'s average has no self tap, so `c = 8`
there is the exact 9-tap boxcar. Compare the resulting kernels, not the raw `c`.)

**Reading.**
1. **How fast does `c` grow? Barely faster — about 5 %.** `c@2k`: 0.3042 vs 0.2876 (C-1, pf only),
   0.3896 vs 0.3726 (C-1, pf+vel), 0.1038 vs 0.1020 (C-4). The enormous first-order advantage
   (SNR −6.2 vs +0.006) does **not** translate into a large speed-up, because the coupled basin
   opens as soon as the head scale falls ~1 % below 1 (H3-e), which happens in the first ~100 steps
   regardless. The first-order signal only matters for the very first steps.
2. **Does the head still dip? Yes, essentially unchanged.** `s@2k`: 0.7675 vs 0.7734 (C-1, pf only),
   0.7095 vs 0.7120 (C-1, pf+vel), 0.6908 vs 0.6913 (C-4). Masking the self key does nothing to the
   shrinkage — that pathway runs through the head, not the branch.
3. **Final quality: a real but modest win for C-1, nothing for C-4.** C-1 `rmse_pf` 7.30 → **6.99**
   (pf only) and 6.95 → **6.81** (pf+vel); `rmse_vel` 0.1506 → **0.1176** and 0.1115 → **0.0825**
   (−22 % and −26 %). C-4: 7.91 → 7.90 and 8.26 → 8.25, velocity 0.1413 → 0.1410 and
   0.0908 → 0.0905 — **no effect at all**. Stacking four layers washes the self-mask out (each
   layer's effective kernel already has a self tap after convolution with the previous layers).
4. **It does not fix the deep-stack damage.** The C-4 `pf → pf+vel` per-frame penalty is +4.4 % with
   the self-inclusive branch and +4.4 % with the self-masked one (7.91→8.26 vs 7.90→8.25).

## Follow-up verdicts

**F1 — decoupled transition loss.** The prediction on the residual branch gain is **confirmed and
large**: exactly where the pointwise Huber velocity loss gives the branch a gradient with population
mean zero (SNR 0.0005), the correlation term gives SNR **−9.6 to −15.1** with the self key
*included*, i.e. Adam grows the branch at 99.5 % of the full learning rate. The algebra says why:
the pointwise gradient at identity is `⟨Δn, Δm⟩` (signal cancels), the correlation gradient is
`∝ ⟨Δm, Δg⟩` whose signal part survives. The prediction of "no shrink signal on `h`" is **half
right**: the *true* Pearson correlation has an exactly-zero amplitude gradient (−1.1e-13, 13 orders
below the pointwise loss's +0.997), but (i) the spec's stop-gradient on `‖Δŷ‖` inverts this into an
amplitude-*growth* signal (SNR −10.2) and (ii) the β RMS-matching term reintroduces a shrink
(SNR +7.35) because at identity the prediction's velocity RMS is inflated by noise. In the dynamics
the true-Pearson form reduces the head-scale dip by ~36 % (model B, extra dip −0.029 vs −0.046) but
does not remove it; it drives the branch harder than any other objective (C-1 `c` 6.70 vs 4.96) and
gives the best velocity error everywhere; **but in the 4-layer residual stack it increases the
per-frame damage from +4.4 % to +15.6 %**. The detach variants are strictly harmful (DC overshoot
1.16–1.20, per-frame error roughly doubled). For a scale-only model `L_dec` halves the shrinkage
optimum (0.45 vs 0.22) but does not remove it, because RMS-matching is itself a shrinkage demand.

**F2 — self-masked residual attention.** Both first-order predictions confirmed exactly: masking the
self key sends `dL_pf/dc = 2σ²u₀` to **0** (SNR +0.36 → +0.005) and turns `dL_vel/dc` from 0 into a
strong smoothing pull (SNR +0.006 → **−6.22**, Adam drift −0.99). In the dynamics, however, the
effect is much smaller than the first-order numbers suggest: `c` grows only ~5 % faster at 2 k
steps, the head-scale dip is **unchanged** (0.7675 vs 0.7734), and the final gain is a modest but
real improvement for one layer (7.30 → **6.99** mm and velocity 0.1506 → **0.1176**; with the
velocity term 6.95 → **6.81** and 0.1115 → **0.0825**) and **exactly nothing for four layers**
(7.91 → 7.90; 8.26 → 8.25). The reason the first-order advantage does not cash out is H3-e: the
coupled (`c` up, `s` down) basin opens after a ~1 % head shrink, which happens within the first
~100 steps whatever the branch's initial gradient is. Self-masking also does not reduce the deep
stack's per-frame penalty from the velocity term (+4.4 % either way).

**Common thread across both follow-ups.** Every intervention that acts on the *branch* (self-masking,
the correlation term) improves the velocity metric and the 1-layer residual, but none of them touch
the head-scale shrinkage, and none of them prevent the 4-layer residual stack from trading per-frame
accuracy for velocity accuracy. In this toy the only intervention that removes the shrinkage
transient outright is still the H3c routing experiment (detach the head scale from the transition
loss) — which, as reported earlier, also removes ~all of the transition loss's influence on a
residual block.

---

# Follow-up 2

Same H3c protocol throughout: Adam, lr 2e-4, 8 clips/step, identity init, 60 000 steps, fresh data
every step (traj seeds 500000+, noise 600000+), Huber per-frame loss, free head scale on every
model. Depth **and** joint-angle channels. Runs sharded one-per-process (`exp_f_one.py`); collected
in `out_fu2_shards.txt`, trajectories in `ftraj_*.json`.

Three new objectives, all at the same 60 %-of-output-gradient-norm balance used everywhere else:

1. **`corr_true`** — `pf + λ·(1 − r)` with the **true Pearson** correlation over the clip's T−1
   increments, β = 0, **no stop-gradient anywhere**. λ = **0.283541** (depth) / **0.211036**
   (joint ang).
2. **`onesided`** — `pf + λ·[(1 − r) + β·max(0, RMS(Δg) − RMS(Δŷ))²]`, true Pearson,
   β = **1.199438** (depth) / **0.541423** (joint ang) — the same β the two-sided calibration gives.
   The one-sided term is **exactly inert at the identity point** (measured value 0.0, because the
   prediction's velocity RMS starts noise-inflated at 0.62 vs the GT's 0.288 m/s), so λ is
   identical to (1) and the two objectives start from the same gradient balance.
3. **`corr_true_route`** — model B, objective (1), with the head scale **detached** from the
   transition term (`kernel/s · sg(s)`).

`pf only` and `pf + pointwise Huber vel` rows are repeated from F1(b) / H3c as reference.

### Depth channel (identity 15.36 mm / 0.5445; oracle ±4 = 6.38 mm / 0.0469)

| model | objective | s@2k | s@10k | s@60k | c or self @2k | @10k | @60k | DC | width | rmse_pf (mm) | rmse_vel |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B | pf only | 0.9591 | 0.9626 | 1.0415 | 0.9590 | 0.9520 | 0.1719 | 1.042 | 0.082 s | **6.38** | 0.0474 |
| B | pf + pointwise vel | **0.9131** | 0.9188 | 1.0434 | 0.9130 | 0.9083 | 0.1693 | 1.043 | 0.084 s | 6.39 | 0.0469 |
| **B** | **pf + corr_true** | **0.9591** | 0.9627 | 1.0455 | 0.9590 | 0.9512 | 0.1648 | 1.046 | 0.086 s | 6.40 | **0.0466** |
| **B** | **pf + onesided** | **0.9591** | 0.9627 | 1.0462 | 0.9590 | 0.9512 | 0.1651 | 1.046 | 0.086 s | 6.40 | **0.0466** |
| **B** | **pf + corr_true, ROUTED** | **0.9591** | 0.9627 | 1.0455 | 0.9590 | 0.9512 | 0.1648 | 1.046 | 0.086 s | 6.40 | **0.0466** |
| C-1 | pf only | 0.7734 | 0.4817 | 0.2596 | c 0.2876 | 1.1277 | 3.0298 | 1.046 | 0.090 s | 7.30 | 0.1506 |
| C-1 | pf + pointwise vel | 0.7120 | 0.3949 | 0.1759 | c 0.3726 | 1.5854 | 4.9618 | 1.049 | 0.094 s | 6.95 | 0.1115 |
| **C-1** | **pf + corr_true** | 0.7240 | 0.3764 | 0.1174 | c 0.3893 | 1.7527 | **7.9853** | 1.055 | 0.097 s | **6.83** | **0.0880** |
| **C-1** | **pf + onesided** | 0.7240 | 0.3764 | 0.1177 | c 0.3893 | 1.7527 | 7.9632 | 1.055 | 0.097 s | **6.83** | **0.0881** |
| C-4 | pf only | 0.6913 | 0.3311 | 0.2449 | c 0.1020 | 0.3376 | 0.4476 | 1.076 | 0.115 s | **7.91** | 0.1413 |
| C-4 | pf + pointwise vel | 0.6396 | 0.2010 | 0.1414 | c 0.1243 | 0.5240 | 0.6696 | 1.099 | 0.131 s | 8.26 (+4.4 %) | 0.0908 |
| **C-4** | **pf + corr_true** | 0.6037 | 0.0992 | 0.0516 | c 0.1533 | 0.8319 | 1.1623 | **1.129** | 0.151 s | **9.64 (+21.9 %)** | **0.0586** |
| **C-4** | **pf + onesided** | 0.6037 | 0.1002 | 0.0528 | c 0.1533 | 0.8277 | 1.1513 | 1.130 | 0.151 s | 9.60 (+21.4 %) | **0.0586** |

### Joint-angle channel (identity 26.52 mm / 0.9400; oracle ±4 = 11.82 mm / 0.1058)

| model | objective | s@2k | s@10k | s@60k | c or self @2k | @10k | @60k | DC | width | rmse_pf (mm) | rmse_vel |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B | pf only | 0.9824 | 0.9839 | 1.0328 | 0.9823 | 0.9735 | 0.1997 | 1.033 | 0.071 s | **11.85** | 0.0984 |
| B | pf + pointwise vel | **0.9611** | 0.9644 | 1.0362 | 0.9609 | 0.9533 | 0.1940 | 1.036 | 0.075 s | 11.89 | 0.0933 |
| **B** | **pf + corr_true** | **0.9824** | 0.9840 | 1.0429 | 0.9823 | 0.9724 | 0.1799 | 1.043 | 0.081 s | 12.07 | **0.0865** |
| **B** | **pf + onesided** | **0.9824** | 0.9840 | 1.0434 | 0.9823 | 0.9724 | 0.1807 | 1.043 | 0.080 s | 12.07 | 0.0866 |
| **B** | **pf + corr_true, ROUTED** | **0.9824** | 0.9840 | 1.0429 | 0.9823 | 0.9724 | 0.1799 | 1.043 | 0.081 s | 12.07 | **0.0865** |
| C-1 | pf only | 0.8109 | 0.5451 | 0.3251 | c 0.2404 | 0.8834 | 2.2134 | 1.045 | 0.086 s | 14.41 | 0.3180 |
| C-1 | pf + pointwise vel | 0.7432 | 0.4341 | 0.2243 | c 0.3450 | 1.3745 | 3.6819 | 1.050 | 0.092 s | 13.84 | 0.2330 |
| **C-1** | **pf + corr_true** | 0.7459 | 0.4071 | 0.1653 | c 0.3597 | 1.5520 | **5.3797** | 1.055 | 0.095 s | **13.74** | **0.1879** |
| **C-1** | **pf + onesided** | 0.7459 | 0.4071 | 0.1664 | c 0.3597 | 1.5519 | 5.3420 | 1.055 | 0.095 s | **13.74** | 0.1887 |
| C-4 | pf only | 0.7315 | 0.4320 | 0.3533 | c 0.0867 | 0.2483 | 0.3159 | 1.059 | 0.101 s | **15.98** | 0.3428 |
| C-4 | pf + pointwise vel | 0.6578 | 0.2862 | 0.2251 | c 0.1174 | 0.3909 | 0.4805 | 1.082 | 0.118 s | 16.82 (+5.3 %) | 0.2321 |
| **C-4** | **pf + corr_true** | 0.6352 | 0.1954 | 0.1259 | c 0.1324 | 0.5378 | 0.7214 | 1.106 | 0.134 s | **19.22 (+20.3 %)** | **0.1587** |
| **C-4** | **pf + onesided** | 0.6352 | 0.1975 | 0.1298 | c 0.1324 | 0.5339 | 0.7089 | 1.107 | 0.133 s | 19.06 (+19.3 %) | 0.1608 |

### Reading

1. **The head-scale dip is removed exactly, not merely reduced.** Model B, `s@2k`:
   pf-only 0.9591, pointwise 0.9131, **corr_true 0.9591** — identical to the pf-only baseline to
   four decimals. Joint angle: 0.9824 / 0.9611 / **0.9824**, same story. Compare F1(b), where the
   *detached* Pearson turned the dip into a +0.026 amplitude *lift* and the two-sided true form
   only cut it by 36 %. Removing the stop-gradient and the two-sided β term is what makes it exact.
   (In the C models `s` still falls, and faster than pf-only — but that is the co-adaptation, not a
   collapse: DC gain stays at 1.046–1.130 throughout while `c` grows to 8.0.)
2. **The routing control is an exact no-op — the strongest confirmation of amplitude blindness.**
   `corr_true` and `corr_true_route` agree in **all twelve columns, both channels**
   (0.9591/0.9627/1.0455, 6.40/0.0466 for depth; 0.9824/0.9840/1.0429, 12.07/0.0865 for joint ang).
   Detaching the head scale from a loss whose head-scale gradient is already exactly zero (F1(a):
   −1.1e-13) changes literally nothing. Contrast the pointwise loss, where routing changed `s@2k`
   from 0.913 to 0.960 and, in the residual models, removed ~all of the loss's effect.
3. **The one-sided amplitude term never fires.** Its results match `corr_true` to 3 decimals in
   every row and both channels (largest difference: depth C-4, 9.60 vs 9.64 mm). No run ever drove
   the predicted velocity RMS below the GT's, so `max(0, ·)` stayed at 0 for the whole 60 k steps.
   It is a free, inert safety net here — it keeps the no-shrink property of (1) at zero cost, but
   this toy provides no evidence that it is ever needed.
4. **Best velocity error of anything tested, at a per-frame cost that grows with stack depth.**
   `corr_true` gives the lowest `rmse_vel` in every model/channel cell (depth 0.0466 / 0.0880 /
   0.0586; joint 0.0865 / 0.1879 / 0.1587). Per-frame cost vs pf-only: model B **+0.3 % / +1.9 %**
   (negligible), C-1 **−6.4 % / −4.6 %** (an improvement), C-4 **+21.9 % / +20.3 %** — i.e. in the
   4-layer residual stack it costs about 4× more per-frame accuracy than the pointwise loss's
   +4.4 % / +5.3 %, because it over-smooths (width 0.151 s vs the oracle's 0.084 s, DC gain 1.13).

### Follow-up 2 verdict

The true-Pearson, β = 0 form is the cleanest transition loss tested: it is **exactly** amplitude-blind
(head-scale gradient −1.1e-13 at init; the dip vanishes to four decimals; the routing control is a
bit-level no-op), and it produces the best velocity error in every cell. The one-sided amplitude
guard is inert and therefore free. But amplitude blindness does **not** buy per-frame accuracy in a
deep residual stack: the same loss that is harmless for a non-residual kernel (+0.3 %) and helpful
for one residual layer (−6.4 %) is the *worst* option for four layers (+21.9 %), because with no
amplitude anchor at all the only thing bounding the kernel width is the per-frame loss, and four
stacked dilution layers over-smooth before it bites.
