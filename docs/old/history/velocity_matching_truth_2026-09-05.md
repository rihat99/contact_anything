# Why velocity matching fights the per-frame losses (2026-09-05)

Question (user, after stopping the `tvel_*` pair): the velocity-matching terms and the per-frame
terms share the same GT, so they should help each other; instead the velocity terms cost pose
accuracy and pushed the body toward the camera. Is matching in the GT body frame conceptually
wrong? The real goal is *transition-dynamics regularisation*: the pose token carries per-frame
white noise and the temporal block should learn to remove it — it does not. No training in this
round: checkpoint forensics, a kernel probe on the frozen token, a toy model of the losses, and a
literature sweep. Raw tables: `docs/velocity_truth/` (copies of the analysis outputs).

## 0. Summary

1. **Both losses have the same minimiser only for a model that filters across time.** For a
   per-frame estimate `x_t = g_t + n_t` with white noise, the position loss wants `x_t` almost
   unchanged (its signal-to-noise ratio is ~100), while the one-step velocity loss wants the
   increment SHRUNK by `SNR_v / (1 + SNR_v)`, where `SNR_v = var(Δg) / var(Δn)` is the
   velocity-domain SNR — below one for every term here (the frozen token's velocity RMS is 2.0–2.5×
   the GT's). The only estimator that satisfies both is the posterior-mean *trajectory* (a temporal
   filter of the tokens). Anything that reaches the velocity optimum per frame is a shortcut.
2. **The model took the shortcut: a uniform amplitude gain, not a filter.** In the stopped runs the
   predicted velocity RMS is 0.49 (joints) / 0.72 (root rotation) of the GT's (ray arm) and
   0.44 / 0.53 (CLIFF arm), versus 0.85–1.6 without velocity matching. Split at 0.2 s, the joint
   velocity is 0.57 of GT in the low band and 0.50 in the high band — a gain, not a denoiser. The
   root-rotation spread is 0.58 of the GT's, per-joint rotation spreads 0.23–0.72. Every mm of the
   MPJPE damage is low-frequency (+56 mm LF, +0.3 mm HF over the frozen model). The SMPL-X head
   regresses rotations as a zero-initialised residual on a mean pose, so "shrink toward the mean"
   is one gain the loss can set directly.
3. **Depth is the same shrinkage with the scale as the gain.** Noise ∝ depth, so the metric
   root-velocity term prefers a nearer body: the ray arm regresses to the mean depth (−934 mm on
   the 7.2 m scene, +200 mm on the 3 m scenes), the CLIFF arm sits at half depth everywhere.
   Robust CVD states the same bias for a spatial residual ("shrinking the whole scene to a point
   would achieve a minimum") and fixes it with a log-ratio.
4. **The frame is not the problem.** Transporting the predicted increment into the GT frame makes
   the comparison equal (in norm) to comparing world-frame increments; the loss IS a world-frame
   velocity loss. Nothing in the failure depends on body vs world frame.
5. **The block never filtered, in any run.** The ±4 block trained with velocity matching attends
   near-uniformly over its window but its branch gain is 0.03–0.04 of the residual stream — an
   identity with a 3 % smoothed copy added. Closing its temporal axis costs 0.08 mm MPJPE. A
   residual softmax layer can only ADD a positive average, never subtract the frame's own token, so
   filtering needs a large branch gain plus a co-adapting head. Toy model, exact: at the identity
   point the velocity loss' first-order gradient on the branch gain is `2σ²(2u₀ − u₁ − u₋₁)/dt²`
   (u = the attention kernel) — exactly ZERO for a uniform window that includes the frame's own
   token, anti-smoothing for a self-peaked kernel, and a full-speed smoothing signal (Adam drift
   0.99 of the learning rate) when the self key is MASKED; the head's amplitude gets the velocity
   gradient at SNR 13 meanwhile, and the per-frame loss' gradient on the branch is anti-smoothing
   (`2σ²u₀`). TCMR found the same architecturally (removing the residual skip: acceleration 29 → 8.7
   AND better PA-MPJPE).
6. **A compact convex kernel on the frozen token is nearly free.** Kernel probe: a single
   non-residual convex layer over ±4 frames (self weight ≤ 0.11) takes the frozen jitter from 118 to
   19–23 (Gaussian σ 0.08 s: 14) at ZERO low-frequency accuracy cost out to a kernel width of ~3
   frames. The high-frequency MPJPE term is 17–20 mm for every kernel and every trained run — the
   per-frame noise is ~5 % of the squared position error but > 90 % of the velocity / jitter error.
   A position-space loss therefore has almost nothing to gain from denoising; only a derivative
   loss carries the incentive — and the pointwise one carries the shrinkage bias with it.
7. **Consequence for the design.** Pointwise velocity matching decomposes as
   `E|Δŷ − Δg|² = (σ_ŷ − σ_g)² + 2 σ_ŷ σ_g (1 − r)`: the coupling term rewards shrinking `σ_ŷ`
   whenever `r < 1`, i.e. whenever there is noise. The shrink-free content of the loss is the
   correlation `r` (needs denoising to rise) and the amplitude match `σ_ŷ = σ_g` (forbids
   shrinking). The toy adds the other half of the picture: once a filter IS available, the joint
   optimum of the two losses is independent of the velocity weight over four decades and its head
   gain is 1.04 (no shrinkage) — the fight is a property of the per-frame-only model class, i.e. of
   the state the model is in while the block stays an identity, and of how long it stays there
   (toy: the shrink is a transient of ~10 k Adam steps at lr 2e-4; the runs were stopped at 909).
   Section 5 lists the options built on that; nothing has been changed in the code.

## 1. Setting and what was measured

Frozen SAM3D pose token → post-decoder temporal block (`model/rope.py`, 4 pre-LN residual layers,
softmax attention, RoPE on the clip row index, hard ±4-frame window, zero-init output projections)
→ from-scratch SMPL-X head (`model/heads.py::SmplxHead`: 6D rotation residuals on a fixed mean pose,
zero-init final linears; camera `ray` = pelvis ray residual on the frozen readout's ray, or `cliff`).
Losses: per-frame kp2d / kp3d / orient / pose / betas (+ depth / bearing in the ray arm, `cam` in the
CLIFF arm) and `velocity_matching` (`model/loss/velocity.py`): one-step se3 / so3 increments of the
world root and the parent-local joints, predicted increments transported into the GT frame, Huber on
value/δ, weights 8 / 8 / 40 (~60 % of the gradient norm at init). Runs: `tvel_ray` (best = epoch 8)
and `tvel_cliff` (best = epoch 3), both stopped; references `static_ray` R1 (no velocity matching;
ray depth/bearing vel + acc stencils), `tb_projzero`, `static_baseline`, the frozen SMPL-X refit.
Test set: the 16 annotated static scenes, whole scenes (3871 frames).

Analyses (scripts and full tables in `docs/velocity_truth/`):

* `forensics_results.md` — whole-scene dumps of the two velocity runs
  (`output/tvel_*/predictions/`, moved to `/data3/rikhat.akizhanov/trash/output_cleanup_20260905/` on 2026-09-05 evening), velocity terms recomputed exactly as the loss defines them,
  0.2 s low/high-frequency split, pose amplitude ratios, depth bias, MPJPE LF/HF split. Pipeline
  verified against the trainer's own eval numbers (r 0.528 / 0.795 / 0.526 at the 120-row cap).
* `block_kernel_results.md` — `scripts/diag_temporal.py` + a branch-gain hook on the trained
  blocks (`tvel_ray`, `tb_projzero`); a kernel-family probe on the frozen pose token
  (identity / Gaussian / residual-dilution / non-residual convex kernels, re-read by the frozen
  heads) with a per-frame accuracy column against the `mhr_sup_1` GT.
* `toy_results.md` — a toy trajectory + white-noise model of the losses: analytic and numerical
  shrinkage optima, first-order gradient signs of the residual and the convex block, Adam
  effective-step experiment, so3 transport identity, Huber vs L2.
* `literature_velocity_losses.md` — 19 methods: where their velocity / acceleration losses attach,
  in which frame, whether they share parameters with the per-frame head, reported trade-offs.

## 2. Theory: two losses, one GT, two different optimal estimators

### 2.1 The posterior-mean argument

Let `g_t` be the GT trajectory and `x_t = g_t + n_t` the per-frame estimate the frozen token
supports, `n_t` white. Under an L2 position loss the best output is `E[g_t | data]`; under an L2
loss on the one-step increment the best output increment is `E[Δg_t | data] = Δ E[g_t | data]`.
If "data" is the whole clip, both are the same object — the posterior-mean trajectory, i.e. a
temporal filter of the tokens — and the two losses do not fight at all. If the model only maps
each frame's token to a pose (the head) plus a small perturbation (the block at gain 0.03), "data"
is effectively `x_t` alone, and the two conditional means differ:

* position: `E[g_t | x_t] = μ + s_p (x_t − μ)` with `s_p = var(g) / (var(g) + var(n)) ≈ 1`
  (pose deviations ~0.3 rad, noise ~0.03 rad; depth 4 m vs 15 mm);
* increment: `E[Δg_t | Δx_t] = s_v Δx_t` with `s_v = var(Δg) / (var(Δg) + var(Δn))`, and the
  one-step difference doubles the noise variance while the GT increment is small.

The fight is the gap between `s_p ≈ 1` and `s_v ≪ 1`. The per-frame loss wants amplitude 1, the
velocity loss wants amplitude `s_v`, and a per-frame head can only pick one gain. Escaping the
dilemma requires conditioning on neighbours — the thing the block was built for and did not do.

### 2.2 The shrinkage factor with the measured noise

Velocity RMS of the frozen refit vs the GT (forensics, 15 scenes): root linear 2.18×, root angular
2.49×, joints 2.01× → noise-to-signal in velocity ≈ √(ratio² − 1) = 1.9 / 2.3 / 1.7, so
`SNR_v ≈ 0.28 / 0.19 / 0.35` and the velocity-only optimum `s_v ≈ 0.22 / 0.16 / 0.26`. The
per-frame terms pull back, and the runs settled at 0.72 (root angular) / 0.49 (joints) in the ray
arm and 0.53 / 0.44 in the CLIFF arm — between the two optima, as a weighted compromise must.
The Huber (δ = GT RMS) does not change this materially: at these SNRs most rows sit in its
quadratic regime (toy §5).

The cleanest signature is the Wiener relation: for a prediction that is a scaled copy of a noisy
estimate, the loss-optimal amplitude satisfies `σ_ŷ / σ_g = r`. Trainer eval of `tvel_ray`:
root_ang_vel r 0.795, amplitude ratio 0.72 (forensics, full protocol r 0.58 / ratio 0.72).

### 2.3 The decomposition that shows the incentive

For zero-mean series, `E|Δŷ − Δg|² = σ_ŷ² + σ_g² − 2 r σ_ŷ σ_g = (σ_ŷ − σ_g)² + 2 σ_ŷ σ_g (1 − r)`.
The first term wants the amplitudes equal; the second is the mismatch of SHAPE weighted by the
prediction's own amplitude — so whenever `r < 1` (noise, or imperfect tracking) the loss can be
lowered by making `σ_ŷ` smaller, without any change in `r`. That is the shrinkage incentive in one
line, and it is present in any pointwise derivative loss (velocity, acceleration, any linear
high-pass of the error). Raising `r` requires removing noise; shrinking `σ_ŷ` requires one gain in
the head. Adam follows the gradient that is available at first order (§2.5).

### 2.4 Depth: the same thing with the scale as the gain

Under a pinhole camera the pelvis' lateral position is `u · z / f`, so a depth error scales the
whole world trajectory AND its noise. With multiplicative depth noise (0.44 %/frame measured) the
metric root-velocity Huber falls monotonically as the trajectory is pulled toward the camera
(0.41 → 0.18 from scale 1 to 0.5, docs/temporal_velocity_2026-09-05.md §2.3). The ray arm has an
absolute log-depth term (weight 10) and still regressed toward the mean depth; the CLIFF arm's
only depth anchor is the proxy `cam` term and it sat at half depth from epoch 0. Dividing the
linear increment by the trajectory's own depth (or matching log-depth rates, as the retired ray
stencils did — `static_ray` reached dlogz 0.16 where `tvel_ray` stayed at 0.38 in the trainer's eval;
0.27 vs 0.52 on whole scenes) removes this
one; it does not remove the amplitude shrinkage of §2.2, which is scale-free.

### 2.5 Why the block does not filter instead

A residual softmax layer computes `x_t + P Σ_j w_j LN(x_j)` with `w ≥ 0`. It cannot subtract the
frame's own token; noise attenuation per layer is `(1 + c w_self) / (1 + c)` for a branch gain `c`,
so the block needs `c ≫ 1` and a head that rescales by `1 / (1 + c)`. Under a per-frame L2 with a
fixed head, the first-order gradient on `c` at init is ANTI-smoothing (`dL_pf/dc = 2σ²u₀`: adding
the local mean adds the neighbours' noise while the frame's own noise is untouched; the gain only
pays at second order through the head's rescale). The velocity loss' first-order gradient on `c` is
`2σ²(2u₀ − u₁ − u₋₁)/dt²` — EXACTLY zero for a uniform window that includes the self key (Stein's
lemma extends this to the Huber), positive (anti-smoothing, Adam drift 0.77–0.84 of the learning
rate) for a self-peaked kernel such as a σ = 2-frame Gaussian, and strongly negative (smoothing,
drift 0.99) when the self key is masked out. The same loss' gradient on the head amplitude has SNR
13 across minibatches (drift 1.0): a single parameter direction with a consistent sign. The toy's
Adam runs at the real learning rate show the sequence: the head shrinks first (to 0.915 for ~10 k
steps), which opens the coupled basin where the branch gain finally grows, then the amplitude
recovers to 1.04 once the kernel exists. Routing the velocity gradient to the block only (head
detached) is a near no-op for the residual block (its kernel stays at the per-frame-only
solution) and works only for a non-residual convex kernel.
Measured (`block_kernel_results.md`): the `tvel_ray` block's attention is a near-uniform 9-frame
box (self 0.14, box 0.11) with branch gain 0.032–0.034 per layer; token third differences fall by
22 % through the four layers; `bypass` (block removed) changes jitter 70 → 77 while `same_frame`
(temporal axis closed, within-frame path kept) changes MPJPE by 0.08 mm and jitter by +10. The
`tb_projzero` block (ray stencils, no velocity matching) has pose-slot gains 0.06–0.09. Every
trained block is a small perturbation of the identity; the amplitude change lives in the head.

A non-residual convex layer (values = the raw tokens, a self-logit bias, no skip) reaches its
oracle in the toy from the identity init, but NOT through a large first-order gradient: at the
identity init the smoothing gradient is exponentially suppressed and only Adam's normalisation of
a small-but-consistent signal recovers it (at a uniform init the gradient even points toward
narrowing). The one lever with a large, consistent first-order smoothing signal under the pointwise
loss is the self-masked residual attention above. The kernel probe (§3.3) shows what a convex layer
with uniform attention gives: jitter 19 at self weight 0.11, no accuracy cost. TCMR (CVPR'21) reports the same lesson for its static→temporal
residual skip: "the identity mapping of the current static feature inside the residual connection
hinders a model from learning meaningful temporal features".

### 2.6 The frame

`E_t = R_gt^T R_pred` applied to the predicted body-frame increment gives
`R_gt^T (R_pred ω_body,pred) = R_gt^T ω_spatial,pred`, and the GT's body increment is
`R_gt^T ω_spatial,gt`; the difference has the norm of the spatial (world) increment difference.
The linear part is a world-frame velocity difference up to the `V^{-1}` coupling (toy §4
quantifies it as negligible at these velocities). So the loss compares WORLD-frame root velocities
and PARENT-frame joint velocities; "GT body frame" is a description of the bookkeeping, not a
different quantity, and none of the failure above depends on it. The literature's choice
(HuMoR, GLAMR, GVHMR, WHAM, RoHM) is a heading-aligned, gravity-preserving, translation-removed
frame — which keeps gravity as an informative direction; a full body frame would discard it. On
the static subset with world-frame comparison this is moot; it matters when the root-velocity
loss is re-introduced in a body frame.

### 2.7 Gradient spectrum and Adam

The velocity loss' gradient w.r.t. the per-frame outputs is a discrete Laplacian of the residual
(gain `4 sin²(ω/2) / dt²`): ~1800× the position loss' L2 gain at Nyquist, ~0 at DC. The
δ-normalised Huber the repo uses undoes most of it (`δ_v / δ_pf = 8` → 39× at Nyquist, and BELOW
1.3 Hz the velocity term has less gain than the per-frame term). Measured at a fixed parameter
point, adding the velocity term at the 60 % balance multiplies Adam's second moment by 2.0–2.7×
uniformly — the head amplitude included — so it is a global slow-down of the per-frame terms, not a
selective suppression. This hypothesis is therefore a minor contributor; the shrinkage of §2.2–2.5
is the mechanism.

## 3. Evidence

### 3.1 Velocity terms and amplitude (forensics, 15 scenes, whole scenes, raw cameras)

Ratio = RMS(pred) / RMS(GT); huber = the loss' per-row value at its δ.

| source | root_vel ratio / r / huber | root_ang_vel ratio / r / huber | joint_ang_vel ratio / r / huber |
|---|---|---|---|
| frozen refit | 2.18 / 0.29 / 0.46 | 2.49 / 0.36 / 0.27 | 2.01 / 0.27 / 0.50 |
| static_baseline | 1.79 / 0.45 / 0.46 | 1.42 / 0.59 / 0.24 | 0.83 / 0.41 / 0.25 |
| static_ray | 1.01 / 0.71 / 0.13 | 1.58 / 0.54 / 0.25 | 0.85 / 0.41 / 0.26 |
| tb_projzero | 1.01 / 0.74 / 0.12 | 1.57 / 0.52 / 0.23 | 0.74 / 0.40 / 0.25 |
| **tvel_ray** | 1.39 / 0.51 / 0.25 | **0.72** / 0.58 / 0.15 | **0.49** / 0.54 / 0.19 |
| **tvel_cliff** | **0.75** / 0.52 / 0.18 | **0.53** / 0.60 / 0.18 | **0.44** / 0.51 / 0.20 |

Low/high-frequency split of the velocity series (Gaussian σ 0.2 s; ratio = pred/GT RMS):

| term | source | LF ratio / r | HF ratio / r |
|---|---|---|---|
| root_ang_vel | frozen | 2.18 / 0.43 | 6.04 / 0.18 |
| | static_ray | 1.79 / 0.59 | 4.03 / 0.22 |
| | tvel_ray | **0.64** / 0.69 | 1.03 / 0.38 |
| joint_ang_vel | frozen | 0.92 / 0.66 | 2.60 / 0.19 |
| | static_ray | 0.77 / 0.62 | 1.00 / 0.27 |
| | tvel_ray | **0.57** / 0.68 | **0.50** / 0.40 |

Pose amplitude, pred/GT ratio of the per-scene temporal spread: `tvel_ray` root rotation 0.58,
per-joint 0.23–0.72, pelvis-camera z 0.85; `tvel_cliff` 0.45 / 0.17–0.73 / 0.48 (x, y 0.50, 0.53:
the half depth scales everything), mean bone length 0.87; the non-velocity runs 0.95–1.05 on the
root and 0.47–0.91 per joint; the frozen refit 0.99 / 0.39–1.18.

Depth: mean bias frozen −18, static_ray +28, tb_projzero +32, tvel_ray −57, tvel_cliff −2110 mm;
tvel_ray per scene: −934 (GT 7.20 m), −446 (5.59), −370 (5.92), −327 (5.28), +202 (3.13),
+239 (2.99) — regression toward a mean depth.

MPJPE (mean-hips aligned) split at σ 0.2 s, 3871 frames:

| source | MPJPE | LF | HF |
|---|---|---|---|
| frozen refit | 59.9 | 54.1 | 16.4 |
| static_ray | 70.8 | 64.8 | 17.4 |
| tb_projzero | 76.0 | 70.0 | 17.4 |
| tvel_ray | 114.6 | 110.0 | 16.7 |
| tvel_cliff | 270.7 | 264.8 | 21.6 |

The high-frequency error is the same 16–17 mm in every run: no trained model denoised the pose, and
the velocity terms' damage is entirely low-frequency. (The 17 mm HF floor is mostly the GT's own
high-frequency content vs the σ 0.2 s band: the kernel probe's σ 0.08 s token average, which cuts
the keypoint third difference 23×, moves the HF accuracy term only 19.9 → 17.0 mm.)

### 3.2 The trained blocks (`diag_temporal.py` + branch-gain hook, 16 clips)

| run | attention self mass | RMS offset (frames) | branch gain attn / ffn per layer | token d3 ratio (4 layers) |
|---|---|---|---|---|
| tvel_ray (±4, K = 1) | 0.136–0.146 (uniform box = 0.111) | 2.2 (box 2.6) | 0.032–0.034 / 0.015–0.017 | 0.78 |
| tb_projzero (±2.5 s, K = 7) | 0.002–0.005 | 36–38 | 0.18–0.33 / 0.06–0.23 (pose slot 0.06–0.09) | 1.49 |

Ablations (`full` / `same_frame` / `bypass`): tvel_ray MPJPE 104.7 / 104.7 / 142.1, jitter
70.2 / 80.5 / 76.7, root_ang_vel r 0.79 / 0.67 / 0.69; tb_projzero MPJPE 72.3 / 72.4 / 91.6, jitter
52.5 / 116.7 / 98.8. The temporal axis buys smoothness metrics only; the within-frame path of the
block (a per-frame correction the head co-adapted to) buys the MPJPE.

### 3.3 Kernel family on the frozen token (frozen heads re-read the averaged token)

Raw crop track (the deployed pipeline): every token kernel crushes rotation / articulation noise
(rot d3 3.98 → 0.08–0.5°, keypoint d3 47.8 → 1–7 mm) but the depth noise gets WORSE
(0.54 → 0.7–1.1 %/frame): the frozen CLIFF-style camera head's `s` tracks the crop scale `b` and
their product cancels the crop jitter; smoothing the token smooths `s` but not `b`. Best jitter in
the family 96. With the crop track smoothed (σ 1 s, diagnostic only) the oracle reproduces:

| kernel | self weight | width (frames) | rot d3 (°) | kp d3 (mm) | jitter | accuracy LF / HF / total (mm) |
|---|---|---|---|---|---|---|
| identity | 1.00 | 0 | 3.81 | 47.7 | 118.2 | 74.0 / 19.8 / 80.6 |
| Gaussian σ 0.04 s | 0.36 | 1.1 | 0.47 | 6.1 | 20.1 | 74.0 / 17.9 / 80.0 |
| Gaussian σ 0.08 s | 0.18 | 2.3 | 0.14 | 2.1 | 14.2 | 74.2 / 17.0 / 80.1 |
| convex, self 0.25 + box | 0.34 | 2.2 | 1.03 | 12.6 | 34.7 | 74.2 / 17.1 / 80.0 |
| convex, self 0.11 + box | 0.21 | 2.4 | 0.57 | 6.9 | 23.0 | 74.3 / 17.0 / 80.2 |
| pure 9-box | 0.11 | 2.6 | 0.37 | 4.8 | 19.0 | 74.3 / 17.2 / 80.3 |
| residual dilution c = 32, 4 layers | 0.08 | 5.0 | 0.08 | 1.1 | (raw crops) 118 | 76.5 / 19.5 / 84.3 |

Accuracy is flat (74.0–74.3 mm LF) out to a kernel width of ~3 frames and rises only beyond ~4
(the deep residual stack is where blur starts to cost). The block could reach the oracle with ONE
convex layer; the residual family gets there only at gains it never reaches.

### 3.4 Toy model (`toy_results.md`; scalar latent, real velocity SNRs, T = 60 at 25 fps)

* Shrinkage optimum: `s* = SNR_v / (1 + SNR_v)` reproduces the numeric optimum to 3–4 decimals.
  Depth (noise velocity 1.9× GT): per-frame optimum 0.964, velocity optimum 0.218 (Huber 0.227);
  the metric velocity Huber falls 2.49× from scale 1 to 0.5 (2.28× measured on the real tracks);
  the log-depth-rate loss is flat in the scale to 4.5e-17. Root angle at 1.25× / 2.0× noise:
  0.389 / 0.199. Huber vs L2 changes `s*` by 3–7 %.
* The conflict lives in the model class: with a per-frame scale only, the velocity optimum costs
  +1200–1900 % per-frame error; with a ±4 kernel available, +1–17 %; the joint optimum's per-frame
  error moves 0.5 % over four decades of velocity weight and its head scale is 1.04–1.05.
* Gradient signs at the identity point (8-clip minibatches, 4000 draws, Huber): velocity loss on the
  head scale SNR 13.2 (Adam drift 1.0 of lr); on the residual branch gain 0.0005 (self-inclusive
  box), −6.2 (self masked, drift −0.99 = smoothing), +1.6 (self-peaked Gaussian, anti-smoothing);
  per-frame loss on the branch +0.36 (anti-smoothing).
* Dynamics at lr 2e-4, 8 clips/step, 60 k steps: the head scale dips to 0.915 for ~10 k steps
  (twice the dip without the velocity term) before the kernel takes over; final per-frame error
  with vs without the velocity term: convex kernel 6.39 vs 6.38 mm, 1-layer residual 6.95 vs 7.30
  (better), 4-layer residual 8.26 vs 7.91 (+4.4 %, the right sign but far below the +27 % MPJPE of
  the real run — the scalar toy has no direction structure and its transient is short relative to
  the real run's 909 steps). Detaching the head from the velocity term leaves the residual kernels
  at the per-frame-only solution (no-op) and removes the dip for the convex kernel.
* Transport identity: `‖E_t ω_body,pred − ω_body,gt‖ = ‖ω_spatial,pred − ω_spatial,gt‖` to 2e-15
  at every rate and pose error; the linear part equals the world velocity difference to 0.6 % RMS.
* Follow-up 1 — decoupled transition loss `(1 − r) + β (RMS(Δŷ) − RMS(Δg))²` (`toy_results.md`
  "Follow-up"). First-order at the identity point, self-INCLUSIVE box: the correlation term gives the
  residual branch a smoothing gradient of SNR −10 to −15 (Adam drift 0.995 of lr) where the
  pointwise loss gives 0.0005 — the prediction of §5.1 holds. Amplitude: a TRUE Pearson term has
  an exactly zero gradient on the head gain (1e-13); a Pearson with the prediction's RMS detached in
  the denominator inverts it into amplitude GROWTH (SNR −10) and every detach run over-amplified
  (DC gain 1.16–1.20, per-frame error doubled) — do not detach. The RMS-matching term reintroduces
  a shrink pull at init (SNR +7) because the noisy prediction's velocity RMS is inflated (0.62 vs
  0.29): filter-less optimum 0.45 instead of 0.22 — halved, not removed. Dynamics (true Pearson +
  RMS): the head dip shrinks 36 % (0.913 → 0.930), the branch is driven harder than by any other loss
  (1-layer residual c 6.7 vs 5.0), the velocity error is the lowest everywhere, the 1-layer residual's
  per-frame error improves (6.95 → 6.86 mm), but the 4-layer residual's worsens (+4.4 % → +15.6 %
  vs per-frame only: the deep stack turns a stronger branch drive into blur).
* Follow-up 2 — self-masked residual attention. First order confirmed: `dL_pf/dc` 0.36 → 0.005,
  `dL_vel/dc` 0.006 → −6.2 (full-speed smoothing). Dynamics: the branch grows only ~5 % faster
  (the coupled basin opens after a ~1 % head shrink in the first ~100 steps anyway), the head dip
  is unchanged, the 1-layer residual gains modestly (7.30 → 6.99 mm per frame, velocity −22 %), the
  4-layer stack gains nothing (7.91 → 7.90; stacking convolves a self tap back in through the
  residual path). A weak lever.
* Follow-up 3 — correlation ONLY, true Pearson, no stop-gradient (`toy_results.md` "Follow-up 2";
  depth and joint-angle channels, 60 k Adam steps at lr 2e-4). The head dip disappears EXACTLY: the
  convex model's amplitude at 2 k steps is 0.9591 with per-frame losses alone, 0.9131 with the
  pointwise velocity loss, 0.9591 with the correlation loss (joint angle 0.9824 / 0.9611 / 0.9824).
  Detaching the head from this loss changes nothing in any column (its head gradient is already
  exactly zero). A one-sided "no shrink below the GT RMS" guard never fires. Velocity error is the
  lowest of every objective in all six cells. Per-frame error vs per-frame-only training: convex
  kernel +0.3 % / +1.9 %, 1-layer residual −6.4 % / −4.6 % (an improvement), 4-layer residual
  +21.9 % / +20.3 % — with the amplitude no longer anchored by the transition loss, nothing but the
  per-frame loss bounds the kernel width, and four stacked dilution layers over-smooth (composed
  width 0.15 s vs the oracle's 0.08 s).
* Toy verdict on the levers: the pointwise loss shrinks the head first and reaches the block late;
  the plain-Pearson transition loss is amplitude-blind and drives the block hardest; what decides
  the per-frame outcome under ANY of these losses is the block's depth — a non-residual convex
  layer is neutral, one residual layer gains, four residual layers lose. Detaching the head from the
  pointwise loss removes its shrink transient for a convex kernel and is a no-op for a residual
  block. Joint-angle channel replicates every depth finding.

## 4. Literature (from `literature_velocity_losses.md`)

* **SmoothNet** (ECCV'22) ran this experiment: same L1 position + L1 acceleration objective; letting
  the acceleration gradient reach the per-frame estimator RAISES MPJPE (83.0 → 84.5 / 86.6 on
  VIBE-3DPW) while the frozen two-stage version gives accel 6.05 at MPJPE 81.4. Their words: the
  acceleration loss "can benefit Accels but harm MPJPEs … due to the optimization bottleneck between
  per-frame precision and smoothness". SmoothNet's best temporal module is a learned SIGNED linear
  filter over the window (accel 4.15), beating self-attention (6.15) and a plain Gaussian (4.95).
* **TCMR** (CVPR'21): removing the residual skip from the per-frame feature into the temporal
  feature — accel 29.2 → 8.7 with PA-MPJPE 55.6 → 54.2.
* **WHAM / GVHMR** never put a one-step difference on the pose path: a separate velocity channel is
  supervised through its INTEGRAL (multi-scale cumulative displacement at windows 1/3/9/27; GVHMR
  rolls the standardised local velocity out with GT orientation), root-velocity weights 0.001 vs
  joint 6.0 (WHAM). Joint velocity losses that do exist (MotionBERT, GLoT, D&D) subtract the pelvis
  first; nobody velocity-supervises an absolute metric root translation on a shared head.
* **Robust CVD** (CVPR'21) on metric residuals: "this biases the solution toward small depths.
  Shrinking the whole scene to a point would achieve a minimum" — fixed with a log-ratio term.
* **Frames**: HuMoR, GLAMR, GVHMR, WHAM, RoHM all use heading-aligned, gravity-preserving,
  translation-removed frames; none a full body frame.
* Not found in the literature: the `s_v = SNR_v / (1 + SNR_v)` derivation for a finite-difference
  loss, the gradient-spectrum argument, log-depth-rate supervision. Johnstone's Gaussian Estimation
  Eq. 1.4 is the scalar Wiener form to cite.

## 5. What "transition-dynamics regularisation" can look like (options, nothing built)

The requirement, from §2: a loss whose value cannot be lowered by an amplitude or scale gain of a
per-frame estimate, whose only way down is removing noise across frames, and a block that has a
first-order gradient toward filtering.

1. **Match the SHAPE of the transition, not its value** (loss only). Keep the se3 / so3 increments
   and the transport; replace the pointwise Huber by, per clip and term, `1 − r(Δŷ, Δg)` with `r` the
   plain Pearson correlation over the clip's rows (pooled over components), NO stop-gradient and NO
   amplitude term — the toy's follow-ups: the detached-denominator form inverts into amplitude
   growth, an RMS-matching term reintroduces half the shrink because the noisy prediction's velocity
   RMS starts above the GT's, and the plain form has an exactly zero amplitude gradient, removes the
   head's shrink transient exactly, is scale-invariant (no depth bias), and gives the residual branch
   a first-order smoothing gradient of SNR −10 to −15 even with the self key included. The per-frame
   losses then own the amplitude alone (their optimum is unshrunk). Cheapest change; the caveat is
   item 4b.
2. **Mask the self key in the temporal attention** (one line in `frame_keep_mask`). Flips the
   pointwise velocity loss' first-order gradient on the branch from zero to a full-speed smoothing
   signal and removes the per-frame loss' anti-smoothing one — but in the toy dynamics it barely
   pays (branch grows 5 % faster, head dip unchanged, 1-layer residual 7.30 → 6.99 mm, 4-layer
   nothing). Not worth a run on its own.
3. **Route the derivative gradient to the block only** (`torch.func.functional_call` with detached
   head parameters for the velocity term; one extra head forward). The toy says this is a near
   no-op for the residual block (its kernel does not move without the head's co-adaptation) and
   works only for a non-residual convex kernel — so on its own it repeats the `tb_stage2`
   outcome.
4. **Block depth and form** — the variable that decides the per-frame outcome under every loss in
   the toy. (a) A first non-residual convex attention layer (values = the raw tokens, self-logit
   bias initialised high, no skip): the kernel probe's convex rows and TCMR's finding; in the toy it
   is per-frame-neutral under any transition loss and reaches the oracle. Learned, data-dependent
   attention, not a fixed output kernel; whether it falls under the user's exclusion of explicit
   smoothing is the user's call. (b) Fewer residual layers: one residual layer GAINS per-frame
   accuracy under the correlation loss (−6 %), four LOSE (+21 %) because stacked dilation
   over-smooths once the amplitude is no longer anchored; the real block has four layers with
   near-uniform attention per layer (§3.2), the kernel probe puts the blur onset at ~4 frames of
   width (§3.3). A 1–2-layer block is the cheap version of (a).
5. **Scale-free root term**: linear increment divided by the trajectory's own depth (or log-depth
   rates, the retired ray stencils that reached dlogz 0.16; flat in the scale to 1e-17 in the toy).
   Necessary whichever of 1–4 is chosen if the metric root velocity stays and 1 is not adopted.
6. **Two-stage (SmoothNet recipe)**: per-frame head trained and frozen, only the block under
   per-frame + transition losses. `tb_stage2` failed with pointwise stencils and a self-inclusive
   residual block — exactly the zero-gradient configuration above; with 1 + 4 it is the cleanest
   test of "the block learns internal smoothing". Not needed for the amplitude problem if 1 is
   adopted (the correlation loss has no head gradient to route).
7. **Launched 2026-09-05 15:53 as `tvel2_ray`** (options 1 + 4b: correlation loss, two-layer block,
   RoPE on seconds; weights 0.3 / 2 / 6 — `docs/temporal_velocity_2026-09-05.md` §7): the shrink is gone
   (amplitudes ≈ GT, best pose of any velocity run at equal epoch) but the block still did not filter in
   10 epochs (gain 0.065, jitter flat) — stopped; **round 3 `tvel3_ray`** adds the block's own lr (× 10),
   `root_vel` × 10 and the non-residual convex first layer (option 4a) — §8 of the round doc.
8. **What to watch** in any relaunch: velocity RMS ratio pred/GT per term (should sit near 1, not
   0.5), the LF/HF split of the velocity and of the MPJPE error, the HF MPJPE (16–17 mm today),
   the block's branch gain and attention self weight — the four numbers that separate filtering
   from shrinking.
