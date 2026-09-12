# Trained-block diagnostics + frozen-token kernel probe (2026-09-05)

Raw numbers only. Everything below was produced on GPU 5 of the shared box, from the dev
worktree `/data3/rikhat.akizhanov/better/contact_anything_dev`, with **no training and no repo
edits**. Scripts and logs live next to this file:

| file | what |
|---|---|
| `diag_tvel_ray/`, `diag_tb_projzero/` | `scripts/diag_temporal.py` output (`summary.json`, 3 PNGs, `console.log`) |
| `branch_gain.py`, `branch_gain_*.{log,json}` | new: per-layer residual-branch gain + temporal high-pass transfer |
| `kernel_probe.py`, `kernel_probe.{log,json}` | new: kernel-family probe on the frozen pose token (raw crops) |
| `kernel_probe_box1s.{log,json}` | the same probe with the crop track Gaussian-smoothed (σ = 1 s) |

Test set for everything: the 16 annotated **static** test scenes
(`configs/datasets/climbing_videos_static.yaml`), eval protocol = one clip per (scene, person),
`eval_max_frames = 120` → **16 clips / 1918 valid frames**; median clip step 0.0357 s (28 fps).

---

## Job 1 — what the trained blocks actually do

Two checkpoints:

* **tvel_ray** — `configs/tvel_ray.yaml` + `output/tvel_ray_20260905_130637/best.pth`
  (epoch 8, stopped). Gate-less pre-LN block (`gate_init: none`), `position: index`,
  `window: 4` (hard ±4 **frames** = 9 keys/op, ±16 over 4 layers), `modalities: [pose]`
  → **K = 1 slot**. Trained WITH Lie-algebra velocity matching (8 / 8 / 40).
* **tb_projzero** — `configs/tb_projzero.yaml` + `output/tb_projzero_20260904_225421/best.pth`
  (epoch 27 = best). `gate_init: zero_proj` (LayerScale γ = 1 on zero projections),
  `position: seconds`, `window: 2.5` s (≈ ±70 frames at 28 fps), `modalities: [pose, contact]`
  → **K = 7 slots** (pose, LH, RH, LF, RF, LA, RA).

Attention recompute verified against the block's own SDPA call: max |diff| 2.4e-7 (tvel_ray),
7.6e-6 (tb_projzero).

### 1.1 Attention mass per layer (heads averaged, over all valid query rows)

`self` = mass on the query's own token; `same_fr` = mass on all K tokens of the query's own
frame; `eff_fr` = exp(entropy over FRAMES) = how many frames a query effectively reads.

**tvel_ray** (K = 1, so `self` = `same_fr`; uniform over the 9-frame window would be 0.111)

| L | self | same_fr | cross_fr | mean \|dt\| s | eff_fr | sf_min/med/max (heads) | ‖W_proj‖ | ‖W_ffn_out‖ |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.1459 | 0.1459 | 0.8541 | 0.0639 | 7.78 | 0.121 / 0.148 / 0.164 | 3.342 | 4.204 |
| 1 | 0.1416 | 0.1416 | 0.8584 | 0.0644 | 7.80 | 0.115 / 0.145 / 0.162 | 3.325 | 4.185 |
| 2 | 0.1356 | 0.1356 | 0.8644 | 0.0654 | 7.82 | 0.111 / 0.134 / 0.158 | 3.342 | 4.183 |
| 3 | 0.1360 | 0.1360 | 0.8640 | 0.0663 | 8.01 | 0.113 / 0.138 / 0.158 | 3.330 | 4.190 |

No gates (`gate_init: none` → `gamma_attn`/`gamma_ffn` are `None`). ‖slot_embed‖ = 0.681.

**tb_projzero** (K = 7)

| L | self | same_fr | cross_fr | mean \|dt\| s | eff_fr | sf_min/med/max | ‖γ_attn‖ (max\|·\|) | ‖γ_ffn‖ (max\|·\|) | ‖W_proj‖ | ‖W_ffn_out‖ |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0054 | 0.0167 | 0.9833 | 0.8422 | 60.4 | 0.0079 / 0.0168 / 0.0254 | 31.19 (1.008) | 30.77 (1.012) | 7.106 | 10.686 |
| 1 | 0.0023 | 0.0110 | 0.9890 | 0.9505 | 63.2 | 0.0041 / 0.0098 / 0.0270 | 31.07 (0.998) | 30.60 (1.000) | 7.179 | 10.205 |
| 2 | 0.0028 | 0.0121 | 0.9879 | 0.9332 | 60.5 | 0.0040 / 0.0093 / 0.0362 | 31.04 (1.004) | 30.61 (1.004) | 7.151 | 9.769 |
| 3 | 0.0031 | 0.0129 | 0.9871 | 0.9377 | 61.8 | 0.0046 / 0.0093 / 0.0261 | 30.93 (0.995) | 30.69 (1.010) | 6.776 | 9.515 |

‖slot_embed‖ per slot: pose 1.029, LH 0.791, RH 0.775, LF 0.763, RF 0.822, LA 0.820, RA 0.838.
Slot mixing, pose row (all frames): pose→pose 0.61 / 0.55 / 0.55 / 0.50 across layers, the rest
spread ≈ evenly over the six contact slots (0.05–0.12 each).

### 1.2 Mass by frame offset (heads + query slots averaged)

| run | L | \|d\|=0 | 1 | 2 | 3 | 4 | 5 | 6–10 | 11–20 | >20 | RMS offset (frames) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tvel_ray | 0 | 0.1459 | 0.2986 | 0.2811 | 0.1926 | 0.0970 | 0 | 0 | 0 | 0 | 2.15 |
| tvel_ray | 1 | 0.1416 | 0.2951 | 0.2858 | 0.1970 | 0.0959 | 0 | 0 | 0 | 0 | 2.16 |
| tvel_ray | 2 | 0.1356 | 0.2890 | 0.2906 | 0.2032 | 0.0971 | 0 | 0 | 0 | 0 | 2.18 |
| tvel_ray | 3 | 0.1360 | 0.2837 | 0.2829 | 0.2073 | 0.1059 | 0 | 0 | 0 | 0 | 2.21 |
| tb_projzero | 0 | 0.0167 | 0.0337 | 0.0355 | 0.0383 | 0.0395 | 0.0380 | 0.1636 | 0.2142 | 0.7380 | 36.1 |
| tb_projzero | 1 | 0.0110 | 0.0223 | 0.0229 | 0.0236 | 0.0244 | 0.0252 | 0.1402 | 0.2226 | 0.8639 | 37.5 |
| tb_projzero | 2 | 0.0121 | 0.0234 | 0.0228 | 0.0229 | 0.0233 | 0.0243 | 0.1504 | 0.2275 | 0.8437 | 37.3 |
| tb_projzero | 3 | 0.0129 | 0.0258 | 0.0259 | 0.0254 | 0.0252 | 0.0262 | 0.1339 | 0.2399 | 0.8397 | 37.7 |

The `|d|` columns are both signs summed (offsets +k and −k). Rows sum to 1.015 (tvel_ray) and
1.32–1.36 (tb_projzero) rather than 1.0 — a normalisation artefact of the diagnostic: each offset
is averaged over only the rows that HAVE that offset, and edge rows (whose softmax renormalises
over fewer keys) put more mass per available key. It biases the wide-window numbers upward; the
`self` / `same_frame` / `eff_frames` columns of §1.1 are computed per row and are not affected.

For reference: a flat kernel over the 9-frame window has self 0.111 and RMS offset 2.58 frames.
tvel_ray sits at self 0.136–0.146 and RMS offset 2.15–2.21 — a **slightly** peaked but essentially
near-uniform 9-frame box. tb_projzero's rows are near-uniform over its whole ±2.5 s window
(self 0.003–0.005 of the 7·(2·70+1)-token row, eff_frames ≈ 60).

### 1.3 Residual-branch gain and temporal high-pass transfer (`branch_gain.py`)

Recomputed layer by layer on the same 16 clips. `gain_attn` = RMS(Proj(Attn(LN x))·γ) / RMS(x)
over ALL slots; `pose_attn` etc. = the same restricted to the pose slot's rows.
`d1/d3_in`, `d1/d3_out` = RMS 1st / 3rd temporal difference of the **pose token** divided by that
token's own RMS (scale-free HF level); `d3_ratio` = out / in per layer (< 1 = the layer removed
per-frame HF from the token, > 1 = added).

**tvel_ray** (K = 1, so the two gain groups coincide)

| L | gain_attn | gain_ffn | gain_block | d1_in | d1_out | d1_ratio | d3_in | d3_out | d3_ratio |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0318 | 0.0174 | 0.0386 | 0.1792 | 0.1729 | 0.9649 | 0.4120 | 0.3930 | 0.9541 |
| 1 | 0.0328 | 0.0164 | 0.0389 | 0.1729 | 0.1657 | 0.9585 | 0.3930 | 0.3707 | 0.9433 |
| 2 | 0.0342 | 0.0156 | 0.0398 | 0.1657 | 0.1578 | 0.9526 | 0.3707 | 0.3455 | 0.9320 |
| 3 | 0.0342 | 0.0153 | 0.0394 | 0.1578 | 0.1504 | 0.9529 | 0.3455 | 0.3210 | 0.9290 |

Whole block: **d1 ratio 0.840, d3 ratio 0.779**. Residual-stream RMS 1.815 → 1.850 (+1.9 %).

**tb_projzero** (K = 7)

| L | gain_attn | gain_ffn | gain_block | pose_attn | pose_ffn | pose_blk | d1_in | d1_out | d1_ratio | d3_in | d3_out | d3_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.3257 | 0.2306 | 0.4228 | 0.0911 | 0.0184 | 0.0955 | 0.1765 | 0.2443 | 1.3840 | 0.4070 | 0.6292 | 1.5461 |
| 1 | 0.1779 | 0.0752 | 0.1969 | 0.0621 | 0.0151 | 0.0665 | 0.2443 | 0.2449 | 1.0023 | 0.6292 | 0.6298 | 1.0009 |
| 2 | 0.1830 | 0.0571 | 0.1965 | 0.0616 | 0.0143 | 0.0655 | 0.2449 | 0.2431 | 0.9927 | 0.6298 | 0.6260 | 0.9940 |
| 3 | 0.1801 | 0.0822 | 0.2056 | 0.0778 | 0.0140 | 0.0807 | 0.2431 | 0.2361 | 0.9711 | 0.6260 | 0.6055 | 0.9673 |

Whole block: **d1 ratio 1.337, d3 ratio 1.488** — the block leaves the pose token with MORE
relative per-frame high-frequency content than it received, essentially all of it written by
layer 0. (Its output metrics are still the best of the round: jitter 52.4. The token-space HF
level and the readout's output jitter are not the same quantity — the readout is non-linear, and
the block also moves the token's low-frequency content, which changes the denominator.)

Note on the two gain groups: over all 7 slots tb_projzero's branch gain is 0.18–0.42, but on the
**pose slot alone** it is 0.062–0.096, i.e. the large branch is mostly written into the six
contact slots. tvel_ray's pose-slot branch gain is 0.032–0.034, ~2–3× smaller than tb_projzero's.

### 1.4 Output ablations (`full` / `same_frame` / `bypass`, 16 test scenes)

`same_frame` = window closed to 1e-4 so only the query's own frame is visible (for K = 1 this
leaves the block as a per-frame non-linearity); `bypass` = every residual branch zeroed (exact
identity, heads read the raw decoder tokens).

**tvel_ray** (no contact head in this build)

| metric | full | same_frame | Δ | bypass | Δ |
|---|---|---|---|---|---|
| metric_pose/mpjpe | 104.67 | 104.74 | +0.08 | 142.12 | +37.46 |
| metric_pose/pa_mpjpe | 69.13 | 69.08 | −0.05 | 88.63 | +19.50 |
| metric_pose/pve | 122.86 | 122.96 | +0.09 | 165.90 | +43.04 |
| metric_pose/accel | 5.129 | 7.533 | +2.404 | 5.718 | +0.589 |
| metric_pose/depth_err | 239.67 | 239.69 | +0.03 | 254.72 | +15.05 |
| metric_pose/depth_bias | −32.76 | −32.72 | +0.05 | −37.47 | −4.71 |
| metric_pose/dlogz_pred (GT 0.2751) | 0.3767 | 0.3988 | +0.0221 | 0.3990 | +0.0222 |
| metric_pose/dlogz_err | 0.4038 | 0.4235 | +0.0198 | 0.4248 | +0.0210 |
| metric_global/lifted_jitter (GT 6.351) | 70.19 | 80.53 | +10.33 | 76.72 | +6.53 |
| metric_global/lifted_rte | 4.081 | 4.104 | +0.024 | 4.126 | +0.045 |
| metric_global/lifted_w_mpjpe100 | 150.10 | 151.16 | +1.06 | 186.48 | +36.37 |
| metric_global/lifted_wa_mpjpe100 | 103.42 | 103.59 | +0.17 | 125.59 | +22.17 |
| metric_velocity/root_vel r / rmse (GT rms 0.288) | 0.528 / 0.301 | 0.512 / 0.313 | −0.016 / +0.012 | 0.516 / 0.313 | −0.012 / +0.012 |
| metric_velocity/root_ang_vel r / rmse (GT rms 0.486) | 0.794 / 0.297 | 0.672 / 0.376 | −0.122 / +0.079 | 0.689 / 0.353 | −0.105 / +0.056 |
| metric_velocity/joint_ang_vel r / rmse (GT rms 0.752) | 0.525 / 0.641 | 0.435 / 0.695 | −0.091 / +0.054 | 0.444 / 0.676 | −0.081 / +0.035 |

Per-clip full-vs-same_frame output movement: joints_cam 3.58 mm mean (p90 6.96, max 19.6),
hip-aligned 2.89 mm (p90 5.39). Removing the temporal axis costs almost nothing on mpjpe
(+0.08 mm) and everything the block bought on smoothness (jitter +10.3, accel +2.4, dlogz +0.022)
and on the velocity correlations (root_ang_vel r −0.12).

**tb_projzero**

| metric | full | same_frame | Δ | bypass | Δ |
|---|---|---|---|---|---|
| metric_pose/mpjpe | 72.33 | 72.43 | +0.11 | 91.60 | +19.27 |
| metric_pose/pa_mpjpe | 50.54 | 50.34 | −0.20 | 58.89 | +8.35 |
| metric_pose/pve | 92.91 | 92.88 | −0.04 | 117.13 | +24.22 |
| metric_pose/accel | 9.342 | 10.265 | +0.923 | 9.564 | +0.222 |
| metric_pose/depth_err | 100.26 | 102.34 | +2.08 | 159.87 | +59.61 |
| metric_pose/dlogz_pred (GT 0.2776) | 0.1576 | 0.5759 | +0.4183 | 0.4826 | +0.3250 |
| metric_pose/dlogz_err | 0.2478 | 0.5678 | +0.3200 | 0.4902 | +0.2424 |
| metric_global/lifted_jitter (GT 6.351) | 52.49 | 116.66 | +64.18 | 98.80 | +46.32 |
| metric_global/lifted_rte | 3.574 | 4.966 | +1.393 | 4.488 | +0.914 |
| metric_global/lifted_w_mpjpe100 | 125.84 | 146.29 | +20.45 | 141.88 | +16.04 |
| metric_contact/f1 | 0.8744 | 0.8447 | −0.0298 | 0.6874 | −0.1871 |
| metric_contact/precision | 0.9008 | 0.8582 | −0.0426 | 0.8826 | −0.0181 |
| metric_contact/recall | 0.8496 | 0.8315 | −0.0181 | 0.5628 | −0.2868 |
| metric_pose/hand_mpjpe | 43.03 | 44.00 | +0.97 | 49.85 | +6.82 |

Per-clip full-vs-same_frame movement: joints_cam 68.4 mm mean (p90 148.7), hip-aligned 32.0 mm
(p90 54.6); contact prob |Δ| mean 0.088 (p50 0.0008, p90 0.354).

Same pattern, much larger: the temporal axis is worth ~0 mm of MPJPE (+0.11) but is responsible
for essentially all of the depth denoising (dlogz_pred 0.158 → 0.576, i.e. WORSE than the frozen
model's 0.596 once the temporal axis is closed) and 64 of the 52→117 jitter.

---

## Job 2 — kernel-family probe on the FROZEN pose token (no training)

`kernel_probe.py`, based on `scratchpad/camera/token_probe.py`. Frozen SAM3D is run on the 16
static test clips through `configs/static_ray.yaml` with the bf16 embedding cache; the decoder
pose token `out["tokens"][:, 0]` is mixed inside each clip by an arbitrary row-stochastic
[T, T] matrix over the clip's VALID frames and re-read with the FROZEN heads
(`model.wrapper.decode_pose(tok, out["ctx"])`). Nothing is trained; no repo file was touched.
The `decode_pose` / embedding-cache APIs used by the original probe are unchanged; the only edits
in my copy are the kernel family, the accuracy metric, and a `--box-gauss` option (below).

Kernels (`U9` = uniform over the ±4-frame ROW-INDEX window, edge- and validity-truncated then
renormalised; `G` = Gaussian σ = 1.5 frames truncated at ±4, likewise renormalised):

* **K0** identity, and Gaussian in SECONDS at σ ∈ {0.04, 0.08, 0.12} (the known oracle line).
* **K1** residual dilution `M = (I + c·U9)/(1+c)`, composed `L` times (matrix power).
* **K2** the same with `G` in place of `U9`.
* **K3** single-layer convex replacement `M = a·I + (1−a)·U9` — what a NON-residual attention with
  a self-logit bias produces.

Columns: `ctr` = mean of diag(M) over valid rows; `width` = sqrt(mean_q Σ_k M[q,k](k−q)²) in
FRAMES; `dep_d1/d3` = 1st/3rd difference of the pelvis log-depth ×100 (%/frame; GT d1 ≈ 0.275);
`bea_d3` bearing 3rd difference ×100; `rot_d1/d3` frozen `global_rot` successive-rotation angle,
degrees; `kp_d3` hips-relative MHR70 3rd difference, mm (all 70) and `kpb_d3` (25 body keypoints
only, fingers/face dropped); `jitter` = GVHMR lifted jitter (GT 6.35); `dep_err` = |pelvis depth −
kindyn SMPL-X pelvis depth| mm; `acc_*` = accuracy, below.

**Accuracy GT**: `batch["kp3d_world"]` — the `mhr_sup_1` MHR70 world keypoints (the SAM3D model's
OWN MHR module evaluated at the kindyn-derived GT parameters), lifted into the camera with
`cam_from_world`. That is same-rig, all-70-keypoint GT, so no kindyn↔MHR70 name mapping is needed
(the mapping the brief refers to no longer exists in `model/loss/keypoint.py`; the current
`KeypointLoss` supervises against `kp3d_world`). This required adding `"keypoints"` to the loader's
signal set — `configs/static_ray.yaml` by itself only loads `"smplx"` (the 22-joint kindyn
`smplx_joints_world`, which I still use for the `dep_err` column, exactly as the original probe
did). Error = hips-relative, **25 body keypoints**, mean mm; `acc_LF` = the error signal smoothed
along time with a σ = 0.2 s Gaussian, `acc_HF` = the residual, `acc_tot` = the unsplit mean.

### 2.1 Raw crop track (embedding cache), 16 clips / 1918 frames

Crop track, kernel-independent: RMS d1 of log bbox_scale = **1.382 %/frame**.

| kernel | ctr | width | dep_d1 | dep_d3 | bea_d3 | rot_d1 | rot_d3 | kp_d3 | kpb_d3 | jitter | dep_err | acc LF | acc HF | acc tot |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K0 identity | 1.000 | 0.00 | 0.544 | 1.367 | 0.248 | 2.15 | 3.98 | 67.1 | 47.8 | 126.4 | 99 | 74.1 | 19.9 | 80.7 |
| K0 gauss 0.04 s | 0.360 | 1.13 | 0.724 | 1.964 | 0.382 | 1.61 | 0.48 | 8.7 | 6.2 | 113.8 | 100 | 74.1 | 17.9 | 80.0 |
| K0 gauss 0.08 s | 0.182 | 2.25 | 0.903 | 2.076 | 0.401 | 1.43 | 0.14 | 2.9 | 2.1 | 116.8 | 104 | 74.2 | 17.0 | 80.1 |
| K0 gauss 0.12 s | 0.123 | 3.36 | 1.016 | 2.086 | 0.400 | 1.31 | 0.08 | 1.8 | 1.4 | 117.5 | 109 | 74.8 | 17.5 | 81.2 |
| K1 c0.5 L1 | 0.705 | 1.48 | 0.528 | 1.130 | 0.228 | 1.79 | 2.68 | 44.4 | 31.9 | 100.6 | 100 | 74.1 | 18.2 | 80.1 |
| K1 c0.5 L2 | 0.508 | 2.09 | 0.647 | 1.293 | 0.265 | 1.59 | 1.81 | 29.7 | 21.4 | 97.0 | 102 | 74.1 | 17.4 | 80.0 |
| K1 c0.5 L4 | 0.286 | 2.95 | 0.854 | 1.692 | 0.334 | 1.39 | 0.84 | 13.6 | 9.8 | 103.1 | 106 | 74.5 | 17.2 | 80.5 |
| K1 c1 L1 | 0.557 | 1.81 | 0.609 | 1.232 | 0.253 | 1.64 | 2.03 | 33.4 | 24.0 | 97.3 | 101 | 74.1 | 17.5 | 80.0 |
| K1 c1 L2 | 0.336 | 2.56 | 0.800 | 1.596 | 0.318 | 1.45 | 1.05 | 17.1 | 12.3 | 101.0 | 104 | 74.3 | 17.1 | 80.2 |
| K1 c1 L4 | 0.160 | 3.60 | 0.996 | 1.953 | 0.378 | 1.30 | 0.31 | 5.1 | 3.7 | 111.7 | 110 | 74.9 | 17.6 | 81.4 |
| K1 c2 L1 | 0.410 | 2.09 | 0.727 | 1.455 | 0.295 | 1.52 | 1.39 | 22.6 | 16.3 | 98.8 | 103 | 74.1 | 17.2 | 80.0 |
| K1 c2 L2 | 0.212 | 2.95 | 0.928 | 1.858 | 0.363 | 1.36 | 0.51 | 8.4 | 6.1 | 108.4 | 107 | 74.5 | 17.2 | 80.6 |
| K1 c2 L4 | 0.107 | 4.15 | 1.072 | 2.057 | 0.395 | 1.24 | 0.12 | 2.1 | 1.7 | 116.3 | 114 | 75.4 | 18.3 | 82.4 |
| K1 c4 L1 | 0.291 | 2.29 | 0.838 | 1.691 | 0.337 | 1.45 | 0.89 | 14.4 | 10.4 | 104.2 | 104 | 74.2 | 17.0 | 80.1 |
| K1 c4 L2 | 0.149 | 3.23 | 1.001 | 1.999 | 0.386 | 1.33 | 0.25 | 4.2 | 3.1 | 113.7 | 109 | 74.6 | 17.4 | 81.0 |
| K1 c4 L4 | 0.089 | 4.54 | 1.106 | 2.083 | 0.399 | 1.21 | 0.08 | 1.6 | 1.3 | 117.6 | 116 | 75.9 | 18.8 | 83.2 |
| K1 c8 L1 | 0.213 | 2.42 | 0.917 | 1.864 | 0.367 | 1.42 | 0.58 | 9.6 | 6.9 | 110.0 | 105 | 74.2 | 17.0 | 80.2 |
| K1 c8 L2 | 0.125 | 3.40 | 1.033 | 2.052 | 0.395 | 1.31 | 0.15 | 2.8 | 2.2 | 115.9 | 110 | 74.8 | 17.6 | 81.2 |
| K1 c8 L4 | 0.083 | 4.78 | 1.122 | 2.087 | 0.400 | 1.19 | 0.08 | 1.5 | 1.2 | 117.9 | 118 | 76.2 | 19.2 | 83.8 |
| K1 c32 L1 | 0.141 | 2.53 | 0.992 | 2.029 | 0.396 | 1.40 | 0.39 | 6.8 | 5.0 | 117.0 | 107 | 74.3 | 17.1 | 80.3 |
| K1 c32 L2 | 0.115 | 3.55 | 1.051 | 2.073 | 0.398 | 1.29 | 0.12 | 2.2 | 1.8 | 116.8 | 111 | 74.9 | 17.8 | 81.5 |
| K1 c32 L4 | 0.079 | 4.99 | 1.133 | 2.086 | 0.400 | 1.17 | 0.08 | 1.4 | 1.1 | 118.0 | 119 | 76.5 | 19.5 | 84.3 |
| K2 c0.5 L1 | 0.757 | 0.85 | 0.493 | 1.117 | 0.224 | 1.88 | 2.69 | 45.1 | 32.2 | 100.3 | 99 | 74.1 | 18.8 | 80.3 |
| K2 c0.5 L2 | 0.586 | 1.20 | 0.551 | 1.268 | 0.259 | 1.71 | 1.82 | 30.5 | 21.8 | **95.9** | 100 | 74.1 | 18.1 | 80.1 |
| K2 c0.5 L4 | 0.378 | 1.70 | 0.700 | 1.656 | 0.329 | 1.55 | 0.85 | 14.2 | 10.2 | 101.4 | 101 | 74.1 | 17.4 | 79.9 |
| K2 c1 L1 | 0.635 | 1.04 | 0.526 | 1.206 | 0.247 | 1.76 | 2.04 | 34.3 | 24.5 | 96.3 | 100 | 74.1 | 18.4 | 80.2 |
| K2 c1 L2 | 0.433 | 1.47 | 0.653 | 1.558 | 0.312 | 1.60 | 1.06 | 17.8 | 12.7 | 99.1 | 101 | 74.1 | 17.6 | 80.0 |
| K2 c1 L4 | 0.250 | 2.08 | 0.822 | 1.930 | 0.376 | 1.47 | 0.32 | 5.6 | 4.1 | 110.4 | 103 | 74.1 | 17.0 | 80.0 |
| K2 c2 L1 | 0.513 | 1.20 | 0.593 | 1.417 | 0.286 | 1.66 | 1.40 | 23.5 | 16.8 | 96.9 | 100 | 74.1 | 18.0 | 80.1 |
| K2 c2 L2 | 0.317 | 1.70 | 0.751 | 1.820 | 0.358 | 1.53 | 0.53 | 9.0 | 6.4 | 106.4 | 101 | 74.1 | 17.3 | 79.9 |
| K2 c2 L4 | 0.185 | 2.39 | 0.897 | 2.048 | 0.395 | 1.42 | 0.15 | 2.9 | 2.2 | 115.4 | 104 | 74.2 | 17.0 | 80.1 |
| K2 c4 L1 | 0.416 | 1.32 | 0.663 | 1.639 | 0.326 | 1.60 | 0.89 | 15.1 | 10.7 | 101.2 | 100 | 74.1 | 17.7 | 80.0 |
| K2 c4 L2 | 0.249 | 1.86 | 0.815 | 1.969 | 0.384 | 1.49 | 0.27 | 4.9 | 3.5 | 112.2 | 102 | 74.1 | 17.1 | 79.9 |
| K2 c4 L4 | 0.159 | 2.62 | 0.935 | 2.077 | 0.400 | 1.39 | 0.12 | 2.4 | 1.8 | 116.9 | 105 | 74.3 | 17.0 | 80.2 |
| K2 c8 L1 | 0.351 | 1.39 | 0.716 | 1.806 | 0.356 | 1.57 | 0.57 | 9.9 | 7.0 | 106.3 | 101 | 74.1 | 17.6 | 80.0 |
| K2 c8 L2 | 0.218 | 1.96 | 0.848 | 2.033 | 0.394 | 1.47 | 0.19 | 3.7 | 2.7 | 115.0 | 103 | 74.1 | 17.1 | 79.9 |
| K2 c8 L4 | 0.148 | 2.76 | 0.955 | 2.084 | 0.401 | 1.38 | 0.11 | 2.2 | 1.7 | 117.2 | 106 | 74.4 | 17.0 | 80.4 |
| K2 c32 L1 | 0.292 | 1.45 | 0.767 | 1.967 | 0.384 | 1.55 | 0.33 | 6.0 | 4.3 | 112.6 | 101 | 74.1 | 17.5 | 79.9 |
| K2 c32 L2 | 0.198 | 2.05 | 0.871 | 2.063 | 0.399 | 1.46 | 0.17 | 3.2 | 2.4 | 116.4 | 103 | 74.1 | 17.0 | 79.9 |
| K2 c32 L4 | 0.140 | 2.88 | 0.971 | 2.087 | 0.401 | 1.36 | 0.10 | 2.1 | 1.6 | 117.4 | 107 | 74.4 | 17.1 | 80.5 |
| K3 a0.5 | 0.557 | 1.81 | 0.609 | 1.232 | 0.253 | 1.64 | 2.03 | 33.4 | 24.0 | 97.3 | 101 | 74.1 | 17.5 | 80.0 |
| K3 a0.25 | 0.336 | 2.22 | 0.795 | 1.598 | 0.320 | 1.47 | 1.07 | 17.4 | 12.5 | 101.7 | 104 | 74.2 | 17.1 | 80.0 |
| K3 a0.11 | 0.212 | 2.42 | 0.918 | 1.866 | 0.367 | 1.42 | 0.58 | 9.5 | 6.9 | 110.1 | 105 | 74.2 | 17.0 | 80.2 |
| K3 a0 | 0.114 | 2.57 | 1.020 | 2.092 | 0.407 | 1.39 | 0.37 | 6.6 | 4.9 | 120.1 | 107 | 74.3 | 17.1 | 80.3 |

Identity reproduces the reference frozen number exactly (jitter 126.4 vs the 126 in `docs/`).
`K3 a0.5` is numerically identical to `K1 c1 L1` (the same matrix, by construction) — a
self-consistency check on the kernel builder.

### 2.2 The oracle line did NOT reproduce at ~14–19 with raw crops

Best jitter over all 44 kernels: **95.9** (K2 c0.5 L2, centre mass 0.586, width 1.20 frames);
the Gaussian-in-seconds arms give 113.8 / 116.8 / 117.5 at σ = 0.04 / 0.08 / 0.12 s, i.e. token
smoothing alone buys 126.4 → ~114–118, not → 14.

The reason is visible in the table: the token kernels crush the ROTATION and ARTICULATION noise
(rot_d3 3.98 → 0.08 = 50×, kp_d3 67.1 → 1.4 = 48×, kpb_d3 47.8 → 1.1) but make the DEPTH noise
**worse** (dep_d1 0.544 → 1.13 %/frame, dep_d3 1.37 → 2.09). The frozen camera head's scale `s`
tracks the crop scale `b` almost exactly (the crop track's own RMS d1 log b is 1.382 %/frame here,
2.5× the raw pelvis depth's 0.544), so the pelvis depth ∝ 1/(s·b) cancels most of the crop jitter;
smoothing the token smooths `s` but not `b` and breaks the cancellation. Depth jitter dominates
the lifted jitter, so the net gain is small.

### 2.3 Same kernels with the crop track Gaussian-smoothed (σ = 1 s)

Run with `--box-gauss 1.0` (image path, embedding cache off — the cache is keyed to the raw crop),
families K0 + K3.

Crop track after smoothing: RMS d1 log bbox_scale = **0.228 %/frame** (raw: 1.382).

| kernel | ctr | width | dep_d1 | dep_d3 | bea_d3 | rot_d1 | rot_d3 | kp_d3 | kpb_d3 | jitter | dep_err | acc LF | acc HF | acc tot |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K0 identity | 1.000 | 0.00 | 0.528 | 1.297 | 0.246 | 2.13 | 3.81 | 72.0 | 47.7 | 118.2 | 97 | 74.0 | 19.8 | 80.6 |
| K0 gauss 0.04 s | 0.360 | 1.13 | 0.308 | 0.244 | 0.039 | 1.61 | 0.47 | 8.3 | 6.1 | 20.1 | 97 | 74.0 | 17.9 | 80.0 |
| K0 gauss 0.08 s | 0.182 | 2.25 | 0.258 | 0.199 | 0.031 | 1.43 | 0.14 | 2.8 | 2.1 | **14.2** | 96 | 74.2 | 17.0 | 80.1 |
| K0 gauss 0.12 s | 0.123 | 3.36 | 0.236 | 0.198 | 0.030 | 1.31 | 0.08 | 1.7 | 1.3 | **13.6** | 95 | 74.8 | 17.5 | 81.2 |
| K3 a0.5 | 0.557 | 1.81 | 0.344 | 0.682 | 0.128 | 1.64 | 1.94 | 35.7 | 24.1 | 61.0 | 96 | 74.1 | 17.5 | 80.0 |
| K3 a0.25 | 0.336 | 2.22 | 0.279 | 0.400 | 0.074 | 1.47 | 1.03 | 18.4 | 12.6 | 34.7 | 96 | 74.2 | 17.1 | 80.0 |
| K3 a0.11 | 0.212 | 2.42 | 0.259 | 0.277 | 0.049 | 1.42 | 0.57 | 9.9 | 6.9 | 23.0 | 96 | 74.3 | 17.0 | 80.2 |
| K3 a0 (pure U9) | 0.114 | 2.57 | 0.254 | 0.234 | 0.040 | 1.39 | 0.37 | 6.7 | 4.8 | 19.0 | 95 | 74.3 | 17.2 | 80.3 |

With the crop track smoothed the oracle line reproduces: identity 118.2 -> **14.2** at a sigma =
0.08 s token Gaussian (GT 6.35), and the depth first difference now FALLS under smoothing
(0.528 -> 0.258 %/frame, GT ~0.275) instead of rising. The single-layer convex replacement kernels
land at 61.0 / 34.7 / 23.0 / 19.0 jitter for self weight 0.557 / 0.336 / 0.212 / 0.114 — i.e. a
NON-residual +-4-frame box with self weight <= 0.11 reaches 19-23, within a factor ~1.5 of the
sigma = 0.08 s Gaussian, from ONE layer.

Accuracy cost of all of this is ~0 on this metric: acc_tot 80.6 -> 80.0-81.2 mm, entirely because
acc_HF falls 19.8 -> 17.0-17.9 while acc_LF is flat at 74.0-74.8 until the widest kernels
(sigma 0.12 s: LF 74.8). `dep_err` also falls slightly (97 -> 95-96 mm). The same pattern holds in
the raw-crop table (2.1), where acc_HF 19.9 -> 17.0 and acc_LF stays 74.1 up to width ~3 frames and
only starts climbing past width ~4 (K1 c32 L4: LF 76.5, tot 84.3).


---

## What I measured / caveats

1. **No training, no repo edits.** Everything ran from the dev worktree on GPU 5
   (`CUDA_VISIBLE_DEVICES=5`); the pre-existing foreign process on that card was left alone. All
   new scripts and all outputs are in this scratchpad directory. `scripts/diag_temporal.py` was run
   unmodified with `--out` pointed here so it wrote nothing into `output/`.
2. **`diag_temporal.py` per-offset profile normalisation.** The `temporal_profile` rows sum to
   1.02 (tvel_ray) / 1.32–1.36 (tb_projzero) instead of 1, because each offset is averaged over
   only the query rows that have that offset available and edge rows carry more mass per key. Use
   the per-row `self` / `same_frame` / `eff_frames` columns for exact statements; the offset table
   is shape information, biased upward at large |offset|.
3. **`branch_gain.py` is a new script written for this task.** It re-runs the block by hand from a
   forward-pre-hook capture of the module's input and reproduces `_RopeBlock.forward` exactly
   (LN → +slot_embed → qkv → RoPE → SDPA with the same mask → proj → γ; then FFN). It does NOT
   re-verify itself against the module's own output — the diag script's SDPA check
   (2e-7 / 8e-6) covers the same recompute path. TF32 matmul is disabled in it.
4. **`d3_ratio` is a token-space quantity, not an output quantity.** It divides the pose token's
   3rd temporal difference by that token's own RMS, per layer. A layer that changes the token's
   low-frequency content changes the denominator too, so `d3_ratio > 1` (tb_projzero) does not by
   itself mean the readout gets noisier — the output ablation table is the output-side statement.
5. **`gain_attn` for tb_projzero mixes 7 slots.** The all-slot number (0.18–0.42) is dominated by
   what the block writes into the six contact tokens; the pose-slot column (0.062–0.096) is the
   one comparable with tvel_ray (0.032–0.034).
6. **The two checkpoints are not comparable head-to-head.** tvel_ray is epoch 8 of an
   interrupted run, pose-only, contacts and hands off, camera smoothing 0.25 s, and trained with
   velocity matching; tb_projzero is epoch 27 (best) with contacts + hands and the R1 loss set.
   Only the within-run ablations are apples-to-apples.
7. **Accuracy GT choice.** `mhr_sup_1` MHR70 keypoints (`batch["kp3d_world"]`), not the kindyn
   SMPL-X 22, and not a name mapping: the mapping the brief mentions is no longer in
   `model/loss/keypoint.py` (that loss now supervises MHR-natively against `kp3d_world`). Using
   `mhr_sup_1` means GT and prediction come from the same rig and the same regressor. `mhr_sup_1`
   is itself derived from the kindyn trajectory, so it inherits kindyn's own errors — the ~74 mm LF
   floor is that plus the frozen model's pose bias, and it is NOT a clean accuracy oracle.
   `dep_err` still uses the kindyn SMPL-X pelvis (`smplx_joints_world[:, 0]`), as the original probe did.
8. **Only 16 clips / 1918 frames.** The eval protocol emits one ≤120-frame clip per test scene, so
   `jitter` is a mean over ~1870 per-frame values from 16 sequences and the between-scene spread is
   large (a single-clip smoke run of the same code gave identity jitter 120.7 and σ = 0.04 s
   jitter 71.4 — very different from the 16-clip means 126.4 / 113.8). Treat differences of a few
   jitter units as noise.
9. **Kernels act on row INDEX inside the clip** (not seconds) for K1/K2/K3, matching the trained
   ±4 window; the K0 Gaussians act on real elapsed seconds, matching the original probe. Both are
   truncated at clip edges and at invalid frames, then row-renormalised, so the centre mass and
   width columns are the realised (edge-corrected) values, not the nominal ones.
10. **What I did not measure**: any effect on the trained SMPL-X head (the kernel probe reads the
    FROZEN MHR/camera heads only); contact metrics under the kernels (the frozen contact head is
    untrained in this build); anything on non-static scenes; and the `tvel_cliff` arm.
11. **The `--box-gauss 1.0` arm is not the deployed pipeline.** It Gaussian-smooths the corpus
    bbox track (centre and size, sigma = 1 s, per person, valid frames) inside the loader before
    anything reads it, which forces the image path (the bf16 embedding cache is keyed to the raw
    crop) and therefore a full backbone forward. It is a diagnostic to isolate the crop track's
    contribution, not a proposal; the raw-crop table (2.1) is the one that matches how the trained
    runs are evaluated. Note the identity row differs slightly between the two tables
    (jitter 126.4 raw vs 118.2 smoothed-crop) because the crops themselves changed.
12. **`K3 a0.5` and `K1 c1 L1` are the same matrix by construction** and come out numerically
    identical in the table — an internal consistency check on the kernel builder.
