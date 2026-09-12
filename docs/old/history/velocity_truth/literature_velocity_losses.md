# Velocity / acceleration supervision in video 3D human pose & mesh recovery — a literature survey

Compiled 2026-09-05. Literature only: no code was run, no repo touched.

**Provenance.** I read the following **first-hand** (arXiv PDF text via `pdftotext`, or the official
repo source fetched directly): WHAM, GVHMR (+ its repo config, pipeline, endecoder, geo utils),
SmoothNet (incl. supplementary), DeciWatch (+ repo), TCMR, MEVA, HuMoR (incl. supplementary), RoHM,
GLAMR, KVAE, and SLAHMR's `confs/optim.yaml`. The papers in **Part 2** were read by a parallel
search agent working from the same brief; their verbatim quotes and file:line references are
reproduced as returned and are marked as second-hand. **Part 3** (theory) is likewise from a
parallel agent. Anything I could not confirm is called out. `[INFERENCE]` marks reasoning that is
not in the cited source.

---

## Executive summary — six things the literature actually says

1. **Almost nobody matches a finite-differenced pose output to a GT velocity.** Of ~19 methods
   surveyed, only **MotionBERT**, **GLoT** (masked frames only), **RoHM** (a denoiser), and **D&D**
   (in code, not in the paper) do it — and all four operate on **root-relative** joints or an
   **integrated** trajectory. Nobody applies a velocity loss to an absolute metric root
   translation / camera depth on a jointly-trained per-frame head.

2. **The dominant pattern is a separate velocity head whose INTEGRAL is supervised.** WHAM, GVHMR
   and GLAMR all regress a root-local velocity as a *channel* and then supervise its cumulative sum:
   WHAM's multi-scale windows `[1, 3, 9, 27]`, GVHMR's `rollout_local_transl_vel` + L1 on world
   translation, GLAMR's `EgoToGlobal` + L2 on the accumulated τ, γ. The derivative is the
   *representation*; the *integral* is the objective. That keeps the DC/low-frequency content in the
   loss instead of high-passing it away.

3. **The accel↔MPJPE trade-off is named, quantified and repeatedly reported.** SmoothNet calls it
   the *"spatio-temporal optimization bottleneck"* and shows adding an acceleration loss end-to-end
   costs +1.35 to +1.5 mm MPJPE. PMCE: *"when MAED reduces PA-MPJPE by 11 mm, it increases ACCEL by
   11.1 mm/s²"*. VIBE: *"There is a trade-off between accuracy and smoothness."* TCMR, MEVA and
   PhysDiff all report it independently.

4. **The published fix is architecture and two-stage training, not loss weighting.** SmoothNet's
   own ablation is decisive: the *identical* network trained **end-to-end** costs +2.5 to +3.5 mm
   MPJPE, but trained **two-stage on a frozen backbone** improves MPJPE *and* Accel simultaneously
   (Accel 23.2 → 6.05, MPJPE 83.0 → 81.4 on 3DPW). PhysPT gets a 6× accel reduction for +0.5 mm from
   a *pure position-space* loss against clean mocap in a fully decoupled second stage. TRAM ships an
   acceleration loss it never enables and still beats everyone on both axes.

5. **The residual-block finding is empirical and published.** TCMR's headline ablation: removing the
   residual skip from the per-frame ("static") feature into the temporal feature took Accel from
   **29.2 → 8.7** *and* improved PA-MPJPE 55.6 → 54.2. Verbatim: *"the identity mapping of the
   current static feature inside the residual connection hinders a model from learning meaningful
   temporal features."* SmoothNet's Table 4 adds that a **self-attention** temporal block (Accel 6.15)
   is beaten by a learned signed FIR filter over the window (4.15) and is barely better than a plain
   Gaussian (4.95). GLAMR: swapping a local LSTM for a Transformer on a derivative-valued output
   raised Accel 5.8 → 121.9.

6. **On the frame question:** every method that had to choose one chose **gravity-preserving,
   heading-aligned, root-translation-removed** — HuMoR's canonical frame, GLAMR's heading
   coordinates, GVHMR's Gravity-View, WHAM's egocentric velocity with world root orientation, RoHM's
   "relative to the current-frame pelvis **projected on the ground**". None uses the full body frame
   (which would also remove roll and pitch), and none compares raw world velocities without first
   factoring out heading.


## The table

"Shares params with the per-frame head?" — *Head* = the module that emits the per-frame pose that
the per-frame losses supervise. *Trunk* = a shared encoder upstream of both.

| Method | Where the velocity/accel loss attaches | Frame | Shares params with per-frame head? | Reported trade-off |
|---|---|---|---|---|
| **WHAM** (CVPR'24, 2312.07531) | Separate **trajectory decoder** `D_T` output `v` (a regressed channel), integrated as `τ=Σ Γ^i v^i`. Loss = **multi-scale cumulative displacement** over windows 1/3/9/27. Plus contact-gated **zero**-velocity foot-slide term on the pose path, and a camera angular-velocity consistency term (skipped 5 epochs). | `v` root-local (egocentric); `Γ` world; foot-slide world; camera terms camera-frame | **No** — separate decoder; shares the motion encoder + feature integrator trunk | None reported |
| **GVHMR** (SIGGRAPH Asia'24, 2409.06662) | MLP head channel `local_transl_vel`, standardized; supervised by (a) MSE on the channel (weight 1) and (b) **L1 on its cumulative roll-out** to world translation (`transl_w`=1), rolled out with **GT** orientation + GT origin | root-local (SMPL-coord); GV frame is gravity+view aligned | **No** — multitask MLP heads off a shared 12-layer transformer trunk | Not stated; their own Tab. 3 shows best-Jitter ablations ≠ best-accuracy |
| **SmoothNet** (ECCV'22, 2112.13715) | **L1 acceleration of the same output** as the position loss, weight 1:1. Network is temporal-only, trained on the **frozen** estimator's dumped outputs | whatever the estimator outputs (2D / 3D / 6D rot); no re-framing | **No parameters shared at all** — two-stage, backbone frozen | **Yes, explicitly.** "adding an acceleration loss … can benefit Accels but harm MPJPEs"; names it the *spatio-temporal optimization bottleneck*; two-stage fixes it |
| **DeciWatch** (ECCV'22, 2203.08713) | **No velocity/accel loss** — L1 positions at two pipeline points (λ=5 on RecoverNet) | pose coordinates | **No** — frozen estimator, two-stage | not discussed |
| **TCMR** (CVPR'21, 2011.08627) | **No velocity/accel loss** — per-frame L2 on three heads, all vs the *current* frame GT | camera / root-relative | jointly trained | **Yes.** Quotes the HMMR-vs-VIBE accuracy↔smoothness trade-off; average filtering "can decrease per-frame accuracy by smoothing out details". Fix is **removing the residual** static→temporal skip (Accel 29.2→8.7, PA-MPJPE 55.6→54.2) |
| **MEVA** (ACCV'20, 2008.03789) | **No velocity/accel/smoothness loss** — L_3D+L_2D+L_SMPL only | camera | jointly trained | **Yes, as motivation:** "using prior knowledge only in the loss function, it is hard to find the balance between smoothness and accuracy" |
| **HuMoR** (ICCV'21, 2105.04668) | L2 on **velocity channels of the state** `x=[r,ṙ,Φ,Φ̇,Θ,J,J̇]`, conditioned on the **GT previous state**; + contact-gated **zero**-velocity (w=0.01) | **heading-aligned, gravity-preserving canonical frame** (up-axis rotation + xy translation removed) | CVAE decoder is `x̂_t = x_{t−1} + Δ_θ(z_t,x_{t−1})` — one model | Not a pose estimator; explicit **stop-gradient** through `x̂_{t−1}` in rollout + scheduled sampling |
| **RoHM** (CVPR'24, 2401.08570) | `L_vel = ‖J̇3D(GT traj, GT pose) − J̇3D(GT traj, P̂0)‖²` — **finite difference of the same output**, λ_vel=1000 vs λ_3D=100 | local frame: joints relative to the current-frame pelvis **projected on the ground** | diffusion denoiser; per-frame estimates are **conditioning inputs**, targets are clean AMASS | Removed velocity channels from the *trajectory* representation "to avoid global drifting caused by inaccurate velocities"; foot-skate term staged 0 → 0.1 |
| **GLAMR** (CVPR'22, 2112.01524) | CVAE trajectory predictor emits **egocentric per-frame deltas**; loss is on the **accumulated** global τ, γ (Eq. 8) | heading coordinates (egocentric); loss in world | **No** — separate predictor fed by the (already estimated) local body motion | Tab. 4: replacing the local **LSTM** with a **Transformer** raised Accel 5.8 → 121.9 and G-MPJPE by ~190 mm |
| **TRAM / VIMO** (ECCV'24, 2403.17346) | **None used.** An `acceleration_loss` exists in the repo but **no shipped config enables it** | (would be root-relative) | ViT backbone frozen; temporal transformers + head trained by per-frame losses | None — wins both axes without any temporal loss (Accel 4.9 vs HMR2.0's 19.9) |
| **SLAHMR** (CVPR'23, 2302.12827) | `E_smooth = Σ‖J^t − J^{t+1}‖²` — a **zero-target** smoothness prior in a test-time optimizer; **turned off (0.0) in the final stage** where the learned HuMoR prior turns on | world | N/A (optimization, no training) | not quantified |
| **PhysPT** (CVPR'24, 2404.04430) | **No velocity-matching loss.** Velocity appears only contact-gated toward zero and inside the Euler-Lagrange residual | world | **Fully decoupled second stage**, trained on AMASS only, bolted on "without the need of model fine-tuning" | **Yes, tabulated.** Recon loss alone: Accel 15.4→2.5 for +0.5 mm MPJPE. Adding the contact (zero-velocity) term "further reduces the errors but **sacrifices the reconstruction accuracy**" |
| **VIBE** (CVPR'20, 1912.05656) | **None** — GRU + motion discriminator only (`batch_smooth_pose_loss` exists but is never called) | — | jointly trained | **Yes.** "There is a trade-off between accuracy and smoothness"; Temporal-HMR "over-smooths the pose predictions while sacrificing accuracy" |
| **MotionBERT** (ICCV'23, 2210.06551) | `L_O = Σ‖Ô_t − O_t‖₂`, one-step difference of the **same output** as the position loss, `λ_O = 20` (position implicit 1) | **root-relative** (`predicted_3d_pos[:,:,0,:]=0`, GT re-centered); mesh path re-centers too | **Yes, fully shared, no stop-gradient** | not discussed |
| **GLoT** (CVPR'23, 2303.14747) | 1-step velocity on the same 2D/3D outputs, **but masked**: `m_t = 1` only on masked frames (MAE-style inpainting). vel_2d=10, vel_3d=100 | root-relative 3D + weak-perspective 2D | yes, shared | Claims to escape it; attributes the trade-off to coupled architecture, fixed by a global/local split |
| **PMCE** (ICCV'23, 2308.10305) | **None** | — | pose stream pretrained separately, then joint | **Yes, the most explicit:** "there exists a trade-off between per-frame accuracy and motion smoothness"; "when MAED reduces PA-MPJPE by 11 mm, it increases ACCEL by 11.1 mm/s²" |
| **4DHumans / HMR2.0** (ICCV'23, 2305.20091) | **None — pure single-frame.** `losses.py` is 92 lines, no time axis | root-relative (pelvis id 39) | single-frame; PHALP′ temporal transformer never backprops into HMR2.0 | n/a (Accel 18-20, the jitteriest strong model) |
| **D&D** (ECCV'22, 2209.08790) | Paper: none. **Code**: vel+accel MSE on root-relative joints at `KP_3D_ACCEL_W=300` (= `KP_3D_W`), **and** on the global translation an **absolute L1 anchor + an accel L1 at the identical weight 100**, both after subtracting frame 0 | non-inertial **camera** frame; joint terms root-relative | shared, but the pose is **integrated from predicted accelerations** by semi-implicit Euler | none reported |
| **PhysDiff** (ICCV'23, 2212.02500) | Velocity only as an **RL reward** for a frozen imitation policy applied at *sampling* time | world (simulator) | no gradient path to the diffusion model | **Yes, non-monotone:** "the motion quality increases before a certain number of steps and decreases after that" |

---

# Part 1 — papers I read first-hand

PDF text or repo source, fetched directly. Anything I could not verify is called out.
`[INFERENCE]` marks reasoning not present in the cited source.

---

## 1. WHAM — Shin, Kim, Halilaj, Black. CVPR 2024. arXiv:2312.07531

**Loss list, verbatim (Sup. Mat. §C, "Losses", p. 13):**

> L_total = [motion reconstruction] L_smpl + L_verts + L_3D + L_2D + L_casc +
>           [trajectory reconstruction] L_root + L_contact + L_ω + L_cam + L_fs .

> L_root = Σ_t λ_Γ ( ||Γ_0^(t) − Γ*^(t)||² + ||Γ^(t) − Γ*^(t)||² ) + Σ_t λ_v ( ||v_0^(t) − v*^(t)||² + ||v^(t) − v*^(t)||² )
> L_contact = Σ_t λ_p ||p^(t) − p*^(t)||²
> L_ω = Σ_t λ_ω ||ω^(t) − ω*^(t)||²      ("ω is the reconstructed camera angular velocities from R")
> L_cam = Σ_t λ_cam ||R^(t) − R*^(t)||²   with R^(t) = Γ_c^(t)⊤ Γ^(t)
> L_fs = Σ_t λ_fs ||p*^(t) ⊙ v_f^(t)||²   ("v_f is the world-coordinate foot velocity, ⊙ denotes the
>                                          masking operation of foot contact based on the ground
>                                          truth contact probability p*^(t)")

**Where the velocity loss attaches.** NOT to the per-frame pose output. WHAM has two
decoders off one shared motion feature φ_m (§3.2):
* Motion decoder `D_M(φ̂_m^(0..t))` → (θ, β, weak-persp camera c, contact p) — the per-frame pose path.
* **Global Trajectory Decoder `D_T`**: "(Γ_0^(t), v_0^(t)) = D_T(φ_m^(0), ω^(0), …, φ_m^(t), ω^(t))"
  — a *separate head* emitting root orientation and **root velocity as a direct regression output**.
  Global translation is obtained by integration: "τ^(t) = Σ_{i=0}^{t−1} Γ^(i) v^(i)" (roll-out).

So `v` is a predicted channel that is integrated, never a finite difference of the pose output.
Code confirms: `pred_vel_root = pred['vel_root']` (lib/core/loss.py:80).

**The velocity loss is NOT a one-step difference — it is a multi-scale cumulative displacement
loss.** `lib/core/loss.py:261-271`, verbatim:

```python
loss_v = 0
T = gt_vel_root.shape[0]
ws_list = [1, 3, 9, 27]
for ws in ws_list:
    tmp_v = 0
    for m in range(T//ws):
        cumulative_v = torch.sum(pred_vel_root[:, m:(m+1)*ws] - gt_vel_root[:, m:(m+1)*ws], dim=1)
        tmp_v += torch.norm(cumulative_v, dim=-1)
    loss_v += tmp_v
loss_v = loss_v[mask_v].mean()
```
Summing the velocity *error* over a window of `ws` frames = the **displacement error over that
window**. So WHAM supervises displacement at 1, 3, 9 and 27 frames — i.e. the loss keeps the DC /
low-frequency content, it is not a pure high-pass differencing penalty. [INFERENCE on the spectral
reading; the code itself is verbatim.]

**Frame.** `v` is *egocentric* (root/body-local): the roll-out multiplies by Γ^(i) to reach world.
The refiner adjusts it with world-frame foot velocity mapped back: `ṽ^(t) = v_0^(t) − (Γ_0^(t))^{-1} v̄_f^(t)`.
Γ (root orientation) is in **world** coordinates; L_cam / L_ω are in camera space.

**Parameter sharing.** The velocity loss trains `D_T` and the refiner `R_T`, plus — through the
shared φ_m — the motion encoder `E_M` and feature integrator `F_I`. It does **not** pass through
the per-frame pose head `D_M`. So: shared *trunk*, separate *head*.

**Weights (repo `configs/yamls/`, verbatim):**

| entry | stage1 (AMASS pretrain) | stage2 (video finetune) |
|---|---|---|
| SHAPE_LOSS_WEIGHT | 0.004 | 0.0 |
| JOINT3D_LOSS_WEIGHT | 0.4 | 6.0 |
| JOINT2D_LOSS_WEIGHT | 0.1 | 3.0 |
| POSE_LOSS_WEIGHT | 8.0 | 1.0 |
| CASCADED_LOSS_WEIGHT | 0.0 | 0.05 |
| SLIDING_LOSS_WEIGHT | 0.5 | 0.5 |
| CAMERA_LOSS_WEIGHT | 0.04 | 0.01 |
| **ROOT_VEL_LOSS_WEIGHT** | **0.001** | **0.001** |
| LOSS_WEIGHT (global) | 50.0 | 60.0 |
| CAMERA_LOSS_SKIP_EPOCH | **5** | 0 |

Two things worth flagging: (i) the root-velocity weight is 0.001 against 6.0 for 3D joints —
three to four orders of magnitude smaller (units differ, so this is not a like-for-like ratio, but
the term is unambiguously a *small* one); (ii) `CAMERA_LOSS_SKIP_EPOCH: 5` — the loss that contains
the **finite-differenced angular velocity** (`camera_loss`, below) is switched OFF for the first 5
epochs of pretraining. A deliberate warm-up on the only differencing term.

**Gradient routing.** `sliding_loss` (loss.py:425-438) detaches the gate:
```python
contact_mask = (contact_prob > 0.5).detach().float()
foot_velocity = foot_position[:, 1:] - foot_position[:, :-1]
loss = (torch.norm(foot_velocity, dim=-1) * contact_mask[:, 1:]).mean()
```
This *is* a finite-difference loss applied to the pose/trajectory output — but it drives the
velocity toward **zero under contact**, it does not match a GT velocity. `camera_loss` (loss.py:398-421)
is the one true finite-difference-vs-GT term:
```python
pred_R = transforms.rotation_6d_to_matrix(pred_cam_r)
cam_angvel_from_R = transforms.matrix_to_rotation_6d(pred_R[:, :-1] @ pred_R[:, 1:].transpose(-1, -2))
cam_angvel_from_R = (cam_angvel_from_R - torch.tensor([[[1, 0, 0, 0, 1, 0]]]).to(cam_angvel)) * 30
loss_a = criterion(cam_angvel, cam_angvel_from_R)[mask].mean()
```
and note it compares the *differenced prediction* against the **input** camera angular velocity
(a consistency term between two predicted/observed quantities), at weight 0.01–0.04, skipped for
5 epochs.

**Reported accel/MPJPE trade-off:** none stated in the WHAM paper. Could not verify any.

---

## 2. GVHMR — Shen, Pi, Xie, Yang, Peng, Zhou et al. SIGGRAPH Asia 2024. arXiv:2409.06662

**Loss list, verbatim (§3.2, "Losses"):**

> "We use the following losses for training: Mean Squared Error (MSE) loss on predicted targets
> except for stationary probability, which uses Binary Cross-Entropy (BCE) loss. Additionally, we
> use L2 loss on 3D joints, 2D joints, vertices, translation in the camera frame, and translation
> in the world coordinate system. More details are provided in the supplementary material."

The predicted targets (§3.2, "Network outputs") are: weak-perspective camera `cw`, camera-frame
human orientation `Γ_c`, SMPL local pose `θ`, shape `β`, stationary label `p_j`, and the global
trajectory representation `Γ_GV` **and `v_root`**.

**Where the velocity loss attaches.** Same structure as WHAM: `v_root` is a **direct MLP output
channel**, and the world trajectory is its cumulative sum (Eq. 1): `τ_w = Σ_{i=0}^{t−1} Γ_w^i v_root^i`.
The repo makes the routing explicit (`hmr4d/model/gvhmr/pipeline/gvhmr_pipeline.py:290-300`, verbatim):

```python
if weights.transl_w > 0:
    # compute pred_transl_w by rollout
    gt_transl_w = inputs["smpl_params_w"]["transl"]
    local_transl_vel = decode_dict["local_transl_vel"]
    pred_transl_w = rollout_local_transl_vel(local_transl_vel, gt_global_orient_w, gt_transl_w[:, [0]])
    trans_w_loss = F.l1_loss(pred_transl_w, gt_transl_w, reduction="none")
    trans_w_loss = (trans_w_loss * mask[..., None]).mean()
    extra_loss += trans_w_loss * weights.transl_w
```
`rollout_local_transl_vel` (`hmr4d/utils/geo/hmr_global.py:151-171`) is a `torch.cumsum` of
`R(global_orient) @ local_transl_vel`. **The world-position loss is the integral of the predicted
velocity** — the low-pass counterpart of a differencing loss. And it is rolled out with the **GT**
global orientation and **GT** first translation (teacher forcing), so no gradient of the trajectory
term reaches the predicted orientation.

The `simple_loss` (`gvhmr_pipeline.py:113-122`) is a plain MSE on the whole predicted target vector
(which includes `local_transl_vel`) with **implicit weight 1** (`total_loss += simple_loss`).

**Frame.** `local_transl_vel` is in **SMPL/root-local** coordinates (docstring: "transl velocity is
in local coordinate (or, SMPL-coord)"); the GV coordinate system itself is gravity-aligned and
camera-view-aligned per frame (§3.1).

**Parameter sharing.** One shared 12-layer relative transformer + multitask MLP heads; velocity
and per-frame pose share the trunk, not the head. [INFERENCE from §3.2's "processed by multitask MLPs".]

**Weights (`hmr4d/configs/exp/gvhmr/mixed/mixed.yaml`, verbatim):**
```yaml
weights:
    cr_j3d: 500.
    transl_c: 1.
    cr_verts: 500.
    j2d: 1000.
    verts2d: 1000.
    transl_w: 1.
    static_conf_bce: 1.
```
plus `simple_loss` at weight 1. Note there is **no separately-weighted velocity term at all** —
velocity gets weight 1 through `simple_loss`, and its *integral* gets weight 1 through `transl_w`.

**GVHMR has no smoothness/jitter/acceleration loss.** Smoothness comes from (a) a 12-layer
transformer over L=120 frames with RoPE, and (b) post-processing (predicted stationary probabilities
→ translation update → CCD IK). They say so about competitors, §4.2, verbatim:
> "Compared to optimization-based algorithms like GLAMR and SLAHMR, our method also achieves better
> smoothness metrics. Although these methods incorporate a smoothness loss, they may struggle due to
> the high difficulty of the actions in the dataset."

**Accel/MPJPE divergence in their own ablation (Tab. 3, RICH):**

| variant | PA-MPJPE | MPJPE | W-MPJPE | Jitter |
|---|---|---|---|---|
| (1) w/o GV | 40.0 | 67.0 | 278.9 | **9.7** |
| (3) w/o Transformer | 43.3 | 73.9 | 138.9 | **7.6** |
| (7) w/o PostProcessing | 39.5 | 66.0 | 145.2 | 14.5 |
| Full Model | **39.5** | **66.0** | **126.3** | 12.8 |

The best-Jitter rows are *not* the best-accuracy rows: the full model is 5.2 units *worse* in Jitter
than "w/o Transformer" while being 8 mm better in MPJPE and 13 mm better in W-MPJPE. GVHMR does not
comment on this. [INFERENCE that this is a smoothness/accuracy divergence — they do not call it one.]

---

## 3. SmoothNet — Zeng, Yang, Ju, Li, Wang, Xu. ECCV 2022. arXiv:2112.13715

**The single most on-point paper for this question.**

**Architecture.** Temporal-**only**, operating directly on the pose *coordinates* of a frozen
estimator, never on images or features. Eq. (1), verbatim:
> Ŷ^{l+1}_{i,t} = σ( Σ_{t=1}^{T} w^l_t * Ŷ^l_{i,t} + b^l )

i.e. a **fully-connected map across the whole temporal window** (a learned FIR filter, weights
shared across joints/axes), LeakyReLU, with N residual-connected blocks; sliding window T with step
s. Motion-aware variant (§4.2, Fig. 6): three parallel branches taking the noisy positions Ŷ, the
computed velocity V̂ and acceleration Â (Eq. 2: `V̂_{i,t} = Ŷ_{i,t} − Ŷ_{i,t−1}`,
`Â_{i,t} = V̂_{i,t} − V̂_{i,t−1}`), concatenated and fused by a linear layer. **The velocity and
acceleration are INPUTS, not losses, in that branch.**
Size: **0.33 M params** (basic variant reported as 0.03 M in the supp. table). Window T = 32
(main experiments), step s = 1; 64 recommended as the accuracy/smoothness balance.

**Losses (§4.3), verbatim:**
> L_pose = 1/(T·C) Σ_t Σ_i |Ĝ_{i,t} − Y_{i,t}|      (3)
> L_acc  = 1/((T−2)·C) Σ_t Σ_i |Ĝ''_{i,t} − A_{i,t}|  (4)
> "where Ĝ''_{i,t} is the computed acceleration from predicted pose Ĝ_{i,t} and A_{i,t} is the
> ground-truth acceleration. **We simply add L_pose and L_acc as our final target.**"

So: L1 position + L1 **acceleration** (second finite difference of the same output), **weight 1:1**,
no velocity term. This is the same functional form as your velocity loss — and it does not collapse.

**Why it does not collapse — their own explanation.** Because it is a *two-stage* method: the
per-frame estimator is frozen and its outputs are dumped to disk; SmoothNet is trained separately on
those. §3.1, verbatim:
> "(ii). further adding an acceleration loss between consecutive frames or enhancing temporal
> modeling in the decoder design **can benefit Accels but harm MPJPEs (increase biased errors S)**,
> due to the optimization bottleneck between per-frame precision and smoothness."
> "With the above, a temporal-only pose smoothing solution is more promising for jitter mitigation."

They name the phenomenon the **"spatio-temporal optimization bottleneck"**, and decompose error
into "the jitter error J between adjacent frames and the biased error S between the ground truth and
smoothed poses" (§2.2).

**The decisive ablation (Supp. §2.1, Table 1 — VIBE on 3DPW).** `×` = acceleration loss added to the
end-to-end objective; `B` = SmoothNet attached and trained end-to-end with the backbone;
`w/ ours` = two-stage (frozen backbone, separately trained SmoothNet):

| Strategy | Accel ↓ | MPJPE ↓ | PA-MPJPE ↓ | MPJVE ↓ |
|---|---|---|---|---|
| In = 1 | 32.69 | 84.54 | 57.94 | 102.05 |
| In = 16 | 23.21 | 83.03 | 56.77 | 99.76 |
| **In = 16 ×** (acc loss, end-to-end) | **20.42** | **84.51** | 57.81 | 101.62 |
| **In = 16 w/ B** (SmoothNet end-to-end) | 21.65 | **86.56** | 59.93 | 105.08 |
| In = 1 w/ ours (two-stage) | 6.12 | 82.98 | 57.27 | 100.67 |
| **In = 16 w/ ours** (two-stage) | **6.05** | **81.42** | **56.21** | **98.83** |

and Supp. Table 2 — VPose on Human3.6M (`◦` = intermediate L1 supervision between backbone and SmoothNet):

| Strategy | Accel ↓ | MPJPE ↓ | Params |
|---|---|---|---|
| In = 27 | 5.07 | 50.13 | 8.61 M |
| **In = 27 w/ ×** | 4.12 | **51.48** | 8.61 M |
| In = 27 w/ B | 2.78 | 52.65 | 8.65 M |
| In = 27 w/ B ◦ | 5.46 | 51.06 | 8.65 M |
| In = 27 w/ B ◦ × | 2.69 | 50.94 | 8.65 M |
| In = 1 w/ ours | 1.03 | 52.72 | **0.03 M** |
| **In = 27 w/ ours** | **0.88** | **50.04** | **0.03 M** |

Reading: adding the acceleration loss **end-to-end** costs +1.5 mm (3DPW) / +1.35 mm (H36M) MPJPE
to buy −2.8 / −0.95 Accel. Bolting SmoothNet on end-to-end costs +3.5 mm / +2.5 mm. Training the
identical network **two-stage on a frozen backbone** improves *both* metrics simultaneously
(−17 Accel and −1.6 mm on 3DPW; −4.2 Accel and −0.1 mm on H36M). Their conclusion, verbatim:
> "Compared with one-stage strategies, two-stage solutions with a refinement network show their
> strengths in boosting both smoothness and precision."

Their proposed *reason* (Supp. §2.1), verbatim and hedged by them:
> "The reasons behind this **may lie in** that temporal and spatial information may generalize and
> overfit at different rates as two different modalities. MPJPEs are always larger than Accels,
> making the models pay more attention to optimizing spatial errors and hard to reduce Accels greatly."

**Headline numbers (Table 1, VIBE on AIST++):** VIBE Accel 31.64 / MPJPE 106.90 / PA 72.84 →
+SmoothNet 4.15 / 97.47 / 69.67. Best Gaussian-1D filter at comparable Accel: 4.47 / 105.71 / 71.49.
**Window-size ablation (Table 5, VIBE-AIST++):**

| W | VIBE | 2 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|
| Accel | 31.63 | 17.89 | 5.76 | 4.54 | 4.15 | 4.07 | 4.04 | 4.03 |
| MPJPE | 106.90 | 102.57 | 99.98 | 98.62 | 97.47 | 97.06 | 93.20 | 94.89 |
| PA-MPJPE | 72.84 | 71.48 | 70.51 | 69.85 | 69.67 | 69.89 | 70.57 | 71.52 |

Accel saturates by W≈8-16; PA-MPJPE has a **minimum at W = 32** and degrades beyond — a mild
over-smoothing signature in a *learned* filter. Also, their filter comparison shows classical
Gaussian filtering over-smooths past kernel 65 (Supp. §2.2: "The filters can relieve jitter errors
with the increase of kernel size but suffers from over-smoothness when the kernel size is larger
than 65, leading to worse position errors.").

---

## 4. DeciWatch — Zeng, Ju, Yang, Wang, Xu et al. ECCV 2022. arXiv:2203.08713

Two-stage again: a frozen per-frame estimator runs on 1/N of the frames; `DenoiseNet` cleans the
sparse estimates; `RecoverNet` interpolates the rest.

**Losses (§3.5), verbatim:**
> L = λ ( 1/T Σ_{t=1}^{T} |P̂^t − P^t| ) + 1/(T/N) Σ_{n=1}^{T/N} |P̂^{sampled(n)}_{clean} − P^{sampled(n)}| ,  (6)
> "where λ is a scalar to balance the losses between RecoverNet and DenoiseNet. We set λ = 5 by default."

**No velocity or acceleration loss at all** — pure L1 on positions, at two places in the pipeline.
Accel improvement is reported as 73–92 % over the estimators, purely from architecture.

**Architecture note relevant to residual-vs-non-residual.** `DenoiseNet` is a transformer *encoder
over the noisy pose coordinates* with a residual in coordinate space
(`lib/models/deciwatch.py:255`, verbatim: `reco = self.encoder_joints_embed(mem) + input`) — the
branch is an unconstrained linear map of the attended features added to the raw input, so it can
subtract. §3.3, verbatim on why they chose global attention:
> "Due to the temporal sparseness and noisy jitters, the key designs of DenoiseNet lie in two
> aspects: (i) A dynamic model for handling diverse possible pose noises; (ii) Global temporal
> receptive fields to capture useful Spatio-temporal information while suppressing distracting
> noises. Based on these two considerations, **local operations, like convolutional or recurrent
> networks, are not well suited.**"

Total DenoiseNet+RecoverNet: 0.60 M params; DenoiseNet M = 5 transformer blocks.

---

## 5. TCMR — Choi, Moon, Chang, Lee. CVPR 2021. arXiv:2011.08627

**Losses (§3.4), verbatim:**
> "For the training, we supervise all three outputs Θ_past, Θ_future, and Θ_int with current frame
> groundtruth. L2 loss between predicted and groundtruth SMPL parameters and 2D/3D joint
> coordinates are used, following VIBE."

**No velocity or acceleration loss.** Everything is per-frame L2. Accel drops 29.2 → 7.7 purely by
architecture.

**The residual-connection result — directly on point for "why residual blocks dilute".**
Table 1 (3DPW, all networks estimate only the middle frame):

| remove residual | PoseForecast | PA-MPJPE ↓ | Accel ↓ |
|---|---|---|---|
| ✗ | ✗ | 55.6 | 29.2 |
| ✗ | ✓ | 55.0 | 24.9 |
| ✓ | ✗ | 54.2 | 8.7 |
| ✓ (Ours) | ✓ | **53.9** | **7.7** |

Verbatim §5.2:
> "As shown in Table 1, removing the residual connection decreases the acceleration error
> significantly, which indicates a considerable improvement in temporal consistency and smoothness
> of 3D human motion. **This finding verifies that the identity mapping of the current static feature
> inside the residual connection hinders a model from learning meaningful temporal features.**
> Moreover, the increased temporal consistency of 3D motion improves the per-frame 3D pose accuracy."

The residual here is VIBE's skip from the per-frame ("static") feature to the temporal feature —
structurally the same object as a residual temporal block on a frozen per-frame token. Removing it
cut Accel 3.4× *and* improved PA-MPJPE.

Table 2 reinforces it: feeding the target frame's own static feature into the temporal encoder
raises Accel 33 % (10.3 vs 7.7). Verbatim:
> "including the target static feature hinders PoseForecast from learning useful temporal
> information for temporally consistent and smooth 3D human motion. **The encoded temporal feature
> is likely to be dominated by the target static feature and marginally leverage temporal
> information from other frames.**"

**Explicit accel/MPJPE trade-off statement (§2, related work), verbatim:**
> "HMMR and VIBE lowered the acceleration error compared with the single image-based methods.
> However, **they revealed a trade-off between per-frame accuracy and temporal consistency.** The
> HMMR outputs smoother 3D human motion but provides low per-frame 3D pose accuracy. Conversely,
> the VIBE shows high per-frame 3D pose accuracy; however, the output is temporally inconsistent
> in quantitative metrics and qualitative results compared with HMMR."

and on post-hoc filtering (§5.4), verbatim:
> "the results imply that the average filtering can decrease the per-frame 3D pose accuracy by
> smoothing out the details of 3D human motion."

---

## 6. MEVA — Luo, Golestaneh, Kitani. ACCV 2020. arXiv:2008.03789

**Losses (§3.5), verbatim:**
> L_meva = L_3D + L_2D + L_SMPL       (3)
> L_3D = Σ_{t=1}^{T} ||jp3d_t − ĵp3d_t||₂    (4)
> L_2D = Σ_{t=1}^{T} ||jp2d_t − ĵp2d_t||₂    (5)
> L_SMPL = ||β − β̂||₂ + Σ_{t=1}^{T} ||θ_t − θ̂_t||₂   (6)

**No velocity, acceleration or smoothness term whatsoever.** Smoothness is architectural: a motion
VAE latent (coarse, inherently smooth motion) decoded by a *pretrained frozen* VAE decoder, plus a
per-frame Motion Residual Regressor that "is only tasked to do small cosmetic changes to the coarse
estimation".

**Their explicit argument against putting smoothness in the loss (§1), verbatim:**
> "Other methods have been developed to enforce temporal smoothness by letting the model predict
> frame ordering. However, **using prior knowledge only in the loss function, it is hard to find the
> balance between smoothness and accuracy.**"

and §2.3, verbatim: "All above methods use a pose or motion prior in an adversarial way, utilizing
the prior knowledge in the loss function." §3: "existing human motion estimation methods often find
it difficult to achieve a balance between temporal smoothness and accuracy."

---

## 7. HuMoR — Rempe, Birdal, Hertzmann, Yang, Sridhar, Guibas. ICCV 2021. arXiv:2105.04668

**Transition model.** State (Eq. 1), verbatim:
> x = [ r  ṙ  Φ  Φ̇  Θ  J  J̇ ],  x ∈ R^{3×69}
i.e. **velocities are explicit state channels**, not finite differences of a free output.
Transition (Eq. 5): `p_θ(x_t|x_{t−1}) = ∫ p_θ(z_t|x_{t−1}) p_θ(x_t|z_t, x_{t−1})`, a CVAE with a
*learned conditional prior* `p_θ(z_t|x_{t−1}) = N(z_t; μ_θ(x_{t−1}), σ_θ(x_{t−1}))` (Eq. 3).
Decoder is an **Euler-integrator residual**: `x̂_t = x_{t−1} + Δ_θ(z_t, x_{t−1})`.
Verbatim §3: "the decoder acts like a combined physical dynamics model and Euler integrator of
generalized position and velocity".

**Losses (§3.1 + Eq. 7-9), verbatim:**
> L_rec + w_KL L_KL + L_reg   (7),  with L_rec = ||x_t − x̂_t||²
> L_reg = L_SMPL + w_contact L_contact ;  L_SMPL = L_joint + L_vtx + L_consist
> L_joint = ||J^SMPL_t − Ĵ^SMPL_t||²  (8)
> L_vtx = ||V_t − V̂_t||² ,  L_consist = ||Ĵ_t − Ĵ^SMPL_t||²   (9)
> L_contact = L_BCE + L_vel,  "the second regularizes joint velocities to be consistent with contacts
> L_vel = Σ_j ĉ^j_t ||v̂^j_t||₂ with v̂_t ∈ Ĵ̇_t"
> "We set w_contact = 0.01 and w_KL = 4e−4."

So the velocity supervision is (a) an L2 on velocity *channels of the state*, conditioned on the
**ground-truth previous state** (teacher forcing), and (b) a contact-gated **zero**-velocity term at
weight 0.01. There is no free-running finite-difference-vs-GT-velocity loss.

**Frame.** Supp. §B.1, verbatim:
> "Canonical Coordinate Frame. To ease learning and improve generalization, our network operates on
> inputs in a canonical coordinate frame. Specifically, based on x_{t−1} we apply a rotation around
> the up (+z) axis and translation in x, y such that the x and y components of r_{t−1} are 0 and the
> person's body right axis (w.r.t. Φ_{t−1}) is facing the +x direction."

A **gravity-preserving, heading-aligned, root-translation-removed** frame — *not* the full body
frame (roll/pitch are never removed) and not the world frame.

**Explicit stop-gradient + curriculum.** Supp. §B.2, verbatim:
> "Importantly for training stability, if using the model's own prediction x̂_{t−1} as input to t, we
> **do not backpropagate gradients from the loss on x̂_t back through x̂_{t−1}**."
> "For CVAE training, we use 10 epochs of regular supervised training, 10 of mixed true and self
> inputs, and the rest using full self-rollouts."

---

## 8. RoHM — Zhang, Bhatnagar, Xu, Winkler, Kadlecek, Tang, Bogo. CVPR 2024. arXiv:2401.08570

A diffusion denoiser that takes the **noisy per-frame estimates** of a frozen regressor (CLIFF /
MeTRAbs) as conditioning and outputs clean motion. Its velocity loss *is* a finite difference of the
same output the position loss touches, and it works — but note the setting.

**Losses (§4.4, Eq. 15-18), verbatim:**
> L_J3D  = || J3D(R0, P0) − J3D(R, P̂0) ||²    (15)
> L_vel  = || J̇3D(R0, P0) − J̇3D(R, P̂0) ||²   (16)
> L_skate = || f0 J̇3D^foot(R0, P̂0) ||²        (17)
> L = L_simple + λ_J3D L_J3D + λ_vel L_vel + λ_skate L_skate   (18)
> "where (R0, P0) is the ground-truth motion; **R refers to ground-truth root trajectory R0 for
> PoseNet**, and predicted root trajectory R̂0 for TrajNet."

**Weights (Supp.), verbatim:** "For both PoseNet and TrajNet, weights λ3D and λvel are set to **100
and 1000**, respectively. λskate is set to **0 during the first training stage, and 0.1 during the
second training stage** in PoseNet."

So: velocity weighted **10× the position term**, and it does not collapse. Three structural reasons
visible in the paper: (i) it is a *denoiser* whose target is clean AMASS GT with synthetic
corruption — the per-frame conditioning is an input, not a parameter being pulled; (ii) the local
pose velocity is computed with the **GT trajectory** substituted in, so trajectory noise never enters
the local velocity term ("PoseNet is trained with the GT trajectory instead of TrajNet output");
(iii) the foot-skate (zero-velocity) term is staged in at weight 0 → 0.1.

**They also removed velocity channels from the trajectory representation because of drift** (§4.1,
verbatim): "trajectory representation (R^t, R̃, R̂0) for TrajNet is parameterized as (r_l, r_a, r_z,
γ, Φ), **excluding first derivatives to avoid global drifting caused by inaccurate velocities**."

**Frame:** "For each frame n, we define a local coordinate system such that local joint positions
are relative to the current frame pelvis joint, **projected on the ground**" — again the
gravity-preserving, root-XY-removed frame.

---

## 9. GLAMR — Yuan, Iqbal, Molchanov, Kitani, Kautz. CVPR 2022 (oral). arXiv:2112.01524

A third "velocity head + integration" instance, with an explicit architecture finding.

**Global Trajectory Predictor `T`** (§3.2): a CVAE mapping *local body motion* Θ to the **egocentric
trajectory** Ψ — per-frame *deltas* in heading coordinates (Eqs. 5-7: `(δx_t, δy_t) = ToHeading(τ^xy...)`,
`η_t = ToHeading(γ_t)`), then `(T, R) = EgoToGlobal(Ψ)` (Eq. 4), which "**accumulates the egocentric
trajectory to obtain the global trajectory**".

**Training loss (Eq. 8), verbatim:**
> L_T = Σ_{t=1}^{m} ( ||τ_t − τ'_t||₂² + ||γ_t ⊖ γ'_t||²_a ) + L^v_KL

i.e. the loss is on the **accumulated global** τ and γ, not on the per-frame deltas. Same pattern as
GVHMR's `transl_w` rollout and WHAM's multi-scale cumulative term: the *derivative* is the
representation, the *integral* is the supervised quantity.

**Architecture finding — a locality argument for derivative-valued outputs.** §3.2, verbatim:
> "we use LSTMs for temporal modeling instead of Transformers since the output of each frame is the
> local trajectory change in our egocentric trajectory representation, **which mainly depends on the
> body motion of nearby frames and does not require long-range temporal modeling.** We will show in
> Sec. 4.2 that the egocentric trajectory and use of LSTMs instead of Transformers are crucial for
> accurate trajectory prediction."

Table 4 (trajectory predictor on AMASS):

| Method | G-MPJPE | G-PVE | Accel |
|---|---|---|---|
| Transformer (instead of LSTM) | 660.1 | 678.6 | **121.9** |
| Ours w/o Ego Trajectory (direct 6-DoF global) | 763.0 | 780.6 | 8.7 |
| Ours (LSTM + ego trajectory) | **466.9** | **472.5** | **5.8** |

Swapping the local recurrent model for a transformer raised Accel from 5.8 to 121.9 (21×) and
G-MPJPE by ~190 mm. Their attributed reason, verbatim: "(1) the positional encoding in Transformers
may not generalize well to longer motions compared to the LSTMs in our approach; (2) directly
predicting the 6-DoF global trajectory offsets instead of egocentric trajectories from local body
motions [is harder]". [INFERENCE: this is at minimum a demonstration that a *global-attention*
temporal model is a poor fit for a derivative-valued output where a local model is not.]

The only *smoothness* penalty in GLAMR (`E_cam`, §3.3) is on the **camera** parameters, not the human.

**Addendum — GVHMR's velocity channel is standardized.** `hmr4d/model/gvhmr/utils/endecoder.py:102-137`:
the target vector is `x = cat([body_pose_r6d(126), betas(10), global_orient_r6d(6),
global_orient_gv_r6d(6), local_transl_vel(3)])` followed by `x_norm = (x - self.mean) / self.std`.
So the MSE on `local_transl_vel` is taken **in units of that channel's own standard deviation** —
an automatic gradient-scale equalizer, not a raw metric-units comparison. And
`mask_simple[inputs["mask"]["spv_incam_only"], :, 142:] = False` (pipeline:119) switches the GV
orientation + velocity channels off for camera-only datasets (3DPW).

**Addendum — SmoothNet Table 4: attention is a WORSE denoiser than a learned FIR filter here.**
Same task, same losses, same inputs (VIBE on AIST++), only the temporal architecture varies
(`×` = overlapped sliding window):

| Method | Gaussian1d | TCN(27) | TCN(81) | TCN(81)× | TCN(243) | Trans.× | Ours☆ | Ours× |
|---|---|---|---|---|---|---|---|---|
| Accel | 4.95 | 14.46 | 11.84 | 8.71 | 10.07 | **6.15** | 5.45 | **4.15** |
| MPJPE | 103.42 | 103.53 | 101.17 | 99.54 | 99.76 | 99.30 | 98.34 | **97.47** |
| PA-MPJPE | 71.11 | 72.99 | 72.30 | 71.80 | 71.92 | 71.89 | 71.02 | **69.67** |

Verbatim §5.4:
> "(ii). the Accel of TCNs are worse than that of the filter, implying **local aggregation of noisy
> poses with the shared kernels cannot handle large and long-term jitters well**; (iii) the MPJPE of
> TCNs and Transformers are lower than that of the filter, indicating learning-based methods can
> further reduce biased errors S with learning the noisy pose prior; (iv) **Transformer achieves a
> good balance between Accel and MPJPE with the global receptive field at each layer, but not as good
> as SmoothNet. We attribute it to the unnecessary self-attention operations for the pose refinement
> task, which is no guarantee to model the smoothness pattern well.**"

A self-attention block over the window (Accel 6.15) is beaten by a plain learned per-window linear
map (4.15) and is barely better than a hand-tuned Gaussian (4.95) on the smoothness metric. Note the
essential difference: SmoothNet's Eq. (1) is an **unconstrained, signed, learned FIR filter across
the window** (`Σ_t w_t^l * Ŷ_t^l`, weights may be negative and need not sum to 1); softmax attention
is a **convex, non-negative, row-stochastic** combination. [INFERENCE: the convexity constraint is
the structural difference — a convex combination of the window can only interpolate between the
inputs, and combined with a residual `x + P·Σ w_j x_j` the block's only reachable behaviour is
dilution. SmoothNet does not make this argument; the table is theirs, the mechanism is mine.]

---

# Part 2 — papers verified by a parallel search (paper + official code)

*These ten were read by a second agent working from the same brief (arXiv PDFs + official repos).
Verbatim quotes and file:line references are theirs; I have not independently re-read them except
where noted. Everything below is reported as they returned it.*

## TRAM / VIMO — Wang et al., ECCV 2024, arXiv:2403.17346
**No velocity/accel loss used.** §3.4 Eq. (3), verbatim:
> ℒ = λ₂D ℒ₂D + λ₃D ℒ₃D + λSMPL ℒSMPL + λV ℒV
> ℒ₂D = ‖Ĵ₂D − Π(J₃D)‖²_F, ℒ₃D = ‖Ĵ₃D − J₃D‖²_F, ℒSMPL = ‖Θ̂ − Θ‖²₂, ℒV = ‖V̂ − V‖²_F

**Notable:** the repo *implements* `acceleration_loss` (`lib/core/losses.py:62-90`, central second
difference on **root-centered** joints, registered at `losses.py:211-213`) but **no shipped config
enables it** — `configs/config_vimo.yaml` → `LOSS: {KPT2D: 5.0, KPT3D: 5.0, SMPL_PLUS: 1.0, V3D: 1.0}`,
and `compile_criterion` only instantiates keys present in `cfg.LOSS`.
ViT-H backbone frozen (`train.py:53-54`). Frame: pelvis-centered camera / image plane.
Table 4 (EMDB): VIMO 45.7 PA / 74.4 MPJPE / **Accel 4.9** vs HMR2.0 60.7 / 98.3 / 19.9. Verbatim:
> "Without domain-specific designs, VIMO outperforms all other methods in both reconstruction
> accuracy and motion smoothness." … "Removing the motion transformer decreases motion smoothness,
> as it plays a key role in denoising to produce smooth and natural motion."

## SLAHMR — Ye et al., CVPR 2023, arXiv:2302.12827
Test-time optimizer, not a trained model. §3.2 Eq. (8), verbatim:
> "We use a simple prior of joint smoothness, or minimal kinematic motion:
> E_smooth = Σ_i^N Σ_t^T ‖J_i^t − J_i^{t+1}‖²"

Code `slahmr/optim/losses.py:486-499` (`joints3d_smooth_loss`), plus `contact_vel_loss`
(`losses.py:589-601`, *"Velocity should be zero at predicted contacts"*). Frame: **world**.
Weights: paper λ_smooth = 5, λ_β = 0.05, λ_pose = 0.04, λ_data = 0.001; stage 3 λ_CVAE = 0.075,
λ_skate = 100, λ_con = 10. **Staged config verified first-hand by me** — see Part 3.

## PhysPT — Zhang, Kephart, Cui, Ji. CVPR 2024, arXiv:2404.04430
**No velocity-matching loss.** §3.2.1 Eq. (3) verbatim:
> ℒ_recon = Σ_t γ_q ℒ_q,t + γ_J ℒ_J,t,  ℒ_q,t = ‖q_t − q̄_t‖²₂,  ℒ_J,t = ‖J_t − J̄_t‖²₂

§3.2.4 Eqs. (6)-(8): ℒ_force (L1 on τ, λ), ℒ_contact (`γ_v‖v‖₁ + γ_z|z|` — contact-gated **zero**
velocity), ℒ_euler (`‖M q̈ + C + g − J_C^T λ̄ − τ̄‖₁`). GT derivatives come from the **target**:
> "Given a 3D trajectory {q̄_t} from training data, we utilize the finite difference to obtain the
> velocity and acceleration {q̄̇_t, q̄̈_t}."

**Fully decoupled second stage**, trained self-supervised on AMASS only:
> "Once the model is trained, it is directly added on top of the kinematics-based model to obtain
> improved motion estimates and infer motion forces, **without the need of model fine-tuning**."

Frame: world. Weights §4: γ_q = 2e3, γ_J = 1e5, γ_τ = 5, γ_λ = 1, γ_v = 100, γ_z = 200.
**Table 3 (Human3.6M) — the cleanest dose-response in the survey:**

| losses | MJE | P-MJE | ACCL | VEL | FS | GP |
|---|---|---|---|---|---|---|
| kinematics baseline (CLIFF) | **52.2** | 36.8 | 15.4 | 6.8 | 8.3 | 9.3 |
| + ℒ_recon | 52.7 | 36.7 | **2.5** | 3.5 | 7.1 | 6.9 |
| + ℒ_force | 52.7 | 36.7 | 2.5 | 3.4 | 6.5 | 5.6 |
| + ℒ_contact | 53.0 | 36.8 | 2.5 | 3.4 | 4.1 | 1.7 |
| + ℒ_euler (full) | 52.7 | 36.7 | 2.5 | 3.4 | 2.6 | 1.5 |

Verbatim §4.2: *"when trained exclusively with the reconstruction loss (Eq. 3), the model maintains
the reconstruction accuracy while reducing the acceleration and velocity errors"* … *"Imposing the
contact loss (Eq. 7) further reduces the errors but **sacrifices the reconstruction accuracy**."*
**6× accel reduction for +0.5 mm MPJPE, from a pure position-space loss with no velocity term.**

## VIBE — Kocabas et al., CVPR 2020, arXiv:1912.05656
**No velocity/accel loss.** `L_G = L_3D + L_2D + L_SMPL + L_adv`; smoothness from the GRU + motion
discriminator. `lib/core/loss.py:22-147` has no finite differences (`batch_smooth_pose_loss` at
`loss.py:368-377` exists but is **never called**). Frame: pelvis-subtracted 3D + image-plane 2D.
Weights `config.py:88-92`: KP_2D_W 60., KP_3D_W 30., SHAPE_W 0.001, POSE_W 1.0, D_MOTION_LOSS_W 1.
**Trade-off, verbatim §4.1:**
> "While we achieve smoother results compared with the baseline frame-based methods, Temporal-HMR
> yields even smoother predictions. However, we note that Temporal-HMR applies aggressive smoothing
> that results in poor accuracy on videos with fast motion or extreme poses. **There is a trade-off
> between accuracy and smoothness.**" … "Temporal-HMR **over-smooths the pose predictions while
> sacrificing accuracy**." … "Once we add our generator, 𝒢, we obtain **slightly worse but smoother**
> results than the frame-based model."

## MotionBERT — Zhu et al., ICCV 2023, arXiv:2210.06551
**The one paper with a first-class one-step velocity-matching term on a shared head.** §3.3 Eq. (6)-(8):
> ℒ₃D = Σ_t Σ_j ‖X̂_{t,j} − X_{t,j}‖₂,  ℒ_O = Σ_{t=2}^T Σ_j ‖Ô_{t,j} − O_{t,j}‖₂
> where Ô_t = X̂_t − X̂_{t−1}, O_t = X_t − X_{t−1};   ℒ = ℒ₃D + λ_O·ℒ_O + ℒ₂D

Code `lib/model/loss.py:133-142`. Fully shared, no stop-gradient.
**Frame: root-relative** — `train.py:167-168` `batch_gt = batch_gt - batch_gt[:,:,0:1,:]`;
`train.py:75-76` `predicted_3d_pos[:,:,0,:] = 0`; mesh path `loss_mesh.py:36-37,43` re-centers too.
Weights: `MB_pretrain.yaml` / `MB_ft_h36m.yaml` `lambda_3d_velocity: 20.0` (position implicit 1);
`MB_ft_pw3d.yaml` `lambda_3d: 0.5, lambda_3dv: 10, lambda_pose: 1000, lambda_shape: 1`.
[INFERENCE, agent's] the 20:1 ratio is roughly a magnitude equalizer, not a 20× emphasis, since
one-step displacements on normalized root-relative joints are 1-2 orders of magnitude below positions.

## GLoT — Shen et al., CVPR 2023, arXiv:2303.14747
Velocity loss **masked**, §3.4 Eq. (3), verbatim:
> "we empirically discover that constraints on the velocity of the predicted 3D/2D joint location can
> help the model learn motion consistency and capture the long-range dependency better **when applying
> the Masked Pose and Shape Estimation strategy to the global transformer**.
> ℒ_vel_2d = Σ_t m_t‖(jt2d^{t+1} − jt2d^t) − (gt2d^{t+1} − gt2d^t)‖₂ … where m_i is **1 when masking
> the i location, otherwise 0**."

Frame: root-relative 3D + weak-perspective 2D. Shared params, no stop-gradient.
Shipped weights (all three configs): vel_2d 10., vel_3d 100., `use_accel: False` (code default is
`True` — configs override it). Claims to escape the trade-off architecturally (global/local split):
*"using a single kind of modeling structure is difficult to balance the learning of short-term and
long-term temporal correlations."*

## PMCE — You et al., ICCV 2023, arXiv:2308.10305
**No temporal loss at all.** §3.4 Eq. (12): `ℒ = λ_m ℒ_mesh + λ_j ℒ_joint + λ_n ℒ_normal + λ_e ℒ_edge`,
λ_m = 1, λ_j = 1, λ_n = 0.1, λ_e = 20 (code says `joint_loss_weight: 1e-3` — paper/code discrepancy).
3DPW: 69.5 MPJPE / 46.7 PA / **6.5 ACCEL** — best on both.
**The most explicit trade-off statement in the literature (§1), verbatim:**
> "Although these video-based methods significantly improve the temporal consistency of 3D human
> motion, **there exists a trade-off between per-frame accuracy and motion smoothness** for the
> following two main reasons: 1) **The highly coupled image feature.** … 2) **The limited
> representation ability of the parametric human model.**"

and §4.3, verbatim:
> "MAED has a trade-off between per-frame accuracy (PA-MPJPE) and temporal consistency (ACCEL).
> Specifically, **when MAED reduces PA-MPJPE by 11 mm, it increases ACCEL by 11.1 mm/s²** compared to
> our PMCE on 3DPW."

## 4DHumans / HMR2.0 — Goel et al., ICCV 2023, arXiv:2305.20091
**Confirmed: zero temporal or velocity loss; pure single-frame.** §3.3 complete loss set:
`ℒ_smpl = ‖θ − θ*‖²₂ + ‖β − β*‖²₂`, `ℒ_kp3D = ‖X − X*‖₁`, `ℒ_kp2D = ‖π(X) − x*‖₁`,
`ℒ_adv = Σ_k (D_k(θ_b, β) − 1)²`. `hmr2/models/losses.py` is 92 lines, three loss classes, no time
axis; `Keypoint3DLoss` re-centers on pelvis id 39. Weights: KEYPOINTS_3D 0.05, KEYPOINTS_2D 0.01,
GLOBAL_ORIENT 0.001, BODY_POSE 0.001, BETAS 0.0005, ADVERSARIAL 0.0005.
Temporal modelling lives entirely downstream in PHALP′ (a separate masked-token transformer over
tracks) and never backprops into HMR2.0. Consequence: Accel 18.1 (3DPW) / 19.9 (EMDB).

## D&D — Li et al., ECCV 2022, arXiv:2209.08790
Paper §3.6 Eq. (16)-(21) lists **no** velocity/accel loss. **The released code has both**,
`dnd/models/criterion.py`:
* `accel_3d_loss` (L275-299): vel + accel MSE on **root-relative** joints
  (`gt_keypoints_3d - gt_pelvis`, `pred_keypoints_3d - pred_pelvis`), weighted
  `KP_3D_ACCEL_W: 300.0` — **equal to `KP_3D_W: 300.0`**.
* Global translation (L190-207) — **the anchor+derivative pairing**:
```python
pred_global_transl = pred_global_transl - pred_global_transl[:, [0], :]
real_global_transl = real_global_transl - real_global_transl[:, [0], :]
...
loss_g_transl   = torch.mean(torch.abs(pred_global_transl - real_global_transl))   # ABSOLUTE
loss_a_g_transl = torch.mean(torch.abs(a_g_transl - a_real_transl))               # ACCELERATION
loss_dict['loss_global_transl']     = self.global_transl_weight * loss_g_transl * 1
loss_dict['loss_acc_global_transl'] = self.global_transl_weight * loss_a_g_transl
```
An absolute-position term and an acceleration term on the **same** trajectory at the **identical
weight** (`G_TRANSL_W: 100.0`), both after subtracting frame 0 so the constant-offset null space is
removed by construction. The pose itself is **integrated** from predicted accelerations by
semi-implicit Euler (Eqs. 11-12), so what the derivative loss shapes is already smooth by construction.
Frame: **non-inertial camera** frame for the dynamics; joint terms root-relative.
Agent's caveats: `self.debug = True` (L62) disables the translation and contact blocks as shipped,
and three weight keys the criterion reads are absent from `configs/dnd.yaml` — treat the yaml as
indicative.
**Anti-jitter mechanism is architectural:** *"The output of the conventional PD controller is
proportional to the distance of the current pose state from the target state, which is sensitive to
the unstable and jittery target. Instead, our attentive PD controller allows accurate control by
globally adjusting the target state and is robust to the jittery target."*

## PhysDiff — Yuan et al., ICCV 2023, arXiv:2212.02500
Physics applied at **sampling** time, not in training: a frozen RL imitation policy (whose reward
includes a velocity term) projects each denoised motion into a physically plausible one. No gradient
path to the diffusion model. No released code, weights unverified.
**Non-monotone dose-response, §4.3 verbatim:**
> "we observe a **trade-off between physical plausibility and motion quality** when varying the number
> of physics-based projection steps. Specifically, while more projection steps always lead to better
> physical plausibility, **the motion quality increases before a certain number of steps and decreases
> after that**."

and on *when* to constrain:
> "We also find that adding the physics-based projection to late diffusion steps performs better than
> early steps. We hypothesize that motions from early diffusion steps may tend toward the **mean
> motion** of the training data and the physics-based projection could push the motion further away
> from the data distribution."

---

# Part 3 — theory: shrinkage, gradient routing, residual denoising, scale-invariant depth rates

*Gathered by a parallel search agent from the primary PDFs. Quotes are reproduced as returned;
second-hand unless I note otherwise.*

## (a) Derivative / velocity losses → amplitude shrinkage, over-smoothing, mean collapse

### The closest published statement of the symptom

**MEVA** — Luo, Golestaneh, Kitani, ACCV 2020, arXiv:2008.03789, **Supplementary §2.3**, verbatim:
> "using only VME will lead to an **overly smoothed motion estimation** and result in a higher
> acceleration error (**underestimating movement also leads to high acceleration error**)."

Main paper §1, verbatim: *"using prior knowledge only in the loss function, **it is hard to find the
balance between smoothness and accuracy**."* Supplementary §2.1: *"average filtering can help reduce
the acceleration error of both VIBE and MEVA **while slightly affecting accuracy**."*

**TCMR** §2 (verified first-hand, see Part 1): *"they revealed a **trade-off between per-frame
accuracy and temporal consistency**."*

**SmoothNet** §1, verbatim: *"applying low-pass filters on each estimated joint with a long filtering
window could reduce jitters to an arbitrarily small value. Nevertheless, **such fixed temporal
filters usually lead to considerable precision loss (e.g., over-smoothing)** without prior knowledge
about the distribution of human motions."* and *"significant jitters occur on those challenging
frames with large estimation errors **as L1/L2 loss optimization is directionless**."* Fig. 2 caption
gives the decomposition to cite: *"Output errors are composed of jitter errors J and biased errors S."*

### Speech / TTS — the canonical derivative-constraint → variance-shrinkage case

This is the textbook instance of exactly your mechanism: MLPG generates a static trajectory under an
explicit constraint linking statics to their **dynamic (delta) features**, and that constraint is
what shrinks the variance.

**Toda & Tokuda**, *"Speech parameter generation algorithm considering global variance for HMM-based
speech synthesis"*, Interspeech 2005, DOI 10.21437/Interspeech.2005-617. Abstract, verbatim:
> "The conventional algorithm generates a trajectory of static features that maximizes an output
> probability of a parameter sequence consisting of the static and dynamic features from HMMs
> **under an actual constraint between the two features. The generated trajectory is often
> excessively smoothed due to the statistical processing.** ... we propose the generation algorithm
> considering not only the output probability used for the conventional method but also that of a
> **global variance (GV)** of the generated trajectory. **The latter probability works as a penalty
> for a reduction of the variance of the generated trajectory.**"

Journal version to cite: Toda & Tokuda, IEICE Trans. Inf. & Syst., **E90-D(5):816–824, 2007**,
DOI 10.1093/ietisy/e90-d.5.816.
MLPG itself: Tokuda, Yoshimura, Masuko, Kobayashi, Kitamura, ICASSP 2000, pp. 1315–1318 (title/venue
confirmed; **DOI unverified**).

**Quantified shrinkage:** Saito, Takamichi, Saruwatari, *"Statistical Parametric Speech Synthesis
Incorporating Generative Adversarial Networks"*, IEEE/ACM TASLP 2018, arXiv:1709.08041, Table III
(mora duration): Natural mean 25.141 / **variance 131.93**; MSE-trained (MGE) 23.492 / **60.891**;
GAN 24.978 / 96.682. **MSE training halves the variance of the generated trajectory.**

The modulation-spectrum post-filter line (Takamichi et al.) exists but was **not read directly** —
treat that citation as unverified.

### Video SR / denoising / prediction

**Lai, Huang, Wang, Shechtman, Yumer, Yang**, *"Learning Blind Video Temporal Consistency"*, ECCV
2018, arXiv:1808.00449. §4.4, verbatim — the trade-off as an explicit loss-weight dial:
> "An extremely blurred video may have high temporal stability but with low perceptual similarity;
> in contrast, the processed video itself has perfect perceptual similarity but is temporally
> unstable. **Due to the trade-off between the temporal stability and perceptual similarity**, it is
> important to balance these two properties" … "When the ratio r > 10, **the output videos become
> overly blurred**."

§6: *"in the way the task is formulated **there is always a trade-off between being temporally
coherent or perceptually similar** to the processed video."*

**TecoGAN** — Chu, Xie, Mayer, Leal-Taixé, Thuerey, ACM TOG (SIGGRAPH 2020), arXiv:1811.09393.
Abstract, verbatim, specifically about **temporal L2 losses**:
> "state-of-the-art methods often favor simpler norm losses such as L2 over adversarial training.
> However, **their averaging nature easily leads to temporally smooth results with an undesirable
> lack of spatial detail.**"

§1: *"While L1 and L2 temporal losses based on warping are generally used to enforce temporal
smoothness … **it leads to an undesirable smooth over spatial detail and temporal changes in
outputs.**"* §3.2 is explicit that L2 is only safe when the target is unimodal.

**SRGAN** (Ledig et al., CVPR 2017, arXiv:1609.04802), Fig. 3 caption: *"**The MSE-based solution
appears overly smooth due to the pixel-wise average of possible solutions** in the pixel space."*

**Mathieu, Couprie, LeCun**, *"Deep multi-scale video prediction beyond mean square error"*, ICLR
2016, arXiv:1511.05440, §2 "Problem 2" — the cleanest mean-collapse statement:
> "**If the probability distribution for an output pixel has two equally likely modes v1 and v2, the
> value v_avg = (v1 + v2)/2 minimizes the ℓ2 loss over the data, even if v_avg has very low
> probability.** In the case of an ℓ1 norm, this effect diminishes, but does not disappear."

(Note the mirror image: their *fix* is an image-**gradient** difference loss added to counter
smoothing.)

### Motion forecasting / generation — mean-trajectory and mean-pose collapse

**Social GAN** (Gupta et al., CVPR 2018, arXiv:1803.10892) §3.3: *"these predictions try to produce
the '**average**' prediction in cases where there can be multiple outputs."*

**Ginosar, Bar, Kohavi, Chan, Owens, Malik**, *"Learning Individual Styles of Conversational
Gesture"*, CVPR 2019, arXiv:1906.04160, §4.2, verbatim — amplitude collapse, and note their design
choice:
> "While L1 regression to keypoints is the only way we can extract a training signal from our data,
> **it suffers from the known issue of regression to the mean which produces overly smooth motion**
> … To combat this … we add an adversarial discriminator D, **conditioned on the differential of the
> predicted sequence of poses**"

i.e. they put a **GAN** on the velocity, not an L2 — because an L2 on the velocity is the thing that
shrinks.

**Kucherenko et al.**, *"Moving fast and slow"*, IJHCI 2021, arXiv:2007.09170. Their Table 1 includes
a **"Static mean pose"** baseline: APE 8.95 cm with speed / acceleration / jerk all exactly **0**,
against their best trained model at APE 7.65 — the degenerate zero-motion solution is within ~15 %
of trained models on position error. §3.2: *"**incorporating the velocity (finite difference) into
predictions did not provide a significant improvement at test time**, [though] including velocity as
a multitask objective helped the network learn motion dynamics during training."*

### The shrinkage theory itself

**Johnstone**, *Gaussian Estimation: Sequence and Wavelet Models* (Cambridge UP; draft PDF
9 Aug 2017), **Eq. (1.4)**, verbatim — this is exactly `s* = SNR/(1+SNR)`:
> θ̂_k = (λ/(λ+1)) y_k,  λ = τ²/σ²
> "**The constant λ is the squared signal-to-noise ratio. The estimator, sometimes called the Wiener
> filter, is optimal in the sense of minimizing the posterior expected squared error.**"

**Correction to the working hypothesis:** Donoho & Johnstone, *"Ideal spatial adaptation by wavelet
shrinkage"*, **Biometrika 81(3):425–455, 1994**, does **not** contain `θ²/(θ²+σ²)`. It gives the
diagonal-*projection* oracle, §1.3, verbatim: *"These ideal coefficients are δ_i = 1{|θ_i|>σ} …
ideal diagonal projection consists in estimating only those θ_i larger than the noise level …
R(DP, θ) = Σ min(θ_i², σ²)."* Cite **Johnstone Eq. 1.4 / the scalar Wiener filter** for the
shrinkage form, DJ94 for keep-or-kill.

**A derivative penalty IS a frequency-dependent shrinkage — the Demmler–Reinsch form.** From
R. Tibshirani's lecture notes §2.5 "Reinsch form" (Berkeley, stat-learn S23), verbatim:
> "S = Σ_j 1/(1+λσ_j) u_j u_jᵀ … **smoothing splines perform a regression on the orthonormal set
> u_1,…,u_n, but they shrink the coefficients, with more shrinkage applied to an eigenvector u_j
> that corresponds to a large eigenvalue σ_j.** … **by increasing λ in the smoothing spline
> estimator, we are tuning out the more wiggly components.**"

Primary sources: Demmler & Reinsch (1975); Hastie–Tibshirani–Friedman, *ESL* 2nd ed. §5.4.1
(**ESL equation numbering unverified**).

**Differencing amplifies noise:** Chartrand, *"Numerical Differentiation of Noisy, Nonsmooth Data"*,
ISRN Applied Mathematics vol. 2011, art. 164564, DOI 10.5402/2011/164564 — abstract: regularizing
differentiation, *"avoiding the noise amplification of finite-difference methods"*. (Abstract only.)

> **Gap, stated honestly.** No paper was found that derives `s* = SNR_v/(1+SNR_v)` for a
> **finite-difference velocity loss** specifically, nor one that writes the loss gradient as a
> discrete Laplacian with gain `4 sin²(ω/2)/dt²`. Both follow from standard results (Johnstone
> Eq. 1.4; the DFT of the second-difference operator), but the *composition* appears unpublished.
> Present it as your own derivation, citing Johnstone for the shrinkage and Chartrand for noise
> amplification under differencing.

---

## (b) Gradient routing / stop-gradient — training only the temporal module

Four directly analogous designs, two of which are *exactly* "frozen per-frame predictor + trained
temporal refiner".

**NVDS+** — Wang et al., *"NVDS+: Towards Efficient and Versatile Neural Stabilizer for Video Depth
Estimation"*, arXiv:2307.08695 (NVDS at ICCV 2023; NVDS+ the TPAMI extension). §3.7, verbatim — the
single most on-point citation:
> "**In the training phase, only the stabilization network is optimized. The depth predictor is the
> freezed pre-trained DPT-Large.**"

**Video Depth Anything** — Chen et al. (ByteDance), arXiv:2501.12375. §3.1, verbatim:
> "We use its trained model as our encoder. **To reduce training costs and preserve well-learned
> features, the encoder is frozen during training.**"

§1, verbatim — the *justification* you want, that letting the temporal signal reach the per-frame
representation corrupts it:
> "**Introducing temporal attention only in the head prevents the learned representation from being
> corrupted by the limited video data.**"

**Eigen, Puhrsch, Fergus**, NeurIPS 2014, arXiv:1406.2283, §3.1, verbatim:
> "We train the coarse network first against the ground-truth targets, then train the fine-scale
> network keeping the coarse-scale output fixed (i.e. **when training the fine network, we do not
> backpropagate through the coarse one**)."

**SmoothNet** (verified first-hand): the per-frame estimator's outputs are dumped to disk and the
refiner trained on them — the estimator is never in the graph. Rationale, §1, verbatim: *"existing
learning-based solutions … employ Spatio-temporal models to co-optimize per-frame precision and
temporal smoothness at all the joints. **This is a highly challenging task.**"*

**Lai et al. 2018**: "blind" temporal consistency — the post-processor is "agnostic to specific image
processing algorithms applied to the original video", so the consistency loss structurally cannot
reach the per-frame stage.

**SimSiam** — Chen & He, CVPR 2021, arXiv:2011.10566, abstract, for the general principle:
> "**collapsing solutions do exist for the loss and structure, but a stop-gradient operation plays an
> essential role in preventing collapsing.**"

Also relevant, verified first-hand elsewhere in this survey: **HuMoR** explicitly stop-gradients the
rollout (*"we do not backpropagate gradients from the loss on x̂_t back through x̂_{t−1}"*),
**WHAM** detaches the contact gate in its foot-slide loss, **GVHMR** rolls out its velocity with the
**GT** orientation and GT origin, and **RoHM** trains PoseNet with the **GT** trajectory substituted
for its own.

> **Honest gap:** no paper was found that (i) trains a velocity/consistency loss end-to-end,
> (ii) reports that it degraded the per-frame estimate, and (iii) therefore adds a `detach`. The
> papers above all *start* from the frozen/two-stage design. The closest causal justification is
> Video Depth Anything's "prevents the learned representation from being corrupted".

---

## (c) Residual vs non-residual temporal blocks for denoising

### Denoising residuals predict the NOISE — they SUBTRACT

**DnCNN** — Zhang, Zuo, Chen, Meng, Zhang, IEEE TIP 26(7) 2017, arXiv:1608.03981. §III, verbatim:
> "For DnCNN, we adopt the residual learning formulation to train a residual mapping **R(y) ≈ v**,
> and then we have **x = y − R(y)**."

§III-B, verbatim:
> "when the original mapping is more like an identity mapping, the residual mapping will be much
> easier to be optimized. Note that **the noisy observation y is much more like the latent clean
> image x than the residual image v** … Thus, F(y) would be more close to an identity mapping than
> R(y), and the residual learning formulation is more suitable for image denoising."

**The structural point:** DnCNN's residual is `out = in − f(in)`. A standard pre-LN transformer block
is `out = in + f(in)`. Opposite signs. To denoise, the branch must subtract.

### ⚠️ Correction: residual connections do NOT bias toward low-pass — they PRESERVE the noise

The stated hypothesis ("residual attention dilutes because residuals bias toward averaging") is
**contradicted in that form**. The literature proves the opposite division of labour: the
**attention branch** is the low-pass part, and the **skip is what prevents** low-pass collapse. This
makes the dilution story stronger, not weaker — the skip faithfully carries the white noise through
untouched, and an additive low-pass branch can only add a smoothed copy on top of it.

**Wang, Zheng, Chen, Wang**, *"Anti-Oversmoothing in Deep Vision Transformers via the Fourier Domain
Analysis"*, **ICLR 2022**, arXiv:2203.05962. Abstract, verbatim:
> "We show that **the self-attention mechanism inherently amounts to a low-pass filter**, which
> indicates when ViT scales up its depth, excessive low-pass filtering will cause feature maps to
> only preserve their Direct-Current (DC) component."

§2.3 "Does residual connection benefit?" + **Proposition 5** (App. C.2), verbatim:
> "**residual connection can effectively prevent high-frequency component from diminishing to zero**
> by promoting the rate σ₁σ₂H√(ne^{2α}/(e^{2α}+n−1)) to 1 + σ₁σ₂H√(ne^{2α}/(e^{2α}+n−1)) > 1."

**Park & Kim**, *"How Do Vision Transformers Work?"*, **ICLR 2022**, arXiv:2202.06709, verbatim:
> "**MSAs are low-pass filters, but Convs are high-pass filters.**" … "MSAs … themselves as a general
> form of **spatial smoothing or an implementation of ensemble averaging for proximate data points**."
> … "**Yu et al. (2022) demonstrated that the MSA layers of ViT can be replaced with non-trainable
> average pooling layers.**"

### Uniform attention / rank collapse — the published name for "wide near-uniform pooling"

**Dong, Cordonnier, Loukas**, *"Attention is not all you need: pure attention loses rank doubly
exponentially with depth"*, **ICML 2021**, arXiv:2103.03404. Abstract, verbatim:
> "we prove that **self-attention possesses a strong inductive bias towards 'token uniformity'.**
> Specifically, **without skip connections or multi-layer perceptrons (MLPs), the output converges
> doubly exponentially to a rank-1 matrix.** On the other hand, **skip connections and MLPs stop the
> output from degeneration.**"

**DeepViT** — Zhou et al., arXiv:2103.11886, abstract, verbatim:
> "such scaling difficulty is caused by the **attention collapse** issue: **as the transformer goes
> deeper, the attention maps gradually become similar and even much the same after certain layers.**"

### Empirical evidence from human pose specifically (verified first-hand — see Part 1)

* **TCMR Table 1**: removing the residual skip from the per-frame static feature → Accel 29.2 → 8.7,
  PA-MPJPE 55.6 → 54.2. *"the identity mapping of the current static feature inside the residual
  connection hinders a model from learning meaningful temporal features."*
* **SmoothNet Table 4**: self-attention block Accel 6.15 vs learned signed FIR filter 4.15 vs plain
  Gaussian 4.95. *"we attribute it to the unnecessary self-attention operations for the pose
  refinement task, which is no guarantee to model the smoothness pattern well."*
* **GLAMR Table 4**: LSTM → Transformer on a derivative-valued output: Accel 5.8 → 121.9.

> **[INFERENCE — not published anywhere found.]** Write one pre-LN block as `out = x + g·P·A·x`,
> with `A` row-stochastic over frames and `g` a zero-init gate. For a mean-zero white component `n`,
> a near-uniform `A ≈ (1/T)11ᵀ` gives `A·n ≈ 0`, so `out ≈ n` — **the additive branch with pooling
> attention leaves the noise untouched and only rescales the DC/mean component**. That is dilution.
> Genuine denoising needs the *convex* form `out = (1−g)x + g·A·x`, i.e. the branch must emit
> `A·x − x`, which requires a second near-diagonal head with a negative output projection to cancel
> `x`. From a zero-init gate, gradient descent reaches the single pooling head first — and that head
> alone is exactly `x + g·mean(x)`: an amplitude change without denoising. The supporting *published*
> facts are DnCNN (denoising needs subtraction), Wang et al. Prop. 5 (the skip preserves the AC
> band), Park & Kim (the attention branch is the averaging part), and SmoothNet's Eq. (1), whose
> filter weights `w_t` are **unconstrained and signed** — unlike a row-stochastic softmax, which can
> only interpolate within the convex hull of the window.

Downloaded but **not** read closely enough to cite: He et al. *"Identity Mappings in Deep Residual
Networks"* (arXiv:1603.05027), Veit et al. *"Residual Networks Behave Like Ensembles"*
(arXiv:1605.06431), Jastrzebski et al. *"Residual Connections Encourage Iterative Inference"*
(arXiv:1710.04773), Greff et al. *"Highway and Residual Networks learn Unrolled Iterative
Estimation"* (arXiv:1612.07771). The last two are the right citations for the "residual block = small
refinement step" view — **unverified**.

---

## (d) Scale-invariant velocity / derivative supervision for monocular depth

### The two canonical scale-invariant losses

**Eigen, Puhrsch, Fergus**, NeurIPS 2014, arXiv:1406.2283, **§3.2 "Scale-Invariant Error", Eq. (1)**:
> D(y, y*) = (1/n) Σ_i (log y_i − log y_i* + α(y,y*))²,  α = (1/n) Σ_i (log y_i* − log y_i)
> "For any prediction y, e^α is the scale that best aligns it to the ground truth. **All scalar
> multiples of y have the same error, hence the scale invariance.**"

Motivation, verbatim — directly relevant to a depth that drifts toward the camera:
> "**much of the error accrued using current elementwise metrics may be explained simply by how well
> the mean depth is predicted.** … just finding the average scale of the scene accounts for a large
> fraction of the total error."

**MiDaS** — Ranftl, Lasinger, Hafner, Schindler, Koltun, IEEE TPAMI 2020/2022, arXiv:1907.01341, §5:
> "**We propose to perform prediction in disparity space (inverse depth up to scale and shift)
> together with a family of scale- and shift-invariant dense losses.**"

### The best single hit for the depth-shrinkage symptom

**Robust CVD** — Kopf, Rong, Huang, *"Robust Consistent Video Depth Estimation"*, CVPR 2021,
arXiv:2012.05901, **§4 "Reprojection loss"**, verbatim:
> "A naïve way would be to simply measure the Euclidean distance L_euclidean(a,b) = ‖a − b‖².
> **However, this biases the solution toward small depths. Shrinking the whole scene to a point would
> achieve a minimum.**"
> "To prevent this, Luo et al. use a split loss where they measure the spatial component L_spatial in
> image space … and the depth L_depth component in disparity space … **The disparity loss actually
> has the opposite bias of Euclidean loss: it is minimized when scene scale grows very large (so that
> the disparities vanish).**"
> "To alleviate this, we propose a new loss that measures the ratio of depth values:
> **L_ratio(a,b) = max(a_z,b_z)/min(a_z,b_z) − 1. This loss does not suffer from any depth bias; it
> does neither encourage growing nor shrinking the scene scale.**"

Since `max/min − 1 ≈ |log(a_z/b_z)|` for small relative differences, **their fix *is* a log-depth
loss, adopted for exactly the no-scale-bias reason.** This is the citation for "a metric-space depth
residual has a built-in bias toward the camera; a ratio/log residual does not."

### The closest published analogue to a depth-velocity loss

**Video Depth Anything**, arXiv:2501.12375, **§3.2 Eq. (3)**, verbatim:
> "we posit that the change in depth of corresponding points between adjacent prediction frames
> should be consistent with the change observed in ground truth … we name it **Temporal Gradient
> Matching Loss**: L_TGM = (1/(N−1)) Σ_i ‖ |d_{i+1} − d_i| − |g_{i+1} − g_i| ‖₁"

Three design details worth copying:
1. **It lives in affine-invariant disparity space, not metric depth**, combined with MiDaS's SSI
   loss: `L_all = α·L_TGM + β·L_ssi`. So the temporal difference *is* normalised — by inverse-depth
   plus scale-shift alignment — though they justify the disparity choice by inheritance from
   MiDaS/DAv2, **not** on scale-bias grounds.
2. **It matches |Δd| — absolute magnitudes**, not signed differences.
3. **Explicit outlier masking**, verbatim: *"we only compute the TGM loss in regions where the change
   in ground truth depth, i.e., |g_{i+1} − g_i| < 0.05. This threshold helps to avoid sudden changes
   in depth map caused by edges, dynamic objects, and other factors that introduce unsteadiness
   during training."*

Their Table 4 is a clean published instance of the trade-off in depth: alignment-only losses
("VideoAlign", "VideoAlign+SSI") give good geometry / poor stability (AbsRel 0.151, TAE 1.326/1.207);
flow-warping OPW+SSI gives good stability / **badly degraded geometry** (AbsRel 0.182, δ1 0.771 vs
0.846). TGM+SSI is the only one that gets both.

**NVDS+** (arXiv:2307.08695) also works in disparity throughout, but Supp. §B.4 gives **pragmatic**
reasons only (dataset annotation format, matching MiDaS/DPT I/O) — no scale-bias argument.

**Consistent Video Depth** — Luo, Huang, Szeliski, Matzen, Kopf, SIGGRAPH 2020, arXiv:2004.15021,
DOI 10.1145/3386569.3392377 — the split spatial/disparity loss that Robust CVD critiques above. CVD's
own framing is **test-time fine-tuning** of a single-image network, a different gradient-routing
answer to the same problem.

> **Gap on (d):** no paper was found that supervises `d(log z)/dt` (or `Δz/z`) as a *temporal*
> derivative and states why. The three nearest published things are Robust CVD's `L_ratio` (the
> log/ratio idea with the exact scale-bias justification, but on a *spatial* reprojection residual),
> Video Depth Anything's TGM (a temporal finite difference that happens to live in SSI disparity
> space, with no scale-bias argument), and Eigen's SILog (scale-invariant but per-frame).
> **[INFERENCE]** Composing Robust CVD's argument with a temporal difference — supervising
> `Δ log z` — appears unpublished, and is defensible directly from the Robust CVD quote.

---

# Part 4 — Transition-dynamics regularisation: how the "noisy observation + learned prior" family supervises transitions

The common structure across this family: **the transition model is a generative prior fitted to
clean motion capture; the noisy per-frame estimates enter only as an observation/data term or as
conditioning. Nothing in the family computes a finite difference of a jointly-trained pose head and
matches it to a GT finite difference.**

**HuMoR** (ICCV'21, arXiv:2105.04668) — see §7 above. `p_θ(x_t|x_{t−1})` is a CVAE with a learned
conditional prior; velocities are *state channels*; the decoder is `x̂_t = x_{t−1} + Δ_θ(z_t, x_{t−1})`;
`L_rec = ||x_t − x̂_t||²` under **teacher forcing on x_{t−1}**, with an explicit stop-gradient once
scheduled sampling switches to self-rollouts. Compared in a **heading-aligned, gravity-preserving
canonical frame** rebuilt at every step from `x_{t−1}` (Supp. §B.1), not in the world frame and not
in a full body frame.

**SLAHMR** (CVPR'23, arXiv:2302.12827) uses HuMoR as its prior. Its own naive derivative penalty
`E_smooth = Σ_i Σ_t ||J_i^t − J_i^{t+1}||²` (Eq. 8, a **zero-target** term, world frame) is used only
as a *warm start*: in the released `slahmr/confs/optim.yaml` the staged weights are
`joints3d_smooth: [1.0, 10.0, 0.0]` — **switched off in the final stage**, exactly where
`motion_prior: [0.0, 0.0, 0.075]` (HuMoR) switches on. The learned prior replaces the derivative
penalty rather than accompanying it. **I verified this config myself** — the relevant lines of
`slahmr/confs/optim.yaml`, verbatim (the three entries are the weights for the `root`, `smooth` and
`motion_chunks` stages):

```yaml
    joints3d_smooth: [1.0, 10.0, 0.0]
    joints3d_rollout: [0.0, 0.0, 0.0]
    motion_prior: [0.0, 0.0, 0.075]
    init_motion_prior: [0.0, 0.0, 0.075]
    joint_consistency: [0.0, 0.0, 100.0]
    bone_length: [0.0, 0.0, 2000.0]
    contact_vel: [0.0, 0.0, 100.0]
    contact_height: [0.0, 0.0, 10.0]
    cam_R_smooth : [0.0, 0.0, 0.0]
    cam_t_smooth : [0.0, 0.0, 0.0]
```
Note also that the contact-gated **zero**-velocity term (`contact_vel`, weight 100) is active only in
the final stage, and that the camera smoothness terms ship at 0.0 (the commented-out alternative is
`[0.0, 1000.0, 1000.0]`).

**PhysPT** (CVPR'24, arXiv:2404.04430) is the purest form of the two-stage answer: a transformer
trained **self-supervised on AMASS only**, then applied on top of a frozen kinematics estimator
"without the need of model fine-tuning". Its reconstruction loss is position-space
(`L_q = ‖q_t − q̄_t‖²`, `L_J = ‖J_t − J̄_t‖²`); GT derivatives are obtained by finite-differencing the
**target**, and appear only inside the Euler-Lagrange residual `‖M q̈ + C + g − J_C^T λ̄ − τ̄‖₁`.
Reported: Accel 15.4 → 2.5 for +0.5 mm MPJPE from the reconstruction loss alone; the contact
(zero-velocity) term "further reduces the errors but **sacrifices the reconstruction accuracy**".
World frame. *(Second-hand from the parallel search.)*

**PhysDiff** (ICCV'23, arXiv:2212.02500): physics enters at **sampling** time via a frozen RL
imitation policy in a simulator (velocity appears only as an RL reward), with no gradient path to
the diffusion model. It reports a **non-monotone** dose-response: "the motion quality increases
before a certain number of steps and decreases after that". *(Second-hand.)*

**RoHM** (CVPR'24, arXiv:2401.08570) — §8 above. The only member of this family whose velocity loss
is a finite difference of the trained output (λ_vel = 1000 vs λ_3D = 100), and it survives because
the target is *clean AMASS with synthetic corruption*, the local velocity term is computed with the
**GT trajectory substituted in**, and the zero-velocity foot term is staged in at 0 → 0.1. They also
**removed velocity channels from the trajectory representation** "to avoid global drifting caused by
inaccurate velocities". Local frame = joints relative to the current-frame pelvis **projected on the
ground** — again gravity-preserving and heading-ish, not a full body frame.

**Kalman-style latent smoothing — KVAE** (Fraccaro, Kamronn, Paquet, Winther, NeurIPS 2017,
arXiv:1710.05741). The canonical "noisy observation + learned prior with exact smoothing": a VAE
encodes each frame into a pseudo-observation `a_t`, and the temporal model is a **linear-Gaussian
state space model** whose parameters `γ_t` are produced by an RNN from past observations. Verbatim
(§2/§3): "the filtered and smoothed posteriors `p(z_t|a_{1:t}, u_{1:t})` and `p(z_t|a, u)` can be
computed **exactly** with the classical Kalman filter and smoother algorithms"; "Smoothing is still
possible as the state transition matrix `A_t` and others in `γ_t` do not have to be constant". The
training objective is an ELBO (reconstruction + KL) — **there is no derivative loss anywhere**. In
their bouncing-ball experiment "`z_t` has to encode the ball's position and velocity", i.e. velocity
lives in the *latent state*, inferred by the smoother, never regressed against a finite-differenced
target. Frame: the model's own latent, so the question does not arise.

**Where they compare — body frame vs world frame.** Every method in this family that had to choose a
frame chose the same thing: **gravity-preserving, heading-aligned, root-translation-removed**.
HuMoR ("rotation around the up (+z) axis and translation in x, y such that the x and y components of
`r_{t−1}` are 0 and the person's body right axis is facing the +x direction"), GLAMR (heading
coordinates, "the heading vector … is parallel to the ground"), GVHMR (GV = gravity × camera-view),
WHAM (egocentric velocity with world root orientation), RoHM ("relative to the current frame pelvis
joint, **projected on the ground**"). None of them compares in the **full body frame** (which would
also remove roll and pitch), and none compares raw world velocities without first factoring out
heading. [INFERENCE: the shared motivation is that the *conditional distribution* of the next
increment is (near-)invariant to heading and to absolute position, but is **not** invariant to
pitch/roll — gravity is a real, informative direction. Removing roll and pitch, as a full body-frame
comparison does, discards that.]

---

# Part 5 — corrections, gaps, and what nobody has published

## Corrections to assumptions in the brief

1. **arXiv 2011.00980 is not MEVA.** That id is Biggs et al., *"3D Multi-bodies: Fitting Sets of
   Plausible 3D Human Models to Ambiguous Image Data"*. **MEVA is arXiv:2008.03789** (Luo,
   Golestaneh, Kitani, ACCV 2020). Verified first-hand — I downloaded both.
2. **Donoho & Johnstone 1994 does not contain the `θ²/(θ²+σ²)` shrinkage form.** Biometrika
   81(3):425–455 gives the diagonal-*projection* oracle `R(DP,θ) = Σ min(θ_i², σ²)` (keep-or-kill).
   The shrinkage form `λ/(λ+1)` with `λ` the squared SNR is **Johnstone, *Gaussian Estimation*,
   Eq. (1.4)** — the scalar Wiener filter.
3. **"Residual attention dilutes because residuals bias toward low-pass averaging" is backwards as
   stated.** Wang et al. (ICLR 2022, Prop. 5) and Dong et al. (ICML 2021) both prove that skip
   connections are what *prevents* low-pass / rank collapse; the **attention branch** is the low-pass
   part. The correct framing — which supports the same conclusion more strongly — is: *the skip
   carries the input, noise and all, through untouched, and an additive low-pass branch can only add
   a smoothed copy on top. It never subtracts.* (DnCNN: denoising is `x = y − R(y)`.)

## Gaps — things that appear to be genuinely unpublished

* **The `s* = SNR_v/(1+SNR_v)` optimum for a finite-difference velocity loss.** The shrinkage form
  (Johnstone Eq. 1.4) and the noise amplification of differencing (Chartrand 2011) are both standard;
  their *composition* into "a velocity loss on a noisy estimate has an amplitude optimum far below 1"
  was not found in any paper.
* **The discrete-Laplacian gradient-spectrum argument** (`4 sin²(ω/2)/dt²`) for why an Adam-normalised
  velocity term suppresses the per-frame term's effective step. Not found.
* **A paper that trained a velocity loss end-to-end, observed per-frame degradation, and responded by
  adding a stop-gradient.** Every frozen/two-stage design found (NVDS+, Video Depth Anything, Eigen
  coarse/fine, SmoothNet, DeciWatch, PhysPT) *starts* there. The nearest causal justification is
  Video Depth Anything's *"Introducing temporal attention only in the head prevents the learned
  representation from being corrupted by the limited video data."*
* **Temporal supervision of `d(log z)/dt`.** Robust CVD gives the scale-bias argument for a *spatial*
  ratio residual; Video Depth Anything differences in SSI disparity space without stating why.
  Combining them is unpublished.
* **The explicit "convex-vs-additive attention branch" argument** for why a residual softmax block
  cannot subtract. The supporting pieces (DnCNN, Wang et al. Prop. 5, Park & Kim, SmoothNet's signed
  FIR filter) are all published; the argument is not.

## What the literature would predict for a jointly-trained one-step Lie-velocity loss on an absolute metric root SE(3)

Stated as prediction, not as a published result:

* The velocity term on the **metric translation** channel has no precedent — every published velocity
  loss on joints first removes the root (MotionBERT, GLoT, D&D's joint term, TRAM's unused
  `acceleration_loss`), and every published velocity loss on the *root* is either a regressed channel
  whose **integral** is supervised (WHAM, GVHMR, GLAMR) or an **anchor + derivative pair at equal
  weight after removing the offset** (D&D's `loss_global_transl` + `loss_acc_global_transl`, both at
  `G_TRANSL_W = 100`, both after `- [:, [0], :]`).
* Robust CVD names the exact failure mode for a metric-space depth residual: *"this biases the
  solution toward small depths. Shrinking the whole scene to a point would achieve a minimum."*
* The MPJPE cost is the documented one: SmoothNet's `In=16 ×` row (+1.5 mm) and `In=27 w/ ×` row
  (+1.35 mm), PhysPT's contact-loss row (+0.3 mm and *"sacrifices the reconstruction accuracy"*),
  PMCE's MAED comparison (11 mm for 11.1 mm/s²).
* The published remedies, in descending order of evidence:
  1. **Two-stage / frozen per-frame estimator** — SmoothNet's own ablation shows the identical
     network flips from "+3.5 mm MPJPE" to "better on both metrics" purely by removing the shared
     gradient path. PhysPT, NVDS+, Video Depth Anything, DeciWatch, Eigen all do this.
  2. **Supervise the integral, not the derivative** — WHAM's `[1,3,9,27]` cumulative windows, GVHMR's
     `cumsum` rollout, GLAMR's `EgoToGlobal`.
  3. **Anchor the null space explicitly at equal weight** — D&D.
  4. **Mask where the model can cheat** — GLoT's `m_t`; Video Depth Anything's `|Δg| < 0.05` outlier
     cut.
  5. **Stage the derivative term in** — WHAM's `CAMERA_LOSS_SKIP_EPOCH: 5`, RoHM's `λ_skate: 0 → 0.1`,
     SLAHMR's `joints3d_smooth: [1.0, 10.0, 0.0]`.
  6. **Normalise the channel** — GVHMR standardizes every predicted channel (`(x − mean)/std`) before
     the MSE, so the velocity term is compared in units of its own σ.
  7. **Replace the derivative penalty with a learned prior** — SLAHMR switches `joints3d_smooth` to
     0.0 exactly where HuMoR's `motion_prior` turns on.
  8. **Change the block, not the loss** — TCMR (remove the residual), SmoothNet (signed FIR over the
     window beats self-attention), GLAMR (local LSTM beats a global transformer for a derivative-valued
     output), MEVA (a smooth latent + a detail residual), TRAM/PMCE (no temporal loss at all).
