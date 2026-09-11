# Proposed joint model: every block, channel and loss (2026-09-07)

*Specification of the model `plan.md` trains. Each block carries a status: **exists** (in the
code today, unchanged), **modify** (exists, changes named here), **new**. Config keys are
proposals; `configs/base.yaml` is the schema once they are built. Companion of `refiner.md`,
which documents the current two-stage refiner this replaces.*

```
                 ┌───────────────── cached, frozen SAM 3D Body ─────────────────┐
frame ──────────►│ pose token (1024)      anchor features 6 × {mean, centre} × 1280│
                 └────────┬───────────────────────────────┬─────────────────────┘
                          ▼                               │
                  SMPL-X + camera head  (trainable)       │
                          │ per-frame body: root, 21 (+30) rotations, betas, pelvis ray
                          ▼                               │
                  world lift (cam_from_world)             │
                          │  ─ ─ ─ stop-gradient (detach_input) ─ ─ ─
                          ▼                               ▼
                  observation token  ◄────────────────────┘   (+ aug: token dropout, frame mask)
                          ▼
                  RoPE temporal block  (2 layers, dim 128, ±0.15 s per layer)
                          ▼
     ┌────────────────────┼──────────────────────────────────────────┐
     ▼                    ▼                                          ▼
convex-kernel smoother   per-frame heads                         causal head
(root pos, root rot,     pose delta · contact · loaded support   transition forecast
 21 joint rots)          force · motion                          (past-only attention)
     ▼
refined world body ─► FK ─► every camera (joints_cam, kp2d) ─► losses / metrics
```

## 1. Inputs and caches

| item | shape / unit | status | notes |
|---|---|---|---|
| pose token | bf16 (1024) per person-frame | exists (`data.pose_token_cache`) | final decoder pose token of the frozen base |
| anchor features | bf16 (6, 2, 1280) | **new** | DINOv3 backbone features grid-sampled (5×5, `grid_radius 0.1`) at the frozen decoder's interm 2D keypoints of the six MHR anchors `[62,41,15,18,17,20]`, pooled to {mean, centre} before storage (30 KB/frame). Built by `precompute_pose_tokens.py`. No learned decoder token exists, so the backbone / decoder never run in training |
| geometry | `bbox`, `cam_int`, `cam_from_world`, `frame_pos_sec`, `frame_valid` | exists | `data.fixed_camera` replaces `cam_from_world` by the per-scene chordal mean (static track) |
| labels | contact (6), forces (6, 3) bw root frame + contact-frame positions, SMPL-X GT, `gravity_world`, `total_mass` | modify | `gravity_world` and `total_mass` loaded for EVERY run (today only with the force group); the force loader keeps the 12 contact-frame world positions for the lever wrenches |

## 2. Per-frame body head — exists (`model/heads.py::SmplxHead`)

Two FFNs of the decoder's shape on the pose token. Outputs: root + 21 body 6D rotations
(+ 30 finger joints with `hands`), 10 betas, pelvis ray `(x/z, y/z, log z)` (`camera: ray`).
FK and full-frame projection inside the head. Changes: `model.smplx.frozen: false` under the
refiner; trained by the per-frame losses only when `detach_input: true`; warm start from the old
stage 1 allowed as an arm.

## 3. World lift — exists (`model/refiner.py`)

`p_w = R^T (p_c − t)` with `cam_from_world`; root rotation likewise. **No smoothing here any
more** (`root_smooth_sec` / `pose_smooth_sec` retired; the smoother block is the only smoothing
and lives at the output). Betas: causal running mean over the clip (**modify**: the clip mean is
a sequence-wide feature that leaks the future into every frame).

## 4. Observation token — modify

Everything is root- or body-frame; nothing refers to the world frame (the frame-independence
test stays). One fixed per-channel standardisation (train-only statistics of the actual input
distribution, recomputed once after the per-frame warm-up) replaces `geometry_norm`.

| channel | dim | status | why |
|---|---|---|---|
| root-frame joint positions (FK of the raw per-frame pose) | 22 × 3 | exists | the body |
| parent-local 6D rotations of the 21 joints | 21 × 6 | **new** | the head corrects rotations; positions do not expose terminal-joint orientation |
| root rotation relative to the gravity frame, 6D | 6 | **new** | "which way is up", frame-independent |
| gravity direction in the body frame | 3 | **new** | needed to read support forces in the body frame |
| body-frame root linear / angular velocity (adjacent increments, causal) | 6 | modify | today central; causal so the forecasting head can share the token |
| `raw − 5-frame-mean` of the root position and of the joint positions | 3 + 22 × 3 | **new** | makes noisiness visible to the smoother (removed in the per-frame twin) |
| frame spacing, standardised | 1 | exists | |
| betas (causal running mean) | 10 | modify | |
| camera context: pelvis→camera direction in the body frame, log depth, box bearing, angular size | 7 | exists | per-frame channel (not constant even under a fixed camera) |
| projected pose token | 1024 → 64 | modify (64, was 256) | appearance |
| projected anchor features | 6 × 2 × (1280 → 16) | **new** | appearance at the extremities |
| "missing" embedding (token dropout) / mask embedding (frame mask) | 128 | **new** | see §9 |

## 5. Temporal block — exists (`model/rope.py::CrossModalRopeModule`), modify size

Pre-LN residual blocks with zero-init output projections (identity at init), RoPE position =
elapsed seconds × `time_scale`, hard window ±0.15 s per layer, `frame_valid` masking, one slot.
Size `dim 128, num_layers 2, num_heads 4, mlp_ratio 2, dropout 0`. Receptive field 0.6 s inside
a 60-frame clip. A second attention mask (past-only) serves the causal head (§8).

## 6. Convex-kernel smoother block — new (`model/refiner.py`)

The operator class a Gaussian belongs to, applied to the RAW per-frame world body at the
output. For each frame `t` and each smoothed quantity `x` (root position; root rotation; each of
the 21 parent-local rotations), weights `w_{t,s} ≥ 0`, `Σ_s w_{t,s} = 1` over the support
`|s − t| ≤ K` (K from `support_sec` 0.24 s at the clip's spacing):

$$
\hat p_t = \sum_s w_{t,s}\, p_s, \qquad
\hat R_t = \operatorname{proj}_{SO(3)}\Big(\sum_s w_{t,s}\, R_s\Big)
$$

(`project_rotation`, the existing scaled-Newton polar projection). Then the residual heads
(§7) act on the smoothed body. Modes (`model.refiner.smoother.kind`):

| mode | weights | parameters | tests |
|---|---|---|---|
| `fixed` | Gaussian of `sigma_sec` | 0 | the old recipe inside the new model; σ sweep = the Pareto front |
| `global` | one learned logit vector per quantity group (root position, root rotation, joint rotations), softmax | 3 × (2K+1) | is the fixed width the right width |
| `adaptive` | logits per frame and quantity emitted by the temporal block, softmax; initialised (zero-init head + bias) to the `fixed` Gaussian | head 128 → 23 × (2K+1) | content dependence |

Properties to log per epoch: effective width `Σ_s w s²`, centre of mass `Σ_s w s` (lag; must
stay at 0 ± 1 frame), width vs joint speed and vs contact state. A convex kernel passes a
common translation bias unchanged (`Σ w (p + b) = Σ w p + b`), so any accuracy gain over the
fixed front comes from the residual heads and is reported separately. Support rows outside a
valid run are renormalised over the valid neighbours; masked observations (§9) are removed
from the support. The per-frame twin sets `support_sec 0`.

## 7. Output heads (zero-init, on the block's final tokens)

| head | output | frame | status | loss |
|---|---|---|---|---|
| pose delta | 6D right-multiplied on root + 21 joints, body-frame root shift | body | exists | §10 pose terms on the refined body |
| contact | 6 logits | – | exists | BCE, heel weight 1 |
| loaded support | 6 logits, target `|f_gt| ≥ 0.05 bw` | – | **new** | BCE; the force-derived readout `sigmoid(k(|f| − θ))` is scored against this target |
| force | 6 × 3, bw, body frame | body | exists | Huber on in-contact rows + noncontact L1; RNEA |
| motion | body-frame vel / acc of 22 joints, root ang vel / acc | body | exists | Huber vs `label_smooth_sec` targets; a diagnostic head only, never consumed by RNEA |
| transition forecast | 6 × 2 logits: release / grab within 0.4 s | – | **new**, causal (§8) | BCE vs label transitions |
| smoother logits (`adaptive`) | 23 × (2K+1) | – | **new** | none direct (through the pose terms) |

Refined body: FK in the world with the running-mean betas, mapped into every camera; the
output keeps the `SmplxHead` layout plus its own camera readout so the per-frame `cam` / ray
terms apply to both outputs.

## 8. Causal head — new

The transition-forecast head reads the temporal block under a past-only attention mask, with
the causal velocity channels and the causal betas (§4) and WITHOUT the smoother's future
neighbours. Baselines it must beat: persistence, a contact-duration hazard model, a per-frame
pose + velocity MLP. It exists to make C5 identifiable, not to improve any other output.

## 9. Augmentations — new (`aug.*`, dataset / collate, cached path)

| key | default | mechanism |
|---|---|---|
| `aug.token_dropout` | per-frame 0.3, whole-clip 0.1 | pose token + anchor features replaced by the "missing" embedding at the refiner input; the per-frame head keeps its token |
| `aug.frame_mask` | 0.2, spans 2–6 frames | the frame's OBSERVATION is removed from every input path (token, residual base pose, smoother support, neighbours' velocity channels) and replaced by the mask embedding; TARGET validity untouched; RoPE position and `frame_valid` kept |
| `aug.synth_noise` | off | per-clip bias + AR(1) corruption of GT clips from out-of-sample residual statistics; mixed with token-dropped real clips so missingness does not identify them; low priority |

## 10. Losses (all on the `Loss` interface of `model/loss/`)

Targets: position, velocity and acceleration of the GT from ONE trajectory (raw, or one
Gaussian on the positions, `smplx_supervision.target_smooth_sec`), stencils aligned:
velocity = adjacent increment at the interval midpoint, acceleration = centred second
difference at the frame. Term weights set from measured gradient norms on the pose output.

| loss | on | term | target / stencil | gate | status |
|---|---|---|---|---|---|
| `smplx_supervision` | per-frame body AND refined body | kp2d, kp3d (pelvis-relative), orient / pose 6D, betas, ray anchors `depth` / `bearing` | GT, per frame | `smplx_valid` | exists; applied to both outputs (**modify**) |
| | refined body | `root_bias` (clip-mean world root error), `root_shape` (per-frame deviation from it, Huber) | GT world root | | exists (not a derivative loss) |
| | refined body | `pose_vel`, `pose_acc` | aligned adjacent / centred differences of the consistent target trajectory | stencil inside a valid run | **modify** (was central vs double-smoothed targets) |
| `contact_supervision` | contact head | confidence-weighted BCE | manual / auto labels | `contact_valid` | exists |
| `support_supervision` | loaded-support head | BCE | `|f_gt| ≥ 0.05 bw` | `force_valid` | **new** |
| `force_supervision` | force head | vector Huber on in-contact rows, noncontact L1, net force / torque | kindyn forces, bw, root frame | `force_confidence^p` | exists |
| `motion_supervision` | motion head | Huber on vel / acc / ang_vel / ang_acc ÷ scale | GT finite differences smoothed ONCE (`label_smooth_sec`), accel from the same smoothed trajectory | support masks | **modify** (no double smoothing) |
| `transition_supervision` | causal head | BCE | label transitions within 0.4 s | valid horizon | **new** |
| `contact_consistency` | refined extremities | L1 world speed on in-contact frames | 0 | GT contact over the WHOLE stencil (not the centre frame only) | **modify**; OFF on the pose path in Phase 1 |
| `force_consistency` | refined motion + forces | RNEA root-wrench residual, pseudo-Huber force (bw) + torque (bw·m) | 0 | detached predicted contact, soft or hard (`gate: soft|hard`) | **modify**, producer parity below |
| `static_balance` | forces + gravity | `Σ f + g` residual (bw) | 0 | same gate | **new** (Phase 3 control) |

**RNEA producer parity** (`force_consistency`): body `bh.SMPLX(use_hands=False, compute_mass=True,
density="Dempster")` (BetterHuman `body_densities.json`: chest 920, pelvis / feet 1010, head
1110, hands 1160 kg/m³; the corpus GT of 2026-09-06 is solved with this table, BVR commit
`c3a0627`); force scale `total_mass · g` from `kindyn_1.npz`, not the model's own mass;
velocity / acceleration by the producer's 3-point manifold stencil at spacing `round(fps/30)`
(`(Δ_fwd[t+h] − Δ_fwd[t]) / (h·dt)²`, `tools/smplx_robot/dynamics.py`), not the composed ±2
central stencil; external forces as wrenches `[f, r × f]` at the contact-frame positions on the
parent joint (`tools/human_optim/kindyn.py::build_fext`), not pure forces at the joint origin;
the loader's dropped groups (~4 % of the force) restored or documented; `detach_pose` detaches
BEFORE the force rotation. The producer's own residual (`base_wrench`, ≈ 0.15 bw mean) is the
floor. Note the producer also carries strength-weighted joint-torque effort terms (commit
`845ef17`: legs / trunk 1, shoulder 2.5, elbow 3.3, wrist 17, fingers 25) that resolve the
statically indeterminate hands-vs-feet split; the learner's root-wrench residual alone leaves
that nullspace (6 equations, 18 unknowns) — it is supervised by the force labels, not by RNEA.

## 11. Metrics (`scripts/evaluate.py`, frozen protocol)

Whole valid runs tiled with overlapping context, each frame scored once; per-clip dumps;
`scripts/paired_ci.py` bootstraps by source video. Pose: MPJPE / PA / PVE per-frame and refined,
referenced accel (hips-aligned, exists) AND world-root / CoM referenced accel (**new**), absolute
world-root error and fixed-scale alignment next to the similarity-aligned GVHMR globals,
`jitter` and `jitter − gt_jitter` (diagnostic), per-joint speed ratio / phase by band and
contact state (shrinkage monitor). Contact: micro / per-group F1, P@R90, onset / offset timing,
boundary F1 ±2 frames, occupancy vs loaded support. Force: MAE, angle, `corrIC`, off-contact
|f|, RNEA residual vs the producer floor. Static and moving cameras separately; `_frame` twins
of every pose metric.

## 12. Config keys introduced

```
model.smplx.frozen: false
model.refiner.detach_input: true            # complete stop-gradient boundary
model.refiner.smoother: {kind: fixed|global|adaptive, support_sec: 0.24, sigma_sec: 0.08}
model.refiner.outputs: [pose, contact, support, force, motion, transition]
model.refiner.anchor_features: true          # cached 6 × 2 × 1280, projected to 16 each
model.refiner.token: {local_rotations: true, gravity: true, raw_minus_mean: true}
data.fixed_camera: false / data.fixed_camera_max_mm: 15
aug.token_dropout: {frame: 0.3, clip: 0.1} / aug.frame_mask: {p: 0.2, span: [2, 6]} / aug.synth_noise: off
optim.temporal_start_step / optim.temporal_ramp_steps
smplx_supervision.target_smooth_sec: 0
force_consistency: {density: Dempster, stencil: producer, levers: true, gate: soft|hard}
```
