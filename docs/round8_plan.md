# Round 8 plan (2026-09-12): gravity, contacts and forces on the smooth body

Start point: `configs/smooth/P_k30.yaml`, run `output_3/P_k30_20260912_011321/best.pth`
(54.86 mm / jitter 5.87 / pelvis 110; `docs/architecture.md`). Contact and force heads were OFF
in round 7. This round puts them back and removes the ground-truth gravity input.

## 0. Facts gathered before designing

**Model size.** 10.47 M trainable parameters: the three RoPE layers 9.46 M (dim 512, 8 heads,
MLP ×4 → 3.15 M per layer), pose-token projection 0.26 M, input projection 0.31 M, feedback
projection 0.10 M, heads 0.33 M. The frozen part is 1.29 B (SAM 3D Body + stage 1).

**Layers.** Round 7 arm M (4 layers) scored 54.90 / 6.64 against K's 54.86 / 5.9 (3 layers):
no gain. The receptive field is already 3 × 0.5 s = ±1.5 s of a 2.4 s clip, so a fourth layer
adds capacity, not context, and the model already overfits in the tail (next point).

**Overfitting.** P_k30, 458 steps × 8 clips per epoch (3.7 k clips from 864 scenes):

| epoch | train loss | test loss | gap | MPJPE | jitter |
|---|---|---|---|---|---|
| 0 | 0.891 | 0.899 | 0.01 | 55.45 | 17.4 |
| 5 | 0.649 | 0.815 | 0.17 | 54.94 | 7.0 |
| 15 | 0.587 | 0.795 | 0.21 | 54.86 | 5.87 (best) |
| 20 | 0.563 | 0.791 | 0.23 | 54.84 | 6.06 |
| 29 | 0.531 | 0.790 | 0.26 | 54.82 | 6.66 |

Train loss keeps falling, test loss is flat from epoch 20, jitter drifts up while MPJPE gains
hundredths: mild overfitting in the cosine tail. Regularisation in place: dropout 0.1 inside the
block (attention and FFN), weight decay 0.01, EMA 0.999, per-epoch clip-window jitter. There is
NO input noise and NO frame / token masking (the masking inputs were removed in the 2026-09-05
simplification; the earlier span-masking arm was on image tokens and showed no gain).

**Where the "GT gravity" comes from** (`features/geocalib/<scene>/gravity.npz`, copied into
`kindyn_1.npz`; the world frame IS the first camera: `extrinsics[0]` = identity on every scene):

| source | test | train | meaning |
|---|---|---|---|
| `ground` | 49 | 365 | normal of a fitted ground plane |
| `geocalib` | 16 | 171 | pooled per-frame GeoCalib estimates, accepted as reliable |
| `fallback_down` | 43 | 328 | **world +y = the first camera's down axis; no measurement** |

So 40 % of the gravity targets are a placeholder. Cues available to a model that may not see
the world frame (frame-independence rule):

| estimate of the down vector | measured test scenes (65) | fallback test scenes (43) |
|---|---|---|
| camera +y axis (OpenCV down), per frame | median 12.2°, p90 30.5° | median 3.2°, p90 8.6° (it IS the target up to camera motion) |
| minus the GT body's own up axis, clip mean | median 36°, p90 64° | — |
| raw per-frame GeoCalib pooled | median 20° (test) / 8° (train) vs the fitted value | — |

Climbers lean and hang (the pelvis axis is a poor gravity cue), while cameras filming climbers
are tilted up by 12° on the median. The camera's orientation relative to the body is not in the
token today (`camera_context` carries the direction TO the camera, not its axes).

## 1. Gravity: predicted, not fed

**Change 1 — drop the GT channel.** `token.gravity: false`. Arm G0 measures what the pose loses
(expected: nothing; round 5 showed the channel helped forces, not pose).

**Change 2 — give the model the camera's orientation.** Add the camera's down (+y) and viewing
(+z) axes expressed in the body frame to `camera_context` (6 channels). This is
`R_body_cam`, which stage 1 already outputs by predicting the body in the camera frame; it is
frame-independent and is the 12°-median cue above.

**Change 3 — a gravity head, iterative like the pose.** After every layer a zero-init head reads
the frame's hidden state and outputs a 3-vector correction in the body frame:

    g_b(t) = normalize( d_cam_b(t) + head(hidden_t) )      # d_cam_b = camera down axis in the body frame
    g_w    = normalize( mean_t rot_wr(t) g_b(t) )           # one vector per clip (gravity is constant);
                                                            # a (near-)cancelling mean falls back to the pooled camera axis
    g_b'(t) = rot_wr(t)^T g_w                               # re-expressed per frame, fed to the next layer

At init the estimate is the camera's down axis (3° / 12°), and the head learns the correction
from the pose and motion. The world is used only as transport between frames (a rotation of the
world rotates both sides), so the test `test_world_frame_independence` still applies. The
per-frame `g_b'` joins the feedback channels of the next layer, and the last layer's `g_w` is
the model's gravity output (`out["gravity"]`).

**Loss / metric.** `1 − g_w · gravity_world` per clip, deep-supervised (`layer_weight`), on
scenes with a MEASURED gravity only (`ground` / `geocalib`; the loader reads the source from
`geocalib/gravity.npz` into a per-frame `gravity_measured` flag). Metrics `metric_gravity/angle_measured` /
`angle_fallback` (mean degrees; the additive-stats interface carries no medians — compute them
offline from a prediction dump if needed) plus `prior_angle_*` for the camera axis alone. Bars: camera axis
12.2° median on measured scenes; a head that does not beat it has learned nothing.

**Downstream.** The predicted gravity (detached) is what the RNEA residual feedback (section 3)
uses, so inference needs no kindyn gravity. The `force_consistency` LOSS keeps the corpus
gravity (it is supervision). Not chosen: GVHMR's gravity-view frame. It solves a problem we do
not have (no world lift) and would not fix the target quality; the pooled per-clip head is the
minimal form of the same idea (the body's orientation relative to gravity is the output).

## 2. Contacts

Contact head on the per-frame hidden state, one token per frame (as round 5 / 6), run after
EVERY layer (deep-supervised BCE at `layer_weight`) because the RNEA feedback of section 3
needs a contact gate at each layer; the six probabilities are fed back to the next layer (6
channels). Loss: confidence-weighted BCE, no class weights (the round-6 recipe).

**Contact consistency**, on with the forward stencil: the extremity's world speed is
`|x(t+1) − x(t)| / dt` on rows where both frames are valid (the current central difference is
Nyquist-blind, the round-7 rule), L1 × label × confidence, gradient into the pose path only.
No acceleration term: the target is stillness, and zero velocity over a contact run already
implies zero acceleration; an acceleration term would only act at contact onsets / offsets
(where the label is least reliable). GT floor 0.13 m/s (the labels are motion-gated, so the
target is stricter than the GT itself; MPJPE tells whether it fights the position losses).
Weight: start at 1.0 (speed in m/s, so a 0.1 m/s excess costs 0.1 — comparable to the
velocity term); one arm at 3.0 if the speed metric does not move.

Bars (different body, so informal): final s42 / s1 on the Gaussian body F1 0.927 / 0.926,
precision 0.924, P@R90 0.938; static-input contact 0.888.

## 3. Forces, with an iterative physics feedback

Force head per layer (single token, 18 outputs = six body-frame forces in body weights),
supervised as round 5 (vector Huber on in-contact rows + noncontact L1, kindyn confidence),
deep-supervised. Bars: MAE 0.183–0.185 bw, angle 19.9–20.7°, off-contact 0.047, RNEA residual
0.154 bw / 0.054 bw·m (the round-5 token-channel arms).

**The feedback (decided 2026-09-12): the RNEA residual itself.** After layer k the refiner
has a body trajectory (pose after k deltas), six forces `f_k`, contact probabilities `c_k` and a
gravity `g_k`. Inverse dynamics with the predicted forces applied as the external forces gives
the unbalanced root wrench:

    r_k = RNEA(q_k, v_k, a_k, gravity = g_k, f_ext = c_k ⊙ f_k)      # 6 per frame, root frame: force (bw), torque about the pelvis (bw·m)

This is the existing `force_consistency.residual` function reused (one RNEA call per layer,
no new physics code). The six numbers go through a LayerNorm and a zero-init linear into the
residual stream before layer k + 1, next to the rate feedback and the contact / gravity
feedback. Layer k + 1 sees what is still missing and corrects the allocation — the force head
becomes an iterative allocator the way the pose head became an iterative smoother. No extra
channels (the required wrench and the current forces add nothing the next layer does not
already hold in its hidden state).

Gradient split. The BODY entering the RNEA is detached: the residual depends on the body's
acceleration, and a live channel would push the pose through the second-difference gradient
the round-7 losses avoid; the pose has its own losses. The FORCES stay live (linear, clean
gradient; the round-7 rate feedback is live too), so layer 3's force loss can improve layer
1's first guess. The contact gate is the detached probability, as in the RNEA loss. The
gravity inside the RNEA is the layer's predicted, pooled vector (detached), so inference
needs no corpus gravity.

Notes. (1) `a_k` is the loss's ±2-frame stencil `(x[t+2] − 2 x[t] + x[t−2]) / (2 dt)²`, which is
BLIND to a period-2 wobble (codex review, 2026-09-12: a ±1 cm alternation at 25 fps leaves the
residual exactly at the static value, though a three-point stencil would read 2.5 bw of
"inertia"). Kept on purpose: the stage-1 jitter lives at Nyquist and is not motion, so a
Nyquist-visible acceleration would feed fake inertial forces into layer 1's channel; the pose's
own velocity losses handle the wobble. (2) After layer 1 the body is still partly jittery, so `r_1` is noisy;
the network can discount it, and a variant feeds the residual from layer 2 only (F1b) if F1 is
worse than F0. (3) Cost: one RNEA per layer per step (the loss already does one); measure,
expect < 10 % of a step.

**RNEA loss** (`force_consistency`) stays a separate factor: round 4 found it buys −1° angle for
2× off-contact force; with the residual feedback the network can satisfy the balance without the
loss, so test both.

## 4. Regularisation (in scope, one arm)

Given the train/test gap, one cheap arm: random FRAME masking of the input token (p = 0.1,
the frame's geometry replaced by a learned embedding; feedback channels still computed, the
losses unchanged). GVHMR trains this way. Run after the ladder above on its best arm; never combined with a
new head in the same arm.

## 5. Experiment ladder

All arms on the P_k30 recipe (30 epochs, cached pose tokens, 8 clips per GPU, one GPU each on
GPUs 5 / 7, ~45 min per arm before the RNEA cost), `best.pth` by jitter for pose-only arms and
by contact F1 once contact is on; every arm evaluated, dumped and paired against P_k30
(`scripts/paired_ci.py`) for MPJPE / jitter, plus the contact / force / gravity metrics.

| arm | change to the previous | question |
|---|---|---|
| G0 | `token.gravity: false` | does the pose need the GT gravity? |
| G1 | + camera axes + gravity head (measured-scene loss) | gravity angle vs the 12° camera bar; pose unchanged? |
| C1 | + contact head (per layer, deep BCE) | F1 on the smooth body vs 0.927 |
| C2 | + contact consistency (forward stencil, w 1.0) | in-contact speed ↓, F1, MPJPE |
| F0 | + force head (supervised only) | MAE / angle vs 0.185 / 20.5° |
| F1 | + RNEA residual feedback | does the physics channel improve allocation (MAE, angle, torque residual)? |
| F2 | F0 + RNEA loss, F3 = F1 + RNEA loss | the 2 × 2 of feedback × loss |
| R1 | best of the above + frame masking | does the train/test gap close? |

Kill rules: MPJPE +0.3 mm or jitter +1 over P_k30 at epoch 10 ends a pose-touching arm;
contact F1 < 0.90 at epoch 10 ends a contact arm. Seed replicate of the final recipe. Noise
rule: 0.005 F1 / 0.1 mm / 0.5° force angle / 0.0005 bw MAE are seed spread.

## 6. Build list (before any launch)

1. Loader: `gravity_measured` flag from `geocalib/gravity.npz` (`source != fallback_down`);
   camera axes in `camera_context`; schema keys `model.refiner.token.gravity` (stays, default
   off), `gravity_supervision` section (weight, measured_only), `contact_consistency.stencil`.
2. Refiner: per-layer contact / force / gravity heads with their feedback channels, the
   gravity pooling, the residual feedback (`model/refiner.py`, reusing `force_consistency.residual`); `out["gravity"]`, `*_layers` for every head.
3. Losses: `gravity` (cosine, measured rows, deep), `contact_consistency` forward stencil,
   deep supervision for contact / force.
4. Tests: frame independence with every head on; identity at init (gravity = camera down
   axis); the feedback residual equals the loss's residual on the same body; forward-stencil rows; gradient reaches each head.
5. Codex read-only review of the residual feedback and the gravity pooling before G1 runs.

## 7. Decisions (user, 2026-09-12)

Gravity supervision on measured scenes only. Feedback = the RNEA residual (6), body detached,
forces live. Contact consistency gated by the GT labels first. Frame masking (R1) in scope.
