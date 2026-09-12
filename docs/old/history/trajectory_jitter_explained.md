# Why the trajectory jitters, and what we are doing about it

*Written 2026-09-04, section 7 rewritten 2026-09-05. A plain-language companion to the terse
investigation logs (`jitter_2026-09-04.md`, `camera_ray_2026-09-04.md`,
`temporal_block_2026-09-04.md`, which have every table). This page explains the camera model, the
measurements, the reasoning, and what followed.*

## 1. The symptom

The model recovers the body well on average (MPJPE ≈ 60 mm, on par with the frozen SAM 3D Body),
but when you play the predicted world trajectory back, it shakes. GVHMR's **jitter** metric puts a
number on that: it takes the third finite difference of the world joint positions (position →
velocity → acceleration → jerk), scales by fps³, averages the vector norm over joints, and reports
it in units of 10 m/s³. The kindyn ground truth scores **6.7**. Our runs score **80–110**. The goal
is jitter *close to the ground truth*; anything in the tens is still a shaking body.

Why can MPJPE be fine while jitter is terrible? MPJPE measures how far each frame is from the truth
and is dominated by slow, systematic errors (a limb a bit too long, the body a bit too deep).
Jitter ignores all of that and measures only what changes *from frame to frame*. An estimator can
be right on average and still flicker.

## 2. The camera model (the CLIFF math)

The network never sees the whole video frame. It sees a **square crop** of side $b$ pixels,
centred at $(c_x, c_y)$ in the full image, resized to the model resolution. Inside that crop the
SMPL-X head predicts a **weak-perspective** camera $(s, t_x^{c}, t_y^{c})$: a body point with
camera-frame coordinates $X$ relative to the pelvis appears in crop-normalized coordinates
(spanning $[-1, 1]$ over the crop) at

$$u^{crop} = s\,(X_x + t_x^{c}), \qquad v^{crop} = s\,(X_y + t_y^{c}).$$

The real camera is a **full-perspective** camera with focal $f$ and principal point $(p_x, p_y)$;
a point at depth $Z$ lands at $f X_x / Z + p_x$ pixels. Re-expressed in the crop's normalized
coordinates that is

$$u^{crop} = \frac{2}{b}\left(\frac{f X_x}{Z} + p_x - c_x\right).$$

Weak perspective is what you get when you replace every point's depth $Z$ by one common depth,
the pelvis depth $t_z$. Matching the two expressions term by term (this is CLIFF's contribution:
the crop cannot know its own bearing angle, so the full-image bbox has to be folded back in):

$$t_z = \frac{2 f}{b\, s}, \qquad
  t_x = t_x^{c} + \frac{2\,(c_x - p_x)}{b\, s}, \qquad
  t_y = t_y^{c} + \frac{2\,(c_y - p_y)}{b\, s}.$$

`utils/geometry.py::cliff_cam_to_translation` is exactly this, and `translation_to_cliff_cam`
is its exact inverse (used to build the proxy target for the `cam` loss). The 2D keypoint loss
projects the predicted camera-frame joints with the true intrinsics and then applies the crop
affine — the same chain the crop was made with. I re-derived and checked all of it: **the camera
math has no bug.**

The formula does tell you where to look, though. Taking logs,

$$\log t_z = \log 2f - \log b - \log s
  \;\;\Rightarrow\;\;
  \frac{\delta t_z}{t_z} = -\frac{\delta b}{b} - \frac{\delta s}{s}.$$

Depth is a *ratio*: every percent of error in the crop scale $s$ (or in the box side $b$ if $s$
does not follow it) is a percent of error in depth — and at the corpus's viewing distances a
percent is several centimetres. The lateral position, by contrast, is pinned by the 2D keypoints
and barely moves. Depth is the soft direction of a monocular camera; that is textbook, and it is
exactly what the data shows next.

## 3. Finding the jitter — an oracle decomposition

The lifted world joints are assembled from four predicted pieces:

$$J_w = R_{cw}^{\top}\left(R_{cb}\, L + p - t_{cw}\right)$$

with $p$ the camera-frame pelvis (from the CLIFF lift), $R_{cb}$ the root rotation, $L$ the
root-local joints (pose + shape), and the extrinsics $(R_{cw}, t_{cw})$ from the dataset. I took
the two valid static-camera runs, replaced each predicted piece by its ground truth in turn, and
recomputed the jitter (19 test clips; static baseline shown):

| what is predicted, everything else GT | jitter |
|---|---|
| all four pieces (the actual prediction) | 107.9 |
| only the pelvis **depth** (2D bearing kept) | 90.9 |
| only the root-local pose | 37.9 (torso 13, hands/feet 81) |
| only the root rotation | 33.8 |
| only the lateral pelvis | 21.6 |
| nothing (all GT) | 6.7 |

$\sqrt{91^2 + 38^2 + 34^2 + 22^2} = 107$: the four contributions add in quadrature, so they are
independent noise sources — and **depth alone is worth 91 of the 108**. Fixing depth and nothing
else would take the jitter to 49; fixing everything but depth would leave it at 91.

Two more rows matter. Smoothing *all* predicted components post hoc with a 0.12 s Gaussian gives
jitter **9.2** with WA-MPJPE unchanged (91.7 vs 95.5). So the information for a smooth trajectory
is already in the per-frame predictions; nothing in the network uses it.

## 4. What the depth noise is — and four things it is not

A second dump added the crop box, the predicted $(s, t^c)$ and the **frozen SAM 3D Body camera
head's own** translation, and I ran the model with clips cut to single frames and on training
clips.

| | baseline T=60 | T=1 | train clips | anchor_raw | frozen SAM3D |
|---|---|---|---|---|---|
| jitter | 107.9 | 113.0 | 97.2 | 83.6 | — |
| jitter, GT body + this model's depth | 90.9 | 94.8 | 85.9 | 66.7 | **85.5** |
| depth error, fast part (< 0.25 s) RMS | 4.4 cm | 4.5 | 4.4 | 3.8 | **3.7** |
| depth error, slow part RMS | 21 cm | 22 | 10.5 | 20 | 19 |
| slope of Δlog s against Δlog b | −0.96 | −0.97 | −0.95 | −1.01 | — |

* **Not the bounding box.** The per-frame SAM3 box side jitters by 3.3 % from frame to frame. If
  the head ignored that, depth would inherit it one for one. It does not: the predicted $s$ moves
  against $b$ with slope −0.96 (correlation −0.98), so $b \cdot s$ — the quantity that sets depth
  — is insensitive to the crop. Smoothing the boxes would not help.
* **Not our head, not CLIFF.** The frozen SAM 3D Body camera head, trained by Meta on far more
  data and read from the very same token, has the same fast depth noise (3.7 cm) and the same
  depth-only jitter (85). That noise is the per-frame **depth ambiguity of the token itself**; the
  anchor run already sits on that floor (3.8 cm).
* **Not overfitting.** Training clips show the same fast noise as test clips (4.4 cm). Only the
  *slow* depth error is memorised (10.5 cm on train vs 21 cm on test; the frozen model has 19 cm
  on both).
* **Not the temporal block doing a bad job — it does no job.** Feeding the frames one at a time
  (T = 1) changes the jitter by 5 % (113 vs 108) and the fast depth noise not at all. The RoPE
  block, as found before, is a per-frame fix (it learned a near-uniform per-clip average that helps
  contact precision), not a temporal filter.

So: the token gives us a per-frame depth that is right to about a percent and wrong by a fresh
few centimetres every frame. **Nothing per frame will fix that; only integration over time can.**

## 5. Why training never learned to integrate over time

Two reasons, one on the loss side and one on the architecture side.

**The losses barely ask for it.** A per-frame loss rewards denoising only in proportion to the
noise's share of the error variance. For depth the fast part is 4.4 cm against a slow part of
21 cm: about **4 %** of the depth-error variance. The pelvis-relative 3D term subtracts the pelvis
(it is blind to depth altogether — it does not "carry the pelvis error", it removes it); depth is
held only by the tiny `cam` Huber on $s$ and, since the anchor arm, by the pelvis anchor. A
post-hoc sweep makes the point concrete: smoothing the predictions with a Gaussian of any width
changes the depth RMSE by 1 %, and MPJPE has a shallow optimum at σ ≈ 0.05–0.08 s worth 0.5 mm —
yet that same σ takes the jitter from 108 to about 10. The jitter is 100 % fast component; the
losses are 96 % slow component. They are looking at different things.

**The architecture makes it expensive.** To smooth the pose token the residual RoPE block would
have to output $\Delta_t \approx \text{mean}_{\text{local}}(x) - x_t$, i.e. cancel its own input
through a 1024×1024 value map learned from zero-initialised gates. It never did. When we pushed
harder through that path (motion matching at 3× weight) the optimizer found a cheaper way to
lower the derivative losses — shrink the per-frame motion — and the pose collapsed. Motion
matching at a sane weight plus the pelvis anchor did move the right way (fast depth noise 4.4 →
3.8 cm, jitter 108 → 84): the loss side works, the model has no cheap lever to pull.

## 6. The pelvis anchor — the piece that already helped

Under a static camera a derivative loss (velocity matching) cannot see a constant depth offset:
`static_matching` drifted the whole body 22 cm too close and the per-frame terms could not pull it
back. The **pelvis anchor** is a Huber loss (δ = 0.1 m, weight 3) on the absolute camera-frame
pelvis position in metres. It closed that null space (depth error 224 → 145 mm, bias +38 → +25 mm
vs the baseline) and let matching keep its smoothing gains (jitter 108 → 84, RTE 5.6 → 4.75) for
2.8 mm of MPJPE. It anchors the *slow* depth. It cannot remove the *fast* noise — nothing per frame
can — which is why it stopped at 84.

## 7. What was tried after this, and where it stands (2026-09-05)

*The original sections 7-11 described a learned convex output smoother and its results. That
approach is explicit smoothing — the user's call is that the model has to learn it — and its code
was removed on 2026-09-05; the numbers survive in `jitter_2026-09-04.md` §4-12.*

1. **Output smoother (rejected).** A learned Gaussian-in-time kernel on the head's per-frame
   outputs, pulled open by velocity / acceleration matching, reached jitter 9.7 (GT 6.7) with the
   per-frame pose unchanged — but only once it acted on the *lifted* pelvis in metres rather than
   on the crop proxy $s$ (averaging $s$ re-injects the 3 % per-frame bbox jitter, section 4).
   That lesson carried over: the camera must be parametrised in a bbox-independent space.
2. **Ray camera head (kept).** The head now regresses the pelvis *ray* — bearing $(x/z, y/z)$
   and log depth — as a residual on the frozen SAM 3D Body's own per-frame pelvis, with depth and
   bearing velocity / acceleration matched to the GT ray. Depth is no longer a jitter source
   (depth-only share 110 → 11-15) and the absolute depth stays at the frozen model's
   (`camera_ray_2026-09-04.md`).
3. **What remains is per-frame noise of the pose readout itself**: root rotation (~34) and joint
   articulation (~32-42) — the same white token noise, created in the backbone feature maps, not
   in the bbox track, the mask prompt or the reconstructed camera. Averaging the *pose token* over
   ~0.1 s takes the frozen model from 126 to 14 with no training, so the temporal block could do
   it in principle.
4. **The temporal block (open).** Its zero-gate parametrisation never learned to average; with unit
   gates and zero output projections (`gate_init: zero_proj`) it is worth ~5 jitter points (52 vs
   57) and converges, but every additive variant settles on *diluting* the token with its
   neighbours' average rather than replacing it (floor ~50). A learnable locality prior, a hard
   0.25 s window, a frozen-head second stage and a first "replace" path were tried and removed;
   `temporal_block_2026-09-04.md` has the diagnosis and the one untested idea (a same-token convex
   replacement gate).

Current best without explicit smoothing, 16-scene static test set: jitter 50-52 (`tb_projzero`),
against 118 for the CLIFF baseline and 6.35 for the GT.
