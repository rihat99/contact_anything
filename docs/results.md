# Results, in easy words (2026-09-12)

What the current model does, how well, and which of the things we tried worked and which did not.
The full numbers are in `docs/round8_2026-09-12.md` (heads) and `docs/old/round7_2026-09-11.md`
(smoothness); every earlier number is in `docs/old/results.md`. The model itself is explained in
`docs/architecture.md`.

## Where we are

One model (`configs/r8/F2m_scale03.yaml`) takes a climbing video with known camera poses and gives,
per frame: a smooth SMPL-X body in the world, which of the six extremities touch the wall, the
force on each of them, and the direction of gravity. On the test videos (107 clips, 120 frames each
at most):

| what | number | what it means |
|---|---|---|
| joint error | 53.98 mm | same as the pose-only body (53.88); the frozen SAM 3D Body was 61 |
| jitter | 5.5 | the real motion scores 6.9; the frozen model 105 |
| contact F1 | 0.927 | precision 0.936, recall 0.918; hands 0.98, toes 0.90, heels 0 |
| force error | 0.179 body weights, 20.6° | the previous best was 0.185 / 20° on a jittery body |
| gravity | 12.7° from the corpus value | the camera's own axis alone is 13.7° off; the corpus value itself is uncertain by 10–20° |

Two training seeds agree to 0.005 mm, 0.001 F1, 0.001 body weights and 0.2°, so differences smaller
than that between arms are noise.

## What worked

**Letting the temporal block smooth the body itself (round 7).** The old fixed Gaussian filters
over-smoothed (jitter 2.7 against a real 6.9) and blurred real motion. Three changes made the
transformer do the job: one-sided (forward / backward) differences everywhere instead of central
ones, which are blind to the frame-to-frame wobble that dominates jitter; per-joint rotation rates
as inputs, so a layer can compute "neighbours minus me"; and a velocity-only loss against the raw
ground truth. Plus the iterative design: every layer applies its correction and the corrected body's
own rates go into the next layer. Result: 54.9 mm instead of 55.5, jitter 5.9 instead of 2.7 (i.e.
the real motion's level), better in-band acceleration (0.77 vs 0.70).

**Contacts on the smooth body.** The contact head reaches F1 0.93 within five epochs, above the
0.927 the previous (Gaussian) body reached, with higher precision.

**Contact stillness as a loss.** Pushing the in-contact extremities' speed towards zero (with a
forward difference) brings them to the ground truth's own level (0.17–0.19 m/s) and removes most of
the jitter the contact head had added. It costs about 0.2 mm on a fully shared trunk and nothing at
the chosen coupling.

**The RNEA (inverse dynamics) residual as a loss.** Asking the predicted forces to balance the
refined motion under gravity is the one thing that improved forces: angle error 22.3° → 20.3°,
error 0.185 → 0.179 body weights, for a small rise of the force on limbs that are not in contact.

**Scaling the head gradients (`head_grad_scale 0.3`).** The contact, force and gravity heads share
the trunk with the pose, and training them at full strength reshaped it: the body jittered 3.5 units
more with the contact head and 1 more with the gravity head. Passing only 30 % of their gradient
into the trunk keeps the forces of the shared version and the smoothness of the pose-only one.

**Predicting gravity instead of reading it.** The corpus gravity channel turned out to be unnecessary
for the pose (removing it changed nothing), and a small head on top of the camera's down axis gives
a gravity that is 1° better than the axis alone, so inference needs no external gravity any more.

## What did not work

**Any acceleration loss for smoothness (round 7).** Pointwise, RMS-matched, smoothed-target or
band-limited: none helped, and the pointwise one shrinks the motion. The pose's velocity loss is
what produces smoothness; in-band acceleration correlation stays at 0.77 whatever we do, which
looks like a ceiling of the whole pipeline.

**A trainable stage-1 head, a fourth layer, delta feedback, a Gaussian reference (round 7).**
None moved the pose beyond seed noise; the trainable per-frame head made the pelvis worse.

**Fully detached heads.** With the heads reading a stop-gradient of the trunk, the body keeps its
smoothness but contact plateaus at F1 0.877: the trunk does need some contact gradient.

**The RNEA residual as per-layer feedback.** Feeding the unbalanced wrench of layer k into layer
k + 1 (the idea of an iterative force allocator) gives 0.003 body weights on its own and nothing on
top of the RNEA loss, for 10 % more training time. The loss already carries the information.

**Frame masking.** Replacing 10 % of the input tokens by a learned embedding during training
changed nothing, on any metric, and the train / test gap stayed the same.

**Better gravity than the camera axis, on this corpus.** 40 % of the corpus gravity values are not
measurements at all (they are the first camera's down axis), and on the scenes that do have a
measurement the two available estimates disagree by 10–20°. A gravity head cannot be validated below
that with these targets; better targets come first.

**Heel contacts.** Still F1 0, as in every round: the heels are labelled in contact 3.5 % of the time
and the labels carry no usable signal for them.

## Things to know before the next round

* Contact F1 peaks at epochs 10–17 of 30 and drifts down after; keep `best.pth` by F1.
* The test split changed during round 8 (108 → 107 scenes, 56 annotations re-edited); arms launched
  after 14:40 on 2026-09-12 use the new split. Paired comparisons use the scenes both runs share.
* Judge new arms against P_k30 / F2m with `scripts/paired_ci.py`; the seed spread above is the
  noise floor.
