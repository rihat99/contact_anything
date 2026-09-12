# The current model, in plain words (2026-09-12, after round 8)

This page describes the model as it stands after round 8 (`configs/r8/F2m_scale03.yaml`): the
round-7 body that smooths itself, with the contact, force and gravity heads back on top of it.
The measurements are in `docs/round8_2026-09-12.md` (heads) and `docs/old/round7_2026-09-11.md`
(smoothness); the earlier designs are in `docs/old/`.

## 1. What the model does

Input: a video clip of a climber (60 frames, about 2.4 seconds) with the camera's position and
orientation for every frame. Output: the climber's SMPL-X body for every frame — where the pelvis
is in the world, how the body is oriented, and the rotation of the 21 body joints. The body is
smooth in time the way the real motion is, and it is placed in the world, not in the camera.
For every frame it also outputs which of the six extremities (hands, toes, heels) are in contact,
the contact force on each of them (in body weights, in the body frame), and the direction of
gravity in the body frame. Nothing about gravity is read from the data at inference.

## 2. Two stages

**Stage 1 — one frame at a time.** The frozen SAM 3D Body model reads each frame and produces its
pose token (a 1024-number summary of the person). A small head trained by us turns that token into
the SMPL-X body in the camera frame: the joint rotations, the body shape, and the pelvis position
through a weak-perspective camera on the crop. This stage knows nothing about the other frames, so
its body is right on average (about 57 mm joint error) but jitters badly from frame to frame
(jitter 105 on a scale where the real motion scores 7).

Stage 1 is trained once and frozen. Its pose tokens are cached on disk, so stage 2 trains in
minutes.

**Stage 2 — the temporal refiner.** It takes the stage-1 bodies of the whole clip, lifts them
into the world with the known camera poses, and corrects them using their neighbours in time.
It outputs corrections, not a new body: a small shift of the pelvis, a small rotation of the root
and of each joint, added onto the stage-1 body. At initialization every correction is zero, so
the untrained refiner is exactly stage 1.

## 3. Inside the refiner

For each frame the refiner builds one token from quantities that do not depend on where the world
origin is (so the model cannot cheat by memorizing world positions):

* the joint positions relative to the pelvis, in the body's own frame;
* how fast the pelvis is moving and turning, expressed in the body frame, measured **one-sidedly**
  (this frame minus the previous one, and the next one minus this one);
* how fast every joint is rotating, again both one-sided rates;
* the time gap to the neighbours, the body shape, the stage-1 pose token, and the camera as seen
  from the body: the direction to it, its distance, the crop box, and the camera's own down and
  viewing axes expressed in the body frame (the camera's tilt relative to the body).

These tokens go through three transformer layers that can look 0.5 seconds to each side per
layer, positions encoded by real elapsed time.

**The iterative part.** The layers are not run as one block. After each layer, a shared head
predicts a correction, the correction is applied to the body, forward kinematics recomputes the
joints, and the corrected body's own velocities are fed back into the token stream before the
next layer. So layer 2 refines the trajectory layer 1 already improved, instead of re-reading the
raw stage-1 one. This is the same idea as the SAM 3D Body decoder, which re-injects its current
estimate after every layer. Each intermediate body is also scored by the loss at half weight
(deep supervision), so every layer is asked for a better body, not just the last one.

**The other heads.** After every layer, three more small heads read the same hidden state:
contact (six probabilities), force (six 3-vectors) and gravity. Their outputs at layer k are fed
back to layer k + 1 next to the velocities (the contact probabilities and the body-frame gravity),
so the later layers see what the earlier ones decided. The gravity head does not predict a
direction from scratch: it predicts a small correction on top of the camera's down axis, the
corrections of all frames are averaged over the clip as unit vectors, and the result is one
down vector per clip (gravity does not change within 2.4 seconds). The camera axis alone is
12° off the corpus gravity on the scenes where that gravity was measured; the head ends 1° better,
which is inside the uncertainty of the targets themselves (see the round-8 write-up).

**The coupling.** These heads share the trunk with the pose, and their gradients reshape it: a
contact head trained at full strength costs the body 3.5 jitter units, and the gravity loss even
reaches the body's rotation through the world lift. So the gradient the heads send into the trunk
is scaled by 0.3 (`head_grad_scale`). At 0 the heads are probes of a trunk only the pose trains,
which keeps the smoothness but loses 0.05 contact F1; at 1 the forces are best but the body
jitters; 0.3 keeps the forces of the shared trunk and the smoothness of the probes.

## 4. How smoothness is achieved

There is no smoothing filter anywhere in the model. The smoothness comes from three things
working together.

**The model can see the jitter.** Frame-to-frame jitter is mostly a wobble that flips sign every
frame. A central difference (next frame minus previous frame) is completely blind to it: the two
neighbours agree, and the wobbling frame in between cancels out. All earlier versions used central
differences for both the velocity inputs and the velocity losses, so the model was trained on a
signal that could not see the error it was judged on. Now every rate, in the inputs and in the
losses, is a one-sided difference, which sees the wobble in full.

**The model has the right inputs to remove it.** To smooth a value, a layer has to compute "what
my neighbours say minus what I say". A residual attention block does this badly when the input
is the absolute value, and well when the input is a rate: then the correction is simply a signed
sum of the neighbours' rates. That is why the pelvis smoothed early (its token always carried
rates) and the joints did not until each joint got its own rate channels. Note there is no
integration: the model still outputs a correction to the anchored per-frame body, so nothing
drifts.

**The loss asks for smooth velocities, not smooth accelerations.** The training target for
motion is the raw ground-truth velocity (one-sided, no smoothing of the target). Raw velocity is
90 % predictable from the clip, so matching it pulls the body towards the true motion. Raw
acceleration is mostly measurement noise; matching it pointwise makes the model shrink its
motion (which costs joint error), matching its amplitude makes the model reproduce the noise,
and smoothing the target makes the model produce a smoothed, too-slow velocity. Position losses
alone do not produce smoothness at all. So the recipe is: keypoint and rotation losses for
accuracy, a clip-level root anchor to keep the world position honest, and a velocity loss for
smoothness — nothing on acceleration.

With these three in place, the iterative refiner reaches the real motion's jitter level after
about 15 epochs. Training longer keeps improving the joint error by hundredths of a millimetre
while jitter slowly rises again. With the contact head on, the checkpoint kept is the one with
the best contact F1, which lands around epoch 10 to 17.

**Two more losses on the body.** Contact consistency: on frames where a limb is labelled in
contact, the world speed of that extremity (a one-sided difference, so it sees the wobble) is
pushed towards zero. It brings the predicted in-contact speed to the level of the ground truth
(0.19 m/s, the labels are not perfectly still) and cancels most of the contact head's jitter, at
about 0.2 mm of joint error. Physics consistency: BetterRobot's inverse dynamics computes the
root wrench the refined motion needs under the corpus gravity, and the six predicted forces have
to supply it; the residual (force in body weights, torque about the pelvis) is penalised. This is
the one force change that helped: the force angle error drops from 22° to 20.5°.

## 5. What it measures

On the test clips (120 frames each at most; the split lost one scene mid-round 8, so the last row
is on 107 clips and the round-7 body re-scored there is given for reference):

| body | joint error (mm) | jitter (real motion: 6.9) | pelvis error (mm) | contact F1 | force MAE (bw) / angle |
|---|---|---|---|---|---|
| stage 1 alone | 56.2 | 108.7 | 120 | – | – |
| stage 1 + fixed Gaussian filters (round 5) | 55.5 | 2.7 (over-smoothed) | 117 | 0.927 | 0.185 / 20° |
| the round-7 body (108 clips) | 54.9 | 5.9 | 110 | – | – |
| the round-7 body (107 clips) | 53.9 | 5.6 | 109 | – | – |
| this model (107 clips) | 53.9 | 6.0 | 109 | 0.924 | 0.179 / 20.6° |

The remaining 110 mm pelvis error is a slow bias of stage 1 (most of its error power is below
0.5 Hz); a temporal model cannot remove it, and letting the stage-1 head train together with the
refiner made it worse.

## 6. Where things are

* Recipe: `configs/r8/F2m_scale03.yaml` (the chain G0 → G1 → C1 → C2 → F0 → F2 → F2m on
  `configs/smooth/P_k30.yaml`); run `output_3/F2m_scale03_20260912_154016/best.pth`. The
  pose-only body stays `configs/smooth/P_k30.yaml` / `output_3/P_k30_20260912_011321/best.pth`.
* Refiner code: `model/refiner.py` (`TemporalRefiner`, the one-sided rate helpers, the iterative
  loop, the per-layer heads, the gravity pooling, `head_grad_scale`); the temporal block:
  `model/rope.py`; the physics: `model/physics.py`; the losses: `model/loss/motion.py` (velocity
  terms), `model/loss/smplx.py` (body terms and the jitter diagnostics), `contact.py`, `force.py`,
  `gravity.py`, `contact_consistency.py`, `force_consistency.py`.
* Evaluate: `scripts/evaluate.py --config configs/r8/F2m_scale03.yaml --checkpoint <run>/best.pth`;
  compare two runs with `scripts/paired_ci.py`; look at the bodies with `scripts/view_results.py
  --output output_3`.
