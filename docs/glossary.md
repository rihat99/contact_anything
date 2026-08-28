# Glossary

Alphabetical reference for the project-specific terms used in these docs and in the code.
Cross-references in *italics* point at other entries; file paths point at the defining code.

**allmod** — shorthand for the all-modality experiment
(`configs/old/climbing_corpus_allmod.yaml`): contact + force + motion + pose branches trained
together on the climbing corpus with the *causal* decoder mask and the *cross-modal temporal* +
*frame attention* bricks (per-modality temporal blocks off). The strongest all-modality recipe;
the contact+force specialist `corpus6_jf_cond_sum1_postdec` still leads on those two tasks alone —
see [experiments.md](experiments.md).

**anchor (token anchor)** — the *MHR70* keypoint a contact/force/motion token is tied to. The
token itself is a plain learned embedding; after every decoder layer except the last, an anchored
update adds a positional encoding of the keypoint's currently-predicted 2-D image location and
backbone features grid-sampled there — so "the left-wrist contact token" keeps looking at the
left wrist as the estimate refines. Configured via `contact_keypoint_indices` /
`force_keypoint_indices` / motion anchors.

**anchor (loss anchor)** — in the *motion-consistency* loss: an absolute supervision term
(`loss.pos`, `loss.rot`) that pins the world-frame root position/orientation to the *kindyn*
trajectory, closing the null spaces that derivative-only terms leave open.

**asymmetric attention mask** — the decoder mask that lets appended tokens attend the original
tokens but never the reverse. The reason the frozen model's outputs are provably unchanged.
See [architecture.md](architecture.md).

**backbone** — DINOv3-H, the frozen vision transformer that turns the input image into a
`32 x 32` grid of 1280-dim embeddings. Runs in bfloat16.

**body-22** — the first 22 joints of the *SMPL-X* skeleton (pelvis through wrists), the label
space of the climbing-corpus contact annotations.

**brick** — informal name for one of the toggleable post-decoder modules
(*cross-modal temporal*, per-modality temporal, *frame attention*, *pose temporal*). Each is
zero-gated and independently enabled from config, like a Lego brick.

**BVR** — the sibling video-reconstruction pipeline that produced the climbing corpus
(person tracking, SMPL-X fitting, camera estimation, *kindyn* solve). Not part of this repo;
its outputs are this repo's ground truth.

**bw (body-weight units)** — forces are expressed as multiples of the subject's body weight
(`|F| / (m g)`), making them dimensionless and comparable across climbers.

**causal / mutual** — the two regimes of `model.extra_token_attention`, governing attention
*among* the appended token blocks. `causal` (the default until 2026-08-27): no earlier block
attends a later one (contact ⊥ force ⊥ motion). `mutual` (the default since): appended blocks
fully inter-attend. Original tokens attend neither, in both regimes.

**clip** — a fixed-length window of `T` consecutive sampled frames from one scene
(`data.sequence.frames_per_clip`, typically 7). Batches flatten clips to `[B_clips * T, ...]`
plus bookkeeping (`seq_len`, `frame_pos_sec`, `frame_valid`).

**cond_input / cond_feat** — an *input*-side conditioning feature (`model.cond_input`): a 10-dim
standardized descriptor of the smoothed root trajectory's kinematics, computed from *kindyn* and
fed into the model so the force branch does not have to infer acceleration from a short clip.
Used by the `corpus6_jf_cond*` line; see [experiments.md](experiments.md).

**contacts_1 / contacts_2** — the two automatic contact-label channels in the corpus.
`contacts_1` is the default training label; `contacts_2` is the (stricter) mask the *kindyn*
force solve was run under, and therefore the gate for the supervised force loss.

**corpus** — the raw `ClimbingVideos` dataset tree read directly by
`contact/data/climbing_corpus.py`: 864 train + 108 test scenes (31 annotated so far;
pre-2026-08-27: 331 + 30) with frames,
contacts, masks, camera geometry, and kindyn ground truth.

**cross-modal temporal** — ONE zero-gated temporal attention block run over the concatenation
of the chosen modality token blocks, across all frames of a clip
(`model.cross_modal_temporal`). The only *brick* where different modalities mix across time.

**D1** — the design invariant "contact outputs have an exactly-zero Jacobian with respect to
every force and motion parameter", guaranteed by the *causal* mask and deliberately relaxed by
`mutual` and by post-decoder bricks that list multiple modalities.

**D8** — the design rule "gate the physics loss on the model's *predicted* contact
probabilities, not on the dataset's labels": the corpus' *stable contact* labels describe
stillness, not instantaneous load, so gating on them would silence real forces.

**DAMON** — the per-vertex contact dataset from the DECO paper: still images with SMPL 6890
vertex-level contact labels. Used for still-image contact training.

**DINOv3** — Meta's self-supervised vision transformer family; the "-H" (Huge) variant is this
project's frozen backbone.

**E-series (E0 … E2b)** — codenames of the pose/motion temporal experiment ladder: E0/E1
variants placed temporal attention *inside* the decoder (retired as controlled negatives); E2 is
the post-decoder pose-token temporal module with q-space supervision, E2b its smoothness variant.
See [experiments.md](experiments.md) §9.

**extrinsics** — the per-frame world-to-camera rigid transform (`cam_from_world`, OpenCV
convention, metric scale) estimated by *BVR* for every corpus video frame. Required by the
physics loss and the motion-consistency world lift; still images have none (`cam_valid=False`).

**focal loss** — binary cross-entropy reweighted to focus on hard, rare positives; the
contact-classification loss (`alpha=0.6`, `gamma=2` in the shipped climbing experiments).

**frame attention** — per-frame, cross-modality zero-gated attention run after every temporal
block; one own-weights module per listed modality, whose keys/values span every enabled
modality's tokens of that frame (`model.frame_attn`,
`sam_3d_body/models/modules/frame_attention.py`). No temporal mixing.

**force_mae** — the force monitor (`test/force_sup/mae`): mean absolute error, in *bw*, between
predicted and kindyn GT forces over the supervised rows.

**free-flyer** — the 6-degree-of-freedom un-actuated joint connecting a floating-base body
(here: the human root) to the world. Its 6D generalized force is what the *RNEA* residual
measures; its pose channels are deliberately never supervised in q-space pose supervision.

**freeze filter** — the name-based rule deciding what trains: only parameters whose dotted
module path contains `contact`, `force`, `motion`, `cross_modal`, `frame_attn`, or
`pose_temporal` receive gradients (`contact/model.py`). Everything else is frozen.

**grouped split** — train/val/test scenes are grouped by *source video*: two chunks of the same
YouTube video can never land in different splits (prevents near-duplicate leakage).

**kindyn** — "kinodynamics": the corpus stage that takes the reconstructed SMPL-X motion and
solves inverse dynamics for the contact forces that explain it. Produces per-scene
`kindyn_1.npz`: 6-group GT forces in *bw* units (body-root frame), the smoothed root
trajectory, and validity masks. The project's force/motion/pose ground truth.

**kindyn_6 groups** — the six force groups the kindyn solve outputs, in order:
left hand, right hand, left toe, right toe, left heel, right heel
(anchor keypoints `[62, 41, 15, 18, 17, 20]`).

**local_world_aligned / local / root (force frames)** — the coordinate frame a predicted force
is expressed in: gravity-aligned camera frame, raw camera frame, or the body-root frame.
The supervised kindyn regime uses `root` (no extrinsics needed anywhere).

**manual test split** — the 30 corpus scenes with human-reviewed contact annotations
(`annotation.npz`): 14 observable joints labeled per frame, fingers folded into the wrists,
the remaining 8 joints defined non-contact. The only evaluation set (there is no validation
split by design).

**MHR (Meta Human Rig)** — the parametric body representation SAM 3D Body predicts: a rigged
skeleton + shape space. A world-frame MHR trajectory is a 132-channel configuration vector `q`
per frame: 7 *free-flyer* root channels (translation + xyzw quaternion) + 125 local pose
channels.

**MHR70** — the 70-keypoint set decoded from MHR predictions; token *anchors* index into it
(e.g. 62 = left wrist, 41 = right wrist).

**mhr_1.npz** — per-scene kindyn-to-MHR conversion (`scripts/convert_kindyn_to_mhr.py`): the
kindyn SMPL-X trajectory re-fitted as a world-frame MHR `q` trajectory (~0.5 cm joint
residual). The pseudo-ground-truth for pose supervision.

**motion-consistency loss** — the pure loss (`contact/motion_consistency.py`, no parameters)
that lifts the *predicted* pose to the metric world, differentiates it, and compares against
kindyn twists and the detached motion head, plus absolute anchors and *rails*. Gradient flows
to the pose path only. Full deep dive: [consistency.md](consistency.md).

**mutual** — see *causal / mutual*.

**observable_14** — the 14 joints an annotator can actually judge in the manual test protocol
(feet, ankles, knees, hips, wrists, elbows, shoulders; `OBSERVABLE_14` in `contact/targets.py`);
the remaining 8 body-22 joints are defined non-contact on reviewed frames.

**pose path** — the set of trainable parameters that can move the pose output:
`pose_temporal`, the pose-designated parts of *cross-modal temporal* / *frame attention*, and
(if `train.finetune_pose_head`) the MHR head's projection `head_pose.proj`.

**pose temporal** — the zero-gated temporal block on the pose token (`model.pose_temporal`);
a deliberate, supervised exception to the frozen-pose rule. The final MHR output is recomputed
from the updated token.

**pred_cam_t** — the frozen model's predicted camera-space translation of the body root; the
quantity whose collapse caused the *mutual* run failure and which the `cam_rail` now anchors.

**q** — a body model's generalized-coordinate vector (root pose + all joint angles). "q-space"
losses compare configurations channel-wise instead of comparing 3D joint positions.

**r3d** — the pooled motion metric: one Pearson correlation computed jointly over all three
spatial components of a target (rather than per-axis), between prediction and kindyn GT.

**rail (trust region)** — a loss term that is exactly zero while a quantity stays within a
margin of the frozen model's own prediction and grows linearly beyond it
(`cam_rail`: `pred_cam_t`, 0.5 m; `rot_rail`: `global_rot`, 0.2 rad). Inert for a healthy
model; blocks collapse escape routes.

**regime (a) / regime (b)** — the two force-training setups. (a): `train.freeze_contact` on — a
pretrained contact branch is loaded and frozen and only the force branch trains (exact gradient
isolation). (b): contact and force train jointly (the physics loss then leaks gradient into
contact through force→contact attention).

**RNEA** — the Recursive Newton-Euler Algorithm: inverse dynamics that, given a trajectory
(q, velocity, acceleration) and external forces, returns the generalized forces required at
every joint. The physics loss runs RNEA and penalizes the resulting 6D *free-flyer* residual.

**scene** — one contiguous chunk of a source climbing video (~hundreds of frames), the corpus'
unit of storage and splitting.

**SMPL / SMPL-X** — parametric human body meshes (6890 / 10475 vertices). SMPL is the label
topology of the still-image datasets; SMPL-X (with articulated hands) is what BVR fits and
what the corpus joint labels index into.

**stable contact** — the corpus' automatic label semantics: a joint counts as "in contact"
only while it is *still* (stillness + hysteresis gating in the estimator). Different from —
and deliberately not derived from — instantaneous vertex proximity.

**stencil** — the (t−1, t, t+1) finite-difference footprint used to compute velocity and
acceleration; rows without full stencil support (clip boundaries) cannot be twist-supervised.

**T** — frames per clip (`frames_per_clip`); T=1 for still images, typically 7 for video.

**temporal module** — the zero-gated pre-LN attention block over a clip's frames
(`sam_3d_body/models/modules/temporal.py`), order-aware via sinusoidal encoding of real
elapsed seconds. Post-decoder only.

**twist (body twist)** — a rigid body's 6D velocity (3 linear + 3 angular) expressed in its
own body frame; the motion targets are the root's twist velocity and acceleration computed
from the kindyn trajectory with the *BVR* stencil convention.

**zero-gating** — initializing each new module's output projection (or a learned gate) at
exactly zero so the module is a provable no-op at initialization: training starts from the
frozen model's behavior, bit-for-bit.
