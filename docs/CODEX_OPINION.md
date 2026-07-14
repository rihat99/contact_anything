# Codex Opinion: Temporal Contact Learning Audit

The core pipeline is aligned correctly. I found no frame shift, label broadcasting, or wrong sliding-window row. The temporal model is instead learning slightly wider contact spans.

### What is already correct

- Every frame has its own image, label, confidence, and prediction. Labels are not copied across the five frames.
- Flattening and reconstruction preserve the order `clip → frame → joint`: [collate.py](/data3/rikhat.akizhanov/better/contact_anything/contact/data/collate.py:107).
- Four distinct contact tokens correctly map to left hand, right hand, left foot, right foot.
- Image patches are added to the contact tokens after decoder layers.
- Confidence is applied once, not twice.
- DDP loss normalization is correct.
- Centered inference selects the third row of each T=5 window correctly: [render_climbing_video_contacts.py](/data3/rikhat.akizhanov/better/contact_anything/scripts/render_climbing_video_contacts.py:73).
- Post-decoder placement is appropriate. It is light and cannot alter the plotted body pose.
- The current temporal module adds about 1.05M parameters. I would keep this placement and size.

### What the empirical audit showed

At threshold 0.5 on the five manually labelled test chunks:

- Precision: 0.931
- Recall: 0.784
- F1: 0.851

So the model does not predict too many contacts globally. Those test chunks are actually extremely contact-heavy: roughly 81–92% of their manual labels are contact.

The problem is local timing:

- Temporal model onset: median 3 frames early.
- Temporal model offset: median 3 frames late.
- Per-frame model onset: median 1 frame early.
- Per-frame model offset: median 2 frames late.

The temporal module therefore widens contact spans by roughly two frames at the beginning and one frame at the end.

Crucially, future frames are not the main cause. I masked both future frames on 171 transition windows:

- Mean probability changed by only 0.005.
- No onset prediction changed class.
- Removing future frames slightly increased contact probability.

There is also no fixed frame shift: comparing predictions against labels shifted from −2 to +2 frames produced almost identical scores.

### What currently encourages wide spans

1. The temporal attention mixes limbs

`attend: joint` lets every output read all four limb tokens from all five frames. A contacting foot can therefore influence a hand token. The temporal FFN can also raise probabilities without using time at all.

2. Temporal order is encoded too weakly

The five timestamps are:

`0, 0.033, 0.067, 0.100, 0.133 seconds`

Their sinusoidal encodings are almost identical. Adjacent encodings have cosine similarity about `0.99997`. This makes “before” and “after” unnecessarily hard to distinguish: [temporal.py](/data3/rikhat.akizhanov/better/contact_anything/sam_3d_body/models/modules/temporal.py:42).

3. Transition supervision is weak

For individual limbs, about 93% of T=5 windows have one constant label.

More importantly:

- Average confidence near transitions: 0.30.
- Average confidence away from transitions: 0.66.

The few frames that teach exact onset and release timing receive about half as much loss weight.

4. Positive pressure is very strong

With confidence included, about 76% of the initial focal-loss pressure comes from positive labels. This happens because:

- Focal alpha is 0.8, giving positive errors four times the class weight.
- Negative labels have much lower confidence.
- Around 15–17% of negative labels have exactly zero confidence, compared with under 0.5% of positives.
- Best-checkpoint selection uses F2, which strongly rewards recall.

This combination naturally makes contacts begin early and finish late.

5. The source labels are already temporally smoothed

Training labels use:

- A centered stillness window of about 0.35 seconds.
- Hysteresis.
- Gap merging of up to 2 seconds.

Therefore the target is “stable, load-bearing contact,” not exact first visible touch. This must be checked visually before blaming the model.

Also, a hand label means any finger or wrist is touching, but the demo plots only the wrist. A foot label means ankle or toe/foot contact, but the demo plots only the ankle. This can make a correct aggregate prediction look visually wrong.

6. Validation does not match video inference

Training validation scores tiled T=5 windows and all five outputs. Video inference uses overlapping centered windows and only their center output. Terminal validation windows also duplicate about 0.6% of physical frames.

T=3 boundary inference is another small mismatch because training only used T=5.

### Proposed minimal next experiment

I recommend keeping the simple design:

- Post-decoder temporal module.
- T=5.
- One layer.
- 256-dimensional bottleneck.
- Four contact tokens.
- Focal loss only.

Change only the following:

1. Use `attend: per_token`.

Each hand or foot then attends only to itself over time. This removes cross-limb contamination while keeping the same four tokens.

2. Strengthen temporal positions.

Use frame offsets `0,1,2,3,4`, or multiply seconds by 30 before the sinusoidal encoding. This adds no meaningful parameters.

3. Keep confidence weighting, but use focal alpha around 0.6.

That still penalizes missed contacts more strongly, but avoids the current extreme positive bias after confidence is included.

4. Select checkpoints by F1, while still reporting F2.

F2 made the almost-untrained, high-recall epoch 0 the current “best” checkpoint.

5. Oversample transition windows.

For example, make approximately half of training clips contain at least one onset or release. Keep the same focal loss and confidence values. This gives the low-confidence boundary examples enough total exposure without inventing another loss.

6. Keep noncausal attention for the first corrected experiment.

The audit showed that future frames are not causing the current anticipation. A causal model should be a separate experiment if online inference is required, using the previous five frames and the final output.

### Evaluation changes before retraining

- Score exactly one prediction per physical frame using the same sliding policy as inference.
- Use masked, padded T=5 windows at video boundaries instead of unseen T=3 inputs.
- Save raw probabilities, labels, confidence, frame identity, and window provenance.
- Report onset error, offset error, transition-match rate, F1, and F2 per limb.
- Select qualitative videos with many manual transitions instead of random contact-heavy chunks.
- Show toe/heel and fingertip locations, or clearly label wrist/ankle circles as aggregate hand/foot predictions.

Threshold 0.3 gives the current temporal model its best overall F1, but lowering the threshold would make contact begin even earlier. Therefore threshold selection should consider onset/offset timing, not F1 alone.

### If span widening remains

The next small ablation would be an attention-only temporal block without its FFN. The FFN is not temporal and may be acting mainly as a positive calibration adapter. I would try this only after `per_token`, stronger time encoding, and transition sampling.

Before another long run, I would also fix reproducibility and resume safeguards: seed every rank, save rank-specific RNG state, correct the cosine scheduler when extending epochs, and apply config checks to explicit resumes. These did not cause the current result, but they matter for trustworthy comparisons.

No files were changed. The read-only audit included real batches, all five test chunks, temporal masking/bypass experiments, the per-frame control checkpoint, and 145 passing CPU tests.
