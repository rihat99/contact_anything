# Claude Opinion: Temporal Contact Learning Audit

I read the full pipeline (loader, collate, targets, losses, temporal module, decoder hooks,
evaluate, video renderer), re-ran every data statistic in the codex audit, and added two
measurements codex did not make. Codex's facts are correct. I disagree with its headline story
and with the priority of some of its fixes.

## Codex's facts check out

I reproduced every number independently:

| Claim | Codex | My measurement |
|---|---|---|
| Constant-label T=5 windows per extremity | ~93% | 93.4% |
| Mean confidence near transitions vs away | 0.30 / 0.66 | 0.314 / 0.664 |
| Zero-confidence negatives vs positives | 15–17% / <0.5% | 15.8% / 0.43% |
| Initial focal pressure from positives | ~76% | 76.9% |
| Adjacent-frame time-encoding cosine similarity | 0.99997 | 0.99997 |
| Best checkpoint picked by F2 is epoch 0 | yes | yes — in **both** runs |

I also confirmed in code: labels are truly per-frame (no broadcasting across the window),
collate order is clip → frame → joint, confidence is applied once, the DDP reduction is exact,
the renderer's centered T=5 window emits the correct center row, and the exporter really uses a
0.35 s stillness window, hysteresis (enter 2 cm / exit 3.5 cm), and gap merging up to 2 s.

So I agree: **there is no alignment or dataloader bug.** The user's hypothesis — "if one of the
five frames is in contact, the others become contact" — is not what happens. Labels differ frame
by frame inside a window, the loss is per-frame, and cross-frame influence is measurably small.

## What codex missed

**1. The trained temporal module is nearly inert — and a net negative.**
The zero-init gates barely moved in 5 epochs: mean |gamma_attn| = 0.004, mean |gamma_ffn| = 0.002.
I ran the temporal `last.pth` on 1,500 val frames twice: module active vs bypassed (gates zeroed,
which makes it an exact identity):

- Mean probability: 0.520 active vs 0.509 bypassed. The module mostly adds a small **positive
  bias** (+0.011 mean, 9% of frame decisions flip at threshold 0.5).
- **Bypassing the module improves val F1: 0.748 vs 0.721** (precision 0.700 vs 0.660, recall
  about equal).

So "the temporal model learned slightly wider contact spans" is better read as: *the temporal
module learned almost no temporal structure; it learned to predict contact a bit more.* Codex
masked future frames (good experiment) but never measured the whole-module bypass, which is the
more basic question: does the module help at all? Right now it does not.

**2. The timing comparison is confounded by operating point.**
Codex compared onset/offset timing between the temporal run and the per-frame run at the same
threshold. But at epoch 4 the two runs sit at very different operating points on val:
temporal = recall 0.82 / precision 0.59; per-frame = recall 0.65 / precision 0.65. A classifier
that simply says "contact" more often will always fire earlier at onsets and hold later at
offsets — no temporal reasoning needed. Most of the "-3/+3 vs -1/+2 frames" gap is likely this
calibration difference, not attention across frames. Timing must be compared at matched
predicted-contact fractions (adjust each model's threshold until both predict the same total
amount of contact).

**3. Both runs are too short to conclude anything.**
Five epochs each. Per-extremity val metrics swing wildly between epochs (right-foot recall:
0.88 → 0.60 → 0.80 → 0.67 → 0.70). The per-frame run's val loss *doubles* after epoch 1
(0.26 → 0.52) while the temporal run's stays flat (~0.27) — one interesting point in the temporal
module's favor (it did not overfit), but mostly a sign that neither run converged. A zero-gated
module in particular needs time: its gates start at zero and grow slowly. Design conclusions
drawn from these two runs are weak.

**4. Train and test have very different class balance.**
Train labels (4-extremity, supervised, confidence-weighted view): only **36% positive**. The five
manual test chunks: 81–92% positive. Codex noted the test chunks are contact-heavy but not the
train rate. This gap means: (a) focal alpha 0.8 gives 4× weight to a class that is only a modest
minority — combined with the confidence asymmetry it produces the 77% positive pressure; (b) any
threshold tuned on val will behave differently on the test chunks; (c) global P/R/F1 on the test
chunks mostly measures "does it say contact", not timing.

## Why the model predicts contact "before the hand touches" — ranked

1. **Operating point.** Alpha 0.8 + confidence asymmetry + F2 selection all push toward "when
   unsure, say contact." A positive-leaning model crosses the threshold frames before the touch.
2. **The labels themselves lead and lag the visible touch.** They mark *stable, load-bearing*
   contact from a 3D-reconstruction pipeline: a 0.35 s stillness window, hysteresis that holds a
   span until the joint moves 3.5 cm away, and gap merging up to 2 s (a hand briefly off the hold
   stays labeled "contact"). Even the per-frame model is 1 frame early / 2 late — that is the
   label, not the model. Nobody has yet measured how early the labels are against the video.
3. **The temporal module adds its own small positive bias** (the bypass measurement above).
4. **Not** label sharing across the window, and (per codex's masking experiment, which I believe)
   not future-frame leakage.

## Plan — what I would actually do, in order

**Step 0 — look at the labels before touching the model (no training).**
Open ~10 onsets/offsets in the dataset viewer and compare the train label timing against the
video. If the labels themselves are 2–3 frames early, no training change can fix the symptom,
and the decision becomes: re-export labels with tighter parameters (smaller stillness window,
shorter/no gap merge) or accept "stable contact" semantics. This is codex's point 5; I would make
it the first action, not a footnote.

**Step 1 — cheap config fixes, then retrain BOTH variants longer (15–20 epochs).**

- `output.monitor: val/joint_f1` (F2 picked the near-untrained epoch 0 in both runs; the F1
  trajectories would have picked epoch 1 and epoch 4 respectively).
- `focal_alpha: 0.5–0.6` (positives are 36% of mass, not a rare class).
- Scale the temporal position encoding to frame offsets (multiply seconds by fps, or pass
  offsets). Confirmed degenerate: adjacent-frame cosine similarity 0.99997 now, 0.972 scaled.
  This is the one small code change and it is required for the module to ever learn order.
- Keep everything else as is: post_decoder, T=5, 1 layer, dim 256, confidence weighting,
  noncausal, `attend: joint` for now.

**Step 2 — fix evaluation so numbers mean something.**

- Score exactly one prediction per physical frame using the same centered sliding policy as the
  renderer (and prefer masked, padded T=5 windows over unseen T=3 boundary windows — training
  never saw T=3).
- Add onset/offset timing error and transition-match metrics per limb.
- Compare temporal vs per-frame **at matched predicted-contact fraction**, not a shared fixed
  threshold.

**Step 3 — decision gate on the temporal module.**
After Step 1+2: if the temporal variant still does not beat the per-frame variant at a matched
operating point, park the module — a per-frame model with a good operating point may be all
this dataset supports. Only if it does help, run the next ablations: `attend: per_token`,
codex's attention-without-FFN, and `frame_stride: 2` (T=5 then spans 0.27 s, closer to the 0.35 s
stillness window the labels were made with — a config-only ablation codex did not consider).

**Postponed, mild disagreement with codex:** transition oversampling. It touches the sampler,
DDP sharding, and resume logic, and the frames it upweights are exactly the frames whose labels
the exporter itself was least sure about (confidence 0.31) — training harder on them may teach
exporter quirks, not touch timing. Try it only if Step 0 shows the labels are trustworthy at
transitions and Steps 1–3 leave a timing gap. Same for causal attention: agree with codex, keep
it out of the first corrected experiment.

## Where codex and I agree on the bottom line

Keep the design simple: frozen base, four tokens, focal-only loss, small post-decoder module.
The pipeline is not buggy. The fixes are about supervision pressure, checkpoint selection, time
encoding, and honest evaluation — not about more architecture.

One caveat on my own numbers: the bypass measurement used the first 25 val batches (1,500 frames,
unshuffled loader), and val metrics here come from tiled windows scoring all five outputs. The
direction (module ≈ positive bias, no help) is clear; exact magnitudes will move once Step 2's
evaluation exists. Codex's reproducibility remarks (per-rank seeding, scheduler on resume) I did
not verify; they are plausible and worth doing before the next long run.
