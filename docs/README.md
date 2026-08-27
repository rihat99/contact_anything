# Project documentation — start here

This repository asks a simple question about rock climbing videos: **where is the climber
holding on, and how hard are they pulling?** It answers it by taking Meta's *SAM 3D Body* —
a large frozen model that reconstructs a 3D human body from a single image — and teaching small
new modules, bolted onto its side, to read three extra things out of the same features:

1. **Contact** — which body parts are holding the wall (per mesh-vertex on still images;
   per joint or extremity group on video, where a label means *stable* contact — see
   [data.md](data.md)),
2. **Force** — the 3D contact force at each extremity (up to six groups: hands, toes, heels),
   in units of body weight,
3. **Motion** — the pelvis' velocity and acceleration over a video clip.

The base model stays frozen. Every new capability lives in added tokens, heads, and small
attention blocks, wired in so that the frozen model's own outputs are provably unchanged
(test-enforced to the GPU's numerical noise floor). The one deliberate, flagged exception:
the current all-modality recipe unfreezes a single projection of the pose head
(`train.finetune_pose_head`). Ground truth for forces and motion comes not from
sensors but from a physics pipeline that reconstructs each climbing video in 3D and solves
inverse dynamics for the forces that explain the motion.

<!-- FIGURE inserted below: full pipeline -->
![One forward pass: frames → frozen backbone → frozen decoder with appended tokens →
post-decoder bricks → heads](figures/pipeline_overview.png)

## Reading map

Read in this order if you are new; each page stands alone if you are not.

| Page | What it explains |
|---|---|
| [architecture.md](architecture.md) | The frozen base model, our added tokens/heads/temporal bricks, the attention mask that keeps pose untouchable, and the freezing machinery |
| [data.md](data.md) | Every dataset, what a contact label actually means (subtler than it sounds), and the physics-derived force/motion ground truth |
| [losses.md](losses.md) | Every training signal, what each one compares, and exactly which parameters its gradient reaches |
| [experiments.md](experiments.md) | The chronological story: what we tried, the numbers, the failures and their post-mortems, and where the project stands |
| [forces.md](forces.md) | Deep dive on the force heads and the physics (inverse-dynamics) loss — full math, frames, and conventions |
| [consistency.md](consistency.md) | Deep dive on the pose→motion consistency loss — the current working frontier: the world-lift geometry, all six terms, the null-space collapses, and the open problems |
| [glossary.md](glossary.md) | A–Z definitions of every project-specific term |

## The project in one paragraph

A frozen single-image body-reconstruction model already computes rich features about a person;
this project shows those features also support *contact* (readable well: four-extremity test
F1 0.935 against a 0.878 trivial baseline on manually annotated climbing video; ~0.83 on the
finer six-group task), *force* (readable moderately: ≈ 0.16 body-weight mean error against
physics-derived targets), and *motion* (readable only weakly — a chain of careful
negative results showed the frozen features carry almost no velocity signal, and temporal
attention over clips recovers only part of it). The recurring engineering theme is isolation:
name-filtered freezing, asymmetric attention masks, zero-initialized gates, and detached loss
paths ensure each new capability can be added, measured, and removed without disturbing
anything else — every experiment is then an honest ablation.

## Where things are

- Configuration: `configs/base.yaml` holds the commented defaults (a few newer sections
  default inside `contact/config.py`); experiment yamls override.
- Code: `contact/` (our library), `sam_3d_body/` (the vendored fork, changes marked),
  `scripts/` (thin CLIs).
- Commands for training/evaluation/rendering: the repo-root `CLAUDE.md` — a dense operational
  reference kept for day-to-day work; these pages are its readable counterpart.
