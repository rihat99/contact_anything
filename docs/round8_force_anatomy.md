# Round 8 force anatomy (2026-09-13): does the RNEA loss help, and why are the feet under-forced?

Two questions after the round-8 recipe (`configs/r8/F2m_scale03.yaml`, run
`output_3/F2m_scale03_20260912_154016`): (1) what the RNEA loss did to the forces, with the
numbers from the arms; (2) why the climb_wall_2 boards show the hands right and the feet low.
Scripts live in the session scratchpad; every table below is in `output_3/logs/` (`pergroup_test_*`,
`perlimb_loss_test.log`, `gt_median_test.log`, `foot_pose_bins_*`, `foot_camside_test_F2m.log`,
`cw2_*_F2m.log`).

## 1. The RNEA loss (arms F0 → F2 → F2m, 107/108-scene test split)

| arm | force MAE (bw) | angle | off-contact | total mean force (bw), GT 0.946 | hands / feet load ratio (pred / GT, loaded rows) |
|---|---|---|---|---|---|
| F0 (force head, no RNEA) | 0.184 | 21.8° | 0.020 | 0.795 | hands 0.85 / 0.89, feet 0.62 / 0.66 |
| F1 (RNEA residual as feedback) | 0.182 | 21.4° | 0.020 | – | – |
| F2 (RNEA loss) | 0.179 | 20.3° | 0.027 | 0.948 | hands 1.00 / 1.01, feet 0.74 / 0.77 |
| F3 (feedback + loss) | 0.178 | 20.8° | 0.027 | – | – |
| F2m (F2 + head_grad_scale 0.3) | 0.179 | 20.6° | 0.028 | 0.947 | hands 0.99 / 1.01, feet 0.74 / 0.77 |
| round 4 A vs B (Gaussian body) | 0.191 → 0.185 | 22.4 → 21.3° | 0.023 → 0.116 | | |

Two seeds of F2m differ by 0.001 bw / 0.2°. So the RNEA loss is worth −0.005 bw and −1.5°
each time it was tried, and its real effect is on the TOTAL: without it the model predicts
0.80 bw of force against 0.95 in the GT (every limb short); the RNEA loss restores the total
exactly — the hands to 1.00 of GT, the feet only to 0.75. The price is +0.007 bw of force on
limbs that are not in contact. Net upward force (world frame, along −gravity): F0 0.70, F2m
0.85, GT 0.82 bw (the GT is not 1 bw: the kindyn solve only accounts for the contacts it labels).

Per group on the test set (world frame, in-contact rows, confidence-weighted like the loss):

| group | Huber | MAE | MAE / mean GT | mean GT | mean pred |
|---|---|---|---|---|---|
| left hand | 0.036 | 0.157 | 0.51 | 0.307 | 0.311 |
| right hand | 0.034 | 0.156 | 0.44 | 0.359 | 0.364 |
| left toe | 0.044 | 0.171 | 0.53 | 0.321 | 0.251 |
| right toe | 0.043 | 0.171 | 0.52 | 0.328 | 0.268 |
| heels | 0.035 / 0.075 | 0.14 / 0.23 | 1.00 | 0.14 / 0.23 | 0.000 |

The loss is not hand-dominated: the relative error is the same for hands and toes, and the
toes get 40 % of the loss mass. The difference is a BIAS: the hands match the GT at every
quantile (median 0.30 vs 0.27, p90 0.56 vs 0.63), the toes are low at every quantile (median
0.20 vs 0.27, p90 0.52 vs 0.62, frac > 0.5 bw 0.12 vs 0.22). Heels are never predicted (F1 0
in contact, force 0), which costs 0.14–0.23 bw on 4–5 % of the rows.

Foot force by foot pose (test set, loaded rows): the ratio pred / GT sits at 0.69–0.86 in every
bin of knee angle, foot height and lateral offset, left and right alike. It drops to 0.61–0.68
when the foot is more than 0.3 m FARTHER from the camera than the pelvis (a camera above the
climber looking down) and rises to 0.87–0.97 when the foot is nearer than the pelvis. The GT
prior itself (200 train scenes): a loaded foot carries 0.53 bw when 0.8 m below the pelvis,
0.25 at pelvis height, 0.19 above it; hands carry 58 % of the corpus force, feet 42 %.

## 2. climb_wall_2 boards (7 trials, 2535 frames, kindyn mass 64–68 kg; 1 bw ≈ 649 N)

| | boards | kindyn optimisation | F2m |
|---|---|---|---|
| net upward force (bw of the kindyn mass) | 1.11 | 0.955 | 0.913 |
| share LH / RH / LF / RF (%) | 23 / 24 / 29 / 24 | 24 / 25 / 28 / 23 | 28 / 35 / 27 / 10.5 |
| hands / feet | 0.90 | 0.96 | 1.69 |
| loaded-row ratio LH / RH / LF / RF | – | 0.80 / 0.80 / 0.73 / 0.72 | 0.90 / 1.04 / 0.66 / 0.33 |
| median angle LH / RH / LF / RF | – | 12.4 / 9.0 / 8.7 / 10.4° | 10.6 / 7.8 / 12.5 / 11.7° |
| contact recall RF (board > 50 N) | – | – | 0.95–1.00, mean p 0.87–0.92 |

* The boards' net upward force is 1.11 bw of the kindyn mass, so the climber weighs ~10 %
  more than the solve's mass estimate; in newtons everything scaled by that mass is 10 % low
  (the optimisation too). The model's own total is a further 9 % short.
* On the boards the feet carry 53 % of the load (hands / feet 0.90); in the corpus GT the
  feet carry 42 % (1.37). The model outputs 1.69 — the corpus prior and then some.
* The left foot behaves like the corpus (0.54–0.97 per trial, 0.66 pooled). The RIGHT foot is
  the outlier in every trial: 0.28–0.44 of the boards, 60–105 N against 250 N, with the contact
  head firing (recall ~1, p ≈ 0.9). It is the FORCE head that starves it, not the contact head.
  The previous learned model (round 4 dump `predictions/pre_allmod_20260830`) had the same
  pattern, milder: RF 0.52–0.73, LF 0.45–0.94.
* Nothing in the pose explains it: the right foot is at the same height (−0.4 to −0.7 m),
  a similar lateral offset (0.5–0.6 m) and a slightly straighter knee than the left, and
  the corpus ratio in those bins is 0.75–0.85 for both feet. The model's body agrees with the
  kindyn body to 5–13 cm on every limb.
* What does single the right foot out is the camera: cam_left looks down at the climber
  (its down axis is 16.6° off the ground-plane gravity) from the climber's left, so the right
  foot is the FAR foot (0.60–0.71 m farther than the pelvis, the left 0.42–0.53) and the
  least visible joint (Sapiens scores 0.55–0.77 for the right small toe / heel vs 0.80–0.96
  on the left). The corpus already shows the model under-forcing far feet (0.61–0.68).
* Gravity: the gravity head returns the camera axis ± 1–5° on every window, so the model
  believes "down" is 16–21° off the true gravity here (`cw2_gravity_F2m.log`). Under that belief
  a foot 0.65 m farther than the pelvis looks 0.19 m higher than it is, which by the corpus
  prior takes ~0.1 bw off it — a contribution, not the 0.25 bw gap, and it does not explain the
  left / right split on its own.

## Reading

The hands are predicted well because the corpus GT hands are the well-determined part of the
kindyn solve and the body pose reads them directly; the feet are systematically low on the
corpus (0.75 of GT, at every quantile, at the same relative error as the hands) and worse when
the foot is far from the camera. The boards add a shift of the true load towards the feet that
the corpus prior does not have, a camera that makes the right foot the far, half-hidden one,
and a believed gravity 17° off. The RNEA loss fixes the total, not the distribution: it fills
the missing force where the model is most confident, the hands.

What would move the feet: (a) a loss that cannot be satisfied by the hands alone — a
per-limb magnitude term or group weights on the toes (both exist in `force_supervision.loss`,
unused); (b) a gravity that is actually measured at inference (the head is a no-op on this
corpus; the ground-plane gravity of the trees is available and could be fed instead of the
camera axis); (c) the far-foot visibility problem is an input problem the image side has to
solve (a second camera, or 2D keypoint confidence as a token channel).
