# Round 11 (2026-09-19): the ladder

The one-off ablations are replaced by a ladder: eight runs, each the previous plus one piece, every
one trained FROM SCRATCH on the same schedule (the stage-1 / stage-2 split is gone; the per-frame
SMPL-X + camera heads train jointly with everything else in every run), plus four ablations off the
last rung. Configs `configs/final/`, runs `output_6/`, rig dumps `predictions/<run>/` under the
climb_wall_3 out-tree, scored by `scripts/score_rig.py`.

## The rungs

| rung | config | adds | trunk |
|---|---|---|---|
| 1 | `final_base` | contact (BCE), force (supervised, zero off-contact) and gravity heads on the pose token (`model.token_heads`); no temporal model, no camera | cached pose token, 1 GPU |
| 2 | `final_temporal` | one 3-layer RoPE block over the pose token (`cross_modal_temporal`, ±0.25 s total); every head reads the mixed token; NO temporal losses (below) | cached, 1 GPU |
| 3 | `final_world` | the world-space refiner instead: extrinsics, world lift, world-independent per-frame token (root-frame joints, local rotations, one-sided rates, projected pose token, camera context + axes), one offset head at the end (`iterative: false`); the velocity-matching and in-contact stillness losses on the refined body; `smplx_supervision.refined` (kp3d 10, pose 2, root_bias 2, root_shape 2) next to the per-frame set | cached, 1 GPU |
| 4 | `final_iter` | `iterative: true` (per-layer heads + feedback, deep supervision 0.5) | cached, 1 GPU |
| 5 | `final_limb` | six contact limb tokens carrying the DECODER contact tokens (`model.contact`, live decoder off the embedding cache), alternating per-slot / within-frame attention | live decoder, 2 GPUs |
| 6 | `final_limb12` | six force limb tokens carrying decoder force tokens (`model.force`, `refiner.force_tokens`); the contact head reads the contact slots, the force head the force slots | live, 2 GPUs |
| 7 | `final_rnea` | the RNEA root-wrench residual as a loss (`force_consistency` force 5, torque 20) | live, 2 GPUs |
| 8 | `final_full` | strength-weighted joint torques (`joint_torque` 0.3) = the model | live, 2 GPUs |

Ablations off `final_full`: `final_nocontact` (no contact anywhere: no decoder contact tokens, no
contact limb tokens, no contact head or loss, RNEA ungated, stillness off — seven body + six force
slots), `final_rf_full` (no window), `final_rf_0.5`, `final_rf_0.125` (receptive field; the default
±0.25 s TOTAL = 3 layers × `window` 0.0833 s).

Schedule everywhere: 10 epochs, 120-frame clips, 32 clips per optimizer step, AdamW lr 3e-4 /
wd 0.01, warm-up 300 steps, cosine to 1e-6, EMA 0.999, eval every epoch; `best.pth` by contact F1,
every table reads `last.pth`. Cached rungs: 3840 frames per micro-batch, ~1 min per epoch. Live
rungs: 480 frames × 4 accumulated per GPU, ~16 min per epoch (2 h 50 min per run). Same supervised
force losses on every rung (force 1, magnitude 1, sum_force 1, noncontact 1, toe weights 2.5),
`force_frame: body`, `head_grad_scale 0.3` from rung 3.

## What changed in the code

* `model/token_heads.py`: `PoseTokenHeads`, zero-init FFN heads on the pose token in the refiner's
  output layout (gravity per frame: the camera's down axis plus a body-frame correction).
* Every build lifts its per-frame body into the world (`out["smplx"]["*_world"]`,
  `out["smplx_per_frame"]` kept when a refiner rewrites `out["smplx"]`), so the motion, stillness
  and RNEA losses / metrics run without a refiner. `smplx_supervision.loss` always scores the
  per-frame heads' body, `.refined` the refined one (`refined_<term>[_layer]`).
* `model.refiner.force_tokens` (force limb slots, alone or next to the contact ones),
  `model.refiner.window: null` (whole clip), `force_consistency` at zero weights = metrics only,
  the gravity loss averages per-frame estimates over the clip.
* Removed: `model.smplx.checkpoint` / `frozen`, `model.warm_start`, `optim.head_lr_scale`,
  `gravity_input`, `configs/stage1.yaml`, `configs/r10/`, `dump_stage1.py`, `analyze_stage1.py`
  (`../trash/ladder_cleanup_20260919/`).
* `scripts/score_rig.py` (new): pose vs the three-camera fit, contacts and forces vs the boards
  (vector MAE in body-weight percent with the reconstructed mass AND with the subject's measured
  mass, per-frame share error, RNEA residual of the predicted pose / contacts / forces).
  `predict_reconstruction.py --pred-dir <run>` writes `predictions/<run>/`; the round-10 rig dumps
  are in `../trash/cw3_predictions_20260919/`. `force_corr_share.py` reads the share per frame.
* Tests: 101 (7 new).

## Rung 2 as first defined collapsed the depth (`../trash/final_temporal_withlosses_20260919/`)

Rung 2 was meant to carry the velocity-matching and stillness losses too, measured on the
per-frame body lifted with the GT extrinsics on the loss side. That run ended at 106.1 mm MPJPE
with a depth BIAS of −336 mm (−959 mm after epoch 1, pelvis error 347 mm; the CLIFF `cam` term
3× the base rung's). Cached 10-epoch arms on the same block (scratch runs, not kept):

| arm | MPJPE | depth bias | jitter | F1 | force MAE |
|---|---|---|---|---|---|
| block, no temporal losses (= the rung 2 kept) | 58.0 | +5 | 89.1 | 0.890 | 0.173 |
| + velocity losses, no stillness | 90.2 | −161 | 66.9 | 0.872 | 0.180 |
| + stillness only | 69.8 | −180 | 67.5 | 0.873 | 0.179 |
| + angular velocity only, no stillness | 69.8 | +11 | 84.0 | 0.879 | 0.181 |
| + both + world root anchor (`loss.root_bias / root_shape` 2) | 102.4 | −20 | 69.2 | 0.870 | 0.180 |
| + both (the trashed run) | 106.1 | −336 | 62.2 | 0.871 | 0.180 |

Every loss that reads a metric motion off the unanchored per-frame body pushes the CLIFF depth
down (a closer body has smaller metric velocities for the same 2D track); anchoring the world root
stops the drift but not the damage. The same losses on the REFINED body of rungs 3 / 4 are
MPJPE-neutral (65.8 vs 67.2 mm, 61.0 vs 61.4) and are what removes the jitter (14.9 vs 82.0,
11.8 vs 74.3) and lowers the RNEA residual (0.37 vs 0.64, 0.35 vs 0.54), so they start at rung 3.

## Corpus test split (109 scenes, one clip per person, 120-frame cap, `last.pth`)

`output_6/logs/eval_table_20260919.txt`; corr / share from `scripts/force_corr_share.py` (per
frame).

| run | F1 | P | R | MPJPE | PA | PVE | accel | jitter | force MAE bw | angle | off-contact | RNEA f | RNEA τ | still | corr | share pp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final_base | 0.867 | 0.849 | 0.885 | 94.3 | 70.1 | 149.1 | 10.7 | 96.7 | 0.188 | 23.2 | 0.057 | 0.746 | 0.129 | – | 0.602 | 14.8 |
| final_temporal | 0.890 | 0.894 | 0.886 | 58.0 | 40.6 | 74.6 | 8.2 | 89.1 | 0.173 | 20.7 | 0.044 | 0.751 | 0.108 | – | 0.680 | 12.9 |
| final_world | 0.895 | 0.890 | 0.900 | 65.8 | 45.3 | 83.1 | 4.2 | 14.9 | 0.185 | 23.3 | 0.050 | 0.372 | 0.086 | 0.221 | 0.653 | 14.0 |
| final_iter | 0.901 | 0.901 | 0.902 | 61.0 | 41.8 | 78.7 | 3.9 | 11.8 | 0.180 | 21.9 | 0.046 | 0.345 | 0.082 | 0.201 | 0.675 | 13.4 |
| final_limb | 0.923 | 0.941 | 0.905 | 60.1 | 41.3 | 77.8 | 3.8 | 11.2 | 0.170 | 21.6 | 0.026 | 0.318 | 0.080 | 0.188 | 0.731 | 10.7 |
| final_limb12 | 0.922 | 0.941 | 0.905 | 60.4 | 41.4 | 78.0 | 3.8 | 11.2 | 0.170 | 21.2 | 0.030 | 0.331 | 0.083 | 0.191 | 0.716 | 11.4 |
| final_rnea | 0.921 | 0.938 | 0.904 | 63.5 | 43.5 | 81.3 | 3.8 | 11.1 | 0.171 | 22.0 | 0.033 | 0.295 | 0.055 | 0.196 | 0.710 | 11.4 |
| final_full | 0.922 | 0.940 | 0.905 | 62.9 | 43.1 | 80.5 | 3.8 | 11.0 | 0.168 | 21.1 | 0.034 | 0.305 | 0.053 | 0.194 | 0.712 | 11.3 |
| final_nocontact | – | – | – | 59.2 | 40.5 | 76.4 | 4.0 | 12.9 | 0.166 | 19.6 | 0.040 | 0.267 | 0.050 | – | 0.702 | 12.0 |
| final_rf_full | 0.922 | 0.940 | 0.904 | 61.8 | 42.5 | 79.9 | 3.8 | 11.6 | 0.167 | 21.2 | 0.034 | 0.298 | 0.054 | 0.173 | 0.712 | 11.4 |
| final_rf_0.5 | 0.924 | 0.943 | 0.907 | 61.8 | 42.2 | 79.7 | 3.8 | 10.7 | 0.167 | 21.1 | 0.033 | 0.294 | 0.052 | 0.177 | 0.720 | 11.1 |
| final_rf_0.125 | 0.918 | 0.934 | 0.903 | 60.9 | 41.6 | 78.5 | 3.9 | 12.1 | 0.168 | 21.2 | 0.035 | 0.314 | 0.054 | 0.212 | 0.708 | 11.5 |

Force MAE is the full 3D vector in body weights (× 100 = BW %). RNEA f / τ = the root-wrench
residual of the model's own pose, contacts (gate) and forces, bw / bw·m; the no-contact row is
ungated. `still` = in-contact extremity speed (GT floor ≈ 0.13 m/s). Gravity: 12.5° (limb, full)
to 13.3° (base) against the measured scenes; the camera axis alone reads 13.5°.

## climb_wall_3 rig (68 trials, left camera, `output_6/logs/rig_12runs_20260919.md`)

Pose vs the three-camera fit; contacts and forces vs the boards, our six groups folded onto the
four limbs (hand = hand, foot = toe + heel); `vec MAE bw%` reads the boards in body weights of the
RECONSTRUCTED body (the optimisation's mass), `GT mass` in body weights of the MEASURED subject
(72.5 / 48.0 kg) against our body-weight output as it is. The optimisation rows are Peter's solve
(their RNEA cells are the solve's own stored residual, not recomputed).

| run | MPJPE | PA | jitter | F1 | P | R | vec MAE bw% | GT mass | vec MAE N | size MAE N | angle | corr | share pp | RNEA f | RNEA τ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| optimisation | 57.5 | 32.9 | 10.2 | 0.916 | 0.902 | 0.931 | 16.7 | – | 94.5 | 80.1 | 14.7 | 0.654 | 10.8 | 0.087 | 0.019 |
| optimisation (GT contact) | 57.1 | 32.9 | 10.1 | 1.000 | 1.000 | 1.000 | 15.4 | – | 87.9 | 72.0 | 16.1 | 0.709 | 10.0 | 0.141 | 0.030 |
| final_base | 85.1 | 63.1 | 69.3 | 0.897 | 0.842 | 0.961 | 17.8 | 16.4 | 100.8 | 84.2 | 16.0 | 0.648 | 10.3 | 0.414 | 0.099 |
| final_temporal | 49.3 | 30.6 | 54.9 | 0.917 | 0.862 | 0.978 | 16.2 | 14.8 | 91.5 | 77.6 | 14.1 | 0.717 | 9.3 | 0.356 | 0.069 |
| final_world | 59.1 | 35.2 | 11.5 | 0.908 | 0.847 | 0.979 | 18.4 | 16.9 | 104.0 | 87.6 | 16.8 | 0.618 | 10.5 | 0.201 | 0.068 |
| final_iter | 55.9 | 31.4 | 11.0 | 0.912 | 0.851 | 0.982 | 17.9 | 16.4 | 101.1 | 85.4 | 16.0 | 0.644 | 10.2 | 0.196 | 0.062 |
| final_limb | 54.7 | 31.1 | 10.5 | 0.939 | 0.894 | 0.988 | 15.2 | 13.9 | 86.3 | 69.8 | 15.6 | 0.758 | 8.1 | 0.151 | 0.059 |
| final_limb12 | 54.8 | 31.5 | 10.6 | 0.939 | 0.895 | 0.987 | 15.8 | 14.4 | 89.3 | 72.8 | 15.7 | 0.738 | 8.6 | 0.166 | 0.059 |
| final_rnea | 57.7 | 33.5 | 10.7 | 0.938 | 0.893 | 0.987 | 16.0 | 14.7 | 90.8 | 73.6 | 15.6 | 0.720 | 8.9 | 0.148 | 0.039 |
| final_full | 58.1 | 33.1 | 10.6 | 0.938 | 0.893 | 0.987 | 15.5 | 14.1 | 87.6 | 71.3 | 15.1 | 0.742 | 8.5 | 0.147 | 0.037 |
| final_nocontact | 54.5 | 31.1 | 11.7 | – | – | – | 15.8 | 14.4 | 89.3 | 74.1 | 14.8 | 0.729 | 8.8 | 0.131 | 0.033 |
| final_rf_full | 55.9 | 32.2 | 11.1 | 0.941 | 0.899 | 0.988 | 15.6 | 14.2 | 88.3 | 72.5 | 15.2 | 0.748 | 8.5 | 0.164 | 0.037 |
| final_rf_0.5 | 56.1 | 31.8 | 10.6 | 0.938 | 0.894 | 0.987 | 15.4 | 14.0 | 87.2 | 70.8 | 15.2 | 0.748 | 8.4 | 0.147 | 0.036 |
| final_rf_0.125 | 55.6 | 31.5 | 10.5 | 0.937 | 0.892 | 0.988 | 15.7 | 14.3 | 88.8 | 72.6 | 15.2 | 0.734 | 8.7 | 0.157 | 0.035 |

## Reading

Raw observations. Round 5's seed spread (F1 0.002, MPJPE 0.02 mm) was measured on 16-30-epoch runs
over a frozen body; the from-scratch 10-epoch runs here are still moving at the last epoch (see the
trajectories), so treat small differences as unresolved.

* **Rung 2, the token block, is the largest single step on the pose**: 94.3 → 58.0 mm on the corpus,
  85.1 → 49.3 on the rig, and it is the best pose of the whole ladder; it also lifts contact
  (+0.023 F1, precision +0.045) and the forces (0.188 → 0.173). It does not smooth (jitter 89).
* **Rungs 3-4, the refiner, buy the trajectory**: jitter 89 → 15 → 12 (rig 55 → 11.5 → 11.0), accel
  8.2 → 3.9, the RNEA force residual halved (0.75 → 0.35), contact +0.011, at a pose COST at this
  schedule (58.0 → 65.8 → 61.0 mm; rig 49.3 → 59.1 → 55.9). Iteration recovers most of the pose
  the non-iterative refiner loses and adds 0.006 F1.
* **Rung 5, the contact limb tokens, is the contact and force step**: F1 0.901 → 0.923 with
  precision 0.90 → 0.94, off-contact force 0.046 → 0.026, force MAE 0.180 → 0.170, corr 0.68 →
  0.73, share 13.4 → 10.7 pp; on the rig F1 0.912 → 0.939 (the optimisation: 0.916), vector MAE
  101 → 86 N (94.5), share 10.2 → 8.1 pp (10.8). Pose −1 mm.
* **Rung 6, the force limb tokens**: nothing measurable (F1 −0.001, MAE 0.170, angle −0.4°;
  rig MAE +3 N, share +0.4 pp).
* **Rung 7, the RNEA loss**: torque residual 0.083 → 0.055 (rig 0.059 → 0.039), force residual
  0.33 → 0.30, otherwise flat on the forces (MAE 0.171, angle +0.8°) and −3 mm on the pose (60.4 →
  63.5; rig 54.8 → 57.7) — the round-10 reading again (RNEA = the residual, not the force error).
* **Rung 8, the joint torques**: MAE 0.171 → 0.168, angle 22.0 → 21.1°, rig share 8.9 → 8.5 pp,
  MAE 90.8 → 87.6 N; pose +0.6 mm.
* **No contact anywhere**: the forces are as good or better (MAE 0.166, angle 19.6°, rig 89 N /
  8.8 pp vs the full model's 0.168 / 21.1° / 88 N / 8.5 pp) and the pose is 3.7 mm better on the
  corpus and on the rig — with more off-contact force (0.040 vs 0.034) and no contact
  output; its RNEA residual is ungated, not comparable.
* **Receptive field**: full = 0.5 s = 0.25 s within noise on contact and forces (F1 0.922 /
  0.924 / 0.922, MAE 0.167-0.168); 0.125 s is the one that drops (F1 0.918, jitter 12.1, still
  0.212 vs 0.173-0.194). The full-clip window has the best world trajectory (WA 68.9, pelvis 98.5)
  and the worst stillness on the corpus but the best contact on the rig; on the rig the four fields
  are within 0.004 F1, 1.6 N and 0.3 pp of each other.
* **Schedule**: every cached rung is still improving steeply at epoch 10 (iter 64.8 → 62.0 → 61.0 mm
  over its last three epochs, base −2 mm per epoch); the round-10 model (frozen 30-epoch stage-1
  body) sat at 52.9 mm. The ladder's pose numbers are 10-epoch numbers.
* **GT mass**: reading the boards in the subject's measured body weight lowers every row by 1.4 bw
  points (14.1 % for the full model): the reconstructed bodies are lighter than the subjects.

## Trajectories worth keeping

Per-epoch test MPJPE of the cached rungs (epochs 1 … 10):

| run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| final_base | 207 | 142 | 122 | 112 | 107 | 105 | 101 | 98 | 96 | 94 |
| final_temporal | 195 | 132 | 96 | 77 | 69 | 64 | 61 | 60 | 58 | 58 |
| final_world | 199 | 141 | 128 | 124 | 118 | 103 | 84 | 73 | 68 | 66 |
| final_iter | 189 | 138 | 126 | 117 | 103 | 81 | 70 | 65 | 62 | 61 |

(the refiner rungs sit on a plateau for five epochs before the pose moves: the refined body's
losses only start to pay once the per-frame body is roughly right.)
