# Velocity-matching forensics — 16 static test scenes

All numbers raw; no interpretation. Sources: the frozen SAM3D SMPL-X refit and five
`scripts/predict_test.py` dumps. GT = the kindyn `kindyn_1.npz` world trajectory.


## 0. Protocol sanity check (reproducing the trainer's eval)

Row protocol sweep for **tvel_ray**: one clip per (scene, person) = the longest valid
run capped at N rows (`data.eval_max_frames`), the trainer's `full_scenes` eval protocol.
Predictions lifted with RAW cameras.  Trainer's reported eval (ep 8): r 0.53 / 0.79 / 0.53, gt_rms 0.288 / 0.486 / 0.752, rmse 0.30 / 0.30 / 0.64.

| cap (rows) | rows | root_vel r | root_ang r | joint_ang r | root_vel gt_rms | root_ang gt_rms | joint_ang gt_rms | root_vel rmse | root_ang rmse | joint_ang rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60 | 944 | 0.531 | 0.782 | 0.522 | 0.324 | 0.466 | 0.800 | 0.314 | 0.290 | 0.685 |
| 120 | 1902 | 0.528 | 0.795 | 0.526 | 0.288 | 0.486 | 0.752 | 0.301 | 0.296 | 0.641 |
| 180 | 2667 | 0.565 | 0.721 | 0.535 | 0.293 | 0.513 | 0.750 | 0.300 | 0.356 | 0.635 |
| 240 | 3190 | 0.595 | 0.676 | 0.550 | 0.328 | 0.554 | 0.782 | 0.328 | 0.409 | 0.655 |
| all | 3855 | 0.377 | 0.591 | 0.537 | 0.369 | 0.601 | 0.812 | 0.727 | 0.491 | 0.685 |

`cap 120` reproduces the trainer's numbers exactly (gt_rms 0.288 / 0.486 / 0.752, r 0.528 / 0.795 / 0.526, rmse 0.301 / 0.296 / 0.641).
Everything below uses the **FULL** protocol (every contiguous run, all rows) unless stated; the cap-120 column is repeated where the two disagree.


## (a) Velocity terms — `model/loss/velocity.py`, FULL protocol

Pooled over components and rows. `ratio` = RMS(pred)/RMS(gt). `huber` = the loss's own value, `mean smooth_l1(x/delta)` at delta = 0.4 / 0.6 / 0.9.


**root_vel (m/s)**

| source | pred RMS | GT RMS | ratio | r | RMSE | huber | cap120 ratio | cap120 r | cap120 RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen | 1.122 | 0.369 | 3.039 | 0.237 | 1.094 | 0.5110 | 1.782 | 0.385 | 0.478 |
| static_baseline | 1.058 | 0.369 | 2.867 | 0.302 | 1.010 | 0.5248 | 1.772 | 0.385 | 0.476 |
| static_ray | 0.944 | 0.369 | 2.558 | 0.310 | 0.901 | 0.1698 | 0.851 | 0.666 | 0.217 |
| tb_projzero | 0.944 | 0.369 | 2.556 | 0.320 | 0.896 | 0.1557 | 0.861 | 0.713 | 0.202 |
| tvel_ray | 0.782 | 0.369 | 2.118 | 0.377 | 0.727 | 0.2811 | 1.167 | 0.528 | 0.301 |
| tvel_cliff | 0.413 | 0.369 | 1.118 | 0.392 | 0.432 | 0.2091 | 0.696 | 0.482 | 0.256 |
| GT (self) | 0.369 | — | 1.000 | 1.000 | 0.000 | 0.0000 | — | — | — |

**root_ang_vel (rad/s)**

| source | pred RMS | GT RMS | ratio | r | RMSE | huber | cap120 ratio | cap120 r | cap120 RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen | 1.519 | 0.601 | 2.527 | 0.400 | 1.392 | 0.2921 | 1.338 | 0.684 | 0.476 |
| static_baseline | 1.069 | 0.601 | 1.779 | 0.548 | 0.894 | 0.2600 | 1.252 | 0.681 | 0.451 |
| static_ray | 1.128 | 0.601 | 1.876 | 0.524 | 0.960 | 0.2664 | 1.268 | 0.689 | 0.450 |
| tb_projzero | 1.129 | 0.601 | 1.879 | 0.507 | 0.974 | 0.2526 | 1.178 | 0.678 | 0.431 |
| tvel_ray | 0.433 | 0.601 | 0.720 | 0.591 | 0.491 | 0.1710 | 0.729 | 0.795 | 0.296 |
| tvel_cliff | 0.320 | 0.601 | 0.533 | 0.577 | 0.492 | 0.1964 | 0.614 | 0.744 | 0.331 |
| GT (self) | 0.601 | — | 1.000 | 1.000 | 0.000 | 0.0000 | — | — | — |

**joint_ang_vel (rad/s)**

| source | pred RMS | GT RMS | ratio | r | RMSE | huber | cap120 ratio | cap120 r | cap120 RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen | 1.611 | 0.812 | 1.984 | 0.270 | 1.596 | 0.5053 | 1.959 | 0.256 | 1.473 |
| static_baseline | 0.687 | 0.812 | 0.846 | 0.404 | 0.825 | 0.2631 | 0.788 | 0.394 | 0.752 |
| static_ray | 0.701 | 0.812 | 0.863 | 0.405 | 0.830 | 0.2658 | 0.801 | 0.398 | 0.753 |
| tb_projzero | 0.609 | 0.812 | 0.751 | 0.388 | 0.804 | 0.2549 | 0.682 | 0.391 | 0.726 |
| tvel_ray | 0.409 | 0.812 | 0.504 | 0.537 | 0.685 | 0.2015 | 0.479 | 0.526 | 0.641 |
| tvel_cliff | 0.362 | 0.812 | 0.445 | 0.504 | 0.703 | 0.2100 | 0.424 | 0.494 | 0.656 |
| GT (self) | 0.812 | — | 1.000 | 1.000 | 0.000 | 0.0000 | — | — | — |

**Same, with the outlier scene `R3KcQ9jBDvw_0011` dropped** (its tail carries a depth blow-up present in EVERY source, incl. frozen: root_vel pred RMS 3.1–3.2 m/s vs GT 0.55; it dominates the pooled root_vel of the full protocol).


**root_vel (m/s)** (15 scenes)

| source | pred RMS | GT RMS | ratio | r | RMSE | huber |
| --- | --- | --- | --- | --- | --- | --- |
| frozen | 0.762 | 0.350 | 2.177 | 0.286 | 0.741 | 0.4595 |
| static_baseline | 0.626 | 0.350 | 1.788 | 0.453 | 0.561 | 0.4608 |
| static_ray | 0.353 | 0.350 | 1.008 | 0.709 | 0.268 | 0.1339 |
| tb_projzero | 0.354 | 0.350 | 1.011 | 0.742 | 0.252 | 0.1171 |
| tvel_ray | 0.487 | 0.350 | 1.392 | 0.512 | 0.430 | 0.2469 |
| tvel_cliff | 0.263 | 0.350 | 0.752 | 0.519 | 0.310 | 0.1845 |

**root_ang_vel (rad/s)** (15 scenes)

| source | pred RMS | GT RMS | ratio | r | RMSE | huber |
| --- | --- | --- | --- | --- | --- | --- |
| frozen | 1.395 | 0.561 | 2.486 | 0.356 | 1.304 | 0.2742 |
| static_baseline | 0.797 | 0.561 | 1.422 | 0.591 | 0.649 | 0.2382 |
| static_ray | 0.888 | 0.561 | 1.583 | 0.540 | 0.752 | 0.2462 |
| tb_projzero | 0.882 | 0.561 | 1.571 | 0.522 | 0.759 | 0.2320 |
| tvel_ray | 0.405 | 0.561 | 0.721 | 0.576 | 0.466 | 0.1542 |
| tvel_cliff | 0.299 | 0.561 | 0.533 | 0.595 | 0.452 | 0.1764 |

**joint_ang_vel (rad/s)** (15 scenes)

| source | pred RMS | GT RMS | ratio | r | RMSE | huber |
| --- | --- | --- | --- | --- | --- | --- |
| frozen | 1.601 | 0.796 | 2.010 | 0.267 | 1.586 | 0.4966 |
| static_baseline | 0.663 | 0.796 | 0.833 | 0.410 | 0.801 | 0.2540 |
| static_ray | 0.677 | 0.796 | 0.850 | 0.409 | 0.807 | 0.2568 |
| tb_projzero | 0.585 | 0.796 | 0.735 | 0.395 | 0.780 | 0.2461 |
| tvel_ray | 0.394 | 0.796 | 0.494 | 0.544 | 0.670 | 0.1945 |
| tvel_cliff | 0.348 | 0.796 | 0.437 | 0.511 | 0.687 | 0.2028 |

### Per-component (GT body frame) + the world vertical

root_vel and root_ang_vel components 0/1/2 are the GT body-frame axes; `up` is the world component along `-gravity_world` of the same (transported) linear increments, rotated into the world by `R_gt`.  FULL protocol.


**root_vel (m/s) per component** — ratio / r / RMSE

| source | c0 ratio | c1 ratio | c2 ratio | c0 r | c1 r | c2 r | c0 RMSE | c1 RMSE | c2 RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen | 3.303 | 2.009 | 3.695 | 0.055 | 0.406 | 0.258 | 1.049 | 0.739 | 1.395 |
| static_baseline | 3.361 | 1.452 | 3.568 | 0.140 | 0.675 | 0.277 | 1.040 | 0.429 | 1.339 |
| static_ray | 2.207 | 1.097 | 3.633 | 0.174 | 0.872 | 0.240 | 0.698 | 0.216 | 1.379 |
| tb_projzero | 2.489 | 1.147 | 3.497 | 0.170 | 0.848 | 0.257 | 0.778 | 0.244 | 1.321 |
| tvel_ray | 2.402 | 1.415 | 2.509 | 0.245 | 0.643 | 0.323 | 0.730 | 0.436 | 0.929 |
| tvel_cliff | 1.332 | 0.668 | 1.331 | 0.261 | 0.776 | 0.307 | 0.445 | 0.256 | 0.545 |
| GT RMS | 0.309 | 0.401 | 0.391 | — | — | — | — | — | — |

**root_ang_vel (rad/s) per component** — ratio / r / RMSE

| source | c0 ratio | c1 ratio | c2 ratio | c0 r | c1 r | c2 r | c0 RMSE | c1 RMSE | c2 RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen | 1.830 | 1.931 | 4.626 | 0.487 | 0.601 | 0.135 | 0.823 | 1.252 | 1.888 |
| static_baseline | 1.449 | 1.994 | 1.300 | 0.511 | 0.573 | 0.531 | 0.653 | 1.324 | 0.469 |
| static_ray | 1.716 | 2.065 | 1.242 | 0.432 | 0.559 | 0.563 | 0.806 | 1.387 | 0.439 |
| tb_projzero | 1.479 | 2.151 | 1.188 | 0.462 | 0.526 | 0.572 | 0.692 | 1.479 | 0.421 |
| tvel_ray | 0.721 | 0.719 | 0.719 | 0.806 | 0.487 | 0.658 | 0.307 | 0.730 | 0.310 |
| tvel_cliff | 0.615 | 0.473 | 0.605 | 0.787 | 0.461 | 0.620 | 0.329 | 0.716 | 0.322 |
| GT RMS | 0.514 | 0.807 | 0.411 | — | — | — | — | — | — |

**root_vel world vertical component** (`v_world · (-gravity)`)

| source | pred RMS | GT RMS | ratio | r | RMSE |
| --- | --- | --- | --- | --- | --- |
| frozen | 0.856 | 0.437 | 1.957 | 0.348 | 0.814 |
| static_baseline | 0.491 | 0.437 | 1.123 | 0.849 | 0.260 |
| static_ray | 0.465 | 0.437 | 1.062 | 0.906 | 0.197 |
| tb_projzero | 0.487 | 0.437 | 1.112 | 0.896 | 0.216 |
| tvel_ray | 0.442 | 0.437 | 1.010 | 0.904 | 0.192 |
| tvel_cliff | 0.246 | 0.437 | 0.563 | 0.900 | 0.241 |

### joint_ang_vel per joint (21 body joints, FULL protocol)

| joint | GT RMS | frozen ratio | static_baseline ratio | static_ray ratio | tb_projzero ratio | tvel_ray ratio | tvel_cliff ratio | frozen r | static_baseline r | static_ray r | tb_projzero r | tvel_ray r | tvel_cliff r |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| left_hip | 0.826 | 1.191 | 0.981 | 0.982 | 0.909 | 0.627 | 0.429 | 0.525 | 0.540 | 0.544 | 0.531 | 0.649 | 0.574 |
| right_hip | 0.768 | 1.105 | 0.913 | 0.922 | 0.855 | 0.634 | 0.459 | 0.606 | 0.643 | 0.636 | 0.631 | 0.722 | 0.688 |
| spine1 | 0.467 | 0.937 | 0.645 | 0.647 | 0.604 | 0.453 | 0.302 | 0.289 | 0.337 | 0.345 | 0.329 | 0.414 | 0.381 |
| left_knee | 1.115 | 1.315 | 1.064 | 1.068 | 0.889 | 0.626 | 0.611 | 0.537 | 0.555 | 0.561 | 0.523 | 0.715 | 0.692 |
| right_knee | 1.011 | 1.139 | 0.904 | 0.947 | 0.822 | 0.629 | 0.629 | 0.600 | 0.578 | 0.585 | 0.554 | 0.734 | 0.714 |
| spine2 | 0.326 | 1.323 | 0.591 | 0.574 | 0.524 | 0.444 | 0.271 | 0.223 | 0.270 | 0.273 | 0.273 | 0.386 | 0.312 |
| left_ankle | 0.939 | 1.065 | 0.476 | 0.484 | 0.450 | 0.304 | 0.261 | 0.229 | 0.192 | 0.204 | 0.173 | 0.347 | 0.301 |
| right_ankle | 0.827 | 1.155 | 0.544 | 0.586 | 0.503 | 0.318 | 0.255 | 0.239 | 0.201 | 0.205 | 0.167 | 0.323 | 0.299 |
| spine3 | 0.411 | 0.851 | 0.612 | 0.577 | 0.484 | 0.405 | 0.377 | 0.175 | 0.238 | 0.240 | 0.233 | 0.372 | 0.293 |
| left_foot | 1.073 | 0.742 | 0.453 | 0.480 | 0.423 | 0.206 | 0.171 | 0.132 | 0.200 | 0.203 | 0.195 | 0.249 | 0.214 |
| right_foot | 0.896 | 0.898 | 0.434 | 0.446 | 0.391 | 0.187 | 0.146 | 0.114 | 0.161 | 0.165 | 0.139 | 0.215 | 0.177 |
| neck | 0.677 | 1.125 | 0.813 | 0.810 | 0.644 | 0.395 | 0.306 | 0.306 | 0.257 | 0.246 | 0.184 | 0.438 | 0.404 |
| left_collar | 0.620 | 1.296 | 0.912 | 0.960 | 0.898 | 0.568 | 0.431 | 0.349 | 0.406 | 0.401 | 0.398 | 0.504 | 0.477 |
| right_collar | 0.546 | 1.248 | 0.931 | 0.921 | 0.829 | 0.515 | 0.461 | 0.292 | 0.326 | 0.317 | 0.293 | 0.422 | 0.402 |
| head | 0.661 | 0.807 | 0.652 | 0.709 | 0.563 | 0.261 | 0.185 | 0.100 | 0.162 | 0.137 | 0.146 | 0.230 | 0.189 |
| left_shoulder | 0.882 | 1.451 | 0.970 | 1.021 | 0.974 | 0.627 | 0.592 | 0.417 | 0.482 | 0.480 | 0.479 | 0.624 | 0.582 |
| right_shoulder | 0.837 | 1.431 | 0.993 | 1.029 | 0.966 | 0.634 | 0.549 | 0.415 | 0.482 | 0.480 | 0.461 | 0.609 | 0.589 |
| left_elbow | 1.098 | 2.012 | 1.168 | 1.168 | 0.985 | 0.655 | 0.612 | 0.391 | 0.511 | 0.514 | 0.512 | 0.695 | 0.660 |
| right_elbow | 0.941 | 2.170 | 1.080 | 1.098 | 0.849 | 0.605 | 0.531 | 0.353 | 0.512 | 0.501 | 0.478 | 0.674 | 0.637 |
| left_wrist | 0.827 | 4.639 | 0.524 | 0.521 | 0.440 | 0.235 | 0.188 | 0.251 | 0.129 | 0.130 | 0.119 | 0.242 | 0.209 |
| right_wrist | 0.677 | 6.062 | 0.641 | 0.648 | 0.523 | 0.257 | 0.213 | 0.187 | 0.095 | 0.097 | 0.076 | 0.219 | 0.194 |

## (b) Frequency split of the velocity series (Gaussian sigma 0.2 s along time)

Each velocity series is low-passed per contiguous run (`scipy.gaussian_filter1d`, `mode='nearest'`); HF = series - LF. Ratios are RMS(pred_part)/RMS(gt_part); RMSE_LF and RMSE_HF are RMS(pred_part - gt_part) (they do NOT add in quadrature to the full RMSE — the Gaussian split is not an orthogonal projection).  FULL protocol.


**root_vel (m/s)** — GT RMS: LF 0.280, HF 0.169

| source | LF ratio | LF r | LF RMSE | HF ratio | HF r | HF RMSE | full RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| frozen | 3.385 | 0.208 | 0.930 | 6.045 | 0.136 | 1.013 | 1.094 |
| static_baseline | 3.233 | 0.359 | 0.846 | 5.639 | 0.130 | 0.947 | 1.010 |
| static_ray | 3.256 | 0.336 | 0.859 | 5.056 | 0.134 | 0.849 | 0.901 |
| tb_projzero | 3.265 | 0.346 | 0.859 | 5.035 | 0.138 | 0.845 | 0.896 |
| tvel_ray | 2.428 | 0.468 | 0.601 | 4.212 | 0.093 | 0.717 | 0.727 |
| tvel_cliff | 1.238 | 0.489 | 0.321 | 2.154 | 0.102 | 0.386 | 0.432 |

**root_ang_vel (rad/s)** — GT RMS: LF 0.479, HF 0.255

| source | LF ratio | LF r | LF RMSE | HF ratio | HF r | HF RMSE | full RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| frozen | 2.180 | 0.430 | 0.943 | 6.042 | 0.181 | 1.513 | 1.392 |
| static_baseline | 1.742 | 0.626 | 0.651 | 3.714 | 0.239 | 0.919 | 0.894 |
| static_ray | 1.790 | 0.592 | 0.691 | 4.033 | 0.222 | 1.002 | 0.960 |
| tb_projzero | 1.781 | 0.583 | 0.693 | 4.083 | 0.205 | 1.019 | 0.974 |
| tvel_ray | 0.637 | 0.691 | 0.347 | 1.029 | 0.377 | 0.289 | 0.491 |
| tvel_cliff | 0.444 | 0.667 | 0.372 | 0.796 | 0.359 | 0.262 | 0.492 |

**joint_ang_vel (rad/s)** — GT RMS: LF 0.392, HF 0.598

| source | LF ratio | LF r | LF RMSE | HF ratio | HF r | HF RMSE | full RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| frozen | 0.921 | 0.657 | 0.313 | 2.602 | 0.192 | 1.557 | 1.596 |
| static_baseline | 0.763 | 0.619 | 0.313 | 0.979 | 0.265 | 0.717 | 0.825 |
| static_ray | 0.772 | 0.621 | 0.313 | 0.999 | 0.267 | 0.724 | 0.830 |
| tb_projzero | 0.727 | 0.578 | 0.325 | 0.856 | 0.249 | 0.684 | 0.804 |
| tvel_ray | 0.565 | 0.677 | 0.292 | 0.496 | 0.397 | 0.552 | 0.685 |
| tvel_cliff | 0.490 | 0.646 | 0.306 | 0.445 | 0.369 | 0.558 | 0.703 |

## (c) Amplitude of the POSES themselves

Per scene: std over time of the pelvis CAMERA-frame position per axis (mm), RMS over time of the root orientation deviation `|log(R_mean^T R_t)|` (rad, `R_mean` = chordal mean), and the per-frame mean bone length of the 21 body joints (mm). Pooled over the 16 scenes as an RMS of the per-scene values (bone length: plain mean).

| source | pelvis cam std x (mm) | y | z | root rot dev RMS (rad) | mean bone (mm) | bone ratio |
| --- | --- | --- | --- | --- | --- | --- |
| frozen | 213.1 | 298.8 | 227.5 | 0.588 | 192.1 | 1.001 |
| static_baseline | 219.3 | 301.3 | 231.5 | 0.576 | 191.3 | 0.997 |
| static_ray | 215.9 | 294.0 | 166.2 | 0.580 | 191.6 | 0.999 |
| tb_projzero | 214.9 | 295.8 | 189.5 | 0.563 | 191.7 | 0.999 |
| tvel_ray | 205.5 | 285.2 | 187.0 | 0.345 | 188.0 | 0.980 |
| tvel_cliff | 107.2 | 158.0 | 105.5 | 0.267 | 166.9 | 0.870 |
| GT | 216.4 | 298.9 | 221.4 | 0.592 | 191.8 | 1.000 |

**Ratios pred/GT of the same quantities**

| source | pelvis std x | y | z | root rot dev | bone length |
| --- | --- | --- | --- | --- | --- |
| frozen | 0.985 | 1.000 | 1.027 | 0.994 | 1.001 |
| static_baseline | 1.013 | 1.008 | 1.046 | 0.974 | 0.997 |
| static_ray | 0.998 | 0.984 | 0.751 | 0.980 | 0.999 |
| tb_projzero | 0.993 | 0.990 | 0.856 | 0.952 | 0.999 |
| tvel_ray | 0.950 | 0.954 | 0.845 | 0.583 | 0.980 |
| tvel_cliff | 0.495 | 0.529 | 0.477 | 0.450 | 0.870 |

### Per-joint rotation spread about the chordal mean (rad), pred / GT ratio

| joint | GT dev RMS | frozen | static_baseline | static_ray | tb_projzero | tvel_ray | tvel_cliff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| left_hip | 0.584 | 0.872 | 0.812 | 0.817 | 0.780 | 0.640 | 0.389 |
| right_hip | 0.475 | 0.915 | 0.898 | 0.894 | 0.866 | 0.719 | 0.465 |
| spine1 | 0.206 | 0.830 | 0.737 | 0.739 | 0.705 | 0.627 | 0.435 |
| left_knee | 0.621 | 0.839 | 0.810 | 0.815 | 0.759 | 0.630 | 0.651 |
| right_knee | 0.524 | 0.852 | 0.820 | 0.824 | 0.763 | 0.685 | 0.727 |
| spine2 | 0.164 | 0.816 | 0.706 | 0.669 | 0.646 | 0.618 | 0.371 |
| left_ankle | 0.325 | 0.654 | 0.607 | 0.630 | 0.610 | 0.477 | 0.413 |
| right_ankle | 0.352 | 0.556 | 0.588 | 0.620 | 0.563 | 0.413 | 0.334 |
| spine3 | 0.178 | 0.510 | 0.742 | 0.707 | 0.657 | 0.625 | 0.556 |
| left_foot | 0.424 | 0.455 | 0.511 | 0.526 | 0.494 | 0.307 | 0.244 |
| right_foot | 0.389 | 0.389 | 0.443 | 0.473 | 0.429 | 0.229 | 0.165 |
| neck | 0.348 | 0.721 | 0.585 | 0.600 | 0.479 | 0.411 | 0.302 |
| left_collar | 0.313 | 0.931 | 0.901 | 0.912 | 0.882 | 0.681 | 0.483 |
| right_collar | 0.301 | 0.926 | 0.931 | 0.907 | 0.838 | 0.631 | 0.560 |
| head | 0.314 | 0.521 | 0.587 | 0.649 | 0.591 | 0.340 | 0.247 |
| left_shoulder | 0.490 | 0.902 | 0.836 | 0.890 | 0.879 | 0.696 | 0.668 |
| right_shoulder | 0.518 | 0.884 | 0.807 | 0.815 | 0.792 | 0.653 | 0.563 |
| left_elbow | 0.518 | 0.898 | 0.784 | 0.790 | 0.753 | 0.657 | 0.632 |
| right_elbow | 0.521 | 0.942 | 0.802 | 0.814 | 0.750 | 0.669 | 0.594 |
| left_wrist | 0.315 | 1.103 | 0.497 | 0.515 | 0.495 | 0.282 | 0.221 |
| right_wrist | 0.274 | 1.183 | 0.650 | 0.680 | 0.621 | 0.378 | 0.326 |

## (d) Pelvis depth

`bias` = mean(pred z - GT z) per scene, then averaged over the 16 scenes; `|err|` = the same with the absolute value; `dlogz` = RMS of the per-step difference of log(pelvis camera z) in %/step at the dump's stride.

| source | depth bias (mm) | depth abs err (mm) | dlogz (%/step) |
| --- | --- | --- | --- |
| frozen | -17.8 | 98.6 | 0.757 |
| static_baseline | 59.4 | 129.4 | 0.725 |
| static_ray | 28.4 | 113.7 | 0.272 |
| tb_projzero | 31.7 | 99.7 | 0.247 |
| tvel_ray | -56.9 | 250.9 | 0.522 |
| tvel_cliff | -2109.8 | 2109.8 | 0.576 |
| GT | 0.0 | 0.0 | 0.331 |

### Per-scene pelvis depth bias (mm)

| scene | GT depth (m) | frozen | static_baseline | static_ray | tb_projzero | tvel_ray | tvel_cliff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0NUdUQZFGjs_0003/0 | 3.13 | 81 | 95 | 62 | 56 | 202 | -1572 |
| 0ZL8elIP9uo_0005/0 | 5.00 | -92 | 88 | -31 | 24 | -259 | -2662 |
| 4HuRoofxxMI_0002/0 | 3.14 | -21 | -9 | -31 | -42 | 128 | -1596 |
| BFFCd9gLmXo_0020/0 | 3.12 | 39 | 148 | 95 | 37 | 161 | -1544 |
| ObOaKNMmG5U_0005/0 | 5.28 | -224 | -41 | -167 | -149 | -327 | -2778 |
| ObOaKNMmG5U_0006/0 | 5.92 | -139 | 5 | -90 | -52 | -370 | -3110 |
| PTX9roUkC0o_0000/0 | 3.94 | -6 | 114 | 101 | 73 | -0 | -2108 |
| R3KcQ9jBDvw_0009/0 | 7.20 | -91 | 34 | -57 | 6 | -934 | -3957 |
| R3KcQ9jBDvw_0011/0 | 5.59 | -72 | -3 | -19 | 16 | -446 | -3032 |
| RVL7DuOL9EU_0114/0 | 3.89 | 21 | 55 | -18 | -7 | 105 | -1974 |
| US6c-J7Rlls_0000/0 | 2.35 | 20 | 55 | 131 | 91 | 190 | -1202 |
| US6c-J7Rlls_0002/0 | 2.41 | 53 | 36 | 151 | 115 | 186 | -1238 |
| UTaaFuZjruc_0008/0 | 4.76 | -38 | 105 | 49 | -2 | -59 | -2419 |
| Ul96DmN2M3s_0011/0 | 2.99 | 88 | 93 | 132 | 197 | 239 | -1533 |
| Ul96DmN2M3s_0013/0 | 3.50 | 37 | 126 | 76 | 90 | 78 | -1788 |
| s-ArwEzr-2M_0025/0 | 2.47 | 57 | 51 | 71 | 55 | 197 | -1244 |

### Per-scene dlogz (%/step)

| scene | GT | frozen | static_baseline | static_ray | tb_projzero | tvel_ray | tvel_cliff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0NUdUQZFGjs_0003/0 | 0.087 | 0.603 | 0.344 | 0.099 | 0.089 | 0.298 | 0.253 |
| 0ZL8elIP9uo_0005/0 | 0.456 | 0.711 | 0.972 | 0.333 | 0.170 | 0.401 | 0.424 |
| 4HuRoofxxMI_0002/0 | 0.224 | 0.513 | 0.520 | 0.161 | 0.140 | 0.366 | 0.579 |
| BFFCd9gLmXo_0020/0 | 0.173 | 0.426 | 0.404 | 0.104 | 0.109 | 0.308 | 0.406 |
| ObOaKNMmG5U_0005/0 | 0.436 | 0.597 | 0.716 | 0.196 | 0.230 | 0.401 | 0.471 |
| ObOaKNMmG5U_0006/0 | 0.134 | 0.524 | 0.561 | 0.148 | 0.131 | 0.335 | 0.402 |
| PTX9roUkC0o_0000/0 | 0.269 | 1.615 | 1.086 | 0.556 | 0.616 | 1.309 | 1.165 |
| R3KcQ9jBDvw_0009/0 | 0.141 | 0.966 | 0.867 | 0.223 | 0.176 | 0.529 | 0.716 |
| R3KcQ9jBDvw_0011/0 | 0.305 | 0.723 | 0.868 | 0.467 | 0.331 | 0.477 | 0.608 |
| RVL7DuOL9EU_0114/0 | 0.454 | 0.468 | 0.478 | 0.102 | 0.092 | 0.281 | 0.433 |
| US6c-J7Rlls_0000/0 | 0.566 | 0.917 | 0.946 | 0.355 | 0.255 | 0.556 | 0.824 |
| US6c-J7Rlls_0002/0 | 0.162 | 0.682 | 0.433 | 0.157 | 0.110 | 0.410 | 0.296 |
| UTaaFuZjruc_0008/0 | 0.125 | 0.390 | 0.426 | 0.084 | 0.089 | 0.246 | 0.302 |
| Ul96DmN2M3s_0011/0 | 0.451 | 0.708 | 0.885 | 0.336 | 0.234 | 0.521 | 0.548 |
| Ul96DmN2M3s_0013/0 | 0.496 | 0.695 | 0.796 | 0.250 | 0.355 | 0.510 | 0.539 |
| s-ArwEzr-2M_0025/0 | 0.204 | 0.705 | 0.707 | 0.221 | 0.202 | 0.491 | 0.547 |

## (e) Pelvis(mean-hips)-aligned MPJPE split into LF / HF (sigma 0.2 s)

The aligned body-22 joint positions of pred and GT are each split into LF and HF along time; `MPJPE_LF` = mean joint distance between the LF parts, `MPJPE_HF` between the HF parts. `MPJPE` is the unsplit value (the repo's metric, world frame = camera frame under the rigid lift).  FULL protocol, 3871 person-frames.

| source | MPJPE (mm) | MPJPE_LF (mm) | MPJPE_HF (mm) | d MPJPE vs frozen | d LF vs frozen | d HF vs frozen | d LF vs static_ray | d HF vs static_ray |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen | 59.9 | 54.1 | 16.4 | 0.0 | 0.0 | 0.0 | -10.8 | -1.0 |
| static_baseline | 66.5 | 60.4 | 17.2 | 6.7 | 6.4 | 0.8 | -4.4 | -0.2 |
| static_ray | 70.8 | 64.8 | 17.4 | 10.9 | 10.8 | 1.0 | 0.0 | 0.0 |
| tb_projzero | 76.0 | 70.0 | 17.4 | 16.1 | 16.0 | 1.1 | 5.2 | 0.1 |
| tvel_ray | 114.6 | 110.0 | 16.7 | 54.8 | 55.9 | 0.3 | 45.2 | -0.7 |
| tvel_cliff | 270.7 | 264.8 | 21.6 | 210.9 | 210.8 | 5.2 | 200.0 | 4.2 |

### Per-scene MPJPE / MPJPE_HF (mm)

| scene | rows | frozen mpjpe | static_baseline mpjpe | static_ray mpjpe | tb_projzero mpjpe | tvel_ray mpjpe | tvel_cliff mpjpe | frozen HF | static_baseline HF | static_ray HF | tb_projzero HF | tvel_ray HF | tvel_cliff HF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0NUdUQZFGjs_0003/0 | 118 | 47.7 | 51.4 | 54.6 | 60.3 | 94.7 | 275.3 | 10.1 | 9.3 | 9.2 | 9.3 | 7.8 | 9.9 |
| 0ZL8elIP9uo_0005/0 | 272 | 52.7 | 52.8 | 60.1 | 65.2 | 82.9 | 276.2 | 14.7 | 14.8 | 14.7 | 14.3 | 13.2 | 17.1 |
| 4HuRoofxxMI_0002/0 | 341 | 47.1 | 57.2 | 57.9 | 63.3 | 113.0 | 254.6 | 10.6 | 11.3 | 11.9 | 12.2 | 10.9 | 16.4 |
| BFFCd9gLmXo_0020/0 | 163 | 60.4 | 56.4 | 60.5 | 58.8 | 99.8 | 256.2 | 9.9 | 10.8 | 10.8 | 10.2 | 10.6 | 15.0 |
| ObOaKNMmG5U_0005/0 | 373 | 77.0 | 84.1 | 96.3 | 97.2 | 127.3 | 277.1 | 21.9 | 22.6 | 23.0 | 22.8 | 22.7 | 28.1 |
| ObOaKNMmG5U_0006/0 | 373 | 75.2 | 75.9 | 79.6 | 79.2 | 147.2 | 294.0 | 17.7 | 19.6 | 20.1 | 19.5 | 17.8 | 21.3 |
| PTX9roUkC0o_0000/0 | 425 | 54.5 | 70.8 | 70.2 | 71.2 | 106.8 | 266.0 | 18.7 | 18.5 | 18.4 | 18.5 | 17.6 | 24.7 |
| R3KcQ9jBDvw_0009/0 | 265 | 63.8 | 74.1 | 82.2 | 99.7 | 163.9 | 321.9 | 22.9 | 23.9 | 24.0 | 26.1 | 27.0 | 32.3 |
| R3KcQ9jBDvw_0011/0 | 288 | 64.2 | 68.4 | 68.1 | 79.1 | 142.4 | 291.2 | 22.2 | 25.0 | 24.9 | 25.2 | 26.1 | 34.3 |
| RVL7DuOL9EU_0114/0 | 188 | 59.8 | 68.6 | 72.9 | 76.3 | 96.4 | 244.2 | 18.6 | 19.6 | 19.8 | 19.5 | 18.6 | 22.8 |
| US6c-J7Rlls_0000/0 | 152 | 58.3 | 57.1 | 62.0 | 69.4 | 93.4 | 254.9 | 18.7 | 19.5 | 19.7 | 20.4 | 18.1 | 23.1 |
| US6c-J7Rlls_0002/0 | 215 | 47.9 | 55.9 | 67.4 | 59.7 | 88.6 | 263.7 | 8.7 | 8.4 | 8.4 | 8.2 | 7.0 | 8.6 |
| UTaaFuZjruc_0008/0 | 148 | 68.3 | 75.5 | 78.9 | 82.3 | 92.4 | 233.2 | 13.2 | 13.0 | 13.2 | 12.7 | 11.7 | 14.2 |
| Ul96DmN2M3s_0011/0 | 248 | 60.9 | 62.3 | 61.9 | 75.5 | 96.4 | 242.2 | 17.8 | 17.9 | 18.6 | 18.7 | 17.5 | 21.5 |
| Ul96DmN2M3s_0013/0 | 153 | 57.4 | 79.1 | 79.6 | 91.5 | 137.4 | 281.3 | 11.7 | 14.2 | 14.3 | 14.7 | 13.6 | 19.6 |
| s-ArwEzr-2M_0025/0 | 149 | 40.9 | 47.0 | 53.3 | 67.9 | 86.8 | 252.8 | 10.3 | 10.9 | 11.0 | 10.8 | 10.1 | 15.1 |

## Camera-smoothing variant (tvel_ray)

The tvel runs trained and evaluated with Gaussian-smoothed cameras (`data.camera_smooth_sec: 0.25`, `data/climbing_videos/camera.py::smooth_cameras`); the tables above lift every source with the RAW `cam_from_world`. Measured effect of the smoothing on a static scene: camera centre moves 0.5 mm mean / 2.1 mm max, |dR|_F 2.5e-4 mean.

| variant | root_vel ratio | root_ang_vel ratio | joint_ang_vel ratio | root_vel r | root_ang_vel r | joint_ang_vel r | root_vel RMSE | root_ang_vel RMSE | joint_ang_vel RMSE | MPJPE (mm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tvel_ray | 2.118 | 0.720 | 0.504 | 0.377 | 0.591 | 0.537 | 0.727 | 0.491 | 0.685 | 114.6 |
| tvel_ray@cams0.25 | 2.032 | 0.722 | 0.504 | 0.390 | 0.592 | 0.537 | 0.694 | 0.491 | 0.685 | 114.6 |

## What I measured / caveats


* **Sources.** GT = `features/human_optim/<shard>/<scene>/kindyn_1.npz` (`q` world +
  `joints_world`, the very arrays `data/climbing_videos/kindyn.py::load_smplx` hands the
  velocity loss). Frozen = `features/sam3d/<shard>/<scene>/smplx_params.npz` through
  `viewer/bodies.py::frozen_source` (classic params -> BetterHuman `q` -> exact per-frame FK ->
  world). Runs = `output/<run>/predictions/<scene>.npz` (`q_cam` = pelvis position, root
  quaternion xyzw, 21 body quaternions, 30 finger quaternions; `joints_cam`), lifted with
  `world_from_cam`.
* **New dumps.** `tvel_ray` (best.pth, epoch 8) and `tvel_cliff` (best.pth, epoch 3) were
  dumped for this analysis with `scripts/predict_test.py` defaults (240-row windows, overlap
  120, auto stride) on GPU 4; logs `predict_tvel_*.log`. The three older dumps
  (`static_baseline` ep29, `static_ray` ep29, `tb_projzero` ep27) are the existing ones.
* **Rows.** Per scene the stride grid is `arange(0, N, stride)` at the dump's own stride
  (auto = fps/25 -> 1 for the 24-30 fps scenes, 2 for the single 60 fps scene). A row enters a
  statistic when the source and the GT are both valid; contiguous runs of at least 12 rows are
  processed separately so no stencil and no Gaussian filter crosses a gap. On these 16 scenes
  every person-frame is valid, so there is exactly ONE run per scene (one person per scene),
  3871 frames / 3855 velocity rows in total.
* **Velocity.** Recomputed with BetterRobot `se3`/`so3` in float64, term by term as
  `model/loss/velocity.py` writes them: `d[t] = se3.log(T_t^-1 T_t+1)/dt` for the root (layout
  [v, omega], body frame at t), `so3.log(R_t^T R_t+1)/dt` for the 21 parent-local joints, the
  predicted increments transported into the GT frame with `E_t = q_gt^-1 q_pred`. The loss runs
  in float32; nothing here depends on that at the reported precision.
* **Protocol.** The trainer's eval is `full_scenes=True, max_frames=data.eval_max_frames` = ONE
  clip per (scene, person): the longest valid run, first 120 rows. That reproduces its
  published numbers exactly (section 0). The FULL protocol adds every later row, and on
  root_vel it is dominated by one scene (`R3KcQ9jBDvw_0011`, rows 120-288) where every source
  including frozen has a depth blow-up; the 15-scene table is given alongside.
* **Cameras.** Everything is lifted with the RAW `cam_from_world` from
  `features/geometry/.../transform.npz`. The tvel runs trained and evaluated with the cameras
  Gaussian-smoothed at sigma 0.25 s; the last section repeats tvel_ray both ways (the
  difference is ~mm on static scenes and does not move any r or the MPJPE).
* **Frequency split.** `scipy.ndimage.gaussian_filter1d(sigma=0.2 s / dt, mode='nearest',
  truncate=4)` along time, per run; HF = signal - LF. sigma is 4.8-6.0 frames here. The split is
  not orthogonal, so the LF and HF RMSEs do not add in quadrature to the full RMSE.
* **Amplitude.** `pelvis cam std` is the std over time of the pelvis in the CAMERA frame (so it
  mixes real motion and per-frame noise). Root/joint rotation spread uses the chordal
  (Frobenius/SVD) mean rotation of each run as the reference. Bone length = the mean over the 21
  body joints of `|joint - parent|` per frame, then over frames — a scale proxy that is
  invariant to the lift.
* **What I did NOT compute.** (i) No per-scene confidence intervals or significance tests —
  16 scenes, one person each. (ii) `tvel_cliff`'s dump is epoch 3 (the run was stopped at
  epoch 5); `tvel_ray`'s is epoch 8. They are not epoch-matched to the 27-29-epoch baselines.
  (iii) The frozen row is the corpus SMPL-X REFIT of the frozen model, not the frozen MHR
  readout. (iv) Nothing here separates the block from the head: the dumps are the joint output.

