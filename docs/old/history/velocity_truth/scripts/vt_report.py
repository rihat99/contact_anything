"""Render ``forensics.json`` into ``forensics_results.md`` (tables only, no verdicts)."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

import vt_common as vt
from vt_forensics import CAPS, TERMS, reduce_all

OUT = Path(__file__).resolve().parent
LBLS = ["frozen", "static_baseline", "static_ray", "tb_projzero", "tvel_ray", "tvel_cliff"]
OUTLIER = "R3KcQ9jBDvw_0011/0"
NICE = {"root_vel": "root_vel (m/s)", "root_ang_vel": "root_ang_vel (rad/s)",
        "joint_ang_vel": "joint_ang_vel (rad/s)"}


def joint_names() -> list[str]:
    from viewer.bodies import load_body
    return list(load_body("cpu").structure.joint_names)[1:22]


def row(cells) -> str:
    return "| " + " | ".join(cells) + " |"


def head(cols) -> str:
    return row(cols) + "\n" + row(["---"] * len(cols))


def f(x, n=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def main() -> None:
    r = json.loads((OUT / "forensics.json").read_text())
    data = pickle.loads((OUT / "collected.pkl").read_bytes())
    r_ex = reduce_all(data, drop=(OUTLIER,))
    names = joint_names()
    L: list[str] = []
    A = L.append

    A("# Velocity-matching forensics — 16 static test scenes\n")
    A("All numbers raw; no interpretation. Sources: the frozen SAM3D SMPL-X refit and five\n"
      "`scripts/predict_test.py` dumps. GT = the kindyn `kindyn_1.npz` world trajectory.\n")

    # ---------------- 0. protocol / sanity
    A("\n## 0. Protocol sanity check (reproducing the trainer's eval)\n")
    A("Row protocol sweep for **tvel_ray**: one clip per (scene, person) = the longest valid\n"
      "run capped at N rows (`data.eval_max_frames`), the trainer's `full_scenes` eval protocol.\n"
      "Predictions lifted with RAW cameras.  Trainer's reported eval (ep 8): "
      "r 0.53 / 0.79 / 0.53, gt_rms 0.288 / 0.486 / 0.752, rmse 0.30 / 0.30 / 0.64.\n")
    A(head(["cap (rows)", "rows", "root_vel r", "root_ang r", "joint_ang r",
            "root_vel gt_rms", "root_ang gt_rms", "joint_ang gt_rms",
            "root_vel rmse", "root_ang rmse", "joint_ang rmse"]))
    for cap in CAPS:
        c = r["tvel_ray"]["caps"][f"cap{cap}"]
        A(row([("all" if cap > 10 ** 5 else str(cap)), str(c["root_vel"]["n"] // 3)]
              + [f(c[t]["r"]) for t in TERMS] + [f(c[t]["gt_rms"]) for t in TERMS]
              + [f(c[t]["rmse"]) for t in TERMS]))
    A("\n`cap 120` reproduces the trainer's numbers exactly (gt_rms 0.288 / 0.486 / 0.752, "
      "r 0.528 / 0.795 / 0.526, rmse 0.301 / 0.296 / 0.641).\n"
      "Everything below uses the **FULL** protocol (every contiguous run, all rows) unless "
      "stated; the cap-120 column is repeated where the two disagree.\n")

    # ---------------- a. velocity terms
    A("\n## (a) Velocity terms — `model/loss/velocity.py`, FULL protocol\n")
    A("Pooled over components and rows. `ratio` = RMS(pred)/RMS(gt). `huber` = the loss's own "
      "value, `mean smooth_l1(x/delta)` at delta = 0.4 / 0.6 / 0.9.\n")
    for t in TERMS:
        A(f"\n**{NICE[t]}**\n")
        A(head(["source", "pred RMS", "GT RMS", "ratio", "r", "RMSE", "huber",
                "cap120 ratio", "cap120 r", "cap120 RMSE"]))
        for lbl in LBLS:
            v, c = r[lbl]["velocity"][t], r[lbl]["caps"]["cap120"][t]
            A(row([lbl, f(v["pred_rms"]), f(v["gt_rms"]), f(v["ratio"]), f(v["r"]), f(v["rmse"]),
                   f(v["huber"], 4), f(c["ratio"]), f(c["r"]), f(c["rmse"])]))
        A(row(["GT (self)", f(r[LBLS[0]]['velocity'][t]['gt_rms']), "—", "1.000", "1.000",
               "0.000", "0.0000", "—", "—", "—"]))

    A("\n**Same, with the outlier scene `R3KcQ9jBDvw_0011` dropped** (its tail carries a "
      "depth blow-up present in EVERY source, incl. frozen: root_vel pred RMS 3.1–3.2 m/s "
      "vs GT 0.55; it dominates the pooled root_vel of the full protocol).\n")
    for t in TERMS:
        A(f"\n**{NICE[t]}** (15 scenes)\n")
        A(head(["source", "pred RMS", "GT RMS", "ratio", "r", "RMSE", "huber"]))
        for lbl in LBLS:
            v = r_ex[lbl]["velocity"][t]
            A(row([lbl, f(v["pred_rms"]), f(v["gt_rms"]), f(v["ratio"]), f(v["r"]), f(v["rmse"]),
                   f(v["huber"], 4)]))

    # per component
    A("\n### Per-component (GT body frame) + the world vertical\n")
    A("root_vel and root_ang_vel components 0/1/2 are the GT body-frame axes; `up` is the "
      "world component along `-gravity_world` of the same (transported) linear increments, "
      "rotated into the world by `R_gt`.  FULL protocol.\n")
    for t in ("root_vel", "root_ang_vel"):
        A(f"\n**{NICE[t]} per component** — ratio / r / RMSE\n")
        A(head(["source"] + [f"c{c} ratio" for c in range(3)] + [f"c{c} r" for c in range(3)]
               + [f"c{c} RMSE" for c in range(3)]))
        for lbl in LBLS:
            pc = r[lbl]["velocity"][t]["per_component"]
            A(row([lbl] + [f(p["ratio"]) for p in pc] + [f(p["r"]) for p in pc]
                  + [f(p["rmse"]) for p in pc]))
        A(row(["GT RMS"] + [f(p["gt_rms"]) for p in r[LBLS[0]]["velocity"][t]["per_component"]]
              + ["—"] * 6))
    A("\n**root_vel world vertical component** (`v_world · (-gravity)`)\n")
    A(head(["source", "pred RMS", "GT RMS", "ratio", "r", "RMSE"]))
    for lbl in LBLS:
        v = r[lbl]["velocity"]["root_vel_up"]
        A(row([lbl, f(v["pred_rms"]), f(v["gt_rms"]), f(v["ratio"]), f(v["r"]), f(v["rmse"])]))

    # per joint
    A("\n### joint_ang_vel per joint (21 body joints, FULL protocol)\n")
    A(head(["joint", "GT RMS"] + [f"{l} ratio" for l in LBLS] + [f"{l} r" for l in LBLS]))
    for j, name in enumerate(names):
        gt = r[LBLS[0]]["velocity"]["joint_ang_vel"]["per_joint"][j]["gt_rms"]
        A(row([name, f(gt)]
              + [f(r[l]["velocity"]["joint_ang_vel"]["per_joint"][j]["ratio"]) for l in LBLS]
              + [f(r[l]["velocity"]["joint_ang_vel"]["per_joint"][j]["r"]) for l in LBLS]))

    # ---------------- b. frequency split
    A("\n## (b) Frequency split of the velocity series (Gaussian sigma 0.2 s along time)\n")
    A("Each velocity series is low-passed per contiguous run (`scipy.gaussian_filter1d`, "
      "`mode='nearest'`); HF = series - LF. Ratios are RMS(pred_part)/RMS(gt_part); RMSE_LF and "
      "RMSE_HF are RMS(pred_part - gt_part) (they do NOT add in quadrature to the full RMSE — the "
      "Gaussian split is not an orthogonal projection).  FULL protocol.\n")
    for t in TERMS:
        gt_lf = r[LBLS[0]]["freq"][t]["lf"]["gt_rms"]
        gt_hf = r[LBLS[0]]["freq"][t]["hf"]["gt_rms"]
        A(f"\n**{NICE[t]}** — GT RMS: LF {gt_lf:.3f}, HF {gt_hf:.3f}\n")
        A(head(["source", "LF ratio", "LF r", "LF RMSE", "HF ratio", "HF r", "HF RMSE",
                "full RMSE"]))
        for lbl in LBLS:
            fq, v = r[lbl]["freq"][t], r[lbl]["velocity"][t]
            A(row([lbl, f(fq["lf"]["ratio"]), f(fq["lf"]["r"]), f(fq["lf"]["rmse"]),
                   f(fq["hf"]["ratio"]), f(fq["hf"]["r"]), f(fq["hf"]["rmse"]), f(v["rmse"])]))

    # ---------------- c. amplitude
    A("\n## (c) Amplitude of the POSES themselves\n")
    A("Per scene: std over time of the pelvis CAMERA-frame position per axis (mm), RMS over time "
      "of the root orientation deviation `|log(R_mean^T R_t)|` (rad, `R_mean` = chordal mean), "
      "and the per-frame mean bone length of the 21 body joints (mm). Pooled over the 16 scenes "
      "as an RMS of the per-scene values (bone length: plain mean).\n")
    A(head(["source", "pelvis cam std x (mm)", "y", "z", "root rot dev RMS (rad)",
            "mean bone (mm)", "bone ratio"]))
    gt_amp = r[LBLS[0]]["amplitude"]
    for lbl in LBLS:
        a = r[lbl]["amplitude"]
        A(row([lbl] + [f(x * 1000, 1) for x in a["pelvis_cam_std_pred"]]
              + [f(a["root_dev_rms_pred"]), f(a["bone_mm_pred"], 1),
                 f(a["bone_mm_pred"] / a["bone_mm_gt"])]))
    A(row(["GT"] + [f(x * 1000, 1) for x in gt_amp["pelvis_cam_std_gt"]]
          + [f(gt_amp["root_dev_rms_gt"]), f(gt_amp["bone_mm_gt"], 1), "1.000"]))
    A("\n**Ratios pred/GT of the same quantities**\n")
    A(head(["source", "pelvis std x", "y", "z", "root rot dev", "bone length"]))
    for lbl in LBLS:
        a = r[lbl]["amplitude"]
        A(row([lbl] + [f(p / g) for p, g in zip(a["pelvis_cam_std_pred"],
                                                a["pelvis_cam_std_gt"])]
              + [f(a["root_dev_rms_pred"] / a["root_dev_rms_gt"]),
                 f(a["bone_mm_pred"] / a["bone_mm_gt"])]))

    A("\n### Per-joint rotation spread about the chordal mean (rad), pred / GT ratio\n")
    A(head(["joint", "GT dev RMS"] + LBLS))
    for j, name in enumerate(names):
        g = r[LBLS[0]]["amplitude"]["joint_dev_rms_gt"][j]
        A(row([name, f(g)] + [f(r[l]["amplitude"]["joint_dev_rms_pred"][j] / g) for l in LBLS]))

    # ---------------- d. depth
    A("\n## (d) Pelvis depth\n")
    A("`bias` = mean(pred z - GT z) per scene, then averaged over the 16 scenes; `|err|` = the "
      "same with the absolute value; `dlogz` = RMS of the per-step difference of "
      "log(pelvis camera z) in %/step at the dump's stride.\n")
    A(head(["source", "depth bias (mm)", "depth abs err (mm)", "dlogz (%/step)"]))
    for lbl in LBLS:
        a = r[lbl]["amplitude"]
        sc = r[lbl]["per_scene"]
        A(row([lbl, f(float(np.mean([s["depth_bias_mm"] for s in sc.values()])), 1),
               f(float(np.mean([s["depth_abs_mm"] for s in sc.values()])), 1),
               f(a["dlogz_pct_pred"])]))
    A(row(["GT", "0.0", "0.0", f(gt_amp["dlogz_pct_gt"])]))

    A("\n### Per-scene pelvis depth bias (mm)\n")
    keys = list(r[LBLS[0]]["per_scene"])
    A(head(["scene", "GT depth (m)"] + LBLS))
    for k in keys:
        A(row([k, f(r[LBLS[0]]["per_scene"][k]["depth_gt_m"], 2)]
              + [f(r[l]["per_scene"][k]["depth_bias_mm"], 0) for l in LBLS]))

    A("\n### Per-scene dlogz (%/step)\n")
    A(head(["scene", "GT"] + LBLS))
    for k in keys:
        A(row([k, f(r[LBLS[0]]["per_scene"][k]["dlogz_gt_pct"])]
              + [f(r[l]["per_scene"][k]["dlogz_pred_pct"]) for l in LBLS]))

    # ---------------- e. mpjpe split
    A("\n## (e) Pelvis(mean-hips)-aligned MPJPE split into LF / HF (sigma 0.2 s)\n")
    A("The aligned body-22 joint positions of pred and GT are each split into LF and HF along "
      "time; `MPJPE_LF` = mean joint distance between the LF parts, `MPJPE_HF` between the HF "
      "parts. `MPJPE` is the unsplit value (the repo's metric, world frame = camera frame under "
      "the rigid lift).  FULL protocol, 3871 person-frames.\n")
    A(head(["source", "MPJPE (mm)", "MPJPE_LF (mm)", "MPJPE_HF (mm)", "d MPJPE vs frozen",
            "d LF vs frozen", "d HF vs frozen", "d LF vs static_ray", "d HF vs static_ray"]))
    fz, sr = r["frozen"]["pose"], r["static_ray"]["pose"]
    for lbl in LBLS:
        p = r[lbl]["pose"]
        A(row([lbl, f(p["mpjpe_mm"], 1), f(p["mpjpe_lf_mm"], 1), f(p["mpjpe_hf_mm"], 1),
               f(p["mpjpe_mm"] - fz["mpjpe_mm"], 1), f(p["mpjpe_lf_mm"] - fz["mpjpe_lf_mm"], 1),
               f(p["mpjpe_hf_mm"] - fz["mpjpe_hf_mm"], 1),
               f(p["mpjpe_lf_mm"] - sr["mpjpe_lf_mm"], 1),
               f(p["mpjpe_hf_mm"] - sr["mpjpe_hf_mm"], 1)]))

    A("\n### Per-scene MPJPE / MPJPE_HF (mm)\n")
    A(head(["scene", "rows"] + [f"{l} mpjpe" for l in LBLS] + [f"{l} HF" for l in LBLS]))
    for k in keys:
        A(row([k, str(r[LBLS[0]]["per_scene"][k]["rows"])]
              + [f(r[l]["per_scene"][k]["mpjpe_mm"], 1) for l in LBLS]
              + [f(r[l]["per_scene"][k]["mpjpe_hf_mm"], 1) for l in LBLS]))

    # ---------------- camera smoothing
    A("\n## Camera-smoothing variant (tvel_ray)\n")
    A("The tvel runs trained and evaluated with Gaussian-smoothed cameras "
      "(`data.camera_smooth_sec: 0.25`, `data/climbing_videos/camera.py::smooth_cameras`); the "
      "tables above lift every source with the RAW `cam_from_world`. Measured effect of the "
      "smoothing on a static scene: camera centre moves 0.5 mm mean / 2.1 mm max, "
      "|dR|_F 2.5e-4 mean.\n")
    A(head(["variant"] + [f"{t} ratio" for t in TERMS] + [f"{t} r" for t in TERMS]
           + [f"{t} RMSE" for t in TERMS] + ["MPJPE (mm)"]))
    for lbl in ("tvel_ray", "tvel_ray@cams0.25"):
        v = r[lbl]["velocity"]
        A(row([lbl] + [f(v[t]["ratio"]) for t in TERMS] + [f(v[t]["r"]) for t in TERMS]
              + [f(v[t]["rmse"]) for t in TERMS] + [f(r[lbl]["pose"]["mpjpe_mm"], 1)]))

    # ---------------- what I measured
    A("\n## What I measured / caveats\n")
    A("""
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
""")

    (OUT / "forensics_results.md").write_text("\n".join(L) + "\n")
    print(f"wrote {OUT / 'forensics_results.md'}")


if __name__ == "__main__":
    main()
