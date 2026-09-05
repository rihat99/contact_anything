"""Velocity-matching forensics on the 16 static test scenes (read-only, no training).

Computes, for the frozen SAM3D refit and five prediction dumps (GT is the
reference of every comparison and is reported through its own RMS):

 (a) the velocity terms of ``model/loss/velocity.py`` (one-step se3 / so3
     increments, predicted increments transported into the GT frame),
 (b) a Gaussian sigma 0.2 s low-pass / high-pass split of every velocity series,
 (c) amplitude statistics of the POSES (pelvis camera-frame spread, root and
     joint rotation spread about the chordal mean, mean bone length),
 (d) pelvis depth bias / error and the per-step log-depth RMS,
 (e) pelvis-aligned MPJPE split into LF and HF along time.

Two row protocols: FULL (every contiguous run at the dump's stride) and EVAL
(the trainer's: one clip per (scene, person) = the longest valid run capped at
``data.eval_max_frames`` = 120 rows). Writes ``forensics.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

import vt_common as vt

sys.path.insert(0, str(vt.REPO))
from better_robot.lie import se3, so3                        # noqa: E402

TERMS = ("root_vel", "root_ang_vel", "joint_ang_vel")
DELTA = {"root_vel": 0.4, "root_ang_vel": 0.6, "joint_ang_vel": 0.9}
SIGMA_SEC = 0.2
HIPS = (1, 2)
EVAL_CAP = 120
CAPS = (60, 120, 180, 240, 10 ** 6)
OUT = Path(__file__).resolve().parent


# ------------------------------------------------------------------ velocity

def velocity_series(pred: dict, gt: dict, idx: np.ndarray, dt: float,
                    gravity: np.ndarray) -> dict:
    """The three velocity terms of one contiguous run, exactly as velocity.py builds them."""
    def pose7(d):
        return torch.as_tensor(np.concatenate([d["root_pos_w"][idx], d["root_quat_w"][idx]], -1),
                               dtype=torch.float64)
    p7, g7 = pose7(pred), pose7(gt)
    dp = se3.log(se3.compose(se3.inverse(p7[:-1]), p7[1:])) / dt
    dg = se3.log(se3.compose(se3.inverse(g7[:-1]), g7[1:])) / dt
    e = so3.compose(so3.inverse(g7[:-1, 3:]), p7[:-1, 3:])
    out = {"root_vel": {"pred": so3.act(e, dp[:, :3]), "gt": dg[:, :3]},
           "root_ang_vel": {"pred": so3.act(e, dp[:, 3:]), "gt": dg[:, 3:]}}
    qjp = torch.as_tensor(pred["body_q"][idx], dtype=torch.float64).transpose(0, 1)
    qjg = torch.as_tensor(gt["body_q"][idx], dtype=torch.float64).transpose(0, 1)
    wp = so3.log(so3.compose(so3.inverse(qjp[:, :-1]), qjp[:, 1:])) / dt
    wg = so3.log(so3.compose(so3.inverse(qjg[:, :-1]), qjg[:, 1:])) / dt
    ej = so3.compose(so3.inverse(qjg[:, :-1]), qjp[:, :-1])
    out["joint_ang_vel"] = {"pred": so3.act(ej, wp).transpose(0, 1).reshape(-1, 63),
                            "gt": wg.transpose(0, 1).reshape(-1, 63)}
    res = {k: {s: v[s].numpy() for s in ("pred", "gt")} for k, v in out.items()}
    rot_gt = vt.quat_to_matrix(gt["root_quat_w"][idx][:-1])
    up = -np.asarray(gravity, np.float64)
    world = {s: np.einsum("nij,nj->ni", rot_gt, res["root_vel"][s]) for s in ("pred", "gt")}
    res["root_vel_world"] = world
    res["root_vel_up"] = {s: (world[s] @ up)[:, None] for s in ("pred", "gt")}
    return res


def split_lf_hf(x: np.ndarray, sigma_frames: float) -> tuple[np.ndarray, np.ndarray]:
    lf = gaussian_filter1d(x, sigma=sigma_frames, axis=0, mode="nearest", truncate=4.0)
    return lf, x - lf


# ------------------------------------------------------------------ statistics

def pooled(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Pooled-over-components RMS / ratio / Pearson r / RMSE (velocity.py's convention)."""
    p, g = np.asarray(pred).reshape(-1), np.asarray(gt).reshape(-1)
    n = p.size
    if n < 2:
        return {k: float("nan") for k in ("pred_rms", "gt_rms", "ratio", "r", "rmse")} | {"n": n}
    cov = n * (p * g).sum() - p.sum() * g.sum()
    vp, vg = n * (p * p).sum() - p.sum() ** 2, n * (g * g).sum() - g.sum() ** 2
    denom = np.sqrt(max(vp, 0.0) * max(vg, 0.0))
    grms, prms = float(np.sqrt((g * g).mean())), float(np.sqrt((p * p).mean()))
    return {"pred_rms": prms, "gt_rms": grms, "ratio": prms / grms if grms > 0 else float("nan"),
            "r": float(cov / denom) if denom > 0 else float("nan"),
            "rmse": float(np.sqrt(((p - g) ** 2).mean())), "n": int(n)}


def huber_mean(pred: np.ndarray, gt: np.ndarray, delta: float) -> float:
    return float(F.smooth_l1_loss(torch.as_tensor(pred / delta), torch.as_tensor(gt / delta),
                                  reduction="mean", beta=1.0))


def chordal_mean(rot: np.ndarray) -> np.ndarray:
    u, _, v = np.linalg.svd(rot.mean(axis=0))
    r = u @ v
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ v
    return r


def so3_log_matrix(r: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(r.reshape(-1, 3, 3)).as_rotvec().reshape(r.shape[:-2] + (3,))


# ------------------------------------------------------------------ collection

def collect() -> dict:
    """Per (source, scene/person): the velocity series, the LF/HF split and the pose stats."""
    scenes = vt.test_scenes()
    parents = vt.smplx_parents()
    variants = [("frozen", 0.0)] + [(s, 0.0) for s in vt.RUNS] + [("tvel_ray", 0.25)]
    data: dict = {}
    for scene in scenes:
        cams = vt.scene_cameras(scene)
        fps, n = cams["fps"], len(cams["extrinsics"])
        base = vt.load_run("static_ray", scene, cams["extrinsics"])
        stride, object_ids = base["stride"], base["object_ids"]
        gtd = vt.load_gt(scene, object_ids)
        dt, grid = stride / fps, np.arange(0, n, stride)
        sigma = SIGMA_SEC / dt
        print(f"[{scene}] fps {fps:.3f} stride {stride} rows {len(grid)} sigma {sigma:.2f}",
              flush=True)
        for src, smooth in variants:
            lbl = f"{src}@cams{smooth:g}" if smooth > 0 else src
            extr = (cams["extrinsics"] if smooth == 0
                    else vt.scene_cameras(scene, smooth)["extrinsics"])
            people = (vt.load_frozen(scene, extr, object_ids, parents)["people"] if src == "frozen"
                      else vt.load_run(src, scene, extr)["people"])
            for p, person in enumerate(people):
                gp = gtd["people"][p]
                if person is None or gp is None:
                    continue
                ok = person["valid"][grid] & gp["valid"][grid]
                if ok.sum() < 12:
                    continue
                edges = np.flatnonzero(np.diff(ok.astype(int)) != 0) + 1
                runs = [b for b in np.split(np.arange(len(grid)), edges)
                        if len(b) >= 12 and ok[b[0]]]
                rec: dict = {"dt": dt, "sigma": sigma, "fps": fps, "stride": stride,
                             "vel": {t: {k: [] for k in
                                         ("pred", "gt", "pred_lf", "gt_lf", "pred_hf", "gt_hf")}
                                     for t in list(TERMS) + ["root_vel_up"]},
                             "eval": {t: {"pred": [], "gt": []} for t in TERMS},
                             "mpjpe": [], "mpjpe_lf": [], "mpjpe_hf": [],
                             "pelvis_cam": {"pred": [], "gt": []},
                             "rot_dev": {"pred": [], "gt": []},
                             "joint_dev": {"pred": [], "gt": []},
                             "bones": {"pred": [], "gt": []},
                             "dlogz": {"pred": [], "gt": []}}
                for b in runs:
                    idx = grid[b]
                    v = velocity_series(person, gp, idx, dt, gtd["gravity"])
                    for t in list(TERMS) + ["root_vel_up"]:
                        for s in ("pred", "gt"):
                            x = v[t][s]
                            lf, hf = split_lf_hf(x, sigma)
                            rec["vel"][t][s].append(x)
                            rec["vel"][t][f"{s}_lf"].append(lf)
                            rec["vel"][t][f"{s}_hf"].append(hf)
                    ec = extr[idx]
                    for payload, s in ((person, "pred"), (gp, "gt")):
                        pc = (np.einsum("nij,nj->ni", ec[:, :3, :3], payload["root_pos_w"][idx])
                              + ec[:, :3, 3])
                        rec["pelvis_cam"][s].append(pc)
                        rec["dlogz"][s].append(np.diff(np.log(np.clip(pc[:, 2], 1e-3, None))))
                        rot = vt.quat_to_matrix(payload["root_quat_w"][idx])
                        rec["rot_dev"][s].append(so3_log_matrix(
                            np.einsum("ab,nac->nbc", chordal_mean(rot), rot)))
                        jr = vt.quat_to_matrix(payload["body_q"][idx])
                        means = np.stack([chordal_mean(jr[:, j]) for j in range(21)])
                        rec["joint_dev"][s].append(so3_log_matrix(
                            np.einsum("jab,njac->njbc", means, jr)))
                        j = payload["joints_w"][idx]
                        rec["bones"][s].append(np.linalg.norm(
                            j[:, 1:vt.NUM_BODY] - j[:, parents[1:vt.NUM_BODY]], axis=-1))
                    ap = (person["joints_w"][idx]
                          - person["joints_w"][idx][:, HIPS].mean(1, keepdims=True))
                    ag = gp["joints_w"][idx] - gp["joints_w"][idx][:, HIPS].mean(1, keepdims=True)
                    rec["mpjpe"].append(np.linalg.norm(ap - ag, axis=-1).mean(1))
                    plf, phf = split_lf_hf(ap, sigma)
                    glf, ghf = split_lf_hf(ag, sigma)
                    rec["mpjpe_lf"].append(np.linalg.norm(plf - glf, axis=-1).mean(1))
                    rec["mpjpe_hf"].append(np.linalg.norm(phf - ghf, axis=-1).mean(1))
                # eval protocol: the longest valid run, capped
                run = max(runs, key=len)
                for cap in CAPS:
                    idx = grid[run][:cap]
                    if len(idx) < 2:
                        continue
                    ev = velocity_series(person, gp, idx, dt, gtd["gravity"])
                    for t in TERMS:
                        rec.setdefault(f"cap{cap}", {}).setdefault(t, {})
                        rec[f"cap{cap}"][t] = {"pred": ev[t]["pred"], "gt": ev[t]["gt"]}
                data.setdefault(lbl, {})[f"{scene}/{p}"] = rec
    return data


# ------------------------------------------------------------------ reduction

def reduce_all(data: dict, drop: tuple = ()) -> dict:
    """Pool every source; ``drop`` removes ``scene/person`` keys from the pooling."""
    results: dict = {}
    for lbl, keys in data.items():
        keys = {k: v for k, v in keys.items() if k not in drop}
        recs = list(keys.values())
        cat = lambda field, t, s: np.concatenate([r[field][t][s] for r in recs])   # noqa: E731
        r: dict = {"velocity": {}, "freq": {}, "caps": {}, "per_scene": {}}
        for t in list(TERMS) + ["root_vel_up"]:
            P = np.concatenate([np.concatenate(x["vel"][t]["pred"]) for x in recs])
            G = np.concatenate([np.concatenate(x["vel"][t]["gt"]) for x in recs])
            entry = pooled(P, G)
            if t in DELTA:
                entry |= {"huber": huber_mean(P, G, DELTA[t]), "delta": DELTA[t]}
            if P.shape[1] == 3:
                entry["per_component"] = [pooled(P[:, c], G[:, c]) for c in range(3)]
            if t == "joint_ang_vel":
                entry["per_joint"] = [pooled(P[:, 3 * j:3 * j + 3], G[:, 3 * j:3 * j + 3])
                                      for j in range(21)]
            r["velocity"][t] = entry
            parts = {}
            for part in ("lf", "hf"):
                Pp = np.concatenate([np.concatenate(x["vel"][t][f"pred_{part}"]) for x in recs])
                Gp = np.concatenate([np.concatenate(x["vel"][t][f"gt_{part}"]) for x in recs])
                parts[part] = pooled(Pp, Gp)
            r["freq"][t] = parts
        for cap in CAPS:
            key = f"cap{cap}"
            r["caps"][key] = {}
            for t in TERMS:
                P = np.concatenate([x[key][t]["pred"] for x in recs if key in x])
                G = np.concatenate([x[key][t]["gt"] for x in recs if key in x])
                r["caps"][key][t] = pooled(P, G) | {"huber": huber_mean(P, G, DELTA[t])}
        mp = np.concatenate([np.concatenate(x["mpjpe"]) for x in recs])
        mlf = np.concatenate([np.concatenate(x["mpjpe_lf"]) for x in recs])
        mhf = np.concatenate([np.concatenate(x["mpjpe_hf"]) for x in recs])
        r["pose"] = {"mpjpe_mm": float(mp.mean() * 1000), "mpjpe_lf_mm": float(mlf.mean() * 1000),
                     "mpjpe_hf_mm": float(mhf.mean() * 1000), "frames": int(mp.size)}

        amp: dict = {}
        for s in ("pred", "gt"):
            std = np.stack([np.concatenate(x["pelvis_cam"][s]).std(0) for x in recs])
            amp[f"pelvis_cam_std_{s}"] = np.sqrt((std ** 2).mean(0)).tolist()
            dev = np.stack([np.sqrt((np.linalg.norm(np.concatenate(x["rot_dev"][s]), axis=-1) ** 2
                                     ).mean()) for x in recs])
            amp[f"root_dev_rms_{s}"] = float(np.sqrt((dev ** 2).mean()))
            jd = np.stack([np.sqrt((np.linalg.norm(np.concatenate(x["joint_dev"][s]), axis=-1) ** 2
                                    ).mean(0)) for x in recs])
            amp[f"joint_dev_rms_{s}"] = np.sqrt((jd ** 2).mean(0)).tolist()
            amp[f"bone_mm_{s}"] = float(np.mean(
                [np.concatenate(x["bones"][s]).mean() for x in recs]) * 1000)
            amp[f"dlogz_pct_{s}"] = float(np.sqrt(np.mean(
                [np.mean(np.concatenate(x["dlogz"][s]) ** 2) for x in recs])) * 100)
        r["amplitude"] = amp

        for key, rec in keys.items():
            pc_p, pc_g = np.concatenate(rec["pelvis_cam"]["pred"]), np.concatenate(
                rec["pelvis_cam"]["gt"])
            entry = {
                "rows": int(sum(len(x) for x in rec["mpjpe"])),
                "mpjpe_mm": float(np.concatenate(rec["mpjpe"]).mean() * 1000),
                "mpjpe_lf_mm": float(np.concatenate(rec["mpjpe_lf"]).mean() * 1000),
                "mpjpe_hf_mm": float(np.concatenate(rec["mpjpe_hf"]).mean() * 1000),
                "depth_bias_mm": float((pc_p[:, 2] - pc_g[:, 2]).mean() * 1000),
                "depth_abs_mm": float(np.abs(pc_p[:, 2] - pc_g[:, 2]).mean() * 1000),
                "depth_gt_m": float(pc_g[:, 2].mean()),
                "pelvis_cam_std_pred": pc_p.std(0).tolist(),
                "pelvis_cam_std_gt": pc_g.std(0).tolist(),
                "dlogz_pred_pct": float(np.sqrt(
                    (np.concatenate(rec["dlogz"]["pred"]) ** 2).mean()) * 100),
                "dlogz_gt_pct": float(np.sqrt(
                    (np.concatenate(rec["dlogz"]["gt"]) ** 2).mean()) * 100),
                "root_dev_rms_pred": float(np.sqrt((np.linalg.norm(
                    np.concatenate(rec["rot_dev"]["pred"]), axis=-1) ** 2).mean())),
                "root_dev_rms_gt": float(np.sqrt((np.linalg.norm(
                    np.concatenate(rec["rot_dev"]["gt"]), axis=-1) ** 2).mean())),
                "bone_mm_pred": float(np.concatenate(rec["bones"]["pred"]).mean() * 1000),
                "bone_mm_gt": float(np.concatenate(rec["bones"]["gt"]).mean() * 1000),
            }
            for t in TERMS:
                entry[t] = pooled(np.concatenate(rec["vel"][t]["pred"]),
                                  np.concatenate(rec["vel"][t]["gt"]))
            r["per_scene"][key] = entry
        results[lbl] = r
    return results


def main() -> None:
    import pickle
    cache = OUT / "collected.pkl"
    if cache.is_file():
        data = pickle.loads(cache.read_bytes())
    else:
        data = collect()
        cache.write_bytes(pickle.dumps(data))
    results = reduce_all(data)
    (OUT / "forensics.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT / 'forensics.json'} and {cache}")


if __name__ == "__main__":
    main()
