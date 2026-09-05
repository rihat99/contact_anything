"""Kernel-family probe on the FROZEN pose token (no training).

Extends scratchpad/camera/token_probe.py: instead of one Gaussian-in-seconds average,
the temporal mixing of the pose token is a general per-clip row-stochastic [T, T] matrix
built from a kernel family, applied INSIDE each clip over its valid frames, and re-read
with the FROZEN MHR + camera heads (``wrapper.decode_pose``).

Families
--------
K0  identity (raw frozen) and Gaussian sigma in SECONDS (the known oracle line).
K1  residual dilution:  M = (I + c*U9)/(1 + c), composed L times (matrix power),
    U9 = uniform over the +-4-frame ROW-INDEX window (edge/validity truncated, renormalised).
K2  same with a local Gaussian (sigma = 1.5 frames, truncated at +-4) in place of U9.
K3  single-layer convex replacement: M = a*I + (1 - a)*U9  (a = the self weight a
    non-residual attention with a self-logit bias would produce).

Reported per kernel (mean over all valid rows of all clips)
-----------------------------------------------------------
centre mass  mean of diag(M_composed) over valid query rows.
width        sqrt(mean_q sum_k M[q,k] (k - q)^2)  -- RMS offset in FRAMES.
depth d1/d3  1st / 3rd temporal difference of the pelvis log-depth, %/frame (x100).
bear d1/d3   same for the pelvis bearing (x/z, y/z), x100.
rot d1/d3    frozen global_rot successive relative rotation angle, degrees.
kp d1/d3     hips-relative MHR70 keypoint differences, mm (all 70 and body-25).
jitter       GVHMR lifted jitter of the world keypoints (10 m/s^3), GT 6.35.
depth err    |pelvis depth - kindyn SMPL-X pelvis depth|, mm (+ signed bias).
accuracy     hips-relative 3D error of the frozen MHR70 body keypoints vs the
             mhr_sup_1 GT (same rig), mm, split by a sigma = 0.2 s Gaussian along
             time into LF (smoothed error) and HF (residual) parts.

argv: out_prefix [--max-clips N] [--families K0,K1,K2,K3]
"""
import json
import sys

import numpy as np
import roma
import torch

REPO = "/data3/rikhat.akizhanov/better/contact_anything_dev"
sys.path.insert(0, REPO)

from data import build_datasets, build_loaders                      # noqa: E402
from data.collate import batch_to_device                            # noqa: E402
from model.loss.keypoint import FACE_KEYPOINTS, FINGER_KEYPOINTS    # noqa: E402
from train.config import load_config, validate                      # noqa: E402
from train.predict import load_model                                # noqa: E402
from utils.geometry import frozen_pelvis_camera, lift_to_world, HIP_KEYPOINTS  # noqa: E402
from utils.gvhmr_metrics import compute_jitter                      # noqa: E402

OUT = sys.argv[1]
MAX_CLIPS = int(sys.argv[sys.argv.index("--max-clips") + 1]) if "--max-clips" in sys.argv else 10 ** 9
FAMILIES = (sys.argv[sys.argv.index("--families") + 1].split(",")
            if "--families" in sys.argv else ["K0", "K1", "K2", "K3"])
# Optional: Gaussian-smooth the CROP TRACK (bbox centre + scale) by this many seconds
# before anything reads it.  The frozen camera head's scale s tracks the bbox scale b
# almost exactly, so the pelvis depth 2f/(s b) cancels most crop jitter; smoothing the
# TOKEN alone breaks that cancellation.  Needs the image path (the embedding cache is
# keyed to the raw crop), so it is much slower.
BOX_GAUSS = (float(sys.argv[sys.argv.index("--box-gauss") + 1])
             if "--box-gauss" in sys.argv else None)

CFG = f"{REPO}/configs/static_ray.yaml"
WINDOW = 4                     # +-4 frames, the trained block's per-layer window
LOCAL_SIGMA_FRAMES = 1.5
LF_SIGMA_SEC = 0.2
BODY_KP = [i for i in range(70)
           if i not in set(FINGER_KEYPOINTS) and i not in set(FACE_KEYPOINTS)]

cfg = load_config(CFG)
cfg["data"]["embedding_cache"] = BOX_GAUSS is None
cfg["data"]["num_workers"] = 6
validate(cfg)


# --------------------------------------------------------------------- kernels
def _rownorm(w, valid):
    w = w * valid[None, :]
    s = w.sum(1, keepdims=True)
    out = np.where(s > 1e-12, w / np.maximum(s, 1e-12), 0.0)
    dead = (s[:, 0] <= 1e-12)
    if dead.any():                                    # no valid key: keep the row itself
        out[dead] = np.eye(len(w))[dead]
    return out


def build_kernels(T, valid, pos_sec):
    """``{name: [T, T] row-stochastic}`` for one clip."""
    eye = np.eye(T)
    off = np.arange(T)[None, :] - np.arange(T)[:, None]
    inside = np.abs(off) <= WINDOW
    uni = _rownorm(inside.astype(np.float64), valid)
    loc = _rownorm(np.exp(-0.5 * (off / LOCAL_SIGMA_FRAMES) ** 2) * inside, valid)
    dt = pos_sec[None, :] - pos_sec[:, None]

    out = {}
    if "K0" in FAMILIES:
        out["K0 identity"] = eye
        for sig in (0.04, 0.08, 0.12):
            out[f"K0 gauss {sig:.2f}s"] = _rownorm(np.exp(-0.5 * (dt / sig) ** 2), valid)
    if "K1" in FAMILIES:
        for c in (0.5, 1.0, 2.0, 4.0, 8.0, 32.0):
            m = (eye + c * uni) / (1.0 + c)
            for lay in (1, 2, 4):
                out[f"K1 c{c:g} L{lay}"] = np.linalg.matrix_power(m, lay)
    if "K2" in FAMILIES:
        for c in (0.5, 1.0, 2.0, 4.0, 8.0, 32.0):
            m = (eye + c * loc) / (1.0 + c)
            for lay in (1, 2, 4):
                out[f"K2 c{c:g} L{lay}"] = np.linalg.matrix_power(m, lay)
    if "K3" in FAMILIES:
        for a in (0.5, 0.25, 0.11, 0.0):
            out[f"K3 a{a:g}"] = a * eye + (1.0 - a) * uni
    return out


def kernel_shape(w, valid):
    """(sum of centre mass, sum of squared offsets, n rows) over valid query rows."""
    t = w.shape[0]
    off = (np.arange(t)[None, :] - np.arange(t)[:, None]).astype(np.float64)
    rows = np.flatnonzero(valid)
    centre = np.diag(w)[rows].sum()
    second = (w[rows] * off[rows] ** 2).sum()
    return centre, second, float(len(rows))


# --------------------------------------------------------------------- helpers
def gauss_time(x, sec, sigma):
    """Gaussian smoothing along axis 0 at irregular times ``sec``."""
    w = np.exp(-0.5 * ((sec[:, None] - sec[None, :]) / sigma) ** 2)
    w = w / w.sum(1, keepdims=True)
    return np.tensordot(w, x, axes=([1], [0]))


def d13(x):
    return np.diff(x, axis=0), x[3:] - 3 * x[2:-1] + 3 * x[1:-2] - x[:-3]


STAT_KEYS = ("depth_d1", "depth_d3", "bear_d1", "bear_d3", "rot_d1", "rot_d3",
             "kp_d1", "kp_d3", "kpb_d1", "kpb_d3", "jit", "depth_err", "depth_bias",
             "acc_tot", "acc_lf", "acc_hf")


def clip_stats(pelvis, global_rot, kp3d, ext, sec, gt_pelvis, gt_kp_cam, gt_ok):
    """All per-clip diagnostics for one kernel (numpy float64, valid rows only)."""
    s = {}
    ray = np.stack([pelvis[:, 0] / pelvis[:, 2], pelvis[:, 1] / pelvis[:, 2],
                    np.log(pelvis[:, 2])], -1) * 100
    d1, d3 = d13(ray[:, 2]); s["depth_d1"], s["depth_d3"] = d1, d3
    d1, d3 = d13(ray[:, :2]); s["bear_d1"], s["bear_d3"] = d1.ravel(), d3.ravel()

    rot = roma.euler_to_rotmat("xyz", torch.tensor(global_rot, dtype=torch.float64)).numpy()
    rel = np.einsum("nji,njk->nik", rot[:-1], rot[1:])
    wvec = roma.rotmat_to_rotvec(torch.tensor(rel)).numpy() * 180 / np.pi
    s["rot_d1"] = np.linalg.norm(wvec, axis=-1)
    s["rot_d3"] = np.linalg.norm(wvec[2:] - 2 * wvec[1:-1] + wvec[:-2], axis=-1)

    rel_kp = (kp3d - kp3d[:, list(HIP_KEYPOINTS)].mean(1, keepdims=True)) * 1000
    d1, d3 = d13(rel_kp)
    s["kp_d1"] = np.linalg.norm(d1, axis=-1).ravel()
    s["kp_d3"] = np.linalg.norm(d3, axis=-1).ravel()
    s["kpb_d1"] = np.linalg.norm(d1[:, BODY_KP], axis=-1).ravel()
    s["kpb_d3"] = np.linalg.norm(d3[:, BODY_KP], axis=-1).ravel()

    abs_kp = kp3d - kp3d[:, list(HIP_KEYPOINTS)].mean(1, keepdims=True) + pelvis[:, None]
    world = lift_to_world(torch.tensor(abs_kp, dtype=torch.float32),
                          torch.tensor(ext, dtype=torch.float32))
    fps = 1.0 / float(np.median(np.diff(sec)))
    s["jit"] = np.asarray(compute_jitter(world, fps=fps)).ravel()
    dz = pelvis[:, 2] - gt_pelvis[:, 2]
    s["depth_err"], s["depth_bias"] = np.abs(dz), dz

    if gt_ok.sum() >= 5:
        pr = rel_kp[gt_ok][:, BODY_KP] / 1000.0
        gt = gt_kp_cam[gt_ok]
        gt = gt - gt[:, list(HIP_KEYPOINTS)].mean(1, keepdims=True)
        diff = (pr - gt[:, BODY_KP]) * 1000.0                    # mm, [n, 25, 3]
        lf = gauss_time(diff, sec[gt_ok], LF_SIGMA_SEC)
        s["acc_tot"] = np.linalg.norm(diff, axis=-1).ravel()
        s["acc_lf"] = np.linalg.norm(lf, axis=-1).ravel()
        s["acc_hf"] = np.linalg.norm(diff - lf, axis=-1).ravel()
    else:
        s["acc_tot"] = s["acc_lf"] = s["acc_hf"] = np.zeros(0)
    return s


# --------------------------------------------------------------------- crop track
if BOX_GAUSS is not None:
    from scipy.ndimage import gaussian_filter1d
    from data.climbing_videos import dataset as ds_mod
    _orig_load = ds_mod.scene_io.load_scene

    def _patched_load(root, scene, split, contact_level):
        data = _orig_load(root, scene, split, contact_level)
        bbox = np.asarray(data["bbox"], np.float32).copy()
        fps = float(data["fps"])
        for p_ in range(bbox.shape[0]):
            idx = np.flatnonzero(np.asarray(data["valid_mask"][p_], bool))
            if len(idx) < 3:
                continue
            b = bbox[p_][idx]
            c = (b[:, :2] + b[:, 2:]) / 2
            sz = b[:, 2:] - b[:, :2]
            c = gaussian_filter1d(c, BOX_GAUSS * fps, axis=0, mode="nearest")
            sz = gaussian_filter1d(sz, BOX_GAUSS * fps, axis=0, mode="nearest")
            bbox[p_][idx] = np.concatenate([c - sz / 2, c + sz / 2], -1)
        data["bbox"] = bbox
        return data

    ds_mod.scene_io.load_scene = _patched_load

# --------------------------------------------------------------------- model
dev = "cuda"
model, _ = load_model(CFG, None, dev)
model.eval()
_, test_sets = build_datasets(cfg, {"smplx", "keypoints"})
_, loader = build_loaders(cfg, [], test_sets)


def run_wrapper(batch):
    learned = [b for b in (model.contact_tokens, model.force_tokens, model.motion_tokens)
               if b is not None]
    n = batch["bbox_center"].shape[0]
    return model.wrapper(
        img=batch.get("img") if batch.get("embedding") is None else None,
        embedding=batch.get("embedding"),
        bbox_center=batch["bbox_center"], bbox_scale=batch["bbox_scale"],
        ori_img_size=batch["ori_img_size"], img_size=batch["img_size"],
        affine_trans=batch["affine_trans"], cam_int=batch["cam_int"],
        mask=batch["mask"], mask_score=batch["mask_score"],
        blocks=[b.as_extra_block(n) for b in learned],
        attention=model.extra_token_attention)


acc = {}
shape_acc = {}
n_clips = 0
n_frames = 0
box_d1 = []          # kernel-independent: the crop track's own log-scale step, %/frame

with torch.no_grad():
    for batch in loader:
        batch = batch_to_device(batch, dev)
        seq = int(batch["seq_len"])
        b = batch["frame_valid"].shape[0]
        out = run_wrapper(batch)
        tok = out["tokens"][:, 0].float()                        # [B, C] pose token
        ctx = out["ctx"]

        pos_all = batch["frame_pos_sec"].float().cpu().numpy().astype(np.float64)
        val_all = (batch["smplx_valid"] & batch["frame_valid"]).cpu().numpy()
        gtok_all = (batch["kp_valid"] & batch["frame_valid"]).cpu().numpy()
        ext_all = batch["cam_from_world"].float().cpu().numpy().astype(np.float64)
        gtw = batch["smplx_joints_world"][:, 0].float()
        gt_pel_all = ((torch.einsum("bij,bj->bi", batch["cam_from_world"].float()[:, :3, :3], gtw)
                       + batch["cam_from_world"].float()[:, :3, 3]).cpu().numpy().astype(np.float64))
        kpw = batch["kp3d_world"].float()
        gt_kp_all = ((torch.einsum("bij,bkj->bki", batch["cam_from_world"].float()[:, :3, :3], kpw)
                      + batch["cam_from_world"].float()[:, :3, 3][:, None])
                     .cpu().numpy().astype(np.float64))

        nc = b // seq
        kernels_per_clip = []
        for c in range(nc):
            sl = slice(c * seq, (c + 1) * seq)
            kernels_per_clip.append(build_kernels(seq, val_all[sl].astype(np.float64),
                                                  pos_all[sl]))
        names = list(kernels_per_clip[0])

        for name in names:
            mats = np.stack([kernels_per_clip[c][name] for c in range(nc)])   # [nc, T, T]
            w = torch.tensor(mats, dtype=torch.float32, device=dev)
            flat = tok.view(nc, seq, -1)
            mixed = torch.einsum("nqk,nkd->nqd", w, flat).reshape(tok.shape)
            mhr = model.wrapper.decode_pose(mixed, ctx)
            pel = frozen_pelvis_camera(mhr).double().cpu().numpy()
            grot = mhr["global_rot"].double().cpu().numpy()
            kp3 = mhr["pred_keypoints_3d"].double().cpu().numpy()
            bucket = acc.setdefault(name, {k: [] for k in STAT_KEYS})
            sh = shape_acc.setdefault(name, [0.0, 0.0, 0.0])
            for c in range(nc):
                sl = slice(c * seq, (c + 1) * seq)
                v = val_all[sl]
                if v.sum() < 10:
                    continue
                idx = np.flatnonzero(v)
                cen, sec2, rows = kernel_shape(mats[c], v.astype(np.float64))
                sh[0] += cen; sh[1] += sec2; sh[2] += rows
                st = clip_stats(pel[sl][idx], grot[sl][idx], kp3[sl][idx],
                                ext_all[sl][idx], pos_all[sl][idx], gt_pel_all[sl][idx],
                                gt_kp_all[sl][idx], gtok_all[sl][idx])
                for k in STAT_KEYS:
                    bucket[k].append(st[k])
        bscale = batch["bbox_scale"][:, 0].float().cpu().numpy().astype(np.float64)
        for c in range(nc):
            sl = slice(c * seq, (c + 1) * seq)
            if val_all[sl].sum() >= 10:
                n_clips += 1
                n_frames += int(val_all[sl].sum())
                box_d1.append(np.diff(np.log(bscale[sl][np.flatnonzero(val_all[sl])])) * 100)
        if n_clips >= MAX_CLIPS:
            break

# --------------------------------------------------------------------- report
def rms(v):
    a = np.concatenate(v)
    return float(np.sqrt(np.mean(a ** 2))) if a.size else float("nan")


def mean(v):
    a = np.concatenate(v)
    return float(np.mean(a)) if a.size else float("nan")


rows = []
for name, bucket in acc.items():
    cen, sec2, nrow = shape_acc[name]
    rows.append({
        "kernel": name,
        "centre_mass": cen / nrow,
        "width_frames": float(np.sqrt(sec2 / nrow)),
        "depth_d1": rms(bucket["depth_d1"]), "depth_d3": rms(bucket["depth_d3"]),
        "bear_d1": rms(bucket["bear_d1"]), "bear_d3": rms(bucket["bear_d3"]),
        "rot_d1": rms(bucket["rot_d1"]), "rot_d3": rms(bucket["rot_d3"]),
        "kp_d1": rms(bucket["kp_d1"]), "kp_d3": rms(bucket["kp_d3"]),
        "kpb_d1": rms(bucket["kpb_d1"]), "kpb_d3": rms(bucket["kpb_d3"]),
        "jitter": mean(bucket["jit"]),
        "depth_err_mm": mean(bucket["depth_err"]) * 1000,
        "depth_bias_mm": mean(bucket["depth_bias"]) * 1000,
        "acc_tot": mean(bucket["acc_tot"]), "acc_lf": mean(bucket["acc_lf"]),
        "acc_hf": mean(bucket["acc_hf"]),
    })

hdr = (f"{'kernel':<16s}{'ctr':>7s}{'width':>7s}{'dep_d1':>8s}{'dep_d3':>8s}"
       f"{'bea_d3':>8s}{'rot_d1':>8s}{'rot_d3':>8s}{'kp_d3':>8s}{'kpb_d3':>8s}"
       f"{'jitter':>8s}{'dep_err':>9s}{'acc_LF':>8s}{'acc_HF':>8s}{'acc_tot':>9s}")
bd1 = float(np.sqrt(np.mean(np.concatenate(box_d1) ** 2)))
box_tag = "raw crops, embedding cache" if BOX_GAUSS is None else f"crop track gauss {BOX_GAUSS} s, image path"
print(f"\nclips {n_clips}  valid frames {n_frames}  config {CFG}  ({box_tag})")
print(f"crop track (kernel independent): RMS d1 log bbox_scale = {bd1:.3f} %/frame")
print(hdr)
for r in rows:
    print(f"{r['kernel']:<16s}{r['centre_mass']:7.3f}{r['width_frames']:7.2f}"
          f"{r['depth_d1']:8.3f}{r['depth_d3']:8.3f}{r['bear_d3']:8.3f}"
          f"{r['rot_d1']:8.2f}{r['rot_d3']:8.2f}{r['kp_d3']:8.1f}{r['kpb_d3']:8.1f}"
          f"{r['jitter']:8.1f}{r['depth_err_mm']:9.0f}{r['acc_lf']:8.1f}"
          f"{r['acc_hf']:8.1f}{r['acc_tot']:9.1f}")

with open(f"{OUT}.json", "w") as fh:
    json.dump({"clips": n_clips, "frames": n_frames, "config": CFG,
               "crop_dlog_b_pct_per_frame": bd1, "box_gauss_sec": BOX_GAUSS,
               "window": WINDOW, "local_sigma_frames": LOCAL_SIGMA_FRAMES,
               "lf_sigma_sec": LF_SIGMA_SEC, "body_keypoints": BODY_KP,
               "rows": rows}, fh, indent=2)
print(f"wrote {OUT}.json")
