"""docs/old/plan.md Phase -1 item 1 — derivative-target consistency, gradient conflict, shrinkage.

On the round-4 arm F checkpoint (cached-token path), test split: (1) the current derivative
targets vs the derivatives of ONE consistently smoothed GT trajectory vs the raw ones (RMS,
differences, effective Gaussian width, per-band attenuation); (2) the cosine between the
position terms' and the pose-derivative terms' gradients under both target recipes; (3) the
band-wise ratio of the predicted joint-velocity amplitude to the GT's. Definitions and
results: ``output_2/audits/targets/RESULTS.md``.

    CUDA_VISIBLE_DEVICES=0 python scripts/audit_targets.py \
        --config configs/final.yaml --checkpoint output_2/<run>/last.pth \
        --limit-scenes 30 --out output_2/audits/targets
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import roma
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import build_datasets                        # noqa: E402
from data.collate import batch_to_device               # noqa: E402
from data.loaders import build_loaders                 # noqa: E402
from model.loss.motion import QUANTITIES, MotionLoss   # noqa: E402
from model.loss.smplx import SmplxLoss                 # noqa: E402
from model.refiner import (NUM_BODY_JOINTS, angular_velocity, gaussian_smooth,  # noqa: E402
                           smooth_rotations, stencil_valid, time_derivative)
from train.config import signal_needs                  # noqa: E402
from train.predict import load_model                   # noqa: E402

JOINT_GROUPS = {"root": (0,), "hips_knees_ankles": (1, 2, 4, 5, 7, 8),
                "shoulders_elbows_wrists": (16, 17, 18, 19, 20, 21),
                "spine_head": (3, 6, 9, 12, 15), "feet_collars": (10, 11, 13, 14),
                "all22": tuple(range(NUM_BODY_JOINTS))}
BANDS = (("<1Hz", 0.0, 1.0), ("1-3Hz", 1.0, 3.0), (">3Hz", 3.0, float("inf")),
         ("broadband", 0.0, float("inf")))
BAND_NAMES = [b for b, _, _ in BANDS]
POS_TERMS = ("kp3d", "orient", "pose", "root_bias", "root_shape")
DER_TERMS = ("pose_vel", "pose_acc", "pose_ang_vel", "pose_ang_acc")
SWEEP = (0.0, 0.05, 0.08, 0.12, 0.17)
SIGMA_GRID = np.round(np.arange(0.0, 0.4001, 0.0025), 6)
BODIES = (("refined", "refined (arm F output)"),
          ("stage1_smoothed", "stage 1 + input smoothing (`joints_world_in`)"),
          ("pose_head_zeroed", "pose-delta head zeroed"),
          ("stage1_unsmoothed", "stage 1, input smoothing off"),
          ("refined_unsmoothed", "refined, input smoothing off"))


# ------------------------------------------------------------------ stencils

def bcast(mask, like):
    return mask.reshape(*mask.shape, *([1] * (like.dim() - 2)))


def central_second(x, seconds, valid):
    """Centred second difference AT the frame (non-uniform spacing); 0 without both neighbours."""
    hb = (seconds[:, 1:-1] - seconds[:, :-2]).clamp(min=1e-6)
    hf = (seconds[:, 2:] - seconds[:, 1:-1]).clamp(min=1e-6)
    inner = 2.0 * ((x[:, 2:] - x[:, 1:-1]) / bcast(hf, x)
                   - (x[:, 1:-1] - x[:, :-2]) / bcast(hb, x)) / bcast(hb + hf, x)
    zero = torch.zeros_like(x[:, :1])
    out = torch.cat([zero, inner, zero], dim=1)
    ok = torch.zeros_like(valid)
    ok[:, 1:-1] = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]
    return torch.where(bcast(ok, out), out, torch.zeros_like(out))


def midpoint_angular(rot, seconds, valid):
    """WORLD-frame angular velocity / acceleration from the midpoint increments of ``rot``.

    ``w[t+1/2] = log(R_t^T R_{t+1}) / h`` is the same vector in the frames of ``t`` and
    ``t+1``; the frame-centred velocity is the mean of the two adjacent midpoints (identical
    to :func:`model.refiner.angular_velocity`) and the acceleration is their difference over
    the half-span — the tight centred second-order stencil, both read in ``R_t``'s frame.
    """
    inc = roma.rotmat_to_rotvec(rot[:, :-1].transpose(-1, -2) @ rot[:, 1:])
    h = (seconds[:, 1:] - seconds[:, :-1]).clamp(min=1e-6)
    ok = valid[:, 1:] & valid[:, :-1]
    w = torch.where(ok[..., None], inc / h[..., None], torch.zeros_like(inc))
    zw, zb = torch.zeros_like(w[:, :1]), torch.zeros_like(ok[:, :1])
    fwd, bwd = torch.cat([w, zw], dim=1), torch.cat([zw, w], dim=1)
    fok, bok = torch.cat([ok, zb], dim=1), torch.cat([zb, ok], dim=1)
    vel_b = (fwd + bwd) / (fok.to(w.dtype) + bok.to(w.dtype)).clamp(min=1.0)[..., None]
    span = torch.zeros_like(seconds)
    span[:, 1:-1] = 0.5 * (seconds[:, 2:] - seconds[:, :-2])
    acc_b = torch.where((fok & bok)[..., None], (fwd - bwd) / span.clamp(min=1e-6)[..., None],
                        torch.zeros_like(fwd))
    return (rot @ vel_b[..., None])[..., 0], (rot @ acc_b[..., None])[..., 0]


def row_masks(valid, seconds, sigma):
    """The MotionLoss row masks: ``stencil_valid`` at radius max(1|2, ceil(2 sigma / dt))."""
    steps = seconds[:, 1:] - seconds[:, :-1]
    dt = float(steps[steps > 0].median()) if bool((steps > 0).any()) else 0.0
    edge = int(math.ceil(2.0 * sigma / dt)) if dt > 0 else 0
    return stencil_valid(valid, max(1, edge)), stencil_valid(valid, max(2, edge)), dt


class ConsistentMotionLoss(MotionLoss):
    """MotionLoss with ONE Gaussian on the GT positions / rotations and ALIGNED stencils.

    Target and prediction both use: velocity = central difference at the frame,
    acceleration = centred second difference at the frame (rotations: midpoint increments).
    """

    @staticmethod
    def _derivs(joints, root, seconds, valid):
        ang, ang_acc = midpoint_angular(root, seconds, valid)
        return {"vel": time_derivative(joints, seconds, valid),
                "acc": central_second(joints, seconds, valid),
                "ang_vel": ang, "ang_acc": ang_acc}

    def _clip(self, batch, n_frames):
        seq_len = int(batch["seq_len"])
        n = n_frames // seq_len
        return n, seq_len, batch["frame_pos_sec"].to(self.device, self.dtype).view(n, seq_len)

    def targets(self, batch, n_frames):
        n, seq_len, seconds = self._clip(batch, n_frames)
        valid = (batch["smplx_valid"] & batch["frame_valid"]).to(self.device).view(n, seq_len)
        joints = batch["smplx_joints_world"][:, :NUM_BODY_JOINTS].to(self.device, self.dtype)
        root = batch["smplx_root_rot"].to(self.device, self.dtype).view(n, seq_len, 3, 3)
        js = gaussian_smooth(joints.view(n, seq_len, NUM_BODY_JOINTS, 3), seconds, valid, self.sigma)
        out = self._derivs(js, smooth_rotations(root, seconds, valid, self.sigma), seconds, valid)
        rows_vel, rows_acc, _ = row_masks(valid, seconds, self.sigma)
        masks = {"vel": rows_vel.reshape(n_frames), "acc": rows_acc.reshape(n_frames),
                 "ang_vel": rows_vel.reshape(n_frames), "ang_acc": rows_acc.reshape(n_frames)}
        return {q: out[q].reshape(n_frames, *out[q].shape[2:]) for q in QUANTITIES}, masks

    def pose_derivatives(self, out, batch):
        smplx = out["smplx"]
        n, seq_len, seconds = self._clip(batch, smplx["joints_world"].shape[0])
        valid = batch["frame_valid"].to(self.device).view(n, seq_len)
        joints = smplx["joints_world"][:, :NUM_BODY_JOINTS].to(self.device, self.dtype)
        d = self._derivs(joints.view(n, seq_len, NUM_BODY_JOINTS, 3),
                         smplx["root_rot_world"].to(self.device, self.dtype).view(n, seq_len, 3, 3),
                         seconds, valid)
        return {q: d[q].reshape(n * seq_len, *d[q].shape[2:]) for q in QUANTITIES}


# ------------------------------------------------------------------ accumulators

class Acc:
    """Sums of squares / element counts keyed by name."""

    def __init__(self):
        self.sq, self.n = {}, {}

    def add(self, key, x, mask):
        w = mask.to(x.dtype)
        self.sq[key] = self.sq.get(key, 0.0) + float((x.pow(2) * bcast(w, x)).sum())
        self.n[key] = self.n.get(key, 0.0) + float(w.sum()) * int(np.prod(x.shape[2:]))

    def rms(self, key):
        return math.sqrt(self.sq[key] / self.n[key]) if self.n.get(key, 0) > 0 else float("nan")


def band_power(x, dt, edge):
    """Per-band power of a fully valid, uniformly spaced clip series ``[1, T, ...]``.

    The ``edge`` truncated-kernel frames are dropped at both ends, the mean removed and a
    Hann window applied (identically to every series, so band RATIOS are unaffected).
    """
    y = x[0, edge:x.shape[1] - edge] if edge > 0 else x[0]
    t = y.shape[0]
    if t < 8:
        return None
    y = (y - y.mean(dim=0, keepdim=True)).reshape(t, -1)
    win = torch.hann_window(t, periodic=False, device=y.device, dtype=y.dtype)[:, None]
    power = torch.fft.rfft(y * win, dim=0).abs().pow(2).sum(dim=1)
    freq = torch.fft.rfftfreq(t, d=dt).to(y.device)
    return {name: float(power[(freq >= lo) & (freq < hi)].sum()) for name, lo, hi in BANDS}


def add_bands(store, key, bands):
    for name, value in (bands or {}).items():
        store[(key, name)] = store.get((key, name), 0.0) + value


def ratio(store, num, den, band):
    a, b = store.get((num, band), 0.0), store.get((den, band), 0.0)
    return math.sqrt(a / b) if b > 0 else float("nan")


# ------------------------------------------------------------------ passes

def gt_series(batch, device, sigma):
    seq_len = int(batch["seq_len"])
    n = batch["frame_pos_sec"].shape[0] // seq_len
    seconds = batch["frame_pos_sec"].to(device, torch.float32).view(n, seq_len)
    valid = (batch["smplx_valid"] & batch["frame_valid"]).to(device).view(n, seq_len)
    joints = batch["smplx_joints_world"][:, :NUM_BODY_JOINTS].to(device, torch.float32)
    root = batch["smplx_root_rot"].to(device, torch.float32).view(n, seq_len, 3, 3)
    rows_vel, rows_acc, dt = row_masks(valid, seconds, sigma)
    return joints.view(n, seq_len, NUM_BODY_JOINTS, 3), root, seconds, valid, rows_vel, rows_acc, dt


def uniform(valid, seconds, dt):
    steps = seconds[:, 1:] - seconds[:, :-1]
    return dt > 0 and bool(valid.all()) and float((steps - dt).abs().max()) < 1e-3 * dt


def part1(batch, device, sigma, acc, fit, spec, sweep):
    """Current vs consistent vs raw GT derivatives, accumulated over clips."""
    joints, root, seconds, valid, rows_vel, rows_acc, dt = gt_series(batch, device, sigma)
    first, second = stencil_valid(valid, 1), stencil_valid(valid, 2)
    v_raw = time_derivative(joints, seconds, valid)                       # current recipe
    v_star = gaussian_smooth(v_raw, seconds, first, sigma)
    a_raw = time_derivative(v_raw, seconds, first)
    a_star = gaussian_smooth(time_derivative(v_star, seconds, first), seconds, second, sigma)
    w_raw = (root @ angular_velocity(root, seconds, valid)[..., None])[..., 0]
    w_star = gaussian_smooth(w_raw, seconds, first, sigma)
    aa_raw = time_derivative(w_raw, seconds, first)
    aa_star = gaussian_smooth(time_derivative(w_star, seconds, first), seconds, second, sigma)
    js = gaussian_smooth(joints, seconds, valid, sigma)                   # consistent recipe
    rs = smooth_rotations(root, seconds, valid, sigma)
    v_c, a_c = time_derivative(js, seconds, valid), central_second(js, seconds, valid)
    a_c_comp = time_derivative(v_c, seconds, first)
    w_c, aa_c = midpoint_angular(rs, seconds, valid)
    a_raw_tight = central_second(joints, seconds, valid)
    _, aa_raw_tight = midpoint_angular(root, seconds, valid)

    for key, x, m in (
            ("v_star", v_star, rows_vel), ("v_c", v_c, rows_vel), ("v_raw", v_raw, rows_vel),
            ("a_star", a_star, rows_acc), ("a_c_tight", a_c, rows_acc),
            ("a_c_comp", a_c_comp, rows_acc), ("a_raw_comp", a_raw, rows_acc),
            ("a_raw_tight", a_raw_tight, rows_acc), ("w_star", w_star, rows_vel),
            ("w_c", w_c, rows_vel), ("w_raw", w_raw, rows_vel), ("aa_star", aa_star, rows_acc),
            ("aa_c", aa_c, rows_acc), ("aa_raw_comp", aa_raw, rows_acc),
            ("aa_raw_tight", aa_raw_tight, rows_acc), ("d_v", v_star - v_c, rows_vel),
            ("d_a_tight", a_star - a_c, rows_acc), ("d_a_comp", a_star - a_c_comp, rows_acc),
            ("d_w", w_star - w_c, rows_vel), ("d_aa", aa_star - aa_c, rows_acc)):
        acc.add(key, x, m)
    for s in SWEEP:                    # consistent targets at other single-Gaussian widths
        a2 = central_second(joints if s == 0.0 else gaussian_smooth(joints, seconds, valid, s),
                            seconds, valid)
        sweep.add(f"a_c@{s}", a2, rows_acc)
        sweep.add(f"d_a@{s}", a_star - a2, rows_acc)
    for key, raw, star, m in (("v", v_raw, v_star, rows_vel), ("a", a_raw, a_star, rows_acc),
                              ("w", w_raw, w_star, rows_vel), ("aa", aa_raw, aa_star, rows_acc)):
        w = bcast(m.to(raw.dtype), raw)                # effective single-Gaussian width of the
        for i, s in enumerate(SIGMA_GRID):             # current target over the raw derivative
            sm = raw if s == 0.0 else gaussian_smooth(raw, seconds, valid, float(s))
            fit[key][i] += float((((sm - star) ** 2) * w).sum())
    if not uniform(valid, seconds, dt):
        return 0
    edge = int(math.ceil(2.0 * sigma / dt))
    for key, x in (("v_star", v_star), ("v_raw", v_raw), ("v_c", v_c), ("a_star", a_star),
                   ("a_raw_comp", a_raw), ("a_c_tight", a_c), ("a_raw_tight", a_raw_tight),
                   ("w_star", w_star), ("w_raw", w_raw), ("aa_star", aa_star),
                   ("aa_raw_comp", aa_raw)):
        add_bands(spec, key, band_power(x, dt, edge))
    return 1


def term_grads(result, inputs, names):
    """Per-term gradients of ``numerator / max(mass, 1)`` — the trainer's single-process form."""
    grads = {}
    for name in names:
        term = result.terms[name]
        g = torch.autograd.grad(term.numerator / max(term.mass, 1.0), inputs,
                                retain_graph=True, allow_unused=True)
        grads[name] = [torch.zeros_like(i) if x is None else x for i, x in zip(inputs, g)]
    return grads


def cosine(a, b):
    na, nb = float(a.norm()), float(b.norm())
    return float((a * b).sum()) / (na * nb) if na > 0 and nb > 0 else float("nan")


def flat(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def part2(model, out, batch, losses, records):
    joints_world, root_world = out["smplx"]["joints_world"], out["smplx"]["root_rot_world"]
    params = list(model.refiner.heads["pose"].parameters())
    inputs = [joints_world, root_world] + params
    pos = term_grads(losses["smplx"](out, batch, train=True), inputs, POS_TERMS)
    for tag in ("current", "consistent"):
        der = term_grads(losses[tag](out, batch, train=True), inputs, DER_TERMS)
        rec = {"terms": {n: {"joints": float(g[0][:, :NUM_BODY_JOINTS].norm()),
                             "root_rot": float(g[1].norm()), "params": float(flat(g[2:]).norm())}
                         for n, g in list(pos.items()) + list(der.items())}}
        gp = [sum(pos[n][i] for n in POS_TERMS) for i in range(len(inputs))]
        gd = [sum(der[n][i] for n in DER_TERMS) for i in range(len(inputs))]
        jp, jd = gp[0][:, :NUM_BODY_JOINTS], gd[0][:, :NUM_BODY_JOINTS]
        rec.update(
            cos_joints_all=cosine(jp.reshape(-1), jd.reshape(-1)),
            cos_per_joint=[cosine(jp[:, j].reshape(-1), jd[:, j].reshape(-1))
                           for j in range(NUM_BODY_JOINTS)],
            cos_root_rot=cosine(gp[1].reshape(-1), gd[1].reshape(-1)),
            cos_params=cosine(flat(gp[2:]), flat(gd[2:])),
            norm_joints_pos=float(jp.norm()), norm_joints_der=float(jd.norm()),
            norm_params_pos=float(flat(gp[2:]).norm()), norm_params_der=float(flat(gd[2:]).norm()))
        records[tag].append(rec)


def part3(batch, series, device, sigma, spec):
    """Band power of the joint VELOCITY of every body plus the GT, on uniform clips."""
    joints, _, seconds, valid, _, _, dt = gt_series(batch, device, sigma)
    if not uniform(valid, seconds, dt):
        return
    edge = int(math.ceil(2.0 * sigma / dt))
    seq_len = joints.shape[1]
    for key, x in [("gt", joints)] + list(series.items()):
        vel = time_derivative(x.view(1, seq_len, NUM_BODY_JOINTS, 3), seconds, valid)
        for group, idx in JOINT_GROUPS.items():
            add_bands(spec, f"{key}|{group}", band_power(vel[:, :, list(idx)], dt, edge))


# ------------------------------------------------------------------ report

def table(header, rows):
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(["| " + " | ".join(header) + " |",
                      "|" + "|".join(["---"] * len(header)) + "|"] + body) + "\n"


def write_report(args, cfg, dump, acc, sweep, best, spec1, spec3, records, scenes):
    sigma, out = dump["sigma"], []
    avg = lambda xs: float(np.nanmean(xs))                                   # noqa: E731
    rel = lambda d, r: f"{100 * acc.rms(d) / acc.rms(r):.1f}"                # noqa: E731
    out.append(HEADER.format(cfg=args.config, ckpt=args.checkpoint, scenes=args.limit_scenes,
                             clips=dump["clips"], cap=cfg["data"]["eval_max_frames"],
                             uniform=dump["uniform_clips"], sigma=sigma))
    out.append(DEFINITIONS)
    out.append("## 1a. RMS of the GT derivative targets\n")
    out.append(table(("quantity", "estimator", "RMS"), [
        (q, e, f"{acc.rms(k):.4f}") for q, e, k in (
            ("velocity (m/s)", "v* = G(D q)  [current]", "v_star"),
            ("", "v_c = D G(q)  [consistent]", "v_c"), ("", "D q  [raw]", "v_raw"),
            ("acceleration (m/s^2)", "a* = G(D G(D q))  [current]", "a_star"),
            ("", "a_c = d2 G(q)  [consistent]", "a_c_tight"), ("", "D2 G(q)", "a_c_comp"),
            ("", "D2 q  [raw]", "a_raw_comp"), ("", "d2 q  [raw]", "a_raw_tight"),
            ("ang. velocity (rad/s)", "w* = G(w)  [current]", "w_star"),
            ("", "w_c from R_s  [consistent]", "w_c"), ("", "w  [raw]", "w_raw"),
            ("ang. accel (rad/s^2)", "aa* = G(D G(w))  [current]", "aa_star"),
            ("", "aa_c from R_s  [consistent]", "aa_c"), ("", "D w  [raw]", "aa_raw_comp"),
            ("", "midpoint d2 R  [raw]", "aa_raw_tight"))]))
    out.append("## 1b. Disagreement between the current and the consistent targets\n")
    out.append(table(("difference", "RMS", "% of the current target's RMS"), [
        (lbl, f"{acc.rms(d):.4f}", rel(d, r)) for lbl, d, r in (
            ("v* - v_c", "d_v", "v_star"), ("a* - a_c (tight d2)", "d_a_tight", "a_star"),
            ("a* - D2 G(q) (composed)", "d_a_comp", "a_star"), ("w* - w_c", "d_w", "w_star"),
            ("aa* - aa_c", "d_aa", "aa_star"))]))
    out.append("Consistent-target width sweep, acceleration `a_c(s) = d2 G_s(q)`:\n")
    out.append(table(("s (s)", "RMS a_c", "RMS(a* - a_c)"),
                     [(s, f"{sweep.rms(f'a_c@{s}'):.4f}", f"{sweep.rms(f'd_a@{s}'):.4f}")
                      for s in SWEEP]))
    out.append("## 1c. Effective single-Gaussian width of the current targets\n")
    out.append("Least-squares fit of s in `G_s(raw derivative) ~ current target`, grid "
               f"{SIGMA_GRID[0]}..{SIGMA_GRID[-1]} s step {SIGMA_GRID[1]:.4f}, over the same "
               "masked rows.\n")
    out.append(table(("target", "raw source", "best-fit sigma (s)", "sigma / label_smooth_sec"),
                     [(t, r, f"{best[k]:.4f}", f"{best[k] / sigma:.2f}") for t, r, k in
                      (("v*", "D q", "v"), ("a*", "D2 q", "a"), ("w*", "w", "w"),
                       ("aa*", "D w", "aa"))]))
    out.append("## 1d. Spectral attenuation of the targets (amplitude ratio per band)\n")
    out.append(f"Per-clip rFFT over the {dump['uniform_clips']} uniformly spaced, fully valid "
               "clips; `ceil(2 sigma / dt)` frames trimmed at both ends, mean removed, Hann "
               "window (identical on both series, so the RATIO is unaffected). Entry = "
               "sqrt(band power / band power of the raw estimator) = an amplitude ratio; "
               "1 = untouched.\n")
    out.append(table(("ratio", *BAND_NAMES), [
        (lbl, *[f"{ratio(spec1, n, d, b):.3f}" for b in BAND_NAMES]) for n, d, lbl in (
            ("v_star", "v_raw", "v* / D q"), ("v_c", "v_raw", "v_c / D q"),
            ("a_star", "a_raw_comp", "a* / D2 q"), ("a_c_tight", "a_raw_tight", "a_c / d2 q"),
            ("w_star", "w_raw", "w* / w"), ("aa_star", "aa_raw_comp", "aa* / D w"))]))

    out.append("## 2. Gradient conflict on the arm-F model\n")
    out.append(GRAD_INTRO.format(
        pos=", ".join(f"{t}={cfg['smplx_supervision']['loss'][t]}" for t in POS_TERMS),
        der=", ".join(f"{t}={cfg['motion_supervision']['loss'][t]}" for t in DER_TERMS)))
    out.append(table(("targets", "quantity", "mean cosine", "s.d. over clips"), [
        (tag, lbl, f"{avg([x[k] for x in records[tag]]):.4f}",
         f"{np.nanstd([x[k] for x in records[tag]]):.4f}")
        for tag in ("current", "consistent")
        for k, lbl in (("cos_joints_all", "cos on joints_world (22 joints)"),
                       ("cos_root_rot", "cos on root_rot_world"),
                       ("cos_params", "cos on the pose-head parameters"))]))
    out.append("Gradient norm of each individual term (mean over clips):\n")
    out.append(table(("term", "targets", "norm on joints_world", "norm on root_rot_world",
                      "norm on pose-head params"), [
        (name, tag if name in DER_TERMS else "-",
         *[f"{avg([x['terms'][name][k] for x in records[tag]]):.4e}"
           for k in ("joints", "root_rot", "params")])
        for name in POS_TERMS + DER_TERMS
        for tag in (("current", "consistent") if name in DER_TERMS else ("current",))]))
    out.append("Group gradient norms (mean over clips):\n")
    out.append(table(("targets", "position on joints", "derivative on joints",
                      "position on params", "derivative on params"), [
        (tag, *[f"{avg([x[k] for x in records[tag]]):.4e}" for k in
                ("norm_joints_pos", "norm_joints_der", "norm_params_pos", "norm_params_der")])
        for tag in ("current", "consistent")]))
    out.append("Per-joint cosine on `joints_world` (mean over clips):\n")
    out.append(table(("SMPL-X body joint", "cos (current targets)", "cos (consistent targets)"),
                     [(j, *[f"{avg([x['cos_per_joint'][j] for x in records[t]]):.4f}"
                            for t in ("current", "consistent")])
                      for j in range(NUM_BODY_JOINTS)]))

    out.append("## 3. Velocity attenuation (shrinkage monitor)\n")
    out.append("Central-difference world velocity of the 22 body joints; entry = "
               "sqrt(band power of the body / band power of the kindyn GT) over the same "
               "uniform clips, trimming and window as 1d. 1.0 = the GT's amplitude.\n")
    for group in JOINT_GROUPS:
        out.append(f"**{group}**\n")
        out.append(table(("body", *BAND_NAMES),
                         [(lbl, *[f"{ratio(spec3, f'{k}|{group}', f'gt|{group}', b):.3f}"
                                  for b in BAND_NAMES]) for k, lbl in BODIES]))
    out.append("Max |joints_world - joints_world_in| with the pose head's last linear zeroed: "
               f"{dump['zero_head_max_gap_m']:.3e} m.\n")
    out.append(DISCREPANCIES.format(
        eff=best["a"], eff_ratio=best["a"] / sigma, a_star=acc.rms("a_star"),
        a_raw=acc.rms("a_raw_comp"), frac=acc.rms("a_star") / acc.rms("a_raw_comp"),
        d_a=rel("d_a_tight", "a_star"), d_aa=rel("d_aa", "aa_star"), d_v=rel("d_v", "v_star"),
        hf=ratio(spec1, "a_star", "a_raw_comp", ">3Hz"),
        s1hf=ratio(spec3, "stage1_unsmoothed|all22", "gt|all22", ">3Hz"),
        ref=ratio(spec3, "refined|all22", "gt|all22", "broadband"),
        ref13=ratio(spec3, "refined|all22", "gt|all22", "1-3Hz"),
        ref_lo=ratio(spec3, "refined|all22", "gt|all22", "<1Hz"),
        **{f"c_{b}{i}": ratio(spec3, f"{k}|all22", "gt|all22", band)
           for b, band in (("lo", "<1Hz"), ("hi", "1-3Hz"))
           for i, k in ((1, "stage1_unsmoothed"), (2, "stage1_smoothed"), (3, "refined"))},
        cos_now=avg([x["cos_params"] for x in records["current"]]),
        cos_new=avg([x["cos_params"] for x in records["consistent"]]),
        cosj_now=avg([x["cos_joints_all"] for x in records["current"]]),
        cosj_new=avg([x["cos_joints_all"] for x in records["consistent"]]),
        gap=dump["zero_head_max_gap_m"]))
    out.append("## Scenes used\n\n```\n" + "\n".join(sorted(set(scenes))) + "\n```\n")
    (args.out / "RESULTS.md").write_text("\n".join(out))


HEADER = """# Target-consistency audit (docs/old/plan.md Phase -1 item 1)

Config `{cfg}`, checkpoint `{ckpt}` — round-4 arm F on the cached pose-token path.
Test split, first {scenes} scenes = {clips} clips (the evaluation protocol: one clip per
(scene, person), the longest valid run at the auto stride, capped at `data.eval_max_frames`
= {cap} rows). {uniform} of those clips are fully valid AND uniformly spaced; only those
enter the spectra. `motion_supervision.label_smooth_sec` sigma = {sigma}. Every number is
also in `raw.json` next to this file.
"""

DEFINITIONS = """## Definitions

* `D` = `model.refiner.time_derivative`: the central difference at the frame at the clip's
  real spacing (one-sided at the end of a valid run, 0 with no valid neighbour). `D2` = the
  COMPOSED `D(D .)`, span +-2 frames — the operator BOTH the current acceleration target and
  `MotionLoss.pose_derivatives` (the prediction side) use. `d2` = the TIGHT centred second
  difference `2((x[t+1]-x[t])/h_f - (x[t]-x[t-1])/h_b)/(h_b+h_f)`, span +-1 frame.
* `G_s` = `model.refiner.gaussian_smooth`, the masked Gaussian of width s seconds;
  `R_s = smooth_rotations(R, s)` its SO(3) counterpart.
* **current targets** (`MotionLoss.targets`, verbatim): `v* = G_s(D q)`, `a* = G_s(D v*)`;
  angular `w* = G_s(R w_body)` with `w_body = angular_velocity(R)`, `aa* = G_s(D w*)`. The
  smoothing supports are `stencil_valid(valid, 1)` / `(valid, 2)` as in the loss.
* **consistent targets** (this script): one Gaussian on the trajectory, then aligned
  stencils — `q_s = G_s(q)`, `v_c = D q_s`, `a_c = d2 q_s`; rotations `R_s`, then
  `w_c[t]` = mean of the two adjacent midpoint increments `log(R_t^T R_t+1)/h` mapped to the
  world (algebraically identical to `angular_velocity(R_s)`), `aa_c[t]` = their difference
  over the half-span `(t[t+1]-t[t-1])/2`. In Part 2 the PREDICTION side uses exactly the
  same stencils, so target and prediction are aligned on both sides.
* **rows**: `stencil_valid(valid, max(1|2, ceil(2 sigma / dt)))` — the MotionLoss mask at
  sigma = 0.12, used for EVERY variant so all RMS values are row-matched. **RMS** = sqrt(sum
  over masked rows and channels of x^2 / element count), on the 22 world body joints.
"""

GRAD_INTRO = """Per clip, `model.eval()`, gradients enabled (only the refiner trains; the
wrapper and the stage-1 SMPL-X head are frozen). Each term is scored the way the trainer
scores it in a single process: `numerator / max(mass, 1)`, the config weight already inside
the numerator. Position group = {pos}. Derivative group = {der} (the four motion-HEAD terms
set to 0 in an in-memory copy of the config; the motion head stays in `outputs`). Gradients
by `torch.autograd.grad` w.r.t. `out["smplx"]["joints_world"]` (first 22 joints),
`out["smplx"]["root_rot_world"]` and the flattened parameters of
`model.refiner.heads["pose"]`. `orient` / `pose` (they read `root_6d` / `body_6d`, siblings
of the FK output) and the two `pose_ang_*` terms have an exactly zero Jacobian w.r.t.
`joints_world`, so the joints-level cosine compares {{kp3d, root_bias, root_shape}} against
{{pose_vel, pose_acc}}; the parameter-level cosine sees every term.
"""

DISCREPANCIES = """## What disagrees with the round-5 context file

Factual list; no verdicts.

1. The context's "Motion targets" line calls the targets "0.12 s-smoothed GT derivatives"
   with "~0.17 s effective on acc". Measured effective single-Gaussian width of `a*` over
   `D2 q`: **{eff:.4f} s** = {eff_ratio:.2f} x `label_smooth_sec` (0.12 x sqrt(2) = 0.1697):
   the parenthetical is right, the headline "0.12 s" is not the width the acceleration
   target carries. `CLAUDE.md`'s loss table gives the single width only.
2. It is a target/prediction mismatch, not just a wider kernel: the prediction side
   (`MotionLoss.pose_derivatives`) is the RAW `D2` of the refined pose, RMS(`a*`) =
   {a_star:.4f} vs RMS(`D2 q`) = {a_raw:.4f} — the target is {frac:.2f} of the estimator it
   is compared against, and above 3 Hz it keeps {hf:.3f} of `D2 q`'s amplitude.
3. Current vs consistent acceleration targets differ by {d_a} % of the current target's own
   RMS ({d_aa} % angular); the velocity targets differ by {d_v} % only — a Gaussian commutes
   with a central difference on a uniform grid, so the inconsistency is created by the
   SECOND application of the Gaussian, not by the smoothing itself.
4. The context reports the velocity attenuation of the pointwise velocity losses as "0.49
   of GT". Measured here on arm F, world joint velocity, all 22 joints: broadband
   {ref:.3f}, 1-3 Hz {ref13:.3f}, <1 Hz {ref_lo:.3f}. Same region, but strongly
   band-dependent, and it is not one mechanism: the chain stage-1-unsmoothed ->
   stage-1-smoothed -> refined is {c_lo1:.3f} -> {c_lo2:.3f} -> {c_lo3:.3f} at <1 Hz and
   {c_hi1:.3f} -> {c_hi2:.3f} -> {c_hi3:.3f} at 1-3 Hz, i.e. the input Gaussian does most of
   the 1-3 Hz reduction and the transformer + pose head do most of the <1 Hz one.
5. A ratio below 1 is not automatically shrinkage of signal: the unsmoothed stage-1 body
   carries {s1hf:.2f} x the GT's velocity amplitude above 3 Hz, so that band of the input is
   dominated by per-frame noise. Only the <1 Hz and 1-3 Hz columns measure shrinkage.
6. "The model with the pose-delta head zeroed" is not a distinct body: with the head's last
   linear zeroed the pose delta is the identity, so the refined FK output equals
   `joints_world_in` to {gap:.1e} m. Reported anyway; a fourth row (stage 1 with the input
   smoothing off) was added in its place.
7. Aligning the stencils changes the conflict on the pose-head parameters from
   {cos_now:.4f} to {cos_new:.4f}, and on `joints_world` from {cosj_now:.4f} to
   {cosj_new:.4f}; the plan's falsification clause for item 1 should be read against these.
   What clearly moves is the derivative group's gradient MAGNITUDE (the tight `d2` stencil
   passes a wider band than the smoothed composed one).
"""


# ------------------------------------------------------------------ driver

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/final.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--limit-scenes", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("output_2/audits/targets"))
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)

    model, cfg = load_model(args.config, args.checkpoint, device)
    sigma = float(cfg["motion_supervision"]["label_smooth_sec"])
    _, test_sets = build_datasets(cfg, signal_needs(cfg), limit_scenes=args.limit_scenes)
    _, loader = build_loaders(cfg, [], test_sets)
    mcfg = copy.deepcopy(cfg)
    for q in QUANTITIES:                    # head terms off: only the pose_* terms are measured
        mcfg["motion_supervision"]["loss"][q] = 0.0
    losses = {"smplx": SmplxLoss(cfg, model, device), "current": MotionLoss(mcfg, model, device),
              "consistent": ConsistentMotionLoss(mcfg, model, device)}

    acc, sweep, spec1, spec3 = Acc(), Acc(), {}, {}
    fit = {k: np.zeros(len(SIGMA_GRID)) for k in ("v", "a", "w", "aa")}
    records = {"current": [], "consistent": []}
    scenes, n_uniform, n_clips = [], 0, 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        scenes.append(batch["key"][0].split("#")[0])
        n_clips += 1
        n_uniform += part1(batch, device, sigma, acc, fit, spec1, sweep)
        with torch.enable_grad():
            out = model(batch)
            part2(model, out, batch, losses, records)
        with torch.no_grad():
            part3(batch, {"refined": out["smplx"]["joints_world"][:, :NUM_BODY_JOINTS].detach(),
                          "stage1_smoothed": out["smplx"]["joints_world_in"][:, :NUM_BODY_JOINTS]},
                  device, sigma, spec3)
        del out

    head = model.refiner.heads["pose"]
    saved = [head[2].weight.detach().clone(), head[2].bias.detach().clone()]
    zero_gap = 0.0
    with torch.no_grad():
        head[2].weight.zero_()
        head[2].bias.zero_()
        for batch in loader:                                     # pose-delta head zeroed
            batch = batch_to_device(batch, device)
            out = model(batch)
            zero_gap = max(zero_gap, float((out["smplx"]["joints_world"]
                                            - out["smplx"]["joints_world_in"]).abs().max()))
            part3(batch, {"pose_head_zeroed": out["smplx"]["joints_world"][:, :NUM_BODY_JOINTS]},
                  device, sigma, spec3)
        head[2].weight.copy_(saved[0])
        head[2].bias.copy_(saved[1])
        widths = model.refiner.root_smooth_sec, model.refiner.pose_smooth_sec
        model.refiner.root_smooth_sec = model.refiner.pose_smooth_sec = 0.0
        for batch in loader:                                     # input smoothing off
            batch = batch_to_device(batch, device)
            out = model(batch)
            part3(batch, {"stage1_unsmoothed": out["smplx"]["joints_world_in"][:, :NUM_BODY_JOINTS],
                          "refined_unsmoothed": out["smplx"]["joints_world"][:, :NUM_BODY_JOINTS]},
                  device, sigma, spec3)
        model.refiner.root_smooth_sec, model.refiner.pose_smooth_sec = widths

    best = {k: float(SIGMA_GRID[int(np.argmin(v))]) for k, v in fit.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    dump = {"config": str(args.config), "checkpoint": args.checkpoint, "sigma": sigma,
            "clips": n_clips, "uniform_clips": n_uniform, "scenes": scenes,
            "rms": {k: acc.rms(k) for k in acc.sq}, "sweep": {k: sweep.rms(k) for k in sweep.sq},
            "effective_sigma": best, "spec_targets": {f"{k}|{b}": v for (k, b), v in spec1.items()},
            "spec_velocity": {f"{k}|{b}": v for (k, b), v in spec3.items()},
            "grad": records, "zero_head_max_gap_m": zero_gap}
    (args.out / "raw.json").write_text(json.dumps(dump, indent=1))
    write_report(args, cfg, dump, acc, sweep, best, spec1, spec3, records, scenes)
    print(f"wrote {args.out / 'RESULTS.md'} ({n_clips} clips, {n_uniform} uniform)")


if __name__ == "__main__":
    main()
