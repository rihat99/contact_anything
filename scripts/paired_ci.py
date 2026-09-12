"""Paired bootstrap (clustered by source video) over whole-scene prediction dumps.

Scores two or more runs' ``predictions/<scene>.npz`` dumps (``scripts/predict_test.py``)
against the corpus test labels on the SAME rows, and reports every metric's difference
to the FIRST run (the reference) with a 95 % percentile interval from a paired bootstrap
that resamples SOURCE VIDEOS (scene id minus its ``_NNNN`` suffix), not frames.

Rows: ``--protocol whole`` (default) scores every ``covered`` person-frame of every dump
(the dump stride); ``--protocol capped`` reproduces ``scripts/evaluate.py`` — one clip per
(scene, person), the longest valid run at stride ``max(1, round(fps / 25))``, first 120
rows. The row set is the intersection of the runs' ``covered`` masks, so every run is
scored on exactly the same frames.

Metrics, aggregated the way the repo aggregates them (pooled counts / sums, never a mean
of per-clip scores):

* contact — micro precision / recall / F1 at 0.5 and per-group F1 over the elements the
  manual annotation supervises (``model/loss/contact.py``);
* contact transitions — onset (0->1) and offset (1->0) boundary F1 with a +-2 row tolerance,
  one-to-one nearest matching of same-sign transitions inside a contiguous supervised stretch;
* force — MAE (bw) on in-contact rows, angle (deg) on rows with ``|f_gt| >= 0.1`` bw, mean
  ``|f|`` off contact (``model/loss/force.py``; the dumps hold WORLD-frame forces, so the
  loader's root-frame GT is rotated back with the kindyn root rotation);
* pose — MPJPE and PA-MPJPE (mm) of the 22 body joints in the camera frame after hip-mean
  alignment (``model/loss/smplx.py::pose_metric_stats``).

    python scripts/paired_ci.py output/<ref_run> output/<run> [...] \
        --protocol whole --json output_2/audits/paired_ci/whole.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.base import longest_valid_run                            # noqa: E402
from data.climbing_videos import kindyn, scene as scene_io         # noqa: E402
from model.loss import KINDYN_GROUP_NAMES                          # noqa: E402

DATASET_YAML = Path(__file__).resolve().parents[1] / "configs" / "datasets" / "climbing_videos.yaml"
CONTACT_LEVEL = 1
EVAL_MAX_FRAMES = 120
REFERENCE_FPS = 25.0
THRESHOLD = 0.5
#: force_supervision defaults the metrics are defined at (configs/base.yaml).
OUTLIER_BW, DIRECTION_MIN_BW = 4.0, 0.1
#: model/loss/smplx.py: GT rows behind the camera are dropped; hips = joints 1, 2.
MIN_DEPTH_M, NUM_BODY_JOINTS, HIPS = 0.25, 22, [1, 2]
#: Rows of tolerance when matching a predicted contact transition to a GT one.
TOLERANCE = 2
EPS = 1e-8

#: Additive sufficient statistics, in vector order.
STAT_NAMES = tuple(
    f"{group}_{count}" for group in KINDYN_GROUP_NAMES for count in ("tp", "fp", "fn", "tn")
) + ("onset_match", "onset_pred", "onset_gt", "offset_match", "offset_pred", "offset_gt",
     "force_sum", "force_n", "angle_sum", "angle_n", "off_sum", "off_n",
     "mpjpe_sum", "mpjpe_n", "pa_sum", "pa_n")
IDX = {name: i for i, name in enumerate(STAT_NAMES)}
CONTACT_SLICE = slice(0, 4 * len(KINDYN_GROUP_NAMES))


def video_id(scene: str) -> str:
    """Source video of a scene id (``RVL7DuOL9EU_0114 -> RVL7DuOL9EU``)."""
    return re.sub(r"_\d+$", "", scene)


def load_labels(root: Path, scene: str) -> dict:
    """Test contact labels + kindyn SMPL-X and force GT of one scene."""
    data = scene_io.load_scene(root, scene, "test", CONTACT_LEVEL)
    n = len(data["frame_indices"])
    gravity = scene_io.gravity_path(root, scene)
    data.update(kindyn.load_smplx(scene, data["human_dir"], data["object_ids"], n,
                                  gravity_path=gravity))
    data.update(kindyn.load_forces(scene, data["human_dir"], data["object_ids"], n,
                                   gravity_path=gravity))
    return data


def protocol_rows(labels: dict, covered: np.ndarray, stride: int,
                  protocol: str) -> list[tuple[int, np.ndarray]]:
    """``(person, source frame indices)`` scored under ``protocol``."""
    out = []
    for person, valid in enumerate(labels["valid_mask"]):
        if protocol == "whole":
            idx = np.flatnonzero(covered[person])
        else:
            base, run_len = longest_valid_run(valid)
            if run_len < 1:
                continue
            idx = base + np.arange(min((run_len - 1) // stride + 1, EVAL_MAX_FRAMES)) * stride
            idx = idx[covered[person][idx]]
        if idx.size:
            out.append((person, idx))
    return out


def transition_counts(gt: np.ndarray, pred: np.ndarray, active: np.ndarray,
                      breaks: np.ndarray) -> np.ndarray:
    """``[onset_match, onset_pred, onset_gt, offset_...]`` over one group's row sequence.

    A transition sits between two rows that are adjacent in the sequence (no ``breaks``
    boundary between them) and both supervised; it is an onset when the label rises.
    Predicted and GT transitions of the same sign are matched one-to-one, nearest first,
    within :data:`TOLERANCE` rows.
    """
    step = active[:-1] & active[1:] & ~breaks[1:]
    out = np.zeros(6)
    for half, rise in enumerate((True, False)):
        gt_at = np.flatnonzero(step & (gt[1:] != gt[:-1]) & (gt[1:] == rise))
        pred_at = np.flatnonzero(step & (pred[1:] != pred[:-1]) & (pred[1:] == rise))
        free = np.ones(len(gt_at), bool)
        matched = 0
        for position in pred_at:
            distance = np.where(free, np.abs(gt_at - position), TOLERANCE + 1)
            if len(distance) and distance.min() <= TOLERANCE:
                free[int(distance.argmin())] = False
                matched += 1
        out[3 * half:3 * half + 3] = (matched, len(pred_at), len(gt_at))
    return out


def person_stats(labels: dict, dump: dict, person: int, idx: np.ndarray,
                 stride: int) -> np.ndarray:
    """The sufficient statistics of one (scene, person) row sequence."""
    stats = np.zeros(len(STAT_NAMES))
    tracked = labels["valid_mask"][person, idx]
    breaks = np.concatenate([[True], np.diff(idx) != stride])          # sequence gaps

    gt = labels["contact_gt"][person, idx] > 0.5                        # (M, 6)
    active = (labels["contact_valid"][person, idx]
              * labels["contact_conf"][person, idx]) > 0
    if "contact_probs" in dump:
        pred = dump["contact_probs"][person, idx] > THRESHOLD
        counts = np.stack([(pred & gt & active).sum(0), (pred & ~gt & active).sum(0),
                           (~pred & gt & active).sum(0), (~pred & ~gt & active).sum(0)], -1)
        stats[CONTACT_SLICE] = counts.reshape(-1)
        for group in range(len(KINDYN_GROUP_NAMES)):
            stats[IDX["onset_match"]:IDX["onset_match"] + 6] += transition_counts(
                gt[:, group], pred[:, group], active[:, group], breaks)

    if "forces_world" in dump:
        pred_f = dump["forces_world"][person, idx].astype(np.float64)   # (M, 6, 3) bw world
        rot = labels["smplx_root_rot"][person, idx].astype(np.float64)  # world-from-root
        gt_f = np.einsum("mij,mkj->mki", rot, labels["force_gt"][person, idx])
        valid = (labels["force_valid"][person, idx] & tracked
                 & np.isfinite(pred_f).all(axis=(-1, -2)))
        mag_gt, mag_pred = np.linalg.norm(gt_f, axis=-1), np.linalg.norm(pred_f, axis=-1)
        contact = labels["force_contact"][person, idx]
        in_contact = valid[:, None] & contact & (mag_gt <= OUTLIER_BW)
        off_contact = valid[:, None] & ~contact
        angle_rows = in_contact & (mag_gt >= DIRECTION_MIN_BW)
        cosine = (pred_f * gt_f).sum(-1) / (np.maximum(mag_gt, 1e-6)
                                            * np.maximum(mag_pred, 1e-6))
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        stats[IDX["force_sum"]] = (np.linalg.norm(pred_f - gt_f, axis=-1) * in_contact).sum()
        stats[IDX["force_n"]] = in_contact.sum()
        stats[IDX["angle_sum"]] = (angle * angle_rows).sum()
        stats[IDX["angle_n"]] = angle_rows.sum()
        stats[IDX["off_sum"]] = (mag_pred * off_contact).sum()
        stats[IDX["off_n"]] = off_contact.sum()

    ext = labels["extrinsics"][idx].astype(np.float64)
    gt_j = np.einsum("mij,mkj->mki", ext[:, :3, :3],
                     labels["smplx_joints_world"][person, idx]) + ext[:, None, :3, 3]
    pred_j = dump["joints_cam"][person, idx].astype(np.float64)
    rows = (labels["smplx_valid"][person, idx] & tracked
            & (gt_j[..., 2] > MIN_DEPTH_M).all(-1) & np.isfinite(pred_j).all(axis=(-1, -2)))
    if rows.any():
        pred_j, gt_j = pred_j[rows][:, :NUM_BODY_JOINTS], gt_j[rows][:, :NUM_BODY_JOINTS]
        pred_j = pred_j - pred_j[:, HIPS].mean(1, keepdims=True)
        gt_j = gt_j - gt_j[:, HIPS].mean(1, keepdims=True)
        stats[IDX["mpjpe_sum"]] = (
            1000.0 * np.linalg.norm(pred_j - gt_j, axis=-1).mean(-1)).sum()
        stats[IDX["pa_sum"]] = (
            1000.0 * np.linalg.norm(procrustes(pred_j, gt_j) - gt_j, axis=-1).mean(-1)).sum()
        stats[IDX["mpjpe_n"]] = stats[IDX["pa_n"]] = rows.sum()
    return stats


def procrustes(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Umeyama similarity alignment of ``pred`` onto ``gt`` per row (``(B, K, 3)``)."""
    mu_p, mu_g = pred.mean(1, keepdims=True), gt.mean(1, keepdims=True)
    p0, g0 = pred - mu_p, gt - mu_g
    u, sigma, vh = np.linalg.svd(np.einsum("bki,bkj->bij", g0, p0))
    d = np.ones_like(sigma)
    d[:, -1] = np.sign(np.linalg.det(u @ vh))
    rot = u @ (d[:, :, None] * vh)
    scale = (sigma * d).sum(-1) / np.maximum((p0 * p0).sum((1, 2)), 1e-12)
    return scale[:, None, None] * np.einsum("bij,bkj->bki", rot, p0) + mu_g


#: metric -> the statistics whose sum is its denominator (0 mass = the metric is NaN).
MASS = {"f1": CONTACT_SLICE, "precision": CONTACT_SLICE, "recall": CONTACT_SLICE,
        "onset_f1": ["onset_pred", "onset_gt"], "offset_f1": ["offset_pred", "offset_gt"],
        "transition_f1": ["onset_pred", "onset_gt", "offset_pred", "offset_gt"],
        "force_mae": ["force_n"], "force_angle_deg": ["angle_n"],
        "force_noncontact": ["off_n"], "mpjpe": ["mpjpe_n"], "pa_mpjpe": ["pa_n"]}
MASS.update({f"f1_{group}": [f"{group}_{c}" for c in ("tp", "fp", "fn", "tn")]
             for group in KINDYN_GROUP_NAMES})
METRIC_NAMES = (
    "f1", "precision", "recall") + tuple(f"f1_{g}" for g in KINDYN_GROUP_NAMES) + (
    "onset_f1", "offset_f1", "transition_f1", "force_mae", "force_angle_deg",
    "force_noncontact", "mpjpe", "pa_mpjpe")


def mass_of(stats: np.ndarray, metric: str) -> np.ndarray:
    """Denominator count of ``metric`` (``(..., S)`` statistics -> ``(...)``)."""
    columns = MASS[metric]
    if isinstance(columns, slice):
        return stats[..., columns].sum(-1)
    return sum(stats[..., IDX[name]] for name in columns)


def compute_metrics(stats: np.ndarray) -> dict[str, np.ndarray]:
    """Every metric from pooled statistics; NaN where nothing was scored."""
    stats = np.asarray(stats, np.float64)
    counts = stats[..., CONTACT_SLICE].reshape(*stats.shape[:-1], len(KINDYN_GROUP_NAMES), 4)
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    out = {"f1": 2 * tp.sum(-1) / (2 * tp.sum(-1) + fp.sum(-1) + fn.sum(-1) + EPS),
           "precision": tp.sum(-1) / (tp.sum(-1) + fp.sum(-1) + EPS),
           "recall": tp.sum(-1) / (tp.sum(-1) + fn.sum(-1) + EPS)}
    for group, name in enumerate(KINDYN_GROUP_NAMES):
        out[f"f1_{name}"] = 2 * tp[..., group] / (
            2 * tp[..., group] + fp[..., group] + fn[..., group] + EPS)
    for kind in ("onset", "offset"):
        matched = stats[..., IDX[f"{kind}_match"]]
        out[f"{kind}_f1"] = 2 * matched / (
            stats[..., IDX[f"{kind}_pred"]] + stats[..., IDX[f"{kind}_gt"]] + EPS)
    matched = stats[..., IDX["onset_match"]] + stats[..., IDX["offset_match"]]
    out["transition_f1"] = 2 * matched / (mass_of(stats, "transition_f1") + EPS)
    for name, (num, den) in {"force_mae": ("force_sum", "force_n"),
                             "force_angle_deg": ("angle_sum", "angle_n"),
                             "force_noncontact": ("off_sum", "off_n"),
                             "mpjpe": ("mpjpe_sum", "mpjpe_n"),
                             "pa_mpjpe": ("pa_sum", "pa_n")}.items():
        out[name] = stats[..., IDX[num]] / np.maximum(stats[..., IDX[den]], 1.0)
    return {name: np.where(mass_of(stats, name) > 0, value, np.nan)
            for name, value in out.items()}


def collect(runs: list[Path], root: Path, protocol: str) -> tuple[dict, list[str], dict]:
    """Per-video statistics of every run, the video order, and the shared row counts."""
    # The scenes every run dumped (the test split can change between dumps).
    scenes = sorted(set.intersection(
        *({p.stem for p in (run / "predictions").glob("*.npz")} for run in runs)))
    videos = sorted({video_id(scene) for scene in scenes})
    order = {video: i for i, video in enumerate(videos)}
    stats = {run.name: np.zeros((len(videos), len(STAT_NAMES))) for run in runs}
    counts = {"scenes": len(scenes), "sequences": 0, "rows": 0}
    for scene in scenes:
        labels = load_labels(root, scene)
        object_ids = labels["object_ids"]
        dumps, covered = {}, None
        for run in runs:
            raw = np.load(run / "predictions" / f"{scene}.npz", allow_pickle=True)
            dump = {key: scene_io.rows_by_object_id(
                np.asarray(raw[key]), raw["object_ids"], object_ids, scene, "prediction dump")
                for key in ("covered", "joints_cam", "contact_probs", "forces_world")
                if key in raw.files}
            dump["stride"] = int(raw["stride"])
            dumps[run.name] = dump
            covered = dump["covered"] if covered is None else covered & dump["covered"]
        stride = dumps[runs[0].name]["stride"]
        if any(d["stride"] != stride for d in dumps.values()):
            raise ValueError(f"{scene}: the runs' dumps use different strides")
        for person, idx in protocol_rows(labels, covered, stride, protocol):
            counts["sequences"] += 1
            counts["rows"] += len(idx)
            for name, dump in dumps.items():
                stats[name][order[video_id(scene)]] += person_stats(
                    labels, dump, person, idx, stride)
    counts["videos"] = len(videos)
    return stats, videos, counts


def bootstrap(stats: dict, names: list[str], resamples: int, seed: int) -> dict:
    """Paired differences ``reference - run`` with 95 % percentile intervals."""
    reference = names[0]
    n_videos = stats[reference].shape[0]
    draws = np.random.default_rng(seed).multinomial(
        n_videos, np.full(n_videos, 1.0 / n_videos), size=resamples)     # (R, V) counts
    point = {name: compute_metrics(stats[name].sum(0)) for name in names}
    sampled = {name: compute_metrics(draws @ stats[name]) for name in names}

    def interval(name: str, metric: str) -> dict:
        finite = (sampled[reference][metric] - sampled[name][metric])
        finite = finite[np.isfinite(finite)]
        low, high = np.percentile(finite, [2.5, 97.5]) if finite.size else (np.nan, np.nan)
        return {"delta": float(point[reference][metric] - point[name][metric]),
                "lo": float(low), "hi": float(high), "n_resamples": int(finite.size)}

    out = {name: {metric: interval(name, metric) for metric in METRIC_NAMES}
           for name in names[1:]}
    return {"point": {name: {m: float(v[m]) for m in METRIC_NAMES} for name, v in point.items()},
            "delta": out}


def fmt(value: float) -> str:
    """Fixed-point with a magnitude-dependent precision; ``-`` for NaN."""
    return f"{value:.{4 if abs(value) < 10 else 2}f}" if np.isfinite(value) else "-"


def tables(result: dict, stats: dict, names: list[str], counts: dict, protocol: str) -> str:
    """The markdown point-estimate and paired-difference tables."""
    reference = names[0]
    lines = [f"### protocol `{protocol}` — {counts['scenes']} scenes, {counts['sequences']} "
             f"(scene, person) sequences, {counts['rows']} rows, {counts['videos']} videos",
             "", "| metric | mass | videos | " + " | ".join(names) + " |",
             "|---|---:|---:|" + "---:|" * len(names)]
    for metric in METRIC_NAMES:
        mass = mass_of(stats[reference].sum(0), metric)
        clusters = int((mass_of(stats[reference], metric) > 0).sum())
        lines.append(f"| {metric} | {mass:.0f} | {clusters} | " + " | ".join(
            fmt(result["point"][name][metric]) for name in names) + " |")
    lines += ["", f"Paired bootstrap differences `{reference} - run` "
                  "(95 % percentile interval, clustered by source video):", "",
              "| metric | " + " | ".join(names[1:]) + " |",
              "|---|" + "---:|" * (len(names) - 1)]
    for metric in METRIC_NAMES:
        cells = []
        for name in names[1:]:
            d = result["delta"][name][metric]
            cells.append("-" if not np.isfinite(d["delta"]) else
                         f"{fmt(d['delta'])} [{fmt(d['lo'])}, {fmt(d['hi'])}]")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="run directories; the first is the reference")
    parser.add_argument("--protocol", choices=("whole", "capped"), default="whole")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--root", type=Path, default=None, help="corpus root (default: the dataset yaml's)")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    if len(args.runs) < 2:
        parser.error("give at least two run directories (the first is the reference)")
    root = args.root or Path(yaml.safe_load(DATASET_YAML.read_text())["root"])

    stats, videos, counts = collect(args.runs, root, args.protocol)
    names = [run.name for run in args.runs]
    result = bootstrap(stats, names, args.resamples, args.seed)
    print(tables(result, stats, names, counts, args.protocol))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "protocol": args.protocol, "runs": names, "reference": names[0],
            "resamples": args.resamples, "seed": args.seed, "counts": counts,
            "videos": videos,
            "mass": {metric: float(mass_of(stats[names[0]].sum(0), metric))
                     for metric in METRIC_NAMES},
            "stats": {name: dict(zip(STAT_NAMES, stats[name].sum(0).tolist()))
                      for name in names},
            **result}, indent=1) + "\n")


if __name__ == "__main__":
    main()
