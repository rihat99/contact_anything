"""Score a motion-token checkpoint on the canonical v1 motion-probe eval rows.

The v1 probe (``output/probe_motion_v1_20260812``) scored **centre frames** of the
30 corpus test scenes: every frame ``n`` with ``12 <= n < N - 12`` whose kindyn
motion target is valid. That is 7,561 rows over 30 scenes (coverage 0.9265) and
it is the ONLY row set the pre-registered bars are stated on, so this script
derives it from the corpus itself (identical masking rule, see
``ClimbingCorpusDataset._load_motion``) and asserts the count.

The stock test loader will not do: its windows tile with step ``T * stride``, so
their centres hit only ~14% of the canonical rows. Instead we override the
dataset's window index with one **centred** clip per canonical row (jitter is off
on test, so ``_window_start`` returns the base unchanged) and assert row identity
through each frame's ``key`` rather than trusting the arithmetic.

Reported per motion slot: 3-D RMSE and the pooled 3-component Pearson r in the
**target** axes (root-local; a twist for the pelvis under ``root_convention:
twist``), the GT 3-D RMS (= the zero prior's RMSE), the **world-vertical**
Pearson r pooled and as a per-scene median, and the least-squares slope of
predicted vs GT world-vertical values (amplitude-shrinkage check). Outliers are
never filtered.

Everything is reported against TWO target definitions: the **primary** one the
config/checkpoint trained on (fixed-seconds smoothed root trajectory) and, when
that is smoothed, the **raw** unsmoothed twist as a secondary section — the one
comparable with the published v1/v2 numbers.

``--baselines`` scores the reference rows instead of a checkpoint, on the same
canonical rows and in the same target axes: the zero prior (the RMSE reference),
the **mean prior** (the null for ``r3d``, which pools components without
centring them, so a constant already scores above zero), Gaussian smoothing of
the frozen model's own lifted pelvis trajectory (the deployable ceiling, uses the
dataset extrinsics) and — on the RAW section only — the same smoothing of the
kindyn GT pelvis (the label ceiling; against a smoothed target it is trivially
near 1.0 because it IS the target), sweeping sigma.

Usage::

    CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_motion.py \
        --config configs/climbing_corpus_motion_pelvis_t7.yaml \
        --checkpoint output/<run>/best.pth --out output/eval_motion.json
    python scripts/evaluate_motion.py --config configs/... --rows-only
    python scripts/evaluate_motion.py --config configs/... --baselines
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from contact import checkpoint as ckpt_io
from contact.config import load_config
from contact.data.climbing_corpus import (
    MOTION_JOINT_NAMES,
    ClimbingCorpusDataset,
    list_corpus_scenes,
)
from contact.data.collate import batch_to_device, make_collate
from contact.engine import forward_model
from contact.model import build_model
from contact.motion_supervision import to_world_linear
from contact.targets import TargetSpec

#: Context radius of the v1 canonical row definition (T = 25 probe window).
REF_HALF = 12
#: Expected canonical row/scene counts for this corpus (assert, never silently drop).
EXPECTED_ROWS = 7561
EXPECTED_SCENES = 30
#: Per-frame pelvis trajectory store the smoothing baselines differentiate.
TRAJ_STORE = REPO / "output" / "motion_probe_geom"
#: Gaussian smoothing widths (seconds) swept by ``--baselines``. 0 = no smoothing
#: (raw central differences of the trajectory).
BASELINE_SIGMAS = (0.0, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.3, 0.48)


def motion_supervision_source(cfg: dict, checkpoint: str | None) -> tuple[dict, str]:
    """The ``motion_supervision`` block to score under, plus where it came from.

    The head learns in the units — and against the target convention — of the
    config it trained under, so the standardize affine, ``joint_names`` and
    ``root_convention`` must all come from the **checkpoint's** stored config;
    reading the current yaml would silently mis-scale or re-frame the comparison
    for any checkpoint predating an edit. Falls back to ``--config`` only when
    there is no checkpoint (the zero-init control and the baselines).

    Checkpoints written before those keys existed get the motion-tokens-v2
    semantics they actually trained with (all seven joints, ``rotated_world``,
    unsmoothed).
    """
    if checkpoint is None:
        return cfg["motion_supervision"], "run config"
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    stored = ((ckpt.get("config") or {}).get("motion_supervision") or {})
    if not (stored.get("standardize") or {}).get("std"):
        raise RuntimeError(
            f"{checkpoint}: no motion_supervision.standardize in the checkpoint's "
            "stored config — cannot de-standardize its predictions faithfully")
    stored = dict(stored)
    stored.setdefault("joint_names", None)
    stored.setdefault("root_convention", "rotated_world")
    stored.setdefault("target_smooth_sec", 0.0)
    # `loss` only supplies the outlier threshold, and evaluation never filters
    # outliers — but a checkpoint predating the sub-schema must not KeyError.
    stored["loss"] = {**cfg["motion_supervision"]["loss"], **(stored.get("loss") or {})}
    note = "checkpoint config"
    if stored != cfg["motion_supervision"]:
        note += " (DIFFERS from --config; the checkpoint's wins)"
    return stored, note


def build_test_dataset(cfg: dict, ms: dict) -> ClimbingCorpusDataset:
    """Corpus test dataset with motion targets, windows still tiled (unused).

    The clip stride follows the run config (``1`` or ``"auto"``), so the centred
    clips this script scores span the same physical window the model trained on.
    """
    entries = [d for d in cfg["data"]["datasets"] if d["name"] == "climbing_corpus"]
    if len(entries) != 1:
        raise ValueError("evaluate_motion requires exactly one climbing_corpus dataset")
    dataset_cfg = (yaml.safe_load(Path(entries[0]["config"]).read_text()) or {})["data"]
    root = dataset_cfg["root"]
    stride = cfg["data"]["sequence"]["frame_stride"]
    return ClimbingCorpusDataset(
        root,
        scenes=list_corpus_scenes(root, "test"),
        split="test",
        frames_per_clip=int(cfg["data"]["sequence"]["frames_per_clip"]),
        frame_stride=stride if stride == "auto" else int(stride),
        jitter=False,
        seed=int(cfg["data"]["seed"]),
        contact_level=int(dataset_cfg.get("contact_level", 1)),
        require_labels=False,          # motion-only build: contact labels unused
        load_motion=True,
        motion_joint_names=ms["joint_names"],
        motion_root_convention=ms["root_convention"],
        motion_target_smooth_sec=float(ms["target_smooth_sec"]),
        motion_outlier_acc_ms2=float(ms["loss"]["outlier_acc_ms2"]),
    )


def canonical_rows(dataset: ClimbingCorpusDataset) -> list[tuple[str, int, int]]:
    """``(scene, person, frame)`` triples of the v1 canonical eval set, in order."""
    rows: list[tuple[str, int, int]] = []
    for scene in sorted(dataset._scenes):
        data = dataset._scenes[scene]
        valid = data["motion_valid"]                       # [P, N]
        n_frames = valid.shape[1]
        window = np.arange(REF_HALF, max(n_frames - REF_HALF, REF_HALF))
        for person in range(valid.shape[0]):
            rows += [(scene, person, int(n)) for n in window[valid[person, window]]]
    return rows


def centered_items(
    dataset: ClimbingCorpusDataset, rows: list[tuple[str, int, int]],
) -> list[tuple[str, int, int, int]]:
    """One centred clip per canonical row, in row order (jitter is off on test).

    Under the ``auto`` stride a clip reaches ``(T // 2) * stride`` frames either
    side of the row. The canonical set's 12-frame margin covers that for every
    corpus frame rate (stride <= 2 at T=7), but it is asserted per row rather
    than assumed — a wider stride or a new fps would silently drop rows.
    """
    half = dataset.T // 2
    items = []
    for scene, person, frame in rows:
        stride = dataset.scene_stride(scene)
        base = frame - half * stride
        positions = base + np.arange(dataset.T) * stride
        n_frames = len(dataset._scenes[scene]["frame_indices"])
        if base < 0 or positions[-1] >= n_frames:
            raise ValueError(
                f"{scene}#{person}@{frame}: no centred T={dataset.T} stride={stride} "
                f"clip fits — the canonical row set is no longer scorable")
        if not dataset._scenes[scene]["valid_mask"][person, positions].all():
            raise ValueError(
                f"{scene}#{person}@{frame}: centred clip crosses an invalid frame")
        items.append((scene, person, base, 1))
    return items


def gather_targets(
    dataset: ClimbingCorpusDataset, rows: list[tuple[str, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GT motion / world-from-root rotation / body angular velocity, in row order.

    Read straight from the scene arrays the loader caches — the same values
    ``__getitem__`` hands the collate, without paying for a second image pass.
    """
    gt, rot, omega = [], [], []
    for scene, person, frame in rows:
        data = dataset._scenes[scene]
        gt.append(data["motion_gt"][person, frame])
        rot.append(data["motion_rot"][person, frame])
        omega.append(data["motion_omega"][person, frame])
    return (torch.from_numpy(np.stack(gt)),
            torch.from_numpy(np.stack(rot)),
            torch.from_numpy(np.stack(omega)))


@torch.no_grad()
def predict(model, loader, rows, seq_len, device: str) -> torch.Tensor:
    """Centre-row **standardized** predictions for every canonical row, in order."""
    center = seq_len // 2
    preds = []
    cursor = 0
    for batch in loader:
        keys = batch.pop("keys")
        batch = batch_to_device(batch, device)
        out = forward_model(model, batch)
        motion = out["motion"]["joint_motion"].float()                 # [B, K, 6]
        n_clips = motion.shape[0] // seq_len
        rows_here = rows[cursor:cursor + n_clips]
        for i, (scene, person, frame) in enumerate(rows_here):
            oid = int(loader.dataset._scenes[scene]["object_ids"][person])
            expected = f"{scene}#{oid}@{frame}"
            got = keys[i * seq_len + center]
            if got != expected:
                raise RuntimeError(
                    f"centre-row misalignment: expected {expected}, got {got}")
        preds.append(motion.reshape(n_clips, seq_len, *motion.shape[1:])[:, center].cpu())
        cursor += n_clips
    if cursor != len(rows):
        raise RuntimeError(f"scored {cursor} rows, expected {len(rows)}")
    return torch.cat(preds)


def destandardize(pred: torch.Tensor, std_cfg: dict) -> torch.Tensor:
    """Standardized head output ``(R, K, 6)`` -> physical target-frame units."""
    mean = torch.tensor(std_cfg["mean"], dtype=torch.float32).reshape(1, -1, 6)
    std = torch.tensor(std_cfg["std"], dtype=torch.float32).reshape(1, -1, 6)
    return pred * std + mean


def twist_slot_mask(names: tuple[str, ...], root_convention: str) -> torch.Tensor:
    """``(K,)`` bool — slots whose target is a body twist (pelvis only, if any)."""
    twist = root_convention == "twist"
    return torch.tensor([twist and n == "pelvis" for n in names], dtype=torch.bool)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _ls_slope(pred: np.ndarray, gt: np.ndarray) -> float:
    """Least-squares slope of ``pred ~ a * gt + b`` — the v1 amplitude-shrinkage
    estimator (``np.polyfit(gt, pred, 1)[0]``, i.e. WITH an intercept).

    The through-origin form agrees with this to ~0.1% on acceleration but differs
    by up to ~17% on velocity (which has a nonzero mean), so the intercept form is
    the one comparable to v1's quoted numbers.
    """
    if len(gt) < 2 or gt.std() == 0:
        return float("nan")
    return float(np.polyfit(gt, pred, 1)[0])


def score(
    pred: torch.Tensor, gt: torch.Tensor, rot: torch.Tensor, omega: torch.Tensor,
    twist_slots: torch.Tensor, names: tuple[str, ...],
    rows: list[tuple[str, int, int]],
) -> dict:
    """Per-slot RMSE3D / GT RMS / pooled-3D r / world-vertical r / medians / slope.

    ``pred``/``gt`` are physical, in target axes. The world-vertical statistics go
    through :func:`to_world_linear` on BOTH sides, so a twist slot picks up its
    ``omega x v`` term consistently.
    """
    pred_world = to_world_linear(pred[..., :3], pred[..., 3:], rot, omega, twist_slots)
    gt_world = to_world_linear(gt[..., :3], gt[..., 3:], rot, omega, twist_slots)
    pred_np = pred.numpy().astype(np.float64)
    gt_np = gt.numpy().astype(np.float64)
    scenes = np.array([row[0] for row in rows])
    report: dict = {"n_rows": len(rows), "n_scenes": int(len(set(scenes)))}
    for q, quantity in enumerate(("vel", "acc")):
        sl = slice(0, 3) if quantity == "vel" else slice(3, 6)
        per_joint = {}
        for k, name in enumerate(names):
            p = pred_np[:, k, sl]
            g = gt_np[:, k, sl]
            p_vert = pred_world[q].numpy().astype(np.float64)[:, k, 1]
            g_vert = gt_world[q].numpy().astype(np.float64)[:, k, 1]
            per_scene_vert, per_scene_3d = [], []
            for s in sorted(set(scenes)):
                sel = scenes == s
                per_scene_vert.append(_pearson(p_vert[sel], g_vert[sel]))
                per_scene_3d.append(_pearson(p[sel].ravel(), g[sel].ravel()))
            per_joint[name] = {
                "rmse3d": float(np.sqrt(((p - g) ** 2).sum(-1).mean())),
                "gt_rms3d": float(np.sqrt((g ** 2).sum(-1).mean())),
                # Pooled over all three target-axis components (one r per slot).
                "r3d": _pearson(p.ravel(), g.ravel()),
                "r3d_scene_median": _median_finite(per_scene_3d),
                "vert_r": _pearson(p_vert, g_vert),
                "vert_r_scene_median": _median_finite(per_scene_vert),
                # Amplitude shrinkage, v1-comparable (with intercept).
                "vert_ls_slope": _ls_slope(p_vert, g_vert),
            }
        report[quantity] = per_joint
    return report


def _median_finite(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


def print_report(report: dict, names: tuple[str, ...], label: str = "") -> None:
    for quantity in ("vel", "acc"):
        print(f"\n{quantity}{'  ' + label if label else ''}:")
        print(f"  {'slot':>12s}  {'rmse3d':>8s}  {'gt_rms':>8s}  {'r3d':>7s}  "
              f"{'r3d_med':>7s}  {'vert_r':>7s}  {'vr_med':>7s}  {'slope':>7s}")
        for name in names:
            m = report[quantity][name]
            print(f"  {name:>12s}  {m['rmse3d']:8.4f}  {m['gt_rms3d']:8.4f}  "
                  f"{m['r3d']:+7.4f}  {m['r3d_scene_median']:+7.4f}  "
                  f"{m['vert_r']:+7.4f}  {m['vert_r_scene_median']:+7.4f}  "
                  f"{m['vert_ls_slope']:+7.4f}")


# ------------------------------------------------------------------- baselines

def _smoothed_derivatives(
    pos: np.ndarray, frame_idx: np.ndarray, fps: float, sigma_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-smoothed central differences of a world trajectory.

    Smoothing and differencing run **inside each contiguous run of frame indices**
    — 4 of the 363 tracked entries have frame gaps, and filtering across one would
    invent motion. Rows outside a run of >= 3 frames keep zero velocity and
    acceleration.

    :returns: ``(v_world, a_world)`` aligned with ``pos``'s rows.
    """
    from scipy.ndimage import gaussian_filter1d

    vel = np.zeros_like(pos, np.float64)
    acc = np.zeros_like(pos, np.float64)
    dt = 1.0 / fps
    breaks = np.flatnonzero(np.diff(frame_idx) != 1) + 1
    for chunk in np.split(np.arange(len(pos)), breaks):
        if len(chunk) < 3:
            continue
        run = pos[chunk].astype(np.float64)
        if sigma_sec > 0:
            run = gaussian_filter1d(run, sigma=sigma_sec * fps, axis=0,
                                    mode="nearest", truncate=4.0)
        vel[chunk[1:-1]] = (run[2:] - run[:-2]) / (2.0 * dt)
        acc[chunk[1:-1]] = (run[2:] - 2.0 * run[1:-1] + run[:-2]) / (dt * dt)
    return vel, acc


def trajectory_baselines(
    dataset: ClimbingCorpusDataset, rows: list[tuple[str, int, int]],
    rot: torch.Tensor, omega: torch.Tensor, twist: bool, source: str,
    sigma_sec: float,
) -> torch.Tensor:
    """Pelvis vel/acc ``(R, 1, 6)`` from a smoothed world trajectory, target axes.

    :param source: ``pred_pelvis_world`` (the frozen model's own pelvis lifted to
        world with the dataset extrinsics — the deployable baseline) or
        ``pelvis_world`` (the kindyn GT trajectory — the label ceiling).
    """
    store = np.load(TRAJ_STORE / "frames.npz", allow_pickle=False)
    entries = json.loads((TRAJ_STORE / "entries.json").read_text())
    entry_id = store["entry_id"]
    frame_idx = store["frame_idx"]
    pos_all = store[source]

    vel_w = np.zeros((len(entry_id), 3), np.float64)
    acc_w = np.zeros_like(vel_w)
    index: dict[tuple[str, int, int], int] = {}
    wanted = {row[0] for row in rows}
    for eid, entry in enumerate(entries):
        if entry["scene"] not in wanted:
            continue
        sel = np.flatnonzero(entry_id == eid)
        if len(sel) == 0:
            continue
        v, a = _smoothed_derivatives(
            pos_all[sel], frame_idx[sel], float(entry["fps"]), sigma_sec)
        vel_w[sel] = v
        acc_w[sel] = a
        for local, row in enumerate(sel):
            index[(entry["scene"], int(entry["object_id"]),
                   int(frame_idx[row]))] = int(row)

    picked = []
    for scene, person, frame in rows:
        oid = int(dataset._scenes[scene]["object_ids"][person])
        key = (scene, oid, frame)
        if key not in index:
            raise RuntimeError(
                f"{TRAJ_STORE}: no {source} row for canonical row {scene}#{oid}@{frame}")
        picked.append(index[key])
    picked = np.asarray(picked)

    # World -> target axes: v_t = R^T v_w; a_t = R^T a_w - omega x v_t (twist only).
    rot_np = rot.numpy().astype(np.float64)
    v_t = np.einsum("rji,rj->ri", rot_np, vel_w[picked])
    a_t = np.einsum("rji,rj->ri", rot_np, acc_w[picked])
    if twist:
        a_t = a_t - np.cross(omega.numpy().astype(np.float64), v_t)
    return torch.from_numpy(
        np.concatenate([v_t, a_t], -1)[:, None, :].astype(np.float32))


def run_baselines(dataset, rows, gt, rot, omega, ms: dict, oracle: bool) -> dict:
    """Zero/mean priors + smoothed pred-pelvis (+ smoothed kindyn), pelvis only.

    :param oracle: include the ``gtsmooth`` (kindyn GT trajectory) rows. They are
        the label ceiling against the RAW target, but against a smoothed target
        they are trivially near 1.0 — the smoothed target IS a smoothed kindyn
        derivative — so they carry no information there.
    """
    names = tuple(ms["joint_names"] or MOTION_JOINT_NAMES)
    if "pelvis" not in names:
        raise ValueError("--baselines scores the pelvis slot; it is not configured")
    k = names.index("pelvis")
    gt_p = gt[:, k:k + 1]
    twist_slots = twist_slot_mask(("pelvis",), ms["root_convention"])
    twist = bool(twist_slots[0])

    out = {"convention": ms["root_convention"],
           "target_smooth_sec": float(ms["target_smooth_sec"]), "rows": {}}
    # Zero prior: the RMSE reference (= GT RMS). Its correlations are nan BY
    # CONSTRUCTION — a constant prediction has zero variance — which is why the
    # mean prior below is the row to read a learned r3d against.
    out["rows"]["zero_prior"] = score(
        torch.zeros_like(gt_p), gt_p, rot, omega, twist_slots, ("pelvis",), rows)
    print_report(out["rows"]["zero_prior"], ("pelvis",), "zero_prior")
    # Mean prior: predict the training mean of every component, forever. This is
    # the NULL for `r3d`, which pools the three components without centring them
    # per component — so a constant that merely reproduces the between-component
    # offsets already scores above zero (the head is zero-init to exactly this
    # prediction, so this row is also its score before any training). `vert_r` is
    # likewise nonzero rather than nan: a CONSTANT ROOT-FRAME vector still has a
    # varying world-vertical projection once the per-frame rotation is applied.
    # The same constant is scored against both target definitions.
    mean_row = torch.tensor(
        ms["standardize"]["mean"], dtype=torch.float32).reshape(-1, 6)[k]
    out["rows"]["mean_prior"] = score(
        mean_row.view(1, 1, 6).expand(len(rows), 1, 6).contiguous(),
        gt_p, rot, omega, twist_slots, ("pelvis",), rows)
    print_report(out["rows"]["mean_prior"], ("pelvis",), "mean_prior")
    sources = [("smooth", "pred_pelvis_world")]
    if oracle:
        sources.append(("gtsmooth", "pelvis_world"))
    for prefix, source in sources:
        for sigma in BASELINE_SIGMAS:
            pred = trajectory_baselines(
                dataset, rows, rot, omega, twist, source, sigma)
            key = f"{prefix}_sigma{sigma:g}"
            out["rows"][key] = score(
                pred, gt_p, rot, omega, twist_slots, ("pelvis",), rows)
            acc = out["rows"][key]["acc"]["pelvis"]
            vel = out["rows"][key]["vel"]["pelvis"]
            print(f"{key:<20s} acc r3d {acc['r3d']:+.4f} vert_r {acc['vert_r']:+.4f} "
                  f"RMSE3D {acc['rmse3d']:7.4f} | vel r3d {vel['r3d']:+.4f} "
                  f"vert_r {vel['vert_r']:+.4f} RMSE3D {vel['rmse3d']:6.4f}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rows-only", action="store_true",
                    help="derive + assert the canonical row set and exit (no model)")
    ap.add_argument("--baselines", action="store_true",
                    help="score the zero/smoothing reference rows instead of a model")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not cfg["model"]["motion_head"]["enabled"]:
        raise ValueError("evaluate_motion requires model.motion_head.enabled=true")

    ms, ms_note = motion_supervision_source(
        cfg, None if args.baselines else args.checkpoint)
    names = tuple(ms["joint_names"] or MOTION_JOINT_NAMES)
    print(f"scoring slots {list(names)} under root_convention="
          f"{ms['root_convention']!r}, target_smooth_sec={ms['target_smooth_sec']} "
          f"from the {ms_note}")

    dataset = build_test_dataset(cfg, ms)
    rows = canonical_rows(dataset)
    n_scenes = len({row[0] for row in rows})
    print(f"canonical rows: {len(rows)} over {n_scenes} scenes  (strides "
          f"{sorted({dataset.scene_stride(s) for s in dataset._scenes})})")
    if len(rows) != EXPECTED_ROWS or n_scenes != EXPECTED_SCENES:
        raise RuntimeError(
            f"canonical row set changed: got {len(rows)} rows / {n_scenes} scenes, "
            f"expected {EXPECTED_ROWS} / {EXPECTED_SCENES} — the v1 comparison is void")
    if args.rows_only:
        return 0

    gt, rot, omega = gather_targets(dataset, rows)
    # Secondary view: the RAW (unsmoothed) twist target, for comparability with
    # the v1/v2 numbers. Its rows are the same — smoothing does not touch the
    # validity rule — which is asserted rather than assumed.
    raw_ms = {**ms, "target_smooth_sec": 0.0}
    raw = None
    if float(ms["target_smooth_sec"]) > 0.0:
        raw_dataset = build_test_dataset(cfg, raw_ms)
        if canonical_rows(raw_dataset) != rows:
            raise RuntimeError("raw-target dataset yields a different canonical row set")
        raw = (raw_dataset,) + gather_targets(raw_dataset, rows)

    if args.baselines:
        print(f"\n=== primary task: target_smooth_sec={ms['target_smooth_sec']} ===")
        report = {"primary": run_baselines(
            dataset, rows, gt, rot, omega, ms, oracle=float(ms["target_smooth_sec"]) == 0.0)}
        if raw is not None:
            print("\n=== secondary: RAW targets (v1/v2-comparable) ===")
            report["raw"] = run_baselines(
                raw[0], rows, raw[1], raw[2], raw[3], raw_ms, oracle=True)
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model, _ = build_model(cfg, args.device)
        if args.checkpoint:
            ckpt_io.load(args.checkpoint, model, config=cfg, map_location=args.device)
        else:
            print("WARNING: no --checkpoint; scoring the zero-init (mean-predicting) head")
        model.eval()

        seq_len = int(cfg["data"]["sequence"]["frames_per_clip"])
        dataset._items = centered_items(dataset, rows)
        base_collate = make_collate(
            tuple(model.cfg.MODEL.IMAGE_SIZE), TargetSpec.from_config(cfg),
            motion_joints=len(names))

        def collate(items):
            out = base_collate(items)
            out["keys"] = [f["key"] for clip in items for f in clip]
            return out

        loader = DataLoader(
            dataset,
            batch_size=max(1, int(cfg["data"]["frames_per_batch"]) // seq_len),
            shuffle=False,
            num_workers=int(cfg["data"]["num_workers"]),
            collate_fn=collate,
            pin_memory=True,
        )
        pred = destandardize(
            predict(model, loader, rows, seq_len, args.device), ms["standardize"])
        twist_slots = twist_slot_mask(names, ms["root_convention"])
        report = {
            "motion_supervision_source": ms_note,
            "target_smooth_sec": float(ms["target_smooth_sec"]),
            "primary": score(pred, gt, rot, omega, twist_slots, names, rows),
        }
        print(f"\n=== primary task: target_smooth_sec={ms['target_smooth_sec']} ===")
        print_report(report["primary"], names)
        if raw is not None:
            # Same predictions, scored against the RAW twist target: the number
            # that lines up with the v1/v2 tables (which had no label smoothing).
            report["raw"] = score(
                pred, raw[1], raw[2], raw[3], twist_slots, names, rows)
            print("\n=== secondary: RAW targets (v1/v2-comparable) ===")
            print_report(report["raw"], names)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
