"""Evaluate a trained contact/force checkpoint on validation or manual video test data.

Builds the model + val loader from ``--config`` (the same
``contact.data.collate.make_loaders`` the trainer uses), runs the frozen base +
loaded contact weights over the val split, and reports micro-averaged per-target
precision / recall / F1 / F2 / IoU via ``contact.metrics``. Works for both a
vertex config (e.g. DAMON) and a joint config. ``--split test`` is supported for
the ClimbingVideos corpus and reads the manually annotated DB test scenes.

When the run enables the force branch + the RNEA physics loss, a second
**physics-consistency** section is reported (headline ``physics_residual``,
per-extremity force magnitudes by predicted contact state, gate-violation rates,
vertical-force-sum distribution) — the contact section is left unchanged. Lacking
a trained force checkpoint, ``--warm-start`` builds the untrained force branch
from the config's ``model.init_contact_checkpoint`` to exercise the pipeline.

Usage::

    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
        --config configs/climbing_videos_joint.yaml \
        --checkpoint output/<run>/best.pth --out output/eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from contact import checkpoint as ckpt_io
from contact.config import load_config
from contact.data.climbing_corpus import ClimbingCorpusDataset
from contact.data.collate import batch_to_device, make_collate, make_loaders
from contact.engine import forward_contact, forward_model, select_temporal_supervision
from contact.metrics import (
    add_counts,
    contact_counts,
    contact_counts_per_dim,
    prf1,
    zero_counts,
)
from contact.model import build_model
from contact.physics import EXTREMITY_OUTPUT_NAMES
from contact.targets import TargetSpec


@torch.no_grad()
def evaluate(
    model,
    loader,
    targets: list[str],
    device: str,
    *,
    threshold: float = 0.5,
    curve_thresholds: tuple[float, ...] = (),
    output_names: dict[str, tuple[str, ...]] | None = None,
    target_frame: str = "all",
) -> dict:
    thresholds = tuple(sorted(set((float(threshold), *map(float, curve_thresholds)))))
    counts = {th: {t: zero_counts() for t in targets} for th in thresholds}
    per_output = {}
    for target, names in (output_names or {}).items():
        per_output[target] = {name: zero_counts() for name in names}
    for batch in loader:
        batch = batch_to_device(batch, device)
        contact = forward_contact(model, batch)
        logits_by_target, selected_targets = select_temporal_supervision(
            {t: contact[f"{t}_logits"] for t in targets},
            batch["targets"],
            int(batch.get("seq_len", 1)),
            target_frame,
        )
        for t in targets:
            tgt = selected_targets[t]
            logits = logits_by_target[t]
            for th in thresholds:
                add_counts(counts[th][t], contact_counts(
                    logits, tgt["gt"], tgt["mask"], threshold=th))
            if t in per_output:
                dim_counts = contact_counts_per_dim(
                    logits, tgt["gt"], tgt["mask"], threshold=threshold)
                for name, current in zip(per_output[t], dim_counts):
                    add_counts(per_output[t][name], current)

    results = {
        t: {**prf1(counts[float(threshold)][t]), **counts[float(threshold)][t]}
        for t in targets
    }
    for target, named_counts in per_output.items():
        results[target]["per_output"] = {
            name: {**prf1(value), **value} for name, value in named_counts.items()
        }
    if curve_thresholds:
        for target in targets:
            results[target]["threshold_curve"] = [
                {"threshold": th, **prf1(counts[th][target]), **counts[th][target]}
                for th in thresholds
            ]
    return results


def _stats(values: torch.Tensor) -> dict:
    """Count + mean + median + p90 of a 1-D magnitude tensor (nan when empty)."""
    if values.numel() == 0:
        nan = float("nan")
        return {"n": 0, "mean": nan, "p50": nan, "p90": nan}
    quant = torch.quantile(values, torch.tensor([0.5, 0.9], dtype=values.dtype))
    return {"n": int(values.numel()), "mean": float(values.mean()),
            "p50": float(quant[0]), "p90": float(quant[1])}


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation of two 1-D tensors (nan when either is constant)."""
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(xc.norm() * yc.norm())
    return float((xc @ yc) / denom) if denom > 0 else float("nan")


def _residual_quantiles(res: torch.Tensor) -> dict:
    """median / p90 / p99 / max of a 1-D per-frame residual tensor."""
    q = torch.quantile(res.double(), torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64))
    return {"p50": float(q[0]), "p90": float(q[1]), "p99": float(q[2]),
            "max": float(res.max())}


def _affine_baselines(by_t: dict, *, n_shuffles: int = 5, seed: int = 0) -> dict:
    """Decisive input-dependence baselines from the affine residual decomposition (§4).

    ``by_t`` maps clip length ``T`` to concatenated per-clip tensors from
    :meth:`PhysicsLoss.affine_residual` (``r0 (Nc, nr, 6)``, ``basis (Nc, nr, 6,
    12)``, ``f_pred (Nc, nr, 12)``, ``probs (Nc, nr, 4)``). Because the normalised
    root wrench is affine in the head-frame forces (``r(f) = r0 + B·vec(f)`` per
    residual frame), the raw residual of any force assignment is closed-form — no
    RNEA re-evaluation. The network is scored against (a) zero forces, (b) the best
    global constant 12-DoF force (least squares on the accumulated normal equations),
    and (d) shuffled per-clip force trajectories: a cyclic rotation of the clips of
    each equal-T group by a random NONZERO offset — guaranteed non-identity for
    group size > 1 (an unconstrained permutation can fix points, and is always the
    identity for a single clip). Size-1 groups cannot be shuffled; they are skipped
    from the shuffled baseline and counted in ``n_unshuffled_clips``. An
    input-dependent model must beat BOTH (b) and (d).
    """
    # Best constant: minimise Σ‖r0 + B·c‖² -> normal equations A·c = -g.
    normal_a = torch.zeros(12, 12, dtype=torch.float64)
    normal_g = torch.zeros(12, dtype=torch.float64)
    for group in by_t.values():
        basis = group["basis"].reshape(-1, 6, 12).double()
        r0 = group["r0"].reshape(-1, 6).double()
        normal_a += torch.einsum("fij,fik->jk", basis, basis)
        normal_g += torch.einsum("fij,fi->j", basis, r0)
    c_star = -(torch.linalg.pinv(normal_a) @ normal_g)             # (12,)

    def residual(f_assign: torch.Tensor, group: dict) -> torch.Tensor:
        pred = group["r0"] + torch.einsum("...ij,...j->...i", group["basis"], f_assign)
        return (pred ** 2).sum(-1).reshape(-1)                     # per-frame raw residual

    zero_res, const_res, net_res = [], [], []
    for group in by_t.values():
        const = c_star.to(group["f_pred"].dtype).expand_as(group["f_pred"])
        zero_res.append(residual(torch.zeros_like(group["f_pred"]), group))
        const_res.append(residual(const, group))
        net_res.append(residual(group["f_pred"], group))
    zero_res = torch.cat(zero_res)
    const_res = torch.cat(const_res)
    net_res = torch.cat(net_res)

    gen = torch.Generator().manual_seed(seed)
    n_unshuffled = sum(
        int(group["f_pred"].shape[0]) for group in by_t.values()
        if group["f_pred"].shape[0] < 2)
    shuffle_means, shuffle_pooled = [], []
    for _ in range(n_shuffles):
        parts = []
        for group in by_t.values():
            n_group = int(group["f_pred"].shape[0])
            if n_group < 2:
                continue                                # unshufflable (counted above)
            offset = int(torch.randint(1, n_group, (1,), generator=gen))
            rolled = torch.roll(group["f_pred"], shifts=offset, dims=0)
            parts.append(residual(rolled, group))
        if parts:
            perm_res = torch.cat(parts)
            shuffle_means.append(float(perm_res.mean()))
            shuffle_pooled.append(perm_res)
    shuffle_means = torch.tensor(shuffle_means) if shuffle_means else torch.tensor([float("nan")])
    shuffle_pooled = (torch.cat(shuffle_pooled) if shuffle_pooled
                      else torch.full((1,), float("nan")))

    # Head-frame per-limb/component std + corr(‖f‖, prob) over all residual frames
    # (flatten each T-group first — groups differ in n_res, only frames concatenate).
    f_all = torch.cat(
        [group["f_pred"].reshape(-1, 12) for group in by_t.values()]).reshape(-1, 4, 3)
    probs_all = torch.cat(
        [group["probs"].reshape(-1, 4) for group in by_t.values()])
    mag = f_all.norm(dim=-1)                                       # (Ntot, 4)
    comp_std = f_all.std(dim=0, unbiased=False)                   # (4, 3)

    def summary(res: torch.Tensor) -> dict:
        return {"mean": float(res.mean()), **_residual_quantiles(res)}

    net_mean = float(net_res.mean())
    beats_constant = net_mean < float(const_res.mean())
    beats_shuffled = net_mean < float(shuffle_means.mean())
    return {
        "n_residual_frames": int(net_res.numel()),
        "zero": summary(zero_res),
        "constant": summary(const_res),
        "network": summary(net_res),
        "shuffled": {"mean": float(shuffle_means.mean()),
                     "std": float(shuffle_means.std(unbiased=False)),
                     **_residual_quantiles(shuffle_pooled)},
        "n_unshuffled_clips": int(n_unshuffled),
        "input_dependent": bool(beats_constant and beats_shuffled),
        "beats_constant": bool(beats_constant),
        "beats_shuffled": bool(beats_shuffled),
        "head_force_component_std": {
            EXTREMITY_OUTPUT_NAMES[e]: comp_std[e].tolist() for e in range(4)},
        "force_prob_pearson": {
            EXTREMITY_OUTPUT_NAMES[e]: _pearson(mag[:, e], probs_all[:, e])
            for e in range(4)},
    }


@torch.no_grad()
def evaluate_physics(model, loader, physics_loss, device, *,
                     threshold: float, contact_min_bw: float) -> dict:
    """Physics-consistency + force-plausibility metrics on the chosen split.

    With no GT forces this is: the headline ``physics_residual`` (the RAW physical
    root-wrench residual, exact mass-weighted mean over residual frames — identical
    to the training monitor), the vertical-force-sum distribution (≈1 body weight
    for quasi-static climbing), the two gate-violation rates, per-extremity force
    magnitudes (body weight + newton) split by predicted contact state, and the
    decisive **affine input-dependence** baselines (§4). The contact metrics are
    computed separately and untouched.
    """
    res_num, res_mass = 0.0, 0.0
    sat_num, jerk_excluded = 0.0, 0
    mags_bw, mags_n, probs_all, vsum_all = [], [], [], []
    by_t: dict[int, dict[str, list]] = {}
    for batch in loader:
        batch = batch_to_device(batch, device)
        out = forward_model(model, batch)
        _, parts = physics_loss(out, batch)
        rterm = parts.get("raw_residual") or parts["terms"].get("residual")
        if rterm is not None:
            res_num += float(rterm["weighted_numerator_tensor"].detach())
            res_mass += rterm["weight_mass"]
            # Mass-weighted saturation aggregate (per-batch sat_frac is a fraction
            # of that batch's residual components; components/frame cancels).
            sat_num += float(parts.get("residual_sat_frac", 0.0)) * rterm["weight_mass"]
        jerk_excluded += int(parts.get("n_jerk_excluded_clips", 0))
        diag = physics_loss.diagnostics(out, batch)
        if diag is not None:
            mags_bw.append(diag["magnitude_bw"])
            mags_n.append(diag["magnitude_newton"])
            probs_all.append(diag["probs"])
            vsum_all.append(diag["vertical_sum_bw"])
        affine = physics_loss.affine_residual(out, batch)
        if affine is not None:
            group = by_t.setdefault(
                affine["seq_len"], {"r0": [], "basis": [], "f_pred": [], "probs": []})
            for key in ("r0", "basis", "f_pred", "probs"):
                group[key].append(affine[key])

    # Zero residual mass (every clip ineligible/jerk-excluded) must read as "no
    # data" (NaN) — never a perfect 0 residual (the trainer-side monitor raises).
    result = {
        "physics_residual": res_num / res_mass if res_mass > 0 else float("nan"),
        "residual_sat_frac": sat_num / res_mass if res_mass > 0 else float("nan"),
        "n_jerk_excluded_clips": int(jerk_excluded),
        "n_frames": 0,
    }
    if by_t:
        result["affine"] = _affine_baselines(
            {t: {key: torch.cat(vals) for key, vals in lists.items()}
             for t, lists in by_t.items()})
    if not mags_bw:
        return result

    mag_bw = torch.cat(mags_bw)                       # (M, 4)
    mag_n = torch.cat(mags_n)                         # (M, 4)
    contact = torch.cat(probs_all) > threshold        # (M, 4)
    vsum = torch.cat(vsum_all)                        # (M,)

    per_extremity = {}
    for e, name in enumerate(EXTREMITY_OUTPUT_NAMES):
        held = contact[:, e]
        per_extremity[name] = {
            "contact_frac": float(held.float().mean()),
            "contact": {"bw": _stats(mag_bw[held, e]), "newton": _stats(mag_n[held, e])},
            "free": {"bw": _stats(mag_bw[~held, e]), "newton": _stats(mag_n[~held, e])},
        }

    free_mag, contact_mag = mag_bw[~contact], mag_bw[contact]
    vquant = torch.quantile(vsum, torch.tensor([0.1, 0.5, 0.9], dtype=vsum.dtype))
    result.update({
        "n_frames": int(mag_bw.shape[0]),
        "threshold": threshold,
        "vertical_force_sum_bw": {
            "mean": float(vsum.mean()), "p10": float(vquant[0]),
            "p50": float(vquant[1]), "p90": float(vquant[2])},
        "gate": {
            "contact_min_bw": contact_min_bw,
            "noncontact_mean_force_bw": (
                float(free_mag.mean()) if free_mag.numel() else 0.0),
            "contact_below_min_frac": (
                float((contact_mag < contact_min_bw).float().mean())
                if contact_mag.numel() else 0.0)},
        "per_extremity": per_extremity,
    })
    return result


def _print_physics(res: dict) -> None:
    """Print the physics metrics in the evaluator's telegraphic table style."""
    print(f"[physics] residual={res['physics_residual']:.6f}  "
          f"sat_frac={res['residual_sat_frac']:.4f}  "
          f"jerk_excluded_clips={res['n_jerk_excluded_clips']}  "
          f"eligible_frames={res['n_frames']}")
    if res["n_frames"] == 0:
        print("  (no physics-eligible clips in this split — nothing to summarise)")
        return
    v = res["vertical_force_sum_bw"]
    print(f"  vertical force sum (bw, +up): mean={v['mean']:.3f} p10={v['p10']:.3f} "
          f"p50={v['p50']:.3f} p90={v['p90']:.3f}  (quasi-static target ~1.0)")
    g = res["gate"]
    print(f"  gate: mean ||f|| on non-contact frames={g['noncontact_mean_force_bw']:.4f} bw   "
          f"contact frames < {g['contact_min_bw']:.2f} bw = "
          f"{100 * g['contact_below_min_frac']:.1f}%")
    print("  per-extremity ||f||  [contact mean/p50/p90 bw (mean N, n) | free mean bw (n)]:")
    for name, pe in res["per_extremity"].items():
        c, fr = pe["contact"], pe["free"]
        print(f"    {name:>11s}: contact {c['bw']['mean']:.3f}/{c['bw']['p50']:.3f}/"
              f"{c['bw']['p90']:.3f} ({c['newton']['mean']:.0f} N, n={c['bw']['n']})   "
              f"free {fr['bw']['mean']:.3f} (n={fr['bw']['n']})")
    if res.get("affine"):
        _print_affine(res["affine"])


def _print_affine(a: dict) -> None:
    """Print the affine input-dependence baselines (§4) in the telegraphic style."""
    verdict = "PASS" if a["input_dependent"] else "FAIL"
    print(f"  [affine input-dependence] {verdict} (network must beat constant AND "
          f"shuffled; residual frames={a['n_residual_frames']})")

    def row(label: str, s: dict) -> str:
        return (f"    {label:>9s}: mean={s['mean']:.4f}  p50={s['p50']:.4f} "
                f"p90={s['p90']:.4f} p99={s['p99']:.4f} max={s['max']:.3f}")

    print(row("zero", a["zero"]))
    print(row("constant", a["constant"]))
    print(row("network", a["network"]))
    sh = a["shuffled"]
    print(f"    {'shuffled':>9s}: mean={sh['mean']:.4f} +/- {sh['std']:.4f}  "
          f"p50={sh['p50']:.4f} p90={sh['p90']:.4f} p99={sh['p99']:.4f} "
          f"max={sh['max']:.3f}")
    print(f"    network beats: constant={a['beats_constant']} "
          f"shuffled={a['beats_shuffled']}"
          + (f"   ({a['n_unshuffled_clips']} clip(s) unshufflable — size-1 T-groups)"
             if a["n_unshuffled_clips"] else ""))
    print("  head-frame corr(||f||, prob) | across-frame component std [x, y, z]:")
    for name in EXTREMITY_OUTPUT_NAMES:
        cstd = a["head_force_component_std"][name]
        print(f"    {name:>11s}: corr={a['force_prob_pearson'][name]:+.3f}  "
              f"std=[{cstd[0]:.3f}, {cstd[1]:.3f}, {cstd[2]:.3f}]")


def _manual_test_loader(cfg: dict, image_size: tuple[int, int], spec: TargetSpec):
    corpus_entries = [d for d in cfg["data"]["datasets"] if d["name"] == "climbing_corpus"]
    if len(corpus_entries) != 1 or len(cfg["data"]["datasets"]) != 1:
        raise ValueError("--split test requires a climbing_corpus-only data config")
    dataset_cfg = (yaml.safe_load(Path(corpus_entries[0]["config"]).read_text()) or {})["data"]
    sequence = cfg["data"]["sequence"]
    ds = ClimbingCorpusDataset(
        dataset_cfg["root"],
        split="test",
        frames_per_clip=int(sequence["frames_per_clip"]),
        frame_stride=int(sequence["frame_stride"]),
        jitter=False,
        seed=int(cfg["data"]["seed"]),
        contact_level=int(dataset_cfg.get("contact_level", 1)),
        use_confidence_weights=bool(
            cfg["contact"]["targets"]["joint"]["use_confidence_weights"]),
        load_forces=bool(dataset_cfg.get("load_forces", False)),
    )
    clips_per_batch = max(
        1, int(cfg["data"]["frames_per_batch"]) // int(sequence["frames_per_clip"]))
    return DataLoader(
        ds,
        batch_size=clips_per_batch,
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        collate_fn=make_collate(image_size, spec),
        pin_memory=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument(
        "--warm-start", action="store_true",
        help="build an untrained force branch warm-started from the config's "
             "model.init_contact_checkpoint (no force checkpoint yet); exercises "
             "the physics-eval pipeline. Force weights stay zero-init.",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument(
        "--curve-thresholds",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="comma-separated thresholds for the saved precision/recall curve; empty disables",
    )
    ap.add_argument("--out", default=None, help="append one result JSON per line here")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, _ = build_model(cfg, args.device)
    if args.warm_start:
        init_ckpt = cfg["model"].get("init_contact_checkpoint")
        if not init_ckpt:
            ap.error("--warm-start requires model.init_contact_checkpoint in the config")
        state = ckpt_io.initialize_common_contact(init_ckpt, model, config=cfg)
    elif args.checkpoint:
        state = ckpt_io.load(args.checkpoint, model, config=cfg)
    else:
        ap.error("--checkpoint is required unless --warm-start is given")
    model.eval()

    spec = TargetSpec.from_config(cfg)
    if args.split == "test":
        val_loader = _manual_test_loader(cfg, tuple(model.cfg.MODEL.IMAGE_SIZE), spec)
    else:
        # Reproduce the checkpoint's exact grouped validation split when this
        # config uses the same datasets as training. Skipped for --warm-start:
        # the source (contact) run may hold a train/test manifest, and there is no
        # trained val split to reproduce — a fresh seeded grouped split suffices.
        manifest = None
        if not args.warm_start and state.get("split_manifest") is not None:
            trained_datasets = (state.get("config", {}) or {}).get("data", {}).get("datasets")
            if trained_datasets == cfg["data"]["datasets"]:
                manifest = state["split_manifest"]
                # Old checkpoints (trained on the exported ClimbingVideos_v1)
                # store "video:<config>" manifest keys; the corpus branch reads
                # "corpus:<config>". A missing corpus key must not make old
                # checkpoints unevaluable — fall back to fresh derivation, loudly.
                missing = [
                    f"corpus:{d['config']}" for d in cfg["data"]["datasets"]
                    if d["name"] == "climbing_corpus"
                    and f"corpus:{d['config']}" not in manifest
                ]
                if missing:
                    print(
                        f"WARNING: checkpoint split manifest has no {missing} "
                        "entry (pre-corpus checkpoint) — re-deriving the grouped "
                        "split fresh instead of reproducing the exact split the "
                        "checkpoint was trained on")
                    manifest = None
            else:
                print(
                    "WARNING: checkpoint data.datasets differ from this config "
                    f"(trained on {trained_datasets!r}) — re-deriving the grouped "
                    "split fresh instead of reproducing the exact split the "
                    "checkpoint was trained on")
        _, val_loader, _ = make_loaders(
            cfg, tuple(model.cfg.MODEL.IMAGE_SIZE), manifest=manifest)
    targets = [t for t in ("vertex", "joint") if cfg["contact"]["targets"][t]["enabled"]]
    curve_thresholds = tuple(
        float(value) for value in args.curve_thresholds.split(",") if value.strip())
    output_names = {"joint": spec.joint_names} if "joint" in targets else {}

    results = evaluate(
        model,
        val_loader,
        targets,
        args.device,
        threshold=args.threshold,
        curve_thresholds=curve_thresholds,
        output_names=output_names,
        target_frame=str(cfg["data"]["sequence"]["target_frame"]),
    )
    for t, res in results.items():
        print(f"[{t}] P={res['precision']:.4f}  R={res['recall']:.4f}  "
              f"F1={res['f1']:.4f}  F2={res['f2']:.4f}  IoU={res['iou']:.4f}  "
              f"(tp={res['tp']} fp={res['fp']} fn={res['fn']})")
        for name, values in res.get("per_output", {}).items():
            print(f"  {name:>10s}: P={values['precision']:.4f} R={values['recall']:.4f} "
                  f"F1={values['f1']:.4f} F2={values['f2']:.4f}")

    # Physics-consistency metrics (only when the run enables the force branch + the
    # RNEA loss). The contact section above is left byte-for-byte unchanged.
    physics_results = None
    if cfg["physics"]["enabled"] and cfg["model"]["force_head"]["enabled"]:
        from contact.physics import PhysicsLoss
        physics_loss = PhysicsLoss(cfg, device=args.device)
        physics_results = evaluate_physics(
            model, val_loader, physics_loss, args.device,
            threshold=args.threshold,
            contact_min_bw=float(cfg["physics"]["loss"]["contact_min_bw"]))
        _print_physics(physics_results)

    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps({"checkpoint": args.checkpoint,
                                "warm_start": args.warm_start,
                                "config": str(args.config), "split": args.split,
                                "threshold": args.threshold, "results": results,
                                "physics": physics_results}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
