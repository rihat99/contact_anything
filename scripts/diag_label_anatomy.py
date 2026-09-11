"""Where do the contact models fail? (round 6 label anatomy, docs/round6_2026-09-11.md)

Capped-protocol test rows (threshold 0.5) split by manual-vs-automatic label agreement, by the
automatic label's confidence and by GT limb stillness (drift of the group joint over 0.35 s
< 8 cm), each run's accuracy per cell; plus the train-label confidence mass over 150 scenes.

    python scripts/diag_label_anatomy.py final=output_2/<run> notoken=output_2/<run> ...

Every run needs a ``predictions/`` dump (``scripts/predict_test.py``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_ci import load_labels, protocol_rows, THRESHOLD, CONTACT_LEVEL   # noqa: E402
from data.climbing_videos import scene as scene_io
from data.climbing_videos.scene import GROUP_BODY22, GROUP_NAMES, list_train_scenes

ROOT = Path("/data3/rikhat.akizhanov/better/data/ClimbingVideos")
runs = dict(a.split("=", 1) for a in sys.argv[1:])
scenes = sorted(p.stem for p in (Path(next(iter(runs.values()))) / "predictions").glob("*.npz"))
BINS = [0, 0.2, 0.5, 0.8, 1.01]

rows = []   # per active row: group, manual, auto, auto_conf, still, {run: pred}
for scene in scenes:
    lab = load_labels(ROOT, scene)
    auto = scene_io.load_scene(ROOT, scene, "train", CONTACT_LEVEL)
    oids = lab["object_ids"]
    dumps, covered = {}, None
    for name, run in runs.items():
        raw = np.load(Path(run) / "predictions" / f"{scene}.npz", allow_pickle=True)
        d = {k: scene_io.rows_by_object_id(np.asarray(raw[k]), raw["object_ids"], oids, scene, "dump")
             for k in ("covered", "contact_probs")}
        d["stride"] = int(raw["stride"]); dumps[name] = d
        covered = d["covered"] if covered is None else covered & d["covered"]
    stride = dumps[next(iter(runs))]["stride"]
    fps = lab["fps"]; half = max(1, int(round(0.35 * fps / 2)))
    jw = lab["smplx_joints_world"]            # (P, N, 52, 3)
    for person, idx in protocol_rows(lab, covered, stride, "capped"):
        active = (lab["contact_valid"][person, idx] * lab["contact_conf"][person, idx]) > 0
        man = lab["contact_gt"][person, idx] > 0.5
        aut = auto["contact_gt"][person, idx] > 0.5
        aconf = auto["contact_conf"][person, idx]
        n = jw.shape[1]
        lo, hi = np.clip(idx - half, 0, n - 1), np.clip(idx + half, 0, n - 1)
        for g in range(6):
            j = GROUP_BODY22[g][0]
            drift = np.linalg.norm(jw[person, hi, j] - jw[person, lo, j], axis=-1)
            for k in np.flatnonzero(active[:, g]):
                rows.append((g, man[k, g], aut[k, g], aconf[k, g], drift[k],
                             {nm: d["contact_probs"][person, idx[k], g] > THRESHOLD for nm, d in dumps.items()}))
print(f"{len(scenes)} scenes, {len(rows)} active rows")
g_ = np.array([r[0] for r in rows]); man = np.array([r[1] for r in rows]); aut = np.array([r[2] for r in rows])
conf = np.array([r[3] for r in rows]); drift = np.array([r[4] for r in rows])
pred = {nm: np.array([r[5][nm] for r in rows]) for nm in runs}
still = np.isfinite(drift) & (drift < 0.08)

def f1(p, g):
    tp = (p & g).sum(); fp = (p & ~g).sum(); fn = (~p & g).sum()
    return 2 * tp / max(1, 2 * tp + fp + fn)
def acc(p, g): return (p == g).mean() if len(g) else float("nan")

print("\n== automatic label vs manual (capped test rows): agreement / F1 / auto-conf mean")
for g in range(6):
    m = g_ == g
    print(f"  {GROUP_NAMES[g]:12s} n {m.sum():6d}  agree {acc(aut[m], man[m]):.3f}  F1(auto) {f1(aut[m], man[m]):.3f}  "
          f"conf {conf[m].mean():.2f}  auto1/man0 {(aut[m] & ~man[m]).sum():5d}  auto0/man1 {(~aut[m] & man[m]).sum():5d}")
print(f"  {'ALL':12s} n {len(g_):6d}  agree {acc(aut, man):.3f}  F1(auto) {f1(aut, man):.3f}")

agree = aut == man
print("\n== runs on agree / disagree rows (micro): F1 vs manual | on disagree: frac siding with MANUAL")
for nm, p in pred.items():
    print(f"  {nm:10s} all {f1(p, man):.3f}  agree {f1(p[agree], man[agree]):.3f} (n {agree.sum()})  "
          f"disagree F1 {f1(p[~agree], man[~agree]):.3f} acc {acc(p[~agree], man[~agree]):.3f} (n {(~agree).sum()})  "
          f"| auto1/man0 sides manual {acc(p[aut & ~man], man[aut & ~man]):.3f}  auto0/man1 sides manual {acc(p[~aut & man], man[~aut & man]):.3f}")

print("\n== runs by the AUTOMATIC label's confidence on the test row (accuracy vs manual, and auto's own accuracy)")
for lo, hi in zip(BINS[:-1], BINS[1:]):
    m = (conf >= lo) & (conf < hi)
    s = f"  conf [{lo:.1f},{min(hi,1):.1f}) n {m.sum():6d} ({m.mean()*100:4.1f}%)  auto {acc(aut[m], man[m]):.3f}"
    for nm, p in pred.items():
        s += f"  {nm} {acc(p[m], man[m]):.3f}"
    print(s)

print("\n== by GT stillness (drift of the group joint over 0.35 s < 8 cm) x manual label: accuracy")
for name, m in (("still & contact", still & man), ("still & free", still & ~man),
                ("moving & contact", ~still & man), ("moving & free", ~still & ~man)):
    s = f"  {name:17s} n {m.sum():6d} ({m.mean()*100:4.1f}%)  auto {acc(aut[m], man[m]):.3f}"
    for nm, p in pred.items():
        s += f"  {nm} {acc(p[m], man[m]):.3f}"
    print(s)

print("\n== TRAIN confidence mass (150 scenes): per group, share of rows / of weight by conf bin, split by label")
tr = list_train_scenes(ROOT)[::len(list_train_scenes(ROOT)) // 150][:150]
C, L, V = [], [], []
for scene in tr:
    d = scene_io.load_scene(ROOT, scene, "train", CONTACT_LEVEL)
    v = d["contact_valid"] > 0
    C.append(d["contact_conf"][v.any(-1)]); L.append(d["contact_gt"][v.any(-1)] > 0.5); V.append(v[v.any(-1)])
C = np.concatenate(C); L = np.concatenate(L); V = np.concatenate(V)
for g in range(6):
    c, l, v = C[:, g], L[:, g], V[:, g]
    c = c[v]; l = l[v]
    line = f"  {GROUP_NAMES[g]:12s} n {len(c):7d} pos {l.mean():.2f}  mean conf pos {c[l].mean():.2f} neg {c[~l].mean():.2f}  weight share of rows w/ conf<0.2: {(c < 0.2).mean():.2f} (their weight {c[c<0.2].sum()/max(1e-9,c.sum()):.3f})"
    line += "  bins(rows%):" + " ".join(f"{((c>=lo)&(c<hi)).mean()*100:4.1f}" for lo, hi in zip(BINS[:-1], BINS[1:]))
    line += "  neg rows conf<0.2: %.2f  pos rows conf<0.2: %.2f" % ((c[~l] < 0.2).mean(), (c[l] < 0.2).mean())
    print(line)
