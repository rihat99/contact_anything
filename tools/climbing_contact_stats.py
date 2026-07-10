"""Dataset-level contact statistics for ClimbingImages.

Produces two figures:

1. ``contact_heatmap.png`` — per-vertex contact *frequency* averaged
   over every climber in the dataset, rendered on the SMPL-X T-pose
   (front + back). Bright = vertices that touch the scene often;
   dark = rarely or never.
2. ``contact_count_hist.png`` — histogram of the number of contact
   vertices per climber. One bar per climber-item; the loader's
   ``contact_surface`` field is what gets counted.

We walk the BetterImageReconstruction sidecars directly instead of
going through ``ClimbingImagesDataset.__getitem__`` to avoid the image
decode on every sample.

Run::

    /data3/rikhat.akizhanov/miniconda3/envs/sam3d/bin/python \\
        tools/climbing_contact_stats.py \\
        --output tools/output
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEFAULT_ROOT  = Path("/data3/rikhat.akizhanov/better/data/ClimbingImages")
SMPLX_NPZ     = "/data3/rikhat.akizhanov/better/better_human/models/smplx/SMPLX_NEUTRAL.npz"
NUM_VERTICES  = 10475


# -------------------------------------------------------------------- aggregation

def _shard(sha: str) -> tuple[str, str]:
    return sha[:2], sha[2:4]


def _scan_entries(db_path: Path) -> list[tuple[str, int]]:
    """Return ``(sha, climber_local_idx)`` pairs for every fully
    processed climber-item in the DB."""
    sql = "SELECT sha256, gemma_climber_indices FROM images WHERE human_optim=1"
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute(sql).fetchall()
    finally:
        con.close()
    entries: list[tuple[str, int]] = []
    for sha, idx_json in rows:
        idxs = json.loads(idx_json or "[]")
        for ci in range(len(idxs)):
            entries.append((sha, ci))
    return entries


CONTACT_KEYS = ("contact_surface", "contact_self", "contact_any")


def _aggregate(root: Path, entries: list[tuple[str, int]],
               keys: tuple[str, ...] = CONTACT_KEYS,
               ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Sum the binary contact maps across all climbers, for every key.

    Returns ``(sums_per_vertex {key: [V]}, counts_per_sample {key: [N]})``.
    """
    totals = {k: np.zeros(NUM_VERTICES, dtype=np.float64) for k in keys}
    counts = {k: np.empty(len(entries),   dtype=np.int64)  for k in keys}

    # One npz can contain multiple climbers — cache the last opened
    # file so we don't pay the np.load cost twice per image.
    last_sha = None
    cached: dict[str, np.ndarray] = {}
    for i, (sha, ci) in enumerate(tqdm(entries, desc="aggregate")):
        if sha != last_sha:
            aa, bb = _shard(sha)
            z = np.load(root / "results" / "human_optim" / aa / bb / sha / "human_optim.npz")
            cached = {k: z[k] for k in keys}
            last_sha = sha
        for k in keys:
            c = cached[k][ci].astype(np.float64)
            totals[k] += c
            counts[k][i] = int(c.sum())
    return totals, counts


# -------------------------------------------------------------------- mesh render

def _load_tpose() -> tuple[np.ndarray, np.ndarray]:
    """SMPL-X T-pose remapped for matplotlib (matches view_dataset.py)."""
    npz = np.load(SMPLX_NPZ, allow_pickle=True)
    v = npz["v_template"].astype(np.float32).copy()
    f = npz["f"].astype(np.int32)
    v[:, [1, 2]] *= -1
    v = np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1)
    return v, f


def _face_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-8)
    return n


def _render_heatmap_view(
    ax, verts: np.ndarray, faces: np.ndarray,
    vert_freq: np.ndarray, vmax: float, cmap,
    title: str, elev: float, azim: float,
) -> None:
    """Render one view with per-face heatmap shading + back-face culling."""
    az, el = np.radians(azim), np.radians(elev)
    cam_dir = np.array([np.cos(el) * np.cos(az),
                        np.cos(el) * np.sin(az),
                        np.sin(el)])
    normals = _face_normals(verts, faces)
    vis = (normals @ cam_dir) > 0
    f_vis, n_vis = faces[vis], normals[vis]

    key  = np.array([0.5, -1.0, 0.8]); key  /= np.linalg.norm(key)
    fill = np.array([-0.4, 1.0, 0.3]); fill /= np.linalg.norm(fill)
    shading = np.clip(0.45
                      + np.clip(n_vis @ key, 0, 1) * 0.7
                      + np.clip(n_vis @ fill, 0, 1) * 0.5, 0, 1)

    # Face frequency = mean of its three vertex frequencies. Normalise
    # by the global vmax so the colour scale is comparable front <> back.
    face_val = vert_freq[f_vis].mean(axis=1)
    face_val = np.clip(face_val / max(vmax, 1e-12), 0, 1)
    rgba = cmap(face_val)                          # (Nf, 4)
    rgba[:, :3] = np.clip(rgba[:, :3] * shading[:, None], 0, 1)

    coll = Poly3DCollection(verts[f_vis], zsort="average",
                            facecolor=rgba, edgecolor="none")
    ax.add_collection3d(coll)

    xlo, xhi = verts[:, 0].min(), verts[:, 0].max()
    zlo, zhi = verts[:, 2].min(), verts[:, 2].max()
    span = max(xhi - xlo, zhi - zlo) * 0.38
    ax.set_xlim((xhi + xlo) / 2 - span, (xhi + xlo) / 2 + span)
    ax.set_zlim((zhi + zlo) / 2 - span, (zhi + zlo) / 2 + span)
    ylo, yhi = verts[:, 1].min(), verts[:, 1].max()
    ax.set_ylim(ylo - 0.05 * (yhi - ylo), yhi + 0.05 * (yhi - ylo))
    ax.set_box_aspect([1, max((yhi - ylo) / (2 * span), 0.05), 1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()


def render_heatmap(verts: np.ndarray, faces: np.ndarray,
                   vert_freq: np.ndarray, out_path: Path,
                   n_samples: int, cmap_name: str = "hot",
                   label: str = "contact_surface") -> None:
    cmap = cm.get_cmap(cmap_name)
    vmax = float(vert_freq.max())

    fig = plt.figure(figsize=(10, 6), dpi=120)
    gs = fig.add_gridspec(1, 2, wspace=0.0, left=0.02, right=0.92, top=0.92, bottom=0.05)
    ax_f = fig.add_subplot(gs[0], projection="3d")
    ax_b = fig.add_subplot(gs[1], projection="3d")
    _render_heatmap_view(ax_f, verts, faces, vert_freq, vmax, cmap,
                         "Front", elev=25,  azim=-90)
    _render_heatmap_view(ax_b, verts, faces, vert_freq, vmax, cmap,
                         "Back",  elev=-25, azim= 90)

    fig.suptitle(
        f"Per-vertex {label} frequency on SMPL-X — averaged over {n_samples:,} climbers"
        f"\n(brightest vertex touches in {100 * vmax:.1f}% of climbers)",
        fontsize=12,
    )

    # Colorbar showing the actual contact fraction the brightest point
    # represents (denormalised back from the [0, 1] face value).
    sm = cm.ScalarMappable(
        norm=plt.Normalize(vmin=0.0, vmax=100 * vmax), cmap=cmap,
    )
    sm.set_array([])
    cax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_label(f"% of climbers with this vertex in {label}", fontsize=10)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def render_heatmap_compare(verts: np.ndarray, faces: np.ndarray,
                           freqs: dict[str, np.ndarray], out_path: Path,
                           n_samples: int, cmap_name: str = "hot") -> None:
    """3-column figure: contact_surface vs contact_self vs contact_any.

    Each column shows the front view only (keeps the row tight). All
    three colour scales are kept independent so the visual contrast is
    not dominated by ``contact_any``, which is the union of the other
    two and therefore always the brightest.
    """
    keys = list(freqs.keys())
    cmap = cm.get_cmap(cmap_name)

    fig = plt.figure(figsize=(12, 5.5), dpi=120)
    gs = fig.add_gridspec(
        1, len(keys), wspace=0.02,
        left=0.02, right=0.98, top=0.86, bottom=0.10,
    )
    for col, k in enumerate(keys):
        freq = freqs[k]
        vmax = float(freq.max())
        ax = fig.add_subplot(gs[0, col], projection="3d")
        _render_heatmap_view(
            ax, verts, faces, freq, vmax, cmap,
            f"{k}   (max {100 * vmax:.1f}%)",
            elev=25, azim=-90,
        )

    fig.suptitle(
        "SMPL-X per-vertex contact frequency — three channels, front view"
        f"\n(averaged over {n_samples:,} climbers; colour scale is independent per column)",
        fontsize=12,
    )
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# -------------------------------------------------------------------- histogram

def render_histogram(counts: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)

    # Long tail — use log y-axis so the bulk and the tail are both readable.
    bins = np.linspace(0, counts.max(), 60) if counts.max() > 0 else np.array([0, 1])
    ax.hist(counts, bins=bins, color="#2563eb", edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("contact vertices per climber (SMPL-X, 10475 verts)")
    ax.set_ylabel("number of climbers  (log scale)")
    ax.set_title("ClimbingImages — distribution of contact-vertex counts")

    mean  = counts.mean()
    median = np.median(counts)
    p95   = np.percentile(counts, 95)
    zero  = int((counts == 0).sum())
    note  = (f"n = {len(counts):,}   mean = {mean:.1f}   "
             f"median = {median:.0f}   p95 = {p95:.0f}   "
             f"zero-contact = {zero:,} ({100 * zero / len(counts):.1f}%)")
    ax.text(0.98, 0.95, note, transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="#d1d5db", boxstyle="round,pad=0.3"))

    ax.grid(True, axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# -------------------------------------------------------------------- entrypoint

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--db",   type=Path, default=None)
    p.add_argument("--key",  choices=CONTACT_KEYS, default="contact_surface",
                   help="which contact channel drives the primary heatmap and histogram")
    p.add_argument("--cmap", default="hot",
                   help="matplotlib colormap for the heatmap mesh")
    p.add_argument("--output", type=Path, default=REPO / "tools" / "output")
    args = p.parse_args()

    db = args.db or args.root / "image_reconstruction.db"
    args.output.mkdir(parents=True, exist_ok=True)

    entries = _scan_entries(db)
    print(f"climbers to aggregate: {len(entries):,} "
          f"(from {len(set(s for s, _ in entries)):,} images)")

    totals, counts = _aggregate(args.root, entries)
    n = max(len(entries), 1)
    freqs = {k: totals[k] / n for k in CONTACT_KEYS}        # each [V] in [0, 1]
    for k in CONTACT_KEYS:
        f, c = freqs[k], counts[k]
        print(f"[{k}] freq: max={f.max():.4f}  mean={f.mean():.4f}  "
              f"nonzero_verts={int((f > 0).sum()):,}/{NUM_VERTICES:,}  "
              f"| count: min={c.min()} max={c.max()} "
              f"mean={c.mean():.1f} median={np.median(c):.0f}")

    verts, faces = _load_tpose()
    heatmap_path = args.output / "contact_heatmap.png"
    compare_path = args.output / "contact_heatmap_compare.png"
    hist_path    = args.output / "contact_count_hist.png"

    render_heatmap(verts, faces, freqs[args.key], heatmap_path,
                   n_samples=len(entries), cmap_name=args.cmap, label=args.key)
    render_heatmap_compare(verts, faces, freqs, compare_path,
                           n_samples=len(entries), cmap_name=args.cmap)
    render_histogram(counts[args.key], hist_path)
    print(f"wrote {heatmap_path}")
    print(f"wrote {compare_path}")
    print(f"wrote {hist_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
