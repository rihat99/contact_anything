"""Browser viewer for the contact datasets.

Shows each sample as an image on the left and a SMPL T-pose mesh
(contacts highlighted) on the right. Selector chips at the top filter
which datasets to walk; Next / Random buttons at the bottom navigate.

Run::

    /data3/rikhat.akizhanov/miniconda3/envs/sam3d/bin/python tools/view_dataset.py --port 8765
"""
from __future__ import annotations

import argparse
import io
import random
import sys
import threading
import webbrowser
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import DamonDataset, LemonDataset, RichDataset  # noqa: E402


SMPL_NPZ = "/data3/rikhat.akizhanov/human_global_motion/better_human/models/smpl/SMPL_NEUTRAL.npz"
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

COLOR_CONTACT     = np.array([0.95, 0.15, 0.15])
COLOR_NO_CONTACT  = np.array([0.55, 0.65, 0.80])
MASK_OUTLINE_BGR  = (30, 230, 30)   # bright green outline on top of RGB→BGR image
MASK_OUTLINE_PX   = 3


# -------------------------------------------------------------------- mask overlay

def _overlay_mask_outline(bgr: np.ndarray, mask: np.ndarray,
                          color=MASK_OUTLINE_BGR,
                          thickness: int = MASK_OUTLINE_PX) -> np.ndarray:
    """Draw a colored outline of ``mask`` on ``bgr`` (in-place safe)."""
    m = mask
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape[:2] != bgr.shape[:2]:
        m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
    binm = (m > 127).astype(np.uint8)
    contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = bgr.copy()
    cv2.drawContours(out, contours, -1, color, thickness, lineType=cv2.LINE_AA)
    return out


# -------------------------------------------------------------------- mesh render

def _face_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-8)
    return n


def _render_view(ax, verts: np.ndarray, faces: np.ndarray,
                 contact_mask: np.ndarray, title: str,
                 elev: float, azim: float) -> None:
    """One T-pose view with simple two-light shading and back-face culling."""
    az, el = np.radians(azim), np.radians(elev)
    cam_dir = np.array([np.cos(el) * np.cos(az),
                        np.cos(el) * np.sin(az),
                        np.sin(el)])
    normals = _face_normals(verts, faces)
    vis = (normals @ cam_dir) > 0
    f_vis, n_vis = faces[vis], normals[vis]

    key  = np.array([0.5, -1.0, 0.8]); key  /= np.linalg.norm(key)
    fill = np.array([-0.4, 1.0, 0.3]); fill /= np.linalg.norm(fill)
    shading = np.clip(0.38
                      + np.clip(n_vis @ key, 0, 1)
                      + np.clip(n_vis @ fill, 0, 1) * 0.7, 0, 1)
    face_hit = contact_mask[f_vis].any(axis=1)
    base = np.where(face_hit[:, None], COLOR_CONTACT[None], COLOR_NO_CONTACT[None])
    rgba = np.concatenate([np.clip(base * shading[:, None], 0, 1),
                           np.ones((len(f_vis), 1))], axis=1)

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


def render_tpose_png(contact_mask: np.ndarray) -> bytes:
    verts = _T_VERTS
    faces = _T_FACES
    contact_mask = contact_mask.astype(bool)
    assert contact_mask.shape == (verts.shape[0],), contact_mask.shape

    fig = plt.figure(figsize=(8, 6), dpi=110)
    gs = fig.add_gridspec(1, 2, wspace=0.0, left=0, right=1, top=1, bottom=0)
    ax_front = fig.add_subplot(gs[0], projection="3d")
    ax_back  = fig.add_subplot(gs[1], projection="3d")
    _render_view(ax_front, verts, faces, contact_mask, "Front",
                 elev=25, azim=-90)
    _render_view(ax_back,  verts, faces, contact_mask, "Back",
                 elev=-25, azim=90)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return buf.getvalue()


# -------------------------------------------------------------------- HTML

PAGE = """<!doctype html>
<html><head><meta charset='utf-8'>
<title>Contact dataset viewer</title>
<style>
  :root { color-scheme: light dark; font-family: system-ui, sans-serif; }
  body { margin: 0; padding: 16px; background: #f3f4f6; color: #111; }
  header { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
  header h1 { font-size: 16px; margin: 0 12px 0 0; }
  .chip { padding: 6px 12px; border-radius: 999px; background: #e5e7eb;
          cursor: pointer; user-select: none; font-size: 13px; }
  .chip.on { background: #2563eb; color: white; }
  main { display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
         background: white; border-radius: 8px; padding: 12px; min-height: 480px; }
  main img { width: 100%; max-height: 78vh; object-fit: contain; background: #f9fafb; }
  footer { margin-top: 12px; display: flex; gap: 8px; align-items: center; }
  button { padding: 8px 14px; border-radius: 6px; border: 1px solid #d1d5db;
           background: white; color: #111; cursor: pointer; font-size: 14px; }
  button:hover { background: #f3f4f6; }
  #info { font-family: ui-monospace, monospace; font-size: 12px; color: #4b5563;
          margin-left: auto; }
</style></head>
<body>
<header>
  <h1>Contact datasets</h1>
  <span class='chip on' data-name='damon'>damon</span>
  <span class='chip on' data-name='lemon'>lemon</span>
  <span class='chip on' data-name='rich'>rich</span>
</header>
<main>
  <img id='img' alt='input image'>
  <img id='mesh' alt='SMPL T-pose with contacts'>
</main>
<footer>
  <button id='next'>Next</button>
  <button id='random'>Random</button>
  <div id='info'>—</div>
</footer>
<script>
let cursor = null;
const chips = document.querySelectorAll('.chip');
chips.forEach(c => c.addEventListener('click', () => { c.classList.toggle('on'); }));

function selected() {
  return Array.from(chips).filter(c => c.classList.contains('on'))
              .map(c => c.dataset.name).join(',');
}

async function load(mode) {
  const params = new URLSearchParams({ mode, datasets: selected() });
  if (cursor) { params.set('ds', cursor.ds); params.set('idx', cursor.idx); }
  const r = await fetch('/sample?' + params);
  if (!r.ok) { document.getElementById('info').textContent = 'no samples'; return; }
  const j = await r.json();
  cursor = { ds: j.dataset, idx: j.local_idx };
  document.getElementById('img').src  = `/image/${j.dataset}/${j.local_idx}?t=${Date.now()}`;
  document.getElementById('mesh').src = `/mesh/${j.dataset}/${j.local_idx}?t=${Date.now()}`;
  document.getElementById('info').textContent =
    `${j.dataset} #${j.local_idx} — ${j.key} — contacts: ${j.contact_count}`;
}

document.getElementById('next').addEventListener('click',   () => load('next'));
document.getElementById('random').addEventListener('click', () => load('random'));
load('random');
</script>
</body></html>
"""


# -------------------------------------------------------------------- app

app = FastAPI(title="Contact dataset viewer", docs_url=None, redoc_url=None)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(PAGE)


def _filtered(names: str) -> list[str]:
    sel = [n for n in (names or "").split(",") if n in DATASETS]
    return sel or list(DATASETS)


@app.get("/sample")
def sample(
    mode: str = Query("random"),
    datasets: str = Query(""),
    ds: str | None = Query(None),
    idx: int | None = Query(None),
) -> dict:
    selected = _filtered(datasets)

    if mode == "random":
        ds_name = random.choice(selected)
        local_idx = random.randrange(len(DATASETS[ds_name]))
    elif mode == "next":
        if ds is None or idx is None or ds not in selected:
            ds_name = selected[0]
            local_idx = 0
        else:
            ds_name = ds
            local_idx = idx + 1
            if local_idx >= len(DATASETS[ds_name]):
                # wrap to next selected dataset
                order = selected
                i = (order.index(ds_name) + 1) % len(order)
                ds_name = order[i]
                local_idx = 0
    else:
        raise HTTPException(400, f"bad mode {mode!r}")

    item = DATASETS[ds_name][local_idx]
    return {
        "dataset":       ds_name,
        "local_idx":     local_idx,
        "key":           item["key"],
        "contact_count": int(item["contact"].sum().item()),
    }


@app.get("/image/{ds_name}/{local_idx}")
def image(ds_name: str, local_idx: int) -> Response:
    if ds_name not in DATASETS:
        raise HTTPException(404, f"unknown dataset {ds_name!r}")
    item = DATASETS[ds_name][local_idx]
    img = item["image"]
    if img is None:
        # placeholder grey image when split has no img.tsv extracted (RICH)
        img = np.full((400, 400, 3), 200, dtype=np.uint8)
        cv2.putText(img, "no image", (110, 210), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (60, 60, 60), 2)
        bgr = img
    else:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        mask = item.get("mask")
        if mask is not None:
            bgr = _overlay_mask_outline(bgr, mask)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(500, "jpeg encode failed")
    return Response(buf.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=60"})


@app.get("/mesh/{ds_name}/{local_idx}")
def mesh(ds_name: str, local_idx: int) -> Response:
    if ds_name not in DATASETS:
        raise HTTPException(404, f"unknown dataset {ds_name!r}")
    item = DATASETS[ds_name][local_idx]
    contact = (item["contact"].numpy() > 0.5)
    return Response(render_tpose_png(contact), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=60"})


# -------------------------------------------------------------------- CLI

def _load_smpl_tpose() -> tuple[np.ndarray, np.ndarray]:
    """Return T-pose vertices remapped for matplotlib (X right, Y depth, Z up).

    Two-step transform from SMPL (Y-up) matches inference_demo.py exactly:
      1. ``[:, [1, 2]] *= -1``  → OpenCV-like (Y down, Z back)
      2. axis swap (x, -y, z) → matplotlib (X right, Y depth, Z up)
    """
    npz = np.load(SMPL_NPZ, allow_pickle=True)
    v = npz["v_template"].astype(np.float32).copy()
    f = npz["f"].astype(np.int32)
    v[:, [1, 2]] *= -1
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1), f


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--damon-split", default="trainval", choices=("trainval", "test"))
    p.add_argument("--lemon-split", default="train", choices=("train", "val"))
    p.add_argument("--rich-split",  default="val",   choices=("train", "val", "test"))
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args()

    global _T_VERTS, _T_FACES, DATASETS
    _T_VERTS, _T_FACES = _load_smpl_tpose()

    print("Loading datasets...")
    damon_cfg = CONFIG_DIR / "damon.yaml"
    if damon_cfg.is_file():
        damon = DamonDataset.from_config(damon_cfg, split=args.damon_split)
    else:
        damon = DamonDataset(split=args.damon_split)
    DATASETS = {
        "damon": damon,
        "lemon": LemonDataset(split=args.lemon_split),
        "rich":  RichDataset(split=args.rich_split),
    }
    for n, ds in DATASETS.items():
        print(f"  {n}: {len(ds)} samples")

    url = f"http://{args.host}:{args.port}/"
    print(f"Serving at {url}")
    if not args.no_open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
