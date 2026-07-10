"""Inference demo for the per-vertex Contact Head.

Adapted to the slim training pipeline (``train/model.py`` + ``train/data.py``):
the contact head predicts directly on the 6890 SMPL vertices, so GT and
predicted contacts share topology with no MHR<->SMPL conversion.

For each sampled val image we save one figure:

  Row 0 (2 panels): input image (+bbox)  |  input image with the model's own
                    predicted body mesh (MHR) overlaid. This is the pose the
                    model infers for this crop+camera — shown plainly, no
                    contact colouring.
  Row 1 (4 panels): GT contacts on the canonical SMPL T-pose (front+back)  |
                    predicted contacts on the T-pose (front+back).

The predicted contacts come straight from ``contact_logits``; the predicted
mesh comes straight from the MHR pose head — nothing here re-poses the GT fit.

Usage::

    CUDA_VISIBLE_DEVICES=0 python scripts/demo.py \
        --checkpoint output/contact_climbing_20260529_151004/best.pth \
        --num_samples 12 --split val
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from contact import checkpoint as ckpt_io
from contact.config import load_config
from contact.data.collate import batch_to_device, make_collate
from contact.engine import forward_model
from contact.data.climbing_images import ClimbingImagesDataset
from contact.data.splits import train_val_indices
from contact.metrics import prf1
from contact.model import build_model
from contact.targets import TargetSpec

SMPL_NPZ = "/data3/rikhat.akizhanov/better/better_human/models/smpl/SMPL_NEUTRAL.npz"

COLOR_CONTACT    = np.array([0.95, 0.15, 0.15])   # red — in contact
COLOR_NO_CONTACT = np.array([0.55, 0.65, 0.80])   # steel-blue — no contact
COLOR_BODY       = np.array([0.62, 0.71, 0.82])   # neutral mesh (predicted pose)
COLOR_BBOX       = (1.0, 0.9, 0.0)                 # yellow bbox


# ---------------------------------------------------------------------------
# Mesh helpers (topology-agnostic: any verts + faces [+ per-vertex contact])
# ---------------------------------------------------------------------------

def _compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-8)
    return n


def _face_colors(contact_mask, faces, normals):
    """Per-face RGBA, two-light Lambertian shading. X=l-r, Y=depth, Z=up."""
    key  = np.array([0.5, -1.0, 0.8]); key  /= np.linalg.norm(key)
    fill = np.array([-0.4, 1.0, 0.3]); fill /= np.linalg.norm(fill)
    shading = np.clip(0.38 + np.clip(normals @ key, 0, 1)
                      + np.clip(normals @ fill, 0, 1) * 0.70, 0, 1)
    face_hit = contact_mask[faces].any(axis=1)
    base = np.where(face_hit[:, None], COLOR_CONTACT[None], COLOR_NO_CONTACT[None])
    lit = np.clip(base * shading[:, None], 0, 1)
    return np.concatenate([lit, np.ones((len(faces), 1))], axis=1)


def render_mesh_3d(ax, vertices, faces, contact_mask, title="",
                   elev=0.0, azim=-90.0):
    """Lit 3-D T-pose with back-face culling. Verts: X=l-r, Y=depth, Z=up."""
    az, el = np.radians(azim), np.radians(elev)
    cam_dir = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    normals = _compute_face_normals(vertices, faces)
    visible = (normals @ cam_dir) > 0
    vis_faces, vis_normals = faces[visible], normals[visible]

    coll = Poly3DCollection(vertices[vis_faces], zsort="average")
    coll.set_facecolor(_face_colors(contact_mask, vis_faces, vis_normals))
    coll.set_edgecolor("none")
    ax.add_collection3d(coll)

    xlo, xhi = vertices[:, 0].min(), vertices[:, 0].max()
    zlo, zhi = vertices[:, 2].min(), vertices[:, 2].max()
    span = max(xhi - xlo, zhi - zlo) * 0.38
    xmid, zmid = (xhi + xlo) / 2, (zhi + zlo) / 2
    ax.set_xlim(xmid - span, xmid + span)
    ax.set_zlim(zmid - span, zmid + span)
    ylo, yhi = vertices[:, 1].min(), vertices[:, 1].max()
    ax.set_ylim(ylo - 0.05 * (yhi - ylo), yhi + 0.05 * (yhi - ylo))
    ax.set_box_aspect([1, max((yhi - ylo) / (2 * span), 0.05), 1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12)


def overlay_mesh_on_image_2d(ax, image, verts_2d, verts_3d, faces,
                              contact_mask=None, title=""):
    """Solid projected mesh on *image*, painter's algorithm + simple shading.

    verts_2d [V,2] pixel, verts_3d [V,3] camera space (depth+shading), faces.
    ``contact_mask`` None -> neutral body colour; else red/blue by contact.
    """
    ax.imshow(image)
    order = np.argsort(-verts_3d[faces, 2].mean(axis=1))   # far first
    sf = faces[order]
    normals = _compute_face_normals(verts_3d, sf)
    shading = 0.35 + 0.65 * np.clip(-normals[:, 2], 0, 1)  # camera looks +Z
    if contact_mask is None:
        base = np.tile(COLOR_BODY, (len(sf), 1))
    else:
        face_hit = contact_mask[sf].any(axis=1)
        base = np.where(face_hit[:, None], COLOR_CONTACT[None], COLOR_NO_CONTACT[None])
    rgba = np.concatenate([np.clip(base * shading[:, None], 0, 1),
                           np.ones((len(sf), 1))], axis=1)
    ax.add_collection(PolyCollection(verts_2d[sf], facecolors=rgba, edgecolors="none"))
    ax.set_xlim(0, image.shape[1]); ax.set_ylim(image.shape[0], 0)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12)


def make_figure(image, bbox, pred_verts_2d, pred_verts_cam, mhr_faces,
                tpose_verts, smpl_faces, gt_mask, pred_mask, title) -> plt.Figure:
    """Row 0: image+bbox | image+predicted MHR mesh.
    Row 1: GT contacts on T-pose (front+back) | predicted contacts (front+back)."""
    fig = plt.figure(figsize=(16, 12), dpi=200)
    fig.suptitle(title, fontsize=15, y=0.99)
    sfig_top, sfig_bot = fig.subfigures(2, 1, height_ratios=[1.15, 1.25], hspace=0.02)

    gs_top = sfig_top.add_gridspec(1, 2, wspace=0.04,
                                   left=0.06, right=0.94, top=0.88, bottom=0.03)
    ax_img = sfig_top.add_subplot(gs_top[0])
    ax_img.imshow(image)
    x1, y1, x2, y2 = bbox.astype(int)
    ax_img.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               linewidth=1.5, edgecolor=COLOR_BBOX, facecolor="none"))
    ax_img.set_title("Input image", fontsize=12); ax_img.set_axis_off()
    overlay_mesh_on_image_2d(sfig_top.add_subplot(gs_top[1]), image, pred_verts_2d,
                             pred_verts_cam, mhr_faces, None, "Predicted pose (MHR)")

    # T-pose verts: SMPL (Y-up, already [:, [1,2]]*=-1) -> mpl (X, depth, Z up)
    tv = np.stack([tpose_verts[:, 0], tpose_verts[:, 2], -tpose_verts[:, 1]], axis=1)
    sfig_gt, sfig_pr = sfig_bot.subfigures(1, 2, wspace=0.10)
    sfig_bot.add_artist(Line2D([0.5, 0.5], [0.03, 0.97],
                               transform=sfig_bot.transSubfigure,
                               color="#888888", linewidth=1.5, linestyle="--"))
    sfig_gt.suptitle("Ground-truth contact (SMPL)", fontsize=14, fontweight="bold", y=0.97)
    sfig_pr.suptitle("Predicted contact (SMPL)", fontsize=14, fontweight="bold", y=0.97)
    for sfig, mask in ((sfig_gt, gt_mask), (sfig_pr, pred_mask)):
        gs = sfig.add_gridspec(1, 2, wspace=0.01, left=0.01, right=0.99,
                               top=0.92, bottom=0.0)
        render_mesh_3d(sfig.add_subplot(gs[0], projection="3d"), tv, smpl_faces, mask,
                       "Front", elev=25, azim=-90)
        render_mesh_3d(sfig.add_subplot(gs[1], projection="3d"), tv, smpl_faces, mask,
                       "Back", elev=-25, azim=90)
    return fig


# ---------------------------------------------------------------------------
# SMPL T-pose template + data
# ---------------------------------------------------------------------------

def load_smpl_template():
    """Return ``(tpose_verts, faces)`` for the canonical SMPL body.

    Only the static template is needed (no posing / no ``smplx``). Verts are
    pre-flipped (``[:, [1,2]] *= -1``) so ``make_figure`` can do its axis swap.
    """
    npz = np.load(SMPL_NPZ, allow_pickle=True)
    v = npz["v_template"].astype(np.float32).copy()
    f = npz["f"].astype(np.int32)
    v[:, [1, 2]] *= -1
    return v, f


def _mhr_overlay_arrays(mhr: dict):
    """Pull the predicted MHR mesh into (verts_2d, verts_cam, faces) numpy."""
    verts_2d  = mhr["pred_keypoints_2d_verts"][0].cpu().numpy()          # full-img px
    verts_cam = (mhr["pred_vertices"][0] + mhr["pred_cam_t"][0]).cpu().numpy()
    faces = mhr["faces"]
    faces = faces.cpu().numpy() if torch.is_tensor(faces) else np.asarray(faces)
    return verts_2d, verts_cam, faces.astype(np.int64)


def split_indices(cfg, n_total, which):
    """Reproduce ``make_loaders``' seed-42 split for the single climbing set."""
    dcfg = cfg["data"]
    train, val = train_val_indices(n_total, dcfg.get("val_ratio", 0.15), dcfg.get("seed", 42))
    return val if which == "val" else train


def _metrics(pred, gt):
    return prf1({
        "tp": int((pred & gt).sum()), "fp": int((pred & ~gt).sum()),
        "fn": int((~pred & gt).sum()), "tn": int((~pred & ~gt).sum()),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True, help="trained contact .pth")
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "climbing_baseline.yaml",
                    help="config to build the (shared) model architecture from")
    ap.add_argument("--num_samples", type=int, default=12)
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--output_dir", default=None,
                    help="default: <checkpoint dir>/inference_<split>")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0, help="sample-selection seed")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output_dir) if args.output_dir else \
        Path(args.checkpoint).resolve().parent / f"inference_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    print("Building model …")
    model, _ = build_model(cfg, args.device)
    ckpt_io.load(args.checkpoint, model)
    model.eval()
    collate = make_collate(tuple(model.cfg.MODEL.IMAGE_SIZE), TargetSpec.from_config(cfg))
    tpose_verts, smpl_faces = load_smpl_template()

    ds = ClimbingImagesDataset()
    pool = split_indices(cfg, len(ds), args.split)
    picks = random.sample(pool, min(args.num_samples, len(pool)))
    print(f"climbing {args.split} split: {len(pool)} samples; rendering {len(picks)}")

    scores = []
    for run_i, ds_idx in enumerate(picks):
        sample = ds[ds_idx]
        batch = batch_to_device(collate([sample]), args.device)
        with torch.no_grad():
            out = forward_model(model, batch)
        logits = out["contact"]["vertex_logits"][0]               # [6890]
        pred_mask = (torch.sigmoid(logits) > args.threshold).cpu().numpy().astype(bool)
        gt_mask = (sample["contact"].numpy() > 0.5).astype(bool)
        pred_v2, pred_vcam, mhr_faces = _mhr_overlay_arrays(out["mhr"])

        m = _metrics(pred_mask, gt_mask)
        scores.append(m)
        title = (f"Climbing {args.split} #{ds_idx}   ·   "
                 f"IoU={m['iou']:.3f}   F1={m['f1']:.3f}   "
                 f"P={m['precision']:.3f}   R={m['recall']:.3f}   ·   "
                 f"GT={int(gt_mask.sum())}   Pred={int(pred_mask.sum())} contacts")
        fig = make_figure(sample["image"], sample["bbox"], pred_v2, pred_vcam,
                          mhr_faces, tpose_verts, smpl_faces, gt_mask, pred_mask, title)
        save = out_dir / f"sample_{run_i:02d}_idx{ds_idx}_iou{m['iou']:.3f}.png"
        fig.savefig(save, bbox_inches="tight"); plt.close(fig)
        print(f"[{run_i + 1}/{len(picks)}] idx {ds_idx}  IoU={m['iou']:.3f}  "
              f"F1={m['f1']:.3f}  -> {save.name}")

    f1 = np.mean([s["f1"] for s in scores]); iou = np.mean([s["iou"] for s in scores])
    print(f"\nDone. {len(scores)} samples · mean F1={f1:.3f} mean IoU={iou:.3f}")
    print(f"Figures: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
