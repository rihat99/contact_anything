"""
Inference demo for per-vertex Contact Head on DAMON dataset.

For each sampled image, generates a figure with:
  Row 1 — Image + 2D projected mesh:
      Left:  plain image with bbox
      Mid:   ground-truth contact vertices highlighted in red
      Right: predicted contact vertices highlighted in red
  Row 2 — T-pose 3D mesh (front + back view, same shape/scale as prediction):
      GT contact coloring: front and back
      Pred contact coloring: front and back

Usage:
    CUDA_VISIBLE_DEVICES=3 python train/inference_demo.py \\
        --config train/config.yaml \\
        --checkpoint train/output/contact_vert_20260222_123456/best_model.pth \\
        --num_samples 20 \\
        --split val
"""

import os
import sys
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "dataset"))
sys.path.insert(0, str(Path(__file__).parent.parent / "mhr_smpl_conversion"))
os.environ["MOMENTUM_ENABLED"] = "1"

from sam_3d_body.build_models import load_sam_3d_body
from sam_3d_body.models.heads.contact_head import ContactHead
from sam_3d_body.models.decoders.interaction_decoder import InteractionDecoder
from sam_3d_body.utils.config import get_config
from damon_mhr import DamonMHRDataset
from damon_smpl import DamonSMPLDataset
from dataset_utils import prepare_damon_batch
from train_contact import damon_collate
from body_converter import BodyConverter


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLOR_CONTACT   = np.array([0.95, 0.15, 0.15])   # red  — in contact
COLOR_NO_CONTACT = np.array([0.55, 0.65, 0.80])  # steel-blue — no contact
COLOR_BBOX      = (1.0, 0.9, 0.0)                 # yellow bbox


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def _compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unit outward face normals. Returns [F, 3]."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-8)
    return n


def _face_colors(contact_mask: np.ndarray, faces: np.ndarray,
                 normals: np.ndarray) -> np.ndarray:
    """
    Per-face RGBA with two-light Lambertian shading, fully opaque.

    Coordinate system: X = left–right,  Y = depth,  Z = up
    Key light: front-right-above  [0.5, -1.0, 0.8]
    Fill light: back-left         [-0.4,  1.0, 0.3]  (40% strength)
    """
    key  = np.array([ 0.5, -1.0,  0.8]); key  /= np.linalg.norm(key)
    fill = np.array([-0.4,  1.0,  0.3]); fill /= np.linalg.norm(fill)

    face_hit = contact_mask[faces].any(axis=1)          # [F]

    ambient  = 0.38
    i_key    = np.clip(normals @ key,  0, 1)            # [F]
    i_fill   = np.clip(normals @ fill, 0, 1) * 0.70    # [F]
    shading  = np.clip(ambient + i_key + i_fill, 0, 1) # [F]

    base_rgb = np.where(
        face_hit[:, None],
        COLOR_CONTACT[None],
        COLOR_NO_CONTACT[None],
    )  # [F, 3]

    lit_rgb = np.clip(base_rgb * shading[:, None], 0, 1)
    return np.concatenate([lit_rgb, np.ones((len(faces), 1))], axis=1)  # [F, 4]


def render_mesh_3d(ax, vertices: np.ndarray, faces: np.ndarray,
                   contact_mask: np.ndarray, title: str = "",
                   elev: float = 0.0, azim: float = -90.0):
    """
    Draw a lit 3-D mesh on *ax* with back-face culling.

    Expects vertices remapped so that:
      axis 0 (X) = left–right,  axis 1 (Y) = depth,  axis 2 (Z) = up
    azim=-90 → front view (camera at −Y), azim=90 → back view.
    """
    # --- Back-face culling ---
    # Camera position direction (scene → camera, unit vector)
    az, el = np.radians(azim), np.radians(elev)
    cam_dir = np.array([np.cos(el) * np.cos(az),
                        np.cos(el) * np.sin(az),
                        np.sin(el)])
    all_normals = _compute_face_normals(vertices, faces)   # [F, 3]
    visible     = (all_normals @ cam_dir) > 0              # [F] bool
    vis_faces   = faces[visible]
    vis_normals = all_normals[visible]

    fcolors = _face_colors(contact_mask, vis_faces, vis_normals)
    tris    = vertices[vis_faces]                          # [F', 3, 3]

    coll = Poly3DCollection(tris, zsort="average")
    coll.set_facecolor(fcolors)
    coll.set_edgecolor("none")
    ax.add_collection3d(coll)

    # --- Axis limits ---
    # Equal span for X (width) and Z (height); proportional Y (depth)
    xlo, xhi = vertices[:, 0].min(), vertices[:, 0].max()
    zlo, zhi = vertices[:, 2].min(), vertices[:, 2].max()
    span = max(xhi - xlo, zhi - zlo) * 0.38   # tighter → mesh fills more viewport
    xmid, zmid = (xhi + xlo) / 2, (zhi + zlo) / 2
    ax.set_xlim(xmid - span, xmid + span)
    ax.set_zlim(zmid - span, zmid + span)
    ylo, yhi = vertices[:, 1].min(), vertices[:, 1].max()
    ypad = (yhi - ylo) * 0.05
    ax.set_ylim(ylo - ypad, yhi + ypad)

    # Shrink the display box so Y (shallow depth) doesn't waste viewport space
    y_ratio = max((yhi - ylo) / (2 * span), 0.05)
    ax.set_box_aspect([1, y_ratio, 1])

    ax.view_init(elev=elev, azim=azim)
    ax.dist = 4  # zoom in (default=10)
    ax.set_title(title, fontsize=13)
    ax.set_axis_off()


def overlay_mesh_on_image_2d(ax, image: np.ndarray,
                              verts_2d: np.ndarray, verts_3d: np.ndarray,
                              faces: np.ndarray, contact_mask: np.ndarray,
                              title: str = ""):
    """
    Render the full projected mesh as solid filled triangles on top of *image*.

    verts_2d:     [V, 2]  pixel (x, y)
    verts_3d:     [V, 3]  camera-space 3-D coords (used for depth sort + shading)
    faces:        [F, 3]  triangle indices
    contact_mask: [V]     bool
    """
    ax.imshow(image)

    # Depth sort: draw farthest faces first (painter's algorithm)
    face_z    = verts_3d[faces, 2].mean(axis=1)   # [F] avg depth
    order     = np.argsort(-face_z)                # descending → far first
    sf        = faces[order]                       # sorted faces

    # Lambertian shading: camera looks along +Z in OpenCV space,
    # so faces with normal pointing toward −Z face the camera.
    normals  = _compute_face_normals(verts_3d, sf)          # [F, 3]
    diffuse  = np.clip(-normals[:, 2], 0, 1)                # [F]
    shading  = 0.35 + 0.65 * diffuse                        # [F]

    face_hit = contact_mask[sf].any(axis=1)
    base_rgb = np.where(face_hit[:, None],
                        COLOR_CONTACT[None], COLOR_NO_CONTACT[None])
    lit_rgb  = np.clip(base_rgb * shading[:, None], 0, 1)
    rgba     = np.concatenate([lit_rgb, np.ones((len(sf), 1))], axis=1)

    polys = verts_2d[sf]          # [F, 3, 2]
    coll  = PolyCollection(polys, facecolors=rgba, edgecolors="none")
    ax.add_collection(coll)

    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)   # flip Y for image coordinates
    ax.set_title(title, fontsize=13)
    ax.set_axis_off()


# ---------------------------------------------------------------------------
# Per-sample figure
# ---------------------------------------------------------------------------

def make_figure(image: np.ndarray,
                bbox: np.ndarray,
                verts_2d: np.ndarray,
                verts_3d_cam: np.ndarray,
                verts_3d_tpose: np.ndarray,
                faces: np.ndarray,
                gt_mask: np.ndarray,
                pred_mask: np.ndarray,
                iou: float,
                sample_idx: int,
                tpose_faces: np.ndarray | None = None,
                tpose_gt_mask: np.ndarray | None = None,
                tpose_pred_mask: np.ndarray | None = None) -> plt.Figure:
    """
    Row 0 (3 panels): plain image | GT mesh overlay | Pred mesh overlay
    Row 1 (4 panels): GT T-pose (front+back)  |gap|  Pred T-pose (front+back)

    Args:
        tpose_faces:     Face connectivity for lower-row T-pose rendering.
            Defaults to ``faces`` (same mesh as 2D overlay).
        tpose_gt_mask:   Per-vertex contact mask for lower-row GT panels.
            Defaults to ``gt_mask``.
        tpose_pred_mask: Per-vertex contact mask for lower-row Pred panels.
            Defaults to ``pred_mask``.
    """
    if tpose_faces is None:
        tpose_faces = faces
    if tpose_gt_mask is None:
        tpose_gt_mask = gt_mask
    if tpose_pred_mask is None:
        tpose_pred_mask = pred_mask

    fig = plt.figure(figsize=(22, 13), dpi=300)
    fig.suptitle(
        f"Sample #{sample_idx}  |  IoU={iou:.3f}  "
        f"GT contacts={gt_mask.sum()}  Pred contacts={pred_mask.sum()}",
        fontsize=16, y=0.995,
    )

    # ── Subfigures: top row + bottom row ──────────────────────────────────────
    sfig_top, sfig_bot = fig.subfigures(2, 1, height_ratios=[1, 1.6], hspace=0.02)

    # ---- Row 0: 3 equal panels -----------------------------------------------
    gs_top = sfig_top.add_gridspec(1, 3, wspace=0.03,
                                   left=0.01, right=0.99, top=0.84, bottom=0.06)
    ax_img  = sfig_top.add_subplot(gs_top[0])
    ax_gt2d = sfig_top.add_subplot(gs_top[1])
    ax_pr2d = sfig_top.add_subplot(gs_top[2])

    ax_img.imshow(image)
    x1, y1, x2, y2 = bbox.astype(int)
    ax_img.add_patch(Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=1.5, edgecolor=COLOR_BBOX, facecolor="none",
    ))
    ax_img.set_title("Input image", fontsize=14)
    ax_img.set_axis_off()

    overlay_mesh_on_image_2d(ax_gt2d, image, verts_2d, verts_3d_cam,
                              faces, gt_mask, title="GT contact (MHR)")
    overlay_mesh_on_image_2d(ax_pr2d, image, verts_2d, verts_3d_cam,
                              faces, pred_mask, title="Pred contact (MHR)")

    # ---- Row 1: GT pair (left) | gap | Pred pair (right) ---------------------
    # Remap T-pose verts: OpenCV (X right, Y down, Z depth) →
    #   matplotlib (X right, Y depth, Z up)
    tv = np.stack([ verts_3d_tpose[:, 0],
                    verts_3d_tpose[:, 2],
                   -verts_3d_tpose[:, 1]], axis=1)

    sfig_gt, sfig_pr = sfig_bot.subfigures(1, 2, wspace=0.12)

    # Vertical separator between GT and Pred halves
    from matplotlib.lines import Line2D
    sfig_bot.add_artist(Line2D(
        [0.5, 0.5], [0.03, 0.97],
        transform=sfig_bot.transSubfigure,
        color="#888888", linewidth=1.5, linestyle="--",
    ))

    sfig_gt.suptitle("Ground Truth (SMPL)", fontsize=15, fontweight="bold", y=0.97)
    sfig_pr.suptitle("Prediction (SMPL)",   fontsize=15, fontweight="bold", y=0.97)

    gs_gt = sfig_gt.add_gridspec(1, 2, wspace=0.01,
                                  left=0.01, right=0.99, top=0.92, bottom=0.00)
    gs_pr = sfig_pr.add_gridspec(1, 2, wspace=0.01,
                                  left=0.01, right=0.99, top=0.92, bottom=0.00)

    ax_gt_front = sfig_gt.add_subplot(gs_gt[0], projection="3d")
    ax_gt_back  = sfig_gt.add_subplot(gs_gt[1], projection="3d")
    ax_pr_front = sfig_pr.add_subplot(gs_pr[0], projection="3d")
    ax_pr_back  = sfig_pr.add_subplot(gs_pr[1], projection="3d")

    render_mesh_3d(ax_gt_front, tv, tpose_faces, tpose_gt_mask,
                   title="Front", elev=30, azim=-90)
    render_mesh_3d(ax_gt_back,  tv, tpose_faces, tpose_gt_mask,
                   title="Back",  elev=-30, azim=90)
    render_mesh_3d(ax_pr_front, tv, tpose_faces, tpose_pred_mask,
                   title="Front", elev=30, azim=-90)
    render_mesh_3d(ax_pr_back,  tv, tpose_faces, tpose_pred_mask,
                   title="Back",  elev=-30, azim=90)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inference demo — per-vertex contact prediction on DAMON"
    )
    parser.add_argument("--config",      type=str, default="configs/step1_contact.yaml")
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="Path to trained checkpoint (.pth)")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to visualize")
    parser.add_argument("--split",       type=str, default="val",
                        choices=["train", "val", "trainval", "test"],
                        help="Which dataset split to sample from")
    parser.add_argument("--output_dir",  type=str,
                        default="train/inference_samples",
                        help="Directory to save figures")
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="Contact probability threshold")
    parser.add_argument("--seed",        type=int, default=0,
                        help="Random seed for sample selection")
    parser.add_argument("--device",      type=str, default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Config ----
    cfg = get_config(args.config)

    # ---- Model ----
    print("Loading SAM-3D-Body model...")
    model, model_cfg = load_sam_3d_body(
        checkpoint_path=cfg.MODEL.CHECKPOINT_PATH,
        device=args.device,
        mhr_path=cfg.MODEL.MHR_MODEL_PATH,
    )

    # Reinitialize InteractionDecoder + ContactHead from training config
    dim = model_cfg.MODEL.DECODER.DIM
    id_cfg = cfg.MODEL.INTERACTION_DECODER
    ch_cfg = cfg.MODEL.CONTACT_HEAD
    model.interaction_decoder = InteractionDecoder(
        d_model=id_cfg.get('D_MODEL', dim),
        image_feat_dim=id_cfg.get('IMAGE_FEAT_DIM', 1280),
        num_contact_tokens=id_cfg.get('NUM_CONTACT_TOKENS', 16),
        num_layers=id_cfg.get('NUM_LAYERS', 4),
        num_heads=id_cfg.get('NUM_HEADS', 8),
        ffn_dim=id_cfg.get('FFN_DIM', 2048),
        dropout=id_cfg.get('DROPOUT', 0.0),
    ).to(args.device)
    model.head_contact = ContactHead(
        input_dim=dim,
        num_contact_tokens=id_cfg.get('NUM_CONTACT_TOKENS', 16),
        num_vertices=ch_cfg.get('NUM_VERTICES', 18439),
        mlp_depth=ch_cfg.get('MLP_DEPTH', 2),
        mlp_channel_div_factor=ch_cfg.get('MLP_CHANNEL_DIV_FACTOR', 4),
        pool_mode=ch_cfg.get('POOL_MODE', 'attention'),
        dropout=ch_cfg.get('DROPOUT', 0.0),
    ).to(args.device)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    lod = cfg.DATASET.get('LOD', 1)

    # MHR mesh faces: [F, 3]  — used for 2D overlay (row 0)
    faces = model.head_pose.faces.cpu().numpy().astype(np.int32)

    # ---- SMPL template for lower-row (row 1) 3-D rendering ----
    print("Loading SMPL template...")
    smpl_model_path = cfg.MODEL.get("SMPL_MODEL_PATH", None)
    if smpl_model_path is None:
        smpl_model_path = "/data3/rikhat.akizhanov/human_global_motion/better_human/models/smpl/SMPL_NEUTRAL.npz"
        print(f"  MODEL.SMPL_MODEL_PATH not set; using default: {smpl_model_path}")
    smpl_npz = np.load(smpl_model_path, allow_pickle=True)
    smpl_template_verts = smpl_npz["v_template"].astype(np.float32)   # [6890, 3]
    smpl_faces          = smpl_npz["f"].astype(np.int32)               # [13776, 3]

    # ---- Contact converter (MHR → SMPL, and SMPL → MHR for back-projection) ----
    print("Initialising MHR↔SMPL contact converter...")
    converter = BodyConverter(smpl_faces=smpl_faces, device="cpu")

    # ---- Dataset ----
    data_root  = cfg.DATASET.get("DATA_ROOT", None)
    val_ratio  = cfg.DATASET.get("VAL_RATIO", 0.2)
    seed_ds    = cfg.DATASET.get("SEED", 42)
    lod        = cfg.DATASET.get('LOD', 1)
    detect_npz = cfg.DATASET.get('DETECT_NPZ', {})

    # Prefer original SMPL GT (6890 verts) when SMPL NPZ paths are configured.
    smpl_trainval = cfg.DATASET.get('SMPL_TRAINVAL_NPZ', None)
    smpl_test     = cfg.DATASET.get('SMPL_TEST_NPZ', None)
    use_smpl_gt   = bool(smpl_trainval or smpl_test)

    if use_smpl_gt:
        print("Using original SMPL GT contacts (DamonSMPLDataset).")
        if args.split == "test":
            if not smpl_test:
                raise ValueError("DATASET.SMPL_TEST_NPZ not set in config.")
            dataset = DamonSMPLDataset(
                smpl_npz_path=smpl_test,
                detect_npz_path=detect_npz.get('TEST', None),
                data_root=data_root,
            )
        elif args.split == "trainval":
            dataset = DamonSMPLDataset(
                smpl_npz_path=smpl_trainval,
                detect_npz_path=detect_npz.get('TRAINVAL', None),
                data_root=data_root,
            )
        else:
            train_ds, val_ds = DamonSMPLDataset.split_train_val(
                smpl_npz_path=smpl_trainval,
                detect_npz_path=detect_npz.get('TRAINVAL', None),
                val_ratio=val_ratio, seed=seed_ds, data_root=data_root,
            )
            dataset = train_ds if args.split == "train" else val_ds
    else:
        print("SMPL NPZ not configured — using MHR LOD GT with conversion.")
        contact_npz = cfg.DATASET.CONTACT_NPZ
        if args.split == "test":
            dataset = DamonMHRDataset(
                contact_npz_path=contact_npz.TEST,
                detect_npz_path=detect_npz.get('TEST', None),
                lod=lod, data_root=data_root,
            )
        elif args.split == "trainval":
            dataset = DamonMHRDataset(
                contact_npz_path=contact_npz.TRAINVAL,
                detect_npz_path=detect_npz.get('TRAINVAL', None),
                lod=lod, data_root=data_root,
            )
        else:
            train_ds, val_ds = DamonMHRDataset.split_train_val(
                contact_npz_path=contact_npz.TRAINVAL,
                detect_npz_path=detect_npz.get('TRAINVAL', None),
                lod=lod, val_ratio=val_ratio, seed=seed_ds, data_root=data_root,
            )
            dataset = train_ds if args.split == "train" else val_ds

    print(f"Dataset split='{args.split}', size={len(dataset)}")

    # Random indices
    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    # ---- Inference loop ----
    ious = []
    for run_idx, ds_idx in enumerate(indices):
        print(f"\n[{run_idx+1}/{len(indices)}]  dataset index {ds_idx}")

        (image_np, bbox, cam_k), contact_label = dataset[ds_idx]

        # Prepare model batch (single sample)
        with torch.no_grad():
            batch = prepare_damon_batch(
                [image_np], [bbox], [cam_k],
                target_size=tuple(cfg.MODEL.IMAGE_SIZE),
                device=args.device,
            )
            model._initialize_batch(batch)
            output = model.forward_step(batch, decoder_type="body")
            contact_tokens = model.interaction_decoder(
                output["image_embeddings"], output["body_tokens"]
            )
            contact_logits = model.head_contact(contact_tokens)  # [1, V_lod]

        # ---- Contact predictions (LOD space) ----
        pred_probs_lod = torch.sigmoid(contact_logits[0]).cpu()  # [V_lod] float

        # ---- Pred: LOD_N → SMPL (Voronoi BFS + barycentric) ----
        pred_smpl_probs = converter.mhr_lod_probs_to_smpl_probs(
            pred_probs_lod.unsqueeze(0), source_lod=lod)              # [1, 6890]
        pred_mask_smpl = (pred_smpl_probs[0] > args.threshold).numpy().astype(bool)  # [6890]

        # ---- GT: original SMPL labels or converted from LOD_N ----
        if use_smpl_gt:
            # contact_label is already [6890] from DamonSMPLDataset
            gt_mask_smpl  = contact_label.numpy().astype(bool)           # [6890]
            gt_smpl_float = contact_label.float().unsqueeze(0)           # [1, 6890]
        else:
            # contact_label is [V_lod] from DamonMHRDataset — convert to SMPL
            gt_smpl_float = converter.mhr_lod_probs_to_smpl_probs(
                contact_label.float().unsqueeze(0), source_lod=lod)      # [1, 6890]
            gt_mask_smpl  = (gt_smpl_float[0] > 0.5).numpy().astype(bool)  # [6890]

        # ---- For 2D overlay: back-project SMPL → LOD1 ----
        # Both rows derive from the same SMPL contacts (consistent body-part regions).
        pred_mask_lod1 = converter.smpl_to_mhr(
            contacts=pred_smpl_probs, threshold=args.threshold, target_lod=1,
        ).contacts[0].numpy().astype(bool)                            # [18439]
        gt_mask_lod1 = converter.smpl_to_mhr(
            contacts=gt_smpl_float, threshold=0.5, target_lod=1,
        ).contacts[0].numpy().astype(bool)                            # [18439]

        # ---- SMPL T-pose template for lower row ----
        verts_3d_tpose = smpl_template_verts.copy()
        verts_3d_tpose[:, [1, 2]] *= -1   # [6890, 3]  (Y-up → OpenCV flip)

        # ---- 3D posed vertices (LOD1) + 2D projection ----
        verts_3d_posed = output["mhr"]["pred_vertices"][0].cpu().numpy()          # [V_lod1, 3]
        pred_cam_t_np  = output["mhr"]["pred_cam_t"][0].cpu().numpy()             # [3]
        verts_3d_cam   = verts_3d_posed + pred_cam_t_np                           # [V_lod1, 3]
        verts_2d       = output["mhr"]["pred_keypoints_2d_verts"][0].cpu().numpy()  # [V_lod1, 2]

        # ---- IoU — computed in SMPL space ----
        tp = (pred_mask_smpl & gt_mask_smpl).sum()
        fp = (pred_mask_smpl & ~gt_mask_smpl).sum()
        fn = (~pred_mask_smpl & gt_mask_smpl).sum()
        iou = float(tp) / (tp + fp + fn + 1e-8)
        ious.append(iou)

        gt_src = "SMPL (original)" if use_smpl_gt else f"LOD{lod}→SMPL"
        print(f"  GT contacts (SMPL, {gt_src}): {gt_mask_smpl.sum()}")
        print(f"  Pred contacts (SMPL):          {pred_mask_smpl.sum()}")
        print(f"  IoU (SMPL):                    {iou:.4f}")

        # ---- Figure ----
        # Row 0: 2D overlay uses LOD1 mesh (pose head always outputs LOD1)
        # Row 1: 3D T-pose uses SMPL template mesh with SMPL-space contact masks
        fig = make_figure(
            image           = image_np,
            bbox            = bbox.numpy(),
            verts_2d        = verts_2d,
            verts_3d_cam    = verts_3d_cam,
            verts_3d_tpose  = verts_3d_tpose,
            faces           = faces,              # LOD1 MHR faces for 2D overlay
            gt_mask         = gt_mask_lod1,       # LOD1 mask for 2D overlay
            pred_mask       = pred_mask_lod1,     # LOD1 mask for 2D overlay
            iou             = iou,
            sample_idx      = ds_idx,
            tpose_faces     = smpl_faces,         # SMPL faces for lower row
            tpose_gt_mask   = gt_mask_smpl,       # SMPL mask for lower row GT
            tpose_pred_mask = pred_mask_smpl,     # SMPL mask for lower row Pred
        )

        save_path = output_dir / f"sample_{run_idx:04d}_idx{ds_idx}_iou{iou:.3f}.png"
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"Done.  {len(ious)} samples  |  mean IoU (SMPL) = {np.mean(ious):.4f}  "
          f"(median {np.median(ious):.4f})")
    print(f"Figures saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
