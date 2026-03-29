"""
Inference demo for per-vertex Contact Head on DAMON dataset.

For each sampled image, generates a figure with:
  Row 0 — Image + 2D projected mesh:
      Left:  plain image with bbox
      Mid:   ground-truth contact vertices highlighted in red
      Right: predicted contact vertices highlighted in red
  Row 1 — T-pose 3D mesh (front + back view, same shape/scale as prediction):
      GT contact coloring: front and back
      Pred contact coloring: front and back
  Row 2 — Body-part contact bar chart (24 SMPL parts, GT vs Predicted)

Usage:
    CUDA_VISIBLE_DEVICES=3 python train/inference_demo.py \\
        --config configs/step1_contact.yaml \\
        --checkpoint train/output/step1_contact_20260228_123456/best_model.pth \\
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
import torch.nn.functional as F
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
from sam_3d_body.models.heads.contact_head import ContactHead, PartContactHead
from sam_3d_body.models.decoders.interaction_decoder import InteractionDecoder
from sam_3d_body.utils.config import get_config
from damon_dataset import DamonDataset
from dataset_utils import prepare_damon_batch
from train_contact import damon_collate, _preprocess_object_mask
from body_converter import BodyConverter


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLOR_CONTACT    = np.array([0.95, 0.15, 0.15])   # red  — in contact
COLOR_NO_CONTACT = np.array([0.55, 0.65, 0.80])   # steel-blue — no contact
COLOR_BBOX       = (1.0, 0.9, 0.0)                 # yellow bbox


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def _compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unit outward face normals. Returns [F, 3]."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    n  = np.cross(v1 - v0, v2 - v0)
    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-8)
    return n


def _face_colors(contact_mask: np.ndarray, faces: np.ndarray,
                 normals: np.ndarray) -> np.ndarray:
    """Per-face RGBA with two-light Lambertian shading, fully opaque."""
    key  = np.array([ 0.5, -1.0,  0.8]); key  /= np.linalg.norm(key)
    fill = np.array([-0.4,  1.0,  0.3]); fill /= np.linalg.norm(fill)

    face_hit = contact_mask[faces].any(axis=1)

    ambient = 0.38
    i_key   = np.clip(normals @ key,  0, 1)
    i_fill  = np.clip(normals @ fill, 0, 1) * 0.70
    shading = np.clip(ambient + i_key + i_fill, 0, 1)

    base_rgb = np.where(face_hit[:, None], COLOR_CONTACT[None], COLOR_NO_CONTACT[None])
    lit_rgb  = np.clip(base_rgb * shading[:, None], 0, 1)
    return np.concatenate([lit_rgb, np.ones((len(faces), 1))], axis=1)  # [F, 4]


def render_mesh_3d(ax, vertices: np.ndarray, faces: np.ndarray,
                   contact_mask: np.ndarray, title: str = "",
                   elev: float = 0.0, azim: float = -90.0):
    """Draw a lit 3-D mesh on *ax* with back-face culling."""
    az, el  = np.radians(azim), np.radians(elev)
    cam_dir = np.array([np.cos(el) * np.cos(az),
                        np.cos(el) * np.sin(az),
                        np.sin(el)])
    all_normals = _compute_face_normals(vertices, faces)
    visible     = (all_normals @ cam_dir) > 0
    vis_faces   = faces[visible]
    vis_normals = all_normals[visible]

    fcolors = _face_colors(contact_mask, vis_faces, vis_normals)
    tris    = vertices[vis_faces]

    coll = Poly3DCollection(tris, zsort="average")
    coll.set_facecolor(fcolors)
    coll.set_edgecolor("none")
    ax.add_collection3d(coll)

    xlo, xhi = vertices[:, 0].min(), vertices[:, 0].max()
    zlo, zhi = vertices[:, 2].min(), vertices[:, 2].max()
    span = max(xhi - xlo, zhi - zlo) * 0.38
    xmid, zmid = (xhi + xlo) / 2, (zhi + zlo) / 2
    ax.set_xlim(xmid - span, xmid + span)
    ax.set_zlim(zmid - span, zmid + span)
    ylo, yhi = vertices[:, 1].min(), vertices[:, 1].max()
    ypad = (yhi - ylo) * 0.05
    ax.set_ylim(ylo - ypad, yhi + ypad)

    y_ratio = max((yhi - ylo) / (2 * span), 0.05)
    ax.set_box_aspect([1, y_ratio, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.dist = 4
    ax.set_title(title, fontsize=13)
    ax.set_axis_off()


def overlay_mesh_on_image_2d(ax, image: np.ndarray,
                              verts_2d: np.ndarray, verts_3d: np.ndarray,
                              faces: np.ndarray, contact_mask: np.ndarray,
                              title: str = ""):
    """Render projected mesh as solid filled triangles on *image*."""
    ax.imshow(image)
    face_z   = verts_3d[faces, 2].mean(axis=1)
    order    = np.argsort(-face_z)
    sf       = faces[order]
    normals  = _compute_face_normals(verts_3d, sf)
    diffuse  = np.clip(-normals[:, 2], 0, 1)
    shading  = 0.35 + 0.65 * diffuse
    face_hit = contact_mask[sf].any(axis=1)
    base_rgb = np.where(face_hit[:, None], COLOR_CONTACT[None], COLOR_NO_CONTACT[None])
    lit_rgb  = np.clip(base_rgb * shading[:, None], 0, 1)
    rgba     = np.concatenate([lit_rgb, np.ones((len(sf), 1))], axis=1)
    polys    = verts_2d[sf]
    coll     = PolyCollection(polys, facecolors=rgba, edgecolors="none")
    ax.add_collection(coll)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_title(title, fontsize=13)
    ax.set_axis_off()


def _draw_mask_overlay(ax, image_hw: tuple, mask: np.ndarray,
                       fill_rgba: tuple, border_rgb: tuple,
                       border_thickness: int = 3) -> None:
    """Draw a boolean mask overlay on *ax*."""
    h, w = image_hw
    if mask.shape != (h, w):
        m8 = cv2.resize(mask.astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST)
    else:
        m8 = mask.astype(np.uint8)

    overlay = np.zeros((h, w, 4), dtype=np.float32)
    overlay[m8.astype(bool)] = fill_rgba
    ax.imshow(overlay)

    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    border_layer = np.zeros((h, w, 4), dtype=np.float32)
    border_u8 = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.drawContours(border_u8, contours, -1,
                     (int(border_rgb[0]*255), int(border_rgb[1]*255), int(border_rgb[2]*255)),
                     thickness=border_thickness)
    drawn = border_u8.sum(axis=2) > 0
    border_layer[drawn, :3] = border_rgb
    border_layer[drawn, 3] = 1.0
    ax.imshow(border_layer)


# ---------------------------------------------------------------------------
# Part contact bar chart
# ---------------------------------------------------------------------------

def _plot_part_bars(ax, gt_parts: np.ndarray, pred_parts: np.ndarray,
                    part_names: list, title: str = "Body-part contact"):
    """
    Grouped bar chart: GT (blue) vs Predicted probability (orange) for 24 body parts.

    Args:
        gt_parts:   [24] bool or int  — GT binary contact per part
        pred_parts: [24] float        — predicted probability per part
        part_names: list of 24 strings
    """
    x     = np.arange(len(part_names))
    width = 0.35

    ax.bar(x - width/2, gt_parts.astype(float),  width, label='GT',
           color='steelblue', edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, pred_parts,               width, label='Predicted',
           color='darkorange', edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(part_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.5, label='Threshold')
    ax.set_ylabel('Contact probability / GT')
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)


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
                tpose_pred_mask: np.ndarray | None = None,
                person_mask: np.ndarray | None = None,
                object_mask: np.ndarray | None = None,
                object_name: str = "",
                gt_parts: np.ndarray | None = None,
                pred_parts: np.ndarray | None = None,
                part_names: list | None = None) -> plt.Figure:
    """
    Row 0 (3 panels): plain image | GT mesh overlay | Pred mesh overlay
    Row 1 (4 panels): GT T-pose (front+back)  |gap|  Pred T-pose (front+back)
    Row 2 (1 panel):  Body-part contact bar chart (only when gt_parts/pred_parts provided)
    """
    if tpose_faces is None:
        tpose_faces = faces
    if tpose_gt_mask is None:
        tpose_gt_mask = gt_mask
    if tpose_pred_mask is None:
        tpose_pred_mask = pred_mask

    has_parts = gt_parts is not None and pred_parts is not None

    height_ratios = [1, 1.6, 0.7] if has_parts else [1, 1.6]
    fig = plt.figure(figsize=(22, 17 if has_parts else 13), dpi=300)
    fig.suptitle(
        f"Sample #{sample_idx}  |  IoU={iou:.3f}  "
        f"GT contacts={gt_mask.sum()}  Pred contacts={pred_mask.sum()}",
        fontsize=16, y=0.995,
    )

    if has_parts:
        sfigs = fig.subfigures(3, 1, height_ratios=height_ratios, hspace=0.02)
        sfig_top, sfig_bot, sfig_parts = sfigs
    else:
        sfigs = fig.subfigures(2, 1, height_ratios=height_ratios, hspace=0.02)
        sfig_top, sfig_bot = sfigs
        sfig_parts = None

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
    hw = image.shape[:2]
    if person_mask is not None:
        _draw_mask_overlay(ax_img, hw, person_mask,
                           fill_rgba=(0.20, 0.55, 1.0, 0.35),
                           border_rgb=(0.10, 0.40, 1.0), border_thickness=4)
    if object_mask is not None:
        _draw_mask_overlay(ax_img, hw, object_mask,
                           fill_rgba=(1.0, 0.50, 0.0, 0.40),
                           border_rgb=(1.0, 0.30, 0.0), border_thickness=4)
    img_title = "Input image" + (f"\n{object_name}" if object_name else "")
    ax_img.set_title(img_title, fontsize=14)
    ax_img.set_axis_off()

    overlay_mesh_on_image_2d(ax_gt2d, image, verts_2d, verts_3d_cam,
                              faces, gt_mask, title="GT contact (LOD1)")
    overlay_mesh_on_image_2d(ax_pr2d, image, verts_2d, verts_3d_cam,
                              faces, pred_mask, title="Pred contact (LOD1)")

    # ---- Row 1: GT pair (left) | gap | Pred pair (right) ---------------------
    tv = np.stack([ verts_3d_tpose[:, 0],
                    verts_3d_tpose[:, 2],
                   -verts_3d_tpose[:, 1]], axis=1)

    sfig_gt, sfig_pr = sfig_bot.subfigures(1, 2, wspace=0.12)

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
                   title="Front", elev=30,  azim=-90)
    render_mesh_3d(ax_gt_back,  tv, tpose_faces, tpose_gt_mask,
                   title="Back",  elev=-30, azim=90)
    render_mesh_3d(ax_pr_front, tv, tpose_faces, tpose_pred_mask,
                   title="Front", elev=30,  azim=-90)
    render_mesh_3d(ax_pr_back,  tv, tpose_faces, tpose_pred_mask,
                   title="Back",  elev=-30, azim=90)

    # ---- Row 2: Part contact bar chart (optional) ----------------------------
    if has_parts and sfig_parts is not None:
        gs_p = sfig_parts.add_gridspec(1, 1, left=0.05, right=0.98,
                                        top=0.85, bottom=0.25)
        ax_parts = sfig_parts.add_subplot(gs_p[0])
        names = part_names or [str(i) for i in range(len(gt_parts))]
        _plot_part_bars(ax_parts, gt_parts, pred_parts, names,
                        title="Body-part contact: GT vs Predicted")

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
    parser.add_argument("--masks_v2_dir", type=str, default=None,
                        help="Path to masks_v2 directory for Step 2 mask-conditioned inference.")
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

    dim    = model_cfg.MODEL.DECODER.DIM
    id_cfg = cfg.MODEL.INTERACTION_DECODER
    ch_cfg = cfg.MODEL.CONTACT_HEAD

    model.interaction_decoder = InteractionDecoder(
        d_model=id_cfg.get('D_MODEL', dim),
        image_feat_dim=id_cfg.get('IMAGE_FEAT_DIM', 1280),
        num_layers=id_cfg.get('NUM_LAYERS', 4),
        num_heads=id_cfg.get('NUM_HEADS', 8),
        ffn_dim=id_cfg.get('FFN_DIM', 2048),
        dropout=id_cfg.get('DROPOUT', 0.0),
    ).to(args.device)
    model.head_contact = ContactHead(
        input_dim=id_cfg.get('D_MODEL', dim),
        num_vertices=ch_cfg.get('NUM_VERTICES', 6890),
        mlp_depth=ch_cfg.get('MLP_DEPTH', 2),
        hidden_dim=ch_cfg.get('HIDDEN_DIM', 512),
        dropout=ch_cfg.get('DROPOUT', 0.0),
    ).to(args.device)
    model.head_part_contact = PartContactHead(
        input_dim=id_cfg.get('D_MODEL', dim),
    ).to(args.device)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location=args.device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    step2_mode = cfg.TRAIN.get('STEP2_MODE', False)
    print(f"{'Step 2' if step2_mode else 'Step 1'} checkpoint.")

    # MHR LOD1 faces for 2D overlay (pose head always outputs LOD1)
    faces = model.head_pose.faces.cpu().numpy().astype(np.int32)

    # SMPL template for 3D T-pose row
    print("Loading SMPL template...")
    smpl_model_path = cfg.MODEL.get("SMPL_MODEL_PATH", None)
    if smpl_model_path is None:
        smpl_model_path = "/data3/rikhat.akizhanov/human_global_motion/better_human/models/smpl/SMPL_NEUTRAL.npz"
        print(f"  MODEL.SMPL_MODEL_PATH not set; using default: {smpl_model_path}")
    smpl_npz            = np.load(smpl_model_path, allow_pickle=True)
    smpl_template_verts = smpl_npz["v_template"].astype(np.float32)  # [6890, 3]
    smpl_faces          = smpl_npz["f"].astype(np.int32)              # [13776, 3]

    # BodyConverter: still needed for SMPL→LOD1 back-projection for 2D overlay
    print("Initialising SMPL↔MHR contact converter for 2D overlay...")
    converter = BodyConverter(smpl_faces=smpl_faces, device="cpu")

    # Part names
    smpl_part_seg = cfg.DATASET.get('SMPL_PART_SEG_PATH', None)
    part_names = None
    if smpl_part_seg:
        from damon_utils import load_smpl_part_segmentation
        part_names, _ = load_smpl_part_segmentation(smpl_part_seg)
        print(f"Loaded {len(part_names)} SMPL part names.")

    # ---- Dataset ----
    data_root  = cfg.DATASET.get("DATA_ROOT", None)
    val_ratio  = cfg.DATASET.get("VAL_RATIO", 0.2)
    seed_ds    = cfg.DATASET.get("SEED", 42)
    detect_npz = cfg.DATASET.get('DETECT_NPZ', {})
    contact_npz = cfg.DATASET.CONTACT_NPZ

    use_instance_mode = bool(args.masks_v2_dir and step2_mode)

    ds_kwargs = dict(
        topology='smpl',
        smpl_part_seg_path=smpl_part_seg,
        data_root=data_root,
    )
    if use_instance_mode:
        ds_kwargs['mode'] = 'instance_contact'
        ds_kwargs['masks_v2_dir'] = args.masks_v2_dir
        print(f"Step 2 mode: loading instance_contact dataset from {args.masks_v2_dir}")

    if args.split == "test":
        dataset = DamonDataset(
            contact_npz_path=contact_npz.TEST,
            detect_npz_path=detect_npz.get('TEST', None),
            **ds_kwargs,
        )
    elif args.split == "trainval":
        dataset = DamonDataset(
            contact_npz_path=contact_npz.TRAINVAL,
            detect_npz_path=detect_npz.get('TRAINVAL', None),
            **ds_kwargs,
        )
    else:
        train_ds, val_ds = DamonDataset.split_train_val(
            contact_npz_path=contact_npz.TRAINVAL,
            detect_npz_path=detect_npz.get('TRAINVAL', None),
            val_ratio=val_ratio, seed=seed_ds, **ds_kwargs,
        )
        dataset = train_ds if args.split == "train" else val_ds

    print(f"Dataset split='{args.split}', size={len(dataset)}")

    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    # ---- Inference loop ----
    ious = []
    for run_idx, ds_idx in enumerate(indices):
        print(f"\n[{run_idx+1}/{len(indices)}]  dataset index {ds_idx}")

        item = dataset[ds_idx]
        if use_instance_mode:
            inputs_dict, label_dict = item
            image_np        = inputs_dict['image']
            bbox            = inputs_dict['person_bbox']
            cam_k           = inputs_dict['cam_k']
            contact_label   = label_dict['contact_label']    # [6890] SMPL
            gt_parts_label  = label_dict.get('part_contact') # [24] or None
            person_mask_raw = inputs_dict.get('person_mask')
            obj_mask_raw    = inputs_dict.get('object_mask')
            object_name     = label_dict.get('object_name', '')
            obj_mask_t         = _preprocess_object_mask(obj_mask_raw, bbox)
            person_mask_t      = _preprocess_object_mask(person_mask_raw, bbox)
            obj_mask_tensor    = obj_mask_t.unsqueeze(0).to(args.device)
            person_mask_tensor = person_mask_t.unsqueeze(0).to(args.device)
        else:
            (image_np, bbox, cam_k), lbl = item
            if isinstance(lbl, dict):
                contact_label  = lbl['contact_label']
                gt_parts_label = lbl.get('part_contact')
            else:
                contact_label  = lbl
                gt_parts_label = None
            person_mask_raw = None
            obj_mask_raw    = None
            obj_mask_tensor = None
            object_name     = ''

        # ---- Model forward ----
        with torch.no_grad():
            batch = prepare_damon_batch(
                [image_np], [bbox], [cam_k],
                target_size=tuple(cfg.MODEL.IMAGE_SIZE),
                device=args.device,
            )
            model._initialize_batch(batch)
            output = model.forward_step(batch, decoder_type="body")

            # Generic (no mask) forward
            tokens = model.interaction_decoder(
                output["image_embeddings"], output["body_tokens"],
            )
            part_tokens  = tokens[:, :InteractionDecoder.NUM_PART_TOKENS, :]
            vertex_token = tokens[:, InteractionDecoder.NUM_PART_TOKENS:,  :]
            vertex_logits = model.head_contact(vertex_token)        # [1, 6890]
            part_logits   = model.head_part_contact(part_tokens)    # [1, 24]

            # Mask-conditioned forward (Step 2)
            vertex_logits_masked = None
            part_logits_masked   = None
            if obj_mask_tensor is not None:
                tokens_m = model.interaction_decoder(
                    output["image_embeddings"], output["body_tokens"],
                    person_mask=person_mask_tensor,
                    object_mask=obj_mask_tensor,
                )
                part_tokens_m  = tokens_m[:, :InteractionDecoder.NUM_PART_TOKENS, :]
                vertex_token_m = tokens_m[:, InteractionDecoder.NUM_PART_TOKENS:,  :]
                vertex_logits_masked = model.head_contact(vertex_token_m)
                part_logits_masked   = model.head_part_contact(part_tokens_m)

        # ---- Predictions (SMPL space, 6890 verts) ----
        pred_probs_smpl  = torch.sigmoid(vertex_logits[0]).cpu()    # [6890]
        pred_mask_smpl   = (pred_probs_smpl > args.threshold).numpy().astype(bool)
        pred_parts_probs = torch.sigmoid(part_logits[0]).cpu().numpy()  # [24]

        # ---- GT (already SMPL) ----
        gt_mask_smpl = contact_label.numpy().astype(bool)   # [6890]

        # ---- For 2D overlay: project SMPL → LOD1 ----
        pred_mask_lod1 = converter.smpl_to_mhr(
            contacts=pred_probs_smpl.unsqueeze(0), threshold=args.threshold, target_lod=1,
        ).contacts[0].numpy().astype(bool)                          # [18439]
        gt_smpl_float  = contact_label.float().unsqueeze(0)        # [1, 6890]
        gt_mask_lod1   = converter.smpl_to_mhr(
            contacts=gt_smpl_float, threshold=0.5, target_lod=1,
        ).contacts[0].numpy().astype(bool)                          # [18439]

        # ---- SMPL T-pose template ----
        verts_3d_tpose = smpl_template_verts.copy()
        verts_3d_tpose[:, [1, 2]] *= -1   # Y-up → OpenCV flip

        # ---- LOD1 posed vertices for 2D projection ----
        verts_3d_posed = output["mhr"]["pred_vertices"][0].cpu().numpy()
        pred_cam_t_np  = output["mhr"]["pred_cam_t"][0].cpu().numpy()
        verts_3d_cam   = verts_3d_posed + pred_cam_t_np
        verts_2d       = output["mhr"]["pred_keypoints_2d_verts"][0].cpu().numpy()

        # ---- IoU (SMPL space) ----
        tp  = (pred_mask_smpl & gt_mask_smpl).sum()
        fp  = (pred_mask_smpl & ~gt_mask_smpl).sum()
        fn  = (~pred_mask_smpl & gt_mask_smpl).sum()
        iou = float(tp) / (tp + fp + fn + 1e-8)
        ious.append(iou)

        print(f"  GT contacts  (SMPL): {gt_mask_smpl.sum()}")
        if object_name:
            print(f"  Object: {object_name}")
        print(f"  Pred contacts (SMPL, no mask): {pred_mask_smpl.sum()}")
        print(f"  IoU (SMPL, no mask):           {iou:.4f}")

        if vertex_logits_masked is not None:
            pred_mask_smpl_m = (torch.sigmoid(vertex_logits_masked[0]).cpu() > args.threshold).numpy()
            tp_m = (pred_mask_smpl_m & gt_mask_smpl).sum()
            fp_m = (pred_mask_smpl_m & ~gt_mask_smpl).sum()
            fn_m = (~pred_mask_smpl_m & gt_mask_smpl).sum()
            iou_m = float(tp_m) / (tp_m + fp_m + fn_m + 1e-8)
            print(f"  Pred contacts (SMPL, with mask): {pred_mask_smpl_m.sum()}")
            print(f"  IoU (SMPL, with mask):           {iou_m:.4f}")

        # ---- Figure ----
        gt_parts_np  = gt_parts_label.numpy().astype(bool) if gt_parts_label is not None else None
        fig = make_figure(
            image           = image_np,
            bbox            = bbox.numpy() if hasattr(bbox, 'numpy') else np.array(bbox),
            verts_2d        = verts_2d,
            verts_3d_cam    = verts_3d_cam,
            verts_3d_tpose  = verts_3d_tpose,
            faces           = faces,
            gt_mask         = gt_mask_lod1,
            pred_mask       = pred_mask_lod1,
            iou             = iou,
            sample_idx      = ds_idx,
            tpose_faces     = smpl_faces,
            tpose_gt_mask   = gt_mask_smpl,
            tpose_pred_mask = pred_mask_smpl,
            person_mask     = person_mask_raw if use_instance_mode else None,
            object_mask     = obj_mask_raw    if use_instance_mode else None,
            object_name     = object_name,
            gt_parts        = gt_parts_np,
            pred_parts      = pred_parts_probs,
            part_names      = part_names,
        )

        save_path = output_dir / f"sample_{run_idx:04d}_idx{ds_idx}_iou{iou:.3f}.png"
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")

    print(f"\n{'='*60}")
    print(f"Done.  {len(ious)} samples  |  mean IoU (SMPL) = {np.mean(ious):.4f}  "
          f"(median {np.median(ious):.4f})")
    print(f"Figures saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
