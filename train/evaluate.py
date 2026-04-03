"""
Evaluation script for the per-vertex Contact Head on DAMON dataset.

Evaluates a trained checkpoint on any of the three splits (train / val / test)
and reports:
  - Per-vertex: accuracy, precision, recall, F1, IoU, geodesic distance
  - Per-body-part: per-part accuracy + mean part accuracy (24 SMPL parts)
  - ROC curve + AUC

Usage:
    python train/evaluate.py \
        --config configs/step1_contact.yaml \
        --checkpoint train/output/step1_contact_20260228_123456/best_model.pth \
        --split val
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "dataset"))
os.environ["MOMENTUM_ENABLED"] = "1"

from sam_3d_body.build_models import load_sam_3d_body
from sam_3d_body.models.heads.contact_head import ContactHead, PartContactHead
from sam_3d_body.models.decoders.interaction_decoder import InteractionDecoder
from sam_3d_body.utils.config import get_config
from damon_dataset import DamonDataset
from dataset_utils import prepare_damon_batch
from train_contact import damon_collate, _preprocess_object_mask


# ---------------------------------------------------------------------------

def _instance_eval_collate(batch):
    """
    Collate for DamonDataset instance_contact mode (raw images).
    Returns: images, bboxes, cam_ks, vertex_labels, part_labels_or_None,
             person_masks [B,1,56,56], obj_masks [B,1,56,56]
    """
    images, bboxes, cam_ks, vlabels, plabels = [], [], [], [], []
    person_masks_prep, obj_masks_prep = [], []
    has_parts = None
    for inputs_dict, label_dict in batch:
        images.append(inputs_dict['image'])
        bboxes.append(inputs_dict['person_bbox'])
        cam_ks.append(inputs_dict['cam_k'])
        vl = label_dict['contact_label']
        pl = label_dict.get('part_contact', None)
        vlabels.append(vl)
        plabels.append(pl)
        if has_parts is None:
            has_parts = pl is not None
        person_masks_prep.append(
            _preprocess_object_mask(
                inputs_dict.get('person_mask'),
                inputs_dict['person_bbox'],
            )
        )
        obj_masks_prep.append(
            _preprocess_object_mask(
                inputs_dict.get('object_mask'),
                inputs_dict['person_bbox'],
            )
        )
    return (
        images,
        torch.stack(bboxes),
        torch.stack(cam_ks),
        torch.stack(vlabels),
        torch.stack(plabels) if has_parts else None,
        torch.stack(person_masks_prep),  # [B, 1, 56, 56]
        torch.stack(obj_masks_prep),     # [B, 1, 56, 56]
    )


class ContactEvaluator:
    """Evaluates per-vertex contact prediction on DAMON dataset."""

    def __init__(self, config_path: str, checkpoint_path: str,
                 split: str = "val", device: str = "cuda",
                 masks_v2_dir: str = None):
        self.device = device
        self.split = split
        self.checkpoint_path = Path(checkpoint_path)
        self.masks_v2_dir = masks_v2_dir

        self.figures_dir = Path(__file__).parent / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.cfg = get_config(config_path)

        # Load model
        print("Loading SAM-3D-Body model...")
        self.model, self.model_cfg = load_sam_3d_body(
            checkpoint_path=self.cfg.MODEL.CHECKPOINT_PATH,
            device=device,
            mhr_path=self.cfg.MODEL.MHR_MODEL_PATH,
        )

        # Reinitialize contact modules from the checkpoint's saved config.yaml
        ckpt_dir = self.checkpoint_path.parent
        ckpt_config_path = ckpt_dir / "config.yaml"
        if ckpt_config_path.exists():
            print(f"Using checkpoint's saved config: {ckpt_config_path}")
            ckpt_cfg = get_config(str(ckpt_config_path))
        else:
            print(f"WARNING: No config.yaml found in {ckpt_dir}, falling back to current config")
            ckpt_cfg = self.cfg

        dim = self.model_cfg.MODEL.DECODER.DIM
        id_cfg = ckpt_cfg.MODEL.INTERACTION_DECODER
        ch_cfg = ckpt_cfg.MODEL.CONTACT_HEAD

        self.model.interaction_decoder = InteractionDecoder(
            d_model=id_cfg.get('D_MODEL', dim),
            image_feat_dim=id_cfg.get('IMAGE_FEAT_DIM', 1280),
            num_layers=id_cfg.get('NUM_LAYERS', 4),
            num_heads=id_cfg.get('NUM_HEADS', 8),
            ffn_dim=id_cfg.get('FFN_DIM', 2048),
            dropout=id_cfg.get('DROPOUT', 0.0),
        ).to(device)
        self.model.head_contact = ContactHead(
            input_dim=id_cfg.get('D_MODEL', dim),
            num_vertices=ch_cfg.get('NUM_VERTICES', 6890),
            mlp_depth=ch_cfg.get('MLP_DEPTH', 2),
            hidden_dim=ch_cfg.get('HIDDEN_DIM', 512),
            dropout=ch_cfg.get('DROPOUT', 0.0),
        ).to(device)
        self.model.head_part_contact = PartContactHead(
            input_dim=id_cfg.get('D_MODEL', dim),
        ).to(device)
        print(f"InteractionDecoder: {InteractionDecoder.NUM_TOKENS} tokens "
              f"({InteractionDecoder.NUM_PART_TOKENS} part + {InteractionDecoder.NUM_VERTEX_TOKENS} vertex), "
              f"{id_cfg.get('NUM_LAYERS', 4)} layers")

        # Load trained checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        missing, unexpected = self.model.load_state_dict(state, strict=False)

        # mask_encoder keys are safe to be absent (zero-initialized → no-op)
        contact_missing = [k for k in missing
                           if ('interaction_decoder' in k or 'head_contact' in k
                               or 'head_part_contact' in k)
                           and 'mask_encoder' not in k]
        mask_enc_missing = [k for k in missing if 'mask_encoder' in k]
        if mask_enc_missing:
            print(f"  mask_encoder absent in checkpoint (zero-init, no-op) — OK.")
        if contact_missing:
            print(f"ERROR: Contact keys MISSING from checkpoint:")
            for k in contact_missing:
                print(f"  {k}")
            raise RuntimeError(
                f"{len(contact_missing)} contact keys missing from checkpoint. "
                f"Architecture mismatch?"
            )
        if missing:
            print(f"  {len(missing)} non-contact keys missing (expected for frozen backbone).")

        self.step2_mode = ckpt_cfg.TRAIN.get('STEP2_MODE', False)
        print(f"{'Step 2' if self.step2_mode else 'Step 1'} checkpoint detected.")

        self.model.eval()

        # Geodesic distance matrix for SMPL (6890 × 6890)
        geo_dist_path = Path(__file__).parent.parent / "data" / "smpl_neutral_geodesic_dist.npy"
        if geo_dist_path.exists():
            print(f"Loading SMPL geodesic distance matrix: {geo_dist_path}")
            self.geo_dist_matrix = torch.from_numpy(np.load(geo_dist_path)).float()
        else:
            print(f"WARNING: Geodesic distance matrix not found at {geo_dist_path}. "
                  f"Geodesic metrics will be skipped.")
            self.geo_dist_matrix = None

        # Part names (for reporting)
        smpl_part_seg = self.cfg.DATASET.get('SMPL_PART_SEG_PATH', None)
        self._part_names = None
        if smpl_part_seg:
            from damon_utils import load_smpl_part_segmentation
            self._part_names, _ = load_smpl_part_segmentation(smpl_part_seg)

        # Dataset
        print(f"Loading {split} dataset...")
        self._setup_dataset()
        self.instance_loader = None
        if self.masks_v2_dir and self.step2_mode:
            print(f"Loading instance_all dataset from {self.masks_v2_dir}...")
            self._setup_instance_dataset()

    # ------------------------------------------------------------------

    def _setup_dataset(self):
        data_root  = self.cfg.DATASET.get('DATA_ROOT', None)
        val_ratio  = self.cfg.DATASET.get('VAL_RATIO', 0.2)
        seed       = self.cfg.DATASET.get('SEED', 42)
        contact_npz = self.cfg.DATASET.CONTACT_NPZ
        detect_npz  = self.cfg.DATASET.get('DETECT_NPZ', {})
        smpl_part_seg = self.cfg.DATASET.get('SMPL_PART_SEG_PATH', None)

        kwargs = dict(
            topology='smpl',
            smpl_part_seg_path=smpl_part_seg,
            data_root=data_root,
        )
        if self.split == 'test':
            dataset = DamonDataset(
                contact_npz_path=contact_npz.TEST,
                detect_npz_path=detect_npz.get('TEST', None),
                **kwargs,
            )
        else:
            train_ds, val_ds = DamonDataset.split_train_val(
                contact_npz_path=contact_npz.TRAINVAL,
                detect_npz_path=detect_npz.get('TRAINVAL', None),
                val_ratio=val_ratio, seed=seed, **kwargs,
            )
            dataset = train_ds if self.split == 'train' else val_ds

        print(f"  {self.split.capitalize()} samples: {len(dataset)}")
        self.loader = DataLoader(
            dataset,
            batch_size=self.cfg.TRAIN.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=False,
            collate_fn=damon_collate,
        )

    def _setup_instance_dataset(self):
        """Build instance_all DataLoader for mask-conditioned evaluation."""
        data_root  = self.cfg.DATASET.get('DATA_ROOT', None)
        val_ratio  = self.cfg.DATASET.get('VAL_RATIO', 0.2)
        seed       = self.cfg.DATASET.get('SEED', 42)
        contact_npz = self.cfg.DATASET.CONTACT_NPZ
        detect_npz  = self.cfg.DATASET.get('DETECT_NPZ', {})
        smpl_part_seg = self.cfg.DATASET.get('SMPL_PART_SEG_PATH', None)

        common_kwargs = dict(
            topology='smpl',
            smpl_part_seg_path=smpl_part_seg,
            mode='instance_all',
            masks_v2_dir=self.masks_v2_dir,
            data_root=data_root,
        )
        if self.split == 'test':
            dataset = DamonDataset(
                contact_npz_path=contact_npz.TEST,
                detect_npz_path=detect_npz.get('TEST', None),
                **common_kwargs,
            )
        else:
            train_ds, val_ds = DamonDataset.split_train_val(
                contact_npz_path=contact_npz.TRAINVAL,
                detect_npz_path=detect_npz.get('TRAINVAL', None),
                val_ratio=val_ratio, seed=seed,
                **common_kwargs,
            )
            dataset = train_ds if self.split == 'train' else val_ds

        print(f"  Instance samples ({self.split}): {len(dataset)}")
        self.instance_loader = DataLoader(
            dataset,
            batch_size=self.cfg.TRAIN.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=False,
            collate_fn=_instance_eval_collate,
        )

    # ------------------------------------------------------------------

    def _prepare_batch(self, images, bboxes, cam_ks):
        return prepare_damon_batch(
            images, bboxes, cam_ks,
            target_size=tuple(self.cfg.MODEL.IMAGE_SIZE),
            device=self.device,
        )

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_eval_loop(self, loader, desc: str, use_mask: bool = False):
        """
        Core evaluation loop.

        Returns:
            (vertex_probs [N,V], vertex_gt [N,V],
             part_probs_or_None [N,24], part_gt_or_None [N,24])
        """
        all_probs, all_gt = [], []
        all_part_probs, all_part_gt = [], []
        has_parts = False

        for batch_data in tqdm(loader, desc=desc):
            if use_mask:
                images, bboxes, cam_ks, vertex_labels, part_labels, person_masks, obj_masks = batch_data
                person_mask = person_masks.to(self.device)
                object_mask = obj_masks.to(self.device)
            else:
                images, bboxes, cam_ks, vertex_labels, part_labels = batch_data
                person_mask = None
                object_mask = None

            batch = self._prepare_batch(images, bboxes, cam_ks)
            self.model._initialize_batch(batch)
            output = self.model.forward_step(batch, decoder_type="body")

            tokens = self.model.interaction_decoder(
                output["image_embeddings"],
                output["body_tokens"],
                person_mask=person_mask,
                object_mask=object_mask,
            )
            part_tokens  = tokens[:, :InteractionDecoder.NUM_PART_TOKENS, :]  # [B, 24, d]
            vertex_token = tokens[:, InteractionDecoder.NUM_PART_TOKENS:,  :]  # [B, 1,  d]

            vertex_logits = self.model.head_contact(vertex_token)        # [B, 6890]
            part_logits   = self.model.head_part_contact(part_tokens)    # [B, 24]

            all_probs.append(torch.sigmoid(vertex_logits).cpu().float().numpy())
            all_gt.append(vertex_labels.numpy())

            if part_labels is not None:
                all_part_probs.append(torch.sigmoid(part_logits).cpu().float().numpy())
                all_part_gt.append(part_labels.numpy())
                has_parts = True

        all_probs = np.concatenate(all_probs, axis=0)
        all_gt    = np.concatenate(all_gt,    axis=0).astype(bool)

        if has_parts:
            part_probs = np.concatenate(all_part_probs, axis=0)
            part_gt    = np.concatenate(all_part_gt,    axis=0).astype(bool)
        else:
            part_probs = part_gt = None

        return all_probs, all_gt, part_probs, part_gt

    # ------------------------------------------------------------------

    def _finalize_metrics(self, all_probs, all_gt, part_probs, part_gt,
                          threshold: float, tag: str = ""):
        """Compute + print + save metrics. Predictions are already in SMPL (6890) space."""
        all_preds = all_probs > threshold
        metrics   = self._print_metrics(all_preds, all_gt, all_probs, threshold,
                                        part_probs, part_gt)
        self._plot_iou_histogram(all_preds, all_gt)
        self._plot_prob_distribution(all_probs, all_gt)
        roc_auc   = self._plot_roc_curve(all_probs, all_gt)
        self._save_results(metrics, roc_auc, threshold, tag=tag)
        return all_probs, all_preds, all_gt

    @torch.no_grad()
    def evaluate(self, threshold: float = 0.5):
        # --- Classic (no mask) evaluation ---
        print("\n=== Evaluation: no object mask (classic) ===")
        all_probs, all_gt, part_probs, part_gt = self._run_eval_loop(
            self.loader, desc="Evaluating (no mask)"
        )
        result = self._finalize_metrics(all_probs, all_gt, part_probs, part_gt, threshold, tag="")

        # --- Mask-conditioned evaluation (Step 2) ---
        if self.instance_loader is not None:
            print("\n=== Evaluation: with object mask (instance_all) ===")
            all_probs_m, all_gt_m, part_probs_m, part_gt_m = self._run_eval_loop(
                self.instance_loader,
                desc="Evaluating (with mask)",
                use_mask=True,
            )
            self._finalize_metrics(all_probs_m, all_gt_m, part_probs_m, part_gt_m,
                                   threshold, tag="with_mask")

        return result

    # ------------------------------------------------------------------
    # Metric reporting
    # ------------------------------------------------------------------

    def _compute_part_metrics(self, part_probs: np.ndarray, part_gt: np.ndarray,
                               threshold: float = 0.5) -> dict:
        """Per-part accuracy + mean accuracy across 24 SMPL body parts."""
        part_preds = part_probs > threshold   # [N, 24] bool
        correct    = (part_preds == part_gt)  # [N, 24] bool
        per_part_acc = correct.mean(axis=0)   # [24] float

        result = {"mean_part_acc": float(per_part_acc.mean())}
        part_names = self._part_names or [str(i) for i in range(part_gt.shape[1])]
        for i, name in enumerate(part_names):
            result[f"part_acc_{name}"] = float(per_part_acc[i])
        return result

    def _print_metrics(self, preds, gt, probs, threshold: float = 0.5,
                       part_probs=None, part_gt=None) -> dict:
        # ------------------------------------------------------------------
        # Global pooled metrics
        # ------------------------------------------------------------------
        tp_g = (preds & gt).sum()
        fp_g = (preds & ~gt).sum()
        fn_g = (~preds & gt).sum()
        tn_g = (~preds & ~gt).sum()

        accuracy        = (tp_g + tn_g) / (tp_g + tn_g + fp_g + fn_g + 1e-10)
        global_precision = tp_g / (tp_g + fp_g + 1e-10)
        global_recall    = tp_g / (tp_g + fn_g + 1e-10)
        global_f1        = 2 * global_precision * global_recall / (global_precision + global_recall + 1e-10)
        global_iou       = tp_g / (tp_g + fp_g + fn_g + 1e-10)

        # ------------------------------------------------------------------
        # Per-sample averaged metrics
        # ------------------------------------------------------------------
        per_tp = (preds & gt).sum(axis=1).astype(float)
        per_fp = (preds & ~gt).sum(axis=1).astype(float)
        per_fn = (~preds & gt).sum(axis=1).astype(float)

        per_precision = per_tp / (per_tp + per_fp + 1e-10)
        per_recall    = per_tp / (per_tp + per_fn + 1e-10)
        per_f1        = 2 * per_precision * per_recall / (per_precision + per_recall + 1e-10)
        per_iou       = per_tp / (per_tp + per_fp + per_fn + 1e-10)

        mean_precision = per_precision.mean()
        mean_recall    = per_recall.mean()
        mean_f1        = per_f1.mean()
        mean_iou       = per_iou.mean()

        # ------------------------------------------------------------------
        # Geodesic distance (SMPL space, 6890 verts)
        # ------------------------------------------------------------------
        geo_results = {}
        if self.geo_dist_matrix is not None:
            fp_geo, fn_geo = self._compute_geo_distance(preds, gt)
            geo_results = {"fp_geo": fp_geo, "fn_geo": fn_geo}

        # ------------------------------------------------------------------
        # Part metrics
        # ------------------------------------------------------------------
        part_metrics = {}
        if part_probs is not None and part_gt is not None:
            part_metrics = self._compute_part_metrics(part_probs, part_gt, threshold)

        # ------------------------------------------------------------------
        # Print
        # ------------------------------------------------------------------
        print("\n" + "=" * 70)
        print(f"EVALUATION RESULTS  [{self.split}]  (SMPL 6890 vertices)")
        print("=" * 70)
        print("  --- Per-sample averaged ---")
        print(f"  F1        : {mean_f1:.4f}")
        print(f"  Precision : {mean_precision:.4f}")
        print(f"  Recall    : {mean_recall:.4f}")
        print(f"  Mean IoU  : {mean_iou:.4f}  (median: {np.median(per_iou):.4f})")
        if geo_results:
            print(f"  FP Geo Dist: {geo_results['fp_geo']:.4f}")
            print(f"  FN Geo Dist: {geo_results['fn_geo']:.4f}")
        print("  --- Global pooled ---")
        print(f"  Accuracy  : {accuracy:.4f}")
        print(f"  Precision : {global_precision:.4f}")
        print(f"  Recall    : {global_recall:.4f}")
        print(f"  F1        : {global_f1:.4f}")
        print(f"  IoU       : {global_iou:.4f}")
        print(f"  GT contact rate  : {gt.mean():.4f}")
        print(f"  Pred contact rate: {preds.mean():.4f}")
        if part_metrics:
            print(f"  --- Part contact (24 SMPL parts) ---")
            print(f"  Mean Part Acc : {part_metrics['mean_part_acc']:.4f}")
            part_names = self._part_names or [str(i) for i in range(24)]
            for i, name in enumerate(part_names):
                acc = part_metrics.get(f"part_acc_{name}", float('nan'))
                print(f"    {name:25s}: {acc:.4f}")
        print("=" * 70)

        result = {
            "mean_f1":          float(mean_f1),
            "mean_precision":   float(mean_precision),
            "mean_recall":      float(mean_recall),
            "mean_iou":         float(mean_iou),
            "median_iou":       float(np.median(per_iou)),
            **{k: float(v) for k, v in geo_results.items()},
            "global_accuracy":  float(accuracy),
            "global_precision": float(global_precision),
            "global_recall":    float(global_recall),
            "global_f1":        float(global_f1),
            "global_iou":       float(global_iou),
            "gt_contact_rate":  float(gt.mean()),
            "pred_contact_rate": float(preds.mean()),
            "num_samples":      int(preds.shape[0]),
            **part_metrics,
        }
        return result

    # ------------------------------------------------------------------

    def _compute_geo_distance(self, preds: np.ndarray, gt: np.ndarray):
        """Mean geodesic distance between predicted and GT contact vertices (SMPL)."""
        dist_matrix = self.geo_dist_matrix  # [6890, 6890] CPU
        N = preds.shape[0]
        fp_dists = np.zeros(N, dtype=float)
        fn_dists = np.zeros(N, dtype=float)

        for b in range(N):
            gt_mask   = torch.from_numpy(gt[b].astype(bool))
            pred_mask = torch.from_numpy(preds[b].astype(bool))

            gt_cols  = dist_matrix[:, gt_mask]   if gt_mask.any()   else dist_matrix
            err_mat  = gt_cols[pred_mask, :]      if pred_mask.any() else gt_cols

            fp_dists[b] = err_mat.min(dim=1).values.mean().item()
            fn_dists[b] = err_mat.min(dim=0).values.mean().item()

        return float(fp_dists.mean()), float(fn_dists.mean())

    def _save_results(self, metrics: dict, roc_auc: float, threshold: float, tag: str = ""):
        out_path = self.checkpoint_path.parent / "eval_results.json"
        if out_path.exists():
            with open(out_path) as f:
                all_results = json.load(f)
        else:
            all_results = {}

        result_key = f"{self.split}{'_' + tag if tag else ''}"
        all_results[result_key] = {
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
            "checkpoint": str(self.checkpoint_path),
            "tag":        tag or "no_mask",
            "threshold":  threshold,
            **metrics,
            "roc_auc": roc_auc,
        }
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved → {out_path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _plot_iou_histogram(self, preds, gt):
        tp  = (preds & gt).sum(axis=1).astype(float)
        fp  = (preds & ~gt).sum(axis=1).astype(float)
        fn  = (~preds & gt).sum(axis=1).astype(float)
        iou = tp / (tp + fp + fn + 1e-8)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(iou, bins=50, color='steelblue', edgecolor='white')
        ax.axvline(iou.mean(), color='red', linestyle='--', label=f'Mean={iou.mean():.3f}')
        ax.set_xlabel('Per-sample IoU')
        ax.set_ylabel('Count')
        ax.set_title(f'Per-sample IoU distribution [{self.split}]')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = self.figures_dir / f"{self.split}_iou_histogram.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved IoU histogram: {out}")

    def _plot_prob_distribution(self, probs, gt):
        flat_probs = probs.ravel()
        flat_gt    = gt.ravel()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(flat_probs[~flat_gt], bins=100, alpha=0.5, label='No contact',
                color='royalblue', density=True)
        ax.hist(flat_probs[flat_gt],  bins=100, alpha=0.5, label='Contact',
                color='crimson', density=True)
        ax.axvline(0.5, color='black', linestyle='--', label='Threshold=0.5')
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Density')
        ax.set_title(f'Probability distribution [{self.split}]')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = self.figures_dir / f"{self.split}_prob_distribution.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved probability distribution: {out}")

    def _plot_roc_curve(self, probs, gt):
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(gt.ravel().astype(int), probs.ravel())
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve [{self.split}]')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = self.figures_dir / f"{self.split}_roc_curve.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved ROC curve: {out}  (AUC = {roc_auc:.4f})")
        return roc_auc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate per-vertex contact head on DAMON")
    parser.add_argument("--config",     type=str, default="configs/step1_contact.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint")
    parser.add_argument("--split",      type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="Binary threshold for contact probability")
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--masks_v2_dir", type=str, default=None,
                        help="Path to masks_v2 directory for Step 2 mask-conditioned evaluation.")
    args = parser.parse_args()

    evaluator = ContactEvaluator(
        args.config, args.checkpoint,
        split=args.split, device=args.device,
        masks_v2_dir=args.masks_v2_dir,
    )
    evaluator.evaluate(threshold=args.threshold)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
