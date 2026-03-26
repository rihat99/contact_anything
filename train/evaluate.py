"""
Evaluation script for the per-vertex Contact Head on DAMON dataset.

Evaluates a trained checkpoint on any of the three splits (train / val / test)
and reports:
  - Overall per-vertex accuracy, precision, recall, F1, IoU
  - Per-sample IoU histogram
  - Probability distribution (contact vs no-contact)
  - ROC curve + AUC

Usage:
    python train/evaluate.py \
        --config train/config.yaml \
        --checkpoint train/output/contact_vert_20260222_123456/best_model.pth \
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
sys.path.insert(0, str(Path(__file__).parent.parent / "mhr_smpl_conversion"))
os.environ["MOMENTUM_ENABLED"] = "1"

from sam_3d_body.build_models import load_sam_3d_body
from sam_3d_body.models.heads.contact_head import ContactHead
from sam_3d_body.models.decoders.interaction_decoder import InteractionDecoder
from sam_3d_body.utils.config import get_config
from damon_dataset import DamonDataset
from dataset_utils import prepare_damon_batch
from train_contact import damon_collate


# ---------------------------------------------------------------------------

class ContactEvaluator:
    """Evaluates per-vertex contact prediction on DAMON dataset."""

    def __init__(self, config_path: str, checkpoint_path: str,
                 split: str = "val", device: str = "cuda", mode: str = "smpl"):
        self.device = device
        self.split = split
        self.mode = mode  # "smpl" or "mhr"
        self.checkpoint_path = Path(checkpoint_path)

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

        # Reinitialize contact modules from the checkpoint's saved config.yaml.
        # Each training run saves config.yaml alongside the checkpoint.
        import yaml

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
            num_contact_tokens=id_cfg.get('NUM_CONTACT_TOKENS', 16),
            num_layers=id_cfg.get('NUM_LAYERS', 4),
            num_heads=id_cfg.get('NUM_HEADS', 8),
            ffn_dim=id_cfg.get('FFN_DIM', 2048),
            dropout=id_cfg.get('DROPOUT', 0.0),
        ).to(device)
        num_vertices = ch_cfg.get('NUM_VERTICES', 18439)
        self.model.head_contact = ContactHead(
            input_dim=id_cfg.get('D_MODEL', dim),
            num_contact_tokens=id_cfg.get('NUM_CONTACT_TOKENS', 16),
            num_vertices=num_vertices,
            mlp_depth=ch_cfg.get('MLP_DEPTH', 2),
            mlp_channel_div_factor=ch_cfg.get('MLP_CHANNEL_DIV_FACTOR', 4),
            pool_mode=ch_cfg.get('POOL_MODE', 'attention'),
            dropout=ch_cfg.get('DROPOUT', 0.0),
        ).to(device)
        print(f"InteractionDecoder: K={id_cfg.get('NUM_CONTACT_TOKENS', 16)} tokens, "
              f"{id_cfg.get('NUM_LAYERS', 4)} layers")

        # Load trained checkpoint (contact head weights override the fresh init above)
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        missing, unexpected = self.model.load_state_dict(state, strict=False)

        # Sanity check: interaction_decoder / head_contact keys must not be missing
        contact_missing = [k for k in missing
                           if 'interaction_decoder' in k or 'head_contact' in k]
        if contact_missing:
            print(f"ERROR: Contact head keys MISSING from checkpoint (architecture mismatch?):")
            for k in contact_missing:
                print(f"  {k}")
            raise RuntimeError(
                f"{len(contact_missing)} contact head keys missing from checkpoint. "
                f"Check that the checkpoint's config.yaml matches the contact head architecture."
            )
        if missing:
            print(f"  {len(missing)} non-contact keys missing (expected for frozen backbone).")

        self.model.eval()

        self.lod = self.cfg.DATASET.get('LOD', 1)

        # Contact converter (SMPL mode only — fast, CPU is fine)
        if self.mode == "smpl":
            from body_converter import BodyConverter
            print("Initialising MHR→SMPL contact converter...")
            self.converter = BodyConverter(device="cpu")

            # Geodesic distance matrix for SMPL (6890 × 6890)
            geo_dist_path = Path(__file__).parent.parent / "data" / "smpl_neutral_geodesic_dist.npy"
            print(f"Loading SMPL geodesic distance matrix: {geo_dist_path}")
            self.geo_dist_matrix = torch.from_numpy(np.load(geo_dist_path)).float()  # CPU

        # Dataset
        print(f"Loading {split} dataset...")
        self._setup_dataset()

    # ------------------------------------------------------------------

    def _setup_dataset(self):
        data_root = self.cfg.DATASET.get('DATA_ROOT', None)
        val_ratio = self.cfg.DATASET.get('VAL_RATIO', 0.2)
        seed      = self.cfg.DATASET.get('SEED', 42)

        lod         = self.cfg.DATASET.get('LOD', 1)
        contact_npz = self.cfg.DATASET.CONTACT_NPZ
        detect_npz  = self.cfg.DATASET.get('DETECT_NPZ', {})

        if self.mode == "smpl":
            # Load GT contacts from original DECO SMPL NPZ
            smpl_trainval = self.cfg.DATASET.get('SMPL_TRAINVAL_NPZ', None)
            smpl_test     = self.cfg.DATASET.get('SMPL_TEST_NPZ', None)

            if self.split == 'test':
                if not smpl_test:
                    raise ValueError("DATASET.SMPL_TEST_NPZ not set in config.")
                dataset = DamonDataset(
                    contact_npz_path=smpl_test,
                    detect_npz_path=detect_npz.get('TEST', None),
                    topology='smpl',
                    data_root=data_root,
                )
            else:
                if not smpl_trainval:
                    raise ValueError("DATASET.SMPL_TRAINVAL_NPZ not set in config.")
                train_ds, val_ds = DamonDataset.split_train_val(
                    contact_npz_path=smpl_trainval,
                    detect_npz_path=detect_npz.get('TRAINVAL', None),
                    topology='smpl',
                    val_ratio=val_ratio,
                    seed=seed,
                    data_root=data_root,
                )
                dataset = train_ds if self.split == 'train' else val_ds
        else:
            # MHR mode
            if self.split == 'test':
                if not contact_npz.get('TEST', None):
                    raise ValueError("DATASET.CONTACT_NPZ.TEST not set in config.")
                dataset = DamonDataset(
                    contact_npz_path=contact_npz.TEST,
                    detect_npz_path=detect_npz.get('TEST', None),
                    topology='mhr',
                    lod=lod,
                    data_root=data_root,
                )
            else:
                # Reproduce the exact same train/val split used during training.
                train_ds, val_ds = DamonDataset.split_train_val(
                    contact_npz_path=contact_npz.TRAINVAL,
                    detect_npz_path=detect_npz.get('TRAINVAL', None),
                    topology='mhr',
                    lod=lod,
                    val_ratio=val_ratio,
                    seed=seed,
                    data_root=data_root,
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

    # ------------------------------------------------------------------

    def _prepare_batch(self, images, bboxes, cam_ks):
        return prepare_damon_batch(
            images, bboxes, cam_ks,
            target_size=tuple(self.cfg.MODEL.IMAGE_SIZE),
            device=self.device,
        )

    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, threshold: float = 0.5):
        all_probs = []   # [N, V_lod]
        all_gt = []      # [N, V_mhr or V_smpl]

        for images, bboxes, cam_ks, contact_labels in tqdm(self.loader, desc="Evaluating"):
            batch = self._prepare_batch(images, bboxes, cam_ks)
            self.model._initialize_batch(batch)
            output = self.model.forward_step(batch, decoder_type="body")
            contact_tokens = self.model.interaction_decoder(
                output["image_embeddings"],
                output["body_tokens"],
            )
            logits = self.model.head_contact(contact_tokens)
            probs = torch.sigmoid(logits).cpu().float()

            all_probs.append(probs.numpy())
            all_gt.append(contact_labels.numpy())

        all_probs = np.concatenate(all_probs, axis=0)            # [N, V_lod]
        all_gt    = np.concatenate(all_gt, axis=0).astype(bool)  # [N, V_smpl or V_lod]

        if self.mode == "smpl":
            # Convert LOD_N predictions → SMPL space.
            # For LOD1: direct barycentric interpolation (exact).
            # For LOD_N (N≠1): Voronoi BFS upsample LOD_N → LOD1, then LOD1 → SMPL.
            print(f"Converting LOD{self.lod} predictions to SMPL space...")
            smpl_probs = self._interpolate_probs_to_smpl(all_probs)  # [N, 6890]
            all_preds  = smpl_probs > threshold
            metrics    = self._print_metrics(all_preds, all_gt, smpl_probs, threshold)
            self._plot_iou_histogram(all_preds, all_gt)
            self._plot_prob_distribution(smpl_probs, all_gt)
            roc_auc    = self._plot_roc_curve(smpl_probs, all_gt)
            self._save_results(metrics, roc_auc, threshold)
            return smpl_probs, all_preds, all_gt
        else:
            all_preds = all_probs > threshold    # [N, V_lod]
            metrics = self._print_metrics(all_preds, all_gt, all_probs, threshold)
            self._plot_iou_histogram(all_preds, all_gt)
            self._plot_prob_distribution(all_probs, all_gt)
            roc_auc = self._plot_roc_curve(all_probs, all_gt)
            self._save_results(metrics, roc_auc, threshold)
            return all_probs, all_preds, all_gt

    def _smpl_gt_to_lod(self, smpl_gt: np.ndarray) -> np.ndarray:
        """
        Convert SMPL GT labels [N, 6890] → LOD_N [N, V_lod] binary.

        Uses the same smpl_to_mhr forward chain as convert_damon.py, so the
        result is consistent with how the training labels were generated.
        Requires SMPL face connectivity (loaded from SMPL_MODEL_PATH).
        """
        smpl_model_path = self.cfg.MODEL.get('SMPL_MODEL_PATH', None)
        if smpl_model_path is None:
            raise ValueError(
                "MODEL.SMPL_MODEL_PATH must be set in config to convert SMPL GT to LOD."
            )
        smpl_npz   = np.load(smpl_model_path, allow_pickle=True)
        smpl_faces = smpl_npz['f'].astype(np.int32)
        conv = BodyConverter(smpl_faces=smpl_faces, device='cpu')

        gt_t = torch.from_numpy(smpl_gt.astype(np.float32))  # [N, 6890]
        result = conv.smpl_to_mhr(contacts=gt_t, target_lod=self.lod, threshold=0.5)
        return result.contacts.numpy().astype(bool)            # [N, V_lod]

    def _interpolate_probs_to_smpl(self, mhr_probs: np.ndarray) -> np.ndarray:
        """
        Convert MHR LOD_N probabilities [N, V_N] → SMPL [N, 6890] as floats.

        For LOD1: direct barycentric interpolation LOD1 → SMPL.
        For other LODs: scatter LOD_N → LOD1 first, then interpolate to SMPL.
        Preserves float values (no thresholding) for ROC / prob-distribution plots.
        """
        probs_t = torch.from_numpy(mhr_probs.astype(np.float32))
        smpl_probs = self.converter.mhr_lod_probs_to_smpl_probs(probs_t, source_lod=self.lod)
        return smpl_probs.numpy()

    # ------------------------------------------------------------------
    # Metric reporting
    # ------------------------------------------------------------------

    def _print_metrics(self, preds, gt, probs, threshold: float = 0.5) -> dict:
        # ------------------------------------------------------------------
        # Global pooled metrics (aggregate TP/FP/FN over all samples)
        # ------------------------------------------------------------------
        tp_g = (preds & gt).sum()
        fp_g = (preds & ~gt).sum()
        fn_g = (~preds & gt).sum()
        tn_g = (~preds & ~gt).sum()

        accuracy       = (tp_g + tn_g) / (tp_g + tn_g + fp_g + fn_g + 1e-10)
        global_precision = tp_g / (tp_g + fp_g + 1e-10)
        global_recall    = tp_g / (tp_g + fn_g + 1e-10)
        global_f1        = 2 * global_precision * global_recall / (global_precision + global_recall + 1e-10)
        global_iou       = tp_g / (tp_g + fp_g + fn_g + 1e-10)

        # ------------------------------------------------------------------
        # Per-sample averaged metrics (matches InteractVLM get_h_contact_metrics)
        # Each sample contributes equally regardless of contact region size.
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
        # Geodesic distance (SMPL only)
        # ------------------------------------------------------------------
        geo_results = {}
        if self.mode == "smpl":
            # Geodesic distance only available when evaluating in SMPL (6890) space.
            fp_geo, fn_geo = self._compute_geo_distance(preds, gt, threshold)
            geo_results = {"fp_geo": fp_geo, "fn_geo": fn_geo}
        else:
            print("  (Geodesic distance skipped — not in SMPL evaluation space)")

        # ------------------------------------------------------------------
        # Print
        # ------------------------------------------------------------------
        mode_label = (
            f"smpl→lod{self.lod}" if self.mode == "_lod_from_smpl"
            else self.mode
        )
        print("\n" + "=" * 70)
        print(f"EVALUATION RESULTS  [{self.split}]  (mode={mode_label}, lod={self.lod})")
        print("=" * 70)
        print(f"  --- Per-sample averaged (matches InteractVLM) ---")
        print(f"  F1        : {mean_f1:.4f}")
        print(f"  Precision : {mean_precision:.4f}")
        print(f"  Recall    : {mean_recall:.4f}")
        print(f"  Mean IoU  : {mean_iou:.4f}  (median: {np.median(per_iou):.4f})")
        if geo_results:
            print(f"  FP Geo Dist: {geo_results['fp_geo']:.4f}")
            print(f"  FN Geo Dist: {geo_results['fn_geo']:.4f}")
        print(f"  --- Global pooled ---")
        print(f"  Accuracy  : {accuracy:.4f}")
        print(f"  Precision : {global_precision:.4f}")
        print(f"  Recall    : {global_recall:.4f}")
        print(f"  F1        : {global_f1:.4f}")
        print(f"  IoU       : {global_iou:.4f}")
        print(f"  GT contact rate  : {gt.mean():.4f}")
        print(f"  Pred contact rate: {preds.mean():.4f}")
        print("=" * 70)

        result = {
            # Per-sample averaged (primary — matches InteractVLM)
            "mean_f1":               float(mean_f1),
            "mean_precision":        float(mean_precision),
            "mean_recall":           float(mean_recall),
            "mean_iou":              float(mean_iou),
            "median_iou":            float(np.median(per_iou)),
            # Geodesic distance (SMPL only)
            **{k: float(v) for k, v in geo_results.items()},
            # Global pooled (for reference)
            "global_accuracy":       float(accuracy),
            "global_precision":      float(global_precision),
            "global_recall":         float(global_recall),
            "global_f1":             float(global_f1),
            "global_iou":            float(global_iou),
            # Meta
            "gt_contact_rate":       float(gt.mean()),
            "pred_contact_rate":     float(preds.mean()),
            "num_samples":           int(preds.shape[0]),
        }
        return result

    # ------------------------------------------------------------------

    def _compute_geo_distance(self, preds: np.ndarray, gt: np.ndarray,
                               threshold: float = 0.5):
        """
        Compute mean geodesic distance between predicted and GT contact vertices.

        Matches InteractVLM's get_h_geo_metric():
          - fp_geo: for each predicted contact vertex, min dist to nearest GT contact
                    vertex, averaged → how far off false positives are
          - fn_geo: for each GT contact vertex, min dist to nearest predicted contact
                    vertex, averaged → how far away missed contacts are

        Args:
            preds: bool array [N, 6890]
            gt:    bool array [N, 6890]
            threshold: not used (preds already thresholded), kept for API symmetry

        Returns:
            fp_geo, fn_geo  (scalars, averaged over N samples)
        """
        dist_matrix = self.geo_dist_matrix  # [6890, 6890] float, CPU torch tensor
        N = preds.shape[0]
        fp_dists = np.zeros(N, dtype=float)
        fn_dists = np.zeros(N, dtype=float)

        for b in range(N):
            gt_mask   = torch.from_numpy(gt[b].astype(bool))    # [6890]
            pred_mask = torch.from_numpy(preds[b].astype(bool)) # [6890]

            # Columns = GT contact vertices; fallback to full matrix if no GT contacts
            gt_cols = dist_matrix[:, gt_mask] if gt_mask.any() else dist_matrix

            # Rows = predicted contact vertices; fallback to gt_cols if no predictions
            err_mat = gt_cols[pred_mask, :] if pred_mask.any() else gt_cols

            # FP geo: avg min dist from each predicted contact to nearest GT contact
            fp_dists[b] = err_mat.min(dim=1).values.mean().item()
            # FN geo: avg min dist from each GT contact to nearest predicted contact
            fn_dists[b] = err_mat.min(dim=0).values.mean().item()

        return float(fp_dists.mean()), float(fn_dists.mean())

    def _save_results(self, metrics: dict, roc_auc: float, threshold: float):
        """Append/update eval_results.json in the checkpoint directory."""
        out_path = self.checkpoint_path.parent / "eval_results.json"

        # Load existing results (other splits may already be present)
        if out_path.exists():
            with open(out_path) as f:
                all_results = json.load(f)
        else:
            all_results = {}

        all_results[self.split] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "checkpoint": str(self.checkpoint_path),
            "mode": self.mode,
            "threshold": threshold,
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
        tp = (preds & gt).sum(axis=1).astype(float)
        fp = (preds & ~gt).sum(axis=1).astype(float)
        fn = (~preds & gt).sum(axis=1).astype(float)
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
        flat_gt = gt.ravel()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(flat_probs[~flat_gt], bins=100, alpha=0.5, label='No contact', color='royalblue',
                density=True)
        ax.hist(flat_probs[flat_gt], bins=100, alpha=0.5, label='Contact', color='crimson',
                density=True)
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

        flat_probs = probs.ravel()
        flat_gt = gt.ravel().astype(int)

        fpr, tpr, _ = roc_curve(flat_gt, flat_probs)
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
    parser.add_argument("--config", type=str, default="configs/step1_contact.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Binary threshold for contact probability")
    parser.add_argument("--mode", type=str, default="smpl",
                        choices=["smpl", "mhr"],
                        help="Mesh topology for metrics: 'smpl' (6890 verts, default) "
                             "or 'mhr' (LOD vertex count from config)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    evaluator = ContactEvaluator(
        args.config, args.checkpoint,
        split=args.split, device=args.device, mode=args.mode,
    )
    evaluator.evaluate(threshold=args.threshold)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
