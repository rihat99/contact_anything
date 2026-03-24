"""
Training script for per-vertex Contact Head on DAMON dataset.

Trains only the contact head (+ contact tokens + update layers) while keeping
the rest of SAM-3D-Body frozen.  Each run writes to a date-time-stamped folder
under OUTPUT.DIR so experiments are easy to track.
"""

import faulthandler
faulthandler.enable()  # print C-level stack trace on segfault

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["MOMENTUM_ENABLED"] = "1"

from sam_3d_body.build_models import load_sam_3d_body
from sam_3d_body.models.heads.contact_head import ContactHead
from sam_3d_body.utils.config import get_config

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "dataset"))
sys.path.insert(0, str(Path(__file__).parent.parent / "mhr_smpl_conversion"))
from damon_mhr import DamonMHRDataset, DamonPrecomputedDataset
from dataset_utils import prepare_damon_batch, prepare_damon_batch_precomputed
from body_converter import BodyConverter
from losses import ContactLoss


# ---------------------------------------------------------------------------
# Custom collate: images are numpy arrays of varying shape — keep as list
# ---------------------------------------------------------------------------

def damon_collate(batch):
    """
    batch: list of ((image, bbox, cam_k), contact_label)
    Returns:
        images    — list of B numpy arrays
        bboxes    — tensor [B, 4]
        cam_ks    — tensor [B, 3, 3]
        contact_labels — tensor [B, num_vertices]
    """
    images, bboxes, cam_ks, contact_labels = [], [], [], []
    for (img, bbox, cam_k), lbl in batch:
        images.append(img)
        bboxes.append(bbox)
        cam_ks.append(cam_k)
        contact_labels.append(lbl)
    return (
        images,
        torch.stack(bboxes),
        torch.stack(cam_ks),
        torch.stack(contact_labels),
    )


def damon_precomputed_collate(batch):
    """
    batch: list of ((feature, bbox, cam_k, ori_img_size, pred_kp2d, pred_kp3d), contact_label)
    Returns:
        features       — tensor [B, C, H, W]
        bboxes         — tensor [B, 4]
        cam_ks         — tensor [B, 3, 3]
        ori_img_sizes  — tensor [B, 2]
        pred_kp2d      — tensor [B, 70, 2]
        pred_kp3d      — tensor [B, 70, 3]
        contact_labels — tensor [B, num_vertices]
    """
    features, bboxes, cam_ks, ori_sizes = [], [], [], []
    pred_kp2ds, pred_kp3ds, contact_labels = [], [], []
    for (feat, bbox, cam_k, ori_size, kp2d, kp3d), lbl in batch:
        features.append(feat)
        bboxes.append(bbox)
        cam_ks.append(cam_k)
        ori_sizes.append(ori_size)
        pred_kp2ds.append(kp2d)
        pred_kp3ds.append(kp3d)
        contact_labels.append(lbl)
    return (
        torch.stack(features),
        torch.stack(bboxes),
        torch.stack(cam_ks),
        torch.stack(ori_sizes),
        torch.stack(pred_kp2ds),
        torch.stack(pred_kp3ds),
        torch.stack(contact_labels),
    )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ContactTrainer:
    """Trains per-vertex contact head on DAMON dataset."""

    def __init__(self, config_path: str, device: str = "cuda"):
        self.cfg = get_config(config_path)
        self.device = device

        # ---- Output directory: base/expname_YYYYMMDD_HHMMSS ----
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.cfg.OUTPUT.EXP_NAME}_{timestamp}"
        self.output_dir = Path(self.cfg.OUTPUT.DIR) / exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

        # Save config
        with open(self.output_dir / "config.yaml", "w") as f:
            f.write(str(self.cfg))

        # TensorBoard
        if self.cfg.OUTPUT.USE_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        else:
            self.writer = None

        # ---- Load model ----
        print("Loading SAM-3D-Body model...")
        self.model, self.model_cfg = load_sam_3d_body(
            checkpoint_path=self.cfg.MODEL.CHECKPOINT_PATH,
            device=device,
            mhr_path=self.cfg.MODEL.MHR_MODEL_PATH,
        )

        # ---- Reinitialize all contact modules from train config ----
        # load_sam_3d_body reads the checkpoint's model_config.yaml, which may have
        # different (or commented-out) CONTACT_HEAD settings. We always reinitialize
        # contact_embedding and head_contact from train/config.yaml so that the
        # architecture matches regardless of what the checkpoint config says.
        import torch.nn as nn
        train_contact_cfg = self.cfg.MODEL.CONTACT_HEAD
        train_num_vertices = train_contact_cfg.get('NUM_VERTICES', 18439)
        num_kp  = train_contact_cfg.get('NUM_CONTACTS', 21)
        num_gbl = train_contact_cfg.get('NUM_GLOBAL_TOKENS', 0)
        total   = num_kp + num_gbl
        dim     = self.model_cfg.MODEL.DECODER.DIM

        old_tokens = getattr(self.model, 'total_contact_tokens', None)
        old_verts  = getattr(self.model.head_contact, 'num_vertices', None) if hasattr(self.model, 'head_contact') else None

        print(f"Reinitializing contact modules from train config: "
              f"tokens {old_tokens}→{total}, verts {old_verts}→{train_num_vertices}")

        self.model.num_contact_tokens        = num_kp
        self.model.num_global_contact_tokens = num_gbl
        self.model.total_contact_tokens      = total
        self.model.contact_keypoint_indices  = list(range(num_kp))
        self.model.contact_grid_size         = train_contact_cfg.get('GRID_SIZE', 1)
        self.model.contact_grid_radius       = train_contact_cfg.get('GRID_RADIUS', 0.1)
        self.model.contact_embedding         = nn.Embedding(total, dim).to(device)
        self.model.head_contact              = ContactHead(
            input_dim=dim,
            num_contact_tokens=total,
            num_vertices=train_num_vertices,
            mlp_depth=train_contact_cfg.get('MLP_DEPTH', 2),
            mlp_channel_div_factor=train_contact_cfg.get('MLP_CHANNEL_DIV_FACTOR', 4),
            pool_mode=train_contact_cfg.get('POOL_MODE', 'attention'),
            dropout=train_contact_cfg.get('DROPOUT', 0.0),
        ).to(device)

        # ---- Freeze all params; unfreeze contact-related ones ----
        print("Freezing all parameters except contact head & tokens...")
        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            if "contact" in name.lower():
                param.requires_grad = True
                print(f"  Unfrozen: {name}")

        # ---- LoRA injection (before counting trainable params) ----
        self.use_lora = self.cfg.MODEL.get('LORA', {}).get('ENABLED', False)
        if self.use_lora:
            from sam_3d_body.models.modules.lora import apply_lora_to_decoder
            lora_cfg = self.cfg.MODEL.LORA
            apply_lora_to_decoder(self.model.decoder, lora_cfg)
            # LoRA params are created with requires_grad=True by default
            # Also explicitly ensure they're unfrozen
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True

        # CRITICAL: decoder must stay in train mode so gradients flow to
        # contact query tokens even though decoder weights are frozen.
        for dec in [getattr(self.model, 'decoder', None),
                    getattr(self.model, 'decoder_hand', None)]:
            if dec is not None:
                for m in dec.modules():
                    m.train()

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable: {trainable:,} / {total_params:,}")

        # ---- Datasets ----
        print("Loading datasets...")
        data_root  = self.cfg.DATASET.get('DATA_ROOT', None)
        val_ratio  = self.cfg.DATASET.get('VAL_RATIO', 0.2)
        seed       = self.cfg.DATASET.get('SEED', 42)
        lod        = self.cfg.DATASET.get('LOD', 1)
        contact_npz = self.cfg.DATASET.CONTACT_NPZ
        detect_npz  = self.cfg.DATASET.get('DETECT_NPZ', {})

        self.use_precomputed = self.cfg.TRAIN.get('USE_PRECOMPUTED_FEATURES', False)

        if self.use_precomputed:
            features_base = self.cfg.TRAIN.get('PRECOMPUTED_FEATURES_DIR',
                                               './dataset/damon_mhr_contact/features')
            predictions_base = self.cfg.TRAIN.get('PRECOMPUTED_PREDICTIONS_DIR',
                                                  './dataset/damon_mhr_contact/predictions')
            self.train_dataset, self.val_dataset = DamonPrecomputedDataset.split_train_val(
                contact_npz_path=contact_npz.TRAINVAL,
                detect_npz_path=detect_npz.get('TRAINVAL', None),
                features_dir=os.path.join(features_base, 'trainval'),
                predictions_npz_path=os.path.join(predictions_base, 'trainval_predictions.npz'),
                lod=lod,
                val_ratio=val_ratio,
                seed=seed,
                data_root=data_root,
            )
            collate_fn = damon_precomputed_collate
            print(f"  Using precomputed features from {features_base}")
        else:
            self.train_dataset, self.val_dataset = DamonMHRDataset.split_train_val(
                contact_npz_path=contact_npz.TRAINVAL,
                detect_npz_path=detect_npz.get('TRAINVAL', None),
                lod=lod,
                val_ratio=val_ratio,
                seed=seed,
                data_root=data_root,
            )
            collate_fn = damon_collate

        print(f"  Train: {len(self.train_dataset)}  Val: {len(self.val_dataset)}")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=False,
            drop_last=True,
            collate_fn=collate_fn,
            persistent_workers=self.cfg.TRAIN.NUM_WORKERS > 0,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.TRAIN.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=False,
            collate_fn=collate_fn,
            persistent_workers=self.cfg.TRAIN.NUM_WORKERS > 0,
        )

        # ---- Loss function ----
        loss_cfg = self.cfg.get("LOSS", {})
        self.loss_fn = ContactLoss(
            focal_alpha=loss_cfg.get("FOCAL_ALPHA", 0.25),
            focal_gamma=loss_cfg.get("FOCAL_GAMMA", 2.0),
            focal_weight=loss_cfg.get("FOCAL_WEIGHT", 2.0),
            dice_weight=loss_cfg.get("DICE_WEIGHT", 0.5),
            dice_eps=loss_cfg.get("DICE_EPS", 1e-5),
            sparsity_weight=loss_cfg.get("SPARSITY_WEIGHT", 0.01),
        )
        self._last_loss_dict: dict = {}
        print(
            f"ContactLoss: focal_weight={self.loss_fn.focal_weight}  "
            f"dice_weight={self.loss_fn.dice_weight}  "
            f"sparsity_weight={self.loss_fn.sparsity_weight}  "
            f"alpha={self.loss_fn.focal_alpha}  gamma={self.loss_fn.focal_gamma}"
        )

        # ---- Pose supervision config ----
        pose_sup_cfg = self.cfg.TRAIN.get('POSE_SUPERVISION', {})
        self.use_pose_supervision = pose_sup_cfg.get('ENABLED', False) and self.use_precomputed
        if self.use_pose_supervision:
            self.kp2d_weight = pose_sup_cfg.get('KP2D_WEIGHT', 0.1)
            self.kp3d_weight = pose_sup_cfg.get('KP3D_WEIGHT', 0.1)
            self.pose_loss_type = pose_sup_cfg.get('LOSS_TYPE', 'smooth_l1')
            print(f"Pose supervision: kp2d_weight={self.kp2d_weight}, "
                  f"kp3d_weight={self.kp3d_weight}, loss={self.pose_loss_type}")

        # ---- Optimizer & scheduler ----
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
        )
        self.scheduler = self._setup_scheduler()

        # ---- OOB vertex weighting ----
        self.oob_weight = self.cfg.TRAIN.get('OOB_VERTEX_WEIGHT', 1.0)
        self.lod = self.cfg.DATASET.get('LOD', 1)
        self.body_converter = None
        if self.oob_weight < 1.0:
            print(f"OOB vertex weighting enabled: out-of-bounds weight = {self.oob_weight}")
            if self.lod != 1:
                print(f"  Loading BodyConverter for LOD{self.lod} mapping...")
                self.body_converter = BodyConverter(device=device)
        else:
            print("OOB vertex weighting disabled (OOB_VERTEX_WEIGHT=1.0)")

        # ---- State ----
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        if self.cfg.TRAIN.RESUME:
            self._load_checkpoint(self.cfg.TRAIN.RESUME)

    # ------------------------------------------------------------------
    # Positive-weight computation
    # ------------------------------------------------------------------

    def _compute_pos_weight(self) -> torch.Tensor:
        if self.cfg.TRAIN.POS_WEIGHT is not None:
            pw = torch.tensor(self.cfg.TRAIN.POS_WEIGHT, dtype=torch.float32)
            return pw.to(self.device)

        print("Computing positive class weight from training set...")
        total_pos = 0
        total_neg = 0
        num_vertices = self.cfg.MODEL.CONTACT_HEAD.get('NUM_VERTICES', 18439)

        for _, _, _, contact_labels in tqdm(self.train_loader, desc="pos_weight"):
            pos = contact_labels.sum().item()
            total_pos += pos
            total_neg += contact_labels.numel() - pos

        pos_weight = total_neg / (total_pos + 1e-6)
        print(f"  pos/neg = {total_pos}/{total_neg}  pos_weight = {pos_weight:.2f}")
        return torch.tensor(pos_weight, dtype=torch.float32).to(self.device)

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def _setup_scheduler(self):
        warmup = self.cfg.TRAIN.get("LR_WARMUP_EPOCHS", 0)
        if self.cfg.TRAIN.LR_SCHEDULER == "cosine":
            main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(self.cfg.TRAIN.EPOCHS - warmup, 1),
                eta_min=self.cfg.TRAIN.LR_MIN,
            )
        elif self.cfg.TRAIN.LR_SCHEDULER == "step":
            main_sched = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )
        else:
            return None

        if warmup > 0:
            warm_sched = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.01, total_iters=warmup
            )
            return torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warm_sched, main_sched], milestones=[warmup]
            )
        return main_sched

    # ------------------------------------------------------------------
    # Batch preparation
    # ------------------------------------------------------------------

    def _prepare_batch(self, *args):
        """Prepare batch for model. Accepts either image-based or precomputed-feature args."""
        if self.use_precomputed:
            features, bboxes, cam_ks, ori_img_sizes = args
            # Use the model's own IMAGE_SIZE (from checkpoint config) to match
            # the resolution at which features were precomputed.
            model_img_size = tuple(self.model_cfg.MODEL.IMAGE_SIZE)
            return prepare_damon_batch_precomputed(
                features, bboxes, cam_ks, ori_img_sizes,
                target_size=model_img_size,
                device=self.device,
            )
        else:
            images, bboxes, cam_ks = args
            return prepare_damon_batch(
                images, bboxes, cam_ks,
                target_size=tuple(self.cfg.MODEL.IMAGE_SIZE),
                device=self.device,
            )

    # ------------------------------------------------------------------
    # Loss & metrics
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        vertex_weights: torch.Tensor = None,
    ):
        """
        Combined contact loss: Focal BCE + Dice + L1 Sparsity.

        Args:
            logits:         [B, num_vertices] raw pre-sigmoid contact logits.
            targets:        [B, num_vertices] int64 or float binary ground-truth.
            vertex_weights: [B, num_vertices] float, optional per-vertex weights.
                            OOB vertices should have weight 0.0; visible vertices 1.0.

        Returns scalar loss. Stores per-component values in self._last_loss_dict.
        """
        total_loss, loss_dict = self.loss_fn(logits, targets.float(), vertex_weights)
        self._last_loss_dict = loss_dict
        return total_loss

    @torch.no_grad()
    def _compute_vertex_weights(self, output: dict, batch: dict) -> torch.Tensor:
        """
        Compute per-vertex loss weights based on whether each vertex projects
        inside the original image bounds.

        Returns [B, num_vertices] float tensor where in-bounds vertices = 1.0
        and out-of-bounds vertices = self.oob_weight.
        Returns None if oob_weight == 1.0 (no masking needed).
        """
        if self.oob_weight >= 1.0:
            return None

        mhr_out = output.get("mhr")
        if mhr_out is None:
            return None
        verts_2d = mhr_out.get("pred_keypoints_2d_verts")  # [B, 18439, 2] pixel coords
        if verts_2d is None:
            return None

        verts_2d = verts_2d.detach()  # no gradient through the weight mask

        # ori_img_size is [B, 1, 2] with (H, W) ordering
        ori_img_size = batch["ori_img_size"][:, 0].to(verts_2d.device)  # [B, 2]
        H = ori_img_size[:, 0:1]  # [B, 1]
        W = ori_img_size[:, 1:2]  # [B, 1]

        # Compute LOD1-level visibility: 1.0 = inside image, 0.0 = OOB
        oob_lod1 = (
            (verts_2d[..., 0] < 0)
            | (verts_2d[..., 0] > W)
            | (verts_2d[..., 1] < 0)
            | (verts_2d[..., 1] > H)
        )  # [B, 18439] bool
        visible_lod1 = (~oob_lod1).float()  # [B, 18439]

        # Downsample to target LOD if needed
        if self.lod == 1:
            visible = visible_lod1
        else:
            # Barycentric interpolation LOD1 → LOD_N via BodyConverter
            visible_float = self.body_converter._apply_lod_mapping_contacts(
                visible_lod1.to(self.body_converter._device), target_lod=self.lod
            ).to(verts_2d.device)
            # A LOD_N vertex is visible if >50% of its LOD1 constituents are visible
            visible = (visible_float > 0.5).float()

        # vertex_weights: 1.0 for visible, oob_weight for OOB
        vertex_weights = visible + (1.0 - visible) * self.oob_weight  # [B, num_vertices]

        oob_frac = 1.0 - visible.mean().item()
        return vertex_weights, oob_frac

    @torch.no_grad()
    def _compute_metrics(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5)
        gt = targets.bool()

        tp = (preds & gt).float().sum()
        fp = (preds & ~gt).float().sum()
        fn = (~preds & gt).float().sum()
        tn = (~preds & ~gt).float().sum()

        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)

        return {
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'iou': iou.item(),
        }

    # ------------------------------------------------------------------
    # Train epoch
    # ------------------------------------------------------------------

    def train_epoch(self):
        self.model.train()
        for dec in [getattr(self.model, 'decoder', None),
                    getattr(self.model, 'decoder_hand', None)]:
            if dec is not None:
                for m in dec.modules():
                    m.train()

        total_loss = 0.0
        total_metrics = {k: 0.0 for k in ('accuracy', 'precision', 'recall', 'f1', 'iou')}

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch_data in enumerate(pbar):
            if self.use_precomputed:
                features, bboxes, cam_ks, ori_img_sizes, pred_kp2d, pred_kp3d, contact_labels = batch_data
                contact_labels = contact_labels.to(self.device)
                batch, precomputed_feats = self._prepare_batch(features, bboxes, cam_ks, ori_img_sizes)
                self.model._initialize_batch(batch)
                output = self.model.forward_step(batch, decoder_type="body",
                                                 precomputed_features=precomputed_feats)
            else:
                images, bboxes, cam_ks, contact_labels = batch_data
                contact_labels = contact_labels.to(self.device)
                batch = self._prepare_batch(images, bboxes, cam_ks)
                self.model._initialize_batch(batch)
                output = self.model.forward_step(batch, decoder_type="body")

            if output["contact"] is None:
                raise RuntimeError(
                    "No contact output — ensure DO_CONTACT_TOKENS: true in config."
                )

            logits = output["contact"]["contact_logits"]  # [B, num_vertices]
            vw_result = self._compute_vertex_weights(output, batch)
            vertex_weights, oob_frac = vw_result if vw_result is not None else (None, 0.0)
            loss = self._compute_loss(logits, contact_labels, vertex_weights)

            # Pose supervision loss (only with precomputed features + LoRA)
            pose_loss_val = 0.0
            if self.use_pose_supervision and output.get("mhr") is not None:
                mhr_out = output["mhr"]
                gt_kp2d = pred_kp2d.to(self.device)
                gt_kp3d = pred_kp3d.to(self.device)
                # Model outputs are [B, 1, 70, D] — squeeze the person dim
                p_kp2d = mhr_out["pred_keypoints_2d"].squeeze(1)  # [B, 70, 2]
                p_kp3d = mhr_out["pred_keypoints_3d"].squeeze(1)  # [B, 70, 3]
                if self.pose_loss_type == 'smooth_l1':
                    kp2d_loss = F.smooth_l1_loss(p_kp2d, gt_kp2d)
                    kp3d_loss = F.smooth_l1_loss(p_kp3d, gt_kp3d)
                else:
                    kp2d_loss = F.mse_loss(p_kp2d, gt_kp2d)
                    kp3d_loss = F.mse_loss(p_kp3d, gt_kp3d)
                pose_loss = self.kp2d_weight * kp2d_loss + self.kp3d_weight * kp3d_loss
                loss = loss + pose_loss
                pose_loss_val = pose_loss.item()

            metrics = self._compute_metrics(logits, contact_labels)

            self.optimizer.zero_grad()
            loss.backward()

            if self.cfg.TRAIN.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.TRAIN.GRAD_CLIP)

            self.optimizer.step()

            # Gradient check on first batch of first epoch
            if self.current_epoch == 0 and batch_idx == 0:
                print("\n=== Gradient Flow Check ===")
                for name, param in self.model.named_parameters():
                    if param.requires_grad and ("contact" in name.lower() or "lora_" in name):
                        status = f"grad_norm={param.grad.norm().item():.6f}" if param.grad is not None else "NO GRAD"
                        print(f"  {name}: {status}")
                print("=" * 40 + "\n")

            total_loss += loss.item()
            for k in total_metrics:
                total_metrics[k] += metrics[k]

            d = self._last_loss_dict
            postfix = {
                'loss': f"{loss.item():.4f}",
                'focal': f"{d.get('focal_bce', 0):.4f}",
                'dice': f"{d.get('dice', 0):.4f}",
                'iou': f"{metrics['iou']:.4f}",
            }
            if pose_loss_val > 0:
                postfix['pose'] = f"{pose_loss_val:.4f}"
            pbar.set_postfix(postfix)

            if self.writer and self.global_step % self.cfg.OUTPUT.LOG_FREQ == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/loss_focal_bce', d.get('focal_bce', 0), self.global_step)
                self.writer.add_scalar('train/loss_dice', d.get('dice', 0), self.global_step)
                self.writer.add_scalar('train/loss_sparsity', d.get('sparsity', 0), self.global_step)
                self.writer.add_scalar('train/iou', metrics['iou'], self.global_step)
                self.writer.add_scalar('train/f1', metrics['f1'], self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                if oob_frac > 0.0:
                    self.writer.add_scalar('train/oob_vertex_frac', oob_frac, self.global_step)
                if pose_loss_val > 0:
                    self.writer.add_scalar('train/loss_pose', pose_loss_val, self.global_step)

            self.global_step += 1

        n = len(self.train_loader)
        return total_loss / n, {k: v / n for k, v in total_metrics.items()}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        total_loss = 0.0
        total_loss_components = {'focal_bce': 0.0, 'dice': 0.0, 'sparsity': 0.0}
        total_metrics = {k: 0.0 for k in ('accuracy', 'precision', 'recall', 'f1', 'iou')}

        for batch_data in tqdm(self.val_loader, desc="Validation"):
            if self.use_precomputed:
                features, bboxes, cam_ks, ori_img_sizes, pred_kp2d, pred_kp3d, contact_labels = batch_data
                contact_labels = contact_labels.to(self.device)
                batch, precomputed_feats = self._prepare_batch(features, bboxes, cam_ks, ori_img_sizes)
                self.model._initialize_batch(batch)
                output = self.model.forward_step(batch, decoder_type="body",
                                                 precomputed_features=precomputed_feats)
            else:
                images, bboxes, cam_ks, contact_labels = batch_data
                contact_labels = contact_labels.to(self.device)
                batch = self._prepare_batch(images, bboxes, cam_ks)
                self.model._initialize_batch(batch)
                output = self.model.forward_step(batch, decoder_type="body")

            logits = output["contact"]["contact_logits"]

            vw_result = self._compute_vertex_weights(output, batch)
            vertex_weights = vw_result[0] if vw_result is not None else None
            loss = self._compute_loss(logits, contact_labels, vertex_weights)
            metrics = self._compute_metrics(logits, contact_labels)

            total_loss += loss.item()
            for k in total_loss_components:
                total_loss_components[k] += self._last_loss_dict.get(k, 0.0)
            for k in total_metrics:
                total_metrics[k] += metrics[k]

        n = len(self.val_loader)
        avg_components = {k: v / n for k, v in total_loss_components.items()}

        if self.writer:
            self.writer.add_scalar('val/loss', total_loss / n, self.current_epoch)
            self.writer.add_scalar('val/loss_focal_bce', avg_components['focal_bce'], self.current_epoch)
            self.writer.add_scalar('val/loss_dice', avg_components['dice'], self.current_epoch)
            self.writer.add_scalar('val/loss_sparsity', avg_components['sparsity'], self.current_epoch)

        return total_loss / n, {k: v / n for k, v in total_metrics.items()}

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def _save_checkpoint(self, filename: str):
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': str(self.cfg),
        }
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def _load_checkpoint(self, checkpoint_path: str):
        print(f"Resuming from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.current_epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']
        self.best_val_loss = ckpt['best_val_loss']
        self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if self.scheduler and ckpt.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        print(f"\nStarting training — {self.cfg.TRAIN.EPOCHS} epochs")
        print(f"Output: {self.output_dir}\n")

        for epoch in range(self.current_epoch, self.cfg.TRAIN.EPOCHS):
            self.current_epoch = epoch

            train_loss, train_metrics = self.train_epoch()
            print(
                f"\nEpoch {epoch} | Train Loss: {train_loss:.4f} | "
                f"IoU: {train_metrics['iou']:.4f}  F1: {train_metrics['f1']:.4f}  "
                f"Prec: {train_metrics['precision']:.4f}  Rec: {train_metrics['recall']:.4f}"
            )

            if epoch % self.cfg.TRAIN.VAL_FREQ == 0:
                val_loss, val_metrics = self.validate()
                print(
                    f"          Val  Loss: {val_loss:.4f} | "
                    f"IoU: {val_metrics['iou']:.4f}  F1: {val_metrics['f1']:.4f}  "
                    f"Prec: {val_metrics['precision']:.4f}  Rec: {val_metrics['recall']:.4f}"
                )

                if self.writer:
                    self.writer.add_scalar('val/loss', val_loss, epoch)
                    self.writer.add_scalar('val/iou', val_metrics['iou'], epoch)
                    self.writer.add_scalar('val/f1', val_metrics['f1'], epoch)
                    self.writer.add_scalar('val/precision', val_metrics['precision'], epoch)
                    self.writer.add_scalar('val/recall', val_metrics['recall'], epoch)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint('best_model.pth')

            if epoch % self.cfg.TRAIN.SAVE_FREQ == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch:04d}.pth')

            if self.scheduler:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']
            print(f"  lr = {lr:.2e}")

        self._save_checkpoint('final_model.pth')
        if self.writer:
            self.writer.close()
        print("\nTraining complete!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train per-vertex Contact Head on DAMON")
    parser.add_argument("--config", type=str, default="train/config.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    trainer = ContactTrainer(args.config, device=args.device)
    trainer.train()


if __name__ == "__main__":
    main()
