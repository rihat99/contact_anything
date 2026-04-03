"""
Training script for SMPL contact prediction on DAMON dataset.

Trains InteractionDecoder + ContactHead + PartContactHead while keeping
SAM-3D-Body frozen.  Each run writes to a timestamped folder under OUTPUT.DIR.

Targets:
  - per-vertex SMPL contacts (6890 verts)   — ContactHead
  - per-body-part contacts (24 SMPL parts)  — PartContactHead (auxiliary)
"""

import faulthandler
faulthandler.enable()

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["MOMENTUM_ENABLED"] = "1"

from sam_3d_body.build_models import load_sam_3d_body
from sam_3d_body.models.heads.contact_head import ContactHead, PartContactHead
from sam_3d_body.models.decoders.interaction_decoder import InteractionDecoder
from sam_3d_body.utils.config import get_config

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "dataset"))
from damon_dataset import DamonDataset, DamonPrecomputedDataset
from dataset_utils import prepare_damon_batch, prepare_damon_batch_precomputed
from losses import ContactLoss


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

# Feature-map spatial size for DINOv3-H at 896×896 input
_FEATURE_HW = (56, 56)


def _preprocess_object_mask(mask, person_bbox, feature_hw=_FEATURE_HW):
    """
    Crop a boolean mask to the person bbox and resize to feature_hw.

    Args:
        mask:        bool numpy array [H, W] or None.
        person_bbox: float32 tensor [4] as (x1, y1, x2, y2) in original image pixels.
        feature_hw:  target (H, W) matching spatial size of precomputed features.

    Returns:
        float32 tensor [1, H_feat, W_feat] with values in {0.0, 1.0}.
        Returns all-zeros if mask is None or bbox is degenerate.
    """
    H_f, W_f = feature_hw
    if mask is None:
        return torch.zeros(1, H_f, W_f, dtype=torch.float32)

    H_img, W_img = mask.shape
    bbox = person_bbox.numpy() if isinstance(person_bbox, torch.Tensor) else person_bbox
    x1, y1, x2, y2 = bbox.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W_img, x2), min(H_img, y2)

    if x2 <= x1 or y2 <= y1:
        return torch.zeros(1, H_f, W_f, dtype=torch.float32)

    cropped = mask[y1:y2, x1:x2].astype(np.float32)
    resized = cv2.resize(cropped, (W_f, H_f), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(resized).unsqueeze(0)  # [1, H_f, W_f]


def _unpack_label(lbl):
    """Return (vertex_label_tensor, part_label_tensor_or_None) from a label item."""
    if isinstance(lbl, dict):
        return lbl['contact_label'], lbl.get('part_contact')
    return lbl, None


def damon_collate(batch):
    """
    Collate for DamonDataset classic mode (raw images).
    Returns: images, bboxes, cam_ks, vertex_labels, part_labels_or_None
    """
    images, bboxes, cam_ks, vlabels, plabels = [], [], [], [], []
    has_parts = None
    for (img, bbox, cam_k), lbl in batch:
        images.append(img)
        bboxes.append(bbox)
        cam_ks.append(cam_k)
        vl, pl = _unpack_label(lbl)
        vlabels.append(vl)
        plabels.append(pl)
        has_parts = pl is not None
    return (
        images,
        torch.stack(bboxes),
        torch.stack(cam_ks),
        torch.stack(vlabels),
        torch.stack(plabels) if has_parts else None,
    )


def damon_precomputed_collate(batch):
    """
    Collate for DamonPrecomputedDataset classic mode.
    Returns: features, bboxes, cam_ks, ori_sizes, vertex_labels, part_labels_or_None
    """
    features, bboxes, cam_ks, ori_sizes, vlabels, plabels = [], [], [], [], [], []
    has_parts = None
    for (feat, bbox, cam_k, ori_size), lbl in batch:
        features.append(feat)
        bboxes.append(bbox)
        cam_ks.append(cam_k)
        ori_sizes.append(ori_size)
        vl, pl = _unpack_label(lbl)
        vlabels.append(vl)
        plabels.append(pl)
        has_parts = pl is not None
    return (
        torch.stack(features),
        torch.stack(bboxes),
        torch.stack(cam_ks),
        torch.stack(ori_sizes),
        torch.stack(vlabels),
        torch.stack(plabels) if has_parts else None,
    )


def damon_instance_precomputed_collate(batch):
    """
    Collate for DamonPrecomputedDataset instance_contact mode.
    Returns: features, bboxes, cam_ks, ori_sizes, vertex_labels, part_labels_or_None,
             person_masks [B,1,56,56], object_masks [B,1,56,56], object_names
    """
    features, bboxes, cam_ks, ori_sizes = [], [], [], []
    vlabels, plabels, person_masks, obj_masks, obj_names = [], [], [], [], []
    has_parts = None

    for inputs_dict, label_dict in batch:
        features.append(inputs_dict['feature'])
        bboxes.append(inputs_dict['person_bbox'])
        cam_ks.append(inputs_dict['cam_k'])
        ori_sizes.append(inputs_dict['ori_img_size'])
        vl, pl = _unpack_label(label_dict)
        vlabels.append(vl)
        plabels.append(pl)
        has_parts = pl is not None
        obj_names.append(label_dict['object_name'])
        person_masks.append(_preprocess_object_mask(
            inputs_dict.get('person_mask'), inputs_dict['person_bbox'],
        ))
        obj_masks.append(_preprocess_object_mask(
            inputs_dict.get('object_mask'), inputs_dict['person_bbox'],
        ))

    return (
        torch.stack(features),
        torch.stack(bboxes),
        torch.stack(cam_ks),
        torch.stack(ori_sizes),
        torch.stack(vlabels),
        torch.stack(plabels) if has_parts else None,
        torch.stack(person_masks),  # [B, 1, 56, 56]
        torch.stack(obj_masks),     # [B, 1, 56, 56]
        obj_names,
    )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ContactTrainer:
    """Trains SMPL contact heads on DAMON dataset."""

    def __init__(self, config_path: str, device: str = "cuda"):
        self.cfg = get_config(config_path)
        self.device = device

        # Output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.cfg.OUTPUT.EXP_NAME}_{timestamp}"
        self.output_dir = Path(self.cfg.OUTPUT.DIR) / exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

        with open(self.output_dir / "config.yaml", "w") as f:
            f.write(str(self.cfg))

        if self.cfg.OUTPUT.USE_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        else:
            self.writer = None

        # ---- SAM-3D-Body base model (frozen) ----
        print("Loading SAM-3D-Body model...")
        self.model, self.model_cfg = load_sam_3d_body(
            checkpoint_path=self.cfg.MODEL.CHECKPOINT_PATH,
            device=device,
            mhr_path=self.cfg.MODEL.MHR_MODEL_PATH,
        )

        # ---- Initialize contact modules ----
        import torch.nn as nn
        dim = self.model_cfg.MODEL.DECODER.DIM  # 1024
        id_cfg = self.cfg.MODEL.INTERACTION_DECODER
        ch_cfg = self.cfg.MODEL.CONTACT_HEAD

        self.model.interaction_decoder = InteractionDecoder(
            d_model=id_cfg.get('D_MODEL', dim),
            image_feat_dim=id_cfg.get('IMAGE_FEAT_DIM', 1280),
            num_layers=id_cfg.get('NUM_LAYERS', 4),
            num_heads=id_cfg.get('NUM_HEADS', 8),
            ffn_dim=id_cfg.get('FFN_DIM', 2048),
            dropout=id_cfg.get('DROPOUT', 0.0),
        ).to(device)

        num_vertices = ch_cfg.get('NUM_VERTICES', 6890)
        self.model.head_contact = ContactHead(
            input_dim=id_cfg.get('D_MODEL', dim),
            num_vertices=num_vertices,
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
        print(f"ContactHead: {num_vertices} vertices")

        # ---- Freeze all except contact modules ----
        for param in self.model.parameters():
            param.requires_grad = False
        for name, param in self.model.named_parameters():
            if any(name.startswith(p) for p in
                   ('interaction_decoder', 'head_contact', 'head_part_contact')):
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable: {trainable:,} / {total:,}")

        self.step2_mode = self.cfg.TRAIN.get('STEP2_MODE', False)

        # ---- Datasets ----
        print("Loading datasets...")
        data_root  = self.cfg.DATASET.get('DATA_ROOT', None)
        val_ratio  = self.cfg.DATASET.get('VAL_RATIO', 0.2)
        seed       = self.cfg.DATASET.get('SEED', 42)
        topology   = self.cfg.DATASET.get('TOPOLOGY', 'mhr')
        part_seg   = self.cfg.DATASET.get('SMPL_PART_SEG_PATH', None)
        contact_npz = self.cfg.DATASET.CONTACT_NPZ
        detect_npz  = self.cfg.DATASET.get('DETECT_NPZ', {})
        self.use_precomputed = self.cfg.TRAIN.get('USE_PRECOMPUTED_FEATURES', False)

        if self.use_precomputed:
            features_base = self.cfg.TRAIN.get('PRECOMPUTED_FEATURES_DIR',
                                               './dataset/damon_mhr_contact/features')
            self.train_dataset, self.val_dataset = DamonPrecomputedDataset.split_train_val(
                contact_npz_path=contact_npz.TRAINVAL,
                detect_npz_path=detect_npz.get('TRAINVAL', None),
                features_dir=os.path.join(features_base, 'trainval'),
                topology=topology,
                val_ratio=val_ratio,
                seed=seed,
                data_root=data_root,
                smpl_part_seg_path=part_seg,
            )
            collate_fn = damon_precomputed_collate
        else:
            self.train_dataset, self.val_dataset = DamonDataset.split_train_val(
                contact_npz_path=contact_npz.TRAINVAL,
                detect_npz_path=detect_npz.get('TRAINVAL', None),
                topology=topology,
                val_ratio=val_ratio,
                seed=seed,
                data_root=data_root,
                smpl_part_seg_path=part_seg,
            )
            collate_fn = damon_collate

        print(f"  Train: {len(self.train_dataset)}  Val: {len(self.val_dataset)}")

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=True, num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=False, drop_last=True, collate_fn=collate_fn,
            persistent_workers=self.cfg.TRAIN.NUM_WORKERS > 0,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.cfg.TRAIN.VAL_BATCH_SIZE,
            shuffle=False, num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=False, collate_fn=collate_fn,
            persistent_workers=self.cfg.TRAIN.NUM_WORKERS > 0,
        )

        # ---- Step 2: instance_contact DataLoaders ----
        self.instance_train_loader = None
        self.instance_val_loader = None
        if self.step2_mode and self.use_precomputed:
            masks_v2_dir = self.cfg.DATASET.get('MASKS_V2_DIR', None)
            if masks_v2_dir is None:
                raise ValueError("DATASET.MASKS_V2_DIR must be set when TRAIN.STEP2_MODE=true")
            inst_batch = self.cfg.TRAIN.get('INSTANCE_BATCH_SIZE', self.cfg.TRAIN.BATCH_SIZE)

            inst_train, inst_val = DamonPrecomputedDataset.split_train_val(
                contact_npz_path=contact_npz.TRAINVAL,
                detect_npz_path=detect_npz.get('TRAINVAL', None),
                features_dir=os.path.join(features_base, 'trainval'),
                topology=topology,
                val_ratio=val_ratio, seed=seed, data_root=data_root,
                mode='instance_all', masks_v2_dir=masks_v2_dir,
                smpl_part_seg_path=part_seg,
            )
            print(f"  Instance Train: {len(inst_train)}  Val: {len(inst_val)}")

            self.instance_train_loader = DataLoader(
                inst_train, batch_size=inst_batch, shuffle=True,
                num_workers=self.cfg.TRAIN.NUM_WORKERS, pin_memory=False,
                drop_last=True, collate_fn=damon_instance_precomputed_collate,
                persistent_workers=self.cfg.TRAIN.NUM_WORKERS > 0,
            )
            self.instance_val_loader = DataLoader(
                inst_val, batch_size=self.cfg.TRAIN.VAL_BATCH_SIZE,
                shuffle=False, num_workers=self.cfg.TRAIN.NUM_WORKERS,
                pin_memory=False, collate_fn=damon_instance_precomputed_collate,
                persistent_workers=self.cfg.TRAIN.NUM_WORKERS > 0,
            )

        # ---- Step 2: optionally load Step 1 checkpoint ----
        if self.step2_mode:
            step1_ckpt = self.cfg.TRAIN.get('STEP1_CHECKPOINT', None)
            if step1_ckpt:
                self._load_step1_checkpoint(step1_ckpt)

        # ---- Loss ----
        loss_cfg = self.cfg.get("LOSS", {})
        self.loss_fn = ContactLoss(
            vertex_pos_weight=loss_cfg.get("VERTEX_POS_WEIGHT", 10.0),
            vertex_weight=loss_cfg.get("VERTEX_WEIGHT", 1.0),
            part_weight=loss_cfg.get("PART_WEIGHT", 0.5),
        ).to(device)
        self._last_loss_dict: dict = {}
        print(f"ContactLoss: vertex_pos_weight={self.loss_fn.pos_weight.item():.1f}  "
              f"vertex_weight={self.loss_fn.vertex_weight}  "
              f"part_weight={self.loss_fn.part_weight}")

        # ---- Optimizer & scheduler ----
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
        )
        self.scheduler = self._setup_scheduler()

        # ---- State ----
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        if self.cfg.TRAIN.RESUME:
            self._load_checkpoint(self.cfg.TRAIN.RESUME)

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def _setup_scheduler(self):
        warmup = self.cfg.TRAIN.get("LR_WARMUP_EPOCHS", 0)
        sched_type = self.cfg.TRAIN.LR_SCHEDULER
        if sched_type == "cosine":
            main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(self.cfg.TRAIN.EPOCHS - warmup, 1),
                eta_min=self.cfg.TRAIN.LR_MIN,
            )
        elif sched_type == "step":
            main_sched = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1)
        else:
            return None

        if warmup > 0:
            warm_sched = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.01, total_iters=warmup)
            return torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warm_sched, main_sched], milestones=[warmup])
        return main_sched

    # ------------------------------------------------------------------
    # Batch preparation
    # ------------------------------------------------------------------

    def _prepare_batch(self, *args):
        if self.use_precomputed:
            features, bboxes, cam_ks, ori_img_sizes = args
            return prepare_damon_batch_precomputed(
                features, bboxes, cam_ks, ori_img_sizes,
                target_size=tuple(self.cfg.MODEL.IMAGE_SIZE),
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
    # Forward: decode batch → contact tokens → logits
    # ------------------------------------------------------------------

    def _forward_contact(self, batch_data, mode: str):
        """
        Run model forward for one batch.

        Returns:
            vertex_logits [B, V], part_logits [B, 24],
            vertex_labels [B, V], part_labels [B, 24] or None
        """
        if mode == 'instance':
            features, bboxes, cam_ks, ori_img_sizes, \
                vertex_labels, part_labels, person_masks, object_masks, _ = batch_data
            batch, precomputed_feats = self._prepare_batch(features, bboxes, cam_ks, ori_img_sizes)
            person_mask = person_masks.to(self.device) if self.step2_mode else None
            object_mask = object_masks.to(self.device) if self.step2_mode else None
        elif self.use_precomputed:
            features, bboxes, cam_ks, ori_img_sizes, vertex_labels, part_labels = batch_data
            batch, precomputed_feats = self._prepare_batch(features, bboxes, cam_ks, ori_img_sizes)
            person_mask = None
            object_mask = None
        else:
            images, bboxes, cam_ks, vertex_labels, part_labels = batch_data
            batch = self._prepare_batch(images, bboxes, cam_ks)
            precomputed_feats = None
            person_mask = None
            object_mask = None

        vertex_labels = vertex_labels.to(self.device)
        if part_labels is not None:
            part_labels = part_labels.to(self.device)

        self.model._initialize_batch(batch)
        output = self.model.forward_step(
            batch, decoder_type="body",
            **({"precomputed_features": precomputed_feats} if precomputed_feats is not None else {}),
        )

        tokens = self.model.interaction_decoder(
            output["image_embeddings"],
            output["body_tokens"],
            person_mask=person_mask,
            object_mask=object_mask,
        )  # [B, 25, d_model]

        part_tokens   = tokens[:, :InteractionDecoder.NUM_PART_TOKENS, :]   # [B, 24, d]
        vertex_token  = tokens[:, InteractionDecoder.NUM_PART_TOKENS:, :]   # [B, 1, d]

        vertex_logits = self.model.head_contact(vertex_token)               # [B, V]
        part_logits   = self.model.head_part_contact(part_tokens)           # [B, 24]

        return vertex_logits, part_logits, vertex_labels, part_labels

    # ------------------------------------------------------------------
    # Loss & metrics
    # ------------------------------------------------------------------

    def _compute_loss(self, vertex_logits, vertex_labels, part_logits, part_labels):
        if part_labels is None:
            # Fallback: vertex-only loss (no part supervision)
            loss = F.binary_cross_entropy_with_logits(
                vertex_logits, vertex_labels.float(),
                pos_weight=self.loss_fn.pos_weight,
            )
            self._last_loss_dict = {'vertex_bce': loss.item(), 'part_bce': 0.0, 'total': loss.item()}
            return loss
        total, d = self.loss_fn(vertex_logits, vertex_labels.float(),
                                part_logits, part_labels.float())
        self._last_loss_dict = d
        return total

    @torch.no_grad()
    def _compute_metrics(self, vertex_logits, vertex_labels, part_logits=None, part_labels=None):
        probs = torch.sigmoid(vertex_logits)
        preds = probs > 0.5
        gt = vertex_labels.bool()

        tp = (preds & gt).float().sum()
        fp = (preds & ~gt).float().sum()
        fn = (~preds & gt).float().sum()
        tn = (~preds & ~gt).float().sum()

        metrics = {
            'accuracy':  ((tp + tn) / (tp + tn + fp + fn + 1e-8)).item(),
            'precision': (tp / (tp + fp + 1e-8)).item(),
            'recall':    (tp / (tp + fn + 1e-8)).item(),
            'f1':        (2*tp / (2*tp + fp + fn + 1e-8)).item(),
            'iou':       (tp / (tp + fp + fn + 1e-8)).item(),
        }

        if part_logits is not None and part_labels is not None:
            part_preds = (torch.sigmoid(part_logits) > 0.5)
            part_acc = (part_preds == part_labels.bool()).float().mean().item()
            metrics['part_acc'] = part_acc

        return metrics

    # ------------------------------------------------------------------
    # Step 2 helpers
    # ------------------------------------------------------------------

    def _load_step1_checkpoint(self, checkpoint_path: str):
        print(f"Loading Step 1 checkpoint for Step 2 init: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        missing, unexpected = self.model.load_state_dict(
            ckpt['model_state_dict'], strict=False)
        if missing:
            print(f"  Missing keys (random-init): {missing}")

    def _mixed_iter(self, epoch: int):
        if self.instance_train_loader is None:
            for batch_data in self.train_loader:
                yield batch_data, 'classic'
            return
        if self.cfg.TRAIN.get('INSTANCE_ONLY', False):
            for batch_data in self.instance_train_loader:
                yield batch_data, 'instance'
            return
        frac = self.cfg.TRAIN.get('INSTANCE_FRACTION', 0.7)
        c_iter = iter(self.train_loader)
        i_iter = iter(self.instance_train_loader)
        n_total = len(self.train_loader) + len(self.instance_train_loader)
        rng = np.random.default_rng(seed=epoch)
        for mode in rng.choice(['instance', 'classic'], size=n_total,
                                p=[frac, 1.0 - frac]):
            try:
                yield (next(i_iter) if mode == 'instance' else next(c_iter)), mode
            except StopIteration:
                return

    def _mixed_val_iter(self):
        if self.instance_val_loader is None:
            for batch_data in self.val_loader:
                yield batch_data, 'classic'
            return
        if self.cfg.TRAIN.get('INSTANCE_ONLY', False):
            for batch_data in self.instance_val_loader:
                yield batch_data, 'instance'
            return
        for batch_data in self.val_loader:
            yield batch_data, 'classic'
        for batch_data in self.instance_val_loader:
            yield batch_data, 'instance'

    # ------------------------------------------------------------------
    # Train epoch
    # ------------------------------------------------------------------

    def train_epoch(self):
        self.model.train()

        total_loss = 0.0
        total_metrics = {k: 0.0 for k in ('accuracy', 'precision', 'recall', 'f1', 'iou')}
        total_part_acc = 0.0
        n_batches = 0

        pbar = tqdm(self._mixed_iter(self.current_epoch),
                    desc=f"Epoch {self.current_epoch}")
        for batch_idx, (batch_data, mode) in enumerate(pbar):
            vertex_logits, part_logits, vertex_labels, part_labels = \
                self._forward_contact(batch_data, mode)

            loss = self._compute_loss(vertex_logits, vertex_labels, part_logits, part_labels)
            metrics = self._compute_metrics(vertex_logits, vertex_labels, part_logits, part_labels)

            self.optimizer.zero_grad()
            loss.backward()
            if self.cfg.TRAIN.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.TRAIN.GRAD_CLIP)
            self.optimizer.step()

            # Gradient check on first batch of first epoch
            if self.current_epoch == 0 and batch_idx == 0:
                print("\n=== Gradient Flow ===")
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        s = f"norm={param.grad.norm().item():.6f}" if param.grad is not None else "NO GRAD"
                        print(f"  {name}: {s}")
                print("=" * 40 + "\n")

            total_loss += loss.item()
            for k in total_metrics:
                total_metrics[k] += metrics.get(k, 0.0)
            total_part_acc += metrics.get('part_acc', 0.0)

            d = self._last_loss_dict
            postfix = {
                'loss': f"{loss.item():.4f}",
                'v_bce': f"{d.get('vertex_bce', 0):.4f}",
                'p_bce': f"{d.get('part_bce', 0):.4f}",
                'iou':   f"{metrics['iou']:.4f}",
            }
            if self.step2_mode:
                postfix['mode'] = mode[0]
            pbar.set_postfix(postfix)

            if self.writer and self.global_step % self.cfg.OUTPUT.LOG_FREQ == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/loss_vertex_bce', d.get('vertex_bce', 0), self.global_step)
                self.writer.add_scalar('train/loss_part_bce', d.get('part_bce', 0), self.global_step)
                self.writer.add_scalar('train/iou', metrics['iou'], self.global_step)
                self.writer.add_scalar('train/f1', metrics['f1'], self.global_step)
                if 'part_acc' in metrics:
                    self.writer.add_scalar('train/part_acc', metrics['part_acc'], self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

            self.global_step += 1
            n_batches += 1

        n = n_batches or 1
        return total_loss / n, {k: v / n for k, v in total_metrics.items()}, total_part_acc / n

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        total_loss = 0.0
        total_loss_components = {'vertex_bce': 0.0, 'part_bce': 0.0}
        total_metrics = {k: 0.0 for k in ('accuracy', 'precision', 'recall', 'f1', 'iou')}
        total_part_acc = 0.0
        n = 0

        for batch_data, mode in tqdm(self._mixed_val_iter(), desc="Validation"):
            vertex_logits, part_logits, vertex_labels, part_labels = \
                self._forward_contact(batch_data, mode)

            loss = self._compute_loss(vertex_logits, vertex_labels, part_logits, part_labels)
            metrics = self._compute_metrics(vertex_logits, vertex_labels, part_logits, part_labels)

            total_loss += loss.item()
            for k in total_loss_components:
                total_loss_components[k] += self._last_loss_dict.get(k, 0.0)
            for k in total_metrics:
                total_metrics[k] += metrics.get(k, 0.0)
            total_part_acc += metrics.get('part_acc', 0.0)
            n += 1

        if n == 0:
            return 0.0, {k: 0.0 for k in total_metrics}, 0.0

        if self.writer:
            self.writer.add_scalar('val/loss', total_loss / n, self.current_epoch)
            self.writer.add_scalar('val/loss_vertex_bce',
                                   total_loss_components['vertex_bce'] / n, self.current_epoch)
            self.writer.add_scalar('val/loss_part_bce',
                                   total_loss_components['part_bce'] / n, self.current_epoch)
            self.writer.add_scalar('val/part_acc', total_part_acc / n, self.current_epoch)

        return total_loss / n, {k: v / n for k, v in total_metrics.items()}, total_part_acc / n

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

            train_loss, train_metrics, train_part_acc = self.train_epoch()
            print(
                f"\nEpoch {epoch} | Train Loss: {train_loss:.4f} | "
                f"IoU: {train_metrics['iou']:.4f}  F1: {train_metrics['f1']:.4f}  "
                f"Prec: {train_metrics['precision']:.4f}  Rec: {train_metrics['recall']:.4f}  "
                f"PartAcc: {train_part_acc:.4f}"
            )

            if epoch % self.cfg.TRAIN.VAL_FREQ == 0:
                val_loss, val_metrics, val_part_acc = self.validate()
                print(
                    f"          Val  Loss: {val_loss:.4f} | "
                    f"IoU: {val_metrics['iou']:.4f}  F1: {val_metrics['f1']:.4f}  "
                    f"Prec: {val_metrics['precision']:.4f}  Rec: {val_metrics['recall']:.4f}  "
                    f"PartAcc: {val_part_acc:.4f}"
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
            print(f"  lr = {self.optimizer.param_groups[0]['lr']:.2e}")

        self._save_checkpoint('final_model.pth')
        if self.writer:
            self.writer.close()
        print("\nTraining complete!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train SMPL contact heads on DAMON")
    parser.add_argument("--config", type=str, default="configs/step1_contact.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    trainer = ContactTrainer(args.config, device=args.device)
    trainer.train()


if __name__ == "__main__":
    main()
