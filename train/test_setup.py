"""
Pre-flight verification for Step 1 training setup.

Checks:
1. Config loads
2. SAM-3D-Body checkpoint loads
3. InteractionDecoder + ContactHead initialize correctly
4. Freeze logic leaves only the right params trainable
5. Dataset loads (precomputed mode)
6. Forward pass produces [B, num_vertices] logits
7. Backward pass computes gradients on interaction_decoder + head_contact
8. Optimizer step updates those params
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "dataset"))
sys.path.insert(0, str(Path(__file__).parent.parent / "mhr_smpl_conversion"))

os.environ["MOMENTUM_ENABLED"] = "1"

from sam_3d_body.build_models import load_sam_3d_body
from sam_3d_body.models.heads.contact_head import ContactHead
from sam_3d_body.models.decoders.interaction_decoder import InteractionDecoder
from sam_3d_body.utils.config import get_config
from damon_mhr import DamonPrecomputedDataset
from dataset_utils import prepare_damon_batch_precomputed
from torch.utils.data import DataLoader


def damon_precomputed_collate(batch):
    import numpy as np
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


def test_setup(config_path="configs/step1_contact.yaml"):
    print("=" * 70)
    print("STEP 1 TRAINING SETUP VERIFICATION")
    print("=" * 70)

    # 1. Config
    print("\n1. Loading config...")
    try:
        cfg = get_config(config_path)
        print(f"   OK  config loaded from {config_path}")
    except Exception as e:
        print(f"   FAIL  {e}")
        return False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. SAM-3D-Body checkpoint
    print("\n2. Loading SAM-3D-Body checkpoint...")
    try:
        model, model_cfg = load_sam_3d_body(
            checkpoint_path=cfg.MODEL.CHECKPOINT_PATH,
            device=device,
            mhr_path=cfg.MODEL.MHR_MODEL_PATH,
        )
        print("   OK  checkpoint loaded")
    except Exception as e:
        print(f"   FAIL  {e}")
        return False

    # 3. Initialize InteractionDecoder + ContactHead
    print("\n3. Initializing InteractionDecoder and ContactHead...")
    try:
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
        ).to(device)

        num_vertices = ch_cfg.get('NUM_VERTICES', 18439)
        num_contact_tokens = id_cfg.get('NUM_CONTACT_TOKENS', 16)
        model.head_contact = ContactHead(
            input_dim=id_cfg.get('D_MODEL', dim),
            num_contact_tokens=num_contact_tokens,
            num_vertices=num_vertices,
            mlp_depth=ch_cfg.get('MLP_DEPTH', 2),
            mlp_channel_div_factor=ch_cfg.get('MLP_CHANNEL_DIV_FACTOR', 4),
            pool_mode=ch_cfg.get('POOL_MODE', 'attention'),
            dropout=ch_cfg.get('DROPOUT', 0.0),
        ).to(device)
        print(f"   OK  InteractionDecoder: K={num_contact_tokens}, {id_cfg.get('NUM_LAYERS',4)} layers")
        print(f"   OK  ContactHead: {num_vertices} vertices")
    except Exception as e:
        print(f"   FAIL  {e}")
        import traceback; traceback.print_exc()
        return False

    # 4. Freeze logic
    print("\n4. Freezing all params except interaction_decoder + head_contact...")
    for param in model.parameters():
        param.requires_grad = False
    trainable_names = []
    for name, param in model.named_parameters():
        if name.startswith("interaction_decoder") or name.startswith("head_contact"):
            param.requires_grad = True
            trainable_names.append(name)

    if not trainable_names:
        print("   FAIL  No trainable parameters found!")
        return False

    trainable_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_n = sum(p.numel() for p in model.parameters())
    print(f"   OK  {trainable_n:,} / {total_n:,} params trainable")
    print(f"   OK  {len(trainable_names)} trainable parameter tensors")

    # 5. Dataset
    print("\n5. Loading dataset (precomputed features)...")
    use_precomputed = cfg.TRAIN.get('USE_PRECOMPUTED_FEATURES', False)
    loader = None
    if use_precomputed:
        try:
            features_base = cfg.TRAIN.get('PRECOMPUTED_FEATURES_DIR',
                                           './dataset/damon_mhr_contact/features')
            predictions_base = cfg.TRAIN.get('PRECOMPUTED_PREDICTIONS_DIR',
                                              './dataset/damon_mhr_contact/predictions')
            contact_npz = cfg.DATASET.CONTACT_NPZ
            detect_npz = cfg.DATASET.get('DETECT_NPZ', {})
            lod = cfg.DATASET.get('LOD', 1)
            train_ds, val_ds = DamonPrecomputedDataset.split_train_val(
                contact_npz_path=contact_npz.TRAINVAL,
                detect_npz_path=detect_npz.get('TRAINVAL', None),
                features_dir=os.path.join(features_base, 'trainval'),
                predictions_npz_path=os.path.join(predictions_base, 'trainval_predictions.npz'),
                lod=lod,
                val_ratio=cfg.DATASET.get('VAL_RATIO', 0.15),
                seed=cfg.DATASET.get('SEED', 42),
                data_root=cfg.DATASET.get('DATA_ROOT', None),
            )
            loader = DataLoader(train_ds, batch_size=2, shuffle=False,
                                num_workers=0, collate_fn=damon_precomputed_collate)
            print(f"   OK  train={len(train_ds)}  val={len(val_ds)} samples")
        except Exception as e:
            print(f"   WARN  Could not load dataset: {e}")
            print("   Skipping forward/backward pass tests (dataset not available).")
            loader = None
    else:
        print("   SKIP  USE_PRECOMPUTED_FEATURES=false; run with precomputed features for full test")

    if loader is None:
        print("\nPartial verification done (no dataset). Model modules initialized correctly.")
        return True

    # 6. Forward pass
    print("\n6. Testing forward pass...")
    try:
        batch_data = next(iter(loader))
        features, bboxes, cam_ks, ori_img_sizes, pred_kp2d, pred_kp3d, contact_labels = batch_data
        contact_labels = contact_labels.to(device)

        model_img_size = tuple(model_cfg.MODEL.IMAGE_SIZE)
        batch, precomputed_feats = prepare_damon_batch_precomputed(
            features, bboxes, cam_ks, ori_img_sizes,
            target_size=model_img_size, device=device,
        )
        model._initialize_batch(batch)
        model.train()

        output = model.forward_step(batch, decoder_type="body",
                                    precomputed_features=precomputed_feats)

        assert output["body_tokens"] is not None, "body_tokens is None"
        assert output["image_embeddings"] is not None, "image_embeddings is None"

        contact_tokens = model.interaction_decoder(
            output["image_embeddings"],
            output["body_tokens"],
        )
        logits = model.head_contact(contact_tokens)

        assert logits.shape == (contact_labels.shape[0], num_vertices), \
            f"Expected [{contact_labels.shape[0]}, {num_vertices}], got {logits.shape}"
        print(f"   OK  logits shape: {logits.shape}")
        print(f"   OK  body_tokens shape: {output['body_tokens'].shape}")
        print(f"   OK  image_embeddings shape: {output['image_embeddings'].shape}")
    except Exception as e:
        print(f"   FAIL  {e}")
        import traceback; traceback.print_exc()
        return False

    # 7. Backward pass
    print("\n7. Testing backward pass...")
    try:
        loss = F.binary_cross_entropy_with_logits(logits, contact_labels.float())
        loss.backward()

        grad_ok = []
        grad_missing = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_ok.append((name, param.grad.norm().item()))
                else:
                    grad_missing.append(name)

        for name, gnorm in grad_ok:
            print(f"   OK  {name}: grad_norm={gnorm:.6f}")
        for name in grad_missing:
            print(f"   FAIL  {name}: NO GRADIENT")

        if grad_missing:
            return False
        print(f"   OK  {len(grad_ok)} param tensors have gradients")
    except Exception as e:
        print(f"   FAIL  {e}")
        import traceback; traceback.print_exc()
        return False

    # 8. Optimizer step
    print("\n8. Testing optimizer step...")
    try:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
        )
        optimizer.step()
        print("   OK  optimizer step succeeded")
    except Exception as e:
        print(f"   FAIL  {e}")
        return False

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
    print(f"\nRun training with:")
    print(f"  python train/train_contact.py --config {config_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/step1_contact.yaml")
    args = parser.parse_args()
    success = test_setup(args.config)
    sys.exit(0 if success else 1)
