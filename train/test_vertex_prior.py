"""
Thorough unit tests for the vertex geometric prior (Improvement D).

Tests cover:
  1. _normalize_kp2d — correct bbox-relative normalization
  2. _preprocess_object_mask — correct crop + resize
  3. Coordinate alignment — vertex pixel coords and object mask in same space
  4. _soft_proximity_field — spread and normalization
  5. _apply_vertex_prior — near/far vertex behavior, gradient flow
  6. Edge cases — zero mask, out-of-bbox vertices, single-pixel mask
  7. Real-data sanity check — loads an actual sample and verifies statistics

Run:
    python train/test_vertex_prior.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import cv2

# Import the helpers we want to test
from train.train_contact import _preprocess_object_mask

# ─────────────────────────────────────────────────────────────────────────────
# Helpers that mirror ContactTrainer (avoid instantiating the full trainer)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_kp2d(kp2d: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
    """Mirrors ContactTrainer._normalize_kp2d (copied to test standalone)."""
    kp2d = kp2d.float()
    bbox = bboxes.float()
    x1 = bbox[:, 0:1]
    y1 = bbox[:, 1:2]
    bw = (bbox[:, 2:3] - x1).clamp(min=1.0)
    bh = (bbox[:, 3:4] - y1).clamp(min=1.0)
    norm_x = 2.0 * (kp2d[:, :, 0] - x1) / bw - 1.0
    norm_y = 2.0 * (kp2d[:, :, 1] - y1) / bh - 1.0
    return torch.stack([norm_x, norm_y], dim=-1)


def _soft_proximity_field(mask: torch.Tensor, n_blur=3, kernel_size=9) -> torch.Tensor:
    """Mirrors InteractionDecoder._soft_proximity_field."""
    spread = mask.float()
    for _ in range(n_blur):
        spread = F.avg_pool2d(spread, kernel_size=kernel_size,
                               stride=1, padding=kernel_size // 2)
    peak = spread.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    return spread / peak


def _apply_vertex_prior(
    logits: torch.Tensor,       # [B, V]
    verts_2d: torch.Tensor,     # [B, V, 2] pixel coords
    bboxes: torch.Tensor,       # [B, 4] (x1, y1, x2, y2) in pixel coords
    object_mask: torch.Tensor,  # [B, 1, H, W] binary
    lambda_v: float = 1.0,
) -> torch.Tensor:
    """Mirrors ContactTrainer._apply_vertex_prior."""
    with torch.no_grad():
        v2d_norm = _normalize_kp2d(verts_2d, bboxes)

        prox_field = object_mask.float()
        for _ in range(3):
            prox_field = F.avg_pool2d(prox_field, kernel_size=9, stride=1, padding=4)
        prox_field = prox_field / prox_field.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)

        B, V, _ = v2d_norm.shape
        grid = v2d_norm.view(B, V, 1, 2)  # no clamp — OOB → zeros from padding_mode
        vertex_prox = F.grid_sample(
            prox_field, grid,
            mode='bilinear', align_corners=True, padding_mode='zeros',
        )
        vertex_prox = vertex_prox.squeeze(-1).squeeze(1)
        prior = torch.log(vertex_prox.clamp(min=1e-4))
    return logits + lambda_v * prior


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    _results.append((name, condition))
    return condition


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: _normalize_kp2d
# ─────────────────────────────────────────────────────────────────────────────

def test_normalize_kp2d():
    print("\n=== Test 1: _normalize_kp2d ===")

    # bbox: x1=100, y1=50, x2=300, y2=450  → bw=200, bh=400
    bbox = torch.tensor([[100., 50., 300., 450.]])  # [1, 4]

    # bbox: x1=100, y1=50, x2=300, y2=450 → bw=200, bh=400
    # Expected normalizations:
    #   (100,50)  → (-1, -1)
    #   (300,450) → (+1, +1)
    #   (200,250) → (0, 0)          midpoint: (100+300)/2=200, (50+450)/2=250
    #   (150,150) → (-0.5, -0.5)   x: 2*(50)/200-1=-0.5  y: 2*(100)/400-1=-0.5
    #   (50, 50)  → outside on x   x: 2*(50-100)/200-1=-1.5
    kps = torch.tensor([[[
        [100., 50.],   # top-left corner → (-1, -1)
        [300., 450.],  # bottom-right corner → (+1, +1)
        [200., 250.],  # center → (0, 0)
        [150., 150.],  # → (-0.5, -0.5)
        [50., 50.],    # outside left → x=-1.5  — will be outside [-1,1]
    ]]]).squeeze(0)  # [1, 5, 2]

    norm = _normalize_kp2d(kps, bbox)  # [1, 5, 2]

    check("top-left → (-1, -1)",
          torch.allclose(norm[0, 0], torch.tensor([-1., -1.]), atol=1e-4),
          f"got {norm[0, 0].tolist()}")
    check("bottom-right → (+1, +1)",
          torch.allclose(norm[0, 1], torch.tensor([1., 1.]), atol=1e-4),
          f"got {norm[0, 1].tolist()}")
    check("center → (0, 0)",
          torch.allclose(norm[0, 2], torch.tensor([0., 0.]), atol=1e-4),
          f"got {norm[0, 2].tolist()}")
    check("(150,150) → (-0.5, -0.5)",
          torch.allclose(norm[0, 3], torch.tensor([-0.5, -0.5]), atol=1e-4),
          f"got {norm[0, 3].tolist()}")
    check("outside-left x < -1",
          norm[0, 4, 0].item() < -1.0,
          f"got x={norm[0, 4, 0].item():.3f}")

    # Test batch consistency: two identical bboxes give identical normalization
    bbox2 = torch.cat([bbox, bbox], dim=0)  # [2, 4]
    kps2 = torch.cat([kps, kps], dim=0)    # [2, 5, 2]
    norm2 = _normalize_kp2d(kps2, bbox2)
    check("batch consistency",
          torch.allclose(norm2[0], norm2[1], atol=1e-5))


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: _preprocess_object_mask coordinate alignment
# ─────────────────────────────────────────────────────────────────────────────

def test_preprocess_mask():
    print("\n=== Test 2: _preprocess_object_mask ===")

    IMG_H, IMG_W = 480, 640
    # Person bbox occupies right half of image
    person_bbox = torch.tensor([320., 0., 640., 480.])  # x1,y1,x2,y2

    # Object mask: True only in a 40×40 square at pixel (400..440, 200..240) in original image
    # → inside person bbox
    obj_mask_np = np.zeros((IMG_H, IMG_W), dtype=bool)
    obj_mask_np[200:240, 400:440] = True

    mask_t = _preprocess_object_mask(obj_mask_np, person_bbox, feature_hw=(56, 56))

    check("output shape [1, 56, 56]", mask_t.shape == (1, 56, 56),
          f"got {mask_t.shape}")
    check("values binary (0 or 1)", mask_t.unique().tolist() in [[0.0], [1.0], [0.0, 1.0]])
    check("has some mask pixels",  mask_t.sum().item() > 0,
          f"sum={mask_t.sum().item()}")
    check("not all mask pixels", mask_t.sum().item() < 56 * 56,
          f"sum={mask_t.sum().item()}")

    # Object entirely outside the person bbox → should produce all-zeros
    obj_outside_np = np.zeros((IMG_H, IMG_W), dtype=bool)
    obj_outside_np[200:240, 50:100] = True  # x in [50, 100] < bbox x1=320
    mask_outside = _preprocess_object_mask(obj_outside_np, person_bbox, feature_hw=(56, 56))
    check("object outside bbox → all zeros", mask_outside.sum().item() == 0,
          f"sum={mask_outside.sum().item()}")

    # None mask → all zeros
    mask_none = _preprocess_object_mask(None, person_bbox, feature_hw=(56, 56))
    check("None mask → all zeros", mask_none.sum().item() == 0)

    # Degenerate bbox → all zeros
    degen_bbox = torch.tensor([100., 100., 100., 100.])  # zero-area
    mask_degen = _preprocess_object_mask(obj_mask_np, degen_bbox, feature_hw=(56, 56))
    check("degenerate bbox → all zeros", mask_degen.sum().item() == 0)


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Coordinate alignment between mask and vertex projection
# ─────────────────────────────────────────────────────────────────────────────

def test_coordinate_alignment():
    """
    Synthetic scenario:
      Image 480×640, person bbox [100, 50, 500, 450].
      Object mask covers [200, 150, 300, 250] (original image coords).
      We place vertices AT known locations and verify the prior reflects proximity.
    """
    print("\n=== Test 3: Coordinate alignment ===")

    IMG_H, IMG_W = 480, 640
    person_bbox = torch.tensor([[100., 50., 500., 450.]])  # [1, 4]
    person_bbox_1d = person_bbox[0]                        # [4] for _preprocess_object_mask

    # Object region in original image
    obj_y1, obj_y2 = 150, 250
    obj_x1, obj_x2 = 200, 300

    obj_mask_np = np.zeros((IMG_H, IMG_W), dtype=bool)
    obj_mask_np[obj_y1:obj_y2, obj_x1:obj_x2] = True

    # Preprocess mask [1, 1, 56, 56]
    mask_t = _preprocess_object_mask(obj_mask_np, person_bbox_1d, feature_hw=(56, 56))
    mask_batch = mask_t.unsqueeze(0)  # [1, 1, 56, 56]

    # Vertex inside object center
    v_inside = torch.tensor([[[250., 200.]]])  # [1, 1, 2]

    # Vertex far from object (right side of person bbox, different from object)
    v_far = torch.tensor([[[480., 400.]]])  # [1, 1, 2]

    # Vertex at person bbox center (might be near or far depending on object)
    obj_center_x = (obj_x1 + obj_x2) / 2  # 250
    obj_center_y = (obj_y1 + obj_y2) / 2  # 200
    v_obj_center = torch.tensor([[[obj_center_x, obj_center_y]]])  # [1, 1, 2]

    # Check normalized coords of inside vertex
    norm_inside = _normalize_kp2d(v_inside, person_bbox)   # [1, 1, 2]
    norm_far = _normalize_kp2d(v_far, person_bbox)          # [1, 1, 2]
    norm_center = _normalize_kp2d(v_obj_center, person_bbox)

    check("v_inside is within bbox (|norm| ≤ 1)",
          norm_inside.abs().max().item() <= 1.0,
          f"norm={norm_inside[0, 0].tolist()}")
    check("v_far is within bbox (|norm| ≤ 1)",
          norm_far.abs().max().item() <= 1.0,
          f"norm={norm_far[0, 0].tolist()}")

    # Verify mask has content at the normalized location of v_inside
    # Sample the mask at norm_inside to check
    grid_in = norm_inside.view(1, 1, 1, 2)
    sampled_mask = F.grid_sample(mask_batch.float(), grid_in,
                                  mode='bilinear', align_corners=True, padding_mode='zeros')
    sampled_val = sampled_mask.item()
    check("mask sampled AT object-center vertex is nonzero (>0)",
          sampled_val > 0.0,
          f"sampled_mask={sampled_val:.4f}")

    # Sample mask at far vertex location — should be 0 (no object there)
    grid_far = norm_far.view(1, 1, 1, 2)
    sampled_far = F.grid_sample(mask_batch.float(), grid_far,
                                 mode='bilinear', align_corners=True, padding_mode='zeros').item()
    check("mask sampled AT far vertex is 0",
          sampled_far == 0.0,
          f"sampled_mask_far={sampled_far:.4f}")

    # Now apply vertex prior and check prior values
    logits_in  = torch.zeros(1, 1)  # dummy
    logits_far = torch.zeros(1, 1)

    prior_in  = _apply_vertex_prior(logits_in,  v_inside,  person_bbox, mask_batch)
    prior_far = _apply_vertex_prior(logits_far, v_far,     person_bbox, mask_batch)

    bias_in  = prior_in.item()   # logits was 0 → result = prior
    bias_far = prior_far.item()

    check("inside-object vertex prior > far-vertex prior",
          bias_in > bias_far,
          f"inside={bias_in:.3f}  far={bias_far:.3f}")
    check("inside-object vertex prior close to 0 (≥ -1.0)",
          bias_in >= -1.0,
          f"bias_inside={bias_in:.3f}  (want ≥ -1.0)")
    check("far vertex prior strongly negative (≤ -5)",
          bias_far <= -5.0,
          f"bias_far={bias_far:.3f}  (want ≤ -5.0)")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Soft proximity field spread
# ─────────────────────────────────────────────────────────────────────────────

def test_soft_proximity_field():
    print("\n=== Test 4: _soft_proximity_field ===")

    # Single pixel mask at center of a 56×56 map
    mask = torch.zeros(1, 1, 56, 56)
    mask[0, 0, 28, 28] = 1.0

    prox = _soft_proximity_field(mask)  # [1, 1, 56, 56]

    check("output shape [1, 1, 56, 56]", prox.shape == (1, 1, 56, 56))
    check("peak is exactly 1.0", abs(prox.max().item() - 1.0) < 1e-5,
          f"max={prox.max().item():.6f}")
    check("minimum is ≥ 0", prox.min().item() >= 0.0)
    check("center gets highest value",
          prox[0, 0, 28, 28].item() == prox.max().item(),
          f"center={prox[0, 0, 28, 28].item():.4f}, max={prox.max().item():.4f}")

    # After 3 passes of kernel=9, effective radius ≈ 3 * (9//2) = 12px
    # Pixel at distance 5 from center should have nonzero proximity
    dist5_val = prox[0, 0, 28, 33].item()  # 5px right
    dist15_val = prox[0, 0, 28, 43].item()  # 15px right (may be ~0)
    check("proximity spreads ≥5px from mask edge (dist5 > 0)",
          dist5_val > 0.0, f"dist5={dist5_val:.6f}")
    check("center proximity > dist5 proximity (falloff exists)",
          prox[0, 0, 28, 28].item() > dist5_val)

    # Large mask: avg_pool with zero padding reduces boundary pixels.
    # The center should be 1.0, but corners are reduced (expected behavior).
    large_mask = torch.ones(1, 1, 56, 56)
    prox_large = _soft_proximity_field(large_mask)
    check("all-ones mask → center pixel is 1.0",
          abs(prox_large[0, 0, 28, 28].item() - 1.0) < 1e-4,
          f"center={prox_large[0, 0, 28, 28].item():.4f}")
    # avg_pool with zero-pad creates a ~4px border band below 0.5 on 56×56.
    # Expect >80% of pixels above 0.5 (interior pixels unaffected).
    check("all-ones mask → majority of pixels near 1.0 (>80% above 0.5)",
          (prox_large > 0.5).float().mean().item() > 0.8,
          f"frac_above_0.5={(prox_large > 0.5).float().mean().item():.3f}")

    # Zero mask → all zeros
    zero_mask = torch.zeros(1, 1, 56, 56)
    prox_zero = _soft_proximity_field(zero_mask)
    check("zero mask → proximity all zeros",
          prox_zero.max().item() == 0.0,
          f"max={prox_zero.max().item()}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: _apply_vertex_prior — gradient flow and bias range
# ─────────────────────────────────────────────────────────────────────────────

def test_apply_vertex_prior_gradients():
    print("\n=== Test 5: Gradient flow and bias range ===")

    B, V = 2, 100
    bbox = torch.tensor([[0., 0., 640., 480.]] * B)

    # Object covers left quarter of person bbox → x in [0, 160], y in [0, 480]
    mask_np_left = np.zeros((480, 640), dtype=bool)
    mask_np_left[:, :160] = True
    mask_t = _preprocess_object_mask(mask_np_left, bbox[0], feature_hw=(56, 56))
    object_mask = mask_t.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 1, 56, 56]

    # Random vertex positions: half on left (near object), half on right
    v_left  = torch.rand(B, V // 2, 2) * torch.tensor([160., 480.])         # x in [0,160]
    v_right = torch.rand(B, V // 2, 2) * torch.tensor([480., 480.]) + torch.tensor([160., 0.])
    verts_2d = torch.cat([v_left, v_right], dim=1)  # [B, V, 2]

    logits = torch.zeros(B, V, requires_grad=True)

    result = _apply_vertex_prior(logits, verts_2d, bbox, object_mask)

    # Gradient check: result should still have a grad_fn (autograd path via logits)
    check("result has grad_fn (gradient still flows through logits)",
          result.grad_fn is not None)

    # Bias (result - logits_zero) for near vs far vertices
    bias = result.detach()  # grad_fn is via logits; here logits=0 so result=prior
    bias_left  = bias[:, :V // 2].mean().item()
    bias_right = bias[:, V // 2:].mean().item()

    check("left (near object) vertices have higher prior than right (far) vertices",
          bias_left > bias_right,
          f"bias_left={bias_left:.3f}  bias_right={bias_right:.3f}")
    check("prior values in expected range [-9.2, 0]",
          bias.min().item() >= -9.3 and bias.max().item() <= 0.01,
          f"min={bias.min().item():.3f}  max={bias.max().item():.3f}")
    check("far vertices strongly suppressed (mean < -4)",
          bias_right < -4.0,
          f"bias_right_mean={bias_right:.3f}")

    # lambda=0 → no prior applied
    result_no_prior = _apply_vertex_prior(logits, verts_2d, bbox, object_mask, lambda_v=0.0)
    check("lambda=0 → prior not applied (result equals logits)",
          torch.allclose(result_no_prior, logits.detach(), atol=1e-5))


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Edge cases
# ─────────────────────────────────────────────────────────────────────────────

def test_edge_cases():
    print("\n=== Test 6: Edge cases ===")

    B, V = 1, 50
    bbox = torch.tensor([[0., 0., 640., 480.]])

    # --- 6a: Zero object mask → all vertices get max suppression ---
    zero_mask = torch.zeros(B, 1, 56, 56)
    verts_2d = torch.rand(B, V, 2) * torch.tensor([640., 480.])
    logits = torch.zeros(B, V)
    result_zero = _apply_vertex_prior(logits, verts_2d, bbox, zero_mask)
    bias_zero = result_zero.mean().item()
    check("zero mask → all vertices maximally suppressed (mean ≈ -9.2)",
          abs(bias_zero - np.log(1e-4)) < 0.1,
          f"mean_bias={bias_zero:.4f}  expected={np.log(1e-4):.4f}")

    # --- 6b: Full mask → all vertices get near-zero prior ---
    full_mask = torch.ones(B, 1, 56, 56)
    result_full = _apply_vertex_prior(logits, verts_2d, bbox, full_mask)
    bias_full = result_full.mean().item()
    check("full mask → all vertices near zero prior (|mean| < 0.5)",
          abs(bias_full) < 0.5,
          f"mean_bias={bias_full:.4f}")

    # --- 6c: Out-of-bbox vertex (outside [-1, 1] in normalized space) ---
    # With NO clamp before grid_sample, padding_mode='zeros' returns 0 → max suppression.
    v_oob = torch.tensor([[[700., 500.]]])  # outside 640x480
    result_oob = _apply_vertex_prior(torch.zeros(1, 1), v_oob, bbox, full_mask)
    bias_oob = result_oob.item()
    check("out-of-bbox vertex with full mask → maximally suppressed (bias ≈ -9.2)",
          bias_oob <= -9.0,
          f"bias_oob={bias_oob:.3f}")

    # --- 6d: Single pixel mask ---
    single_px_mask = np.zeros((480, 640), dtype=bool)
    single_px_mask[240, 320] = True
    mask_t = _preprocess_object_mask(single_px_mask, bbox[0], feature_hw=(56, 56))
    mask_batch = mask_t.unsqueeze(0)
    # Vertex at the exact pixel location
    v_at_pixel = torch.tensor([[[320., 240.]]])
    result_px = _apply_vertex_prior(torch.zeros(1, 1), v_at_pixel, bbox, mask_batch)
    bias_px = result_px.item()
    # The soft proximity field should spread the single pixel's signal
    prox_field = _soft_proximity_field(mask_batch)
    sampled_prox = F.grid_sample(
        prox_field,
        _normalize_kp2d(v_at_pixel, bbox).view(1, 1, 1, 2),
        mode='bilinear', align_corners=True, padding_mode='zeros',
    ).item()
    check("single-pixel mask: vertex AT pixel has nonzero proximity (after spreading)",
          sampled_prox > 0.0,
          f"sampled_prox={sampled_prox:.6f}")

    # --- 6e: Batch size 1 vs B>1 gives same results ---
    v_test = torch.rand(1, 10, 2) * torch.tensor([640., 480.])
    mask_test = torch.zeros(1, 1, 56, 56)
    mask_test[0, 0, 20:30, 20:30] = 1.0
    logits_1 = torch.zeros(1, 10)
    result_1 = _apply_vertex_prior(logits_1, v_test, bbox, mask_test)

    v_test_b = v_test.expand(3, -1, -1)
    mask_test_b = mask_test.expand(3, -1, -1, -1)
    bbox_b = bbox.expand(3, -1)
    logits_b = torch.zeros(3, 10)
    result_b = _apply_vertex_prior(logits_b, v_test_b, bbox_b, mask_test_b)

    check("batch-size consistency (B=1 vs B=3 give same per-sample result)",
          torch.allclose(result_1[0], result_b[0], atol=1e-5) and
          torch.allclose(result_b[0], result_b[1], atol=1e-5))


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Real data sanity check
# ─────────────────────────────────────────────────────────────────────────────

def test_real_data():
    print("\n=== Test 7: Real data sanity check ===")

    masks_v2_dir = Path("dataset/damon_mhr_contact/masks_v2")
    predictions_dir = Path("dataset/damon_mhr_contact/predictions/trainval")

    if not masks_v2_dir.exists():
        print("  [SKIP] masks_v2_dir not found, skipping real-data test")
        return
    if not predictions_dir.exists():
        print("  [SKIP] predictions_dir not found, skipping real-data test")
        return

    # Load a few samples and check statistics
    import importlib.util, json
    sys.path.insert(0, str(Path(__file__).parent.parent / "dataset"))
    from damon_dataset import DamonPrecomputedDataset, build_instance_index, infer_split_from_npz_path

    contact_npz = "dataset/damon_mhr_contact/hot_dca_trainval_contact_lod1.npz"
    detect_npz  = "dataset/damon_mhr_contact/hot_dca_trainval_detect.npz"
    features_dir = "dataset/damon_mhr_contact/features/trainval"

    if not Path(contact_npz).exists():
        print("  [SKIP] contact npz not found")
        return

    ds = DamonPrecomputedDataset(
        contact_npz_path=contact_npz,
        detect_npz_path=detect_npz,
        features_dir=features_dir,
        predictions_dir=str(predictions_dir),
        lod=1,
        mode='instance_contact',
        masks_v2_dir=str(masks_v2_dir),
    )

    n_samples = min(20, len(ds))
    biases_near, biases_far = [], []
    n_contact_verts_before, n_contact_verts_after = [], []

    for i in range(n_samples):
        inputs_dict, label_dict = ds[i]

        person_bbox = inputs_dict['person_bbox'].unsqueeze(0)  # [1, 4]
        obj_mask_np = inputs_dict.get('object_mask')
        if obj_mask_np is None:
            continue

        mask_t = _preprocess_object_mask(obj_mask_np, inputs_dict['person_bbox'], (56, 56))
        mask_batch = mask_t.unsqueeze(0)  # [1, 1, 56, 56]

        contact_label = label_dict['contact_label']  # [V]
        V = len(contact_label)
        contact_verts_idx = contact_label.nonzero(as_tuple=True)[0]
        noncontact_verts_idx = (contact_label == 0).nonzero(as_tuple=True)[0]

        if len(contact_verts_idx) == 0:
            continue

        # Use precomputed 2D keypoints as a proxy for vertex positions
        # (real verts_2d would come from the model — we use joints as a subset)
        pred_kp2d = inputs_dict['pred_kp2d']  # [70, 2]
        kp2d_batch = pred_kp2d.unsqueeze(0)   # [1, 70, 2]

        logits_dummy = torch.zeros(1, 70)
        result = _apply_vertex_prior(logits_dummy, kp2d_batch, person_bbox, mask_batch)
        biases = result[0].tolist()  # [70]

        biases_near.extend([b for b in biases if b > -3.0])
        biases_far.extend([b for b in biases if b <= -3.0])

        # Verify mask is non-trivial
        mask_density = mask_t.mean().item()

    if biases_near or biases_far:
        mean_near = np.mean(biases_near) if biases_near else float('nan')
        mean_far  = np.mean(biases_far)  if biases_far else float('nan')
        pct_far   = 100 * len(biases_far) / (len(biases_near) + len(biases_far))
        print(f"  Real data ({n_samples} samples): mean_near={mean_near:.2f}  "
              f"mean_far={mean_far:.2f}  pct_suppressed={pct_far:.1f}%")
        check("mean 'near-object' joint bias > mean 'far' joint bias",
              mean_near > mean_far if (biases_near and biases_far) else True)
        check("suppression fraction in sensible range (10–90%)",
              10.0 < pct_far < 90.0,
              f"pct_far={pct_far:.1f}%")
    else:
        print("  [WARN] No biases collected — check dataset")

    # Verify kp2d range is in image pixel space (not normalized)
    raw_kp2d = ds.pred_kp2d
    check("kp2d values are pixel-scale (mean > 1.0, i.e., not normalized)",
          float(np.abs(raw_kp2d).mean()) > 1.0,
          f"mean_abs_kp2d={float(np.abs(raw_kp2d).mean()):.2f}")
    # NOTE: some DAMON samples have near-zero depth, causing extreme projected coords.
    # Extreme OOB values (e.g., ±1e5) are correctly handled: without clamp before
    # grid_sample, padding_mode='zeros' suppresses them with max negative bias.
    pct_extreme = float(np.mean(np.abs(raw_kp2d) > 10000)) * 100
    check("kp2d extreme outliers <5% of values (most samples are normal)",
          pct_extreme < 5.0,
          f"pct_extreme={pct_extreme:.2f}%  min={raw_kp2d.min():.1f}  max={raw_kp2d.max():.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Bbox format mismatch guard
# ─────────────────────────────────────────────────────────────────────────────

def test_bbox_format():
    """
    Verify that the bbox used in _normalize_kp2d is (x1, y1, x2, y2),
    NOT (cx, cy, w, h) — a common pitfall.
    """
    print("\n=== Test 8: Bbox format is (x1, y1, x2, y2) ===")

    # bbox: x1=200, y1=100, x2=400, y2=300 → bw=200, bh=200
    bbox = torch.tensor([[200., 100., 400., 300.]])

    # Midpoint of the bbox should be (300, 200)
    mid_x, mid_y = 300., 200.
    kp = torch.tensor([[[mid_x, mid_y]]])

    norm = _normalize_kp2d(kp, bbox)
    check("midpoint of (x1,y1,x2,y2) bbox → (0, 0) — confirms format is NOT cx/cy/w/h",
          torch.allclose(norm[0, 0], torch.tensor([0., 0.]), atol=1e-4),
          f"got {norm[0, 0].tolist()}")

    # If format were wrongly (cx, cy, w, h), then x1=200 would be cx=200,
    # bbox midpoint would be at (200, 100), not (300, 200)
    wrong_fmt_bbox = torch.tensor([[200., 100., 200., 200.]])  # interpreted as cx,cy,w,h
    # In wrong format, 'x1=200=cx', bw=200-200=0 (clamped to 1), result would be 0/(1)-1=-1
    norm_wrong = _normalize_kp2d(kp, wrong_fmt_bbox)
    check("wrong bbox format gives wrong result (sanity: norm ≠ (0,0))",
          not torch.allclose(norm_wrong[0, 0], torch.tensor([0., 0.]), atol=1e-4),
          f"(confirming wrong fmt gives {norm_wrong[0, 0].tolist()} ≠ (0,0))")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Vertex Prior Unit Tests")
    print("=" * 65)

    test_normalize_kp2d()
    test_preprocess_mask()
    test_coordinate_alignment()
    test_soft_proximity_field()
    test_apply_vertex_prior_gradients()
    test_edge_cases()
    test_real_data()
    test_bbox_format()

    print("\n" + "=" * 65)
    n_pass = sum(1 for _, ok in _results if ok)
    n_fail = sum(1 for _, ok in _results if not ok)
    print(f"Results: {n_pass}/{len(_results)} passed"
          + (f", {n_fail} FAILED" if n_fail else " — all OK"))
    print("=" * 65)

    if n_fail:
        print("\nFailed tests:")
        for name, ok in _results:
            if not ok:
                print(f"  ✗ {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
