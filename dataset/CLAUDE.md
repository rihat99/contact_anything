# dataset/ — Data Processing Pipeline

## Source Dataset: DAMON / DECO

**Location:** `/data3/rikhat.akizhanov/DECO/datasets/Release_Datasets/damon/`

| File | Samples | Description |
|------|---------|-------------|
| `hot_dca_trainval.npz` | 4,384 | Full training+val split with all annotations |
| `hot_dca_test.npz` | 785 | Test split |

**Key keys in source NPZ:**

| Key | Shape | Description |
|-----|-------|-------------|
| `imgname` | (N,) str | Path relative to `/data3/rikhat.akizhanov/DECO/datasets/` |
| `contact_label` | (N, 6890) float64 | Binary per-vertex contact, SMPL topology |
| `contact_label_objectwise` | (N,) object | Dict per sample: `{object_name: [smpl_vertex_indices]}` |
| `contact_label_smplx` | (N, 10475) float32 | Same in SMPLx topology |
| `cam_k` | (N, 3, 3) float64 | Camera intrinsics (focal length ~800px) |
| `pose` | (N, 72) float32 | SMPL pose parameters |
| `shape` | (N, 10) float32 | SMPL shape parameters |

**Contact objects:** 61 COCO categories + `supporting` (ground/walls). `supporting` is skipped in segmentation. Mean ~2.5 objects per sample; 77% of samples have exactly one contact object + optional `supporting`.

---

## Processed Data: `damon_mhr_contact/`

All derived files live here. **Never regenerate the detect/contact NPZs unless the source changes** — they took significant compute.

### Ready-to-use NPZ files

| File | Keys | Description |
|------|------|-------------|
| `hot_dca_trainval_detect.npz` | imgname (4384,), bbox (4384,4), cam_k (4384,3,3) | Person bboxes (ViTDet) + camera intrinsics (MoGe2) |
| `hot_dca_test_detect.npz` | same, (785,) | Test split |
| `hot_dca_trainval_contact_lod1.npz` | imgname (4384,), contact_label (4384, **18439**) | Contacts converted to MHR LOD1 topology |
| `hot_dca_test_contact_lod1.npz` | same, (785,) | Test split |
| `hot_dca_trainval_contact_lod6.npz` | imgname (4384,), contact_label (4384, **595**) | LOD6 (low-res, for ablations) |
| `hot_dca_test_contact_lod6.npz` | same | |

The contact LOD1 files are what the training pipeline uses (`DATASET.CONTACT_NPZ` in config).

### Precomputed features & predictions

```
features/{split}/{idx:04d}.pt        # DINOv3-H encoder output [1280, 56, 56] float16
predictions/{split}/{idx:04d}.npz    # Per-sample pose predictions
```

**Predictions NPZ keys** (per sample):

| Key | Shape | Description |
|-----|-------|-------------|
| `pred_keypoints_2d` | (70, 2) | MHR 70-joint 2D positions in original image pixels |
| `pred_keypoints_3d` | (70, 3) | 3D positions in camera space |
| `pred_cam_t` | (3,) | Camera translation |
| `focal_length` | scalar | Focal length in pixels |
| `body_pose_params` | (130,) | Body pose |
| `hand_pose_params` | (108,) | Hand pose |
| `shape_params` | (45,) | Body shape |
| `scale_params` | (28,) | Scale |
| `mhr_model_params` | (195,) | Combined MHR params |
| `pred_joint_coords` | (127, 3) | Full joint set |
| `bbox_used` | (4,) | Bbox used for this prediction |
| `global_rot` | (3,) | Global rotation (axis-angle) |

Both trainval (4,384) and test (785) are complete.

### SAM3 segmentation masks

Two versions exist — **v2 is current**:

```
masks/     ← v1 (legacy, index-based names, no contact vertex info)
masks_v2/  ← v2 (current, name-based, full contact metadata)
```

**v2 structure per sample** (`masks_v2/{split}/{idx:04d}/`):

```
metadata.npz
masks/
  {idx:04d}_person_0.png
  {idx:04d}_{object_name}_{det_idx}.png
  ...
```

**v2 metadata.npz keys** (additive over v1):

| Key | Description |
|-----|-------------|
| `imgname` | Image path relative to data_root |
| `object_names` | Array of object names, index 0 always = `person` |
| `num_detections` | Number of SAM3 detections per object |
| `bboxes_{i}` | float32 (n_det, 4) bounding boxes for object i |
| `scores_{i}` | float32 (n_det,) SAM3 confidence scores |
| `contact_vertices_smpl_{i}` | int64 array of SMPL vertex indices in contact |
| `contact_vertices_mhr_{i}` | int64 array of MHR LOD1 vertex indices in contact |
| `contact_body_parts_{i}` | int array of SMPL body part ids (0–23) |
| `best_detection_{i}` | int, disambiguated best detection index (-1 if no detections) |
| `disambiguation_scores_{i}` | float32 (n_det,) combined disambiguation scores |
| `metadata_version` | 2 |

**Disambiguation scoring** (for multi-detection objects):
- 30% — person-mask pixel overlap + adjacency (10px dilation)
- 50% — 2D joint proximity (contact SMPL parts → MHR keypoints → pred_kp2d)
- 20% — SAM3 confidence score

### Visualisation output

```
test_masks/{split}/{idx:04d}.jpg   # mask overlay on original image
```
Blue = person, Green = best-detection contacted object, Red = non-contact or non-best detections.

---

## Scripts

### Data preparation (run once, in order)

| Script | Purpose | Runtime |
|--------|---------|---------|
| `convert_damon.py` | SMPL (6890) → MHR LOD-N contact labels via barycentric interpolation. Requires SMPL_NEUTRAL.npz | ~minutes |
| `damon_append.py` | Produce detect NPZ: ViTDet human bboxes + MoGe2 camera intrinsics | ~hours |
| `split_existing_npz.py` | One-time migration: split combined MHR NPZ into separate contact + detect files | seconds |

### Precomputation (run once per split, resumable)

| Script | Purpose | Runtime |
|--------|---------|---------|
| `damon_sam3d_precompute.py` | Run SAM-3D-Body on each image → save DINOv3 features `.pt` + pose predictions `.npz` per sample | ~hours (GPU) |
| `damon_sam3_segment.py` | Run SAM3 open-vocabulary segmentation → save person + contact object masks + metadata | ~hours (GPU) |

### Dataset classes

All dataset logic lives in two files:

| File | Contents |
|------|----------|
| `damon_utils.py` | Shared utilities: mask loading, instance index building, label conversion, split helpers, `load_smpl_part_segmentation`, `part_contact_from_vertex_label` |
| `damon_dataset.py` | `DamonDataset`, `DamonPrecomputedDataset`, `build_datasets(cfg)` |

#### `DamonDataset`

Raw-image dataset. `topology` and `mode` are constructor parameters.

```python
DamonDataset(
    contact_npz_path,           # 'hot_dca_{split}_contact_lod1.npz' (MHR) or source DAMON NPZ (SMPL)
    detect_npz_path=None,       # 'hot_dca_{split}_detect.npz'
    topology='mhr',             # 'mhr' | 'smpl'
    lod=1,                      # MHR LOD 0–6; ignored for SMPL
    mode='classic',             # see modes below
    masks_v2_dir=None,          # required for instance modes
    data_root=None,             # image root; defaults to DAMON_DATA_ROOT env var
    transform=None,
    smpl_part_seg_path=None,    # SMPL vertex segmentation JSON; enables per-body-part labels
                                # only valid with topology='smpl'
)
```

**Per-body-part contact labels** (`smpl_part_seg_path`): when provided with `topology='smpl'`, each item
also returns a 24-dim binary tensor indicating which SMPL body parts are in contact. Computed on the fly
from the per-vertex label — no precomputation needed.

- Segmentation NPY path: `/data3/rikhat.akizhanov/human_global_motion/better_human/src/better_human/smpl/config/smpl_3d_segmentation.npy`
  (key `body_vertices` → dict of int part id 0–23 → vertex index list; one sentinel index ≥ 6890 per part is filtered automatically)
- 24 parts in **SMPL joint order** (index = part id): `hips(0), leftUpLeg(1), rightUpLeg(2), spine(3), leftLeg(4), rightLeg(5), spine1(6), leftFoot(7), rightFoot(8), spine2(9), leftToeBase(10), rightToeBase(11), neck(12), leftShoulder(13), rightShoulder(14), head(15), leftArm(16), rightArm(17), leftForeArm(18), rightForeArm(19), leftHand(20), rightHand(21), leftHandIndex1(22), rightHandIndex1(23)`
- Part names also exported as `damon_utils.SMPL_PART_NAMES` (list, index = part id)
- Part i is active (=1) if **any** of its vertices has contact
- **Classic mode**: label becomes a dict `{'contact_label': tensor[6890], 'part_contact': tensor[24]}`
  (without `smpl_part_seg_path`, label remains a flat tensor — backward compatible)
- **Instance modes**: `'part_contact': tensor[24]` added to `label_dict`

```python
# Example
ds = DamonDataset(
    "hot_dca_trainval.npz",   # source SMPL NPZ
    topology='smpl',
    smpl_part_seg_path="/data3/.../smpl_vert_segmentation.json",
)
inputs, labels = ds[0]
labels['contact_label']   # int64 [6890] per-vertex
labels['part_contact']    # int64 [24]   per-body-part
```

#### `DamonPrecomputedDataset`

Loads precomputed DINOv3 features (`.pt`) instead of raw images. Always MHR topology. Used by the training pipeline to skip the frozen backbone.

```python
DamonPrecomputedDataset(
    contact_npz_path, detect_npz_path, features_dir,
    predictions_npz_path=None,  # merged [N,70,2] / [N,70,3] predictions
    lod=1, mode='classic', masks_v2_dir=None, data_root=None,
)
```

#### Operation modes

| Mode | Index | Return format |
|------|-------|---------------|
| `classic` | one item per image | `(image_or_feature, bbox, cam_k[, img_size, kp2d, kp3d]), contact_label[V]` |
| `instance_contact` | one per (image, contact-object) | `inputs_dict, label_dict` — only objects with ≥1 contact vertex + valid detection |
| `instance_all` | one per (image, any-object) | `inputs_dict, label_dict` — all non-person objects; zeros label if no contact |

Instance mode `inputs_dict` keys: `image` (or `feature`), `person_mask`, `object_mask`, `person_bbox`, `object_bbox`, `cam_k`
Instance mode `label_dict` keys: `contact_label [V]`, `object_name` (str), `has_contact` (bool), `part_contact [24]` *(SMPL only, when `smpl_part_seg_path` set)*

> `object_name` is a string — use a custom `collate_fn` when batching instance-mode items with DataLoader.

Both classes share `split_train_val(...)` classmethod that always splits at **image level** (deterministic, seed=42), returning `(train_Subset, val_Subset)`. Instance modes produce a `Subset` with flat indices into the full instance index.

`build_datasets(cfg)` in `damon_dataset.py` is the config-driven factory used by the training pipeline (returns `train, val, test`).

### Utilities

| Script | Purpose |
|--------|---------|
| `visualize_masks.py` | Overlay v2 masks on original images with colour-coded contact status |
| `example_damon_mhr.py` | Usage example / sanity check (uses `DamonDataset`) |
| `test_smpl_part_contact.py` | Tests for per-body-part contact labels (17 tests) |

---

## Key Commands

```bash
PYTHON=/data3/rikhat.akizhanov/miniconda3/envs/sam3d/bin/python

# Convert SMPL contacts → MHR LOD1
$PYTHON dataset/convert_damon.py \
  --input_path /data3/rikhat.akizhanov/DECO/datasets/Release_Datasets/damon/hot_dca_trainval.npz \
  --output_path dataset/damon_mhr_contact/hot_dca_trainval_contact_lod1.npz \
  --smpl_model_path /data3/rikhat.akizhanov/human_global_motion/better_human/models/smpl/SMPL_NEUTRAL.npz

# Precompute features + predictions (resumable)
CUDA_VISIBLE_DEVICES=0 $PYTHON dataset/damon_sam3d_precompute.py --resume --splits trainval test

# Generate SAM3 masks v2 (requires predictions to be done first)
CUDA_VISIBLE_DEVICES=0 $PYTHON dataset/damon_sam3_segment.py --resume --splits trainval test

# Visualise masks for a split
$PYTHON dataset/visualize_masks.py --split trainval --end_idx 100
```

---

## Data Flow

```
Source DAMON NPZ (SMPL 6890-vert contacts, raw images)
        │
        ├─ convert_damon.py ──────────────► hot_dca_{split}_contact_lod1.npz  (MHR 18439 verts)
        │
        ├─ damon_append.py ───────────────► hot_dca_{split}_detect.npz  (bbox, cam_k)
        │
        └─ damon_sam3d_precompute.py ─────► features/{split}/*.pt  +  predictions/{split}/*.npz
                                                    │
                                                    └─ damon_sam3_segment.py ──► masks_v2/{split}/*
                                                                                        │
                                                                                        └─ visualize_masks.py → test_masks/
```

Training uses `DamonPrecomputedDataset` (classic mode) which reads:
- `hot_dca_{split}_contact_lod1.npz` — MHR LOD1 contact labels
- `hot_dca_{split}_detect.npz` — bboxes + camera intrinsics
- `features/{split}/*.pt` — precomputed DINOv3 features
- `predictions/{split}/*.npz` — precomputed 2D/3D keypoints (optional, for InteractionDecoder)

Instance modes additionally read `masks_v2/{split}/{idx:04d}/` for per-object masks and contact vertex lists.

---

## SMPL → MHR Vertex Mapping

Conversion uses `mhr_smpl_conversion/body_converter.py` with precomputed barycentric mapping files in `mhr_smpl_conversion/assets/`. No SMPL forward pass needed at inference — only SMPL face connectivity (`SMPL_NEUTRAL.npz` key `f`, shape `[13776, 3]`).

SMPL part segmentation (used for contact part labels and disambiguation):
- `/data3/rikhat.akizhanov/human_global_motion/better_human/src/better_human/smpl/config/smpl_3d_segmentation.npy`
- Key `body_vertices`: dict int→list, 24 parts in SMPL joint order (see `SMPL_PART_NAMES` in `damon_utils.py`)
- Loaded via `load_smpl_part_segmentation(npy_path)` in `damon_utils.py`
