# Step 6: Object Position Prediction

**Duration**: ~1-2 weeks
**Depends on**: Step 3 (correspondence vectors), Step 2 (object mask prompts)
**Goal**: Predict object 6DoF pose and scale relative to the human body, using contact as the bridge

---

## Motivation

Contact tells us *where* the body touches the object. Correspondence vectors (from Step 3) tell us *which direction* the object is relative to the body. The next step is predicting the full object 6DoF pose and scale relative to the body.

PICO does this via optimization (slow, fragile). LEMON predicts object center position but not full 6DoF. No feed-forward model jointly predicts contact + object pose from a single image.

**Key insight from PICO**: Contact correspondences are the strongest cue for object placement. Body vertex v contacts object point p -> the vector (v -> p) constrains where the object must be. With enough contact correspondences, object pose is highly constrained.

---

## Architecture

```
                    +------------------+
                    |   SAM3DB Encoder |
                    +--------+---------+
                             |
                        Image Features F
                             |
              +--------------+------------------+
              |              |                  |
     +--------v--------+    |         +--------v---------+
     |  Body Decoder    |    |         | Interaction      |
     |                  |    |         | Decoder          |
     +--------+---------+    |         |                  |
              |              |         | T_contact (K=16) |
         Body Mesh           |         | T_object (2-4)   | <-- expanded
                             |         | T_scene (2-4)    |
                             |         |                  |
           +------+          |         | Cross-Attn to:   |
           | SAM  |--mask--->|-------->|  - Image F       |
           +------+          |         |  - Obj prompts   |
                             |         |  - SAM3D tokens  | <-- NEW
           +------+          |         |                  |
           |SAM3D |--tokens->|-------->|                  | <-- NEW
           +------+          |         +-----+----+-------+
                             |               |    |
                             |          Contact   Object Pose Head
                             |          Head      (NEW)
                             |                    |
                             |              +-----v------+
                             |              | 6DoF + Scl |
                             |              | relative   |
                             |              | to pelvis  |
                             |              +------------+
```

---

## New Components

### SAM 3D Object Token Prompt Encoder
- Input: SAM 3D object tokens (latent representation encoding object geometry + appearance)
- Encoding: Linear projection to d_model
- These tokens provide 3D object shape/appearance understanding without explicit mesh

### Object Point Cloud Prompt Encoder (Optional)
- Input: Object point cloud from SAM 3D output (N points x 3)
- Encoding: Mini PointNet (3 FC layers: 3->64->128->d_model, with max-pool)
- Provides explicit 3D geometry when available

### Object Pose Head
From object query tokens after decoding:

| Output | Representation | Dim |
|--------|---------------|-----|
| Rotation | 6D continuous rotation (Zhou et al.) | 6 |
| Translation | 3D offset from human pelvis | 3 |
| Scale | Log-scale (isotropic or 3D anisotropic) | 1 or 3 |
| Confidence | Sigmoid | 1 |

**All relative to the human pelvis joint**, not absolute camera coordinates. This makes the prediction invariant to camera position.

### Object Contact Head (Optional)
If object geometry is provided as prompt (point cloud or SAM3D tokens), predict contact on the object surface:

| Output | Description |
|--------|------------|
| Object contact probability | Per-point on object point cloud |
| Affordance type | Per-point classification (17 IAG categories) |

### Loss
```
L_obj = L_rot + L_trans + L_scale + lambda_pen * L_penetration + lambda_chamfer * L_chamfer

L_rot:         L1 on 6D rotation representation
L_trans:       L1 on pelvis-relative translation
L_scale:       L1 on log-scale
L_penetration: SDF-based body-object interpenetration penalty (from PICO)
               Penalize object vertices that are inside the body SDF
L_chamfer:     Chamfer distance between placed object and GT mesh (if GT available)
```

**Contact-pose consistency loss** (new):
```
L_consistency = sum over contacting vertices v of ||v + delta(v) - T(p_obj)||^2

Where:
  v:        body contact vertex
  delta(v): predicted correspondence vector (from Step 3)
  T(p_obj): object point after applying predicted 6DoF + scale
  p_obj:    nearest object surface point

This loss enforces that the correspondence vectors agree with the object placement.
```

---

## Training

**What trains**: SAM3D prompt encoder + object pose head + object contact head
**What's frozen**: Encoder (or lightly unfrozen). Body decoder frozen. Interaction decoder finetuned.

**Data**:

| Dataset | Object 6DoF GT | Object Meshes | Contact GT | Size |
|---------|---------------|---------------|------------|------|
| BEHAVE | Fitted object 6DoF | 20 object templates | Proximity-based | 15K frames |
| InterCap | Fitted object 6DoF | 10 categories | Proximity-based | Multi-view |
| 3DIR (LEMON) | Object center position + spatial relation | Retrieved meshes (21 classes) | Contact + affordance | 5K images |
| PICO-db | Not directly (optimization-based) | Retrieved meshes | Correspondences | 4.1K images |

**Object mesh for training**: BEHAVE and InterCap provide GT object meshes and 6DoF poses per frame. 3DIR provides spatial relations. Use BEHAVE + InterCap as primary.

**Training modes**:
1. **With SAM3D object tokens** (30%): Full geometry understanding
2. **With object mask only** (40%): Mask from SAM
3. **With object mask + point cloud** (20%): Explicit geometry
4. **No object prompt** (10%): Model predicts from contact alone

---

## Evaluation

| Metric | Dataset | Comparison |
|--------|---------|------------|
| Rotation error (deg) | BEHAVE, InterCap | vs. PICO-fit |
| Translation error (cm) | BEHAVE, InterCap | vs. PICO-fit |
| Scale error | BEHAVE, InterCap | vs. PICO-fit |
| Chamfer distance (placed object) | BEHAVE, InterCap | vs. PICO-fit, LEMON |
| Object contact F1 | 3DIR, PIAD | vs. InteractVLM, LEMON |
| Contact correspondence error | PICO-db | vs. PICO nearest-neighbor |

**Key comparison**: PICO-fit (optimization-based, uses DECO + OpenShape + GPT-4V + SAM). Our model should be competitive on metrics while being orders of magnitude faster.

---

## Ablations

1. With SAM3D tokens vs. mask-only vs. no object prompt
2. Contact-pose consistency loss vs. without
3. Correspondence vectors -> object pose vs. direct pose prediction
4. Object contact prediction with/without object geometry
5. Pelvis-relative vs. camera-relative object pose

---

## Success Criteria

- Rotation error < 30 deg on BEHAVE (competitive with PICO-fit)
- Translation error < 15cm on BEHAVE
- Chamfer distance competitive with PICO-fit
- Object placement is physically plausible (no major penetration or floating)
- Speed: < 200ms total (vs. PICO-fit's minutes)

---

## References

- PICO (Cseke 2025): 3-stage optimization for object placement, body-object correspondences
- LEMON (Yang 2024): Object center position prediction, 3DIR dataset, 21 object classes
- SAM 3D (Team 2025): Object tokens (DINOv2 crop+mask), voxel geometry, layout prediction
- InteractVLM (Dwivedi 2025): Object affordance from VLM, object mesh from OpenShape
- IAG (Yang 2023): 3D object affordance grounding, PIAD dataset, 17 affordance types
- BEHAVE (Bhatnagar 2022): 15K frames, 20 objects, GT 6DoF, proximity-based contact
