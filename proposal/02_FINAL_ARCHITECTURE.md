# SAM 3D Contact: Target Architecture

## Overview

The final architecture extends SAM 3D Body with three new components: (1) an interaction decoder for contact, (2) prompt encoders for objects and scenes, and (3) cross-attention bridges between body decoder and interaction decoder for mutual refinement.

```
                              +------------------+
                              |   Input Image    |
                              +--------+---------+
                                       |
                          +------------v-----------+
                          |    SAM3DB Encoder       |
                          |    (ViT-H / DINOv3)    |
                          |    frozen or lightly    |
                          |    finetuned            |
                          +------------+-----------+
                                       |
                              Image Features F
                                       |
             +------------+------------+------------+------------+
             |            |            |            |            |
        +----v----+  +----v----+  +----v----+  +----v----+  +---v----+
        |  Body   |  |  Hand   |  | Contact |  | Object  |  | Scene  |
        |  Query  |  |  Query  |  |  Query  |  |  Query  |  | Query  |
        | Tokens  |  | Tokens  |  | Tokens  |  | Tokens  |  | Tokens |
        | (pose)  |  | (hands) |  | (K=16)  |  | (obj)   |  | (scn)  |
        +----+----+  +----+----+  +----+----+  +----+----+  +---+----+
             |            |            |            |            |
             |            |            |       +----v----+  +---v----+
             |            |            |       | Obj Mask|  | Depth/ |
             |            |            |       | SAM3D   |  | Floor  |
             |            |            |       | Tokens  |  | Plane  |
             |            |            |       +---------+  +--------+
             |            |            |            |            |
        +----v-----------v----+  +----v-----------v-----------v----+
        |   Body Decoder      |  |     Interaction Decoder          |
        |   (from SAM3DB)     |  |     (new)                       |
        |                     |  |                                  |
        | - Self-attention    |  | - Self-attention on contact,     |
        | - Cross-attn to F   |  |   object, scene tokens          |
        | - Body param heads  |  | - Cross-attn to F               |
        |                     |  | - Contact/object/scene heads     |
        +----+-------+--------+  +----+-------+-------+------------+
             |       |                |       |       |
             |       +<-- cross-attn->+       |       |      <-- iterative (Step 7)
             |                        |       |       |
        +----v----+          +--------v--+ +--v---+ +-v--------+
        | Body    |          | Contact   | | Obj  | | Scene    |
        | Mesh    |          | Field     | | 6DoF | | Contact  |
        | (MHR)   |          | + Corr    | | +Scl | | (floor,  |
        +---------+          +----------+ +------+ | wall)    |
                                                    +----------+
```

---

## Component Details

### A. Encoder (Inherited from SAM3DB)

**Backbone**: ViT-H (632M) or DINOv3 (840M), producing dense feature map F from 512x512 human crop. Stays frozen in early stages, lightly finetuned later.

**Note on dual-stream**: The old proposal considered a separate DINOv2 encoder for scene semantics. This may be unnecessary since SAM3DB already uses DINOv3 which has strong scene understanding. **Ablate in Step 2** -- if object-conditioned contact does not improve with SAM3DB features alone, add DINOv2 stream.

---

### B. Body Decoder (Inherited from SAM3DB)

Predicts MHR/SMPL-X body parameters: pose P, shape S, camera C, skeleton S_k. Cross-attends to image features F. This decoder is well-trained on ~7M images -- do not destroy it.

In the final architecture (Step 7), it also receives cross-attention from interaction decoder tokens, so contact predictions can refine pose.

---

### C. Interaction Decoder (New -- Core Contribution)

A transformer decoder that processes contact, object, and scene query tokens.

**Token types**:

| Token Type | Count | Purpose |
|-----------|-------|---------|
| Contact query tokens | K=16 | Decode into contact patches on body |
| Object query tokens | 1-4 | Aggregate object information, predict object pose |
| Scene query tokens | 2-4 | Aggregate scene information (floor, walls) |

**Decoder layers** (each iteration):
```
1. Self-attention:   all interaction tokens attend to each other
2. Cross-attn to F:  interaction tokens attend to image features
3. Cross-attn to prompts: interaction tokens attend to prompt tokens (obj mask, depth, etc.)
4. [Step 7] Cross-attn to body: interaction tokens attend to body decoder intermediate tokens
```

**Output heads** (MLPs applied to decoded tokens):

| Head | Input Tokens | Output | Loss |
|------|-------------|--------|------|
| Contact head | Contact tokens | Per-vertex contact field (CDF + correspondence) | L1 + BCE |
| Semantic head | Contact tokens | Contact type + target type per token | Cross-entropy |
| Object pose head | Object tokens | 6DoF (6D rotation + translation) + scale | L1 |
| Scene contact head | Scene tokens | Floor contact vertices + floor plane | L1 + BCE |

---

### D. Prompt Encoders

| Prompt | Encoder | Source | Added In |
|--------|---------|--------|----------|
| Human mask | SAM3DB mask encoder (inherited) | SAM / manual | Step 1 |
| 2D keypoints | SAM3DB prompt encoder (inherited) | Detection / manual | Step 1 |
| Object mask | Same mask encoder, different type embedding | SAM | Step 2 |
| Object category | Learned embedding (N classes) | Classifier / manual | Step 2 |
| Depth map | Lightweight CNN (4 conv layers) | DepthPro / MoGe | Step 4 |
| Floor plane | Linear projection (4D -> d_model) | RANSAC on depth | Step 4 |
| SAM 3D object tokens | Linear projection | SAM 3D | Step 6 |
| Object point cloud | Mini PointNet (3 layers) | SAM 3D output | Step 6 |

All prompt tokens are projected to d_model dimension and concatenated with the interaction decoder's query tokens.

---

### E. Cross-Attention Bridge (Step 7)

The key architectural contribution. Lightweight cross-attention between body decoder and interaction decoder at each iteration.

```
Body Decoder Iteration t:
  body_t = BodySelfAttn(body_{t-1})
  body_t = BodyCrossAttn(body_t, F)
  body_t = BodyCrossAttn(body_t, contact_t)    <-- NEW: contact informs pose

Interaction Decoder Iteration t:
  inter_t = InterSelfAttn(inter_{t-1})
  inter_t = InterCrossAttn(inter_t, F)
  inter_t = InterCrossAttn(inter_t, body_t)    <-- NEW: pose informs contact
```

**Design constraints**:
- Single-head cross-attention with small projection (d=64) to avoid slowing body decoder
- Gradient stopping from interaction->body in early training stages
- Deep supervision: losses at every iteration

---

## Training Curriculum (Staged)

| Stage | What Trains | What's Frozen | Data |
|-------|-------------|---------------|------|
| Step 1 | Interaction decoder + contact head | Encoder + body decoder | DAMON + HOT |
| Step 2 | + Object mask prompt encoder | Encoder + body decoder | DAMON + obj masks |
| Step 3 | + New contact representation heads | Encoder + body decoder | DAMON + BEHAVE + PICO-db |
| Step 4 | + Scene prompt encoders + scene head | Encoder + body decoder | + PROX + RICH + COFE |
| Step 5 | + Hand decoder cross-attention | Encoder + body decoder (hands unfrozen) | + BEHAVE + InterCap |
| Step 6 | + Object pose head + SAM3D prompt encoder | Encoder (lightly unfrozen) | + 3DIR + BEHAVE |
| Step 7 | + Cross-attention bridge, full iterative loop | Nothing frozen | All data combined |

---

## Loss Functions Summary

**Contact losses**:
- `L_bce`: Binary cross-entropy for per-vertex contact (backward compat with DAMON)
- `L_cdf`: L1 on contact distance field vs. GT geodesic distance
- `L_corr`: L1 on correspondence vectors (weighted by contact probability)
- `L_semantic`: Cross-entropy for contact type classification
- `L_target`: Cross-entropy for target type (object/floor/wall/furniture)
- `L_pal`: Pixel Anchoring Loss -- project 3D contacts to 2D, match with HOT 2D evidence (from DECO)

**Object losses**:
- `L_obj_pose`: L1 on object 6DoF relative to pelvis
- `L_obj_scale`: L1 on log-scale
- `L_penetration`: Body-object interpenetration (SDF-based, from PICO)

**Scene losses**:
- `L_floor`: Distance from floor-contacting vertices to floor plane
- `L_stability`: CoP-CoM alignment for standing poses (from IPMAN)
- `L_scene_pen`: Body vertices below floor or inside walls

**Body losses**: Inherited from SAM3DB (joints, vertices, SMPL parameters).

---

## Model Size Estimate

| Component | Parameters | Status |
|-----------|-----------|--------|
| SAM3DB Encoder | 632M-840M | Frozen (mostly) |
| SAM3DB Body Decoder | ~50M | Frozen in Steps 1-6 |
| SAM3DB Hand Decoder | ~30M | Frozen until Step 5 |
| Interaction Decoder (new) | ~15-25M | Trained from scratch |
| Prompt Encoders (new) | ~5-10M | Trained from scratch |
| Cross-attention bridge (new) | ~2-5M | Trained in Step 7 |
| **Total new parameters** | **~22-40M** | **~3-5% of base model** |

This is comparable to DECO's overhead (~1% for context branches). The interaction decoder is lightweight relative to the base model.

---

## References

- SAM 3D Body (Yang 2026): Base architecture, ViT-H/DINOv3 encoder, body decoder, hand decoder, MHR
- SAM 3D (Team 2025): Object tokens, DINOv2 conditioning, mask-based prompting
- DECO (Tripathi 2023): Dual-branch scene/part context, PAL loss
- BSTRO (Huang 2022): Per-vertex queries in transformer, MVM training
- PICO (Cseke 2025): Correspondence vectors, SDF penetration loss
- IPMAN (Tripathi 2023): CoP-CoM stability loss, differentiable pressure
- MCC-HO (Wu 2025): Hand-normalized coordinate system, joint occupancy + segmentation
