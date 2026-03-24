# Contact Representations

## Current State of the Art

| Method | Representation | Pros | Cons |
|--------|---------------|------|------|
| DECO | Binary per-vertex (6890 SMPL) | Simple, established | Lossy, no semantics, topology-dependent |
| BSTRO | Binary per-vertex (6890 SMPL) | Transformer-based, hallucination via MVM | Same limitations as DECO |
| InteractVLM | Binary per-vertex + semantic conditioning | Semantic contact per object | Still binary, pipeline-based |
| PICO | Dense vertex-to-point correspondences | Body-object spatial link | Requires optimization, not learned |
| LEMON | Binary per-vertex + affordance + spatial | Joint prediction | Requires pre-computed mesh |
| POSA | Binary per-vertex + semantic label (40 classes) | Pose-dependent, generative | Only scenes, not objects |
| IPMAN | Continuous pressure field | Differentiable, physics-based | Floor-only |
| FECO | Binary per-vertex on foot mesh (265 verts) | Multi-scale hierarchy | Foot-only |
| HOT | 2D image-space heatmaps + body part | Scalable, large dataset | 2D only, no 3D |

---

## Proposed Representations

### Option A: Binary Per-Vertex (Baseline)

Same as DECO. Per-vertex probability on SMPL mesh (6890 vertices).

```
Output: p_contact(v) in [0, 1]  for each vertex v
Loss: Binary cross-entropy
GT: DAMON labels (vertex-level binary)
```

**Use as**: Baseline to beat. Already implemented in current SAM3DB + contact tokens.

---

### Option B: Contact Distance Field (CDF)

For each body vertex, predict geodesic distance to nearest contact region on the body surface.

```
Output: d_geodesic(v) in [0, inf)  for each vertex v
  d = 0  -> vertex is in contact
  d > 0  -> geodesic distance to nearest contact (cm)
Loss: L1 or smooth-L1 on geodesic distance
GT: Precomputed from DAMON binary labels on SMPL mesh
```

**Advantages**:
- Continuous supervision -- easier to learn than sharp binary boundaries
- Captures contact "proximity" -- useful for near-contact reasoning
- Resolution-independent
- Threshold at inference for backward-compatible binary output

**GT generation**: For each SMPL mesh in DAMON, compute geodesic distance from every vertex to the nearest contact vertex. This is precomputable using Dijkstra on the mesh graph. Cost: ~0.1s per mesh.

**Recommended as**: Primary replacement for binary. Simple upgrade with clear training signal.

---

### Option C: Correspondence Vectors

For each vertex predicted to be in contact, additionally predict a 3D offset vector pointing toward the contacted surface.

```
Output per vertex:
  p_contact(v)  in [0, 1]       -- contact probability
  delta(v)      in R^3           -- 3D offset from body vertex to contacted surface point
Loss: L1 on delta, weighted by p_contact
GT: From BEHAVE/InterCap/PICO-db body-object pairs
```

**Advantages**:
- Directly encodes body-to-object spatial relationship
- Eliminates need for separate correspondence retrieval (as in PICO)
- Enables object placement from contact alone

**GT generation**: For each contacting body vertex in BEHAVE/InterCap, compute vector to nearest object surface point. Available in PICO-db annotations.

**Recommended as**: Add-on to CDF for steps that involve object reasoning (Step 3+).

---

### Option D: Contact Query Tokens (DETR-style)

A set of K learnable tokens (K=16-32) that decode into structured contact patches.

```
Each active token outputs:
  existence     in [0, 1]        -- is this query active?
  body_region   in UV space      -- Gaussian (center, spread, rotation) on body surface
  confidence    in [0, 1]        -- contact intensity within the patch
  semantic      in {support, grasp, lean, press, wrap, step, sit, rest}
  target_type   in {object, floor, wall, furniture, person}
  correspondence in R^3          -- offset to contacted surface
  target_embed  in R^d           -- learned embedding of contacted entity

Training: Hungarian matching loss (like DETR)
GT: Convert DAMON/BEHAVE/PICO-db to contact patches via connected-component grouping
```

**Advantages**:
- Structured: contact patches have spatial coherence
- Semantic: each token knows what kind of contact and what target
- Variable count: handles 0-N simultaneous contacts naturally
- Unified: same representation for object grasps, floor support, wall leaning

**Backward compatibility**: Compose soft body-region masks from active tokens (union with max) to produce standard per-vertex probabilities.

**Recommended as**: The most architecturally novel representation. Explore in Step 3 ablation.

---

### Option E: Neural Contact Maps (UV Space)

Predict contact in UV-space rather than on mesh vertices.

```
Output: Multi-channel UV map at resolution HxW
  Channel 0: contact_prob       [0, 1]
  Channel 1: geodesic_dist      [0, inf)
  Channel 2-4: correspondence   R^3
  Channel 5+: object_class      logits
Loss: Per-pixel BCE + L1 + CE
GT: Project 3D contact labels to SMPL UV atlas (DensePose-style)
```

**Advantages**:
- Decouples contact from mesh topology
- Higher resolution in important regions (hands, feet) via non-uniform UV
- Compatible with 2D convolutional decoders
- Can leverage HOT 2D contact maps as auxiliary supervision

**Recommended as**: Optional exploration. More complex than CDF but resolution-independent.

---

## Ablation Plan (Step 3)

| Representation | Training Data | Metrics | Expected Outcome |
|---------------|--------------|---------|-----------------|
| Binary per-vertex (baseline) | DAMON | F1, Precision, Recall | Baseline numbers |
| CDF (continuous) | DAMON (precomputed geodesic GT) | F1 (thresholded), AUC, Geodesic Error | Better boundary handling |
| CDF + Correspondence vectors | DAMON + BEHAVE + PICO-db | F1 + Correspondence error (mm) | Body-object spatial link |
| Contact query tokens (K=16) | DAMON + BEHAVE + PICO-db | F1 + Semantic accuracy | Structured, semantic output |
| CDF + Query tokens (combined) | All | All metrics | Best of both |

**Recommendation**: Start with **CDF** (Option B) as primary upgrade from binary. Add **correspondence vectors** (Option C) when object data is available. Explore **query tokens** (Option D) as the architecturally novel direction. Compare all in Step 3.

---

## Key Data for Each Representation

| Representation | Requires | Available From |
|---------------|----------|---------------|
| Binary | Body contact labels | DAMON (5.5K), RICH (90K), HOT (35K via PAL) |
| CDF | Same as binary (precomputed) | Same + geodesic computation |
| Correspondence vectors | Body-object pairs with GT meshes | BEHAVE (15K), InterCap, PICO-db (4.1K) |
| Query tokens | Grouped contact patches | Derived from binary labels |
| UV maps | Contact projected to UV | Derived from binary + DensePose UV |

---

## References

- DECO (Tripathi 2023): Binary per-vertex + PAL loss
- BSTRO (Huang 2022): Per-vertex queries in transformer
- POSA (Hassan 2021): Per-vertex binary + semantic (40 Matterport categories)
- PICO (Cseke 2025): Dense vertex-to-point correspondences
- IPMAN (Tripathi 2023): Continuous pressure field from penetration depth
- HOT (Chen 2023): 2D heatmaps with body-part labels
