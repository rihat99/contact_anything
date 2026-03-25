# Step 3: Contact Representation Ablation

**Duration**: ~1-2 weeks
**Depends on**: Step 2 (interaction decoder with object prompts)
**Goal**: Find the best contact representation by comparing binary, CDF, correspondence vectors, and contact query tokens

---

## Motivation

Binary per-vertex contact is the standard (DECO, BSTRO, InteractVLM) but is fundamentally limited: it's topology-dependent, loses boundary information, and carries no spatial relationship to the contacted surface. This step systematically compares richer representations.

See `01_CONTACT_REPRESENTATIONS.md` for full description of each representation.

---

## Architecture Changes

The interaction decoder stays the same (including body→contact cross-attention from Step 1). Only the **output heads** change per representation.

```
Body Decoder (FROZEN)
    T_pose ──body tokens──> Interaction Decoder (from Step 2)
                                     |
                                Contact Tokens (K=16)
                                     |
                +----v----+----v----+----v----+----v----+
                | Binary  |   CDF   |  Corr.  | Query   |
                | Head    |   Head  |  Head   | Token   |
                | (base)  |  (new)  |  (new)  | Head    |
                +----+----+----+----+----+----+----+----+
                     |         |         |         |
                p(v) [0,1] d_geo(v)  delta(v)  patch_i
                6890 verts 6890 verts 6890x3   K structs
```

---

## Representations to Compare

### A. Binary Per-Vertex (Baseline from Step 1)
```
Head: Linear(d_model, 6890) + Sigmoid
Loss: BCE with class weight 10:1
GT:   DAMON binary labels
```

### B. Contact Distance Field (CDF)
```
Head: Linear(d_model, 6890)  -- no sigmoid, predicts positive reals
Loss: Smooth-L1 on geodesic distance
GT:   Precomputed geodesic distances from DAMON binary labels on SMPL mesh
      For each vertex, d = shortest geodesic path to nearest contact vertex
      Precomputation: Dijkstra on SMPL mesh graph (~0.1s per mesh, do once)
```

**GT precomputation script needed**: Take DAMON's binary contact labels, compute geodesic distance field on SMPL mesh. Store as .npy per sample.

### C. CDF + Correspondence Vectors
```
Head: Linear(d_model, 6890 * 4)  -- (d_geo, dx, dy, dz) per vertex
Loss: Smooth-L1 on d_geo + L1 on correspondence vectors (weighted by contact_prob)
GT:   CDF from DAMON + correspondence vectors from BEHAVE/InterCap/PICO-db
      For DAMON-only samples: correspondence vectors masked out (only CDF supervised)
```

**Data for correspondence**: BEHAVE (15K frames, 20 objects), InterCap, PICO-db (4.1K images, 627 object instances). For each contacting body vertex, the correspondence vector points to the nearest object surface point.

### D. Contact Query Tokens (Structured)
```
Each of K=16 tokens decodes to:
  existence:     Sigmoid(Linear(d_model, 1))
  body_center:   Sigmoid(Linear(d_model, 2))     -- UV coordinates on body
  body_spread:   Softplus(Linear(d_model, 2))     -- Gaussian spread
  confidence:    Sigmoid(Linear(d_model, 1))
  semantic:      Linear(d_model, N_semantic)       -- contact type logits

Loss: Hungarian matching (bipartite assignment between predicted and GT patches)
      + Gaussian NLL for body region
      + Cross-entropy for semantics
GT:   Convert DAMON binary labels to contact patches:
      1. Connected component analysis on contact vertices (geodesic clustering)
      2. Fit Gaussian to each component in UV space
      3. Assign semantic label from DAMON object annotations
```

**GT precomputation**: Connected component analysis on SMPL contact vertices, Gaussian fitting per component.

### E. CDF + Query Tokens (Combined)
```
CDF head provides smooth per-vertex field
Query token head provides structured patches with semantics
Both supervised jointly
Inference: use CDF for per-vertex output, query tokens for semantic/structured output
```

---

## Training

**What trains**: Contact output heads (each variant)
**What's frozen**: Encoder + body decoder. Interaction decoder finetuned with new heads.

**Data per representation**:

| Representation | Primary Data | Additional Data | Total |
|---------------|-------------|-----------------|-------|
| Binary | DAMON (4.4K) | HOT via PAL (15K) | ~19K |
| CDF | DAMON (4.4K) | HOT via PAL (15K) | ~19K |
| CDF + Corr | DAMON (4.4K) + BEHAVE (15K) + PICO-db (4.1K) | HOT via PAL | ~38K |
| Query Tokens | DAMON (4.4K) | HOT via PAL | ~19K |
| CDF + Tokens | DAMON (4.4K) + BEHAVE (15K) + PICO-db (4.1K) | HOT via PAL | ~38K |

**Training**: Same recipe as Step 1. Each representation variant is a separate run. Compare after convergence.

---

## Evaluation

| Metric | What It Measures | Applicable To |
|--------|-----------------|---------------|
| F1 (binary threshold) | Contact detection accuracy | All (CDF thresholded at d=0) |
| Geodesic Error (cm) | How far predictions are from true contact | CDF, Binary |
| AUC | Ranking quality of contact probability | All |
| Correspondence Error (mm) | Accuracy of body-to-object vectors | CDF+Corr, CDF+Tokens |
| Semantic Accuracy | Contact type classification | Query Tokens |
| Patch Coherence | Are predicted patches spatially coherent? | Query Tokens |

**Datasets for evaluation**:
- DAMON test: F1, Geodesic Error, AUC
- BEHAVE test: Correspondence Error
- PICO-db test: Correspondence Error
- InterCap: Out-of-distribution evaluation

---

## Expected Outcomes

| Representation | Expected F1 | Expected Advantage |
|---------------|------------|-------------------|
| Binary (baseline) | ~55-58% | Simple, established |
| CDF | ~57-60% | Better boundary handling, smoother gradients |
| CDF + Corr | ~57-60% + corr. error | Body-object spatial link |
| Query Tokens | ~55-58% | Structured, semantic output |
| CDF + Tokens | ~58-62% | Best of both worlds |

**Hypothesis**: CDF will match or beat binary on F1 while providing a smoother, more informative output. Correspondence vectors will be essential for downstream object placement (Step 6).

---

## Decision Point

After this step, choose the primary representation going forward:
- If CDF clearly wins on F1 and geodesic error -> use CDF as default
- If correspondence vectors add meaningful signal on BEHAVE/InterCap -> keep them
- If query tokens provide useful semantic structure -> keep as secondary output
- Most likely outcome: **CDF + Correspondence Vectors** as primary, with optional query token semantic output

---

## References

- DECO (Tripathi 2023): Binary per-vertex baseline, ~55% F1
- PICO (Cseke 2025): Correspondence vector formulation, ContactEdit patches
- POSA (Hassan 2021): Per-vertex binary + semantic (40 Matterport categories)
- IPMAN (Tripathi 2023): Continuous pressure field idea
- BEHAVE (Bhatnagar 2022): Body-object GT meshes for correspondence supervision
