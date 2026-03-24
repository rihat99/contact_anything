# Step 1: Contact Tokens + Interaction Decoder

**Duration**: ~1 week
**Depends on**: Current SAM3DB baseline with contact tokens
**Goal**: Separate the contact prediction into its own decoder, train it properly on DAMON + HOT

---

## Motivation

Currently, contact query tokens are appended to the body decoder. This couples contact learning with pose learning, which is risky: the body decoder is trained on 7M images, and contact data is only 5.5K images. A separate interaction decoder can be trained independently without disrupting body pose quality.

---

## Architecture

```
                    +------------------+
                    |   SAM3DB Encoder |   (FROZEN)
                    |   (ViT-H/DINOv3)|
                    +--------+---------+
                             |
                        Image Features F
                             |
              +--------------+--------------+
              |                             |
     +--------v--------+          +--------v--------+
     |  Body Decoder    |          | Interaction     |
     |  (FROZEN)        |          | Decoder (NEW)   |
     |                  |          |                  |
     |  T_pose          |          | T_contact (K=16)|
     |  T_prompt        |          |                  |
     |  T_keypoint      |          | Self-Attention   |
     |  T_hand          |          | Cross-Attn to F  |
     |                  |          |                  |
     +--------+---------+          +--------+---------+
              |                             |
         Body Mesh (MHR)              Contact Head
                                     (per-vertex BCE)
```

**Key change**: Contact tokens move from body decoder to their own interaction decoder. The interaction decoder has its own self-attention and cross-attention to image features F.

---

## Implementation Details

### Interaction Decoder
- **Layers**: 4 transformer decoder layers (matching body decoder depth, or ablate 2/4/6)
- **Token count**: K=16 learnable contact query tokens
- **Attention**: Self-attention among contact tokens + cross-attention to image features F
- **Hidden dim**: Same as body decoder (d_model)
- **No cross-attention to body decoder yet** (that's Step 7)

### Contact Head
- Input: K contact tokens from interaction decoder
- Linear projection: each token -> per-vertex logits (K tokens x 6890 vertices)
- Aggregation: max-pool across K tokens per vertex -> final per-vertex probability
- Sigmoid activation
- Alternative: each token predicts a soft Gaussian on body surface (center + spread), compose into per-vertex map

### Loss
```
L = L_bce + lambda_pal * L_pal

L_bce:  Binary cross-entropy on per-vertex contact probabilities vs. DAMON GT
        Use class weighting (contact vertices are ~5% of mesh) -- weight ~10:1
L_pal:  Pixel Anchoring Loss (from DECO) -- render contact-colored mesh to 2D,
        compare with HOT 2D contact maps via BCE
        Requires: differentiable rendering (PyTorch3D), GT SMPL mesh from body decoder
```

---

## Training

**What trains**: Interaction decoder (K contact tokens + 4 transformer layers + contact head)
**What's frozen**: SAM3DB encoder + body decoder + hand decoder

**Data**:
| Dataset | Images | Contact Labels | Usage |
|---------|--------|---------------|-------|
| DAMON train | ~4.4K | Dense vertex-level binary (SMPL) | Primary 3D supervision |
| HOT-Annotated | ~15K | 2D contact heatmaps + body part | PAL loss (2D auxiliary) |

**Training recipe**:
- Optimizer: AdamW, lr=1e-4, weight decay 1e-2
- Batch size: 32 (small data, can fit easily)
- Epochs: ~50-100 (small dataset, need many passes)
- Augmentation: random flip, color jitter, crop (inherited from SAM3DB)
- Contact class weighting: 10:1 for contact vs. non-contact vertices

**Body mesh for PAL loss**: Use the frozen body decoder's output mesh. This mesh is good enough for rendering since the body decoder is pretrained on 7M images.

---

## Evaluation

| Metric | Dataset | Target |
|--------|---------|--------|
| F1 | DAMON test | > 55% (beat DECO's ~55%) |
| Precision | DAMON test | Track |
| Recall | DAMON test | Track |
| Geodesic Error | DAMON test | Track (lower is better) |
| Body pose (MPJPE) | 3DPW | No degradation vs. SAM3DB baseline |

**Key check**: Body pose must NOT degrade. If it does, the interaction decoder is leaking bad gradients (shouldn't happen since body decoder is frozen, but verify).

---

## Ablations for This Step

1. K=8 vs. K=16 vs. K=32 contact tokens
2. Interaction decoder depth: 2 vs. 4 vs. 6 layers
3. With vs. without PAL loss
4. Per-vertex logits vs. Gaussian patch composition from tokens
5. SAM3DB encoder features vs. intermediate body decoder features as input to interaction decoder

---

## Success Criteria

- F1 on DAMON >= 55% (matches or beats DECO)
- Body pose unchanged from SAM3DB baseline
- Clean separation: interaction decoder operates independently
- Qualitative: contact predictions are spatially coherent (not scattered noise)

---

## References

- DECO (Tripathi 2023): PAL loss formulation, class-weighted BCE, ~55% F1 on DAMON
- BSTRO (Huang 2022): Per-vertex transformer queries, Masked Vertex Modeling
- HOT (Chen 2023): 35K images with 2D contact heatmaps, body-part attention
