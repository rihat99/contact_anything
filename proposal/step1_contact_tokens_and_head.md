# Step 1: Contact Tokens + Interaction Decoder

**Duration**: ~1 week
**Depends on**: Current SAM3DB baseline with contact tokens
**Goal**: Separate the contact prediction into its own decoder with one-directional body→contact cross-attention, train it on DAMON; address overfitting via regularization and RICH data

---

## Motivation

Currently, contact query tokens are appended to the body decoder. This couples contact learning with pose learning, which is risky: the body decoder is trained on 7M images, and contact data is only 5.5K images. A separate interaction decoder can be trained independently without disrupting body pose quality.

Additionally, the interaction decoder should have access to body pose information from the start. Contact is inherently pose-dependent: knowing where body parts are in 3D directly constrains which vertices can be in contact. Every prior method implicitly uses pose for contact (DECO uses body-part features, BSTRO queries per-vertex on the predicted mesh, PICO starts from body pose). An interaction decoder without pose information would need to re-discover body configuration from raw image features alone — duplicating work the body decoder already did.

Since the body decoder is frozen, one-directional body→contact cross-attention is risk-free: no gradients flow back to the body decoder, so body pose quality is guaranteed to be preserved.

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
     |  T_pose  --------|--body--->| T_contact (K=16)|
     |  T_prompt        |  tokens  |                  |
     |  T_keypoint      |          | Self-Attention   |
     |  T_hand          |          | Cross-Attn to F  |
     |                  |          | Cross-Attn to    |
     |                  |          |   body tokens    |
     +--------+---------+          +--------+---------+
              |                             |
         Body Mesh (MHR)              Contact Head
                                     (per-vertex BCE)
```

**Key changes**:
1. Contact tokens move from body decoder to their own interaction decoder.
2. The interaction decoder cross-attends to body decoder's decoded T_pose tokens (one-directional: body→contact, no reverse yet).

---

## Which Body Decoder Tokens Are Used

The body decoder contains multiple token types after processing:

| Token | Count | Contains | Used for cross-attention? |
|-------|-------|----------|--------------------------|
| **T_pose (decoded)** | ~N_pose | Body pose, shape, camera, skeleton after self-attn + cross-attn to F | **Yes — primary source** |
| T_prompt (decoded) | Variable | Processed mask/camera prompt information | No (input conditioning, less useful) |
| T_keypoint (decoded) | Variable | Processed 2D keypoint information | No (optional, not always present) |
| T_hand (decoded) | ~N_hand | Hand-specific parameters | No (used in Step 5 via hand decoder) |

**Rationale for using T_pose only**: After body decoder processing, T_pose tokens encode the model's best estimate of body joint positions, body shape, and global orientation. These are the tokens that directly decode into SMPL/MHR parameters via the body heads. They provide the richest 3D body-aware representation for contact reasoning.

**Implementation**: Extract T_pose tokens from the body decoder's last layer output (or intermediate layer — ablate which layer is best). These are treated as read-only keys/values in the interaction decoder's cross-attention.

---

## Implementation Details

### Interaction Decoder
- **Layers**: 4 transformer decoder layers (matching body decoder depth, or ablate 2/4/6)
- **Token count**: K=16 learnable contact query tokens
- **Attention per layer**:
  1. Self-attention among contact tokens
  2. Cross-attention to image features F
  3. Cross-attention to body decoder T_pose tokens (NEW)
- **Hidden dim**: Same as body decoder (d_model)

### Body→Contact Cross-Attention
```python
class BodyToContactCrossAttn(nn.Module):
    def __init__(self, d_model, n_heads=4):
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, contact_tokens, body_pose_tokens):
        # contact_tokens: [K, B, d_model]  (queries)
        # body_pose_tokens: [N_pose, B, d_model]  (keys, values — DETACHED)
        body_pose_tokens = body_pose_tokens.detach()  # no gradients to body decoder
        attn_out = self.cross_attn(
            query=contact_tokens,
            key=body_pose_tokens,
            value=body_pose_tokens
        )[0]
        return self.norm(contact_tokens + attn_out)
```

**Note**: `.detach()` is a safety measure. Since the body decoder is frozen (requires_grad=False), gradients wouldn't flow anyway, but explicit detach makes the intent clear and protects against accidental unfreezing.

### Contact Head
- Input: K contact tokens from interaction decoder
- Linear projection: each token -> per-vertex logits (K tokens x 6890 vertices)
- Aggregation: max-pool across K tokens per vertex -> final per-vertex probability
- Sigmoid activation
- Alternative: each token predicts a soft Gaussian on body surface (center + spread), compose into per-vertex map

### Loss
```
L = L_bce

L_bce:  Binary cross-entropy on per-vertex contact probabilities vs. DAMON/RICH GT
        Use class weighting (contact vertices are ~5% of mesh) -- weight ~10:1
```

**Note on PAL loss**: DECO's ablations show marginal gain from PAL. The core issue is that HOT 2D contact annotations are coarse and imprecise, and projecting GT 3D contact from DAMON to 2D adds no new information beyond the 3D BCE (it's the same GT projected to a lower-dimensional, ambiguous space). PAL is dropped.

---

## Training

**What trains**: Interaction decoder (K contact tokens + 4 transformer layers + body→contact cross-attn + contact head)
**What's frozen**: SAM3DB encoder + body decoder + hand decoder

**Data**:
| Dataset | Images | Contact Labels | Usage |
|---------|--------|---------------|-------|
| DAMON train | ~4.4K | Dense vertex-level binary (SMPL) | Primary 3D supervision |
| RICH | ~577K | Body-scene contact (proximity-based, floor/furniture/wall) | Regularization — massive diversity boost |

**Overfitting mitigation** (DAMON alone is 4.4K images — too small):
- **RICH data**: Pull RICH into training now even though it's "Step 4 data". Body-scene contact on the same SMPL vertices; label schema is compatible with DAMON's binary labels. Balances DAMON:RICH sampling to avoid scene-contact bias (e.g., 1:4 ratio).
- **Dropout**: Add dropout 0.1–0.2 after each cross-attention layer in the interaction decoder
- **Weight decay**: 1e-1 (stronger than default 1e-2)
- **Reduce K**: Try K=8 before K=16 — fewer parameters to overfit

**Training recipe**:
- Optimizer: AdamW, lr=1e-4, weight decay 1e-1
- Batch size: 32
- Epochs: ~50-100 on DAMON; monitor val F1 on both DAMON and RICH separately
- Augmentation: random flip, color jitter, crop (inherited from SAM3DB)
- Contact class weighting: 10:1 for contact vs. non-contact vertices

---

## Evaluation

| Metric | Dataset | Target |
|--------|---------|--------|
| F1 | DAMON test | > 55% (beat DECO's ~55%) |
| Precision | DAMON test | Track |
| Recall | DAMON test | Track |
| Geodesic Error | DAMON test | Track (lower is better) |
| Body pose (MPJPE) | 3DPW | No degradation vs. SAM3DB baseline |

**Key check**: Body pose must NOT degrade. Since body decoder is frozen and T_pose tokens are detached, this is guaranteed — but verify.

---

## Ablations for This Step

1. **With vs. without body→contact cross-attention** (key ablation: does pose info help contact?)
2. K=8 vs. K=16 vs. K=32 contact tokens
3. Interaction decoder depth: 2 vs. 4 vs. 6 layers
4. With vs. without RICH data (does it reduce overfitting without hurting DAMON F1?)
5. Per-vertex logits vs. Gaussian patch composition from tokens
6. Cross-attend to T_pose from last body decoder layer vs. intermediate layer vs. all layers (concatenated)
7. Cross-attention heads: 1 vs. 4 vs. 8

**Ablation 1 is the most important**: it validates that providing body pose information to the contact decoder improves contact prediction. Expected outcome: significant improvement, since pose is directly informative for contact.

---

## Success Criteria

- F1 on DAMON >= 55% (matches or beats DECO)
- Body→contact cross-attention improves F1 over image-features-only baseline (ablation 1)
- Body pose unchanged from SAM3DB baseline
- Qualitative: contact predictions are spatially coherent (not scattered noise)

---

## References

- DECO (Tripathi 2023): Class-weighted BCE, ~55% F1 on DAMON; PAL ablation shows marginal gain
- BSTRO (Huang 2022): Per-vertex transformer queries, Masked Vertex Modeling
- RICH/BSTRO (Huang 2022): 577K images, proximity-based body-scene contact labels
