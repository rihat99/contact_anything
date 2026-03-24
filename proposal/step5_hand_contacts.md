# Step 5: Hand Contact Integration

**Duration**: ~1 week
**Depends on**: Step 3 (contact representation), optionally Step 4 (scene contact)
**Goal**: Feed contact information into the hand decoder so hand pose is contact-aware, and predict fine-grained hand-object contact

---

## Motivation

SAM3DB has a separate hand decoder that predicts hand pose from hand-crop features. Currently it operates independently from contact. But hand-object contact is the most common and most important interaction: grasping, holding, manipulating.

Problems with current approach:
- Hand decoder doesn't know if the hand is touching something
- Contact decoder doesn't have fine hand features (only full-body features)
- CARI4D showed even binary hand contact (2 bits!) significantly improves reconstruction

By connecting the interaction decoder to the hand decoder, we get:
1. **Hand pose informed by contact**: If the hand grasps a cup, fingers should curl around it
2. **Contact informed by hand pose**: Refined hand pose tells us which fingers are actually in contact

---

## Architecture

```
                    +------------------+
                    |   SAM3DB Encoder |
                    +--------+---------+
                             |
                +------------+------------+
                |                         |
           Full Image F              Hand Crop F_hand
                |                         |
     +----------+---------+     +--------v--------+
     | Interaction Decoder |     | Hand Decoder    |
     |                     |     |                  |
     | T_contact           |     | T_hand          |
     | T_object            |     |                  |
     |                     |     | Cross-Attn to    |
     | Cross-Attn to F     |     | F_hand          |
     |                     |     |                  |
     +----+----+-----------+     +---+----+---------+
          |    |                     |    |
          |    +-- cross-attn -------+    |          <-- NEW
          |    +------- cross-attn --+    |          <-- NEW
          |                               |
     Contact Field                   Hand Pose
     + Hand Contact Map              (MANO params)
```

**New cross-attention**: Bidirectional lightweight cross-attention between interaction decoder and hand decoder.

---

## New Components

### Hand-Contact Cross-Attention
- **Contact -> Hand**: Hand decoder tokens attend to contact tokens from interaction decoder. This informs hand pose about whether the hand is in contact and where.
- **Hand -> Contact**: Contact tokens attend to hand decoder tokens. This gives the contact decoder access to fine-grained hand features for better hand-region contact prediction.
- **Implementation**: Single-head cross-attention with small projection (d=64), added as an extra layer in both decoders.

### Hand Contact Head (Fine-Grained)
From contact tokens that have attended to hand features, predict:

| Output | Description | Vertices |
|--------|------------|----------|
| Hand contact probability | Per-vertex on MANO/SMPL-X hand mesh | 778 vertices per hand |
| Fingertip contact | Binary per finger (5 per hand) | 10 total |
| Grasp type | Classification: power, precision, pinch, palm, wrap | 1 per hand |

### Loss
```
L_hand_contact = L_bce_hand + lambda_grasp * L_grasp_type

L_bce_hand:    BCE on per-vertex hand contact
               GT from BEHAVE/InterCap (proximity-based, threshold 2cm on hand vertices)
L_grasp_type:  Cross-entropy on grasp classification (if GT available)
```

---

## Training

**What trains**: Cross-attention layers (hand <-> contact) + hand contact head
**What's frozen**: Encoder. Body decoder frozen. Hand decoder: **lightly finetuned** (low lr) since we're modifying its input.

**Data**:

| Dataset | Hand Contact GT | Object Info | Size |
|---------|----------------|-------------|------|
| BEHAVE | Proximity-based (2cm threshold) | 20 object meshes | 15K frames |
| InterCap | Proximity-based | 10 object categories | Multi-view |
| DexYCB | GT hand-object contact from RGBD | 21 YCB objects | 1K videos |
| DAMON | Body contact (includes hand region) | 84 categories | 4.4K images |

**Hand contact GT from BEHAVE/InterCap**: For each hand vertex in the fitted SMPL/MANO mesh, compute distance to nearest object surface vertex. If distance < 2cm, label as contact.

**Training recipe**:
- Interaction decoder: normal lr (1e-4)
- Hand decoder: low lr (1e-5) to avoid destroying pretrained hand quality
- Cross-attention layers: normal lr (1e-4)
- Hand contact head: normal lr (1e-4)

---

## Evaluation

| Metric | Dataset | Comparison |
|--------|---------|------------|
| Hand contact F1 | BEHAVE test | vs. proximity-based baseline |
| Hand pose (MPJPE hand) | FreiHand, DexYCB | vs. SAM3DB baseline |
| Fingertip contact accuracy | DexYCB | Track |
| Object contact improvement | BEHAVE | Does hand contact help object placement? |
| Body contact F1 | DAMON test | No degradation |

**Key check**: Hand pose quality must not degrade. The cross-attention should help (contact constrains pose) but could also hurt if poorly tuned.

---

## Ablations

1. Contact -> Hand only vs. bidirectional cross-attention
2. Hand decoder frozen vs. lightly finetuned
3. With vs. without hand contact head (does the cross-attention alone help?)
4. Hand contact from interaction decoder vs. from hand decoder tokens
5. Cross-attention dimension: d=32 vs. d=64 vs. d=128

---

## Success Criteria

- Hand contact F1 on BEHAVE/InterCap is meaningful (>0.4)
- Hand pose (MPJPE) does not degrade, ideally improves slightly for grasping poses
- Qualitative: when grasping objects, contact is on the correct fingers/palm regions
- The cross-attention bridge works without destabilizing either decoder

---

## Connection to MCC-HO

MCC-HO (Wu 2025) showed that hand-normalized coordinate systems and joint occupancy + segmentation predictions naturally encode the hand-object contact boundary. Our approach is complementary: instead of reconstructing the object in hand-normalized space, we predict contact on the hand surface and use it to constrain both hand pose and object placement.

---

## References

- SAM3DB (Yang 2026): Separate hand decoder, hand-crop features F_hand, MANO parameters
- CARI4D (Xie 2025): Binary hand contact (2 bits) significantly improves HOI reconstruction
- MCC-HO (Wu 2025): Hand-normalized coordinate system, joint hand-object reconstruction
- BEHAVE (Bhatnagar 2022): 15K RGBD frames, hand-object proximity contact
- DexYCB: GT hand-object contact from multi-view RGBD
