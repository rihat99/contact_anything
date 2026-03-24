# Step 7: Mutual Refinement of Pose, Contact, and Objects

**Duration**: ~1-2 weeks
**Depends on**: All previous steps (Step 1-6)
**Goal**: Enable iterative cross-attention between body decoder and interaction decoder so pose, contact, and object predictions mutually refine each other

---

## Motivation

This is the core architectural contribution of the project. All previous steps built the components independently. Now we connect them in a feedback loop.

**Why this matters**:
- Contact constrains pose: if the right hand touches a table, the arm should reach toward it
- Pose constrains contact: if the arm is extended, only fingertip contact is plausible, not palm
- Object constrains both: if the object is small, contact area should be limited
- These mutual constraints should improve all three predictions iteratively

**Evidence it works**:
- PICO shows 3-stage optimization (body -> contact -> object) improves all metrics
- PhySIC shows contact + penetration losses improve pose (V2V: 218 -> 167mm on PROX)
- SAM3DB's own iterative decoder (body tokens refined over iterations) improves pose
- CARI4D's CoCoNet shows even binary hand contact improves HOI reconstruction

---

## Architecture

```
Iteration t = 1, 2, ..., T  (T = 3-6)

+-------------------+                    +-------------------+
|  Body Decoder     |                    | Interaction       |
|  Iteration t      |                    | Decoder Iter t    |
|                   |                    |                   |
| body_t = SelfAttn |                    | inter_t = SelfAttn|
|   (body_{t-1})    |                    |   (inter_{t-1})   |
|                   |                    |                   |
| body_t = CrossAttn|<-- image F         | inter_t = CrossAttn|<-- image F
|   (body_t, F)     |                    |   (inter_t, F)    |
|                   |                    |                   |
| body_t = CrossAttn|<-- contact tokens  | inter_t = CrossAttn|<-- body tokens
|   (body_t,        |    (NEW)           |   (inter_t,       |    (NEW)
|    contact_t)     |                    |    body_t)        |
|                   |                    |                   |
+--------+----------+                    +--------+----------+
         |                                        |
    Intermediate:                           Intermediate:
    body_params_t                           contact_t
    (pose, shape)                           obj_pose_t
                                            scene_contact_t
         |                                        |
         v                                        v
    [Deep supervision]                      [Deep supervision]
    L_body_t                                L_contact_t + L_obj_t
```

**Each iteration**: Both decoders run one layer and exchange information via cross-attention. The key is bidirectional: body tokens inform contact, contact tokens inform body.

---

## Implementation Details

### Cross-Attention Bridge

```python
# In Body Decoder layer t:
body_tokens = body_self_attention(body_tokens)
body_tokens = body_cross_attention(body_tokens, image_features)  # existing
body_tokens = contact_bridge(body_tokens, contact_tokens)        # NEW

# In Interaction Decoder layer t:
inter_tokens = inter_self_attention(inter_tokens)
inter_tokens = inter_cross_attention(inter_tokens, image_features)  # existing
inter_tokens = body_bridge(inter_tokens, body_tokens)              # NEW
```

**Bridge design** (lightweight):
```python
class CrossBridge(nn.Module):
    def __init__(self, d_model, d_bridge=64, n_heads=1):
        self.q_proj = nn.Linear(d_model, d_bridge)
        self.k_proj = nn.Linear(d_model, d_bridge)
        self.v_proj = nn.Linear(d_model, d_bridge)
        self.out_proj = nn.Linear(d_bridge, d_model)
        self.gate = nn.Parameter(torch.zeros(1))  # learnable gate, starts at 0

    def forward(self, x, context):
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)
        attn = softmax(q @ k.T / sqrt(d_bridge)) @ v
        return x + self.gate * self.out_proj(attn)  # residual with gate
```

**The learnable gate starts at 0** -- this means at the beginning of training, the cross-attention has no effect, and the model behaves like Steps 1-6. As training progresses, the gate opens and mutual refinement kicks in.

### Deep Supervision
Every iteration t produces intermediate outputs. All are supervised:
```
L_total = sum over t=1..T of lambda_t * (L_body_t + L_contact_t + L_obj_t)

lambda_t increases with t (later iterations get higher weight):
  lambda_t = t / T  (linear schedule)
  or lambda_t = 1 for all t (uniform)
```

### Iteration Schedule
- T=1: No iterative refinement (equivalent to Steps 1-6)
- T=3: Light refinement (recommended starting point)
- T=6: Full refinement (may not be needed if T=3 converges)

---

## Training Strategy

### Phase 1: Warm Up (Frozen Gates)
- Load all weights from Steps 1-6
- Add cross-attention bridges with gates initialized to 0
- Train for a few epochs with gates frozen at 0 (verify no degradation)
- Unfreeze gates, train with low lr on bridge parameters

### Phase 2: Iterative Training
- Enable deep supervision at all iterations
- Increase T gradually: start with T=2, then T=3, then T=4
- Gradient stopping: in early epochs, stop gradients from interaction decoder to body decoder to prevent instability
- Unfreeze body decoder (low lr: 1e-5) to allow pose adaptation

### Phase 3: End-to-End Finetuning
- All components unfrozen (encoder at very low lr: 1e-6)
- Full iterative loop with T=3-4
- All data combined:

| Dataset | Supervision Signal |
|---------|-------------------|
| SAM3DB data (subsampled) | Body pose |
| DAMON + HOT | Body contact |
| BEHAVE + InterCap | Body-object contact + correspondence + object 6DoF |
| PICO-db | Body-object correspondence |
| PROX + RICH | Scene contact |
| COFE + MoYo | Foot/floor contact + stability |
| 3DIR | Contact + affordance + spatial relation |

---

## Evaluation

### Does Iterative Refinement Help?

| Experiment | Comparison |
|-----------|------------|
| T=1 vs. T=3 vs. T=6 | Does more iteration = better? |
| Body pose (MPJPE) with vs. without contact feedback | Does contact improve pose? |
| Contact F1 with vs. without pose feedback | Does pose improve contact? |
| Object pose error with vs. without mutual refinement | Does the loop help objects? |
| Per-iteration metrics | Do predictions improve at each step? |

### Full Benchmark

| Task | Metric | Target |
|------|--------|--------|
| Body contact F1 | DAMON test | > 60% (beat DECO + InteractVLM) |
| Semantic contact | DAMON test | Beat InteractVLM |
| Body-object correspondence | PICO-db, InterCap | Competitive with PICO-fit |
| Object 6DoF | BEHAVE, InterCap | Competitive with PICO-fit |
| Scene contact F1 | PROX, RICH | Competitive with PhySIC |
| Body pose | 3DPW, RICH | No degradation, ideally improvement |
| Hand pose | FreiHand, DexYCB | No degradation |
| Inference speed | All | < 200ms (real-time capable) |

---

## Potential Issues and Mitigations

| Issue | Symptom | Mitigation |
|-------|---------|-----------|
| Oscillation between iterations | Metrics don't improve or oscillate | Reduce gate magnitude; add momentum; fewer iterations |
| Body pose degradation | MPJPE increases | Freeze body decoder; gradient stopping from interaction to body |
| Training instability | Loss diverges or NaN | Gradient clipping; lower lr; staged unfreezing |
| No improvement from iteration | T=3 same as T=1 | The cross-attention bridge may be too weak; increase d_bridge |
| Slow convergence | Takes too many epochs | Pre-train bridges separately on synthetic paired data |

---

## Ablations

1. T=1 vs. T=2 vs. T=3 vs. T=4 vs. T=6
2. Bidirectional bridge vs. contact->body only vs. body->contact only
3. Gated bridge vs. ungated
4. Deep supervision vs. final-iteration-only supervision
5. Gradient stopping schedule
6. Bridge dimension d=32 vs. d=64 vs. d=128

---

## Success Criteria

- **Both pose and contact improve** with iterative refinement (the core hypothesis)
- Per-iteration improvement visible (iteration 3 > iteration 2 > iteration 1)
- Overall metrics competitive with or exceeding:
  - DECO on body contact
  - InteractVLM on semantic contact
  - PICO-fit on object placement
  - PhySIC on scene contact
- Inference speed remains practical (< 200ms)

---

## Expected Contributions (from Full System)

1. **First promptable end-to-end model** for joint pose + contact + object/scene from single image
2. **Iterative cross-modal refinement** between body, contact, and object decoders
3. **Rich contact representation** beyond binary per-vertex labels
4. **Unified object + scene contact** in one architecture
5. **Practical speed** vs. optimization-based methods (100x faster than PICO/PhySIC)

---

## References

- SAM3DB (Yang 2026): Iterative body decoder, deep supervision per iteration
- PICO (Cseke 2025): 3-stage optimization shows contact -> pose -> object improves all
- PhySIC (Muralidhar 2025): Contact + penetration losses improve pose on PROX/RICH
- CARI4D (Xie 2025): CoCoNet refinement, even binary hand contact helps
- DETR (Carion 2020): Iterative decoder with deep supervision
- SAM (Kirillov 2023): Promptable architecture, iterative mask refinement
