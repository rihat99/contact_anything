# Step 7: Mutual Refinement — Adding Contact→Body Feedback

**Duration**: ~1-2 weeks
**Depends on**: All previous steps (Step 1-6)
**Goal**: Complete the bidirectional loop by adding contact→body cross-attention and unfreezing the body decoder, enabling iterative mutual refinement of pose, contact, and objects

---

## Motivation

Steps 1-6 established **one-directional** body→contact cross-attention: the interaction decoder reads body pose tokens (T_pose) to inform contact prediction. This already helps contact quality significantly.

This step adds the **reverse direction**: body decoder tokens attend to contact/object tokens from the interaction decoder. This enables:

- **Contact constrains pose**: if the right hand touches a table, the arm should reach toward it
- **Object constrains pose**: if the object is at a specific 6DoF, pose should be consistent
- **Iterative refinement**: multiple rounds of body↔contact exchange improve both predictions

**What changed vs. the original plan**: The body→contact direction was already established in Step 1, so this step is specifically about:
1. Adding contact→body cross-attention (the risky direction — requires unfreezing body decoder)
2. Making the loop iterative (T iterations)
3. End-to-end finetuning of all components

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
| body_t = CrossAttn|<-- contact tokens  | inter_t = CrossAttn|<-- body T_pose
|   (body_t,        |    (NEW in S7)     |   (inter_t,       |    (from Step 1)
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

**Step 1-6 already had**: inter_t cross-attends to body T_pose tokens (body→contact).
**New in Step 7**: body_t cross-attends to contact/object/scene tokens (contact→body). This is the risky direction since it requires unfreezing the body decoder.

---

## Which Tokens Flow in Each Direction

### Body → Contact (established in Step 1)
| Source | Tokens | Information |
|--------|--------|------------|
| Body decoder | T_pose (decoded) | Joint positions, body shape, global orientation |

### Contact → Body (NEW in Step 7)
| Source | Tokens | Information |
|--------|--------|------------|
| Interaction decoder | T_contact (K=16) | Where contact is happening on body surface |
| Interaction decoder | T_object (2-4) | Object identity, location, geometry |
| Interaction decoder | T_scene (2-4) | Floor/wall geometry and contact |

**All interaction decoder tokens** are concatenated as keys/values for the body decoder's new cross-attention layer. This gives the body decoder access to the full interaction context.

---

## Implementation Details

### Contact→Body Cross-Attention Bridge (NEW)
```python
class ContactToBodyBridge(nn.Module):
    def __init__(self, d_model, d_bridge=64, n_heads=1):
        self.q_proj = nn.Linear(d_model, d_bridge)
        self.k_proj = nn.Linear(d_model, d_bridge)
        self.v_proj = nn.Linear(d_model, d_bridge)
        self.out_proj = nn.Linear(d_bridge, d_model)
        self.gate = nn.Parameter(torch.zeros(1))  # starts at 0

    def forward(self, body_tokens, interaction_tokens):
        # body_tokens: [N_pose, B, d_model]  (queries)
        # interaction_tokens: [K+N_obj+N_scene, B, d_model]  (keys, values)
        q = self.q_proj(body_tokens)
        k = self.k_proj(interaction_tokens)
        v = self.v_proj(interaction_tokens)
        attn = softmax(q @ k.T / sqrt(d_bridge)) @ v
        return body_tokens + self.gate * self.out_proj(attn)  # residual with gate
```

**The learnable gate starts at 0** — at the beginning of training, the contact→body cross-attention has zero effect. The model behaves exactly like Steps 1-6. As training progresses, the gate opens and mutual refinement kicks in. This ensures smooth transition.

**Note**: The body→contact cross-attention (Step 1) does NOT use a gate — it's a standard cross-attention that was trained from the start.

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
- Add contact→body cross-attention bridges with gates initialized to 0
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
| DAMON + RICH | Body contact (object + scene) |
| BEHAVE + InterCap | Body-object contact + correspondence + object 6DoF |
| PICO-db | Body-object correspondence |
| PROX + RICH | Scene contact |
| COFE + MoYo | Foot/floor contact + stability |
| 3DIR | Contact + affordance + spatial relation |

---

## Evaluation

### Does Adding Contact→Body Help? (Core Question)

| Experiment | Comparison |
|-----------|------------|
| Body→contact only (Steps 1-6) vs. bidirectional | Does the reverse direction improve anything? |
| Body pose (MPJPE) with vs. without contact→body feedback | Does contact improve pose? |
| Contact F1: Steps 1-6 vs. Step 7 (iterative) | Does iteration improve contact? |
| T=1 vs. T=3 vs. T=6 | Does more iteration = better? |
| Per-iteration metrics | Do predictions improve at each step? |
| Object pose error with vs. without mutual refinement | Does the loop help objects? |

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
| Body pose degradation from contact→body | MPJPE increases | Reduce gate magnitude; gradient stopping; lower lr for body decoder |
| Oscillation between iterations | Metrics don't improve or oscillate | Add momentum; fewer iterations; tighter gate |
| Training instability | Loss diverges or NaN | Gradient clipping; lower lr; staged unfreezing |
| No improvement from iteration | T=3 same as T=1 | Contact→body bridge too weak; increase d_bridge |
| Slow convergence | Takes too many epochs | Pre-train bridges separately on synthetic paired data |

---

## Ablations

1. **Unidirectional (body→contact only, Steps 1-6) vs. bidirectional (Step 7)** — core ablation
2. T=1 vs. T=2 vs. T=3 vs. T=4 vs. T=6
3. Gated bridge vs. ungated for contact→body direction
4. Deep supervision vs. final-iteration-only supervision
5. Gradient stopping schedule for contact→body
6. Bridge dimension d=32 vs. d=64 vs. d=128
7. Which interaction tokens the body decoder attends to: all vs. contact-only vs. contact+object

---

## Success Criteria

- **Bidirectional improves over unidirectional** (the core hypothesis of this step)
- Per-iteration improvement visible (iteration 3 > iteration 2 > iteration 1)
- Body pose does not degrade, ideally improves for interaction poses
- Overall metrics competitive with or exceeding:
  - DECO on body contact
  - InteractVLM on semantic contact
  - PICO-fit on object placement
  - PhySIC on scene contact
- Inference speed remains practical (< 200ms)

---

## Expected Contributions (from Full System)

1. **First promptable end-to-end model** for joint pose + contact + object/scene from single image
2. **Bidirectional cross-modal refinement** between body, contact, and object decoders
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
