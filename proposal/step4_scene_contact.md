# Step 4: Scene Contact (Floor, Wall, Surfaces)

**Duration**: ~1 week
**Depends on**: Step 3 (contact representation chosen)
**Goal**: Add scene contact prediction -- feet on floor, body against wall, sitting on furniture surfaces

---

## Motivation

Most real interactions involve the scene: standing on a floor, leaning against a wall, sitting on furniture. DECO/InteractVLM focus on object contact and ignore the scene. PhySIC handles scene contact but through slow optimization using DECO's noisy predictions. FECO handles feet only.

Scene contact is critical for physical plausibility: a person sitting in mid-air is obviously wrong. Adding floor/wall contact directly to the model enables physically grounded pose estimation.

---

## Architecture

```
                    +------------------+
                    |   SAM3DB Encoder |   (FROZEN)
                    +--------+---------+
                             |
                        Image Features F
                             |
              +--------------+--------------+
              |                             |
     +--------v--------+          +--------v--------+
     |  Body Decoder    |          | Interaction     |
     |  (FROZEN)        |          | Decoder         |
     |                  |          |                  |
     |  T_pose ---------|--body--->| T_contact (K=16)|
     |                  |  tokens  | T_object (1-2)  |
     +--------+---------+          | T_scene (2-4)   | <-- NEW
              |                    |                  |
         Body Mesh                 | Cross-Attn to F  |
                                   | Cross-Attn to    |
              +---+                |   body T_pose    | (from Step 1)
              |Dep|--depth-------->| Cross-Attn to    |
              |Pro|                |   scene prompts  | <-- NEW
              +---+                +--------+---------+
                                            |
                               +-----+-----+------+
                               |     |            |
                          Contact  Scene      Floor Plane
                          Head     Contact    Head
                                   Head
```

---

## New Components

### Scene Query Tokens
- 2-4 learnable tokens added to interaction decoder
- Dedicated to aggregating scene geometry information
- Attend to image features and scene prompt tokens

### Scene Prompt Encoders

| Prompt | Encoder | Input | When Available |
|--------|---------|-------|---------------|
| Depth map | 4-layer CNN (Conv-BN-ReLU, stride 2) -> flatten -> Linear(d_model) | HxW depth from DepthPro/MoGe | Always (run monocular depth) |
| Floor plane | Linear(4, d_model) | [a, b, c, d] plane coefficients | RANSAC on depth or GT |
| No scene prompt | Learned "no-scene" embedding | None | Fallback mode |

**Depth map encoding**: Encode depth map into tokens that the interaction decoder can attend to. Lightweight CNN produces spatial tokens at 1/16 resolution, projected to d_model.

### Scene Contact Head
From scene query tokens, predict:

| Output | Representation | Loss |
|--------|---------------|------|
| Floor contact vertices | Per-vertex probability for floor contact | BCE |
| Wall contact vertices | Per-vertex probability for wall contact | BCE |
| Floor plane parameters | [a, b, c, d] in camera/pelvis frame | L1 |
| Contact target type | {floor, wall, furniture_surface} per contact token | CE |

### Stability Loss (from IPMAN)
```
L_stability = ||CoP_xy - CoM_xy||^2

CoP: pressure-weighted average of floor-contacting vertex positions
     projected onto the ground plane
CoM: volume-weighted center of mass of SMPL body (differentiable, from IPMAN)

Only active for standing poses (feet are primary contact).
```

### Floor Contact Loss
```
L_floor = sum over v in V_floor of |h(v) - 0|    (h = signed height above floor)
L_push  = sum over v of max(0, -h(v))^2           (penalize below-floor vertices)
L_pull  = sum over v in V_floor of max(0, h(v))^2 (snap hovering contact to floor)
```

---

## Training

**What trains**: Scene query tokens + scene prompt encoders + scene contact head + stability loss
**What's frozen**: Encoder + body decoder

**Data**:

| Dataset | Images | Scene Contact GT | Floor Info |
|---------|--------|-----------------|------------|
| PROX | ~100K frames | Proximity-based (SDF threshold 2.5cm, 5cm feet) | 12 scene scans |
| RICH | 577K images (90K bodies) | Proximity-based (threshold + normal compat.) | 5 scene scans |
| COFE | ~31K images | Foot contact (3 keypoints + dense) | Ground plane fitted |
| MoYo | 1.75M frames | Pressure mat GT + CoM | Known floor plane |
| DAMON | 4.4K images | Body contact (some is floor/furniture) | No explicit floor |

**Contact GT generation for PROX/RICH**: For each body vertex, compute distance to scene mesh. If distance < threshold AND surface normal is compatible, label as scene contact. Separate floor vs. wall vs. furniture using scene semantic labels (available in PROX-E with 40 Matterport3D categories).

**Training modes**:
1. **With depth prompt** (50%): Provide monocular depth map
2. **With floor plane prompt** (20%): Provide GT floor plane
3. **Without scene prompt** (30%): Model must infer from image alone

---

## Evaluation

| Metric | Dataset | Comparison |
|--------|---------|------------|
| Floor contact F1 | PROX test | vs. PhySIC, FECO |
| Wall contact F1 | PROX test | vs. PhySIC |
| Foot contact F1 | COFE test | vs. FECO |
| Physical plausibility (penetration) | PROX, RICH | vs. PhySIC |
| Stability (CoP-CoM distance) | MoYo | vs. IPMAN |
| Body contact F1 | DAMON test | No degradation vs. Step 3 |

**Key comparison**: PhySIC achieves F1=0.51 on PROX and F1=0.43 on RICH for contact. Our model should be competitive while being ~100x faster (feed-forward vs. 27s optimization).

---

## Ablations

1. With depth prompt vs. without depth prompt (does depth help?)
2. With floor plane vs. inferred floor plane
3. Stability loss weight: 0 vs. 0.01 vs. 0.1
4. Scene tokens count: 2 vs. 4
5. Floor contact from scene tokens vs. from existing contact tokens (separate vs. unified)

---

## Success Criteria

- Floor contact F1 on PROX >= 0.45 (competitive with PhySIC's 0.51)
- Foot contact on COFE competitive with FECO
- No degradation on DAMON body contact
- Qualitative: standing people have feet on floor, sitting people have buttocks on surface
- Inference is real-time (< 100ms) vs. PhySIC's 27s

---

## References

- PhySIC (Muralidhar 2025): Scene contact optimization, F1=0.51 on PROX, 0.43 on RICH
- PROX (Hassan 2019): Scene SDF, contact vertices, penetration loss
- POSA (Hassan 2021): Per-vertex contact + 40 semantic labels on scenes
- IPMAN (Tripathi 2023): CoP-CoM stability loss, differentiable pressure, MoYo dataset
- FECO (Jung 2025): Dense foot contact, COFE dataset, ground-aware learning
- RICH/BSTRO (Huang 2022): 577K images, scene contact from proximity, MVM training
