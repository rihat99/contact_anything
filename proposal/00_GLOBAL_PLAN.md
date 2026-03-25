# SAM 3D Contact: Global Plan

## The Problem

No single model jointly predicts human pose, dense contact, and object/scene spatial arrangement from one image. Current methods are either:
- **Body-only contact** (DECO): binary per-vertex labels, no object awareness, no pose feedback
- **Pipeline-based** (PICO, InteractVLM, PhySIC): chain of separate models, error accumulation, slow
- **Specialized** (FECO: feet only, IPMAN: floor stability only, CARI4D: hand binary only)

**Gap**: A promptable, end-to-end model that reasons about pose, contact, and objects/scenes jointly with iterative refinement.

---

## Current State

You have SAM 3D Body with learnable contact query tokens and a contact head trained on DAMON, achieving results slightly above DECO but below InteractVLM.

---

## Landscape

| Method | Pose | Body Contact | Object Contact | Scene Contact | Correspondences | Object Pose | End-to-End |
|--------|------|-------------|----------------|---------------|-----------------|-------------|------------|
| DECO | No | Binary vertex | No | No | No | No | Yes |
| InteractVLM | No | Semantic vertex | Affordance proxy | No | No | No | No (pipeline) |
| PICO | Refine | Binary vertex | Retrieved | No | Retrieved | Optimized | No (pipeline) |
| LEMON | No | Dense vertex | Affordance | No | Implicit | Spatial relation | Partial |
| PhySIC | Refine | From DECO | No | Floor/scene | No | Scene aligned | No (optim) |
| **Ours (target)** | **Joint** | **Rich + semantic** | **Predicted** | **Predicted** | **Predicted** | **Predicted** | **Yes** |

---

## Target Architecture (High Level)

```
Image --> [SAM3DB Encoder] --> [Image Features F] --> [Body Decoder]
                                      |                    |
                                      |              T_pose tokens (detached)
                                      |                    | (body→contact, from Step 1)
      [Prompts] --> [Prompt Encoder] -+--> [Interaction Decoder]
                                      |        |          |
                                      |   [Contact Field] [Object 6DoF]
                                      |        |
                                      |   (contact→body, Step 7 only, gated)
                                      |
                                      +--> [Scene Tokens] --> [Floor/Wall Contact]
```

See `02_FINAL_ARCHITECTURE.md` for details.

---

## Sub-Projects (Steps)

Each step is ~1 week of work, builds on the previous, and is independently publishable/evaluable.

| Step | Focus | Key Output | Cross-Attention | Key Data |
|------|-------|------------|-----------------|----------|
| **Step 1** | Contact tokens + interaction decoder | Binary contact head + body→contact cross-attn | Body T_pose → contact (one-directional) | DAMON + RICH |
| **Step 2** | Object mask prompts | Contact conditioned on which object | Body→contact (inherited) | DAMON + object masks from SAM |
| **Step 3** | Contact representation ablation | Best repr: CDF vs. binary vs. correspondence | Body→contact (inherited) | DAMON, BEHAVE, PICO-db |
| **Step 4** | Scene contact (floor/wall) | Floor + wall contact prediction | Body→contact (inherited) | PROX, RICH, COFE, MoYo |
| **Step 5** | Hand contact integration | Contact tokens feed into hand decoder | Body→contact + hand↔contact | BEHAVE, InterCap, DexYCB |
| **Step 6** | Object position prediction | Object 6DoF relative to body | Body→contact + hand↔contact | BEHAVE, InterCap, 3DIR |
| **Step 7** | Mutual refinement (reverse direction) | Add contact→body, iterative loop | **Bidirectional** body↔contact | All datasets combined |

See individual `step*.md` files for detailed plans.

---

## Data Summary

| Dataset | Size | Contact Type | Object/Scene Info | Role |
|---------|------|-------------|-------------------|------|
| DAMON | 5.5K images | Dense binary body vertex | Object class (84 categories) | Primary body contact |
| HOT | 35K images | 2D contact heatmaps + body part | Object class | Not used (PAL dropped — coarse 2D annotations, marginal gain per DECO ablations) |
| RICH | 577K images (90K bodies) | Dense vertex contact | 5 scene scans | Body-scene contact |
| BEHAVE | 15K RGBD frames | Object vertex contact + SMPL correspondence | 20 object meshes | Body-object correspondence |
| InterCap | Multi-view | Contact from proximity | 10 object categories | Evaluation + training |
| PICO-db | 4.1K images | Body + object contact + correspondences | Retrieved meshes (627 instances) | Correspondence supervision |
| PROX | 100K frames | Proximity-based body-scene | 12 scene scans | Scene contact + floor/wall |
| 3DIR | 5K images | Contact + affordance + spatial relation | Meshes (21 object classes) | Joint relation supervision |
| PIAD | 7K point clouds + 5K images | Object affordance (17 types) | Point clouds | Object-side contact |
| MoYo | 1.75M frames | Pressure mat GT + CoM | Floor plane | Physical plausibility |
| COFE | 33K images | Foot contact (3 keypoints) | Ground info | Foot contact |

---

## Key Risks

| Risk | Mitigation |
|------|------------|
| Limited contact data (5.5K DAMON vs. 7M SAM3DB) | Freeze encoder; pull RICH (577K) into training from Step 1 for regularization; dropout + strong weight decay in interaction decoder |
| Interaction decoder hurts body pose | Body→contact is risk-free (frozen decoder, detached tokens); contact→body (Step 7) uses gated bridge starting at 0 |
| Object position from single image is ambiguous | Start with relative position; contact constrains arrangement |
| Too many loss terms cause instability | Staged curriculum; loss weighting search |

---

## Evaluation Plan

| Task | Metrics | Datasets |
|------|---------|----------|
| Body contact (binary) | F1, Precision, Recall, Geodesic error | DAMON, RICH |
| Body contact (continuous) | AUC, Geodesic error (cm) | DAMON |
| Contact correspondence | Mean correspondence error (mm) | BEHAVE, InterCap, PICO-db |
| Object 6DoF | Rotation error (deg), Translation error (cm) | BEHAVE, InterCap |
| Floor/wall contact | F1 on PROX, RICH; stability (CoP-CoM) on MoYo | PROX, RICH, MoYo |
| Joint pose + contact | MPJPE with/without contact feedback | All |
| Physical plausibility | Penetration depth, floating distance | All |

---

## File Index

| File | Contents |
|------|----------|
| `00_GLOBAL_PLAN.md` | This file -- master plan |
| `01_CONTACT_REPRESENTATIONS.md` | Contact representation options and comparison |
| `02_FINAL_ARCHITECTURE.md` | Full target architecture |
| `step1_contact_tokens_and_head.md` | Step 1: Contact tokens + interaction decoder |
| `step2_object_mask_prompts.md` | Step 2: Object mask prompts |
| `step3_contact_representation_ablation.md` | Step 3: Contact representation ablation |
| `step4_scene_contact.md` | Step 4: Scene contact (floor/wall) |
| `step5_hand_contacts.md` | Step 5: Hand contact integration |
| `step6_object_position.md` | Step 6: Object position prediction |
| `step7_mutual_refinement.md` | Step 7: Mutual refinement loop |
