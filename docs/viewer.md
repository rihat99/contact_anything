# Results viewer (2026-09-05)

*The runs whose dumps this page mentions (tb_projzero, static_ray, hands) were trashed in the
2026-09-05 simplification; the viewer itself is unchanged and reads any run dumped by
`scripts/predict_test.py` from the current code.*

A viser 3D viewer of the test-set predictions of every trained model next to the
kindyn ground truth and the frozen SAM 3D Body, in the spirit of
BetterVideoReconstruction's `scripts/view_3d.py`.

## Two steps

1. **Dump a run's predictions** (once per run, its best checkpoint):

   ```bash
   CUDA_VISIBLE_DEVICES=0 python scripts/predict_test.py \
       --config configs/baseline.yaml --checkpoint output/<run>/best.pth
   ```

   writes `output/<run>/predictions/<scene>.npz` + `manifest.json`. The test split
   is the config's dataset yaml (its `camera` filter included: `static_ray.yaml`
   dumps the 16 static test scenes, `baseline.yaml` every annotated test scene). Every
   contiguous tracked run of every person is predicted at the evaluation stride
   (`auto` = ~25 fps) in windows of `--max-frames` rows overlapping by `--overlap`
   rows; a row keeps the window it sits deepest inside. The defaults 240 / 120
   (the 2026-09-05 dumps; ~18 GiB peak on a 360-frame scene) put every row at
   least 60 rows inside its window. That is one per-layer RoPE window
   (`window` 2.5 s under `position: seconds`), not the block's full receptive field (4 layers, ~250
   rows per side), so the tiling is NOT bit-identical to a single whole-scene
   pass: measured on `4HuRoofxxMI_0002` (341 rows, TF32 off) the body-22 joints
   differ by 4.6 mm max / 0.33 mm mean, the pelvis by 3.1 mm max, contact
   probabilities by 0.025 max — negligible next to a 72 mm MPJPE. Smaller
   windows (120 / 30) leave only 15 rows at a crossover and are not recommended.

   Per person and source frame the npz holds the head's camera-frame BetterHuman
   configuration `q_cam (P, N, 211)` (identity finger quaternions for a hands-free
   head), `betas (P, N, 10)`, `joints_cam (P, N, J, 3)`, `pelvis_cam`, `covered`,
   `contact_probs (P, N, 6)` (contact builds), the dataset's `tracked` mask, the
   stride and the windows. The forward runs under TF32 like `evaluate.py`, so
   `joints_cam` is the head's own FK of `q_cam` to ~0.6 mm (the viewer re-runs
   the FK from `q_cam`).

2. **Serve the viewer** (every run with a dump, one server):

   ```bash
   CUDA_VISIBLE_DEVICES=5 python scripts/view_results.py --port 8090      # GPU only for the FK
   python scripts/view_results.py --device cpu --no-video --run <run dir name>
   ```

   Open `http://localhost:8090` (port-forward from the box). Port 8082 is the BVR
   viewer's, so this one defaults to 8090.

## What is drawn

Three bodies, one colour each, in the metric world frame:

| source | data | colour |
|---|---|---|
| predicted | the run's `predictions/<scene>.npz` (`q_cam` folded into the world with the frame extrinsics) | red |
| GT | `human_optim/kindyn_1.npz` (world `q`, one betas per person) | green |
| frozen | `sam3d/smplx_params.npz` — the frozen SAM 3D Body refit to SMPL-X by the corpus pipeline (classic params in the camera frame, converted to `q` per frame, folded into the world) | blue |

Each body has one checkbox (its mesh and skeleton together); a global **skeletons
(all)** switch shows the 22 body joints + bones of every shown body; one slider
sets the mesh opacity. Layers: the fused scene cloud
(`geometry/scene.npz`), the cameras, the gravity arrow (world regime). The
sidebar **video** pane shows the corpus JPEG frame of the slider position;
**playback** plays at the scene's fps; **onboard camera** rides the source camera.

### The two regimes (`view / frame`)

Everything is built once in the world frame under one root node; the regime is
that root's pose.

* **camera** — the root is posed at `cam_from_world(f)` every frame, so the world
  is re-expressed in the current camera's OpenCV axes: the camera is a fixed
  frustum at the origin, the predicted and frozen bodies appear exactly as the
  models output them (verified: lifting and re-expressing is exact to 1e-7 m), and
  the GT is lifted INTO the camera the way the losses see it. Up is `-y`.
* **world** — the root is the identity; the per-frame frustums, the camera path,
  the current-frame cursor and the gravity arrow show; up is the scene's fitted
  gravity (`kindyn_1.npz gravity_world`).

The corpus "static" scenes still carry per-frame extrinsics, which is why the
camera regime is per frame rather than a fixed transform.

### Approximations (viewer only, never in the metrics)

* Meshes are viser **skinned meshes** (LBS in the browser, 52 bone poses per
  frame, one upload per person). Browser LBS drops SMPL-X's pose correctives:
  ~6 mm mean / 26 mm max on a climbing pose against BetterHuman's own vertices
  (the same approximation the BVR viewer makes).
* The predicted and frozen heads regress **betas per frame**; the mesh is
  uploaded once at the person's median identity, while the skeleton and the bone
  poses come from the exact per-frame FK. The per-frame betas spread is printed in
  the info panel (`betas per-frame std`, ~0.1 for tb_projzero, ~0.3 for the frozen
  refit).
* Scenes with a prediction stride > 1 (60 fps sources) show the previous predicted
  row on the tracked frames in between (`held` count in the info panel); the
  metrics ignore those rows.
* The scene cache keeps the last 6 scenes (~100 MB each with the video pane).

The info panel also reports, per scene, each source's body-22 MPJPE (mean-hips
aligned) and absolute pelvis error against the GT over the frames both are valid
(real predictions only, never the held rows). The formula is `metric_pose/mpjpe`
/ `pelvis_err`'s (bit-identical on the frozen json), but the FRAME SET is not the
evaluation protocol's: the viewer averages every valid frame of the whole scene,
`docs/results.md` one clip per person capped at `eval_max_frames` (120 rows).
Over the 16 static scenes the whole-scene numbers run higher — frozen 59.9 vs
57.9 mm MPJPE and 119 vs 105 mm pelvis, tb_projzero 76.0 vs 72.8 mm — so compare
sources within the viewer, not a viewer number against the results table.

## Code

| file | role |
|---|---|
| `scripts/predict_test.py` | the dump (tiled whole-scene windows, manifest) |
| `scripts/view_results.py` | CLI |
| `viewer/bodies.py` | the three sources → one skinning payload (rest mesh, rest skeleton, per-frame world bone poses) via BetterHuman FK |
| `viewer/loading.py` | one `(run, scene)` → `SceneData` (cameras, footage, cloud, bodies, metrics) |
| `viewer/scene.py` | the viser nodes of one scene, the regime root pose, per-frame updates |
| `viewer/app.py` | the sidebar, run/scene switching, playback loop |

Not yet drawn: contacts and forces (the dump already stores `contact_probs`).

## Contacts and forces (2026-09-05)

The dump carries `contact_probs (P, N, 6)` and, for a build with a force head, `forces_world
(P, N, 6, 3)` — body-weight units in the WORLD frame (the refiner's body-frame forces rotated with
its own root). The viewer draws, on the body they belong to:

* **contact markers** — six discs at the group joints (wrists, toes, heels; body-22 joints
  20, 21, 10, 11, 7, 8), coloured grey → green by probability (predicted) or by the manual test
  label (GT; dark = not annotated);
* **force arrows** — one line per group from the joint along the force, `force scale` metres
  per body weight (default 0.3). GT forces come from `kindyn_1.npz` folded into the six groups
  and rotated to the world with the kindyn root.

Toggles live in the *contacts & forces* folder (predicted on, GT off by default). The info panel
adds the contact rate and the mean / max force magnitude per source. Stride-held frames repeat
the previous prediction's contacts and forces, like the body.
