# Step 02 — Camera extrinsics: pipeline export + dataset + loader plumbing

Independent of step 01. Read `plan/README.md` §2 (reconstruction pipeline block), D7, D13,
R2–R4. This step spans two repos: the exporter lives in
`/data3/rikhat.akizhanov/better/BetterVideoReconstruction` (we may modify it), the loader in
this repo. Goal: every ClimbingVideos_v1 scene carries per-frame metric camera extrinsics
and a per-scene gravity direction; the training batch exposes them.

## Verified facts (2026-07-22 — re-verify line numbers, trust names)

- Pipeline working data: `/data3/rikhat.akizhanov/better/data/ClimbingVideos/`. Per-scene
  cameras: `features/geometry/<sid[:2]>/<sid[2:4]>/<sid>/transform.npz` with
  `extrinsics (N,4,4) f32` — **camera-from-world (w2c), OpenCV convention** (x right,
  y down, z forward; docstring in `scripts/stages/estimate_camera_vggt.py:18`), rotation as
  3×3 matrix, bottom row `[0,0,0,1]`. After the scale stage the translation is **metric**:
  `scale` (cumulative factor, e.g. 7.4647), `metric=True`, `scale_method` recorded;
  `estimate_scale.py::_metricize_geometry` (~273–302) does `extr[:, :3, 3] *= s`.
  Frame 0 ≈ identity (VGGT anchors the first camera), so world ≈ metric camera-0 frame.
- Frame alignment: `frame_indices` is sequential `0..N-1`; `extrinsics[k]` ↔ exported
  `frames/{k:06d}.jpg` ↔ row k of every per-frame array (the exporter asserts this via
  `_require_sequential_frames`).
- Exporter: `BetterVideoReconstruction/scripts/export_contact_dataset.py`. `export_scene`
  (~265–313) already `np.load`s `geometry/transform.npz` (~276) but reads only
  `intrinsics_px_orig` (~284). A `common` dict (~303–313) lands in BOTH train `labels.npz`
  (~352) and test `inputs.npz` (~359). `dataset_info.json` schema text is built at ~721–739.
  The exporter has sha256 staleness checks (~337–347) that will *skip* already-exported
  scenes — adding output keys does not change input hashes.
- Exported dataset: `/data3/rikhat.akizhanov/datasets/ClimbingVideos_v1` — 331 train + 30
  test scenes, `schema_version=2`. **No extrinsics anywhere today**; only per-frame
  `intrinsics`.

## Part A — exporter (BetterVideoReconstruction)

1. In `export_scene`, read from the already-loaded `transform.npz`: `extrinsics`
   (validate shape `(N,4,4)` next to the existing checks), `scale`, `metric`.
   **Hard-fail with the scene id if `metric` is not True** — never export up-to-scale
   cameras silently.
2. Add to the `common` dict (so labels.npz and inputs.npz stay symmetric):
   - `extrinsics (N,4,4) f32` — verbatim camera-from-world, OpenCV, metric;
   - `gravity_world (3,) f32` — unit vector, downward in world, computed as
     `R_c2w[0] @ [0,1,0]` with `R_c2w[0] = extrinsics[0,:3,:3].T` (first camera's +y =
     image-down ≈ physical down for a level camera; with frame-0-anchored worlds this is
     ≈ `[0,1,0]`). Normalize; this key is the *assumption made explicit* — if the pipeline
     later estimates true gravity, only this exporter line changes;
   - `cam_scale () f32` — provenance for scale debugging (risk R4).
3. `contacts.npz` / `--fill-test` need no change (extrinsics ride in `inputs.npz`).
4. Document the three keys in the `dataset_info.json` schema text; bump `schema_version`
   to 3.
5. **Backfill**: the 331+30 existing scenes must gain the keys **without re-decoding
   videos** (frames/masks/labels are expensive and unchanged). Add a small backfill mode or
   script that, per exported scene, loads its `labels.npz`/`inputs.npz`, reads the scene's
   `transform.npz`, asserts `N` matches the npz's per-frame arrays, and rewrites the npz
   atomically (tmp + rename) with the new keys. Idempotent (running twice is a no-op).
   Run it over both splits; report scene counts and any scene lacking a metric
   `transform.npz` (fail loudly, don't skip silently).
6. Verify on a sample of ≥5 scenes per split: keys present, `R@R.T≈I` (|det−1|<1e-4) per
   frame, `gravity_world` unit norm, translation magnitudes plausible (meters, camera
   center `-Rᵀt` moving smoothly).

## Part B — loader + collate (this repo)

7. `contact/data/climbing_videos.py::_load_scene`: read `extrinsics` and `gravity_world`
   from the npz (labels.npz / inputs.npz). **Hard-require them** for this dataset after the
   backfill — a missing key means a stale export; raise with the scene id and a pointer to
   the backfill script. In `__getitem__`, add per-frame fields:
   `"cam_from_world": data["extrinsics"][pos]` (`[4,4] f32`) and
   `"gravity_world": data["gravity_world"]` (`[3] f32`, same for all frames of a scene).
8. `contact/data/collate.py::make_collate`: stack into the batch —
   `out["cam_from_world"] [B,4,4] f32`, `out["gravity_world"] [B,3] f32`,
   `out["cam_valid"] [B] bool`. Frames from datasets that don't provide cameras (all
   still-image datasets) get identity extrinsics, zero gravity, `cam_valid=False`. Keep it
   to a few lines mirroring the `frame_pos_sec` handling.
9. No model or config change — the model never consumes extrinsics (D13); they flow only
   into the physics loss (step 06).

## Tests (this repo, CPU fast suite)

- Loader: build a tiny synthetic scene dir (mirror any existing loader-test fixture
  pattern) with `extrinsics`/`gravity_world` in the npz → clip frames carry the right
  per-frame `cam_from_world` (window offsets respected) and constant `gravity_world`;
  a scene npz *without* the keys raises the stale-export error.
- Collate: a mixed batch of synthetic video frames (with cams) and a still frame (without)
  yields correct `cam_valid`, identity fallback, shapes `[B,4,4]`/`[B,3]`.

## Acceptance

- Backfill ran over the full dataset; reported `train=331, test=30` (or documented
  exceptions) with verification (item 6) passing.
- This repo's fast suite green, including the new loader/collate tests against synthetic
  fixtures (no dependence on the real dataset in the fast suite).
- One real-scene spot check (script or slow test): load a clip via `ClimbingVideosDataset`,
  confirm `cam_from_world` matches the scene's `transform.npz` rows for the window
  positions.

## Out of scope

Adapter math and any use of the extrinsics (step 03), physics loss (step 06), gravity
estimation beyond the first-camera rule, re-export of frames/masks/labels.
