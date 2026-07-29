# legacy/

Code kept for provenance but no longer wired into the active pipeline. Nothing
here is imported by `contact/`, `scripts/`, or `tools/` (`viewer/` is the one
deliberate exception — see `climbing_videos.py`). Each entry says why it is
legacy, what functionality is lost by retiring it, and how to resurrect it.

## `damon_sam3_segment.py`

SAM3 open-vocabulary segmentation for DAMON: it segmented `"person"` (object
order 0) **and** each contact object name from `contact_label_objectwise`
(skipping `"supporting"`), writing per-object masks + bboxes + scores.

- **Why legacy:** person masks are now produced by
  `scripts/precompute_masks_damon.py` (single `"person"` prompt, largest-bbox
  detection), which is all the training pipeline consumes.
- **Functionality lost:** the **per-contact-object** masks/bboxes (segmentation
  of the objects a person is touching). The active precompute path produces only
  the person mask; contact-object segmentation is not replicated anywhere.
- **Resurrect:** run this script against the DAMON split to regenerate the full
  `{split}/{idx:04d}/` metadata + per-object mask tree; re-point any consumer at
  that output layout.

## `damon_sam3d_precompute.py`

Orphan precompute: ran the SAM-3D-Body body decoder over DAMON to dump pose
predictions (keypoints, MHR params, camera) per split and DINOv3 encoder
features per sample (`[1280, 56, 56]` float16).

- **Why legacy:** nothing in the current training/eval loop reads these dumps;
  the model is run live from images each step.
- **Functionality lost:** the pose-prediction dump and the **feature cache**.
  Note the cached features were **post-mask-conditioning** embeddings (mask + ray
  conditioning already fused into the decoder's image tensor). A future feature
  cache must instead store **raw backbone outputs** (pre mask-conditioning) plus
  a manifest (checkpoint hash, crop affine, precision) and a live-vs-cache
  equivalence check — otherwise it bakes in one mask/camera and cannot be reused.
- **Resurrect:** re-run to regenerate the dumps, but rewrite the feature path to
  cache raw backbone features before reusing them (see note above).

## `contact.py` (`ContactDataset`)

Unified loader that concatenated DAMON + LEMON + RICH behind one index-
translation layer.

- **Why legacy:** superseded by explicit `data.datasets: [{name, config, split}]`
  lists in the training configs, which `contact/data/collate.py::make_loaders`
  concatenates. Its export was removed from the datasets package
  (`contact/data/__init__.py` no longer re-exports `ContactDataset`).
- **Functionality lost:** the single `names=[...]`-addressable concat wrapper.
- **Resurrect:** `from legacy.contact import ContactDataset` (the individual
  loaders it wraps now live in `contact/data/`).

## `climbing_videos.py` (`ClimbingVideosDataset`)

Training/eval loader for the **exported** ClimbingVideos_v1 dataset
(`/data3/rikhat.akizhanov/datasets/ClimbingVideos_v1`, `train/` + `test/`
directories produced by BetterVideoReconstruction's `export_contact_dataset.py`).

- **Why legacy:** training now reads the raw pipeline corpus directly via
  `contact/data/climbing_corpus.py` (`ClimbingCorpusDataset`, which duck-types
  this class), so the export step — and this loader — left the training loop.
  A config naming dataset `climbing_videos` now hard-errors with a pointer here.
- **Functionality lost:** reading the *exported* v1 tree (per-scene
  `contacts.npz` with the `pending` test-label flag, v1's camera-0-derived
  `gravity_world`, `cam_scale`). The corpus loader reads `features/` +
  `frames/` instead and uses the exact kindyn `[0, 1, 0]` gravity.
- **Still imported by `viewer/`** (as `legacy.climbing_videos`): the dataset
  browser still displays the v1 export from disk (user decision); its relative
  import was rewritten absolute so the module works from `legacy/`.
- **Resurrect:** `from legacy.climbing_videos import ClimbingVideosDataset`
  against an existing v1 export.

## `demo_climbing_videos.py`

Qualitative still-panel demo for v1 video checkpoints: GT-vs-predicted contact
disks per extremity plus force arrows drawn via the model's own intrinsics with
screen-space length.

- **Why legacy:** it is hard-wired to the v1 loader/dataset entry
  (`ClimbingVideosDataset`, `name: climbing_videos` configs). Qualitative video
  checks now go through `scripts/render_climbing_video_contacts.py` (ported to
  the corpus loader; per-frame dataset-intrinsics force projection).
- **Functionality lost:** the still-image *panel montage* output (one grid image
  per sample rather than a rendered video), and its simpler screen-space force
  arrows.
- **Resurrect:** run it from the repo root against a v1 export after restoring
  the imports it shares with the retired loader (it imports
  `contact.data.climbing_videos`, now `legacy.climbing_videos`).

## `configs/`

The retired experiment yamls from the 2026-07 corpus sweep: the superseded
climbing_videos_* family (`_joint_temporal{,_causal,_center}`,
`_force_scratch{,_temporal}`, `_force_warmstart{,_temporal,_t5temporal,_t16,_t7mid}`)
and the still-image baselines (`damon_baseline`, `climbing_baseline`,
`climbing_damon_baseline`).

- **Why legacy:** `configs/` now keeps only `base.yaml`, the four configs
  matching the runs in `output/` (flattened into self-contained overrides of
  `base.yaml`, verified to resolve identically before flattening), the
  supervised-force corpus experiment, and `configs/datasets/*`.
- **Functionality lost:** none as *records* — every recipe and its annotated
  rationale is preserved here, and the kept configs inline the rationale of
  their former include chains. Note the `base:` include chains in these files
  point at `configs/` paths that have since been flattened or moved, so most no
  longer `load_config` as-is.
- **Resurrect:** copy one back into `configs/` and re-point (or inline) its
  `base:` chain; for the still-image baselines also note `damon_baseline` was
  the only config exercising the vertex target end-to-end.

## `tests/`

`test_climbing_videos.py` and `test_demo_climbing_videos.py` — the unit tests of
the two modules above, moved with them.

- **Why legacy:** they import `contact.data.climbing_videos` /
  `scripts.demo_climbing_videos`, both retired; a `conftest.py` guard keeps
  pytest from collecting them (the active suite lives in `tests/`, where the
  corpus loader has its own coverage in `test_climbing_corpus.py`).
- **Resurrect:** move back alongside resurrected modules and fix the imports.

## `mhr_smpl_conversion/`

Standalone bidirectional SMPL (6890) ↔ MHR LOD1 (18439) and MHR LOD→LOD
converter (`body_converter.py`), driven by precomputed barycentric mapping files.

- **Why legacy:** the required barycentric mapping assets are **missing on disk**,
  so the module cannot run as-is.
- **Functionality lost:** SMPL↔MHR and MHR-LOD conversion (e.g. mapping MHR
  native 18439-vertex contact to SMPL 6890, or across LODs).
- **Resurrect:** restore the barycentric mapping asset files the module loads,
  then `from legacy.mhr_smpl_conversion.body_converter import ...`.
