# legacy/

Code kept for provenance but no longer wired into the active pipeline. Nothing
here is imported by `contact/`, `scripts/`, or `tools/`. Each entry says why it
is legacy, what functionality is lost by retiring it, and how to resurrect it.

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

## `mhr_smpl_conversion/`

Standalone bidirectional SMPL (6890) ↔ MHR LOD1 (18439) and MHR LOD→LOD
converter (`body_converter.py`), driven by precomputed barycentric mapping files.

- **Why legacy:** the required barycentric mapping assets are **missing on disk**,
  so the module cannot run as-is.
- **Functionality lost:** SMPL↔MHR and MHR-LOD conversion (e.g. mapping MHR
  native 18439-vertex contact to SMPL 6890, or across LODs).
- **Resurrect:** restore the barycentric mapping asset files the module loads,
  then `from legacy.mhr_smpl_conversion.body_converter import ...`.
