# Contact Atlas dataset viewer

A light, standalone web application for inspecting all contact datasets in this
repository. It is intentionally separate from `tools/` and loads datasets only
when selected.

```bash
/data3/rikhat.akizhanov/miniconda3/envs/sam3d/bin/python -m viewer --port 8765
```

Use `--no-open` on a remote machine. The API documentation is available at
`/api/docs` while the server is running.

## ClimbingVideos v1

- **Frame** mode uses the dataloader with `frames_per_clip=1`, so each tracked
  person-frame is a separate viewer instance.
- **Sequence** mode uses the native clip dataloader with configurable length and
  stride. Stride 1 shows consecutive frames as one instance.
- **Train/Test** selects the physical corpus split. Viewer windows are fixed
  (`mode="val"`, no jitter); this is separate from the grouped train/validation
  split used during model training.
- Contact joints are red, non-contact joints are green, and unannotated joints
  are hollow gray. When confidence display is enabled, red/green is interpolated
  toward gray as label confidence falls.

ClimbingVideos exports contact labels, confidence, masks, boxes, and camera
intrinsics, but it does not export 2-D or 3-D joint positions. The skeleton is
therefore a canonical SMPL-X body-22 diagram beside each image, not a pose
overlay. This is deliberate: the viewer does not run pose inference or depend
on reconstruction sidecars outside the dataset.

## Other datasets

DAMON, LEMON, RICH, and ClimbingImages retain the source-image plus canonical
SMPL contact-mesh view. Person mask outlines and bounding boxes can be toggled.

Keyboard shortcuts: Left/Right browse instances; `R` chooses a random instance.
