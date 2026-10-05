"""Prepare in-the-wild videos for ``scripts/predict_reconstruction.py``: person tracks + a camera.

For every ``<videos>/<stem>.mp4`` builds the out-tree ``<out>/<stem>/`` the reconstruction
loader reads (:class:`data.reconstruction.ReconstructionSceneDataset`):

``sam3/``
    BetterVideoReconstruction's SAM 3 tracker stage, run in ITS venv as a subprocess with
    ``configs/wild_sam3.toml``: the largest frame-0 ``person`` detection only, propagated in
    200-frame parts, so a video of any length works (masks per frame, ``bboxes.npz``).
``geometry/transform.npz``
    a STATIC pinhole camera: ``cam_from_world`` is the identity on every frame (the world IS
    the camera frame) and the intrinsics are MoGe-2's estimate on the first frame, held for
    the whole clip.

A stem that already has a full pipeline tree under ``--moving-root`` (the clips whose camera
moves: VGGT trajectory, metric scale, ``sam3/`` from the same source video; built by BVR's
``scripts/pipeline.py`` with ``configs/wild_pipeline.toml``) gets both folders symlinked from
there instead — nothing is recomputed.

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/prepare_wild.py \\
        --videos ../data/willd_videos/videos --out ../data/willd_videos/out --stems full_body_1
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
BVR = REPO.parent / "BetterVideoReconstruction"
SAM3_CONFIG = REPO / "configs" / "wild_sam3.toml"
MOGE_MODEL = "Ruicheng/moge-2-vitl-normal"


def track_people(video: Path, out_root: Path, config: Path) -> None:
    """BVR's SAM 3 stage on one video → ``<out_root>/<stem>/sam3/`` (skips a finished one)."""
    subprocess.run(
        [str(BVR / ".venv" / "bin" / "python"), str(BVR / "scripts" / "stages" / "track_sam3.py"),
         "--source", str(video.resolve()), "--save-dir", str(out_root.resolve()),
         "--config", str(config.resolve()), "--skip-existing"],
        cwd=BVR, check=True)


def first_frame(video: Path) -> tuple[np.ndarray, float]:
    """The first decodable frame (RGB) and the container fps."""
    cap = cv2.VideoCapture(str(video))
    ok, bgr = cap.read()
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot decode {video}")
    return bgr[..., ::-1].copy(), fps


def moge_intrinsics(model, image: np.ndarray, device: str) -> np.ndarray:
    """MoGe-2's pinhole intrinsics of one RGB uint8 image, in pixels ``(3, 3)``."""
    height, width = image.shape[:2]
    tensor = torch.as_tensor(image, device=device).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        normalized = model.infer(tensor)["intrinsics"].double().cpu().numpy()
    return (normalized * np.array([[width, width, width], [height, height, height], [1, 1, 1]],
                                  np.float64)).astype(np.float32)


def write_static_camera(video: Path, tree: Path, model, device: str) -> None:
    """``geometry/transform.npz`` of a static camera with MoGe-2's first-frame intrinsics."""
    boxes = np.load(tree / "sam3" / "bboxes.npz", allow_pickle=True)
    n_frames = int(boxes["num_frames"])
    image, fps = first_frame(video)
    height, width = image.shape[:2]
    if (int(boxes["video_height"]), int(boxes["video_width"])) != (height, width):
        raise ValueError(f"{tree.name}: the tracker saw {int(boxes['video_width'])}x"
                         f"{int(boxes['video_height'])}, the video decodes at {width}x{height}")
    intrinsics = np.tile(moge_intrinsics(model, image, device), (n_frames, 1, 1))
    path = tree / "geometry" / "transform.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, extrinsics=np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1)),
        intrinsics_px_orig=intrinsics, intrinsics=intrinsics,
        frame_indices=np.arange(n_frames, dtype=np.int32), fps=np.float32(fps),
        image_width=np.int32(width), image_height=np.int32(height),
        static_camera=np.bool_(True), metric=np.bool_(True), scale=np.float32(1.0),
        camera_source=f"static camera; intrinsics from {MOGE_MODEL} on the first frame")
    print(f"  {tree.name}: {n_frames} frames, {width}x{height} @ {fps:.3f} fps, "
          f"focal {intrinsics[0, 0, 0]:.0f} / {intrinsics[0, 1, 1]:.0f} px, "
          f"centre ({intrinsics[0, 0, 2]:.0f}, {intrinsics[0, 1, 2]:.0f})")


def link_pipeline_tree(source: Path, tree: Path) -> None:
    """Symlink ``sam3/`` and ``geometry/`` of a full pipeline tree into ``tree``.

    Every person the pipeline tree tracks is predicted (``configs/wild_pipeline*.toml``
    decide how many the tracker keeps).
    """
    n_objects = int(np.load(source / "sam3" / "bboxes.npz", allow_pickle=True)["num_objects"])
    tree.mkdir(parents=True, exist_ok=True)
    for name in ("sam3", "geometry"):
        link, target = tree / name, (source / name).resolve()
        if link.is_symlink() and link.resolve() == target:
            continue
        if link.is_symlink() or link.exists():
            raise FileExistsError(f"{link} exists and is not a link to {target}")
        link.symlink_to(target)
    print(f"  {tree.name}: sam3/ ({n_objects} people) and geometry/ linked from {source}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos", type=Path, required=True, help="directory of <stem>.mp4")
    parser.add_argument("--out", type=Path, required=True, help="root of the out-trees")
    parser.add_argument("--moving-root", type=Path, default=BVR / "out_2",
                        help="full pipeline trees of the clips whose camera moves")
    parser.add_argument("--stems", nargs="*", default=None, help="subset (default: every mp4)")
    parser.add_argument("--device", default="cuda", help="MoGe-2 device")
    parser.add_argument("--sam3-config", type=Path, default=SAM3_CONFIG,
                        help="the tracker stage's toml (wild_sam3_small.toml: a far, small athlete)")
    args = parser.parse_args()

    videos = {p.stem: p for p in sorted(args.videos.glob("*.mp4"))}
    stems = args.stems or list(videos)
    missing = [s for s in stems if s not in videos]
    if missing:
        raise SystemExit(f"no video for {missing} under {args.videos}")
    args.out.mkdir(parents=True, exist_ok=True)
    model = None
    for index, stem in enumerate(stems, start=1):
        print(f"[{index}/{len(stems)}] {stem}", flush=True)
        tree = args.out / stem
        pipeline_tree = args.moving_root / stem
        if (pipeline_tree / "geometry" / "transform.npz").is_file():
            link_pipeline_tree(pipeline_tree, tree)
            continue
        track_people(videos[stem], args.out, args.sam3_config)
        if (tree / "geometry" / "transform.npz").is_file():
            print(f"  {stem}: camera exists, skipping")
            continue
        if model is None:
            from moge.model.v2 import MoGeModel
            model = MoGeModel.from_pretrained(MOGE_MODEL).to(args.device).eval()
        write_static_camera(videos[stem], tree, model, args.device)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
