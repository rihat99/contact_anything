"""Precompute the frozen SAM 3D Body pose token for the ClimbingVideos corpus.

Stage 2 trains on the frozen model's FINAL pose token (decoder sequence index 0;
``data.pose_token_cache``), which is a fixed function of the person-frame. This
writes it once for every valid ``(scene, person, frame)`` row as ONE npz per
scene (:func:`data.climbing_videos.scene.pose_token_path`):

    tokens (P, N, C) int16   bf16 bits of the token (zeros on invalid rows)
    valid  (P, N)   bool     rows that carry a token (= the scene's valid_mask)
    object_ids (P,)          the scene's person ids, in tracking order
    img_wh (N, 2)  int64     full-frame (W, H) — the crop geometry needs it and
                             the training loader opens no image afterwards

The token is computed on the live training path: the dataset's own
:meth:`~data.climbing_videos.dataset.ClimbingVideosDataset._frame` (corpus JPEG
+ SAM-3 mask + bbox) -> the training collate -> the frozen wrapper (backbone +
decoder) in chunks of ``--chunk`` frames. Complete scenes are skipped (safe to
re-run); writes are atomic. Shard over GPUs by scene, one process per GPU:

    CUDA_VISIBLE_DEVICES=2 python scripts/data/precompute_pose_tokens.py \\
        --split all --shard-index 0 --num-shards 8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data.climbing_videos.dataset import ClimbingVideosDataset          # noqa: E402
from data.climbing_videos.scene import list_scenes, pose_token_path     # noqa: E402
from data.collate import batch_to_device, make_collate                  # noqa: E402
from data.transforms import crop_size                                   # noqa: E402
from model.wrapper import SAM3DBodyWrapper                              # noqa: E402
from train.config import load_config                                    # noqa: E402

DEFAULT_ROOT = "/home/rikhat.akizhanov/better/data/ClimbingVideos"
GEOMETRY_KEYS = ("bbox_center", "bbox_scale", "ori_img_size", "img_size",
                 "affine_trans", "cam_int", "mask", "mask_score")


class SceneFrames(Dataset):
    """Every valid person-frame of one scene as a one-frame clip (training's frame dict)."""

    def __init__(self, dataset: ClimbingVideosDataset, scene: str, data: dict):
        self.dataset, self.scene, self.data = dataset, scene, data
        valid = data["valid_mask"]
        self.rows = [(p, f) for p in range(valid.shape[0])
                     for f in np.flatnonzero(valid[p]).tolist()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> list[dict]:
        person, pos = self.rows[index]
        frame = self.dataset._frame(self.scene, self.data, person, pos, 0, np.array([pos]))
        frame["_row"] = index
        return [frame]


def compute_scene(wrapper: SAM3DBodyWrapper, dataset: ClimbingVideosDataset, scene: str,
                  collate, *, chunk: int, num_workers: int) -> dict:
    """Tokens of every valid person-frame of ``scene`` -> the cache arrays."""
    data = dataset._load_scene(scene)
    frames = SceneFrames(dataset, scene, data)
    n_people, n = data["valid_mask"].shape
    dim = wrapper.decoder_dim
    tokens = np.zeros((n_people, n, dim), np.int16)
    valid = np.zeros((n_people, n), bool)
    img_wh = np.zeros((n, 2), np.int64)
    for pos in range(n):
        with Image.open(data["frames_dir"] / f"{pos:06d}.jpg") as im:
            img_wh[pos] = im.size
    loader = DataLoader(frames, batch_size=chunk, num_workers=num_workers,
                        collate_fn=collate, pin_memory=True)
    with torch.inference_mode():
        for batch in loader:
            rows = batch.pop("_row").tolist()
            batch = batch_to_device(batch, "cuda")
            out = wrapper(img=batch["img"], blocks=[], **{k: batch[k] for k in GEOMETRY_KEYS})
            bits = (out["tokens"][:, 0].to(torch.bfloat16).contiguous()
                    .view(torch.int16).cpu().numpy())
            for row, index in zip(bits, rows):
                person, pos = frames.rows[index]
                tokens[person, pos] = row
                valid[person, pos] = True
    if not np.array_equal(valid, data["valid_mask"]):
        raise RuntimeError(f"{scene}: cached rows do not match valid_mask")
    return {"tokens": tokens, "valid": valid, "object_ids": data["object_ids"],
            "img_wh": img_wh, "dim": np.array(dim)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/base.yaml",
                        help="any config: only model.checkpoint_path / mhr_model_path are read")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--token-dir", default=None,
                        help="output root (default: <root>/features/pose_token)")
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=48, help="frames per frozen pass")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-scenes", type=int, default=0, help="0 = all (smoke)")
    parser.add_argument("--scenes", nargs="*", default=None, help="explicit scene ids (smoke)")
    args = parser.parse_args()

    corpus_root = Path(args.root)
    token_root = Path(args.token_dir) if args.token_dir else corpus_root / "features" / "pose_token"
    splits = ("train", "test") if args.split == "all" else (args.split,)
    scenes = args.scenes or [s for split in splits for s in list_scenes(corpus_root, split)]
    scenes = scenes[args.shard_index::args.num_shards]
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]
    todo = [s for s in scenes if not pose_token_path(token_root, s).is_file()]
    print(f"shard {args.shard_index}/{args.num_shards}: {len(scenes)} scenes, "
          f"{len(todo)} to compute", flush=True)
    if not todo:
        return

    cfg = load_config(args.config)
    wrapper = SAM3DBodyWrapper(
        cfg["model"]["checkpoint_path"], cfg["model"]["mhr_model_path"]).to("cuda")
    wrapper.eval()
    collate = make_collate(crop_size(cfg["model"]["checkpoint_path"]))
    # A scene-less dataset: only its per-frame assembly is used (labels play no
    # part, so every scene is read on the automatic-label path).
    dataset = ClimbingVideosDataset(corpus_root, scenes=[], split="train", clip_frames=1, load=())

    start = time.time()
    frames_done = 0
    failed: list[str] = []
    for scene in tqdm(todo, desc="scenes"):
        try:
            arrays = compute_scene(wrapper, dataset, scene, collate,
                                   chunk=args.chunk, num_workers=args.num_workers)
        except Exception as exc:                          # noqa: BLE001
            failed.append(f"{scene}: {exc}")
            continue
        path = pose_token_path(token_root, scene)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.replace(path)
        frames_done += int(arrays["valid"].sum())
    elapsed = time.time() - start
    print(f"wrote {len(todo) - len(failed)} scenes / {frames_done} tokens in "
          f"{elapsed:.0f}s ({frames_done / max(elapsed, 1e-9):.1f} frames/s)")
    for line in failed:
        print(f"  FAILED {line}")


if __name__ == "__main__":
    main()
