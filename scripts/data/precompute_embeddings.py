"""Precompute frozen-backbone embeddings for the ClimbingVideos corpus.

Training re-runs the frozen DINOv3 backbone over the same crops every epoch —
the dominant step cost. This writes, for every croppable ``(scene, person,
frame)`` row (``valid_mask`` — tracked frames with a good bbox, all persons of
multi-person scenes), the RAW bf16 ``[1280, 32, 32]`` backbone output to
``<corpus>/features/embedding/<s[0:2]>/<s[2:4]>/<scene>/<oid:02d>/<pos:06d>.npy``
as an int16 bit view (numpy has no bf16). The crop is bit-identical to
training: the same :func:`data.transforms.process_frame` transform on the same bbox,
then ``data_preprocess`` + backbone in bf16. Mask and ray conditioning are NOT
baked in — they run live downstream of the cache.

Training consumes the cache via ``data.embedding_cache: true``.

Complete files are skipped (safe to re-run); writes are atomic (tmp + rename).
Shard over GPUs by scene, one process per GPU:

    CUDA_VISIBLE_DEVICES=0 python scripts/data/precompute_embeddings.py \
        --split all --shard-index 0 --num-shards 4
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data.climbing_videos.scene import (                              # noqa: E402
    embedding_path, list_scenes, load_scene)
from data.transforms import build_transform, process_frame            # noqa: E402
from model.wrapper import SAM3DBodyWrapper                            # noqa: E402
from train.config import load_config                                  # noqa: E402

DEFAULT_ROOT = "/data3/rikhat.akizhanov/better/data/ClimbingVideos"


class CropDataset(Dataset):
    """Training-identical person crops for a list of cache entries.

    Each entry is ``(frames_dir, scene, oid, pos, bbox)``; the item is the
    ``[3, H, W]`` float crop :func:`process_frame` produces (the mask plays no
    part in the backbone input, so none is loaded).
    """

    def __init__(self, entries: list[tuple], image_size: tuple[int, int]):
        self.entries = entries
        self.transform = build_transform(image_size)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        frames_dir, scene, oid, pos, bbox = self.entries[index]
        image = np.array(
            Image.open(Path(frames_dir) / f"{pos:06d}.jpg").convert("RGB"),
            np.uint8)
        frame = {"image": image, "img_wh": None, "mask": None, "bbox": bbox,
                 "key": f"{scene}#{oid}@{pos}"}
        return process_frame(frame, self.transform)["img"], index


def collect_entries(
    corpus_root: Path, scenes: list[str], embedding_root: Path,
) -> tuple[list[tuple], int, list[str]]:
    """Enumerate missing cache entries for ``scenes``.

    Returns ``(entries, num_existing, failed_scenes)``. Scene loading reuses
    :func:`load_scene` one scene at a time, so a scene with missing or broken
    features is skipped rather than fatal — the ``valid_mask`` / ``bbox`` the
    cache keys on are exactly the training loader's. Labels play no part here,
    so every scene is read on the automatic-label path (test scenes without a
    manual annotation are cached too).
    """
    entries: list[tuple] = []
    num_existing = 0
    failed: list[str] = []
    for scene in tqdm(scenes, desc="index scenes"):
        try:
            data = load_scene(corpus_root, scene, "train", 1)
        except Exception as exc:                      # noqa: BLE001
            failed.append(f"{scene}: {exc}")
            continue
        for person, oid in enumerate(data["object_ids"].tolist()):
            for pos in np.flatnonzero(data["valid_mask"][person]).tolist():
                if embedding_path(embedding_root, scene, oid, pos).is_file():
                    num_existing += 1
                    continue
                entries.append((str(data["frames_dir"]), scene, int(oid),
                                int(pos), data["bbox"][person, pos].copy()))
    return entries, num_existing, failed


def _save_atomic(path: Path, bits: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, bits)
    tmp.replace(path)


def compute_entries(
    wrapper: SAM3DBodyWrapper,
    entries: list[tuple],
    embedding_root: Path,
    *,
    batch_size: int = 48,
    num_workers: int = 16,
) -> float:
    """Run the frozen backbone over ``entries`` and write the cache files.

    The exact training input path: :func:`process_frame` crop ->
    ``data_preprocess`` -> backbone in ``backbone_dtype``; the raw bf16 output
    is stored as int16 bits. Returns the elapsed seconds.
    """
    model = wrapper.model
    image_size = wrapper.image_size
    loader = DataLoader(
        CropDataset(entries, image_size), batch_size=batch_size,
        num_workers=num_workers, pin_memory=True)
    writer = ThreadPoolExecutor(max_workers=8)
    pending: list[Future] = []
    start = time.time()
    with torch.inference_mode():
        for images, indices in tqdm(loader, desc="backbone"):
            x = model.data_preprocess(images.to("cuda", non_blocking=True))
            emb = model.backbone(x.type(model.backbone_dtype), extra_embed=None)
            if isinstance(emb, tuple):
                emb = emb[-1]
            bits = emb.contiguous().view(torch.int16).cpu().numpy()
            for row, index in zip(bits, indices.tolist()):
                _, scene, oid, pos, _ = entries[index]
                pending.append(writer.submit(
                    _save_atomic, embedding_path(embedding_root, scene, oid, pos),
                    row.copy()))
            if len(pending) > 1024:                 # surface write errors early
                for future in pending[:512]:
                    future.result()
                pending = pending[512:]
    for future in pending:
        future.result()
    writer.shutdown()
    return time.time() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml",
                        help="experiment yaml (only the model section is used)")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--split", choices=("train", "test", "all"), default="all",
                        help="corpus DB split(s) to cover (test = all 108 "
                             "scenes, not just the annotated ones)")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=35,
                        help="backbone batch ROWS. At a fixed row count the "
                             "embedding of a crop is bit-exact regardless of "
                             "its batch neighbours; a different row count "
                             "shifts bf16 values by ~1 ulp (the same family "
                             "of realizations live training spans when "
                             "frames_per_batch changes). 35 matches the "
                             "current configs' frames_per_batch")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after this many frames (0 = all; benchmarking)")
    args = parser.parse_args()

    corpus_root = Path(args.root)
    embedding_root = corpus_root / "features" / "embedding"
    splits = ("train", "test") if args.split == "all" else (args.split,)
    scenes = [s for split in splits for s in list_scenes(corpus_root, split)]
    scenes = scenes[args.shard_index::args.num_shards]
    print(f"shard {args.shard_index}/{args.num_shards}: {len(scenes)} scenes")

    entries, num_existing, failed = collect_entries(
        corpus_root, scenes, embedding_root)
    if args.max_frames:
        entries = entries[: args.max_frames]
    print(f"{len(entries)} frames to compute ({num_existing} already cached, "
          f"{len(failed)} scenes failed to index)")
    for line in failed:
        print(f"  SKIPPED {line}")
    if not entries:
        return

    cfg = load_config(args.config)
    wrapper = SAM3DBodyWrapper(
        cfg["model"]["checkpoint_path"], cfg["model"]["mhr_model_path"]).to("cuda")
    wrapper.eval()

    elapsed = compute_entries(
        wrapper, entries, embedding_root,
        batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"wrote {len(entries)} embeddings in {elapsed:.0f}s "
          f"({len(entries) / max(elapsed, 1e-9):.1f} frames/s)")


if __name__ == "__main__":
    main()
