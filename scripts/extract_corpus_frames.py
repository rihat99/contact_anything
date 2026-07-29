"""Pre-extract ClimbingVideos corpus scene videos into a ``frames/`` tree.

Training reads individual frames; decoding mp4s on the fly is too slow. This
writes ``<corpus>/frames/<s[0:2]>/<s[2:4]>/<scene>/<pos:06d>.jpg`` (JPEG
quality 95, sequential-decode order — the exact convention of
``BetterVideoReconstruction/scripts/export_contact_dataset.py``, so row k of
every feature array is frame ``k``).

Scenes come from ``scenes/scenes.db`` under the curated-corpus filter
(``human_selected=1 AND vlm_category IN (1,2) AND vlm_rope_supported=0``,
331 train + 30 test). Each scene is additionally guarded against
``rope_supported`` flips via its ``features/vlm/<shard>/<scene>.json``.
Complete extractions are skipped, so the script is safe to re-run.

    python scripts/extract_corpus_frames.py --workers 16
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from contact.data.reconstruction_scenes import extract_frames  # noqa: E402

DEFAULT_CORPUS = Path("/data3/rikhat.akizhanov/better/data/ClimbingVideos")
SCENE_QUERY = (
    "SELECT scene_id FROM scenes WHERE human_selected=1 "
    "AND vlm_category IN (1,2) AND vlm_rope_supported=0 ORDER BY scene_id"
)


def shard(scene: str) -> str:
    """Two-level shard prefix used throughout the corpus tree."""
    return f"{scene[0:2]}/{scene[2:4]}"


def list_scenes(corpus: Path) -> list[str]:
    """Curated boulder scene ids from ``scenes/scenes.db``."""
    with sqlite3.connect(corpus / "scenes" / "scenes.db") as db:
        return [row[0] for row in db.execute(SCENE_QUERY)]


def extract_one(args: tuple[str, str]) -> tuple[str, str]:
    """Extract one scene; returns ``(scene, "ok"|"skipped"|error text)``."""
    corpus_str, scene = args
    corpus = Path(corpus_str)
    try:
        vlm = json.loads(
            (corpus / "features" / "vlm" / shard(scene) / f"{scene}.json").read_text())
        if vlm["decision"]["rope_supported"]:
            raise ValueError("vlm says rope_supported — refusing to extract")
        contacts = np.load(
            corpus / "features" / "human_optim" / shard(scene) / scene / "contacts_1.npz")
        n_expected = int(contacts["num_frames"])
        out_dir = corpus / "frames" / shard(scene) / scene
        if len(list(out_dir.glob("*.jpg"))) == n_expected:
            return scene, "skipped"
        extract_frames(corpus / "scenes" / shard(scene) / f"{scene}.mp4",
                       out_dir, n_expected)
        return scene, "ok"
    except Exception as exc:  # keep the batch alive; report at the end
        return scene, f"FAILED: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    scenes = list_scenes(args.corpus)
    print(f"{len(scenes)} scenes")
    failed = []
    with Pool(args.workers) as pool:
        jobs = [(str(args.corpus), scene) for scene in scenes]
        for i, (scene, status) in enumerate(pool.imap_unordered(extract_one, jobs), 1):
            print(f"[{i}/{len(scenes)}] {scene}: {status}", flush=True)
            if status.startswith("FAILED"):
                failed.append((scene, status))
    if failed:
        print(f"\n{len(failed)} scenes failed:")
        for scene, status in failed:
            print(f"  {scene}: {status}")
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
