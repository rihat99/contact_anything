"""Cached-embedding equivalence: precomputed backbone vs live, outputs and grads.

The cache stores the RAW bf16 backbone output (``scripts/precompute_embeddings``);
mask and ray conditioning run live in both paths, so the only degree of freedom
is the backbone computation itself. Two measured facts (2026-08-28, RTX 6000
Ada) shape the assertions:

* At a FIXED backbone batch shape, a crop's bf16 embedding is bit-exact
  regardless of its batch neighbours (0/31M differing elements across two
  shifted 48-row batches). A cache built at batch size B therefore reproduces
  live training at ``frames_per_batch == B`` bit-exactly.
* A DIFFERENT row count is a different bf16 rounding realization: ~80 % of
  elements move by ~1 ulp, which the decoder amplifies to ~2e-3 absolute on
  MHR/pose outputs (a 2pi axis-angle wrap on ``global_rot``) and a few percent
  on contact grads — the same noise family live training itself spans when
  ``frames_per_batch`` changes. Not a cache artifact.

The equivalence tests therefore compare at MATCHED shape, where the remaining
divergence is the CUDA run-to-run floor (the frozen forward is not
bit-deterministic; see ``test_temporal_invariance``): two live passes measure
the floor, cached must stay within a small margin of it.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from contact.config import load_config
from contact.data.climbing_corpus import (
    ClimbingCorpusDataset, DEFAULT_ROOT, embedding_path, list_corpus_scenes)
from contact.data.collate import batch_to_device, make_collate
from contact.engine import forward_model
from contact.model import build_model
from contact.targets import TargetSpec
from scripts.precompute_embeddings import (
    _save_atomic, collect_entries, compute_entries)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BASE_CFG = os.path.join(REPO, "configs", "base.yaml")
_CKPT = load_config(_BASE_CFG)["model"]["checkpoint_path"]

_NOISE_MARGIN = 8.0
_NOISE_FLOOR_EPS = 1e-6
# base.yaml: T=8 clips; 6 clips -> 48 backbone rows, matching the cache build.
_NUM_CLIPS = 6
_BATCH_ROWS = 48

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint missing"),
    pytest.mark.skipif(not os.path.isdir(DEFAULT_ROOT), reason="corpus missing"),
]


def test_save_roundtrip_preserves_bits(tmp_path):
    """int16 .npy storage is a bit-exact round trip for bf16 tensors."""
    torch.manual_seed(0)
    emb = torch.randn(2, 4, 3, dtype=torch.bfloat16)
    path = tmp_path / "aa/bb/scene/00/000000.npy"
    _save_atomic(path, emb.view(torch.int16).numpy())
    back = torch.from_numpy(np.load(path)).view(torch.bfloat16)
    assert torch.equal(back, emb)


def test_embedding_path_layout(tmp_path):
    path = embedding_path(tmp_path, "abcdef_0001", 3, 42)
    assert path == tmp_path / "ab/cd/abcdef_0001/03/000042.npy"


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    cfg = load_config(_BASE_CFG)
    model, _ = build_model(cfg, "cuda")
    model.eval()

    scene = list_corpus_scenes(DEFAULT_ROOT, "train")[0]
    cache_root = tmp_path_factory.mktemp("embedding")
    entries, _, failed = collect_entries(DEFAULT_ROOT, [scene], cache_root)
    assert not failed, failed
    assert entries, "first train scene produced no cache entries"
    compute_entries(model, entries, cache_root,
                    batch_size=_BATCH_ROWS, num_workers=8)

    image_size = tuple(model.cfg.MODEL.IMAGE_SIZE)
    collate = make_collate(image_size, TargetSpec.from_config(cfg))

    def _batch(embedding_dir):
        ds = ClimbingCorpusDataset(
            DEFAULT_ROOT, scenes=[scene], split="train", jitter=False,
            embedding_dir=embedding_dir)
        # Early-scene clips only: the shard's LAST script batch has fewer rows
        # (a different shape realization); staying below it keeps the matched
        # -shape guarantee exact.
        assert len(ds) >= _NUM_CLIPS
        return batch_to_device(collate([ds[i] for i in range(_NUM_CLIPS)]), "cuda")

    return model, _batch(None), _batch(cache_root)


def test_cache_matches_live_backbone_bits(setup):
    """At matched batch shape the cache equals the live backbone bitwise.

    The script computed these rows among ~600 scene entries in 48-row batches
    with different neighbours — bit equality here is exactly the row-content
    independence the cached training path relies on.
    """
    model, batch_live, batch_cached = setup
    images = batch_live["img"][:, 0]
    assert images.shape[0] == _BATCH_ROWS
    with torch.inference_mode():
        x = model.data_preprocess(images)
        emb = model.backbone(x.type(model.backbone_dtype), extra_embed=None)
    if isinstance(emb, tuple):
        emb = emb[-1]
    cached = batch_cached["embedding"]
    assert cached.dtype == torch.bfloat16
    assert torch.equal(emb.contiguous().view(torch.int16).cpu(),
                       cached.view(torch.int16).cpu())


def _flat_outputs(out: dict) -> dict[str, torch.Tensor]:
    flat = {}
    for group in ("mhr", "contact", "force", "motion"):
        sub = out.get(group)
        if sub is None:
            continue
        for key, value in sub.items():
            if torch.is_tensor(value) and value.is_floating_point():
                flat[f"{group}.{key}"] = value.detach()
    flat["image_embeddings"] = out["image_embeddings"].detach()
    return flat


def _max_diffs(a: dict, b: dict) -> dict[str, float]:
    assert a.keys() == b.keys()
    return {k: (a[k] - b[k]).abs().max().item() for k in a}


def _forward_and_grads(model, batch):
    model.zero_grad(set_to_none=True)
    out = forward_model(model, batch)
    loss = sum(v.sum() for k, v in out["contact"].items() if k.endswith("_logits"))
    loss.backward()
    grads = {
        name: p.grad.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }
    assert grads, "no trainable parameter received a gradient"
    return _flat_outputs(out), grads


def test_cached_matches_live_outputs_and_grads(setup):
    model, batch_live, batch_cached = setup
    assert "embedding" not in batch_live

    out_a, grads_a = _forward_and_grads(model, batch_live)
    out_b, grads_b = _forward_and_grads(model, batch_live)
    out_c, grads_c = _forward_and_grads(model, batch_cached)

    floor_out = _max_diffs(out_a, out_b)
    diff_out = _max_diffs(out_a, out_c)
    for key, diff in diff_out.items():
        bound = max(_NOISE_MARGIN * floor_out[key], _NOISE_FLOOR_EPS)
        assert diff <= bound, (
            f"output {key}: cached-vs-live diff {diff:.3e} exceeds "
            f"noise-floor bound {bound:.3e} (floor {floor_out[key]:.3e})")

    floor_grad = _max_diffs(grads_a, grads_b)
    diff_grad = _max_diffs(grads_a, grads_c)
    for name, diff in diff_grad.items():
        bound = max(_NOISE_MARGIN * floor_grad[name], _NOISE_FLOOR_EPS)
        assert diff <= bound, (
            f"grad {name}: cached-vs-live diff {diff:.3e} exceeds "
            f"noise-floor bound {bound:.3e} (floor {floor_grad[name]:.3e})")


def test_missing_cache_file_is_a_hard_error(setup, tmp_path):
    scene = list_corpus_scenes(DEFAULT_ROOT, "train")[0]
    ds = ClimbingCorpusDataset(
        DEFAULT_ROOT, scenes=[scene], split="train", jitter=False,
        embedding_dir=tmp_path / "empty")
    with pytest.raises(FileNotFoundError):
        ds[0]
