"""Embedding augmentation: the two components, the anneal, and config gating.

The properties that matter are structural, not statistical: the corruption must be
per frame (so neighbouring frames stay a useful source), CutMix must paste real
features from a frame of ANOTHER CLIP into a rectangle, the same rectangle must
land on the mask (a second clean per-frame input to the same features), and both
components must vanish at ``scale == 0`` so the annealed tail trains on clean data.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from contact.config import load_config
from contact.embedding_augment import anneal_scale, augment_batch

REPO = Path(__file__).resolve().parents[1]

CFG = {
    "enabled": True,
    "gaussian_alpha": 0.1,
    "cutmix_prob": 0.5,
    "cutmix_area": [0.1, 0.4],
    "anneal_start_frac": 0.8,
}


def _cfg(**overrides) -> dict:
    return {**CFG, **overrides}


def _batch(rows: int, seq_len: int, channels: int = 2, grid: int = 16,
           mask_px: int = 64, dtype=torch.float32) -> dict:
    """A batch whose row ``i`` is filled with the constant ``i`` throughout.

    Constant frames make a pasted cell name its own source frame, so the tests can
    assert *where* content came from rather than merely that it changed.
    """
    ramp = torch.arange(rows, dtype=torch.float32)
    return {
        "seq_len": seq_len,
        "embedding": ramp.view(rows, 1, 1, 1).expand(
            rows, channels, grid, grid).contiguous().to(dtype),
        "mask": ramp.view(rows, 1, 1, 1, 1).expand(
            rows, 1, 1, mask_px, mask_px).contiguous(),
    }


def test_disabled_and_zero_scale_leave_the_batch_untouched():
    for cfg, scale in ((_cfg(enabled=False), 1.0), (_cfg(), 0.0)):
        batch = _batch(8, 4)
        before = {k: v.clone() for k, v in batch.items() if torch.is_tensor(v)}
        augment_batch(batch, cfg, scale)
        for key, value in before.items():
            assert torch.equal(batch[key], value), key


def test_gaussian_is_per_channel_and_per_frame():
    torch.manual_seed(0)
    # Channel 1 is 100x channel 0, so a per-channel sigma must scale with it.
    batch = {"seq_len": 8, "embedding": torch.randn(64, 2, 8, 8) * torch.tensor(
        [1.0, 100.0]).view(1, 2, 1, 1)}
    clean = batch["embedding"].clone()
    augment_batch(batch, _cfg(gaussian_alpha=0.1, cutmix_prob=0.0))
    delta = batch["embedding"] - clean

    per_channel = delta.std(dim=(0, 2, 3))
    assert per_channel[1] / per_channel[0] == pytest.approx(100.0, rel=0.1)
    assert per_channel[0] == pytest.approx(0.1, rel=0.15)
    # Drawn per frame, not broadcast over the batch — a shared draw would be a
    # silent no-op for the whole design (every frame corrupted identically).
    assert not torch.equal(delta[0], delta[1])


def test_cutmix_pastes_a_rectangle_from_another_clip():
    rows, seq_len, grid = 24, 6, 16
    batch = _batch(rows, seq_len, grid=grid)
    clean = batch["embedding"].clone()
    torch.manual_seed(0)
    augment_batch(batch, _cfg(
        gaussian_alpha=0.0, cutmix_prob=1.0, cutmix_area=[0.25, 0.25]))

    for row in range(rows):
        changed = batch["embedding"][row, 0] != clean[row, 0]
        assert changed.any(), "cutmix_prob=1.0 must corrupt every frame"
        # Exactly one source frame, and it belongs to a DIFFERENT clip.
        sources = batch["embedding"][row, 0][changed].unique()
        assert sources.numel() == 1
        assert int(sources) // seq_len != row // seq_len
        # The changed cells form one axis-aligned rectangle of ~the asked area.
        ys, xs = changed.nonzero(as_tuple=True)
        box = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        assert int(changed.sum()) == int(box)
        assert 0.15 <= box / (grid * grid) <= 0.35
        # Channels move together: the corruption is shared by the whole frame.
        assert torch.equal(changed, batch["embedding"][row, 1] != clean[row, 1])


def test_cutmix_box_lands_on_the_mask_at_its_own_resolution():
    """The mask is a second clean input to the same features; it must follow."""
    rows, seq_len, grid, mask_px = 12, 4, 16, 64
    batch = _batch(rows, seq_len, grid=grid, mask_px=mask_px)
    clean_emb, clean_mask = batch["embedding"].clone(), batch["mask"].clone()
    torch.manual_seed(0)
    augment_batch(batch, _cfg(
        gaussian_alpha=0.0, cutmix_prob=1.0, cutmix_area=[0.25, 0.25]))

    scale = mask_px // grid
    for row in range(rows):
        emb_box = (batch["embedding"][row, 0] != clean_emb[row, 0])
        mask_box = (batch["mask"][row, 0, 0] != clean_mask[row, 0, 0])
        # Same normalized region, so the mask box is the feature box upsampled.
        upsampled = emb_box.repeat_interleave(scale, 0).repeat_interleave(scale, 1)
        assert torch.equal(mask_box, upsampled)
        # ...and pasted from the same source frame.
        assert torch.equal(
            batch["mask"][row, 0, 0][mask_box].unique(),
            batch["embedding"][row, 0][emb_box].unique())


@pytest.mark.parametrize("rows,seq_len", [(240, 60), (24, 1), (25, 8), (6, 3)])
def test_cutmix_source_is_never_from_the_destination_clip(rows, seq_len):
    """Holds for T=1 and for a ragged final clip, neither of which is obvious."""
    from contact.embedding_augment import _cross_clip_source

    torch.manual_seed(0)
    destination = torch.arange(rows)
    for _ in range(50):
        source = _cross_clip_source(rows, seq_len, torch.device("cpu"))
        assert source.min() >= 0 and source.max() < rows
        assert not (source // seq_len == destination // seq_len).any()


def test_mask_is_untouched_when_only_the_gaussian_runs():
    """The always-on component corrupts appearance only — a documented contract."""
    batch = _batch(8, 4)
    clean_mask = batch["mask"].clone()
    torch.manual_seed(0)
    augment_batch(batch, _cfg(gaussian_alpha=0.1, cutmix_prob=0.0))
    assert torch.equal(batch["mask"], clean_mask)
    assert not torch.equal(batch["embedding"], _batch(8, 4)["embedding"])


def test_cutmix_probability_selects_a_subset_of_frames():
    batch = _batch(256, 8, channels=1, grid=8, mask_px=8)
    clean = batch["embedding"].clone()
    torch.manual_seed(0)
    augment_batch(batch, _cfg(gaussian_alpha=0.0, cutmix_prob=0.5))
    hit = (batch["embedding"] != clean).flatten(1).any(dim=1).float().mean()
    assert 0.35 <= float(hit) <= 0.65


def test_bfloat16_batches_keep_their_dtype():
    batch = _batch(16, 4, dtype=torch.bfloat16)
    torch.manual_seed(0)
    augment_batch(batch, _cfg())
    assert batch["embedding"].dtype == torch.bfloat16


def test_anneal_holds_then_reaches_zero():
    assert anneal_scale(0, 10, 0.8) == 1.0
    assert anneal_scale(7, 10, 0.8) == 1.0            # 80% done: still full
    assert 0.0 < anneal_scale(8, 10, 0.8) < 1.0
    assert anneal_scale(9, 10, 0.8) == pytest.approx(0.0, abs=1e-9)
    assert anneal_scale(9, 10, 1.0) == 1.0            # start_frac=1.0 disables it
    assert anneal_scale(0, 1, 0.8) == 0.0             # rejected by the validator


def test_config_rejects_the_silent_no_op_setups(tmp_path):
    def _load(body: str, tail: str = ""):
        path = tmp_path / "run.yaml"
        path.write_text(f"base: {REPO / 'configs' / 'base.yaml'}\n"
                        f"data:\n  embedding_cache: true\n{body}{tail}")
        return load_config(path)

    corpus = ("  datasets:\n    - name: climbing_corpus\n"
              "      config: configs/datasets/climbing_corpus.yaml\n")
    on = "  embedding_augment:\n    enabled: true\n"

    with pytest.raises(ValueError, match="requires data.embedding_cache"):
        load_config(_write_no_cache(tmp_path))
    with pytest.raises(ValueError, match="both components off"):
        _load(corpus + on + "    gaussian_alpha: 0.0\n    cutmix_prob: 0.0\n")
    with pytest.raises(ValueError, match="every dataset to be climbing_corpus"):
        _load("  datasets:\n    - name: damon\n"
              "      config: configs/datasets/damon.yaml\n" + on)
    with pytest.raises(ValueError, match="annealed out for the whole run"):
        _load(corpus + on, "\noptim:\n  epochs: 1\n")
    with pytest.raises(ValueError, match=">= 2 clips per batch"):
        _load(corpus + on + "  frames_per_batch: 8\n"
              "  sequence:\n    frames_per_clip: 8\n")

    cfg = _load(corpus + on, "\noptim:\n  epochs: 20\n")
    assert cfg["data"]["embedding_augment"]["cutmix_area"] == [0.1, 0.4]


def _write_no_cache(tmp_path: Path) -> Path:
    path = tmp_path / "no_cache.yaml"
    path.write_text(f"base: {REPO / 'configs' / 'base.yaml'}\n"
                    "data:\n  embedding_augment:\n    enabled: true\n")
    return path
