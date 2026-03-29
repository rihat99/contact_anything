"""
Tests for SMPL per-body-part contact labels in DamonDataset.

Exercises:
  - load_smpl_part_segmentation: correct shape, unique verts, no missing verts
  - part_contact_from_vertex_label: known contact pattern → expected parts hot
  - DamonDataset (topology='smpl', classic mode): label dict structure, shapes
  - DamonDataset part labels disabled when topology='mhr'
  - Backward compat: without smpl_part_seg_path, label is still a flat tensor
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Allow running from within dataset/ directly
sys.path.insert(0, str(Path(__file__).parent))

from damon_utils import (
    SMPL_NUM_VERTS,
    SMPL_PART_NAMES,
    load_smpl_part_segmentation,
    part_contact_from_vertex_label,
)

SMPL_SEG_PATH = (
    "/data3/rikhat.akizhanov/human_global_motion/better_human/"
    "src/better_human/smpl/config/smpl_3d_segmentation.npy"
)


# ---------------------------------------------------------------------------
# load_smpl_part_segmentation
# ---------------------------------------------------------------------------

class TestLoadSmplPartSegmentation:
    def test_part_count(self):
        names, arrays = load_smpl_part_segmentation(SMPL_SEG_PATH)
        assert len(names) == 24
        assert len(arrays) == 24

    def test_names_match_smpl_part_names(self):
        names, _ = load_smpl_part_segmentation(SMPL_SEG_PATH)
        assert names == SMPL_PART_NAMES, "Part names must match SMPL_PART_NAMES (joint order)"

    def test_vertex_coverage(self):
        """All 6890 SMPL vertices must appear in at least one part."""
        _, arrays = load_smpl_part_segmentation(SMPL_SEG_PATH)
        all_verts = np.concatenate(arrays)
        assert set(all_verts) == set(range(SMPL_NUM_VERTS)), (
            "Segmentation must cover all 6890 SMPL vertices"
        )

    def test_vertex_range(self):
        _, arrays = load_smpl_part_segmentation(SMPL_SEG_PATH)
        for arr in arrays:
            assert arr.min() >= 0
            assert arr.max() < SMPL_NUM_VERTS

    def test_returns_int32_arrays(self):
        _, arrays = load_smpl_part_segmentation(SMPL_SEG_PATH)
        for arr in arrays:
            assert arr.dtype == np.int32

    def test_no_sentinel_indices(self):
        """Indices >= SMPL_NUM_VERTS (sentinel values in the npy) must be filtered out."""
        _, arrays = load_smpl_part_segmentation(SMPL_SEG_PATH)
        for arr in arrays:
            assert arr.max() < SMPL_NUM_VERTS, "Sentinel indices >= 6890 must be filtered"


# ---------------------------------------------------------------------------
# part_contact_from_vertex_label
# ---------------------------------------------------------------------------

class TestPartContactFromVertexLabel:
    def setup_method(self):
        self.part_names, self.part_arrays = load_smpl_part_segmentation(SMPL_SEG_PATH)

    def test_all_zeros_no_contact(self):
        label = np.zeros(SMPL_NUM_VERTS, dtype=np.int64)
        out = part_contact_from_vertex_label(label, self.part_arrays)
        assert out.sum() == 0

    def test_all_ones_all_contact(self):
        label = np.ones(SMPL_NUM_VERTS, dtype=np.int64)
        out = part_contact_from_vertex_label(label, self.part_arrays)
        assert out.sum() == 24

    def test_single_part_contact(self):
        """Touching only 'head' vertices should activate only the head part."""
        head_idx = self.part_names.index('head')
        head_verts = self.part_arrays[head_idx]

        label = np.zeros(SMPL_NUM_VERTS, dtype=np.int64)
        label[head_verts[0]] = 1  # single vertex in head

        out = part_contact_from_vertex_label(label, self.part_arrays)
        assert out[head_idx] == 1, "Head part should be active"
        # All other parts that don't share this vertex must be inactive
        for i, arr in enumerate(self.part_arrays):
            if i != head_idx and head_verts[0] not in arr:
                assert out[i] == 0, f"Part '{self.part_names[i]}' should be inactive"

    def test_output_shape_and_dtype(self):
        label = np.zeros(SMPL_NUM_VERTS, dtype=np.int64)
        out = part_contact_from_vertex_label(label, self.part_arrays)
        assert out.shape == (24,)
        assert out.dtype == np.int64

    def test_output_binary(self):
        rng = np.random.RandomState(0)
        label = rng.randint(0, 2, size=SMPL_NUM_VERTS).astype(np.int64)
        out = part_contact_from_vertex_label(label, self.part_arrays)
        assert set(out.tolist()).issubset({0, 1})


# ---------------------------------------------------------------------------
# DamonDataset integration
# ---------------------------------------------------------------------------

def _make_minimal_smpl_npz(tmp_path, n=5):
    """Create a tiny SMPL-topology contact NPZ for dataset tests."""
    npz_path = str(tmp_path / "contact.npz")
    rng = np.random.RandomState(42)
    labels = rng.randint(0, 2, size=(n, SMPL_NUM_VERTS)).astype(np.int64)
    # Use fake image paths (dataset won't load images in this test)
    imgnames = np.array([f"fake/img_{i:04d}.jpg" for i in range(n)])
    np.savez(npz_path, imgname=imgnames, contact_label=labels)
    return npz_path, labels


class TestDamonDatasetPartLabels:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.tmp_path = Path(self.tmp)
        self.npz_path, self.raw_labels = _make_minimal_smpl_npz(self.tmp_path)
        self.part_names, self.part_arrays = load_smpl_part_segmentation(SMPL_SEG_PATH)

    def _make_dataset(self, smpl_part_seg_path=None):
        from damon_dataset import DamonDataset

        class _Stub(DamonDataset):
            """Override _getitem_classic to skip image IO."""
            def _getitem_classic(self, idx):
                import torch
                contact_label = self.contact_labels[idx]
                bbox = torch.zeros(4)
                cam_k = torch.eye(3)
                if self._part_vert_arrays is not None:
                    from damon_utils import part_contact_from_vertex_label
                    part_contact = part_contact_from_vertex_label(
                        contact_label, self._part_vert_arrays
                    )
                    return (None, bbox, cam_k), {
                        'contact_label': torch.from_numpy(contact_label).long(),
                        'part_contact': torch.from_numpy(part_contact).long(),
                    }
                return (None, bbox, cam_k), torch.from_numpy(contact_label).long()

        return _Stub(
            contact_npz_path=self.npz_path,
            topology='smpl',
            smpl_part_seg_path=smpl_part_seg_path,
        )

    def test_without_part_seg_returns_flat_tensor(self):
        ds = self._make_dataset(smpl_part_seg_path=None)
        _, label = ds[0]
        assert isinstance(label, torch.Tensor)
        assert label.shape == (SMPL_NUM_VERTS,)

    def test_with_part_seg_returns_dict(self):
        ds = self._make_dataset(smpl_part_seg_path=SMPL_SEG_PATH)
        _, label = ds[0]
        assert isinstance(label, dict)
        assert 'contact_label' in label
        assert 'part_contact' in label

    def test_contact_label_shape(self):
        ds = self._make_dataset(smpl_part_seg_path=SMPL_SEG_PATH)
        _, label = ds[0]
        assert label['contact_label'].shape == (SMPL_NUM_VERTS,)
        assert label['contact_label'].dtype == torch.int64

    def test_part_contact_shape(self):
        ds = self._make_dataset(smpl_part_seg_path=SMPL_SEG_PATH)
        _, label = ds[0]
        assert label['part_contact'].shape == (24,)
        assert label['part_contact'].dtype == torch.int64

    def test_part_contact_binary(self):
        ds = self._make_dataset(smpl_part_seg_path=SMPL_SEG_PATH)
        for i in range(len(ds)):
            _, label = ds[i]
            vals = label['part_contact'].unique().tolist()
            assert set(vals).issubset({0, 1})

    def test_part_contact_consistent_with_vertex_label(self):
        """part_contact[i]=1 iff any vertex in part i has contact."""
        ds = self._make_dataset(smpl_part_seg_path=SMPL_SEG_PATH)
        for sample_idx in range(len(ds)):
            _, label = ds[sample_idx]
            vert = label['contact_label'].numpy()
            part = label['part_contact'].numpy()
            expected = part_contact_from_vertex_label(vert, self.part_arrays)
            np.testing.assert_array_equal(part, expected,
                err_msg=f"Mismatch at sample {sample_idx}")

    def test_raises_on_mhr_with_part_seg(self):
        from damon_dataset import DamonDataset
        import pytest
        # Build a minimal MHR NPZ
        from damon_utils import LOD_VERTEX_COUNTS
        n_verts = LOD_VERTEX_COUNTS[1]
        mhr_npz = str(self.tmp_path / "mhr.npz")
        np.savez(mhr_npz,
                 imgname=np.array(["x.jpg"]),
                 contact_label=np.zeros((1, n_verts), dtype=np.int64))
        with pytest.raises(ValueError, match="topology='smpl'"):
            DamonDataset(mhr_npz, topology='mhr', smpl_part_seg_path=SMPL_SEG_PATH)


# ---------------------------------------------------------------------------
# Standalone smoke-test (no pytest)
# ---------------------------------------------------------------------------

def _smoke_test():
    print("=== Smoke test: load_smpl_part_segmentation ===")
    names, arrays = load_smpl_part_segmentation(SMPL_SEG_PATH)
    print(f"  Parts ({len(names)}): {names}")
    total_unique = len(set(np.concatenate(arrays).tolist()))
    print(f"  Unique vertices covered: {total_unique} / {SMPL_NUM_VERTS}")
    print(f"  Max index: {max(v for a in arrays for v in a)}")

    print("\n=== Smoke test: part_contact_from_vertex_label ===")
    rng = np.random.RandomState(7)
    label = rng.randint(0, 2, size=SMPL_NUM_VERTS).astype(np.int64)
    out = part_contact_from_vertex_label(label, arrays)
    active = [names[i] for i in range(len(names)) if out[i]]
    print(f"  Active parts ({out.sum()}): {active}")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    _smoke_test()
    # Also run pytest programmatically
    sys.exit(pytest.main([__file__, "-v"]))
