"""Unit tests of :class:`~model.tokens.LearnedTokenBlock`'s anchors (CPU, no base model).

The block is driven with a FAKE interm ``pose_output`` — the fork's contract — so the
anchor arithmetic is checked without the frozen decoder:

* ``vertex_anchors`` picks each SMPL-X vertex's MHR triangle out of the mapping and
  resolves it through the MHR face table;
* a vertex-anchored token lands at the barycentric combination of its three MHR
  vertices' projections, converted to crop coordinates exactly like the fork's
  ``_full_to_crop``;
* anchors outside the crop or behind the camera contribute nothing (zero posemb row,
  zero feature update), and the ones inside do;
* the keypoint-anchored path (``kindyn6``) is untouched: it reads
  ``pred_keypoints_2d_cropped`` / ``_depth`` straight.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from model.contact_frames import KINDYN_GROUP_KEYPOINTS, MHR_MAPPING_NPZ, contact_set
from model.tokens import LearnedTokenBlock, vertex_anchors

DIM, BACKBONE_DIM = 32, 16
N_MHR_VERTICES, N_MHR_FACES = 18439, 36874
GRID = 8


def fake_faces() -> torch.Tensor:
    """An MHR-shaped face table whose rows are distinct vertex triples."""
    torch.manual_seed(0)
    return torch.randint(0, N_MHR_VERTICES, (N_MHR_FACES, 3))


def fake_pose_output(batch: int, crop: torch.Tensor | None = None) -> dict:
    """A fake interm readout: random vertex projections / depths and MHR70 keypoints."""
    torch.manual_seed(1)
    out = {
        "pred_keypoints_2d_verts": 400.0 * torch.rand(batch, N_MHR_VERTICES, 2) + 100.0,
        "pred_vertices": torch.randn(batch, N_MHR_VERTICES, 3) * 0.3,
        "pred_cam_t": torch.tensor([0.0, 0.0, 4.0]).expand(batch, 3).clone(),
        "pred_keypoints_2d_cropped": (torch.rand(batch, 70, 2) - 0.5) * 0.8,
        "pred_keypoints_2d_depth": torch.full((batch, 70), 3.0),
    }
    if crop is not None:
        out["pred_keypoints_2d_cropped"] = crop
    return out


def crop_geometry(batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    affine = torch.tensor([[0.5, 0.0, -20.0], [0.0, 0.5, -30.0]]).expand(batch, 2, 3).contiguous()
    return affine, torch.full((batch, 2), 256.0)


def test_vertex_anchors_resolve_the_mapping_through_the_faces():
    slots = contact_set("frames35")
    faces = fake_faces()
    ids, weights = vertex_anchors(slots.vertex_ids, faces)
    assert ids.shape == (35, 3) and weights.shape == (35, 3)
    mapping = np.load(MHR_MAPPING_NPZ)
    for slot, vertex in enumerate(slots.vertex_ids):
        triangle = int(mapping["triangle_ids"][vertex])
        assert torch.equal(ids[slot], faces[triangle].to(torch.long))
        assert np.allclose(weights[slot].numpy(), mapping["baryc_coords"][vertex], atol=1e-6)
    # Barycentric coordinates: non-negative and summing to one.
    assert float(weights.sum(dim=-1).sub(1.0).abs().max()) < 1e-5
    assert float(weights.min()) >= -1e-6


def make_block(anchors) -> LearnedTokenBlock:
    torch.manual_seed(2)
    return LearnedTokenBlock("contact", DIM, BACKBONE_DIM, anchors=anchors).eval()


def test_vertex_anchor_is_the_barycentric_point_in_crop_coordinates():
    slots = contact_set("frames35")
    ids, weights = vertex_anchors(slots.vertex_ids, fake_faces())
    block = make_block((ids, weights))
    batch = 3
    pose_output = fake_pose_output(batch)
    affine, img_size = crop_geometry(batch)
    crop, depth = block._vertex_anchors(pose_output, affine, img_size)
    assert crop.shape == (batch, 35, 2) and depth.shape == (batch, 35)

    full = pose_output["pred_keypoints_2d_verts"]
    z = pose_output["pred_vertices"][..., 2] + pose_output["pred_cam_t"][:, None, 2]
    for slot in range(35):
        triple, w = ids[slot], weights[slot]
        point = (full[:, triple] * w[None, :, None]).sum(dim=1)              # [B, 2] px
        homogeneous = torch.cat([point, torch.ones(batch, 1)], dim=-1)
        expected = torch.einsum("bj,bij->bi", homogeneous, affine) / img_size - 0.5
        assert torch.allclose(crop[:, slot], expected, atol=1e-5), slot
        assert torch.allclose(depth[:, slot], (z[:, triple] * w[None]).sum(dim=1), atol=1e-5)


def test_vertex_anchors_outside_the_crop_or_behind_the_camera_contribute_nothing():
    slots = contact_set("frames35")
    ids, weights = vertex_anchors(slots.vertex_ids, fake_faces())
    block = make_block((ids, weights))
    batch = 2
    pose_output = fake_pose_output(batch)
    affine, img_size = crop_geometry(batch)
    # Slot 0's triangle projects far outside the crop, slot 1's sits behind the camera.
    pose_output["pred_keypoints_2d_verts"][:, ids[0]] = 100000.0
    pose_output["pred_vertices"][:, ids[1], 2] = -10.0
    crop, depth = block._vertex_anchors(pose_output, affine, img_size)
    assert (crop[:, 0].abs() > 0.5).all() and (depth[:, 1] < 0).all()

    start = 1
    n_tokens = block.num_tokens
    tokens = torch.zeros(batch, start + n_tokens, DIM)
    augment = torch.randn(batch, start + n_tokens, DIM)
    embeddings = torch.randn(batch, BACKBONE_DIM, GRID, GRID)
    updated, aug = block._update(start, embeddings, tokens, augment, pose_output,
                                 affine_trans=affine, img_size=img_size)
    assert torch.count_nonzero(aug[:, start]) == 0            # outside the crop
    assert torch.count_nonzero(aug[:, start + 1]) == 0        # behind the camera
    # An invalid anchor samples zero features, so the token gains only the projection's bias.
    bias = block.feat_linear.bias
    assert torch.allclose(updated[:, start], bias.expand(batch, DIM), atol=1e-6)
    assert torch.allclose(updated[:, start + 1], bias.expand(batch, DIM), atol=1e-6)
    inside = ((crop.abs() <= 0.5).all(dim=-1) & (depth >= 1e-5))
    assert inside.any()
    rows = torch.nonzero(inside[0], as_tuple=False)[:, 0]
    assert torch.count_nonzero(aug[0, start + rows]) > 0
    assert (updated[0, start + rows] - bias).abs().max() > 1e-6
    assert torch.equal(aug[:, :start], augment[:, :start])    # rows before the block untouched


def test_keypoint_anchors_read_the_interm_crop_keypoints():
    torch.manual_seed(3)
    block = LearnedTokenBlock("contact", DIM, BACKBONE_DIM,
                              keypoint_indices=KINDYN_GROUP_KEYPOINTS).eval()
    assert not block.vertex_anchored and block.num_tokens == 6
    batch = 2
    pose_output = fake_pose_output(batch)
    crop, depth = block._keypoint_anchors(pose_output)
    assert torch.equal(crop, pose_output["pred_keypoints_2d_cropped"][:, list(KINDYN_GROUP_KEYPOINTS)])
    assert torch.equal(depth, pose_output["pred_keypoints_2d_depth"][:, list(KINDYN_GROUP_KEYPOINTS)])
    block.as_extra_block(batch)                               # needs no crop geometry


def test_the_two_anchor_kinds_are_exclusive():
    with pytest.raises(ValueError):
        LearnedTokenBlock("contact", DIM, BACKBONE_DIM)
    ids, weights = vertex_anchors(contact_set("frames35").vertex_ids, fake_faces())
    with pytest.raises(ValueError):
        LearnedTokenBlock("contact", DIM, BACKBONE_DIM,
                          keypoint_indices=KINDYN_GROUP_KEYPOINTS, anchors=(ids, weights))
    with pytest.raises(ValueError):
        make_block((ids, weights)).as_extra_block(2)          # vertex anchors need the crop
