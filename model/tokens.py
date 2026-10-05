"""Learned decoder token blocks and their keypoint-anchored per-layer updates.

A :class:`LearnedTokenBlock` owns everything the frozen decoder does *not*:
the learnable token embeddings of one modality (contact / force) and the two
projections of the per-layer update — a 2D positional-encoding FFN and a
backbone-feature linear. Per forward it produces an
:class:`~model.sam_3d_body.models.meta_arch.sam3d_body.ExtraTokenBlock`
carrying the batch-expanded tokens and a bound callback; the fork appends the
tokens behind its asymmetric mask and invokes the callback after every
intermediate decoder layer.

The anchored update (a port of the fork's former ``_anchored_token_update``):
each token is tied to one point of the body. After every intermediate layer, the
layer's interm pose prediction gives that point's 2D crop position; the
update (1) writes ``posemb_linear`` of that position into the token's augment
row and (2) adds ``feat_linear`` of the (ray-conditioned) image features
grid-sampled there to the token itself. Anchors outside the crop or behind the
camera contribute zero.

Two kinds of anchor, one per contact set (:mod:`model.contact_frames`):

* **keypoint** (``kindyn6``) — one MHR70 index per token, read straight off the
  interm ``pred_keypoints_2d_cropped`` / ``pred_keypoints_2d_depth``.
* **vertex** (``frames35``) — one SMPL-X vertex per token, which is a barycentric
  point of ONE MHR triangle (``MHR_MAPPING_NPZ`` + the MHR faces). The interm
  readout already carries every MHR vertex's full-image projection
  (``pred_keypoints_2d_verts``) and camera depth (``pred_vertices`` +
  ``pred_cam_t``), so the anchor is one gather of the token's three MHR vertices
  and their barycentric combination, converted to crop coordinates with the
  batch's affine.

The square-backbone assumption: grid-sample x coordinates are used as-is,
which is exact for the DINOv3 backbones (square input).
"""
from __future__ import annotations

from functools import partial
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.contact_frames import MHR_MAPPING_NPZ
from model.sam_3d_body.models.meta_arch.sam3d_body import ExtraTokenBlock
from model.sam_3d_body.models.modules.transformer import FFN


def vertex_anchors(vertex_ids: Sequence[int], faces: torch.Tensor) -> tuple[torch.Tensor,
                                                                           torch.Tensor]:
    """The MHR triangle of each SMPL-X vertex: ``(K, 3)`` MHR vertex ids + ``(K, 3)`` weights.

    ``MHR_MAPPING_NPZ`` gives one MHR face id and three barycentric coordinates per
    SMPL-X vertex; ``faces`` ``(F, 3)`` is the MHR mesh's own face table (read off the
    frozen MHR body model), so vertex ``v`` of SMPL-X is
    ``sum_k w[v, k] * mhr_vertices[ids[v, k]]``.
    """
    mapping = np.load(MHR_MAPPING_NPZ)
    rows = np.asarray(list(vertex_ids), np.int64)
    triangles = torch.as_tensor(mapping["triangle_ids"][rows], dtype=torch.long)
    weights = torch.as_tensor(mapping["baryc_coords"][rows], dtype=torch.float32)
    return faces.to(torch.long)[triangles], weights


class LearnedTokenBlock(nn.Module):
    """One modality's learned decoder tokens + anchored-update projections.

    :param name: block name (keys the wrapper's ``blocks`` bounds dict).
    :param dim: decoder token width.
    :param backbone_dim: backbone feature channels (feat-linear input).
    :param keypoint_indices: MHR70 anchor index per token (the keypoint anchors);
        the list length is the token count. Mutually exclusive with ``anchors``.
    :param anchors: ``(ids (K, 3) long, weights (K, 3))`` vertex anchors from
        :func:`vertex_anchors`; the token count is ``K``.
    :param grid_size: K — sample a K x K grid around each anchor and average
        (``1`` = single-point bilinear sample).
    :param grid_radius: grid spacing in normalized [-1, 1] sample coordinates.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        backbone_dim: int,
        keypoint_indices: Optional[Sequence[int]] = None,
        anchors: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        grid_size: int = 1,
        grid_radius: float = 0.1,
    ):
        super().__init__()
        if (keypoint_indices is None) == (anchors is None):
            raise ValueError(f"{name}: pass exactly one of keypoint_indices= or anchors=")
        self.name = str(name)
        self.keypoint_indices = None
        if keypoint_indices is not None:
            keypoint_indices = [int(i) for i in keypoint_indices]
            assert keypoint_indices and all(0 <= i < 70 for i in keypoint_indices), (
                f"{name}: anchor indices must be MHR70 indices in [0, 70); "
                f"got {keypoint_indices}")
            self.keypoint_indices = keypoint_indices
            self.num_tokens = len(keypoint_indices)
        else:
            ids, weights = anchors
            if ids.shape != weights.shape or ids.dim() != 2 or ids.shape[1] != 3:
                raise ValueError(f"{name}: vertex anchors must be (K, 3) ids + (K, 3) weights; "
                                 f"got {tuple(ids.shape)} / {tuple(weights.shape)}")
            self.num_tokens = int(ids.shape[0])
            self.register_buffer("anchor_vertex_ids", ids.to(torch.long), persistent=False)
            self.register_buffer("anchor_weights", weights.to(torch.float32), persistent=False)
        self.grid_size = int(grid_size)
        self.grid_radius = float(grid_radius)

        self.embedding = nn.Embedding(self.num_tokens, dim)
        # Positional encoding: 2D crop position -> decoder dim
        self.posemb_linear = FFN(
            embed_dims=2,
            feedforward_channels=dim,
            output_dims=dim,
            num_fcs=2,
            add_identity=False,
        )
        # Feature projection: sampled backbone features -> decoder dim
        self.feat_linear = nn.Linear(backbone_dim, dim)

    @property
    def vertex_anchored(self) -> bool:
        """Whether the tokens anchor at mesh vertices (else at MHR70 keypoints)."""
        return self.keypoint_indices is None

    def as_extra_block(self, batch_size: int, affine_trans: Optional[torch.Tensor] = None,
                       img_size: Optional[torch.Tensor] = None) -> ExtraTokenBlock:
        """The fork-facing block for one forward pass.

        :param affine_trans: ``[B, 2, 3]`` full-image -> crop affine and ``img_size``
            ``[B, 2]`` the crop size — required by the vertex anchors, whose projections
            come out of the interm readout in full-image pixels.
        """
        if self.vertex_anchored and (affine_trans is None or img_size is None):
            raise ValueError(f"{self.name}: vertex anchors need affine_trans= and img_size= "
                             "to reach crop coordinates")
        return ExtraTokenBlock(
            name=self.name,
            tokens=self.embedding.weight[None].expand(batch_size, -1, -1),
            update_fn=partial(self._update, affine_trans=affine_trans, img_size=img_size),
        )

    def _keypoint_anchors(self, pose_output: Dict) -> tuple[torch.Tensor, torch.Tensor]:
        """``([B, K, 2]`` crop positions, ``[B, K]`` depths) of the MHR70 anchors."""
        return (pose_output["pred_keypoints_2d_cropped"][:, self.keypoint_indices],
                pose_output["pred_keypoints_2d_depth"][:, self.keypoint_indices])

    def _vertex_anchors(self, pose_output: Dict, affine_trans: torch.Tensor,
                        img_size: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``([B, K, 2]`` crop positions, ``[B, K]`` depths) of the vertex anchors.

        One gather of the 3K MHR vertices per layer: their full-image projections and
        camera depths, barycentrically combined, then the fork's own full -> crop map
        (``[u, v, 1] @ affine.mT / img_size - 0.5``).
        """
        ids = self.anchor_vertex_ids.reshape(-1)                              # [3K]
        weights = self.anchor_weights[None, :, :, None]                       # [1, K, 3, 1]
        full = pose_output["pred_keypoints_2d_verts"][:, ids].reshape(
            -1, self.num_tokens, 3, 2).to(weights)                            # [B, K, 3, 2] px
        depth = (pose_output["pred_vertices"][..., 2]
                 + pose_output["pred_cam_t"][:, None, 2])[:, ids].reshape(
                     -1, self.num_tokens, 3, 1).to(weights)                   # [B, K, 3, 1] m
        point = (full * weights).sum(dim=2)                                   # [B, K, 2] px
        homogeneous = torch.cat([point, torch.ones_like(point[..., :1])], dim=-1)
        crop = homogeneous @ affine_trans.to(point).mT / img_size.to(point)[:, None] - 0.5
        return crop, (depth * weights).sum(dim=2).squeeze(-1)

    def _update(
        self,
        start_idx: int,
        image_embeddings: torch.Tensor,
        token_embeddings: torch.Tensor,
        token_augment: torch.Tensor,
        pose_output: Dict,
        *,
        affine_trans: Optional[torch.Tensor] = None,
        img_size: Optional[torch.Tensor] = None,
    ):
        """Anchored per-layer update of this block's token rows.

        Contract fixed by the fork's extra-token-block hook:
        ``(start_idx, image_embeddings, token_embeddings, token_augment,
        pose_output) -> (token_embeddings, token_augment)``; the crop geometry is
        bound by :meth:`as_extra_block`.
        """
        # Anchor positions in crop space (-0.5 to 0.5) and their camera depths.
        if self.vertex_anchored:
            anchor_kps_2d, anchor_kps_depth = self._vertex_anchors(
                pose_output, affine_trans, img_size)
        else:
            anchor_kps_2d, anchor_kps_depth = self._keypoint_anchors(pose_output)

        # Validity: outside image bounds or behind camera
        anchor_kps_01 = anchor_kps_2d + 0.5
        invalid_mask = (
            (anchor_kps_01[:, :, 0] < 0)
            | (anchor_kps_01[:, :, 0] > 1)
            | (anchor_kps_01[:, :, 1] < 0)
            | (anchor_kps_01[:, :, 1] > 1)
            | (anchor_kps_depth < 1e-5)
        )  # [B, K]

        # 1. Positional encoding into the augment rows
        token_augment = token_augment.clone()
        token_augment[:, start_idx : start_idx + self.num_tokens, :] = (
            self.posemb_linear(anchor_kps_2d) * (~invalid_mask[:, :, None])
        )

        # 2. Grid-sampled image features added to the tokens
        sample_points = anchor_kps_2d * 2  # [-0.5, 0.5] -> [-1, 1]
        gs = self.grid_size
        if gs > 1:
            half = gs // 2
            offsets = torch.tensor(
                [
                    [dy * self.grid_radius, dx * self.grid_radius]
                    for dy in range(-half, half + 1)
                    for dx in range(-half, half + 1)
                ],
                dtype=sample_points.dtype,
                device=sample_points.device,
            )  # [K*K, 2]
            pts = sample_points.unsqueeze(2) + offsets[None, None]  # [B, K, gs*gs, 2]
            b, k, kk, _ = pts.shape
            feats_flat = (
                F.grid_sample(
                    image_embeddings,
                    pts.reshape(b, k * kk, 1, 2),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                .squeeze(3)
                .permute(0, 2, 1)
            )  # [B, K*gs*gs, C_backbone]
            sampled_feats = feats_flat.reshape(b, k, kk, -1).mean(dim=2)
        else:
            sampled_feats = (
                F.grid_sample(
                    image_embeddings,
                    sample_points[:, :, None, :],
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                .squeeze(3)
                .permute(0, 2, 1)
            )  # [B, K, C_backbone]

        sampled_feats = sampled_feats * (~invalid_mask[:, :, None])

        token_embeddings = token_embeddings.clone()
        token_embeddings[:, start_idx : start_idx + self.num_tokens, :] += (
            self.feat_linear(sampled_feats)
        )

        return token_embeddings, token_augment
