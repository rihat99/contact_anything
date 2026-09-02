"""Learned decoder token blocks and their keypoint-anchored per-layer updates.

A :class:`LearnedTokenBlock` owns everything the frozen decoder does *not*:
the learnable token embeddings of one modality (contact / force / motion) and,
for anchored blocks, the two projections of the per-layer update — a 2D
positional-encoding FFN and a backbone-feature linear. Per forward it produces
an :class:`~model.sam_3d_body.models.meta_arch.sam3d_body.ExtraTokenBlock`
carrying the batch-expanded tokens and a bound callback; the fork appends the
tokens behind its asymmetric mask and invokes the callback after every
intermediate decoder layer.

The anchored update (a port of the fork's former ``_anchored_token_update``):
each of the first ``num_anchored`` tokens is tied to one MHR70 keypoint. After
every intermediate layer, the layer's interm pose prediction gives the
keypoint's 2D crop position; the update (1) writes ``posemb_linear`` of that
position into the token's augment row and (2) adds ``feat_linear`` of the
(ray-conditioned) image features grid-sampled there to the token itself.
Anchors outside the crop or behind the camera contribute zero. Trailing
*global* tokens (contact only) and unanchored blocks never receive the update.

The square-backbone assumption: grid-sample x coordinates are used as-is,
which is exact for the DINOv3 backbones (square input). The retired ViT-HMR
variants (width-cropped input) would need the historical ``x * 16 / 12``
correction — deliberately not carried.
"""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.sam_3d_body.models.meta_arch.sam3d_body import ExtraTokenBlock
from model.sam_3d_body.models.modules.transformer import FFN


class LearnedTokenBlock(nn.Module):
    """One modality's learned decoder tokens + anchored-update projections.

    :param name: block name (keys the wrapper's ``blocks`` bounds dict).
    :param dim: decoder token width.
    :param backbone_dim: backbone feature channels (feat-linear input).
    :param keypoint_indices: MHR70 anchor index per anchored token; the list
        length is the anchored-token count.
    :param num_global_tokens: extra learned tokens appended after the anchored
        ones; they join the block (same mask rows) but are never anchored.
    :param anchored: ``False`` = pure learned queries — no per-layer update,
        no projections built; ``keypoint_indices`` then only names/counts slots.
    :param blind_to_image: gate the block's image cross-attention to zero AND
        drop the anchored update — the tokens' only input is self-attention
        over the visible part of the sequence. Projections are not built (they
        would be params that never receive a gradient, which DDP rejects).
    :param grid_size: K — sample a K x K grid around each anchor and average
        (``1`` = single-point bilinear sample).
    :param grid_radius: grid spacing in normalized [-1, 1] sample coordinates.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        backbone_dim: int,
        keypoint_indices: Sequence[int],
        num_global_tokens: int = 0,
        anchored: bool = True,
        blind_to_image: bool = False,
        grid_size: int = 1,
        grid_radius: float = 0.1,
    ):
        super().__init__()
        keypoint_indices = [int(i) for i in keypoint_indices]
        assert all(0 <= i < 70 for i in keypoint_indices), (
            f"{name}: anchor indices must be MHR70 indices in [0, 70); "
            f"got {keypoint_indices}")
        self.name = str(name)
        self.keypoint_indices = keypoint_indices
        self.num_anchored = len(keypoint_indices)
        self.num_global_tokens = int(num_global_tokens)
        self.num_tokens = self.num_anchored + self.num_global_tokens
        self.anchored = bool(anchored)
        self.blind_to_image = bool(blind_to_image)
        self.grid_size = int(grid_size)
        self.grid_radius = float(grid_radius)

        self.embedding = nn.Embedding(self.num_tokens, dim)
        if self.anchored and not self.blind_to_image:
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

    def as_extra_block(self, batch_size: int) -> ExtraTokenBlock:
        """The fork-facing block for one forward pass."""
        has_update = self.anchored and not self.blind_to_image
        return ExtraTokenBlock(
            name=self.name,
            tokens=self.embedding.weight[None].expand(batch_size, -1, -1),
            update_fn=self._update if has_update else None,
            blind_to_image=self.blind_to_image,
        )

    def _update(
        self,
        start_idx: int,
        image_embeddings: torch.Tensor,
        token_embeddings: torch.Tensor,
        token_augment: torch.Tensor,
        pose_output: Dict,
    ):
        """Anchored per-layer update of this block's token rows.

        Contract fixed by the fork's extra-token-block hook:
        ``(start_idx, image_embeddings, token_embeddings, token_augment,
        pose_output) -> (token_embeddings, token_augment)``. Only rows
        ``[start_idx, start_idx + num_anchored)`` are touched (global tokens
        excluded).
        """
        # Predicted 2D keypoint positions in crop space (-0.5 to 0.5)
        pred_kps_2d = pose_output["pred_keypoints_2d_cropped"].clone()  # [B, 70, 2]
        pred_kps_depth = pose_output["pred_keypoints_2d_depth"].clone()  # [B, 70]

        anchor_kps_2d = pred_kps_2d[:, self.keypoint_indices]       # [B, K, 2]
        anchor_kps_depth = pred_kps_depth[:, self.keypoint_indices]  # [B, K]

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
        token_augment[:, start_idx : start_idx + self.num_anchored, :] = (
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
        token_embeddings[:, start_idx : start_idx + self.num_anchored, :] += (
            self.feat_linear(sampled_feats)
        )

        return token_embeddings, token_augment
