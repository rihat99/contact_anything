# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn

from ..modules.transformer import FFN


class ContactHead(nn.Module):
    """
    Predict per-vertex contact states from contact query tokens.

    Takes all contact tokens (corresponding to the first 21 MHR70 keypoints:
    body joints + toes/heels) and predicts binary contact for each of the
    18439 MHR mesh vertices.

    Two pooling modes:
        - "attention": Learnable query attends over tokens -> [B, 1, C], then MLP.
        - "concat": Flatten all tokens -> [B, num_tokens * C], project down to C,
          then MLP.  Preserves per-token information without the bottleneck of
          compressing everything into a single attention query, while keeping
          parameter count manageable via the linear projection.
    """

    NUM_VERTICES = 18439

    def __init__(
        self,
        input_dim: int,
        num_contact_tokens: int = 21,
        num_vertices: int = 18439,
        mlp_depth: int = 2,
        mlp_channel_div_factor: int = 4,
        pool_num_heads: int = 8,
        pool_mode: str = "attention",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_contact_tokens = num_contact_tokens
        self.num_vertices = num_vertices
        self.pool_mode = pool_mode

        if pool_mode == "attention":
            self.pool_query = nn.Parameter(torch.zeros(1, 1, input_dim))
            nn.init.trunc_normal_(self.pool_query, std=0.02)
            self.pool_attn = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=pool_num_heads,
                batch_first=True,
            )
            mlp_input_dim = input_dim
        elif pool_mode == "concat":
            concat_dim = num_contact_tokens * input_dim
            self.concat_proj = nn.Sequential(
                nn.Linear(concat_dim, input_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            mlp_input_dim = input_dim
        else:
            raise ValueError(f"Unknown pool_mode: {pool_mode!r}")

        self.proj = FFN(
            embed_dims=mlp_input_dim,
            feedforward_channels=mlp_input_dim // mlp_channel_div_factor,
            output_dims=num_vertices,
            num_fcs=mlp_depth,
            ffn_drop=dropout,
            add_identity=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: contact tokens  [B, num_contact_tokens, C]

        Returns:
            contact_logits: [B, num_vertices]  (un-sigmoid-ed)
        """
        if self.pool_mode == "attention":
            batch_size = x.shape[0]
            query = self.pool_query.expand(batch_size, -1, -1)
            x_pooled, _ = self.pool_attn(query, x, x)  # [B, 1, C]
            contact_logits = self.proj(x_pooled).squeeze(1)
        else:  # concat
            x_flat = x.flatten(1)            # [B, num_tokens * C]
            x_proj = self.concat_proj(x_flat)  # [B, C]
            contact_logits = self.proj(x_proj)  # [B, num_vertices]

        return contact_logits
