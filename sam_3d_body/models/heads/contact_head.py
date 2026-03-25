"""
ContactHead: maps K contact tokens from the InteractionDecoder to per-vertex contact logits.

Input:  [B, K, d_model]  — contact tokens from InteractionDecoder
Output: [B, num_vertices] — raw (pre-sigmoid) contact logits
"""

import torch
import torch.nn as nn


class ContactHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_contact_tokens: int,
        num_vertices: int,
        mlp_depth: int = 2,
        mlp_channel_div_factor: int = 4,
        pool_mode: str = "attention",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_vertices = num_vertices
        self.pool_mode = pool_mode

        if pool_mode == "attention":
            self.pool_attn = nn.Linear(input_dim, 1)

        hidden_dim = max(input_dim // mlp_channel_div_factor, 64)

        layers = []
        in_dim = input_dim
        for i in range(mlp_depth):
            out_dim = num_vertices if i == mlp_depth - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < mlp_depth - 1:
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
                layers.append(nn.GELU())
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, contact_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            contact_tokens: [B, K, d_model]
        Returns:
            logits: [B, num_vertices]
        """
        if self.pool_mode == "attention":
            # Learned attention weights over K tokens
            weights = self.pool_attn(contact_tokens)  # [B, K, 1]
            weights = weights.softmax(dim=1)
            pooled = (contact_tokens * weights).sum(dim=1)  # [B, d_model]
        else:
            pooled = contact_tokens.mean(dim=1)  # [B, d_model]

        return self.mlp(pooled)  # [B, num_vertices]
