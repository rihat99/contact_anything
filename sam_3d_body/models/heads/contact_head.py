"""
ContactHead and PartContactHead for human body contact prediction.

ContactHead:
    Maps the single vertex query token from InteractionDecoder to per-vertex
    contact logits.
    Input:  [B, 1, d_model]  (the last token from InteractionDecoder)
    Output: [B, num_vertices] raw logits

PartContactHead:
    Maps the 24 body-part query tokens to per-part contact logits.
    Input:  [B, 24, d_model]
    Output: [B, 24] raw logits
    Uses a shared linear projection across all part tokens.
"""

import torch
import torch.nn as nn


class ContactHead(nn.Module):
    """
    Per-vertex contact prediction head.

    Takes the single vertex token [B, 1, d_model], squeezes it, and runs an MLP
    to produce [B, num_vertices] logits.  No pooling needed — single token.

    Args:
        input_dim:    Token dimension (d_model from InteractionDecoder).
        num_vertices: Number of output vertices (6890 for SMPL).
        mlp_depth:    Number of linear layers (default 2).
        hidden_dim:   Hidden dimension in MLP (default 512).
        dropout:      Dropout before each non-final layer (default 0.0).
    """

    def __init__(
        self,
        input_dim: int,
        num_vertices: int,
        mlp_depth: int = 2,
        hidden_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_vertices = num_vertices

        layers = []
        in_dim = input_dim
        for i in range(mlp_depth):
            out_dim = num_vertices if i == mlp_depth - 1 else hidden_dim
            if dropout > 0.0 and i < mlp_depth - 1:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(in_dim, out_dim))
            if i < mlp_depth - 1:
                layers.append(nn.GELU())
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, vertex_token: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vertex_token: [B, 1, d_model]  — last token from InteractionDecoder
        Returns:
            logits: [B, num_vertices]
        """
        x = vertex_token.squeeze(1)  # [B, d_model]
        return self.mlp(x)           # [B, num_vertices]


class PartContactHead(nn.Module):
    """
    Per-body-part contact prediction head.

    Takes the 24 body-part tokens [B, 24, d_model] and applies a shared linear
    projection to each token independently, producing one logit per part.

    Args:
        input_dim: Token dimension (d_model from InteractionDecoder).
        num_parts: Number of body parts (default 24, matching SMPL_PART_NAMES).
    """

    def __init__(self, input_dim: int, num_parts: int = 24):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def forward(self, part_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            part_tokens: [B, 24, d_model]
        Returns:
            logits: [B, 24]
        """
        return self.proj(part_tokens).squeeze(-1)  # [B, 24]
