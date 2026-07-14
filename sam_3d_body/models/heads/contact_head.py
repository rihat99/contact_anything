# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn

from ..modules.transformer import FFN


class ContactHead(nn.Module):
    """Predict contact logits from a bank of contact query tokens.

    Three output modes are supported:
        - "attention": Learnable query attends over tokens -> [B, 1, C], then MLP.
        - "concat": Flatten all tokens -> [B, num_tokens * C], project down to C,
          then MLP.  Preserves per-token information without the bottleneck of
          compressing everything into a single attention query, while keeping
          parameter count manageable via the linear projection.
        - "per_token": Apply one shared MLP independently to every token, producing
          one logit per token.  This mode requires the output dimension to equal the
          total number of contact tokens.

    ``attention`` and ``concat`` can emit an arbitrary target dimension (for
    example an entire body-22 or vertex target). ``per_token`` preserves token
    identity and is intended for explicitly aligned contact-token targets.
    """

    def __init__(
        self,
        input_dim: int,
        num_contact_tokens: int,
        output_dims: int,
        mlp_depth: int = 2,
        mlp_channel_div_factor: int = 4,
        pool_num_heads: int = 8,
        pool_mode: str = "attention",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_contact_tokens = num_contact_tokens
        self.output_dims = output_dims
        self.pool_mode = pool_mode

        if pool_mode == "per_token":
            if output_dims != num_contact_tokens:
                raise ValueError(
                    "pool_mode='per_token' requires output_dims to equal the total "
                    f"contact-token count; got output_dims={output_dims}, "
                    f"num_contact_tokens={num_contact_tokens}"
                )
            # FFN operates on the last dimension, so the same weights are applied
            # independently to every token: [B, K, C] -> [B, K, 1].
            self.proj = FFN(
                embed_dims=input_dim,
                feedforward_channels=input_dim // mlp_channel_div_factor,
                output_dims=1,
                num_fcs=mlp_depth,
                ffn_drop=dropout,
                add_identity=False,
            )
        elif pool_mode == "attention":
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

        if pool_mode != "per_token":
            self.proj = FFN(
                embed_dims=mlp_input_dim,
                feedforward_channels=mlp_input_dim // mlp_channel_div_factor,
                output_dims=output_dims,
                num_fcs=mlp_depth,
                ffn_drop=dropout,
                add_identity=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: contact tokens  [B, num_contact_tokens, C]

        Returns:
            contact_logits: [B, output_dims] (raw, before sigmoid)
        """
        if self.pool_mode == "per_token":
            if x.shape[1] != self.num_contact_tokens:
                raise ValueError(
                    "per-token input token count does not match the configured head: "
                    f"input={x.shape[1]}, configured={self.num_contact_tokens}"
                )
            contact_logits = self.proj(x).squeeze(-1)  # [B, K]
        elif self.pool_mode == "attention":
            batch_size = x.shape[0]
            query = self.pool_query.expand(batch_size, -1, -1)
            x_pooled, _ = self.pool_attn(query, x, x)  # [B, 1, C]
            contact_logits = self.proj(x_pooled).squeeze(1)
        else:  # concat
            x_flat = x.flatten(1)            # [B, num_tokens * C]
            x_proj = self.concat_proj(x_flat)  # [B, C]
            contact_logits = self.proj(x_proj)  # [B, output_dims]

        return contact_logits
