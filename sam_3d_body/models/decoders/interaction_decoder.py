"""
InteractionDecoder: separate transformer decoder for contact prediction.

Architecture (per layer):
  1. Self-attention among K contact query tokens
  2. Cross-attention to image features (spatial)
  3. Cross-attention to body decoder pose token (detached — one-directional)
  4. FFN
  All sub-layers use pre-LayerNorm + residual connections.

The body decoder's output tokens are detached before being used as keys/values,
so no gradients flow back to the (frozen) body decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InteractionDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        # 1. Self-attention among contact tokens
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # 2. Cross-attention to image features
        self.cross_attn_image = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        # 3. Cross-attention to body token (one-directional, body->contact)
        self.cross_attn_body = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)

        # 4. FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,          # [B, K, d_model]  contact queries
        img_feats: torch.Tensor,  # [B, HW, d_model] flattened image features
        body_token: torch.Tensor, # [B, 1, d_model]  body pose token (already detached)
    ) -> torch.Tensor:
        # 1. Self-attention
        x = self.norm1(q)
        x, _ = self.self_attn(x, x, x)
        q = q + self.dropout(x)

        # 2. Cross-attention to image
        x = self.norm2(q)
        x, _ = self.cross_attn_image(x, img_feats, img_feats)
        q = q + self.dropout(x)

        # 3. Cross-attention to body token (one-directional; body_token already detached)
        x = self.norm3(q)
        x, _ = self.cross_attn_body(x, body_token, body_token)
        q = q + self.dropout(x)

        # 4. FFN
        x = self.norm4(q)
        x = self.ffn(x)
        q = q + self.dropout(x)

        return q


class InteractionDecoder(nn.Module):
    """
    Separate decoder for contact prediction.

    Args:
        d_model:            Hidden dimension (should match body decoder dim, typically 1024).
        image_feat_dim:     Backbone output channels (1280 for DINOv3-H).
        num_contact_tokens: Number of learnable contact query tokens (K).
        num_layers:         Number of decoder layers.
        num_heads:          Number of attention heads.
        ffn_dim:            Hidden dim of per-layer FFN.
        dropout:            Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 1024,
        image_feat_dim: int = 1280,
        num_contact_tokens: int = 16,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_contact_tokens = num_contact_tokens

        # Learnable contact query embeddings
        self.contact_queries = nn.Embedding(num_contact_tokens, d_model)

        # Project backbone features (image_feat_dim) → d_model if they differ
        if image_feat_dim != d_model:
            self.image_proj = nn.Linear(image_feat_dim, d_model)
        else:
            self.image_proj = nn.Identity()

        self.layers = nn.ModuleList([
            InteractionDecoderLayer(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        image_features: torch.Tensor,  # [B, C, H, W]
        body_tokens: torch.Tensor,     # [B, N, d_model] — caller should pass pose token only
    ) -> torch.Tensor:
        """
        Returns contact tokens [B, K, d_model].
        body_tokens is detached inside this forward to enforce one-directional flow.
        """
        B = image_features.shape[0]

        # Flatten spatial dims and project to d_model
        # image_features: [B, C, H, W] → [B, H*W, C] → [B, H*W, d_model]
        img_flat = image_features.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        img_flat = self.image_proj(img_flat)                   # [B, HW, d_model]

        # Detach body tokens — no gradient flows back to body decoder
        body_kv = body_tokens.detach()  # [B, 1, d_model]

        # Expand learnable queries to batch
        q = self.contact_queries.weight.unsqueeze(0).expand(B, -1, -1)  # [B, K, d_model]
        q = q.contiguous()

        for layer in self.layers:
            q = layer(q, img_flat, body_kv)

        return self.norm(q)  # [B, K, d_model]
