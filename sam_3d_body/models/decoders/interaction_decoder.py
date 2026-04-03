"""
InteractionDecoder: transformer decoder for contact prediction.

Token layout (fixed):
  - 24 body-part query tokens: auxiliary supervision, one per SMPL joint/part.
    These mimic the SAM-3D body decoder's query structure.
  - 1 vertex query token: drives the dense per-vertex contact map.
  Total = 25 tokens. Output is split by the caller:
      tokens[:, :24, :] → PartContactHead  (part logits)
      tokens[:, 24:, :] → ContactHead      (vertex logits)

Architecture (per layer):
  1. Self-attention among all 25 tokens
  2. Cross-attention to image features (mask-conditioned, see below)
  3. Cross-attention to SAM-3D body pose token (detached — one-directional)
  4. FFN
  All sub-layers use pre-LayerNorm + residual connections.

Mask conditioning (Step 2 only):
  person_mask [B, 1, H, W] and object_mask [B, 1, H, W] are stacked into a
  2-channel input [B, 2, H, W] and passed through a small CNN (MaskEncoder).
  The output [B, d_model, H, W] is added to image features before any attention —
  the same additive fusion used by the SAM3D body decoder for human mask conditioning.
  The CNN's final layer is zero-initialized so the mask has no effect at init,
  and the model learns how much spatial context to inject from each mask channel.

Body token is detached before use as keys/values — no gradients flow to the frozen
body decoder.
"""

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """Channel-last LayerNorm for 2D feature maps [B, C, H, W]."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias   = nn.Parameter(torch.zeros(num_channels))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MaskEncoder(nn.Module):
    """
    Small CNN that encodes person + object masks into dense image-space embeddings.

    Input:  [B, 2, H, W]  — channel 0: person_mask, channel 1: object_mask
    Output: [B, d_model, H, W]

    The final 1×1 conv is zero-initialized so at init the mask contributes
    nothing, and the model learns how much mask context to inject.
    """

    def __init__(self, d_model: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden, kernel_size=3, padding=1),
            LayerNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden * 4, kernel_size=3, padding=1),
            LayerNorm2d(hidden * 4),
            nn.GELU(),
            nn.Conv2d(hidden * 4, d_model, kernel_size=1),  # ← zero-initialized
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, d_model, H, W]


class InteractionDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        # 1. Self-attention among all contact tokens
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # 2. Cross-attention to image features
        self.cross_attn_image = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        # 3. Cross-attention to body token (one-directional, body→contact, detached)
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
        q: torch.Tensor,          # [B, 25, d_model]
        img_feats: torch.Tensor,  # [B, HW, d_model]
        body_token: torch.Tensor, # [B, 1, d_model]  (already detached)
    ) -> torch.Tensor:
        # 1. Self-attention
        x = self.norm1(q)
        x, _ = self.self_attn(x, x, x)
        q = q + self.dropout(x)

        # 2. Cross-attention to image features
        x = self.norm2(q)
        x, _ = self.cross_attn_image(x, img_feats, img_feats)
        q = q + self.dropout(x)

        # 3. Cross-attention to body token
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
    Transformer decoder for contact prediction.

    Uses 24 body-part tokens (auxiliary) + 1 vertex token (dense contact map).
    The split is fixed: caller receives [B, 25, d_model] and slices as needed.

    Args:
        d_model:            Hidden dimension (should match body decoder dim, typically 1024).
        image_feat_dim:     Backbone output channels (1280 for DINOv3-H).
        num_layers:         Number of decoder layers.
        num_heads:          Number of attention heads.
        ffn_dim:            Hidden dim of per-layer FFN.
        dropout:            Dropout probability.
        mask_encoder_hidden: Hidden channels in MaskEncoder CNN (default 16).
    """

    NUM_PART_TOKENS = 24
    NUM_VERTEX_TOKENS = 1
    NUM_TOKENS = NUM_PART_TOKENS + NUM_VERTEX_TOKENS  # 25

    def __init__(
        self,
        d_model: int = 1024,
        image_feat_dim: int = 1280,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.0,
        mask_encoder_hidden: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # 24 learnable body-part query embeddings (auxiliary)
        self.part_queries = nn.Embedding(self.NUM_PART_TOKENS, d_model)
        # 1 learnable global vertex query embedding
        self.vertex_query = nn.Embedding(self.NUM_VERTEX_TOKENS, d_model)

        # Project backbone features → d_model if they differ
        self.image_proj = (
            nn.Linear(image_feat_dim, d_model)
            if image_feat_dim != d_model
            else nn.Identity()
        )

        # Mask encoder: person + object mask → dense additive embedding on image features
        # Zero-initialized final layer → no-op at init, model learns mask influence
        self.mask_encoder = MaskEncoder(d_model, hidden=mask_encoder_hidden)

        self.layers = nn.ModuleList([
            InteractionDecoderLayer(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        image_features: torch.Tensor,       # [B, C, H, W]
        body_tokens: torch.Tensor,          # [B, N, d_model] — pose token [B, 1, d_model]
        person_mask: torch.Tensor = None,   # [B, 1, H, W]  — Step 2 only
        object_mask: torch.Tensor = None,   # [B, 1, H, W]  — Step 2 only
    ) -> torch.Tensor:
        """
        Returns all contact tokens [B, 25, d_model].
          [:, :24, :] — body-part tokens → PartContactHead
          [:, 24:, :] — vertex token    → ContactHead
        """
        B, C, H, W = image_features.shape

        # Flatten spatial dims and project to d_model
        img_flat = image_features.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        img_flat = self.image_proj(img_flat)                    # [B, HW, d_model]

        # Additive mask conditioning (Step 2 only)
        if person_mask is not None or object_mask is not None:
            zeros = torch.zeros(B, 1, H, W, device=image_features.device)
            pm = person_mask.float() if person_mask is not None else zeros
            om = object_mask.float() if object_mask is not None else zeros
            mask_in  = torch.cat([pm, om], dim=1)               # [B, 2, H, W]
            mask_emb = self.mask_encoder(mask_in)               # [B, d_model, H, W]
            mask_flat = mask_emb.flatten(2).permute(0, 2, 1)   # [B, HW, d_model]
            img_flat = img_flat + mask_flat.to(dtype=img_flat.dtype)

        # Detach body tokens — no gradient flows back to body decoder
        body_kv = body_tokens.detach()

        # Concatenate part + vertex queries → [B, 25, d_model]
        q = torch.cat([
            self.part_queries.weight.unsqueeze(0).expand(B, -1, -1).contiguous(),
            self.vertex_query.weight.unsqueeze(0).expand(B, -1, -1).contiguous(),
        ], dim=1)

        for layer in self.layers:
            q = layer(q, img_flat, body_kv)

        return self.norm(q)  # [B, 25, d_model]
