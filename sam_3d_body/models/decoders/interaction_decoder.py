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
  2. Cross-attention to image features (spatial, hard-masked)
  3. Cross-attention to SAM-3D body pose token (detached — one-directional)
  4. FFN
  All sub-layers use pre-LayerNorm + residual connections.

Mask conditioning (Step 2 only):
  person_mask [B, 1, H, W] and object_mask [B, 1, H, W] — their union is used
  to zero out image features outside the combined region before any attention.
  The decoder only attends to features inside the human + object area.

Body token is detached before use as keys/values — no gradients flow to the frozen
body decoder.
"""

import torch
import torch.nn as nn


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
        img_feats: torch.Tensor,  # [B, HW, d_model]  (already masked by caller)
        body_token: torch.Tensor, # [B, 1, d_model]   (already detached)
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
        d_model:        Hidden dimension (should match body decoder dim, typically 1024).
        image_feat_dim: Backbone output channels (1280 for DINOv3-H).
        num_layers:     Number of decoder layers.
        num_heads:      Number of attention heads.
        ffn_dim:        Hidden dim of per-layer FFN.
        dropout:        Dropout probability.
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
        B = image_features.shape[0]

        # Flatten spatial dims and project to d_model
        img_flat = image_features.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        img_flat = self.image_proj(img_flat)                    # [B, HW, d_model]

        # Hard masking: zero out features outside union(person_mask, object_mask)
        if person_mask is not None or object_mask is not None:
            if person_mask is not None and object_mask is not None:
                combined = (person_mask.float() + object_mask.float()).clamp(max=1.0)
            elif person_mask is not None:
                combined = person_mask.float()
            else:
                combined = object_mask.float()
            mask_flat = combined.flatten(2).permute(0, 2, 1)   # [B, HW, 1]
            img_flat = img_flat * mask_flat.to(dtype=img_flat.dtype)

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
