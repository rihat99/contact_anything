"""
ObjectMaskEncoder: encode a binary object mask into a small set of object tokens
for use as a prompt in the InteractionDecoder (Step 2).

Input:  object_mask [B, 1, H_feat, W_feat]  — binary float, already cropped to the
        person bbox and resized to the feature-map spatial resolution (e.g. 56×56).
Output: object_tokens [B, num_obj_tokens, d_model]
"""

import torch
import torch.nn as nn


class ObjectMaskEncoder(nn.Module):
    """
    Encodes a spatial binary mask into object prompt tokens.

    Architecture:
        4 strided conv layers (1→32→64→128→d_model), 3×3 kernel, stride 2, GELU.
        At 56×56 input: 56→28→14→7→4 spatial dims.
        AdaptiveAvgPool2d(1) → [B, d_model] → unsqueeze → [B, 1, d_model].
        A learnable type_embed [1, 1, d_model] is added to distinguish object tokens
        from body tokens and image features.

    Args:
        d_model:          Token dimension (must match InteractionDecoder d_model).
        num_obj_tokens:   Currently only 1 is supported (global avg pool).
        dropout:          Dropout probability applied to output tokens.
    """

    def __init__(
        self,
        d_model: int = 1024,
        num_obj_tokens: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_obj_tokens != 1:
            raise ValueError("ObjectMaskEncoder currently supports only num_obj_tokens=1.")
        self.d_model = d_model
        self.num_obj_tokens = num_obj_tokens

        # Small CNN: 4 strided conv layers
        # 56×56 → 28 → 14 → 7 → 4 (spatial)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # → 28
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # → 14
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # → 7
            nn.GELU(),
            nn.Conv2d(128, d_model, kernel_size=3, stride=2, padding=1),  # → 4
            nn.GELU(),
        )

        # Global average pool → single token
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Learnable type embedding — distinguishes object tokens from body/image tokens
        self.type_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.type_embed, std=0.02)

        self.dropout = nn.Dropout(dropout)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mask: [B, 1, H, W] float32 binary mask (values in {0, 1}).
                  Should be pre-resized to the feature map spatial resolution (e.g. 56×56).

        Returns:
            object_tokens: [B, 1, d_model]
        """
        B = mask.shape[0]
        x = self.conv(mask)            # [B, d_model, ~4, ~4]
        x = self.pool(x)               # [B, d_model, 1, 1]
        x = x.view(B, self.d_model)    # [B, d_model]
        x = x.unsqueeze(1)             # [B, 1, d_model]
        x = x + self.type_embed        # add type embedding
        return self.dropout(x)         # [B, 1, d_model]
