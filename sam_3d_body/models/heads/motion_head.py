# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn

from ..modules.transformer import FFN


class MotionHead(nn.Module):
    """Regress linear velocity + acceleration per motion token.

    Mirrors :class:`ForceHead`: a single FFN applied independently along the
    token axis, mapping ``[B, K, C] -> [B, K, 6]``. The six outputs are the
    **standardized** root-frame linear velocity (first three) and linear
    acceleration (last three) of the token's kindyn joint; the supervised loss
    owns the mean/std table, so the head never sees physical units.

    The final linear is zero-initialised, so at init every token predicts the
    standardized mean (the dataset's average velocity/acceleration) — the same
    curriculum start the force head uses.
    """

    def __init__(
        self,
        input_dim: int,
        num_motion_tokens: int,
        mlp_depth: int = 2,
        mlp_channel_div_factor: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_motion_tokens = num_motion_tokens

        # FFN operates on the last dimension, so the same weights are applied
        # independently to every token: [B, K, C] -> [B, K, 6].
        self.proj = FFN(
            embed_dims=input_dim,
            feedforward_channels=input_dim // mlp_channel_div_factor,
            output_dims=6,
            num_fcs=mlp_depth,
            ffn_drop=dropout,
            add_identity=False,
        )

        # Zero-init the final linear -> standardized-mean prediction at init.
        final_linear = [m for m in self.proj.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: motion tokens ``[B, num_motion_tokens, C]``

        Returns:
            motion: ``[B, num_motion_tokens, 6]`` — standardized ``(vel, acc)``
            in root axes.
        """
        if x.shape[1] != self.num_motion_tokens:
            raise ValueError(
                "motion-head input token count does not match the configured head: "
                f"input={x.shape[1]}, configured={self.num_motion_tokens}"
            )
        return self.proj(x)  # [B, K, 6]
