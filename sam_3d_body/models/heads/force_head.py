# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn

from ..modules.transformer import FFN


class ForceHead(nn.Module):
    """Regress one 3D force vector per force token.

    Mirrors :class:`ContactHead`'s ``per_token`` branch: a single FFN applied
    independently along the token axis, mapping ``[B, K, C] -> [B, K, 3]``. No
    sigmoid — the output is a dimensionless force in units of body weight (the
    physics loss converts to newtons).

    The final linear is zero-initialised so the model starts predicting zero
    force everywhere; the RNEA residual then reduces to the pure-kinematics
    baseline before any force is learned (a meaningful curriculum start).
    """

    def __init__(
        self,
        input_dim: int,
        num_force_tokens: int,
        mlp_depth: int = 2,
        mlp_channel_div_factor: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_force_tokens = num_force_tokens

        # FFN operates on the last dimension, so the same weights are applied
        # independently to every token: [B, K, C] -> [B, K, 3].
        self.proj = FFN(
            embed_dims=input_dim,
            feedforward_channels=input_dim // mlp_channel_div_factor,
            output_dims=3,
            num_fcs=mlp_depth,
            ffn_drop=dropout,
            add_identity=False,
        )

        # Zero-init the final linear -> zero force at init (D5).
        final_linear = [m for m in self.proj.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: force tokens  ``[B, num_force_tokens, C]``

        Returns:
            joint_forces: ``[B, num_force_tokens, 3]`` (dimensionless, body-weight units)
        """
        if x.shape[1] != self.num_force_tokens:
            raise ValueError(
                "force-head input token count does not match the configured head: "
                f"input={x.shape[1]}, configured={self.num_force_tokens}"
            )
        return self.proj(x)  # [B, K, 3]
