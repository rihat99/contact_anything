"""Prediction heads for the contact / force / motion token blocks.

Every head reads its block's final decoder tokens (``[B, K, C]``) after the
post-decoder temporal mixing and regresses the block's output. They are plain
trainable modules — the frozen base model never sees them.

* :class:`ContactHead` — contact logits (``per_token`` / ``attention`` /
  ``concat`` pooling).
* :class:`ForceHead` — one 3D force vector per token, body-weight units,
  zero-initialised.
* :class:`MotionHead` — standardized vel/acc (+ optional root angular pair)
  per token, zero-initialised.
* :func:`contact_gate_forces` — the parameter-free contact gate multiplying
  the final force output by the (detached) per-group contact probabilities.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model.sam_3d_body.models.modules.transformer import FFN


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


FORCE_GATE_CONTACT_MAP = (0, 1, 2, 3, 4, 5)


def contact_gate_forces(
    joint_forces: torch.Tensor, contact_logits: torch.Tensor, sharpness: float
) -> torch.Tensor:
    """Gate the final force output by the (detached) per-group contact logits.

    ``gate_k = sigmoid(sharpness * contact_logits[:, MAP[k]])`` — nearly 1 for a
    confident contact (logit 1.0 -> gate 0.982 at sharpness 4.0), nearly 0 for a
    confident free limb. The logits are detached unconditionally: the force loss
    must not rewrite the calibrated contact probabilities through this product
    (contact trains purely from its labels).

    Args:
        joint_forces: raw force output ``[B, 6, 3]`` (kindyn group order).
        contact_logits: kindyn_6 contact logits ``[B, 6]`` (same group order).
        sharpness: gate steepness multiplier on the logits.

    Returns:
        gated forces ``[B, 6, 3]``.
    """
    if joint_forces.shape[1] != len(FORCE_GATE_CONTACT_MAP):
        raise ValueError(
            "contact-gated forces need one group per gate-map entry: "
            f"forces={joint_forces.shape[1]}, map={len(FORCE_GATE_CONTACT_MAP)}"
        )
    if contact_logits.shape[-1] <= max(FORCE_GATE_CONTACT_MAP):
        raise ValueError(
            "contact-gate map indexes contact output "
            f"{max(FORCE_GATE_CONTACT_MAP)} but the contact head produced only "
            f"{contact_logits.shape[-1]} logits"
        )
    gate = torch.sigmoid(
        sharpness * contact_logits.detach()[:, list(FORCE_GATE_CONTACT_MAP)]
    )  # [B, 6]
    return joint_forces * gate.unsqueeze(-1)


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


class MotionHead(nn.Module):
    """Regress velocity + acceleration per motion token.

    Mirrors :class:`ForceHead`: a single FFN applied independently along the
    token axis, mapping ``[B, K, C] -> [B, K, output_dims]``. The outputs are
    the **standardized** root-frame linear velocity (``[..., 0:3]``) and linear
    acceleration (``[..., 3:6]``) of the token's kindyn joint — plus, when
    ``output_dims`` is 12, the root body angular velocity (``[..., 6:9]``) and
    angular acceleration (``[..., 9:12]``); the supervised loss owns the
    mean/std table, so the head never sees physical units.

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
        output_dims: int = 6,
    ):
        super().__init__()

        self.num_motion_tokens = num_motion_tokens

        # FFN operates on the last dimension, so the same weights are applied
        # independently to every token: [B, K, C] -> [B, K, output_dims].
        self.proj = FFN(
            embed_dims=input_dim,
            feedforward_channels=input_dim // mlp_channel_div_factor,
            output_dims=output_dims,
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
            motion: ``[B, num_motion_tokens, output_dims]`` — standardized
            ``(vel, acc[, ang_vel, ang_acc])`` in root axes.
        """
        if x.shape[1] != self.num_motion_tokens:
            raise ValueError(
                "motion-head input token count does not match the configured head: "
                f"input={x.shape[1]}, configured={self.num_motion_tokens}"
            )
        return self.proj(x)  # [B, K, 6|12]
