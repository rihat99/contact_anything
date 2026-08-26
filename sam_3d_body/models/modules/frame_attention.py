"""Per-frame attention over a modality's tokens (no temporal mixing).

:class:`FrameAttentionModule` runs AFTER the temporal modules: within every
single frame, one modality's tokens (contact, force, motion, or the pose
token) attend a ``context`` block spanning the tokens of ALL modalities of
that frame — so modalities exchange information at the current timestep
without any cross-frame path. Frames are independent by construction (the
batch dim is the flattened frame dim), so the module works identically for
clips and single images and needs no ``seq_len`` plumbing.

Like :class:`.temporal.ContactTemporalModule`, every attention/FFN branch is
gated by a zero-initialised per-channel ``gamma`` and the optional bottleneck
adapter projects only the delta (bias-free out-projection), so the module is
an exact identity at init. No positional encoding: tokens are distinguished by
their content/embeddings, and there is no time axis to encode.

The owning model binds instances under a ``frame_attn`` attribute (one per
modality), which the freeze/eval filters in :mod:`contact.model` match on.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class _FrameBlock(nn.Module):
    """Pre-LN cross-attention block with zero-initialised residual gates.

    ``x -> x + gamma_attn * MHA(q=LN(x), kv=LN(context));
    x -> x + gamma_ffn * FFN(LN(x))``. Both gammas start at zero, so the block
    is an exact identity at init; ``context`` is read, never written.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.gamma_attn = nn.Parameter(torch.zeros(dim))

        hidden = int(dim * mlp_ratio)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.gamma_ffn = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        normed_kv = self.norm_kv(context)
        attn_out, _ = self.attn(
            self.norm_attn(x), normed_kv, normed_kv, need_weights=False
        )
        x = x + self.gamma_attn * attn_out
        x = x + self.gamma_ffn * self.ffn(self.norm_ffn(x))
        return x


class FrameAttentionModule(nn.Module):
    """Zero-gated per-frame attention over one modality's tokens.

    :param dim: token working dim (``DECODER.DIM``).
    :param num_layers: number of stacked :class:`_FrameBlock` s.
    :param num_heads: attention heads per block.
    :param mlp_ratio: FFN hidden expansion factor.
    :param dropout: dropout inside attention/FFN (default 0.0).
    :param bottleneck_dim: attention width: project ``dim -> bottleneck_dim ->
        dim`` and add only the delta. ``None`` = attend at full ``dim``.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        bottleneck_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.bottleneck_dim = int(dim if bottleneck_dim is None else bottleneck_dim)
        block_dim = self.bottleneck_dim
        if block_dim != dim:
            self.token_in_proj = nn.Linear(dim, block_dim)
            self.context_in_proj = nn.Linear(dim, block_dim)
            # Bias-free so a zero delta maps to an exact zero and the complete
            # adapter remains bitwise identity at initialization.
            self.token_out_proj = nn.Linear(block_dim, dim, bias=False)
        self.blocks = nn.ModuleList(
            _FrameBlock(block_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        )

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Attend within each frame; return the updated ``tokens``.

        :param tokens: one modality's tokens ``[B, K, C]`` (``B`` = flattened
            frames; each row is one frame — no cross-frame mixing).
        :param context: keys/values ``[B, S, C]``: every modality's tokens of
            the same frame (``tokens`` is one of its slices). Read, never
            written.
        """
        b, num_k, dim = tokens.shape
        assert dim == self.dim, f"token dim {dim} != configured {self.dim}"
        assert context.shape[0] == b and context.shape[2] == dim, (
            f"context {tuple(context.shape)} does not match tokens "
            f"{tuple(tokens.shape)}")

        if hasattr(self, "token_in_proj"):
            working = self.token_in_proj(tokens)
            working_ctx = self.context_in_proj(context)
        else:
            working, working_ctx = tokens, context

        x = working
        for block in self.blocks:
            x = block(x, working_ctx)

        if hasattr(self, "token_out_proj"):
            # The blocks are exact identities while their gammas are zero.
            # Project only their delta, not the token itself.
            return tokens + self.token_out_proj(x - working)
        return x
