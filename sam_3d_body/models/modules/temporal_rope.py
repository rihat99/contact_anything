"""RoPE-based temporal transformer over per-frame decoder tokens.

:class:`RopeTemporalModule` is the long-sequence temporal transformer for the
pose path (GVHMR-style, Shen et al. 2024). A batch of
``B_flat = B_clips * T`` flattened frames is reshaped to ``[B_clips, T, K, C]``
and self-attention runs across the ``T`` frames of each clip, independently per
token slot ``K`` (the pose path uses ``K = 1``).

Design points (vs the retired sliding-window / sinusoidal module):

* **Rotary position encoding on q/k** instead of an additive sinusoidal
  encoding: attention logits depend only on *relative* time offsets, so a
  model trained at ``T = 60`` runs single-pass on a whole scene.
* **Time-valued positions**: RoPE positions are ``frame_pos_sec * time_scale``
  (real elapsed seconds, ``time_scale = 25`` makes one 25-fps step ~= 1.0), so
  the corpus's variable fps is encoded exactly rather than approximated by
  frame indices.
* **Seconds-based local window**: attention is restricted to relative offsets
  ``|dt| <= max_rel_sec`` (the span seen in training), GVHMR's
  never-see-unseen-offsets rule. Inert on training-length clips; activates
  automatically on longer inference sequences.
* **Bidirectional only** — no causal option (offline video; no paper supports
  causal masking here).

Every attention/FFN branch is gated by a per-channel ``gamma = zeros``
parameter, so at initialisation the module is an exact identity
(``torch.equal``-verified), matching the repo's zero-init invariant. The
model attribute an instance is bound to (``pose_temporal``) carries the
substring the freeze/eval filters in :mod:`contact.model` match on.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_cos_sin(
    positions: torch.Tensor, head_dim: int, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cos/sin tables for rotary embedding at real-valued positions.

    :param positions: rotation positions (any real values), shape ``[..., T]``.
    :param head_dim: per-head channel count (must be even).
    :param base: RoPE frequency base.
    :returns: ``(cos, sin)`` each of shape ``[..., T, head_dim]`` (the
        ``head_dim/2`` frequencies duplicated for the two rotated halves),
        computed in float32.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even; got {head_dim}")
    half = head_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(half, device=positions.device, dtype=torch.float32)
                 * 2.0 / head_dim)
    )                                                       # [half]
    angles = positions.float().unsqueeze(-1) * inv_freq     # [..., T, half]
    angles = torch.cat([angles, angles], dim=-1)            # [..., T, head_dim]
    return angles.cos(), angles.sin()


def rope_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply a rotary embedding (rotate-half convention).

    :param x: ``[B, H, T, head_dim]`` queries or keys.
    :param cos: ``[B, 1, T, head_dim]`` (broadcast over heads).
    :param sin: ``[B, 1, T, head_dim]``.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


def frame_keep_mask(
    pos_sec: torch.Tensor,
    frame_valid: Optional[torch.Tensor],
    max_rel_sec: Optional[float],
) -> Optional[torch.Tensor]:
    """Frame-level attention keep-mask ``[n_clips, T, T]`` (``None`` = all-visible).

    Rules: keys further than ``max_rel_sec`` away in time are hidden; an invalid
    key frame is hidden from every query except its own diagonal, so no softmax
    row is ever fully masked. ``None`` is returned when neither rule bites, which
    lets the caller skip building a mask at all.

    :param pos_sec: elapsed seconds ``[n_clips, T]``.
    :param frame_valid: bool ``[n_clips, T]``, or ``None`` for all-valid.
    :param max_rel_sec: attention window half-width in seconds (``None`` = off).
    """
    n_clips, t = pos_sec.shape
    keep = None
    if max_rel_sec is not None:
        dt = (pos_sec[:, :, None] - pos_sec[:, None, :]).abs()
        if bool((dt > max_rel_sec).any()):
            keep = dt <= max_rel_sec                        # [n_clips, T, T]
    if frame_valid is not None and not bool(frame_valid.all()):
        valid_key = frame_valid[:, None, :].expand(n_clips, t, t)
        diag = torch.eye(t, dtype=torch.bool, device=pos_sec.device)
        valid_keep = valid_key | diag
        keep = valid_keep if keep is None else keep & valid_keep
    if keep is None:
        return None
    # A query must always see itself even inside the time window.
    return keep | torch.eye(t, dtype=torch.bool, device=pos_sec.device)


class _RopeBlock(nn.Module):
    """Pre-LN transformer block: RoPE attention + FFN, zero-gated residuals.

    ``x -> x + gamma_attn * Attn(LN(x));  x -> x + gamma_ffn * FFN(LN(x))``
    with q/k rotated by RoPE before the dot product. Both gammas start at
    zero, so the block is an exact identity at init.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = float(dropout)

        self.norm_attn = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
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

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        token_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one block.

        :param x: tokens ``[B, T, C]``.
        :param cos: RoPE table ``[B, 1, T, head_dim]``.
        :param sin: RoPE table ``[B, 1, T, head_dim]``.
        :param attn_mask: bool ``[B, 1, T, T]`` (``True`` = may attend) or
            ``None`` for all-visible.
        :param token_emb: optional identity embedding broadcastable onto
            ``[B, T, C]``, added to the LayerNormed input before the q/k/v
            projection. RoPE encodes *time* only, so a sequence carrying
            several tokens per frame needs this to tell its slots apart. Added
            INSIDE the gated branch, never to the residual stream, so the
            zero-gate identity at init is preserved exactly. ``None`` (the
            pose path) leaves the block bit-identical to before.
        """
        b, t, c = x.shape
        normed = self.norm_attn(x)
        if token_emb is not None:
            normed = normed + token_emb
        qkv = self.qkv(normed).reshape(b, t, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)      # each [B, H, T, hd]
        q = rope_rotate(q, cos, sin)
        k = rope_rotate(k, cos, sin)
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )                                                   # [B, H, T, hd]
        attn = attn.transpose(1, 2).reshape(b, t, c)
        attn = self.proj_drop(self.proj(attn))
        x = x + self.gamma_attn * attn
        x = x + self.gamma_ffn * self.ffn(self.norm_ffn(x))
        return x


class RopeTemporalModule(nn.Module):
    """Zero-gated RoPE temporal self-attention over a token block.

    :param dim: token working dim (``DECODER.DIM``); blocks run natively at
        this width (no bottleneck adapter).
    :param num_layers: number of stacked :class:`_RopeBlock` s.
    :param num_heads: attention heads per block (``dim / num_heads`` even).
    :param mlp_ratio: FFN hidden expansion factor.
    :param dropout: dropout inside attention/FFN.
    :param time_scale: multiplier from elapsed seconds to RoPE positions;
        ``25`` calibrates one 25-fps frame step to a unit position.
    :param max_rel_sec: attention window half-width in *seconds* (``None``
        disables). Frames further apart than this never attend each other;
        set it to the training clip span so inference on arbitrarily long
        sequences only ever sees trained relative offsets.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 4,
        num_heads: int = 16,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        time_scale: float = 25.0,
        max_rel_sec: Optional[float] = 2.5,
    ):
        super().__init__()
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale must be finite and positive")
        if max_rel_sec is not None and (
            not math.isfinite(max_rel_sec) or max_rel_sec <= 0
        ):
            raise ValueError("max_rel_sec must be finite and positive, or None")
        self.dim = dim
        self.time_scale = float(time_scale)
        self.max_rel_sec = None if max_rel_sec is None else float(max_rel_sec)
        self.blocks = nn.ModuleList(
            _RopeBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        )
        self.head_dim = self.blocks[0].head_dim if num_layers > 0 else dim // num_heads

    def _attn_mask(
        self, pos_sec: torch.Tensor, frame_valid: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Bool keep-mask ``[n_clips, 1, T, T]``, or ``None`` when all-visible.

        Thin head-dim wrapper around :func:`frame_keep_mask`.

        :param pos_sec: elapsed seconds ``[n_clips, T]``.
        :param frame_valid: bool ``[n_clips, T]`` or ``None``.
        """
        keep = frame_keep_mask(pos_sec, frame_valid, self.max_rel_sec)
        if keep is None:
            return None
        return keep[:, None]                                # [n_clips, 1, T, T]

    def forward(
        self,
        tokens: torch.Tensor,
        seq_len: int,
        frame_pos_sec: Optional[torch.Tensor] = None,
        frame_valid: Optional[torch.Tensor] = None,
        attend: Optional[str] = None,
        causal: Optional[bool] = None,
    ) -> torch.Tensor:
        """Temporal self-attention across the frames of each clip.

        :param tokens: ``[B_flat, K, C]`` with ``B_flat = B_clips * seq_len``
            (clip-major, frame-minor order); slots attend independently.
        :param seq_len: frames per clip ``T``.
        :param frame_pos_sec: elapsed seconds per flattened frame ``[B_flat]``;
            ``None`` (single images) falls back to zero positions.
        :param frame_valid: per-frame validity ``[B_flat]`` bool.
        :param attend: accepted for signature compatibility; must be ``None``
            or ``'per_token'`` (the only supported mode).
        :param causal: accepted for signature compatibility; must be falsy —
            the module is bidirectional by design.
        :returns: updated tokens ``[B_flat, K, C]``.
        """
        if attend not in (None, "per_token"):
            raise ValueError(f"RopeTemporalModule only attends per_token; got {attend!r}")
        if causal:
            raise ValueError("RopeTemporalModule is bidirectional; causal is unsupported")

        b_flat, num_slots, input_dim = tokens.shape
        if input_dim != self.dim:
            raise AssertionError(
                f"token dim {input_dim} does not match configured dim {self.dim}")
        if b_flat % seq_len != 0:
            raise AssertionError(f"batch {b_flat} not divisible by seq_len {seq_len}")
        n_clips = b_flat // seq_len

        if frame_pos_sec is None:
            pos_sec = torch.zeros(
                n_clips, seq_len, device=tokens.device, dtype=torch.float32)
        else:
            pos_sec = frame_pos_sec.to(tokens.device).float().view(n_clips, seq_len)

        cos, sin = rope_cos_sin(pos_sec * self.time_scale, self.head_dim)
        cos = cos.to(tokens.dtype)[:, None]                 # [n_clips, 1, T, hd]
        sin = sin.to(tokens.dtype)[:, None]
        valid = None
        if frame_valid is not None:
            valid = frame_valid.to(
                device=tokens.device, dtype=torch.bool).view(n_clips, seq_len)
        mask = self._attn_mask(pos_sec, valid)

        # Fold slots into the batch: each slot attends over its own T frames.
        x = (tokens.view(n_clips, seq_len, num_slots, input_dim)
             .permute(0, 2, 1, 3).reshape(n_clips * num_slots, seq_len, input_dim))
        if num_slots > 1:
            cos = cos.repeat_interleave(num_slots, dim=0)
            sin = sin.repeat_interleave(num_slots, dim=0)
            mask = None if mask is None else mask.repeat_interleave(num_slots, dim=0)
        for block in self.blocks:
            x = block(x, cos, sin, mask)
        return (x.view(n_clips, num_slots, seq_len, input_dim)
                .permute(0, 2, 1, 3).reshape(b_flat, num_slots, input_dim))
