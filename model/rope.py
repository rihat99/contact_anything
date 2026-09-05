"""RoPE temporal transformer over the per-frame decoder tokens.

:class:`CrossModalRopeModule` is the single post-decoder mixing brick
(``model.cross_modal_temporal``): ONE self-attention over the concatenation of
the chosen modality token blocks across the frames of a clip.

Design points:

* **Rotary position encoding on q/k**: attention logits depend only on
  *relative* positions, so a model trained on ``T = 60`` clips runs single-pass
  on a whole scene.
* **Positions are real elapsed seconds** (``frame_pos_sec x time_scale``), so
  the block is exact under the corpus's variable frame rates.
* **Local window** (``window``, seconds): a key further than ``window`` from
  the query is hidden; the receptive field grows by ``window`` per layer, and
  inference on a longer sequence never exposes an untrained relative offset.
  ``None`` = every frame of the clip.
* **Bidirectional only** — offline video; no causal option.
* **Pre-LN residual blocks** with the attention and FFN output projections
  zero-initialised: the module is an exact identity at initialisation and the
  projections receive a first-order gradient from step one.

The module is bound outside the frozen wrapper (:class:`model.network.
ContactAnything`), so it trains by construction.
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
    seconds: torch.Tensor,
    frame_valid: Optional[torch.Tensor],
    window: Optional[float],
) -> Optional[torch.Tensor]:
    """Frame-level attention keep-mask ``[n_clips, T, T]`` (``None`` = all-visible).

    Rules: keys further than ``window`` seconds away are hidden; an invalid key
    frame is hidden from every query except its own diagonal, so no softmax row
    is ever fully masked. ``None`` is returned when neither rule bites, which
    lets the caller skip building a mask at all.

    :param seconds: frame times ``[n_clips, T]``.
    :param frame_valid: bool ``[n_clips, T]``, or ``None`` for all-valid.
    :param window: attention half-width in seconds (``None`` = off).
    """
    n_clips, t = seconds.shape
    keep = None
    if window is not None:
        dist = (seconds[:, :, None] - seconds[:, None, :]).abs()
        if bool((dist > window).any()):
            keep = dist <= window                           # [n_clips, T, T]
    if frame_valid is not None and not bool(frame_valid.all()):
        valid_key = frame_valid[:, None, :].expand(n_clips, t, t)
        diag = torch.eye(t, dtype=torch.bool, device=seconds.device)
        valid_keep = valid_key | diag
        keep = valid_keep if keep is None else keep & valid_keep
    if keep is None:
        return None
    # A query must always see itself even inside the window rule.
    return keep | torch.eye(t, dtype=torch.bool, device=seconds.device)


class _RopeBlock(nn.Module):
    """Pre-LN transformer block: RoPE attention + FFN, both residual.

    ``x -> x + Proj(Attn(LN(x)));  x -> x + FFN(LN(x))`` with q/k rotated by
    RoPE before the dot product and both output projections zero-initialised
    (an exact identity at init).
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

        hidden = int(dim * mlp_ratio)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.ffn[3].weight)
        nn.init.zeros_(self.ffn[3].bias)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        token_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Run one block.

        :param x: tokens ``[B, T, C]``.
        :param cos: RoPE table ``[B, 1, T, head_dim]``.
        :param sin: RoPE table ``[B, 1, T, head_dim]``.
        :param attn_mask: bool ``[B, 1, T, T]`` (``True`` = may attend) or
            ``None`` for all-visible.
        :param token_emb: slot identity embedding broadcastable onto
            ``[B, T, C]``, added to the LayerNormed input before the q/k/v
            projection. RoPE encodes *time* only, so a sequence carrying
            several tokens per frame needs this to tell its slots apart. Added
            INSIDE the attention branch, never to the residual stream, so the
            identity at init is preserved exactly.
        """
        b, t, c = x.shape
        normed = self.norm_attn(x) + token_emb
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
        x = x + self.proj_drop(self.proj(attn))
        return x + self.ffn(self.norm_ffn(x))


class CrossModalRopeModule(nn.Module):
    """RoPE self-attention over all modality tokens of a clip, across its frames.

    One sequence per clip of ``T`` frames x ``K`` tokens, frame-major
    (``index = t * K + k``), ``K`` = the total token count of the listed
    modalities concatenated in canonical order (pose, contact, force). Every
    token of a frame gets the SAME RoPE position, so within-frame pairs attend
    un-rotated and across-frame pairs see only their relative offset. A learned
    ``[K, C]`` slot embedding (added to the LayerNormed input inside each
    block's attention branch, never to the residual stream) tells the slots
    apart. The frame keep-mask (window + ``frame_valid``) is expanded to the
    token grid, so all ``K`` tokens of a frame share one visibility row.

    :param dim: token working dim (``DECODER.DIM``).
    :param num_slots: tokens per frame ``K`` (sizes the slot embedding, so it
        is part of the architecture).
    :param num_layers: number of stacked RoPE blocks.
    :param num_heads: attention heads per block (``dim / num_heads`` even).
    :param mlp_ratio: FFN hidden expansion factor.
    :param dropout: dropout inside attention/FFN.
    :param window: attention half-width in seconds (``None`` = whole clip).
    :param time_scale: RoPE rotation units per second (``25`` makes one
        25-fps step a unit position).
    """

    def __init__(
        self,
        dim: int,
        num_slots: int,
        num_layers: int = 4,
        num_heads: int = 16,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        window: Optional[float] = 2.5,
        time_scale: float = 25.0,
    ):
        super().__init__()
        if num_slots <= 0:
            raise ValueError(f"num_slots must be positive; got {num_slots}")
        if window is not None and (not math.isfinite(window) or window <= 0):
            raise ValueError("window must be finite and positive, or None")
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale must be finite and positive")
        self.dim = dim
        self.num_slots = int(num_slots)
        self.window = None if window is None else float(window)
        self.time_scale = float(time_scale)
        self.blocks = nn.ModuleList(
            _RopeBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers))
        self.head_dim = dim // num_heads
        # Small init (ViT positional-embedding convention): slots are
        # distinguishable from step one without swamping the LayerNormed
        # features they are added to.
        self.slot_embed = nn.Parameter(torch.zeros(self.num_slots, dim))
        nn.init.trunc_normal_(self.slot_embed, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        seq_len: int,
        frame_pos_sec: torch.Tensor,
        frame_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mix every modality token with every other across the clip's frames.

        :param tokens: ``[B_flat, K, C]`` with ``B_flat = B_clips * seq_len``
            (clip-major, frame-minor), ``K`` = :attr:`num_slots`, the modality
            blocks already concatenated in canonical order.
        :param seq_len: frames per clip ``T``. ``T = 1`` (still images) is legal
            and degenerates to within-frame cross-modal attention.
        :param frame_pos_sec: elapsed seconds per flattened frame ``[B_flat]``.
        :param frame_valid: per-frame validity ``[B_flat]`` bool.
        :returns: updated tokens ``[B_flat, K, C]``.
        """
        b_flat, num_slots, input_dim = tokens.shape
        if input_dim != self.dim:
            raise AssertionError(
                f"token dim {input_dim} does not match configured dim {self.dim}")
        if num_slots != self.num_slots:
            raise AssertionError(
                f"token count {num_slots} does not match configured num_slots "
                f"{self.num_slots}")
        if b_flat % seq_len != 0:
            raise AssertionError(f"batch {b_flat} not divisible by seq_len {seq_len}")
        n_clips = b_flat // seq_len

        seconds = frame_pos_sec.to(tokens.device).float().view(n_clips, seq_len)
        # Every token of a frame shares that frame's position: within-frame
        # pairs see offset 0 (un-rotated), across-frame pairs their relative offset.
        token_pos = seconds.repeat_interleave(num_slots, dim=1)     # [n_clips, T*K]
        cos, sin = rope_cos_sin(token_pos * self.time_scale, self.head_dim)
        cos = cos.to(tokens.dtype)[:, None]                 # [n_clips, 1, T*K, hd]
        sin = sin.to(tokens.dtype)[:, None]

        valid = None
        if frame_valid is not None:
            valid = frame_valid.to(
                device=tokens.device, dtype=torch.bool).view(n_clips, seq_len)
        mask = frame_keep_mask(seconds, valid, self.window)
        if mask is not None:
            # [n_clips, T, T] -> [n_clips, 1, T*K, T*K]: all K tokens of a frame
            # share one visibility row (frame-major token order).
            mask = mask.repeat_interleave(num_slots, dim=1)
            mask = mask.repeat_interleave(num_slots, dim=2)[:, None]

        slot_emb = self.slot_embed.repeat(seq_len, 1)[None]  # [1, T*K, C]
        x = tokens.reshape(n_clips, seq_len * num_slots, input_dim)
        for block in self.blocks:
            x = block(x, cos, sin, mask, slot_emb)
        return x.reshape(b_flat, num_slots, input_dim)
