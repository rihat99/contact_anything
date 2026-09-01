"""One RoPE temporal transformer over ALL modality tokens of a clip.

:class:`CrossModalRopeModule` is the single post-decoder mixing brick: it runs
self-attention over the concatenation, across a clip's frames, of every listed
modality's token block (pose, contact, force, motion). It replaces the retired
per-modality sliding-window temporal blocks, the sinusoidal
``cross_modal_temporal`` and the per-frame ``frame_attn`` module at once —
within-frame cross-modal attention is simply the ``dt = 0`` diagonal of joint
attention.

Sequence layout, per clip: ``T`` frames x ``K`` tokens, frame-major
(``index = t * K + k``), where ``K`` is the total token count of the listed
modalities concatenated in canonical order (pose, contact, force, motion).

Positions
    Rotary embedding only, at ``frame_pos_sec * time_scale``. Every token of a
    frame gets the SAME position, so within-frame pairs attend un-rotated and
    across-frame pairs see relative elapsed time — no absolute or sinusoidal
    encoding anywhere, and a model trained at ``T = 60`` runs single-pass on a
    whole scene.

Slot identity
    RoPE carries time only, so a learned ``[K, C]`` embedding indexed by slot
    position tells the concatenated tokens apart. It is added to the
    LayerNormed input *inside* each block's gated attention branch, never to
    the residual stream, so the module stays an exact identity at
    initialisation (every ``gamma`` starts at zero) — the repo's zero-init
    invariant.

Masking
    Frame-level: keys further than ``max_rel_sec`` away are hidden, and an
    invalid frame's tokens are hidden from every other frame (its own frame
    stays visible, so no softmax row is ever empty). The frame mask is expanded
    to the token grid, so all ``K`` tokens of a frame share one visibility row.

The model attribute an instance is bound to (``cross_modal_temporal``) carries
the substring the freeze/eval filters in :mod:`contact.model` match on.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .temporal_rope import _RopeBlock, frame_keep_mask, rope_cos_sin


class CrossModalRopeModule(nn.Module):
    """Zero-gated RoPE self-attention over all modality tokens of a clip.

    :param dim: token working dim (``DECODER.DIM``); blocks run natively at
        this width (no bottleneck adapter).
    :param num_slots: tokens per frame ``K`` = the concatenated width of the
        participating modality blocks. Sizes the learned slot embedding, so it
        is part of the architecture (a different modality list is a different
        module).
    :param num_layers: number of stacked RoPE blocks.
    :param num_heads: attention heads per block (``dim / num_heads`` even).
    :param mlp_ratio: FFN hidden expansion factor.
    :param dropout: dropout inside attention/FFN.
    :param time_scale: multiplier from elapsed seconds to RoPE positions;
        ``25`` calibrates one 25-fps frame step to a unit position.
    :param max_rel_sec: attention window half-width in *seconds* (``None``
        disables). Set it to the training clip span so inference on arbitrarily
        long sequences only ever sees trained relative offsets.
    """

    def __init__(
        self,
        dim: int,
        num_slots: int,
        num_layers: int = 4,
        num_heads: int = 16,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        time_scale: float = 25.0,
        max_rel_sec: Optional[float] = 2.5,
    ):
        super().__init__()
        if num_slots <= 0:
            raise ValueError(f"num_slots must be positive; got {num_slots}")
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale must be finite and positive")
        if max_rel_sec is not None and (
            not math.isfinite(max_rel_sec) or max_rel_sec <= 0
        ):
            raise ValueError("max_rel_sec must be finite and positive, or None")
        self.dim = dim
        self.num_slots = int(num_slots)
        self.time_scale = float(time_scale)
        self.max_rel_sec = None if max_rel_sec is None else float(max_rel_sec)
        self.blocks = nn.ModuleList(
            _RopeBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        )
        self.head_dim = self.blocks[0].head_dim if num_layers > 0 else dim // num_heads
        # Small init (ViT positional-embedding convention): slots are
        # distinguishable from step one without swamping the LayerNormed
        # features they are added to.
        self.slot_embed = nn.Parameter(torch.zeros(self.num_slots, dim))
        nn.init.trunc_normal_(self.slot_embed, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        seq_len: int,
        frame_pos_sec: Optional[torch.Tensor] = None,
        frame_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mix every modality token with every other across the clip's frames.

        :param tokens: ``[B_flat, K, C]`` with ``B_flat = B_clips * seq_len``
            (clip-major, frame-minor), ``K`` = :attr:`num_slots`, the modality
            blocks already concatenated in canonical order.
        :param seq_len: frames per clip ``T``. ``T = 1`` (still images) is legal
            and degenerates to within-frame cross-modal attention.
        :param frame_pos_sec: elapsed seconds per flattened frame ``[B_flat]``;
            ``None`` (single images) falls back to zero positions.
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

        if frame_pos_sec is None:
            pos_sec = torch.zeros(
                n_clips, seq_len, device=tokens.device, dtype=torch.float32)
        else:
            pos_sec = frame_pos_sec.to(tokens.device).float().view(n_clips, seq_len)

        # Every token of a frame shares that frame's position: within-frame
        # pairs see dt = 0 (un-rotated), across-frame pairs see relative time.
        token_pos = pos_sec.repeat_interleave(num_slots, dim=1)   # [n_clips, T*K]
        cos, sin = rope_cos_sin(token_pos * self.time_scale, self.head_dim)
        cos = cos.to(tokens.dtype)[:, None]                 # [n_clips, 1, T*K, hd]
        sin = sin.to(tokens.dtype)[:, None]

        valid = None
        if frame_valid is not None:
            valid = frame_valid.to(
                device=tokens.device, dtype=torch.bool).view(n_clips, seq_len)
        frame_mask = frame_keep_mask(pos_sec, valid, self.max_rel_sec)
        mask = None
        if frame_mask is not None:
            # [n_clips, T, T] -> [n_clips, 1, T*K, T*K]: all K tokens of a frame
            # share one visibility row (frame-major token order).
            mask = frame_mask.repeat_interleave(num_slots, dim=1)
            mask = mask.repeat_interleave(num_slots, dim=2)[:, None]

        slot_emb = self.slot_embed.repeat(seq_len, 1)[None]  # [1, T*K, C]
        x = tokens.reshape(n_clips, seq_len * num_slots, input_dim)
        for block in self.blocks:
            x = block(x, cos, sin, mask, slot_emb)
        return x.reshape(b_flat, num_slots, input_dim)
