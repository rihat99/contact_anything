"""Temporal attention over contact tokens for clip (multi-frame) training.

:class:`ContactTemporalModule` gives the contact pipeline a temporal receptive
field: a batch of ``B_flat = B_clips * T`` flattened frames is reshaped to
``[B_clips, T, K, C]`` and self-attention is run across the ``T`` frames of each
clip (optionally jointly with the ``K`` contact-token slots). The module is a
**zero-initialised residual add** — every attention/FFN branch is gated by a
per-channel ``gamma = zeros`` parameter, so at initialisation the module is an
exact identity (``torch.equal``-verified). Training moves the gammas off zero.

Only params of this module carry ``contact`` in their dotted name (the model
attribute is ``contact_temporal``), so the contact-only freeze/eval filters in
:mod:`contact.model` pick them up automatically.

Three placements are wired in :class:`sam_3d_body.models.meta_arch.SAM3DBody`:

* ``post_decoder`` / ``between_layers`` — attention over the ``[B, K, C]`` contact
  tokens through a configurable low-dimensional residual adapter
  (:meth:`ContactTemporalModule.forward`).
* ``pre_decoder`` — per-location temporal attention over the decoder image tensor
  through a ``1280 -> 256 -> 1280`` bottleneck (:meth:`forward_image`). The single
  output gate ``img_gamma`` (zero-init) sits downstream of the attention blocks,
  so at init it also zeros the gradient to the inner block gammas: ``img_gamma``
  trains first and unlocks the blocks once it moves off zero (nested-gate warm-up).
  This is inherent to a single "zero-gated residual" and is why ``pre_decoder`` is
  experimental.

Design note: the fork's :class:`LayerScale` wrapper turns ``scale <= 0`` into an
``nn.Identity`` (``transformer.py:163-166,243-249``), which would silently drop
the gate and its gradient. We therefore keep explicit ``nn.Parameter(zeros)``
gates and plain ``nn.MultiheadAttention`` here instead of reusing those wrappers.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def sinusoidal_time_encoding(
    pos_sec: torch.Tensor, dim: int, max_period: float = 10000.0
) -> torch.Tensor:
    """Sinusoidal encoding of continuous timestamps.

    :param pos_sec: elapsed-seconds positions, shape ``[...]`` (any shape).
    :param dim: embedding dimension (must be even).
    :param max_period: longest sinusoid period.
    :returns: encoding of shape ``[..., dim]`` in ``pos_sec``'s dtype/device.
    """
    if dim % 2 != 0:
        raise ValueError(f"time-encoding dim must be even; got {dim}")
    half = dim // 2
    exponents = torch.arange(half, device=pos_sec.device, dtype=torch.float32) / half
    freqs = torch.exp(-math.log(max_period) * exponents)          # [half]
    args = pos_sec.float().unsqueeze(-1) * freqs                  # [..., half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [..., dim]
    return emb.to(pos_sec.dtype)


def frame_visibility(
    seq_len: int, frame_valid: torch.Tensor, causal: bool
) -> torch.Tensor:
    """Boolean frame-level attention-visibility matrix for one clip.

    ``allowed[q, k]`` is ``True`` when query frame ``q`` may attend to key frame
    ``k``. Rules:

    * causal (if requested): ``k <= q``;
    * an **invalid** key frame is hidden from every query...
    * ...except a query may always see its **own** frame (the diagonal), so no
      softmax row is ever fully masked (which would produce NaNs).

    :param seq_len: number of frames ``T``.
    :param frame_valid: bool tensor ``[T]``.
    :param causal: restrict to non-future keys when ``True``.
    :returns: bool ``allowed`` matrix ``[T, T]`` (row = query, col = key).
    """
    device = frame_valid.device
    idx = torch.arange(seq_len, device=device)
    if causal:
        causal_ok = idx[None, :] <= idx[:, None]                  # k <= q
    else:
        causal_ok = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    valid_key = frame_valid.to(torch.bool)[None, :].expand(seq_len, seq_len)
    self_diag = torch.eye(seq_len, dtype=torch.bool, device=device)
    return causal_ok & (valid_key | self_diag)


class _TemporalBlock(nn.Module):
    """Pre-LN transformer block with zero-initialised residual gates.

    ``x -> x + gamma_attn * MHA(LN(x)+pe);  x -> x + gamma_ffn * FFN(LN(x))``.
    Both gammas start at zero, so the block is an exact identity at init.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
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

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run one block.

        :param x: tokens ``[B, L, C]``.
        :param pos_emb: positional encoding ``[B, L, C]``, added to the attention
            query/key branch only (never to the residual stream).
        :param attn_mask: additive/bool attention mask accepted by
            :class:`nn.MultiheadAttention`, or ``None``.
        """
        normed = self.norm_attn(x)
        query = key = normed + pos_emb
        attn_out, _ = self.attn(
            query, key, normed, attn_mask=attn_mask, need_weights=False
        )
        x = x + self.gamma_attn * attn_out
        x = x + self.gamma_ffn * self.ffn(self.norm_ffn(x))
        return x


class ContactTemporalModule(nn.Module):
    """Zero-gated temporal self-attention over contact tokens / image features.

    :param dim: token working dim (``DECODER.DIM``) for ``post_decoder`` /
        ``between_layers``; unused for ``pre_decoder`` blocks.
    :param num_layers: number of stacked :class:`_TemporalBlock` s.
    :param num_heads: attention heads per block.
    :param mlp_ratio: FFN hidden expansion factor.
    :param attend: ``'joint'`` (attend over ``T*K`` tokens per clip) or
        ``'per_token'`` (attend over ``T`` per token-slot).
    :param causal: default causal setting for :meth:`forward`.
    :param dropout: dropout inside attention/FFN (default 0.0).
    :param placement: ``post_decoder`` | ``between_layers`` | ``pre_decoder``.
    :param image_dim: backbone feature dim; required for ``pre_decoder``.
    :param bottleneck_dim: attention width. Token placements project
        ``dim -> bottleneck_dim -> dim`` and add only the temporal delta; the
        image placement uses the same width for its image-feature bottleneck.
    :param position_scale: multiplier applied to elapsed-second timestamps before
        sinusoidal encoding. ``30`` turns 30-fps timestamps into frame offsets.
    :param window_frames: optional odd ``>= 3`` centered attention window. When set
        and a clip is longer, :meth:`forward` attends only the central
        ``window_frames`` frames (positions re-zeroed to the window start), letting
        a checkpoint trained at ``T = window_frames`` run inside a longer clip while
        seeing exactly its native window. ``None`` disables windowing (attend the
        whole clip). Unsupported with ``pre_decoder`` (:meth:`forward_image`).
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attend: str = "joint",
        causal: bool = False,
        dropout: float = 0.0,
        placement: str = "post_decoder",
        image_dim: Optional[int] = None,
        bottleneck_dim: Optional[int] = None,
        position_scale: float = 1.0,
        window_frames: Optional[int] = None,
    ):
        super().__init__()
        if attend not in ("joint", "per_token"):
            raise ValueError(f"attend must be 'joint' or 'per_token'; got {attend!r}")
        if placement not in ("post_decoder", "between_layers", "pre_decoder"):
            raise ValueError(f"unknown temporal placement {placement!r}")
        if window_frames is not None:
            if int(window_frames) < 3 or int(window_frames) % 2 == 0:
                raise ValueError(
                    f"window_frames must be an odd int >= 3; got {window_frames}")
            if placement == "pre_decoder":
                raise ValueError(
                    "window_frames is unsupported with placement='pre_decoder' "
                    "(forward_image has no windowing path)")

        self.dim = dim
        self.attend = attend
        self.causal = bool(causal)
        self.placement = placement
        self.window_frames = None if window_frames is None else int(window_frames)
        self.position_scale = float(position_scale)
        if not math.isfinite(self.position_scale) or self.position_scale <= 0:
            raise ValueError("position_scale must be finite and positive")

        if placement == "pre_decoder":
            if image_dim is None:
                raise ValueError("pre_decoder placement needs image_dim")
            self.image_dim = int(image_dim)
            self.bottleneck_dim = int(256 if bottleneck_dim is None else bottleneck_dim)
            block_dim = self.bottleneck_dim
            self.img_in_proj = nn.Linear(self.image_dim, block_dim)
            self.img_out_proj = nn.Linear(block_dim, self.image_dim)
            # img_gamma is the single zero-init gate: the private image copy equals
            # the shared tensor at init (identity residual) regardless of the
            # projection weights, mirroring the gamma_attn/gamma_ffn token gates.
            self.img_gamma = nn.Parameter(torch.zeros(self.image_dim))
        else:
            self.bottleneck_dim = int(dim if bottleneck_dim is None else bottleneck_dim)
            block_dim = self.bottleneck_dim
            if block_dim != dim:
                self.token_in_proj = nn.Linear(dim, block_dim)
                # Bias-free so a zero temporal delta maps to an exact zero and the
                # complete adapter remains bitwise identity at initialization.
                self.token_out_proj = nn.Linear(block_dim, dim, bias=False)

        self.blocks = nn.ModuleList(
            _TemporalBlock(block_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        )

    # ------------------------------------------------------------------ masks/PE

    def _attn_mask(
        self,
        seq_len: int,
        frame_valid: Optional[torch.Tensor],
        n_clips: int,
        num_heads: int,
        per_slot: int,
        causal: bool,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Build a per-clip boolean attention mask, or ``None`` when unneeded.

        :param per_slot: ``K`` for ``attend='joint'`` (token-level expansion),
            or ``1`` for ``attend='per_token'`` (frame-level only).
        :returns: bool mask ``[n_eff * num_heads, L, L]`` where ``True`` = blocked,
            with ``L = seq_len * per_slot``; ``None`` if all-visible.
        """
        if frame_valid is None:
            valid = torch.ones(n_clips, seq_len, dtype=torch.bool, device=device)
        else:
            valid = frame_valid.to(device=device, dtype=torch.bool).view(n_clips, seq_len)

        if not causal and bool(valid.all()):
            return None                                   # nothing to mask

        # Per-clip frame-level visibility -> token-level (kron with per_slot).
        allowed = torch.stack(
            [frame_visibility(seq_len, valid[c], causal) for c in range(n_clips)], dim=0
        )                                                 # [n_clips, T, T]
        if per_slot > 1:
            # attend='joint': expand frames -> tokens (each frame owns K slots).
            allowed = allowed.repeat_interleave(per_slot, dim=1).repeat_interleave(
                per_slot, dim=2
            )                                             # [n_clips, T*K, T*K]
        # per_token leaves the frame-level [n_clips, T, T] mask; the caller tiles
        # it across the K token-slots.
        blocked = ~allowed                                # True = blocked
        return blocked.repeat_interleave(num_heads, dim=0)  # [n_clips*H, L, L]

    def _pos_emb(
        self,
        pos_sec: Optional[torch.Tensor],
        seq_len: int,
        n_clips: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Per-frame sinusoidal encoding ``[n_clips, T, dim]``.

        A lone frame (``seq_len == 1``) has no temporal position, so the encoding
        is exactly zero. ``pos_sec`` (elapsed seconds, relative to window start)
        keeps the encoding stride/fps aware; ``position_scale`` controls how
        strongly nearby frames are separated.
        """
        if seq_len == 1 or pos_sec is None:
            return torch.zeros(n_clips, seq_len, dim, device=device, dtype=dtype)
        pos = pos_sec.to(device=device).view(n_clips, seq_len) * self.position_scale
        return sinusoidal_time_encoding(pos, dim).to(dtype)

    # ------------------------------------------------------------------ forwards

    def forward(
        self,
        tokens: torch.Tensor,
        seq_len: int,
        frame_pos_sec: Optional[torch.Tensor] = None,
        frame_valid: Optional[torch.Tensor] = None,
        attend: Optional[str] = None,
        causal: Optional[bool] = None,
    ) -> torch.Tensor:
        """Temporal self-attention over contact tokens.

        :param tokens: contact tokens ``[B_flat, K, C]`` with
            ``B_flat = B_clips * seq_len`` (clip-major, frame-minor order).
        :param seq_len: frames per clip ``T``.
        :param frame_pos_sec: elapsed seconds per flattened frame ``[B_flat]``.
        :param frame_valid: per-frame validity ``[B_flat]`` bool.
        :param attend: override the configured attend mode for this call.
        :param causal: override the configured causal flag for this call.
        :returns: updated tokens ``[B_flat, K, C]``.
        """
        attend = self.attend if attend is None else attend
        causal = self.causal if causal is None else causal

        b_flat, K, input_dim = tokens.shape
        if input_dim != self.dim:
            raise AssertionError(
                f"token dim {input_dim} does not match configured dim {self.dim}")
        if b_flat % seq_len != 0:
            raise AssertionError(
                f"batch {b_flat} not divisible by seq_len {seq_len}")
        n_clips = b_flat // seq_len

        # --- centered attention window (e.g. a T=5-native checkpoint in T=7) ---
        # Attend only the central window; frames outside pass through unchanged.
        if self.window_frames is not None and seq_len > self.window_frames:
            w = self.window_frames
            if (seq_len - w) % 2 != 0:
                raise AssertionError(
                    f"window_frames {w} must be exactly centered in seq_len "
                    f"{seq_len} (seq_len - window_frames must be even)")
            lo = (seq_len - w) // 2
            clips = tokens.reshape(n_clips, seq_len, K, input_dim)
            win_tokens = clips[:, lo:lo + w].reshape(n_clips * w, K, input_dim)
            if frame_pos_sec is not None:
                pos = frame_pos_sec.view(n_clips, seq_len)[:, lo:lo + w]
                # Re-zero to the window's first frame: frame_pos_sec is elapsed
                # seconds from the clip's first frame, but the windowed weights
                # were trained on windows starting at zero.
                win_pos = (pos - pos[:, :1]).reshape(n_clips * w)
            else:
                win_pos = None
            if frame_valid is not None:
                win_valid = frame_valid.view(n_clips, seq_len)[:, lo:lo + w].reshape(
                    n_clips * w)
            else:
                win_valid = None
            # Recurse: window == seq_len inside, so this branch is inert there.
            win_out = self.forward(
                win_tokens, w, win_pos, win_valid, attend=attend, causal=causal
            ).reshape(n_clips, w, K, input_dim)
            return torch.cat(
                [clips[:, :lo], win_out, clips[:, lo + w:]], dim=1
            ).reshape(b_flat, K, input_dim)
        # --- end centered attention window ---

        residual = tokens
        working = self.token_in_proj(tokens) if hasattr(self, "token_in_proj") else tokens
        C = working.shape[-1]
        device, dtype = working.device, working.dtype
        num_heads = self.blocks[0].attn.num_heads

        clips = working.view(n_clips, seq_len, K, C)
        frame_pe = self._pos_emb(frame_pos_sec, seq_len, n_clips, C, device, dtype)

        if attend == "joint":
            x = clips.reshape(n_clips, seq_len * K, C)
            pe = frame_pe.unsqueeze(2).expand(-1, -1, K, -1).reshape(
                n_clips, seq_len * K, C)
            mask = self._attn_mask(
                seq_len, frame_valid, n_clips, num_heads, K, causal, device)
        else:  # per_token: [n_clips*K, T, C]
            x = clips.permute(0, 2, 1, 3).reshape(n_clips * K, seq_len, C)
            pe = frame_pe.unsqueeze(1).expand(-1, K, -1, -1).reshape(
                n_clips * K, seq_len, C)
            mask = self._attn_mask(
                seq_len, frame_valid, n_clips, num_heads, 1, causal, device)
            if mask is not None:
                # Reuse each clip's [T,T] mask across its K token-slots.
                mask = mask.view(n_clips, num_heads, seq_len, seq_len)
                mask = mask.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(
                    n_clips * K * num_heads, seq_len, seq_len)

        for block in self.blocks:
            x = block(x, pe, mask)

        if attend == "joint":
            out = x.reshape(n_clips, seq_len, K, C)
        else:
            out = x.reshape(n_clips, K, seq_len, C).permute(0, 2, 1, 3)
        out = out.reshape(b_flat, K, C)
        if hasattr(self, "token_out_proj"):
            # The temporal blocks are exact identities while their gammas are
            # zero. Project only their delta, not the spatial token itself.
            return residual + self.token_out_proj(out - working)
        return out

    def forward_image(
        self,
        image_embeddings: torch.Tensor,
        seq_len: int,
        frame_pos_sec: Optional[torch.Tensor] = None,
        frame_valid: Optional[torch.Tensor] = None,
        causal: Optional[bool] = None,
    ) -> torch.Tensor:
        """Per-location temporal attention over the decoder image tensor.

        Bottlenecks ``image_dim -> bottleneck_dim``, runs per-location attention
        over the ``T`` frames of each clip, projects back and adds a zero-gated
        residual. Returns a **new** tensor (the caller keeps the original for the
        decoder cross-attention).

        :param image_embeddings: ``[B_flat, image_dim, H, W]``.
        :returns: private image copy ``[B_flat, image_dim, H, W]``.
        """
        if self.placement != "pre_decoder":
            raise RuntimeError("forward_image is only valid for pre_decoder placement")
        causal = self.causal if causal is None else causal

        b_flat, Cimg, H, W = image_embeddings.shape
        if b_flat % seq_len != 0:
            raise AssertionError(
                f"batch {b_flat} not divisible by seq_len {seq_len}")
        n_clips = b_flat // seq_len
        HW = H * W
        device, dtype = image_embeddings.device, image_embeddings.dtype
        num_heads = self.blocks[0].attn.num_heads
        bdim = self.bottleneck_dim

        # [B_flat, Cimg, H, W] -> [B_flat, HW, Cimg] -> bottleneck
        feats = image_embeddings.flatten(2).transpose(1, 2)          # [B_flat, HW, Cimg]
        feats = self.img_in_proj(feats)                 # [B_flat, HW, bdim]

        # -> per-location sequences [n_clips*HW, T, bdim]
        feats = feats.view(n_clips, seq_len, HW, bdim).permute(0, 2, 1, 3)
        x = feats.reshape(n_clips * HW, seq_len, bdim)

        frame_pe = self._pos_emb(frame_pos_sec, seq_len, n_clips, bdim, device, dtype)
        pe = frame_pe.unsqueeze(1).expand(-1, HW, -1, -1).reshape(
            n_clips * HW, seq_len, bdim)

        mask = self._attn_mask(seq_len, frame_valid, n_clips, num_heads, 1, causal, device)
        if mask is not None:
            mask = mask.view(n_clips, num_heads, seq_len, seq_len)
            mask = mask.unsqueeze(1).expand(-1, HW, -1, -1, -1).reshape(
                n_clips * HW * num_heads, seq_len, seq_len)

        for block in self.blocks:
            x = block(x, pe, mask)

        # back to [B_flat, HW, Cimg]
        x = x.reshape(n_clips, HW, seq_len, bdim).permute(0, 2, 1, 3).reshape(
            b_flat, HW, bdim)
        delta = self.img_out_proj(x)                    # [B_flat, HW, Cimg]
        delta = (self.img_gamma * delta).transpose(1, 2).view(
            b_flat, Cimg, H, W)
        return image_embeddings + delta
