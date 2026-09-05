"""Extra per-frame inputs to the pose token, and train-time token corruption.

:class:`KeypointInput` is GVHMR-style early fusion of 2D keypoints: the frame's
detected keypoints, crop-normalised exactly like SAM 3D Body's own
``_full_to_crop`` (``[-0.5, 0.5]`` spans the crop) together with their detector
score, go through a zero-init MLP whose output is ADDED to the pose token
before the cross-modal temporal block. Zero-init keeps the network's initial
function exactly the one without the input. The same module serves the corpus
sapiens keypoints (training) and the frozen readout's own ``pred_keypoints_2d``
(deployment); which one is the caller's choice.

:class:`TokenMasking` is a train-only corruption of the temporal block's input:
contiguous spans of frames have ALL their token slots replaced — by a learned
per-slot mask embedding, or by the same slots of a random frame of another clip
in the batch (``swap``). The losses stay on the corrupted frames, so the block
has to reconstruct them from their neighbours instead of copying each frame's
own decoder token through.

:class:`BboxInput` adds the CLIFF bbox vector (crop centre offset and crop side
over the focal) to the pose token the same way, so the temporal block can see
the crop-to-metric factor the depth lift applies after it.

:class:`FrozenCameraInput` adds the frozen readout's own pelvis ray (bearing
and log depth) the same way — the per-frame depth MEASUREMENT the temporal
block is to denoise.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from utils.geometry import translation_to_ray


class KeypointInput(nn.Module):
    """2D keypoints + scores -> an additive pose-token embedding.

    :param dim: pose-token width (output).
    :param num_keypoints: keypoints per frame ``K``.
    :param min_score: keypoints scoring below are dropped (coords and score
        zeroed); so are keypoints further than half a crop outside it.
    :param noise_std: train-only Gaussian noise on the crop-normalised coords.
    :param drop_prob: train-only per-keypoint drop probability.
    """

    def __init__(
        self,
        dim: int,
        num_keypoints: int,
        min_score: float = 0.3,
        noise_std: float = 0.0,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        self.num_keypoints = int(num_keypoints)
        self.min_score = float(min_score)
        self.noise_std = float(noise_std)
        self.drop_prob = float(drop_prob)
        self.mlp = nn.Sequential(
            nn.Linear(3 * self.num_keypoints, dim), nn.GELU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(
        self,
        pixels: Tensor,
        score: Tensor,
        valid: Tensor,
        affine_trans: Tensor,
        img_size: Tensor,
    ) -> Tensor:
        """Embed one batch of frames.

        :param pixels: ``[B, K, 2]`` full-image pixels.
        :param score: ``[B, K]`` detector confidence in ``[0, 1]``.
        :param valid: ``[B]`` bool — a False frame contributes an all-zero input.
        :param affine_trans: ``[B, 2, 3]`` full-image -> crop affine.
        :param img_size: ``[B, 2]`` crop (W, H) px.
        :returns: ``[B, C]``.
        """
        homog = torch.cat([pixels, torch.ones_like(pixels[..., :1])], dim=-1)
        crop = (homog @ affine_trans.mT) / img_size[:, None] - 0.5         # [B, K, 2]
        keep = (score >= self.min_score) & valid[:, None] & (crop.abs().amax(-1) <= 1.0)
        if self.training and self.noise_std > 0:
            crop = crop + self.noise_std * torch.randn_like(crop)
        if self.training and self.drop_prob > 0:
            keep = keep & (torch.rand_like(score) >= self.drop_prob)
        crop = torch.where(keep[..., None], crop.clamp(-1.0, 1.0), torch.zeros_like(crop))
        score = torch.where(keep, score, torch.zeros_like(score))
        return self.mlp(torch.cat([crop, score[..., None]], dim=-1).flatten(1))


class TokenMasking(nn.Module):
    """Train-time span corruption of whole frames of a ``[B_flat, K, C]`` token sequence.

    Every frame starts a span with probability ``frac / mean span length`` and
    a span runs ``span_min..span_max`` frames (uniform), so about ``frac`` of the
    frames end up corrupted (a little less where spans overlap).

    :param num_slots: tokens per frame ``K``.
    :param dim: token width.
    :param frac: target fraction of corrupted frames.
    :param span_min: shortest span (frames).
    :param span_max: longest span (frames).
    :param replace: ``"mask"`` — a learned ``[K, C]`` embedding; ``"swap"`` —
        the same ``K`` slots of a random frame of another clip in the batch
        (another frame of the same clip when the batch holds one clip).
    """

    def __init__(
        self,
        num_slots: int,
        dim: int,
        frac: float = 0.15,
        span_min: int = 1,
        span_max: int = 5,
        replace: str = "mask",
    ):
        super().__init__()
        if replace not in ("mask", "swap"):
            raise ValueError(f"replace must be 'mask' or 'swap'; got {replace!r}")
        if not 0.0 < float(frac) <= 1.0:
            raise ValueError(f"frac must lie in (0, 1]; got {frac}")
        if not 1 <= int(span_min) <= int(span_max):
            raise ValueError(
                f"need 1 <= span_min <= span_max; got {span_min}, {span_max}")
        self.replace = replace
        self.span_min = int(span_min)
        self.span_max = int(span_max)
        self.p_start = float(frac) / ((self.span_min + self.span_max) / 2.0)
        self.mask_embed = None
        if replace == "mask":
            self.mask_embed = nn.Parameter(torch.zeros(int(num_slots), dim))
            nn.init.trunc_normal_(self.mask_embed, std=0.02)

    def sample(self, n_clips: int, seq_len: int, device) -> Tensor:
        """Corrupted-frame mask ``[n_clips, seq_len]`` (bool)."""
        starts = torch.rand(n_clips, seq_len, device=device) < self.p_start
        lengths = torch.randint(
            self.span_min, self.span_max + 1, (n_clips, seq_len), device=device)
        masked = torch.zeros(n_clips, seq_len, dtype=torch.bool, device=device)
        for offset in range(min(self.span_max, seq_len)):
            live = starts & (lengths > offset)      # spans still running `offset` frames on
            masked[:, offset:] |= live[:, :seq_len - offset]
        return masked

    def _donors(self, n_clips: int, seq_len: int, device) -> Tensor:
        """Flat index of a random frame of another clip, per frame ``[n_clips * seq_len]``."""
        count = n_clips * seq_len
        clip = torch.arange(n_clips, device=device).repeat_interleave(seq_len)
        frame = torch.randint(0, seq_len, (count,), device=device)
        if n_clips > 1:
            other = (clip + torch.randint(1, n_clips, (count,), device=device)) % n_clips
            return other * seq_len + frame
        own = torch.arange(seq_len, device=device).repeat(n_clips)
        if seq_len > 1:
            frame = (own + torch.randint(1, seq_len, (count,), device=device)) % seq_len
        return clip * seq_len + frame

    def forward(self, tokens: Tensor, seq_len: int) -> tuple[Tensor, Tensor]:
        """Corrupt spans of frames.

        :param tokens: ``[B_flat, K, C]`` clip-major, frame-minor.
        :param seq_len: frames per clip ``T``.
        :returns: ``(tokens [B_flat, K, C], masked [B_flat] bool)``.
        """
        b_flat = tokens.shape[0]
        n_clips = b_flat // seq_len
        masked = self.sample(n_clips, seq_len, tokens.device).reshape(-1)
        if self.mask_embed is not None:
            fill = self.mask_embed.to(tokens.dtype)[None].expand(b_flat, -1, -1)
        else:
            fill = tokens[self._donors(n_clips, seq_len, tokens.device)]
        return torch.where(masked[:, None, None], fill, tokens), masked


class BboxInput(nn.Module):
    """The CLIFF bbox vector -> an additive pose-token embedding.

    The pose token encodes the person's CROP-relative scale ``s``; the metric
    depth ``2 f / (b s)`` also needs the crop side ``b`` (full-image px) and
    the focal, which nothing before the SMPL-X head's lift ever sees. Feeding
    ``[(cx - px) / f, (cy - py) / f, log(b / f)]`` to the pose token gives the
    temporal block what it lacks to average DEPTH (not ``s``) across frames.
    Zero-init like :class:`KeypointInput`.

    :param dim: token width (output).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, bbox_center: Tensor, bbox_size: Tensor, cam_int: Tensor) -> Tensor:
        """``bbox_center [B, 2]`` px, ``bbox_size [B]`` px, ``cam_int [B, 3, 3]`` -> ``[B, C]``."""
        focal = cam_int[:, 0, 0]
        offset = (bbox_center - cam_int[:, :2, 2]) / focal[:, None]
        size = torch.log(bbox_size / focal)
        return self.mlp(torch.cat([offset, size[:, None]], dim=-1))


class FrozenCameraInput(nn.Module):
    """The frozen readout's pelvis ray ``[x/z, y/z, log z]`` -> an additive pose-token embedding.

    The frozen SAM 3D Body camera head is a per-frame depth measurement with
    the same white noise the pose token carries (0.55 %/frame on the static
    corpus, 2026-09-04 diagnostic); handing it to the temporal block in the
    bbox-free ray form makes the block's depth job an explicit denoising of a
    measurement sequence. Zero-init like :class:`KeypointInput`.

    :param dim: token width (output).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, pelvis_cam: Tensor) -> Tensor:
        """``pelvis_cam [B, 3]`` camera metres (detached upstream) -> ``[B, C]``."""
        return self.mlp(translation_to_ray(pelvis_cam.float()))


class TwistInput(nn.Module):
    """A 6-vector twist ``[linear, angular]`` -> an additive token embedding.

    Zero-init like :class:`KeypointInput`: the network's initial function is
    exactly the one without the input. Consumes physical units (m/s, rad/s);
    the first linear layer owns the scale.

    :param dim: token width (output).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, twist: Tensor) -> Tensor:
        """``[B, 6] -> [B, C]``."""
        return self.mlp(twist)
