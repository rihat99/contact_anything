"""Train-time corruption of the cached frozen-backbone embedding.

Both components perturb one frame independently of its neighbours, so a per-frame
estimate cannot recover the clean features while a cross-frame one can. Corrupting
the SHARED source of a frame's decoder tokens (rather than any one token) is what
closes the within-frame leak: every token of that frame is then consistently
wrong, so a temporal block cannot route around the corruption by reading a clean
same-frame modality. Because the frozen decoder maps the corrupted embedding
through its own weights, the induced token perturbation is on-manifold by
construction — it is what the model would genuinely predict from a different image.

Two stacked components, both applied per frame (each batch row is one frame in the
collator's clip-major layout):

- **Gaussian**, on every frame: additive noise at ``gaussian_alpha`` times the
  per-channel feature std. Always on and therefore undetectable, so the model
  cannot learn to gate its response on a "this frame is corrupted" signal.
- **Patch CutMix**, with probability ``cutmix_prob``: a rectangle of the feature
  grid is replaced by the same region of a frame from ANOTHER CLIP. Real features,
  so the frozen decoder stays in distribution, and the affected frame genuinely
  loses evidence that its neighbours keep. The source is drawn per row and always
  from a different clip: a same-clip source would paste a near-identical
  neighbouring frame, i.e. exactly the content the temporal block is meant to go
  and fetch for itself.

The CutMix rectangle is also applied to ``batch["mask"]``. The SAM3 person mask is
a SECOND clean per-frame input — ``sam3d_body.py`` encodes it to a dense
``[B, C, h, w]`` map and *adds* it onto these features, and it does so for cached
embeddings too. Pasting appearance while leaving the silhouette untouched would
both leave the frame's pose readable from the mask alone and hand the frozen
decoder a combination it has never seen. Note the Gaussian has no mask
counterpart: the always-on component corrupts appearance only.

Train-only: :meth:`scripts.train.Trainer._train_epoch` is the sole caller, so no
evaluation, demo or render path ever sees a corrupted batch.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def augment_batch(batch: dict, cfg: dict, scale: float = 1.0) -> None:
    """Corrupt ``batch["embedding"]`` (and the matching mask region) in place.

    :param batch: a collated training batch on the model's device; reads
        ``"embedding"`` ``[B, C, h, w]``, ``"mask"`` and ``"seq_len"``.
    :param cfg: the resolved ``data.embedding_augment`` config section.
    :param scale: strength multiplier from the anneal schedule (see
        :func:`anneal_scale`); ``0`` leaves the batch untouched.
    """
    if not cfg["enabled"] or scale <= 0.0:
        return
    embedding = batch["embedding"]
    rows = embedding.shape[0]

    # CutMix first: pasting after the Gaussian would carry the source frame's
    # noise into the destination frame, correlating what must stay independent.
    prob = float(cfg["cutmix_prob"]) * scale
    if prob > 0.0:
        source = _cross_clip_source(rows, int(batch["seq_len"]), embedding.device)
        box = _sample_boxes(rows, cfg["cutmix_area"], prob,
                            embedding.shape[-2], embedding.shape[-1],
                            embedding.device)
        batch["embedding"] = _paste(embedding, source, box)
        batch["mask"] = _paste(batch["mask"], source, box)
        embedding = batch["embedding"]

    alpha = float(cfg["gaussian_alpha"]) * scale
    if alpha > 0.0:
        # Per-channel: DINOv3 channel scales differ by orders of magnitude, so a
        # single isotropic sigma would swamp some channels and vanish in others.
        # Upcast because bf16 accumulation over the reduction is lossy; strided
        # so the fp32 temporary stays a quarter of the batch (still ~60k
        # samples/channel at the shipped shape).
        std = embedding[:, :, ::2, ::2].float().std(
            dim=(0, 2, 3), keepdim=True).to(embedding.dtype)
        batch["embedding"] = embedding + (alpha * std) * torch.randn_like(embedding)


def anneal_scale(epoch: int, epochs: int, start_frac: float) -> float:
    """Augmentation strength for ``epoch``: 1.0, then a cosine ramp down to 0.

    Full strength while less than ``start_frac`` of training is done, then a
    cosine anneal reaching exactly 0 in the final epoch, so the model finishes on
    clean batches and inference has no train/test mismatch. ``start_frac = 1.0``
    disables the anneal (constant full strength).

    :param epoch: 0-based epoch index.
    :param epochs: total number of epochs in the run.
    :param start_frac: fraction of training completed before the anneal begins.
    :returns: a multiplier in ``[0, 1]``.
    """
    done = (epoch + 1) / max(epochs, 1)
    if done <= start_frac:
        return 1.0
    ramp = (done - start_frac) / max(1.0 - start_frac, 1e-6)
    return 0.5 * (1.0 + math.cos(math.pi * min(ramp, 1.0)))


def _cross_clip_source(rows: int, seq_len: int, device) -> Tensor:
    """Per-row source index, drawn uniformly over every row of the OTHER clips.

    Uniform over rows rather than over clips: matching the destination's in-clip
    index too would leave only ``clips - 1`` candidates per row (3 at the shipped
    240/60 shape) instead of ``rows - seq_len``.
    """
    if rows <= seq_len:
        raise RuntimeError(
            f"cutmix needs >= 2 clips per batch to paste across clips; got "
            f"{rows} rows at seq_len={seq_len} (check data.frames_per_batch "
            "against data.sequence.frames_per_clip)")
    clip_start = (torch.arange(rows, device=device) // seq_len) * seq_len
    # Draw over the rows outside this row's own clip, then step over that clip.
    drawn = torch.randint(0, rows - seq_len, (rows,), device=device)
    return drawn + seq_len * (drawn >= clip_start).long()


def _sample_boxes(rows: int, area: list, prob: float,
                  num_y: int, num_x: int, device) -> tuple:
    """Per-frame rectangles as ``(top, left, height, width)`` fractions of the frame.

    Expressed as fractions so one box can be rasterised onto the feature grid and
    onto the much larger mask, and snapped to feature-cell edges so the two cover
    exactly the same cells rather than rounding apart at the seam. Frames not
    selected by ``prob`` get a zero-size box.
    """
    frac = torch.empty(rows, device=device).uniform_(float(area[0]), float(area[1]))
    frac *= (torch.rand(rows, device=device) < prob)
    aspect = torch.empty(rows, device=device).uniform_(
        -math.log(2.0), math.log(2.0)).exp()          # log-uniform in [1/2, 2]
    # A selected row must get at least one cell: rounding a small area down to
    # zero would consume the draw and paste nothing. Unselected rows keep the
    # zero-size box that encodes "not selected".
    height = ((frac * aspect).sqrt() * num_y).round().clamp(max=num_y)
    width = ((frac / aspect).sqrt() * num_x).round().clamp(max=num_x)
    selected = frac > 0
    height = torch.where(selected, height.clamp(min=1), height)
    width = torch.where(selected, width.clamp(min=1), width)
    top = (torch.rand(rows, device=device) * (num_y - height + 1)).floor()
    left = (torch.rand(rows, device=device) * (num_x - width + 1)).floor()
    return top / num_y, left / num_x, height / num_y, width / num_x


def _paste(x: Tensor, source: Tensor, box: tuple) -> Tensor:
    """``x`` with each row's ``box`` region replaced by row ``source``'s.

    :param x: ``[rows, ..., H, W]`` — any number of middle dims (``[B, C, h, w]``
        for the embedding, ``[B, 1, 1, H, W]`` for the mask).
    :param source: ``[rows]`` source row indices.
    :param box: the ``(top, left, height, width)`` fractions from
        :func:`_sample_boxes`.
    """
    top, left, height, width = box
    rows, num_y, num_x = x.shape[0], x.shape[-2], x.shape[-1]
    # Cell centres, so a box covers the same fraction of either resolution.
    ys = ((torch.arange(num_y, device=x.device) + 0.5) / num_y).view(1, num_y)
    xs = ((torch.arange(num_x, device=x.device) + 0.5) / num_x).view(1, num_x)
    inside = (
        ((ys >= top.view(-1, 1)) & (ys < (top + height).view(-1, 1))).view(rows, num_y, 1)
        & ((xs >= left.view(-1, 1)) & (xs < (left + width).view(-1, 1))).view(rows, 1, num_x)
    )
    shape = (rows,) + (1,) * (x.dim() - 3) + (num_y, num_x)
    return torch.where(inside.view(shape), x[source], x)
