"""Diagnose whether the post-decoder RoPE temporal block really attends across frames.

    python scripts/diag_temporal.py --config configs/temporal_tokens_b8_lr2.yaml \
        --checkpoint output/temporal_tokens_b8_lr2_20260902_203707/best.pth

``model.cross_modal_temporal`` (:class:`model.rope.CrossModalRopeModule`) runs one
sequence of ``T`` frames x ``K`` tokens per clip, frame-major (``index = t*K + k``),
every token of a frame sharing that frame's RoPE position. It CAN therefore collapse
to within-frame ("diagonal") cross-modal mixing while looking, from the outside, like
a temporal model. This script measures whether it does, on a trained checkpoint, and
how much the temporal context is worth at the output.

Numbers and how to read them
----------------------------

*Attention mass* (per layer, per head, averaged over query tokens and clips; the
attention is recomputed from the block's own inputs, captured with forward
pre-hooks, and verified once against ``F.scaled_dot_product_attention``):

``self``
    mass a query puts on its own token. 1.0 = the block reads nothing at all.
``same_frame``
    mass on all ``K`` tokens of the query's own frame (``dt = 0``, self included).
    ``same_frame ~ 1`` is the COLLAPSED case: the block is a per-frame cross-modal
    mixer with no temporal content.
``cross_frame = 1 - same_frame``
    mass spent on other frames. This is the temporal budget.
``mean |dt|``
    attention-weighted mean absolute time offset, in seconds (from the clip's real
    ``frame_pos_sec``). Small = local smoothing, large = long-range context.
``eff_frames``
    ``exp(H)`` of the per-row entropy over FRAMES (the ``K`` tokens of a frame summed
    first), averaged over rows: how many frames a query effectively reads. 1.0 =
    collapsed, ``~2 * window`` = near-uniform over the whole window.

*Temporal profile*: mean mass as a function of the frame offset ``t' - t`` (clip
steps; the median seconds per step is reported alongside). A collapsed block is a
spike at 0 with nothing beside it; a temporal block has a decaying skirt.

*Slot mixing* ``[K, K]``: mass from query slot ``k`` to key slot ``k'``, summed over
ALL frames. Off-diagonal mass in row ``pose`` means the pose token reads contact
tokens, and vice versa.

*Gate norms*: ``||gamma_attn||`` / ``||gamma_ffn||`` per layer. The gates are
zero-initialised, so they are the direct "does this block write anything at all"
measure — a layer with a ~0 gate is inert whatever its attention looks like.

*Output ablations* (the decisive functional test, full test protocol):

``full``
    the model as trained.
``same_frame``
    ``max_rel_sec`` set to 1e-4, so the frame keep-mask hides every frame but the
    query's own: within-frame cross-modal mixing survives, temporal context is gone.
``bypass``
    every ``gamma`` zeroed: the block is an exact identity and the heads read the raw
    decoder tokens.

``full`` == ``same_frame`` means the temporal axis buys nothing; ``same_frame`` ==
``bypass`` means the block as a whole buys nothing.

*Per-clip sensitivity*: on the attention subset, how far the outputs actually move
between ``full`` and ``same_frame`` (contact probability delta; SMPL-X ``joints_cam``
displacement in mm, raw and hip-aligned).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path
from typing import Iterator, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt                                     # noqa: E402
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import torch.nn.functional as F                                     # noqa: E402
from tqdm import tqdm                                               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import build_datasets                                     # noqa: E402
from data.collate import batch_to_device                            # noqa: E402
from data.loaders import build_loaders                              # noqa: E402
from model.loss import KINDYN_GROUP_NAMES, build_losses             # noqa: E402
from model.loss.contact import CURVE_THRESHOLDS, THRESHOLD          # noqa: E402
from model.loss.smplx import SMPLX_HIPS                             # noqa: E402
from model.rope import rope_rotate                                  # noqa: E402
from train.config import signal_needs                               # noqa: E402
from train.predict import load_model                                # noqa: E402
from train.trainer import evaluate_losses                           # noqa: E402

#: Short labels for the six kindyn contact groups, in corpus order.
GROUP_SHORT = {"left_hand": "LH", "right_hand": "RH", "left_foot": "LF",
               "right_foot": "RF", "left_ankle": "LA", "right_ankle": "RA"}
#: ``max_rel_sec`` of the ``same_frame`` arm — below any real inter-frame spacing.
SAME_FRAME_SEC = 1e-4
_EPS = 1e-12


@contextlib.contextmanager
def exact_matmul() -> Iterator[None]:
    """Disable TF32 matmul: the attention recompute is checked against SDPA at fp32.

    TF32 costs ~3 decimal digits in the ``q @ k^T`` product, which shows up as a
    ~1e-2 disagreement with the block's own SDPA call — precision, not a
    different formula. The model forward itself keeps whatever the caller set,
    so the ablation metrics stay comparable with ``scripts/evaluate.py``.
    """
    saved = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved


def slot_labels(net) -> list[str]:
    """Per-slot names of the concatenated cross-modal token sequence."""
    labels: list[str] = []
    for modality in net.cross_modal_modalities:
        if modality == "pose":
            labels.append("pose")
        elif modality == "contact":
            count = net.contact_tokens.num_tokens
            labels += ([GROUP_SHORT[g] for g in KINDYN_GROUP_NAMES]
                       if count == len(KINDYN_GROUP_NAMES)
                       else [f"c{i}" for i in range(count)])
        elif modality == "force":
            labels += [f"f{i}" for i in range(net.force_tokens.num_tokens)]
        else:
            labels += [f"m{i}" for i in range(net.motion_tokens.num_tokens)]
    return labels


class BlockCapture:
    """Forward pre-hook storing every ``_RopeBlock`` call's positional arguments.

    One instance is registered on every block, so ``self.args`` fills in layer
    order (the blocks run sequentially inside one module forward).
    """

    def __init__(self) -> None:
        self.enabled = False
        self.args: list[tuple] = []

    def __call__(self, module, args: tuple) -> None:
        if self.enabled:
            self.args.append(args)


def block_attention(
    block, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    mask: Optional[torch.Tensor], token_emb: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute one block's attention probabilities exactly as ``_RopeBlock`` does.

    :returns: ``(probs [B, H, N, N], v [B, H, N, hd])`` in float32.
    """
    b, n, _ = x.shape
    normed = block.norm_attn(x)
    if token_emb is not None:
        normed = normed + token_emb
    qkv = block.qkv(normed).reshape(b, n, 3, block.num_heads, block.head_dim)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)          # each [B, H, N, hd]
    q = rope_rotate(q, cos, sin).float()
    k = rope_rotate(k, cos, sin).float()
    logits = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(block.head_dim))
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    return logits.softmax(dim=-1), v.float()


def verify_attention(
    block, args: tuple, probs: torch.Tensor, v: torch.Tensor,
) -> float:
    """Max abs deviation of ``probs @ v`` from the block's own SDPA attention."""
    x, cos, sin, mask, token_emb = args
    normed = block.norm_attn(x)
    if token_emb is not None:
        normed = normed + token_emb
    qkv = block.qkv(normed).reshape(x.shape[0], x.shape[1], 3,
                                    block.num_heads, block.head_dim)
    q, k, _ = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    q = rope_rotate(q, cos, sin)
    k = rope_rotate(k, cos, sin)
    reference = F.scaled_dot_product_attention(q, k, v.to(q.dtype), attn_mask=mask)
    return float((probs.to(v.dtype) @ v - reference).abs().max())


class AttentionStats:
    """Float64 CPU accumulators for the per-layer / per-head attention statistics."""

    def __init__(self, num_layers: int, num_heads: int, num_slots: int,
                 max_offset: int) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_slots = num_slots
        self.max_offset = max_offset
        width = 2 * max_offset + 1
        zeros_lh = torch.zeros(num_layers, num_heads, dtype=torch.float64)
        self.rows = torch.zeros(num_layers, dtype=torch.float64)
        self.self_sum = zeros_lh.clone()
        self.same_sum = zeros_lh.clone()
        self.wdt_sum = zeros_lh.clone()
        self.ent_sum = zeros_lh.clone()
        self.eff_sum = zeros_lh.clone()
        self.prof_num = torch.zeros(num_layers, width, dtype=torch.float64)
        self.prof_den = torch.zeros(num_layers, width, dtype=torch.float64)
        self.dt_num = torch.zeros(width, dtype=torch.float64)
        self.dt_den = torch.zeros(width, dtype=torch.float64)
        self.slot_num = torch.zeros(num_layers, num_slots, num_slots,
                                    dtype=torch.float64)
        self.slot_den = torch.zeros(num_layers, dtype=torch.float64)
        self.clips = 0
        self.frames = 0

    def add_clip_time(self, pos_sec: torch.Tensor, valid: torch.Tensor) -> None:
        """Accumulate the seconds-per-offset table of one clip (layer independent)."""
        t = pos_sec.shape[0]
        offset = (torch.arange(t)[None, :] - torch.arange(t)[:, None]) + self.max_offset
        dt = (pos_sec[None, :] - pos_sec[:, None]).abs().to(torch.float64)
        keep = valid[:, None].expand(t, t)
        self.dt_num.index_add_(0, offset[keep], dt[keep])
        self.dt_den.index_add_(0, offset[keep], torch.ones(int(keep.sum()),
                                                           dtype=torch.float64))
        self.clips += 1
        self.frames += int(valid.sum())

    def add_layer(self, layer: int, probs: torch.Tensor, pos_sec: torch.Tensor,
                  valid: torch.Tensor) -> None:
        """Accumulate one clip's statistics for one layer.

        :param probs: attention probabilities ``[H, N, N]`` (float32, on device).
        :param pos_sec: clip frame positions ``[T]`` seconds (CPU).
        :param valid: per-frame validity ``[T]`` bool (CPU).
        """
        heads, n, _ = probs.shape
        slots = self.num_slots
        t = n // slots
        device = probs.device
        frame_of_row = torch.arange(n, device=device) // slots
        row_keep = valid.to(device).repeat_interleave(slots)             # [N]
        keep_f = row_keep.to(torch.float32)

        by_frame = probs.reshape(heads, n, t, slots).sum(-1)             # [H, N, T]
        self_mass = probs[:, torch.arange(n, device=device),
                          torch.arange(n, device=device)]                # [H, N]
        same_mass = by_frame[:, torch.arange(n, device=device), frame_of_row]
        dt_abs = (pos_sec[None, :] - pos_sec[:, None]).abs().to(device)  # [T, T]
        weighted_dt = (by_frame * dt_abs[frame_of_row][None]).sum(-1)    # [H, N]
        entropy = -(by_frame * (by_frame + _EPS).log()).sum(-1)          # [H, N]

        self.rows[layer] += float(keep_f.sum())
        self.self_sum[layer] += (self_mass * keep_f).sum(1).double().cpu()
        self.same_sum[layer] += (same_mass * keep_f).sum(1).double().cpu()
        self.wdt_sum[layer] += (weighted_dt * keep_f).sum(1).double().cpu()
        self.ent_sum[layer] += (entropy * keep_f).sum(1).double().cpu()
        self.eff_sum[layer] += (entropy.exp() * keep_f).sum(1).double().cpu()

        # Head-averaged frame x frame profile (query slots averaged) and slot mixing.
        mean_probs = probs.mean(0)                                       # [N, N]
        frame_frame = (mean_probs.reshape(n, t, slots).sum(-1)
                       .reshape(t, slots, t).mean(1)).double().cpu()     # [T, T]
        offset = (torch.arange(t)[None, :] - torch.arange(t)[:, None]) + self.max_offset
        keep = valid[:, None].expand(t, t)
        self.prof_num[layer].index_add_(0, offset[keep], frame_frame[keep])
        self.prof_den[layer].index_add_(
            0, offset[keep], torch.ones(int(keep.sum()), dtype=torch.float64))

        slot_mix = mean_probs.reshape(t, slots, t, slots).sum(2).double().cpu()
        self.slot_num[layer] += slot_mix[valid].sum(0)
        self.slot_den[layer] += float(valid.sum())

    def summary(self) -> dict:
        """Reduce the accumulators to the reported numbers."""
        rows = self.rows.clamp_min(1.0)[:, None]
        self_mass = (self.self_sum / rows).tolist()
        same_mass = (self.same_sum / rows).tolist()
        mean_dt = (self.wdt_sum / rows).tolist()
        entropy = (self.ent_sum / rows).tolist()
        eff_frames = (self.eff_sum / rows).tolist()
        offsets = list(range(-self.max_offset, self.max_offset + 1))
        seen = (self.prof_den[0] > 0).tolist()
        profile = {
            "offsets": [o for o, s in zip(offsets, seen) if s],
            "count": [c for c, s in zip(self.prof_den[0].tolist(), seen) if s],
            "sec": [n / max(d, 1.0) for n, d, s
                    in zip(self.dt_num.tolist(), self.dt_den.tolist(), seen) if s],
            "mass": [
                [n / max(d, 1.0) for n, d, s
                 in zip(self.prof_num[layer].tolist(),
                        self.prof_den[layer].tolist(), seen) if s]
                for layer in range(self.num_layers)],
        }
        slot_den = self.slot_den.clamp_min(1.0)[:, None, None]
        return {
            "rows": self.rows.tolist(),
            "clips": self.clips,
            "frames": self.frames,
            "per_head": {
                "self_mass": self_mass,
                "same_frame_mass": same_mass,
                "cross_frame_mass": [[1.0 - v for v in row] for row in same_mass],
                "mean_abs_dt_sec": mean_dt,
                "frame_entropy_nats": entropy,
                "eff_frames": eff_frames,
            },
            "per_layer": {
                "self_mass": [float(np.mean(r)) for r in self_mass],
                "same_frame_mass": [float(np.mean(r)) for r in same_mass],
                "cross_frame_mass": [1.0 - float(np.mean(r)) for r in same_mass],
                "mean_abs_dt_sec": [float(np.mean(r)) for r in mean_dt],
                "eff_frames": [float(np.mean(r)) for r in eff_frames],
                "same_frame_mass_head_min": [float(np.min(r)) for r in same_mass],
                "same_frame_mass_head_median": [float(np.median(r)) for r in same_mass],
                "same_frame_mass_head_max": [float(np.max(r)) for r in same_mass],
            },
            "temporal_profile": profile,
            "slot_mixing": (self.slot_num / slot_den).tolist(),
        }


def gate_summary(module) -> dict:
    """Zero-init gate magnitudes per layer plus the slot-embedding norms."""
    blocks = list(module.blocks)
    return {
        "gamma_attn_l2": [float(b.gamma_attn.detach().norm()) for b in blocks],
        "gamma_attn_absmax": [float(b.gamma_attn.detach().abs().max()) for b in blocks],
        "gamma_ffn_l2": [float(b.gamma_ffn.detach().norm()) for b in blocks],
        "gamma_ffn_absmax": [float(b.gamma_ffn.detach().abs().max()) for b in blocks],
        "slot_embed_l2": module.slot_embed.detach().norm(dim=-1).tolist(),
    }


def joint_displacement(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor,
                                                                  torch.Tensor]:
    """Per-frame mean joint displacement (mm) between two ``[B, 22, 3]`` sets.

    :returns: ``(raw, hip_aligned)`` — the second with each set's mean-hip origin
        removed first, matching the MPJPE convention.
    """
    raw = (a - b).norm(dim=-1).mean(-1) * 1000.0
    hips = list(SMPLX_HIPS)
    ca = a - a[:, hips].mean(dim=1, keepdim=True)
    cb = b - b[:, hips].mean(dim=1, keepdim=True)
    return raw, (ca - cb).norm(dim=-1).mean(-1) * 1000.0


@torch.no_grad()
def collect_attention(model, loader, device: str, stats: AttentionStats,
                      capture: BlockCapture) -> dict:
    """Run the attention subset: per-layer statistics + full-vs-same_frame outputs.

    Each clip is forwarded twice — once as trained (hooks on) and once with the
    temporal window closed to the query's own frame — so the output sensitivity is
    measured on exactly the clips the attention statistics come from.
    """
    module = model.cross_modal_temporal
    blocks = list(module.blocks)
    trained_window = module.max_rel_sec
    prob_deltas: list[torch.Tensor] = []
    raw_mm: list[torch.Tensor] = []
    aligned_mm: list[torch.Tensor] = []
    verification: list[float] = []

    for batch in tqdm(loader, desc="attention"):
        batch = batch_to_device(batch, device)
        seq_len = int(batch["seq_len"])
        pos_sec = batch["frame_pos_sec"].detach().float().cpu()[:seq_len]
        valid = batch["frame_valid"].detach().bool().cpu()[:seq_len]
        if not bool(valid.any()):
            continue

        capture.args.clear()
        capture.enabled = True
        out_full = model(batch)
        capture.enabled = False
        assert len(capture.args) == len(blocks), (
            f"captured {len(capture.args)} block calls, expected {len(blocks)}")

        stats.add_clip_time(pos_sec, valid)
        with exact_matmul():
            for layer, (block, args) in enumerate(zip(blocks, capture.args)):
                x, cos, sin, mask, token_emb = args
                probs, value = block_attention(block, x, cos, sin, mask, token_emb)
                if not verification:
                    verification.append(verify_attention(block, args, probs, value))
                stats.add_layer(layer, probs[0], pos_sec, valid)
                del probs, value
        capture.args.clear()

        module.max_rel_sec = SAME_FRAME_SEC
        out_same = model(batch)
        module.max_rel_sec = trained_window

        contact_valid = batch["contact_valid"].detach().float().cpu() > 0
        delta = (out_full["contact"]["joint_probs"].detach().float().cpu()
                 - out_same["contact"]["joint_probs"].detach().float().cpu()).abs()
        prob_deltas.append(delta[contact_valid])
        raw, aligned = joint_displacement(
            out_full["smplx"]["joints_cam"].detach().float().cpu(),
            out_same["smplx"]["joints_cam"].detach().float().cpu())
        frame_keep = batch["frame_valid"].detach().bool().cpu()
        raw_mm.append(raw[frame_keep])
        aligned_mm.append(aligned[frame_keep])

    def describe(values: list[torch.Tensor], name: str) -> dict:
        joined = torch.cat(values) if values else torch.zeros(0)
        array = joined.numpy()
        return {
            f"{name}_n": int(array.size),
            f"{name}_mean": float(array.mean()) if array.size else float("nan"),
            f"{name}_p50": float(np.percentile(array, 50)) if array.size else float("nan"),
            f"{name}_p90": float(np.percentile(array, 90)) if array.size else float("nan"),
            f"{name}_max": float(array.max()) if array.size else float("nan"),
        }

    return {
        "sdpa_max_abs_diff": verification[0] if verification else float("nan"),
        **describe(prob_deltas, "contact_prob_abs_delta"),
        **describe(raw_mm, "joints_cam_raw_mm"),
        **describe(aligned_mm, "joints_cam_hip_aligned_mm"),
    }


def run_ablations(model, cfg: dict, device: str, limit_scenes: Optional[int]) -> dict:
    """Score ``full`` / ``same_frame`` / ``bypass`` on the test protocol."""
    _, test_sets = build_datasets(cfg, signal_needs(cfg), limit_scenes=limit_scenes)
    _, loader = build_loaders(cfg, [], test_sets)
    losses = build_losses(cfg, model, device)
    module = model.cross_modal_temporal
    blocks = list(module.blocks)
    saved_window = module.max_rel_sec
    saved_gammas = [(b.gamma_attn.detach().clone(), b.gamma_ffn.detach().clone())
                    for b in blocks]

    results: dict[str, dict] = {}
    for arm in ("full", "same_frame", "bypass"):
        if arm == "same_frame":
            module.max_rel_sec = SAME_FRAME_SEC
        if arm == "bypass":
            for block in blocks:
                block.gamma_attn.data.zero_()
                block.gamma_ffn.data.zero_()
        print(f"\n=== ablation arm: {arm} ===", flush=True)
        results[arm] = evaluate_losses(model, loader, losses, device)
        module.max_rel_sec = saved_window
        for block, (gamma_attn, gamma_ffn) in zip(blocks, saved_gammas):
            block.gamma_attn.data.copy_(gamma_attn)
            block.gamma_ffn.data.copy_(gamma_ffn)

    tags = sorted(t for t in results["full"] if t.startswith("metric_"))
    return {
        "scenes": sum(len(s) for s in test_sets),
        "metrics": {arm: {t: results[arm][t] for t in tags} for arm in results},
        "delta_vs_full": {
            arm: {t: results[arm][t] - results["full"][t] for t in tags}
            for arm in ("same_frame", "bypass")},
    }


def write_figures(out_dir: Path, summary: dict, labels: list[str]) -> list[str]:
    """Write the three diagnostic PNGs; returns their paths."""
    attn = summary["attention"]
    profile = attn["temporal_profile"]
    layers = len(profile["mass"])
    paths: list[str] = []

    figure, axis = plt.subplots(figsize=(9, 5))
    offsets = np.asarray(profile["offsets"])
    mass = np.asarray(profile["mass"])
    floor = max(float(mass[mass > 0].min()) / 10.0, 1e-12) if (mass > 0).any() else 1e-12
    for layer in range(layers):
        axis.plot(offsets, np.maximum(mass[layer], floor),
                  linewidth=1.2, label=f"layer {layer}")
    axis.set_yscale("log")
    axis.set_ylim(bottom=floor)
    axis.set_xlabel("frame offset  t' - t  (clip steps)")
    axis.set_ylabel("mean attention mass on that frame")
    axis.set_title("temporal attention profile (heads and query slots averaged)")
    axis.grid(alpha=0.3)
    axis.legend()
    path = out_dir / "temporal_profile.png"
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    paths.append(str(path))

    mixing = np.asarray(attn["slot_mixing"])
    figure, axes = plt.subplots(1, layers, figsize=(3.4 * layers, 3.8))
    axes = np.atleast_1d(axes)
    for layer in range(layers):
        image = axes[layer].imshow(mixing[layer], cmap="viridis", vmin=0.0)
        for i in range(len(labels)):
            for j in range(len(labels)):
                axes[layer].text(j, i, f"{mixing[layer, i, j]:.2f}", fontsize=5,
                                 ha="center", va="center", color="w")
        axes[layer].set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        axes[layer].set_yticks(range(len(labels)), labels, fontsize=7)
        axes[layer].set_title(f"layer {layer}", fontsize=9)
        axes[layer].set_xlabel("key slot", fontsize=8)
        if layer == 0:
            axes[layer].set_ylabel("query slot", fontsize=8)
        figure.colorbar(image, ax=axes[layer], fraction=0.046)
    figure.suptitle("slot mixing: mass from query slot to key slot (all frames)",
                    fontsize=10)
    path = out_dir / "slot_mixing.png"
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    paths.append(str(path))

    same = np.asarray(attn["per_head"]["same_frame_mass"])
    figure, axis = plt.subplots(figsize=(0.4 * same.shape[1] + 3, 0.5 * layers + 2))
    image = axis.imshow(same, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
    for layer in range(layers):
        for head in range(same.shape[1]):
            axis.text(head, layer, f"{same[layer, head]:.3f}", fontsize=5,
                      ha="center", va="center", color="w")
    axis.set_xlabel("head")
    axis.set_ylabel("layer")
    axis.set_yticks(range(layers), [str(i) for i in range(layers)])
    axis.set_title("same-frame attention mass (1.0 = collapsed to the diagonal)")
    figure.colorbar(image, ax=axis)
    path = out_dir / "same_frame_mass.png"
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    paths.append(str(path))
    return paths


def print_report(summary: dict, labels: list[str]) -> None:
    """Compact human-readable dump of everything in ``summary``."""
    attn = summary["attention"]
    per_layer = attn["per_layer"]
    gates = summary["gates"]
    layers = len(per_layer["self_mass"])

    print("\n" + "=" * 96)
    print(f"clips {attn['clips']}  valid frames {attn['frames']}  "
          f"slots {len(labels)} ({', '.join(labels)})")
    print(f"recomputed attention vs SDPA, max |diff|: "
          f"{summary['sensitivity']['sdpa_max_abs_diff']:.3e}")
    print("\nper layer (heads averaged)")
    print(f"  {'L':>2s} {'self':>7s} {'same_fr':>8s} {'cross_fr':>9s} "
          f"{'|dt| s':>8s} {'eff_fr':>7s} {'sf_min':>7s} {'sf_med':>7s} "
          f"{'sf_max':>7s} {'|g_attn|':>9s} {'|g_ffn|':>8s}")
    for layer in range(layers):
        print(f"  {layer:2d} {per_layer['self_mass'][layer]:7.4f} "
              f"{per_layer['same_frame_mass'][layer]:8.4f} "
              f"{per_layer['cross_frame_mass'][layer]:9.4f} "
              f"{per_layer['mean_abs_dt_sec'][layer]:8.4f} "
              f"{per_layer['eff_frames'][layer]:7.3f} "
              f"{per_layer['same_frame_mass_head_min'][layer]:7.4f} "
              f"{per_layer['same_frame_mass_head_median'][layer]:7.4f} "
              f"{per_layer['same_frame_mass_head_max'][layer]:7.4f} "
              f"{gates['gamma_attn_l2'][layer]:9.4f} "
              f"{gates['gamma_ffn_l2'][layer]:8.4f}")

    profile = attn["temporal_profile"]
    offsets = np.asarray(profile["offsets"])
    print("\ntemporal profile — mass summed over |offset| bands")
    bands = [(0, 0), (1, 1), (2, 5), (6, 20), (21, 10 ** 6)]
    print(f"  {'L':>2s}" + "".join(f" {f'|d|{lo}-{hi}':>13s}" if hi < 10 ** 6
                                   else f" {'|d|>20':>13s}" for lo, hi in bands))
    for layer in range(layers):
        mass = np.asarray(profile["mass"][layer])
        row = ""
        for lo, hi in bands:
            keep = (np.abs(offsets) >= lo) & (np.abs(offsets) <= hi)
            row += f" {mass[keep].sum():13.6f}"
        print(f"  {layer:2d}{row}")
    step = np.asarray(profile["sec"])[offsets == 1]
    print(f"  one clip step = {float(step[0]) if step.size else float('nan'):.4f} s "
          f"(mean over the subset); window max_rel_sec = "
          f"{summary['protocol']['max_rel_sec']}")

    print("\nslot mixing (rows = query slot, columns = key slot, all frames)")
    mixing = np.asarray(attn["slot_mixing"])
    for layer in range(layers):
        print(f"  layer {layer}")
        print("       " + "".join(f"{name:>8s}" for name in labels))
        for i, name in enumerate(labels):
            print(f"  {name:>5s}" + "".join(f"{v:8.4f}" for v in mixing[layer, i]))

    print("\ngates")
    for layer in range(layers):
        print(f"  layer {layer}: |gamma_attn| {gates['gamma_attn_l2'][layer]:.4f} "
              f"(max |.| {gates['gamma_attn_absmax'][layer]:.4f})   "
              f"|gamma_ffn| {gates['gamma_ffn_l2'][layer]:.4f} "
              f"(max |.| {gates['gamma_ffn_absmax'][layer]:.4f})")
    print("  ||slot_embed|| per slot: " + "  ".join(
        f"{name}={value:.4f}" for name, value
        in zip(labels, gates["slot_embed_l2"])))

    sensitivity = summary["sensitivity"]
    print("\nper-clip output sensitivity (full vs same_frame, attention subset)")
    for name, unit in (("contact_prob_abs_delta", ""),
                       ("joints_cam_raw_mm", " mm"),
                       ("joints_cam_hip_aligned_mm", " mm")):
        print(f"  {name:<28s} n {sensitivity[f'{name}_n']:7d}  "
              f"mean {sensitivity[f'{name}_mean']:.5f}{unit}  "
              f"p50 {sensitivity[f'{name}_p50']:.5f}{unit}  "
              f"p90 {sensitivity[f'{name}_p90']:.5f}{unit}  "
              f"max {sensitivity[f'{name}_max']:.5f}{unit}")

    if summary["ablations"] is None:
        print("\nablations: skipped")
        return
    ablations = summary["ablations"]
    tags = sorted(ablations["metrics"]["full"])
    print(f"\noutput ablations on {ablations['scenes']} test scenes")
    print(f"  {'metric':<40s} {'full':>11s} {'same_frame':>11s} {'delta':>10s} "
          f"{'bypass':>11s} {'delta':>10s}")
    for tag in tags:
        print(f"  {tag:<40s} {ablations['metrics']['full'][tag]:11.5f} "
              f"{ablations['metrics']['same_frame'][tag]:11.5f} "
              f"{ablations['delta_vs_full']['same_frame'][tag]:+10.5f} "
              f"{ablations['metrics']['bypass'][tag]:11.5f} "
              f"{ablations['delta_vs_full']['bypass'][tag]:+10.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=None,
                        help="output dir (default: <checkpoint dir>/diag_temporal)")
    parser.add_argument("--attn-scenes", type=int, default=24,
                        help="test scenes used for the attention statistics")
    parser.add_argument("--limit-scenes", type=int, default=None,
                        help="test scenes used for the output ablations (default: all)")
    parser.add_argument("--skip-ablations", action="store_true")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, cfg = load_model(args.config, args.checkpoint, args.device)
    module = model.cross_modal_temporal
    if module is None:
        raise ValueError(f"{args.config}: model.cross_modal_temporal is not enabled")
    labels = slot_labels(model)
    assert len(labels) == module.num_slots, (
        f"{len(labels)} slot labels for {module.num_slots} slots")

    out_dir = (Path(args.out) if args.out is not None
               else Path(args.checkpoint).resolve().parent / "diag_temporal")
    out_dir.mkdir(parents=True, exist_ok=True)

    capture = BlockCapture()
    handles = [block.register_forward_pre_hook(capture) for block in module.blocks]

    _, attn_sets = build_datasets(cfg, signal_needs(cfg),
                                  limit_scenes=args.attn_scenes)
    _, attn_loader = build_loaders(cfg, [], attn_sets)
    stats = AttentionStats(len(module.blocks), module.blocks[0].num_heads,
                           module.num_slots, int(cfg["data"]["eval_max_frames"]) - 1)
    sensitivity = collect_attention(model, attn_loader, args.device, stats, capture)
    for handle in handles:
        handle.remove()

    summary = {
        "protocol": {
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "device": args.device,
            "modalities": list(model.cross_modal_modalities),
            "slots": labels,
            "num_layers": len(module.blocks),
            "num_heads": module.blocks[0].num_heads,
            "head_dim": module.blocks[0].head_dim,
            "time_scale": module.time_scale,
            "max_rel_sec": module.max_rel_sec,
            "same_frame_max_rel_sec": SAME_FRAME_SEC,
            "attn_scenes_requested": args.attn_scenes,
            "attn_scenes": sum(len(s) for s in attn_sets),
            "attn_clips": stats.clips,
            "attn_frames": stats.frames,
            "eval_max_frames": int(cfg["data"]["eval_max_frames"]),
            "clip_frames_train": int(cfg["data"]["clip"]["frames"]),
            "clip_stride": cfg["data"]["clip"]["stride"],
            "contact_threshold": THRESHOLD,
            "curve_thresholds": list(CURVE_THRESHOLDS),
        },
        "gates": gate_summary(module),
        "attention": stats.summary(),
        "sensitivity": sensitivity,
        "ablations": None,
    }
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2))

    if not args.skip_ablations:
        summary["ablations"] = run_ablations(model, cfg, args.device,
                                             args.limit_scenes)
        path.write_text(json.dumps(summary, indent=2))

    figures = write_figures(out_dir, summary, labels)
    print_report(summary, labels)
    print(f"\nwrote {path}")
    for figure in figures:
        print(f"wrote {figure}")


if __name__ == "__main__":
    main()
