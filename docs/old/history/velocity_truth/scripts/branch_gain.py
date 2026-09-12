"""Per-layer branch gain + temporal high-pass transfer of a trained cross-modal RoPE block.

For every ``_RopeBlock`` of ``model.cross_modal_temporal`` this recomputes, on the
static test clips, the two residual branches exactly as the block does and reports:

* ``gain_attn``  RMS(Proj(Attn(LN x)) * gamma)      / RMS(x)          -- the "c" of x + c*avg
* ``gain_ffn``   RMS(FFN(LN x_mid)  * gamma)        / RMS(x_mid)
* ``gain_block`` RMS(out - x)                        / RMS(x)
* ``d3_in`` / ``d3_out``: RMS third temporal difference of the POSE slot's token,
  divided by that slot's own RMS (a scale-free high-frequency level), and their
  ratio ``d3_ratio`` = how much of the token's per-frame white noise the layer removed
  (< 1 = smoothing, > 1 = sharpening).  Same for first differences (``d1_ratio``).

argv: <config> <checkpoint> [--out json]
"""
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/data3/rikhat.akizhanov/better/contact_anything_dev")

from data import build_datasets                                       # noqa: E402
from data.collate import batch_to_device                              # noqa: E402
from data.loaders import build_loaders                                # noqa: E402
from model.rope import rope_rotate, frame_positions, frame_keep_mask, rope_cos_sin  # noqa: E402
from train.config import signal_needs                                 # noqa: E402
from train.predict import load_model                                  # noqa: E402

CFG, CKPT = sys.argv[1], sys.argv[2]
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False

model, cfg = load_model(CFG, CKPT, dev)
module = model.cross_modal_temporal
blocks = list(module.blocks)
K = module.num_slots
L = len(blocks)

_, test_sets = build_datasets(cfg, signal_needs(cfg))
_, loader = build_loaders(cfg, [], test_sets)

mod_in = {}
acc = {k: [0.0, 0.0] for k in
       [f"{n}_{i}" for i in range(L)
        for n in ("attn", "x", "ffn", "xmid", "delta", "d3in", "d3out", "d1in", "d1out",
                  "slot_in", "slot_out", "pattn", "px", "pffn", "pxmid", "pdelta")]}


def add(key, sq, n):
    acc[key][0] += float(sq)
    acc[key][1] += float(n)


def rms(key):
    s, n = acc[key]
    return math.sqrt(s / n) if n else float("nan")


def diff_stats(x, n_clips, T, valid, tag, layer):
    """RMS of the pose slot's temporal 1st/3rd differences and its own RMS."""
    v = x.view(n_clips, T, K, -1)[:, :, 0]                   # [n, T, C] pose slot
    vm = valid.view(n_clips, T)
    for c in range(n_clips):
        idx = torch.nonzero(vm[c], as_tuple=True)[0]
        if idx.numel() < 5:
            continue
        y = v[c, idx].double()
        y = y - y.mean(0, keepdim=True)                       # remove the clip mean
        add(f"slot_{tag}_{layer}", (y ** 2).sum(), y.numel())
        d1 = y[1:] - y[:-1]
        d3 = y[3:] - 3 * y[2:-1] + 3 * y[1:-2] - y[:-3]
        add(f"d1{tag}_{layer}", (d1 ** 2).sum(), d1.numel())
        add(f"d3{tag}_{layer}", (d3 ** 2).sum(), d3.numel())


def mod_pre2(m, args):
    mod_in["x"] = args[0].detach()
    mod_in["seq_len"] = int(args[1])
    mod_in["frame_pos_sec"] = args[2]
    mod_in["frame_valid"] = args[3]


module.register_forward_pre_hook(mod_pre2)

with torch.no_grad():
    for batch in loader:
        batch = batch_to_device(batch, dev)
        model(batch)
        tokens = mod_in["x"]
        T = mod_in["seq_len"]
        fv = mod_in["frame_valid"]
        b_flat = tokens.shape[0]
        n_clips = b_flat // T
        valid = fv.to(device=tokens.device, dtype=torch.bool)
        pos = frame_positions(module.position, mod_in["frame_pos_sec"], n_clips, T,
                              tokens.device)
        token_pos = pos.repeat_interleave(K, dim=1)
        cos, sin = rope_cos_sin(token_pos * module.rope_scale, module.head_dim)
        cos = cos.to(tokens.dtype)[:, None]
        sin = sin.to(tokens.dtype)[:, None]
        fmask = frame_keep_mask(pos, valid.view(n_clips, T), module.window)
        mask = None
        if fmask is not None:
            mask = fmask.repeat_interleave(K, dim=1).repeat_interleave(K, dim=2)[:, None]
        slot_emb = module.slot_embed.repeat(T, 1)[None]
        x = tokens.reshape(n_clips, T * K, tokens.shape[-1])

        for i, blk in enumerate(blocks):
            x_flat = x.reshape(b_flat, K, -1)
            diff_stats(x_flat, n_clips, T, valid, "in", i)
            normed = blk.norm_attn(x) + slot_emb
            qkv = blk.qkv(normed).reshape(n_clips, T * K, 3, blk.num_heads, blk.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            q = rope_rotate(q, cos, sin)
            k = rope_rotate(k, cos, sin)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            attn = attn.transpose(1, 2).reshape(n_clips, T * K, -1)
            attn = blk.proj(attn)
            if blk.gamma_attn is not None:
                attn = blk.gamma_attn * attn
            add(f"x_{i}", (x.double() ** 2).sum(), x.numel())
            add(f"attn_{i}", (attn.double() ** 2).sum(), attn.numel())
            px = x.reshape(n_clips, T, K, -1)[:, :, 0]
            pa = attn.reshape(n_clips, T, K, -1)[:, :, 0]
            add(f"px_{i}", (px.double() ** 2).sum(), px.numel())
            add(f"pattn_{i}", (pa.double() ** 2).sum(), pa.numel())
            x_mid = x + attn
            ffn = blk.ffn(blk.norm_ffn(x_mid))
            if blk.gamma_ffn is not None:
                ffn = blk.gamma_ffn * ffn
            add(f"xmid_{i}", (x_mid.double() ** 2).sum(), x_mid.numel())
            add(f"ffn_{i}", (ffn.double() ** 2).sum(), ffn.numel())
            pm = x_mid.reshape(n_clips, T, K, -1)[:, :, 0]
            pf = ffn.reshape(n_clips, T, K, -1)[:, :, 0]
            add(f"pxmid_{i}", (pm.double() ** 2).sum(), pm.numel())
            add(f"pffn_{i}", (pf.double() ** 2).sum(), pf.numel())
            out = x_mid + ffn
            add(f"delta_{i}", ((out - x).double() ** 2).sum(), out.numel())
            pd = (out - x).reshape(n_clips, T, K, -1)[:, :, 0]
            add(f"pdelta_{i}", (pd.double() ** 2).sum(), pd.numel())
            diff_stats(out.reshape(b_flat, K, -1), n_clips, T, valid, "out", i)
            x = out

rows = []
print(f"\nconfig {CFG}  checkpoint {CKPT}   slots K={K}  layers {L}")
print(f"{'L':>2s} {'gain_attn':>10s} {'gain_ffn':>9s} {'gain_block':>11s} "
      f"{'pose_attn':>10s} {'pose_ffn':>9s} {'pose_blk':>9s} "
      f"{'d1_in':>8s} {'d1_out':>8s} {'d1_ratio':>9s} {'d3_in':>8s} {'d3_out':>8s} {'d3_ratio':>9s}")
for i in range(L):
    ga = rms(f"attn_{i}") / rms(f"x_{i}")
    gf = rms(f"ffn_{i}") / rms(f"xmid_{i}")
    gb = rms(f"delta_{i}") / rms(f"x_{i}")
    d1i = rms(f"d1in_{i}") / rms(f"slot_in_{i}")
    d1o = rms(f"d1out_{i}") / rms(f"slot_out_{i}")
    d3i = rms(f"d3in_{i}") / rms(f"slot_in_{i}")
    d3o = rms(f"d3out_{i}") / rms(f"slot_out_{i}")
    pa_ = rms(f"pattn_{i}") / rms(f"px_{i}")
    pf_ = rms(f"pffn_{i}") / rms(f"pxmid_{i}")
    pb_ = rms(f"pdelta_{i}") / rms(f"px_{i}")
    print(f"{i:2d} {ga:10.4f} {gf:9.4f} {gb:11.4f} {pa_:10.4f} {pf_:9.4f} {pb_:9.4f} "
          f"{d1i:8.4f} {d1o:8.4f} "
          f"{d1o/d1i:9.4f} {d3i:8.4f} {d3o:8.4f} {d3o/d3i:9.4f}")
    rows.append(dict(layer=i, gain_attn=ga, gain_ffn=gf, gain_block=gb,
                     pose_gain_attn=pa_, pose_gain_ffn=pf_, pose_gain_block=pb_,
                     d1_in=d1i, d1_out=d1o, d1_ratio=d1o / d1i,
                     d3_in=d3i, d3_out=d3o, d3_ratio=d3o / d3i,
                     rms_x=rms(f"x_{i}"), rms_attn=rms(f"attn_{i}"),
                     rms_ffn=rms(f"ffn_{i}"), rms_delta=rms(f"delta_{i}")))
tot_d1 = rms(f"d1out_{L-1}") / rms(f"slot_out_{L-1}") / (rms("d1in_0") / rms("slot_in_0"))
tot_d3 = rms(f"d3out_{L-1}") / rms(f"slot_out_{L-1}") / (rms("d3in_0") / rms("slot_in_0"))
print(f"whole block: d1 ratio {tot_d1:.4f}   d3 ratio {tot_d3:.4f}")
if OUT:
    Path(OUT).write_text(json.dumps(
        {"config": CFG, "checkpoint": CKPT, "num_slots": K, "layers": rows,
         "block_d1_ratio": tot_d1, "block_d3_ratio": tot_d3}, indent=2))
    print(f"wrote {OUT}")
