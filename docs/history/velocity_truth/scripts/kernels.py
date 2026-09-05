"""Kernel-space toy models.  Every model is LTI, so it is fully described by an
effective kernel of half-width M; the loss is evaluated on the interior only."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import common as C

M = 16                     # max half-width (4 layers x +/-4)
HALF = C.HALF              # 4
DT = C.DT


def apply_k(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x (B,T) -> y (B, T-2M) using kernel w (2M+1,) centred."""
    B, T = x.shape
    y = x.unfold(1, 2 * M + 1, 1)          # (B, T-2M, 2M+1)
    return (y * w).sum(-1)


def targets(g: torch.Tensor):
    G = g[:, M:g.shape[1] - M]
    return G, (G[:, 1:] - G[:, :-1]) / DT


def loss_terms(y, G, dG, kind, d_pf, d_v):
    e = y - G
    dv = (y[:, 1:] - y[:, :-1]) / DT - dG
    if kind == 'l2':
        return (e ** 2).mean(), (dv ** 2).mean()
    return C.huber_norm(e, d_pf).mean(), C.huber_norm(dv, d_v).mean()


def boxcar(dtype=torch.float64):
    u = torch.zeros(2 * M + 1, dtype=dtype)
    u[M - HALF:M + HALF + 1] = 1.0 / (2 * HALF + 1)
    return u


def _emb(v, n):
    """centred kernel (2M+1,) -> circular buffer of length n (lag 0 at index 0)."""
    z = torch.zeros(n - (2 * M + 1), dtype=v.dtype)
    return torch.cat([v[M:], z, v[:M]])


def conv_k(a, b):
    """convolve two centred kernels, keep half-width M (support must fit +/-M)."""
    n = 4 * M + 2
    c = torch.fft.irfft(torch.fft.rfft(_emb(a, n), n=n)
                        * torch.fft.rfft(_emb(b, n), n=n), n=n)
    return torch.cat([c[n - M:], c[:M + 1]])


class ModelA(nn.Module):                    # per-frame affine scale only
    def __init__(self):
        super().__init__(); self.s = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    def kernel(self):
        k = torch.zeros(2 * M + 1, dtype=torch.float64); k = k.clone(); k[M] = 1.0
        return self.s * k


class ModelB(nn.Module):
    """non-residual convex kernel over +/-4, softmax logits (free or self-bias only),
    optional free head scale s."""
    def __init__(self, free_scale=False, self_bias_only=False, init_beta=0.0):
        super().__init__()
        self.self_bias_only = self_bias_only
        if self_bias_only:
            self.beta = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float64))
        else:
            l = torch.zeros(2 * HALF + 1, dtype=torch.float64); l[HALF] = init_beta
            self.logits = nn.Parameter(l)
        self.free_scale = free_scale
        self.s = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    def kernel(self):
        if self.self_bias_only:
            l = torch.zeros(2 * HALF + 1, dtype=torch.float64)
            l = l.clone(); l[HALF] = self.beta
        else:
            l = self.logits
        w = torch.softmax(l, 0)
        k = torch.zeros(2 * M + 1, dtype=torch.float64)
        k = torch.cat([k[:M - HALF], w, k[M + HALF + 1:]])
        return (self.s if self.free_scale else 1.0) * k


class ModelC(nn.Module):
    """residual dilution:  y <- h*(y + c*mean_{+/-4}(y)), L layers.
    norm=True forces h = 1/(1+c) (no free amplitude); else h is learnable."""
    def __init__(self, layers=1, norm=False, c0=0.0, extra_scale=False, linear_c=False):
        super().__init__()
        self.L = layers; self.norm = norm; self.linear_c = linear_c
        inv = 0.0 if linear_c else float(np.log(np.expm1(c0)) if c0 > 0 else -6.0)
        self.craw = nn.Parameter(torch.full((layers,), inv, dtype=torch.float64))
        self.h = nn.Parameter(torch.ones(layers, dtype=torch.float64))
        self.extra_scale = extra_scale
        self.s = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    def kernel(self):
        u = boxcar(); d = torch.zeros(2 * M + 1, dtype=torch.float64)
        d = d.clone(); d[M] = 1.0
        k = d
        for i in range(self.L):
            c = self.craw[i] if self.linear_c else torch.nn.functional.softplus(self.craw[i])
            h = 1.0 / (1.0 + c) if self.norm else self.h[i]
            k = conv_k(k, h * (d + c * u))
        return (self.s if self.extra_scale else 1.0) * k


def train(model, x, g, kind, d_pf, d_v, w_pf, w_vel, lr=2e-4, steps=20000,
          log_every=0, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    G, dG = targets(g)
    hist = []
    for it in range(steps):
        opt.zero_grad()
        y = apply_k(x, model.kernel())
        lp, lv = loss_terms(y, G, dG, kind, d_pf, d_v)
        (w_pf * lp + w_vel * lv).backward()
        opt.step()
        if log_every and it % log_every == 0:
            hist.append((it, float(lp), float(lv)))
    return hist


def report(model, x, g, name):
    with torch.no_grad():
        k = model.kernel().numpy()
    return dict(name=name, **eval_kernel(k, x.numpy(), g.numpy()))


def eval_kernel(k, x, g):
    xt = torch.as_tensor(x); kt = torch.as_tensor(k)
    y = apply_k(xt, kt).numpy()
    G = g[:, M:g.shape[1] - M]
    e = y - G
    dv = (np.diff(y, axis=1) - np.diff(G, axis=1)) / DT
    j = np.arange(-M, M + 1, dtype=float)
    p = np.abs(k) / (np.abs(k).sum() + 1e-30)
    mu = (p * j).sum(); sd = np.sqrt((p * (j - mu) ** 2).sum())
    return dict(dc=float(k.sum()), self=float(k[M]), width_f=float(sd),
                width_s=float(sd * DT), negmass=float(np.clip(-k, 0, None).sum()),
                rmse_pf=float(np.sqrt((e ** 2).mean())),
                rmse_vel=float(np.sqrt((dv ** 2).mean())),
                l2_pf=float((e ** 2).mean()), l2_vel=float((dv ** 2).mean()))
