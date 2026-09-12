"""Shared machinery for the follow-up experiments F1 (decoupled transition loss) and
F2 (self-masked residual attention)."""
from __future__ import annotations
import numpy as np, torch
import common as C, kernels as K

torch.set_default_dtype(torch.float64)
DT = C.DT
CH = dict(depth=(C.GT_V_ROOT_POS, C.SIGMA_DEPTH, 0.05, 0.4),
          joint_ang=(C.GT_V_JOINT_ANG, 1.25 * C.GT_V_JOINT_ANG * C.DT / np.sqrt(2), 0.1, 0.9))
CLIPS = 8
EPS = 1e-12


# ------------------------------------------------------------------ losses
def corr_term(y, dG, detach_norm=True):
    """1 - Pearson r between the clip's T-1 predicted and GT increments.
    detach_norm=True puts a stop-gradient on the prediction's own RMS in the
    denominator (the spec's 'RMS(dyhat) detached inside the correlation term');
    the VALUE is the true Pearson r either way."""
    dy = (y[:, 1:] - y[:, :-1]) / DT
    a = dy - dy.mean(1, keepdim=True)
    b = dG - dG.mean(1, keepdim=True)
    na = a.norm(dim=1)
    nb = b.norm(dim=1)
    na_use = na.detach() if detach_norm else na
    r = (a * b).sum(1) / (na_use * nb + EPS)
    return (1.0 - r).mean()


def rms_term(y, dG):
    dy = (y[:, 1:] - y[:, :-1]) / DT
    return (((dy ** 2).mean(1).sqrt() - (dG ** 2).mean(1).sqrt()) ** 2).mean()


def rms_term_onesided(y, dG):
    """penalise only SHRINKAGE below the GT velocity RMS, never excess."""
    dy = (y[:, 1:] - y[:, :-1]) / DT
    gap = (dG ** 2).mean(1).sqrt() - (dy ** 2).mean(1).sqrt()
    return (torch.clamp(gap, min=0.0) ** 2).mean()


def dec_loss(y, dG, beta, detach_norm=True):
    return corr_term(y, dG, detach_norm) + beta * rms_term(y, dG)


def dec_loss_onesided(y, dG, beta, detach_norm=False):
    return corr_term(y, dG, detach_norm) + beta * rms_term_onesided(y, dG)


# ------------------------------------------------------------------ balances
def _ident_y(x):
    d = torch.zeros(2 * K.M + 1, dtype=torch.float64); d[K.M] = 1.0
    return K.apply_k(x, d).clone().requires_grad_(True)


def calibrate(ch, n=384, seed_g=11, seed_n=12, detach_norm=True):
    """beta (the two parts of L_dec have equal output-gradient norm at identity) and
    lambda60-style weights for the pointwise Huber velocity loss and for L_dec."""
    v, sig, dp, dv = CH[ch]
    g = torch.as_tensor(C.make_traj(n, v, seed_g))
    x = torch.as_tensor(C.add_white(g.numpy(), sig, seed_n))
    G, dG = K.targets(g)
    y = _ident_y(x)
    gc, = torch.autograd.grad(corr_term(y, dG, detach_norm), [y], retain_graph=True)
    gr, = torch.autograd.grad(rms_term(y, dG), [y], retain_graph=True)
    beta = float(gc.norm()) / float(gr.norm())
    lp, lv = K.loss_terms(y, G, dG, 'hb', dp, dv)
    gp, = torch.autograd.grad(lp, [y], retain_graph=True)
    gv, = torch.autograd.grad(lv, [y], retain_graph=True)
    gd, = torch.autograd.grad(dec_loss(y, dG, beta, detach_norm), [y])
    return dict(beta=beta, n_corr=float(gc.norm()), n_rms=float(gr.norm()),
                n_pf=float(gp.norm()), n_vel=float(gv.norm()), n_dec=float(gd.norm()),
                lam_vel=1.5 * float(gp.norm()) / float(gv.norm()),
                lam_dec=1.5 * float(gp.norm()) / float(gd.norm()),
                lam_dec_b0=1.5 * float(gp.norm()) / float(gc.norm()))


# ------------------------------------------------------------------ models
def make_model(kind, branch='boxcar9'):
    if kind == 'B':
        m = K.ModelB(free_scale=True, init_beta=12.0)
        return m, list(m.parameters()), ['logit'] * 9 + ['s']
    n = int(kind[1:])
    m = K.ModelC(n, norm=False, extra_scale=True, linear_c=True, branch=branch)
    with torch.no_grad():
        m.craw.fill_(0.0); m.h.fill_(1.0)
    return m, [m.craw, m.s], ['c'] * n + ['s']


def flat(ts):
    return torch.cat([(t if t is not None else torch.zeros(1)).reshape(-1) for t in ts])


# ------------------------------------------------------------------ runner
def run(ch, model_kind, aux, steps=60000, lr=2e-4, branch='boxcar9', seed0=500000,
        log_at=(2000, 10000, 60000), route=False):
    """aux: None | ('vel', lam) | ('dec', lam, beta, detach_norm)
             | ('dec1', lam, beta, detach_norm)   [one-sided amplitude term]
    route=True: the transition term sees a DETACHED head scale."""
    v, sig, dp, dv = CH[ch]
    m, params, names = make_model(model_kind, branch)
    opt = torch.optim.Adam(params, lr=lr)
    snap, traj = {}, []
    for it in range(steps):
        g = torch.as_tensor(C.make_traj(CLIPS, v, seed0 + it))
        x = torch.as_tensor(C.add_white(g.numpy(), sig, seed0 + 100000 + it))
        G, dG = K.targets(g)
        kern = m.kernel()
        y = K.apply_k(x, kern)
        lp, lv = K.loss_terms(y, G, dG, 'hb', dp, dv)
        loss = lp
        if aux is not None:
            ya = K.apply_k(x, kern / m.s * m.s.detach()) if route else y
            if aux[0] == 'vel':
                loss = lp + aux[1] * K.loss_terms(ya, G, dG, 'hb', dp, dv)[1]
            elif aux[0] == 'dec1':
                loss = lp + aux[1] * dec_loss_onesided(ya, dG, aux[2], aux[3])
            else:
                loss = lp + aux[1] * dec_loss(ya, dG, aux[2], aux[3])
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) in log_at or it == 0 or it % 500 == 0:
            with torch.no_grad():
                kk = m.kernel().numpy()
            rec = dict(it=it + 1, s=float(m.s), dc=float(kk.sum()), self=float(kk[K.M]))
            if model_kind != 'B':
                rec['c'] = float(m.craw[0])
            traj.append(rec)
            if (it + 1) in log_at:
                snap[it + 1] = rec
    gg = C.make_traj(384, v, 11); xx = C.add_white(gg, sig, 12)
    with torch.no_grad():
        kk = m.kernel().numpy()
    return m, K.eval_kernel(kk, xx, gg), snap, traj
