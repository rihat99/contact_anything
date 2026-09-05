"""H1 + item 2: do the model families reach the oracle filter, and does a free head
scale steal the improvement?   Full-batch Adam to the family optimum."""
from __future__ import annotations
import numpy as np, torch
import common as C, kernels as K
from scipy.optimize import minimize

N = 384
CH = dict(
    depth=(C.GT_V_ROOT_POS, C.SIGMA_DEPTH, 0.05, 0.4),
    joint_ang=(C.GT_V_JOINT_ANG, 1.25 * C.GT_V_JOINT_ANG * C.DT / np.sqrt(2), 0.1, 0.9),
)


def oracle(x, g, hw, w_pf, w_vel, sum_one=False, kind='l2', dp=1., dv=1.):
    """best LTI kernel of half-width hw for the weighted loss (interior margin M)."""
    n = 2 * hw + 1
    xt, gt = torch.as_tensor(x), torch.as_tensor(g)
    G, dG = K.targets(gt)
    def full(w):
        k = torch.zeros(2 * K.M + 1, dtype=torch.float64)
        return torch.cat([k[:K.M - hw], w, k[K.M + hw + 1:]])
    def f(wv):
        w = torch.tensor(wv, dtype=torch.float64, requires_grad=True)
        y = K.apply_k(xt, full(w))
        lp, lv = K.loss_terms(y, G, dG, kind, dp, dv)
        L = w_pf * lp + w_vel * lv
        gr, = torch.autograd.grad(L, [w])
        return float(L.detach()), gr.numpy()
    w0 = np.zeros(n); w0[hw] = 1.0
    cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1},) if sum_one else ()
    r = minimize(f, w0, jac=True, method='SLSQP' if sum_one else 'BFGS',
                 constraints=cons, options=dict(maxiter=2000))
    k = np.zeros(2 * K.M + 1); k[K.M - hw:K.M + hw + 1] = r.x
    return k


def lam_60(x, g, kind, dp, dv):
    """weight ratio making the velocity term 60% of the output-gradient norm at identity."""
    xt, gt = torch.as_tensor(x), torch.as_tensor(g)
    G, dG = K.targets(gt)
    y = K.apply_k(xt, torch.eye(2 * K.M + 1, dtype=torch.float64)[K.M]).requires_grad_(True)
    lp, lv = K.loss_terms(y, G, dG, kind, dp, dv)
    gp, = torch.autograd.grad(lp, [y], retain_graph=True)
    gv, = torch.autograd.grad(lv, [y])
    return 1.5 * float(gp.norm()) / float(gv.norm()), float(gp.norm()), float(gv.norm())


def line(tag, d, extra=""):
    return (f"{tag:<26}{d['dc']:>8.3f}{d['self']:>8.3f}{d['width_s']:>9.3f}"
            f"{d['negmass']:>9.3f}{d['rmse_pf']*1000:>11.2f}{d['rmse_vel']:>11.4f}  {extra}")


for ch, (v, sig, dp, dv) in CH.items():
    g = C.make_traj(N, v, 11); x = C.add_white(g, sig, 12)
    xt, gt = torch.as_tensor(x), torch.as_tensor(g)
    for kind in ('l2', 'hb'):
        lam, np_, nv_ = lam_60(x, g, kind, dp, dv)
        print("\n" + "=" * 118)
        print(f"CHANNEL {ch}   loss={kind}   sigma={sig:.5g}  "
              f"deltas pf={dp} vel={dv}   |dLpf/dy|={np_:.4g} |dLvel/dy|={nv_:.4g}")
        print(f"weights: w_pf=1, w_vel=lambda; lambda60 (velocity = 60% of the output-grad "
              f"norm at identity) = {lam:.6g}")
        print("=" * 118)
        print(f"{'model':<26}{'DCgain':>8}{'self':>8}{'width_s':>9}{'negmass':>9}"
              f"{'rmse_pf(mm)':>11}{'rmse_vel':>11}")
        # reference points
        idk = np.zeros(2 * K.M + 1); idk[K.M] = 1.0
        print(line('identity (raw x)', K.eval_kernel(idk, x, g)))
        for hw, tag in ((4, 'oracle+-4'), (16, 'oracle+-16')):
            for wp, wv, nm in ((1, 0, 'pf-only'), (0, 1, 'vel-only'), (1, lam, 'joint')):
                k = oracle(x, g, hw, wp, wv, False, kind, dp, dv)
                print(line(f'{tag} {nm}', K.eval_kernel(k, x, g)))
        k = oracle(x, g, 4, 1, lam, True, kind, dp, dv)
        print(line('oracle+-4 joint DC=1', K.eval_kernel(k, x, g)))
        # trained families
        models = [
            ('A scale-only', K.ModelA()),
            ('B convex(free logits)', K.ModelB(free_scale=False, init_beta=0.0)),
            ('B convex init=identity', K.ModelB(free_scale=False, init_beta=12.0)),
            ('D = B + free scale', K.ModelB(free_scale=True, init_beta=0.0)),
            ('D = B(id init) + scale', K.ModelB(free_scale=True, init_beta=12.0)),
            ('C1 norm (DC=1)', K.ModelC(1, norm=True)),
            ('C1 free h (=D)', K.ModelC(1, norm=False)),
            ('C4 norm (DC=1)', K.ModelC(4, norm=True)),
            ('C4 free h (=D)', K.ModelC(4, norm=False)),
        ]
        for nm, m in models:
            K.train(m, xt, gt, kind, dp, dv, 1.0, lam, lr=1e-2, steps=5000)
            K.train(m, xt, gt, kind, dp, dv, 1.0, lam, lr=1e-3, steps=2500)
            d = K.report(m, xt, gt, nm)
            extra = ''
            if isinstance(m, K.ModelC):
                cs = torch.nn.functional.softplus(m.craw).detach().numpy()
                extra = f"c={np.array2string(cs, precision=2)} h={np.array2string(m.h.detach().numpy(), precision=3)}"
            if isinstance(m, (K.ModelB, K.ModelA)):
                extra = f"s={float(m.s):.4f}"
            print(line(nm, d, extra))
