"""Item 2 follow-up: as a function of the velocity weight lambda, what does the JOINT
OPTIMUM look like for (A) scale-only, (B) convex kernel, (D) kernel+free scale?
Does the free scale steal the kernel's improvement AT THE OPTIMUM?"""
from __future__ import annotations
import numpy as np, torch
import common as C, kernels as K
from scipy.optimize import minimize

N = 384
CH = dict(depth=(C.GT_V_ROOT_POS, C.SIGMA_DEPTH, 0.05, 0.4),
          joint_ang=(C.GT_V_JOINT_ANG, 1.25 * C.GT_V_JOINT_ANG * C.DT / np.sqrt(2), 0.1, 0.9))


def lam60(x, g, kind, dp, dv):
    G, dG = K.targets(torch.as_tensor(g))
    y = K.apply_k(torch.as_tensor(x), torch.eye(2*K.M+1, dtype=torch.float64)[K.M]).clone()
    y.requires_grad_(True)
    lp, lv = K.loss_terms(y, G, dG, kind, dp, dv)
    a, = torch.autograd.grad(lp, [y], retain_graph=True)
    b, = torch.autograd.grad(lv, [y])
    return 1.5 * float(a.norm()) / float(b.norm())


def opt_kernel(x, g, hw, w_pf, w_vel, kind, dp, dv, sum_one=False):
    n = 2 * hw + 1
    xt, gt = torch.as_tensor(x), torch.as_tensor(g)
    G, dG = K.targets(gt)
    def f(wv):
        w = torch.tensor(wv, dtype=torch.float64, requires_grad=True)
        k = torch.zeros(2*K.M+1, dtype=torch.float64)
        k = torch.cat([k[:K.M-hw], w, k[K.M+hw+1:]])
        lp, lv = K.loss_terms(K.apply_k(xt, k), G, dG, kind, dp, dv)
        L = w_pf*lp + w_vel*lv
        gr, = torch.autograd.grad(L, [w])
        return float(L.detach()), gr.numpy()
    w0 = np.zeros(n); w0[hw] = 1.0
    cons = ({'type':'eq','fun':lambda w: w.sum()-1},) if sum_one else ()
    r = minimize(f, w0, jac=True, method='SLSQP' if sum_one else 'BFGS',
                 constraints=cons, options=dict(maxiter=3000))
    k = np.zeros(2*K.M+1); k[K.M-hw:K.M+hw+1] = r.x
    return k


def s_star(x, g, w_pf, w_vel, kind, dp, dv):
    from scipy.optimize import minimize_scalar
    xt, gt = torch.as_tensor(x), torch.as_tensor(g)
    G, dG = K.targets(gt)
    d = torch.zeros(2*K.M+1, dtype=torch.float64); d[K.M] = 1.0
    def f(s):
        lp, lv = K.loss_terms(K.apply_k(xt, s*d), G, dG, kind, dp, dv)
        return float(w_pf*lp + w_vel*lv)
    return minimize_scalar(f, bounds=(0.0, 2.0), method='bounded',
                           options=dict(xatol=1e-9)).x


for ch, (v, sig, dp, dv) in CH.items():
    g = C.make_traj(N, v, 11); x = C.add_white(g, sig, 12)
    for kind in ('hb', 'l2'):
        L60 = lam60(x, g, kind, dp, dv)
        print("\n" + "=" * 122)
        print(f"lambda sweep -- channel {ch}, loss {kind}, lambda60={L60:.5g}  "
              f"(N={N} clips, seeds 11/12)")
        print("=" * 122)
        print(f"{'lam/lam60':>10}{'lambda':>11} | {'A: s*':>7} {'A rmse_pf':>10} {'A rmse_v':>9}"
              f" | {'D: s(DC)':>9} {'D self':>7} {'D width_s':>10} {'D rmse_pf':>10} {'D rmse_v':>9}"
              f" | {'B(DC=1) rmse_pf':>16} {'B rmse_v':>9}")
        for mult in (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
            lam = L60 * mult
            s = s_star(x, g, 1.0, lam, kind, dp, dv)
            d = torch.zeros(2*K.M+1, dtype=torch.float64).numpy(); d[K.M] = s
            ea = K.eval_kernel(d, x, g)
            kD = opt_kernel(x, g, 4, 1.0, lam, kind, dp, dv, False)
            eD = K.eval_kernel(kD, x, g)
            kB = opt_kernel(x, g, 4, 1.0, lam, kind, dp, dv, True)
            eB = K.eval_kernel(kB, x, g)
            print(f"{mult:>10.3g}{lam:>11.4g} | {s:>7.4f} {ea['rmse_pf']*1000:>10.2f}"
                  f" {ea['rmse_vel']:>9.4f} | {eD['dc']:>9.4f} {eD['self']:>7.3f}"
                  f" {eD['width_s']:>10.3f} {eD['rmse_pf']*1000:>10.2f} {eD['rmse_vel']:>9.4f}"
                  f" | {eB['rmse_pf']*1000:>16.2f} {eB['rmse_vel']:>9.4f}")
        # dump the actual taps at lambda60
        kD = opt_kernel(x, g, 4, 1.0, L60, kind, dp, dv, False)
        print("  optimal +-4 taps at lambda60 (D):",
              np.array2string(kD[K.M-4:K.M+5], precision=4, suppress_small=True))
