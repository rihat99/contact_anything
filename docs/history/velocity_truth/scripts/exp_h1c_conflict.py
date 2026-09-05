"""H1 core test: how much do the per-frame and velocity objectives actually CONFLICT,
as a function of the model class?  For each class: the pf-optimum and the vel-optimum,
each evaluated under BOTH objectives.  Excess = how much worse the other optimum is."""
from __future__ import annotations
import numpy as np, torch
import common as C, kernels as K
from scipy.optimize import minimize, minimize_scalar

N = 384
CH = dict(depth=(C.GT_V_ROOT_POS, C.SIGMA_DEPTH, 0.05, 0.4),
          root_ang=(C.GT_V_ROOT_ANG, 1.25*C.GT_V_ROOT_ANG*C.DT/np.sqrt(2), 0.1, 0.6),
          joint_ang=(C.GT_V_JOINT_ANG, 1.25*C.GT_V_JOINT_ANG*C.DT/np.sqrt(2), 0.1, 0.9))


def solve(x, g, hw, w_pf, w_vel, kind, dp, dv, sum_one=False):
    xt, gt = torch.as_tensor(x), torch.as_tensor(g)
    G, dG = K.targets(gt)
    def emb(w):
        k = torch.zeros(2*K.M+1, dtype=torch.float64)
        return torch.cat([k[:K.M-hw], w, k[K.M+hw+1:]])
    def f(wv):
        w = torch.tensor(wv, dtype=torch.float64, requires_grad=True)
        lp, lv = K.loss_terms(K.apply_k(xt, emb(w)), G, dG, kind, dp, dv)
        L = w_pf*lp + w_vel*lv
        gr, = torch.autograd.grad(L, [w])
        return float(L.detach()), gr.numpy()
    w0 = np.zeros(2*hw+1); w0[hw] = 1.0
    cons = ({'type':'eq','fun':lambda w: w.sum()-1},) if sum_one else ()
    best = None
    for m_ in (['SLSQP'] if sum_one else ['BFGS']):
        r = minimize(f, w0, jac=True, method=m_, constraints=cons,
                     options=dict(maxiter=5000))
        if best is None or r.fun < best.fun:
            best = r
    k = np.zeros(2*K.M+1); k[K.M-hw:K.M+hw+1] = best.x
    return k


def scale_only(x, g, w_pf, w_vel, kind, dp, dv):
    xt, gt = torch.as_tensor(x), torch.as_tensor(g)
    G, dG = K.targets(gt)
    d = torch.zeros(2*K.M+1, dtype=torch.float64); d[K.M] = 1.0
    def f(s):
        lp, lv = K.loss_terms(K.apply_k(xt, s*d), G, dG, kind, dp, dv)
        return float(w_pf*lp + w_vel*lv)
    s = minimize_scalar(f, bounds=(0., 2.), method='bounded', options=dict(xatol=1e-10)).x
    return (s*d).numpy()


def losses(x, g, k, kind, dp, dv):
    xt, gt = torch.as_tensor(x), torch.as_tensor(g)
    G, dG = K.targets(gt)
    lp, lv = K.loss_terms(K.apply_k(xt, torch.as_tensor(k)), G, dG, kind, dp, dv)
    return float(lp), float(lv)


for kind in ('hb', 'l2'):
    print("\n" + "=" * 118)
    print(f"CONFLICT between the per-frame and velocity objectives, loss={kind}, "
          f"N={N} clips (seeds 11/12)")
    print("  excess_pf  = L_pf(vel-optimum)/L_pf(pf-optimum) - 1   (how much per-frame accuracy "
          "the velocity optimum costs)")
    print("  excess_vel = L_vel(pf-optimum)/L_vel(vel-optimum) - 1")
    print("=" * 118)
    print(f"{'channel':<11}{'model class':<26}{'L_pf(pf*)':>12}{'L_pf(vel*)':>12}"
          f"{'excess_pf':>11}{'L_vel(vel*)':>13}{'L_vel(pf*)':>12}{'excess_vel':>11}"
          f"{'DC(pf*)':>9}{'DC(vel*)':>9}")
    for ch, (v, sig, dp, dv) in CH.items():
        g = C.make_traj(N, v, 11); x = C.add_white(g, sig, 12)
        classes = [
            ('A  scale only', lambda wp, wv: scale_only(x, g, wp, wv, kind, dp, dv)),
            ('B  +-4 convex (DC=1)', lambda wp, wv: solve(x, g, 4, wp, wv, kind, dp, dv, True)),
            ('D  +-4 free (kern+scale)', lambda wp, wv: solve(x, g, 4, wp, wv, kind, dp, dv)),
            ('   +-16 free', lambda wp, wv: solve(x, g, 16, wp, wv, kind, dp, dv)),
        ]
        for nm, fn in classes:
            kp = fn(1.0, 0.0); kv = fn(0.0, 1.0)
            pp, pv = losses(x, g, kp, kind, dp, dv)
            vp, vv = losses(x, g, kv, kind, dp, dv)
            print(f"{ch:<11}{nm:<26}{pp:>12.5g}{vp:>12.5g}{vp/pp-1:>11.2%}"
                  f"{vv:>13.5g}{pv:>12.5g}{pv/vv-1:>11.2%}{kp.sum():>9.3f}{kv.sum():>9.3f}")
