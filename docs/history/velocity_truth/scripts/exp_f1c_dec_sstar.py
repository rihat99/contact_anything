"""F1 extra: what is the shrinkage optimum s* for the SCALE-ONLY model (A) under the
decoupled loss, compared with the pointwise Huber velocity loss?
Pearson r is scale-invariant (true form) so only the RMS-matching term can shrink."""
from __future__ import annotations
import numpy as np, torch
from scipy.optimize import minimize_scalar
import common as C, kernels as K, follow as F

N = 384
d = torch.zeros(2 * K.M + 1, dtype=torch.float64); d[K.M] = 1.0
print("=" * 116)
print("Model A (scale only), s* under each transition objective.  N=384 clips, seeds 11/12.")
print("  RMS-matching alone has the closed form  s* = RMS(dg)/RMS(dx) = 1/sqrt(1 + var(dn)/var(dg))")
print("=" * 116)
print(f"{'channel':<11}{'objective':<40}{'s*':>9}{'rmse_pf(mm)@s*':>16}{'rmse_vel@s*':>13}")
for ch in F.CH:
    v, sig, dp, dv = F.CH[ch]
    cal = F.calibrate(ch, detach_norm=True)
    calt = F.calibrate(ch, detach_norm=False)
    g = torch.as_tensor(C.make_traj(N, v, 11))
    x = torch.as_tensor(C.add_white(g.numpy(), sig, 12))
    G, dG = K.targets(g)
    st = C.interior_stats(g.numpy(), x.numpy())
    closed = 1.0 / np.sqrt(1.0 + st['var_dn'] / st['var_dg'])
    def ev(name, f):
        r = minimize_scalar(f, bounds=(0.0, 2.0), method='bounded', options=dict(xatol=1e-9))
        s = float(r.x)
        e = K.eval_kernel((s * d).numpy(), x.numpy(), g.numpy())
        print(f"{ch:<11}{name:<40}{s:>9.4f}{e['rmse_pf']*1000:>16.2f}{e['rmse_vel']:>13.4f}")
    ev('pointwise Huber velocity only',
       lambda s: float(K.loss_terms(K.apply_k(x, s*d), G, dG, 'hb', dp, dv)[1]))
    ev('RMS-matching term only', lambda s: float(F.rms_term(K.apply_k(x, s*d), dG)))
    ev('L_dec (true-Pearson) only',
       lambda s: float(F.dec_loss(K.apply_k(x, s*d), dG, calt['beta'], False)))
    ev('L_dec (detach) only',
       lambda s: float(F.dec_loss(K.apply_k(x, s*d), dG, cal['beta'], True)))
    ev('pf + pointwise Huber vel (lambda60)',
       lambda s: float(sum(w*t for w, t in zip((1.0, cal['lam_vel']),
                       K.loss_terms(K.apply_k(x, s*d), G, dG, 'hb', dp, dv)))))
    ev('pf + L_dec (true-Pearson, matched)',
       lambda s: float(K.loss_terms(K.apply_k(x, s*d), G, dG, 'hb', dp, dv)[0]
                       + calt['lam_dec']*F.dec_loss(K.apply_k(x, s*d), dG, calt['beta'], False)))
    ev('pf + L_dec (detach, matched)',
       lambda s: float(K.loss_terms(K.apply_k(x, s*d), G, dG, 'hb', dp, dv)[0]
                       + cal['lam_dec']*F.dec_loss(K.apply_k(x, s*d), dG, cal['beta'], True)))
    print(f"{'':<11}{'[closed form for RMS-matching alone]':<40}{closed:>9.4f}")
