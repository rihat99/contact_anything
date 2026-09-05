"""H3-d/H3-e (standalone): the c landscape, and the sign of dL/dc at c=0 vs the head scale s."""
from __future__ import annotations
import numpy as np, torch
import common as C, kernels as K
torch.set_default_dtype(torch.float64)
CH = dict(depth=(C.GT_V_ROOT_POS, C.SIGMA_DEPTH, 0.05, 0.4),
          joint_ang=(C.GT_V_JOINT_ANG, 1.25*C.GT_V_JOINT_ANG*C.DT/np.sqrt(2), 0.1, 0.9))

def lam60(ch):
    v, sig, dp, dv = CH[ch]
    g = C.make_traj(384, v, 11); x = C.add_white(g, sig, 12)
    G, dG = K.targets(torch.as_tensor(g))
    y = K.apply_k(torch.as_tensor(x), torch.eye(2*K.M+1, dtype=torch.float64)[K.M]).clone()
    y.requires_grad_(True)
    lp, lv = K.loss_terms(y, G, dG, 'hb', dp, dv)
    a, = torch.autograd.grad(lp, [y], retain_graph=True)
    b, = torch.autograd.grad(lv, [y])
    return 1.5*float(a.norm())/float(b.norm())
LAM = {c: lam60(c) for c in CH}
print("lambda60 (Huber):", {k: round(v, 5) for k, v in LAM.items()})

print("\n" + "="*112)
print("H3-d  landscape: gradient of each loss w.r.t. the LINEAR residual branch gain c")
print("      y = s*(x + c*mean_{+/-4} x), 1 layer.  s=1 (head does NOT rescale) vs s=1/(1+c) (head co-adapts)")
print("="*112)
for ch in CH:
    v, sig, dp, dv = CH[ch]
    g = torch.as_tensor(C.make_traj(1024, v, 77))
    x = torch.as_tensor(C.add_white(g.numpy(), sig, 78))
    G, dG = K.targets(g)
    u = K.boxcar(); d = torch.zeros(2*K.M+1, dtype=torch.float64); d[K.M] = 1.0
    print(f"\n  {ch}: {'c':>7}{'s=1: dLpf/dc':>15}{'dLvel/dc':>12}{'| DC=1: s':>12}"
          f"{'dLpf/dc':>11}{'dLvel/dc':>11}{'rmse_pf(mm)':>13}{'rmse_vel':>10}")
    for cval in (0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        row = []
        for snorm in (1.0, 1.0/(1.0+cval)):
            c = torch.tensor(cval, dtype=torch.float64, requires_grad=True)
            y = K.apply_k(x, snorm*(d + c*u))
            lp, lv = K.loss_terms(y, G, dG, 'hb', dp, dv)
            a, = torch.autograd.grad(lp, [c], retain_graph=True)
            b, = torch.autograd.grad(lv, [c])
            row += [float(a), float(b)]
        e = K.eval_kernel(((1.0/(1.0+cval))*(d + cval*u)).numpy(), x.numpy(), g.numpy())
        print(f"  {'':>4}{cval:>7.2f}{row[0]:>15.5g}{row[1]:>12.5g}{1.0/(1.0+cval):>12.4f}"
              f"{row[2]:>11.5g}{row[3]:>11.5g}{e['rmse_pf']*1000:>13.2f}{e['rmse_vel']:>10.4f}")

print("\n" + "="*112)
print("H3-e  sign of dL/dc AT c=0 as a function of the head scale s  (1 residual layer)")
print("      dL/dc > 0 => Adam pushes c NEGATIVE (anti-smoothing);  < 0 => c grows (smoothing)")
print("="*112)
for ch in CH:
    v, sig, dp, dv = CH[ch]
    lam = LAM[ch]
    g = torch.as_tensor(C.make_traj(1024, v, 77))
    x = torch.as_tensor(C.add_white(g.numpy(), sig, 78))
    G, dG = K.targets(g)
    u = K.boxcar(); d = torch.zeros(2*K.M+1, dtype=torch.float64); d[K.M] = 1.0
    print(f"\n  {ch} (lambda={lam:.4g}):")
    print(f"  {'s':>8}{'dLpf/dc':>14}{'dLvel/dc':>14}{'joint dL/dc':>14}"
          f"{'dLpf/ds':>12}{'dLvel/ds':>12}{'joint dL/ds':>13}")
    for sval in (1.0, 0.999, 0.99, 0.98, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2):
        c = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        sp = torch.tensor(sval, dtype=torch.float64, requires_grad=True)
        y = K.apply_k(x, sp*(d + c*u))
        lp, lv = K.loss_terms(y, G, dG, 'hb', dp, dv)
        a = torch.autograd.grad(lp, [c, sp], retain_graph=True)
        b = torch.autograd.grad(lv, [c, sp])
        print(f"  {sval:>8.3f}{float(a[0]):>14.5g}{float(b[0]):>14.5g}"
              f"{float(a[0])+lam*float(b[0]):>14.5g}{float(a[1]):>12.5g}"
              f"{float(b[1]):>12.5g}{float(a[1])+lam*float(b[1]):>13.5g}")
