"""H4 first-order check: exact gradient signs/magnitudes at the IDENTITY init.

(C) residual dilution  y = h*(x + c*mean_{+/-4} x)      at c=0, h=1  (= identity)
(B) non-residual convex softmax kernel, self-logit bias beta            at several beta
"""
from __future__ import annotations
import numpy as np, torch
import common as C, kernels as K

N = 512
CH = dict(
    depth=(C.GT_V_ROOT_POS, C.SIGMA_DEPTH, 0.05, 0.4),
    root_ang=(C.GT_V_ROOT_ANG, 1.25 * C.GT_V_ROOT_ANG * C.DT / np.sqrt(2), 0.1, 0.6),
    joint_ang=(C.GT_V_JOINT_ANG, 1.25 * C.GT_V_JOINT_ANG * C.DT / np.sqrt(2), 0.1, 0.9),
)

def data(ch, seed_g=11, seed_n=12):
    v, sig, dp, dv = CH[ch]
    g = C.make_traj(N, v, seed_g); x = C.add_white(g, sig, seed_n)
    return (torch.as_tensor(x), torch.as_tensor(g), dp, dv, sig)

print("=" * 100)
print("H4-a : gradient of each loss w.r.t. the residual branch gain c and the head scale h")
print("       evaluated EXACTLY AT THE IDENTITY INIT (c=0, h=1).  N=%d clips, seeds 11/12" % N)
print("       sign > 0  => gradient descent DECREASES the parameter")
print("=" * 100)
hdr = f"{'channel':<10}{'loss':<12}{'dL/dc':>14}{'dL/dh':>14}{'dL/dbeta_B(uniform)':>22}"
print(hdr)
for ch in CH:
    x, g, dp, dv, sig = data(ch)
    G, dG = K.targets(g)
    for kind, lab in [('l2', 'L2'), ('hb', 'Huber')]:
        for term in ('pf', 'vel'):
            m = K.ModelC(layers=1, norm=False, c0=0.0)
            with torch.no_grad():
                m.craw.fill_(-40.0)          # softplus(-40) = 0 exactly -> identity
                m.h.fill_(1.0)
            y = K.apply_k(x, m.kernel())
            lp, lv = K.loss_terms(y, G, dG, kind, dp, dv)
            loss = lp if term == 'pf' else lv
            gc, gh = torch.autograd.grad(loss, [m.craw, m.h], retain_graph=False)
            # analytic dL/dc uses dc/dcraw = sigmoid(craw) ~ 0 at -40; recompute wrt c directly
            u = K.boxcar(); d = torch.zeros(2 * K.M + 1, dtype=torch.float64); d[K.M] = 1.0
            c = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            h = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
            y2 = K.apply_k(x, h * (d + c * u))
            lp2, lv2 = K.loss_terms(y2, G, dG, kind, dp, dv)
            l2 = lp2 if term == 'pf' else lv2
            gc2, gh2 = torch.autograd.grad(l2, [c, h])
            # model B, self-bias only, at beta = 0 (uniform kernel)
            b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            lg = torch.zeros(2 * C.HALF + 1, dtype=torch.float64)
            lg = torch.cat([lg[:C.HALF], b.reshape(1), lg[C.HALF + 1:]])
            w = torch.softmax(lg, 0)
            kk = torch.zeros(2 * K.M + 1, dtype=torch.float64)
            kk = torch.cat([kk[:K.M - C.HALF], w, kk[K.M + C.HALF + 1:]])
            y3 = K.apply_k(x, kk)
            lp3, lv3 = K.loss_terms(y3, G, dG, kind, dp, dv)
            gb, = torch.autograd.grad(lp3 if term == 'pf' else lv3, [b])
            print(f"{ch:<10}{lab+'/'+term:<12}{float(gc2):>14.6g}{float(gh2):>14.6g}{float(gb):>22.6g}")

print()
print("Analytic check (L2, population):  dL_pf/dc = 2*sigma^2*u_0 = 2*sigma^2/9 > 0 ;"
      "  dL_vel/dc = 2*sigma^2*(2u_0-u_1-u_-1)/dt^2 = 0 for a boxcar")
for ch in CH:
    x, g, dp, dv, sig = data(ch)
    print(f"  {ch:<10} sigma={sig:.5g}  predicted dL_pf/dc = {2*sig**2/9:.6g}"
          f"   predicted dL_vel/dc = 0   predicted dL_pf/dh = 2*sigma^2 = {2*sig**2:.6g}"
          f"   predicted dL_vel/dh = 2*var(dn) = {2*2*sig**2/C.DT**2:.6g}")

print()
print("=" * 100)
print("H4-b : model B self-logit-bias gradient vs beta  (beta -> inf == identity init)")
print("       w_self = e^b/(e^b+8).  dL/dbeta < 0 => descent WIDENS is wrong; "
      "dL/dbeta > 0 => descent widens (lowers self weight)")
print("=" * 100)
for ch in ('depth', 'joint_ang'):
    x, g, dp, dv, sig = data(ch)
    G, dG = K.targets(g)
    print(f"\n-- {ch} --")
    print(f"{'beta':>6}{'w_self':>9}{'dLpf/db (L2)':>15}{'dLvel/db (L2)':>16}"
          f"{'dLpf/db (Hub)':>16}{'dLvel/db (Hub)':>16}")
    for beta in (0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0):
        row = [beta, float(np.exp(beta) / (np.exp(beta) + 8))]
        for kind in ('l2', 'hb'):
            for term in ('pf', 'vel'):
                b = torch.tensor(beta, dtype=torch.float64, requires_grad=True)
                lg = torch.zeros(2 * C.HALF + 1, dtype=torch.float64)
                lg = torch.cat([lg[:C.HALF], b.reshape(1), lg[C.HALF + 1:]])
                w = torch.softmax(lg, 0)
                kk = torch.zeros(2 * K.M + 1, dtype=torch.float64)
                kk = torch.cat([kk[:K.M - C.HALF], w, kk[K.M + C.HALF + 1:]])
                y = K.apply_k(x, kk)
                lp, lv = K.loss_terms(y, G, dG, kind, dp, dv)
                gb, = torch.autograd.grad(lp if term == 'pf' else lv, [b])
                row.append(float(gb))
        print(f"{row[0]:>6.1f}{row[1]:>9.4f}{row[2]:>15.4g}{row[4]:>16.4g}"
              f"{row[3]:>16.4g}{row[5]:>16.4g}")

print()
print("=" * 100)
print("H4-c : residual dilution attenuation  (1 + c*w_self)/(1+c) with w_self=1/9,")
print("       and the head rescale needed to keep DC gain 1")
print("=" * 100)
print(f"{'c':>8}{'noise gain (1+c/9)/(1+c)':>28}{'h for DC=1':>13}{'eff self weight':>18}")
for c in (0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 20.0, 1e6):
    ng = (1 + c / 9) / (1 + c)
    print(f"{c:>8.3g}{ng:>28.4f}{1/(1+c):>13.4f}{(1+c/9)/(1+c):>18.4f}")
