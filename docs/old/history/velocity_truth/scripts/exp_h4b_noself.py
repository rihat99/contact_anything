"""Follow-up to H4: the residual branch's first-order gradient depends on whether the
branch average INCLUDES the frame's own token.  Population value of dL/dc at identity is
2 sigma^2 (2 u_0 - u_1 - u_-1)/dt^2 for the velocity loss -- 0 for a symmetric boxcar that
includes self, NEGATIVE (= descent grows c = smoothing) if self is excluded."""
from __future__ import annotations
import numpy as np, torch
import common as C, kernels as K

CH = dict(depth=(C.GT_V_ROOT_POS, C.SIGMA_DEPTH, 0.05, 0.4),
          joint_ang=(C.GT_V_JOINT_ANG, 1.25*C.GT_V_JOINT_ANG*C.DT/np.sqrt(2), 0.1, 0.9))
CLIPS, NB = 8, 4000


def branch(kind):
    u = torch.zeros(2*K.M+1, dtype=torch.float64)
    if kind == 'boxcar9':          # includes self  (the real attention window)
        u[K.M-4:K.M+5] = 1/9
    elif kind == 'noself8':        # self key masked out
        u[K.M-4:K.M+5] = 1/8; u[K.M] = 0.0
    elif kind == 'gauss9':         # gaussian, sigma 2 frames, includes self
        j = np.arange(-4, 5); w = np.exp(-j**2/(2*2.0**2)); w /= w.sum()
        u[K.M-4:K.M+5] = torch.as_tensor(w)
    return u


print("=" * 112)
print("dL/dc at the identity point (c=0, h=1), for different residual-branch averages.")
print("sign > 0 => descent DECREASES c (anti-smoothing);  < 0 => descent GROWS c (smoothing).")
print(f"per-minibatch stats: {CLIPS} clips x {NB} fresh draws.  Huber losses.")
print("=" * 112)
print(f"{'channel':<11}{'branch':<10}{'2u0-u1-u-1':>12}{'loss':<10}{'mean dL/dc':>13}"
      f"{'std':>12}{'SNR':>9}{'Adam drift/lr':>15}")
for ch, (v, sig, dp, dv) in CH.items():
    for bk in ('boxcar9', 'noself8', 'gauss9'):
        u = branch(bk)
        pred = float(2*u[K.M] - u[K.M+1] - u[K.M-1])
        acc = {'pf': [], 'vel': []}
        d = torch.zeros(2*K.M+1, dtype=torch.float64); d[K.M] = 1.0
        for it in range(NB):
            g = torch.as_tensor(C.make_traj(CLIPS, v, 10000+it))
            x = torch.as_tensor(C.add_white(g.numpy(), sig, 900000+it))
            G, dG = K.targets(g)
            c = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            y = K.apply_k(x, d + c*u)
            lp, lv = K.loss_terms(y, G, dG, 'hb', dp, dv)
            a, = torch.autograd.grad(lp, [c], retain_graph=True)
            b, = torch.autograd.grad(lv, [c])
            acc['pf'].append(float(a)); acc['vel'].append(float(b))
        for term in ('pf', 'vel'):
            a = np.array(acc[term]); m, s = a.mean(), a.std()
            snr = m/s
            print(f"{ch:<11}{bk:<10}{pred:>12.4f}{term:<10}{m:>13.5g}{s:>12.4g}"
                  f"{snr:>9.4f}{snr/np.sqrt(1+snr**2):>15.4f}")
