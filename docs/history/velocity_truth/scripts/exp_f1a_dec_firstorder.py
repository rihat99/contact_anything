"""F1(a): first-order gradient of the DECOUPLED transition loss at the identity point
(c = 0, h = 1), self-INCLUSIVE +-4 boxcar branch.  Same protocol as H4b/H4e:
8 clips per minibatch, 4000 fresh draws, mean / std / SNR / Adam drift-per-lr.

L_dec = (1 - r(dyhat, dg)) + beta * (RMS(dyhat) - RMS(dg))^2
  'detach'      = spec: stop-gradient on the prediction's own RMS in r's denominator
  'true-Pearson'= no stop-gradient (r is then EXACTLY scale-invariant)
Sign convention: gradient > 0 => descent DECREASES the parameter.
"""
from __future__ import annotations
import numpy as np, torch
import common as C, kernels as K, follow as F

NB = 4000
CAL = {ch: {dn: F.calibrate(ch, detach_norm=dn) for dn in (True, False)} for ch in F.CH}

print("=" * 122)
print(f"F1(a)  first-order gradients at the identity point (c=0, h=1), branch = self-INCLUSIVE "
      f"+-4 boxcar")
print(f"       {F.CLIPS} clips/minibatch x {NB} fresh draws (traj seeds 10000+, noise seeds 900000+)")
print("       Adam drift/lr = SNR/sqrt(1+SNR^2)   (1.0 = full learning rate)")
print("=" * 122)
for ch in F.CH:
    v, sig, dp, dv = F.CH[ch]
    print(f"\n### {ch}  (sigma={sig:.5g})   beta(detach)={CAL[ch][True]['beta']:.4f}  "
          f"beta(true-Pearson)={CAL[ch][False]['beta']:.4f}")
    print(f"{'loss term':<34}{'param':<6}{'mean':>13}{'std':>12}{'SNR':>10}"
          f"{'Adam drift/lr':>15}{'descent moves':>15}")
    u = K.boxcar(); d = torch.zeros(2 * K.M + 1, dtype=torch.float64); d[K.M] = 1.0
    specs = []
    for dn, tag in ((True, 'detach'), (False, 'true-Pearson')):
        b = CAL[ch][dn]['beta']
        specs += [(f'L_dec  ({tag}, beta={b:.3f})', 'dec', b, dn),
                  (f'  corr only ({tag}, beta=0)', 'dec', 0.0, dn)]
    specs += [('pointwise Huber velocity', 'vel', 0, True),
              ('pointwise Huber per-frame', 'pf', 0, True)]
    for name, kind, beta, dn in specs:
        acc = []
        for it in range(NB):
            g = torch.as_tensor(C.make_traj(F.CLIPS, v, 10000 + it))
            x = torch.as_tensor(C.add_white(g.numpy(), sig, 900000 + it))
            G, dG = K.targets(g)
            c = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            h = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
            y = K.apply_k(x, h * (d + c * u))
            if kind == 'dec':
                L = F.dec_loss(y, dG, beta, dn)
            else:
                lp, lv = K.loss_terms(y, G, dG, 'hb', dp, dv)
                L = lv if kind == 'vel' else lp
            acc.append([float(t) for t in torch.autograd.grad(L, [c, h])])
        a = np.array(acc)
        for i, pn in enumerate(['c', 'h']):
            m, s = a[:, i].mean(), a[:, i].std()
            snr = m / s if s > 0 else np.inf
            drift = snr / np.sqrt(1 + snr ** 2)
            move = 'UP' if m < 0 else 'DOWN'
            if abs(snr) < 0.05:
                move = 'nothing'
            print(f"{name:<34}{pn:<6}{m:>13.5g}{s:>12.4g}{snr:>10.4f}"
                  f"{drift:>15.4f}{move:>15}")
