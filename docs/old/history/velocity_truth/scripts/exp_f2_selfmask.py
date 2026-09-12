"""F2: self-MASKED residual attention in the dynamics.  Model C-1 / C-4 whose branch
average excludes the frame's own token (8 taps, 1/8), free head scale, H3c protocol.
Control = the same models with the self-inclusive boxcar."""
from __future__ import annotations
import json, sys
import numpy as np, torch
import common as C, kernels as K, follow as F

torch.set_num_threads(1)
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
CHS = sys.argv[2].split(',') if len(sys.argv) > 2 else ['depth']
LOG = (2000, 10000, STEPS)

print("=" * 128)
print("F2-a  first-order gradient on c at the identity point: self-inclusive vs self-masked branch")
print("      population prediction: dL_pf/dc = 2 sigma^2 u_0  ->  0 when the self tap is masked;")
print("                             dL_vel/dc = 2 sigma^2 (2u_0 - u_1 - u_-1)/dt^2")
print("      (8 clips/minibatch x 2000 fresh draws, Huber losses)")
print("=" * 128)
print(f"{'channel':<11}{'branch':<10}{'2u0-u1-u-1':>12}{'u_0':>7}{'loss':<6}"
      f"{'mean dL/dc':>13}{'std':>11}{'SNR':>9}{'drift/lr':>10}{'2*sig^2*u0':>12}")
for ch in F.CH:
    v, sig, dp, dv = F.CH[ch]
    d = torch.zeros(2 * K.M + 1, dtype=torch.float64); d[K.M] = 1.0
    for bk in ('boxcar9', 'noself8'):
        u = K.boxcar(kind=bk)
        pred = float(2 * u[K.M] - u[K.M + 1] - u[K.M - 1])
        acc = {'pf': [], 'vel': []}
        for it in range(2000):
            g = torch.as_tensor(C.make_traj(F.CLIPS, v, 10000 + it))
            x = torch.as_tensor(C.add_white(g.numpy(), sig, 900000 + it))
            G, dG = K.targets(g)
            c = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            y = K.apply_k(x, d + c * u)
            lp, lv = K.loss_terms(y, G, dG, 'hb', dp, dv)
            acc['pf'].append(float(torch.autograd.grad(lp, [c], retain_graph=True)[0]))
            acc['vel'].append(float(torch.autograd.grad(lv, [c])[0]))
        for term in ('pf', 'vel'):
            a = np.array(acc[term]); m, s = a.mean(), a.std(); snr = m / s
            print(f"{ch:<11}{bk:<10}{pred:>12.4f}{float(u[K.M]):>7.4f}{term:<6}"
                  f"{m:>13.5g}{s:>11.4g}{snr:>9.4f}{snr/np.sqrt(1+snr**2):>10.4f}"
                  f"{(2*sig**2*float(u[K.M]) if term=='pf' else float('nan')):>12.5g}")

for ch in CHS:
    cal = F.calibrate(ch, detach_norm=True)
    print("\n" + "=" * 132)
    print(f"F2-b  dynamics, channel {ch}: residual model with a SELF-MASKED branch vs the "
          f"self-inclusive control")
    print(f"      lr 2e-4, {F.CLIPS} clips/step, {STEPS} steps, identity init, "
          f"lambda_vel = {cal['lam_vel']:.6g}")
    print("=" * 132)
    print(f"{'model':<5}{'branch':<10}{'objective':<26}{'s@2k':>8}{'s@10k':>8}{'s@end':>8}"
          f"{'c@2k':>8}{'c@10k':>8}{'c@end':>8}{'DC':>7}{'self':>7}{'width_s':>8}"
          f"{'rmse_pf(mm)':>12}{'rmse_vel':>10}")
    for mk in ('C1', 'C4'):
        for br in ('noself8', 'boxcar9'):
            for tag, aux in [('pf only', None),
                             ('pf + pointwise Huber vel', ('vel', cal['lam_vel']))]:
                m, dd, snap, traj = F.run(ch, mk, aux, steps=STEPS, branch=br,
                                          log_at=LOG, seed0=500000)
                def g(step, f):
                    return snap[step][f] if step in snap else float('nan')
                print(f"{mk:<5}{br:<10}{tag:<26}{g(2000,'s'):>8.4f}{g(10000,'s'):>8.4f}"
                      f"{g(STEPS,'s'):>8.4f}{g(2000,'c'):>8.4f}{g(10000,'c'):>8.4f}"
                      f"{g(STEPS,'c'):>8.4f}{dd['dc']:>7.3f}{dd['self']:>7.3f}"
                      f"{dd['width_s']:>8.3f}{dd['rmse_pf']*1000:>12.2f}{dd['rmse_vel']:>10.4f}")
                json.dump(traj, open(f"f2_{ch}_{mk}_{br}_{tag.split()[0]}.json", 'w'))
