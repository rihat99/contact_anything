"""F1(b): dynamic Adam runs, exactly the H3c protocol (lr 2e-4, 8 clips/step, identity
init, fresh data every step), comparing
   pf only | pf + pointwise Huber velocity (lambda60) | pf + L_dec (same grad-norm balance)
for models B (softmax kernel + free head scale), C-1 and C-4 (residual + free head scale)."""
from __future__ import annotations
import json, sys
import numpy as np, torch
import common as C, kernels as K, follow as F

torch.set_num_threads(1)
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
CHS = sys.argv[2].split(',') if len(sys.argv) > 2 else ['depth']
LOG = (2000, 10000, STEPS)

for ch in CHS:
    cal_d = F.calibrate(ch, detach_norm=True)
    cal_t = F.calibrate(ch, detach_norm=False)
    print("=" * 132)
    print(f"F1(b)  channel {ch}   lr 2e-4, {F.CLIPS} clips/step, {STEPS} steps, identity init, "
          f"traj seeds 500000+, noise seeds 600000+")
    print(f"  lambda_vel (pointwise Huber, 60% grad-norm) = {cal_d['lam_vel']:.6g}")
    print(f"  L_dec detach:       beta = {cal_d['beta']:.6g}  lambda_dec = {cal_d['lam_dec']:.6g}"
          f"   (|dcorr/dy|={cal_d['n_corr']:.4g}, |drms/dy|={cal_d['n_rms']:.4g})")
    print(f"  L_dec true-Pearson: beta = {cal_t['beta']:.6g}  lambda_dec = {cal_t['lam_dec']:.6g}")
    print(f"  L_dec detach, beta=0 (corr only): lambda = {cal_d['lam_dec_b0']:.6g}")
    print("=" * 132)
    hdr = (f"{'model':<5}{'objective':<30}{'s@2k':>8}{'s@10k':>8}{'s@end':>8}"
           f"{'k@2k':>8}{'k@10k':>8}{'k@end':>8}{'DC':>7}{'self':>7}{'width_s':>8}"
           f"{'rmse_pf(mm)':>12}{'rmse_vel':>10}")
    print(hdr)
    for mk in ('B', 'C1', 'C4'):
        objs = [('pf only', None),
                ('pf + pointwise Huber vel', ('vel', cal_d['lam_vel'])),
                ('pf + L_dec (detach)', ('dec', cal_d['lam_dec'], cal_d['beta'], True)),
                ('pf + L_dec (true-Pearson)', ('dec', cal_t['lam_dec'], cal_t['beta'], False)),
                ('pf + corr only (detach)', ('dec', cal_d['lam_dec_b0'], 0.0, True))]
        for tag, aux in objs:
            m, d, snap, traj = F.run(ch, mk, aux, steps=STEPS, log_at=LOG, seed0=500000)
            key = 'c' if mk != 'B' else 'self'
            def g(step, f):
                return snap[step][f] if step in snap else float('nan')
            print(f"{mk:<5}{tag:<30}{g(2000,'s'):>8.4f}{g(10000,'s'):>8.4f}"
                  f"{g(STEPS,'s'):>8.4f}{g(2000,key):>8.4f}{g(10000,key):>8.4f}"
                  f"{g(STEPS,key):>8.4f}{d['dc']:>7.3f}{d['self']:>7.3f}{d['width_s']:>8.3f}"
                  f"{d['rmse_pf']*1000:>12.2f}{d['rmse_vel']:>10.4f}")
            json.dump(traj, open(f"f1b_{ch}_{mk}_{tag.replace(' ','_').replace('+','')}.json", 'w'))
