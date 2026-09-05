"""Run ONE follow-up dynamic configuration and print one result line.
usage: exp_f_one.py <channel> <model B|C1|C4> <branch boxcar9|noself8> <obj> <steps>
obj in: pf | vel | dec_detach | dec_true | corr_detach
"""
from __future__ import annotations
import json, sys
import numpy as np, torch
import common as C, kernels as K, follow as F

torch.set_num_threads(1)
ch, mk, br, obj, steps = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
route = len(sys.argv) > 6 and sys.argv[6] == 'route'
LOG = (2000, 10000, steps)
cd = F.calibrate(ch, detach_norm=True)
ct = F.calibrate(ch, detach_norm=False)
aux = {'pf': None,
       'vel': ('vel', cd['lam_vel']),
       'dec_detach': ('dec', cd['lam_dec'], cd['beta'], True),
       'dec_true': ('dec', ct['lam_dec'], ct['beta'], False),
       'corr_detach': ('dec', cd['lam_dec_b0'], 0.0, True),
       'corr_true': ('dec', ct['lam_dec_b0'], 0.0, False),
       'onesided': ('dec1', ct['lam_dec_b0'], ct['beta'], False)}[obj]
m, d, snap, traj = F.run(ch, mk, aux, steps=steps, branch=br, log_at=LOG,
                         seed0=500000, route=route)
key = 'c' if mk != 'B' else 'self'
def g(st, f):
    return snap[st][f] if st in snap else float('nan')
print(f"{ch:<10}{mk:<4}{br:<9}{obj + ("_route" if route else ""):<18}{g(2000,'s'):>8.4f}{g(10000,'s'):>8.4f}"
      f"{g(steps,'s'):>8.4f}{g(2000,key):>8.4f}{g(10000,key):>8.4f}{g(steps,key):>8.4f}"
      f"{d['dc']:>7.3f}{d['self']:>7.3f}{d['width_s']:>8.3f}{d['rmse_pf']*1000:>12.2f}"
      f"{d['rmse_vel']:>10.4f}", flush=True)
json.dump(traj, open(f"ftraj_{ch}_{mk}_{br}_{obj}.json", 'w'))
