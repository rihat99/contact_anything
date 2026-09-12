"""H4 follow-up: is the first-order pull on c REAL or finite-sample noise, and what
does Adam actually do with it?  Per-minibatch gradient SNR (mean/std) at identity init."""
from __future__ import annotations
import numpy as np, torch
import common as C, kernels as K

CH = dict(
    depth=(C.GT_V_ROOT_POS, C.SIGMA_DEPTH, 0.05, 0.4),
    joint_ang=(C.GT_V_JOINT_ANG, 1.25 * C.GT_V_JOINT_ANG * C.DT / np.sqrt(2), 0.1, 0.9),
)
CLIPS_PER_STEP = 8          # the real run's batch (round-2 memory: 8 clips/step)
N_BATCH = 4000


def grads_at_identity(x, g, kind, dp, dv, term):
    G, dG = K.targets(g)
    u = K.boxcar(); d = torch.zeros(2 * K.M + 1, dtype=torch.float64); d[K.M] = 1.0
    c = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    h = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)   # B, uniform init
    lg = torch.zeros(2 * C.HALF + 1, dtype=torch.float64)
    lg = torch.cat([lg[:C.HALF], b.reshape(1), lg[C.HALF + 1:]])
    w = torch.softmax(lg, 0)
    kk = torch.zeros(2 * K.M + 1, dtype=torch.float64)
    kk = torch.cat([kk[:K.M - C.HALF], w, kk[K.M + C.HALF + 1:]])
    y1 = K.apply_k(x, h * (d + c * u)); y2 = K.apply_k(x, kk)
    l1 = K.loss_terms(y1, G, dG, kind, dp, dv)[0 if term == 'pf' else 1]
    l2 = K.loss_terms(y2, G, dG, kind, dp, dv)[0 if term == 'pf' else 1]
    gc, gh = torch.autograd.grad(l1, [c, h]); gb, = torch.autograd.grad(l2, [b])
    return float(gc), float(gh), float(gb)


print("=" * 108)
print(f"Per-minibatch ({CLIPS_PER_STEP} clips, {N_BATCH} independent draws, fresh GT+noise seeds) "
      "gradient statistics AT IDENTITY INIT")
print("SNR = mean/std.  Adam's steady drift ~ lr*mean/sqrt(mean^2+std^2) = lr*SNR/sqrt(1+SNR^2).")
print("=" * 108)
for ch, (v, sig, dp, dv) in CH.items():
    print(f"\n### {ch}  (sigma={sig:.5g}) ###")
    print(f"{'loss':<12}{'param':<8}{'mean':>13}{'std':>13}{'SNR':>10}"
          f"{'Adam drift/lr':>15}{'|mean|/|mean_h|':>17}")
    store = {}
    for kind in ('l2', 'hb'):
        for term in ('pf', 'vel'):
            acc = []
            for b_i in range(N_BATCH):
                g = C.make_traj(CLIPS_PER_STEP, v, 10000 + b_i)
                x = C.add_white(g, sig, 900000 + b_i)
                acc.append(grads_at_identity(torch.as_tensor(x), torch.as_tensor(g),
                                             kind, dp, dv, term))
            a = np.array(acc)
            store[(kind, term)] = a
            names = ['c (C branch)', 'h (head)', 'beta (B unif)']
            mh = abs(a[:, 1].mean())
            for i, nm in enumerate(names):
                m, s = a[:, i].mean(), a[:, i].std()
                snr = m / s if s > 0 else np.inf
                drift = snr / np.sqrt(1 + snr ** 2)
                print(f"{(kind+'/'+term):<12}{nm:<8}{m:>13.5g}{s:>13.5g}{snr:>10.4f}"
                      f"{drift:>15.4f}{abs(m)/mh:>17.5g}")
    np.save(f"gradsnr_{ch}.npy", np.stack([store[k] for k in store]))
