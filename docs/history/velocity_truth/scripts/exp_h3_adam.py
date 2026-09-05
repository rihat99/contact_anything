"""H3: (a) gradient spectrum, (b) Adam second-moment shrink of the per-frame step at a
FIXED parameter point (no trajectory confound), (c) dynamic runs incl. stop-grad routing."""
from __future__ import annotations
import json, sys
import numpy as np, torch
import common as C, kernels as K

torch.set_default_dtype(torch.float64)
CH = dict(depth=(C.GT_V_ROOT_POS, C.SIGMA_DEPTH, 0.05, 0.4),
          joint_ang=(C.GT_V_JOINT_ANG, 1.25 * C.GT_V_JOINT_ANG * C.DT / np.sqrt(2), 0.1, 0.9))
CLIPS = 8
KIND = 'hb'

# ---------------------------------------------------------------- (a) spectrum
print("=" * 104)
print("H3-a  gradient of each loss w.r.t. the PER-FRAME OUTPUTS yhat: transfer gain vs frequency")
print("  per-frame L2 : dL/dy = 2(y-g)                -> gain 2                       (flat)")
print("  velocity  L2 : dL/dy = (2/dt^2) D^T D (y-g)  -> gain 8 sin^2(w/2)/dt^2, dt=1/25")
print("  the repo's delta-normalised Huber divides each by delta^2 in the quadratic region")
print("=" * 104)
dt = C.DT
print(f"{'f (Hz)':>9}{'w/pi':>8}{'vel L2 gain':>14}{'pf L2 gain':>12}{'ratio L2':>11}"
      f"{'ratio Huber (d_pf=.05, d_v=.4)':>33}")
for f in (0.0, 0.25, 0.5, 1.0, 1.6, 2.0, 4.0, 8.0, 12.5):
    w = 2 * np.pi * f * dt
    gv = 8 * np.sin(w / 2) ** 2 / dt ** 2
    print(f"{f:>9.2f}{w/np.pi:>8.3f}{gv:>14.4g}{2.0:>12.4g}{gv/2:>11.4g}"
          f"{(gv/2/0.4**2)/(2/2/0.05**2):>33.4g}")
Tn = 512
r = np.random.default_rng(0).normal(size=Tn)
y = torch.tensor(r, requires_grad=True)
gv, = torch.autograd.grad((((y[1:] - y[:-1]) / dt) ** 2).sum(), [y])
Fy, Fg = np.fft.rfft(r), np.fft.rfft(gv.numpy())
w = 2 * np.pi * np.arange(len(Fy)) / Tn
sel = (w > 0.05 * np.pi) & (w < 0.95 * np.pi)
print(f"  numeric symbol check: max rel dev of |Fg/Fy| from 8 sin^2(w/2)/dt^2 = "
      f"{np.abs(np.abs(Fg/Fy)[sel] / (8*np.sin(w/2)**2/dt**2)[sel] - 1).max():.3e}")


def lam60(ch, kind=KIND):
    v, sig, dp, dv = CH[ch]
    g = C.make_traj(384, v, 11); x = C.add_white(g, sig, 12)
    G, dG = K.targets(torch.as_tensor(g))
    y = K.apply_k(torch.as_tensor(x), torch.eye(2*K.M+1, dtype=torch.float64)[K.M]).clone()
    y.requires_grad_(True)
    lp, lv = K.loss_terms(y, G, dG, kind, dp, dv)
    a, = torch.autograd.grad(lp, [y], retain_graph=True)
    b, = torch.autograd.grad(lv, [y])
    return 1.5 * float(a.norm()) / float(b.norm())


LAM = {c: lam60(c) for c in CH}
print("\nlambda60 (velocity = 60% of the output-gradient norm at identity, Huber):",
      {k: round(v, 5) for k, v in LAM.items()})


def make_model(kind):
    if kind == 'D_B':          # convex softmax kernel (identity init) + free head scale
        m = K.ModelB(free_scale=True, init_beta=12.0)
        return m, list(m.parameters()), ['logits'] * 9 + ['s']
    if kind == 'D_C4':         # 4 residual layers, LINEAR zero-init branch gain + head scale
        m = K.ModelC(4, norm=False, extra_scale=True, linear_c=True)
        with torch.no_grad():
            m.craw.fill_(0.0); m.h.fill_(1.0)
        return m, [m.craw, m.s], ['c'] * 4 + ['s']
    if kind == 'D_C1':         # 1 residual layer, LINEAR zero-init branch gain + head scale
        m = K.ModelC(1, norm=False, extra_scale=True, linear_c=True)
        with torch.no_grad():
            m.craw.fill_(0.0); m.h.fill_(1.0)
        return m, [m.craw, m.s], ['c', 's']
    raise ValueError(kind)


def grads(m, x, g, lam, use_vel, kind=KIND, dp=0.05, dv=0.4, route=False):
    G, dG = K.targets(g)
    kern = m.kernel()
    y = K.apply_k(x, kern)
    lp, lv = K.loss_terms(y, G, dG, kind, dp, dv)
    if route:      # velocity term sees a DETACHED head scale
        kv = kern / m.s * m.s.detach()
        lv = K.loss_terms(K.apply_k(x, kv), G, dG, kind, dp, dv)[1]
    return lp, lv


def flat(ts):
    return torch.cat([(t if t is not None else torch.zeros(1)).reshape(-1) for t in ts])


# ---------------------------------------------------------------- (b) static second moment
print()
print("=" * 104)
print(f"H3-b  AT A FIXED PARAMETER POINT (identity init), {CLIPS} clips/step, 3000 fresh "
      "minibatches: Adam second moment")
print("      shrink = sqrt(v_joint)/sqrt(v_pf-only) = the factor by which the PER-FRAME")
print("      gradient's Adam step is divided when the velocity term is added (same params).")
print("=" * 104)
for ch in CH:
    v, sig, dp, dv = CH[ch]
    lam = LAM[ch]
    for mk in ('D_B', 'D_C4'):
        m, params, names = make_model(mk)
        acc_pf, acc_j, acc_m = [], [], []
        for it in range(3000):
            g = torch.as_tensor(C.make_traj(CLIPS, v, 300000 + it))
            x = torch.as_tensor(C.add_white(g.numpy(), sig, 400000 + it))
            lp, lv = grads(m, x, g, lam, True, dp=dp, dv=dv)
            gp = flat(torch.autograd.grad(lp, params, retain_graph=True, allow_unused=True))
            gvv = flat(torch.autograd.grad(lv, params, allow_unused=True))
            acc_pf.append(gp.numpy()); acc_j.append((gp + lam * gvv).numpy())
        A, B_ = np.array(acc_pf), np.array(acc_j)
        vpf, vj = np.sqrt((A ** 2).mean(0)), np.sqrt((B_ ** 2).mean(0))
        mpf = A.mean(0)
        print(f"\n  {ch} / {mk}   lambda={lam:.4g}")
        print(f"    {'param':>8}{'mean g_pf':>13}{'sqrt(v) pf':>13}{'sqrt(v) joint':>15}"
              f"{'SHRINK':>9}{'|step_pf| w/o vel':>19}{'with vel':>11}")
        for i, nm in enumerate(names):
            sh = vj[i] / vpf[i]
            print(f"    {nm+str(i):>8}{mpf[i]:>13.4g}{vpf[i]:>13.4g}{vj[i]:>15.4g}"
                  f"{sh:>9.2f}{abs(mpf[i])/vpf[i]:>19.4f}{abs(mpf[i])/vj[i]:>11.4f}")

# ---------------------------------------------------------------- (c) dynamic runs
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
print()
print("=" * 104)
print(f"H3-c  dynamic Adam runs, lr 2e-4, {CLIPS} clips/step, {STEPS} steps, fresh data each step")
print("      proj = <-update, ghat_pf> = the distance moved per step DOWN the per-frame-loss "
      "gradient direction")
print("=" * 104)
for ch in CH:
    v, sig, dp, dv = CH[ch]
    lam = LAM[ch]
    print(f"\n--- {ch}  lambda={lam:.5g} ---")
    for mk in ('D_B', 'D_C1', 'D_C4'):
        for tag, uv, rt in [('pf only', False, False), ('pf+vel', True, False),
                            ('pf+vel  vel->kernel only', True, True)]:
            m, params, names = make_model(mk)
            opt = torch.optim.Adam(params, lr=2e-4)
            proj = []; traj = []
            for it in range(STEPS):
                g = torch.as_tensor(C.make_traj(CLIPS, v, 500000 + it))
                x = torch.as_tensor(C.add_white(g.numpy(), sig, 600000 + it))
                lp, lv = grads(m, x, g, lam, uv, dp=dp, dv=dv, route=rt)
                gp = flat(torch.autograd.grad(lp, params, retain_graph=True, allow_unused=True))
                gv2 = flat(torch.autograd.grad(lv, params, allow_unused=True))
                gt = gp + lam * gv2 if uv else gp
                o = 0
                for p in params:
                    n = p.numel(); p.grad = gt[o:o+n].reshape(p.shape).clone(); o += n
                prev = [p.detach().clone() for p in params]
                opt.step()
                u = flat([(p.detach() - q) for p, q in zip(params, prev)])
                proj.append(-float(u @ gp) / (float(gp.norm()) + 1e-30))
                if it % 500 == 0:
                    with torch.no_grad():
                        kk = m.kernel().numpy()
                    traj.append(dict(it=it, s=float(m.s), dc=float(kk.sum()),
                                     self=float(kk[K.M])))
            gg = C.make_traj(384, v, 11); xx = C.add_white(gg, sig, 12)
            with torch.no_grad():
                kk = m.kernel().numpy()
            d = K.eval_kernel(kk, xx, gg)
            pr = np.array(proj)
            extra = (f"c={np.array2string(m.craw.detach().numpy(), precision=3)}"
                     if mk.startswith('D_C') else "")
            print(f"  {mk:<6}{tag:<26} s={float(m.s):.4f} DC={d['dc']:.3f} self={d['self']:.3f} "
                  f"width={d['width_s']:.3f}s rmse_pf={d['rmse_pf']*1000:.2f}mm "
                  f"rmse_vel={d['rmse_vel']:.4f}  proj: all {pr.mean():.3g} "
                  f"first10% {pr[:len(pr)//10].mean():.3g} last10% {pr[-len(pr)//10:].mean():.3g} {extra}")
            json.dump(traj, open(f'h3c_{ch}_{mk}_{tag.split()[0]}{"_route" if rt else ""}.json', 'w'))


# ---------------------------------------------------------------- (d) c landscape
print()
print("=" * 104)
print("H3-d  landscape: population gradient of each loss w.r.t. the LINEAR residual branch gain c")
print("      (1 layer, y = s*(x + c*mean_{+/-4} x)); s at 1.0 and at the value that keeps DC gain 1")
print("=" * 104)
for ch in CH:
    v, sig, dp, dv = CH[ch]
    lam = LAM[ch]
    g = torch.as_tensor(C.make_traj(1024, v, 77))
    x = torch.as_tensor(C.add_white(g.numpy(), sig, 78))
    G, dG = K.targets(g)
    u = K.boxcar(); d = torch.zeros(2 * K.M + 1, dtype=torch.float64); d[K.M] = 1.0
    print(f"\n  {ch}: {'c':>7}{'s=1: dLpf/dc':>15}{'dLvel/dc':>12}{'| DC=1: s':>12}"
          f"{'dLpf/dc':>11}{'dLvel/dc':>11}{'rmse_pf(mm)':>13}{'rmse_vel':>10}")
    for cval in (0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        row = []
        for snorm in (1.0, 1.0 / (1.0 + cval)):
            c = torch.tensor(cval, dtype=torch.float64, requires_grad=True)
            y = K.apply_k(x, snorm * (d + c * u))
            lp, lv = K.loss_terms(y, G, dG, 'hb', dp, dv)
            a, = torch.autograd.grad(lp, [c], retain_graph=True)
            b, = torch.autograd.grad(lv, [c])
            row += [float(a), float(b)]
        e = K.eval_kernel(((1.0/(1.0+cval)) * (d + cval * u)).numpy(), x.numpy(), g.numpy())
        print(f"  {'':>4}{cval:>7.2f}{row[0]:>15.5g}{row[1]:>12.5g}"
              f"{1.0/(1.0+cval):>12.4f}{row[2]:>11.5g}{row[3]:>11.5g}"
              f"{e['rmse_pf']*1000:>13.2f}{e['rmse_vel']:>10.4f}")

# --------------------------------------------------- (e) c-gradient sign vs the head scale s
print()
print("=" * 104)
print("H3-e  sign of dL/dc AT c=0 as a function of the head scale s  (1 residual layer)")
print("      c>0 = 'add the local mean'.  dL/dc > 0 => Adam pushes c NEGATIVE (anti-smoothing).")
print("=" * 104)
for ch in CH:
    v, sig, dp, dv = CH[ch]
    lam = LAM[ch]
    g = torch.as_tensor(C.make_traj(1024, v, 77))
    x = torch.as_tensor(C.add_white(g.numpy(), sig, 78))
    G, dG = K.targets(g)
    u = K.boxcar(); d = torch.zeros(2 * K.M + 1, dtype=torch.float64); d[K.M] = 1.0
    print(f"\n  {ch} (lambda={lam:.4g}):")
    print(f"  {'s':>8}{'dLpf/dc':>14}{'dLvel/dc':>14}{'joint dL/dc':>14}"
          f"{'dLpf/ds':>12}{'dLvel/ds':>12}{'joint dL/ds':>13}")
    for sval in (1.0, 0.999, 0.99, 0.98, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2):
        c = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        sp = torch.tensor(sval, dtype=torch.float64, requires_grad=True)
        y = K.apply_k(x, sp * (d + c * u))
        lp, lv = K.loss_terms(y, G, dG, 'hb', dp, dv)
        a = torch.autograd.grad(lp, [c, sp], retain_graph=True)
        b = torch.autograd.grad(lv, [c, sp])
        print(f"  {sval:>8.3f}{float(a[0]):>14.5g}{float(b[0]):>14.5g}"
              f"{float(a[0])+lam*float(b[0]):>14.5g}"
              f"{float(a[1]):>12.5g}{float(b[1]):>12.5g}{float(a[1])+lam*float(b[1]):>13.5g}")
