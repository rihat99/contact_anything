"""H2 (shrinkage shortcut) + item 5 (Huber vs L2) + the depth/multiplicative case.

Model (A): per-frame affine head  yhat_t = s * x_t + b   (b free -> only the
centred moments matter).  x = g + n, n white.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar
import common as C

np.set_printoptions(precision=4, suppress=True)
N_CLIPS = 512
SEED_G, SEED_N = 11, 12


def s_star_l2(x, g, mode, lam=0.0):
    """Exact LS minimiser of  w_pf*||s*xc - gc||^2 + w_vel*||s*dx - dg||^2."""
    xi = x[:, C.INTERIOR]; gi = g[:, C.INTERIOR]
    xc = xi - xi.mean(1, keepdims=True); gc = gi - gi.mean(1, keepdims=True)
    dx = np.diff(x, axis=1)[:, C.HALF:C.T - C.HALF - 1] / C.DT
    dg = np.diff(g, axis=1)[:, C.HALF:C.T - C.HALF - 1] / C.DT
    num = den = 0.0
    if mode in ('pf', 'both'):
        w = 1.0 if mode == 'pf' else 1.0
        num += w * (xc * gc).mean(); den += w * (xc * xc).mean()
    if mode in ('vel', 'both'):
        w = 1.0 if mode == 'vel' else lam
        num += w * (dx * dg).mean(); den += w * (dx * dx).mean()
    return num / den


def loss_num(x, g, s, kind, delta=None, w_pf=0.0, w_vel=0.0):
    xi = x[:, C.INTERIOR]; gi = g[:, C.INTERIOR]
    xc = xi - xi.mean(1, keepdims=True); gc = gi - gi.mean(1, keepdims=True)
    dx = np.diff(x, axis=1)[:, C.HALF:C.T - C.HALF - 1] / C.DT
    dg = np.diff(g, axis=1)[:, C.HALF:C.T - C.HALF - 1] / C.DT
    e_p = s * xc - gc; e_v = s * dx - dg
    if kind == 'l2':
        return w_pf * (e_p ** 2).mean() + w_vel * (e_v ** 2).mean()
    return (w_pf * C.huber_np(e_p, delta[0]).mean()
            + w_vel * C.huber_np(e_v, delta[1]).mean())


def argmin_s(f, lo=-0.5, hi=2.0):
    r = minimize_scalar(f, bounds=(lo, hi), method='bounded',
                        options=dict(xatol=1e-8))
    return float(r.x)


def grad_ratio_lambda(x, g):
    """lambda = w_vel/w_pf such that ||dL_vel/dyhat|| = 0.6 * total at init (s=1)."""
    n = (x - g)
    gp = 2.0 * n[:, C.INTERIOR]                       # d/dyhat of ||y-g||^2
    e = np.diff(n, axis=1) / C.DT
    gv = np.zeros_like(n)
    gv[:, :-1] -= 2 * e / C.DT
    gv[:, 1:] += 2 * e / C.DT
    gv = gv[:, C.INTERIOR]
    np_, nv = np.linalg.norm(gp), np.linalg.norm(gv)
    return 1.5 * np_ / nv, np_, nv        # 60/40 -> ratio 1.5


def block(name, v_rms, sigma, delta_pf, delta_v):
    g = C.make_traj(N_CLIPS, v_rms, SEED_G)
    x = C.add_white(g, sigma, SEED_N)
    st = C.interior_stats(g, x)
    snr_p = st['var_g'] / st['var_n']
    snr_v = st['var_dg'] / st['var_dn']
    lam60, np_, nv = grad_ratio_lambda(x, g)
    print(f"\n=== {name}  v_rms={v_rms}  sigma={sigma:.5g} "
          f"noise_vel_rms={st['rms_dn']:.4f} (x{st['rms_dn']/st['rms_dg']:.2f} GT) ===")
    print(f"  var_g={st['var_g']:.6g} var_n={st['var_n']:.6g} "
          f"var_dg={st['var_dg']:.6g} var_dn={st['var_dn']:.6g}")
    print(f"  SNR_pf={snr_p:.4g} -> s*_pf(formula)={snr_p/(1+snr_p):.4f} | "
          f"SNR_vel={snr_v:.4g} -> s*_vel(formula)={snr_v/(1+snr_v):.4f}")
    print(f"  ||dLpf/dy||={np_:.4g} ||dLvel/dy||={nv:.4g}  "
          f"lambda(vel=60% of grad norm)={lam60:.6g}")
    rows = []
    for mode, lam in [('pf', 0), ('vel', 0), ('both', 0.01), ('both', 0.1),
                      ('both', 1.0), ('both', lam60), ('both', 10.0)]:
        w_pf = 0.0 if mode == 'vel' else 1.0
        w_vel = 1.0 if mode == 'vel' else (0.0 if mode == 'pf' else lam)
        s_ana = s_star_l2(x, g, mode, lam)
        s_l2 = argmin_s(lambda s: loss_num(x, g, s, 'l2', w_pf=w_pf, w_vel=w_vel))
        s_hb = argmin_s(lambda s: loss_num(x, g, s, 'hb', (delta_pf, delta_v),
                                           w_pf, w_vel))
        rows.append((mode, lam, s_ana, s_l2, s_hb))
    print(f"  {'loss':<6}{'lambda':>12}{'s*_L2(closed)':>15}{'s*_L2(num)':>12}{'s*_Huber':>11}")
    for m, lam, sa, sl, sh in rows:
        lab = {'pf': 'per-frame', 'vel': 'velocity', 'both': 'sum'}[m]
        print(f"  {lab:<6}{lam:>12.4g}{sa:>15.4f}{sl:>12.4f}{sh:>11.4f}")
    return dict(name=name, snr_p=snr_p, snr_v=snr_v, lam60=lam60, rows=rows, st=st)


print("#" * 78)
print("# H2 / item5 : per-frame affine head, additive white noise")
print(f"# T={C.T} dt={C.DT} clips={N_CLIPS} seeds g={SEED_G} n={SEED_N}; "
      f"interior rows only (t in [{C.HALF},{C.T-C.HALF-1}])")
print("#" * 78)

res = []
res.append(block('DEPTH  (0.44%/frame of 3.5 m)', C.GT_V_ROOT_POS, C.SIGMA_DEPTH, 0.05, 0.4))
for r in (0.5, 1.0, 1.25, 1.5, 2.0):
    sig = r * C.GT_V_ROOT_ANG * C.DT / np.sqrt(2)
    res.append(block(f'ROOT-ANG noise-vel x{r}', C.GT_V_ROOT_ANG, sig, 0.1, 0.6))
for r in (1.0, 1.25, 1.5):
    sig = r * C.GT_V_JOINT_ANG * C.DT / np.sqrt(2)
    res.append(block(f'JOINT-ANG noise-vel x{r}', C.GT_V_JOINT_ANG, sig, 0.1, 0.9))

# ------------------------------------------------------------------ depth: multiplicative
print("\n" + "#" * 78)
print("# DEPTH with MULTIPLICATIVE noise  x_t = z_t*(1+eps_t), eps~N(0,0.0044^2)")
print("# yhat = s*x  (shrinking the whole trajectory toward the camera)")
print("#" * 78)
zc = C.make_traj(N_CLIPS, C.GT_V_ROOT_POS, SEED_G)
z = zc + C.DEPTH_MEAN
rng = np.random.default_rng(SEED_N)
eps = rng.normal(0, C.DEPTH_NOISE_FRAC, size=z.shape)
xm = z * (1.0 + eps)

def metric_vel_loss(s, kind='hb', delta=0.4):
    dY = np.diff(s * xm, axis=1)[:, C.HALF:C.T - C.HALF - 1] / C.DT
    dG = np.diff(z, axis=1)[:, C.HALF:C.T - C.HALF - 1] / C.DT
    e = dY - dG
    return float((e ** 2).mean()) if kind == 'l2' else float(C.huber_np(e, delta).mean())

def dimless_vel_loss(s, kind='hb', delta=None):
    """velocity normalised by each trajectory's OWN depth: dlog z."""
    Y = s * xm
    dY = np.diff(np.log(Y), axis=1)[:, C.HALF:C.T - C.HALF - 1] / C.DT
    dG = np.diff(np.log(z), axis=1)[:, C.HALF:C.T - C.HALF - 1] / C.DT
    e = dY - dG
    d = delta if delta is not None else float(np.sqrt((dG ** 2).mean()))
    return float((e ** 2).mean()) if kind == 'l2' else float(C.huber_np(e, d).mean())

def pf_loss(s, kind='l2'):
    e = s * xm - z
    return float((e ** 2).mean()) if kind == 'l2' else float(C.huber_np(e, 0.05).mean())

print(f"  {'s':>6}{'metric vel L2':>15}{'metric vel Huber(0.4)':>23}"
      f"{'dimless vel L2':>16}{'dimless vel Huber':>19}{'per-frame L2':>14}")
for s in (1.0, 0.9, 0.75, 0.5, 0.3, 0.221, 0.1, 0.0):
    print(f"  {s:>6.3f}{metric_vel_loss(s,'l2'):>15.5f}{metric_vel_loss(s,'hb'):>23.5f}"
          f"{dimless_vel_loss(s,'l2'):>16.6f}{dimless_vel_loss(s,'hb'):>19.6f}"
          f"{pf_loss(s):>14.5f}")
print(f"  argmin s (metric vel, L2)    = {argmin_s(lambda s: metric_vel_loss(s,'l2'), 0.0, 2.0):.4f}")
print(f"  argmin s (metric vel, Huber) = {argmin_s(lambda s: metric_vel_loss(s,'hb'), 0.0, 2.0):.4f}")
print(f"  argmin s (dimless vel, L2)   = {argmin_s(lambda s: dimless_vel_loss(s,'l2'), 0.05, 3.0):.4f}")
print(f"  dimless vel L2 spread over s in [0.1,2]: "
      f"{max(dimless_vel_loss(s,'l2') for s in np.linspace(0.1,2,40)) - min(dimless_vel_loss(s,'l2') for s in np.linspace(0.1,2,40)):.3e}")
print(f"  argmin s (per-frame L2)      = {argmin_s(lambda s: pf_loss(s), 0.0, 2.0):.4f}")
