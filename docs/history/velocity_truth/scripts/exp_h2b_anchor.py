"""How strong must the per-frame anchor be to keep the free scale near 1, given the
velocity term?  (Model A, L2, exact.)   s*(r) = (var_g + r*var_dg)/((var_g+var_n) + r*(var_dg+var_dn))
with r = w_vel/w_pf; solve for the r that yields a given s*."""
from __future__ import annotations
import numpy as np
import common as C

CH = [('depth (real 0.44%/frame)', C.GT_V_ROOT_POS, C.SIGMA_DEPTH),
      ('root ang, noise-vel x1.25', C.GT_V_ROOT_ANG, 1.25*C.GT_V_ROOT_ANG*C.DT/np.sqrt(2)),
      ('joint ang, noise-vel x1.25', C.GT_V_JOINT_ANG, 1.25*C.GT_V_JOINT_ANG*C.DT/np.sqrt(2))]
print("=" * 104)
print("Model A, L2.  r = w_vel/w_pf in the SAME units (metres / metres-per-second etc).")
print("s*(r) = (Vg + r Vdg)/((Vg+Vn) + r (Vdg+Vdn));   r for a target s* solves a linear equation.")
print("=" * 104)
print(f"{'channel':<28}{'var_g':>11}{'var_n':>11}{'var_dg':>10}{'var_dn':>10}"
      f"{'r for s*=0.99':>15}{'0.95':>10}{'0.90':>10}{'0.50':>10}")
for nm, v, sig in CH:
    g = C.make_traj(1024, v, 11); x = C.add_white(g, sig, 12)
    st = C.interior_stats(g, x)
    Vg, Vn, Vdg, Vdn = st['var_g'], st['var_n'], st['var_dg'], st['var_dn']
    def r_for(s):
        # s*((Vg+Vn) + r(Vdg+Vdn)) = Vg + r Vdg
        num = Vg - s*(Vg+Vn); den = s*(Vdg+Vdn) - Vdg
        return num/den if den > 0 else np.inf
    print(f"{nm:<28}{Vg:>11.5g}{Vn:>11.5g}{Vdg:>10.5g}{Vdn:>10.5g}"
          f"{r_for(0.99):>15.4g}{r_for(0.95):>10.4g}{r_for(0.90):>10.4g}{r_for(0.50):>10.4g}")
print()
print("Interpretation: r is the velocity weight (per unit per-frame weight) at which a purely")
print("per-frame model would still keep the given amplitude.  Compare with the toy's lambda60")
print("(L2) = 9.65e-4 and lambda60 (Huber) = 0.109.")
