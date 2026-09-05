"""H5: is the GT-body-frame transported comparison the same as a spatial/world one?

E_t = R_gt,t^T R_pred,t  (so3.act).  Claim:  ||E_t w_body,pred - w_body,gt||
                                     ==  ||w_spatial,pred - w_spatial,gt||.
And: how far is the transported se3 LINEAR part from the plain world-velocity difference?
"""
from __future__ import annotations
import sys
sys.path.insert(0, '/data3/rikhat.akizhanov/better/BetterRobot/src')
import numpy as np, torch
from better_robot.lie import se3, so3
import common as C

torch.set_default_dtype(torch.float64)
N, T, DT = 256, C.T, C.DT


def smooth_vec(n, T, v_rms, seed):
    out = np.stack([C.make_traj(n, 1.0, seed + 100 * k) for k in range(3)], -1)  # (n,T,3)
    d = np.diff(out, axis=1) / DT
    out *= v_rms / np.sqrt((d ** 2).mean())
    return out


def build(seed, w_rms, v_rms, rot_err_deg, pos_err_m, rot_noise_deg):
    """GT + predicted SE3 trajectories as (N,T,7) se3 tensors, plus the R matrices."""
    rng = np.random.default_rng(seed)
    # GT rotation: integrate a smooth body angular velocity
    wb = smooth_vec(N, T, w_rms, seed + 1)                     # (N,T,3) body ang vel target
    cur = so3.identity(batch_shape=(N,), dtype=torch.float64)
    qs = []
    for t in range(T):
        qs.append(cur.clone())
        cur = so3.compose(cur, so3.exp(torch.as_tensor(wb[:, t]) * DT))
    q_gt = torch.stack(qs, 1)                                   # (N,T,4)
    p_gt = torch.as_tensor(smooth_vec(N, T, v_rms, seed + 7))    # (N,T,3)
    # predicted = GT * exp(const offset + white noise)
    off = torch.as_tensor(rng.normal(0, 1, (N, 1, 3)))
    off = off / off.norm(dim=-1, keepdim=True) * np.deg2rad(rot_err_deg)
    nz = torch.as_tensor(rng.normal(0, np.deg2rad(rot_noise_deg) / np.sqrt(3), (N, T, 3)))
    q_pr = so3.compose(q_gt, so3.exp(off + nz))
    p_pr = p_gt + torch.as_tensor(rng.normal(0, pos_err_m, (N, T, 3)))
    T_gt = torch.cat([p_gt, q_gt], -1)
    T_pr = torch.cat([p_pr, q_pr], -1)
    return T_gt, T_pr, q_gt, q_pr, p_gt, p_pr


def analyse(tag, **kw):
    T_gt, T_pr, q_gt, q_pr, p_gt, p_pr = build(**kw)
    a, b = slice(None, -1), slice(1, None)
    d_gt = se3.log(se3.compose(se3.inverse(T_gt[:, a]), T_gt[:, b])) / DT
    d_pr = se3.log(se3.compose(se3.inverse(T_pr[:, a]), T_pr[:, b])) / DT
    E = so3.compose(so3.inverse(q_gt[:, a]), q_pr[:, a])
    # --- angular
    w_body_gt, w_body_pr = d_gt[..., 3:], d_pr[..., 3:]
    lhs = (so3.act(E, w_body_pr) - w_body_gt).norm(dim=-1)
    w_sp_gt = so3.log(so3.compose(q_gt[:, b], so3.inverse(q_gt[:, a]))) / DT
    w_sp_pr = so3.log(so3.compose(q_pr[:, b], so3.inverse(q_pr[:, a]))) / DT
    rhs = (w_sp_pr - w_sp_gt).norm(dim=-1)
    # also check w_spatial == R_t * w_body
    chk = (so3.act(q_gt[:, a], w_body_gt) - w_sp_gt).abs().max()
    # --- linear
    v_gt, v_pr = d_gt[..., :3], d_pr[..., :3]
    lin_se3 = (so3.act(E, v_pr) - v_gt).norm(dim=-1)
    dworld = ((p_pr[:, b] - p_pr[:, a]) - (p_gt[:, b] - p_gt[:, a])) / DT
    lin_world = dworld.norm(dim=-1)
    # "no V^-1" body variant: v' = R_t^T dp/dt
    vp_pr = so3.act(so3.inverse(q_pr[:, a]), (p_pr[:, b] - p_pr[:, a]) / DT)
    vp_gt = so3.act(so3.inverse(q_gt[:, a]), (p_gt[:, b] - p_gt[:, a]) / DT)
    lin_noV = (so3.act(E, vp_pr) - vp_gt).norm(dim=-1)
    rel = ((lin_se3 - lin_world) / lin_world.clamp(min=1e-9))
    print(f"\n### {tag}")
    print(f"  w_spatial == R_t*w_body : max abs dev = {float(chk):.3e}")
    print(f"  ANGULAR  max|transported-body  -  spatial| = {float((lhs-rhs).abs().max()):.3e}"
          f"   (rms level {float(rhs.mean()):.4f} rad/s)")
    print(f"  LINEAR   rms ||E v_pred - v_gt||           = {float((lin_se3**2).mean().sqrt()):.5f} m/s")
    print(f"           rms ||dp_pred - dp_gt||/dt (world)= {float((lin_world**2).mean().sqrt()):.5f} m/s")
    print(f"           rms  no-V^-1 body variant         = {float((lin_noV**2).mean().sqrt()):.5f} m/s"
          f"  (max dev vs world {float((lin_noV-lin_world).abs().max()):.3e})")
    print(f"           rms(lin_se3-lin_world)/rms(lin_world) = "
          f"{float(((lin_se3-lin_world)**2).mean().sqrt()/(lin_world**2).mean().sqrt())*100:.4f} %")
    print(f"           relative dev se3-vs-world: mean {float(rel.mean())*100:+.3f} %  "
          f"p50 {float(rel.median())*100:+.3f} %  p95 {float(rel.quantile(0.95))*100:+.3f} %  "
          f"max|.| {float(rel.abs().max())*100:.3f} %")
    return float(rel.abs().mean())


print("=" * 100)
print("H5: transported (GT-body-frame) vs spatial/world comparison.  N=%d clips, T=%d, dt=%.3f"
      % (N, T, DT))
print("Real magnitudes: root_ang_vel 0.486 rad/s, root_vel 0.288 m/s per component.")
print("=" * 100)
base = dict(seed=5, w_rms=C.GT_V_ROOT_ANG, v_rms=C.GT_V_ROOT_POS,
            rot_err_deg=10.0, pos_err_m=0.0154, rot_noise_deg=2.0)
analyse("REAL magnitudes (rot offset 10 deg, rot noise 2 deg, pos noise 15.4 mm)", **base)
analyse("no rotation error at all", **{**base, 'rot_err_deg': 0.0, 'rot_noise_deg': 0.0})
analyse("large rotation offset 55 deg (the v3 probe number)", **{**base, 'rot_err_deg': 55.0})
analyse("5x angular rate (2.43 rad/s)", **{**base, 'w_rms': 5 * C.GT_V_ROOT_ANG})
analyse("5x angular rate + 55 deg offset", **{**base, 'w_rms': 5 * C.GT_V_ROOT_ANG,
                                              'rot_err_deg': 55.0})
analyse("POSITION PERFECT, rot offset 10 deg + 2 deg/frame noise (isolates V^-1 coupling)",
        **{**base, 'pos_err_m': 0.0})
analyse("POSITION PERFECT, rot noise 5 deg/frame",
        **{**base, 'pos_err_m': 0.0, 'rot_noise_deg': 5.0})
analyse("POSITION PERFECT, constant rot offset only, no per-frame rot noise",
        **{**base, 'pos_err_m': 0.0, 'rot_noise_deg': 0.0})
analyse("20x angular rate (9.7 rad/s, unphysical stress test)",
        **{**base, 'w_rms': 20 * C.GT_V_ROOT_ANG})
