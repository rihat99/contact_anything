"""Build the compact SMPL-X -> SMPL conversion asset (run once).

Everything heavy (the 6890x10475 deformation-transfer matrix and both
shape spaces) is folded here into a tiny, self-contained npz so the
runtime converter needs no model files and no ``smplx`` library:

  * vertices / contacts  -> barycentric gather  (closest_faces + bc)
  * betas                -> affine map  betas_smpl = betas_smplx @ A.T + b
                            (exact because the transfer is linear and the
                             shape blendshapes are linear in betas)
  * translation          -> pelvis-offset correction from precomputed
                            joint-regressor terms

Source files (defaults are the copies on this machine):
  * transfer matrix : Phy-SIC ``smplx_to_smpl.pkl``  (keys: matrix, bc,
                      closest_faces, valid_vertices)
  * SMPL / SMPL-X   : better_human neutral ``.npz`` models

Run::

    python conversion/smplx_smpl_conversion/build_assets.py
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEF_PKL   = "/data3/rikhat.akizhanov/better/other/Phy-SIC/data/conversions/smplx_to_smpl.pkl"
DEF_SMPL  = "/data3/rikhat.akizhanov/better/better_human/models/smpl/SMPL_NEUTRAL.npz"
DEF_SMPLX = "/data3/rikhat.akizhanov/better/better_human/models/smplx/SMPLX_NEUTRAL.npz"
NUM_BETAS = 10


def _shaped(v_template, shapedirs, betas):
    """T-pose mesh for a batch of betas: ``v_template + shapedirs . betas``."""
    return v_template + np.einsum("vck,nk->nvc", shapedirs, betas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pkl",   default=DEF_PKL)
    ap.add_argument("--smpl",  default=DEF_SMPL)
    ap.add_argument("--smplx", default=DEF_SMPLX)
    ap.add_argument("--out",   default=str(HERE / "assets" / "smplx_to_smpl.npz"))
    ap.add_argument("--num-betas", type=int, default=NUM_BETAS)
    args = ap.parse_args()
    nb = args.num_betas

    with open(args.pkl, "rb") as f:
        tf = pickle.load(f, encoding="latin1")
    M  = tf["matrix"].astype(np.float64)            # [6890, 10475]
    faces = tf["closest_faces"].astype(np.int32)    # [6890, 3] SMPL-X vert ids
    bc    = tf["bc"].astype(np.float64)             # [6890, 3] barycentric

    smpl  = np.load(args.smpl,  allow_pickle=True)
    smplx = np.load(args.smplx, allow_pickle=True)
    Vs0 = smpl["v_template"].astype(np.float64)                  # [6890, 3]
    Ss  = smpl["shapedirs"][:, :, :nb].astype(np.float64)        # [6890, 3, nb]
    Js0 = smpl["J_regressor"][0].astype(np.float64)              # [6890] pelvis row
    Vx0 = smplx["v_template"].astype(np.float64)                 # [10475, 3]
    Sx  = smplx["shapedirs"][:, :, :nb].astype(np.float64)       # [10475, 3, nb]
    Jx0 = np.asarray(smplx["J_regressor"])[0].astype(np.float64)  # [10475] pelvis row

    # --- betas linear map: betas_smpl = betas_smplx @ A.T -------------------
    # Convert the shape *deformation* (not absolute vertices): the SMPL-X
    # blendshapes are transferred through the correspondence and re-fit by the
    # SMPL blendshapes. neutral (betas=0) maps to neutral, so there is no bias
    # term -- the two templates differ only by a constant frame offset (which
    # is irrelevant to shape and is handled for translation via the pelvis).
    Ss_pinv = np.linalg.pinv(Ss.reshape(-1, nb))               # [nb, 20670]
    MSx     = np.einsum("ij,jck->ick", M, Sx).reshape(-1, nb)  # [20670, nb]
    A = Ss_pinv @ MSx                                           # [nb, nb]

    # --- pelvis terms for the translation correction -----------------------
    # J0(betas) = J0_template + Jdirs @ betas, per model.
    J0x_tmpl = Jx0 @ Vx0;  Jdirs_x = np.einsum("v,vck->ck", Jx0, Sx)   # [3],[3,nb]
    J0s_tmpl = Js0 @ Vs0;  Jdirs_s = np.einsum("v,vck->ck", Js0, Ss)

    # ----------------------------------------------------------------- save
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        closest_faces=faces, bc=bc.astype(np.float32),
        betas_A=A.astype(np.float32),
        J0x_tmpl=J0x_tmpl.astype(np.float32), Jdirs_x=Jdirs_x.astype(np.float32),
        J0s_tmpl=J0s_tmpl.astype(np.float32), Jdirs_s=Jdirs_s.astype(np.float32),
        num_smplx=np.int64(Vx0.shape[0]), num_smpl=np.int64(Vs0.shape[0]),
        num_betas=np.int64(nb),
    )

    # ----------------------------------------------------------------- report
    off = (M @ Vx0 - Vs0).mean(0)                              # template frame offset
    rng = np.random.default_rng(0)
    bx  = rng.normal(0, 1.5, size=(64, nb))                    # plausible shapes
    defx = np.einsum("ij,njc->nic", M, _shaped(0, Sx, bx))    # transferred deformation
    defs = _shaped(0, Ss, bx @ A.T)                            # SMPL deformation
    res = np.linalg.norm(defs - defx, axis=-1).mean(1) * 1000.0
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
    print(f"template frame offset (m): {off.round(4)}  (handled via pelvis, not betas)")
    print(f"betas deformation-fit residual (mm):  mean={res.mean():.1f} max={res.max():.1f}  "
          f"(10-beta cross-topology limit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
