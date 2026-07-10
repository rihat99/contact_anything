"""SMPL-X (10475) -> SMPL (6890) converter — contacts, vertices, params.

Self-contained: loads only the small ``assets/smplx_to_smpl.npz`` produced by
``build_assets.py`` (no ``smplx`` library, no model files at runtime).

  * contacts / vertices : barycentric gather through the deformation-transfer
    correspondence (``v_smpl[i] = sum_k bc[i,k] * v_smplx[face[i,k]]``).
  * params : ``betas`` via a precomputed linear map, ``body_pose`` /
    ``global_orient`` copied (the SMPL-X body skeleton is the SMPL skeleton;
    SMPL's two extra hand joints are zeroed), ``transl`` pelvis-corrected so the
    SMPL body overlays the SMPL-X body in the canonical frame. Apply BIR's
    ``pre_*`` similarity afterwards exactly as for SMPL-X to reach camera space.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor

ArrayLike = Union[Tensor, np.ndarray]
_ASSET = Path(__file__).parent / "assets" / "smplx_to_smpl.npz"


class SmplxToSmpl:
    """Convert SMPL-X data to SMPL. All cached tensors live on ``device``."""

    def __init__(
        self,
        assets_path: Optional[str] = None,
        device: str = "cpu",
        threshold: float = 0.5,
    ):
        self.device = torch.device(device)
        self.threshold = threshold
        z = np.load(assets_path or _ASSET)

        def t(key, dtype=torch.float32):
            return torch.as_tensor(z[key], dtype=dtype, device=self.device)

        self.faces = t("closest_faces", torch.long)   # [6890, 3] SMPL-X vert ids
        self.bc    = t("bc")                           # [6890, 3] barycentric
        self.A     = t("betas_A")                      # [nb, nb] betas map
        self.J0x_tmpl, self.Jdirs_x = t("J0x_tmpl"), t("Jdirs_x")  # [3], [3, nb]
        self.J0s_tmpl, self.Jdirs_s = t("J0s_tmpl"), t("Jdirs_s")
        self.num_smplx = int(z["num_smplx"])
        self.num_smpl  = int(z["num_smpl"])

    # ------------------------------------------------------------------ verts / contacts

    def convert_contacts(self, contacts: ArrayLike, threshold: Optional[float] = None) -> Tensor:
        """``[*, 10475]`` (bool/float) -> ``[*, 6890]`` long binary contacts."""
        thr = self.threshold if threshold is None else threshold
        c, single = self._batch(contacts, ndim=2)                  # [B, 10475]
        interp = (c[:, self.faces] * self.bc).sum(-1)              # [B, 6890]
        out = (interp > thr).long()
        return out.squeeze(0) if single else out

    def convert_vertices(self, verts: ArrayLike) -> Tensor:
        """``[*, 10475, 3]`` -> ``[*, 6890, 3]`` barycentric-interpolated verts."""
        v, single = self._batch(verts, ndim=3)                     # [B, 10475, 3]
        out = (v[:, self.faces] * self.bc.unsqueeze(-1)).sum(-2)   # [B, 6890, 3]
        return out.squeeze(0) if single else out

    # ------------------------------------------------------------------ params

    def convert_params(
        self,
        betas: ArrayLike,           # [*, nb]
        body_pose: ArrayLike,       # [*, 63]  (21 SMPL-X body joints)
        global_orient: ArrayLike,   # [*, 3]
        transl: ArrayLike,          # [*, 3]
    ) -> dict[str, Tensor]:
        """Convert canonical-frame SMPL-X body params to canonical-frame SMPL.

        Returns ``betas`` ``[*, nb]``, ``body_pose`` ``[*, 69]``,
        ``global_orient`` ``[*, 3]``, ``transl`` ``[*, 3]``.
        """
        bx, single = self._batch(betas, ndim=2)
        bp, _ = self._batch(body_pose, ndim=2)
        go, _ = self._batch(global_orient, ndim=2)
        tr, _ = self._batch(transl, ndim=2)

        bs = bx @ self.A.T                                         # betas
        body = torch.cat([bp, bp.new_zeros(bp.shape[0], 6)], -1)  # +L/R hand = 0
        j0x = self.J0x_tmpl + bx @ self.Jdirs_x.T                 # pelvis(betas)
        j0s = self.J0s_tmpl + bs @ self.Jdirs_s.T
        transl_s = tr + (j0x - j0s)                               # align pelvis

        out = {"betas": bs, "body_pose": body, "global_orient": go.clone(),
               "transl": transl_s}
        return {k: (v.squeeze(0) if single else v) for k, v in out.items()}

    # ------------------------------------------------------------------ helper

    def _batch(self, x: ArrayLike, ndim: int) -> tuple[Tensor, bool]:
        t = torch.as_tensor(np.asarray(x) if isinstance(x, np.ndarray) else x,
                            dtype=torch.float32, device=self.device)
        single = t.dim() == ndim - 1
        return (t.unsqueeze(0) if single else t), single
