"""Bridge the frozen SAM-3D-Body outputs onto a BetterHuman MHR body.

The single place that knows how the frozen model's per-frame MHR parameters map
onto BetterHuman's ``MHRClassic`` compact vector and how per-frame camera
extrinsics place every frame's body in one static metric reconstruction world
(plan ``README.md`` §2, D7). Everything runs under :func:`torch.no_grad`: the
physics loss (step 06) receives ``q`` (and the shaped body) as constants — only
the predicted forces carry gradients.

Two facts, both verified against the vendored model on real checkpoint outputs,
fix the mapping:

* The 204-slot ``mhr_model_params`` vector fed to SAM's MHR *is* Momentum's
  ``compact_v6`` vector, so ``MHRClassic(model_parameters=mhr_model_params)``
  reproduces SAM's native joint centres bit-for-bit (0.0 mm) — no permutation.
* SAM's per-frame world composition is ``X_cam = D @ X_native + pred_cam_t`` with
  ``D = diag(1, -1, -1)`` (``mhr_head`` axes-1,2 flip + ``camera_head``
  translation). Composing ``T_w<-c = cam_from_world**-1`` on top of that places
  the native body in the reconstruction world.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import Tensor

import better_human as bh
from better_human.bodies import MHRClassic
from better_robot import Model
from better_robot.lie import se3, so3

_MODELS_ENV = "BETTERHUMAN_MODELS_DIR"

#: Native MHR joint names for the four climbing extremities, in output order
#: ``left_hand, right_hand, left_foot, right_foot``. MHR has no ``l_ankle`` /
#: ``r_ankle`` joints, so the ankle anchors resolve to the coincident foot
#: origins (``01_results.md``); resolution is by NAME, never by MHR70 index.
EXTREMITY_OUTPUT_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot")
_EXTREMITY_NATIVE_NAMES = ("l_wrist", "r_wrist", "l_foot", "r_foot")


def _resolve_model_path(model_path: str | None, lod: int) -> str | None:
    """Resolve the MHR archive path: explicit, then ``$BETTERHUMAN_MODELS_DIR``,
    then the sibling BetterHuman checkout (mirrors ``01_results.md``).

    Returns ``None`` when the environment root is set, letting :class:`bh.MHR`
    resolve the licensed file itself.
    """
    if model_path is not None:
        return model_path
    if os.environ.get(_MODELS_ENV):
        return None
    sibling = (
        Path(__file__).resolve().parents[3]
        / "BetterHuman" / "models" / "MHR" / "converted" / f"mhr_lod{lod}.npz"
    )
    if not sibling.is_file():
        raise FileNotFoundError(
            f"MHR LOD{lod} archive not found at {sibling}; pass model_path=... or set "
            f"${_MODELS_ENV}"
        )
    return str(sibling)


def _extremity_joint_ids(body: bh.MHR) -> Tensor:
    """Resolve the four extremity BetterRobot joint ids by native joint name.

    :returns: ``(4,)`` long tensor of BetterRobot joint ids in
        :data:`EXTREMITY_OUTPUT_NAMES` order.
    """
    native_names = body.structure.joint_names
    native_pose_ids = body.structure.native_pose_joint_indices
    if native_pose_ids is None:
        raise AssertionError("MHR must expose native_pose_joint_indices")
    ids = [int(native_pose_ids[native_names.index(name)]) for name in _EXTREMITY_NATIVE_NAMES]
    if len(set(ids)) != len(ids):
        raise AssertionError(f"extremity joints must be unique, got {ids}")
    return torch.as_tensor(ids, dtype=torch.long, device=native_pose_ids.device)


def _with_time_axis(robot: Model) -> Model:
    """Insert a singleton time axis into the shaped robot's batched value tables.

    A per-clip shaped body carries batched value tables (e.g. ``joint_placements
    [n_clips, njoints, 7]``). BetterRobot right-aligns batch axes, so evaluating
    against ``q [n_clips, T, nq]`` fails unless the values expose ``[n_clips, 1,
    ...]`` so the model broadcasts over ``T``
    (``better_robot/data_model/execution_batch.py``).
    """
    values = robot.values
    updated = {
        name: getattr(values, name).unsqueeze(-3)
        for name in ("joint_placements", "body_inertias", "frame_placements")
        if getattr(values, name).ndim > 2
    }
    return robot.with_values(**updated) if updated else robot


class MHRAdapter:
    """Map frozen MHR outputs + camera extrinsics to a world-frame ``q`` trajectory.

    :param model_path: MHR archive path; ``None`` resolves via
        ``$BETTERHUMAN_MODELS_DIR`` then the sibling BetterHuman checkout.
    :param lod: MHR level of detail (1 matches SAM-3D-Body).
    """

    def __init__(
        self,
        model_path: str | None = None,
        lod: int = 1,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.body = bh.MHR(
            _resolve_model_path(model_path, lod),
            lod=lod,
            use_expression=False,
            use_correctives=False,
            compute_mass=True,
            dtype=dtype,
            device=self.device,
        )
        #: ``(4,)`` BetterRobot joint ids for the extremities (loss gate order).
        self.extremity_joint_ids = _extremity_joint_ids(self.body)
        #: Camera-vs-native axis flip ``diag(1, -1, -1)`` (``det = +1``, a rotation).
        self._flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=self.device, dtype=dtype))

    @torch.no_grad()
    def q_from_mhr_out(
        self,
        mhr_out: dict,
        cam_from_world: Tensor,
        n_clips: int,
        seq_len: int,
    ) -> tuple[bh.MHR, Tensor]:
        """Return ``(shaped_body, q)`` in the metric reconstruction world.

        The batch is flat clip-major ``B = n_clips * seq_len``. One body is baked
        per clip from the clip's centre-frame shape/proportions (shape varies per
        frame only slightly); ``q`` is per frame. The returned body's ``robot``
        carries a singleton time axis so downstream FK/RNEA broadcast over ``T``.

        :param mhr_out: ``out["mhr"]`` from the frozen forward — reads
            ``mhr_model_params (B, 204)``, ``shape (B, 45)``, ``pred_cam_t (B, 3)``.
        :param cam_from_world: ``(B, 4, 4)`` camera-from-world (OpenCV, metric).
        :param n_clips: number of clips in the flat batch.
        :param seq_len: frames per clip ``T``.
        :returns: shaped :class:`bh.MHR` (``robot`` time-axised) and ``q``
            ``(n_clips, T, 132)`` — free-flyer ``[tx, ty, tz, qx, qy, qz, qw]``
            then 125 pose channels.
        """
        model_params = mhr_out["mhr_model_params"].detach().to(self.device, self.dtype)
        shape = mhr_out["shape"].detach().to(self.device, self.dtype)
        pred_cam_t = mhr_out["pred_cam_t"].detach().to(self.device, self.dtype)
        cam_from_world = cam_from_world.detach().to(self.device, self.dtype)
        batch = n_clips * seq_len
        if model_params.shape[0] != batch or cam_from_world.shape[0] != batch:
            raise ValueError(
                f"expected {batch} = n_clips*seq_len rows; got model_params "
                f"{model_params.shape[0]}, cam_from_world {cam_from_world.shape[0]}")

        # One body per clip from the centre frame; from_classic's q is
        # shape-independent, so per-frame pose comes from the full flat batch.
        center = model_params.view(n_clips, seq_len, -1)[:, seq_len // 2]
        shape_center = shape.view(n_clips, seq_len, -1)[:, seq_len // 2]
        body, _ = self.body.from_classic(
            MHRClassic(identity_coeffs=shape_center, model_parameters=center))
        _, q = self.body.from_classic(
            MHRClassic(identity_coeffs=shape, model_parameters=model_params))
        q = q.view(n_clips, seq_len, -1)

        root_world = self._root_to_world(
            body.robot, q[..., :7], pred_cam_t, cam_from_world, n_clips, seq_len)
        q = torch.cat((root_world, q[..., 7:]), dim=-1)
        body = body._replace(values=body.values, robot=_with_time_axis(body.robot))
        return body, q

    def total_mass(self, body: bh.MHR) -> Tensor:
        """Per-clip total mass ``(n_clips,)`` in kg from the shaped body inertias."""
        return body.robot.values.body_inertias[..., 0].sum(-1).squeeze(-1)

    def _root_to_world(
        self,
        robot: Model,
        root_native: Tensor,
        pred_cam_t: Tensor,
        cam_from_world: Tensor,
        n_clips: int,
        seq_len: int,
    ) -> Tensor:
        """Recompose the free-flyer so the native config lands in the world.

        The native-to-camera map is ``T_c<-native = (D, pred_cam_t)``; the
        world places it via ``T_w<-native = T_w<-c @ T_c<-native`` with
        ``T_w<-c = cam_from_world**-1`` (transpose form). The root free-flyer
        origin ``O = joint_placements[1]`` is non-trivial (pelvis offset), so the
        joint config is ``Q_world = O**-1 . T_w<-native . O . Q_native`` — left-
        composing ``T_w<-native`` onto every joint's world pose.

        :returns: world free-flyer block ``(n_clips, T, 7)``.
        """
        cam = cam_from_world.view(n_clips, seq_len, 4, 4)
        rot_ext = cam[..., :3, :3]
        trans_ext = cam[..., :3, 3]
        rot_w_c = rot_ext.transpose(-1, -2)                          # R_w<-c
        trans_w_c = -(rot_w_c @ trans_ext.unsqueeze(-1)).squeeze(-1)
        cam_t = pred_cam_t.view(n_clips, seq_len, 3)

        rot_w_native = rot_w_c @ self._flip
        trans_w_native = (rot_w_c @ cam_t.unsqueeze(-1)).squeeze(-1) + trans_w_c
        t_w_native = torch.cat((trans_w_native, so3.from_matrix(rot_w_native)), dim=-1)

        origin = robot.joint_placements[..., 1, :].unsqueeze(1)       # [n_clips, 1, 7]
        root_body_world = se3.compose(t_w_native, se3.compose(origin, root_native))
        return se3.compose(se3.inverse(origin), root_body_world)
