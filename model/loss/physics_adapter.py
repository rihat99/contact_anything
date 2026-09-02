"""Bridge the frozen SAM-3D-Body outputs onto a BetterHuman MHR body.

The single place that knows how the frozen model's per-frame MHR parameters map
onto BetterHuman's ``MHRClassic`` compact vector, and how per-frame camera
extrinsics place every frame's body in one static metric reconstruction world.
Everything here runs under :func:`torch.no_grad`: the physics loss receives the
configuration ``q`` (and the shaped body) as constants — only the predicted
forces carry gradients.

Two facts, both verified against the vendored model on real checkpoint outputs,
fix the mapping:

* The 204-slot ``mhr_model_params`` vector fed to SAM's MHR *is* Momentum's
  ``compact_v6`` vector, so ``MHRClassic(model_parameters=mhr_model_params)``
  reproduces SAM's native joint centres bit-for-bit (0.0 mm) — no permutation.
* SAM's per-frame world composition is ``X_cam = D @ X_native + pred_cam_t``
  with ``D = diag(1, -1, -1)`` (``mhr_head`` axes-1,2 flip + ``camera_head``
  translation). Composing ``T_w<-c = cam_from_world**-1`` on top of that places
  the native body in the metric reconstruction world.

The six kindyn force/contact groups anchor onto MHR joints BY NAME (never by
index): the hands on the wrists, the toe groups on the ball-of-foot joints, the
heel groups on the foot origins (MHR has no separate ankle joint — the foot
origin sits at the ankle, 6 cm behind and 5 cm above the ball).

Lengths are metres, masses kilograms; ``q`` is a free-flyer
``[tx, ty, tz, qx, qy, qz, qw]`` followed by 125 revolute channels.
"""
from __future__ import annotations


import torch
from torch import Tensor

import better_human as bh
from better_human.bodies import MHRClassic
from better_robot import Model
from better_robot.lie import se3, so3

from model.loss import KINDYN_GROUP_NAMES
from utils.betterhuman import resolve_mhr_archive

GROUP_NATIVE_JOINTS = (
    "l_wrist", "r_wrist", "l_ball", "r_ball", "l_foot", "r_foot",
)

#: Native name of the MHR free-flyer root joint (the pelvis body).
_ROOT_NATIVE_JOINT = "root"


def _native_joint_id(body: bh.MHR, name: str) -> int:
    """BetterRobot joint id of the MHR joint called ``name``."""
    native_names = body.structure.joint_names
    native_pose_ids = body.structure.native_pose_joint_indices
    if native_pose_ids is None:
        raise RuntimeError("MHR must expose native_pose_joint_indices")
    if name not in native_names:
        raise ValueError(f"MHR rig has no joint named {name!r}")
    return int(native_pose_ids[native_names.index(name)])


def _group_joint_ids(body: bh.MHR) -> Tensor:
    """Resolve the six kindyn-group BetterRobot joint ids by native joint name.

    :returns: ``(6,)`` long tensor of BetterRobot joint ids in
        :data:`~model.loss.KINDYN_GROUP_NAMES` order.
    """
    if len(GROUP_NATIVE_JOINTS) != len(KINDYN_GROUP_NAMES):
        raise RuntimeError("one MHR joint per kindyn group")
    ids = [_native_joint_id(body, name) for name in GROUP_NATIVE_JOINTS]
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"kindyn group joints must be distinct, got {dict(zip(GROUP_NATIVE_JOINTS, ids))}")
    device = body.structure.native_pose_joint_indices.device
    return torch.as_tensor(ids, dtype=torch.long, device=device)


def _with_time_axis(robot: Model) -> Model:
    """Insert a singleton time axis into the shaped robot's batched value tables.

    A per-clip shaped body carries batched value tables (e.g. ``joint_placements
    [n_clips, njoints, 7]``). BetterRobot right-aligns batch axes, so evaluating
    against ``q [n_clips, T, nq]`` fails unless the values expose ``[n_clips, 1,
    ...]`` so the model broadcasts over ``T``.
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
            resolve_mhr_archive(model_path, lod),
            lod=lod,
            use_expression=False,
            use_correctives=False,
            compute_mass=True,
            dtype=dtype,
            device=self.device,
        )
        #: ``(6,)`` BetterRobot joint ids for the kindyn groups (loss gate order).
        self.group_joint_ids = _group_joint_ids(self.body)
        #: BetterRobot joint id of the free-flyer root (world-from-root rotation).
        self.root_joint_id = _native_joint_id(self.body, _ROOT_NATIVE_JOINT)
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

        :param mhr_out: ``out["mhr"]`` from the forward — reads
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


__all__ = ["MHRAdapter", "GROUP_NATIVE_JOINTS"]
