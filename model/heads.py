"""Prediction heads for the contact / force token blocks and the pose token.

Every head reads its block's final decoder tokens (``[B, K, C]``) after the
post-decoder temporal mixing and regresses the block's output. They are plain
trainable modules — the frozen base model never sees them.

* :class:`ContactHead` — one contact logit per token (a shared FFN applied
  independently along the token axis).
* :class:`ForceHead` — one 3D force vector per token, body-weight units,
  zero-initialised.
* :class:`SmplxHead` — from-scratch SMPL-X pose + camera heads on the final
  pose token, with BetterHuman FK inside.
"""
from __future__ import annotations

import math

import roma
import torch
import torch.nn as nn

from model.sam_3d_body.models.modules.transformer import FFN
from utils.geometry import (cliff_cam_to_translation, project_to_crop, ray_to_translation,
                            rot6d_to_rotmat, translation_to_ray)


def _per_token_ffn(input_dim: int, output_dims: int, mlp_depth: int,
                   mlp_channel_div_factor: int, dropout: float) -> FFN:
    """``[B, K, C] -> [B, K, output_dims]``: the same weights on every token."""
    return FFN(
        embed_dims=input_dim,
        feedforward_channels=input_dim // mlp_channel_div_factor,
        output_dims=output_dims,
        num_fcs=mlp_depth,
        ffn_drop=dropout,
        add_identity=False,
    )


def _zero_last_linear(module: nn.Module) -> None:
    final_linear = [m for m in module.modules() if isinstance(m, nn.Linear)][-1]
    nn.init.zeros_(final_linear.weight)
    nn.init.zeros_(final_linear.bias)


class ContactHead(nn.Module):
    """One contact logit per contact token: ``[B, K, C] -> [B, K]``."""

    def __init__(self, input_dim: int, mlp_depth: int = 2,
                 mlp_channel_div_factor: int = 4, dropout: float = 0.0):
        super().__init__()
        self.proj = _per_token_ffn(input_dim, 1, mlp_depth, mlp_channel_div_factor, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: contact tokens ``[B, K, C]`` -> raw logits ``[B, K]``."""
        return self.proj(x).squeeze(-1)


class ForceHead(nn.Module):
    """One 3D force vector per force token: ``[B, K, C] -> [B, K, 3]``.

    No sigmoid — the output is a dimensionless force in units of body weight.
    The final linear is zero-initialised so the model starts predicting zero
    force everywhere.
    """

    def __init__(self, input_dim: int, mlp_depth: int = 2,
                 mlp_channel_div_factor: int = 4, dropout: float = 0.0):
        super().__init__()
        self.proj = _per_token_ffn(input_dim, 3, mlp_depth, mlp_channel_div_factor, dropout)
        _zero_last_linear(self.proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: force tokens ``[B, K, C]`` -> forces ``[B, K, 3]`` (body-weight units)."""
        return self.proj(x)


class SmplxHead(nn.Module):
    """From-scratch SMPL-X readout of the pose token.

    Two FFNs of SAM 3D Body's own head shape (``C -> C -> D``, ReLU, last
    linear zero-initialised) read the FINAL pose token — the same token the
    frozen MHR and camera heads read — and regress, as a residual on a fixed
    mean, the BetterHuman SMPL-X body in the CAMERA frame:

    * ``proj_pose`` -> root orientation (6D), the 21 body-joint rotations (6D
      each, parent-local) and 10 betas. Mean = upright facing the camera
      (180 deg about x in OpenCV axes), identity joints, zero betas.
    * ``proj_cam`` -> the pelvis position in full-image camera metres, in one
      of two parametrizations (``camera``): ``cliff`` — CLIFF crop
      weak-perspective ``(s, tx, ty)``, mean ``(1, 0, 0)``, lifted with the crop
      box and the true focal (:func:`utils.geometry.cliff_cam_to_translation`);
      ``ray`` — the bearing ``(x/z, y/z)`` and the log depth about a fixed mean
      depth of :data:`RAY_MEAN_DEPTH_M`. Under ``ray`` the crop box never enters
      the lift, so nothing of the per-frame crop jitter reaches the depth.

    The root of the BetterHuman ``q`` IS the pelvis pose, so no shape-dependent
    pelvis offset couples the camera and shape outputs. With ``hands`` the
    30 finger joints (15 per hand, left then right, raw parent-local
    rotations — the classic hand mean is not part of the ``q`` convention) are
    regressed too and the body is the 52-joint build; otherwise the 22-joint
    build (fingers never move a body joint). The forward runs BetterHuman's
    differentiable FK on the assembled ``q`` and projects the joints with the
    frame intrinsics, so the output is self-contained: parameters,
    camera-frame joints, full-image and crop-normalized 2D points. The body
    models are plain tensor holders (not modules) built lazily on the input
    device; :meth:`body_flat` is always the 22-joint build (the metric
    protocol's flat-hand vertices).
    """

    NUM_BODY_JOINTS = 21
    NUM_HAND_JOINTS = 30
    NUM_BETAS = 10
    #: Mean pelvis depth (m) of the ``ray`` camera (the corpus's typical climber distance).
    RAY_MEAN_DEPTH_M = 3.5
    _MEAN_ROOT_6D = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    _MEAN_JOINT_6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    def __init__(
        self,
        input_dim: int,
        model_path: str,
        mlp_depth: int = 2,
        mlp_channel_div_factor: int = 1,
        dropout: float = 0.0,
        hands: bool = False,
        camera: str = "cliff",
    ):
        super().__init__()
        self.model_path = str(model_path)
        self.hands = bool(hands)
        if camera not in ("cliff", "ray"):
            raise ValueError(f"smplx.camera must be cliff | ray, got {camera!r}")
        self.camera = camera
        self.num_hand_joints = self.NUM_HAND_JOINTS if self.hands else 0
        n_pose = (6 + 6 * (self.NUM_BODY_JOINTS + self.num_hand_joints)
                  + self.NUM_BETAS)
        self.proj_pose = _per_token_ffn(input_dim, n_pose, mlp_depth,
                                        mlp_channel_div_factor, dropout)
        self.proj_cam = _per_token_ffn(input_dim, 3, mlp_depth,
                                       mlp_channel_div_factor, dropout)
        # Zero-init the final linears -> the mean body at init, learn a delta.
        _zero_last_linear(self.proj_pose)
        _zero_last_linear(self.proj_cam)
        mean_pose = torch.tensor(
            list(self._MEAN_ROOT_6D)
            + list(self._MEAN_JOINT_6D) * (self.NUM_BODY_JOINTS + self.num_hand_joints)
            + [0.0] * self.NUM_BETAS)
        self.register_buffer("mean_pose", mean_pose, persistent=False)
        mean_cam = ([1.0, 0.0, 0.0] if camera == "cliff"
                    else [0.0, 0.0, math.log(self.RAY_MEAN_DEPTH_M)])
        self.register_buffer("mean_cam", torch.tensor(mean_cam), persistent=False)
        self._bodies: dict = {}

    @property
    def num_joints(self) -> int:
        """Joints in ``joints_cam``: 22, or 52 with hands."""
        return 1 + self.NUM_BODY_JOINTS + self.num_hand_joints

    def _body(self, device: torch.device, hands: bool):
        key = (device, hands)
        if key not in self._bodies:
            import better_human as bh
            self._bodies[key] = bh.SMPLX(
                model_path=self.model_path, gender="neutral", num_betas=self.NUM_BETAS,
                use_hands=hands, use_face=False, compute_mass=False,
                dtype=torch.float32, device=device)
        return self._bodies[key]

    def body(self, device: torch.device):
        """The head's BetterHuman SMPL-X body on ``device`` (22 or 52 joints; built once)."""
        return self._body(device, self.hands)

    def body_flat(self, device: torch.device):
        """The 22-joint body on ``device`` — flat-hand vertices for the metrics."""
        return self._body(device, False)

    def forward(
        self,
        pose_token: torch.Tensor,
        *,
        bbox_center: torch.Tensor,
        bbox_size: torch.Tensor,
        cam_int: torch.Tensor,
        affine_trans: torch.Tensor,
        img_size: torch.Tensor,
    ) -> dict:
        """
        Args:
            pose_token: final decoder pose token ``[B, C]``
            bbox_center: crop-box centre ``[B, 2]`` full-image px
            bbox_size: square crop-box side ``[B]`` full-image px
            cam_int: full-frame intrinsics ``[B, 3, 3]``
            affine_trans: full -> crop affine ``[B, 2, 3]``
            img_size: crop size ``[B, 2]``

        Returns:
            ``root_6d [B, 6]``, ``body_6d [B, 21, 6]`` (raw, mean included),
            ``hand_6d [B, 30, 6]`` (``None`` without hands), ``root_rot [B, 3, 3]``
            camera-from-root, ``body_rot [B, 21, 3, 3]``, ``hand_rot [B, 30, 3, 3]``
            (``None`` without hands), ``betas [B, 10]``, ``cam [B, 3]`` (s, tx, ty;
            ``None`` under ``camera: ray``), ``ray [B, 3]`` (x/z, y/z, log z of the
            pelvis), ``pelvis_cam [B, 3]`` m, ``q_cam [B, 91 | 211]`` BetterHuman
            configuration (body first, finger quaternions last), ``joints_cam
            [B, J, 3]`` m with ``J`` = :attr:`num_joints` (22 body joints first),
            ``kp2d_full [B, J, 2]`` px, ``kp2d_crop [B, J, 2]`` in ``[-0.5, 0.5]``.
        """
        pose_token = pose_token.float()
        batch = pose_token.shape[0]
        pose = self.proj_pose(pose_token) + self.mean_pose
        cam_out = self.proj_cam(pose_token) + self.mean_cam
        n_body, n_hand = 6 * self.NUM_BODY_JOINTS, 6 * self.num_hand_joints
        cam = None
        if self.camera == "cliff":
            cam = cam_out
            pelvis_cam = cliff_cam_to_translation(
                cam, bbox_center.float(), bbox_size.float(), cam_int.float())
        else:
            pelvis_cam = ray_to_translation(cam_out)
        root_6d = pose[:, :6]
        body_6d = pose[:, 6:6 + n_body].reshape(batch, self.NUM_BODY_JOINTS, 6)
        betas = pose[:, 6 + n_body + n_hand:]
        root_rot = rot6d_to_rotmat(root_6d)
        body_rot = rot6d_to_rotmat(body_6d)
        quats = [
            pelvis_cam,
            roma.rotmat_to_unitquat(root_rot),                            # xyzw
            roma.rotmat_to_unitquat(body_rot).reshape(batch, -1),
        ]
        hand_6d = hand_rot = None
        if self.hands:
            hand_6d = pose[:, 6 + n_body:6 + n_body + n_hand].reshape(
                batch, self.NUM_HAND_JOINTS, 6)
            hand_rot = rot6d_to_rotmat(hand_6d)
            quats.append(roma.rotmat_to_unitquat(hand_rot).reshape(batch, -1))
        q_cam = torch.cat(quats, dim=-1)                                  # [B, 91 | 211]
        shaped = self.body(pose_token.device).with_shape(betas=betas)
        joints_cam = shaped.fk(q_cam).joint_pose_world[..., 1:, :3]      # drop the universe row
        kp2d_full, kp2d_crop = project_to_crop(
            joints_cam, cam_int.float(), affine_trans.float(), img_size.float())
        return {
            "root_6d": root_6d, "body_6d": body_6d, "hand_6d": hand_6d,
            "root_rot": root_rot, "body_rot": body_rot, "hand_rot": hand_rot,
            "betas": betas, "cam": cam, "ray": translation_to_ray(pelvis_cam),
            "pelvis_cam": pelvis_cam, "q_cam": q_cam,
            "joints_cam": joints_cam, "kp2d_full": kp2d_full, "kp2d_crop": kp2d_crop,
        }
