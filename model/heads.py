"""Prediction heads for the contact / force / motion token blocks and the pose token.

Every head reads its block's final decoder tokens (``[B, K, C]``) after the
post-decoder temporal mixing and regresses the block's output. They are plain
trainable modules — the frozen base model never sees them.

* :class:`ContactHead` — contact logits (``per_token`` / ``attention`` /
  ``concat`` pooling).
* :class:`ForceHead` — one 3D force vector per token, body-weight units,
  zero-initialised.
* :class:`MotionHead` — standardized vel/acc (+ optional root angular pair)
  per token, zero-initialised.
* :func:`contact_gate_forces` — the parameter-free contact gate multiplying
  the final force output by the (detached) per-group contact probabilities.
* :class:`SmplxHead` — the pose-token probe: from-scratch SMPL-X pose + CLIFF
  camera heads on the final pose token, with BetterHuman FK inside.
"""
from __future__ import annotations

import roma
import torch
import torch.nn as nn

from model.sam_3d_body.models.modules.transformer import FFN
from utils.geometry import cliff_cam_to_translation, project_to_crop, rot6d_to_rotmat


class ContactHead(nn.Module):
    """Predict contact logits from a bank of contact query tokens.

    Four output modes are supported:
        - "attention": Learnable query attends over tokens -> [B, 1, C], then MLP.
        - "concat": Flatten all tokens -> [B, num_tokens * C], project down to C,
          then MLP.  Preserves per-token information without the bottleneck of
          compressing everything into a single attention query, while keeping
          parameter count manageable via the linear projection.
        - "per_token": Apply one shared MLP independently to every token, producing
          one logit per token.  This mode requires the output dimension to equal the
          total number of contact tokens.
        - "flat": One MLP straight from the flattened tokens
          ``[B, num_tokens * C]`` to ``output_dims`` — the same depth as
          ``per_token`` (no extra projection layer), so a head reading the single
          POSE token (``[B, 1, C] -> [B, 6]``) is directly comparable with the
          per-token head over six contact tokens.

    ``attention``, ``concat`` and ``flat`` can emit an arbitrary target dimension
    (for example an entire body-22 or vertex target). ``per_token`` preserves
    token identity and is intended for explicitly aligned contact-token targets.
    """

    def __init__(
        self,
        input_dim: int,
        num_contact_tokens: int,
        output_dims: int,
        mlp_depth: int = 2,
        mlp_channel_div_factor: int = 4,
        pool_num_heads: int = 8,
        pool_mode: str = "attention",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_contact_tokens = num_contact_tokens
        self.output_dims = output_dims
        self.pool_mode = pool_mode

        if pool_mode == "per_token":
            if output_dims != num_contact_tokens:
                raise ValueError(
                    "pool_mode='per_token' requires output_dims to equal the total "
                    f"contact-token count; got output_dims={output_dims}, "
                    f"num_contact_tokens={num_contact_tokens}"
                )
            # FFN operates on the last dimension, so the same weights are applied
            # independently to every token: [B, K, C] -> [B, K, 1].
            self.proj = FFN(
                embed_dims=input_dim,
                feedforward_channels=input_dim // mlp_channel_div_factor,
                output_dims=1,
                num_fcs=mlp_depth,
                ffn_drop=dropout,
                add_identity=False,
            )
        elif pool_mode == "attention":
            self.pool_query = nn.Parameter(torch.zeros(1, 1, input_dim))
            nn.init.trunc_normal_(self.pool_query, std=0.02)
            self.pool_attn = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=pool_num_heads,
                batch_first=True,
            )
            mlp_input_dim = input_dim
        elif pool_mode == "concat":
            concat_dim = num_contact_tokens * input_dim
            self.concat_proj = nn.Sequential(
                nn.Linear(concat_dim, input_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            mlp_input_dim = input_dim
        elif pool_mode == "flat":
            mlp_input_dim = num_contact_tokens * input_dim
        else:
            raise ValueError(f"Unknown pool_mode: {pool_mode!r}")

        if pool_mode != "per_token":
            self.proj = FFN(
                embed_dims=mlp_input_dim,
                feedforward_channels=input_dim // mlp_channel_div_factor,
                output_dims=output_dims,
                num_fcs=mlp_depth,
                ffn_drop=dropout,
                add_identity=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: contact tokens  [B, num_contact_tokens, C]

        Returns:
            contact_logits: [B, output_dims] (raw, before sigmoid)
        """
        if self.pool_mode == "per_token":
            if x.shape[1] != self.num_contact_tokens:
                raise ValueError(
                    "per-token input token count does not match the configured head: "
                    f"input={x.shape[1]}, configured={self.num_contact_tokens}"
                )
            contact_logits = self.proj(x).squeeze(-1)  # [B, K]
        elif self.pool_mode == "attention":
            batch_size = x.shape[0]
            query = self.pool_query.expand(batch_size, -1, -1)
            x_pooled, _ = self.pool_attn(query, x, x)  # [B, 1, C]
            contact_logits = self.proj(x_pooled).squeeze(1)
        elif self.pool_mode == "flat":
            if x.shape[1] != self.num_contact_tokens:
                raise ValueError(
                    "flat-head input token count does not match the configured head: "
                    f"input={x.shape[1]}, configured={self.num_contact_tokens}"
                )
            contact_logits = self.proj(x.flatten(1))  # [B, output_dims]
        else:  # concat
            x_flat = x.flatten(1)            # [B, num_tokens * C]
            x_proj = self.concat_proj(x_flat)  # [B, C]
            contact_logits = self.proj(x_proj)  # [B, output_dims]

        return contact_logits


FORCE_GATE_CONTACT_MAP = (0, 1, 2, 3, 4, 5)


def contact_gate_forces(
    joint_forces: torch.Tensor, contact_logits: torch.Tensor, sharpness: float
) -> torch.Tensor:
    """Gate the final force output by the (detached) per-group contact logits.

    ``gate_k = sigmoid(sharpness * contact_logits[:, MAP[k]])`` — nearly 1 for a
    confident contact (logit 1.0 -> gate 0.982 at sharpness 4.0), nearly 0 for a
    confident free limb. The logits are detached unconditionally: the force loss
    must not rewrite the calibrated contact probabilities through this product
    (contact trains purely from its labels).

    Args:
        joint_forces: raw force output ``[B, 6, 3]`` (kindyn group order).
        contact_logits: kindyn_6 contact logits ``[B, 6]`` (same group order).
        sharpness: gate steepness multiplier on the logits.

    Returns:
        gated forces ``[B, 6, 3]``.
    """
    if joint_forces.shape[1] != len(FORCE_GATE_CONTACT_MAP):
        raise ValueError(
            "contact-gated forces need one group per gate-map entry: "
            f"forces={joint_forces.shape[1]}, map={len(FORCE_GATE_CONTACT_MAP)}"
        )
    if contact_logits.shape[-1] <= max(FORCE_GATE_CONTACT_MAP):
        raise ValueError(
            "contact-gate map indexes contact output "
            f"{max(FORCE_GATE_CONTACT_MAP)} but the contact head produced only "
            f"{contact_logits.shape[-1]} logits"
        )
    gate = torch.sigmoid(
        sharpness * contact_logits.detach()[:, list(FORCE_GATE_CONTACT_MAP)]
    )  # [B, 6]
    return joint_forces * gate.unsqueeze(-1)


class ForceHead(nn.Module):
    """Regress one 3D force vector per force token.

    Mirrors :class:`ContactHead`'s ``per_token`` branch: a single FFN applied
    independently along the token axis, mapping ``[B, K, C] -> [B, K, 3]``. No
    sigmoid — the output is a dimensionless force in units of body weight (the
    physics loss converts to newtons).

    The final linear is zero-initialised so the model starts predicting zero
    force everywhere; the RNEA residual then reduces to the pure-kinematics
    baseline before any force is learned (a meaningful curriculum start).
    """

    def __init__(
        self,
        input_dim: int,
        num_force_tokens: int,
        mlp_depth: int = 2,
        mlp_channel_div_factor: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_force_tokens = num_force_tokens

        # FFN operates on the last dimension, so the same weights are applied
        # independently to every token: [B, K, C] -> [B, K, 3].
        self.proj = FFN(
            embed_dims=input_dim,
            feedforward_channels=input_dim // mlp_channel_div_factor,
            output_dims=3,
            num_fcs=mlp_depth,
            ffn_drop=dropout,
            add_identity=False,
        )

        # Zero-init the final linear -> zero force at init (D5).
        final_linear = [m for m in self.proj.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: force tokens  ``[B, num_force_tokens, C]``

        Returns:
            joint_forces: ``[B, num_force_tokens, 3]`` (dimensionless, body-weight units)
        """
        if x.shape[1] != self.num_force_tokens:
            raise ValueError(
                "force-head input token count does not match the configured head: "
                f"input={x.shape[1]}, configured={self.num_force_tokens}"
            )
        return self.proj(x)  # [B, K, 3]


class MotionHead(nn.Module):
    """Regress velocity + acceleration per motion token.

    Mirrors :class:`ForceHead`: a single FFN applied independently along the
    token axis, mapping ``[B, K, C] -> [B, K, output_dims]``. The outputs are
    the **standardized** root-frame linear velocity (``[..., 0:3]``) and linear
    acceleration (``[..., 3:6]``) of the token's kindyn joint — plus, when
    ``output_dims`` is 12, the root body angular velocity (``[..., 6:9]``) and
    angular acceleration (``[..., 9:12]``); the supervised loss owns the
    mean/std table, so the head never sees physical units.

    The final linear is zero-initialised, so at init every token predicts the
    standardized mean (the dataset's average velocity/acceleration) — the same
    curriculum start the force head uses.
    """

    def __init__(
        self,
        input_dim: int,
        num_motion_tokens: int,
        mlp_depth: int = 2,
        mlp_channel_div_factor: int = 4,
        dropout: float = 0.0,
        output_dims: int = 6,
    ):
        super().__init__()

        self.num_motion_tokens = num_motion_tokens

        # FFN operates on the last dimension, so the same weights are applied
        # independently to every token: [B, K, C] -> [B, K, output_dims].
        self.proj = FFN(
            embed_dims=input_dim,
            feedforward_channels=input_dim // mlp_channel_div_factor,
            output_dims=output_dims,
            num_fcs=mlp_depth,
            ffn_drop=dropout,
            add_identity=False,
        )

        # Zero-init the final linear -> standardized-mean prediction at init.
        final_linear = [m for m in self.proj.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: motion tokens ``[B, num_motion_tokens, C]``

        Returns:
            motion: ``[B, num_motion_tokens, output_dims]`` — standardized
            ``(vel, acc[, ang_vel, ang_acc])`` in root axes.
        """
        if x.shape[1] != self.num_motion_tokens:
            raise ValueError(
                "motion-head input token count does not match the configured head: "
                f"input={x.shape[1]}, configured={self.num_motion_tokens}"
            )
        return self.proj(x)  # [B, K, 6|12]


class SmplxHead(nn.Module):
    """From-scratch SMPL-X readout of the frozen pose token (the pose-token probe).

    Two FFNs of SAM 3D Body's own head shape (``C -> C -> D``, ReLU, last
    linear zero-initialised) read the FINAL pose token — the same token the
    frozen MHR and camera heads read — and regress, as a residual on a fixed
    mean, the BetterHuman SMPL-X body in the CAMERA frame:

    * ``proj_pose`` -> root orientation (6D), the 21 body-joint rotations (6D
      each, parent-local) and 10 betas. Mean = upright facing the camera
      (180 deg about x in OpenCV axes), identity joints, zero betas.
    * ``proj_cam`` -> CLIFF crop weak-perspective ``(s, tx, ty)``, mean
      ``(1, 0, 0)``, lifted with the crop box and the true focal to the pelvis
      position in full-image camera metres (:func:`utils.geometry.cliff_cam_to_translation`).

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
    ):
        super().__init__()
        self.model_path = str(model_path)
        self.hands = bool(hands)
        self.num_hand_joints = self.NUM_HAND_JOINTS if self.hands else 0
        n_pose = (6 + 6 * (self.NUM_BODY_JOINTS + self.num_hand_joints)
                  + self.NUM_BETAS)
        self.proj_pose = FFN(
            embed_dims=input_dim,
            feedforward_channels=input_dim // mlp_channel_div_factor,
            output_dims=n_pose,
            num_fcs=mlp_depth,
            ffn_drop=dropout,
            add_identity=False,
        )
        self.proj_cam = FFN(
            embed_dims=input_dim,
            feedforward_channels=input_dim // mlp_channel_div_factor,
            output_dims=3,
            num_fcs=mlp_depth,
            ffn_drop=dropout,
            add_identity=False,
        )
        # Zero-init the final linears -> the mean body at init, learn a delta.
        for proj in (self.proj_pose, self.proj_cam):
            final_linear = [m for m in proj.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)
        mean_pose = torch.tensor(
            list(self._MEAN_ROOT_6D)
            + list(self._MEAN_JOINT_6D) * (self.NUM_BODY_JOINTS + self.num_hand_joints)
            + [0.0] * self.NUM_BETAS)
        self.register_buffer("mean_pose", mean_pose, persistent=False)
        self.register_buffer("mean_cam", torch.tensor([1.0, 0.0, 0.0]), persistent=False)
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
            (``None`` without hands), ``betas [B, 10]``, ``cam [B, 3]`` (s, tx, ty),
            ``pelvis_cam [B, 3]`` m, ``q_cam [B, 91 | 211]`` BetterHuman
            configuration (body first, finger quaternions last), ``joints_cam
            [B, J, 3]`` m with ``J`` = :attr:`num_joints` (22 body joints first),
            ``kp2d_full [B, J, 2]`` px, ``kp2d_crop [B, J, 2]`` in ``[-0.5, 0.5]``.
        """
        pose_token = pose_token.float()
        batch = pose_token.shape[0]
        pose = self.proj_pose(pose_token) + self.mean_pose
        cam = self.proj_cam(pose_token) + self.mean_cam
        n_body, n_hand = 6 * self.NUM_BODY_JOINTS, 6 * self.num_hand_joints
        root_6d = pose[:, :6]
        body_6d = pose[:, 6:6 + n_body].reshape(batch, self.NUM_BODY_JOINTS, 6)
        betas = pose[:, 6 + n_body + n_hand:]
        root_rot = rot6d_to_rotmat(root_6d)
        body_rot = rot6d_to_rotmat(body_6d)
        pelvis_cam = cliff_cam_to_translation(
            cam, bbox_center.float(), bbox_size.float(), cam_int.float())
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
            "betas": betas, "cam": cam, "pelvis_cam": pelvis_cam, "q_cam": q_cam,
            "joints_cam": joints_cam, "kp2d_full": kp2d_full, "kp2d_crop": kp2d_crop,
        }
