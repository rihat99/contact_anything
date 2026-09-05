"""ContactAnything — the composed trainable model over the frozen SAM-3D-Body.

Assembly (everything trainable lives HERE, nothing trainable inside the
wrapper):

1. :class:`~model.wrapper.SAM3DBodyWrapper` runs the frozen body decoder with
   our learned token blocks (:class:`~model.tokens.LearnedTokenBlock`)
   appended behind the asymmetric mask — original tokens never attend them, so
   the frozen pose/MHR outputs have an exactly-zero Jacobian w.r.t. every
   parameter of this module unless a pose-writing brick below is enabled.
2. :class:`~model.inputs.KeypointInput` (optional) — the frame's 2D keypoints
   (corpus sapiens detections, or at deployment the frozen readout's own
   ``pred_keypoints_2d``) crop-normalised and ADDED to the pose token through
   a zero-init MLP; :class:`~model.inputs.TwistInput` (optional) likewise adds
   the camera's own twist to the motion tokens (or to the pose token when the
   motion head reads that); :class:`~model.inputs.BboxInput` (optional) adds
   the CLIFF bbox vector to the pose token; :class:`~model.inputs.TokenMasking`
   (optional, train only) then corrupts random spans of frames of the temporal
   block's input.
3. :class:`~model.rope.CrossModalRopeModule` (optional) — ONE post-decoder
   RoPE temporal transformer over the concatenation of the chosen modality
   token blocks, across the clip's frames (within-frame mixing is its
   ``dt = 0`` diagonal). A single listed modality (``[pose]``) is legal: the
   block is then a plain temporal transformer over that token block.
4. :class:`~model.rope.RopeTemporalModule` (optional, ``pose_temporal``) —
   the pose token's own temporal block, run after the cross-modal brick.
5. Final readout: when anything wrote the pose token (cross-modal with
   ``pose`` listed, ``pose_temporal``, the keypoint input) or a split-head
   fine-tune copy exists,
   the final MHR + camera output is recomputed from the updated token via
   ``wrapper.decode_pose`` — through the trainable projection copies when
   enabled — while the frozen forward's own readout supplies the
   ``*_frozen`` trust-region anchors. With an :class:`~model.heads.SmplxHead`
   the SMPL-X body IS the pose output and the MHR recompute is skipped:
   ``out["mhr"]`` then stays the frozen model's own readout (the config layer
   forbids the MHR-consuming losses in that build).
6. Heads (:mod:`model.heads`) read each block's final tokens: contact logits
   per target (from the contact tokens, or — ``contact.source: pose_token`` —
   straight from the final pose token with no learned tokens at all),
   per-token 3D forces (optionally gated by the detached contact logits),
   standardized root-twist motion (from the motion tokens, or —
   ``motion.source: pose_token`` — a flat head on the final pose token). The
   optional :class:`~model.heads.SmplxHead`
   reads the final POSE token (index 0) and regresses an SMPL-X body from
   scratch; it never writes the token.

The sub-configs are plain dicts mirroring the experiment yaml sections; the
config layer maps yaml -> these kwargs 1:1.
"""
from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn

from model.heads import (
    FORCE_GATE_CONTACT_MAP,
    ContactHead,
    ForceHead,
    MotionHead,
    SmplxHead,
    contact_gate_forces,
)
from model.inputs import (BboxInput, FrozenCameraInput, KeypointInput, TokenMasking,
                          TwistInput)
from model.rope import CrossModalRopeModule, RopeTemporalModule
from model.tokens import LearnedTokenBlock
from model.wrapper import SAM3DBodyWrapper
from utils.geometry import frozen_pelvis_camera

_MODALITY_ORDER = ("pose", "contact", "force", "motion")
#: The motion head's possible output triples, in channel order.
MOTION_TERM_NAMES = ("vel", "acc", "ang_vel", "ang_acc")


class ContactAnything(nn.Module):
    """Frozen SAM-3D-Body + trainable contact / force / motion branches.

    :param wrapper: the frozen base (registered as a submodule; all its params
        stay ``requires_grad=False`` and eval-pinned).
    :param contact: contact branch config or ``None`` —
        ``{source: "tokens" | "pose_token", keypoint_indices,
        num_global_tokens, targets: {name: output_dims}, pool_mode, mlp_depth,
        mlp_channel_div_factor, dropout, grid_size, grid_radius,
        blind_to_image}``. ``source: pose_token`` builds no token block: the
        heads read the final pose token (``[B, 1, C]``) instead.
    :param force: force branch config or ``None`` —
        ``{keypoint_indices (None = inherit contact anchors), mlp_depth,
        mlp_channel_div_factor, dropout,
        contact_gate: {enabled, sharpness}}``.
    :param motion: motion branch config or ``None`` —
        ``{source: "tokens" | "pose_token", keypoint_indices, anchored, terms,
        mlp_depth, mlp_channel_div_factor, dropout}``; the head emits 3
        channels per listed term (``vel``/``acc``/``ang_vel``/``ang_acc``).
    :param cross_modal: cross-modal temporal config or ``None`` —
        ``{modalities, num_layers, num_heads, mlp_ratio, dropout, time_scale,
        max_rel_sec}``.
    :param pose_temporal: pose-token temporal config or ``None`` — same keys
        minus ``modalities``.
    :param finetune_pose_head / finetune_camera_head: build trainable COPIES
        of the frozen projection FFNs, applied to the FINAL readout only.
    :param extra_token_attention: ``"mutual" | "causal"`` decoder mask regime
        among the appended blocks.
    :param smplx: SMPL-X head config or ``None`` — ``{model_path, hands,
        mlp_depth, mlp_channel_div_factor, dropout, camera, depth_prior,
        depth_prior_m}``; a from-scratch pose + camera readout of the final
        pose token (reads it, never writes it).
    :param keypoints2d: keypoint-input config or ``None`` — ``{source:
        "sapiens" | "mhr", indices (MHR70), min_score, noise_std, drop_prob}``;
        added to the pose token before the temporal bricks (a pose writer).
    :param token_masking: train-only span corruption of the cross-modal block's
        input or ``None`` — ``{frac, span_min, span_max, replace: "mask" |
        "swap"}``; requires ``cross_modal``.
    :param camera_twist: ``{}`` to add the camera's own twist (``cam_twist``)
        to the motion tokens — or to the pose token when the motion head reads
        the pose token / no motion branch exists (then a pose writer).
    :param bbox: ``{}`` to add the CLIFF bbox vector (``bbox_center``,
        ``bbox_scale``, ``cam_int``) to the pose token before the temporal
        bricks (a pose writer).
    :param frozen_camera: ``{}`` to add the frozen readout's own pelvis ray
        (:class:`~model.inputs.FrozenCameraInput`) to the pose token before
        the temporal bricks (a pose writer).
    """

    def __init__(
        self,
        wrapper: SAM3DBodyWrapper,
        contact: Optional[dict] = None,
        force: Optional[dict] = None,
        motion: Optional[dict] = None,
        cross_modal: Optional[dict] = None,
        pose_temporal: Optional[dict] = None,
        finetune_pose_head: bool = False,
        finetune_camera_head: bool = False,
        extra_token_attention: str = "mutual",
        smplx: Optional[dict] = None,
        keypoints2d: Optional[dict] = None,
        token_masking: Optional[dict] = None,
        camera_twist: Optional[dict] = None,
        bbox: Optional[dict] = None,
        frozen_camera: Optional[dict] = None,
    ):
        super().__init__()
        assert extra_token_attention in ("mutual", "causal")
        self.wrapper = wrapper
        self.extra_token_attention = extra_token_attention
        dim = wrapper.decoder_dim
        backbone_dim = wrapper.backbone_dim

        # --- contact branch ---
        self.contact_tokens = None
        self.head_contact = None
        if contact is not None:
            source = str(contact["source"])
            assert source in ("tokens", "pose_token"), (
                f"contact.source must be 'tokens' or 'pose_token'; got {source!r}")
            if source == "tokens":
                self.contact_tokens = LearnedTokenBlock(
                    "contact", dim, backbone_dim,
                    keypoint_indices=contact["keypoint_indices"],
                    num_global_tokens=int(contact["num_global_tokens"]),
                    blind_to_image=bool(contact["blind_to_image"]),
                    grid_size=int(contact["grid_size"]),
                    grid_radius=float(contact["grid_radius"]),
                )
            head_tokens = (self.contact_tokens.num_tokens
                           if self.contact_tokens is not None else 1)
            self.head_contact = nn.ModuleDict({
                str(name): ContactHead(
                    input_dim=dim,
                    num_contact_tokens=head_tokens,
                    output_dims=int(dims),
                    mlp_depth=int(contact["mlp_depth"]),
                    mlp_channel_div_factor=int(contact["mlp_channel_div_factor"]),
                    pool_mode=str(contact["pool_mode"]),
                    dropout=float(contact["dropout"]),
                )
                for name, dims in contact["targets"].items()
            })

        # --- force branch ---
        self.force_tokens = None
        self.head_force = None
        self.force_contact_gate = False
        if force is not None:
            force_kp = force["keypoint_indices"]
            if force_kp is None:
                assert contact is not None, (
                    "force.keypoint_indices=None inherits the contact anchors "
                    "and therefore requires the contact branch")
                force_kp = contact["keypoint_indices"]
            self.force_tokens = LearnedTokenBlock(
                "force", dim, backbone_dim,
                keypoint_indices=force_kp,
                grid_size=int(contact["grid_size"]) if contact else 1,
                grid_radius=float(contact["grid_radius"]) if contact else 0.1,
            )
            self.head_force = ForceHead(
                input_dim=dim,
                num_force_tokens=self.force_tokens.num_tokens,
                mlp_depth=int(force["mlp_depth"]),
                mlp_channel_div_factor=int(force["mlp_channel_div_factor"]),
                dropout=float(force["dropout"]),
            )
            gate = force["contact_gate"]
            self.force_contact_gate = bool(gate["enabled"])
            self.force_contact_gate_sharpness = float(gate["sharpness"])
            if self.force_contact_gate:
                assert self.head_contact is not None and "joint" in self.head_contact, (
                    "force.contact_gate reads the per-group 'joint' contact logits")
                assert self.force_tokens.num_tokens == len(FORCE_GATE_CONTACT_MAP), (
                    f"force.contact_gate requires {len(FORCE_GATE_CONTACT_MAP)} "
                    f"force tokens (kindyn groups); got {self.force_tokens.num_tokens}")

        # --- motion branch ---
        self.motion_tokens = None
        self.head_motion = None
        self.motion_terms: list[str] = []
        if motion is not None:
            source = str(motion["source"])
            assert source in ("tokens", "pose_token"), (
                f"motion.source must be 'tokens' or 'pose_token'; got {source!r}")
            self.motion_terms = [str(t) for t in motion["terms"]]
            assert self.motion_terms and all(
                t in MOTION_TERM_NAMES for t in self.motion_terms), (
                f"motion.terms must be a non-empty subset of {MOTION_TERM_NAMES}; "
                f"got {self.motion_terms}")
            if source == "tokens":
                self.motion_tokens = LearnedTokenBlock(
                    "motion", dim, backbone_dim,
                    keypoint_indices=motion["keypoint_indices"],
                    anchored=bool(motion["anchored"]),
                    grid_size=int(contact["grid_size"]) if contact else 1,
                    grid_radius=float(contact["grid_radius"]) if contact else 0.1,
                )
            self.head_motion = MotionHead(
                input_dim=dim,
                num_motion_tokens=(self.motion_tokens.num_tokens
                                   if self.motion_tokens is not None else 1),
                mlp_depth=int(motion["mlp_depth"]),
                mlp_channel_div_factor=int(motion["mlp_channel_div_factor"]),
                dropout=float(motion["dropout"]),
                output_dims=3 * len(self.motion_terms),
            )

        # --- SMPL-X pose-token probe ---
        self.head_smplx = None
        if smplx is not None:
            self.head_smplx = SmplxHead(
                input_dim=dim,
                model_path=str(smplx["model_path"]),
                mlp_depth=int(smplx["mlp_depth"]),
                mlp_channel_div_factor=int(smplx["mlp_channel_div_factor"]),
                dropout=float(smplx["dropout"]),
                hands=bool(smplx["hands"]),
                camera=str(smplx["camera"]),
                depth_prior=str(smplx["depth_prior"]),
                depth_prior_m=float(smplx["depth_prior_m"]),
            )

        # --- keypoint input (pose token) ---
        self.keypoint_input = None
        if keypoints2d is not None:
            self.keypoint_source = str(keypoints2d["source"])
            assert self.keypoint_source in ("sapiens", "mhr"), (
                f"keypoints2d.source must be 'sapiens' or 'mhr'; got "
                f"{self.keypoint_source!r}")
            indices = [int(i) for i in keypoints2d["indices"]]
            assert indices and all(0 <= i < 70 for i in indices), (
                f"keypoints2d.indices must be MHR70 indices; got {indices}")
            self.register_buffer(
                "keypoint_indices", torch.tensor(indices), persistent=False)
            self.keypoint_input = KeypointInput(
                dim, len(indices),
                min_score=float(keypoints2d["min_score"]),
                noise_std=float(keypoints2d["noise_std"]),
                drop_prob=float(keypoints2d["drop_prob"]),
            )

        # --- camera twist input (motion tokens, else the pose token) ---
        self.camera_input = None
        if camera_twist is not None:
            self.camera_input = TwistInput(dim)

        # --- bbox input (pose token) ---
        self.bbox_input = BboxInput(dim) if bbox is not None else None

        # --- frozen camera input (pose token) ---
        self.frozen_camera_input = (
            FrozenCameraInput(dim) if frozen_camera is not None else None)

        # --- cross-modal temporal brick ---
        self.cross_modal_temporal = None
        self.cross_modal_modalities = []
        if cross_modal is not None:
            requested = [str(m) for m in cross_modal["modalities"]]
            available = {
                "pose": True,
                "contact": self.contact_tokens is not None,
                "force": self.force_tokens is not None,
                "motion": self.motion_tokens is not None,
            }
            missing = [m for m in requested if not available[m]]
            assert not missing, (
                f"cross_modal modalities {missing} have no token block in this "
                "build (enable the corresponding branch)")
            assert requested, "cross_modal needs at least one modality"
            self.cross_modal_modalities = [
                m for m in _MODALITY_ORDER if m in requested]
            self.cross_modal_temporal = CrossModalRopeModule(
                dim=dim,
                num_slots=sum(self._modality_token_count(m)
                              for m in self.cross_modal_modalities),
                num_layers=int(cross_modal["num_layers"]),
                num_heads=int(cross_modal["num_heads"]),
                mlp_ratio=float(cross_modal["mlp_ratio"]),
                dropout=float(cross_modal["dropout"]),
                time_scale=float(cross_modal["time_scale"]),
                max_rel_sec=cross_modal["max_rel_sec"],
                gate_init=str(cross_modal["gate_init"]),
            )

        # --- train-time span masking of the cross-modal input ---
        self.token_masking = None
        if token_masking is not None:
            assert self.cross_modal_temporal is not None, (
                "token_masking corrupts the cross-modal temporal block's input; "
                "enable cross_modal")
            self.token_masking = TokenMasking(
                num_slots=self.cross_modal_temporal.num_slots, dim=dim,
                frac=float(token_masking["frac"]),
                span_min=int(token_masking["span_min"]),
                span_max=int(token_masking["span_max"]),
                replace=str(token_masking["replace"]),
            )

        # --- pose temporal brick ---
        self.pose_temporal = None
        if pose_temporal is not None:
            self.pose_temporal = RopeTemporalModule(
                dim=dim,
                num_layers=int(pose_temporal["num_layers"]),
                num_heads=int(pose_temporal["num_heads"]),
                mlp_ratio=float(pose_temporal["mlp_ratio"]),
                dropout=float(pose_temporal["dropout"]),
                time_scale=float(pose_temporal["time_scale"]),
                max_rel_sec=pose_temporal["max_rel_sec"],
            )

        # --- split-head fine-tune copies (final readout only) ---
        # deepcopy-initialised, so before any optimizer step the recomputed
        # readout is bit-identical to the frozen one.
        self.head_pose_ft_proj = (
            copy.deepcopy(wrapper.model.head_pose.proj)
            if finetune_pose_head else None)
        self.head_camera_ft_proj = (
            copy.deepcopy(wrapper.model.head_camera.proj)
            if finetune_camera_head else None)
        if self.head_pose_ft_proj is not None:
            for p in self.head_pose_ft_proj.parameters():
                p.requires_grad = True
        if self.head_camera_ft_proj is not None:
            for p in self.head_camera_ft_proj.parameters():
                p.requires_grad = True

    def _modality_token_count(self, modality: str) -> int:
        return {
            "pose": 1,
            "contact": self.contact_tokens.num_tokens if self.contact_tokens else 0,
            "force": self.force_tokens.num_tokens if self.force_tokens else 0,
            "motion": self.motion_tokens.num_tokens if self.motion_tokens else 0,
        }[modality]

    @property
    def writes_pose(self) -> bool:
        """Whether the final MHR output can differ from the frozen model's."""
        return (
            "pose" in self.cross_modal_modalities
            or self.pose_temporal is not None
            or self.keypoint_input is not None
            or self.bbox_input is not None
            or self.frozen_camera_input is not None
            or (self.camera_input is not None and self.motion_tokens is None)
            or self.head_pose_ft_proj is not None
            or self.head_camera_ft_proj is not None
        )

    def _keypoint_embedding(self, batch: Dict, mhr: Dict) -> torch.Tensor:
        """Additive pose-token embedding of the frame's 2D keypoints, ``[B, C]``.

        ``sapiens``: the corpus detections (``kp2d_in`` px + score, MHR70
        order). ``mhr``: the frozen readout's own ``pred_keypoints_2d`` (full
        image px, score 1) — what deployment has.
        """
        indices = self.keypoint_indices
        if self.keypoint_source == "sapiens":
            kp = batch["kp2d_in"][:, indices]
            pixels, score, valid = kp[..., :2], kp[..., 2], batch["kp2d_in_valid"]
        else:
            pixels = mhr["pred_keypoints_2d"].detach()[:, indices]
            score = torch.ones_like(pixels[..., 0])
            valid = torch.ones(pixels.shape[0], dtype=torch.bool, device=pixels.device)
        return self.keypoint_input(
            pixels, score, valid, batch["affine_trans"], batch["img_size"])

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, batch: Dict) -> Dict:
        """Forward one collated batch (flattened clips, ``[B_clips * T, ...]``).

        Consumes the collate layout: flat per-frame geometry ``[B, ...]`` (no
        person dimension), optional ``embedding`` (cached backbone output), and
        the ``seq_len`` / ``frame_pos_sec`` / ``frame_valid`` clip fields.

        :returns: ``{"mhr", "contact", "force", "motion", "smplx", "tokens",
            "blocks", "ctx"}`` — head outputs are ``None`` for disabled branches.
        """
        embedding = batch.get("embedding")          # optional: the cached backbone path
        img = batch["img"] if embedding is None else None
        batch_size = (img if embedding is None else embedding).shape[0]

        learned = [b for b in (self.contact_tokens, self.force_tokens,
                               self.motion_tokens) if b is not None]
        out = self.wrapper(
            img=img,
            embedding=embedding,
            bbox_center=batch["bbox_center"],
            bbox_scale=batch["bbox_scale"],
            ori_img_size=batch["ori_img_size"],
            img_size=batch["img_size"],
            affine_trans=batch["affine_trans"],
            cam_int=batch["cam_int"],
            mask=batch["mask"],
            mask_score=batch["mask_score"],
            blocks=[b.as_extra_block(batch_size) for b in learned],
            attention=self.extra_token_attention,
        )
        tokens = out["tokens"]
        bounds = dict(out["blocks"])
        bounds["pose"] = (0, 1)

        if self.keypoint_input is not None:
            embedding = self._keypoint_embedding(batch, out["mhr"])
            tokens = torch.cat([tokens[:, :1] + embedding[:, None], tokens[:, 1:]], dim=1)
        if self.bbox_input is not None:
            embedding = self.bbox_input(
                batch["bbox_center"], batch["bbox_scale"][:, 0], batch["cam_int"])
            tokens = torch.cat([tokens[:, :1] + embedding[:, None], tokens[:, 1:]], dim=1)
        # The frozen readout's pelvis: the measurement the frozen-camera input
        # and the ray head's frozen prior consume (no gradient into the base).
        frozen_pelvis = None
        if self.frozen_camera_input is not None or (
                self.head_smplx is not None and self.head_smplx.camera == "ray"
                and self.head_smplx.depth_prior == "frozen"):
            frozen_pelvis = frozen_pelvis_camera(out["mhr"]).detach()
        if self.frozen_camera_input is not None:
            embedding = self.frozen_camera_input(frozen_pelvis)
            tokens = torch.cat([tokens[:, :1] + embedding[:, None], tokens[:, 1:]], dim=1)
        if self.camera_input is not None:
            embedding = self.camera_input(batch["cam_twist"])[:, None]     # [B, 1, C]
            lo, hi = bounds["motion"] if self.motion_tokens is not None else (0, 1)
            tokens = torch.cat(
                [tokens[:, :lo], tokens[:, lo:hi] + embedding, tokens[:, hi:]], dim=1)

        # --- temporal bricks (post-decoder) ---
        seq_len = int(batch["seq_len"])
        frame_pos_sec = batch["frame_pos_sec"]
        frame_valid = batch["frame_valid"]

        if self.cross_modal_temporal is not None:
            slices = [bounds[m] for m in self.cross_modal_modalities]
            mixed = torch.cat([tokens[:, lo:hi] for lo, hi in slices], dim=1)
            if self.token_masking is not None and self.training:
                mixed, _ = self.token_masking(mixed, seq_len)
            updated = self.cross_modal_temporal(
                mixed, seq_len, frame_pos_sec, frame_valid)
            # Scatter the updated slices back (ordered + disjoint by the
            # canonical modality order; everything between them is untouched).
            pieces, off, prev = [], 0, 0
            for lo, hi in slices:
                k = hi - lo
                pieces += [tokens[:, prev:lo], updated[:, off:off + k]]
                off, prev = off + k, hi
            pieces.append(tokens[:, prev:])
            tokens = torch.cat(pieces, dim=1)

        if self.pose_temporal is not None:
            updated = self.pose_temporal(
                tokens[:, 0:1], seq_len, frame_pos_sec, frame_valid)
            tokens = torch.cat([updated, tokens[:, 1:]], dim=1)

        # --- final MHR readout ---
        # Skipped with the SMPL-X head: that head is the pose output, so the
        # (expensive) MHR recompute would feed nothing.
        mhr = out["mhr"]
        if self.writes_pose and self.head_smplx is None:
            frozen = mhr
            mhr = self.wrapper.decode_pose(
                tokens[:, 0], out["ctx"],
                proj_pose=self.head_pose_ft_proj,
                proj_camera=self.head_camera_ft_proj,
            )
            # The frozen model's own readout anchors the trust-region rails.
            mhr["pred_cam_t_frozen"] = frozen["pred_cam_t"].detach()
            mhr["global_rot_frozen"] = frozen["global_rot"].detach()
            mhr["shape_frozen"] = frozen["shape"].detach()
            mhr["scale_frozen"] = frozen["scale"].detach()

        # --- heads ---
        contact_output = None
        if self.head_contact is not None:
            # Learned contact tokens, or (source: pose_token) the pose token itself.
            lo, hi = bounds["contact"] if self.contact_tokens is not None else (0, 1)
            contact_output = {}
            for name, head in self.head_contact.items():
                logits = head(tokens[:, lo:hi])
                contact_output[f"{name}_logits"] = logits
                contact_output[f"{name}_probs"] = torch.sigmoid(logits)

        force_output = None
        if self.force_tokens is not None:
            lo, hi = bounds["force"]
            joint_forces = self.head_force(tokens[:, lo:hi])       # [B, K, 3]
            force_output = {"joint_forces": joint_forces}
            if self.force_contact_gate:
                # Gate by the DETACHED per-group contact logits so eval /
                # rendering see gated forces; the raw tensor stays for
                # diagnostics. The detach keeps the force loss from rewriting
                # the calibrated contact probabilities through this product.
                force_output = {
                    "joint_forces": contact_gate_forces(
                        joint_forces,
                        contact_output["joint_logits"],
                        self.force_contact_gate_sharpness,
                    ),
                    "joint_forces_raw": joint_forces,
                }

        motion_output = None
        if self.head_motion is not None:
            # Learned motion tokens, or (source: pose_token) the pose token itself.
            lo, hi = bounds["motion"] if self.motion_tokens is not None else (0, 1)
            motion = self.head_motion(tokens[:, lo:hi])            # [B, K, 3 * terms]
            motion_output = {"joint_motion": motion}
            for i, term in enumerate(self.motion_terms):
                motion_output[f"joint_{term}"] = motion[..., 3 * i:3 * i + 3]

        smplx_output = None
        if self.head_smplx is not None:
            smplx_output = self.head_smplx(
                tokens[:, 0],
                bbox_center=batch["bbox_center"],
                bbox_size=batch["bbox_scale"][:, 0],
                cam_int=batch["cam_int"],
                affine_trans=batch["affine_trans"],
                img_size=batch["img_size"],
                frozen_pelvis_cam=frozen_pelvis,
            )

        return {
            "mhr": mhr,
            "contact": contact_output,
            "force": force_output,
            "motion": motion_output,
            "smplx": smplx_output,
            "tokens": tokens,
            "blocks": bounds,
            "ctx": out["ctx"],
        }
