# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Any, Dict, Optional, Tuple

import numpy as np
import roma
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam_3d_body.data.utils.prepare_batch import prepare_batch
from sam_3d_body.models.decoders.prompt_encoder import PositionEmbeddingRandom
from sam_3d_body.models.modules.mhr_utils import (
    fix_wrist_euler,
    rotation_angle_difference,
)
from sam_3d_body.utils import recursive_to
from sam_3d_body.utils.logging import get_pylogger

from ..backbones import create_backbone
from ..decoders import build_decoder, build_keypoint_sampler, PromptEncoder
from ..heads import build_head
# --- force contact-gate hook (import) ---
from ..heads.force_head import FORCE_GATE_CONTACT_MAP, contact_gate_forces
# --- end force contact-gate hook ---
from ..modules.camera_embed import CameraEncoder
from ..modules.transformer import FFN, MLP

from .base_model import BaseModel


logger = get_pylogger(__name__)


# fmt: off
PROMPT_KEYPOINTS = {  # keypoint_idx: prompt_idx
    "mhr70": {
        i: i for i in range(70)
    },  # all 70 keypoints are supported for prompting
}
KEY_BODY = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 41, 62]  # key body joints for prompting
KEY_RIGHT_HAND = list(range(21, 42))
# fmt: on


class SAM3DBody(BaseModel):
    pelvis_idx = [9, 10]  # left_hip, right_hip

    def _initialze_model(self):
        self.register_buffer(
            "image_mean", torch.tensor(self.cfg.MODEL.IMAGE_MEAN).view(-1, 1, 1), False
        )
        self.register_buffer(
            "image_std", torch.tensor(self.cfg.MODEL.IMAGE_STD).view(-1, 1, 1), False
        )

        # Create backbone feature extractor for human crops
        self.backbone = create_backbone(self.cfg.MODEL.BACKBONE.TYPE, self.cfg)

        # Create header for pose estimation output
        self.head_pose = build_head(self.cfg, self.cfg.MODEL.PERSON_HEAD.POSE_TYPE)
        self.head_pose.hand_pose_comps_ori = nn.Parameter(
            self.head_pose.hand_pose_comps.clone(), requires_grad=False
        )
        self.head_pose.hand_pose_comps.data = (
            torch.eye(54).to(self.head_pose.hand_pose_comps.data).float()
        )

        # Initialize pose token with learnable params
        # Note: bias/initial value should be zero-pose in cont, not all-zeros
        self.init_pose = nn.Embedding(1, self.head_pose.npose)

        # Define header for hand pose estimation
        self.head_pose_hand = build_head(
            self.cfg, self.cfg.MODEL.PERSON_HEAD.POSE_TYPE, enable_hand_model=True
        )
        self.head_pose_hand.hand_pose_comps_ori = nn.Parameter(
            self.head_pose_hand.hand_pose_comps.clone(), requires_grad=False
        )
        self.head_pose_hand.hand_pose_comps.data = (
            torch.eye(54).to(self.head_pose_hand.hand_pose_comps.data).float()
        )
        self.init_pose_hand = nn.Embedding(1, self.head_pose_hand.npose)

        self.head_camera = build_head(self.cfg, self.cfg.MODEL.PERSON_HEAD.CAMERA_TYPE)
        self.init_camera = nn.Embedding(1, self.head_camera.ncam)
        nn.init.zeros_(self.init_camera.weight)

        self.head_camera_hand = build_head(
            self.cfg,
            self.cfg.MODEL.PERSON_HEAD.CAMERA_TYPE,
            default_scale_factor=self.cfg.MODEL.CAMERA_HEAD.get(
                "DEFAULT_SCALE_FACTOR_HAND", 1.0
            ),
        )
        self.init_camera_hand = nn.Embedding(1, self.head_camera_hand.ncam)
        nn.init.zeros_(self.init_camera_hand.weight)

        self.camera_type = "perspective"

        # Support conditioned information for decoder
        cond_dim = 3
        init_dim = self.head_pose.npose + self.head_camera.ncam + cond_dim
        self.init_to_token_mhr = nn.Linear(init_dim, self.cfg.MODEL.DECODER.DIM)
        self.prev_to_token_mhr = nn.Linear(
            init_dim - cond_dim, self.cfg.MODEL.DECODER.DIM
        )
        self.init_to_token_mhr_hand = nn.Linear(init_dim, self.cfg.MODEL.DECODER.DIM)
        self.prev_to_token_mhr_hand = nn.Linear(
            init_dim - cond_dim, self.cfg.MODEL.DECODER.DIM
        )

        # Create prompt encoder
        self.max_num_clicks = 0
        if self.cfg.MODEL.PROMPT_ENCODER.ENABLE:
            self.max_num_clicks = self.cfg.MODEL.PROMPT_ENCODER.MAX_NUM_CLICKS
            self.prompt_keypoints = PROMPT_KEYPOINTS[
                self.cfg.MODEL.PROMPT_ENCODER.PROMPT_KEYPOINTS
            ]

            self.prompt_encoder = PromptEncoder(
                embed_dim=self.backbone.embed_dims,  # need to match backbone dims for PE
                num_body_joints=len(set(self.prompt_keypoints.values())),
                frozen=self.cfg.MODEL.PROMPT_ENCODER.get("frozen", False),
                mask_embed_type=self.cfg.MODEL.PROMPT_ENCODER.get(
                    "MASK_EMBED_TYPE", None
                ),
            )
            self.prompt_to_token = nn.Linear(
                self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM
            )

            self.keypoint_prompt_sampler = build_keypoint_sampler(
                self.cfg.MODEL.PROMPT_ENCODER.get("KEYPOINT_SAMPLER", {}),
                prompt_keypoints=self.prompt_keypoints,
                keybody_idx=(
                    KEY_BODY
                    if not self.cfg.MODEL.PROMPT_ENCODER.get("SAMPLE_HAND", False)
                    else KEY_RIGHT_HAND
                ),
            )
            # To keep track of prompting history
            self.prompt_hist = np.zeros(
                (len(set(self.prompt_keypoints.values())) + 2, self.max_num_clicks),
                dtype=np.float32,
            )

            if self.cfg.MODEL.DECODER.FROZEN:
                for param in self.prompt_to_token.parameters():
                    param.requires_grad = False

        # Create promptable decoder
        self.decoder = build_decoder(
            self.cfg.MODEL.DECODER, context_dim=self.backbone.embed_dims
        )
        # shared config for the two decoders
        self.decoder_hand = build_decoder(
            self.cfg.MODEL.DECODER, context_dim=self.backbone.embed_dims
        )
        self.hand_pe_layer = PositionEmbeddingRandom(self.backbone.embed_dims // 2)

        # --- contact efficiency hook (detach_interm_preds) ---
        # When set, the decoder runs its per-layer interm MHR/camera predictions
        # under no_grad (they feed keypoint/contact-token *sampling locations* only,
        # and every grad path through them dead-ends in frozen params). Absent key =
        # old full-graph behaviour, so the fork stays usable standalone.
        _detach_interm = bool(self.cfg.MODEL.get("EFFICIENCY", {}).get("DETACH_INTERM_PREDS", False))
        self.decoder.detach_interm_preds = _detach_interm
        self.decoder_hand.detach_interm_preds = _detach_interm
        # --- end contact efficiency hook ---

        # Manually convert the torso of the model to fp16.
        if self.cfg.TRAIN.USE_FP16:
            self.convert_to_fp16()
            if self.cfg.TRAIN.get("FP16_TYPE", "float16") == "float16":
                self.backbone_dtype = torch.float16
            else:
                self.backbone_dtype = torch.bfloat16
        else:
            self.backbone_dtype = torch.float32

        self.ray_cond_emb = CameraEncoder(
            self.backbone.embed_dim,
            self.backbone.patch_size,
        )
        self.ray_cond_emb_hand = CameraEncoder(
            self.backbone.embed_dim,
            self.backbone.patch_size,
        )

        self.keypoint_embedding_idxs = list(range(70))
        self.keypoint_embedding = nn.Embedding(
            len(self.keypoint_embedding_idxs), self.cfg.MODEL.DECODER.DIM
        )
        self.keypoint_embedding_idxs_hand = list(range(70))
        self.keypoint_embedding_hand = nn.Embedding(
            len(self.keypoint_embedding_idxs_hand), self.cfg.MODEL.DECODER.DIM
        )

        if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
            self.hand_box_embedding = nn.Embedding(
                2, self.cfg.MODEL.DECODER.DIM
            )  # for two hands
            # decice if there is left or right hand inside the image
            self.hand_cls_embed = nn.Linear(self.cfg.MODEL.DECODER.DIM, 2)
            self.bbox_embed = MLP(
                self.cfg.MODEL.DECODER.DIM, self.cfg.MODEL.DECODER.DIM, 4, 3
            )

        # Contact prediction head and tokens
        if self.cfg.MODEL.DECODER.get("DO_CONTACT_TOKENS", False):
            contact_head_cfg = self.cfg.MODEL.get("CONTACT_HEAD", dict())
            # Each anchored contact token grid-samples image features at one
            # explicitly selected MHR70 keypoint. The list is fully configurable
            # (for example [62, 41, 13, 14] for both wrists and ankles), and the
            # number of anchored tokens follows its length.
            kp_indices = contact_head_cfg.get("KEYPOINT_INDICES", None)
            if kp_indices is None:
                kp_indices = list(range(21))
            kp_indices = list(kp_indices)
            assert all(0 <= int(i) < 70 for i in kp_indices), (
                f"contact keypoint indices must be MHR70 indices in [0, 70); got {kp_indices}"
            )
            self.contact_keypoint_indices = [int(i) for i in kp_indices]
            self.num_contact_tokens = len(self.contact_keypoint_indices)
            # Extra global token(s) not updated with image features
            self.num_global_contact_tokens = contact_head_cfg.get("NUM_GLOBAL_TOKENS", 0)
            self.total_contact_tokens = self.num_contact_tokens + self.num_global_contact_tokens
            # Learnable query tokens for contact prediction
            self.contact_embedding = nn.Embedding(
                self.total_contact_tokens, self.cfg.MODEL.DECODER.DIM
            )
            # One independent prediction head per enabled target. TARGETS maps a
            # target name to its configured output dimension.
            targets = contact_head_cfg.get(
                "TARGETS", {"VERTEX": contact_head_cfg.get("NUM_VERTICES", 18439)}
            )
            self.contact_target_names = [str(name).lower() for name in targets]
            self.head_contact = nn.ModuleDict({
                str(name).lower(): build_head(self.cfg, "contact", output_dims=int(dims))
                for name, dims in targets.items()
            })

            # --- contact blind hook (ablation: no image features at all) ---
            # When set, the contact tokens lose every path to the image: the
            # decoder cross-attention is gated off for their rows (see
            # forward_decoder) and the anchored update below never runs, so their
            # only input is self-attention over the preceding (body) tokens. The
            # two anchored-update projections are then not built at all — keeping
            # them would leave params that never receive a gradient, which DDP
            # rejects without find_unused_parameters.
            self.contact_blind_to_image = bool(
                contact_head_cfg.get("BLIND_TO_IMAGE", False)
            )
            if not self.contact_blind_to_image:
                # Positional encoding: project 2D keypoint position -> decoder dim
                self.contact_posemb_linear = FFN(
                    embed_dims=2,
                    feedforward_channels=self.cfg.MODEL.DECODER.DIM,
                    output_dims=self.cfg.MODEL.DECODER.DIM,
                    num_fcs=2,
                    add_identity=False,
                )
                # Feature projection: project sampled image features -> decoder dim
                self.contact_feat_linear = nn.Linear(
                    self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM
                )
            # --- end contact blind hook ---
            # K×K grid sampling params
            self.contact_grid_size   = contact_head_cfg.get("GRID_SIZE", 1)
            self.contact_grid_radius = contact_head_cfg.get("GRID_RADIUS", 0.1)

        # --- force hook (module construction) ---
        # Force tokens/head mirror the contact machinery: keypoint anchors, an
        # own embedding + posemb/feat linears, and a per-token regression head.
        # Every param carries "force" in its name so the generalized freeze/eval
        # filters ("contact" OR "force") pick it up.
        if self.cfg.MODEL.DECODER.get("DO_FORCE_TOKENS", False):
            force_head_cfg = self.cfg.MODEL.get("FORCE_HEAD", dict())
            # Force anchors: FORCE_HEAD.KEYPOINT_INDICES when explicitly set,
            # otherwise the contact anchors (D2, legacy default — requires the
            # contact tokens to exist). No global force tokens either way.
            force_kp_indices = force_head_cfg.get("KEYPOINT_INDICES", None)
            if force_kp_indices is None:
                assert self.cfg.MODEL.DECODER.get("DO_CONTACT_TOKENS", False), (
                    "DO_FORCE_TOKENS without FORCE_HEAD.KEYPOINT_INDICES requires "
                    "DO_CONTACT_TOKENS: the force tokens then reuse the contact "
                    "keypoint anchors (contact_keypoint_indices)."
                )
                force_kp_indices = self.contact_keypoint_indices
            force_kp_indices = list(force_kp_indices)
            assert all(0 <= int(i) < 70 for i in force_kp_indices), (
                f"force keypoint indices must be MHR70 indices in [0, 70); "
                f"got {force_kp_indices}"
            )
            self.force_keypoint_indices = [int(i) for i in force_kp_indices]
            self.num_force_tokens = len(self.force_keypoint_indices)
            if not self.cfg.MODEL.DECODER.get("DO_CONTACT_TOKENS", False):
                # Force-only build: the anchored update's grid-sampling params
                # are normally set by the contact block above.
                contact_head_cfg = self.cfg.MODEL.get("CONTACT_HEAD", dict())
                self.contact_grid_size   = contact_head_cfg.get("GRID_SIZE", 1)
                self.contact_grid_radius = contact_head_cfg.get("GRID_RADIUS", 0.1)
            self.force_embedding = nn.Embedding(
                self.num_force_tokens, self.cfg.MODEL.DECODER.DIM
            )
            self.head_force = build_head(self.cfg, "force")
            self.force_posemb_linear = FFN(
                embed_dims=2,
                feedforward_channels=self.cfg.MODEL.DECODER.DIM,
                output_dims=self.cfg.MODEL.DECODER.DIM,
                num_fcs=2,
                add_identity=False,
            )
            self.force_feat_linear = nn.Linear(
                self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM
            )

            # --- force contact-gate hook (module construction) ---
            # Contact-gated final force output (no params — a head-level product;
            # config validation requires the six-token per_token kindyn_6 contact
            # target, matched 1:1 to the six force groups, whenever the gate is on).
            self.force_contact_gate = bool(
                force_head_cfg.get("CONTACT_GATE_ENABLED", False)
            )
            self.force_contact_gate_sharpness = float(
                force_head_cfg.get("CONTACT_GATE_SHARPNESS", 4.0)
            )
            if self.force_contact_gate:
                assert self.cfg.MODEL.DECODER.get("DO_CONTACT_TOKENS", False), (
                    "FORCE_HEAD.CONTACT_GATE_ENABLED requires DO_CONTACT_TOKENS: "
                    "the gate reads the per-group contact logits"
                )
                assert self.num_force_tokens == len(FORCE_GATE_CONTACT_MAP), (
                    "FORCE_HEAD.CONTACT_GATE_ENABLED requires "
                    f"{len(FORCE_GATE_CONTACT_MAP)} force tokens (kindyn groups); "
                    f"got {self.num_force_tokens}"
                )
            # --- end force contact-gate hook ---
        # --- end force hook ---

        # --- cond input hook (module construction) ---
        # Per-frame conditioning feature (smoothed root velocity/acceleration
        # derived from the model's OWN reconstructed pelvis; see
        # contact/data/climbing_corpus.py::cond_feature_rows) projected into the
        # decoder dim and ADDED to the contact/force token stream. INJECTION
        # picks where: "pre_decoder" = the INITIAL token embeddings (the feature
        # integrates with image evidence over all decoder layers); "post_decoder"
        # = the decoder's token OUTPUTS, right before the heads.
        # Zero-init, so an enabled build starts bit-identical to the unconditioned
        # one. Param names carry "contact"/"force" for the freeze/eval filters.
        # No mask involvement: only token *values* change, so the frozen pose/MHR
        # outputs keep their exactly-zero Jacobian w.r.t. these params.
        self.cond_input_dim = 0
        self.cond_input_injection = "pre_decoder"
        cond_cfg = self.cfg.MODEL.get("COND_INPUT", None)
        if cond_cfg is not None and cond_cfg.get("ENABLED", False):
            self.cond_input_dim = int(cond_cfg.get("FEAT_DIM", 10))
            self.cond_input_injection = str(cond_cfg.get("INJECTION", "pre_decoder"))
            if self.cond_input_injection not in ("pre_decoder", "post_decoder"):
                raise ValueError(
                    f"COND_INPUT.INJECTION: {self.cond_input_injection!r}")
            # `nn.Linear` draws its default init before we overwrite it with
            # zeros, which would shift the global RNG stream and de-align every
            # module built after this block. Forked so an enabled build gives
            # every SHARED parameter exactly the weights the unconditioned build
            # gets — the A/B pair then differs only by these two zero tensors.
            # The OUTPUT layer is bias-free on purpose: a bias is a per-token-block
            # CONSTANT that would receive gradient even on an all-zero (invalid)
            # feature row, so a conditioned arm could drift from its baseline
            # through a channel that carries no motion information — and the token
            # embeddings already provide exactly that learnable constant. The MLP
            # variant's HIDDEN layer keeps its bias deliberately (a useful GELU
            # operating point; the constant it induces once the output layer is
            # non-zero is redundant with the token embedding, not harmful).
            # ENCODER_HIDDEN=None keeps the original bare linear; an int H swaps
            # in a small MLP (Linear(feat,H) + GELU + Linear(H,dim)). Either way
            # the OUTPUT layer is zero-init, so the projection is an exact no-op
            # at initialisation and every invariant above still holds. The MLP
            # variant keeps the *_cond_linear names: the freeze filter matches on
            # "contact"/"force" and the warm-start exemption on "cond_linear".
            hidden = self.cfg.MODEL.COND_INPUT.get("ENCODER_HIDDEN", None)

            def _cond_projection() -> nn.Module:
                if hidden is None:
                    linear = nn.Linear(
                        self.cond_input_dim, self.cfg.MODEL.DECODER.DIM, bias=False)
                    nn.init.zeros_(linear.weight)
                    return linear
                out = nn.Linear(int(hidden), self.cfg.MODEL.DECODER.DIM, bias=False)
                nn.init.zeros_(out.weight)
                return nn.Sequential(
                    nn.Linear(self.cond_input_dim, int(hidden)), nn.GELU(), out)

            with torch.random.fork_rng(devices=[]):
                if self.cfg.MODEL.DECODER.get("DO_CONTACT_TOKENS", False):
                    self.contact_cond_linear = _cond_projection()
                if self.cfg.MODEL.DECODER.get("DO_FORCE_TOKENS", False):
                    self.force_cond_linear = _cond_projection()
        # --- end cond input hook ---

        # --- motion hook (module construction) ---
        # Motion tokens/head mirror the force machinery: explicit keypoint
        # anchors (no contact inheritance, no global tokens), an own embedding +
        # posemb/feat linears, and a per-token vel/acc regression head. Every
        # param carries "motion" in its name so the generalized freeze/eval
        # filters ("contact" OR "force" OR "motion") pick it up.
        if self.cfg.MODEL.DECODER.get("DO_MOTION_TOKENS", False):
            motion_head_cfg = self.cfg.MODEL.get("MOTION_HEAD", dict())
            motion_kp_indices = list(motion_head_cfg["KEYPOINT_INDICES"])
            assert all(0 <= int(i) < 70 for i in motion_kp_indices), (
                f"motion keypoint indices must be MHR70 indices in [0, 70); "
                f"got {motion_kp_indices}"
            )
            self.motion_keypoint_indices = [int(i) for i in motion_kp_indices]
            self.num_motion_tokens = len(self.motion_keypoint_indices)
            if not hasattr(self, "contact_grid_size"):
                # Motion-only build: the anchored update's grid-sampling params
                # are normally set by the contact (or force-only) block above.
                contact_head_cfg = self.cfg.MODEL.get("CONTACT_HEAD", dict())
                self.contact_grid_size   = contact_head_cfg.get("GRID_SIZE", 1)
                self.contact_grid_radius = contact_head_cfg.get("GRID_RADIUS", 0.1)
            self.motion_embedding = nn.Embedding(
                self.num_motion_tokens, self.cfg.MODEL.DECODER.DIM
            )
            self.head_motion = build_head(self.cfg, "motion")
            # --- motion unanchored hook (pure learned queries) ---
            # ANCHORED=False drops the per-layer anchored update: the motion
            # tokens enter the sequence with no positional embedding and read the
            # image only through the decoder's cross-attention, so the anchor list
            # merely names/counts the slots. The two projections are then not built
            # at all — keeping them would leave params that never receive a
            # gradient, which DDP rejects without find_unused_parameters.
            self.motion_anchored = bool(motion_head_cfg.get("ANCHORED", True))
            if self.motion_anchored:
                # Positional encoding: project 2D keypoint position -> decoder dim
                self.motion_posemb_linear = FFN(
                    embed_dims=2,
                    feedforward_channels=self.cfg.MODEL.DECODER.DIM,
                    output_dims=self.cfg.MODEL.DECODER.DIM,
                    num_fcs=2,
                    add_identity=False,
                )
                # Feature projection: project sampled image features -> decoder dim
                self.motion_feat_linear = nn.Linear(
                    self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM
                )
            # --- end motion unanchored hook ---
        # --- end motion hook ---

        # --- pose temporal hook (module construction) ---
        # E2: temporal attention over the POSE token (sequence index 0), the ONE
        # deliberate exception to the frozen-pose rule — the pose modality's
        # OWN temporal block, run after cross_modal_temporal. The FINAL pose
        # output is recomputed from the updated token; zero-init gates make
        # init behavior exactly frozen. Every param carries "pose_temporal" in
        # its name.
        ptcfg = self.cfg.MODEL.get("POSE_TEMPORAL", None)
        if ptcfg is not None and ptcfg.get("ENABLED", False):
            from ..modules.temporal_rope import RopeTemporalModule
            self.pose_temporal = RopeTemporalModule(
                dim=self.cfg.MODEL.DECODER.DIM,
                num_layers=ptcfg.get("NUM_LAYERS", 4),
                num_heads=ptcfg.get("NUM_HEADS", 16),
                mlp_ratio=ptcfg.get("MLP_RATIO", 2.0),
                dropout=ptcfg.get("DROPOUT", 0.0),
                time_scale=ptcfg.get("TIME_SCALE", 25.0),
                max_rel_sec=ptcfg.get("MAX_REL_SEC", 2.5),
            )
        # --- end pose temporal hook ---

        # --- cross-modal temporal hook (module construction) ---
        # THE post-decoder mixing brick: ONE RoPE temporal transformer over the
        # CONCATENATION of the chosen modality token blocks. Every participating
        # token attends every other one across all of the clip's frames, so
        # contact/force/motion see the pose token and vice versa, and the dt = 0
        # diagonal gives within-frame cross-modal attention for free. This
        # deliberately relaxes the per-modality gradient isolation (D1) AMONG
        # the participating blocks; the frozen pose/MHR outputs stay isolated
        # unless 'pose' participates (the final pose output is then recomputed
        # from the updated token, like pose_temporal). Params carry
        # "cross_modal" in their names for the freeze/eval filters. Attribute
        # absent when disabled.
        xmcfg = self.cfg.MODEL.get("CROSS_MODAL_TEMPORAL", None)
        if xmcfg is not None and xmcfg.get("ENABLED", False):
            requested = [str(m) for m in xmcfg.get("MODALITIES", [])]
            missing = [m for m in requested if not self._modality_available(m)]
            assert not missing, (
                f"cross_modal_temporal modalities {missing} have no token "
                "block in this build (enable the corresponding head)")
            # Canonical sequence order (pose < contact < force < motion)
            # regardless of the config list's order: the forward hook
            # concatenates the slices in token-sequence order.
            self.cross_modal_modalities = [
                m for m in ("pose", "contact", "force", "motion")
                if m in requested]
            from ..modules.cross_modal_rope import CrossModalRopeModule
            self.cross_modal_temporal = CrossModalRopeModule(
                dim=self.cfg.MODEL.DECODER.DIM,
                num_slots=sum(self._modality_token_count(m)
                              for m in self.cross_modal_modalities),
                num_layers=xmcfg.get("NUM_LAYERS", 4),
                num_heads=xmcfg.get("NUM_HEADS", 16),
                mlp_ratio=xmcfg.get("MLP_RATIO", 2.0),
                dropout=xmcfg.get("DROPOUT", 0.0),
                time_scale=xmcfg.get("TIME_SCALE", 25.0),
                max_rel_sec=xmcfg.get("MAX_REL_SEC", 2.5),
            )
        # --- end cross-modal temporal hook ---

        self.keypoint_posemb_linear = FFN(
            embed_dims=2,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
        )
        self.keypoint_posemb_linear_hand = FFN(
            embed_dims=2,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
        )
        self.keypoint_feat_linear = nn.Linear(
            self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM
        )
        self.keypoint_feat_linear_hand = nn.Linear(
            self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM
        )

        # Do all KPS
        self.keypoint3d_embedding_idxs = list(range(70))
        self.keypoint3d_embedding = nn.Embedding(
            len(self.keypoint3d_embedding_idxs), self.cfg.MODEL.DECODER.DIM
        )

        # Assume always do full body for the hand decoder
        self.keypoint3d_embedding_idxs_hand = list(range(70))
        self.keypoint3d_embedding_hand = nn.Embedding(
            len(self.keypoint3d_embedding_idxs_hand), self.cfg.MODEL.DECODER.DIM
        )

        self.keypoint3d_posemb_linear = FFN(
            embed_dims=3,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
        )
        self.keypoint3d_posemb_linear_hand = FFN(
            embed_dims=3,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
        )

    def _get_decoder_condition(self, batch: Dict) -> Optional[torch.Tensor]:
        num_person = batch["img"].shape[1]

        if self.cfg.MODEL.DECODER.CONDITION_TYPE == "cliff":
            # CLIFF-style condition info (cx/f, cy/f, b/f)
            cx, cy = torch.chunk(
                self._flatten_person(batch["bbox_center"]), chunks=2, dim=-1
            )
            img_w, img_h = torch.chunk(
                self._flatten_person(batch["ori_img_size"]), chunks=2, dim=-1
            )
            b = self._flatten_person(batch["bbox_scale"])[:, [0]]

            focal_length = self._flatten_person(
                batch["cam_int"]
                .unsqueeze(1)
                .expand(-1, num_person, -1, -1)
                .contiguous()
            )[:, 0, 0]
            if not self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False):
                condition_info = torch.cat(
                    [cx - img_w / 2.0, cy - img_h / 2.0, b], dim=-1
                )
            else:
                full_img_cxy = self._flatten_person(
                    batch["cam_int"]
                    .unsqueeze(1)
                    .expand(-1, num_person, -1, -1)
                    .contiguous()
                )[:, [0, 1], [2, 2]]
                condition_info = torch.cat(
                    [cx - full_img_cxy[:, [0]], cy - full_img_cxy[:, [1]], b], dim=-1
                )
            condition_info[:, :2] = condition_info[:, :2] / focal_length.unsqueeze(
                -1
            )  # [-1, 1]
            condition_info[:, 2] = condition_info[:, 2] / focal_length  # [-1, 1]
        elif self.cfg.MODEL.DECODER.CONDITION_TYPE == "none":
            return None
        else:
            raise NotImplementedError

        return condition_info.type(batch["img"].dtype)

    def _modality_available(self, modality: str) -> bool:
        """Whether ``modality`` has a token block in this build ('pose' always)."""
        return {
            "pose": True,
            "contact": bool(self.cfg.MODEL.DECODER.get("DO_CONTACT_TOKENS", False)),
            "force": bool(self.cfg.MODEL.DECODER.get("DO_FORCE_TOKENS", False)),
            "motion": bool(self.cfg.MODEL.DECODER.get("DO_MOTION_TOKENS", False)),
        }.get(modality, False)

    def _modality_token_count(self, modality: str) -> int:
        """How many decoder tokens ``modality``'s block contributes per frame."""
        return {
            "pose": 1,
            "contact": getattr(self, "total_contact_tokens", 0),
            "force": getattr(self, "num_force_tokens", 0),
            "motion": getattr(self, "num_motion_tokens", 0),
        }[modality]

    def _contact_temporal_fields(self, batch):
        """Temporal-clip fields for the body samples (the temporal hooks).

        Returns ``(seq_len, frame_pos_sec, frame_valid)`` where the per-frame
        tensors are indexed by ``self.body_batch_idx`` so they line up row-for-row
        with the contact tokens / image features seen inside ``forward_decoder``.
        Defaults to ``(1, None, None)`` for single-image batches (no ``seq_len``).
        """
        if batch is None:
            return 1, None, None
        seq_len = int(batch.get("seq_len", 1))
        pos = batch.get("frame_pos_sec")
        valid = batch.get("frame_valid")
        idx = self.body_batch_idx
        if idx is not None and len(idx):
            if pos is not None:
                pos = pos[idx]
            if valid is not None:
                valid = valid[idx]
        return seq_len, pos, valid

    def _cond_input_feature(self, batch, batch_size, ref):
        """Per-body-row conditioning feature (cond input hook).

        Returns ``[batch_size, cond_input_dim]`` on ``ref``'s device/dtype,
        indexed by ``self.body_batch_idx`` so it lines up row-for-row with the
        contact/force tokens. Batches without the key (still-image collates that
        predate it, or a bare inference batch) contribute exact zeros rather than
        skipping the projection, so its params are used on every step — DDP
        rejects a parameter some ranks leave unused.
        """
        feat = None if batch is None else batch.get("cond_feat")
        if feat is None:
            return torch.zeros(
                batch_size, self.cond_input_dim, dtype=ref.dtype, device=ref.device)
        idx = self.body_batch_idx
        if idx is not None and len(idx):
            feat = feat[idx]
        return feat.to(dtype=ref.dtype, device=ref.device)

    def forward_decoder(
        self,
        image_embeddings: torch.Tensor,
        init_estimate: Optional[torch.Tensor] = None,
        keypoints: Optional[torch.Tensor] = None,
        prev_estimate: Optional[torch.Tensor] = None,
        condition_info: Optional[torch.Tensor] = None,
        batch=None,
    ):
        """
        Args:
            image_embeddings: image features from the backbone, shape (B, C, H, W)
            init_estimate: initial estimate to be refined on, shape (B, 1, C)
            keypoints: optional prompt input, shape (B, N, 3),
                3 for coordinates (x,y) + label.
                (x, y) should be normalized to range [0, 1].
                label==-1 indicates incorrect points,
                label==-2 indicates invalid points
            prev_estimate: optional prompt input, shape (B, 1, C),
                previous estimate for pose refinement.
            condition_info: optional condition information that is concatenated with
                the input tokens, shape (B, c)
        """
        batch_size = image_embeddings.shape[0]

        # Initial estimation for residual prediction.
        if init_estimate is None:
            init_pose = self.init_pose.weight.expand(batch_size, -1).unsqueeze(dim=1)
            if hasattr(self, "init_camera"):
                init_camera = self.init_camera.weight.expand(batch_size, -1).unsqueeze(
                    dim=1
                )

            init_estimate = (
                init_pose
                if not hasattr(self, "init_camera")
                else torch.cat([init_pose, init_camera], dim=-1)
            )  # This is basically pose & camera translation at the end. B x 1 x (404 + 3)

        if condition_info is not None:
            init_input = torch.cat(
                [condition_info.view(batch_size, 1, -1), init_estimate], dim=-1
            )  # B x 1 x 410 (this is with the CLIFF condition)
        else:
            init_input = init_estimate
        token_embeddings = self.init_to_token_mhr(init_input).view(
            batch_size, 1, -1
        )  # B x 1 x 1024 (linear layered)

        num_pose_token = token_embeddings.shape[1]
        assert num_pose_token == 1

        image_augment, token_augment, token_mask = None, None, None
        token_context_gate = None          # contact blind hook, set with token_mask
        if hasattr(self, "prompt_encoder") and keypoints is not None:
            if prev_estimate is None:
                # Use initial embedding if no previous embedding
                prev_estimate = init_estimate
            # Previous estimate w/o the CLIFF condition.
            prev_embeddings = self.prev_to_token_mhr(prev_estimate).view(
                batch_size, 1, -1
            )  # 407 -> B x 1 x 1024; linear layer-ed

            if self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr",
                "vit",
                "vit_b",
                "vit_l",
            ]:
                # ViT backbone assumes a different aspect ratio as input size
                image_augment = self.prompt_encoder.get_dense_pe((16, 16))[
                    :, :, :, 2:-2
                ]
            elif self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr_512_384",
            ]:
                # ViT backbone assumes a different aspect ratio as input size
                image_augment = self.prompt_encoder.get_dense_pe((32, 32))[
                    :, :, :, 4:-4
                ]
            else:
                image_augment = self.prompt_encoder.get_dense_pe(
                    image_embeddings.shape[-2:]
                )  # (1, C, H, W)

            image_embeddings = self.ray_cond_emb(image_embeddings, batch["ray_cond"])

            # To start, keypoints is all [0, 0, -2]. The points get sent into self.pe_layer._pe_encoding,
            # the labels determine the embedding weight (special one for -2, -1, then each of joint.)
            prompt_embeddings, prompt_mask = self.prompt_encoder(
                keypoints=keypoints
            )  # B x 1 x 1280
            prompt_embeddings = self.prompt_to_token(
                prompt_embeddings
            )  # Linear layered: B x 1 x 1024

            # Concatenate pose tokens and prompt embeddings as decoder input
            token_embeddings = torch.cat(
                [
                    token_embeddings,
                    prev_embeddings,
                    prompt_embeddings,
                ],
                dim=1,
            )

            token_augment = torch.zeros_like(token_embeddings)
            token_augment[:, [num_pose_token]] = prev_embeddings
            token_augment[:, (num_pose_token + 1) :] = prompt_embeddings
            token_mask = None

            if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
                # Put in a token for each hand
                hand_det_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.hand_box_embedding.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )  # B x 5 + 70 x 1024
                # No positional embeddings
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )  # B x 5 + 70 x 1024

            assert self.cfg.MODEL.DECODER.get("DO_KEYPOINT_TOKENS", False)
            # Put in a token for each keypoint
            kps_emb_start_idx = token_embeddings.shape[1]
            token_embeddings = torch.cat(
                [
                    token_embeddings,
                    self.keypoint_embedding.weight[None, :, :].repeat(batch_size, 1, 1),
                ],
                dim=1,
            )  # B x 3 + 70 x 1024
            # No positional embeddings
            token_augment = torch.cat(
                [
                    token_augment,
                    torch.zeros_like(token_embeddings[:, token_augment.shape[1] :, :]),
                ],
                dim=1,
            )  # B x 3 + 70 x 1024
            if self.cfg.MODEL.DECODER.get("DO_KEYPOINT3D_TOKENS", False):
                # Put in a token for each keypoint
                kps3d_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.keypoint3d_embedding.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )  # B x 3 + 70 + 70 x 1024
                # No positional embeddings
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )  # B x 3 + 70 + 70 x 1024

            # Add contact tokens if enabled
            do_contact_tokens = self.cfg.MODEL.DECODER.get("DO_CONTACT_TOKENS", False)
            do_force_tokens = self.cfg.MODEL.DECODER.get("DO_FORCE_TOKENS", False)
            do_motion_tokens = self.cfg.MODEL.DECODER.get("DO_MOTION_TOKENS", False)
            # --- cond input hook (per-frame token conditioning) ---
            # One 10-d feature row per frame, projected (zero-init) and added to
            # both extra token blocks below.
            cond_feature = (
                self._cond_input_feature(batch, batch_size, token_embeddings)
                if self.cond_input_dim else None
            )
            # --- end cond input hook ---

            if do_contact_tokens:
                contact_emb_start_idx = token_embeddings.shape[1]
                contact_emb = self.contact_embedding.weight[None, :, :].repeat(
                    batch_size, 1, 1
                )
                # --- cond input hook (contact tokens, pre_decoder) ---
                if (cond_feature is not None
                        and self.cond_input_injection == "pre_decoder"
                        and hasattr(self, "contact_cond_linear")):
                    contact_emb = contact_emb + self.contact_cond_linear(
                        cond_feature).unsqueeze(1)
                # --- end cond input hook ---
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        contact_emb,
                    ],
                    dim=1,
                )
                # No positional embeddings for contact tokens
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )

            # --- force hook (append tokens after contact) ---
            # Force tokens live after the contact block (or directly after the
            # original tokens in a force-only build); the mask below is built
            # once the sequence is complete so N_total is correct.
            if do_force_tokens:
                force_emb_start_idx = token_embeddings.shape[1]
                force_emb = self.force_embedding.weight[None, :, :].repeat(
                    batch_size, 1, 1
                )
                # --- cond input hook (force tokens, pre_decoder) ---
                if (cond_feature is not None
                        and self.cond_input_injection == "pre_decoder"
                        and hasattr(self, "force_cond_linear")):
                    force_emb = force_emb + self.force_cond_linear(
                        cond_feature).unsqueeze(1)
                # --- end cond input hook ---
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        force_emb,
                    ],
                    dim=1,
                )
                # No positional embeddings for force tokens
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )
            # --- end force hook ---

            # --- motion hook (append tokens last) ---
            # Motion tokens live after every other appended block, so the mask
            # below gives original/contact/force ⊥ motion for free.
            if do_motion_tokens:
                motion_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.motion_embedding.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )
                # No positional embeddings for motion tokens
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )
            # --- end motion hook ---

            if do_contact_tokens or do_force_tokens or do_motion_tokens:
                # Asymmetric attention mask (True=allowed). Original tokens
                # never attend contact, force or motion tokens. Among the
                # appended blocks, EXTRA_TOKEN_ATTENTION picks the regime:
                # 'causal' bars every earlier block from the later ones
                # (contact ⊥ force ⊥ motion), 'mutual' keeps only the barrier
                # in front of the FIRST appended block so contact, force and
                # motion tokens fully inter-attend. Built after all extra
                # blocks so N_total is final.
                _block_starts = (
                    ([contact_emb_start_idx] if do_contact_tokens else [])
                    + ([force_emb_start_idx] if do_force_tokens else [])
                    + ([motion_emb_start_idx] if do_motion_tokens else []))
                if self.cfg.MODEL.get("EXTRA_TOKEN_ATTENTION", "causal") == "mutual":
                    _block_starts = _block_starts[:1]
                token_mask = self._build_block_token_mask(
                    batch_size,
                    token_embeddings.shape[1],
                    _block_starts,
                    token_embeddings.device,
                )

                # --- contact blind hook (gate the image cross-attention) ---
                # Every token cross-attends the image embeddings, and that
                # attention is unmasked. A fully-masked query row would make
                # softmax produce NaN, so the ablation instead zeroes the contact
                # rows of the cross-attention *output* before its residual add.
                # Cross-attention is independent per query row (keys/values are
                # image-only), so this removes the contact tokens' image access
                # without perturbing any other row by a single ulp.
                if do_contact_tokens and self.contact_blind_to_image:
                    token_context_gate = torch.ones(
                        1, token_embeddings.shape[1], 1,
                        dtype=token_embeddings.dtype,
                        device=token_embeddings.device,
                    )
                    token_context_gate[
                        :, contact_emb_start_idx :
                             contact_emb_start_idx + self.total_contact_tokens
                    ] = 0.0
                # --- end contact blind hook ---

        # We're doing intermediate model predictions
        def token_to_pose_output_fn(tokens, prev_pose_output, layer_idx,
                                    use_ft_heads=False):
            # Get the pose token
            pose_token = tokens[:, 0]

            prev_pose = init_pose.view(batch_size, -1)
            prev_camera = init_camera.view(batch_size, -1)

            # --- contact split-head hook ---
            # In-decoder (interm) calls always use the frozen original heads,
            # so the per-layer keypoint-token refresh — and with it every
            # frozen token trajectory — stays bit-identical to the base model.
            # Only the FINAL readout (the post-brick recompute below) applies
            # the fine-tuned copies, when they exist.
            _proj_pose = (getattr(self, "head_pose_ft_proj", None)
                          if use_ft_heads else None)
            _proj_cam = (getattr(self, "head_camera_ft_proj", None)
                         if use_ft_heads else None)
            # --- end contact split-head hook ---

            # Get pose outputs
            pose_output = self.head_pose(pose_token, prev_pose, proj=_proj_pose)
            # Get Camera Translation
            if hasattr(self, "head_camera"):
                pred_cam = self.head_camera(pose_token, prev_camera,
                                            proj=_proj_cam)
                pose_output["pred_cam"] = pred_cam
            # Run camera projection
            pose_output = self.camera_project(pose_output, batch)

            # Get 2D KPS in crop
            pose_output["pred_keypoints_2d_cropped"] = self._full_to_crop(
                batch, pose_output["pred_keypoints_2d"], self.body_batch_idx
            )

            return pose_output

        kp_token_update_fn = self.keypoint_token_update_fn

        # Now for 3D
        kp3d_token_update_fn = self.keypoint3d_token_update_fn

        # Contact token update (PE + local image-feature sampling at its anchors).
        ct_token_update_fn = (
            self.contact_token_update_fn
            if self.cfg.MODEL.DECODER.get("DO_CONTACT_TOKENS", False)
            else None
        )

        # --- force hook (per-layer anchored update) ---
        ft_token_update_fn = (
            self.force_token_update_fn
            if self.cfg.MODEL.DECODER.get("DO_FORCE_TOKENS", False)
            else None
        )
        # --- end force hook ---

        # --- motion hook (per-layer anchored update) ---
        # Not registered at all under `model.motion_head.anchored: false`: the
        # motion tokens are pure learned queries and their two projections do
        # not exist (see the motion unanchored hook in __init__).
        mt_token_update_fn = (
            self.motion_token_update_fn
            if (self.cfg.MODEL.DECODER.get("DO_MOTION_TOKENS", False)
                and getattr(self, "motion_anchored", True))
            else None
        )
        # --- end motion hook ---

        # Combine the 2D, 3D, and contact update functions
        def keypoint_token_update_fn_comb(*args):
            if kp_token_update_fn is not None:
                args = kp_token_update_fn(kps_emb_start_idx, image_embeddings, *args)
            if kp3d_token_update_fn is not None:
                args = kp3d_token_update_fn(kps3d_emb_start_idx, *args)
            if ct_token_update_fn is not None:
                args = ct_token_update_fn(
                    contact_emb_start_idx, image_embeddings,
                    self.decoder.layers, batch, *args
                )
            # --- force hook: force tokens sample the shared decoder image
            # tensor, anchored at the contact anchors ---
            if ft_token_update_fn is not None:
                args = ft_token_update_fn(
                    force_emb_start_idx, image_embeddings, self.decoder.layers, *args
                )
            # --- motion hook: motion tokens sample the shared decoder image
            # tensor, anchored at their own MHR70 keypoints ---
            if mt_token_update_fn is not None:
                args = mt_token_update_fn(
                    motion_emb_start_idx, image_embeddings, self.decoder.layers,
                    batch, *args
                )
            return args

        pose_token, pose_output = self.decoder(
            token_embeddings,
            image_embeddings,
            token_augment,
            image_augment,
            token_mask,
            token_to_pose_output_fn=token_to_pose_output_fn,
            keypoint_token_update_fn=keypoint_token_update_fn_comb,
            token_context_gate=token_context_gate,
        )

        _ft_recompute_done = [False]

        def _recompute_final_pose_output(pose_output):
            # Recompute the FINAL pose output from the CURRENT (updated) pose
            # token. The decoder returns the interm list with the final output
            # last — only the final one is replaced. Shared by every hook that
            # is allowed to move the pose token (pose_temporal, or the
            # cross-modal block with the 'pose' modality). Uses the
            # fine-tuned head copies when they exist (split-head): the frozen
            # anchors below therefore really are the FROZEN model's outputs —
            # the in-decoder final entry was produced by the original heads.
            _old = (pose_output[-1] if isinstance(pose_output, (list, tuple))
                    else pose_output)
            _final = token_to_pose_output_fn(
                pose_token, None, len(self.decoder.layers) - 1,
                use_ft_heads=True)
            _ft_recompute_done[0] = True
            # The first recompute's predecessor is the frozen model's own final
            # output; carry its camera translation, global orientation and
            # shape/scale coefficients through every later recompute as the
            # anchors for the trust-region / rail losses.
            _final["pred_cam_t_frozen"] = _old.get(
                "pred_cam_t_frozen", _old["pred_cam_t"].detach())
            _final["global_rot_frozen"] = _old.get(
                "global_rot_frozen", _old["global_rot"].detach())
            _final["shape_frozen"] = _old.get(
                "shape_frozen", _old["shape"].detach())
            _final["scale_frozen"] = _old.get(
                "scale_frozen", _old["scale"].detach())
            if isinstance(pose_output, (list, tuple)):
                return list(pose_output[:-1]) + [_final]
            return _final

        # --- cross-modal temporal hook (post_decoder) ---
        # ONE RoPE temporal block over the CONCATENATION of the participating
        # token blocks: every participating token attends every other one
        # across all of the clip's frames (and, at dt = 0, within its own
        # frame). Runs BEFORE pose_temporal below (mix across modalities first,
        # refine the pose slot after). Each participating slice is written
        # back; when 'pose' participates the final pose output is recomputed
        # like the pose temporal hook below.
        _xm = getattr(self, "cross_modal_temporal", None)
        if _xm is not None:
            _bounds = {"pose": (0, 1)}
            if self.cfg.MODEL.DECODER.get("DO_CONTACT_TOKENS", False):
                _bounds["contact"] = (
                    contact_emb_start_idx,
                    contact_emb_start_idx + self.total_contact_tokens)
            if self.cfg.MODEL.DECODER.get("DO_FORCE_TOKENS", False):
                _bounds["force"] = (
                    force_emb_start_idx,
                    force_emb_start_idx + self.num_force_tokens)
            if self.cfg.MODEL.DECODER.get("DO_MOTION_TOKENS", False):
                _bounds["motion"] = (
                    motion_emb_start_idx,
                    motion_emb_start_idx + self.num_motion_tokens)
            _sl, _pos, _valid = self._contact_temporal_fields(batch)
            _slices = [_bounds[m] for m in self.cross_modal_modalities]
            _updated = _xm(
                torch.cat([pose_token[:, lo:hi] for lo, hi in _slices], dim=1),
                _sl, _pos, _valid)
            # Scatter the updated slices back (ordered + disjoint by the
            # canonical modality order; everything between them is untouched).
            _pieces, _off, _prev = [], 0, 0
            for _lo, _hi in _slices:
                _k = _hi - _lo
                _pieces += [pose_token[:, _prev:_lo], _updated[:, _off:_off + _k]]
                _off, _prev = _off + _k, _hi
            _pieces.append(pose_token[:, _prev:])
            pose_token = torch.cat(_pieces, dim=1)
            if "pose" in self.cross_modal_modalities:
                pose_output = _recompute_final_pose_output(pose_output)
        # --- end cross-modal temporal hook ---

        # --- pose temporal hook (pose slot only) ---
        # E2, the DELIBERATE exception to the frozen-pose rule: the pose
        # modality's own temporal block. Mixes the pose token (index 0) across
        # the clip's frames and recomputes the FINAL pose output from the
        # updated token. Intermediate predictions, the keypoint-token updates
        # and every other token block read the untouched token —
        # contact/force/motion outputs cannot move. Zero-init gates keep init
        # behavior exactly frozen.
        _pt = getattr(self, "pose_temporal", None)
        if _pt is not None:
            _sl, _pos, _valid = self._contact_temporal_fields(batch)
            _updated = _pt(pose_token[:, 0:1], _sl, _pos, _valid)
            pose_token = torch.cat([_updated, pose_token[:, 1:]], dim=1)
            pose_output = _recompute_final_pose_output(pose_output)
        # --- end pose temporal hook ---

        # --- modality token prep (slice + conditioning) ---
        # Everything cross-frame and cross-modal already happened above, in the
        # single cross_modal_temporal block; here the per-modality slices are
        # taken and handed to their heads.
        contact_tokens = None
        if self.cfg.MODEL.DECODER.get("DO_CONTACT_TOKENS", False):
            contact_tokens = pose_token[
                :, contact_emb_start_idx : contact_emb_start_idx + self.total_contact_tokens
            ]
            # --- cond input hook (contact tokens, post_decoder) ---
            # Added out-of-place to the decoder's contact-token outputs (the
            # returned pose_token stays unconditioned). Same zero-init
            # projection as pre_decoder — only the injection site moves.
            if (cond_feature is not None
                    and self.cond_input_injection == "post_decoder"
                    and hasattr(self, "contact_cond_linear")):
                contact_tokens = contact_tokens + self.contact_cond_linear(
                    cond_feature).unsqueeze(1)
            # --- end cond input hook ---

        # --- force hook (prep force tokens) ---
        force_tokens = None
        if self.cfg.MODEL.DECODER.get("DO_FORCE_TOKENS", False):
            force_tokens = pose_token[
                :, force_emb_start_idx : force_emb_start_idx + self.num_force_tokens
            ]
            # --- cond input hook (force tokens, post_decoder) ---
            if (cond_feature is not None
                    and self.cond_input_injection == "post_decoder"
                    and hasattr(self, "force_cond_linear")):
                force_tokens = force_tokens + self.force_cond_linear(
                    cond_feature).unsqueeze(1)
            # --- end cond input hook ---
        # --- end force hook ---

        # --- motion hook (prep motion tokens) ---
        motion_tokens = None
        if self.cfg.MODEL.DECODER.get("DO_MOTION_TOKENS", False):
            motion_tokens = pose_token[
                :, motion_emb_start_idx : motion_emb_start_idx + self.num_motion_tokens
            ]
        # --- end motion hook ---

        # --- contact split-head hook (final readout) ---
        # A fine-tuned head copy must always produce the FINAL output, even
        # when no pose-writing brick triggered a recompute above (finetune-only
        # runs with no temporal module).
        if (not _ft_recompute_done[0]) and (
                getattr(self, "head_pose_ft_proj", None) is not None
                or getattr(self, "head_camera_ft_proj", None) is not None):
            pose_output = _recompute_final_pose_output(pose_output)
        # --- end contact split-head hook ---

        # Process contact tokens if enabled
        contact_output = None
        if contact_tokens is not None:
            # One head per target -> {"<target>_logits": [B, D], "<target>_probs": ...}
            contact_output = {}
            for name, head in self.head_contact.items():
                logits = head(contact_tokens)
                contact_output[f"{name}_logits"] = logits
                contact_output[f"{name}_probs"] = torch.sigmoid(logits)

        # --- force hook (process force tokens) ---
        force_output = None
        if force_tokens is not None:
            # Dimensionless per-extremity force vectors (units of body weight, D5).
            joint_forces = self.head_force(force_tokens)                # [B, K, 3]
            force_output = {"joint_forces": joint_forces}
            # --- force contact-gate hook (final output) ---
            # Gate the FINAL force output (post temporal mixing) by the DETACHED
            # per-group contact logits, so eval/inference/rendering all see gated
            # forces unchanged; the ungated tensor stays for diagnostics. The
            # detach keeps the force loss from rewriting the calibrated contact
            # probabilities through this product (contact trains from its labels).
            if getattr(self, "force_contact_gate", False):
                force_output = {
                    "joint_forces": contact_gate_forces(
                        joint_forces,
                        contact_output["joint_logits"],
                        self.force_contact_gate_sharpness,
                    ),
                    "joint_forces_raw": joint_forces,
                }
            # --- end force contact-gate hook ---
        # --- end force hook ---

        # --- motion hook (process motion tokens) ---
        motion_output = None
        if motion_tokens is not None:
            # Standardized root-frame linear velocity + acceleration per token
            # (+ the root angular pair on 12-wide heads); the supervised loss
            # owns the mean/std table.
            motion = self.head_motion(motion_tokens)                # [B, K, 6|12]
            motion_output = {
                "joint_vel": motion[..., 0:3],
                "joint_acc": motion[..., 3:6],
                "joint_motion": motion,
            }
            if motion.shape[-1] == 12:
                motion_output["joint_ang_vel"] = motion[..., 6:9]
                motion_output["joint_ang_acc"] = motion[..., 9:12]
        # --- end motion hook ---

        if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
            return (
                pose_token[:, hand_det_emb_start_idx : hand_det_emb_start_idx + 2],
                pose_output,
                contact_output,
                force_output,
                motion_output,
            )
        else:
            return pose_token, pose_output, contact_output, force_output, motion_output

    def forward_decoder_hand(
        self,
        image_embeddings: torch.Tensor,
        init_estimate: Optional[torch.Tensor] = None,
        keypoints: Optional[torch.Tensor] = None,
        prev_estimate: Optional[torch.Tensor] = None,
        condition_info: Optional[torch.Tensor] = None,
        batch=None,
    ):
        """
        Args:
            image_embeddings: image features from the backbone, shape (B, C, H, W)
            init_estimate: initial estimate to be refined on, shape (B, 1, C)
            keypoints: optional prompt input, shape (B, N, 3),
                3 for coordinates (x,y) + label.
                (x, y) should be normalized to range [0, 1].
                label==-1 indicates incorrect points,
                label==-2 indicates invalid points
            prev_estimate: optional prompt input, shape (B, 1, C),
                previous estimate for pose refinement.
            condition_info: optional condition information that is concatenated with
                the input tokens, shape (B, c)
        """
        batch_size = image_embeddings.shape[0]

        # Initial estimation for residual prediction.
        if init_estimate is None:
            init_pose = self.init_pose_hand.weight.expand(batch_size, -1).unsqueeze(
                dim=1
            )
            if hasattr(self, "init_camera_hand"):
                init_camera = self.init_camera_hand.weight.expand(
                    batch_size, -1
                ).unsqueeze(dim=1)

            init_estimate = (
                init_pose
                if not hasattr(self, "init_camera_hand")
                else torch.cat([init_pose, init_camera], dim=-1)
            )  # This is basically pose & camera translation at the end. B x 1 x (404 + 3)

        if condition_info is not None:
            init_input = torch.cat(
                [condition_info.view(batch_size, 1, -1), init_estimate], dim=-1
            )  # B x 1 x 410 (this is with the CLIFF condition)
        else:
            init_input = init_estimate
        token_embeddings = self.init_to_token_mhr_hand(init_input).view(
            batch_size, 1, -1
        )  # B x 1 x 1024 (linear layered)
        num_pose_token = token_embeddings.shape[1]

        image_augment, token_augment, token_mask = None, None, None
        if hasattr(self, "prompt_encoder") and keypoints is not None:
            if prev_estimate is None:
                # Use initial embedding if no previous embedding
                prev_estimate = init_estimate
            # Previous estimate w/o the CLIFF condition.
            prev_embeddings = self.prev_to_token_mhr_hand(prev_estimate).view(
                batch_size, 1, -1
            )  # 407 -> B x 1 x 1024; linear layer-ed

            if self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr",
                "vit",
                "vit_b",
                "vit_l",
            ]:
                # ViT backbone assumes a different aspect ratio as input size
                image_augment = self.hand_pe_layer((16, 16)).unsqueeze(0)[:, :, :, 2:-2]
            elif self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr_512_384",
            ]:
                # ViT backbone assumes a different aspect ratio as input size
                image_augment = self.hand_pe_layer((32, 32)).unsqueeze(0)[:, :, :, 4:-4]
            else:
                image_augment = self.hand_pe_layer(
                    image_embeddings.shape[-2:]
                ).unsqueeze(
                    0
                )  # (1, C, H, W)

            image_embeddings = self.ray_cond_emb_hand(
                image_embeddings, batch["ray_cond_hand"]
            )

            # To start, keypoints is all [0, 0, -2]. The points get sent into self.pe_layer._pe_encoding,
            # the labels determine the embedding weight (special one for -2, -1, then each of joint.)
            prompt_embeddings, prompt_mask = self.prompt_encoder(
                keypoints=keypoints
            )  # B x 1 x 1280
            prompt_embeddings = self.prompt_to_token(
                prompt_embeddings
            )  # Linear layered: B x 1 x 1024

            # Concatenate pose tokens and prompt embeddings as decoder input
            token_embeddings = torch.cat(
                [
                    token_embeddings,
                    prev_embeddings,
                    prompt_embeddings,
                ],
                dim=1,
            )

            token_augment = torch.zeros_like(token_embeddings)
            token_augment[:, [num_pose_token]] = prev_embeddings
            token_augment[:, (num_pose_token + 1) :] = prompt_embeddings
            token_mask = None

            if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
                # Put in a token for each hand
                hand_det_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.hand_box_embedding.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )  # B x 5 + 70 x 1024
                # No positional embeddings
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )  # B x 5 + 70 x 1024

            assert self.cfg.MODEL.DECODER.get("DO_KEYPOINT_TOKENS", False)
            # Put in a token for each keypoint
            kps_emb_start_idx = token_embeddings.shape[1]
            token_embeddings = torch.cat(
                [
                    token_embeddings,
                    self.keypoint_embedding_hand.weight[None, :, :].repeat(
                        batch_size, 1, 1
                    ),
                ],
                dim=1,
            )  # B x 3 + 70 x 1024
            # No positional embeddings
            token_augment = torch.cat(
                [
                    token_augment,
                    torch.zeros_like(token_embeddings[:, token_augment.shape[1] :, :]),
                ],
                dim=1,
            )  # B x 3 + 70 x 1024

            if self.cfg.MODEL.DECODER.get("DO_KEYPOINT3D_TOKENS", False):
                # Put in a token for each keypoint
                kps3d_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.keypoint3d_embedding_hand.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )  # B x 3 + 70 + 70 x 1024
                # No positional embeddings
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )  # B x 3 + 70 + 70 x 1024

        # We're doing intermediate model predictions
        def token_to_pose_output_fn(tokens, prev_pose_output, layer_idx):
            # Get the pose token
            pose_token = tokens[:, 0]

            prev_pose = init_pose.view(batch_size, -1)
            prev_camera = init_camera.view(batch_size, -1)

            # Get pose outputs
            pose_output = self.head_pose_hand(pose_token, prev_pose)

            # Get Camera Translation
            if hasattr(self, "head_camera_hand"):
                pred_cam = self.head_camera_hand(pose_token, prev_camera)
                pose_output["pred_cam"] = pred_cam
            # Run camera projection
            pose_output = self.camera_project_hand(pose_output, batch)

            # Get 2D KPS in crop
            pose_output["pred_keypoints_2d_cropped"] = self._full_to_crop(
                batch, pose_output["pred_keypoints_2d"], self.hand_batch_idx
            )

            return pose_output

        kp_token_update_fn = self.keypoint_token_update_fn_hand

        # Now for 3D
        kp3d_token_update_fn = self.keypoint3d_token_update_fn_hand

        # Combine the 2D and 3D update functions
        def keypoint_token_update_fn_comb(*args):
            if kp_token_update_fn is not None:
                args = kp_token_update_fn(kps_emb_start_idx, image_embeddings, *args)
            if kp3d_token_update_fn is not None:
                args = kp3d_token_update_fn(kps3d_emb_start_idx, *args)
            return args

        pose_token, pose_output = self.decoder_hand(
            token_embeddings,
            image_embeddings,
            token_augment,
            image_augment,
            token_mask,
            token_to_pose_output_fn=token_to_pose_output_fn,
            keypoint_token_update_fn=keypoint_token_update_fn_comb,
        )

        if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
            return (
                pose_token[:, hand_det_emb_start_idx : hand_det_emb_start_idx + 2],
                pose_output,
            )
        else:
            return pose_token, pose_output

    @torch.no_grad()
    def _get_keypoint_prompt(self, batch, pred_keypoints_2d, force_dummy=False):
        if self.camera_type == "perspective":
            pred_keypoints_2d = self._full_to_crop(batch, pred_keypoints_2d)

        gt_keypoints_2d = self._flatten_person(batch["keypoints_2d"]).clone()

        keypoint_prompt = self.keypoint_prompt_sampler.sample(
            gt_keypoints_2d,
            pred_keypoints_2d,
            is_train=self.training,
            force_dummy=force_dummy,
        )
        return keypoint_prompt

    def _get_mask_prompt(self, batch, image_embeddings):
        x_mask = self._flatten_person(batch["mask"])
        mask_embeddings, no_mask_embeddings = self.prompt_encoder.get_mask_embeddings(
            x_mask, image_embeddings.shape[0], image_embeddings.shape[2:]
        )
        if self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr",
            "vit",
        ]:
            # ViT backbone assumes a different aspect ratio as input size
            mask_embeddings = mask_embeddings[:, :, :, 2:-2]
        elif self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr_512_384",
        ]:
            # for x2 resolution
            mask_embeddings = mask_embeddings[:, :, :, 4:-4]

        mask_score = self._flatten_person(batch["mask_score"]).view(-1, 1, 1, 1)
        mask_embeddings = torch.where(
            mask_score > 0,
            mask_score * mask_embeddings.to(image_embeddings),
            no_mask_embeddings.to(image_embeddings),
        )
        return mask_embeddings

    def _one_prompt_iter(self, batch, output, prev_prompt, full_output):
        image_embeddings = output["image_embeddings"]
        condition_info = output["condition_info"]

        if "mhr" in output and output["mhr"] is not None:
            pose_output = output["mhr"]  # body-only output
            # Use previous estimate as initialization
            prev_estimate = torch.cat(
                [
                    pose_output["pred_pose_raw"].detach(),  # (B, 6)
                    pose_output["shape"].detach(),
                    pose_output["scale"].detach(),
                    pose_output["hand"].detach(),
                    pose_output["face"].detach(),
                ],
                dim=1,
            ).unsqueeze(dim=1)
            if hasattr(self, "init_camera"):
                prev_estimate = torch.cat(
                    [prev_estimate, pose_output["pred_cam"].detach().unsqueeze(1)],
                    dim=-1,
                )
            prev_shape = prev_estimate.shape[1:]

            pred_keypoints_2d = output["mhr"]["pred_keypoints_2d"].detach().clone()
            kpt_shape = pred_keypoints_2d.shape[1:]

        if "mhr_hand" in output and output["mhr_hand"] is not None:
            pose_output_hand = output["mhr_hand"]
            # Use previous estimate as initialization
            prev_estimate_hand = torch.cat(
                [
                    pose_output_hand["pred_pose_raw"].detach(),  # (B, 6)
                    pose_output_hand["shape"].detach(),
                    pose_output_hand["scale"].detach(),
                    pose_output_hand["hand"].detach(),
                    pose_output_hand["face"].detach(),
                ],
                dim=1,
            ).unsqueeze(dim=1)
            if hasattr(self, "init_camera_hand"):
                prev_estimate_hand = torch.cat(
                    [
                        prev_estimate_hand,
                        pose_output_hand["pred_cam"].detach().unsqueeze(1),
                    ],
                    dim=-1,
                )
            prev_shape = prev_estimate_hand.shape[1:]

            pred_keypoints_2d_hand = (
                output["mhr_hand"]["pred_keypoints_2d"].detach().clone()
            )
            kpt_shape = pred_keypoints_2d_hand.shape[1:]

        all_prev_estimate = torch.zeros(
            (image_embeddings.shape[0], *prev_shape), device=image_embeddings.device
        )
        if "mhr" in output and output["mhr"] is not None:
            all_prev_estimate[self.body_batch_idx] = prev_estimate
        if "mhr_hand" in output and output["mhr_hand"] is not None:
            all_prev_estimate[self.hand_batch_idx] = prev_estimate_hand

        # Get keypoint prompts
        all_pred_keypoints_2d = torch.zeros(
            (image_embeddings.shape[0], *kpt_shape), device=image_embeddings.device
        )
        if "mhr" in output and output["mhr"] is not None:
            all_pred_keypoints_2d[self.body_batch_idx] = pred_keypoints_2d
        if "mhr_hand" in output and output["mhr_hand"] is not None:
            all_pred_keypoints_2d[self.hand_batch_idx] = pred_keypoints_2d_hand

        keypoint_prompt = self._get_keypoint_prompt(batch, all_pred_keypoints_2d)
        if len(prev_prompt):
            cur_keypoint_prompt = torch.cat(prev_prompt + [keypoint_prompt], dim=1)
        else:
            cur_keypoint_prompt = keypoint_prompt  # [B, 1, 3]

        pose_output, pose_output_hand = None, None
        contact_output = None
        force_output = None
        motion_output = None
        if len(self.body_batch_idx):
            (tokens_output, pose_output, contact_output, force_output,
             motion_output) = self.forward_decoder(
                image_embeddings[self.body_batch_idx],
                init_estimate=None,  # not recurring previous estimate
                keypoints=cur_keypoint_prompt[self.body_batch_idx],
                prev_estimate=all_prev_estimate[self.body_batch_idx],
                condition_info=condition_info[self.body_batch_idx],
                batch=batch,
                full_output=None,
            )
            pose_output = pose_output[-1]

        # Update prediction output
        output.update(
            {
                "mhr": pose_output,
                "mhr_hand": pose_output_hand,
                "contact": contact_output,
                "force": force_output,
                "motion": motion_output,
            }
        )

        return output, keypoint_prompt

    def _full_to_crop(
        self,
        batch: Dict,
        pred_keypoints_2d: torch.Tensor,
        batch_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        """Convert full-image keypoints coordinates to crop and normalize to [-0.5. 0.5]"""
        pred_keypoints_2d_cropped = torch.cat(
            [pred_keypoints_2d, torch.ones_like(pred_keypoints_2d[:, :, [-1]])], dim=-1
        )
        if batch_idx is not None:
            affine_trans = self._flatten_person(batch["affine_trans"])[batch_idx].to(
                pred_keypoints_2d_cropped
            )
            img_size = self._flatten_person(batch["img_size"])[batch_idx].unsqueeze(1)
        else:
            affine_trans = self._flatten_person(batch["affine_trans"]).to(
                pred_keypoints_2d_cropped
            )
            img_size = self._flatten_person(batch["img_size"]).unsqueeze(1)
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped @ affine_trans.mT
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[..., :2] / img_size - 0.5

        return pred_keypoints_2d_cropped

    def camera_project(self, pose_output: Dict, batch: Dict) -> Dict:
        """
        Project 3D keypoints to 2D using the camera parameters.
        Args:
            pose_output (Dict): Dictionary containing the pose output.
            batch (Dict): Dictionary containing the batch data.
        Returns:
            Dict: Dictionary containing the projected 2D keypoints.
        """
        if hasattr(self, "head_camera"):
            head_camera = self.head_camera
            pred_cam = pose_output["pred_cam"]
        else:
            assert False

        cam_out = head_camera.perspective_projection(
            pose_output["pred_keypoints_3d"],
            pred_cam,
            self._flatten_person(batch["bbox_center"])[self.body_batch_idx],
            self._flatten_person(batch["bbox_scale"])[self.body_batch_idx, 0],
            self._flatten_person(batch["ori_img_size"])[self.body_batch_idx],
            self._flatten_person(
                batch["cam_int"]
                .unsqueeze(1)
                .expand(-1, batch["img"].shape[1], -1, -1)
                .contiguous()
            )[self.body_batch_idx],
            use_intrin_center=self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False),
        )

        if pose_output.get("pred_vertices", None) is not None:
            cam_out_vertices = head_camera.perspective_projection(
                pose_output["pred_vertices"],
                pred_cam,
                self._flatten_person(batch["bbox_center"])[self.body_batch_idx],
                self._flatten_person(batch["bbox_scale"])[self.body_batch_idx, 0],
                self._flatten_person(batch["ori_img_size"])[self.body_batch_idx],
                self._flatten_person(
                    batch["cam_int"]
                    .unsqueeze(1)
                    .expand(-1, batch["img"].shape[1], -1, -1)
                    .contiguous()
                )[self.body_batch_idx],
                use_intrin_center=self.cfg.MODEL.DECODER.get(
                    "USE_INTRIN_CENTER", False
                ),
            )
            pose_output["pred_keypoints_2d_verts"] = cam_out_vertices[
                "pred_keypoints_2d"
            ]

        pose_output.update(cam_out)

        return pose_output

    def camera_project_hand(self, pose_output: Dict, batch: Dict) -> Dict:
        """
        Project 3D keypoints to 2D using the camera parameters.
        Args:
            pose_output (Dict): Dictionary containing the pose output.
            batch (Dict): Dictionary containing the batch data.
        Returns:
            Dict: Dictionary containing the projected 2D keypoints.
        """
        if hasattr(self, "head_camera_hand"):
            head_camera = self.head_camera_hand
            pred_cam = pose_output["pred_cam"]
        else:
            assert False

        cam_out = head_camera.perspective_projection(
            pose_output["pred_keypoints_3d"],
            pred_cam,
            self._flatten_person(batch["bbox_center"])[self.hand_batch_idx],
            self._flatten_person(batch["bbox_scale"])[self.hand_batch_idx, 0],
            self._flatten_person(batch["ori_img_size"])[self.hand_batch_idx],
            self._flatten_person(
                batch["cam_int"]
                .unsqueeze(1)
                .expand(-1, batch["img"].shape[1], -1, -1)
                .contiguous()
            )[self.hand_batch_idx],
            use_intrin_center=self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False),
        )

        if pose_output.get("pred_vertices", None) is not None:
            cam_out_vertices = head_camera.perspective_projection(
                pose_output["pred_vertices"],
                pred_cam,
                self._flatten_person(batch["bbox_center"])[self.hand_batch_idx],
                self._flatten_person(batch["bbox_scale"])[self.hand_batch_idx, 0],
                self._flatten_person(batch["ori_img_size"])[self.hand_batch_idx],
                self._flatten_person(
                    batch["cam_int"]
                    .unsqueeze(1)
                    .expand(-1, batch["img"].shape[1], -1, -1)
                    .contiguous()
                )[self.hand_batch_idx],
                use_intrin_center=self.cfg.MODEL.DECODER.get(
                    "USE_INTRIN_CENTER", False
                ),
            )
            pose_output["pred_keypoints_2d_verts"] = cam_out_vertices[
                "pred_keypoints_2d"
            ]

        pose_output.update(cam_out)

        return pose_output

    def get_ray_condition(self, batch):
        B, N, _, H, W = batch["img"].shape
        meshgrid_xy = (
            torch.stack(
                torch.meshgrid(torch.arange(H), torch.arange(W), indexing="xy"), dim=2
            )[None, None, :, :, :]
            .repeat(B, N, 1, 1, 1)
            .cuda()
        )  # B x N x H x W x 2
        meshgrid_xy = (
            meshgrid_xy / batch["affine_trans"][:, :, None, None, [0, 1], [0, 1]]
        )
        meshgrid_xy = (
            meshgrid_xy
            - batch["affine_trans"][:, :, None, None, [0, 1], [2, 2]]
            / batch["affine_trans"][:, :, None, None, [0, 1], [0, 1]]
        )

        # Subtract out center & normalize to be rays
        meshgrid_xy = (
            meshgrid_xy - batch["cam_int"][:, None, None, None, [0, 1], [2, 2]]
        )
        meshgrid_xy = (
            meshgrid_xy / batch["cam_int"][:, None, None, None, [0, 1], [0, 1]]
        )

        return meshgrid_xy.permute(0, 1, 4, 2, 3).to(
            batch["img"].dtype
        )  # This is B x num_person x 2 x H x W

    def forward_pose_branch(self, batch: Dict, precomputed_features=None) -> Dict:
        """Run a forward pass for the crop-image (pose) branch.

        Args:
            batch: standard SAM-3D-Body batch dict.
            precomputed_features: optional [B*N, C, H, W] backbone embeddings.
                When provided, the backbone is skipped entirely.
        """
        batch_size, num_person = batch["img"].shape[:2]

        if precomputed_features is not None:
            # --- Skip backbone, use precomputed features ---
            # The cache stores the RAW backbone output (bf16, pre-cast, pre
            # mask-conditioning); this cast reproduces the live path's
            # bf16 -> img-dtype cast bit-exactly. Mask conditioning below and
            # ray conditioning inside forward_decoder still run live.
            if self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr", "vit", "vit_b", "vit_l", "vit_hmr_512_384",
            ]:
                raise NotImplementedError(
                    "precomputed_features assumes the full-square backbone input; "
                    f"backbone {self.cfg.MODEL.BACKBONE.TYPE!r} width-crops it")
            image_embeddings = precomputed_features.type(batch["img"].dtype)

            expected_h = self.cfg.MODEL.IMAGE_SIZE[0] // self.ray_cond_emb.patch_size
            expected_w = self.cfg.MODEL.IMAGE_SIZE[1] // self.ray_cond_emb.patch_size
            if image_embeddings.shape[-2:] != (expected_h, expected_w):
                raise ValueError(
                    f"precomputed features grid {tuple(image_embeddings.shape[-2:])} "
                    f"does not match the backbone output ({expected_h}, {expected_w}) "
                    "— the cache was built for a different model")

            # Still need ray conditioning for the decoder
            ray_cond = self.get_ray_condition(batch)
            ray_cond = self._flatten_person(ray_cond)
            if len(self.body_batch_idx):
                batch["ray_cond"] = ray_cond[self.body_batch_idx].clone()
            if len(self.hand_batch_idx):
                batch["ray_cond_hand"] = ray_cond[self.hand_batch_idx].clone()
        else:
            # --- Original backbone path ---
            # Forward backbone encoder
            x = self.data_preprocess(
                self._flatten_person(batch["img"]),
                crop_width=(
                    self.cfg.MODEL.BACKBONE.TYPE
                    in [
                        "vit_hmr",
                        "vit",
                        "vit_b",
                        "vit_l",
                        "vit_hmr_512_384",
                    ]
                ),
            )

            # Optionally get ray conditioining
            ray_cond = self.get_ray_condition(batch)  # This is B x num_person x 2 x H x W
            ray_cond = self._flatten_person(ray_cond)
            if self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr",
                "vit",
                "vit_b",
                "vit_l",
            ]:
                ray_cond = ray_cond[:, :, :, 32:-32]
            elif self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr_512_384",
            ]:
                ray_cond = ray_cond[:, :, :, 64:-64]

            if len(self.body_batch_idx):
                batch["ray_cond"] = ray_cond[self.body_batch_idx].clone()
            if len(self.hand_batch_idx):
                batch["ray_cond_hand"] = ray_cond[self.hand_batch_idx].clone()
            ray_cond = None

            # --- contact efficiency hook (backbone_no_grad) ---
            # The backbone is fully frozen (asserted at build in contact/model.py),
            # so wrapping ONLY this call in no_grad drops its activation graph without
            # touching any trainable param. Absent key = old behaviour.
            if self.cfg.MODEL.get("EFFICIENCY", {}).get("BACKBONE_NO_GRAD", False):
                with torch.no_grad():
                    image_embeddings = self.backbone(
                        x.type(self.backbone_dtype), extra_embed=ray_cond
                    )  # (B, C, H, W)
            else:
                image_embeddings = self.backbone(
                    x.type(self.backbone_dtype), extra_embed=ray_cond
                )  # (B, C, H, W)
            # --- end contact efficiency hook ---

            if isinstance(image_embeddings, tuple):
                image_embeddings = image_embeddings[-1]
            image_embeddings = image_embeddings.type(x.dtype)

        # Mask condition if available. Runs for precomputed features too: the
        # cache holds the raw backbone output, and this addition depends only on
        # batch["mask"]/["mask_score"] (never on embedding values).
        if self.cfg.MODEL.PROMPT_ENCODER.get("MASK_EMBED_TYPE", None) is not None:
            # v1: non-iterative mask conditioning
            if self.cfg.MODEL.PROMPT_ENCODER.get("MASK_PROMPT", "v1") == "v1":
                mask_embeddings = self._get_mask_prompt(batch, image_embeddings)
                image_embeddings = image_embeddings + mask_embeddings
            else:
                raise NotImplementedError

        # Prepare input for promptable decoder
        condition_info = self._get_decoder_condition(batch)

        # Initial estimate with a dummy prompt
        keypoints_prompt = torch.zeros((batch_size * num_person, 1, 3)).to(batch["img"])
        keypoints_prompt[:, :, -1] = -2

        # Forward promptable decoder to get updated pose tokens and regression output
        pose_output, pose_output_hand = None, None
        contact_output = None
        force_output = None
        motion_output = None
        if len(self.body_batch_idx):
            (tokens_output, pose_output, contact_output, force_output,
             motion_output) = self.forward_decoder(
                image_embeddings[self.body_batch_idx],
                init_estimate=None,
                keypoints=keypoints_prompt[self.body_batch_idx],
                prev_estimate=None,
                condition_info=condition_info[self.body_batch_idx],
                batch=batch,
            )
            pose_output = pose_output[-1]
        if len(self.hand_batch_idx):
            tokens_output_hand, pose_output_hand = self.forward_decoder_hand(
                image_embeddings[self.hand_batch_idx],
                init_estimate=None,
                keypoints=keypoints_prompt[self.hand_batch_idx],
                prev_estimate=None,
                condition_info=condition_info[self.hand_batch_idx],
                batch=batch,
            )
            pose_output_hand = pose_output_hand[-1]

        output = {
            # "pose_token": pose_token,
            "mhr": pose_output,  # mhr prediction output
            "mhr_hand": pose_output_hand,  # mhr prediction output
            "contact": contact_output,  # contact prediction output (body decoder only)
            "force": force_output,  # per-extremity force output (body decoder only)
            "motion": motion_output,  # per-joint vel/acc output (body decoder only)
            "condition_info": condition_info,
            "image_embeddings": image_embeddings,
        }

        if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
            if len(self.body_batch_idx):
                output_hand_box_tokens = tokens_output
                hand_coords = self.bbox_embed(
                    output_hand_box_tokens
                ).sigmoid()  # x1, y1, w, h for body samples, 0 ~ 1
                hand_logits = self.hand_cls_embed(output_hand_box_tokens)

                output["mhr"]["hand_box"] = hand_coords
                output["mhr"]["hand_logits"] = hand_logits

            if len(self.hand_batch_idx):
                output_hand_box_tokens_hand_batch = tokens_output_hand

                hand_coords_hand_batch = self.bbox_embed(
                    output_hand_box_tokens_hand_batch
                ).sigmoid()  # x1, y1, w, h for hand samples
                hand_logits_hand_batch = self.hand_cls_embed(
                    output_hand_box_tokens_hand_batch
                )

                output["mhr_hand"]["hand_box"] = hand_coords_hand_batch
                output["mhr_hand"]["hand_logits"] = hand_logits_hand_batch

        return output

    def forward_step(
        self, batch: Dict, decoder_type: str = "body", precomputed_features=None,
    ) -> Tuple[Dict, Dict]:
        batch_size, num_person = batch["img"].shape[:2]

        if decoder_type == "body":
            self.hand_batch_idx = []
            self.body_batch_idx = list(range(batch_size * num_person))
        elif decoder_type == "hand":
            self.hand_batch_idx = list(range(batch_size * num_person))
            self.body_batch_idx = []
        else:
            ValueError("Invalid decoder type: ", decoder_type)

        # Crop-image (pose) branch
        pose_output = self.forward_pose_branch(batch, precomputed_features=precomputed_features)

        return pose_output

    def run_inference(
        self,
        img,
        batch: Dict,
        inference_type: str = "full",
        transform_hand: Any = None,
        thresh_wrist_angle=1.4,
    ):
        """
        Run 3DB inference (optionally with hand detector).

        inference_type:
            - full: full-body inference with both body and hand decoders
            - body: inference with body decoder only (still full-body output)
            - hand: inference with hand decoder only (only hand output)
        """

        height, width = img.shape[:2]
        cam_int = batch["cam_int"].clone()

        if inference_type == "body":
            pose_output = self.forward_step(batch, decoder_type="body")
            return pose_output
        elif inference_type == "hand":
            pose_output = self.forward_step(batch, decoder_type="hand")
            return pose_output
        elif not inference_type == "full":
            ValueError("Invalid inference type: ", inference_type)

        # Step 1. For full-body inference, we first inference with the body decoder.
        pose_output = self.forward_step(batch, decoder_type="body")
        left_xyxy, right_xyxy = self._get_hand_box(pose_output, batch)
        ori_local_wrist_rotmat = roma.euler_to_rotmat(
            "XZY",
            pose_output["mhr"]["body_pose"][:, [41, 43, 42, 31, 33, 32]].unflatten(
                1, (2, 3)
            ),
        )

        # Step 2. Re-run with each hand
        ## Left... Flip image & box
        flipped_img = img[:, ::-1]
        tmp = left_xyxy.copy()
        left_xyxy[:, 0] = width - tmp[:, 2] - 1
        left_xyxy[:, 2] = width - tmp[:, 0] - 1

        batch_lhand = prepare_batch(
            flipped_img, transform_hand, left_xyxy, cam_int=cam_int.clone()
        )
        batch_lhand = recursive_to(batch_lhand, "cuda")
        lhand_output = self.forward_step(batch_lhand, decoder_type="hand")

        # Unflip output
        ## Flip scale
        ### Get MHR values
        scale_r_hands_mean = self.head_pose.scale_mean[8].item()
        scale_l_hands_mean = self.head_pose.scale_mean[9].item()
        scale_r_hands_std = self.head_pose.scale_comps[8, 8].item()
        scale_l_hands_std = self.head_pose.scale_comps[9, 9].item()
        ### Apply
        lhand_output["mhr_hand"]["scale"][:, 9] = (
            (
                scale_r_hands_mean
                + scale_r_hands_std * lhand_output["mhr_hand"]["scale"][:, 8]
            )
            - scale_l_hands_mean
        ) / scale_l_hands_std
        ## Get the right hand global rotation, flip it, put it in as left.
        lhand_output["mhr_hand"]["joint_global_rots"][:, 78] = lhand_output["mhr_hand"][
            "joint_global_rots"
        ][:, 42].clone()
        lhand_output["mhr_hand"]["joint_global_rots"][:, 78, [1, 2], :] *= -1
        ### Flip hand pose
        lhand_output["mhr_hand"]["hand"][:, :54] = lhand_output["mhr_hand"]["hand"][
            :, 54:
        ]
        ### Unflip box
        batch_lhand["bbox_center"][:, :, 0] = (
            width - batch_lhand["bbox_center"][:, :, 0] - 1
        )

        ## Right...
        batch_rhand = prepare_batch(
            img, transform_hand, right_xyxy, cam_int=cam_int.clone()
        )
        batch_rhand = recursive_to(batch_rhand, "cuda")
        rhand_output = self.forward_step(batch_rhand, decoder_type="hand")

        # Step 3. replace hand pose estimation from the body decoder.
        ## CRITERIA 1: LOCAL WRIST POSE DIFFERENCE
        joint_rotations = pose_output["mhr"]["joint_global_rots"]
        ### Get lowarm
        lowarm_joint_idxs = torch.LongTensor([76, 40]).cuda()  # left, right
        lowarm_joint_rotations = joint_rotations[:, lowarm_joint_idxs]  # B x 2 x 3 x 3
        ### Get zero-wrist pose
        wrist_twist_joint_idxs = torch.LongTensor([77, 41]).cuda()  # left, right
        wrist_zero_rot_pose = (
            lowarm_joint_rotations
            @ self.head_pose.joint_rotation[wrist_twist_joint_idxs]
        )
        ### Get globals from left & right
        left_joint_global_rots = lhand_output["mhr_hand"]["joint_global_rots"]
        right_joint_global_rots = rhand_output["mhr_hand"]["joint_global_rots"]
        pred_global_wrist_rotmat = torch.stack(
            [
                left_joint_global_rots[:, 78],
                right_joint_global_rots[:, 42],
            ],
            dim=1,
        )
        ### Get the local poses that lead to the wrist being pred_global_wrist_rotmat
        fused_local_wrist_rotmat = torch.einsum(
            "kabc,kabd->kadc", pred_global_wrist_rotmat, wrist_zero_rot_pose
        )
        angle_difference = rotation_angle_difference(
            ori_local_wrist_rotmat, fused_local_wrist_rotmat
        )  # B x 2 x 3 x3
        angle_difference_valid_mask = angle_difference < thresh_wrist_angle

        ## CRITERIA 2: hand box size
        hand_box_size_thresh = 64
        hand_box_size_valid_mask = torch.stack(
            [
                (batch_lhand["bbox_scale"].flatten(0, 1) > hand_box_size_thresh).all(
                    dim=1
                ),
                (batch_rhand["bbox_scale"].flatten(0, 1) > hand_box_size_thresh).all(
                    dim=1
                ),
            ],
            dim=1,
        )

        ## CRITERIA 3: all hand 2D KPS (including wrist) inside of box.
        hand_kps2d_thresh = 0.5
        hand_kps2d_valid_mask = torch.stack(
            [
                lhand_output["mhr_hand"]["pred_keypoints_2d_cropped"]
                .abs()
                .amax(dim=(1, 2))
                < hand_kps2d_thresh,
                rhand_output["mhr_hand"]["pred_keypoints_2d_cropped"]
                .abs()
                .amax(dim=(1, 2))
                < hand_kps2d_thresh,
            ],
            dim=1,
        )

        ## CRITERIA 4: 2D wrist distance.
        hand_wrist_kps2d_thresh = 0.25
        kps_right_wrist_idx = 41
        kps_left_wrist_idx = 62
        right_kps_full = rhand_output["mhr_hand"]["pred_keypoints_2d"][
            :, [kps_right_wrist_idx]
        ].clone()
        left_kps_full = lhand_output["mhr_hand"]["pred_keypoints_2d"][
            :, [kps_right_wrist_idx]
        ].clone()
        left_kps_full[:, :, 0] = width - left_kps_full[:, :, 0] - 1  # Flip left hand
        body_right_kps_full = pose_output["mhr"]["pred_keypoints_2d"][
            :, [kps_right_wrist_idx]
        ].clone()
        body_left_kps_full = pose_output["mhr"]["pred_keypoints_2d"][
            :, [kps_left_wrist_idx]
        ].clone()
        right_kps_dist = (right_kps_full - body_right_kps_full).flatten(0, 1).norm(
            dim=-1
        ) / batch_lhand["bbox_scale"].flatten(0, 1)[:, 0]
        left_kps_dist = (left_kps_full - body_left_kps_full).flatten(0, 1).norm(
            dim=-1
        ) / batch_rhand["bbox_scale"].flatten(0, 1)[:, 0]
        hand_wrist_kps2d_valid_mask = torch.stack(
            [
                left_kps_dist < hand_wrist_kps2d_thresh,
                right_kps_dist < hand_wrist_kps2d_thresh,
            ],
            dim=1,
        )
        ## Left-right
        hand_valid_mask = (
            angle_difference_valid_mask
            & hand_box_size_valid_mask
            & hand_kps2d_valid_mask
            & hand_wrist_kps2d_valid_mask
        )

        # Keypoint prompting with the body decoder.
        # We use the wrist location from the hand decoder and the elbow location
        # from the body decoder as prompts to get an updated body pose estimation.
        batch_size, num_person = batch["img"].shape[:2]
        self.hand_batch_idx = []
        self.body_batch_idx = list(range(batch_size * num_person))

        ## Get right & left wrist keypoints from crops; full image. Each are B x 1 x 2
        kps_right_wrist_idx = 41
        kps_left_wrist_idx = 62
        right_kps_full = rhand_output["mhr_hand"]["pred_keypoints_2d"][
            :, [kps_right_wrist_idx]
        ].clone()
        left_kps_full = lhand_output["mhr_hand"]["pred_keypoints_2d"][
            :, [kps_right_wrist_idx]
        ].clone()
        left_kps_full[:, :, 0] = width - left_kps_full[:, :, 0] - 1  # Flip left hand

        # Next, get them to crop-normalized space.
        right_kps_crop = self._full_to_crop(batch, right_kps_full)
        left_kps_crop = self._full_to_crop(batch, left_kps_full)

        # Get right & left elbow keypoints from crops; full image. Each are B x 1 x 2
        kps_right_elbow_idx = 8
        kps_left_elbow_idx = 7
        right_kps_elbow_full = pose_output["mhr"]["pred_keypoints_2d"][
            :, [kps_right_elbow_idx]
        ].clone()
        left_kps_elbow_full = pose_output["mhr"]["pred_keypoints_2d"][
            :, [kps_left_elbow_idx]
        ].clone()

        # Next, get them to crop-normalized space.
        right_kps_elbow_crop = self._full_to_crop(batch, right_kps_elbow_full)
        left_kps_elbow_crop = self._full_to_crop(batch, left_kps_elbow_full)

        # Assemble them into keypoint prompts
        keypoint_prompt = torch.cat(
            [right_kps_crop, left_kps_crop, right_kps_elbow_crop, left_kps_elbow_crop],
            dim=1,
        )
        keypoint_prompt = torch.cat(
            [keypoint_prompt, keypoint_prompt[..., [-1]]], dim=-1
        )
        keypoint_prompt[:, 0, -1] = kps_right_wrist_idx
        keypoint_prompt[:, 1, -1] = kps_left_wrist_idx
        keypoint_prompt[:, 2, -1] = kps_right_elbow_idx
        keypoint_prompt[:, 3, -1] = kps_left_elbow_idx

        if keypoint_prompt.shape[0] > 1:
            # Replace invalid keypoints to dummy prompts
            invalid_prompt = (
                (keypoint_prompt[..., 0] < -0.5)
                | (keypoint_prompt[..., 0] > 0.5)
                | (keypoint_prompt[..., 1] < -0.5)
                | (keypoint_prompt[..., 1] > 0.5)
                | (~hand_valid_mask[..., [1, 0, 1, 0]])
            ).unsqueeze(-1)
            dummy_prompt = torch.zeros((1, 1, 3)).to(keypoint_prompt)
            dummy_prompt[:, :, -1] = -2
            keypoint_prompt[:, :, :2] = torch.clamp(
                keypoint_prompt[:, :, :2] + 0.5, min=0.0, max=1.0
            )  # [-0.5, 0.5] --> [0, 1]
            keypoint_prompt = torch.where(invalid_prompt, dummy_prompt, keypoint_prompt)
        else:
            # Only keep valid keypoints
            valid_keypoint = (
                torch.all(
                    (keypoint_prompt[:, :, :2] > -0.5)
                    & (keypoint_prompt[:, :, :2] < 0.5),
                    dim=2,
                )
                & hand_valid_mask[..., [1, 0, 1, 0]]
            ).squeeze()
            keypoint_prompt = keypoint_prompt[:, valid_keypoint]
            keypoint_prompt[:, :, :2] = torch.clamp(
                keypoint_prompt[:, :, :2] + 0.5, min=0.0, max=1.0
            )  # [-0.5, 0.5] --> [0, 1]

        if keypoint_prompt.numel() != 0:
            pose_output, _ = self.run_keypoint_prompt(
                batch, pose_output, keypoint_prompt
            )

        ##############################################################################

        # Drop in hand pose
        left_hand_pose_params = lhand_output["mhr_hand"]["hand"][:, :54]
        right_hand_pose_params = rhand_output["mhr_hand"]["hand"][:, 54:]
        updated_hand_pose = torch.cat(
            [left_hand_pose_params, right_hand_pose_params], dim=1
        )

        # Drop in hand scales
        updated_scale = pose_output["mhr"]["scale"].clone()
        updated_scale[:, 9] = lhand_output["mhr_hand"]["scale"][:, 9]
        updated_scale[:, 8] = rhand_output["mhr_hand"]["scale"][:, 8]
        updated_scale[:, 18:] = (
            lhand_output["mhr_hand"]["scale"][:, 18:]
            + rhand_output["mhr_hand"]["scale"][:, 18:]
        ) / 2

        # Update hand shape
        updated_shape = pose_output["mhr"]["shape"].clone()
        updated_shape[:, 40:] = (
            lhand_output["mhr_hand"]["shape"][:, 40:]
            + rhand_output["mhr_hand"]["shape"][:, 40:]
        ) / 2

        ############################ Doing IK ############################

        # First, forward just FK
        joint_rotations = self.head_pose.mhr_forward(
            global_trans=pose_output["mhr"]["global_rot"] * 0,
            global_rot=pose_output["mhr"]["global_rot"],
            body_pose_params=pose_output["mhr"]["body_pose"],
            hand_pose_params=updated_hand_pose,
            scale_params=updated_scale,
            shape_params=updated_shape,
            expr_params=pose_output["mhr"]["face"],
            return_joint_rotations=True,
        )[1]

        # Get lowarm
        lowarm_joint_idxs = torch.LongTensor([76, 40]).cuda()  # left, right
        lowarm_joint_rotations = joint_rotations[:, lowarm_joint_idxs]  # B x 2 x 3 x 3

        # Get zero-wrist pose
        wrist_twist_joint_idxs = torch.LongTensor([77, 41]).cuda()  # left, right
        wrist_zero_rot_pose = (
            lowarm_joint_rotations
            @ self.head_pose.joint_rotation[wrist_twist_joint_idxs]
        )

        # Get globals from left & right
        left_joint_global_rots = lhand_output["mhr_hand"]["joint_global_rots"]
        right_joint_global_rots = rhand_output["mhr_hand"]["joint_global_rots"]
        pred_global_wrist_rotmat = torch.stack(
            [
                left_joint_global_rots[:, 78],
                right_joint_global_rots[:, 42],
            ],
            dim=1,
        )

        # Now we want to get the local poses that lead to the wrist being pred_global_wrist_rotmat
        fused_local_wrist_rotmat = torch.einsum(
            "kabc,kabd->kadc", pred_global_wrist_rotmat, wrist_zero_rot_pose
        )
        wrist_xzy = fix_wrist_euler(
            roma.rotmat_to_euler("XZY", fused_local_wrist_rotmat)
        )

        # Put it in.
        angle_difference = rotation_angle_difference(
            ori_local_wrist_rotmat, fused_local_wrist_rotmat
        )  # B x 2 x 3 x3
        valid_angle = angle_difference < thresh_wrist_angle
        valid_angle = valid_angle & hand_valid_mask
        valid_angle = valid_angle.unsqueeze(-1)

        body_pose = pose_output["mhr"]["body_pose"][
            :, [41, 43, 42, 31, 33, 32]
        ].unflatten(1, (2, 3))
        updated_body_pose = torch.where(valid_angle, wrist_xzy, body_pose)
        pose_output["mhr"]["body_pose"][:, [41, 43, 42, 31, 33, 32]] = (
            updated_body_pose.flatten(1, 2)
        )

        hand_pose = pose_output["mhr"]["hand"].unflatten(1, (2, 54))
        pose_output["mhr"]["hand"] = torch.where(
            valid_angle, updated_hand_pose.unflatten(1, (2, 54)), hand_pose
        ).flatten(1, 2)

        hand_scale = torch.stack(
            [pose_output["mhr"]["scale"][:, 9], pose_output["mhr"]["scale"][:, 8]],
            dim=1,
        )
        updated_hand_scale = torch.stack(
            [updated_scale[:, 9], updated_scale[:, 8]], dim=1
        )
        masked_hand_scale = torch.where(
            valid_angle.squeeze(-1), updated_hand_scale, hand_scale
        )
        pose_output["mhr"]["scale"][:, 9] = masked_hand_scale[:, 0]
        pose_output["mhr"]["scale"][:, 8] = masked_hand_scale[:, 1]

        # Replace shared shape and scale
        pose_output["mhr"]["scale"][:, 18:] = torch.where(
            valid_angle.squeeze(-1).sum(dim=1, keepdim=True) > 0,
            (
                lhand_output["mhr_hand"]["scale"][:, 18:]
                * valid_angle.squeeze(-1)[:, [0]]
                + rhand_output["mhr_hand"]["scale"][:, 18:]
                * valid_angle.squeeze(-1)[:, [1]]
            )
            / (valid_angle.squeeze(-1).sum(dim=1, keepdim=True) + 1e-8),
            pose_output["mhr"]["scale"][:, 18:],
        )
        pose_output["mhr"]["shape"][:, 40:] = torch.where(
            valid_angle.squeeze(-1).sum(dim=1, keepdim=True) > 0,
            (
                lhand_output["mhr_hand"]["shape"][:, 40:]
                * valid_angle.squeeze(-1)[:, [0]]
                + rhand_output["mhr_hand"]["shape"][:, 40:]
                * valid_angle.squeeze(-1)[:, [1]]
            )
            / (valid_angle.squeeze(-1).sum(dim=1, keepdim=True) + 1e-8),
            pose_output["mhr"]["shape"][:, 40:],
        )

        ########################################################

        # Re-run forward
        with torch.no_grad():
            verts, j3d, jcoords, mhr_model_params, joint_global_rots = (
                self.head_pose.mhr_forward(
                    global_trans=pose_output["mhr"]["global_rot"] * 0,
                    global_rot=pose_output["mhr"]["global_rot"],
                    body_pose_params=pose_output["mhr"]["body_pose"],
                    hand_pose_params=pose_output["mhr"]["hand"],
                    scale_params=pose_output["mhr"]["scale"],
                    shape_params=pose_output["mhr"]["shape"],
                    expr_params=pose_output["mhr"]["face"],
                    return_keypoints=True,
                    return_joint_coords=True,
                    return_model_params=True,
                    return_joint_rotations=True,
                )
            )
            j3d = j3d[:, :70]  # 308 --> 70 keypoints
            verts[..., [1, 2]] *= -1  # Camera system difference
            j3d[..., [1, 2]] *= -1  # Camera system difference
            jcoords[..., [1, 2]] *= -1
            pose_output["mhr"]["pred_keypoints_3d"] = j3d
            pose_output["mhr"]["pred_vertices"] = verts
            pose_output["mhr"]["pred_joint_coords"] = jcoords
            pose_output["mhr"]["pred_pose_raw"][
                ...
            ] = 0  # pred_pose_raw is not valid anymore
            pose_output["mhr"]["mhr_model_params"] = mhr_model_params

        ########################################################
        # Project to 2D
        pred_keypoints_3d_proj = (
            pose_output["mhr"]["pred_keypoints_3d"]
            + pose_output["mhr"]["pred_cam_t"][:, None, :]
        )
        pred_keypoints_3d_proj[:, :, [0, 1]] *= pose_output["mhr"]["focal_length"][
            :, None, None
        ]
        pred_keypoints_3d_proj[:, :, [0, 1]] = (
            pred_keypoints_3d_proj[:, :, [0, 1]]
            + torch.FloatTensor([width / 2, height / 2]).to(pred_keypoints_3d_proj)[
                None, None, :
            ]
            * pred_keypoints_3d_proj[:, :, [2]]
        )
        pred_keypoints_3d_proj[:, :, :2] = (
            pred_keypoints_3d_proj[:, :, :2] / pred_keypoints_3d_proj[:, :, [2]]
        )
        pose_output["mhr"]["pred_keypoints_2d"] = pred_keypoints_3d_proj[:, :, :2]

        return pose_output, batch_lhand, batch_rhand, lhand_output, rhand_output

    def run_keypoint_prompt(self, batch, output, keypoint_prompt):
        image_embeddings = output["image_embeddings"]
        condition_info = output["condition_info"]
        pose_output = output["mhr"]  # body-only output
        # Use previous estimate as initialization
        prev_estimate = torch.cat(
            [
                pose_output["pred_pose_raw"].detach(),  # (B, 6)
                pose_output["shape"].detach(),
                pose_output["scale"].detach(),
                pose_output["hand"].detach(),
                pose_output["face"].detach(),
            ],
            dim=1,
        ).unsqueeze(dim=1)
        if hasattr(self, "init_camera"):
            prev_estimate = torch.cat(
                [prev_estimate, pose_output["pred_cam"].detach().unsqueeze(1)],
                dim=-1,
            )

        (tokens_output, pose_output, contact_output, force_output,
         motion_output) = self.forward_decoder(
            image_embeddings,
            init_estimate=None,  # not recurring previous estimate
            keypoints=keypoint_prompt,
            prev_estimate=prev_estimate,
            condition_info=condition_info,
            batch=batch,
        )
        pose_output = pose_output[-1]

        output.update({"mhr": pose_output, "contact": contact_output,
                       "force": force_output, "motion": motion_output})
        return output, keypoint_prompt

    def _get_hand_box(self, pose_output, batch):
        """Get hand bbox from the hand detector"""
        pred_left_hand_box = (
            pose_output["mhr"]["hand_box"][:, 0].detach().cpu().numpy()
            * self.cfg.MODEL.IMAGE_SIZE[0]
        )
        pred_right_hand_box = (
            pose_output["mhr"]["hand_box"][:, 1].detach().cpu().numpy()
            * self.cfg.MODEL.IMAGE_SIZE[0]
        )

        # Change boxes into squares
        batch["left_center"] = pred_left_hand_box[:, :2]
        batch["left_scale"] = (
            pred_left_hand_box[:, 2:].max(axis=1, keepdims=True).repeat(2, axis=1)
        )
        batch["right_center"] = pred_right_hand_box[:, :2]
        batch["right_scale"] = (
            pred_right_hand_box[:, 2:].max(axis=1, keepdims=True).repeat(2, axis=1)
        )

        # Crop to full. batch["affine_trans"] is full-to-crop, right application
        batch["left_scale"] = (
            batch["left_scale"]
            / batch["affine_trans"][0, :, 0, 0].cpu().numpy()[:, None]
        )
        batch["right_scale"] = (
            batch["right_scale"]
            / batch["affine_trans"][0, :, 0, 0].cpu().numpy()[:, None]
        )
        batch["left_center"] = (
            batch["left_center"]
            - batch["affine_trans"][0, :, [0, 1], [2, 2]].cpu().numpy()
        ) / batch["affine_trans"][0, :, 0, 0].cpu().numpy()[:, None]
        batch["right_center"] = (
            batch["right_center"]
            - batch["affine_trans"][0, :, [0, 1], [2, 2]].cpu().numpy()
        ) / batch["affine_trans"][0, :, 0, 0].cpu().numpy()[:, None]

        left_xyxy = np.concatenate(
            [
                (
                    batch["left_center"][:, 0] - batch["left_scale"][:, 0] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["left_center"][:, 1] - batch["left_scale"][:, 1] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["left_center"][:, 0] + batch["left_scale"][:, 0] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["left_center"][:, 1] + batch["left_scale"][:, 1] * 1 / 2
                ).reshape(-1, 1),
            ],
            axis=1,
        )
        right_xyxy = np.concatenate(
            [
                (
                    batch["right_center"][:, 0] - batch["right_scale"][:, 0] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["right_center"][:, 1] - batch["right_scale"][:, 1] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["right_center"][:, 0] + batch["right_scale"][:, 0] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["right_center"][:, 1] + batch["right_scale"][:, 1] * 1 / 2
                ).reshape(-1, 1),
            ],
            axis=1,
        )

        return left_xyxy, right_xyxy

    def keypoint_token_update_fn(
        self,
        kps_emb_start_idx,
        image_embeddings,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        # It's already after the last layer, we're done.
        if layer_idx == len(self.decoder.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        # Clone
        token_embeddings = token_embeddings.clone()
        token_augment = token_augment.clone()

        num_keypoints = self.keypoint_embedding.weight.shape[0]

        # Get current 2D KPS predictions
        pred_keypoints_2d_cropped = pose_output[
            "pred_keypoints_2d_cropped"
        ].clone()  # These are -0.5 ~ 0.5
        pred_keypoints_2d_depth = pose_output["pred_keypoints_2d_depth"].clone()

        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[
            :, self.keypoint_embedding_idxs
        ]
        pred_keypoints_2d_depth = pred_keypoints_2d_depth[
            :, self.keypoint_embedding_idxs
        ]

        # Get 2D KPS to be 0 ~ 1
        pred_keypoints_2d_cropped_01 = pred_keypoints_2d_cropped + 0.5

        # Get a mask of those that are 1) beyond image boundaries or 2) behind the camera
        invalid_mask = (
            (pred_keypoints_2d_cropped_01[:, :, 0] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 0] > 1)
            | (pred_keypoints_2d_cropped_01[:, :, 1] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 1] > 1)
            | (pred_keypoints_2d_depth[:, :] < 1e-5)
        )

        # Run them through the prompt encoder's pos emb function
        token_augment[:, kps_emb_start_idx : kps_emb_start_idx + num_keypoints, :] = (
            self.keypoint_posemb_linear(pred_keypoints_2d_cropped)
            * (~invalid_mask[:, :, None])
        )

        # Also maybe update token_embeddings with the grid sampled 2D feature.
        # Remember that pred_keypoints_2d_cropped are -0.5 ~ 0.5. We want -1 ~ 1
        # Sample points...
        ## Get sampling points
        pred_keypoints_2d_cropped_sample_points = pred_keypoints_2d_cropped * 2
        if self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr",
            "vit",
            "vit_b",
            "vit_l",
            "vit_hmr_512_384",
        ]:
            # Need to go from 256 x 256 coords to 256 x 192 (HW) because image_embeddings is 16x12
            # Aka, for x, what was normally -1 ~ 1 for 256 should be -16/12 ~ 16/12 (since to sample at original 256, need to overflow)
            pred_keypoints_2d_cropped_sample_points[:, :, 0] = (
                pred_keypoints_2d_cropped_sample_points[:, :, 0] / 12 * 16
            )

        # Version 2 is projecting & bilinear sampling
        pred_keypoints_2d_cropped_feats = (
            F.grid_sample(
                image_embeddings,
                pred_keypoints_2d_cropped_sample_points[:, :, None, :],  # -1 ~ 1, xy
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            .squeeze(3)
            .permute(0, 2, 1)
        )  # B x kps x C
        # Zero out invalid locations...
        pred_keypoints_2d_cropped_feats = pred_keypoints_2d_cropped_feats * (
            ~invalid_mask[:, :, None]
        )
        # This is ADDING
        token_embeddings = token_embeddings.clone()
        token_embeddings[
            :,
            kps_emb_start_idx : kps_emb_start_idx + num_keypoints,
            :,
        ] += self.keypoint_feat_linear(pred_keypoints_2d_cropped_feats)

        return token_embeddings, token_augment, pose_output, layer_idx

    def keypoint3d_token_update_fn(
        self,
        kps3d_emb_start_idx,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        # It's already after the last layer, we're done.
        if layer_idx == len(self.decoder.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        num_keypoints3d = self.keypoint3d_embedding.weight.shape[0]

        # Get current 3D kps predictions
        pred_keypoints_3d = pose_output["pred_keypoints_3d"].clone()

        # Now, pelvis normalize
        pred_keypoints_3d = (
            pred_keypoints_3d
            - (
                pred_keypoints_3d[:, [self.pelvis_idx[0]], :]
                + pred_keypoints_3d[:, [self.pelvis_idx[1]], :]
            )
            / 2
        )

        # Get the kps we care about, _after_ pelvis norm (just in case idxs shift)
        pred_keypoints_3d = pred_keypoints_3d[:, self.keypoint3d_embedding_idxs]

        # Run through embedding MLP & put in
        token_augment = token_augment.clone()
        token_augment[
            :,
            kps3d_emb_start_idx : kps3d_emb_start_idx + num_keypoints3d,
            :,
        ] = self.keypoint3d_posemb_linear(pred_keypoints_3d)

        return token_embeddings, token_augment, pose_output, layer_idx

    @staticmethod
    def _build_block_token_mask(batch_size, num_total, block_starts, device):
        """Asymmetric token-token attention mask for the appended token blocks.

        For each start ``s`` in ``block_starts`` (ascending), every token before
        ``s`` is barred from attending any token at ``>= s`` (True=allowed), while
        tokens at ``>= s`` still attend everything before them. Passing every
        appended block's start gives the block-causal regime (original tokens
        never attend contact/force/motion, contact never attends force/motion,
        force never attends motion); passing only the FIRST start gives the
        mutual regime (the appended blocks fully inter-attend, the original
        tokens still attend none of them). Returns a bool mask of shape
        ``(batch_size, num_total, num_total)``.
        """
        token_mask = torch.ones(
            batch_size, num_total, num_total, dtype=torch.bool, device=device,
        )
        for start in block_starts:
            token_mask[:, :start, start:] = False
        return token_mask

    def _anchored_token_update(
        self,
        start_idx,
        num_anchor,
        kp_indices,
        posemb_linear,
        feat_linear,
        image_embeddings,
        pose_output,
        token_embeddings,
        token_augment,
    ):
        """Anchored per-layer token update shared by contact and force tokens.

        For the ``num_anchor`` tokens starting at ``start_idx``:
        1. writes a 2D-keypoint positional encoding into ``token_augment`` and
        2. adds grid-sampled backbone features into ``token_embeddings``,

        both anchored at the caller's ``kp_indices`` MHR70 anchor list
        (``contact_keypoint_indices`` / ``force_keypoint_indices``) with the
        caller's own ``posemb_linear`` / ``feat_linear``. Anchors outside the
        frame or behind the camera contribute zero. Returns the updated
        ``(token_embeddings, token_augment)``.
        """
        # Get predicted 2D keypoint positions in crop space (-0.5 to 0.5)
        pred_kps_2d = pose_output["pred_keypoints_2d_cropped"].clone()  # [B, 70, 2]
        pred_kps_depth = pose_output["pred_keypoints_2d_depth"].clone()  # [B, 70]

        # Select the configured keypoint anchor for every anchored token.
        anchor_kps_2d = pred_kps_2d[:, kp_indices]       # [B, K_anchor, 2]
        anchor_kps_depth = pred_kps_depth[:, kp_indices]  # [B, K_anchor]

        # Validity check: outside image bounds or behind camera
        anchor_kps_01 = anchor_kps_2d + 0.5  # convert to 0-1 range
        invalid_mask = (
            (anchor_kps_01[:, :, 0] < 0)
            | (anchor_kps_01[:, :, 0] > 1)
            | (anchor_kps_01[:, :, 1] < 0)
            | (anchor_kps_01[:, :, 1] > 1)
            | (anchor_kps_depth < 1e-5)
        )  # [B, K_anchor]

        # --- 1. Update positional encoding ---
        token_augment = token_augment.clone()
        token_augment[
            :, start_idx : start_idx + num_anchor, :
        ] = (
            posemb_linear(anchor_kps_2d)
            * (~invalid_mask[:, :, None])
        )

        # --- 2. Sample image features at predicted body part locations ---
        # Convert from [-0.5, 0.5] to [-1, 1] for grid_sample
        sample_points = anchor_kps_2d * 2
        # Handle backbone-specific coordinate adjustments
        if self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr", "vit", "vit_b", "vit_l", "vit_hmr_512_384",
        ]:
            sample_points[:, :, 0] = sample_points[:, :, 0] / 12 * 16

        # Bilinear sampling with optional K×K neighbourhood grid
        gs = self.contact_grid_size
        if gs > 1:
            half = gs // 2
            offsets = torch.tensor(
                [
                    [dy * self.contact_grid_radius, dx * self.contact_grid_radius]
                    for dy in range(-half, half + 1)
                    for dx in range(-half, half + 1)
                ],
                dtype=sample_points.dtype,
                device=sample_points.device,
            )  # [K*K, 2]
            # [B, num_anchor, K*K, 2]
            pts = sample_points.unsqueeze(2) + offsets[None, None]
            B_s, nc, KK, _ = pts.shape
            pts_flat = pts.reshape(B_s, nc * KK, 1, 2)
            feats_flat = (
                F.grid_sample(
                    image_embeddings,
                    pts_flat,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                .squeeze(3)
                .permute(0, 2, 1)
            )  # [B, nc*KK, C_backbone]
            sampled_feats = feats_flat.reshape(B_s, nc, KK, -1).mean(dim=2)  # [B, nc, C_backbone]
        else:
            # Original single-point sampling
            sampled_feats = (
                F.grid_sample(
                    image_embeddings,
                    sample_points[:, :, None, :],  # [B, num_anchor, 1, 2]
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                .squeeze(3)
                .permute(0, 2, 1)
            )  # [B, num_anchor, C_backbone]

        # Zero out features for invalid locations
        sampled_feats = sampled_feats * (~invalid_mask[:, :, None])

        # Project from backbone dim to decoder dim and add to the tokens
        token_embeddings = token_embeddings.clone()
        token_embeddings[
            :, start_idx : start_idx + num_anchor, :
        ] += feat_linear(sampled_feats)

        return token_embeddings, token_augment

    def contact_token_update_fn(
        self,
        contact_emb_start_idx,
        image_embeddings,
        decoder_layers,
        batch,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        """
        Update contact tokens after each intermediate decoder layer.

        Mirrors the keypoint token update mechanism:
        1. Updates positional encoding (token_augment) with predicted 2D positions
           of the corresponding MHR70 keypoint.
        2. Samples image features at those 2D positions via grid_sample and adds
           them to the contact token embeddings.

        Every anchored contact token corresponds to the MHR70 keypoint at the
        same position in ``contact_keypoint_indices``. Global contact tokens are
        deliberately excluded from this local update.
        """
        # Skip after the last layer (same pattern as keypoint_token_update_fn)
        if layer_idx == len(decoder_layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        # --- contact blind hook ---
        # The anchored update is the tokens' only *direct* image path (grid-sampled
        # features) and their only keypoint-position path (posemb). The ablation
        # drops both. Attribute absent -> unablated (standalone use).
        if not getattr(self, "contact_blind_to_image", False):
            token_embeddings, token_augment = self._anchored_token_update(
                contact_emb_start_idx, self.num_contact_tokens,
                self.contact_keypoint_indices,
                self.contact_posemb_linear, self.contact_feat_linear,
                image_embeddings, pose_output, token_embeddings, token_augment,
            )
        # --- end contact blind hook ---

        return token_embeddings, token_augment, pose_output, layer_idx

    def force_token_update_fn(
        self,
        force_emb_start_idx,
        image_embeddings,
        decoder_layers,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        """Update force tokens after each intermediate decoder layer.

        Same anchored update as :meth:`contact_token_update_fn` (2D-keypoint posemb
        + grid-sampled features at ``force_keypoint_indices`` — the contact anchors
        when inherited, D2), applied to the force-token slice with the force
        linears. No temporal hook — all temporal mixing is post_decoder.
        """
        # Skip after the last layer (same pattern as contact_token_update_fn)
        if layer_idx == len(decoder_layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        token_embeddings, token_augment = self._anchored_token_update(
            force_emb_start_idx, self.num_force_tokens,
            self.force_keypoint_indices,
            self.force_posemb_linear, self.force_feat_linear,
            image_embeddings, pose_output, token_embeddings, token_augment,
        )
        return token_embeddings, token_augment, pose_output, layer_idx

    def motion_token_update_fn(
        self,
        motion_emb_start_idx,
        image_embeddings,
        decoder_layers,
        batch,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        """Update motion tokens after each intermediate decoder layer.

        Same anchored update as :meth:`force_token_update_fn` (2D-keypoint posemb
        + grid-sampled features at ``motion_keypoint_indices``), applied to the
        motion-token slice with the motion linears. Every motion token is
        anchored (no global tokens). No temporal hook — all temporal mixing is
        post_decoder. Never registered under ``MOTION_HEAD.ANCHORED=False``
        (the motion linears do not exist then).
        """
        # Skip after the last layer (same pattern as force_token_update_fn)
        if layer_idx == len(decoder_layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        token_embeddings, token_augment = self._anchored_token_update(
            motion_emb_start_idx, self.num_motion_tokens,
            self.motion_keypoint_indices,
            self.motion_posemb_linear, self.motion_feat_linear,
            image_embeddings, pose_output, token_embeddings, token_augment,
        )
        return token_embeddings, token_augment, pose_output, layer_idx

    def keypoint_token_update_fn_hand(
        self,
        kps_emb_start_idx,
        image_embeddings,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        # It's already after the last layer, we're done.
        if layer_idx == len(self.decoder_hand.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        # Clone
        token_embeddings = token_embeddings.clone()
        token_augment = token_augment.clone()

        num_keypoints = self.keypoint_embedding_hand.weight.shape[0]

        # Get current 2D KPS predictions
        pred_keypoints_2d_cropped = pose_output[
            "pred_keypoints_2d_cropped"
        ].clone()  # These are -0.5 ~ 0.5
        pred_keypoints_2d_depth = pose_output["pred_keypoints_2d_depth"].clone()

        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[
            :, self.keypoint_embedding_idxs_hand
        ]
        pred_keypoints_2d_depth = pred_keypoints_2d_depth[
            :, self.keypoint_embedding_idxs_hand
        ]

        # Get 2D KPS to be 0 ~ 1
        pred_keypoints_2d_cropped_01 = pred_keypoints_2d_cropped + 0.5

        # Get a mask of those that are 1) beyond image boundaries or 2) behind the camera
        invalid_mask = (
            (pred_keypoints_2d_cropped_01[:, :, 0] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 0] > 1)
            | (pred_keypoints_2d_cropped_01[:, :, 1] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 1] > 1)
            | (pred_keypoints_2d_depth[:, :] < 1e-5)
        )

        # Run them through the prompt encoder's pos emb function
        token_augment[:, kps_emb_start_idx : kps_emb_start_idx + num_keypoints, :] = (
            self.keypoint_posemb_linear_hand(pred_keypoints_2d_cropped)
            * (~invalid_mask[:, :, None])
        )

        # Also maybe update token_embeddings with the grid sampled 2D feature.
        # Remember that pred_keypoints_2d_cropped are -0.5 ~ 0.5. We want -1 ~ 1
        # Sample points...
        ## Get sampling points
        pred_keypoints_2d_cropped_sample_points = pred_keypoints_2d_cropped * 2
        if self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr",
            "vit",
            "vit_b",
            "vit_l",
            "vit_hmr_512_384",
        ]:
            # Need to go from 256 x 256 coords to 256 x 192 (HW) because image_embeddings is 16x12
            # Aka, for x, what was normally -1 ~ 1 for 256 should be -16/12 ~ 16/12 (since to sample at original 256, need to overflow)
            pred_keypoints_2d_cropped_sample_points[:, :, 0] = (
                pred_keypoints_2d_cropped_sample_points[:, :, 0] / 12 * 16
            )

        # Version 2 is projecting & bilinear sampling
        pred_keypoints_2d_cropped_feats = (
            F.grid_sample(
                image_embeddings,
                pred_keypoints_2d_cropped_sample_points[:, :, None, :],  # -1 ~ 1, xy
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            .squeeze(3)
            .permute(0, 2, 1)
        )  # B x kps x C
        # Zero out invalid locations...
        pred_keypoints_2d_cropped_feats = pred_keypoints_2d_cropped_feats * (
            ~invalid_mask[:, :, None]
        )
        # This is ADDING
        token_embeddings = token_embeddings.clone()
        token_embeddings[
            :,
            kps_emb_start_idx : kps_emb_start_idx + num_keypoints,
            :,
        ] += self.keypoint_feat_linear_hand(pred_keypoints_2d_cropped_feats)

        return token_embeddings, token_augment, pose_output, layer_idx

    def keypoint3d_token_update_fn_hand(
        self,
        kps3d_emb_start_idx,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        # It's already after the last layer, we're done.
        if layer_idx == len(self.decoder_hand.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        num_keypoints3d = self.keypoint3d_embedding_hand.weight.shape[0]

        # Get current 3D kps predictions
        pred_keypoints_3d = pose_output["pred_keypoints_3d"].clone()

        # Now, pelvis normalize
        pred_keypoints_3d = (
            pred_keypoints_3d
            - (
                pred_keypoints_3d[:, [self.pelvis_idx[0]], :]
                + pred_keypoints_3d[:, [self.pelvis_idx[1]], :]
            )
            / 2
        )

        # Get the kps we care about, _after_ pelvis norm (just in case idxs shift)
        pred_keypoints_3d = pred_keypoints_3d[:, self.keypoint3d_embedding_idxs_hand]

        # Run through embedding MLP & put in
        token_augment = token_augment.clone()
        token_augment[
            :,
            kps3d_emb_start_idx : kps3d_emb_start_idx + num_keypoints3d,
            :,
        ] = self.keypoint3d_posemb_linear_hand(pred_keypoints_3d)

        return token_embeddings, token_augment, pose_output, layer_idx
