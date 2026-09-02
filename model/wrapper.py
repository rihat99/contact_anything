"""Frozen SAM-3D-Body wrapper: batched crops (or cached embeddings) in, tokens out.

:class:`SAM3DBodyWrapper` is the single boundary between our trainable code and
the vendored fork in :mod:`model.sam_3d_body`. It owns the frozen base model
end to end — building it from the shipped checkpoint, freezing every
parameter, and pinning it to eval so stochastic depth / dropout in the frozen
backbone can never fire — and exposes exactly two operations:

``forward``
    One body decoder pass. Input is a flat batch of person crops
    (``[B, ...]``, no person dimension — the wrapper adds the fork's
    ``num_person = 1`` axis internally) plus the camera/bbox geometry the
    model conditions on, and optionally a list of externally-owned
    :class:`~model.sam_3d_body.models.meta_arch.sam3d_body.ExtraTokenBlock`
    appended to the decoder sequence behind an asymmetric attention mask.
    Output: the final token sequence, each block's bounds, and the frozen
    final MHR + camera readout.

``decode_pose``
    Recompute the final MHR + camera readout from a (temporally mixed or
    otherwise updated) pose token, optionally through externally-owned
    fine-tuned projection copies. Bit-identical to the in-forward readout
    when called with the unmodified pose token and no projection overrides.

Conditioning notes (the bug-prone part — identical for both input paths):

* **CLIFF condition** (``bbox_center``/``ori_img_size``/``bbox_scale``/
  ``cam_int``) is computed live from the geometry arguments and concatenated
  into the init pose token.
* **Ray map** is built from ``cam_int`` + ``affine_trans`` at crop resolution
  and applied to the *backbone output* inside the decoder forward
  (``ray_cond_emb``), never inside the backbone itself.
* **Mask conditioning** (``mask``/``mask_score``, SAM-3 person masks) is added
  to the backbone output before the decoder.

Because all three run downstream of the backbone, a cached raw backbone
embedding (``embedding=...``) goes through byte-for-byte the same conditioning
as a live image forward.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from model.sam_3d_body.models.meta_arch.sam3d_body import ExtraTokenBlock, SAM3DBody
from model.sam_3d_body.utils.checkpoint import load_state_dict
from model.sam_3d_body.utils.config import get_config


class SAM3DBodyWrapper(nn.Module):
    """Own and drive the frozen SAM-3D-Body base model (body branch only).

    :param checkpoint_path: the shipped ``model.ckpt`` (a ``model_config.yaml``
        must live next to it or one directory up, as in the HF snapshot).
    :param mhr_model_path: the shipped ``mhr_model.pt`` MHR body model.
    :param mask_embed_type: mask-conditioning variant to enable (the shipped
        checkpoints carry ``"v2"`` weights); ``None`` disables it.
    :param backbone_no_grad: run the backbone under ``no_grad`` (it is frozen;
        this only drops its activation graph).
    :param detach_interm_preds: run the decoder's per-layer interm MHR/camera
        readouts under ``no_grad`` (they only supply grid-sample locations for
        the token updates; every grad path through them ends in frozen params).
    """

    def __init__(
        self,
        checkpoint_path: str,
        mhr_model_path: str,
        mask_embed_type: Optional[str] = "v2",
        backbone_no_grad: bool = True,
        detach_interm_preds: bool = True,
    ):
        super().__init__()
        from yacs.config import CfgNode

        cfg_path = os.path.join(os.path.dirname(checkpoint_path), "model_config.yaml")
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(
                os.path.dirname(os.path.dirname(checkpoint_path)), "model_config.yaml"
            )
        model_cfg = get_config(cfg_path)
        model_cfg.defrost()
        model_cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = mhr_model_path
        model_cfg.MODEL.PROMPT_ENCODER.MASK_EMBED_TYPE = mask_embed_type
        model_cfg.MODEL.EFFICIENCY = CfgNode(
            {
                "BACKBONE_NO_GRAD": bool(backbone_no_grad),
                "DETACH_INTERM_PREDS": bool(detach_interm_preds),
            }
        )
        model_cfg.freeze()

        self.model = SAM3DBody(model_cfg)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        load_state_dict(self.model, state_dict, strict=False)

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    # The whole wrapper is frozen: train(True) from an enclosing module must
    # never re-enable the backbone's stochastic depth / dropout.
    def train(self, mode: bool = True) -> "SAM3DBodyWrapper":
        return super().train(False)

    @property
    def decoder_dim(self) -> int:
        return int(self.model.cfg.MODEL.DECODER.DIM)

    @property
    def backbone_dim(self) -> int:
        return int(self.model.backbone.embed_dims)

    @property
    def image_size(self) -> tuple:
        h, w = self.model.cfg.MODEL.IMAGE_SIZE
        return int(h), int(w)

    def _assemble_batch(
        self,
        img: Optional[torch.Tensor],
        embedding: Optional[torch.Tensor],
        bbox_center: torch.Tensor,
        bbox_scale: torch.Tensor,
        ori_img_size: torch.Tensor,
        img_size: torch.Tensor,
        affine_trans: torch.Tensor,
        cam_int: torch.Tensor,
        mask: Optional[torch.Tensor],
        mask_score: Optional[torch.Tensor],
    ) -> Dict:
        """Build the fork-facing batch dict (adds the ``num_person = 1`` axis)."""
        if (img is None) == (embedding is None):
            raise ValueError("pass exactly one of img= or embedding=")
        if img is None:
            # The fork reads batch["img"] only for shape/dtype (ray map grid,
            # embedding cast) on the precomputed path — a zero crop suffices.
            h, w = self.image_size
            img = torch.zeros(
                embedding.shape[0], 3, h, w,
                dtype=torch.float32, device=embedding.device,
            )
        batch_size = img.shape[0]
        device = img.device
        if mask is None:
            # mask_score <= 0 selects the learned no-mask embedding, so a zero
            # mask + zero score is the model's own "no mask given" input.
            mask = torch.zeros(batch_size, 1, *img.shape[-2:], device=device)
            mask_score = torch.zeros(batch_size, device=device)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)                    # [B, H, W] -> [B, 1, H, W]
        if mask.dim() != 4 or mask.shape[1] != 1:
            # The fork consumes [B, 1, H, W] after the person flatten; anything
            # else silently reaches its convs as an UNBATCHED 3-D tensor (the
            # LayerNorm2d then normalizes over height, destroying the
            # embeddings) or hard-errors at B > 1.
            raise ValueError(
                f"mask must be [B, H, W] or [B, 1, H, W]; got {tuple(mask.shape)}")
        if mask_score is None:
            raise ValueError("mask= requires mask_score= (SAM-3 mask confidence)")

        batch = {
            "img": img.unsqueeze(1),                    # [B, 1, 3, H, W]
            "bbox_center": bbox_center.unsqueeze(1),    # [B, 1, 2]
            "bbox_scale": bbox_scale.unsqueeze(1),      # [B, 1, 2]
            "ori_img_size": ori_img_size.unsqueeze(1),  # [B, 1, 2]
            "img_size": img_size.unsqueeze(1),          # [B, 1, 2]
            "affine_trans": affine_trans.unsqueeze(1),  # [B, 1, 2, 3]
            "cam_int": cam_int,                         # [B, 3, 3] (per-sample)
            "mask": mask.unsqueeze(1),                  # [B, 1, 1, H, W]
            "mask_score": mask_score.unsqueeze(1),      # [B, 1]
            "person_valid": torch.ones(batch_size, 1, device=device),
        }
        return batch

    def forward(
        self,
        *,
        img: Optional[torch.Tensor] = None,
        embedding: Optional[torch.Tensor] = None,
        bbox_center: torch.Tensor,
        bbox_scale: torch.Tensor,
        ori_img_size: torch.Tensor,
        img_size: torch.Tensor,
        affine_trans: torch.Tensor,
        cam_int: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_score: Optional[torch.Tensor] = None,
        blocks: Sequence[ExtraTokenBlock] = (),
        attention: str = "mutual",
    ) -> Dict:
        """Run one frozen body decoder pass.

        :param img: ``[B, 3, H, W]`` person crops (uint8-range or [0, 1] floats;
            the model normalizes either). Mutually exclusive with ``embedding``.
        :param embedding: ``[B, C, h, w]`` RAW backbone output (the
            ``precompute_embeddings`` cache); skips the backbone entirely.
        :param bbox_center: ``[B, 2]`` crop bbox center, full-image px.
        :param bbox_scale: ``[B, 2]`` crop bbox size, full-image px.
        :param ori_img_size: ``[B, 2]`` full image (W, H) px.
        :param img_size: ``[B, 2]`` crop (W, H) px.
        :param affine_trans: ``[B, 2, 3]`` full-image -> crop affine.
        :param cam_int: ``[B, 3, 3]`` pinhole intrinsics of the full image.
        :param mask: optional ``[B, H, W]`` or ``[B, 1, H, W]`` person masks
            (SAM-3); ``None`` = the model's learned no-mask embedding.
        :param mask_score: ``[B]`` mask confidence (``<= 0`` = ignore mask).
        :param blocks: externally-owned token blocks appended to the decoder
            sequence (order = sequence order); may be empty.
        :param attention: ``"mutual" | "causal"`` mask regime among the blocks.
        :returns: dict with

            - ``"tokens"``: ``[B, N_seq, C]`` final decoder token sequence
            - ``"blocks"``: ``{name: (start, end)}`` bounds of each block
            - ``"pose_token"``: ``[B, C]`` final body pose token (sequence 0)
            - ``"mhr"``: frozen final MHR + camera readout dict
            - ``"image_embeddings"``: mask-conditioned backbone features
            - ``"condition_info"``: the CLIFF condition vector
            - ``"ctx"``: opaque context to hand back to :meth:`decode_pose`
        """
        batch = self._assemble_batch(
            img, embedding, bbox_center, bbox_scale, ori_img_size, img_size,
            affine_trans, cam_int, mask, mask_score,
        )
        self.model._initialize_batch(batch)
        out = self.model.forward_step(
            batch,
            decoder_type="body",
            precomputed_features=embedding,
            extra_blocks=list(blocks),
            extra_token_attention=attention,
        )
        return {
            "tokens": out["tokens"],
            "blocks": out["blocks"],
            "pose_token": out["tokens"][:, 0],
            "mhr": out["mhr"],
            "image_embeddings": out["image_embeddings"],
            "condition_info": out["condition_info"],
            "ctx": batch,
        }

    def decode_pose(
        self,
        pose_token: torch.Tensor,
        ctx: Dict,
        proj_pose: Optional[nn.Module] = None,
        proj_camera: Optional[nn.Module] = None,
    ) -> Dict:
        """Final MHR + camera readout from an (updated) pose token.

        :param pose_token: ``[B, C]`` pose token (e.g. after temporal mixing).
        :param ctx: the ``"ctx"`` entry of a prior :meth:`forward` output for
            the same rows (supplies the camera/bbox geometry).
        :param proj_pose: optional trainable copy of ``head_pose.proj``
            applied instead of the frozen original (split-head fine-tune).
        :param proj_camera: optional trainable copy of ``head_camera.proj``.
        """
        self.model._initialize_batch(ctx)
        self.model.hand_batch_idx = []
        self.model.body_batch_idx = list(range(pose_token.shape[0]))
        return self.model.readout_pose(
            pose_token, ctx, proj_pose=proj_pose, proj_camera=proj_camera
        )
