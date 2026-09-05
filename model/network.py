"""ContactAnything — the composed trainable model over the frozen SAM-3D-Body.

Assembly (everything trainable lives HERE, nothing trainable inside the
wrapper):

1. :class:`~model.wrapper.SAM3DBodyWrapper` runs the frozen body decoder with
   our learned token blocks (:class:`~model.tokens.LearnedTokenBlock`)
   appended behind the asymmetric mask — original tokens never attend them, so
   the frozen pose/MHR outputs have an exactly-zero Jacobian w.r.t. every
   parameter of this module.
2. :class:`~model.rope.CrossModalRopeModule` (optional) — ONE post-decoder
   RoPE temporal transformer over the concatenation of the chosen modality
   token blocks, across the clip's frames (within-frame mixing is its
   ``dt = 0`` diagonal). A single listed modality (``[pose]``) is legal: the
   block is then a plain temporal transformer over that token block.
3. Heads (:mod:`model.heads`) read each block's final tokens: one contact
   logit per contact token, one 3D force per force token, and — from the
   (possibly mixed) pose token — the :class:`~model.heads.SmplxHead` SMPL-X
   body, which IS the pose output. ``out["mhr"]`` stays the frozen model's own
   readout (the contact anchors sample around its intermediate keypoints).
4. :class:`~model.refiner.TemporalRefiner` (optional, stage 2) — the
   world-space temporal refiner behind the per-frame body: it rewrites
   ``out["smplx"]`` and owns whichever of ``contact`` / ``force`` / ``motion``
   its ``outputs`` list (the decoder-level contact head is then not built).

The sub-configs are plain dicts mirroring the experiment yaml sections; the
config layer maps yaml -> these kwargs 1:1.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from model.heads import ContactHead, ForceHead, SmplxHead
from model.refiner import TemporalRefiner
from model.rope import CrossModalRopeModule
from model.tokens import LearnedTokenBlock
from model.wrapper import SAM3DBodyWrapper

_MODALITY_ORDER = ("pose", "contact", "force")


class ContactAnything(nn.Module):
    """Frozen SAM-3D-Body + trainable contact / force / SMPL-X / refiner branches.

    :param wrapper: the frozen base (registered as a submodule; all its params
        stay ``requires_grad=False`` and eval-pinned).
    :param contact: contact branch config or ``None`` — ``{keypoint_indices,
        mlp_depth, mlp_channel_div_factor, dropout, grid_size, grid_radius}``.
    :param force: force branch config or ``None`` — the same keys.
    :param cross_modal: cross-modal temporal config or ``None`` —
        ``{modalities, num_layers, num_heads, mlp_ratio, dropout, window,
        time_scale}``.
    :param smplx: SMPL-X head config or ``None`` — ``{model_path, hands,
        mlp_depth, mlp_channel_div_factor, dropout, camera}`` (extra keys such
        as ``checkpoint`` / ``frozen`` are the builder's business).
    :param refiner: :class:`~model.refiner.TemporalRefiner` config or ``None``
        — ``{outputs, dim, num_layers, num_heads, mlp_ratio, dropout, window,
        time_scale, depth_smooth_sec, pose_token, pose_token_dim,
        contact_token_dim}``; needs ``smplx``.
    """

    def __init__(
        self,
        wrapper: SAM3DBodyWrapper,
        contact: Optional[dict] = None,
        force: Optional[dict] = None,
        cross_modal: Optional[dict] = None,
        smplx: Optional[dict] = None,
        refiner: Optional[dict] = None,
    ):
        super().__init__()
        self.wrapper = wrapper
        dim = wrapper.decoder_dim
        backbone_dim = wrapper.backbone_dim
        refiner_outputs = set(str(o) for o in refiner["outputs"]) if refiner is not None else set()

        self.contact_tokens, self.head_contact = self._token_branch(
            "contact", contact, dim, backbone_dim, ContactHead)
        self.force_tokens, self.head_force = self._token_branch(
            "force", force, dim, backbone_dim, ForceHead)
        if "contact" in refiner_outputs:
            self.head_contact = None                    # the refiner predicts contact
        assert not ("force" in refiner_outputs and self.force_tokens is not None), (
            "refiner 'force' output and decoder force tokens are two force heads")

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
            )

        self.cross_modal_temporal = None
        self.cross_modal_modalities: list[str] = []
        if cross_modal is not None:
            requested = [str(m) for m in cross_modal["modalities"]]
            assert requested, "cross_modal needs at least one modality"
            missing = [m for m in requested if self._token_count(m) == 0]
            assert not missing, (
                f"cross_modal modalities {missing} have no token block in this "
                "build (enable the corresponding branch)")
            self.cross_modal_modalities = [
                m for m in _MODALITY_ORDER if m in requested]
            self.cross_modal_temporal = CrossModalRopeModule(
                dim=dim,
                num_slots=sum(self._token_count(m) for m in self.cross_modal_modalities),
                num_layers=int(cross_modal["num_layers"]),
                num_heads=int(cross_modal["num_heads"]),
                mlp_ratio=float(cross_modal["mlp_ratio"]),
                dropout=float(cross_modal["dropout"]),
                window=cross_modal["window"],
                time_scale=float(cross_modal["time_scale"]),
            )

        self.refiner = None
        if refiner is not None:
            assert self.head_smplx is not None, "the refiner needs the per-frame SMPL-X head"
            self.refiner = TemporalRefiner(
                decoder_dim=dim,
                outputs=refiner["outputs"],
                num_contact_tokens=self._token_count("contact"),
                dim=int(refiner["dim"]),
                num_layers=int(refiner["num_layers"]),
                num_heads=int(refiner["num_heads"]),
                mlp_ratio=float(refiner["mlp_ratio"]),
                dropout=float(refiner["dropout"]),
                window=float(refiner["window"]),
                time_scale=float(refiner["time_scale"]),
                depth_smooth_sec=float(refiner["depth_smooth_sec"]),
                pose_token=bool(refiner["pose_token"]),
                pose_token_dim=int(refiner["pose_token_dim"]),
                contact_token_dim=int(refiner["contact_token_dim"]),
            )

    @staticmethod
    def _token_branch(name: str, cfg: Optional[dict], dim: int, backbone_dim: int,
                      head_cls):
        if cfg is None:
            return None, None
        tokens = LearnedTokenBlock(
            name, dim, backbone_dim,
            keypoint_indices=cfg["keypoint_indices"],
            grid_size=int(cfg["grid_size"]),
            grid_radius=float(cfg["grid_radius"]),
        )
        head = head_cls(
            input_dim=dim,
            mlp_depth=int(cfg["mlp_depth"]),
            mlp_channel_div_factor=int(cfg["mlp_channel_div_factor"]),
            dropout=float(cfg["dropout"]),
        )
        return tokens, head

    def _token_count(self, modality: str) -> int:
        block = {"pose": None, "contact": self.contact_tokens,
                 "force": self.force_tokens}[modality]
        if modality == "pose":
            return 1
        return block.num_tokens if block is not None else 0

    def _refiner_has(self, output: str) -> bool:
        return self.refiner is not None and output in self.refiner.outputs

    @property
    def has_contact(self) -> bool:
        """Whether the forward emits ``out["contact"]`` (decoder head or refiner)."""
        return self.head_contact is not None or self._refiner_has("contact")

    @property
    def has_force(self) -> bool:
        """Whether the forward emits ``out["force"]`` (decoder head or refiner)."""
        return self.head_force is not None or self._refiner_has("force")

    @property
    def has_motion(self) -> bool:
        """Whether the forward emits ``out["motion"]`` (refiner only)."""
        return self._refiner_has("motion")

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, batch: Dict) -> Dict:
        """Forward one collated batch (flattened clips, ``[B_clips * T, ...]``).

        Consumes the collate layout: flat per-frame geometry ``[B, ...]`` (no
        person dimension), optional ``embedding`` (cached backbone output), and
        the ``seq_len`` / ``frame_pos_sec`` / ``frame_valid`` clip fields.

        :returns: ``{"mhr", "contact", "force", "motion", "smplx", "tokens",
            "blocks"}`` — head outputs are ``None`` for disabled branches;
            ``contact`` is ``{"logits", "probs"} [B, K]``, ``force`` is
            ``{"forces"} [B, K, 3]``, ``motion`` the refiner's body-frame
            velocities / accelerations.
        """
        embedding = batch.get("embedding")          # optional: the cached backbone path
        img = batch["img"] if embedding is None else None
        batch_size = (img if embedding is None else embedding).shape[0]

        learned = [b for b in (self.contact_tokens, self.force_tokens) if b is not None]
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
        )
        tokens = out["tokens"]
        bounds = dict(out["blocks"])
        bounds["pose"] = (0, 1)

        if self.cross_modal_temporal is not None:
            slices = [bounds[m] for m in self.cross_modal_modalities]
            mixed = torch.cat([tokens[:, lo:hi] for lo, hi in slices], dim=1)
            updated = self.cross_modal_temporal(
                mixed, int(batch["seq_len"]), batch["frame_pos_sec"], batch["frame_valid"])
            # Scatter the updated slices back (ordered + disjoint by the
            # canonical modality order; everything between them is untouched).
            pieces, off, prev = [], 0, 0
            for lo, hi in slices:
                k = hi - lo
                pieces += [tokens[:, prev:lo], updated[:, off:off + k]]
                off, prev = off + k, hi
            pieces.append(tokens[:, prev:])
            tokens = torch.cat(pieces, dim=1)

        contact_output = None
        if self.head_contact is not None:
            lo, hi = bounds["contact"]
            logits = self.head_contact(tokens[:, lo:hi])
            contact_output = {"logits": logits, "probs": torch.sigmoid(logits)}

        force_output = None
        if self.head_force is not None:
            lo, hi = bounds["force"]
            force_output = {"forces": self.head_force(tokens[:, lo:hi])}      # [B, K, 3]

        smplx_output = None
        if self.head_smplx is not None:
            smplx_output = self.head_smplx(
                tokens[:, 0],
                bbox_center=batch["bbox_center"],
                bbox_size=batch["bbox_scale"][:, 0],
                cam_int=batch["cam_int"],
                affine_trans=batch["affine_trans"],
                img_size=batch["img_size"],
            )

        motion_output = None
        if self.refiner is not None:
            refined = self.refiner(smplx_output, tokens, bounds, batch,
                                   body=self.head_smplx.body(tokens.device))
            smplx_output = refined["smplx"]
            if refined["contact"] is not None:
                contact_output = refined["contact"]
            if refined["force"] is not None:
                force_output = refined["force"]
            motion_output = refined["motion"]

        return {
            "mhr": out["mhr"],
            "contact": contact_output,
            "force": force_output,
            "motion": motion_output,
            "smplx": smplx_output,
            "tokens": tokens,
            "blocks": bounds,
        }
