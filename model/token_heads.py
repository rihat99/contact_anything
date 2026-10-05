"""Contact, force and gravity heads on the pose token itself (no temporal refiner).

The readout of the ladder's first rungs: every output is one zero-initialised
FFN (``dim -> dim -> k``, the refiner's head shape) on the (mixed) pose token of
a frame, and the output dicts follow the refiner's layout exactly, so the same
losses, metrics and dump scripts read them:

* ``contact`` — one logit per contact slot, ``{"logits", "probs", "logits_layers"}``.
* ``force`` — one 3D force per slot in the PER-FRAME body's root frame (body-weight
  units); ``frame`` is world-from-root of that body, which lets the force loss
  rotate the kindyn GT into it.
* ``gravity`` — per FRAME (there is no world model to pool over): the camera's
  down axis plus a body-frame correction, normalised; expressed in the world
  with the frame's extrinsics purely as transport, like the refiner's votes.
  ``prior_world`` is the camera axis alone, ``given`` all false.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from model.refiner import TemporalRefiner

OUTPUTS = ("contact", "force", "gravity")


class PoseTokenHeads(nn.Module):
    """Zero-init FFN heads on the pose token, one per listed output.

    :param dim: pose-token width.
    :param outputs: subset of :data:`OUTPUTS`.
    :param num_slots: K — the contact set's slot count.
    """

    def __init__(self, dim: int, outputs: Sequence[str], num_slots: int):
        super().__init__()
        self.num_slots = int(num_slots)
        outputs = [str(o) for o in outputs]
        if not outputs or any(o not in OUTPUTS for o in outputs) or len(set(outputs)) != len(outputs):
            raise ValueError(f"outputs must be a non-empty subset of {OUTPUTS}; got {outputs}")
        self.outputs = tuple(o for o in OUTPUTS if o in outputs)
        sizes = {"contact": self.num_slots, "force": 3 * self.num_slots, "gravity": 3}
        self.heads = nn.ModuleDict(
            {name: TemporalRefiner._zero_head(dim, sizes[name]) for name in self.outputs})

    def forward(self, pose_token: Tensor, root_rot_world: Tensor, cam_from_world: Tensor) -> dict:
        """Read the heads.

        :param pose_token: ``[B, C]`` final pose token.
        :param root_rot_world: ``[B, 3, 3]`` world-from-root of the per-frame body.
        :param cam_from_world: ``[B, 4, 4]`` extrinsics of the frame.
        :returns: ``{"contact", "force", "gravity"}`` in the refiner's layout, ``None``
            for outputs not built.
        """
        token = pose_token.float()
        n_frames = token.shape[0]
        out: dict = {name: None for name in OUTPUTS}
        if "contact" in self.outputs:
            logits = self.heads["contact"](token)
            out["contact"] = {"logits": logits, "probs": torch.sigmoid(logits),
                              "logits_layers": [logits]}
        if "force" in self.outputs:
            forces = self.heads["force"](token).reshape(n_frames, self.num_slots, 3)
            out["force"] = {"forces": forces, "frame": root_rot_world, "forces_layers": [forces]}
        if "gravity" in self.outputs:
            down_cam_w = cam_from_world[:, :3, :3].transpose(1, 2)[:, :, 1]   # camera +y in the world
            valid = torch.ones(n_frames, 1, dtype=torch.bool, device=token.device)
            world = TemporalRefiner.pool_gravity(
                self.heads["gravity"](token), root_rot_world, down_cam_w, n_frames, 1, valid,
                per_frame=True)
            prior = down_cam_w / down_cam_w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            out["gravity"] = {
                "world": world, "body": (root_rot_world.transpose(1, 2) @ world[..., None])[..., 0],
                "prior_world": prior, "world_layers": [world],
                "given": torch.zeros(n_frames, dtype=torch.bool, device=token.device)}
        return out


__all__ = ["PoseTokenHeads", "OUTPUTS"]
