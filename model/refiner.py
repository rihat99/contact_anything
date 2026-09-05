"""World-space temporal refiner — stage 2 behind the per-frame model (docs/refiner.md).

The per-frame model (frozen SAM3D decoder + the frozen stage-1 SMPL-X / CLIFF
heads) gives a camera-frame body per frame. This module turns the clip into a
WORLD-space motion, encodes it frame by frame in a way that is independent of
the world frame, runs a local temporal transformer over the frames, and decodes
corrections — again without any reference to the world frame:

1. **Depth smoothing** (``depth_smooth_sec``): Gaussian smoothing of the pelvis
   LOG depth along time in camera coordinates, bearing kept, so the body stays
   on its image ray. Depth is the per-frame model's dominant noise.
2. **Lift** with the frame extrinsics: ``p_w = R^T (p_c - t)``,
   ``R_world_root = R^T R_cam_root``; betas averaged over the clip.
3. **Per-frame token** = 21 body-joint positions in the ROOT frame, the root's
   linear and angular velocity in the BODY frame (finite differences of the
   lifted trajectory), the frame spacing, the mean betas, the projected frozen
   pose token and the projected six contact tokens. No absolute position, no
   heading: a rigid re-definition of the world leaves every input unchanged.
4. **Temporal transformer**: :class:`~model.rope.CrossModalRopeModule` with a
   single slot — RoPE positions are seconds, a hard ``window`` per layer bounds
   the receptive field to ``num_layers x window``.
5. **Zero-initialised heads**: contact logits; pose offset = 6D rotation deltas
   right-multiplied onto the root (body frame) and the 21 body joints
   (parent-local) plus a root shift in the body frame; motion = world velocity /
   acceleration of the 22 joints and the root's angular velocity / acceleration,
   all expressed in the (input) body frame; forces in that same body frame.
6. **Decode**: FK in the world with the mean betas, then back into each camera
   with the extrinsics — the output dict has the :class:`~model.heads.SmplxHead`
   keys, so the existing SMPL-X loss and pose metrics apply unchanged.

At initialisation the refiner is exactly "per-frame model + depth smoothing":
the RoPE blocks are identities and every head is zero.
"""
from __future__ import annotations

from typing import Optional, Sequence

import roma
import torch
import torch.nn as nn
from torch import Tensor

from model.rope import CrossModalRopeModule
from utils.geometry import (project_to_crop, ray_to_translation, rot6d_to_rotmat,
                            rotmat_to_rot6d, smplx_q, translation_to_ray)

OUTPUTS = ("pose", "contact", "motion", "force")
NUM_BODY_JOINTS = 22
NUM_GROUPS = 6
_IDENTITY_6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
#: The frame-spacing input is expressed in 25-fps frames (1.0 at the corpus's reference rate).
_DT_SCALE = 25.0
#: Geometry features per frame: 21 root-frame joint positions (the pelvis IS the root, so
#: its row is identically zero and omitted), root linear + angular velocity, dt, 10 betas.
_GEOMETRY_DIM = 3 * (NUM_BODY_JOINTS - 1) + 3 + 3 + 1 + 10


# ------------------------------------------------------------------ time series helpers

def _trailing(mask: Tensor, like: Tensor) -> Tensor:
    """Reshape a ``[n, T]`` mask so it broadcasts over ``like``'s trailing dims."""
    return mask.reshape(*mask.shape, *([1] * (like.dim() - 2)))


def gaussian_smooth(x: Tensor, seconds: Tensor, valid: Tensor, sigma: float) -> Tensor:
    """Masked Gaussian smoothing along time.

    :param x: ``[n, T, ...]`` series.
    :param seconds: ``[n, T]`` frame times.
    :param valid: ``[n, T]`` bool; invalid frames never contribute to others.
    :param sigma: kernel width in seconds (``<= 0`` returns ``x``).
    """
    if sigma <= 0.0:
        return x
    n, t = seconds.shape
    dt = seconds[:, :, None] - seconds[:, None, :]
    weights = torch.exp(-0.5 * (dt / sigma) ** 2) * valid[:, None, :].to(x.dtype)
    eye = torch.eye(t, dtype=x.dtype, device=x.device)[None]
    weights = torch.maximum(weights, eye)                    # a frame always sees itself
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return torch.bmm(weights, x.reshape(n, t, -1)).reshape(x.shape)


def _shifted(x: Tensor, valid: Tensor, seconds: Tensor):
    prev_ok = torch.zeros_like(valid)
    prev_ok[:, 1:] = valid[:, :-1]
    next_ok = torch.zeros_like(valid)
    next_ok[:, :-1] = valid[:, 1:]
    x_prev = torch.cat([x[:, :1], x[:, :-1]], dim=1)
    x_next = torch.cat([x[:, 1:], x[:, -1:]], dim=1)
    t_prev = torch.cat([seconds[:, :1], seconds[:, :-1]], dim=1)
    t_next = torch.cat([seconds[:, 1:], seconds[:, -1:]], dim=1)
    lo = torch.where(_trailing(prev_ok, x), x_prev, x)
    hi = torch.where(_trailing(next_ok, x), x_next, x)
    t_lo = torch.where(prev_ok, t_prev, seconds)
    t_hi = torch.where(next_ok, t_next, seconds)
    return lo, hi, t_lo, t_hi, prev_ok, next_ok


def time_derivative(x: Tensor, seconds: Tensor, valid: Tensor) -> Tensor:
    """Finite-difference time derivative of ``x`` ``[n, T, ...]``.

    Central where both neighbours are valid, one-sided at the ends of a valid
    run, zero where a frame has no valid neighbour.
    """
    lo, hi, t_lo, t_hi, _, _ = _shifted(x, valid, seconds)
    dt = t_hi - t_lo
    deriv = (hi - lo) / _trailing(dt.clamp(min=1e-6), x)
    return torch.where(_trailing(dt > 0, x), deriv, torch.zeros_like(deriv))


def local_dt(seconds: Tensor, valid: Tensor) -> Tensor:
    """Per-frame spacing ``[n, T]``: mean step to the valid neighbours (0 if none)."""
    _, _, t_lo, t_hi, prev_ok, next_ok = _shifted(seconds, valid, seconds)
    count = (prev_ok.to(seconds.dtype) + next_ok.to(seconds.dtype))
    return (t_hi - t_lo) / count.clamp(min=1.0)


def angular_velocity(rot: Tensor, seconds: Tensor, valid: Tensor) -> Tensor:
    """Body-frame angular velocity of world-from-body rotations.

    The increment ``log(R_t^T R_{t+1})`` is the same vector in the frames of
    ``t`` and ``t + 1`` (a rotation fixes its own axis), so its rate over the
    step is attributed to both frames; a frame's velocity is the mean of the
    backward and forward steps that join two valid frames, zero when none.

    :param rot: ``[n, T, 3, 3]``; ``seconds`` / ``valid`` ``[n, T]``.
    :returns: ``[n, T, 3]`` rad/s.
    """
    n, t = seconds.shape
    zeros = torch.zeros(n, 1, 3, dtype=rot.dtype, device=rot.device)
    if t < 2:
        return zeros.expand(n, t, 3)
    inc = roma.rotmat_to_rotvec(rot[:, :-1].transpose(-1, -2) @ rot[:, 1:])   # [n, T-1, 3]
    dt = seconds[:, 1:] - seconds[:, :-1]
    ok = valid[:, 1:] & valid[:, :-1] & (dt > 0)
    rate = torch.where(ok[..., None], inc / dt.clamp(min=1e-6)[..., None],
                       torch.zeros_like(inc))
    false = torch.zeros(n, 1, dtype=torch.bool, device=rot.device)
    fwd, fwd_ok = torch.cat([rate, zeros], dim=1), torch.cat([ok, false], dim=1)
    bwd, bwd_ok = torch.cat([zeros, rate], dim=1), torch.cat([false, ok], dim=1)
    count = fwd_ok.to(rot.dtype) + bwd_ok.to(rot.dtype)
    return (fwd + bwd) / count.clamp(min=1.0)[..., None]


def stencil_valid(valid: Tensor, radius: int) -> Tensor:
    """Frames whose ``+- radius`` neighbours (and themselves) are all valid."""
    out = valid.clone()
    for k in range(1, radius + 1):
        prev = torch.zeros_like(valid)
        prev[:, k:] = valid[:, :-k]
        nxt = torch.zeros_like(valid)
        nxt[:, :-k] = valid[:, k:]
        out = out & prev & nxt
    return out


# ------------------------------------------------------------------ the module

class TemporalRefiner(nn.Module):
    """Local temporal transformer over the world-lifted per-frame body.

    :param decoder_dim: width of the frozen decoder tokens.
    :param outputs: subset of :data:`OUTPUTS` to build heads for.
    :param num_contact_tokens: contact tokens fed as features (0 = none).
    :param dim: transformer width.
    :param num_layers: RoPE blocks.
    :param num_heads: attention heads.
    :param mlp_ratio: FFN expansion.
    :param dropout: dropout inside attention / FFN.
    :param window: attention half-width per layer, seconds.
    :param time_scale: RoPE rotation units per second.
    :param depth_smooth_sec: pelvis log-depth Gaussian sigma, seconds (0 = off).
    :param pose_token: feed the frozen pose token (projected).
    :param pose_token_dim: pose-token projection width.
    :param contact_token_dim: per-contact-token projection width.
    """

    def __init__(
        self,
        decoder_dim: int,
        outputs: Sequence[str],
        num_contact_tokens: int,
        dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        window: float = 0.5,
        time_scale: float = 25.0,
        depth_smooth_sec: float = 0.0,
        pose_token: bool = True,
        pose_token_dim: int = 256,
        contact_token_dim: int = 64,
    ):
        super().__init__()
        outputs = [str(o) for o in outputs]
        if not outputs or any(o not in OUTPUTS for o in outputs) or len(set(outputs)) != len(outputs):
            raise ValueError(f"outputs must be a non-empty subset of {OUTPUTS}; got {outputs}")
        self.outputs = tuple(o for o in OUTPUTS if o in outputs)
        self.num_contact_tokens = int(num_contact_tokens)
        self.depth_smooth_sec = float(depth_smooth_sec)
        self.time_scale = float(time_scale)

        self.proj_pose_token = nn.Linear(decoder_dim, pose_token_dim) if pose_token else None
        self.proj_contact_tokens = (nn.Linear(decoder_dim, contact_token_dim)
                                    if self.num_contact_tokens > 0 else None)
        token_dim = (pose_token_dim if pose_token else 0) + self.num_contact_tokens * contact_token_dim
        # Two LayerNorms: the 80 geometry numbers and the ~640 projected token channels are
        # normalised separately, so neither group's scale rides on the other's width.
        self.geometry_norm = nn.LayerNorm(_GEOMETRY_DIM)
        self.token_norm = nn.LayerNorm(token_dim) if token_dim > 0 else None
        self.input_proj = nn.Linear(_GEOMETRY_DIM + token_dim, dim)
        self.temporal = CrossModalRopeModule(
            dim=dim, num_slots=1, num_layers=num_layers, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout, window=window, time_scale=time_scale)
        self.output_norm = nn.LayerNorm(dim)
        sizes = {"pose": 6 * NUM_BODY_JOINTS + 3, "contact": NUM_GROUPS,
                 "motion": 6 * NUM_BODY_JOINTS + 6, "force": 3 * NUM_GROUPS}
        self.heads = nn.ModuleDict()
        for name in self.outputs:
            head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, sizes[name]))
            nn.init.zeros_(head[2].weight)
            nn.init.zeros_(head[2].bias)
            self.heads[name] = head
        self.register_buffer("identity_6d", torch.tensor(_IDENTITY_6D), persistent=False)

    # ------------------------------------------------------------------ forward

    def forward(self, smplx_out: dict, tokens: Tensor, blocks: dict, batch: dict, body) -> dict:
        """Refine one batch of flattened clips.

        :param smplx_out: the per-frame :class:`~model.heads.SmplxHead` output.
        :param tokens: final decoder tokens ``[B, N, C]`` (pose token at 0).
        :param blocks: token-block bounds (``blocks["contact"]`` when present).
        :param batch: collated batch (``seq_len``, ``frame_pos_sec``,
            ``frame_valid``, ``cam_from_world``, ``cam_int``, ``affine_trans``,
            ``img_size``).
        :param body: the head's BetterHuman SMPL-X body (22 or 52 joints).
        :returns: ``{"smplx", "contact", "force", "motion"}`` — ``smplx`` in the
            SmplxHead layout plus ``pelvis_world`` / ``root_rot_world`` /
            ``joints_world`` and the un-refined ``pelvis_world_in`` /
            ``root_rot_world_in``; absent heads are ``None``.
        """
        n_frames = tokens.shape[0]
        seq_len = int(batch["seq_len"])
        n_clips = n_frames // seq_len
        device = tokens.device
        ext = batch["cam_from_world"].to(device, torch.float32)
        rot_cw, t_cw = ext[:, :3, :3], ext[:, :3, 3]
        seconds = batch["frame_pos_sec"].to(device, torch.float32).view(n_clips, seq_len)
        valid = batch["frame_valid"].to(device, torch.bool).view(n_clips, seq_len)

        pelvis_cam = smplx_out["pelvis_cam"].float()
        root_rot_cam = smplx_out["root_rot"].float()
        body_rot = smplx_out["body_rot"].float()
        hand_rot = smplx_out["hand_rot"]
        hand_rot = None if hand_rot is None else hand_rot.float()
        betas = smplx_out["betas"].float()
        joints_cam = smplx_out["joints_cam"].float()

        # 1. depth smoothing in camera coordinates, bearing kept.
        ray = translation_to_ray(pelvis_cam)
        log_z = gaussian_smooth(ray[:, 2].view(n_clips, seq_len), seconds, valid,
                                self.depth_smooth_sec).reshape(n_frames)
        pelvis_s = ray_to_translation(torch.stack([ray[:, 0], ray[:, 1], log_z], dim=-1))

        # 2. lift to world; clip-mean betas.
        rot_wc = rot_cw.transpose(1, 2)
        p_w = (rot_wc @ (pelvis_s - t_cw)[..., None])[..., 0]
        rot_wr = rot_wc @ root_rot_cam
        w = valid.to(torch.float32)[..., None]
        betas_clip = (betas.view(n_clips, seq_len, -1) * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        betas_mean = betas_clip[:, None].expand(n_clips, seq_len, -1).reshape(n_frames, -1)

        # 3. world-independent per-frame features.
        joints_root = torch.einsum(
            "bji,bkj->bki", root_rot_cam, joints_cam[:, 1:NUM_BODY_JOINTS] - pelvis_cam[:, None])
        vel_w = time_derivative(p_w.view(n_clips, seq_len, 3), seconds, valid).reshape(n_frames, 3)
        vel_b = (rot_wr.transpose(1, 2) @ vel_w[..., None])[..., 0]
        ang_b = angular_velocity(rot_wr.view(n_clips, seq_len, 3, 3), seconds, valid)
        dt = local_dt(seconds, valid).reshape(n_frames, 1) * _DT_SCALE
        geometry = torch.cat(
            [joints_root.reshape(n_frames, -1), vel_b, ang_b.reshape(n_frames, 3), dt, betas_mean],
            dim=-1)
        feats = [self.geometry_norm(geometry)]
        token_feats = []
        if self.proj_pose_token is not None:
            token_feats.append(self.proj_pose_token(tokens[:, 0].float()))
        if self.proj_contact_tokens is not None:
            lo, hi = blocks["contact"]
            if hi - lo != self.num_contact_tokens:
                raise AssertionError(
                    f"contact block has {hi - lo} tokens; refiner built for {self.num_contact_tokens}")
            token_feats.append(
                self.proj_contact_tokens(tokens[:, lo:hi].float()).reshape(n_frames, -1))
        if token_feats:
            feats.append(self.token_norm(torch.cat(token_feats, dim=-1)))
        x = self.input_proj(torch.cat(feats, dim=-1))

        # 4. temporal transformer (one slot per frame).
        x = self.temporal(x[:, None], seq_len, batch["frame_pos_sec"], batch["frame_valid"])[:, 0]
        hidden = self.output_norm(x)
        raw = {name: head(hidden) for name, head in self.heads.items()}

        # 5. decode the pose offset in the body / parent-local frames.
        rot_wr2, body_rot2, p_w2 = rot_wr, body_rot, p_w
        if "pose" in raw:
            delta = raw["pose"]
            six = delta[:, :6 * NUM_BODY_JOINTS].reshape(n_frames, NUM_BODY_JOINTS, 6) + self.identity_6d
            d_rot = rot6d_to_rotmat(six)                                  # [B, 22, 3, 3]
            rot_wr2 = rot_wr @ d_rot[:, 0]
            body_rot2 = body_rot @ d_rot[:, 1:]
            p_w2 = p_w + (rot_wr @ delta[:, 6 * NUM_BODY_JOINTS:, None])[..., 0]

        # 6. FK in the world, then into every camera.
        q_world = smplx_q(p_w2, rot_wr2, body_rot2, hand_rot)
        joints_world = body.with_shape(betas=betas_mean).fk(q_world).joint_pose_world[..., 1:, :3]
        joints_cam2 = torch.einsum("bij,bkj->bki", rot_cw, joints_world) + t_cw[:, None]
        pelvis_cam2 = (rot_cw @ p_w2[..., None])[..., 0] + t_cw
        root_rot_cam2 = rot_cw @ rot_wr2
        kp2d_full, kp2d_crop = project_to_crop(
            joints_cam2, batch["cam_int"].float(), batch["affine_trans"].float(),
            batch["img_size"].float())
        smplx = {
            "root_6d": rotmat_to_rot6d(root_rot_cam2), "body_6d": rotmat_to_rot6d(body_rot2),
            "hand_6d": None if hand_rot is None else rotmat_to_rot6d(hand_rot),
            "root_rot": root_rot_cam2, "body_rot": body_rot2, "hand_rot": hand_rot,
            "betas": betas_mean, "cam": None, "ray": translation_to_ray(pelvis_cam2),
            "pelvis_cam": pelvis_cam2,
            "q_cam": smplx_q(pelvis_cam2, root_rot_cam2, body_rot2, hand_rot),
            "joints_cam": joints_cam2, "kp2d_full": kp2d_full, "kp2d_crop": kp2d_crop,
            "pelvis_world": p_w2, "root_rot_world": rot_wr2, "joints_world": joints_world,
            "pelvis_world_in": p_w, "root_rot_world_in": rot_wr,
        }
        contact = force = motion = None
        if "contact" in raw:
            contact = {"logits": raw["contact"], "probs": torch.sigmoid(raw["contact"])}
        if "force" in raw:
            # Forces live in the INPUT body frame; `frame` lets the loss rotate the kindyn
            # GT (given in the GT root frame) into it.
            force = {"forces": raw["force"].reshape(n_frames, NUM_GROUPS, 3), "frame": rot_wr}
        if "motion" in raw:
            m = raw["motion"]
            k = 3 * NUM_BODY_JOINTS
            motion = {
                "vel": m[:, :k].reshape(n_frames, NUM_BODY_JOINTS, 3),
                "acc": m[:, k:2 * k].reshape(n_frames, NUM_BODY_JOINTS, 3),
                "ang_vel": m[:, 2 * k:2 * k + 3], "ang_acc": m[:, 2 * k + 3:],
                "frame": rot_wr,                                        # world-from-body
            }
        return {"smplx": smplx, "contact": contact, "force": force, "motion": motion}


__all__ = ["TemporalRefiner", "OUTPUTS", "gaussian_smooth", "time_derivative",
           "angular_velocity", "local_dt", "stencil_valid"]
