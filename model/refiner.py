"""World-space temporal refiner behind the per-frame model (docs/refiner.md, docs/architecture_2.md).

The per-frame model (frozen SAM3D decoder + the SMPL-X / CLIFF heads) gives a
camera-frame body per frame. This module turns the clip into a WORLD-space
motion, encodes it frame by frame in a way that is independent of the world
frame, runs a local temporal transformer over the frames, and decodes
corrections — again without any reference to the world frame:

1. **Lift** with the frame extrinsics: ``p_w = R^T (p_c - t)``,
   ``R_world_root = R^T R_cam_root``; betas averaged over the clip.
2. **Input smoothing** (``root_smooth_sec`` / ``pose_smooth_sec``, the round-4
   recipe; 0 = off): Gaussian smoothing of the WORLD pelvis position after the
   lift (never in camera coordinates — those carry the camera's own motion and
   the lift no longer cancels it) and of the world root rotation and the
   parent-local joint rotations (matrix means projected onto SO(3)).
   ``learn_smoothing`` makes the widths parameters.
3. **Per-frame token** = 21 body-joint positions in the ROOT frame, the root's
   linear and angular velocity in the BODY frame, the frame spacing, the mean
   betas, optionally the camera context (``camera_context``: pelvis->camera
   direction in the body frame, log depth, crop-box bearing and angular size),
   optionally (``token``) the parent-local 6D joint rotations, the gravity
   direction in the body frame, the ``raw - 5-frame-mean`` channels of the
   root position and the root-frame joints (the noisiness, made visible) and
   the FORWARD and BACKWARD one-sided root rates in place of the central ones
   (``one_sided_velocity``: a central difference cancels a period-2 wobble), the
   projected frozen pose token and the projected contact tokens. Nothing refers
   to the world frame: a rigid re-definition of the world leaves every input
   unchanged.
4. **Temporal transformer**: :class:`~model.rope.CrossModalRopeModule` with a
   single slot — RoPE positions are seconds, a hard ``window`` per layer bounds
   the receptive field to ``num_layers x window``. With ``iterative`` the
   layers run one at a time: the shared pose head reads every layer, the deltas
   compose, and the corrected trajectory's own rate features go back into the
   residual stream before the next layer (SAM3D's decoder updates its anchored
   tokens with the intermediate keypoint readout the same way).
5. **Zero-initialised heads**: contact logits; pose offset = 6D rotation deltas
   right-multiplied onto the root (body frame) and the 21 body joints
   (parent-local) plus a root shift in the body frame; motion = world velocity /
   acceleration of the 22 joints and the root's angular velocity / acceleration,
   all expressed in the (input) body frame; forces in that same body frame.
6. **Decode**: FK in the world with the mean betas, then back into each camera
   with the extrinsics — the output dict has the :class:`~model.heads.SmplxHead`
   keys, so the existing SMPL-X loss and pose metrics apply unchanged.

At initialisation the refiner is exactly "per-frame model + smoothing": the
RoPE blocks are identities and every head is zero.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import roma
import torch
import torch.nn as nn
from torch import Tensor

from model.rope import CrossModalRopeModule
from utils.geometry import (project_to_crop, rot6d_to_rotmat, rotmat_to_rot6d, smplx_q,
                            translation_to_ray)

OUTPUTS = ("pose", "contact", "motion", "force")
NUM_BODY_JOINTS = 22
NUM_GROUPS = 6
_IDENTITY_6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
#: The frame-spacing input is expressed in 25-fps frames (1.0 at the corpus's reference rate).
_DT_SCALE = 25.0
#: Geometry features per frame: 21 root-frame joint positions (the pelvis IS the root, so
#: its row is identically zero and omitted), root linear + angular velocity, dt, 10 betas.
_GEOMETRY_DIM = 3 * (NUM_BODY_JOINTS - 1) + 3 + 3 + 1 + 10
#: Camera-context features: pelvis->camera direction in the body frame (3), pelvis log
#: depth (1), crop-box bearing (2) and angular size (1).
_CAMERA_DIM = 3 + 1 + 2 + 1
#: Feedback features of the iterative mode: the corrected trajectory's root-frame joint
#: positions, its one-sided root linear / angular rates and its per-joint ones.
_FEEDBACK_DIM = 3 * (NUM_BODY_JOINTS - 1) + 6 + 6 + 6 * (NUM_BODY_JOINTS - 1)
#: ... and with ``feedback_delta`` the cumulative correction: the root shift in the input
#: body frame and the composed root + 21 joint rotation deltas as 6D.
_FEEDBACK_DELTA_DIM = 3 + 6 * NUM_BODY_JOINTS
#: Half-width (frames) of the mean the ``raw - mean`` token channels subtract.
_RAW_MEAN_RADIUS = 2
#: Attribute names of the learnable smoothing widths (``learn_smoothing``); the trainer
#: gives them their own optimizer group (``optim.smoothing_lr_scale``).
SMOOTHING_PARAM_NAMES = ("log_root_sigma", "log_pose_sigma")
#: Bounds of the learnable log-widths: 1 ms .. 10 s.
_LOG_SIGMA_MIN, _LOG_SIGMA_MAX = math.log(1e-3), math.log(10.0)


# ------------------------------------------------------------------ time series helpers

def _trailing(mask: Tensor, like: Tensor) -> Tensor:
    """Reshape a ``[n, T]`` mask so it broadcasts over ``like``'s trailing dims."""
    return mask.reshape(*mask.shape, *([1] * (like.dim() - 2)))


def gaussian_smooth(x: Tensor, seconds: Tensor, valid: Tensor, sigma: float | Tensor) -> Tensor:
    """Masked Gaussian smoothing along time.

    :param x: ``[n, T, ...]`` series.
    :param seconds: ``[n, T]`` frame times.
    :param valid: ``[n, T]`` bool; invalid frames never contribute to others.
    :param sigma: kernel width in seconds: a float (``<= 0`` returns ``x``), or a
        tensor of ``J`` widths, one per channel of ``x``'s third axis (``x`` read
        as ``[n, T, J, ...]``; a one-element tensor is one width for all of ``x``).
        The output is differentiable w.r.t. a tensor ``sigma``.
    """
    if not torch.is_tensor(sigma):
        if sigma <= 0.0:
            return x
        sigma = torch.tensor(float(sigma), dtype=x.dtype, device=x.device)
    n, t = seconds.shape
    channels = sigma.numel()
    dt = seconds[:, :, None] - seconds[:, None, :]                              # [n, T, T]
    weights = torch.exp(-0.5 * (dt[..., None] / sigma.reshape(1, 1, 1, -1)) ** 2)   # [n, T, T, J]
    weights = weights * valid[:, None, :, None].to(x.dtype)
    eye = torch.eye(t, dtype=x.dtype, device=x.device)[None, :, :, None]
    weights = torch.maximum(weights, eye)                    # a frame always sees itself
    weights = weights / weights.sum(dim=2, keepdim=True)
    series = x.reshape(n, t, channels, -1)
    return torch.einsum("ntsj,nsjc->ntjc", weights, series).reshape(x.shape)


def project_rotation(mat: Tensor, iterations: int = 5) -> Tensor:
    """Nearest rotation to a non-singular ``(..., 3, 3)`` matrix (its polar factor).

    Higham's scaled Newton iteration ``X <- (g X + X^-T / g) / 2`` with
    ``g = |det X|^(-1/3)``: quadratically convergent to the orthogonal polar
    factor from any non-singular start, ~1e-6 in five steps. A batched 3x3
    inverse instead of the batched SVD of :func:`roma.special_procrustes`,
    which is serial on the GPU (seconds for a few thousand matrices).
    """
    x = mat
    for _ in range(iterations):
        gamma = torch.linalg.det(x).abs().clamp(min=1e-12).pow(-1.0 / 3.0)[..., None, None]
        x = 0.5 * (gamma * x + torch.linalg.inv(x).transpose(-1, -2) / gamma)
    return x


def smooth_rotations(rot: Tensor, seconds: Tensor, valid: Tensor, sigma: float | Tensor) -> Tensor:
    """Masked Gaussian smoothing of rotations ``[n, T, ..., 3, 3]``.

    The Gaussian-weighted mean of the matrices, projected back onto SO(3)
    (:func:`project_rotation`) — the chordal mean, exact for small spreads.
    ``sigma`` as in :func:`gaussian_smooth` (a float ``<= 0`` returns ``rot``).
    """
    if not torch.is_tensor(sigma) and sigma <= 0.0:
        return rot
    return project_rotation(gaussian_smooth(rot, seconds, valid, sigma))


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


def forward_difference(x: Tensor, seconds: Tensor, valid: Tensor) -> Tensor:
    """Forward difference ``(x[t + 1] - x[t]) / h`` of ``x`` ``[n, T, ...]`` at frame ``t``.

    Zero at the last frame and wherever ``t`` or ``t + 1`` is invalid
    (:func:`forward_valid` is the matching row mask). Unlike the central
    :func:`time_derivative` this has a non-zero response to a period-2
    component, so a Nyquist wobble is visible to it.
    """
    if seconds.shape[1] < 2:
        return torch.zeros_like(x)
    h = seconds[:, 1:] - seconds[:, :-1]
    ok = valid[:, 1:] & valid[:, :-1] & (h > 0)
    rate = (x[:, 1:] - x[:, :-1]) / _trailing(h.clamp(min=1e-6), x)
    rate = torch.where(_trailing(ok, rate), rate, torch.zeros_like(rate))
    return torch.cat([rate, torch.zeros_like(x[:, :1])], dim=1)


def backward_difference(x: Tensor, seconds: Tensor, valid: Tensor) -> Tensor:
    """Backward difference ``(x[t] - x[t - 1]) / h`` at frame ``t``: the forward
    difference of the previous frame (zero at the first frame and across holes)."""
    rate = forward_difference(x, seconds, valid)
    return torch.cat([torch.zeros_like(rate[:, :1]), rate[:, :-1]], dim=1)


def forward_valid(valid: Tensor) -> Tensor:
    """Frames whose own and next frame are valid — the forward difference's support."""
    out = torch.zeros_like(valid)
    out[:, :-1] = valid[:, :-1] & valid[:, 1:]
    return out


def second_difference(x: Tensor, seconds: Tensor, valid: Tensor) -> Tensor:
    """Centred second difference of ``x`` ``[n, T, ...]`` at the frame (span ``+-1``).

    ``2 ((x[t+1] - x[t]) / h_f - (x[t] - x[t-1]) / h_b) / (h_b + h_f)``; zero where a
    neighbour is missing (rows are masked by :func:`stencil_valid` downstream).
    """
    lo, hi, t_lo, t_hi, prev_ok, next_ok = _shifted(x, valid, seconds)
    h_b = (seconds - t_lo).clamp(min=1e-6)
    h_f = (t_hi - seconds).clamp(min=1e-6)
    fwd = (hi - x) / _trailing(h_f, x)
    bwd = (x - lo) / _trailing(h_b, x)
    acc = 2.0 * (fwd - bwd) / _trailing(h_b + h_f, x)
    return torch.where(_trailing(prev_ok & next_ok, x), acc, torch.zeros_like(acc))


def angular_acceleration(rot: Tensor, seconds: Tensor, valid: Tensor) -> Tensor:
    """Body-frame angular acceleration of world-from-body rotations ``[n, T, 3, 3]``.

    The forward and backward step rates ``log(R_t^T R_{t+-1}) / h`` are the two
    adjacent midpoint velocities; their difference over the half-span
    ``(h_b + h_f) / 2`` is the centred second difference (span ``+-1``, the
    rotational twin of :func:`second_difference`). Zero where a neighbour is missing.
    """
    n, t = seconds.shape
    zeros = torch.zeros(n, 1, 3, dtype=rot.dtype, device=rot.device)
    if t < 3:
        return zeros.expand(n, t, 3)
    inc = roma.rotmat_to_rotvec(rot[:, :-1].transpose(-1, -2) @ rot[:, 1:])   # [n, T-1, 3]
    dt = (seconds[:, 1:] - seconds[:, :-1]).clamp(min=1e-6)
    ok = valid[:, 1:] & valid[:, :-1]
    rate = inc / dt[..., None]
    span = 0.5 * (dt[:, 1:] + dt[:, :-1])
    acc = (rate[:, 1:] - rate[:, :-1]) / span[..., None]                      # frames 1..T-2
    acc = torch.where((ok[:, 1:] & ok[:, :-1])[..., None], acc, torch.zeros_like(acc))
    return torch.cat([zeros, acc, zeros], dim=1)


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


def forward_angular_velocity(rot: Tensor, seconds: Tensor, valid: Tensor) -> Tensor:
    """Forward angular rate ``log(R_t^T R_{t+1}) / h`` of ``[n, T, 3, 3]``, in the frame of ``t``.

    The one-sided twin of :func:`angular_velocity` (which averages the forward
    and backward steps and is therefore blind to a period-2 alternation).
    Zero at the last frame and wherever ``t`` or ``t + 1`` is invalid.
    """
    n, t = seconds.shape
    zeros = torch.zeros(n, 1, 3, dtype=rot.dtype, device=rot.device)
    if t < 2:
        return zeros.expand(n, t, 3)
    inc = roma.rotmat_to_rotvec(rot[:, :-1].transpose(-1, -2) @ rot[:, 1:])   # [n, T-1, 3]
    dt = seconds[:, 1:] - seconds[:, :-1]
    ok = valid[:, 1:] & valid[:, :-1] & (dt > 0)
    rate = torch.where(ok[..., None], inc / dt.clamp(min=1e-6)[..., None], torch.zeros_like(inc))
    return torch.cat([rate, zeros], dim=1)


def backward_angular_velocity(rot: Tensor, seconds: Tensor, valid: Tensor) -> Tensor:
    """Backward angular rate ``log(R_{t-1}^T R_t) / h`` at frame ``t``.

    The increment fixes its own axis, so the previous step's rate is the same
    vector in the frames of ``t - 1`` and ``t``: the forward rate, shifted.
    """
    rate = forward_angular_velocity(rot, seconds, valid)
    return torch.cat([torch.zeros_like(rate[:, :1]), rate[:, :-1]], dim=1)


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


def neighbours(x: Tensor, valid: Tensor, radius: int) -> tuple[Tensor, Tensor]:
    """The ``+- radius`` neighbours of every frame of ``x`` ``[n, T, ...]``.

    :returns: ``(stack [n, T, 2 radius + 1, ...], mask [n, T, 2 radius + 1])`` —
        offsets ``-radius .. radius`` in order; a neighbour outside the clip or
        invalid is masked (its value is a clamped copy of the frame's own).
    """
    n, t = valid.shape
    index = torch.arange(t, device=x.device)[:, None] + torch.arange(
        -radius, radius + 1, device=x.device)[None, :]                        # [T, 2r+1]
    inside = (index >= 0) & (index < t)
    index = index.clamp(0, t - 1)
    mask = valid[:, index] & inside[None]                                     # [n, T, 2r+1]
    mask[:, :, radius] = True                                # a frame always sees itself
    stack = x[:, index]                                                       # [n, T, 2r+1, ...]
    return stack, mask


def local_mean(x: Tensor, valid: Tensor, radius: int) -> Tensor:
    """Mean of ``x`` ``[n, T, ...]`` over the valid ``+- radius`` neighbours (self included)."""
    stack, mask = neighbours(x, valid, radius)
    weight = mask.to(x.dtype).reshape(*mask.shape, *([1] * (x.dim() - 2)))
    return (stack * weight).sum(dim=2) / weight.sum(dim=2).clamp(min=1.0)


# ------------------------------------------------------------------ body-relative features

def world_joints(shaped, pelvis_world: Tensor, rot_wr: Tensor, body_rot: Tensor,
                 hand_rot: Optional[Tensor]) -> Tensor:
    """FK of a world trajectory with an already-shaped body: world joints ``[n, J, 3]``."""
    return shaped.fk(smplx_q(pelvis_world, rot_wr, body_rot, hand_rot)
                     ).joint_pose_world[..., 1:, :3]


def root_frame_joints(joints_world: Tensor, pelvis_world: Tensor, rot_wr: Tensor) -> Tensor:
    """The 21 body joints of a world trajectory in its root frame, ``[n, 21, 3]``."""
    return torch.einsum("bji,bkj->bki", rot_wr,
                        joints_world[:, 1:NUM_BODY_JOINTS] - pelvis_world[:, None])


def one_sided_root_rates(pelvis_world: Tensor, rot_wr: Tensor, seconds: Tensor,
                         valid: Tensor) -> tuple[Tensor, Tensor]:
    """Forward AND backward one-sided root rates in the body frame, ``[n, 6]`` each.

    A central difference averages the two steps and cancels a period-2 alternation
    exactly; the two one-sided rates keep it. ``seconds`` / ``valid`` are ``[n_clips, T]``.
    """
    n_clips, seq_len = seconds.shape
    n_frames = n_clips * seq_len
    p_clip = pelvis_world.view(n_clips, seq_len, 3)
    rot_clip = rot_wr.view(n_clips, seq_len, 3, 3)
    to_body = rot_wr.transpose(1, 2)
    rates = [forward_difference(p_clip, seconds, valid).reshape(n_frames, 3),
             backward_difference(p_clip, seconds, valid).reshape(n_frames, 3)]
    vel_b = torch.cat([(to_body @ r[..., None])[..., 0] for r in rates], dim=-1)
    ang_b = torch.cat([forward_angular_velocity(rot_clip, seconds, valid),
                       backward_angular_velocity(rot_clip, seconds, valid)],
                      dim=-1).reshape(n_frames, 6)
    return vel_b, ang_b


def joint_rates(body_rot: Tensor, seconds: Tensor, valid: Tensor) -> Tensor:
    """Forward AND backward one-sided parent-local angular rates of the 21 joints, ``[n, 126]``.

    The joints' twin of :func:`one_sided_root_rates`; each rate lives in the frame of
    ``t``, so a rigid re-definition of the world leaves it untouched.
    """
    n_clips, seq_len = seconds.shape
    n_frames = n_clips * seq_len
    n_joints = NUM_BODY_JOINTS - 1
    rot = body_rot.view(n_clips, seq_len, n_joints, 3, 3).transpose(1, 2)
    rot = rot.reshape(n_clips * n_joints, seq_len, 3, 3)
    sec_j = seconds.repeat_interleave(n_joints, dim=0)
    valid_j = valid.repeat_interleave(n_joints, dim=0)
    rates = [forward_angular_velocity(rot, sec_j, valid_j),
             backward_angular_velocity(rot, sec_j, valid_j)]              # [n J, T, 3]
    return torch.cat([r.view(n_clips, n_joints, seq_len, 3).transpose(1, 2).reshape(n_frames, -1)
                      for r in rates], dim=-1)


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
    :param root_smooth_sec: input Gaussian sigma (s) of the world pelvis
        position after the lift (0 = off).
    :param pose_smooth_sec: input Gaussian sigma (s) of the root / joint
        rotation smoothing (0 = off).
    :param learn_smoothing: make the input widths trainable — one log-sigma
        for the root position and one per rotation (root + 21 joints),
        initialised from the two ``*_smooth_sec`` values (both must be > 0).
    :param token: extra token channels, ``{local_rotations, gravity,
        raw_minus_mean, one_sided_velocity, joint_velocity}`` (bools).
    :param iterative: run the layers one at a time, apply the shared pose
        head's delta after every one of them and feed the corrected
        trajectory's rate features back into the residual stream (needs the
        ``pose`` output and the ``one_sided_velocity`` / ``joint_velocity``
        token channels, whose features the feedback recomputes).
    :param feedback_delta: add the cumulative pose correction (root shift +
        composed 6D rotation deltas) to that feedback (``iterative`` only).
    :param camera_context: append the camera-context features to the token.
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
        root_smooth_sec: float = 0.0,
        pose_smooth_sec: float = 0.0,
        learn_smoothing: bool = False,
        token: Optional[dict] = None,
        iterative: bool = False,
        feedback_delta: bool = False,
        camera_context: bool = False,
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
        self.root_smooth_sec = float(root_smooth_sec)
        self.pose_smooth_sec = float(pose_smooth_sec)
        self.learn_smoothing = bool(learn_smoothing)
        if self.learn_smoothing:
            if not (self.root_smooth_sec > 0.0 and self.pose_smooth_sec > 0.0):
                raise ValueError(
                    "learn_smoothing needs positive root_smooth_sec / pose_smooth_sec as the "
                    "initial widths")
            self.log_root_sigma = nn.Parameter(torch.full((1,), math.log(self.root_smooth_sec)))
            self.log_pose_sigma = nn.Parameter(
                torch.full((NUM_BODY_JOINTS,), math.log(self.pose_smooth_sec)))
        token = {"local_rotations": False, "gravity": False, "raw_minus_mean": False,
                 "one_sided_velocity": False, "joint_velocity": False, **(token or {})}
        self.token_local_rotations = bool(token["local_rotations"])
        self.token_gravity = bool(token["gravity"])
        self.token_raw_minus_mean = bool(token["raw_minus_mean"])
        self.token_one_sided_velocity = bool(token["one_sided_velocity"])
        self.token_joint_velocity = bool(token["joint_velocity"])
        self.iterative = bool(iterative)
        self.feedback_delta = bool(feedback_delta)
        if self.feedback_delta and not self.iterative:
            raise ValueError("feedback_delta is the iterative feedback's extra channels: "
                             "it needs iterative")
        if self.iterative:
            if "pose" not in self.outputs:
                raise ValueError("iterative refinement corrects the pose after every layer: "
                                 "list 'pose' in outputs")
            if not (self.token_one_sided_velocity and self.token_joint_velocity):
                raise ValueError(
                    "iterative refinement feeds the corrected trajectory's one-sided root and "
                    "joint rates back: token.one_sided_velocity and token.joint_velocity must "
                    "be on, so the input token carries the same channels")
        self.camera_context = bool(camera_context)
        self.time_scale = float(time_scale)

        self.proj_pose_token = nn.Linear(decoder_dim, pose_token_dim) if pose_token else None
        self.proj_contact_tokens = (nn.Linear(decoder_dim, contact_token_dim)
                                    if self.num_contact_tokens > 0 else None)
        token_dim = (pose_token_dim if pose_token else 0) + self.num_contact_tokens * contact_token_dim
        # Two LayerNorms: the geometry numbers and the projected token channels are
        # normalised separately, so neither group's scale rides on the other's width.
        geometry_dim = _GEOMETRY_DIM + (_CAMERA_DIM if self.camera_context else 0)
        geometry_dim += 6 if self.token_one_sided_velocity else 0   # two one-sided rates, not one central
        geometry_dim += 6 * (NUM_BODY_JOINTS - 1) if self.token_joint_velocity else 0
        geometry_dim += 6 * (NUM_BODY_JOINTS - 1) if self.token_local_rotations else 0
        geometry_dim += 3 if self.token_gravity else 0
        geometry_dim += 3 + 3 * (NUM_BODY_JOINTS - 1) if self.token_raw_minus_mean else 0
        self.geometry_norm = nn.LayerNorm(geometry_dim)
        self.token_norm = nn.LayerNorm(token_dim) if token_dim > 0 else None
        self.input_proj = nn.Linear(geometry_dim + token_dim, dim)
        self.temporal = CrossModalRopeModule(
            dim=dim, num_slots=1, num_layers=num_layers, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout, window=window, time_scale=time_scale)
        self.output_norm = nn.LayerNorm(dim)
        self.feedback_norm = self.feedback_proj = None
        if self.iterative:
            # Zero-initialised, like the heads: at init the extra path contributes nothing.
            feedback_dim = _FEEDBACK_DIM + (_FEEDBACK_DELTA_DIM if self.feedback_delta else 0)
            self.feedback_norm = nn.LayerNorm(feedback_dim)
            self.feedback_proj = nn.Linear(feedback_dim, dim)
            nn.init.zeros_(self.feedback_proj.weight)
            nn.init.zeros_(self.feedback_proj.bias)
        sizes = {"pose": 6 * NUM_BODY_JOINTS + 3, "contact": NUM_GROUPS,
                 "motion": 6 * NUM_BODY_JOINTS + 6, "force": 3 * NUM_GROUPS}
        self.heads = nn.ModuleDict()
        for name in self.outputs:
            self.heads[name] = self._zero_head(dim, sizes[name])
        self.register_buffer("identity_6d", torch.tensor(_IDENTITY_6D), persistent=False)

    @staticmethod
    def _zero_head(dim: int, out: int) -> nn.Sequential:
        head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, out))
        nn.init.zeros_(head[2].weight)
        nn.init.zeros_(head[2].bias)
        return head

    # ------------------------------------------------------------------ smoothing

    def smoothing_sigmas(self) -> tuple[float | Tensor, float | Tensor, float | Tensor]:
        """Input kernel widths in seconds: (root position, root rotation, the 21 joint rotations).

        Floats from the config when fixed; tensors ``(1,)``, ``(1,)``, ``(21,)`` of the
        clamped learnable widths under ``learn_smoothing``.
        """
        if not self.learn_smoothing:
            return self.root_smooth_sec, self.pose_smooth_sec, self.pose_smooth_sec
        pose = self.log_pose_sigma.clamp(_LOG_SIGMA_MIN, _LOG_SIGMA_MAX).exp()
        root = self.log_root_sigma.clamp(_LOG_SIGMA_MIN, _LOG_SIGMA_MAX).exp()
        return root, pose[:1], pose[1:]

    def smoothing_scalars(self) -> dict[str, float]:
        """The learned widths for the train log (empty when the widths are fixed)."""
        if not self.learn_smoothing:
            return {}
        with torch.no_grad():
            root, root_rot, joints = self.smoothing_sigmas()
        return {"smoothing/root_sigma": float(root), "smoothing/root_rot_sigma": float(root_rot),
                "smoothing/joint_sigma_min": float(joints.min()),
                "smoothing/joint_sigma_mean": float(joints.mean()),
                "smoothing/joint_sigma_max": float(joints.max())}

    # ------------------------------------------------------------------ pose update

    def _apply_pose_delta(self, delta: Tensor, pelvis_world: Tensor, rot_wr: Tensor,
                          body_rot: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """One pose-head output applied to a world trajectory.

        The 22 6D deltas are right-multiplied onto the root (body frame) and the 21
        parent-local joint rotations; the root shift is read in the trajectory's CURRENT
        body frame.
        """
        n_frames = delta.shape[0]
        six = delta[:, :6 * NUM_BODY_JOINTS].reshape(n_frames, NUM_BODY_JOINTS, 6) + self.identity_6d
        d_rot = rot6d_to_rotmat(six)                                      # [B, 22, 3, 3]
        shift = (rot_wr @ delta[:, 6 * NUM_BODY_JOINTS:, None])[..., 0]
        return pelvis_world + shift, rot_wr @ d_rot[:, 0], body_rot @ d_rot[:, 1:]

    def _feedback(self, pelvis_world: Tensor, rot_wr: Tensor, body_rot: Tensor,
                  joints_world: Tensor, pelvis_in: Tensor, rot_wr_in: Tensor,
                  body_rot_in: Tensor, seconds: Tensor, valid: Tensor) -> Tensor:
        """The corrected trajectory's own features, projected into the residual stream.

        The channels the input token carries (root-frame joint positions, the one-sided
        root and per-joint rates, all body-relative) recomputed on the trajectory as the
        layers so far left it, so the next layer reads ITS rates and not the raw ones.
        With ``feedback_delta`` the CUMULATIVE correction is appended: the root shift in
        the input body frame and the composed root / joint rotation deltas as 6D — rates
        are blind to the slow drift the layers have already accumulated, and the
        correction measured against the un-refined trajectory is frame-independent too.
        """
        n_frames = pelvis_world.shape[0]
        vel_b, ang_b = one_sided_root_rates(pelvis_world, rot_wr, seconds, valid)
        feats = [root_frame_joints(joints_world, pelvis_world, rot_wr).reshape(n_frames, -1),
                 vel_b, ang_b, joint_rates(body_rot, seconds, valid)]
        if self.feedback_delta:
            to_in = rot_wr_in.transpose(1, 2)
            feats += [(to_in @ (pelvis_world - pelvis_in)[..., None])[..., 0],
                      rotmat_to_rot6d(to_in @ rot_wr),
                      rotmat_to_rot6d(body_rot_in.transpose(2, 3) @ body_rot).reshape(n_frames, -1)]
        return self.feedback_proj(self.feedback_norm(torch.cat(feats, dim=-1)))

    # ------------------------------------------------------------------ forward

    def forward(self, smplx_out: dict, tokens: Tensor, blocks: dict, batch: dict, body) -> dict:
        """Refine one batch of flattened clips.

        :param smplx_out: the per-frame :class:`~model.heads.SmplxHead` output.
        :param tokens: final decoder tokens ``[B, N, C]`` (pose token at 0).
        :param blocks: token-block bounds (``blocks["contact"]`` when present).
        :param batch: collated batch (``seq_len``, ``frame_pos_sec``,
            ``frame_valid``, ``cam_from_world``, ``cam_int``, ``affine_trans``,
            ``img_size``, ``bbox_center``, ``bbox_scale``; ``gravity_world`` with
            the gravity token channel).
        :param body: the head's BetterHuman SMPL-X body (22 or 52 joints).
        :returns: ``{"smplx", "contact", "force", "motion"}`` — ``smplx`` in the
            SmplxHead layout plus ``pelvis_world`` / ``root_rot_world`` /
            ``joints_world``, the per-layer world joints ``joints_world_layers``
            (one entry unless ``iterative``, the last one IS ``joints_world``)
            and the smoothed, un-refined ``pelvis_world_in`` /
            ``root_rot_world_in`` / ``body_rot_in`` / ``joints_world_in``;
            absent heads are ``None``.
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

        # 1. lift the pelvis to the world and smooth it THERE: camera coordinates carry the
        #    camera's own motion, smoothing them strips it from the signal and the lift no
        #    longer cancels it (2026-09-06: jitter 30 -> the GT floor on moving cameras).
        root_sigma, root_rot_sigma, joint_sigma = self.smoothing_sigmas()
        rot_wc = rot_cw.transpose(1, 2)
        p_w = (rot_wc @ (pelvis_cam - t_cw)[..., None])[..., 0]
        p_w = gaussian_smooth(p_w.view(n_clips, seq_len, 3), seconds, valid,
                              root_sigma).reshape(n_frames, 3)
        pelvis_s = (rot_cw @ p_w[..., None])[..., 0] + t_cw          # the smoothed root, per camera

        # 2. smooth the rotations in the world / parent-local frames; clip-mean betas.
        rot_wr = smooth_rotations((rot_wc @ root_rot_cam).view(n_clips, seq_len, 3, 3), seconds,
                                  valid, root_rot_sigma).reshape(n_frames, 3, 3)
        body_rot = smooth_rotations(body_rot.view(n_clips, seq_len, NUM_BODY_JOINTS - 1, 3, 3),
                                    seconds, valid, joint_sigma
                                    ).reshape(n_frames, NUM_BODY_JOINTS - 1, 3, 3)
        w = valid.to(torch.float32)[..., None]
        betas_clip = (betas.view(n_clips, seq_len, -1) * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        betas_mean = betas_clip[:, None].expand(n_clips, seq_len, -1).reshape(n_frames, -1)
        shaped = body.with_shape(betas=betas_mean)
        joints_world_in = world_joints(shaped, p_w, rot_wr, body_rot, hand_rot)

        # 3. world-independent per-frame features.
        joints_root = root_frame_joints(joints_world_in, p_w, rot_wr)
        dt = local_dt(seconds, valid).reshape(n_frames, 1) * _DT_SCALE
        if self.token_one_sided_velocity:
            vel_b, ang_b = one_sided_root_rates(p_w, rot_wr, seconds, valid)
        else:
            vel_w = time_derivative(p_w.view(n_clips, seq_len, 3), seconds, valid
                                    ).reshape(n_frames, 3)
            vel_b = (rot_wr.transpose(1, 2) @ vel_w[..., None])[..., 0]
            ang_b = angular_velocity(rot_wr.view(n_clips, seq_len, 3, 3), seconds, valid
                                     ).reshape(n_frames, 3)
        geometry = [joints_root.reshape(n_frames, -1), vel_b, ang_b, dt, betas_mean]
        if self.camera_context:
            # The camera seen from the body: where the per-frame depth error points, and how
            # far / how large the crop was (the CLIFF lift's inputs). All frame-independent.
            root_rot_cam_s = rot_cw @ rot_wr
            to_camera = -pelvis_s / pelvis_s.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            cam_dir_b = (root_rot_cam_s.transpose(1, 2) @ to_camera[..., None])[..., 0]
            cam_int = batch["cam_int"].float()
            focal = cam_int[:, 0, 0].clamp(min=1.0)
            centre = batch["bbox_center"].float()
            box_bearing = (centre - cam_int[:, :2, 2]) / focal[:, None]
            box_size = batch["bbox_scale"].float()[:, 0] / focal
            log_z = torch.log(pelvis_s[:, 2].clamp(min=1e-3))
            geometry += [cam_dir_b, log_z[:, None], box_bearing, box_size[:, None]]
        if self.token_local_rotations:
            geometry.append(rotmat_to_rot6d(body_rot).reshape(n_frames, -1))
        if self.token_joint_velocity:
            # The root's smoothing is a signed sum of its one-sided rates over the neighbours;
            # give every joint's parent-local rotation the same two rates instead of only its
            # absolute value.
            geometry.append(joint_rates(body_rot, seconds, valid))
        if self.token_gravity:
            # The scene's down vector seen from the body: pitch / roll relative to gravity
            # (a heading relative to gravity would need a world reference, so none is given).
            gravity_w = batch["gravity_world"].to(device, torch.float32)
            geometry.append((rot_wr.transpose(1, 2) @ gravity_w[..., None])[..., 0])
        if self.token_raw_minus_mean:
            p_mean = local_mean(p_w.view(n_clips, seq_len, 3), valid, _RAW_MEAN_RADIUS)
            dp_w = p_w - p_mean.reshape(n_frames, 3)
            dp_b = (rot_wr.transpose(1, 2) @ dp_w[..., None])[..., 0]
            j_mean = local_mean(joints_root.reshape(n_clips, seq_len, -1), valid, _RAW_MEAN_RADIUS)
            geometry += [dp_b, joints_root.reshape(n_frames, -1) - j_mean.reshape(n_frames, -1)]
        geometry = torch.cat(geometry, dim=-1)
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

        # 4. temporal transformer (one slot per frame), 5. the pose offset in the body /
        #    parent-local frames and 6. FK in the world. Under `iterative` the three steps
        #    interleave: every layer's delta lands on the trajectory the previous layers
        #    left and its rate features go back into the residual stream.
        n_layers = self.temporal.num_layers
        states: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        rot_wr2, body_rot2, p_w2 = rot_wr, body_rot, p_w
        if self.iterative:
            tables = self.temporal.prepare(x[:, None], seq_len, batch["frame_pos_sec"],
                                           batch["frame_valid"])
            h = x[:, None]
            for layer in range(n_layers):
                h = self.temporal.run_block(layer, h, tables)
                hidden = self.output_norm(h[:, 0])
                p_w2, rot_wr2, body_rot2 = self._apply_pose_delta(
                    self.heads["pose"](hidden), p_w2, rot_wr2, body_rot2)
                states.append((p_w2, rot_wr2, body_rot2,
                               world_joints(shaped, p_w2, rot_wr2, body_rot2, hand_rot)))
                if layer + 1 < n_layers:
                    h = h + self._feedback(*states[-1], p_w, rot_wr, body_rot,
                                           seconds, valid)[:, None]
        else:
            x = self.temporal(x[:, None], seq_len, batch["frame_pos_sec"],
                              batch["frame_valid"])[:, 0]
            hidden = self.output_norm(x)
            if "pose" in self.outputs:
                p_w2, rot_wr2, body_rot2 = self._apply_pose_delta(
                    self.heads["pose"](hidden), p_w2, rot_wr2, body_rot2)
            states.append((p_w2, rot_wr2, body_rot2,
                           world_joints(shaped, p_w2, rot_wr2, body_rot2, hand_rot)))
        raw = {name: head(hidden) for name, head in self.heads.items() if name != "pose"}

        # 7. back into every camera — the intermediate layers too (deep supervision reads
        #    them; with one layer the lists are the final tensors and nothing extra runs).
        joints_world = states[-1][3]
        joints_cam2 = torch.einsum("bij,bkj->bki", rot_cw, joints_world) + t_cw[:, None]
        pelvis_cam2 = (rot_cw @ p_w2[..., None])[..., 0] + t_cw
        root_rot_cam2 = rot_cw @ rot_wr2
        kp2d_full, kp2d_crop = project_to_crop(
            joints_cam2, batch["cam_int"].float(), batch["affine_trans"].float(),
            batch["img_size"].float())
        root_6d, body_6d = rotmat_to_rot6d(root_rot_cam2), rotmat_to_rot6d(body_rot2)
        layers = [
            {"pelvis_world": p, "root_rot_world": r, "joints_world": j,
             "joints_cam": torch.einsum("bij,bkj->bki", rot_cw, j) + t_cw[:, None],
             "root_6d": rotmat_to_rot6d(rot_cw @ r), "body_6d": rotmat_to_rot6d(b)}
            for p, r, b, j in states[:-1]]
        layers.append({"pelvis_world": p_w2, "root_rot_world": rot_wr2,
                       "joints_world": joints_world, "joints_cam": joints_cam2,
                       "root_6d": root_6d, "body_6d": body_6d})
        smplx = {
            "root_6d": root_6d, "body_6d": body_6d,
            "hand_6d": None if hand_rot is None else rotmat_to_rot6d(hand_rot),
            "root_rot": root_rot_cam2, "body_rot": body_rot2, "hand_rot": hand_rot,
            "betas": betas_mean, "cam": None, "ray": translation_to_ray(pelvis_cam2),
            "pelvis_cam": pelvis_cam2,
            "q_cam": smplx_q(pelvis_cam2, root_rot_cam2, body_rot2, hand_rot),
            "joints_cam": joints_cam2, "kp2d_full": kp2d_full, "kp2d_crop": kp2d_crop,
            "pelvis_world": p_w2, "root_rot_world": rot_wr2, "joints_world": joints_world,
            "pelvis_world_in": p_w, "root_rot_world_in": rot_wr, "body_rot_in": body_rot,
            "joints_world_in": joints_world_in,
            **{f"{key}_layers": [layer[key] for layer in layers] for key in layers[0]},
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


__all__ = ["TemporalRefiner", "OUTPUTS", "SMOOTHING_PARAM_NAMES", "world_joints",
           "root_frame_joints", "one_sided_root_rates", "joint_rates",
           "gaussian_smooth", "smooth_rotations", "project_rotation", "time_derivative",
           "second_difference", "angular_velocity", "angular_acceleration", "local_dt",
           "stencil_valid", "neighbours", "local_mean", "forward_difference",
           "backward_difference", "forward_valid", "forward_angular_velocity",
           "backward_angular_velocity"]
