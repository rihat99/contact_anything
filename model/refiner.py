"""World-space temporal refiner behind the per-frame model (docs/old/refiner.md, docs/old/architecture_2.md).

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
   projected frozen pose token and the projected contact tokens. With
   ``gravity_input`` the scene's gravity is HANDED to the clip (its direction in
   the body frame plus a flag) instead of being guessed. Nothing refers
   to the world frame: a rigid re-definition of the world leaves every input
   unchanged.
4. **Temporal transformer**: :class:`~model.rope.CrossModalRopeModule` with a
   single slot — RoPE positions are seconds, a hard ``window`` per layer bounds
   the receptive field to ``num_layers x window``. With ``iterative`` the
   layers run one at a time: the shared pose head reads every layer, the deltas
   compose, and the corrected trajectory's own rate features go back into the
   residual stream before the next layer (SAM3D's decoder updates its anchored
   tokens with the intermediate keypoint readout the same way). With
   ``limb_tokens`` a frame carries ``1 + K`` tokens — the body token (whose decoder
   contact tokens move to the limbs) and one per slot of the contact set holding
   that slot's geometry and its decoder contact token — through alternating
   temporal / within-frame attention; the contact and force heads then read the
   limb tokens, and the limbs get a feedback path of their own. With
   ``force_tokens`` K more limb tokens carry the decoder FORCE tokens: the
   force head reads those, the contact head the contact ones (``1 + 2K`` slots).
5. **Zero-initialised heads**: contact logits; pose offset = 6D rotation deltas
   right-multiplied onto the root (body frame) and the 21 body joints
   (parent-local) plus a root shift in the body frame; motion = world velocity /
   acceleration of the 22 joints and the root's angular velocity / acceleration,
   all expressed in the (input) body frame; forces in that same body frame, or
   (``force_frame: gravity``) in a gravity-aligned, body-headed one, with
   ``out["force"]["frame"]`` the world-from-that rotation either way;
   gravity = a body-frame correction on top of the camera's down axis
   (``camera_axes``), averaged over the clip in the world and normalised — one
   down vector per clip, so the world serves only as transport between frames;
   a clip that was GIVEN its gravity (``gravity_input``) returns that vector and
   the head's votes are discarded.
   Under ``iterative`` every head reads every layer, and the contact
   probabilities, the body-frame gravity and (``residual_feedback``) the RNEA
   root-wrench residual of the layer's body under its gated forces
   (:class:`~model.physics.RootWrench`; body detached, forces live) are fed back
   with the rate features. ``frame_mask_p`` replaces random input tokens by a
   learned embedding during training (the feedback and the losses unchanged).
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

from model.contact_frames import ContactSet, contact_set
from model.rope import CrossModalRopeModule
from utils.geometry import (project_to_crop, rot6d_to_rotmat, rotmat_to_rot6d, smplx_q,
                            translation_to_ray)

OUTPUTS = ("pose", "contact", "motion", "force", "gravity")
NUM_BODY_JOINTS = 22
#: Heads that read the limb tokens under ``limb_tokens`` (the body token otherwise).
_LIMB_HEADS = ("contact", "force")
#: Frames the ``force`` output may be read in (``force_frame``).
FORCE_FRAMES = ("body", "gravity")
#: ``force_frame: gravity``: a body axis whose horizontal part is shorter than this cannot
#: head the frame (the body lies along gravity), so the next axis takes over.
_FORCE_FRAME_MIN_HORIZONTAL = 0.2
_IDENTITY_6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
#: The frame-spacing input is expressed in 25-fps frames (1.0 at the corpus's reference rate).
_DT_SCALE = 25.0
#: Geometry features per frame: 21 root-frame joint positions (the pelvis IS the root, so
#: its row is identically zero and omitted), root linear + angular velocity, dt, 10 betas.
_GEOMETRY_DIM = 3 * (NUM_BODY_JOINTS - 1) + 3 + 3 + 1 + 10
#: Camera-context features: pelvis->camera direction in the body frame (3), pelvis log
#: depth (1), crop-box bearing (2) and angular size (1).
_CAMERA_DIM = 3 + 1 + 2 + 1
#: ``camera_axes``: the camera's down (+y) and viewing (+z) axes in the body frame.
_CAMERA_AXES_DIM = 6
#: ``gravity_input``: the given gravity in the body frame (3) and the flag that says
#: it was given (1); both zero on a clip that was not given it.
_GRAVITY_INPUT_DIM = 4
#: ``residual_feedback``: the root-wrench residual, force (bw) + torque (bw*m).
_RESIDUAL_DIM = 6
#: Pooled gravity votes whose mean is shorter than this fall back to the camera axis.
_GRAVITY_MIN_NORM = 0.1
#: Frames a clip needs for the RNEA residual's +-2 stencil to have one interior row.
_RESIDUAL_MIN_FRAMES = 5
#: Feedback features of the iterative mode: the corrected trajectory's root-frame joint
#: positions, its one-sided root linear / angular rates and its per-joint ones.
_FEEDBACK_DIM = 3 * (NUM_BODY_JOINTS - 1) + 6 + 6 + 6 * (NUM_BODY_JOINTS - 1)
#: ... and with ``feedback_delta`` the cumulative correction: the root shift in the input
#: body frame and the composed root + 21 joint rotation deltas as 6D.
_FEEDBACK_DELTA_DIM = 3 + 6 * NUM_BODY_JOINTS
#: ``limb_tokens``: per-limb geometry — the extremity joint's root-frame position and its
#: forward + backward one-sided body-frame rates.
_LIMB_GEOMETRY_DIM = 3 + 6
#: ``per_frame``: the attention window (seconds) that keeps every query on its own frame
#: (frames are never closer than 1/120 s).
_PER_FRAME_WINDOW = 1e-3
#: ... and with ``camera_context`` its crop-space 2D position and log depth ratio to the pelvis.
_LIMB_CAMERA_DIM = 3
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

def slot_frame_ids(slots: ContactSet, body) -> Optional[list[int]]:
    """The BetterRobot frame ids of a set's posed contact frames (``None`` for joint slots).

    Hard-fails when the body's frame set is not the contact set's: every slot tensor in
    the repository is indexed by :attr:`ContactSet.slot_names`, in that order.
    """
    if not slots.uses_frames:
        return None
    frames = body.contact_frames
    if frames is None or tuple(frames.names) != slots.slot_names:
        raise ValueError(
            f"the SMPL-X body carries {None if frames is None else tuple(frames.names)}, "
            f"not the {slots.name} frames")
    return list(frames.frame_ids)


def world_points(shaped, pelvis_world: Tensor, rot_wr: Tensor, body_rot: Tensor,
                 hand_rot: Optional[Tensor], slots: ContactSet,
                 frame_ids: Optional[Sequence[int]]) -> tuple[Tensor, Tensor]:
    """FK of a world trajectory with an already-shaped body.

    :returns: ``(world joints [n, J, 3], the set's slot points [n, K, 3])`` — the posed
        contact frames when the set has them (``frame_ids`` from :func:`slot_frame_ids`),
        else the slots' own body joints.
    """
    data = shaped.fk(smplx_q(pelvis_world, rot_wr, body_rot, hand_rot),
                     compute_frames=frame_ids is not None)
    joints = data.joint_pose_world[..., 1:, :3]
    points = (data.frame_pose_world[..., frame_ids, :3] if frame_ids is not None
              else joints[:, list(slots.parent_joint52)])
    return joints, points


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


def limb_geometry(points_world: Tensor, pelvis_world: Tensor, rot_wr: Tensor, seconds: Tensor,
                  valid: Tensor, rates: bool = True) -> Tensor:
    """The set's slots seen from the body, ``[n, K, 9]`` (``[n, K, 3]`` without ``rates``).

    Per slot: its position in the root frame (3) and the forward AND backward
    one-sided rates of its world position rotated into the body frame (6) — the
    limbs' twin of :func:`one_sided_root_rates`, so a rigid re-definition of the
    world leaves every channel untouched. ``seconds`` / ``valid`` are ``[n_clips, T]``.
    """
    n_clips, seq_len = seconds.shape
    n_frames, n_slots = points_world.shape[:2]
    clip = points_world.view(n_clips, seq_len, n_slots, 3)
    parts = [points_world - pelvis_world[:, None]]
    if rates:
        parts += [forward_difference(clip, seconds, valid).reshape(n_frames, n_slots, 3),
                  backward_difference(clip, seconds, valid).reshape(n_frames, n_slots, 3)]
    world = torch.stack(parts, dim=2)                                         # [n, K, k, 3]
    return torch.einsum("bji,bkmj->bkmi", rot_wr, world).reshape(n_frames, n_slots, -1)


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
    :param contact_set: name of the contact set (:mod:`model.contact_frames`) whose
        K slots the contact / force outputs, the limb tokens and the slot points are
        indexed by.
    :param num_contact_tokens: contact tokens fed as features (0 = none).
    :param dim: transformer width.
    :param num_layers: RoPE blocks.
    :param num_heads: attention heads.
    :param mlp_ratio: FFN expansion.
    :param dropout: dropout inside attention / FFN.
    :param window: attention half-width per layer, seconds (``None`` = the whole clip).
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
    :param camera_axes: ... plus the camera's down and viewing axes in the body
        frame (needs ``camera_context``; the ``gravity`` output's prior).
    :param gravity_input: known-gravity input ``{enabled, p_given,
        measured_only, eval_given}`` (needs the ``gravity`` output): the scene's
        gravity is given to a clip as four token channels, and that clip's
        ``gravity`` output IS the given vector.
    :param residual_feedback: feed each layer's RNEA root-wrench residual back
        (``iterative`` with the contact / force / gravity outputs; needs
        ``smplx_model_path`` for the dynamics body).
    :param frame_mask_p: training-time probability of replacing a frame's input
        token by the learned mask embedding (0 = off).
    :param head_grad_scale: factor on the gradient the non-pose heads (and
        their fed-back values) send into the trunk: 1 = fully shared, 0 = the
        heads are probes of a trunk only the pose losses shape.
    :param smplx_model_path: BetterHuman SMPL-X model file of the dynamics body.
    :param pose_token: feed the frozen pose token (projected).
    :param pose_token_dim: pose-token projection width.
    :param contact_token_dim: per-contact-token projection width.
    :param force_frame: frame the ``force`` head's three numbers per limb are
        read in — ``body`` (the input body frame) or ``gravity`` (a
        gravity-aligned, body-headed frame; needs the ``gravity`` output).
    :param limb_tokens: run ``1 + K`` tokens per frame through an alternating
        transformer — the body token (without the decoder contact tokens) plus
        one token per contact slot carrying that slot's geometry and its decoder
        contact token; the ``contact`` and ``force`` heads then read the limb
        tokens instead of the body one.
    :param force_tokens: K FORCE limb tokens (besides the contact ones when
        ``limb_tokens``: ``1 + 2K`` slots; alone: ``1 + K``): each carries its
        slot's geometry and its decoder force token (``num_force_tokens``); the
        ``force`` head reads them, the ``contact`` head the contact tokens, and
        each set feeds its own head's output back. Needs the ``force`` output.
    :param num_force_tokens: decoder force tokens fed to the force limb tokens
        (0 or K; needs ``force_tokens``).
    :param per_frame: NO information between frames — the token carries no
        root velocity / dt channels and the limb tokens no rates, the attention
        keeps every query on its own frame (``window`` is ignored), and the
        betas and the gravity vote are per frame instead of per clip. Needs
        ``iterative`` off, the velocity / ``raw_minus_mean`` token channels off
        and no input smoothing.
    """

    def __init__(
        self,
        decoder_dim: int,
        outputs: Sequence[str],
        num_contact_tokens: int,
        contact_set_name: str = "kindyn6",
        dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        window: Optional[float] = 0.5,
        time_scale: float = 25.0,
        root_smooth_sec: float = 0.0,
        pose_smooth_sec: float = 0.0,
        learn_smoothing: bool = False,
        token: Optional[dict] = None,
        iterative: bool = False,
        feedback_delta: bool = False,
        camera_context: bool = False,
        camera_axes: bool = False,
        gravity_input: Optional[dict] = None,
        residual_feedback: bool = False,
        frame_mask_p: float = 0.0,
        head_grad_scale: float = 1.0,
        smplx_model_path: Optional[str] = None,
        pose_token: bool = True,
        pose_token_dim: int = 256,
        contact_token_dim: int = 64,
        force_frame: str = "body",
        limb_tokens: bool = False,
        force_tokens: bool = False,
        num_force_tokens: int = 0,
        per_frame: bool = False,
    ):
        super().__init__()
        outputs = [str(o) for o in outputs]
        if not outputs or any(o not in OUTPUTS for o in outputs) or len(set(outputs)) != len(outputs):
            raise ValueError(f"outputs must be a non-empty subset of {OUTPUTS}; got {outputs}")
        self.outputs = tuple(o for o in OUTPUTS if o in outputs)
        self.slots = contact_set(str(contact_set_name))
        self.num_slots = self.slots.count
        self.num_contact_tokens = int(num_contact_tokens)
        self.register_buffer("slot_parent22",
                             torch.tensor(self.slots.parent_joint22, dtype=torch.long),
                             persistent=False)
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
        self.camera_axes = bool(camera_axes)
        if self.camera_axes and not self.camera_context:
            raise ValueError("camera_axes extends the camera context: it needs camera_context")
        self.per_frame = bool(per_frame)
        if self.per_frame:
            if self.iterative or self.token_one_sided_velocity or self.token_joint_velocity \
                    or self.token_raw_minus_mean:
                raise ValueError(
                    "per_frame refines every frame on its own: it needs iterative off and the "
                    "one_sided_velocity / joint_velocity / raw_minus_mean token channels off")
            if float(root_smooth_sec) > 0.0 or float(pose_smooth_sec) > 0.0:
                raise ValueError("per_frame needs root_smooth_sec and pose_smooth_sec 0")
            window = _PER_FRAME_WINDOW
        if "gravity" in self.outputs and not self.camera_axes:
            raise ValueError("the gravity output corrects the camera's down axis: it needs camera_axes")
        gravity_input = {"enabled": False, "p_given": 0.5, "measured_only": True,
                         "eval_given": True, **(gravity_input or {})}
        self.gravity_input = bool(gravity_input["enabled"])
        self.gravity_p_given = float(gravity_input["p_given"])
        self.gravity_measured_only = bool(gravity_input["measured_only"])
        self.gravity_eval_given = bool(gravity_input["eval_given"])
        if self.gravity_input:
            if "gravity" not in self.outputs:
                raise ValueError("gravity_input replaces the gravity estimate of the clips it is "
                                 "given on: list 'gravity' in outputs")
            if not 0.0 <= self.gravity_p_given <= 1.0:
                raise ValueError("gravity_input p_given must be in [0, 1]")
        self.residual_feedback = bool(residual_feedback)
        if self.residual_feedback:
            missing = [o for o in ("contact", "force", "gravity") if o not in self.outputs]
            if not self.iterative or missing:
                raise ValueError(
                    "residual_feedback runs the RNEA on every layer's body, gated forces and "
                    f"gravity: it needs iterative and the contact / force / gravity outputs (missing {missing})")
            if smplx_model_path is None:
                raise ValueError("residual_feedback needs smplx_model_path for the dynamics body")
        self.smplx_model_path = None if smplx_model_path is None else str(smplx_model_path)
        self._wrenches: dict = {}
        self.frame_mask_p = float(frame_mask_p)
        if not 0.0 <= self.frame_mask_p < 1.0:
            raise ValueError("frame_mask_p must be in [0, 1)")
        self.head_grad_scale = float(head_grad_scale)
        if not 0.0 <= self.head_grad_scale <= 1.0:
            raise ValueError("head_grad_scale must be in [0, 1]")
        self.time_scale = float(time_scale)
        self.force_frame = str(force_frame)
        if self.force_frame not in FORCE_FRAMES:
            raise ValueError(f"force_frame must be one of {FORCE_FRAMES}; got {force_frame!r}")
        if self.force_frame == "gravity" and "gravity" not in self.outputs:
            raise ValueError("force_frame 'gravity' aligns the force frame with the clip's "
                             "gravity estimate: list 'gravity' in outputs")
        self.limb_tokens = bool(limb_tokens)
        if self.limb_tokens and self.num_contact_tokens not in (0, self.num_slots):
            raise ValueError(
                f"limb_tokens gives each of the {self.num_slots} {self.slots.name} slots its "
                f"own decoder contact token; this build has {self.num_contact_tokens}")
        self.force_tokens = bool(force_tokens)
        self.num_force_tokens = int(num_force_tokens)
        if self.force_tokens and "force" not in self.outputs:
            raise ValueError("force_tokens adds force limb tokens: needs the force output")
        if self.limb_tokens and "contact" not in self.outputs:
            raise ValueError("limb_tokens adds contact limb tokens: needs the contact output")
        if self.num_force_tokens not in (0, self.num_slots) or (
                self.num_force_tokens and not self.force_tokens):
            raise ValueError(
                f"decoder force tokens feed the force limb tokens (force_tokens) one per "
                f"{self.slots.name} slot; this build has {self.num_force_tokens} and "
                f"force_tokens {self.force_tokens}")

        self.proj_pose_token = nn.Linear(decoder_dim, pose_token_dim) if pose_token else None
        self.proj_contact_tokens = (nn.Linear(decoder_dim, contact_token_dim)
                                    if self.num_contact_tokens > 0 else None)
        # The contact tokens ride the body token, or (limb_tokens) their own limb token.
        token_dim = (pose_token_dim if pose_token else 0)
        token_dim += 0 if self.limb_tokens else self.num_contact_tokens * contact_token_dim
        # Two LayerNorms: the geometry numbers and the projected token channels are
        # normalised separately, so neither group's scale rides on the other's width.
        geometry_dim = _GEOMETRY_DIM - (7 if self.per_frame else 0)     # no root rates, no dt
        geometry_dim += _CAMERA_DIM if self.camera_context else 0
        geometry_dim += _CAMERA_AXES_DIM if self.camera_axes else 0
        geometry_dim += 6 if self.token_one_sided_velocity else 0   # two one-sided rates, not one central
        geometry_dim += 6 * (NUM_BODY_JOINTS - 1) if self.token_joint_velocity else 0
        geometry_dim += 6 * (NUM_BODY_JOINTS - 1) if self.token_local_rotations else 0
        geometry_dim += 3 if self.token_gravity else 0
        geometry_dim += 3 + 3 * (NUM_BODY_JOINTS - 1) if self.token_raw_minus_mean else 0
        geometry_dim += _GRAVITY_INPUT_DIM if self.gravity_input else 0   # the last channels
        self.geometry_norm = nn.LayerNorm(geometry_dim)
        self.token_norm = nn.LayerNorm(token_dim) if token_dim > 0 else None
        self.input_proj = nn.Linear(geometry_dim + token_dim, dim)
        #: Any limb tokens at all (contact and / or force): alternating attention, limb geometry.
        self.any_limb = self.limb_tokens or self.force_tokens
        self.limb_geometry_norm = self.limb_token_norm = self.limb_input_proj = None
        self.limb_output_norm = None
        self.proj_force_tokens = self.force_limb_token_norm = self.force_limb_input_proj = None
        if self.any_limb:
            limb_geometry_dim = (3 if self.per_frame else _LIMB_GEOMETRY_DIM) + (
                _LIMB_CAMERA_DIM if self.camera_context else 0)
            self.limb_geometry_norm = nn.LayerNorm(limb_geometry_dim)
            self.limb_output_norm = nn.LayerNorm(dim)
        if self.limb_tokens:
            limb_token_dim = contact_token_dim if self.num_contact_tokens > 0 else 0
            self.limb_token_norm = nn.LayerNorm(limb_token_dim) if limb_token_dim > 0 else None
            # One projection shared by the six limbs: the transformer's slot embedding is
            # what tells them apart.
            self.limb_input_proj = nn.Linear(limb_geometry_dim + limb_token_dim, dim)
        if self.force_tokens:
            force_token_dim = contact_token_dim if self.num_force_tokens > 0 else 0
            if force_token_dim > 0:
                self.proj_force_tokens = nn.Linear(decoder_dim, contact_token_dim)
                self.force_limb_token_norm = nn.LayerNorm(force_token_dim)
            self.force_limb_input_proj = nn.Linear(limb_geometry_dim + force_token_dim, dim)
        transformer_slots = (1 + (self.num_slots if self.limb_tokens else 0)
                             + (self.num_slots if self.force_tokens else 0))
        self.temporal = CrossModalRopeModule(
            dim=dim, num_slots=transformer_slots,
            num_layers=num_layers, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout, window=window, time_scale=time_scale,
            alternating=self.any_limb)
        self.output_norm = nn.LayerNorm(dim)
        self.mask_token = nn.Parameter(torch.zeros(dim)) if self.frame_mask_p > 0.0 else None
        self.feedback_norm = self.feedback_proj = None
        self.limb_feedback_norm = self.limb_feedback_proj = None
        self.force_limb_feedback_norm = self.force_limb_feedback_proj = None
        if self.iterative:
            # Zero-initialised, like the heads: at init the extra path contributes nothing.
            feedback_dim = _FEEDBACK_DIM + (_FEEDBACK_DELTA_DIM if self.feedback_delta else 0)
            feedback_dim += self.num_slots if "contact" in self.outputs else 0
            feedback_dim += 3 if "gravity" in self.outputs else 0
            feedback_dim += _RESIDUAL_DIM if self.residual_feedback else 0
            self.feedback_norm = nn.LayerNorm(feedback_dim)
            self.feedback_proj = nn.Linear(feedback_dim, dim)
            nn.init.zeros_(self.feedback_proj.weight)
            nn.init.zeros_(self.feedback_proj.bias)
            if self.limb_tokens:
                # The contact limb tokens get the contact probability back, and the force too
                # when the force head reads them (no force tokens of their own).
                limb_feedback_dim = _LIMB_GEOMETRY_DIM + 1
                limb_feedback_dim += 3 if "force" in self.outputs and not self.force_tokens else 0
                self.limb_feedback_norm = nn.LayerNorm(limb_feedback_dim)
                self.limb_feedback_proj = nn.Linear(limb_feedback_dim, dim)
                nn.init.zeros_(self.limb_feedback_proj.weight)
                nn.init.zeros_(self.limb_feedback_proj.bias)
            if self.force_tokens:
                self.force_limb_feedback_norm = nn.LayerNorm(_LIMB_GEOMETRY_DIM + 3)
                self.force_limb_feedback_proj = nn.Linear(_LIMB_GEOMETRY_DIM + 3, dim)
                nn.init.zeros_(self.force_limb_feedback_proj.weight)
                nn.init.zeros_(self.force_limb_feedback_proj.bias)
        sizes = {"pose": 6 * NUM_BODY_JOINTS + 3, "contact": self.num_slots,
                 "motion": 6 * NUM_BODY_JOINTS + 6, "force": 3 * self.num_slots, "gravity": 3}
        if self.limb_tokens:
            sizes["contact"] = 1            # one head per limb TOKEN, shared by all of them
        if self.any_limb:
            sizes["force"] = 3              # ... the force head reads limb tokens either way
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
                  body_rot_in: Tensor, seconds: Tensor, valid: Tensor,
                  extra: Sequence[Tensor] = ()) -> Tensor:
        """The corrected trajectory's own features, projected into the residual stream.

        The channels the input token carries (root-frame joint positions, the one-sided
        root and per-joint rates, all body-relative) recomputed on the trajectory as the
        layers so far left it, so the next layer reads ITS rates and not the raw ones.
        With ``feedback_delta`` the CUMULATIVE correction is appended: the root shift in
        the input body frame and the composed root / joint rotation deltas as 6D — rates
        are blind to the slow drift the layers have already accumulated, and the
        correction measured against the un-refined trajectory is frame-independent too.
        ``extra`` holds the layer's head feedback (contact probabilities, body-frame
        gravity, root-wrench residual), each ``[B, k]`` and body-relative.
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
        feats += list(extra)
        return self.feedback_proj(self.feedback_norm(torch.cat(feats, dim=-1)))

    def _limb_features(self, points_world: Tensor, pelvis_world: Tensor, rot_wr: Tensor,
                       points_cam: Optional[Tensor], pelvis_depth: Optional[Tensor],
                       batch: dict, seconds: Tensor, valid: Tensor) -> Tensor:
        """The limbs' input geometry ``[B, K, k]``, LayerNormed.

        :func:`limb_geometry` (the slot's root-frame position and its one-sided
        body-frame rates), under ``camera_context`` the slot's crop-space position
        and its log depth ratio to the pelvis — camera-relative, like the body token's
        camera context, never world-relative.
        """
        feats = [limb_geometry(points_world, pelvis_world, rot_wr, seconds, valid,
                               rates=not self.per_frame)]
        if self.camera_context:
            _, crop = project_to_crop(points_cam, batch["cam_int"].float(),
                                      batch["affine_trans"].float(), batch["img_size"].float())
            log_ratio = (torch.log(points_cam[..., 2].clamp(min=1e-3))
                         - torch.log(pelvis_depth.clamp(min=1e-3))[:, None])
            feats += [crop, log_ratio[..., None]]
        return self.limb_geometry_norm(torch.cat(feats, dim=-1))

    @staticmethod
    def _limb_input(geometry: Tensor, token_proj: Optional[Tensor], norm, proj) -> Tensor:
        """The limb tokens ``[B, K, dim]``: the geometry plus the limbs' projected decoder
        tokens when the build has them. One projection for all the limbs (the
        transformer's slot embedding tells them apart)."""
        parts = [geometry] if norm is None else [geometry, norm(token_proj)]
        return proj(torch.cat(parts, dim=-1))

    @staticmethod
    def _limb_feedback(geometry: Tensor, extra: Sequence[Tensor], norm, proj) -> Tensor:
        """The corrected trajectory's per-limb features, projected into six limb tokens.

        The limbs' twin of :meth:`_feedback`: :func:`limb_geometry` recomputed on the
        trajectory as the layers so far left it, plus the layer's own per-limb head
        outputs (``extra``, each ``[B, K, k]`` and body-relative: the contact probability
        and / or the force). Returns ``[B, K, dim]``.
        """
        return proj(norm(torch.cat([geometry, *extra], dim=-1)))

    def _scale_grad(self, x: Tensor) -> Tensor:
        """``x`` unchanged in value, its gradient scaled by ``head_grad_scale``."""
        s = self.head_grad_scale
        if s >= 1.0:
            return x
        if s <= 0.0:
            return x.detach()
        return x * s + x.detach() * (1.0 - s)

    # ------------------------------------------------------------------ gravity / physics

    @staticmethod
    def pool_gravity(delta_b: Tensor, rot_wr: Tensor, down_cam_w: Tensor, n_clips: int,
                     seq_len: int, valid: Tensor, per_frame: bool = False) -> Tensor:
        """One unit down vector per clip, expanded to its frames ``[B, 3]`` (world).

        Per frame, the camera's down axis plus the head's body-frame correction
        ``delta_b`` (zero at init), normalised so every frame votes with a unit vector;
        then the masked clip mean, normalised again. A mean that (nearly) cancels falls
        back to the pooled camera axis, so the output is always a unit vector. The world
        enters only as the transport between the frames' body frames. With ``per_frame``
        there is no pooling: every frame keeps its own vote (same fallback).
        """
        n_frames = delta_b.shape[0]
        votes = down_cam_w + (rot_wr @ delta_b[..., None])[..., 0]
        if per_frame:
            degenerate = votes.norm(dim=-1, keepdim=True) < _GRAVITY_MIN_NORM
            votes = torch.where(degenerate, down_cam_w, votes)
            return votes / votes.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        w = valid.to(delta_b.dtype).reshape(n_clips, seq_len, 1)
        count = w.sum(dim=1).clamp(min=1.0)

        def clip_mean(vectors: Tensor) -> Tensor:
            unit = vectors / vectors.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            return (unit.view(n_clips, seq_len, 3) * w).sum(dim=1) / count

        prior = clip_mean(down_cam_w)
        pooled = clip_mean(votes)
        degenerate = pooled.norm(dim=-1, keepdim=True) < _GRAVITY_MIN_NORM
        pooled = torch.where(degenerate, prior, pooled)
        pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return pooled[:, None].expand(n_clips, seq_len, 3).reshape(n_frames, 3)

    def _force_frame(self, gravity_w: Optional[Tensor], rot_wr: Tensor) -> Tensor:
        """World-from-force-frame rotation ``[B, 3, 3]`` the ``force`` head predicts in.

        ``body``: the input body frame itself. ``gravity``: columns
        ``[e_side, e_up, e_fwd]`` with ``e_up = -g`` (the layer's own gravity estimate,
        DETACHED), ``e_fwd`` the input body's ``+z`` axis with its vertical component
        removed (its ``+y`` axis instead when that horizontal part nearly vanishes — the
        body lies along gravity) and ``e_side = e_up x e_fwd``. A function of the gravity
        and the body only, so a rigid re-definition of the world turns it with them and
        the force in the WORLD is unchanged.
        """
        if self.force_frame == "body":
            return rot_wr
        up = -gravity_w.detach()
        up = up / up.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        def horizontal(axis: Tensor) -> Tensor:
            return axis - (axis * up).sum(dim=-1, keepdim=True) * up

        forward = horizontal(rot_wr[:, :, 2])
        flat = forward.norm(dim=-1, keepdim=True) < _FORCE_FRAME_MIN_HORIZONTAL
        forward = torch.where(flat, horizontal(rot_wr[:, :, 1]), forward)
        forward = forward / forward.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return torch.stack([torch.cross(up, forward, dim=-1), up, forward], dim=-1)

    def _given_gravity(self, batch: dict, n_clips: int, seq_len: int,
                       device: torch.device) -> Tensor:
        """Clips that are HANDED the scene's gravity, ``[n_clips]`` bool (all false when off).

        Eligible clips are those whose corpus gravity is a measurement
        (``measured_only``) or all of them; training draws each eligible clip with
        probability ``p_given`` (torch's generator on the batch device, so the draws
        replay with the run's data seed), eval gives every eligible clip its gravity
        when ``eval_given`` and draws nothing.
        """
        if not self.gravity_input:
            return torch.zeros(n_clips, dtype=torch.bool, device=device)
        if self.gravity_measured_only:
            eligible = batch["gravity_measured"].to(device).view(n_clips, seq_len).all(dim=1)
        else:
            eligible = torch.ones(n_clips, dtype=torch.bool, device=device)
        if self.training:
            return eligible & (torch.rand(n_clips, device=device) < self.gravity_p_given)
        return eligible if self.gravity_eval_given else torch.zeros_like(eligible)

    def _wrench(self, device: torch.device):
        if device not in self._wrenches:
            from model.physics import RootWrench
            self._wrenches[device] = RootWrench(self.smplx_model_path, device)
        return self._wrenches[device]

    def _residual(self, pelvis_world: Tensor, rot_wr: Tensor, body_rot: Tensor,
                  betas_clip: Tensor, points_world: Tensor, forces_b: Tensor, frame_in: Tensor,
                  probs: Tensor, gravity_w: Tensor, seconds: Tensor, valid: Tensor) -> Tensor:
        """The layer's root-wrench residual ``[B, 6]`` (root frame; 0 outside the stencil).

        Body, betas, gravity and the contact gate are detached; the forces are live.
        """
        n_clips, seq_len = seconds.shape
        n_frames = pelvis_world.shape[0]
        if seq_len < _RESIDUAL_MIN_FRAMES:
            # No interior row exists; a graph-connected zero keeps the forces on the path.
            return forces_b.reshape(n_frames, -1)[:, :1].expand(n_frames, _RESIDUAL_DIM) * 0.0
        gated = forces_b * probs.detach()[..., None]
        forces_world = torch.einsum("bij,bkj->bki", frame_in.detach(), gated)
        res_f, res_t, rows, _ = self._wrench(pelvis_world.device).residual(
            pelvis_world.detach().view(n_clips, seq_len, 3),
            rot_wr.detach().view(n_clips, seq_len, 3, 3),
            body_rot.detach().view(n_clips, seq_len, NUM_BODY_JOINTS - 1, 3, 3),
            betas_clip.detach(), forces_world.view(n_clips, seq_len, self.num_slots, 3),
            points_world.detach().view(n_clips, seq_len, self.num_slots, 3), self.slot_parent22,
            gravity_w.detach().view(n_clips, seq_len, 3)[:, 0], seconds, valid)
        residual = torch.cat([res_f, res_t], dim=-1) * rows.to(res_f.dtype)[..., None]
        return residual.reshape(n_frames, _RESIDUAL_DIM)

    # ------------------------------------------------------------------ forward

    def forward(self, smplx_out: dict, tokens: Tensor, blocks: dict, batch: dict, body) -> dict:
        """Refine one batch of flattened clips.

        :param smplx_out: the per-frame :class:`~model.heads.SmplxHead` output.
        :param tokens: final decoder tokens ``[B, N, C]`` (pose token at 0).
        :param blocks: token-block bounds (``blocks["contact"]`` when present).
        :param batch: collated batch (``seq_len``, ``frame_pos_sec``,
            ``frame_valid``, ``cam_from_world``, ``cam_int``, ``affine_trans``,
            ``img_size``, ``bbox_center``, ``bbox_scale``; ``gravity_world`` with
            the gravity token channel, plus ``gravity_measured`` under
            ``gravity_input``).
        :param body: the head's BetterHuman SMPL-X body (22 or 52 joints).
        :returns: ``{"smplx", "contact", "force", "motion", "gravity"}`` — ``smplx`` in the
            SmplxHead layout plus ``pelvis_world`` / ``root_rot_world`` /
            ``joints_world``, the contact set's ``slot_points_world`` ``[B, K, 3]``,
            the per-layer ``joints_world_layers`` / ``slot_points_world_layers``
            (one entry unless ``iterative``, the last one IS the final tensor)
            and the smoothed, un-refined ``pelvis_world_in`` /
            ``root_rot_world_in`` / ``body_rot_in`` / ``joints_world_in`` /
            ``slot_points_world_in``;
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
        betas_mean = (betas if self.per_frame
                      else betas_clip[:, None].expand(n_clips, seq_len, -1).reshape(n_frames, -1))
        shaped = body.with_shape(betas=betas_mean)
        frame_ids = slot_frame_ids(self.slots, body)
        joints_world_in, points_world_in = world_points(
            shaped, p_w, rot_wr, body_rot, hand_rot, self.slots, frame_ids)

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
        geometry = ([joints_root.reshape(n_frames, -1), betas_mean] if self.per_frame
                    else [joints_root.reshape(n_frames, -1), vel_b, ang_b, dt, betas_mean])
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
            if self.camera_axes:
                # Rows of cam-from-root = the camera's axes in the body frame: its down (+y)
                # axis is the gravity prior, its viewing (+z) axis the tilt's other half.
                geometry += [root_rot_cam_s[:, 1, :], root_rot_cam_s[:, 2, :]]
        down_cam_w = rot_wc[:, :, 1]                              # camera +y axis in the world
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
        given = self._given_gravity(batch, n_clips, seq_len, device)
        given_frames = given[:, None].expand(n_clips, seq_len).reshape(n_frames)
        gravity_given = None
        if self.gravity_input:
            # The scene's gravity, handed over on the clips drawn for it: its direction in
            # the body frame (frame-independent, as the token.gravity channel) and the flag
            # that says so; both zero on a clip that has to guess.
            gravity_given = batch["gravity_world"].to(device, torch.float32)
            flag = given_frames.to(gravity_given.dtype)[:, None]
            geometry += [(rot_wr.transpose(1, 2) @ gravity_given[..., None])[..., 0] * flag, flag]
        geometry = torch.cat(geometry, dim=-1)
        feats = [self.geometry_norm(geometry)]
        token_feats = []
        if self.proj_pose_token is not None:
            token_feats.append(self.proj_pose_token(tokens[:, 0].float()))
        contact_proj = None
        if self.proj_contact_tokens is not None:
            lo, hi = blocks["contact"]
            if hi - lo != self.num_contact_tokens:
                raise AssertionError(
                    f"contact block has {hi - lo} tokens; refiner built for {self.num_contact_tokens}")
            contact_proj = self.proj_contact_tokens(tokens[:, lo:hi].float())   # [B, 6, C]
            if not self.limb_tokens:
                token_feats.append(contact_proj.reshape(n_frames, -1))
        if token_feats:
            feats.append(self.token_norm(torch.cat(token_feats, dim=-1)))
        x = self.input_proj(torch.cat(feats, dim=-1))
        if self.training and self.mask_token is not None:
            dropped = torch.rand(n_frames, device=device) < self.frame_mask_p
            x = torch.where(dropped[:, None], self.mask_token[None].to(x.dtype), x)
        stream = x[:, None]                                        # [B, 1, dim]
        if self.any_limb:
            points_cam_in = None
            if self.camera_context:
                points_cam_in = torch.einsum(
                    "bij,bkj->bki", rot_cw, points_world_in) + t_cw[:, None]
            geometry_in = self._limb_features(points_world_in, p_w, rot_wr, points_cam_in,
                                              pelvis_s[:, 2], batch, seconds, valid)
            slots = [stream]
            if self.limb_tokens:
                slots.append(self._limb_input(geometry_in, contact_proj, self.limb_token_norm,
                                              self.limb_input_proj))
            if self.force_tokens:
                force_proj = None
                if self.proj_force_tokens is not None:
                    lo, hi = blocks["force"]
                    if hi - lo != self.num_force_tokens:
                        raise AssertionError(
                            f"force block has {hi - lo} tokens; refiner built for {self.num_force_tokens}")
                    force_proj = self.proj_force_tokens(tokens[:, lo:hi].float())   # [B, 6, C]
                slots.append(self._limb_input(geometry_in, force_proj, self.force_limb_token_norm,
                                              self.force_limb_input_proj))
            stream = torch.cat(slots, dim=1)                        # [B, 7 | 13, dim]

        # 4. temporal transformer (one slot per frame, seven under `limb_tokens`), 5. the
        #    pose offset in the body / parent-local frames and 6. FK in the world. Under
        #    `iterative` the three steps interleave: every layer's delta lands on the
        #    trajectory the previous layers left and its rate features go back into the
        #    residual stream.
        n_layers = self.temporal.num_layers
        # Per layer: (pelvis, world-from-root, parent-local joints, world joints, slot points).
        states: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = []
        rot_wr2, body_rot2, p_w2 = rot_wr, body_rot, p_w
        head_names = [name for name in self.outputs if name != "pose"]
        per_layer: dict[str, list[Tensor]] = {name: [] for name in head_names}
        force_frames: list[Tensor] = []      # world-from-force-frame, one per layer

        def read_heads(hidden: Tensor, limb: Optional[Tensor]) -> dict[str, Tensor]:
            # The pooled gravity also depends on the body's rotation: scale that path too, or
            # the gravity loss would still reach the pose path through the lift.
            hidden, rot = self._scale_grad(hidden), self._scale_grad(rot_wr2)
            limb = None if limb is None else self._scale_grad(limb)

            def slots_of(name: str) -> Optional[Tensor]:
                """The limb tokens a head reads: contact ones first, force ones after."""
                if limb is None or name not in _LIMB_HEADS:
                    return None
                if name == "contact":
                    return limb[:, :self.num_slots] if self.limb_tokens else None
                if self.force_tokens:
                    return (limb[:, self.num_slots:] if self.limb_tokens
                            else limb[:, :self.num_slots])
                return limb[:, :self.num_slots]

            raw = {name: (self.heads[name](slots_of(name)).reshape(n_frames, -1)
                          if slots_of(name) is not None else self.heads[name](hidden))
                   for name in head_names}
            if "gravity" in raw:
                pooled = self.pool_gravity(raw["gravity"], rot, down_cam_w,
                                           n_clips, seq_len, valid, self.per_frame)
                # A clip that was given its gravity keeps it: the head's votes are dropped
                # there (and the supervision with them), so it never learns to copy its input.
                raw["gravity"] = pooled if gravity_given is None else torch.where(
                    given_frames[:, None], gravity_given, pooled)
            for name, value in raw.items():
                per_layer[name].append(value)
            if "force" in raw:
                # The layer's own gravity estimate heads its force frame (`body`: the input body).
                force_frames.append(self._force_frame(
                    raw["gravity"] if "gravity" in raw else None, rot_wr))
            return raw

        limb_hidden = None
        if self.iterative:
            tables = self.temporal.prepare(stream, seq_len, batch["frame_pos_sec"],
                                           batch["frame_valid"])
            h = stream
            for layer in range(n_layers):
                h = self.temporal.run_block(layer, h, tables)
                hidden = self.output_norm(h[:, 0])
                if self.any_limb:
                    limb_hidden = self.limb_output_norm(h[:, 1:])
                p_w2, rot_wr2, body_rot2 = self._apply_pose_delta(
                    self.heads["pose"](hidden), p_w2, rot_wr2, body_rot2)
                states.append((p_w2, rot_wr2, body_rot2, *world_points(
                    shaped, p_w2, rot_wr2, body_rot2, hand_rot, self.slots, frame_ids)))
                raw = read_heads(hidden, limb_hidden)
                if layer + 1 < n_layers:
                    fed = {k: self._scale_grad(v) for k, v in raw.items()}
                    extra = []
                    if "contact" in fed:
                        extra.append(torch.sigmoid(fed["contact"]))
                    if "gravity" in fed:
                        extra.append((rot_wr2.transpose(1, 2) @ fed["gravity"][..., None])[..., 0])
                    if self.residual_feedback:
                        extra.append(self._residual(
                            p_w2, rot_wr2, body_rot2, betas_clip, states[-1][4],
                            fed["force"].reshape(n_frames, self.num_slots, 3), force_frames[-1],
                            torch.sigmoid(fed["contact"]), fed["gravity"], seconds, valid))
                    update = self._feedback(*states[-1][:4], p_w, rot_wr, body_rot,
                                            seconds, valid, extra)[:, None]
                    if self.any_limb:
                        geometry = limb_geometry(states[-1][4], p_w2, rot_wr2, seconds, valid)
                        limb_extra, force_extra = [], []
                        if self.limb_tokens:
                            limb_extra.append(torch.sigmoid(fed["contact"])[..., None])
                        if "force" in fed:
                            # The token stream is body-relative: a force read in the
                            # gravity frame comes back into the input body frame.
                            force_b = fed["force"].reshape(n_frames, self.num_slots, 3)
                            if self.force_frame != "body":
                                force_b = torch.einsum(
                                    "bij,bkj->bki", rot_wr.transpose(1, 2) @ force_frames[-1],
                                    force_b)
                            (force_extra if self.force_tokens else limb_extra).append(force_b)
                        updates = [update]
                        if self.limb_tokens:
                            updates.append(self._limb_feedback(
                                geometry, limb_extra, self.limb_feedback_norm, self.limb_feedback_proj))
                        if self.force_tokens:
                            updates.append(self._limb_feedback(
                                geometry, force_extra, self.force_limb_feedback_norm,
                                self.force_limb_feedback_proj))
                        update = torch.cat(updates, dim=1)
                    h = h + update
        else:
            h = self.temporal(stream, seq_len, batch["frame_pos_sec"], batch["frame_valid"])
            hidden = self.output_norm(h[:, 0])
            if self.any_limb:
                limb_hidden = self.limb_output_norm(h[:, 1:])
            if "pose" in self.outputs:
                p_w2, rot_wr2, body_rot2 = self._apply_pose_delta(
                    self.heads["pose"](hidden), p_w2, rot_wr2, body_rot2)
            states.append((p_w2, rot_wr2, body_rot2, *world_points(
                shaped, p_w2, rot_wr2, body_rot2, hand_rot, self.slots, frame_ids)))
            raw = read_heads(hidden, limb_hidden)

        # 7. back into every camera — the intermediate layers too (deep supervision reads
        #    them; with one layer the lists are the final tensors and nothing extra runs).
        joints_world, slot_points_world = states[-1][3], states[-1][4]
        joints_cam2 = torch.einsum("bij,bkj->bki", rot_cw, joints_world) + t_cw[:, None]
        pelvis_cam2 = (rot_cw @ p_w2[..., None])[..., 0] + t_cw
        root_rot_cam2 = rot_cw @ rot_wr2
        kp2d_full, kp2d_crop = project_to_crop(
            joints_cam2, batch["cam_int"].float(), batch["affine_trans"].float(),
            batch["img_size"].float())
        root_6d, body_6d = rotmat_to_rot6d(root_rot_cam2), rotmat_to_rot6d(body_rot2)
        layers = [
            {"pelvis_world": p, "root_rot_world": r, "joints_world": j, "slot_points_world": s,
             "joints_cam": torch.einsum("bij,bkj->bki", rot_cw, j) + t_cw[:, None],
             "root_6d": rotmat_to_rot6d(rot_cw @ r), "body_6d": rotmat_to_rot6d(b)}
            for p, r, b, j, s in states[:-1]]
        layers.append({"pelvis_world": p_w2, "root_rot_world": rot_wr2,
                       "joints_world": joints_world, "slot_points_world": slot_points_world,
                       "joints_cam": joints_cam2, "root_6d": root_6d, "body_6d": body_6d})
        smplx = {
            "root_6d": root_6d, "body_6d": body_6d,
            "hand_6d": None if hand_rot is None else rotmat_to_rot6d(hand_rot),
            "root_rot": root_rot_cam2, "body_rot": body_rot2, "hand_rot": hand_rot,
            "betas": betas_mean, "cam": None, "ray": translation_to_ray(pelvis_cam2),
            "pelvis_cam": pelvis_cam2,
            "q_cam": smplx_q(pelvis_cam2, root_rot_cam2, body_rot2, hand_rot),
            "joints_cam": joints_cam2, "kp2d_full": kp2d_full, "kp2d_crop": kp2d_crop,
            "pelvis_world": p_w2, "root_rot_world": rot_wr2, "joints_world": joints_world,
            "slot_points_world": slot_points_world,
            "pelvis_world_in": p_w, "root_rot_world_in": rot_wr, "body_rot_in": body_rot,
            "joints_world_in": joints_world_in, "slot_points_world_in": points_world_in,
            **{f"{key}_layers": [layer[key] for layer in layers] for key in layers[0]},
        }
        contact = force = motion = gravity = None
        if "contact" in raw:
            contact = {"logits": raw["contact"], "probs": torch.sigmoid(raw["contact"]),
                       "logits_layers": per_layer["contact"]}
        if "force" in raw:
            # Forces live in the INPUT body frame, or (`force_frame: gravity`) in the final
            # layer's gravity-aligned frame; `frame` is world-from-that, which lets the loss
            # rotate the kindyn GT (given in the GT root frame) into it.
            layer_forces = [f.reshape(n_frames, self.num_slots, 3) for f in per_layer["force"]]
            if self.force_frame != "body":
                # Deep supervision compares the layers in ONE frame, and each layer's frame
                # follows its own gravity estimate.
                to_final = force_frames[-1].detach().transpose(1, 2)
                layer_forces = [torch.einsum("bij,bkj->bki", to_final @ frame.detach(), value)
                                for frame, value in zip(force_frames, layer_forces)]
            force = {"forces": raw["force"].reshape(n_frames, self.num_slots, 3),
                     "frame": force_frames[-1], "forces_layers": layer_forces}
        if "gravity" in raw:
            rot = self._scale_grad(rot_wr2)
            gravity = {"world": raw["gravity"],
                       "body": (rot.transpose(1, 2) @ raw["gravity"][..., None])[..., 0],
                       "prior_world": self.pool_gravity(torch.zeros_like(raw["gravity"]), rot_wr2,
                                                        down_cam_w, n_clips, seq_len, valid),
                       "given": given_frames, "world_layers": per_layer["gravity"]}
        if "motion" in raw:
            m = raw["motion"]
            k = 3 * NUM_BODY_JOINTS
            motion = {
                "vel": m[:, :k].reshape(n_frames, NUM_BODY_JOINTS, 3),
                "acc": m[:, k:2 * k].reshape(n_frames, NUM_BODY_JOINTS, 3),
                "ang_vel": m[:, 2 * k:2 * k + 3], "ang_acc": m[:, 2 * k + 3:],
                "frame": rot_wr,                                        # world-from-body
            }
        return {"smplx": smplx, "contact": contact, "force": force, "motion": motion,
                "gravity": gravity}


__all__ = ["TemporalRefiner", "OUTPUTS", "SMOOTHING_PARAM_NAMES", "world_points",
           "slot_frame_ids", "root_frame_joints", "one_sided_root_rates", "joint_rates",
           "limb_geometry",
           "gaussian_smooth", "smooth_rotations", "project_rotation", "time_derivative",
           "second_difference", "angular_velocity", "angular_acceleration", "local_dt",
           "stencil_valid", "neighbours", "local_mean", "forward_difference",
           "backward_difference", "forward_valid", "forward_angular_velocity",
           "backward_angular_velocity"]
