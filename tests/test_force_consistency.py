"""Unit tests of the RNEA force-consistency residual (CPU, 22-joint SMPL-X).

* a body at rest under gravity with no external force needs exactly one body
  weight at the root, along -gravity (the residual is expressed in the root's
  local frame — BetterRobot's free-flyer tangent convention);
* the same body supported by one body weight at a foot joint has a vanishing
  force residual (the torque residual is the lever's moment);
* the residual is invariant to a rigid re-definition of the world (gravity
  rotates with it);
* gradients reach the forces and the pose.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import roma
import torch
import yaml

from model.loss.force_consistency import ForceConsistencyLoss

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def loss():
    cfg = yaml.safe_load((REPO / "configs" / "base.yaml").read_text())
    cfg["force_consistency"]["enabled"] = True
    cfg["force_consistency"]["smooth_sec"] = 0.0
    return ForceConsistencyLoss(cfg, None, "cpu")


def still_clip(n: int = 2, t: int = 9):
    torch.manual_seed(0)
    root_rot = roma.random_rotmat(n)[:, None].expand(n, t, 3, 3).contiguous()
    body_rot = roma.rotvec_to_rotmat(0.2 * torch.randn(n, 21, 3))[:, None].expand(n, t, 21, 3, 3).contiguous()
    pelvis = torch.tensor([0.3, 1.0, 2.0]).expand(n, t, 3).contiguous()
    betas = 0.3 * torch.randn(n, 10)
    seconds = (torch.arange(t, dtype=torch.float32) / 25.0)[None].expand(n, t).contiguous()
    valid = torch.ones(n, t, dtype=torch.bool)
    down = torch.tensor([0.0, -1.0, 0.0]).expand(n, 3).contiguous()
    return pelvis, root_rot, body_rot, betas, seconds, valid, down


def test_rest_needs_one_body_weight(loss):
    pelvis, root_rot, body_rot, betas, seconds, valid, down = still_clip()
    forces = torch.zeros(*seconds.shape, 6, 3)
    res_f, res_t, rows = loss.residual(pelvis, root_rot, body_rot, betas, forces, down, seconds, valid)
    assert rows[:, 2:-2].all() and not rows[:, :2].any()
    up_local = torch.einsum("ntji,nj->nti", root_rot, -down)         # -g in the root frame
    assert torch.allclose(res_f[rows], up_local[rows], atol=2e-3)    # exactly 1 bw, along -g


def test_support_at_a_foot_balances_the_force(loss):
    pelvis, root_rot, body_rot, betas, seconds, valid, down = still_clip()
    forces = torch.zeros(*seconds.shape, 6, 3)
    forces[..., 2, :] = -down[:, None]                                # 1 bw up at the left big toe
    res_f, res_t, rows = loss.residual(pelvis, root_rot, body_rot, betas, forces, down, seconds, valid)
    assert res_f[rows].abs().max() < 2e-3
    assert res_t[rows].norm(dim=-1).mean() > 0.05                    # the toe's lever about the pelvis


def test_world_frame_independence(loss):
    pelvis, root_rot, body_rot, betas, seconds, valid, down = still_clip()
    torch.manual_seed(3)
    forces = 0.5 * torch.randn(*seconds.shape, 6, 3)
    pelvis = pelvis + torch.cumsum(0.01 * torch.randn_like(pelvis), dim=1)
    res_f, res_t, rows = loss.residual(pelvis, root_rot, body_rot, betas, forces, down, seconds, valid)
    rot0 = roma.random_rotmat(1)[0]
    t0 = torch.tensor([2.0, -1.0, 0.5])
    moved = loss.residual((pelvis @ rot0.T) + t0, rot0 @ root_rot, body_rot, betas,
                          forces @ rot0.T, down @ rot0.T, seconds, valid)
    # The residual is expressed in the root's local frame (RNEA free-flyer convention):
    # identical before and after the rigid world change.
    assert torch.allclose(res_f[rows], moved[0][rows], atol=1e-3)
    assert torch.allclose(res_t[rows], moved[1][rows], atol=1e-3)


def test_gradients_reach_forces_and_pose(loss):
    pelvis, root_rot, body_rot, betas, seconds, valid, down = still_clip()
    forces = (0.3 * torch.randn(*seconds.shape, 6, 3)).requires_grad_(True)
    pelvis = pelvis.clone().requires_grad_(True)
    res_f, res_t, rows = loss.residual(pelvis, root_rot, body_rot, betas, forces, down, seconds, valid)
    (res_f[rows].square().sum() + res_t[rows].square().sum()).backward()
    assert forces.grad is not None and torch.isfinite(forces.grad).all() and forces.grad.abs().sum() > 0
    assert pelvis.grad is not None and torch.isfinite(pelvis.grad).all() and pelvis.grad.abs().sum() > 0
