"""Physics loss: static equilibrium, gradients, stencil/eligibility, gating,
dimensionlessness, and frame equivariance.

All CPU, using the real MHR body at LOD1 with tiny synthetic "predictions" (no
checkpoint, no GPU). The adapter (MHR body load) is shared across tests via a
module fixture and injected into each :class:`PhysicsLoss`.
"""
from __future__ import annotations

import copy
import dataclasses

import pytest
import torch

import better_robot as br
from better_robot.lie import so3

from contact.losses import ddp_global_mean_term
from contact.physics.adapter import MHRAdapter, _resolve_model_path
from contact.physics.loss import PhysicsLoss, _residual_frame_indices

try:
    _MHR_PATH = _resolve_model_path(None, 1)
    _HAS_MHR = True
except FileNotFoundError:
    _MHR_PATH, _HAS_MHR = None, False

pytestmark = pytest.mark.skipif(not _HAS_MHR, reason="MHR archive unavailable")


@pytest.fixture(scope="module")
def adapter() -> MHRAdapter:
    return MHRAdapter(model_path=_MHR_PATH, lod=1, device="cpu")


# --------------------------------------------------------------------- helpers

def _cfg(
    frame: str = "local_world_aligned",
    kernel=None,
    use_warp: bool = False,
    max_cam_jump_m=None,
    **loss_overrides,
) -> dict:
    """Minimal resolved-config slice the loss reads (``physics`` + head frame).

    ``residual_robust`` may be passed as a ``loss_overrides`` mapping; ``.get`` in
    ``PhysicsLoss`` defaults it to the square path when absent (backwards-identical).
    """
    loss = {
        "residual": 1.0, "force_noncontact": 1.0, "force_at_contact": 0.1,
        "contact_min_bw": 0.05, "force_smooth": 0.1, "force_l2": 0.01,
        "torque_l2": 0.01, "torque_smooth": 0.0,
    }
    loss.update(loss_overrides)
    return {
        "physics": {
            "enabled": True, "model_path": None, "lod": 1, "gravity": 9.81,
            "use_warp": use_warp, "max_cam_jump_m": max_cam_jump_m,
            "min_frames": 5, "smoothing_kernel": [1.0] if kernel is None else kernel,
            "loss": loss,
        },
        "model": {"force_head": {"frame": frame}},
    }


def _mhr_out(adapter: MHRAdapter, n_clips: int, seq_len: int, *, moving: bool, seed: int = 0):
    """Synthetic ``out["mhr"]`` — neutral static, or a per-frame perturbed clip."""
    batch = n_clips * seq_len
    gen = torch.Generator().manual_seed(seed)
    if moving:
        robot = adapter.body.robot
        q_rand = robot.integrate(
            robot.q_neutral.expand(batch, -1),
            torch.randn(batch, robot.nv, generator=gen) * 0.05)
        params = adapter.body.to_classic(q_rand).model_parameters.clone()
        params[:, :3] = 0.0
    else:
        params = torch.zeros(batch, 204)
    return {
        "mhr_model_params": params,
        "shape": torch.zeros(batch, 45),
        "pred_cam_t": torch.tensor([0.0, 0.0, 3.0]).expand(batch, 3).contiguous(),
    }


def _batch(n_clips: int, seq_len: int, *, cam=None, gravity_world=(0.0, 1.0, 0.0),
           dt: float = 0.05) -> dict:
    batch = n_clips * seq_len
    if cam is None:
        cam = torch.eye(4).expand(batch, 4, 4).contiguous()
    return {
        "seq_len": seq_len,
        "frame_pos_sec": (torch.arange(batch) % seq_len).float() * dt,
        "frame_valid": torch.ones(batch, dtype=torch.bool),
        "cam_from_world": cam,
        "gravity_world": torch.tensor(gravity_world).expand(batch, 3).contiguous(),
        "cam_valid": torch.ones(batch, dtype=torch.bool),
    }


def _out(mhr, forces, probs) -> dict:
    return {"mhr": mhr, "force": {"joint_forces": forces}, "contact": {"joint_probs": probs}}


# ------------------------------------------------------------------- 1. static

def test_static_equilibrium_and_support(adapter):
    """Static neutral clip, zero forces -> residual force == 1 body weight; the
    analytic two-foot support (in the joint-local head frame) drops it >= 10x."""
    n_clips, seq_len = 1, 5
    batch = _batch(n_clips, seq_len)
    mhr = _mhr_out(adapter, n_clips, seq_len, moving=False)
    forces0 = torch.zeros(n_clips * seq_len, 4, 3)
    probs = torch.zeros(n_clips * seq_len, 4)

    loss = PhysicsLoss(_cfg(frame="local"), device="cpu", adapter=adapter)
    _, parts0 = loss(_out(mhr, forces0, probs), batch)
    assert parts0["n_residual_frames"] == 1
    assert abs(parts0["residual_force"] - 1.0) < 1e-3

    # Analytic support: -m*g_world/2 per foot, rotated into each foot-joint frame,
    # expressed dimensionlessly (units of body weight) as the head predicts.
    body, q = adapter.q_from_mhr_out(mhr, batch["cam_from_world"], n_clips, seq_len)
    fk = br.forward_kinematics(body.robot, q.index_select(1, torch.tensor([2])))
    foot_ids = adapter.extremity_joint_ids[2:]
    world_to_local = so3.to_matrix(fk.joint_pose_world.index_select(-2, foot_ids)[..., 3:]) \
        .transpose(-1, -2)                                           # (1, 1, 2, 3, 3)
    gravity_world = torch.tensor([0.0, 1.0, 0.0])
    support = (world_to_local @ (-gravity_world / 2).view(1, 1, 1, 3, 1)).squeeze(-1)

    forces_support = torch.zeros(n_clips * seq_len, 4, 3)
    forces_support[:, 2] = support[0, 0, 0]
    forces_support[:, 3] = support[0, 0, 1]
    _, parts1 = loss(_out(mhr, forces_support, probs), batch)
    assert parts1["residual_force"] < parts0["residual_force"] / 10.0


# ---------------------------------------------------------------- 2. gradients

def test_gradients_and_detachment(adapter):
    """backward gives finite nonzero grad on the force leaf, None on the (detached)
    contact-probs leaf, and None on the MHR-params leaf (no grad reaches q)."""
    n_clips, seq_len = 2, 7
    batch = _batch(n_clips, seq_len)
    mhr = _mhr_out(adapter, n_clips, seq_len, moving=True)
    mhr["mhr_model_params"] = mhr["mhr_model_params"].requires_grad_(True)
    forces = (torch.randn(n_clips * seq_len, 4, 3) * 0.1).requires_grad_(True)
    probs = torch.full((n_clips * seq_len, 4), 0.5).requires_grad_(True)

    loss = PhysicsLoss(_cfg(), device="cpu", adapter=adapter)
    total, _ = loss(_out(mhr, forces, probs), batch)
    total.backward()

    assert forces.grad is not None and torch.isfinite(forces.grad).all()
    assert float(forces.grad.abs().sum()) > 0.0
    assert probs.grad is None            # D8: probs detached before use
    assert mhr["mhr_model_params"].grad is None   # adapter detaches -> no grad to q


def test_warp_selector_is_forwarded_to_fk_and_rnea(adapter, monkeypatch):
    """The physics selector reaches both BetterRobot whole-pass APIs."""
    selectors = {"fk": [], "rnea": []}
    original_fk = br.forward_kinematics
    original_rnea = br.rnea

    def forward_kinematics(*args, **kwargs):
        if "use_warp" in kwargs:
            selectors["fk"].append(kwargs.pop("use_warp"))
        return original_fk(*args, **kwargs)

    def rnea(*args, **kwargs):
        selectors["rnea"].append(kwargs.pop("use_warp"))
        return original_rnea(*args, **kwargs)

    monkeypatch.setattr(br, "forward_kinematics", forward_kinematics)
    monkeypatch.setattr(br, "rnea", rnea)

    seq_len = 5
    loss = PhysicsLoss(_cfg(use_warp=True), device="cpu", adapter=adapter)
    loss(
        _out(
            _mhr_out(adapter, 1, seq_len, moving=False),
            torch.zeros(seq_len, 4, 3),
            torch.zeros(seq_len, 4),
        ),
        _batch(1, seq_len),
    )

    assert selectors == {"fk": [True], "rnea": [True]}


# --------------------------------------------------- 3. stencil / eligibility / DDP

def test_residual_frame_scheme():
    """Documented stencil: {t : 2+r <= t <= T-3-r}."""
    assert _residual_frame_indices(5, 0) == [2]
    assert _residual_frame_indices(7, 0) == [2, 3, 4]
    assert _residual_frame_indices(8, 1) == [3, 4]
    assert _residual_frame_indices(5, 1) == []       # default kernel needs T>=7


def test_ineligible_batches_are_graph_connected_zero(adapter):
    """Still images (T=1) and clips with an invalid frame give zero mass, but the
    returned loss is still graph-connected to the force tensor (DDP guarantee)."""
    loss = PhysicsLoss(_cfg(), device="cpu", adapter=adapter)

    # T=1 still-image batch: whole batch ineligible.
    forces = torch.zeros(3, 4, 3, requires_grad=True)
    out = _out(_mhr_out(adapter, 3, 1, moving=False), forces, torch.zeros(3, 4))
    total, parts = loss(out, _batch(3, 1))
    assert parts["n_eligible_clips"] == 0
    assert all(term["weight_mass"] == 0.0 for term in parts["terms"].values())
    total.backward()
    assert forces.grad is not None and float(forces.grad.abs().sum()) == 0.0

    # T=5 clip with one invalid frame -> ineligible, zero mass, still connected.
    forces = torch.zeros(5, 4, 3, requires_grad=True)
    batch = _batch(1, 5)
    batch["frame_valid"][2] = False
    out = _out(_mhr_out(adapter, 1, 5, moving=False), forces, torch.zeros(5, 4))
    total, parts = loss(out, batch)
    assert parts["n_eligible_clips"] == 0
    total.backward()
    assert forces.grad is not None and float(forces.grad.abs().sum()) == 0.0


def test_missing_camera_on_eligible_clip_raises(adapter):
    """An otherwise-eligible video clip with cam_valid=False raises (step-02 export
    missing must never become a silent no-op)."""
    loss = PhysicsLoss(_cfg(), device="cpu", adapter=adapter)
    batch = _batch(1, 5)
    batch["cam_valid"][3] = False
    out = _out(_mhr_out(adapter, 1, 5, moving=False),
               torch.zeros(5, 4, 3), torch.zeros(5, 4))
    with pytest.raises(ValueError, match="cam_valid=False"):
        loss(out, batch)


# ------------------------------------------------------------------- 4. gating

def test_gating(adapter):
    """p=0 -> force_noncontact strictly increasing in ||f||; p=1,f=0 ->
    force_at_contact>0; p=1,||f||>f_min -> force_at_contact==0."""
    # Only the gating terms active (skips the adapter/RNEA path entirely).
    cfg = _cfg(residual=0.0, force_smooth=0.0, torque_l2=0.0, torque_smooth=0.0)
    loss = PhysicsLoss(cfg, device="cpu", adapter=adapter)
    n = 5
    mhr = _mhr_out(adapter, 1, n, moving=False)

    def gate(forces, probs):
        _, parts = loss(_out(mhr, forces, probs), _batch(1, n))
        return parts["terms"]

    small = gate(torch.full((n, 4, 3), 0.1), torch.zeros(n, 4))
    large = gate(torch.full((n, 4, 3), 0.5), torch.zeros(n, 4))
    assert large["force_noncontact"]["loss"] > small["force_noncontact"]["loss"] > 0.0

    at_zero = gate(torch.zeros(n, 4, 3), torch.ones(n, 4))
    assert at_zero["force_at_contact"]["loss"] > 0.0
    above = gate(torch.full((n, 4, 3), 1.0), torch.ones(n, 4))  # ||f||=sqrt(3)>f_min
    assert above["force_at_contact"]["loss"] == 0.0


# ---------------------------------------------------------- 5. dimensionlessness

def test_dimensionless_under_mass_doubling(adapter):
    """Doubling the body mass (2x inertias) leaves the normalized residual force
    invariant (guards D12: fext and the normalizer both scale with mass)."""
    n_clips, seq_len = 1, 7
    batch = _batch(n_clips, seq_len)
    mhr = _mhr_out(adapter, n_clips, seq_len, moving=True)
    forces = torch.randn(n_clips * seq_len, 4, 3) * 0.2
    probs = torch.zeros(n_clips * seq_len, 4)

    loss = PhysicsLoss(_cfg(), device="cpu", adapter=adapter)
    _, base = loss(_out(mhr, forces, probs), batch)

    heavy = copy.copy(adapter)
    original = adapter.q_from_mhr_out
    # Density doubling in the packed [mass, com(3), inertia(6)] layout: mass and
    # inertia scale by 2, com stays put (scaling com would move the COM).
    density_scale = torch.tensor([2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    def doubled(*args, **kwargs):
        body, q = original(*args, **kwargs)
        values = body.robot.values
        robot = dataclasses.replace(
            body.robot,
            values=dataclasses.replace(values, body_inertias=values.body_inertias * density_scale))
        return body._replace(values=body.values, robot=robot), q

    heavy.q_from_mhr_out = doubled
    loss_heavy = PhysicsLoss(_cfg(), device="cpu", adapter=heavy)
    _, scaled = loss_heavy(_out(mhr, forces, probs), batch)

    assert abs(scaled["residual_force"] - base["residual_force"]) < 1e-4 * base["residual_force"]


# ------------------------------------------------------------ 6. frame equivariance

def test_frame_equivariance(adapter):
    """Re-anchoring the world (cam_from_world @ G) with a consistently rotated
    gravity_world leaves every loss term unchanged (catches w2c-vs-c2w and
    flip-sign mistakes in the LWA conversion)."""
    n_clips, seq_len = 2, 7
    mhr = _mhr_out(adapter, n_clips, seq_len, moving=True, seed=3)
    forces = torch.randn(n_clips * seq_len, 4, 3, generator=torch.Generator().manual_seed(1)) * 0.15
    probs = torch.rand(n_clips * seq_len, 4, generator=torch.Generator().manual_seed(2))

    gen = torch.Generator().manual_seed(7)
    rot = so3.to_matrix(so3.exp(torch.randn(n_clips * seq_len, 3, generator=gen) * 0.4))
    trans = torch.randn(n_clips * seq_len, 3, generator=gen) * 0.3
    cam = torch.eye(4).expand(n_clips * seq_len, 4, 4).clone()
    cam[:, :3, :3] = rot
    cam[:, :3, 3] = trans
    batch = _batch(n_clips, seq_len, cam=cam)

    loss = PhysicsLoss(_cfg(), device="cpu", adapter=adapter)
    _, parts = loss(_out(mhr, forces, probs), batch)

    rg = so3.to_matrix(so3.exp(torch.tensor([0.3, -0.5, 0.7])))
    g_transform = torch.eye(4)
    g_transform[:3, :3] = rg
    g_transform[:3, 3] = torch.tensor([0.4, -0.2, 0.9])
    batch_g = copy.copy(batch)
    batch_g["cam_from_world"] = torch.matmul(cam, g_transform)
    batch_g["gravity_world"] = (rg.transpose(-1, -2) @ batch["gravity_world"].unsqueeze(-1)).squeeze(-1)
    _, parts_g = loss(_out(mhr, forces, probs), batch_g)

    for name in parts["terms"]:
        assert abs(parts_g["terms"][name]["loss"] - parts["terms"][name]["loss"]) < 1e-3, name
    assert abs(parts_g["residual_force"] - parts["residual_force"]) < 1e-3
    assert abs(parts_g["residual_torque"] - parts["residual_torque"]) < 1e-3


# ------------------------------------------------------- DDP exact-mean reduction

def test_ddp_physics_masses_reduce_to_global_mean(adapter):
    """Two simulated ranks with different physics eligibility (different per-term
    masses) reproduce the single-process global mass-weighted mean and gradient —
    the same exact-mean contract the contact loss uses, now for PhysicsLoss's
    ``(weighted_numerator, mass)`` terms folded through ``ddp_global_mean_term``
    (the trainer's ``_ddp_physics_loss`` path, step 07)."""
    loss = PhysicsLoss(_cfg(), device="cpu", adapter=adapter)
    seq_len = 7

    def rank_terms(n_clips: int, scale) -> dict:
        """One rank's PhysicsLoss terms; forces = ``scale * base`` (base fixed)."""
        batch = _batch(n_clips, seq_len)
        mhr = _mhr_out(adapter, n_clips, seq_len, moving=True, seed=n_clips)
        gen = torch.Generator().manual_seed(100 + n_clips)
        base = torch.randn(n_clips * seq_len, 4, 3, generator=gen) * 0.2
        probs = torch.full((n_clips * seq_len, 4), 0.3)
        _, parts = loss(_out(mhr, scale * base, probs), batch)
        return parts["terms"]

    # Rank 0 has 1 eligible clip, rank 1 has 2 -> per-term masses differ.
    t0 = rank_terms(1, torch.tensor(1.0))
    t1 = rank_terms(2, torch.tensor(1.0))
    common = set(t0) & set(t1)
    assert {"residual", "force_l2", "force_noncontact"} <= common

    # Value: DDP averages the per-rank terms; that must equal the global mean.
    for name in common:
        num0, mass0 = t0[name]["weighted_numerator_tensor"].detach(), t0[name]["weight_mass"]
        num1, mass1 = t1[name]["weighted_numerator_tensor"].detach(), t1[name]["weight_mass"]
        global_mass = torch.tensor(mass0 + mass1)
        assert mass0 != mass1                                  # genuinely different masses
        ddp_value = torch.stack([
            ddp_global_mean_term(num0, global_mass, world_size=2),
            ddp_global_mean_term(num1, global_mass, world_size=2),
        ]).mean()
        reference = (num0 + num1) / global_mass.clamp(min=1.0)
        assert ddp_value.item() == pytest.approx(reference.item(), rel=1e-6), name

    # Gradient: through a shared force scale, the DDP-averaged gradient equals the
    # single-process global-mean gradient (force_l2 is a clean quadratic in scale).
    scale = torch.tensor(0.7, requires_grad=True)
    r0 = rank_terms(1, scale)["force_l2"]
    r1 = rank_terms(2, scale)["force_l2"]
    global_mass = torch.tensor(r0["weight_mass"] + r1["weight_mass"])
    reference = (r0["weighted_numerator_tensor"]
                 + r1["weighted_numerator_tensor"]) / global_mass.clamp(min=1.0)
    reference_grad = torch.autograd.grad(reference, scale)[0]

    s0 = torch.tensor(0.7, requires_grad=True)
    s1 = torch.tensor(0.7, requires_grad=True)
    d0 = ddp_global_mean_term(
        rank_terms(1, s0)["force_l2"]["weighted_numerator_tensor"], global_mass, 2)
    d1 = ddp_global_mean_term(
        rank_terms(2, s1)["force_l2"]["weighted_numerator_tensor"], global_mass, 2)
    ddp_grad = torch.stack([
        torch.autograd.grad(d0, s0)[0], torch.autograd.grad(d1, s1)[0]]).mean()
    assert ddp_grad.item() == pytest.approx(reference_grad.item(), rel=1e-5)


# ------------------------------------------------------- robust residual (§1)

def test_pseudo_huber_limits_and_finite_gradient():
    """ρ_δ is quadratic (≈ ½x²) near 0, linear (δ|x|−δ²) far out, with a finite,
    zero-at-zero gradient everywhere — including x=0 (which the classic Huber lacks)."""
    from contact.physics.loss import _pseudo_huber

    delta = 1.0
    # Standard pseudo-Huber near-zero limit is ½x² (a quadratic, half the square
    # scale). Use a moderately small x — a tinier one triggers float32 catastrophic
    # cancellation in √(1+u)−1, which is a precision artefact, not a limit failure.
    small = torch.tensor([0.02, -0.03])
    assert torch.allclose(_pseudo_huber(small, delta), 0.5 * small ** 2, rtol=1e-2)
    large = torch.tensor([50.0, -80.0])
    linear = delta * large.abs() - delta * delta
    assert torch.allclose(_pseudo_huber(large, delta), linear, rtol=1e-2)
    # ρ ≤ ½x² ≤ x² everywhere (it never exceeds the square term it robustifies).
    x = torch.linspace(-6, 6, 121)
    assert (_pseudo_huber(x, delta) <= 0.5 * x ** 2 + 1e-6).all()

    x_leaf = torch.linspace(-4, 4, 33, requires_grad=True)
    _pseudo_huber(x_leaf, delta).sum().backward()
    assert torch.isfinite(x_leaf.grad).all()
    zero = torch.zeros(1, requires_grad=True)
    _pseudo_huber(zero, delta).sum().backward()
    assert torch.isfinite(zero.grad).all() and float(zero.grad.abs().sum()) == 0.0


def test_residual_robust_and_raw_residual(adapter):
    """``raw_residual`` is the physical residual under BOTH kinds and equals the
    ``residual`` term exactly under ``square``; ``pseudo_huber`` down-weights the
    objective (num strictly smaller) but leaves the raw headline unchanged; and
    ``residual_sat_frac`` tracks the fraction of components past δ (0 under square)."""
    n_clips, seq_len = 2, 7
    batch = _batch(n_clips, seq_len)
    mhr = _mhr_out(adapter, n_clips, seq_len, moving=True, seed=5)
    forces = torch.randn(n_clips * seq_len, 4, 3,
                         generator=torch.Generator().manual_seed(9)) * 0.3
    probs = torch.zeros(n_clips * seq_len, 4)
    out = _out(mhr, forces, probs)

    square = PhysicsLoss(_cfg(), device="cpu", adapter=adapter)
    _, parts_sq = square(out, batch)
    raw = parts_sq["raw_residual"]
    res = parts_sq["terms"]["residual"]
    # square: raw_residual == residual term (numerator + mass), sat_frac == 0.
    assert raw["weight_mass"] == res["weight_mass"]
    assert float(raw["weighted_numerator_tensor"]) == pytest.approx(
        float(res["weighted_numerator_tensor"]), rel=1e-6)
    assert parts_sq["residual_sat_frac"] == 0.0

    huber = PhysicsLoss(
        _cfg(residual_robust={"kind": "pseudo_huber",
                              "delta_force": 1.0, "delta_torque": 0.5}),
        device="cpu", adapter=adapter)
    _, parts_hb = huber(out, batch)
    raw_num = float(raw["weighted_numerator_tensor"])
    # raw headline is invariant to the robustifier; the objective term shrinks.
    assert float(parts_hb["raw_residual"]["weighted_numerator_tensor"]) == pytest.approx(
        raw_num, rel=1e-5)
    assert 0.0 < float(parts_hb["terms"]["residual"]["weighted_numerator_tensor"]) < raw_num
    assert 0.0 <= parts_hb["residual_sat_frac"] <= 1.0

    # sat_frac endpoints: a tiny δ saturates ~everything, a huge δ saturates nothing.
    tiny = PhysicsLoss(
        _cfg(residual_robust={"kind": "pseudo_huber",
                              "delta_force": 1e-6, "delta_torque": 1e-6}),
        device="cpu", adapter=adapter)
    assert tiny(out, batch)[1]["residual_sat_frac"] > 0.99
    huge = PhysicsLoss(
        _cfg(residual_robust={"kind": "pseudo_huber",
                              "delta_force": 1e6, "delta_torque": 1e6}),
        device="cpu", adapter=adapter)
    assert huge(out, batch)[1]["residual_sat_frac"] == 0.0


# ------------------------------------------------------ camera-jerk filter (§2)

def test_camera_jerk_filter(adapter):
    """A per-frame camera jump above ``physics.max_cam_jump_m`` makes only that clip
    physics-ineligible (no raise, counted); ``null`` threshold is a no-op."""
    seq_len = 7
    mhr = _mhr_out(adapter, 2, seq_len, moving=True, seed=1)
    forces = torch.zeros(2 * seq_len, 4, 3)
    probs = torch.zeros(2 * seq_len, 4)
    out = _out(mhr, forces, probs)

    cam_jump = torch.full((2, seq_len), 0.1)
    cam_jump[1, 3] = 0.9                                     # clip 1: one >0.5 m jump
    batch = _batch(2, seq_len)
    batch["cam_jump_m"] = cam_jump.reshape(-1)

    off = PhysicsLoss(_cfg(), device="cpu", adapter=adapter)   # max_cam_jump_m=None
    _, parts_off = off(out, batch)
    assert parts_off["n_eligible_clips"] == 2
    assert parts_off["n_jerk_excluded_clips"] == 0

    on = PhysicsLoss(_cfg(max_cam_jump_m=0.5), device="cpu", adapter=adapter)
    _, parts_on = on(out, batch)
    assert parts_on["n_eligible_clips"] == 1                 # clip 1 dropped, no raise
    assert parts_on["n_jerk_excluded_clips"] == 1


# ------------------------------------------------ affine residual decomposition (§4)

def test_affine_residual_matches_full_rnea(adapter):
    """``r0 + B·vec(f)`` reproduces the full-RNEA raw residual (mean over residual
    frames) for random forces — validating the basis used by the eval baselines."""
    n_clips, seq_len = 2, 7
    batch = _batch(n_clips, seq_len)
    loss = PhysicsLoss(_cfg(), device="cpu", adapter=adapter)
    for seed in (11, 21):
        mhr = _mhr_out(adapter, n_clips, seq_len, moving=True, seed=seed)
        forces = torch.randn(n_clips * seq_len, 4, 3,
                             generator=torch.Generator().manual_seed(seed)) * 0.25
        probs = torch.rand(n_clips * seq_len, 4,
                           generator=torch.Generator().manual_seed(seed + 1))
        out = _out(mhr, forces, probs)

        aff = loss.affine_residual(out, batch)
        assert aff is not None
        pred6 = aff["r0"] + torch.einsum("...ij,...j->...i", aff["basis"], aff["f_pred"])
        affine_mean = float((pred6 ** 2).sum(-1).mean())

        _, parts = loss(out, batch)                         # independent full RNEA
        assert affine_mean == pytest.approx(parts["raw_residual"]["loss"], rel=1e-4)


def test_affine_basis_matches_autograd(adapter):
    """The basis ``B`` (columns = ∂r/∂f_k) matches autograd: the forward's square
    residual gradient w.r.t. the force leaf equals the analytic ``2·Bᵀ(r0 + B·f)``
    placed at the (eligible clip, residual frame) rows."""
    n_clips, seq_len = 2, 7                                  # default kernel [1] -> r=0
    batch = _batch(n_clips, seq_len)
    mhr = _mhr_out(adapter, n_clips, seq_len, moving=True, seed=11)
    forces = torch.randn(n_clips * seq_len, 4, 3,
                         generator=torch.Generator().manual_seed(11)) * 0.25
    probs = torch.rand(n_clips * seq_len, 4,
                       generator=torch.Generator().manual_seed(12))
    loss = PhysicsLoss(_cfg(), device="cpu", adapter=adapter)

    aff = loss.affine_residual(_out(mhr, forces, probs), batch)
    forces_leaf = forces.clone().requires_grad_(True)
    _, parts = loss(_out(mhr, forces_leaf, probs), batch)
    # square residual numerator == Σ‖r‖²; its grad w.r.t. the forces is the autograd
    # reference (zero_touch adds an exactly-zero-gradient term, so it is transparent).
    parts["terms"]["residual"]["weighted_numerator_tensor"].backward()

    r_full = aff["r0"] + torch.einsum("...ij,...j->...i", aff["basis"], aff["f_pred"])
    analytic = 2.0 * torch.einsum("...ij,...i->...j", aff["basis"], r_full)  # (Nc, nr, 12)
    sel = _residual_frame_indices(seq_len, 0)
    expected = torch.zeros(n_clips * seq_len, 4, 3)
    for clip in range(n_clips):
        for j, frame in enumerate(sel):
            expected[clip * seq_len + frame] = analytic[clip, j].reshape(4, 3)
    assert torch.allclose(forces_leaf.grad, expected, atol=1e-4, rtol=1e-3)


# --------------------------------------------------------------- config validation

def test_physics_config_validation():
    """physics.enabled cross-key requirements and weight/kernel sanity."""
    from contact.config import _validate_physics

    force_off = {"enabled": False}
    force_on = {"enabled": True}
    base = {
        "physics": {
            "enabled": False, "model_path": None, "lod": 1, "gravity": 9.81,
            "use_warp": False,
            "min_frames": 5, "smoothing_kernel": [0.25, 0.5, 0.25],
            "loss": {"residual": 1.0, "force_noncontact": 1.0, "force_at_contact": 0.1,
                     "contact_min_bw": 0.05, "force_smooth": 0.1, "force_l2": 0.01,
                     "torque_l2": 0.01, "torque_smooth": 0.0},
        },
        "data": {"datasets": [{"name": "climbing_videos", "config": "x"}],
                 "sequence": {"frames_per_clip": 8}},
    }
    _validate_physics(base, force_off)                       # disabled: only leaf sanity

    enabled = copy.deepcopy(base)
    enabled["physics"]["enabled"] = True
    _validate_physics(enabled, force_on)                     # ok

    with pytest.raises(ValueError, match="force_head.enabled"):
        _validate_physics(enabled, force_off)

    no_video = copy.deepcopy(enabled)
    no_video["data"]["datasets"] = [{"name": "damon", "config": "x"}]
    with pytest.raises(ValueError, match="climbing_videos"):
        _validate_physics(no_video, force_on)

    short = copy.deepcopy(enabled)
    short["data"]["sequence"]["frames_per_clip"] = 4
    with pytest.raises(ValueError, match="frames_per_clip"):
        _validate_physics(short, force_on)

    bad_kernel = copy.deepcopy(base)
    bad_kernel["physics"]["smoothing_kernel"] = [0.5, 0.5]   # even length
    with pytest.raises(ValueError, match="smoothing_kernel"):
        _validate_physics(bad_kernel, force_off)

    bad_weight = copy.deepcopy(base)
    bad_weight["physics"]["loss"]["residual"] = -1.0
    with pytest.raises(ValueError, match="residual"):
        _validate_physics(bad_weight, force_off)
