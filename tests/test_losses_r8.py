"""Round-8 supervision unit tests (CPU, synthetic batches, float32).

* ``contact_consistency.stencil``: the forward speed sees a period-2 wobble that
  the central one cancels exactly, its rows are :func:`model.refiner.forward_valid`'s,
  and the ``gt_speed`` floor uses the same stencil;
* deep supervision (``layer_weight``) of the contact BCE and of the force loss's
  four per-limb terms: the pooled ``*_layer`` term is the sum of the intermediate
  layers' own terms at ``layer_weight``, masses add, the gradient reaches every
  intermediate layer, the sums never get a layer twin, and at ``layer_weight: 0``
  the term is gone and the base terms are bit-identical;
* the loader's gravity source -> ``gravity_measured`` flag on the three corpus
  sources, a missing file and an unknown source;
* ``force_consistency``'s joint-torque term: the per-DOF multipliers follow the
  joint-GROUP table (and an incomplete table is refused), a static standing body
  routes its load through whichever limbs hold it up — legs when the feet carry it,
  shoulders and elbows when the hands do, which the strength-weighted term prices
  two orders of magnitude apart — and the term moves the forces only, never the body.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import roma
import torch
import torch.nn.functional as F
import yaml

from data.climbing_videos.kindyn import gravity_measured
from model.loss import KINDYN_GROUP_NAMES
from model.loss.contact import ContactLoss
from model.loss.contact_consistency import GROUP_JOINTS, ContactConsistencyLoss
from model.loss.force import LAYER_TERM_NAMES, ForceLoss
from model.loss.force_consistency import ForceConsistencyLoss
from model.refiner import forward_valid, stencil_valid

REPO = Path(__file__).resolve().parents[1]
FPS = 25.0
#: Amplitude (m) of the period-2 wobble the two stencils are compared on.
WOBBLE_M = 0.02


@pytest.fixture(scope="module")
def base_cfg():
    return yaml.safe_load((REPO / "configs" / "base.yaml").read_text())


# ------------------------------------------------------------------ contact_consistency

def wobble(n_clips: int, seq_len: int, joint: int, amplitude: float) -> torch.Tensor:
    """``(n, 22, 3)`` world joints: all at the origin but ``joint``, which flips
    between ``+-amplitude`` along x every frame (a period-2 = Nyquist wobble)."""
    sign = torch.tensor([1.0 if t % 2 == 0 else -1.0 for t in range(seq_len)]).repeat(n_clips)
    joints = torch.zeros(n_clips * seq_len, 22, 3)
    joints[:, joint, 0] = amplitude * sign
    return joints


def stillness_batch(n_clips: int = 2, seq_len: int = 8) -> tuple[dict, torch.Tensor]:
    """A labelled two-clip batch (one dropped frame) + the GT's own wobbling joints."""
    n = n_clips * seq_len
    valid = torch.ones(n_clips, seq_len, dtype=torch.bool)
    valid[1, 3] = False                                  # a tracking gap inside clip 1
    contact_gt = torch.zeros(n, 6)
    contact_gt[:, 0] = 1.0                               # only the left hand is in contact
    batch = {
        "seq_len": seq_len,
        "frame_pos_sec": (torch.arange(seq_len, dtype=torch.float32) / FPS).repeat(n_clips),
        "frame_valid": valid.reshape(n),
        "contact_gt": contact_gt,
        "contact_valid": torch.ones(n, 6),
        "contact_conf": torch.ones(n, 6),
        "smplx_valid": torch.ones(n, dtype=torch.bool),
        "smplx_joints_world": wobble(n_clips, seq_len, GROUP_JOINTS[0], 0.5 * WOBBLE_M),
    }
    return batch, valid


def consistency_loss(base_cfg, stencil: str) -> ContactConsistencyLoss:
    cfg = copy.deepcopy(base_cfg)
    cfg["contact_consistency"].update(enabled=True, weight=1.0, confidence_weights=True,
                                      stencil=stencil)
    return ContactConsistencyLoss(cfg, None, "cpu")


def test_forward_stencil_sees_the_wobble_the_central_one_cancels(base_cfg):
    batch, valid = stillness_batch()
    n_clips, seq_len = valid.shape
    pred = wobble(n_clips, seq_len, GROUP_JOINTS[0], WOBBLE_M)
    out = {"smplx": {"joints_world": pred}}

    forward = consistency_loss(base_cfg, "forward")(out, batch, train=True)
    central = consistency_loss(base_cfg, "central")(out, batch, train=True)

    # Rows are each stencil's own support, and only the labelled group is weighted.
    assert forward.terms["still"].mass == pytest.approx(float(forward_valid(valid).sum()))
    assert central.terms["still"].mass == pytest.approx(float(stencil_valid(valid, 1).sum()))
    # |x(t+1) - x(t)| / dt of a +-A flip is 2 A fps; the central difference is exactly 0.
    speed = 2.0 * WOBBLE_M * FPS
    assert float(forward.terms["still"].numerator) == pytest.approx(
        speed * forward.terms["still"].mass, rel=1e-5)
    assert float(central.terms["still"].numerator) == 0.0
    assert consistency_loss(base_cfg, "forward").metrics(forward.stats)["speed"] == pytest.approx(
        speed, rel=1e-5)
    assert consistency_loss(base_cfg, "central").metrics(central.stats)["speed"] == 0.0
    # The GT floor is measured with the SAME stencil (its wobble is half as wide).
    assert consistency_loss(base_cfg, "forward").metrics(forward.stats)["gt_speed"] == pytest.approx(
        0.5 * speed, rel=1e-5)
    assert consistency_loss(base_cfg, "central").metrics(central.stats)["gt_speed"] == 0.0


def test_forward_stencil_gradient_reaches_the_pose(base_cfg):
    batch, valid = stillness_batch()
    n_clips, seq_len = valid.shape
    pred = wobble(n_clips, seq_len, GROUP_JOINTS[0], WOBBLE_M).requires_grad_(True)
    result = consistency_loss(base_cfg, "forward")(
        {"smplx": {"joints_world": pred}}, batch, train=True)
    result.terms["still"].numerator.backward()
    # Only the labelled extremity of rows the stencil supports carries gradient.
    assert float(pred.grad[:, GROUP_JOINTS[0]].abs().sum()) > 0.0
    unsupervised = [j for j in range(22) if j != GROUP_JOINTS[0]]
    assert float(pred.grad[:, unsupervised].abs().sum()) == 0.0


def test_unknown_stencil_is_rejected(base_cfg):
    with pytest.raises(ValueError, match="contact_consistency.stencil"):
        consistency_loss(base_cfg, "backward")


# ------------------------------------------------------------------ deep supervision

def label_batch(n: int = 12, seed: int = 3) -> dict:
    """Contact labels with unlabelled rows and fractional confidences."""
    torch.manual_seed(seed)
    return {
        "contact_gt": (torch.rand(n, 6) > 0.5).float(),
        "contact_valid": (torch.rand(n, 6) > 0.2).float(),
        "contact_conf": torch.rand(n, 6),
    }


def contact_cfg(base_cfg, layer_weight: float) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["contact_supervision"].update(
        enabled=True, weight=2.0, confidence_weights=True, layer_weight=layer_weight,
        class_weights={"positive": [1.0, 1.5, 2.0, 1.0, 3.0, 3.0],
                       "negative": [1.0, 1.0, 1.0, 0.5, 1.0, 1.0]})
    return cfg


def contact_out(logits: torch.Tensor, layers: list[torch.Tensor] | None = None) -> dict:
    contact = {"logits": logits, "probs": torch.sigmoid(logits)}
    if layers is not None:
        contact["logits_layers"] = layers
    return {"contact": contact}


def test_contact_bce_matches_an_explicit_weighted_reference(base_cfg):
    batch = label_batch()
    logits = torch.randn(12, 6)
    result = ContactLoss(contact_cfg(base_cfg, 0.0), None, "cpu")(
        contact_out(logits), batch, train=True)

    gt, conf = batch["contact_gt"], batch["contact_valid"] * batch["contact_conf"]
    pos = torch.tensor([1.0, 1.5, 2.0, 1.0, 3.0, 3.0])
    neg = torch.tensor([1.0, 1.0, 1.0, 0.5, 1.0, 1.0])
    weight = conf * (gt * pos + (1.0 - gt) * neg)
    reference = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
    assert float(result.terms["bce"].numerator) == pytest.approx(
        2.0 * float((reference * weight).sum()), rel=1e-6)
    assert result.terms["bce"].mass == pytest.approx(float(weight.sum()))


def test_contact_layer_terms_pool_the_intermediate_layers(base_cfg):
    batch = label_batch()
    torch.manual_seed(7)
    layers = [torch.randn(12, 6, requires_grad=True) for _ in range(3)]
    loss = ContactLoss(contact_cfg(base_cfg, 0.5), None, "cpu")
    assert loss.term_names == ("bce", "bce_layer")

    result = loss(contact_out(layers[-1], layers), batch, train=True)
    assert set(result.terms) == set(loss.term_names)

    # The pooled term is the intermediate layers' own BCE, summed, at `layer_weight`.
    single = ContactLoss(contact_cfg(base_cfg, 0.0), None, "cpu")
    per_layer = [single(contact_out(logits.detach()), batch, train=True).terms["bce"]
                 for logits in layers[:-1]]
    assert float(result.terms["bce_layer"].numerator) == pytest.approx(
        0.5 * sum(float(term.numerator) for term in per_layer), rel=1e-6)
    assert result.terms["bce_layer"].mass == pytest.approx(
        sum(term.mass for term in per_layer))
    assert result.terms["bce_layer"].mass == pytest.approx(2.0 * result.terms["bce"].mass)
    # The final layer's own term is untouched by the extra supervision.
    assert float(result.terms["bce"].numerator) == pytest.approx(
        float(single(contact_out(layers[-1].detach()), batch, train=True)
              .terms["bce"].numerator), rel=1e-6)

    # Every intermediate layer gets gradient from the pooled term; the final one none.
    result.terms["bce_layer"].numerator.backward()
    assert all(float(layer.grad.abs().sum()) > 0.0 for layer in layers[:-1])
    assert float(layers[-1].grad.abs().sum()) == 0.0


def test_contact_layer_weight_zero_is_the_previous_loss(base_cfg):
    batch = label_batch()
    logits = torch.randn(12, 6)
    off = ContactLoss(contact_cfg(base_cfg, 0.0), None, "cpu")
    assert off.term_names == ("bce",)
    plain = off(contact_out(logits), batch, train=True)
    assert set(plain.terms) == {"bce"}
    # Bit-identical to the same batch scored by a deep-supervised build.
    deep = ContactLoss(contact_cfg(base_cfg, 0.5), None, "cpu")(
        contact_out(logits, [logits, logits]), batch, train=True)
    assert float(plain.terms["bce"].numerator) == float(deep.terms["bce"].numerator)
    assert plain.terms["bce"].mass == deep.terms["bce"].mass
    assert torch.equal(plain.stats, deep.stats)


def force_cfg(base_cfg, layer_weight: float) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["force_supervision"].update(enabled=True, confidence=True, layer_weight=layer_weight)
    cfg["force_supervision"]["loss"].update(
        force=1.0, magnitude=0.5, direction=0.25, noncontact=2.0,
        sum_force=1.0, sum_torque=1.0)
    return cfg


def force_batch(n: int = 12, seed: int = 5) -> dict:
    """In- and off-contact rows with lever arms, confidences and a body-frame GT."""
    torch.manual_seed(seed)
    contact = torch.rand(n, 6) > 0.4
    gt = 0.5 * torch.randn(n, 6, 3) * contact[..., None].float()
    return {
        "force_gt": gt,
        "force_lever": 0.3 * torch.randn(n, 6, 3),
        "force_contact": contact,
        "force_valid": torch.ones(n, dtype=torch.bool),
        "frame_valid": torch.ones(n, dtype=torch.bool),
        "force_conf": torch.rand(n).clamp(min=0.1),
        "smplx_root_rot": roma.random_rotmat(n),
    }


def force_out(forces: torch.Tensor, frame: torch.Tensor,
              layers: list[torch.Tensor] | None = None) -> dict:
    out = {"forces": forces, "frame": frame}
    if layers is not None:
        out["forces_layers"] = layers
    return {"force": out}


def test_force_layer_terms_pool_the_intermediate_layers(base_cfg):
    batch = force_batch()
    torch.manual_seed(11)
    frame = roma.random_rotmat(12)
    layers = [(0.4 * torch.randn(12, 6, 3)).requires_grad_(True) for _ in range(3)]
    loss = ForceLoss(force_cfg(base_cfg, 0.5), None, "cpu")
    assert loss.layer_terms == LAYER_TERM_NAMES
    assert loss.term_names == (
        "force", "magnitude", "direction", "noncontact", "sum_force", "sum_torque",
        "force_layer", "magnitude_layer", "direction_layer", "noncontact_layer")

    result = loss(force_out(layers[-1], frame, layers), batch, train=True)
    assert set(result.terms) == set(loss.term_names)

    single = ForceLoss(force_cfg(base_cfg, 0.0), None, "cpu")
    for name in LAYER_TERM_NAMES:
        per_layer = [single(force_out(f.detach(), frame), batch, train=True).terms[name]
                     for f in layers[:-1]]
        layer = result.terms[f"{name}_layer"]
        assert float(layer.numerator) == pytest.approx(
            0.5 * sum(float(term.numerator) for term in per_layer), rel=1e-5)
        assert layer.mass == pytest.approx(sum(term.mass for term in per_layer))
        assert layer.mass == pytest.approx(2.0 * result.terms[name].mass)
        assert result.terms[name].mass > 0

    # The gradient of the pooled terms reaches every intermediate layer, and only those.
    sum(result.terms[f"{name}_layer"].numerator for name in LAYER_TERM_NAMES).backward()
    assert all(float(f.grad.abs().sum()) > 0.0 for f in layers[:-1])
    assert float(layers[-1].grad.abs().sum()) == 0.0


def test_force_layer_weight_zero_is_the_previous_loss(base_cfg):
    batch = force_batch()
    torch.manual_seed(13)
    frame = roma.random_rotmat(12)
    forces = 0.4 * torch.randn(12, 6, 3)
    off = ForceLoss(force_cfg(base_cfg, 0.0), None, "cpu")
    assert off.term_names == ("force", "magnitude", "direction", "noncontact",
                              "sum_force", "sum_torque")
    plain = off(force_out(forces, frame), batch, train=True)
    assert set(plain.terms) == set(off.term_names)

    deep = ForceLoss(force_cfg(base_cfg, 0.5), None, "cpu")(
        force_out(forces, frame, [forces, forces]), batch, train=True)
    for name in off.term_names:
        assert float(plain.terms[name].numerator) == float(deep.terms[name].numerator)
        assert plain.terms[name].mass == deep.terms[name].mass
    assert torch.equal(plain.stats, deep.stats)


def test_force_layer_terms_follow_the_enabled_per_limb_terms(base_cfg):
    cfg = force_cfg(base_cfg, 0.5)
    cfg["force_supervision"]["loss"].update(magnitude=0.0, direction=0.0)
    loss = ForceLoss(cfg, None, "cpu")
    assert loss.layer_terms == ("force", "noncontact")
    assert loss.term_names == ("force", "noncontact", "sum_force", "sum_torque",
                               "force_layer", "noncontact_layer")


# ------------------------------------------------------------------ the gravity flag

def write_gravity(path: Path, source: str) -> Path:
    np.savez(path, gravity_world=np.array([0.0, 1.0, 0.0], np.float32), source=np.array(source))
    return path


@pytest.mark.parametrize("source, measured", [("ground", True), ("geocalib", True),
                                              ("ground-plane-3view", True),
                                              ("fallback_down", False)])
def test_gravity_measured_reads_the_geocalib_source(tmp_path, source, measured):
    assert gravity_measured(write_gravity(tmp_path / f"{source}.npz", source)) is measured


def test_gravity_measured_hard_fails_on_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        gravity_measured(tmp_path / "absent.npz")


# ------------------------------------------------------------------ joint torques

#: Frames of the static-body clips (the residual's +-2 stencil needs five).
STATIC_FRAMES = 7
#: Middle frame of such a clip — an interior row of the stencil.
MID = STATIC_FRAMES // 2


@pytest.fixture(scope="module")
def torque_loss(base_cfg):
    """``force_consistency`` with only the joint-torque term weighted (ungated forces)."""
    cfg = copy.deepcopy(base_cfg)
    cfg["force_consistency"].update(enabled=True, gate_by_contact=False)
    cfg["force_consistency"]["loss"].update(force=0.0, torque=0.0, joint_torque=1.0)
    return ForceConsistencyLoss(cfg, None, "cpu")


def standing_body() -> dict:
    """A motionless SMPL-X neutral body, upright at the world origin, y up."""
    return {
        "pelvis": torch.zeros(1, STATIC_FRAMES, 3),
        "root_rot": torch.eye(3).expand(1, STATIC_FRAMES, 3, 3).contiguous(),
        "body_rot": torch.eye(3).expand(1, STATIC_FRAMES, 21, 3, 3).contiguous(),
        "betas": torch.zeros(1, 10),
        "seconds": (torch.arange(STATIC_FRAMES, dtype=torch.float32) / FPS)[None],
        "valid": torch.ones(1, STATIC_FRAMES, dtype=torch.bool),
        "gravity": torch.tensor([[0.0, -1.0, 0.0]]),
    }


def support_forces(groups: tuple[str, ...], magnitude: float = 0.5) -> torch.Tensor:
    """``(1, T, 6, 3)`` world forces: ``magnitude`` bw straight up on each named group."""
    forces = torch.zeros(1, STATIC_FRAMES, len(KINDYN_GROUP_NAMES), 3)
    for group in groups:
        forces[:, :, KINDYN_GROUP_NAMES.index(group), 1] = magnitude
    return forces


def named_torque(wrench, tau: torch.Tensor, joint: str) -> float:
    """``|tau|`` (bw·m) of one named joint at the clip's middle frame."""
    robot = wrench.body.robot
    start = int(robot.idx_vs[robot.joint_names.index(joint)]) - 6
    return float(tau[0, MID, start: start + 3].norm())


def static_residual(wrench, groups: tuple[str, ...]):
    """The residual of the standing body held up by ``groups`` (1 bw of support in all)."""
    body = standing_body()
    return wrench.residual(body["pelvis"], body["root_rot"], body["body_rot"], body["betas"],
                           support_forces(groups), body["gravity"], body["seconds"],
                           body["valid"])


def test_dof_weights_follow_the_joint_group_table(torque_loss, base_cfg):
    table = base_cfg["force_consistency"]["joint_torque_multipliers"]
    weight = torque_loss.dof_weight
    robot = torque_loss.wrench.body.robot
    assert weight.shape == (int(robot.nv) - 6,)
    for joint, group in (("left_hip", "hip"), ("spine2", "spine"), ("right_wrist", "wrist"),
                         ("left_foot", "foot"), ("head", "head")):
        start = int(robot.idx_vs[robot.joint_names.index(joint)]) - 6
        assert weight[start: start + 3].tolist() == pytest.approx([table[group]] * 3)
    with pytest.raises(ValueError, match="joint group"):
        torque_loss.wrench.dof_weights({"hip": 1.0})


def test_static_body_routes_its_load_through_the_supported_limbs(torque_loss):
    wrench = torque_loss.wrench
    torques, weighted = {}, {}
    for label, groups in (("feet", ("left_ankle", "right_ankle")),
                          ("hands", ("left_hand", "right_hand"))):
        res_f, _, rows, tau = static_residual(wrench, groups)
        # One body weight of support leaves the root force balanced, so the joints carry it.
        assert bool(rows[0, MID]) and float(res_f[0, MID].norm()) < 1e-4
        torques[label] = {joint: named_torque(wrench, tau, joint) for joint in
                          ("left_hip", "left_knee", "left_shoulder", "left_elbow")}
        weighted[label] = float(torque_loss._joint_torque_sq(tau)[0, MID])

    # Standing on the heels: the legs carry the weight close to their own axis.
    assert torques["feet"]["left_hip"] < 0.05
    assert torques["feet"]["left_knee"] < 0.05
    # Hanging from the hands: the whole weight passes through the arms instead.
    assert torques["hands"]["left_shoulder"] > 0.15
    assert torques["hands"]["left_elbow"] > 0.05
    assert torques["hands"]["left_shoulder"] > 10.0 * torques["feet"]["left_shoulder"]
    assert torques["hands"]["left_hip"] < torques["feet"]["left_hip"]
    # Which is what the strength-weighted term prices: the same support costs far more
    # on the arms (wrist 17, shoulder 2.5) than on the legs (hip / knee 1.0).
    assert weighted["hands"] > 100.0 * weighted["feet"]


def torque_batch() -> dict:
    """A one-clip batch of the standing body; its GT floor is the same body, unforced."""
    body, n = standing_body(), STATIC_FRAMES
    return {
        "seq_len": n,
        "frame_pos_sec": body["seconds"].reshape(n),
        "frame_valid": body["valid"].reshape(n),
        "gravity_world": body["gravity"].expand(n, 3),
        "smplx_joints_world": torch.zeros(n, 22, 3),
        "smplx_root_rot": body["root_rot"].reshape(n, 3, 3),
        "smplx_body_rot": body["body_rot"].reshape(n, 21, 3, 3),
        "smplx_betas": body["betas"].expand(n, 10),
        "smplx_valid": torch.ones(n, dtype=torch.bool),
        "force_gt": torch.zeros(n, len(KINDYN_GROUP_NAMES), 3),
        "force_contact": torch.zeros(n, len(KINDYN_GROUP_NAMES), dtype=torch.bool),
        "force_valid": torch.ones(n, dtype=torch.bool),
    }


def test_joint_torque_term_moves_only_the_forces(torque_loss):
    body, n = standing_body(), STATIC_FRAMES
    pose = {name: body[name].clone().requires_grad_(True)
            for name in ("pelvis", "root_rot", "body_rot", "betas")}
    forces = support_forces(("left_hand", "right_hand")).reshape(
        n, len(KINDYN_GROUP_NAMES), 3).clone().requires_grad_(True)
    out = {"smplx": {"pelvis_world": pose["pelvis"].reshape(n, 3),
                     "root_rot_world": pose["root_rot"].reshape(n, 3, 3),
                     "body_rot": pose["body_rot"].reshape(n, 21, 3, 3),
                     "betas": pose["betas"].expand(n, 10)},
           "force": {"forces": forces, "frame": torch.eye(3).expand(n, 3, 3)}}

    assert torque_loss.term_names == ("joint_torque",)
    result = torque_loss(out, torque_batch(), train=True)
    assert set(result.terms) == {"joint_torque"}

    # The term is the strength-weighted squared torque of the same (fixed) body.
    _, _, rows, tau = static_residual(torque_loss.wrench, ("left_hand", "right_hand"))
    mask = rows.to(torch.float32)
    assert result.terms["joint_torque"].mass == pytest.approx(float(mask.sum()))
    assert float(result.terms["joint_torque"].numerator) == pytest.approx(
        float((torque_loss._joint_torque_sq(tau) * mask).sum()), rel=1e-5)
    metrics = torque_loss.metrics(result.stats)
    assert metrics["joint_torque"] == pytest.approx(
        float(result.terms["joint_torque"].numerator) / result.terms["joint_torque"].mass,
        rel=1e-5)
    # The GT floor is the same body under the (zero) kindyn GT forces: it hangs on nothing.
    assert 0.0 < metrics["gt_joint_torque"] < metrics["joint_torque"]

    # Only the forces may move to lighten it: the body it is measured on is detached.
    result.terms["joint_torque"].numerator.backward()
    assert float(forces.grad.abs().sum()) > 0.0
    for tensor in pose.values():
        assert tensor.grad is None or float(tensor.grad.abs().sum()) == 0.0
