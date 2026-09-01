from __future__ import annotations

import math

import pytest
import torch

import scripts.evaluate as ev
from contact.targets import TargetSpec
from scripts.evaluate import evaluate


class _ContactModel(torch.nn.Module):
    def _initialize_batch(self, batch):
        pass

    def forward_step(self, batch, decoder_type="body", precomputed_features=None):
        assert decoder_type == "body"
        return {"contact": {"joint_logits": batch["logits"]}}


def test_evaluate_reports_named_outputs_f2_and_threshold_curve():
    loader = [{
        "logits": torch.tensor([[10.0, -10.0], [-10.0, 10.0]]),
        "targets": {"joint": {
            "gt": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "mask": torch.ones(2, 2),
        }},
    }]
    result = evaluate(
        _ContactModel(),
        loader,
        ["joint"],
        "cpu",
        threshold=0.5,
        curve_thresholds=(0.3, 0.7),
        output_names={"joint": ("left_hand", "right_hand")},
    )["joint"]

    assert result["f1"] > 0.999
    assert result["f2"] > 0.999
    assert set(result["per_output"]) == {"left_hand", "right_hand"}
    assert all(value["f2"] > 0.999 for value in result["per_output"].values())
    assert [point["threshold"] for point in result["threshold_curve"]] == [0.3, 0.5, 0.7]


def test_evaluate_center_policy_ignores_wrong_noncenter_rows():
    # Two T=5 clips. Only flattened rows 2 and 7 are correct; every other row is
    # deliberately wrong and must not affect center-only metrics.
    gt = torch.zeros(10, 2)
    gt[:, 0] = 1.0
    logits = torch.full((10, 2), -10.0)
    logits[:, 1] = 10.0
    logits[[2, 7], 0] = 10.0
    logits[[2, 7], 1] = -10.0
    loader = [{
        "seq_len": 5,
        "logits": logits,
        "targets": {"joint": {"gt": gt, "mask": torch.ones(10, 2)}},
    }]

    result = evaluate(
        _ContactModel(), loader, ["joint"], "cpu", target_frame="center",
    )["joint"]
    assert result["f1"] > 0.999
    assert (result["tp"], result["fp"], result["fn"]) == (2, 0, 0)


# ------------------------------------------------------------ affine baselines

def _affine_group(n_clips: int, n_res: int, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    return {
        "r0": torch.randn(n_clips, n_res, 6, generator=gen),
        "basis": torch.randn(n_clips, n_res, 6, 12, generator=gen) * 0.3,
        "f_pred": torch.randn(n_clips, n_res, 12, generator=gen) * 0.2,
        "probs": torch.rand(n_clips, n_res, 4, generator=gen),
    }


def test_affine_baselines_structure_and_math():
    """Network mean matches an independent recomputation; the fitted constant can
    only improve on zero forces; the shuffled baseline carries pooled per-frame
    quantiles; a size-1 T-group is counted unshufflable."""
    by_t = {7: _affine_group(3, 2, seed=0), 16: _affine_group(1, 10, seed=1)}
    res = ev._affine_baselines(by_t)

    manual = []
    for group in by_t.values():
        pred = group["r0"] + torch.einsum(
            "...ij,...j->...i", group["basis"], group["f_pred"])
        manual.append((pred ** 2).sum(-1).reshape(-1))
    manual = torch.cat(manual)
    assert res["network"]["mean"] == pytest.approx(float(manual.mean()), rel=1e-5)
    assert res["n_residual_frames"] == manual.numel()
    assert res["constant"]["mean"] <= res["zero"]["mean"] + 1e-6

    for key in ("mean", "std", "p50", "p90", "p99", "max"):
        assert key in res["shuffled"] and math.isfinite(res["shuffled"][key]), key
    assert res["n_unshuffled_clips"] == 1                    # the size-1 T=16 group
    assert set(res["head_force_component_std"]) == {
        "left_hand", "right_hand", "left_foot", "right_foot"}
    assert isinstance(res["input_dependent"], bool)


def test_affine_shuffle_is_never_identity():
    """Two clips whose predictions exactly zero their own residual: every cyclic
    rotation must swap them, giving a strictly positive shuffled residual — an
    unconstrained permutation could draw the identity and (wrongly) report 0."""
    basis = torch.cat((torch.eye(6), torch.zeros(6, 6)), dim=1)   # (6, 12)
    c0 = torch.zeros(12)
    c0[0] = 1.0
    c1 = torch.zeros(12)
    c1[1] = -1.0
    group = {
        "r0": torch.stack((-(basis @ c0), -(basis @ c1))).unsqueeze(1),   # (2, 1, 6)
        "basis": basis.expand(2, 1, 6, 12).contiguous(),
        "f_pred": torch.stack((c0, c1)).unsqueeze(1),                     # (2, 1, 12)
        "probs": torch.full((2, 1, 4), 0.5),
    }
    res = ev._affine_baselines({7: group}, n_shuffles=5)
    assert res["network"]["mean"] == pytest.approx(0.0, abs=1e-10)
    # Size-2 group: the only nonzero cyclic offset is 1 (a swap) — every shuffle is
    # identical (std 0) and strictly worse than the network.
    assert res["shuffled"]["mean"] == pytest.approx(2.0, rel=1e-5)
    assert res["shuffled"]["std"] == pytest.approx(0.0, abs=1e-9)
    assert res["n_unshuffled_clips"] == 0
    assert res["beats_shuffled"] is True


def test_affine_all_groups_unshufflable_fails_conservatively():
    """Only size-1 groups: the shuffled baseline is NaN, beats_shuffled is False
    (input-dependence cannot be proven without a shuffle), every clip counted."""
    by_t = {7: _affine_group(1, 2, seed=2), 9: _affine_group(1, 3, seed=3)}
    res = ev._affine_baselines(by_t)
    assert math.isnan(res["shuffled"]["mean"])
    assert res["n_unshuffled_clips"] == 2
    assert res["beats_shuffled"] is False
    assert res["input_dependent"] is False


# --------------------------------------------------------- evaluate_physics stubs

class _StubPhysicsLoss:
    """Duck-typed PhysicsLoss yielding one pre-built ``parts`` dict per batch."""

    def __init__(self, parts_sequence):
        self._parts = list(parts_sequence)
        self._index = 0

    def __call__(self, out, batch):
        parts = self._parts[self._index]
        self._index += 1
        return torch.tensor(0.0), parts

    def diagnostics(self, out, batch):
        return None

    def affine_residual(self, out, batch):
        return None


def _phys_parts(numerator: float, mass: float, sat: float, jerk: int) -> dict:
    return {
        "terms": {},
        "raw_residual": {
            "weighted_numerator_tensor": torch.tensor(numerator),
            "weight_mass": mass, "loss": numerator / max(mass, 1.0)},
        "residual_sat_frac": sat,
        "n_jerk_excluded_clips": jerk,
    }


def test_evaluate_physics_zero_mass_is_nan(monkeypatch):
    """Zero residual mass (everything ineligible/jerk-excluded) must report NaN —
    never a perfect 0 residual — while the jerk count still surfaces."""
    monkeypatch.setattr(ev, "forward_model", lambda model, batch: {})
    monkeypatch.setattr(ev, "batch_to_device", lambda batch, device: batch)
    stub = _StubPhysicsLoss([_phys_parts(0.0, 0.0, 0.0, jerk=3)])
    res = ev.evaluate_physics(None, [None], stub, "cpu",
                              threshold=0.5, contact_min_bw=0.05)
    assert math.isnan(res["physics_residual"])
    assert math.isnan(res["residual_sat_frac"])
    assert res["n_jerk_excluded_clips"] == 3
    assert res["n_frames"] == 0


def test_evaluate_physics_aggregates_sat_and_jerk(monkeypatch):
    """Headline = exact mass-weighted mean over batches; sat_frac mass-weighted;
    jerk exclusions summed."""
    monkeypatch.setattr(ev, "forward_model", lambda model, batch: {})
    monkeypatch.setattr(ev, "batch_to_device", lambda batch, device: batch)
    stub = _StubPhysicsLoss([
        _phys_parts(6.0, 4.0, sat=0.25, jerk=1),
        _phys_parts(2.0, 2.0, sat=0.10, jerk=0),
    ])
    res = ev.evaluate_physics(None, [None, None], stub, "cpu",
                              threshold=0.5, contact_min_bw=0.05)
    assert res["physics_residual"] == pytest.approx(8.0 / 6.0)
    assert res["residual_sat_frac"] == pytest.approx((0.25 * 4 + 0.10 * 2) / 6.0)
    assert res["n_jerk_excluded_clips"] == 1


def test_manual_test_loader_passes_auto_frame_stride_through(tmp_path, monkeypatch):
    # Regression: `_manual_test_loader` int()-cast the stride, so every config
    # using `frame_stride: auto` (all the T=60 ones) died with
    # "invalid literal for int() with base 10: 'auto'" before scoring a batch.
    from contact.config import load_config

    cfg_path = tmp_path / "eval_auto.yaml"
    cfg_path.write_text(
        "base: configs/base.yaml\n"
        "data:\n"
        "  datasets:\n"
        "    - {name: climbing_corpus, config: configs/datasets/climbing_corpus.yaml}\n"
        "  eval_split: test\n"
        "  sequence: {frames_per_clip: 8, frame_stride: auto}\n"
        "contact:\n"
        "  primary_target: joint\n"
        "  targets:\n"
        "    vertex: {enabled: false}\n"
        "    joint: {enabled: true, joint_set: kindyn_6}\n"
        # `frame_stride: auto` is only permitted alongside a motion/pose
        # pipeline (see _validate_data), which is precisely the case that
        # used to crash here.
        "pose_supervision: {enabled: true}\n"
        "train: {finetune_pose_head: true}\n")
    cfg = load_config(cfg_path)

    seen = {}

    class _Stub:
        def __init__(self, root, **kwargs):
            seen.update(kwargs)

        def __len__(self):
            return 0

    monkeypatch.setattr(ev, "ClimbingCorpusDataset", _Stub)
    ev._manual_test_loader(cfg, (256, 256), TargetSpec.from_config(cfg))
    assert seen["frame_stride"] == "auto"
