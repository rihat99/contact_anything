"""Validate BetterHuman MHR + BetterRobot inverse dynamics end to end.

The CPU section checks gravity, external-force frame/sign conventions,
autograd through ``fext``, manifold trajectory derivatives, and shape-dependent
mass.  The optional CUDA section benchmarks the realistic explicit-FK + RNEA
forward/backward workload at the clip batches planned for contact-force
training.  Results are printed as JSON so this file can serve as a reproducible
Step 01 evidence artifact.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import better_human as bh
import better_robot as br
import numpy as np
import torch
from better_human.bodies import MHRClassic
from better_robot.lie import so3


GRAVITY_Y_UP = (0.0, -9.81, 0.0, 0.0, 0.0, 0.0)
EXTREMITY_NATIVE_NAMES = ("l_wrist", "r_wrist", "l_foot", "r_foot")
EXTREMITY_OUTPUT_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot")
EXTREMITY_SAM_ANCHOR_NAMES = ("l_wrist", "r_wrist", "l_ankle", "r_ankle")
EXTREMITY_SAM_ANCHOR_IDS = (62, 41, 13, 14)
EXPECTED_BR_JOINT_IDS = (139, 87, 8, 31)
DEFAULT_BATCHES = (40, 128)
DEFAULT_WARMUP = 1
DEFAULT_SAMPLES = 5
DEFAULT_SHAPE_BATCH = 1


def _resolve_model_path(explicit: Path | None) -> tuple[Path, str]:
    if explicit is not None:
        path, source = explicit.expanduser(), "--model-path"
    elif models_root := os.environ.get("BETTERHUMAN_MODELS_DIR"):
        path = Path(models_root).expanduser() / "MHR" / "converted" / "mhr_lod1.npz"
        source = "$BETTERHUMAN_MODELS_DIR"
    else:
        better_root = Path(__file__).resolve().parents[4]
        path = better_root / "BetterHuman" / "models" / "MHR" / "converted" / "mhr_lod1.npz"
        source = "sibling BetterHuman checkout"
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"MHR LOD1 archive not found at {path}; pass --model-path or set "
            "$BETTERHUMAN_MODELS_DIR"
        )
    return path, source


def _load_neutral_mhr(
    model_path: Path,
    *,
    device: torch.device,
) -> tuple[bh.MHR, bh.MHR, torch.Tensor]:
    body = bh.MHR(
        model_path,
        lod=1,
        use_expression=False,
        use_correctives=False,
        compute_mass=True,
        dtype=torch.float32,
        device=device,
    )
    exemplar = body.values.v_template
    shaped, q = body.from_classic(
        MHRClassic(
            identity_coeffs=exemplar.new_zeros(45),
            model_parameters=exemplar.new_zeros(204),
        )
    )
    return body, shaped, q


def _with_y_up_gravity(model: br.Model) -> br.Model:
    gravity = model.q_neutral.new_tensor(GRAVITY_Y_UP)
    return dataclasses.replace(
        model,
        values=dataclasses.replace(model.values, gravity=gravity),
    )


def _total_mass(model: br.Model) -> torch.Tensor:
    return model.values.body_inertias[..., 0].sum(dim=-1)


def _extremity_mapping(body: bh.MHR) -> tuple[torch.Tensor, list[dict[str, object]]]:
    native_names = body.structure.joint_names
    native_pose_ids = body.structure.native_pose_joint_indices
    if native_pose_ids is None:
        raise AssertionError("MHR must expose native_pose_joint_indices")

    ids: list[int] = []
    records: list[dict[str, object]] = []
    for output_name, anchor_name, anchor_id, native_name in zip(
        EXTREMITY_OUTPUT_NAMES,
        EXTREMITY_SAM_ANCHOR_NAMES,
        EXTREMITY_SAM_ANCHOR_IDS,
        EXTREMITY_NATIVE_NAMES,
        strict=True,
    ):
        native_index = native_names.index(native_name)
        joint_id = int(native_pose_ids[native_index])
        ids.append(joint_id)
        records.append(
            {
                "output_name": output_name,
                "sam_anchor_name": anchor_name,
                "sam_anchor_mhr70_index": anchor_id,
                "native_name": native_name,
                "native_index": native_index,
                "better_robot_joint_id": joint_id,
                "better_robot_joint_name": body.robot.joint_names[joint_id],
            }
        )
    if len(set(ids)) != len(ids):
        raise AssertionError(f"extremity joints must be unique, got {ids}")
    if tuple(ids) != EXPECTED_BR_JOINT_IDS:
        raise AssertionError(
            f"unexpected BetterRobot extremity ids {ids}; expected {EXPECTED_BR_JOINT_IDS}"
        )
    return torch.tensor(ids, dtype=torch.long, device=native_pose_ids.device), records


def _trajectory_derivatives(
    model: br.Model,
    q: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return q-aligned derivatives using BetterRobot's audited task pattern."""
    time_steps = q.shape[-2]
    if time_steps == 1:
        zeros = q.new_zeros((*q.shape[:-2], 1, model.nv))
        return zeros, zeros

    adjacent = model.difference(q[..., :-1, :], q[..., 1:, :]) / dt
    if time_steps == 2:
        velocity = torch.cat((adjacent[..., :1, :], adjacent[..., -1:, :]), dim=-2)
    else:
        central = model.difference(q[..., :-2, :], q[..., 2:, :]) / (2.0 * dt)
        velocity = torch.cat((adjacent[..., :1, :], central, adjacent[..., -1:, :]), dim=-2)

    delta_v = (velocity[..., 1:, :] - velocity[..., :-1, :]) / dt
    if time_steps == 2:
        acceleration = torch.cat((delta_v[..., :1, :], delta_v[..., -1:, :]), dim=-2)
    else:
        central_a = (velocity[..., 2:, :] - velocity[..., :-2, :]) / (2.0 * dt)
        acceleration = torch.cat((delta_v[..., :1, :], central_a, delta_v[..., -1:, :]), dim=-2)
    return velocity, acceleration


def _cpu_checks(model_path: Path) -> tuple[dict[str, object], br.Model]:
    base_body, shaped_body, q = _load_neutral_mhr(model_path, device=torch.device("cpu"))
    source_default_gravity = shaped_body.robot.values.gravity.detach().clone()
    model = _with_y_up_gravity(shaped_body.robot)
    extremity_ids, extremity_records = _extremity_mapping(shaped_body)

    if (model.nq, model.nv, model.njoints) != (132, 131, 203):
        raise AssertionError(
            f"unexpected MHR dimensions {(model.nq, model.nv, model.njoints)}"
        )
    if model.joint_models[1].kind != "free_flyer":
        raise AssertionError("MHR joint 1 must be a free flyer")
    if shaped_body.structure.up_axis != "+y":
        raise AssertionError(f"unexpected MHR up axis {shaped_body.structure.up_axis!r}")

    mass = _total_mass(model)
    if mass.ndim != 0 or not torch.isfinite(mass) or float(mass) <= 0.0:
        raise AssertionError(f"invalid total mass {mass}")
    torch.testing.assert_close(
        mass,
        mass.new_tensor(81.481842),
        rtol=0.0,
        atol=2e-4,
    )

    zero = q.new_zeros(model.nv)
    static_data = br.forward_kinematics(model, q)
    static_tau = br.rnea(model, q, zero, zero)
    if static_data.joint_pose_world is None:
        raise AssertionError("FK did not populate world joint poses")
    root_rotation = so3.to_matrix(static_data.joint_pose_world[1, 3:])
    expected_force_world = -mass * model.values.gravity[:3]
    expected_force_local = root_rotation.mT @ expected_force_world
    torch.testing.assert_close(
        static_tau[:3],
        expected_force_local,
        rtol=2e-5,
        atol=2e-3,
    )
    static_linear_norm = static_tau[:3].norm()
    static_wrench_norm = static_tau[:6].norm()

    # MHR70's ankle semantics correspond to the native l_foot/r_foot origins.
    foot_ids = extremity_ids[2:]
    foot_poses = static_data.joint_pose_world.index_select(-2, foot_ids)
    world_to_local = so3.to_matrix(foot_poses[..., 3:]).mT
    support_world = (-mass * model.values.gravity[:3] / len(foot_ids)).expand(
        len(foot_ids), -1
    )
    support_local = (world_to_local @ support_world.unsqueeze(-1)).squeeze(-1)
    support_wrenches = q.new_zeros(len(foot_ids), 6)
    support_wrenches[:, :3] = support_local
    fext_support = q.new_zeros(model.njoints, 6).index_copy(
        -2,
        foot_ids,
        support_wrenches,
    )
    supported_tau = br.rnea(model, q, zero, zero, fext=fext_support)
    supported_linear_norm = supported_tau[:3].norm()
    if float(supported_linear_norm / static_linear_norm) >= 1e-5:
        raise AssertionError(
            "two-foot support did not close the root force balance: "
            f"{float(supported_linear_norm)} N vs {float(static_linear_norm)} N"
        )

    fext_leaf = q.new_zeros(model.njoints, 6).requires_grad_()
    autograd_tau = br.rnea(model, q, zero, zero, fext=fext_leaf)
    autograd_loss = autograd_tau[:6].square().sum()
    autograd_loss.backward()
    fext_gradient = fext_leaf.grad
    if fext_gradient is None or not torch.isfinite(fext_gradient).all():
        raise AssertionError("RNEA produced a missing or non-finite fext gradient")
    target_gradient = fext_gradient.index_select(-2, extremity_ids)
    if int(torch.count_nonzero(target_gradient)) == 0:
        raise AssertionError("RNEA root residual has zero gradient at all extremities")

    dt = 0.05
    time_steps = 9
    time_axis = torch.arange(time_steps, dtype=q.dtype) * dt
    bob = 0.10 * torch.sin(2.0 * math.pi * 2.0 * time_axis)
    displacement = q.new_zeros(time_steps, model.nv)
    displacement[:, 1] = bob
    q_trajectory = model.integrate(q.expand(time_steps, -1), displacement)
    velocity, acceleration = _trajectory_derivatives(model, q_trajectory, dt)
    motion_tau = br.rnea(model, q_trajectory, velocity, acceleration)
    if not all(torch.isfinite(value).all() for value in (velocity, acceleration, motion_tau)):
        raise AssertionError("synthetic-motion derivatives or RNEA residual are non-finite")
    motion_wrench_norm = motion_tau[..., :6].norm(dim=-1)
    if float(motion_wrench_norm.max()) <= float(static_wrench_norm) * 1.1:
        raise AssertionError(
            "synthetic root bob did not produce a clearly larger dynamic residual"
        )

    identity = torch.linspace(-0.5, 0.5, 45, dtype=q.dtype)
    nonneutral = base_body.with_shape(
        identity=identity,
        proportions=torch.zeros(73, dtype=q.dtype),
    )
    nonneutral_mass = _total_mass(nonneutral.robot)
    mass_delta = (nonneutral_mass - mass).abs()
    if float(mass_delta) <= 1e-5:
        raise AssertionError("non-neutral MHR identity did not change total mass")

    report: dict[str, object] = {
        "model": {
            "lod": 1,
            "dtype": str(q.dtype),
            "nq": model.nq,
            "nv": model.nv,
            "njoints": model.njoints,
            "nbodies": model.nbodies,
            "up_axis": shaped_body.structure.up_axis,
            "total_mass_kg": float(mass),
            "extremities": extremity_records,
            "ankle_mapping_note": (
                "MHR70 left/right ankle map to native l_foot/r_foot joint origins"
            ),
        },
        "gravity_wrench": {
            "better_robot_source_default": source_default_gravity.tolist(),
            "gravity_world": model.values.gravity.tolist(),
            "root_linear_force": static_tau[:3].tolist(),
            "root_torque": static_tau[3:6].tolist(),
            "linear_norm_n": float(static_linear_norm),
            "expected_mass_times_g_n": float(mass * model.values.gravity[:3].norm()),
        },
        "support_force": {
            "support_joint_ids": foot_ids.tolist(),
            "force_world_per_joint_n": support_world.tolist(),
            "force_local_per_joint_n": support_local.tolist(),
            "root_linear_norm_n": float(supported_linear_norm),
            "root_torque_norm_nm": float(supported_tau[3:6].norm()),
            "linear_reduction_ratio": float(supported_linear_norm / static_linear_norm),
        },
        "autograd": {
            "loss": float(autograd_loss.detach()),
            "fext_gradient_norm": float(fext_gradient.norm()),
            "extremity_gradient_norm": float(target_gradient.norm()),
            "all_finite": True,
        },
        "motion": {
            "time_steps": time_steps,
            "dt_s": dt,
            "root_bob_amplitude_m": 0.10,
            "root_bob_frequency_hz": 2.0,
            "static_wrench_norm": float(static_wrench_norm),
            "motion_wrench_norm_min": float(motion_wrench_norm.min()),
            "motion_wrench_norm_max": float(motion_wrench_norm.max()),
            "all_finite": True,
        },
        "shape_mass": {
            "identity_definition": "linspace(-0.5, 0.5, 45)",
            "neutral_mass_kg": float(mass),
            "nonneutral_mass_kg": float(nonneutral_mass),
            "absolute_change_kg": float(mass_delta),
            "with_shape_recomputes_inertias": True,
        },
    }
    return report, model


def _quartiles(samples: list[float]) -> tuple[float, float]:
    if len(samples) == 1:
        return samples[0], samples[0]
    values = statistics.quantiles(samples, n=4, method="inclusive")
    return values[0], values[2]


def _measure_cuda(
    function: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    samples: int,
) -> dict[str, object]:
    for _ in range(warmup):
        output = function()
        del output
    torch.cuda.synchronize(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)

    samples_ms: list[float] = []
    for _ in range(samples):
        torch.cuda.synchronize(device)
        start_ns = time.perf_counter_ns()
        output = function()
        torch.cuda.synchronize(device)
        samples_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000.0)
        del output
    q1_ms, q3_ms = _quartiles(samples_ms)
    return {
        "median_ms": statistics.median(samples_ms),
        "q1_ms": q1_ms,
        "q3_ms": q3_ms,
        "iqr_ms": q3_ms - q1_ms,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "samples_ms": samples_ms,
        "cuda_memory": {
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
    }


def _cuda_benchmarks(
    model_path: Path,
    cpu_model: br.Model,
    *,
    device: torch.device,
    batches: Sequence[int],
    shape_batch: int,
    warmup: int,
    samples: int,
) -> dict[str, object]:
    if device.type != "cuda":
        raise ValueError(f"benchmark device must be CUDA, got {device}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --skip-cuda for CPU checks only")

    model = cpu_model.to(device=device, dtype=torch.float32)
    dynamics_measurements: list[dict[str, object]] = []
    for batch_size in batches:
        q = model.q_neutral.expand(batch_size, -1).contiguous()
        velocity = q.new_zeros(batch_size, model.nv)
        acceleration = torch.zeros_like(velocity)
        fext = q.new_zeros(batch_size, model.njoints, 6).requires_grad_()

        def rnea_forward_backward() -> None:
            fext.grad = None
            tau = br.rnea(model, q, velocity, acceleration, fext=fext)
            tau[..., :6].square().mean().backward()

        def explicit_fk_rnea_forward_backward() -> None:
            fext.grad = None
            br.forward_kinematics(model, q)
            tau = br.rnea(model, q, velocity, acceleration, fext=fext)
            tau[..., :6].square().mean().backward()

        for operation, function, note in (
            (
                "rnea_forward_backward_fext",
                rnea_forward_backward,
                "RNEA includes its internal FK traversal",
            ),
            (
                "explicit_fk_plus_rnea_forward_backward_fext",
                explicit_fk_rnea_forward_backward,
                "explicit FK for force-frame conversion plus RNEA's internal FK",
            ),
        ):
            measurement = _measure_cuda(
                function,
                device=device,
                warmup=warmup,
                samples=samples,
            )
            if fext.grad is None or not torch.isfinite(fext.grad).all():
                raise AssertionError(f"{operation} produced invalid fext gradients")
            dynamics_measurements.append(
                {
                    "operation": operation,
                    "batch_size": batch_size,
                    "note": note,
                    **measurement,
                }
            )

    shape_body = bh.MHR(
        model_path,
        lod=1,
        use_expression=False,
        use_correctives=False,
        compute_mass=True,
        dtype=torch.float32,
        device=device,
    )
    identity = torch.linspace(
        -0.5,
        0.5,
        45,
        dtype=torch.float32,
        device=device,
    ).expand(shape_batch, -1)
    proportions = torch.zeros(shape_batch, 73, dtype=torch.float32, device=device)

    def shape_bake() -> bh.MHR:
        return shape_body.with_shape(identity=identity, proportions=proportions)

    with torch.inference_mode():
        shape_measurement = _measure_cuda(
            shape_bake,
            device=device,
            warmup=warmup,
            samples=samples,
        )

    properties = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "device_name": properties.name,
        "dtype": "torch.float32",
        "warmup_iterations": warmup,
        "samples": samples,
        "statistic": "synchronized wall-clock median with inclusive Q1/Q3",
        "dynamics": dynamics_measurements,
        "with_shape": {
            "operation": "with_shape_forward",
            "batch_size": shape_batch,
            "note": "identity/proportion bake, including neutral FK and inertia recomputation",
            **shape_measurement,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    parser.add_argument("--shape-batch", type=int, default=DEFAULT_SHAPE_BATCH)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(batch <= 0 for batch in args.batches):
        parser.error("--batches values must be positive")
    if args.shape_batch <= 0:
        parser.error("--shape-batch must be positive")
    if args.warmup < 0 or args.samples <= 0:
        parser.error("--warmup must be non-negative and --samples must be positive")
    return args


def main() -> int:
    args = _parse_args()
    model_path, model_path_source = _resolve_model_path(args.model_path)
    torch.manual_seed(20260722)

    print("[validate] running CPU MHR dynamics checks", file=sys.stderr, flush=True)
    checks, cpu_model = _cpu_checks(model_path)
    if args.skip_cuda:
        cuda_report: dict[str, object] = {
            "skipped": True,
            "reason": "--skip-cuda",
        }
    else:
        device = torch.device(args.device)
        print(
            f"[validate] benchmarking CUDA on {device} at batches {args.batches}",
            file=sys.stderr,
            flush=True,
        )
        cuda_report = _cuda_benchmarks(
            model_path,
            cpu_model,
            device=device,
            batches=args.batches,
            shape_batch=args.shape_batch,
            warmup=args.warmup,
            samples=args.samples,
        )

    report = {
        "schema_version": 1,
        "status": "PASS",
        "model_path": str(model_path),
        "model_path_source": model_path_source,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "better_robot": br.__version__,
            "better_robot_source": str(Path(br.__file__).resolve()),
            "better_human": bh.__version__,
            "better_human_source": str(Path(bh.__file__).resolve()),
        },
        "cpu_checks": checks,
        "cuda_benchmark": cuda_report,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
