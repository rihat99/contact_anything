"""Reproduce stored BVR root wrenches and audit the learner's GT residual on CPU."""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "output_2/audits/rnea_v2"
BVR = REPO.parent / "BetterVideoReconstruction"
ROOT = REPO.parent / "data/ClimbingVideos"
OUT.mkdir(parents=True, exist_ok=True)
sys.dont_write_bytecode = True
os.environ["MPLCONFIGDIR"] = str(OUT / "matplotlib")
os.environ["WARP_CACHE_PATH"] = str(OUT / "warp")
sys.path[:0] = [str(REPO), str(BVR)]

import better_human as bh
import better_robot as br
import numpy as np
import roma
import torch
from torch import Tensor

from data.climbing_videos.kindyn import GROUPS_52, load_forces, load_smplx
from model.loss.force_consistency import ForceConsistencyLoss, GROUP_ROBOT_JOINTS, trajectory_derivatives
from tools.body import load_body, _DEFAULT_CONTACT_FRAMES
from tools.human_optim.kindyn import build_dynamics_spec
from tools.smplx_robot.dynamics import velocity_acceleration_from_trajectory
from train.config import load_config
from utils.geometry import smplx_q

STEPS = ("learner", "a_density_mass", "b_stencil", "b2_full_body", "c0_group_frame_wrenches", "c_contact_wrenches", "d_dyn_row")


def measure(residual: np.ndarray, stored: np.ndarray) -> dict:
    """Component agreement and mean vector norms; inputs are selected rows in bw."""
    result = {"rows": len(residual)}
    for name, columns in (("force", slice(0, 3)), ("torque", slice(3, 6))):
        actual, target = residual[:, columns], stored[:, columns]
        result[name] = float(np.linalg.norm(actual, axis=-1).mean()) if len(actual) else float("nan")
        result[f"stored_{name}"] = float(np.linalg.norm(target, axis=-1).mean()) if len(actual) else float("nan")
        result[f"mae_{name}"] = float(np.abs(actual - target).mean()) if len(actual) else float("nan")
        result[f"max_{name}"] = float(np.abs(actual - target).max()) if len(actual) else float("nan")
        result[f"corr_{name}"] = (float(np.corrcoef(actual.ravel(), target.ravel())[0, 1])
                                 if actual.size and actual.std() > 0 and target.std() > 0 else float("nan"))
    return result


def pure_force_residual(robot: object, q: Tensor, forces: Tensor, mass_g: float,
                        seconds: Tensor, spacing: int | None, dt: float) -> Tensor:
    """Evaluate one trajectory with six pure forces at the learner group joints."""
    if spacing is None:
        velocity, acceleration = trajectory_derivatives(robot, q[None], seconds[None])
        velocity, acceleration = velocity[0], acceleration[0]
    else:
        velocity, acceleration = velocity_acceleration_from_trajectory(robot, q, dt, spacing)
    joint_ids = torch.tensor([robot.joint_id(name) for name in GROUP_ROBOT_JOINTS], device=q.device)
    poses = br.forward_kinematics(robot, q).joint_pose_world
    rotation = roma.unitquat_to_rotmat(poses[:, joint_ids, 3:7])
    local = (rotation.transpose(-1, -2) @ forces.unsqueeze(-1)).squeeze(-1)
    external = q.new_zeros(len(q), robot.njoints, 6)
    external[:, joint_ids, :3] = local
    return br.rnea(robot, q, velocity, acceleration, fext=external)[:, :6] / mass_g


def audit_person(data: dict, person: int, body: object, dense22: bh.SMPLX,
                 loss: ForceConsistencyLoss, targets: dict, forces: dict) -> tuple[dict, dict, list]:
    """Compute the cumulative ladder on raw and auto-strided stored trajectories."""
    def tensor(value: object, dtype: torch.dtype = torch.float32) -> Tensor:
        return torch.as_tensor(value, dtype=dtype, device=loss.device)

    q_all, beta = tensor(data["q"][person]), tensor(data["betas"][person])
    assert torch.isfinite(q_all).all(), "Stored trajectory contains non-finite values"
    fps, mass = float(data["fps"]), float(data["total_mass"][person])
    raw_h, stride = max(1, round(fps / 30)), max(1, round(fps / 25))
    names = list(map(str, data["contact_frame_names"]))
    assert names == list(body.contact_frame_names)
    assert list(data["contact_frame_parents"]) == list(body.contact_frame_parents)
    assert list(map(str, data["joint_names"])) == list(body.joint_names)
    raw_q = q_all.cpu().numpy()
    metadata = {"person": person, "object_id": int(data["object_ids"][person]), "fps": fps,
                "raw_h": raw_h, "auto_stride": stride, "mass_stored_kg": mass,
                "frames": len(q_all), "scale": float(data["scale"][person]),
                "max_quaternion_norm_error": float(np.abs(np.linalg.norm(
                    raw_q[:, 3:].reshape(len(q_all), -1, 4), axis=-1) - 1).max())}
    metadata["mass_uniform22_kg"] = float(loss.body.with_shape(betas=beta).robot.values.body_inertias[:, 0].sum())
    saved, rows = {}, []
    raw_gate = None
    for mode, step in (("raw", 1), ("strided", stride)):
        index = np.arange(0, len(q_all), step)
        q = q_all[index]
        tracked = data["valid_mask"][person, index]
        spec = build_dynamics_spec(
            body, beta, tensor(data["frame_contact"][person, index]), tracked,
            data["gravity_world"], weights={}, dyn_opt={}, fps=fps / step,
            device=str(loss.device), q_world=q, cones=False)
        h = raw_h if mode == "raw" else 1
        assert spec.spacing == h
        gate = spec.dyn_row.bool().cpu().numpy()
        if mode == "raw":
            raw_gate = gate
        root_rotation = tensor(targets["smplx_root_rot"][person, index])
        pelvis = tensor(targets["smplx_joints_world"][person, index, 0])
        body_rotation = tensor(targets["smplx_body_rot"][person, index])
        root_forces = tensor(forces["force_gt"][person, index])
        root_forces *= tensor(forces["force_contact"][person, index])[..., None]
        world_bw = torch.einsum("tij,tkj->tki", root_rotation, root_forces)
        valid = tensor(targets["smplx_valid"][person, index] & forces["force_valid"][person, index], torch.bool)
        seconds = tensor(index / fps)
        linear, angular, learner_gate = loss.residual(
            pelvis[None], root_rotation[None], body_rotation[None], beta[None], world_bw[None],
            tensor(forces["gravity_world"])[None], seconds[None], valid[None])
        result = {"learner": torch.cat((linear, angular), dim=-1)[0]}
        if mode == "raw":
            batch = {key: tensor(targets[key][person, index]) for key in
                     ("smplx_joints_world", "smplx_root_rot", "smplx_body_rot")}
            batch["smplx_hand_rot"] = tensor(targets["smplx_hand_rot"][person, index])
            batch.update(smplx_betas=beta[None].expand(len(index), -1), smplx_valid=valid,
                         force_gt=tensor(forces["force_gt"][person, index]),
                         force_contact=tensor(forces["force_contact"][person, index]), force_valid=valid)
            exact = loss._gt_stats(batch, 1, len(index), seconds[None], valid[None],
                                   tensor(forces["gravity_world"])[None])
            mask = learner_gate.float()
            observed = [float((linear.norm(dim=-1) * mask).sum()), float(mask.sum()),
                        float((angular.norm(dim=-1) * mask).sum())]
            metadata["gt_stats_max_difference"] = float(np.abs(np.asarray(exact) - observed).max())
            assert metadata["gt_stats_max_difference"] == 0
        q22 = smplx_q(pelvis, root_rotation, body_rotation)
        shaped22 = dense22.with_shape(betas=beta).robot
        shaped22 = dataclasses.replace(shaped22, values=dataclasses.replace(
            shaped22.values, gravity=spec.model.values.gravity))
        result["a_density_mass"] = pure_force_residual(shaped22, q22, world_bw * (mass * 9.81),
                                                      mass * 9.81, seconds, None, step / fps)
        result["b_stencil"] = pure_force_residual(shaped22, q22, world_bw * (mass * 9.81),
                                                mass * 9.81, seconds, h, step / fps)
        result["b2_full_body"] = pure_force_residual(spec.model, q, world_bw * (mass * 9.81),
                                                   mass * 9.81, seconds, h, step / fps)
        frame_forces = tensor(data["frame_forces"][person, index])
        in_group = torch.as_tensor(np.isin(np.asarray(data["contact_frame_parents"]),
                                           np.concatenate([np.asarray(g) for g in GROUPS_52])),
                                   device=frame_forces.device)
        tau, _ = spec.forward(q, frame_forces * in_group.to(frame_forces.dtype)[None, :, None])
        result["c0_group_frame_wrenches"] = tau[:, :6] / (mass * 9.81)   # levers, six-group frames only
        tau, _ = spec.forward(q, frame_forces)
        result["c_contact_wrenches"] = tau[:, :6] / (mass * 9.81)
        result["d_dyn_row"] = result["c_contact_wrenches"]
        learner_rows = learner_gate[0].cpu().numpy()
        common = learner_rows & gate & raw_gate[index]
        agreement = gate & raw_gate[index]
        interior = agreement.copy()
        interior[:h] = interior[-h:] = False
        positions, _ = spec.placements(q)
        metadata[f"{mode}_fk_max_error_m"] = float((positions[:, 1:, :3] - tensor(
            data["joints_world"][person, index])).abs().max())
        metadata["mass_dem22_kg"] = float(shaped22.values.body_inertias[:, 0].sum())
        metadata["mass_dem52_kg"] = spec.total_mass
        metadata["finger_mass_kg"] = float(spec.model.values.body_inertias[23:, 0].sum())
        metadata[f"{mode}_gate_only_rows"] = int((gate & ~raw_gate[index]).sum())
        metadata[f"{mode}_endpoint_rows"] = int((gate & ~interior).sum())
        saved[f"{mode}_index"], saved[f"{mode}_common"] = index, common
        saved[f"{mode}_agreement"], saved[f"{mode}_interior"] = agreement, interior
        target = data["base_wrench"][person, index].astype(np.float64) / (mass * 9.81)
        saved[f"{mode}_stored"] = target
        for name, value in result.items():
            mask = gate if name == "d_dyn_row" else learner_rows
            actual = value.cpu().numpy().astype(np.float64)
            saved[f"{mode}_{name}"], saved[f"{mode}_{name}_rows"] = actual, mask
            record = {"mode": mode, "step": name, **measure(actual[mask], target[mask])}
            record.update({f"common_{key}": value for key, value in measure(actual[common], target[common]).items()})
            rows.append(record)
    parent = data["contact_frame_parents"]
    dropped = ~np.isin(parent, np.concatenate([np.asarray(group) for group in GROUPS_52]))
    magnitudes = np.linalg.norm(data["frame_forces"][person], axis=-1)[data["valid_mask"][person]]
    metadata["dropped_magnitude_n_sum"] = float(magnitudes[:, dropped].sum(dtype=np.float64))
    metadata["total_magnitude_n_sum"] = float(magnitudes.sum(dtype=np.float64))
    metadata["off_contact_max_n"] = float(np.linalg.norm(data["frame_forces"][person][
        ~data["frame_contact"][person]], axis=-1).max(initial=0))
    metadata["stored_ungated_max"] = float(np.abs(data["base_wrench"][person, ~raw_gate]).max(initial=0))
    return metadata, saved, rows


def write_report(records: list[dict], people: list[dict], buckets: dict, manifest: dict) -> None:
    """Write reference tables with both row-weighted and equally weighted scene means."""
    lines = ["# Stored-GT RNEA audit", "", f"{manifest['device']} float32; residuals are root-local. Force units: bw; torque: bw·m.",
             "Means are vector norms; agreement MAE is over signed Cartesian components, and correlation pools components.",
             "", "## Sample and aggregation", "",
             f"Seed 20260907; draw 40 scenes without replacement per split, test then train; use the first {manifest['scenes_per_split']} sorted IDs.",
             "Full raw trajectories of every stored person, no 120-frame cap or longest-run selection.",
             "Strided trajectories start at source frame 0, stride=max(1,round(fps/25)); no interpolation.",
             "The CSV identifies every scene/person, fps, stride, step, row count and raw sums for dropped-force magnitude.",
             "Per-person NPZs preserve residual vectors, source indices and every reporting mask. manifest.json records source hashes.",
             "", "## Cumulative ladder", "",
             "a changes density and uses stored mass for both force scaling and residual normalization. b changes only derivatives.",
             "b2 additionally restores the full 52-joint body and stored finger pose; fingers are not massless.",
             "c uses the producer's contact-frame wrenches. d changes only the reporting mask to producer dyn_row.",
             "Earlier rungs use the learner ±2 mask. Stored columns use exactly the same rows as each rung.",
             "Scene means first pool all people/rows within each scene, then weight scenes equally.",
             "", "| Split | Frames | Step | Rows | Force | Torque | Stored force | Stored torque | Scene force | Scene torque |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    findings = ["The measured learner floor contains both a nonzero producer residual and a convention mismatch.",
                "The largest ladder change is restoring all contact-frame wrenches (c); this rung jointly changes",
                "force aggregation, dropped contacts and lever moments, so it does not isolate those effects."]
    for split in ("test", "train"):
        stats = [measure(*[np.concatenate(parts) for parts in zip(*buckets[split, "raw", step])])
                 for step in ("learner", "c_contact_wrenches")]
        findings.append(f"{split}: on identical learner ±2 rows, force/torque means change from "
                        f"{stats[0]['force']:.6f}/{stats[0]['torque']:.6f} to {stats[1]['force']:.6f}/{stats[1]['torque']:.6f}.")
    lines[5:5] = ["", *findings, ""]
    for split in ("test", "train"):
        for mode in ("raw", "strided"):
            for step in STEPS:
                selected = [row for row in records if (row["split"], row["mode"], row["step"]) == (split, mode, step)]
                stats = measure(*[np.concatenate(parts) for parts in zip(*buckets[split, mode, step])])
                scene_groups = defaultdict(list)
                for row in selected:
                    scene_groups[row["scene"]].append(row)
                macro = [np.mean([sum(row[key] * row["rows"] for row in group) / sum(row["rows"] for row in group)
                                  for group in scene_groups.values()]) for key in ("force", "torque")]
                lines.append(f"| {split} | {mode} | {step} | {stats['rows']} | " + " | ".join(
                    f"{value:.6f}" for value in [stats[key] for key in ("force", "torque", "stored_force", "stored_torque")] + macro) + " |")
    lines += ["", "## Reproduction agreement", "",
              "d agreement uses the intersection of sampled dyn_row and the original raw dyn_row, so stored zero-filled gap rows are excluded.",
              "Interior additionally excludes the sampled first/last h rows. Decimated endpoints can differ from raw endpoint treatment.",
              "The `strided_gt1` rows isolate people whose auto stride exceeds one; they are not diluted by stride-one scenes.",
              "", "| Split | Comparison | Rows | Force MAE | Torque MAE | Force max | Torque max | Force r | Torque r |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split in ("test", "train"):
        for comparison in ("raw", "strided", "strided_interior", "strided_gt1", "strided_gt1_interior", "wrong_world_rotation"):
            parts = buckets[split, comparison, "agreement"]
            if not parts:
                continue
            stats = measure(*[np.concatenate(items) for items in zip(*parts)])
            lines.append(f"| {split} | {comparison} | {stats['rows']} | " + " | ".join(
                f"{stats[key]:.8g}" for key in ("mae_force", "mae_torque", "max_force", "max_torque", "corr_force", "corr_torque")) + " |")
    lines += ["", "## Mass, geometry and dropped contacts", "",
              "| Split | People | Tracked force magnitude sum (N) | Dropped sum (N) | Dropped share | Max mass error (kg) | Max FK error (m) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for split in ("test", "train"):
        selected = [person for person in people if person["split"] == split]
        total = sum(person["total_magnitude_n_sum"] for person in selected)
        dropped = sum(person["dropped_magnitude_n_sum"] for person in selected)
        lines.append(f"| {split} | {len(selected)} | {total:.6f} | {dropped:.6f} | {dropped/total:.6%} | "
                     f"{max(abs(person['mass_dem52_kg']-person['mass_stored_kg']) for person in selected):.8g} | "
                     f"{max(person['raw_fk_max_error_m'] for person in selected):.8g} |")
    lines += ["", f"Full-body finger mass per person spans {min(p['finger_mass_kg'] for p in people):.6f}–"
              f"{max(p['finger_mass_kg'] for p in people):.6f} kg. Off-contact force and ungated stored-wrench maxima: "
              f"{max(p['off_contact_max_n'] for p in people):.6g} N and {max(p['stored_ungated_max'] for p in people):.6g} N/N·m.",
              "Dropped share is sum of per-frame force magnitudes on tracked rows for parents outside GROUPS_52,",
              "divided by the sum for all contact frames; forces are not vector-summed before taking magnitudes.",
              "The sampled test share differs from the handoff's approximate corpus-wide 4%; this is not a corpus-wide estimate."]
    lines += ["", "## Verified recipe and discrepancies", "",
              "The reproduction directly calls BVR build_dynamics_spec and DynamicsSpec.forward (no optimizer or force refit).",
              "Body: bh.SMPLX(model_path=the configured SMPLX_NEUTRAL.npz, gender='neutral', num_betas=10,",
              "use_hands=True, use_face=False, density='Dempster', compute_mass=True, dtype=torch.float32);",
              f"contact_frames={_DEFAULT_CONTACT_FRAMES}. Bake the stored per-person betas, then use the stored q (211).",
              "Gravity is normalized stored gravity_world times 9.81. Use stored total_mass×9.81 for normalization.",
              "v[t]=(difference(q[t-h],q[t])+difference(q[t],q[t+h]))/(2h·dt);",
              "a[t]=(difference(q[t],q[t+h])-difference(q[t-h],q[t]))/(h·dt)²; v=a=0 at first/last h rows.",
              "Raw dt=1/fps, h=max(1,round(fps/30)); strided dt=stride/fps, h=1.",
              "Contact positions are NOT stored. BetterHuman rebakes authored contact-frame vertex placements from betas;",
              "BVR placements runs FK plus update_frame_placements. This uses rigid attached frames, not pose-deformed mesh vertices.",
              "Apply frame_forces×frame_contact, rotate force and (contact position−parent origin) into each parent's local frame,",
              "form [f_local, lever_local×f_local], and scatter-add all 35 frames to their 52-joint parents (including fingers).",
              "Do not multiply forces by total_mass again: stored frame_forces are already in newtons.",
              "dyn_row starts as valid_mask and ANDs all IN-BOUNDS neighbours at offsets 1…h on each side.",
              "This differs from a complete-stencil mask: eligible scene endpoints remain counted with quasi-static derivatives.",
              "br.rnea returns tau[:6] in the pelvis free-flyer LOCAL axes [force, torque]; the motion subspace is identity.",
              "No world rotation or moment shift is needed against base_wrench. The wrong-rotation control quantifies this.",
              "", "The handoff's 12-frame description is stale (stored and current authored sets contain 35). The producer body.py comment also says 33.",
              "The learner's 'fingers are massless' comment is false: the full body's finger inertias contain nonzero mass.",
              "Reducing to 22 joints changes the mesh partition/inertias as well as removing finger motion; b2 measures their combined effect.",
              "The handoff's stencil-gating description omits the producer's endpoint exception. These endpoints are included in d.",
              "The task's claim that model=None cannot initialize this loss is also false: getattr(None,'module',None) returns None;",
              "the existing tests use None. This audit follows the requested dummy-object construction.",
              "The reported ~0.209/0.050 test floor uses a different, capped evaluation sample; the learner rows here are full-scene GT.",
              "The stored residual remains nonzero even when exactly reproduced. Convention corrections cannot make it vanish.",
              "Every raw learner result is also checked against the actual _gt_stats method; the three additive statistics agree exactly.",
              "Sampled people with stride>1: " + ", ".join(f"{split}={sum(p['split'] == split and p['auto_stride'] > 1 for p in people)}" for split in ("test", "train")) + ".",
              f"Every sampled auto stride equals raw producer h: {all(p['auto_stride'] == p['raw_h'] for p in people)}.",
              "When stride=raw h, strided h=1 uses identical stencil source frames; both endpoints also align with the producer's zero-derivative bands.",
              "This sample does not establish equivalence when round(fps/25) differs from round(fps/30), or a strided track skips a gap.",
              "", "## Provenance and execution", "",
              f"Command: `{manifest['command']}`. Elapsed seconds: {manifest['elapsed_sec']:.2f}.",
              "Only the new audit script and audit output directory are intentionally written; imports disable Python bytecode and redirect matplotlib/Warp caches here.",
              "A preliminary interactive import, before cache redirection was installed, reported creating /tmp/matplotlib-q32ybeh7; no producer source was edited.",
              "Other processes edited existing learner files during the audit session; this audit did not modify them.",
              f"Inputs changed between this run's start/end hashes: {manifest['changed_inputs_during_run']}.",
              "See manifest.json for repository commits, source SHA-256 hashes, exact sampled scenes and data keys."]
    for split in ("test", "train"):
        lines += ["", f"{split} scenes: " + ", ".join(f"`{item['scene']}`" for item in manifest["sources"] if item["split"] == split) + "."]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Run the fixed-seed scene sample and save the numerical audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scenes-per-split", type=int, default=40)
    args = parser.parse_args()
    if not 1 <= args.scenes_per_split <= 40:
        parser.error("--scenes-per-split must be between 1 and 40")
    torch.set_num_threads(2)
    started = time.monotonic()
    cfg = load_config(REPO / "configs/final_rnea.yaml")
    assert cfg["force_consistency"]["smooth_sec"] == 0
    loss = ForceConsistencyLoss({"model": cfg["model"], "force_consistency": cfg["force_consistency"]}, object(), args.device)
    body = load_body(model_path=cfg["model"]["smplx"]["model_path"], device=args.device)
    dense22 = bh.SMPLX(model_path=cfg["model"]["smplx"]["model_path"], num_betas=10,
                       use_hands=False, use_face=False, density="Dempster", device=args.device)
    rng, records, people, buckets = np.random.default_rng(20260907), [], [], defaultdict(list)
    manifest = {"scenes_per_split": args.scenes_per_split, "command": " ".join([sys.executable, "-B", *sys.argv]),
                "sources": [], "repos": {}, "inputs": {}, "device": args.device}
    for path in (REPO, BVR, REPO.parent / "BetterHuman", REPO.parent / "BetterRobot"):
        manifest["repos"][path.name] = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    source_files = [Path(__file__), REPO / "model/loss/force_consistency.py", REPO / "data/climbing_videos/kindyn.py",
                    BVR / "tools/human_optim/kindyn.py", BVR / "tools/smplx_robot/dynamics.py", BVR / "tools/body.py",
                    Path(_DEFAULT_CONTACT_FRAMES), REPO / "configs/base.yaml", REPO / "configs/final_rnea.yaml",
                    Path(bh.__file__).parent / "bodies/smpl_family/smplx.py",
                    Path(bh.__file__).parent / "bodies/smpl_family/shared.py",
                    Path(bh.__file__).parent / "core/mass.py", Path(br.__file__).parent / "dynamics/rnea.py",
                    Path(br.__file__).parent / "data_model/joint_models/free_flyer.py"]
    manifest["inputs"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    with sqlite3.connect(f"file:{ROOT}/scenes/scenes.db?mode=ro", uri=True) as db, torch.no_grad():
        for split in ("test", "train"):
            eligible = [row[0] for row in db.execute("SELECT scene_id FROM scenes WHERE human_selected=1 "
                "AND vlm_category IN (1,2) AND vlm_rope_supported=0 AND dataset_split=? ORDER BY scene_id", (split,))]
            selected = sorted(rng.choice(eligible, 40, replace=False))[:args.scenes_per_split]
            for scene in selected:
                path = ROOT / f"features/human_optim/{scene[:2]}/{scene[2:4]}/{scene}/kindyn_1.npz"
                with np.load(path, allow_pickle=True) as archive:
                    data = {key: archive[key] for key in archive.files}
                manifest["sources"].append({"split": split, "scene": scene, "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "keys": list(data)})
                targets = load_smplx(scene, path.parent, data["object_ids"], int(data["num_frames"]))
                forces = load_forces(scene, path.parent, data["object_ids"], int(data["num_frames"]))
                for person in range(len(data["object_ids"])):
                    metadata, saved, rows = audit_person(data, person, body, dense22, loss, targets, forces)
                    metadata.update(split=split, scene=scene)
                    people.append(metadata)
                    records.extend({**metadata, **row} for row in rows)
                    np.savez_compressed(OUT / f"{split}_{scene}_p{person}.npz", **saved)
                    for mode in ("raw", "strided"):
                        target = saved[f"{mode}_stored"]
                        for step in STEPS:
                            mask = saved[f"{mode}_{step}_rows"]
                            buckets[split, mode, step].append((saved[f"{mode}_{step}"][mask], target[mask]))
                        for suffix, mask_name in (("", "agreement"), ("_interior", "interior")):
                            mask = saved[f"{mode}_{mask_name}"]
                            pair = (saved[f"{mode}_d_dyn_row"][mask], target[mask])
                            buckets[split, mode + suffix, "agreement"].append(pair)
                            if mode == "strided" and metadata["auto_stride"] > 1:
                                buckets[split, "strided_gt1" + suffix, "agreement"].append(pair)
                    rotation = targets["smplx_root_rot"][person]
                    wrong = np.einsum("tij,tkj->tki", rotation, saved["raw_d_dyn_row"].reshape(-1, 2, 3)).reshape(-1, 6)
                    mask = saved["raw_agreement"]
                    buckets[split, "wrong_world_rotation", "agreement"].append((wrong[mask], saved["raw_stored"][mask]))
                    stats = measure(saved["raw_d_dyn_row"][mask], saved["raw_stored"][mask])
                    print(f"{split} {scene} p{person} rows={stats['rows']} raw MAE={stats['mae_force']:.3g}/{stats['mae_torque']:.3g}", flush=True)
    with (OUT / "per_scene.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    manifest["elapsed_sec"] = time.monotonic() - started
    manifest["changed_inputs_during_run"] = [path for path, digest in manifest["inputs"].items()
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest]
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_report(records, people, buckets, manifest)


if __name__ == "__main__":
    main()
