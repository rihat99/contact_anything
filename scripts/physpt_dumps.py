"""Turn PhysPT (Zhang et al. CVPR'24) output into prediction dumps the rig scorer reads.

PhysPT (``../other/PhysPT``) was run from OUR SAM-3D initialisation (``scripts/
parkour_physpt_inputs.py`` in BVR) and writes ``<dataset>/<clip>_output.npz``: the SMPL meshes
of its kinematic and physics stages in its own world, and a ground-reaction force on each
of its 29 contact vertices (newtons, ``T-2`` frames). It saves no pose parameters, so this
script rebuilds each clip as ``<clip>/predictions/physpt/{smplx,contacts,forces_sup}.npz``
(the ``predict_reconstruction.py`` layout, contact set ``kindyn6``) in three steps:

* PhysPT's world is brought into our camera frame by ONE rotation per clip, fitted between
  its kinematic-stage joints and our initialisation's with every frame centred (it keeps our
  orientation but predicts its own root path, which drifts), plus one mean offset; the
  centred residual printed per clip (~30-40 mm, mostly SMPL-vs-SMPL-X joint placement) is a
  check;
* our SMPL-X body (SAM-3D mean shape) is posed onto its physics-stage joints: starting from
  the initialisation, pelvis and every joint rotation are refined by Adam on the 22 shared
  joints; the per-clip joint RMSE is printed;
* its vertex forces are summed per foot and applied at their force-weighted centroid, in
  body weights of the reconstructed body; a foot is "in contact" while it carries more
  than ``LOADED_N``. PhysPT models ground contact only, so the hands never carry force.

    python scripts/physpt_dumps.py --dataset parkour
    python scripts/physpt_dumps.py --dataset climb_wall_3
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import roma
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
PHYSPT = _ROOT.parent / "other" / "PhysPT"

from model.contact_frames import contact_set                                    # noqa: E402
from model.physics import GRAVITY                                               # noqa: E402
from trivial_baselines import BODY_Q, DATASETS, dump, init_body                 # noqa: E402
from utils.geometry import smplx_q                                              # noqa: E402
from viewer.bodies import load_body                                             # noqa: E402

#: kindyn6 slot of each foot -> the SMPL parts (ankle, foot) its contact vertices belong to.
FOOT_PARTS = {2: {7, 10}, 3: {8, 11}}
#: A foot is in contact while PhysPT's force on it exceeds this (N).
LOADED_N = 20.0
FIT_ITERS = 400
SHARED = 22


def physpt_model() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PhysPT's SMPL joint regressor ``(24, 6890)``, its 29 contact vertices and their parts."""
    with open(PHYSPT / "assets" / "data" / "smpl" / "neutral.pkl", "rb") as handle:
        regressor = np.asarray(pickle.load(handle)["V_regressor"], np.float64)
    with open(PHYSPT / "assets" / "data" / "physics.pkl", "rb") as handle:
        physics = pickle.load(handle)
    vertices = np.asarray(physics["contact_vertexidxall"], np.int64)
    parts = np.concatenate([[part] * len(v) for part, v in enumerate(physics["contact_vertexidx"])])
    return regressor, vertices, parts.astype(np.int64)


def rigid(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch: ``R, t`` with ``target ~ source @ R.T + t`` over ``(M, 3)`` point pairs."""
    src_mean, dst_mean = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - src_mean).T @ (target - dst_mean))
    v = vt.T
    rotation = v @ np.diag([1.0, 1.0, np.linalg.det(v) * np.linalg.det(u)]) @ u.T
    return rotation, dst_mean - src_mean @ rotation.T


def fit_pose(shaped, q0: torch.Tensor, target22: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Refine ``q0 (T, 211)`` so the body's 22 joints meet ``target22 (T, 22, 3)``; RMSE in mm."""
    n = q0.shape[0]
    pelvis0, root0 = q0[:, :3], roma.unitquat_to_rotmat(q0[:, 3:7])
    joints0 = roma.unitquat_to_rotmat(q0[:, 7:BODY_Q].reshape(n, 21, 4))
    deltas = [torch.zeros(n, 3), torch.zeros(n, 3), torch.zeros(n, 21, 3)]
    deltas = [d.to(q0.device).requires_grad_(True) for d in deltas]

    def compose() -> torch.Tensor:
        q91 = smplx_q(pelvis0 + deltas[0], roma.rotvec_to_rotmat(deltas[1]) @ root0,
                      roma.rotvec_to_rotmat(deltas[2]) @ joints0)
        return torch.cat([q91, q0[:, BODY_Q:]], dim=-1)

    def error(q: torch.Tensor) -> torch.Tensor:
        return shaped.fk(q).joint_pose_world[:, 1:SHARED + 1, :3] - target22

    optimizer = torch.optim.Adam(deltas, lr=0.02)
    for _ in range(FIT_ITERS):
        optimizer.zero_grad()
        loss = error(compose()).square().sum(-1).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        q = compose()
        rmse = error(q).square().sum(-1).mean().sqrt().item() * 1e3
    return q.detach(), rmse


def convert(clip: str, tree: Path, dataset: str, run: str, body, device) -> dict[str, float] | None:
    path = PHYSPT / dataset / f"{clip}_output.npz"
    if not path.is_file():
        return None
    saved = np.load(path)
    transform = np.load(tree / "geometry" / "transform.npz")
    n = len(transform["frame_indices"])
    ext = np.asarray(transform["extrinsics"], np.float32)[:n]
    nf = min(len(saved["frame_idx"]), n)
    mass = float(np.load(tree / "human_optim" / "kindyn_1.npz",
                         allow_pickle=True)["total_mass"][0])
    regressor, vertices, parts = physpt_model()
    kin_joints = np.einsum("jv,tvc->tjc", regressor, np.asarray(saved["kinematics_vertices_world"], np.float64))[:nf]
    phys_mesh = np.asarray(saved["physics_vertices_world"], np.float64)[:nf]
    phys_joints = np.einsum("jv,tvc->tjc", regressor, phys_mesh)
    grf = np.zeros((nf, len(vertices), 3))
    force = np.asarray(saved["pred_grf"][0], np.float64)
    grf[:len(force)] = force[:nf]

    q0, betas, _ = init_body(tree, body, device)
    tensor = lambda a: torch.as_tensor(np.ascontiguousarray(a, np.float32), device=device)  # noqa: E731
    q0 = tensor(q0[:nf])
    with torch.no_grad():
        shaped = body.with_shape(betas=tensor(betas)[None].expand(nf, -1))
        init22 = shaped.fk(q0).joint_pose_world[:, 1:SHARED + 1, :3].cpu().numpy().astype(np.float64)
    # PhysPT keeps our orientation but predicts its own root trajectory (drifts by up to a
    # metre), so the rotation comes from a per-frame-centred fit and the translation is one
    # mean offset; its trajectory is left as it predicted it.
    kin22, kin_centre = kin_joints[:, :SHARED], kin_joints[:, :SHARED].mean(1, keepdims=True)
    init_centre = init22.mean(1, keepdims=True)
    rot, _ = rigid((kin22 - kin_centre).reshape(-1, 3), (init22 - init_centre).reshape(-1, 3))
    shift = (init_centre - kin_centre @ rot.T).mean(0)[0]
    rigid_mm = float(np.linalg.norm((kin22 - kin_centre) @ rot.T - (init22 - init_centre), axis=-1).mean() * 1e3)
    target22 = phys_joints[:, :SHARED] @ rot.T + shift
    points_cam = phys_mesh[:, vertices] @ rot.T + shift                  # (nf, 29, 3)
    force_cam = grf @ rot.T

    q, fit_mm = fit_pose(shaped, q0, tensor(target22))
    with torch.no_grad():
        joints_cam = shaped.fk(q).joint_pose_world[:, 1:SHARED + 1, :3]
    rot_wc = torch.as_tensor(ext[:nf, :3, :3], device=device).transpose(-1, -2)
    t_cw = torch.as_tensor(ext[:nf, :3, 3], device=device)
    joints_world = torch.einsum("tij,tkj->tki", rot_wc, joints_cam - t_cw[:, None])

    slots = contact_set("kindyn6")
    anchor_cam = joints_cam[:, list(slots.parent_joint22)].cpu().numpy().astype(np.float64)
    forces_cam6 = np.zeros((nf, 6, 3))
    for slot, part_set in FOOT_PARTS.items():
        columns = [c for c, part in enumerate(parts) if part in part_set]
        f = force_cam[:, columns]
        forces_cam6[:, slot] = f.sum(1)
        weight = np.linalg.norm(f, axis=-1, keepdims=True)
        loaded = weight.sum(1)[:, 0] > 0
        anchor_cam[loaded, slot] = (points_cam[:, columns] * weight).sum(1)[loaded] / weight.sum(1)[loaded]
    contact6 = np.linalg.norm(forces_cam6, axis=-1) > LOADED_N
    forces_world = torch.einsum("tij,tkj->tki", rot_wc,
                                torch.as_tensor(forces_cam6, dtype=torch.float32, device=device)
                                ).cpu().numpy() / (mass * GRAVITY)

    covered = np.zeros(n, bool)
    covered[:nf] = True
    pad = lambda a: np.concatenate([a, np.full((n - nf, *a.shape[1:]), np.nan, a.dtype)])  # noqa: E731
    sx = np.load(tree / "sam3d" / "smplx_params.npz")
    sx_out = {"q_cam": pad(q.cpu().numpy()), "betas": np.broadcast_to(betas, (n, 10)),
              "joints_cam": pad(joints_cam.cpu().numpy()),
              "joints_world": pad(joints_world.cpu().numpy()), "covered": covered}
    identity = {"limbs": np.asarray(slots.slot_names), "contact_set": "kindyn6",
                "object_ids": np.asarray(sx["object_ids"], np.int32),
                "frame_indices": np.arange(n, dtype=np.int32), "valid_mask": covered[None],
                "tracked": covered[None], "fps": np.float32(transform["fps"]), "stride": np.int32(1),
                "source": f"PhysPT CVPR'24 from our SAM-3D init, {path}; pose fitted to its joints"}
    dump(tree / "predictions" / run, sx_out, pad(contact6.astype(np.float32)) > 0.5,
         pad(forces_world.astype(np.float32)), pad(anchor_cam.astype(np.float32)), identity)
    loaded = contact6.any(1)
    return {"frames": nf, "rigid mm": rigid_mm, "fit mm": fit_mm, "loaded": int(loaded.sum()),
            "|f| bw": float(np.linalg.norm(forces_world, axis=-1).sum(1)[loaded].mean()) if loaded.any() else 0.0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="parkour", choices=sorted(DATASETS))
    parser.add_argument("--run", default="physpt", help="dump name under <clip>/predictions/")
    parser.add_argument("--clips", default=None, help="comma-separated clip names (default: all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    model_path = yaml.safe_load((_ROOT / "configs" / "base.yaml").read_text())["model"]["smplx"]["model_path"]
    device = torch.device(args.device)
    body = load_body(device, model_path)
    wanted = set(args.clips.split(",")) if args.clips else None
    done = 0
    for clip, tree in DATASETS[args.dataset][0]():
        if wanted and clip not in wanted:
            continue
        stats = convert(clip, tree, args.dataset, args.run, body, device)
        if stats is None:
            print(f"{clip}: no PhysPT output, skipped")
            continue
        done += 1
        print(f"{clip}: " + ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"
                                     for k, v in stats.items()), flush=True)
    print(f"{done} clip(s) -> predictions/{args.run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
