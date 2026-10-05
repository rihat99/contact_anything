"""Turn Li et al. (CVPR'19) estimator output into prediction dumps the rig scorers read.

The estimator (``data/Estimating-3D-Motion-Forces``) writes one ``<clip>_pred.npz`` per clip
with its SMPL pose (``poses_smpl``, ``trans``, ``betas_smpl``), its 24 joints, and the linear
contact force per limb in its world, which is the camera frame. This script rewrites each into
``<clip>/predictions/estmf/{smplx,contacts,forces_sup}.npz`` (the ``predict_reconstruction.py``
layout, contact set ``kindyn6``), so ``score_rig.py`` and ``score_parkour_rig.py`` score it like
every other row:

* the pose goes onto OUR SMPL-X body: SMPL's 21 body-joint axis angles are SMPL-X's, joint for
  joint, and the shape is the SAM-3D mean shape the estimator's SMPL betas were fitted from;
  the pelvis is then placed exactly where the estimator's pelvis joint is;
* forces become body weights of the reconstructed body (the scorer multiplies the same mass
  back) and act at the estimator's own contact points (sole centre, finger joint);
* contact probabilities are its contact states (1 where it used the limb, else 0).

    python scripts/estmf_dumps.py --dataset climb_wall_3
    python scripts/estmf_dumps.py --dataset parkour
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
_ESTMF = _ROOT.parent / "data" / "Estimating-3D-Motion-Forces" / "validation" / "results"

from model.contact_frames import contact_set                          # noqa: E402
from model.physics import GRAVITY                                     # noqa: E402
from trivial_baselines import clips_climb_wall_3, clips_parkour, dump  # noqa: E402
from viewer.bodies import load_body                                   # noqa: E402

#: The estimator's force channels -> our kindyn6 slot (left_hand, right_hand, left_foot, right_foot).
CHANNEL_SLOT = {0: 2, 1: 3, 2: 0, 3: 1}            # l_sole, r_sole, l_hand, r_hand
#: Rows of the estimator's ``contact_states`` (its joint numbering) behind each channel.
#: A sole channel sums the knee, the ankle and the toes of that leg
#: (``optimizer.py::joint_contact_force_mapping``), so a toe-only frame is a loaded foot.
CHANNEL_JOINT = ([2, 3, 4], [6, 7, 8], [18], [23])
DATASETS = {
    "climb_wall_3": (clips_climb_wall_3, _ESTMF / "climb_wall_3" / "run" / "evaluation" / "ours_s7"),
    "parkour": (clips_parkour, _ESTMF / "ours" / "run" / "evaluation" / "ours_fps25_s7"),
}


def convert(clip: str, tree: Path, pred_path: Path, run: str, body, device) -> dict[str, float]:
    from better_human.bodies.smpl_family.smplx import SMPLXClassic

    pred = np.load(pred_path)
    n = len(np.load(tree / "geometry" / "transform.npz")["frame_indices"])
    nf = min(int(pred["nf"]), n)
    ext = np.asarray(np.load(tree / "geometry" / "transform.npz")["extrinsics"], np.float32)[:n]
    sx = np.load(tree / "sam3d" / "smplx_params.npz")
    betas = np.asarray(sx["betas"][0][np.asarray(sx["valid_mask"][0], bool)], np.float32).mean(0)
    mass = float(np.load(tree / ("human_optim" if (tree / "human_optim" / "kindyn_1.npz").is_file()
                                 else "human_optim_gt_contact") / "kindyn_1.npz",
                         allow_pickle=True)["total_mass"][0])

    poses = np.asarray(pred["poses_smpl"], np.float32)[:nf]
    tensor = lambda a: torch.as_tensor(np.ascontiguousarray(a, np.float32), device=device)  # noqa: E731
    with torch.no_grad():
        shaped = body.with_shape(betas=tensor(betas)[None].expand(nf, -1))
        q = shaped.from_classic(SMPLXClassic(
            global_orient=tensor(poses[:, :3]), body_pose=tensor(poses[:, 3:66]),
            transl=tensor(np.zeros((nf, 3))), left_hand_pose=tensor(np.zeros((nf, 45))),
            right_hand_pose=tensor(np.zeros((nf, 45))), num_pca_comps=None))
        pelvis = shaped.fk(q).joint_pose_world[:, 1, :3]
        q[:, :3] += tensor(pred["joints_24"][:nf, 0]) - pelvis          # pelvis where theirs is
        joints_cam = shaped.fk(q).joint_pose_world[:, 1:23, :3]
    rot_wc = torch.as_tensor(ext[:nf, :3, :3], device=device).transpose(-1, -2)
    t_cw = torch.as_tensor(ext[:nf, :3, 3], device=device)
    joints_world = torch.einsum("tij,tkj->tki", rot_wc, joints_cam - t_cw[:, None])

    slots = contact_set("kindyn6")
    anchor_cam = joints_cam[:, list(slots.parent_joint22)].cpu().numpy()   # (nf, 6, 3)
    forces_world = np.zeros((nf, 6, 3))
    contact6 = np.zeros((nf, 6), bool)
    force_n = np.asarray(pred["forces_world"], np.float64)[:nf]
    points = np.asarray(pred["force_points_world"], np.float64)[:nf]
    states = np.asarray(pred["contact_states"])[:nf]
    for channel, slot in CHANNEL_SLOT.items():
        anchor_cam[:, slot] = points[:, channel]
        contact6[:, slot] = (states[:, CHANNEL_JOINT[channel]] > 0).any(1)
        # the estimator's world is the camera; the dump's forces are in the tree's world
        f_cam = torch.as_tensor(force_n[:, channel], dtype=torch.float32, device=device)
        forces_world[:, slot] = torch.einsum("tij,tj->ti", rot_wc, f_cam).cpu().numpy() / (mass * GRAVITY)

    covered = np.zeros(n, bool)
    covered[:nf] = True
    pad = lambda a: np.concatenate([a, np.full((n - nf, *a.shape[1:]), np.nan, a.dtype)])  # noqa: E731
    sx_out = {"q_cam": pad(q.cpu().numpy()), "betas": np.broadcast_to(betas, (n, 10)),
              "joints_cam": pad(joints_cam.cpu().numpy()),
              "joints_world": pad(joints_world.cpu().numpy()), "covered": covered}
    identity = {"limbs": np.asarray(slots.slot_names), "contact_set": "kindyn6",
                "object_ids": np.asarray(sx["object_ids"], np.int32),
                "frame_indices": np.arange(n, dtype=np.int32), "valid_mask": covered[None],
                "tracked": covered[None], "fps": np.float32(pred["fps"]), "stride": np.int32(1),
                "source": f"Li et al. CVPR'19 estimator, {pred_path}"}
    dump(tree / "predictions" / run, sx_out, pad(contact6.astype(np.float32)) > 0.5,
         pad(forces_world.astype(np.float32)), pad(anchor_cam.astype(np.float32)), identity)
    loaded = contact6.any(1)
    return {"frames": nf, "loaded": int(loaded.sum()),
            "|f| bw": float(np.linalg.norm(forces_world, axis=-1).sum(1)[loaded].mean())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--results", type=Path, default=None,
                        help="directory of <clip>_pred.npz (default per dataset)")
    parser.add_argument("--run", default="estmf", help="dump name under <clip>/predictions/")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    model_path = yaml.safe_load((_ROOT / "configs" / "base.yaml").read_text())["model"]["smplx"]["model_path"]
    device = torch.device(args.device)
    body = load_body(device, model_path)
    list_clips, results = DATASETS[args.dataset]
    results = args.results or results
    done = 0
    for clip, tree in list_clips():
        path = results / f"{clip}_pred.npz"
        if not path.is_file():
            print(f"{clip}: no {path.name}, skipped")
            continue
        stats = convert(clip, tree, path, args.run, body, device)
        done += 1
        print(f"{clip}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                     for k, v in stats.items()), flush=True)
    print(f"{done} clip(s) -> predictions/{args.run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
