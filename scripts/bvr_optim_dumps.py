"""BVR's optimisation (``human_optim/kindyn_1.npz``) as a prediction dump of an out-tree.

Writes ``<tree>/predictions/<run>/{smplx,contacts,forces_sup}.npz`` in the
``scripts/predict_reconstruction.py`` layout (contact set ``frames35``: the 35 contact frames
are the optimisation's own) so ``scripts/render_wild_overlays.py`` and the viewer draw the
optimised body and its forces the way they draw the learned ones: the kindyn body's world
joints mapped into every camera with the tree's extrinsics, its contact labels as 0 / 1
probabilities, its newtons in body weights of the mass it solved with.

    .venv/bin/python scripts/bvr_optim_dumps.py --out-root ../data/willd_videos/bvr_out \\
        --videos ../data/willd_videos/videos --run bvr_optim [--stems ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.contact_frames import contact_set                          # noqa: E402
from model.physics import GRAVITY                                     # noqa: E402


def q_to_camera(q_world: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    """``q`` (N, 211) of BetterHuman's world body under the per-frame ``cam_from_world`` (N, 4, 4)."""
    rot = extrinsics[:, :3, :3]
    q_cam = q_world.copy()
    q_cam[:, :3] = np.einsum("tij,tj->ti", rot, q_world[:, :3]) + extrinsics[:, :3, 3]
    q_cam[:, 3:7] = (Rotation.from_matrix(rot) * Rotation.from_quat(q_world[:, 3:7])).as_quat()
    return q_cam


def convert(tree: Path, video: Path, run: str) -> dict[str, float]:
    solve = np.load(tree / "human_optim" / "kindyn_1.npz", allow_pickle=True)
    camera = np.load(tree / "geometry" / "transform.npz", allow_pickle=True)
    extrinsics = np.asarray(camera["extrinsics"], np.float32)
    n = len(extrinsics)
    if int(solve["num_frames"]) != n:
        raise ValueError(f"{tree.name}: kindyn has {int(solve['num_frames'])} frames, the camera {n}")
    slots = contact_set("frames35")
    names = [str(s) for s in solve["contact_frame_names"]]
    order = [names.index(s) for s in slots.slot_names]
    parents = np.asarray(slots.parent_joint52, np.int64)

    valid = np.asarray(solve["valid_mask"], bool)                                   # (P, N)
    joints_world = np.asarray(solve["joints_world"], np.float32)                     # (P, N, 52, 3)
    joints_cam = (np.einsum("tij,ptkj->ptki", extrinsics[:, :3, :3], joints_world)
                  + extrinsics[None, :, None, :3, 3])
    q_cam = np.stack([q_to_camera(np.asarray(q, np.float32), extrinsics) for q in solve["q"]])
    mass = np.asarray(solve["total_mass"], np.float32)                               # (P,)
    forces_world = (np.asarray(solve["frame_forces"], np.float32)[:, :, order]
                    / (mass * GRAVITY)[:, None, None, None])
    contact = np.asarray(solve["frame_contact"], bool)[:, :, order]
    hidden = ~valid[:, :, None, None]
    forces_world = np.where(hidden, np.nan, forces_world)
    joints_cam = np.where(hidden, np.nan, joints_cam)

    identity = {"limbs": np.asarray(slots.slot_names), "contact_set": "frames35",
                "object_ids": np.asarray(solve["object_ids"], np.int32),
                "frame_indices": np.asarray(camera["frame_indices"], np.int32),
                "valid_mask": valid, "tracked": valid, "fps": np.float32(solve["fps"]),
                "stride": np.int32(1), "source_video": str(video.resolve()),
                "source": f"BVR human optimisation, {tree / 'human_optim' / 'kindyn_1.npz'}"}
    pred_dir = tree / "predictions" / run
    pred_dir.mkdir(parents=True, exist_ok=True)
    betas = np.broadcast_to(np.asarray(solve["betas"], np.float32)[:, None], (len(mass), n, 10))
    np.savez_compressed(pred_dir / "smplx.npz", q_cam=q_cam, betas=betas, joints_cam=joints_cam,
                        joints_world=np.where(hidden, np.nan, joints_world),
                        pelvis_cam=joints_cam[:, :, 0], covered=valid, hands=np.bool_(True),
                        **identity)
    probs = contact.astype(np.float32)
    np.savez_compressed(pred_dir / "contacts.npz", probs=probs, contacts=contact,
                        threshold=np.float32(0.5), **identity)
    np.savez_compressed(pred_dir / "forces_sup.npz", forces=forces_world, forces_world=forces_world,
                        anchor_cam=joints_cam[:, :, parents], units="body_weight",
                        force_frame="world", contact_probs=probs, **identity)
    loaded = contact.any(-1) & valid
    return {"people": len(mass), "frames": n, "loaded": int(loaded.sum()),
            "mass": float(mass.mean()),
            "|f| bw": float(np.nansum(np.linalg.norm(forces_world, axis=-1), -1)[loaded].mean())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-root", type=Path, required=True, help="the BVR out-trees")
    parser.add_argument("--videos", type=Path, required=True, help="<stem>.mp4 per tree")
    parser.add_argument("--run", default="bvr_optim", help="dump name under <tree>/predictions/")
    parser.add_argument("--stems", nargs="*", default=None, help="default: every tree with a kindyn_1")
    args = parser.parse_args()
    trees = ([args.out_root / s for s in args.stems] if args.stems else
             sorted(d for d in args.out_root.iterdir() if (d / "human_optim" / "kindyn_1.npz").is_file()))
    for tree in trees:
        stats = convert(tree, args.videos / f"{tree.name}.mp4", args.run)
        print(f"{tree.name}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                          for k, v in stats.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
