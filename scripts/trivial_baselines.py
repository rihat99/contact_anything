"""Two trivial force baselines on a measured rig: a smoothed init body plus the measured contacts.

Reviewers ask what a force estimate is worth against the simplest thing one could do with
the same inputs. Both baselines here take the per-frame SAM-3D-Body fit (the initialisation
every one of our rows starts from), smooth it in time with one Gaussian kernel, and take the
contacts from ``--contacts`` — no optimisation, no learning:

* **equal** — the body weight split equally between the limbs in contact, each force
  pointing straight up (against gravity). The static answer.
* **rnea** — the six root-wrench equations of the smoothed body (inverse dynamics on the
  22-joint SMPL-X, :class:`model.physics.RootWrench`) solved for the smallest force over the
  limbs in contact, one 3-D force per limb. Nothing else: no cones, no smoothness, no torque
  cost. Where no limb touches the force is zero and the residual stays.

Each limb's force acts at one joint: the wrist for a hand, the toe base (``left_foot`` /
``right_foot``) for a foot. The output is a prediction dump in the layout of
``scripts/predict_reconstruction.py`` (``<clip>/predictions/<run>/{smplx,contacts,forces_sup}.npz``,
contact set ``kindyn6``, forces in body weights, world frame), so ``scripts/score_rig.py``
and its Parkour twin score the rows exactly like the learned models.

``--contacts measured`` reads the rig's own sensors (climb_wall_3: the board labels;
Parkour: the dataset's contact file); ``--contacts interactvlm`` reads InteractVLM's
per-vertex contact probability instead (``others/interactvlm_<query>/<clip>/contacts_vertex.npz``
next to each rig's trees, ``--query`` naming the prompt it was run with), folded onto the four
limbs exactly as BVR's ``scripts/diagnostics/score_gt_contacts.py`` folds it under ``lbs`` /
``frac10`` — every SMPL vertex belongs to the joint of its largest skinning weight, and a limb
is in contact when a tenth of its vertices are above the file's threshold. Frames InteractVLM
has no row for are left free.

    python scripts/trivial_baselines.py --dataset climb_wall_3 --contacts interactvlm
    python scripts/trivial_baselines.py --dataset parkour --contacts interactvlm --query scene
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import roma
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
_BVR = _ROOT.parent / "BetterVideoReconstruction"
_PARKOUR = _ROOT.parent / "data" / "Parkour-dataset" / "processed"
#: Where each rig keeps the published models' predictions (``interactvlm_<query>/<clip>/``).
_OTHERS = {"climb_wall_3": _BVR / "peter" / "climb_wall_3_gt" / "others",
           "parkour": _PARKOUR / "others"}
_SMPL_NEUTRAL = _ROOT.parent / "BetterHuman" / "models" / "smpl" / "SMPL_NEUTRAL.npz"

from model.contact_frames import contact_set                          # noqa: E402
from model.physics import RootWrench                                  # noqa: E402
from model.refiner import gaussian_smooth, smooth_rotations           # noqa: E402
from utils.geometry import smplx_q                                    # noqa: E402
from viewer.bodies import Q_FULL, load_body                           # noqa: E402

#: The rig's four measured limbs, in the order the sensor files use.
LIMBS = ("left_hand", "right_hand", "left_foot", "right_foot")
BODY_Q = 91
#: Singular values below this fraction of the largest are treated as zero in the least-norm
#: solve: two contact points cannot produce a moment about the line through them, and the
#: float32 residual would otherwise turn that null direction into a force of millions.
RCOND = 1e-3
RUNS = ("baseline_equal", "baseline_rnea")
#: SMPL skinning joints of each limb (BVR ``score_gt_contacts.py``'s ``lbs`` grouping).
LBS_JOINTS = {"left_hand": [20, 22], "right_hand": [21, 23],
              "left_foot": [7, 10], "right_foot": [8, 11]}
#: ``frac10``: a limb is in contact when a tenth of its vertices are above the threshold.
VERTEX_FRACTION = 0.10


def clips_climb_wall_3() -> list[tuple[str, Path]]:
    gt = _BVR / "peter" / "climb_wall_3_gt" / "human"
    return [(d.name, _BVR / "peter" / "out_climb_wall_3_single" / d.name)
            for d in sorted(gt.iterdir()) if d.is_dir() and not d.name.startswith("_")]


def contacts_climb_wall_3(clip: str, n: int) -> np.ndarray:
    """``(n, 4)`` board contact labels of the three-camera reference, padded to the video."""
    ref = np.load(_BVR / "peter" / "climb_wall_3_gt" / "human" / clip / "human_optim"
                  / "contacts_1.npz", allow_pickle=True)
    names = [str(x) for x in ref["limb_names"]]
    contact = np.asarray(ref["limb_contact"][0], bool)[:, [names.index(l) for l in LIMBS]]
    pad = np.repeat(contact[-1:], max(n - len(contact), 0), axis=0)
    return np.concatenate([contact, pad])[:n]


def contacts_interactvlm(root: Path, clip: str, n: int) -> np.ndarray:
    """``(n, 4)`` InteractVLM contact, folded onto the limbs by ``lbs`` / ``frac10``."""
    owner = np.asarray(np.load(_SMPL_NEUTRAL, allow_pickle=True)["weights"]).argmax(1)
    groups = {limb: np.flatnonzero(np.isin(owner, ids)) for limb, ids in LBS_JOINTS.items()}
    vertex = np.load(root / clip / "contacts_vertex.npz", allow_pickle=True)
    above = np.asarray(vertex["contact_smpl"], np.float32) > float(vertex["threshold"])
    hit = np.stack([above[:, groups[limb]].mean(1) >= VERTEX_FRACTION for limb in LIMBS], axis=1)
    frames = np.asarray(vertex["frame_indices"], int)
    keep = frames < n
    contact = np.zeros((n, 4), bool)
    contact[frames[keep]] = hit[keep]
    return contact


def clips_parkour() -> list[tuple[str, Path]]:
    return [(d.name, d) for d in sorted((_PARKOUR / "out").iterdir())
            if (d / "sam3d" / "smplx_params.npz").is_file()]


def contacts_parkour(clip: str, n: int) -> np.ndarray:
    truth = np.load(_PARKOUR / clip / "contacts.npz")
    names = [str(x) for x in truth["limbs"]]
    return np.asarray(truth["contacts"], bool)[:n, [names.index(l) for l in LIMBS]]


DATASETS = {"climb_wall_3": (clips_climb_wall_3, contacts_climb_wall_3),
            "parkour": (clips_parkour, contacts_parkour)}


def init_body(tree: Path, body, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SAM-3D's classic camera-frame fit -> ``(q_cam (N, 211), betas (10,), valid (N,))``.

    Frames SAM-3D did not fit hold the nearest fitted pose, so the smoothing below never
    sees a hole; they stay marked invalid.
    """
    from better_human.bodies.smpl_family.smplx import SMPLXClassic

    sx = np.load(tree / "sam3d" / "smplx_params.npz")
    valid = np.asarray(sx["valid_mask"][0], bool)
    idx = np.flatnonzero(valid)
    nearest = idx[np.clip(np.searchsorted(idx, np.arange(len(valid))), 0, len(idx) - 1)]
    betas = np.asarray(sx["betas"][0][idx], np.float32).mean(0)
    tensor = lambda key: torch.as_tensor(np.ascontiguousarray(sx[key][0][nearest], np.float32),  # noqa: E731
                                         device=device)
    with torch.no_grad():
        shaped = body.with_shape(betas=torch.as_tensor(betas, device=device)[None].expand(len(valid), -1))
        q_cam = shaped.from_classic(SMPLXClassic(
            global_orient=tensor("global_orient"), body_pose=tensor("body_pose"),
            transl=tensor("transl"), left_hand_pose=tensor("left_hand_pose"),
            right_hand_pose=tensor("right_hand_pose"), num_pca_comps=None))
    return q_cam.cpu().numpy(), betas, valid


def smooth_q(q_cam: np.ndarray, valid: np.ndarray, fps: float, sigma: float, device):
    """Gaussian-smooth the pelvis path and every joint rotation of ``q (N, 211)``."""
    q = torch.as_tensor(q_cam, device=device)[None]
    seconds = torch.arange(q.shape[1], device=device, dtype=torch.float32)[None] / fps
    ok = torch.as_tensor(valid, device=device)[None]
    pelvis = gaussian_smooth(q[..., :3], seconds, ok, sigma)
    root = smooth_rotations(roma.unitquat_to_rotmat(q[..., 3:7]), seconds, ok, sigma)
    joints = smooth_rotations(roma.unitquat_to_rotmat(q[..., 7:BODY_Q].reshape(1, -1, 21, 4)),
                              seconds, ok, sigma)
    q91 = smplx_q(pelvis[0], root[0], joints[0])
    return torch.cat([q91, q[0, :, BODY_Q:]], dim=-1), pelvis[0], root[0], joints[0]


def least_norm_forces(residual_of, contact: np.ndarray, slots: int) -> np.ndarray:
    """``(N, slots, 3)`` body-weight forces zeroing the root wrench with the least norm.

    ``residual_of(f)`` maps a batch of force fields ``(B, N, slots, 3)`` to residuals
    ``(B, N, 6)``; it is affine in ``f``, so one zero field and ``3 * slots`` unit fields
    give the per-frame system ``r0 + A f = 0``, solved by the pseudo-inverse over the
    columns of the limbs in contact.
    """
    n = contact.shape[0]
    probes = torch.zeros(1 + 3 * slots, n, slots, 3)
    for k in range(3 * slots):
        probes[1 + k, :, k // 3, k % 3] = 1.0
    r = residual_of(probes)                                            # (1 + 3K, N, 6)
    r0, columns = r[0], np.transpose(r[1:] - r[0], (1, 2, 0))          # (N, 6), (N, 6, 3K)
    forces = np.zeros((n, slots, 3))
    for t in range(n):
        on = np.flatnonzero(np.repeat(contact[t], 3))
        if on.size == 0 or not np.isfinite(r0[t]).all():
            continue
        forces[t].reshape(-1)[on] = np.linalg.lstsq(columns[t][:, on], -r0[t], rcond=RCOND)[0]
    return forces


def dump(pred_dir: Path, sx: dict, contact6: np.ndarray, forces_world: np.ndarray,
         anchor_cam: np.ndarray, identity: dict) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    one = lambda a: np.asarray(a, np.float32)[None]  # noqa: E731
    np.savez_compressed(pred_dir / "smplx.npz", q_cam=one(sx["q_cam"]), betas=one(sx["betas"]),
                        joints_cam=one(sx["joints_cam"]), joints_world=one(sx["joints_world"]),
                        pelvis_cam=one(sx["joints_cam"][:, 0]), covered=sx["covered"][None],
                        hands=np.bool_(False), **identity)
    probs = contact6.astype(np.float32)
    np.savez_compressed(pred_dir / "contacts.npz", probs=probs[None], contacts=contact6[None],
                        threshold=np.float32(0.5), **identity)
    np.savez_compressed(pred_dir / "forces_sup.npz", forces=one(forces_world),
                        forces_world=one(forces_world), anchor_cam=one(anchor_cam),
                        units="body_weight", force_frame="world", contact_probs=probs[None],
                        **identity)


def run_clip(clip: str, tree: Path, contact4: np.ndarray, sigma: float, body, wrench: RootWrench,
             device, source: str) -> dict[str, float]:
    transform = np.load(tree / "geometry" / "transform.npz")
    ext = np.asarray(transform["extrinsics"], np.float32)
    fps = float(transform["fps"])
    n = len(ext)
    gravity = np.asarray(np.load(tree / "geocalib" / "gravity.npz")["gravity_world"], np.float32)
    gravity = gravity / np.linalg.norm(gravity)
    slots = contact_set("kindyn6")
    parent22 = torch.as_tensor(np.asarray(slots.parent_joint22, np.int64), device=device)

    q_raw, betas, valid = init_body(tree, body, device)
    q_cam, pelvis_cam, root_cam, body_rot = smooth_q(q_raw[:n], valid[:n], fps, sigma, device)
    with torch.no_grad():
        fk = body.with_shape(betas=torch.as_tensor(betas, device=device)[None].expand(n, -1)).fk(q_cam)
    joints_cam = fk.joint_pose_world[:, 1:23, :3]                                  # (N, 22, 3)
    rot_wc = torch.as_tensor(ext[:, :3, :3], device=device).transpose(-1, -2)
    t_cw = torch.as_tensor(ext[:, :3, 3], device=device)
    to_world = lambda p: torch.einsum("tij,tkj->tki", rot_wc, p - t_cw[:, None])  # noqa: E731
    joints_world = to_world(joints_cam)
    anchor_cam = joints_cam[:, list(slots.parent_joint22)]                           # (N, 6, 3)
    anchor_world = joints_world[:, list(slots.parent_joint22)]
    root_world = rot_wc @ root_cam

    seconds = torch.arange(n, device=device, dtype=torch.float32)[None] / fps
    ok = torch.as_tensor(valid[:n], device=device)[None]
    down = torch.as_tensor(gravity, device=device)[None]

    def residual_of(forces: torch.Tensor) -> np.ndarray:
        b = forces.shape[0]
        expand = lambda x: x[None].expand(b, *x.shape).contiguous()  # noqa: E731
        with torch.no_grad():
            res_f, res_t, _, _ = wrench.residual(
                expand(joints_world[:, 0]), expand(root_world), expand(body_rot),
                torch.as_tensor(betas, device=device)[None].expand(b, -1),
                forces.to(device), expand(anchor_world), parent22,
                down.expand(b, -1), seconds.expand(b, -1), ok.expand(b, -1))
        return torch.cat([res_f, res_t], dim=-1).cpu().numpy()

    contact6 = np.concatenate([contact4, np.zeros((n, 2), bool)], axis=1)
    up = -gravity
    count = np.maximum(contact4.sum(1, keepdims=True), 1)
    equal = np.zeros((n, 6, 3))
    equal[:, :4] = (contact4 / count)[..., None] * up
    rnea = least_norm_forces(residual_of, contact6, 6)

    sx = {"q_cam": q_cam.cpu().numpy(), "betas": np.broadcast_to(betas, (n, 10)),
          "joints_cam": joints_cam.cpu().numpy(), "joints_world": joints_world.cpu().numpy(),
          "covered": valid[:n]}
    identity = {"limbs": np.asarray(slots.slot_names), "contact_set": "kindyn6",
                "object_ids": np.asarray(np.load(tree / "sam3d" / "smplx_params.npz")["object_ids"],
                                         np.int32),
                "frame_indices": np.arange(n, dtype=np.int32), "valid_mask": valid[None, :n],
                "tracked": valid[None, :n], "fps": np.float32(fps), "stride": np.int32(1),
                "smooth_sigma_s": np.float32(sigma), "gravity_world": gravity,
                "source": f"SAM-3D-Body init, Gaussian-smoothed, {source} (trivial baseline)"}
    for run, forces in (("baseline_equal", equal), ("baseline_rnea", rnea)):
        dump(tree / "predictions" / run, sx, contact6, forces, anchor_cam.cpu().numpy(), identity)

    check = residual_of(torch.stack([torch.zeros(n, 6, 3), torch.as_tensor(equal, dtype=torch.float32),
                                     torch.as_tensor(rnea, dtype=torch.float32)]))
    loaded = contact4.any(1)
    return {"frames": n, "loaded": int(loaded.sum()),
            "|f| equal": float(np.linalg.norm(equal, axis=-1).sum(1)[loaded].mean()),
            "|f| rnea": float(np.linalg.norm(rnea, axis=-1).sum(1)[loaded].mean()),
            "max |f| rnea": float(np.linalg.norm(rnea, axis=-1).max()),
            "resid none": float(np.nanmean(np.linalg.norm(check[0, loaded, :3], axis=-1))),
            "resid equal": float(np.nanmean(np.linalg.norm(check[1, loaded, :3], axis=-1))),
            "resid rnea": float(np.nanmean(np.linalg.norm(check[2, loaded, :3], axis=-1)))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--contacts", choices=["measured", "interactvlm"], default="measured",
                        help="where the contact labels come from")
    parser.add_argument("--query", default="wall",
                        help="the InteractVLM prompt whose predictions are read")
    parser.add_argument("--sigma", type=float, default=0.1, help="Gaussian width, seconds")
    parser.add_argument("--clips", nargs="*", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_path = yaml.safe_load((_ROOT / "configs" / "base.yaml").read_text())["model"]["smplx"]["model_path"]
    device = torch.device(args.device)
    body, wrench = load_body(device, model_path), RootWrench(model_path, device)
    list_clips, contacts_of = DATASETS[args.dataset]
    source = "measured contacts"
    if args.contacts == "interactvlm":
        root = _OTHERS[args.dataset] / f"interactvlm_{args.query}"
        contacts_of = lambda clip, n: contacts_interactvlm(root, clip, n)  # noqa: E731
        source = f"InteractVLM (query: {args.query}, lbs/frac10) contacts"
    clips = [(c, t) for c, t in list_clips() if not args.clips or c in args.clips]
    print(f"{args.dataset}: {len(clips)} clips, sigma {args.sigma:g} s, {source} "
          f"-> predictions/{RUNS}")
    for index, (clip, tree) in enumerate(clips, start=1):
        n = len(np.load(tree / "geometry" / "transform.npz")["frame_indices"])
        stats = run_clip(clip, tree, contacts_of(clip, n), args.sigma, body, wrench, device,
                         source)
        print(f"[{index}/{len(clips)}] {clip}: " + ", ".join(
            f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in stats.items()),
            flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
