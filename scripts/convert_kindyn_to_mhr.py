"""Convert kindyn SMPL-X reconstructions to MHR pseudo-GT (``mhr_1.npz``).

The kindyn stage solved a smooth, physically-consistent SMPL-X trajectory per
scene, but the frozen SAM-3D-Body model predicts MHR — so pose supervision
needs the kindyn result re-expressed as an MHR trajectory. This script fits one
world-frame MHR ``q`` trajectory per tracked person to the kindyn
``joints_world`` targets and writes it next to the kindyn file.

Method (validated on corpus scenes; per-joint median residual ~0.5 cm):

1. **Init**: the frozen model's own per-frame MHR predictions
   (``features/sam3d/<scene>/params.npz``) lifted into the metric
   reconstruction world with the dataset extrinsics, via the physics
   :class:`~contact.physics.adapter.MHRAdapter` composition. One body is baked
   per person from the mean valid-frame shape.
2. **Pass 1**: Adam on the free-flyer root + 125 pose channels, Huber on the
   matched-joint world positions (:data:`JOINT_MAP`), a rest prior toward the
   init pose and a pose-acceleration smoothness term.
3. **Offsets**: SMPL-X and MHR place anatomically-matched joints differently
   (spine3 ~10 cm, head/collars/pelvis 4-6 cm). A constant per-joint offset in
   the joint's LOCAL frame (median over valid frames of ``R^T (target - fk)``)
   absorbs the rig mismatch without absorbing time-varying pose corrections.
4. **Pass 2**: refit against the offset-corrected targets.

The fitted trajectory inherits kindyn's smoothness (pose acceleration RMS
drops ~10x vs the per-frame init on the validation scene).

Output ``features/human_optim/<shard>/<scene>/mhr_1.npz``:

* ``q_world (P, N, 132)`` float32 — ``[tx, ty, tz, qx, qy, qz, qw]`` (xyzw,
  hemisphere-aligned along time) + 125 MHR pose channels, world frame.
* ``identity (P, 45)`` float32 — the baked per-person identity coefficients.
* ``valid_mask (P, N)`` bool — rows that were actually fitted (sam3d AND
  kindyn valid AND finite sam3d params); other rows are only loosely
  constrained (rest prior / smoothness) or, for skipped persons, identity-quat
  zeros — never supervise on them.
* ``object_ids (P,)``, ``num_frames``, ``fps``, ``converter_version``.
* Diagnostics: ``residual_med_cm (P,)`` (vs the OFFSET-CORRECTED targets — the
  retarget consistency), ``residual_vs_kindyn_med_cm (P,)`` (vs the raw kindyn
  joints, i.e. including the constant rig offset), ``per_joint_residual_cm
  (P, J)``, ``offsets_local (P, J, 3)`` (metres), ``joint_map (J, 2)``.

Usage::

    CUDA_VISIBLE_DEVICES=0 python scripts/convert_kindyn_to_mhr.py            # all scenes
    python scripts/convert_kindyn_to_mhr.py --scenes MuVpoovQl2M_0001 --overwrite
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from contact.data.climbing_corpus import hemisphere_align
from contact.physics.adapter import MHRAdapter

CONVERTER_VERSION = 1

#: (SMPL-X kindyn joint, MHR native joint) pairs the fit matches. Fingers are
#: excluded (kindyn hands are coarse); every pair was probed on real scenes.
JOINT_MAP = (
    ("pelvis", "root"), ("left_hip", "l_upleg"), ("right_hip", "r_upleg"),
    ("left_knee", "l_lowleg"), ("right_knee", "r_lowleg"),
    ("left_ankle", "l_talocrural"), ("right_ankle", "r_talocrural"),
    ("left_foot", "l_ball"), ("right_foot", "r_ball"),
    ("spine1", "c_spine1"), ("spine2", "c_spine2"), ("spine3", "c_spine3"),
    ("neck", "c_neck"), ("head", "c_head"),
    ("left_collar", "l_clavicle"), ("right_collar", "r_clavicle"),
    ("left_shoulder", "l_uparm"), ("right_shoulder", "r_uparm"),
    ("left_elbow", "l_lowarm"), ("right_elbow", "r_lowarm"),
    ("left_wrist", "l_wrist"), ("right_wrist", "r_wrist"),
)

#: Fit hyperparameters (per pass). Huber delta in metres; the rest prior keeps
#: target-unconstrained channels (fingers, twists) at their init; the
#: acceleration term keeps them SMOOTH where the init is jittery.
FIT_ITERS = 300
FIT_LR = 2e-2
HUBER_DELTA_M = 0.05
W_REST = 1e-3
W_SMOOTH = 0.1


def _fit(body, mids: torch.Tensor, q_init: torch.Tensor, targets: torch.Tensor,
         vmask: torch.Tensor) -> torch.Tensor:
    """One Adam pass; ``q_init (1, N, 132)``, ``targets (1, N, J, 3)``.

    :param vmask: ``(1, N, 1, 1)`` float validity weights.
    :returns: fitted ``q (1, N, 132)`` (detached).
    """
    from better_robot import forward_kinematics

    t = q_init[..., :3].clone().requires_grad_(True)
    quat = q_init[..., 3:7].clone().requires_grad_(True)
    pose = q_init[..., 7:].clone().requires_grad_(True)
    pose0 = q_init[..., 7:].clone()
    opt = torch.optim.Adam([t, quat, pose], lr=FIT_LR)
    denom = vmask.sum().clamp(min=1.0) * len(JOINT_MAP) * 3
    for _ in range(FIT_ITERS):
        opt.zero_grad()
        qn = torch.cat([t, F.normalize(quat, dim=-1), pose], -1)
        fk = forward_kinematics(body.robot, qn)
        r = fk.joint_pose_world[..., :3][:, :, mids] - targets
        loss = (F.huber_loss(r, torch.zeros_like(r), delta=HUBER_DELTA_M,
                             reduction="none") * vmask).sum() / denom
        loss = loss + W_REST * ((pose - pose0) ** 2).mean()
        acc = pose[:, 2:] - 2 * pose[:, 1:-1] + pose[:, :-2]
        loss = loss + W_SMOOTH * (acc ** 2).mean()
        loss.backward()
        opt.step()
    return torch.cat([t, F.normalize(quat, dim=-1), pose], -1).detach()


def convert_scene(scene_dir: Path, features: Path, adapter: MHRAdapter,
                  device: str) -> dict:
    """Fit every tracked person of one scene; return the ``mhr_1.npz`` payload."""
    from better_robot import forward_kinematics
    from better_robot.lie import so3
    from better_human.bodies import MHRClassic

    scene = scene_dir.name
    shard = scene_dir.parent.relative_to(features / "human_optim")
    kin = np.load(scene_dir / "kindyn_1.npz", allow_pickle=True)
    sam = np.load(features / "sam3d" / shard / scene / "params.npz",
                  allow_pickle=True)
    tra = np.load(features / "geometry" / shard / scene / "transform.npz",
                  allow_pickle=True)

    kj_names = [str(x) for x in kin["joint_names"]]
    kids = [kj_names.index(s) for s, _ in JOINT_MAP]
    mhr_names = list(adapter.body.structure.joint_names)
    npi = adapter.body.structure.native_pose_joint_indices
    mids = torch.tensor([int(npi[mhr_names.index(m)]) for _, m in JOINT_MAP],
                        device=device)

    n = int(kin["num_frames"])
    if int(np.asarray(sam["num_frames"])) != n:
        raise ValueError(f"{scene}: sam3d has {int(np.asarray(sam['num_frames']))} "
                         f"frames, kindyn {n}")
    ext = np.asarray(tra["extrinsics"], np.float32)
    if not np.isfinite(ext).all():
        raise ValueError(f"{scene}: non-finite extrinsics")
    cam = torch.tensor(ext, device=device)
    kin_ids = [int(i) for i in np.asarray(kin["object_ids"]).ravel()]
    sam_ids = [int(i) for i in np.asarray(sam["object_ids"]).ravel()]

    n_joints = len(JOINT_MAP)
    q_default = np.zeros((len(kin_ids), n, 132), np.float32)
    q_default[..., 6] = 1.0                # identity quat (xyzw) for unfitted rows
    out = {
        "q_world": q_default,
        "identity": np.zeros((len(kin_ids), 45), np.float32),
        "valid_mask": np.zeros((len(kin_ids), n), bool),
        "residual_med_cm": np.full((len(kin_ids),), np.nan, np.float32),
        "residual_vs_kindyn_med_cm": np.full((len(kin_ids),), np.nan, np.float32),
        "per_joint_residual_cm": np.full((len(kin_ids), n_joints), np.nan, np.float32),
        "offsets_local": np.zeros((len(kin_ids), n_joints, 3), np.float32),
    }
    for p_kin, oid in enumerate(kin_ids):
        if oid not in sam_ids:
            print(f"  {scene}: person {oid} missing from sam3d params — skipped")
            continue
        p_sam = sam_ids.index(oid)
        valid = (np.asarray(sam["valid_mask"], bool)[p_sam]
                 & np.asarray(kin["valid_mask"], bool)[p_kin])

        params = torch.tensor(sam["mhr_model_params"][p_sam], device=device)
        shape = torch.tensor(sam["shape_params"][p_sam], device=device)
        cam_t = torch.tensor(sam["pred_cam_t"][p_sam], device=device)
        # Several corpus scenes carry NaN sam3d params on frames already flagged
        # invalid. A NaN anywhere in the init would poison the rest-prior and
        # smoothness means (0 * nan = nan) and one Adam step would NaN the whole
        # trajectory — so non-finite rows are replaced by the nearest finite
        # frame's values and excluded from validity.
        finite = (torch.isfinite(params).all(-1) & torch.isfinite(shape).all(-1)
                  & torch.isfinite(cam_t).all(-1)).cpu().numpy()
        valid &= finite
        if valid.sum() < 3:
            print(f"  {scene}: person {oid} has {int(valid.sum())} valid frames — skipped")
            continue
        if not finite.all():
            fin_idx = np.flatnonzero(finite)
            near = fin_idx[np.abs(
                np.arange(n)[:, None] - fin_idx[None, :]).argmin(1)]
            fill = np.arange(n)
            fill[~finite] = near[~finite]
            sel = torch.tensor(fill, device=device)
            params, shape, cam_t = params[sel], shape[sel], cam_t[sel]
        vrow = torch.tensor(valid, device=device)

        # One body from the mean valid-frame identity; per-frame native q from
        # the full trajectory (from_classic's q is shape-independent).
        shape_mean = shape[vrow].mean(0, keepdim=True)
        body, _ = adapter.body.from_classic(MHRClassic(
            identity_coeffs=shape_mean, model_parameters=params[:1]))
        _, q_native = adapter.body.from_classic(MHRClassic(
            identity_coeffs=shape, model_parameters=params))
        q_native = q_native.view(1, n, -1)
        root_world = adapter._root_to_world(
            body.robot, q_native[..., :7], cam_t, cam, 1, n)
        q0 = torch.cat([root_world, q_native[..., 7:]], -1)

        from contact.physics.adapter import _with_time_axis
        body = body._replace(values=body.values, robot=_with_time_axis(body.robot))

        tgt = torch.tensor(np.asarray(kin["joints_world"], np.float32)[p_kin],
                           device=device)[None, :, kids]
        vmask = vrow[None, :, None, None].float()

        q1 = _fit(body, mids, q0, tgt, vmask)
        with torch.no_grad():
            fk1 = forward_kinematics(body.robot, q1)
            rot_w = so3.to_matrix(fk1.joint_pose_world[..., 3:][:, :, mids])
            delta_w = tgt - fk1.joint_pose_world[..., :3][:, :, mids]
            local = torch.einsum("bnjxy,bnjx->bnjy", rot_w, delta_w)
            off_local = local[0][vrow].median(dim=0).values             # [J, 3]
            tgt2 = tgt - torch.einsum("bnjxy,jy->bnjx", rot_w, off_local)
        q2 = _fit(body, mids, q1, tgt2, vmask)

        with torch.no_grad():
            fk2 = forward_kinematics(body.robot, q2)
            fk2_pos = fk2.joint_pose_world[..., :3][:, :, mids]
            resid = (fk2_pos - tgt2).norm(dim=-1)[0][vrow]              # [rows, J]
            resid_kindyn = (fk2_pos - tgt).norm(dim=-1)[0][vrow]
        if not torch.isfinite(q2).all():
            raise RuntimeError(f"{scene}: person {oid} fit produced non-finite q")
        q_out = q2[0].cpu().numpy()
        q_out[:, 3:7] = hemisphere_align(q_out[:, 3:7].astype(np.float64))
        out["q_world"][p_kin] = q_out
        out["identity"][p_kin] = shape_mean[0].cpu().numpy()
        out["valid_mask"][p_kin] = valid
        out["residual_med_cm"][p_kin] = float(resid.median()) * 100.0
        out["residual_vs_kindyn_med_cm"][p_kin] = float(resid_kindyn.median()) * 100.0
        out["per_joint_residual_cm"][p_kin] = (
            resid.median(dim=0).values.cpu().numpy() * 100.0)
        out["offsets_local"][p_kin] = off_local.cpu().numpy()

    out.update(
        object_ids=np.asarray(kin_ids, np.int32),
        num_frames=np.int32(n),
        fps=np.asarray(kin["fps"], np.float32),
        joint_map=np.asarray([[s, m] for s, m in JOINT_MAP]),
        converter_version=np.int32(CONVERTER_VERSION),
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", type=Path,
                    default=Path("/data3/rikhat.akizhanov/better/data/ClimbingVideos"))
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="scene names; default = every scene with a kindyn_1.npz")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    features = args.corpus / "features"
    scene_dirs = sorted(
        p.parent for p in (features / "human_optim").glob("*/*/*/kindyn_1.npz"))
    if args.scenes:
        wanted = set(args.scenes)
        scene_dirs = [d for d in scene_dirs if d.name in wanted]
        missing = wanted - {d.name for d in scene_dirs}
        if missing:
            raise SystemExit(f"scenes not found: {sorted(missing)}")
    if args.limit:
        scene_dirs = scene_dirs[: args.limit]

    adapter = MHRAdapter(device=args.device)
    done = skipped = failed = 0
    start = time.time()
    for i, scene_dir in enumerate(scene_dirs):
        target = scene_dir / "mhr_1.npz"
        if target.exists() and not args.overwrite:
            skipped += 1
            continue
        tmp = target.with_name("mhr_1.tmp.npz")
        try:
            payload = convert_scene(scene_dir, features, adapter, args.device)
            np.savez_compressed(tmp, **payload)
            tmp.rename(target)
        except Exception as exc:                       # keep the sweep alive
            failed += 1
            tmp.unlink(missing_ok=True)
            print(f"[{i + 1}/{len(scene_dirs)}] {scene_dir.name} FAILED: {exc}")
            continue
        done += 1
        med = np.nanmedian(payload["residual_med_cm"])
        elapsed = time.time() - start
        print(f"[{i + 1}/{len(scene_dirs)}] {scene_dir.name}: "
              f"P={len(payload['object_ids'])} med {med:.2f} cm "
              f"({elapsed / max(done, 1):.0f}s/scene)", flush=True)
    print(f"done={done} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
