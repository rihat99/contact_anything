"""Fit MHR to the kindyn SMPL-X surface — the ``mhr_1.npz`` pose pseudo-GT (v2).

v1 matched 22 joint POSITIONS. A joint's own rotation is only observable through
its fitted children, so head rotation, all finger channels and the foot
intrinsics carried rest-prior/smoothing artifacts — which the pose q loss then
imposed on the model (probe 2026-08-28: the head channel sat 26 deg from the
frozen model and training converged onto the artifact). v2 is an exact
mesh-to-mesh conversion instead:

1. **Target**: the kindyn body itself — BVR ``tools.body.load_body``
   (better_human SMPLX, ``use_face=False``, ``q (211)``) posed at the stored
   per-frame ``q``; world frame, metres. Reproduces kindyn ``joints_world`` to
   0.00 cm, hands and head included.
2. **Fit**: Meta's converter (``third_party/MHR/tools/mhr_smpl_conversion``),
   ``convert_smpl2mhr(smpl_vertices=..., method="pytorch",
   single_identity=True)``: every MHR vertex is a fixed barycentric point on the
   SMPL-X surface (18439 exact correspondences), staged Adam over per-frame
   root+pose (136) with shared identity (45) + scales (68).
3. **q**: better_human ``MHR.from_classic(MHRClassic(identity, lbs_params))`` —
   the SAME mapping the training-time pred path uses (``contact/pose_supervision``),
   so GT and prediction q are consistent by construction. Root stays in the
   kindyn world; ``POSE_SLOTS`` supervises only the 125 local channels.

Run with the **BVR venv** python (pymomentum + smplx + better_human)::

    /data3/rikhat.akizhanov/better/BetterVideoReconstruction/.venv/bin/python \
        scripts/convert_kindyn_to_mhr.py --num-shards 10 --shard-index 0

Output schema (per scene): ``q_world (P, N, 132)`` f32, ``valid_mask (P, N)``
bool, ``identity (P, 45)``, ``lbs_params (P, N, 204)`` (the full fitted MHR
``lbs_model_params`` row per frame — root+pose+proportions; NaN on unfitted
rows; v3), ``object_ids (P,)``, ``num_frames``, ``fps``,
``fit_err_cm (P, N)`` (MHR-vertex mesh error, NaN on unfitted rows),
``joint_vs_kindyn_med_cm (P,)`` (FK cross-check), ``converter_version = 2``.
The loader consumes q_world / valid_mask / object_ids / num_frames only.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BVR = Path("/data3/rikhat.akizhanov/better/BetterVideoReconstruction")
_TOOL = BVR / "third_party" / "MHR" / "tools" / "mhr_smpl_conversion"
sys.path.insert(0, str(BVR / "scripts"))
import _pym_preload  # noqa: F401  (re-execs with the pymomentum libstdc++ preloaded)

sys.path.insert(0, str(BVR))                        # tools.body
sys.path.insert(0, str(BVR / "third_party" / "MHR"))  # the mhr package
sys.path.insert(0, str(_TOOL))                      # flat conversion imports
os.chdir(_TOOL)                                     # ./assets/* are cwd-relative

import argparse   # noqa: E402
import time       # noqa: E402
from dataclasses import dataclass  # noqa: E402

import numpy as np    # noqa: E402
import torch          # noqa: E402
import smplx as smplx_lib          # noqa: E402
from conversion import Conversion  # noqa: E402
from mhr.mhr import MHR as MetaMHR  # noqa: E402

import better_human as bh                       # noqa: E402
from better_human.bodies.mhr.body import MHRClassic  # noqa: E402
from tools.body import load_body                # noqa: E402

CONVERTER_VERSION = 3
_MODELS = Path("/data3/rikhat.akizhanov/better/BetterHuman/models")
#: Fit batch cap: ~13 MB GPU per frame in the fit -> ~6 GB at 450.
MAX_FIT_BATCH = 450
#: (kindyn SMPL-X joint, MHR native joint) name pairs for the FK cross-check
#: diagnostic (v1's fit targets; diagnostics only in v2).
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


def hemisphere_align(quat: np.ndarray) -> np.ndarray:
    """Flip quaternion signs for consecutive-row continuity. ``quat (N, 4)``."""
    out = quat.copy()
    for i in range(1, len(out)):
        if float((out[i] * out[i - 1]).sum()) < 0.0:
            out[i] = -out[i]
    return out


@dataclass
class Ctx:
    """Models shared across scenes (built once per process)."""

    body: object            # BVR Body (bh SMPLX q(211) + vertices)
    smplx_model: object     # pip smplx (topology carrier for Conversion)
    meta_mhr: object        # Meta MHR (cm, 18439 verts) — the fit model
    bh_mhr: object          # better_human MHR — the q mapping + FK check
    device: str


def build_ctx(device: str) -> Ctx:
    body = load_body(device=device)
    smplx_model = smplx_lib.create(
        model_path=str(_MODELS), model_type="smplx", gender="neutral",
        use_pca=False, num_betas=10, batch_size=1)
    faces_bh = body.faces.cpu().numpy().astype(np.int64)
    faces_pip = smplx_model.faces.astype(np.int64)
    if not np.array_equal(faces_bh, faces_pip):
        raise RuntimeError("bh SMPLX and pip smplx topologies differ")
    meta_mhr = MetaMHR.from_files(lod=1, device=torch.device(device))
    bh_mhr = bh.MHR(_MODELS / "MHR" / "converted" / "mhr_lod1.npz", lod=1,
                    use_expression=False, use_correctives=False,
                    compute_mass=False, device=device)
    return Ctx(body, smplx_model, meta_mhr, bh_mhr, device)


def convert_scene(scene_dir: Path, ctx: Ctx) -> dict:
    """Fit every tracked person of one scene; return the npz payload."""
    from better_robot import forward_kinematics

    kin = np.load(scene_dir / "kindyn_1.npz", allow_pickle=True)
    q_s = np.asarray(kin["q"], np.float32)             # (P, N, 211)
    vm = np.asarray(kin["valid_mask"], bool)           # (P, N)
    betas = np.asarray(kin["betas"], np.float32)       # (P, 10)
    ids = np.asarray(kin["object_ids"], np.int32)
    n_people, n = vm.shape

    q_world = np.zeros((n_people, n, 132), np.float32)
    q_world[..., 6] = 1.0                              # identity quat (xyzw)
    identity = np.zeros((n_people, 45), np.float32)
    lbs_params = np.full((n_people, n, 204), np.nan, np.float32)
    fit_err = np.full((n_people, n), np.nan, np.float32)
    joint_med = np.full((n_people,), np.nan, np.float32)

    kn = [str(x) for x in kin["joint_names"]]
    # fk.joint_pose_world rows are the robot's BODY list, not structure joints.
    mhr_names = list(ctx.bh_mhr.robot.body_names)
    kids = [kn.index(s) for s, _ in JOINT_MAP]
    mids = [mhr_names.index(m) for _, m in JOINT_MAP]

    for p in range(n_people):
        rows = np.flatnonzero(vm[p])
        if rows.size == 0:
            continue
        with torch.no_grad():
            verts = ctx.body.forward(
                torch.tensor(betas[p], device=ctx.device),
                torch.tensor(q_s[p, rows], device=ctx.device),
            ).vertices                                  # (R, 10475, 3) m, world
        conv = Conversion(
            mhr_model=ctx.meta_mhr, smpl_model=ctx.smplx_model,
            method="pytorch", batch_size=min(int(rows.size), MAX_FIT_BATCH))
        res = conv.convert_smpl2mhr(
            smpl_vertices=verts, single_identity=True, exclude_expression=True,
            is_tracking=False, return_mhr_meshes=False, return_mhr_vertices=False,
            return_mhr_parameters=True, return_fitting_errors=True)
        params = res.result_parameters

        def _t(x) -> torch.Tensor:
            if isinstance(x, torch.Tensor):
                return x.detach().to(device=ctx.device, dtype=torch.float32)
            return torch.as_tensor(np.asarray(x), dtype=torch.float32,
                                   device=ctx.device)

        lbs = _t(params["lbs_model_params"])                            # (R, 204)
        iden = _t(params["identity_coeffs"])                            # (R, 45)
        _, q_fit = ctx.bh_mhr.from_classic(
            MHRClassic(identity_coeffs=iden, model_parameters=lbs))     # (R, 132)
        if not torch.isfinite(q_fit).all():
            raise RuntimeError(f"{scene_dir.name}: person {int(ids[p])} "
                               "fit produced non-finite q")
        q_np = q_fit.detach().cpu().numpy().astype(np.float32)
        q_np[:, 3:7] = hemisphere_align(q_np[:, 3:7])
        q_world[p, rows] = q_np
        identity[p] = iden[0].detach().cpu().numpy()
        lbs_params[p, rows] = lbs.detach().cpu().numpy().astype(np.float32)
        err = _t(res.result_errors).cpu().numpy().astype(np.float32).reshape(-1)
        fit_err[p, rows] = err[: rows.size]

        with torch.no_grad():
            fk = forward_kinematics(
                ctx.bh_mhr.robot, torch.as_tensor(q_np, device=ctx.device)[None])
            pred_j = fk.joint_pose_world[..., :3][0][:, mids].cpu().numpy()
        gt_j = np.asarray(kin["joints_world"], np.float32)[p][rows][:, kids]
        joint_med[p] = float(np.median(
            np.linalg.norm(pred_j - gt_j, axis=-1))) * 100.0

    return dict(
        q_world=q_world, valid_mask=vm, identity=identity,
        lbs_params=lbs_params,
        fit_err_cm=fit_err, joint_vs_kindyn_med_cm=joint_med,
        object_ids=ids, num_frames=np.int32(n),
        fps=np.asarray(kin["fps"], np.float32),
        converter_version=np.int32(CONVERTER_VERSION),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", type=Path,
                    default=Path("/data3/rikhat.akizhanov/better/data/ClimbingVideos"))
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="scene names; default = every scene with a kindyn_1.npz")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split the scene list into this many interleaved shards")
    ap.add_argument("--shard-index", type=int, default=0)
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
    if args.num_shards > 1:
        scene_dirs = scene_dirs[args.shard_index :: args.num_shards]

    ctx = build_ctx(args.device)
    done = skipped = failed = 0
    start = time.time()
    for i, scene_dir in enumerate(scene_dirs):
        target = scene_dir / "mhr_1.npz"
        if target.exists() and not args.overwrite:
            skipped += 1
            continue
        tmp = target.with_name("mhr_1.tmp.npz")
        try:
            payload = convert_scene(scene_dir, ctx)
            np.savez_compressed(tmp, **payload)
            tmp.rename(target)
        except Exception as exc:                       # keep the sweep alive
            failed += 1
            tmp.unlink(missing_ok=True)
            print(f"[{i + 1}/{len(scene_dirs)}] {scene_dir.name} FAILED: {exc}",
                  flush=True)
            continue
        done += 1
        med = np.nanmedian(payload["fit_err_cm"])
        jmed = np.nanmedian(payload["joint_vs_kindyn_med_cm"])
        elapsed = time.time() - start
        print(f"[{i + 1}/{len(scene_dirs)}] {scene_dir.name}: "
              f"P={len(payload['object_ids'])} mesh {med:.2f} cm "
              f"joints {jmed:.2f} cm ({elapsed / max(done, 1):.0f}s/scene)",
              flush=True)
    print(f"done={done} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
