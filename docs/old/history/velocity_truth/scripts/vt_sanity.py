"""Pipeline sanity checks before the forensics run (read-only)."""
from __future__ import annotations

import numpy as np

import vt_common as vt


def main() -> None:
    scenes = vt.test_scenes()
    print(f"{len(scenes)} test scenes: {scenes[:3]} ...")
    parents = vt.smplx_parents()
    print("parents[:22] =", parents[:22].tolist())

    scene = scenes[2]
    cams = vt.scene_cameras(scene)
    run = vt.load_run("static_ray", scene, cams["extrinsics"])
    gt = vt.load_gt(scene, run["object_ids"])
    print(f"\nscene {scene}: fps {cams['fps']:.3f}, stride {run['stride']}, "
          f"people {len(run['people'])}, frames {len(cams['extrinsics'])}")
    print("gravity_world:", np.round(gt["gravity"], 4).tolist())

    # --- 1. kindyn q[:3] == joints_world[0] -------------------------------
    g = gt["people"][0]
    v = g["valid"]
    d = np.linalg.norm(g["root_pos_w"][v] - g["joints_w"][v, 0], axis=-1)
    print(f"[1] GT q[:3] vs joints_world[0]: max {d.max() * 1000:.4f} mm")

    # --- 2. dump q_cam[:3] == pelvis_cam == joints_cam[0] -----------------
    raw = np.load(vt.REPO / vt.RUNS["static_ray"][0] / "predictions" / f"{scene}.npz")
    cov = np.asarray(raw["covered"], bool)[0]
    q_cam = np.asarray(raw["q_cam"], np.float64)[0][cov]
    pel = np.asarray(raw["pelvis_cam"], np.float64)[0][cov]
    jc = np.asarray(raw["joints_cam"], np.float64)[0][cov]
    print(f"[2] q_cam[:3] vs pelvis_cam: max {np.abs(q_cam[:, :3] - pel).max() * 1000:.4f} mm; "
          f"vs joints_cam[0]: max {np.abs(q_cam[:, :3] - jc[:, 0]).max() * 1000:.4f} mm")

    # --- 3. parent-local-from-world FK == kindyn q body quats -------------
    from viewer.bodies import gt_source
    shard_dir = vt.CORPUS / "features" / "human_optim" / vt.scene_shard(scene)
    src = gt_source(shard_dir / scene / "kindyn_1.npz", run["object_ids"],
                    len(cams["extrinsics"]), "cpu")
    person = src.people[0]
    quat_w = np.asarray(person.bone_wxyz, np.float64)[..., [1, 2, 3, 0]]
    pos_w = np.asarray(person.bone_pos, np.float64)
    ok = np.asarray(person.valid, bool) & v
    rot_w = vt.quat_to_matrix(np.where(ok[:, None, None], quat_w, [0.0, 0.0, 0.0, 1.0]))
    local = np.einsum("njab,njac->njbc", rot_w[:, parents[1:22]], rot_w[:, 1:22])
    ref = vt.quat_to_matrix(g["body_q"])
    err = np.linalg.norm((local[ok] - ref[ok]).reshape(-1, 9), axis=-1)
    print(f"[3] FK parent-local vs kindyn q body quats: max Frobenius {err.max():.3e}")
    dj = np.linalg.norm(pos_w[ok, :22] - g["joints_w"][ok], axis=-1)
    print(f"    FK joints vs kindyn joints_world: mean {dj.mean() * 1000:.3f} mm, "
          f"max {dj.max() * 1000:.3f} mm")
    dr = np.linalg.norm(vt.quat_to_matrix(quat_w[ok, 0]).reshape(-1, 9)
                        - vt.quat_to_matrix(g["root_quat_w"][ok]).reshape(-1, 9), axis=-1)
    print(f"    FK root rot vs kindyn q root quat: max Frobenius {dr.max():.3e}")

    # --- 4. frozen source -------------------------------------------------
    fr = vt.load_frozen(scene, cams["extrinsics"], run["object_ids"], parents)
    f0 = fr["people"][0]
    print(f"[4] frozen valid frames {int(f0['valid'].sum())}/{len(f0['valid'])}; "
          f"pelvis world span {np.ptp(f0['root_pos_w'][f0['valid']], axis=0).round(3).tolist()} m")

    # --- 5. camera smoothing magnitude on a static scene ------------------
    cs = vt.scene_cameras(scene, 0.25)
    dc = np.linalg.norm(cs["extrinsics"][:, :3, 3] - cams["extrinsics"][:, :3, 3], axis=-1)
    dr = np.linalg.norm((cs["extrinsics"][:, :3, :3] - cams["extrinsics"][:, :3, :3]
                         ).reshape(-1, 9), axis=-1)
    print(f"[5] camera smoothing sigma 0.25 s: |dt| mean {dc.mean() * 1000:.3f} mm "
          f"max {dc.max() * 1000:.3f} mm; |dR|_F mean {dr.mean():.2e} max {dr.max():.2e}")

    # --- 6. strides / fps per scene ---------------------------------------
    print("\nscene, fps, stride, frames, people, gt_valid, covered(static_ray)")
    for s in scenes:
        c = vt.scene_cameras(s)
        r = vt.load_run("static_ray", s, c["extrinsics"])
        gg = vt.load_gt(s, r["object_ids"])
        nv = sum(int(p["valid"].sum()) for p in gg["people"] if p is not None)
        nc = sum(int(p["valid"].sum()) for p in r["people"])
        print(f"  {s:20s} {c['fps']:7.3f} {r['stride']:2d} {len(c['extrinsics']):5d} "
              f"{len(r['people'])} {nv:5d} {nc:5d}")


if __name__ == "__main__":
    main()
