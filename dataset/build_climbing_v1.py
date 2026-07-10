"""Build ``ClimbingImages_v1`` — a self-contained, fast-to-load SMPL contact dataset.

Reads the BetterImageReconstruction outputs (DB + NPZ sidecars), keeps only
climbers whose ``contact_surface`` count is within ``[min_contacts,
max_contacts]``, converts everything from SMPL-X (10475) to SMPL (6890), and
writes a flat copy:

    <out>/
      images/<idx>.jpg     — copied source image (one per climber-item)
      masks/<idx>.png      — that climber's binary person mask (full image size)
      metadata.npz         — all labels, stacked by index (see schema below)
      dataset_info.json    — counts, thresholds, provenance

Nothing is referenced back to the source tree at load time. ``contact`` is the
exact barycentric transfer of the SMPL-X label; the SMPL params are the
closed-form conversion (pose copied, betas via the precomputed linear map,
translation pelvis-corrected) and carry ~2-3 cm of mesh error vs SMPL-X.
The SMPL-X source params + contacts and the ``pre_*`` similarity are kept so the
camera-space mesh can be reconstructed and the conversion re-run if improved.

Run::

    python dataset/build_climbing_v1.py                 # full build from config
    python dataset/build_climbing_v1.py --limit 20      # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from conversion.smplx_smpl_conversion import SmplxToSmpl  # noqa: E402

NUM_SMPL, NUM_SMPLX = 6890, 10475
DEFAULT_CONFIG = REPO / "configs" / "climbing.yaml"


def _shard(sha: str) -> tuple[str, str]:
    return sha[:2], sha[2:4]


def _mask_to_bbox(mask: np.ndarray, pad_frac: float = 0.05) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
    H, W = mask.shape[:2]
    return np.array([max(0.0, x0 - px), max(0.0, y0 - py),
                     min(W - 1.0, x1 + px), min(H - 1.0, y1 + py)], np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--limit", type=int, default=0, help="stop after N kept items (smoke test)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())["build"]
    src = Path(cfg["source_root"])
    out = Path(cfg["out"])
    db  = Path(cfg.get("db") or src / "image_reconstruction.db")
    require = tuple(cfg.get("require", ("human_optim",)))
    lo, hi = int(cfg["min_contacts"]), int(cfg["max_contacts"])

    conv = SmplxToSmpl(device=args.device)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    flags = " AND ".join(f"{c}=1" for c in require)
    con = sqlite3.connect(str(db))
    rows = con.execute(f"SELECT sha256, gemma_climber_indices FROM images WHERE {flags}").fetchall()
    con.close()

    rec: dict[str, list] = {k: [] for k in (
        "key", "gender", "betas", "body_pose", "global_orient", "transl",
        "contact", "bbox", "cam_int", "pre_scale", "scale_delta",
        "pre_rotation", "pre_translation", "smplx_betas", "smplx_body_pose",
        "smplx_global_orient", "smplx_transl", "smplx_contact",
        "sapiens_scale", "sapiens_shift", "sapiens_inliers", "body_rmse",
        "body_inliers", "body_valid", "stitch_rmse", "stitch_err_local",
        "stitch_err_global", "stitch_ok")}

    idx = dropped_low = dropped_high = missing = 0
    for sha, idx_json in tqdm(rows, desc="build"):
        idxs = json.loads(idx_json or "[]")
        if not idxs:
            continue
        aa, bb = _shard(sha)
        h = np.load(src / "results" / "human_optim" / aa / bb / sha / "human_optim.npz")
        cs_all = h["contact_surface"]                       # [n, 10475] bool
        pre = np.load(src / "results" / "pre_optim" / aa / bb / sha / "pre_optim.npz")
        img_src = src / "images" / aa / bb / f"{sha}.jpg"

        for ci, mask_i in enumerate(idxs):
            n = int(cs_all[ci].sum())
            if n < lo:
                dropped_low += 1; continue
            if n > hi:
                dropped_high += 1; continue
            mask_path = src / "features" / "sam3" / aa / bb / sha / f"{int(mask_i):02d}.png"
            if not mask_path.exists():
                missing += 1; continue
            mask = np.array(Image.open(mask_path))
            bbox = _mask_to_bbox(mask)
            if bbox is None:
                missing += 1; continue

            smpl = conv.convert_params(h["refined_betas"][ci, 0], h["refined_body_pose"][ci, 0],
                                       h["refined_global_orient"][ci, 0], h["refined_transl"][ci, 0])
            contact = conv.convert_contacts(cs_all[ci].astype(np.float32)).numpy().astype(bool)

            shutil.copyfile(img_src, out / "images" / f"{idx:05d}.jpg")
            Image.fromarray(((mask > 0).astype(np.uint8) * 255), mode="L").save(out / "masks" / f"{idx:05d}.png")

            rec["key"].append(f"{sha}#{ci}")
            rec["gender"].append(str(h["gender"]))
            rec["betas"].append(smpl["betas"].numpy())
            rec["body_pose"].append(smpl["body_pose"].numpy())
            rec["global_orient"].append(smpl["global_orient"].numpy())
            rec["transl"].append(smpl["transl"].numpy())
            rec["contact"].append(np.packbits(contact))
            rec["bbox"].append(bbox)
            rec["cam_int"].append(h["cam_intrinsics"].astype(np.float32))
            rec["pre_scale"].append(np.float32(h["pre_scale"][ci]))
            rec["scale_delta"].append(np.float32(h["refined_scale_delta"][ci]))
            rec["pre_rotation"].append(h["pre_rotation"][ci].astype(np.float32))
            rec["pre_translation"].append(h["pre_translation"][ci].astype(np.float32))
            rec["smplx_betas"].append(h["refined_betas"][ci, 0].astype(np.float32))
            rec["smplx_body_pose"].append(h["refined_body_pose"][ci, 0].astype(np.float32))
            rec["smplx_global_orient"].append(h["refined_global_orient"][ci, 0].astype(np.float32))
            rec["smplx_transl"].append(h["refined_transl"][ci, 0].astype(np.float32))
            rec["smplx_contact"].append(np.packbits(cs_all[ci].astype(bool)))
            rec["sapiens_scale"].append(np.float32(pre["sapiens_scale"]))
            rec["sapiens_shift"].append(np.float32(pre["sapiens_shift"]))
            rec["sapiens_inliers"].append(np.int32(pre["sapiens_inliers"]))
            rec["body_rmse"].append(np.float32(pre["body_rmse"][ci]))
            rec["body_inliers"].append(np.int32(pre["body_inliers"][ci]))
            rec["body_valid"].append(bool(pre["body_valid"][ci]))
            rec["stitch_rmse"].append(np.float32(pre["stitch_rmse"][ci]))
            rec["stitch_err_local"].append(np.float32(pre["stitch_err_local"][ci]))
            rec["stitch_err_global"].append(np.float32(pre["stitch_err_global"][ci]))
            rec["stitch_ok"].append(bool(pre["stitch_ok"][ci]))
            idx += 1
            if args.limit and idx >= args.limit:
                break
        if args.limit and idx >= args.limit:
            break

    meta = {k: np.stack(v) if k not in ("key", "gender") else np.asarray(v) for k, v in rec.items()}
    np.savez(out / "metadata.npz", **meta,
             num_vertices=np.int64(NUM_SMPL), num_vertices_smplx=np.int64(NUM_SMPLX))

    info = dict(
        schema_version=1, n_samples=idx,
        n_images=len({k.split("#")[0] for k in rec["key"]}),
        num_vertices=NUM_SMPL, num_vertices_smplx=NUM_SMPLX,
        thresholds=dict(min_contacts=lo, max_contacts=hi),
        dropped=dict(too_few=dropped_low, too_many=dropped_high, no_mask=missing),
        source_root=str(src),
        transfer=("Phy-SIC smplx_to_smpl.pkl barycentric; SMPL params closed-form "
                  "(pose copy + linear betas + pelvis-corrected transl), ~2-3cm mesh error"),
    )
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2))

    print(f"\nwrote {out}")
    print(f"  kept {idx} climbers from {info['n_images']} images")
    print(f"  dropped: too_few={dropped_low} too_many={dropped_high} no_mask={missing}")
    print(f"  metadata.npz = {(out/'metadata.npz').stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
