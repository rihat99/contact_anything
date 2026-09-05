"""Score the frozen SAM 3D Body, as SMPL-X, on the test protocol (the `frozen` line).

    python scripts/eval_frozen_smplx.py --config configs/static_ray.yaml \
        --out output/frozen_sam3d_smplx_static16.json

The corpus ships the frozen model's per-frame output refit to SMPL-X
(``features/sam3d/<shard>/<scene>/smplx_params.npz``: classic camera-frame
``global_orient`` / ``body_pose`` / ``betas`` / ``transl``), so the MHR-vs-SMPL-X
topology gap never enters the comparison. Exactly the frames the config's
evaluation scores (one clip per (scene, person), longest valid run, capped at
``data.eval_max_frames``, auto stride) are scored with exactly the metric code
the trainer uses (:func:`model.loss.smplx.pose_metric_stats`: hip-aligned
MPJPE / PA-MPJPE / PVE in mm, dt-exact Accel in m/s^2; flat hands on both
sides). The result json is what ``output.frozen_metrics`` points at: the trainer
re-emits it at every evaluation step as the ``frozen`` tensorboard run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import roma
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import build_datasets                                     # noqa: E402
from data.climbing_videos.scene import scene_shard                  # noqa: E402
from data.loaders import build_loaders                              # noqa: E402
from model.loss.smplx import (                                      # noqa: E402
    POSE_METRICS,
    gt_smplx_camera,
    pose_metric_stats,
    pose_metrics_from_stats,
    smplx_q,
    smplx_vertices,
)
from train.config import load_config                                # noqa: E402


class FrozenSmplx:
    """Per-scene reader of the frozen model's SMPL-X refit, keyed by object id."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._cache: dict[str, dict] = {}

    def scene(self, scene: str) -> dict:
        if scene not in self._cache:
            path = (self.root / "features" / "sam3d" / scene_shard(scene) / scene
                    / "smplx_params.npz")
            z = np.load(path, allow_pickle=True)
            if str(z["model_type"]) != "smplx" or int(z["num_betas"]) != 10:
                raise ValueError(f"{path}: expected smplx / 10 betas, got "
                                 f"{z['model_type']} / {z['num_betas']}")
            self._cache = {scene: {                                  # one scene at a time
                "rows": {int(o): i for i, o in enumerate(
                    np.asarray(z["object_ids"]).reshape(-1))},
                "global_orient": np.asarray(z["global_orient"], np.float32),
                "body_pose": np.asarray(z["body_pose"], np.float32),
                "betas": np.asarray(z["betas"], np.float32),
                "transl": np.asarray(z["transl"], np.float32),
                "valid": np.asarray(z["valid_mask"], bool),
            }}
        return self._cache[scene]

    def rows(self, keys: list[str]) -> dict[str, np.ndarray]:
        """Stack the parameter rows of ``"{scene}#{oid}@{position}"`` keys."""
        out = {k: [] for k in ("global_orient", "body_pose", "betas", "transl", "valid")}
        for key in keys:
            scene, rest = key.split("#")
            oid, position = (int(x) for x in rest.split("@"))
            data = self.scene(scene)
            row = data["rows"][oid]
            for k in out:
                out[k].append(data[k][row, position])
        return {k: np.stack(v) for k, v in out.items()}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-scenes", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    _, test_sets = build_datasets(cfg, {"smplx"}, limit_scenes=args.limit_scenes)
    _, loader = build_loaders(cfg, [], test_sets)
    device = torch.device(args.device)

    import better_human as bh
    body = bh.SMPLX(
        model_path=cfg["model"]["smplx"]["model_path"], gender="neutral", num_betas=10,
        use_hands=False, use_face=False, compute_mass=False,
        dtype=torch.float32, device=device)
    frozen = FrozenSmplx(test_sets[0].root)

    stats = torch.zeros(2 * len(POSE_METRICS), dtype=torch.float64)
    n_clips = 0
    for batch in tqdm(loader, desc="frozen"):
        rows = frozen.rows(batch["key"])
        betas = torch.from_numpy(rows["betas"]).to(device)
        root_rot = roma.rotvec_to_rotmat(torch.from_numpy(rows["global_orient"]).to(device))
        body_rot = roma.rotvec_to_rotmat(
            torch.from_numpy(rows["body_pose"]).to(device).reshape(-1, 21, 3))
        # Classic transl -> BetterHuman root (= the pelvis): add the shaped pelvis offset.
        pelvis = (torch.from_numpy(rows["transl"]).to(device)
                  + body.with_shape(betas=betas).values.pelvis_offset)
        q = smplx_q(pelvis, root_rot, body_rot)
        shaped = body.with_shape(betas=betas)
        fk = shaped.fk(q)
        joints = fk.joint_pose_world[..., 1:, :3]
        verts = body.vertices_from_data(fk)

        gt = gt_smplx_camera(batch, device)
        gt_verts = smplx_vertices(body, gt["betas"], gt["q"])
        valid = gt["valid"] & torch.from_numpy(rows["valid"]).to(device)
        stats += pose_metric_stats(
            joints, verts, gt["joints"], gt_verts, valid,
            int(batch["seq_len"]), batch["frame_pos_sec"].to(device))
        n_clips += 1

    metrics = pose_metrics_from_stats(stats)
    result = {
        "metrics": {f"metric_pose/{name}": value for name, value in metrics.items()},
        "protocol": {
            "config": str(args.config),
            "scenes": sum(len(ds.scenes) for ds in test_sets),
            "clips": n_clips,
            "frames": float(stats[1]),
            "accel_rows": float(stats[7]),
            "eval_max_frames": int(cfg["data"]["eval_max_frames"]),
            "stride": cfg["data"]["clip"]["stride"],
        },
        "source": "features/sam3d/<shard>/<scene>/smplx_params.npz (frozen SAM3D refit to SMPL-X)",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
