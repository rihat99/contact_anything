"""Evaluate SAM-3D-Body's OWN MHR module at the ``mhr_1`` v3 GT parameters.

``mhr_1.npz`` (converter v3) stores, per tracked person and frame, the fitted
MHR ``lbs_params (204)`` plus a per-person ``identity (45)``. Those are exactly
the two tensors :meth:`MHRHead.mhr_forward` feeds the MHR module, so pushing
them through the *same* module and the *same* sapiens-308 keypoint regressor
(sliced to the first 70) yields GT keypoints and vertices that live on the
model's own rig, with the model's own keypoint definition. Nothing is fitted
here — this is a pure FK + LBS replay of the stored fit.

Frame: the v3 fit targeted WORLD-frame SMPL-X vertices, so the module's raw
output is already the kindyn metric world (metres after the ``/100`` cm
conversion). The ``[1, 2]`` axis flip :meth:`MHRHead.forward` applies to
``pred_keypoints_3d`` is a NATIVE->CAMERA change of basis and must NOT be
applied here — verified: with the flip the 13 name-matched joints sit 21.8 m
from kindyn ``joints_world``, without it 3.1 cm (the known cross-rig offset),
and BetterHuman FK at the stored ``q_world`` reproduces the module's 127 joint
centres to 0.0001 cm.

Output schema (per scene, ``mhr_sup_1.npz`` next to ``mhr_1.npz``)::

    kp_world            (P, N, 70, 3) f32   MHR70 keypoints, world, NaN-padded
    verts_world         (P, N, V, 3)  f32   vertex subset, world, NaN-padded
    vert_indices        (V,)          i64   MHR lod1 vertex ids (same every scene)
    kp_vs_kindyn_med_cm (P,)          f32   diagnostic: median distance over the
                                            13 name-matched joints vs kindyn
    schema_version, source_converter_version, num_frames, fps

Run with the sam3d env python::

    CUDA_VISIBLE_DEVICES=1 python scripts/precompute_mhr_supervision.py \\
        --num-shards 3 --shard-index 0
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

SCHEMA_VERSION = 1
#: Only the v3 converter stores ``lbs_params``; earlier files cannot be replayed.
REQUIRED_CONVERTER_VERSION = 3
#: ``pred_keypoints_3d`` is the sapiens-308 regressor sliced to its first 70 rows.
NUM_KEYPOINTS = 70
#: Farthest-point-sampling seed for the vertex subset (fixed => same ids forever).
FPS_SEED = 0
_DEFAULT_CKPT_DIR = Path(
    "/data3/rikhat.akizhanov/.cache/huggingface/hub/models--facebook--sam-3d-body-dinov3"
    "/snapshots/11aaa346c7204874a1cbafe3d39a979080b2c55a")
#: (kindyn SMPL-X joint, MHR70 keypoint index) pairs for the cross-rig diagnostic
#: — ``contact.keypoint_supervision.KP_MHR70_INDICES`` in ``KP_JOINT_NAMES`` order.
KINDYN_JOINT_CHECK = (
    ("left_shoulder", 5), ("right_shoulder", 6), ("left_elbow", 7),
    ("right_elbow", 8), ("left_wrist", 62), ("right_wrist", 41),
    ("left_hip", 9), ("right_hip", 10), ("left_knee", 11), ("right_knee", 12),
    ("left_ankle", 13), ("right_ankle", 14), ("neck", 69),
)


class MHRGroundTruth:
    """SAM-3D-Body's MHR module + sapiens keypoint regressor, replay-only.

    :param ckpt_dir: SAM-3D-Body snapshot directory (``model.ckpt`` +
        ``assets/mhr_model.pt``).
    :param num_vertices: size of the farthest-point vertex subset.
    :param device: torch device string.
    """

    def __init__(self, ckpt_dir: Path, num_vertices: int, device: str = "cuda") -> None:
        self.device = device
        self.mhr = torch.jit.load(str(ckpt_dir / "assets" / "mhr_model.pt"),
                                  map_location=device)
        state = torch.load(str(ckpt_dir / "model.ckpt"), map_location="cpu",
                           weights_only=False)
        state = state.get("state_dict", state)
        self.keypoint_mapping = state["head_pose.keypoint_mapping"].to(device).float()
        scale_mean = state["head_pose.scale_mean"].to(device).float()
        self.vert_indices = _farthest_point_indices(
            self._template_vertices(scale_mean), num_vertices, FPS_SEED)

    def _template_vertices(self, scale_mean: Tensor) -> Tensor:
        """Rest-pose mean-identity template mesh, ``(num_verts, 3)`` metres.

        Zero identity, zero pose/translation, mean bone scales — the same
        ``scales = scale_mean + scale_params @ scale_comps`` the head computes at
        ``scale_params = 0``.
        """
        model_params = torch.cat(
            (torch.zeros(1, 136, device=self.device), scale_mean[None]), dim=1)
        with torch.no_grad():
            verts, _ = self.mhr(torch.zeros(1, 45, device=self.device), model_params,
                                torch.zeros(1, 72, device=self.device))
        return verts[0] / 100.0

    @torch.no_grad()
    def evaluate(self, identity: Tensor, model_params: Tensor) -> tuple[Tensor, Tensor]:
        """Replay the MHR module at GT parameters.

        :param identity: ``(R, 45)`` MHR identity blendshape coefficients.
        :param model_params: ``(R, 204)`` MHR ``lbs_model_params`` rows.
        :returns: ``(kp_world (R, 70, 3), verts_world (R, V, 3))`` in metres,
            WORLD frame (no native->camera flip — see the module docstring).
        """
        rows = model_params.shape[0]
        verts, skel_state = self.mhr(
            identity, model_params, torch.zeros(rows, 72, device=self.device))
        verts = verts / 100.0
        joint_coords = skel_state[..., :3] / 100.0
        vert_joints = torch.cat((verts, joint_coords), dim=1)
        keypoints = (
            (self.keypoint_mapping @ vert_joints.permute(1, 0, 2).flatten(1, 2))
            .reshape(-1, rows, 3).permute(1, 0, 2))
        return keypoints[:, :NUM_KEYPOINTS], verts[:, self.vert_indices]


def _farthest_point_indices(points: Tensor, num_samples: int, seed: int) -> Tensor:
    """Deterministic farthest-point sampling over ``points (N, 3)``.

    :returns: ``(num_samples,)`` long tensor of point indices, sorted ascending.
    """
    start = int(np.random.default_rng(seed).integers(points.shape[0]))
    chosen = torch.empty(num_samples, dtype=torch.long, device=points.device)
    chosen[0] = start
    dist = (points - points[start]).norm(dim=-1)
    for i in range(1, num_samples):
        nxt = int(dist.argmax())
        chosen[i] = nxt
        dist = torch.minimum(dist, (points - points[nxt]).norm(dim=-1))
    return chosen.sort().values


def _kindyn_check(scene_dir: Path, keypoints: np.ndarray, person: int,
                  rows: np.ndarray) -> float:
    """Median distance (cm) between our MHR70 keypoints and kindyn ``joints_world``.

    A cross-rig diagnostic — a few cm is expected, metres means a frame bug.
    """
    with np.load(scene_dir / "kindyn_1.npz", allow_pickle=True) as kin:
        names = [str(x) for x in kin["joint_names"]]
        gt = np.asarray(kin["joints_world"], np.float32)[person][rows]
    gt = gt[:, [names.index(n) for n, _ in KINDYN_JOINT_CHECK]]
    ours = keypoints[:, [i for _, i in KINDYN_JOINT_CHECK]]
    return float(np.median(np.linalg.norm(ours - gt, axis=-1))) * 100.0


def process_scene(scene_dir: Path, gt: MHRGroundTruth, batch_frames: int) -> dict:
    """Replay every tracked person of one scene; return the npz payload."""
    with np.load(scene_dir / "mhr_1.npz", allow_pickle=True) as src:
        version = int(src["converter_version"])
        if version != REQUIRED_CONVERTER_VERSION:
            raise ValueError(f"converter_version {version} != {REQUIRED_CONVERTER_VERSION}")
        lbs_params = np.asarray(src["lbs_params"], np.float32)
        identity = np.asarray(src["identity"], np.float32)
        valid_mask = np.asarray(src["valid_mask"], bool)
        num_frames = np.asarray(src["num_frames"])
        fps = np.asarray(src["fps"], np.float32)

    n_people, n_frames = valid_mask.shape
    num_verts = int(gt.vert_indices.numel())
    kp_world = np.full((n_people, n_frames, NUM_KEYPOINTS, 3), np.nan, np.float32)
    verts_world = np.full((n_people, n_frames, num_verts, 3), np.nan, np.float32)
    kindyn_med = np.full((n_people,), np.nan, np.float32)

    for person in range(n_people):
        rows = np.flatnonzero(valid_mask[person]
                              & np.isfinite(lbs_params[person]).all(-1))
        if rows.size != int(valid_mask[person].sum()):
            print(f"  person {person}: {int(valid_mask[person].sum()) - rows.size} "
                  f"valid rows carry non-finite lbs_params (left NaN)", flush=True)
        if rows.size == 0:
            continue
        iden = torch.tensor(identity[person], device=gt.device)[None]
        for start in range(0, rows.size, batch_frames):
            chunk = rows[start:start + batch_frames]
            params = torch.tensor(lbs_params[person, chunk], device=gt.device)
            keypoints, verts = gt.evaluate(iden.expand(len(chunk), -1).contiguous(),
                                           params)
            kp_world[person, chunk] = keypoints.cpu().numpy()
            verts_world[person, chunk] = verts.cpu().numpy()
        kindyn_med[person] = _kindyn_check(scene_dir, kp_world[person, rows], person,
                                           rows)

    return dict(
        kp_world=kp_world, verts_world=verts_world,
        vert_indices=gt.vert_indices.cpu().numpy().astype(np.int64),
        kp_vs_kindyn_med_cm=kindyn_med,
        schema_version=np.int32(SCHEMA_VERSION),
        source_converter_version=np.int32(REQUIRED_CONVERTER_VERSION),
        num_frames=num_frames, fps=fps,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", type=Path,
                    default=Path("/data3/rikhat.akizhanov/better/data/ClimbingVideos"))
    ap.add_argument("--checkpoint-dir", type=Path, default=_DEFAULT_CKPT_DIR)
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="scene names; default = every scene with an mhr_1.npz")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-vertices", type=int, default=384)
    ap.add_argument("--batch-frames", type=int, default=256)
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split the scene list into this many interleaved shards")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    scene_dirs = sorted(
        p.parent for p in
        (args.corpus / "features" / "human_optim").glob("*/*/*/mhr_1.npz"))
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

    gt = MHRGroundTruth(args.checkpoint_dir, args.num_vertices, args.device)
    print(f"vertex subset: {args.num_vertices} of "
          f"{int(gt.keypoint_mapping.shape[1]) - 127} template vertices "
          f"(fps seed {FPS_SEED}, checksum {int(gt.vert_indices.sum())})", flush=True)

    done = skipped = failed = 0
    start_time = time.time()
    for i, scene_dir in enumerate(scene_dirs):
        target = scene_dir / "mhr_sup_1.npz"
        if target.exists() and not args.overwrite:
            skipped += 1
            continue
        tmp = target.with_name(f"mhr_sup_1.tmp{os.getpid()}.npz")
        try:
            payload = process_scene(scene_dir, gt, args.batch_frames)
            np.savez_compressed(tmp, **payload)
            tmp.rename(target)
        except Exception as exc:                           # keep the sweep alive
            failed += 1
            tmp.unlink(missing_ok=True)
            print(f"[{i + 1}/{len(scene_dirs)}] {scene_dir.name} FAILED: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            continue
        done += 1
        elapsed = time.time() - start_time
        print(f"[{i + 1}/{len(scene_dirs)}] {scene_dir.name}: "
              f"P={payload['kp_world'].shape[0]} N={payload['kp_world'].shape[1]} "
              f"kindyn {np.nanmedian(payload['kp_vs_kindyn_med_cm']):.2f} cm "
              f"({target.stat().st_size / 1e6:.1f} MB, "
              f"{elapsed / max(done, 1):.2f}s/scene)", flush=True)
    print(f"done={done} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
