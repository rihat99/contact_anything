"""Predict extremity contacts + 3D forces on BetterVideoReconstruction out-trees.

For every scene ``<out-root>/<stem>/`` with pipeline inputs (``sam3/bboxes.npz``,
``geometry/transform.npz``), the source video is decoded to frames and two
checkpoints run over them with the same centered sliding-window inference as
``scripts/render_climbing_video_contacts.py`` (each model uses its own config's
``data.sequence``; every valid (person, frame) is predicted exactly once):

  * ``--contact-checkpoint`` -> ``<stem>/predictions/contacts.npz``:
    per-limb probabilities + thresholded booleans.
  * ``--force-checkpoint``   -> ``<stem>/predictions/<--force-name>`` (default
    ``forces.npz``): per-limb 3D forces in body-weight units, in the head's own
    frame plus a world-frame copy. A ``local_world_aligned`` head is rotated
    into world via the scene's ``cam_from_world`` extrinsics; a ``root``-frame
    head (e.g. the supervised six-group model) is rotated by the kindyn root
    quaternion (``human_optim/kindyn_1.npz`` ``q[..., 3:7]``, xyzw,
    world-from-root), so that tree must have run the dynamics stage.

Limb order matches the checkpoint: ``left_hand, right_hand, left_foot,
right_foot`` for four-output heads, plus ``left_ankle, right_ankle`` (kindyn's
big-toe/heel split) for the six-group supervised head. Force-only builds (no
contact head) are supported for ``--force-checkpoint``; they store no
``contact_probs``.

Example (the ``out/`` corpus, videos named ``<stem>.mp4``)::

    python scripts/predict_reconstruction.py \
        --out-root ../BetterVideoReconstruction/out \
        --videos ../BetterVideoReconstruction/data \
        --contact-checkpoint output/<contact_run>/best.pth \
        --contact-config output/<contact_run>/config.yaml \
        --force-checkpoint output/<force_run>/best.pth \
        --force-config output/<force_run>/config.yaml

For per-scene video folders (e.g. campus board), use
``--video-pattern "{scene}/cam_left.mp4"``. The supervised six-group model
writes next to the originals via ``--force-name forces_sup.npz``.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import render_climbing_video_contacts as rcv

from contact import checkpoint as ckpt_io
from contact.config import load_config
from contact.data.climbing_corpus import (
    FORCE_GROUP_NAMES, _rows_by_object_id, quat_xyzw_to_matrix,
)
from contact.data.collate import make_collate
from contact.data.reconstruction_scenes import ReconstructionSceneDataset, extract_frames
from contact.engine import forward_model  # noqa: F401  (imported by rcv internally)
from contact.model import build_model
from contact.targets import EXTREMITY_4_NAMES, TargetSpec


def _load(checkpoint: Path, config: Path, device: str, kind: str) -> dict:
    """Build a model from ``config``, load ``checkpoint``, return its bundle."""
    cfg = load_config(config)
    model, _ = build_model(cfg, device)
    state = ckpt_io.load(checkpoint, model, config=cfg, map_location=device)
    model.eval()
    spec = TargetSpec.from_config(cfg)
    has_contact = getattr(model, "num_contact_tokens", 0) > 0
    if has_contact:
        if spec.joint_names != EXTREMITY_4_NAMES:
            raise ValueError(
                f"{checkpoint}: reconstruction prediction requires extremities_4; "
                f"got {spec.joint_names}")
        anchors = list(model.contact_keypoint_indices)
        if len(anchors) != 4:
            raise ValueError(f"{checkpoint}: expected four MHR anchors; got {anchors}")
        limb_names = list(EXTREMITY_4_NAMES)
    else:                                   # force-only build: anchors are the force tokens'
        if kind != "force":
            raise ValueError(f"{checkpoint}: contact prediction needs a contact head")
        anchors = list(model.force_keypoint_indices)
        if len(anchors) == len(FORCE_GROUP_NAMES):
            limb_names = list(FORCE_GROUP_NAMES)
        elif len(anchors) == 4:
            limb_names = list(EXTREMITY_4_NAMES)
        else:
            raise ValueError(
                f"{checkpoint}: force-only build with {len(anchors)} outputs — "
                f"only 4 (extremities) or {len(FORCE_GROUP_NAMES)} (kindyn groups) "
                f"have a known limb order")
    seq = cfg["data"]["sequence"]
    return {
        "model": model,
        "cfg": cfg,
        "collate": make_collate(tuple(model.cfg.MODEL.IMAGE_SIZE), spec),
        "anchors": anchors,
        "limb_names": limb_names,
        "has_contact": has_contact,
        "seq_len": int(seq["frames_per_clip"]),
        "stride": int(seq["frame_stride"]),
        "checkpoint": str(checkpoint),
        "epoch": int(state["epoch"]),
        "exp_name": str(cfg["output"]["exp_name"]),
    }


def _predict(bundle: dict, ds: ReconstructionSceneDataset, batch_size: int,
             device: str, collect_force: bool):
    """Sliding-window inference of one bundle over one scene.

    :returns: ``probs [P,N,K]``, anchor keypoints ``[P,N,K,2]`` and (when
        ``collect_force``) ``{"forces": [P,N,K,3], "anchor_cam": [P,N,K,3]}``;
        NaN rows are frames without a valid prediction (``probs`` stays NaN
        throughout for a force-only build).
    """
    data = ds._scenes[ds.scene]
    n_people, n_frames = data["valid_mask"].shape
    n_out = len(bundle["anchors"])
    probs = np.full((n_people, n_frames, n_out), np.nan, dtype=np.float32)
    points = np.full((n_people, n_frames, n_out, 2), np.nan, dtype=np.float32)
    force_data = None
    if collect_force:
        force_data = {
            "forces": np.full((n_people, n_frames, n_out, 3), np.nan, dtype=np.float32),
            "anchor_cam": np.full((n_people, n_frames, n_out, 3), np.nan, dtype=np.float32),
        }
    requests_by_t = rcv.sliding_window_requests(
        data["valid_mask"], bundle["seq_len"], bundle["stride"])
    for seq_len in sorted(requests_by_t):
        rcv._predict_requests(
            bundle["model"], ds, ds.scene, requests_by_t[seq_len], seq_len,
            batch_size, device, bundle["collate"], bundle["anchors"],
            probs, points, force_data,
        )
    return probs, points, force_data


def _provenance(bundle: dict, ds: ReconstructionSceneDataset, video: Path) -> dict:
    data = ds._scenes[ds.scene]
    return {
        "limbs": np.asarray(bundle["limb_names"]),
        "object_ids": data["object_ids"].astype(np.int32),
        "frame_indices": data["frame_indices"].astype(np.int32),
        "valid_mask": data["valid_mask"],
        "fps": np.float32(data["fps"]),
        "source_video": str(video),
        "checkpoint": bundle["checkpoint"],
        "checkpoint_epoch": np.int32(bundle["epoch"]),
        "exp_name": bundle["exp_name"],
        "windows": (
            f"centered sliding windows T={bundle['seq_len']} "
            f"stride={bundle['stride']}; each frame predicted once"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-root", type=Path, required=True,
                        help="root of pipeline out-trees (one subdir per scene)")
    parser.add_argument("--videos", type=Path, required=True,
                        help="root the video pattern is resolved against")
    parser.add_argument("--video-pattern", default="{scene}.mp4",
                        help="source video path relative to --videos ({scene} placeholder)")
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="scene subset (default: every out-root subdir with pipeline inputs)")
    parser.add_argument("--contact-checkpoint", type=Path, default=None)
    parser.add_argument("--contact-config", type=Path, default=None)
    parser.add_argument("--force-checkpoint", type=Path, default=None)
    parser.add_argument("--force-config", type=Path, default=None)
    parser.add_argument("--force-name", default="forces.npz",
                        help="filename for the force predictions inside predictions/")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="probability threshold for the stored contact booleans")
    parser.add_argument("--batch-size", type=int, default=16, help="windows per forward")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="frame-extraction scratch dir (default: a temp dir)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip scenes whose requested prediction files already exist")
    args = parser.parse_args()

    if args.contact_checkpoint is None and args.force_checkpoint is None:
        parser.error("provide --contact-checkpoint and/or --force-checkpoint")
    for kind in ("contact", "force"):
        ckpt = getattr(args, f"{kind}_checkpoint")
        if ckpt is not None and getattr(args, f"{kind}_config") is None:
            parser.error(f"--{kind}-checkpoint requires --{kind}-config")

    scenes = args.scenes or sorted(
        d.name for d in args.out_root.iterdir()
        if d.is_dir() and not d.name.startswith("_")
        and (d / "sam3" / "bboxes.npz").is_file()
        and (d / "geometry" / "transform.npz").is_file()
    )
    if not scenes:
        raise SystemExit(f"no scenes with pipeline inputs under {args.out_root}")
    videos = {s: args.videos / args.video_pattern.format(scene=s) for s in scenes}
    missing = [s for s in scenes if not videos[s].is_file()]
    if missing:
        raise SystemExit(f"missing source videos for scenes: {missing}")

    bundles = {}
    if args.contact_checkpoint is not None:
        print(f"Loading contact model {args.contact_checkpoint} …")
        bundles["contact"] = _load(
            args.contact_checkpoint, args.contact_config, args.device, "contact")
    if args.force_checkpoint is not None:
        print(f"Loading force model {args.force_checkpoint} …")
        bundles["force"] = _load(
            args.force_checkpoint, args.force_config, args.device, "force")
        force_cfg = bundles["force"]["cfg"]["model"]["force_head"]
        if not force_cfg["enabled"]:
            raise ValueError(f"{args.force_checkpoint}: config has no force head")
        if force_cfg["frame"] not in ("local_world_aligned", "root"):
            raise ValueError(
                f"{args.force_checkpoint}: only local_world_aligned or root force "
                f"heads are supported; got {force_cfg['frame']!r}")
        bundles["force"]["frame"] = force_cfg["frame"]

    work_root = args.work_dir or Path(tempfile.mkdtemp(prefix="predict_reconstruction_"))
    work_root.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for index, scene in enumerate(scenes, start=1):
        try:
            _run_scene(args, bundles, scene, index, len(scenes), videos[scene], work_root)
        except Exception as error:                # a broken scene must not kill the batch
            print(f"[{index}/{len(scenes)}] {scene}: FAILED — {error}")
            failures.append(scene)

    if args.work_dir is None:
        shutil.rmtree(work_root, ignore_errors=True)
    if failures:
        print(f"Done with {len(failures)} failed scene(s): {failures}")
        return 1
    print("Done.")
    return 0


def _run_scene(args, bundles: dict, scene: str, index: int, total: int,
               video: Path, work_root: Path) -> None:
    out_dir = args.out_root / scene
    pred_dir = out_dir / "predictions"
    targets = {
        kind: pred_dir / name
        for kind, name in (("contact", "contacts.npz"), ("force", args.force_name))
        if kind in bundles
    }
    if args.skip_existing and all(p.is_file() for p in targets.values()):
        print(f"[{index}/{total}] {scene}: predictions exist, skipping")
        return

    n_frames = len(np.load(out_dir / "geometry" / "transform.npz")["frame_indices"])
    frames_dir = work_root / scene
    print(f"[{index}/{total}] {scene}: extracting {n_frames} frames …")
    extract_frames(video, frames_dir, n_frames)
    ds = ReconstructionSceneDataset(out_dir, frames_dir, scene=scene)
    data = ds._scenes[scene]
    pred_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        if "contact" in bundles:
            probs, points, _ = _predict(
                bundles["contact"], ds, args.batch_size, args.device, False)
            contacts = np.where(np.isfinite(probs), probs >= args.threshold, False)
            np.savez_compressed(
                targets["contact"],
                probs=probs,
                contacts=contacts.astype(bool),
                threshold=np.float32(args.threshold),
                anchor_points_2d=points,
                **_provenance(bundles["contact"], ds, video),
            )
            print(f"  contacts.npz: {int(np.isfinite(probs).all(-1).sum())} predicted "
                  f"person-frames, contact fraction "
                  f"{float(contacts[np.isfinite(probs)].mean()):.3f}")

        if "force" in bundles:
            gate_probs, points, force_data = _predict(
                bundles["force"], ds, args.batch_size, args.device, True)
            forces = force_data["forces"]                       # [P,N,K,3] head frame, bw
            if bundles["force"]["frame"] == "local_world_aligned":
                # LWA (camera y-up) -> OpenCV camera frame, then cam -> world via
                # the metric cam_from_world rotation (rotation only: scale-free).
                f_cam = forces * rcv.FORCE_FLIP[None, None, None, :].astype(np.float32)
                rot = data["extrinsics"][:, :3, :3]             # [N,3,3] cam-from-world
                forces_world = np.einsum("nji,pnkj->pnki", rot, f_cam).astype(np.float32)
                extra = {"lwa_to_opencv_cam_flip": rcv.FORCE_FLIP.astype(np.float32)}
            else:                                               # root frame
                kindyn = np.load(
                    out_dir / "human_optim" / "kindyn_1.npz", allow_pickle=True)
                q = _rows_by_object_id(
                    np.asarray(kindyn["q"], np.float32), kindyn["object_ids"],
                    data["object_ids"], scene, "kindyn")        # [P,N,nq]
                kindyn_valid = _rows_by_object_id(
                    np.asarray(kindyn["valid_mask"], bool), kindyn["object_ids"],
                    data["object_ids"], scene, "kindyn")        # [P,N]
                if q.shape[1] != forces.shape[1]:
                    raise ValueError(
                        f"{scene}: kindyn covers {q.shape[1]} frames but the tree "
                        f"has {forces.shape[1]}")
                rot_wr = quat_xyzw_to_matrix(q[..., 3:7])       # [P,N,3,3] world-from-root
                forces_world = np.einsum(
                    "pnij,pnkj->pnki", rot_wr, forces).astype(np.float32)
                forces_world[~kindyn_valid] = np.nan            # no root orientation there
                extra = {"root_rotation_source":
                         "human_optim/kindyn_1.npz q[...,3:7] xyzw world-from-root"}
            if bundles["force"]["has_contact"]:
                extra["contact_probs"] = gate_probs
            np.savez_compressed(
                targets["force"],
                forces=forces,
                forces_world=forces_world,
                anchor_points_2d=points,
                anchor_cam=force_data["anchor_cam"],
                units="body_weight",
                force_frame=bundles["force"]["frame"],
                **extra,
                **_provenance(bundles["force"], ds, video),
            )
            mag = np.linalg.norm(forces, axis=-1)
            mag = mag[np.isfinite(mag)]
            print(f"  {targets['force'].name}: mean |f| {float(mag.mean()):.3f} bw, "
                  f"frac >0.05 bw {float((mag > 0.05).mean()):.3f}")

    shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
