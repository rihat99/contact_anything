"""Render the predicted body and its contact state onto ClimbingVideos scenes.

One disk per kindyn group is drawn at that group's anchor on the predicted
body: red in contact, green free. ``--overlay-labels`` splits each disk — the
outer ring is the prediction, the inner disk the corpus label (blank where the
label is not supervised).

**With an SMPL-X head** the body IS that head's prediction: its mesh
(painter-sorted, lambert-shaded, alpha-blended) and its 22 body joints are
drawn, and the disks sit on the SMPL-X joints of the six groups
(:data:`SMPLX_GROUP_JOINTS`). ``--gt-panel`` adds a second panel to the left
carrying the kindyn SMPL-X GT — its mesh, its joints and its label disks, the
latter drawn only where the label is supervised. Two panels are twice as wide
as the frame, so ``--scale 0.5`` is what keeps the canvas at one frame's width.

**Without an SMPL-X head** the body is the frozen MHR readout: the disks sit on
its MHR70 anchors and, when the build has a force head, the predicted 3D force
is drawn on top as a metric segment of :data:`METERS_PER_BW` metres per body
weight starting at the anchor's camera-frame position and perspective-projected
through the scene's own intrinsics, so its on-image direction and
foreshortening are the real camera's. A ``root``-frame head (the supervised
regime) is rotated to the camera with the GT world-from-root rotation;
``local_world_aligned`` by the fixed axis flip. Under ``--overlay-labels`` the
kindyn GT force is drawn as a thinner white arrow at the same anchor.

Only the frames of the evaluation clip are written (see
:mod:`scripts._render_common`), at the source fps divided by the clip stride.
Under ``torchrun`` the scenes are sharded over ranks; every rank writes its own
files, no process group needed.

    python scripts/render_video.py --config configs/temporal_tokens_b8_lr2.yaml \
        --checkpoint output/<run>/best.pth --split test --scenes 5 \
        --out output/<run>/render_contact --overlay-labels --gt-panel --scale 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _render_common as rc                                     # noqa: E402
from render_pose_video import draw_keypoints, draw_mesh         # noqa: E402
from data.climbing_videos.scene import GROUP_BODY22             # noqa: E402
from model.loss import (                                        # noqa: E402
    KINDYN_GROUP_KEYPOINTS,
    KINDYN_GROUP_NAMES,
    NUM_KINDYN_GROUPS,
)
from model.loss.smplx import BODY_Q_DIM, gt_smplx_camera, smplx_vertices    # noqa: E402
from train.predict import load_model                            # noqa: E402

#: BGR disk colours.
CONTACT_BGR = (45, 45, 235)
FREE_BGR = (55, 185, 75)
OUTLINE_BGR = (245, 245, 245)

#: SMPL-X body-22 joint anchoring each kindyn group, in group order — the very
#: joints the corpus folds its 52-joint labels onto (wrists, big toes, ankles).
SMPLX_GROUP_JOINTS = tuple(joints[0] for joints in GROUP_BODY22)
#: The panels of the SMPL-X path, left to right, and their banner text.
PANEL_LABELS = {
    "gt": "kindyn SMPL-X GT",
    "pred": "SMPL-X head + contact (outer = pred, inner = label)",
}
MESH_BGR = {"gt": (120, 200, 120), "pred": (208, 178, 130)}
KP_BGR = {"gt": (80, 220, 80), "pred": (200, 80, 220)}
BANNER_BGR = (25, 25, 25)
BANNER_TEXT_BGR = (240, 240, 240)

#: One arrow colour per kindyn group (LH, RH, LF, RF, LA, RA), BGR.
FORCE_BGR = ((15, 83, 224), (0, 165, 240), (216, 114, 28),
             (173, 179, 47), (201, 65, 139), (127, 54, 209))
FORCE_OUTLINE_BGR = (25, 25, 25)
GT_FORCE_BGR = (245, 245, 245)
METERS_PER_BW = 1.0        # arrow length in metres per unit body weight
MIN_FORCE_BW = 1.0e-3      # below this the arrow is noise (e.g. an untrained head)
FORCE_THICKNESS = 8
GT_FORCE_THICKNESS = 3
#: local_world_aligned (camera y-up) -> OpenCV camera frame (y-down, z-forward).
LWA_TO_CAM = np.array([1.0, -1.0, -1.0], np.float32)


def predict_scene(model, ds, cfg: dict, device: str, force_frame: str | None,
                  arms: tuple[str, ...]) -> dict:
    """Run every whole-scene clip; scatter the outputs into ``[P, N, ...]`` arrays.

    Rows the evaluation clips do not cover stay NaN and are never drawn.

    :param arms: the SMPL-X panels to build — ``()`` for the frozen-MHR path,
        ``("pred",)`` or ``("gt", "pred")`` for the SMPL-X head's.
    """
    data = ds.scene_data(ds.clips[0].scene)
    n_people, n_frames = data["valid_mask"].shape
    out: dict = {
        "probs": np.full((n_people, n_frames, NUM_KINDYN_GROUPS), np.nan, np.float32),
        "kp2d": np.full((n_people, n_frames, 70, 2), np.nan, np.float32),
        "kp3d_cam": np.full((n_people, n_frames, 70, 3), np.nan, np.float32),
        "covered": np.zeros((n_people, n_frames), bool),
    }
    if model.force_tokens is not None:
        out["forces"] = np.full(
            (n_people, n_frames, model.force_tokens.num_tokens, 3), np.nan, np.float32)
        if force_frame == "root":
            out["world_from_root"] = np.full(
                (n_people, n_frames, 3, 3), np.nan, np.float32)
    body = model.head_smplx.body(torch.device(device)) if arms else None
    if body is not None:
        out["faces"] = np.asarray(body.structure.faces.cpu().numpy(), np.int64)
        n_verts = int(out["faces"].max()) + 1
        for arm in arms:
            # fp16 vertices: 10475 of them per person-frame, and a pixel of
            # rounding is invisible at these magnitudes.
            out[f"{arm}_verts2d"] = np.full(
                (n_people, n_frames, n_verts, 2), np.nan, np.float16)
            out[f"{arm}_verts_cam"] = np.full(
                (n_people, n_frames, n_verts, 3), np.nan, np.float16)
            out[f"{arm}_kp2d"] = np.full((n_people, n_frames, 22, 2), np.nan, np.float32)

    for clip, batch, output in rc.clip_batches(ds, cfg, model, device):
        rows = batch["frame_index"].tolist()
        person = clip.person
        mhr = output["mhr"]
        kp2d = rc.to_numpy(mhr["pred_keypoints_2d"])
        kp3d_cam = rc.to_numpy(mhr["pred_keypoints_3d"] + mhr["pred_cam_t"][:, None, :])
        probs = (rc.to_numpy(output["contact"]["joint_probs"])
                 if output["contact"] is not None else None)
        forces = (rc.to_numpy(output["force"]["joint_forces"])
                  if output["force"] is not None else None)
        rot = (rc.to_numpy(batch["motion_rot"])
               if "world_from_root" in out else None)
        bodies = (smplx_arms(body, batch, output, arms, torch.device(device))
                  if body is not None else {})
        cam_int = rc.to_numpy(batch["cam_int"])
        for row, position in enumerate(rows):
            out["covered"][person, position] = True
            out["kp2d"][person, position] = kp2d[row]
            out["kp3d_cam"][person, position] = kp3d_cam[row]
            if probs is not None:
                out["probs"][person, position] = probs[row]
            if forces is not None:
                out["forces"][person, position] = forces[row]
            if rot is not None:
                out["world_from_root"][person, position] = rot[row]
            for arm, (verts, joints, valid) in bodies.items():
                if not valid[row]:
                    continue
                out[f"{arm}_verts_cam"][person, position] = verts[row]
                out[f"{arm}_verts2d"][person, position] = rc.project(verts[row], cam_int[row])
                out[f"{arm}_kp2d"][person, position] = rc.project(joints[row], cam_int[row])
    return out


def smplx_arms(body, batch: dict, output: dict, arms: tuple[str, ...],
               device: torch.device) -> dict:
    """Skin one clip's SMPL-X arms. ``arm -> (vertices, joints, valid)``, camera metres."""
    out = {}
    if "pred" in arms:
        smplx = output["smplx"]
        verts = smplx_vertices(body, smplx["betas"], smplx["q_cam"])
        out["pred"] = (rc.to_numpy(verts), rc.to_numpy(smplx["joints_cam"]),
                       np.ones(verts.shape[0], bool))
    if "gt" in arms:
        gt = gt_smplx_camera(batch, device, hands=body.robot.nq > BODY_Q_DIM)
        out["gt"] = (rc.to_numpy(smplx_vertices(body, gt["betas"], gt["q"])),
                     rc.to_numpy(gt["joints"]), rc.to_numpy(gt["valid"]) > 0)
    return out


def draw_contacts(frame: np.ndarray, xy: np.ndarray, probs: np.ndarray,
                  threshold: float, labels: np.ndarray | None,
                  label_valid: np.ndarray | None) -> None:
    """Draw one disk per group at ``xy [K, 2]`` for one person on one frame."""
    height, width = frame.shape[:2]
    radius = max(6, int(round(min(height, width) * 0.009)))
    outline = max(2, radius // 4)
    if labels is not None:
        radius = max(9, int(round(min(height, width) * 0.012)))
        inner = max(4, int(round(radius * 0.48)))
    for group, (probability, point) in enumerate(zip(probs, xy)):
        if not np.isfinite(probability) or not np.isfinite(point).all():
            continue
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        if not (0 <= x < width and 0 <= y < height):
            continue
        colour = CONTACT_BGR if probability >= threshold else FREE_BGR
        cv2.circle(frame, (x, y), radius + outline, OUTLINE_BGR, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), radius, colour, -1, cv2.LINE_AA)
        if labels is None:
            continue
        # The prediction stays readable as the outer ring; the label fills the
        # centre, and a white centre means "not supervised here".
        cv2.circle(frame, (x, y), inner + 2, OUTLINE_BGR, -1, cv2.LINE_AA)
        if label_valid[group] > 0:
            cv2.circle(frame, (x, y), inner,
                       CONTACT_BGR if labels[group] > 0.5 else FREE_BGR,
                       -1, cv2.LINE_AA)


def _arrow(force_cam: np.ndarray, point_cam: np.ndarray, anchor: np.ndarray,
           cam_int: np.ndarray, size: tuple[int, int]):
    """Project one camera-frame force to ``(start_px, end_px, length_px)`` or None."""
    height, width = size
    if not np.isfinite(point_cam).all() or point_cam[2] <= 1e-3:
        return None
    if not np.isfinite(anchor).all():
        return None
    if not (0 <= anchor[0] < width and 0 <= anchor[1] < height):
        return None
    tip_cam = point_cam + METERS_PER_BW * force_cam
    if tip_cam[2] <= 1e-3:
        return None
    base_px = rc.project(point_cam, cam_int)
    delta = rc.project(tip_cam, cam_int) - base_px
    length = float(np.linalg.norm(delta))
    if length < 2.0:                       # force nearly along the optical axis
        return None
    start = (int(round(anchor[0])), int(round(anchor[1])))
    return start, (start[0] + int(round(delta[0])), start[1] + int(round(delta[1]))), length


def draw_forces(frame: np.ndarray, anchors: np.ndarray, anchors_cam: np.ndarray,
                forces: np.ndarray, cam_int: np.ndarray, colours,
                thickness: int, tip_scale: float) -> None:
    """Draw one arrow per group; ``forces [K, 3]`` are already camera-frame."""
    size = frame.shape[:2]
    for group in range(forces.shape[0]):
        force = np.asarray(forces[group], np.float64)
        if not np.isfinite(force).all() or np.linalg.norm(force) < MIN_FORCE_BW:
            continue
        arrow = _arrow(force, np.asarray(anchors_cam[group], np.float64),
                       np.asarray(anchors[group], np.float64), cam_int, size)
        if arrow is None:
            continue
        start, end, length = arrow
        tip = float(np.clip(tip_scale / max(length, 1.0), 0.1, 0.5))
        cv2.arrowedLine(frame, start, end, FORCE_OUTLINE_BGR, thickness + 3,
                        cv2.LINE_AA, tipLength=tip)
        cv2.arrowedLine(frame, start, end, colours[group % len(colours)],
                        thickness, cv2.LINE_AA, tipLength=tip)


def forces_to_camera(forces: np.ndarray, frame_index: int, person: int,
                     preds: dict, extrinsics: np.ndarray, force_frame: str):
    """Rotate one person-frame's force vectors into the OpenCV camera frame."""
    if force_frame != "root":
        return forces * LWA_TO_CAM
    world_from_root = preds["world_from_root"][person, frame_index]
    if not np.isfinite(world_from_root).all():
        return None
    return np.einsum(
        "ij,jk,nk->ni", extrinsics[frame_index, :3, :3], world_from_root, forces)


def banner(img: np.ndarray, text: str) -> np.ndarray:
    """Stamp a panel's name into a dark strip along its top edge.

    The corpus frames are as narrow as 480 px, so the font is shrunk until the
    label fits the panel rather than running off its right edge.
    """
    font = min(0.8, img.shape[1] / 640.0)
    (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, 2)
    font *= min(1.0, (img.shape[1] - 24) / max(text_width, 1))
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), BANNER_BGR, -1)
    cv2.putText(img, text, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, font,
                BANNER_TEXT_BGR, 2, cv2.LINE_AA)
    return img


def smplx_panel(frame: np.ndarray, preds: dict, data: dict, position: int,
                arm: str, args) -> np.ndarray:
    """One SMPL-X panel: that arm's mesh, its 22 joints and its contact disks.

    The ``pred`` disks are the head's probabilities against ``--threshold``; the
    ``gt`` disks are the corpus labels, drawn only where they are supervised.
    """
    img = frame.copy()
    for person in range(preds["covered"].shape[0]):
        if not preds["covered"][person, position]:
            continue
        verts2d = preds[f"{arm}_verts2d"][person, position].astype(np.float32)
        if np.isfinite(verts2d).all():
            draw_mesh(img, verts2d,
                      preds[f"{arm}_verts_cam"][person, position].astype(np.float32),
                      preds["faces"], MESH_BGR[arm])
        kp2d = preds[f"{arm}_kp2d"][person, position]
        draw_keypoints(img, kp2d, KP_BGR[arm], 2)
        anchors = kp2d[list(SMPLX_GROUP_JOINTS)]
        labels = data["contact_gt"][person, position]
        valid = data["contact_valid"][person, position]
        if arm == "gt":
            draw_contacts(img, anchors, np.where(valid > 0, labels, np.nan), 0.5,
                          None, None)
        else:
            draw_contacts(img, anchors, preds["probs"][person, position],
                          args.threshold,
                          labels if args.overlay_labels else None,
                          valid if args.overlay_labels else None)
    return banner(img, PANEL_LABELS[arm])


def mhr_panel(frame: np.ndarray, preds: dict, data: dict, position: int,
              contact_anchors: list[int], force_anchors: list[int],
              force_frame: str | None, args) -> np.ndarray:
    """The frozen-MHR panel: contact disks and force arrows on its MHR70 anchors."""
    for person in range(preds["covered"].shape[0]):
        if not preds["covered"][person, position]:
            continue
        kp2d = preds["kp2d"][person, position]
        if contact_anchors:
            draw_contacts(
                frame, kp2d[contact_anchors], preds["probs"][person, position],
                args.threshold,
                data["contact_gt"][person, position] if args.overlay_labels else None,
                data["contact_valid"][person, position] if args.overlay_labels else None)
        if not force_anchors:
            continue
        cam_int = data["intrinsics"][position]
        anchors = kp2d[force_anchors]
        anchors_cam = preds["kp3d_cam"][person, position][force_anchors]
        forces = forces_to_camera(
            preds["forces"][person, position], position, person, preds,
            data["extrinsics"], force_frame)
        if forces is not None:
            draw_forces(frame, anchors, anchors_cam, forces, cam_int,
                        FORCE_BGR, FORCE_THICKNESS, 32.0)
        if not args.overlay_labels or "force_gt" not in data:
            continue
        gt = np.where(
            data["force_contact"][person, position][:, None],
            data["force_gt"][person, position], np.nan)
        gt_cam = forces_to_camera(
            gt, position, person, preds, data["extrinsics"], force_frame)
        if gt_cam is not None:
            draw_forces(frame, anchors, anchors_cam, gt_cam, cam_int,
                        (GT_FORCE_BGR,) * len(force_anchors),
                        GT_FORCE_THICKNESS, 24.0)
    return frame


def render_scene(ds, scene: str, preds: dict, args, model, force_frame: str | None,
                 arms: tuple[str, ...], out_path: Path) -> dict:
    """Write the scene's mp4; returns a small stats record."""
    data = ds.scene_data(scene)
    stride = ds.scene_stride(scene)
    positions = np.flatnonzero(preds["covered"].any(axis=0))
    if positions.size == 0:
        raise RuntimeError(f"{scene}: no frame was covered by an evaluation clip")
    contact_anchors = ([] if model.head_contact is None
                       else list(model.contact_tokens.keypoint_indices)
                       if model.contact_tokens is not None
                       else list(KINDYN_GROUP_KEYPOINTS))
    force_anchors = (list(model.force_tokens.keypoint_indices)
                     if model.force_tokens is not None else [])

    first = rc.read_frame(data["frames_dir"], int(positions[0]))
    height, width = first.shape[:2]
    size = (int(round(max(len(arms), 1) * width * args.scale)),
            int(round(height * args.scale)))
    fps = float(data["fps"]) / stride
    writer = rc.open_writer(out_path, fps, size)
    try:
        for position in positions:
            frame = rc.read_frame(data["frames_dir"], int(position))
            if arms:
                canvas = np.concatenate(
                    [smplx_panel(frame, preds, data, int(position), arm, args)
                     for arm in arms], axis=1)
            else:
                canvas = mhr_panel(frame, preds, data, int(position), contact_anchors,
                                   force_anchors, force_frame, args)
            if canvas.shape[1::-1] != size:
                canvas = cv2.resize(canvas, size, interpolation=cv2.INTER_AREA)
            writer.write(canvas)
    finally:
        writer.release()

    has_contact = model.head_contact is not None
    known = np.isfinite(preds["probs"]) if has_contact else np.zeros(0, bool)
    record = {"scene": scene, "output": str(out_path), "frames": int(positions.size),
              "size": list(size), "fps": round(fps, 3), "stride": stride,
              "people": int(preds["covered"].shape[0])}
    if has_contact:
        record["predicted_contact_fraction"] = round(float(
            (preds["probs"][known] >= args.threshold).mean()), 4)
        if args.overlay_labels:
            valid = known & (data["contact_valid"] > 0)
            agree = ((preds["probs"] >= args.threshold)
                     == (data["contact_gt"] > 0.5))[valid]
            record["label_agreement"] = round(float(agree.mean()), 4)
    if force_anchors:
        magnitude = np.linalg.norm(preds["forces"], axis=-1)
        record["mean_force_bw"] = round(float(
            np.nanmean(magnitude)) if np.isfinite(magnitude).any() else float("nan"), 4)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", default="none",
                        help="checkpoint path, or 'none' for the untrained model")
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--scenes", default=None,
                        help="scene count (e.g. '5') or a comma-separated id list; "
                             "default: every scene of the split")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overlay-labels", action="store_true",
                        help="corpus label as the inner disk, GT force as a white arrow")
    parser.add_argument("--gt-panel", action="store_true",
                        help="second panel on the left with the kindyn SMPL-X GT "
                             "body and its label disks (needs the SMPL-X head)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="scale factor of the written canvas; 0.5 keeps the "
                             "two panels of --gt-panel at one frame's width")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="test-split clip-length cap "
                             "(default: data.eval_max_frames; inert on --split train, "
                             "which runs data.clip.frames tiles)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    checkpoint = None if str(args.checkpoint).lower() == "none" else args.checkpoint
    model, cfg = load_model(args.config, checkpoint, args.device)
    if model.head_contact is None and model.force_tokens is None:
        raise SystemExit("this build has neither a contact nor a force head — "
                         "there is nothing to draw")
    if args.gt_panel and model.head_smplx is None:
        raise SystemExit("--gt-panel draws the kindyn SMPL-X GT against the SMPL-X "
                         "head's body; this build has no model.smplx head")
    root, contact_level = rc.dataset_spec(cfg)
    max_frames = args.max_frames or int(cfg["data"]["eval_max_frames"])

    force_frame = cfg["model"]["force"]["frame"] if model.force_tokens else None
    if force_frame == "local":
        raise SystemExit("force.frame 'local' needs forward kinematics that this "
                         "script does not run — no drawable arrows")
    load = set()
    if force_frame == "root":
        load.add("motion")            # motion_rot = the GT world-from-root rotation
    if args.overlay_labels and model.force_tokens is not None:
        load.add("forces")
    if args.gt_panel:
        load.add("smplx")             # the kindyn SMPL-X body of the GT panel
    arms = () if model.head_smplx is None else ("gt", "pred") if args.gt_panel else ("pred",)

    scenes, rank, world_size = rc.shard(
        rc.resolve_scenes(root, args.split, args.scenes))
    print(f"[rank {rank}/{world_size}] {len(scenes)} scene(s) on {args.device}; "
          f"checkpoint {checkpoint or 'none (untrained)'}; "
          f"groups {list(KINDYN_GROUP_NAMES)}")
    for index, scene in enumerate(scenes, start=1):
        ds = rc.build_dataset(cfg, root, contact_level, scene, args.split,
                              load, max_frames)
        preds = predict_scene(model, ds, cfg, args.device, force_frame, arms)
        record = render_scene(ds, scene, preds, args, model, force_frame, arms,
                              args.out / f"{scene}.mp4")
        print(f"[rank {rank}] [{index}/{len(scenes)}] {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
