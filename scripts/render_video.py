"""Render the SMPL-X head's body with its contact state (and forces) onto corpus scenes.

Every test scene is run under the evaluation protocol (one clip per person,
see :mod:`scripts._render_common`) and written as an mp4. The panel carries the
SMPL-X head's mesh (painter-sorted, lambert-shaded, alpha-blended), its 22 body
joints and one disk per kindyn group at that group's joint
(:data:`SMPLX_GROUP_JOINTS`): red in contact, green free. ``--overlay-labels``
splits each disk — the outer ring is the prediction, the inner disk the corpus
label (blank where the label is not supervised). ``--gt-panel`` adds a second
panel on the left carrying the kindyn SMPL-X GT — its mesh, its joints and its
label disks. Two panels are twice as wide as the frame, so ``--scale 0.5`` keeps
the canvas at one frame's width.

With a force head the predicted 3D force of every group (body-weight units,
body-root frame) is rotated into the camera with the head's OWN predicted root
rotation and drawn as an arrow of :data:`METERS_PER_BW` metres per body weight
from the group joint, perspective-projected through the scene's intrinsics.
Under ``--overlay-labels`` the kindyn GT force (rotated with the GT root) is a
thinner white arrow at the same joint.

Under ``torchrun`` the scenes are sharded over ranks; every rank writes its own
files, no process group needed.

    python scripts/render_video.py --config configs/baseline.yaml \\
        --checkpoint output/<run>/best.pth --scenes 5 \\
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
from _render_common import draw_keypoints, draw_mesh            # noqa: E402
from data.climbing_videos.scene import GROUP_BODY22             # noqa: E402
from model.loss import KINDYN_GROUP_NAMES, NUM_KINDYN_GROUPS    # noqa: E402
from model.loss.smplx import gt_smplx_camera, smplx_vertices    # noqa: E402
from train.predict import load_model                            # noqa: E402

#: BGR disk colours.
CONTACT_BGR = (45, 45, 235)
FREE_BGR = (55, 185, 75)
OUTLINE_BGR = (245, 245, 245)

#: SMPL-X body-22 joint anchoring each kindyn group, in group order — the very
#: joints the corpus folds its 52-joint labels onto (wrists, big toes, ankles).
SMPLX_GROUP_JOINTS = tuple(joints[0] for joints in GROUP_BODY22)
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


def predict_scene(model, ds, cfg: dict, device: str, arms: tuple[str, ...]) -> dict:
    """Run every whole-scene clip; scatter the outputs into ``[P, N, ...]`` arrays.

    Rows the evaluation clips do not cover stay NaN and are never drawn. Forces
    (predicted and, when loaded, GT) are stored already rotated into the camera.
    """
    data = ds.scene_data(ds.clips[0].scene)
    n_people, n_frames = data["valid_mask"].shape
    body = model.head_smplx.body(torch.device(device))
    faces = np.asarray(body.structure.faces.cpu().numpy(), np.int64)
    n_verts = int(faces.max()) + 1
    out: dict = {
        "faces": faces,
        "probs": np.full((n_people, n_frames, NUM_KINDYN_GROUPS), np.nan, np.float32),
        "covered": np.zeros((n_people, n_frames), bool),
    }
    if model.head_force is not None:
        out["forces_cam"] = np.full((n_people, n_frames, NUM_KINDYN_GROUPS, 3), np.nan, np.float32)
        if "force_gt" in data:
            out["gt_forces_cam"] = np.full_like(out["forces_cam"], np.nan)
    for arm in arms:
        # fp16 vertices: 10475 of them per person-frame, and a pixel of
        # rounding is invisible at these magnitudes.
        out[f"{arm}_verts2d"] = np.full((n_people, n_frames, n_verts, 2), np.nan, np.float16)
        out[f"{arm}_verts_cam"] = np.full((n_people, n_frames, n_verts, 3), np.nan, np.float16)
        out[f"{arm}_kp2d"] = np.full((n_people, n_frames, 22, 2), np.nan, np.float32)
        out[f"{arm}_kp3d"] = np.full((n_people, n_frames, 22, 3), np.nan, np.float32)

    for clip, batch, output in rc.clip_batches(ds, cfg, model, device):
        rows = batch["frame_index"].tolist()
        person = clip.person
        smplx = output["smplx"]
        probs = (rc.to_numpy(output["contact"]["probs"])
                 if output["contact"] is not None else None)
        cam_int = rc.to_numpy(batch["cam_int"])
        bodies = {"pred": (rc.to_numpy(smplx_vertices(body, smplx["betas"], smplx["q_cam"])),
                           rc.to_numpy(smplx["joints_cam"][:, :22]),
                           np.ones(len(rows), bool))}
        gt = None
        if "gt" in arms or "gt_forces_cam" in out:
            gt = gt_smplx_camera(batch, torch.device(device), hands=model.head_smplx.hands)
        if "gt" in arms:
            bodies["gt"] = (rc.to_numpy(smplx_vertices(body, gt["betas"], gt["q"])),
                            rc.to_numpy(gt["joints"][:, :22]), rc.to_numpy(gt["valid"]) > 0)
        forces_cam = gt_forces_cam = None
        if output["force"] is not None:
            root_rot = rc.to_numpy(smplx["root_rot"])                        # cam-from-root
            forces_cam = np.einsum("bij,bkj->bki", root_rot, rc.to_numpy(output["force"]["forces"]))
            if gt is not None and "gt_forces_cam" in out:
                gt_force = np.where(rc.to_numpy(batch["force_contact"])[..., None] > 0,
                                    rc.to_numpy(batch["force_gt"]), np.nan)
                gt_forces_cam = np.einsum("bij,bkj->bki", rc.to_numpy(gt["root_rot"]), gt_force)
        for row, position in enumerate(rows):
            out["covered"][person, position] = True
            if probs is not None:
                out["probs"][person, position] = probs[row]
            if forces_cam is not None:
                out["forces_cam"][person, position] = forces_cam[row]
            if gt_forces_cam is not None:
                out["gt_forces_cam"][person, position] = gt_forces_cam[row]
            for arm, (verts, joints, valid) in bodies.items():
                if not valid[row]:
                    continue
                out[f"{arm}_verts_cam"][person, position] = verts[row]
                out[f"{arm}_verts2d"][person, position] = rc.project(verts[row], cam_int[row])
                out[f"{arm}_kp3d"][person, position] = joints[row]
                out[f"{arm}_kp2d"][person, position] = rc.project(joints[row], cam_int[row])
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


def draw_forces(frame: np.ndarray, anchors: np.ndarray, anchors_cam: np.ndarray,
                forces_cam: np.ndarray, cam_int: np.ndarray, colours,
                thickness: int, tip_scale: float) -> None:
    """One arrow per group from the joint pixel ``anchors [K, 2]``; forces are camera-frame."""
    height, width = frame.shape[:2]
    for group in range(forces_cam.shape[0]):
        force = np.asarray(forces_cam[group], np.float64)
        point_cam = np.asarray(anchors_cam[group], np.float64)
        anchor = np.asarray(anchors[group], np.float64)
        if (not np.isfinite(force).all() or np.linalg.norm(force) < MIN_FORCE_BW
                or not np.isfinite(point_cam).all() or point_cam[2] <= 1e-3
                or not np.isfinite(anchor).all()
                or not (0 <= anchor[0] < width and 0 <= anchor[1] < height)):
            continue
        tip_cam = point_cam + METERS_PER_BW * force
        if tip_cam[2] <= 1e-3:
            continue
        delta = rc.project(tip_cam, cam_int) - rc.project(point_cam, cam_int)
        length = float(np.linalg.norm(delta))
        if length < 2.0:                       # force nearly along the optical axis
            continue
        start = (int(round(anchor[0])), int(round(anchor[1])))
        end = (start[0] + int(round(delta[0])), start[1] + int(round(delta[1])))
        tip = float(np.clip(tip_scale / max(length, 1.0), 0.1, 0.5))
        cv2.arrowedLine(frame, start, end, FORCE_OUTLINE_BGR, thickness + 3,
                        cv2.LINE_AA, tipLength=tip)
        cv2.arrowedLine(frame, start, end, colours[group % len(colours)],
                        thickness, cv2.LINE_AA, tipLength=tip)


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


def panel(frame: np.ndarray, preds: dict, data: dict, position: int, arm: str, args
          ) -> np.ndarray:
    """One panel: that arm's mesh, its 22 joints, its contact disks and force arrows.

    The ``pred`` disks are the head's probabilities against ``--threshold``; the
    ``gt`` disks are the corpus labels, drawn only where they are supervised.
    """
    img = frame.copy()
    groups = list(SMPLX_GROUP_JOINTS)
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
        anchors = kp2d[groups]
        labels = data["contact_gt"][person, position]
        valid = data["contact_valid"][person, position]
        if arm == "gt":
            draw_contacts(img, anchors, np.where(valid > 0, labels, np.nan), 0.5, None, None)
        else:
            draw_contacts(img, anchors, preds["probs"][person, position], args.threshold,
                          labels if args.overlay_labels else None,
                          valid if args.overlay_labels else None)
        key = "forces_cam" if arm == "pred" else "gt_forces_cam"
        if key in preds:
            cam_int = data["intrinsics"][position]
            anchors_cam = preds[f"{arm}_kp3d"][person, position][groups]
            colours, thick, tip = ((FORCE_BGR, FORCE_THICKNESS, 32.0) if arm == "pred"
                                   else ((GT_FORCE_BGR,) * 6, GT_FORCE_THICKNESS, 24.0))
            draw_forces(img, anchors, anchors_cam, preds[key][person, position], cam_int,
                        colours, thick, tip)
            if arm == "pred" and args.overlay_labels and "gt_forces_cam" in preds:
                draw_forces(img, anchors, anchors_cam, preds["gt_forces_cam"][person, position],
                            cam_int, (GT_FORCE_BGR,) * 6, GT_FORCE_THICKNESS, 24.0)
    return banner(img, PANEL_LABELS[arm])


def render_scene(ds, scene: str, preds: dict, args, arms: tuple[str, ...], out_path: Path
                 ) -> dict:
    """Write the scene's mp4; returns a small stats record."""
    data = ds.scene_data(scene)
    stride = ds.scene_stride(scene)
    positions = np.flatnonzero(preds["covered"].any(axis=0))
    if positions.size == 0:
        raise RuntimeError(f"{scene}: no frame was covered by an evaluation clip")
    first = rc.read_frame(data["frames_dir"], int(positions[0]))
    height, width = first.shape[:2]
    size = (int(round(len(arms) * width * args.scale)), int(round(height * args.scale)))
    fps = float(data["fps"]) / stride
    writer = rc.open_writer(out_path, fps, size)
    try:
        for position in positions:
            frame = rc.read_frame(data["frames_dir"], int(position))
            canvas = np.concatenate(
                [panel(frame, preds, data, int(position), arm, args) for arm in arms], axis=1)
            if canvas.shape[1::-1] != size:
                canvas = cv2.resize(canvas, size, interpolation=cv2.INTER_AREA)
            writer.write(canvas)
    finally:
        writer.release()

    record = {"scene": scene, "output": str(out_path), "frames": int(positions.size),
              "size": list(size), "fps": round(fps, 3), "stride": stride,
              "people": int(preds["covered"].shape[0])}
    known = np.isfinite(preds["probs"])
    if known.any():
        record["predicted_contact_fraction"] = round(float(
            (preds["probs"][known] >= args.threshold).mean()), 4)
        if args.overlay_labels:
            valid = known & (data["contact_valid"] > 0)
            agree = ((preds["probs"] >= args.threshold) == (data["contact_gt"] > 0.5))[valid]
            record["label_agreement"] = round(float(agree.mean()), 4)
    if "forces_cam" in preds:
        magnitude = np.linalg.norm(preds["forces_cam"], axis=-1)
        record["mean_force_bw"] = round(float(
            np.nanmean(magnitude)) if np.isfinite(magnitude).any() else float("nan"), 4)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", default="none",
                        help="checkpoint path, or 'none' for the untrained model")
    parser.add_argument("--scenes", default=None,
                        help="scene count (e.g. '5') or a comma-separated id list; "
                             "default: every test scene")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overlay-labels", action="store_true",
                        help="corpus label as the inner disk, GT force as a white arrow")
    parser.add_argument("--gt-panel", action="store_true",
                        help="second panel on the left with the kindyn SMPL-X GT body")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="scale factor of the written canvas; 0.5 keeps the "
                             "two panels of --gt-panel at one frame's width")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="clip-length cap (default: data.eval_max_frames)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    checkpoint = None if str(args.checkpoint).lower() == "none" else args.checkpoint
    model, cfg = load_model(args.config, checkpoint, args.device)
    if model.head_smplx is None:
        raise SystemExit("this build has no model.smplx head — nothing to draw the contacts on")
    root, contact_level = rc.dataset_spec(cfg)
    max_frames = args.max_frames or int(cfg["data"]["eval_max_frames"])
    load = {"smplx"} if args.gt_panel or model.head_force is not None else set()
    if args.overlay_labels and model.head_force is not None:
        load.add("forces")
    arms = ("gt", "pred") if args.gt_panel else ("pred",)

    scenes, rank, world_size = rc.shard(
        rc.resolve_scenes(root, "test", args.scenes, rc.dataset_camera(cfg)))
    print(f"[rank {rank}/{world_size}] {len(scenes)} scene(s) on {args.device}; "
          f"checkpoint {checkpoint or 'none (untrained)'}; "
          f"groups {list(KINDYN_GROUP_NAMES)}")
    for index, scene in enumerate(scenes, start=1):
        ds = rc.build_dataset(cfg, root, contact_level, scene, "test", load, max_frames)
        preds = predict_scene(model, ds, cfg, args.device, arms)
        record = render_scene(ds, scene, preds, args, arms, args.out / f"{scene}.mp4")
        print(f"[rank {rank}] [{index}/{len(scenes)}] {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
