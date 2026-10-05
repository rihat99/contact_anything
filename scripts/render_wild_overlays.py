"""Overlay a run's predictions on an in-the-wild video: skeleton + force arrows.

For every out-tree ``<out-root>/<stem>/`` with ``predictions/<run>/{smplx,forces_sup}.npz``
(``scripts/predict_reconstruction.py``) the source video is decoded once more and, resized so
its long side is at most ``--max-side`` pixels, written as ``<stem>/predictions/<run>/overlay.mp4``
(the dump's fps; ``--results-dir DIR`` writes ``DIR/<stem>.mp4`` instead) and, with ``--png``,
frame by frame as well:

* ``<stem>/frames/<frame>.png`` — the original frame, and
* ``<stem>/predictions/<run>/overlay/<frame>.png`` — the overlay (``--png-dir DIR``: the overlays
  into ``DIR/<stem>/`` and no originals; ``--frame-ext jpg``: JPEG frames at quality 92;
  ``--crop-square SIDE``: every written image is a SIDE x SIDE square around the person — the
  box of the projected joints and arrow tips padded by ``CROP_PAD``, smoothed over
  ``CROP_SMOOTH_SEC`` — cut from the full-resolution frame).

The overlay is the frame fogged towards white
  (``FOG``, ``--fog 0`` leaves the frame as is), the predicted 22-joint SMPL-X body skeleton in white (camera-frame joints projected with
  the tree's intrinsics, bones to the parent joint) and one red (``--arrow-color yellow``: yellow) arrow per BODY joint that
  carries contact slots (the hand frames — palm, fingers, thumb — all fold onto the wrist, as
  in BVR, and the foot frames — toes, balls, heel — all fold onto the ANKLE, ``FOLD_JOINT``; the
  wrist's arrow is DRAWN from the base of the middle finger, where the palm ends,
  ``ANCHOR_JOINT``): the world-frame forces of the joint's slots whose contact probability is at least
  ``CONTACT_THRESHOLD`` are summed (an unlikely contact's phantom force is dropped), rotated
  into the camera, and drawn as a shaded 3D-looking arrow from the joint along the force at
  ``ARROW_M_PER_BW`` metres per body weight (so a near arrow is longer than a far one).

Frames without a prediction are fogged but carry no drawing.

    .venv/bin/python scripts/render_wild_overlays.py --out-root ../data/willd_videos/out \\
        --pred-dir joint_frames35 --stems full_body_1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _render_common import project                                          # noqa: E402
from model.contact_frames import body22_parent, contact_set                 # noqa: E402

FOG = 0.45                          # blend towards white: 0 = the frame as is, 1 = white
BONE_BGR = (255, 255, 255)          # white skeleton and joints with a thin black border
OUTLINE_BGR = (0, 0, 0)
#: Arrow palettes (body, lit stripe, shaded stripe), shaded like a 3D tube + cone.
CROP_PAD = 0.3        # square crop: side = the longest side of the person box x (1 + 2 x CROP_PAD)
CROP_SMOOTH_SEC = 0.3  # Gaussian over time on the crop centre and side
IMWRITE_PARAMS = {"png": [], "jpg": [cv2.IMWRITE_JPEG_QUALITY, 92]}
ARROW_PALETTES = {"red": ((30, 30, 215), (110, 110, 255), (10, 10, 130)),
                  "yellow": ((0, 200, 240), (120, 245, 255), (0, 120, 160))}
ARROW_M_PER_BW = 0.7                # arrow length in metres per body weight of force
ARROW_MIN_BW = 0.05                 # a joint below this force draws no arrow (a tiny arrow is clutter)
CONTACT_THRESHOLD = 0.5             # slots below this contact probability contribute no force
NEAR_M = 0.05                       # camera near plane for the arrows' end points
BONE_M = 0.02                       # stroke widths in metres at the pelvis depth, so a far
JOINT_M = 0.03                      # person is drawn as thin as a near one is thick
ARROW_M = 0.055
#: Parent of each of the 22 SMPL-X body joints (BetterHuman's joint order).
BODY22_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
FOLD_JOINT = {10: 7, 11: 8}         # foot (toes, balls) -> ankle: one arrow per foot
ANCHOR_JOINT = {20: 25, 21: 40}     # wrist -> middle-finger base (left / right) in the 52-joint body


def joint_forces_cam(forces_world: np.ndarray, probs: np.ndarray, slot_parent: np.ndarray,
                     joints: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    """Per-joint camera-frame forces ``(N, J, 3)``: the joint's in-contact slots summed, NaN where unknown."""
    n, _, _ = forces_world.shape
    out = np.full((n, len(joints), 3), np.nan, np.float32)
    for j, joint in enumerate(joints):
        members = slot_parent == joint
        part = forces_world[:, members]                                       # (N, k, 3)
        known = np.isfinite(part).all(-1)                                     # (N, k) whole vectors
        on = known & (np.nan_to_num(probs[:, members]) >= CONTACT_THRESHOLD)
        rows = known.any(-1)
        out[rows, j] = np.where(on[rows, :, None], part[rows], 0.0).sum(axis=1)
    return np.einsum("nij,nkj->nki", extrinsics[:, :3, :3], out)


def crop_windows(people: list[dict], intrinsics: np.ndarray, n_frames: int, width: int, height: int,
                 fps: float, pad: float = CROP_PAD, tips: bool = True) -> np.ndarray:
    """Per-frame square crop ``(N, 3)`` = ``x0, y0, side`` in original pixels around every drawn person.

    The box spans the projected 22 joints (and, with ``tips``, the arrow tips) of every covered
    person, padded by ``pad`` of its side on each side, smoothed over time and kept inside the
    image; frames without a person take the nearest frame's window.
    """
    centre = np.full((n_frames, 2), np.nan, np.float32)
    side = np.full(n_frames, np.nan, np.float32)
    for f in range(n_frames):
        pts = []
        for person in people:
            if not person["covered"][f]:
                continue
            body = person["body"][f]
            pts.append(project(body[body[:, 2] > NEAR_M], intrinsics[f]))
            if tips:
                force = np.nan_to_num(person["force"][f])
                ends = person["anchors"][f] + force * ARROW_M_PER_BW
                pts.append(project(ends[ends[:, 2] > NEAR_M], intrinsics[f]))
        pts = np.concatenate(pts) if pts else np.zeros((0, 2))
        if len(pts) == 0:
            continue
        lo, hi = pts.min(0), pts.max(0)
        centre[f] = (lo + hi) / 2
        side[f] = float((hi - lo).max()) * (1 + 2 * pad)
    known = np.isfinite(side)
    if not known.any():
        raise ValueError("no person to crop around")
    idx = np.arange(n_frames)
    for k in range(2):
        centre[:, k] = np.interp(idx, idx[known], centre[known, k])
    side = np.interp(idx, idx[known], side[known])
    sigma = CROP_SMOOTH_SEC * fps
    if sigma > 0:
        radius = int(3 * sigma)
        kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
        kernel /= kernel.sum()
        pad = lambda x: np.pad(x, radius, mode="edge")
        centre = np.stack([np.convolve(pad(centre[:, k]), kernel, mode="valid") for k in range(2)], -1)
        side = np.convolve(pad(side), kernel, mode="valid")
    side = np.minimum(side, min(width, height)).round().astype(np.int64)
    x0 = np.clip((centre[:, 0] - side / 2).round().astype(np.int64), 0, width - side)
    y0 = np.clip((centre[:, 1] - side / 2).round().astype(np.int64), 0, height - side)
    return np.stack([x0, y0, side], -1)


def fog(img: np.ndarray, amount: float) -> np.ndarray:
    """The frame blended towards white by ``amount`` (0 = untouched)."""
    if amount <= 0:
        return img
    white = np.full_like(img, 255)
    return cv2.addWeighted(img, 1.0 - amount, white, amount, 0.0)


def pixels_per_metre(joints_cam: np.ndarray, intrinsics: np.ndarray, scale: float) -> float:
    """Image pixels per metre at the pelvis depth."""
    depth = max(float(joints_cam[0, 2]), NEAR_M)
    return float(intrinsics[0, 0]) * scale / depth


def draw_skeleton(img: np.ndarray, joints_cam: np.ndarray, intrinsics: np.ndarray,
                  scale: float) -> None:
    """The 22-joint body: white bones to the parent joint and a white dot per joint, black-rimmed."""
    px_per_m = pixels_per_metre(joints_cam, intrinsics, scale)
    thickness = max(2, int(round(BONE_M * px_per_m)))
    border = max(1, thickness // 4)
    pts = project(joints_cam, intrinsics) * scale
    visible = joints_cam[:, 2] > NEAR_M
    bound = 4 * max(img.shape[:2])
    visible &= np.abs(pts).max(-1) < bound
    px = [(int(round(x)), int(round(y))) for x, y in pts]
    bones = [(j, p) for j, p in enumerate(BODY22_PARENTS) if p >= 0 and visible[j] and visible[p]]
    for j, p in bones:
        cv2.line(img, px[j], px[p], OUTLINE_BGR, thickness + 2 * border, cv2.LINE_AA)
    for j, p in bones:
        cv2.line(img, px[j], px[p], BONE_BGR, thickness, cv2.LINE_AA)
    radius = max(thickness, int(round(JOINT_M * px_per_m)))
    for j in np.flatnonzero(visible):
        cv2.circle(img, px[j], radius, BONE_BGR, -1, cv2.LINE_AA)
        cv2.circle(img, px[j], radius, OUTLINE_BGR, border, cv2.LINE_AA)


def draw_arrow_3d(img: np.ndarray, start: np.ndarray, end: np.ndarray, thickness: int,
                  palette: tuple) -> None:
    """One shaded arrow from ``start`` to ``end`` (pixels): a tube shaft and a cone head.

    The shaft is a thick line with a lighter stripe off-centre and a darker one on the other
    side (a lit cylinder); the head is a triangle split along its axis into a lit and a
    shaded half. Both carry a black rim. ``palette`` = (body, lit, shaded) BGR (``ARROW_PALETTES``).
    """
    body_bgr, light_bgr, dark_bgr = palette
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1.0:
        return
    axis = direction / length
    normal = np.array([-axis[1], axis[0]])
    head_len = min(length * 0.45, 3.0 * thickness)
    head_half = 1.6 * thickness
    base = end - axis * head_len
    rim = max(2, thickness // 3)

    def pt(p):
        return int(round(p[0])), int(round(p[1]))

    head = np.array([pt(end), pt(base + normal * head_half), pt(base - normal * head_half)], np.int32)
    cv2.line(img, pt(start), pt(base), OUTLINE_BGR, thickness + 2 * rim, cv2.LINE_AA)
    cv2.polylines(img, [head], True, OUTLINE_BGR, 2 * rim, cv2.LINE_AA)
    cv2.line(img, pt(start), pt(base), body_bgr, thickness, cv2.LINE_AA)
    offset = normal * thickness * 0.28
    cv2.line(img, pt(start - offset), pt(base - offset), light_bgr, max(1, thickness // 3),
             cv2.LINE_AA)
    cv2.line(img, pt(start + offset), pt(base + offset), dark_bgr, max(1, thickness // 4),
             cv2.LINE_AA)
    lit = np.array([pt(end), pt(base - normal * head_half), pt(base)], np.int32)
    shaded = np.array([pt(end), pt(base), pt(base + normal * head_half)], np.int32)
    cv2.fillPoly(img, [lit], light_bgr, cv2.LINE_AA)
    cv2.fillPoly(img, [shaded], dark_bgr, cv2.LINE_AA)
    cv2.polylines(img, [head], True, OUTLINE_BGR, rim, cv2.LINE_AA)


def draw_arrows(img: np.ndarray, anchors: np.ndarray, tips: np.ndarray, intrinsics: np.ndarray,
                scale: float, px_per_m: float, palette: tuple, thickness_m: float = ARROW_M) -> None:
    """Thick shaded arrows from ``anchors`` to ``tips`` (camera metres) on the resized ``img``.

    ``thickness_m`` is the shaft width in metres at ``px_per_m``. An arrow with an end
    behind the near plane, or one projecting far outside the image, is not drawn (its
    projection is meaningless).
    """
    thickness = max(4, int(round(thickness_m * px_per_m)))
    start = project(anchors, intrinsics) * scale
    end = project(tips, intrinsics) * scale
    bound = 4 * max(img.shape[:2])
    for a, b, za, zb in zip(start, end, anchors[:, 2], tips[:, 2]):
        if za > NEAR_M and zb > NEAR_M and np.abs(a).max() < bound and np.abs(b).max() < bound:
            draw_arrow_3d(img, a, b, thickness, palette)


def load_people(tree: Path, pred_dir: str) -> tuple[list[dict], dict]:
    """A run's dump on ``tree`` as drawable people plus the clip's camera and video facts.

    Each person: ``covered (N,)``, ``body (N, 22, 3)`` camera-frame joints, ``anchors (N, J, 3)``
    (the arrow start per drawn joint) and ``force (N, J, 3)`` camera-frame forces in body weight.
    The dict: ``intrinsics (N, 3, 3)``, ``video``, ``width``, ``height``, ``fps``, ``n_frames``.
    """
    pred = np.load(tree / "predictions" / pred_dir / "smplx.npz", allow_pickle=True)
    forces = np.load(tree / "predictions" / pred_dir / "forces_sup.npz", allow_pickle=True)
    camera = np.load(tree / "geometry" / "transform.npz", allow_pickle=True)
    intrinsics = np.asarray(camera["intrinsics_px_orig"], np.float32)
    extrinsics = np.asarray(camera["extrinsics"], np.float32)
    covered = np.asarray(pred["covered"], bool)                                # (P, N)
    n_people, n_frames = covered.shape
    slots = contact_set(str(pred["contact_set"]))
    slot_parent = np.asarray([FOLD_JOINT.get(body22_parent(j), body22_parent(j))
                              for j in slots.parent_joint52], np.int64)
    joints = np.unique(slot_parent)
    people = []
    for p in range(n_people):
        force = joint_forces_cam(forces["forces_world"][p], forces["contact_probs"][p], slot_parent,
                                 joints, extrinsics)
        people.append({"covered": covered[p], "body": pred["joints_cam"][p][:, :22],
                       "anchors": pred["joints_cam"][p][:, [ANCHOR_JOINT.get(int(j), int(j)) for j in joints]],
                       "force": force})
    clip = {"intrinsics": intrinsics, "video": Path(str(pred["source_video"])),
            "width": int(camera["image_width"]), "height": int(camera["image_height"]),
            "fps": float(pred["fps"]), "n_frames": n_frames}
    return people, clip


def crop_frame(img: np.ndarray, window: np.ndarray, side_out: int) -> np.ndarray:
    """The ``x0, y0, side`` square of ``img`` resized to ``side_out`` pixels."""
    x0, y0, side = window
    return cv2.resize(img[y0:y0 + side, x0:x0 + side], (side_out, side_out),
                      interpolation=cv2.INTER_AREA if side > side_out else cv2.INTER_LINEAR)


def render_tree(tree: Path, pred_dir: str, max_side: int, png: bool, fog_amount: float,
                results_dir: Path | None, png_dir: Path | None = None,
                frame_ext: str = "png", crop_square: int = 0,
                palette: tuple = ARROW_PALETTES["red"]) -> None:
    people, clip = load_people(tree, pred_dir)
    intrinsics, video, n_frames = clip["intrinsics"], clip["video"], clip["n_frames"]
    width, height = clip["width"], clip["height"]
    scale = 1.0 if crop_square else min(1.0, max_side / max(width, height))
    size = (crop_square, crop_square) if crop_square else (int(round(width * scale)),
                                                           int(round(height * scale)))
    windows = (crop_windows(people, intrinsics, n_frames, width, height, clip["fps"])
               if crop_square else None)
    frames_dir = tree / "frames"
    overlay_dir = tree / "predictions" / pred_dir / "overlay" if png_dir is None else png_dir / tree.name
    originals = png and png_dir is None
    if originals:
        frames_dir.mkdir(exist_ok=True)
    if png:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    mp4 = overlay_dir.with_suffix(".mp4") if results_dir is None else results_dir / f"{tree.name}.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), clip["fps"], size)
    drawn = 0
    try:
        for f in range(n_frames):
            ok, img = cap.read()
            if not ok:
                raise ValueError(f"{video}: decoded {f} frames, the dump has {n_frames}")
            if scale < 1.0:
                img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            if originals:
                cv2.imwrite(str(frames_dir / f"{f:06d}.{frame_ext}"), img, IMWRITE_PARAMS[frame_ext])
            img = fog(img, fog_amount)
            for person in people:
                if not person["covered"][f]:
                    continue
                draw_skeleton(img, person["body"][f], intrinsics[f], scale)
                force = person["force"][f]
                anchors = person["anchors"][f]
                on = np.isfinite(force).all(-1) & (np.linalg.norm(np.nan_to_num(force), axis=-1)
                                                   >= ARROW_MIN_BW)
                if on.any():
                    draw_arrows(img, anchors[on], anchors[on] + force[on] * ARROW_M_PER_BW,
                                intrinsics[f], scale,
                                pixels_per_metre(person["body"][f], intrinsics[f], scale), palette)
                drawn += 1
            if crop_square:
                img = crop_frame(img, windows[f], crop_square)
            writer.write(img)
            if png:
                cv2.imwrite(str(overlay_dir / f"{f:06d}.{frame_ext}"), img, IMWRITE_PARAMS[frame_ext])
    finally:
        cap.release()
        writer.release()
    print(f"  {mp4}: {n_frames} frames at {size[0]}x{size[1]}, "
          f"{drawn} person-frames drawn" + (f"; frames in {overlay_dir}" if png else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--pred-dir", required=True, help="the run's subfolder of predictions/")
    parser.add_argument("--stems", nargs="*", default=None,
                        help="subset (default: every tree with that run's dump)")
    parser.add_argument("--max-side", type=int, default=1920)
    parser.add_argument("--png", action="store_true",
                        help="also write the originals and the overlays as PNG frames")
    parser.add_argument("--png-dir", type=Path, default=None,
                        help="with --png: the overlay frames go to DIR/<stem>/ instead of the tree")
    parser.add_argument("--crop-square", type=int, default=0, metavar="SIDE",
                        help="write SIDExSIDE squares around the person (padded, time-smoothed) "
                             "instead of the whole frame")
    parser.add_argument("--frame-ext", choices=["png", "jpg"], default="png",
                        help="image format of the dumped frames (jpg = quality 92)")
    parser.add_argument("--arrow-color", choices=sorted(ARROW_PALETTES), default="red")
    parser.add_argument("--fog", type=float, default=FOG, help="blend towards white, 0 = none")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="write DIR/<stem>.mp4 instead of the tree's overlay.mp4")
    args = parser.parse_args()
    stems = args.stems or sorted(
        d.name for d in args.out_root.iterdir()
        if (d / "predictions" / args.pred_dir / "smplx.npz").is_file())
    if not stems:
        raise SystemExit(f"no predictions/{args.pred_dir} under {args.out_root}")
    if args.results_dir is not None:
        args.results_dir.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(stems, start=1):
        print(f"[{index}/{len(stems)}] {stem}", flush=True)
        render_tree(args.out_root / stem, args.pred_dir, args.max_side, args.png, args.fog,
                    args.results_dir, args.png_dir, args.frame_ext, args.crop_square,
                    ARROW_PALETTES[args.arrow_color])
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
