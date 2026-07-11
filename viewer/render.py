"""Server-side raster rendering shared by the viewer's asset endpoints."""
from __future__ import annotations

import io
from functools import lru_cache

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

SMPL_NPZ = "/data3/rikhat.akizhanov/better/better_human/models/smpl/SMPL_NEUTRAL.npz"
SMPLX_NPZ = "/data3/rikhat.akizhanov/better/better_human/models/smplx/SMPLX_NEUTRAL.npz"
TEMPLATE_PATHS = {"smpl": SMPL_NPZ, "smplx": SMPLX_NPZ}

COLOR_CONTACT = np.array([0.86, 0.15, 0.20])
COLOR_NO_CONTACT = np.array([0.36, 0.48, 0.68])
MASK_OUTLINE_BGR = (48, 190, 96)


def overlay_annotations(bgr: np.ndarray, mask: np.ndarray | None,
                        bbox: np.ndarray | None, show_mask: bool,
                        show_bbox: bool) -> np.ndarray:
    out = bgr.copy()
    if show_mask and mask is not None:
        m = mask[..., 0] if mask.ndim == 3 else mask
        if m.shape[:2] != out.shape[:2]:
            m = cv2.resize(m, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, MASK_OUTLINE_BGR, 3, lineType=cv2.LINE_AA)
    if show_bbox and bbox is not None:
        x0, y0, x1, y1 = np.asarray(bbox).round().astype(int).tolist()
        cv2.rectangle(out, (x0, y0), (x1, y1), (245, 150, 36), 3, cv2.LINE_AA)
    return out


def encode_frame_jpeg(frame: dict, show_mask: bool = True,
                      show_bbox: bool = False) -> bytes:
    image = frame.get("image")
    if image is None:
        bgr = np.full((640, 640, 3), 241, dtype=np.uint8)
        cv2.putText(bgr, "Image unavailable", (150, 325), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (100, 110, 125), 2, cv2.LINE_AA)
    else:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        bgr = overlay_annotations(
            bgr, frame.get("mask"), frame.get("bbox"), show_mask, show_bbox)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


@lru_cache(maxsize=2)
def load_tpose(topology: str) -> tuple[np.ndarray, np.ndarray]:
    if topology not in TEMPLATE_PATHS:
        raise ValueError(f"unknown topology {topology!r}")
    d = np.load(TEMPLATE_PATHS[topology], allow_pickle=True)
    vertices = d["v_template"].astype(np.float32).copy()
    faces = d["f"].astype(np.int32)
    vertices[:, [1, 2]] *= -1
    vertices = np.stack([vertices[:, 0], vertices[:, 2], -vertices[:, 1]], axis=1)
    return vertices, faces


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-8)
    return normals


def _render_view(ax, vertices: np.ndarray, faces: np.ndarray,
                 contact: np.ndarray, title: str, elev: float, azim: float) -> None:
    azimuth, elevation = np.radians(azim), np.radians(elev)
    camera = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])
    normals = _face_normals(vertices, faces)
    visible = (normals @ camera) > 0
    shown_faces, shown_normals = faces[visible], normals[visible]

    key = np.array([0.5, -1.0, 0.8]); key /= np.linalg.norm(key)
    fill = np.array([-0.4, 1.0, 0.3]); fill /= np.linalg.norm(fill)
    shade = np.clip(0.40 + np.clip(shown_normals @ key, 0, 1)
                    + 0.65 * np.clip(shown_normals @ fill, 0, 1), 0, 1)
    face_contact = contact[shown_faces].any(axis=1)
    base = np.where(face_contact[:, None], COLOR_CONTACT, COLOR_NO_CONTACT)
    rgba = np.c_[np.clip(base * shade[:, None], 0, 1), np.ones(len(shown_faces))]
    ax.add_collection3d(Poly3DCollection(
        vertices[shown_faces], zsort="average", facecolor=rgba, edgecolor="none"))

    xlo, xhi = vertices[:, 0].min(), vertices[:, 0].max()
    ylo, yhi = vertices[:, 1].min(), vertices[:, 1].max()
    zlo, zhi = vertices[:, 2].min(), vertices[:, 2].max()
    span = max(xhi - xlo, zhi - zlo) * 0.38
    ax.set_xlim((xlo + xhi) / 2 - span, (xlo + xhi) / 2 + span)
    ax.set_ylim(ylo - 0.05 * (yhi - ylo), yhi + 0.05 * (yhi - ylo))
    ax.set_zlim((zlo + zhi) / 2 - span, (zlo + zhi) / 2 + span)
    ax.set_box_aspect([1, max((yhi - ylo) / (2 * span), 0.05), 1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=10, color="#5f6877", pad=0)
    ax.set_axis_off()


def render_tpose_png(contact: np.ndarray, topology: str = "smpl") -> bytes:
    vertices, faces = load_tpose(topology)
    contact = np.asarray(contact).astype(bool)
    if contact.shape != (vertices.shape[0],):
        raise ValueError(f"contact shape {contact.shape} does not match {topology} template")

    fig = plt.figure(figsize=(8, 6), dpi=110, facecolor="white")
    grid = fig.add_gridspec(1, 2, wspace=0, left=0, right=1, top=.96, bottom=0)
    front = fig.add_subplot(grid[0], projection="3d")
    back = fig.add_subplot(grid[1], projection="3d")
    front.set_facecolor("white"); back.set_facecolor("white")
    _render_view(front, vertices, faces, contact, "Front", 25, -90)
    _render_view(back, vertices, faces, contact, "Back", -25, 90)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return buf.getvalue()
