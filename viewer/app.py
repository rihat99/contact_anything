"""FastAPI application for the contact dataset viewer."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .datasets import DatasetManager, ViewerDataset
from .render import encode_frame_jpeg, render_tpose_png
from .skeleton import skeleton_payload

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _dataset_or_error(manager: DatasetManager, dataset: str, split: str,
                      mode: str, clip_length: int, stride: int) -> ViewerDataset:
    try:
        return manager.get(dataset, split, mode, clip_length, stride)
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _item_or_error(ds: ViewerDataset, index: int):
    try:
        return ds.item(index)
    except IndexError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"index {index} is outside [0, {max(len(ds) - 1, 0)}]",
        ) from exc


def _frames(item, target: str) -> list[dict]:
    if target == "joint":
        return item
    return [item]


def _asset_url(path: str, dataset: str, split: str, mode: str,
               clip_length: int, stride: int, index: int, frame: int,
               **extra) -> str:
    query = {
        "dataset": dataset, "split": split, "mode": mode,
        "clip_length": clip_length, "stride": stride,
        "index": index, "frame": frame, **extra,
    }
    return f"{path}?{urlencode(query)}"


def _serialize_frame(frame: dict, frame_number: int, ds: ViewerDataset,
                     index: int, show_mask: bool, show_bbox: bool) -> dict:
    base = {
        "number": frame_number,
        "key": str(frame.get("key", index)),
        "image_url": _asset_url(
            "/api/image", ds.spec.id, ds.split, ds.mode, ds.clip_length,
            ds.stride, index, frame_number,
            show_mask=str(show_mask).lower(), show_bbox=str(show_bbox).lower(),
        ),
        "bbox": None if frame.get("bbox") is None
                else _as_numpy(frame["bbox"]).astype(float).tolist(),
    }
    if ds.spec.target == "joint":
        contact = _as_numpy(frame["joint_contact"]) > 0.5
        supervised = _as_numpy(frame.get("joint_supervised", frame["joint_mask"])) > 0.5
        confidence_value = frame.get("joint_confidence")
        confidence = (np.ones_like(contact, dtype=np.float32) if confidence_value is None
                      else _as_numpy(confidence_value).astype(np.float32))
        confidence = np.clip(confidence, 0.0, 1.0)
        scored_contact = contact & supervised
        base.update({
            "frame_valid": bool(frame.get("frame_valid", True)),
            "frame_position": int(frame.get("frame_position", -1)),
            "frame_index": int(frame.get("frame_index", -1)),
            "time_sec": float(frame.get("frame_pos_sec", 0.0)),
            "joint_contact": contact.tolist(),
            "joint_supervised": supervised.tolist(),
            "joint_confidence": confidence.astype(float).tolist(),
            "contact_count": int(scored_contact.sum()),
            "supervised_count": int(supervised.sum()),
            "mean_confidence": (float(confidence[supervised].mean())
                                if supervised.any() else None),
        })
    else:
        contact = _as_numpy(frame["contact"]) > 0.5
        topology = str(frame.get("vertex_topology", "smpl"))
        base.update({
            "contact_count": int(contact.sum()),
            "vertex_count": int(contact.size),
            "topology": topology,
            "mesh_url": _asset_url(
                "/api/mesh", ds.spec.id, ds.split, ds.mode, ds.clip_length,
                ds.stride, index, frame_number,
            ),
        })
        if frame.get("focal") is not None:
            base["focal"] = float(frame["focal"])
        if frame.get("depth_errors") is not None:
            base["depth_errors"] = {
                str(k): (v.item() if hasattr(v, "item") else v)
                for k, v in frame["depth_errors"].items()
            }
    return base


def create_app(manager: DatasetManager | None = None) -> FastAPI:
    manager = manager or DatasetManager()
    app = FastAPI(title="Contact dataset viewer", docs_url="/api/docs", redoc_url=None)
    app.state.dataset_manager = manager
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/catalog")
    async def catalog() -> dict:
        return {
            "datasets": manager.catalog(),
            "skeleton": skeleton_payload(),
            "sequence": {"default_length": 5, "max_length": 16, "default_stride": 1},
        }

    @app.get("/api/sample")
    async def sample(
        dataset: str = Query("climbing_videos"),
        split: str = Query("train"),
        mode: str = Query("frame"),
        clip_length: int = Query(5, ge=1, le=16),
        stride: int = Query(1, ge=1, le=8),
        index: int = Query(0, ge=0),
        show_mask: bool = Query(True),
        show_bbox: bool = Query(False),
    ) -> dict:
        ds = _dataset_or_error(manager, dataset, split, mode, clip_length, stride)
        item = _item_or_error(ds, index)
        frames = [
            _serialize_frame(frame, i, ds, index, show_mask, show_bbox)
            for i, frame in enumerate(_frames(item, ds.spec.target))
        ]
        return {
            "dataset": ds.spec.id,
            "dataset_label": ds.spec.label,
            "target": ds.spec.target,
            "split": ds.split,
            "mode": ds.mode,
            "index": index,
            "total": len(ds),
            "key": frames[0]["key"] if len(frames) == 1
                   else f'{frames[0]["key"]} → {frames[-1]["key"]}',
            "confidence_available": ds.spec.id == "climbing_videos" and ds.split == "train",
            "frames": frames,
        }

    @app.get("/api/image")
    async def image(
        dataset: str, split: str, mode: str = "frame",
        clip_length: int = Query(5, ge=1, le=16),
        stride: int = Query(1, ge=1, le=8),
        index: int = Query(..., ge=0), frame: int = Query(0, ge=0),
        show_mask: bool = True, show_bbox: bool = False,
    ) -> Response:
        ds = _dataset_or_error(manager, dataset, split, mode, clip_length, stride)
        item = _item_or_error(ds, index)
        frames = _frames(item, ds.spec.target)
        if frame >= len(frames):
            raise HTTPException(404, f"frame {frame} is outside this instance")
        try:
            payload = encode_frame_jpeg(frames[frame], show_mask, show_bbox)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(500, str(exc)) from exc
        return Response(payload, media_type="image/jpeg",
                        headers={"Cache-Control": "private, max-age=120"})

    @app.get("/api/mesh")
    async def mesh(
        dataset: str, split: str, mode: str = "frame",
        clip_length: int = Query(1, ge=1, le=16),
        stride: int = Query(1, ge=1, le=8),
        index: int = Query(..., ge=0), frame: int = Query(0, ge=0),
    ) -> Response:
        ds = _dataset_or_error(manager, dataset, split, mode, clip_length, stride)
        if ds.spec.target != "vertex":
            raise HTTPException(400, "joint-label datasets use the skeleton view, not a mesh")
        item = _item_or_error(ds, index)
        try:
            payload = render_tpose_png(
                _as_numpy(item["contact"]) > 0.5,
                str(item.get("vertex_topology", "smpl")),
            )
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(500, str(exc)) from exc
        return Response(payload, media_type="image/png",
                        headers={"Cache-Control": "private, max-age=120"})

    return app


app = create_app()
