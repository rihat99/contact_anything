"""Focused API and schema tests for the standalone dataset viewer."""
from __future__ import annotations

import asyncio

import httpx
import numpy as np
import torch

from viewer.app import create_app
from viewer.datasets import DatasetSpec, ViewerDataset
from viewer.skeleton import JOINT_COORDS, JOINT_EDGES, skeleton_payload


def _frame(position: int, confidence: float = 0.5) -> dict:
    contact = torch.zeros(22)
    contact[20] = 1
    supervised = torch.ones(22)
    supervised[15] = 0
    return {
        "image": np.full((16, 20, 3), 128 + position, np.uint8),
        "mask": np.full((16, 20), 255, np.uint8),
        "bbox": np.array([1, 2, 18, 14], np.float32),
        "joint_contact": contact,
        "joint_mask": supervised,
        "joint_supervised": supervised,
        "joint_confidence": torch.full((22,), confidence),
        "frame_pos_sec": position / 30,
        "frame_position": position,
        "frame_index": 100 + position,
        "frame_valid": True,
        "key": f"scene#7@{position}",
    }


class _Manager:
    def __init__(self):
        spec = DatasetSpec(
            "fake_video", "Fake video", "joint", ("train",), "train",
            ("frame", "sequence"), "Synthetic test dataset",
        )
        self.ds = ViewerDataset([[_frame(3), _frame(4, 0.25)]], spec, "train", "sequence", 2, 1)

    def catalog(self):
        return [self.ds.spec.as_dict()]

    def get(self, dataset_id, split, mode="frame", clip_length=5, stride=1):
        if dataset_id != "fake_video":
            raise ValueError("unknown dataset")
        return self.ds


def test_body_22_skeleton_schema_is_complete():
    payload = skeleton_payload()
    assert len(payload["joint_names"]) == len(JOINT_COORDS) == 22
    assert len(JOINT_EDGES) == 21
    assert payload["joint_names"][20:22] == ["left_wrist", "right_wrist"]
    assert all(0 <= parent < 22 and 0 <= child < 22 for parent, child in JOINT_EDGES)
    assert len({child for _, child in JOINT_EDGES}) == 21


def test_sequence_sample_exposes_labels_confidence_and_source_frame_metadata():
    async def request():
        transport = httpx.ASGITransport(app=create_app(_Manager()))
        async with httpx.AsyncClient(transport=transport, base_url="http://viewer") as client:
            return await client.get(
                "/api/sample",
                params={"dataset": "fake_video", "split": "train", "mode": "sequence",
                        "clip_length": 2, "stride": 1, "index": 0},
            )

    response = asyncio.run(request())
    assert response.status_code == 200
    sample = response.json()
    assert sample["target"] == "joint"
    assert len(sample["frames"]) == 2
    assert sample["frames"][0]["frame_position"] == 3
    assert sample["frames"][0]["frame_index"] == 103
    assert sample["frames"][0]["joint_contact"][20] is True
    assert sample["frames"][0]["joint_supervised"][15] is False
    assert sample["frames"][1]["joint_confidence"][0] == 0.25


def test_image_endpoint_and_errors():
    async def requests():
        transport = httpx.ASGITransport(app=create_app(_Manager()))
        async with httpx.AsyncClient(transport=transport, base_url="http://viewer") as client:
            params = {"dataset": "fake_video", "split": "train", "mode": "sequence",
                      "clip_length": 2, "stride": 1, "index": 0, "frame": 1}
            return (
                await client.get("/api/image", params=params),
                await client.get("/api/image", params={**params, "frame": 2}),
                await client.get("/api/mesh", params=params),
                await client.get("/api/sample", params={
                    "dataset": "fake_video", "split": "train", "mode": "sequence", "index": 9}),
            )

    response, missing_frame, joint_mesh, missing_item = asyncio.run(requests())
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xff\xd8")
    assert missing_frame.status_code == 404
    assert joint_mesh.status_code == 400
    assert missing_item.status_code == 404
