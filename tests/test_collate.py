"""Clip collate: T=1 / T=4 shapes, mask_score, homogeneous-T, targets dict."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from contact.data.collate import DistributedEvalSampler, InterleavedLoader, make_collate
from contact.targets import NUM_BODY_22, TargetSpec

_IMG = (128, 128)
_VDIM = 32   # small vertex head for a fast test


def _frame(with_mask: bool = True, contact=None, joint=None):
    frame = {
        "image": (np.random.rand(96, 96, 3) * 255).astype(np.uint8),
        "mask": np.ones((96, 96), np.uint8) * 255 if with_mask else None,
        "bbox": np.array([8.0, 8.0, 88.0, 88.0], np.float32),
        "cam_int": (np.eye(3, dtype=np.float32) * 100.0),
    }
    if contact is not None:
        frame["contact"] = contact
    if joint is not None:
        frame["joint_contact"] = joint["gt"]
        frame["joint_mask"] = joint["mask"]
        frame["frame_pos_sec"] = joint.get("pos_sec", 0.0)
        frame["frame_valid"] = joint.get("valid", True)
    return frame


def _vertex_spec():
    return TargetSpec(enabled=["vertex"], topology="smpl", vertex_dims=_VDIM)


def _joint_spec():
    return TargetSpec(enabled=["joint"], topology="smpl", vertex_dims=_VDIM)


def test_t1_image_batch_shapes_and_targets():
    collate = make_collate(_IMG, _vertex_spec())
    contact = torch.zeros(_VDIM)
    contact[:5] = 1.0
    batch = collate([_frame(contact=contact), _frame(contact=contact)])

    assert batch["img"].shape == (2, 1, 3, *_IMG)
    assert batch["mask"].shape == (2, 1, 1, *_IMG)
    assert batch["seq_len"] == 1
    assert batch["frame_pos_sec"].shape == (2,)
    assert torch.all(batch["frame_pos_sec"] == 0.0)
    assert batch["frame_valid"].tolist() == [True, True]

    vt = batch["targets"]["vertex"]
    assert vt["gt"].shape == (2, _VDIM)
    assert vt["mask"].shape == (2, _VDIM)
    assert torch.all(vt["mask"] == 1.0)             # vertex fully supervised for image data
    assert float(vt["gt"][0].sum()) == 5.0


def test_t4_clip_batch_flattens_and_carries_joint_targets():
    collate = make_collate(_IMG, _joint_spec())
    clip = []
    for t in range(4):
        gt = torch.zeros(NUM_BODY_22)
        gt[t] = 1.0
        clip.append(_frame(joint={"gt": gt, "mask": torch.ones(NUM_BODY_22),
                                  "pos_sec": t * 0.1, "valid": True}))
    batch = collate([clip, clip])                    # 2 clips of T=4 -> B=8

    assert batch["img"].shape == (8, 1, 3, *_IMG)
    assert batch["seq_len"] == 4
    jt = batch["targets"]["joint"]
    assert jt["gt"].shape == (8, NUM_BODY_22)
    assert jt["mask"].shape == (8, NUM_BODY_22)
    assert torch.all(jt["mask"] == 1.0)
    # per-clip frame_pos_sec resets
    assert batch["frame_pos_sec"].tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3] * 2)


def test_t5_clip_batch_keeps_clip_major_frame_minor_order():
    collate = make_collate(_IMG, _joint_spec())
    clips = []
    for clip_id in range(2):
        clip = []
        for frame_id in range(5):
            gt = torch.zeros(NUM_BODY_22)
            gt[5 * clip_id + frame_id] = 1.0
            clip.append(_frame(joint={
                "gt": gt,
                "mask": torch.ones(NUM_BODY_22),
                "pos_sec": frame_id / 30.0,
                "valid": True,
            }))
        clips.append(clip)
    batch = collate(clips)
    assert batch["seq_len"] == 5
    assert batch["targets"]["joint"]["gt"].argmax(dim=1).tolist() == list(range(10))
    assert batch["frame_pos_sec"].tolist() == pytest.approx(
        [i / 30.0 for i in range(5)] * 2)


def test_missing_mask_scores_zero():
    collate = make_collate(_IMG, _vertex_spec())
    contact = torch.zeros(_VDIM)
    batch = collate([_frame(with_mask=True, contact=contact),
                     _frame(with_mask=False, contact=contact)])
    scores = batch["mask_score"].flatten().tolist()
    assert scores[0] == 1.0
    assert scores[1] == 0.0


def test_degenerate_bbox_fails_before_crop_transform():
    frame = _frame(joint={
        "gt": torch.zeros(NUM_BODY_22),
        "mask": torch.ones(NUM_BODY_22),
    })
    frame["bbox"] = np.zeros(4, np.float32)
    with pytest.raises(RuntimeError, match="invalid xyxy bbox"):
        make_collate(_IMG, _joint_spec())([frame])


def test_invalid_frame_mask_is_zero_for_joint():
    collate = make_collate(_IMG, _joint_spec())
    gt = torch.ones(NUM_BODY_22)
    valid = _frame(joint={"gt": gt, "mask": torch.ones(NUM_BODY_22), "valid": True})
    invalid = _frame(joint={"gt": gt, "mask": torch.zeros(NUM_BODY_22), "valid": False})
    batch = collate([[valid, invalid]])              # one clip, T=2
    jm = batch["targets"]["joint"]["mask"]
    assert float(jm[0].sum()) == NUM_BODY_22
    assert float(jm[1].sum()) == 0.0


def test_homogeneous_t_assertion():
    collate = make_collate(_IMG, _joint_spec())
    gt = torch.ones(NUM_BODY_22)
    clip4 = [_frame(joint={"gt": gt, "mask": torch.ones(NUM_BODY_22)}) for _ in range(4)]
    single = _frame(joint={"gt": gt, "mask": torch.ones(NUM_BODY_22)})   # T=1 item
    with pytest.raises(AssertionError, match="homogeneous-T"):
        collate([clip4, single])


def test_distributed_eval_sampler_is_exact_without_padding():
    dataset = list(range(7))
    shards = [
        list(DistributedEvalSampler(dataset, num_replicas=3, rank=rank))
        for rank in range(3)
    ]
    flat = [index for shard in shards for index in shard]
    assert sorted(flat) == list(range(7))
    assert len(flat) == len(set(flat))


def test_interleaved_loader_forwards_epoch_to_sampler_and_dataset():
    class EpochDataset(torch.utils.data.Dataset):
        epoch = None

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return index

        def set_epoch(self, epoch):
            self.epoch = epoch

    class EpochSampler(torch.utils.data.Sampler):
        epoch = None

        def __init__(self, dataset):
            self.dataset = dataset

        def __iter__(self):
            return iter(range(len(self.dataset)))

        def __len__(self):
            return len(self.dataset)

        def set_epoch(self, epoch):
            self.epoch = epoch

    dataset = EpochDataset()
    sampler = EpochSampler(dataset)
    child = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=sampler)
    loader = InterleavedLoader([child])
    loader.set_epoch(9)
    assert dataset.epoch == 9
    assert sampler.epoch == 9
