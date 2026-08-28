"""Batch assembly — datasets -> SAM-3D-Body batch, images and clips alike.

Each dataset item is either a single frame-dict (still-image datasets) or a
**clip** (list of ``T`` frame-dicts, from :class:`ClimbingCorpusDataset`). The
collate normalises stills to length-1 clips, asserts the batch is homogeneous in
``T``, flattens ``[B_clips, T, ...]`` to a model batch ``[B_clips*T, ...]``, runs
the SAM-3D-Body top-down transform per frame, and builds the per-target
``{name: {"gt": [B, D], "mask": [B, D]}}`` supervision from
:meth:`contact.targets.TargetSpec.assemble_batch`.

Mixed image+video training uses **batch-level interleaving**: one DataLoader per
T-group, interleaved by remaining length (:class:`InterleavedLoader`) — never a
mixed-T batch.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Sampler, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import ToTensor

from sam_3d_body.data.transforms import (
    Compose, GetBBoxCenterScale, TopdownAffine, VisionTransformWrapper,
)
from ..targets import TargetSpec, validate_targets
from .climbing_corpus import (
    COND_FEATURE_DIM,
    FORCE_GROUP_NAMES,
    KP_JOINT_NAMES,
    MOTION_JOINT_NAMES,
    NUM_MOTION_JOINTS,
    ClimbingCorpusDataset,
    list_annotated_test_scenes,
    list_corpus_scenes,
)
from .climbing_images import ClimbingImagesDataset
from .damon import DamonDataset
from .splits import group_train_val_split, train_val_indices, video_id_from_scene


class DistributedEvalSampler(Sampler[int]):
    """Exact, non-padding validation shard for distributed evaluation."""

    def __init__(self, dataset, *, num_replicas: int, rank: int):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"rank {self.rank} outside [0, {self.num_replicas})")

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.num_replicas - 1) // self.num_replicas)


# -------------------------------------------------------------------- transform

def _build_transform(image_size: Tuple[int, int], *, imageless: bool = False):
    """SAM-3D-Body's standard top-down crop pipeline at the model resolution.

    :param imageless: drop the torchvision wrapper (it requires ``img``); the
        geometric transforms run identically without one (``TopdownAffine``
        skips a missing ``img`` and the warp matrix depends only on the bbox).
    """
    transforms = [
        GetBBoxCenterScale(),
        TopdownAffine(input_size=image_size, use_udp=False),
    ]
    if not imageless:
        transforms.append(VisionTransformWrapper(ToTensor()))
    return Compose(transforms)


def _process_sample(frame: dict, transform, transform_imageless=None):
    """Run the SAM-3D-Body transform on a single frame dict (image/mask/bbox).

    Frames with ``image is None`` and a header-read ``img_wh`` (the
    embedding-cache load path) take an imageless fast path: same geometry
    (bbox center/scale, affine, ``ori_img_size``) and the same mask warp, but
    no JPEG decode and a zero crop for ``img`` — the model provably never
    reads its values when ``batch["embedding"]`` is present.
    """
    img  = frame["image"]
    mask = frame["mask"]
    bbox = frame["bbox"]
    if bbox is None:
        raise RuntimeError(f"frame {frame.get('key', '?')} has no bbox")
    bbox = np.asarray(bbox, dtype=np.float32)
    if (bbox.shape != (4,) or not np.isfinite(bbox).all()
            or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
        raise RuntimeError(
            f"frame {frame.get('key', '?')} has invalid xyxy bbox {bbox.tolist()}")
    imageless = img is None and frame.get("img_wh") is not None
    has_mask = mask is not None
    if mask is None:
        if imageless:
            W, H = frame["img_wh"]
        else:
            H, W = img.shape[:2]
        mask = np.zeros((H, W, 1), dtype=np.uint8)
    elif mask.ndim == 2:
        mask = mask[..., None]

    # mask_score>0 tells the model this is a real mask (sam3d_body.py:883-889);
    # a substituted all-zeros mask must score 0.0 so it is treated as "no mask".
    data_info = dict(
        bbox=bbox,
        bbox_format="xyxy",
        mask=mask,
        mask_score=np.array(1.0 if has_mask else 0.0, dtype=np.float32),
    )
    if imageless:
        if transform_imageless is None:
            raise RuntimeError(
                f"frame {frame.get('key', '?')} has no decoded image "
                "(embedding-cache fast path) but no imageless transform was given")
        out = transform_imageless(data_info)
        out["ori_img_size"] = np.array(frame["img_wh"])             # [W, H]
        w, h = (int(v) for v in out["img_size"])
        out["img"] = torch.zeros(3, h, w)
    else:
        data_info["img"] = img
        out = transform(data_info)
    m = out["mask"]
    if m.ndim == 3:
        m = m[..., 0]
    out["mask"] = (m.astype(np.float32) / 255.0)
    return out


# -------------------------------------------------------------------- collate

def make_collate(image_size: Tuple[int, int], spec: TargetSpec,
                 motion_joints: int = NUM_MOTION_JOINTS, motion_dim: int = 6):
    """Build the batch collate.

    :param motion_joints: number of motion slots ``K`` the run's datasets emit
        (``motion_supervision.joint_names``); only the zero fallback for frames
        without motion targets depends on it.
    :param motion_dim: per-slot ``motion_gt`` width — ``6`` (vel|acc) or ``12``
        when ``motion_supervision.angular`` adds the root angular pair.
    """
    transform = _build_transform(image_size)
    transform_imageless = _build_transform(image_size, imageless=True)

    def _collate(batch):
        clips = [item if isinstance(item, list) else [item] for item in batch]
        seq_len = len(clips[0])
        if any(len(c) != seq_len for c in clips):
            raise AssertionError("homogeneous-T batches only: all clips must share the same length")
        frames = [f for clip in clips for f in clip]   # flatten -> B = B_clips * T

        per_sample = [_process_sample(f, transform, transform_imageless)
                      for f in frames]
        cam_ints = torch.stack([
            torch.as_tensor(f["cam_int"], dtype=torch.float32) for f in frames
        ], dim=0)

        keys = ["img", "img_size", "ori_img_size", "bbox_center", "bbox_scale",
                "bbox", "affine_trans", "mask", "mask_score"]
        out = {}
        for k in keys:
            if k not in per_sample[0]:
                continue
            tensors = [
                s[k] if isinstance(s[k], torch.Tensor) else torch.as_tensor(s[k])
                for s in per_sample
            ]
            out[k] = torch.stack(tensors, dim=0).float().unsqueeze(1)   # [B, 1, ...]
        if "mask" in out and out["mask"].dim() == 4:
            out["mask"] = out["mask"].unsqueeze(2)                      # [B, 1, 1, H, W]

        # Precomputed backbone embeddings (climbing_corpus with embedding_dir):
        # kept bf16 — forward_pose_branch casts to the img dtype, reproducing
        # the live path's bf16 -> fp32 cast bit-exactly.
        if "embedding" in frames[0]:
            out["embedding"] = torch.stack(
                [f["embedding"] for f in frames], dim=0)        # [B, C, h, w] bf16

        out["person_valid"] = torch.ones((len(frames), 1))
        out["cam_int"]      = cam_ints
        out["targets"]      = spec.assemble_batch(frames)
        out["seq_len"]      = seq_len
        out["frame_pos_sec"] = torch.tensor(
            [float(f.get("frame_pos_sec", 0.0)) for f in frames], dtype=torch.float32)
        out["frame_valid"] = torch.tensor(
            [bool(f.get("frame_valid", True)) for f in frames], dtype=torch.bool)

        # Per-frame camera pose + gravity for the physics loss. Datasets without
        # cameras (all still images) fall back to identity/zeros with cam_valid=False.
        out["cam_from_world"] = torch.stack([
            torch.as_tensor(f["cam_from_world"], dtype=torch.float32)
            if "cam_from_world" in f else torch.eye(4, dtype=torch.float32)
            for f in frames], dim=0)                                       # [B, 4, 4]
        out["gravity_world"] = torch.stack([
            torch.as_tensor(f["gravity_world"], dtype=torch.float32)
            if "gravity_world" in f else torch.zeros(3, dtype=torch.float32)
            for f in frames], dim=0)                                       # [B, 3]
        out["cam_valid"] = torch.tensor(
            ["cam_from_world" in f for f in frames], dtype=torch.bool)     # [B]
        # Camera-center jump (metres) between consecutive SAMPLED clip frames for
        # the physics camera-jerk filter (stride-consistent; see climbing_corpus.py);
        # 0.0 for frames without cameras (still images) or the first row of a clip.
        out["cam_jump_m"] = torch.tensor(
            [float(f.get("cam_jump_m", 0.0)) for f in frames], dtype=torch.float32)  # [B]

        # Supervised GT forces (climbing_corpus with load_forces): body-root
        # frame, body-weight units, kindyn group order. Frames without forces
        # fall back to zeros with force_valid=False, so mixed batches collate.
        n_groups = len(FORCE_GROUP_NAMES)
        out["force_gt"] = torch.stack([
            torch.as_tensor(f["force_gt"], dtype=torch.float32)
            if "force_gt" in f else torch.zeros(n_groups, 3, dtype=torch.float32)
            for f in frames], dim=0)                                       # [B, 6, 3]
        out["force_contact"] = torch.stack([
            torch.as_tensor(f["force_contact"], dtype=torch.bool)
            if "force_contact" in f else torch.zeros(n_groups, dtype=torch.bool)
            for f in frames], dim=0)                                       # [B, 6]
        # Root-frame lever arms (m) for the net-torque loss; the zeros fallback
        # is inert because those frames also carry force_valid=False.
        out["force_lever"] = torch.stack([
            torch.as_tensor(f["force_lever"], dtype=torch.float32)
            if "force_lever" in f else torch.zeros(n_groups, 3, dtype=torch.float32)
            for f in frames], dim=0)                                       # [B, 6, 3]
        out["force_conf"] = torch.tensor(
            [float(f.get("force_conf", 1.0)) for f in frames], dtype=torch.float32)
        out["force_valid"] = torch.tensor(
            [bool(f.get("force_valid", False)) for f in frames],
            dtype=torch.bool)                                              # [B]

        # Supervised GT motion (climbing_corpus with load_motion): root-frame
        # linear vel|acc per motion slot (m/s, m/s^2), the per-slot outlier bit,
        # the world-from-root rotation and the root body angular velocity. Frames
        # without motion fall back to zeros / False / identity with
        # motion_valid=False, so mixed batches collate.
        n_motion = motion_joints
        out["motion_gt"] = torch.stack([
            torch.as_tensor(f["motion_gt"], dtype=torch.float32)
            if "motion_gt" in f else torch.zeros(n_motion, motion_dim,
                                                 dtype=torch.float32)
            for f in frames], dim=0)                                       # [B, K, 6|12]
        out["motion_outlier"] = torch.stack([
            torch.as_tensor(f["motion_outlier"], dtype=torch.bool)
            if "motion_outlier" in f else torch.zeros(n_motion, dtype=torch.bool)
            for f in frames], dim=0)                                       # [B, K]
        out["motion_rot"] = torch.stack([
            torch.as_tensor(f["motion_rot"], dtype=torch.float32)
            if "motion_rot" in f else torch.eye(3, dtype=torch.float32)
            for f in frames], dim=0)                                       # [B, 3, 3]
        out["motion_omega"] = torch.stack([
            torch.as_tensor(f["motion_omega"], dtype=torch.float32)
            if "motion_omega" in f else torch.zeros(3, dtype=torch.float32)
            for f in frames], dim=0)                                       # [B, 3]
        out["motion_valid"] = torch.tensor(
            [bool(f.get("motion_valid", False)) for f in frames],
            dtype=torch.bool)                                              # [B]
        out["motion_root_pos"] = torch.stack([
            torch.as_tensor(f["motion_root_pos"], dtype=torch.float32)
            if "motion_root_pos" in f else torch.zeros(3, dtype=torch.float32)
            for f in frames], dim=0)                                       # [B, 3] world
        out["motion_root_valid"] = torch.tensor(
            [bool(f.get("motion_root_valid", False)) for f in frames],
            dtype=torch.bool)                                              # [B]

        # Kindyn-MHR pseudo-GT pose (climbing_corpus with load_pose): the fitted
        # world-frame MHR q per frame. Frames without targets fall back to zeros
        # with pose_valid=False, so mixed batches collate.
        out["pose_gt_q"] = torch.stack([
            torch.as_tensor(f["pose_gt_q"], dtype=torch.float32)
            if "pose_gt_q" in f else torch.zeros(132, dtype=torch.float32)
            for f in frames], dim=0)                                       # [B, 132]
        out["pose_valid"] = torch.tensor(
            [bool(f.get("pose_valid", False)) for f in frames],
            dtype=torch.bool)                                              # [B]

        # Kindyn GT keypoints (climbing_corpus with load_keypoints): the 13
        # KP_JOINT_NAMES joints in the metric world frame. Frames without
        # keypoints fall back to zeros with kp_valid=False, so mixed batches
        # collate.
        n_kp = len(KP_JOINT_NAMES)
        out["kp3d_world"] = torch.stack([
            torch.as_tensor(f["kp3d_world"], dtype=torch.float32)
            if "kp3d_world" in f else torch.zeros(n_kp, 3, dtype=torch.float32)
            for f in frames], dim=0)                                       # [B, 13, 3] world
        out["kp_valid"] = torch.tensor(
            [bool(f.get("kp_valid", False)) for f in frames],
            dtype=torch.bool)                                              # [B]

        # Input-side conditioning feature (climbing_corpus with cond_features_path):
        # standardized root-frame smoothed vel/acc + gravity direction + validity
        # bit. ALWAYS emitted so the model's zero-init projections are used on
        # every step (DDP rejects a param that some ranks leave unused); frames
        # and datasets without features contribute exact zeros, which is what the
        # bit-0 rows of the artifact itself look like.
        out["cond_feat"] = torch.stack([
            torch.as_tensor(f["cond_feat"], dtype=torch.float32)
            if "cond_feat" in f else torch.zeros(COND_FEATURE_DIM, dtype=torch.float32)
            for f in frames], dim=0)                                       # [B, 10]
        return out

    return _collate


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    return value


def batch_to_device(batch: dict, device: str) -> dict:
    for k, v in batch.items():
        if k == "img" and "embedding" in batch:
            # Embedding-cache batches carry a zero img stub the model reads
            # only for shape/dtype — allocate it on-device instead of copying
            # ~220 MB of zeros over PCIe every batch.
            batch[k] = torch.zeros(tuple(v.shape), dtype=v.dtype, device=device)
            continue
        batch[k] = _to_device(v, device)
    return batch


# -------------------------------------------------------------------- interleave

class InterleavedLoader:
    """Interleave whole batches from several DataLoaders (one per T-group).

    Batches are drawn proportionally to each loader's remaining length, seeded
    per epoch, so image (T=1) and video (T) batches mix without ever forming a
    mixed-T batch. :meth:`set_epoch` reseeds the interleave order and forwards to
    any child dataset exposing ``set_epoch`` (the video jitter).
    """

    def __init__(self, loaders: list[DataLoader], seed: int = 42):
        self.loaders = loaders
        self.seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        for loader in self.loaders:
            sampler = getattr(loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            ds = loader.dataset
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

    def __len__(self) -> int:
        return sum(len(loader) for loader in self.loaders)

    def __iter__(self):
        iters = [iter(loader) for loader in self.loaders]
        remaining = [len(loader) for loader in self.loaders]
        rng = random.Random(self.seed + self._epoch)
        while any(remaining):
            choice = rng.choices(range(len(iters)), weights=remaining)[0]
            try:
                yield next(iters[choice])
                remaining[choice] -= 1
            except StopIteration:
                remaining[choice] = 0


# -------------------------------------------------------------------- builders

def _build_image_dataset(spec: dict):
    """Construct one still-image sub-dataset from a ``data.datasets`` entry."""
    name = spec["name"]
    if name == "damon":
        return DamonDataset.from_config(spec["config"], split=spec.get("split", "trainval"))
    if name == "climbing":
        return ClimbingImagesDataset.from_config(spec["config"])
    raise ValueError(f"{name!r} is not a still-image dataset")


def _manifest_image_indices(manifest: dict, n: int) -> Tuple[list, list]:
    """Read the still-image split from a manifest; reject out-of-range indices."""
    if "images" not in manifest:
        raise ValueError("split manifest has no 'images' entry but the config has image datasets")
    train_idx = [int(i) for i in manifest["images"]["train"]]
    val_idx = [int(i) for i in manifest["images"]["val"]]
    referenced = set(train_idx) | set(val_idx)
    if referenced and max(referenced) >= n:
        raise ValueError(
            f"split manifest references image index {max(referenced)} but the current "
            f"dataset has only {n} items — the source data changed since the checkpoint")
    return train_idx, val_idx


def _manifest_video_ids(manifest: dict, key: str, scenes: list) -> Tuple[set, set]:
    """Read the video-group split from a manifest; reject ids missing from disk."""
    if key not in manifest:
        raise ValueError(f"split manifest has no {key!r} entry but the config lists that video dataset")
    train_vids = set(manifest[key]["train"])
    val_vids = set(manifest[key]["val"])
    present = {video_id_from_scene(sc) for sc in scenes}
    missing = (train_vids | val_vids) - present
    if missing:
        raise ValueError(
            f"split manifest {key!r} references source videos not present on disk: "
            f"{sorted(missing)} — the source data changed since the checkpoint")
    return train_vids, val_vids


def _manifest_train_test_scenes(
    manifest: dict, key: str, train_scenes: list[str], test_scenes: list[str],
) -> tuple[list[str], list[str]]:
    """Restore an exact all-train/manual-test scene manifest and detect drift."""
    if key not in manifest:
        raise ValueError(f"split manifest has no {key!r} entry but the config lists it")
    entry = manifest[key]
    if "train" not in entry or "test" not in entry:
        raise ValueError(
            f"split manifest {key!r} is not a train/test manifest: expected train and test")
    selected_train = [str(scene) for scene in entry["train"]]
    selected_test = [str(scene) for scene in entry["test"]]
    missing_train = set(selected_train) - set(train_scenes)
    missing_test = set(selected_test) - set(test_scenes)
    if missing_train or missing_test:
        raise ValueError(
            f"split manifest {key!r} references scenes no longer present/labelled; "
            f"missing train={sorted(missing_train)}, test={sorted(missing_test)}")
    return selected_train, selected_test


def make_loaders(
    cfg: dict,
    image_size: Tuple[int, int],
    manifest: dict | None = None,
    *,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
):
    """Build interleaved train + configured-evaluation loaders.

    Still-image datasets (damon/climbing) are concatenated and split randomly by
    ``val_ratio``/``seed`` (T=1 clips). ClimbingCorpus datasets are split by
    source video (no video crosses train/val) into windowed clips (T), unless
    ``data.eval_split=test``: then every curated train scene is used for
    training and annotated manual test scenes form evaluation. Batches keep a
    fixed ``frames_per_batch`` budget: ``B_clips = frames_per_batch // T``.

    :param manifest: when given, splits are taken **from the manifest** instead of
        re-derived (``{"images": {"train": [idx...], "val": [idx...]},
        "corpus:<config>": {"train": [vid...], "val": [vid...]}}``); missing
        members raise. Used by evaluate/resume to reproduce the exact split a
        checkpoint was trained on rather than the current directory's derivation.
    :param distributed_rank: Process rank for ``DistributedSampler`` sharding.
    :param distributed_world_size: Number of data-parallel processes. ``1`` keeps
        the original single-process loader behavior.
    :returns: ``(train_loader, eval_loader, split_manifest)`` — the manifest is the
        one used (echoing the input when supplied), so callers can persist it.
    """
    dcfg = cfg["data"]
    specs = list(dcfg["datasets"])
    spec_obj = TargetSpec.from_config(cfg)
    motion_names = tuple(cfg["motion_supervision"]["joint_names"] or MOTION_JOINT_NAMES)
    motion_angular = bool(cfg["motion_supervision"].get("angular", False))
    collate = make_collate(image_size, spec_obj, motion_joints=len(motion_names),
                           motion_dim=12 if motion_angular else 6)

    frames_per_batch = int(dcfg["frames_per_batch"])
    num_workers = int(dcfg["num_workers"])
    val_ratio = float(dcfg["val_ratio"])
    eval_split = str(dcfg["eval_split"])
    seed = int(dcfg["seed"])
    seq = dcfg["sequence"]
    clip_len = int(seq["frames_per_clip"])
    use_conf = bool(cfg["contact"]["targets"]["joint"]["use_confidence_weights"])

    image_specs = [s for s in specs if s["name"] in ("damon", "climbing")]
    corpus_specs = [s for s in specs if s["name"] == "climbing_corpus"]

    datasets_for_validation = []
    train_parts: list[tuple] = []   # (dataset, batch_size, shuffle)
    eval_parts: list[tuple] = []
    out_manifest: dict = {}

    if image_specs:
        subsets = [_build_image_dataset(s) for s in image_specs]
        datasets_for_validation += subsets
        ds = subsets[0] if len(subsets) == 1 else ConcatDataset(subsets)
        if manifest is not None:
            train_idx, val_idx = _manifest_image_indices(manifest, len(ds))
        else:
            train_idx, val_idx = train_val_indices(len(ds), val_ratio, seed)
        train_parts.append((Subset(ds, train_idx), frames_per_batch, True))
        eval_parts.append((Subset(ds, val_idx), frames_per_batch, False))
        out_manifest["images"] = {"train": [int(i) for i in train_idx],
                                  "val": [int(i) for i in val_idx]}
        sizes = " + ".join(f"{s['name']}={len(d)}" for s, d in zip(image_specs, subsets))
        if distributed_rank == 0:
            print(f"Image datasets [{sizes}] -> total={len(ds)} "
                  f"train={len(train_idx)} val={len(val_idx)} (val_ratio={val_ratio}, seed={seed})")

    if corpus_specs:
        clips_per_batch = max(1, frames_per_batch // clip_len)
        cond_cfg = cfg["model"]["cond_input"]
        for s in corpus_specs:
            ccfg = yaml.safe_load(Path(s["config"]).read_text())["data"]
            root = ccfg["root"]
            key = f"corpus:{s['config']}"
            common = dict(
                frames_per_clip=clip_len,
                # "auto" = per-scene max(1, round(fps/25)); see scene_stride.
                frame_stride=(seq["frame_stride"] if seq["frame_stride"] == "auto"
                              else int(seq["frame_stride"])),
                seed=seed, contact_level=int(ccfg.get("contact_level", 1)),
                use_confidence_weights=use_conf,
                load_forces=bool(ccfg.get("load_forces", False)),
                force_frame=str(cfg["force_supervision"]["gt_frame"]),
                force_units=str(cfg["force_supervision"]["units"]),
                load_motion=bool(ccfg.get("load_motion", False)),
                motion_joint_names=motion_names,
                motion_root_convention=str(
                    cfg["motion_supervision"]["root_convention"]),
                motion_target_smooth_sec=float(
                    cfg["motion_supervision"]["target_smooth_sec"]),
                motion_outlier_acc_ms2=float(
                    cfg["motion_supervision"]["loss"]["outlier_acc_ms2"]),
                motion_angular=motion_angular,
                load_pose=bool(cfg["pose_supervision"]["enabled"]),
                load_keypoints=bool(cfg["keypoint_supervision"]["enabled"]),
                # Input conditioning (model.cond_input): label-free, so it is a
                # loader option rather than a target — null path = no cond_feat.
                cond_features_path=(
                    cond_cfg["features_path"] if cond_cfg["enabled"] else None),
                cond_standardize=cond_cfg["standardize"],
                cond_clip=float(cond_cfg["clip"]),
                embedding_dir=(Path(root) / "features" / "embedding"
                               if bool(dcfg["embedding_cache"]) else None))
            all_train_scenes = list_corpus_scenes(root, "train")
            if eval_split == "test":
                all_test_scenes = list_annotated_test_scenes(root)
                if manifest is not None:
                    train_scenes, eval_scenes = _manifest_train_test_scenes(
                        manifest, key, all_train_scenes, all_test_scenes)
                else:
                    train_scenes, eval_scenes = all_train_scenes, all_test_scenes
                train_ds = ClimbingCorpusDataset(
                    root, scenes=train_scenes, split="train",
                    jitter=bool(seq["jitter"]), **common)
                eval_ds = ClimbingCorpusDataset(
                    root, scenes=eval_scenes, split="test", jitter=False, **common)
                out_manifest[key] = {
                    "train": sorted(train_scenes), "test": sorted(eval_scenes)}
                if distributed_rank == 0:
                    print(
                        f"ClimbingCorpus [{s['config']}]: all train scenes={len(train_scenes)} "
                        f"manual test scenes={len(eval_scenes)} | clips train={len(train_ds)} "
                        f"test={len(eval_ds)} (T={clip_len})")
            else:
                scenes = all_train_scenes
                if manifest is not None:
                    train_vids, val_vids = _manifest_video_ids(manifest, key, scenes)
                else:
                    train_vids, val_vids = group_train_val_split(
                        (video_id_from_scene(sc) for sc in scenes), val_ratio, seed)
                train_scenes = [sc for sc in scenes if video_id_from_scene(sc) in train_vids]
                eval_scenes = [sc for sc in scenes if video_id_from_scene(sc) in val_vids]
                train_ds = ClimbingCorpusDataset(
                    root, scenes=train_scenes, split="train",
                    jitter=bool(seq["jitter"]), **common)
                eval_ds = ClimbingCorpusDataset(
                    root, scenes=eval_scenes, split="val", jitter=False, **common)
                out_manifest[key] = {"train": sorted(train_vids), "val": sorted(val_vids)}
                if distributed_rank == 0:
                    print(
                        f"ClimbingCorpus [{s['config']}]: videos train={len(train_vids)} "
                        f"val={len(val_vids)} | scenes train={len(train_scenes)} "
                        f"val={len(eval_scenes)} | clips train={len(train_ds)} "
                        f"val={len(eval_ds)} (T={clip_len})")
            datasets_for_validation.extend((train_ds, eval_ds))
            train_parts.append((train_ds, clips_per_batch, True))
            eval_parts.append((eval_ds, clips_per_batch, False))

    validate_targets(cfg, datasets_for_validation)

    def _loader(dataset, batch_size, shuffle):
        # persistent_workers=False so each epoch's workers fork fresh from the main
        # process *after* set_epoch — the stateless video jitter reads the updated
        # epoch (persistent workers would freeze it at epoch 0). Re-fork is cheap:
        # the dataset (window index / npz metadata) is built once and only copied.
        sampler = None
        if distributed_world_size > 1:
            if shuffle:
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=distributed_world_size,
                    rank=distributed_rank,
                    shuffle=True,
                    seed=seed,
                    drop_last=True,
                )
            else:
                sampler = DistributedEvalSampler(
                    dataset,
                    num_replicas=distributed_world_size,
                    rank=distributed_rank,
                )
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=num_workers, drop_last=shuffle, collate_fn=collate,
            pin_memory=True, persistent_workers=False)

    train_loader = InterleavedLoader([_loader(*p) for p in train_parts], seed=seed)
    eval_loader = InterleavedLoader([_loader(*p) for p in eval_parts], seed=seed)
    return train_loader, eval_loader, out_manifest
