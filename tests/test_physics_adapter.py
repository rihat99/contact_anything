"""MHR adapter: 204-param mapping, world composition, and FK acceptance.

The adapter reproduces SAM-3D-Body's own MHR joints, so correctness is proven by
re-projecting the adapter's world-frame FK back through the same camera and the
axes flip and matching the model's ``pred_joint_coords`` (+ ``pred_cam_t``).

Two GPU proofs (mirroring ``test_temporal_invariance.py``'s real-checkpoint
fixture):

* **Exactness** — with one body per frame (``seq_len=1``) the re-projection is
  bit-exact (< 0.5 mm), proving the 204-slot mapping, the ``cam_from_world``
  inversion, and the ``diag(1,-1,-1)`` flip composition are all correct.
* **Clip contract** — one body per clip from the centre-frame shape leaves only
  the deliberate centre-shape approximation on shape-sensitive joints; the root
  stays exact and the mean stays within the 5 mm target.
"""
from __future__ import annotations

import os

import pytest
import torch

import better_robot as br
from better_human.bodies import MHRClassic
from better_robot.lie import so3

from contact.config import load_config
from contact.physics import EXTREMITY_OUTPUT_NAMES, MHRAdapter
from contact.physics.adapter import _resolve_model_path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CKPT = load_config(os.path.join(REPO, "configs", "base.yaml"))["model"]["checkpoint_path"]

try:
    _MHR_PATH = _resolve_model_path(None, 1)
    _HAS_MHR = True
except FileNotFoundError:
    _MHR_PATH, _HAS_MHR = None, False

_FLIP = torch.diag(torch.tensor([1.0, -1.0, -1.0]))


# --------------------------------------------------------------------- helpers

def _synthetic_cameras(batch: int, device, generator) -> torch.Tensor:
    """Random non-identity ``cam_from_world`` rigids ``(B, 4, 4)``."""
    rot = so3.to_matrix(so3.exp(torch.randn(batch, 3, generator=generator, device=device) * 0.5))
    trans = torch.randn(batch, 3, generator=generator, device=device) * 0.3
    cam = torch.eye(4, device=device).expand(batch, 4, 4).clone()
    cam[:, :3, :3] = rot
    cam[:, :3, 3] = trans
    return cam


def _reproject(body, q, cam_from_world, n_clips, seq_len) -> torch.Tensor:
    """World-frame FK mapped back to the camera: ``(n_clips, T, 127, 3)``."""
    data = body.fk(q)
    native = data.joint_pose_world.index_select(-2, body.structure.native_pose_joint_indices)
    world_pos = native[..., :3]
    cam = cam_from_world.view(n_clips, seq_len, 4, 4)
    rot_ext = cam[..., :3, :3].unsqueeze(2)
    trans_ext = cam[..., :3, 3].unsqueeze(2)
    return (rot_ext @ world_pos.unsqueeze(-1)).squeeze(-1) + trans_ext


def _expected_camera(adapter, mhr_out, n_clips, seq_len) -> torch.Tensor:
    """Model's own native joints flipped + translated: ``(n_clips, T, 127, 3)``.

    Independent of the adapter's world composition — native FK only.
    """
    shape = mhr_out["shape"]
    model_params = mhr_out["mhr_model_params"]
    pred_cam_t = mhr_out["pred_cam_t"]
    native_body, native_q = adapter.body.from_classic(
        MHRClassic(identity_coeffs=shape, model_parameters=model_params))
    data = br.forward_kinematics(native_body.robot, native_q)
    native = data.joint_pose_world.index_select(
        -2, native_body.structure.native_pose_joint_indices)[..., :3]
    flip = _FLIP.to(native)
    cam = (flip @ native.unsqueeze(-1)).squeeze(-1) + pred_cam_t.unsqueeze(1)
    return cam.view(n_clips, seq_len, 127, 3)


# ---------------------------------------------------------------- CPU synthetic

@pytest.mark.skipif(not _HAS_MHR, reason="MHR archive unavailable")
def test_world_composition_roundtrip_cpu():
    """Adapter world q, re-projected through cam_from_world + the flip, reproduces
    the native joints + pred_cam_t under a non-identity camera per frame.

    Shape is constant within a clip so the centre-shape body is exact; this
    isolates the 204-slot mapping, the translation handling, and the w2c inversion.
    """
    device = torch.device("cpu")
    adapter = MHRAdapter(model_path=_MHR_PATH, lod=1, device=device)
    n_clips, seq_len = 2, 3
    batch = n_clips * seq_len
    gen = torch.Generator(device=device).manual_seed(0)

    # Per-frame pose from a perturbed neutral q -> valid 204 params via to_classic;
    # zero the global translation as the frozen model does (mhr_head ~296).
    q_rand = adapter.body.robot.integrate(
        adapter.body.robot.q_neutral.expand(batch, -1),
        torch.randn(batch, adapter.body.robot.nv, generator=gen, device=device) * 0.2)
    model_params = adapter.body.to_classic(q_rand).model_parameters.clone()
    model_params[:, :3] = 0.0
    # Shape constant within each clip (so centre-frame shape == per-frame shape).
    shape_clip = torch.randn(n_clips, 45, generator=gen, device=device) * 0.3
    shape = shape_clip[:, None, :].expand(n_clips, seq_len, 45).reshape(batch, 45).contiguous()
    pred_cam_t = torch.tensor([0.0, 0.0, 3.0], device=device) + torch.randn(
        batch, 3, generator=gen, device=device) * 0.1
    cam_from_world = _synthetic_cameras(batch, device, gen)
    mhr_out = {"mhr_model_params": model_params, "shape": shape, "pred_cam_t": pred_cam_t}

    body, q = adapter.q_from_mhr_out(mhr_out, cam_from_world, n_clips, seq_len)
    assert q.shape == (n_clips, seq_len, 132)

    actual = _reproject(body, q, cam_from_world, n_clips, seq_len)
    expected = _expected_camera(adapter, mhr_out, n_clips, seq_len)
    err_mm = (actual - expected).norm(dim=-1)[..., 1:] * 1e3   # drop body_world reference
    assert err_mm.max() < 0.05, f"world composition drift {err_mm.max():.4f} mm"


@pytest.mark.skipif(not _HAS_MHR, reason="MHR archive unavailable")
def test_extremity_ids_resolve_by_name_cpu():
    """The four extremity BR joints resolve by native name (l_wrist/r_wrist/l_foot/
    r_foot); assert the resolved names, let the numeric ids float (``01_results.md``).
    """
    adapter = MHRAdapter(model_path=_MHR_PATH, lod=1, device="cpu")
    assert EXTREMITY_OUTPUT_NAMES == ("left_hand", "right_hand", "left_foot", "right_foot")
    assert adapter.extremity_joint_ids.shape == (4,)
    resolved = [adapter.body.robot.joint_names[i] for i in adapter.extremity_joint_ids.tolist()]
    assert resolved == ["l_wrist_ry", "r_wrist_ry", "l_lowleg_twist", "r_lowleg_twist"]


# ------------------------------------------------------- GPU real-checkpoint

def _requires_gpu_checkpoint(fn):
    for mark in (
        pytest.mark.slow,
        pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
        pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint missing"),
        pytest.mark.skipif(not _HAS_MHR, reason="MHR archive unavailable"),
    ):
        fn = mark(fn)
    return fn


def _real_clip_batch(seq_len: int, n_clips: int):
    from contact.data.climbing_corpus import (
        DEFAULT_ROOT, ClimbingCorpusDataset, list_corpus_scenes)
    from contact.data.collate import batch_to_device, make_collate
    from contact.model import build_model
    from contact.targets import TargetSpec

    try:
        scenes = list_corpus_scenes(DEFAULT_ROOT, "train")
    except FileNotFoundError:
        scenes = []
    if not scenes:
        pytest.skip("ClimbingVideos corpus unavailable")
    cfg = load_config(os.path.join(REPO, "configs", "climbing_videos_joint.yaml"))
    model, _ = build_model(cfg, "cuda")
    model.eval()
    dataset = ClimbingCorpusDataset(
        DEFAULT_ROOT, scenes=scenes[:1], split="val",
        frames_per_clip=seq_len, frame_stride=2, jitter=False)
    if len(dataset) < n_clips:
        pytest.skip("scene too short for the requested clip batch")
    collate = make_collate(tuple(model.cfg.MODEL.IMAGE_SIZE), TargetSpec.from_config(cfg))
    batch = batch_to_device(collate([dataset[i] for i in range(n_clips)]), "cuda")
    model._initialize_batch(batch)
    with torch.no_grad():
        out = model.forward_step(batch, decoder_type="body")
    return model, out["mhr"], batch


@_requires_gpu_checkpoint
def test_fk_exactness_per_frame_gpu():
    """One body per frame (seq_len=1): re-projection is bit-exact.

    This is the step's definition of done — it proves the 204-slot mapping, the
    ``cam_from_world`` inversion, and the flip composition on real model outputs.
    """
    model, mhr_out, batch = _real_clip_batch(seq_len=1, n_clips=6)
    try:
        adapter = MHRAdapter(model_path=_MHR_PATH, lod=1, device="cuda")
        n_clips = batch["cam_from_world"].shape[0]
        body, q = adapter.q_from_mhr_out(mhr_out, batch["cam_from_world"], n_clips, 1)
        actual = _reproject(body, q, batch["cam_from_world"], n_clips, 1)
        expected = _expected_camera(adapter, mhr_out, n_clips, 1)
        err_mm = (actual - expected).norm(dim=-1)[..., 1:] * 1e3
        assert err_mm.max() < 0.5, f"per-frame FK not exact: {err_mm.max():.4f} mm"
    finally:
        del model
        torch.cuda.empty_cache()


@_requires_gpu_checkpoint
def test_fk_acceptance_clip_gpu():
    """One body per clip (centre-frame shape): FK acceptance on real clips.

    The mapping is exact (see the per-frame test); the only residual is the
    deliberate centre-shape approximation on shape-sensitive joints. The physics-
    relevant quantities stay tight — the root is exact and the mean is well under
    5 mm — while the worst finger-joint error reaches ~1-3 cm in high-shape-
    variation clips (bounded loosely below, documented, not a mapping error).
    """
    seq_len, n_clips = 5, 3
    model, mhr_out, batch = _real_clip_batch(seq_len=seq_len, n_clips=n_clips)
    try:
        adapter = MHRAdapter(model_path=_MHR_PATH, lod=1, device="cuda")
        body, q = adapter.q_from_mhr_out(mhr_out, batch["cam_from_world"], n_clips, seq_len)
        actual = _reproject(body, q, batch["cam_from_world"], n_clips, seq_len)
        expected = _expected_camera(adapter, mhr_out, n_clips, seq_len)
        err = (actual - expected).norm(dim=-1)                       # (n_clips, T, 127)

        real = err[..., 1:] * 1e3                                    # drop body_world
        npi = body.structure.native_pose_joint_indices
        ext_native = [int((npi == jid).nonzero().flatten()[0]) for jid in adapter.extremity_joint_ids]
        root_mm = err[..., 1] * 1e3
        ext_mm = err[..., ext_native] * 1e3

        assert real.mean() < 5.0, f"mean joint error {real.mean():.3f} mm"
        assert root_mm.max() < 1.0, f"root not exact: {root_mm.max():.4f} mm"
        assert ext_mm.mean() < 10.0, f"extremity mean {ext_mm.mean():.3f} mm"
        assert real.max() < 35.0, f"centre-shape worst joint {real.max():.3f} mm"

        masses = adapter.total_mass(body)
        assert masses.shape == (n_clips,)
        assert torch.all((masses > 30.0) & (masses < 150.0))
    finally:
        del model
        torch.cuda.empty_cache()


@_requires_gpu_checkpoint
def test_extremity_fk_matches_mhr70_keypoints_gpu():
    """Extremity joints' FK positions match the model's MHR70 keypoints
    ``[62, 41, 13, 14]`` (per-frame, so exact-mapping; keypoints are regressed
    surface points a few mm off the joint origins).
    """
    model, mhr_out, batch = _real_clip_batch(seq_len=1, n_clips=6)
    try:
        adapter = MHRAdapter(model_path=_MHR_PATH, lod=1, device="cuda")
        n_clips = batch["cam_from_world"].shape[0]
        body, q = adapter.q_from_mhr_out(mhr_out, batch["cam_from_world"], n_clips, 1)
        data = body.fk(q)
        ext_world = data.joint_pose_world.index_select(-2, adapter.extremity_joint_ids)[..., :3]
        cam = batch["cam_from_world"].view(n_clips, 1, 4, 4)
        ext_cam = (cam[..., :3, :3].unsqueeze(2) @ ext_world.unsqueeze(-1)).squeeze(-1) \
            + cam[..., :3, 3].unsqueeze(2)
        kp = mhr_out["pred_keypoints_3d"][:, [62, 41, 13, 14]].view(n_clips, 1, 4, 3)
        kp_cam = kp + mhr_out["pred_cam_t"].view(n_clips, 1, 1, 3)
        err_mm = (ext_cam - kp_cam).norm(dim=-1) * 1e3
        assert err_mm.max() < 20.0, f"extremity vs MHR70 keypoint {err_mm.max():.3f} mm"
    finally:
        del model
        torch.cuda.empty_cache()
