"""GPU slow tests for the force-branch training integration (step 07).

Real SAM-3D-Body checkpoint required. Two things the trainer wiring must get right:

* **Warm-start behavioral** — after ``initialize_common_contact`` + ``freeze_contact``
  (regime (a)), the force model's contact ``joint_logits`` reproduce the source
  contact model's within the CUDA noise floor. This leans on the D1 asymmetric mask
  (force tokens never perturb contact logits) proved exactly in
  ``test_force_invariance.py``; here it is checked end-to-end through a real
  checkpoint. Skips (does NOT fabricate) when no ``climb4_frame`` contact run exists.
* **End-to-end smoke** — two DDP-wrapped training steps on a real climbing_corpus
  micro-batch with physics enabled: finite loss, force params move, frozen base does
  not; then one physics-INACTIVE batch through the same ``find_unused_parameters=False``
  DDP step (relies on PhysicsLoss's always-graph-connected ``joint_forces`` zero).
"""
from __future__ import annotations

import glob
import importlib.util
import os

import numpy as np
import pytest
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel

from contact import checkpoint as ckpt_io
from contact.config import load_config
from contact.data.climbing_corpus import ClimbingCorpusDataset, list_corpus_scenes
from contact.data.collate import batch_to_device, make_collate
from contact.engine import forward_model
from contact.losses import MultiTargetContactLoss
from contact.model import build_model
from contact.physics.adapter import _resolve_model_path
from contact.physics.loss import PhysicsLoss
from contact.targets import NUM_BODY_22, TargetSpec

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOINT_CFG = os.path.join(REPO, "configs", "climbing_videos_joint.yaml")
_CKPT = load_config(os.path.join(REPO, "configs", "base.yaml"))["model"]["checkpoint_path"]

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint missing"),
]


# ---------------------------------------------------------------- helpers

def _best_contact_ckpt() -> str | None:
    """Newest ``output/climb4_frame_*/best.pth`` (a per-frame extremities_4 contact
    run), or ``None`` — never fabricate a checkpoint."""
    cands = sorted(glob.glob(os.path.join(REPO, "output", "climb4_frame_*", "best.pth")),
                   key=os.path.getmtime)
    return cands[-1] if cands else None


def _force_cfg(*, freeze_contact: bool, use_warp: bool) -> dict:
    """Rebuild the retired warmstart / scratch experiment configs over the kept
    flattened joint config (the original yamls live in legacy/configs/): force
    head + RNEA physics on T=8 clips, contact frozen (regime a) or trainable (b).
    """
    cfg = load_config(JOINT_CFG)
    cfg["model"]["force_head"]["enabled"] = True
    cfg["physics"]["enabled"] = True
    cfg["physics"]["use_warp"] = use_warp
    cfg["train"]["freeze_contact"] = freeze_contact
    cfg["data"]["frames_per_batch"] = 32
    cfg["data"]["sequence"] = {"frames_per_clip": 8, "frame_stride": 1,
                               "jitter": True, "target_frame": "all"}
    return cfg


def _mhr_available() -> bool:
    try:
        _resolve_model_path(None, 1)
        return True
    except FileNotFoundError:
        return False


def _synth_frames(n: int):
    rng = np.random.RandomState(1234)
    frames = []
    for t in range(n):
        gt = torch.zeros(NUM_BODY_22)
        gt[t % NUM_BODY_22] = 1.0
        frames.append({
            "image": (rng.rand(200, 160, 3) * 255).astype(np.uint8),
            "mask": (np.ones((200, 160), np.uint8) * 255),
            "bbox": np.array([10.0, 10.0, 150.0, 190.0], np.float32),
            "cam_int": (np.eye(3, dtype=np.float32) * 500.0),
            "joint_contact": gt,
            "joint_mask": torch.ones(NUM_BODY_22),
            "joint_supervised": torch.ones(NUM_BODY_22),
            "joint_confidence": torch.ones(NUM_BODY_22),
            "frame_pos_sec": t * 0.1,
            "frame_valid": True,
        })
    return frames


def _batch(cfg, model, frames, seq_len=1):
    spec = TargetSpec.from_config(cfg)
    collate = make_collate(tuple(model.cfg.MODEL.IMAGE_SIZE), spec)
    if seq_len == 1:
        items = list(frames)
    else:
        assert len(frames) % seq_len == 0
        items = [frames[i:i + seq_len] for i in range(0, len(frames), seq_len)]
    return batch_to_device(collate(items), "cuda")


def _contact_logits(model, batch):
    out = forward_model(model, batch)
    return out["contact"]["joint_logits"].detach().float().clone()


def _max_abs(a, b):
    return float((a - b).abs().max())


# ---------------------------------------------------------- warm-start behavioral

def test_warm_start_preserves_contact_logits():
    """Regime (a) warm-start reproduces the source model's contact logits (D1)."""
    ckpt = _best_contact_ckpt()
    if ckpt is None:
        pytest.skip("no output/climb4_frame_*/best.pth contact checkpoint to warm-start from")

    # Source: a per-frame extremities_4 contact model with the checkpoint's contact
    # weights loaded. (The checkpoint predates the force arch keys, so strict
    # ckpt_io.load would reject it on the force-key diff — that resume path is not
    # what we test here; we only need the reference weights for the contact logits.)
    src_cfg = load_config(JOINT_CFG)
    torch.manual_seed(0)
    src_model, _ = build_model(src_cfg, "cuda")
    raw = torch.load(ckpt, map_location="cuda", weights_only=False)
    src_model.load_state_dict(raw["trainable_state_dict"], strict=False)
    src_model.eval()

    batch = _batch(src_cfg, src_model, _synth_frames(4), seq_len=1)
    a = _contact_logits(src_model, batch)
    b = _contact_logits(src_model, batch)
    floor = _max_abs(a, b)                       # base CUDA nondeterminism
    src_logits = a
    del src_model
    torch.cuda.empty_cache()

    # Force model: freeze_contact build (contact frozen) + warm-start the contact
    # branch from the same checkpoint (force branch stays fresh, zero-init).
    force_cfg = _force_cfg(freeze_contact=True, use_warp=True)
    force_cfg["model"]["init_contact_checkpoint"] = ckpt
    torch.manual_seed(0)
    force_model, _ = build_model(force_cfg, "cuda")
    state = ckpt_io.initialize_common_contact(
        ckpt, force_model, config=force_cfg, map_location="cuda")
    force_model.eval()
    assert any("force" in n for n in state["warm_start_new_names"]), state["warm_start_new_names"]
    # Contact params were frozen by freeze_contact, then loaded from the checkpoint.
    assert all(not p.requires_grad for n, p in force_model.named_parameters()
               if "contact" in n.lower())

    try:
        warm_logits = _contact_logits(force_model, batch)
        diff = _max_abs(warm_logits, src_logits)
        assert diff <= 8.0 * floor + 1e-6, (
            f"warm-started contact logits moved {diff:.2e} > 8x floor {floor:.2e} — "
            f"the force branch / warm start perturbed a frozen contact output")
    finally:
        del force_model
        torch.cuda.empty_cache()


# --------------------------------------------------------------- DDP smoke

def _real_micro_batch(cfg, model, n_clips=2, seq_len=8):
    """Two real climbing_corpus clips (T=8) collated + moved to CUDA, or None."""
    ds_cfg = yaml.safe_load(open(os.path.join(REPO, "configs", "datasets", "climbing_corpus.yaml")))
    root = ds_cfg["data"]["root"]
    if not os.path.isfile(os.path.join(root, "scenes", "scenes.db")):
        return None
    scenes = list_corpus_scenes(root, "train")[:6]
    ds = ClimbingCorpusDataset(
        root, scenes=scenes, split="train",
        frames_per_clip=seq_len, frame_stride=1, jitter=False,
        contact_level=int(ds_cfg["data"].get("contact_level", 1)),
        use_confidence_weights=True)
    if len(ds) < n_clips:
        return None
    spec = TargetSpec.from_config(cfg)
    collate = make_collate(tuple(model.cfg.MODEL.IMAGE_SIZE), spec)
    return batch_to_device(collate([ds[i] for i in range(n_clips)]), "cuda")


def test_force_physics_ddp_training_smoke():
    """Two DDP steps with physics move the force params and leave the frozen base
    fixed; a physics-inactive batch still passes the find_unused_parameters=False
    reducer (force params reached through PhysicsLoss's graph-connected zero)."""
    if not _mhr_available():
        pytest.skip("MHR archive unavailable — physics loss cannot be built")

    cfg = _force_cfg(freeze_contact=False, use_warp=False)
    torch.manual_seed(0)
    model, _ = build_model(cfg, "cuda")
    batch = _real_micro_batch(cfg, model)
    if batch is None:
        pytest.skip("no real climbing_corpus train clips available for the smoke batch")

    physics = PhysicsLoss(cfg, device="cuda")
    loss_fn = MultiTargetContactLoss(cfg).to("cuda")

    # The real _ContactForward wrapper, DDP-wrapped (world_size=1) so the reducer
    # runs with find_unused_parameters=False exactly as in distributed training.
    spec = importlib.util.spec_from_file_location(
        "train_mod", os.path.join(REPO, "scripts", "train.py"))
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = "29517"
    dist.init_process_group("nccl", rank=0, world_size=1)
    try:
        ddp = DistributedDataParallel(
            tm._ContactForward(model), device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3)

        force_before = {n: p.detach().clone()
                        for n, p in model.named_parameters() if "force" in n.lower()}
        frozen_sample = {
            n: p.detach().clone()
            for n, p in list(model.named_parameters())
            if not p.requires_grad}
        frozen_sample = dict(list(frozen_sample.items())[:5])
        assert force_before and frozen_sample

        def _step(step_batch) -> float:
            opt.zero_grad(set_to_none=True)
            model.train()
            out = ddp(step_batch)
            logits = {t: out["contact"][f"{t}_logits"] for t in loss_fn.target_names}
            contact_loss, _ = loss_fn(logits, step_batch["targets"])
            phys_total, phys_parts = physics(out, step_batch)
            loss = contact_loss + phys_total
            loss.backward()
            opt.step()
            return float(loss.detach()), phys_parts

        loss0, parts0 = _step(batch)
        loss1, _ = _step(batch)
        assert np.isfinite(loss0) and np.isfinite(loss1), (loss0, loss1)
        assert parts0["n_eligible_clips"] > 0 and parts0["n_residual_frames"] > 0, parts0

        # Force params moved; frozen base params did not.
        moved = [n for n, p in model.named_parameters()
                 if "force" in n.lower() and not torch.equal(p.detach(), force_before[n])]
        assert moved, "no force parameter changed after two physics training steps"
        for n, before in frozen_sample.items():
            after = dict(model.named_parameters())[n].detach()
            assert torch.equal(after, before), f"frozen base param {n} changed"

        # Physics-INACTIVE batch (all frames invalid -> zero eligible clips, no raise
        # since no frame is camera-valid-but-frame-invalid): the step must still run,
        # exercising the graph-connected joint_forces zero under find_unused=False.
        inactive = dict(batch)
        inactive["frame_valid"] = torch.zeros_like(batch["frame_valid"])
        loss2, parts2 = _step(inactive)
        assert np.isfinite(loss2), loss2
        assert parts2["n_eligible_clips"] == 0
        assert all(t["weight_mass"] == 0.0 for t in parts2["terms"].values())
    finally:
        dist.destroy_process_group()
        del model
        torch.cuda.empty_cache()
