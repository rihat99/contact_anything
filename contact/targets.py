"""Contact targets: topologies, joint sets, and vertex->joint ownership.

Two supervision targets share the 24 contact tokens through independent heads:

* ``vertex`` — per-vertex contact in the body-model topology (SMPL 6890 or
  SMPL-X 10475).
* ``joint`` — per-joint contact over the 22-joint SMPL-X body set.

Vertex labels come from the still-image datasets; joint labels come from the
ClimbingVideos dataset.

.. warning::

   **Video joint labels and still-image derived joint labels are different
   tasks.** ClimbingVideos joint labels are motion-gated *stable* contact
   (stillness + hysteresis + min-duration + gap-merge, see
   ``BetterVideoReconstruction/scripts/stages/estimate_contacts.py``). A joint
   label obtained by lifting a *still* image's per-vertex contact
   (:func:`derive_joint_contact`) is instantaneous *surface* contact — it has no
   temporal gating. Mixing the two supervises one head with two different label
   semantics, so ``joint.derive_from_vertex`` defaults **off**; enabling it is a
   deliberate per-experiment choice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor

# Body-model vertex counts. "mhr" (18439) is the model's native topology but is
# not a supported *training* target yet -> NotImplementedError at config time.
TOPOLOGY_VERTS = {"smpl": 6890, "smplx": 10475}

# SMPL-X body-22 joint set (copied from ClimbingVideos_v1/dataset_info.json).
SMPLX_BODY_22 = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]
NUM_BODY_22 = len(SMPLX_BODY_22)

# Joints an annotator can label in the manual test set (evaluation only). The
# other 8 (pelvis, spine1/2/3, neck, collars, head) stay non-contact.
OBSERVABLE_14 = [1, 2, 4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21]

# SMPL joints 22/23 are the hands; fold them onto the wrists (20/21) so a
# 24-joint SMPL ownership map lands in the 22-joint body set.
_SMPL_HAND_TO_WRIST = {22: 20, 23: 21}

SMPL_NEUTRAL_NPZ = "/data3/rikhat.akizhanov/better/better_human/models/smpl/SMPL_NEUTRAL.npz"

_OWNER_CACHE: dict[str, Tensor] = {}


def topology_num_vertices(topology: str) -> int:
    """Return the vertex count for a topology name.

    :param topology: ``"smpl"`` or ``"smplx"``.
    :raises NotImplementedError: for ``"mhr"`` (native topology, not a target yet).
    :raises ValueError: for any other name.
    """
    if topology == "mhr":
        raise NotImplementedError(
            "topology 'mhr' (18439 native verts) is not a supported training target")
    if topology not in TOPOLOGY_VERTS:
        raise ValueError(f"unknown topology {topology!r}; choose from {sorted(TOPOLOGY_VERTS)}")
    return TOPOLOGY_VERTS[topology]


def compute_vertex_joint_owner(
    betas: Optional[Sequence[float]] = None,
    smpl_npz: str = SMPL_NEUTRAL_NPZ,
) -> Tensor:
    """Nearest-joint vertex ownership for the SMPL rest pose ``(6890,)`` long, values in ``[0, 22)``.

    Ports ``BetterVideoReconstruction/tools/human_optim/contacts.py::vertex_to_joint``:
    each vertex is assigned to the spatially closest joint center of the *shaped
    rest pose* (NOT the LBS-argmax joint that deforms it). Shaped rest pose is
    ``v = v_template + shapedirs @ betas`` and ``J = J_regressor @ v``. SMPL's 24
    joints are folded to 22 by mapping the hand joints (22/23) onto the wrists
    (20/21).

    :param betas: Optional shape coefficients; ``None`` uses the neutral shape.
    :param smpl_npz: Path to the SMPL ``.npz`` (needs ``v_template``,
        ``shapedirs``, ``J_regressor``).
    """
    if betas is None:
        cached = _OWNER_CACHE.get(smpl_npz)
        if cached is not None:
            return cached

    data = np.load(smpl_npz, allow_pickle=True)
    v_template = torch.as_tensor(np.asarray(data["v_template"]), dtype=torch.float32)   # [6890, 3]
    shapedirs = torch.as_tensor(np.asarray(data["shapedirs"]), dtype=torch.float32)     # [6890, 3, n]
    j_regressor = torch.as_tensor(np.asarray(data["J_regressor"]), dtype=torch.float32)  # [24, 6890]

    verts = v_template
    if betas is not None:
        beta = torch.as_tensor(np.asarray(betas), dtype=torch.float32).reshape(-1)
        n = min(beta.shape[0], shapedirs.shape[2])
        verts = verts + torch.einsum("vcn,n->vc", shapedirs[:, :, :n], beta[:n])

    joints = j_regressor @ verts                                    # [24, 3]
    owner = torch.cdist(verts[None], joints[None])[0].argmin(dim=1)  # [6890] in [0, 24)
    for hand, wrist in _SMPL_HAND_TO_WRIST.items():
        owner[owner == hand] = wrist

    owner = owner.long()
    if betas is None:
        _OWNER_CACHE[smpl_npz] = owner
    return owner


def derive_joint_contact(vertex_contact: Tensor, owner: Tensor) -> Tensor:
    """Lift per-vertex contact to per-joint by max over each joint's vertices.

    :param vertex_contact: ``(..., 6890)`` float/bool per-vertex contact.
    :param owner: ``(6890,)`` long vertex->joint owner in ``[0, 22)``.
    :returns: ``(..., 22)`` float per-joint contact (1 if any owned vertex touches).
    """
    lead = vertex_contact.shape[:-1]
    flat = vertex_contact.reshape(-1, vertex_contact.shape[-1]).to(torch.float32)  # [N, V]
    idx = owner[None].expand(flat.shape[0], -1).to(flat.device)                    # [N, V]
    out = torch.zeros(flat.shape[0], NUM_BODY_22, dtype=torch.float32, device=flat.device)
    out.scatter_reduce_(1, idx, flat, reduce="amax", include_self=True)
    return out.reshape(*lead, NUM_BODY_22)


def _joint_subset_mask(supervise_subset) -> Tensor:
    """Return a ``(22,)`` float mask of supervised joints (1=supervised)."""
    mask = torch.zeros(NUM_BODY_22, dtype=torch.float32)
    if supervise_subset is None:
        mask[:] = 1.0
    elif supervise_subset == "observable_14":
        mask[OBSERVABLE_14] = 1.0
    elif isinstance(supervise_subset, (list, tuple)):
        for i in supervise_subset:
            mask[int(i)] = 1.0
    else:
        raise ValueError(
            f"joint.supervise_subset must be null, 'observable_14', or an index list; "
            f"got {supervise_subset!r}")
    return mask


@dataclass
class TargetSpec:
    """Resolved supervision-target spec, built once per run from the config.

    :param enabled: Enabled target names in canonical order (``vertex`` before ``joint``).
    :param topology: Vertex topology name (``"smpl"`` / ``"smplx"``).
    :param vertex_dims: Vertex head output size (topology vertex count).
    :param joint_dims: Joint head output size (always 22).
    :param derive_from_vertex: Lift vertex labels to joint labels for image data.
    :param joint_subset_mask: ``(22,)`` mask restricting supervised joints.
    :param owner: ``(6890,)`` vertex->joint owner, present iff ``derive_from_vertex``.
    """

    enabled: list[str]
    topology: str
    vertex_dims: int
    joint_dims: int = NUM_BODY_22
    derive_from_vertex: bool = False
    joint_subset_mask: Tensor = field(default_factory=lambda: torch.ones(NUM_BODY_22))
    owner: Optional[Tensor] = None
    #: Bounded cache of per-``betas`` ownership maps (key = rounded-betas bytes).
    _betas_owner_cache: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_config(cls, cfg: dict) -> "TargetSpec":
        topology = cfg["contact"]["topology"]
        targets = cfg["contact"]["targets"]
        enabled = [name for name in ("vertex", "joint") if targets[name]["enabled"]]
        jcfg = targets["joint"]
        derive = bool(jcfg["derive_from_vertex"])
        owner = compute_vertex_joint_owner() if (derive and "joint" in enabled) else None
        return cls(
            enabled=enabled,
            topology=topology,
            vertex_dims=topology_num_vertices(topology),
            derive_from_vertex=derive,
            joint_subset_mask=_joint_subset_mask(jcfg["supervise_subset"]),
            owner=owner,
        )

    def output_dims(self) -> dict[str, int]:
        """``{target_name: head_output_size}`` for the enabled targets."""
        dims = {"vertex": self.vertex_dims, "joint": self.joint_dims}
        return {name: dims[name] for name in self.enabled}

    def _owner_for_frame(self, frame: dict) -> Tensor:
        """Vertex->joint ownership for a frame: shaped by its ``betas`` if present.

        ClimbingImages supplies per-sample SMPL ``betas``; a contacted boundary
        vertex whose nearest joint moves with body shape would otherwise be lifted
        to the wrong joint. Falls back to the neutral :attr:`owner` when a frame
        carries no betas (e.g. DAMON). Maps are cached by rounded betas to bound
        recompute; the cache is capped so a large, varied corpus cannot grow it
        without limit.
        """
        smpl = frame.get("smpl")
        betas = smpl.get("betas") if isinstance(smpl, dict) else None
        if betas is None:
            return self.owner
        arr = np.asarray(betas, dtype=np.float32).reshape(-1)
        key = np.round(arr, 3).tobytes()
        cached = self._betas_owner_cache.get(key)
        if cached is None:
            cached = compute_vertex_joint_owner(betas=arr)
            if len(self._betas_owner_cache) < 4096:
                self._betas_owner_cache[key] = cached
        return cached

    def assemble_batch(self, frames: list[dict]) -> dict[str, dict[str, Tensor]]:
        """Build ``{target: {'gt': [B, D], 'mask': [B, D]}}`` from per-frame dicts.

        A frame carries native labels: image frames a ``contact`` ``[V]`` vertex
        tensor; video frames ``joint_contact`` / ``joint_mask`` ``[22]``. Targets a
        frame does not supervise get all-zero masks (ignored by the loss).
        """
        batch_size = len(frames)
        out: dict[str, dict[str, Tensor]] = {}

        if "vertex" in self.enabled:
            gt = torch.zeros(batch_size, self.vertex_dims, dtype=torch.float32)
            mask = torch.zeros(batch_size, self.vertex_dims, dtype=torch.float32)
            for i, frame in enumerate(frames):
                contact = frame.get("contact")
                if contact is not None:
                    gt[i] = torch.as_tensor(contact, dtype=torch.float32)
                    mask[i] = 1.0
            out["vertex"] = {"gt": gt, "mask": mask}

        if "joint" in self.enabled:
            gt = torch.zeros(batch_size, self.joint_dims, dtype=torch.float32)
            mask = torch.zeros(batch_size, self.joint_dims, dtype=torch.float32)
            subset = self.joint_subset_mask
            for i, frame in enumerate(frames):
                if "joint_contact" in frame:
                    gt[i] = torch.as_tensor(frame["joint_contact"], dtype=torch.float32)
                    mask[i] = torch.as_tensor(frame["joint_mask"], dtype=torch.float32) * subset
                elif self.derive_from_vertex and frame.get("contact") is not None:
                    contact = torch.as_tensor(frame["contact"], dtype=torch.float32)
                    gt[i] = derive_joint_contact(contact, self._owner_for_frame(frame))
                    mask[i] = subset
            out["joint"] = {"gt": gt, "mask": mask}

        return out


def _dataset_supervised(native: set[str], enabled: set[str], derive: bool) -> set[str]:
    """Enabled targets a dataset supervises, natively or via enabled derivation."""
    supervised = set(native) & enabled
    if derive and "vertex" in native and "joint" in enabled:
        supervised.add("joint")
    return supervised


def validate_targets(cfg: dict, datasets: Sequence) -> None:
    """Reject configs whose enabled targets cannot be supervised by the datasets.

    Every enabled target must be supervised by at least one dataset, **and every
    configured dataset must supervise at least one enabled target** (natively or
    via enabled ``joint.derive_from_vertex``) — otherwise that dataset's batches
    are all-masked yet still take optimiser steps. Every dataset that supplies
    vertex labels must also match ``contact.topology``.

    :param cfg: Resolved run config.
    :param datasets: Built dataset objects exposing ``supervised_targets``
        (set of names) and ``topology`` (str or ``None``).
    :raises ValueError: on any unsatisfiable (dataset, target, topology) combo, or
        a dataset that supervises none of the enabled targets.
    """
    topology = cfg["contact"]["topology"]
    targets = cfg["contact"]["targets"]
    enabled = {name for name in ("vertex", "joint") if targets[name]["enabled"]}
    derive = bool(targets["joint"]["derive_from_vertex"])

    supplied: set[str] = set()
    for ds in datasets:
        native = set(getattr(ds, "supervised_targets", set()))
        ds_supervised = _dataset_supervised(native, enabled, derive)
        supplied |= ds_supervised
        if "vertex" in native:
            ds_topology = getattr(ds, "topology", None)
            if ds_topology != topology:
                raise ValueError(
                    f"dataset {getattr(ds, 'name', ds)!r} supplies vertex labels in "
                    f"topology {ds_topology!r} but contact.topology is {topology!r}")
        if not ds_supervised:
            raise ValueError(
                f"dataset {getattr(ds, 'name', ds)!r} supervises none of the enabled "
                f"target(s) {sorted(enabled)} (it supplies {sorted(native)}); every "
                f"configured dataset must supervise ≥1 enabled target or it only "
                f"contributes all-masked batches")

    missing = enabled - supplied
    if missing:
        raise ValueError(
            f"enabled target(s) {sorted(missing)} are not supervised by any dataset "
            f"(set joint.derive_from_vertex, or add a dataset that supplies them)")
