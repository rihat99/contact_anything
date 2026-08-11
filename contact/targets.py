"""Contact targets: topologies, joint sets, and vertex->joint ownership.

The model may expose independent heads over the configured contact-token bank:

* ``vertex`` — per-vertex contact in the body-model topology (SMPL 6890 or
  SMPL-X 10475).
* ``joint`` — contact over the 22-joint SMPL-X body set or a reduced set
  (four extremities, or the six kindyn force groups) selected by
  ``contact.targets.joint.joint_set``.

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

# Four contact endpoints used by the climbing-only classifier. Hands are already
# wrist+finger aggregates in ClimbingVideos_v1; each foot combines the distinct
# ankle and big-toe/foot labels with a tri-state OR (see
# :func:`reduce_body22_to_groups`).
EXTREMITY_4_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot")
EXTREMITY_4_GROUPS = ((20,), (21,), (7, 10), (8, 11))
NUM_EXTREMITY_4 = len(EXTREMITY_4_NAMES)

# Six contact outputs matched 1:1 to the corpus kindyn force groups, in kindyn
# column order: LH, RH, LF=big-toe, RF=big-toe, LA=heel, RA=heel. Every group
# has a single body-22 source joint — the hands are the wrists (fingers already
# folded there by the 52->22 fold, exactly how ``extremities_4`` sources its
# hands), the toe groups the foot joints (10/11), the heel groups the ankles (7/8).
KINDYN_6_NAMES = (
    "left_hand", "right_hand", "left_foot", "right_foot", "left_ankle", "right_ankle",
)
KINDYN_6_GROUPS = ((20,), (21,), (10,), (11,), (7,), (8,))
NUM_KINDYN_6 = len(KINDYN_6_NAMES)

JOINT_SET_NAMES = {
    "smplx_body_22": tuple(SMPLX_BODY_22),
    "extremities_4": EXTREMITY_4_NAMES,
    "kindyn_6": KINDYN_6_NAMES,
}
# Body-22 source groups of the reduced joint sets (assemble_batch reduction).
JOINT_SET_GROUPS = {
    "extremities_4": EXTREMITY_4_GROUPS,
    "kindyn_6": KINDYN_6_GROUPS,
}

# Joints an annotator can label in the manual test set (evaluation only). The
# other 8 (pelvis, spine1/2/3, neck, collars, head) stay non-contact.
OBSERVABLE_14 = [1, 2, 4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21]

# The manual test protocol does not expose these joints. On a reviewed frame the
# dataset schema defines them as non-contact rather than unknown.
ALWAYS_NON_CONTACT_8 = [0, 3, 6, 9, 12, 13, 14, 15]

# SMPL joints 22/23 are the hands; fold them onto the wrists (20/21) so a
# 24-joint SMPL ownership map lands in the 22-joint body set.
_SMPL_HAND_TO_WRIST = {22: 20, 23: 21}

SMPL_NEUTRAL_NPZ = "/data3/rikhat.akizhanov/better/BetterHuman/models/smpl/SMPL_NEUTRAL.npz"

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


def joint_set_names(joint_set: str) -> tuple[str, ...]:
    """Return ordered output names for a supported joint-contact set."""
    try:
        return JOINT_SET_NAMES[joint_set]
    except KeyError as exc:
        raise ValueError(
            f"unknown joint_set {joint_set!r}; choose from {sorted(JOINT_SET_NAMES)}") from exc


def joint_set_num_outputs(joint_set: str) -> int:
    """Return the contact-head output dimension for ``joint_set``."""
    return len(joint_set_names(joint_set))


def reduce_body22_to_extremities(
    contact: Tensor,
    supervised: Tensor,
    confidence: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reduce body-22 labels to the four extremities (:data:`EXTREMITY_4_GROUPS`)."""
    return reduce_body22_to_groups(contact, supervised, confidence, EXTREMITY_4_GROUPS)


def reduce_body22_to_groups(
    contact: Tensor,
    supervised: Tensor,
    confidence: Tensor,
    groups: Sequence[Sequence[int]],
) -> tuple[Tensor, Tensor, Tensor]:
    """Reduce body-22 labels onto joint groups with tri-state OR semantics.

    ``contact``, ``supervised`` and ``confidence`` must have the same shape
    ``(..., 22)``. A group is a known positive when any *supervised* member is
    positive, even if another member is unknown. It is a known negative only
    when every member is supervised and free; a partial negative stays ignored.
    Positive confidence is the maximum over supervised positive members, while
    known-free confidence is the mean over all group members. This matches the
    ClimbingVideos exporter convention used to fold fingers into each hand.
    A single-member group (every hand, and every ``kindyn_6`` group) degenerates
    to a passthrough of its joint's label, supervision and confidence.

    :param groups: Body-22 index groups, one output per group
        (e.g. :data:`EXTREMITY_4_GROUPS` / :data:`KINDYN_6_GROUPS`).
    :returns: ``(contact_G, supervised_G, confidence_G)`` as float tensors.
    """
    contact = torch.as_tensor(contact, dtype=torch.float32)
    supervised = torch.as_tensor(supervised, dtype=torch.float32, device=contact.device)
    confidence = torch.as_tensor(confidence, dtype=torch.float32, device=contact.device)
    if contact.shape != supervised.shape or contact.shape != confidence.shape:
        raise ValueError(
            "body-22 contact/supervised/confidence shapes must match; got "
            f"{tuple(contact.shape)}, {tuple(supervised.shape)}, {tuple(confidence.shape)}")
    if contact.ndim < 1 or contact.shape[-1] != NUM_BODY_22:
        raise ValueError(f"body-22 reduction expects (..., 22); got {tuple(contact.shape)}")

    is_contact = contact > 0.5
    is_supervised = supervised > 0
    confidence = torch.nan_to_num(confidence, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
    out_contact = []
    out_supervised = []
    out_confidence = []
    for group in groups:
        indices = list(group)
        group_contact = is_contact[..., indices]
        group_supervised = is_supervised[..., indices]
        group_confidence = confidence[..., indices]
        supervised_positive = group_contact & group_supervised
        positive = supervised_positive.any(dim=-1)
        all_known = group_supervised.all(dim=-1)
        known = positive | all_known

        positive_confidence = torch.where(
            supervised_positive,
            group_confidence,
            torch.full_like(group_confidence, -torch.inf),
        ).amax(dim=-1)
        free_confidence = group_confidence.mean(dim=-1)
        reduced_confidence = torch.where(
            positive,
            positive_confidence,
            torch.where(all_known, free_confidence, torch.zeros_like(free_confidence)),
        )
        out_contact.append(positive.to(torch.float32))
        out_supervised.append(known.to(torch.float32))
        out_confidence.append(reduced_confidence)

    return (
        torch.stack(out_contact, dim=-1),
        torch.stack(out_supervised, dim=-1),
        torch.stack(out_confidence, dim=-1),
    )


def _joint_subset_mask(supervise_subset, joint_set: str) -> Tensor:
    """Return the output-space float supervision-subset mask."""
    if joint_set in JOINT_SET_GROUPS:
        if supervise_subset is not None:
            raise ValueError(
                f"joint.supervise_subset must be null for joint_set={joint_set!r}")
        return torch.ones(len(JOINT_SET_NAMES[joint_set]), dtype=torch.float32)

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
    :param joint_set: Semantic joint-contact output set.
    :param joint_names: Ordered semantic output names.
    :param joint_dims: Joint head output size (22, 4 or 6).
    :param derive_from_vertex: Lift vertex labels to joint labels for image data.
    :param joint_subset_mask: Output-space mask restricting supervised joints.
    :param owner: ``(6890,)`` vertex->joint owner, present iff ``derive_from_vertex``.
    """

    enabled: list[str]
    topology: str
    vertex_dims: int
    joint_set: str = "smplx_body_22"
    joint_names: tuple[str, ...] = tuple(SMPLX_BODY_22)
    joint_dims: int = NUM_BODY_22
    derive_from_vertex: bool = False
    use_confidence_weights: bool = False
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
        joint_set = str(jcfg.get("joint_set", "smplx_body_22"))
        names = joint_set_names(joint_set)
        derive = bool(jcfg["derive_from_vertex"])
        owner = compute_vertex_joint_owner() if (derive and "joint" in enabled) else None
        return cls(
            enabled=enabled,
            topology=topology,
            vertex_dims=topology_num_vertices(topology),
            joint_set=joint_set,
            joint_names=names,
            joint_dims=len(names),
            derive_from_vertex=derive,
            use_confidence_weights=bool(jcfg.get("use_confidence_weights", False)),
            joint_subset_mask=_joint_subset_mask(jcfg["supervise_subset"], joint_set),
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
        tensor; video frames carry raw body-22 contact, supervision and confidence.
        Targets a frame does not supervise get all-zero masks (ignored by the loss).
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
                    if self.joint_set == "smplx_body_22":
                        gt[i] = torch.as_tensor(frame["joint_contact"], dtype=torch.float32)
                        mask[i] = torch.as_tensor(frame["joint_mask"], dtype=torch.float32) * subset
                    else:
                        missing = {
                            name for name in ("joint_supervised", "joint_confidence")
                            if name not in frame
                        }
                        if missing:
                            raise ValueError(
                                f"joint_set={self.joint_set!r} requires raw body-22 "
                                f"{sorted(missing)} in each video frame")
                        reduced_gt, reduced_supervised, reduced_confidence = (
                            reduce_body22_to_groups(
                                frame["joint_contact"],
                                frame["joint_supervised"],
                                frame["joint_confidence"],
                                JOINT_SET_GROUPS[self.joint_set],
                            )
                        )
                        gt[i] = reduced_gt
                        mask[i] = reduced_supervised * (
                            reduced_confidence if self.use_confidence_weights else 1.0)
                elif self.derive_from_vertex and frame.get("contact") is not None:
                    contact = torch.as_tensor(frame["contact"], dtype=torch.float32)
                    body22 = derive_joint_contact(contact, self._owner_for_frame(frame))
                    if self.joint_set == "smplx_body_22":
                        gt[i] = body22
                        mask[i] = subset
                    else:
                        gt[i], mask[i], _ = reduce_body22_to_groups(
                            body22, torch.ones_like(body22), torch.ones_like(body22),
                            JOINT_SET_GROUPS[self.joint_set])
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
    # A supervised-force run (force_supervision.enabled) counts a dataset that
    # supplies GT forces as supervising — force-only configs have no contact
    # target at all, and mixed configs may include a forces-only dataset.
    force_supervised = bool((cfg.get("force_supervision") or {}).get("enabled", False))

    supplied: set[str] = set()
    for ds in datasets:
        native = set(getattr(ds, "supervised_targets", set()))
        ds_supervised = _dataset_supervised(native, enabled, derive)
        ds_forces = force_supervised and bool(getattr(ds, "load_forces", False))
        supplied |= ds_supervised
        if "vertex" in native:
            ds_topology = getattr(ds, "topology", None)
            if ds_topology != topology:
                raise ValueError(
                    f"dataset {getattr(ds, 'name', ds)!r} supplies vertex labels in "
                    f"topology {ds_topology!r} but contact.topology is {topology!r}")
        if not ds_supervised and not ds_forces:
            raise ValueError(
                f"dataset {getattr(ds, 'name', ds)!r} supervises none of the enabled "
                f"target(s) {sorted(enabled)} (it supplies {sorted(native)}); every "
                f"configured dataset must supervise ≥1 enabled target (or supply GT "
                f"forces under force_supervision) or it only contributes all-masked "
                f"batches")

    missing = enabled - supplied
    if missing:
        raise ValueError(
            f"enabled target(s) {sorted(missing)} are not supervised by any dataset "
            f"(set joint.derive_from_vertex, or add a dataset that supplies them)")
