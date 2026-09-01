"""Contact targets: the joint sets the contact head predicts over.

``joint`` is the only target: contact over the 22-joint SMPL-X body set or a
reduced set (four extremities, or the six kindyn force groups) selected by
``contact.targets.joint.joint_set``. Labels come from the ClimbingVideos
corpus, where they are motion-gated *stable* contact (stillness + hysteresis +
min-duration + gap-merge, see
``BetterVideoReconstruction/scripts/stages/estimate_contacts.py``) — not
instantaneous surface contact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor

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

    :param enabled: Enabled target names (``["joint"]`` or empty).
    :param joint_set: Semantic joint-contact output set.
    :param joint_names: Ordered semantic output names.
    :param joint_dims: Joint head output size (22, 4 or 6).
    :param joint_subset_mask: Output-space mask restricting supervised joints.
    """

    enabled: list[str]
    joint_set: str = "smplx_body_22"
    joint_names: tuple[str, ...] = tuple(SMPLX_BODY_22)
    joint_dims: int = NUM_BODY_22
    use_confidence_weights: bool = False
    joint_subset_mask: Tensor = field(default_factory=lambda: torch.ones(NUM_BODY_22))

    @classmethod
    def from_config(cls, cfg: dict) -> "TargetSpec":
        targets = cfg["contact"]["targets"]
        enabled = ["joint"] if targets["joint"]["enabled"] else []
        jcfg = targets["joint"]
        joint_set = str(jcfg.get("joint_set", "smplx_body_22"))
        names = joint_set_names(joint_set)
        return cls(
            enabled=enabled,
            joint_set=joint_set,
            joint_names=names,
            joint_dims=len(names),
            use_confidence_weights=bool(jcfg.get("use_confidence_weights", False)),
            joint_subset_mask=_joint_subset_mask(jcfg["supervise_subset"], joint_set),
        )

    def output_dims(self) -> dict[str, int]:
        """``{target_name: head_output_size}`` for the enabled targets."""
        return {name: self.joint_dims for name in self.enabled}

    def assemble_batch(self, frames: list[dict]) -> dict[str, dict[str, Tensor]]:
        """Build ``{target: {'gt': [B, D], 'mask': [B, D]}}`` from per-frame dicts.

        Video frames carry raw body-22 contact, supervision and confidence.
        A frame that does not supervise the target gets an all-zero mask
        (ignored by the loss).
        """
        batch_size = len(frames)
        out: dict[str, dict[str, Tensor]] = {}

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
            out["joint"] = {"gt": gt, "mask": mask}

        return out


def validate_targets(cfg: dict, datasets: Sequence) -> None:
    """Reject configs whose enabled targets cannot be supervised by the datasets.

    The joint target, when enabled, must be supervised by at least one dataset,
    **and every configured dataset must supervise it** — otherwise that dataset's
    batches are all-masked yet still take optimiser steps. A dataset supplying GT
    forces (``force_supervision``) or GT motion (``motion_supervision``) counts as
    supervising: those builds have no contact target at all.

    :param cfg: Resolved run config.
    :param datasets: Built dataset objects exposing ``supervised_targets``
        (set of names).
    :raises ValueError: on any unsatisfiable (dataset, target) combo, or a dataset
        that supervises none of the enabled targets.
    """
    targets = cfg["contact"]["targets"]
    enabled = {"joint"} if targets["joint"]["enabled"] else set()
    # A supervised-force run (force_supervision.enabled) counts a dataset that
    # supplies GT forces as supervising — force-only configs have no contact
    # target at all, and mixed configs may include a forces-only dataset.
    force_supervised = bool((cfg.get("force_supervision") or {}).get("enabled", False))
    # Same exemption for a supervised-motion run (motion-only configs have no
    # contact target either; their supervision is the kindyn vel/acc targets).
    motion_supervised = bool((cfg.get("motion_supervision") or {}).get("enabled", False))
    # And for a pose-temporal run (E2): supervision is the kindyn-MHR pose
    # pseudo-GT the corpus loader supplies under load_pose.
    pose_supervised = bool((cfg.get("pose_supervision") or {}).get("enabled", False))
    # And for a keypoint-supervised run (stage-1 pose/camera fine-tune): GT is
    # the kindyn joints_world the corpus loader supplies under load_keypoints.
    kp_supervised = bool(
        (cfg.get("keypoint_supervision") or {}).get("enabled", False))

    supplied: set[str] = set()
    for ds in datasets:
        native = set(getattr(ds, "supervised_targets", set()))
        ds_supervised = native & enabled
        ds_forces = force_supervised and bool(getattr(ds, "load_forces", False))
        ds_motion = motion_supervised and bool(getattr(ds, "load_motion", False))
        ds_pose = pose_supervised and bool(getattr(ds, "load_pose", False))
        ds_kp = kp_supervised and bool(getattr(ds, "load_keypoints", False))
        supplied |= ds_supervised
        if (not ds_supervised and not ds_forces and not ds_motion
                and not ds_pose and not ds_kp):
            raise ValueError(
                f"dataset {getattr(ds, 'name', ds)!r} supervises none of the enabled "
                f"target(s) {sorted(enabled)} (it supplies {sorted(native)}); every "
                f"configured dataset must supervise ≥1 enabled target (or supply GT "
                f"forces under force_supervision / GT motion under motion_supervision "
                f"/ pose pseudo-GT under pose_supervision / GT keypoints under "
                f"keypoint_supervision) "
                f"or it only contributes all-masked batches")

    missing = enabled - supplied
    if missing:
        raise ValueError(
            f"enabled target(s) {sorted(missing)} are not supervised by any dataset")
