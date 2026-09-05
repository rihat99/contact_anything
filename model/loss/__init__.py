"""Supervision terms: one interface, one reduction rule, one metric protocol.

Every loss in this package is a weighted mean over rows the batch happens to
supervise, and the batches are not homogeneous — a clip may carry no valid
extrinsics, a rank may draw a batch with no in-contact limb at all. So no loss
ever returns a normalised scalar. It returns, per term, the ADDITIVE pair
``(numerator, mass)`` of :class:`LossTerm`; the trainer all-reduces the masses
once, divides once, and gets the exact global weighted mean — including when a
rank's mass is below one or zero (:func:`utils.distributed.ddp_global_mean_term`).

Three rules follow from that and every loss respects them:

* **The term set is fixed by config, not by the batch.** A term with no rows
  this batch reports ``mass = 0``, never a missing key, so every rank iterates
  the same names in the same order and the all-reduce lines up.
* **Every numerator is graph-connected**, even at zero mass: each carries a
  ``0 * (sum of the tensors the loss consumes)`` anchor, so no parameter drops
  off the backward graph under ``find_unused_parameters=False``.
* **Term weights (the yaml ``loss.*`` block) are applied INSIDE the loss**, to
  the numerator. The trainer only reduces.

Evaluation uses a second channel: :attr:`LossResult.stats`, a float64 vector of
additive sufficient statistics summed over the split and all-reduced once, from
which :meth:`Loss.metrics` computes the reported numbers. Means of per-batch
means are wrong under uneven shards; this is why.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
from torch import Tensor

#: The six contact / force groups, in kindyn's ``contact_force_joints`` column
#: order. ``*_foot`` is the big-toe joint, ``*_ankle`` the heel. This order is
#: the contract between the loaders, the six contact/force tokens and every
#: loss and metric here.
KINDYN_GROUP_NAMES = (
    "left_hand", "right_hand", "left_foot", "right_foot",
    "left_ankle", "right_ankle",
)
#: MHR70 keypoint anchoring each group, same order — wrists, big-toe tips, heels.
KINDYN_GROUP_KEYPOINTS = (62, 41, 15, 18, 17, 20)
NUM_KINDYN_GROUPS = len(KINDYN_GROUP_NAMES)


class LossTerm(NamedTuple):
    """One additive term of a loss.

    :param numerator: graph-live WEIGHTED numerator summed over this rank's
        rows (a scalar tensor).
    :param mass: this rank's weight mass, ``>= 0``. Mass 0 means the term is
        inert this batch — its numerator is then a graph-connected zero.
    """

    numerator: Tensor
    mass: float


@dataclass
class LossResult:
    """What one loss returns for one batch.

    :param terms: the loss's fixed term set (same keys every batch).
    :param scalars: detached per-batch diagnostics for the train log.
    :param stats: float64 ``[len(stat_names)]`` additive sufficient statistics.
    """

    terms: dict[str, LossTerm]
    scalars: dict[str, float] = field(default_factory=dict)
    stats: Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.float64))


class Loss(ABC):
    """Base class of every supervision term.

    :param cfg: the FULL resolved run config — a loss reads its own yaml
        section plus whatever model facts it needs.
    :param model: the built :class:`~model.network.ContactAnything` (branch
        availability, anchor indices).
    :param device: device the loss runs on; predictions are moved to it.

    Subclasses set :attr:`name` (the yaml section stem), :attr:`stat_names` and
    :attr:`term_names`, and implement :meth:`__call__` and :meth:`metrics`.
    """

    #: Loss identity: the yaml section stem, and the loss-term log namespace.
    name: str = ""
    #: Tensorboard section of the reported metrics (``metric_<group>/<metric>``);
    #: defaults to :attr:`name`. Set in ``__init__`` when it differs.
    metric_group: str = ""
    #: Names of the entries of :attr:`LossResult.stats`.
    stat_names: tuple[str, ...] = ()
    #: The fixed term set this loss emits every batch (set in ``__init__`` from
    #: the config). The trainer lays out its all-reduce buffers from it, so a
    #: rank that sees no batch still reduces the same shape as every other.
    term_names: tuple[str, ...] = ()

    def __init__(self, cfg: dict, model, device: torch.device | str) -> None:
        self.cfg = cfg
        self.model = getattr(model, "module", model)
        self.device = torch.device(device)
        self.dtype = torch.float32
        if not self.metric_group:
            self.metric_group = self.name

    @abstractmethod
    def __call__(self, out: dict, batch: dict, *, train: bool) -> LossResult:
        """Evaluate the loss on one forward output.

        :param out: :meth:`~model.network.ContactAnything.forward` output.
        :param batch: the collated batch on the same device.
        :param train: ``False`` disables train-only row filtering, so the
            reported evaluation is protocol-stable.
        """

    @abstractmethod
    def metrics(self, stats: Tensor) -> dict[str, float]:
        """Reported metrics from the SUMMED (all-reduced) statistics vector."""

    # ------------------------------------------------------------------ helpers

    def empty_stats(self) -> Tensor:
        """A zero statistics vector of this loss's shape."""
        return torch.zeros(len(self.stat_names), dtype=torch.float64,
                           device=self.device)

    @staticmethod
    def _terms(raw: dict[str, tuple[Tensor, float]], anchor: Tensor
               ) -> dict[str, LossTerm]:
        """Attach the graph-connected zero to every numerator."""
        return {name: LossTerm(numerator + anchor, float(mass))
                for name, (numerator, mass) in raw.items()}


def build_losses(cfg: dict, model, device: torch.device | str) -> list[Loss]:
    """Instantiate every enabled loss, in a fixed order (contact, force, smplx, motion).

    The order is what makes the trainer's packed mass all-reduce identical on
    every rank.

    :raises ValueError: when an enabled loss has no branch to supervise.
    """
    net = getattr(model, "module", model)
    losses: list[Loss] = []
    if cfg["contact_supervision"]["enabled"]:
        _require(net.has_contact, "contact_supervision",
                 "model.contact.enabled or a refiner 'contact' output")
        from model.loss.contact import ContactLoss
        losses.append(ContactLoss(cfg, model, device))
    if cfg["force_supervision"]["enabled"]:
        _require(net.has_force, "force_supervision",
                 "model.force.enabled or a refiner 'force' output")
        from model.loss.force import ForceLoss
        losses.append(ForceLoss(cfg, model, device))
    if cfg["smplx_supervision"]["enabled"]:
        _require(net.head_smplx is not None, "smplx_supervision", "model.smplx.enabled")
        from model.loss.smplx import SmplxLoss
        losses.append(SmplxLoss(cfg, model, device))
    if cfg["motion_supervision"]["enabled"]:
        _require(net.has_motion, "motion_supervision", "a refiner 'motion' output")
        from model.loss.motion import MotionLoss
        losses.append(MotionLoss(cfg, model, device))
    return losses


def _require(condition: bool, section: str, requirement: str) -> None:
    if not condition:
        raise ValueError(f"{section} is enabled but requires {requirement}")


__all__ = ["Loss", "LossResult", "LossTerm", "build_losses",
           "KINDYN_GROUP_NAMES", "KINDYN_GROUP_KEYPOINTS", "NUM_KINDYN_GROUPS"]
