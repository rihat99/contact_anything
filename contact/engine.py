"""Single forward pass through SAM-3D-Body for contact training / eval / demo.

The trainer, evaluator and demo all run the model the same way: initialise the
batch, forward the *body* decoder, and read the output. Keeping that here is the
one place the forward lives — it removes the last script->script import
(``evaluate``/``demo`` previously reached into ``scripts.train``).
"""
from __future__ import annotations

import torch.nn as nn


def forward_model(model: nn.Module, batch: dict) -> dict:
    """Forward one collated batch through the body decoder; return the raw output.

    :param model: a built SAM-3D-Body model (see :func:`contact.model.build_model`).
    :param batch: collated batch already moved to the model's device.
    :returns: the full model output dict (``"mhr"``, ``"contact"``, ...).
    """
    model._initialize_batch(batch)
    return model.forward_step(batch, decoder_type="body")


def forward_contact(model: nn.Module, batch: dict) -> dict:
    """Forward one batch and return just the contact output dict.

    :param model: a built SAM-3D-Body model.
    :param batch: collated batch already moved to the model's device.
    :returns: ``{"<target>_logits": [B, D], "<target>_probs": [B, D], ...}``.
    :raises RuntimeError: if the model produced no contact output.
    """
    out = forward_model(model, batch)
    if out.get("contact") is None:
        raise RuntimeError("model produced no contact output — check DO_CONTACT_TOKENS.")
    return out["contact"]
