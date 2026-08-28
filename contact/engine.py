"""Single forward pass through SAM-3D-Body for contact training / eval / demo.

The trainer, evaluator and demo all run the model the same way: initialise the
batch, forward the *body* decoder, and read the output. Keeping that here is the
one place the forward lives — it removes the last script->script import
(``evaluate``/``demo`` previously reached into ``scripts.train``).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def forward_model(model: nn.Module, batch: dict) -> dict:
    """Forward one collated batch through the body decoder; return the raw output.

    Batches carrying precomputed backbone embeddings (``batch["embedding"]``,
    bf16 ``[B, C, h, w]`` from the ``data.embedding_cache`` load path) skip the
    frozen backbone; mask/ray conditioning still run live inside the model.

    :param model: a built SAM-3D-Body model (see :func:`contact.model.build_model`).
    :param batch: collated batch already moved to the model's device.
    :returns: the full model output dict (``"mhr"``, ``"contact"``, ...).
    """
    model._initialize_batch(batch)
    return model.forward_step(
        batch, decoder_type="body", precomputed_features=batch.get("embedding"))


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


def select_temporal_supervision(
    logits: dict[str, torch.Tensor],
    targets: dict[str, dict[str, torch.Tensor]],
    seq_len: int,
    target_frame: str,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    """Select which temporal rows contribute to loss and metrics.

    Inputs use the collator's clip-major, frame-minor layout: the first ``T``
    rows are clip 0, the next ``T`` rows clip 1, and so on. ``"center"`` keeps
    exactly row ``T // 2`` from every clip. The temporal model still consumes
    all ``T`` frames, so the selected prediction can attend to the full window.

    ``"all"`` returns the original mappings unchanged. Center selection requires
    an odd sequence length so there is one unambiguous middle frame.
    """
    if target_frame == "all":
        return logits, targets
    if target_frame != "center":
        raise ValueError(
            f"target_frame must be 'all' or 'center'; got {target_frame!r}")

    seq_len = int(seq_len)
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive; got {seq_len}")
    if seq_len % 2 == 0:
        raise ValueError(
            f"center-frame supervision requires an odd seq_len; got {seq_len}")

    tensors = list(logits.values()) + [
        value for target in targets.values() for value in target.values()
        if torch.is_tensor(value)
    ]
    if not tensors:
        raise ValueError("cannot select temporal supervision from empty mappings")
    num_rows = int(tensors[0].shape[0])
    if num_rows % seq_len:
        raise ValueError(
            f"flattened row count {num_rows} is not divisible by seq_len {seq_len}")
    mismatched = [tuple(value.shape) for value in tensors if value.shape[0] != num_rows]
    if mismatched:
        raise ValueError(
            f"logit/target row counts disagree with {num_rows}: {mismatched}")

    num_clips = num_rows // seq_len
    center = seq_len // 2

    def _center_rows(value: torch.Tensor) -> torch.Tensor:
        shape = (num_clips, seq_len, *value.shape[1:])
        return value.reshape(shape)[:, center]

    selected_logits = {name: _center_rows(value) for name, value in logits.items()}
    selected_targets = {
        name: {
            key: _center_rows(value) if torch.is_tensor(value) else value
            for key, value in target.items()
        }
        for name, target in targets.items()
    }
    return selected_logits, selected_targets
