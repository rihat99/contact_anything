"""Locate the BetterHuman MHR archive used by the pose loss and the physics adapter."""
from __future__ import annotations

import os
from pathlib import Path

MODELS_ENV = "BETTERHUMAN_MODELS_DIR"


def resolve_mhr_archive(model_path: str | None, lod: int) -> str | None:
    """Resolve the MHR archive: explicit path, then ``$BETTERHUMAN_MODELS_DIR``,
    then the sibling BetterHuman checkout.

    Returns ``None`` when the environment root is set, letting
    ``better_human.MHR`` resolve the licensed file itself.

    :raises FileNotFoundError: when neither is available.
    """
    if model_path is not None:
        return model_path
    if os.environ.get(MODELS_ENV):
        return None
    sibling = (Path(__file__).resolve().parents[2]
               / "BetterHuman" / "models" / "MHR" / "converted" / f"mhr_lod{lod}.npz")
    if not sibling.is_file():
        raise FileNotFoundError(
            f"no MHR LOD{lod} archive: set mhr_body.model_path, ${MODELS_ENV}, or "
            f"place the BetterHuman checkout at {sibling}")
    return str(sibling)
