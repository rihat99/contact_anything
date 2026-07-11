"""Lightweight browser application for inspecting contact datasets."""

__all__ = ["create_app"]


def create_app(*args, **kwargs):
    """Create the FastAPI app without importing rendering dependencies eagerly."""
    from .app import create_app as factory
    return factory(*args, **kwargs)
