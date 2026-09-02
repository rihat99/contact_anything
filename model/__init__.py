"""ContactAnything model package.

``sam_3d_body/`` is the vendored SAM-3D-Body fork (near-upstream; its only
extension is the generic extra-token-block mechanism). Everything trainable
lives in the sibling modules:

* :mod:`model.wrapper`  — frozen-base wrapper (build / freeze / forward / readout)
* :mod:`model.tokens`   — learned token blocks + anchored per-layer updates
* :mod:`model.rope`     — RoPE temporal transformers (pose + cross-modal)
* :mod:`model.heads`    — contact / force / motion prediction heads
* :mod:`model.network`  — :class:`ContactAnything`, the composed model
"""
from model.network import ContactAnything
from model.wrapper import SAM3DBodyWrapper

__all__ = ["ContactAnything", "SAM3DBodyWrapper"]
