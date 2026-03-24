"""Low-Rank Adaptation (LoRA) for nn.Linear layers.

Usage:
    from sam_3d_body.models.modules.lora import apply_lora_to_decoder

    apply_lora_to_decoder(model.decoder, lora_cfg)
    # lora_cfg keys: RANK, ALPHA, DROPOUT, TARGET_MODULES, TARGET_PROJECTIONS
"""

import math
import torch
import torch.nn as nn
from typing import List


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with a parallel low-rank adapter.

    forward(x) = original_linear(x) + scaling * lora_B(lora_A(dropout(x)))

    The original linear is kept frozen; only lora_A and lora_B are trainable.
    B is zero-initialized so the adapter starts as identity (no change).
    """

    def __init__(self, original: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.original = original
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        in_features = original.in_features
        out_features = original.out_features
        self.scaling = alpha / rank

        device = original.weight.device
        self.lora_A = nn.Linear(in_features, rank, bias=False, device=device)
        self.lora_B = nn.Linear(rank, out_features, bias=False, device=device)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Kaiming init for A, zero init for B (adapter starts as zero)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        lora = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return base + self.scaling * lora

    @property
    def in_features(self):
        return self.original.in_features

    @property
    def out_features(self):
        return self.original.out_features


def apply_lora_to_decoder(
    decoder: nn.Module,
    lora_cfg,
) -> int:
    """Inject LoRA adapters into a PromptableDecoder's TransformerDecoderLayers.

    Args:
        decoder: The PromptableDecoder module (has `decoder.layers`).
        lora_cfg: Config node with RANK, ALPHA, DROPOUT, TARGET_MODULES, TARGET_PROJECTIONS.

    Returns:
        Total number of LoRA parameters added.
    """
    rank = lora_cfg.get("RANK", 16)
    alpha = lora_cfg.get("ALPHA", 16)
    dropout = lora_cfg.get("DROPOUT", 0.0)
    target_modules: List[str] = list(lora_cfg.get("TARGET_MODULES", ["self_attn", "cross_attn"]))
    target_projections: List[str] = list(lora_cfg.get("TARGET_PROJECTIONS", ["q_proj", "v_proj"]))

    total_params = 0
    count = 0

    for layer_idx, layer in enumerate(decoder.layers):
        for mod_name in target_modules:
            attn = getattr(layer, mod_name, None)
            if attn is None:
                continue
            for proj_name in target_projections:
                linear = getattr(attn, proj_name, None)
                if linear is None or not isinstance(linear, nn.Linear):
                    continue
                lora_layer = LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout)
                setattr(attn, proj_name, lora_layer)
                n_params = sum(p.numel() for p in [lora_layer.lora_A.weight, lora_layer.lora_B.weight])
                total_params += n_params
                count += 1

    print(f"LoRA: injected {count} adapters (rank={rank}, alpha={alpha}) — "
          f"{total_params:,} trainable LoRA params")
    return total_params
