"""Contact-head output modes: token-aligned and pooled targets."""
from __future__ import annotations

import pytest
import torch
from yacs.config import CfgNode

from sam_3d_body.models.heads import build_head
from sam_3d_body.models.heads.contact_head import ContactHead
from sam_3d_body.models.meta_arch.sam3d_body import SAM3DBody


def _head(pool_mode: str, *, tokens: int = 4, outputs: int = 4) -> ContactHead:
    return ContactHead(
        input_dim=16,
        num_contact_tokens=tokens,
        output_dims=outputs,
        mlp_depth=2,
        mlp_channel_div_factor=2,
        pool_mode=pool_mode,
        dropout=0.0,
    )


def test_per_token_returns_one_logit_per_token_and_backpropagates():
    head = _head("per_token")
    tokens = torch.randn(3, 4, 16, requires_grad=True)

    logits = head(tokens)
    assert logits.shape == (3, 4)

    logits.sum().backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    assert float(tokens.grad.abs().sum()) > 0.0
    assert all(p.grad is not None for p in head.parameters())


def test_per_token_mlp_is_shared_and_preserves_token_identity():
    head = _head("per_token").eval()
    tokens = torch.randn(2, 4, 16)
    permutation = torch.tensor([2, 0, 3, 1])

    expected = head(tokens)[:, permutation]
    actual = head(tokens[:, permutation])
    assert torch.equal(actual, expected)


def test_per_token_rejects_output_token_count_mismatch():
    with pytest.raises(ValueError, match="output_dims.*token count"):
        _head("per_token", tokens=4, outputs=22)


def test_per_token_rejects_runtime_token_count_mismatch():
    head = _head("per_token")
    with pytest.raises(ValueError, match="input=5, configured=4"):
        head(torch.randn(2, 5, 16))


@pytest.mark.parametrize("pool_mode", ["attention", "concat"])
def test_pooled_modes_keep_arbitrary_output_dimensions(pool_mode):
    head = _head(pool_mode, tokens=4, outputs=22)
    logits = head(torch.randn(2, 4, 16))
    assert logits.shape == (2, 22)


def test_unknown_pool_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown pool_mode"):
        _head("not-a-mode")


def test_builder_counts_anchored_and_global_tokens_for_per_token_mode():
    cfg = CfgNode({
        "MODEL": {
            "DECODER": {"DIM": 16},
            "CONTACT_HEAD": {
                "NUM_CONTACTS": 4,
                "NUM_GLOBAL_TOKENS": 0,
                "POOL_MODE": "per_token",
                "MLP_DEPTH": 2,
                "MLP_CHANNEL_DIV_FACTOR": 2,
                "DROPOUT": 0.0,
            },
        },
    })

    head = build_head(cfg, "contact", output_dims=4)
    assert head.num_contact_tokens == 4
    assert head(torch.randn(2, 4, 16)).shape == (2, 4)

    cfg.MODEL.CONTACT_HEAD.NUM_GLOBAL_TOKENS = 1
    with pytest.raises(ValueError, match="num_contact_tokens=5"):
        build_head(cfg, "contact", output_dims=4)


def test_four_extremity_anchors_use_existing_iterative_patch_update():
    class CoordinateProjection(torch.nn.Module):
        def forward(self, coordinates):
            return torch.cat([coordinates, coordinates], dim=-1)

    class UpdateHarness(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_contact_tokens = 4
            self.contact_keypoint_indices = [62, 41, 13, 14]
            self.contact_posemb_linear = CoordinateProjection()
            self.contact_feat_linear = torch.nn.Identity()
            self.contact_grid_size = 1
            self.contact_grid_radius = 0.1
            self.cfg = CfgNode({"MODEL": {"BACKBONE": {"TYPE": "dinov3_vith16plus"}}})
            # contact_token_update_fn delegates the anchored sampling to this
            # shared helper (also used by the force tokens).
            self._anchored_token_update = SAM3DBody._anchored_token_update.__get__(self)

    harness = UpdateHarness()
    batch_size, channels, token_start = 2, 4, 3
    pose_coordinates = torch.zeros(batch_size, 70, 2)
    selected = torch.tensor([
        [[-0.3, -0.2], [0.3, -0.2], [-0.2, 0.3], [0.2, 0.3]],
        [[-0.2, -0.1], [0.2, -0.1], [-0.1, 0.2], [0.1, 0.2]],
    ])
    pose_coordinates[:, harness.contact_keypoint_indices] = selected
    pose_output = {
        "pred_keypoints_2d_cropped": pose_coordinates,
        "pred_keypoints_2d_depth": torch.ones(batch_size, 70),
    }
    token_embeddings = torch.zeros(batch_size, token_start + 4, channels)
    token_augment = torch.zeros_like(token_embeddings)
    image_embeddings = torch.randn(batch_size, channels, 4, 4)

    updated, augment, _, _ = SAM3DBody.contact_token_update_fn(
        harness,
        token_start,
        image_embeddings,
        decoder_layers=[object(), object()],
        batch={},
        token_embeddings=token_embeddings,
        token_augment=token_augment,
        pose_output=pose_output,
        layer_idx=0,
    )

    assert torch.equal(
        augment[:, token_start: token_start + 4],
        torch.cat([selected, selected], dim=-1),
    )
    assert float(updated[:, token_start: token_start + 4].abs().sum()) > 0.0
    assert torch.equal(updated[:, :token_start], token_embeddings[:, :token_start])
