from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from cps.models import dinov3_mask_rcnn
from cps.models.dinov3_mask_rcnn import (
    DINOv3Backbone,
    DINOv3MaskRCNNConfig,
    DINOv3MaskRCNNSegmenter,
    model_config_from_cfg,
)


class _FakeDINOBody(torch.nn.Module):
    def __init__(self, embed_dim: int = 32, blocks: int = 12) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.blocks = torch.nn.ModuleList(torch.nn.Identity() for _ in range(blocks))
        self.grad_checkpointing_enabled = False
        self.forward_calls: list[dict[str, object]] = []

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing_enabled = bool(enable)

    def forward_intermediates(
        self,
        x: torch.Tensor,
        indices: list[int],
        output_fmt: str,
        intermediates_only: bool,
        norm: bool,
    ) -> list[torch.Tensor]:
        self.forward_calls.append(
            {
                "indices": list(indices),
                "output_fmt": output_fmt,
                "intermediates_only": intermediates_only,
                "norm": norm,
            }
        )
        size = max(int(x.shape[-1]) // 16, 1)
        return [
            x.new_zeros((x.shape[0], self.embed_dim, size, size)) + float(i)
            for i, _ in enumerate(indices)
        ]


class DINOv3MaskRCNNConfigTest(unittest.TestCase):
    def test_model_yaml_defaults_enable_fast_dinov3_knobs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cfg = OmegaConf.load(repo_root / "configs/model/dinov3_mask_rcnn.yaml")

        model_cfg = model_config_from_cfg(cfg)

        self.assertFalse(model_cfg.backbone_grad_checkpointing)
        self.assertAlmostEqual(model_cfg.backbone_drop_path_rate, 0.1)
        self.assertTrue(model_cfg.compile_backbone)
        self.assertEqual(model_cfg.compile_mode, "reduce-overhead")
        self.assertEqual(model_cfg.image_size, 256)
        self.assertEqual(model_cfg.mask_head_channels, 128)
        self.assertEqual(model_cfg.mask_head_num_convs, 2)
        self.assertEqual(model_cfg.rpn_pre_nms_top_n_train, 500)
        self.assertEqual(model_cfg.rpn_post_nms_top_n_train, 500)
        self.assertEqual(model_cfg.anchor_sizes[0], (16,))
        self.assertEqual(model_cfg.anchor_sizes[-1], (256,))

    def test_base_model_yaml_selects_stronger_backbone(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cfg = OmegaConf.load(repo_root / "configs/model/dinov3_mask_rcnn_base.yaml")

        model_cfg = model_config_from_cfg(cfg)

        self.assertEqual(model_cfg.backbone_name, "vit_base_patch16_dinov3")
        self.assertTrue(model_cfg.backbone_grad_checkpointing)
        self.assertAlmostEqual(model_cfg.backbone_drop_path_rate, 0.15)
        self.assertEqual(model_cfg.mask_head_channels, 128)

    def test_custom_mask_and_box_heads_are_passed_to_torchvision(self) -> None:
        fake_body = _FakeDINOBody()
        create_kwargs: dict[str, object] = {}

        def create_model(*args: object, **kwargs: object) -> _FakeDINOBody:
            del args
            create_kwargs.update(kwargs)
            return fake_body

        config = DINOv3MaskRCNNConfig(
            backbone_pretrained=False,
            backbone_drop_path_rate=0.2,
            compile_backbone=False,
            image_size=64,
            fpn_out_channels=32,
            mask_head_channels=64,
            mask_head_num_convs=2,
            box_batch_size_per_image=16,
            rpn_batch_size_per_image=16,
        )

        with (
            patch.object(dinov3_mask_rcnn.timm, "create_model", side_effect=create_model),
            patch.object(
                dinov3_mask_rcnn, "MaskRCNN", return_value=torch.nn.Identity()
            ) as mask_rcnn_cls,
        ):
            DINOv3MaskRCNNSegmenter(num_classes=3, config=config)

        self.assertEqual(create_kwargs["drop_path_rate"], 0.2)
        _, kwargs = mask_rcnn_cls.call_args
        self.assertIsNone(kwargs["num_classes"])
        self.assertIsNotNone(kwargs["box_head"])
        self.assertIsNotNone(kwargs["box_predictor"])
        self.assertEqual(kwargs["mask_head"][0][0].out_channels, 64)
        self.assertEqual(kwargs["mask_predictor"].conv5_mask.in_channels, 64)
        self.assertEqual(kwargs["mask_predictor"].mask_fcn_logits.out_channels, 4)
        self.assertEqual(kwargs["box_batch_size_per_image"], 16)
        self.assertEqual(kwargs["rpn_batch_size_per_image"], 16)

    def test_compile_backbone_uses_configured_mode(self) -> None:
        fake_body = _FakeDINOBody()

        def compile_identity(fn: object, **kwargs: object) -> object:
            self.assertEqual(kwargs["mode"], "reduce-overhead")
            return fn

        config = DINOv3MaskRCNNConfig(
            backbone_pretrained=False,
            compile_backbone=True,
            compile_mode="reduce-overhead",
            image_size=64,
        )

        with (
            patch.object(dinov3_mask_rcnn.timm, "create_model", return_value=fake_body),
            patch.object(dinov3_mask_rcnn.torch, "compile", side_effect=compile_identity),
        ):
            backbone = DINOv3Backbone(config)

        feat = backbone(torch.rand(1, 3, 64, 64))

        self.assertEqual(tuple(feat.shape), (1, 32, 4, 4))
        self.assertEqual(fake_body.forward_calls[-1]["indices"], [0, 5, 11])


if __name__ == "__main__":
    unittest.main()
