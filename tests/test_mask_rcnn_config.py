from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from cps.models import mask_rcnn
from cps.models.mask_rcnn import MaskRCNNConfig, MaskRCNNSegmenter, model_config_from_cfg


class _FakeFeatureInfo:
    def channels(self) -> list[int]:
        return [8, 16, 32, 64]


class _FakeTimmFeatures(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_info = _FakeFeatureInfo()
        self.grad_checkpointing_enabled = False

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing_enabled = bool(enable)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        batch = x.shape[0]
        return [
            torch.zeros(batch, 8, 8, 8, device=x.device),
            torch.zeros(batch, 4, 4, 16, device=x.device),
            torch.zeros(batch, 2, 2, 32, device=x.device),
            torch.zeros(batch, 1, 1, 64, device=x.device),
        ]


class MaskRCNNMemoryConfigTest(unittest.TestCase):
    def test_premade_config_uses_large_batch_memory_knobs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cfg = OmegaConf.load(repo_root / "configs/model/maskrcnn_swin_t.yaml")

        model_cfg = model_config_from_cfg(cfg)

        self.assertTrue(model_cfg.backbone_grad_checkpointing)
        self.assertEqual(model_cfg.box_batch_size_per_image, 128)
        self.assertEqual(model_cfg.box_positive_fraction, 0.25)
        self.assertEqual(model_cfg.rpn_batch_size_per_image, 128)
        self.assertEqual(model_cfg.rpn_positive_fraction, 0.5)
        self.assertEqual(model_cfg.rpn_pre_nms_top_n_train, 1000)

    def test_overrides_are_passed_to_torchvision_and_timm(self) -> None:
        fake_body = _FakeTimmFeatures()
        config = MaskRCNNConfig(
            backbone_pretrained=False,
            backbone_grad_checkpointing=True,
            image_size=64,
            fpn_out_channels=16,
            box_batch_size_per_image=96,
            box_positive_fraction=0.5,
            rpn_batch_size_per_image=64,
            rpn_positive_fraction=0.25,
            rpn_pre_nms_top_n_train=333,
            rpn_pre_nms_top_n_test=222,
            rpn_post_nms_top_n_train=111,
            rpn_post_nms_top_n_test=77,
        )

        with (
            patch.object(mask_rcnn.timm, "create_model", return_value=fake_body),
            patch.object(mask_rcnn, "MaskRCNN", return_value=torch.nn.Identity()) as mask_rcnn_cls,
        ):
            model = MaskRCNNSegmenter(num_classes=3, config=config)

        self.assertTrue(model.backbone.body.grad_checkpointing_enabled)
        _, kwargs = mask_rcnn_cls.call_args
        self.assertEqual(kwargs["box_batch_size_per_image"], 96)
        self.assertEqual(kwargs["box_positive_fraction"], 0.5)
        self.assertEqual(kwargs["rpn_batch_size_per_image"], 64)
        self.assertEqual(kwargs["rpn_positive_fraction"], 0.25)
        self.assertEqual(kwargs["rpn_pre_nms_top_n_train"], 333)
        self.assertEqual(kwargs["rpn_pre_nms_top_n_test"], 222)
        self.assertEqual(kwargs["rpn_post_nms_top_n_train"], 111)
        self.assertEqual(kwargs["rpn_post_nms_top_n_test"], 77)

    def test_grad_checkpointing_can_be_disabled(self) -> None:
        fake_body = _FakeTimmFeatures()
        config = MaskRCNNConfig(
            backbone_pretrained=False,
            backbone_grad_checkpointing=False,
            image_size=64,
            fpn_out_channels=16,
        )

        with (
            patch.object(mask_rcnn.timm, "create_model", return_value=fake_body),
            patch.object(mask_rcnn, "MaskRCNN", return_value=torch.nn.Identity()),
        ):
            model = MaskRCNNSegmenter(num_classes=3, config=config)

        self.assertFalse(model.backbone.body.grad_checkpointing_enabled)


if __name__ == "__main__":
    unittest.main()
