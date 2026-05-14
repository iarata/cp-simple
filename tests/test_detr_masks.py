from __future__ import annotations

import unittest

import torch
from cps.models.detr import ModelConfig, TinyDETRSegmenter, outputs_to_predictions


class DETRMaskMemoryTest(unittest.TestCase):
    def test_forward_keeps_pred_masks_at_feature_resolution(self) -> None:
        config = ModelConfig(
            hidden_dim=16,
            num_queries=3,
            num_encoder_layers=1,
            num_decoder_layers=1,
            nheads=4,
            dim_feedforward=32,
        )
        model = TinyDETRSegmenter(num_classes=2, config=config)
        model.eval()
        images = [torch.rand(3, 64, 80), torch.rand(3, 48, 64)]

        with torch.no_grad():
            outputs = model(images)

        self.assertEqual(tuple(outputs["pred_masks"].shape), (2, 3, 4, 5))
        self.assertNotIn("cross_attention", outputs)
        self.assertIsNone(model.decoder_layers[-1].last_cross_attention)

    def test_forward_stores_attention_only_when_requested(self) -> None:
        config = ModelConfig(
            hidden_dim=16,
            num_queries=3,
            num_encoder_layers=1,
            num_decoder_layers=1,
            nheads=4,
            dim_feedforward=32,
        )
        model = TinyDETRSegmenter(num_classes=2, config=config)
        model.eval()
        images = [torch.rand(3, 64, 80)]

        with torch.no_grad():
            outputs = model(images, return_attention=True)

        self.assertIn("cross_attention", outputs)
        self.assertEqual(outputs["cross_attention"].shape[:3], (1, 4, 3))

    def test_forward_can_store_first_decoder_attention(self) -> None:
        config = ModelConfig(
            hidden_dim=16,
            num_queries=3,
            num_encoder_layers=1,
            num_decoder_layers=2,
            nheads=4,
            dim_feedforward=32,
            attention_layer="first",
        )
        model = TinyDETRSegmenter(num_classes=2, config=config)
        model.eval()
        images = [torch.rand(3, 64, 80)]

        with torch.no_grad():
            outputs = model(images, return_attention=True)

        self.assertIn("cross_attention", outputs)
        self.assertEqual(int(outputs["attention_layer_index"].item()), 0)
        self.assertIsNotNone(model.decoder_layers[0].last_cross_attention)
        self.assertIsNone(model.decoder_layers[1].last_cross_attention)

    def test_forward_supports_timm_backbone_without_pretrained_download(self) -> None:
        config = ModelConfig(
            hidden_dim=16,
            num_queries=3,
            num_encoder_layers=1,
            num_decoder_layers=1,
            nheads=4,
            dim_feedforward=32,
            backbone="timm",
            backbone_name="resnet18",
            backbone_pretrained=False,
            backbone_out_index=3,
            attention_layer="first",
        )
        model = TinyDETRSegmenter(num_classes=2, config=config)
        model.eval()
        images = [torch.rand(3, 64, 80)]

        with torch.no_grad():
            outputs = model(images, return_attention=True)

        self.assertEqual(tuple(outputs["pred_masks"].shape), (1, 3, 4, 5))
        self.assertIn("cross_attention", outputs)

    def test_forward_supports_timm_efficientnet_backbone(self) -> None:
        config = ModelConfig(
            hidden_dim=16,
            num_queries=3,
            num_encoder_layers=1,
            num_decoder_layers=1,
            nheads=4,
            dim_feedforward=32,
            backbone="timm",
            backbone_name="efficientnet_b0",
            backbone_pretrained=False,
            backbone_out_index=3,
            attention_layer="first",
        )
        model = TinyDETRSegmenter(num_classes=2, config=config)
        model.eval()
        images = [torch.rand(3, 64, 80)]

        with torch.no_grad():
            outputs = model(images, return_attention=True)

        self.assertEqual(tuple(outputs["pred_masks"].shape), (1, 3, 4, 5))
        self.assertEqual(outputs["attention_hw"].tolist(), [4, 5])

    def test_outputs_to_predictions_upsamples_selected_masks(self) -> None:
        outputs = {
            "pred_logits": torch.tensor([[[0.0, 7.0, 0.0], [0.0, 0.0, 5.0]]]),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 1.0, 1.0], [0.25, 0.25, 0.2, 0.2]]]),
            "pred_masks": torch.tensor(
                [
                    [
                        [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]],
                        [[-10.0, 10.0, -10.0], [10.0, -10.0, 10.0]],
                    ]
                ]
            ),
        }
        targets = [
            {
                "orig_size": torch.tensor([8, 12]),
                "image_id": torch.tensor([123]),
            }
        ]

        predictions = outputs_to_predictions(
            outputs,
            targets,
            label_to_cat_id={1: 10, 2: 20},
            score_threshold=0.05,
            max_detections=1,
        )

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["category_id"], 10)
        self.assertEqual(predictions[0]["mask"].shape, (8, 12))

    def test_outputs_to_predictions_can_skip_masks_for_bbox_only_eval(self) -> None:
        outputs = {
            "pred_logits": torch.tensor([[[0.0, 7.0, 0.0]]]),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 1.0, 1.0]]]),
            "pred_masks": torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]]),
        }
        targets = [
            {
                "orig_size": torch.tensor([8, 12]),
                "image_id": torch.tensor([123]),
            }
        ]

        predictions = outputs_to_predictions(
            outputs,
            targets,
            label_to_cat_id={1: 10},
            include_masks=False,
        )

        self.assertEqual(len(predictions), 1)
        self.assertNotIn("mask", predictions[0])

    def test_outputs_to_predictions_can_use_model_input_size_for_visualizations(self) -> None:
        outputs = {
            "pred_logits": torch.tensor([[[0.0, 7.0, 0.0]]]),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 1.0, 1.0]]]),
            "pred_masks": torch.ones((1, 1, 2, 2)),
        }
        targets = [
            {
                "orig_size": torch.tensor([80, 120]),
                "size": torch.tensor([8, 12]),
                "image_id": torch.tensor([123]),
            }
        ]

        metric_predictions = outputs_to_predictions(outputs, targets, label_to_cat_id={1: 10})
        viz_predictions = outputs_to_predictions(
            outputs,
            targets,
            label_to_cat_id={1: 10},
            target_size_key="size",
        )

        self.assertEqual(metric_predictions[0]["mask"].shape, (80, 120))
        self.assertEqual(viz_predictions[0]["mask"].shape, (8, 12))
        self.assertEqual(metric_predictions[0]["bbox_xyxy"], [0.0, 0.0, 120.0, 80.0])
        self.assertEqual(viz_predictions[0]["bbox_xyxy"], [0.0, 0.0, 12.0, 8.0])


if __name__ == "__main__":
    unittest.main()
