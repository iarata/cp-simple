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


if __name__ == "__main__":
    unittest.main()
