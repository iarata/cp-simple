from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from cps.models.detr import DETRCriterion, ModelConfig
from cps.training.validate import validation_loop


class RecordingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, images, return_attention: bool = False):
        batch_size = len(images)
        self.batch_sizes.append(batch_size)
        return {
            "pred_logits": torch.zeros((batch_size, 2, 3), device=images[0].device),
            "pred_boxes": torch.zeros((batch_size, 2, 4), device=images[0].device),
            "pred_masks": torch.zeros((batch_size, 2, 4, 4), device=images[0].device),
        }


class ValidationLoopTest(unittest.TestCase):
    def test_forward_batch_size_splits_large_loader_batches(self) -> None:
        images = [torch.zeros(3, 16, 16) for _ in range(5)]
        targets = [
            {
                "orig_size": torch.tensor([16, 16]),
                "size": torch.tensor([16, 16]),
                "image_id": torch.tensor([idx]),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "category_ids": torch.zeros((0,), dtype=torch.int64),
                "boxes": torch.zeros((0, 4)),
                "masks": torch.zeros((0, 16, 16)),
            }
            for idx in range(5)
        ]
        model = RecordingModel()

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("cps.training.validate.outputs_to_predictions", return_value=[]),
            patch("cps.training.validate.evaluate_coco_predictions", return_value={}),
        ):
            validation_loop(
                model=model,
                criterion=None,
                dataloader=[(images, targets)],
                device=torch.device("cpu"),
                label_to_cat_id={},
                categories={},
                annotation_json=Path(tmp) / "annotations.json",
                output_dir=tmp,
                visualize_batches=0,
                attention_samples=0,
                forward_batch_size=2,
            )

        self.assertEqual(model.batch_sizes, [2, 2, 1])

    def test_loss_mode_skips_prediction_export_and_coco_eval(self) -> None:
        images = [torch.zeros(3, 16, 16)]
        targets = [
            {
                "orig_size": torch.tensor([16, 16]),
                "size": torch.tensor([16, 16]),
                "image_id": torch.tensor([1]),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "category_ids": torch.zeros((0,), dtype=torch.int64),
                "boxes": torch.zeros((0, 4)),
                "masks": torch.zeros((0, 16, 16)),
            }
        ]

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("cps.training.validate.outputs_to_predictions") as to_predictions,
            patch("cps.training.validate.evaluate_coco_predictions") as evaluate,
        ):
            metrics = validation_loop(
                model=RecordingModel(),
                criterion=None,
                dataloader=[(images, targets)],
                device=torch.device("cpu"),
                label_to_cat_id={},
                categories={},
                annotation_json=Path(tmp) / "annotations.json",
                output_dir=tmp,
                mode="loss",
            )

        to_predictions.assert_not_called()
        evaluate.assert_not_called()
        self.assertFalse(metrics["available"])

    def test_gpu_loss_mode_stays_on_loss_path_without_prediction_export(self) -> None:
        images = [torch.zeros(3, 16, 16)]
        targets = [
            {
                "orig_size": torch.tensor([16, 16]),
                "size": torch.tensor([16, 16]),
                "image_id": torch.tensor([1]),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "category_ids": torch.zeros((0,), dtype=torch.int64),
                "boxes": torch.zeros((0, 4)),
                "masks": torch.zeros((0, 16, 16)),
            }
        ]
        criterion = DETRCriterion(num_classes=2, config=ModelConfig(num_queries=2))

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("cps.training.validate.outputs_to_predictions") as to_predictions,
            patch("cps.training.validate.evaluate_coco_predictions") as evaluate,
        ):
            metrics = validation_loop(
                model=RecordingModel(),
                criterion=criterion,
                dataloader=[(images, targets)],
                device=torch.device("cpu"),
                label_to_cat_id={},
                categories={},
                annotation_json=Path(tmp) / "annotations.json",
                output_dir=tmp,
                mode="gpu_loss",
            )

        to_predictions.assert_not_called()
        evaluate.assert_not_called()
        self.assertFalse(metrics["available"])
        self.assertIn("loss", metrics["losses"])

    def test_bbox_mode_skips_mask_prediction_export(self) -> None:
        images = [torch.zeros(3, 16, 16)]
        targets = [
            {
                "orig_size": torch.tensor([16, 16]),
                "size": torch.tensor([16, 16]),
                "image_id": torch.tensor([1]),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "category_ids": torch.zeros((0,), dtype=torch.int64),
                "boxes": torch.zeros((0, 4)),
                "masks": torch.zeros((0, 16, 16)),
            }
        ]

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "cps.training.validate.outputs_to_predictions", return_value=[]
            ) as to_predictions,
            patch("cps.training.validate.evaluate_coco_predictions", return_value={}) as evaluate,
        ):
            validation_loop(
                model=RecordingModel(),
                criterion=None,
                dataloader=[(images, targets)],
                device=torch.device("cpu"),
                label_to_cat_id={},
                categories={},
                annotation_json=Path(tmp) / "annotations.json",
                output_dir=tmp,
                mode="bbox",
                visualize_batches=0,
                attention_samples=0,
            )

        self.assertFalse(to_predictions.call_args.kwargs["include_masks"])
        self.assertEqual(evaluate.call_args.kwargs["iou_types"], ("bbox",))


if __name__ == "__main__":
    unittest.main()
