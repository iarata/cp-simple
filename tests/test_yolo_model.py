from __future__ import annotations

import unittest

import torch
from cps.config import load_config
from cps.models import build_model_and_criterion, outputs_to_predictions
from cps.models.yolo import YOLO_MODEL_TYPE, yolo_config_from_cfg
from omegaconf import OmegaConf


def _target(image_id: int = 1, label: int = 1) -> dict[str, torch.Tensor]:
    mask = torch.zeros((1, 128, 128), dtype=torch.uint8)
    mask[:, 10:50, 10:50] = 1
    return {
        "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
        "labels": torch.tensor([label], dtype=torch.int64),
        "category_ids": torch.tensor([10 + label], dtype=torch.int64),
        "masks": mask,
        "image_id": torch.tensor([image_id], dtype=torch.int64),
        "orig_size": torch.tensor([128, 128], dtype=torch.int64),
        "size": torch.tensor([128, 128], dtype=torch.int64),
    }


class YOLOModelTest(unittest.TestCase):
    def test_yolo26_override_is_available_for_all_premade_train_configs(self) -> None:
        for config_name in (
            "train_premade/none_online_normal",
            "train_premade/simple_copy_paste",
            "train_premade/pctnet_copy_paste",
            "train_premade/lbm_copy_paste",
        ):
            cfg = load_config(config_name, overrides=["model=yolo26n"])
            self.assertEqual(cfg.model.name, "yolo26n")
            self.assertEqual(cfg.model.architecture, "yolo26n-seg.yaml")
            self.assertFalse(cfg.model.pretrained)

    def test_yolo26_model_uses_yaml_architecture_and_native_loss(self) -> None:
        cfg = OmegaConf.create(
            {
                "name": "yolo26n",
                "family": "yolo26",
                "architecture": "yolo26n-seg.yaml",
                "pretrained": False,
                "verbose": False,
            }
        )
        train_cfg = OmegaConf.create({"epochs": 2})
        model, criterion = build_model_and_criterion(cfg, num_classes=3, train_cfg=train_cfg)
        self.assertEqual(model.config.architecture, "yolo26n-seg.yaml")
        self.assertFalse(model.config.pretrained)

        model.train()
        criterion.train()
        outputs = model([torch.zeros(3, 128, 128)])
        losses = criterion(outputs, [_target()])

        self.assertIn("loss", losses)
        self.assertIn("loss_seg", losses)
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()

    def test_yolo26_rejects_pretrained_weight_files(self) -> None:
        cfg = OmegaConf.create(
            {
                "name": "yolo26n",
                "family": "yolo26",
                "architecture": "yolo26n-seg.pt",
                "pretrained": False,
            }
        )

        with self.assertRaisesRegex(ValueError, "yaml"):
            yolo_config_from_cfg(cfg)

    def test_yolo_outputs_to_predictions_maps_zero_based_classes_to_coco_ids(self) -> None:
        detections = torch.zeros((1, 2, 38), dtype=torch.float32)
        detections[0, 0, :6] = torch.tensor([2.0, 4.0, 10.0, 12.0, 0.9, 1.0])
        detections[0, 0, 6:] = 1.0
        detections[0, 1, :6] = torch.tensor([0.0, 0.0, 4.0, 4.0, 0.01, 0.0])
        outputs = {
            "model_type": YOLO_MODEL_TYPE,
            "detections": detections,
            "proto": torch.ones((1, 32, 16, 16), dtype=torch.float32),
            "image_size": torch.tensor([16, 16], dtype=torch.int64),
        }
        target = {
            "image_id": torch.tensor([123], dtype=torch.int64),
            "orig_size": torch.tensor([32, 32], dtype=torch.int64),
            "size": torch.tensor([16, 16], dtype=torch.int64),
        }

        predictions = outputs_to_predictions(
            outputs,
            [target],
            label_to_cat_id={1: 10, 2: 20},
            score_threshold=0.05,
            max_detections=10,
        )

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["image_id"], 123)
        self.assertEqual(predictions[0]["category_id"], 20)
        self.assertEqual(predictions[0]["bbox_xyxy"], [4.0, 8.0, 20.0, 24.0])
        self.assertEqual(predictions[0]["mask"].shape, (32, 32))


if __name__ == "__main__":
    unittest.main()
