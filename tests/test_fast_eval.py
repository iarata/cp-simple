from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from cps.training.fast_eval import ProbeSet, build_probe_set, run_fast_validation


def _sample(image_id: int, category_id: int) -> dict:
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    image[..., 0] = (image_id * 17) % 255
    mask = np.zeros((24, 24), dtype=bool)
    mask[5:17, 5:17] = True
    return {
        "image": image,
        "instances": [
            {
                "annotation_id": image_id,
                "category_id": category_id,
                "bbox_xyxy": [5.0, 5.0, 16.0, 16.0],
                "bbox": [5.0, 5.0, 16.0, 16.0],
                "bbox_mode": "xyxy",
                "area": float(mask.sum()),
                "iscrowd": 0,
                "mask": mask,
            }
        ],
        "image_info": {"id": image_id, "file_name": f"{image_id}.jpg"},
        "height": 24,
        "width": 24,
        "orig_height": 24,
        "orig_width": 24,
        "augmentation_meta": {"method": "none"},
    }


class FakeDataset:
    def __init__(self, category_ids: list[int]) -> None:
        self.samples = [_sample(idx + 1, category_id) for idx, category_id in enumerate(category_ids)]
        self.images = [dict(sample["image_info"]) for sample in self.samples]
        self.cat_id_to_label = {1: 1, 2: 2}
        self.label_to_cat_id = {1: 1, 2: 2}
        self.cat_id_to_name = {1: "common", 2: "rare"}
        annotations = [
            {
                "id": idx + 1,
                "image_id": idx + 1,
                "category_id": category_id,
                "iscrowd": 0,
            }
            for idx, category_id in enumerate(category_ids)
        ]
        self.anns_by_image = {
            int(ann["image_id"]): [ann] for ann in annotations
        }
        self.coco = {
            "images": self.images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "common"}, {"id": 2, "name": "rare"}],
        }

    def get_raw_sample(self, index: int) -> dict:
        sample = self.samples[int(index) % len(self.samples)]
        return {
            **sample,
            "image": sample["image"].copy(),
            "image_info": dict(sample["image_info"]),
            "augmentation_meta": dict(sample["augmentation_meta"]),
            "instances": [
                {**inst, "mask": np.asarray(inst["mask"], dtype=bool).copy()}
                for inst in sample["instances"]
            ],
        }


class FakeImage:
    def __init__(self, data, caption: str | None = None) -> None:
        self.data = data
        self.caption = caption


class FakeTable:
    def __init__(self, columns: list[str], data: list[list]) -> None:
        self.columns = columns
        self.data = data


class FakeRun:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def log(self, payload: dict) -> None:
        self.payloads.append(payload)

    def __bool__(self) -> bool:
        return True


class ProbeModel(torch.nn.Module):
    def forward(self, images, return_attention: bool = False):
        assert return_attention
        device = images[0].device
        return {
            "pred_logits": torch.tensor([[[0.0, 8.0, 0.0], [0.0, 0.0, 1.0]]], device=device),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.1, 0.1]]], device=device),
            "pred_masks": torch.ones((1, 2, 2, 2), device=device),
            "cross_attention": torch.tensor(
                [[[[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]]], device=device
            ),
            "attention_hw": torch.tensor([2, 2], device=device),
        }


class FastEvalTest(unittest.TestCase):
    def test_probe_set_contains_three_normal_and_three_copy_paste_rare_samples(self) -> None:
        dataset = FakeDataset([1, 1, 1, 1, 1, 1, 2, 2, 2])

        probe = build_probe_set(
            val_dataset=dataset,
            train_dataset=dataset,
            num_normal=3,
            num_underrepresented=3,
            seed=7,
            quantile=0.25,
        )

        kinds = [kind for _, _, kind in probe.samples]
        self.assertEqual(len(probe), 6)
        self.assertEqual(kinds.count("normal"), 3)
        self.assertEqual(kinds.count("underrepresented+cp"), 3)
        for _, target, kind in probe.samples:
            if kind == "underrepresented+cp":
                self.assertTrue(target["augmentation_meta"]["applied"])

    def test_fast_validation_logs_gt_predictions_attention_and_restores_mode(self) -> None:
        image = torch.zeros(3, 8, 8)
        target = {
            "boxes": torch.tensor([[1.0, 1.0, 5.0, 5.0]]),
            "labels": torch.tensor([1]),
            "category_ids": torch.tensor([10]),
            "masks": torch.ones((1, 8, 8), dtype=torch.uint8),
            "image_id": torch.tensor([123]),
            "orig_size": torch.tensor([16, 16]),
            "size": torch.tensor([8, 8]),
            "augmentation_meta": {"method": "none"},
        }
        probe = ProbeSet(
            samples=[(image, target, "normal")],
            label_to_cat_id={1: 10, 2: 20},
            cat_id_to_name={10: "common", 20: "rare"},
        )
        model = ProbeModel()
        model.train()
        run = FakeRun()
        fake_wandb = SimpleNamespace(Image=FakeImage, Table=FakeTable)

        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            run_fast_validation(
                model=model,
                probe=probe,
                device=torch.device("cpu"),
                run=run,
                epoch=4,
            )

        self.assertTrue(model.training)
        self.assertEqual(len(run.payloads), 1)
        payload = run.payloads[0]
        self.assertEqual(len(payload["fast_eval/gt"]), 1)
        self.assertEqual(len(payload["fast_eval/pred"]), 1)
        self.assertEqual(len(payload["fast_eval/attention"]), 1)
        table = payload["fast_eval/samples"]
        self.assertEqual(table.columns[0:5], ["epoch", "sample_index", "image_id", "probe_type", "copy_paste_applied"])
        self.assertEqual(table.data[0][0:5], [4, 0, 123, "normal", False])
        self.assertIsInstance(table.data[0][5], FakeImage)
        self.assertIsInstance(table.data[0][6], FakeImage)
        self.assertIsInstance(table.data[0][7], FakeImage)


if __name__ == "__main__":
    unittest.main()
