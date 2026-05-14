from __future__ import annotations

import unittest

import numpy as np

from cps.data.coco import sample_to_torch


class COCOSampleResizeTest(unittest.TestCase):
    def test_sample_to_torch_keeps_original_size_separate_from_model_input_size(self) -> None:
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        mask = np.zeros((8, 10), dtype=bool)
        mask[1:6, 2:8] = True
        _, target = sample_to_torch(
            {
                "image": image,
                "instances": [{"category_id": 10, "mask": mask, "iscrowd": 0}],
                "image_info": {"id": 42},
                "orig_height": 80,
                "orig_width": 100,
                "augmentation_meta": {"method": "none"},
            },
            {10: 1},
        )

        self.assertEqual(target["size"].tolist(), [8, 10])
        self.assertEqual(target["orig_size"].tolist(), [80, 100])


if __name__ == "__main__":
    unittest.main()
