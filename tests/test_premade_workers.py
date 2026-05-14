from __future__ import annotations

import unittest
from unittest.mock import patch

from cps.data import subsets


class PremadeWorkerSelectionTest(unittest.TestCase):
    def test_lbm_libcom_auto_uses_cuda_vram_budget(self) -> None:
        premade_cfg = {
            "num_workers": 0,
            "device": "cuda",
            "lbm_auto_max_workers": 4,
            "lbm_worker_vram_gb": 9.0,
            "lbm_vram_reserve_gb": 4.0,
        }
        method_cfg = {"name": "lbm_copy_paste", "harmonizer_backend": "libcom"}

        with patch.object(subsets.subprocess, "check_output", return_value="32768\n"):
            self.assertEqual(subsets._premade_num_workers(premade_cfg, method_cfg), 3)

    def test_lbm_libcom_auto_stays_conservative_on_small_cuda_gpu(self) -> None:
        premade_cfg = {
            "num_workers": 0,
            "device": "cuda",
            "lbm_auto_max_workers": 4,
            "lbm_worker_vram_gb": 9.0,
            "lbm_vram_reserve_gb": 4.0,
        }
        method_cfg = {"name": "lbm_copy_paste", "harmonizer_backend": "libcom"}

        with patch.object(subsets.subprocess, "check_output", return_value="16384\n"):
            self.assertEqual(subsets._premade_num_workers(premade_cfg, method_cfg), 1)

    def test_configured_workers_override_lbm_auto(self) -> None:
        premade_cfg = {"num_workers": 5, "device": "cuda"}
        method_cfg = {"name": "lbm_copy_paste", "harmonizer_backend": "libcom"}

        self.assertEqual(subsets._premade_num_workers(premade_cfg, method_cfg), 5)

    def test_non_lbm_libcom_auto_remains_single_worker(self) -> None:
        premade_cfg = {"num_workers": 0, "device": "cuda"}
        method_cfg = {"name": "pctnet_copy_paste", "harmonizer_backend": "libcom"}

        self.assertEqual(subsets._premade_num_workers(premade_cfg, method_cfg), 1)


class PremadeBalanceTest(unittest.TestCase):
    def test_balance_applies_to_copy_paste_methods_only(self) -> None:
        premade_cfg = {
            "balance": {
                "enabled": True,
                "methods": ["simple_copy_paste", "pctnet_copy_paste", "lbm_copy_paste"],
            }
        }

        self.assertFalse(subsets._balance_enabled_for_method("none", premade_cfg))
        self.assertTrue(subsets._balance_enabled_for_method("simple_copy_paste", premade_cfg))

    def test_balance_drops_person_heavy_images_when_rare_targets_are_met(self) -> None:
        premade = {
            "images": [
                {"id": image_id, "file_name": f"{image_id}.jpg"} for image_id in range(1, 7)
            ],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "person"},
                {"id": 2, "name": "rare_a"},
                {"id": 3, "name": "rare_b"},
            ],
        }
        next_ann_id = 1
        for image_id, category_id, count in [
            (1, 1, 10),
            (2, 1, 10),
            (3, 1, 10),
            (4, 1, 1),
            (5, 2, 3),
            (6, 3, 3),
        ]:
            for _ in range(count):
                premade["annotations"].append(
                    {
                        "id": next_ann_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [0, 0, 1, 1],
                        "area": 1,
                        "iscrowd": 0,
                    }
                )
                next_ann_id += 1
        premade_cfg = {
            "balance": {
                "target_strategy": "min_count_multiplier",
                "min_count_multiplier": 1.0,
                "min_target_instances": None,
                "max_target_instances": None,
                "dominant_top_k": 1,
                "dominant_max_overshoot_ratio": 1.5,
            }
        }

        balanced, metadata = subsets._balance_premade_subset(premade, premade_cfg, seed=1337)
        selected_image_ids = {int(image["id"]) for image in balanced["images"]}
        counts = subsets.instance_count_per_class(balanced)

        self.assertNotIn(1, selected_image_ids)
        self.assertNotIn(2, selected_image_ids)
        self.assertNotIn(3, selected_image_ids)
        self.assertEqual(counts[1], 1)
        self.assertEqual(counts[2], 3)
        self.assertEqual(counts[3], 3)
        self.assertEqual(metadata["target_instances_per_class"], 3)


if __name__ == "__main__":
    unittest.main()
