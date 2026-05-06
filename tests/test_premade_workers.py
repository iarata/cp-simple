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


if __name__ == "__main__":
    unittest.main()
