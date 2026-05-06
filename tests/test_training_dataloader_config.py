from __future__ import annotations

import unittest
from unittest.mock import patch

from cps.training import train
from omegaconf import OmegaConf


def training_cfg(**overrides):
    cfg = OmegaConf.create(
        {
            "train": {
                "batch_size": 64,
                "num_workers": 12,
                "prefetch_factor": 1,
                "pin_memory": None,
                "persistent_workers": False,
                "multiprocessing_sharing_strategy": "file_system",
            },
            "eval": {
                "batch_size": 32,
                "num_workers": None,
            },
        }
    )
    for key, value in overrides.items():
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = getattr(node, part)
        setattr(node, parts[-1], value)
    return cfg


class TrainingDataLoaderConfigTest(unittest.TestCase):
    def test_worker_kwargs_include_prefetch_only_with_workers(self) -> None:
        cfg = training_cfg()

        kwargs = train._dataloader_kwargs(cfg, "train", shuffle=True)

        self.assertEqual(kwargs["num_workers"], 12)
        self.assertEqual(kwargs["prefetch_factor"], 1)
        self.assertFalse(kwargs["persistent_workers"])

    def test_worker_zero_omits_prefetch_and_persistent_workers(self) -> None:
        cfg = training_cfg(**{"train.num_workers": 0})

        kwargs = train._dataloader_kwargs(cfg, "train", shuffle=True)

        self.assertEqual(kwargs["num_workers"], 0)
        self.assertNotIn("prefetch_factor", kwargs)
        self.assertNotIn("persistent_workers", kwargs)

    def test_eval_num_workers_can_override_train_workers(self) -> None:
        cfg = training_cfg(**{"eval.num_workers": 2})

        kwargs = train._dataloader_kwargs(cfg, "val", shuffle=False)

        self.assertEqual(kwargs["batch_size"], 32)
        self.assertEqual(kwargs["num_workers"], 2)

    def test_configure_multiprocessing_sets_supported_strategy(self) -> None:
        cfg = training_cfg()

        with (
            patch.object(
                train.torch.multiprocessing,
                "get_all_sharing_strategies",
                return_value={"file_descriptor", "file_system"},
            ),
            patch.object(
                train.torch.multiprocessing,
                "get_sharing_strategy",
                side_effect=["file_descriptor", "file_system"],
            ),
            patch.object(train.torch.multiprocessing, "set_sharing_strategy") as set_strategy,
        ):
            train.configure_torch_multiprocessing(cfg)

        set_strategy.assert_called_once_with("file_system")


if __name__ == "__main__":
    unittest.main()
