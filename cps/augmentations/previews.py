"""Augmentation preview generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cps.augmentations import build_augmentation
from cps.augmentations.normal import NormalAugmentation
from cps.data.coco import COCODataset
from cps.data.visualization import overlay_instances, save_image_grid
from cps.paths import project_path
from cps.utils.device import get_device


def generate_augmentation_previews(cfg: Any) -> list[Path]:
    dataset_cfg = cfg.dataset
    preview_cfg = cfg.augmentation_preview
    device = get_device(str(getattr(cfg.train, "device", "auto")))
    dataset = COCODataset(
        image_dir=dataset_cfg.train_images,
        annotation_json=dataset_cfg.train_annotations,
        augmentation=None,
        seed=int(getattr(cfg.train, "seed", 1337)),
        max_images=getattr(preview_cfg, "dataset_max_images", None),
    )
    output_dir = project_path(
        getattr(preview_cfg, "output_dir", "data/interim/augmentation_previews")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    max_samples = min(int(getattr(preview_cfg, "num_samples", 8)), len(dataset))
    rng = np.random.default_rng(int(getattr(preview_cfg, "seed", getattr(cfg.train, "seed", 1337))))
    saved: list[Path] = []
    categories = dataset.cat_id_to_name

    method_cfgs = _method_cfgs(cfg)
    for sample_idx in range(max_samples):
        index = int(rng.integers(0, len(dataset)))
        raw = dataset.get_raw_sample(index)
        images = [overlay_instances(raw["image"], raw["instances"], categories)]
        labels = ["original"]
        for name, aug_cfg in method_cfgs:
            if name == "normal":
                augmenter = NormalAugmentation(
                    probability=1.0,
                    horizontal_flip_p=float(getattr(aug_cfg, "horizontal_flip_p", 0.5)),
                    brightness_limit=float(getattr(aug_cfg, "brightness_limit", 0.2)),
                    contrast_limit=float(getattr(aug_cfg, "contrast_limit", 0.2)),
                    use_albumentations=bool(getattr(aug_cfg, "use_albumentations", False)),
                )
            else:
                augmenter = build_augmentation(
                    aug_cfg,
                    dataset.coco,
                    dataset.images,
                    donor_getter=dataset.get_raw_sample,
                    device=str(device),
                )
            aug_sample = augmenter(raw, np.random.default_rng(int(rng.integers(0, 1_000_000_000))))
            images.append(
                overlay_instances(aug_sample["image"], aug_sample["instances"], categories)
            )
            labels.append(name)
        out_path = output_dir / f"preview_{sample_idx:04d}_image_{raw['image_info']['id']}.png"
        save_image_grid(images, labels, out_path, columns=len(images))
        saved.append(out_path)
    return saved


def _method_cfgs(cfg: Any) -> list[tuple[str, Any]]:
    from omegaconf import OmegaConf

    base = OmegaConf.to_container(cfg.augmentation, resolve=True)
    names = ["normal", "simple_copy_paste", "pctnet_copy_paste", "lbm_copy_paste"]
    configs = []
    for name in names:
        item = dict(base)
        item["name"] = name
        item["probability"] = 1.0
        if name in {"pctnet_copy_paste", "lbm_copy_paste"}:
            item.setdefault("harmonizer_backend", "local")
        configs.append((name, OmegaConf.create(item)))
    return configs
