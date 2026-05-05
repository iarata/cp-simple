"""Augmentation factory."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from cps.augmentations.base import cfg_get, copy_paste_config_from_cfg
from cps.augmentations.lbm import LBMCopyPasteAugmentation
from cps.augmentations.normal import NormalAugmentation
from cps.augmentations.pctnet import PCTNetCopyPasteAugmentation
from cps.augmentations.simple_copy_paste import SimpleCopyPasteAugmentation
from cps.data.stats import underrepresented_classes


def category_to_image_indices(
    coco: dict[str, Any], images: list[dict[str, Any]]
) -> dict[int, list[int]]:
    image_id_to_index = {int(img["id"]): idx for idx, img in enumerate(images)}
    mapping: dict[int, set[int]] = defaultdict(set)
    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        if image_id not in image_id_to_index:
            continue
        mapping[int(ann["category_id"])].add(image_id_to_index[image_id])
    return {cat_id: sorted(indices) for cat_id, indices in mapping.items()}


def _allowed_categories(coco: dict[str, Any], cfg: Any) -> set[int] | None:
    policy = str(cfg_get(cfg, "target_policy", "all"))
    class_ids = {int(x) for x in list(cfg_get(cfg, "class_ids", []) or [])}
    if policy == "all":
        return class_ids or None
    if policy == "underrepresented":
        allowed = underrepresented_classes(coco, float(cfg_get(cfg, "rare_quantile", 0.25)))
        return allowed & class_ids if class_ids else allowed
    if policy in {"class_ids", "class_groups", "custom"}:
        return class_ids
    raise ValueError(
        f"Unknown copy-paste target_policy='{policy}'. Use all, underrepresented, or class_ids."
    )


def build_augmentation(
    cfg: Any,
    coco: dict[str, Any],
    images: list[dict[str, Any]],
    donor_getter: Callable[[int], dict[str, Any]] | None = None,
    device: str | int = "cpu",
):
    name = str(cfg_get(cfg, "name", "normal"))
    if name == "none":
        return None
    if name == "normal":
        return NormalAugmentation(
            probability=float(cfg_get(cfg, "probability", 1.0)),
            horizontal_flip_p=float(cfg_get(cfg, "horizontal_flip_p", 0.5)),
            brightness_limit=float(cfg_get(cfg, "brightness_limit", 0.2)),
            contrast_limit=float(cfg_get(cfg, "contrast_limit", 0.2)),
            use_albumentations=bool(cfg_get(cfg, "use_albumentations", True)),
            min_area=int(cfg_get(cfg, "min_area", 16)),
            min_bbox_size=int(cfg_get(cfg, "min_bbox_size", 2)),
        )
    if donor_getter is None:
        raise ValueError(f"Augmentation '{name}' requires a donor_getter.")
    cp_cfg = copy_paste_config_from_cfg(cfg)
    allowed = _allowed_categories(coco, cfg)
    cat_to_indices = category_to_image_indices(coco, images)
    donor_count = len(images)
    if name == "simple_copy_paste":
        return SimpleCopyPasteAugmentation(
            donor_getter=donor_getter,
            donor_count=donor_count,
            category_to_indices=cat_to_indices,
            allowed_category_ids=allowed,
            config=cp_cfg,
            name="simple_copy_paste",
        )
    if name == "pctnet_copy_paste":
        return PCTNetCopyPasteAugmentation(
            donor_getter=donor_getter,
            donor_count=donor_count,
            category_to_indices=cat_to_indices,
            allowed_category_ids=allowed,
            config=cp_cfg,
            device=device,
        )
    if name == "lbm_copy_paste":
        return LBMCopyPasteAugmentation(
            donor_getter=donor_getter,
            donor_count=donor_count,
            category_to_indices=cat_to_indices,
            allowed_category_ids=allowed,
            config=cp_cfg,
            device=device,
        )
    raise ValueError(f"Unknown augmentation name: {name}")
