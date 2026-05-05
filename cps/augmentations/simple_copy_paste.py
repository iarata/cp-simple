"""Practical Simple Copy-Paste augmentation for COCO instance segmentation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cps.augmentations.base import CopyPasteConfig
from cps.augmentations.masks import (
    crop_instance,
    mask_to_bbox_xyxy,
    paste_foreground,
    place_crop_on_canvas,
    remove_tiny_instances,
    resize_image,
    resize_mask,
    subtract_occlusion,
)

DonorGetter = Callable[[int], dict[str, Any]]


@dataclass
class SimpleCopyPasteAugmentation:
    """Randomly paste object masks from donor images onto the current image.

    This implementation follows the practical spirit of Simple Copy-Paste:
    object instances are selected independently of semantic context, pasted at
    random valid locations, existing annotations are occlusion-updated, and tiny
    residual masks are removed.
    """

    donor_getter: DonorGetter
    donor_count: int
    category_to_indices: dict[int, list[int]]
    allowed_category_ids: set[int] | None = None
    config: CopyPasteConfig = field(default_factory=CopyPasteConfig)
    name: str = "simple_copy_paste"

    def __call__(self, sample: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
        if self.donor_count <= 0 or float(rng.random()) > self.config.probability:
            out = dict(sample)
            out["augmentation_meta"] = {"method": self.name, "applied": False}
            return out

        out = self._clone_sample(sample)
        paste_count = int(rng.integers(1, max(2, self.config.max_paste_objects + 1)))
        pasted_meta = []
        for _ in range(paste_count):
            donor = self._sample_donor(rng)
            if donor is None:
                continue
            donor_inst = self._sample_instance(donor, rng)
            if donor_inst is None:
                continue
            result = self._paste_one(out, donor, donor_inst, rng)
            if result is not None:
                out, meta = result
                pasted_meta.append(meta)
        out["instances"] = remove_tiny_instances(
            out.get("instances", []), self.config.min_area, self.config.min_bbox_size
        )
        out["augmentation_meta"] = {
            "method": self.name,
            "applied": bool(pasted_meta),
            "pasted_objects": pasted_meta,
        }
        return out

    def _clone_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        out = dict(sample)
        out["image"] = np.asarray(sample["image"], dtype=np.uint8).copy()
        out["instances"] = [
            {**inst, "mask": np.asarray(inst["mask"], dtype=bool).copy()}
            for inst in sample.get("instances", [])
        ]
        return out

    def _sample_donor(self, rng: np.random.Generator) -> dict[str, Any] | None:
        if self.allowed_category_ids:
            candidates = []
            for cat_id in self.allowed_category_ids:
                candidates.extend(self.category_to_indices.get(int(cat_id), []))
            if not candidates:
                return None
            donor_index = int(candidates[int(rng.integers(0, len(candidates)))])
        else:
            donor_index = int(rng.integers(0, self.donor_count))
        return self.donor_getter(donor_index)

    def _sample_instance(
        self, donor: dict[str, Any], rng: np.random.Generator
    ) -> dict[str, Any] | None:
        instances = donor.get("instances", [])
        if self.allowed_category_ids:
            instances = [
                inst for inst in instances if int(inst["category_id"]) in self.allowed_category_ids
            ]
        instances = [
            inst for inst in instances if np.asarray(inst.get("mask"), dtype=bool).sum() > 0
        ]
        if not instances:
            return None
        return instances[int(rng.integers(0, len(instances)))]

    def _paste_one(
        self,
        sample: dict[str, Any],
        donor: dict[str, Any],
        donor_inst: dict[str, Any],
        rng: np.random.Generator,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        target_image = np.asarray(sample["image"], dtype=np.uint8)
        height, width = target_image.shape[:2]
        donor_image = np.asarray(donor["image"], dtype=np.uint8)
        donor_mask = np.asarray(donor_inst["mask"], dtype=bool)
        bbox = mask_to_bbox_xyxy(donor_mask)
        if bbox is None:
            return None
        crop_img, crop_mask = crop_instance(donor_image, donor_mask, bbox)
        if crop_mask.sum() < self.config.min_area:
            return None
        crop_h, crop_w = crop_mask.shape[:2]
        scale = float(rng.uniform(*self.config.paste_scale_jitter))
        if crop_w > 0 and crop_h > 0:
            scale = min(scale, 0.9 * width / max(crop_w, 1), 0.9 * height / max(crop_h, 1))
        new_w = max(1, round(crop_w * max(scale, 0.05)))
        new_h = max(1, round(crop_h * max(scale, 0.05)))
        if new_w < self.config.min_bbox_size or new_h < self.config.min_bbox_size:
            return None
        crop_img = resize_image(crop_img, (new_w, new_h))
        crop_mask = resize_mask(crop_mask, (new_w, new_h))
        if crop_mask.sum() < self.config.min_area:
            return None

        x_max = max(0, width - new_w)
        y_max = max(0, height - new_h)
        xy = (int(rng.integers(0, x_max + 1)), int(rng.integers(0, y_max + 1)))
        fg_canvas, pasted_mask = place_crop_on_canvas(crop_img, crop_mask, (height, width), xy)
        if pasted_mask.sum() < self.config.min_area:
            return None
        before = target_image.copy()
        composite = paste_foreground(target_image, fg_canvas, pasted_mask)
        composite = self.harmonize(composite, before, pasted_mask, rng)

        updated_instances = subtract_occlusion(sample.get("instances", []), pasted_mask)
        new_bbox = mask_to_bbox_xyxy(pasted_mask)
        if new_bbox is None:
            return None
        pasted_instance = {
            "annotation_id": -1,
            "category_id": int(donor_inst["category_id"]),
            "bbox_xyxy": new_bbox,
            "bbox": new_bbox,
            "bbox_mode": "xyxy",
            "area": float(pasted_mask.sum()),
            "iscrowd": 0,
            "mask": pasted_mask,
            "source_image_id": int(donor.get("image_info", {}).get("id", -1)),
            "source_annotation_id": int(donor_inst.get("annotation_id", -1)),
            "is_copy_paste": True,
        }
        updated_instances.append(pasted_instance)
        out = dict(sample)
        out["image"] = composite
        out["instances"] = remove_tiny_instances(
            updated_instances, self.config.min_area, self.config.min_bbox_size
        )
        meta = {
            "category_id": int(donor_inst["category_id"]),
            "source_image_id": int(donor.get("image_info", {}).get("id", -1)),
            "source_annotation_id": int(donor_inst.get("annotation_id", -1)),
            "bbox_xyxy": [float(v) for v in new_bbox],
            "scale": scale,
            "xy": [int(xy[0]), int(xy[1])],
        }
        return out, meta

    def harmonize(
        self,
        composite: np.ndarray,
        background: np.ndarray,
        pasted_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del background, pasted_mask, rng
        return composite
