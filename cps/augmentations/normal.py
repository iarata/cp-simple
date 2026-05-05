"""Normal non-copy-paste augmentations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cps.augmentations.masks import mask_to_bbox_xyxy, remove_tiny_instances


@dataclass
class NormalAugmentation:
    """Lightweight image/mask-safe augmentation with optional Albumentations.

    The fallback path intentionally keeps geometry simple. It uses horizontal
    flips and photometric jitter and therefore has no dependency on compiled
    torchvision operators.
    """

    probability: float = 1.0
    horizontal_flip_p: float = 0.5
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    use_albumentations: bool = True
    min_area: int = 16
    min_bbox_size: int = 2

    name: str = "normal"

    def __post_init__(self) -> None:
        self._albumentations = None
        if self.use_albumentations:
            try:
                import albumentations as A

                self._albumentations = A.Compose(
                    [
                        A.HorizontalFlip(p=self.horizontal_flip_p),
                        A.RandomBrightnessContrast(
                            brightness_limit=self.brightness_limit,
                            contrast_limit=self.contrast_limit,
                            p=0.7,
                        ),
                    ]
                )
            except Exception:
                self._albumentations = None

    def __call__(self, sample: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
        if float(rng.random()) > self.probability:
            out = dict(sample)
            out["augmentation_meta"] = {"method": self.name, "applied": False}
            return out
        if self._albumentations is not None:
            return self._apply_albumentations(sample)
        return self._apply_local(sample, rng)

    def _apply_albumentations(self, sample: dict[str, Any]) -> dict[str, Any]:
        image = np.asarray(sample["image"], dtype=np.uint8)
        masks = [np.asarray(inst["mask"], dtype=np.uint8) for inst in sample.get("instances", [])]
        transformed = self._albumentations(image=image, masks=masks)
        out_instances = []
        for inst, mask in zip(
            sample.get("instances", []), transformed.get("masks", []), strict=False
        ):
            updated = dict(inst)
            updated["mask"] = np.asarray(mask).astype(bool)
            bbox = mask_to_bbox_xyxy(updated["mask"])
            if bbox is not None:
                updated["bbox_xyxy"] = bbox
                updated["bbox"] = bbox
                updated["bbox_mode"] = "xyxy"
                updated["area"] = float(updated["mask"].sum())
                out_instances.append(updated)
        out = dict(sample)
        out["image"] = np.asarray(transformed["image"], dtype=np.uint8)
        out["instances"] = remove_tiny_instances(out_instances, self.min_area, self.min_bbox_size)
        out["augmentation_meta"] = {
            "method": self.name,
            "applied": True,
            "backend": "albumentations",
        }
        return out

    def _apply_local(self, sample: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
        image = np.asarray(sample["image"], dtype=np.float32).copy()
        instances = [dict(inst) for inst in sample.get("instances", [])]
        height, width = image.shape[:2]
        if float(rng.random()) < self.horizontal_flip_p:
            image = image[:, ::-1].copy()
            for inst in instances:
                inst["mask"] = np.asarray(inst["mask"], dtype=bool)[:, ::-1].copy()
        contrast = 1.0 + float(rng.uniform(-self.contrast_limit, self.contrast_limit))
        brightness = float(rng.uniform(-255 * self.brightness_limit, 255 * self.brightness_limit))
        image = np.clip((image - 127.5) * contrast + 127.5 + brightness, 0, 255).astype(np.uint8)
        out_instances = []
        for inst in instances:
            bbox = mask_to_bbox_xyxy(inst["mask"])
            if bbox is None:
                continue
            inst["bbox_xyxy"] = bbox
            inst["bbox"] = bbox
            inst["bbox_mode"] = "xyxy"
            inst["area"] = float(np.asarray(inst["mask"], dtype=bool).sum())
            out_instances.append(inst)
        out = dict(sample)
        out["image"] = image
        out["instances"] = remove_tiny_instances(out_instances, self.min_area, self.min_bbox_size)
        out["height"] = height
        out["width"] = width
        out["augmentation_meta"] = {"method": self.name, "applied": True, "backend": "local"}
        return out
