"""COCO2017-style dataset utilities for instance segmentation."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from cps.augmentations.masks import bbox_xywh_to_xyxy, mask_to_bbox_xyxy
from cps.paths import project_path

Sample = dict[str, Any]
AugmentationFn = Callable[[Sample, np.random.Generator], Sample]


def load_coco_json(path: str | Path) -> dict[str, Any]:
    path = project_path(path)
    if not path.exists():
        raise FileNotFoundError(f"COCO annotation JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise ValueError(f"Invalid COCO file {path}: missing key '{key}'")
    return data


def save_coco_json(data: dict[str, Any], path: str | Path) -> None:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def annotations_by_image(coco: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    anns: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns[int(ann["image_id"])].append(ann)
    return dict(anns)


def category_maps(coco: dict[str, Any]) -> tuple[dict[int, int], dict[int, int], dict[int, str]]:
    categories = sorted(coco.get("categories", []), key=lambda c: int(c["id"]))
    cat_id_to_label = {int(cat["id"]): idx + 1 for idx, cat in enumerate(categories)}
    label_to_cat_id = {label: cat_id for cat_id, label in cat_id_to_label.items()}
    cat_id_to_name = {int(cat["id"]): cat.get("name", str(cat["id"])) for cat in categories}
    return cat_id_to_label, label_to_cat_id, cat_id_to_name


def _polygon_to_mask(polygons: list[list[float]], height: int, width: int) -> np.ndarray:
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    for polygon in polygons:
        if len(polygon) < 6:
            continue
        coords = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon), 2)]
        draw.polygon(coords, outline=1, fill=1)
    return np.asarray(mask_img, dtype=bool)


def segmentation_to_mask(
    segmentation: Any,
    height: int,
    width: int,
    bbox_xywh: list[float] | tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Decode a COCO segmentation to a boolean mask.

    Polygon masks are decoded locally. RLE masks use pycocotools when available;
    if unavailable, a bbox mask is used as a conservative fallback so dry runs can
    still execute without pycocotools.
    """

    if isinstance(segmentation, list):
        return _polygon_to_mask(segmentation, height, width)
    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as mask_utils

            decoded = mask_utils.decode(segmentation)
            if decoded.ndim == 3:
                decoded = decoded.any(axis=2)
            return decoded.astype(bool)
        except Exception:
            pass
    mask = np.zeros((height, width), dtype=bool)
    if bbox_xywh is not None:
        x, y, w, h = [round(v) for v in bbox_xywh]
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(width, x + max(0, w))
        y1 = min(height, y + max(0, h))
        mask[y0:y1, x0:x1] = True
    return mask


def annotation_to_instance(ann: dict[str, Any], height: int, width: int) -> dict[str, Any] | None:
    bbox_xywh = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
    mask = segmentation_to_mask(ann.get("segmentation"), height, width, bbox_xywh)
    if mask.sum() == 0:
        x0, y0, x1, y1 = bbox_xywh_to_xyxy(bbox_xywh)
        if x1 <= x0 or y1 <= y0:
            return None
    bbox = mask_to_bbox_xyxy(mask)
    if bbox is None:
        bbox = bbox_xywh_to_xyxy(bbox_xywh)
    area = (
        float(mask.sum()) if mask.sum() > 0 else float(ann.get("area", bbox_xywh[2] * bbox_xywh[3]))
    )
    return {
        "annotation_id": int(ann.get("id", -1)),
        "category_id": int(ann["category_id"]),
        "bbox_xyxy": [float(v) for v in bbox],
        "bbox": [float(v) for v in bbox],
        "bbox_mode": "xyxy",
        "area": area,
        "iscrowd": int(ann.get("iscrowd", 0)),
        "mask": mask.astype(bool),
        "source_image_id": int(ann.get("image_id", -1)),
    }


def mask_to_coco_segmentation(mask: np.ndarray) -> dict[str, Any] | list[Any]:
    """Encode a boolean mask as COCO compressed RLE when pycocotools is available."""

    mask_uint8 = np.asfortranarray(np.asarray(mask, dtype=np.uint8))
    try:
        from pycocotools import mask as mask_utils

        rle = mask_utils.encode(mask_uint8)
        counts = rle.get("counts")
        if isinstance(counts, bytes):
            rle["counts"] = counts.decode("ascii")
        rle["size"] = [int(v) for v in rle["size"]]
        return rle
    except Exception:
        return []


def instances_to_annotations(
    instances: list[dict[str, Any]],
    image_id: int,
    start_ann_id: int = 1,
) -> list[dict[str, Any]]:
    anns = []
    ann_id = start_ann_id
    for inst in instances:
        mask = np.asarray(inst["mask"], dtype=bool)
        bbox = mask_to_bbox_xyxy(mask)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        anns.append(
            {
                "id": ann_id,
                "image_id": int(image_id),
                "category_id": int(inst["category_id"]),
                "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                "area": float(mask.sum()),
                "iscrowd": int(inst.get("iscrowd", 0)),
                "segmentation": mask_to_coco_segmentation(mask),
            }
        )
        ann_id += 1
    return anns


class COCODataset:
    """Torch-compatible COCO instance segmentation dataset.

    The dataset returns ``(image_tensor, target_dict)``. Labels are mapped to a
    contiguous ``1..K`` range while ``target['category_ids']`` preserves original
    COCO category IDs for metrics.
    """

    def __init__(
        self,
        image_dir: str | Path,
        annotation_json: str | Path,
        augmentation: AugmentationFn | None = None,
        seed: int = 0,
        max_images: int | None = None,
    ) -> None:
        self.image_dir = project_path(image_dir)
        self.annotation_json = project_path(annotation_json)
        self.coco = load_coco_json(self.annotation_json)
        self.images = sorted(self.coco["images"], key=lambda img: int(img["id"]))
        if max_images is not None:
            self.images = self.images[: int(max_images)]
        self.anns_by_image = annotations_by_image(self.coco)
        self.cat_id_to_label, self.label_to_cat_id, self.cat_id_to_name = category_maps(self.coco)
        self.augmentation = augmentation
        self.seed = int(seed)
        self.epoch = 0

    @property
    def num_classes(self) -> int:
        return len(self.cat_id_to_label)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.images)

    def image_path(self, image_info: dict[str, Any]) -> Path:
        file_name = image_info["file_name"]
        path = self.image_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        return path

    def load_image(self, image_info: dict[str, Any]) -> np.ndarray:
        path = self.image_path(image_info)
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    def get_raw_sample(self, index: int) -> Sample:
        image_info = self.images[int(index) % len(self.images)]
        image = self.load_image(image_info)
        height, width = image.shape[:2]
        instances = []
        for ann in self.anns_by_image.get(int(image_info["id"]), []):
            inst = annotation_to_instance(ann, height, width)
            if inst is not None:
                instances.append(inst)
        return {
            "image": image,
            "instances": instances,
            "image_info": dict(image_info),
            "height": height,
            "width": width,
            "augmentation_meta": {"method": "none"},
        }

    def __getitem__(self, index: int):
        sample = self.get_raw_sample(index)
        if self.augmentation is not None:
            rng = np.random.default_rng(self.seed + int(index) + self.epoch * 1_000_003)
            sample = self.augmentation(sample, rng)
        return sample_to_torch(sample, self.cat_id_to_label)


def sample_to_torch(sample: Sample, cat_id_to_label: dict[int, int]):
    import torch

    image = np.asarray(sample["image"], dtype=np.uint8)
    height, width = image.shape[:2]
    image_tensor = torch.as_tensor(image.transpose(2, 0, 1).copy(), dtype=torch.float32) / 255.0
    instances = sample.get("instances", [])
    boxes = []
    labels = []
    category_ids = []
    masks = []
    areas = []
    iscrowd = []
    for inst in instances:
        mask = np.asarray(inst.get("mask"), dtype=bool)
        bbox = mask_to_bbox_xyxy(mask) or inst.get("bbox_xyxy")
        if bbox is None:
            continue
        x0, y0, x1, y1 = [float(v) for v in bbox]
        if x1 <= x0 or y1 <= y0 or mask.sum() <= 0:
            continue
        cat_id = int(inst["category_id"])
        label = cat_id_to_label.get(cat_id)
        if label is None:
            continue
        boxes.append([x0, y0, x1, y1])
        labels.append(label)
        category_ids.append(cat_id)
        masks.append(mask.astype(np.uint8))
        areas.append(float(mask.sum()))
        iscrowd.append(int(inst.get("iscrowd", 0)))
    if boxes:
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)
        category_ids_t = torch.as_tensor(category_ids, dtype=torch.int64)
        masks_t = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.float32)
        areas_t = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd_t = torch.as_tensor(iscrowd, dtype=torch.int64)
    else:
        boxes_t = torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.zeros((0,), dtype=torch.int64)
        category_ids_t = torch.zeros((0,), dtype=torch.int64)
        masks_t = torch.zeros((0, height, width), dtype=torch.float32)
        areas_t = torch.zeros((0,), dtype=torch.float32)
        iscrowd_t = torch.zeros((0,), dtype=torch.int64)
    image_id = int(sample.get("image_info", {}).get("id", 0))
    target = {
        "boxes": boxes_t,
        "labels": labels_t,
        "category_ids": category_ids_t,
        "masks": masks_t,
        "area": areas_t,
        "iscrowd": iscrowd_t,
        "image_id": torch.tensor([image_id], dtype=torch.int64),
        "orig_size": torch.tensor([height, width], dtype=torch.int64),
        "size": torch.tensor([height, width], dtype=torch.int64),
        "augmentation_meta": sample.get("augmentation_meta", {}),
    }
    return image_tensor, target


def collate_fn(batch: list[tuple[Any, dict[str, Any]]]) -> tuple[list[Any], list[dict[str, Any]]]:
    return [item[0] for item in batch], [item[1] for item in batch]
