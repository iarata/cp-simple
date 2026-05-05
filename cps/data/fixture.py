"""Tiny synthetic COCO fixture for smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from cps.paths import project_path

CATEGORIES = [
    {"id": 1, "name": "common_square", "supercategory": "shape"},
    {"id": 2, "name": "medium_circle", "supercategory": "shape"},
    {"id": 3, "name": "rare_triangle", "supercategory": "shape"},
]


def create_tiny_coco_fixture(
    output_dir: str | Path, seed: int = 1337, num_train: int = 12, num_val: int = 4
) -> dict[str, str]:
    root = project_path(output_dir)
    rng = np.random.default_rng(seed)
    train = _make_split(root / "train", rng, num_train, start_image_id=1, start_ann_id=1)
    val = _make_split(root / "val", rng, num_val, start_image_id=10_001, start_ann_id=100_001)
    return {
        "root": str(root),
        "train_images": str(root / "train" / "images"),
        "train_annotations": str(train),
        "val_images": str(root / "val" / "images"),
        "val_annotations": str(val),
    }


def _make_split(
    split_dir: Path,
    rng: np.random.Generator,
    num_images: int,
    start_image_id: int,
    start_ann_id: int,
) -> Path:
    image_dir = split_dir / "images"
    ann_dir = split_dir / "annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    images: list[dict[str, Any]] = []
    anns: list[dict[str, Any]] = []
    ann_id = start_ann_id
    for i in range(num_images):
        image_id = start_image_id + i
        width, height = 160, 128
        base = np.zeros((height, width, 3), dtype=np.uint8)
        base[..., 0] = np.uint8(rng.integers(40, 110))
        base[..., 1] = np.uint8(rng.integers(40, 110))
        base[..., 2] = np.uint8(rng.integers(40, 110))
        image = Image.fromarray(base, mode="RGB")
        draw = ImageDraw.Draw(image)
        objects = [(1, "square")]
        if i % 2 == 0:
            objects.append((2, "circle"))
        if i in {0, max(0, num_images - 1)}:
            objects.append((3, "triangle"))
        for cat_id, shape in objects:
            x = int(rng.integers(8, width - 56))
            y = int(rng.integers(8, height - 48))
            w = int(rng.integers(22, 44))
            h = int(rng.integers(20, 40))
            color = _color(cat_id)
            if shape == "square":
                poly = [x, y, x + w, y, x + w, y + h, x, y + h]
                draw.rectangle([x, y, x + w, y + h], fill=color)
            elif shape == "circle":
                poly = _ellipse_polygon(x, y, w, h)
                draw.ellipse([x, y, x + w, y + h], fill=color)
            else:
                poly = [x + w / 2, y, x + w, y + h, x, y + h]
                draw.polygon([(poly[j], poly[j + 1]) for j in range(0, len(poly), 2)], fill=color)
            anns.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "segmentation": [[float(v) for v in poly]],
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
        file_name = f"{image_id:012d}.jpg"
        image.save(image_dir / file_name, quality=95)
        images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})
    coco = {
        "info": {"description": "Synthetic tiny COCO fixture for CPS smoke tests"},
        "licenses": [],
        "images": images,
        "annotations": anns,
        "categories": CATEGORIES,
    }
    ann_path = ann_dir / "instances.json"
    with ann_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)
    return ann_path


def _color(cat_id: int) -> tuple[int, int, int]:
    return {
        1: (220, 70, 70),
        2: (70, 220, 80),
        3: (80, 120, 235),
    }[cat_id]


def _ellipse_polygon(x: int, y: int, w: int, h: int, points: int = 16) -> list[float]:
    coords = []
    for idx in range(points):
        theta = 2 * np.pi * idx / points
        coords.extend([x + w / 2 + np.cos(theta) * w / 2, y + h / 2 + np.sin(theta) * h / 2])
    return [float(v) for v in coords]
