"""COCO class distribution and imbalance statistics."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ImbalanceStats:
    num_classes_with_instances: int
    min_instances: int
    max_instances: int
    max_min_ratio: float
    gini: float
    entropy: float
    rare_threshold: float
    common_threshold: float


def gini(counts: list[int]) -> float:
    values = np.asarray([c for c in counts if c >= 0], dtype=np.float64)
    if values.size == 0 or np.sum(values) == 0:
        return 0.0
    values = np.sort(values)
    n = values.size
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * values)) / (n * np.sum(values)))


def entropy(counts: list[int]) -> float:
    values = np.asarray([c for c in counts if c > 0], dtype=np.float64)
    if values.size == 0:
        return 0.0
    probs = values / values.sum()
    return float(-(probs * np.log2(probs)).sum())


def instance_count_per_class(coco: dict[str, Any]) -> dict[int, int]:
    counter: Counter[int] = Counter()
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        counter[int(ann["category_id"])] += 1
    return dict(counter)


def images_per_class(coco: dict[str, Any]) -> dict[int, int]:
    image_sets: dict[int, set[int]] = defaultdict(set)
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        image_sets[int(ann["category_id"])].add(int(ann["image_id"]))
    return {cat_id: len(image_ids) for cat_id, image_ids in image_sets.items()}


def class_distribution(coco: dict[str, Any]) -> dict[str, Any]:
    categories = sorted(coco.get("categories", []), key=lambda c: int(c["id"]))
    instance_counts = instance_count_per_class(coco)
    image_counts = images_per_class(coco)
    rows = []
    for cat in categories:
        cat_id = int(cat["id"])
        rows.append(
            {
                "category_id": cat_id,
                "name": cat.get("name", str(cat_id)),
                "supercategory": cat.get("supercategory", ""),
                "instances": int(instance_counts.get(cat_id, 0)),
                "images": int(image_counts.get(cat_id, 0)),
            }
        )
    counts = [row["instances"] for row in rows if row["instances"] > 0]
    if counts:
        rare_threshold = float(np.quantile(counts, 0.25))
        common_threshold = float(np.quantile(counts, 0.75))
        min_instances = int(min(counts))
        max_instances = int(max(counts))
        max_min_ratio = float(max_instances / max(min_instances, 1))
    else:
        rare_threshold = 0.0
        common_threshold = 0.0
        min_instances = 0
        max_instances = 0
        max_min_ratio = 0.0
    stats = ImbalanceStats(
        num_classes_with_instances=len(counts),
        min_instances=min_instances,
        max_instances=max_instances,
        max_min_ratio=max_min_ratio,
        gini=gini(counts),
        entropy=entropy(counts),
        rare_threshold=rare_threshold,
        common_threshold=common_threshold,
    )
    return {
        "classes": rows,
        "instance_count_per_class": {str(k): int(v) for k, v in instance_counts.items()},
        "images_per_class": {str(k): int(v) for k, v in image_counts.items()},
        "imbalance": asdict(stats),
    }


def underrepresented_classes(coco: dict[str, Any], quantile: float = 0.25) -> set[int]:
    counts = instance_count_per_class(coco)
    nonzero = np.asarray([value for value in counts.values() if value > 0], dtype=np.float64)
    if nonzero.size == 0:
        return set()
    threshold = float(np.quantile(nonzero, quantile))
    return {cat_id for cat_id, count in counts.items() if count <= threshold}
