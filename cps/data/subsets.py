"""Percentage-based nested COCO subset generation."""

from __future__ import annotations

import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from cps.data.coco import (
    annotation_to_instance,
    annotations_by_image,
    load_coco_json,
    save_coco_json,
)
from cps.data.stats import class_distribution, instance_count_per_class
from cps.data.visualization import (
    overlay_instances,
    save_class_frequency_plot,
    save_distribution_comparison_plot,
    save_image_grid,
    save_long_tail_plot,
)
from cps.paths import project_path


def percent_slug(percent: float | int) -> str:
    value = float(percent)
    if value.is_integer():
        return f"pct_{int(value):03d}"
    return "pct_" + str(value).replace(".", "p")


def build_nested_image_order(coco: dict[str, Any], seed: int) -> list[int]:
    """Build a single image-id order whose prefixes approximate full distribution.

    Images are assigned to a primary bucket using their rarest annotated class;
    this protects long-tail classes better than pure random sampling while still
    preserving the original bucket proportions. Prefixes of this order are used
    for all requested subset percentages, so smaller subsets are nested in larger
    subsets by construction.
    """

    images = sorted(coco.get("images", []), key=lambda img: int(img["id"]))
    anns = annotations_by_image(coco)
    counts = instance_count_per_class(coco)
    rng = np.random.default_rng(seed)
    buckets: dict[int, list[int]] = defaultdict(list)
    for image in images:
        image_id = int(image["id"])
        cat_ids = [
            int(ann["category_id"]) for ann in anns.get(image_id, []) if not ann.get("iscrowd", 0)
        ]
        if cat_ids:
            primary = min(cat_ids, key=lambda cat_id: (counts.get(cat_id, 0), cat_id))
        else:
            primary = -1
        buckets[int(primary)].append(image_id)
    for bucket_id, image_ids in buckets.items():
        bucket_rng = np.random.default_rng(seed + int(bucket_id) * 104_729)
        bucket_rng.shuffle(image_ids)
    bucket_ids = sorted(buckets)
    total = len(images)
    proportions = {bucket_id: len(buckets[bucket_id]) / max(total, 1) for bucket_id in bucket_ids}
    selected_counts = dict.fromkeys(bucket_ids, 0)
    tie_break = {bucket_id: float(rng.random()) for bucket_id in bucket_ids}
    order: list[int] = []
    for step in range(1, total + 1):
        best_bucket = None
        best_score = -math.inf
        for bucket_id in bucket_ids:
            if selected_counts[bucket_id] >= len(buckets[bucket_id]):
                continue
            deficit = proportions[bucket_id] * step - selected_counts[bucket_id]
            score = deficit + 1e-9 * tie_break[bucket_id]
            if score > best_score:
                best_score = score
                best_bucket = bucket_id
        if best_bucket is None:
            break
        idx = selected_counts[best_bucket]
        order.append(buckets[best_bucket][idx])
        selected_counts[best_bucket] += 1
    return order


def subset_coco(coco: dict[str, Any], selected_image_ids: set[int]) -> dict[str, Any]:
    selected_images = [
        img for img in coco.get("images", []) if int(img["id"]) in selected_image_ids
    ]
    selected_annotations = [
        ann for ann in coco.get("annotations", []) if int(ann["image_id"]) in selected_image_ids
    ]
    return {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": coco.get("categories", []),
    }


def build_subset_metadata(
    subset: dict[str, Any],
    full: dict[str, Any],
    percent: float,
    seed: int,
) -> dict[str, Any]:
    distribution = class_distribution(subset)
    full_distribution = class_distribution(full)
    return {
        "percentage": float(percent),
        "seed": int(seed),
        "number_of_images": len(subset.get("images", [])),
        "number_of_annotations": len(subset.get("annotations", [])),
        "class_distribution": distribution["classes"],
        "instance_count_per_class": distribution["instance_count_per_class"],
        "images_per_class": distribution["images_per_class"],
        "imbalance_statistics": distribution["imbalance"],
        "full_imbalance_statistics": full_distribution["imbalance"],
    }


def materialize_images(
    subset: dict[str, Any],
    source_image_dir: Path,
    output_image_dir: Path,
    mode: str = "symlink",
) -> None:
    if mode == "none":
        return
    output_image_dir.mkdir(parents=True, exist_ok=True)
    for image in subset.get("images", []):
        rel = Path(image["file_name"])
        src = source_image_dir / rel
        dst = output_image_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            continue
        if not src.exists():
            logger.warning("Skipping missing image while materializing subset: {}", src)
            continue
        if mode == "copy":
            shutil.copy2(src, dst)
        elif mode == "symlink":
            try:
                os.symlink(os.path.relpath(src, dst.parent), dst)
            except OSError:
                shutil.copy2(src, dst)
        else:
            raise ValueError("image materialization mode must be one of: none, symlink, copy")


def save_subset_visualizations(
    subset: dict[str, Any],
    full: dict[str, Any],
    image_dir: Path,
    output_dir: Path,
    max_preview_images: int,
    seed: int,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_dist = class_distribution(subset)
    full_dist = class_distribution(full)
    paths = {
        "class_frequency": output_dir / "class_frequency.png",
        "long_tail": output_dir / "long_tail.png",
        "comparison_full": output_dir / "comparison_full_distribution.png",
        "sample_grid": output_dir / "sample_grid.png",
    }
    save_class_frequency_plot(
        subset_dist["classes"], paths["class_frequency"], "Subset class frequency"
    )
    save_long_tail_plot(subset_dist["classes"], paths["long_tail"], "Subset long-tail distribution")
    save_distribution_comparison_plot(
        full_dist["classes"],
        subset_dist["classes"],
        paths["comparison_full"],
        "Subset vs full COCO class distribution",
    )
    _save_subset_sample_grid(subset, image_dir, paths["sample_grid"], max_preview_images, seed)
    return {key: str(value) for key, value in paths.items()}


def _save_subset_sample_grid(
    subset: dict[str, Any],
    image_dir: Path,
    output_path: Path,
    max_preview_images: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    images = list(subset.get("images", []))
    if not images:
        save_image_grid([], [], output_path)
        return
    rng.shuffle(images)
    images = images[: max(0, int(max_preview_images))]
    anns = annotations_by_image(subset)
    categories = {
        int(cat["id"]): cat.get("name", str(cat["id"])) for cat in subset.get("categories", [])
    }
    rendered = []
    labels = []
    for image_info in images:
        path = image_dir / image_info["file_name"]
        if not path.exists():
            continue
        from PIL import Image

        with Image.open(path) as pil_img:
            rgb = pil_img.convert("RGB")
            arr = np.asarray(rgb, dtype=np.uint8)
        height, width = arr.shape[:2]
        instances = []
        for ann in anns.get(int(image_info["id"]), []):
            inst = annotation_to_instance(ann, height, width)
            if inst is not None:
                instances.append(inst)
        rendered.append(overlay_instances(arr, instances, categories))
        labels.append(str(image_info["id"]))
    save_image_grid(rendered, labels, output_path)


def build_coco_subsets(cfg: Any) -> list[dict[str, Any]]:
    subset_cfg = cfg.subset
    annotation_json = project_path(subset_cfg.annotation_json)
    image_dir = project_path(subset_cfg.image_dir)
    output_root = project_path(subset_cfg.output_dir)
    seed = int(subset_cfg.seed)
    percentages = [float(p) for p in list(subset_cfg.percentages)]
    if any(p <= 0 or p > 100 for p in percentages):
        raise ValueError("Subset percentages must be in the interval (0, 100].")
    coco = load_coco_json(annotation_json)
    order = build_nested_image_order(coco, seed)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    full_count = len(order)
    for percent in sorted(percentages):
        subset_count = max(1, round(full_count * percent / 100.0))
        selected_ids = set(order[:subset_count])
        subset = subset_coco(coco, selected_ids)
        slug = percent_slug(percent)
        subset_dir = output_root / f"{slug}_seed_{seed}"
        ann_path = subset_dir / "annotations.json"
        save_coco_json(subset, ann_path)
        image_mode = str(getattr(subset_cfg, "image_mode", "symlink"))
        image_output_dir = subset_dir / "images"
        materialize_images(subset, image_dir, image_output_dir, image_mode)
        metadata = build_subset_metadata(subset, coco, percent, seed)
        viz_paths = save_subset_visualizations(
            subset=subset,
            full=coco,
            image_dir=image_output_dir if image_mode != "none" else image_dir,
            output_dir=subset_dir / "visualizations",
            max_preview_images=int(getattr(subset_cfg, "max_preview_images", 16)),
            seed=seed,
        )
        metadata["annotation_json"] = str(ann_path)
        metadata["image_dir"] = str(image_output_dir if image_mode != "none" else image_dir)
        metadata["visualizations"] = viz_paths
        meta_path = subset_dir / "metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(
            "Wrote {}: {} images, {} annotations",
            ann_path,
            len(subset.get("images", [])),
            len(subset.get("annotations", [])),
        )
        summaries.append(metadata)
    summary_path = output_root / f"summary_seed_{seed}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    return summaries
