"""Percentage-based nested COCO subset generation."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from cps.augmentations import build_augmentation
from cps.augmentations.base import cfg_get
from cps.data.coco import (
    COCODataset,
    annotation_to_instance,
    annotations_by_image,
    instances_to_annotations,
    load_coco_json,
    save_coco_json,
)
from cps.data.stats import class_distribution, instance_count_per_class, underrepresented_classes
from cps.data.visualization import (
    overlay_instances,
    save_class_frequency_plot,
    save_distribution_comparison_plot,
    save_image_grid,
    save_instance_count_comparison_plot,
    save_instance_delta_plot,
    save_long_tail_comparison_plot,
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
    for bucket_index, bucket_id in enumerate(sorted(buckets)):
        image_ids = buckets[bucket_id]
        bucket_rng = np.random.default_rng(seed + bucket_index * 104_729)
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


def save_before_after_distribution_visualizations(
    before: dict[str, Any],
    after: dict[str, Any],
    output_dir: Path,
    method: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_dist = class_distribution(before)
    after_dist = class_distribution(after)
    paths = {
        "long_tail": output_dir / "long_tail.png",
        "comparison_before_after_counts": output_dir / "comparison_before_after_counts.png",
        "comparison_before_after_relative": output_dir / "comparison_before_after_relative.png",
        "instance_delta_before_after": output_dir / "instance_delta_before_after.png",
    }
    save_long_tail_comparison_plot(
        before_dist["classes"],
        after_dist["classes"],
        paths["long_tail"],
        f"{method}: before vs after augmentation long-tail distribution",
        before_label="before augmentation",
        after_label="after augmentation",
    )
    save_instance_count_comparison_plot(
        before_dist["classes"],
        after_dist["classes"],
        paths["comparison_before_after_counts"],
        f"{method}: before vs after augmentation instance counts",
        before_label="before augmentation",
        after_label="after augmentation",
    )
    save_distribution_comparison_plot(
        before_dist["classes"],
        after_dist["classes"],
        paths["comparison_before_after_relative"],
        f"{method}: before vs after augmentation relative distribution",
    )
    save_instance_delta_plot(
        before_dist["classes"],
        after_dist["classes"],
        paths["instance_delta_before_after"],
        f"{method}: instance count changes after augmentation",
    )
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


COPY_PASTE_METHODS = ("simple_copy_paste", "pctnet_copy_paste", "lbm_copy_paste")

_PREMADE_WORKER_DATASET: COCODataset | None = None
_PREMADE_WORKER_AUGMENTATION: Any | None = None
_PREMADE_WORKER_SELECTED_IDS: set[int] = set()
_PREMADE_WORKER_IMAGE_DIR: Path | None = None
_PREMADE_WORKER_IMAGE_MODE = "symlink"
_PREMADE_WORKER_APPEND_AUGMENTED = True
_PREMADE_WORKER_SEED = 0
_PREMADE_WORKER_SEED_OFFSET = 1_000_003


def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    return dict(OmegaConf.to_container(cfg, resolve=True) or {})


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if overwrite and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _materialize_one_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if not src.exists():
        logger.warning("Skipping missing image while materializing premade subset: {}", src)
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        try:
            os.symlink(os.path.relpath(src, dst.parent), dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError("Premade image materialization mode must be one of: symlink, copy")


def _premade_num_workers(premade_cfg: Any, method_cfg: dict[str, Any] | None = None) -> int:
    configured = int(cfg_get(premade_cfg, "num_workers", 0) or 0)
    if configured < 0:
        raise ValueError("subset.premade.num_workers must be >= 0.")
    if configured > 0:
        return configured
    if method_cfg and str(method_cfg.get("harmonizer_backend", "local")) == "libcom":
        if str(method_cfg.get("name", "")) == "lbm_copy_paste":
            return _premade_lbm_libcom_auto_workers(premade_cfg)
        return 1
    return max(1, min(os.cpu_count() or 1, 8))


def _premade_lbm_libcom_auto_workers(premade_cfg: Any) -> int:
    device = str(cfg_get(premade_cfg, "device", "cpu"))
    if not _is_cuda_device(device):
        return 1

    max_workers = max(1, int(cfg_get(premade_cfg, "lbm_auto_max_workers", 3) or 1))
    worker_vram_gb = max(1.0, float(cfg_get(premade_cfg, "lbm_worker_vram_gb", 9.0) or 9.0))
    reserve_gb = max(0.0, float(cfg_get(premade_cfg, "lbm_vram_reserve_gb", 4.0) or 4.0))
    total_vram_gb = _cuda_total_vram_gb(device)
    if total_vram_gb is None:
        logger.warning(
            "Could not detect CUDA VRAM for LBM premade auto workers; using 1 worker. "
            "Set subset.premade.num_workers explicitly to override."
        )
        return 1

    vram_workers = int(max(1.0, math.floor((total_vram_gb - reserve_gb) / worker_vram_gb)))
    cpu_workers = max(1, os.cpu_count() or 1)
    workers = max(1, min(max_workers, vram_workers, cpu_workers))
    logger.info(
        "Auto-selected {} LBM libcom worker(s) for {:.1f} GiB CUDA VRAM "
        "({:.1f} GiB/worker, {:.1f} GiB reserve, max {}).",
        workers,
        total_vram_gb,
        worker_vram_gb,
        reserve_gb,
        max_workers,
    )
    return workers


def _is_cuda_device(device: str) -> bool:
    value = str(device).lower()
    return value == "cuda" or value.startswith("cuda:") or value.isdigit()


def _cuda_total_vram_gb(device: str) -> float | None:
    selector = _nvidia_smi_device_selector(device)
    cmd = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
    if selector is not None:
        cmd.extend(["-i", selector])
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=5)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            return float(stripped.split()[0]) / 1024.0
        except ValueError:
            return None
    return None


def _nvidia_smi_device_selector(device: str) -> str | None:
    value = str(device).lower()
    if value == "cuda":
        cuda_index = 0
    elif value.startswith("cuda:"):
        _, _, suffix = value.partition(":")
        if not suffix:
            cuda_index = 0
        elif suffix.isdigit():
            cuda_index = int(suffix)
        else:
            return suffix
    elif value.isdigit():
        cuda_index = int(value)
    else:
        return None

    visible_devices = [
        item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    ]
    visible_devices = [item for item in visible_devices if item]
    if visible_devices and cuda_index < len(visible_devices):
        return visible_devices[cuda_index]
    return str(cuda_index)


def _premade_worker_chunksize(premade_cfg: Any) -> int:
    return max(1, int(cfg_get(premade_cfg, "worker_chunksize", 16) or 1))


def _materialize_images_for_premade(
    subset: dict[str, Any],
    source_image_dir: Path,
    output_image_dir: Path,
    mode: str,
    num_workers: int,
) -> None:
    if num_workers <= 1:
        materialize_images(subset, source_image_dir, output_image_dir, mode)
        return
    output_image_dir.mkdir(parents=True, exist_ok=True)
    images = list(subset.get("images", []))
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for image in images:
            rel = Path(image["file_name"])
            futures.append(
                executor.submit(
                    _materialize_one_image, source_image_dir / rel, output_image_dir / rel, mode
                )
            )
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="premade none materialize",
            leave=False,
        ):
            future.result()


def _write_augmented_image(image: np.ndarray, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    suffix = dst.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        pil_image.save(dst, quality=95)
    else:
        pil_image.save(dst)


def _augmented_file_name(rel: Path, method: str) -> str:
    suffix = rel.suffix or ".jpg"
    stem = rel.name[: -len(rel.suffix)] if rel.suffix else rel.name
    parent = rel.parent if str(rel.parent) != "." else Path()
    return str(Path("augmented") / parent / f"{stem}__{method}{suffix}")


def _renumber_original_annotations(
    annotations: list[dict[str, Any]],
    start_ann_id: int,
    image_id: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    next_ann_id = int(start_ann_id)
    out = []
    for ann in annotations:
        updated = deepcopy(ann)
        updated["id"] = next_ann_id
        if image_id is not None:
            updated["image_id"] = int(image_id)
        next_ann_id += 1
        out.append(updated)
    return out, next_ann_id


def _premade_methods(premade_cfg: Any) -> list[str]:
    methods = [
        str(name) for name in list(cfg_get(premade_cfg, "methods", ["none", *COPY_PASTE_METHODS]))
    ]
    unknown = sorted(set(methods) - {"none", *COPY_PASTE_METHODS})
    if unknown:
        raise ValueError(f"Unknown premade subset method(s): {unknown}")
    return methods


def _premade_output_subdir(premade_cfg: Any) -> str:
    return str(cfg_get(premade_cfg, "output_subdir", "premade"))


def _percent_value_slug(value: float | int) -> str:
    value = float(value)
    if value.is_integer():
        return f"{int(value):03d}"
    return str(value).replace(".", "p")


def _premade_target_slug(premade_cfg: Any) -> str:
    policy = str(cfg_get(premade_cfg, "target_images", "all")).lower()
    if policy == "all":
        return "all_images"
    if policy in {"random", "random_percent", "percentage", "percent"}:
        return (
            f"random_{_percent_value_slug(float(cfg_get(premade_cfg, 'random_percent', 100.0)))}pct"
        )
    if policy in {"underrepresented", "imbalanced", "rare"}:
        class_ids = [int(x) for x in list(cfg_get(premade_cfg, "class_ids", []) or [])]
        if class_ids:
            return "underrepresented_classes_" + "_".join(str(x) for x in sorted(class_ids))
        quantile = float(cfg_get(premade_cfg, "rare_quantile", 0.25))
        return f"underrepresented_q{_percent_value_slug(quantile * 100)}"
    raise ValueError(
        "subset.premade.target_images must be one of: all, random_percent, underrepresented"
    )


def _premade_variant_dir(subset_dir: Path, premade_cfg: Any, variant: str) -> Path:
    variant_path = Path(variant)
    if len(variant_path.parts) > 1:
        return subset_dir / _premade_output_subdir(premade_cfg) / variant_path
    return (
        subset_dir
        / _premade_output_subdir(premade_cfg)
        / _premade_target_slug(premade_cfg)
        / variant
    )


def _legacy_premade_variant_dir(subset_dir: Path, premade_cfg: Any, variant: str) -> Path:
    return subset_dir / _premade_output_subdir(premade_cfg) / variant


def _image_category_lookup(coco: dict[str, Any]) -> dict[int, set[int]]:
    lookup: dict[int, set[int]] = defaultdict(set)
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        lookup[int(ann["image_id"])].add(int(ann["category_id"]))
    return dict(lookup)


def _selected_premade_image_ids(
    subset: dict[str, Any],
    premade_cfg: Any,
    seed: int,
) -> tuple[set[int], dict[str, Any]]:
    policy = str(cfg_get(premade_cfg, "target_images", "all")).lower()
    image_ids = [int(img["id"]) for img in subset.get("images", [])]
    if policy == "all":
        return set(image_ids), {"target_images": "all"}
    if policy in {"random", "random_percent", "percentage", "percent"}:
        percent = float(cfg_get(premade_cfg, "random_percent", 100.0))
        if percent <= 0 or percent > 100:
            raise ValueError("subset.premade.random_percent must be in the interval (0, 100].")
        rng = np.random.default_rng(
            seed + int(cfg_get(premade_cfg, "selection_seed_offset", 911_503))
        )
        shuffled = list(image_ids)
        rng.shuffle(shuffled)
        count = max(1, round(len(shuffled) * percent / 100.0)) if shuffled else 0
        return set(shuffled[:count]), {
            "target_images": "random_percent",
            "random_percent": percent,
        }
    if policy in {"underrepresented", "imbalanced", "rare"}:
        class_ids = {int(x) for x in list(cfg_get(premade_cfg, "class_ids", []) or [])}
        if not class_ids:
            class_ids = underrepresented_classes(
                subset, float(cfg_get(premade_cfg, "rare_quantile", 0.25))
            )
        image_to_categories = _image_category_lookup(subset)
        selected = {
            image_id
            for image_id in image_ids
            if image_to_categories.get(image_id, set()) & class_ids
        }
        return selected, {
            "target_images": "underrepresented",
            "class_ids": sorted(class_ids),
            "rare_quantile": float(cfg_get(premade_cfg, "rare_quantile", 0.25)),
        }
    raise ValueError(
        "subset.premade.target_images must be one of: all, random_percent, underrepresented"
    )


def _copy_paste_cfg_for_premade(method: str, premade_cfg: Any) -> dict[str, Any]:
    cp_cfg = {
        "name": method,
        "probability": 1.0,
        "max_paste_objects": 3,
        "target_policy": "all",
        "class_ids": [],
        "rare_quantile": float(cfg_get(premade_cfg, "rare_quantile", 0.25)),
        "min_area": 16,
        "min_bbox_size": 2,
        "paste_scale_jitter": [0.75, 1.25],
        "max_placement_attempts": 20,
        "harmonizer_backend": "local",
        "feather_sigma": 2.0,
        "lbm_steps": 4,
        "lbm_resolution": 1024,
    }
    configured_cp = cfg_get(premade_cfg, "copy_paste", None)
    cp_cfg.update(_cfg_to_dict(configured_cp))
    cp_cfg["name"] = method

    class_ids = list(cfg_get(premade_cfg, "class_ids", []) or [])
    if class_ids and not list(cp_cfg.get("class_ids", []) or []):
        cp_cfg["class_ids"] = [int(x) for x in class_ids]

    target_images = str(cfg_get(premade_cfg, "target_images", "all")).lower()
    if (
        target_images in {"underrepresented", "imbalanced", "rare"}
        and str(cp_cfg.get("target_policy", "all")) == "all"
    ):
        cp_cfg["target_policy"] = "underrepresented"
    return cp_cfg


def _balance_cfg_for_premade(premade_cfg: Any) -> Any:
    return cfg_get(premade_cfg, "balance", None)


def _balance_enabled_for_method(method: str, premade_cfg: Any) -> bool:
    balance_cfg = _balance_cfg_for_premade(premade_cfg)
    if balance_cfg is None or not bool(cfg_get(balance_cfg, "enabled", False)):
        return False
    methods = {str(name) for name in list(cfg_get(balance_cfg, "methods", COPY_PASTE_METHODS))}
    return method in methods


def _annotation_counts_by_image(
    coco: dict[str, Any],
) -> tuple[dict[int, Counter[int]], dict[int, list[int]], Counter[int]]:
    image_counts: dict[int, Counter[int]] = defaultdict(Counter)
    category_to_images: dict[int, set[int]] = defaultdict(set)
    pool_counts: Counter[int] = Counter()
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])
        image_counts[image_id][category_id] += 1
        category_to_images[category_id].add(image_id)
        pool_counts[category_id] += 1
    return (
        dict(image_counts),
        {cat_id: sorted(image_ids) for cat_id, image_ids in category_to_images.items()},
        pool_counts,
    )


def _balance_target_instances(pool_counts: Counter[int], balance_cfg: Any) -> int:
    positive_counts = [int(count) for count in pool_counts.values() if int(count) > 0]
    if not positive_counts:
        return 0

    explicit = cfg_get(balance_cfg, "target_instances_per_class", None)
    strategy = str(cfg_get(balance_cfg, "target_strategy", "min_count_multiplier")).lower()
    if explicit is not None:
        target = float(explicit)
    elif strategy in {"fixed", "target", "manual"}:
        raise ValueError(
            "subset.premade.balance.target_instances_per_class must be set when "
            "target_strategy=fixed."
        )
    elif strategy in {"quantile", "percentile"}:
        quantile = float(cfg_get(balance_cfg, "target_quantile", 0.25))
        if quantile <= 0 or quantile > 1:
            raise ValueError("subset.premade.balance.target_quantile must be in (0, 1].")
        target = float(np.quantile(np.asarray(positive_counts, dtype=np.float64), quantile))
    elif strategy in {"min_count", "min_count_multiplier", "minimum"}:
        multiplier = float(cfg_get(balance_cfg, "min_count_multiplier", 1.0))
        if multiplier <= 0:
            raise ValueError("subset.premade.balance.min_count_multiplier must be > 0.")
        target = float(min(positive_counts)) * multiplier
    else:
        raise ValueError(
            "subset.premade.balance.target_strategy must be one of: "
            "fixed, min_count_multiplier, quantile."
        )

    min_target = cfg_get(balance_cfg, "min_target_instances", None)
    max_target = cfg_get(balance_cfg, "max_target_instances", None)
    if min_target is not None:
        target = max(target, float(min_target))
    if max_target is not None:
        target = min(target, float(max_target))
    return max(1, round(target))


def _filter_coco_to_image_ids(coco: dict[str, Any], selected_image_ids: set[int]) -> dict[str, Any]:
    return {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": [
            image for image in coco.get("images", []) if int(image["id"]) in selected_image_ids
        ],
        "annotations": [
            ann for ann in coco.get("annotations", []) if int(ann["image_id"]) in selected_image_ids
        ],
        "categories": coco.get("categories", []),
    }


def _top_class_counts(counts: Counter[int], limit: int = 10) -> list[dict[str, int]]:
    return [
        {"category_id": int(category_id), "instances": int(instances)}
        for category_id, instances in counts.most_common(limit)
    ]


def _dominant_balance_categories(pool_counts: Counter[int], balance_cfg: Any) -> set[int]:
    configured = {
        int(cat_id) for cat_id in list(cfg_get(balance_cfg, "dominant_class_ids", []) or [])
    }
    if configured:
        return configured & set(pool_counts)
    top_k = int(cfg_get(balance_cfg, "dominant_top_k", 1) or 0)
    if top_k <= 0:
        return set()
    return {int(category_id) for category_id, _ in pool_counts.most_common(top_k)}


def _balance_premade_subset(
    premade: dict[str, Any],
    premade_cfg: Any,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    balance_cfg = _balance_cfg_for_premade(premade_cfg)
    image_counts, category_to_images, pool_counts = _annotation_counts_by_image(premade)
    if not pool_counts:
        return premade, {"enabled": False, "reason": "no_non_crowd_annotations"}

    target_instances = _balance_target_instances(pool_counts, balance_cfg)
    per_class_targets = {
        int(category_id): min(int(count), target_instances)
        for category_id, count in pool_counts.items()
    }
    dominant_categories = _dominant_balance_categories(pool_counts, balance_cfg)
    dominant_overshoot_ratio = float(cfg_get(balance_cfg, "dominant_max_overshoot_ratio", 1.5))
    if dominant_overshoot_ratio < 1.0:
        raise ValueError("subset.premade.balance.dominant_max_overshoot_ratio must be >= 1.0.")
    dominant_max_counts = {
        category_id: max(
            per_class_targets[category_id],
            math.ceil(per_class_targets[category_id] * dominant_overshoot_ratio),
        )
        for category_id in dominant_categories
    }
    overrepresented = {
        category_id
        for category_id, count in pool_counts.items()
        if int(count) > per_class_targets[category_id]
    }

    image_ids = [int(image["id"]) for image in premade.get("images", [])]
    rng = np.random.default_rng(seed + int(cfg_get(balance_cfg, "seed_offset", 271_828)))
    tie_break = {image_id: float(rng.random()) for image_id in image_ids}
    selected_image_ids: set[int] = set()
    current_counts: Counter[int] = Counter()

    def sort_key(category_id: int, image_id: int) -> tuple[float, int, int, float]:
        counts = image_counts.get(image_id, Counter())
        dominant_load = sum(
            int(count)
            for cat_id, count in counts.items()
            if cat_id in dominant_categories and cat_id != category_id
        )
        category_count = int(counts.get(category_id, 0))
        category_order = category_count if category_id in overrepresented else -category_count
        return (
            dominant_load,
            category_order,
            sum(int(count) for count in counts.values()),
            tie_break.get(image_id, 0.0),
        )

    for category_id in sorted(pool_counts, key=lambda cat_id: (pool_counts[cat_id], cat_id)):
        target = per_class_targets[category_id]
        candidates = sorted(
            category_to_images[category_id], key=lambda img_id: sort_key(category_id, img_id)
        )

        for image_id in candidates:
            if current_counts[category_id] >= target:
                break
            if image_id in selected_image_ids:
                continue
            counts = image_counts.get(image_id, Counter())
            if (
                category_id in dominant_categories
                and current_counts[category_id] + counts.get(category_id, 0)
                > dominant_max_counts[category_id]
            ):
                continue
            selected_image_ids.add(image_id)
            current_counts.update(counts)

    balanced = _filter_coco_to_image_ids(premade, selected_image_ids)
    selected_counts = instance_count_per_class(balanced)
    selected_counter = Counter({int(k): int(v) for k, v in selected_counts.items()})
    reached_targets = sum(
        1
        for category_id, target in per_class_targets.items()
        if selected_counter[category_id] >= target
    )
    meta = {
        "enabled": True,
        "target_strategy": str(cfg_get(balance_cfg, "target_strategy", "min_count_multiplier")),
        "target_instances_per_class": int(target_instances),
        "dominant_class_ids": sorted(dominant_categories),
        "dominant_max_overshoot_ratio": float(dominant_overshoot_ratio),
        "input_images": len(premade.get("images", [])),
        "input_annotations": len(premade.get("annotations", [])),
        "output_images": len(balanced.get("images", [])),
        "output_annotations": len(balanced.get("annotations", [])),
        "classes_reaching_target": int(reached_targets),
        "classes_with_instances": len(pool_counts),
        "input_top_classes": _top_class_counts(pool_counts),
        "output_top_classes": _top_class_counts(selected_counter),
    }
    return balanced, meta


def _init_premade_worker(
    source_image_dir: str,
    subset_ann_path: str,
    method_cfg: dict[str, Any],
    selected_ids: list[int],
    image_dir: str,
    image_mode: str,
    append_augmented: bool,
    seed: int,
    seed_offset: int,
    device: str,
) -> None:
    global _PREMADE_WORKER_AUGMENTATION
    global _PREMADE_WORKER_APPEND_AUGMENTED
    global _PREMADE_WORKER_DATASET
    global _PREMADE_WORKER_IMAGE_DIR
    global _PREMADE_WORKER_IMAGE_MODE
    global _PREMADE_WORKER_SEED
    global _PREMADE_WORKER_SEED_OFFSET
    global _PREMADE_WORKER_SELECTED_IDS

    dataset = COCODataset(
        image_dir=source_image_dir,
        annotation_json=subset_ann_path,
        augmentation=None,
        seed=seed,
    )
    augmentation = build_augmentation(
        method_cfg,
        dataset.coco,
        dataset.images,
        donor_getter=dataset.get_raw_sample,
        device=device,
    )
    if augmentation is None:
        raise ValueError(f"Premade copy-paste method unexpectedly disabled: {method_cfg['name']}")

    _PREMADE_WORKER_DATASET = dataset
    _PREMADE_WORKER_AUGMENTATION = augmentation
    _PREMADE_WORKER_SELECTED_IDS = {int(image_id) for image_id in selected_ids}
    _PREMADE_WORKER_IMAGE_DIR = Path(image_dir)
    _PREMADE_WORKER_IMAGE_MODE = image_mode
    _PREMADE_WORKER_APPEND_AUGMENTED = bool(append_augmented)
    _PREMADE_WORKER_SEED = int(seed)
    _PREMADE_WORKER_SEED_OFFSET = int(seed_offset)


def _process_premade_image_worker(index: int) -> dict[str, Any]:
    if _PREMADE_WORKER_DATASET is None or _PREMADE_WORKER_AUGMENTATION is None:
        raise RuntimeError("Premade worker was not initialized.")
    if _PREMADE_WORKER_IMAGE_DIR is None:
        raise RuntimeError("Premade worker image directory was not initialized.")
    return _process_premade_image_index(
        index=index,
        dataset=_PREMADE_WORKER_DATASET,
        augmentation=_PREMADE_WORKER_AUGMENTATION,
        selected_ids=_PREMADE_WORKER_SELECTED_IDS,
        image_dir=_PREMADE_WORKER_IMAGE_DIR,
        image_mode=_PREMADE_WORKER_IMAGE_MODE,
        append_augmented=_PREMADE_WORKER_APPEND_AUGMENTED,
        seed=_PREMADE_WORKER_SEED,
        seed_offset=_PREMADE_WORKER_SEED_OFFSET,
    )


def _process_premade_image_index(
    index: int,
    dataset: COCODataset,
    augmentation: Any,
    selected_ids: set[int],
    image_dir: Path,
    image_mode: str,
    append_augmented: bool,
    seed: int,
    seed_offset: int,
) -> dict[str, Any]:
    image_info = dataset.images[int(index)]
    image_id = int(image_info["id"])
    rel = Path(image_info["file_name"])
    src = dataset.image_path(image_info)
    dst = image_dir / rel

    if image_id not in selected_ids:
        _materialize_one_image(src, dst, image_mode)
        return {
            "index": int(index),
            "image_info": dict(image_info),
            "annotations": [deepcopy(ann) for ann in dataset.anns_by_image.get(image_id, [])],
            "applied": False,
        }

    if append_augmented:
        _materialize_one_image(src, dst, image_mode)

    raw = dataset.get_raw_sample(index)
    rng = np.random.default_rng(seed + seed_offset + int(index))
    augmented = augmentation(raw, rng)
    if bool(augmented.get("augmentation_meta", {}).get("applied", False)):
        augmented_info = dict(image_info)
        if append_augmented:
            augmented_info["file_name"] = _augmented_file_name(rel, str(augmentation.name))
            augmented_dst = image_dir / augmented_info["file_name"]
        else:
            augmented_dst = dst
        _write_augmented_image(augmented["image"], augmented_dst)
        annotations = instances_to_annotations(
            augmented.get("instances", []), image_id, start_ann_id=1
        )
        result = {
            "index": int(index),
            "image_info": dict(image_info),
            "annotations": annotations,
            "applied": True,
        }
        if append_augmented:
            result["original_annotations"] = [
                deepcopy(ann) for ann in dataset.anns_by_image.get(image_id, [])
            ]
            result["augmented_image_info"] = augmented_info
            result["augmented_annotations"] = annotations
        return result

    if not append_augmented:
        _materialize_one_image(src, dst, image_mode)
    return {
        "index": int(index),
        "image_info": dict(image_info),
        "annotations": [deepcopy(ann) for ann in dataset.anns_by_image.get(image_id, [])],
        "applied": False,
    }


def _process_premade_images(
    dataset: COCODataset,
    method_cfg: dict[str, Any],
    selected_ids: set[int],
    image_dir: Path,
    image_mode: str,
    append_augmented: bool,
    seed: int,
    seed_offset: int,
    source_image_dir: Path,
    subset_ann_path: Path,
    premade_cfg: Any,
) -> tuple[list[dict[str, Any]], int]:
    num_workers = _premade_num_workers(premade_cfg, method_cfg)
    total = len(dataset.images)
    logger.info("Building premade {} with {} worker(s).", method_cfg["name"], num_workers)
    if num_workers <= 1:
        augmentation = build_augmentation(
            method_cfg,
            dataset.coco,
            dataset.images,
            donor_getter=dataset.get_raw_sample,
            device=str(cfg_get(premade_cfg, "device", "cpu")),
        )
        if augmentation is None:
            raise ValueError(
                f"Premade copy-paste method unexpectedly disabled: {method_cfg['name']}"
            )
        results = [
            _process_premade_image_index(
                index=index,
                dataset=dataset,
                augmentation=augmentation,
                selected_ids=selected_ids,
                image_dir=image_dir,
                image_mode=image_mode,
                append_augmented=append_augmented,
                seed=seed,
                seed_offset=seed_offset,
            )
            for index in tqdm(range(total), desc=f"premade {method_cfg['name']}", leave=False)
        ]
        return results, sum(1 for result in results if bool(result["applied"]))

    results: list[dict[str, Any] | None] = [None] * total
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_premade_worker,
        initargs=(
            str(source_image_dir),
            str(subset_ann_path),
            method_cfg,
            sorted(int(image_id) for image_id in selected_ids),
            str(image_dir),
            image_mode,
            bool(append_augmented),
            int(seed),
            int(seed_offset),
            str(cfg_get(premade_cfg, "device", "cpu")),
        ),
    ) as executor:
        for result in tqdm(
            executor.map(
                _process_premade_image_worker,
                range(total),
                chunksize=_premade_worker_chunksize(premade_cfg),
            ),
            total=total,
            desc=f"premade {method_cfg['name']}",
            leave=False,
        ):
            results[int(result["index"])] = result

    ordered = [result for result in results if result is not None]
    if len(ordered) != total:
        raise RuntimeError(f"Premade {method_cfg['name']} produced incomplete worker results.")
    return ordered, sum(1 for result in ordered if bool(result["applied"]))


def _save_premade_metadata(
    premade: dict[str, Any],
    before: dict[str, Any] | None,
    full: dict[str, Any],
    percent: float,
    seed: int,
    ann_path: Path,
    image_dir: Path,
    variant_dir: Path,
    method: str,
    selection_meta: dict[str, Any],
    selected_count: int,
    save_visualizations: bool,
    max_preview_images: int,
) -> dict[str, Any]:
    metadata = build_subset_metadata(premade, full, percent, seed)
    metadata["annotation_json"] = str(ann_path)
    metadata["image_dir"] = str(image_dir)
    metadata["premade"] = {
        "method": method,
        "variant": variant_dir.name,
        "target_slug": variant_dir.parent.name,
        "selected_images": int(selected_count),
        **selection_meta,
    }
    if save_visualizations:
        visualizations = save_subset_visualizations(
            subset=premade,
            full=full,
            image_dir=image_dir,
            output_dir=variant_dir / "visualizations",
            max_preview_images=max_preview_images,
            seed=seed,
        )
        if before is not None:
            visualizations.update(
                save_before_after_distribution_visualizations(
                    before=before,
                    after=premade,
                    output_dir=variant_dir / "visualizations",
                    method=method,
                )
            )
        metadata["visualizations"] = visualizations
    with (variant_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def _build_none_premade_subset(
    subset: dict[str, Any],
    full: dict[str, Any],
    subset_dir: Path,
    source_image_dir: Path,
    percent: float,
    seed: int,
    premade_cfg: Any,
) -> dict[str, Any]:
    variant = "none"
    variant_dir = _premade_variant_dir(subset_dir, premade_cfg, variant)
    _prepare_output_dir(variant_dir, bool(cfg_get(premade_cfg, "overwrite", True)))
    image_dir = variant_dir / "images"
    image_mode = str(cfg_get(premade_cfg, "image_mode", "symlink"))
    if image_mode == "none":
        raise ValueError("subset.premade.image_mode must be symlink or copy for premade datasets.")
    # Match the copy-paste variants: when ``append_augmented=False`` and a
    # selection policy (e.g. ``target_images=underrepresented``) is active, the
    # "none" baseline should be the SAME set of base images the copy-paste
    # variants train on, just without paste. Otherwise epochs aren't
    # comparable across variants (copy-paste = 4.3k underrepresented images,
    # none = full 29.6k -> ~7x more steps per epoch). With ``append_augmented``
    # left at its True default the full base subset is written, preserving the
    # legacy meaning of "premade/none = the 25%-COCO sample".
    append_augmented = bool(cfg_get(premade_cfg, "append_augmented", True))
    selected_ids, selection_meta = _selected_premade_image_ids(subset, premade_cfg, seed)
    if append_augmented:
        emitted = subset
        selection_meta = {**selection_meta, "filtered_to_selected": False}
    else:
        emitted = {
            "info": subset.get("info", {}),
            "licenses": subset.get("licenses", []),
            "images": [img for img in subset.get("images", []) if int(img["id"]) in selected_ids],
            "annotations": [
                ann
                for ann in subset.get("annotations", [])
                if int(ann["image_id"]) in selected_ids
            ],
            "categories": subset.get("categories", []),
        }
        selection_meta = {
            **selection_meta,
            "filtered_to_selected": True,
            "base_images": len(subset.get("images", [])),
            "base_annotations": len(subset.get("annotations", [])),
            "output_images": len(emitted["images"]),
            "output_annotations": len(emitted["annotations"]),
        }
    num_workers = _premade_num_workers(premade_cfg)
    logger.info(
        "Building premade none with {} worker(s) ({} images).",
        num_workers,
        len(emitted.get("images", [])),
    )
    _materialize_images_for_premade(
        subset=emitted,
        source_image_dir=source_image_dir,
        output_image_dir=image_dir,
        mode=image_mode,
        num_workers=num_workers,
    )
    ann_path = variant_dir / "annotations.json"
    save_coco_json(emitted, ann_path)
    metadata = _save_premade_metadata(
        premade=emitted,
        before=None,
        full=full,
        percent=percent,
        seed=seed,
        ann_path=ann_path,
        image_dir=image_dir,
        variant_dir=variant_dir,
        method=variant,
        selection_meta=selection_meta,
        selected_count=len(selected_ids),
        save_visualizations=bool(cfg_get(premade_cfg, "save_visualizations", True)),
        max_preview_images=int(cfg_get(premade_cfg, "max_preview_images", 16)),
    )
    logger.info("Wrote premade subset {}: no augmentation", ann_path)
    return metadata


def _build_copy_paste_premade_subset(
    subset_ann_path: Path,
    subset: dict[str, Any],
    full: dict[str, Any],
    subset_dir: Path,
    source_image_dir: Path,
    percent: float,
    seed: int,
    premade_cfg: Any,
    method: str,
) -> dict[str, Any]:
    variant_dir = _premade_variant_dir(subset_dir, premade_cfg, method)
    _prepare_output_dir(variant_dir, bool(cfg_get(premade_cfg, "overwrite", True)))
    image_dir = variant_dir / "images"
    image_mode = str(cfg_get(premade_cfg, "image_mode", "symlink"))
    if image_mode == "none":
        raise ValueError("subset.premade.image_mode must be symlink or copy for premade datasets.")

    selected_ids, selection_meta = _selected_premade_image_ids(subset, premade_cfg, seed)
    method_cfg = _copy_paste_cfg_for_premade(method, premade_cfg)
    append_augmented = bool(cfg_get(premade_cfg, "append_augmented", True))
    dataset = COCODataset(
        image_dir=source_image_dir,
        annotation_json=subset_ann_path,
        augmentation=None,
        seed=seed,
    )
    worker_results, applied_count = _process_premade_images(
        dataset=dataset,
        method_cfg=method_cfg,
        selected_ids=selected_ids,
        image_dir=image_dir,
        image_mode=image_mode,
        append_augmented=append_augmented,
        seed=seed,
        seed_offset=int(cfg_get(premade_cfg, "augmentation_seed_offset", 1_000_003)),
        source_image_dir=source_image_dir,
        subset_ann_path=subset_ann_path,
        premade_cfg=premade_cfg,
    )
    premade_images = []
    premade_annotations = []
    next_ann_id = 1
    next_image_id = max((int(image["id"]) for image in subset.get("images", [])), default=0) + 1
    added_images = 0
    if append_augmented:
        for result in worker_results:
            image_info = dict(result["image_info"])
            premade_images.append(image_info)
            original_annotations = result.get("original_annotations", result["annotations"])
            annotations, next_ann_id = _renumber_original_annotations(
                original_annotations, next_ann_id
            )
            premade_annotations.extend(annotations)
        for result in worker_results:
            if not bool(result["applied"]):
                continue
            augmented_info = dict(result["augmented_image_info"])
            augmented_info["id"] = next_image_id
            next_image_id += 1
            premade_images.append(augmented_info)
            annotations, next_ann_id = _renumber_original_annotations(
                result["augmented_annotations"], next_ann_id, image_id=int(augmented_info["id"])
            )
            premade_annotations.extend(annotations)
            added_images += 1
    else:
        premade_images = [dict(result["image_info"]) for result in worker_results]
        for result in worker_results:
            annotations, next_ann_id = _renumber_original_annotations(
                result["annotations"], next_ann_id
            )
            premade_annotations.extend(annotations)

    premade = {
        "info": subset.get("info", {}),
        "licenses": subset.get("licenses", []),
        "images": premade_images,
        "annotations": premade_annotations,
        "categories": subset.get("categories", []),
    }
    generated_images = len(premade.get("images", []))
    generated_annotations = len(premade.get("annotations", []))
    balance_meta = None
    if _balance_enabled_for_method(method, premade_cfg):
        premade, balance_meta = _balance_premade_subset(premade, premade_cfg, seed)
    ann_path = variant_dir / "annotations.json"
    save_coco_json(premade, ann_path)
    selection_meta = {
        **selection_meta,
        "applied_images": int(applied_count),
        "append_augmented": append_augmented,
        "added_images": int(added_images),
        "base_images": len(subset.get("images", [])),
        "base_annotations": len(subset.get("annotations", [])),
        "generated_images": int(generated_images),
        "generated_annotations": int(generated_annotations),
        "output_images": len(premade.get("images", [])),
        "output_annotations": len(premade.get("annotations", [])),
        "added_annotations": len(premade.get("annotations", []))
        - len(subset.get("annotations", [])),
        "copy_paste_config": method_cfg,
        "num_workers": _premade_num_workers(premade_cfg, method_cfg),
    }
    if balance_meta is not None:
        selection_meta["balance"] = balance_meta
    metadata = _save_premade_metadata(
        premade=premade,
        before=subset,
        full=full,
        percent=percent,
        seed=seed,
        ann_path=ann_path,
        image_dir=image_dir,
        variant_dir=variant_dir,
        method=method,
        selection_meta=selection_meta,
        selected_count=len(selected_ids),
        save_visualizations=bool(cfg_get(premade_cfg, "save_visualizations", True)),
        max_preview_images=int(cfg_get(premade_cfg, "max_preview_images", 16)),
    )
    logger.info(
        "Wrote premade subset {}: {} selected, {} augmented",
        ann_path,
        len(selected_ids),
        applied_count,
    )
    return metadata


def build_premade_subsets(
    subset_ann_path: Path,
    subset: dict[str, Any],
    full: dict[str, Any],
    subset_dir: Path,
    source_image_dir: Path,
    percent: float,
    seed: int,
    premade_cfg: Any,
) -> list[dict[str, Any]]:
    """Build fixed offline augmentation variants for one materialized subset."""

    if not bool(cfg_get(premade_cfg, "enabled", False)):
        return []
    summaries = []
    for method in _premade_methods(premade_cfg):
        if method == "none":
            summaries.append(
                _build_none_premade_subset(
                    subset=subset,
                    full=full,
                    subset_dir=subset_dir,
                    source_image_dir=source_image_dir,
                    percent=percent,
                    seed=seed,
                    premade_cfg=premade_cfg,
                )
            )
            continue
        summaries.append(
            _build_copy_paste_premade_subset(
                subset_ann_path=subset_ann_path,
                subset=subset,
                full=full,
                subset_dir=subset_dir,
                source_image_dir=source_image_dir,
                percent=percent,
                seed=seed,
                premade_cfg=premade_cfg,
                method=method,
            )
        )
    return summaries


def premade_train_variant(subset_cfg: Any) -> str:
    premade_cfg = cfg_get(subset_cfg, "premade", None)
    return str(cfg_get(premade_cfg, "train_variant", "") or "")


def premade_train_paths(subset_cfg: Any, percent: float) -> tuple[Path, Path] | None:
    variant = premade_train_variant(subset_cfg)
    if not variant:
        return None
    seed = int(subset_cfg.seed)
    subset_dir = project_path(subset_cfg.output_dir) / f"{percent_slug(percent)}_seed_{seed}"
    premade_cfg = cfg_get(subset_cfg, "premade", None)
    variant_dir = _premade_variant_dir(subset_dir, premade_cfg, variant)
    resolved = _premade_paths_from_dir(variant_dir)
    if resolved is not None:
        return resolved

    legacy_dir = _legacy_premade_variant_dir(subset_dir, premade_cfg, variant)
    resolved = _premade_paths_from_dir(legacy_dir)
    if resolved is not None:
        return resolved

    if len(Path(variant).parts) == 1:
        candidates = sorted(
            path.parent
            for path in (subset_dir / _premade_output_subdir(premade_cfg)).glob(
                f"*/{variant}/metadata.json"
            )
        )
        if len(candidates) == 1:
            resolved = _premade_paths_from_dir(candidates[0])
            if resolved is not None:
                return resolved
        if len(candidates) > 1:
            candidate_names = [
                str(path.relative_to(subset_dir / _premade_output_subdir(premade_cfg)))
                for path in candidates
            ]
            raise ValueError(
                f"Found multiple premade variants named {variant!r}: {candidate_names}. "
                "Set subset.premade.target_images to the generation policy, or use "
                "subset.premade.train_variant=<target_slug>/<method>."
            )

    ann_path = variant_dir / "annotations.json"
    available = _available_premade_variants(subset_dir, premade_cfg)
    available_msg = (
        f" Available premade variants for this subset: {available}."
        if available
        else " No premade variants were found for this subset."
    )
    command = _premade_generation_command(
        percent=percent,
        seed=seed,
        premade_cfg=premade_cfg,
        method=Path(variant).name,
    )
    raise FileNotFoundError(
        f"Requested subset.premade.train_variant={variant!r}, but {ann_path} does not exist. "
        f"{available_msg} Generate it first with: `{command}`"
    )


def _percent_cli_value(percent: float) -> str:
    value = float(percent)
    return str(int(value)) if value.is_integer() else str(value)


def _premade_generation_command(
    percent: float,
    seed: int,
    premade_cfg: Any,
    method: str,
) -> str:
    target_images = str(cfg_get(premade_cfg, "target_images", "all")).lower()
    parts = [
        "uv run python -m cps.cli make-premade-subsets --config-name subset",
        f"subset.percentages='[{_percent_cli_value(percent)}]'",
        f"subset.seed={seed}",
        f"subset.premade.target_images={target_images}",
    ]
    if target_images in {"random", "random_percent", "percentage", "percent"}:
        parts.append(f"subset.premade.random_percent={cfg_get(premade_cfg, 'random_percent', 100)}")
    if target_images in {"underrepresented", "imbalanced", "rare"}:
        parts.append(f"subset.premade.rare_quantile={cfg_get(premade_cfg, 'rare_quantile', 0.25)}")
        class_ids = list(cfg_get(premade_cfg, "class_ids", []) or [])
        if class_ids:
            parts.append("subset.premade.class_ids='[" + ",".join(str(x) for x in class_ids) + "]'")
    parts.append(f"subset.premade.methods='[{method}]'")
    return " ".join(parts)


def _available_premade_variants(subset_dir: Path, premade_cfg: Any) -> list[str]:
    premade_root = subset_dir / _premade_output_subdir(premade_cfg)
    if not premade_root.exists():
        return []
    variants = []
    for metadata_path in sorted(premade_root.glob("**/metadata.json")):
        try:
            variants.append(str(metadata_path.parent.relative_to(premade_root)))
        except ValueError:
            continue
    for ann_path in sorted(premade_root.glob("**/annotations.json")):
        try:
            variant = str(ann_path.parent.relative_to(premade_root))
        except ValueError:
            continue
        if variant not in variants:
            variants.append(variant)
    return variants


def _premade_paths_from_dir(variant_dir: Path) -> tuple[Path, Path] | None:
    metadata_path = variant_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        return project_path(metadata["image_dir"]), project_path(metadata["annotation_json"])
    ann_path = variant_dir / "annotations.json"
    img_dir = variant_dir / "images"
    if not ann_path.exists():
        return None
    return img_dir, ann_path


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
        premade_summaries = build_premade_subsets(
            subset_ann_path=ann_path,
            subset=subset,
            full=coco,
            subset_dir=subset_dir,
            source_image_dir=image_output_dir if image_mode != "none" else image_dir,
            percent=percent,
            seed=seed,
            premade_cfg=getattr(subset_cfg, "premade", None),
        )
        if premade_summaries:
            metadata["premade_variants"] = premade_summaries
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
