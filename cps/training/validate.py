"""Validation loop, COCO metric computation, and qualitative visualizations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from cps.analysis.attention import save_decoder_attention_maps
from cps.data.visualization import overlay_instances, save_image_grid
from cps.evaluation.coco_metrics import evaluate_coco_predictions
from cps.models.detr import outputs_to_predictions
from cps.paths import project_path


def move_targets_to_device(
    targets: list[dict[str, Any]], device: torch.device
) -> list[dict[str, Any]]:
    moved = []
    for target in targets:
        moved_target = {}
        for key, value in target.items():
            moved_target[key] = value.to(device) if torch.is_tensor(value) else value
        moved.append(moved_target)
    return moved


def validation_loop(
    model: torch.nn.Module,
    criterion: torch.nn.Module | None,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    label_to_cat_id: dict[int, int],
    categories: dict[int, str],
    annotation_json: str | Path,
    output_dir: str | Path,
    score_threshold: float = 0.05,
    max_detections: int = 100,
    max_batches: int | None = None,
    visualize_batches: int = 2,
    attention_samples: int = 2,
) -> dict[str, Any]:
    model.eval()
    output_dir = project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    loss_sums: dict[str, float] = {}
    batches = 0
    saved_viz = 0
    saved_attention = 0
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="validate", leave=False):
            images = [img.to(device) for img in images]
            targets_device = move_targets_to_device(targets, device)
            return_attention = saved_attention < attention_samples
            outputs = model(images, return_attention=return_attention)
            if criterion is not None:
                losses = criterion(outputs, targets_device)
                for key, value in losses.items():
                    loss_sums[key] = loss_sums.get(key, 0.0) + float(value.detach().cpu())
            batch_predictions = outputs_to_predictions(
                outputs,
                targets_device,
                label_to_cat_id=label_to_cat_id,
                score_threshold=score_threshold,
                max_detections=max_detections,
            )
            predictions.extend(batch_predictions)
            if saved_viz < visualize_batches:
                save_validation_grid(
                    images=images,
                    targets=targets_device,
                    predictions=batch_predictions,
                    categories=categories,
                    output_path=viz_dir / f"gt_vs_pred_{batches:04d}.png",
                )
                saved_viz += 1
            if return_attention and "cross_attention" in outputs:
                saved_attention += save_decoder_attention_maps(
                    images=images,
                    targets=targets_device,
                    outputs=outputs,
                    output_dir=viz_dir / "attention",
                    batch_offset=batches,
                    max_images=attention_samples - saved_attention,
                )
            batches += 1
            if max_batches is not None and batches >= int(max_batches):
                break
    loss_means = {key: value / max(batches, 1) for key, value in loss_sums.items()}
    metrics = evaluate_coco_predictions(annotation_json, predictions, output_dir=output_dir)
    metrics["losses"] = loss_means
    with (output_dir / "validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Validation metrics written to {}", output_dir)
    return metrics


def save_validation_grid(
    images: list[torch.Tensor],
    targets: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    categories: dict[int, str],
    output_path: str | Path,
) -> None:
    by_image: dict[int, list[dict[str, Any]]] = {}
    for pred in predictions:
        by_image.setdefault(int(pred["image_id"]), []).append(pred)
    rendered = []
    labels = []
    for image, target in zip(images, targets, strict=False):
        height, width = [int(v) for v in target["orig_size"].detach().cpu().tolist()]
        arr = (image[:, :height, :width].detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(
            0, 255
        )
        arr = arr.astype(np.uint8)
        gt_instances = []
        for idx in range(len(target["labels"])):
            cat_id = int(target["category_ids"][idx].detach().cpu().item())
            gt_instances.append(
                {
                    "category_id": cat_id,
                    "bbox_xyxy": target["boxes"][idx].detach().cpu().tolist(),
                    "mask": target["masks"][idx].detach().cpu().numpy().astype(bool),
                }
            )
        image_id = int(target["image_id"].item())
        pred_instances = by_image.get(image_id, [])
        rendered.append(overlay_instances(arr, gt_instances, categories))
        labels.append(f"GT {image_id}")
        rendered.append(overlay_instances(arr, pred_instances, categories))
        labels.append(f"Pred {image_id}")
    save_image_grid(rendered, labels, output_path, columns=2)
