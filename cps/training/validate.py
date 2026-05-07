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
from cps.models.detr import (
    box_xyxy_to_cxcywh,
    normalize_boxes_xyxy,
    outputs_to_predictions,
)
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
    visualize_max_images: int = 4,
    attention_samples: int = 2,
    forward_batch_size: int | None = None,
    mode: str = "full",
    iou_types: tuple[str, ...] = ("segm", "bbox"),
    empty_cache: bool = True,
    empty_cache_between_chunks: bool = False,
) -> dict[str, Any]:
    model.eval()
    mode = str(mode).lower()
    if mode not in {"full", "bbox", "loss", "gpu_loss"}:
        raise ValueError("eval.mode must be one of: full, bbox, loss, gpu_loss.")
    if mode == "bbox":
        iou_types = ("bbox",)
    elif mode in {"loss", "gpu_loss"}:
        iou_types = ()
        visualize_batches = 0
        attention_samples = 0
    compute_predictions = mode in {"full", "bbox"}
    compute_coco_metrics = bool(iou_types)
    include_prediction_masks = "segm" in iou_types
    output_dir = project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Validation mode: {}, forward_batch_size: {}, iou_types: {}, empty_cache: {}",
        mode,
        forward_batch_size if forward_batch_size is not None else "loader_batch",
        list(iou_types),
        empty_cache,
    )
    predictions: list[dict[str, Any]] = []
    loss_sums: dict[str, float] = {}
    batches = 0
    loss_batches = 0
    saved_viz = 0
    saved_attention = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for images, targets in tqdm(dataloader, desc="validate", leave=False):
            chunk_size = len(images) if forward_batch_size is None else int(forward_batch_size)
            if chunk_size <= 0:
                raise ValueError("eval.forward_batch_size must be > 0 when set.")
            for chunk_start in range(0, len(images), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(images))
                image_chunk = [img.to(device) for img in images[chunk_start:chunk_end]]
                target_chunk = move_targets_to_device(targets[chunk_start:chunk_end], device)
                return_attention = saved_attention < attention_samples
                outputs = model(image_chunk, return_attention=return_attention)
                losses = None
                if mode == "gpu_loss" and criterion is not None:
                    losses = gpu_greedy_validation_losses(outputs, target_chunk, criterion)
                elif criterion is not None:
                    losses = criterion(outputs, target_chunk)
                if losses is not None:
                    for key, value in losses.items():
                        loss_sums[key] = loss_sums.get(key, 0.0) + float(value.detach().cpu())
                    loss_batches += 1
                chunk_predictions = []
                if compute_predictions:
                    chunk_predictions = outputs_to_predictions(
                        outputs,
                        target_chunk,
                        label_to_cat_id=label_to_cat_id,
                        score_threshold=score_threshold,
                        max_detections=max_detections,
                        include_masks=include_prediction_masks,
                    )
                predictions.extend(chunk_predictions)
                if saved_viz < visualize_batches:
                    save_validation_grid(
                        images=image_chunk,
                        targets=target_chunk,
                        predictions=chunk_predictions,
                        categories=categories,
                        output_path=viz_dir / f"gt_vs_pred_{batches:04d}_{chunk_start:03d}.png",
                        max_images=visualize_max_images,
                    )
                    saved_viz += 1
                if return_attention and "cross_attention" in outputs:
                    saved_attention += save_decoder_attention_maps(
                        images=image_chunk,
                        targets=target_chunk,
                        outputs=outputs,
                        output_dir=viz_dir / "attention",
                        batch_offset=batches,
                        max_images=attention_samples - saved_attention,
                    )
                del outputs, image_chunk, target_chunk, chunk_predictions
                if losses is not None:
                    del losses
                if empty_cache_between_chunks:
                    clear_cuda_cache(device)
            batches += 1
            if max_batches is not None and batches >= int(max_batches):
                break
    loss_means = {key: value / max(loss_batches, 1) for key, value in loss_sums.items()}
    if empty_cache:
        clear_cuda_cache(device)
    log_cuda_validation_memory(device)
    if compute_coco_metrics:
        metrics = evaluate_coco_predictions(
            annotation_json, predictions, output_dir=output_dir, iou_types=iou_types
        )
    else:
        metrics = {
            "available": False,
            "reason": f"COCO metrics disabled for eval.mode={mode}",
            "num_predictions": len(predictions),
        }
    metrics["losses"] = loss_means
    with (output_dir / "validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Validation metrics written to {}", output_dir)
    return metrics


def gpu_greedy_validation_losses(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    criterion: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """GPU-resident validation loss proxy using greedy query/target assignment.

    Exact DETR validation loss uses SciPy's Hungarian matcher on CPU. This proxy
    avoids that CPU handoff for training-time validation; use eval.mode=full for
    final COCO metrics.
    """

    matcher = getattr(criterion, "matcher", None)
    config = getattr(criterion, "config", None)
    if matcher is None or config is None:
        return {}
    indices = greedy_match_on_device(outputs, targets, criterion)
    num_boxes = sum(len(target["labels"]) for target in targets)
    num_boxes_f = float(max(num_boxes, 1))
    losses = {}
    losses.update(criterion.loss_labels(outputs, targets, indices))
    losses.update(criterion.loss_boxes(outputs, targets, indices, num_boxes_f))
    losses.update(criterion.loss_masks(outputs, targets, indices, num_boxes_f))
    losses["loss"] = (
        losses["loss_ce"]
        + config.bbox_loss_coef * losses["loss_bbox"]
        + config.giou_loss_coef * losses["loss_giou"]
        + config.mask_loss_coef * losses["loss_mask"]
        + config.dice_loss_coef * losses["loss_dice"]
    )
    return losses


def greedy_match_on_device(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    criterion: torch.nn.Module,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    matcher = criterion.matcher
    config = criterion.config
    out_prob = outputs["pred_logits"].softmax(-1)
    out_bbox = outputs["pred_boxes"]
    out_masks = outputs["pred_masks"].sigmoid()
    indices = []
    for batch_idx in range(out_prob.shape[0]):
        labels = targets[batch_idx]["labels"].to(out_prob.device)
        if labels.numel() == 0:
            empty = torch.empty(0, dtype=torch.int64, device=out_prob.device)
            indices.append((empty, empty))
            continue
        tgt_bbox = box_xyxy_to_cxcywh(
            normalize_boxes_xyxy(
                targets[batch_idx]["boxes"].to(out_prob.device), targets[batch_idx]["size"]
            )
        )
        cost_class = -out_prob[batch_idx][:, labels]
        cost_bbox = torch.cdist(out_bbox[batch_idx], tgt_bbox, p=1)
        cost_mask = matcher._mask_cost(
            out_masks[batch_idx], targets[batch_idx]["masks"].to(out_prob.device)
        )
        cost = (
            config.cost_class * cost_class
            + config.cost_bbox * cost_bbox
            + config.cost_mask * cost_mask
        )
        indices.append(greedy_unique_assignment(cost))
    return indices


def greedy_unique_assignment(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_queries, num_targets = cost.shape
    if num_queries == 0 or num_targets == 0:
        empty = torch.empty(0, dtype=torch.int64, device=cost.device)
        return empty, empty
    remaining_queries = torch.ones(num_queries, dtype=torch.bool, device=cost.device)
    remaining_targets = torch.ones(num_targets, dtype=torch.bool, device=cost.device)
    src_idx = []
    tgt_idx = []
    inf = torch.tensor(float("inf"), dtype=cost.dtype, device=cost.device)
    for _ in range(min(num_queries, num_targets)):
        masked_cost = cost.masked_fill(
            ~(remaining_queries[:, None] & remaining_targets[None, :]), inf
        )
        flat_idx = torch.argmin(masked_cost)
        best = masked_cost.flatten()[flat_idx]
        if not torch.isfinite(best).item():
            break
        query_idx = flat_idx // num_targets
        target_idx = flat_idx % num_targets
        src_idx.append(query_idx)
        tgt_idx.append(target_idx)
        remaining_queries[query_idx] = False
        remaining_targets[target_idx] = False
    if not src_idx:
        empty = torch.empty(0, dtype=torch.int64, device=cost.device)
        return empty, empty
    return torch.stack(src_idx).long(), torch.stack(tgt_idx).long()


def clear_cuda_cache(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


def log_cuda_validation_memory(device: torch.device) -> None:
    if device.type != "cuda":
        return
    allocated_gib = torch.cuda.memory_allocated(device) / 1024**3
    reserved_gib = torch.cuda.memory_reserved(device) / 1024**3
    peak_allocated_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    logger.info(
        "Validation CUDA memory - allocated: {:.2f} GiB, reserved: {:.2f} GiB, "
        "peak allocated: {:.2f} GiB",
        allocated_gib,
        reserved_gib,
        peak_allocated_gib,
    )


def save_validation_grid(
    images: list[torch.Tensor],
    targets: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    categories: dict[int, str],
    output_path: str | Path,
    max_images: int | None = None,
) -> None:
    by_image: dict[int, list[dict[str, Any]]] = {}
    for pred in predictions:
        by_image.setdefault(int(pred["image_id"]), []).append(pred)
    rendered = []
    labels = []
    image_targets = list(zip(images, targets, strict=False))
    if max_images is not None:
        image_targets = image_targets[: max(0, int(max_images))]
    for image, target in image_targets:
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
