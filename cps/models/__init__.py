"""Model factory and prediction conversion helpers."""

from __future__ import annotations

from typing import Any

import torch

from cps.models.detr import (
    DETRCriterion,
    TinyDETRSegmenter,
)
from cps.models.detr import (
    build_model_and_criterion as build_detr_model_and_criterion,
)
from cps.models.detr import (
    outputs_to_predictions as detr_outputs_to_predictions,
)
from cps.models.dinov3_mask_rcnn import (
    DINOV3_MASK_RCNN_MODEL_TYPE,
    DINOv3MaskRCNNCriterion,
    DINOv3MaskRCNNSegmenter,
)
from cps.models.dinov3_mask_rcnn import (
    build_model_and_criterion as build_dinov3_mask_rcnn_model_and_criterion,
)
from cps.models.dinov3_mask_rcnn import (
    outputs_to_predictions as dinov3_mask_rcnn_outputs_to_predictions,
)
from cps.models.mask_rcnn import (
    MASK_RCNN_MODEL_TYPE,
    MaskRCNNCriterion,
    MaskRCNNSegmenter,
)
from cps.models.mask_rcnn import (
    build_model_and_criterion as build_mask_rcnn_model_and_criterion,
)
from cps.models.mask_rcnn import (
    outputs_to_predictions as mask_rcnn_outputs_to_predictions,
)
from cps.models.yolo import (
    YOLO_MODEL_TYPE,
    YOLO26SegmentationCriterion,
    YOLO26SegmentationModel,
    build_yolo_model_and_criterion,
    yolo_outputs_to_predictions,
)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _is_yolo26_config(cfg: Any) -> bool:
    family = str(_cfg_get(cfg, "family", "")).lower()
    name = str(_cfg_get(cfg, "name", "")).lower()
    architecture = str(_cfg_get(cfg, "architecture", "")).lower()
    return family == "yolo26" or name.startswith("yolo26") or architecture.startswith("yolo26")


def _is_mask_rcnn_config(cfg: Any) -> bool:
    family = str(_cfg_get(cfg, "family", "")).lower()
    name = str(_cfg_get(cfg, "name", "")).lower()
    architecture = str(_cfg_get(cfg, "architecture", "")).lower()
    return (
        family in {"maskrcnn", "mask_rcnn"}
        or name.startswith("maskrcnn")
        or name.startswith("mask_rcnn")
        or architecture.startswith("maskrcnn")
        or architecture.startswith("mask_rcnn")
    )


def _is_dinov3_mask_rcnn_config(cfg: Any) -> bool:
    family = str(_cfg_get(cfg, "family", "")).lower()
    name = str(_cfg_get(cfg, "name", "")).lower()
    architecture = str(_cfg_get(cfg, "architecture", "")).lower()
    triggers = ("dinov3", "dinov3_maskrcnn", "dinov3_mask_rcnn")
    return (
        family in {"dinov3", "dinov3_maskrcnn", "dinov3_mask_rcnn"}
        or any(name.startswith(t) for t in triggers)
        or any(architecture.startswith(t) for t in triggers)
    )


def build_model_and_criterion(
    cfg: Any, num_classes: int, train_cfg: Any | None = None
) -> tuple[torch.nn.Module, torch.nn.Module]:
    if _is_yolo26_config(cfg):
        return build_yolo_model_and_criterion(cfg, num_classes=num_classes, train_cfg=train_cfg)
    if _is_dinov3_mask_rcnn_config(cfg):
        return build_dinov3_mask_rcnn_model_and_criterion(
            cfg, num_classes=num_classes, train_cfg=train_cfg
        )
    if _is_mask_rcnn_config(cfg):
        return build_mask_rcnn_model_and_criterion(
            cfg, num_classes=num_classes, train_cfg=train_cfg
        )
    return build_detr_model_and_criterion(cfg, num_classes=num_classes)


@torch.no_grad()
def outputs_to_predictions(
    outputs: dict[str, Any],
    targets: list[dict[str, torch.Tensor]],
    label_to_cat_id: dict[int, int],
    score_threshold: float = 0.05,
    max_detections: int = 100,
    mask_threshold: float = 0.5,
    include_masks: bool = True,
    target_size_key: str = "orig_size",
    class_agnostic_nms_iou: float | None = None,
) -> list[dict[str, Any]]:
    """Dispatch to the right per-model converter based on ``outputs["model_type"]``.

    ``class_agnostic_nms_iou`` is the viz-only knob that suppresses duplicate
    boxes spanning different classes (e.g. cat-vs-dog over the same dog),
    which is what produces the "many overlapping low-score boxes" the user
    sees in fast-eval. Only the Mask R-CNN converters honour it today;
    everything else ignores it harmlessly.
    """

    if outputs.get("model_type") == YOLO_MODEL_TYPE:
        return yolo_outputs_to_predictions(
            outputs,
            targets,
            label_to_cat_id,
            score_threshold=score_threshold,
            max_detections=max_detections,
            mask_threshold=mask_threshold,
            include_masks=include_masks,
            target_size_key=target_size_key,
        )
    if outputs.get("model_type") == DINOV3_MASK_RCNN_MODEL_TYPE:
        return dinov3_mask_rcnn_outputs_to_predictions(
            outputs,
            targets,
            label_to_cat_id,
            score_threshold=score_threshold,
            max_detections=max_detections,
            mask_threshold=mask_threshold,
            include_masks=include_masks,
            target_size_key=target_size_key,
            class_agnostic_nms_iou=class_agnostic_nms_iou,
        )
    if outputs.get("model_type") == MASK_RCNN_MODEL_TYPE:
        return mask_rcnn_outputs_to_predictions(
            outputs,
            targets,
            label_to_cat_id,
            score_threshold=score_threshold,
            max_detections=max_detections,
            mask_threshold=mask_threshold,
            include_masks=include_masks,
            target_size_key=target_size_key,
        )
    return detr_outputs_to_predictions(
        outputs,
        targets,
        label_to_cat_id,
        score_threshold=score_threshold,
        max_detections=max_detections,
        mask_threshold=mask_threshold,
        include_masks=include_masks,
        target_size_key=target_size_key,
    )


__all__ = [
    "DETRCriterion",
    "DINOV3_MASK_RCNN_MODEL_TYPE",
    "DINOv3MaskRCNNCriterion",
    "DINOv3MaskRCNNSegmenter",
    "MASK_RCNN_MODEL_TYPE",
    "MaskRCNNCriterion",
    "MaskRCNNSegmenter",
    "TinyDETRSegmenter",
    "YOLO26SegmentationCriterion",
    "YOLO26SegmentationModel",
    "build_model_and_criterion",
    "outputs_to_predictions",
]
