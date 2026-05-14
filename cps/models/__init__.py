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


def build_model_and_criterion(
    cfg: Any, num_classes: int, train_cfg: Any | None = None
) -> tuple[torch.nn.Module, torch.nn.Module]:
    if _is_yolo26_config(cfg):
        return build_yolo_model_and_criterion(cfg, num_classes=num_classes, train_cfg=train_cfg)
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
) -> list[dict[str, Any]]:
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
    "TinyDETRSegmenter",
    "YOLO26SegmentationCriterion",
    "YOLO26SegmentationModel",
    "build_model_and_criterion",
    "outputs_to_predictions",
]
