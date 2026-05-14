"""Ultralytics YOLO26 instance-segmentation adapter.

The project trains from its own COCO dataloader and augmentation pipeline. This
module keeps that surface intact while using Ultralytics' random-initialized
YOLO26 segmentation architecture and native loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

YOLO_MODEL_TYPE = "yolo26_segmenter"
YOLO_LOSS_KEYS = ("loss_box", "loss_seg", "loss_cls", "loss_dfl", "loss_sem")


@dataclass(frozen=True)
class YOLO26Config:
    name: str = "yolo26n"
    size: str = "n"
    architecture: str = "yolo26n-seg.yaml"
    pretrained: bool = False
    overlap_mask: bool = False
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    epochs: int = 100
    verbose: bool = False
    stride_multiple: int = 32


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def yolo_config_from_cfg(cfg: Any, train_cfg: Any | None = None) -> YOLO26Config:
    name = str(_cfg_get(cfg, "name", "yolo26n"))
    size = str(_cfg_get(cfg, "size", _size_from_name(name)))
    architecture = str(_cfg_get(cfg, "architecture", f"yolo26{size}-seg.yaml"))
    pretrained = bool(_cfg_get(cfg, "pretrained", False))
    if pretrained:
        raise ValueError(
            "YOLO26 must train from raw architecture weights in this project. "
            "Use model.pretrained=false and an architecture *.yaml, not a *.pt checkpoint."
        )
    if architecture.endswith(".pt"):
        raise ValueError(
            f"YOLO26 architecture must be a *.yaml file, got {architecture!r}. "
            "*.pt files load pretrained weights."
        )
    epochs = int(_cfg_get(train_cfg, "epochs", _cfg_get(cfg, "epochs", 100)) or 100)
    return YOLO26Config(
        name=name,
        size=size,
        architecture=architecture,
        pretrained=False,
        overlap_mask=bool(_cfg_get(cfg, "overlap_mask", False)),
        box=float(_cfg_get(cfg, "box", 7.5)),
        cls=float(_cfg_get(cfg, "cls", 0.5)),
        dfl=float(_cfg_get(cfg, "dfl", 1.5)),
        epochs=epochs,
        verbose=bool(_cfg_get(cfg, "verbose", False)),
        stride_multiple=int(_cfg_get(cfg, "stride_multiple", 32) or 32),
    )


def _size_from_name(name: str) -> str:
    lowered = str(name).lower()
    for size in ("n", "s", "m", "l", "x"):
        if lowered == f"yolo26{size}" or lowered.endswith(f"yolo26{size}"):
            return size
    return "n"


class YOLO26SegmentationModel(nn.Module):
    """Torch module wrapper around Ultralytics' YOLO26 segmentation model."""

    def __init__(self, num_classes: int, config: YOLO26Config) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.config = config
        try:
            from ultralytics.cfg import get_cfg
            from ultralytics.nn.tasks import SegmentationModel
        except ImportError as exc:  # pragma: no cover - exercised by dependency resolution
            raise ImportError(
                "model=yolo26* requires the `ultralytics` package. "
                "Install project dependencies with `uv sync`."
            ) from exc

        self.model = SegmentationModel(
            config.architecture,
            ch=3,
            nc=self.num_classes,
            verbose=config.verbose,
        )
        self.model.args = get_cfg(
            overrides={
                "task": "segment",
                "mode": "train",
                "model": config.architecture,
                "overlap_mask": config.overlap_mask,
                "box": config.box,
                "cls": config.cls,
                "dfl": config.dfl,
                "epochs": config.epochs,
            }
        )
        self.model.names = {idx: str(idx) for idx in range(self.num_classes)}
        self.model.nc = self.num_classes
        stride = getattr(self.model, "stride", None)
        if torch.is_tensor(stride) and stride.numel() > 0:
            self.stride_multiple = int(stride.max().detach().cpu().item())
        else:
            self.stride_multiple = int(config.stride_multiple)
        self.stride_multiple = max(1, self.stride_multiple)

    def forward(self, images: list[torch.Tensor], return_attention: bool = False) -> dict[str, Any]:
        del return_attention  # YOLO has no decoder attention map to expose.
        batch = pad_images_to_tensor(images, stride_multiple=self.stride_multiple)
        preds = self.model(batch)
        outputs: dict[str, Any] = {
            "model_type": YOLO_MODEL_TYPE,
            "preds": preds,
            "image_size": torch.tensor(batch.shape[-2:], device=batch.device),
        }
        if not self.training:
            detections, proto = parse_yolo_eval_output(preds)
            outputs["detections"] = detections
            outputs["proto"] = proto
        return outputs


class YOLO26SegmentationCriterion(nn.Module):
    """Criterion wrapper that converts project targets to Ultralytics batches."""

    def __init__(self, model: YOLO26SegmentationModel) -> None:
        super().__init__()
        self._model = model
        self._native_criterion: Any | None = None
        self._criterion_device: torch.device | None = None

    def _criterion(self) -> Any:
        device = next(self._model.model.parameters()).device
        if self._native_criterion is None or self._criterion_device != device:
            self._model.model.criterion = self._model.model.init_criterion()
            self._native_criterion = self._model.model.criterion
            self._criterion_device = device
        return self._native_criterion

    def update(self) -> None:
        criterion = self._criterion()
        if hasattr(criterion, "update"):
            criterion.update()

    def forward(
        self, outputs: dict[str, Any], targets: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        if outputs.get("model_type") != YOLO_MODEL_TYPE:
            raise TypeError("YOLO26SegmentationCriterion received non-YOLO outputs.")
        device = outputs["image_size"].device
        image_h, image_w = [int(v) for v in outputs["image_size"].detach().cpu().tolist()]
        batch = targets_to_yolo_batch(
            targets,
            image_size=(image_h, image_w),
            device=device,
            num_classes=self._model.num_classes,
        )
        loss_vector, loss_items = self._criterion()(outputs["preds"], batch)
        loss = loss_vector.sum()
        losses: dict[str, torch.Tensor] = {"loss": loss}
        detached = loss_items.detach()
        for idx, key in enumerate(YOLO_LOSS_KEYS[: int(detached.numel())]):
            losses[key] = detached[idx]
        return losses


def pad_images_to_tensor(images: list[torch.Tensor], stride_multiple: int = 32) -> torch.Tensor:
    if not images:
        raise ValueError("YOLO26SegmentationModel requires at least one image.")
    max_h = max(int(image.shape[-2]) for image in images)
    max_w = max(int(image.shape[-1]) for image in images)
    stride = max(1, int(stride_multiple))
    pad_h = ((max_h + stride - 1) // stride) * stride
    pad_w = ((max_w + stride - 1) // stride) * stride
    batch = images[0].new_zeros((len(images), 3, pad_h, pad_w))
    for idx, image in enumerate(images):
        if image.shape[0] != 3:
            raise ValueError(f"YOLO26 expects 3-channel RGB images, got shape {tuple(image.shape)}")
        h, w = int(image.shape[-2]), int(image.shape[-1])
        batch[idx, :, :h, :w] = image
    return batch


def targets_to_yolo_batch(
    targets: list[dict[str, torch.Tensor]],
    *,
    image_size: tuple[int, int],
    device: torch.device,
    num_classes: int,
) -> dict[str, torch.Tensor]:
    image_h, image_w = image_size
    all_boxes: list[torch.Tensor] = []
    all_cls: list[torch.Tensor] = []
    all_batch_idx: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    sem_masks = torch.zeros((len(targets), image_h, image_w), dtype=torch.long, device=device)
    for batch_idx, target in enumerate(targets):
        labels = target["labels"].to(device=device, dtype=torch.long) - 1
        valid = (labels >= 0) & (labels < int(num_classes))
        if not valid.any():
            continue
        boxes_xyxy = target["boxes"].to(device=device, dtype=torch.float32)[valid]
        labels = labels[valid]
        masks = target["masks"].to(device=device)[valid]
        if masks.ndim != 3:
            raise ValueError(
                f"Expected target masks to have shape (N,H,W), got {tuple(masks.shape)}"
            )
        masks = pad_masks(masks, image_size)
        boxes = boxes_xyxy_to_xywh_normalized(boxes_xyxy, image_size)
        nonempty = (boxes[:, 2] > 0) & (boxes[:, 3] > 0) & (masks.flatten(1).sum(1) > 0)
        if not nonempty.any():
            continue
        boxes = boxes[nonempty]
        labels = labels[nonempty]
        masks = masks[nonempty]
        all_boxes.append(boxes)
        all_cls.append(labels.to(dtype=torch.float32))
        all_batch_idx.append(
            torch.full((labels.numel(),), batch_idx, dtype=torch.float32, device=device)
        )
        all_masks.append(masks)
        for label, mask in zip(labels, masks, strict=False):
            sem_masks[batch_idx][mask.bool()] = int(label.item())
    if all_boxes:
        bboxes = torch.cat(all_boxes, dim=0)
        cls = torch.cat(all_cls, dim=0)
        batch_idx = torch.cat(all_batch_idx, dim=0)
        masks = torch.cat(all_masks, dim=0)
    else:
        bboxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
        cls = torch.zeros((0,), dtype=torch.float32, device=device)
        batch_idx = torch.zeros((0,), dtype=torch.float32, device=device)
        masks = torch.zeros((0, image_h, image_w), dtype=torch.uint8, device=device)
    return {
        "bboxes": bboxes,
        "cls": cls,
        "batch_idx": batch_idx,
        "masks": masks,
        "sem_masks": sem_masks,
    }


def pad_masks(masks: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    image_h, image_w = image_size
    mask_h, mask_w = int(masks.shape[-2]), int(masks.shape[-1])
    if mask_h > image_h or mask_w > image_w:
        masks = F.interpolate(
            masks[:, None].float(),
            size=(min(mask_h, image_h), min(mask_w, image_w)),
            mode="nearest",
        )[:, 0].to(dtype=masks.dtype)
        mask_h, mask_w = int(masks.shape[-2]), int(masks.shape[-1])
    padded = torch.zeros((masks.shape[0], image_h, image_w), dtype=masks.dtype, device=masks.device)
    padded[:, :mask_h, :mask_w] = masks[:, :image_h, :image_w]
    return padded


def boxes_xyxy_to_xywh_normalized(
    boxes_xyxy: torch.Tensor, image_size: tuple[int, int]
) -> torch.Tensor:
    image_h, image_w = image_size
    boxes = boxes_xyxy.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, image_w)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, image_h)
    xywh = torch.empty_like(boxes)
    xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) * 0.5 / max(float(image_w), 1.0)
    xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) * 0.5 / max(float(image_h), 1.0)
    xywh[:, 2] = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) / max(float(image_w), 1.0)
    xywh[:, 3] = (boxes[:, 3] - boxes[:, 1]).clamp(min=0) / max(float(image_h), 1.0)
    return xywh.clamp(0, 1)


def parse_yolo_eval_output(preds: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        isinstance(preds, tuple)
        and len(preds) == 2
        and isinstance(preds[0], tuple)
        and len(preds[0]) == 2
    ):
        detections, proto = preds[0]
        if isinstance(proto, tuple):
            proto = proto[0]
        return detections, proto
    raise TypeError(f"Unsupported YOLO26 evaluation output type: {type(preds).__name__}")


@torch.no_grad()
def yolo_outputs_to_predictions(
    outputs: dict[str, Any],
    targets: list[dict[str, torch.Tensor]],
    label_to_cat_id: dict[int, int],
    score_threshold: float = 0.05,
    max_detections: int = 100,
    mask_threshold: float = 0.5,
    include_masks: bool = True,
    target_size_key: str = "orig_size",
) -> list[dict[str, Any]]:
    try:
        from ultralytics.utils import ops
    except ImportError as exc:  # pragma: no cover
        raise ImportError("YOLO prediction conversion requires `ultralytics`.") from exc

    detections = outputs["detections"]
    proto = outputs.get("proto")
    predictions: list[dict[str, Any]] = []
    for batch_idx, target in enumerate(targets):
        target_h, target_w = [
            int(v) for v in target.get(target_size_key, target["orig_size"]).tolist()
        ]
        input_h, input_w = [int(v) for v in target["size"].tolist()]
        image_id = int(target["image_id"].item())
        det = detections[batch_idx]
        keep = det[:, 4] >= float(score_threshold)
        det = det[keep]
        if det.numel() == 0:
            continue
        det = det[torch.argsort(det[:, 4], descending=True)[: int(max_detections)]]
        boxes_input = det[:, :4].clone()
        boxes_input[:, 0::2] = boxes_input[:, 0::2].clamp(0, input_w)
        boxes_input[:, 1::2] = boxes_input[:, 1::2].clamp(0, input_h)
        masks = None
        if include_masks and proto is not None and det.shape[1] > 6:
            masks = ops.process_mask(
                proto[batch_idx],
                det[:, 6:],
                boxes_input,
                tuple(int(v) for v in outputs["image_size"].detach().cpu().tolist()),
                upsample=True,
            )
            masks = masks[:, :input_h, :input_w]
            if (input_h, input_w) != (target_h, target_w):
                masks = F.interpolate(
                    masks[:, None].float(),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
        scale = boxes_input.new_tensor(
            [
                target_w / max(float(input_w), 1.0),
                target_h / max(float(input_h), 1.0),
                target_w / max(float(input_w), 1.0),
                target_h / max(float(input_h), 1.0),
            ]
        )
        boxes_out = boxes_input * scale
        for idx in range(det.shape[0]):
            class_idx = int(det[idx, 5].item())
            label = class_idx + 1
            pred: dict[str, Any] = {
                "image_id": image_id,
                "category_id": int(label_to_cat_id.get(label, label)),
                "score": float(det[idx, 4].detach().cpu().item()),
                "bbox_xyxy": [float(v) for v in boxes_out[idx].detach().cpu().tolist()],
            }
            if masks is not None:
                pred["mask"] = (
                    masks[idx].detach().cpu().float().ge(float(mask_threshold)).numpy().astype(bool)
                )
            predictions.append(pred)
    return predictions


def build_yolo_model_and_criterion(
    cfg: Any, num_classes: int, train_cfg: Any | None = None
) -> tuple[YOLO26SegmentationModel, YOLO26SegmentationCriterion]:
    model_cfg = yolo_config_from_cfg(cfg, train_cfg=train_cfg)
    model = YOLO26SegmentationModel(num_classes=num_classes, config=model_cfg)
    criterion = YOLO26SegmentationCriterion(model)
    return model, criterion
