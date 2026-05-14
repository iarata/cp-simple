"""A small DETR-style instance segmentation baseline.

This is intentionally minimal: a CNN backbone, Transformer encoder/decoder,
fixed object queries, class and box heads, and a query-conditioned mask head.
It is suitable for controlled augmentation studies, not for state-of-the-art
COCO accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    num_queries: int = 100
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    nheads: int = 8
    dim_feedforward: int = 512
    dropout: float = 0.1
    eos_coef: float = 0.1
    cost_class: float = 1.0
    cost_bbox: float = 5.0
    cost_mask: float = 1.0
    bbox_loss_coef: float = 5.0
    giou_loss_coef: float = 2.0
    mask_loss_coef: float = 1.0
    dice_loss_coef: float = 1.0
    mask_loss_size: int = 128
    backbone: str = "tiny_cnn"
    backbone_name: str = "resnet18"
    backbone_pretrained: bool = True
    backbone_out_index: int = 3
    backbone_freeze: bool = False
    normalize_backbone_inputs: bool = True
    attention_layer: str = "last"


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def model_config_from_cfg(cfg: Any) -> ModelConfig:
    return ModelConfig(
        hidden_dim=int(_cfg_get(cfg, "hidden_dim", 128)),
        num_queries=int(_cfg_get(cfg, "num_queries", 100)),
        num_encoder_layers=int(_cfg_get(cfg, "num_encoder_layers", 3)),
        num_decoder_layers=int(_cfg_get(cfg, "num_decoder_layers", 3)),
        nheads=int(_cfg_get(cfg, "nheads", 8)),
        dim_feedforward=int(_cfg_get(cfg, "dim_feedforward", 512)),
        dropout=float(_cfg_get(cfg, "dropout", 0.1)),
        eos_coef=float(_cfg_get(cfg, "eos_coef", 0.1)),
        cost_class=float(_cfg_get(cfg, "cost_class", 1.0)),
        cost_bbox=float(_cfg_get(cfg, "cost_bbox", 5.0)),
        cost_mask=float(_cfg_get(cfg, "cost_mask", 1.0)),
        bbox_loss_coef=float(_cfg_get(cfg, "bbox_loss_coef", 5.0)),
        giou_loss_coef=float(_cfg_get(cfg, "giou_loss_coef", 2.0)),
        mask_loss_coef=float(_cfg_get(cfg, "mask_loss_coef", 1.0)),
        dice_loss_coef=float(_cfg_get(cfg, "dice_loss_coef", 1.0)),
        mask_loss_size=int(_cfg_get(cfg, "mask_loss_size", 128)),
        backbone=str(_cfg_get(cfg, "backbone", "tiny_cnn")),
        backbone_name=str(_cfg_get(cfg, "backbone_name", "resnet18")),
        backbone_pretrained=bool(_cfg_get(cfg, "backbone_pretrained", True)),
        backbone_out_index=int(_cfg_get(cfg, "backbone_out_index", 3)),
        backbone_freeze=bool(_cfg_get(cfg, "backbone_freeze", False)),
        normalize_backbone_inputs=bool(_cfg_get(cfg, "normalize_backbone_inputs", True)),
        attention_layer=str(_cfg_get(cfg, "attention_layer", "last")),
    )


def nested_tensor_from_tensor_list(
    tensors: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    max_size = [max(s) for s in zip(*[img.shape for img in tensors], strict=False)]
    batch_shape = [len(tensors), *max_size]
    batched = tensors[0].new_zeros(batch_shape)
    mask = torch.ones(
        (len(tensors), max_size[1], max_size[2]), dtype=torch.bool, device=tensors[0].device
    )
    for img, pad_img, pad_mask in zip(tensors, batched, mask, strict=True):
        c, h, w = img.shape
        pad_img[:c, :h, :w].copy_(img)
        pad_mask[:h, :w] = False
    return batched, mask


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int = 64, temperature: int = 10_000) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * torch.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * torch.pi
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class TinyBackbone(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        mid = max(32, hidden_dim // 2)
        # Four stride-2 stages keep transformer token counts manageable on COCO
        # while preserving enough spatial resolution for a lightweight mask head.
        self.body = nn.Sequential(
            nn.Conv2d(3, mid, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(_group_count(mid), mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(mid), mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class TimmBackbone(nn.Module):
    """Pretrained timm visual encoder adapter for DETR features."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "model.backbone=timm requires the `timm` package. "
                "Install project dependencies with `uv sync`."
            ) from exc

        self.body = timm.create_model(
            config.backbone_name,
            pretrained=config.backbone_pretrained,
            features_only=True,
            out_indices=(config.backbone_out_index,),
        )
        channels = self.body.feature_info.channels()
        if not channels:
            raise ValueError(f"timm model {config.backbone_name!r} did not expose feature info.")
        in_channels = int(channels[-1])
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, config.hidden_dim, kernel_size=1),
            nn.GroupNorm(_group_count(config.hidden_dim), config.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.normalize_inputs = bool(config.normalize_backbone_inputs)
        default_cfg = getattr(self.body, "default_cfg", {}) or {}
        mean = torch.tensor(default_cfg.get("mean", (0.485, 0.456, 0.406)), dtype=torch.float32)
        std = torch.tensor(default_cfg.get("std", (0.229, 0.224, 0.225)), dtype=torch.float32)
        self.register_buffer("input_mean", mean.view(1, 3, 1, 1), persistent=False)
        self.register_buffer("input_std", std.view(1, 3, 1, 1), persistent=False)
        if config.backbone_freeze:
            for parameter in self.body.parameters():
                parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_inputs:
            x = (x - self.input_mean) / self.input_std.clamp(min=1e-6)
        features = self.body(x)
        feature = self._last_feature(features)
        return self.proj(feature)

    def _last_feature(self, features: Any) -> torch.Tensor:
        if torch.is_tensor(features):
            feature = features
        elif isinstance(features, dict):
            if not features:
                raise ValueError(f"timm model {self.config.backbone_name!r} returned no features.")
            feature = next(reversed(tuple(features.values())))
        elif isinstance(features, (list, tuple)):
            if not features:
                raise ValueError(f"timm model {self.config.backbone_name!r} returned no features.")
            feature = features[-1]
        else:
            raise TypeError(
                f"timm model {self.config.backbone_name!r} returned unsupported feature type "
                f"{type(features).__name__}."
            )
        if feature.ndim != 4:
            raise ValueError(
                f"timm model {self.config.backbone_name!r} returned a non-spatial feature tensor "
                f"with shape {tuple(feature.shape)}."
            )
        return feature


def build_backbone(config: ModelConfig) -> nn.Module:
    backbone = config.backbone.lower()
    if backbone in {"tiny", "tiny_cnn", "cnn"}:
        return TinyBackbone(config.hidden_dim)
    if backbone in {"timm", "pretrained", "timm_pretrained"}:
        return TimmBackbone(config)
    raise ValueError("model.backbone must be one of: tiny_cnn, timm.")


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d_model = config.hidden_dim
        self.self_attn = nn.MultiheadAttention(
            d_model, config.nheads, config.dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, config.nheads, config.dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        self.activation = nn.ReLU()
        self.last_cross_attention: torch.Tensor | None = None

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
        store_attention: bool = False,
    ) -> torch.Tensor:
        tgt2, _ = self.self_attn(tgt, tgt, tgt, need_weights=False)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2, attn = self.cross_attn(
            tgt,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=store_attention,
            average_attn_weights=False,
        )
        self.last_cross_attention = attn.detach() if store_attention and attn is not None else None
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


class TinyDETRSegmenter(nn.Module):
    def __init__(self, num_classes: int, config: ModelConfig) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.config = config
        hidden_dim = config.hidden_dim
        self.backbone = build_backbone(config)
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=config.nheads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_encoder_layers, enable_nested_tensor=False
        )
        self.query_embed = nn.Embedding(config.num_queries, hidden_dim)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_decoder_layers)]
        )
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # 0 is no-object/background.
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.mask_feature_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        # Bias the classifier so the initial softmax matches the empirical class
        # prior (mostly background). Without this the first hundreds of iterations
        # are wasted learning to predict "no object" against random logits.
        with torch.no_grad():
            self.class_embed.bias.zero_()
            background_prior = 1.0 - 1.0 / float(num_classes + 1)
            self.class_embed.bias[0] = float(
                torch.log(torch.tensor(background_prior / (1 - background_prior)))
            )

    def attention_layer_index(self) -> int:
        layer = str(self.config.attention_layer).strip().lower()
        last_idx = len(self.decoder_layers) - 1
        if layer in {"first", "0"}:
            return 0
        if layer in {"last", "-1"}:
            return last_idx
        try:
            layer_idx = int(layer)
        except ValueError as exc:
            raise ValueError(
                "model.attention_layer must be 'first', 'last', or a decoder layer index."
            ) from exc
        if layer_idx < 0:
            layer_idx = len(self.decoder_layers) + layer_idx
        if layer_idx < 0 or layer_idx > last_idx:
            raise ValueError(
                f"model.attention_layer={self.config.attention_layer!r} is outside 0..{last_idx}."
            )
        return layer_idx

    def forward(
        self, images: list[torch.Tensor], return_attention: bool = False
    ) -> dict[str, torch.Tensor]:
        src, mask = nested_tensor_from_tensor_list(images)
        features = self.backbone(src)
        small_mask = F.interpolate(mask[:, None].float(), size=features.shape[-2:]).to(torch.bool)[
            :, 0
        ]
        pos = self.position_embedding(features, small_mask)
        memory = features.flatten(2).permute(0, 2, 1)
        pos_flat = pos.flatten(2).permute(0, 2, 1)
        mask_flat = small_mask.flatten(1)
        memory = self.encoder(memory + pos_flat, src_key_padding_mask=mask_flat)
        queries = self.query_embed.weight.unsqueeze(0).repeat(src.shape[0], 1, 1)
        hs = queries
        attention_layer_idx = self.attention_layer_index() if return_attention else -1
        for layer_idx, layer in enumerate(self.decoder_layers):
            hs = layer(
                hs,
                memory,
                memory_key_padding_mask=mask_flat,
                store_attention=return_attention and layer_idx == attention_layer_idx,
            )
        logits = self.class_embed(hs)
        boxes = self.bbox_embed(hs).sigmoid()
        mask_features = self.mask_feature_proj(features)
        mask_embed = self.mask_embed(hs)
        # Keep mask logits at feature resolution; losses downsample targets, and
        # evaluation upsamples only selected masks to avoid B*Q*H*W allocations.
        pred_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        out: dict[str, torch.Tensor] = {
            "pred_logits": logits,
            "pred_boxes": boxes,
            "pred_masks": pred_masks,
            "feature_mask": small_mask,
        }
        if (
            return_attention
            and attention_layer_idx >= 0
            and self.decoder_layers[attention_layer_idx].last_cross_attention is not None
        ):
            out["cross_attention"] = self.decoder_layers[attention_layer_idx].last_cross_attention
            out["attention_hw"] = torch.tensor(features.shape[-2:], device=src.device)
            out["attention_layer_index"] = torch.tensor(attention_layer_idx, device=src.device)
        return out


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        layers = []
        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            out_dim = output_dim if layer_idx == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack((x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h), dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0), dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area.clamp(min=1e-6)


def normalize_boxes_xyxy(boxes: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    height, width = size.to(boxes.device).float()
    scale = torch.tensor([width, height, width, height], device=boxes.device)
    return boxes / scale.clamp(min=1.0)


class HungarianMatcher(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.cost_class = config.cost_class
        self.cost_bbox = config.cost_bbox
        self.cost_mask = config.cost_mask

    @torch.no_grad()
    def forward(
        self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        bs, _num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].softmax(-1)
        out_bbox = outputs["pred_boxes"]
        out_masks = outputs["pred_masks"].sigmoid()
        indices = []
        for batch_idx in range(bs):
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
            cost_mask = self._mask_cost(
                out_masks[batch_idx], targets[batch_idx]["masks"].to(out_prob.device)
            )
            cost = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_mask * cost_mask
            )
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            indices.append(
                (
                    torch.as_tensor(row_ind, dtype=torch.int64, device=out_prob.device),
                    torch.as_tensor(col_ind, dtype=torch.int64, device=out_prob.device),
                )
            )
        return indices

    def _mask_cost(self, pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        if target_masks.numel() == 0:
            return pred_masks.new_zeros((pred_masks.shape[0], 0))
        pred_masks, target_masks = resize_masks_for_loss(
            pred_masks, target_masks, max_size=self.config.mask_loss_size
        )
        pred_flat = pred_masks.flatten(1)
        tgt_flat = target_masks.flatten(1)
        numerator = 2 * torch.einsum("qd,nd->qn", pred_flat, tgt_flat)
        denominator = pred_flat.sum(-1)[:, None] + tgt_flat.sum(-1)[None, :]
        return 1 - (numerator + 1) / (denominator + 1)


class DETRCriterion(nn.Module):
    def __init__(self, num_classes: int, config: ModelConfig) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.config = config
        self.matcher = HungarianMatcher(config)
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[0] = config.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(
        self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(target["labels"]) for target in targets)
        num_boxes_t = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=outputs["pred_logits"].device
        )
        num_boxes_f = torch.clamp(num_boxes_t, min=1).item()
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes_f))
        losses.update(self.loss_masks(outputs, targets, indices, num_boxes_f))
        losses["loss"] = (
            losses["loss_ce"]
            + self.config.bbox_loss_coef * losses["loss_bbox"]
            + self.config.giou_loss_coef * losses["loss_giou"]
            + self.config.mask_loss_coef * losses["loss_mask"]
            + self.config.dice_loss_coef * losses["loss_dice"]
        )
        return losses

    def loss_labels(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        src_logits = outputs["pred_logits"]
        target_classes = torch.zeros(
            src_logits.shape[:2], dtype=torch.int64, device=src_logits.device
        )
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() > 0:
                target_classes[batch_idx, src_idx] = targets[batch_idx]["labels"].to(
                    src_logits.device
                )[tgt_idx]
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {"loss_ce": loss_ce}

    def loss_boxes(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
    ) -> dict[str, torch.Tensor]:
        src_boxes_list = []
        tgt_boxes_list = []
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            src_boxes_list.append(outputs["pred_boxes"][batch_idx, src_idx])
            tgt_boxes = normalize_boxes_xyxy(
                targets[batch_idx]["boxes"].to(outputs["pred_boxes"].device),
                targets[batch_idx]["size"],
            )
            tgt_boxes_list.append(box_xyxy_to_cxcywh(tgt_boxes)[tgt_idx])
        if not src_boxes_list:
            zero = outputs["pred_boxes"].sum() * 0.0
            return {"loss_bbox": zero, "loss_giou": zero}
        src_boxes = torch.cat(src_boxes_list, dim=0)
        tgt_boxes = torch.cat(tgt_boxes_list, dim=0)
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="sum") / num_boxes
        src_xyxy = box_cxcywh_to_xyxy(src_boxes)
        tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
        loss_giou = (1 - torch.diag(generalized_box_iou(src_xyxy, tgt_xyxy))).sum() / num_boxes
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def loss_masks(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
    ) -> dict[str, torch.Tensor]:
        pred_masks_list = []
        tgt_masks_list = []
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            pred_masks = outputs["pred_masks"][batch_idx, src_idx]
            tgt_masks = targets[batch_idx]["masks"].to(outputs["pred_masks"].device)[tgt_idx]
            pred_masks, tgt_masks = resize_masks_for_loss(
                pred_masks, tgt_masks, max_size=self.config.mask_loss_size
            )
            pred_masks_list.append(pred_masks)
            tgt_masks_list.append(tgt_masks)
        if not pred_masks_list:
            zero = outputs["pred_masks"].sum() * 0.0
            return {"loss_mask": zero, "loss_dice": zero}
        pred_masks = torch.cat(pred_masks_list, dim=0)
        tgt_masks = torch.cat(tgt_masks_list, dim=0)
        loss_mask = F.binary_cross_entropy_with_logits(pred_masks, tgt_masks, reduction="mean")
        loss_dice = dice_loss(pred_masks.sigmoid(), tgt_masks, num_boxes)
        return {"loss_mask": loss_mask, "loss_dice": loss_dice}


def resize_masks_for_loss(
    pred_masks: torch.Tensor, target_masks: torch.Tensor, max_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize predicted and target masks to a bounded size for memory-conscious losses.

    Targets come in as uint8 from the dataloader; we downsample at uint8 (much
    smaller transfer) and only cast to float once at the reduced resolution.
    """

    height, width = pred_masks.shape[-2:]
    if max(height, width) > max_size > 0:
        scale = max_size / float(max(height, width))
        new_h = max(1, round(height * scale))
        new_w = max(1, round(width * scale))
        size = (new_h, new_w)
    else:
        size = (height, width)
    if pred_masks.shape[-2:] != size:
        pred_masks = F.interpolate(
            pred_masks[:, None], size=size, mode="bilinear", align_corners=False
        )[:, 0]
    if target_masks.shape[-2:] != size:
        target_masks = F.interpolate(target_masks[:, None], size=size, mode="nearest")[:, 0]
    target_masks = target_masks.float()
    return pred_masks, target_masks


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_boxes: float) -> torch.Tensor:
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def build_model_and_criterion(
    cfg: Any, num_classes: int
) -> tuple[TinyDETRSegmenter, DETRCriterion]:
    model_cfg = model_config_from_cfg(cfg)
    model = TinyDETRSegmenter(num_classes=num_classes, config=model_cfg)
    criterion = DETRCriterion(num_classes=num_classes, config=model_cfg)
    return model, criterion


@torch.no_grad()
def outputs_to_predictions(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    label_to_cat_id: dict[int, int],
    score_threshold: float = 0.05,
    max_detections: int = 100,
    mask_threshold: float = 0.5,
    include_masks: bool = True,
    target_size_key: str = "orig_size",
) -> list[dict[str, Any]]:
    prob = outputs["pred_logits"].softmax(-1)
    scores, labels = prob[..., 1:].max(-1)
    labels = labels + 1
    boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"])
    masks = outputs["pred_masks"].sigmoid() if include_masks else None
    predictions: list[dict[str, Any]] = []
    for batch_idx, target in enumerate(targets):
        size_tensor = target.get(target_size_key, target["orig_size"])
        height, width = [int(v) for v in size_tensor.tolist()]
        image_id = int(target["image_id"].item())
        keep = scores[batch_idx] >= score_threshold
        if keep.sum() == 0:
            continue
        kept_scores = scores[batch_idx][keep]
        kept_labels = labels[batch_idx][keep]
        kept_boxes = boxes[batch_idx][keep]
        kept_masks = masks[batch_idx][keep] if masks is not None else None
        order = torch.argsort(kept_scores, descending=True)[:max_detections]
        ordered_scores = kept_scores[order]
        ordered_labels = kept_labels[order]
        ordered_boxes = kept_boxes[order]
        ordered_masks = kept_masks[order].detach().cpu() if kept_masks is not None else None
        for idx in range(int(order.numel())):
            label = int(ordered_labels[idx].item())
            cat_id = int(label_to_cat_id.get(label, label))
            box = ordered_boxes[idx].clone()
            scale = torch.tensor([width, height, width, height], device=box.device)
            box = (box * scale).clamp(min=0)
            box[0::2] = box[0::2].clamp(max=width)
            box[1::2] = box[1::2].clamp(max=height)
            pred = {
                "image_id": image_id,
                "category_id": cat_id,
                "score": float(ordered_scores[idx].item()),
                "bbox_xyxy": [float(v) for v in box.detach().cpu().tolist()],
            }
            if ordered_masks is not None:
                mask_prob = ordered_masks[idx]
                if tuple(mask_prob.shape[-2:]) != (height, width):
                    mask_prob = F.interpolate(
                        mask_prob[None, None],
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                pred["mask"] = (mask_prob >= mask_threshold).detach().cpu().numpy().astype(bool)
            predictions.append(pred)
    return predictions
