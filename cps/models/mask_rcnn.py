"""Mask R-CNN with a timm Swin-Tiny backbone for the copy-paste study.

The earlier DETR baseline failed to converge on 25%-COCO in <300 epochs. This
module is the fast-converging replacement: torchvision's mature Mask R-CNN
detection stack on top of a Swin-Tiny ImageNet-pretrained backbone (timm). It
typically reaches usable mask mAP within ~12 epochs.

Backbone-level spatial focus is captured as a feature-norm map at the deepest
Swin stage and exposed through the same ``outputs`` dict shape used by DETR so
the existing attention-visualization pipeline (cps/analysis/attention.py and
cps/data/visualization.py:save_attention_overlay) keeps working unchanged. This
is what lets the copy-paste study read off shortcut behaviour from the model.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

MASK_RCNN_MODEL_TYPE = "maskrcnn"


@dataclass
class MaskRCNNConfig:
    """Hyperparameters for ``MaskRCNNSegmenter``. Read from Hydra cfg.model."""

    backbone_name: str = "swin_tiny_patch4_window7_224"
    backbone_pretrained: bool = True
    backbone_freeze: bool = False
    # If True, calls timm's ``set_grad_checkpointing(True)`` on the Swin body.
    # This is the biggest backbone-side VRAM win for large batches: Swin
    # otherwise keeps every block's activations on the forward tape. Trades
    # extra compute for lower activation memory.
    backbone_grad_checkpointing: bool = True
    image_size: int = 512
    fpn_out_channels: int = 256
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    # Detection-head settings
    box_detections_per_image: int = 100
    box_score_thresh: float = 0.0  # let outputs_to_predictions do the filtering
    box_nms_thresh: float = 0.5
    # Sampled RoIs per image used for box + mask loss.
    # Torchvision's default is 512. At batch=64 that sends 32k RoIs through the
    # box head and up to 8k positive RoIs through the mask head, producing very
    # large mask-logit and autograd tensors. 128 is the standard memory-friendly
    # value used for large-batch Mask R-CNN training; positives still cover the
    # gradient mix well at fraction=0.25.
    box_batch_size_per_image: int = 128
    box_positive_fraction: float = 0.25
    # Sampled anchors per image for RPN loss. Torchvision default is 256;
    # halving lets us double batch size without hurting RPN learnability.
    rpn_batch_size_per_image: int = 128
    rpn_positive_fraction: float = 0.5
    # Top-K anchor candidates before NMS, per FPN level. The torchvision
    # default is 2000 per level, or 10000 boxes/image with this 5-level FPN.
    # At batch=64 that is 640k candidate boxes carried into NMS each step.
    rpn_pre_nms_top_n_train: int = 1000
    rpn_pre_nms_top_n_test: int = 1000
    # Top-K proposals after NMS that are passed to the RoI heads. The post-NMS
    # number multiplied by batch size sets the pool the box-sampler draws from;
    # 1000 is the standard.
    rpn_post_nms_top_n_train: int = 1000
    rpn_post_nms_top_n_test: int = 1000
    # Anchor pyramid: 5 levels (P2..P6 from LastLevelMaxPool)
    anchor_sizes: tuple[tuple[int, ...], ...] = (
        (32,), (64,), (128,), (256,), (512,),
    )
    anchor_aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0)
    # Which Swin stage's feature norm to use as the spatial-focus map.
    # 0..3, where 3 is the deepest (stride 32) — strongest semantic signal.
    attention_stage: int = 3
    # Loss weighting passed through to ``MaskRCNNCriterion.forward``.
    loss_weights: dict[str, float] = field(default_factory=dict)


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def model_config_from_cfg(cfg: Any) -> MaskRCNNConfig:
    base = MaskRCNNConfig()
    image_mean = _cfg_get(cfg, "image_mean", base.image_mean)
    image_std = _cfg_get(cfg, "image_std", base.image_std)
    anchor_sizes = _cfg_get(cfg, "anchor_sizes", base.anchor_sizes)
    anchor_aspects = _cfg_get(cfg, "anchor_aspect_ratios", base.anchor_aspect_ratios)
    loss_weights = _cfg_get(cfg, "loss_weights", base.loss_weights)
    return MaskRCNNConfig(
        backbone_name=str(_cfg_get(cfg, "backbone_name", base.backbone_name)),
        backbone_pretrained=bool(_cfg_get(cfg, "backbone_pretrained", base.backbone_pretrained)),
        backbone_freeze=bool(_cfg_get(cfg, "backbone_freeze", base.backbone_freeze)),
        backbone_grad_checkpointing=bool(
            _cfg_get(cfg, "backbone_grad_checkpointing", base.backbone_grad_checkpointing)
        ),
        image_size=int(_cfg_get(cfg, "image_size", base.image_size)),
        fpn_out_channels=int(_cfg_get(cfg, "fpn_out_channels", base.fpn_out_channels)),
        image_mean=tuple(float(v) for v in image_mean),
        image_std=tuple(float(v) for v in image_std),
        box_detections_per_image=int(
            _cfg_get(cfg, "box_detections_per_image", base.box_detections_per_image)
        ),
        box_score_thresh=float(_cfg_get(cfg, "box_score_thresh", base.box_score_thresh)),
        box_nms_thresh=float(_cfg_get(cfg, "box_nms_thresh", base.box_nms_thresh)),
        box_batch_size_per_image=int(
            _cfg_get(cfg, "box_batch_size_per_image", base.box_batch_size_per_image)
        ),
        box_positive_fraction=float(
            _cfg_get(cfg, "box_positive_fraction", base.box_positive_fraction)
        ),
        rpn_batch_size_per_image=int(
            _cfg_get(cfg, "rpn_batch_size_per_image", base.rpn_batch_size_per_image)
        ),
        rpn_positive_fraction=float(
            _cfg_get(cfg, "rpn_positive_fraction", base.rpn_positive_fraction)
        ),
        rpn_pre_nms_top_n_train=int(
            _cfg_get(cfg, "rpn_pre_nms_top_n_train", base.rpn_pre_nms_top_n_train)
        ),
        rpn_pre_nms_top_n_test=int(
            _cfg_get(cfg, "rpn_pre_nms_top_n_test", base.rpn_pre_nms_top_n_test)
        ),
        rpn_post_nms_top_n_train=int(
            _cfg_get(cfg, "rpn_post_nms_top_n_train", base.rpn_post_nms_top_n_train)
        ),
        rpn_post_nms_top_n_test=int(
            _cfg_get(cfg, "rpn_post_nms_top_n_test", base.rpn_post_nms_top_n_test)
        ),
        anchor_sizes=tuple(tuple(int(v) for v in level) for level in anchor_sizes),
        anchor_aspect_ratios=tuple(float(v) for v in anchor_aspects),
        attention_stage=int(_cfg_get(cfg, "attention_stage", base.attention_stage)),
        loss_weights=dict(loss_weights) if loss_weights else {},
    )


class TimmSwinBackboneBody(nn.Module):
    """timm Swin features_only backbone returning NCHW feature maps.

    Also captures a per-image spatial-focus map (feature-norm) at a chosen stage
    for shortcut-learning analysis. The focus map is the channel-mean of the
    absolute activations, which is a standard, model-agnostic proxy for "where
    the backbone is responding strongly" — directly usable as a heatmap overlay.
    """

    def __init__(self, config: MaskRCNNConfig) -> None:
        super().__init__()
        self.body = timm.create_model(
            config.backbone_name,
            pretrained=config.backbone_pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=config.image_size,
        )
        self.feature_info = self.body.feature_info
        self._attention_stage = int(config.attention_stage)
        self._latest_attention: torch.Tensor | None = None
        if config.backbone_grad_checkpointing and hasattr(self.body, "set_grad_checkpointing"):
            # timm exposes ``set_grad_checkpointing(True)`` to recompute each
            # transformer block in backward instead of keeping its activations.
            # Cuts the Swin tape memory roughly in half for ~25% extra compute.
            self.body.set_grad_checkpointing(True)
        if config.backbone_freeze:
            for p in self.body.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats_nhwc = self.body(x)
        feats_nchw = [f.permute(0, 3, 1, 2).contiguous() for f in feats_nhwc]
        stage = self._attention_stage
        if 0 <= stage < len(feats_nchw):
            self._latest_attention = feats_nchw[stage].detach().abs().mean(dim=1)
        else:
            self._latest_attention = None
        return feats_nchw


class _BackboneAdapter(nn.Module):
    """Adapter exposed to torchvision ``MaskRCNN`` as its ``backbone`` argument.

    Holds the actual ``TimmSwinBackboneBody`` and FPN by *reference* (bypassing
    ``nn.Module`` child registration) so each parameter is still listed under a
    single canonical path on the parent wrapper. That lets the existing
    ``build_optimizer`` name-prefix split (``backbone.*`` → low LR, everything
    else → head LR) keep working without changes.
    """

    def __init__(self, backbone: TimmSwinBackboneBody, fpn: FeaturePyramidNetwork, out_channels: int) -> None:
        super().__init__()
        object.__setattr__(self, "_backbone_ref", backbone)
        object.__setattr__(self, "_fpn_ref", fpn)
        self.out_channels = int(out_channels)

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        feats = self._backbone_ref(x)
        feat_dict: OrderedDict[str, torch.Tensor] = OrderedDict(
            (str(i), f) for i, f in enumerate(feats)
        )
        return self._fpn_ref(feat_dict)


class MaskRCNNSegmenter(nn.Module):
    """Top-level wrapper exposing the project's unified model contract.

    Forward signature mirrors ``TinyDETRSegmenter``:
        ``model(images, targets=None, return_attention=False) -> dict``

    Training (``self.training==True``): ``targets`` are required; the dict
    contains ``"losses"`` (torchvision's loss dict). The trainer's criterion
    just sums these — no Hungarian matcher, no manual loss computation.

    Eval: the dict contains ``"predictions"`` (torchvision's list of per-image
    dicts with ``boxes``/``labels``/``scores``/``masks``) and, when requested,
    a spatial-focus map under the DETR-shaped keys
    (``cross_attention``/``attention_hw``/``pred_logits`` so the existing
    ``attention_for_top_query`` visualizer works unchanged).
    """

    def __init__(self, num_classes: int, config: MaskRCNNConfig) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.config = config
        # `backbone` (pretrained, low LR) is registered FIRST so that
        # `named_parameters()` lists each Swin parameter under `backbone.body.*`
        # exactly once. The same module is reused via reference by `detector`
        # below, but PyTorch dedupes by identity so no duplicate path appears.
        self.backbone = TimmSwinBackboneBody(config)
        in_channels = list(self.backbone.feature_info.channels())
        # FPN (random init, head LR) sits OUTSIDE `backbone.*` so the optimizer
        # treats it as a head — that's important: FPN converges fastest with
        # the head's larger LR.
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels,
            out_channels=config.fpn_out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        adapter = _BackboneAdapter(self.backbone, self.fpn, config.fpn_out_channels)
        anchor_sizes = tuple(tuple(s) for s in config.anchor_sizes)
        aspect_ratios = (tuple(config.anchor_aspect_ratios),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        featmap_names = [str(i) for i in range(len(in_channels))]
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=featmap_names, output_size=7, sampling_ratio=2,
        )
        mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=featmap_names, output_size=14, sampling_ratio=2,
        )
        self.detector = MaskRCNN(
            backbone=adapter,
            num_classes=num_classes + 1,
            min_size=config.image_size,
            max_size=config.image_size,
            image_mean=list(config.image_mean),
            image_std=list(config.image_std),
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=box_roi_pool,
            mask_roi_pool=mask_roi_pool,
            box_detections_per_img=config.box_detections_per_image,
            box_score_thresh=config.box_score_thresh,
            box_nms_thresh=config.box_nms_thresh,
            box_batch_size_per_image=config.box_batch_size_per_image,
            box_positive_fraction=config.box_positive_fraction,
            rpn_batch_size_per_image=config.rpn_batch_size_per_image,
            rpn_positive_fraction=config.rpn_positive_fraction,
            rpn_pre_nms_top_n_train=config.rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test=config.rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train=config.rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test=config.rpn_post_nms_top_n_test,
        )

    @staticmethod
    def _prepare_targets(targets: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
        """Re-pack dataset targets into torchvision MaskRCNN's expected schema.

        Dataset masks arrive as uint8 at the (possibly resized) image resolution;
        torchvision accepts uint8/bool/float, but its internal loss expects mask
        floats in [0, 1]. We feed uint8 — MaskRCNN handles the cast.
        """

        prepared: list[dict[str, torch.Tensor]] = []
        for t in targets:
            boxes = t["boxes"].float()
            labels = t["labels"].long()
            masks = t.get("masks")
            entry: dict[str, torch.Tensor] = {"boxes": boxes, "labels": labels}
            if masks is not None and masks.numel() > 0:
                entry["masks"] = masks.to(dtype=torch.uint8)
            else:
                size = t.get("size")
                if size is not None:
                    H, W = int(size[0].item()), int(size[1].item())
                else:
                    H, W = 1, 1
                entry["masks"] = torch.zeros(
                    (0, H, W), dtype=torch.uint8, device=boxes.device
                )
            prepared.append(entry)
        return prepared

    def _attention_outputs(self, return_attention: bool) -> dict[str, torch.Tensor]:
        """Translate the captured feature-norm map into the DETR-shaped contract.

        Why mimic the DETR keys: ``cps/analysis/attention.py:attention_for_top_query``
        reads ``cross_attention``/``attention_hw``/``pred_logits`` and indexes the
        top-scoring "query". We expose a single synthetic query whose attention
        is the spatial-focus map, and a synthetic logits tensor that picks that
        query. The visualizer then renders identical to the DETR case.
        """

        if not return_attention:
            return {}
        att = self.backbone._latest_attention
        if att is None:
            return {}
        B, H, W = att.shape
        cross_attn = att.view(B, 1, 1, H * W)
        fake_logits = torch.zeros(
            B, 1, self.num_classes + 1, device=att.device, dtype=att.dtype
        )
        fake_logits[..., 1] = 10.0
        return {
            "cross_attention": cross_attn,
            "attention_hw": torch.tensor([H, W], device=att.device),
            "attention_layer_index": torch.tensor(
                self.config.attention_stage, device=att.device
            ),
            "pred_logits": fake_logits,
        }

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
        return_attention: bool = False,
    ) -> dict[str, Any]:
        if self.training:
            if targets is None:
                raise ValueError("MaskRCNNSegmenter.forward requires targets in train mode.")
            tv_targets = self._prepare_targets(targets)
            losses = self.detector(images, tv_targets)
            out: dict[str, Any] = {
                "model_type": MASK_RCNN_MODEL_TYPE,
                "losses": losses,
            }
            out.update(self._attention_outputs(return_attention))
            return out
        predictions = self.detector(images)
        out = {
            "model_type": MASK_RCNN_MODEL_TYPE,
            "predictions": predictions,
        }
        out.update(self._attention_outputs(return_attention))
        return out


class MaskRCNNCriterion(nn.Module):
    """Thin wrapper that sums torchvision's loss dict into the project's shape.

    Train mode: outputs carry a ``losses`` dict — we just sum (optionally
    re-weighted) and return ``{"loss": total, **components}``.
    Eval mode: torchvision MRCNN doesn't compute losses in eval mode without
    re-running the heads against targets, which would double the validation
    cost. We simply return an empty/zero loss so the validation loop's mAP
    is the source of truth.
    """

    def __init__(self, loss_weights: dict[str, float] | None = None) -> None:
        super().__init__()
        self.loss_weights: dict[str, float] = dict(loss_weights or {})

    def forward(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        del targets  # unused — loss is already computed by torchvision in train mode
        losses = outputs.get("losses")
        if losses is None:
            device = self._pick_device(outputs)
            return {"loss": torch.zeros((), device=device)}
        components: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None
        for key, value in losses.items():
            weight = float(self.loss_weights.get(key, 1.0))
            scaled = value * weight if weight != 1.0 else value
            components[key] = scaled
            total = scaled if total is None else total + scaled
        if total is None:
            device = self._pick_device(outputs)
            total = torch.zeros((), device=device)
        return {"loss": total, **components}

    @staticmethod
    def _pick_device(outputs: dict[str, Any]) -> torch.device:
        for value in outputs.values():
            if torch.is_tensor(value):
                return value.device
        return torch.device("cpu")


def build_model_and_criterion(
    cfg: Any, num_classes: int, train_cfg: Any | None = None
) -> tuple[MaskRCNNSegmenter, MaskRCNNCriterion]:
    del train_cfg  # no train-config-dependent knobs at construction time
    model_cfg = model_config_from_cfg(cfg)
    model = MaskRCNNSegmenter(num_classes=num_classes, config=model_cfg)
    criterion = MaskRCNNCriterion(loss_weights=model_cfg.loss_weights)
    return model, criterion


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
    """Convert torchvision MaskRCNN predictions to the project's COCO-eval shape."""

    predictions: list[dict[str, Any]] = []
    raw = outputs.get("predictions")
    if not raw:
        return predictions
    for batch_idx, (per_image, target) in enumerate(zip(raw, targets, strict=False)):
        size_tensor = target.get(target_size_key, target["orig_size"])
        height, width = [int(v) for v in size_tensor.tolist()]
        image_id = int(target["image_id"].item())
        # Input-resolution boxes (torchvision MRCNN returns at the resized
        # forward-pass resolution, which matches `target["size"]`). When the
        # caller asks for `orig_size` outputs we rescale; for `size` we don't.
        in_size = target.get("size", size_tensor)
        in_h, in_w = [int(v) for v in in_size.tolist()]
        scale_x = float(width) / float(max(in_w, 1))
        scale_y = float(height) / float(max(in_h, 1))
        scores = per_image.get("scores")
        labels = per_image.get("labels")
        boxes = per_image.get("boxes")
        masks = per_image.get("masks") if include_masks else None
        if scores is None or labels is None or boxes is None:
            continue
        keep = scores >= score_threshold
        if keep.sum() == 0:
            continue
        order_all = torch.argsort(scores[keep], descending=True)[:max_detections]
        kept_scores = scores[keep][order_all]
        kept_labels = labels[keep][order_all]
        kept_boxes = boxes[keep][order_all]
        kept_masks = (
            masks[keep][order_all] if masks is not None and masks.numel() > 0 else None
        )
        for idx in range(int(order_all.numel())):
            label = int(kept_labels[idx].item())
            cat_id = int(label_to_cat_id.get(label, label))
            box = kept_boxes[idx].detach().clone()
            box[0::2] = (box[0::2] * scale_x).clamp(min=0, max=float(width))
            box[1::2] = (box[1::2] * scale_y).clamp(min=0, max=float(height))
            pred: dict[str, Any] = {
                "image_id": image_id,
                "category_id": cat_id,
                "score": float(kept_scores[idx].item()),
                "bbox_xyxy": [float(v) for v in box.detach().cpu().tolist()],
            }
            if kept_masks is not None:
                # torchvision masks are (N, 1, H_in, W_in) float in [0,1].
                mask_prob = kept_masks[idx]
                if mask_prob.dim() == 3:
                    mask_prob = mask_prob[0]
                if tuple(mask_prob.shape[-2:]) != (height, width):
                    mask_prob = F.interpolate(
                        mask_prob[None, None],
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                pred["mask"] = (
                    (mask_prob >= mask_threshold).detach().cpu().numpy().astype(bool)
                )
            predictions.append(pred)
        del batch_idx
    return predictions


__all__ = [
    "MASK_RCNN_MODEL_TYPE",
    "MaskRCNNConfig",
    "MaskRCNNCriterion",
    "MaskRCNNSegmenter",
    "build_model_and_criterion",
    "outputs_to_predictions",
]
