"""DINOv3 ViT-S/16 + ViTDet Simple Feature Pyramid + Mask R-CNN heads.

Why this exists: the Swin-T baseline plateaus at mask mAP ~0.09 on 25%-COCO
after 100 epochs because the backbone features aren't strong enough on small
data. DINOv3's self-supervised features carry rich object-aware information
out of the box, so the downstream detection heads converge in a fraction of
the wall time and reach a much higher ceiling.

Architecture in one paragraph:
* Backbone: ``timm.create_model("vit_small_patch16_dinov3", pretrained=True)``.
  Single-scale ViT, 12 transformer blocks, embed_dim=384, patch=16. At 512×512
  input it produces a 32×32 feature grid plus prefix tokens (CLS + 4 register
  tokens) that we discard before reshaping back to NCHW.
* Feature pyramid: Li et al. (2022) "Simple Feature Pyramid" — from one feature
  map at stride 16, build P2..P5 by deconv-upsampling for fine levels and
  strided-conv downsampling for the coarse level. Cheap; competitive with FPN.
* P6 comes from ``LastLevelMaxPool`` on P5, giving the 5 levels Mask R-CNN's
  default anchor pyramid expects.
* Heads: torchvision ``MaskRCNN`` end-to-end (RPN + box head + mask head), same
  configuration as the Swin baseline.
* Attention: a single ``forward_intermediates`` call returns NCHW features at
  the first / middle / last blocks; we expose channel-mean of |features| at
  each as a spatial focus map. Three depths reveal which features the model
  actually relies on (early-layer texture cues vs. late-layer object cues) —
  the central readout for shortcut-learning analysis.
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
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

DINOV3_MASK_RCNN_MODEL_TYPE = "dinov3_maskrcnn"


@dataclass
class DINOv3MaskRCNNConfig:
    """Hyperparameters for ``DINOv3MaskRCNNSegmenter``."""

    # ViT backbone — any timm DINOv3 variant works; default is the small one
    # for a good speed/quality trade-off on 25%-COCO. Patch-16 keeps 512 input
    # divisible (32x32 tokens), avoiding the resize that patch-14 forces.
    backbone_name: str = "vit_small_patch16_dinov3"
    backbone_pretrained: bool = True
    backbone_freeze: bool = False
    # OFF by default for DINOv3. Grad checkpointing trades ~25% compute for
    # roughly half the activation memory; on a 96 GB card with batch=128 we
    # have headroom to skip it and get the wall-clock back.
    backbone_grad_checkpointing: bool = False
    # Stochastic depth on the ViT. 0.1 is a no-cost regularizer that
    # consistently helps when fine-tuning a pretrained ViT on small data -
    # 25%-COCO is exactly that regime. Set to 0.0 to disable.
    backbone_drop_path_rate: float = 0.1
    # ``torch.compile`` the backbone forward. Compile of the full Mask R-CNN
    # tree is unsafe because RPN/RoI ops have data-dependent shapes, so we
    # scope it to the fixed-shape ViT body only.
    compile_backbone: bool = True
    compile_mode: str = "default"  # default | reduce-overhead | max-autotune
    image_size: int = 512
    fpn_out_channels: int = 256
    # Channels for the 4-conv Mask R-CNN mask head (256 in torchvision's
    # default; 128 is the standard "lite" setting). The mask head dominates
    # the per-iteration cost at large batch + many positives, and DINOv3's
    # features are strong enough that the extra capacity stops mattering
    # very early. Drops ~25% off epoch wall time at ~0 mAP cost in our regime.
    mask_head_channels: int = 128
    mask_head_num_convs: int = 4
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    # Detection-head settings (memory-conscious defaults from the Swin model)
    box_detections_per_image: int = 100
    box_score_thresh: float = 0.0
    box_nms_thresh: float = 0.5
    box_batch_size_per_image: int = 128
    box_positive_fraction: float = 0.25
    rpn_batch_size_per_image: int = 128
    rpn_positive_fraction: float = 0.5
    rpn_pre_nms_top_n_train: int = 1000
    rpn_pre_nms_top_n_test: int = 1000
    rpn_post_nms_top_n_train: int = 1000
    rpn_post_nms_top_n_test: int = 1000
    anchor_sizes: tuple[tuple[int, ...], ...] = (
        (32,),
        (64,),
        (128,),
        (256,),
        (512,),
    )
    anchor_aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0)
    # ViT block indices whose feature-norms are exposed as attention maps. The
    # default samples first / middle / last for a 12-block ViT-S; the wrapper
    # clips out-of-range indices for other ViT sizes automatically.
    attention_block_indices: tuple[int, ...] = (0, 5, 11)
    attention_layer_names: tuple[str, ...] = ("first", "middle", "last")
    loss_weights: dict[str, float] = field(default_factory=dict)


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def model_config_from_cfg(cfg: Any) -> DINOv3MaskRCNNConfig:
    base = DINOv3MaskRCNNConfig()
    image_mean = _cfg_get(cfg, "image_mean", base.image_mean)
    image_std = _cfg_get(cfg, "image_std", base.image_std)
    anchor_sizes = _cfg_get(cfg, "anchor_sizes", base.anchor_sizes)
    anchor_aspects = _cfg_get(cfg, "anchor_aspect_ratios", base.anchor_aspect_ratios)
    attention_indices = _cfg_get(cfg, "attention_block_indices", base.attention_block_indices)
    attention_names = _cfg_get(cfg, "attention_layer_names", base.attention_layer_names)
    loss_weights = _cfg_get(cfg, "loss_weights", base.loss_weights)
    return DINOv3MaskRCNNConfig(
        backbone_name=str(_cfg_get(cfg, "backbone_name", base.backbone_name)),
        backbone_pretrained=bool(_cfg_get(cfg, "backbone_pretrained", base.backbone_pretrained)),
        backbone_freeze=bool(_cfg_get(cfg, "backbone_freeze", base.backbone_freeze)),
        backbone_grad_checkpointing=bool(
            _cfg_get(cfg, "backbone_grad_checkpointing", base.backbone_grad_checkpointing)
        ),
        backbone_drop_path_rate=float(
            _cfg_get(cfg, "backbone_drop_path_rate", base.backbone_drop_path_rate)
        ),
        compile_backbone=bool(_cfg_get(cfg, "compile_backbone", base.compile_backbone)),
        compile_mode=str(_cfg_get(cfg, "compile_mode", base.compile_mode)),
        image_size=int(_cfg_get(cfg, "image_size", base.image_size)),
        fpn_out_channels=int(_cfg_get(cfg, "fpn_out_channels", base.fpn_out_channels)),
        mask_head_channels=int(_cfg_get(cfg, "mask_head_channels", base.mask_head_channels)),
        mask_head_num_convs=int(_cfg_get(cfg, "mask_head_num_convs", base.mask_head_num_convs)),
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
        attention_block_indices=tuple(int(v) for v in attention_indices),
        attention_layer_names=tuple(str(v) for v in attention_names),
        loss_weights=dict(loss_weights) if loss_weights else {},
    )


class DINOv3Backbone(nn.Module):
    """timm DINOv3 ViT wrapped to return final NCHW features + multi-block attention.

    A single call to ``forward_intermediates`` does double duty: the final
    block's feature map is what the detection head sees, and the earlier
    blocks' feature maps are what we render as attention overlays. One forward
    pass — no second backbone call for visualization.
    """

    def __init__(self, config: DINOv3MaskRCNNConfig) -> None:
        super().__init__()
        # ``drop_path_rate`` is forwarded to timm so each transformer block
        # randomly drops with a linearly-scheduled probability, equivalent to
        # the standard stochastic-depth regularizer. timm silently accepts the
        # kwarg even on models that don't honour it, so this stays safe.
        self.body = timm.create_model(
            config.backbone_name,
            pretrained=config.backbone_pretrained,
            num_classes=0,
            img_size=config.image_size,
            drop_path_rate=float(config.backbone_drop_path_rate),
        )
        self.embed_dim = int(self.body.embed_dim)
        self.num_blocks = len(self.body.blocks)
        # Clip caller-provided indices to the actual block count of the variant
        # they chose (small=12, base=12, large=24, etc.) so configs stay portable.
        clipped: list[int] = []
        for idx in config.attention_block_indices:
            idx = int(idx)
            if idx < 0:
                idx = self.num_blocks + idx
            idx = max(0, min(self.num_blocks - 1, idx))
            if idx not in clipped:
                clipped.append(idx)
        # The final block is always part of the request — it's both the
        # detection feature and the "last" attention layer.
        if (self.num_blocks - 1) not in clipped:
            clipped.append(self.num_blocks - 1)
        self.attention_block_indices = tuple(sorted(clipped))
        self.attention_layer_names = tuple(config.attention_layer_names[: len(clipped)])
        if config.backbone_grad_checkpointing and hasattr(self.body, "set_grad_checkpointing"):
            self.body.set_grad_checkpointing(True)
        if config.backbone_freeze:
            for p in self.body.parameters():
                p.requires_grad = False
        self._latest_attentions: list[torch.Tensor] | None = None
        self._latest_attention_indices: tuple[int, ...] = self.attention_block_indices
        # The compiled fast path is built once and used in every forward. We
        # compile the ``forward_intermediates`` slice with the chosen block
        # indices baked in so torch.compile sees a static graph. Runtime
        # compile failures are handled in ``forward`` by falling back to eager.
        self._compiled_forward = None
        if config.compile_backbone and hasattr(torch, "compile"):
            indices = list(self.attention_block_indices)

            def _runner(x: torch.Tensor, body=self.body, indices=indices) -> list[torch.Tensor]:
                return body.forward_intermediates(
                    x,
                    indices=indices,
                    output_fmt="NCHW",
                    intermediates_only=True,
                    norm=True,
                )

            try:
                self._compiled_forward = torch.compile(_runner, mode=config.compile_mode)
            except Exception:  # pragma: no cover - depends on torch + hardware
                self._compiled_forward = None

    def _forward_intermediates_eager(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.body.forward_intermediates(
            x,
            indices=list(self.attention_block_indices),
            output_fmt="NCHW",
            intermediates_only=True,
            norm=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._compiled_forward is None:
            intermediates = self._forward_intermediates_eager(x)
        else:
            try:
                intermediates = self._compiled_forward(x)
            except Exception:  # pragma: no cover - depends on torch + hardware
                self._compiled_forward = None
                intermediates = self._forward_intermediates_eager(x)
        # Per-block channel-mean of |features|. This is a feature-norm map —
        # the standard interpretability proxy for DINO-style backbones, where
        # high values mark spatially-discriminative regions. Computed under
        # no_grad so it never participates in the autograd tape.
        with torch.no_grad():
            self._latest_attentions = [f.detach().abs().mean(dim=1) for f in intermediates]
        # The detection pipeline only needs the final-block feature.
        return intermediates[-1]


class SimpleFeaturePyramid(nn.Module):
    """ViTDet single-scale → multi-scale adapter (Li et al. 2022).

    Inputs: one feature map at stride 16 with ``in_channels`` channels.
    Outputs: an ``OrderedDict`` of P2..P5 (strides 4, 8, 16, 32), each with
    ``out_channels`` channels. The downstream P6 level Mask R-CNN expects is
    produced by ``LastLevelMaxPool`` after this adapter — kept separate so the
    adapter remains drop-in for non-MaskRCNN heads.

    Each branch is a small conv stack rather than a sequence of FPN top-down
    additions, which is the whole point of "simple": no cross-level fusion is
    needed because the ViT features are already highly fused globally.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # P2 (stride 4): 4× upsample. Two deconvs with a GN+GELU bottleneck so
        # the spatial-doubling layers can learn something useful instead of a
        # blind nearest-neighbour-style upsample.
        self.p2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.GroupNorm(32, in_channels),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
        )
        # P3 (stride 8): 2× upsample.
        self.p3 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        # P4 (stride 16): identity.
        self.p4 = nn.Identity()
        # P5 (stride 32): 2× downsample via strided conv.
        self.p5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

        # Per-level 1×1 lateral + 3×3 smooth, mirroring FPN's final two ops.
        # The lateral lets each level pick its own channel mix; the smooth
        # removes upsampling aliasing.
        def lateral_smooth() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.GroupNorm(32, out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(32, out_channels),
            )

        self.lateral_p2 = lateral_smooth()
        self.lateral_p3 = lateral_smooth()
        self.lateral_p4 = lateral_smooth()
        self.lateral_p5 = lateral_smooth()
        self.out_channels = int(out_channels)

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict(
            [
                ("0", self.lateral_p2(self.p2(x))),
                ("1", self.lateral_p3(self.p3(x))),
                ("2", self.lateral_p4(self.p4(x))),
                ("3", self.lateral_p5(self.p5(x))),
            ]
        )


class _ViTBackboneAdapter(nn.Module):
    """Glue between ``DINOv3Backbone`` + ``SimpleFeaturePyramid`` and torchvision's
    ``MaskRCNN``. Holds the backbone/pyramid by reference (``object.__setattr__``)
    so parameter paths stay canonical under the parent wrapper.
    """

    def __init__(
        self,
        backbone: DINOv3Backbone,
        pyramid: SimpleFeaturePyramid,
        extra_block: LastLevelMaxPool,
        out_channels: int,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_backbone_ref", backbone)
        object.__setattr__(self, "_pyramid_ref", pyramid)
        object.__setattr__(self, "_extra_block_ref", extra_block)
        self.out_channels = int(out_channels)

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        feat = self._backbone_ref(x)
        levels = self._pyramid_ref(feat)
        # LastLevelMaxPool turns 4 levels into 5 by max-pooling the last one.
        # Its signature is (results, x, names) → expanded results+names; we
        # only need the result list back.
        names = list(levels.keys())
        results = list(levels.values())
        x_extra: list[torch.Tensor] = []
        results, names = self._extra_block_ref(results, x_extra, names)
        return OrderedDict(zip(names, results, strict=False))


class DINOv3MaskRCNNSegmenter(nn.Module):
    """Top-level wrapper matching the project's unified model contract.

    ``forward(images, targets=None, return_attention=False) -> dict``

    Train mode (``self.training``): targets required; returns
    ``{"losses": dict, "model_type": "dinov3_maskrcnn"}``.

    Eval mode: returns
    ``{"predictions": [...], "model_type": "dinov3_maskrcnn"}`` plus, when
    ``return_attention=True``, ``"multi_layer_attention"`` (a dict mapping
    layer name → (B, H, W) heatmap) and DETR-shaped attention keys
    (``cross_attention``/``attention_hw``/``pred_logits``) for the existing
    single-layer visualizer.
    """

    def __init__(self, num_classes: int, config: DINOv3MaskRCNNConfig) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.config = config
        # Pretrained backbone (low-LR group) is registered FIRST so
        # ``named_parameters`` lists each ViT parameter under ``backbone.body.*``.
        self.backbone = DINOv3Backbone(config)
        self.pyramid = SimpleFeaturePyramid(self.backbone.embed_dim, config.fpn_out_channels)
        self.extra_block = LastLevelMaxPool()
        adapter = _ViTBackboneAdapter(
            self.backbone, self.pyramid, self.extra_block, config.fpn_out_channels
        )
        anchor_sizes = tuple(tuple(s) for s in config.anchor_sizes)
        aspect_ratios = (tuple(config.anchor_aspect_ratios),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        featmap_names = [str(i) for i in range(len(anchor_sizes) - 1)]
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=featmap_names,
            output_size=7,
            sampling_ratio=2,
        )
        mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=featmap_names,
            output_size=14,
            sampling_ratio=2,
        )
        if config.mask_head_channels < 1:
            raise ValueError("mask_head_channels must be >= 1.")
        if config.mask_head_num_convs < 1:
            raise ValueError("mask_head_num_convs must be >= 1.")
        # Lighter mask head: by default torchvision builds a 256-channel
        # 4-conv stack which is the cost driver per training step (it
        # processes positive RoIs * num_classes mask logits at 14x14 -> 28x28).
        # With DINOv3 features 128 channels is empirically sufficient: the
        # head is consuming feature maps that are already highly informative.
        # When ``mask_predictor`` is supplied to MaskRCNN we must also supply
        # ``box_predictor`` and pass ``num_classes=None``, so we build both
        # heads explicitly here to keep the dependency clear.
        box_repr_size = 1024
        box_head = TwoMLPHead(
            config.fpn_out_channels * box_roi_pool.output_size[0] * box_roi_pool.output_size[1],
            box_repr_size,
        )
        box_predictor = FastRCNNPredictor(box_repr_size, num_classes + 1)
        mask_layers = tuple([config.mask_head_channels] * config.mask_head_num_convs)
        mask_head = MaskRCNNHeads(config.fpn_out_channels, mask_layers, dilation=1)
        mask_predictor = MaskRCNNPredictor(
            in_channels=mask_layers[-1],
            dim_reduced=mask_layers[-1],
            num_classes=num_classes + 1,
        )
        self.detector = MaskRCNN(
            backbone=adapter,
            num_classes=None,
            min_size=config.image_size,
            max_size=config.image_size,
            image_mean=list(config.image_mean),
            image_std=list(config.image_std),
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            mask_roi_pool=mask_roi_pool,
            mask_head=mask_head,
            mask_predictor=mask_predictor,
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
                entry["masks"] = torch.zeros((0, H, W), dtype=torch.uint8, device=boxes.device)
            prepared.append(entry)
        return prepared

    def _attention_outputs(self, return_attention: bool) -> dict[str, Any]:
        """Translate captured per-block feature-norm maps into the output dict.

        Three things are exposed:

        * ``multi_layer_attention`` — dict ``{layer_name: (B, H, W) heatmap}``.
          This is the new, multi-depth view that the fast-eval renderer reads
          to log first/middle/last overlays for shortcut analysis.
        * ``cross_attention`` + ``attention_hw`` + ``pred_logits`` — the
          single-map DETR-shaped fallback for the older single-layer
          ``attention_for_top_query`` visualizer. We use the LAST block's map
          here because that's the one closest to the detection decision.
        """

        if not return_attention:
            return {}
        atts = self.backbone._latest_attentions
        if not atts:
            return {}
        # Map block index → friendly layer name (first / middle / last).
        names = self.backbone.attention_layer_names
        attn_dict = {name: att for name, att in zip(names, atts, strict=False)}
        last = atts[-1]
        B, H, W = last.shape
        cross_attn = last.view(B, 1, 1, H * W)
        fake_logits = torch.zeros(B, 1, self.num_classes + 1, device=last.device, dtype=last.dtype)
        fake_logits[..., 1] = 10.0
        return {
            "multi_layer_attention": attn_dict,
            "multi_layer_attention_block_indices": tuple(
                int(i) for i in self.backbone.attention_block_indices
            ),
            "cross_attention": cross_attn,
            "attention_hw": torch.tensor([H, W], device=last.device),
            "attention_layer_index": torch.tensor(
                int(self.backbone.attention_block_indices[-1]), device=last.device
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
                raise ValueError("DINOv3MaskRCNNSegmenter.forward requires targets in train mode.")
            tv_targets = self._prepare_targets(targets)
            losses = self.detector(images, tv_targets)
            out: dict[str, Any] = {
                "model_type": DINOV3_MASK_RCNN_MODEL_TYPE,
                "losses": losses,
            }
            out.update(self._attention_outputs(return_attention))
            return out
        predictions = self.detector(images)
        out = {
            "model_type": DINOV3_MASK_RCNN_MODEL_TYPE,
            "predictions": predictions,
        }
        out.update(self._attention_outputs(return_attention))
        return out


class DINOv3MaskRCNNCriterion(nn.Module):
    """Thin wrapper that sums torchvision's loss dict into the project's shape.

    Mirrors ``MaskRCNNCriterion`` — the only reason this exists as a separate
    class is so the model factory can return symmetric ``(model, criterion)``
    pairs across model families without ambiguity.
    """

    def __init__(self, loss_weights: dict[str, float] | None = None) -> None:
        super().__init__()
        self.loss_weights: dict[str, float] = dict(loss_weights or {})

    def forward(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        del targets
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
) -> tuple[DINOv3MaskRCNNSegmenter, DINOv3MaskRCNNCriterion]:
    del train_cfg
    model_cfg = model_config_from_cfg(cfg)
    model = DINOv3MaskRCNNSegmenter(num_classes=num_classes, config=model_cfg)
    criterion = DINOv3MaskRCNNCriterion(loss_weights=model_cfg.loss_weights)
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
    class_agnostic_nms_iou: float | None = None,
) -> list[dict[str, Any]]:
    """Convert torchvision MaskRCNN predictions to the project's COCO-eval shape.

    Identical to the Swin model's converter, with one extra optional pass:
    when ``class_agnostic_nms_iou`` is set, a second NMS that ignores class
    labels runs on top of torchvision's per-class NMS. This is what the W&B
    visualization layer turns on to suppress the "many overlapping low-score
    boxes around the same object" the user sees in fast-eval logs — the
    duplicates come from per-class NMS letting through cat/dog/etc. variants
    of the same region.
    """

    from torchvision.ops import nms as torchvision_nms

    predictions: list[dict[str, Any]] = []
    raw = outputs.get("predictions")
    if not raw:
        return predictions
    for per_image, target in zip(raw, targets, strict=False):
        size_tensor = target.get(target_size_key, target["orig_size"])
        height, width = [int(v) for v in size_tensor.tolist()]
        image_id = int(target["image_id"].item())
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
        order = torch.argsort(scores[keep], descending=True)[:max_detections]
        kept_scores = scores[keep][order]
        kept_labels = labels[keep][order]
        kept_boxes = boxes[keep][order]
        kept_masks = masks[keep][order] if masks is not None and masks.numel() > 0 else None
        if class_agnostic_nms_iou is not None and kept_boxes.numel() > 0:
            survivors = torchvision_nms(
                kept_boxes.float(), kept_scores.float(), float(class_agnostic_nms_iou)
            )
            kept_scores = kept_scores[survivors]
            kept_labels = kept_labels[survivors]
            kept_boxes = kept_boxes[survivors]
            if kept_masks is not None:
                kept_masks = kept_masks[survivors]
        for idx in range(int(kept_scores.numel())):
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
                pred["mask"] = (mask_prob >= mask_threshold).detach().cpu().numpy().astype(bool)
            predictions.append(pred)
    return predictions


__all__ = [
    "DINOV3_MASK_RCNN_MODEL_TYPE",
    "DINOv3Backbone",
    "DINOv3MaskRCNNConfig",
    "DINOv3MaskRCNNCriterion",
    "DINOv3MaskRCNNSegmenter",
    "SimpleFeaturePyramid",
    "build_model_and_criterion",
    "outputs_to_predictions",
]
