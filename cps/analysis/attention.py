"""DETR decoder attention visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from cps.data.visualization import save_attention_overlay
from cps.paths import project_path


def attention_for_top_query(outputs: dict[str, torch.Tensor], batch_idx: int) -> np.ndarray | None:
    if "cross_attention" not in outputs or "attention_hw" not in outputs:
        return None
    logits = outputs["pred_logits"][batch_idx].softmax(-1)
    scores = logits[:, 1:].max(-1).values
    query_idx = int(torch.argmax(scores).detach().cpu().item())
    att = outputs["cross_attention"][batch_idx, :, query_idx, :].mean(0)
    h, w = [int(v) for v in outputs["attention_hw"].detach().cpu().tolist()]
    return att.reshape(h, w).detach().cpu().numpy()


def multi_layer_attention_for_image(
    outputs: dict[str, Any],
    batch_idx: int,
) -> dict[str, np.ndarray] | None:
    """Return per-layer attention maps for one image, keyed by layer name.

    Models that expose multi-depth attention (DINOv3 ViT, future ViT-DETR
    variants) populate ``outputs["multi_layer_attention"]`` as a dict of
    ``(B, H, W)`` tensors. This helper slices the batch dimension and converts
    each entry to a CPU NumPy array for visualisation.

    Falls back to ``attention_for_top_query`` for legacy single-layer models so
    callers don't need to special-case the older DETR/Swin attention shape.
    """

    multi = outputs.get("multi_layer_attention")
    if isinstance(multi, dict) and multi:
        out: dict[str, np.ndarray] = {}
        for name, tensor in multi.items():
            if not torch.is_tensor(tensor):
                continue
            if tensor.dim() == 2:
                out[str(name)] = tensor.detach().cpu().numpy()
            elif tensor.dim() == 3:
                out[str(name)] = tensor[batch_idx].detach().cpu().numpy()
        if out:
            return out
    fallback = attention_for_top_query(outputs, batch_idx)
    if fallback is not None:
        return {"attention": fallback}
    return None


def save_decoder_attention_maps(
    images: list[torch.Tensor],
    targets: list[dict[str, Any]],
    outputs: dict[str, torch.Tensor],
    output_dir: str | Path,
    batch_offset: int,
    max_images: int = 2,
) -> int:
    output_dir = project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for batch_idx, (image, target) in enumerate(zip(images, targets, strict=False)):
        if saved >= max_images:
            break
        attention = attention_for_top_query(outputs, batch_idx)
        if attention is None:
            continue
        height, width = [int(v) for v in target["size"].detach().cpu().tolist()]
        arr = (image[:, :height, :width].detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(
            0, 255
        )
        arr = arr.astype(np.uint8)
        image_id = int(target["image_id"].item())
        layer_idx = outputs.get("attention_layer_index")
        title = f"decoder cross-attention image {image_id}"
        if torch.is_tensor(layer_idx):
            title = (
                f"decoder layer {int(layer_idx.detach().cpu().item())} attention image {image_id}"
            )
        save_attention_overlay(
            arr,
            attention,
            output_dir / f"attention_batch{batch_offset:04d}_image{image_id}.png",
            title=title,
        )
        saved += 1
    return saved
