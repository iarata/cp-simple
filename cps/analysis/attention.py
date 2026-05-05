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
        height, width = [int(v) for v in target["orig_size"].detach().cpu().tolist()]
        arr = (image[:, :height, :width].detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(
            0, 255
        )
        arr = arr.astype(np.uint8)
        image_id = int(target["image_id"].item())
        save_attention_overlay(
            arr,
            attention,
            output_dir / f"attention_batch{batch_offset:04d}_image{image_id}.png",
            title=f"decoder cross-attention image {image_id}",
        )
        saved += 1
    return saved
