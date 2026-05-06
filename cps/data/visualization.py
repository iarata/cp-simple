"""Visualization helpers for COCO samples, subsets, predictions, and reports."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

Color = tuple[int, int, int]


def color_for_category(category_id: int) -> Color:
    rng = np.random.default_rng(int(category_id) * 9973 + 17)
    return tuple(int(x) for x in rng.integers(32, 230, size=3))  # type: ignore[return-value]


def _as_pil_rgb(image: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")


def overlay_instances(
    image: np.ndarray | Image.Image,
    instances: Iterable[dict[str, Any]],
    categories: dict[int, str] | None = None,
    draw_masks: bool = True,
    draw_boxes: bool = True,
    alpha: float = 0.45,
) -> Image.Image:
    base = _as_pil_rgb(image)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    categories = categories or {}
    for inst in instances:
        cat_id = int(inst.get("category_id", inst.get("label", 0)))
        color = color_for_category(cat_id)
        mask = inst.get("mask")
        if draw_masks and mask is not None:
            mask_arr = np.asarray(mask).astype(bool)
            if mask_arr.shape[:2] == (base.height, base.width):
                rgba = np.zeros((base.height, base.width, 4), dtype=np.uint8)
                rgba[mask_arr, :3] = color
                rgba[mask_arr, 3] = int(255 * alpha)
                overlay = Image.alpha_composite(overlay, Image.fromarray(rgba, mode="RGBA"))
                draw = ImageDraw.Draw(overlay)
        bbox = inst.get("bbox_xyxy") or inst.get("box") or inst.get("bbox")
        if draw_boxes and bbox is not None and len(bbox) == 4:
            if inst.get("bbox_mode") == "xywh":
                x, y, w, h = [float(v) for v in bbox]
                box = [x, y, x + w, y + h]
            else:
                box = [float(v) for v in bbox]
            draw.rectangle(box, outline=(*color, 255), width=2)
            label = categories.get(cat_id, str(cat_id))
            score = inst.get("score")
            if score is not None:
                label = f"{label} {float(score):.2f}"
            text_xy = (box[0] + 2, max(0, box[1] - 12))
            draw.rectangle(
                [text_xy[0] - 1, text_xy[1], text_xy[0] + 7 * len(label), text_xy[1] + 12],
                fill=(*color, 220),
            )
            draw.text(text_xy, label, fill=(255, 255, 255, 255))
    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


def save_class_frequency_plot(
    rows: list[dict[str, Any]], output_path: str | Path, title: str
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = [row["name"] for row in rows]
    counts = [row["instances"] for row in rows]
    fig_width = max(10, min(28, len(names) * 0.25))
    plt.figure(figsize=(fig_width, 5))
    plt.bar(range(len(names)), counts)
    plt.xticks(range(len(names)), names, rotation=90, fontsize=6)
    plt.ylabel("Instances")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_long_tail_plot(rows: list[dict[str, Any]], output_path: str | Path, title: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = sorted([row["instances"] for row in rows if row["instances"] > 0], reverse=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(counts) + 1), counts, marker="o", linewidth=1)
    plt.yscale("log")
    plt.xlabel("Class rank")
    plt.ylabel("Instances, log scale")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_long_tail_comparison_plot(
    before_rows: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
    output_path: str | Path,
    title: str,
    before_label: str = "before",
    after_label: str = "after",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    before_counts = sorted(
        [int(row["instances"]) for row in before_rows if int(row["instances"]) > 0],
        reverse=True,
    )
    after_counts = sorted(
        [int(row["instances"]) for row in after_rows if int(row["instances"]) > 0],
        reverse=True,
    )
    plt.figure(figsize=(8, 5))
    if before_counts:
        plt.plot(
            range(1, len(before_counts) + 1),
            before_counts,
            marker="o",
            linewidth=1,
            label=before_label,
        )
    if after_counts:
        plt.plot(
            range(1, len(after_counts) + 1),
            after_counts,
            marker="o",
            linewidth=1,
            label=after_label,
        )
    plt.yscale("log")
    plt.xlabel("Class rank")
    plt.ylabel("Instances, log scale")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_distribution_comparison_plot(
    full_rows: list[dict[str, Any]],
    subset_rows: list[dict[str, Any]],
    output_path: str | Path,
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full = {int(row["category_id"]): row for row in full_rows}
    subset = {int(row["category_id"]): row for row in subset_rows}
    cat_ids = sorted(full)
    names = [full[cid]["name"] for cid in cat_ids]
    full_total = max(sum(full[cid]["instances"] for cid in cat_ids), 1)
    subset_total = max(sum(subset.get(cid, {}).get("instances", 0) for cid in cat_ids), 1)
    full_freq = [full[cid]["instances"] / full_total for cid in cat_ids]
    subset_freq = [subset.get(cid, {}).get("instances", 0) / subset_total for cid in cat_ids]
    plt.figure(figsize=(max(10, min(28, len(cat_ids) * 0.25)), 5))
    plt.plot(range(len(cat_ids)), full_freq, label="full")
    plt.plot(range(len(cat_ids)), subset_freq, label="subset")
    plt.xticks(range(len(cat_ids)), names, rotation=90, fontsize=6)
    plt.ylabel("Relative instance frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_instance_count_comparison_plot(
    before_rows: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
    output_path: str | Path,
    title: str,
    before_label: str = "before",
    after_label: str = "after",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    before = {int(row["category_id"]): row for row in before_rows}
    after = {int(row["category_id"]): row for row in after_rows}
    cat_ids = sorted(set(before) | set(after))
    names = [str((before.get(cid) or after.get(cid) or {}).get("name", cid)) for cid in cat_ids]
    before_counts = [before.get(cid, {}).get("instances", 0) for cid in cat_ids]
    after_counts = [after.get(cid, {}).get("instances", 0) for cid in cat_ids]
    x = np.arange(len(cat_ids), dtype=np.float32)
    width = 0.42
    plt.figure(figsize=(max(10, min(30, len(cat_ids) * 0.3)), 5.5))
    plt.bar(x - width / 2, before_counts, width=width, label=before_label)
    plt.bar(x + width / 2, after_counts, width=width, label=after_label)
    plt.xticks(x, names, rotation=90, fontsize=6)
    plt.ylabel("Instances")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_instance_delta_plot(
    before_rows: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
    output_path: str | Path,
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    before = {int(row["category_id"]): row for row in before_rows}
    after = {int(row["category_id"]): row for row in after_rows}
    cat_ids = sorted(set(before) | set(after))
    deltas = [
        (
            cid,
            str((before.get(cid) or after.get(cid) or {}).get("name", cid)),
            int(after.get(cid, {}).get("instances", 0))
            - int(before.get(cid, {}).get("instances", 0)),
        )
        for cid in cat_ids
    ]
    nonzero = [item for item in deltas if item[2] != 0]
    rows = nonzero or deltas
    rows = sorted(rows, key=lambda item: (item[2], item[1]), reverse=True)
    names = [row[1] for row in rows]
    values = [row[2] for row in rows]
    colors = ["#2f9e44" if value >= 0 else "#c92a2a" for value in values]
    plt.figure(figsize=(max(10, min(30, len(rows) * 0.35)), 5.5))
    plt.bar(range(len(rows)), values, color=colors)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(range(len(rows)), names, rotation=90, fontsize=6)
    plt.ylabel("Instance count delta")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_image_grid(
    images: list[Image.Image], labels: list[str], output_path: str | Path, columns: int = 4
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not images:
        Image.new("RGB", (512, 128), "white").save(output_path)
        return
    thumb_w = 320
    thumb_h = 240
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + 24)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, image in enumerate(images):
        row, col = divmod(idx, columns)
        thumb = image.copy()
        thumb.thumbnail((thumb_w, thumb_h))
        x = col * thumb_w + (thumb_w - thumb.width) // 2
        y = row * (thumb_h + 24) + 20
        draw.text((col * thumb_w + 4, row * (thumb_h + 24) + 4), labels[idx], fill=(0, 0, 0))
        canvas.paste(thumb, (x, y))
    canvas.save(output_path)


def save_attention_overlay(
    image: np.ndarray | Image.Image,
    attention: np.ndarray,
    output_path: str | Path,
    title: str = "attention",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = _as_pil_rgb(image)
    att = np.asarray(attention, dtype=np.float32)
    if att.ndim == 3:
        att = att.mean(axis=0)
    att = att - float(att.min())
    if float(att.max()) > 0:
        att = att / float(att.max())
    att_img = Image.fromarray(np.uint8(att * 255)).resize(img.size, Image.BILINEAR)
    plt.figure(figsize=(6, 5))
    plt.imshow(img)
    plt.imshow(att_img, alpha=0.45)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
