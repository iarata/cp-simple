"""Mask, bbox, and copy-paste geometry utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage


def bbox_xywh_to_xyxy(bbox: list[float] | tuple[float, float, float, float]) -> list[float]:
    x, y, w, h = [float(v) for v in bbox]
    return [x, y, x + w, y + h]


def bbox_xyxy_to_xywh(bbox: list[float] | tuple[float, float, float, float]) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]


def mask_to_bbox_xyxy(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(np.asarray(mask).astype(bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def mask_area(mask: np.ndarray) -> float:
    return float(np.asarray(mask, dtype=bool).sum())


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize mask to ``(width, height)`` using nearest-neighbor sampling."""

    img = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    return np.asarray(img.resize(size, Image.NEAREST)) > 127


def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    return np.asarray(img.resize(size, Image.BILINEAR), dtype=np.uint8)


def crop_instance(
    image: np.ndarray, mask: np.ndarray, bbox: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = [round(v) for v in bbox]
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return image[:1, :1].copy(), np.zeros((1, 1), dtype=bool)
    return image[y0:y1, x0:x1].copy(), np.asarray(mask[y0:y1, x0:x1], dtype=bool).copy()


def place_crop_on_canvas(
    crop_image: np.ndarray,
    crop_mask: np.ndarray,
    canvas_shape: tuple[int, int],
    xy: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = canvas_shape
    canvas_img = np.zeros((height, width, 3), dtype=np.uint8)
    canvas_mask = np.zeros((height, width), dtype=bool)
    x, y = xy
    ch, cw = crop_mask.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + cw)
    y1 = min(height, y + ch)
    if x1 <= x0 or y1 <= y0:
        return canvas_img, canvas_mask
    src_x0 = x0 - x
    src_y0 = y0 - y
    src_x1 = src_x0 + (x1 - x0)
    src_y1 = src_y0 + (y1 - y0)
    canvas_img[y0:y1, x0:x1] = crop_image[src_y0:src_y1, src_x0:src_x1]
    canvas_mask[y0:y1, x0:x1] = crop_mask[src_y0:src_y1, src_x0:src_x1]
    return canvas_img, canvas_mask


def remove_tiny_instances(
    instances: list[dict[str, Any]],
    min_area: int = 16,
    min_bbox_size: int = 2,
) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for inst in instances:
        mask = np.asarray(inst.get("mask"), dtype=bool)
        if int(mask.sum()) < min_area:
            continue
        bbox = mask_to_bbox_xyxy(mask)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        if (x1 - x0) < min_bbox_size or (y1 - y0) < min_bbox_size:
            continue
        updated = dict(inst)
        updated["mask"] = mask
        updated["bbox_xyxy"] = [float(x0), float(y0), float(x1), float(y1)]
        updated["bbox"] = updated["bbox_xyxy"]
        updated["bbox_mode"] = "xyxy"
        updated["area"] = float(mask.sum())
        valid.append(updated)
    return valid


def subtract_occlusion(
    instances: list[dict[str, Any]], occluder: np.ndarray
) -> list[dict[str, Any]]:
    occluder = np.asarray(occluder, dtype=bool)
    updated = []
    for inst in instances:
        new_inst = dict(inst)
        new_inst["mask"] = np.asarray(inst["mask"], dtype=bool) & ~occluder
        updated.append(new_inst)
    return updated


def paste_foreground(
    background: np.ndarray,
    foreground: np.ndarray,
    mask: np.ndarray,
    alpha: np.ndarray | None = None,
) -> np.ndarray:
    bg = np.asarray(background, dtype=np.float32).copy()
    fg = np.asarray(foreground, dtype=np.float32)
    mask_bool = np.asarray(mask, dtype=bool)
    if alpha is None:
        bg[mask_bool] = fg[mask_bool]
    else:
        a = np.asarray(alpha, dtype=np.float32)
        if a.ndim == 2:
            a = a[..., None]
        bg = bg * (1.0 - a) + fg * a
    return np.clip(bg, 0, 255).astype(np.uint8)


def feather_alpha(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    mask_f = np.asarray(mask, dtype=np.float32)
    blurred = ndimage.gaussian_filter(mask_f, sigma=max(float(sigma), 0.01))
    if blurred.max() > 0:
        blurred = blurred / blurred.max()
    return np.clip(blurred, 0.0, 1.0)


def mask_boundary(mask: np.ndarray, width: int = 3) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if not mask_bool.any():
        return mask_bool
    dilated = ndimage.binary_dilation(mask_bool, iterations=max(1, width))
    eroded = ndimage.binary_erosion(mask_bool, iterations=max(1, width))
    return dilated ^ eroded
