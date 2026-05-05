"""Shortcut-learning diagnostics for copy-paste artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from cps.augmentations.masks import mask_boundary


@dataclass(frozen=True)
class BoundaryAttentionReport:
    boundary_attention_fraction: float
    foreground_attention_fraction: float
    boundary_to_foreground_ratio: float
    note: str


def analyze_boundary_attention(
    attention_map: np.ndarray, pasted_mask: np.ndarray
) -> BoundaryAttentionReport:
    att = np.asarray(attention_map, dtype=np.float32)
    mask = np.asarray(pasted_mask, dtype=bool)
    if att.shape != mask.shape:
        from PIL import Image

        att_img = Image.fromarray(np.uint8(255 * _normalize(att)))
        att = np.asarray(
            att_img.resize((mask.shape[1], mask.shape[0]), Image.BILINEAR), dtype=np.float32
        )
        att = _normalize(att)
    boundary = mask_boundary(mask, width=3)
    total = float(att.sum()) + 1e-6
    boundary_fraction = float(att[boundary].sum() / total) if boundary.any() else 0.0
    fg_fraction = float(att[mask].sum() / total) if mask.any() else 0.0
    ratio = boundary_fraction / max(fg_fraction, 1e-6)
    if ratio > 0.5:
        note = "attention concentrates strongly near pasted-object boundaries"
    elif boundary_fraction > 0.15:
        note = "attention has a visible boundary component"
    else:
        note = "no obvious boundary shortcut signal in this attention map"
    return BoundaryAttentionReport(boundary_fraction, fg_fraction, ratio, note)


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - float(arr.min())
    if float(arr.max()) > 0:
        arr = arr / float(arr.max())
    return arr


def report_to_dict(report: BoundaryAttentionReport) -> dict[str, float | str]:
    return asdict(report)
