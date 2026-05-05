"""Augmentation interfaces and builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

Sample = dict[str, Any]
DonorGetter = Any


class Augmenter(Protocol):
    name: str

    def __call__(self, sample: Sample, rng: np.random.Generator) -> Sample: ...


@dataclass
class CopyPasteConfig:
    probability: float = 1.0
    max_paste_objects: int = 3
    target_policy: str = "all"
    class_ids: list[int] = field(default_factory=list)
    rare_quantile: float = 0.25
    min_area: int = 16
    min_bbox_size: int = 2
    paste_scale_jitter: tuple[float, float] = (0.75, 1.25)
    max_placement_attempts: int = 20
    harmonizer_backend: str = "local"
    feather_sigma: float = 2.0
    lbm_steps: int = 4
    lbm_resolution: int = 1024


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def copy_paste_config_from_cfg(cfg: Any) -> CopyPasteConfig:
    jitter = cfg_get(cfg, "paste_scale_jitter", [0.75, 1.25])
    return CopyPasteConfig(
        probability=float(cfg_get(cfg, "probability", 1.0)),
        max_paste_objects=int(cfg_get(cfg, "max_paste_objects", 3)),
        target_policy=str(cfg_get(cfg, "target_policy", "all")),
        class_ids=[int(x) for x in list(cfg_get(cfg, "class_ids", []) or [])],
        rare_quantile=float(cfg_get(cfg, "rare_quantile", 0.25)),
        min_area=int(cfg_get(cfg, "min_area", 16)),
        min_bbox_size=int(cfg_get(cfg, "min_bbox_size", 2)),
        paste_scale_jitter=(float(jitter[0]), float(jitter[1])),
        max_placement_attempts=int(cfg_get(cfg, "max_placement_attempts", 20)),
        harmonizer_backend=str(cfg_get(cfg, "harmonizer_backend", "local")),
        feather_sigma=float(cfg_get(cfg, "feather_sigma", 2.0)),
        lbm_steps=int(cfg_get(cfg, "lbm_steps", 4)),
        lbm_resolution=int(cfg_get(cfg, "lbm_resolution", 1024)),
    )
