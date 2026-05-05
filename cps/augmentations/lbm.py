"""LBM-style copy-paste harmonization.

The local default is a boundary-aware, multi-scale blend with context color
matching. It is intentionally cheap and deterministic. Set
``harmonizer_backend=libcom`` to route through libcom's diffusion LBM backend
when that optional dependency and checkpoints are available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from cps.augmentations.base import CopyPasteConfig
from cps.augmentations.masks import feather_alpha, mask_boundary
from cps.augmentations.simple_copy_paste import DonorGetter, SimpleCopyPasteAugmentation


@dataclass
class LBMStyleHarmonizer:
    backend: str = "local"
    device: str | int = "cpu"
    steps: int = 4
    resolution: int = 1024
    feather_sigma: float = 2.0
    _legacy_model: Any | None = field(default=None, init=False, repr=False)

    def harmonize(
        self,
        composite: np.ndarray,
        background: np.ndarray,
        pasted_mask: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if self.backend == "libcom":
            result = self._try_libcom(composite, pasted_mask)
            if result is not None:
                return result
        del rng
        return self._local_boundary_blend(composite, background, pasted_mask)

    def _try_libcom(self, composite: np.ndarray, pasted_mask: np.ndarray) -> np.ndarray | None:
        try:
            if self._legacy_model is None:
                from libcom.image_harmonization import ImageHarmonizationModel

                self._legacy_model = ImageHarmonizationModel(device=self.device, model_type="LBM")
            bgr = np.asarray(composite, dtype=np.uint8)[..., ::-1]
            mask = np.asarray(pasted_mask, dtype=np.uint8) * 255
            result_bgr = self._legacy_model(
                bgr,
                mask,
                steps=int(self.steps),
                resolution=int(self.resolution),
            )
            return np.asarray(result_bgr, dtype=np.uint8)[..., ::-1]
        except Exception:
            return None

    def _local_boundary_blend(
        self, composite: np.ndarray, background: np.ndarray, pasted_mask: np.ndarray
    ) -> np.ndarray:
        mask = np.asarray(pasted_mask, dtype=bool)
        if not mask.any():
            return composite
        comp = np.asarray(composite, dtype=np.float32).copy()
        bg = np.asarray(background, dtype=np.float32)
        ring = mask_boundary(mask, width=10) & ~mask
        if int(ring.sum()) >= 8:
            fg = comp[mask]
            ctx = bg[ring]
            fg_mean = fg.mean(axis=0)
            ctx_mean = ctx.mean(axis=0)
            # Conservative color shift: enough to reduce obvious paste artifacts
            # without inventing a heavy diffusion pipeline.
            comp[mask] = fg + 0.55 * (ctx_mean - fg_mean)
        alpha = feather_alpha(mask, sigma=self.feather_sigma)
        core = ndimage.binary_erosion(mask, iterations=2)
        alpha[core] = 1.0
        alpha[~mask & (alpha < 0.02)] = 0.0
        blended = bg * (1.0 - alpha[..., None]) + comp * alpha[..., None]
        return np.clip(blended, 0, 255).astype(np.uint8)


@dataclass
class LBMCopyPasteAugmentation(SimpleCopyPasteAugmentation):
    harmonizer: LBMStyleHarmonizer | None = None
    name: str = "lbm_copy_paste"

    def __init__(
        self,
        donor_getter: DonorGetter,
        donor_count: int,
        category_to_indices: dict[int, list[int]],
        allowed_category_ids: set[int] | None = None,
        config: CopyPasteConfig | None = None,
        device: str | int = "cpu",
    ) -> None:
        cfg = config or CopyPasteConfig()
        super().__init__(
            donor_getter=donor_getter,
            donor_count=donor_count,
            category_to_indices=category_to_indices,
            allowed_category_ids=allowed_category_ids,
            config=cfg,
            name="lbm_copy_paste",
        )
        self.harmonizer = LBMStyleHarmonizer(
            backend=cfg.harmonizer_backend,
            device=device,
            steps=cfg.lbm_steps,
            resolution=cfg.lbm_resolution,
            feather_sigma=cfg.feather_sigma,
        )

    def harmonize(
        self,
        composite: np.ndarray,
        background: np.ndarray,
        pasted_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        assert self.harmonizer is not None
        return self.harmonizer.harmonize(composite, background, pasted_mask, rng)
