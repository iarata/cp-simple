"""PCTNet-style copy-paste harmonization.

The project-local default is a deterministic, lightweight color-transform
harmonizer inspired by PCTNet's foreground-only pixel-wise color transforms. A
legacy libcom backend can be enabled explicitly, but the training pipeline does
not depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger
from scipy import ndimage

from cps.augmentations.base import CopyPasteConfig
from cps.augmentations.masks import mask_boundary
from cps.augmentations.simple_copy_paste import DonorGetter, SimpleCopyPasteAugmentation


@dataclass
class PCTNetStyleHarmonizer:
    backend: str = "local"
    device: str | int = "cpu"
    _legacy_model: Any | None = field(default=None, init=False, repr=False)

    def harmonize(
        self,
        composite: np.ndarray,
        background: np.ndarray,
        pasted_mask: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if self.backend == "libcom":
            return self._run_libcom(composite, pasted_mask)
        if self.backend != "local":
            raise ValueError("PCTNet harmonizer_backend must be 'local' or 'libcom'.")
        del rng
        return self._local_color_transform(composite, background, pasted_mask)

    def _run_libcom(self, composite: np.ndarray, pasted_mask: np.ndarray) -> np.ndarray:
        try:
            if self._legacy_model is None:
                from libcom.image_harmonization import ImageHarmonizationModel

                logger.info(
                    "Initializing libcom ImageHarmonizationModel(model_type='PCTNet', device={!r})",
                    self.device,
                )
                self._legacy_model = ImageHarmonizationModel(
                    device=self.device, model_type="PCTNet"
                )
                logger.info("Initialized libcom PCTNet harmonizer.")
            bgr = np.asarray(composite, dtype=np.uint8)[..., ::-1]
            mask = np.asarray(pasted_mask, dtype=np.uint8) * 255
            result_bgr = self._legacy_model(bgr, mask)
            return np.asarray(result_bgr, dtype=np.uint8)[..., ::-1]
        except ImportError as exc:
            raise RuntimeError(
                "PCTNet harmonizer_backend=libcom was requested, but libcom or one of its "
                "optional dependencies is not installed. "
                "Install the optional dependency with `uv sync --extra legacy-libcom` or run "
                "commands with `uv run --extra legacy-libcom ...`."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "PCTNet harmonizer_backend=libcom failed during initialization or inference. "
                "The local fallback was not used because libcom was explicitly requested."
            ) from exc

    def _local_color_transform(
        self, composite: np.ndarray, background: np.ndarray, pasted_mask: np.ndarray
    ) -> np.ndarray:
        mask = np.asarray(pasted_mask, dtype=bool)
        if not mask.any():
            return composite
        comp = np.asarray(composite, dtype=np.float32).copy()
        bg = np.asarray(background, dtype=np.float32)
        # Context ring estimates the surrounding image statistics while avoiding
        # the pasted foreground itself.
        ring = mask_boundary(mask, width=8) & ~mask
        if int(ring.sum()) < 8:
            ring = ~mask
        if int(ring.sum()) < 8:
            return composite
        fg_pixels = comp[mask]
        bg_pixels = bg[ring]
        fg_mean = fg_pixels.mean(axis=0)
        fg_std = fg_pixels.std(axis=0) + 1e-6
        bg_mean = bg_pixels.mean(axis=0)
        bg_std = bg_pixels.std(axis=0) + 1e-6
        adjusted = (fg_pixels - fg_mean) * (bg_std / fg_std) + bg_mean

        # Add a low-frequency spatial correction to emulate spatially varying
        # foreground color transforms rather than a single global color transfer.
        residual = np.zeros_like(comp)
        residual[mask] = adjusted - fg_pixels
        for channel in range(3):
            residual[..., channel] = ndimage.gaussian_filter(residual[..., channel], sigma=9.0)
        comp[mask] = fg_pixels + residual[mask]
        return np.clip(comp, 0, 255).astype(np.uint8)


@dataclass
class PCTNetCopyPasteAugmentation(SimpleCopyPasteAugmentation):
    harmonizer: PCTNetStyleHarmonizer | None = None
    name: str = "pctnet_copy_paste"

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
            name="pctnet_copy_paste",
        )
        self.harmonizer = PCTNetStyleHarmonizer(backend=cfg.harmonizer_backend, device=device)

    def harmonize(
        self,
        composite: np.ndarray,
        background: np.ndarray,
        pasted_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        assert self.harmonizer is not None
        return self.harmonizer.harmonize(composite, background, pasted_mask, rng)
