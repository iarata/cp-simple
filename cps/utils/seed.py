"""Deterministic seeding utilities."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedState:
    seed: int
    deterministic: bool = True


def seed_everything(seed: int, deterministic: bool = True) -> SeedState:
    """Seed Python, NumPy, and PyTorch if it is available.

    Determinism is best effort: some CUDA/MPS kernels remain nondeterministic or
    slower when deterministic algorithms are requested.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except Exception:
        # Torch is an optional import for non-training commands.
        pass
    return SeedState(seed=seed, deterministic=deterministic)


def rng_from_seed(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)
