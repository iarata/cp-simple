"""Project logging setup."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True, backtrace=False, diagnose=False)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            path, level=level, rotation="20 MB", retention=5, backtrace=False, diagnose=False
        )
