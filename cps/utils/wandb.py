"""Optional Weights & Biases integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import OmegaConf


class NullRun:
    """Drop-in no-op object for disabled or unavailable W&B."""

    def log(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def finish(self) -> None:
        return None

    def __bool__(self) -> bool:
        return False


def init_wandb(cfg: Any, job_type: str) -> Any:
    wandb_cfg = getattr(cfg, "wandb", {})
    if not bool(getattr(wandb_cfg, "enabled", False)):
        return NullRun()
    try:
        import wandb
    except ImportError:
        logger.warning(
            "W&B is enabled in config but wandb is not installed; continuing without it."
        )
        return NullRun()

    mode = getattr(wandb_cfg, "mode", "online")
    run = wandb.init(
        project=getattr(wandb_cfg, "project", "cps-copy-paste"),
        entity=getattr(wandb_cfg, "entity", None) or None,
        group=getattr(wandb_cfg, "group", None) or None,
        tags=list(getattr(wandb_cfg, "tags", [])),
        job_type=job_type,
        mode=mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return run


def log_artifact(run: Any, path: str | Path, artifact_type: str, name: str | None = None) -> None:
    if not run:
        return
    try:
        import wandb

        path = Path(path)
        artifact = wandb.Artifact(name or path.stem, type=artifact_type)
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
        run.log_artifact(artifact)
    except Exception as exc:  # pragma: no cover - defensive around external service
        logger.warning("Failed to log W&B artifact {}: {}", path, exc)


def wandb_image(path: str | Path) -> Any | None:
    try:
        import wandb

        return wandb.Image(str(path))
    except Exception:
        return None
