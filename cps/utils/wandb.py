"""Optional Weights & Biases integration."""

from __future__ import annotations

import math
from collections.abc import Iterable
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


def wandb_image(path: str | Path, caption: str | None = None) -> Any | None:
    try:
        import wandb

        return wandb.Image(str(path), caption=caption)
    except Exception:
        return None


def log_validation_outputs(
    run: Any,
    metrics: dict[str, Any],
    output_dir: str | Path,
    *,
    epoch: int | None = None,
    max_visualizations: int = 16,
    log_plots: bool = True,
    log_per_class_ap: bool = True,
) -> None:
    if not run:
        return

    payload = validation_scalar_logs(metrics)
    if epoch is not None:
        payload["epoch"] = epoch
    if log_plots:
        payload.update(validation_plot_logs(metrics, log_per_class_ap=log_per_class_ap))

    payload.update(validation_image_logs(output_dir, max_images=max_visualizations))

    if payload:
        run.log(payload)


def validation_scalar_logs(metrics: dict[str, Any], prefix: str = "val") -> dict[str, float]:
    logs: dict[str, float] = {}
    num_predictions = _finite_float(metrics.get("num_predictions"))
    if num_predictions is not None:
        logs[f"{prefix}/num_predictions"] = num_predictions

    losses = metrics.get("losses", {})
    if isinstance(losses, dict):
        for key, value in losses.items():
            scalar = _finite_float(value)
            if scalar is not None:
                logs[f"{prefix}/{key}"] = scalar

    for iou_type in ("segm", "bbox"):
        iou_metrics = metrics.get(iou_type, {})
        if not isinstance(iou_metrics, dict):
            continue
        for key, value in iou_metrics.items():
            if key == "per_class_AP":
                continue
            scalar = _finite_float(value)
            if scalar is not None:
                logs[f"{prefix}/{iou_type}_{key}"] = scalar
    return logs


def validation_plot_logs(
    metrics: dict[str, Any],
    prefix: str = "val",
    *,
    log_per_class_ap: bool = True,
) -> dict[str, Any]:
    try:
        import wandb
    except Exception:
        return {}

    logs: dict[str, Any] = {}
    summary_rows: list[list[str | float]] = []
    chart_rows: list[list[str | float]] = []
    chart_metrics = {"mAP", "AP50", "AP75", "AR100"}
    for iou_type in ("segm", "bbox"):
        iou_metrics = metrics.get(iou_type, {})
        if not isinstance(iou_metrics, dict):
            continue
        for key, value in iou_metrics.items():
            if key == "per_class_AP":
                continue
            scalar = _finite_float(value)
            if scalar is None:
                continue
            summary_rows.append([iou_type, key, scalar])
            if key in chart_metrics:
                chart_rows.append([f"{iou_type}/{key}", scalar])
    if summary_rows:
        logs[f"{prefix}/tables/coco_summary"] = wandb.Table(
            data=summary_rows, columns=["iou_type", "metric", "value"]
        )
    if chart_rows:
        chart_table = wandb.Table(data=chart_rows, columns=["metric", "value"])
        logs[f"{prefix}/plots/coco_summary"] = wandb.plot.bar(
            chart_table, "metric", "value", title="COCO validation metrics"
        )

    if log_per_class_ap:
        for iou_type in ("segm", "bbox"):
            per_class = metrics.get(iou_type, {}).get("per_class_AP", {})
            if not isinstance(per_class, dict):
                continue
            rows = [
                [str(cat_id), scalar]
                for cat_id, value in sorted(per_class.items(), key=lambda item: int(item[0]))
                if (scalar := _finite_float(value)) is not None
            ]
            if not rows:
                continue
            table = wandb.Table(data=rows, columns=["category_id", "AP"])
            logs[f"{prefix}/tables/{iou_type}_per_class_AP"] = table
            logs[f"{prefix}/plots/{iou_type}_per_class_AP"] = wandb.plot.bar(
                table, "category_id", "AP", title=f"{iou_type.upper()} per-class AP"
            )
    return logs


def validation_image_logs(output_dir: str | Path, max_images: int = 16) -> dict[str, Any]:
    if max_images <= 0:
        return {}
    output_dir = Path(output_dir)
    image_paths = list(_iter_image_paths(output_dir / "visualizations"))
    logs = {}
    for path in image_paths[:max_images]:
        rel_path = path.relative_to(output_dir)
        caption = str(rel_path)
        key = f"val/{_wandb_path_key(rel_path.with_suffix(''))}"
        image = wandb_image(path, caption=caption)
        if image is not None:
            logs[key] = image
    return logs


def _iter_image_paths(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    suffixes = {".png", ".jpg", ".jpeg"}
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in suffixes)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    return None


def _wandb_path_key(path: Path) -> str:
    return "/".join(part.replace(" ", "_") for part in path.parts)
