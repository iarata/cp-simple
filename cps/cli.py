"""Typer command line interface for CPS experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from cps.utils.logging import setup_logging

CONTEXT_SETTINGS = {"allow_extra_args": True, "ignore_unknown_options": True}

app = typer.Typer(
    name="cps",
    help="Copy-paste augmentation experiments for COCO instance segmentation.",
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=True,
)


def _load(config_name: str, overrides: list[str]):
    from cps.config import load_config

    return load_config(config_name=config_name, overrides=overrides)


@app.command("make-subsets", context_settings=CONTEXT_SETTINGS)
def make_subsets(
    ctx: typer.Context,
    config_name: Annotated[
        str, typer.Option("--config-name", help="Hydra config name.")
    ] = "subset",
) -> None:
    """Create nested percentage-based COCO subsets and visualizations."""

    setup_logging()
    cfg = _load(config_name, list(ctx.args))
    from cps.data.subsets import build_coco_subsets

    summaries = build_coco_subsets(cfg)
    logger.info("Generated {} subset(s).", len(summaries))


@app.command("make-premade-subsets", context_settings=CONTEXT_SETTINGS)
def make_premade_subsets(
    ctx: typer.Context,
    config_name: Annotated[
        str, typer.Option("--config-name", help="Hydra config name.")
    ] = "subset",
) -> None:
    """Create fixed no-augmentation and copy-paste subset variants for offline training."""

    setup_logging()
    cfg = _load(config_name, list(ctx.args))
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.subset.premade.enabled = True
    from cps.data.subsets import build_coco_subsets

    summaries = build_coco_subsets(cfg)
    logger.info("Generated {} subset(s) with premade variants.", len(summaries))


@app.command("analyze-subsets", context_settings=CONTEXT_SETTINGS)
def analyze_subsets(
    ctx: typer.Context,
    config_name: Annotated[
        str, typer.Option("--config-name", help="Hydra config name.")
    ] = "subset",
) -> None:
    """Recompute subset metadata/plots by rerunning the subset analysis step."""

    make_subsets(ctx, config_name=config_name)


@app.command("preview-augmentations", context_settings=CONTEXT_SETTINGS)
def preview_augmentations(
    ctx: typer.Context,
    config_name: Annotated[
        str, typer.Option("--config-name", help="Hydra config name.")
    ] = "augment",
) -> None:
    """Save original/normal/SimpleCP/PCTNet-style/LBM-style preview grids."""

    setup_logging()
    cfg = _load(config_name, list(ctx.args))
    from cps.augmentations.previews import generate_augmentation_previews

    paths = generate_augmentation_previews(cfg)
    for path in paths:
        logger.info("Wrote preview {}", path)


@app.command("train", context_settings=CONTEXT_SETTINGS)
def train(
    ctx: typer.Context,
    config_name: Annotated[
        str, typer.Option("--config-name", help="Hydra config name.")
    ] = "train_config",
) -> None:
    """Train one instance segmentation experiment."""

    setup_logging()
    cfg = _load(config_name, list(ctx.args))
    from cps.training.train import run_training

    result = run_training(cfg)
    logger.info("Training finished: {}", result["output_dir"])


@app.command("evaluate", context_settings=CONTEXT_SETTINGS)
def evaluate(
    ctx: typer.Context,
    config_name: Annotated[
        str, typer.Option("--config-name", help="Hydra config name.")
    ] = "eval_config",
) -> None:
    """Evaluate one checkpoint and save COCO metrics/visualizations."""

    setup_logging()
    cfg = _load(config_name, list(ctx.args))
    from cps.training.train import run_evaluation

    metrics = run_evaluation(cfg)
    logger.info("Evaluation complete. Available metrics: {}", list(metrics.keys()))


@app.command("report", context_settings=CONTEXT_SETTINGS)
def report(
    ctx: typer.Context,
    config_name: Annotated[
        str, typer.Option("--config-name", help="Hydra config name.")
    ] = "report",
) -> None:
    """Generate final cross-method comparison reports from saved metrics."""

    setup_logging()
    cfg = _load(config_name, list(ctx.args))
    from cps.evaluation.reports import generate_report

    outputs = generate_report(cfg)
    for name, path in outputs.items():
        logger.info("{}: {}", name, path)


@app.command("create-tiny-fixture")
def create_tiny_fixture(
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Where to create the synthetic COCO fixture.")
    ] = Path("data/interim/tiny_coco"),
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 1337,
) -> None:
    """Create a tiny synthetic COCO fixture for local smoke tests."""

    setup_logging()
    from cps.data.fixture import create_tiny_coco_fixture

    paths = create_tiny_coco_fixture(output_dir, seed=seed)
    for key, value in paths.items():
        logger.info("{}: {}", key, value)


if __name__ == "__main__":
    app()
