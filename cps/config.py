"""Hydra config loading helpers used by the Typer CLI."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from cps.paths import CONFIG_DIR

ALIASES = {
    "subset": "subset",
    "augment": "augment",
    "train": "train_config",
    "eval": "eval_config",
    "report": "report",
}


def load_config(config_name: str = "config", overrides: Sequence[str] | None = None):
    """Load a Hydra config from the project-local ``configs`` directory."""

    from hydra import compose, initialize_config_dir

    config_name = ALIASES.get(config_name, config_name)
    overrides = list(overrides or [])
    config_dir = str(Path(CONFIG_DIR).resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name=config_name, overrides=overrides)
