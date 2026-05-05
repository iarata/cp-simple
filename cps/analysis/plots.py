"""Plotting helpers for final experiment reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from cps.paths import project_path


def save_metric_barplot(
    df: pd.DataFrame, x: str, y: str, output_path: str | Path, title: str
) -> None:
    output_path = project_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(6, len(df) * 0.7), 5))
    plt.bar(df[x].astype(str), df[y])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_lineplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    output_path: str | Path,
    title: str,
) -> None:
    output_path = project_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for group_name, group_df in df.groupby(group):
        group_df = group_df.sort_values(x)
        plt.plot(group_df[x], group_df[y], marker="o", label=str(group_name))
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def summarize_shortcut_notes(reports: list[dict[str, Any]]) -> dict[str, Any]:
    notes = [str(report.get("note", "")) for report in reports]
    return {"num_reports": len(reports), "notes": notes[:20]}
