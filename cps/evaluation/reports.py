"""Final report orchestration."""

from __future__ import annotations

from typing import Any

from cps.evaluation.comparisons import generate_comparison_report


def generate_report(cfg: Any) -> dict[str, str]:
    return generate_comparison_report(cfg.report.metrics_root, cfg.report.output_dir)
