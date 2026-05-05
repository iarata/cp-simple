"""Cross-experiment comparison plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cps.analysis.plots import save_lineplot, save_metric_barplot
from cps.paths import project_path


def collect_metric_reports(root: str | Path) -> pd.DataFrame:
    root = project_path(root)
    rows: list[dict[str, Any]] = []
    for path in root.rglob("metrics.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                report = json.load(f)
        except json.JSONDecodeError:
            continue
        parts = path.parts
        method = _infer_part(parts, "augmentation", default="unknown")
        subset = _infer_subset(parts)
        config_path = path.parent / "config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            method = cfg.get("augmentation", {}).get("name", method)
            subset = cfg.get("subset", {}).get("percent", subset)
        for iou_type in ("segm", "bbox"):
            metrics = report.get(iou_type, {})
            if not isinstance(metrics, dict):
                continue
            rows.append(
                {
                    "path": str(path),
                    "method": method,
                    "subset_percent": float(subset) if subset is not None else float("nan"),
                    "iou_type": iou_type,
                    "mAP": metrics.get("mAP", float("nan")),
                    "AP50": metrics.get("AP50", float("nan")),
                    "AP75": metrics.get("AP75", float("nan")),
                    "AR100": metrics.get("AR100", float("nan")),
                    "per_class_AP": metrics.get("per_class_AP", {}),
                }
            )
    return pd.DataFrame(rows)


def generate_comparison_report(metrics_root: str | Path, output_dir: str | Path) -> dict[str, str]:
    output_dir = project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = collect_metric_reports(metrics_root)
    outputs: dict[str, str] = {}
    csv_path = output_dir / "comparison_metrics.csv"
    if df.empty:
        df.to_csv(csv_path, index=False)
        outputs["csv"] = str(csv_path)
        return outputs
    df.to_csv(csv_path, index=False)
    outputs["csv"] = str(csv_path)
    segm = df[df["iou_type"] == "segm"].copy()
    if not segm.empty:
        latest = segm.sort_values(["subset_percent", "method"])
        bar_path = output_dir / "map_by_method.png"
        save_metric_barplot(
            latest, x="method", y="mAP", output_path=bar_path, title="Segm mAP by method"
        )
        outputs["map_by_method"] = str(bar_path)
        line_path = output_dir / "map_vs_subset.png"
        save_lineplot(
            latest,
            x="subset_percent",
            y="mAP",
            group="method",
            output_path=line_path,
            title="Segm mAP vs subset percentage",
        )
        outputs["map_vs_subset"] = str(line_path)
        delta_csv = output_dir / "per_class_ap_delta_vs_normal.csv"
        _write_per_class_deltas(segm, delta_csv)
        outputs["per_class_ap_delta_vs_normal"] = str(delta_csv)
    return outputs


def _write_per_class_deltas(df: pd.DataFrame, output_path: Path) -> None:
    rows = []
    for subset_percent, group in df.groupby("subset_percent"):
        normal_rows = group[group["method"] == "normal"]
        if normal_rows.empty:
            continue
        baseline = normal_rows.iloc[0]["per_class_AP"] or {}
        for _, row in group.iterrows():
            per_class = row["per_class_AP"] or {}
            for cat_id, ap in per_class.items():
                base_ap = baseline.get(cat_id)
                if base_ap is None:
                    continue
                rows.append(
                    {
                        "subset_percent": subset_percent,
                        "method": row["method"],
                        "category_id": cat_id,
                        "ap": ap,
                        "baseline_normal_ap": base_ap,
                        "delta_vs_normal": ap - base_ap,
                    }
                )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _infer_part(parts: tuple[str, ...], marker: str, default: str) -> str:
    for part in reversed(parts):
        if marker in part:
            return part
    return default


def _infer_subset(parts: tuple[str, ...]) -> float | None:
    for part in reversed(parts):
        if part.startswith("pct_"):
            try:
                return float(part.split("_")[1].replace("p", "."))
            except Exception:
                continue
    return None
