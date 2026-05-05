"""COCO metric reporting with graceful pycocotools fallback."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from cps.augmentations.masks import bbox_xyxy_to_xywh
from cps.paths import project_path


def encode_binary_mask(mask: np.ndarray) -> dict[str, Any]:
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(v) for v in rle["size"]], "counts": counts}


def predictions_to_coco_results(
    predictions: list[dict[str, Any]], iou_type: str
) -> list[dict[str, Any]]:
    results = []
    for pred in predictions:
        bbox_xyxy = pred.get("bbox_xyxy") or pred.get("box")
        if bbox_xyxy is None:
            continue
        item = {
            "image_id": int(pred["image_id"]),
            "category_id": int(pred["category_id"]),
            "score": float(pred.get("score", 1.0)),
            "bbox": [float(v) for v in bbox_xyxy_to_xywh(bbox_xyxy)],
        }
        if iou_type == "segm":
            if "mask" not in pred:
                continue
            item["segmentation"] = encode_binary_mask(np.asarray(pred["mask"], dtype=bool))
        results.append(item)
    return results


def summarize_coco_eval(coco_eval: Any, categories: list[dict[str, Any]]) -> dict[str, Any]:
    stats = (
        coco_eval.stats.tolist() if getattr(coco_eval, "stats", None) is not None else [0.0] * 12
    )
    metrics = {
        "mAP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR1": float(stats[6]),
        "AR10": float(stats[7]),
        "AR100": float(stats[8]),
        "AR_small": float(stats[9]),
        "AR_medium": float(stats[10]),
        "AR_large": float(stats[11]),
    }
    precision = coco_eval.eval.get("precision") if getattr(coco_eval, "eval", None) else None
    per_class_ap: dict[str, float] = {}
    if precision is not None:
        # precision: [TxRxKxAxM]. Average over IoU thresholds, recall, all-area, maxDets last.
        for class_idx, cat in enumerate(categories):
            values = precision[:, :, class_idx, 0, -1]
            values = values[values > -1]
            per_class_ap[str(int(cat["id"]))] = (
                float(np.mean(values)) if values.size else float("nan")
            )
    metrics["per_class_AP"] = per_class_ap
    return metrics


def evaluate_coco_predictions(
    ground_truth_json: str | Path,
    predictions: list[dict[str, Any]],
    output_dir: str | Path,
    iou_types: tuple[str, ...] = ("segm", "bbox"),
) -> dict[str, Any]:
    output_dir = project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_json = project_path(ground_truth_json)
    report: dict[str, Any] = {
        "ground_truth_json": str(ground_truth_json),
        "num_predictions": len(predictions),
    }
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as exc:
        logger.warning(
            "pycocotools unavailable; writing predictions but skipping COCOeval: {}", exc
        )
        report["available"] = False
        report["reason"] = f"pycocotools unavailable: {exc}"
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return report

    coco_gt = COCO(str(ground_truth_json))
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    for iou_type in iou_types:
        result_path = output_dir / f"predictions_{iou_type}.json"
        results = predictions_to_coco_results(predictions, iou_type)
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(results, f)
        if not results:
            report[iou_type] = _empty_metrics(categories)
            continue
        coco_dt = coco_gt.loadRes(str(result_path))
        coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        report[iou_type] = summarize_coco_eval(coco_eval, categories)
    report["available"] = True
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_metrics_csv(report, output_dir / "metrics.csv")
    return report


def _empty_metrics(categories: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mAP": 0.0,
        "AP50": 0.0,
        "AP75": 0.0,
        "AP_small": 0.0,
        "AP_medium": 0.0,
        "AP_large": 0.0,
        "AR1": 0.0,
        "AR10": 0.0,
        "AR100": 0.0,
        "AR_small": 0.0,
        "AR_medium": 0.0,
        "AR_large": 0.0,
        "per_class_AP": {str(int(cat["id"])): 0.0 for cat in categories},
    }


def write_metrics_csv(report: dict[str, Any], output_path: str | Path) -> None:
    output_path = project_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for iou_type in ("segm", "bbox"):
        metrics = report.get(iou_type)
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if key == "per_class_AP":
                continue
            rows.append({"iou_type": iou_type, "metric": key, "value": value})
        for cat_id, value in metrics.get("per_class_AP", {}).items():
            rows.append({"iou_type": iou_type, "metric": f"AP_class_{cat_id}", "value": value})
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iou_type", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
