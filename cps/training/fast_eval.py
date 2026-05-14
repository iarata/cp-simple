"""Fast in-training validation that logs a tiny fixed probe set to W&B.

The full COCO benchmark is expensive (whole val split + pycocotools). For
shortcut diagnosis we only need a few frozen samples re-evaluated each epoch:
ground truth, predictions, and the decoder's top-query attention map. If the
model is learning copy-paste artifacts, the attention map on the augmented
probes will lock onto the paste boundary instead of the object.

Probes are picked once at training start with a fixed seed so the wandb time
series is comparable across epochs. Three probes come from "normal" classes
(no underrepresented category in their annotations) and three come from images
that contain at least one underrepresented class, with a copy-paste augmentor
applied at probe-build time so we can see the model's response to the same
augmentation pipeline it is being trained against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from loguru import logger
from PIL import Image

from cps.analysis.attention import attention_for_top_query
from cps.augmentations.base import CopyPasteConfig
from cps.augmentations.simple_copy_paste import SimpleCopyPasteAugmentation
from cps.data.coco import COCODataset, sample_to_torch
from cps.data.stats import underrepresented_classes
from cps.data.visualization import overlay_instances
from cps.models.detr import outputs_to_predictions

ProbeSample = tuple[torch.Tensor, dict[str, Any], str]  # (image, target, label)


@dataclass
class ProbeSet:
    """Frozen probe samples and the metadata wandb needs to label them."""

    samples: list[ProbeSample]
    label_to_cat_id: dict[int, int]
    cat_id_to_name: dict[int, str]

    def __len__(self) -> int:
        return len(self.samples)


def _annotations_by_image(coco: dict[str, Any]) -> dict[int, list[int]]:
    by_image: dict[int, list[int]] = {}
    for ann in coco.get("annotations", []):
        by_image.setdefault(int(ann["image_id"]), []).append(int(ann["category_id"]))
    return by_image


def _select_indices(
    images: list[dict[str, Any]],
    anns_by_image: dict[int, list[int]],
    rare_cat_ids: set[int],
    num_normal: int,
    num_rare: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    normal_pool: list[int] = []
    rare_pool: list[int] = []
    for idx, img in enumerate(images):
        cats = set(anns_by_image.get(int(img["id"]), []))
        if not cats:
            continue
        if cats & rare_cat_ids:
            rare_pool.append(idx)
        else:
            normal_pool.append(idx)

    def _pick(pool: list[int], k: int) -> list[int]:
        if not pool or k <= 0:
            return []
        replace = len(pool) < k
        if replace:
            logger.warning(
                "Fast-eval probe pool has only {} samples for requested {}; sampling with replacement.",
                len(pool),
                k,
            )
        chosen = rng.choice(len(pool), size=k, replace=replace)
        return [int(pool[i]) for i in chosen]

    return _pick(normal_pool, num_normal), _pick(rare_pool, num_rare)


def _build_probe_augmentation(
    train_dataset: COCODataset,
    rare_cat_ids: set[int],
    max_paste_objects: int,
) -> SimpleCopyPasteAugmentation | None:
    """A self-contained simple-copy-paste augmentor used only for probes.

    This is independent of whatever augmentation the training run uses, so the
    probe set stays comparable across runs (e.g., a `none` run still gets to
    see "model behavior under copy-paste" without changing training data).
    """

    images = train_dataset.images
    if not images:
        return None
    cat_to_indices: dict[int, list[int]] = {}
    for idx, img in enumerate(images):
        for ann in train_dataset.anns_by_image.get(int(img["id"]), []):
            cat_id = int(ann["category_id"])
            cat_to_indices.setdefault(cat_id, []).append(idx)
    if not cat_to_indices:
        return None
    return SimpleCopyPasteAugmentation(
        donor_getter=train_dataset.get_raw_sample,
        donor_count=len(images),
        category_to_indices=cat_to_indices,
        allowed_category_ids=rare_cat_ids or None,
        config=CopyPasteConfig(
            probability=1.0,
            max_paste_objects=max(1, int(max_paste_objects)),
            paste_scale_jitter=(1.0, 1.0),
        ),
        name="probe_copy_paste",
    )


def _apply_probe_copy_paste(
    sample: dict[str, Any],
    augmentation: SimpleCopyPasteAugmentation,
    seed: int,
    *,
    max_attempts: int = 8,
) -> dict[str, Any]:
    last = sample
    for attempt in range(max_attempts):
        rng = np.random.default_rng(int(seed) + attempt * 1_000_003)
        candidate = augmentation(sample, rng)
        last = candidate
        meta = candidate.get("augmentation_meta", {})
        if bool(meta.get("applied", False)):
            return candidate
    logger.warning(
        "Fast-eval copy-paste probe did not paste after {} attempts; logging unpasted sample.",
        max_attempts,
    )
    return last


def build_probe_set(
    val_dataset: COCODataset,
    train_dataset: COCODataset,
    *,
    num_normal: int = 3,
    num_underrepresented: int = 3,
    seed: int = 1337,
    quantile: float = 0.25,
    max_paste_objects: int = 2,
) -> ProbeSet:
    rare_cat_ids = underrepresented_classes(val_dataset.coco, quantile)
    if not rare_cat_ids:
        logger.warning(
            "Fast-eval probe set: no underrepresented classes found at quantile={}; "
            "all probes will come from the normal pool.",
            quantile,
        )
    anns_by_image = _annotations_by_image(val_dataset.coco)
    rng = np.random.default_rng(int(seed))
    normal_idx, rare_idx = _select_indices(
        val_dataset.images,
        anns_by_image,
        rare_cat_ids,
        num_normal=num_normal,
        num_rare=num_underrepresented if rare_cat_ids else 0,
        rng=rng,
    )

    samples: list[ProbeSample] = []
    # Normal probes: no augmentation.
    for idx in normal_idx:
        raw = val_dataset.get_raw_sample(idx)
        image, target = sample_to_torch(raw, val_dataset.cat_id_to_label)
        samples.append((image, target, "normal"))

    # Underrepresented probes: apply copy-paste so the attention map reflects
    # how the model handles augmented content.
    if rare_idx:
        probe_aug = _build_probe_augmentation(train_dataset, rare_cat_ids, max_paste_objects)
        for k, idx in enumerate(rare_idx):
            raw = val_dataset.get_raw_sample(idx)
            if probe_aug is not None:
                raw = _apply_probe_copy_paste(raw, probe_aug, int(seed) * 9973 + k + 1)
            image, target = sample_to_torch(raw, val_dataset.cat_id_to_label)
            applied = bool(target.get("augmentation_meta", {}).get("applied", False))
            label = "underrepresented+cp" if applied else "underrepresented"
            samples.append((image, target, label))

    logger.info(
        "Fast-eval probe set: {} normal samples, {} underrepresented samples "
        "(with copy-paste applied to underrepresented probes).",
        sum(1 for _, _, kind in samples if kind == "normal"),
        sum(1 for _, _, kind in samples if kind != "normal"),
    )
    return ProbeSet(
        samples=samples,
        label_to_cat_id=val_dataset.label_to_cat_id,
        cat_id_to_name=val_dataset.cat_id_to_name,
    )


def _render_overlay(
    image: torch.Tensor,
    instances: list[dict[str, Any]],
    categories: dict[int, str],
) -> np.ndarray:
    arr = (image.detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return np.asarray(overlay_instances(arr, instances, categories), dtype=np.uint8)


def _gt_instances(target: dict[str, Any]) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for idx in range(len(target["labels"])):
        instances.append(
            {
                "category_id": int(target["category_ids"][idx].item()),
                "bbox_xyxy": target["boxes"][idx].detach().cpu().tolist(),
                "mask": target["masks"][idx].detach().cpu().numpy().astype(bool),
            }
        )
    return instances


def _attention_heatmap(attention: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    att = attention.astype(np.float32)
    if att.ndim == 3:
        att = att.mean(axis=0)
    finite = np.isfinite(att)
    if not finite.any():
        att = np.zeros_like(att, dtype=np.float32)
    else:
        att = np.where(finite, att, 0.0)
        lo, hi = np.percentile(att[finite], [5, 99])
        if hi > lo:
            att = np.clip((att - lo) / (hi - lo), 0.0, 1.0)
            att = np.sqrt(att)
        elif att.max() > 0:
            att = np.ones_like(att, dtype=np.float32)
    pil = Image.fromarray((att * 255).astype(np.uint8), mode="L").resize(
        (size[1], size[0]), Image.BILINEAR
    )
    heat = np.asarray(pil, dtype=np.float32) / 255.0
    rgb = np.zeros((heat.shape[0], heat.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = (255.0 * heat).astype(np.uint8)
    rgb[..., 1] = (220.0 * np.power(heat, 1.4)).astype(np.uint8)
    rgb[..., 2] = (80.0 * np.power(1.0 - heat, 2.0)).astype(np.uint8)
    return rgb


def _attention_overlay(image: torch.Tensor, attention: np.ndarray) -> np.ndarray:
    height, width = int(image.shape[-2]), int(image.shape[-1])
    base = (image.detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    gray = np.dot(base[..., :3], [0.299, 0.587, 0.114])[..., None]
    base = np.repeat(gray, 3, axis=-1).astype(np.float32)
    heat = _attention_heatmap(attention, (height, width))
    blended = (0.28 * base + 0.72 * heat).clip(0, 255).astype(np.uint8)
    return blended


def run_fast_validation(
    *,
    model: torch.nn.Module,
    probe: ProbeSet,
    device: torch.device,
    run: Any,
    epoch: int,
    step: int | None = None,
    score_threshold: float = 0.05,
    max_detections: int = 20,
) -> None:
    """Forward the probe set, log GT/pred/attention overlays to W&B."""

    if not run or not probe.samples:
        return
    try:
        import wandb
    except Exception:  # pragma: no cover
        return

    was_training = model.training
    model.eval()
    gt_logs: list[Any] = []
    pred_logs: list[Any] = []
    attn_logs: list[Any] = []
    table_rows: list[list[Any]] = []
    try:
        with torch.inference_mode():
            for sample_idx, (image, target, kind) in enumerate(probe.samples):
                img_dev = image.to(device)
                target_dev = {
                    k: v.to(device) if torch.is_tensor(v) else v for k, v in target.items()
                }
                outputs = model([img_dev], return_attention=True)
                preds = outputs_to_predictions(
                    outputs,
                    [target_dev],
                    label_to_cat_id=probe.label_to_cat_id,
                    score_threshold=score_threshold,
                    max_detections=max_detections,
                    include_masks=True,
                    target_size_key="size",
                )
                gt_img = _render_overlay(img_dev, _gt_instances(target_dev), probe.cat_id_to_name)
                pred_img = _render_overlay(img_dev, preds, probe.cat_id_to_name)
                image_id = int(target_dev["image_id"].item())
                caption = f"{kind} [{sample_idx}] image_id={image_id}"
                gt_caption = f"GT {caption}"
                pred_caption = f"Pred {caption}"
                gt_log = wandb.Image(gt_img, caption=gt_caption)
                pred_log = wandb.Image(pred_img, caption=pred_caption)
                gt_table = wandb.Image(gt_img, caption=gt_caption)
                pred_table = wandb.Image(pred_img, caption=pred_caption)
                attn_table = None
                gt_logs.append(gt_log)
                pred_logs.append(pred_log)
                attention = attention_for_top_query(outputs, batch_idx=0)
                if attention is not None:
                    attn_img = _attention_overlay(img_dev, attention)
                    attn_caption = f"Attn {caption}"
                    attn_logs.append(wandb.Image(attn_img, caption=attn_caption))
                    attn_table = wandb.Image(attn_img, caption=attn_caption)
                meta = target.get("augmentation_meta", {})
                table_rows.append(
                    [
                        int(epoch),
                        int(sample_idx),
                        int(image_id),
                        kind,
                        bool(meta.get("applied", False)) if isinstance(meta, dict) else False,
                        gt_table,
                        pred_table,
                        attn_table,
                    ]
                )
                del outputs

        payload: dict[str, Any] = {
            "fast_eval/gt": gt_logs,
            "fast_eval/pred": pred_logs,
            "fast_eval/samples": wandb.Table(
                columns=[
                    "epoch",
                    "sample_index",
                    "image_id",
                    "probe_type",
                    "copy_paste_applied",
                    "ground_truth",
                    "prediction",
                    "attention",
                ],
                data=table_rows,
            ),
            "epoch": epoch,
        }
        if attn_logs:
            payload["fast_eval/attention"] = attn_logs
        if step is not None:
            payload["step"] = step
        run.log(payload)
    finally:
        if was_training:
            model.train()
