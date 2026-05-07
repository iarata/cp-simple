"""Training and evaluation entrypoints."""

from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from cps.augmentations import build_augmentation
from cps.data.coco import COCODataset, collate_fn
from cps.data.subsets import percent_slug, premade_train_paths, premade_train_variant
from cps.models.detr import build_model_and_criterion
from cps.paths import project_path
from cps.training.checkpoints import load_checkpoint, save_checkpoint
from cps.training.validate import move_targets_to_device, validation_loop
from cps.utils.device import device_info, get_device
from cps.utils.seed import seed_everything
from cps.utils.wandb import init_wandb, log_artifact, log_validation_outputs


def configure_torch_multiprocessing(cfg: Any) -> None:
    """Configure multiprocessing before DataLoader workers are created."""

    strategy = str(
        getattr(cfg.train, "multiprocessing_sharing_strategy", "file_system") or ""
    ).strip()
    if not strategy or strategy.lower() in {"default", "none"}:
        return
    available = torch.multiprocessing.get_all_sharing_strategies()
    if strategy not in available:
        raise ValueError(
            f"Unsupported train.multiprocessing_sharing_strategy={strategy!r}. "
            f"Available strategies: {sorted(available)}"
        )
    if torch.multiprocessing.get_sharing_strategy() != strategy:
        torch.multiprocessing.set_sharing_strategy(strategy)
    logger.info(
        "Torch multiprocessing sharing strategy: {}", torch.multiprocessing.get_sharing_strategy()
    )


def configure_torch_threads(cfg: Any) -> None:
    """Optionally cap CPU thread pools for reproducible smoke tests and small machines."""

    num_threads = int(getattr(cfg.train, "num_threads", 0) or 0)
    num_interop_threads = int(getattr(cfg.train, "num_interop_threads", 0) or 0)
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    if num_interop_threads > 0:
        # PyTorch only allows this before parallel work starts in a process.
        with suppress(RuntimeError):
            torch.set_num_interop_threads(num_interop_threads)


def _pin_memory_enabled(cfg: Any) -> bool:
    configured = getattr(cfg.train, "pin_memory", None)
    if configured is None:
        return torch.cuda.is_available()
    return bool(configured)


def _num_workers(cfg: Any, split: str) -> int:
    configured = getattr(cfg.eval, "num_workers", None) if split == "val" else None
    if configured is None:
        configured = cfg.train.num_workers
    num_workers = int(configured)
    if num_workers < 0:
        raise ValueError(f"{split} num_workers must be >= 0.")
    return num_workers


def _dataloader_kwargs(cfg: Any, split: str, *, shuffle: bool) -> dict[str, Any]:
    num_workers = _num_workers(cfg, split)
    kwargs: dict[str, Any] = {
        "batch_size": int(cfg.eval.batch_size) if split == "val" else int(cfg.train.batch_size),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": _pin_memory_enabled(cfg),
    }
    if num_workers > 0:
        prefetch_factor = int(getattr(cfg.train, "prefetch_factor", 1) or 1)
        if prefetch_factor < 1:
            raise ValueError("train.prefetch_factor must be >= 1 when num_workers > 0.")
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = bool(getattr(cfg.train, "persistent_workers", False))
    return kwargs


def resolve_train_paths(cfg: Any) -> tuple[Path, Path]:
    percent = float(getattr(cfg.subset, "percent", 100))
    premade_paths = premade_train_paths(cfg.subset, percent)
    if premade_paths is not None:
        return premade_paths
    if percent >= 100:
        return project_path(cfg.dataset.train_images), project_path(cfg.dataset.train_annotations)
    slug = percent_slug(percent)
    subset_dir = project_path(cfg.subset.output_dir) / f"{slug}_seed_{int(cfg.subset.seed)}"
    metadata_path = subset_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        return project_path(metadata["image_dir"]), project_path(metadata["annotation_json"])
    ann_path = subset_dir / "annotations.json"
    img_dir = subset_dir / "images"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"Requested subset.percent={percent}, but {ann_path} does not exist. "
            "Run `uv run python -m cps.cli make-subsets --config-name subset` first."
        )
    return img_dir if img_dir.exists() else project_path(cfg.dataset.train_images), ann_path


def build_datasets(cfg: Any) -> tuple[COCODataset, COCODataset]:
    train_images, train_annotations = resolve_train_paths(cfg)
    train_dataset = COCODataset(
        image_dir=train_images,
        annotation_json=train_annotations,
        augmentation=None,
        seed=int(cfg.train.seed),
        max_images=getattr(cfg.train, "max_train_images", None),
    )
    premade_variant = premade_train_variant(cfg.subset)
    if premade_variant and str(getattr(cfg.augmentation, "name", "none")) != "none":
        logger.warning(
            "Training from premade subset variant '{}' with online augmentation '{}'. "
            "Use augmentation=none to train only on the premade dataset.",
            premade_variant,
            cfg.augmentation.name,
        )
    augmentation = build_augmentation(
        cfg.augmentation,
        train_dataset.coco,
        train_dataset.images,
        donor_getter=train_dataset.get_raw_sample,
        device=str(get_device(str(cfg.train.device))),
    )
    train_dataset.augmentation = augmentation
    val_dataset = COCODataset(
        image_dir=cfg.dataset.val_images,
        annotation_json=cfg.dataset.val_annotations,
        augmentation=None,
        seed=int(cfg.train.seed),
        max_images=getattr(cfg.train, "max_val_images", None),
    )
    return train_dataset, val_dataset


def build_dataloaders(cfg: Any) -> tuple[DataLoader, DataLoader, COCODataset, COCODataset]:
    train_dataset, val_dataset = build_datasets(cfg)
    train_kwargs = _dataloader_kwargs(cfg, "train", shuffle=True)
    val_kwargs = _dataloader_kwargs(cfg, "val", shuffle=False)
    logger.info(
        "DataLoader settings - train workers: {}, val workers: {}, prefetch: {}, "
        "persistent: {}, pin_memory: {}",
        train_kwargs["num_workers"],
        val_kwargs["num_workers"],
        train_kwargs.get("prefetch_factor", "n/a"),
        train_kwargs.get("persistent_workers", False),
        train_kwargs["pin_memory"],
    )
    train_loader = DataLoader(train_dataset, **train_kwargs)
    val_loader = DataLoader(val_dataset, **val_kwargs)
    return train_loader, val_loader, train_dataset, val_dataset


def validation_checkpoint_score(metrics: dict[str, Any]) -> float:
    for iou_type in ("segm", "bbox"):
        value = metrics.get(iou_type, {}).get("mAP")
        if value is not None:
            return float(value)
    loss = metrics.get("losses", {}).get("loss")
    if loss is not None:
        return -float(loss)
    return 0.0


def run_training(cfg: Any) -> dict[str, Any]:
    configure_torch_multiprocessing(cfg)
    configure_torch_threads(cfg)
    seed_everything(int(cfg.train.seed), deterministic=bool(cfg.train.deterministic))
    info = device_info(str(cfg.train.device))
    device = get_device(str(cfg.train.device))
    logger.info("Using device: {} ({})", info.selected, info.name)
    logger.info(
        "Dataset {} - Batch size: {}, Epochs: {}, Eval every: {}",
        cfg.dataset.name,
        cfg.train.batch_size,
        cfg.train.epochs,
        cfg.train.eval_every,
    )
    run = init_wandb(cfg, job_type="train")
    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders(cfg)
    model, criterion = build_model_and_criterion(cfg.model, num_classes=train_dataset.num_classes)
    model.to(device)
    criterion.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay)
    )
    start_epoch = 0
    if getattr(cfg.train, "resume", None):
        payload = load_checkpoint(cfg.train.resume, model, optimizer, map_location=device)
        start_epoch = int(payload.get("epoch", -1)) + 1
        logger.info("Resumed from {} at epoch {}", cfg.train.resume, start_epoch)

    output_dir = experiment_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)
    metrics: dict[str, Any] = {}
    best_score = float("-inf")
    for epoch in range(start_epoch, int(cfg.train.epochs)):
        train_dataset.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            cfg=cfg,
            run=run,
        )
        metrics = {"train": train_metrics, "epoch": epoch}
        should_eval = (epoch + 1) % int(cfg.train.eval_every) == 0 or epoch == int(
            cfg.train.epochs
        ) - 1
        if should_eval:
            val_dir = output_dir / f"val_epoch_{epoch:03d}"
            val_metrics = validation_loop(
                model=model,
                criterion=criterion,
                dataloader=val_loader,
                device=device,
                label_to_cat_id=val_dataset.label_to_cat_id,
                categories=val_dataset.cat_id_to_name,
                annotation_json=val_dataset.annotation_json,
                output_dir=val_dir,
                score_threshold=float(cfg.eval.score_threshold),
                max_detections=int(cfg.eval.max_detections),
                max_batches=getattr(cfg.eval, "max_batches", None),
                visualize_batches=int(cfg.eval.visualize_batches),
                visualize_max_images=int(getattr(cfg.eval, "visualize_max_images", 4)),
                attention_samples=int(cfg.analysis.attention_samples),
                forward_batch_size=getattr(cfg.eval, "forward_batch_size", None),
                mode=str(getattr(cfg.eval, "mode", "full")),
                iou_types=tuple(getattr(cfg.eval, "iou_types", ("segm", "bbox"))),
                empty_cache=bool(getattr(cfg.eval, "empty_cache", True)),
                empty_cache_between_chunks=bool(
                    getattr(cfg.eval, "empty_cache_between_chunks", False)
                ),
            )
            metrics["val"] = val_metrics
            val_score = validation_checkpoint_score(val_metrics)
            log_validation_outputs(
                run,
                val_metrics,
                val_dir,
                epoch=epoch,
                max_visualizations=int(getattr(cfg.wandb, "max_visualizations", 16)),
                log_plots=bool(getattr(cfg.wandb, "log_plots", True)),
                log_per_class_ap=bool(getattr(cfg.wandb, "log_per_class_ap", True)),
            )
            if val_score > best_score:
                best_score = val_score
                best_path = save_checkpoint(
                    output_dir / "checkpoint_best.pt", model, optimizer, epoch, cfg, metrics
                )
                log_artifact(run, best_path, "model")
        ckpt_path = save_checkpoint(
            output_dir / f"checkpoint_epoch_{epoch:03d}.pt", model, optimizer, epoch, cfg, metrics
        )
        save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, epoch, cfg, metrics)
        if bool(getattr(cfg.wandb, "log_checkpoints", False)):
            log_artifact(run, ckpt_path, "model")
    if run:
        run.finish()
    return {"output_dir": str(output_dir), "metrics": metrics}


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    cfg: Any,
    run: Any,
) -> dict[str, float]:
    model.train()
    criterion.train()
    loss_sums: dict[str, float] = {}
    num_batches = 0
    progress = tqdm(data_loader, desc=f"train epoch {epoch}", leave=False)
    for step, (images, targets) in enumerate(progress):
        images = [img.to(device) for img in images]
        targets = move_targets_to_device(targets, device)
        outputs = model(images)
        losses = criterion(outputs, targets)
        loss = losses["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if float(cfg.train.gradient_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.gradient_clip_norm))
        optimizer.step()
        for key, value in losses.items():
            loss_sums[key] = loss_sums.get(key, 0.0) + float(value.detach().cpu())
        num_batches += 1
        progress.set_postfix(loss=float(loss.detach().cpu()))
        global_step = epoch * len(data_loader) + step
        if run and step % int(cfg.train.log_every) == 0:
            payload = {
                "train/lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
                "step": global_step,
            }
            for key, value in losses.items():
                payload[f"train/{key}"] = float(value.detach().cpu())
            run.log(payload)
        max_batches = getattr(cfg.train, "max_train_batches", None)
        if max_batches is not None and step + 1 >= int(max_batches):
            break
    return {key: value / max(num_batches, 1) for key, value in loss_sums.items()}


def experiment_output_dir(cfg: Any) -> Path:
    method = str(cfg.augmentation.name)
    percent = float(getattr(cfg.subset, "percent", 100))
    seed = int(cfg.train.seed)
    name = str(getattr(cfg.train, "run_name", ""))
    if not name:
        premade_variant = premade_train_variant(cfg.subset)
        if premade_variant:
            method = (
                f"premade_{premade_variant}"
                if method == "none"
                else f"premade_{premade_variant}_online_{method}"
            )
        name = f"{method}_{percent_slug(percent)}_seed_{seed}"
    return project_path(cfg.train.output_dir) / name


def run_evaluation(cfg: Any) -> dict[str, Any]:
    configure_torch_multiprocessing(cfg)
    configure_torch_threads(cfg)
    seed_everything(int(cfg.train.seed), deterministic=bool(cfg.train.deterministic))
    device = get_device(str(cfg.train.device))
    run = init_wandb(cfg, job_type="eval")
    val_dataset = COCODataset(
        image_dir=cfg.dataset.val_images,
        annotation_json=cfg.dataset.val_annotations,
        augmentation=None,
        seed=int(cfg.train.seed),
        max_images=getattr(cfg.eval, "max_images", None),
    )
    val_loader = DataLoader(
        val_dataset,
        **_dataloader_kwargs(cfg, "val", shuffle=False),
    )
    model, criterion = build_model_and_criterion(cfg.model, num_classes=val_dataset.num_classes)
    model.to(device)
    criterion.to(device)
    checkpoint = getattr(cfg.eval, "checkpoint", None) or getattr(cfg, "checkpoint", None)
    if not checkpoint:
        raise ValueError("Evaluation requires eval.checkpoint=... or checkpoint=...")
    load_checkpoint(checkpoint, model, optimizer=None, map_location=device)
    output_dir = project_path(cfg.eval.output_dir)
    metrics = validation_loop(
        model=model,
        criterion=criterion,
        dataloader=val_loader,
        device=device,
        label_to_cat_id=val_dataset.label_to_cat_id,
        categories=val_dataset.cat_id_to_name,
        annotation_json=val_dataset.annotation_json,
        output_dir=output_dir,
        score_threshold=float(cfg.eval.score_threshold),
        max_detections=int(cfg.eval.max_detections),
        max_batches=getattr(cfg.eval, "max_batches", None),
        visualize_batches=int(cfg.eval.visualize_batches),
        visualize_max_images=int(getattr(cfg.eval, "visualize_max_images", 4)),
        attention_samples=int(cfg.analysis.attention_samples),
        forward_batch_size=getattr(cfg.eval, "forward_batch_size", None),
        mode=str(getattr(cfg.eval, "mode", "full")),
        iou_types=tuple(getattr(cfg.eval, "iou_types", ("segm", "bbox"))),
        empty_cache=bool(getattr(cfg.eval, "empty_cache", True)),
        empty_cache_between_chunks=bool(getattr(cfg.eval, "empty_cache_between_chunks", False)),
    )
    log_validation_outputs(
        run,
        metrics,
        output_dir,
        max_visualizations=int(getattr(cfg.wandb, "max_visualizations", 16)),
        log_plots=bool(getattr(cfg.wandb, "log_plots", True)),
        log_per_class_ap=bool(getattr(cfg.wandb, "log_per_class_ap", True)),
    )
    if run:
        run.finish()
    return metrics
