"""Training and evaluation entrypoints."""

from __future__ import annotations

import json
import math
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
from cps.models import build_model_and_criterion
from cps.paths import project_path
from cps.training.checkpoints import load_checkpoint, save_checkpoint
from cps.training.fast_eval import ProbeSet, build_probe_set, run_fast_validation
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
    image_size = getattr(cfg.dataset, "image_size", None)
    train_dataset = COCODataset(
        image_dir=train_images,
        annotation_json=train_annotations,
        augmentation=None,
        seed=int(cfg.train.seed),
        max_images=getattr(cfg.train, "max_train_images", None),
        image_size=image_size,
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
        image_size=image_size,
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


def build_optimizer(model: torch.nn.Module, cfg: Any) -> torch.optim.Optimizer:
    """Build AdamW with a lower LR group for the pretrained backbone.

    Without this, a pretrained backbone (timm) is destroyed at the head's LR
    in the first epoch, exactly the symptom of "loss decreases very slowly."
    """

    base_lr = float(cfg.train.lr)
    weight_decay = float(cfg.train.weight_decay)
    backbone_lr_cfg = getattr(cfg.train, "backbone_lr", None)
    backbone_lr = float(backbone_lr_cfg) if backbone_lr_cfg is not None else base_lr
    backbone_params: list[torch.nn.Parameter] = []
    head_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)
    param_groups = [{"params": head_params, "lr": base_lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    logger.info(
        "Optimizer LR groups - head: {} ({} params), backbone: {} ({} params)",
        base_lr,
        sum(p.numel() for p in head_params),
        backbone_lr,
        sum(p.numel() for p in backbone_params),
    )
    return torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)


def _build_probe_set_if_enabled(
    cfg: Any, train_dataset: COCODataset, val_dataset: COCODataset
) -> ProbeSet | None:
    """Build the fixed 6-sample probe set unless the user disables it."""

    fast_cfg = getattr(cfg.eval, "fast_eval", None)
    enabled = True
    if fast_cfg is not None:
        enabled = bool(getattr(fast_cfg, "enabled", True))
    if not enabled:
        return None
    num_normal = int(getattr(fast_cfg, "num_normal", 3)) if fast_cfg is not None else 3
    num_rare = int(getattr(fast_cfg, "num_underrepresented", 3)) if fast_cfg is not None else 3
    probe_seed = (
        int(getattr(fast_cfg, "probe_seed", cfg.train.seed))
        if fast_cfg is not None
        else int(cfg.train.seed)
    )
    quantile = (
        float(getattr(fast_cfg, "underrepresented_quantile", 0.25))
        if fast_cfg is not None
        else 0.25
    )
    max_paste_objects = (
        int(getattr(fast_cfg, "max_paste_objects", 2)) if fast_cfg is not None else 2
    )
    try:
        return build_probe_set(
            val_dataset=val_dataset,
            train_dataset=train_dataset,
            num_normal=num_normal,
            num_underrepresented=num_rare,
            seed=probe_seed,
            quantile=quantile,
            max_paste_objects=max_paste_objects,
        )
    except Exception as exc:  # pragma: no cover - never break training on probe issues
        logger.warning("Fast-eval probe set build failed; disabling: {}", exc)
        return None


def _fast_eval_every(cfg: Any) -> int:
    fast_cfg = getattr(cfg.eval, "fast_eval", None)
    if fast_cfg is not None:
        return int(getattr(fast_cfg, "every", 1) or 0)
    return int(getattr(cfg.eval, "fast_eval_every", 1) or 0)


def _fast_eval_score_threshold(cfg: Any) -> float:
    fast_cfg = getattr(cfg.eval, "fast_eval", None)
    if fast_cfg is not None:
        return float(getattr(fast_cfg, "score_threshold", cfg.eval.score_threshold))
    return float(getattr(cfg.eval, "score_threshold", 0.05))


def _fast_eval_max_detections(cfg: Any) -> int:
    fast_cfg = getattr(cfg.eval, "fast_eval", None)
    if fast_cfg is not None:
        return int(getattr(fast_cfg, "max_detections", 20))
    return 20


def _should_run_full_validation(epoch: int, total_epochs: int, eval_every: int) -> bool:
    is_final_epoch = epoch == total_epochs - 1
    return is_final_epoch or (eval_every > 0 and (epoch + 1) % eval_every == 0)


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
    eval_every = int(getattr(cfg.train, "eval_every", 0) or 0)
    logger.info(
        "Dataset {} - Batch size: {}, Epochs: {}, Eval every: {}",
        cfg.dataset.name,
        cfg.train.batch_size,
        cfg.train.epochs,
        eval_every if eval_every > 0 else "final_only",
    )
    run = init_wandb(cfg, job_type="train")
    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders(cfg)
    model, criterion = build_model_and_criterion(
        cfg.model, num_classes=train_dataset.num_classes, train_cfg=cfg.train
    )
    model.to(device)
    criterion.to(device)
    optimizer = build_optimizer(model, cfg)
    start_epoch = 0
    if getattr(cfg.train, "resume", None):
        payload = load_checkpoint(cfg.train.resume, model, optimizer, map_location=device)
        start_epoch = int(payload.get("epoch", -1)) + 1
        logger.info("Resumed from {} at epoch {}", cfg.train.resume, start_epoch)

    output_dir = experiment_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)
    fast_eval_every = _fast_eval_every(cfg)
    probe = (
        _build_probe_set_if_enabled(cfg, train_dataset, val_dataset)
        if fast_eval_every > 0
        else None
    )
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
        if hasattr(criterion, "update"):
            criterion.update()
        metrics = {"train": train_metrics, "epoch": epoch}
        is_final_epoch = epoch == int(cfg.train.epochs) - 1
        if probe is not None and fast_eval_every > 0 and (epoch + 1) % fast_eval_every == 0:
            try:
                run_fast_validation(
                    model=model,
                    probe=probe,
                    device=device,
                    run=run,
                    epoch=epoch,
                    score_threshold=_fast_eval_score_threshold(cfg),
                    max_detections=_fast_eval_max_detections(cfg),
                )
            except Exception as exc:  # pragma: no cover - W&B/media failures are non-fatal
                logger.warning("Fast validation failed at epoch {}; continuing: {}", epoch, exc)
        should_eval = _should_run_full_validation(epoch, int(cfg.train.epochs), eval_every)
        if should_eval:
            val_dir_name = (
                f"final_eval_epoch_{epoch:03d}" if is_final_epoch else f"val_epoch_{epoch:03d}"
            )
            val_dir = output_dir / val_dir_name
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


def _amp_dtype(cfg: Any, device: torch.device) -> torch.dtype | None:
    """Resolve mixed-precision dtype. Returns None if AMP is disabled."""

    if device.type != "cuda":
        return None
    amp_cfg = getattr(cfg.train, "amp", None)
    if amp_cfg is None or amp_cfg is False or str(amp_cfg).lower() in {"false", "off", "none"}:
        return None
    if amp_cfg is True or str(amp_cfg).lower() in {"true", "on", "auto", "bfloat16", "bf16"}:
        return torch.bfloat16
    if str(amp_cfg).lower() in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"Unknown train.amp={amp_cfg!r}; use true/false/bfloat16/float16.")


def _warmup_factor(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0 or step >= warmup_steps:
        return 1.0
    return float(step + 1) / float(warmup_steps)


def _lr_factor(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    schedule: str,
    min_ratio: float,
) -> float:
    """LR scale at ``step``. Warmup ramps linearly, then optional cosine decay.

    Without post-warmup decay the loss plateaus at a saddle — that was a major
    cause of the DETR run stalling at loss 3.7. Cosine is the default schedule
    for the MaskRCNN swin-t configs; older runs that omit the field stay on
    ``warmup_only`` and behave exactly as before.
    """

    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    if schedule == "warmup_only" or total_steps <= warmup_steps:
        return 1.0
    if schedule == "cosine":
        denom = max(1, total_steps - warmup_steps)
        progress = (step - warmup_steps) / float(denom)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_ratio + (1.0 - min_ratio) * cosine)
    raise ValueError(f"Unknown train.lr_schedule={schedule!r}; use warmup_only or cosine.")


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
    amp_dtype = _amp_dtype(cfg, device)
    use_grad_scaler = amp_dtype == torch.float16
    grad_scaler = torch.amp.GradScaler("cuda") if use_grad_scaler else None
    warmup_steps = int(getattr(cfg.train, "warmup_steps", 0) or 0)
    schedule = str(getattr(cfg.train, "lr_schedule", "warmup_only") or "warmup_only").lower()
    min_lr_ratio = float(getattr(cfg.train, "min_lr_ratio", 0.01))
    total_epochs = int(cfg.train.epochs)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    log_every = max(int(cfg.train.log_every), 1)
    loss_sums_gpu: dict[str, torch.Tensor] = {}
    num_batches = 0
    progress = tqdm(data_loader, desc=f"train epoch {epoch}", leave=False)
    last_logged_losses: dict[str, float] = {}
    steps_per_epoch = len(data_loader)
    total_steps = max(1, total_epochs * steps_per_epoch)
    for step, (images, targets) in enumerate(progress):
        global_step = epoch * steps_per_epoch + step
        factor = _lr_factor(
            global_step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            schedule=schedule,
            min_ratio=min_lr_ratio,
        )
        for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=False):
            group["lr"] = base_lr * factor
        images = [img.to(device, non_blocking=True) for img in images]
        targets = move_targets_to_device(targets, device)
        optimizer.zero_grad(set_to_none=True)
        if amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(images, targets=targets)
                losses = criterion(outputs, targets)
            loss = losses["loss"]
        else:
            outputs = model(images, targets=targets)
            losses = criterion(outputs, targets)
            loss = losses["loss"]
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            if float(cfg.train.gradient_clip_norm) > 0:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(cfg.train.gradient_clip_norm)
                )
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            if float(cfg.train.gradient_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(cfg.train.gradient_clip_norm)
                )
            optimizer.step()
        # Accumulate per-key losses on the GPU. Pulling values to CPU via .item()
        # every step forces a host sync that single-handedly halves throughput
        # on a fast GPU like the RTX PRO 6000. We only sync on log boundaries.
        for key, value in losses.items():
            detached = value.detach()
            if key in loss_sums_gpu:
                loss_sums_gpu[key] = loss_sums_gpu[key] + detached
            else:
                loss_sums_gpu[key] = detached.clone()
        num_batches += 1
        if step % log_every == 0:
            payload_tensors = {key: value.detach() for key, value in losses.items()}
            payload = {key: float(value.item()) for key, value in payload_tensors.items()}
            last_logged_losses = payload
            progress.set_postfix(loss=payload.get("loss", 0.0))
            if run:
                wandb_payload: dict[str, Any] = {
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    "step": global_step,
                }
                for key, val in payload.items():
                    wandb_payload[f"train/{key}"] = val
                run.log(wandb_payload)
        max_batches = getattr(cfg.train, "max_train_batches", None)
        if max_batches is not None and step + 1 >= int(max_batches):
            break
    # Restore configured LRs so subsequent epochs see post-warmup values.
    for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=False):
        group["lr"] = base_lr
    if num_batches == 0:
        return last_logged_losses
    return {key: float((value / num_batches).item()) for key, value in loss_sums_gpu.items()}


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
        image_size=getattr(cfg.dataset, "image_size", None),
    )
    val_loader = DataLoader(
        val_dataset,
        **_dataloader_kwargs(cfg, "val", shuffle=False),
    )
    model, criterion = build_model_and_criterion(
        cfg.model, num_classes=val_dataset.num_classes, train_cfg=cfg.train
    )
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
