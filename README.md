# CPS: Copy-Paste Augmentation Studies for COCO Instance Segmentation

`cps` is a reproducible research codebase for comparing four augmentation strategies on COCO2017-style instance segmentation:

1. normal image augmentation,
2. Simple Copy-Paste,
3. PCTNet-style copy-paste harmonization,
4. LBM-style copy-paste harmonization.

The baseline is intentionally simple: a small DETR-style instance segmenter with a CNN backbone, transformer encoder/decoder, fixed object queries, class/box heads, and a query-conditioned mask head. It is designed for controlled augmentation experiments, not for state-of-the-art COCO results.

## Design notes

- COCO annotations are loaded directly from `instances_*.json` files.
- Subsets are nested: the same deterministic image order is used for all requested percentages.
- Copy-paste methods update masks, boxes, areas, labels, image IDs, occlusion, and tiny-mask filtering.
- Premade copy-paste variants keep the original subset and append augmented copies by default, so the training set grows.
- PCTNet and LBM support `harmonizer_backend=libcom` through the in-repository `cps/libcom` wrapper; the default remains a lightweight project-local fallback.
- W&B is optional and disabled by default.
- CUDA, Apple Silicon MPS, and CPU fallback are supported through `train.device=auto`.

## Install

```bash
uv python install 3.12
uv venv --python 3.12
uv sync --extra dev
```

Optional ModelScope download fallback for the in-repository libcom wrapper:

```bash
uv sync --extra dev --extra legacy-libcom
```

The default `harmonizer_backend=local` does not load the in-repository `cps/libcom` wrapper.
When `harmonizer_backend=libcom` is requested, the code imports `cps/libcom` directly rather than the stale PyPI `libcom` package.
Downloaded libcom weights default to `data.nosync/libcom_models`; set `LIBCOM_MODEL_DIR=/path/to/models` to override that location.

## Expected COCO2017 layout

```text
data/raw/coco2017/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
└── val2017/
```

Override paths with Hydra-style CLI overrides when your data lives elsewhere.

## Create nested subsets

```bash
uv run python -m cps.cli make-subsets --config-name subset
```

Example with explicit paths:

```bash
uv run python -m cps.cli make-subsets --config-name subset \
  subset.image_dir=/path/to/train2017 \
  subset.annotation_json=/path/to/annotations/instances_train2017.json \
  subset.percentages='[1,5,10,25]' \
  subset.output_dir=data/processed/coco_subsets
```

Each subset directory contains `annotations.json`, optional symlinked/copied images, `metadata.json`, and distribution/sample visualizations.

## Create premade offline augmentation datasets

Use the same subset seed and percentage, then materialize one no-augmentation dataset plus the three copy-paste variants:

```bash
uv run python -m cps.cli make-premade-subsets --config-name subset \
  subset.percentages='[25]' \
  subset.seed=1337 \
  subset.premade.target_images=all \
  subset.premade.num_workers=0
```

To augment the same random percentage of images across all three copy-paste methods:

```bash
uv run python -m cps.cli make-premade-subsets --config-name subset \
  subset.percentages='[25]' \
  subset.seed=1337 \
  subset.premade.target_images=random_percent \
  subset.premade.random_percent=50
```

To target images that contain underrepresented classes and paste underrepresented-class donors:

```bash
uv run python -m cps.cli make-premade-subsets --config-name subset \
  subset.percentages='[25]' \
  subset.seed=1337 \
  subset.premade.target_images=underrepresented \
  subset.premade.rare_quantile=0.25
```

Premade datasets are written under each subset directory grouped by target policy, for example `premade/all_images/simple_copy_paste`, `premade/random_050pct/simple_copy_paste`, or `premade/underrepresented_q025/simple_copy_paste`. Copy-paste variants keep every original image and add one augmented copy for each image where copy-paste succeeds; set `subset.premade.append_augmented=false` to use the older replace-in-place behavior. Each variant writes before/after count, relative-frequency, and delta plots under `visualizations/`. `subset.premade.num_workers=0` uses an automatic worker count capped at 8, or one worker when `harmonizer_backend=libcom`; set it to `1` for sequential debugging or to an explicit worker count for your machine. Train from one of them with online augmentation disabled:

```bash
uv run python -m cps.cli train --config-name train \
  subset.percent=25 \
  subset.premade.target_images=underrepresented \
  subset.premade.train_variant=simple_copy_paste \
  augmentation=none
```

You can also address the folder directly:

```bash
uv run python -m cps.cli train --config-name train \
  subset.percent=25 \
  subset.premade.train_variant=underrepresented_q025/simple_copy_paste \
  augmentation=none
```

Full training configs for the 25% premade datasets are available under `configs/train_premade`:

```bash
uv run python -m cps.cli train --config-name train_premade/none_online_normal
uv run python -m cps.cli train --config-name train_premade/simple_copy_paste
uv run python -m cps.cli train --config-name train_premade/pctnet_copy_paste
uv run python -m cps.cli train --config-name train_premade/lbm_copy_paste
```

The `none_online_normal` config trains from `premade/underrepresented_q025/none` and applies online normal Albumentations. The copy-paste configs train directly from the premade augmented datasets with `augmentation=none`.

## Preview augmentations

```bash
uv run python -m cps.cli preview-augmentations --config-name augment
```

This writes grids with original, normal, Simple Copy-Paste, PCTNet-style, and LBM-style images under `data/interim/augmentation_previews`.

## Train one experiment

```bash
uv run python -m cps.cli train --config-name train augmentation=normal subset.percent=100
uv run python -m cps.cli train --config-name train augmentation=simple_copy_paste subset.percent=10
uv run python -m cps.cli train --config-name train augmentation=pctnet_copy_paste subset.percent=10 augmentation.target_policy=underrepresented
uv run python -m cps.cli train --config-name train augmentation=lbm_copy_paste subset.percent=10 augmentation.target_policy=underrepresented
```

Checkpoints and config snapshots are saved under `models/runs/<method>_<subset>_seed_<seed>/` by default.

Large server runs with many workers and large batches use
`train.multiprocessing_sharing_strategy=file_system` by default to avoid PyTorch
DataLoader shared-memory file descriptor exhaustion. If a server still reports
`Too many open files`, raise the shell limit before launching training, for
example `ulimit -n 65535`, or reduce the number of queued batches with
`train.num_workers`, `train.prefetch_factor`, or `eval.num_workers` overrides.

## Evaluate a checkpoint

```bash
uv run python -m cps.cli evaluate --config-name eval checkpoint=models/runs/simple_copy_paste_pct_010_seed_1337/checkpoint_best.pt
```

Evaluation writes COCO-style metrics, CSV reports, GT/prediction grids, and attention maps under `data/processed/evaluation` unless overridden.

## Run a grid

```bash
make grid
```

or explicitly:

```bash
for pct in 1 5 10 25; do
  for aug in normal simple_copy_paste pctnet_copy_paste lbm_copy_paste; do
    uv run python -m cps.cli train --config-name train augmentation=$aug subset.percent=$pct
  done
done
```

## Generate final reports

```bash
uv run python -m cps.cli report --config-name report
```

Outputs include `comparison_metrics.csv`, mAP-by-method plots, mAP-vs-subset plots, and per-class AP deltas versus normal augmentation when baseline reports are available.

## Tiny fixture smoke test

The codebase includes a synthetic COCO-style fixture that does not require COCO2017:

```bash
uv run python -m cps.cli create-tiny-fixture --output-dir data/interim/tiny_coco
make smoke-test
```

## W&B

Enable W&B with overrides:

```bash
uv run python -m cps.cli train --config-name train \
  wandb.enabled=true wandb.project=cps-copy-paste wandb.mode=online
```

Training logs every configured train step's loss components (`train/loss_ce`,
`train/loss_bbox`, `train/loss_giou`, `train/loss_mask`, `train/loss_dice`,
and total `train/loss`), validation loss components, segmentation and bbox COCO
metrics including `val/bbox_mAP`, W&B COCO summary/per-class AP plots, and saved
validation PNGs such as GT-vs-pred grids and attention maps. Use
`wandb.mode=offline` for air-gapped runs. Set `wandb.max_visualizations=0` to
skip image uploads.

## Shortcut-learning analysis

Validation saves decoder cross-attention overlays for a configurable number of samples:

```bash
uv run python -m cps.cli evaluate --config-name eval \
  checkpoint=models/runs/.../checkpoint_best.pt \
  analysis.attention_samples=8
```

The `cps.analysis.shortcuts` module also provides boundary-attention diagnostics for samples where pasted-object masks are available.
