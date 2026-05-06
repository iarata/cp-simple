.PHONY: requirements clean lint format create_environment data make-subsets make-premade-subsets analyze-subsets preview-augmentations train evaluate grid report smoke-test

PYTHON_VERSION ?= 3.12
UV ?= uv

requirements:
	$(UV) sync --extra dev

create_environment:
	$(UV) python install $(PYTHON_VERSION)
	$(UV) venv --python $(PYTHON_VERSION)

clean:
	rm -rf .ruff_cache .pytest_cache htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff format .

format-check:
	$(UV) run ruff format --check .

data:
	$(UV) run python -m cps.cli create-tiny-fixture --output-dir data/interim/tiny_coco

make-subsets:
	$(UV) run python -m cps.cli make-subsets --config-name subset

make-premade-subsets:
	$(UV) run python -m cps.cli make-premade-subsets --config-name subset

analyze-subsets:
	$(UV) run python -m cps.cli analyze-subsets --config-name subset

preview-augmentations:
	$(UV) run python -m cps.cli preview-augmentations --config-name augment

train:
	$(UV) run python -m cps.cli train --config-name train augmentation=normal subset.percent=100

evaluate:
	$(UV) run python -m cps.cli evaluate --config-name eval checkpoint=$(CHECKPOINT)

grid:
	@for pct in 1 5 10 25; do \
	  for aug in normal simple_copy_paste pctnet_copy_paste lbm_copy_paste; do \
	    $(UV) run python -m cps.cli train --config-name train augmentation=$$aug subset.percent=$$pct; \
	  done; \
	done

report:
	$(UV) run python -m cps.cli report --config-name report

smoke-test:
	$(UV) run python -m cps.cli create-tiny-fixture --output-dir data/interim/tiny_coco
	$(UV) run python -m cps.cli make-subsets --config-name subset dataset.root=data/interim/tiny_coco/train dataset.train_images=data/interim/tiny_coco/train/images dataset.train_annotations=data/interim/tiny_coco/train/annotations/instances.json subset.output_dir=data/interim/tiny_subsets subset.percentages='[25,50]'
	$(UV) run python -m cps.cli preview-augmentations --config-name augment dataset.train_images=data/interim/tiny_coco/train/images dataset.train_annotations=data/interim/tiny_coco/train/annotations/instances.json train.device=cpu augmentation_preview.num_samples=2 augmentation_preview.output_dir=data/interim/tiny_previews
	$(UV) run python -m cps.cli train --config-name train dataset.train_images=data/interim/tiny_coco/train/images dataset.train_annotations=data/interim/tiny_coco/train/annotations/instances.json dataset.val_images=data/interim/tiny_coco/val/images dataset.val_annotations=data/interim/tiny_coco/val/annotations/instances.json train.device=cpu train.deterministic=false train.num_threads=1 train.num_interop_threads=1 train.epochs=1 train.batch_size=1 train.num_workers=0 train.max_train_batches=1 eval.batch_size=1 eval.max_batches=1 eval.visualize_batches=1 analysis.attention_samples=1 model.hidden_dim=16 model.nheads=4 model.dim_feedforward=32 model.num_encoder_layers=1 model.num_decoder_layers=1 model.num_queries=6 model.mask_loss_size=32 augmentation=normal augmentation.use_albumentations=false train.output_dir=models/smoke
