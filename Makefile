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

.PHONY: download-coco
download-coco:
		@test -n "$(OUTPUT)" || (echo "OUTPUT is required. Usage: make download-coco-aria2 OUTPUT=/path/to/file.zip" >&2; exit 2)
		aria2c \
			--max-connection-per-server=16 \
			--split=16 \
			--min-split-size=1M \
			--continue \
			--console-log-level=error \
			-o $(OUTPUT) \
			https://www.kaggle.com/api/v1/datasets/download/awsaf49/coco-2017-dataset

make-all-subsets:
	$(UV) run python -m cps.cli make-premade-subsets --config-name subset subset.percentages='[25]' subset.seed=1337 subset.premade.num_workers=0 subset.premade.target_images=underrepresented subset.premade.methods='[none,simple_copy_paste,pctnet_copy_paste,lbm_copy_paste]' subset.premade.image_mode=copy subset.premade.copy_paste.harmonizer_backend=libcom

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
