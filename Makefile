PYTHON ?= python

.PHONY: install test audit data sft-dry dpo-dry eval benchmark verify-all

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest -v

audit:
	$(PYTHON) -m src.eval.run_baseline_check

data:
	$(PYTHON) -m src.data_gen.synthesizer --mode procedural --sft-count 800 --dpo-count 800 --holdout-count 80
	$(PYTHON) -m src.data_gen.validate_datasets

sft-dry:
	$(PYTHON) -m src.train.train_sft --config configs/sft_config.yaml --dry-run

dpo-dry:
	$(PYTHON) -m src.train.train_dpo --config configs/dpo_config.yaml --dry-run

eval:
	$(PYTHON) -m src.eval.evaluate_checkpoint --mode simulate-stage --stage all

benchmark:
	$(PYTHON) -m src.benchmark.kv_cache_profiler
	$(PYTHON) -m src.benchmark.load_tester --mode simulate --concurrency 1 8 16 32

verify-all:
	$(PYTHON) -m src.cli verify-all
