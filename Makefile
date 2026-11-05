.PHONY: install test lint sft-lora sft-qlora dpo-lora dpo-qlora serve benchmark api

install:
	pip install -e '.[train,spark,serve,eval,dev]'

test:
	pytest -q

lint:
	ruff check src tests scripts

sft-lora:
	python scripts/train_sft.py --config configs/sft_lora.yaml

sft-qlora:
	python scripts/train_sft.py --config configs/sft_qlora.yaml

dpo-lora:
	python scripts/train_dpo.py --config configs/dpo_lora.yaml

dpo-qlora:
	python scripts/train_dpo.py --config configs/dpo_qlora.yaml

serve:
	bash scripts/serve_vllm.sh

api:
	uvicorn posttrainx.api.app:app --host 0.0.0.0 --port 8080

benchmark:
	python scripts/benchmark_serving.py --config configs/benchmark.yaml
