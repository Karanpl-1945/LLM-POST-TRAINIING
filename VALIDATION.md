# Validation status

Checks reported when this repository was generated (not rerun after cleanup):

- `pyproject.toml` parsed successfully.
- all YAML files in `configs/` parsed successfully.
- Python source and scripts passed `compileall` syntax validation.
- `scripts/serve_vllm.sh` passed `bash -n` syntax validation.
- lightweight unit tests: **4 passed**.

Not executed (requires user environment/resources):

- Hugging Face model or dataset downloads
- PySpark preprocessing job
- LoRA / QLoRA SFT
- LoRA / QLoRA DPO
- adapter merging on a real checkpoint
- vLLM GPU server
- FP8 KV-cache runtime compatibility
- Docker/Kubernetes deployment
- serving benchmarks

Run these on the CUDA/Spark environment you intend to use and pin compatible package/image versions after validation.
