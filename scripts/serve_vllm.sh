#!/usr/bin/env bash
set -euo pipefail

MODEL="${FINAL_MODEL_PATH:-outputs/final-merged}"
SERVED_NAME="${SERVED_MODEL_NAME:-posttrainx}"
API_KEY="${VLLM_API_KEY:-local-dev-key}"
KV_DTYPE="${KV_CACHE_DTYPE:-fp8}"
TENSOR_PARALLEL="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_LEN="${MAX_MODEL_LEN:-4096}"

exec vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "$API_KEY" \
  --tensor-parallel-size "$TENSOR_PARALLEL" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_LEN" \
  --enable-prefix-caching \
  --kv-cache-dtype "$KV_DTYPE"
