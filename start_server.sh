#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-1}"

if ss -ltn | awk '{print $4}' | grep -q ":${PORT}$"; then
    echo "[FAIL] Port ${PORT} is already in use."
    echo "[HINT] Use a different port, e.g.: PORT=8002 bash start_server.sh"
    exit 1
fi

echo "[INFO] Starting vLLM on ${HOST}:${PORT} with model ${MODEL_ID}"
vllm serve "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype bfloat16 \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization 0.98 \
    --max-model-len 12000 \
    --max-num-seqs 1 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
