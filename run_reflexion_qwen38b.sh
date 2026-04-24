#!/usr/bin/env bash
set -euo pipefail

# Start the matching server separately, for example:
#   MODEL_ID=Qwen/Qwen3-8B PORT=8001 bash start_server.sh

export CAB_HISTORY_CONTEXT_CHARS="${CAB_HISTORY_CONTEXT_CHARS:-8000}"
export CAB_HISTORY_RECENT_MESSAGES="${CAB_HISTORY_RECENT_MESSAGES:-3}"
export CAB_HISTORY_MESSAGE_CHARS="${CAB_HISTORY_MESSAGE_CHARS:-2000}"
export CAB_ORIGINAL_QUESTION_CHARS="${CAB_ORIGINAL_QUESTION_CHARS:-3500}"
export CAB_EXPLORATION_CONTEXT_CHARS="${CAB_EXPLORATION_CONTEXT_CHARS:-3000}"
export CAB_EXPLORATION_COMMAND_CHARS="${CAB_EXPLORATION_COMMAND_CHARS:-1200}"
export CAB_REFLEXION_MEMORY_CHARS="${CAB_REFLEXION_MEMORY_CHARS:-2500}"
export VLLM_MAX_OUTPUT_TOKENS="${VLLM_MAX_OUTPUT_TOKENS:-1536}"
export VLLM_CONTEXT_SAFETY_MARGIN="${VLLM_CONTEXT_SAFETY_MARGIN:-512}"

PYTHONPATH=src python -m cab_evaluation.cli reflexion-dataset \
  dataset/verified_smoke_1_python.jsonl \
  --output-dir results/reflexion_qwen3_8b_smoke \
  --agent-models '{"maintainer":"qwen3_8b_vllm","user":"qwen3_8b_vllm"}' \
  --language python \
  --max-conversation-rounds 3 \
  --checkpoint-steps 0,1 \
  --reflexion-memory-chars "${CAB_REFLEXION_MEMORY_CHARS}"
