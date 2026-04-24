#!/usr/bin/env bash
set -euo pipefail

# Start the matching server separately, for example:
#   MODEL_ID=Qwen/Qwen3-Coder-30B-A3B-Instruct PORT=8000 bash start_server.sh

export CAB_HISTORY_CONTEXT_CHARS="${CAB_HISTORY_CONTEXT_CHARS:-12000}"
export CAB_HISTORY_RECENT_MESSAGES="${CAB_HISTORY_RECENT_MESSAGES:-4}"
export CAB_HISTORY_MESSAGE_CHARS="${CAB_HISTORY_MESSAGE_CHARS:-3000}"
export CAB_ORIGINAL_QUESTION_CHARS="${CAB_ORIGINAL_QUESTION_CHARS:-5000}"
export CAB_EXPLORATION_CONTEXT_CHARS="${CAB_EXPLORATION_CONTEXT_CHARS:-12000}"
export CAB_EXPLORATION_COMMAND_CHARS="${CAB_EXPLORATION_COMMAND_CHARS:-3000}"
export CAB_REFLEXION_MEMORY_CHARS="${CAB_REFLEXION_MEMORY_CHARS:-4000}"
export VLLM_MAX_OUTPUT_TOKENS="${VLLM_MAX_OUTPUT_TOKENS:-2048}"
export VLLM_CONTEXT_SAFETY_MARGIN="${VLLM_CONTEXT_SAFETY_MARGIN:-512}"

PYTHONPATH=src python -m cab_evaluation.cli reflexion-dataset \
  dataset/verified_smoke_1_python.jsonl \
  --output-dir results/reflexion_qwen3coder_smoke \
  --agent-models '{"maintainer":"qwen3coder_vllm","user":"qwen3coder_vllm"}' \
  --language python \
  --max-conversation-rounds 3 \
  --checkpoint-steps 0,1 \
  --reflexion-memory-chars "${CAB_REFLEXION_MEMORY_CHARS}"
