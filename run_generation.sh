#!/usr/bin/env bash
set -euo pipefail

# context management
export CAB_HISTORY_CONTEXT_CHARS="${CAB_HISTORY_CONTEXT_CHARS:-12000}"
export CAB_HISTORY_RECENT_MESSAGES="${CAB_HISTORY_RECENT_MESSAGES:-4}"
export CAB_HISTORY_MESSAGE_CHARS="${CAB_HISTORY_MESSAGE_CHARS:-3000}"
export CAB_ORIGINAL_QUESTION_CHARS="${CAB_ORIGINAL_QUESTION_CHARS:-5000}"
export CAB_EXPLORATION_CONTEXT_CHARS="${CAB_EXPLORATION_CONTEXT_CHARS:-12000}"
export CAB_EXPLORATION_COMMAND_CHARS="${CAB_EXPLORATION_COMMAND_CHARS:-3000}"
export VLLM_MAX_OUTPUT_TOKENS="${VLLM_MAX_OUTPUT_TOKENS:-4096}"
export VLLM_CONTEXT_SAFETY_MARGIN="${VLLM_CONTEXT_SAFETY_MARGIN:-512}"


python -m cab_evaluation.cli generation-dataset \
  dataset/verified_smoke_1_python.jsonl \
  --output results/generation_vllm_smokeCoder.jsonl \
  --agent-models '{"maintainer":"qwen3coder_vllm","user":"qwen3coder_vllm"}' \
  --language python \
  --max-conversation-rounds 3
