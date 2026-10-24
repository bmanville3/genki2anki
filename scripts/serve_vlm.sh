#!/usr/bin/env bash
# Serve a local vision-language model with vLLM (OpenAI-compatible API on :8000).
#
#   scripts/serve_vlm.sh                       # Qwen3-VL-32B, FP8 weights, ~70% of the GPU
#   MODEL=<any <=35B vision model> scripts/serve_vlm.sh
#   QUANT=none GPU_UTIL=0.88 scripts/serve_vlm.sh   # full bf16 (stop it before `transcribe`)
#
# The served model name must match `--vlm-model` / vlm.model in genki2anki.
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
QUANT="${QUANT:-fp8}"          # fp8 leaves room for Whisper on an 80GB A100; "none" = bf16
UTIL="${GPU_UTIL:-0.70}"
PORT="${PORT:-8000}"
MAXLEN="${MAX_LEN:-12288}"     # one page image (~4k tokens) + ~6k tokens of JSON output

args=(serve "$MODEL" --port "$PORT" --max-model-len "$MAXLEN"
      --gpu-memory-utilization "$UTIL" --max-num-seqs 16 --served-model-name "$MODEL")
if [[ "$QUANT" != "none" ]]; then args+=(--quantization "$QUANT"); fi
exec vllm "${args[@]}"
