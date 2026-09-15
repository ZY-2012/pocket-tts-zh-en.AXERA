#!/usr/bin/env bash
# One-command board run for pocket-tts-zh-en (AX650N).
#
# Prereq (board local disk, copied once):
#   /root/pocket_tts_zh_en/models/{flow_step_windowed.onnx,step_encoder.onnx,step_config.json,
#       flow_step_weighted? , flow_ar_step(.int8).onnx (flow/), flow_net_step(.fp32).axmodel,
#       mimi_split/{mimi_transformer_step.onnx,mimi_conv_step.onnx}, mimi_conv_step.axmodel,
#       chn_jpn_yue_eng_ko_spectok.bpe.model, Vivian.wav}
#
# Usage (board):
#   bash scripts/board_run.sh "你好，世界。" out.wav
set -euo pipefail

if [ -f /root/miniforge3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /root/miniforge3/etc/profile.d/conda.sh
  conda activate base
fi

TEXT="${1:-今天天气不错，我们一起去公园散步吧。}"
OUTPUT="${2:-/root/pocket_tts_zh_en/output/board.wav}"
MODELS="${MODELS:-/root/pocket_tts_zh_en/models}"
BOARD_SCRIPT="${BOARD_SCRIPT:-/root/huyuan/workspace/TTS_quant/pocket-tts-zh-en-axera/board/pocket_tts_axera.py}"
REF="${REF:-$MODELS/Vivian.wav}"
THREADS="${THREADS:-4}"
PREFILL_THREADS="${PREFILL_THREADS:-8}"
FLOW_AR_MODEL="${FLOW_AR_MODEL:-flow/flow_ar_step_int8.onnx}"   # or flow/flow_ar_step.onnx
FLOW_NET_MODEL="${FLOW_NET_MODEL:-flow_net_step_fp32.axmodel}" # or flow_net_step.axmodel (U16)

python3 -B "$BOARD_SCRIPT" \
  --text "$TEXT" \
  --reference "$REF" \
  --output "$OUTPUT" \
  --onnx-dir "$MODELS" \
  --axmodel-dir "$MODELS" \
  --cpu-model-dir "$MODELS" \
  --mimi-split-dir "$MODELS/mimi_split" \
  --spm "$MODELS/chn_jpn_yue_eng_ko_spectok.bpe.model" \
  --threads "$THREADS" --prefill-threads "$PREFILL_THREADS" \
  --npu-flow-net 1 --npu-mimi-conv 1 \
  --flow-ar-model "$FLOW_AR_MODEL" --flow-net-model "$FLOW_NET_MODEL" \
  --chunk 1
