#!/usr/bin/env bash
set -euo pipefail

AUDIO_PATH="${1:?Usage: ./synthesis/timbre_remap/run_pipeline.sh /path/to/tonbak_audio [run_name] [arch] [onset_window]}"
RUN_NAME="${2:-tonbak_mlp_256}"
ARCH="${3:-mlp}"
ONSET_WINDOW="${4:-256}"
WORK_DIR="."

export PYTHONPATH="${WORK_DIR}"

CKPT="${WORK_DIR}/runs/${RUN_NAME}/best.ckpt"

python "${WORK_DIR}/3_infer.py" \
  --ckpt "${CKPT}" \
  --preset "${WORK_DIR}/presets/tonbak_808.json" \
  --arch "${ARCH}" \
  --controls 0.1 0.1 0.1 \
  --out_wav "${WORK_DIR}/generated_samples/${RUN_NAME}/soft_dark.wav"

python "${WORK_DIR}/3_infer.py" \
  --ckpt "${CKPT}" \
  --preset "${WORK_DIR}/presets/tonbak_808.json" \
  --arch "${ARCH}" \
  --controls 0.9 0.9 0.9 \
  --out_wav "${WORK_DIR}/generated_samples/${RUN_NAME}/loud_bright.wav"