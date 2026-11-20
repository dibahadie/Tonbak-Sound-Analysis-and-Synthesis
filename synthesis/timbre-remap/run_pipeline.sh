#!/usr/bin/env bash
set -euo pipefail

AUDIO_PATH="${1:?Usage: ./synthesis/timbre_remap/run_pipeline.sh /path/to/tonbak_audio [run_name] [arch] [onset_window]}"
RUN_NAME="${2:-tonbak_mlp_256}"
ARCH="${3:-mlp}"
ONSET_WINDOW="${4:-256}"
WORK_DIR="."

export PYTHONPATH="${WORK_DIR}"

python "${WORK_DIR}/1_make_preset.py" \
  --out "${WORK_DIR}/presets/tonbak_808.json"

# python "${WORK_DIR}/2_train.py" \
#   --audio_path "${AUDIO_PATH}" \
#   --preset "${WORK_DIR}/presets/tonbak_808.json" \
#   --work_dir "${WORK_DIR}" \
#   --run_name "${RUN_NAME}" \
#   --arch "${ARCH}" \
#   --onset_window "${ONSET_WINDOW}" \
#   --sample_rate 44100 \
#   --batch_size 64 \
#   --lr 5e-4 \
#   --max_epochs 250 \
#   --seed 3608013887

# CKPT="${WORK_DIR}/runs/${RUN_NAME}/best.ckpt"

# python "${WORK_DIR}/3_infer.py" \
#   --ckpt "${CKPT}" \
#   --preset "${WORK_DIR}/presets/tonbak_808.json" \
#   --arch "${ARCH}" \
#   --controls 0.2 0.2 0.2 \
#   --out_wav "${WORK_DIR}/runs/${RUN_NAME}/soft_dark.wav"

# python "${WORK_DIR}/3_infer.py" \
#   --ckpt "${CKPT}" \
#   --preset "${WORK_DIR}/presets/tonbak_808.json" \
#   --arch "${ARCH}" \
#   --controls 0.8 0.8 0.7 \
#   --out_wav "${WORK_DIR}/runs/${RUN_NAME}/loud_bright.wav"