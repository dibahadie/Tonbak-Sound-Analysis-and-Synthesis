#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${1:?Usage: ./synthesis/drumblender/run_pipeline.sh /path/to/extracted_one_shots [run_name]}"
RUN_NAME="${2:-tonbak_ts_n}"
WORK_DIR="."

export PYTHONPATH="/data/anasynth_nonbp/hadi/Tonbak/Tonbak-Sound-Analysis-and-Synthesis/synthesis/drumblender"

# python "${WORK_DIR}/1_prepare_data.py" \
#   --source_dir "${SOURCE_DIR}" \
#   --work_dir "${WORK_DIR}" \
#   --sample_rate 48000 \
#   --duration 2.0

# python "${WORK_DIR}/2_extract_modal.py" \
#   --work_dir "${WORK_DIR}" \
#   --sample_rate 48000 \
#   --num_modes 64 \
#   --hop_length 256

python "${WORK_DIR}/3_train.py" \
  --work_dir "${WORK_DIR}" \
  --run_name "${RUN_NAME}" \
  --mixing "T(S)+N" \
  --batch_size 4 \
  --lr 1e-4 \
  --accumulate_grad_batches 3