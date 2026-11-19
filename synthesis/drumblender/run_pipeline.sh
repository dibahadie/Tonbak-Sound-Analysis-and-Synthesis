#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${1:?Usage: ./run_pipeline.sh ../../data_processing/extracted_one_shots [run_name]}"
RUN_NAME="${2:-tonbak_ts_n}"
WORK_DIR="synthesis/drumblender"

python "${WORK_DIR}/1_prepare_data.py" \
  --source_dir "${SOURCE_DIR}" \
  --work_dir "${WORK_DIR}" \
  --sr 48000 \
  --duration 2.0 \
  --seed 42

python "${WORK_DIR}/2_extract_sines.py" \
  --work_dir "${WORK_DIR}" \
  --n_sines 64 \
  --hop_length 256

python "${WORK_DIR}/3_train.py" \
  --work_dir "${WORK_DIR}" \
  --run_name "${RUN_NAME}" \
  --mixing "T(S)+N" \
  --batch_size 12 \
  --lr 1e-4 \
  --seed 42