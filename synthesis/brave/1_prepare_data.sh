#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${1:?Usage: ./synthesis/brave/1_prepare_data.sh /path/to/tonbak_wavs}"
OUT_DIR="./data"

mkdir -p "${OUT_DIR}"

python ./concat_hits.py --input_dir "${INPUT_DIR}"

rave preprocess \
  --input_path "./concat_wavs" \
  --output_path "${OUT_DIR}" \
  --sampling_rate 44100 \
  --num_signal 65536