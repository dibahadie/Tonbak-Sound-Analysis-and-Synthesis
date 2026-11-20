#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${1:?Usage: ./synthesis/modal_resonator/run_pipeline.sh /path/to/extracted_one_shots [run_name]}"
RUN_NAME="${2:-tonbak_direct_32x2}"
WORK_DIR="."

# Keep imports scoped to synthesis/modal_resonator so neuralresonator/ is found cleanly
export PYTHONPATH="${WORK_DIR}"

python "${WORK_DIR}/1_prepare_data.py" \
  --source_dir "${SOURCE_DIR}" \
  --work_dir "${WORK_DIR}" \
  --sample_rate 32000 \
  --audio_length 2.0 \
  --seed 42

python "${WORK_DIR}/2_train.py" \
  --work_dir "${WORK_DIR}" \
  --run_name "${RUN_NAME}" \
  --n_parallel 32 \
  --n_biquads 2 \
  --sample_rate 32000 \
  --audio_length 2.0 \
  --batch_size 8 \
  --lr 1e-4 \
  --seed 42

# Render two quick test strokes (tom-like and bak-like) from the best checkpoint
CKPT="${WORK_DIR}/runs/${RUN_NAME}/best.pt"
OUT="${WORK_DIR}/generated_samples"
mkdir -p "$OUT"

python ${WORK_DIR}/3_infer.py --ckpt "$CKPT" --controls 0.10 0.10 0.05 0.95 1.0 --out_wav "$OUT/01_open_tom.wav"
python ${WORK_DIR}/3_infer.py --ckpt "$CKPT" --controls 0.15 0.15 0.15 0.25 1.0 --out_wav "$OUT/02_muted_tom.wav"
python ${WORK_DIR}/3_infer.py --ckpt "$CKPT" --controls 0.75 0.80 0.45 0.60 1.0 --out_wav "$OUT/03_open_bak.wav"
python ${WORK_DIR}/3_infer.py --ckpt "$CKPT" --controls 0.90 0.95 0.80 0.10 1.0 --out_wav "$OUT/04_pelang_snap.wav"
python ${WORK_DIR}/3_infer.py --ckpt "$CKPT" --controls 0.60 0.65 0.50 0.30 1.0 --out_wav "$OUT/05_riz_soft.wav"
python ${WORK_DIR}/3_infer.py --ckpt "$CKPT" --controls 0.45 0.55 0.70 0.20 1.0 --out_wav "$OUT/06_mid_slap.wav"