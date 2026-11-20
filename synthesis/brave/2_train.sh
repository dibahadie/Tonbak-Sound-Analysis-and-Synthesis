#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-tonbak_brave}"
DB_PATH="./data"
OUT_PATH="./runs"
export PYTHONPATH="/data/anasynth_nonbp/hadi/Tonbak/Tonbak-Sound-Analysis-and-Synthesis/synthesis/brave"

rave train \
  --config synthesis/brave/configs/brave.gin \
  --db_path "${DB_PATH}" \
  --out_path "${OUT_PATH}" \
  --name "${RUN_NAME}" \
  --val_every 10000 \
  --max_steps 1500000 \
  --gpu 0