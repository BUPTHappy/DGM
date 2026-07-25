#!/bin/bash
# Multi-seed inference for a PushT DGM checkpoint.
# Default: 50 runs varying model RNG seed (sampling stochasticity).

set -euo pipefail

CHECKPOINT="${CHECKPOINT:-checkpoints/pusht_dgm.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_out/pusht_dgm_multiseed}"
NUM_RUNS="${NUM_RUNS:-50}"
SEED_MODE="${SEED_MODE:-model}"   # model | env | both
BASE_SEED="${BASE_SEED:-0}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-2}"
DEVICE="${DEVICE:-cuda:0}"
GPU="${GPU:-0}"

CUDA_VISIBLE_DEVICES="${GPU}" python scripts/eval/multi_seed_eval.py \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_runs "${NUM_RUNS}" \
  --seed_mode "${SEED_MODE}" \
  --base_seed "${BASE_SEED}" \
  --device "${DEVICE}" \
  --num_sampling_steps "${NUM_SAMPLING_STEPS}" \
  --skip_existing \
  "$@"
