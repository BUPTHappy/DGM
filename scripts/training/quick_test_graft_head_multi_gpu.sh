#!/bin/bash

# Quick multi-GPU test script for Stage 2 only
# 快速多GPU测试脚本：仅运行Stage 2

set -e

# Configuration - adjust these paths
INIT_WEIGHTS_DIR="outputs/graft_head_training/stage1/init_weights"
OUTPUT_DIR="outputs/graft_head_test_multi_gpu"
CHECKPOINT_PATH="checkpoints/pusht.ckpt"
NUM_GPUS=4
GPU_IDS="0,1,2,3"

echo "=========================================="
echo "Quick Multi-GPU Test: Stage 2 Fine-tuning"
echo "=========================================="
echo "Using ${NUM_GPUS} GPUs: ${GPU_IDS}"

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "Running Stage 2 with distilled weights..."
echo "----------------------------------------"

# Stage 2: End-to-end fine-tuning with multi-GPU
accelerate launch --num_processes=${NUM_GPUS} train.py \
    --config-name=train_diffusion_graft_workspace \
    model.policy.action_model_params.head_init_paths.cross_attn=${INIT_WEIGHTS_DIR}/cross_attn.pt \
    model.policy.action_model_params.head_init_paths.self_attn=${INIT_WEIGHTS_DIR}/self_attn.pt \
    model.policy.action_model_params.head_init_paths.mlp=${INIT_WEIGHTS_DIR}/mlp.pt \
    model.policy.finetune.freeze_encoder=true \
    training.steps=10000 \
    training.data_fraction=0.05 \
    training.num_gpus=${NUM_GPUS} \
    training.batch_size_per_gpu=16 \
    model.policy.optimizer.learning_rate=1e-4 \
    hydra.run.dir=${OUTPUT_DIR}

echo ""
echo "Quick multi-GPU test completed!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo "=========================================="

