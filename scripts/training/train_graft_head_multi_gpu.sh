#!/bin/bash

# Multi-GPU two-stage training script for diffusion action head grafting
# 多GPU两阶段训练脚本：将MLP头替换为cross-attention + self-attention + MLP结构

set -e

# Change to project root directory
cd "$(dirname "$0")/../.."

# Configuration
CHECKPOINT_PATH="checkpoints/pusht.ckpt"  # Path to your pretrained model
DATASET_PATH="data/pusht/"  # Path to your dataset
OUTPUT_DIR="outputs/graft_head_training_multi_gpu"
NUM_SAMPLES=8000  # Number of samples for Stage 1
BATCH_SIZE=32
NUM_GPUS=4  # Number of GPUs to use
GPU_IDS="0,1,2,3"  # Specific GPU IDs (adjust based on your setup)

echo "=========================================="
echo "Multi-GPU Two-Stage Diffusion Head Grafting Training"
echo "=========================================="
echo "Using ${NUM_GPUS} GPUs: ${GPU_IDS}"
echo "Working directory: $(pwd)"

# Create output directories
mkdir -p ${OUTPUT_DIR}/stage1/cached_activations
mkdir -p ${OUTPUT_DIR}/stage1/init_weights
mkdir -p ${OUTPUT_DIR}/stage2/checkpoints

echo "Stage 1: Caching teacher activations..."
echo "----------------------------------------"

# Stage 1: Cache teacher activations with multi-GPU
CUDA_VISIBLE_DEVICES=${GPU_IDS} python unified_video_action/tools/cache_activation.py \
    --checkpoint ${CHECKPOINT_PATH} \
    --dataset ${DATASET_PATH} \
    --out ${OUTPUT_DIR}/stage1/cached_activations \
    --num_samples ${NUM_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --num_gpus ${NUM_GPUS} \
    --gpu_ids ${GPU_IDS}

echo ""
echo "Stage 1: Distilling individual operators in parallel..."
echo "-----------------------------------------------------"

# Stage 1: Distill operators in parallel using different GPUs
echo "Starting parallel distillation..."

# Distill cross-attention operator on GPU 0
CUDA_VISIBLE_DEVICES=0 python unified_video_action/training/distill_operator.py \
    --cached_dir ${OUTPUT_DIR}/stage1/cached_activations \
    --target_layer cross_attn \
    --loss l1 \
    --out ${OUTPUT_DIR}/stage1/init_weights/cross_attn.pt \
    --epochs 200 \
    --lr 1e-4 \
    --batch_size 64 \
    --num_gpus 1 \
    --gpu_ids 0 &

# Distill self-attention operator on GPU 1
CUDA_VISIBLE_DEVICES=1 python unified_video_action/training/distill_operator.py \
    --cached_dir ${OUTPUT_DIR}/stage1/cached_activations \
    --target_layer self_attn \
    --loss l1 \
    --out ${OUTPUT_DIR}/stage1/init_weights/self_attn.pt \
    --epochs 200 \
    --lr 1e-4 \
    --batch_size 64 \
    --num_gpus 1 \
    --gpu_ids 1 &

# Distill MLP operator on GPU 2
CUDA_VISIBLE_DEVICES=2 python unified_video_action/training/distill_operator.py \
    --cached_dir ${OUTPUT_DIR}/stage1/cached_activations \
    --target_layer mlp \
    --loss l2 \
    --out ${OUTPUT_DIR}/stage1/init_weights/mlp.pt \
    --epochs 200 \
    --lr 1e-4 \
    --batch_size 64 \
    --num_gpus 1 \
    --gpu_ids 2 &

# Wait for all parallel distillation jobs to complete
echo "Waiting for parallel distillation to complete..."
wait

echo "All distillation jobs completed!"

# Verify all weights were created
echo "Verifying distilled weights..."
for weight_file in cross_attn.pt self_attn.pt mlp.pt; do
    if [ -f "${OUTPUT_DIR}/stage1/init_weights/${weight_file}" ]; then
        echo "✓ ${weight_file} created successfully"
    else
        echo "✗ ${weight_file} missing!"
        exit 1
    fi
done

echo ""
echo "Stage 2: End-to-end fine-tuning with multi-GPU..."
echo "------------------------------------------------"

# Stage 2: End-to-end fine-tuning with multi-GPU
accelerate launch --num_processes=${NUM_GPUS} train.py \
    --config-name=train_diffusion_graft_workspace \
    model.policy.action_model_params.head_init_paths.cross_attn=${OUTPUT_DIR}/stage1/init_weights/cross_attn.pt \
    model.policy.action_model_params.head_init_paths.self_attn=${OUTPUT_DIR}/stage1/init_weights/self_attn.pt \
    model.policy.action_model_params.head_init_paths.mlp=${OUTPUT_DIR}/stage1/init_weights/mlp.pt \
    model.policy.finetune.freeze_encoder=true \
    training.steps=50000 \
    training.data_fraction=0.10 \
    model.policy.optimizer.learning_rate=1e-4 \
    hydra.run.dir=${OUTPUT_DIR}/stage2/checkpoints

echo ""
echo "Multi-GPU training completed!"
echo "Checkpoints saved to: ${OUTPUT_DIR}/stage2/checkpoints"
echo "=========================================="
