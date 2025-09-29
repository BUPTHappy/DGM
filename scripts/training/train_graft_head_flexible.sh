#!/bin/bash

# Flexible multi-GPU training script with configurable GPU usage
# 灵活的多GPU训练脚本，可配置GPU使用

set -e

# Default configuration - modify these based on your setup
CHECKPOINT_PATH="checkpoints/pusht.ckpt"
DATASET_PATH="data/pusht/"
OUTPUT_DIR="outputs/graft_head_training"
NUM_SAMPLES=8000
BATCH_SIZE=32

# GPU configuration - modify these based on your available GPUs
NUM_GPUS=4
GPU_IDS="0,1,2,3"  # Adjust based on your GPU setup
CACHE_GPUS=4       # GPUs for caching (can use all)
DISTILL_GPUS=3     # GPUs for distillation (one per operator)
STAGE2_GPUS=4      # GPUs for Stage 2 fine-tuning

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gpus)
            GPU_IDS="$2"
            NUM_GPUS=$(echo "$2" | tr ',' '\n' | wc -l)
            shift 2
            ;;
        --cache-gpus)
            CACHE_GPUS="$2"
            shift 2
            ;;
        --distill-gpus)
            DISTILL_GPUS="$2"
            shift 2
            ;;
        --stage2-gpus)
            STAGE2_GPUS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --checkpoint PATH     Path to pretrained model checkpoint"
            echo "  --dataset PATH        Path to dataset"
            echo "  --output PATH         Output directory"
            echo "  --num-samples N       Number of samples for Stage 1"
            echo "  --batch-size N        Batch size"
            echo "  --gpus IDS            GPU IDs to use (e.g., '0,1,2,3')"
            echo "  --cache-gpus N        Number of GPUs for caching"
            echo "  --distill-gpus N      Number of GPUs for distillation"
            echo "  --stage2-gpus N       Number of GPUs for Stage 2"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Flexible Multi-GPU Training Configuration"
echo "=========================================="
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Total GPUs: ${NUM_GPUS} (${GPU_IDS})"
echo "Cache GPUs: ${CACHE_GPUS}"
echo "Distill GPUs: ${DISTILL_GPUS}"
echo "Stage 2 GPUs: ${STAGE2_GPUS}"
echo "=========================================="

# Create output directories
mkdir -p ${OUTPUT_DIR}/stage1/cached_activations
mkdir -p ${OUTPUT_DIR}/stage1/init_weights
mkdir -p ${OUTPUT_DIR}/stage2/checkpoints

echo "Stage 1: Caching teacher activations..."
echo "----------------------------------------"

# Stage 1: Cache teacher activations
CUDA_VISIBLE_DEVICES=${GPU_IDS} python unified_video_action/tools/cache_activation.py \
    --checkpoint ${CHECKPOINT_PATH} \
    --dataset ${DATASET_PATH} \
    --out ${OUTPUT_DIR}/stage1/cached_activations \
    --num_samples ${NUM_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --num_gpus ${CACHE_GPUS} \
    --gpu_ids ${GPU_IDS}

echo ""
echo "Stage 1: Distilling individual operators..."
echo "------------------------------------------"

# Parse GPU IDs for distillation
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"

# Stage 1: Distill operators (can be sequential or parallel)
echo "Distilling cross-attention operator..."
CUDA_VISIBLE_DEVICES=${GPU_ARRAY[0]} python unified_video_action/training/distill_operator.py \
    --cached_dir ${OUTPUT_DIR}/stage1/cached_activations \
    --target_layer cross_attn \
    --loss l1 \
    --out ${OUTPUT_DIR}/stage1/init_weights/cross_attn.pt \
    --epochs 200 \
    --lr 1e-4 \
    --batch_size 64 \
    --num_gpus 1 \
    --gpu_ids ${GPU_ARRAY[0]}

echo "Distilling self-attention operator..."
CUDA_VISIBLE_DEVICES=${GPU_ARRAY[1]} python unified_video_action/training/distill_operator.py \
    --cached_dir ${OUTPUT_DIR}/stage1/cached_activations \
    --target_layer self_attn \
    --loss l1 \
    --out ${OUTPUT_DIR}/stage1/init_weights/self_attn.pt \
    --epochs 200 \
    --lr 1e-4 \
    --batch_size 64 \
    --num_gpus 1 \
    --gpu_ids ${GPU_ARRAY[1]}

echo "Distilling MLP operator..."
CUDA_VISIBLE_DEVICES=${GPU_ARRAY[2]} python unified_video_action/training/distill_operator.py \
    --cached_dir ${OUTPUT_DIR}/stage1/cached_activations \
    --target_layer mlp \
    --loss l2 \
    --out ${OUTPUT_DIR}/stage1/init_weights/mlp.pt \
    --epochs 200 \
    --lr 1e-4 \
    --batch_size 64 \
    --num_gpus 1 \
    --gpu_ids ${GPU_ARRAY[2]}

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
echo "Stage 2: End-to-end fine-tuning..."
echo "--------------------------------"

# Stage 2: End-to-end fine-tuning
accelerate launch --num_processes=${STAGE2_GPUS} train.py \
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
echo "Training completed!"
echo "Checkpoints saved to: ${OUTPUT_DIR}/stage2/checkpoints"
echo "=========================================="

