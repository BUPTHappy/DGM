#!/bin/bash

# Example script demonstrating memory-optimized evaluation
# This script shows how to use the enhanced eval_sim_memory_optim.py with various memory optimization options

echo "Running memory-optimized evaluation with various configurations..."

# Basic memory optimization (recommended for most cases)
echo "=== Basic Memory Optimization ==="
CUDA_VISIBLE_DEVICES=0 python eval_sim_memory_optim.py \
  --checkpoint checkpoints/libero10_MLP.ckpt \
  --output_dir checkpoints/libero10_MLP_rollout_optimized/ \
  --enable_grad_checkpointing \
  --reduce_batch_size \
  --enable_memory_monitoring \
  --aggressive_memory_cleanup \
  --optimize_data_loading \
  --vae_chunk_size 2 \
  --vae_batch_chunk_size 4

# Maximum memory optimization (for very limited GPU memory)
echo "=== Maximum Memory Optimization ==="
CUDA_VISIBLE_DEVICES=0 python eval_sim_memory_optim.py \
  --checkpoint checkpoints/libero10_MLP.ckpt \
  --output_dir checkpoints/libero10_MLP_rollout_max_optimized/ \
  --enable_grad_checkpointing \
  --reduce_batch_size \
  --enable_memory_monitoring \
  --aggressive_memory_cleanup \
  --optimize_data_loading \
  --pin_memory \
  --enable_model_sharding \
  --shard_size_mb 500 \
  --vae_chunk_size 1 \
  --vae_batch_chunk_size 2

# Performance-focused optimization (balanced memory and speed)
echo "=== Performance-Focused Optimization ==="
CUDA_VISIBLE_DEVICES=0 python eval_sim_memory_optim.py \
  --checkpoint checkpoints/libero10_DiT_hybrid.ckpt \
  --output_dir checkpoints/libero10_DiT_hybrid_rollout_optimized/ \
  --enable_grad_checkpointing \
  --enable_memory_monitoring \
  --optimize_data_loading \
  --pin_memory \
  --vae_chunk_size 4 \
  --vae_batch_chunk_size 8

echo "Memory optimization examples completed!"
echo "Check the output directories for evaluation results and memory usage logs."
