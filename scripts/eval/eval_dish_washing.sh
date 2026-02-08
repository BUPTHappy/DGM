#!/bin/bash
# Evaluate dish_washing checkpoint (real-world dataset, no simulator)

# Default training checkpoint
CUDA_VISIBLE_DEVICES=0 python eval_offline.py \
    --checkpoint checkpoints/dish_washing_dgm_default/checkpoints/latest.ckpt \
    --output_dir eval_results/dish_washing_default \
    --device cuda:0 \
    --use_ucgm

# Uncomment to also compute FVD (slower):
# CUDA_VISIBLE_DEVICES=0 python eval_offline.py \
#     --checkpoint checkpoints/dish_washing_dgm_default/checkpoints/latest.ckpt \
#     --output_dir eval_results/dish_washing_default \
#     --device cuda:0 \
#     --use_ucgm \
#     --fvd
