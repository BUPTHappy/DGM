#!/bin/bash
# Train DGM on dish_washing (bimanual) dataset
# With Bayesian optimization enabled starting at epoch 140

task_name='dish_washing'

# Train video and action model with Bayesian optimization
accelerate launch --num_processes=8 train.py \
    --config-dir=unified_video_action/config \
    --config-name=dgm_dish_washing_optimized.yaml \
    logging.project=dgm_${task_name}_optimized \
    hydra.run.dir="checkpoints/${task_name}_dgm_optimized"
