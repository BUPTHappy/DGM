#!/bin/bash
# Train DGM on dish_washing (bimanual) dataset
# Normal training with default hyperparameters, no Bayesian optimization

task_name='dish_washing'

# Train video and action model
accelerate launch --num_processes=8 train.py \
    --config-dir=unified_video_action/config \
    --config-name=dgm_dish_washing.yaml \
    logging.project=dgm_${task_name} \
    hydra.run.dir="checkpoints/${task_name}_dgm"
