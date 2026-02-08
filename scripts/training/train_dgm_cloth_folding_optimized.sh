#!/bin/bash
# Train DGM on cloth_folding (bimanual) dataset
# With Bayesian optimization enabled starting at epoch 140

task_name='cloth_folding'

accelerate launch --num_processes=6 train.py \
    --config-name=dgm_cloth_folding_optimized.yaml \
    model.policy.selected_training_mode=policy_model \
    model.policy.autoregressive_model_params.use_ucgm=true \
    logging.project=cloth_folding_dgm_optimized \
    hydra.run.dir="checkpoints/${task_name}_dgm_optimized"
