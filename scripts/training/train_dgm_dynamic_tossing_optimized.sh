#!/bin/bash
# Train DGM on dynamic_tossing (bimanual) dataset
# With Bayesian optimization enabled starting at epoch 140

task_name='dynamic_tossing'

accelerate launch --num_processes=6 train.py \
    --config-name=dgm_dynamic_tossing_optimized.yaml \
    model.policy.selected_training_mode=policy_model \
    model.policy.autoregressive_model_params.use_ucgm=true \
    logging.project=dynamic_tossing_dgm_optimized \
    hydra.run.dir="checkpoints/${task_name}_dgm_optimized"
