#!/bin/bash
# Train DGM on cloth_folding (bimanual) dataset
# Normal training with default hyperparameters, no Bayesian optimization

task_name='cloth_folding'

accelerate launch --num_processes=6 train.py \
    --config-name=dgm_cloth_folding.yaml \
    model.policy.selected_training_mode=policy_model \
    model.policy.autoregressive_model_params.use_ucgm=true \
    logging.project=cloth_folding_dgm_default \
    hydra.run.dir="checkpoints/${task_name}_dgm_default"
