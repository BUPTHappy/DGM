#!/bin/bash
# Train DGM on dynamic_tossing (bimanual) dataset
# Normal training with default hyperparameters, no Bayesian optimization

task_name='dynamic_tossing'

accelerate launch --num_processes=6 train.py \
    --config-name=dgm_dynamic_tossing.yaml \
    model.policy.selected_training_mode=policy_model \
    model.policy.autoregressive_model_params.use_ucgm=true \
    logging.project=dynamic_tossing_dgm_default \
    hydra.run.dir="checkpoints/${task_name}_dgm_default"
