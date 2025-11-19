#!/bin/bash

model_dir='checkpoints'

CUDA_VISIBLE_DEVICES=0 python eval_sim.py \
    --checkpoint ${model_dir}/pusht.ckpt \
    --output_dir ${model_dir}/pusht


