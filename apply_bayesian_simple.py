#!/usr/bin/env python3
"""
Simple script to apply Bayesian optimization parameters to checkpoint.
"""

import os
import sys
import torch
import dill
import json
from omegaconf import OmegaConf, open_dict

def main():
    # Bayesian optimization results from your training
    bayesian_params = {
        'consistc_ratio': 0.9870031611149999,
        'rfba_gap_end': 0.22533496150457658,
        'temperature': 0.8836599795042932,
        'num_sampling_steps': 2,
        'cfg': 1.0260267723044316,
        'extrapol_ratio': 0.5318789824701121,
        'window_size': 8,
        'lambda_local': 0.3566535175072145,
        'ucgmts_config': {
            'transport_type': 'Linear',
            'consistc_ratio': 0.9870031611149999,
            'scaled_cbl_eps': 0.0,
            'ema_decay_rate': 0.0,
            'rfba_gap_steps': [0.001, 0.22533496150457658],
            'extrapol_ratio': 0.5318789824701121
        }
    }
    
    input_checkpoint = "checkpoints/pusht_production_opt/checkpoints/epoch=0050-test_mean_score=0.991.ckpt"
    output_checkpoint = "checkpoints/pusht_production_opt/checkpoints/epoch=0050-test_mean_score=0.991_optimized.ckpt"
    
    print(f"Loading checkpoint: {input_checkpoint}")
    
    # Load original checkpoint
    payload = torch.load(open(input_checkpoint, "rb"), pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    
    print("Original parameters:")
    print(f"  num_sampling_steps: {cfg.model.policy.autoregressive_model_params.num_sampling_steps}")
    print(f"  cfg: {cfg.model.policy.autoregressive_model_params.cfg}")
    print(f"  temperature: {cfg.model.policy.autoregressive_model_params.temperature}")
    print(f"  window_size: {getattr(cfg.model.policy.autoregressive_model_params, 'window_size', 'N/A')}")
    print(f"  lambda_local: {getattr(cfg.model.policy.autoregressive_model_params, 'lambda_local', 'N/A')}")
    
    # Update config with Bayesian optimization parameters
    with open_dict(cfg.model.policy.autoregressive_model_params):
        # Ensure ucgmts_config exists
        if "ucgmts_config" not in cfg.model.policy.autoregressive_model_params:
            cfg.model.policy.autoregressive_model_params.ucgmts_config = OmegaConf.create({})
        
        # Apply optimized parameters
        cfg.model.policy.autoregressive_model_params.use_ucgm = True
        cfg.model.policy.autoregressive_model_params.num_sampling_steps = bayesian_params['num_sampling_steps']
        cfg.model.policy.autoregressive_model_params.cfg = bayesian_params['cfg']
        cfg.model.policy.autoregressive_model_params.temperature = bayesian_params['temperature']
        cfg.model.policy.autoregressive_model_params.window_size = bayesian_params['window_size']
        cfg.model.policy.autoregressive_model_params.lambda_local = bayesian_params['lambda_local']
        
        # Update UCGMTS config
        cfg.model.policy.autoregressive_model_params.ucgmts_config.transport_type = bayesian_params['ucgmts_config']['transport_type']
        cfg.model.policy.autoregressive_model_params.ucgmts_config.consistc_ratio = bayesian_params['ucgmts_config']['consistc_ratio']
        cfg.model.policy.autoregressive_model_params.ucgmts_config.scaled_cbl_eps = bayesian_params['ucgmts_config']['scaled_cbl_eps']
        cfg.model.policy.autoregressive_model_params.ucgmts_config.ema_decay_rate = bayesian_params['ucgmts_config']['ema_decay_rate']
        cfg.model.policy.autoregressive_model_params.ucgmts_config.rfba_gap_steps = bayesian_params['ucgmts_config']['rfba_gap_steps']
        cfg.model.policy.autoregressive_model_params.ucgmts_config.extrapol_ratio = bayesian_params['ucgmts_config']['extrapol_ratio']
    
    print("\nUpdated parameters:")
    print(f"  num_sampling_steps: {cfg.model.policy.autoregressive_model_params.num_sampling_steps}")
    print(f"  cfg: {cfg.model.policy.autoregressive_model_params.cfg}")
    print(f"  temperature: {cfg.model.policy.autoregressive_model_params.temperature}")
    print(f"  window_size: {cfg.model.policy.autoregressive_model_params.window_size}")
    print(f"  lambda_local: {cfg.model.policy.autoregressive_model_params.lambda_local}")
    print(f"  ucgmts_config:")
    print(f"    transport_type: {cfg.model.policy.autoregressive_model_params.ucgmts_config.transport_type}")
    print(f"    consistc_ratio: {cfg.model.policy.autoregressive_model_params.ucgmts_config.consistc_ratio}")
    print(f"    scaled_cbl_eps: {cfg.model.policy.autoregressive_model_params.ucgmts_config.scaled_cbl_eps}")
    print(f"    ema_decay_rate: {cfg.model.policy.autoregressive_model_params.ucgmts_config.ema_decay_rate}")
    print(f"    rfba_gap_steps: {cfg.model.policy.autoregressive_model_params.ucgmts_config.rfba_gap_steps}")
    print(f"    extrapol_ratio: {cfg.model.policy.autoregressive_model_params.ucgmts_config.extrapol_ratio}")
    
    # Update payload with new config
    payload["cfg"] = cfg
    
    # Save new checkpoint
    print(f"\nSaving optimized checkpoint to: {output_checkpoint}")
    os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)
    torch.save(payload, open(output_checkpoint, "wb"), pickle_module=dill)
    
    print("✓ Successfully created checkpoint with Bayesian optimization parameters!")
    
    print(f"\n{'='*60}")
    print("SUCCESS!")
    print(f"{'='*60}")
    print(f"Original checkpoint: {input_checkpoint}")
    print(f"Optimized checkpoint: {output_checkpoint}")
    print(f"Expected performance: 0.9919")
    print(f"\nYou can now evaluate the optimized checkpoint:")
    print(f"CUDA_VISIBLE_DEVICES=0 python eval_sim.py \\")
    print(f"    --checkpoint {output_checkpoint} \\")
    print(f"    --output_dir checkpoints/pusht_optimized_eval/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
