#!/usr/bin/env python3
"""
Create a truly optimized checkpoint with ALL Bayesian optimization parameters.
This script ensures the checkpoint contains the exact same parameters as manual evaluation.
"""

import os
import torch
import dill
from omegaconf import OmegaConf, open_dict

def create_perfect_optimized_checkpoint():
    """
    Create a checkpoint that contains ALL the optimized parameters.
    """
    
    # Bayesian optimization results (exact values from your training)
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
    output_checkpoint = "checkpoints/pusht_production_opt/checkpoints/epoch=0050-test_mean_score=0.991_PERFECT_OPTIMIZED.ckpt"
    
    print(f"{'='*60}")
    print("CREATING PERFECT OPTIMIZED CHECKPOINT")
    print(f"{'='*60}")
    print(f"Input: {input_checkpoint}")
    print(f"Output: {output_checkpoint}")
    
    # Load original checkpoint
    print(f"\nLoading checkpoint...")
    payload = torch.load(open(input_checkpoint, "rb"), pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    
    print("Original parameters:")
    params = cfg.model.policy.autoregressive_model_params
    print(f"  num_sampling_steps: {params.num_sampling_steps}")
    print(f"  cfg: {params.cfg}")
    print(f"  temperature: {params.temperature}")
    print(f"  window_size: {getattr(params, 'window_size', 'N/A')}")
    print(f"  lambda_local: {getattr(params, 'lambda_local', 'N/A')}")
    print(f"  consistc_ratio: {params.ucgmts_config.consistc_ratio}")
    print(f"  rfba_gap_steps: {params.ucgmts_config.rfba_gap_steps}")
    print(f"  extrapol_ratio: {getattr(params.ucgmts_config, 'extrapol_ratio', 'N/A')}")
    
    # Apply ALL optimized parameters
    print(f"\nApplying optimized parameters...")
    with open_dict(cfg.model.policy.autoregressive_model_params):
        # Ensure ucgmts_config exists
        if "ucgmts_config" not in cfg.model.policy.autoregressive_model_params:
            cfg.model.policy.autoregressive_model_params.ucgmts_config = OmegaConf.create({})
        
        # Apply ALL optimized parameters
        cfg.model.policy.autoregressive_model_params.use_ucgm = True
        cfg.model.policy.autoregressive_model_params.num_sampling_steps = bayesian_params['num_sampling_steps']
        cfg.model.policy.autoregressive_model_params.cfg = bayesian_params['cfg']
        cfg.model.policy.autoregressive_model_params.temperature = bayesian_params['temperature']
        cfg.model.policy.autoregressive_model_params.window_size = bayesian_params['window_size']
        cfg.model.policy.autoregressive_model_params.lambda_local = bayesian_params['lambda_local']
        
        # Update UCGMTS config with ALL parameters
        cfg.model.policy.autoregressive_model_params.ucgmts_config.transport_type = bayesian_params['ucgmts_config']['transport_type']
        cfg.model.policy.autoregressive_model_params.ucgmts_config.consistc_ratio = bayesian_params['ucgmts_config']['consistc_ratio']
        cfg.model.policy.autoregressive_model_params.ucgmts_config.scaled_cbl_eps = bayesian_params['ucgmts_config']['scaled_cbl_eps']
        cfg.model.policy.autoregressive_model_params.ucgmts_config.ema_decay_rate = bayesian_params['ucgmts_config']['ema_decay_rate']
        cfg.model.policy.autoregressive_model_params.ucgmts_config.rfba_gap_steps = bayesian_params['ucgmts_config']['rfba_gap_steps']
        cfg.model.policy.autoregressive_model_params.ucgmts_config.extrapol_ratio = bayesian_params['ucgmts_config']['extrapol_ratio']
    
    print("Updated parameters:")
    params = cfg.model.policy.autoregressive_model_params
    print(f"  num_sampling_steps: {params.num_sampling_steps}")
    print(f"  cfg: {params.cfg}")
    print(f"  temperature: {params.temperature}")
    print(f"  window_size: {params.window_size}")
    print(f"  lambda_local: {params.lambda_local}")
    print(f"  consistc_ratio: {params.ucgmts_config.consistc_ratio}")
    print(f"  rfba_gap_steps: {params.ucgmts_config.rfba_gap_steps}")
    print(f"  extrapol_ratio: {params.ucgmts_config.extrapol_ratio}")
    
    # Save optimized checkpoint
    payload["cfg"] = cfg
    print(f"\nSaving optimized checkpoint...")
    os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)
    torch.save(payload, open(output_checkpoint, "wb"), pickle_module=dill)
    
    print(f"\n{'='*60}")
    print("SUCCESS!")
    print(f"{'='*60}")
    print(f"✓ Created perfect optimized checkpoint: {output_checkpoint}")
    print(f"✓ Contains ALL Bayesian optimization parameters")
    print(f"✓ Expected performance: 0.9919")
    print(f"\nNow you can evaluate without any manual parameters:")
    print(f"CUDA_VISIBLE_DEVICES=0 python eval_sim.py \\")
    print(f"    --checkpoint {output_checkpoint} \\")
    print(f"    --output_dir checkpoints/pusht_perfect_eval/")
    print(f"{'='*60}")

if __name__ == "__main__":
    create_perfect_optimized_checkpoint()
