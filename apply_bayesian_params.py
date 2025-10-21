#!/usr/bin/env python3
"""
Script to manually apply Bayesian optimization parameters to an existing checkpoint.
This creates a new checkpoint with the optimized parameters.
"""

import os
import sys
import torch
import dill
import json
from omegaconf import OmegaConf, open_dict

def apply_bayesian_params_to_checkpoint(input_checkpoint_path, output_checkpoint_path, bayesian_params):
    """
    Apply Bayesian optimization parameters to a checkpoint and save as new checkpoint.
    
    Args:
        input_checkpoint_path: Path to original checkpoint
        output_checkpoint_path: Path to save new checkpoint with optimized parameters
        bayesian_params: Dictionary containing optimized parameters
    """
    print(f"Loading checkpoint: {input_checkpoint_path}")
    
    # Load original checkpoint
    payload = torch.load(open(input_checkpoint_path, "rb"), pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    
    print("Original parameters:")
    print(f"  num_sampling_steps: {cfg.model.policy.autoregressive_model_params.num_sampling_steps}")
    print(f"  cfg: {cfg.model.policy.autoregressive_model_params.cfg}")
    print(f"  temperature: {cfg.model.policy.autoregressive_model_params.temperature}")
    print(f"  window_size: {cfg.model.policy.autoregressive_model_params.window_size}")
    print(f"  lambda_local: {cfg.model.policy.autoregressive_model_params.lambda_local}")
    
    # Update config with Bayesian optimization parameters
    with open_dict(cfg.model.policy.autoregressive_model_params):
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
    print(f"\nSaving optimized checkpoint to: {output_checkpoint_path}")
    os.makedirs(os.path.dirname(output_checkpoint_path), exist_ok=True)
    torch.save(payload, open(output_checkpoint_path, "wb"), pickle_module=dill)
    
    print("✓ Successfully created checkpoint with Bayesian optimization parameters!")
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply Bayesian optimization parameters to checkpoint')
    parser.add_argument('--input_checkpoint', required=True, help='Path to input checkpoint')
    parser.add_argument('--output_checkpoint', required=True, help='Path to output checkpoint')
    parser.add_argument('--bayesian_log', help='Path to Bayesian optimization log file (optional)')
    
    args = parser.parse_args()
    
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
    
    # Try to load parameters from Bayesian optimization log if provided
    if args.bayesian_log and os.path.exists(args.bayesian_log):
        try:
            with open(args.bayesian_log, 'r') as f:
                log_data = json.load(f)
                if 'best_params' in log_data:
                    bayesian_params = log_data['best_params']
                    print(f"Loaded parameters from: {args.bayesian_log}")
        except Exception as e:
            print(f"Failed to load from log file: {e}")
            print("Using default parameters")
    
    # Use provided paths
    input_checkpoint = args.input_checkpoint
    output_checkpoint = args.output_checkpoint
    
    # Apply parameters
    success = apply_bayesian_params_to_checkpoint(input_checkpoint, output_checkpoint, bayesian_params)
    
    if success:
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
