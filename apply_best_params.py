#!/usr/bin/env python3
"""
Apply Bayesian-optimized parameters to a checkpoint and save as a new checkpoint.

The new checkpoint is self-contained: eval_offline.py can load it directly
without passing any extra --num_sampling_steps / --stochasticity_rate / etc.

Usage:
    # From best_params.json (output of bayesian_optimization_offline.py)
    python apply_best_params.py \
        --checkpoint checkpoints/dish_washing_dgm_default/checkpoints/latest.ckpt \
        --best_params bayesian_optimization_results/best_params.json \
        --output checkpoints/dish_washing_dgm_default/checkpoints/optimized.ckpt

    # Then evaluate directly - no extra flags needed:
    python eval_offline.py \
        --checkpoint checkpoints/dish_washing_dgm_default/checkpoints/optimized.ckpt \
        --output_dir eval_results/dish_washing_optimized \
        --use_ucgm
"""

import os
import sys
import json
import torch
import dill
from omegaconf import OmegaConf, open_dict


def apply_params_to_checkpoint(checkpoint_path: str, 
                                best_params: dict, 
                                output_path: str):
    """
    Load checkpoint, write best UCGM params into cfg, save new checkpoint.
    
    Args:
        checkpoint_path: Original checkpoint path
        best_params: Dict of optimized parameters
        output_path: Where to save the new checkpoint
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    
    # Print original params
    print("\n--- ORIGINAL PARAMS ---")
    arp = cfg.model.policy.autoregressive_model_params
    print(f"  use_ucgm: {getattr(arp, 'use_ucgm', 'N/A')}")
    print(f"  num_sampling_steps: {getattr(arp, 'num_sampling_steps', 'N/A')}")
    print(f"  cfg: {getattr(arp, 'cfg', 'N/A')}")
    print(f"  temperature: {getattr(arp, 'temperature', 'N/A')}")
    print(f"  window_size: {getattr(arp, 'window_size', 'N/A')}")
    print(f"  lambda_local: {getattr(arp, 'lambda_local', 'N/A')}")
    if hasattr(arp, 'ucgmts_config'):
        print(f"  ucgmts_config.consistc_ratio: {getattr(arp.ucgmts_config, 'consistc_ratio', 'N/A')}")
        print(f"  ucgmts_config.rfba_gap_steps: {getattr(arp.ucgmts_config, 'rfba_gap_steps', 'N/A')}")
        print(f"  ucgmts_config.extrapol_ratio: {getattr(arp.ucgmts_config, 'extrapol_ratio', 'N/A')}")
    
    # Apply optimized params
    with open_dict(cfg.model.policy.autoregressive_model_params):
        if "ucgmts_config" not in cfg.model.policy.autoregressive_model_params:
            cfg.model.policy.autoregressive_model_params.ucgmts_config = OmegaConf.create({})
        
        cfg.model.policy.autoregressive_model_params.use_ucgm = True
        
        # Core params
        if 'num_sampling_steps' in best_params:
            cfg.model.policy.autoregressive_model_params.num_sampling_steps = best_params['num_sampling_steps']
        if 'cfg' in best_params:
            cfg.model.policy.autoregressive_model_params.cfg = best_params['cfg']
        if 'temperature' in best_params:
            cfg.model.policy.autoregressive_model_params.temperature = best_params['temperature']
        if 'window_size' in best_params:
            cfg.model.policy.autoregressive_model_params.window_size = best_params['window_size']
        if 'lambda_local' in best_params:
            cfg.model.policy.autoregressive_model_params.lambda_local = best_params['lambda_local']
        
        # UCGMTS config
        ucgmts = best_params.get('ucgmts_config', {})
        if ucgmts:
            if 'transport_type' in ucgmts:
                cfg.model.policy.autoregressive_model_params.ucgmts_config.transport_type = ucgmts['transport_type']
            if 'consistc_ratio' in ucgmts:
                cfg.model.policy.autoregressive_model_params.ucgmts_config.consistc_ratio = ucgmts['consistc_ratio']
            if 'scaled_cbl_eps' in ucgmts:
                cfg.model.policy.autoregressive_model_params.ucgmts_config.scaled_cbl_eps = ucgmts['scaled_cbl_eps']
            if 'ema_decay_rate' in ucgmts:
                cfg.model.policy.autoregressive_model_params.ucgmts_config.ema_decay_rate = ucgmts['ema_decay_rate']
            if 'rfba_gap_steps' in ucgmts:
                cfg.model.policy.autoregressive_model_params.ucgmts_config.rfba_gap_steps = ucgmts['rfba_gap_steps']
            if 'extrapol_ratio' in ucgmts:
                cfg.model.policy.autoregressive_model_params.ucgmts_config.extrapol_ratio = ucgmts['extrapol_ratio']
    
    # Write back to payload
    payload["cfg"] = cfg
    
    # Print new params
    print("\n--- OPTIMIZED PARAMS (written to checkpoint) ---")
    arp = cfg.model.policy.autoregressive_model_params
    print(f"  use_ucgm: {arp.use_ucgm}")
    print(f"  num_sampling_steps: {arp.num_sampling_steps}")
    print(f"  cfg: {arp.cfg}")
    print(f"  temperature: {arp.temperature}")
    print(f"  window_size: {arp.window_size}")
    print(f"  lambda_local: {arp.lambda_local}")
    print(f"  ucgmts_config.transport_type: {arp.ucgmts_config.transport_type}")
    print(f"  ucgmts_config.consistc_ratio: {arp.ucgmts_config.consistc_ratio}")
    print(f"  ucgmts_config.rfba_gap_steps: {arp.ucgmts_config.rfba_gap_steps}")
    print(f"  ucgmts_config.extrapol_ratio: {arp.ucgmts_config.extrapol_ratio}")
    
    # Save new checkpoint
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"\nSaving optimized checkpoint to: {output_path}")
    torch.save(payload, open(output_path, "wb"), pickle_module=dill)
    
    # Verify
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved! File size: {file_size_mb:.1f} MB")
    print(f"\nYou can now evaluate directly:")
    print(f"  python eval_offline.py --checkpoint {output_path} --output_dir eval_results/ --use_ucgm")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Apply Bayesian-optimized params to checkpoint, save as new checkpoint"
    )
    parser.add_argument("--checkpoint", required=True, help="Original checkpoint path")
    parser.add_argument("--best_params", required=True, 
                        help="Path to best_params.json (from bayesian_optimization_offline.py)")
    parser.add_argument("--output", default=None,
                        help="Output checkpoint path (default: <checkpoint_dir>/optimized.ckpt)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    if not os.path.exists(args.best_params):
        print(f"Error: best_params.json not found: {args.best_params}")
        sys.exit(1)
    
    # Load best params
    with open(args.best_params, 'r') as f:
        best_params = json.load(f)
    
    # Default output path
    if args.output is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        args.output = os.path.join(ckpt_dir, "optimized.ckpt")
    
    apply_params_to_checkpoint(args.checkpoint, best_params, args.output)
