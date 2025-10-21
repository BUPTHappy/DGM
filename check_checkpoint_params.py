#!/usr/bin/env python3
"""
Check what parameters are actually saved in our optimized checkpoint
"""

import torch
import dill
from omegaconf import OmegaConf

def check_checkpoint_params(checkpoint_path):
    print(f"Checking parameters in: {checkpoint_path}")
    
    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    
    params = cfg.model.policy.autoregressive_model_params
    
    print("Current parameters in checkpoint:")
    print(f"  use_ucgm: {params.use_ucgm}")
    print(f"  num_sampling_steps: {params.num_sampling_steps}")
    print(f"  cfg: {params.cfg}")
    print(f"  temperature: {params.temperature}")
    print(f"  window_size: {getattr(params, 'window_size', 'N/A')}")
    print(f"  lambda_local: {getattr(params, 'lambda_local', 'N/A')}")
    
    if hasattr(params, 'ucgmts_config') and params.ucgmts_config:
        print(f"  ucgmts_config:")
        print(f"    transport_type: {params.ucgmts_config.transport_type}")
        print(f"    consistc_ratio: {params.ucgmts_config.consistc_ratio}")
        print(f"    scaled_cbl_eps: {params.ucgmts_config.scaled_cbl_eps}")
        print(f"    ema_decay_rate: {params.ucgmts_config.ema_decay_rate}")
        print(f"    rfba_gap_steps: {params.ucgmts_config.rfba_gap_steps}")
        print(f"    extrapol_ratio: {getattr(params.ucgmts_config, 'extrapol_ratio', 'N/A')}")
    else:
        print("  ucgmts_config: Not found or empty")

if __name__ == "__main__":
    # Check original checkpoint
    print("="*60)
    print("ORIGINAL CHECKPOINT")
    print("="*60)
    check_checkpoint_params("checkpoints/pusht_production_opt/checkpoints/epoch=0050-test_mean_score=0.991.ckpt")
    
    print("\n" + "="*60)
    print("OPTIMIZED CHECKPOINT")
    print("="*60)
    check_checkpoint_params("checkpoints/pusht_production_opt/checkpoints/epoch=0050-test_mean_score=0.991_FINAL_OPTIMIZED.ckpt")
