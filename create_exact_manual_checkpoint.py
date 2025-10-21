#!/usr/bin/env python3
"""
Create a checkpoint with the EXACT parameters that manual evaluation uses.
This script replicates exactly what eval_sim.py does when manual parameters are passed.
"""

import os
import torch
import dill
import hydra
from omegaconf import OmegaConf, open_dict
from unified_video_action.workspace.train_unified_video_action_workspace import TrainUnifiedVideoActionWorkspace

def create_exact_manual_params_checkpoint():
    """
    Create a checkpoint with the exact parameters that manual evaluation uses.
    This replicates the exact behavior of eval_sim.py with manual parameters.
    """
    
    # Manual evaluation parameters (exactly what you passed)
    manual_params = {
        'use_ucgm': True,
        'num_sampling_steps': 2,
        'stochasticity_rate': 0.9870031611149999,
        'window_size': 8,
        'lambda_local': 0.3566535175072145
    }
    
    input_checkpoint = "checkpoints/pusht_production_opt/checkpoints/epoch=0050-test_mean_score=0.991.ckpt"
    output_checkpoint = "checkpoints/pusht_production_opt/checkpoints/epoch=0050-test_mean_score=0.991_EXACT_MANUAL.ckpt"
    
    print(f"{'='*60}")
    print("CREATING CHECKPOINT WITH EXACT MANUAL PARAMETERS")
    print(f"{'='*60}")
    print(f"Input: {input_checkpoint}")
    print(f"Output: {output_checkpoint}")
    print(f"Manual parameters:")
    for key, value in manual_params.items():
        print(f"  {key}: {value}")
    
    # Load original checkpoint
    print(f"\nLoading checkpoint...")
    payload = torch.load(open(input_checkpoint, "rb"), pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    
    # Apply parameters EXACTLY as eval_sim.py does
    print(f"\nApplying parameters exactly as eval_sim.py does...")
    
    with open_dict(cfg.model.policy.autoregressive_model_params):
        # Create ucgmts_config if it doesn't exist (exactly as eval_sim.py does)
        if "ucgmts_config" not in cfg.model.policy.autoregressive_model_params:
            cfg.model.policy.autoregressive_model_params.ucgmts_config = OmegaConf.create({})
            cfg.model.policy.autoregressive_model_params.ucgmts_config.transport_type = "Linear"
            cfg.model.policy.autoregressive_model_params.ucgmts_config.scaled_cbs_eps = 0.0
            cfg.model.policy.autoregressive_model_params.ucgmts_config.ema_decay_rate = 0.0
        
        # Apply stochasticity_rate (consistc_ratio)
        if manual_params['stochasticity_rate'] is not None:
            cfg.model.policy.autoregressive_model_params.ucgmts_config.consistc_ratio = manual_params['stochasticity_rate']
        
        # Apply num_sampling_steps and rfba_gap_steps (exactly as eval_sim.py does)
        if manual_params['num_sampling_steps']:
            cfg.model.policy.autoregressive_model_params.num_sampling_steps = manual_params['num_sampling_steps']
            if manual_params['num_sampling_steps'] <= 2:
                cfg.model.policy.autoregressive_model_params.ucgmts_config.rfba_gap_steps = [0.001, 0.5] 
            else:
                cfg.model.policy.autoregressive_model_params.ucgmts_config.rfba_gap_steps = [0.001, 0.001]
        
        # Apply local attention parameters
        if manual_params['window_size'] is not None:
            cfg.model.policy.autoregressive_model_params.window_size = manual_params['window_size']
        if manual_params['lambda_local'] is not None:
            cfg.model.policy.autoregressive_model_params.lambda_local = manual_params['lambda_local']
    
    # Apply use_ucgm flag (exactly as eval_sim.py does)
    if manual_params['use_ucgm']:
        OmegaConf.set_struct(cfg, False)  # Allow new keys
        cfg.model.policy.autoregressive_model_params.use_ucgm = True
        print("Using UCGM mode")
    
    # Apply pruning settings (exactly as eval_sim.py does)
    OmegaConf.set_struct(cfg, False)  # Allow new keys
    cfg.model.policy.autoregressive_model_params.pruning_ratios = None
    cfg.model.policy.autoregressive_model_params.token_pruning = False
    cfg.model.policy.autoregressive_model_params.restore_after_encoder = False
    
    print("Parameters applied exactly as eval_sim.py:")
    params = cfg.model.policy.autoregressive_model_params
    print(f"  use_ucgm: {params.use_ucgm}")
    print(f"  num_sampling_steps: {params.num_sampling_steps}")
    print(f"  window_size: {params.window_size}")
    print(f"  lambda_local: {params.lambda_local}")
    print(f"  ucgmts_config:")
    print(f"    transport_type: {params.ucgmts_config.transport_type}")
    print(f"    consistc_ratio: {params.ucgmts_config.consistc_ratio}")
    print(f"    scaled_cbs_eps: {params.ucgmts_config.scaled_cbs_eps}")
    print(f"    ema_decay_rate: {params.ucgmts_config.ema_decay_rate}")
    print(f"    rfba_gap_steps: {params.ucgmts_config.rfba_gap_steps}")
    
    # Create workspace with updated config
    print(f"\nCreating workspace with exact manual parameters...")
    workspace = TrainUnifiedVideoActionWorkspace(cfg, output_dir="./temp_workspace")
    
    # Load the model weights
    print(f"\nLoading model weights...")
    workspace.load_payload_new(payload, exclude_keys=None, include_keys=None, strict=False)
    
    # Apply parameters to model components (same as before)
    print(f"\nApplying parameters to model components...")
    model = workspace.model
    
    # Apply to autoregressive_model_params
    if hasattr(model, 'autoregressive_model_params'):
        autoregressive_params = model.autoregressive_model_params
    elif hasattr(model, 'model') and hasattr(model.model, 'autoregressive_model_params'):
        autoregressive_params = model.model.autoregressive_model_params
    else:
        print("❌ Cannot find autoregressive_model_params")
        return False
    
    autoregressive_params.use_ucgm = True
    autoregressive_params.num_sampling_steps = manual_params['num_sampling_steps']
    autoregressive_params.window_size = manual_params['window_size']
    autoregressive_params.lambda_local = manual_params['lambda_local']
    
    if not hasattr(autoregressive_params, 'ucgmts_config'):
        autoregressive_params.ucgmts_config = OmegaConf.create({})
    
    autoregressive_params.ucgmts_config.transport_type = "Linear"
    autoregressive_params.ucgmts_config.consistc_ratio = manual_params['stochasticity_rate']
    autoregressive_params.ucgmts_config.scaled_cbs_eps = 0.0
    autoregressive_params.ucgmts_config.ema_decay_rate = 0.0
    autoregressive_params.ucgmts_config.rfba_gap_steps = [0.001, 0.5]  # Exactly as eval_sim.py sets it
    
    print("✓ Updated autoregressive_model_params")
    
    # Apply to actual model components
    if hasattr(model, 'model') and hasattr(model.model, 'diffactloss'):
        diffactloss = model.model.diffactloss
        diffactloss.num_sampling_steps = manual_params['num_sampling_steps']
        
        if hasattr(diffactloss, 'ucgmts'):
            ucgmts = diffactloss.ucgmts
            ucgmts.transport_type = "Linear"
            ucgmts.consistc_ratio = manual_params['stochasticity_rate']
            ucgmts.scaled_cbs_eps = 0.0
            ucgmts.ema_decay_rate = 0.0
            ucgmts.rfba_gap_steps = [0.001, 0.5]  # Exactly as eval_sim.py sets it
            print("✓ Updated UCGMTS model components")
    
    # Also update EMA model if it exists
    if hasattr(workspace, 'ema_model') and workspace.ema_model is not None:
        print(f"\nUpdating EMA model...")
        ema_model = workspace.ema_model
        
        if hasattr(ema_model, 'autoregressive_model_params'):
            ema_autoregressive_params = ema_model.autoregressive_model_params
        elif hasattr(ema_model, 'model') and hasattr(ema_model.model, 'autoregressive_model_params'):
            ema_autoregressive_params = ema_model.model.autoregressive_model_params
        else:
            print("❌ Cannot find EMA autoregressive_model_params")
            return False
        
        ema_autoregressive_params.use_ucgm = True
        ema_autoregressive_params.num_sampling_steps = manual_params['num_sampling_steps']
        ema_autoregressive_params.window_size = manual_params['window_size']
        ema_autoregressive_params.lambda_local = manual_params['lambda_local']
        
        if not hasattr(ema_autoregressive_params, 'ucgmts_config'):
            ema_autoregressive_params.ucgmts_config = OmegaConf.create({})
        
        ema_autoregressive_params.ucgmts_config.transport_type = "Linear"
        ema_autoregressive_params.ucgmts_config.consistc_ratio = manual_params['stochasticity_rate']
        ema_autoregressive_params.ucgmts_config.scaled_cbs_eps = 0.0
        ema_autoregressive_params.ucgmts_config.ema_decay_rate = 0.0
        ema_autoregressive_params.ucgmts_config.rfba_gap_steps = [0.001, 0.5]  # Exactly as eval_sim.py sets it
        
        # Also update EMA model components
        if hasattr(ema_model, 'model') and hasattr(ema_model.model, 'diffactloss'):
            ema_diffactloss = ema_model.model.diffactloss
            ema_diffactloss.num_sampling_steps = manual_params['num_sampling_steps']
            
            if hasattr(ema_diffactloss, 'ucgmts'):
                ema_ucgmts = ema_diffactloss.ucgmts
                ema_ucgmts.transport_type = "Linear"
                ema_ucgmts.consistc_ratio = manual_params['stochasticity_rate']
                ema_ucgmts.scaled_cbs_eps = 0.0
                ema_ucgmts.ema_decay_rate = 0.0
                ema_ucgmts.rfba_gap_steps = [0.001, 0.5]  # Exactly as eval_sim.py sets it
        
        print("✓ Updated EMA model")
    
    # Save the workspace as a new checkpoint
    print(f"\nSaving exact manual parameters checkpoint...")
    os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)
    workspace.save_checkpoint(path=output_checkpoint, tag="exact_manual")
    
    print(f"\n{'='*60}")
    print("SUCCESS!")
    print(f"{'='*60}")
    print(f"✓ Created checkpoint with EXACT manual parameters: {output_checkpoint}")
    print(f"✓ Replicates exact eval_sim.py behavior")
    print(f"✓ Expected performance: 0.991887845932303")
    print(f"\nNow you can evaluate without any manual parameters:")
    print(f"CUDA_VISIBLE_DEVICES=0 python eval_sim.py \\")
    print(f"    --checkpoint {output_checkpoint} \\")
    print(f"    --output_dir checkpoints/pusht_exact_eval/")
    print(f"{'='*60}")
    
    return True

if __name__ == "__main__":
    create_exact_manual_params_checkpoint()
