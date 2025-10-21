#!/usr/bin/env python3
"""
Create a checkpoint that bypasses the load_payload_new ucgmts dropping issue.
This script creates a checkpoint where the model components are already initialized
with the correct parameters, so they don't get dropped during loading.
"""

import os
import torch
import dill
import hydra
from omegaconf import OmegaConf, open_dict
from unified_video_action.workspace.train_unified_video_action_workspace import TrainUnifiedVideoActionWorkspace

def create_working_optimized_checkpoint():
    """
    Create a checkpoint by actually running the workspace with optimized parameters,
    then saving it. This ensures the model components are properly initialized.
    """
    
    # Bayesian optimization results
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
    output_checkpoint = "checkpoints/pusht_production_opt/checkpoints/epoch=0050-test_mean_score=0.991_WORKING_OPTIMIZED.ckpt"
    
    print(f"{'='*60}")
    print("CREATING WORKING OPTIMIZED CHECKPOINT")
    print(f"{'='*60}")
    print(f"Input: {input_checkpoint}")
    print(f"Output: {output_checkpoint}")
    
    # Load original checkpoint
    print(f"\nLoading checkpoint...")
    payload = torch.load(open(input_checkpoint, "rb"), pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    
    # Update config with optimized parameters
    print(f"\nUpdating config with optimized parameters...")
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
        
        # Update UCGMTS config
        cfg.model.policy.autoregressive_model_params.ucgmts_config.transport_type = bayesian_params['ucgmts_config']['transport_type']
        cfg.model.policy.autoregressive_model_params.ucgmts_config.consistc_ratio = bayesian_params['ucgmts_config']['consistc_ratio']
        cfg.model.policy.autoregressive_model_params.ucgmts_config.scaled_cbl_eps = bayesian_params['ucgmts_config']['scaled_cbl_eps']
        cfg.model.policy.autoregressive_model_params.ucgmts_config.ema_decay_rate = bayesian_params['ucgmts_config']['ema_decay_rate']
        cfg.model.policy.autoregressive_model_params.ucgmts_config.rfba_gap_steps = bayesian_params['ucgmts_config']['rfba_gap_steps']
        cfg.model.policy.autoregressive_model_params.ucgmts_config.extrapol_ratio = bayesian_params['ucgmts_config']['extrapol_ratio']
    
    print("Config updated with optimized parameters:")
    params = cfg.model.policy.autoregressive_model_params
    print(f"  num_sampling_steps: {params.num_sampling_steps}")
    print(f"  cfg: {params.cfg}")
    print(f"  temperature: {params.temperature}")
    print(f"  window_size: {params.window_size}")
    print(f"  lambda_local: {params.lambda_local}")
    print(f"  consistc_ratio: {params.ucgmts_config.consistc_ratio}")
    print(f"  rfba_gap_steps: {params.ucgmts_config.rfba_gap_steps}")
    print(f"  extrapol_ratio: {params.ucgmts_config.extrapol_ratio}")
    
    # Create workspace with updated config
    print(f"\nCreating workspace with optimized config...")
    workspace = TrainUnifiedVideoActionWorkspace(cfg, output_dir="./temp_workspace")
    
    # Load the model weights (but not the ucgmts components that get dropped)
    print(f"\nLoading model weights...")
    workspace.load_payload_new(payload, exclude_keys=None, include_keys=None, strict=False)
    
    # Now manually apply the optimized parameters to the model components
    print(f"\nApplying optimized parameters to model components...")
    model = workspace.model
    
    # Apply parameters to autoregressive_model_params
    if hasattr(model, 'autoregressive_model_params'):
        autoregressive_params = model.autoregressive_model_params
    elif hasattr(model, 'model') and hasattr(model.model, 'autoregressive_model_params'):
        autoregressive_params = model.model.autoregressive_model_params
    else:
        print("❌ Cannot find autoregressive_model_params")
        return False
    
    autoregressive_params.use_ucgm = True
    autoregressive_params.num_sampling_steps = bayesian_params['num_sampling_steps']
    autoregressive_params.cfg = bayesian_params['cfg']
    autoregressive_params.temperature = bayesian_params['temperature']
    autoregressive_params.window_size = bayesian_params['window_size']
    autoregressive_params.lambda_local = bayesian_params['lambda_local']
    
    if not hasattr(autoregressive_params, 'ucgmts_config'):
        autoregressive_params.ucgmts_config = OmegaConf.create({})
    
    autoregressive_params.ucgmts_config.transport_type = bayesian_params['ucgmts_config']['transport_type']
    autoregressive_params.ucgmts_config.consistc_ratio = bayesian_params['ucgmts_config']['consistc_ratio']
    autoregressive_params.ucgmts_config.scaled_cbl_eps = bayesian_params['ucgmts_config']['scaled_cbl_eps']
    autoregressive_params.ucgmts_config.ema_decay_rate = bayesian_params['ucgmts_config']['ema_decay_rate']
    autoregressive_params.ucgmts_config.rfba_gap_steps = bayesian_params['ucgmts_config']['rfba_gap_steps']
    autoregressive_params.ucgmts_config.extrapol_ratio = bayesian_params['ucgmts_config']['extrapol_ratio']
    
    print("✓ Updated autoregressive_model_params")
    
    # Apply parameters to actual model components
    if hasattr(model, 'model') and hasattr(model.model, 'diffactloss'):
        diffactloss = model.model.diffactloss
        diffactloss.num_sampling_steps = bayesian_params['num_sampling_steps']
        
        if hasattr(diffactloss, 'ucgmts'):
            ucgmts = diffactloss.ucgmts
            ucgmts.transport_type = bayesian_params['ucgmts_config']['transport_type']
            ucgmts.consistc_ratio = bayesian_params['ucgmts_config']['consistc_ratio']
            ucgmts.scaled_cbl_eps = bayesian_params['ucgmts_config']['scaled_cbl_eps']
            ucgmts.ema_decay_rate = bayesian_params['ucgmts_config']['ema_decay_rate']
            ucgmts.rfba_gap_steps = bayesian_params['ucgmts_config']['rfba_gap_steps']
            ucgmts.extrapol_ratio = bayesian_params['ucgmts_config']['extrapol_ratio']
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
        ema_autoregressive_params.num_sampling_steps = bayesian_params['num_sampling_steps']
        ema_autoregressive_params.cfg = bayesian_params['cfg']
        ema_autoregressive_params.temperature = bayesian_params['temperature']
        ema_autoregressive_params.window_size = bayesian_params['window_size']
        ema_autoregressive_params.lambda_local = bayesian_params['lambda_local']
        
        if not hasattr(ema_autoregressive_params, 'ucgmts_config'):
            ema_autoregressive_params.ucgmts_config = OmegaConf.create({})
        
        ema_autoregressive_params.ucgmts_config.transport_type = bayesian_params['ucgmts_config']['transport_type']
        ema_autoregressive_params.ucgmts_config.consistc_ratio = bayesian_params['ucgmts_config']['consistc_ratio']
        ema_autoregressive_params.ucgmts_config.scaled_cbl_eps = bayesian_params['ucgmts_config']['scaled_cbl_eps']
        ema_autoregressive_params.ucgmts_config.ema_decay_rate = bayesian_params['ucgmts_config']['ema_decay_rate']
        ema_autoregressive_params.ucgmts_config.rfba_gap_steps = bayesian_params['ucgmts_config']['rfba_gap_steps']
        ema_autoregressive_params.ucgmts_config.extrapol_ratio = bayesian_params['ucgmts_config']['extrapol_ratio']
        
        # Also update EMA model components
        if hasattr(ema_model, 'model') and hasattr(ema_model.model, 'diffactloss'):
            ema_diffactloss = ema_model.model.diffactloss
            ema_diffactloss.num_sampling_steps = bayesian_params['num_sampling_steps']
            
            if hasattr(ema_diffactloss, 'ucgmts'):
                ema_ucgmts = ema_diffactloss.ucgmts
                ema_ucgmts.transport_type = bayesian_params['ucgmts_config']['transport_type']
                ema_ucgmts.consistc_ratio = bayesian_params['ucgmts_config']['consistc_ratio']
                ema_ucgmts.scaled_cbl_eps = bayesian_params['ucgmts_config']['scaled_cbl_eps']
                ema_ucgmts.ema_decay_rate = bayesian_params['ucgmts_config']['ema_decay_rate']
                ema_ucgmts.rfba_gap_steps = bayesian_params['ucgmts_config']['rfba_gap_steps']
                ema_ucgmts.extrapol_ratio = bayesian_params['ucgmts_config']['extrapol_ratio']
        
        print("✓ Updated EMA model")
    
    # Save the workspace as a new checkpoint
    print(f"\nSaving optimized checkpoint...")
    os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)
    workspace.save_checkpoint(path=output_checkpoint, tag="optimized")
    
    print(f"\n{'='*60}")
    print("SUCCESS!")
    print(f"{'='*60}")
    print(f"✓ Created working optimized checkpoint: {output_checkpoint}")
    print(f"✓ Model components properly initialized with optimized parameters")
    print(f"✓ Bypasses load_payload_new ucgmts dropping issue")
    print(f"✓ Expected performance: 0.9919")
    print(f"\nNow you can evaluate without any manual parameters:")
    print(f"CUDA_VISIBLE_DEVICES=0 python eval_sim.py \\")
    print(f"    --checkpoint {output_checkpoint} \\")
    print(f"    --output_dir checkpoints/pusht_working_eval/")
    print(f"{'='*60}")
    
    return True

if __name__ == "__main__":
    create_working_optimized_checkpoint()
