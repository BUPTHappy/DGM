#!/usr/bin/env python3
"""
Enhanced script to apply Bayesian optimization parameters to checkpoint.
This script ensures parameters are applied both to config and to the actual model components.
"""

import os
import sys
import torch
import dill
import json
from omegaconf import OmegaConf, open_dict

def apply_params_to_model_components(policy, params):
    """Apply parameters to actual model components, not just config."""
    try:
        print("Applying parameters to model components...")
        
        # Update autoregressive_model_params
        if hasattr(policy, 'autoregressive_model_params'):
            autoregressive_params = policy.autoregressive_model_params
            
            # Update UCGM parameters
            autoregressive_params.use_ucgm = True
            autoregressive_params.num_sampling_steps = params['num_sampling_steps']
            autoregressive_params.cfg = params['cfg']
            autoregressive_params.temperature = params['temperature']
            autoregressive_params.window_size = params['window_size']
            autoregressive_params.lambda_local = params['lambda_local']
            
            # Update UCGMTS config
            if not hasattr(autoregressive_params, 'ucgmts_config'):
                autoregressive_params.ucgmts_config = OmegaConf.create({})
            
            autoregressive_params.ucgmts_config.transport_type = params['ucgmts_config']['transport_type']
            autoregressive_params.ucgmts_config.consistc_ratio = params['ucgmts_config']['consistc_ratio']
            autoregressive_params.ucgmts_config.scaled_cbl_eps = params['ucgmts_config']['scaled_cbl_eps']
            autoregressive_params.ucgmts_config.ema_decay_rate = params['ucgmts_config']['ema_decay_rate']
            autoregressive_params.ucgmts_config.rfba_gap_steps = params['ucgmts_config']['rfba_gap_steps']
            autoregressive_params.ucgmts_config.extrapol_ratio = params['ucgmts_config']['extrapol_ratio']
            
            print("✓ Updated autoregressive_model_params")
        
        # Update actual model components
        if hasattr(policy, 'model') and hasattr(policy.model, 'diffactloss'):
            diffactloss = policy.model.diffactloss
            diffactloss.num_sampling_steps = params['num_sampling_steps']
            
            if hasattr(diffactloss, 'ucgmts'):
                ucgmts = diffactloss.ucgmts
                ucgmts.transport_type = params['ucgmts_config']['transport_type']
                ucgmts.consistc_ratio = params['ucgmts_config']['consistc_ratio']
                ucgmts.scaled_cbl_eps = params['ucgmts_config']['scaled_cbl_eps']
                ucgmts.ema_decay_rate = params['ucgmts_config']['ema_decay_rate']
                ucgmts.rfba_gap_steps = params['ucgmts_config']['rfba_gap_steps']
                ucgmts.extrapol_ratio = params['ucgmts_config']['extrapol_ratio']
                
                print("✓ Updated UCGMTS model components")
        
        # Also update EMA model if it exists
        if hasattr(policy, 'ema_model') and policy.ema_model is not None:
            print("Updating EMA model components...")
            apply_params_to_model_components(policy.ema_model, params)
        
        return True
        
    except Exception as e:
        print(f"Error applying parameters to model components: {e}")
        return False

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
    output_checkpoint = "checkpoints/pusht_production_opt/checkpoints/epoch=0050-test_mean_score=0.991_optimized_v2.ckpt"
    
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
    
    print("\nUpdated config parameters:")
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
    
    # Now we need to load the workspace and apply parameters to model components
    print("\nLoading workspace to apply parameters to model components...")
    
    # Import necessary modules
    import hydra
    from unified_video_action.workspace.base_workspace import BaseWorkspace
    
    # Create workspace
    cls = hydra.utils.get_class(cfg.model._target_)
    workspace = cls(cfg, output_dir="temp_output")
    
    # Load the payload
    workspace.load_payload_new(payload, exclude_keys=None, include_keys=None, strict=False)
    
    # Apply parameters to model components
    policy = workspace.model
    if cfg.training.use_ema and hasattr(workspace, 'ema_model'):
        policy = workspace.ema_model
    
    success = apply_params_to_model_components(policy, bayesian_params)
    
    if success:
        print("✓ Successfully applied parameters to model components")
        
        # Save the updated workspace state
        print(f"\nSaving optimized checkpoint to: {output_checkpoint}")
        os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)
        
        # Create new payload with updated workspace
        new_payload = {"cfg": cfg, "state_dicts": dict(), "pickles": dict()}
        
        # Save model state
        if hasattr(workspace, 'model'):
            new_payload["state_dicts"]["model"] = workspace.model.state_dict()
        if hasattr(workspace, 'ema_model') and workspace.ema_model is not None:
            new_payload["state_dicts"]["ema_model"] = workspace.ema_model.state_dict()
        
        # Save other workspace attributes
        for key, value in workspace.__dict__.items():
            if key in ["global_step", "epoch"]:
                new_payload["pickles"][key] = dill.dumps(value)
        
        torch.save(new_payload, open(output_checkpoint, "wb"), pickle_module=dill)
        
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
        print(f"    --output_dir checkpoints/pusht_optimized_eval_v2/")
        print(f"{'='*60}")
    else:
        print("❌ Failed to apply parameters to model components")

if __name__ == "__main__":
    main()
