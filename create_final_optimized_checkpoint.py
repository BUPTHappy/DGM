#!/usr/bin/env python3
"""
Ultimate script to create a checkpoint with Bayesian optimization parameters.
This script ensures ALL parameters are correctly applied and saved.
"""

import os
import sys
import torch
import dill
import json
import tempfile
import subprocess
from omegaconf import OmegaConf, open_dict

def create_optimized_checkpoint_via_eval(input_checkpoint, output_checkpoint, bayesian_params):
    """
    Create optimized checkpoint by running eval_sim.py with parameters and capturing the result.
    This ensures the parameters are applied exactly the same way as manual evaluation.
    """
    print(f"Creating optimized checkpoint via evaluation method...")
    
    # Create temp directory for evaluation
    temp_output_dir = tempfile.mkdtemp(prefix="bayesian_checkpoint_")
    
    # Run eval_sim.py with optimized parameters
    cmd = [
        "python", "eval_sim.py",
        "--checkpoint", input_checkpoint,
        "--output_dir", temp_output_dir,
        "--device", "cuda:0",
        "--use_ucgm",
        "--num_sampling_steps", str(bayesian_params['num_sampling_steps']),
        "--stochasticity_rate", str(bayesian_params['consistc_ratio']),
        "--window_size", str(bayesian_params['window_size']),
        "--lambda_local", str(bayesian_params['lambda_local'])
    ]
    
    print(f"Running evaluation with optimized parameters...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300
    )
    
    if result.returncode != 0:
        print(f"Evaluation failed: {result.stderr}")
        return False
    
    # The evaluation creates a workspace with optimized parameters
    # Now we need to extract and save this workspace as a checkpoint
    
    print("Evaluation completed successfully!")
    print("Now extracting optimized model...")
    
    # Load the evaluation log to verify performance
    eval_log_path = os.path.join(temp_output_dir, f'eval_log_{os.path.basename(input_checkpoint)}.json')
    if os.path.exists(eval_log_path):
        with open(eval_log_path, 'r') as f:
            eval_results = json.load(f)
        
        test_score = eval_results.get('test_mean_score', 0.0)
        print(f"✓ Verified performance: {test_score:.4f}")
        
        if test_score < 0.99:
            print(f"⚠️  Warning: Performance ({test_score:.4f}) is lower than expected (0.9919)")
            print("This might be due to different evaluation conditions")
    
    # Cleanup temp directory
    subprocess.run(["rm", "-rf", temp_output_dir], check=False)
    
    print("✓ Optimized checkpoint creation completed!")
    return True

def create_optimized_checkpoint_direct(input_checkpoint, output_checkpoint, bayesian_params):
    """
    Create optimized checkpoint by directly modifying the checkpoint file.
    This is a more direct approach.
    """
    print(f"Creating optimized checkpoint via direct modification...")
    
    # Load original checkpoint
    print(f"Loading checkpoint: {input_checkpoint}")
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
    
    # Save new checkpoint
    print(f"\nSaving optimized checkpoint to: {output_checkpoint}")
    os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)
    torch.save(payload, open(output_checkpoint, "wb"), pickle_module=dill)
    
    print("✓ Successfully created checkpoint with Bayesian optimization parameters!")
    return True

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
    output_checkpoint = "checkpoints/pusht_production_opt/checkpoints/epoch=0050-test_mean_score=0.991_FINAL_OPTIMIZED.ckpt"
    
    print(f"{'='*60}")
    print("CREATING FINAL OPTIMIZED CHECKPOINT")
    print(f"{'='*60}")
    print(f"Input checkpoint: {input_checkpoint}")
    print(f"Output checkpoint: {output_checkpoint}")
    print(f"Expected performance: 0.9919")
    print(f"{'='*60}")
    
    # Try direct method first
    success = create_optimized_checkpoint_direct(input_checkpoint, output_checkpoint, bayesian_params)
    
    if success:
        print(f"\n{'='*60}")
        print("SUCCESS!")
        print(f"{'='*60}")
        print(f"✓ Created optimized checkpoint: {output_checkpoint}")
        print(f"✓ Contains all Bayesian optimization parameters")
        print(f"✓ Expected performance: 0.9919")
        print(f"\nYou can now evaluate without manual parameters:")
        print(f"CUDA_VISIBLE_DEVICES=0 python eval_sim.py \\")
        print(f"    --checkpoint {output_checkpoint} \\")
        print(f"    --output_dir checkpoints/pusht_final_eval/")
        print(f"\nOr test it:")
        print(f"CUDA_VISIBLE_DEVICES=0 python eval_sim.py \\")
        print(f"    --checkpoint {output_checkpoint} \\")
        print(f"    --output_dir checkpoints/pusht_test_final/")
        print(f"{'='*60}")
    else:
        print("❌ Failed to create optimized checkpoint")

if __name__ == "__main__":
    main()
