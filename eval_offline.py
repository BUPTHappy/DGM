#!/usr/bin/env python3
"""
Offline evaluation for real-world UMI datasets (no env_runner / simulator needed).

Computes:
  - Action L2 distance
  - End-effector trajectory error (per arm for bimanual)
  - Final state distance (per arm for bimanual)
  - FVD (optional, with --fvd flag)

Usage:
    python eval_offline.py \
        --checkpoint checkpoints/dish_washing_dgm_default/checkpoints/latest.ckpt \
        --output_dir eval_results/dish_washing \
        --device cuda:0
"""

import sys
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import os
import json
import time
import hydra
import torch
import dill
import numpy as np
import random
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader
import click

from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.eval.eval import (
    test_action_l2,
    test_eef_trajectory_error,
    test_video_fvd,
)


@click.command()
@click.option("-c", "--checkpoint", required=True, help="Path to model checkpoint")
@click.option("-o", "--output_dir", required=True, help="Output directory for results")
@click.option("-d", "--device", default="cuda:0", help="Device (default: cuda:0)")
@click.option("--no_ema", is_flag=True, help="Disable EMA, use raw model")
@click.option("--fvd", is_flag=True, help="Also compute FVD (slower)")
@click.option("--use_ucgm", is_flag=True, help="Enable UCGM mode")
# UCGM parameters (all optimized by Bayesian optimization)
@click.option("--num_sampling_steps", type=int, default=None, help="Override num_sampling_steps")
@click.option("--stochasticity_rate", type=float, default=None, help="Override consistc_ratio")
@click.option("--temperature", type=float, default=None, help="Override temperature")
@click.option("--cfg_scale", type=float, default=None, help="Override cfg (classifier-free guidance scale)")
@click.option("--window_size", type=int, default=None, help="Override window_size")
@click.option("--lambda_local", type=float, default=None, help="Override lambda_local")
@click.option("--rfba_gap_end", type=float, default=None, help="Override rfba_gap_steps end value")
@click.option("--extrapol_ratio", type=float, default=None, help="Override extrapol_ratio")
def main(checkpoint, output_dir, device, no_ema, fvd, use_ucgm,
         num_sampling_steps, stochasticity_rate, temperature, cfg_scale,
         window_size, lambda_local, rfba_gap_end, extrapol_ratio):
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("OFFLINE EVALUATION (Real-world UMI dataset)")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    
    # Apply UCGM overrides
    with open_dict(cfg.model.policy.autoregressive_model_params):
        if "ucgmts_config" not in cfg.model.policy.autoregressive_model_params:
            cfg.model.policy.autoregressive_model_params.ucgmts_config = OmegaConf.create({})
            cfg.model.policy.autoregressive_model_params.ucgmts_config.transport_type = "Linear"
            cfg.model.policy.autoregressive_model_params.ucgmts_config.scaled_cbs_eps = 0.0
            cfg.model.policy.autoregressive_model_params.ucgmts_config.ema_decay_rate = 0.0
        
        if use_ucgm:
            cfg.model.policy.autoregressive_model_params.use_ucgm = True
        if num_sampling_steps is not None:
            cfg.model.policy.autoregressive_model_params.num_sampling_steps = num_sampling_steps
        if temperature is not None:
            cfg.model.policy.autoregressive_model_params.temperature = temperature
        if cfg_scale is not None:
            cfg.model.policy.autoregressive_model_params.cfg = cfg_scale
        if window_size is not None:
            cfg.model.policy.autoregressive_model_params.window_size = window_size
        if lambda_local is not None:
            cfg.model.policy.autoregressive_model_params.lambda_local = lambda_local
        if stochasticity_rate is not None:
            cfg.model.policy.autoregressive_model_params.ucgmts_config.consistc_ratio = stochasticity_rate
        if rfba_gap_end is not None:
            cfg.model.policy.autoregressive_model_params.ucgmts_config.rfba_gap_steps = [0.001, rfba_gap_end]
        if extrapol_ratio is not None:
            cfg.model.policy.autoregressive_model_params.ucgmts_config.extrapol_ratio = extrapol_ratio
    
    # Set seed
    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    with open_dict(cfg):
        cfg.output_dir = output_dir
    
    # Build workspace and load weights
    cls = hydra.utils.get_class(cfg.model._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload_new(payload, exclude_keys=None, include_keys=None, strict=False)
    
    # Re-copy encoder parameters if needed
    if hasattr(workspace.model, 'model') and hasattr(workspace.model.model, 'copy_encoder_parameters'):
        workspace.model.model.copy_encoder_parameters()
        print("Re-copied encoder parameters to local causal encoder blocks")
    
    # Select policy
    if no_ema:
        with open_dict(cfg):
            cfg.training.use_ema = False
    if cfg.training.use_ema and workspace.ema_model is not None:
        print("Using EMA policy")
        policy = workspace.ema_model
    else:
        print("Using regular policy")
        policy = workspace.model
    
    # Build dataset
    print("\nBuilding dataset...")
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    val_dataset = dataset.get_validation_dataset()
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)
    
    policy.to(device)
    policy.eval()
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.val_dataloader.batch_size,
        num_workers=cfg.val_dataloader.num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=False,
    )
    
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Validation batches: {len(val_dataloader)}")
    
    # Run evaluation
    eval_log = {}
    
    # 1. Action L2 Distance
    print("\n" + "=" * 60)
    print("Computing Action L2 Distance...")
    print("=" * 60)
    with torch.no_grad():
        act_log = test_action_l2(
            cfg, policy, val_dataloader, 0, output_dir, device
        )
        eval_log.update(act_log)
    
    # 2. End-Effector Trajectory Error
    print("\n" + "=" * 60)
    print("Computing EEF Trajectory Error...")
    print("=" * 60)
    with torch.no_grad():
        try:
            eef_log = test_eef_trajectory_error(
                cfg, policy, val_dataloader, 0, output_dir, device
            )
            eval_log.update(eef_log)
        except Exception as e:
            print(f"Warning: trajectory error failed: {e}")
    
    # 3. FVD (optional)
    if fvd:
        print("\n" + "=" * 60)
        print("Computing FVD...")
        print("=" * 60)
        with torch.no_grad():
            try:
                fvd_log = test_video_fvd(
                    cfg, policy, val_dataloader, 0, output_dir, device
                )
                eval_log.update(fvd_log)
            except Exception as e:
                print(f"Warning: FVD computation failed: {e}")
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    json_log = {}
    for key, value in sorted(eval_log.items()):
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
            json_log[key] = value
        else:
            # wandb Video objects etc.
            json_log[key] = str(value)
    
    # Save results
    ckpt_name = os.path.basename(checkpoint)
    out_path = os.path.join(output_dir, f"eval_log_{ckpt_name}.json")
    with open(out_path, "w") as f:
        json.dump(json_log, f, indent=2, sort_keys=True)
    print(f"\nResults saved to: {out_path}")
    
    print("=" * 60)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
