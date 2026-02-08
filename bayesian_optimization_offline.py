#!/usr/bin/env python3
"""
Offline Bayesian Optimization for real-world UMI datasets (no env_runner needed).

Evaluates UCGM parameters using action L2 distance and trajectory error
on the validation set directly (in-process, no subprocess).

Usage:
    python bayesian_optimization_offline.py \
        --checkpoint checkpoints/dish_washing_dgm_default/checkpoints/latest.ckpt \
        --max_trials 20 \
        --device cuda:0
"""

import os
import sys
import json
import time
import copy
import random
import hydra
import torch
import dill
import numpy as np
from typing import Dict, Any, Optional, Tuple
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from unified_video_action.optimization.bayesian_optimizer import DGMBayesianOptimizer
from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.eval.eval import (
    test_action_l2,
    test_eef_trajectory_error,
    test_video_fvd,
)
from unified_video_action.utils.data_utils import resize_image


def load_model_and_data(checkpoint_path: str, device: str = "cuda:0"):
    """
    Load model from checkpoint and build validation dataloader.
    
    Returns:
        (policy, cfg, val_dataloader)
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    
    # Build workspace and load weights
    cls = hydra.utils.get_class(cfg.model._target_)
    workspace = cls(cfg, output_dir="./bayesian_optimization_tmp")
    workspace.load_payload_new(payload, exclude_keys=None, include_keys=None, strict=False)
    
    # Re-copy encoder parameters if needed
    if hasattr(workspace.model, 'model') and hasattr(workspace.model.model, 'copy_encoder_parameters'):
        workspace.model.model.copy_encoder_parameters()
    
    # Use EMA model if available
    if cfg.training.use_ema and workspace.ema_model is not None:
        policy = workspace.ema_model
        print("Using EMA policy")
    else:
        policy = workspace.model
        print("Using regular policy")
    
    # Build dataset and dataloader
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
    
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Validation dataloader batches: {len(val_dataloader)}")
    
    return policy, cfg, val_dataloader, workspace


def apply_params_to_config(cfg, params: Dict[str, Any]):
    """Apply UCGM parameters to the config for inference."""
    with open_dict(cfg.model.policy.autoregressive_model_params):
        if "ucgmts_config" not in cfg.model.policy.autoregressive_model_params:
            cfg.model.policy.autoregressive_model_params.ucgmts_config = OmegaConf.create({})
        
        cfg.model.policy.autoregressive_model_params.use_ucgm = True
        cfg.model.policy.autoregressive_model_params.num_sampling_steps = params['num_sampling_steps']
        cfg.model.policy.autoregressive_model_params.cfg = params['cfg']
        cfg.model.policy.autoregressive_model_params.temperature = params['temperature']
        cfg.model.policy.autoregressive_model_params.window_size = params['window_size']
        cfg.model.policy.autoregressive_model_params.lambda_local = params['lambda_local']
        
        cfg.model.policy.autoregressive_model_params.ucgmts_config.transport_type = params['ucgmts_config']['transport_type']
        cfg.model.policy.autoregressive_model_params.ucgmts_config.consistc_ratio = params['ucgmts_config']['consistc_ratio']
        cfg.model.policy.autoregressive_model_params.ucgmts_config.scaled_cbl_eps = params['ucgmts_config']['scaled_cbl_eps']
        cfg.model.policy.autoregressive_model_params.ucgmts_config.ema_decay_rate = params['ucgmts_config']['ema_decay_rate']
        cfg.model.policy.autoregressive_model_params.ucgmts_config.rfba_gap_steps = params['ucgmts_config']['rfba_gap_steps']
        cfg.model.policy.autoregressive_model_params.ucgmts_config.extrapol_ratio = params['ucgmts_config']['extrapol_ratio']
    
    return cfg


def evaluate_with_params(policy, cfg, val_dataloader, params: Dict[str, Any], 
                         device: str = "cuda:0", max_batches: int = None) -> Dict[str, float]:
    """
    Evaluate model with given UCGM parameters on validation set.
    
    Args:
        policy: Model policy
        cfg: Config (will be modified in-place)
        val_dataloader: Validation dataloader
        params: UCGM parameters to test
        device: Device
        max_batches: Max batches to evaluate (None = all)
    
    Returns:
        Dict with evaluation metrics
    """
    # Apply params to config
    cfg = apply_params_to_config(cfg, params)
    
    # Also update model's internal parameters if accessible
    if hasattr(policy, 'autoregressive_model_params'):
        arp = policy.autoregressive_model_params
        arp.use_ucgm = True
        arp.num_sampling_steps = params['num_sampling_steps']
        arp.cfg = params['cfg']
        arp.temperature = params['temperature']
        arp.window_size = params['window_size']
        arp.lambda_local = params['lambda_local']
        if hasattr(arp, 'ucgmts_config'):
            arp.ucgmts_config.consistc_ratio = params['ucgmts_config']['consistc_ratio']
            arp.ucgmts_config.rfba_gap_steps = params['ucgmts_config']['rfba_gap_steps']
            arp.ucgmts_config.extrapol_ratio = params['ucgmts_config']['extrapol_ratio']
    
    # Update diffactloss if present
    actual_model = getattr(policy, 'model', policy)
    if hasattr(actual_model, 'diffactloss'):
        actual_model.diffactloss.num_sampling_steps = params['num_sampling_steps']
        if hasattr(actual_model.diffactloss, 'ucgmts'):
            ucgmts = actual_model.diffactloss.ucgmts
            ucgmts.consistc_ratio = params['ucgmts_config']['consistc_ratio']
            ucgmts.rfba_gap_steps = params['ucgmts_config']['rfba_gap_steps']
            ucgmts.extrapol_ratio = params['ucgmts_config']['extrapol_ratio']
    
    results = {}
    
    # Run action L2 evaluation
    with torch.no_grad():
        act_log = test_action_l2(
            cfg, policy, val_dataloader, 0, "./bayesian_optimization_tmp", device
        )
        results.update(act_log)
        
        # Run trajectory error evaluation
        try:
            eef_log = test_eef_trajectory_error(
                cfg, policy, val_dataloader, 0, "./bayesian_optimization_tmp", device
            )
            results.update(eef_log)
        except Exception as e:
            print(f"  Warning: trajectory error evaluation failed: {e}")
    
    return results


def run_offline_bayesian_optimization(
    checkpoint_path: str,
    output_dir: str = "bayesian_optimization_results",
    max_trials: int = 20,
    device: str = "cuda:0",
    optimization_mode: str = "speed_priority",
    metric: str = "all_combined",
    max_batches: int = None,
):
    """
    Run standalone Bayesian optimization on a trained checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save results
        max_trials: Number of optimization trials
        device: CUDA device
        optimization_mode: "speed_priority", "balanced", or "performance_priority"
        metric: Metric to optimize:
            - "action_l2": only action L2 distance
            - "trajectory_error": only trajectory error (avg of both arms)
            - "combined": weighted 0.6*action_l2 + 0.4*trajectory_error
            - "all_combined": all metrics averaged (action_l2, traj_error_r0, traj_error_r1, final_dist_r0, final_dist_r1)
        max_batches: Max validation batches per evaluation (None = all)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("OFFLINE BAYESIAN OPTIMIZATION")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Trials: {max_trials}")
    print(f"Mode: {optimization_mode}")
    print(f"Metric: {metric}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load model and data once
    policy, cfg, val_dataloader, workspace = load_model_and_data(checkpoint_path, device)
    
    # Save original config for restoration between trials
    original_cfg = copy.deepcopy(cfg)
    
    # Track all results
    all_trial_results = []
    
    def objective_function(params: Dict[str, Any]) -> float:
        """Objective function for Bayesian optimization.
        
        NOTE: The optimizer MAXIMIZES the score, but all our metrics are
        "lower is better" (L2 error, trajectory error, etc.).
        So we negate: score = -metric_value.
        A score of -0.021 means the actual metric value is 0.021.
        """
        trial_num = len(all_trial_results) + 1
        print(f"\n{'='*50}")
        print(f"Trial {trial_num}/{max_trials}")
        print(f"{'='*50}")
        print(f"Params: num_sampling_steps={params['num_sampling_steps']}, "
              f"temperature={params['temperature']:.3f}, "
              f"cfg={params['cfg']:.3f}, "
              f"consistc_ratio={params['consistc_ratio']:.3f}, "
              f"window_size={params['window_size']}, "
              f"lambda_local={params['lambda_local']:.3f}")
        
        try:
            start_time = time.time()
            
            # Evaluate
            results = evaluate_with_params(
                policy, cfg, val_dataloader, params, device, max_batches
            )
            
            elapsed = time.time() - start_time
            
            # Print all available metrics
            print(f"\n  --- All Metrics ---")
            for k, v in sorted(results.items()):
                if isinstance(v, (int, float)):
                    print(f"    {k}: {v:.6f}")
            
            # Compute score based on metric
            # NOTE: lower error = better, but optimizer MAXIMIZES
            # So score = -(metric), higher score = lower error = better
            if metric == "action_l2":
                val = results.get("val_action_l2_distances", None)
                if val is None:
                    print("  No action L2 metric found!")
                    score = -1000.0
                else:
                    score = -val
                    print(f"\n  >> Optimizing: Action L2 = {val:.6f} (lower is better)")
            
            elif metric == "trajectory_error":
                val = results.get("val_eef_trajectory_error", None)
                if val is None:
                    print("  No trajectory error metric found!")
                    score = -1000.0
                else:
                    score = -val
                    print(f"\n  >> Optimizing: Trajectory Error = {val:.6f} (lower is better)")
            
            elif metric == "combined":
                l2 = results.get("val_action_l2_distances", None)
                traj = results.get("val_eef_trajectory_error", None)
                if l2 is None:
                    score = -1000.0
                elif traj is not None:
                    combined_val = 0.6 * l2 + 0.4 * traj
                    score = -combined_val
                    print(f"\n  >> Optimizing: 0.6*L2({l2:.6f}) + 0.4*TrajErr({traj:.6f}) = {combined_val:.6f}")
                else:
                    score = -l2
                    print(f"\n  >> Optimizing: Action L2 = {l2:.6f} (no traj error available)")
            
            elif metric == "all_combined":
                # Collect all available error metrics and average them
                metric_values = []
                metric_names = []
                
                # Action L2
                l2 = results.get("val_action_l2_distances", None)
                if l2 is not None:
                    metric_values.append(l2)
                    metric_names.append(f"action_l2={l2:.6f}")
                
                # Per-arm trajectory errors (bimanual)
                traj_r0 = results.get("val_eef_trajectory_error_robot0", None)
                traj_r1 = results.get("val_eef_trajectory_error_robot1", None)
                if traj_r0 is not None:
                    metric_values.append(traj_r0)
                    metric_names.append(f"traj_err_r0={traj_r0:.6f}")
                if traj_r1 is not None:
                    metric_values.append(traj_r1)
                    metric_names.append(f"traj_err_r1={traj_r1:.6f}")
                
                # Per-arm final state distances (bimanual)
                final_r0 = results.get("val_final_state_distance_robot0", None)
                final_r1 = results.get("val_final_state_distance_robot1", None)
                if final_r0 is not None:
                    metric_values.append(final_r0)
                    metric_names.append(f"final_dist_r0={final_r0:.6f}")
                if final_r1 is not None:
                    metric_values.append(final_r1)
                    metric_names.append(f"final_dist_r1={final_r1:.6f}")
                
                # Fallback: use combined trajectory error if per-arm not available
                if traj_r0 is None and traj_r1 is None:
                    traj = results.get("val_eef_trajectory_error", None)
                    if traj is not None:
                        metric_values.append(traj)
                        metric_names.append(f"traj_err={traj:.6f}")
                    final = results.get("val_final_state_distance", None)
                    if final is not None:
                        metric_values.append(final)
                        metric_names.append(f"final_dist={final:.6f}")
                
                if len(metric_values) == 0:
                    print("  No metrics found!")
                    score = -1000.0
                else:
                    avg_val = np.mean(metric_values)
                    score = -avg_val
                    print(f"\n  >> Optimizing: avg of {len(metric_values)} metrics = {avg_val:.6f} (lower is better)")
                    for name in metric_names:
                        print(f"       {name}")
            
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            print(f"  Time: {elapsed:.1f}s")
            
            # Record trial
            trial_result = {
                "trial": trial_num,
                "params": {k: v for k, v in params.items() if k != 'ucgmts_config'},
                "ucgmts_config": params.get('ucgmts_config', {}),
                "results": results,
                "score": score,
                "actual_metric_value": -score if score > -999 else None,
                "elapsed_seconds": elapsed,
            }
            all_trial_results.append(trial_result)
            
            return score
            
        except Exception as e:
            print(f"  Evaluation error: {e}")
            import traceback
            traceback.print_exc()
            all_trial_results.append({
                "trial": trial_num,
                "params": {k: v for k, v in params.items() if k != 'ucgmts_config'},
                "error": str(e),
                "score": -1000.0,
            })
            return -1000.0
    
    # Run optimization
    optimizer = DGMBayesianOptimizer(max_trials=max_trials, optimization_mode=optimization_mode)
    
    print(f"\nStarting optimization with {max_trials} trials...")
    best_params, best_score = optimizer.optimize(objective_function)
    
    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best score: {best_score:.6f} (optimizer internal, higher=better)")
    print(f"Best metric value: {-best_score:.6f} (actual error, lower=better)")
    print(f"Best params:")
    for k, v in best_params.items():
        if k != 'ucgmts_config':
            print(f"  {k}: {v}")
    print(f"  ucgmts_config:")
    for k, v in best_params.get('ucgmts_config', {}).items():
        print(f"    {k}: {v}")
    
    # Save results
    final_results = {
        "checkpoint": checkpoint_path,
        "optimization_mode": optimization_mode,
        "metric": metric,
        "max_trials": max_trials,
        "best_score": best_score,
        "best_params": best_params,
        "all_trials": all_trial_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    results_file = os.path.join(output_dir, "bayesian_optimization_results.json")
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
    
    # Also save a simple best params file for easy loading
    best_params_file = os.path.join(output_dir, "best_params.json")
    with open(best_params_file, 'w') as f:
        json.dump(best_params, f, indent=2, default=str)
    print(f"Best params saved to: {best_params_file}")
    
    # Auto-save optimized checkpoint with best params baked in
    from apply_best_params import apply_params_to_checkpoint
    optimized_ckpt_path = os.path.join(output_dir, "optimized.ckpt")
    print(f"\n{'='*60}")
    print("AUTO-SAVING OPTIMIZED CHECKPOINT")
    print(f"{'='*60}")
    try:
        apply_params_to_checkpoint(checkpoint_path, best_params, optimized_ckpt_path)
        print(f"\nOptimized checkpoint ready! Evaluate directly:")
        print(f"  python eval_offline.py --checkpoint {optimized_ckpt_path} --output_dir eval_results/ --use_ucgm")
    except Exception as e:
        print(f"Warning: Failed to save optimized checkpoint: {e}")
        print(f"You can manually run:")
        print(f"  python apply_best_params.py --checkpoint {checkpoint_path} --best_params {best_params_file}")
    
    return best_params, best_score


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Offline Bayesian optimization for UMI datasets")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--output_dir", default="bayesian_optimization_results", help="Output directory")
    parser.add_argument("--max_trials", type=int, default=20, help="Max optimization trials")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--optimization_mode", default="speed_priority", 
                        choices=["speed_priority", "balanced", "performance_priority"],
                        help="Optimization mode (default: speed_priority)")
    parser.add_argument("--metric", default="all_combined",
                        choices=["action_l2", "trajectory_error", "combined", "all_combined"],
                        help="Metric to optimize: action_l2, trajectory_error, combined (weighted), all_combined (avg all metrics including per-arm)")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Max validation batches per trial (None=all, set lower for faster trials)")
    parser.add_argument("--quick_test", action="store_true", help="Quick test: 3 trials")
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.max_trials = 3
        args.max_batches = 5
        print("Quick test mode: 3 trials, 5 batches each")
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    try:
        run_offline_bayesian_optimization(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            max_trials=args.max_trials,
            device=args.device,
            optimization_mode=args.optimization_mode,
            metric=args.metric,
            max_batches=args.max_batches,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
