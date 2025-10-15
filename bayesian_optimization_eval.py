#!/usr/bin/env python3

import os
import sys
import torch
import dill
import json
import tempfile
import subprocess
import numpy as np
from typing import Dict, Any
from omegaconf import OmegaConf, open_dict
from unified_video_action.optimization.bayesian_optimizer import UCGMBayesianOptimizer


def evaluate_model_with_params(params: Dict[str, Any], 
                              checkpoint_path: str,
                              output_dir: str,
                              device: str = "cuda:0",
                              n_test: int = 10) -> float:
    try:
        print(f"Evaluating params: {params}")
        
        # Load checkpoint and config
        payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill, weights_only=False)
        cfg = payload["cfg"]
        
        # Update config parameters
        with open_dict(cfg.model.policy.autoregressive_model_params):
            if "ucgmts_config" not in cfg.model.policy.autoregressive_model_params:
                cfg.model.policy.autoregressive_model_params.ucgmts_config = OmegaConf.create({})
            
            cfg.model.policy.autoregressive_model_params.use_ucgm = True
            cfg.model.policy.autoregressive_model_params.num_sampling_steps = params['num_sampling_steps']
            cfg.model.policy.autoregressive_model_params.cfg = params['cfg']
            cfg.model.policy.autoregressive_model_params.temperature = params['temperature']
            
            # 新增的local attention参数
            cfg.model.policy.autoregressive_model_params.window_size = params['window_size']
            cfg.model.policy.autoregressive_model_params.lambda_local = params['lambda_local']
            
            cfg.model.policy.autoregressive_model_params.ucgmts_config.transport_type = params['ucgmts_config']['transport_type']
            cfg.model.policy.autoregressive_model_params.ucgmts_config.consistc_ratio = params['ucgmts_config']['consistc_ratio']
            cfg.model.policy.autoregressive_model_params.ucgmts_config.scaled_cbl_eps = params['ucgmts_config']['scaled_cbl_eps']
            cfg.model.policy.autoregressive_model_params.ucgmts_config.ema_decay_rate = params['ucgmts_config']['ema_decay_rate']
            cfg.model.policy.autoregressive_model_params.ucgmts_config.rfba_gap_steps = params['ucgmts_config']['rfba_gap_steps']
            cfg.model.policy.autoregressive_model_params.ucgmts_config.extrapol_ratio = params['ucgmts_config']['extrapol_ratio']
        
        # Set test count
        if "libero" in cfg.task.name:
            cfg.task.env_runner.n_test = n_test
        else:
            cfg.task.env_runner.n_test = min(n_test * 5, 50)
        
        # Create temp output directory
        temp_output_dir = tempfile.mkdtemp(prefix="bayesian_eval_")
        
        # Run evaluation
        cmd = [
            "python", "eval_sim.py",
            "--checkpoint", checkpoint_path,
            "--output_dir", temp_output_dir,
            "--device", device,
            "--use_ucgm",
            "--num_sampling_steps", str(params['num_sampling_steps']),
            "--stochasticity_rate", str(params['consistc_ratio']),
            "--window_size", str(params['window_size']),
            "--lambda_local", str(params['lambda_local'])
        ]
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1] if ":" in device else "0"
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=300
        )
        
        if result.returncode != 0:
            print(f"Evaluation failed: {result.stderr}")
            return -1000.0
        
        # Parse results
        eval_log_path = os.path.join(temp_output_dir, f'eval_log_{os.path.basename(checkpoint_path)}.json')
        
        if not os.path.exists(eval_log_path):
            print(f"Eval log not found: {eval_log_path}")
            return -1000.0
        
        with open(eval_log_path, 'r') as f:
            eval_results = json.load(f)
        
        # Extract score
        if "test_mean_score" in eval_results:
            score = eval_results["test_mean_score"]
        elif "test/pusht_mean_score" in eval_results:
            score = eval_results["test/pusht_mean_score"]
        elif "test/libero10_mean_score" in eval_results:
            score = eval_results["test/libero10_mean_score"]
        else:
            score_keys = [k for k in eval_results.keys() if "score" in k.lower()]
            if score_keys:
                score = eval_results[score_keys[0]]
            else:
                print(f"No score found, available keys: {list(eval_results.keys())}")
                return -1000.0
        
        print(f"Score: {score}")
        
        # Cleanup
        subprocess.run(["rm", "-rf", temp_output_dir], check=False)
        
        return float(score)
        
    except subprocess.TimeoutExpired:
        print("Evaluation timeout")
        return -1000.0
    except Exception as e:
        print(f"Evaluation error: {e}")
        return -1000.0


def run_bayesian_optimization(checkpoint_path: str,
                             output_dir: str,
                             max_trials: int = 30,
                             device: str = "cuda:0",
                             n_test: int = 10):
    os.makedirs(output_dir, exist_ok=True)
    
    def objective_function(params):
        return evaluate_model_with_params(
            params=params,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            device=device,
            n_test=n_test
        )
    
    optimizer = UCGMBayesianOptimizer(max_trials=max_trials) #创建优化器
    
    print(f"Starting Bayesian optimization: {max_trials} trials")
    print(f"Checkpoint: {checkpoint_path}")
    
    best_params, best_score = optimizer.optimize(objective_function) #优化参数
    
    results_file = os.path.join(output_dir, "bayesian_optimization_results.json")
    optimizer.save_results(results_file)
    
    print(f"Optimization completed!")
    print(f"Best score: {best_score:.4f}")
    print(f"Best params: {best_params}")
    print(f"Results saved to: {results_file}")
    
    return best_params, best_score


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bayesian optimization for UCGM parameters")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--output_dir", default="bayesian_optimization_results", help="Output directory")
    parser.add_argument("--max_trials", type=int, default=20, help="Max optimization trials")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--n_test", type=int, default=5, help="Test count per evaluation")
    parser.add_argument("--quick_test", action="store_true", help="Quick test mode")
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.max_trials = 5
        args.n_test = 2
        print("Quick test mode: max_trials=5, n_test=2")
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_dir}")
    print(f"Trials: {args.max_trials}")
    print(f"Tests: {args.n_test}")
    
    try:
        run_bayesian_optimization(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            max_trials=args.max_trials,
            device=args.device,
            n_test=args.n_test
        )
    except KeyboardInterrupt:
        print("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)