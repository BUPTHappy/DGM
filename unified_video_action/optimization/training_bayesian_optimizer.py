"""
Training-integrated Bayesian Optimizer for UCGM parameters.
This module integrates Bayesian optimization into the training loop,
allowing for adaptive hyperparameter optimization during training.
"""

import os
import json
import tempfile
import subprocess
import torch
import dill
import numpy as np
from typing import Dict, Any, Optional, Tuple
from omegaconf import OmegaConf, open_dict
from unified_video_action.optimization.bayesian_optimizer import UCGMBayesianOptimizer


class TrainingBayesianOptimizer:
    """
    Bayesian optimizer integrated into the training process.
    Performs optimization at specified intervals during training.
    """
    
    def __init__(self, 
                 config: OmegaConf,
                 start_epoch: int = 100,  # Start optimization after 50% of training
                 interval: int = 10,      # Optimize every 10 epochs
                 max_trials: int = 15,    # Reduced trials for training integration
                 n_test: int = 5,         # Reduced test count for faster evaluation
                 device: str = "cuda:0",
                 output_dir: str = "./bayesian_optimization_logs"):
        """
        Initialize the training-integrated Bayesian optimizer.
        
        Args:
            config: Training configuration
            start_epoch: Epoch to start Bayesian optimization
            interval: Interval between optimization runs
            max_trials: Maximum trials per optimization run
            n_test: Number of test runs per evaluation
            device: Device for evaluation
            output_dir: Output directory for optimization logs
        """
        self.config = config
        self.start_epoch = start_epoch
        self.interval = interval
        self.max_trials = max_trials
        self.n_test = n_test
        self.device = device
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Track optimization history
        self.optimization_history = []
        self.best_params = None
        self.best_score = -float('inf')
        
        # Initialize optimizer
        self.optimizer = UCGMBayesianOptimizer(max_trials=max_trials)
        
        print(f"TrainingBayesianOptimizer initialized:")
        print(f"  Start epoch: {start_epoch}")
        print(f"  Interval: {interval}")
        print(f"  Max trials: {max_trials}")
        print(f"  Output dir: {output_dir}")
    
    def should_optimize(self, current_epoch: int) -> bool:
        """Check if optimization should be performed at current epoch."""
        return (current_epoch >= self.start_epoch and 
                (current_epoch - self.start_epoch) % self.interval == 0)
    
    def evaluate_model_with_params(self, 
                                 params: Dict[str, Any], 
                                 checkpoint_path: str) -> float:
        """
        Evaluate model with given parameters.
        This is a simplified version of the evaluation function optimized for training integration.
        """
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
                
                # Local attention parameters
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
                cfg.task.env_runner.n_test = self.n_test
            else:
                cfg.task.env_runner.n_test = min(self.n_test * 5, 50)
            
            # Create temp output directory
            temp_output_dir = tempfile.mkdtemp(prefix="bayesian_eval_")
            
            # Run evaluation
            cmd = [
                "python", "eval_sim.py",
                "--checkpoint", checkpoint_path,
                "--output_dir", temp_output_dir,
                "--device", self.device,
                "--use_ucgm",
                "--num_sampling_steps", str(params['num_sampling_steps']),
                "--stochasticity_rate", str(params['consistc_ratio']),
                "--window_size", str(params['window_size']),
                "--lambda_local", str(params['lambda_local'])
            ]
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = self.device.split(":")[-1] if ":" in self.device else "0"
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=180  # Reduced timeout for training integration
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
    
    def run_optimization(self, 
                        checkpoint_path: str, 
                        current_epoch: int) -> Optional[Dict[str, Any]]:
        """
        Run Bayesian optimization for current epoch.
        
        Args:
            checkpoint_path: Path to current checkpoint
            current_epoch: Current training epoch
            
        Returns:
            Best parameters found, or None if optimization failed
        """
        print(f"\n{'='*60}")
        print(f"Starting Bayesian optimization at epoch {current_epoch}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*60}")
        
        def objective_function(params):
            return self.evaluate_model_with_params(params, checkpoint_path)
        
        try:
            # Run optimization
            best_params, best_score = self.optimizer.optimize(objective_function)
            
            if best_params is not None and best_score > -1000.0:
                print(f"Optimization completed!")
                print(f"Best score: {best_score:.4f}")
                print(f"Best params: {best_params}")
                
                # Update best parameters if this is the best so far
                if best_score > self.best_score:
                    self.best_score = best_score
                    self.best_params = best_params
                    print(f"New best parameters found! Score: {best_score:.4f}")
                
                # Save optimization results
                optimization_result = {
                    'epoch': current_epoch,
                    'best_params': best_params,
                    'best_score': best_score,
                    'checkpoint_path': checkpoint_path,
                    'timestamp': torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True))
                }
                
                self.optimization_history.append(optimization_result)
                
                # Save to file
                results_file = os.path.join(self.output_dir, f"optimization_epoch_{current_epoch}.json")
                with open(results_file, 'w') as f:
                    json.dump(optimization_result, f, indent=2)
                
                # Save complete history
                history_file = os.path.join(self.output_dir, "optimization_history.json")
                with open(history_file, 'w') as f:
                    json.dump(self.optimization_history, f, indent=2)
                
                return best_params
            else:
                print("Optimization failed or no valid parameters found")
                return None
                
        except Exception as e:
            print(f"Optimization error: {e}")
            return None
    
    def apply_best_params_to_model(self, 
                                 model, 
                                 params: Dict[str, Any]) -> bool:
        """
        Apply the best parameters to the model.
        
        Args:
            model: The model to update
            params: Parameters to apply
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Applying parameters to model: {params}")
            
            # Update model parameters
            if hasattr(model, 'autoregressive_model_params'):
                autoregressive_params = model.autoregressive_model_params
                
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
                
                print("Parameters successfully applied to model")
                return True
            else:
                print("Model does not have autoregressive_model_params attribute")
                return False
                
        except Exception as e:
            print(f"Error applying parameters to model: {e}")
            return False
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization history."""
        return {
            'total_optimizations': len(self.optimization_history),
            'best_score': self.best_score,
            'best_params': self.best_params,
            'optimization_history': self.optimization_history
        }
