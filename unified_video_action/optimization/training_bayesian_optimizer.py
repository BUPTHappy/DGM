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
from unified_video_action.optimization.adaptive_bayesian_optimizer import AdaptiveUCGMBayesianOptimizer, OptimizationMode


class TrainingBayesianOptimizer:
    """
    Bayesian optimizer integrated into the training process.
    Performs optimization at specified intervals during training.
    """
    
    def __init__(self, 
                 config: OmegaConf,
                 start_epoch: int = 150,  # Start optimization at 3/4 of training
                 interval: int = 10,       # Optimize every 10 epochs
                 max_trials: int = 8,      # Normal trials for mid-training
                 final_trials: int = 15,   # Final optimization trials
                 n_test: int = 3,          # Normal test count
                 final_n_test: int = 5,    # Final optimization test count
                 device: str = "cuda:0",
                 output_dir: str = "./bayesian_optimization_logs",
                 use_best_checkpoint_for_final: bool = True,
                 optimization_mode: str = "balanced"):
        """
        Initialize the training-integrated Bayesian optimizer.
        
        Args:
            config: Training configuration
            start_epoch: Epoch to start Bayesian optimization
            interval: Interval between optimization runs
            max_trials: Maximum trials per optimization run (normal phase)
            final_trials: Maximum trials for final optimization
            n_test: Number of test runs per evaluation (normal phase)
            final_n_test: Number of test runs for final optimization
            device: Device for evaluation
            output_dir: Output directory for optimization logs
            use_best_checkpoint_for_final: Whether to use best checkpoint for final optimization
            optimization_mode: Optimization mode ("performance", "speed", "balanced")
        """
        self.config = config
        self.start_epoch = start_epoch
        self.interval = interval
        self.max_trials = max_trials
        self.final_trials = final_trials
        self.n_test = n_test
        self.final_n_test = final_n_test
        self.device = device
        self.output_dir = output_dir
        self.use_best_checkpoint_for_final = use_best_checkpoint_for_final
        self.optimization_mode = optimization_mode
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Track optimization history
        self.optimization_history = []
        self.best_params = None
        self.best_score = -float('inf')
        
        # Initialize adaptive optimizer
        optimization_mode_enum = OptimizationMode(optimization_mode)
        self.optimizer = AdaptiveUCGMBayesianOptimizer(
            max_trials=max_trials, 
            optimization_mode=optimization_mode_enum
        )
        
        print(f"TrainingBayesianOptimizer initialized:")
        print(f"  Start epoch: {start_epoch}")
        print(f"  Interval: {interval}")
        print(f"  Max trials (normal): {max_trials}")
        print(f"  Max trials (final): {final_trials}")
        print(f"  N test (normal): {n_test}")
        print(f"  N test (final): {final_n_test}")
        print(f"  Optimization mode: {optimization_mode}")
        print(f"  Use best checkpoint for final: {use_best_checkpoint_for_final}")
        print(f"  Output dir: {output_dir}")
    
    def should_optimize(self, current_epoch: int) -> bool:
        """Check if optimization should be performed at current epoch."""
        return (current_epoch >= self.start_epoch and 
                (current_epoch - self.start_epoch) % self.interval == 0)
    
    def find_best_checkpoint(self, checkpoints_dir: str) -> Optional[str]:
        """
        Find the best checkpoint based on test_mean_score.
        
        Args:
            checkpoints_dir: Directory containing checkpoints
            
        Returns:
            Path to the best checkpoint, or None if not found
        """
        if not os.path.exists(checkpoints_dir):
            return None
            
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt')]
        if not checkpoint_files:
            return None
        
        best_score = -float('inf')
        best_checkpoint = None
        
        for checkpoint_file in checkpoint_files:
            # Skip latest.ckpt as it's not performance-based
            if checkpoint_file == 'latest.ckpt':
                continue
                
            # Extract score from filename if possible
            if 'test_mean_score=' in checkpoint_file:
                try:
                    # Extract score from filename like "epoch=XXXX-test_mean_score=X.XXX.ckpt"
                    score_str = checkpoint_file.split('test_mean_score=')[1].split('.ckpt')[0]
                    score = float(score_str)
                    
                    if score > best_score:
                        best_score = score
                        best_checkpoint = os.path.join(checkpoints_dir, checkpoint_file)
                except (ValueError, IndexError):
                    continue
        
        if best_checkpoint:
            print(f"Found best checkpoint: {best_checkpoint} (score: {best_score:.3f})")
        else:
            print(f"No best checkpoint found in {checkpoints_dir}")
            
        return best_checkpoint
    
    def evaluate_model_with_params(self, 
                                 params: Dict[str, Any], 
                                 checkpoint_path: str) -> float:
        """
        Evaluate model with given parameters.
        This is a simplified version of the evaluation function optimized for training integration.
        """
        try:
            print(f"Evaluating params: {params}")
            print(f"Using checkpoint: {checkpoint_path}")
            print(f"Using device: {self.device}")
            
            # Clear CUDA cache before evaluation
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Wait a bit to ensure checkpoint is fully written
            import time
            time.sleep(5)  # Increased wait time
            
            # Check if checkpoint file is valid before loading
            if not self._is_checkpoint_valid(checkpoint_path):
                print(f"Checkpoint file is invalid or corrupted: {checkpoint_path}")
                return -1000.0
            
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
                
                # UCGM Training parameters
                cfg.model.policy.autoregressive_model_params.ucgmts_config.transport_type = params['ucgmts_config']['transport_type']
                cfg.model.policy.autoregressive_model_params.ucgmts_config.consistc_ratio = params['ucgmts_config']['consistc_ratio']
                cfg.model.policy.autoregressive_model_params.ucgmts_config.scaled_cbl_eps = params['ucgmts_config']['scaled_cbl_eps']
                cfg.model.policy.autoregressive_model_params.ucgmts_config.ema_decay_rate = params['ucgmts_config']['ema_decay_rate']
                cfg.model.policy.autoregressive_model_params.ucgmts_config.rfba_gap_steps = params['ucgmts_config']['rfba_gap_steps']
                cfg.model.policy.autoregressive_model_params.ucgmts_config.extrapol_ratio = params['ucgmts_config']['extrapol_ratio']
                cfg.model.policy.autoregressive_model_params.ucgmts_config.lab_drop_ratio = params['ucgmts_config']['lab_drop_ratio']
                cfg.model.policy.autoregressive_model_params.ucgmts_config.enhanced_ratio = params['ucgmts_config']['enhanced_ratio']
                cfg.model.policy.autoregressive_model_params.ucgmts_config.wt_cosine_loss = params['ucgmts_config']['wt_cosine_loss']
                cfg.model.policy.autoregressive_model_params.ucgmts_config.weight_function = params['ucgmts_config']['weight_function']
                cfg.model.policy.autoregressive_model_params.ucgmts_config.time_dist_ctrl = params['ucgmts_config']['time_dist_ctrl']
            
            # Set test count - reduce for training integration to avoid timeouts
            if "libero" in cfg.task.name:
                cfg.task.env_runner.n_test = min(self.n_test, 3)  # Cap at 3 for libero
            else:
                cfg.task.env_runner.n_test = min(self.n_test, 5)  # Cap at 5 for pusht
            
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
                timeout=600  # Increased timeout to 10 minutes for evaluation
            )
            
            if result.returncode != 0:
                print(f"Evaluation failed with return code {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                print(f"Command: {' '.join(cmd)}")
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
            print("Evaluation timeout - this might indicate the evaluation is taking too long")
            print("Consider reducing n_test or increasing timeout")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return -1000.0
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e).lower():
                print(f"CUDA error during evaluation: {e}")
                print("Clearing CUDA cache and retrying...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                return -1000.0
            else:
                print(f"Runtime error during evaluation: {e}")
                return -1000.0
        except Exception as e:
            print(f"Evaluation error: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return -1000.0
    
    def run_optimization(self, 
                        checkpoint_path: str, 
                        current_epoch: int,
                        checkpoints_dir: str = None) -> Optional[Dict[str, Any]]:
        """
        Run Bayesian optimization for current epoch.
        
        Args:
            checkpoint_path: Path to current checkpoint
            current_epoch: Current training epoch
            checkpoints_dir: Directory containing checkpoints (for finding best checkpoint)
            
        Returns:
            Best parameters found, or None if optimization failed
        """
        print(f"\n{'='*60}")
        print(f"Starting Bayesian optimization at epoch {current_epoch}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*60}")
        
        # For final optimization, use best checkpoint if available
        total_epochs = self.config.training.num_epochs
        is_final_optimization = current_epoch >= total_epochs - 5
        
        if is_final_optimization and self.use_best_checkpoint_for_final and checkpoints_dir:
            best_checkpoint = self.find_best_checkpoint(checkpoints_dir)
            if best_checkpoint and best_checkpoint != checkpoint_path:
                print(f"Using best checkpoint for final optimization: {best_checkpoint}")
                checkpoint_path = best_checkpoint
        
        # Adjust optimization parameters based on epoch (progressive strategy)
        total_epochs = self.config.training.num_epochs
        progress = current_epoch / total_epochs
        
        if current_epoch >= total_epochs - 5:
            # Final optimization phase - use maximum resources
            max_trials = self.final_trials
            n_test = self.final_n_test
            print(f"Final optimization phase: max_trials={max_trials}, n_test={n_test}")
        elif progress >= 0.9:
            # Near-final phase - increase resources
            max_trials = int(self.max_trials * 1.5)
            n_test = int(self.n_test * 1.5)
            print(f"Near-final phase: max_trials={max_trials}, n_test={n_test}")
        else:
            # Normal phase - use standard resources
            max_trials = self.max_trials
            n_test = self.n_test
            print(f"Normal phase: max_trials={max_trials}, n_test={n_test}")
        
        # Create a temporary optimizer with adjusted parameters
        optimization_mode_enum = OptimizationMode(self.optimization_mode)
        temp_optimizer = AdaptiveUCGMBayesianOptimizer(
            max_trials=max_trials,
            optimization_mode=optimization_mode_enum
        )
        
        def objective_function(params):
            return self.evaluate_model_with_params(params, checkpoint_path)
        
        try:
            # First, evaluate current model performance as baseline
            print("Evaluating current model performance as baseline...")
            current_score = self.evaluate_model_with_params({}, checkpoint_path)  # Empty params = use current config
            print(f"Current model score: {current_score:.4f}")
            
            # Run optimization with temporary optimizer
            result = temp_optimizer.optimize(objective_function)
            
            # Handle case where optimization returns None or fails
            if result is None:
                print("Optimization returned None - all trials failed")
                return None
            
            # Unpack result safely
            if isinstance(result, tuple) and len(result) == 2:
                best_params, best_score = result
            else:
                print(f"Unexpected optimization result format: {result}")
                return None
            
            if best_params is not None and best_score > -1000.0:
                print(f"Optimization completed!")
                print(f"Current model score: {current_score:.4f}")
                print(f"Best optimization score: {best_score:.4f}")
                print(f"Best params: {best_params}")
                
                # Only apply parameters if optimization result is better than current model
                if best_score > current_score:
                    print(f"✅ Optimization improved performance! ({current_score:.4f} → {best_score:.4f})")
                    
                    # Update best parameters if this is the best so far
                    if best_score > self.best_score:
                        self.best_score = best_score
                        self.best_params = best_params
                        print(f"New best parameters found! Score: {best_score:.4f}")
                    
                    # Save optimization results
                    import time
                    optimization_result = {
                        'epoch': current_epoch,
                        'current_score': current_score,
                        'best_params': best_params,
                        'best_score': best_score,
                        'improvement': best_score - current_score,
                        'checkpoint_path': checkpoint_path,
                        'timestamp': time.time()
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
                    print(f"❌ Optimization did not improve performance ({current_score:.4f} ≥ {best_score:.4f})")
                    print("Keeping current model parameters")
                    
                    # Still save the optimization results for analysis
                    import time
                    optimization_result = {
                        'epoch': current_epoch,
                        'current_score': current_score,
                        'best_params': best_params,
                        'best_score': best_score,
                        'improvement': best_score - current_score,
                        'checkpoint_path': checkpoint_path,
                        'timestamp': time.time(),
                        'applied': False
                    }
                    
                    self.optimization_history.append(optimization_result)
                    
                    # Save to file
                    results_file = os.path.join(self.output_dir, f"optimization_epoch_{current_epoch}.json")
                    with open(results_file, 'w') as f:
                        json.dump(optimization_result, f, indent=2)
                    
                    return None  # Don't apply parameters
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
            
            # Print current model parameters before update
            print(f"\n{'='*50}")
            print(f"MODEL PARAMETERS BEFORE UPDATE:")
            print(f"{'='*50}")
            if hasattr(model, 'autoregressive_model_params'):
                autoregressive_params = model.autoregressive_model_params
                print(f"Current num_sampling_steps: {getattr(autoregressive_params, 'num_sampling_steps', 'N/A')}")
                print(f"Current cfg: {getattr(autoregressive_params, 'cfg', 'N/A')}")
                print(f"Current temperature: {getattr(autoregressive_params, 'temperature', 'N/A')}")
                print(f"Current window_size: {getattr(autoregressive_params, 'window_size', 'N/A')}")
                print(f"Current lambda_local: {getattr(autoregressive_params, 'lambda_local', 'N/A')}")
                print(f"Current use_ucgm: {getattr(autoregressive_params, 'use_ucgm', 'N/A')}")
                
                if hasattr(autoregressive_params, 'ucgmts_config'):
                    ucgmts_config = autoregressive_params.ucgmts_config
                    print(f"Current ucgmts_config:")
                    print(f"  transport_type: {getattr(ucgmts_config, 'transport_type', 'N/A')}")
                    print(f"  consistc_ratio: {getattr(ucgmts_config, 'consistc_ratio', 'N/A')}")
                    print(f"  scaled_cbl_eps: {getattr(ucgmts_config, 'scaled_cbl_eps', 'N/A')}")
                    print(f"  ema_decay_rate: {getattr(ucgmts_config, 'ema_decay_rate', 'N/A')}")
                    print(f"  rfba_gap_steps: {getattr(ucgmts_config, 'rfba_gap_steps', 'N/A')}")
                    print(f"  extrapol_ratio: {getattr(ucgmts_config, 'extrapol_ratio', 'N/A')}")
                
                # Also check the actual model components
                if hasattr(model, 'model') and hasattr(model.model, 'diffactloss'):
                    diffactloss = model.model.diffactloss
                    print(f"Current DiffActLoss num_sampling_steps: {getattr(diffactloss, 'num_sampling_steps', 'N/A')}")
                    if hasattr(diffactloss, 'ucgmts'):
                        ucgmts = diffactloss.ucgmts
                        print(f"Current UCGMTS parameters:")
                        print(f"  transport_type: {getattr(ucgmts, 'transport_type', 'N/A')}")
                        print(f"  consistc_ratio: {getattr(ucgmts, 'consistc_ratio', 'N/A')}")
                        print(f"  rfba_gap_steps: {getattr(ucgmts, 'rfba_gap_steps', 'N/A')}")
                        print(f"  extrapol_ratio: {getattr(ucgmts, 'extrapol_ratio', 'N/A')}")
            
            # Update model parameters
            # Check if model has autoregressive_model_params attribute
            autoregressive_params = None
            
            # Try different ways to access autoregressive_model_params
            if hasattr(model, 'autoregressive_model_params'):
                autoregressive_params = model.autoregressive_model_params
                print("Found autoregressive_model_params directly on model")
            elif hasattr(model, 'model') and hasattr(model.model, 'autoregressive_model_params'):
                # Handle case where model is wrapped (e.g., DDP wrapper)
                autoregressive_params = model.model.autoregressive_model_params
                print("Found autoregressive_model_params on model.model")
            elif hasattr(model, 'module') and hasattr(model.module, 'autoregressive_model_params'):
                # Handle case where model is wrapped with module attribute
                autoregressive_params = model.module.autoregressive_model_params
                print("Found autoregressive_model_params on model.module")
            
            # Check if we found autoregressive_model_params
            if autoregressive_params is None:
                print(f"Model type: {type(model)}")
                print(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
                
                # Check if model has a model attribute
                if hasattr(model, 'model'):
                    print(f"model.model type: {type(model.model)}")
                    print(f"model.model attributes: {[attr for attr in dir(model.model) if not attr.startswith('_')]}")
                
                # Check if model has a module attribute
                if hasattr(model, 'module'):
                    print(f"model.module type: {type(model.module)}")
                    print(f"model.module attributes: {[attr for attr in dir(model.module) if not attr.startswith('_')]}")
                
                print("Model does not have autoregressive_model_params attribute")
                return False
            
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
            
            # Also update the actual model components
            if hasattr(model, 'model') and hasattr(model.model, 'diffactloss'):
                diffactloss = model.model.diffactloss
                diffactloss.num_sampling_steps = params['num_sampling_steps']
                
                if hasattr(diffactloss, 'ucgmts'):
                    ucgmts = diffactloss.ucgmts
                    ucgmts.transport_type = params['ucgmts_config']['transport_type']
                    ucgmts.consistc_ratio = params['ucgmts_config']['consistc_ratio']
                    ucgmts.scaled_cbl_eps = params['ucgmts_config']['scaled_cbl_eps']
                    ucgmts.ema_decay_rate = params['ucgmts_config']['ema_decay_rate']
                    ucgmts.rfba_gap_steps = params['ucgmts_config']['rfba_gap_steps']
                    ucgmts.extrapol_ratio = params['ucgmts_config']['extrapol_ratio']
            
            # CRITICAL: Also update the config to ensure parameters are saved in checkpoint
            if hasattr(model, 'cfg') and model.cfg is not None:
                with open_dict(model.cfg.model.policy.autoregressive_model_params):
                    if "ucgmts_config" not in model.cfg.model.policy.autoregressive_model_params:
                        model.cfg.model.policy.autoregressive_model_params.ucgmts_config = OmegaConf.create({})
                    
                    model.cfg.model.policy.autoregressive_model_params.use_ucgm = True
                    model.cfg.model.policy.autoregressive_model_params.num_sampling_steps = params['num_sampling_steps']
                    model.cfg.model.policy.autoregressive_model_params.cfg = params['cfg']
                    model.cfg.model.policy.autoregressive_model_params.temperature = params['temperature']
                    model.cfg.model.policy.autoregressive_model_params.window_size = params['window_size']
                    model.cfg.model.policy.autoregressive_model_params.lambda_local = params['lambda_local']
                    
                    model.cfg.model.policy.autoregressive_model_params.ucgmts_config.transport_type = params['ucgmts_config']['transport_type']
                    model.cfg.model.policy.autoregressive_model_params.ucgmts_config.consistc_ratio = params['ucgmts_config']['consistc_ratio']
                    model.cfg.model.policy.autoregressive_model_params.ucgmts_config.scaled_cbl_eps = params['ucgmts_config']['scaled_cbl_eps']
                    model.cfg.model.policy.autoregressive_model_params.ucgmts_config.ema_decay_rate = params['ucgmts_config']['ema_decay_rate']
                    model.cfg.model.policy.autoregressive_model_params.ucgmts_config.rfba_gap_steps = params['ucgmts_config']['rfba_gap_steps']
                    model.cfg.model.policy.autoregressive_model_params.ucgmts_config.extrapol_ratio = params['ucgmts_config']['extrapol_ratio']
                    
                    print("✓ Updated model.cfg with optimized parameters")
                
                # Print updated model parameters
                print(f"\n{'='*50}")
                print(f"MODEL PARAMETERS AFTER UPDATE:")
                print(f"{'='*50}")
                print(f"Updated num_sampling_steps: {autoregressive_params.num_sampling_steps}")
                print(f"Updated cfg: {autoregressive_params.cfg}")
                print(f"Updated temperature: {autoregressive_params.temperature}")
                print(f"Updated window_size: {autoregressive_params.window_size}")
                print(f"Updated lambda_local: {autoregressive_params.lambda_local}")
                print(f"Updated use_ucgm: {autoregressive_params.use_ucgm}")
                
                print(f"Updated ucgmts_config:")
                print(f"  transport_type: {autoregressive_params.ucgmts_config.transport_type}")
                print(f"  consistc_ratio: {autoregressive_params.ucgmts_config.consistc_ratio}")
                print(f"  scaled_cbl_eps: {autoregressive_params.ucgmts_config.scaled_cbl_eps}")
                print(f"  ema_decay_rate: {autoregressive_params.ucgmts_config.ema_decay_rate}")
                print(f"  rfba_gap_steps: {autoregressive_params.ucgmts_config.rfba_gap_steps}")
                print(f"  extrapol_ratio: {autoregressive_params.ucgmts_config.extrapol_ratio}")
                
                # Also print the actual model components
                if hasattr(model, 'model') and hasattr(model.model, 'diffactloss'):
                    diffactloss = model.model.diffactloss
                    print(f"Updated DiffActLoss num_sampling_steps: {diffactloss.num_sampling_steps}")
                    if hasattr(diffactloss, 'ucgmts'):
                        ucgmts = diffactloss.ucgmts
                        print(f"Updated UCGMTS parameters:")
                        print(f"  transport_type: {ucgmts.transport_type}")
                        print(f"  consistc_ratio: {ucgmts.consistc_ratio}")
                        print(f"  rfba_gap_steps: {ucgmts.rfba_gap_steps}")
                        print(f"  extrapol_ratio: {ucgmts.extrapol_ratio}")
                print(f"{'='*50}")
                
                print("Parameters successfully applied to model")
                return True
            else:
                print("Model does not have autoregressive_model_params attribute")
                return False
                
        except Exception as e:
            print(f"Error applying parameters to model: {e}")
            return False
    
    def _is_checkpoint_valid(self, checkpoint_path: str) -> bool:
        """
        Check if checkpoint file is valid and not corrupted.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            True if checkpoint is valid, False otherwise
        """
        try:
            if not os.path.exists(checkpoint_path):
                return False
            
            # Check file size (should be reasonable)
            file_size = os.path.getsize(checkpoint_path)
            if file_size < 1024:  # Less than 1KB is suspicious
                print(f"Checkpoint file too small: {file_size} bytes")
                return False
            
            # Try to actually load the checkpoint to verify it's valid
            try:
                with open(checkpoint_path, "rb") as f:
                    # Try to load just the header to check if it's valid
                    import dill
                    # Read a small portion to check if it's a valid pickle file
                    f.seek(0)
                    # Try to load the checkpoint
                    checkpoint = torch.load(f, pickle_module=dill, weights_only=False)
                    
                    # Check if it has the expected structure
                    if isinstance(checkpoint, dict) and 'cfg' in checkpoint:
                        return True
                    else:
                        print(f"Checkpoint doesn't have expected structure: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}")
                        return False
                        
            except Exception as load_error:
                print(f"Failed to load checkpoint: {load_error}")
                return False
                    
        except Exception as e:
            print(f"Error checking checkpoint validity: {e}")
            return False
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization history."""
        return {
            'total_optimizations': len(self.optimization_history),
            'best_score': self.best_score,
            'best_params': self.best_params,
            'optimization_history': self.optimization_history
        }
