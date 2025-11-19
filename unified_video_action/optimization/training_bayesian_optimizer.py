
import os
import json
import tempfile
import subprocess
import torch
import dill
import numpy as np
from typing import Dict, Any, Optional, Tuple
from omegaconf import OmegaConf, open_dict
from unified_video_action.optimization.bayesian_optimizer import DGMBayesianOptimizer


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
            optimization_mode: Optimization mode - "speed_priority", "performance_priority", or "balanced"
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
        
        # Validate optimization mode
        valid_modes = ["speed_priority", "performance_priority", "balanced"]
        if optimization_mode not in valid_modes:
            raise ValueError(f"Invalid optimization_mode: {optimization_mode}. Must be one of {valid_modes}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Track optimization history
        self.optimization_history = []
        self.best_params = None
        self.best_score = -float('inf')
        
        # Initialize optimizer
        self.optimizer = DGMBayesianOptimizer(max_trials=max_trials)
        
        print(f"TrainingBayesianOptimizer initialized:")
        print(f"  Start epoch: {start_epoch}")
        print(f"  Interval: {interval}")
        print(f"  Max trials (normal): {max_trials}")
        print(f"  Max trials (final): {final_trials}")
        print(f"  N test (normal): {n_test}")
        print(f"  N test (final): {final_n_test}")
        print(f"  Use best checkpoint for final: {use_best_checkpoint_for_final}")
        print(f"  Output dir: {output_dir}")
        print(f"  Optimization mode: {optimization_mode}")
    
    def should_optimize(self, current_epoch: int) -> bool:
        """Check if optimization should be performed at current epoch."""
        return (current_epoch >= self.start_epoch and 
                (current_epoch - self.start_epoch) % self.interval == 0)
    
    def get_current_checkpoint_score(self, checkpoint_path: str) -> float:
        """
        Get the current checkpoint score for comparison.
        
        Args:
            checkpoint_path: Path to current checkpoint
            
        Returns:
            Current checkpoint score, or -1000.0 if evaluation fails
        """
        try:
            print(f"Evaluating current checkpoint: {checkpoint_path}")
            
            # Clear CUDA cache before evaluation
            torch.cuda.empty_cache()
            
            # Wait a bit to ensure checkpoint is fully written
            import time
            time.sleep(5)
            
            # Check if checkpoint file is valid before loading
            if not self._is_checkpoint_valid(checkpoint_path):
                print(f"Checkpoint file is invalid or corrupted: {checkpoint_path}")
                return -1000.0
            
            # Load checkpoint and config
            payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill, weights_only=False)
            cfg = payload["cfg"]
            
            # Set test count
            if "libero" in cfg.task.name:
                cfg.task.env_runner.n_test = self.n_test
                timeout = max(1200, self.n_test * 60 * 10)  
            else:
                cfg.task.env_runner.n_test = min(self.n_test * 5, 50)
                timeout = 300  # 5 minutes for other tasks
            
            # Create temp output directory
            temp_output_dir = tempfile.mkdtemp(prefix="current_eval_")
            
            # Run evaluation with current parameters (no parameter modification)
            cmd = [
                "python", "eval_sim.py",
                "--checkpoint", checkpoint_path,
                "--output_dir", temp_output_dir,
                "--device", self.device,
                "--use_ucgm"
            ]
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = self.device.split(":")[-1] if ":" in self.device else "0"
            
            print(f"Running evaluation with timeout: {timeout} seconds")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout
            )
            
            if result.returncode != 0:
                print(f"Current checkpoint evaluation failed: {result.stderr}")
                return -1000.0
            
            # Parse results
            eval_log_path = os.path.join(temp_output_dir, f'eval_log_{os.path.basename(checkpoint_path)}.json')
            
            if not os.path.exists(eval_log_path):
                print(f"Current eval log not found: {eval_log_path}")
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
                    print(f"No score found in current checkpoint, available keys: {list(eval_results.keys())}")
                    return -1000.0
            
            print(f"Current checkpoint score: {score}")
            
            # Cleanup
            subprocess.run(["rm", "-rf", temp_output_dir], check=False)
            
            return float(score)
            
        except subprocess.TimeoutExpired:
            print(f"Current checkpoint evaluation timeout after {timeout} seconds")
            print(f"Evaluation may need more time. Consider increasing timeout or reducing n_test.")
            torch.cuda.empty_cache()
            return -1000.0
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e).lower():
                print(f"CUDA error during current checkpoint evaluation: {e}")
                print("Clearing CUDA cache and retrying...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                return -1000.0
            else:
                print(f"Runtime error during current checkpoint evaluation: {e}")
                return -1000.0
        except Exception as e:
            print(f"Current checkpoint evaluation error: {e}")
            torch.cuda.empty_cache()
            return -1000.0

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
            
            # Clear CUDA cache before evaluation
            torch.cuda.empty_cache()
            
            # Wait a bit to ensure checkpoint is fully written
            import time
            time.sleep(5)
            
            # Check if checkpoint file is valid before loading
            if not self._is_checkpoint_valid(checkpoint_path):
                print(f"Checkpoint file is invalid or corrupted: {checkpoint_path}")
                return -1000.0
            
            # Load checkpoint and config
            payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill, weights_only=False)
            cfg = payload["cfg"] #model config
            
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
                # For libero10, each task needs more time (10 tasks * n_test * max_steps)
                # Estimate: ~30-60 seconds per test per task depending on complexity
                # Calculate timeout with buffer: at least 20 minutes, scale with n_test
                # Formula: base_time (20 min) + n_test * tasks * time_per_test
                timeout = max(1200, self.n_test * 60 * 10) 
            else:
                cfg.task.env_runner.n_test = min(self.n_test * 5, 50)
                timeout = 300  # 5 minutes for other tasks
            
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
            
            print(f"Running evaluation with timeout: {timeout} seconds")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout
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
            print(f"Evaluation timeout after {timeout} seconds")
            print(f"Evaluation may need more time. Consider increasing timeout or reducing n_test.")
            torch.cuda.empty_cache()
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
            torch.cuda.empty_cache()
            return -1000.0
    
    def run_optimization(self, 
                        checkpoint_path: str, 
                        current_epoch: int,
                        checkpoints_dir: str = None) -> Optional[Tuple[Dict[str, Any], float]]:
        """
        Run Bayesian optimization for current epoch.
        
        Args:
            checkpoint_path: Path to current checkpoint
            current_epoch: Current training epoch
            checkpoints_dir: Directory containing checkpoints (for finding best checkpoint)
            
        Returns:
            Tuple of (best parameters, best score) if optimization succeeded, None otherwise
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
        
        # Create a temporary optimizer with adjusted parameters and mode-specific ranges
        temp_optimizer = DGMBayesianOptimizer(max_trials=max_trials, optimization_mode=self.optimization_mode)
        
        def objective_function(params):
            return self.evaluate_model_with_params(params, checkpoint_path)
        
        try:
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
                print(f"Best score: {best_score:.4f}")
                print(f"Best params: {best_params}")
                
                # Update best parameters if this is the best so far
                if best_score > self.best_score:
                    self.best_score = best_score
                    self.best_params = best_params
                    print(f"New best parameters found! Score: {best_score:.4f}")
                
                # Save optimization results
                import time
                optimization_result = {
                    'epoch': current_epoch,
                    'best_params': best_params,
                    'best_score': best_score,
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
                
                return (best_params, best_score)
            else:
                print("Optimization failed or no valid parameters found")
                return None
                
        except Exception as e:
            print(f"Optimization error: {e}")
            return None
    
    def apply_best_params_to_model(self, 
                                 model, 
                                 params: Dict[str, Any],
                                 workspace=None,
                                 current_checkpoint_path: str = None,
                                 optimized_score: float = None) -> bool:
        """
        Apply the best parameters to the model, but only if they perform better than current checkpoint.
        
        Args:
            model: The model to update
            params: Parameters to apply
            workspace: Workspace object for updating config
            current_checkpoint_path: Path to current checkpoint for score comparison
            optimized_score: Score achieved by optimized parameters
            
        Returns:
            True if parameters were applied, False otherwise
        """
        try:
            # Score comparison logic
            if current_checkpoint_path is not None and optimized_score is not None:
                print(f"\n{'='*60}")
                print(f"SCORE COMPARISON:")
                print(f"{'='*60}")
                print(f"Optimized score: {optimized_score:.4f}")
                
                # Get current checkpoint score
                current_score = self.get_current_checkpoint_score(current_checkpoint_path)
                
                if current_score > -1000.0:  # Valid score
                    print(f"Current checkpoint score: {current_score:.4f}")
                    improvement = optimized_score - current_score
                    print(f"Improvement: {improvement:+.4f}")
                    
                    if optimized_score <= current_score:
                        print(f"   Optimized: {optimized_score:.4f} <= Current: {current_score:.4f}")
                        print(f"   Skipping parameter application to avoid performance degradation")
                        return False
                    else:
                        print(f"   Optimized: {optimized_score:.4f} > Current: {current_score:.4f}")
                        print(f"   Proceeding with parameter application")
                else:
                    print(f"Failed to get current checkpoint score, proceeding with parameter application")
            else:
                print(f"No score comparison available, proceeding with parameter application")
            
            print(f"\n{'='*60}")
            print(f"APPLYING PARAMETERS TO MODEL:")
            print(f"{'='*60}")
            print(f"Parameters: {params}")
            print(f"Model type: {type(model)}")
            print(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
            
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
            if hasattr(model, 'autoregressive_model_params'):
                autoregressive_params = model.autoregressive_model_params
            elif hasattr(model, 'model') and hasattr(model.model, 'autoregressive_model_params'):
                # Handle case where model is wrapped (e.g., DDP wrapper)
                autoregressive_params = model.model.autoregressive_model_params
            elif hasattr(model, 'module') and hasattr(model.module, 'autoregressive_model_params'):
                # Handle case where model is wrapped by accelerator (e.g., AcceleratedModel)
                autoregressive_params = model.module.autoregressive_model_params
            else:
                print(f"Model type: {type(model)}")
                print(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
                
                # Try to find autoregressive_model_params in nested structure
                if hasattr(model, 'module'):
                    print(f"Model.module type: {type(model.module)}")
                    print(f"Model.module attributes: {[attr for attr in dir(model.module) if not attr.startswith('_')]}")
                    if hasattr(model.module, 'autoregressive_model_params'):
                        print("Found autoregressive_model_params in model.module")
                        autoregressive_params = model.module.autoregressive_model_params
                    else:
                        print("Model does not have autoregressive_model_params attribute")
                        return False
                elif hasattr(model, 'model'):
                    print(f"Model.model type: {type(model.model)}")
                    print(f"Model.model attributes: {[attr for attr in dir(model.model) if not attr.startswith('_')]}")
                    if hasattr(model.model, 'autoregressive_model_params'):
                        print("Found autoregressive_model_params in model.model")
                        autoregressive_params = model.model.autoregressive_model_params
                    else:
                        print("Model does not have autoregressive_model_params attribute")
                        return False
                else:
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
            
            # Only update the optimized parameters, keep default values for fixed parameters
            autoregressive_params.ucgmts_config.consistc_ratio = params['ucgmts_config']['consistc_ratio']
            autoregressive_params.ucgmts_config.rfba_gap_steps = params['ucgmts_config']['rfba_gap_steps']
            autoregressive_params.ucgmts_config.extrapol_ratio = params['ucgmts_config']['extrapol_ratio']
            # Note: transport_type, scaled_cbl_eps, ema_decay_rate are kept as default values
            
            # Also update the actual model components
            # Handle different model wrapping scenarios
            actual_model = None
            if hasattr(model, 'module'):
                # Accelerator wrapped model
                actual_model = model.module
            elif hasattr(model, 'model'):
                # DDP wrapped model
                actual_model = model.model
            else:
                # Direct model
                actual_model = model
            
            if hasattr(actual_model, 'diffactloss'):
                diffactloss = actual_model.diffactloss
                diffactloss.num_sampling_steps = params['num_sampling_steps']
                
                if hasattr(diffactloss, 'ucgmts'):
                    ucgmts = diffactloss.ucgmts
                    # Only update the optimized parameters, keep default values for fixed parameters
                    ucgmts.consistc_ratio = params['ucgmts_config']['consistc_ratio']
                    ucgmts.rfba_gap_steps = params['ucgmts_config']['rfba_gap_steps']
                    ucgmts.extrapol_ratio = params['ucgmts_config']['extrapol_ratio']
                    # Note: transport_type, scaled_cbl_eps, ema_decay_rate are kept as default values
            
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
            
            print(f"Updated ucgmts_config (optimized parameters only):")
            print(f"  consistc_ratio: {autoregressive_params.ucgmts_config.consistc_ratio}")
            print(f"  rfba_gap_steps: {autoregressive_params.ucgmts_config.rfba_gap_steps}")
            print(f"  extrapol_ratio: {autoregressive_params.ucgmts_config.extrapol_ratio}")
            print(f"  (transport_type, scaled_cbl_eps, ema_decay_rate kept as default)")
            
            # Also print the actual model components
            if hasattr(actual_model, 'diffactloss'):
                diffactloss = actual_model.diffactloss
                print(f"Updated DiffActLoss num_sampling_steps: {diffactloss.num_sampling_steps}")
                if hasattr(diffactloss, 'ucgmts'):
                    ucgmts = diffactloss.ucgmts
                    print(f"Updated UCGMTS parameters (optimized only):")
                    print(f"  consistc_ratio: {ucgmts.consistc_ratio}")
                    print(f"  rfba_gap_steps: {ucgmts.rfba_gap_steps}")
                    print(f"  extrapol_ratio: {ucgmts.extrapol_ratio}")
                    print(f"  (transport_type, scaled_cbl_eps, ema_decay_rate kept as default)")
            print(f"{'='*50}")
            
            # CRITICAL: Also update the workspace cfg to ensure parameters are saved in checkpoint
            if workspace is not None and hasattr(workspace, 'cfg') and workspace.cfg is not None:
                with open_dict(workspace.cfg.model.policy.autoregressive_model_params):
                    if "ucgmts_config" not in workspace.cfg.model.policy.autoregressive_model_params:
                        workspace.cfg.model.policy.autoregressive_model_params.ucgmts_config = OmegaConf.create({})
                    
                    workspace.cfg.model.policy.autoregressive_model_params.use_ucgm = True
                    workspace.cfg.model.policy.autoregressive_model_params.num_sampling_steps = params['num_sampling_steps']
                    workspace.cfg.model.policy.autoregressive_model_params.cfg = params['cfg']
                    workspace.cfg.model.policy.autoregressive_model_params.temperature = params['temperature']
                    workspace.cfg.model.policy.autoregressive_model_params.window_size = params['window_size']
                    workspace.cfg.model.policy.autoregressive_model_params.lambda_local = params['lambda_local']
                    
                    # Only update the optimized parameters, keep default values for fixed parameters
                    workspace.cfg.model.policy.autoregressive_model_params.ucgmts_config.consistc_ratio = params['ucgmts_config']['consistc_ratio']
                    workspace.cfg.model.policy.autoregressive_model_params.ucgmts_config.rfba_gap_steps = params['ucgmts_config']['rfba_gap_steps']
                    workspace.cfg.model.policy.autoregressive_model_params.ucgmts_config.extrapol_ratio = params['ucgmts_config']['extrapol_ratio']
                    # Note: transport_type, scaled_cbl_eps, ema_decay_rate are kept as default values
                    
                    print("✓ Updated workspace.cfg with optimized parameters")
            
            print("Parameters successfully applied to model")
            return True
                
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
