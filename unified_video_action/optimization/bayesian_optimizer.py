import optuna
import json
from typing import Dict, Any, Tuple

class UCGMBayesianOptimizer:
    def __init__(self, max_trials: int = 30, optimization_mode: str = "balanced"):
        self.max_trials = max_trials
        self.optimization_mode = optimization_mode
        self.study = optuna.create_study(direction='maximize')
        self.best_params = None
        
        # Validate optimization mode
        valid_modes = ["speed_priority", "performance_priority", "balanced"]
        if optimization_mode not in valid_modes:
            raise ValueError(f"Invalid optimization_mode: {optimization_mode}. Must be one of {valid_modes}")

    def optimize_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        # Define parameter ranges based on optimization mode
        if self.optimization_mode == "speed_priority":
            # Speed priority: focus on fast inference with limited sampling steps
            # For few-step (1-2 steps), higher consistc_ratio is optimal (0.8-1.0)
            num_sampling_steps_options = [1, 2]
            consistc_ratio_range = (0.75, 1.0)  # Higher consistency for few-step
            rfba_gap_end_range = (0.3, 0.8)  # Mid to high for few-step
            temperature_range = (0.7, 1.0)  # Lower temperature for faster convergence
            cfg_range = (0.8, 1.2)  # Narrower CFG range
            extrapol_ratio_range = (0.0, 0.3)  # Lower extrapolation for speed
            window_size_range = (0, 10)  # Smaller window for speed
            lambda_local_range = (0.01, 0.5)  # Lower lambda for speed
            
        elif self.optimization_mode == "performance_priority":
            # Performance priority: allow more sampling steps and wider parameter ranges
            # For multi-step (5-15 steps), consistc_ratio can be lower (0.5-0.9)
            num_sampling_steps_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            consistc_ratio_range = (0.5, 0.9)  # Lower consistency for multi-step
            rfba_gap_end_range = (0.1, 0.5)  # Lower for multi-step
            temperature_range = (0.7, 1.3)  # Wider temperature range
            cfg_range = (0.8, 1.5)  # Wider CFG range
            extrapol_ratio_range = (0.0, 0.8)  # Higher extrapolation for performance
            window_size_range = (0, 20)  # Larger window for performance
            lambda_local_range = (0.01, 1.0)  # Higher lambda for performance
            
        else:  # balanced
            # Balanced: moderate ranges for good speed-performance tradeoff
            # Typical 2-3 steps, optimal consistc_ratio around 0.75-0.95
            num_sampling_steps_options = [1, 2, 3]
            consistc_ratio_range = (0.7, 1.0)  # Mid-high for balanced
            rfba_gap_end_range = (0.2, 0.7)  # Medium range
            temperature_range = (0.7, 1.2)
            cfg_range = (0.8, 1.5)
            extrapol_ratio_range = (0.0, 0.6)
            window_size_range = (0, 15)
            lambda_local_range = (0.01, 0.8)
        
        params = {
            'consistc_ratio': trial.suggest_float('consistc_ratio', *consistc_ratio_range),
            'rfba_gap_end': trial.suggest_float('rfba_gap_end', *rfba_gap_end_range),
            'temperature': trial.suggest_float('temperature', *temperature_range), #0.95
            'num_sampling_steps': trial.suggest_categorical('num_sampling_steps', num_sampling_steps_options), # 当前值：2
            'cfg': trial.suggest_float('cfg', *cfg_range), # 当前值：1
            'extrapol_ratio': trial.suggest_float('extrapol_ratio', *extrapol_ratio_range), # 当前值：0.0 (加速采样)
            # 新增的local attention参数
            'window_size': trial.suggest_int('window_size', *window_size_range), # 当前值：15
            'lambda_local': trial.suggest_float('lambda_local', *lambda_local_range) # 当前值：0.1
        }
        
        params['ucgmts_config'] = {
            'transport_type': "Linear",
            'consistc_ratio': params['consistc_ratio'],
            'scaled_cbl_eps': 0.0,
            'ema_decay_rate': 0.0,
            'rfba_gap_steps': [0.001, params['rfba_gap_end']],
            'extrapol_ratio': params['extrapol_ratio']
        }
        
        return params
    
    def optimize(self, objective_func) -> Tuple[Dict[str, Any], float]: 
        def objective(trial):
            params = self.optimize_params(trial) #生成新的参数组合
            score = objective_func(params) #评估参数组合
            if score > (self.best_params[1] if self.best_params else -1):
                self.best_params = (params, score) #更新最佳参数组合
            return score
        
        self.study.optimize(objective, n_trials=self.max_trials)
        return self.best_params #返回最佳参数和对应的分数
    
    def save_results(self, filename: str = "optimization_results.json"):
        if self.best_params:
            results = {
                'best_params': self.best_params[0],
                'best_score': self.best_params[1],
                'all_trials': [
                    {'params': trial.params, 'value': trial.value}
                    for trial in self.study.trials
                ]
            }
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)