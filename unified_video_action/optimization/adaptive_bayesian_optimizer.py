import optuna
import json
from typing import Dict, Any, Tuple
from enum import Enum

class OptimizationMode(Enum):
    PERFORMANCE = "performance"  # performance priority 
    SPEED = "speed"            # speed priority 
    BALANCED = "balanced"      # (default)

class AdaptiveUCGMBayesianOptimizer:

    def __init__(self, max_trials: int = 30, optimization_mode: OptimizationMode = OptimizationMode.BALANCED):
        self.max_trials = int(max_trials) 
        self.optimization_mode = optimization_mode
        self.study = optuna.create_study(direction='maximize')
        self.best_params = None
        
        print(f"AdaptiveUCGMBayesianOptimizer initialized with mode: {optimization_mode.value}")

    def optimize_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Generate different parameter search spaces based on the optimization mode
        """
        if self.optimization_mode == OptimizationMode.SPEED:
            return self._optimize_for_speed(trial)
        elif self.optimization_mode == OptimizationMode.PERFORMANCE:
            return self._optimize_for_performance(trial)
        else:  # BALANCED
            return self._optimize_for_balanced(trial)

    def _optimize_for_speed(self, trial: optuna.Trial) -> Dict[str, Any]:
        print("Optimizing for SPEED priority...")
        
        num_sampling_steps = trial.suggest_categorical('num_sampling_steps', [1, 2])
        

        consistc_ratio = trial.suggest_float('consistc_ratio', 0.6, 1.0)  # 重叠参数
        rfba_gap_end = trial.suggest_float('rfba_gap_end', 0.1, 0.7)  # 重叠参数
        extrapol_ratio = trial.suggest_float('extrapol_ratio', 0.0, 0.6)  # 重叠参数

        # 只保留需要优化的参数
        params = {
            # UCGM Sampling parameters
            'consistc_ratio': consistc_ratio,
            'rfba_gap_end': rfba_gap_end,
            'temperature': trial.suggest_float('temperature', 0.7, 1.2),
            'num_sampling_steps': num_sampling_steps,
            'cfg': trial.suggest_float('cfg', 0.8, 1.5),
            'extrapol_ratio': extrapol_ratio,
            
            # Local attention parameters
            'window_size': trial.suggest_int('window_size', 0, 15),
            'lambda_local': trial.suggest_float('lambda_local', 0.01, 0.8),
        }
        
        return self._build_ucgmts_config(params)

    def _optimize_for_performance(self, trial: optuna.Trial) -> Dict[str, Any]:

        print("Optimizing for PERFORMANCE priority...")
        
        # 允许更多步数
        num_sampling_steps = trial.suggest_categorical('num_sampling_steps', [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        

        consistc_ratio = trial.suggest_float('consistc_ratio', 0.0, 1.0) 
        rfba_gap_end = trial.suggest_float('rfba_gap_end', 0.001, 0.8)
        extrapol_ratio = trial.suggest_float('extrapol_ratio', 0.0, 0.6) 

        params = {
            # UCGM Sampling parameters
            'consistc_ratio': consistc_ratio,
            'rfba_gap_end': rfba_gap_end,
            'temperature': trial.suggest_float('temperature', 0.8, 1.2),
            'num_sampling_steps': num_sampling_steps,
            'cfg': trial.suggest_float('cfg', 1.0, 2.0),  # 允许更高的CFG
            'extrapol_ratio': extrapol_ratio,
            
            # Local attention parameters
            'window_size': trial.suggest_int('window_size', 4, 16),
            'lambda_local': trial.suggest_float('lambda_local', 0.1, 1.0),
        }
        
        return self._build_ucgmts_config(params)

    def _optimize_for_balanced(self, trial: optuna.Trial) -> Dict[str, Any]:

        print("Optimizing for BALANCED mode...")
        
        # 平衡的步数范围
        num_sampling_steps = trial.suggest_categorical('num_sampling_steps', [1, 2, 3])
        
        # 只优化重叠参数，移除训练独有参数以保证稳定性
        # 重叠参数：consistc_ratio, rfba_gap_end, extrapol_ratio
        consistc_ratio = trial.suggest_float('consistc_ratio', 0.5, 1.0)  # 重叠参数
        rfba_gap_end = trial.suggest_float('rfba_gap_end', 0.1, 0.8)  # 重叠参数
        extrapol_ratio = trial.suggest_float('extrapol_ratio', 0.0, 0.6)  # 重叠参数

        params = {
            # UCGM Sampling parameters
            'consistc_ratio': consistc_ratio,
            'rfba_gap_end': rfba_gap_end,
            'temperature': trial.suggest_float('temperature', 0.7, 1.2),
            'num_sampling_steps': num_sampling_steps,
            'cfg': trial.suggest_float('cfg', 0.8, 1.5),
            'extrapol_ratio': extrapol_ratio,
            
            # Local attention parameters
            'window_size': trial.suggest_int('window_size', 0, 15),
            'lambda_local': trial.suggest_float('lambda_local', 0.01, 0.8),
        }
        
        return self._build_ucgmts_config(params)

    def _build_ucgmts_config(self, params: Dict[str, Any]) -> Dict[str, Any]:

        params['ucgmts_config'] = {
            'consistc_ratio': params['consistc_ratio'],
            'rfba_gap_steps': [0.001, params['rfba_gap_end']],
            'extrapol_ratio': params['extrapol_ratio']
        }
        
        return params
    
    def optimize(self, objective_func) -> Tuple[Dict[str, Any], float]: 

        def objective(trial):
            params = self.optimize_params(trial)
            score = objective_func(params)
            return score
        
        self.study.optimize(objective, n_trials=self.max_trials)
        
        # Process the best parameters to include ucgmts_config structure
        raw_best_params = self.study.best_params
        
        self.best_params = self._build_ucgmts_config(raw_best_params.copy())
        best_score = self.study.best_value
        
        print(f"\nOptimization completed!")
        print(f"Best score: {best_score:.4f}")
        print(f"Best parameters:")
        for key, value in self.best_params.items():
            if key == 'ucgmts_config':
                print(f"   {key}:")
                for sub_key, sub_value in value.items():
                    print(f"     {sub_key}: {sub_value}")
            else:
                print(f"   {key}: {value}")
        
        return self.best_params, best_score

    def get_optimization_summary(self) -> Dict[str, Any]:

        return {
            'optimization_mode': self.optimization_mode.value,
            'max_trials': self.max_trials,
            'best_score': self.study.best_value if self.study.best_value else None,
            'best_params': self.best_params,
            'num_trials': len(self.study.trials),
            'optimization_completed': len(self.study.trials) >= self.max_trials
        }
