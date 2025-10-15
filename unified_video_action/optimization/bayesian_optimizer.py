import optuna
import json
from typing import Dict, Any, Tuple

class UCGMBayesianOptimizer:
    def __init__(self,max_trials: int =30):
        self.max_trials = max_trials
        self.study = optuna.create_study(direction='maximize')
        self.best_params = None

    def optimize_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {
            'consistc_ratio': trial.suggest_float('consistc_ratio', 0.5, 1.0), # 当前值：1.0
            'rfba_gap_end': trial.suggest_float('rfba_gap_end', 0.1, 0.8), # rfba_gap_end = 0.5 (优化这个结束点)
            'temperature': trial.suggest_float('temperature', 0.7, 1.2), #0.95
            'num_sampling_steps': trial.suggest_categorical('num_sampling_steps', [1, 2, 3]), # 当前值：2
            'cfg': trial.suggest_float('cfg', 0.8, 1.5), # 当前值：1
            'extrapol_ratio': trial.suggest_float('extrapol_ratio', 0.0, 0.6), # 当前值：0.0 (加速采样)
            # 新增的local attention参数
            'window_size': trial.suggest_int('window_size', 0, 25), # 当前值：15
            'lambda_local': trial.suggest_float('lambda_local', 0.01, 0.8) # 当前值：0.1
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