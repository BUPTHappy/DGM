import optuna
import json
from typing import Dict, Any, Tuple
from enum import Enum

class OptimizationMode(Enum):
    """优化模式枚举"""
    PERFORMANCE = "performance"  # 性能优先
    SPEED = "speed"            # 速度优先  
    BALANCED = "balanced"      # 平衡模式（默认）

class AdaptiveUCGMBayesianOptimizer:
    """
    自适应UCGM贝叶斯优化器
    
    支持三种优化模式：
    1. PERFORMANCE: 性能优先，允许更多步数以获得最佳效果
    2. SPEED: 速度优先，最小化步数，其他参数相应调整
    3. BALANCED: 平衡模式，兼顾性能和速度
    """
    
    def __init__(self, max_trials: int = 30, optimization_mode: OptimizationMode = OptimizationMode.BALANCED):
        self.max_trials = max_trials
        self.optimization_mode = optimization_mode
        self.study = optuna.create_study(direction='maximize')
        self.best_params = None
        
        print(f"AdaptiveUCGMBayesianOptimizer initialized with mode: {optimization_mode.value}")

    def optimize_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        根据优化模式生成不同的参数搜索空间
        """
        if self.optimization_mode == OptimizationMode.SPEED:
            return self._optimize_for_speed(trial)
        elif self.optimization_mode == OptimizationMode.PERFORMANCE:
            return self._optimize_for_performance(trial)
        else:  # BALANCED
            return self._optimize_for_balanced(trial)

    def _optimize_for_speed(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        速度优先优化：最小化采样步数，其他参数相应调整
        """
        print("Optimizing for SPEED priority...")
        
        # 强制使用少步数采样
        num_sampling_steps = trial.suggest_categorical('num_sampling_steps', [1, 2])
        
        # 使用固定的categorical choices避免Optuna错误
        transport_type = trial.suggest_categorical('transport_type', ['Linear', 'TrigFlow', 'Cosine', 'DDPM'])
        wt_cosine_loss = trial.suggest_categorical('wt_cosine_loss', [True, False])
        weight_function = trial.suggest_categorical('weight_function', [None, 'Cosine'])
        
        # 统一的参数范围，不根据步数强制划分
        # 让优化器自由探索，只在步数上做限制
        consistc_ratio = trial.suggest_float('consistc_ratio', 0.0, 1.0)  # 全范围
        ema_decay_rate = trial.suggest_float('ema_decay_rate', 0.0, 0.999)  # 全范围
        scaled_cbl_eps = trial.suggest_float('scaled_cbl_eps', 0.0, 10.0)  # 全范围
        rfba_gap_end = trial.suggest_float('rfba_gap_end', 0.001, 0.8)  # 全范围
        extrapol_ratio = trial.suggest_float('extrapol_ratio', 0.0, 0.6)  # 全范围
        time_dist_ctrl = [
            trial.suggest_float('time_dist_ctrl_0', 0.5, 2.5),
            trial.suggest_float('time_dist_ctrl_1', 0.5, 2.5),
            trial.suggest_float('time_dist_ctrl_2', 0.5, 2.5)
        ]

        # 其他参数保持相对宽松的范围
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
            
            # UCGM Training parameters
            'ema_decay_rate': ema_decay_rate,
            'scaled_cbl_eps': scaled_cbl_eps,
            'transport_type': transport_type,
            'lab_drop_ratio': trial.suggest_float('lab_drop_ratio', 0.0, 0.3),
            'enhanced_ratio': trial.suggest_float('enhanced_ratio', 0.0, 2.0),
            'wt_cosine_loss': wt_cosine_loss,
            'weight_function': weight_function,
            'time_dist_ctrl_0': time_dist_ctrl[0],
            'time_dist_ctrl_1': time_dist_ctrl[1],
            'time_dist_ctrl_2': time_dist_ctrl[2],
        }
        
        return self._build_ucgmts_config(params)

    def _optimize_for_performance(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        性能优先优化：允许更多步数以获得最佳效果
        """
        print("🎯 Optimizing for PERFORMANCE priority...")
        
        # 允许更多步数
        num_sampling_steps = trial.suggest_categorical('num_sampling_steps', [2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # 使用固定的categorical choices避免Optuna错误
        transport_type = trial.suggest_categorical('transport_type', ['Linear', 'TrigFlow', 'Cosine', 'DDPM'])
        wt_cosine_loss = trial.suggest_categorical('wt_cosine_loss', [True, False])
        weight_function = trial.suggest_categorical('weight_function', [None, 'Cosine'])
        
        # 根据步数调整参数
        # 使用更灵活的参数范围，避免武断的步数划分
        # 所有参数都使用统一的宽松范围，让优化器自由探索
        
        # 统一的参数范围，不根据步数强制划分
        consistc_ratio = trial.suggest_float('consistc_ratio', 0.0, 1.0)  # 全范围
        ema_decay_rate = trial.suggest_float('ema_decay_rate', 0.0, 0.999)  # 全范围
        scaled_cbl_eps = trial.suggest_float('scaled_cbl_eps', 0.0, 10.0)  # 全范围
        rfba_gap_end = trial.suggest_float('rfba_gap_end', 0.001, 0.8)  # 全范围
        extrapol_ratio = trial.suggest_float('extrapol_ratio', 0.0, 0.6)  # 全范围
        time_dist_ctrl = [
            trial.suggest_float('time_dist_ctrl_0', 0.5, 2.5),
            trial.suggest_float('time_dist_ctrl_1', 0.5, 2.5),
            trial.suggest_float('time_dist_ctrl_2', 0.5, 2.5)
        ]

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
            
            # UCGM Training parameters
            'ema_decay_rate': ema_decay_rate,
            'scaled_cbl_eps': scaled_cbl_eps,
            'transport_type': transport_type,
            'lab_drop_ratio': trial.suggest_float('lab_drop_ratio', 0.0, 0.3),
            'enhanced_ratio': trial.suggest_float('enhanced_ratio', 0.0, 2.0),
            'wt_cosine_loss': wt_cosine_loss,
            'weight_function': weight_function,
            'time_dist_ctrl_0': time_dist_ctrl[0],
            'time_dist_ctrl_1': time_dist_ctrl[1],
            'time_dist_ctrl_2': time_dist_ctrl[2],
        }
        
        return self._build_ucgmts_config(params)

    def _optimize_for_balanced(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        平衡模式优化：兼顾性能和速度
        """
        print("⚖️ Optimizing for BALANCED mode...")
        
        # 平衡的步数范围
        num_sampling_steps = trial.suggest_categorical('num_sampling_steps', [1, 2, 3])
        
        # 使用固定的categorical choices避免Optuna错误
        transport_type = trial.suggest_categorical('transport_type', ['Linear', 'TrigFlow', 'Cosine', 'DDPM'])
        wt_cosine_loss = trial.suggest_categorical('wt_cosine_loss', [True, False])
        weight_function = trial.suggest_categorical('weight_function', [None, 'Cosine'])
        
        # 统一的参数范围，不根据步数强制划分
        # 让优化器自由探索，只在步数上做限制
        consistc_ratio = trial.suggest_float('consistc_ratio', 0.0, 1.0)  # 全范围
        ema_decay_rate = trial.suggest_float('ema_decay_rate', 0.0, 0.999)  # 全范围
        scaled_cbl_eps = trial.suggest_float('scaled_cbl_eps', 0.0, 10.0)  # 全范围
        rfba_gap_end = trial.suggest_float('rfba_gap_end', 0.001, 0.8)  # 全范围
        extrapol_ratio = trial.suggest_float('extrapol_ratio', 0.0, 0.6)  # 全范围
        time_dist_ctrl = [
            trial.suggest_float('time_dist_ctrl_0', 0.5, 2.5),
            trial.suggest_float('time_dist_ctrl_1', 0.5, 2.5),
            trial.suggest_float('time_dist_ctrl_2', 0.5, 2.5)
        ]

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
            
            # UCGM Training parameters
            'ema_decay_rate': ema_decay_rate,
            'scaled_cbl_eps': scaled_cbl_eps,
            'transport_type': trial.suggest_categorical('transport_type', ['Linear', 'TrigFlow', 'Cosine', 'DDPM']),
            'lab_drop_ratio': trial.suggest_float('lab_drop_ratio', 0.0, 0.3),
            'enhanced_ratio': trial.suggest_float('enhanced_ratio', 0.0, 2.0),
            'wt_cosine_loss': trial.suggest_categorical('wt_cosine_loss', [True, False]),
            'weight_function': trial.suggest_categorical('weight_function', [None, 'Cosine']),
            'time_dist_ctrl_0': trial.suggest_float('time_dist_ctrl_0', 0.5, 2.5),
            'time_dist_ctrl_1': trial.suggest_float('time_dist_ctrl_1', 0.5, 2.5),
            'time_dist_ctrl_2': trial.suggest_float('time_dist_ctrl_2', 0.5, 2.5),
        }
        
        return self._build_ucgmts_config(params)

    def _build_ucgmts_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建UCGMTS配置
        """
        params['ucgmts_config'] = {
            'transport_type': params['transport_type'],
            'consistc_ratio': params['consistc_ratio'],
            'scaled_cbl_eps': params['scaled_cbl_eps'],
            'ema_decay_rate': params['ema_decay_rate'],
            'rfba_gap_steps': [0.001, params['rfba_gap_end']],
            'extrapol_ratio': params['extrapol_ratio'],
            'lab_drop_ratio': params['lab_drop_ratio'],
            'enhanced_ratio': params['enhanced_ratio'],
            'wt_cosine_loss': params['wt_cosine_loss'],
            'weight_function': params['weight_function'],
            'time_dist_ctrl': [params['time_dist_ctrl_0'], params['time_dist_ctrl_1'], params['time_dist_ctrl_2']]
        }
        
        return params
    
    def optimize(self, objective_func) -> Tuple[Dict[str, Any], float]: 
        """
        执行贝叶斯优化
        """
        def objective(trial):
            params = self.optimize_params(trial)
            score = objective_func(params)
            return score
        
        self.study.optimize(objective, n_trials=self.max_trials)
        self.best_params = self.study.best_params
        best_score = self.study.best_value
        
        print(f"\n🎉 Optimization completed!")
        print(f"📊 Best score: {best_score:.4f}")
        print(f"⚙️ Best parameters:")
        for key, value in self.best_params.items():
            print(f"   {key}: {value}")
        
        return self.best_params, best_score

    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        获取优化总结
        """
        return {
            'optimization_mode': self.optimization_mode.value,
            'max_trials': self.max_trials,
            'best_score': self.study.best_value if self.study.best_value else None,
            'best_params': self.best_params,
            'num_trials': len(self.study.trials),
            'optimization_completed': len(self.study.trials) >= self.max_trials
        }
