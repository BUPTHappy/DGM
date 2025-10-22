# 当前项目文件结构总结

## 📁 主要配置文件

### 训练配置
- `pusht_MLP.yaml` - Pusht任务的MLP配置（主要使用）
- `libero10_MLP.yaml` - Libero10任务的MLP配置

### 模型配置
- `model/fast_policy.yaml` - MLP模型配置

### 任务配置
- `task/pusht.yaml` - Pusht任务配置
- `task/libero10.yaml` - Libero10任务配置
- 其他任务配置文件保留

## 🔧 优化器文件

### 当前使用
- `optimization/adaptive_bayesian_optimizer.py` - 自适应贝叶斯优化器（主要使用）
- `optimization/training_bayesian_optimizer.py` - 训练集成优化器

### 已删除
- ~~`optimization/bayesian_optimizer.py`~~ - 旧的贝叶斯优化器（已删除）

## 🗑️ 已清理的文件

### DiT相关配置（已删除）
- ~~`libero10_DiT_hybrid_bayesian.yaml`~~
- ~~`libero10_DiT_hybrid.yaml`~~
- ~~`libero10_tp.yaml`~~
- ~~`pusht_DiT_hybrid_production.yaml`~~
- ~~`pusht_DiT_hybrid.yaml`~~
- ~~`pusht_tp.yaml`~~

### 旧配置文件（已删除）
- ~~`uva_pusht_ucgm.yaml`~~
- ~~`uva_pusht.yaml`~~
- ~~`uva_libero10.yaml`~~
- ~~`libero10_extracted.yaml`~~
- ~~`model/uva.yaml`~~

## 🚀 使用方法

### 训练Pusht任务
```bash
python train.py --config-name=pusht_MLP
```

### 训练Libero10任务
```bash
python train.py --config-name=libero10_MLP
```

### 优化模式选择
在YAML配置文件中设置：
```yaml
bayesian_optimization:
  optimization_mode: "balanced"  # 可选: "performance", "speed", "balanced"
```

## ✅ 当前状态

- ✅ 使用MLP架构 + UCGM
- ✅ 使用自适应贝叶斯优化器
- ✅ 支持三种优化模式（speed/performance/balanced）
- ✅ 清理了所有DiT相关文件
- ✅ 保留了基本的任务和模型配置
