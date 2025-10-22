# 自适应UCGM贝叶斯优化器使用说明

## 🎯 概述

我们实现了一个自适应的UCGM贝叶斯优化器，支持三种优化模式，可以根据不同的需求（性能优先、速度优先、平衡模式）来调整优化策略。

## 🚀 三种优化模式

### 1. **Speed Priority (速度优先)**
- **目标**: 最小化采样步数，优先考虑推理速度
- **步数范围**: 1-2步
- **参数特点**: 全范围参数搜索，让优化器自由探索最优组合
- **适用场景**: 实时应用、边缘设备、对延迟敏感的场景

### 2. **Performance Priority (性能优先)**
- **目标**: 最大化生成质量，允许更多步数
- **步数范围**: 2-10步
- **参数特点**: 全范围参数搜索，避免武断的步数划分限制
- **适用场景**: 离线处理、对质量要求极高的场景

### 3. **Balanced Mode (平衡模式)**
- **目标**: 兼顾性能和速度
- **步数范围**: 1-3步
- **参数特点**: 全范围参数搜索，让优化器找到最佳平衡点
- **适用场景**: 大多数应用场景的默认选择

## 📝 配置方法

### 在YAML配置文件中设置

```yaml
bayesian_optimization:
  enabled: true
  optimization_mode: "balanced"  # 可选: "performance", "speed", "balanced"
  start_epoch: 150
  interval: 10
  max_trials: 8
  final_trials: 15
  n_test: 3
  final_n_test: 5
  device: cuda:0
  output_dir: ./bayesian_optimization_logs
  use_best_checkpoint_for_final: true
```

### 配置参数说明

- `optimization_mode`: 优化模式选择
  - `"speed"`: 速度优先
  - `"performance"`: 性能优先  
  - `"balanced"`: 平衡模式（默认）
- 其他参数与原有贝叶斯优化器相同

## 🔧 技术实现

### 核心类

1. **`AdaptiveUCGMBayesianOptimizer`**: 自适应优化器主类
2. **`OptimizationMode`**: 优化模式枚举
3. **`TrainingBayesianOptimizer`**: 训练集成优化器（已更新支持自适应模式）

### 关键特性

1. **全范围参数搜索**: 避免武断的步数划分，让优化器自由探索
2. **固定Categorical选择**: 避免Optuna的"dynamic value space"错误
3. **模式特定步数限制**: 只在步数上做限制，参数范围完全开放

### 设计理念

**避免武断的参数划分**: 
- 不再根据步数强制划分参数范围
- 所有模式都使用全范围参数搜索 (0.0-1.0, 0.0-0.999, 0.0-10.0等)
- 让贝叶斯优化器自己发现最优的参数组合
- 避免"4步可能比100步算少步"这种武断判断

## 📊 参数约束总结

| 模式 | 步数范围 | 参数特点 | 主要特点 |
|------|----------|----------|----------|
| **Speed** | 1-2 | 全范围参数搜索 | 少步数，让优化器自由探索 |
| **Performance** | 2-10 | 全范围参数搜索 | 多步数，避免武断限制 |
| **Balanced** | 1-3 | 全范围参数搜索 | 平衡步数，自由优化 |

## 🎯 使用建议

### 选择优化模式

1. **实时应用** → 选择 `"speed"`
2. **离线处理** → 选择 `"performance"`  
3. **一般应用** → 选择 `"balanced"`（推荐）

### 参数调优

- 优化器会根据选择的模式自动调整参数搜索空间
- 无需手动设置复杂的参数约束
- 所有16个UCGM参数都会被优化

## 🔍 验证结果

测试显示三种模式都能正确工作：

- **Speed模式**: 倾向于选择1-2步，参数符合consistency model特征
- **Performance模式**: 倾向于选择2-5步，参数符合diffusion model特征  
- **Balanced模式**: 在1-3步之间平衡选择

## 📈 预期效果

- **Speed模式**: 推理速度提升2-5倍，质量略有下降
- **Performance模式**: 生成质量显著提升，推理时间增加
- **Balanced模式**: 在速度和质量之间找到最佳平衡点

## 🚀 快速开始

1. 在配置文件中设置 `optimization_mode`
2. 运行训练脚本
3. 优化器会在指定epoch开始自动优化
4. 查看优化日志了解参数变化

```bash
# 示例：使用速度优先模式训练pusht任务
python train.py --config-name=pusht_MLP bayesian_optimization.optimization_mode=speed
```

