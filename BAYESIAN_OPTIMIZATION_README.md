# 训练中集成贝叶斯优化使用说明

## 概述

本项目已经成功集成了贝叶斯优化功能到训练循环中，允许在训练过程中自适应地优化超参数。贝叶斯优化会在训练的中后段定期运行，找到最优的超参数组合并应用到模型中。

## 主要特性

1. **渐进式优化**: 在训练进行到一半后开始贝叶斯优化
2. **定期优化**: 每隔指定epoch运行一次优化
3. **动态参数更新**: 将找到的最优参数直接应用到模型中
4. **完整日志记录**: 记录所有优化过程和结果
5. **非破坏性集成**: 不影响原有训练流程

## 使用方法

### 1. 使用新的配置文件（推荐）

```bash
yes n | accelerate launch --num_processes=4 train.py \
    --config-dir=. \
    --config-name=pusht_DiT_hybrid_with_bayesian.yaml \
    model.policy.selected_training_mode=policy_model \
    logging.project=pusht_MLP \
    hydra.run.dir="checkpoints/pusht"
```

### 2. 使用原有配置文件并启用贝叶斯优化

```bash
yes n | accelerate launch --num_processes=4 train.py \
    --config-dir=. \
    --config-name=pusht_DiT_hybrid.yaml \
    model.policy.selected_training_mode=policy_model \
    logging.project=pusht_MLP \
    hydra.run.dir="checkpoints/pusht" \
    bayesian_optimization.enabled=true \
    bayesian_optimization.start_epoch=100 \
    bayesian_optimization.interval=10
```

### 3. 自定义贝叶斯优化参数

```bash
yes n | accelerate launch --num_processes=4 train.py \
    --config-dir=. \
    --config-name=pusht_DiT_hybrid.yaml \
    model.policy.selected_training_mode=policy_model \
    logging.project=pusht_MLP \
    hydra.run.dir="checkpoints/pusht" \
    bayesian_optimization.enabled=true \
    bayesian_optimization.start_epoch=80 \
    bayesian_optimization.interval=15 \
    bayesian_optimization.max_trials=20 \
    bayesian_optimization.n_test=8
```

## 配置参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bayesian_optimization.enabled` | `false` | 是否启用贝叶斯优化 |
| `bayesian_optimization.start_epoch` | `num_epochs // 2` | 开始优化的epoch（训练的一半） |
| `bayesian_optimization.interval` | `10` | 优化间隔（每N个epoch） |
| `bayesian_optimization.max_trials` | `15` | 每次优化的最大试验次数 |
| `bayesian_optimization.n_test` | `5` | 每次评估的测试次数 |
| `bayesian_optimization.device` | `cuda:0` | 评估使用的设备 |
| `bayesian_optimization.output_dir` | `./bayesian_optimization_logs` | 优化日志输出目录 |

## 优化的超参数

贝叶斯优化会优化以下UCGM相关参数：

1. **consistc_ratio**: 一致性比例 (0.5-1.0)
2. **rfba_gap_end**: RFBA间隙结束点 (0.1-0.8)
3. **temperature**: 温度参数 (0.7-1.2)
4. **num_sampling_steps**: 采样步数 (1, 2, 3)
5. **cfg**: CFG参数 (0.8-1.5)
6. **extrapol_ratio**: 外推比例 (0.0-0.6)
7. **window_size**: 局部注意力窗口大小 (0-25)
8. **lambda_local**: 局部注意力权重 (0.01-0.8)

## 输出文件

优化过程会生成以下文件：

1. `bayesian_optimization_logs/optimization_epoch_X.json`: 每个epoch的优化结果
2. `bayesian_optimization_logs/optimization_history.json`: 完整的优化历史
3. 训练日志中会包含优化分数和参数信息

## 工作流程

1. **训练开始**: 正常训练流程，贝叶斯优化器初始化但未激活
2. **达到起始epoch**: 开始第一次贝叶斯优化
3. **定期优化**: 每隔指定间隔运行优化
4. **参数应用**: 将找到的最优参数应用到模型中
5. **继续训练**: 使用优化后的参数继续训练
6. **训练结束**: 输出优化总结信息

## 注意事项

1. **计算开销**: 贝叶斯优化会增加训练时间，建议在资源充足时使用
2. **评估依赖**: 需要确保`eval_sim.py`脚本可用且配置正确
3. **检查点**: 优化需要最新的检查点文件，确保检查点保存正常
4. **设备配置**: 确保指定的CUDA设备可用
5. **内存使用**: 优化过程会创建临时文件，注意磁盘空间

## 故障排除

### 常见问题

1. **"Checkpoint not found"**: 检查检查点保存路径和文件名
2. **"Evaluation failed"**: 检查`eval_sim.py`脚本和依赖
3. **"CUDA out of memory"**: 减少`n_test`参数或使用更小的设备
4. **"Optimization timeout"**: 增加超时时间或减少`max_trials`

### 调试模式

启用调试模式可以减少评估开销：

```bash
bayesian_optimization.max_trials=5 \
bayesian_optimization.n_test=2
```

## 性能建议

1. **起始epoch**: 建议设置为总epoch数的50-60%
2. **优化间隔**: 建议10-15个epoch，避免过于频繁
3. **试验次数**: 训练集成建议10-20次，平衡效果和效率
4. **测试次数**: 建议5-10次，确保评估稳定性

通过这种方式，你可以在训练过程中自动找到最优的超参数组合，提高模型的最终性能。
