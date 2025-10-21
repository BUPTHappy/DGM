# 贝叶斯优化参数应用脚本使用说明

## 问题总结
你的贝叶斯优化确实找到了最优参数（score: 0.9919），但是这些参数没有成功保存到checkpoint中。所以：
- 贝叶斯优化评估时使用优化参数 → 得到0.9919的性能
- 手动评估checkpoint时使用原始参数 → 得到较低的性能

## 解决方案

### 方案1：使用脚本创建包含优化参数的checkpoint

在远程服务器上运行：

```bash
# 1. 复制脚本到远程服务器
# 2. 运行脚本创建优化后的checkpoint
python apply_bayesian_params.py \
    --input_checkpoint checkpoints/pusht_production_opt/checkpoints/epoch=0050-test_mean_score=0.991.ckpt \
    --output_checkpoint checkpoints/pusht_production_opt/checkpoints/epoch=0050-test_mean_score=0.991_optimized.ckpt

# 3. 评估优化后的checkpoint
CUDA_VISIBLE_DEVICES=0 python eval_sim.py \
    --checkpoint checkpoints/pusht_production_opt/checkpoints/epoch=0050-test_mean_score=0.991_optimized.ckpt \
    --output_dir checkpoints/pusht_optimized_eval/
```

### 方案2：手动传递优化参数（你已经验证过这个方法有效）

```bash
CUDA_VISIBLE_DEVICES=0 python eval_sim.py \
    --checkpoint checkpoints/pusht_production_opt/checkpoints/epoch=0050-test_mean_score=0.991.ckpt \
    --output_dir checkpoints/pusht_DiT_hybrid_rollout/ \
    --use_ucgm \
    --num_sampling_steps 2 \
    --stochasticity_rate 0.9870031611149999 \
    --window_size 8 \
    --lambda_local 0.3566535175072145
```

## 贝叶斯优化找到的最优参数

```json
{
    "consistc_ratio": 0.9870031611149999,
    "rfba_gap_end": 0.22533496150457658,
    "temperature": 0.8836599795042932,
    "num_sampling_steps": 2,
    "cfg": 1.0260267723044316,
    "extrapol_ratio": 0.5318789824701121,
    "window_size": 8,
    "lambda_local": 0.3566535175072145,
    "ucgmts_config": {
        "transport_type": "Linear",
        "consistc_ratio": 0.9870031611149999,
        "scaled_cbl_eps": 0.0,
        "ema_decay_rate": 0.0,
        "rfba_gap_steps": [0.001, 0.22533496150457658],
        "extrapol_ratio": 0.5318789824701121
    }
}
```

## 预期结果
使用这些优化参数，你应该能够达到 **0.9919** 的性能分数，而不是原始checkpoint的较低性能。

## 长期解决方案
我已经修改了训练代码，确保贝叶斯优化的参数会正确保存到checkpoint中。下次训练时，这个问题就不会再出现了。
