#!/usr/bin/env python3

"""
自适应训练脚本 - 基于现有框架的简洁版本
在训练过程中定期运行贝叶斯优化来调整超参数
"""

import os
import sys
import torch
import dill
import json
import copy
import subprocess
import tempfile
import glob
import numpy as np
import random
from typing import Dict, Any, Optional
from omegaconf import OmegaConf, open_dict
import hydra
import pathlib
from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.optimization.bayesian_optimizer import UCGMBayesianOptimizer

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

if "WANDB_API_KEY" in os.environ:
    import wandb
    wandb.login(key=os.environ["WANDB_API_KEY"])


class AdaptiveTrainer:
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.current_params = self._extract_current_params()
        self.optimization_history = []
        self.best_score = -float('inf')
        
    def _extract_current_params(self) -> Dict[str, Any]:
        """提取当前的超参数"""
        autoregressive_params = self.cfg.model.policy.autoregressive_model_params
        ucgmts_config = autoregressive_params.get('ucgmts_config', {})
        
        return {
            'consistc_ratio': ucgmts_config.get('consistc_ratio', 1.0),
            'rfba_gap_end': ucgmts_config.get('rfba_gap_steps', [0.001, 0.5])[1],
            'temperature': autoregressive_params.get('temperature', 0.95),
            'num_sampling_steps': autoregressive_params.get('num_sampling_steps', 2),
            'cfg': autoregressive_params.get('cfg', 1.0),
            'extrapol_ratio': ucgmts_config.get('extrapol_ratio', 0.0),
            'window_size': autoregressive_params.get('window_size', 15),
            'lambda_local': autoregressive_params.get('lambda_local', 0.1)
        }
    
    def _update_config_with_params(self, params: Dict[str, Any]):
        """更新配置文件的超参数"""
        with open_dict(self.cfg.model.policy.autoregressive_model_params):
            # 更新基本参数
            self.cfg.model.policy.autoregressive_model_params.num_sampling_steps = params['num_sampling_steps']
            self.cfg.model.policy.autoregressive_model_params.cfg = params['cfg']
            self.cfg.model.policy.autoregressive_model_params.temperature = params['temperature']
            self.cfg.model.policy.autoregressive_model_params.window_size = params['window_size']
            self.cfg.model.policy.autoregressive_model_params.lambda_local = params['lambda_local']
            
            # 更新UCGM配置
            if 'ucgmts_config' not in self.cfg.model.policy.autoregressive_model_params:
                self.cfg.model.policy.autoregressive_model_params.ucgmts_config = OmegaConf.create({})
            
            ucgmts_config = self.cfg.model.policy.autoregressive_model_params.ucgmts_config
            ucgmts_config.transport_type = "Linear"
            ucgmts_config.consistc_ratio = params['consistc_ratio']
            ucgmts_config.scaled_cbl_eps = 0.0
            ucgmts_config.ema_decay_rate = 0.0
            ucgmts_config.rfba_gap_steps = [0.001, params['rfba_gap_end']]
            ucgmts_config.extrapol_ratio = params['extrapol_ratio']
    
    def _update_model_params(self, workspace: BaseWorkspace, params: Dict[str, Any]):
        """直接更新模型组件的参数"""
        try:
            # 获取原始模型 - 处理accelerator.prepare()的包装
            original_model = workspace.model
            if hasattr(workspace.model, 'module'):
                # 如果被accelerator包装了，通过module访问原始模型
                original_model = workspace.model.module
                print("✅ Detected accelerator-wrapped model, accessing via .module")
            else:
                print("ℹ️  Model not wrapped by accelerator")
            
            # 更新模型中的超参数
            if hasattr(original_model, 'model') and hasattr(original_model.model, 'diffactloss'):
                diffactloss = original_model.model.diffactloss
                
                # 更新UCGMTS参数
                if hasattr(diffactloss, 'ucgmts'):
                    ucgmts = diffactloss.ucgmts
                    
                    # 记录更新前的值
                    old_cor = getattr(ucgmts, 'cor', 'N/A')
                    old_tdr = getattr(ucgmts, 'tdr', 'N/A')
                    old_tdc = getattr(ucgmts, 'tdc', 'N/A')
                    
                    # 更新关键参数 - 使用UCGMTS的实际参数名
                    if hasattr(ucgmts, 'cor'):  # consistc_ratio
                        ucgmts.cor = params['consistc_ratio']
                    
                    if hasattr(ucgmts, 'tdr'):  # lab_drop_ratio (stochasticity)
                        ucgmts.tdr = params['consistc_ratio']  # 使用consistc_ratio作为stochasticity
                    
                    # 更新rfba_gap_steps相关的参数
                    # 注意：rfba_gap_steps主要用于采样，在训练中可能不直接使用
                    # 但我们可以更新相关的time distribution control
                    if hasattr(ucgmts, 'tdc'):  # time_dist_ctrl
                        # 根据rfba_gap_end调整time distribution
                        rfba_gap_end = params['rfba_gap_end']
                        if rfba_gap_end > 0.5:  # few-step mode
                            ucgmts.tdc = [0.8, 1.0, 1.0]
                        else:  # multi-step mode
                            ucgmts.tdc = [1.0, 1.0, 1.0]
                    
                    # 验证更新是否成功
                    new_cor = getattr(ucgmts, 'cor', 'N/A')
                    new_tdr = getattr(ucgmts, 'tdr', 'N/A')
                    new_tdc = getattr(ucgmts, 'tdc', 'N/A')
                    
                    print(f"Model parameters updated successfully!")
                    print(f"  - consistc_ratio (cor): {old_cor} -> {new_cor}")
                    print(f"  - stochasticity (tdr): {old_tdr} -> {new_tdr}")
                    print(f"  - time_dist_ctrl (tdc): {old_tdc} -> {new_tdc}")
                    print(f"  - rfba_gap_end: {params['rfba_gap_end']}")
                    print(f"  - num_sampling_steps: {params['num_sampling_steps']}")
                    print(f"  - temperature: {params['temperature']}")
                    
                    # 验证参数确实被更新了
                    if new_cor != old_cor:
                        print("✅ consistc_ratio successfully updated!")
                    else:
                        print("❌ consistc_ratio update failed!")
                        
                else:
                    print("❌ UCGMTS not found in diffactloss!")
                
                # 更新其他参数
                if hasattr(diffactloss, 'num_sampling_steps'):
                    old_steps = diffactloss.num_sampling_steps
                    diffactloss.num_sampling_steps = params['num_sampling_steps']
                    print(f"  - num_sampling_steps: {old_steps} -> {params['num_sampling_steps']}")
                
            else:
                print("❌ Model structure not found!")
                print(f"  - workspace.model: {hasattr(workspace, 'model')}")
                print(f"  - original_model: {hasattr(original_model, 'model') if 'original_model' in locals() else 'Not created'}")
                print(f"  - diffactloss: {hasattr(original_model.model, 'diffactloss') if hasattr(original_model, 'model') else False}")
                
        except Exception as e:
            print(f"Warning: Could not update model parameters directly: {e}")
            print("Parameters updated in config only.")
            import traceback
            traceback.print_exc()
    
    def should_optimize(self, epoch: int) -> bool:
        """判断是否应该进行优化"""
        # 前150个epoch正常训练，之后每隔20个epoch优化一次
        if epoch < 150:
            return False
        
        optimization_schedule = {
            150: True,   # 第一次优化
            170: True,   # 每20个epoch优化一次
            190: True,
            210: True,
            230: True,
            250: True,
            270: True,
            290: True,
            310: True,
            330: True,
            350: True,
            370: True,
            390: True,
            400: True   # 最终优化
        }
        return optimization_schedule.get(epoch, False)
    
    def evaluate_with_checkpoint(self, checkpoint_path: str, params: Dict[str, Any]) -> float:
        """使用checkpoint评估参数组合 - 基于bayesian_optimization_eval.py"""
        try:
            # 创建临时输出目录
            temp_output_dir = tempfile.mkdtemp(prefix="bayesian_eval_")
            
            # 运行评估命令 - 直接使用eval_sim.py
            cmd = [
                "python", "eval_sim.py",
                "--checkpoint", checkpoint_path,
                "--output_dir", temp_output_dir,
                "--use_ucgm",
                "--num_sampling_steps", str(params['num_sampling_steps']),
                "--stochasticity_rate", str(params['consistc_ratio']),
                "--window_size", str(params['window_size']),
                "--lambda_local", str(params['lambda_local'])
            ]
            
            print(f"Running evaluation: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode != 0:
                print(f"Evaluation failed: {result.stderr}")
                return -1000.0
            
            # 解析结果
            eval_log_path = os.path.join(temp_output_dir, f'eval_log_{os.path.basename(checkpoint_path)}.json')
            
            if not os.path.exists(eval_log_path):
                print(f"Eval log not found: {eval_log_path}")
                return -1000.0
            
            with open(eval_log_path, 'r') as f:
                eval_results = json.load(f)
            
            # 提取test_mean_score
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
            
            print(f"Evaluation score: {score}")
            
            # 清理临时文件
            subprocess.run(["rm", "-rf", temp_output_dir], check=False)
            
            return float(score)
            
        except subprocess.TimeoutExpired:
            print("Evaluation timeout")
            return -1000.0
        except Exception as e:
            print(f"Evaluation error: {e}")
            return -1000.0
    
    def run_bayesian_optimization(self, checkpoint_path: str, epoch: int) -> Optional[Dict[str, Any]]:
        """运行贝叶斯优化"""
        print(f"\n=== Starting Bayesian Optimization at Epoch {epoch} ===")
        
        # 固定15个trial
        max_trials = 15
        
        def objective_function(params):
            return self.evaluate_with_checkpoint(checkpoint_path, params)
        
        optimizer = UCGMBayesianOptimizer(max_trials=max_trials)
        
        try:
            best_params, best_score = optimizer.optimize(objective_function)
            
            print(f"Optimization completed! Best score: {best_score:.4f}")
            print(f"Best params: {best_params}")
            
            # 只有当新参数明显更好时才更新
            improvement_threshold = 0.01  # 1%的改进阈值
            if best_score > self.best_score + improvement_threshold:
                print(f"Significant improvement detected! Updating parameters.")
                self.best_score = best_score
                self.current_params = best_params
                
                # 记录优化历史
                self.optimization_history.append({
                    'epoch': epoch,
                    'best_score': best_score,
                    'best_params': best_params
                })
                
                return best_params
            else:
                print(f"No significant improvement. Keeping current parameters.")
                return None
                
        except Exception as e:
            print(f"Bayesian optimization failed: {e}")
            return None
    
    def save_optimization_history(self, filename: str = "optimization_history.json"):
        """保存优化历史"""
        with open(filename, 'w') as f:
            json.dump(self.optimization_history, f, indent=2)
        print(f"Optimization history saved to {filename}")


@hydra.main(
    version_base=None,
    config_path=str(
        pathlib.Path(__file__).parent.joinpath("unified_video_action", "config")
    ),
)

def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    
    # 修改总epoch数为400
    cfg.training.num_epochs = 400
    
    if cfg.model.policy.action_model_params.predict_action == False:
        cfg.checkpoint.topk.monitor_key = "video_fvd"
        cfg.checkpoint.topk.format_str = (
            "epoch={epoch:04d}-video_fvd={video_fvd:.3f}.ckpt"
        )
        cfg.checkpoint.topk.mode = "min"

    with open_dict(cfg):
        cfg.n_gpus = torch.cuda.device_count()
        cfg.model.policy.debug = cfg.training.debug

    if cfg.training.debug:
        cfg.dataloader.batch_size = 2
        cfg.val_dataloader.batch_size = 2
        cfg.dataloader.shuffle = False
        cfg.val_dataloader.shuffle = False
        cfg.training.num_epochs = 10  # 调试模式减少epoch

    # 创建自适应训练器
    adaptive_trainer = AdaptiveTrainer(cfg)
    
    # 创建workspace
    cls = hydra.utils.get_class(cfg.model._target_)
    workspace: BaseWorkspace = cls(cfg)

    if cfg.payload:
        print(f"Loading checkpoint from: {cfg.payload}")
        payload = torch.load(open(cfg.payload, "rb"), pickle_module=dill)
        if cfg.training.use_ema:
            workspace.load_payload_new(payload, exclude_keys=None, include_keys=None, diffhead_finetuning=False, strict=False)
        else:
            workspace.load_payload_new(payload, exclude_keys=['ema_model'], include_keys=None, diffhead_finetuning=True, strict=False)
        
        # Re-copy encoder parameters to local causal encoder blocks after loading checkpoint
        if hasattr(workspace.model, 'model') and hasattr(workspace.model.model, 'copy_encoder_parameters'):
            workspace.model.model.copy_encoder_parameters()
            print("Re-copied encoder parameters to local causal encoder blocks")

    if cfg.freeze_submodules:
        workspace.freeze_submodules(action_only=True)

    # 修改workspace的run方法以支持自适应训练
    original_run = workspace.run
    
    def adaptive_run():
        """修改后的run方法，集成贝叶斯优化"""
        cfg = copy.deepcopy(workspace.cfg)
        
        # 导入必要的模块
        from accelerate import Accelerator
        from accelerate.utils import DeepSpeedPlugin, DistributedDataParallelKwargs
        from unified_video_action.common.checkpoint_util import TopKCheckpointManager
        from unified_video_action.model.autoregressive.ema_model import EMAModel
        from unified_video_action.model.common.lr_scheduler import get_scheduler
        from unified_video_action.utils.data_utils import resize_image
        from unified_video_action.common.pytorch_util import dict_apply
        from unified_video_action.dataset.base_dataset import BaseImageDataset
        from unified_video_action.dataset.umi_multi_dataset import UmiMultiDataset
        from unified_video_action.utils.load_env import load_env_runner, env_rollout
        from unified_video_action.eval.eval import test_video_fvd, test_action_l2
        import tqdm
        import math
        import random
        from torch.utils.data import DataLoader
        
        # 设置随机种子
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # 初始化accelerator
        accelerator = Accelerator()
        
        # 配置数据加载器
        train_dataset = hydra.utils.instantiate(cfg.task.dataset)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.dataloader.batch_size,
            shuffle=cfg.dataloader.shuffle,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory,
            persistent_workers=cfg.dataloader.persistent_workers,
        )
        
        val_dataset = hydra.utils.instantiate(cfg.task.val_dataset)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.val_dataloader.batch_size,
            shuffle=cfg.val_dataloader.shuffle,
            num_workers=cfg.val_dataloader.num_workers,
            pin_memory=cfg.val_dataloader.pin_memory,
            persistent_workers=cfg.val_dataloader.persistent_workers,
        )
        
        # 配置优化器和调度器
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=workspace.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=cfg.training.num_epochs * len(train_dataloader),
        )
        
        # 配置EMA
        ema_model = None
        if cfg.training.use_ema:
            ema_model = EMAModel(workspace.model)
        
        # 配置checkpoint管理器
        checkpoint_manager = TopKCheckpointManager(
            save_dir=pathlib.Path(workspace.output_dir).joinpath("checkpoints"),
            **cfg.checkpoint.topk
        )
        
        # 准备accelerator
        (
            train_dataloader,
            val_dataloader,
            workspace.model,
            workspace.optimizer,
            lr_scheduler,
        ) = accelerator.prepare(
            train_dataloader,
            val_dataloader,
            workspace.model,
            workspace.optimizer,
            lr_scheduler,
        )
        
        device = workspace.model.device
        
        # Re-copy encoder parameters to local causal encoder blocks after accelerator.prepare
        if hasattr(workspace.model, 'model') and hasattr(workspace.model.model, 'copy_encoder_parameters'):
            workspace.model.model.copy_encoder_parameters()
            print("Re-copied encoder parameters to local causal encoder blocks after accelerator.prepare")
        
        if ema_model is not None:
            ema_model.to(device)
        
        # 打印可训练参数信息
        total_params = sum(p.numel() for p in workspace.model.parameters())
        trainable_params = sum(p.numel() for p in workspace.model.parameters() if p.requires_grad)
        trainable_percent = 100.0 * trainable_params / total_params
        
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
              f"({trainable_percent:.2f}%)")
        print(f"Starting adaptive training for {cfg.training.num_epochs} epochs")
        print(f"Initial parameters: {adaptive_trainer.current_params}")
        
        # 主训练循环
        for local_epoch_idx in range(cfg.training.num_epochs):
            workspace.epoch = local_epoch_idx
            
            print(f"\n=== Epoch {local_epoch_idx+1}/{cfg.training.num_epochs} ===")
            
            # 检查是否需要优化
            if adaptive_trainer.should_optimize(local_epoch_idx + 1):
                # 找到最新的checkpoint
                checkpoint_pattern = f"{workspace.output_dir}/checkpoints/epoch={local_epoch_idx:04d}-*.ckpt"
                checkpoints = glob.glob(checkpoint_pattern)
                
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=os.path.getctime)
                    print(f"Using checkpoint: {latest_checkpoint}")
                    
                    optimized_params = adaptive_trainer.run_bayesian_optimization(latest_checkpoint, local_epoch_idx + 1)
                    
                    if optimized_params:
                        print(f"Parameters updated: {optimized_params}")
                        # 更新配置
                        adaptive_trainer._update_config_with_params(optimized_params)
                        # 直接更新模型参数
                        adaptive_trainer._update_model_params(workspace, optimized_params)
                        # 可选：降低学习率以稳定训练
                        for param_group in workspace.optimizer.param_groups:
                            param_group['lr'] *= 0.95
                else:
                    print(f"No checkpoint found for epoch {local_epoch_idx}")
            
            # 训练一个epoch
            step_log = dict()
            train_losses = list()
            
            with tqdm.tqdm(
                train_dataloader,
                desc=f"Training epoch {workspace.epoch}",
                leave=False,
                mininterval=cfg.training.tqdm_interval_sec,
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    # resize image
                    batch = resize_image(cfg, batch)
                    
                    # compute loss
                    if (
                        "deepspeed_config" in cfg.training
                        and cfg.training.deepspeed_config is not None
                    ): 
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            raw_loss, (loss_diffusion, loss_action) = workspace.model(batch)
                    else:
                        raw_loss, (loss_diffusion, loss_action) = workspace.model(batch)
                    
                    # backward pass
                    accelerator.backward(raw_loss)
                    
                    scale = accelerator.scaler.get_scale()
                    
                    # step optimizer
                    if workspace.global_step % cfg.training.gradient_accumulate_every == 0:
                        workspace.optimizer.step()
                        workspace.optimizer.zero_grad()
                        lr_scheduler.step()
                    
                    # update ema
                    if cfg.training.use_ema:
                        ema_model.step(accelerator.unwrap_model(workspace.model))
                    
                    # logging
                    raw_loss_cpu = raw_loss.item()
                    
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    
                    step_log = {
                        "AMP scale": scale,
                        "train_loss": raw_loss_cpu,
                        "global_step": workspace.global_step,
                        "epoch": workspace.epoch,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    
                    is_last_batch = batch_idx == (len(train_dataloader) - 1)
                    if not is_last_batch:
                        accelerator.log(step_log, step=workspace.global_step)
                        workspace.global_step += 1
                    
                    if (cfg.training.max_train_steps is not None) and batch_idx >= (
                        cfg.training.max_train_steps - 1
                    ):
                        break
            
            train_loss = np.mean(train_losses)
            step_log["train_loss"] = train_loss
            
            # 验证和保存checkpoint
            if (local_epoch_idx + 1) % cfg.training.val_every == 0:
                # 这里可以添加验证逻辑
                pass
            
            if (local_epoch_idx + 1) % cfg.training.checkpoint_every == 0:
                checkpoint_manager.save(
                    accelerator.unwrap_model(workspace.model),
                    step=workspace.global_step,
                    epoch=workspace.epoch,
                    test_mean_score=0.0,  # 这里需要实际的验证分数
                )
        
        print("Adaptive training completed!")
    
    # 替换run方法
    workspace.run = adaptive_run
    
    # 运行训练
    workspace.run()
    
    # 保存优化历史
    adaptive_trainer.save_optimization_history("final_optimization_history.json")
    print("All training completed!")


if __name__ == "__main__":
    print(sys.argv)
    for arg in sys.argv:
        if "local_rank" in arg:  # For deepspeed compatibility
            sys.argv.remove(arg)
    main()
